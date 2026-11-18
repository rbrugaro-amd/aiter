"""S1 (GLM-5.2 MXFP4 member, bf16 W): fused qkv_a projection + dual RMSNorm.

T : glm52/qkv_a_norm_xl/qkv_a_norm_template.py::qkv_a_norm_tp4 / qkv_a_norm_tp8 (bf16 x, bf16 W
    [2624,6144] -> q, kv, kr bf16)
B : SGLang 703e763d5f as called for GLM-5.2 MXFP4 (q_b weight uint8): qkv_a through
    UnquantizedLinearMethod.process_weights_after_loading + apply (gfx950: small-M tiled /
    rowmajor / Triton a16w16 / tgemm.mm dispatch as SGLang decides), split, then aiter
    fused_rms_mxfp4_quant(q, kv) [also emits MXFP4 q for q_b -- extra work vs T]
B1: same-work composition: the same SGLang linear + fused_qk_rmsnorm_bf16 (AITER fused_qk_rmsnorm)
"""
import torch

from xbench import core

K, NQ, NKV, NR = 6144, 2048, 512, 64
N = NQ + NKV + NR
EPS = 1e-5
TPL = "glm52/qkv_a_norm_xl/qkv_a_norm_template.py"
SHAPES = ([dict(label=f"t4m{m}", tp=4, m=m) for m in (1, 2, 4, 8, 16, 32, 64, 128, 256)]
          + [dict(label=f"t8m{m}", tp=8, m=m) for m in (1, 8, 64, 256)]
          + [dict(label=f"t8m{m}", tp=8, m=m, caches=["rotating"]) for m in (1024, 4193, 8192, 16384)])
MODE = "graph"
SEEDS = (0, 100)
AA = ["T", "B1"]


def gen(shape, seed):
    import glm52__qkv_a_norm as sp
    inp = sp._make_inputs((shape["tp"], shape["m"], 0), seed, "cuda")
    proj = inp["hidden"].double() @ inp["weight"].double().t()

    def rms(t, g):
        t = t.float()
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + EPS) * g.float()
    return inp, dict(q=rms(proj[:, :NQ], inp["q_gamma"]), kv=rms(proj[:, NQ:NQ + NKV], inp["kv_gamma"]),
                     kr=proj[:, NQ + NKV:].float())


def _names(inp):
    return [k for k in ("hidden", "weight", "q_gamma", "kv_gamma") if k in inp]


def _T(shape):
    def b(inp):
        from xbench import tload
        tp = shape["tp"]
        fn = tload.load(TPL, f"qkv_a_norm_tp{tp}", f"glm52_mxfp4_tp{tp}_gfx950/qkv_a_norm_templated.json",
                        "qkv_a_norm_xl/qkv_a_norm_template.py")
        args = [inp[k] for k in _names(inp)]

        def call():
            q, kv, kr = fn(*args, eps=EPS, rope_dim=NR)
            return dict(q=q, kv=kv, kr=kr)
        return call
    return b


def _linear(w):
    uq = core.sglang_import("sglang.srt.layers.quantization.unquant")
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w, requires_grad=False)
    meth = uq.UnquantizedLinearMethod()
    meth.process_weights_after_loading(layer)
    return lambda x: meth.apply(layer, x)


def _B(inp):
    lin = _linear(inp["weight"])
    fmr = core.sglang_import("sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm")
    x, qg, kvg = inp["hidden"], inp["q_gamma"], inp["kv_gamma"]

    def call():
        y = lin(x)
        q, latent = y.split([NQ, NKV + NR], dim=-1)
        qm, qms, kvn, *_ = fmr.fused_rms_mxfp4_quant(q, qg, EPS, latent[..., :NKV], kvg, EPS)
        return dict(q_mxfp4=qm, q_mxfp4_scale=qms, kv=kvn, kr=latent[..., NKV:], q=None)
    return call


def _B1(inp):
    lin = _linear(inp["weight"])
    fmr = core.sglang_import("sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm")
    x, qg, kvg = inp["hidden"], inp["q_gamma"], inp["kv_gamma"]

    def call():
        y = lin(x)
        q, latent = y.split([NQ, NKV + NR], dim=-1)
        qn, kvn = fmr.fused_qk_rmsnorm_bf16(q, qg, EPS, latent[..., :NKV], kvg, EPS)
        return dict(q=qn, kv=kvn, kr=latent[..., NKV:])
    return call


def arms(shape):
    return {"T": _T(shape), "B": _B, "B1": _B1}


def check(arm, outs, ref, inp, shape):
    st = {k: core.cmp_close(outs[k], ref[k]) for k in ("kv", "kr")}
    if outs.get("q") is not None:
        st["q"] = core.cmp_close(outs["q"], ref["q"])
    return st
