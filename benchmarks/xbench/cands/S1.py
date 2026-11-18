"""S1: fused MLA q/kv-a projection + dual RMSNorm (GLM-5.2 FP8 lead template).

T  : glm52_fp8/qkv_a_norm_r3/qkv_a_norm_template.py::mla_qkv_a_norm (bf16 x -> q, kv, kr bf16)
B  : SGLang 703e763d5f stock, as called (forward_absorb_rocm_prepare, GLM-5.2 FP8 = use_dsa and
     block-fp8 q_b): fp8_utils.aiter_w8a8_block_fp8_linear (per1x128 quant + AITER bpreshuffle
     GEMM, tuned-CSV dispatch) -> split -> aiter fused_rms_fp8_group_quant(q, kv,
     output_unquantized_inp1=True, transpose_scale=M>=2) [also emits fp8 q + scales for q_b]
B1 : same-work AITER composition (no fp8 q output): aiter_w8a8_block_fp8_linear + SGLang's
     fused_qk_rmsnorm_bf16 (AITER fused_qk_rmsnorm, quant_type No) -- SGLang's branch for
     non-fp8 q_b models.
"""
import torch

from xbench import core

K, NQ, NKV, NR = 6144, 2048, 512, 64
N = NQ + NKV + NR
EPS = 1e-5
PROFILE = "glm52_fp8_tp4_gfx950/qkv_a_norm_templated.json"
TPL = "glm52_fp8/qkv_a_norm_r3/qkv_a_norm_template.py"

DECODE = [1, 2, 4, 8, 16, 32, 64, 128, 256]
PREFILL = [1024, 2048, 4096, 8193, 16384]
SHAPES = [dict(label=f"m{m}", m=m) for m in DECODE] + [
    dict(label=f"m{m}", m=m, caches=["rotating"]) for m in PREFILL]
MODE = "graph"
SEEDS = (0, 100)
AA = ["T", "B"]


def _spec():
    import glm52_fp8__qkv_a_norm as sp  # specs dir on PYTHONPATH
    return sp


def _unpack(wp):
    n, k = wp.shape
    return wp.view(n // 16, k // 16, 16, 16).permute(0, 2, 1, 3).contiguous().view(n, k)


def _quant_1x128(x):
    """Reference dynamic per-(row,128) fp8 e4m3fn quant (scale = amax/448)."""
    m, k = x.shape
    xg = x.float().view(m, k // 128, 128)
    s = xg.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448.0
    q = (xg / s).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (q.float() * s).view(m, k), s


def gen(shape, seed):
    inp = _spec()._make_inputs(shape["m"], seed, "cuda")
    x = inp["hidden"]
    w = _unpack(inp["weight"]).double()
    ws = inp["weight_scales"].double()
    wdq = w * ws.repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)
    xdq, _ = _quant_1x128(x)
    proj = (xdq.double() @ wdq.t())
    qp, kvp, kr = proj[:, :NQ], proj[:, NQ:NQ + NKV], proj[:, NQ + NKV:]

    def rms(t, g):
        t = t.float()
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + EPS) * g.float()

    ref = dict(q=rms(qp, inp["q_gamma"]), kv=rms(kvp, inp["kv_gamma"]), kr=kr.float())
    return inp, ref


def _tpl():
    from xbench import tload
    return tload.load(TPL, "mla_qkv_a_norm", PROFILE, "qkv_a_norm_r3/qkv_a_norm_template.py")


def _T(inp):
    fn = _tpl()
    kw = dict(rope_dim=NR, eps=EPS, weight_preshuffled=True)

    def call():
        q, kv, kr = fn(**{k: inp[k] for k in ("hidden", "weight", "q_gamma", "kv_gamma", "weight_scales")}, **kw)
        return dict(q=q, kv=kv, kr=kr)
    return call


def _sgl():
    fu = core.sglang_import("sglang.srt.layers.quantization.fp8_utils")
    fmr = core.sglang_import("sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm")
    return fu, fmr


def _weight(inp):
    w = inp["weight"]  # already AITER (16,16)-shuffled bytes (spec pack == aiter shuffle_weight)
    w.is_shuffled = True
    return w


def _B(inp):
    fu, fmr = _sgl()
    w, ws = _weight(inp), inp["weight_scales"]
    x, qg, kvg = inp["hidden"], inp["q_gamma"], inp["kv_gamma"]
    m = x.shape[0]
    emit = fu.emit_transposed_bpreshuffle_scale(m, on_bpreshuffle_gfx95=fu._use_aiter_bpreshuffle_gfx95)

    def call():
        y = fu.aiter_w8a8_block_fp8_linear(x, w, [128, 128], ws)
        q, latent = y.split([NQ, NKV + NR], dim=-1)
        k_nope = latent[..., :NKV]
        q_quanted, q_lora, k_nope, _ = fmr.fused_rms_fp8_group_quant(
            q, qg, EPS, k_nope, kvg, EPS, group_size=128, dtype_quant=torch.float8_e4m3fn,
            res1=None, output_unquantized_inp1=True, transpose_scale=emit)
        if emit:
            q_quanted = fu.view_aiter_fused_rms_transposed_fp8_scale_tuple(q_quanted)
        else:
            q_quanted = fu.materialize_bpreshuffle_fp8_scale_tuple(q_quanted)
        return dict(q=q_lora, kv=k_nope, kr=latent[..., NKV:], q_fp8=q_quanted[0], q_scale=q_quanted[1])
    return call


def _B1(inp):
    fu, fmr = _sgl()
    w, ws = _weight(inp), inp["weight_scales"]
    x, qg, kvg = inp["hidden"], inp["q_gamma"], inp["kv_gamma"]

    def call():
        y = fu.aiter_w8a8_block_fp8_linear(x, w, [128, 128], ws)
        q, latent = y.split([NQ, NKV + NR], dim=-1)
        qn, kvn = fmr.fused_qk_rmsnorm_bf16(q, qg, EPS, latent[..., :NKV], kvg, EPS)
        return dict(q=qn, kv=kvn, kr=latent[..., NKV:])
    return call


def _Xgemm(inp):
    """Context (A2 cross-check): the AITER bpreshuffle GEMM alone on a pre-quantized input,
    dispatched through aiter.gemm_a8w8_blockscale_bpreshuffle (tuned CSV), comparable to
    op_tests/test_gemm_a8w8_blockscale.py --ck_preshuffle True."""
    import aiter
    fu, _ = _sgl()
    x, w, ws = inp["hidden"], _weight(inp), inp["weight_scales"]
    xq, xs = fu.aiter_per1x128_quant(x, quant_dtype=aiter.dtypes.fp8, transpose_scale=False)
    xs = fu.materialize_bpreshuffle_fp8_scale(xs)

    def call():
        return dict(y=aiter.gemm_a8w8_blockscale_bpreshuffle(xq, w, xs, ws, torch.bfloat16))
    return call


def arms(shape):
    return {"T": _T, "B": _B, "B1": _B1, "X_gemm_only": _Xgemm}


def check(arm, outs, ref, inp, shape):
    st = {k: core.cmp_close(outs[k], ref[k]) for k in ("q", "kv")}
    st["kr"] = core.cmp_close(outs["kr"], ref["kr"])
    return st
