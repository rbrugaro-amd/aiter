"""S1 (DeepSeek-R1 member): fused MLA q/kv-a projection + dual RMSNorm + FP8 group quant of q.

T : deepseek_r1/qkv_a_norm_quant_templated/qkv_a_norm_quant_template.py::mla_qkv_a_norm_quant
    (fp8 x [M,7168] + row-major group scales [M,56], plain fp8 W [2112,7168] + [17,56] block
    scales -> q fp8 + q scales [M,12], kv bf16, kr bf16)
B : SGLang 703e763d5f as called for DeepSeek-R1 FP8 on gfx950 (forward_absorb_rocm_prepare,
    q_b block-fp8, no DSA): (2112,7168) is in use_aiter_triton_gemm_w8a8_tuned_gfx950, so the
    weight stays plain and aiter_w8a8_block_fp8_linear runs AITER's Triton gemm_a8w8_blockscale
    on the producer-quantized input (scale handed over in the transposed-emit layout); then
    aiter fused_rms_fp8_group_quant(q, kv, output_unquantized_inp1=False, transpose_scale=M>=2).
"""
import torch

from xbench import core

K, NQ, NKV, NR = 7168, 1536, 512, 64
N = NQ + NKV + NR
EPS = 1e-6
PROFILE = "deepseek_r1_fp8_tp8_gfx950/qkv_a_norm_quant_templated.json"
TPL = "deepseek_r1/qkv_a_norm_quant_templated/qkv_a_norm_quant_template.py"
DECODE = [1, 2, 4, 8, 16, 32, 64, 128, 256]
PREFILL = [1024, 4096, 8192, 16384]
SHAPES = [dict(label=f"m{m}", m=m) for m in DECODE] + [
    dict(label=f"m{m}", m=m, caches=["rotating"]) for m in PREFILL]
MODE = "graph"
SEEDS = (0, 100)
AA = ["T", "B"]


def gen(shape, seed):
    import deepseek_r1__qkv_a_norm_quant as sp
    inp = sp._make_inputs(shape["m"], seed, "cuda")
    m = shape["m"]
    x = inp["hidden"].double().view(m, K // 128, 128) * inp["hidden_scales"].double()[..., None]
    w = inp["weight"].double() * inp["weight_scales"].double().repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)
    proj = x.view(m, K) @ w.t()

    def rms(t, g):
        t = t.float()
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + EPS) * g.float()
    q = rms(proj[:, :NQ], inp["q_gamma"])
    qb = q.to(torch.bfloat16).float().view(m, NQ // 128, 128)
    qs = qb.abs().amax(-1).clamp_min(1e-10) / 448.0
    q8 = (qb / qs[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(m, NQ)
    ref = dict(q=q, q8=q8, qs=qs.float(), kv=rms(proj[:, NQ:NQ + NKV], inp["kv_gamma"]),
               kr=proj[:, NQ + NKV:].float())
    return inp, ref


def _T(inp):
    from xbench import tload
    fn = tload.load(TPL, "mla_qkv_a_norm_quant", PROFILE, "qkv_a_norm_quant_templated/qkv_a_norm_quant_template.py")
    names = ("hidden", "hidden_scales", "weight", "weight_scales", "q_gamma", "kv_gamma")

    def call():
        q, qs, kv, kr = fn(**{k: inp[k] for k in names}, eps=EPS, bf16_q_hip_quant=True)
        return dict(q8=q, qs=qs, kv=kv, kr=kr)
    return call


def _B(inp):
    fu = core.sglang_import("sglang.srt.layers.quantization.fp8_utils")
    fmr = core.sglang_import("sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm")
    x, w, ws = inp["hidden"], inp["weight"], inp["weight_scales"]
    m = x.shape[0]
    assert fu.use_aiter_triton_gemm_w8a8_tuned_gfx950(N, K)
    # producer (fused input RMSNorm + quant) emits the scale already transposed: bytes [56, M]
    xs_t = inp["hidden_scales"].t().contiguous().view(m, K // 128)
    qg, kvg = inp["q_gamma"], inp["kv_gamma"]
    emit = fu.emit_transposed_bpreshuffle_scale(m, on_bpreshuffle_gfx95=fu._use_aiter_bpreshuffle_gfx95)

    def call():
        y = fu.aiter_w8a8_block_fp8_linear(x, w, [128, 128], ws, input_scale=xs_t)
        q, latent = y.split([NQ, NKV + NR], dim=-1)
        qq, _, kvn, _ = fmr.fused_rms_fp8_group_quant(
            q, qg, EPS, latent[..., :NKV], kvg, EPS, group_size=128, dtype_quant=torch.float8_e4m3fn,
            res1=None, output_unquantized_inp1=False, transpose_scale=emit)
        qq = fu.view_aiter_fused_rms_transposed_fp8_scale_tuple(qq) if emit else fu.materialize_bpreshuffle_fp8_scale_tuple(qq)
        return dict(q8=qq[0], qs=qq[1], kv=kvn, kr=latent[..., NKV:])
    return call


def arms(shape):
    return {"T": _T, "B": _B}


def check(arm, outs, ref, inp, shape):
    st = dict(q=core.cmp_fp8(outs["q8"], outs["qs"].contiguous().float(), ref["q8"], ref["qs"]),
              kv=core.cmp_close(outs["kv"], ref["kv"]), kr=core.cmp_close(outs["kr"], ref["kr"]))
    m = outs["q8"].shape[0]
    deq = outs["q8"].float().view(m, NQ // 128, 128) * outs["qs"].contiguous().float()[..., None]
    st["q_dequant_info"] = core.cmp_close(deq.view(m, NQ), ref["q"], atol=2 ** -4, rtol=2 ** -3)
    st["q_dequant_info"]["pass_"] = True  # informative only
    return st
