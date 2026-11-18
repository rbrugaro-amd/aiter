"""S1 GLM-5.2 FP8 qkv_a_norm: AITER PR branch op vs the AITER chain (PR gate).

A  : aiter.ops.triton.fusions.fused_mla_qkv_a_proj_rmsnorm (branch mla-qkv-a-norm-quant-fused;
     fp32 split-K partials), M in {1, 2, 4, 8, 16, 32}.
B1 : AITER chain as SGLang composes it (per-1x128 quant + bpreshuffle GEMM by tuned CSV +
     fused_qk_rmsnorm), same as S1.B1 -- unchanged files between upstream main and the branch.
B2 : pure-AITER chain without SGLang: get_hip_quant(per_1x128) + gemm_a8w8_blockscale_bpreshuffle
     + fused_qk_rmsnorm (what the AITER benchmark script times as --baseline).
T  : the served Artemis template (bf16 partials), for reference.
Run with the PR AITER copy on PYTHONPATH (s1glm_pr/prelude.sh pr); with the main copy A is
Unavailable and only B1/B2 run (baseline cross-check).
"""
import torch

from xbench import core
from xbench.cands import S1 as s1

SHAPES = [dict(label=f"m{m}", m=m) for m in (1, 2, 4, 8, 16, 32)]
MODE = "graph"
SEEDS = (0, 100)
AA = ["A", "B1"]
gen = s1.gen
check = s1.check


def _A(inp):
    try:
        from aiter.ops.triton.fusions.fused_mla_qkv_a_proj_rmsnorm import (
            fused_mla_qkv_a_proj_rmsnorm, fused_mla_qkv_a_proj_rmsnorm_supported)
    except ImportError as e:
        raise core.Unavailable(f"branch op not importable: {e}")
    w = inp["weight"]
    args = (inp["hidden"], w, inp["weight_scales"], inp["q_gamma"], inp["kv_gamma"])
    ok, reason = fused_mla_qkv_a_proj_rmsnorm_supported(*args, rope_dim=s1.NR)
    if not ok:
        raise core.Unavailable(reason)

    def call():
        q, kv, kr = fused_mla_qkv_a_proj_rmsnorm(*args, rope_dim=s1.NR, eps=s1.EPS)
        return dict(q=q, kv=kv, kr=kr)
    return call


def _B2(inp):
    from aiter import dtypes
    from aiter.ops.enum import QuantType
    from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle
    from aiter.ops.quant import get_hip_quant
    x, w, ws, qg, kvg = inp["hidden"], s1._weight(inp), inp["weight_scales"], inp["q_gamma"], inp["kv_gamma"]
    m = x.shape[0]
    quant = get_hip_quant(QuantType.per_1x128)

    def call():
        xq, xs = quant(x, quant_dtype=dtypes.fp8, transpose_scale=m >= 2)
        if m >= 2:
            xs = xs.as_strided((m, s1.K // 128), (1, m))
        y = gemm_a8w8_blockscale_bpreshuffle(xq, w, xs, ws, dtypes.bf16)
        q = torch.empty((m, s1.NQ), dtype=torch.bfloat16, device=x.device)
        kv = torch.empty((m, s1.NKV), dtype=torch.bfloat16, device=x.device)
        fused_qk_rmsnorm(q_out_quantized=q, q=y[:, :s1.NQ], q_weight=qg, q_epsilon=s1.EPS,
                         k_out=kv, k=y[:, s1.NQ:s1.NQ + s1.NKV], k_weight=kvg, k_epsilon=s1.EPS,
                         quant_type=QuantType.No)
        return dict(q=q, kv=kv, kr=y[:, s1.NQ + s1.NKV:])
    return call


def arms(shape):
    return {"A": _A, "B1": s1._B1, "B2": _B2, "T": s1._T}
