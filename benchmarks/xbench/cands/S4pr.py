"""S4pr: AITER PR sparse-mla-decode-gfx950 -- sparse_mla_fwd on GLM-5.2 FP8 EP8 decode
(64 heads, unit-scale fp8 q, fp8_scalar KV [P, 576], -1-padded [m, 2048] top-k table,
has_invalid=True, dot_precision="fp8"), branch vs upstream main ca7cfa1c3.

Inputs: S4's EP8 draws (glm52_fp8__sparse_paged_mla.make_inputs, seeds 0 / 100), q cast to
fp8 (unit scale, the EP8 adapters' contract), pool 262144 rows (151 MB < 2**31).
Arms:
  T    : aiter.ops.triton.attention.sparse_mla.sparse_mla_fwd as installed (branch: routes to
         the new 64-head decode kernels; main-copy run: upstream main's generic kernel).
  Bm2  : upstream main's sparse_mla_fwd, same call (fp8 dots).  The module is upstream main's
         attention/sparse_mla.py loaded from /scratch/aiter_main_s4ep8 under another name; the
         kernels it imports are byte-identical on the branch (the PR does not touch them).
  Bm1  : upstream main's sparse_mla_fwd with dot_precision="bf16" (fastest passing main arm).
  Tpk  : the Artemis pack template the kernels were ported from (parity control).
"""
import importlib.util
import os

import torch

from xbench import core
from xbench.cands import S4 as base

SCALE, TOPK, DV = base.SCALE, base.TOPK, base.DV
MAIN_SRC = os.environ.get("S4PR_MAIN", "/scratch/aiter_main_s4ep8/aiter/ops/triton/attention/sparse_mla.py")

SHAPES = [s for s in base.SHAPES if s["prof"] == "fp8ep8"]
# Served-size pools: LARGE 4194304 rows = 2.42 GB (> 2**31, buffer_load's offset range) and
# XLARGE 7456541 rows = 4.29 GB (> 2**32, the GLM-5.2 FP8 EP8 serving pool; the uint32 slot
# offset forms of rows 2 and 8 switch to int64 there).
XLARGE = 7456541
for _s in list(SHAPES):
    for _pool, _sfx in ((base.LARGE, "_L"), (XLARGE, "_XL")):
        SHAPES.append({**_s, "pool": _pool, "label": _s["label"] + _sfx})
MODE = os.environ.get("S4PR_MODE", "graph")
SEEDS = (0, 100)
AA = ["T", "Bm2"]
_ONLY = os.environ.get("S4_ONLY")
if _ONLY:
    SHAPES = [s for s in SHAPES if s["label"] in _ONLY.split(",")]

rotate, gen = base.rotate, base.gen

_MAIN = None


def _main():
    global _MAIN
    if _MAIN is None:
        spec = importlib.util.spec_from_file_location("aiter_main_sparse_mla", MAIN_SRC)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert not hasattr(mod, "_h64_decode_config"), "main copy carries the PR"
        _MAIN = mod
    return _MAIN


def _call(fwd, dp):
    def build(inp):
        q, kv, slots = inp["query"], inp["kv"], inp["slots"]
        m = q.shape[0]
        indptr = torch.arange(0, (m + 1) * TOPK, TOPK, dtype=torch.int32, device=q.device)
        flat = slots.reshape(-1)
        one = torch.ones(1, dtype=torch.float32, device=q.device)
        u8 = kv.view(torch.uint8)

        def call():
            out, _ = fwd()(q, u8, indptr, flat, SCALE, kv_scale=one, kv_lora_rank=512, qk_rope_head_dim=64,
                           has_invalid=True, dot_precision=dp, q_scale=one)
            return dict(out=out)
        return call
    return build


def _branch():
    from aiter.ops.triton.attention.sparse_mla import sparse_mla_fwd
    return sparse_mla_fwd


def arms(shape):
    return {"T": _call(_branch, "fp8"), "Bm2": _call(lambda: _main().sparse_mla_fwd, "fp8"),
            "Bm1": _call(lambda: _main().sparse_mla_fwd, "bf16"), "Tpk": base._T(shape)}


def check(arm, outs, ref, inp, shape):
    return {"out": core.cmp_close(outs["out"], ref["out"])}
