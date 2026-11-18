"""S12pr: AITER PR gdn-prefill-qwen35 -- fused_gdn_prefill_qkvz(qkvz_layout="flat") (branch) vs
the AITER composition a Qwen3.5 caller must run on upstream main (the op rejects Qwen3.5 there).

Reuses S12's inputs semantics, fp32 reference, stock-envelope pass rule and the B arms verbatim.
Arms:
  T    : aiter fused_gdn_prefill_qkvz(..., conv_bias=None, qkvz_layout="flat")  (the PR)
  Tpk  : the Artemis pack template (xl) the PR was ported from (parity control; S12 'T')
  B1   : causal_conv1d_split_qkv_triton_fn -> torch gates -> chunk_gated_delta_rule_opt ->
         FLA RMSNormGated (S12 B1)
  B2   : as B1 with chunk_gated_delta_rule (S12 B2)
  X_fla: stock SGLang path (context; envelope)
The B arms' AITER functions are byte-identical on the branch and on upstream main ca7cfa1c3
(the PR touches only fused_gdn_prefill_qkvz*.py); the separate main-copy run (AITER from
/scratch/aiter_main_gdnp) confirms it.
"""
import torch

from xbench import core
from xbench.cands import S12 as base

HK, HV, D, CONV, SCALE, EPS = base.HK, base.HV, base.D, base.CONV, base.SCALE, base.EPS
SPEC_SEED, SEEDS, MODE = base.SPEC_SEED, base.SEEDS, "eager"
AA = ["T", "B1"]
_sh = base._sh

SHAPES = [s for s in base.SHAPES if s["tpl"] == "xl"] + [
    _sh((10240,), 0), _sh((4096, 5120, 5120, 2048), 8),     # S12's r3 rows, now via xl
    _sh((2048,), 1), _sh((3000, 2000), 2), _sh((8192, 8192), 3),
    _sh((4096, 4096, 4096), 5), _sh((1500, 7, 900, 3000, 33, 100, 64, 3000), 37),
]
gen, reference, check, reset = base.gen, base.reference, base.check, base.reset


def _T(shape):
    def build(inp):
        import importlib
        m = importlib.import_module("aiter.ops.triton.gated_delta_net.fused_gdn_prefill_qkvz")
        core.restore_triton()
        args = (inp["projected_qkvz"], inp["projected_ba"], inp["conv_state"], inp["delta_state"],
                inp["cache_indices"], inp["cu_seqlens"], inp["has_initial_state"], inp["conv_weight"],
                None, inp["a_log"], inp["dt_bias"], inp["norm_weight"])
        ok, reason = m.fused_gdn_prefill_qkvz_supported(*args[:9], None, a_log=args[9], dt_bias=args[10],
                                                         norm_weight=args[11], qkvz_layout="flat")
        if not ok:
            raise core.Unavailable(f"flat layout unsupported: {reason}")

        def call():
            o, c, d, q, s = m.fused_gdn_prefill_qkvz(*args, scale=SCALE, eps=EPS, qkvz_layout="flat")
            return dict(output=o, conv_state=c, delta_state=d)
        return call
    return build


def _Tpk(shape):
    return base._T(dict(shape, tpl="xl"))


def raw_arms(shape):
    return {"T": _T(shape), "Tpk": _Tpk(shape), "B1": base._aiter_chain(shape),
            "B2": base._aiter_chain(shape, gdr="plain"), "X_fla": base._stock(shape)}


def arms(shape):
    return {k: base._cycled(k, v) for k, v in raw_arms(shape).items()}
