"""S1 (GLM-5.2 MXFP4 TP8-EP8-DP8 member, bf16 W): glm52_ep8/qkv_a_norm_xl/decode_async_x.py::qkv_a_norm
at its served rows M 4, 8, 16 (per-DP-rank decode M).  B/B1 as in S1g (SGLang linear dispatch +
fused_rms_mxfp4_quant / fused_qk_rmsnorm_bf16).  Inputs are the glm52 base draw at (tp=8, M)."""
from xbench.cands import S1g as g

SHAPES = [dict(label=f"m{m}", tp=8, m=m) for m in (4, 8, 16)]
MODE = "graph"
SEEDS = (0, 100)
AA = ["T", "B1"]
gen = g.gen
check = g.check


def _T(inp):
    from xbench import tload
    fn = tload.load("glm52_ep8/qkv_a_norm_xl/decode_async_x.py", "qkv_a_norm",
                    "glm52_mxfp4_tp8_ep8_dp8_gfx950/qkv_a_norm_templated.json", "qkv_a_norm_xl/decode_async_x.py")
    args = [inp[k] for k in ("hidden", "weight", "q_gamma", "kv_gamma")]

    def call():
        q, kv, kr = fn(*args, eps=g.EPS, rope_dim=g.NR)
        return dict(q=q, kv=kv, kr=kr)
    return call


def arms(shape):
    return {"T": _T, "B": g._B, "B1": g._B1}
