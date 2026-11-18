"""C1 output-gate member (parity only): Qwen3.5 MXFP4 decode attention with bf16 sigmoid output
gate, Hq 8 / Hk 1 / D 256, FP8 page-1 KV, graph capacity 131072.

T   : qwen3_5/mxfp4/paged_attention_output_gate_r3/r3_paog_template.py (rows 1, 4, 8).
B   : AITER port of our kernel, paged_attention_output_gate_group_fp8_quant(quant_dtype=None,
      max_context=131072) -- Triton >= 3.8 only; raises 'unsupported' in the unmodified nightly.
B1  : what SGLang 703e763d5f actually runs (it never calls the port): aiter
      paged_attention_ragged (partitions per SGLang's graph bucket, workspace for max_bs 256) +
      SGLang fused_sigmoid_mul(inplace=True).  B37 = B/B1 in the unmodified nightly (Triton 3.7).
Inputs: the family spec make_inputs (qwen3_5_mxfp4__paged_attention_output_gate), unit scales.
"""
import torch

from xbench import core
from xbench.cands import C1_common as C

CAP = 131072
PROF = "qwen3_5_mxfp4_tp4_gfx950/paged_attention_output_gate_templated.json"
SHAPES = [dict(label=f"g{m}", m=m) for m in (1, 4, 8)]
MODE = "graph"
SEEDS = (0, 100)
AA = ["T", "B1"]
B37_ARMS = ["B", "B1"]
HQ, D = 8, 256
SC = D ** -0.5


def rotate(shape):
    return ["query", "gate"]


def gen(shape, seed):
    import qwen3_5_mxfp4__paged_attention_output_gate as sp
    inp = sp._make_inputs(shape["m"], seed, "cuda")
    m = shape["m"]
    out = torch.empty(m, HQ * D, device="cuda")
    ip = inp["kv_indptr"]
    ks, vs = float(inp["k_scale"]), float(inp["v_scale"])
    for r in range(m):
        sl = inp["kv_indices"][int(ip[r]):int(ip[r + 1])].long()
        k = inp["key_cache"][sl, 0].float() * ks
        v = inp["value_cache"][sl, 0].float() * vs
        q = inp["query"][r].float()
        p = torch.softmax(q @ k.t() * SC, -1)
        out[r] = (p @ v).reshape(-1) * torch.sigmoid(inp["gate"][r].float())
    return inp, dict(output=out)


def _T(inp):
    fn = C.tpl("qwen3_5/mxfp4/paged_attention_output_gate_r3/r3_paog_template.py", "paged_attention_output_gate",
               PROF, "mxfp4/paged_attention_output_gate_r3/r3_paog_template.py")

    def call():
        o = fn(inp["query"], inp["key_cache"], inp["value_cache"], inp["kv_indptr"], inp["kv_indices"], inp["gate"],
               scale=SC, max_context=CAP, k_scale=inp["k_scale"], v_scale=inp["v_scale"])
        return dict(output=o)
    return call


def _B(inp):
    from aiter.ops.triton.attention import paged_attention_output_gate as pog
    ok, why = pog.paged_attention_output_gate_supported(inp["query"], inp["key_cache"], inp["value_cache"], inp["gate"], None)
    if not ok:
        raise core.Unavailable(f"paged_attention_output_gate_supported: {why}")

    def call():
        o, _, _ = pog.paged_attention_output_gate_group_fp8_quant(
            inp["query"], inp["key_cache"], inp["value_cache"], inp["kv_indptr"], inp["kv_indices"], inp["gate"],
            scale=SC, max_context=CAP, k_scale=inp["k_scale"], v_scale=inp["v_scale"], quant_dtype=None)
        return dict(output=o)
    return call


_WS = {}


def _B1(inp):
    ab = C.ab()
    fsm = core.sglang_import("sglang.kernels.ops.elementwise.elementwise").fused_sigmoid_mul
    if "ws" not in _WS:
        maxp = CAP // 256
        _WS["ws"] = torch.empty(256 * HQ * maxp * D * 4 + 2 * 256 * HQ * maxp * 4, dtype=torch.uint8, device="cuda")
        _WS["lpl"] = torch.ones(256, dtype=torch.int32, device="cuda")
    ws, lpl = _WS["ws"], _WS["lpl"]
    ip = inp["kv_indptr"]
    maxlen = int((ip[1:] - ip[:-1]).max())
    parts = 16 if maxlen <= 4096 else 128 if maxlen <= 32768 else CAP // 256
    kc = inp["key_cache"].view(-1, 1, 1, D)
    vc = inp["value_cache"].view(-1, 1, 1, D)
    ks, vs = inp["k_scale"].view(1), inp["v_scale"].view(1)
    q, g = inp["query"], inp["gate"]

    def call():
        o = torch.empty(q.shape[0], HQ * D, dtype=torch.bfloat16, device="cuda")
        ab.paged_attention_ragged(o.view(-1, HQ, D), ws, q, kc, vc, SC, ip, inp["kv_indices"], lpl, 1, parts, None,
                                  "fp8_e4m3", "NHD", 0.0, ks, vs, None, 256)
        o = fsm(o, g, inplace=True)
        return dict(output=o)
    return call


def arms(shape):
    return {"T": _T, "B": _B, "B1": _B1}


def check(arm, outs, ref, inp, shape):
    return {"output": core.cmp_close(outs["output"], ref["output"])}
