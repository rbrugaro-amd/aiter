"""C1 decode: MiniMax-M2.5 paged GQA decode, FP8 KV (Hq 12 / Hk 2 / D 128, page 1).

T     : templated row of the fp8 TP4-EP4 profile (rows 1..16 -> paged_gqa_decode_split_r3_template
        ::paged_gqa_attention_ep4, 32/64 -> paged_gqa_decode_r3_template::paged_gqa_attention).
Ttp4  : rows 4 only, fp8 TP4 profile row (split_r3::paged_gqa_attention_tp4).
B     : SGLang 703e763d5f aiter backend forward_decode as called for MiniMax (no SWA, default
        paged pool): aiter.paged_attention_ragged(o, workspace, q, k.view(-1,1,2,128), ...,
        block 1, num_partitions per SGLang's graph bucket, "fp8_e4m3", "NHD", 0.0, ks, vs,
        None, 256).  Graph capacity 65536 -> buckets p16 (max_seq <= 4096) / pmax 256.
B2    : same op with the partitions the request needs (ceil(max_seq/256)) - best AITER config.
B1    : aiter unified_attention (Triton), SGLang's SGLANG_USE_AITER_UNIFIED_ATTN path (q fp8
        quant via scaled_fp8_quant counted; 2-D page table prepared outside the graph).
kv_indices: graph capacity = 256 * 65536 entries (live CSR prefix + stale valid slots), as
served; '_tight' rows use exactly the live entries.
"""
import torch

from xbench import core
from xbench.cands import C1_common as C

CAP = 65536
GRAPH_NUMEL = 256 * CAP
PROF_E = "minimax_m2_5_fp8_tp4_ep4_gfx950/paged_gqa_decode_templated.json"
PROF_F = "minimax_m2_5_fp8_tp4_gfx950/paged_gqa_decode_templated.json"

_rows_ctx = [(m, 16384) for m in (1, 2, 4, 8, 16, 32, 64)] + \
    [(m, c) for c in (2048, 65536) for m in (1, 8, 64)]
SHAPES = [dict(label=f"m{m}c{c // 1024}k", m=m, ctx=c, tight=False) for m, c in _rows_ctx] + \
    [dict(label=f"m{m}c16k_tight", m=m, ctx=16384, tight=True) for m in (1, 8, 64)]
MODE = "graph"
SEEDS = (0, 100)
AA = ["T", "B"]

_rot_pending = False
_rot_mode = False


def rotate(shape):
    global _rot_pending
    _rot_pending = True
    return ["query", "kv_indices"]


def gen(shape, seed):
    m, ctx = shape["m"], shape["ctx"]
    pseed = C.pool_seed(seed)
    pool = C.get_pool(pseed)
    g = torch.Generator(device="cuda").manual_seed(seed * 1_000_003 + m * 131 + ctx)
    live = m * ctx
    numel = live if shape["tight"] else GRAPH_NUMEL
    if pseed == 0:
        idx = torch.randint(0, C.POOL, (numel,), generator=g, device="cuda", dtype=torch.int32)
    else:
        half = C.POOL // 2
        idx = torch.randint(0, half, (numel,), generator=g, device="cuda", dtype=torch.int32)
        npair = live // 2
        idx[1:2 * npair:2] = idx[0:2 * npair:2] + half
    indptr = torch.arange(0, live + 1, ctx, device="cuda", dtype=torch.int32)
    q = torch.randn(m, C.HQ, C.D, generator=g, device="cuda").to(torch.bfloat16)
    inp = dict(query=q, kv_indptr=indptr, kv_indices=idx,
               qo_indptr=torch.arange(m + 1, device="cuda", dtype=torch.int32),
               k_scale=torch.full((1,), C.K_SCALE, device="cuda"),
               v_scale=torch.full((1,), C.V_SCALE, device="cuda"),
               pool_seed=torch.tensor([pseed]))
    out = torch.empty(m, C.HQ, C.D, device="cuda")
    for r in range(m):
        out[r] = C.ref_attention(q[r:r + 1], pool["key_cache"], pool["value_cache"],
                                 idx[r * ctx:(r + 1) * ctx].long())[0]
    return inp, dict(output=out)


def _shift(m):
    """rotating ring: give every ring member its own slot set (distinct KV lines)."""
    if _rot_mode:
        t = m["kv_indices"]
        if not getattr(t, "_c1_shifted", False):
            off = (t.data_ptr() // 256) % C.POOL
            t.copy_(((t.long() + off) % C.POOL).to(torch.int32))
            t._c1_shifted = True
    return m


def _T_fn(shape, tp4=False):
    m = shape["m"]
    if tp4:
        return C.tpl("minimax_m2_5/paged_gqa_decode_r3/paged_gqa_decode_split_r3_template.py",
                     "paged_gqa_attention_tp4", PROF_F, "paged_gqa_decode_r3/paged_gqa_decode_split_r3_template.py")
    if m <= 16:
        return C.tpl("minimax_m2_5/paged_gqa_decode_r3/paged_gqa_decode_split_r3_template.py",
                     "paged_gqa_attention_ep4", PROF_E, "paged_gqa_decode_r3/paged_gqa_decode_split_r3_template.py")
    return C.tpl("minimax_m2_5/paged_gqa_decode_r3/paged_gqa_decode_r3_template.py",
                 "paged_gqa_attention", PROF_E, "paged_gqa_decode_r3/paged_gqa_decode_r3_template.py")


def _mk_T(shape, tp4=False):
    def build(inp):
        m = C.with_pool(_shift(inp))
        fn = _T_fn(shape, tp4)

        def call():
            o = fn(m["query"], m["key_cache"], m["value_cache"], m["kv_indptr"], m["kv_indices"],
                   m["qo_indptr"], m["k_scale"], m["v_scale"], softmax_scale=C.SCALE,
                   max_query_length=1, q_scale=None)
            return dict(output=o)
        return call
    return build


_WS = {}


def _workspace(parts):
    """SGLang sizes it once for max_bs 256 x max_num_partitions (CAP / 256)."""
    if "ws" not in _WS:
        maxp = (CAP + 255) // 256
        n = 256 * C.HQ * maxp * C.D * 4 + 2 * (256 * C.HQ * maxp) * 4
        _WS["ws"] = torch.empty(n, dtype=torch.uint8, device="cuda")
        _WS["lpl"] = torch.ones(256, dtype=torch.int32, device="cuda")
    return _WS["ws"], _WS["lpl"]


def _mk_B(shape, partitions):
    def build(inp):
        ab = C.ab()
        m = C.with_pool(_shift(inp))
        ws, lpl = _workspace(partitions)
        kv = m["key_cache"].view(-1, 1, C.HK, C.D)
        vv = m["value_cache"].view(-1, 1, C.HK, C.D)
        q = m["query"]

        def call():
            o = torch.empty_like(q)
            ab.paged_attention_ragged(o.view(-1, C.HQ, C.D), ws, q.view(-1, C.HQ, C.D), kv, vv, C.SCALE,
                                      m["kv_indptr"], m["kv_indices"], lpl, 1, partitions, None,
                                      "fp8_e4m3", "NHD", 0.0, m["k_scale"], m["v_scale"], None, 256)
            return dict(output=o)
        return call
    return build


def _mk_B1(shape):
    def build(inp):
        ab = C.ab()
        m = C.with_pool(_shift(inp))
        rows, ctx = shape["m"], shape["ctx"]
        live = m["kv_indices"][:rows * ctx]
        # SGLang unified decode reads a (bs, max_context) page table; built outside the graph.
        bt = torch.zeros(rows, CAP, dtype=torch.int32, device="cuda")
        bt[:, :ctx] = live.view(rows, ctx)
        seqused = torch.full((rows,), ctx, dtype=torch.int64, device="cuda")
        qs = torch.ones(1, dtype=torch.float32, device="cuda")
        kv = m["key_cache"].view(-1, 1, C.HK, C.D)
        vv = m["value_cache"].view(-1, 1, C.HK, C.D)
        q = m["query"]

        def call():
            o = torch.empty_like(q)
            q8, _ = ab.scaled_fp8_quant(q.reshape(rows, -1).contiguous(), qs)
            ab.unified_attention(q=q8.view(rows, C.HQ, C.D), k=kv, v=vv, out=o, cu_seqlens_q=m["qo_indptr"],
                                 seqused_k=seqused, max_seqlen_q=1, max_seqlen_k=CAP, softmax_scale=C.SCALE,
                                 causal=True, window_size=(-1, -1), block_table=bt, softcap=0.0,
                                 q_descale=qs, k_descale=m["k_scale"], v_descale=m["v_scale"], sinks=None)
            return dict(output=o)
        return call
    return build


def arms(shape):
    global _rot_mode, _rot_pending
    _rot_mode, _rot_pending = _rot_pending, False
    ctx = shape["ctx"]
    sg_parts = 16 if ctx <= 4096 else (CAP + 255) // 256
    need = (ctx + 255) // 256
    a = {"T": _mk_T(shape), "B": _mk_B(shape, sg_parts)}
    if need != sg_parts:
        a["B2"] = _mk_B(shape, need)
    if not shape["tight"]:
        a["B1"] = _mk_B1(shape)
    if shape["m"] == 4:
        a["Ttp4"] = _mk_T(shape, tp4=True)
    return a


def check(arm, outs, ref, inp, shape):
    return {"output": core.cmp_close(outs["output"], ref["output"])}
