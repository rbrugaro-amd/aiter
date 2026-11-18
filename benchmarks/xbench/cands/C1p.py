"""C1 prefill: MiniMax-M2.5 paged GQA causal prefill, FP8 KV (page 1).

T  : paged_gqa_prefill_r3_template.py::paged_gqa_attention (all fp8/mxfp4 profiles, ranges
     1024-4096 / 4097-6143 / 6144-8192), called as the binder does: q_scale = k_scale.
B  : SGLang 703e763d5f aiter backend forward_extend for MiniMax (head_dim 128, GQA 6: the HD128
     FP8 asm prefill needs a GQA ratio in {1,2,4,8,16} and the paged asm needs hd 256, so it
     falls through to the "NHD path"): q -> fp8 via scaled_fp8_quant at the backend's static
     q_scale 1.0, then aiter.mha_batch_prefill_func(q8, k, v, qo_indptr, kv_indptr, kv_indices,
     max_q, max_k, causal, q/k/v_descale) (CK batch prefill).
B1 : aiter unified_attention (Triton, SGLANG_USE_AITER_UNIFIED_ATTN path), q fp8 quant counted.
kv_indices: live slots + 256-entry tail filled with kv_indices[0] (AiterIndicesUpdaterPrefill).
"""
import torch

from xbench import core
from xbench.cands import C1_common as C

PROF = "minimax_m2_5_fp8_tp4_gfx950/paged_gqa_prefill_templated.json"
# (tokens, sequences, prefix per sequence)
_SC = [(1024, 1, 0), (4096, 1, 0), (6143, 1, 0), (8192, 1, 0), (4096, 4, 0), (8192, 1, 8192)]
SHAPES = [dict(label=f"p{t}x{s}" + (f"+{p}" if p else ""), tokens=t, seqs=s, prefix=p,
               caches=["rotating"]) for t, s, p in _SC]
MODE = "graph"
SEEDS = (0, 100)
AA = ["T", "B"]


def _qlens(s):
    base = s["tokens"] // s["seqs"]
    return [base] * (s["seqs"] - 1) + [s["tokens"] - base * (s["seqs"] - 1)]


def gen(shape, seed):
    pseed = C.pool_seed(seed)
    pool = C.get_pool(pseed)
    g = torch.Generator(device="cuda").manual_seed(seed * 1_000_003 + shape["tokens"] * 7 + shape["prefix"])
    qlens = _qlens(shape)
    kvlens = [q + shape["prefix"] for q in qlens]
    total = sum(kvlens)
    idx = torch.empty(total + 256, dtype=torch.int32, device="cuda")
    if pseed == 0:
        idx[:total] = torch.randperm(C.POOL, generator=g, device="cuda")[:total].to(torch.int32)
    else:
        half = C.POOL // 2
        base = torch.randperm(half, generator=g, device="cuda")[:(total + 1) // 2].to(torch.int32)
        n = total // 2
        idx[0:2 * n:2] = base[:n]
        idx[1:2 * n:2] = base[:n] + half
        if total % 2:
            idx[total - 1] = base[-1]
    idx[total:] = idx[0]
    q = torch.randn(shape["tokens"], C.HQ, C.D, generator=g, device="cuda").to(torch.bfloat16)
    cq = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0).tolist()), dtype=torch.int32, device="cuda")
    ck = torch.tensor([0] + list(torch.tensor(kvlens).cumsum(0).tolist()), dtype=torch.int32, device="cuda")
    out = torch.empty(shape["tokens"], C.HQ, C.D, device="cuda")
    for i in range(shape["seqs"]):
        sl = idx[ck[i]:ck[i + 1]].long()
        out[cq[i]:cq[i + 1]] = C.ref_attention(q[cq[i]:cq[i + 1]], pool["key_cache"], pool["value_cache"],
                                               sl, causal_prefix=shape["prefix"])
    ks = torch.full((1,), C.K_SCALE, device="cuda")
    inp = dict(query=q, kv_indptr=ck, kv_indices=idx, qo_indptr=cq, k_scale=ks,
               v_scale=torch.full((1,), C.V_SCALE, device="cuda"), q_scale=ks.clone(),
               pool_seed=torch.tensor([pseed]))
    return inp, dict(output=out)


def rotate(shape):
    return ["query"]


def _T(shape):
    def build(inp):
        m = C.with_pool(inp)
        fn = C.tpl("minimax_m2_5/paged_gqa_prefill_r3/paged_gqa_prefill_r3_template.py", "paged_gqa_attention",
                   PROF, "paged_gqa_prefill_r3/paged_gqa_prefill_r3_template.py")
        mq = max(_qlens(shape))

        def call():
            o = fn(m["query"], m["key_cache"], m["value_cache"], m["kv_indptr"], m["kv_indices"], m["qo_indptr"],
                   m["k_scale"], m["v_scale"], softmax_scale=C.SCALE, max_query_length=mq, q_scale=m["q_scale"])
            return dict(output=o)
        return call
    return build


def _B(shape):
    def build(inp):
        ab = C.ab()
        m = C.with_pool(inp)
        T = shape["tokens"]
        mq = max(_qlens(shape))
        mk = mq + shape["prefix"]
        qs = torch.ones(1, dtype=torch.float32, device="cuda")
        bs0 = shape["seqs"] + 1

        def call():
            q8, _ = ab.scaled_fp8_quant(m["query"].reshape(T, -1).contiguous(), qs)
            o = ab.mha_batch_prefill_func(
                q8.view(T, C.HQ, C.D), m["key_cache"], m["value_cache"], m["qo_indptr"][:bs0],
                m["kv_indptr"][:bs0], m["kv_indices"], mq, mk, softmax_scale=C.SCALE, causal=True,
                logits_soft_cap=0.0, alibi_slopes=None, return_lse=False, return_attn_probs=False,
                window_size=(-1, -1), sink_ptr=None, q_descale=qs, k_descale=m["k_scale"],
                v_descale=m["v_scale"])
            if o.dtype != torch.bfloat16:
                o = o.to(torch.bfloat16)
            return dict(output=o)
        return call
    return build


def _B1(shape):
    def build(inp):
        ab = C.ab()
        m = C.with_pool(inp)
        T, S = shape["tokens"], shape["seqs"]
        mq = max(_qlens(shape))
        mk = mq + shape["prefix"]
        kvl = (m["kv_indptr"][1:] - m["kv_indptr"][:-1])
        bt = torch.zeros(S, mk, dtype=torch.int32, device="cuda")
        for i in range(S):
            n = int(kvl[i])
            bt[i, :n] = m["kv_indices"][int(m["kv_indptr"][i]):int(m["kv_indptr"][i]) + n]
        seqused = kvl.to(torch.int64)
        qs = torch.ones(1, dtype=torch.float32, device="cuda")
        kv = m["key_cache"].view(-1, 1, C.HK, C.D)
        vv = m["value_cache"].view(-1, 1, C.HK, C.D)

        def call():
            o = torch.empty_like(m["query"])
            q8, _ = ab.scaled_fp8_quant(m["query"].reshape(T, -1).contiguous(), qs)
            ab.unified_attention(q=q8.view(T, C.HQ, C.D), k=kv, v=vv, out=o, cu_seqlens_q=m["qo_indptr"],
                                 seqused_k=seqused, max_seqlen_q=mq, max_seqlen_k=mk, softmax_scale=C.SCALE,
                                 causal=True, window_size=(-1, -1), block_table=bt, softcap=0.0,
                                 q_descale=qs, k_descale=m["k_scale"], v_descale=m["v_scale"], sinks=None)
            return dict(output=o)
        return call
    return build


def arms(shape):
    return {"T": _T(shape), "B": _B(shape), "B1": _B1(shape)}


def check(arm, outs, ref, inp, shape):
    return {"output": core.cmp_close(outs["output"], ref["output"])}
