"""S4: sparse (DSA top-k) MLA decode, GLM-5.2, fp8_scalar KV [P, 576] (unit scale), top-k 2048.

T  : the template the *_templated.json manifest serves at that row (sha checked):
       mx4 (MXFP4 TP4, H=16): glm52/sparse_paged_mla_xl/decode_tp4_small.py      M 1, 2, 16
       mx8 (MXFP4 TP8, H=8) : glm52/sparse_paged_mla_xl/decode_tp8_narrow.py     M 32
                              glm52/sparse_paged_mla_xl/decode_tp8_deep.py       M 128, 256
       fp8 (FP8 TP8, H=8)   : glm52_fp8/sparse_paged_mla_xl/decode_tp8_m8_x.py   M 1, 2, 4, 8, 16
                              glm52_fp8/sparse_paged_mla_r3/spm_tp8_policy_template.py  M 12, 24, 48, 96, 128, 192, 256
                              glm52_fp8/sparse_paged_mla_r3/spm_tp8_split_template.py   M 32, 64
       ep8 (EP8, H=64)      : see EP8 rows below (if present)
B  : AITER exactly as SGLang 703e763d5f calls it with --dsa-decode-backend aiter
     (DeepseekSparseAttnBackend._forward_aiter, run verbatim on a stand-in `self`):
     H<16 -> q.repeat_interleave(16//H) head padding, kv_indptr = cumsum(valid counts),
     get_valid_kv_indices (SGLang Triton compaction), get_mla_metadata_v1 (persistent
     schedule, fp8 KV), aiter.mla.mla_decode_fwd (asm), o[:, ::rep] un-padding.  All per call.
B1 : aiter.ops.triton.attention.sparse_mla.sparse_mla_fwd(dot_precision="bf16")  (not used by SGLang)
B2 : sparse_mla_fwd(dot_precision="fp8")  (fp8_scalar Gluon fp8-dot kernel; bf16 q quantized in-kernel)
X_triton  : SGLang default HIP DSA decode backend ("triton": triton_sparse_mla_decode_splitk) - non-AITER context
X_tilelang: SGLang tilelang_sparse_fwd (--dsa-decode-backend tilelang, our FP8/EP8 serving runs) - context
"""
import os
import types
from types import SimpleNamespace

import torch

from xbench import core

SCALE = 0.0625
TOPK, ROW, DV = 2048, 576, 512
SMALL, LARGE = 262144, 4194304          # LARGE * 576 B = 2.42 GB > 2**31

_MX4 = "glm52_mxfp4_tp4_gfx950/sparse_paged_mla_templated.json"
_MX8 = "glm52_mxfp4_tp8_gfx950/sparse_paged_mla_templated.json"
_FP8 = "glm52_fp8_tp8_gfx950/sparse_paged_mla_templated.json"
_EP8F = "glm52_fp8_tp8_ep8_dp8_gfx950/sparse_paged_mla_templated.json"


def _row(prof, tp, m, H, tpl, profile, pool=SMALL, sfx="", qfp8=False, srcf=None):
    return dict(label=f"{prof}_tp{tp}_h{H}_m{m}{sfx}", prof=prof, tp=tp, m=m, H=H, tpl=tpl,
                profile=profile, pool=pool, qfp8=qfp8, srcf=srcf)


SHAPES = []
for m in (1, 2, 16):
    SHAPES.append(_row("mx", 4, m, 16, "glm52/sparse_paged_mla_xl/decode_tp4_small.py", _MX4))
SHAPES.append(_row("mx", 8, 32, 8, "glm52/sparse_paged_mla_xl/decode_tp8_narrow.py", _MX8))
for m in (128, 256):
    SHAPES.append(_row("mx", 8, m, 8, "glm52/sparse_paged_mla_xl/decode_tp8_deep.py", _MX8))
for m in (1, 2, 4, 8, 16):
    SHAPES.append(_row("fp8", 8, m, 8, "glm52_fp8/sparse_paged_mla_xl/decode_tp8_m8_x.py", _FP8))
for m in (12, 24, 48, 96, 128, 192, 256):
    SHAPES.append(_row("fp8", 8, m, 8, "glm52_fp8/sparse_paged_mla_r3/spm_tp8_policy_template.py", _FP8))
for m in (32, 64):
    SHAPES.append(_row("fp8", 8, m, 8, "glm52_fp8/sparse_paged_mla_r3/spm_tp8_split_template.py", _FP8))
# large-pool (> 2**31 bytes) variants of lead rows
SHAPES.append(_row("mx", 4, 1, 16, "glm52/sparse_paged_mla_xl/decode_tp4_small.py", _MX4, LARGE, "_L"))
SHAPES.append(_row("mx", 4, 16, 16, "glm52/sparse_paged_mla_xl/decode_tp4_small.py", _MX4, LARGE, "_L"))
SHAPES.append(_row("mx", 8, 32, 8, "glm52/sparse_paged_mla_xl/decode_tp8_narrow.py", _MX8, LARGE, "_L"))
SHAPES.append(_row("fp8", 8, 8, 8, "glm52_fp8/sparse_paged_mla_xl/decode_tp8_m8_x.py", _FP8, LARGE, "_L"))
SHAPES.append(_row("fp8", 8, 64, 8, "glm52_fp8/sparse_paged_mla_r3/spm_tp8_split_template.py", _FP8, LARGE, "_L"))

# EP8 (H=64 per rank), unit-scale fp8 query (contract query_quantization unit_scale_fp8)
_EP8F_ACTIVE = "glm52_fp8_tp8_ep8_dp8_gfx950/sparse_paged_mla.json"   # no *_templated.json for this profile
for m, f in ((1, "decode_ep8_m1_m8_x"), (2, "decode_ep8_m1_m8_x"), (4, "decode_ep8_m1_m8_x"), (8, "decode_ep8_m1_m8_x"),
             (12, "decode_ep8_m12_m24_x"), (24, "decode_ep8_m12_m24_x"), (16, "decode_ep8_m16_m32_x"),
             (32, "decode_ep8_m16_m32_x")):
    pr = f"glm52_fp8_ep8/sparse_paged_mla_xl/{f}.py"
    SHAPES.append(_row("fp8ep8", 8, m, 64, pr, _EP8F_ACTIVE, qfp8=True, srcf=pr))
_MXEP8 = "glm52_mxfp4_tp8_ep8_dp8_gfx950/sparse_paged_mla_templated.json"
for m, f in ((12, "decode_ep8_template"), (24, "decode_ep8_template"), (16, "decode_ep8_d16_d32_template"),
             (32, "decode_ep8_d16_d32_template")):
    SHAPES.append(_row("mxep8", 8, m, 64, f"glm52_ep8/sparse_paged_mla_r3/{f}.py", _MXEP8, qfp8=True))

MODE = "graph"
SEEDS = (0, 100)
AA = ["T", "B"]
_ONLY = os.environ.get("S4_ONLY")          # optional label filter for quick runs
if _ONLY:
    SHAPES = [s for s in SHAPES if s["label"] in _ONLY.split(",")]


def rotate(shape):
    return ["query", "slots"]          # KV pool shared (it is the serving working set)


def gen(shape, seed):
    if shape["prof"] == "mx":
        import glm52__sparse_paged_mla as sp
        key = f"tp{shape['tp']}m{shape['m']}"
    else:
        import glm52_fp8__sparse_paged_mla as sp
        key = f"tp{shape['tp']}m{shape['m']}"
    if shape["prof"] in ("fp8ep8", "mxep8"):
        import glm52_fp8__sparse_paged_mla as sp
    sp._POOL = shape["pool"]
    sp._HEADS = dict(sp._HEADS)
    sp._HEADS[shape["tp"]] = shape["H"]
    d = sp.make_inputs(key, seed, "cuda")
    q = d["query"]
    kv = d.get("kv_cache", d.get("cache"))
    slots = d.get("selected_slots", d.get("slots"))
    assert q.shape == (shape["m"], shape["H"], ROW) and kv.shape == (shape["pool"], ROW)
    if shape.get("qfp8"):
        q = q.to(torch.float8_e4m3fn)          # unit-scale fp8 query, as the EP8 adapters pass it
    ref = dict(out=_reference(q, kv, slots))
    return dict(query=q, kv=kv, slots=slots), ref


def _reference(q, kv, slots):
    m, H, _ = q.shape
    out = torch.empty(m, H, DV, dtype=torch.float32, device=q.device)
    for i in range(m):
        idx = slots[i][slots[i] >= 0].long()
        K = kv[idx].float()                                  # [n, 576]
        s = (q[i].float() @ K.t()) * SCALE         # q may be fp32 (q8 reference)                   # [H, n]
        p = torch.softmax(s, -1)
        out[i] = p @ K[:, :DV]
    return out


# ------------------------------------------------------------------ arms
def _T(shape):
    def build(inp):
        from xbench import tload
        fn = tload.load(shape["tpl"], "sparse_paged_mla", shape["profile"], shape.get("srcf"))
        q, kv, sl = inp["query"], inp["kv"], inp["slots"]
        if shape["prof"] == "mx":
            return lambda: dict(out=fn(q, kv, sl, softmax_scale=SCALE))
        return lambda: dict(out=fn(q, kv, sl, softmax_scale=SCALE, value_dim=DV, quantize_query=True))
    return build


_DSA = None


def _dsa():
    global _DSA
    if _DSA is None:
        _DSA = core.sglang_import("sglang.srt.layers.attention.dsa_backend")
    return _DSA


def _B(inp):
    dsa = _dsa()
    Cls = dsa.DeepseekSparseAttnBackend
    q, kv, slots = inp["query"], inp["kv"], inp["slots"]
    m, H, D = q.shape
    dev = q.device
    ns = SimpleNamespace(
        device=dev, num_q_heads=H, need_pad_heads=H < 16,
        head_repeat_factor=16 // H if H < 16 else 1, num_head_padded=max(H, 16),
        aiter_dsa_max_split_per_batch=64, aiter_dsa_metadata_capacity=0,
        aiter_dsa_metadata_max_seqlen_q=0, aiter_dsa_metadata_q_dtype=None,
        aiter_dsa_metadata_kv_dtype=None, aiter_dsa_kv_last_page_lens=None,
        aiter_dsa_work_metadata=None,
        aiter_dsa_identity_scale=torch.ones((), dtype=torch.float32, device=dev),
        kv_indptr=torch.zeros(m + 1, dtype=torch.int32, device=dev),
        kv_indices=torch.zeros(m * TOPK, dtype=torch.int32, device=dev),
    )
    for name in ("_make_aiter_dsa_decode_metadata_buffer", "_ensure_aiter_dsa_decode_metadata_buffer",
                 "_prepare_aiter_dsa_decode_metadata"):
        setattr(ns, name, types.MethodType(getattr(Cls, name), ns))
    # SGLang allocates the persistent-metadata buffers at init (max_bs); do it here, untimed.
    ns._ensure_aiter_dsa_decode_metadata_buffer(max_seqlen_q=1, batch_size=m,
                                                q_dtype=q.dtype, kv_dtype=kv.dtype)
    layer = SimpleNamespace(tp_q_head_num=H, head_dim=D, v_head_dim=DV, scaling=SCALE,
                            logit_cap=0.0, k_scale=None)
    meta = SimpleNamespace(cu_seqlens_q=torch.arange(m + 1, dtype=torch.int32, device=dev),
                           max_seq_len_q=1)
    fwd = types.MethodType(Cls._forward_aiter, ns)

    def call():
        o = fwd(q_all=q, kv_cache=kv, page_table_1=slots, layer=layer, metadata=meta, bs=m)
        return dict(out=o.reshape(m, H, DV))
    return call


def _sparse(dp):
    def build(inp):
        from aiter.ops.triton.attention.sparse_mla import sparse_mla_fwd
        q, kv, slots = inp["query"], inp["kv"], inp["slots"]
        m = q.shape[0]
        dev = q.device
        indptr = torch.arange(0, (m + 1) * TOPK, TOPK, dtype=torch.int32, device=dev)
        flat = slots.reshape(-1)
        ks = torch.ones(1, dtype=torch.float32, device=dev)
        qs = ks if q.dtype == torch.float8_e4m3fn else None

        def call():
            out, _ = sparse_mla_fwd(q, kv, indptr, flat, SCALE, kv_scale=ks, kv_lora_rank=512,
                                    qk_rope_head_dim=64, has_invalid=True, dot_precision=dp, q_scale=qs)
            return dict(out=out)
        return call
    return build


def _compact(slots):
    """SGLang's per-call compaction (as in _forward_aiter): valid counts -> indptr, Triton compaction."""
    dsa = _dsa()
    m = slots.shape[0]
    indptr = torch.zeros(m + 1, dtype=torch.int32, device=slots.device)
    indices = torch.zeros(m * TOPK, dtype=torch.int32, device=slots.device)

    def run():
        indptr[1:m + 1] = torch.cumsum((slots != -1).sum(dim=1), dim=0)
        dsa.get_valid_kv_indices(slots, indptr, indices, m)
        return indptr, indices
    return run


def _B3(inp):
    """sparse_mla_fwd fp8 dots on the dedicated Gluon fp8_scalar kernel: needs has_invalid=False,
    so the -1 sentinels are compacted per call exactly like SGLang's aiter path does (counted)."""
    from aiter.ops.triton.attention.sparse_mla import sparse_mla_fwd
    q, kv, slots = inp["query"], inp["kv"], inp["slots"]
    ks = torch.ones(1, dtype=torch.float32, device=q.device)
    comp = _compact(slots)

    def call():
        indptr, indices = comp()
        out, _ = sparse_mla_fwd(q, kv, indptr, indices, SCALE, kv_scale=ks, kv_lora_rank=512,
                                qk_rope_head_dim=64, has_invalid=False, dot_precision="fp8",
                                q_scale=ks if q.dtype == torch.float8_e4m3fn else None)
        return dict(out=out)
    return call


def _Xk_fp8gluon(inp):
    """Kernel-only lower bound: B3 with the compacted list precomputed (not timed)."""
    from aiter.ops.triton.attention.sparse_mla import sparse_mla_fwd
    q, kv, slots = inp["query"], inp["kv"], inp["slots"]
    ks = torch.ones(1, dtype=torch.float32, device=q.device)
    indptr, indices = _compact(slots)()
    indptr, indices = indptr.clone(), indices[: int(indptr[-1].item())].clone()

    def call():
        out, _ = sparse_mla_fwd(q, kv, indptr, indices, SCALE, kv_scale=ks, kv_lora_rank=512,
                                qk_rope_head_dim=64, has_invalid=False, dot_precision="fp8",
                                q_scale=ks if q.dtype == torch.float8_e4m3fn else None)
        return dict(out=out)
    return call


def _Xk_asm(inp):
    """Kernel-only lower bound for B: mla_decode_fwd asm with padded q, compacted indices and the
    persistent metadata all prepared once outside the timed region."""
    dsa = _dsa()
    q, kv, slots = inp["query"], inp["kv"], inp["slots"]
    m, H, D = q.shape
    dev = q.device
    rep = 16 // H if H < 16 else 1
    qk = q.repeat_interleave(rep, dim=1).contiguous() if rep > 1 else q
    ok = torch.empty(m, H * rep, DV, dtype=torch.bfloat16, device=dev)
    indptr, indices = _compact(slots)()
    qo = torch.arange(m + 1, dtype=torch.int32, device=dev)
    ns = SimpleNamespace(device=dev, num_head_padded=H * rep, aiter_dsa_max_split_per_batch=64,
                         aiter_dsa_metadata_capacity=0, aiter_dsa_metadata_max_seqlen_q=0,
                         aiter_dsa_metadata_q_dtype=None, aiter_dsa_metadata_kv_dtype=None,
                         aiter_dsa_kv_last_page_lens=None, aiter_dsa_work_metadata=None)
    Cls = dsa.DeepseekSparseAttnBackend
    for name in ("_make_aiter_dsa_decode_metadata_buffer", "_ensure_aiter_dsa_decode_metadata_buffer",
                 "_prepare_aiter_dsa_decode_metadata"):
        setattr(ns, name, types.MethodType(getattr(Cls, name), ns))
    kw = ns._prepare_aiter_dsa_decode_metadata(qo, indptr, m, 1, qk.dtype, kv.dtype)
    lpl = kw.pop("kv_last_page_lens")
    one = torch.ones((), dtype=torch.float32, device=dev)

    def call():
        dsa.mla_decode_fwd(qk, kv.view(-1, 1, 1, D), ok, qo, indptr, indices, lpl, 1, sm_scale=SCALE,
                           logit_cap=0.0, q_scale=one, kv_scale=one, **kw)
        return dict(out=ok[:, ::rep, :])
    return call


def _X_triton(inp):
    _dsa()
    mod = core.sglang_import("sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode")
    q, kv, slots = inp["query"], inp["kv"], inp["slots"]
    ws = []
    kv3 = kv.view(-1, 1, ROW)
    idx = slots.unsqueeze(1)

    def call():
        o = mod.triton_sparse_mla_decode_splitk(q_nope=q[..., :DV], q_rope=q[..., DV:], kv=kv3,
                                                indices=idx, sm_scale=SCALE, d_v=DV, workspace=ws)
        return dict(out=o.squeeze(0))
    return call


def _X_tilelang(inp):
    _dsa()
    mod = core.sglang_import("sglang.kernels.ops.attention.dsa.tilelang_kernel")
    q, kv, slots = inp["query"], inp["kv"], inp["slots"]
    kv3 = kv.view(-1, 1, ROW)
    idx = slots.unsqueeze(1)

    def call():
        return dict(out=mod.tilelang_sparse_fwd(q=q, kv=kv3, indices=idx, sm_scale=SCALE, d_v=DV).squeeze(0))
    return call


def arms(shape):
    return {"T": _T(shape), "B": _B, "B1": _sparse("bf16"), "B2": _sparse("fp8"), "B3": _B3,
            "Xk_asm": _Xk_asm, "Xk_fp8gluon": _Xk_fp8gluon,
            "X_triton": _X_triton, "X_tilelang": _X_tilelang}


def _q8(q):
    """Reference fp8 query: per-(row, head) e4m3 with scale amax/448 (the kernels' fp8-q paths
    quantize per head or per head-tile)."""
    qf = q.float()
    s = qf.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
    return ((qf / s).to(torch.float8_e4m3fn).float() * s)


def check(arm, outs, ref, inp, shape):
    """bf16 criterion vs the fp32 reference with the bf16 query ('out').  Arms that quantize the
    query to fp8 by design (fp8-profile T with quantize_query=True, fp8-dot AITER arms, SGLang's
    triton/tilelang) are also checked vs a reference on an fp8-quantized query ('out_q8ref');
    their verdict uses that reference (stated in the notes)."""
    st = {"out": core.cmp_close(outs["out"], ref["out"])}
    unit = (arm == "T" and shape["prof"] == "fp8") or arm in ("X_triton", "X_tilelang")
    perhead = arm in ("B2", "B3", "Xk_fp8gluon")
    if (unit or perhead) and inp["query"].dtype != torch.float8_e4m3fn:
        q = inp["query"]
        q8 = q.to(torch.float8_e4m3fn).float() if unit else _q8(q)
        r8 = _reference(q8, inp["kv"], inp["slots"])
        k = "out_q8unit_ref" if unit else "out_q8head_ref"
        st[k] = core.cmp_close(outs["out"], r8)
        st["out"]["pass_"] = st[k]["pass_"]   # judged on a reference with the same fp8 query
        st["out"]["judged_on"] = k
    return st
