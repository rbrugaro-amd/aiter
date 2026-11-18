"""S6: DSA indexer prepare (GLM-5.2 MXFP4; indexer weights bf16, replicated -> same shapes TP4/TP8).

Full prepare (decode):  q = q_lora @ wq_b.T, k = x @ wk.T, gate = x @ weights_proj.T;
  k: LayerNorm -> interleaved rope(64) -> Hadamard(128) -> ue8m0 fp8 quant -> index-K cache (page 64,
     16x16 preshuffled, fp32 scale tail);  q: rope -> Hadamard -> ue8m0 fp8;  weights = gate*32^-.5*q_scale*128^-.5
Key-only prefill: only the k path + cache write.

Arms
  T        : template (r3 decode_tp4_template m<=16; xl decode_tp8_lds_x m128/256; xl prefill_tp4_x key-only)
  B        : full: SGLang 703e763d5f Indexer._aiter_fused_fp8_prepare_and_store, replicated line by line:
             3 x ReplicatedLinear bf16 (UnquantizedLinearMethod.apply -> AITER tgemm / AITER triton a16w16)
             + aiter.indexer_qk_rope_quant_and_cache (via sglang aiter_fused_fp8_qk_write).  This AITER
             writer omits the Hadamard (SGLang then skips it on q and k: q.k is invariant), so its outputs are
             checked against the no-Hadamard reference.
             key-only: SGLang Indexer._get_k_bf16 (called unbound on a stand-in module: wk linear, sglang
             LayerNorm, AITER RotaryEmbedding, fast_hadamard_transform) + aiter.indexer_k_quant_and_cache
             (SGLang _write_index_k_cache aiter branch).  This is SGLang's default k-only path on gfx950.
  B1       : key-only with the AITER-writer configuration (_get_k_bf16(rotate=False): no Hadamard).
  X_fused  : full, m<=48: SGLang's default gfx950 decode path (Gfx950FusedIndexer.run nodes 1-2: sglang JIT
             HIP dual_gemv_bf16 + indexer_qk_rope_hadamard_quant_and_cache).  NOT AITER; context only.
"""
import types

import torch

from xbench import core

H, D, HID, QL, PAGE = 32, 128, 6144, 2048, 64
EPS = 1e-6
FP8 = torch.float8_e4m3fn
FULL_PROFILE4 = "glm52_mxfp4_tp4_gfx950/indexer_prepare_templated.json"
FULL_PROFILE8 = "glm52_mxfp4_tp8_gfx950/indexer_prepare_templated.json"
KO_PROFILE4 = "glm52_mxfp4_tp4_gfx950/indexer_prepare_key_only_templated.json"

SHAPES = ([dict(label=f"m{m}", m=m, kind="full", tp=4) for m in (1, 2, 4, 8, 16)]
          + [dict(label=f"m{m}", m=m, kind="full", tp=8) for m in (128, 256)]
          + [dict(label=f"k{m}", m=m, kind="key", tp=4, caches=["rotating"]) for m in (4193, 8192, 16384)])
MODE = "graph"
SEEDS = (0, 100)
AA = ["T", "B"]


# ----------------------------------------------------------------------------- inputs / reference
def gen(shape, seed):
    if shape["kind"] == "full":
        import glm52__indexer_prepare as sp
        inp = sp._make_inputs((shape["tp"], shape["m"], 0), seed, "cuda")
    else:
        import glm52__indexer_prepare_key_only as sp
        inp = sp._make_inputs((shape["tp"], shape["m"], 0), seed, "cuda")
        inp.pop("q_latent")   # a view of x; rebuilt in the builders
    ref = {"cache_orig": inp["cache"].clone()}
    for hada in (True, False):
        r = _reference(inp, shape["kind"] == "full", hada)
        sfx = "" if hada else "_noH"
        for k, v in r.items():
            ref[k + sfx] = v
    return inp, ref


def _bf(t):
    return t.to(torch.bfloat16).float()


def _hadamard(n):
    i = torch.arange(n)
    bits = (i[:, None] & i[None, :])
    par = torch.zeros_like(bits)
    b = bits.clone()
    while bool((b > 0).any()):
        par ^= b & 1
        b >>= 1
    return (1.0 - 2.0 * par.float())


def _rope_interleaved(x, cos, sin):
    # x [..., 128]; pairs (2i, 2i+1) for i < 32
    x = x.clone()
    e, o = x[..., 0:64:2].clone(), x[..., 1:64:2].clone()
    x[..., 0:64:2] = e * cos - o * sin
    x[..., 1:64:2] = o * cos + e * sin
    return x


def _quant(v):
    amax = v.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    codes = (v / scale).to(FP8)
    return codes, scale.squeeze(-1)


def _reference(inp, full, hada):
    """fp64 projections, fp32 elementwise; bf16 rounding at the points every implementation
    (template and SGLang/AITER) materializes bf16: GEMM output, after norm, after rope, after Hadamard."""
    x = inp["x"].double()
    pos = inp["positions"]
    cs = inp["cos_sin"].float()[pos]
    cos, sin = cs[:, :32], cs[:, 32:]
    hm = _hadamard(D).to(x.device) * D ** -0.5
    out = {}
    k = _bf((x @ inp["wk"].double().t()).float())
    mu = k.mean(-1, keepdim=True)
    var = ((k - mu) ** 2).mean(-1, keepdim=True)
    k = _bf((k - mu) * torch.rsqrt(var + EPS) * inp["k_gamma"].float() + inp["k_beta"].float())
    k = _bf(_rope_interleaved(k, cos, sin))
    if hada:
        k = _bf(k @ hm.t())
    out["k_codes"], out["k_scale"] = _quant(k)
    out["k_val"] = k
    if full:
        q = _bf((inp["q_latent"].double() @ inp["wq"].double().t()).float()).view(-1, H, D)
        q = _bf(_rope_interleaved(q, cos[:, None], sin[:, None]))
        if hada:
            q = _bf(q @ hm.t())
        out["q_codes"], out["q_scale"] = _quant(q)
        out["q_val"] = q
        gate = _bf((x @ inp["wgate"].double().t()).float())
        out["weights"] = gate * H ** -0.5 * out["q_scale"] * D ** -0.5
    return out


def _decode_cache(cache_u8, slots):
    """index-K cache [pages,64,132] u8 (16x16 preshuffled codes + fp32 scale tail) -> codes [m,128], scale [m]."""
    flat = cache_u8.reshape(-1)
    s = slots.clamp_min(0)
    page, off = s // PAGE, s % PAGE
    d = torch.arange(D, device=flat.device)
    base = page * PAGE * 132
    idx = base[:, None] + (off // 16 * 2048)[:, None] + (d // 16 * 256)[None, :] + (off % 16 * 16)[:, None] + (d % 16)[None, :]
    codes = flat[idx].view(FP8)
    sidx = base + PAGE * 128 + off * 4
    sb = torch.stack([flat[sidx + i] for i in range(4)], -1).contiguous().view(torch.float32).view(-1)
    return codes, sb, idx, sidx


# ----------------------------------------------------------------------------- arms
def _tpl(shape):
    from xbench import tload
    if shape["kind"] == "key":
        return tload.load("glm52/indexer_prepare_key_only_xl/prefill_tp4_x.py", "indexer_prepare_key_only",
                          KO_PROFILE4, "indexer_prepare_key_only_xl/prefill_tp4_x.py")
    if shape["m"] <= 16:
        return tload.load("glm52/indexer_prepare_r3/decode_tp4_template.py", "indexer_prepare",
                          FULL_PROFILE4, "indexer_prepare_r3/decode_tp4_template.py")
    return tload.load("glm52/indexer_prepare_xl/decode_tp8_lds_x.py", "indexer_prepare",
                      FULL_PROFILE8, "indexer_prepare_xl/decode_tp8_lds_x.py")


def _mk_T(shape):
    full = shape["kind"] == "full"

    def build(inp):
        fn = _tpl(shape)
        ql = inp["q_latent"] if full else inp["x"][:, :QL]
        args = (inp["x"], ql, inp["wq"], inp["wk"], inp["wgate"], inp["k_gamma"], inp["k_beta"],
                inp["cos_sin"], inp["positions"], inp["slots"], inp["cache"])

        def call():
            q, w = fn(*args, key_only=not full, eps=EPS)
            o = {"cache": inp["cache"]}
            if full:
                o.update(q_fp8=q, weights=w)
            return o
        return call
    return build


def _sgl():
    un = core.sglang_import("sglang.srt.layers.quantization.unquant")
    fw = core.sglang_import("sglang.kernels.ops.attention.dsa.fp8_fused_writer_hip")
    di = core.sglang_import("sglang.srt.layers.attention.dsa.dsa_indexer")
    return un, fw, di


def _linear(un, w):
    layer = types.SimpleNamespace(weight=w)
    meth = un.UnquantizedLinearMethod()
    return lambda x: (meth.apply(layer, x), None)


def _mk_B_full(inp):
    un, fw, di = _sgl()
    wq_b, wk, wproj = _linear(un, inp["wq"]), _linear(un, inp["wk"]), _linear(un, inp["wgate"])
    cos = inp["cos_sin"][:, :32].contiguous()
    sin = inp["cos_sin"][:, 32:].contiguous()
    gamma, beta = inp["k_gamma"].float().contiguous(), inp["k_beta"].float().contiguous()
    kv_cache = inp["cache"].view(-1, PAGE, D + 4).view(FP8)
    x, q_lora, pos, loc = inp["x"], inp["q_latent"], inp["positions"], inp["slots"]
    wscale = D ** -0.5 * H ** -0.5

    def call():
        # == Indexer._aiter_fused_fp8_prepare_and_store (dsa_indexer.py:785-848)
        query, _ = wq_b(q_lora)
        query = query.view(-1, H, D)
        key, _ = wk(x)
        weights_raw, _ = wproj(x)          # _project_head_gates on HIP: bf16
        q_fp8 = torch.empty_like(query, dtype=FP8)
        weights = torch.empty_like(weights_raw, dtype=torch.float32)
        fw.aiter_fused_fp8_qk_write(query, q_fp8, weights_raw, weights, key, kv_cache, loc, gamma, beta,
                                    pos, cos, sin, EPS, 128, "ue8m0", wscale,
                                    preshuffle=di._use_aiter_preshuffle, is_neox=False, compute_all_q_rope=True)
        return {"q_fp8": q_fp8, "weights": weights, "cache": inp["cache"]}
    return call


def _rope_module(inp):
    re = core.sglang_import("sglang.srt.layers.rotary_embedding")
    rot = re.get_rope_wrapper(64, rotary_dim=64, max_position=inp["cos_sin"].shape[0], base=1000000,
                              is_neox_style=False, rope_scaling=None, dtype=torch.bfloat16, device="cuda")
    p = inp["cos_sin"].shape[0]
    # replace the generated tables by the drawn ones (same tables every arm uses)
    rot.cos_cache = inp["cos_sin"][:, :32].reshape(p, 1, 1, 32).contiguous()
    rot.sin_cache = inp["cos_sin"][:, 32:].reshape(p, 1, 1, 32).contiguous()
    if hasattr(rot, "cos_sin_cache"):
        rot.cos_sin_cache = inp["cos_sin"].contiguous()
    return rot


def _mk_B_key(rotate):
    def build(inp):
        un, fw, di = _sgl()
        ln = core.sglang_import("sglang.srt.layers.layernorm")
        from aiter.ops.cache import indexer_k_quant_and_cache
        k_norm = ln.LayerNorm(D, dtype=torch.bfloat16, eps=EPS).cuda()
        with torch.no_grad():
            k_norm.weight.copy_(inp["k_gamma"].to(torch.bfloat16))
            k_norm.bias.copy_(inp["k_beta"].to(torch.bfloat16))
        me = types.SimpleNamespace(
            wk=_linear(un, inp["wk"]), k_norm=k_norm, skip_rope=False, rope_head_dim=64, head_dim=D,
            rotary_emb=_rope_module(inp), apply_hadamard=True,
            _update_rope_guarded=di.Indexer._update_rope_guarded)
        kv_cache = inp["cache"].view(-1, PAGE, D + 4).view(FP8)
        x, pos, loc = inp["x"], inp["positions"], inp["slots"]

        def call():
            key = di.Indexer._get_k_bf16(me, x, pos, rotate=rotate)
            indexer_k_quant_and_cache(key, kv_cache, loc, 128, "ue8m0", preshuffle=di._use_aiter_preshuffle)
            return {"cache": inp["cache"]}
        return call
    return build


def _mk_X_fused(inp):
    _sgl()
    ld = core.sglang_import("sglang.kernels.ops.attention.dsa.hip_gfx950.loader")
    gemv, qk, _, _ = ld.modules()
    m = inp["x"].shape[0]
    if m > 48:
        raise core.Unavailable("gfx950 fused indexer: MAX_ROWS 48")
    w_kw = torch.cat([inp["wk"], inp["wgate"]], 0).contiguous()
    norm_w, norm_b = inp["k_gamma"].float().contiguous(), inp["k_beta"].float().contiguous()
    cos = inp["cos_sin"][:, :32].contiguous()
    sin = inp["cos_sin"][:, 32:].contiguous()
    kv_cache = inp["cache"].view(FP8).view(-1, PAGE, 132)
    q_proj = torch.empty(m, H * D, dtype=torch.bfloat16, device="cuda")
    kw = torch.empty(m, D + H, dtype=torch.bfloat16, device="cuda")
    q_fp8 = torch.empty(m, H, D, dtype=FP8, device="cuda")
    head_gate = torch.empty(m, H, dtype=torch.float32, device="cuda")
    wscale = float(H) ** -0.5 * D ** -0.5

    def call():
        gemv.dual_gemv_bf16(inp["q_latent"], inp["wq"], q_proj, inp["x"], w_kw, kw)
        k, weights = kw[:, :D], kw[:, D:]
        qk.indexer_qk_rope_hadamard_quant_and_cache(q_proj.view(m, H, D), q_fp8, weights, head_gate, k, kv_cache,
                                                    inp["slots"], norm_w, norm_b, inp["positions"], cos, sin,
                                                    EPS, wscale)
        return {"q_fp8": q_fp8, "weights": head_gate, "cache": inp["cache"]}
    return call


def arms(shape):
    if shape["kind"] == "full":
        return {"T": _mk_T(shape), "B": _mk_B_full, "X_fused": _mk_X_fused}
    return {"T": _mk_T(shape), "B": _mk_B_key(True), "B1": _mk_B_key(False)}


def rotate(shape):
    names = ["x", "wk", "k_gamma", "k_beta", "positions", "slots"]
    if shape["kind"] == "full":
        names += ["q_latent", "wq", "wgate"]
    return names


# ----------------------------------------------------------------------------- numerics
# Criterion (documented deviation from A5's code-point rule, which is unusable across implementations
# with different bf16 rounding points / GEMM split-K): dequantized fp8 value vs the reference value,
# relative to the row's |max| (e4m3 top-binade code spacing is 32/448 = 0.071 of amax).
# pass: max err/amax <= 0.071 (at most ~1 top-binade code); scale mismatches (2x) reported only.
# Code-point mismatch rates are reported as well.  weights: bf16 criterion (rtol 1e-2).
VAL_TOL = 32 / 448


def _valstat(codes, scale, ref_val, ref_codes, ref_scale):
    deq = codes.float() * scale.float().unsqueeze(-1)
    rv = ref_val.float().to(deq.device)
    amax = rv.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    e = ((deq - rv).abs() / amax).reshape(-1)
    d = core.cmp_fp8(codes, scale, ref_codes, ref_scale)
    d.update(err_amax_max=e.max().item(), err_amax_p999=torch.quantile(e[:1 << 24].float(), 0.999).item(),
             err_amax_mean=e.mean().item())
    # ue8m0 scales may legitimately differ by 2x where the row amax sits on a power-of-two boundary
    # under a different rounding chain; the value error (with each kernel's own scale) is what counts.
    d["pass_"] = bool(d["err_amax_max"] <= VAL_TOL)
    return d


def check(arm, outs, ref, inp, shape):
    hada = arm not in ("B", "B1") or (arm == "B" and shape["kind"] == "key")
    sfx = "" if hada else "_noH"
    st = {}
    slots = inp["slots"]
    valid = slots >= 0
    codes, scale, idx, sidx = _decode_cache(outs["cache"], slots)
    st["k_cache"] = _valstat(codes[valid], scale[valid], ref["k_val" + sfx][valid],
                             ref["k_codes" + sfx][valid], ref["k_scale" + sfx][valid])
    mask = torch.ones(outs["cache"].numel(), dtype=torch.bool, device=codes.device)
    mask[idx[valid].reshape(-1)] = False
    for i in range(4):
        mask[sidx[valid] + i] = False
    nd = int((outs["cache"].reshape(-1)[mask] != ref["cache_orig"].reshape(-1).to(codes.device)[mask]).sum().item())
    st["cache_untouched"] = dict(bytes_differ=nd, pass_=nd == 0)
    if shape["kind"] == "full":
        q = outs["q_fp8"].reshape(-1, H, D)
        # the q scale is folded into weights, not returned: dequantize with the reference ue8m0 scale
        # (a wrong kernel scale shows up as a 2x value error)
        st["q_fp8"] = _valstat(q, ref["q_scale" + sfx], ref["q_val" + sfx], ref["q_codes" + sfx], ref["q_scale" + sfx])
        w = outs["weights"].reshape(ref["weights" + sfx].shape)
        st["weights"] = core.cmp_close(w, ref["weights" + sfx], atol=0.0, rtol=1e-2, ulp=False)
    if not hada:
        st["note"] = dict(info="vs no-Hadamard reference (AITER writer path omits the Hadamard)", pass_=True)
    return st
