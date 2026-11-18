"""S5: Qwen3.5 attention prologue (Gemma q/k RMSNorm + interleaved mRoPE(11,11,10) on 64 of 256
dims + q|gate deinterleave + FP8 KV-cache write, page 1, unit scales).

Input: projected_qkv_gate [M, 4608] bf16 = 8 heads x (256 q | 256 gate) + 256 k + 256 v.

T  : M in {1,2,4,8}  -> qwen3_5/mxfp4 .../xl_decode_small_template.py  (labels m*)
     M in {8..64}    -> qwen3_5/fp8   .../faqk_template.py             (labels f*)
Stock SGLang 703e763d5f on ROCm (models/qwen3_5.py forward_prepare_fused_gate + aiter backend
_use_fused_fp8_kv_write) runs NO AITER kernel in this prologue:
  X  : sglang Triton _fused_qk_gemma_rmsnorm_gate_kernel (models/utils.py) ->
       MRotaryEmbedding.forward -> sglang Triton _triton_mrope_forward_fused ->
       sglang Triton reshape_and_cache_flash (kernels/ops/kvcache/cache_ops.py).   [context arm]
So the AITER arms are the plan's compositions (no "AITER as SGLang calls it" exists):
  B1 : torch cat (deinterleave q + k + v) + gate .contiguous() + AITER HIP
       fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(gemma_norm=True, rotary_dim=64,
       interleaved mRoPE, fp8 NHD cache write, CPU unit scales) -- called like SGLang's
       models/qwen3.py aiter_fused_qk_norm_rope_set_kv does for Qwen3-VL.  Same work as T.
  B2 : AITER Triton fused_qkv_split_qk_norm_rope_cache(attn_output_gate, interleaved, NHD,
       fp8 cache) with (1+w) folded into bf16 weights; plain 1-D RoPE -> only valid for the
       identical-rows (text-only decode) draw.  Same work as T.  Its RMSNorm applies (1+w)
       itself (measured), so raw Gemma weights are passed.
  B3 : AITER Triton fused_qk_norm_rope_gate_fp8_quant (1-D positions, Gemma in-kernel) +
       sglang reshape_and_cache_flash KV write.  EXTRA work (per-sequence fp8 Q/K/V quant for
       FMHA); identical-rows only.
"""
import torch

from xbench import core

W, H, D = 4608, 8, 256
EPS = 1e-6
SECT = (11, 11, 10)
TRIG = 262144
POSBUF = 512
SMALL_SLOTS = 32768
BIG_SLOTS = (1 << 23) + (1 << 20)   # 9,437,184 slots x 256 B = 2.42 GB per cache (> 2^31 B)

XL = ("qwen3_5/mxfp4/fused_attention_qk_norm_rope_kv_cache_xl/xl_decode_small_template.py",
      "qwen3_5_mxfp4_tp4_gfx950/fused_attention_qk_norm_rope_kv_cache_templated.json",
      "mxfp4/fused_attention_qk_norm_rope_kv_cache_xl/xl_decode_small_template.py")
FAQK = ("qwen3_5/fp8/fused_attention_qk_norm_rope_kv_cache_templated/faqk_template.py",
        "qwen3_5_fp8_tp4_gfx950/fused_attention_qk_norm_rope_kv_cache_templated.json",
        "fused_attention_qk_norm_rope_kv_cache_templated/faqk_template.py")

SHAPES = (
    [dict(label=f"m{m}", m=m, tpl="xl", draw="id", pool="small") for m in (1, 2, 4, 8)]
    + [dict(label=f"f{m}", m=m, tpl="faqk", draw="id", pool="small") for m in (8, 16, 32, 64)]
    + [dict(label=f"m{m}_dr", m=m, tpl="xl", draw="dr", pool="small", caches=["hot"]) for m in (1, 8)]
    + [dict(label="f64_dr", m=64, tpl="faqk", draw="dr", pool="small", caches=["hot"])]
    + [dict(label=f"m{m}_big", m=m, tpl="xl", draw="id", pool="big", caches=["hot"]) for m in (1, 8)]
    + [dict(label="f64_big", m=64, tpl="faqk", draw="id", pool="big", caches=["hot"])]
)
MODE = "graph"
SEEDS = (0, 100)
AA = ["T", "B1"]
# Decode M weights: no recorded per-concurrency decode-M distribution for Qwen3.5 here;
# uniform over the served id-draw rows (stated in the notes).


# ----------------------------------------------------------------------------- inputs
def _axis_of_lane():
    ax = []
    for f in range(32):
        if f % 3 == 1 and f < 3 * SECT[1]:
            ax.append(1)
        elif f % 3 == 2 and f < 3 * SECT[2]:
            ax.append(2)
        else:
            ax.append(0)
    return torch.tensor(ax)


def _pattern_rows(rows, salt):
    """Deterministic fp8 bytes for big-pool rows (no RNG state): byte = (r*131 + d*7 + salt) % 251."""
    r = rows.long()[:, None, None]
    d = torch.arange(D, device=rows.device)[None, None, :]
    return ((r * 131 + d * 7 + salt) % 251).to(torch.uint8).view(torch.float8_e4m3fn)


_BIG = {}


def _big_pool(dev="cuda"):
    if "kc" not in _BIG:
        kc = torch.empty((BIG_SLOTS, 1, D), dtype=torch.float8_e4m3fn, device=dev)
        vc = torch.empty_like(kc)
        step = 1 << 20
        for s in range(0, BIG_SLOTS, step):
            rows = torch.arange(s, min(s + step, BIG_SLOTS), device=dev)
            kc[s:s + len(rows)] = _pattern_rows(rows, 11)
            vc[s:s + len(rows)] = _pattern_rows(rows, 29)
        _BIG["kc"], _BIG["vc"] = kc, vc
    return _BIG["kc"], _BIG["vc"]


def gen(shape, seed):
    m = shape["m"]
    g = torch.Generator(device="cuda").manual_seed(seed * 7919 + m * 31 + (0 if shape["draw"] == "id" else 1)
                                                   + (0 if shape["pool"] == "small" else 2))

    def randn(*s):
        return torch.randn(*s, generator=g, device="cuda", dtype=torch.float32)

    packed = randn(m, W)
    if seed >= 100:  # wide magnitudes 2^-8..2^8 (spec adversarial draw)
        packed = packed * torch.exp2(torch.randint(-8, 9, (m, W), generator=g, device="cuda").float())
    packed = packed.to(torch.bfloat16)
    qw = (randn(D) * 0.2).to(torch.bfloat16)
    kw = (randn(D) * 0.2).to(torch.bfloat16)
    pos_buf = torch.randint(0, TRIG, (3, POSBUF), generator=g, device="cuda", dtype=torch.int64)
    if shape["draw"] == "id":
        pos_buf[1] = pos_buf[0]
        pos_buf[2] = pos_buf[0]
    ang = randn(TRIG, 32) * 3.0
    cos_sin = torch.cat([ang.cos(), ang.sin()], 1).to(torch.bfloat16).contiguous()
    if shape["pool"] == "small":
        slots = torch.randperm(SMALL_SLOTS, generator=g, device="cuda")[:m].to(torch.int64)
        kc0 = (randn(SMALL_SLOTS, 1, D) * 0.5).to(torch.float8_e4m3fn)
        vc0 = (randn(SMALL_SLOTS, 1, D) * 0.5).to(torch.float8_e4m3fn)
    else:  # top of the big pool: slot * 256 B >= 2^31
        slots = (BIG_SLOTS - 1 - torch.randperm(1 << 16, generator=g, device="cuda")[:m]).to(torch.int64)
    inp = dict(packed=packed.contiguous(), qw=qw, kw=kw, pos_buf=pos_buf, cos_sin=cos_sin, slots=slots)
    if shape["pool"] == "small":
        inp.update(kc=kc0, vc=vc0)

    # ---- fp32 reference (bf16 boundary after the norm, as the serving contract)
    pos = pos_buf[:, :m]
    qg = packed[:, :4096].view(m, H, 512)
    q, gate = qg[..., :D], qg[..., D:]
    k = packed[:, 4096:4352].view(m, 1, D)
    v = packed[:, 4352:].view(m, 1, D)
    axis = _axis_of_lane().to(pos.device)
    lane_pos = pos[axis, :].t()                     # [m, 32]
    cs = cos_sin.float()
    cos = cs[lane_pos, torch.arange(32, device=pos.device)[None, :]]       # [m, 32]
    sin = cs[lane_pos, 32 + torch.arange(32, device=pos.device)[None, :]]

    def nr(x, w):
        x = x.float()
        n = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * (1.0 + w.float())).to(torch.bfloat16).float()
        a, b = n[..., :32], n[..., 32:64]
        c, s = cos[:, None, :], sin[:, None, :]
        return torch.cat([a * c - b * s, b * c + a * s, n[..., 64:]], -1)

    qr = nr(q, qw)
    kr = nr(k, kw)
    ref = dict(q=qr, gate=gate.reshape(m, H * D).contiguous(),
               k_codes=kr.to(torch.bfloat16).float().clamp(-448, 448).to(torch.float8_e4m3fn),
               v_codes=v.float().clamp(-448, 448).to(torch.float8_e4m3fn))   # satfinite (all kernels)
    if shape["pool"] == "small":
        kcr, vcr = kc0.clone(), vc0.clone()
        kcr[slots] = ref["k_codes"]
        vcr[slots] = ref["v_codes"]
        ref.update(kc=kcr, vc=vcr)
    return inp, ref


def rotate(shape):
    names = ["packed", "qw", "kw", "pos_buf", "slots"]
    if shape["pool"] == "small":
        names += ["kc", "vc"]
    return names


def _caches(inp, shape):
    if shape["pool"] == "small":
        return inp["kc"], inp["vc"]
    return _big_pool()


def _ones():
    return torch.ones(1, dtype=torch.float32, device="cuda")


# ----------------------------------------------------------------------------- arms
def _T(shape):
    from xbench import tload
    rel, prof, src = XL if shape["tpl"] == "xl" else FAQK

    def build(inp):
        fn = tload.load(rel, "fused_attention_qk_norm_rope_kv_cache", prof, src)
        m = shape["m"]
        kc, vc = _caches(inp, shape)
        ks, vs = _ones(), _ones()
        pos = inp["pos_buf"][:, :m]

        def call():
            q, gate, k, v, kc_, vc_ = fn(inp["packed"], inp["qw"], inp["kw"], pos, inp["cos_sin"],
                                         inp["slots"], kc, vc, eps=EPS, rotary_dim=64, k_scale=ks,
                                         v_scale=vs, mrope_section=SECT)
            return dict(q=q, gate=gate, kc=kc_, vc=vc_)
        return call
    return build


def _B1(shape):
    def build(inp):
        import aiter
        op = aiter.fused_qk_norm_mrope_3d_cache_pts_quant_shuffle
        m = shape["m"]
        kc, vc = _caches(inp, shape)
        ks = torch.ones(1, dtype=torch.float32)   # CPU: the HIP op reads scales on the host
        vs = torch.ones(1, dtype=torch.float32)
        packed = inp["packed"]
        pos = inp["pos_buf"][:, :m]
        qg = packed[:, :4096].view(m, H, 512)
        kv = packed[:, 4096:].view(m, 2, D)

        def call():
            qkv = torch.cat((qg[..., :D], kv), dim=1)          # deinterleave copy (counted)
            gate = qg[..., D:].reshape(m, H * D)               # gate copy (counted)
            q_out = torch.empty((m, H, D), dtype=torch.bfloat16, device=packed.device)
            op(qkv=qkv, qw=inp["qw"], kw=inp["kw"], cos_sin=inp["cos_sin"], positions=pos,
               num_tokens=m, num_heads_q=H, num_heads_k=1, num_heads_v=1, head_size=D,
               is_neox_style=True, mrope_section_=list(SECT), is_interleaved=True, eps=EPS,
               q_out=q_out, k_cache=kc.unsqueeze(1), v_cache=vc.unsqueeze(1),
               slot_mapping=inp["slots"], per_tensor_k_scale=ks, per_tensor_v_scale=vs,
               k_out=None, v_out=None, return_kv=False, use_shuffle_layout=False,
               block_size=0, x=0, rotary_dim=64, gemma_norm=True)
            return dict(q=q_out, gate=gate, kc=kc, vc=vc)
        return call
    return build


def _B2(shape):
    def build(inp):
        if shape["draw"] != "id":
            raise core.Unavailable("1-D RoPE only: invalid for distinct mRoPE rows")
        from aiter.ops.triton.rope.fused_qkv_split_qk_norm_rope_cache import (
            fused_qkv_split_qk_norm_rope_cache)
        m = shape["m"]
        kc, vc = _caches(inp, shape)
        ks, vs = _ones(), _ones()
        # Measured: this kernel's RMSNorm already applies (1 + w) (Gemma form), so the raw
        # zero-centred weights are passed (plan said fold 1+w; that double-adds 1).
        qw1, kw1 = inp["qw"], inp["kw"]
        cos, sin = inp["cos_sin"][:, :32], inp["cos_sin"][:, 32:]
        pos1 = inp["pos_buf"][0, :m]
        kc4, vc4 = kc.view(-1, 1, 1, D), vc.view(-1, 1, 1, D)

        def call():
            q, gate, k, v = fused_qkv_split_qk_norm_rope_cache(
                inp["packed"], qw1, kw1, cos, sin, pos1, kc4, vc4, inp["slots"], qh=H, kvh=1,
                head_dim=D, is_neox=True, reuse_freqs_front_part=True, attn_output_gate=True,
                k_scale=ks, v_scale=vs, eps=EPS, gated_qkv_layout="interleaved",
                kv_cache_layout="NHD")
            return dict(q=q, gate=gate.reshape(m, H * D), kc=kc, vc=vc)
        return call
    return build


def _sgl_kv():
    return core.sglang_import("sglang.kernels.ops.kvcache.cache_ops").launch_reshape_and_cache_flash


def _B3(shape):
    def build(inp):
        if shape["draw"] != "id":
            raise core.Unavailable("1-D positions only: invalid for distinct mRoPE rows")
        from aiter.ops.triton.rope.fused_qk_norm_rope_gate_fp8_quant import (
            fused_qk_norm_rope_gate_fp8_quant)
        kvw = _sgl_kv()
        m = shape["m"]
        kc, vc = _caches(inp, shape)
        ks, vs = _ones(), _ones()
        packed = inp["packed"]
        pos1 = inp["pos_buf"][0, :m]
        cu = torch.arange(m + 1, dtype=torch.int32, device=packed.device)

        def call():
            r = fused_qk_norm_rope_gate_fp8_quant(
                packed[:, :4096], packed[:, 4096:4352], packed[:, 4352:], inp["qw"], inp["kw"],
                inp["cos_sin"], pos1, cu, num_actual_tokens=m, num_query_heads=H, num_kv_heads=1,
                head_dim=D, rotary_dim=64, eps=EPS)
            kvw(r.key.view(m, 1, D), packed[:, 4352:].view(m, 1, D), kc.view(-1, 1, 1, D),
                vc.view(-1, 1, 1, D), inp["slots"], k_scale=ks, v_scale=vs)
            return dict(q=r.query.view(m, H, D), gate=r.gate.reshape(m, H * D), kc=kc, vc=vc)
        return call
    return build


_ROPE = {}


def _X(shape):
    def build(inp):
        mu = core.sglang_import("sglang.srt.models.utils")
        mr = core.sglang_import("sglang.srt.layers.rotary_embedding.mrope")
        rc = core.sglang_import("sglang.srt.runtime_context")
        kvw = _sgl_kv()
        m = shape["m"]
        kc, vc = _caches(inp, shape)
        ks, vs = _ones(), _ones()
        # keyed by data_ptr with the tensor kept alive (an id()-key was reused across freed
        # inputs and silently paired a stale rope table with new inputs).
        key = inp["cos_sin"].data_ptr()
        if key not in _ROPE:
            with rc.get_context().override_server_args():
                rope = mr.MRotaryEmbedding(D, 64, TRIG, 10000000, True, torch.bfloat16,
                                           mrope_section=list(SECT), mrope_interleaved=True)
            core.restore_triton()
            rope = rope.cuda()
            # HIP keeps the table fp32 (mrope.py _match_cos_sin_cache_dtype); same values.
            rope.cos_sin_cache = inp["cos_sin"].float()
            _ROPE[key] = (inp["cos_sin"], rope)
        rope = _ROPE[key][1]
        packed = inp["packed"]
        pos = inp["pos_buf"][:, :m]

        def call():
            q_gate, k, v = packed.split([4096, 256, 256], dim=-1)
            q, k2, gate = mu.fused_qk_gemma_rmsnorm_with_gate(q_gate, k, inp["qw"], inp["kw"], EPS, D, H)
            q = q.view(m, -1)
            k2 = k2.view(m, -1)
            q, k2 = rope(pos, q, k2)
            kvw(k2.view(m, 1, D), v.view(m, 1, D), kc.view(-1, 1, 1, D), vc.view(-1, 1, 1, D),
                inp["slots"], k_scale=ks, v_scale=vs)
            return dict(q=q.view(m, H, D), gate=gate.view(m, H * D), kc=kc, vc=vc)
        return call
    return build


def arms(shape):
    return {"T": _T(shape), "B1": _B1(shape), "B2": _B2(shape), "B3": _B3(shape), "X": _X(shape)}


# ----------------------------------------------------------------------------- numerics
def check(arm, outs, ref, inp, shape):
    st = {"q": core.cmp_close(outs["q"], ref["q"])}
    st["gate"] = core.cmp_bytes(outs["gate"].contiguous(), ref["gate"])
    slots = inp["slots"]
    kc, vc = outs["kc"], outs["vc"]
    one = torch.ones(1, device=kc.device)
    st["k_cache_slots"] = core.cmp_fp8(kc[slots], one, ref["k_codes"], one)
    st["v_cache_slots"] = core.cmp_bytes(vc[slots], ref["v_codes"])
    if shape["pool"] == "small":
        untouched = torch.ones(kc.shape[0], dtype=torch.bool, device=kc.device)
        untouched[slots] = False
        st["untouched"] = core.cmp_bytes(
            torch.cat([kc[untouched], vc[untouched]]), torch.cat([ref["kc"][untouched], ref["vc"][untouched]]))
    else:
        nb = (slots - 1).clamp_min(0)
        nb = nb[~torch.isin(nb, slots)]
        st["untouched"] = core.cmp_bytes(torch.cat([kc[nb], vc[nb]]),
                                         torch.cat([_pattern_rows(nb, 11), _pattern_rows(nb, 29)]))
    return st
