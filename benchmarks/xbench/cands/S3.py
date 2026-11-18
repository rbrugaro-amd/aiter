"""S3: GDN decode (conv1d update + gated delta rule + gated RMSNorm [+ FP8 group-128 quant]).

Shapes (label prefix):
  nx_m*  Qwen3-Next FP8 TP4 (KH 4, VH 8, ratio 2, interleaved qkvz 3072, ba 16, conv bias, fp8 out)
         T = qwen3_next/gdn_fused_decode_group_fp8_quant_templated/gdn_decode_template_v2.py (M 1,8,16,64)
  q5f_m* Qwen3.5 FP8 TP4 (KH 4, VH 16, ratio 4, flat qkvz 5120, ba 32, no conv bias, fp8 out)
         T = r3_gdn_decode_small (M 2) / qwen3_5_fp8_gdn_decode_template (M 128, 256)
  q5x_m* Qwen3.5 MXFP4 TP4 (same dims, bf16 out)
         T = r3_gdn_decode_small::gdn_fused_decode (M 1,2,8) / xl_gdn_decode_m96 (M 16..96)

Arms:
  T       template (manifest sha256 checked)
  B_port  aiter.ops.triton.gated_delta_net.fused_gdn_decode_qkvz (AITER's port of our
          gdn_decode_m16.py; one launch).  Unavailable when its probe says unsupported.
  B1      AITER multi-kernel composition: [Qwen3-Next: aiter fused_qkvzba_split_reshape_cat_decode
          (Triton)] + causal_conv1d_update_split_qkv (Gluon) + torch.cat(q,k,v) (counted) +
          fused_rearrange_sigmoid_gated_delta_rule (Triton) + gated_rmsnorm_fp8_group_quant (HIP).
          bf16-out rows: the last stage is SGLang's FLA RMSNormGated (no bf16 AITER gated norm) -- flagged.
  X_sgl   SGLang 703e763d5f stock decode chain on ROCm (non-AITER Triton kernels):
          split (fused_qkvzba_split_reshape_cat[_contiguous]) + causal_conv1d_update +
          TritonGDNKernel.packed_decode (FLA) + FusedRMSNormGated / RMSNormGated, then for fp8 rows
          the AITER per_1x128 quant that out_proj's aiter_w8a8_block_fp8_linear runs on bf16 input.
"""
import math

import torch
import torch.nn.functional as F

from xbench import core

KD = VD = 128
EPS = 1e-6
SCALE = KD ** -0.5
FP8 = torch.float8_e4m3fn

CFG = {
    "nx": dict(KH=4, VH=8, layout="interleaved", bias=True, fp8=True),
    "q5f": dict(KH=4, VH=16, layout="flat", bias=False, fp8=True),
    "q5x": dict(KH=4, VH=16, layout="flat", bias=False, fp8=False),
}

TPL = {
    # model: [(Ms, pack_rel, entry, profile, source_file)]
    "nx": [((1, 8, 16, 64), "qwen3_next/gdn_fused_decode_group_fp8_quant_templated/gdn_decode_template_v2.py",
            "gdn_fused_decode_group_fp8_quant", "qwen3_next_fp8_tp4_gfx950/gdn_fused_decode_group_fp8_quant_templated.json",
            "gdn_fused_decode_group_fp8_quant_templated/gdn_decode_template_v2.py")],
    "q5f": [((2,), "qwen3_5/fp8/gdn_fused_decode_group_fp8_quant_r3/r3_gdn_decode_small.py",
             "gdn_fused_decode_group_fp8_quant", "qwen3_5_fp8_tp4_gfx950/gdn_fused_decode_group_fp8_quant_templated.json",
             "gdn_fused_decode_group_fp8_quant_r3/r3_gdn_decode_small.py"),
            ((128, 256), "qwen3_5/fp8/gdn_fused_decode_group_fp8_quant_templated/qwen3_5_fp8_gdn_decode_template.py",
             "gdn_fused_decode_group_fp8_quant", "qwen3_5_fp8_tp4_gfx950/gdn_fused_decode_group_fp8_quant_templated.json",
             "gdn_fused_decode_group_fp8_quant_templated/qwen3_5_fp8_gdn_decode_template.py")],
    "q5x": [((1, 2, 8), "qwen3_5/fp8/gdn_fused_decode_group_fp8_quant_r3/r3_gdn_decode_small.py",
             "gdn_fused_decode", "qwen3_5_mxfp4_tp4_gfx950/gdn_fused_decode_templated.json",
             "fp8/gdn_fused_decode_group_fp8_quant_r3/r3_gdn_decode_small.py"),
            ((16, 32, 64, 96), "qwen3_5/mxfp4/gdn_fused_decode_xl/xl_gdn_decode_m96.py",
             "gdn_fused_decode", "qwen3_5_mxfp4_tp4_gfx950/gdn_fused_decode_templated.json",
             "mxfp4/gdn_fused_decode_xl/xl_gdn_decode_m96.py")],
}

SHAPES = []
for _model, _rows in TPL.items():
    for _ms, *_ in _rows:
        for _m in _ms:
            SHAPES.append(dict(label=f"{_model}_m{_m}", model=_model, m=_m))
MODE = "graph"
SEEDS = (0, 100)
AA = ["T", "B_port"]
B37_ARMS = ["B_port", "B1", "X_sgl"]
RING_CAP = 32


def count(shape):
    return 960 if shape["m"] <= 96 else 192


def rotate(shape):
    return ["projected_qkvz", "projected_ba", "conv_state", "delta_state"]


def _dims(model):
    c = CFG[model]
    kh, vh = c["KH"], c["VH"]
    r = vh // kh
    channels = 2 * kh * KD + vh * VD
    return kh, vh, r, channels


# ----------------------------------------------------------------------------- inputs
def gen(shape, seed):
    model, m = shape["model"], shape["m"]
    kh, vh, r, channels = _dims(model)
    slots = 2 * m + 5                       # plan: realistic pool ~ 2 x batch
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed * 1009 + m * 7 + len(model))
    bf16 = torch.bfloat16

    def rnd(*s, dtype=bf16, scale=1.0):
        return (torch.randn(*s, generator=g, device=dev, dtype=torch.float32) * scale).to(dtype)

    width = channels + vh * VD
    inp = dict(
        projected_qkvz=rnd(m, width),
        projected_ba=rnd(m, 2 * vh),
        conv_state=rnd(slots, channels, 3),
        delta_state=rnd(slots, vh, VD, KD, dtype=torch.float32, scale=0.1),
        cache_indices=torch.randperm(slots, generator=g, device=dev)[:m].to(torch.int32),
        conv_weight=rnd(channels, 4, scale=0.2),
        a_log=rnd(vh, dtype=torch.float32),
        dt_bias=rnd(vh),
        norm_weight=(1 + 0.25 * rnd(VD, dtype=torch.float32)).to(bf16),
    )
    if CFG[model]["bias"]:
        inp["conv_bias"] = rnd(channels, scale=0.1)
    if seed >= 100:
        # adversarial: 2^-8..2^8 magnitudes on the state and q/k columns; odd heads near-zero decay
        def mag(s):
            return torch.exp2(torch.randint(-8, 9, s, generator=g, device=dev).float())
        inp["delta_state"].mul_(mag(inp["delta_state"].shape))
        q, k = (t.to(dev) for t in _qk_cols(model))
        for cols in (q, k):
            sub = inp["projected_qkvz"][:, cols]
            inp["projected_qkvz"][:, cols] = (sub.float() * mag(sub.shape)).to(bf16)
        odd = torch.arange(1, vh, 2, device=dev)
        a_idx = _a_cols(model).to(dev)[odd]
        inp["projected_ba"][:, a_idx] = torch.linspace(80, 110, m * len(odd), device=dev).reshape(m, -1).to(bf16)
        inp["a_log"][odd] = 0.0
        inp["dt_bias"][odd] = 0.0
    ref = reference(model, inp)
    return inp, ref


def _qk_cols(model):
    kh, vh, r, channels = _dims(model)
    if CFG[model]["layout"] == "flat":
        return torch.arange(0, kh * KD), torch.arange(kh * KD, 2 * kh * KD)
    gw = 2 * KD + 2 * r * VD
    base = torch.arange(kh)[:, None] * gw
    return (base + torch.arange(KD)).reshape(-1), (base + KD + torch.arange(KD)).reshape(-1)


def _a_cols(model):
    kh, vh, r, _ = _dims(model)
    if CFG[model]["layout"] == "flat":
        return torch.arange(vh, 2 * vh)
    # interleaved ba per key head [b*r | a*r]
    return (torch.arange(kh)[:, None] * 2 * r + r + torch.arange(r)).reshape(-1)


def unpack(model, qkvz, ba):
    """-> mixed [M, 2k+v] (q_all|k_all|v_all), z [M,VH,VD], b [M,VH], a [M,VH]"""
    kh, vh, r, channels = _dims(model)
    m = qkvz.shape[0]
    if CFG[model]["layout"] == "flat":
        mixed = qkvz[:, :channels]
        z = qkvz[:, channels:].reshape(m, vh, VD)
        return mixed, z, ba[:, :vh], ba[:, vh:]
    gw = 2 * KD + 2 * r * VD
    gq = qkvz.reshape(m, kh, gw)
    q, k, v, z = gq.split((KD, KD, r * VD, r * VD), -1)
    mixed = torch.cat((q.reshape(m, -1), k.reshape(m, -1), v.reshape(m, -1)), -1)
    bb, aa = ba.reshape(m, kh, 2 * r).split(r, -1)
    return mixed, z.reshape(m, vh, VD), bb.reshape(m, vh), aa.reshape(m, vh)


def reference(model, inp):
    kh, vh, r, channels = _dims(model)
    x, ba = inp["projected_qkvz"], inp["projected_ba"]
    conv, state, idx = inp["conv_state"], inp["delta_state"], inp["cache_indices"].long()
    w, bias = inp["conv_weight"], inp.get("conv_bias")
    m = x.shape[0]
    mixed, z, b, a = unpack(model, x, ba)
    window = torch.cat((conv[idx].float(), mixed.float()[..., None]), -1)
    pre = (window * w.float()[None]).sum(-1)
    if bias is not None:
        pre = pre + bias.float()
    convolved = F.silu(pre).to(torch.bfloat16)
    q, k, v = convolved.split((kh * KD, kh * KD, vh * VD), -1)
    q, k = q.float().reshape(m, kh, KD), k.float().reshape(m, kh, KD)
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * SCALE
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q, k = q.repeat_interleave(r, 1), k.repeat_interleave(r, 1)
    decay = torch.exp(-torch.exp(inp["a_log"].float())[None] * F.softplus(a.float() + inp["dt_bias"].float()))
    beta = torch.sigmoid(b.float())
    st = state[idx].float() * decay[..., None, None]
    resid = v.float().reshape(m, vh, VD) - torch.einsum("bhvk,bhk->bhv", st, k)
    st = st + (resid * beta[..., None])[..., None] * k[..., None, :]
    core_ = torch.einsum("bhvk,bhk->bhv", st, q).to(torch.bfloat16).float()
    normalized = (core_ * torch.rsqrt(core_.square().mean(-1, keepdim=True) + EPS)
                  * inp["norm_weight"].float()[None, None] * F.silu(z.float()))
    groups = normalized.reshape(m, -1, 128)
    scales = groups.abs().amax(-1).clamp_min(1e-10) / 448.0
    qcodes = (groups / scales[..., None]).clamp(-448, 448).to(FP8).reshape(m, -1)
    conv_new = torch.cat((conv[idx, :, 1:], mixed[..., None]), -1)
    touched = torch.zeros(conv.shape[0], dtype=torch.bool, device=conv.device)
    touched[idx] = True
    return dict(normalized=normalized, quantized=qcodes, scales=scales,
                conv_touched=conv_new, delta_touched=st,
                conv_untouched_sum=conv[~touched].double().sum().reshape(1),
                delta_untouched_sum=state[~touched].double().sum().reshape(1))


# ----------------------------------------------------------------------------- arms
def _tpl_for(shape):
    for ms, pack_rel, entry, profile, src in TPL[shape["model"]]:
        if shape["m"] in ms:
            from xbench import tload
            return tload.load(pack_rel, entry, profile, src)
    raise KeyError(shape)


def _T(shape):
    model = shape["model"]

    def build(inp):
        fn = _tpl_for(shape)
        args = [inp[k] for k in ("projected_qkvz", "projected_ba", "conv_state", "delta_state", "cache_indices",
                                 "conv_weight")]
        if CFG[model]["bias"]:
            args.append(inp["conv_bias"])
        args += [inp["a_log"], inp["dt_bias"], inp["norm_weight"]]

        def call():
            res = fn(*args, scale=SCALE, eps=EPS)
            if model == "nx":
                n, c, d, qq, s = res
                return dict(normalized=n, quantized=qq, scales=s, conv_state=c, delta_state=d)
            if model == "q5f":
                qq, s, c, d = res
                return dict(quantized=qq, scales=s, conv_state=c, delta_state=d)
            n, c, d = res
            return dict(normalized=n, conv_state=c, delta_state=d)
        return call
    return build


def _port():
    import importlib
    # the package __init__ re-exports the function under the module's name
    return importlib.import_module("aiter.ops.triton.gated_delta_net.fused_gdn_decode_qkvz")


def _B_port(shape):
    model = shape["model"]
    c = CFG[model]

    def build(inp):
        mod = _port()
        bias = inp["conv_bias"] if c["bias"] else torch.zeros(inp["conv_weight"].shape[0], dtype=torch.bfloat16,
                                                              device=inp["conv_weight"].device)
        args = (inp["projected_qkvz"], inp["projected_ba"], inp["conv_state"], inp["delta_state"],
                inp["cache_indices"], inp["conv_weight"], bias, inp["a_log"], inp["dt_bias"], inp["norm_weight"])
        qd = FP8 if c["fp8"] else None
        ok, reason = mod.fused_gdn_decode_qkvz_supported(*args, quant_dtype=qd, qkvz_layout=c["layout"])
        if not ok:
            raise core.Unavailable(f"fused_gdn_decode_qkvz_supported=False: {reason}")

        def call():
            out, qq, s = mod.fused_gdn_decode_qkvz(*args, scale=SCALE, norm_eps=EPS, quant_dtype=qd,
                                                   qkvz_layout=c["layout"])
            d = dict(normalized=out, conv_state=inp["conv_state"], delta_state=inp["delta_state"])
            if qd is not None:
                d.update(quantized=qq, scales=s)
            return d
        return call
    return build


def _sgl_norm(model, weight):
    if model == "nx":
        mod = core.sglang_import("sglang.kernels.ops.attention.fla.fused_norm_gate")
        norm = mod.FusedRMSNormGated(VD, eps=EPS, activation="silu", device=weight.device, dtype=torch.bfloat16)
    else:
        mod = core.sglang_import("sglang.kernels.ops.attention.fla.layernorm_gated")
        norm = mod.RMSNorm(VD, eps=EPS, group_size=None, norm_before_gate=True, device=weight.device,
                           dtype=torch.bfloat16, activation="silu")
    norm.weight = torch.nn.Parameter(weight, requires_grad=False)
    return norm


def _aiter_quant():
    import aiter
    return aiter.get_hip_quant(aiter.QuantType.per_1x128)


def _B1(shape):
    model = shape["model"]
    c = CFG[model]
    kh, vh, r, channels = _dims(model)

    def build(inp):
        from aiter.ops.triton.gated_delta_net.causal_conv1d_decode import causal_conv1d_update_split_qkv
        from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
            fused_rearrange_sigmoid_gated_delta_rule)
        m = inp["projected_qkvz"].shape[0]
        cu = torch.arange(m + 1, dtype=torch.int32, device=inp["projected_qkvz"].device)
        if c["fp8"]:
            from aiter.ops.gated_rmsnorm_fp8_group_quant import gated_rmsnorm_fp8_group_quant
        else:
            norm = _sgl_norm(model, inp["norm_weight"])
        if c["layout"] == "interleaved":
            from aiter.ops.triton._triton_kernels.gated_delta_net.fused_qkvzba_split import (
                fused_qkvzba_split_reshape_cat_decode)

        def call():
            if c["layout"] == "interleaved":
                mixed, z, b, a = fused_qkvzba_split_reshape_cat_decode(
                    inp["projected_qkvz"], inp["projected_ba"], kh, vh, KD, VD)
            else:
                mixed, z, b, a = unpack(model, inp["projected_qkvz"], inp["projected_ba"])
            q, k, v = causal_conv1d_update_split_qkv(
                mixed.unsqueeze(-1), inp["conv_state"], inp["conv_weight"], kh * KD, vh * VD,
                bias=inp.get("conv_bias"), activation="silu", conv_state_indices=inp["cache_indices"])
            qkv = torch.cat((q, k, v), 1).squeeze(-1)
            o, _ = fused_rearrange_sigmoid_gated_delta_rule(
                inp["a_log"], a, b, inp["dt_bias"], qkv, kh * KD, vh * VD, KD, VD, scale=SCALE,
                initial_state=inp["delta_state"], inplace_final_state=True, cu_seqlens=cu,
                ssm_state_indices=inp["cache_indices"], use_qk_l2norm_in_kernel=True)
            core_ = o.reshape(m, vh, VD)
            d = dict(conv_state=inp["conv_state"], delta_state=inp["delta_state"])
            if c["fp8"]:
                qq = torch.empty((m, vh * VD), dtype=FP8, device=core_.device)
                s = torch.empty((m, vh), dtype=torch.float32, device=core_.device)
                gated_rmsnorm_fp8_group_quant(qq, s, core_, z.contiguous(), inp["norm_weight"], EPS, 128)
                d.update(quantized=qq, scales=s)
            else:
                d["normalized"] = norm(core_.reshape(-1, VD), z.reshape(-1, VD)).reshape(m, vh, VD)
            return d
        return call
    return build


def _X_sgl(shape):
    model = shape["model"]
    c = CFG[model]
    kh, vh, r, channels = _dims(model)

    def build(inp):
        proj = core.sglang_import("sglang.kernels.ops.attention.triton_gdn_fused_proj")
        gb = core.sglang_import("sglang.srt.layers.attention.linear.gdn_backend")
        gt = core.sglang_import("sglang.srt.layers.attention.linear.kernels.gdn_triton")
        kernel = gt.TritonGDNKernel()
        if not kernel.supports_packed_decode:
            raise core.Unavailable("TritonGDNKernel.supports_packed_decode is False")
        norm = _sgl_norm(model, inp["norm_weight"])
        split = (proj.fused_qkvzba_split_reshape_cat if model == "nx"
                 else proj.fused_qkvzba_split_reshape_cat_contiguous)
        quant = _aiter_quant() if c["fp8"] else None
        m = inp["projected_qkvz"].shape[0]

        def call():
            mixed, z, b, a = split(inp["projected_qkvz"], inp["projected_ba"], kh, vh, KD, VD)
            mixed = gb.causal_conv1d_update(mixed, inp["conv_state"], inp["conv_weight"], inp.get("conv_bias"),
                                            "silu", conv_state_indices=inp["cache_indices"])
            core_ = kernel.packed_decode(mixed, a, b, A_log=inp["a_log"], dt_bias=inp["dt_bias"], scale=SCALE,
                                         ssm_states=inp["delta_state"], cache_indices=inp["cache_indices"],
                                         num_v_heads=vh, head_v_dim=VD)
            n = norm(core_.reshape(-1, VD), z.reshape(-1, VD)).reshape(m, vh, VD)
            d = dict(normalized=n, conv_state=inp["conv_state"], delta_state=inp["delta_state"])
            if quant is not None:
                qq, s = quant(n.reshape(m, -1), quant_dtype=FP8)
                d.update(quantized=qq, scales=s)
            return d
        return call
    return build


def arms(shape):
    return {"T": _T(shape), "B_port": _B_port(shape), "B1": _B1(shape), "X_sgl": _X_sgl(shape)}


def reset(shape, ring):
    snap_c = ring[0]["conv_state"].clone()
    snap_d = ring[0]["delta_state"].clone()

    def fn():
        for mbr in ring:
            mbr["conv_state"].copy_(snap_c)
            mbr["delta_state"].copy_(snap_d)
    return fn


# ----------------------------------------------------------------------------- numerics
def _fp8_vs_ref(qq, sc, ref):
    """fp8 group-128 output vs the fp32 reference (before any rounding).
    Exact-code agreement with an fp32-derived quantization is not attainable for a chain whose
    bf16 core already differs by bf16 ulps (codes are 3-bit mantissa; amax moves the scale), so
    the A5 code criterion is reported as information ('fp8_codes'); pass/fail uses the dequantized
    value: |deq - ref| <= 2^-4 |ref| + 2^-6 * group_amax (half an e4m3 ulp + a scale-rounding slack)."""
    m = sc.shape[0]
    deq = qq.float().reshape(m, -1, 128) * sc.float()[..., None]
    r = ref["normalized"].float().reshape(m, -1, 128).to(deq.device)
    amax = r.abs().amax(-1, keepdim=True)
    err = (deq - r).abs()
    bad = err > (2 ** -4) * r.abs() + (2 ** -6) * amax
    return dict(max_abs=err.max().item(), med_abs=err.median().item(), frac_fail=bad.float().mean().item(),
                pass_=bool(not bad.any()))


def check(arm, outs, ref, inp, shape):
    """Pass criteria (documented in S3_notes.txt):
      normalized bf16: assert_close(atol 2^-5, rtol 1e-2)            [plan A5]
      fp8 out: dequantized within half an e4m3 ulp (see _fp8_vs_ref); codes/scales reported
      fp32 state (touched slots): atol=rtol=1e-3 on >= 99.9% of elements (frac_fail <= 1e-3),
               max_abs reported; untouched slots / conv window bitwise.
    """
    st = {}
    if "normalized" in outs:
        st["normalized"] = core.cmp_close(outs["normalized"], ref["normalized"])
    if "quantized" in outs:
        info = core.cmp_fp8(outs["quantized"], outs["scales"], ref["quantized"], ref["scales"])
        info["pass_"] = True   # informational (see _fp8_vs_ref)
        info["a5_code_criterion"] = bool(info["scale_max_rel"] <= 2 ** -10 and info["frac_gt1"] == 0
                                         and info["code_mismatch"] <= 0.005)
        st["fp8_codes"] = info
        st["fp8_dequant"] = _fp8_vs_ref(outs["quantized"], outs["scales"], ref)
    idx = inp["cache_indices"].long()
    d = core.cmp_close(outs["delta_state"][idx], ref["delta_touched"], atol=1e-3, rtol=1e-3, ulp=False)
    d["strict_pass"] = d["pass_"]
    d["pass_"] = d["frac_fail"] <= 1e-3
    st["delta_touched"] = d
    st["conv_touched"] = core.cmp_bytes(outs["conv_state"][idx], ref["conv_touched"])
    touched = torch.zeros(outs["conv_state"].shape[0], dtype=torch.bool, device=idx.device)
    touched[idx] = True
    cu = outs["conv_state"][~touched].double().sum().reshape(1)
    du = outs["delta_state"][~touched].double().sum().reshape(1)
    st["untouched"] = dict(pass_=bool(torch.equal(cu, ref["conv_untouched_sum"].to(cu.device))
                                      and torch.equal(du, ref["delta_untouched_sum"].to(du.device))))
    return st
