"""S12f: fused GDN prefill + gated RMSNorm + group-128 FP8 quant, Qwen3.5 FP8 pack (TP4 shard:
qkvz 5120, ba 32, 4 k-heads, 16 v-heads, d 128, conv bias), state pool [slots, 16, V, K] fp32.

T     : qwen3_5/fp8/gdn_prefill_group_fp8_quant_templated/gdn_prefill_template.py
        ::gdn_prefill_group_fp8_quant.  Contiguous (contiguous.v1) rows take the serving projections
        as is; v2 rows (every sequence <= 2048) take the head-interleaved layout, so for those T
        includes the adapter's per-call qwen35_prefill_layout.pack() (served path).
B1    : AITER composition: causal_conv1d_split_qkv_triton_fn (conv+bias+silu+split) -> torch gates
        -> chunk_gated_delta_rule_opt (state gather/transposed scatter counted) ->
        aiter gated_rmsnorm_fp8_group_quant (HIP, fp8 values + group-128 scales).
B1h   : as B1 with the HIP conv.   B2: as B1 with chunk_gated_delta_rule (q/k expanded).
X_fla : stock SGLang 703e763d5f prefill (no AITER until out_proj): sglang conv -> split -> gating
        -> FLA chunk GDR -> FLA RMSNormGated (bf16) -> AITER per_1x128 quant (what
        aiter_w8a8_block_fp8_linear runs on the bf16 out_proj input).  Context only.
Inputs are stored in the contiguous serving layout (v2 draws unpacked from the spec's packed
layout).  Seeds: 0 benign, 100 adversarial (spec seeds).
"""
import torch

from xbench import core
from xbench.cands import S12 as base

HK, HV, D, CONV, KD, VD = base.HK, base.HV, base.D, base.CONV, base.KD, base.VD
SCALE, EPS = base.SCALE, base.EPS
PROFILE = "qwen3_5_fp8_tp4_gfx950/gdn_prefill_group_fp8_quant_templated.json"
TPL_REL = "qwen3_5/fp8/gdn_prefill_group_fp8_quant_templated/gdn_prefill_template.py"
TPL_SRC = "gdn_prefill_group_fp8_quant_templated/gdn_prefill_template.py"
SEEDS = (0, 100)
MODE = "eager"
AA = ["T", "B1"]
B37_ARMS = ["B1", "B1h", "B2"]


def _sh(lengths):
    lengths = tuple(lengths)
    s = "x".join(map(str, lengths)) if len(lengths) <= 3 else f"{lengths[0]}x{len(lengths)}_m{sum(lengths)}"
    return dict(label=f"s{s}", lengths=lengths, packed=max(lengths) <= 2048)


SHAPES = [_sh((5000,)), _sh((5000, 5000, 5000)), _sh((16384,)), _sh((10000, 6384)),
          _sh((2048,) * 7 + (1024,)), _sh((2048,) * 8)]


def _spec():
    import qwen3_5_fp8__gdn_prefill_group_fp8_quant as sp
    return sp


def gen(shape, seed):
    sp = _spec()
    inp = sp.make_inputs(shape["lengths"], seed, "cuda")
    if shape["packed"]:
        src, bsrc = sp._pack_index("cuda")
        q = torch.empty_like(inp["projected_qkvz"])
        q[:, src] = inp["projected_qkvz"]
        b = torch.empty_like(inp["projected_ba"])
        b[:, bsrc] = inp["projected_ba"]
        inp["projected_qkvz"], inp["projected_ba"] = q.contiguous(), b.contiguous()
    ref = reference(inp, shape)
    m = core.clone(inp)
    outs = _stock(shape)(m)()
    torch.cuda.synchronize()
    idx = m["cache_indices"].long()
    so = core.cmp_close(_deq(outs), ref["output"], ulp=False)
    sd = core.cmp_close(outs["delta_state"][idx], ref["delta"], atol=1e-3, rtol=1e-3, ulp=False)
    ref["env"] = torch.tensor([so["med_abs"], so["frac_fail"], sd["max_abs"]], dtype=torch.float64)
    return inp, ref


def reference(inp, shape):
    """S12 fp32 recurrence with conv bias (same semantics)."""
    x = inp["projected_qkvz"].float()
    ba = inp["projected_ba"].float()
    w = inp["conv_weight"].float()
    bias = inp["conv_bias"].float()
    A = torch.exp(inp["a_log"].float())
    dtb = inp["dt_bias"].float()
    nw = inp["norm_weight"].float()
    cu = inp["cu_seqlens"].tolist()
    flags = inp["has_initial_state"].tolist()
    idx = inp["cache_indices"].long().tolist()
    out = torch.empty(x.shape[0], HV, D, device=x.device)
    fd, fc = [], []
    for i in range(len(cu) - 1):
        s, e = cu[i], cu[i + 1]
        hist = inp["conv_state"][idx[i]].float().t() if flags[i] else torch.zeros(3, CONV, device=x.device)
        xe = torch.cat([hist, x[s:e, :CONV]], 0)
        y = torch.nn.functional.silu(sum(xe[j:j + (e - s)] * w[:, j] for j in range(4)) + bias)
        fc.append(xe[-3:].t().contiguous())
        q = y[:, :KD].view(-1, HK, D)
        k = y[:, KD:2 * KD].view(-1, HK, D)
        v = y[:, 2 * KD:].view(-1, HV, D)
        q = (q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * SCALE).repeat_interleave(4, 1)
        k = (k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)).repeat_interleave(4, 1)
        gdec = torch.exp(-A * torch.nn.functional.softplus(ba[s:e, HV:] + dtb))
        beta = torch.sigmoid(ba[s:e, :HV])
        S = inp["delta_state"][idx[i]].float().transpose(-1, -2).contiguous()
        o = torch.empty(e - s, HV, D, device=x.device)
        for t in range(e - s):
            S.mul_(gdec[t][:, None, None])
            dl = (v[t] - torch.einsum("hk,hkv->hv", k[t], S)) * beta[t][:, None]
            S.add_(k[t][:, :, None] * dl[:, None, :])
            o[t] = torch.einsum("hk,hkv->hv", q[t], S)
        fd.append(S.transpose(-1, -2).contiguous())
        z = x[s:e, CONV:].view(-1, HV, D)
        out[s:e] = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS) * nw * torch.nn.functional.silu(z)
    return dict(output=out, delta=torch.stack(fd), conv=torch.stack(fc))


def _deq(outs):
    v = outs["values"].float().view(-1, HV, D)
    return v * outs["scales"].float().reshape(-1, HV, 1)


def _T(shape):
    def build(inp):
        from types import SimpleNamespace
        from xbench import tload
        fn = tload.load(TPL_REL, "gdn_prefill_group_fp8_quant", PROFILE, TPL_SRC)
        prec = SimpleNamespace(dtype=torch.float8_e4m3fn, max_finite=448.0, group_quant_max=448.0)
        names = ("projected_qkvz", "projected_ba", "conv_state", "delta_state", "cache_indices",
                 "cu_seqlens", "has_initial_state", "conv_weight", "conv_bias", "a_log", "dt_bias",
                 "norm_weight")
        pack = None
        if shape["packed"]:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "xb_qwen35_prefill_layout", "/work/src/artemis_sglang/adapters/qwen35_prefill_layout.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            pack = mod.pack

        def call():
            kw = {k: inp[k] for k in names}
            if pack is not None:   # served v2 path: adapter packs the projections per call
                kw["projected_qkvz"], kw["projected_ba"] = pack(inp["projected_qkvz"], inp["projected_ba"])
            n, c, d, v, s = fn(**kw, scale=SCALE, eps=EPS, precision_config=prec)
            return dict(normalized=n, conv_state=c, delta_state=d, values=v, scales=s)
        return call
    return build


def _chain(shape, conv="triton", gdr="opt"):
    def build(inp):
        from aiter.ops.triton.gated_delta_net import causal_conv1d_prefill as cp
        from aiter.ops.triton.gated_delta_net import gated_delta_rule as gd
        from aiter.ops.gated_rmsnorm_fp8_group_quant import gated_rmsnorm_fp8_group_quant
        core.restore_triton()
        lengths = list(shape["lengths"])
        T = sum(lengths)
        qkvz, ba = inp["projected_qkvz"], inp["projected_ba"]
        x = qkvz[:, :CONV].t()
        cu, cu64, idx = inp["cu_seqlens"], inp["cu_seqlens"].long(), inp["cache_indices"].long()
        if conv == "hip":
            from aiter.ops.causal_conv1d_fwd_split_qkv import causal_conv1d_split_qkv_hip_fn as hipc
        fn = gd.chunk_gated_delta_rule_opt if gdr == "opt" else gd.chunk_gated_delta_rule

        def call():
            if conv == "hip":
                q, k, v = hipc(x, inp["conv_weight"], inp["conv_bias"], inp["conv_state"], cu, KD, VD,
                               cache_indices=inp["cache_indices"], has_initial_state=inp["has_initial_state"])
            else:
                q, k, v = cp.causal_conv1d_split_qkv_triton_fn(
                    x, inp["conv_weight"], inp["conv_bias"], inp["conv_state"], cu, lengths, KD, VD,
                    cache_indices=inp["cache_indices"], has_initial_state=inp["has_initial_state"])
            g = -torch.exp(inp["a_log"].float()) * torch.nn.functional.softplus(ba[:, HV:].float() + inp["dt_bias"].float())
            beta = torch.sigmoid(ba[:, :HV].float())
            h0 = inp["delta_state"][idx].transpose(-1, -2).contiguous()
            qq, kk = q.view(1, T, HK, D), k.view(1, T, HK, D)
            if gdr != "opt":
                qq, kk = qq.repeat_interleave(4, 2), kk.repeat_interleave(4, 2)
            o, hT = fn(qq, kk, v.view(1, T, HV, D), g.view(1, T, HV), beta.view(1, T, HV), scale=SCALE,
                       initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu64)
            inp["delta_state"][idx] = hT.transpose(-1, -2).to(inp["delta_state"].dtype)
            vals = torch.empty(T, HV * D, dtype=torch.float8_e4m3fn, device=qkvz.device)
            scl = torch.empty(T, HV, dtype=torch.float32, device=qkvz.device)
            gated_rmsnorm_fp8_group_quant(vals, scl, o.view(T, HV, D), qkvz[:, CONV:].view(T, HV, D),
                                          inp["norm_weight"], EPS, 128)
            return dict(conv_state=inp["conv_state"], delta_state=inp["delta_state"], values=vals, scales=scl)
        return call
    return build


def _stock(shape):
    def build(inp):
        import aiter
        cc = core.sglang_import("sglang.kernels.ops.mamba.causal_conv1d_triton")
        fp = core.sglang_import("sglang.kernels.ops.attention.triton_gdn_fused_proj")
        gt = core.sglang_import("sglang.kernels.ops.attention.linear.fused_gdn_gating")
        ch = core.sglang_import("sglang.kernels.ops.attention.fla.chunk")
        quant = aiter.get_hip_quant(aiter.QuantType.per_1x128)
        lengths = list(shape["lengths"])
        T = sum(lengths)
        qkvz, ba, cu = inp["projected_qkvz"], inp["projected_ba"], inp["cu_seqlens"]
        norm = base._gated_norm(inp["norm_weight"])

        def call():
            y = cc.causal_conv1d_fn(qkvz[:, :CONV].transpose(0, 1), inp["conv_weight"], inp["conv_bias"],
                                    activation="silu", conv_states=inp["conv_state"],
                                    has_initial_state=inp["has_initial_state"], cache_indices=inp["cache_indices"],
                                    query_start_loc=cu, seq_lens_cpu=lengths).transpose(0, 1)[:T]
            q, k, v = fp.fused_qkv_split_gdn_prefill(y, HK, HK, HV, D, D, D)
            g, beta = gt.fused_gdn_gating(inp["a_log"], ba[:, HV:], ba[:, :HV], inp["dt_bias"])
            o = ch.chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=inp["delta_state"],
                                          initial_state_indices=inp["cache_indices"], cu_seqlens=cu,
                                          head_first=False, use_qk_l2norm_in_kernel=True, inplace_update=True)[0]
            n = norm(o.reshape(T * HV, D), qkvz[:, CONV:].reshape(T * HV, D)).view(T, HV * D)
            vals, scl = quant(n, quant_dtype=torch.float8_e4m3fn)
            return dict(normalized=n, conv_state=inp["conv_state"], delta_state=inp["delta_state"],
                        values=vals, scales=scl)
        return call
    return build


def arms(shape):
    a = {"T": _T(shape), "B1": _chain(shape), "B1h": _chain(shape, conv="hip"),
         "B2": _chain(shape, gdr="plain"), "X_fla": _stock(shape)}
    return {k: base._cycled(k, v) for k, v in a.items()}


reset = base.reset


def check(arm, outs, ref, inp, shape):
    idx = inp["cache_indices"].long()
    env_med, env_frac, env_state = ref["env"].tolist()
    o = core.cmp_close(_deq(outs), ref["output"])        # dequantized fp8 output vs fp32 reference
    d = core.cmp_close(outs["delta_state"][idx], ref["delta"], atol=1e-3, rtol=1e-3, ulp=False)
    for x in (o, d):
        x["strict_pass"] = x["pass_"]
    o["stock_med_abs"], o["stock_frac_fail"], d["stock_max_abs"] = env_med, env_frac, env_state
    o["pass_"] = bool(o["nonfinite"] == 0 and o["med_abs"] <= 1.25 * env_med + 1e-6
                      and o["frac_fail"] <= 1.25 * env_frac + 1e-5)
    d["pass_"] = bool(d["nonfinite"] == 0 and d["max_abs"] <= 2 * env_state + 1e-4)
    st = {"fp8_out_dequant": o, "delta_state": d,
          "conv_state": core.cmp_bytes(outs["conv_state"][idx], ref["conv"].to(torch.bfloat16))}
    sc = outs["scales"]
    st["scales_finite"] = dict(pass_=bool(torch.isfinite(sc).all().item() and (sc > 0).all().item()))
    return st
