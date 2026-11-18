"""S12: fused GDN prefill (conv1d + gating + chunked gated delta rule + gated RMSNorm), Qwen3.5
MXFP4 pack (BF16 output), TP4 shard: qkvz 5120 = q 512 | k 512 | v 2048 | z 2048, ba 32 = b 16 | a 16,
4 k-heads, 16 v-heads, d 128, conv width 4, fp32 recurrent state pool [slots, 16, V, K].

T       : qwen3_5/mxfp4/gdn_prefill_xl/xl_gdn_prefill_dynamic.py::gdn_prefill (lead, range row
          1024..16384 x b1..8 + the 16x1024 row); r3_gdn_prefill_template.py for its two keys.
B_port  : aiter fused_gdn_prefill_qkvz (ported from our Qwen3-Next pack).  Qwen3-Next topology
          only -> unsupported for every Qwen3.5 shape (reason recorded from its probe).
B1      : AITER multi-kernel composition: causal_conv1d_split_qkv_triton_fn (conv+split) ->
          gates (torch, as plain ops) -> chunk_gated_delta_rule_opt (l2norm in kernel) with
          state gather/transposed scatter -> gated RMSNorm.  AITER has no bf16-output gated
          RMSNorm, so stage 4 is SGLang's FLA RMSNormGated (non-AITER, flagged).
B1h     : as B1 with the HIP conv (causal_conv1d_split_qkv_hip_fn).
B2      : as B1 with chunk_gated_delta_rule (non-opt).
X_fla   : stock SGLang 703e763d5f prefill path (no AITER op at all): sglang triton
          causal_conv1d_fn -> fused_qkv_split_gdn_prefill -> fused_gdn_gating -> FLA
          chunk_gated_delta_rule (indexed in-place state) -> FLA RMSNormGated.  Context only.
Timing is eager (template not graph-safe).  Harness seeds map to spec seeds: 0 -> 300
("fast", the served fast path), 100 -> 100 (adversarial), 1 -> 0 (benign; exact path).
"""
import torch

from xbench import core

HK, HV, D = 4, 16, 128
QKVZ, CONV = (2 * HK + 2 * HV) * D, (2 * HK + HV) * D      # 5120, 3072
KD, VD = HK * D, HV * D                                   # 512, 2048
SLOTS = 32                                                # pool = 2 x max batch (16)
SCALE, EPS = D ** -0.5, 1e-6
PROFILE = "qwen3_5_mxfp4_tp4_gfx950/gdn_prefill_templated.json"
TPL = {"xl": ("qwen3_5/mxfp4/gdn_prefill_xl/xl_gdn_prefill_dynamic.py", "mxfp4/gdn_prefill_xl/xl_gdn_prefill_dynamic.py"),
       "r3": ("qwen3_5/mxfp4/gdn_prefill_r3/r3_gdn_prefill_template.py", "mxfp4/gdn_prefill_r3/r3_gdn_prefill_template.py")}
SPEC_SEED = {0: 300, 100: 100, 1: 0}
SEEDS = (0, 100, 1)
MODE = "eager"
AA = ["T", "B1"]
B37_ARMS = ["B_port", "B1", "B1h", "B2"]


def _sh(lengths, bits, tpl="xl"):
    lengths = tuple(lengths)
    s = "x".join(map(str, lengths)) if len(lengths) <= 4 else f"{lengths[0]}x{len(lengths)}"
    return dict(label=f"s{s}_i{bits}", lengths=lengths, bits=bits, tpl=tpl)


B8 = (1024,) * 8
SHAPES = [
    _sh((1024,), 0), _sh((1024,), 1), _sh((5120,), 0), _sh((5120,), 1),
    _sh((8192,), 0), _sh((8192,), 1), _sh((16384,), 0), _sh((16384,), 1),
    _sh(B8, 0), _sh(B8, 255), _sh((1024,) * 16, 0),
    _sh((4096, 10240), 2), _sh((2048, 5120, 5120, 4096), 8),
    _sh((10240,), 0, "r3"), _sh((4096, 5120, 5120, 2048), 8, "r3"),
]


def _spec():
    import qwen3_5_mxfp4__gdn_prefill as sp
    return sp


def _flags(shape):
    b = len(shape["lengths"])
    return [bool((shape["bits"] >> (b - 1 - i)) & 1) for i in range(b)]


# ----------------------------------------------------------------------------- inputs / reference
def gen(shape, seed):
    sp = _spec()
    key = (shape["lengths"], shape["bits"])
    inp = sp.make_inputs(key, SPEC_SEED[seed], "cuda")
    batch = len(shape["lengths"])
    old = inp["cache_indices"].long()
    g = torch.Generator(device="cuda").manual_seed(1234 + seed)
    new = torch.randperm(SLOTS, generator=g, device="cuda")[:batch]
    conv = inp["conv_state"][:SLOTS].clone()
    delta = inp["delta_state"][:SLOTS].clone()
    conv[new] = inp["conv_state"][old]
    delta[new] = inp["delta_state"][old]
    inp["conv_state"], inp["delta_state"] = conv.contiguous(), delta.contiguous()
    inp["cache_indices"] = new.to(torch.int32)
    torch.cuda.empty_cache()
    ref = reference(inp, shape)
    # Stock envelope: error stats of the shipped SGLang FLA path vs the same fp32 reference.
    # Pass rule (A5 adapted to a long chained recurrence with bf16 intermediates, where every
    # implementation incl. stock misses the strict bf16 atol on a few outliers): an arm passes if
    # its output med-abs and beyond-tolerance fraction are <= 1.25x stock's and its final-state
    # max-abs is <= 2x stock's.  Strict A5 stats are kept in the record (strict_pass).
    m = core.clone(inp)
    outs = _stock(shape)(m)()
    torch.cuda.synchronize()
    idx = m["cache_indices"].long()
    so = core.cmp_close(outs["output"], ref["output"], ulp=False)
    sd = core.cmp_close(outs["delta_state"][idx], ref["delta"], atol=1e-3, rtol=1e-3, ulp=False)
    ref["env"] = torch.tensor([so["med_abs"], so["frac_fail"], sd["max_abs"]], dtype=torch.float64)
    del m, outs
    return inp, ref


def reference(inp, shape, zero_state_if_no_initial=False):
    """fp32 sequential recurrence (FLA / SGLang semantics; state stored [H, V, K])."""
    x = inp["projected_qkvz"].float()
    ba = inp["projected_ba"].float()
    w = inp["conv_weight"].float()
    A = torch.exp(inp["a_log"].float())
    dtb = inp["dt_bias"].float()
    nw = inp["norm_weight"].float()
    cu = inp["cu_seqlens"].tolist()
    flags = _flags(shape)
    idx = inp["cache_indices"].long().tolist()
    T = x.shape[0]
    out = torch.empty(T, HV, D, device=x.device)
    fin_delta, fin_conv = [], []
    for i in range(len(cu) - 1):
        s, e = cu[i], cu[i + 1]
        xi = x[s:e, :CONV]
        hist = inp["conv_state"][idx[i]].float().t() if flags[i] else torch.zeros(3, CONV, device=x.device)
        xe = torch.cat([hist, xi], 0)                                 # [L+3, C]
        y = sum(xe[j:j + (e - s)] * w[:, j] for j in range(4))
        y = torch.nn.functional.silu(y)
        fin_conv.append(xe[-3:].t().contiguous())                    # [C, 3] oldest..newest
        q = y[:, :KD].view(-1, HK, D)
        k = y[:, KD:2 * KD].view(-1, HK, D)
        v = y[:, 2 * KD:].view(-1, HV, D)
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * SCALE
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        q = q.repeat_interleave(HV // HK, 1)
        k = k.repeat_interleave(HV // HK, 1)
        bb, aa = ba[s:e, :HV], ba[s:e, HV:]
        gdec = torch.exp(-A * torch.nn.functional.softplus(aa + dtb))  # [L, HV]
        beta = torch.sigmoid(bb)
        if zero_state_if_no_initial and not flags[i]:
            S = torch.zeros(HV, D, D, device=x.device)
        else:
            S = inp["delta_state"][idx[i]].float().transpose(-1, -2).contiguous()   # -> [H, K, V]
        o = torch.empty(e - s, HV, D, device=x.device)
        for t in range(e - s):
            S.mul_(gdec[t][:, None, None])
            kt = k[t]
            mem = torch.einsum("hk,hkv->hv", kt, S)
            dlt = (v[t] - mem) * beta[t][:, None]
            S.add_(kt[:, :, None] * dlt[:, None, :])
            o[t] = torch.einsum("hk,hkv->hv", q[t], S)
        fin_delta.append(S.transpose(-1, -2).contiguous())           # back to [H, V, K]
        z = x[s:e, CONV:].view(-1, HV, D)
        rms = torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS)
        out[s:e] = o * rms * nw * torch.nn.functional.silu(z)
    return dict(output=out, delta=torch.stack(fin_delta), conv=torch.stack(fin_conv))


# ----------------------------------------------------------------------------- arms
def _T(shape):
    def build(inp):
        from xbench import tload
        rel, src = TPL[shape["tpl"]]
        fn = tload.load(rel, "gdn_prefill", PROFILE, src)
        names = ("projected_qkvz", "projected_ba", "conv_state", "delta_state", "cache_indices",
                 "cu_seqlens", "has_initial_state", "conv_weight", "a_log", "dt_bias", "norm_weight")

        def call():
            o, c, d = fn(**{k: inp[k] for k in names}, scale=SCALE, eps=EPS)
            return dict(output=o, conv_state=c, delta_state=d)
        return call
    return build


def _port(shape):
    def build(inp):
        import importlib
        m = importlib.import_module("aiter.ops.triton.gated_delta_net.fused_gdn_prefill_qkvz")
        core.restore_triton()
        ok, reason = m.fused_gdn_prefill_qkvz_supported(
            inp["projected_qkvz"], inp["projected_ba"], inp["conv_state"], inp["delta_state"],
            inp["cache_indices"], inp["cu_seqlens"], inp["has_initial_state"], inp["conv_weight"],
            torch.zeros(CONV, dtype=torch.bfloat16, device="cuda"), torch.float8_e4m3fn,
            a_log=inp["a_log"], dt_bias=inp["dt_bias"], norm_weight=inp["norm_weight"])
        if not ok:
            raise core.Unavailable(f"fused_gdn_prefill_qkvz_supported -> False: {reason}")
        raise core.Unavailable("unexpectedly supported; arm not wired for Qwen3.5")
    return build


def _gated_norm(nw):
    """SGLang FLA RMSNormGated (what Qwen3_5GatedDeltaNet.norm runs); built per arm build
    (weights copied once, untimed)."""
    lg = core.sglang_import("sglang.kernels.ops.attention.fla.layernorm_gated")
    n = lg.RMSNorm(D, eps=EPS, group_size=None, norm_before_gate=True,
                   device=nw.device, dtype=torch.bfloat16)
    with torch.no_grad():
        n.weight.copy_(nw)
    return n


def _aiter_chain(shape, conv="triton", gdr="opt"):
    def build(inp):
        from aiter.ops.triton.gated_delta_net import causal_conv1d_prefill as cp
        from aiter.ops.triton.gated_delta_net import gated_delta_rule as gd
        core.restore_triton()
        lengths = list(shape["lengths"])
        T = sum(lengths)
        qkvz, ba = inp["projected_qkvz"], inp["projected_ba"]
        x = qkvz[:, :CONV].t()                         # [C, T] strided view (no copy)
        cu = inp["cu_seqlens"]
        cu64 = cu.long()
        idx = inp["cache_indices"].long()
        hasi = inp["has_initial_state"]
        A_log, dtb = inp["a_log"], inp["dt_bias"]
        norm = _gated_norm(inp["norm_weight"])
        if conv == "hip":
            from aiter.ops.causal_conv1d_fwd_split_qkv import causal_conv1d_split_qkv_hip_fn as hipc
        fn = gd.chunk_gated_delta_rule_opt if gdr == "opt" else gd.chunk_gated_delta_rule

        def call():
            if conv == "hip":
                q, k, v = hipc(x, inp["conv_weight"], None, inp["conv_state"], cu, KD, VD,
                               cache_indices=inp["cache_indices"], has_initial_state=hasi)
            else:
                q, k, v = cp.causal_conv1d_split_qkv_triton_fn(
                    x, inp["conv_weight"], None, inp["conv_state"], cu, lengths, KD, VD,
                    cache_indices=inp["cache_indices"], has_initial_state=hasi)
            b, a = ba[:, :HV], ba[:, HV:]
            g = -torch.exp(A_log.float()) * torch.nn.functional.softplus(a.float() + dtb.float())
            beta = torch.sigmoid(b.float())
            # [N, H, K, V] contiguous (gather + transpose, counted: AITER reads h0 as contiguous)
            h0 = inp["delta_state"][idx].transpose(-1, -2).contiguous()
            qq, kk = q.view(1, T, HK, D), k.view(1, T, HK, D)
            if gdr != "opt":   # chunk_gated_delta_rule has no GVA: expand q/k to 16 heads (counted)
                qq, kk = qq.repeat_interleave(HV // HK, 2), kk.repeat_interleave(HV // HK, 2)
            o, hT = fn(qq, kk, v.view(1, T, HV, D),
                       g.view(1, T, HV), beta.view(1, T, HV), scale=SCALE, initial_state=h0,
                       output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu64)
            inp["delta_state"][idx] = hT.transpose(-1, -2).to(inp["delta_state"].dtype)
            z = qkvz[:, CONV:].reshape(T * HV, D)
            out = norm(o.reshape(T * HV, D), z).view(T, HV, D)
            return dict(output=out, conv_state=inp["conv_state"], delta_state=inp["delta_state"])
        return call
    return build


def _stock(shape):
    def build(inp):
        cc = core.sglang_import("sglang.kernels.ops.mamba.causal_conv1d_triton")
        fp = core.sglang_import("sglang.kernels.ops.attention.triton_gdn_fused_proj")
        gt = core.sglang_import("sglang.kernels.ops.attention.linear.fused_gdn_gating")
        ch = core.sglang_import("sglang.kernels.ops.attention.fla.chunk")
        lengths = list(shape["lengths"])
        T = sum(lengths)
        qkvz, ba = inp["projected_qkvz"], inp["projected_ba"]
        cu = inp["cu_seqlens"]
        norm = _gated_norm(inp["norm_weight"])

        def call():
            mixed = qkvz[:, :CONV]
            y = cc.causal_conv1d_fn(mixed.transpose(0, 1), inp["conv_weight"], None, activation="silu",
                                    conv_states=inp["conv_state"], has_initial_state=inp["has_initial_state"],
                                    cache_indices=inp["cache_indices"], query_start_loc=cu,
                                    seq_lens_cpu=lengths).transpose(0, 1)[:T]
            q, k, v = fp.fused_qkv_split_gdn_prefill(y, HK, HK, HV, D, D, D)
            g, beta = gt.fused_gdn_gating(inp["a_log"], ba[:, HV:], ba[:, :HV], inp["dt_bias"])
            res = ch.chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=inp["delta_state"],
                                            initial_state_indices=inp["cache_indices"], cu_seqlens=cu,
                                            head_first=False, use_qk_l2norm_in_kernel=True, inplace_update=True)
            o = res[0]
            z = qkvz[:, CONV:].reshape(T * HV, D)
            out = norm(o.reshape(T * HV, D), z).view(T, HV, D)
            return dict(output=out, conv_state=inp["conv_state"], delta_state=inp["delta_state"])
        return call
    return build


NPOOL = 20   # = calls per eager round: every timed call starts from the drawn state


def _cycled(name, build):
    """Each call gets its own copy of the (conv, delta) state pools, cycling over NPOOL copies,
    so chained in-place updates never feed the next call (the template takes data-dependent
    exact/fast paths on |h0|).  reset() restores every copy before each timed round."""
    def outer(inp):
        pools = []
        calls = []
        for _ in range(NPOOL):
            v = dict(inp, conv_state=inp["conv_state"].clone(), delta_state=inp["delta_state"].clone())
            pools.append(v)
            calls.append(build(v))
        inp.setdefault("_pools", []).append((pools, inp["conv_state"], inp["delta_state"]))
        it = [0]

        def call():
            c = calls[it[0] % NPOOL]
            it[0] += 1
            return c()
        return call
    return outer


def arms(shape):
    a = {"T": _T(shape), "B_port": _port(shape), "B1": _aiter_chain(shape),
         "B1h": _aiter_chain(shape, conv="hip"), "B2": _aiter_chain(shape, gdr="plain"),
         "X_fla": _stock(shape)}
    return {k: (v if k == "B_port" else _cycled(k, v)) for k, v in a.items()}


def reset(shape, ring):
    def go():
        for m in ring:
            for pools, c0, d0 in m.get("_pools", []):
                for v in pools:
                    v["conv_state"].copy_(c0)
                    v["delta_state"].copy_(d0)
    return go


def check(arm, outs, ref, inp, shape):
    idx = inp["cache_indices"].long()
    env_med, env_frac, env_state = ref["env"].tolist()
    o = core.cmp_close(outs["output"], ref["output"])
    d = core.cmp_close(outs["delta_state"][idx], ref["delta"], atol=1e-3, rtol=1e-3, ulp=False)
    for x in (o, d):
        x["strict_pass"] = x["pass_"]
    o["stock_med_abs"], o["stock_frac_fail"], d["stock_max_abs"] = env_med, env_frac, env_state
    o["pass_"] = bool(o["nonfinite"] == 0 and o["med_abs"] <= 1.25 * env_med + 1e-6
                      and o["frac_fail"] <= 1.25 * env_frac + 1e-5)
    d["pass_"] = bool(d["nonfinite"] == 0 and d["max_abs"] <= 2 * env_state + 1e-4)
    st = {"output": o, "delta_state": d}
    st["conv_state"] = core.cmp_bytes(outs["conv_state"][idx], ref["conv"].to(torch.bfloat16))
    return st
