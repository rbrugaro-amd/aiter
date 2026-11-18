"""S10: Kimi-K3 attention-residual gate (H=7168), key (M, nvb, mode), mode = addend | write<<1 | norm<<2.

Trace (SGLang 703e763d5f, srt/layers/attn_residual.py): on ROCm the K3 attention residual never
calls AITER.  _aggregate / _aggregate_fused_add dispatch to `_aggregate_hip` whenever
supports_attn_res_hip(H, nvb) holds (H=7168, nvb<=8: always for K3), which launches SGLang's own
single Triton kernel sglang.kernels.ops.attention.attn_res_hip._agg_kernel (pending add, bank
snapshot and output RMSNorm folded in).  That kernel is the stock op our template replaced.

T  : kimi_k3/attention_residual_norm_xl/arn_wide_xl.py (M 32..256) / arn_fused_xl.py (M 1,2,8)
B  : AITER aiter.ops.triton.fusions.attn_res.attn_res_gate per bench_plan S10 (SGLang does not use
     AITER here, so this is "AITER's op for the same computation"); write modes use
     close_block=True as-is (returns a new [M, nvb+1, H] block instead of writing in place).
B_strided : same op with the SGLang-layout bank view bank[:, :nvb] (AITER makes it contiguous
     per call: one extra copy kernel) -- what a drop-in into SGLang's AttnResidual would pay.
B1 : write modes only: attn_res_gate(close_block=False) + bank[:, nvb].copy_(prefix_out)
     ("AITER, in-place equivalent").
X_native : SGLang's stock ROCm kernel attn_res_hip (context, non-AITER).
"""
import torch

from xbench import core

H = 7168
BANK_ROWS = 9
EPS = 1e-5
PROFILE = "kimi_k3_mxfp4_tp8_gfx950/kimi_attention_residual_norm_templated.json"

_FUSED = [(1, 4, 5), (2, 1, 5), (2, 1, 6), (2, 4, 4), (2, 4, 5), (2, 4, 6), (2, 8, 4),
          (8, 1, 4), (8, 1, 5), (8, 1, 6), (8, 4, 4), (8, 4, 6), (8, 8, 4)]
_WIDE = [(m, nvb, mode) for m in (32, 64, 128, 256)
         for (nvb, mode) in ((1, 5), (1, 6), (4, 4), (4, 6), (8, 4))]
SHAPES = [dict(label=f"m{m}_v{nvb}_md{mode}", key=(m, nvb, mode)) for (m, nvb, mode) in _FUSED + _WIDE]
MODE = "graph"
SEEDS = (0, 100, 200)
AA = ["T", "B"]


def _spec():
    import kimi_k3__attention_residual_norm as sp
    return sp


def gen(shape, seed):
    m, nvb, mode = shape["key"]
    inp = _spec()._make_inputs(shape["key"], seed, "cuda")
    has_add, write, norm = bool(mode & 1), bool(mode & 2), bool(mode & 4)
    prefix, addend, bank = inp["prefix"], inp["addend"], inp["bank"]
    cw, ow = inp["score_weight"], inp["output_weight"]
    cur = (prefix.float() + addend.float()).to(torch.bfloat16) if has_add else prefix
    rows = torch.cat([bank[:, :nvb, :], cur[:, None, :]], 1).double()       # [M, nvb+1, H]
    rs = torch.rsqrt(rows.pow(2).mean(-1) + EPS)
    scores = (rows * cw.double()).sum(-1) * rs
    p = torch.softmax(scores, -1)
    mixed = (p[..., None] * rows).sum(1)
    if norm:
        out = mixed * torch.rsqrt(mixed.pow(2).mean(-1, keepdim=True) + EPS) * ow.double()
    else:
        out = mixed
    rbank = bank.clone()
    if write:
        rbank[:, nvb, :] = cur
    return inp, dict(out=out.float(), current=cur, bank=rbank)


def _flags(shape):
    m, nvb, mode = shape["key"]
    return nvb, bool(mode & 1), bool(mode & 2), bool(mode & 4)


def _tpl(m):
    from xbench import tload
    if m <= 8:
        f = "attention_residual_norm_xl/arn_fused_xl.py"
        return tload.load("kimi_k3/" + f, "attention_residual_norm_fused_xl", PROFILE, f)
    f = "attention_residual_norm_xl/arn_wide_xl.py"
    return tload.load("kimi_k3/" + f, "attention_residual_norm_wide_xl", PROFILE, f)


def _T(shape):
    nvb, add, write, norm = _flags(shape)
    fn = _tpl(shape["key"][0])

    def build(inp):
        kw = dict(valid_rows=nvb, has_addend=add, write_bank=write, apply_output_norm=norm,
                  score_eps=EPS, output_eps=EPS)
        a = (inp["prefix"], inp["addend"], inp["bank"], inp["score_weight"], inp["output_weight"])

        def call():
            out, cur, bank = fn(*a, **kw)
            return dict(out=out, current=cur, bank=bank)
        return call
    return build


def _aiter():
    from aiter.ops.triton.fusions.attn_res import attn_res_gate
    return attn_res_gate


def _B(shape, inplace=False, strided=False):
    nvb, add, write, norm = _flags(shape)
    gate = _aiter()

    def build(inp):
        prefix, bank = inp["prefix"], inp["bank"]
        # packed contiguous [M, nvb, H] block (ATOM layout; untimed prep) unless strided=True,
        # where the [M, 9, H] bank view is passed and AITER copies it to contiguous per call.
        br = bank[:, :nvb, :] if strided else bank[:, :nvb, :].contiguous()
        kw = dict(add_hidden=inp["addend"] if add else None,
                  output_rms_weight=inp["output_weight"] if norm else None, output_rms_eps=EPS,
                  close_block=write and not inplace)
        sw = inp["score_weight"]

        def call():
            r = gate(prefix, br, sw, EPS, **kw)
            d = dict(out=r[0], current=r[1])
            if write and not inplace:
                d["bank_block"] = r[2]
            if write and inplace:
                bank[:, nvb].copy_(r[1])
                d["bank"] = bank
            return d
        return call
    return build


def _X(shape):
    nvb, add, write, norm = _flags(shape)
    mod = core.sglang_import("sglang.kernels.ops.attention.attn_res_hip")

    def build(inp):
        prefix, bank, cw, ow = inp["prefix"], inp["bank"], inp["score_weight"], inp["output_weight"]
        addend = inp["addend"] if add else None

        def call():
            # mirrors attn_residual._aggregate_hip
            prefix_out = prefix if addend is None else torch.empty_like(prefix)
            out = torch.empty_like(prefix)
            mod.attn_res_hip(prefix, bank, cw, ow if norm else None, out, nvb, EPS, EPS if norm else 0.0,
                             addend=addend, prefix_out=prefix_out, write_prefix=write)
            return dict(out=out, current=prefix_out, bank=bank)
        return call
    return build


def arms(shape):
    _, _, write, _ = _flags(shape)
    a = {"T": _T(shape), "B": _B(shape), "B_strided": _B(shape, strided=True), "X_native": _X(shape)}
    if write:
        a["B1"] = _B(shape, inplace=True)
    return a


def check(arm, outs, ref, inp, shape):
    nvb, add, write, norm = _flags(shape)
    st = {"out": core.cmp_close(outs["out"], ref["out"])}
    st["current"] = core.cmp_bytes(outs["current"], ref["current"])
    if "bank" in outs:
        st["bank"] = core.cmp_bytes(outs["bank"], ref["bank"])
    if "bank_block" in outs:
        st["bank_block"] = core.cmp_bytes(outs["bank_block"], ref["bank"][:, :nvb + 1].contiguous())
    return st
