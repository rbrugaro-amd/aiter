import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
_SCHEDULE = {32: dict(GROUPS=4, CYCLIC=True, FOLD=4, BUFFER_SCALES=False, BUFFER_STORE=True, waves_per_eu=2)}
_N_TILE = 64
_SPLITS = 6

@gluon.jit
def _load_record(X, W, WS, ao, wo, ga, gb, gs, start, scale_offset, BUFFER_SCALES: gl.constexpr):
    GROUPS: gl.constexpr = ga.shape[0]
    M: gl.constexpr = ao.shape[1]
    BN: gl.constexpr = wo.shape[2]
    am = gl.full((GROUPS, M, 1), True, gl.int1, ao.type.layout)
    wm = gl.full((GROUPS, 1, BN), True, gl.int1, wo.type.layout)
    xx = gl.amd.cdna4.buffer_load(X + start * 128, ao + ga[:, None, None] * 128, am, 0.0)
    ww = gl.amd.cdna4.buffer_load(W + start * 2048, wo + gb[:, None, None] * 2048, wm, 0.0)
    scale_base = WS + scale_offset + start
    if BUFFER_SCALES:
        sw = gl.amd.cdna4.buffer_load(scale_base, gs, gl.full((GROUPS,), True, gl.int1, gs.type.layout), 0.0)
    else:
        sw = gl.load(scale_base + gs)
    if M == 32:
        xx = xx.to(gl.float32)
    return (xx, ww, sw)

@gluon.jit
def _apply_record(xx, ww, sw, zeros):
    ml: gl.constexpr = zeros.type.layout
    ad: gl.constexpr = gl.DotOperandLayout(0, ml, 16)
    xx = xx.to(gl.float32)
    scale = gl.maximum(gl.max(gl.abs(xx), 2), 1e-10) * (1.0 / 448.0)
    aa = gl.convert_layout((xx * gl.div_rn(1.0, scale)[:, :, None]).to(ww.dtype), ad)
    sa = gl.convert_layout(scale, gl.SliceLayout(2, ml))
    part = gl.amd.cdna4.mfma(aa, ww, zeros)
    return part * (sa * sw[:, None])[:, :, None]

@gluon.jit
def _project32(X, W, WS, P, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, SX: gl.constexpr, GROUPS: gl.constexpr, CYCLIC: gl.constexpr, FOLD: gl.constexpr, BUFFER_SCALES: gl.constexpr, BUFFER_STORE: gl.constexpr, BN: gl.constexpr, SPLIT: gl.constexpr, PS: gl.constexpr):
    gl.static_assert(M == 32)
    gl.static_assert(BN == 64 and N % BN == 0)
    gl.static_assert(K % (128 * GROUPS * SPLIT) == 0)
    row_waves: gl.constexpr = gl.num_warps() // GROUPS
    linear = gl.program_id(0)
    si = linear % SPLIT
    ni = linear // SPLIT % gl.cdiv(N, BN)
    ml: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [GROUPS, row_waves, 1])
    bd: gl.constexpr = gl.DotOperandLayout(1, ml, 16)
    al: gl.constexpr = gl.BlockedLayout([1, 1, 16], [1, 16, 4], [GROUPS, row_waves, 1], [2, 1, 0])
    ga = gl.arange(0, GROUPS, gl.SliceLayout(1, gl.SliceLayout(2, al)))
    ra = gl.arange(0, M, gl.SliceLayout(0, gl.SliceLayout(2, al)))
    ka = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(1, al)))
    gb = gl.arange(0, GROUPS, gl.SliceLayout(1, gl.SliceLayout(2, bd)))
    nb = gl.arange(0, BN, gl.SliceLayout(0, gl.SliceLayout(1, bd))) + ni * BN
    kb = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(2, bd)))
    gs = gl.arange(0, GROUPS, gl.SliceLayout(1, gl.SliceLayout(2, ml)))
    ITERS: gl.constexpr = K // (128 * GROUPS * SPLIT)
    ao = ra[None, :, None] * SX + ka[None, None, :]
    W += ni * BN * K
    wn = nb - ni * BN
    wo = wn[None, None, :] // 16 * K * 16 + kb[None, :, None] // 32 * 512
    wo += kb[None, :, None] // 16 % 2 * 256 + wn[None, None, :] % 16 * 16 + kb[None, :, None] % 16
    first = si * GROUPS if CYCLIC else si * GROUPS * ITERS
    stride: gl.constexpr = GROUPS * SPLIT if CYCLIC else GROUPS
    WS += ni * BN // 128 * (K // 128)
    scale_offset = 0
    zeros = gl.zeros((GROUPS, M, BN), gl.float32, ml)
    acc = zeros
    for it in gl.static_range(ITERS):
        xx, ww, sw = _load_record(X, W, WS, ao, wo, ga, gb, gs, first + it * stride, scale_offset, BUFFER_SCALES)
        acc += _apply_record(xx, ww, sw, zeros)
    fold_order: gl.constexpr = [2, 0, 1]
    fold: gl.constexpr = gl.BlockedLayout([GROUPS, 1, FOLD], [1, FOLD, 64 // FOLD], [1, gl.num_warps(), 1], fold_order)
    merged = gl.sum(gl.convert_layout(acc, fold), 0)
    cl: gl.constexpr = merged.type.layout
    row = gl.arange(0, M, gl.SliceLayout(1, cl))
    col = ni * BN + gl.arange(0, BN, gl.SliceLayout(0, cl))
    if BUFFER_STORE:
        gl.amd.cdna4.buffer_store(stored_value=merged.to(P.dtype.element_ty), ptr=P + si * M * PS, offsets=row[:, None] * PS + col[None, :], cache='.wt')
    else:
        gl.store(P + si * M * PS + row[:, None] * PS + col[None, :], merged, (row[:, None] < M) & (col[None, :] < N), cache_modifier='.wt')

@gluon.jit
def _finish_role(P, Gamma, Out, Rope, row, M: gl.constexpr, N: gl.constexpr, OFFSET: gl.constexpr, WIDTH: gl.constexpr, R: gl.constexpr, EPS: gl.constexpr, SPLIT: gl.constexpr, VEC: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [gl.num_warps()], [0])
    d = gl.arange(0, triton.next_power_of_2(WIDTH + R), layout)
    offsets = row * N + OFFSET + d
    x = gl.amd.cdna4.buffer_load(P, offsets, d < WIDTH + R, 0.0).to(gl.float32)
    for split in gl.static_range(1, SPLIT):
        x += gl.amd.cdna4.buffer_load(P, split * M * N + offsets, d < WIDTH + R, 0.0).to(gl.float32)
    x = x.to(gl.bfloat16).to(gl.float32)
    gam = gl.load(Gamma + d, d < WIDTH, 0).to(gl.float32)
    xn = gl.where(d < WIDTH, x, 0.0)
    variance = gl.sum(xn * xn, 0) / WIDTH
    y = x * gl.rsqrt(variance + EPS) * gam
    gl.store(Out + row * WIDTH + d, y.to(gl.bfloat16), d < WIDTH, cache_modifier='.wt')
    if R:
        gl.store(Rope + row * R + d - WIDTH, x.to(gl.bfloat16), (d >= WIDTH) & (d < WIDTH + R), cache_modifier='.wt')

@gluon.jit
def _finish(P, QG, KG, QO, KO, RO, M: gl.constexpr, N: gl.constexpr, Q: gl.constexpr, KV: gl.constexpr, R: gl.constexpr, EPS: gl.constexpr, SPLIT: gl.constexpr):
    row = gl.program_id(0)
    if gl.program_id(1) == 0:
        _finish_role(P, QG, QO, RO, row, M, N, 0, Q, 0, EPS, SPLIT, 8)
    else:
        _finish_role(P, KG, KO, RO, row, M, N, Q, KV, R, EPS, SPLIT, 4)

@gluon.jit
def _load64(X, W, WS, ao, wo, gs, start, ni, BN: gl.constexpr, K_GROUPS: gl.constexpr):
    xx = gl.amd.cdna4.buffer_load(X + start * 128, ao)
    ww = gl.amd.cdna4.buffer_load(W + start * 2048, wo)
    sw = gl.load(WS + ni * BN // 128 * K_GROUPS + start + gs)
    return (xx.to(gl.float32), ww, sw)

@gluon.jit
def _apply64(xx, ww, sw, zeros):
    ml: gl.constexpr = zeros.type.layout
    ad: gl.constexpr = gl.DotOperandLayout(0, ml, 16)
    scale = gl.maximum(gl.max(gl.abs(xx), 2), 1e-10) * (1.0 / 448.0)
    xx = gl.convert_layout((xx * gl.div_rn(1.0, scale)[:, :, None]).to(ww.dtype), ad)
    sa = gl.convert_layout(scale, gl.SliceLayout(2, ml))
    part = gl.amd.cdna4.mfma(xx, ww, zeros)
    return part * (sa * sw[:, None])[:, :, None]

@gluon.jit
def _project64(X, W, WS, P, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, SX: gl.constexpr, BN: gl.constexpr, LG: gl.constexpr, SPLIT: gl.constexpr, FOLD_VALUES: gl.constexpr, FOLD_ROWS: gl.constexpr, PS: gl.constexpr):
    gl.static_assert(N % BN == 0 and K % (128 * LG * SPLIT) == 0, 'The decode schedule requires complete N tiles and K groups.')
    linear = gl.program_id(0).to(gl.uint32)
    si = linear % SPLIT
    ni = linear // SPLIT % gl.cdiv(N, BN)
    WM: gl.constexpr = gl.num_warps() // LG
    ml: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [LG, WM, 1])
    bd: gl.constexpr = gl.DotOperandLayout(1, ml, 16)
    al: gl.constexpr = gl.BlockedLayout([1, 1, 16], [1, 16, 4], [LG, WM, 1], [2, 1, 0])
    ga = gl.arange(0, LG, gl.SliceLayout(1, gl.SliceLayout(2, al)))
    ra = gl.arange(0, M, gl.SliceLayout(0, gl.SliceLayout(2, al)))
    ka = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(1, al)))
    gb = gl.arange(0, LG, gl.SliceLayout(1, gl.SliceLayout(2, bd)))
    nb = gl.arange(0, BN, gl.SliceLayout(0, gl.SliceLayout(1, bd))) + ni * BN
    kb = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(2, bd)))
    gs = gl.arange(0, LG, gl.SliceLayout(1, gl.SliceLayout(2, ml)))
    ao = ra[None, :, None] * SX + ka[None, None, :]
    ao += ga[:, None, None] * 128
    wo = nb[None, None, :] // 16 * K * 16 + kb[None, :, None] // 32 * 512
    wo += kb[None, :, None] // 16 % 2 * 256 + nb[None, None, :] % 16 * 16 + kb[None, :, None] % 16
    wo += gb[:, None, None] * 2048
    zeros = gl.zeros((LG, M, BN), gl.float32, ml)
    acc = zeros
    ITERS: gl.constexpr = K // (128 * LG * SPLIT)
    first = si * LG
    xx, ww, sw = _load64(X, W, WS, ao, wo, gs, first, ni, BN, K // 128)
    for it in gl.static_range(ITERS - 1):
        nx, nw, nsw = _load64(X, W, WS, ao, wo, gs, first + (it + 1) * LG * SPLIT, ni, BN, K // 128)
        acc += _apply64(xx, ww, sw, zeros)
        xx, ww, sw = (nx, nw, nsw)
    acc += _apply64(xx, ww, sw, zeros)
    fold: gl.constexpr = gl.BlockedLayout([LG, 1, FOLD_VALUES], [1, FOLD_ROWS, 64 // FOLD_ROWS], [1, gl.num_warps(), 1], [0, 2, 1])
    merged = gl.sum(gl.convert_layout(acc, fold), 0)
    cl: gl.constexpr = merged.type.layout
    row = gl.arange(0, M, gl.SliceLayout(1, cl))
    col = ni * BN + gl.arange(0, BN, gl.SliceLayout(0, cl))
    offsets = si * M * PS + row[:, None] * PS + col[None, :]
    gl.amd.cdna4.buffer_store(merged.to(P.dtype.element_ty), P, offsets, cache='.wt')

def mla_qkv_a_norm(hidden: torch.Tensor, weight: torch.Tensor, q_gamma: torch.Tensor, kv_gamma: torch.Tensor, weight_scales: torch.Tensor | None=None, *, rope_dim: int=64, eps: float=1e-05, weight_preshuffled: bool=False):
    m, k = hidden.shape
    q, kv = (q_gamma.numel(), kv_gamma.numel())
    n = q + kv + rope_dim
    assert weight_preshuffled and weight_scales is not None
    qo = torch.empty((m, q), device=hidden.device, dtype=torch.bfloat16)
    ko = torch.empty((m, kv), device=hidden.device, dtype=torch.bfloat16)
    ro = torch.empty((m, rope_dim), device=hidden.device, dtype=torch.bfloat16)
    pitch = n if m == 32 else triton.cdiv(n, 1024) * 1024
    p = torch.empty((_SPLITS, m, pitch), device=hidden.device, dtype=torch.float32)
    if m == 32:
        _project32[triton.cdiv(n, _N_TILE) * _SPLITS,](hidden, weight, weight_scales, p, M=m, N=n, K=k, SX=hidden.stride(0), BN=_N_TILE, SPLIT=_SPLITS, PS=pitch, **_SCHEDULE[m], num_warps=8, enable_fp_fusion=True)
    else:
        _project64[triton.cdiv(n, _N_TILE) * _SPLITS,](hidden, weight, weight_scales, p, M=m, N=n, K=k, SX=hidden.stride(0), BN=_N_TILE, LG=2, SPLIT=_SPLITS, FOLD_VALUES=8, FOLD_ROWS=8, PS=pitch, num_warps=8, waves_per_eu=4, enable_fp_fusion=True)
    _finish[m, 2](p, q_gamma, kv_gamma, qo, ko, ro, m, pitch, q, kv, rope_dim, eps, _SPLITS, num_warps=4, enable_fp_fusion=False)
    return (qo, ko, ro)
