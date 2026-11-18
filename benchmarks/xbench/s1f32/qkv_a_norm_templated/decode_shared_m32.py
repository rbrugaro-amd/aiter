import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
_N_TILE, _SPLITS = (64, 6)
_TUNING = {32: dict(CHAINS=4, PREFETCH=False, CYCLIC=False, BUFFER_SCALES=True, ROTATE=True, FOLD_VEC=4, FOLD_ROW_LANES=4, FOLD_ORDER=(2, 0, 1), INDEX_2D=True, waves_per_eu=2), 64: dict(CHAINS=2, PREFETCH=True, CYCLIC=True, BUFFER_SCALES=False, ROTATE=False, FOLD_VEC=8, FOLD_ROW_LANES=8, FOLD_ORDER=(0, 2, 1), INDEX_2D=False, waves_per_eu=4)}
_FINISH_CONFIG = dict(Q_VEC=8, KV_VEC=4, num_warps=4, enable_fp_fusion=False)

@gluon.jit
def _indices(g: gl.constexpr, rows: gl.constexpr, cols: gl.constexpr, layout: gl.constexpr):
    gi = gl.arange(0, g, gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    ri = gl.arange(0, rows, gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    ci = gl.arange(0, cols, gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    return (gi[:, None, None], ri[None, :, None], ci[None, None, :])

@gluon.jit
def _load_record(X, W, WS, offsets, start, BUFFER_SCALES: gl.constexpr, PREFETCH: gl.constexpr):
    ao, wo, gs = offsets
    x = gl.amd.cdna4.buffer_load(X + start * 128, ao)
    w = gl.amd.cdna4.buffer_load(W + start * 2048, wo)
    sw = gl.amd.cdna4.buffer_load(WS + start, gs) if BUFFER_SCALES else gl.load(WS + start + gs)
    return (x if PREFETCH else x.to(gl.float32), w, sw)

@gluon.jit
def _apply_record(record, zeros):
    x, w, sw = record
    ml: gl.constexpr = zeros.type.layout
    ad: gl.constexpr = gl.DotOperandLayout(0, ml, 16)
    x = x.to(gl.float32)
    scale = gl.maximum(gl.max(gl.abs(x), 2), 1e-10) * (1.0 / 448.0)
    a = gl.convert_layout((x * gl.div_rn(1.0, scale)[:, :, None]).to(w.dtype), ad)
    sa = gl.convert_layout(scale, gl.SliceLayout(2, ml))
    return gl.amd.cdna4.mfma(a, w, zeros) * (sa * sw[:, None])[:, :, None]

@gluon.jit
def _project(X, W, WS, P, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, SX: gl.constexpr, BN: gl.constexpr, SPLIT: gl.constexpr, CHAINS: gl.constexpr, PREFETCH: gl.constexpr, CYCLIC: gl.constexpr, BUFFER_SCALES: gl.constexpr, FOLD_VEC: gl.constexpr, FOLD_ROW_LANES: gl.constexpr, FOLD_ORDER: gl.constexpr, INDEX_2D: gl.constexpr, ROTATE: gl.constexpr):
    gl.static_assert(M == 32 or M == 64)
    gl.static_assert(BN == 64 and N % BN == 0)
    gl.static_assert(K % (128 * CHAINS * SPLIT) == 0)
    if INDEX_2D:
        si, ni = (gl.program_id(0).to(gl.uint32), gl.program_id(1).to(gl.uint32))
        gl.assume(si < SPLIT)
        gl.assume(ni < N // BN)
    else:
        linear = gl.program_id(0)
        si, ni = (linear % SPLIT, linear // SPLIT)
        if M == 32:
            ni %= N // BN
    if ROTATE:
        si = (si + ni) % SPLIT
    row_waves: gl.constexpr = gl.num_warps() // CHAINS
    ml: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [CHAINS, row_waves, 1])
    bd: gl.constexpr = gl.DotOperandLayout(1, ml, 16)
    al: gl.constexpr = gl.BlockedLayout([1, 1, 16], [1, 16, 4], [CHAINS, row_waves, 1], [2, 1, 0])
    ga, ra, ka = _indices(CHAINS, M, 128, al)
    gb, kb, nb = _indices(CHAINS, 128, BN, bd)
    gs = gl.arange(0, CHAINS, gl.SliceLayout(1, gl.SliceLayout(2, ml)))
    ao = ra * SX + ka + ga * 128
    wo = nb // 16 * K * 16 + kb // 32 * 512
    wo += kb // 16 % 2 * 256 + nb % 16 * 16 + kb % 16 + gb * 2048
    W += ni * BN * K
    WS += ni * BN // 128 * (K // 128)
    offsets = (ao, wo, gs)
    ITERS: gl.constexpr = K // (128 * CHAINS * SPLIT)
    first = si * CHAINS if CYCLIC else si * CHAINS * ITERS
    step: gl.constexpr = CHAINS * SPLIT if CYCLIC else CHAINS
    zeros = gl.zeros((CHAINS, M, BN), gl.float32, ml)
    acc = zeros
    if PREFETCH:
        record = _load_record(X, W, WS, offsets, first, BUFFER_SCALES, PREFETCH)
    for it in gl.static_range(ITERS):
        if it + PREFETCH < ITERS:
            upcoming = _load_record(X, W, WS, offsets, first + (it + PREFETCH) * step, BUFFER_SCALES, PREFETCH)
        if not PREFETCH:
            record = upcoming
        acc += _apply_record(record, zeros)
        if PREFETCH and it + 1 < ITERS:
            record = upcoming
    fold: gl.constexpr = gl.BlockedLayout([CHAINS, 1, FOLD_VEC], [1, FOLD_ROW_LANES, 64 // FOLD_ROW_LANES], [1, gl.num_warps(), 1], FOLD_ORDER)
    merged = gl.sum(gl.convert_layout(acc, fold), 0)
    cl: gl.constexpr = merged.type.layout
    row = gl.arange(0, M, gl.SliceLayout(1, cl))
    col = ni * BN + gl.arange(0, BN, gl.SliceLayout(0, cl))
    gl.amd.cdna4.buffer_store(merged.to(P.dtype.element_ty), P + si * N, row[:, None] * (SPLIT * N) + col[None, :], cache='.wt')

@gluon.jit
def _finish_role(P, Gamma, Out, Rope, row, N: gl.constexpr, OFFSET: gl.constexpr, WIDTH: gl.constexpr, R: gl.constexpr, EPS: gl.constexpr, SPLIT: gl.constexpr, VEC: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [gl.num_warps()], [0])
    d = gl.arange(0, triton.next_power_of_2(WIDTH + R), layout)
    P += row * (SPLIT * N) + OFFSET
    x = gl.amd.cdna4.buffer_load(P, d, d < WIDTH + R, 0.0).to(gl.float32)
    for split in gl.static_range(1, SPLIT):
        x += gl.amd.cdna4.buffer_load(P + split * N, d, d < WIDTH + R, 0.0).to(gl.float32)
    x = x.to(gl.bfloat16).to(gl.float32)
    gamma = gl.load(Gamma + d, d < WIDTH, 0).to(gl.float32)
    xn = gl.where(d < WIDTH, x, 0.0)
    variance = gl.sum(xn * xn, 0) / WIDTH
    y = x * gl.rsqrt(variance + EPS) * gamma
    gl.store(Out + row * WIDTH + d, y.to(gl.bfloat16), d < WIDTH, cache_modifier='.wt')
    if R:
        gl.store(Rope + row * R + d - WIDTH, x.to(gl.bfloat16), (d >= WIDTH) & (d < WIDTH + R), cache_modifier='.wt')

@gluon.jit
def _finish(P, QG, KG, QO, KO, RO, N: gl.constexpr, Q: gl.constexpr, KV: gl.constexpr, R: gl.constexpr, EPS: gl.constexpr, SPLIT: gl.constexpr, Q_VEC: gl.constexpr, KV_VEC: gl.constexpr):
    row = gl.program_id(0)
    if gl.program_id(1) == 0:
        _finish_role(P, QG, QO, RO, row, N, 0, Q, 0, EPS, SPLIT, Q_VEC)
    else:
        _finish_role(P, KG, KO, RO, row, N, Q, KV, R, EPS, SPLIT, KV_VEC)

def mla_qkv_a_norm(hidden: torch.Tensor, weight: torch.Tensor, q_gamma: torch.Tensor, kv_gamma: torch.Tensor, weight_scales: torch.Tensor | None=None, *, rope_dim: int=64, eps: float=1e-05, weight_preshuffled: bool=False):
    m, k = hidden.shape
    q, kv = (q_gamma.numel(), kv_gamma.numel())
    n = q + kv + rope_dim
    assert weight_preshuffled and weight_scales is not None
    qo = torch.empty((m, q), device=hidden.device, dtype=torch.bfloat16)
    ko = torch.empty((m, kv), device=hidden.device, dtype=torch.bfloat16)
    ro = torch.empty((m, rope_dim), device=hidden.device, dtype=torch.bfloat16)
    p = torch.empty((m, _SPLITS, n), device=hidden.device, dtype=torch.float32)
    tuning = _TUNING[m]
    grid = (_SPLITS, n // _N_TILE) if tuning['INDEX_2D'] else (n // _N_TILE * _SPLITS,)
    _project[grid](hidden, weight, weight_scales, p, m, n, k, hidden.stride(0), _N_TILE, _SPLITS, **tuning, num_warps=8, enable_fp_fusion=True)
    _finish[m, 2](p, q_gamma, kv_gamma, qo, ko, ro, n, q, kv, rope_dim, eps, _SPLITS, **_FINISH_CONFIG)
    return (qo, ko, ro)
