from typing import NamedTuple
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._core import builtin
from triton.experimental.gluon.language.amd.cdna4 import async_copy
_SCALE_LAYOUT = gl.BlockedLayout([1], [64], [4], [0])

@builtin
def _mfma_unscaled(a, b, acc, _semantic=None):
    result = _semantic.dot_scaled(a, None, 'e4m3', b, None, 'e4m3', acc, fast_math=False, lhs_k_pack=True, rhs_k_pack=True, out_dtype=gl.float32)
    return gl.tensor(result.handle, acc.type)

@gluon.constexpr_function
def _packed_lds_layout(rows, transpose=False):
    bases = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2], [0, 4], [0, 8], [16, 0], [32, 0], [64, 0]]
    step = 16
    while step < rows:
        bases.append([0, step])
        step *= 2
    if transpose:
        bases = [[b[1], b[0]] for b in bases]
    return gl.SharedLinearLayout(bases)

class _ProjectionPlan(NamedTuple):
    tile_m: int = 256
    phase_m: int = 128
    rope_tile_m: int = 0
    group_m: int = 64
    store_rows: int = 1
    store_lanes_m: int = 4
    select_output_early: bool = False
    fence_pack: int = 32
    early_scale: bool = False
    scale_first: bool = False
    initial_copy_early: bool = False
    scale_vec: int = 1
    unsigned_pid: bool = False
    scale_after_a: bool = False

class _NormPlan(NamedTuple):
    q_rows: int
    kv_rows: int
    warps: int = 4
    load_cache: str = ''
    store_cache: str = ''
    kv_lanes: int = 64
    full_tiles: bool = False
    q_vec: int = 8
    packed_input: bool = False

class _ProjectionShape(NamedTuple):
    rows: int
    padded_rows: int
    hidden: int
    q: int
    kv: int
    rope: int
_SCHEDULES = {8193: (_ProjectionPlan(tile_m=128, phase_m=64, rope_tile_m=64, store_lanes_m=8, fence_pack=1, early_scale=True, scale_first=True, initial_copy_early=True, scale_after_a=True), _NormPlan(16, 32, kv_lanes=16)), 10240: (_ProjectionPlan(store_rows=4, select_output_early=True, fence_pack=0, scale_first=True, scale_vec=4, unsigned_pid=True, scale_after_a=True), _NormPlan(2, 4, 2, '.cg', '.cs', full_tiles=True, q_vec=4, packed_input=True)), 12288: (_ProjectionPlan(store_rows=4, select_output_early=True, fence_pack=0, initial_copy_early=True, scale_vec=4, unsigned_pid=True), _NormPlan(2, 2, 2, '.cg', '.cs', full_tiles=True)), 16384: (_ProjectionPlan(rope_tile_m=128, group_m=4, select_output_early=True, fence_pack=0, scale_first=True, initial_copy_early=True, scale_vec=4, scale_after_a=True), _NormPlan(4, 4, 1, '.cg', '.cs', full_tiles=True))}

def _schedule(m):
    if m in _SCHEDULES:
        return _SCHEDULES[m]
    medium, large = (m >= 10000, m >= 15000)
    projection = _ProjectionPlan(tile_m=256 if medium else 128, store_rows=4 if medium and (not large) else 1, select_output_early=large)
    norm = _NormPlan(16, 8 if medium else 32, load_cache='.cg' if medium else '', store_cache='.cs' if m >= 11000 else '')
    return (projection, norm)

@gluon.jit
def _map_tiles(pid, nm: gl.constexpr, nn: gl.constexpr, GM: gl.constexpr):
    if nm <= GM:
        return (pid % nm, pid // nm)
    group = pid // (GM * nn)
    first = group * GM
    within = pid % (GM * nn)
    if nm % GM == 0:
        pm, pn = (first + within % GM, within // GM)
    elif group == nm // GM:
        pm, pn = (first + within % (nm % GM), within // (nm % GM))
    else:
        pm, pn = (first + within % GM, within // GM)
    return (pm, pn)

@gluon.jit
def _packed_offset(row, k, K: gl.constexpr):
    return row // 16 * (K * 16) + k // 16 * 256 + row % 16 * 16 + k % 16

@gluon.jit
def _quantize(X, A, S, M: gl.constexpr, MP: gl.constexpr, K: gl.constexpr, SX: gl.constexpr, BR: gl.constexpr=16, W: gl.constexpr=2, BUFFER: gl.constexpr=False, GR: gl.constexpr=1):
    L: gl.constexpr = gl.BlockedLayout([1, 16], [8, 8], [W, 1], [1, 0])
    ng: gl.constexpr = K // 128
    pid = gl.program_id(0)
    if GR == 1:
        pm, g = (pid // ng, pid % ng)
    else:
        pm, g = _map_tiles(pid, MP // BR, ng, GR)
    r = pm * BR + gl.arange(0, BR, gl.SliceLayout(1, L))
    d = gl.arange(0, 128, gl.SliceLayout(0, L))
    valid = gl.full((BR, 1), True, gl.int1, L) if M == MP else r[:, None] < M
    if BUFFER:
        x = gl.amd.cdna4.buffer_load(X, r[:, None] * SX + g * 128 + d[None, :], valid, other=0.0).to(gl.float32)
    else:
        x = gl.load(X + r[:, None] * SX + g * 128 + d[None, :], valid, 0.0).to(gl.float32)
    scale = gl.maximum(gl.max(gl.abs(x), 1), 1e-10) * (1.0 / 448.0)
    a = (x * gl.div_rn(1.0, scale)[:, None]).to(gl.float8e4nv)
    off = _packed_offset(r[:, None], d[None, :], K) + g * 2048
    if BUFFER:
        gl.amd.cdna4.buffer_store(a, A, off)
        gl.amd.cdna4.buffer_store(scale, S, g * MP + r)
    else:
        gl.store(A + off, a)
        gl.store(S + g * MP + r, scale)

@gluon.jit
def _scale_plane(sd, g):
    return sd.index(g % sd.shape[0])

@gluon.jit
def _phase_scale(sd, g, ROWS: gl.constexpr, ML: gl.constexpr, OFFSET: gl.constexpr=0):
    if sd is None:
        return None
    else:
        return async_copy.load_shared_relaxed(_scale_plane(sd, g).slice(OFFSET, ROWS), gl.SliceLayout(1, ML))

@gluon.jit
def _copy_scales(AS, sd, g, MP: gl.constexpr, BM: gl.constexpr, FULL_AS: gl.constexpr, V: gl.constexpr):
    if sd is not None:
        SL: gl.constexpr = gl.BlockedLayout([V], [64], [4], [0])
        so = gl.arange(0, 256, SL)
        async_copy.buffer_load_to_shared(_scale_plane(sd, g), AS + g * MP, so, (so < BM) | FULL_AS, other=0.0)

@gluon.jit
def _stream_copy(A, W, AS, ad, bd, sd, ao, bo, g, MP: gl.constexpr, BM: gl.constexpr, FULL_AS: gl.constexpr, SCALE_FIRST: gl.constexpr, V: gl.constexpr, AFTER_A: gl.constexpr):
    if SCALE_FIRST and (not AFTER_A):
        _copy_scales(AS, sd, g, MP, BM, FULL_AS, V)
    if ad is not None:
        async_copy.buffer_load_to_shared(ad, A + g * 2048, ao)
    if SCALE_FIRST and AFTER_A:
        _copy_scales(AS, sd, g, MP, BM, FULL_AS, V)
    async_copy.buffer_load_to_shared(bd, W + g * 2048, bo)
    if not SCALE_FIRST:
        _copy_scales(AS, sd, g, MP, BM, FULL_AS, V)
    async_copy.commit_group()

@gluon.jit
def _output_tile(P, KO, RO, row_base, pn, Q: gl.constexpr, KV: gl.constexpr, RD: gl.constexpr, BN: gl.constexpr):
    CL: gl.constexpr = _SCALE_LAYOUT
    if BN == 64:
        O = RO + row_base * RD
        stride = gl.full((), RD, gl.int32, CL)
    elif pn < Q // 128:
        O = P + row_base * Q + pn * 128
        stride = gl.full((), Q, gl.int32, CL)
    else:
        O = KO + row_base * KV + pn * 128 - Q
        stride = gl.full((), KV, gl.int32, CL)
    return (O, stride)

@gluon.jit
def _store_phase(value, O, stride, row_base, M: gl.constexpr, FULL: gl.constexpr, ROW_OFFSET: gl.constexpr=0):
    LO: gl.constexpr = value.type.layout
    r = gl.arange(0, value.shape[0], gl.SliceLayout(1, LO))
    c = gl.arange(0, value.shape[1], gl.SliceLayout(0, LO))
    valid = gl.full((value.shape[0], 1), True, gl.int1, LO) if FULL else row_base + r[:, None] + ROW_OFFSET < M
    gl.amd.cdna4.buffer_store(value, O, (r[:, None] + ROW_OFFSET) * stride + c[None, :], valid)

@gluon.constexpr_function
def _fence_constraints(pack):
    return ','.join(['=v'] * pack + [str(i) for i in range(pack)] + ['~{memory}'])

@gluon.jit
def _schedule_fence(value, PACK: gl.constexpr):
    return gl.inline_asm_elementwise('', constraints=_fence_constraints(PACK), args=[value], dtype=gl.float32, is_pure=False, pack=PACK)

@gluon.jit
def _weight_scale(wd, g):
    wi = gl.full((1,), g, gl.int32, _SCALE_LAYOUT)
    return gl.sum(wd.gather(wi, 0), 0)

@gluon.jit
def _accumulate(acc, a, b, row_scale, weight_scale):
    part = _mfma_unscaled(a, b, gl.zeros(acc.shape, gl.float32, acc.type.layout))
    if row_scale is None:
        return acc + part * weight_scale
    else:
        return acc + part * (row_scale * weight_scale)[:, None]

@gluon.jit
def _stream_tile(inputs, outputs, SHAPE: gl.constexpr, row_base, pn, BM: gl.constexpr, BN: gl.constexpr, PLAN: gl.constexpr, FULL: gl.constexpr, REPLICATE: gl.constexpr=False):
    A, W, AS, WS = inputs
    P, KO, RO = outputs
    M, MP, K, Q, KV, RD = SHAPE
    LA: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [4, 1], [0, 1])
    LB: gl.constexpr = gl.BlockedLayout([16, 1], [4, 16], [1, 4], [1, 0])
    TM: gl.constexpr = min(BM, PLAN.phase_m)
    SPLIT: gl.constexpr = BM == 2 * TM
    gl.static_assert(BM == TM or SPLIT)
    gl.static_assert(not REPLICATE or BM == 16)
    ML: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 4] if REPLICATE else [4, 1] if BM == 256 else [2, 2])
    DA: gl.constexpr = gl.DotOperandLayout(0, ML, 32)
    DB: gl.constexpr = gl.DotOperandLayout(1, ML, 32)
    A = A + row_base * K
    W = W + pn * 128 * K
    AS = AS + row_base
    if PLAN.select_output_early:
        O, stride = _output_tile(P, KO, RO, row_base, pn, Q, KV, RD, BN)
    else:
        O, stride = (None, None)
    ar = gl.full((BM, 1), 0, gl.int32, LA) if REPLICATE else gl.arange(0, BM, gl.SliceLayout(1, LA))[:, None]
    ak = gl.arange(0, 128, gl.SliceLayout(0, LA))
    bc = gl.arange(0, BN, gl.SliceLayout(0, LB))
    bk = gl.arange(0, 128, gl.SliceLayout(1, LB))
    ao = _packed_offset(ar, ak[None, :], K)
    bo = _packed_offset(bc[None, :], bk[:, None], K)
    ao = gl.max_contiguous(gl.multiple_of(ao, [1, 16]), [1, 16])
    bo = gl.max_contiguous(gl.multiple_of(bo, [16, 1]), [16, 1])
    ad, sd = (None, None)
    if not REPLICATE:
        ad = gl.allocate_shared_memory(gl.uint8, [BM, 128], _packed_lds_layout(BM, True))
    bd = gl.allocate_shared_memory(gl.uint8, [128, BN], _packed_lds_layout(BN))
    wg = gl.arange(0, 64, _SCALE_LAYOUT)
    if REPLICATE:
        sa = gl.load(AS + wg * MP, wg < K // 128, 0.0)
    else:
        sd = gl.allocate_shared_memory(gl.float32, [BM // TM, 256], gl.SwizzledSharedLayout(1, 1, 1, [0]))
    wv = gl.load(WS + pn * (K // 128) + wg, wg < K // 128, 0.0)
    if REPLICATE:
        wv = sa * wv
    FULL_AS: gl.constexpr = M == 8193 and BM == 128 and SPLIT
    if PLAN.initial_copy_early:
        _stream_copy(A, W, AS, ad, bd, sd, ao, bo, 0, MP, BM, FULL_AS, PLAN.scale_first, PLAN.scale_vec, PLAN.scale_after_a)
    wd = gl.allocate_shared_memory(gl.float32, [64], gl.SwizzledSharedLayout(1, 1, 1, [0]), wv)
    gl.barrier()
    if not PLAN.initial_copy_early:
        _stream_copy(A, W, AS, ad, bd, sd, ao, bo, 0, MP, BM, FULL_AS, PLAN.scale_first, PLAN.scale_vec, PLAN.scale_after_a)
    acc0 = gl.zeros((TM, BN), gl.float32, ML)
    if SPLIT:
        acc1 = gl.zeros((TM, BN), gl.float32, ML)
    for g in range(K // 128):
        if PLAN.early_scale and (not REPLICATE):
            ws = _weight_scale(wd, g)
        if REPLICATE:
            a0 = gl.convert_layout(gl.amd.cdna4.buffer_load(A + g * 2048, ao), DA)
        async_copy.wait_group(0)
        if SPLIT:
            b = async_copy.load_shared_relaxed(bd, DB)
        if not REPLICATE:
            a0 = async_copy.load_shared_relaxed(ad.slice(0, TM, 0), DA)
        if not SPLIT:
            b = async_copy.load_shared_relaxed(bd, DB)
        s0 = _phase_scale(sd, g, TM, ML)
        if not PLAN.early_scale or REPLICATE:
            ws = _weight_scale(wd, g)
        if SPLIT:
            a1 = async_copy.load_shared_relaxed(ad.slice(TM, TM, 0), DA)
        gl.barrier()
        if g + 1 < K // 128:
            _stream_copy(A, W, AS, ad, bd, sd, ao, bo, g + 1, MP, BM, FULL_AS, PLAN.scale_first, PLAN.scale_vec, PLAN.scale_after_a)
        acc0 = _accumulate(acc0, a0, b, s0, ws)
        if SPLIT:
            if PLAN.fence_pack == 0:
                ws = _schedule_fence(ws, 1)
            else:
                acc0 = _schedule_fence(acc0, PLAN.fence_pack)
            s1 = _phase_scale(sd, g, TM, ML, OFFSET=TM)
            acc1 = _accumulate(acc1, a1, b, s1, ws)
    LO: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0]) if REPLICATE else gl.BlockedLayout([PLAN.store_rows, 8], [PLAN.store_lanes_m, 64 // PLAN.store_lanes_m], [4, 1], [1, 0])
    value0 = gl.convert_layout(acc0.to(gl.bfloat16), LO)
    if O is None:
        O, stride = _output_tile(P, KO, RO, row_base, pn, Q, KV, RD, BN)
    _store_phase(value0, O, stride, row_base, M, FULL)
    if SPLIT:
        value1 = gl.convert_layout(acc1.to(gl.bfloat16), LO)
        _store_phase(value1, O, stride, row_base, M, FULL, ROW_OFFSET=TM)

@gluon.jit
def _dispatch_tile(inputs, outputs, SHAPE: gl.constexpr, pm, pn, BM: gl.constexpr, BN: gl.constexpr, PLAN: gl.constexpr):
    M: gl.constexpr = SHAPE.rows
    row_base = pm * BM
    TAIL: gl.constexpr = M % BM
    SHORT: gl.constexpr = TAIL > 0 and TAIL <= 32
    if SHORT and pm == M // BM:
        BT: gl.constexpr = 16 if TAIL <= 16 else 32
        _stream_tile(inputs, outputs, SHAPE, row_base, pn, BT, BN, PLAN, False, REPLICATE=PLAN.rope_tile_m != 0 and TAIL == 1)
    else:
        _stream_tile(inputs, outputs, SHAPE, row_base, pn, BM, BN, PLAN, TAIL == 0 or SHORT)

@gluon.jit
def _project(inputs, outputs, SHAPE: gl.constexpr, PLAN: gl.constexpr):
    pid = gl.program_id(0).to(gl.uint32) if PLAN.unsigned_pid else gl.program_id(0)
    BM: gl.constexpr = PLAN.tile_m
    RB: gl.constexpr = PLAN.rope_tile_m
    GM: gl.constexpr = PLAN.group_m
    nm: gl.constexpr = gl.cdiv(SHAPE.rows, BM)
    nn: gl.constexpr = (SHAPE.q + SHAPE.kv) // 128
    if RB:
        if pid < nm * nn:
            pm, pn = _map_tiles(pid, nm, nn, GM)
        else:
            pm, pn = (pid - nm * nn, nn)
    else:
        pm, pn = _map_tiles(pid, nm, nn + 1, GM)
    if pn == nn:
        _dispatch_tile(inputs, outputs, SHAPE, pm, pn, RB or BM, 64, PLAN)
    else:
        _dispatch_tile(inputs, outputs, SHAPE, pm, pn, BM, 128, PLAN)

@gluon.jit
def _norm_rows(Y, G, pid, M: gl.constexpr, WIDTH: gl.constexpr, EPS: gl.constexpr, BR: gl.constexpr, PLAN: gl.constexpr, LANES: gl.constexpr=64, V: gl.constexpr=8):
    L: gl.constexpr = gl.BlockedLayout([1, V], [64 // LANES, LANES], [PLAN.warps, 1], [1, 0])
    r = pid * BR + gl.arange(0, BR, gl.SliceLayout(1, L))
    d = gl.arange(0, WIDTH, gl.SliceLayout(0, L))
    valid = gl.full((BR, 1), True, gl.int1, L) if PLAN.full_tiles and M % BR == 0 else r[:, None] < M
    x16 = gl.amd.cdna4.buffer_load(Y, r[:, None] * WIDTH + d[None, :], valid, other=0.0, cache=PLAN.load_cache)
    if PLAN.packed_input:
        gamma = gl.load(G + d).to(gl.float32)
        x = x16.to(gl.float32)
    else:
        x = x16.to(gl.float32)
        gamma = gl.load(G + d).to(gl.float32)
    inv = gl.rsqrt(gl.sum(x * x, 1) / WIDTH + EPS)
    gl.amd.cdna4.buffer_store((x * inv[:, None] * gamma[None, :]).to(gl.bfloat16), Y, r[:, None] * WIDTH + d[None, :], valid, cache=PLAN.store_cache)

@gluon.jit
def _norm_roles(QO, KO, QG, KG, M: gl.constexpr, Q: gl.constexpr, KV: gl.constexpr, EPS: gl.constexpr, PLAN: gl.constexpr):
    pid = gl.program_id(0)
    QR: gl.constexpr = PLAN.q_rows
    KR: gl.constexpr = PLAN.kv_rows
    if pid < gl.cdiv(M, QR):
        _norm_rows(QO, QG, pid, M, Q, EPS, QR, PLAN, V=PLAN.q_vec)
    else:
        _norm_rows(KO, KG, pid - gl.cdiv(M, QR), M, KV, EPS, KR, PLAN, PLAN.kv_lanes)

def _run_projection_norm(hidden, weight, q_gamma, kv_gamma, weight_scales, rope_dim, eps, plan, norm):
    m, k = hidden.shape
    q, kv = (q_gamma.numel(), kv_gamma.numel())
    bm = plan.tile_m
    mp = triton.cdiv(m, bm) * bm
    a = torch.empty((mp, k), device=hidden.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((k // 128, mp), device=hidden.device, dtype=torch.float32)
    qo = torch.empty((m, q), device=hidden.device, dtype=torch.bfloat16)
    ko = torch.empty((m, kv), device=hidden.device, dtype=torch.bfloat16)
    ro = torch.empty((m, rope_dim), device=hidden.device, dtype=torch.bfloat16)
    if m == 16384:
        qr, qw, buffer, gr = (8, 1, False, 32)
    elif m == 10240:
        qr, qw, buffer, gr = (8, 1, True, 1)
    else:
        qr, qw, buffer, gr = (16, 2, False, 1)
    _quantize[mp // qr * (k // 128),](hidden, a, scales, m, mp, k, hidden.stride(0), qr, qw, buffer, gr, num_warps=qw)
    dense_tiles = triton.cdiv(m, bm) * ((q + kv) // 128)
    rope_tiles = triton.cdiv(m, plan.rope_tile_m or bm)
    inputs = (a.view(torch.uint8), weight.view(torch.uint8), scales, weight_scales)
    outputs = (qo, ko, ro)
    shape = _ProjectionShape(m, mp, k, q, kv, rope_dim)
    _project[dense_tiles + rope_tiles,](inputs, outputs, shape, plan, num_warps=4, enable_fp_fusion=True)
    _norm_roles[triton.cdiv(m, norm.q_rows) + triton.cdiv(m, norm.kv_rows),](qo, ko, q_gamma, kv_gamma, m, q, kv, eps, norm, num_warps=norm.warps, enable_fp_fusion=False)
    return outputs

def mla_qkv_a_norm(hidden: torch.Tensor, weight: torch.Tensor, q_gamma: torch.Tensor, kv_gamma: torch.Tensor, weight_scales: torch.Tensor | None=None, *, rope_dim: int=64, eps: float=1e-05, weight_preshuffled: bool=False):
    plan, norm = _schedule(hidden.shape[0])
    return _run_projection_norm(hidden, weight, q_gamma, kv_gamma, weight_scales, rope_dim, eps, plan, norm)
