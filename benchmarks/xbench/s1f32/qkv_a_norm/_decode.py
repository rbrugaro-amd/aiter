import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def _axes3(D0: gl.constexpr, D1: gl.constexpr, D2: gl.constexpr, LAYOUT: gl.constexpr):
    d0 = gl.arange(0, D0, gl.SliceLayout(1, gl.SliceLayout(2, LAYOUT)))
    d1 = gl.arange(0, D1, gl.SliceLayout(0, gl.SliceLayout(2, LAYOUT)))
    d2 = gl.arange(0, D2, gl.SliceLayout(0, gl.SliceLayout(1, LAYOUT)))
    return (d0, d1, d2)

@gluon.jit
def _quantize_block(x, DTYPE: gl.constexpr):
    axis: gl.constexpr = len(x.type.shape) - 1
    scale = gl.maximum(gl.max(gl.abs(x), axis), 1e-10) * (1.0 / 448.0)
    return ((x * gl.expand_dims(gl.div_rn(1.0, scale), axis)).to(DTYPE), scale)

@gluon.jit
def _packed_offset(k, n, K: gl.constexpr, group=0):
    return n // 16 * K * 16 + group * 2048 + k // 16 * 256 + n % 16 * 16 + k % 16

@gluon.jit
def _store_partial(value, P, tile, split, M: gl.constexpr, PITCH: gl.constexpr, BN: gl.constexpr, BUFFER: gl.constexpr=True, CACHE: gl.constexpr=None):
    layout: gl.constexpr = value.type.layout
    row = gl.arange(0, M, gl.SliceLayout(1, layout))
    col = gl.arange(0, BN, gl.SliceLayout(0, layout))
    if BUFFER:
        gl.amd.cdna4.buffer_store(value.to(P.dtype.element_ty), P + split * M * PITCH + tile * BN, row[:, None] * PITCH + col[None, :], cache=CACHE)
    else:
        gl.store(P + split * M * PITCH + row[:, None] * PITCH + tile * BN + col[None, :], value, cache_modifier=CACHE)

@gluon.jit
def _diagonal_two(dot, M: gl.constexpr, BN: gl.constexpr, WAVES: gl.constexpr):
    blocks = dot.reshape((WAVES, 2, M, 2, BN)).permute((0, 1, 2, 4, 3))
    a, b = gl.split(blocks.reshape((WAVES, 2, M, BN, 2)))
    group = gl.arange(0, 2, gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, a.type.layout))))
    return gl.where(group[None, :, None, None] == 0, a, b)

@gluon.jit
def _project_two(X, W, WS, P, M: gl.constexpr, PITCH: gl.constexpr, K: gl.constexpr, SX: gl.constexpr, BN: gl.constexpr, PACK: gl.constexpr, WAVES: gl.constexpr, AV: gl.constexpr, FN: gl.constexpr):
    gl.static_assert(M == 4 and PACK == 2 and (WAVES == 8) and (BN == 32) and (FN == 16))
    BM: gl.constexpr = M * PACK
    al: gl.constexpr = gl.BlockedLayout([1, 1, AV], [1, BM, 64 // BM], [WAVES, 1, 1], [2, 1, 0])
    ml: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [WAVES, 1, 1])
    bl: gl.constexpr = gl.BlockedLayout([1, 16, 1], [1, 4, 16], [WAVES, 1, 1], [2, 1, 0])
    rl: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 2, 32], [1, WAVES // 2, 2], [2, 1, 0])
    ga, ra, ka = _axes3(WAVES, BM, 128, al)
    gb, kb, nb = _axes3(WAVES, 128, PACK * FN, bl)
    gc = gl.arange(0, WAVES, gl.SliceLayout(1, gl.SliceLayout(2, ml)))
    rc = gl.arange(0, BM, gl.SliceLayout(0, gl.SliceLayout(2, ml)))
    tile, split = (gl.program_id(0).to(gl.uint32), gl.program_id(1).to(gl.uint32))
    first = split * WAVES * PACK
    ag = ga[:, None, None] * PACK + ra[None, :, None] // M
    x = gl.amd.cdna4.buffer_load(X + first * 128, ra[None, :, None] % M * SX + ag * 128 + ka[None, None, :]).to(gl.float32)
    a, sa = _quantize_block(x, W.dtype.element_ty)
    a = gl.convert_layout(a, gl.DotOperandLayout(0, ml, 16))
    sa = gl.convert_layout(sa, gl.SliceLayout(2, ml))
    sw = gl.amd.cdna4.buffer_load(WS + tile * BN // 128 * (K // 128) + first, gc[:, None] * PACK + rc[None, :] // M)
    factor = sa * sw
    shared = gl.allocate_shared_memory(gl.float32, (WAVES, M, BN), gl.SwizzledSharedLayout(4, 2, 4, [2, 1, 0]))
    for frag in gl.static_range(BN // FN):
        bk = (gb[:, None, None] * PACK + nb[None, None, :] // FN) * 128 + kb[None, :, None]
        wo = _packed_offset(bk, nb[None, None, :] % FN, K)
        b = gl.convert_layout(gl.amd.cdna4.buffer_load(W + (tile * BN + frag * FN) * K + first * 2048, wo), gl.DotOperandLayout(1, ml, 16))
        dot = gl.amd.cdna4.mfma(a, b, gl.full((WAVES, BM, PACK * FN), 0, gl.float32, ml))
        dot = _diagonal_two(dot, M, FN, WAVES).reshape((WAVES, BM, FN))
        factors = gl.convert_layout(factor, gl.SliceLayout(2, dot.type.layout))
        value = gl.sum((dot * factors[:, :, None]).reshape((WAVES, PACK, M, FN)), 1)
        shared.slice(frag * FN, FN, 2).store(value)
    value = gl.sum(shared.load(rl), 0)
    _store_partial(value, P, tile, split, M, PITCH, BN, CACHE='.wt')

@triton.constexpr_function
def _packed_shared(outer, panel):
    return gl.SharedLinearLayout([[0, bit] for bit in (1, 2, 4, 8)] + [[1 << bit, 0] for bit in range(outer.bit_length() - 1)] + [[0, 1 << bit] for bit in range(4, panel.bit_length() - 1)])

@gluon.jit
def _tile_ids(M: gl.constexpr, N: gl.constexpr, BM: gl.constexpr, SPLITS: gl.constexpr, STRIPES: gl.constexpr):
    total: gl.constexpr = M // BM * (N // 64) * SPLITS
    pid = gl.program_id(0).to(gl.uint32)
    pid = pid % STRIPES * (total // STRIPES) + gl.minimum(pid % STRIPES, total % STRIPES) + pid // STRIPES
    return (pid // (N // 64) % (M // BM), pid % (N // 64), pid // (M // BM * (N // 64)))

@gluon.jit
def _enqueue(amem, bmem, abase, wbase, ao, bo, GROUP: gl.constexpr, STAGES: gl.constexpr, PANEL: gl.constexpr):
    slot: gl.constexpr = GROUP % STAGES
    gl.amd.cdna4.async_copy.buffer_load_to_shared(amem.index(slot), abase + GROUP * (16 * PANEL), ao)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(bmem.index(slot), wbase + GROUP * (16 * PANEL), bo)
    gl.amd.cdna4.async_copy.commit_group()

@gluon.jit
def _read_panel(amem, bmem, abase, wbase, ao, bo, INDEX: gl.constexpr, PANELS: gl.constexpr, STAGES: gl.constexpr, MMA: gl.constexpr, PANEL: gl.constexpr):
    gl.amd.cdna4.async_copy.wait_group(min(STAGES - 1, PANELS - INDEX - 1))
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(amem.index(INDEX % STAGES), gl.DotOperandLayout(0, MMA, 16))
    b = gl.amd.cdna4.async_copy.load_shared_relaxed(bmem.index(INDEX % STAGES).permute((1, 0)), gl.DotOperandLayout(1, MMA, 16))
    gl.barrier()
    if INDEX + STAGES < PANELS:
        _enqueue(amem, bmem, abase, wbase, ao, bo, INDEX + STAGES, STAGES, PANEL)
    return (a, b)

@gluon.jit
def _store_split(acc, P, pm, pn, part, M: gl.constexpr, N: gl.constexpr, BM: gl.constexpr, VEC: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, VEC], [VEC, 64 // VEC], [4, 1], [1, 0])
    value = gl.convert_layout(acc.to(P.dtype.element_ty), layout)
    r = pm * BM + gl.arange(0, BM, gl.SliceLayout(1, layout))
    c = pn * 64 + gl.arange(0, 64, gl.SliceLayout(0, layout))
    gl.amd.cdna4.buffer_store(value, P + part * M * N, r[:, None] * N + c[None, :], cache='.wt')

@gluon.jit
def _merge(P, offsets, valid, SPLITS: gl.constexpr, PLANE: gl.constexpr):
    x = gl.amd.cdna4.buffer_load(P, offsets, valid, 0.0, cache='.cg').to(gl.float32)
    for part in gl.static_range(1, SPLITS):
        x += gl.amd.cdna4.buffer_load(P + part * PLANE, offsets, valid, 0.0, cache='.cg').to(gl.float32)
    return x.to(gl.bfloat16).to(gl.float32)

@gluon.jit
def _project_stream_exchange(X, W, WS, P, M: gl.constexpr, PITCH: gl.constexpr, K: gl.constexpr, SX: gl.constexpr, BN: gl.constexpr, PACK: gl.constexpr, WAVES: gl.constexpr, AV: gl.constexpr, FN: gl.constexpr, BF16_EXCHANGE: gl.constexpr=False):
    gl.static_assert(PACK == 1 and (M == 8 or M == 16))
    gl.static_assert(FN % 16 == 0 and BN % FN == 0 and (BN <= 128))
    al: gl.constexpr = gl.BlockedLayout([1, 1, AV], [1, M, 64 // M], [WAVES, 1, 1], [2, 1, 0])
    ml: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [WAVES, 1, 1])
    if M == 8:
        bl: gl.constexpr = gl.DotOperandLayout(1, ml, 16)
        rl: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 2, 32], [1, WAVES // 2, 2], [2, 1, 0])
    else:
        bl: gl.constexpr = gl.BlockedLayout([1, 16, 1], [1, 4, 16], [WAVES, 1, 1], [2, 1, 0])
        rl: gl.constexpr = gl.BlockedLayout([1, 1, 2], [1, 4, 16], [1, WAVES // 2, 2], [2, 1, 0])
    ga, ra, ka = _axes3(WAVES, M, 128, al)
    gb, kb, nb = _axes3(WAVES, 128, FN, bl)
    gc = gl.arange(0, WAVES, gl.SliceLayout(1, gl.SliceLayout(2, ml)))
    tile, split = (gl.program_id(0), gl.program_id(1))
    if M == 16:
        tile, split = (tile.to(gl.uint32), split.to(gl.uint32))
    first = split * WAVES
    sw = gl.amd.cdna4.buffer_load(WS + tile * BN // 128 * (K // 128) + first, gc)[:, None]
    if M == 16:
        xo = ra[None, :, None] * SX + ga[:, None, None] * 128 + ka[None, None, :]
        x = gl.amd.cdna4.buffer_load(X + first * 128, xo).to(gl.float32)
    else:
        xo = ra[None, :, None] * SX + (first + ga[:, None, None]) * 128 + ka[None, None, :]
        x = gl.amd.cdna4.buffer_load(X, xo).to(gl.float32)
    a, sa = _quantize_block(x, W.dtype.element_ty)
    a = gl.convert_layout(a, gl.DotOperandLayout(0, ml, 16))
    sa = gl.convert_layout(sa, gl.SliceLayout(2, ml))
    factor = sa * sw
    exchange_dtype: gl.constexpr = gl.bfloat16 if BF16_EXCHANGE else gl.float32
    shared = gl.allocate_shared_memory(exchange_dtype, (WAVES, M, BN), gl.SwizzledSharedLayout(8 if BF16_EXCHANGE else 4, 1, 8, [2, 1, 0]))
    for frag in gl.static_range(BN // FN):
        if M == 16:
            bk = gb[:, None, None] * 128 + kb[None, :, None]
            bn = nb[None, None, :]
            wo = bn // 16 * K * 16 + bk // 32 * 512 + bk // 16 % 2 * 256 + bn % 16 * 16 + bk % 16
        else:
            wo = _packed_offset(kb[None, :, None], nb[None, None, :], K, gb[:, None, None])
        b = gl.convert_layout(gl.amd.cdna4.buffer_load(W + (tile * BN + frag * FN) * K + first * 2048, wo), gl.DotOperandLayout(1, ml, 16))
        dot = gl.amd.cdna4.mfma(a, b, gl.full((WAVES, M, FN), 0, gl.float32, ml))
        shared.slice(frag * FN, FN, 2).store((dot * factor[:, :, None]).to(exchange_dtype))
    value = gl.sum(shared.load(rl).to(gl.float32), 0)
    _store_partial(value, P, tile, split, M, PITCH, BN, BUFFER=M != 8, CACHE='.wt')
