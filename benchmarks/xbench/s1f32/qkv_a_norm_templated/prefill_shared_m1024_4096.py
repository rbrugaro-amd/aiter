from typing import NamedTuple
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout
from triton.experimental.gluon.language.amd.cdna4 import async_copy, buffer_load, buffer_store, mfma, mfma_scaled
_MMA = AMDMFMALayout(4, [16, 16, 128], True, [2, 2])

@gluon.jit
def _packed_offset(row, channel, K: gl.constexpr):
    return row // 16 * K * 16 + channel // 32 * 512 + channel // 16 % 2 * 256 + row % 16 * 16 + channel % 16

@gluon.constexpr_function
def _packed_shared(outer, transpose, interval=1024, padding=32):
    bases = [[0, 1 << i] for i in range(4)]
    bases += [[1 << i, 0] for i in range(4)]
    bases += [[0, 1 << i] for i in range(4, 7)]
    bases += [[1 << i, 0] for i in range(4, outer.bit_length() - 1)]
    if transpose:
        bases = [[y, x] for x, y in bases]
    return gl.PaddedSharedLayout([(interval, padding)], bases, [], [128, outer] if transpose else [outer, 128])

@gluon.constexpr_function
def _copy_layout(shared, shape):
    bases = shared.offset_bases
    return gl.DistributedLinearLayout(bases[:4] + bases[12:], bases[4:10], bases[10:12], [], shape)

@gluon.jit
def _quantize(X, A, AS, M: gl.constexpr, MP: gl.constexpr, K: gl.constexpr, SX: gl.constexpr, ROWS: gl.constexpr, VEC: gl.constexpr, CACHE: gl.constexpr, REBASE: gl.constexpr, ROW_FIRST: gl.constexpr, GROUP_MAJOR: gl.constexpr=False):
    layout: gl.constexpr = gl.BlockedLayout([1, VEC], [VEC, 64 // VEC], [gl.num_warps(), 1], [1, 0])
    group = gl.program_id(1 if ROW_FIRST else 0)
    base = gl.program_id(0 if ROW_FIRST else 1) * ROWS
    row = gl.arange(0, ROWS, gl.SliceLayout(1, layout))
    channel = gl.arange(0, 128, gl.SliceLayout(0, layout))
    mask = None if M == MP else base + row[:, None] < M
    if REBASE:
        x = buffer_load(X + base * SX + group * 128, row[:, None] * SX + channel[None, :], mask, None if M == MP else 0.0).to(gl.float32)
    else:
        x = buffer_load(X, (base + row[:, None]) * SX + group * 128 + channel[None, :], mask, None if M == MP else 0.0).to(gl.float32)
    scale = gl.maximum(gl.max(gl.abs(x), 1), 1e-10) * (1.0 / 448.0)
    encoded = (x * gl.div_rn(1.0, scale)[:, None]).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
    if GROUP_MAJOR:
        off = _packed_offset(row[:, None], channel[None, :], 128)
        buffer_store(encoded, A + base * 128 + group * MP * 128, off, cache=CACHE)
        buffer_store(scale, AS + group * MP + base, row)
    elif REBASE:
        off = _packed_offset(row[:, None], channel[None, :], K)
        buffer_store(encoded, A + base * K + group * 2048, off, cache=CACHE)
        buffer_store(scale, AS + group * MP + base, row)
    else:
        off = _packed_offset(base + row[:, None], channel[None, :], K) + group * 2048
        buffer_store(encoded, A, off, cache=CACHE)
        gl.store(AS + group * MP + base + row, scale)

@gluon.jit
def _quantize_multi(X, A, AS, M: gl.constexpr, MP: gl.constexpr, K: gl.constexpr, SX: gl.constexpr, ROWS: gl.constexpr, VEC: gl.constexpr, CACHE: gl.constexpr, GROUPS: gl.constexpr, GROUP_MAJOR: gl.constexpr=False, ROW_FIRST: gl.constexpr=False, SCALE_CACHE: gl.constexpr=''):
    layout: gl.constexpr = gl.BlockedLayout([1, 1, VEC], [VEC, 1, 64 // VEC], [gl.num_warps(), 1, 1], [2, 0, 1])
    base_group = gl.program_id(1 if ROW_FIRST else 0) * GROUPS
    base = gl.program_id(0 if ROW_FIRST else 1) * ROWS
    row = gl.arange(0, ROWS, gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    group = gl.arange(0, GROUPS, gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    channel = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    offsets = row[:, None, None] * SX + group[None, :, None] * 128 + channel[None, None, :]
    mask = None if M == MP else base + row[:, None, None] < M
    x = buffer_load(X + base * SX + base_group * 128, offsets, mask, None if M == MP else 0.0).to(gl.float32)
    scale = gl.maximum(gl.max(gl.abs(x), 2), 1e-10) * (1.0 / 448.0)
    encoded = (x * gl.div_rn(1.0, scale)[:, :, None]).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
    if GROUP_MAJOR:
        packed = _packed_offset(row[:, None, None], channel[None, None, :], 128)
        buffer_store(encoded, A + base * 128 + base_group * MP * 128, packed + group[None, :, None] * MP * 128, cache=CACHE)
    else:
        packed = _packed_offset(row[:, None, None], channel[None, None, :], K)
        buffer_store(encoded, A + base * K + base_group * 2048, packed + group[None, :, None] * 2048, cache=CACHE)
    buffer_store(scale, AS + base_group * MP + base, group[None, :] * MP + row[:, None], cache=SCALE_CACHE)

@gluon.jit
def _scale_factors(AS, WS, rm, rn, start, MP: gl.constexpr, NG: gl.constexpr, BM: gl.constexpr, SG: gl.constexpr, BUFFER: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    group = start + gl.arange(0, SG, gl.SliceLayout(1, layout))
    row = rm + gl.arange(0, BM, gl.SliceLayout(0, layout))
    if BUFFER:
        sa = buffer_load(AS, group[:, None] * MP + row[None, :], group[:, None] < NG, 0.0)
        sw = buffer_load(WS, rn // 128 * NG + group, group < NG, 0.0)
    else:
        sa = gl.load(AS + group[:, None] * MP + row[None, :], group[:, None] < NG, 0.0)
        sw = gl.load(WS + rn // 128 * NG + group, group < NG, 0.0)
    return sa * sw[:, None]

@gluon.jit
def _panel_offsets(pm, pn, K: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, INTERVAL: gl.constexpr, PAD: gl.constexpr, GROUP_MAJOR: gl.constexpr=False):
    la: gl.constexpr = _copy_layout(_packed_shared(BM, False, INTERVAL, PAD), [BM, 128])
    lb: gl.constexpr = _copy_layout(_packed_shared(BN, True, INTERVAL, PAD), [128, BN])
    row = pm * BM + gl.arange(0, BM, gl.SliceLayout(1, la))
    ak = gl.arange(0, 128, gl.SliceLayout(0, la))
    bk = gl.arange(0, 128, gl.SliceLayout(1, lb))
    col = pn * BN + gl.arange(0, BN, gl.SliceLayout(0, lb))
    return (_packed_offset(row[:, None], ak[None, :], 128 if GROUP_MAJOR else K), _packed_offset(col[None, :], bk[:, None], K))

@gluon.jit
def _copy_panel(a_slot, b_slot, A, W, ao, bo):
    async_copy.buffer_load_to_shared(a_slot, A, ao)
    async_copy.buffer_load_to_shared(b_slot, W, bo)
    async_copy.commit_group()

@gluon.jit
def _load_panel(a_slot, b_slot, K_WIDTH: gl.constexpr=16):
    a = async_copy.load_shared_relaxed(a_slot, gl.DotOperandLayout(0, _MMA, K_WIDTH)).to(gl.float8e4nv, bitcast=True)
    b = async_copy.load_shared_relaxed(b_slot, gl.DotOperandLayout(1, _MMA, K_WIDTH)).to(gl.float8e4nv, bitcast=True)
    return (a, b)

@gluon.jit
def _drain_lds():
    gl.inline_asm_elementwise('s_waitcnt lgkmcnt(0)\n s_mov_b32 $0, 0', '=s,~{memory}', [], dtype=gl.int32, is_pure=False, pack=1)
    gl.barrier()

@gluon.jit
def _accumulate(a, b, factor, acc):
    partial = mfma_scaled(a, None, 'e4m3', b, None, 'e4m3', gl.zeros(acc.shape, gl.float32, _MMA))
    return acc + partial * factor[:, None]

@gluon.jit
def _single_tile(A, AS, W, WS, pm, pn, start, MP: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, SG: gl.constexpr, GROUPS: gl.constexpr, STREAM: gl.constexpr, INTERVAL: gl.constexpr, PAD: gl.constexpr, K_WIDTH: gl.constexpr=16, GROUP_MAJOR: gl.constexpr=False):
    NG: gl.constexpr = K // 128
    ASTEP: gl.constexpr = MP * 128 if GROUP_MAJOR else 2048
    APITCH: gl.constexpr = 128 if GROUP_MAJOR else K
    a_slot = gl.allocate_shared_memory(gl.uint8, [BM, 128], _packed_shared(BM, False, INTERVAL, PAD))
    b_slot = gl.allocate_shared_memory(gl.uint8, [128, BN], _packed_shared(BN, True, INTERVAL, PAD))
    if STREAM:
        ao, bo = _panel_offsets(0, 0, K, BM, BN, INTERVAL, PAD, GROUP_MAJOR)
        A, W = (A + pm * BM * APITCH + start * ASTEP, W + pn * BN * K + start * 2048)
        flat = gl.allocate_shared_memory(gl.float32, [SG * BM], gl.SwizzledSharedLayout(1, 1, 1, [0]))
    else:
        ao, bo = _panel_offsets(pm, pn, K, BM, BN, INTERVAL, PAD, GROUP_MAJOR)
        ao, bo = (ao + start * ASTEP, bo + start * 2048)
        factors = _scale_factors(AS, WS, pm * BM, pn * BN, start, MP, NG, BM, SG, False)
        flat = gl.allocate_shared_memory(gl.float32, [SG * BM], gl.SwizzledSharedLayout(1, 1, 1, [0]), factors.reshape([SG * BM]))
    scales = flat._reinterpret(shape=[SG, BM])
    acc = gl.zeros((BM, BN), gl.float32, _MMA)
    if STREAM:
        _copy_panel(a_slot, b_slot, A, W, ao, bo)
        for chunk in range(triton.cdiv(GROUPS, SG)):
            factors = _scale_factors(AS, WS, pm * BM, pn * BN, start + chunk * SG, MP, NG, BM, SG, True)
            _drain_lds()
            flat.store(factors.reshape([SG * BM]))
            gl.barrier()
            for j in range(gl.minimum(SG, GROUPS - chunk * SG)):
                group = chunk * SG + j
                async_copy.wait_group(0)
                a, b = _load_panel(a_slot, b_slot, K_WIDTH)
                factor = async_copy.load_shared_relaxed(scales.index(j), gl.SliceLayout(1, _MMA))
                if group + 1 < GROUPS:
                    _drain_lds()
                    _copy_panel(a_slot, b_slot, A + (group + 1) * ASTEP, W + (group + 1) * 2048, ao, bo)
                acc = _accumulate(a, b, factor, acc)
    else:
        gl.barrier()
        _copy_panel(a_slot, b_slot, A, W, ao, bo)
        for group in range(GROUPS - 1):
            async_copy.wait_group(0)
            a, b = _load_panel(a_slot, b_slot, K_WIDTH)
            factor = async_copy.load_shared_relaxed(scales.index(group), gl.SliceLayout(1, _MMA))
            _drain_lds()
            _copy_panel(a_slot, b_slot, A, W, ao + (group + 1) * ASTEP, bo + (group + 1) * 2048)
            acc = _accumulate(a, b, factor, acc)
        async_copy.wait_group(0)
        a, b = _load_panel(a_slot, b_slot, K_WIDTH)
        factor = async_copy.load_shared_relaxed(scales.index(GROUPS - 1), gl.SliceLayout(1, _MMA))
        acc = _accumulate(a, b, factor, acc)
    return acc

@gluon.jit
def _coordinates(tile, M: gl.constexpr, N: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, PANEL: gl.constexpr, ROW_SUB: gl.constexpr):
    rows: gl.constexpr = triton.cdiv(M, BM)
    columns: gl.constexpr = triton.cdiv(N, BN)
    tile = tile.to(gl.uint32)
    if PANEL == 0:
        return (tile % rows, tile // rows)
    else:
        full: gl.constexpr = columns // PANEL
        remainder: gl.constexpr = columns % PANEL
        if remainder > 0 and tile >= full * rows * PANEL:
            tail = tile - full * rows * PANEL
            return (tail // remainder, full * PANEL + tail % remainder)
        else:
            if ROW_SUB > 1:
                pm = tile // (PANEL * ROW_SUB) % (rows // ROW_SUB) * ROW_SUB + tile % ROW_SUB
                pn = tile // (rows * PANEL) * PANEL + tile // ROW_SUB % PANEL
            else:
                pm = tile // PANEL % rows
                pn = tile // (rows * PANEL) * PANEL + tile % PANEL
            return (pm, pn)

@gluon.jit
def _partial_offset(row, channel, WIDTH: gl.constexpr, ROW_STEP: gl.constexpr, PACK_ROWS: gl.constexpr, PACK_COLS: gl.constexpr=4):
    if PACK_ROWS:
        return row // PACK_ROWS * (WIDTH * ROW_STEP * PACK_ROWS) + channel // PACK_COLS * (PACK_ROWS * PACK_COLS) + row % PACK_ROWS * PACK_COLS + channel % PACK_COLS
    return row * (WIDTH * ROW_STEP) + channel

@gluon.jit
def _store_tile(acc, O, pm, pn, M: gl.constexpr, WIDTH: gl.constexpr, VEC: gl.constexpr, CACHE: gl.constexpr, ROW_STEP: gl.constexpr=1, PACK_ROWS: gl.constexpr=0, PACK_COLS: gl.constexpr=4):
    BM: gl.constexpr = acc.shape[0]
    BN: gl.constexpr = acc.shape[1]
    layout: gl.constexpr = _MMA if PACK_ROWS else gl.BlockedLayout([1, VEC], [4, 16], [gl.num_warps(), 1], [1, 0])
    value = gl.convert_layout(acc.to(O.dtype.element_ty), layout)
    row = pm * BM + gl.arange(0, BM, gl.SliceLayout(1, layout))
    col = pn * BN + gl.arange(0, BN, gl.SliceLayout(0, layout))
    mask = None if M % BM == 0 else row[:, None] < M
    buffer_store(value, O, _partial_offset(row[:, None], col[None, :], WIDTH, ROW_STEP, PACK_ROWS, PACK_COLS), mask, cache=CACHE)

@gluon.jit
def _partial_plane(O, split, M: gl.constexpr, WIDTH: gl.constexpr, INTERLEAVED: gl.constexpr, PACK_ROWS: gl.constexpr=0):
    return O + split * (WIDTH * (PACK_ROWS or 1) if INTERLEAVED else M * WIDTH)

@gluon.jit
def _project(A, AS, W, WS, QO, KO, RO, M: gl.constexpr, MP: gl.constexpr, K: gl.constexpr, Q: gl.constexpr, KV: gl.constexpr, R: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, SG: gl.constexpr, SPLITS: gl.constexpr, STREAM: gl.constexpr, PANEL: gl.constexpr, ROW_SUB: gl.constexpr, INTERVAL: gl.constexpr, PAD: gl.constexpr, OUT_VEC: gl.constexpr, OUT_CACHE: gl.constexpr, GROUP_MAJOR: gl.constexpr, JOINT: gl.constexpr=False, INTERLEAVED: gl.constexpr=False, PACK_ROWS: gl.constexpr=0, PACK_COLS: gl.constexpr=4):
    ROW_STEP: gl.constexpr = SPLITS if INTERLEAVED else 1
    if PACK_ROWS:
        gl.static_assert(SPLITS > 1 and M % PACK_ROWS == 0 and (BM % PACK_ROWS == 0))
        gl.static_assert(Q % PACK_COLS == 0 and KV % PACK_COLS == 0 and (R % PACK_COLS == 0))
    if SPLITS == 1:
        tile = gl.program_id(0)
        split = 0
    else:
        tile = gl.program_id(0) // SPLITS
        split = gl.program_id(0) % SPLITS
    pm, pn = _coordinates(tile, M, Q + KV + R, BM, BN, PANEL, ROW_SUB)
    start = split * (K // 128 // SPLITS)
    if pn < (Q + KV) // BN:
        acc = _single_tile(A, AS, W, WS, pm, pn, start, MP, K, BM, BN, SG, K // 128 // SPLITS, STREAM, INTERVAL, PAD, 16, GROUP_MAJOR)
        if JOINT:
            _store_tile(acc, _partial_plane(QO, split, M, Q + KV + R, INTERLEAVED, PACK_ROWS), pm, pn, M, Q + KV + R, OUT_VEC, OUT_CACHE, ROW_STEP, PACK_ROWS, PACK_COLS)
        elif pn < Q // BN:
            _store_tile(acc, _partial_plane(QO, split, M, Q, INTERLEAVED, PACK_ROWS), pm, pn, M, Q, OUT_VEC, OUT_CACHE, ROW_STEP, PACK_ROWS, PACK_COLS)
        else:
            _store_tile(acc, _partial_plane(KO, split, M, KV, INTERLEAVED, PACK_ROWS), pm, pn - Q // BN, M, KV, OUT_VEC, OUT_CACHE, ROW_STEP, PACK_ROWS, PACK_COLS)
    else:
        rp = pn - (Q + KV) // BN
        rope = _single_tile(A, AS, W + (Q + KV) * K, WS + (Q + KV) // 128 * (K // 128), pm, rp, start, MP, K, BM, 64, SG, K // 128 // SPLITS, STREAM, INTERVAL, PAD, 16, GROUP_MAJOR)
        if JOINT:
            _store_tile(rope, _partial_plane(QO, split, M, Q + KV + R, INTERLEAVED, PACK_ROWS) + (Q + KV) * (PACK_ROWS or 1), pm, rp, M, Q + KV + R, OUT_VEC, OUT_CACHE, ROW_STEP, PACK_ROWS, PACK_COLS)
        else:
            _store_tile(rope, _partial_plane(RO, split, M, R, INTERLEAVED, PACK_ROWS), pm, rp, M, R, OUT_VEC, OUT_CACHE, ROW_STEP, PACK_ROWS, PACK_COLS)

@gluon.jit
def _normalize_row(source, gamma, output, WIDTH: gl.constexpr, EPS: gl.constexpr, VEC: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [gl.num_warps()], [0])
    d = gl.arange(0, triton.next_power_of_2(WIDTH), layout)
    x = gl.load(source + d, d < WIDTH, 0).to(gl.float32)
    gain = gl.load(gamma + d, d < WIDTH, 0).to(gl.float32)
    y = x * gl.rsqrt(gl.sum(x * x, 0) / WIDTH + EPS) * gain
    gl.store(output + d, y.to(gl.bfloat16), d < WIDTH)

@gluon.jit
def _normalize(QG, KG, QO, KO, Q: gl.constexpr, KV: gl.constexpr, EPS: gl.constexpr, VEC: gl.constexpr):
    row = gl.program_id(0)
    if gl.program_id(1) == 0:
        _normalize_row(QO + row * Q, QG, QO + row * Q, Q, EPS, VEC)
    else:
        _normalize_row(KO + row * KV, KG, KO + row * KV, KV, EPS, VEC)

@gluon.jit
def _merge_norm_rows(P, G, Y, base, M: gl.constexpr, WIDTH: gl.constexpr, EPS: gl.constexpr, SPLITS: gl.constexpr, ROWS: gl.constexpr, VEC: gl.constexpr, PITCH: gl.constexpr, OFFSET: gl.constexpr, INTERLEAVED: gl.constexpr=False, PACK_ROWS: gl.constexpr=0, PACK_COLS: gl.constexpr=4):
    layout: gl.constexpr = gl.BlockedLayout([1, PACK_COLS], [PACK_ROWS, 64 // PACK_ROWS], [1, gl.num_warps()], [1, 0]) if PACK_ROWS else gl.BlockedLayout([1, VEC], [1, 64], [gl.num_warps(), 1], [1, 0])
    rows = base + gl.arange(0, ROWS, gl.SliceLayout(1, layout))
    d = gl.arange(0, triton.next_power_of_2(WIDTH), gl.SliceLayout(0, layout))
    ROW_PITCH: gl.constexpr = PITCH * SPLITS if INTERLEAVED else PITCH
    SPLIT_PITCH: gl.constexpr = PITCH * (PACK_ROWS or 1) if INTERLEAVED else M * PITCH
    row_base = base // PACK_ROWS * PACK_ROWS if PACK_ROWS else base
    off = _partial_offset(rows[:, None] - row_base, d[None, :], PITCH, SPLITS if INTERLEAVED else 1, PACK_ROWS, PACK_COLS)
    out_off = (rows[:, None] - base) * WIDTH + d[None, :]
    mask = (rows[:, None] < M) & (d[None, :] < WIDTH)
    x = buffer_load(P + row_base * ROW_PITCH + OFFSET * (PACK_ROWS or 1), off, mask, 0).to(gl.float32)
    for split in gl.static_range(1, SPLITS):
        x += buffer_load(P + split * SPLIT_PITCH + row_base * ROW_PITCH + OFFSET * (PACK_ROWS or 1), off, mask, 0).to(gl.float32)
    gain = buffer_load(G, d, d < WIDTH, 0).to(gl.float32)
    x = x.to(gl.bfloat16).to(gl.float32)
    inv = gl.rsqrt(gl.sum(x * x, 1) / WIDTH + EPS)
    y = x * inv[:, None] * gain[None, :]
    buffer_store(y.to(gl.bfloat16), Y + base * WIDTH, out_off, mask, cache='.wt')

@gluon.jit
def _merge_rope_rows(P, Y, base, M: gl.constexpr, R: gl.constexpr, SPLITS: gl.constexpr, ROWS: gl.constexpr, PITCH: gl.constexpr, OFFSET: gl.constexpr, INTERLEAVED: gl.constexpr=False, PACK_ROWS: gl.constexpr=0, PACK_COLS: gl.constexpr=4):
    layout: gl.constexpr = gl.BlockedLayout([1, PACK_COLS], [PACK_ROWS, 64 // PACK_ROWS], [gl.num_warps(), 1], [1, 0]) if PACK_ROWS else gl.BlockedLayout([1, 8], [8, 8], [gl.num_warps(), 1], [1, 0])
    rows = base + gl.arange(0, ROWS, gl.SliceLayout(1, layout))
    d = gl.arange(0, R, gl.SliceLayout(0, layout))
    ROW_PITCH: gl.constexpr = PITCH * SPLITS if INTERLEAVED else PITCH
    SPLIT_PITCH: gl.constexpr = PITCH * (PACK_ROWS or 1) if INTERLEAVED else M * PITCH
    row_base = base // PACK_ROWS * PACK_ROWS if PACK_ROWS else base
    off = _partial_offset(rows[:, None] - row_base, d[None, :], PITCH, SPLITS if INTERLEAVED else 1, PACK_ROWS, PACK_COLS)
    out_off = (rows[:, None] - base) * R + d[None, :]
    mask = rows[:, None] < M
    x = buffer_load(P + row_base * ROW_PITCH + OFFSET * (PACK_ROWS or 1), off, mask, 0).to(gl.float32)
    for split in gl.static_range(1, SPLITS):
        x += buffer_load(P + split * SPLIT_PITCH + row_base * ROW_PITCH + OFFSET * (PACK_ROWS or 1), off, mask, 0).to(gl.float32)
    buffer_store(x.to(gl.bfloat16), Y + base * R, out_off, mask, cache='.wt')

@gluon.jit
def _finish_compact(QP, KP, RP, QG, KG, QO, KO, RO, M: gl.constexpr, Q: gl.constexpr, KV: gl.constexpr, R: gl.constexpr, EPS: gl.constexpr, SPLITS: gl.constexpr, VEC: gl.constexpr, ROWS: gl.constexpr, ROPE_ROWS: gl.constexpr, JOINT: gl.constexpr=False, INTERLEAVED: gl.constexpr=False, PACK_ROWS: gl.constexpr=0, PACK_COLS: gl.constexpr=4):
    pid = gl.program_id(0)
    tasks: gl.constexpr = triton.cdiv(M, ROWS)
    if pid < tasks:
        _merge_norm_rows(QP, QG, QO, pid * ROWS, M, Q, EPS, SPLITS, ROWS, VEC, Q + KV + R if JOINT else Q, 0, INTERLEAVED, PACK_ROWS, PACK_COLS)
    elif pid < 2 * tasks:
        base = (pid - tasks) * ROWS
        _merge_norm_rows(KP, KG, KO, base, M, KV, EPS, SPLITS, ROWS, VEC, Q + KV + R if JOINT else KV, Q if JOINT else 0, INTERLEAVED, PACK_ROWS, PACK_COLS)
    elif SPLITS > 1:
        _merge_rope_rows(RP, RO, (pid - 2 * tasks) * ROPE_ROWS, M, R, SPLITS, ROPE_ROWS, Q + KV + R if JOINT else R, Q + KV if JOINT else 0, INTERLEAVED, PACK_ROWS, PACK_COLS)

class _Schedule(NamedTuple):
    bm: int = 64
    bn: int = 128
    splits: int = 1
    sg: int = 64
    stream: bool = False
    qr: int = 64
    qv: int = 16
    qgroups: int = 1
    rebase: bool = True
    row_first: bool = False
    panel: int = 0
    row_sub: int = 1
    interval: int = 1024
    padding: int = 32
    out_vec: int = 8
    norm_warps: int = 1
    norm_vec: int = 8
    a_major: bool = False
    compact_finish: bool = False
    finish_rows: int = 1
    joint_partials: bool = False
    quant_warps: int = 4
    interleaved: bool = False
    partial_pack: int = 0
    partial_vec: int = 4
    scale_cache: str = ''
_SMALL = _Schedule(bm=128, splits=3, sg=16, rebase=False, row_first=True, qgroups=2, panel=4, row_sub=2, interval=4096, padding=64, a_major=True, compact_finish=True, finish_rows=4, norm_warps=4, partial_pack=4, partial_vec=8, scale_cache='.wt')
_MEDIUM = _Schedule(bm=128, splits=2, sg=16, stream=True, qr=32, qgroups=4, quant_warps=2, panel=2, out_vec=4, a_major=True, compact_finish=True, joint_partials=True, interleaved=True, finish_rows=4, norm_warps=4, partial_pack=4, row_first=True, scale_cache='.wt')
_STREAM_4096 = _Schedule(bm=128, sg=16, stream=True, qgroups=2, interval=2048, compact_finish=True, finish_rows=4, norm_warps=4)
_STREAM_8192 = _Schedule(bm=128, sg=16, stream=True, qr=32, qv=8, interval=2048, padding=64, norm_vec=2, a_major=True, compact_finish=True, finish_rows=4, norm_warps=4)
_TAIL = _Schedule(qr=32, qv=8)
_LARGE_TAIL = _Schedule(bm=128, qr=32, qv=8, norm_vec=2)

def _schedule(m):
    if m == 1024:
        return _SMALL
    if m == 2048:
        return _MEDIUM
    if m == 4096:
        return _STREAM_4096
    if m == 8192:
        return _STREAM_8192
    return _LARGE_TAIL if m >= 8192 else _TAIL

@gluon.jit
def _quantize_generic(X, A, AS, M: gl.constexpr, K: gl.constexpr, SX: gl.constexpr, MAX: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    row = gl.program_id(0) * 32 + gl.arange(0, 32, gl.SliceLayout(1, layout))
    col = gl.program_id(1) * 128 + gl.arange(0, 128, gl.SliceLayout(0, layout))
    mask = (row[:, None] < M) & (col[None, :] < K)
    x = gl.load(X + row[:, None] * SX + col[None, :], mask, 0).to(gl.float32)
    scale = gl.maximum(gl.max(gl.abs(x), 1), 1e-10) * (1.0 / MAX)
    value = (x * gl.div_rn(1.0, scale)[:, None]).to(A.dtype.element_ty)
    gl.store(A + row[:, None] * K + col[None, :], value, mask)
    gl.store(AS + row * gl.cdiv(K, 128) + gl.program_id(1), scale, row < M)

@gluon.jit
def _bf16_project(X, W, P, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, SX: gl.constexpr, SW: gl.constexpr):
    mma: gl.constexpr = AMDMFMALayout(4, [32, 32, 16], True, [4, 1])
    al: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    bl: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    row = gl.program_id(0) * 16 + gl.arange(0, 16, gl.SliceLayout(1, al))
    ak = gl.arange(0, 32, gl.SliceLayout(0, al))
    col = gl.program_id(1) * 64 + gl.arange(0, 64, gl.SliceLayout(0, bl))
    bk = gl.arange(0, 32, gl.SliceLayout(1, bl))
    acc = gl.zeros((16, 64), gl.float32, mma)
    for group in range(gl.cdiv(K, 32)):
        a = gl.load(X + row[:, None] * SX + group * 32 + ak[None, :], (row[:, None] < M) & (group * 32 + ak[None, :] < K), 0)
        b = gl.load(W + col[None, :] * SW + group * 32 + bk[:, None], (col[None, :] < N) & (group * 32 + bk[:, None] < K), 0)
        acc = mfma(a, b, acc)
    rr = gl.program_id(0) * 16 + gl.arange(0, 16, gl.SliceLayout(1, mma))
    cc = gl.program_id(1) * 64 + gl.arange(0, 64, gl.SliceLayout(0, mma))
    gl.store(P + rr[:, None] * N + cc[None, :], acc.to(gl.bfloat16), (rr[:, None] < M) & (cc[None, :] < N))

@gluon.jit
def _fp8_generic(A, AS, W, WS, P, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, PACKED: gl.constexpr):
    mma: gl.constexpr = AMDMFMALayout(4, [32, 32, 16], True, [4, 1])
    al: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    bl: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    la: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    lb: gl.constexpr = gl.BlockedLayout([4, 1], [16, 4], [1, 4], [0, 1])
    ar = gl.program_id(0) * 32 + gl.arange(0, 32, gl.SliceLayout(1, la))
    ak = gl.arange(0, 128, gl.SliceLayout(0, la))
    bk = gl.arange(0, 128, gl.SliceLayout(1, lb))
    bc = gl.program_id(1) * 32 + gl.arange(0, 32, gl.SliceLayout(0, lb))
    row = gl.program_id(0) * 32 + gl.arange(0, 32, gl.SliceLayout(1, mma))
    col = gl.program_id(1) * 32 + gl.arange(0, 32, gl.SliceLayout(0, mma))
    acc = gl.zeros((32, 32), gl.float32, mma)
    for group in range(gl.cdiv(K, 128)):
        a = gl.load(A + ar[:, None] * K + group * 128 + ak[None, :], (ar[:, None] < M) & (group * 128 + ak[None, :] < K), 0.0)
        k = group * 128 + bk
        if PACKED:
            off = _packed_offset(bc[None, :], k[:, None], K)
        else:
            off = bc[None, :] * K + k[:, None]
        b = gl.load(W + off, (bc[None, :] < N) & (k[:, None] < K), 0.0)
        a = gl.convert_layout(a.to(gl.bfloat16), al)
        b = gl.convert_layout(b.to(gl.bfloat16), bl)
        part = mfma(a, b, gl.zeros((32, 32), gl.float32, mma))
        sa = gl.load(AS + row * gl.cdiv(K, 128) + group, row < M, 0)
        sw = gl.load(WS + col // 128 * gl.cdiv(K, 128) + group, col < N, 0)
        acc += part * sa[:, None] * sw[None, :]
    gl.store(P + row[:, None] * N + col[None, :], acc.to(gl.bfloat16), (row[:, None] < M) & (col[None, :] < N))

@gluon.jit
def _norm_from_projection(P, G, Y, row, WIDTH: gl.constexpr, N: gl.constexpr, OFFSET: gl.constexpr, EPS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [1], [0])
    d = gl.arange(0, triton.next_power_of_2(WIDTH), layout)
    x = gl.load(P + row * N + OFFSET + d, d < WIDTH, 0).to(gl.float32)
    gain = gl.load(G + d, d < WIDTH, 0).to(gl.float32)
    y = x * gl.rsqrt(gl.sum(x * x, 0) / WIDTH + EPS) * gain
    gl.store(Y + row * WIDTH + d, y.to(gl.bfloat16), d < WIDTH)

@gluon.jit
def _finish_generic(P, QG, KG, QO, KO, RO, Q: gl.constexpr, KV: gl.constexpr, R: gl.constexpr, EPS: gl.constexpr):
    row = gl.program_id(0)
    if gl.program_id(1) == 0:
        _norm_from_projection(P, QG, QO, row, Q, Q + KV + R, 0, EPS)
    elif gl.program_id(1) == 1:
        _norm_from_projection(P, KG, KO, row, KV, Q + KV + R, Q, EPS)
    else:
        layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
        d = gl.arange(0, triton.next_power_of_2(R), layout)
        x = gl.load(P + row * (Q + KV + R) + Q + KV + d, d < R, 0)
        gl.store(RO + row * R + d, x, d < R)

def _generic(hidden, weight, q_gamma, kv_gamma, weight_scales, rope_dim, eps, packed):
    m, k = hidden.shape
    q, kv = (q_gamma.numel(), kv_gamma.numel())
    n = q + kv + rope_dim
    projected = torch.empty((m, n), device=hidden.device, dtype=torch.bfloat16)
    qo = torch.empty((m, q), device=hidden.device, dtype=torch.bfloat16)
    ko = torch.empty((m, kv), device=hidden.device, dtype=torch.bfloat16)
    ro = torch.empty((m, rope_dim), device=hidden.device, dtype=torch.bfloat16)
    if weight_scales is None:
        _bf16_project[triton.cdiv(m, 16), triton.cdiv(n, 64)](hidden, weight, projected, m, n, k, hidden.stride(0), weight.stride(0))
    else:
        a = torch.empty((m, k), device=hidden.device, dtype=weight.dtype)
        scales = torch.empty((m, triton.cdiv(k, 128)), device=hidden.device, dtype=torch.float32)
        _quantize_generic[triton.cdiv(m, 32), triton.cdiv(k, 128)](hidden, a, scales, m, k, hidden.stride(0), torch.finfo(weight.dtype).max)
        _fp8_generic[triton.cdiv(m, 32), triton.cdiv(n, 32)](a, scales, weight, weight_scales, projected, m, n, k, packed)
    _finish_generic[m, 3](projected, q_gamma, kv_gamma, qo, ko, ro, q, kv, rope_dim, eps, num_warps=1, enable_fp_fusion=False)
    return (qo, ko, ro)

def mla_qkv_a_norm(hidden: torch.Tensor, weight: torch.Tensor, q_gamma: torch.Tensor, kv_gamma: torch.Tensor, weight_scales: torch.Tensor | None=None, *, rope_dim: int=64, eps: float=1e-05, weight_preshuffled: bool=False):
    m, k = hidden.shape
    q, kv = (q_gamma.numel(), kv_gamma.numel())
    assert m > 0 and k > 0 and (q > 0) and (kv > 0) and (0 < rope_dim <= 256) and (eps > 0)
    assert weight.shape == (q + kv + rope_dim, k) and q_gamma.ndim == kv_gamma.ndim == 1
    fast = weight_scales is not None and weight_preshuffled and (weight.dtype == torch.float8_e4m3fn) and (k == 6144) and (q % 128 == 0) and (kv % 128 == 0) and (rope_dim == 64)
    if not fast:
        return _generic(hidden, weight, q_gamma, kv_gamma, weight_scales, rope_dim, eps, weight_preshuffled)
    return _fast(hidden, weight, q_gamma, kv_gamma, weight_scales, rope_dim, eps, _schedule(m))

def _fast(hidden, weight, q_gamma, kv_gamma, weight_scales, rope_dim, eps, cfg):
    m, k = hidden.shape
    q, kv = (q_gamma.numel(), kv_gamma.numel())
    mp = triton.cdiv(m, cfg.bm) * cfg.bm
    groups = k // 128
    qo = torch.empty((m, q), dtype=torch.bfloat16, device=hidden.device)
    ko = torch.empty((m, kv), dtype=torch.bfloat16, device=hidden.device)
    ro = torch.empty((m, rope_dim), dtype=torch.bfloat16, device=hidden.device)
    a = torch.empty((mp, k), dtype=torch.uint8, device=hidden.device)
    scales = torch.empty((groups, mp), dtype=torch.float32, device=hidden.device)
    if cfg.splits == 1:
        qp, kp, rp = (qo, ko, ro)
    elif cfg.joint_partials:
        shape = (m, cfg.splits, q + kv + rope_dim) if cfg.interleaved else (cfg.splits, m, q + kv + rope_dim)
        qp = torch.empty(shape, dtype=torch.float32, device=hidden.device)
        kp = rp = qp
    else:
        qp = torch.empty((cfg.splits, m, q), dtype=torch.float32, device=hidden.device)
        kp = torch.empty((cfg.splits, m, kv), dtype=torch.float32, device=hidden.device)
        rp = torch.empty((cfg.splits, m, rope_dim), dtype=torch.float32, device=hidden.device)
    if cfg.qgroups == 1:
        grid = (mp // cfg.qr, groups) if cfg.row_first else (groups, mp // cfg.qr)
        _quantize[grid](hidden, a, scales, m, mp, k, hidden.stride(0), cfg.qr, cfg.qv, '.wt', cfg.rebase, cfg.row_first, cfg.a_major, num_warps=cfg.quant_warps)
    else:
        grid = (mp // cfg.qr, groups // cfg.qgroups) if cfg.row_first else (groups // cfg.qgroups, mp // cfg.qr)
        _quantize_multi[grid](hidden, a, scales, m, mp, k, hidden.stride(0), cfg.qr, cfg.qv, '.wt', cfg.qgroups, cfg.a_major, cfg.row_first, cfg.scale_cache, num_warps=cfg.quant_warps)
    tiles = triton.cdiv(m, cfg.bm) * triton.cdiv(q + kv + rope_dim, cfg.bn)
    _project[tiles * cfg.splits,](a, scales, weight.view(torch.uint8), weight_scales, qp, kp, rp, m, mp, k, q, kv, rope_dim, cfg.bm, cfg.bn, cfg.sg, cfg.splits, cfg.stream, cfg.panel, cfg.row_sub, cfg.interval, cfg.padding, cfg.out_vec, '.wt', cfg.a_major, cfg.joint_partials, cfg.interleaved, cfg.partial_pack, cfg.partial_vec, num_warps=4)
    if cfg.compact_finish:
        tasks = 2 * triton.cdiv(m, cfg.finish_rows)
        if cfg.splits > 1:
            tasks += triton.cdiv(m, 32)
        _finish_compact[tasks,](qp, kp, rp, q_gamma, kv_gamma, qo, ko, ro, m, q, kv, rope_dim, eps, cfg.splits, cfg.norm_vec, cfg.finish_rows, 32, cfg.joint_partials, cfg.interleaved, cfg.partial_pack, cfg.partial_vec, num_warps=cfg.norm_warps, enable_fp_fusion=False)
    else:
        _normalize[m, 2](q_gamma, kv_gamma, qo, ko, q, kv, eps, cfg.norm_vec, num_warps=cfg.norm_warps, enable_fp_fusion=False)
    return (qo, ko, ro)
