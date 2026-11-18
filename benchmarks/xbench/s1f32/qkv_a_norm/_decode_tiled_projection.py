'Shared tiled decode projection with packed FP8 weights.'

from xbench.s1f32.qkv_a_norm._decode import _enqueue, _packed_shared, _read_panel, _store_split, _tile_ids

from triton.experimental.gluon import language as gl

from triton.experimental import gluon

@gluon.jit
def _packed_offset(row, k, K: gl.constexpr):
    return row // 16 * K * 16 + k // 16 * 256 + row % 16 * 16 + k % 16

@gluon.jit
def _packed_tile(OUTER: gl.constexpr, K: gl.constexpr, PANEL: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 16], [64, 1], [1, 4], [1, 0])
    row = gl.arange(0, OUTER, gl.SliceLayout(1, layout))
    k = gl.arange(0, PANEL, gl.SliceLayout(0, layout))
    return _packed_offset(row[:, None], k[None, :], K)

@gluon.jit
def _project_lds(A, AS, W, WS, P, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, SPLITS: gl.constexpr, STAGES: gl.constexpr=2, STRIPES: gl.constexpr=8, WM: gl.constexpr=2):
    gl.static_assert(K % (128 * SPLITS) == 0 and M % BM == 0 and (N % 64 == 0))
    gl.static_assert(BM % 16 == 0)
    GROUPS: gl.constexpr = K // 128 // SPLITS
    gl.static_assert((STAGES >= 1) & (STAGES <= GROUPS))
    pm, pn, part = _tile_ids(M, N, BM, SPLITS, STRIPES)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[32, 32, 64], transposed=True, warps_per_cta=[WM, 4 // WM])
    rows = pm * BM + gl.arange(0, BM, gl.SliceLayout(1, mma))
    start = part * GROUPS
    abase, wbase = (A + pm * BM * K + start * 2048, W + pn * 64 * K + start * 2048)
    sa, sw = ((), ())
    for i in gl.static_range(GROUPS):
        sa += (gl.amd.cdna4.buffer_load(AS, (start + i) * M + rows),)
        sw += (gl.load(WS + pn // 2 * (K // 128) + start + i),)
    ao, bo = (_packed_tile(BM, K, 128), _packed_tile(64, K, 128))
    amem = gl.allocate_shared_memory(A.dtype.element_ty, (STAGES, BM, 128), _packed_shared(BM, 128))
    bmem = gl.allocate_shared_memory(W.dtype.element_ty, (STAGES, 64, 128), _packed_shared(64, 128))
    for i in gl.static_range(STAGES):
        _enqueue(amem, bmem, abase, wbase, ao, bo, i, STAGES, 128)
    zero = gl.zeros((BM, 64), gl.float32, mma)
    acc = zero
    for i in gl.static_range(GROUPS):
        a, b = _read_panel(amem, bmem, abase, wbase, ao, bo, i, GROUPS, STAGES, mma, 128)
        dot = gl.amd.cdna4.mfma_scaled(a, None, 'e4m3', b, None, 'e4m3', zero)
        acc = gl.fma(dot, (sa[i] * sw[i])[:, None], acc)
        acc = gl.inline_asm_elementwise('', '=v,0,~{memory}', [acc], dtype=gl.float32, is_pure=False, pack=1)
    _store_split(acc, P, pm, pn, part, M, N, BM, 8)
