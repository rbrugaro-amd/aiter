import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import range as loop_range
from xbench.s1f32.qkv_a_norm._decode import _merge
from xbench.s1f32.qkv_a_norm._decode_tiled_projection import _packed_offset

@gluon.jit
def _striped_id(TOTAL: gl.constexpr, STRIPES: gl.constexpr=8):
    pid = gl.program_id(0).to(gl.uint32)
    return pid % STRIPES * (TOTAL // STRIPES) + gl.minimum(pid % STRIPES, TOTAL % STRIPES) + pid // STRIPES

@gluon.jit
def _quantize(X, A, S, M: gl.constexpr, K: gl.constexpr, SX: gl.constexpr, RB: gl.constexpr, NW: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 16], [8, 8], [NW, 1], [1, 0])
    pid = _striped_id(M // RB * (K // 128))
    tile = pid // 8 % (M // RB)
    group = pid % 8 + pid // (8 * (M // RB)) * 8
    rows = tile * RB + gl.arange(0, RB, gl.SliceLayout(1, layout))
    ks = group * 128 + gl.arange(0, 128, gl.SliceLayout(0, layout))
    x = gl.amd.cdna4.buffer_load(X, rows[:, None] * SX + ks[None, :]).to(gl.float32)
    scale = gl.maximum(gl.max(gl.abs(x), 1), 1e-10) * (1.0 / 448.0)
    value = (x * gl.div_rn(1.0, scale)[:, None]).to(A.dtype.element_ty)
    gl.amd.cdna4.buffer_store(value, A, _packed_offset(rows[:, None], ks[None, :], K), cache='.wt')
    gl.amd.cdna4.buffer_store(scale, S, group * M + rows, cache='.wt')

@triton.constexpr_function
def _panel_shared_layout(outer, panel_k, row_lanes, row_waves):
    rows = [[1 << b, 0] for b in range(outer.bit_length() - 1)]
    ks = [[0, 1 << b] for b in range(panel_k.bit_length() - 1)]
    if row_lanes == 64 and row_waves == 1:
        return gl.SharedLinearLayout(ks[:4] + rows + ks[4:])
    lane_rows = row_lanes.bit_length() - 1
    lane_ks = 4 + (64 // row_lanes).bit_length() - 1
    wave_rows = lane_rows + row_waves.bit_length() - 1
    wave_ks = lane_ks + (4 // row_waves).bit_length() - 1
    return gl.SharedLinearLayout(ks[:4] + rows[:lane_rows] + ks[4:lane_ks] + rows[lane_rows:wave_rows] + ks[lane_ks:wave_ks] + rows[wave_rows:] + ks[wave_ks:])

@gluon.jit
def _panel_offsets(OUTER: gl.constexpr, K: gl.constexpr, PANEL_K: gl.constexpr, ROW_LANES: gl.constexpr, ROW_WAVES: gl.constexpr, UNROLLED: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 16], [ROW_LANES, 64 // ROW_LANES], [ROW_WAVES, 4 // ROW_WAVES], [0, 1] if UNROLLED else [1, 0])
    rows = gl.arange(0, OUTER, gl.SliceLayout(1, layout))
    ks = gl.arange(0, PANEL_K, gl.SliceLayout(0, layout))
    return _packed_offset(rows[:, None], ks[None, :], K)

@gluon.jit
def _enqueue(pipe, panel, SLOT: gl.constexpr, PANEL_K: gl.constexpr):
    amem, bmem, abase, wbase, ao, bo = pipe
    gl.amd.cdna4.async_copy.buffer_load_to_shared(amem.index(SLOT), abase + panel * (16 * PANEL_K), ao)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(bmem.index(SLOT), wbase + panel * (16 * PANEL_K), bo)
    gl.amd.cdna4.async_copy.commit_group()

@gluon.jit
def _dot_panel(pipe, dot, panel, SLOT: gl.constexpr, PANEL_K: gl.constexpr, MMA: gl.constexpr, WAIT: gl.constexpr, REFILL: gl.constexpr, DRAIN_FENCE: gl.constexpr=False):
    gl.amd.cdna4.async_copy.wait_group(WAIT)
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(pipe[0].index(SLOT), gl.DotOperandLayout(0, MMA, 16))
    b = gl.amd.cdna4.async_copy.load_shared_relaxed(pipe[1].index(SLOT).permute((1, 0)), gl.DotOperandLayout(1, MMA, 16))
    if REFILL or DRAIN_FENCE:
        gl.barrier()
    if REFILL:
        _enqueue(pipe, panel + 2, SLOT, PANEL_K)
    return gl.amd.cdna4.mfma_scaled(a, None, 'e4m3', b, None, 'e4m3', dot)

@gluon.jit
def _static_panel(pipe, dot, PANEL: gl.constexpr, PANELS: gl.constexpr, PANEL_K: gl.constexpr, MMA: gl.constexpr, FENCE_DRAIN: gl.constexpr):
    return _dot_panel(pipe, dot, PANEL, PANEL % 2, PANEL_K, MMA, min(1, PANELS - PANEL - 1), PANEL + 2 < PANELS, FENCE_DRAIN or PANEL == PANELS - 1)

@triton.constexpr_function
def _anchor_constraints(pack):
    return ','.join(['=v'] * pack + [str(i) for i in range(pack)] + ['~{memory}'])

@gluon.jit
def _accumulate(dot, a_scale, w_scale, acc, PACK: gl.constexpr):
    acc = gl.fma(dot, (a_scale * w_scale)[:, None], acc)
    gl.static_assert(PACK > 0 and PACK & PACK - 1 == 0)
    return gl.inline_asm_elementwise('', _anchor_constraints(PACK), [acc], dtype=gl.float32, is_pure=False, pack=PACK)

@gluon.jit
def _project(A, AS, W, WS, P, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, PANEL_K: gl.constexpr, M_WAVES: gl.constexpr, ROW_LANES: gl.constexpr, ROW_WAVES: gl.constexpr, UNROLLED: gl.constexpr, ANCHOR_PACK: gl.constexpr, SPLITS: gl.constexpr):
    gl.static_assert(K % (128 * SPLITS) == 0 and M % BM == 0 and (N % 64 == 0))
    gl.static_assert(BM % 16 == 0 and (PANEL_K == 64 or PANEL_K == 128))
    GROUPS: gl.constexpr = K // 128 // SPLITS
    PER_GROUP: gl.constexpr = 128 // PANEL_K
    PANELS: gl.constexpr = GROUPS * PER_GROUP
    pid = _striped_id(M // BM * (N // 64) * SPLITS)
    pm, pn = (pid // (N // 64) % (M // BM), pid % (N // 64))
    part = pid // (M // BM * (N // 64))
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[32, 32, 64], transposed=True, warps_per_cta=[M_WAVES, 4 // M_WAVES])
    rows = pm * BM + gl.arange(0, BM, gl.SliceLayout(1, mma))
    start = part * GROUPS
    sa, sw = ((), ())
    if UNROLLED:
        abase = A + pm * BM * K + start * 2048
        wbase = W + pn * 64 * K + start * 2048
        for i in gl.static_range(GROUPS):
            sa += (gl.amd.cdna4.buffer_load(AS, (start + i) * M + rows),)
            sw += (gl.load(WS + pn // 2 * (K // 128) + start + i),)
        ao = _panel_offsets(BM, K, PANEL_K, ROW_LANES, ROW_WAVES, True)
        bo = _panel_offsets(64, K, PANEL_K, ROW_LANES, ROW_WAVES, True)
    amem = gl.allocate_shared_memory(A.dtype.element_ty, (2, BM, PANEL_K), _panel_shared_layout(BM, PANEL_K, ROW_LANES, ROW_WAVES))
    bmem = gl.allocate_shared_memory(W.dtype.element_ty, (2, 64, PANEL_K), _panel_shared_layout(64, PANEL_K, ROW_LANES, ROW_WAVES))
    if UNROLLED:
        pipe = (amem, bmem, abase, wbase, ao, bo)
    else:
        pipe = (amem, bmem, A + pm * BM * K + start * 2048, W + pn * 64 * K + start * 2048, _panel_offsets(BM, K, PANEL_K, ROW_LANES, ROW_WAVES, False), _panel_offsets(64, K, PANEL_K, ROW_LANES, ROW_WAVES, False))
    _enqueue(pipe, 0, 0, PANEL_K)
    _enqueue(pipe, 1, 1, PANEL_K)
    zero = gl.zeros((BM, 64), gl.float32, mma)
    acc = zero
    if UNROLLED:
        for i in gl.static_range(GROUPS):
            dot = zero
            for j in gl.static_range(PER_GROUP):
                dot = _static_panel(pipe, dot, i * PER_GROUP + j, PANELS, PANEL_K, mma, False)
            acc = _accumulate(dot, sa[i], sw[i], acc, ANCHOR_PACK)
    else:
        gl.static_assert(GROUPS >= 2 and GROUPS % 2 == 0)
        for outer in loop_range(0, GROUPS - 2, 2, loop_unroll_factor=1):
            for j in gl.static_range(2):
                group = outer + j
                a_scale = gl.amd.cdna4.buffer_load(AS + (start + group) * M, rows)
                w_scale = gl.load(WS + pn // 2 * (K // 128) + start + group)
                dot = zero
                for half in gl.static_range(PER_GROUP):
                    dot = _dot_panel(pipe, dot, group * PER_GROUP + half, (j * PER_GROUP + half) % 2, PANEL_K, mma, 1, True)
                acc = _accumulate(dot, a_scale, w_scale, acc, ANCHOR_PACK)
        for last in gl.static_range(GROUPS - 2, GROUPS):
            a_scale = gl.amd.cdna4.buffer_load(AS + (start + last) * M, rows)
            w_scale = gl.load(WS + pn // 2 * (K // 128) + start + last)
            dot = zero
            for half in gl.static_range(PER_GROUP):
                dot = _static_panel(pipe, dot, last * PER_GROUP + half, PANELS, PANEL_K, mma, True)
            acc = _accumulate(dot, a_scale, w_scale, acc, ANCHOR_PACK)
    store: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    value = gl.convert_layout(acc.to(P.dtype.element_ty), store)
    r = pm * BM + gl.arange(0, BM, gl.SliceLayout(1, store))
    c = pn * 64 + gl.arange(0, 64, gl.SliceLayout(0, store))
    gl.amd.cdna4.buffer_store(value, P + part * M * N, r[:, None] * N + c[None, :], cache='.wt')

@gluon.jit
def _normalize(P, G, Y, RO, row, N: gl.constexpr, PLANE: gl.constexpr, OFFSET: gl.constexpr, WIDTH: gl.constexpr, ROPE: gl.constexpr, EPS: gl.constexpr, SPLITS: gl.constexpr, VEC: gl.constexpr, BUFFER_GAMMA: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [4], [0])
    d = gl.arange(0, triton.next_power_of_2(WIDTH + ROPE), layout)
    x = _merge(P, row * N + OFFSET + d, d < WIDTH + ROPE, SPLITS, PLANE)
    if BUFFER_GAMMA:
        gamma = gl.amd.cdna4.buffer_load(G, d, d < WIDTH, 0.0, cache='.ca').to(gl.float32)
    else:
        gamma = gl.load(G + d, d < WIDTH, 0.0, cache_modifier='.ca').to(gl.float32)
    norm_x = gl.where(d < WIDTH, x, 0.0)
    inv = gl.rsqrt(gl.sum(norm_x * norm_x, 0) / WIDTH + EPS)
    y = (x * inv * gamma).to(gl.bfloat16)
    gl.amd.cdna4.buffer_store(y, Y, row * WIDTH + d, d < WIDTH, cache='.wt')
    if ROPE:
        gl.amd.cdna4.buffer_store(x.to(gl.bfloat16), RO, row * ROPE + d - WIDTH, (d >= WIDTH) & (d < WIDTH + ROPE), cache='.wt')

@gluon.jit
def _finish(P, QG, KG, QO, KO, RO, M: gl.constexpr, Q: gl.constexpr, KV: gl.constexpr, R: gl.constexpr, EPS: gl.constexpr, SPLITS: gl.constexpr, BATCH_ROPE: gl.constexpr, ROPE_ROWS: gl.constexpr, KV_VEC: gl.constexpr):
    N: gl.constexpr = Q + KV + R
    pid = gl.program_id(0).to(gl.uint32)
    if pid < M:
        _normalize(P, QG, QO, RO, pid, N, M * N, 0, Q, 0, EPS, SPLITS, 8, BATCH_ROPE)
    elif not BATCH_ROPE:
        _normalize(P, KG, KO, RO, pid - M, N, M * N, Q, KV, R, EPS, SPLITS, KV_VEC, False)
    elif pid < 2 * M:
        _normalize(P, KG, KO, RO, pid - M, N, M * N, Q, KV, 0, EPS, SPLITS, KV_VEC, True)
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
        rows = (pid - 2 * M) * ROPE_ROWS + gl.arange(0, ROPE_ROWS, gl.SliceLayout(1, layout))
        d = gl.arange(0, R, gl.SliceLayout(0, layout))
        offsets = rows[:, None] * N + Q + KV + d[None, :]
        x = _merge(P, offsets, rows[:, None] < M, SPLITS, M * N)
        gl.amd.cdna4.buffer_store(x.to(gl.bfloat16), RO, rows[:, None] * R + d[None, :], rows[:, None] < M, cache='.wt')

def mla_qkv_a_norm(hidden: torch.Tensor, weight: torch.Tensor, q_gamma: torch.Tensor, kv_gamma: torch.Tensor, weight_scales: torch.Tensor | None=None, *, rope_dim: int=64, eps: float=1e-05, weight_preshuffled: bool=False):
    m, k = hidden.shape
    q, kv = (q_gamma.numel(), kv_gamma.numel())
    n = q + kv + rope_dim
    assert m in (128, 256) and k == 6144 and ((q, kv, rope_dim) == (2048, 512, 64))
    assert weight_preshuffled and weight_scales is not None and (weight.dtype == torch.float8_e4m3fn)
    splits = 6
    small = m == 128
    if small:
        qo = torch.empty((m, q), device=hidden.device, dtype=torch.bfloat16)
        ko = torch.empty((m, kv), device=hidden.device, dtype=torch.bfloat16)
        ro = torch.empty((m, rope_dim), device=hidden.device, dtype=torch.bfloat16)
    else:
        partials = torch.empty((splits, m, n), device=hidden.device, dtype=torch.float32)
        outputs = torch.empty((m * n,), device=hidden.device, dtype=torch.bfloat16)
        qo = outputs[:m * q].view(m, q)
        ko = outputs[m * q:m * (q + kv)].view(m, kv)
        ro = outputs[m * (q + kv):].view(m, rope_dim)
    a = torch.empty((m, k), device=hidden.device, dtype=weight.dtype)
    scales = torch.empty((k // 128, m), device=hidden.device, dtype=torch.float32)
    if small:
        partials = torch.empty((splits, m, n), device=hidden.device, dtype=torch.float32)
    rb, quant_waves = (8, 1) if small else (16, 2)
    block_m, panel_k, m_waves = (64, 128, 2) if small else (128, 64, 4)
    row_lanes, row_waves, anchor_pack = (16, 4, 4) if small else (64, 1, 2)
    batch_rope, rope_rows, kv_vec = (not small, 8, 1 if small else 2)
    _quantize[m // rb * (k // 128),](hidden, a, scales, m, k, hidden.stride(0), rb, quant_waves, num_warps=quant_waves)
    _project[m // block_m * (n // 64) * splits,](a, scales, weight, weight_scales, partials, m, n, k, block_m, panel_k, m_waves, row_lanes, row_waves, small, anchor_pack, splits, num_warps=4, waves_per_eu=0 if small else 2)
    _finish[m * 2 + (m // rope_rows if batch_rope else 0),](partials, q_gamma, kv_gamma, qo, ko, ro, m, q, kv, rope_dim, eps, splits, batch_rope, rope_rows, kv_vec, num_warps=4, enable_fp_fusion=False)
    return (qo, ko, ro)
