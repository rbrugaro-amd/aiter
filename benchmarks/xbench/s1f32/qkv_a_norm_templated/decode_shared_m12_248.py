import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from xbench.s1f32.qkv_a_norm._decode import _enqueue, _merge, _packed_shared, _read_panel, _store_split, _tile_ids
from xbench.s1f32.qkv_a_norm._decode_tiled_projection import _project_lds, _packed_tile, _packed_offset

@gluon.jit
def _quantize_wide(X, A, S, M: gl.constexpr, K: gl.constexpr, SX: gl.constexpr, LIVE_M: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 16], [1, 1], [1, 0])
    pid = gl.program_id(0).to(gl.uint32)
    tile, group = (pid % (M // 4), pid // (M // 4))
    rows = tile * 4 + gl.arange(0, 4, gl.SliceLayout(1, layout))
    k = group * 256 + gl.arange(0, 256, gl.SliceLayout(0, layout))
    x = gl.amd.cdna4.buffer_load(X, rows[:, None] * SX + k[None, :], rows[:, None] < LIVE_M, 0.0).to(gl.float32)
    x = gl.reshape(x, (4, 2, 128))
    scale = gl.maximum(gl.max(gl.abs(x), 2), 1e-10) * (1.0 / 448.0)
    value = (x * gl.div_rn(1.0, scale)[:, :, None]).to(A.dtype.element_ty)
    value = gl.reshape(value, (4, 256))
    gl.amd.cdna4.buffer_store(value, A, _packed_offset(rows[:, None], k[None, :], K), cache='.wt')
    sr = gl.convert_layout(rows, gl.SliceLayout(1, scale.type.layout))
    sg = group * 2 + gl.arange(0, 2, gl.SliceLayout(0, scale.type.layout))
    gl.amd.cdna4.buffer_store(scale, S, sg[None, :] * M + sr[:, None], cache='.wt')

@gluon.jit
def _norm_region(P, G, Y, RO, row, OFFSET: gl.constexpr, WIDTH: gl.constexpr, ROPE: gl.constexpr, EPS: gl.constexpr, SPLITS: gl.constexpr, N: gl.constexpr, PLANE: gl.constexpr, VEC: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [4], [0])
    d = gl.arange(0, triton.next_power_of_2(WIDTH + ROPE), layout)
    x = _merge(P, row * N + OFFSET + d, d < WIDTH + ROPE, SPLITS, PLANE)
    gamma = gl.load(G + d, d < WIDTH, 0.0, cache_modifier='.ca').to(gl.float32)
    norm_x = gl.where(d < WIDTH, x, 0.0)
    inv = gl.rsqrt(gl.sum(norm_x * norm_x, 0) / WIDTH + EPS)
    y = (x * inv * gamma).to(gl.bfloat16)
    gl.amd.cdna4.buffer_store(y, Y, row * WIDTH + d, d < WIDTH, cache='.wt')
    if ROPE:
        gl.amd.cdna4.buffer_store(x.to(gl.bfloat16), RO, row * ROPE + d - WIDTH, (d >= WIDTH) & (d < WIDTH + ROPE), cache='.wt')

@gluon.jit
def _finish(P, QG, KG, QO, KO, RO, M: gl.constexpr, Q: gl.constexpr, KV: gl.constexpr, R: gl.constexpr, EPS: gl.constexpr, SPLITS: gl.constexpr):
    N: gl.constexpr = Q + KV + R
    pid = gl.program_id(0).to(gl.uint32)
    row, is_q = (pid // 2, pid % 2 == 0)
    KV_VEC: gl.constexpr = 4
    if is_q:
        _norm_region(P, QG, QO, RO, row, 0, Q, 0, EPS, SPLITS, N, M * N, 8)
    else:
        _norm_region(P, KG, KO, RO, row, Q, KV, R, EPS, SPLITS, N, M * N, KV_VEC)

def mla_qkv_a_norm(hidden: torch.Tensor, weight: torch.Tensor, q_gamma: torch.Tensor, kv_gamma: torch.Tensor, weight_scales: torch.Tensor | None=None, *, rope_dim: int=64, eps: float=1e-05, weight_preshuffled: bool=False):
    m, k = hidden.shape
    q, kv = (q_gamma.numel(), kv_gamma.numel())
    n = q + kv + rope_dim
    assert 1 <= m <= 256 and k == 6144 and ((q, kv, rope_dim) == (2048, 512, 64))
    assert weight_preshuffled and weight_scales is not None
    assert weight.dtype == torch.float8_e4m3fn
    bm, splits = (64, 6)
    mp = triton.cdiv(m, bm) * bm
    qo = torch.empty((m, q), device=hidden.device, dtype=torch.bfloat16)
    ko = torch.empty((m, kv), device=hidden.device, dtype=torch.bfloat16)
    ro = torch.empty((m, rope_dim), device=hidden.device, dtype=torch.bfloat16)
    a = torch.empty((mp, k), device=hidden.device, dtype=weight.dtype)
    scales = torch.empty((k // 128, mp), device=hidden.device, dtype=torch.float32)
    partials = torch.empty((splits, mp, n), device=hidden.device, dtype=torch.float32)
    _quantize_wide[mp // 4 * (k // 256),](hidden, a, scales, mp, k, hidden.stride(0), m, num_warps=1)
    _project_lds[mp // bm * (n // 64) * splits,](a, scales, weight, weight_scales, partials, mp, n, k, bm, splits, num_warps=4, waves_per_eu=0)
    _finish[m * 2,](partials, q_gamma, kv_gamma, qo, ko, ro, mp, q, kv, rope_dim, eps, splits, num_warps=4, enable_fp_fusion=False)
    return (qo, ko, ro)
