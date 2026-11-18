"""glm52_fp8 qkv_a_norm small decode (round 3, triage item #59).

Verbatim copy of glm52_fp8/qkv_a_norm/decode_shared_m8.py (= round-1 regime
`small`) except _SCHEDULE[16].  The original file's M = 16 plan was never
selected (the manifest routes M = 16 to decode_shared_m2_m16.py) and is a
different arithmetic (FN 16, BF16 exchange, packed buffers; ~75% of elements
differ).  It is replaced by decode_shared_m2_m16.py's _SCHEDULE[16] expressed in
this file's plan vocabulary: _project_stream (-> _decode._project_stream_exchange)
with FN = 32, BF16_EXCHANGE = False, tile_n 64, groups/wave 1, 8 waves,
activation vec 16, q/kv shards 4/2, scratch alignment 256, BF16 partials,
waves_per_eu 4; finish with NARROW_ROPE / BUFFER_* / UNSIGNED off, KV_VEC 0
(-> 4); unpacked work buffers.  M = 1, 2, 4, 8 plans are unchanged.
"""
from typing import NamedTuple
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from xbench.s1f32.qkv_a_norm._decode import _axes3, _diagonal_two, _packed_offset, _project_two, _quantize_block, _store_partial, _project_stream_exchange

@gluon.jit
def _diagonal_eight(dot, M: gl.constexpr, BN: gl.constexpr):
    blocks = dot.reshape((8, M, 8, BN)).permute((0, 1, 3, 2))
    even, odd = gl.split(blocks.reshape((8, M, BN, 4, 2)))
    ae, cg = gl.split(even.reshape((8, M, BN, 2, 2)))
    bf, dh = gl.split(odd.reshape((8, M, BN, 2, 2)))
    a, e = gl.split(ae)
    c, g = gl.split(cg)
    b, f = gl.split(bf)
    d, h = gl.split(dh)
    layout: gl.constexpr = a.type.layout
    group = gl.arange(0, 8, gl.SliceLayout(1, gl.SliceLayout(2, layout)))[:, None, None]
    if M == 2:
        bit0, bit1, bit2 = (group & 1 != 0, group & 2 != 0, group & 4 != 0)
        low = gl.where(bit1, gl.where(bit0, d, c), gl.where(bit0, b, a))
        high = gl.where(bit1, gl.where(bit0, h, g), gl.where(bit0, f, e))
        selected = gl.where(bit2, high, low)
    else:
        choices = (a, b, c, d, e, f, g)
        selected = h
        for index in gl.static_range(6, -1, -1):
            selected = gl.where(group == index, choices[index], selected)
    return selected

@gluon.jit
def _project_native(X, W, WS, P, M: gl.constexpr, PITCH: gl.constexpr, K: gl.constexpr, SX: gl.constexpr, BN: gl.constexpr, PACK: gl.constexpr, WAVES: gl.constexpr, AV: gl.constexpr):
    gl.static_assert(WAVES == 1 and PACK == 8 and (M <= 2))
    BM: gl.constexpr = M * PACK
    al: gl.constexpr = gl.BlockedLayout([1, AV], [BM, 64 // BM], [1, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([32, 1], [4, 16], [1, 1], [1, 0])
    ml: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 128], False, [1, 1])
    ra = gl.arange(0, BM, gl.SliceLayout(1, al))
    ka = gl.arange(0, 128, gl.SliceLayout(0, al))
    kb = gl.arange(0, 128, gl.SliceLayout(1, bl))
    nb = gl.arange(0, PACK * BN, gl.SliceLayout(0, bl))
    rc = gl.arange(0, BM, gl.SliceLayout(1, ml))
    tile, split = (gl.program_id(0), gl.program_id(1))
    first = split * PACK
    xo = ra[:, None] % M * SX + ra[:, None] // M * 128 + ka[None, :]
    wo = _packed_offset(nb[None, :] // BN * 128 + kb[:, None], nb[None, :] % BN, K)
    x = gl.amd.cdna4.buffer_load(X + first * 128, xo).to(gl.float32)
    a, sa = _quantize_block(x, W.dtype.element_ty)
    b = gl.amd.cdna4.buffer_load(W + tile * BN * K + first * 2048, wo)
    if M == 1:
        sw = gl.amd.cdna4.buffer_load(WS + tile * BN // 128 * (K // 128) + first, rc // M)
    else:
        sw = gl.load(WS + tile * BN // 128 * (K // 128) + first + rc // M)
    dot = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, gl.DotOperandLayout(0, ml, 32)).to(gl.uint8, bitcast=True), None, 'e4m3', gl.convert_layout(b, gl.DotOperandLayout(1, ml, 32)).to(gl.uint8, bitcast=True), None, 'e4m3', gl.full((BM, PACK * BN), 0, gl.float32, ml))
    dot = _diagonal_eight(dot, M, BN)
    dl: gl.constexpr = dot.type.layout
    sa = gl.convert_layout(sa.reshape((PACK, M)), gl.SliceLayout(2, dl))
    sw = gl.convert_layout(sw.reshape((PACK, M)), gl.SliceLayout(2, dl))
    value = gl.sum(dot * sa[:, :, None] * sw[:, :, None], 0)
    _store_partial(value, P, tile, split, M, PITCH, BN, CACHE='.wt' if M == 2 else None)

@gluon.jit
def _project_stream(X, W, WS, P, M: gl.constexpr, PITCH: gl.constexpr, K: gl.constexpr, SX: gl.constexpr, BN: gl.constexpr, PACK: gl.constexpr, WAVES: gl.constexpr, AV: gl.constexpr, FN: gl.constexpr, BF16_EXCHANGE: gl.constexpr=False):
    _project_stream_exchange(X, W, WS, P, M, PITCH, K, SX, BN, PACK, WAVES, AV, FN, BF16_EXCHANGE)

@gluon.jit
def _finish_row(P, G, Y, RO, row, shard, M: gl.constexpr, PITCH: gl.constexpr, OFFSET: gl.constexpr, WIDTH: gl.constexpr, TAIL: gl.constexpr, EPS: gl.constexpr, PARTS: gl.constexpr, SHARDS: gl.constexpr, VEC: gl.constexpr, NARROW_TAIL: gl.constexpr=False, BUFFER_GAMMA: gl.constexpr=False, BUFFER_OUTPUT: gl.constexpr=False):
    LOAD_WIDTH: gl.constexpr = WIDTH if NARROW_TAIL else WIDTH + TAIL
    B: gl.constexpr = triton.next_power_of_2(LOAD_WIDTH)
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [4], [0])
    d = gl.arange(0, B, layout)
    acc = gl.full((B,), 0, gl.float32, layout)
    if NARROW_TAIL:
        rope_layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
        RB: gl.constexpr = triton.next_power_of_2(TAIL)
        rd = gl.arange(0, RB, rope_layout)
        racc = gl.full((RB,), 0, gl.float32, rope_layout)
    for part in gl.static_range(PARTS):
        acc += gl.amd.cdna4.buffer_load(P + part * M * PITCH + row * PITCH + OFFSET, d, d < LOAD_WIDTH, 0).to(gl.float32)
        if NARROW_TAIL:
            racc += gl.amd.cdna4.buffer_load(P + part * M * PITCH + row * PITCH + OFFSET + WIDTH, rd, rd < TAIL, 0).to(gl.float32)
    x = acc.to(gl.bfloat16).to(gl.float32)
    if not NARROW_TAIL:
        if BUFFER_GAMMA:
            gamma = gl.amd.cdna4.buffer_load(G, d, d < WIDTH, 0).to(gl.float32)
        else:
            gamma = gl.load(G + d, d < WIDTH, 0).to(gl.float32)
    norm_x = gl.where(d < WIDTH, x, 0.0)
    inv = gl.rsqrt(gl.sum(norm_x * norm_x, 0) / WIDTH + EPS)
    owned = d < WIDTH
    if SHARDS > 1:
        owned = owned & (d // gl.cdiv(WIDTH, SHARDS) == shard)
    if NARROW_TAIL:
        gamma = gl.amd.cdna4.buffer_load(G, d, owned, 0).to(gl.float32)
        gl.amd.cdna4.buffer_store((x * inv * gamma).to(gl.bfloat16), Y + row * WIDTH, d, owned, cache='.wt')
        gl.amd.cdna4.buffer_store(racc.to(gl.bfloat16), RO + row * TAIL, rd, (rd < TAIL) & (shard == 0), cache='.wt')
    else:
        if BUFFER_OUTPUT:
            gl.amd.cdna4.buffer_store((x * inv * gamma).to(gl.bfloat16), Y + row * WIDTH, d, owned, cache='.wt')
        else:
            gl.store(Y + row * WIDTH + d, (x * inv * gamma).to(gl.bfloat16), owned, cache_modifier='.wt')
        if TAIL:
            rope_owned = (d >= WIDTH) & (d < WIDTH + TAIL)
            if SHARDS > 1:
                rope_owned = rope_owned & (shard == 0)
            gl.store(RO + row * TAIL + d - WIDTH, x.to(gl.bfloat16), rope_owned, cache_modifier='.wt')

@gluon.jit
def _finish(P, QG, KG, QO, KO, RO, M: gl.constexpr, PITCH: gl.constexpr, Q: gl.constexpr, KV: gl.constexpr, R: gl.constexpr, EPS: gl.constexpr, PARTS: gl.constexpr, QS: gl.constexpr, KS: gl.constexpr, NARROW_ROPE: gl.constexpr, BUFFER_Q_GAMMA: gl.constexpr=False, UNSIGNED: gl.constexpr=False, BUFFER_Q_OUTPUT: gl.constexpr=False, KV_VEC: gl.constexpr=0):
    row, role = (gl.program_id(0), gl.program_id(1))
    if UNSIGNED:
        row, role = (row.to(gl.uint32), role.to(gl.uint32))
    if M == 1:
        row = 0
    if role < QS:
        _finish_row(P, QG, QO, RO, row, role, M, PITCH, 0, Q, 0, EPS, PARTS, QS, 8, False, BUFFER_Q_GAMMA, BUFFER_Q_OUTPUT)
    else:
        _finish_row(P, KG, KO, RO, row, role - QS, M, PITCH, Q, KV, R, EPS, PARTS, KS, KV_VEC if KV_VEC else 2 if NARROW_ROPE else 4, NARROW_ROPE)

class _Plan(NamedTuple):
    producer: object
    tile_n: int
    groups_per_wave: int
    waves: int
    activation_vec: int
    q_shards: int = 1
    kv_shards: int = 1
    scratch_alignment: int = 1
    partial_dtype: torch.dtype = torch.float32
    waves_per_eu: int = 0
    narrow_rope: bool = False
    fragment_n: int = 0
    packed: bool = False
    bf16_exchange: bool = False
    buffer_q_gamma: bool = False
    unsigned_finish: bool = False
    packed_outputs: bool = False
    buffer_q_output: bool = False
    kv_vec: int = 0
    packed_alignment: int = 1
_SCHEDULE = {1: _Plan(_project_native, 16, 8, 1, 4, scratch_alignment=16384, narrow_rope=True, waves_per_eu=5), 2: _Plan(_project_native, 16, 8, 1, 32, scratch_alignment=2048, narrow_rope=True, waves_per_eu=4, buffer_q_gamma=True, buffer_q_output=True, kv_vec=8, packed=True, packed_alignment=2048), 4: _Plan(_project_two, 32, 2, 8, 8, scratch_alignment=4096, partial_dtype=torch.float32, waves_per_eu=3, narrow_rope=True, fragment_n=16, packed=True), 8: _Plan(_project_stream, 64, 1, 8, 8, q_shards=8, kv_shards=2, scratch_alignment=16384, waves_per_eu=8, fragment_n=32, bf16_exchange=False, narrow_rope=True, packed=True, buffer_q_gamma=True), 16: _Plan(_project_stream, 64, 1, 8, 16, q_shards=4, kv_shards=2, scratch_alignment=256, waves_per_eu=4, fragment_n=32)}

def _work_buffers(device, partial_dtype, parts, m, pitch, q, kv, rope, packed, packed_outputs=False, packed_alignment=1):
    output_size = m * (q + kv + rope)
    if packed:
        pnum = parts * m * pitch
        factor = 2 if partial_dtype == torch.float32 else 1
        partial_start = triton.cdiv(output_size, packed_alignment) * packed_alignment
        storage = torch.empty((pnum * factor + partial_start,), device=device, dtype=torch.bfloat16)
        partial = storage[partial_start:].view(partial_dtype).view(parts, m, pitch)
        outputs = storage[:output_size]
    else:
        partial = torch.empty((parts, m, pitch), device=device, dtype=partial_dtype)
        if packed_outputs:
            outputs = torch.empty((output_size,), device=device, dtype=torch.bfloat16)
        else:
            qo = torch.empty((m, q), device=device, dtype=torch.bfloat16)
            ko = torch.empty((m, kv), device=device, dtype=torch.bfloat16)
            ro = torch.empty((m, rope), device=device, dtype=torch.bfloat16)
            return (partial, (qo, ko, ro))
    qo = outputs[:m * q].view(m, q)
    ko = outputs[m * q:m * (q + kv)].view(m, kv)
    ro = outputs[m * (q + kv):output_size].view(m, rope)
    return (partial, (qo, ko, ro))

def _run_plan(hidden, weight, q_gamma, kv_gamma, weight_scales, plan, rope_dim, eps):
    m, k = hidden.shape
    q, kv = (q_gamma.numel(), kv_gamma.numel())
    n = q + kv + rope_dim
    group_span = 128 * plan.groups_per_wave * plan.waves
    assert k % group_span == 0 and n % plan.tile_n == 0
    parts = k // group_span
    pitch = triton.cdiv(n, plan.scratch_alignment) * plan.scratch_alignment
    partial, (qo, ko, ro) = _work_buffers(hidden.device, plan.partial_dtype, parts, m, pitch, q, kv, rope_dim, plan.packed, plan.packed_outputs, plan.packed_alignment)
    producer_options = {}
    if plan.waves_per_eu:
        producer_options['waves_per_eu'] = plan.waves_per_eu
    if plan.fragment_n:
        producer_options['FN'] = plan.fragment_n
    if plan.producer is _project_stream:
        producer_options['BF16_EXCHANGE'] = plan.bf16_exchange
    plan.producer[n // plan.tile_n, parts](hidden, weight, weight_scales, partial, m, pitch, k, hidden.stride(0), plan.tile_n, plan.groups_per_wave, plan.waves, plan.activation_vec, num_warps=plan.waves, enable_fp_fusion=plan.waves != 1, **producer_options)
    _finish[m, plan.q_shards + plan.kv_shards](partial, q_gamma, kv_gamma, qo, ko, ro, m, pitch, q, kv, rope_dim, eps, parts, plan.q_shards, plan.kv_shards, plan.narrow_rope, plan.buffer_q_gamma, plan.unsigned_finish, plan.buffer_q_output, plan.kv_vec, num_warps=4, enable_fp_fusion=False)
    return (qo, ko, ro)

def mla_qkv_a_norm(hidden: torch.Tensor, weight: torch.Tensor, q_gamma: torch.Tensor, kv_gamma: torch.Tensor, weight_scales: torch.Tensor | None=None, *, rope_dim: int=64, eps: float=1e-05, weight_preshuffled: bool=False):
    assert weight_preshuffled and weight_scales is not None and (weight.dtype == torch.float8_e4m3fn)
    return _run_plan(hidden, weight, q_gamma, kv_gamma, weight_scales, _SCHEDULE[hidden.shape[0]], rope_dim, eps)
