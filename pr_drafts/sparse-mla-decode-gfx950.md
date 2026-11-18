# [Triton/Gluon] sparse_mla_fwd: 64-head fp8 decode kernels for GLM-5.2 EP8 on gfx950

Branch: `sparse-mla-decode-gfx950` (fork `rbrugaro-amd/aiter`), base upstream/main `ca7cfa1c3`, head `9b366a032` (4th commit = black/ruff formatting only).

## Summary

`sparse_mla_fwd` gets three Gluon kernels for the GLM-5.2 FP8 EP8 decode call: 64 query heads
per rank, fp8 e4m3 q, an fp8_scalar `[slots, 576]` cache, fp8 dots, and a `[C, 2048]` top-k
table padded with -1 (`has_invalid=True`). The call keeps the same signature. The new kernels
run for C in {1, 2, 4, 8, 12, 16, 24, 32} query rows on gfx950 with Triton >= 3.8; every other
call takes the path it took before.

## Motivation

GLM-5.2 FP8 at TP8/EP8/DP8 decodes with 64 heads per rank and a -1-padded `[C, 2048]` DSA
top-k table (SGLang `dsa_backend.py`). On upstream main this exact call runs the generic
`_sparse_mla` kernel: the dedicated `_sparse_mla_fp8` kernel is gated on `has_invalid=False`,
so the padded table never reaches it without a per-call compaction (what SGLang's
`_forward_aiter` does, about 10 launches). In addition, `sparse_mla_fwd` with fp8 dots fails
the bf16-criterion numerics check at 64 heads on wide-magnitude draws (per-head q
requantization), and SGLang's asm path (`mla_decode_fwd`) fails the same draws at H=64.

The three kernels come from the Artemis `glm52_fp8_ep8` kernel pack (sparse_paged_mla_xl:
`decode_ep8_m1_m8_x`, `decode_ep8_m12_m24_x`, `decode_ep8_m16_m32_x`), measured 1.36-2.6x
faster than upstream's fastest arm and numerically passing on both seeds.

## Changes

- `aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla_fp8_h64_small.py` (C 1-8),
  `..._split.py` (C 12, 24), `..._panel.py` (C 16, 32): one partials kernel per
  (row, head tile, key split) + one merge kernel. fp8 MFMA QK and rope, bf16 hi/lo (or fp16/bf16)
  PV, FP32 split partials. The kernels read `kv_indices[kv_indptr[row] : kv_indptr[row] + 2048]`,
  mask `-1` and positions past the row's count, apply `sm_scale * q_scale * kv_scale` to the
  scores and `kv_scale` to the output, and store into `out` through its strides.
- `aiter/ops/triton/attention/sparse_mla.py`: `_h64_decode_config` (shape/contract check +
  config lookup), `_sparse_mla_fwd_gfx950_fp8_h64` (launcher), route in `sparse_mla_fwd`
  gated on gfx950, fp8_scalar, fp8 dots, fp8 q, `has_invalid`, 512/64 geometry, no
  `kv_splits`/`skip_reduce`/`return_lse`/sink/extra/epilogue, C * 2048 indices, flat cache
  under 2^31 B, Triton >= 3.8 (`_gluon_supported`, same probe as
  `fused_gdn_decode_qkvz._gluon_supported`). A miss falls through to the existing kernels.
- `configs/gfx950/gluon/attention/sparse_mla/DEFAULT.json`: new `_sparse_mla_fp8_h64` family,
  one entry per row count (`M_1` ... `M_32`) with the kernel name and the partials/merge knobs.
- `op_tests/triton_tests/attention/test_sparse_mla.py`: `test_sparse_mla_fp8_h64_decode`
  (8 row counts x 3 draws: benign, wide-magnitude q with clustered keys and 5% holes,
  non-unit q/cache scales; one empty row), `..._out_buffer`, `..._fallback` (rows=3, bf16 q,
  bf16 dots, no sentinels, > 2 GB pool keep the previous path and stay correct).
- `op_tests/op_benchmarks/triton/bench_sparse_mla.py`: `--has_invalid` builds -1-padded
  `topk`-wide rows so the benchmark can reach the new path (`--num_heads 64 --has_invalid`).

## API impact

None. No new public function; `sparse_mla_fwd`'s signature and documented contract are
unchanged. The docstring of `has_invalid` names the new route.

## Gating and fallback

Same style as `fused_gdn_decode_qkvz` / `paged_attention_output_gate`: a cached
`_gluon_supported()` version probe (Triton >= 3.8). Below that, or for any shape / contract the
kernels do not take, `sparse_mla_fwd` runs exactly what upstream main runs today.

## Performance

MI355X (gfx950), Triton 3.8.0. Inputs: S4's GLM-5.2 EP8 draws (`glm52_fp8__sparse_paged_mla.make_inputs`,
seed 0), fp8 q, pool 262144 x 576 fp8 (151 MB), top-k 2048 with -1 padding, 64 heads. Device time by
CUDA-graph replay (xbench: 1000/200-call graphs, 3 blocks x 15 rounds pooled, ABBA order, A/A control `#2`).
`hot` = one input set; `rotating` = a 256-copy ring of q and slot tables (KV pool shared), so q/slots come
from HBM. Arms: **T** = this branch's `sparse_mla_fwd`; **Bm2** = upstream main `ca7cfa1c3`
`sparse_mla_fwd(dot_precision="fp8")` (its generic `_sparse_mla` kernel: the fp8 kernel is gated on
`has_invalid=False`); **Bm1** = upstream main with `dot_precision="bf16"` (its fastest arm that passes
numerics); **Tpk** = the Artemis pack template the kernels were ported from.

#### device time, graph replay (branch AITER; median of pooled rounds, us)

| rows | cache | T (branch) | T#2 (A/A) | Bm2 main fp8 dots | Bm2#2 | Bm1 main bf16 dots | Tpk (pack) | T vs Bm2 | T vs Bm1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | hot | 5.43 | 5.40 | 13.76 | 13.80 | 17.58 | 5.10 | 2.53x | 3.23x |
| 1 | rotating | 5.82 | 5.78 | 14.57 | 14.58 | 18.56 | 5.41 | 2.50x | 3.19x |
| 2 | hot | 6.73 | 6.60 | 13.37 | 13.37 | 17.09 | 6.14 | 1.99x | 2.54x |
| 2 | rotating | 7.08 | 7.04 | 14.08 | 14.06 | 18.35 | 6.45 | 1.99x | 2.59x |
| 4 | hot | 7.83 | 7.67 | 13.56 | 13.55 | 17.29 | 6.63 | 1.73x | 2.21x |
| 4 | rotating | 8.17 | 8.15 | 14.05 | 14.04 | 18.73 | 6.96 | 1.72x | 2.29x |
| 8 | hot | 9.91 | 9.84 | 13.38 | 13.35 | 17.25 | 8.84 | 1.35x | 1.74x |
| 8 | rotating | 10.31 | 10.23 | 14.11 | 14.12 | 18.42 | 9.19 | 1.37x | 1.79x |
| 12 | hot | 12.37 | 12.41 | 19.90 | 19.84 | 22.23 | 11.75 | 1.61x | 1.80x |
| 12 | rotating | 12.70 | 12.65 | 20.14 | 20.11 | 22.51 | 11.99 | 1.59x | 1.77x |
| 16 | hot | 14.42 | 14.40 | 21.30 | 21.11 | 23.74 | 14.19 | 1.48x | 1.65x |
| 16 | rotating | 14.81 | 14.75 | 21.69 | 21.59 | 24.08 | 14.56 | 1.46x | 1.63x |
| 24 | hot | 21.82 | 21.54 | 29.27 | 29.07 | 33.51 | 21.31 | 1.34x | 1.54x |
| 24 | rotating | 21.80 | 21.66 | 29.48 | 29.43 | 33.72 | 21.69 | 1.35x | 1.55x |
| 32 | hot | 20.87 | 20.84 | 28.45 | 28.37 | 33.15 | 18.59 | 1.36x | 1.59x |
| 32 | rotating | 21.16 | 21.18 | 28.88 | 28.78 | 33.82 | 19.22 | 1.37x | 1.60x |

rotating geomean T vs Bm2: 1.631x  (min 1.35x, n=8)
rotating geomean T vs Bm1: 1.985x  (min 1.55x)

The same call on an unmodified upstream-main copy (`/scratch/aiter_main_s4ep8`, `has_h64 False`): T
equals Bm2 on every row, so the in-process Bm2 arm is the true upstream baseline.

#### device time, graph replay (main-copy AITER; median of pooled rounds, us)

| rows | cache | T (branch) | T#2 (A/A) | Bm2 main fp8 dots | Bm2#2 | Bm1 main bf16 dots | Tpk (pack) | T vs Bm2 | T vs Bm1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | rotating | 14.60 | 14.57 | 14.58 | 14.60 | - | - | 1.00x | - |
| 2 | rotating | 14.08 | 14.07 | 14.08 | 14.10 | - | - | 1.00x | - |
| 4 | rotating | 14.06 | 14.04 | 14.08 | 14.06 | - | - | 1.00x | - |
| 8 | rotating | 14.16 | 14.15 | 14.15 | 14.16 | - | - | 1.00x | - |
| 12 | rotating | 20.13 | 20.24 | 20.14 | 20.16 | - | - | 1.00x | - |
| 16 | rotating | 21.79 | 21.73 | 21.71 | 21.79 | - | - | 1.00x | - |
| 24 | rotating | 29.32 | 29.46 | 29.44 | 29.29 | - | - | 1.00x | - |
| 32 | rotating | 28.83 | 28.86 | 28.84 | 28.81 | - | - | 1.00x | - |

rotating geomean T vs Bm2: 1.000x  (min 1.00x, n=8)

Eager wall time per call (host-bound at these sizes; 7 rounds x 50 calls, median; branch / upstream main):

| call | branch us | main us |
|---|---:|---:|
| H16 C=4 fp8 dots, no sentinels, bf16 q (not routed) | 64.4 | 66.6 |
| H64 C=8 fp8 dots, no sentinels, fp8 q (not routed) | 69.3 | 72.2 |
| H64 C=8 bf16 dots, sentinels, fp8 q (not routed) | 85.9 | 90.6 |
| H64 C=3 fp8 dots, sentinels, fp8 q (not routed: no config row) | 88.0 | 90.7 |
| H64 C=8 fp8 dots, sentinels, fp8 q (routed) | 51.6 | 91.0 |
| H64 C=1 fp8 dots, sentinels, fp8 q (routed) | 55.5 | 90.7 |

No host-side regression on the calls that keep their path; the routed call is cheaper to submit (two
launches, no split-K partials of the generic path). The launch cache (`load_config_json` LRU,
`_h64_rows` LRU) is kept.

T vs Tpk: the port is 0.3-2.3 us slower than the pack template on some rows (m4, m32). The one device-side
difference is that the kernels now read the row's index window from `kv_indptr[row]` (a dependent load
before the slot gathers) instead of a `row * 2048` offset. Still 1.35x+ over upstream everywhere.

## Numerics

fp32 reference over the dequantized cache (`-1` masked), bf16 criterion (`atol 2^-5`, `rtol 1e-2`, every
element). Seed 0 = benign draw; seed 100 = wide-magnitude q (8x) with a clustered key window and 5% holes.
T matches the pack template bit-for-bit on max-abs. Upstream's fp8-dot path (Bm2) fails seed 100 on every
row (its per-head q requantization); its bf16-dot path (Bm1) passes.

#### numerics vs fp32 reference (bf16 criterion atol 2^-5, rtol 1e-2): pass / max_abs / frac out of tol

| rows | seed | T | Tpk | Bm2 | Bm1 |
|---|---|---|---|---|---|
| 1 | 0 | PASS 0.0037 0.00e+00 | PASS 0.0037 0.00e+00 | PASS 0.0166 0.00e+00 | PASS 0.0052 0.00e+00 |
| 1 | 100 | PASS 0.0078 0.00e+00 | PASS 0.0078 0.00e+00 | FAIL 0.0372 3.05e-05 | PASS 0.0130 0.00e+00 |
| 2 | 0 | PASS 0.0037 0.00e+00 | PASS 0.0037 0.00e+00 | PASS 0.0166 0.00e+00 | PASS 0.0052 0.00e+00 |
| 2 | 100 | PASS 0.0086 0.00e+00 | PASS 0.0086 0.00e+00 | FAIL 0.0372 1.53e-05 | PASS 0.0142 0.00e+00 |
| 4 | 0 | PASS 0.0037 0.00e+00 | PASS 0.0037 0.00e+00 | PASS 0.0166 0.00e+00 | PASS 0.0052 0.00e+00 |
| 4 | 100 | PASS 0.0086 0.00e+00 | PASS 0.0086 0.00e+00 | FAIL 0.0484 2.29e-05 | PASS 0.0149 0.00e+00 |
| 8 | 0 | PASS 0.0037 0.00e+00 | PASS 0.0037 0.00e+00 | PASS 0.0166 0.00e+00 | PASS 0.0052 0.00e+00 |
| 8 | 100 | PASS 0.0086 0.00e+00 | PASS 0.0086 0.00e+00 | FAIL 0.0484 1.91e-05 | PASS 0.0149 0.00e+00 |
| 12 | 0 | PASS 0.0037 0.00e+00 | PASS 0.0037 0.00e+00 | PASS 0.0166 0.00e+00 | PASS 0.0052 0.00e+00 |
| 12 | 100 | PASS 0.0146 0.00e+00 | PASS 0.0146 0.00e+00 | FAIL 0.0658 6.61e-05 | PASS 0.0224 0.00e+00 |
| 16 | 0 | PASS 0.0037 0.00e+00 | PASS 0.0037 0.00e+00 | PASS 0.0166 0.00e+00 | PASS 0.0052 0.00e+00 |
| 16 | 100 | PASS 0.0146 0.00e+00 | PASS 0.0146 0.00e+00 | FAIL 0.0658 1.35e-04 | PASS 0.0224 0.00e+00 |
| 24 | 0 | PASS 0.0037 0.00e+00 | PASS 0.0037 0.00e+00 | PASS 0.0203 0.00e+00 | PASS 0.0052 0.00e+00 |
| 24 | 100 | PASS 0.0146 0.00e+00 | PASS 0.0146 0.00e+00 | FAIL 0.0658 1.70e-04 | PASS 0.0227 0.00e+00 |
| 32 | 0 | PASS 0.0037 0.00e+00 | PASS 0.0037 0.00e+00 | PASS 0.0203 0.00e+00 | PASS 0.0052 0.00e+00 |
| 32 | 100 | PASS 0.0146 0.00e+00 | PASS 0.0146 0.00e+00 | FAIL 0.0749 1.97e-04 | PASS 0.0189 0.00e+00 |

The new unit tests (`test_sparse_mla_fp8_h64_decode`) check 8 row counts x 3 draws (benign, wide, non-unit
q and cache scales) against the same fp32 reference at `rel_err < 2e-2`, the repo's bf16 tolerance for this
op; the previous fp8 path is held at `7e-2` in the existing tests.

## Testing

- [x] Unit tests added or updated (`test_sparse_mla.py`: 30 new cases, all pass; full
  `test_sparse_mla.py` suite: 172 passed, 2 skipped (pre-existing arch skips))
- [x] Performance benchmarks run (tables above)
- [ ] Tested on MI250X (not applicable: gfx950-only path)
- [ ] Tested on MI300X (gfx942 keeps its path; `_gluon_supported`/route never fires there)

## Documentation

- [x] Docstrings updated (`sparse_mla_fwd` `has_invalid`, kernel module docstrings)
- [ ] User guide updated
- [ ] Performance guide updated

## Dependencies

- [x] No new third-party dependencies added

## Breaking Changes

None.

## Environment

Image `nightly1117-triton38` (SGLang nightly 20261117): Triton 3.8.0, torch 2.11.0+rocm7.2,
MI355X (gfx950), GPU 2 of node crsuse2-m2m-v3-114. AITER branch overlaid from
`/scratch/aiter_pr_s4ep8`; upstream main `ca7cfa1c3` from `/scratch/aiter_main_s4ep8`
(`aiter.__file__` printed in every run).

## How to reproduce

```
# correctness
pytest -q op_tests/triton_tests/attention/test_sparse_mla.py -k h64
# device time, new path vs the generic kernel (flip has_invalid off to compare)
python op_tests/op_benchmarks/triton/bench_sparse_mla.py --num_heads 64 --num_seqs 1 2 4 8 12 16 24 32 --has_invalid
# the A/B harness used for the tables: templating-infra/aiter/bench/s4pr/{push,rn}.sh,
# xbench/cands/S4pr.py (arms T=branch, Bm2/Bm1=upstream main fp8/bf16 dots, Tpk=Artemis pack)
```

## Attribution

Kernels ported from the Artemis kernel packs, `glm52_fp8_ep8/sparse_paged_mla_xl`
(GLM-5.2 FP8 TP8/EP8/DP8 pack, artemis PR #33).

## PR Self-Review

### Reviewed revision

9b366a0329ea72cfef112d08f11ecdbd147aff76

### Behavior and intent

`sparse_mla_fwd` keeps its signature; one call pattern changes implementation: 64 query
heads, fp8 e4m3 q with a per-tensor scale, fp8_scalar `[slots, 576]` cache, fp8 dots,
`has_invalid=True` with a `[C, 2048]` -1-padded top-k table, C in {1, 2, 4, 8, 12, 16, 24,
32}, flat cache below 2^31 bytes, gfx950, Triton >= 3.8. That call (GLM-5.2 FP8 TP8/EP8/DP8
decode) previously ran the generic `_sparse_mla` kernel because the existing fp8 kernel is
gated on `has_invalid=False`; it now runs three dedicated Gluon kernels that are 1.35-2.50x
faster and pass the wide-magnitude numerics draw that the previous fp8 path fails. Every other
call runs exactly what upstream main runs today.

### Validation and unverified scope

Hardware: MI355X (gfx950), ROCm 7.2.4, Triton 3.8.0, torch 2.11.0+rocm7.2, AITER base
`ca7cfa1c3`. Commands: `pytest -q op_tests/triton_tests/attention/test_sparse_mla.py` (172
passed, 2 skipped: pre-existing arch skips; the new `-k h64` tests are 30 of them and were
re-run after the formatting commit: 30 passed) and
`python op_tests/op_benchmarks/triton/bench_sparse_mla.py --num_heads 64 --num_seqs 1 2 4 8 12 16 24 32 --has_invalid`.
Device time by CUDA-graph replay (rotating and hot inputs, ABBA, 3 x 15 rounds) against
upstream main's fp8 and bf16 paths, in-process and on a separate unmodified copy of upstream
main (Performance tables); numerics vs an fp32 reference on seeds 0 and 100 (Numerics table);
eager wall time of routed and not-routed calls. `black --check` and `ruff check` pass (ruff
0.17.5; CI pins 0.16.0). Unverified: caches >= 2^31 bytes (the route excludes them and they
fall to the generic kernel), Triton 3.9 (`_HAS_REINTERPRET` branch is the same switch
`sparse_mla_fp8.py` uses; not executed), architectures other than gfx950 (route never fires,
not executed), non-unit q/cache scales are tested for correctness but not benchmarked, and
end-to-end serving (SGLang does not call `sparse_mla_fwd` today).

### PR-specific question

Which exact workload keys consume these configuration changes, and what evidence shows that
unrelated shapes keep their previous selection? `configs/gfx950/gluon/attention/sparse_mla/DEFAULT.json`
gains one new top-level family `_sparse_mla_fp8_h64` with keys `M_1`, `M_2`, `M_4`, `M_8`,
`M_12`, `M_16`, `M_24`, `M_32` (row count C); only `_h64_decode_config` in `sparse_mla.py`
reads that family, and only after every contract check listed above passes. The existing
`_sparse_mla` entry in the same file is untouched (diff shows additions only), so bf16-q,
bf16-dot, H != 64, `has_invalid=False`, > 2 GB and non-listed-C calls select what they did
before: `test_sparse_mla_fp8_h64_fallback` asserts that five such calls do not route and stay
correct, the full existing `test_sparse_mla.py` suite passes unchanged, and the eager timing
of four not-routed calls matches upstream main within noise (Performance, eager table).

### Changed op tests

- `op_tests/triton_tests/attention/test_sparse_mla.py`: adds `test_sparse_mla_fp8_h64_decode`
  (8 row counts x 3 draws: benign, wide-magnitude q with clustered keys and 5% holes, non-unit
  q/cache scales, with one empty row), `test_sparse_mla_fp8_h64_out_buffer` (caller-provided
  `out` with strides) and `test_sparse_mla_fp8_h64_fallback` (five contracts that must not
  route and must stay correct). The existing tests never reach the new kernels (they use
  bf16 q or `has_invalid=False`), so this coverage is not redundant; no existing test changed.
- `op_tests/op_benchmarks/triton/bench_sparse_mla.py`: adds `--has_invalid`, which builds
  -1-padded `topk`-wide rows so the benchmark can reach the new route; default behavior is
  unchanged.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
