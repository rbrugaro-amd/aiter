# [Triton/Gluon] [Feature] Fused DSA indexer prepare for block-FP8 indexer weights (gfx950)

## Summary

New op `aiter.ops.triton.attention.dsa_indexer_prepare.dsa_indexer_prepare` (Gluon, gfx950, Triton >= 3.8). One
call runs the decode step a DSA indexer performs before its top-k logits when its `wq_b` / `wk` weights are
block-FP8 (GLM-5.x FP8 checkpoints): the two block-FP8 projections with their 1x128 activation quant, the bf16
`weights_proj` projection, K LayerNorm, interleaved RoPE, the 128-point Hadamard rotation, ue8m0 FP8 quant, the
preshuffled index-K cache write and the head gate weights. Two launches replace six. `key_only=True` runs the K
projection and the cache write alone. `dsa_indexer_prepare_supported()` reports `(ok, reason)`; an unsupported
call raises, so a caller keeps its unfused chain (the contract of `fused_gdn_decode_qkvz`).

## Motivation

GLM-5.2 FP8 decodes its DSA indexer at 1..32 rows per step. On gfx950 with block-FP8 indexer weights the chain is,
per layer: `per_1x128` quant + CK `gemm_a8w8_blockscale_bpreshuffle` for `wq_b`, the same pair for `wk`,
`tgemm` for `weights_proj`, then `indexer_qk_rope_quant_and_cache` (or a framework-side rope/Hadamard/quant
kernel). The projections are tiny (M <= 32; N = 128, 4096, 32; K = 6144, 2048) and the chain is launch-bound:
25-37 us for six launches against 4.7-9.4 us for the fused pair.

No aiter op owns the projections together with the epilogue. `indexer_qk_rope_quant_and_cache`
(`aiter/ops/cache.py`) starts from the projected q/k/gate and omits the Hadamard rotation. The fused op writes the
Hadamard-rotated index-K that the gfx950 fused indexer decode path reads.

## Changes

- `aiter/ops/triton/_gluon_kernels/gfx950/attention/dsa_indexer_prepare.py`: `_projections` (single-wave
  16 x BN x 128*bundle split-K tiles; fp8 `mfma_scaled` with the activation quant fused, bf16 MFMA for the gate),
  `_packed_key` (key-only: up to 4 rows x 4 K-groups per MFMA tile), `_finish` (per (row, head): partial merge,
  LayerNorm, RoPE, Hadamard, ue8m0 quant, cache write / q and weights). 32-bit buffer addressing when every operand
  is below 2 GiB, int64 otherwise.
- `aiter/ops/triton/attention/dsa_indexer_prepare.py`: `dsa_indexer_prepare`, `dsa_indexer_prepare_supported`,
  config lookup through `resolve_config_dir` / `load_config_json` / `select_leq_config`.
- `aiter/ops/triton/configs/gfx950/gluon/attention/dsa_indexer_prepare/DEFAULT.json`: new family. `full` has the
  projection ladder per row bucket (`M_LEQ_2`, `M_LEQ_4`, `M_LEQ_8`, `M_LEQ_16`, `any`); `key_only` has
  one `any` entry.
- `op_tests/triton_tests/attention/test_dsa_indexer_prepare.py` and
  `op_tests/op_benchmarks/triton/bench_dsa_indexer_prepare.py` (see Testing).

API: one new public wrapper and its support probe under `aiter/ops/triton/attention/`, placed like
`fused_gdn_decode_qkvz` / `paged_attention_output_gate`. No existing signature changes. Weights are the standard
`shuffle_weight(w, layout=(16, 16))` bytes; the cache is the layout `indexer_k_quant_and_cache(..., preshuffle=True)`
writes, so both writers can share a pool.

Gating: gfx950 (`arch_info`), Triton >= 3.8 (the `fused_gdn_decode_qkvz` version probe), 1 <= T <= 32, K and QK
multiples of 512, 32 or 64 heads, fp8 e4m3fn shuffled weights with fp32 128x128 block scales, bf16 activations and
rope tables, int64 positions and slots, `[pages, 64, 132]` one-byte cache. Everything else is rejected with a reason.

## Performance

MI355X (gfx950), Triton 3.8.0, torch 2.11.0+rocm7.2. Device time by CUDA-graph replay (1000-call graphs, 3 blocks x
15 rounds, alternating arm order, median of block medians). `hot` = one input set; `rotating` = a 32-copy ring of
activations, weights, scales, rope tables and slots, so inputs come from HBM. Baseline = the unfused aiter chain
described above (key-only: quant + GEMM, `layernorm2d_fwd`, aiter RoPE, `indexer_k_quant_and_cache`), run from
upstream main `ca7cfa1c3` in its own process and from this branch's checkout in the same process as the op. Shapes:
GLM-5.2 indexer (K 6144, QK 2048, 32 heads); cache 34.6 MB (32-bit addressing) and 2.15 GB (int64 addressing).

| rows | path | cache | pool | branch op us | aiter chain us (main ca7cfa1c3) | speedup | chain us (branch checkout, same process) | speedup (min-max over 3 blocks) |
|---|---|---|---|---:|---:|---:|---:|---|
| 2 | full | hot | 34.6 MB | 4.68 | 24.62 | 5.26x | 24.60 | 5.26 (5.20-5.27) |
| 2 | full | rotating | 34.6 MB | 6.05 | 35.41 | 5.85x | 35.43 | 5.85 (5.75-5.87) |
| 4 | full | hot | 34.6 MB | 5.13 | 25.21 | 4.91x | 24.96 | 4.86 (4.86-4.88) |
| 4 | full | rotating | 34.6 MB | 6.94 | 35.71 | 5.15x | 35.72 | 5.15 (5.08-5.20) |
| 8 | full | hot | 34.6 MB | 5.30 | 26.35 | 4.97x | 26.17 | 4.94 (4.89-5.05) |
| 8 | full | rotating | 34.6 MB | 7.25 | 36.76 | 5.07x | 36.96 | 5.10 (5.06-5.15) |
| 16 | full | hot | 34.6 MB | 6.26 | 25.84 | 4.13x | 25.72 | 4.11 (4.10-4.18) |
| 16 | full | rotating | 34.6 MB | 8.34 | 34.64 | 4.15x | 34.75 | 4.17 (4.15-4.27) |
| 32 | full | hot | 34.6 MB | 8.18 | 26.89 | 3.29x | 26.71 | 3.27 (3.26-3.28) |
| 32 | full | rotating | 34.6 MB | 9.41 | 33.70 | 3.58x | 33.10 | 3.52 (3.50-3.55) |
| 2 | key-only | hot | 34.6 MB | 4.03 | 17.29 | 4.29x | 17.57 | 4.35 (4.34-4.44) |
| 2 | key-only | rotating | 34.6 MB | 4.81 | 19.88 | 4.13x | 20.10 | 4.18 (4.13-4.19) |
| 32 | key-only | hot | 34.6 MB | 4.42 | 19.62 | 4.44x | 19.88 | 4.50 (4.43-4.64) |
| 32 | key-only | rotating | 34.6 MB | 5.88 | 21.78 | 3.70x | 22.26 | 3.78 (3.70-3.82) |
| 2 | full | hot | 2.15 GB | 4.72 | 24.52 | 5.20x | 24.63 | 5.22 (5.18-5.26) |
| 2 | full | rotating | 2.15 GB | 6.13 | 35.55 | 5.80x | 35.54 | 5.80 (5.77-5.83) |
| 4 | full | hot | 2.15 GB | 4.75 | 25.10 | 5.28x | 25.24 | 5.31 (5.23-5.32) |
| 4 | full | rotating | 2.15 GB | 6.16 | 35.82 | 5.82x | 35.73 | 5.80 (5.77-5.82) |
| 8 | full | hot | 2.15 GB | 5.32 | 25.94 | 4.88x | 26.15 | 4.92 (4.88-4.94) |
| 8 | full | rotating | 2.15 GB | 7.20 | 37.17 | 5.16x | 36.99 | 5.14 (5.07-5.23) |
| 16 | full | hot | 2.15 GB | 5.99 | 25.60 | 4.27x | 25.60 | 4.27 (4.26-4.33) |
| 16 | full | rotating | 2.15 GB | 7.89 | 34.83 | 4.41x | 34.94 | 4.43 (4.42-4.43) |
| 32 | full | hot | 2.15 GB | 8.91 | 27.15 | 3.05x | 27.42 | 3.08 (3.07-3.10) |
| 32 | full | rotating | 2.15 GB | 9.30 | 29.07 | 3.12x | 29.24 | 3.14 (3.14-3.15) |
| 2 | key-only | hot | 2.15 GB | 4.08 | 17.57 | 4.31x | 17.30 | 4.24 (4.22-4.35) |
| 2 | key-only | rotating | 2.15 GB | 5.10 | 20.06 | 3.93x | 19.77 | 3.87 (3.87-4.30) |
| 32 | key-only | hot | 2.15 GB | 4.45 | 19.57 | 4.39x | 19.98 | 4.49 (4.33-4.62) |
| 32 | key-only | rotating | 2.15 GB | 6.02 | 22.54 | 3.75x | 23.00 | 3.82 (3.80-3.92) |

Rotating geomean 4.47x over the 14 cells, minimum 3.12x (32 rows, 2.15 GB cache); hot 4.43x, minimum 3.05x.
Per-block spread <= 0.1x on every cell.

`op_tests/op_benchmarks/triton/bench_dsa_indexer_prepare.py --eager` (RotatingBufferBench, 4096-page cache; eager =
host wall time per call):

| rows | path | fused us | unfused chain us | speedup | fused eager us | unfused eager us |
|---|---|---:|---:|---:|---:|---:|
| 1 | full | 5.59 | 31.41 | 5.62 | 72.0 | 130.4 |
| 2 | full | 5.73 | 30.87 | 5.39 | 72.9 | 172.6 |
| 4 | full | 6.42 | 31.73 | 4.94 | 72.7 | 171.1 |
| 8 | full | 6.29 | 32.01 | 5.09 | 73.2 | 171.9 |
| 16 | full | 7.30 | 30.53 | 4.18 | 73.1 | 172.8 |
| 32 | full | 8.57 | 31.97 | 3.73 | 73.6 | 173.7 |
| 1 | key-only | 3.91 | 24.25 | 6.20 | 46.7 | 70.3 |
| 2 | key-only | 4.06 | 24.34 | 6.00 | 46.9 | 71.3 |
| 4 | key-only | 4.26 | 24.80 | 5.83 | 47.2 | 70.6 |
| 8 | key-only | 5.02 | 25.27 | 5.03 | 47.1 | 70.9 |
| 16 | key-only | 5.25 | 23.72 | 4.51 | 46.7 | 70.9 |
| 32 | key-only | 5.75 | 24.90 | 4.33 | 47.7 | 71.3 |

## Numerics

Reference: fp64 projections after the kernel's 1x128 activation quant, fp32 elementwise, bf16 rounding where every
implementation materializes bf16. Pass bound: dequantized fp8 value within 32/448 of the row amax (one e4m3 code at
the top binade); untouched cache bytes identical; weights at rtol 1e-2 with ue8m0 scale flips matched. Draws: benign,
padded (-1) slots, cancellation (x = [a, -a], weight halves equal up to 1/128 nudged codes), both. The op passes every
cell (max k/q error 0.016 x amax). The unfused chain passes k and q but its bf16 gate GEMM (`tgemm`, N = 32,
K = 6144) leaves 8-15% of gate weights beyond rtol 1e-2 on benign draws; the fused op's fp32 split-K gate does not.

## Testing

- [x] Unit tests added or updated
- [x] Performance benchmarks run
- [ ] Tested on MI250X (gfx950-only op)
- [ ] Tested on MI300X (gfx950-only op; `_supported` reports the reason there)

## Documentation

- [x] Docstrings updated
- [ ] User guide updated
- [ ] Performance guide updated
- [x] No further documentation change is needed: the op follows the existing Gluon op layout and config conventions.

## Dependencies

- [x] No new third-party dependencies added

## Breaking Changes

None.

## PR Self-Review

### Reviewed revision

2c285e7d7509b082185ebf969c66943f80ac0760

### Behavior and intent

Adds one fused Gluon op for the block-FP8 DSA indexer decode prepare on gfx950; nothing existing changes behavior.
The unfused aiter chain is six launches and 25-37 us per layer at decode batch sizes; the fused pair is 4.7-9.4 us
and also produces exactly rounded fp32 gate weights where the chain's bf16 GEMM does not.

### Validation and unverified scope

Commands (gfx950, Triton 3.8.0, torch 2.11.0+rocm7.2):
- `python -m pytest op_tests/triton_tests/attention/test_dsa_indexer_prepare.py -q`: 40 passed.
- `python op_tests/op_benchmarks/triton/bench_dsa_indexer_prepare.py --eager` and `--key-only --eager`: tables above.
- Independent graph-replay A/B of the op against the chain from upstream main `ca7cfa1c3` in a separate process
  (14 cells, 3 + 2 blocks; tables above) and a numerics pass on 14 shapes x 4 draws.
- `black --check` and `ruff check` on the five changed files: clean.
Unverified: gfx942 / gfx1250 (the op rejects them); heads = 64 is accepted by `_supported` but was only tested with
32 heads; T > 32 is rejected and was not measured; the baseline's `layernorm2d_fwd` module was built against the
CK tree of the benchmark image, not the pinned submodule (affects only the key-only baseline timing, not the op).

### PR-specific question

Which exact workload keys consume these configuration changes, and what evidence shows that unrelated shapes keep their
previous selection? The new file `configs/gfx950/gluon/attention/dsa_indexer_prepare/DEFAULT.json` is read only by
`dsa_indexer_prepare.py` (`resolve_config_dir("attention", "DSA_INDEXER_PREPARE", backend="gluon")`). Its keys are
the row buckets `full.M_LEQ_2 / M_LEQ_4 / M_LEQ_8 / M_LEQ_16 / any` (T = 1-2, 3-4, 5-8, 9-16, 17-32) and
`key_only.any`, selected with `select_leq_config` on the token count. No existing config file, loader or op is
touched, so no other shape or op can change selection; `git diff --stat ca7cfa1c3` shows only the five new files.

### Changed op tests

- `op_tests/triton_tests/attention/test_dsa_indexer_prepare.py` (new): reference self-tests; full and key-only
  correctness at 1, 2, 3, 4, 8, 12, 16, 32 rows x benign and cancellation draws with padded slots; cache layout
  against `indexer_k_quant_and_cache(preshuffle=True)`; pad rows skip the cache write; int64 vs 32-bit addressing
  bit-equality on a 2 GiB cache; preallocated outputs; determinism; `_supported` rejections. Not redundant: no
  other test covers a fused projection + indexer epilogue.
- `op_tests/op_benchmarks/triton/bench_dsa_indexer_prepare.py` (new): rotating-buffer graph benchmark of the fused
  op against the unfused aiter chain, `--key-only` and `--eager` options.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
