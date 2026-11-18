# [Triton/Gluon] [Feature] Fused DSA indexer prepare for block-FP8 indexer weights (gfx950)

Branch: `glm52-fp8-indexer-prepare` (fork `rbrugaro-amd/aiter`), base upstream/main `ca7cfa1c3`.
Status: see "Push status" at the end.

## Summary

New op `aiter.ops.triton.attention.dsa_indexer_prepare.dsa_indexer_prepare` (Gluon, gfx950,
Triton >= 3.8): one call replaces the decode chain a DSA indexer runs before its top-k logits
when its `wq_b` / `wk` weights are block-FP8 (GLM-5.x; the same layout DeepSeek-V3.2 FP8
checkpoints use): the two block-FP8 projections with their 1x128 activation quant, the bf16
`weights_proj` projection, K LayerNorm, interleaved RoPE, the 128-point Hadamard rotation,
ue8m0 FP8 quant, the preshuffled index-K cache write and the head gate weights. Two launches
instead of six. `key_only=True` runs the K projection + cache write alone (SGLang's
`_forward_cuda_k_only`). Support is reported by `dsa_indexer_prepare_supported()`; unsupported
calls raise (same contract as `fused_gdn_decode_qkvz`) so a caller keeps its unfused chain.

## Motivation

GLM-5.2 FP8 (TP4, TP8, TP8/EP8/DP8) decodes its DSA indexer at 1..32 rows per step (CUDA-graph
batches; 32 at bs 8 x MTP). On gfx950 with block-FP8 indexer weights, SGLang (`dsa_indexer.py`,
`gfx950_fused_indexer.py`) runs per layer: `dynamic_per_group_scaled_quant` + CK
`gemm_a8w8_blockscale_bpreshuffle` for `wq_b`, the same pair for `wk`, `tgemm` for
`weights_proj`, then either `aiter::indexer_qk_rope_quant_and_cache` (AITER fused writer, no
Hadamard) or SGLang's own HIP `qk_rope_hadamard_quant_kernel`. The projections are tiny
(M <= 32, N = 128 / 4096 / 32, K = 6144 / 2048) and launch-bound: 25-36 us for the six
launches against 4.7-8.2 us for the fused pair (below).

No aiter op owns the projections together with the epilogue: `indexer_qk_rope_quant_and_cache`
(`aiter/ops/cache.py:123`) starts at the projected q/k/gate and omits the Hadamard. The op is a
port of the Artemis GLM-5.2 FP8 kernel pack (`glm52_fp8/indexer_prepare_xl/ipx_decode_template.py`,
served rows M 2/4/8/16 full and M 2 key-only of the `glm52_fp8_tp4`, `glm52_fp8_tp8` and
`glm52_fp8_tp8_ep8_dp8` profiles), with one finish form and a config table instead of the
pack's per-row files.

## Changes

- `aiter/ops/triton/_gluon_kernels/gfx950/attention/dsa_indexer_prepare.py`: `_projections`
  (single-wave 16 x BN x 128*bundle split-K tiles; fp8 `mfma_scaled` with the activation quant
  fused, bf16 MFMA for the gate), `_packed_key` (key-only: up to 4 rows x 4 K-groups per MFMA),
  `_finish` (per (row, head): partial merge, LayerNorm, RoPE, Hadamard, ue8m0 quant, cache
  write / q + weights). 32-bit buffer addressing when every operand is < 2 GiB, int64 otherwise.
- `aiter/ops/triton/attention/dsa_indexer_prepare.py`: `dsa_indexer_prepare`,
  `dsa_indexer_prepare_supported` (gfx950, Triton >= 3.8, shapes, dtypes, layouts), config
  lookup through `resolve_config_dir` / `load_config_json` / `select_leq_config`.
- `aiter/ops/triton/configs/gfx950/gluon/attention/dsa_indexer_prepare/DEFAULT.json`: the
  projection ladder per row bucket (`M_LEQ_2/4/8/16`, `any`) and the key-only tile.
- `op_tests/triton_tests/attention/test_dsa_indexer_prepare.py`: 8 row counts x 2 draws
  (benign, cancellation) for full and key-only, padded (-1) slots, int64 vs buffer arm
  bit-equality on a 2 GiB cache, cache layout vs `indexer_k_quant_and_cache(preshuffle=True)`,
  preallocated outputs, determinism, `_supported` rejections.
- `op_tests/op_benchmarks/triton/bench_dsa_indexer_prepare.py`: rotating-buffer graph bench,
  fused op vs the unfused aiter chain (`--key-only`).

## API impact

New public wrapper (one function + its `_supported` probe) under `aiter/ops/triton/attention/`,
following `fused_gdn_decode_qkvz` / `paged_attention_output_gate` (wrapper + `_gluon_kernels/
gfx950/attention/` bodies + `configs/gfx950/gluon/attention/<op>/DEFAULT.json` + test + bench).
No existing signature changes. Weights are the standard `shuffle_weight(w, layout=(16, 16))`
bytes; the cache is the layout `indexer_k_quant_and_cache(..., preshuffle=True)` writes.

## Gating and fallback

`dsa_indexer_prepare_supported()` -> `(ok, reason)`; `dsa_indexer_prepare()` raises
`ValueError(reason)` when not ok. Gates: gfx950 (`arch_info`), Triton >= 3.8 version probe
(same `_gluon_supported` as `fused_gdn_decode_qkvz`), 1 <= T <= 32, K and QK multiples of 512,
H in {32, 64}, fp8 e4m3fn shuffled weights with fp32 128x128 block scales, bf16 activations /
rope tables, int64 positions / slots, `[pages, 64, 132]` 1-byte cache. Everything else is the
caller's existing path.

## Performance

MI355X (gfx950), Triton 3.8.0. Device time by CUDA-graph replay (xbench: 1000-call graphs, 3 blocks x 15 rounds,
ABBA order, median of block medians). `hot` = one input set; `rotating` = a 32-copy ring of activations, weights,
scales, rope tables and slots (cache shared), so inputs come from HBM. Arms: **A** = this branch's
`dsa_indexer_prepare`; **B** = the unfused aiter chain SGLang runs (full: `per_1x128` quant + CK
`gemm_a8w8_blockscale_bpreshuffle` for wq_b and wk, `tgemm` for weights_proj, `indexer_qk_rope_quant_and_cache`;
key-only: quant + GEMM, `layernorm2d_fwd`, aiter RoPE, `indexer_k_quant_and_cache`, without the Hadamard (B1) = its
fastest configuration), run from upstream main `ca7cfa1c3` in its own process and from the branch checkout in the same
process as A; **T** = the Artemis template the kernels were ported from. Shapes: GLM-5.2 indexer (K 6144, QK 2048,
32 heads); pool `s` = 262144 slots (34.6 MB cache), `b` = 16269312 slots (2.15 GB cache, int64 addressing arm).
Inputs: `glm52_fp8` spec draws, seed 0.

#### device time (us)

| shape | cache | A (branch op) us | B chain us, main ca7cfa1c3 | A vs B main | B chain us, branch checkout | A vs B (min-max) | T template us | A vs T |
|---|---|---:|---:|---:|---:|---|---:|---:|
| m2s | hot | 4.68 | 24.62 | 5.26x | 24.60 | 5.26 (5.20-5.27) | 4.65 | 0.99 |
| m2s | rotating | 6.05 | 35.41 | 5.85x | 35.43 | 5.85 (5.75-5.87) | 6.06 | 1.00 |
| m4s | hot | 5.13 | 25.21 | 4.91x | 24.96 | 4.86 (4.86-4.88) | 4.86 | 0.95 |
| m4s | rotating | 6.94 | 35.71 | 5.15x | 35.72 | 5.15 (5.08-5.20) | 6.55 | 0.94 |
| m8s | hot | 5.30 | 26.35 | 4.97x | 26.17 | 4.94 (4.89-5.05) | 5.34 | 1.01 |
| m8s | rotating | 7.25 | 36.76 | 5.07x | 36.96 | 5.10 (5.06-5.15) | 7.15 | 0.99 |
| m16s | hot | 6.26 | 25.84 | 4.13x | 25.72 | 4.11 (4.10-4.18) | 6.11 | 0.98 |
| m16s | rotating | 8.34 | 34.64 | 4.15x | 34.75 | 4.17 (4.15-4.27) | 8.06 | 0.97 |
| m32s | hot | 8.18 | 26.89 | 3.29x | 26.71 | 3.27 (3.26-3.28) | 5.69 | 0.70 |
| m32s | rotating | 9.41 | 33.70 | 3.58x | 33.10 | 3.52 (3.50-3.55) | 8.16 | 0.87 |
| ko2s | hot | 4.03 | 17.29 | 4.29x | 17.57 | 4.35 (4.34-4.44) | 3.93 | 0.97 |
| ko2s | rotating | 4.81 | 19.88 | 4.13x | 20.10 | 4.18 (4.13-4.19) | 4.63 | 0.96 |
| ko32s | hot | 4.42 | 19.62 | 4.44x | 19.88 | 4.50 (4.43-4.64) | 3.76 | 0.85 |
| ko32s | rotating | 5.88 | 21.78 | 3.70x | 22.26 | 3.78 (3.70-3.82) | 4.45 | 0.76 |
| m2b | hot | 4.72 | 24.52 | 5.20x | 24.63 | 5.22 (5.18-5.26) | 4.69 | 0.99 |
| m2b | rotating | 6.13 | 35.55 | 5.80x | 35.54 | 5.80 (5.77-5.83) | 6.18 | 1.01 |
| m4b | hot | 4.75 | 25.10 | 5.28x | 25.24 | 5.31 (5.23-5.32) | 5.04 | 1.06 |
| m4b | rotating | 6.16 | 35.82 | 5.82x | 35.73 | 5.80 (5.77-5.82) | 6.62 | 1.07 |
| m8b | hot | 5.32 | 25.94 | 4.88x | 26.15 | 4.92 (4.88-4.94) | 5.29 | 1.00 |
| m8b | rotating | 7.20 | 37.17 | 5.16x | 36.99 | 5.14 (5.07-5.23) | 7.19 | 1.00 |
| m16b | hot | 5.99 | 25.60 | 4.27x | 25.60 | 4.27 (4.26-4.33) | 5.94 | 0.99 |
| m16b | rotating | 7.89 | 34.83 | 4.41x | 34.94 | 4.43 (4.42-4.43) | 7.75 | 0.98 |
| m32b | hot | 8.91 | 27.15 | 3.05x | 27.42 | 3.08 (3.07-3.10) | 202.71 | 22.75 |
| m32b | rotating | 9.30 | 29.07 | 3.12x | 29.24 | 3.14 (3.14-3.15) | 249.55 | 26.82 |
| ko2b | hot | 4.08 | 17.57 | 4.31x | 17.30 | 4.24 (4.22-4.35) | 4.10 | 1.00 |
| ko2b | rotating | 5.10 | 20.06 | 3.93x | 19.77 | 3.87 (3.87-4.30) | 4.84 | 0.95 |
| ko32b | hot | 4.45 | 19.57 | 4.39x | 19.98 | 4.49 (4.33-4.62) | 74.75 | 16.78 |
| ko32b | rotating | 6.02 | 22.54 | 3.75x | 23.00 | 3.82 (3.80-3.92) | 104.21 | 17.32 |

rotating: uniform geomean A vs B (branch checkout) 4.47x over 14 shapes, minimum 3.14x (m32, 2.15 GB cache); hot 4.43x,
minimum 3.08x. A vs B (upstream main process): 3.05x-5.85x. Per-block speedup spread <= 0.1 on every cell; T#2 / B#2 A/A
controls <= 0.5% hot. A vs T: parity (0.94-1.07) on the served M 2..16 rows and key-only M 2; 0.70-0.87 at M 32 on the
small pool (the pack's dedicated M=32 file is faster there); 17-27x at M 32 on the served 2.15 GB pool, where the pack
falls back to its compat kernels.

#### `op_tests/op_benchmarks/triton/bench_dsa_indexer_prepare.py --eager` (branch checkout; RotatingBufferBench graph replay, us; eager = host wall per call incl. launches, 200 calls)

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

Eager wall time: the fused op costs 72-74 us (full) / 47 us (key-only) per call on the host against 130-174 / 70-71 us
for the chain, so no host-side regression; the Gluon launch itself (two kernels) is the bulk of the 72 us.

Full per-arm records: `bench/results/S6fp8/n38_b1[123]_pr.jsonl` (branch), `n38_b2[12]_main.jsonl` (main),
`bench/results/S6fp8_pr.md`. The earlier template-vs-nightly table (T vs AITER 581579ab5, image AITER) is
`bench/results/S6fp8.md`.

## Numerics

Criterion (as S6): fp64 projections after the kernel's own 1x128 activation quant, fp32 elementwise, bf16 rounding at the
materialization points (GEMM output, after norm, after RoPE, after Hadamard); dequantized fp8 value vs reference
<= 32/448 of the row amax (one e4m3 code at the top binade); untouched cache bytes identical; weights = bf16(gate) *
q_scale * weights_scale at rtol 1e-2, with ue8m0 scale flips (2x) matched and an fp32 accumulation-order allowance of
2^-16 x sum|x w| on the cancellation draw. Draws: seeds 0 (benign), 3 (padded -1 slots), 100 (cancellation: x = [a, -a],
weights' halves equal up to 1/128 nudged codes), 103 (both).

- Branch op (A): PASS on all 14 shapes x 4 seeds (k cache, q, weights, untouched bytes); k/q max error <= 0.016 x amax.
- Template (T): PASS on the same 56 cells.
- aiter chain (B, main and branch checkout): k cache and q PASS; gate weights fail the strict bf16 rtol on 8-15% of
  entries on benign draws and 13-64% on the cancellation draws on every full shape. Root cause is the chain's bf16 gate
  GEMM (`tgemm` hgemm_bf16 for N=32, K=6144), the same finding as S6 (not correctly rounded bf16 output); the fused op's
  fp32 split-K gate passes. Key-only: every arm passes.
- pytest `op_tests/triton_tests/attention/test_dsa_indexer_prepare.py`: 40 passed (branch checkout, image Triton 3.8.0).

## Environment

Image `nightly1117-triton38` (SGLang nightly 20261117 base): torch 2.11.0+rocm7.2, Triton 3.8.0,
flydsl 0.5.0 (added for the main checkout), AITER = this branch (`/scratch/aiter_pr_idxfp8`) and
upstream main `ca7cfa1c3` (`/scratch/aiter_main_idxfp8`), both built with their own JIT
modules (`aiter.__file__` printed in every run). GPU 3 of an MI355X (gfx950) node
(crsuse2-m2m-v3-114, job 296860).

## How to reproduce

```
# correctness
python -m pytest op_tests/triton_tests/attention/test_dsa_indexer_prepare.py -q
# device time, fused vs the unfused aiter chain
python op_tests/op_benchmarks/triton/bench_dsa_indexer_prepare.py --tokens 1,2,4,8,16,32
python op_tests/op_benchmarks/triton/bench_dsa_indexer_prepare.py --tokens 2,32 --key-only
```
Artemis-side A/B (templates vs this op vs upstream main, inputs with sha256 manifests):
`templating-infra/aiter/bench/xbench/cands/S6fp8.py`, results `bench/results/S6fp8*`.

## Attribution

Kernels ported from the Artemis GLM-5.2 FP8 kernel packs (`glm52_fp8/indexer_prepare_xl/
ipx_decode_template.py` and its owners `indexer_prepare/decode_shared_m2_m8.py`, `_decode.py`,
`decode_memory.py`), served by the `glm52_fp8_tp8_ep8_dp8_gfx950` profile (artemis PR #33) and
the shared `glm52_fp8_tp4/tp8` profiles.

## Likely reviewer questions

- Why a new op and not a flag on `indexer_qk_rope_quant_and_cache`? That op is HIP (`csrc/`), takes the already
  projected q/k/gate and omits the Hadamard; the fusion here is the projections + epilogue in one Gluon pair. Adding it
  to the HIP op would mix backends in one PR.
- Why Gluon / Triton >= 3.8 only, and gfx950 only? `mfma_scaled` e4m3 MFMAs, CDNA4 buffer loads and explicit layouts;
  same gate and fallback contract as `fused_gdn_decode_qkvz`.
- M > 32? Not enabled: the single-wave projection tiles were measured for decode rows 1..32; `_supported` rejects
  larger T so a caller keeps its GEMM chain for prefill.
- Weight layout: the standard `shuffle_weight(layout=(16, 16))` bytes, read in place (no copy); non-shuffled weights
  are rejected rather than reshuffled per call.
- Hadamard always on: this is the format the gfx950 SGLang fused indexer writes and reads; a no-Hadamard mode (the AITER
  fused-writer configuration) is a one-constexpr addition if wanted.
- The 72 us eager wall time per call: two Gluon launches through `JITFunction.run`; graph capture (the decode case)
  removes it. A launch cache like `attn_res` could follow if eager decode matters.
- CK submodule: the baseline's `layernorm2d_fwd` module (key-only chain) was built against the image's CK
  (96afae1d) because the pinned CK commit was not fetchable from the bench host; it affects only the B/B1 key-only
  baseline, not the op.

## Push status

Pushed: `git push origin glm52-fp8-indexer-prepare` (3 commits 2c285e7d7, c5129bece, 62d7cf959 on ca7cfa1c3). All gates
passed: pytest 40/40, numerics PASS on every cell, >= 3.05x vs upstream main on every enabled shape (gate 1.10x),
no eager regression, black/ruff clean. Upstream-facing description: `pr_drafts/upstream/glm52-fp8-indexer-prepare.md`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
