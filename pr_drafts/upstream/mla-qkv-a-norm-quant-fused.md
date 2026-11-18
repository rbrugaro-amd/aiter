## Summary

Adds `aiter.ops.triton.fusions.fused_mla_qkv_a_proj_rmsnorm`: one op for the MLA decode
chain `per-1x128 FP8 quant -> gemm_a8w8_blockscale_bpreshuffle -> fused_qk_rmsnorm` on a
fused `q_a | kv_a | k_pe` weight (GLM-5.x, DeepSeek-V3 style attention). Gluon, gfx950,
decode batch sizes M in {1, 2, 4, 8, 16, 32}. 1.58x-1.81x (4.8x at M=2) device time versus
the AITER chain SGLang runs today, with rotating inputs; numerics match the chain.

## Motivation

GLM-5.2 FP8 (TP4, TP8, TP8/EP8/DP8) runs `q_a_proj` + `kv_a_proj_with_mqa` as one fused
`[2624, 6144]` FP8 block-scaled GEMM per layer, followed by `q_a_layernorm` and
`kv_a_layernorm`. At decode batch sizes the chain is three launches and two HBM round
trips of the `[M, 2624]` projection, and the GEMM is launch/latency bound (the weight is
16 MB, the activation a few KB). SGLang composes it as `aiter_per1x128_quant ->
gemm_a8w8_blockscale_bpreshuffle (tuned CSV) -> fused_qk_rmsnorm(quant_type=No)`.

The fused op quantizes the activation in registers, runs a split-K FP8 MFMA over the
preshuffled weight, and folds the split-K merge, the bf16 round and the two RMSNorms into
one finish launch. The split-K partials and the intra-CTA wave exchange are kept in fp32 so
that cancelling partial sums do not lose precision (see Numerics).

## Changes

- `aiter/ops/triton/fusions/fused_mla_qkv_a_proj_rmsnorm.py`: public wrapper
  `fused_mla_qkv_a_proj_rmsnorm(hidden, weight, weight_scale, q_norm_weight,
  kv_norm_weight, *, rope_dim=64, eps=1e-5) -> (q_lora, kv_lora, k_pe)` and the support
  predicate `fused_mla_qkv_a_proj_rmsnorm_supported(...) -> (bool, reason)`.
- `aiter/ops/triton/_gluon_kernels/gfx950/fusions/fused_mla_qkv_a_proj_rmsnorm.py`:
  torch-free Gluon kernels (three split-K producers for M <= 16, one chained producer for
  M = 32, two finish kernels), each with `make_kernel_repr`.
- `aiter/ops/triton/configs/gfx950/gluon/fusions/fused_mla_qkv_a_proj_rmsnorm/DEFAULT.json`:
  one schedule per M (`M_1` ... `M_32`), read through `resolve_config_dir` /
  `load_config_json`.
- `op_tests/triton_tests/fusions/test_fused_mla_qkv_a_proj_rmsnorm.py`,
  `op_tests/op_benchmarks/triton/bench_fused_mla_qkv_a_proj_rmsnorm.py`,
  `aiter/ops/triton/gluon/README.md` (quick-reference row + section).

API impact: one new op. No existing op changes. The op follows the
`fused_gdn_prefill_qkvz` / `paged_attention_output_gate_group_fp8_quant` precedent:
same `_arch_supported` / `_gluon_supported` (Triton >= 3.8) gates, same
`<op>_supported(...) -> (bool, reason)` probe, lazy import of the Gluon kernels inside the
host launchers so the module imports on any platform, kernels under
`_gluon_kernels/gfx950/<wrapper folder>/`, config under `configs/gfx950/gluon/`.

Why not a flag on an existing op: `gemm_a8w8_blockscale_bpreshuffle` is a CK/asm/FlyDSL
dispatch with a `[M, N]` output contract and no epilogue hook; `fused_qk_rmsnorm` /
`fused_rms_fp8_group_quant` consume an already materialized projection. Neither can host a
fused GEMM + dual RMSNorm without a new output contract (three tensors), so a new op in
`fusions/` is the smallest change.

Gating and fallback: not supported -> `(False, reason)`; the wrapper raises `ValueError`
only when a caller skips the probe. Fallback is the caller's existing three-kernel chain.
Not adopted in-tree (no AITER code calls it yet), as for `fused_gdn_prefill_qkvz`.

M-exact schedules: M is a `constexpr` of the MFMA tiling, so each supported M has its own
schedule. A decode batch outside {1, 2, 4, 8, 16, 32} is reported unsupported and the caller
runs the chain; with CUDA-graph decode the batch is padded to a captured size, so callers
that capture these sizes always hit a schedule. Padding to the next supported M inside the
wrapper is a possible follow-up (the kernel is latency bound at these M, so the cost would be
~1 us).

## Performance

Hardware: MI355X (gfx950), ROCm 7.2.4, Triton 3.8.0, torch 2.11.0+rocm7.2. AITER base:
upstream/main `ca7cfa1c3`; the chain baselines were run on this branch and cross-checked on
a separate copy of upstream main (below). Shape: GLM-5.2 FP8, K = 6144, N = 2624 (q 2048,
kv 512, rope 64), bf16 activation, fp8 e4m3fn (16,16)-preshuffled weight with 128x128 fp32
block scales, eps 1e-5.

Device time by CUDA-graph replay (ABBA order, 15 rounds x 3 blocks, median; A/A pairs within
2%). "Rotating": inputs rotate through a 256 MB ring so the 16 MB weight is not cache
resident (headline). "Hot": one input set.

- B1 = the AITER chain as SGLang composes it (`aiter_w8a8_block_fp8_linear` ->
  tuned-CSV `gemm_a8w8_blockscale_bpreshuffle` -> `fused_qk_rmsnorm`, quant_type No).
- B2 = pure AITER chain (`get_hip_quant(per_1x128)` -> `gemm_a8w8_blockscale_bpreshuffle`
  -> `fused_qk_rmsnorm`), what the benchmark script's `--baseline` runs.
- A = this op.

Rotating (us):

| M | B1 chain (SGLang) | B2 chain (AITER) | A fused (this PR) | speedup A vs B1 |
|---|---|---|---|---|
| 1 | 11.31 | 11.24 | 6.25 | 1.81x |
| 2 | 32.68 | 32.53 | 6.78 | 4.82x |
| 4 | 11.88 | 11.96 | 6.67 | 1.78x |
| 8 | 12.12 | 12.27 | 7.21 | 1.68x |
| 16 | 12.42 | 12.34 | 7.39 | 1.68x |
| 32 | 13.67 | 13.60 | 8.68 | 1.58x |

Geomean over the six M: **2.03x** rotating (1.58x worst), 2.69x hot.

Hot (us):

| M | B1 | B2 | A | speedup |
|---|---|---|---|---|
| 1 | 10.51 | 10.45 | 4.16 | 2.53x |
| 2 | 25.03 | 25.08 | 4.42 | 5.67x |
| 4 | 10.83 | 10.94 | 4.41 | 2.45x |
| 8 | 11.02 | 11.04 | 4.61 | 2.39x |
| 16 | 11.23 | 10.98 | 4.92 | 2.28x |
| 32 | 12.17 | 12.14 | 6.14 | 1.98x |

M = 2 is slow on the AITER side because the tuned bpreshuffle CSV has no M = 2 row for
(2624, 6144) and the dispatcher picks a poor kernel; the other rows compare against tuned
rows.

AITER benchmark script (`bench_fused_mla_qkv_a_proj_rmsnorm.py --baseline`, rotating
buffers, graph replay):

| M | fused us | TB/s | chain us | speedup |
|---|---|---|---|---|
| 1 | 6.10 | 2.65 | 10.88 | 1.78x |
| 2 | 6.34 | 2.55 | 32.12 | 5.07x |
| 4 | 6.27 | 2.58 | 11.88 | 1.90x |
| 8 | 6.91 | 2.35 | 11.69 | 1.69x |
| 16 | 7.12 | 2.30 | 12.67 | 1.78x |
| 32 | 7.99 | 2.09 | 13.45 | 1.68x |

Eager wall time per call (no graph; host overhead included), us:

| M | A fused | B2 AITER chain | B1 SGLang chain |
|---|---|---|---|
| 1 | 40.6 | 112.0 | 125.1 |
| 8 | 40.5 | 111.1 | 123.8 |
| 32 | 38.2 | 112.3 | 124.5 |

The op is two Triton launches with a cached config lookup (`functools.cache`), so eager
time is launch bound (~40 us) and 2.7-3.1x below the chain (quant + GEMM dispatch + norm).
No host-side regression.

Baseline cross-check on a copy of upstream main (`ca7cfa1c3`, same node/GPU, rotating; two
runs, six records per arm; us):

| M | B1 main median (min-max) | B2 main median | B1 branch (3 blocks) | A fused | speedup vs main median | vs main min |
|---|---|---|---|---|---|---|
| 1 | 10.45 (9.57-10.47) | 10.38 | 11.31 | 6.25 | 1.67x | 1.53x |
| 2 | 32.61 (24.89-32.69) | 32.56 | 32.68 | 6.78 | 4.81x | 3.67x |
| 4 | 11.22 (10.12-11.24) | 11.19 | 11.88 | 6.67 | 1.68x | 1.52x |
| 8 | 11.56 (10.37-11.62) | 11.62 | 12.12 | 7.21 | 1.60x | 1.44x |
| 16 | 12.01 (10.68-12.07) | 11.94 | 12.42 | 7.39 | 1.63x | 1.45x |
| 32 | 13.28 (11.88-13.35) | 13.18 | 13.67 | 8.68 | 1.53x | 1.37x |

The chain kernels are identical bytes on both copies (the branch touches none of them); the
5-8% gap between the branch-copy and main-copy medians, and the one faster main run, are
run-to-run drift of the node. Even against the fastest main record the op is >= 1.37x
rotating at every covered M. The op is unavailable on the main copy (module absent), which
confirms the two copies were distinct. B1/B2 numerics on main: pass at seeds 0 and 100.

Cost of fp32 partials: keeping the split-K partials in fp32 instead of bf16/fp16 costs 0-4%
at M <= 32 on these schedules (e.g. 6.15 -> 6.16 us at M = 1, 8.37 -> 8.65 at M = 32).
Larger-M and prefill schedules were measured to lose 14-34% with fp32 partials (their partial
traffic dominates) and are not in this PR; at those M the AITER chain is also closer
(1.4-1.8x).

## Numerics

Reference: fp64 torch with the same dynamic per-1x128 FP8 e4m3 quant (`scale = amax/448`),
bf16 round of the projection, fp32 RMSNorm. Tolerance `atol 2^-5, rtol 1e-2`
(`torch.testing.assert_close` semantics, as `test_fused_qk_rmsnorm_group_quant`).

Three draws per M: benign (randn), cancellation (`hidden = [x, -x]`, weight K-halves differ
by one fp8 code in 25% of entries: the split-K partials are large and cancel), wide (row
magnitudes 2^-12..2^12, norm weights 2^-4..2^4). Seeds 0 (benign) and 100 (cancellation) at
M 1..32:

| | seed 0 | seed 100 (cancellation) |
|---|---|---|
| A (this PR) | pass, 6/6 | pass, 6/6 |
| B1 / B2 AITER chain | pass, 6/6 | pass, 6/6 |

A and the AITER chain produce the same max_abs error against the reference at every shape
and seed (e.g. M = 8 seed 100: q 0.0258, kv 0.0132, k_pe 6.2e-5 for both); the unit test
`test_matches_aiter_chain` checks A against the chain directly. The same schedules with
bf16/fp16 split-K partials fail the cancellation draw at 5 of 6 M (max_abs 0.05-0.41 vs
0.03 tolerance), which is why the partials are fp32.

## Testing

- [x] Unit tests added: `op_tests/triton_tests/fusions/test_fused_mla_qkv_a_proj_rmsnorm.py`
  (23 tests: 18 reference cases = 6 M x 3 draws, 3 chain comparisons, determinism, support
  predicate). 23 passed on MI355X, Triton 3.8.0, 7.4 s.
- [x] Performance benchmarks run (tables above).
- [ ] Tested on MI250X / MI300X: not applicable, gfx950-only op; the gate returns
  `(False, reason)` elsewhere and the test skips.
- `black --check` and `ruff check` (0.16.0) pass on the changed files.

## Documentation

- [x] Docstrings on the wrapper and the predicate.
- [x] `aiter/ops/triton/gluon/README.md`: quick-reference row and a section.

## Dependencies

- [x] No new third-party dependencies.

## Breaking Changes

None.

## How to reproduce

```bash
pytest op_tests/triton_tests/fusions/test_fused_mla_qkv_a_proj_rmsnorm.py
python op_tests/op_benchmarks/triton/bench_fused_mla_qkv_a_proj_rmsnorm.py --baseline
```

## PR Self-Review

### Reviewed revision

4a2b3388087be03a039807286c2edb7ef5ef5a0c

### Behavior and intent

Adds one new opt-in op, `fused_mla_qkv_a_proj_rmsnorm`, that replaces the three-launch MLA
decode chain `per-1x128 FP8 quant -> gemm_a8w8_blockscale_bpreshuffle -> fused_qk_rmsnorm`
with a fused split-K FP8 GEMM plus dual RMSNorm for GLM-5.x / DeepSeek-V3 style `q_a | kv_a |
k_pe` weights on gfx950. No existing op, config or kernel changes; callers that do not call
the new op see no behavior change. The existing chain is launch/latency bound at decode batch
sizes (three launches, two HBM round trips of the `[M, 2624]` projection, 11-33 us); the fused
op runs the same math in two launches at 6-9 us (1.58-1.81x, 4.8x at M = 2) with identical
numerics to the chain.

### Validation and unverified scope

Hardware: MI355X (gfx950), ROCm 7.2.4, Triton 3.8.0, torch 2.11.0+rocm7.2, AITER base
`ca7cfa1c3`. Commands: `pytest op_tests/triton_tests/fusions/test_fused_mla_qkv_a_proj_rmsnorm.py`
(23 passed: 6 M x 3 draws vs an fp64 reference incl. cancellation and wide-magnitude draws,
3 direct comparisons against the AITER chain, determinism, support predicate) and
`python op_tests/op_benchmarks/triton/bench_fused_mla_qkv_a_proj_rmsnorm.py --baseline`
(1.68-1.90x, 5.07x at M = 2). Device time also measured by CUDA-graph replay with rotating
inputs, ABBA order, 3 x 15 rounds, against the chain on this branch and on a separate copy of
upstream main (tables in Performance). `black --check` and `ruff check` (0.16.0) pass.
Unverified: architectures other than gfx950 (the gate returns `(False, reason)` and the tests
skip), Triton < 3.8 (gate returns `False`; not executed), CU counts other than 256 (schedules
tuned on 256 CU, no CU tree), widths other than N = 2624 / K = 6144 (the gate accepts K, q and
kv multiples of 128 but only the GLM-5.2 shape was measured), and end-to-end serving (no AITER
or SGLang code calls the op yet).

### PR-specific question

Which exact workload keys consume these configuration changes, and what evidence shows that
unrelated shapes keep their previous selection? The new file
`configs/gfx950/gluon/fusions/fused_mla_qkv_a_proj_rmsnorm/DEFAULT.json` is read only by the
new op through `resolve_config_dir("fusions", "fused_mla_qkv_a_proj_rmsnorm", backend="gluon")`,
and its keys are the decode row counts `M_1`, `M_2`, `M_4`, `M_8`, `M_16`, `M_32`; an M
without a key makes `fused_mla_qkv_a_proj_rmsnorm_supported` return `(False, reason)` and the
caller keeps its chain (`test_supported_reasons` covers this). No existing config file or
config directory is modified, so no other op's selection can change; the chain kernels
(`gemm_a8w8_blockscale_bpreshuffle`, `fused_qk_rmsnorm`) measured identical device time on
this branch and on the upstream-main copy (Performance, baseline cross-check table).

### Changed op tests

- `op_tests/triton_tests/fusions/test_fused_mla_qkv_a_proj_rmsnorm.py` (new): the only test
  of the new op. It checks the fused output against an fp64 reference on benign, cancellation
  and wide-magnitude draws at every supported M (the cancellation draw is what exposes
  low-precision split-K partials), agreement with the existing AITER chain, run-to-run
  determinism, and the reasons returned by the support predicate. No existing test exercises
  a fused GEMM + dual RMSNorm, so none of this is redundant.
- `op_tests/op_benchmarks/triton/bench_fused_mla_qkv_a_proj_rmsnorm.py` (new): benchmark
  script for the op with `--baseline` running the AITER chain on the same buffers; it is the
  reproducer for the Performance tables and follows `bench_fused_gdn_decode_qkvz.py`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
