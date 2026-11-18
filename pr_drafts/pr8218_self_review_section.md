## PR Self-Review

### Reviewed revision

cae0256954aae92cac746b75351b36172b5a935b

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
