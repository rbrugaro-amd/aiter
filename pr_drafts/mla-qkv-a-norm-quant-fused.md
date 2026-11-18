# [Triton/Gluon] Fused MLA qkv_a projection + q/kv RMSNorm for FP8 block-scaled weights (gfx950)

Branch: `mla-qkv-a-norm-quant-fused` on `rbrugaro-amd/aiter` (base: upstream/main `ca7cfa1c3`).
Commits: `569c84010` (op, Gluon kernels, config), `cae025695` (tests, benchmark, README).

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
one finish launch. The kernels are the GLM-5.2 FP8 `qkv_a_norm` schedules from the Artemis
kernel packs (`glm52_fp8/qkv_a_norm_r3`, served by the `glm52_fp8_ep8` pack and the
`glm52_fp8` TP4/TP8 packs), with one change: the split-K partials are **fp32**. The pack
stores them in bf16/fp16 and fails a cancellation draw (details below).

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
`fused_gdn_prefill_qkvz` / `paged_attention_output_gate_group_fp8_quant` precedent exactly:
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

## Performance

Hardware: MI355X (gfx950), ROCm 7.2.4, Triton 3.8.0, torch 2.11.0+rocm7.2, image
`nightly1117-triton38` (SGLang 703e763d5f). AITER base: upstream/main `ca7cfa1c3`; the chain
baselines use the same AITER copy as the op (branch) and were cross-checked against a copy
of upstream main (identical numbers, see below). Shape: GLM-5.2 FP8, K = 6144, N = 2624
(q 2048, kv 512, rope 64), bf16 activation, fp8 e4m3fn (16,16)-preshuffled weight with
128x128 fp32 block scales, eps 1e-5.

Device time by CUDA-graph replay (xbench: ABBA order, 15 rounds x 3 blocks, median; `A/A`
pairs within 2%). "Rotating": inputs rotate through a 256 MB ring so the 16 MB weight is
not cache resident (headline). "Hot": one input set.

- B1 = the AITER chain as SGLang 703e763d5f composes it (`aiter_w8a8_block_fp8_linear` ->
  tuned-CSV `gemm_a8w8_blockscale_bpreshuffle` -> `fused_qk_rmsnorm`, quant_type No).
- B2 = pure AITER chain (`get_hip_quant(per_1x128)` -> `gemm_a8w8_blockscale_bpreshuffle`
  -> `fused_qk_rmsnorm`), what the benchmark script's `--baseline` runs.
- A = this op (fp32 partials). T = the Artemis template (bf16 partials), for reference.

Rotating (us):

| M | B1 chain (SGLang) | B2 chain (AITER) | A fused (this PR) | speedup A vs B1 | T (bf16 partials) |
|---|---|---|---|---|---|
| 1 | 11.31 | 11.24 | 6.25 | 1.81x | 6.15 |
| 2 | 32.68 | 32.53 | 6.78 | 4.82x | 6.58 |
| 4 | 11.88 | 11.96 | 6.67 | 1.78x | 6.62 |
| 8 | 12.12 | 12.27 | 7.21 | 1.68x | 7.03 |
| 16 | 12.42 | 12.34 | 7.39 | 1.68x | 7.39 |
| 32 | 13.67 | 13.60 | 8.68 | 1.58x | 8.41 |

Geomean over the six M: **2.03x** rotating (1.58x worst), 2.69x hot.

Hot (us):

| M | B1 | B2 | A | speedup | T |
|---|---|---|---|---|---|
| 1 | 10.51 | 10.45 | 4.16 | 2.53x | 4.03 |
| 2 | 25.03 | 25.08 | 4.42 | 5.67x | 4.25 |
| 4 | 10.83 | 10.94 | 4.41 | 2.45x | 4.39 |
| 8 | 11.02 | 11.04 | 4.61 | 2.39x | 4.45 |
| 16 | 11.23 | 10.98 | 4.92 | 2.28x | 4.69 |
| 32 | 12.17 | 12.14 | 6.14 | 1.98x | 5.49 |

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

Baseline cross-check on a copy of upstream main (`ca7cfa1c3`, same node/GPU, block 1,
rotating): B1/B2 on the upstream-main copy
(`bench/results/S1pr/n38_b1_main.jsonl`, two appended runs, six records per arm; us):

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
rotating at every covered M. The op is `Unavailable` on the main copy (module absent),
which proves the two copies were distinct. B1/B2 numerics on main: pass at seeds 0 and 100.

Cost of fp32 partials (template with bf16 partials vs the same kernels with fp32 partials,
rotating): M <= 16 within 0-4% (6.15 -> 6.16 us at M = 1, 7.39 -> 7.42 at M = 16), 3% at
M = 32 (8.37 -> 8.65). The pack's M = 64..256 and prefill schedules lose 14-34% with fp32
partials (their partial traffic dominates) and are not in this PR; at those M the AITER
chain is also closer (1.4-1.8x) and the pack serves them with bf16 partials that fail the
cancellation draw.

## Numerics

Reference: fp64 torch with the same dynamic per-1x128 FP8 e4m3 quant (`scale = amax/448`),
bf16 round of the projection, fp32 RMSNorm. Tolerance `atol 2^-5, rtol 1e-2`
(`torch.testing.assert_close` semantics, as `test_fused_qk_rmsnorm_group_quant`).

Three draws per M: benign (randn), cancellation (`hidden = [x, -x]`, weight K-halves differ
by one fp8 code in 25% of entries: the split-K partials are large and cancel), wide (row
magnitudes 2^-12..2^12, norm weights 2^-4..2^4). xbench seeds 0 (benign) and 100
(cancellation) at M 1..32:

| | seed 0 | seed 100 (cancellation) |
|---|---|---|
| A (this PR) | pass, 6/6 | pass, 6/6 |
| B1 / B2 AITER chain | pass, 6/6 | pass, 6/6 |
| T (Artemis, bf16 partials) | pass, 6/6 | **FAIL 5/6** (M 1, 2, 8, 16, 32; max_abs 0.05-0.41 vs 0.03 tolerance) |

A and the AITER chain produce the same max_abs error against the reference at every shape
and seed (e.g. M = 8 seed 100: q 0.0258, kv 0.0132, k_pe 6.2e-5 for both); the unit test
`test_matches_aiter_chain` checks A against the chain directly.

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

xbench (this repo's harness): `bench/xbench/cands/S1pr.py` with `bench/s1glm_pr/{push,rn}.sh`
(`rn.sh pr|main <timeout> "python -m xbench.run --cand S1pr --block N --arms A,B1,B2,T
--numerics"`); raw results in `bench/results/S1pr/n38_b{1,2,3}_pr.jsonl` and
`n38_b1_main.jsonl`. fp32-partials experiment on the pack kernels: `cands/S1f32.py` +
`xbench/s1f32/`, results in `bench/results/S1f32/`.

## Attribution

Kernels ported from the Artemis GLM-5.2 FP8 kernel pack
(`src/kernel_packs/gfx950/glm52_fp8/qkv_a_norm_r3`, regimes `decode_shared_m8.py`,
`decode_shared_m32.py`, shared `_decode.py`), served by the `glm52_fp8_tp8_ep8_dp8_gfx950`
(artemis PR #33) and `glm52_fp8_tp{4,8}_gfx950` packs. Partials changed from bf16/fp16 to
fp32.

## Likely reviewer questions

- Why Gluon and gfx950 only: the producers use `mfma_scaled` (fp8 MFMA), CDNA4 buffer
  loads and explicit register layouts; same answer as `fused_gdn_prefill_qkvz`.
- Why M-exact schedules instead of M buckets: each M is a distinct register layout (M is a
  `constexpr` in the MFMA tiling). A decode batch that is not in {1, 2, 4, 8, 16, 32} is
  reported unsupported and the caller runs the chain; CUDA-graph capture batches of SGLang
  are powers of two in this range for GLM-5.2 DP8 decode (M 1..32 per rank).
- Why not also M 64..256 / prefill: with fp32 partials those schedules lose 14-34% and
  fall to 1.1-1.4x over the chain; left out until re-tuned.
- Does the bf16 round of the projection before the norm matter: it matches the unfused
  chain, where the GEMM writes bf16; the reference and the chain comparison both include it.
- The weight scale layout is row-major `[ceil(N/128), K/128]`, the same tensor the CK
  bpreshuffle GEMM takes; the activation scale is internal (no transpose_scale question).

## PR Self-Review

Reviewed revision: `cae025695`.

Behavior and intent: one new op; no existing behavior changes.

Validation and unverified scope: commands, hardware and results above. Unverified: other
architectures (the gate returns `(False, reason)`, tests skip), Triton 3.7 (gate returns
`False`, not executed), CU counts other than 256 (no CU tree; schedules tuned on 256 CU),
widths other than N = 2624 / K = 6144 (the gate accepts K, q, kv multiples of 128, but only
the GLM-5.2 shape was measured).

Changed op tests: `op_tests/triton_tests/fusions/test_fused_mla_qkv_a_proj_rmsnorm.py` (new)
covers the new op only: reference agreement on three draws at every covered M, agreement with
the AITER chain, determinism, support-probe reasons.

## Finding for the Artemis packs (not part of this PR)

The GLM-5.2 FP8 `qkv_a_norm` kernels store split-K partials in bf16/fp16 and the M = 8
schedule exchanges wave partials through LDS in bf16:
`glm52_fp8/qkv_a_norm_templated/decode_shared_m256.py:188,197` (bf16 partials, M 128/256),
`decode_shared_m32.py:119` and `decode_shared_m64.py:174` (fp16), `decode_shared_m12_248.py:63`
(bf16), `prefill_shared_m1024_4096.py:477-484` (bf16), `qkv_a_norm_r3/decode_shared_m8.py`
`_Plan.partial_dtype=bfloat16` (M 1, 2, 8, 16; M 4 already fp32) and `bf16_exchange=True`
(M 8). On the seed-100 cancellation draw this fails at M 1, 2, 8, 16, 32, 128, 256, 1024,
2048 (max_abs 0.27-0.67 vs tolerance 0.03); M 4, 64, 4096, 8193, 16384 pass. Switching
every partial buffer (and the M = 8 exchange) to fp32 makes all 14 shapes pass with no
other change (`bench/xbench/s1f32/`). Cost: <= 4% at M <= 32; 14% at M 64, 21-34% at
M 128/256, 19% at M 1024, 9% at M 2048 (rotating). The large-M schedules need a re-tune
(e.g. fewer splits, or fp32 only on the merge path) rather than a dtype switch.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
