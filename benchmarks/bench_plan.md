# Phase 2 GPU bench plan: shortlisted templates vs AITER

This plan covers the 12 shortlisted candidates (S1-S12) and the conditional ones (C1-C4) from
`assessment.md`. It runs on one MI355X (gfx950, 256 CUs) across **two containers**:

| container | image | what runs there | stack |
|---|---|---|---|
| **N** ("nightly") | `lmsysorg/sglang-rocm:v0.5.23-rocm724-mi35x-20261117`, as is, with no AITER rebuild | AITER arms (B, B+) and stock-SGLang arms (S) | AITER 581579ab5 (2026-11-07, amd-aiter 0.1.29.dev156) at `/sgl-workspace/aiter`, torch 2.11.0+rocm7.2, **Triton 3.7.0**, SGLang 703e763d5f |
| **O** ("ours") | `artemis-templating:rocm724` = `lmsysorg/sglang:v0.5.22-rocm724-mi35x` + Triton 3.8.0+gitdf3f91dd | template arms (T); optional arm B38 | torch 2.11 + ROCm 7.2.4, **Triton 3.8**. Our templates are gated on Triton 3.8 (105 of 106 are Gluon in the 3.8 dialect) |

Path conventions:
- `wt_next/` = the `rbrugaro/templating-gfx950-next` checkout at `88081cdc`.
- `nightly/` = the AITER package of image N. Its source snapshot is
  `templating-infra/aiter/aiter_from_sglang_nightly_20261117/aiter/`.
- `op_tests/` = AITER's tests in image N.

Every AITER call below was checked against the 581579ab5 snapshot. Top-level `aiter.X` names are
the ones re-exported by `nightly/__init__.py:79-209`; everything else is given with its module
path. Signatures are unchanged from e7d2453f2 (2026-09-25, the phase-1 copy), except where a
candidate section says "nightly note". **Before running, confirm that
`git -C /sgl-workspace/aiter rev-parse HEAD` in image N starts with `581579ab5`.**

**Bench order (user priority).** All fused and non-GEMM candidates run first. All plain GEMMs
run after them. Part C has the exact order.

---

## Part A: comparison methodology (applies to every candidate)

### A0. Two containers, one harness

The templates (container O) and AITER/stock (container N) cannot share a process: the templates
need Triton 3.8, and image N must stay as shipped. So one standalone harness, `xbench/`, runs
unchanged in both containers on the **same GPU**, and the results are merged afterwards.

**Layout.** `xbench/` is mounted read-only into both containers, together with `wt_next/` and a
shared results dir:
- `xbench/inputs.py`: deterministic input builders per candidate and shape. They are thin
  wrappers around the family-spec `make_inputs` (`wt_next/tools/kernel-templating/specs/*.py`).
- `xbench/arms_T.py`: template arms, imported only in O. Load them with
  `src/bench/common.py:14 load_kernel()` plus the manifest sha256.
- `xbench/arms_N.py`: AITER and stock arms, imported only in N.
- `xbench/arms_B38.py`: optional, imported only in O; see below.
- `xbench/timing.py`: a copy of `wt_next/src/bench/common.py:63-104 measure()`, extended to k
  arms and R rounds.
- `xbench/run.py --container {O,N} --candidate S4 --shapes ... --cache {hot,rotating} --block <i> --out results/<cand>/<container>_<block>.jsonl`
- `xbench/merge.py`: joins the two containers and computes speedups, noise and numerics.

**Inputs are identical bytes in both containers.**
- Generate every input once, in container O, on CPU with a fixed seed. Save it as
  `inputs/<cand>/<shape>.pt` with a sha256 manifest.
- Both containers `torch.load` the file, check the sha256, and `.to("cuda")`.
- Weights are preshuffled with AITER's `shuffle_weight` in container N. This is a one-time
  layout step, not timed. Save the result so that both containers can also load it.
- Rotating rings (A3) are built from these tensors by `.clone()` in each container, with the same
  ring length.

**Same timing method in both containers.**
- Graph replay of `count` calls per graph, `torch.cuda._sleep` before the start event, CUDA-event
  timing, median over R=15 rounds.
- ABBA alternation **within** a container: (B, S, B', S'...) in N; (T, T') in O.
- Containers alternate at **block** level: O1, N1, O2, N2, O3, N3. Run at least 3 blocks per
  container, one container at a time, never concurrently, and keep `HIP_VISIBLE_DEVICES` pinned
  to the same physical GPU.
- Each block restarts the container process, so the JIT/graph state is fresh. Compile and warm up
  before the timed rounds.

**Controls.**
- **A/A per container.** O: T vs T, in two independently captured graphs. N: B vs B. Run it in
  every block. The noise floor of a cross-container ratio is
  `sqrt(spread_O^2 + spread_N^2) + calibration drift`.
- **Cross-container calibration.** Each block runs the same two reference kernels in both
  containers:
  - (i) `torch.Tensor.copy_` of a 1 GiB bf16 buffer, bandwidth-bound;
  - (ii) `torch.mm` bf16 8192^3, compute-bound. hipBLASLt may differ between images, so record
    the version from each image (`/opt/rocm/.info/version`, `hipblaslt` package version).

  Report the O/N ratio of each reference per block. If the copy kernel differs by more than 2%
  between containers in the same block, the block is invalid: re-run after a cool-down.
- **Ordering bias.** Alternate which container goes first in each pair of blocks
  (O-N, N-O, O-N).

**Cross-container caveats to record and report.**
- **Clock and power state.** Record `rocm-smi --showclocks --showpower --showtemp --showperflevel`
  before and after each block. Keep the default perf level (do not pin clocks unless both
  containers use the same setting). Start a block only when the edge temperature is within 5 C of
  the first block's start temperature.
- **Runtime libraries differ.** Both images use torch 2.11+rocm7.2, but the ROCm user-space
  libraries (hipBLASLt, rocBLAS, comgr) and the HIP runtime build may differ. Record
  `torch.version.hip` and `torch.version.git_version` in each. Under graph replay, host-side
  differences mostly vanish, but kernel launch latency inside a graph can differ by runtime.
  That is why the copy-kernel calibration is required, and why sub-3 us kernels need
  `count` >= 2000.
- **Triton differs** (3.7 in N vs 3.8 in O). It only affects Triton/Gluon arms: our T arms and
  AITER's Triton/Gluon B arms. CK, asm, FlyDSL and hipBLASLt AITER kernels do not depend on
  Triton. For AITER Triton kernels, the comparison is against what image N actually ships, which
  is what a user of that image gets.
- **AITER ops ported from Artemis are disabled in N.** `fused_gdn_decode_qkvz`,
  `fused_gdn_prefill_qkvz` and `paged_attention_output_gate_*` require Triton >= 3.8
  (`nightly/ops/triton/gated_delta_net/fused_gdn_decode_qkvz.py:89-114`,
  `nightly/ops/triton/attention/paged_attention_output_gate.py:94`). In N their `*_supported()`
  probe returns False, so the "AITER as shipped" arm for those computations is the multi-kernel
  fallback chain.
- **Optional arm B38** ("AITER's own op on Triton 3.8", container O). It shows whether the
  upstream op already matches us.
  - Vendor these pure-Python modules from the nightly snapshot into `xbench/aiter_ports/`,
    rewriting `from aiter.` to `from xbench.aiter_ports.`:
    - the three op wrappers, their Gluon kernel modules
      (`_gluon_kernels/gfx950/gated_delta_net/fused_gdn_{decode,prefill}_qkvz*.py`,
      `_gluon_kernels/gfx950/attention/paged_attention_output_gate*.py`);
    - their helper imports: `utils/_triton/arch_info.py`, `utils/_triton/kernel_repr.py`,
      `utils/logger.py`, `_triton_kernels/common/activation.py`, `_gluon_kernels/common/utils.py`,
      and `_triton_kernels/rope/rope.py`.
  - Phase 1 checked this import set: no compiled AITER module is needed. Run them under Triton
    3.8 in O.
  - Report B38 separately. It is not "AITER as shipped".
- **Stock arms (S) use SGLang 703e763d5f in N.** Our templates were validated against SGLang
  0.5.18/0.5.22-era stock, so a stock arm can differ from the stock in `docs/`. That is the
  point: we compare against today's stock.

**Merge rules.**
- speedup = median_over_blocks(B_N) / median_over_blocks(T_O). Report the min and max over
  blocks as the spread.
- A cell is "win" only if speedup - 1 > 2 x noise floor and the numerics pass in **both**
  containers. Numerics are compared on outputs saved to `results/<cand>/outputs_<container>.pt`,
  against the same fp32/fp64 reference computed once in O.

### A1. What is compared

Every (candidate, shape) cell has up to five arms. Arms T, B and A0 are mandatory; S, B+ and B*
apply where noted.

| arm | what runs | notes |
|---|---|---|
| **T** (container O) | the template's host entry, loaded from `wt_next/` | Load it with `src/bench/common.py:14 load_kernel(path, entrypoint, expected_sha256)`, taking the sha256 from the manifest row. The entry name is in the candidate section. |
| **B** (container N) | the AITER op that serving would call for the **same computation**, at AITER's tuned/default config, as shipped in image N (no rebuild) | For GEMMs, record which backend AITER dispatched to (CK, cktile, asm, flydsl, hipBLASLt, skinny or Triton). Run with `AITER_LOG_TUNED_CONFIG=1` (or the logger in `nightly/tuned_gemm.py` / `gemm_op_a8w8.py`) and record whether the (M,N,K) row was found in the merged tuned CSVs (`nightly/configs/*.csv` + `nightly/configs/model_configs/*.csv`, merged per `nightly/jit/core.py:805-830`). |
| **B+** (container N) | GEMMs whose shape is **not** in any tuned CSV. The tuner writes CSVs only; AITER is not rebuilt | Run AITER's tuner for exactly the served shapes, point the matching env var at the result, and re-run B. Env vars: `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE[_BPRESHUFFLE]` (`nightly/jit/core.py:175`) and `AITER_CONFIG_GEMM_BF16` (`:225`). Tuners: the CK blockscale and bpreshuffle tune scripts under `/sgl-workspace/aiter/csrc/ck_gemm_a8w8_blockscale*/` and gradlib for bf16 (locate them in the container). Report B and B+; the claim is made against B+. |
| **S** (container N) | the stock serving sequence (SGLang 703e763d5f) for fused templates (every op SGLang runs today, in order), and the best AITER composition of the same stages if that is faster | Each candidate lists its ops. Where SGLang's stock path is not AITER (TileLang sparse MLA, FLA GDN, native attention residual), include that op as its own arm. |
| **B38** (container O, optional) | the nightly AITER op ported from Artemis, run on Triton 3.8 (see A0) | Only for S3, S12 (Qwen3-Next reference topology) and the C1 output gate |
| **A0** | A/A control in **each** container: T vs T in O, B vs B in N, in separately captured graphs | This gives the noise floor. Any |T-B| difference below max(2 x A/A spread, 2%) is reported as "parity". |

Rules:
- **Same work on both sides.** If the template fuses activation quantization, B and S include
  it, using `aiter.get_hip_quant(QuantType.per_1x128)` (`nightly/ops/quant.py:368`).
- **Layout conversions.** A conversion that serving would pay per call is counted. For example,
  serving converts activation scales to AITER's column-major `x_scale` for bpreshuffle
  (`op_tests/test_gemm_a8w8_blockscale.py:152`, `src/bench/gemm.py:105-110`). One-time
  conversions are excluded (weight preshuffle, `w_kc` transpose views).
- **Report both** "AITER kernel only" and "AITER as served (incl. conversion)", so the PR can
  argue either way.

### A2. Timing

- **Primary: device time under graph replay**, via `xbench/timing.py` in both containers (A0).
  It is the same algorithm as `wt_next/src/bench/common.py:63-104 measure()`: N calls captured
  per graph, a `torch.cuda._sleep` before the start event, and ABBA alternation of the arms
  present in that container. Across containers, blocks alternate O/N (A0).
  - **N**: 1000 for kernels under 20 us, 200 for 20-200 us, 20 above that.
  - **Rounds**: raise from 7 to 15 (same code; just loop `measure` twice with fresh graphs and pool
    the rounds).
  - Report the median and the spread as (p90 - p10) / median over the rounds.
  - **Warmup**: one eager call per callable before capture (already done in `measure`), plus one
    discarded replay.
- **Cross-check: AITER's own harness** at a few shapes per candidate, so AITER reviewers see
  their own numbers. Use `aiter.test_common.perftest` (`nightly/test_common.py:49`; default
  `num_iters=101`, rotates args to defeat cache) or the `op_tests/` script named per candidate.
  If arm B in our harness and AITER's harness disagree by more than 10% at the same cache state,
  stop and explain the gap before quoting any speedup.
- **Alternatively:** `wt_next/tools/kernel-templating/verify.py --graph --graph-calls N --bench`
  (`verify.py:537-577`) gives T5 for T only. Use it to confirm T matches the T5 numbers already
  recorded in `notes/` (it is a sanity check, not the comparison).

### A3. Cache state

Run two variants for every decode and memory-bound op.
- **hot**: the same inputs and weights every call (what `measure` does by default; it matches
  our recorded T5 numbers).
- **cold/rotating**: a ring of distinct weight and activation buffers. Use
  `src/bench/gemm.py:weight_copies()` sizing: a 512 MiB budget, power of two, at most 256
  copies. The callable list cycles through the ring (`measure` already cycles `calls[i % len]`).
  For attention, rotate the KV pool slices and the index tensors. MI355X has 4 MB of L2 per XCD
  and 256 MB of MALL, so the ring must exceed 256 MB.

Which one matches serving:
- **Decode GEMMs and norms**: rotating (weights of other layers evict). Quote rotating as the
  headline and hot as secondary.
- **Prefill** (M >= 1024): weights are re-read in tiles anyway; report rotating only.
- **Attention decode**: the KV working set is large by construction; report hot only if both
  variants agree within noise.

AITER's tuned-CSV `us` columns come from rotating args (`nightly/test_common.py:60-80`). Never
compare our hot numbers with them.

### A4. Inputs

- **Distributions.** Use the family spec draw classes: the `make_inputs(key, seed)` of
  `wt_next/tools/kernel-templating/specs/<pack>__<family>.py`, which are the same tensors as the
  T3 checks. Use seeds 0 and 100 (100+ are the adversarial/near-eps draws, e.g. the
  `fadd_rmsnorm_template.py` header). For FP8 GEMMs, also use `src/bench` `channel_outliers`
  (every 256th channel x16, `src/bench/README.md:44-46`).
- **Layouts exactly as served:**
  - AITER-preshuffled 16x16 FP8 weights: `aiter.ops.shuffle.shuffle_weight(w, (16,16))`.
  - Scale layouts: row-major `[M, K/128]` fp32 for our kernels; AITER bpreshuffle wants the
    column-major copy.
  - Page sizes as served: page 1 for MiniMax, Qwen3.5 and sparse MLA; page 64 for the GLM
    kc/indexer caches.
- **KV pools.** Use at least two arms per attention candidate: a small pool (262,144 slots) and
  a large pool whose byte size is at least 2^31. Several templates switch code paths at 2^31 /
  2^32 bytes (`decode_tp8_m8_x.py` `_supports_buffer_copy`, `decode_ep8_m1_m8_x.py:899-903`).
- **Same dtype and quant scheme on both sides.** Do not compare a bf16-q AITER path with an
  fp8-q template path without saying so; run both AITER `dot_precision` settings where relevant.
- **Graph-capacity-dependent schedules (C1).** The MiniMax templates key the schedule on
  `kv_indices.numel()`. Set `kv_indices` to the served CUDA-graph capacity (`256 * ctx`) and
  also to the tight size.

### A5. Numerics (pass criteria; bit-exactness across different kernels is not expected)

Compute a float64 (GEMM) or float32 (attention, norms, GDN) torch reference for every cell, and
report three things for both T and B:
- `max_abs`, `max_rel` (relative errors where |ref| > 1e-3), and the fraction of elements
  beyond tolerance;
- the ULP histogram at bf16 (0, 1, 2, >2);
- for fp8 outputs: the code-point mismatch rate, plus the scale tensor's max rel diff.

Pass criteria:

| output dtype | criterion |
|---|---|
| bf16 GEMM / attention out | `torch.testing.assert_close(out, ref, atol=2**-5, rtol=1e-2)` (same as `src/bench/gemm.py:113-120`), **and** T's error statistics no worse than B's by more than 1.25x (median abs err) |
| fp32 recurrent state (GDN, S3/S12) | atol 1e-3, rtol 1e-3 after 1 step; after 64 chained steps (state evolution), T and B errors vs fp32 reference within 2x of each other |
| fp8 e4m3 quantized activations + fp32 group scales | scales: rtol 2**-10; codes: at most 1 code point difference on at most 0.5% of elements |
| index outputs (top-k) | set equality per row (order-insensitive), as in `docs/glm5.2/README.md:95-97` |
| in-place cache writes | compare the written cache slots bytewise against the reference writer, and check that untouched slots stay unchanged |

Any T cell that fails while B passes is a blocker for that row. Report it, do not average it
away. The known numerics flags from phase 1 are the Kimi 12-head MLA BF16 partials (not
shortlisted) and the scale-dependent BF16 replay in `xl_native_rows` (S11).

### A6. Reporting

Write one CSV per candidate with these columns: `candidate, template, shape(M,N,K / rows,heads,...),
cache(hot|rotating), T_us, B_us, Bplus_us, S_us, A0_spread_pct, speedup_vs_B, speedup_vs_Bplus,
speedup_vs_S, aiter_backend, aiter_tuned_row(y/n), T_maxabs, B_maxabs, T_frac_fail, B_frac_fail, pass`.

Then one summary row per candidate: the geometric mean of speedup vs max(B, B+), weighted by
serving frequency. Weights:
- **decode**: the CUDA-graph batch-size distribution of the model's serving runs (C1..C256 in
  `docs/*/measurements.csv` or `results.csv`). Use the per-concurrency decode M where recorded.
  Otherwise weight the served M rows uniformly and say so.
- **prefill**: the prompt-length mix of the serving suites (1024/1024, 5120/1024, 8192/1024).

### A7. Environment pinning

- **Both images.** Record the image digest, `torch.__version__`, `torch.version.hip`,
  `torch.version.git_version`, `triton.__version__`, `/opt/rocm/.info/version` and the hipBLASLt
  version.
  - **N:** `git -C /sgl-workspace/aiter rev-parse HEAD` (expect `581579ab5...`), Triton 3.7.0,
    SGLang 703e763d5f.
  - **O:** Triton `3.8.0+gitdf3f91dd` and `wt_next` HEAD (88081cdc).
- **One physical GPU for both containers.** Pin the same `HIP_VISIBLE_DEVICES`/BDF in both, with
  no co-scheduled work on the node's other GPUs during a block (shared power and thermal budget).
  Record `rocm-smi --showclocks --showpower --showtemp --showperflevel` before and after every
  block.
- Do not change clocks or power caps. If they must be set (e.g. by the cluster's determinism
  mode), set them identically before both containers' blocks and record it.
- Record the AITER env (`AITER_*`, `SGLANG_USE_AITER`) and each container's Triton cache dir.
  Clear the Triton/AITER JIT caches once per container before block 1, and exclude compile time.
- **Order.** Follow Part C. Within a candidate, interleave shapes and alternate the container
  order per block pair. Re-run the per-container A/A and the calibration kernels in every block.

### A8. Optional Triton 3.7 data point (can any template run unchanged in image N?)

**Expected answer: no.**
- 105 of the 106 templates are Gluon kernels written in the Triton 3.8 dialect.
- AITER itself gates its own ports of these kernels at Triton >= 3.8, because "ROCm backported an
  earlier, incompatible Gluon into some 3.7 builds where `triton.experimental.gluon` imports fine
  but the layout ops fail to compile" (`nightly/ops/triton/gated_delta_net/fused_gdn_decode_qkvz.py:93-99`)
  or miscompile (`nightly/ops/triton/attention/paged_attention_output_gate.py:69-80`).
- The only plain-Triton template, `qwen3_5/fp8/fp8_block_scaled_gemm_xl/xl_paired_template.py`
  (DNU), passes the patched `llvm_fn_attrs` launch option.
- Many shortlisted templates also use `llvm_fn_attrs`, `gl.amd.cdna4` intrinsics or inline asm
  (`assessment.json` portability deps).

**Optional probe** (costs minutes, informative for upstreaming).
- Targets: the three shortlisted templates with the fewest Triton-specific features:
  - S10 `kimi_k3/attention_residual_norm_xl/arn_{wide,fused}_xl.py` (Gluon `cdna3` only, no
    `llvm_fn_attrs`, no inline asm);
  - S5 `qwen3_5/fp8/fused_attention_qk_norm_rope_kv_cache_templated/faqk_template.py`;
  - S12 `qwen3_5/fp8/gdn_prefill_group_fp8_quant_templated/gdn_prefill_template.py`.
- In image N, compile each at two served shapes and record pass/fail.
- If one compiles, compare its outputs **bitwise** against the same template's output from O on
  identical inputs. A silent miscompile is the known 3.7 failure mode.
- Only if bit-exact, also time it with the same harness. Report it as "template on Triton 3.7"
  next to T. Never substitute it for T.

---

## Part B: per-candidate plans

Shapes are the served signatures from `template_inventory.json` (rows listed in
`assessment.json`). For range rows, run both endpoints plus the midpoint. "Spec" below means
`wt_next/tools/kernel-templating/specs/<file>`. Its `make_inputs`/`call_kwargs` produce the T
inputs; build the B/S inputs from the same tensors.

### S1. Fused MLA q/kv-a projection + dual RMSNorm (+rope split, +FP8 group quant)

**Nightly note (581579ab5).** There is still no fused AITER op. The GLM FP8 GEMM leg got
faster: the 2624x6144 bpreshuffle decode rows are now Triton, 6.5 us at M=1 vs 17.0 us CK in
e7d2453f2 (`nightly/configs/model_configs/glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv`).
The bf16 2624x6144 M=1 row is 7.4 us. Record which backend arm B dispatches to. The composition
B is now the bar to beat.

| template | entry | spec | shapes |
|---|---|---|---|
| `glm52_fp8/qkv_a_norm_r3/qkv_a_norm_template.py` (lead) | `mla_qkv_a_norm` | `glm52_fp8__qkv_a_norm.py` | M in {1,2,4,8,16,32,64,128,256} plus 1024, 2048, 4096, 8193, 16384; N=2624 (2048 q + 512 kv + 64 rope), K=6144 |
| `glm52/qkv_a_norm_xl/qkv_a_norm_template.py` | `qkv_a_norm_tp4`, `qkv_a_norm_tp8` | `glm52__qkv_a_norm.py` | M in {1..256 pow2} plus 4193, 6144, 8192, 12288, 16384 (bf16 W) |
| `glm52_ep8/qkv_a_norm_xl/decode_async_x.py` | `qkv_a_norm` | `glm52_ep8__qkv_a_norm.py` | M in {4,8,16} (bf16 W) |
| `deepseek_r1/qkv_a_norm_quant_templated/qkv_a_norm_quant_template.py` | `mla_qkv_a_norm_quant` | `deepseek_r1__qkv_a_norm_quant.py` | M in {1..256 pow2, 1024, 4096, 8192, 16384}; N=2112 (1536+512+64), K=7168 |

**B (AITER composition; no single AITER op exists):**
- **GLM FP8** (input bf16 `x[M,6144]`, W fp8 preshuffled `[2624,6144]`, `w_scale[21,48]`):
  ```python
  xq, xs = aiter.get_hip_quant(aiter.QuantType.per_1x128)(x, quant_dtype=aiter.dtypes.fp8)
  xs_t = xs.transpose(0, 1).contiguous().view(*xs.shape)          # counted (served path)
  y = aiter.gemm_a8w8_blockscale_bpreshuffle(xq, w_shuf, xs_t, w_scale, torch.bfloat16)  # nightly/ops/gemm_op_a8w8.py:1495
  q, kv, k_rope = y.split([2048, 512, 64], dim=-1)
  aiter.fused_qk_rmsnorm(q=q, q_weight=q_gamma, q_epsilon=eps, q_out_unquantized=q_out,
                         k=kv, k_weight=kv_gamma, k_epsilon=eps, k_out=kv_out)   # nightly/ops/fused_qk_rmsnorm_group_quant.py:179
  ```
  Also run the Triton alternatives:
  `aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale.gemm_a8w8_blockscale_preshuffle`
  (`nightly/ops/triton/gemm/basic/gemm_a8w8_blockscale.py:250`) and
  `aiter.ops.triton.quant.fused_fp8_quant.fused_rms_fp8_group_quant(q, q_gamma, eps, inp2=kv, inp2_weight=kv_gamma, inp2_epsilon=eps, output_unquantized_inp1=True)`
  (`nightly/ops/triton/quant/fused_fp8_quant.py:164`). B = the fastest composition.
- **GLM BF16:** `aiter.tuned_gemm.tgemm.mm(x, W)` (`nightly/tuned_gemm.py:832`), then the same
  `fused_qk_rmsnorm`. The tuned rows exist: `model_configs/glm5_bf16_tuned_gemm.csv:115-254`.
- **DeepSeek-R1** (input already fp8 `x[M,7168]` + `xs[M,56]`):
  `aiter.gemm_a8w8_blockscale(xq, W, xs, ws)` (plain W; `gemm_op_a8w8.py:1289`) and the
  bpreshuffle variant. Then `fused_rms_fp8_group_quant(q, q_gamma, eps, inp2=kv, inp2_weight=kv_gamma, inp2_epsilon=eps, group_size=128)`,
  which gives the fp8 q with group-128 scales that q_b consumes.
- **S:** the SGLang `forward_absorb_rocm_prepare` q_lora branch:
  - `fused_qkv_a_proj_with_mqa` (FP8 via `fp8_utils.aiter_w8a8_block_fp8_linear`, or bf16 linear);
  - split;
  - `q_a_layernorm` and `kv_a_layernorm` (R1: `fused_qk_rmsnorm_bf16`, then `per1x128` quant).

  Capture this from the adapter's native fallback (`src/artemis_sglang/adapters/glm/`, AST
  rewrite of `_glm_project_norm`). If that is impractical, B is an acceptable stand-in.

**Numerics:** q, kv bf16 against the fp64 reference GEMM + fp32 RMSNorm (bf16 criterion);
k_rope must be bit-identical to the GEMM slice. For R1: fp8 q codes and scales per A5.

**AITER harness cross-check:** `op_tests/test_gemm_a8w8_blockscale.py -m 1 2 4 8 16 32 64 128 256 -nk 2624,6144 --ck_preshuffle True`
gives the GEMM-only lower bound. Our fused T must beat GEMM-only B for the "new op" claim to be
compelling.

### S2. Small-M FP8 block-scale GEMM (M <= 256)

**Nightly note.** Decode-M (<=256) tuned rows are identical to e7d2453f2 for every served
(N,K), and (2560,2048) is still absent from all bpreshuffle tables. New FlyDSL blockscale
backends exist (`nightly/ops/flydsl/gemm_a8w8_blockscale.py`) but are not selected at these rows.
Signatures are unchanged (`nightly/ops/gemm_op_a8w8.py:1289,1495`).

| template | entry | shapes (M; N,K) |
|---|---|---|
| `qwen3_next/fp8_preshuffled_gemm_xl/xl_register_template.py` (lead) | from manifest `qwen3_next_fp8_tp4_gfx950/*fp8*_templated.json` | (1;2560,2048) (4;2048,1024) (4;2560,2048) (4;3072,2048) (8;2048,1024) |
| `qwen3_next/fp8_preshuffled_gemm_xl/xl_slot_template.py` | same | (2;2560,2048) (2;3072,2048) (8;2560,2048) (8;3072,2048) |
| `glm52_fp8/o_gemm_xl/split_k_decode_x.py` | `fp8_a8w8_block_scaled_gemm` (spec `glm52_fp8__o_gemm.py`) | (8;6144,4096) (32;2048,2048) (32;4096,2048) (32;6144,2048) (64;2048,2048) (64;4096,2048) (256;2048,2048) |
| `glm52_fp8/o_gemm_xl/tp8_decode_x.py` | same | (1,2,8,16;6144,2048) |
| `deepseek_r1/o_gemm_xl/o_gemm_xl_template.py` | spec `deepseek_r1__o_gemm.py` | M in {1..256 pow2}; 7168,2048 |
| `deepseek_r1/q_b_gemm_templated/q_b_gemm_template.py` | `fp8_a8w8_block_scaled_gemm_fn` | M in {1..256 pow2}; 3072,1536 (decode rows only here) |
| `minimax_m2_5/fp8_gemm_xl/fp8_gemm_xl_template.py` | `bf16_fp8_block_scaled_gemm` | (4,8,16;3072,1536), **bf16 input** |
| `minimax_m2_5/fp8_gemm_r3/fp8_gemm_r3_template.py` | same | (128,256;3072,1536), bf16 input |

**Arms:**
- **B (fp8 input):** `aiter.gemm_a8w8_blockscale_bpreshuffle(xq, shuffle_weight(W,(16,16)), xs_t, ws, torch.bfloat16)`
  for the preshuffled templates (qwen3_next, glm52_fp8). For plain-W templates (R1), run both
  `aiter.gemm_a8w8_blockscale(xq, W, xs, ws)` and the bpreshuffle variant, and report the min.
- **B (bf16 input; MiniMax):** `aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale.gemm_a16w8_blockscale(x, W, ws, prequant=True)`
  (`nightly/ops/triton/gemm/basic/gemm_a16w8_blockscale.py:21`) **and** per_1x128 quant + bpreshuffle.
- **S (container N):** port `wt_next/src/bench/gemm.py:80-112 native()` into
  `xbench/arms_N.py`. It calls `fp8_utils.aiter_w8a8_block_fp8_linear(x, packed, [128,128], ws, input_scale=xs)`
  of SGLang 703e763d5f. Check that the nightly SGLang still has this function and the
  `_use_aiter_bpreshuffle_gfx95` flag; if it was renamed, use whatever `Fp8LinearMethod.apply`
  calls.
- **Same-process cross-check (container O, optional):**
  `python src/bench/qwen3_next.py --family fp8_block_scaled_gemm --m 1 2 4 8 16 32 64 128 256 --csv s2_qwen3next.csv`
  reproduces `docs/qwen3-next-80b/microbenchmarks.csv` against image O's older stock.
- **Tuned-row status:** most shapes are in merged model_configs, e.g.
  `a8w8_blockscale_tuned_gemm_qwen3_next_80b_a3b.csv:704` (M=1, N=2560, K=2048, CK, 6.26 us),
  `a8w8_blockscale_bpreshuffle_tuned_gemm_glm5.2.csv`, `a8w8_blockscale_tuned_gemm_ds_v3.csv`.
  Run B+ only where a row is missing. Phase 1 found (2560,2048) absent from all *bpreshuffle*
  tables.
- **Cache:** rotating is the headline (A3). The earlier decode numbers were hot
  (`src/bench/README.md:50`).

**Numerics:** fp64 dequant reference, `assert_close(atol=2**-5, rtol=1e-2)`; ordinary and
channel-outlier draws.

**AITER harness:** `python op_tests/test_gemm_a8w8_blockscale.py -m 1 2 4 8 -nk 2560,2048 3072,2048 2048,1024 --ck_preshuffle True`
(check the `-nk` tuple syntax via `dtypes.str2tuple` in the container), and
`op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py` for the Triton backend.

### S3. GDN decode: conv1d update + gated delta rule + gated RMSNorm (+FP8 group quant)

**Nightly note: the fused op is already upstream.**
`aiter.ops.triton.gated_delta_net.fused_gdn_decode_qkvz.fused_gdn_decode_qkvz` (`nightly/ops/triton/gated_delta_net/fused_gdn_decode_qkvz.py:311`)
was generated from our `qwen3_next/gdn_fused_decode_group_fp8_quant/gdn_decode_m16.py`, with the
same sha256 `16ec8eaf...`. It covers Qwen3.5 through `qkvz_layout="flat"` and ratio 4. It needs
Triton >= 3.8, so in image N it is unsupported and the B arm is the 4-kernel fallback below.
Add the B38 arm (container O, vendored op, A0):
```python
out, q8, sc = fused_gdn_decode_qkvz(qkvz, ba, conv_state, ssm_state, cache_indices, conv_weight, conv_bias,
    A_log, dt_bias, norm_weight, scale=128**-0.5, norm_eps=eps,
    quant_dtype=torch.float8_e4m3fn,            # None for the bf16-output (mxfp4) rows
    qkvz_layout="interleaved" if qwen3_next else "flat")
```
Probe `fused_gdn_decode_qkvz_supported(...)` first and record its reason string if False. Also
run AITER's `op_tests/op_benchmarks/triton/bench_fused_gdn_decode_qkvz.py` in O as the
AITER-harness cross-check.

**Decision.** T is worth a PR only where it beats B38 by >= 1.10x (rotating). Expected candidates
are our M=1/8/64 schedules and the Qwen3.5 rows, which run AITER's ratio-generic kernel.

| template | entry | shapes |
|---|---|---|
| `qwen3_next/gdn_fused_decode_group_fp8_quant_templated/gdn_decode_template_v2.py` (lead) | default spec `gdn_qwen3next` (`tools/kernel-templating/family_spec.py:186`) | M in {1,8,16,64}; qkvz 3072 (q512 k512 v1024 z1024), ba 16, k_heads 4, v_heads 8, d 128; fp8 out + group-128 scales |
| `qwen3_5/fp8/gdn_fused_decode_group_fp8_quant_r3/r3_gdn_decode_small.py` | `gdn_fused_decode` | M in {1,2,8}; qkvz 5120, ba 32, k_heads 4, v_heads 16 (bf16 out on the mxfp4 rows, fp8 on fp8) |
| `qwen3_5/fp8/gdn_fused_decode_group_fp8_quant_templated/qwen3_5_fp8_gdn_decode_template.py` | spec `qwen3_5_fp8__gdn_fused_decode_group_fp8_quant.py` | M in {128,256} |
| `qwen3_5/mxfp4/gdn_fused_decode_xl/xl_gdn_decode_m96.py` | `gdn_fused_decode` | M in {16,32,64,96} (bf16 out) |

**B (AITER composition, 3 launches),** with `x = qkvz[:, :key_dim*2+value_dim]` and
`z = qkvz[:, key_dim*2+value_dim:]`:
```python
q, k, v = aiter.ops.triton.gated_delta_net.causal_conv1d_decode.causal_conv1d_update_split_qkv(
    x.unsqueeze(-1), conv_state, conv_weight, key_dim, value_dim, bias=conv_bias,
    activation="silu", conv_state_indices=cache_indices)                         # causal_conv1d_decode.py:25
core, _ = aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr.fused_rearrange_sigmoid_gated_delta_rule(
    A_log, a, b, dt_bias, qkv_post_conv, key_dim, value_dim, 128, 128, scale=d**-0.5,
    initial_state=ssm_state, inplace_final_state=True, ssm_state_indices=cache_indices,
    use_qk_l2norm_in_kernel=True)                                                # fused_rearrange_sigmoid_gdr.py:186
from aiter.ops.gated_rmsnorm_fp8_group_quant import gated_rmsnorm_fp8_group_quant  # not re-exported by aiter/__init__.py
gated_rmsnorm_fp8_group_quant(out_fp8, out_scale, core, z, norm_w, eps, 128)   # nightly/ops/gated_rmsnorm_fp8_group_quant.py:24
```
- Alternative stage 2: HIP `aiter.fused_split_gdr_update(mixed_qkv[B,2k+v,T], A_log, a, dt_bias, b, ssm_state, cache_indices, key_dim, value_dim, 4, v_heads, 128)`
  (`nightly/ops/fused_split_gdr_update.py:37`). It fuses the q/k/v split but not the conv.
- For bf16-output rows, stage 3 is `aiter.ops.triton.quant.fused_fp8_quant.fused_rms_gated_fp8_group_quant`
  (`:396`) if it can emit bf16. Otherwise use SGLang's FLA `RMSNormGated`.
- Check that `causal_conv1d_update_split_qkv` returns tensors that the GDR op accepts as packed
  qkv. If it does not, add the concat to B and say so.

**S (container N):** port the stock path of
`wt_next/src/bench/gdn_fused_decode_group_fp8_quant.py` into `xbench/arms_N.py`. It runs SGLang
703e763d5f's GDN decode chain; with Triton 3.7 that chain cannot take AITER's fused op. Use count
972 (27 passes over 36 independent states, `src/bench/README.md:57`) in both containers, and
extend it to the Qwen3.5 dims. The same-process cross-check in O is
`python src/bench/qwen3_next.py --family gdn_fused_decode_group_fp8_quant --m 1 8 16 64`.

**State handling:** 36 independent `(conv_state, ssm_state)` slot sets cycled inside the graph,
with a reset between rounds (`measure(reset=...)`). Use realistic pool sizes (slots = 2 x max
batch).

**Numerics:** after one step, fp8 codes/scales per A5 and the fp32 state per A5. Then chain 64
steps and compare T vs B drift against an fp32 torch recurrence.

**AITER harness:** `op_tests/test_split_gdr_update.py`, `op_tests/test_causal_conv1d_update.py`,
`op_tests/test_gated_rmsnorm_fp8_quant.py`, `op_tests/op_benchmarks/triton/bench_gated_delta_rule_block.py`.

### S4. Sparse MLA decode (GLM-5.2, fp8_scalar KV [slots,576], top-k 2048)

**Nightly note.** `sparse_mla_fwd` (`nightly/ops/triton/attention/sparse_mla.py:696`)
now launches a dedicated Gluon fp8-dot kernel for fp8_scalar caches
(`_gluon_kernels/gfx950/attention/sparse_mla_fp8.py`). The heads-per-program policy is at
`sparse_mla.py:475-489`, and the generic path uses block_m=8 for H<16 (`:998`). The B arm runs
on image N's Triton 3.7. Log which kernel was launched (`AITER_LOG_MORE=1`, or by checking the
`_sparse_mla_fp8` compile in the Triton cache). The new kwargs (`extra_kv_scale`, `inv_rope_*`,
`out_mxfp8`) stay at their defaults.

| template | entry | heads | shapes M (spec flag) |
|---|---|---|---|
| `glm52/sparse_paged_mla_xl/decode_tp4_small.py` (lead) | `sparse_paged_mla` | 16 | 1,2,16 (spec 0/1) |
| `glm52/sparse_paged_mla_xl/decode_tp8_narrow.py` | same | 8 | 32 (0/1), 128, 256 (spec 1) |
| `glm52/sparse_paged_mla_xl/decode_tp8_deep.py` | same | 8 | 128, 256 |
| `glm52_fp8/sparse_paged_mla_xl/decode_tp8_m8_x.py` | same | 8 | 1,2,4,8,16 |
| `glm52_fp8/sparse_paged_mla_r3/spm_tp8_policy_template.py` | same | 8 | 12, 24, 40..256 step 8 (sample 12, 24, 48, 96, 128, 192, 256) |
| `glm52_fp8/sparse_paged_mla_r3/spm_tp8_split_template.py` | same | 8 | 32, 64 |
| `glm52_fp8_ep8/sparse_paged_mla_xl/decode_ep8_{m1_m8,m12_m24,m16_m32}_x.py`, `glm52_ep8/sparse_paged_mla_r3/decode_ep8{,_d16_d32}_template.py` | same | 64 | 1,2,4,8,12,16,24,32 |

Specs: `glm52__sparse_paged_mla.py`, `glm52_fp8__sparse_paged_mla.py`,
`glm52_fp8_ep8__sparse_paged_mla.py`, `glm52_ep8__sparse_paged_mla.py`.

Inputs:
- `q [M,H,576]` (bf16 for the MXFP4 TP rows; fp8 with unit scale for the FP8/EP8 rows);
- `kv [P,576]` fp8 e4m3 with `kv_scale = tensor([s], fp32)`;
- `slots [M,2048]` int32 global slot ids, with -1 sentinels in 1% of entries (realistic short
  contexts);
- `softmax_scale = 576**-0.5` (or the served value from the spec);
- P in {262144, 4194304} (the second gives 2.4 GB, above 2^31 bytes).

**B:**
```python
indptr = torch.arange(0, (M + 1) * 2048, 2048, dtype=torch.int32, device=dev)
out, _ = aiter.ops.triton.attention.sparse_mla.sparse_mla_fwd(
    q, kv, indptr, slots.reshape(-1), softmax_scale,
    kv_scale=kv_scale, kv_lora_rank=512, qk_rope_head_dim=64,
    has_invalid=True, dot_precision=dp, q_scale=q_scale_if_fp8)        # nightly sparse_mla.py:696
```
Run `dp` in {"bf16", "fp8"} (fp8 is valid only for fp8_scalar caches, `:736`) and report
the faster. Count the `indptr`/flatten construction as free: it is static under graphs.

**S:**
- MXFP4 TP4/TP8: AITER asm `aiter.mla.mla_decode_fwd` (`nightly/mla.py:990`) with page_size 1,
  as SGLang's DSA aiter backend calls it. At H=8, pad to 16 heads (`head_repeat_factor=2`), as
  stock does (`src/artemis_sglang/adapters/glm/attention.py:567-575`).
- FP8 and EP8 profiles: SGLang's TileLang sparse decode (`--dsa-decode-backend tilelang`). Call
  the kernel the backend's `_forward_tilelang` uses (locate it in the container's sglang).

**Numerics:** an fp32 reference attention over the gathered slots, using the bf16 criterion on
the output `[M,H,512]`. The EP8 templates return a transposed-stride view, so compare values,
not strides. Then measure the cost of making the output contiguous separately and report it.

**AITER harness:** `op_tests/triton_tests/attention/test_sparse_mla.py`,
`op_tests/op_benchmarks/triton/bench_sparse_mla.py`, `op_tests/test_mla_sparse.py`.

**Decision rule:** if B is within noise of T at H=8, S4 reduces to the H=64 rows.

### S5. Qwen3.5 attention prologue (Gemma q/k norm + mRoPE + q|gate split + FP8 KV write)

**Nightly note.** There is a new partial overlap:
`aiter.ops.triton.rope.fused_qk_norm_rope_gate_fp8_quant.fused_qk_norm_rope_gate_fp8_quant`
(`nightly/ops/triton/rope/fused_qk_norm_rope_gate_fp8_quant.py:65`). It has the interleaved
q|gate input, Gemma `+1` in the kernel and partial rotary. It has **no KV-cache write** and
1-D positions, and it quantizes Q/K/V per sequence for FMHA.

Add **arm B3**:
1. `fused_qk_norm_rope_gate_fp8_quant(q_gate, key, value, qw, kw, cos_sin, positions[0], cu_seqlens, num_actual_tokens=M, num_query_heads=8, num_kv_heads=1, head_dim=256, rotary_dim=64)`
   (identical-rows draw only);
2. then the KV-cache write as stock does it: SGLang `set_kv_buffer`, or `k.to(fp8)` and
   `index_copy_` into `[slots,1,256]`, counted.

B3 does extra work (fp8 Q/V outputs) that the decode path does not need. Report it, but do not
use it as the headline B.

| template | entry | shapes |
|---|---|---|
| `qwen3_5/mxfp4/fused_attention_qk_norm_rope_kv_cache_xl/xl_decode_small_template.py` (lead) | `fused_attention_qk_norm_rope_kv_cache` | rows in {1,2,4,8}; qkv_gate width 4608 = 8 q heads x (256 q + 256 gate) + 256 k + 256 v; rotary_dim 64 |
| `qwen3_5/fp8/fused_attention_qk_norm_rope_kv_cache_templated/faqk_template.py` | spec `qwen3_5_fp8__fused_attention_qk_norm_rope_kv_cache.py` | rows in {8,16,32,64} |

Inputs: positions `[3,M]` (mRoPE; text-only decode uses identical rows, so run both an
identical-rows and a distinct-rows draw), `cos_sin_cache [262144,64]`, KV cache
`[slots,1,256]` fp8, page 1, slots P in {262144, >=2^31/256 bytes}.

**B1 (HIP + gate split):**
1. `q_gate = qkv[:, :4096].view(M,8,512)`; `q = q_gate[..., :256]`; `gate = q_gate[..., 256:]`.
   This copy is counted.
2. Then:
   ```python
   aiter.fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(qkv_packed, qw, kw, cos_sin, positions, M, 8, 1, 1, 256,
       is_neox_style=..., mrope_section_=[...], is_interleaved=..., eps=1e-6, q_out=..., k_cache=..., v_cache=...,
       slot_mapping=..., per_tensor_k_scale=ks, per_tensor_v_scale=vs, k_out=None, v_out=None, return_kv=False,
       use_shuffle_layout=False, block_size=1, x=..., rotary_dim=64, gemma_norm=True)
   ```
   (`nightly/ops/fused_qk_norm_mrope_cache_quant.py:10-39`; take the mrope_section and the
   interleave flags from the model config).

**B2 (Triton, gate-aware, plain RoPE; valid only for the identical-rows draw):**
`aiter.ops.triton.rope.fused_qkv_split_qk_norm_rope_cache.fused_qkv_split_qk_norm_rope_cache(qkv, qw+1, kw+1, cos, sin, positions[0], k_cache, v_cache, slots, qh=8, kvh=1, head_dim=256, attn_output_gate=True, gated_qkv_layout="interleaved", kv_cache_layout="NHD", k_scale=ks, v_scale=vs)`
(`:35`). It has no Gemma flag, so fold `1+w` into the weights and note the rounding difference.

**S:** SGLang `Qwen3_5AttentionDecoderLayer.self_attention` prologue: split, `q_norm`/`k_norm`
GemmaRMSNorm, `rotary_emb` (mRoPE), backend `set_kv_buffer`.

**Numerics:** q (bf16 criterion), gate passthrough bit-exact, and cache slots per A5 (fp8 codes;
unit and non-unit scales 0.75/1.25 as in `src/bench/README.md:61`).

**AITER harness:** `op_tests/test_fused_qk_norm_mrope_cache_quant.py`.

### S6. DSA indexer prepare (GLM-5.2 MXFP4)

**Nightly note.** No change: `indexer_qk_rope_quant_and_cache` is identical
(`nightly/ops/cache.py:123`); `rotate_activation` is now at `nightly/ops/quant.py:1361`.

| template | entry | shapes |
|---|---|---|
| `glm52/indexer_prepare_r3/decode_tp4_template.py` (lead) | `indexer_prepare` | m in {1,2,4,8,16} x spec {0,1}; hidden 6144, q_lora 2048, 32 index heads x 128 |
| `glm52/indexer_prepare_xl/decode_tp8_lds_x.py` | same | m in {128,256} |
| `glm52/indexer_prepare_key_only_xl/prefill_tp4_x.py` | `indexer_prepare_key_only` | m in {4193, 6144, 8192, 12288, 16384, 16768} |

Specs: `glm52__indexer_prepare.py`, `glm52__indexer_prepare_key_only.py`. Index cache:
uint8 `[pages,64,132]` (128 fp8 + fp32 scale, preshuffled).

**B (AITER composition):**
1. `qb = tgemm.mm(q_lora, wq_b)` `[m,4096]`, `kp = tgemm.mm(x, wk)` `[m,128]`,
   `wt = tgemm.mm(x, wgate)` `[m,32]` (3 bf16 GEMMs; `nightly/tuned_gemm.py:832`).
2. The k LayerNorm, RoPE, Hadamard and fp8 quant/cache, either via
   `aiter.indexer_qk_rope_quant_and_cache(q, q_out, weights, weights_out, k, kv_cache, slot_mapping, norm_weight, norm_bias, positions, cos, sin, eps, quant_block_size=128, scale_fmt=..., weights_scale=..., preshuffle=True, is_neox=False)`
   (`nightly/ops/cache.py:123`), plus the Hadamard rotation via
   `aiter.ops.quant.rotate_activation` or `rope_rotate_activation` (`nightly/ops/quant.py:1361,1442`)
   if `indexer_qk_rope_quant_and_cache` does not apply it.
3. Confirm from the GLM-5.2 config which stages the AITER op covers, and add any missing stage
   as torch.

Key-only: drop qb/weights.

**S:** SGLang DSA `Indexer.forward_cuda` / `_forward_cuda_k_only` (native GEMMs + AITER
quant/cache).

**Numerics:** q fp8 codes/scales, weights fp32 (rtol 1e-3), and index cache bytes per A5.

**AITER harness:** `op_tests/test_indexer_qk_rope_quant_and_cache.py`,
`op_tests/test_indexer_k_quant_and_cache.py`.

### S7. MLA kc absorb: FP8 BMM + RoPE + concat + KV-cache write (GLM K=192)

**Classification:** GEMM-dominated fused op. The FP8 BMM (H x M x 512 x 192) dominates
prefill time; RoPE, concat and cache write are epilogue. Per the user's priority it runs after
the non-GEMM block and before the plain GEMMs.

**Nightly note.** `fused_fp8_bmm_rope_cat_and_cache_mla` is unchanged apart from an import
refactor (`nightly/ops/triton/fusions/fused_bmm_rope_kv_cache.py:384`). For the FP8 decode rows
(tp4_small/tp4_large/tp8_large) add **arm B4**:
`aiter.ops.triton.fusions.fused_bf16_bmm_cat_quant_pad_mla.fused_bf16_bmm_cat_quant_pad_mla(q_nope, w_uk_t_bf16, q_pe_roped, q_scale)`
(new in the nightly), plus RoPE and the cache write it does not do. It uses bf16 weights, so it is
only a lower bound on what AITER offers for decode q-prep.

| template | entry | H | shapes M |
|---|---|---|---|
| `glm52/kc_rope_cache_xl/kc_prefill_xl.py` (lead) | `kc_rope_cache` | 8 | 1025, 2048, 3072, 4096, 4193, 8192, 16384 |
| `glm52_ep8/kc_rope_cache_r3/prefill_ep8_m128_to_2048_tpl.py` | `kc_rope_cache` | 64 | 128, 512, 1024, 2048 |
| `glm52_fp8/kc_rope_cache_xl/tp4_large_xl.py` / `tp4_small_xl.py` | `mla_fp8_kc_rope_cache` | 16 | 2,4,8, 72..256 step 8 (sample) |
| `glm52_fp8/kc_rope_cache_xl/tp8_large_xl.py` | same | 8 | 72..128 step 8, 256 |

Specs: `glm52__kc_rope_cache.py`, `glm52_ep8__kc_rope_cache.py`, `glm52_fp8__kc_rope_cache.py`.
Inputs:
- `query [M,H,256]` bf16 (192 nope + 64 rope), `kv_latent [M,512]`, `k_rope [M,64]`;
- `w_kc^T [H,512,192]` fp8 + scalar scale;
- `cos/sin [P,32]`, positions, `cache_locations` int64;
- cache `[slots,576]` fp8, page 64.

**B:**
```python
aiter.ops.triton.fusions.fused_bmm_rope_kv_cache.fused_fp8_bmm_rope_cat_and_cache_mla(
    q_nope, w_k, w_k_scale, q_pe, k_nope, k_rope, kv_cache, slot_mapping, positions, cos, sin,
    group_size=128, transpose_bm=True, k_scale=ks, is_neox=False, q_out_dtype=<served>)   # nightly fused_bmm_rope_kv_cache.py:384
```
- **First check that it accepts K=192** (`q_nope [M,H,192]`, `w_k [H,512,192]`). If it asserts,
  record "unsupported" and that is the PR content. Then B falls back to the AITER composition:
  `batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(q_nope, w_k, w_k_scale, transpose_bm=True)`
  + `aiter.ops.triton.fusions.fused_kv_cache.fused_qk_rope_cat_and_cache_mla` (`fused_kv_cache.py:106`).
- Our templates also zero padded rows (`native_zero_padding`) and emit unit-scale fp8 q. Match
  these on the AITER side or document the difference.

**S:** SGLang `forward_absorb_rocm_prepare` W_UK bmm + `rotary_emb` + backend cache save.

**Numerics:** q_out (bf16 criterion, or fp8 codes when fp8 out) and cache slots bytewise per A5.

**AITER harness:** `op_tests/triton_tests/fusions/test_fused_bmm_rope_kv_cache.py`.

### S8. Skinny-N BF16 GEMM, N=32, K=4096 (Qwen3.5 GDN in_proj_ba)

**Nightly note.** N=32/K=4096 still has only an M=16 Triton row (5.6 us). A new FlyDSL
skinny gfx950 kernel exists (`nightly/ops/flydsl/kernels/gemm_a16w16_skinny_gfx950.py`) but is
only reachable through tuned rows. Arm B+ (tuning in N) is the decisive arm.

| template | entry | shapes M |
|---|---|---|
| `qwen3_5/mxfp4/bf16_gemm_xl/xl_n32_fold_template.py` (lead) | `bf16_gemm` | 16,32,64,96,128 |
| `qwen3_5/mxfp4/bf16_gemm_xl/xl_n32_gemv_template.py` | `bf16_gemm` | 1,2,4 |
| `qwen3_5/fp8/bf16_gemm_xl/xl_gdn_gemv_template.py` | spec `qwen3_5_fp8__bf16_gemm_xl.py` | 1,2,4,16 |
| `qwen3_5/mxfp4/bf16_gemm_templated/qwen3_5_mxfp4_bf16_gemm_n32_template.py` | `bf16_gemm` | 2048,3072,4096,5120,7168 |
| `qwen3_5/fp8/bf16_gemm_xl/xl_n32_template.py` | spec as above | 1024,2048,2232,3072,3616,5000,7168,8192 |
| `qwen3_5/fp8/bf16_gemm_xl/xl_splitk_template.py` | same | 4096, 6144 |

**B:** `aiter.tuned_gemm.tgemm.mm(x, W)`, which records the dispatched backend. For M<=4 this
is the skinny path per `is_skinny_default_shape` (`nightly/tuned_gemm.py:88-109`). Also run
`aiter.ops.triton.gemm.basic.gemm_a16w16.gemm_a16w16(x, W)` (`:578`) and `torch.nn.functional.linear`
(hipBLASLt), and report the min.

**B+:** N=32/K=4096 is absent from every bf16 tuned CSV (`model_configs/qwen3_5_397b_bf16_tuned_gemm.csv`
has only N=64), so tune all listed M and re-run. **This arm decides S8:** if B+ is within 10%,
the PR becomes "add tuned rows", not "add a kernel".

**S:** SGLang `UnquantizedLinearMethod.apply` (`src/bench/gemm.py:86-89` shows how to call it).

**Numerics:** fp64 reference, bf16 criterion. **Cache:** rotating (W is only 256 KB, so rotate
x as well).

**AITER harness:** `op_tests/test_gemm_a16w16.py`, `op_tests/op_benchmarks/triton/bench_gemm_a16w16.py`.

### S9. MLA vc absorb FP8 batched GEMM

**Nightly note.** The batched op gained `backend=` and `y_scale_out=` (per-token FP8
output quant; `nightly/ops/triton/gemm/batched/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py:30-44`).
For the R1 template (group-128 output quant), B = the op with `y_scale_out=None` + `per_1x128`
quant (as before). Also report the op with `y_scale_out` (per-token) as an informative arm with a
different quant scheme, labelled so. The gfx950 config dir is still empty, so B+ (config sweep)
still applies.

| template | entry | shapes |
|---|---|---|
| `glm52/vc_reconstruct_r3/vc_prefill_tpl.py` (lead) | `vc_reconstruct` | H=8, K=512, N=256; M in {4097, 4193, 6144, 8192, 12288, 16384} |
| `deepseek_r1/vc_reconstruct_quant_templated/vc_reconstruct_quant_template.py` | `mla_fp8_vc_reconstruct_group_fp8_quant` | H=16, V=128, K=512; M in {1..64 pow2}; output fp8 + group-128 scales |

Inputs: `latent [M,H,512]` bf16; W `[H,N,512]` fp8 (K-contiguous, served as `w_vc.transpose(1,2).contiguous()`)
+ scalar fp32 scale.

**B:**
`aiter.ops.triton.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(latent, W, w_scale, group_size=128, dtype=torch.bfloat16, transpose_bm=True)`
(`:30`). Choose `transpose_bm`/`transpose_bm_in` to match the `[M,H,K]` input and the
`[M,H*N]` output. For R1, add `aiter.get_hip_quant(QuantType.per_1x128)` on the output (the
o_proj input), because our kernel fuses it.

**B+:** there is no gfx950 tuned config for this op. Sweep its `config=` dict (BLOCK_M/N/K,
num_warps, num_stages, waves_per_eu) over a small grid at each M, and report the best.

**S:** SGLang `rocm_absorb_v_bmm` (+ R1's per1x128 quant).

**Numerics:** bf16 criterion on output; R1 fp8 codes/scales per A5.

**AITER harness:**
`op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py`,
`op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py`.

### S10. Kimi attention-residual gate

**Nightly note.** `attn_res_gate` signature is unchanged (`nightly/ops/triton/fusions/attn_res.py:680`).
It now picks num_warps/num_stages from config tables (`attn_res.py:276-295`). No other change.

| template | entry | shapes (M, valid_rows nvb, mode) |
|---|---|---|
| `kimi_k3/attention_residual_norm_xl/arn_wide_xl.py` (lead) | `attention_residual_norm_wide_xl` | M in {32,64,128,256} x nvb 1..8 x mode {4,5,6}; sample nvb {1,4,8} |
| `kimi_k3/attention_residual_norm_xl/arn_fused_xl.py` | `attention_residual_norm_fused_xl` | M in {1,2,8} x nvb {1,4..8} x mode {4,5,6} |

The mode bits are addend | write<<1 | norm<<2 (`assessment.json`, kimi_k3 entries). Mode 4 = norm only, 5 = norm +
addend, 6 = norm + bank write. H=7168. Spec: `kimi_k3__attention_residual_norm.py`.

**B:**
```python
res = aiter.ops.triton.fusions.attn_res.attn_res_gate(
    prefix, bank[:, :nvb, :], score_weight, eps,
    add_hidden=addend if mode & 1 else None,
    output_rms_weight=output_weight, output_rms_eps=eps2,
    close_block=bool(mode & 2))                                           # nightly attn_res.py:680
```
`score_weight` = the same pre-folded `rms_weight*query` vector we pass (`get_cw`). With
`close_block=True`, AITER returns a **new** `[M,nvb+1,7168]` tensor, while we write row `nvb` in
place. Time it as is (that is what an AITER user pays). Also time
`bank[:, nvb].copy_(prefix_out)` + `close_block=False` as "AITER, in-place equivalent".

**Host overhead:** AITER's comment says decode is host-bound (~50 us of Python per call,
`attn_res.py:65-70`). Graph replay removes that for both arms. Also report eager wall time (T4
style) once, because a vLLM/ATOM user without graphs would see it.

**S:** SGLang `sglang.srt.layers.attn_residual._aggregate_hip` (the native fallback the
adapter patches).

**Numerics:** out and current (bf16 criterion); bank row bytewise vs reference.

**AITER harness:** `op_tests/triton_tests/fusions/test_attn_res.py`,
`op_tests/op_benchmarks/triton/bench_attn_res.py`.

### S11. FP8 block-scale GEMM prefill (M 1024..32768)

**Nightly note: expect this candidate to shrink or drop.** The AITER rows at M>=8192
were retuned to FlyDSL and are 15-48% faster than in e7d2453f2. Examples: 7168x2048 M8192
221.4 -> 128.4 us; 4096x2048 bpreshuffle M16384 187.6 -> 130.0 us; 3072x1536 M8192
82.3 -> 56.8 us. Run S11 last.

| template | shapes |
|---|---|
| `qwen3_next/fp8_preshuffled_gemm_xl/xl_packed_template.py` (lead) | (1024..2048; 2560,2048), (1024..1280, 2048; 3072,2048) |
| `qwen3_next/fp8_preshuffled_gemm_xl/xl_native_rows_template.py` | 4096, 7168, 8192, 8193..16384 on (2560,2048), (3072,2048), (2048,1024) |
| `glm52_fp8/q_b_gemm_xl/prefill_x.py` | 1024..8192 on (4096,2048); 8193..16384 on (2048,2048), (4096,2048) |
| `deepseek_r1/o_gemm_prefill_templated/o_gemm_prefill_template.py` | 1024..16384 on (7168,2048), **bf16 input** |
| `deepseek_r1/kv_b_gemm_templated/kv_b_gemm_template.py` | 1024..16384 on (4096,512), **bf16 input** |

**Arms:** as in S2. For the bf16-input R1 templates, B =
`gemm_a16w8_blockscale(x, W, ws, prequant=True)` and per_1x128 quant + `gemm_a8w8_blockscale[_bpreshuffle]`,
with the min reported. S for qwen3_next = the ported `native()` in N with the rotating-weight ring of `src/bench/gemm.py`.
The same-process cross-check in O is `python src/bench/qwen3_next.py --family fp8_block_scaled_gemm`. Use ordinary and `channel_outliers` draws.

**Extra check for `xl_native_rows`:**
- It replays tiles in exact BF16 when scales exceed a bound.
- Add a draw where 1% of the 128x128 weight blocks have scales above that bound, and time and
  check both paths.
- An upstream reviewer will ask about this first.

### S12. GDN prefill fused (conv + chunk GDR + gated norm), conditional

**Nightly note.** AITER now has `fused_gdn_prefill_qkvz`
(`nightly/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py:1625`), ported from our
Qwen3-Next pack. It supports the Qwen3-Next topology only (qkvz 3072, ba 16; `:73-89`) and needs
Triton >= 3.8, so it does **not** apply to the Qwen3.5 templates here. The FlyDSL
`gdn_fused_prefill_fwd` (`nightly/ops/flydsl/gdn_fused_prefill.py:147`) is chunk-core only with
a bf16 state, so it is not applicable to our fp32 state either.

B stays the multi-kernel composition below. Optionally, run B38 at Qwen3-Next shapes in O. That
calibrates the upstream port against its own source and supports a "extend to Qwen3.5" PR.

| template | entry | shapes (tokens, batch) |
|---|---|---|
| `qwen3_5/mxfp4/gdn_prefill_xl/xl_gdn_prefill_dynamic.py` (lead) | `gdn_prefill` | (1024,1), (5120,1), (8192,1), (16384,1), (8x1024), (16384 as 2/4 seqs) |
| `qwen3_5/mxfp4/gdn_prefill_r3/r3_gdn_prefill_template.py` | same | (10240, served tuple), (16384, served tuple) |
| `qwen3_5/fp8/gdn_prefill_group_fp8_quant_templated/gdn_prefill_template.py` | spec `qwen3_5_fp8__gdn_prefill_group_fp8_quant.py` | 5000, 15000, 15360, 16384 |

Spec: `qwen3_5_mxfp4__gdn_prefill.py`. Inputs:
- `projected_qkvz [T,5120]`, `projected_ba [T,32]`;
- `conv_state [slots,3072,3]`, `delta_state [slots,16,128,128]` fp32;
- `cu_seqlens`, `has_initial_state` (run both all-False and all-True).

**B (AITER composition):**
1. conv: `aiter.ops.triton.gated_delta_net.causal_conv1d_prefill.causal_conv1d_split_qkv_triton_fn`
   (`causal_conv1d_prefill.py:43`), or the HIP `aiter.ops.causal_conv1d_fwd_split_qkv.causal_conv1d_split_qkv_hip_fn` (`nightly/ops/causal_conv1d_fwd_split_qkv.py:90`); confirm the public wrapper names in the container;
2. gates: g, beta from a, b, A_log, dt_bias (torch, as SGLang does);
3. `aiter.ops.triton.gated_delta_net.gated_delta_rule.chunk_gated_delta_rule_opt(q,k,v,g,beta, scale, initial_state, output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)`
   (`:354`; also run `chunk_gated_delta_rule`, `:211`);
4. gated RMSNorm (`gated_rmsnorm_fp8_group_quant` for fp8 out, or the bf16 equivalent).

The AITER `chunk_gdn_h` tuned table for Qwen3.5-397B is gfx942-only
(`model_configs/qwen3_5_397b_chunk_gdn_h_opt_tuned.csv`), so also run the
`chunk_gated_delta_rule_fwd_h` HIP/flydsl variants if they dispatch on gfx950.

**S:** the SGLang linear-attention backend prefill (FLA). The adapter's native fallback is
`src/artemis_sglang/adapters/qwen35_prefill.py`.

**Timing:** eager, because the template is not graph-safe (`capture_safe False`). Use
`torch.cuda.Event` around 20 calls, ABBA over 7 rounds. Prefill is not graph-captured in serving
either.

**Numerics:** output (bf16 criterion), final state (fp32 criterion).

**AITER harness:** `op_tests/test_gated_delta_rule.py`,
`op_tests/op_benchmarks/triton/bench_gated_delta_rule_block.py`.

**Decision rule:** keep S12 only if T beats the best B by >=1.3x at 5120 and 16384 tokens.

### Conditional candidates (see Part C for where they slot in)

- **C1. MiniMax paged GQA** (non-GEMM). The Qwen3.5 output-gate member was dropped: it is already upstream as `paged_attention_output_gate_group_fp8_quant` (`nightly/ops/triton/attention/paged_attention_output_gate.py:192`, Triton >= 3.8). Optionally measure our `r3_paog_template.py` against it via B38 for parity only.
  - Decode: `aiter.paged_attention_ragged(out, workspace, q, k_cache, v_cache, scale, kv_indptr, kv_page_indices, kv_last_page_lens, 1, max_num_partitions, None, "fp8", "NHD", 0.0, k_scale, v_scale)`
    (`nightly/ops/attention.py:792`). Also run `aiter.ops.triton.gluon.pa_decode_gluon.pa_decode_gluon` (`:4770`).
  - Prefill: `aiter.mha_batch_prefill_func(q, k, v, cu_seqlens_q, kv_indptr, kv_page_indices, max_q, max_k, causal=True, q_descale=..., k_descale=..., v_descale=...)`
    (`nightly/ops/mha.py:4514`).
  - Shapes: decode M {1,2,4,8,16,32,64} with Hq=12, Hk=2, D=128 at ctx {2k, 16k, 64k}; prefill
    {1024, 4096, 6143, 8192} tokens.
  - Run each with `kv_indices` sized to the CUDA-graph capacity and to the tight size (A4).
- **C2. Kimi GEMM+add2.**
  - B = `aiter.ops.triton.gemm.fused.fused_gemm_a16w16_mul_add.fused_gemm_a16w16_mul_add(x, W, 1.0, b)` (new in the nightly, `nightly/ops/triton/gemm/fused/fused_gemm_a16w16_mul_add.py:35`). Use `b = shared` (no prefix), or `b = shared + prefix` with that one add counted.
  - Also run `tgemm.mm(x, W)` + `torch.add(out, shared).add_(prefix)`.
  - Shapes: M {1,2,4,8,16}, N=7168, K=3584, with and without prefix.
  - Tuned rows exist: `model_configs/kimik3_bf16_tuned_gemm.csv:75-91`.
- **C3. GLM index scores top-k** (non-GEMM). Nightly note: GLM-5 tuned mqa-logits tables (`nightly/configs/model_configs/paged_mqa_logits_tuned_glm5.csv`), FlyDSL `fp8_paged_mqa_logits_gfx950` and `topk_per_row_radix_stream` are new, so arm B is stronger.
  - Only after the per-call `torch.cuda.Stream()` is removed.
  - B = `aiter.ops.triton.attention.fp8_mqa_logits` / `pa_mqa_logits` + `top_k_per_row_prefill`
    (`op_tests/test_topk_row_prefill.py`).
- **C4. Untuned bf16 decode shapes.**
  - Only B+ vs T at (2,4,8; 6144,16384) for `glm52_ep8/o_gemm_xl/decode_folded_x.py`.
  - If B+ is within 10%, submit the tuned CSV rows instead of the kernel.

---

## Part C: run order and stop rules

### C.0 Setup (once)

1. Build `xbench/` and generate all inputs in container O (A0). Verify the input sha256 values
   load identically in N.
2. Smoke-run block 0 (discarded) in both containers: the calibration kernels, plus one shape
   each of S1 and S4. This confirms the O/N copy-kernel ratio is within 2% and both A/A floors
   are below 2%.
3. Run the optional Triton 3.7 compile probe (A8). It takes minutes and does not block anything.

### C.1 Non-GEMM / fused candidates first (user priority)

Within this block, order follows the nightly priority:

| order | id | why here |
|---:|---|---|
| 1 | **S1** fused qkv-a + dual RMSNorm (+quant) | rank 1; new-op candidate; GLM FP8 baseline GEMM got ~2x faster, so verify first |
| 2 | **S4** sparse MLA decode | largest e2e upside; `sparse_mla_fwd` got a new fp8 kernel, so decisive |
| 3 | **S6** DSA indexer prepare | new-op candidate; AITER unchanged |
| 4 | **S5** Qwen3.5 prologue | new partial AITER overlap (B3 arm) |
| 5 | **S12** GDN prefill (Qwen3.5) | AITER has only the Qwen3-Next port; eager timing (not graph-safe) |
| 6 | **S3** GDN decode | AITER already has our kernel's port; T vs B (fallback in N) **and** B38 (port on Triton 3.8 in O) |
| 7 | **S10** Kimi attention residual | cheap; `attn_res_gate` only gained config tables |
| 8 | **C1** MiniMax paged GQA decode/prefill | conditional; graph-capacity schedule caveat (A4) |
| 9 | **C3** GLM index top-k | only if the per-call stream has been removed; otherwise record "blocked" |

### C.2 GEMM-dominated fused

| order | id | note |
|---:|---|---|
| 10 | **S7** MLA kc BMM + RoPE + cache (K=192) | classified GEMM-dominated (BMM is the bulk); includes arm B4 for decode rows |

### C.3 Plain GEMMs last

| order | id | note |
|---:|---|---|
| 11 | **S2** small-M FP8 block-scale GEMM | rank-2 candidate; AITER decode rows unchanged |
| 12 | **S8** BF16 N=32 skinny GEMM | B+ (tune in N) decides kernel-PR vs CSV-rows PR |
| 13 | **S9** MLA vc FP8 batched GEMM | B+ config sweep in N |
| 14 | **C4** untuned bf16 decode shape (GLM EP8 o_proj) | B+ only |
| 15 | **C2** Kimi GEMM + residual adds | vs new `fused_gemm_a16w16_mul_add` |
| 16 | **S11** FP8 prefill GEMM | AITER prefill rows 15-48% faster in nightly; most likely to drop |

### C.4 Stop rules and cross-checks

- **Stop rule per candidate.** Drop it from the PR list if the weighted speedup vs the best
  AITER arm is below 1.10x on rotating cache. The best AITER arm is max(B, B+), or B38 where it
  applies, reported separately. Also drop it if any T numerics cell fails while B passes, or if
  the cross-container calibration is out of tolerance for that block (then re-run, not drop).
- **Same-process cross-check (optional, container O only).** `wt_next/src/bench/` still runs T
  and SGLang-0.5.22 stock in one process (`python src/bench/qwen3_next.py ...`). Use it for S2,
  S3 and S11 to confirm the cross-container speedup is not an artifact. Its stock is the older
  SGLang/AITER of image O, so it is not the headline.
- **AITER-harness cross-check** per A2, using `op_tests/` in N. The ported-op tests and benches
  (`op_tests/triton_tests/gated_delta_net/test_fused_gdn_decode_qkvz.py`,
  `op_tests/op_benchmarks/triton/bench_fused_gdn_{decode,prefill}_qkvz.py`,
  `op_tests/op_benchmarks/triton/bench_paged_attention_output_gate.py`,
  `op_tests/op_benchmarks/triton/bench_fused_qk_norm_rope_gate_fp8_quant.py`) need Triton 3.8,
  so they run in O against the vendored ports.

### C.5 Deliverables

- `results/<candidate>/{O,N}_<block>.jsonl` (raw) and `results/<candidate>.csv` per A6. The CSV
  adds the columns `container_T=O`, `container_B=N`, `B38_us`, `calib_copy_ratio`, `aa_O`, `aa_N`.
- `results/summary.md`: one line per candidate (keep, drop or retune-only), the AITER backend
  each comparison ran against, and whether the ported AITER op (B38) already matches us.
- The raw `rocm-smi`, version and environment captures from both containers.
