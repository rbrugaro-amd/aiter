# AITER upstreaming assessment: templated gfx950 kernels (phase 1, CPU-only)

Scope: the 106 template files in `template_inventory.json` that serve rows on
`rbrugaro/templating-gfx950-next` (checkout `/home/rbrugaro/code/wt_next`, HEAD `88081cdc`).
Paths below are relative to `wt_next/` unless they start with `aiter/` or `nightly/`:
- `aiter/` = `templating-infra/aiter/aiter_src/aiter/`, AITER **e7d2453f2** (2026-09-25,
  0.1.23.dev183). This is the phase-1 copy.
- `nightly/` = `templating-infra/aiter/aiter_from_sglang_nightly_20261117/aiter/`, AITER
  **581579ab5** (2026-11-07, 0.1.29.dev156). It is the copy shipped in
  `lmsysorg/sglang-rocm:v0.5.23-rocm724-mi35x-20261117` (torch 2.11.0+rocm7.2, Triton 3.7.0,
  SGLang 703e763d5f) and is 816 commits newer.

**The nightly is now the baseline.** Section 0a lists what changed against it. The
closest-AITER column in the section-5 tables still cites e7d2453f2, and every group header there
carries the nightly delta. In `assessment.json`, every template has an `aiter_nightly` field
(`closest`, `changed_vs_e7d2453f2`, `tuned_shapes`). The GPU plan is in `bench_plan.md`.

Method: six read-only sub-passes, one per pack group, gathered the facts for every template
(op, stock path, closest AITER op, portability, knobs, evidence). I then grouped near-duplicates
into 25 op groups, re-checked the load-bearing claims myself, and scored everything with one
explicit formula (below). Nothing in any worktree was modified.

## 0. Caveats that apply to everything below

1. **The AITER we compare against is not the AITER behind our evidence.** The source tree here
   is `0.1.23.dev183+ge7d2453f2.d20261016` (`aiter/_version.py:1`). All of our vs-stock serving
   evidence was measured with AITER `0.1.20`, commit `fc2e5d57` (`docs/qwen3.5/README.md:31`,
   `docs/glm5.2/runtime-patches/README.md:9`, `docs/kimi-k3/README.md:13`,
   `docs/minimax-m2.5/README.md:34`, `docs/deepseek-r1-mi355x.md:12`). The newer tree adds
   per-model tuned tables for most of our exact models (`aiter/configs/model_configs/`:
   `*_glm5.2.csv`, `*_kimik3.csv`, `*_qwen3.5_397b.csv`, `*_qwen3_next_80b_a3b.csv`,
   `*_mm2.5.csv`, `dsv3_*`), and they are merged by default (`aiter/jit/core.py:556-577`). It
   also adds ops that overlap ours: `sparse_mla_fwd` (`aiter/ops/triton/attention/sparse_mla.py:326`,
   which names the GLM-5.x fp8_scalar format), `attn_res_gate`
   (`aiter/ops/triton/fusions/attn_res.py:656`) and `fused_fp8_bmm_rope_cat_and_cache_mla`
   (`aiter/ops/triton/fusions/fused_bmm_rope_kv_cache.py:388`). **Every "we beat stock" number
   is therefore a claim against an older baseline.** Phase 2 has to re-establish each one
   against the nightly AITER 581579ab5. e7d2453f2 is dated 2026-09-25 (an earlier note said
   2025, which was a typo in the brief). The nightly adds more again; see 0a.
2. **Most templating evidence is parity, not benefit.** T1/T3/T5 compare a template with the
   hand-tuned file it replaced (`notes/templating/gfx950/SUMMARY.md`). Benefit over stock comes
   from three other places:
   - (a) `docs/qwen3-next-80b/microbenchmarks.csv`. This is the only same-harness, per-kernel
     stock-vs-ours table. It came from `src/bench/` (graph replay, ABBA, 7 rounds, median;
     `src/bench/README.md:54-63`).
   - (b) family-level e2e ablations: `docs/*/ablation.{png,csv}`. These cover whole families on
     the pre-templating files.
   - (c) manifest `serving_speedup`, present for 17 templates (all Qwen3.5). The field is only
     declared (`src/artemis_runtime/contracts.py:432`), and nothing in the tree says what it
     compares against. I treat it as "candidate vs stock at selection time, comparator
     undocumented".
3. **Cross-harness numbers are biased in our favour.** Many reports contrast our T5 (hot cache,
   graph replay, same inputs every call) with AITER tuned-CSV `us` columns. AITER's `perftest`
   rotates through enough copies of the arguments to defeat the cache (`aiter/test_common.py:49-80`,
   `num_rotate_args` sized from 64x128 x L2). Its numbers are cold-cache and ours are hot. **None
   of those "2-3x vs AITER CSV" claims is evidence until it is re-measured in one harness.**
   They are listed in `assessment.json` only as hints.
4. **No SGLang source in the tree.** Stock paths are taken from the adapters' monkeypatch
   targets, plus `src/bench/gemm.py:80-112` for the FP8 linear. Whether stock actually reached
   AITER is confirmed only in three places:
   - FP8 block linears go through `fp8_utils.aiter_w8a8_block_fp8_linear` with bpreshuffle
     (`src/bench/gemm.py:92-110`).
   - GLM-5.2 MXFP4 TP4/TP8 sparse MLA uses `--dsa-decode-backend aiter`
     (`docs/glm5.2/tp4-ep1/serve_args.json:24-25`).
   - The GLM-5.2 FP8 and EP8 profiles use TileLang instead (`docs/glm52-fp8-tp8/runtime.json:42`,
     `docs/glm5.2/fp8-tp8-ep8-dp8/serve_args.json:30-31`).
5. **All 106 templates are gfx950 Gluon or Triton.** Many use `gl.amd.cdna4` intrinsics and
   inline asm. Some pass `llvm_fn_attrs`, which is only known to work on the pinned Triton.
   AITER already ships gfx950 Gluon kernels (for example `sparse_mla_fwd`), so Gluon itself is
   not a blocker. The blockers are hard-coded model dimensions and exact-M tables.

## 0a. What changed with the nightly AITER (581579ab5 vs e7d2453f2)

Method: file-list diff (665 new and 121 removed py/csv/md files), signature checks on every
baseline function cited here, and tuned-CSV coverage and `us` comparisons at the served
(M,N,K) of every GEMM candidate. The scripts are in `/tmp/aiter_assess/main/csv{cov,us}.py`
and are not shipped.

**1. Three of our kernels are already upstream as AITER ports.** The nightly ships three ops
ported from Artemis kernel-pack sources:
- **`fused_gdn_decode_qkvz`** (`nightly/ops/triton/gated_delta_net/fused_gdn_decode_qkvz.py:311`).
  The Gluon module says "Generated from Artemis kernel pack source" and cites `gdn_decode_m16.py`
  sha256 `16ec8eaf...`. That is byte-identical to our
  `qwen3_next/gdn_fused_decode_group_fp8_quant/gdn_decode_m16.py`, which the templated manifest
  still serves (`src/kernel_packs/profiles/qwen3_next_fp8_tp4_gfx950/gdn_fused_decode_group_fp8_quant_templated.json:83`).
  The op covers value/key-head ratios 1-4 and the Qwen3.5 "flat" layout
  (`fused_gdn_decode_qkvz.py:52,183,360-363`).
- **`fused_gdn_prefill_qkvz`** (`nightly/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py:1625`).
  It is ported from the "Artemis MI355 v1" Qwen3-Next pack (`:47`) and covers the Qwen3-Next
  topology only (`:73-89`, qkvz 3072 / ba 16).
- **`paged_attention_output_gate_group_fp8_quant`** (`nightly/ops/triton/attention/paged_attention_output_gate.py:192`).
  It is ported from Artemis sources (`_gluon_kernels/gfx950/attention/paged_attention_output_gate_short_context.py:38`).

All three are **gated on Triton >= 3.8** because ROCm's Triton 3.7 Gluon is incompatible
(`fused_gdn_decode_qkvz.py:89-114`, `paged_attention_output_gate.py:94`). In the nightly image,
which has Triton 3.7, they report unsupported and callers fall back to the multi-kernel chains.
Consequences for the shortlist:
- **S3 (GDN decode)** is no longer a new-op PR. What remains is a perf PR: our per-M and Qwen3.5
  schedules against AITER's generated pair kernel and ratio-generic kernel. Gap 2 -> 1; priority
  3 -> 8.
- **C1 / Qwen3.5 `r3_paog`** is already upstream, so it moves to do-not-upstream (benchmark for
  parity only).
- **S12 (GDN prefill, Qwen3.5)**: AITER now has the Qwen3-Next fused prefill but not Qwen3.5
  (qkvz 5120, ratio 4, fp32 state). The new FlyDSL `gdn_fused_prefill_fwd`
  (`nightly/ops/flydsl/gdn_fused_prefill.py:147`) is chunk-core only and needs a bf16 state.
  The gap stays 2, and the route is now concrete: extend `fused_gdn_prefill_qkvz` to the Qwen3.5
  topology. Priority 12 -> 7.

**2. New overlapping ops.**
- **`fused_qk_norm_rope_gate_fp8_quant`** (`nightly/ops/triton/rope/fused_qk_norm_rope_gate_fp8_quant.py:65`).
  It has the interleaved q/gate split, Gemma `+1` in the kernel and partial rotary. It has no KV
  cache write and 1-D positions only, and it quantizes per sequence for FMHA. S5 remains a
  partial gap, narrower than before.
- **`fused_gemm_a16w16_mul_add`** (`nightly/ops/triton/gemm/fused/fused_gemm_a16w16_mul_add.py:35`),
  `out = a*(xW^T+bias)+b`. It covers C2 with at most one pre-add. C2 gap 2 -> 1.
- **`fused_bf16_bmm_cat_quant_pad_mla`.** This is MLA decode q-prep without the cache write. It
  is an extra baseline arm for S7's decode rows.
- **`batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant`** gained
  `y_scale_out`, a per-token FP8 output quant. It is not the group-128 quant the R1 S9 template
  fuses. The gfx950 config directory for this op is still empty.

**3. Stronger baselines for existing ops.**
- **`sparse_mla_fwd`** now launches a dedicated Gluon fp8-dot kernel for fp8_scalar caches, with
  block_m=8 for H<16 (`nightly/ops/triton/attention/sparse_mla.py:475-489,998`). This bears on S4.
- **GLM FP8 qkv_a GEMM (2624x6144):** the bpreshuffle decode rows moved from CK to Triton,
  17.0 -> 6.5 us at M=1 (`nightly/configs/model_configs/glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv`).
  S1's margin shrinks but survives, since our fused op is ~3.8-5 us at M=1.
- **FP8 prefill GEMM rows at M>=8192** were retuned to FlyDSL and are 15-48% faster (G01 header
  in section 5). S11's evidence is stale: evidence 3 -> 1, priority 11 -> 12.

**4. Unchanged.**
- **Decode-M FP8 rows (S2):** every served (N,K) at M<=256 has identical tuned `us`. (2560,2048)
  is still absent from all bpreshuffle tables.
- **S6** (`indexer_qk_rope_quant_and_cache`: identical `cache.py`).
- **S7** (`fused_fp8_bmm_rope_cat_and_cache_mla`: import refactor only).
- **S8:** N=32/K=4096 still has only an M=16 row. A new FlyDSL skinny kernel is selectable only
  via tuned rows.
- **S10:** `attn_res_gate` only gained config tables.
- **MiniMax paged GQA:** `paged_attention_ragged` signature unchanged (`nightly/ops/attention.py:792`).
- **Baseline signatures** used in `bench_plan.md` are unchanged, but their line numbers moved.
  The new line numbers are in `bench_plan.md`.

**5. Triton 3.7 vs 3.8.**
- The nightly image cannot run 105 of our 106 templates, which are Gluon in the Triton 3.8
  dialect. The 106th, `qwen3_5/fp8/fp8_block_scaled_gemm_xl/xl_paired_template.py`, is plain
  Triton but uses the patched `llvm_fn_attrs` launch option.
- For the same reason, AITER's own ports of our kernels are disabled in that image.
- `bench_plan.md` therefore runs the templates in our Triton 3.8 container and the AITER and
  stock arms in the nightly image.

## 1. Scoring (explicit)

`score = 2*gap + 2*evidence + reach + portability + generality + maintenance`, max 24.

| term | 3 | 2 | 1 | 0 |
|---|---|---|---|---|
| gap vs AITER (group, per-template override) | no AITER op covers the fused computation | AITER has the stages, not the fusion | AITER has the op; evidence of a shape/schedule win | AITER has the op, tuned at our shapes, no win |
| evidence (group, per-template override) | same-harness per-kernel vs stock | positive family-level e2e ablation vs stock | cross-harness hint / undocumented `serving_speedup` | none or negative |
| reach (group) | >=3 base models | 2 base models or >=40 rows | 1 model, <40 rows | - |
| portability | plain tensor API, few deps | adapter-prepared views, sys.path/sibling imports, exact lookup | mutates framework state pools, imports sglang, custom weight repacks, rank-specific code | - |
| generality | (2 if knobs<=5, 1 if <=15, else 0) + 1 if any range dispatch | | | |
| maintenance | knobs+3*bodies <=10 | <=25 | <=50 | >50 |

The gap and evidence terms are doubled on purpose: an upstream PR only lands if AITER lacks
the op or ours is measurably faster. Plain GEMMs therefore cannot score above gap 1, and they
reach the shortlist only on evidence. Reach counts base models (GLM-5.2 FP8 and MXFP4 count as
one), because a pile of exact-M rows for one model is not reach.

## 2. Shortlist (12 PR candidates)

### 2a. Re-ranked against the nightly AITER 581579ab5 (current priority)

IDs are kept from the first pass so references stay stable.

| new priority | id | candidate | moved? | why (vs e7d2453f2) |
|---:|---|---|---|---|
| 1 | S1 | Fused MLA q/kv-a projection + dual RMSNorm (+quant) | = | Still no AITER fused op. The baseline GEMM for GLM FP8 got ~2x faster (CK -> Triton rows), so the margin shrinks |
| 2 | S2 | Small-M FP8 block-scale GEMM | = | Decode-M tuned rows unchanged at every served shape |
| 3 | S4 | Sparse MLA decode, H=8/H=64 | up 1 | Unchanged rank logic; AITER `sparse_mla_fwd` is stronger (dedicated fp8-dot Gluon kernel, block_m=8 for H<16). Strictly bench-gated |
| 4 | S6 | DSA indexer prepare | up 2 | No change in AITER (`cache.py` identical) |
| 5 | S7 | MLA kc BMM + RoPE + cache (K=192) | up 2 | AITER fused op unchanged; new bf16 q-prep op is an extra arm only |
| 6 | S5 | Qwen3.5 attention prologue | down 1 | New `fused_qk_norm_rope_gate_fp8_quant` covers the gate split and Gemma norm, but not the cache write or mRoPE. The gap narrows |
| 7 | S12 | Fused GDN prefill, Qwen3.5 | up 5 | AITER ported our Qwen3-Next fused prefill (`fused_gdn_prefill_qkvz`) but not Qwen3.5. A ready extension path, and still not covered |
| 8 | S3 | GDN decode fused | **down 5** | **Already upstream:** `fused_gdn_decode_qkvz` was generated from our `gdn_decode_m16.py` and covers Qwen3.5 (flat layout, ratio 4). This is now a perf PR only, and only if our per-M schedules win |
| 9 | S8 | BF16 N=32 skinny GEMM | down 1 | Still untuned; may become a tuned-row PR |
| 10 | S9 | MLA vc FP8 batched GEMM | down 1 | gfx950 configs still empty; new `y_scale_out` is per-token, not group-128 |
| 11 | S10 | Kimi attention residual (`attn_res_gate`) | down 1 | AITER only added config tables |
| 12 | S11 | FP8 prefill GEMM | down 1 | AITER prefill rows at M>=8192 are 15-48% faster (FlyDSL). Our evidence was against the old CK rows; likely to drop |

Changes outside the shortlist:
- **C1:** the Qwen3.5 `r3_paog` member moves to do-not-upstream. It is already upstream as
  `paged_attention_output_gate_group_fp8_quant`. C1 keeps the MiniMax paged GQA templates.
- **C2** weakens: `fused_gemm_a16w16_mul_add` exists.
- **DNU:** no DNU item moved back into the list. The MoE DNU is reinforced by
  `qwen3_5_397b_tuned_grouped_fmoe.csv`.

### 2b. First-pass table (against e7d2453f2; "AITER today" column = e7d2453f2)

The rows below are the original analysis. Read the "AITER today" column together with 2a and
section 0a.

| # | candidate (PR shape) | lead template | templates / rows / models | AITER today | why it is on the list | main risk |
|---|---|---|---|---|---|---|
| S1 | **New op: fused MLA q/kv-a projection + dual RMSNorm (+rope split, +FP8 group quant)** | `glm52_fp8/qkv_a_norm_r3/qkv_a_norm_template.py` | 4 / 164 / GLM-5.2 (fp8, mxfp4, ep8), DeepSeek-R1 | Pieces only: `gemm_a8w8_blockscale_bpreshuffle` (`aiter/ops/gemm_op_a8w8.py:1006`) or `gemm_a16w16` + `fused_qk_rmsnorm` (`aiter/ops/fused_qk_rmsnorm_group_quant.py:179`) / `fused_rms_fp8_group_quant` (`aiter/ops/triton/quant/fused_fp8_quant.py:164`) | Only true "no-equivalent" fusion with large reach. Family ablations save 1.0-2.0 ms per decode step on GLM-5.2 FP8 (`docs/glm5.2/fp8-tp8-ep8-dp8/ablation.csv:74-82`, `docs/glm52-fp8-tp{4,8}/ablation.png`). Lead already has a plain tensor API | Hard-coded N=2624/K=6144 (GLM) and 2112/7168 (R1); regime modules loaded by path; R1 variant returns as_strided views of one backing buffer |
| S2 | **Small-M (M<=256) FP8 block-scale GEMM path** for `gemm_a8w8_blockscale[_bpreshuffle]` | `qwen3_next/fp8_preshuffled_gemm_xl/xl_register_template.py` | 8 / 79 / Qwen3-Next, GLM-5.2, DeepSeek-R1, MiniMax-M2.5 | Equivalent op: CK/asm `gemm_a8w8_blockscale[_bpreshuffle]` (`aiter/ops/gemm_op_a8w8.py:862,1006`), tuned rows exist (e.g. `model_configs/a8w8_blockscale_tuned_gemm_qwen3_next_80b_a3b.csv:704`, M=1 N=2560 K=2048, 6.26 us) | Strongest same-harness evidence: originals 2.4-5.5x vs stock AITER bpreshuffle at decode M (`docs/qwen3-next-80b/microbenchmarks.csv`; e.g. 1x2560x2048 7.77 -> 1.67 us). 1.67 us for 5.2 MB of weights is plausible, not a hot-cache artifact on its own. Reach is 4 models | Decode microbench used one weight set, i.e. hot cache (`src/bench/README.md:50`). The newer AITER tables may have closed part of the gap. Knob tables are exact-shape |
| S3 | **New op: GDN decode = conv1d update + gated delta rule + gated RMSNorm + FP8 group quant** | `qwen3_next/gdn_fused_decode_group_fp8_quant_templated/gdn_decode_template_v2.py` | 4 / 14 / Qwen3-Next, Qwen3.5 | Stages only: `causal_conv1d_update_split_qkv` (`aiter/ops/triton/gated_delta_net/causal_conv1d_decode.py:25`), `fused_rearrange_sigmoid_gated_delta_rule` (`.../fused_rearrange_sigmoid_gdr.py:186`) or HIP `fused_split_gdr_update` (`aiter/ops/fused_split_gdr_update.py:37`), `gated_rmsnorm_fp8_group_quant` (`aiter/ops/gated_rmsnorm_fp8_group_quant.py:24`) | Same-harness 1.5-3.1x vs stock (`docs/qwen3-next-80b/microbenchmarks.csv` rows 2-37), ~21% TPOT at C1 (`docs/qwen3-next-80b/ablation.csv`). End-to-end parity with the originals in serving (`docs/validation-qwen3-next-gdn-templated.md`) | Mutates SGLang mamba state pools in place; 52 knobs; needs an API that takes `conv_state, ssm_state, indices` like AITER's existing GDR ops |
| S4 | **Sparse MLA decode schedules for H=8 (TP8) and H=64 (DP/EP)** contributed into `sparse_mla_fwd` | `glm52/sparse_paged_mla_xl/decode_tp4_small.py` (1 knob, plain API) | 11 / 60 / GLM-5.2 | Equivalent: `sparse_mla_fwd` (`aiter/ops/triton/attention/sparse_mla.py:326`, fp8_scalar = GLM-5.x, `:358-361`); stock was asm `mla_decode_fwd` with 8->16 head padding (`src/artemis_sglang/adapters/glm/attention.py:567-575`) or TileLang | Largest e2e numbers in the inventory: +4.5 ms/forward at TP4 C1-C16 and +8.9/+13.8 ms at TP8 C128/C256 (`docs/glm5.2/tp*-ep1/ablation.png`), up to +28 ms per step on FP8 TP8 C256 (`docs/glm52-fp8-tp8/ablation.png`) | Baseline was asm/TileLang, **never** `sparse_mla_fwd`. API mismatch: we take dense `slots[M,2048]`, AITER takes CSR `kv_indptr/kv_indices`. EP8 kernels return transposed-stride outputs |
| S5 | **Extend `fused_qk_norm_mrope_3d_cache_pts_quant_shuffle` with the Qwen3.5 q/gate split**, or contribute our Triton op | `qwen3_5/mxfp4/fused_attention_qk_norm_rope_kv_cache_xl/xl_decode_small_template.py` | 2 / 8 / Qwen3.5 | Partial: the HIP op has gemma_norm, mrope and rotary_dim (`aiter/ops/fused_qk_norm_mrope_cache_quant.py:10-39`) but no gate split; Triton `fused_qkv_split_qk_norm_rope_cache` (`aiter/ops/triton/rope/fused_qkv_split_qk_norm_rope_cache.py:34`) has the gate but no mRoPE | Clear functional gap (the union of two AITER ops). Recorded 1.37x on rows 1/2 (manifest), and the sibling unfused-stock comparison on Qwen3-Next is 4-16x (`docs/qwen3-next-80b/microbenchmarks.csv`) | Single model; must be timed against AITER's HIP op plus a separate gate split, not against stock |
| S6 | **New op: DSA indexer prepare** (wq_b/wk/weights projections + LN + RoPE + Hadamard + FP8 quant + index-cache write) | `glm52/indexer_prepare_r3/decode_tp4_template.py` (2 knobs, 1 body) | 3 / 26 / GLM-5.2 | Partial: `indexer_qk_rope_quant_and_cache` covers the post-projection epilogue only (`aiter/ops/cache.py:123`) | +0.8-1.3 ms/forward in decode, +6-7 ms in TP4 prefill (`docs/glm5.2/tp4-ep1/ablation.png`). Applies to every DSA model (DeepSeek-V3.2, GLM-5.x) | Fuses BF16 projections, so it competes with tuned GEMMs; adapter-prepared rope table and uint8 index-cache view (`adapters/glm/attention.py:216-266`) |
| S7 | **K=192 nope-dim support and gfx950 tuning for `fused_fp8_bmm_rope_cat_and_cache_mla`** | `glm52/kc_rope_cache_xl/kc_prefill_xl.py` | 5 / 48 / GLM-5.2 | Equivalent fusion exists (`aiter/ops/triton/fusions/fused_bmm_rope_kv_cache.py:388`); no gfx950 config | Big prefill savings: +4-14 ms/forward TP8 (`docs/glm5.2/tp8-ep1/ablation.png`), +10-28 ms on FP8 TP4 (`docs/glm52-fp8-tp4/ablation.png`) | Baseline was SGLang's unfused path, not AITER's fused op. Most knobs only reproduce code order |
| S8 | **Skinny-N (N=32, K=4096) BF16 GEMM path/configs** in `gemm_a16w16` | `qwen3_5/mxfp4/bf16_gemm_xl/xl_n32_fold_template.py` | 6 / 27 / Qwen3.5 | Equivalent op; shape untuned. `qwen3_5_397b_bf16_tuned_gemm.csv` only has N=64 (TP2) at M<=8; skinny dispatch per `aiter/tuned_gemm.py:87-109` | Recorded per-row 1.4-4.4x (manifests); simple API, high portability | Launch-floor kernels (~2 us); a tuned CSV row may capture most of the gain. Same class exists for Qwen3-Next N=16 (1.2-2.8x, `docs/qwen3-next-80b/microbenchmarks.csv`), not templated |
| S9 | **gfx950 configs/kernel for `batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant`** (+optional fused output group quant from R1) | `glm52/vc_reconstruct_r3/vc_prefill_tpl.py` | 2 / 16 / GLM-5.2, DeepSeek-R1 | Equivalent math (`aiter/ops/triton/gemm/batched/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py:13`); no tuned rows for B=8/16, N=128/256, K=512 | +1.7-6.3 ms/forward TP8 prefill (`docs/glm5.2/tp8-ep1/ablation.png`) | R1 template has no benefit data; `vc_onewave_tpl.py` is negative and excluded |
| S10 | **Perf PR to `attn_res_gate`** (Kimi attention residual: in-place bank row write, decode M<=256) | `kimi_k3/attention_residual_norm_xl/arn_wide_xl.py` | 2 / 118 / Kimi-K3 | Equivalent op with the same contract, including `add_hidden`, `output_rms_weight` and `close_block` (`aiter/ops/triton/fusions/attn_res.py:656-711`) | AITER's own comment puts its device time at ~10-25 us for N<=128 (`attn_res.py:60-66`); ours is 3-5 us at T5 (`notes/.../kimi_k3_attention_residual_norm/t5_wide_a.log`). High call count (every layer) | Unpaired and cross-harness. `close_block` returns a new tensor while ours writes in place, so the contracts differ. No vs-stock data at all |
| S11 | **FP8 block-scale GEMM prefill path (M 1024..32768)**, preshuffled and bf16-act dyn-quant | `qwen3_next/fp8_preshuffled_gemm_xl/xl_packed_template.py` | 5 / 57 / Qwen3-Next, GLM-5.2, DeepSeek-R1 | Equivalent (CK/asm bpreshuffle; Triton `gemm_a16w8_blockscale(prequant=True)`, `aiter/ops/triton/gemm/basic/gemm_a16w8_blockscale.py:21`) | Same-harness 1.4-2.2x vs stock with rotating weights (`docs/qwen3-next-80b/microbenchmarks.csv`, `src/bench/README.md:41-50`); GLM Q-B prefill +1.5-5.5 ms per forward | `xl_native_rows` switches path based on scale magnitude (exact-BF16 replay), which upstream will question; R1 prefill has no isolated data |
| S12 | **Fused GDN prefill (conv + chunk GDR + gated norm)**, conditional | `qwen3_5/mxfp4/gdn_prefill_xl/xl_gdn_prefill_dynamic.py` (range dispatch) | 3 / 18 / Qwen3.5 | Stages: `chunk_gated_delta_rule[_opt]` (`aiter/ops/triton/gated_delta_net/gated_delta_rule.py:211,354`), conv prefill, gated norm; the AITER chunk_gdn_h table for Qwen3.5-397B is gfx942-only | `serving_speedup` 2.8-3.8x on 11 rows (comparator undocumented); GDN family TTFT +6-16% at 1024/1024 | The MXFP4 profile routes GDN prefill to native for a forward with exactly one uncached 5120-token sequence (`docs/qwen3.5/README.md:21-24`). The whole MXFP4 pack regresses TTFT at 8 of 72 points (worst 8192/1024 C1 +97.6%, `docs/qwen3.5/README.md:119-121`), with no attribution to a family. Not graph-safe; r3 rows only match exact length tuples |

**Conditional (decide after the GPU phase), not PRs yet:**
- **C1. Paged GQA decode/prefill, FP8 KV (MiniMax 12q/2kv/128).** The Qwen3.5 output-gate member was dropped after the nightly check: it is already upstream as `nightly/ops/triton/attention/paged_attention_output_gate.py:192`.
  - Family ablations: +0.1-0.6 ms (decode) and +3.4-4.7 ms (prefill) per forward.
  - Not upstreamable as is: the schedule is keyed on `kv_indices.numel()`, i.e. the CUDA-graph
    buffer capacity, not the actual sequence lengths (`out_D.json` notes for the r3 templates).
- **C2. Kimi BF16 GEMM + two residual adds** (`kimi_k3/bf16_gemm_add_xl/gemm_add_m1_16.py`). The nightly AITER has `fused_gemm_a16w16_mul_add` (`out=a*GEMM+b`), so only the second addend is missing. Expect DNU.
  - This is a real epilogue gap in AITER, whose fused GEMMs offer `*_mul_add` epilogues only for
    fp8/fp4.
  - The evidence is cross-harness only.
- **C3. DSA index scores + top-k, prefill** (`glm52/paged_index_scores_topk_xl/prefill_tp4_large_x.py`).
  - Family gain is +5-11 ms per forward at C>=4.
  - It allocates a new `torch.cuda.Stream` on every call
    (`.../prefill_tp4_large_x.py:893-898`) and hard-codes the index-cache page layout. Fix both
    before any PR.
- **C4. Untuned BF16 decode shapes**, e.g. the GLM-5.2 EP8 o_proj N=6144, K=16384
  (`glm52_ep8/o_gemm_xl/decode_folded_x.py`).
  - Contribute AITER tuner CSV rows rather than kernels, since the kernel runs near the HBM roof.

## 3. Do not upstream (with reason)

| group | templates | reason |
|---|---|---|
| G16 BF16 GEMM medium/large M | `glm52/o_gemm_xl/tp4_prefill_panel_x.py`, `glm52/q_b_gemm_r3/glm52_q_b_gemm_n2048_m4193_template.py`, `glm52/q_b_gemm_templated/glm52_q_b_gemm_n2048_prefill_template.py`, `kimi_k3/bf16_gemm_r3/n7168_m128_256_template.py`, `minimax_m2_5/bf16_gemm_xl/bf16_gemm_xl_template.py` | No demonstrated benefit. The ablation picks native or is slightly negative: GLM TP8 Q-B -1.86 ms at C4, MiniMax O-proj -0.035/-0.097 ms. The q_b range template also compiles once per distinct M in mode 0 |
| G20 MoE blocks | `qwen3_5/fp8/qwen_router_complete_moe_templated/router_moe_{small,mid}.py`, `qwen3_5/mxfp4/mxfp4_router_complete_moe_{r3,xl}/*`, `glm52_ep8/fused_moe_xl/decode_ep8_m8_m16_x.py` | AITER `fused_moe` is already tuned at the exact Qwen3.5 shapes (`aiter/configs/model_configs/qwen3_5_397b_fp4_tuned_fmoe.csv`, `qwen3_5_397b_fp8_ptpc_tuned_fmoe.csv`). Our FP8 rows gain only 1.01-1.07x. The kernels import sglang at module level and need AITER-shuffled weights. The EP8 kernel hard-codes rank offsets and has a 76-knob sibling |
| G18 RMSNorm family | `qwen3_5/fp8/fused_add_rmsnorm_bf16_group_fp8_quant_templated/fadd_rmsnorm_template.py`, `.../fused_add_gemma_rmsnorm_group_fp8_quant_templated/fagr_template.py`, `qwen3_5/mxfp4/bf16_*gemma_rmsnorm*` (3), `qwen3_coder/rmsnorm_templated/qc_rms_template.py` | `add_rmsnorm_quant(..., group_size, gemma_norm)` (`aiter/ops/rmsnorm.py:586-597`) and `rmsnorm2d_fwd[_with_add]` cover these. The e2e norm-family ablation is within noise. The kernels sit at the launch floor and carry address-offset tricks (e.g. a 16384x4096 backing buffer allocated per call for row 15360). One thing to file as an AITER issue: whether `add_rmsnorm_quant` accepts an fp32 gamma (the `fadd` numerics) |
| G15 BF16 skinny decode, general N | 8 templates (Kimi n2112/n2304/n1536, GLM o/q_b decode, GLM EP8 q_b ring, DSv4 n512) | AITER has tuned rows for nearly all of these shapes in merged model_configs. Our wins are cross-harness only (hot vs rotating) or small at family level. For DSv4 n512, stock measured 22 us while AITER's own table says 5.7 us, so stock was not on AITER's best path. Exception: C4 above |
| G14 BF16 LM head | 8 templates | Wins exist only against hipBLASLt/torch (1.26-1.65x). End-to-end is about zero (`docs/qwen3-next-80b/ablation.csv`, lm_head 0.01 ms), and AITER has tuned vocab shapes. At most contribute CSV rows |
| G21 dense MLA decode | `deepseek_r1/mla_decode_xl/dsr1xl_mla_template.py`, `kimi_k3/paged_attention_decode_{r3,xl}/*` | Stock for R1 is AITER asm `mla_decode_fwd`, and there is no vs-stock data. The Kimi 12-head owners store BF16 partials: 3.5-4% of elements are more than 2 ulp from an fp32 reference, and 20-24% on cancellation draws |
| G19 dense prefill MHA d256 | `glm52/dense_prefill_attention_xl/dense_prefill_x.py`, `glm52_ep8/dense_prefill_attention_xl/prefill_ep8_m128_m1024_x.py` | AITER CK `flash_attn_varlen_func` covers it. The kernels need patched Triton (`llvm_fn_attrs`), and their numerics differ from the neighbouring range files |
| G23 KDA whole-layer decode | `kimi_k3/kda_layer_decode_r3/kda_m32_m64_template.py` | It owns the projections and the sglang KDA state-pool ABI. Only Artemis-vs-Artemis timings exist. AITER `fused_kda_decode_unified` (`aiter/ops/triton/gated_delta_net/fused_kda_decode_unified.py:24`) covers the non-GEMM part |
| G24 mHC pre | `deepseek_v4_pro/mhc_pre_templated/mhc_pre_decode_m8_m32.py` | AITER `mhc_pre_dsv4` has the same ABI. Gains are 0-18% at M=32, with no data at M=8/16. Uses raw LLVM intrinsics |
| G25 MTP helpers | `glm52/mtp_softmax_top1_r3/glm52_mst_template.py`, `glm52/mtp_input_combine_r3/glm52_mic_m8_m16_template.py` | No benefit data. softmax_top1 hard-codes vocab 154880 and differs from torch.softmax in ~39% of elements |
| G05 (Coder) | `qwen3_coder/fused_attention_qk_norm_rope_fp8_kv_cache_xl/xl_split_half_template.py` | AITER `fused_qkv_split_qk_norm_rope_cache` and `fused_qk_norm_rope_cache_quant_shuffle` cover it. No data |
| G12 (Qwen3.5 output gate) | `qwen3_5/mxfp4/paged_attention_output_gate_r3/r3_paog_template.py` | Already upstream in AITER 581579ab5 (`nightly/ops/triton/attention/paged_attention_output_gate.py:192`, ported from Artemis; Triton>=3.8). Benchmark for parity only |
| G11 (one-wave) | `glm52/vc_reconstruct_r3/vc_onewave_tpl.py` | Slightly negative vs native at the M=1/2 rows it serves |
| G01 (paired) | `qwen3_5/fp8/fp8_block_scaled_gemm_xl/xl_paired_template.py` | The recorded 3.3-5.8x is implausible against AITER's tuned 9.7-20.5 us for 128-256 x 4608 x 4096. It needs a row-major W copy. Re-measure before anything else |

## 4. Open questions (ranked by how much they can change the shortlist)

1. **Who ported our kernels into AITER, and is there a channel?**
   - The nightly AITER already carries ports of our Qwen3-Next GDN decode, GDN prefill and
     paged-attention-output-gate kernels. Their headers cite our file names and the sha256 of
     `gdn_decode_m16.py`.
   - Before opening any PR we need to know who did this (AMD, or someone on our side) and through
     which process. The answer sets the upstream route for S3, S12 and S5, since the pattern is
     "port a kernel-pack file, add a `*_supported()` gate".
   - It also bears on attribution and on whether AITER wants the template or the generated
     single-shape files.
2. **Does the nightly AITER (581579ab5) still lose?** All vs-stock evidence used AITER fc2e5d57
   (0.1.20). Most at risk:
   - S4: `sparse_mla_fwd` now has a dedicated fp8-dot kernel.
   - S11: prefill GEMM rows are 15-48% faster.
   - S1: the GLM FP8 qkv_a GEMM rows are ~2x faster.
   - S3: the AITER op is a derivative of our own kernel.

   Phase 2 runs exactly this check.
3. **AITER's ported ops are off in the nightly image.** They are gated on Triton >= 3.8, and
   the nightly has Triton 3.7. The "AITER as shipped" arm for GDN decode/prefill and the output
   gate is therefore the multi-kernel fallback. The fair upstream comparison also needs the
   AITER op itself, run under Triton 3.8 (arm B38 in `bench_plan.md`). Does SGLang 703e763d5f
   call these ops at all when Triton >= 3.8? The SGLang source is not in the tree.
4. **What does `serving_speedup` compare against?** It is undocumented
   (`src/artemis_runtime/contracts.py:432`), and it is the only per-row benefit number for S5, S8
   and S12.
5. **Hot vs cold weights for decode GEMMs.** The Qwen3-Next decode GEMM microbench used one
   weight set (`src/bench/README.md:50`). Phase 2 must report rotating-weight numbers.
6. **API redesign for S1 and S12.**
   - S12: AITER's `fused_gdn_prefill_qkvz` signature is the template to follow (`projected_qkvz`,
     `projected_ba`, `conv_state`, `delta_state`, `cache_indices`, `cu_seqlens`,
     `has_initial_state`, ...). The open part is a Qwen3.5 topology parameter.
   - S1: the R1 variant returns views of a shared backing buffer and needs an output-buffer API.
7. **Generality vs exact-M tables.**
   - AITER's own ports keep per-band schedules, selected by a runtime band dispatch
     (`fused_gdn_prefill_qkvz.py:141 _select_tile_key`).
   - That is a precedent for band tables over pure heuristics. Is the same acceptable for the
     GEMM candidates, or do maintainers want tuned-CSV rows there?
8. **Cross-container comparability.** Templates must run in our Triton 3.8 container and AITER
   in the nightly image (Triton 3.7, different SGLang). Torch is 2.11+rocm7.2 in both, so the
   cross-container error should be small but has to be measured (A/A per container, plus a shared
   reference kernel run in both; see `bench_plan.md` A8).
9. **Licensing/attribution and the generated-file style.** Does an upstream PR ship our
   generator, or hand-cleaned per-band kernels? AITER's ports say "Generated from Artemis kernel
   pack source ... see build_aiter_kernel.py".
10. **AITER dates (resolved).** e7d2453f2 is 2026-09-25; 581579ab5 is 2026-11-07 (per the
    coordinator and `aiter_from_sglang_nightly_20261117/SOURCE.txt`).

## 5. Ranked table, all 106 templates, grouped by op

Groups are ordered by their best template score; templates within a group by score. `rec`:
shortlist id (S#), conditional (C#), or DNU (do not upstream; reason in section 3 or in
`assessment.json` `recommendation_note`). `gap` and `ev` include per-template overrides.
Closest-AITER text is abbreviated; full text and citations are in `assessment.json`.

### G02: Fused MLA q/kv-a projection GEMM + dual RMSNorm (+rope split, +FP8 group quant)

4 templates, 164 rows, models: DeepSeek-R1, GLM-5.2. Base gap 3, base evidence 2, reach 2.

**AITER 581579ab5 (nightly):** No new fused op. Composition = gemm_a8w8_blockscale_bpreshuffle / tgemm.mm + fused_qk_rmsnorm (aiter_from_sglang_nightly_20261117/aiter/ops/fused_qk_rmsnorm_group_quant.py:179) or fused_rms_fp8_group_quant (ops/triton/quant/fused_fp8_quant.py:164). ops/fused_norm_quant_gemm.py is RMSNorm->quant->GEMM (wrong order for this op).  
**Changed vs e7d2453f2:** Gap unchanged (3). The AITER baseline GEMM got faster for GLM FP8: 2624x6144 bpreshuffle decode rows moved CK->Triton, M=1 17.0->6.5 us, M=16 15.5->8.4 us (configs/model_configs/glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv; M=2 row dropped); bf16 2624x6144 M=1 8.7->7.4 us. Our fused kernel (~3.8-5 us at M=1 incl. both norms) is still below the GEMM-only time, but the margin shrank.

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `glm52_fp8/qkv_a_norm_r3/qkv_a_norm_template.py` | 83 | mixed | 1/7 | medium | two ops: gemm_a8w8_blockscale_bpreshuffle | 3 | 2 | 19 | S1 (lead) |
| 2 | `glm52_ep8/qkv_a_norm_xl/decode_async_x.py` | 3 | exact | 8/1 | high | tuned_gemm gemm_a16w16 | 3 | 2 | 18 | S1 |
| 3 | `deepseek_r1/qkv_a_norm_quant_templated/qkv_a_norm_quant_template.py` | 29 | mixed | 1/10 | medium | gemm_a8w8_blockscale | 3 | 1 | 16 | S1 |
| 4 | `glm52/qkv_a_norm_xl/qkv_a_norm_template.py` | 49 | mixed | 49/16 | medium | No single-op equivalent. Closest is two ops: the tuned bf16 GEMM | 3 | 2 | 15 | S1 |

### G01: FP8 block-scaled GEMM (A fp8 per-1x128 / bf16 dyn-quant, W fp8 128x128; plain or AITER-preshuffled W)

14 templates, 124 rows, models: DeepSeek-R1, GLM-5.2, MiniMax-M2.5, Qwen3-Next-80B, Qwen3.5-397B. Base gap 1, base evidence 3, reach 3.

**AITER 581579ab5 (nightly):** Same ops: gemm_a8w8_blockscale (aiter_from_sglang_nightly_20261117/aiter/ops/gemm_op_a8w8.py:1289), gemm_a8w8_blockscale_bpreshuffle (:1495), Triton gemm_a16w8_blockscale (ops/triton/gemm/basic/gemm_a16w8_blockscale.py:21); new FlyDSL blockscale backends (ops/flydsl/gemm_a8w8_blockscale.py, kernels/gemm_a8w8_blockscale_8wave.py).  
**Changed vs e7d2453f2:** Signatures unchanged (line numbers moved). Tuned rows at served decode M (<=256) are unchanged for every served (N,K). Prefill rows at M>=8192 were re-tuned to FlyDSL and are 15-48% faster: e.g. 7168x2048 M8192 221.4->128.4 us (plain), 4096x2048 M16384 187.6->130.0 us (bpreshuffle), 3072x1536 M8192 82.3->56.8 us. (2560,2048) is still absent from every bpreshuffle table. Effect: S2 unchanged; the S11 prefill evidence (measured against the older CK rows) is now stale.

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `qwen3_next/fp8_preshuffled_gemm_xl/xl_register_template.py` | 5 | exact | 1/3 | medium | gemm_a8w8_blockscale_bpreshuffle | 1 | 3 | 18 | S2 (lead) |
| 2 | `minimax_m2_5/fp8_gemm_r3/fp8_gemm_r3_template.py` | 4 | exact | 5/1 | high | aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale(prequant=True) | 1 | 2 | 17 | S2 |
| 3 | `glm52_fp8/q_b_gemm_xl/prefill_x.py` | 3 | ranges | 1/1 | medium | aiter gemm_a8w8_blockscale_bpreshuffle | 1 | 2 | 17 | S11 |
| 4 | `deepseek_r1/o_gemm_prefill_templated/o_gemm_prefill_template.py` | 19 | mixed | 1/3 | high | gemm_a16w8_blockscale(prequant=True) / per1x128 quant + gemm_a8w8_bloc | 1 | 1 | 16 | S11 |
| 5 | `deepseek_r1/kv_b_gemm_templated/kv_b_gemm_template.py` | 20 | mixed | 1/4 | high | gemm_a16w8_blockscale(prequant=True) | 1 | 1 | 15 | S11 |
| 6 | `minimax_m2_5/fp8_gemm_xl/fp8_gemm_xl_template.py` | 6 | exact | 14/2 | high | aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale(prequant=True) | 1 | 2 | 15 | S2 |
| 7 | `deepseek_r1/q_b_gemm_templated/q_b_gemm_template.py` | 28 | mixed | 2/6 | medium | gemm_a8w8_blockscale | 1 | 1 | 14 | S2 |
| 8 | `qwen3_next/fp8_preshuffled_gemm_xl/xl_slot_template.py` | 4 | exact | 27/2 | medium | gemm_a8w8_blockscale_bpreshuffle | 1 | 3 | 14 | S2 |
| 9 | `qwen3_next/fp8_preshuffled_gemm_xl/xl_packed_template.py` | 3 | mixed | 0/3 | low | gemm_a8w8_blockscale_bpreshuffle | 1 | 1 | 14 | S11 (lead) |
| 10 | `qwen3_5/fp8/fp8_block_scaled_gemm_xl/xl_paired_template.py` | 3 | exact | 15/1 | high | Triton gemm_a8w8_blockscale | 1 | 1 | 13 | DNU |
| 11 | `qwen3_next/fp8_preshuffled_gemm_xl/xl_native_rows_template.py` | 9 | mixed | 6/4 | low | gemm_a8w8_blockscale_bpreshuffle | 1 | 1 | 12 | S11 |
| 12 | `glm52_fp8/o_gemm_xl/split_k_decode_x.py` | 7 | exact | 26/1 | medium | aiter gemm_a8w8_blockscale_bpreshuffle | 1 | 2 | 12 | S2 |
| 13 | `glm52_fp8/o_gemm_xl/tp8_decode_x.py` | 4 | exact | 17/2 | low | aiter gemm_a8w8_blockscale_bpreshuffle | 1 | 2 | 12 | S2 |
| 14 | `deepseek_r1/o_gemm_xl/o_gemm_xl_template.py` | 9 | exact | 12/6 | medium | gemm_a8w8_blockscale | 1 | 1 | 11 | S2 |

### G03: Sparse (DSA top-k) MLA decode, fp8_scalar KV [slots,576], GLM-5.2 H=8/16/64

11 templates, 60 rows, models: GLM-5.2. Base gap 1, base evidence 2, reach 2.

**AITER 581579ab5 (nightly):** sparse_mla_fwd (aiter_from_sglang_nightly_20261117/aiter/ops/triton/attention/sparse_mla.py:696) now dispatches a dedicated Gluon fp8-dot kernel for fp8_scalar (GLM-5.x) caches (_gluon_kernels/gfx950/attention/sparse_mla_fp8.py), with a heads-per-program policy (sparse_mla.py:475-489) and block_m=8 for H<16 (:998).  
**Changed vs e7d2453f2:** Baseline materially stronger and now explicitly handles H=8 without padding; new kwargs (extra_kv_scale, inv_rope_*, out_mxfp8), positional API unchanged. Still no head-to-head data; S4 value depends entirely on the GPU result.

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `glm52/sparse_paged_mla_xl/decode_tp4_small.py` | 6 | exact | 1/2 | high | aiter.mla.mla_decode_fwd | 1 | 2 | 16 | S4 (lead) |
| 2 | `glm52/sparse_paged_mla_xl/decode_tp8_narrow.py` | 4 | exact | 4/1 | high | aiter.mla.mla_decode_fwd | 1 | 2 | 16 | S4 |
| 3 | `glm52_fp8_ep8/sparse_paged_mla_xl/decode_ep8_m12_m24_x.py` | 2 | exact | 4/1 | medium | aiter sparse_mla_fwd | 1 | 2 | 15 | S4 |
| 4 | `glm52_fp8/sparse_paged_mla_r3/spm_tp8_split_template.py` | 2 | exact | 4/1 | medium | aiter sparse_mla_fwd | 1 | 2 | 15 | S4 |
| 5 | `glm52_ep8/sparse_paged_mla_r3/decode_ep8_template.py` | 2 | exact | 4/1 | medium | aiter sparse_mla_fwd | 1 | 2 | 15 | S4 |
| 6 | `glm52/sparse_paged_mla_xl/decode_tp8_deep.py` | 2 | exact | 7/2 | high | aiter.mla.mla_decode_fwd | 1 | 2 | 14 | S4 |
| 7 | `glm52_fp8/sparse_paged_mla_r3/spm_tp8_policy_template.py` | 29 | exact | 13/2 | medium | aiter sparse_mla_fwd | 1 | 2 | 13 | S4 |
| 8 | `glm52_fp8_ep8/sparse_paged_mla_xl/decode_ep8_m16_m32_x.py` | 2 | exact | 18/1 | medium | aiter sparse_mla_fwd | 1 | 2 | 12 | S4 |
| 9 | `glm52_ep8/sparse_paged_mla_r3/decode_ep8_d16_d32_template.py` | 2 | exact | 19/1 | medium | aiter sparse_mla_fwd | 1 | 2 | 12 | S4 |
| 10 | `glm52_fp8/sparse_paged_mla_xl/decode_tp8_m8_x.py` | 5 | exact | 29/2 | medium | aiter sparse_mla_fwd | 1 | 2 | 11 | S4 |
| 11 | `glm52_fp8_ep8/sparse_paged_mla_xl/decode_ep8_m1_m8_x.py` | 4 | exact | 34/2 | medium | aiter sparse_mla_fwd | 1 | 2 | 11 | S4 |

### G09: DSA indexer prepare: q/k/weights projections + LN + RoPE + Hadamard + FP8 quant + index-cache write

3 templates, 26 rows, models: GLM-5.2. Base gap 2, base evidence 2, reach 1.

**AITER 581579ab5 (nightly):** indexer_qk_rope_quant_and_cache unchanged (aiter_from_sglang_nightly_20261117/aiter/ops/cache.py:123); rotate_activation now at ops/quant.py:1361.  
**Changed vs e7d2453f2:** Unchanged.

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `glm52/indexer_prepare_r3/decode_tp4_template.py` | 10 | exact | 2/1 | medium | aiter indexer_qk_rope_quant_and_cache | 2 | 2 | 16 | S6 (lead) |
| 2 | `glm52/indexer_prepare_xl/decode_tp8_lds_x.py` | 2 | exact | 4/2 | medium | aiter indexer_qk_rope_quant_and_cache | 2 | 2 | 16 | S6 |
| 3 | `glm52/indexer_prepare_key_only_xl/prefill_tp4_x.py` | 14 | mixed | 6/5 | medium | aiter indexer_k_quant_and_cache | 2 | 2 | 15 | S6 |

### G10: MLA kc absorb: FP8 BMM (q_nope x W_UK) + RoPE + concat + KV-cache write

5 templates, 48 rows, models: GLM-5.2. Base gap 1, base evidence 2, reach 2.

**AITER 581579ab5 (nightly):** fused_fp8_bmm_rope_cat_and_cache_mla (aiter_from_sglang_nightly_20261117/aiter/ops/triton/fusions/fused_bmm_rope_kv_cache.py:384), only import refactor; NEW fused_bf16_bmm_cat_quant_pad_mla (decode q-prep: bf16 W_UK BMM + cat + per-tensor FP8 + head padding, no cache write) and fused_mla_prefill_qkv_fp8.  
**Changed vs e7d2453f2:** Unchanged gap; the new bf16 BMM op is an extra baseline arm for the decode rows (it skips the cache write).

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `glm52_ep8/kc_rope_cache_r3/prefill_ep8_m128_to_2048_tpl.py` | 4 | exact | 5/1 | high | aiter/ops/triton/fusions/fused_bmm_rope_kv_cache.py:fused_fp8_bmm_rope | 1 | 2 | 16 | S7 |
| 2 | `glm52/kc_rope_cache_xl/kc_prefill_xl.py` | 8 | mixed | 13/1 | medium | fused_fp8_bmm_rope_cat_and_cache_mla: FP8 BMM with per-token-group act | 1 | 2 | 14 | S7 (lead) |
| 3 | `glm52_fp8/kc_rope_cache_xl/tp4_large_xl.py` | 24 | exact | 14/1 | medium | aiter/ops/triton/fusions/fused_bmm_rope_kv_cache.py:fused_fp8_bmm_rope | 1 | 2 | 13 | S7 |
| 4 | `glm52_fp8/kc_rope_cache_xl/tp4_small_xl.py` | 3 | exact | 8/1 | medium | aiter/ops/triton/fusions/fused_bmm_rope_kv_cache.py:fused_fp8_bmm_rope | 1 | 2 | 13 | S7 |
| 5 | `glm52_fp8/kc_rope_cache_xl/tp8_large_xl.py` | 9 | exact | 18/1 | medium | aiter/ops/triton/fusions/fused_bmm_rope_kv_cache.py:fused_fp8_bmm_rope | 1 | 2 | 12 | S7 |

### G22: DSA index scores + top-k + page map (paged, prefill)

1 templates, 6 rows, models: GLM-5.2. Base gap 2, base evidence 2, reach 1.

**AITER 581579ab5 (nightly):** NEW GLM-5 paged/plain mqa_logits tuned tables (configs/model_configs/paged_mqa_logits_tuned_glm5.csv, glm5_tuned_mqa_logits.csv), FlyDSL fp8_paged_mqa_logits_gfx950, topk_per_row_radix_stream.  
**Changed vs e7d2453f2:** Baseline stronger; C3 still blocked on the per-call stream.

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `glm52/paged_index_scores_topk_xl/prefill_tp4_large_x.py` | 6 | exact | 4/1 | medium | aiter fp8_mqa_logits / deepgemm_fp8_paged_mqa_logits | 2 | 2 | 16 | C3 |

### G05: Attention prologue: q/k RMSNorm + (m)RoPE (+q/gate split) + FP8 KV-cache write

3 templates, 12 rows, models: Qwen3-Coder, Qwen3.5-397B. Base gap 2, base evidence 1, reach 2.

**AITER 581579ab5 (nightly):** Qwen3.5: fused_qk_norm_mrope_3d_cache_pts_quant_shuffle (unchanged, ops/fused_qk_norm_mrope_cache_quant.py:10) + NEW fused_qk_norm_rope_gate_fp8_quant (aiter_from_sglang_nightly_20261117/aiter/ops/triton/rope/fused_qk_norm_rope_gate_fp8_quant.py:65): interleaved q|gate split, Gemma (+1 in kernel), partial rotary, dynamic per-sequence FP8 Q/K/V for FMHA.  
**Changed vs e7d2453f2:** Gap narrower but still partial: the new op has the gate split + Gemma norm but writes no KV cache, takes 1-D positions (no mRoPE) and quantizes per sequence for FMHA (prefill-shaped); the HIP mRoPE op still has no gate split. Add arm B3 (new op + separate cache write).

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `qwen3_5/fp8/fused_attention_qk_norm_rope_kv_cache_templated/faqk_template.py` | 4 | exact | 3/1 | medium | fused_qk_norm_mrope_3d_cache_pts_quant_shuffle | 2 | 1 | 15 | S5 |
| 2 | `qwen3_5/mxfp4/fused_attention_qk_norm_rope_kv_cache_xl/xl_decode_small_template.py` | 4 | exact | 3/1 | medium | aiter fused_qk_norm_mrope_3d_cache_pts_quant_shuffle | 2 | 1 | 15 | S5 (lead) |
| 3 | `qwen3_coder/fused_attention_qk_norm_rope_fp8_kv_cache_xl/xl_split_half_template.py` | 4 | exact | 1/2 | medium | Triton fused_qkv_split_qk_norm_rope_cache | 0 | 0 | 9 | DNU |

### G15: BF16 skinny decode GEMM, general N (M<=64)

9 templates, 43 rows, models: DeepSeek-V4-Pro, GLM-5.2, Kimi-K3. Base gap 0, base evidence 1, reach 3.

**AITER 581579ab5 (nightly):** no material change vs e7d2453f2 for this group (same op/signature; closest-AITER column still valid, line numbers may have moved).

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `glm52_ep8/o_gemm_xl/decode_folded_x.py` | 3 | exact | 13/1 | high | aiter tuned_gemm.gemm_a16w16 / tgemm.mm | 1 | 2 | 15 | C4 |
| 2 | `glm52/o_gemm_xl/tp4_tiny_decode_x.py` | 8 | exact | 5/1 | medium | aiter/tuned_gemm.py:TunedGemm.mm:680 -> gemm_a16w16:328 | 0 | 2 | 14 | DNU |
| 3 | `glm52/q_b_gemm_xl/tp8_small_decode_x.py` | 6 | exact | 4/2 | medium | aiter/tuned_gemm.py:TunedGemm.mm:680 -> gemm_a16w16:328 | 0 | 2 | 14 | DNU |
| 4 | `glm52/q_b_gemm_r3/glm52_q_b_gemm_n4096_decode_template.py` | 8 | exact | 12/1 | medium | aiter/tuned_gemm.py:TunedGemm.mm:680 -> gemm_a16w16:328 | 0 | 2 | 12 | DNU |
| 5 | `deepseek_v4_pro/bf16_gemm_r3/n512_decode_template.py` | 6 | exact | 18/2 | high | tuned_gemm mm/tgemm | 0 | 2 | 12 | DNU |
| 6 | `glm52_ep8/q_b_gemm_xl/decode_ring_x.py` | 2 | exact | 11/1 | medium | aiter tuned_gemm.gemm_a16w16 / tgemm.mm | 0 | 2 | 12 | DNU |
| 7 | `kimi_k3/bf16_gemm_xl/n1536_paired_template.py` | 2 | exact | 5/1 | medium | aiter.tuned_gemm.gemm_a16w16 / tgemm.mm | 0 | 1 | 12 | DNU |
| 8 | `kimi_k3/bf16_gemm_xl/n2304_folded_template.py` | 3 | exact | 15/1 | medium | aiter.tuned_gemm.gemm_a16w16 / tgemm.mm | 0 | 1 | 10 | DNU |
| 9 | `kimi_k3/bf16_gemm_r3/n2112_small_template.py` | 5 | exact | 4/3 | medium | aiter.tuned_gemm.gemm_a16w16 / tgemm.mm | 0 | 0 | 9 | DNU |

### G04: GDN decode: conv1d update + gated delta rule + gated RMSNorm (+FP8 group quant)

4 templates, 14 rows, models: Qwen3-Next-80B, Qwen3.5-397B. Base gap 1, base evidence 3, reach 2.

**AITER 581579ab5 (nightly):** fused_gdn_decode_qkvz (aiter_from_sglang_nightly_20261117/aiter/ops/triton/gated_delta_net/fused_gdn_decode_qkvz.py:311): one Gluon launch for split + conv1d + recurrence + gated RMSNorm + optional group-128 FP8; ratios 1-4, "interleaved" (Qwen3-Next) and "flat" (Qwen3.5) qkvz layouts. Generated from the Artemis qwen3_next gdn_decode_m16.py (sha256 16ec8eaf... matches wt_next manifest) and a gdn_decode_m128_m256.py variant (_gluon_kernels/gfx950/gated_delta_net/fused_gdn_decode_qkvz.py:19-24).  
**Changed vs e7d2453f2:** NEW: the fused op now exists upstream (a port of our own kernel). Gated on Triton>=3.8 (fused_gdn_decode_qkvz.py:89-114), so in the nightly image (Triton 3.7) it reports unsupported and callers take the 4-kernel fallback. Gap 2->1: what is left is a perf PR (our per-M schedules m1/m8/m64 and the Qwen3.5 r3/xl rows vs AITER's generated pair kernel / ratio-generic kernel).

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `qwen3_5/fp8/gdn_fused_decode_group_fp8_quant_templated/qwen3_5_fp8_gdn_decode_template.py` | 2 | exact | 3/1 | low | causal_conv1d_update_split_qkv + fused_rearrange_sigmoid_gated_delta_r | 1 | 2 | 14 | S3 |
| 2 | `qwen3_next/gdn_fused_decode_group_fp8_quant_templated/gdn_decode_template_v2.py` | 4 | exact | 52/4 | low | causal_conv1d_update_split_qkv + fused_rearrange_sigmoid_gated_delta_r | 1 | 3 | 11 | S3 (lead) |
| 3 | `qwen3_5/mxfp4/gdn_fused_decode_xl/xl_gdn_decode_m96.py` | 4 | exact | 10/3 | medium | aiter fused_rearrange_sigmoid_gated_delta_rule / fused_split_gdr_updat | 1 | 1 | 11 | S3 |
| 4 | `qwen3_5/fp8/gdn_fused_decode_group_fp8_quant_r3/r3_gdn_decode_small.py` | 4 | exact | 23/1 | low | causal_conv1d_update_split_qkv + fused_rearrange_sigmoid_gated_delta_r | 1 | 2 | 10 | S3 |

### G07: BF16 skinny-N GEMM, N=32 K=4096 (GDN in_proj_ba)

6 templates, 27 rows, models: Qwen3.5-397B. Base gap 1, base evidence 2, reach 1.

**AITER 581579ab5 (nightly):** tgemm.mm / gemm_a16w16 (aiter_from_sglang_nightly_20261117/aiter/tuned_gemm.py:375,832); NEW FlyDSL skinny gfx950 kernel (ops/flydsl/kernels/gemm_a16w16_skinny_gfx950.py), selected only through tuned rows.  
**Changed vs e7d2453f2:** Unchanged: N=32,K=4096 still has only an M=16 Triton row (5.6 us); default skinny predicate unchanged (tuned_gemm.py:88-109).

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `qwen3_5/mxfp4/bf16_gemm_xl/xl_n32_fold_template.py` | 5 | exact | 7/1 | high | aiter tuned_gemm.gemm_a16w16 | 1 | 2 | 14 | S8 (lead) |
| 2 | `qwen3_5/mxfp4/bf16_gemm_xl/xl_n32_gemv_template.py` | 3 | exact | 7/1 | high | aiter tuned_gemm.gemm_a16w16 -> skinny_gemm | 1 | 2 | 14 | S8 |
| 3 | `qwen3_5/fp8/bf16_gemm_xl/xl_gdn_gemv_template.py` | 4 | exact | 12/1 | high | skinny_gemm / gemm_a16w16 | 1 | 2 | 13 | S8 |
| 4 | `qwen3_5/fp8/bf16_gemm_xl/xl_splitk_template.py` | 2 | exact | 16/1 | high | gemm_a16w16 | 1 | 2 | 12 | S8 |
| 5 | `qwen3_5/fp8/bf16_gemm_xl/xl_n32_template.py` | 8 | exact | 22/3 | high | tgemm.mm -> gemm_a16w16 | 1 | 2 | 11 | S8 |
| 6 | `qwen3_5/mxfp4/bf16_gemm_templated/qwen3_5_mxfp4_bf16_gemm_n32_template.py` | 5 | exact | 38/2 | high | aiter tuned_gemm.gemm_a16w16 / triton gemm_a16w16 | 1 | 2 | 11 | S8 |

### G11: MLA vc absorb: FP8 batched GEMM (latent x W_UV), optional fused output FP8 group quant

3 templates, 20 rows, models: DeepSeek-R1, GLM-5.2. Base gap 1, base evidence 1, reach 2.

**AITER 581579ab5 (nightly):** batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant (aiter_from_sglang_nightly_20261117/aiter/ops/triton/gemm/batched/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py:30) gained backend= and y_scale_out (per-token FP8 output quant); NEW batched_gemm_a16w8 (gfx1250-only Gluon).  
**Changed vs e7d2453f2:** gfx950 config dir for the op is still empty. y_scale_out is per-token, not the group-128 output quant the R1 template fuses, so the R1 part remains a partial gap.

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `glm52/vc_reconstruct_r3/vc_prefill_tpl.py` | 9 | mixed | 9/5 | medium | batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quan | 1 | 2 | 14 | S9 (lead) |
| 2 | `deepseek_r1/vc_reconstruct_quant_templated/vc_reconstruct_quant_template.py` | 7 | exact | 16/2 | high | batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quan | 2 | 0 | 11 | S9 |
| 3 | `glm52/vc_reconstruct_r3/vc_onewave_tpl.py` | 4 | exact | 8/1 | medium | batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quan | 1 | 0 | 9 | DNU |

### G14: BF16 LM-head GEMM, wide N (vocab shard), small M

8 templates, 44 rows, models: DeepSeek-R1, GLM-5.2, Qwen3-Next-80B, Qwen3.5-397B. Base gap 0, base evidence 1, reach 3.

**AITER 581579ab5 (nightly):** no material change vs e7d2453f2 for this group (same op/signature; closest-AITER column still valid, line numbers may have moved).

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `qwen3_next/bf16_gemm_xl/xl_gemv_template.py` | 2 | exact | 6/1 | high | skinny_gemm / gemm_a16w16 | 0 | 2 | 14 | DNU |
| 2 | `qwen3_next/bf16_gemm_xl/xl_lmhead_stream_template.py` | 5 | exact | 13/2 | medium | tgemm.mm / gemm_a16w16 / Triton gemm_a16w16 | 0 | 2 | 12 | DNU |
| 3 | `glm52/mtp_lm_head_templated/glm52_lmh_template.py` | 2 | exact | 2/1 | medium | aiter/tuned_gemm.py:TunedGemm.mm:680 -> gemm_a16w16:328 | 0 | 1 | 12 | DNU |
| 4 | `deepseek_r1/lm_head_xl/dsr1_lmh_xl_template.py` | 8 | exact | 7/5 | high | tuned_gemm mm/tgemm | 0 | 1 | 11 | DNU |
| 5 | `qwen3_5/fp8/bf16_gemm_xl/xl_lmhead_packed_template.py` | 3 | exact | 18/1 | high | skinny_gemm / gemm_a16w16 | 0 | 1 | 10 | DNU |
| 6 | `glm52_fp8/lm_head_xl/glm52f_lmhx_template.py` | 14 | exact | 10/10 | medium | aiter tuned_gemm.gemm_a16w16 / tgemm.mm | 0 | 1 | 9 | DNU |
| 7 | `qwen3_5/fp8/bf16_gemm_xl/xl_lmhead_mfma_template.py` | 6 | exact | 23/1 | high | tgemm.mm / gemm_a16w16 | 0 | 1 | 9 | DNU |
| 8 | `glm52/mtp_lm_head_r3/lmh_mfma_tpl.py` | 4 | exact | 16/1 | medium | aiter/tuned_gemm.py:TunedGemm.mm:680 -> gemm_a16w16:328 | 0 | 1 | 9 | DNU |

### G06: Kimi attention-residual gate (softmax over residual bank + mix + RMSNorm, in-place bank write)

2 templates, 118 rows, models: Kimi-K3. Base gap 1, base evidence 1, reach 2.

**AITER 581579ab5 (nightly):** attn_res_gate (aiter_from_sglang_nightly_20261117/aiter/ops/triton/fusions/attn_res.py:680), signature unchanged.  
**Changed vs e7d2453f2:** Only per-shape num_warps/num_stages config tables added (attn_res.py:276-295). Gap/evidence unchanged.

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `kimi_k3/attention_residual_norm_xl/arn_wide_xl.py` | 87 | exact | 3/2 | medium | aiter.ops.triton.fusions.attn_res.attn_res_gate | 1 | 1 | 13 | S10 (lead) |
| 2 | `kimi_k3/attention_residual_norm_xl/arn_fused_xl.py` | 31 | exact | 6/5 | medium | aiter.ops.triton.fusions.attn_res.attn_res_gate | 1 | 1 | 11 | S10 |

### G08: GDN prefill: conv1d + chunked gated delta rule + gated RMSNorm (+FP8 quant)

3 templates, 18 rows, models: Qwen3.5-397B. Base gap 2, base evidence 1, reach 1.

**AITER 581579ab5 (nightly):** NEW fused_gdn_prefill_qkvz (aiter_from_sglang_nightly_20261117/aiter/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py:1625), ported from the Artemis MI355 v1 Qwen3-Next pack (4 M-tile schedules), Qwen3-Next topology only (qkvz 3072, ba 16), FP8 epilogue, Triton>=3.8; NEW FlyDSL gdn_fused_prefill_fwd (ops/flydsl/gdn_fused_prefill.py:147), chunk core only, bf16 state.  
**Changed vs e7d2453f2:** Qwen3.5 (qkvz 5120, ratio 4, fp32 state) is not covered by either op, so the gap stays 2. The upstream route is now clear: extend fused_gdn_prefill_qkvz to the Qwen3.5 topology.

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `qwen3_5/mxfp4/gdn_prefill_xl/xl_gdn_prefill_dynamic.py` | 7 | mixed | 14/2 | medium | aiter triton chunk_gated_delta_rule | 2 | 1 | 13 | S12 (lead) |
| 2 | `qwen3_5/fp8/gdn_prefill_group_fp8_quant_templated/gdn_prefill_template.py` | 6 | mixed | 4/2 | low | chunk_gated_delta_rule_opt + causal_conv1d_split_qkv + gated_rmsnorm_f | 2 | 0 | 12 | S12 |
| 3 | `qwen3_5/mxfp4/gdn_prefill_r3/r3_gdn_prefill_template.py` | 5 | exact | 14/1 | medium | aiter triton chunk_gated_delta_rule + causal conv1d + gated rmsnorm | 2 | 1 | 12 | S12 |

### G12: Paged GQA decode, FP8 KV (MiniMax 12q/2kv/128; Qwen3.5 + sigmoid output gate)

3 templates, 19 rows, models: MiniMax-M2.5, Qwen3.5-397B. Base gap 1, base evidence 2, reach 2.

**AITER 581579ab5 (nightly):** MiniMax: paged_attention_ragged (aiter_from_sglang_nightly_20261117/aiter/ops/attention.py:792), signature unchanged. Qwen3.5 output gate: NEW paged_attention_output_gate_group_fp8_quant (ops/triton/attention/paged_attention_output_gate.py:192), ported from the Artemis kernels.  
**Changed vs e7d2453f2:** Qwen3.5 r3_paog is now upstream (DNU). MiniMax rows unchanged.

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `minimax_m2_5/paged_gqa_decode_r3/paged_gqa_decode_r3_template.py` | 8 | exact | 10/2 | medium | aiter.ops.attention.paged_attention_ragged | 1 | 2 | 13 | C1 |
| 2 | `minimax_m2_5/paged_gqa_decode_r3/paged_gqa_decode_split_r3_template.py` | 8 | exact | 24/2 | low | aiter.ops.attention.paged_attention_ragged | 1 | 2 | 10 | C1 |
| 3 | `qwen3_5/mxfp4/paged_attention_output_gate_r3/r3_paog_template.py` | 3 | exact | 6/1 | medium | aiter paged attention decode | 0 | 1 | 10 | DNU |

### G20: MoE block: router/top-k + routed experts (+shared expert) in one pack

5 templates, 14 rows, models: GLM-5.2, Qwen3.5-397B. Base gap 0, base evidence 1, reach 2.

**AITER 581579ab5 (nightly):** fused_moe + NEW qwen3_5_397b_tuned_grouped_fmoe.csv.  
**Changed vs e7d2453f2:** AITER even more tuned at our shapes; DNU strengthened.

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `glm52_ep8/fused_moe_xl/decode_ep8_m8_m16_x.py` | 2 | exact | 2/7 | low | aiter fused_moe | 1 | 2 | 13 | DNU |
| 2 | `qwen3_5/fp8/qwen_router_complete_moe_templated/router_moe_mid.py` | 2 | exact | 0/1 | low | fused_moe / fused_moe_2stages | 0 | 1 | 10 | DNU |
| 3 | `qwen3_5/mxfp4/mxfp4_router_complete_moe_r3/r3_router_moe_template.py` | 3 | mixed | 12/2 | low | aiter fused_moe | 0 | 0 | 7 | DNU |
| 4 | `qwen3_5/fp8/qwen_router_complete_moe_templated/router_moe_small.py` | 4 | exact | 51/1 | low | fused_moe / fused_moe_2stages | 0 | 1 | 5 | DNU |
| 5 | `qwen3_5/mxfp4/mxfp4_router_complete_moe_xl/router_moe_native_xl.py` | 3 | exact | 76/2 | low | aiter fused_moe a4w4 per_1x32 + fused_topk | 0 | 0 | 3 | DNU |

### G25: MTP/draft helpers: softmax-top1, eh_proj input combine

2 templates, 7 rows, models: GLM-5.2. Base gap 2, base evidence 0, reach 1.

**AITER 581579ab5 (nightly):** no material change vs e7d2453f2 for this group (same op/signature; closest-AITER column still valid, line numbers may have moved).

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `glm52/mtp_softmax_top1_r3/glm52_mst_template.py` | 4 | exact | 2/1 | high | No fused equivalent. Pieces exist: triton softmax, triton topk, greedy | 2 | 0 | 13 | DNU |
| 2 | `glm52/mtp_input_combine_r3/glm52_mic_m8_m16_template.py` | 3 | exact | 1/3 | medium | No fused equivalent. Pieces: rmsnorm | 2 | 0 | 12 | DNU |

### G19: Dense causal varlen prefill MHA, head dim 256 (GLM-5.2 short-context)

2 templates, 11 rows, models: GLM-5.2. Base gap 0, base evidence 1, reach 1.

**AITER 581579ab5 (nightly):** no material change vs e7d2453f2 for this group (same op/signature; closest-AITER column still valid, line numbers may have moved).

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `glm52_ep8/dense_prefill_attention_xl/prefill_ep8_m128_m1024_x.py` | 2 | exact | 1/2 | medium | aiter flash_attn_varlen_func | 0 | 2 | 12 | DNU |
| 2 | `glm52/dense_prefill_attention_xl/dense_prefill_x.py` | 9 | mixed | 4/2 | medium | aiter.flash_attn_varlen_func | 0 | 0 | 9 | DNU |

### G13: Paged GQA prefill/extend, FP8 KV (MiniMax 12q/2kv/128)

1 templates, 12 rows, models: MiniMax-M2.5. Base gap 1, base evidence 2, reach 1.

**AITER 581579ab5 (nightly):** mha_batch_prefill_func (aiter_from_sglang_nightly_20261117/aiter/ops/mha.py:4514).  
**Changed vs e7d2453f2:** Unchanged.

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `minimax_m2_5/paged_gqa_prefill_r3/paged_gqa_prefill_r3_template.py` | 12 | ranges | 22/3 | medium | aiter.ops.mha.mha_batch_prefill_func | 1 | 2 | 11 | C1 |

### G18: RMSNorm family: (add+) RMSNorm / Gemma RMSNorm (+FP8 group quant)

6 templates, 35 rows, models: Qwen3-Coder, Qwen3.5-397B. Base gap 0, base evidence 1, reach 2.

**AITER 581579ab5 (nightly):** no material change vs e7d2453f2 for this group (same op/signature; closest-AITER column still valid, line numbers may have moved).

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `qwen3_5/fp8/fused_add_rmsnorm_bf16_group_fp8_quant_templated/fadd_rmsnorm_template.py` | 11 | exact | 13/2 | medium | aiter add_rmsnorm_quant(group_size=128, gemma_norm=True) | 0 | 2 | 11 | DNU |
| 2 | `qwen3_5/fp8/fused_add_gemma_rmsnorm_group_fp8_quant_templated/fagr_template.py` | 9 | exact | 4/5 | medium | aiter add_rmsnorm_quant(group_size=128, gemma_norm=True) | 0 | 1 | 10 | DNU |
| 3 | `qwen3_5/mxfp4/bf16_residual_gemma_rmsnorm_xl/xl_add_gemma_rmsnorm_decode.py` | 2 | exact | 4/1 | high | aiter rmsnorm2d_fwd_with_add | 0 | 0 | 10 | DNU |
| 4 | `qwen3_5/mxfp4/bf16_gemma_rmsnorm_r3/r3_gemma_rmsnorm_template.py` | 6 | mixed | 22/3 | high | aiter rms_norm / rmsnorm2d_fwd | 0 | 1 | 9 | DNU |
| 5 | `qwen3_5/mxfp4/bf16_residual_gemma_rmsnorm_r3/r3_residual_gemma_rmsnorm_template.py` | 2 | exact | 6/1 | high | aiter rmsnorm2d_fwd_with_add | 0 | 0 | 9 | DNU |
| 6 | `qwen3_coder/rmsnorm_templated/qc_rms_template.py` | 5 | exact | 21/1 | medium | rms_norm / rmsnorm2d_fwd | 0 | 0 | 6 | DNU |

### G16: BF16 GEMM, medium/large M (128..16384)

5 templates, 24 rows, models: GLM-5.2, Kimi-K3, MiniMax-M2.5. Base gap 0, base evidence 0, reach 3.

**AITER 581579ab5 (nightly):** no material change vs e7d2453f2 for this group (same op/signature; closest-AITER column still valid, line numbers may have moved).

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `glm52/q_b_gemm_templated/glm52_q_b_gemm_n2048_prefill_template.py` | 2 | ranges | 1/5 | medium | aiter/tuned_gemm.py:TunedGemm.mm:680 -> gemm_a16w16:328 | 0 | 0 | 10 | DNU |
| 2 | `minimax_m2_5/bf16_gemm_xl/bf16_gemm_xl_template.py` | 4 | exact | 12/1 | high | aiter.tuned_gemm.gemm_a16w16 / tgemm.mm | 0 | 0 | 9 | DNU |
| 3 | `kimi_k3/bf16_gemm_r3/n7168_m128_256_template.py` | 2 | exact | 7/1 | medium | aiter.tuned_gemm.gemm_a16w16 / tgemm.mm | 0 | 0 | 9 | DNU |
| 4 | `glm52/o_gemm_xl/tp4_prefill_panel_x.py` | 10 | exact | 12/4 | medium | aiter/tuned_gemm.py:TunedGemm.mm:680 -> gemm_a16w16:328 | 0 | 0 | 8 | DNU |
| 5 | `glm52/q_b_gemm_r3/glm52_q_b_gemm_n2048_m4193_template.py` | 6 | exact | 14/3 | medium | aiter/tuned_gemm.py:TunedGemm.mm:680 -> gemm_a16w16:328 | 0 | 0 | 8 | DNU |

### G17: BF16 skinny GEMM + two residual adds epilogue (Kimi MoE tail)

1 templates, 10 rows, models: Kimi-K3. Base gap 1, base evidence 1, reach 1.

**AITER 581579ab5 (nightly):** NEW fused_gemm_a16w16_mul_add (aiter_from_sglang_nightly_20261117/aiter/ops/triton/gemm/fused/fused_gemm_a16w16_mul_add.py:35): out = a*(x@W^T+bias)+b.  
**Changed vs e7d2453f2:** Covers has_prefix=0 directly (b=shared) and has_prefix=1 with one pre-add; gap 2->1.

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `kimi_k3/bf16_gemm_add_xl/gemm_add_m1_16.py` | 10 | exact | 14/3 | medium | aiter.tuned_gemm.gemm_a16w16 / tgemm.mm | 1 | 1 | 10 | C2 |

### G23: KDA whole-layer decode (projections + conv + recurrence + gated norm)

1 templates, 3 rows, models: Kimi-K3. Base gap 2, base evidence 0, reach 1.

**AITER 581579ab5 (nightly):** no material change vs e7d2453f2 for this group (same op/signature; closest-AITER column still valid, line numbers may have moved).

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `kimi_k3/kda_layer_decode_r3/kda_m32_m64_template.py` | 3 | mixed | 7/2 | low | aiter.ops.triton.gated_delta_net.fused_kda_decode_unified | 2 | 0 | 10 | DNU |

### G21: Dense paged MLA decode, FP8 KV (DeepSeek-R1 16 heads, Kimi-K3 12 heads)

3 templates, 14 rows, models: DeepSeek-R1, Kimi-K3. Base gap 1, base evidence 0, reach 2.

**AITER 581579ab5 (nightly):** no material change vs e7d2453f2 for this group (same op/signature; closest-AITER column still valid, line numbers may have moved).

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `kimi_k3/paged_attention_decode_xl/pa_m24_64.py` | 3 | exact | 9/2 | medium | aiter.mla.mla_decode_fwd | 1 | 0 | 9 | DNU |
| 2 | `kimi_k3/paged_attention_decode_r3/pa_m4_m8_template.py` | 2 | exact | 8/2 | medium | aiter.mla.mla_decode_fwd | 1 | 0 | 9 | DNU |
| 3 | `deepseek_r1/mla_decode_xl/dsr1xl_mla_template.py` | 9 | exact | 17/5 | medium | aiter.mla.mla_decode_fwd | 1 | 0 | 7 | DNU |

### G24: DeepSeek-V4 mHC pre (hyper-connection mix)

1 templates, 4 rows, models: DeepSeek-V4-Pro. Base gap 0, base evidence 1, reach 1.

**AITER 581579ab5 (nightly):** no material change vs e7d2453f2 for this group (same op/signature; closest-AITER column still valid, line numbers may have moved).

| # | template (under src/kernel_packs/gfx950/) | rows | dispatch | knobs/bodies | port | closest AITER (e7d2453f2) | gap | ev | score | rec |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| 1 | `deepseek_v4_pro/mhc_pre_templated/mhc_pre_decode_m8_m32.py` | 4 | exact | 16/3 | medium | mhc_pre_dsv4 | 0 | 1 | 7 | DNU |
