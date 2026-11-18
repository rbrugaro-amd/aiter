## Phase 2 GPU bench (non-GEMM candidates): templates vs AITER as SGLang 703e763d5f calls it

All requested non-GEMM candidates were measured except C3, which is blocked. The harness, raw data, merged tables and notes are in `/home/rbrugaro/code/templating-infra/aiter/bench/`.

Per-shape tables: `results/<cand>.md` and `results/<cand>.csv` (merged by `xbench/merge.py`); raw records in `results/<cand>/*.jsonl`; per-candidate notes in `results/*_notes.txt`. Node job 295273, 2026-11-18.

### Setup, as you changed it
- **Image:** one image for all timed arms, `nightly1117-triton38`, built from `bench/docker/Dockerfile`. It is the SGLang nightly plus `triton==3.8.0+gitdf3f91dd`.
  - Versions inside: torch 2.11.0+rocm7.2, AITER 581579ab51ca, SGLang 703e763d5f, ROCm 7.2.4, hipBLASLt 1.2.2.
  - Before timing, templates were checked in this image: S1 (m64, m1024) and S10 (64/4/4) were T1 IDENTICAL and T3 bit-exact against their owner files.
- **Code version:** wt_next is at c41c6780, not the 88081cdc the plan names.
- **Arms:**
  - T is the template, loaded with its manifest sha256 checked.
  - B is AITER exactly as SGLang calls it. Where possible I called SGLang's own functions.
  - B1, B2… are better AITER compositions.
  - X arms are non-AITER stock paths, timed once as context.
  - A/A is a second, separately captured graph per block.
  - B37 is the unmodified nightly (Triton 3.7), run only for S3, S12/S12f and the C1 output-gate member.
- **Trace:** each candidate's SGLang dispatch was traced once with the profiler and the AITER tuned-config log.
- **Timing:** graph replay, ABBA order, 15 rounds, 3 blocks; S12/S12f were timed eagerly because they are not graph-safe. Inputs were generated once with sha256 checks. Rotating cache is the headline.

### Results (rotating cache; speedup = best AITER arm / T)

| cand | scope | T vs best AITER | numerics | verdict |
|---|---|---|---|---|
| **S1** GLM FP8 lead (`qkv_a_norm_r3`) | M 1..256 + prefill | **1.94x** decode (weighted uniformly over M 1..256, matching the GLM C1..C256 suites); 1.56–1.82x per M, 4.9x at M=2. Prefill: 1.64x at 1024, 1.46x at 2048, 0.96x at 4096, 1.13x at 8193, 1.07x at 16384 | **T fails the seed-100 cancellation draw at 11 of 14 shapes; AITER passes** | **Needs fix before a PR.** The split-K partials are stored in bf16/fp16 (`qkv_a_norm_templated/decode_shared_m256.py:188,197`, `decode_shared_m32.py:119`). This is inherited from the hand-tuned owners. |
| S1 DeepSeek-R1 (`qkv_a_norm_quant`) | M 1..256, 1k–16k | **1.65x** decode, 2.06–2.44x prefill | T within ~1.2x of B on every statistic | **PR** |
| S1 GLM MXFP4 bf16-W, plus EP8 member | TP4/TP8/EP8 | **1.13x** (EP8: 1.13x) | T passes; SGLang's AITER bf16 GEMM fails seed 100 at M≥16 | **PR, marginal.** Losses at TP8 m1 (0.96x) and m16384 (0.90x) |
| **S4** sparse MLA decode | 37 shapes | MXFP4 H16/H8 **1.85x**; EP8 H64 1.36–2.6x; FP8 TP8 1.00x overall | The fastest AITER arm (sparse_mla_fwd fp8 dots) fails 35/37 shapes; against the fastest passing AITER arm T is 2.1–2.5x. T fails 1 shape | **PR:** `decode_tp4_small`, `decode_tp8_deep`, all EP8 H=64. **Fix first:** `decode_tp8_narrow` (wrong output on pools >2^31 bytes), `decode_tp8_m8_x` (0.47x on such pools). **No PR:** `spm_tp8_split`/`spm_tp8_policy` (15–30x slower when slot ids spread over >64–128k rows) |
| **S6** DSA indexer prepare | decode + key-only | **2.12x** (hot 3.05x), worst block ≥1.25x | T passes; AITER fails (FlyDSL split-K bf16 GEMM accuracy) | **PR** as a new fused AITER op. SGLang's default decode at M≤48 is its own fused HIP kernel, which T beats by only 1.20–1.27x |
| **S5** Qwen3.5 attention prologue | M 1–64 | **1.26x** vs B2 | T passes everything; B2 fails the fp8 K-cache code check | **PR**. m1 is the weakest cell at 1.15x |
| **S12** GDN prefill, Qwen3.5 MXFP4 | 1k–16k tokens | **4.66x** eager (3.1–4.4x on GPU time alone); 4.95x at 5120, 3.96x at 16384 | pass, 45/45 cells | **PR** |
| S12f GDN prefill, Qwen3.5 FP8 | 5k–16k | 1.74x eager, but only 1.03–1.53x on GPU time alone | pass | **Weak.** Fails the 1.3x rule at 5000 tokens on GPU time. Rebuild it on the S12 xl template |
| **S3** GDN decode | Qwen3-Next + Qwen3.5 | Qwen3-Next **1.00–1.06x** vs AITER's port of our kernel; Qwen3.5 1.57–2.88x (FP8 M256 1.07x) | Qwen3.5: T fails seed 0 where the port passes; T's error equals SGLang stock, and the port is 2–3x closer to fp32 | **No PR** for Qwen3-Next. Qwen3.5 small-M schedules: **PR after a numerics fix** (or an explicit stock-equivalent acceptance) |
| **S10** Kimi attention residual | 33 keys | **1.46x** (1.07–1.86x) | T passes; AITER `attn_res_gate` fails seed 100 at nvb=1 mode 5 | **PR**. Without CUDA graphs, AITER's host time per call is 1.2–1.8x lower than T's |
| **C1** MiniMax paged GQA decode | rows ≤64 | **1.28x**; rows 64 at parity | T passes and is ~10x closer to fp32 | **PR after one change.** The schedule is keyed on `kv_indices.numel()` (graph capacity); with exact-size indices T raises an error |
| C1 MiniMax prefill | 1024–8192 | 0.86–1.46x | **T fails seed 0 while AITER passes** | **No PR** |
| C1 Qwen3.5 output gate (parity member) | rows 1/4/8 | AITER's port does not run in either image; vs SGLang's path T is 1.5–2.3x | pass | Parity check impossible |
| **C3** GLM index top-k | – | – | – | **Blocked.** The per-call `torch.cuda.Stream()` is still at `glm52/paged_index_scores_topk_xl/prefill_tp4_large_x.py:894-896` |

GEMM candidates were not run, as instructed. Where no decode M distribution is recorded, served rows are weighted uniformly.

### What stock SGLang dispatches (traced)
- **S1 GLM FP8:** per-1x128 quant, then AITER's bpreshuffle GEMM chosen by tuned CSV. M≤64 gets the Triton split-K kernel, M 256–1024 CK, M=16384 FlyDSL. Then `fused_rms_fp8_group_quant`.
- **S1 R1:** plain-weight Triton GEMM, then `fused_rms_fp8_group_quant`.
- **S1 GLM bf16:** the linear goes to AITER skinny/hgemm kernels at small M and hipBLASLt (untuned) at M≥1024, then `fused_rms_mxfp4_quant`.
- **S4:** SGLang's default is its own Triton kernel. `--dsa-decode-backend aiter` runs the `mla_decode_fwd` asm kernel with head padding plus about 10 prep kernels per call, 4–7x slower than T. SGLang never calls `sparse_mla_fwd`.
- **S5:** no AITER at all (SGLang Triton kernels plus `reshape_and_cache_flash`).
- **S10:** no AITER (SGLang's `_agg_kernel`).
- **S6:** SGLang's own fused HIP indexer at decode M≤48; otherwise AITER GEMMs plus `indexer_qk_rope_quant_and_cache`.
- **S3/S12:** SGLang Triton and FLA kernels only.
- **C1 decode:** `paged_attention_ragged`. **C1 prefill:** CK `mha_batch_prefill_func`.

### Does SGLang 703e763d5f call the AITER ops ported from Artemis?
**No, under no condition.**
- `fused_gdn_decode_qkvz`, `fused_gdn_prefill_qkvz` and `paged_attention_output_gate*` do not appear anywhere in `sglang/srt` or `sglang/kernels`, and nothing inside AITER calls them either.
- GDN on ROCm runs SGLang's Triton and FLA kernels. The Qwen3.5 output gate is backend attention plus `fused_sigmoid_mul` (`qwen3_5.py:1632`). AITER GDN kernels are used only for speculative target-verify with `--linear-attn-verify-backend aiter`.
- In the shipped Triton 3.7 image all three ports report unsupported.
- With our Triton 3.8 pin:
  - `fused_gdn_decode_qkvz` runs.
  - `fused_gdn_prefill_qkvz` rejects every Qwen3.5 shape (it covers Qwen3-Next only).
  - `paged_attention_output_gate` still reports unsupported: "Gluon build predates `shared_memory_descriptor.reinterpret`", so our pin is too old for it.

### Cross-cutting findings
1. **SGLang monkeypatches Triton.** It replaces `triton.next_power_of_2` at import (`sglang/srt/utils/triton_utils.py:49`). Under Triton 3.8 that breaks Gluon kernels that use it in-kernel, including our S1 templates. The harness restores the original after every SGLang import.
2. **AITER bf16 GEMMs lose accuracy on cancellation inputs.** The split-K kernels SGLang dispatches on gfx950 fail where our templates pass (S1 bf16 member, S6).
3. **AITER fails on adversarial draws** in: sparse_mla fp8 dots, asm MLA at H=64, `attn_res_gate` mode 5, and S5's B2 fp8 K codes.
4. **Cross-check against AITER's own harness (S1 GEMM leg).** AITER's `perftest` reads 5–32% slower than our graph-replay number for the same kernel; the gap is largest at M≤64 and comes from launch overhead in eager timing. Our M=1 result (6.46 us) matches the tuned CSV (6.5 us), so our B numbers are, if anything, favourable to AITER. The AITER op_test script itself crashes on gfx950 in its asm sub-test.
5. **Triton 3.7 probe.** The plan expected no template to run on Triton 3.7, but the S10 and S5 templates compile there and are bitwise equal to Triton 3.8. S12 and S3 Qwen3-Next m8 fail on `llvm_fn_attrs`.

### Deviations and incidents
- **Parallel GPUs:** candidates ran in parallel on GPUs 0–6. The reference copy kernel read 4.8–5.05k GB/s in some blocks against about 5.5k when the node was quiet. T and B always ran in the same block, so per-block speedups stayed tight (min–max mostly under 2%).
- **Node outage:** the node daemon was down 06:27–06:43 UTC. Incomplete blocks were discarded and re-run. C1 was re-run entirely on GPU 6 when GPU 7 was reassigned.
- **fp8 code criterion relaxed:** the plan's code-point rule fails every implementation, AITER and stock included, against an fp32 reference. S3, S6 and S12/S12f use a dequantized-value check instead (S12/S12f judge against the stock path's error envelope), and S1-R1 is judged relative to B. Each candidate's notes file states its rule.
- **Stock arm:** the stock SGLang sequence was not timed, only traced, as you asked.

Files are in `/home/rbrugaro/code/templating-infra/aiter/bench/`:
- `docker/Dockerfile`
- `rn`, `sync`, `pull` (node helpers)
- `xbench/` (`core.py`, `run.py`, `merge.py`, `tload.py`, `envinfo.py`, `cands/*.py`, `xcheck_s1.py`, `a8_probe*.py`)
- `results/<cand>/*.jsonl` (raw records, including environment and rocm-smi captures)
- `results/<cand>.csv` and `results/<cand>.md` (merged per-shape tables) for S1, S1r1, S1g, S1e, S4, S5, S6, S10, S12, S12f, S3, C1d, C1p, C1g
- `results/*_notes.txt`
- `sglang_703e763d5f/` (source snapshot used for the traces)

Inputs remain on node scratch under `/mnt/m2m_nobackup/rbrugaro/templating/xb/inputs`. The S4 tilelang context arm and the MXFP4 notes are in `results/S4_notes.txt`.