# Handoff: GLM-5.2 (artemis PR #33) AITER PRs: branches, images, what to test

State as of 2026-11-18 17:30 UTC. Everything needed off-cluster is on GitHub:

| what | where |
|---|---|
| AITER code branches (one per PR) | `rbrugaro-amd/aiter` (fork), branches below; base `ROCm/aiter` main `ca7cfa1c3` |
| PR drafts, this handoff, bench harness + Dockerfile + results | `rbrugaro-amd/aiter`, branch `pr-drafts` (`pr_drafts/`, `benchmarks/`) |
| Our kernels (source of the ports, read-only) | `AMD-MLPerf/artemis-kernel-integrations`, branch `rbrugaro/templating-gfx950-next` @ `c41c6780` (PR #26 head + PR #33 merged, all 6 PR #33 templating units active) |

**Code state right now:** no PR code is pushed yet. Three agents are implementing on the cluster;
the node allocation ended before any GPU gate ran. A branch appears on the fork only after its gates
pass. If a branch is missing when you pick this up, start from `upstream/main` (`ca7cfa1c3`) and use
the plan below. Uncommitted cluster work-in-progress lives in
`/home/rbrugaro/code/aiter_wt/<branch>/` (shared /home), and the agents' PR drafts in
`/home/rbrugaro/code/templating-infra/aiter/pr_drafts/`.

## Image (same for all three PRs)

`nightly1117-triton38`: `benchmarks/docker/Dockerfile`

```Dockerfile
FROM lmsysorg/sglang-rocm:v0.5.23-rocm724-mi35x-20261117
RUN /opt/venv/bin/python -m pip install --no-deps --force-reinstall --no-cache-dir \
      --index-url https://download.pytorch.org/whl/nightly/rocm7.2 "triton==3.8.0+gitdf3f91dd"
```

- Contents: SGLang `703e763d5f`, bundled AITER `581579ab5` (2026-11-07), torch `2.11.0+rocm7.2`,
  Triton upgraded to 3.8 so templates (Gluon) and AITER run in one container.
- GPU: MI355X (gfx950). Single GPU per PR is enough. GLM weights are not needed (microbenchmarks only).
- Run the PR branch from source over the bundled AITER, and print `aiter.__file__` in every run:
  `pip install --no-deps --no-build-isolation -e <branch checkout>`. Use a separate checkout of
  `upstream/main` as the baseline; the gate is "branch vs current upstream main", not vs the nightly.
- Do not `import sglang` in benches: it monkeypatches `triton.next_power_of_2` and breaks Gluon kernels
  (the harness restores it).

## Harness (`benchmarks/xbench/`)

```sh
# in the container; XB_ROOT = scratch dir for inputs/results; /work = checkout of the artemis branch above
export XB_ROOT=/scratch/xb XB_IMG=n38 PYTHONPATH=<dir containing xbench>
python -m xbench.run --cand S4 --gen                     # once: inputs + fp32 refs
python -m xbench.run --cand S4 --block 1 --numerics      # timing block + numerics
python -m xbench.run --cand S4 --block 2 ; --block 3     # more blocks (A/A, ABBA)
python3 -m xbench.merge S4 --root $XB_ROOT/results    # host-side -> S4.md / .csv
```

- `xbench/tload.py` loads our templates from `/work` and checks their sha256 against the served
  manifest. So mount the artemis checkout at `/work`, or edit `WORK` there.
- Timing uses graph replay, rotating cache (headline) and hot cache. Numerics are vs an fp32
  reference, on seeds 0 (benign) and 100 (peaked/cancellation).
- For the PR gate, add an arm that calls the AITER op from the branch checkout and one from the main
  checkout. Also run eager wall time (no host regression; keep AITER's launch cache).
- Gate per enabled shape: ≥1.10x rotating vs upstream main, numerics pass. The op's existing tests
  plus new tests in the same style must pass in `op_tests/`.

## Rules for the AITER code (from Rita)

- No monkey patches.
- Avoid new public API: extend an existing op behind the same signature. If a new op is
  unavoidable, copy the closest upstream precedent exactly (file placement, naming, config table,
  tests in `op_tests/`, bench in `op_tests/op_benchmarks/`).
- Gluon paths need Triton ≥3.8. Gate them like `fused_gdn_decode_qkvz` /
  `paged_attention_output_gate_group_fp8_quant` are gated, with the same fallback.
- Enable only the shapes that measure faster.
- Push to the fork only; Rita opens the upstream PRs.

---

## PR 1: Sparse MLA decode, EP8 H=64, fp8 KV. Branch `sparse-mla-decode-gfx950`

- **Source kernels:** `src/kernel_packs/gfx950/glm52_fp8_ep8/sparse_paged_mla_xl/{decode_ep8_m1_m8_x,decode_ep8_m12_m24_x,decode_ep8_m16_m32_x}.py` + configs. Served by the `glm52_fp8_ep8` profile, manifest `sparse_paged_mla.json`.
- **Contract:**
  - GLM-5.2 DSA decode, 64 heads per rank (EP8).
  - KV is `fp8_scalar` `[P,576]` with unit scale; the query is fp8 with unit scale.
  - top-k 2048, slot table `[m, 2048]` with `-1` sentinels.
  - Pools both below and above 2^31 bytes.
- **AITER target:** extend the existing `sparse_mla_fwd`. It already has a Gluon fp8_scalar kernel, but that kernel only runs with `dot_precision="fp8"` and no `-1` sentinels; with the served slot table, it falls back to the generic Triton `_sparse_mla`. Route the H=64 unit-scale fp8 shapes to the new kernel.
- **Measured so far** (nightly AITER, `benchmarks/results/S4.md`, rows `fp8ep8_tp8_h64_*`):

  | M | template (µs, rotating) | sparse_mla_fwd fp8 dots (B2) | speedup vs B2 | vs bf16 dots (B1) |
  |---|---|---|---|---|
  | 1 | 5.62 | 14.62 | 2.60x | 3.32x |
  | 2 | 6.58 | 14.24 | 2.16x | 2.83x |
  | 4 | 7.13 | 14.16 | 1.99x | 2.62x |
  | 8 | 9.52 | 14.28 | 1.50x | 1.94x |
  | 12 | 12.13 | 20.22 | 1.67x | 1.86x |
  | 16 | 14.84 | 21.81 | 1.47x | 1.63x |
  | to M=32 | | | down to 1.36x | ~2.1x geomean |

  Template numerics pass on both seeds. Problems in the AITER paths:
  - B2 (fp8 dots) and B3 (Gluon fp8) fail numerics.
  - SGLang's AITER asm path (`mla_decode_fwd` `a16w8_qh16`) fails seed 100 at H=64.
  - B1 (bf16 dots) passes.
- **To test:**
  1. Run `op_tests` for `sparse_mla_fwd`: existing tests plus new tests at H=64, M ∈ {1,2,4,8,12,16,24,32}. Cover fp8 unit-scale query, `-1` sentinels, pool < and > 2^31 B, and seeds 0 and 100.
  2. Benchmark branch vs main per M (rotating + hot), and eager wall time.
  3. Confirm that H≠64 and non-unit-scale calls still take the old path and match main bit-for-bit.
  4. Check behaviour on Triton < 3.8: falls back and tests are skipped.
- **Out of scope:** the fp8 TP8 H=8 templates (`decode_tp8_narrow` is wrong on pools above 2 GB; `policy` M12/24 loses; split/policy M≥64 has a 0.05x slot-spread pathology) and the MXFP4 rows.

## PR 2: GLM-5.2 FP8 DSA indexer prepare. Branch `glm52-fp8-indexer-prepare`

- **Source kernels:**
  - `src/kernel_packs/gfx950/glm52_fp8/indexer_prepare_xl/ipx_decode_template.py` + `ipx_decode_config.py`
  - `glm52_fp8/indexer_prepare_key_only_xl/decode_shared_m32_x.py`
  - EP8 top-k row `glm52_fp8/paged_index_scores_topk/prefill_shared_m1024_8191_m8192.py`

  Served rows are in the `glm52_fp8_ep8` and `glm52_fp8` (TP4/TP8) profiles; the adapter is in `src/artemis_sglang/adapters/glm/attention.py`. Templating evidence: `notes/templating/gfx950/pr33/idx/`.
- **What the kernel fuses:**
  1. wq_b / wk / weights projections
  2. LayerNorm
  3. RoPE
  4. Hadamard
  5. FP8 quant
  6. index-cache write

  AITER only covers the epilogue: `indexer_qk_rope_quant_and_cache` and `indexer_k_quant_and_cache` in `aiter/ops/cache.py:123`. A PR likely needs a **new op** following that file's precedent.
- **Measured so far:** the FP8 variant has **not been benchmarked**. The MXFP4 sibling (S6, `benchmarks/results/S6*`, `xbench/cands/S6.py`) measured 2.12x and passes numerics; AITER's bf16 GEMMs fail the cancellation draw there.
- **To test:**
  1. Trace what SGLang nightly calls on the FP8 path (trace only, do not time it).
  2. Create `cands/S6fp8.py` from `S6.py` with the FP8 served shapes (TP4/TP8/EP8 decode M rows, key-only prefill, EP8 top-k rows).
  3. Time the template vs the best AITER composition, with numerics on seeds 0 and 100.
  4. Only if ≥1.10x rotating and numerics pass, implement it, then run the same op_tests + branch-vs-main gates as PR 1.

## PR 3: GLM-5.2 FP8 qkv_a + RMSNorm + FP8 quant. Branch `mla-qkv-a-norm-quant-fused`

- **Source kernel:** `src/kernel_packs/gfx950/glm52_fp8/qkv_a_norm_r3/qkv_a_norm_template.py`. It serves the EP8 qkv_a_norm rows (repointed by PR #33) and the glm52_fp8 TP4/TP8 rows.
- **What SGLang/AITER does today:**
  1. per-1x128 quant
  2. AITER bpreshuffle GEMM chosen by the tuned CSV (Triton split-K at M≤64, CK at 256–1024, FlyDSL at 16384)
  3. `fused_rms_fp8_group_quant`
- **Measured so far** (`benchmarks/results/S1.md`):
  - **1.94x** decode over M 1..256 (1.56–1.82x per M, 4.9x at M=2).
  - Prefill: 1.64x at 1024, 1.46x at 2048, 0.96x at 4096, 1.13x at 8193, 1.07x at 16384.
  - **Blocker:** the template fails the seed-100 cancellation draw at 11/14 shapes; AITER passes. The cause is split-K partials stored in bf16/fp16 (`glm52_fp8/qkv_a_norm_templated/decode_shared_m256.py:188,197`, `decode_shared_m32.py:119`), inherited from the hand-tuned owners. The served artemis kernels have the same issue; that's a separate fix in artemis.
- **Fix in progress:** `xbench/cands/S1f32.py` plus `xbench/s1f32/` hold copies of the kernels with every split-K partial (and the M=8 wave exchange) in fp32, arm `Tf`.
- **To test:**
  1. Run S1f32: does `Tf` pass seeds 0 and 100 on all 14 shapes, and how much of the 1.94x survives?
  2. If ≥1.10x rotating and it passes, port it into AITER. Prefer extending the `fused_rms_fp8_group_quant` / GEMM+norm fusion family; otherwise copy the closest fused GEMM+RMSNorm+quant precedent. Enable only the M range that wins (prefill at 4096 loses).
  3. Run op_tests + branch-vs-main gates as above.

---

## Not in scope / dropped

- Kimi attn_res (S10), Qwen3.5 GDN/prologue (S5/S12), DeepSeek-R1 qkv_a_norm (S1-R1), MiniMax: stopped. Fork worktrees for some of them exist but have no commits.
- PR #33 kernels with no templating fold and no AITER bench yet: MoE, GEMMs, dp_gather, dense prefill. Benchmarking them vs AITER is an open question.
- `glm52_fp8_ep8/fused_moe/decode_m128_m192.py` is non-deterministic at M=128 under hot routing. That's a PR #33 authors' bug; evidence in `notes/templating/gfx950/pr33/moe/det_*` on the artemis branch.
