# Kimi-K3 latent-projection MXFP4 preshuffled GEMM — tuning note

## Status: NOT tuned. Investigated, then deliberately not pursued.

`GEMM-AFP4WFP4_PRESHUFFLED` has no tuned entry on gfx950 for the two Kimi-K3
latent-MoE projection shapes:

    N=3584, K=7168     (routed_expert_down_proj, logical K)
    N=7168, K=3584     (routed_expert_up_proj,  logical K)

Verified absent in v0.1.19 and in main as of 878d60d77 (2026-08-15). The whole
preshuffled family is stale: the 28 gfx950 shape files are byte-identical
across 182 commits, last substantively touched 2026-02-24 (#2095), default
2026-04-10 (#2671). Both shapes therefore fall back to the default heuristic
(`BLOCK_N=64, BLOCK_K=256, NUM_KSPLIT=1`).

Prior art: aiter#4603 (XiaobingSuper, opened 2026-08-06, closed unmerged
2026-08-11) tuned *these exact shapes* for the CK/ASM `a4w4_blockscale` path
and was abandoned with no successor. Worth asking why before redoing it.

## Why we stopped

Measured on MI355X gfx950 for the consuming vLLM change:

- The preshuffled Triton GEMM is not the best kernel for these shapes at
  prefill sizes anyway. At M=4096/8192 the plain Triton GEMM wins for
  `down_proj` and the ASM `gemm_a4w4(bpreshuffle=True)` wins for `up_proj`;
  the preshuffled path is the worst of the three there — consistent with it
  running an untuned heuristic config.
- But the ceiling is what killed it: end to end, MXFP4 on these two layers is
  1.003× on a prefill-heavy workload (they are ~6% of prefill compute) and
  0.892× on a decode-heavy one. Recovering 20–30% of the kernel through tuning
  maps to roughly 0.3% end to end in the best case.

So tuning is real, open, and genuinely useful to someone — just not
load-bearing for the change that motivated it.

## If someone does pick it up

- Tooling: `aiter/ops/triton/utils/_triton/tunning/screen.py` with
  `ut_afp4wfp4_gemm_preshuffle.py`, one M bucket per GPU.
- `GEMM-AFP4WFP4_PRESHUFFLED` is still a **legacy** family, so files go in
  `aiter/ops/triton/configs/gemm/` with the arch prefix, NOT the new nested
  `configs/<arch>/<backend>/<op>/<dtype>/` layout. Per `configs/CLAUDE.md`, a
  lone nested file for a family whose default lives in legacy is silently
  invisible — the directory probe picks one directory and ignores the other.
- `K` in AFP4WFP4 filenames is the **logical** K (`2 * K_bytes`). Tuning output
  named by the packed byte width will never be found.
- Kernel constraints: `BLOCK_SIZE_M < 32` for M<32 and `>= 32` for M>=32;
  `BLOCK_SIZE_K >= 256`; `NUM_KSPLIT=1` (split-K for this kernel is still a
  TODO at `gemm/basic/gemm_afp4wfp4.py`).
- Note aiter#4630 (2026-08-10) fixed `_get_config` to take `K_bytes` rather
  than `K_elems`. vLLM pins v0.1.19, which predates it, so on that pin the
  lookup probes for 2× the true K. Measured impact on these two shapes: none —
  both lookups return the identical config because neither shape is tuned.
