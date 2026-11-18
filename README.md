# AITER PR drafts (templated gfx950 kernels, GLM-5.2 / artemis PR #33 scope)

Docs-only branch: PR descriptions for AITER PRs prepared from the Artemis templated kernels of the
GLM-5.2 FP8 TP8/EP8/DP8 pack (`glm52_fp8_ep8`, artemis PR #33) and the shared `glm52_fp8` kernels
it uses. Each PR's code is on its own branch of this fork; nothing has been opened upstream.

| PR draft | code branch | bench headline | status |
|---|---|---|---|
| [Sparse MLA decode, EP8 H=64 fp8 KV (`sparse_mla_fwd`)](pr_drafts/sparse-mla-decode-gfx950.md) | `sparse-mla-decode-gfx950` @ `bf1304b7e` | 1.35-2.50x rotating vs upstream main fp8-dot path (geomean 1.63x), 1.55-3.19x vs the bf16-dot path; numerics pass both seeds | **ready for review** |
| GLM-5.2 FP8 DSA indexer prepare | `glm52-fp8-indexer-prepare` | not benchmarked yet | in progress |
| GLM-5.2 FP8 qkv_a + RMSNorm + quant | `mla-qkv-a-norm-quant-fused` | 1.94x, but bf16 split-K partials fail a cancellation draw; fix in progress | in progress |

Handoff (branches, image, what to test per PR): [HANDOFF_glm52_aiter_prs.md](HANDOFF_glm52_aiter_prs.md).

Assessment and benchmark data behind these drafts: `benchmarks/` on this branch, and
`/home/rbrugaro/code/templating-infra/aiter/` on the cluster (assessment.md, bench_plan.md, bench/results/).
