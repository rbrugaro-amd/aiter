# AITER PR drafts (templated gfx950 kernels, GLM-5.2 / artemis PR #33 scope)

Docs-only branch: PR descriptions for AITER PRs prepared from the Artemis templated kernels of the
GLM-5.2 FP8 TP8/EP8/DP8 pack (`glm52_fp8_ep8`, artemis PR #33) and the shared `glm52_fp8` kernels
it uses. Each PR's code is on its own branch of this fork; nothing has been opened upstream.

| PR draft | code branch | bench headline | status |
|---|---|---|---|
| [Sparse MLA decode, EP8 H=64 fp8 KV (`sparse_mla_fwd`)](pr_drafts/sparse-mla-decode-gfx950.md) | `sparse-mla-decode-gfx950` @ `bf1304b7e` | 1.35-2.50x rotating vs upstream main fp8-dot path (geomean 1.63x), 1.55-3.19x vs the bf16-dot path; numerics pass both seeds | **ready for review** |
| [GLM-5.2 FP8 DSA indexer prepare (new op `dsa_indexer_prepare`)](pr_drafts/glm52-fp8-indexer-prepare.md) | `glm52-fp8-indexer-prepare` (head 2c285e7) | 4.47x rotating geomean vs upstream main chain (3.1-5.9x), 2 launches vs 6; numerics pass | **branch pushed, ready for review** |
| [GLM-5.2 FP8 qkv_a + RMSNorm (new op `fused_mla_qkv_a_proj_rmsnorm`)](pr_drafts/mla-qkv-a-norm-quant-fused.md) | `mla-qkv-a-norm-quant-fused` (head 0393961) | 2.03x rotating geomean vs AITER chain, M 1-32; numerics pass both seeds | **branch pushed, ready for review** |

Upstream-facing PR bodies (no internal references; paste these): `pr_drafts/upstream/<branch>.md`. Internal versions with the pre-port columns and origin notes stay in `pr_drafts/<branch>.md`.

Handoff (branches, image, what to test per PR): [HANDOFF_glm52_aiter_prs.md](HANDOFF_glm52_aiter_prs.md).

Assessment and benchmark data behind these drafts: `benchmarks/` on this branch, and
`/home/rbrugaro/code/templating-infra/aiter/` on the cluster (assessment.md, bench_plan.md, bench/results/).
