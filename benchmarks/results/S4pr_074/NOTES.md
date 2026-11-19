# S4pr re-run with the pool-size gate lifted (2026-11-19, node crsuse2-m2m-v3-074, GPU 3)

Branch `sparse-mla-decode-gfx950` @ 24b3a0468 (PR ROCm/aiter#8220): `_h64_decode_config` no longer
rejects caches past 2^31 B; past 2^32 B the small kernel's NARROW_FORM 1/2 rows fall back to the int64
offset form (`_h64_cache_forms`). Same image (nightly1117-triton38), harness (`xbench/cands/S4pr.py`,
now with `_L` = 4194304-row / 2.42 GB and `_XL` = 7456541-row / 4.29 GB variants of every EP8 row),
3 blocks x 15 rounds, arms T / Bm2 / Bm1 / Tpk. Tables: `S4pr_tables.md` (from `summarize074.py`).

Headline: small-pool rows reproduce the node-114 numbers within 1%; large pools 1.22-2.2x vs upstream
main fp8 dots (geomean 1.49x at both sizes), all T numerics pass. Tests: `-k h64` 46 passed (16 new
`_big_pool` cases), full `test_sparse_mla.py` 187 passed, 2 skipped. A Kimi-K3 decode server was
co-resident on the node (GPU 3 use 1% at bench start).
