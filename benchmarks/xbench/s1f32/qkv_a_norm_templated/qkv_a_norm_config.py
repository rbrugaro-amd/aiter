"""Instance table for qkv_a_norm_template.py: M -> regime module.

Packs: glm52_fp8_tp4_gfx950 and glm52_fp8_tp8_gfx950.  Their qkv_a_norm
manifests are byte-identical (the qkv_a weight is replicated), so the key is M
alone (signature (M, 0, 0)).  Shapes: hidden [M, 6144] BF16, weight [2624, 6144]
FP8 e4m3fn AITER-packed with 128x128 block scales [21, 48]; q 2048 / kv 512
(RMSNorm, eps 1e-5) / rope 64, all BF16 out.

The 11 hand-written files collapse to 8 programs.  The equivalence experiment
(tools/kernel-templating/equiv_matrix.py; benign, cancellation and wide draws,
then a 300-draw seed volume at every moved row with a positive control; see
notes/templating/gfx950/glm52_fp8/qkv_a_norm/REPORT.md) found these bit-exact
substitutes serving at device-time parity (Tier B):

  small  decode_shared_m8.py      M = 1, 2, 4 (replaces decode_shared_m1 and
                                  decode_shared_m4; M = 2 moves off decode_shared_m2_m16)
  wide   decode_shared_m12_248.py M = 128 (replaces decode_shared_m128; same
                                  AMDGCN as the owner, T1 IDENTICAL)

Every other row stays on its owner's code (Tier A, verbatim copy).  Rejected
move (kept on the owner): M = 64 -> decode_shared_m32.py (_TUNING[64]) is
bit-exact on 300 draws but +1.85% T5 (ABBA x12, 11/12 sessions > 0; the
T1-identical M = 32 control in the same sessions -0.1%): perf.

Lookup mirrors the runtime (exact rows first, then the profile range rows);
anything else raises KeyError (fail closed).  M 4097..8192 has no row in the
manifests (native fallback) and raises here too.  Every row carries `origin`:
the owning file in the original manifest and, for substituted rows, the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

# regime name -> module file (beside this table; verbatim copies)
REGIMES = {
    "small": "decode_shared_m8.py",
    "m16": "decode_shared_m2_m16.py",
    "wide": "decode_shared_m12_248.py",
    "m32": "decode_shared_m32.py",
    "m64": "decode_shared_m64.py",
    "m256": "decode_shared_m256.py",
    "prefill": "prefill_shared_m1024_4096.py",
    "prefill_large": "prefill_shared_m8193_16384.py",
}


@dataclass(frozen=True)
class Instance:
    m: int
    regime: str
    origin: str


@dataclass(frozen=True)
class RangeInstance:
    lo: int
    hi: int
    regime: str
    origin: str


_TIER_B = "Tier B: BIT 9 draws + 300-draw volume; T5 ABBA x12 at parity (-1.25 .. -0.1%)"
_WIDE_M = (12, 24, 40, 48, 56, 72, 80, 88, 96, 104, 112, 120, 136, 144, 152, 160,
           168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248)

CONFIGS = (
    Instance(1, "small", f"owner decode_shared_m1.py; served by decode_shared_m8.py:_SCHEDULE[1] ({_TIER_B})"),
    Instance(2, "small", f"owner decode_shared_m2_m16.py; served by decode_shared_m8.py:_SCHEDULE[2] ({_TIER_B})"),
    Instance(4, "small", f"owner decode_shared_m4.py; served by decode_shared_m8.py:_SCHEDULE[4], T1 IDENTICAL ({_TIER_B})"),
    Instance(8, "small", "decode_shared_m8.py:_SCHEDULE[8] (owner)"),
    Instance(16, "m16", "decode_shared_m2_m16.py:_SCHEDULE[16] (owner)"),
    Instance(32, "m32", "decode_shared_m32.py:_TUNING[32] (owner)"),
    Instance(64, "m64", "decode_shared_m64.py:_project64 (owner; decode_shared_m32.py is bit-exact but +1.85% T5)"),
    Instance(128, "wide", f"owner decode_shared_m128.py; served by decode_shared_m12_248.py (bm=64, splits=6) ({_TIER_B})"),
    Instance(256, "m256", "decode_shared_m256.py (owner)"),
) + tuple(Instance(m, "wide", "decode_shared_m12_248.py (owner)") for m in _WIDE_M)

RANGES = (
    RangeInstance(1024, 4096, "prefill", "prefill_shared_m1024_4096.py range [1024, 4096] (owner)"),
    RangeInstance(8193, 16384, "prefill_large", "prefill_shared_m8193_16384.py range [8193, 16384] (owner)"),
)

_EXACT = {row.m: row for row in CONFIGS}
assert len(_EXACT) == len(CONFIGS), "duplicate exact row"
assert all(row.regime in REGIMES for row in CONFIGS + RANGES)


def lookup(m: int):
    """Exact rows, then range rows; anything else raises KeyError."""
    row = _EXACT.get(m)
    if row is not None:
        return row
    for r in RANGES:
        if r.lo <= m <= r.hi:
            return r
    raise KeyError(f"glm52_fp8 qkv_a_norm: no instance for M={m}")
