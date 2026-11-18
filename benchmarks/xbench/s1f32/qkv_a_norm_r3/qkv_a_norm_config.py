"""Round-3 instance table for glm52_fp8 qkv_a_norm (triage item #59).

Loads the round-1 table (`glm52_fp8/qkv_a_norm_templated/qkv_a_norm_config.py`)
unchanged by path and makes one change: M = 16 moves from regime `m16`
(decode_shared_m2_m16.py) to regime `small`, which now loads this directory's
decode_shared_m8.py (round-1 file with _SCHEDULE[16] = decode_shared_m2_m16's
plan).  Regime `m16` is dropped (8 -> 7 regimes).  Lookup semantics are the
round-1 ones.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_R1_CONFIG = (_HERE / "../qkv_a_norm_templated/qkv_a_norm_config.py").resolve()


def _load_r1():
    spec = importlib.util.spec_from_file_location(
        "_s1f32_glm52_fp8_qkv_a_norm_r1_config_for_r3", _R1_CONFIG)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_r1 = _load_r1()

REGIMES = {regime: f"../qkv_a_norm_templated/{f}" for regime, f in _r1.REGIMES.items()
           if regime != "m16"}
REGIMES["small"] = "decode_shared_m8.py"

Instance = _r1.Instance
RangeInstance = _r1.RangeInstance

_MOVED = {16: Instance(16, "small", "owner decode_shared_m2_m16.py:_SCHEDULE[16]; served by round-3 "
                                    "decode_shared_m8.py:_SCHEDULE[16] (same plan, T1-identical)")}
CONFIGS = tuple(_MOVED.get(r.m, r) for r in _r1.CONFIGS)
RANGES = _r1.RANGES
_EXACT = {row.m: row for row in CONFIGS}
assert len(_EXACT) == len(CONFIGS)
assert all(row.regime in REGIMES for row in CONFIGS + RANGES)


def lookup(m: int):
    row = _EXACT.get(m)
    if row is not None:
        return row
    for r in RANGES:
        if r.lo <= m <= r.hi:
            return r
    raise KeyError(f"glm52_fp8 qkv_a_norm: no instance for M={m}")
