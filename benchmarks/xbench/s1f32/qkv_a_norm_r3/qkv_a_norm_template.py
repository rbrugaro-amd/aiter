"""GLM-5.2 FP8 MLA qkv_a projection + q/kv latent RMSNorm, both glm52_fp8 TP profiles (round-3 dispatcher).

Round 3 (triage item #59): same host as glm52_fp8/qkv_a_norm_templated/qkv_a_norm_template.py,
with this directory's table (M = 16 on regime `small`, regime `m16` dropped).
Regime paths in REGIMES are relative to this directory.

One table-driven host for every shipped signature of the glm52_fp8_tp4_gfx950
and glm52_fp8_tp8_gfx950 packs (their qkv_a_norm manifests are identical: the
qkv_a weight is replicated, so the key is M alone).  The kernel bodies live in
regime modules beside this file -- verbatim copies of the hand-written files
that survive the merge -- and `qkv_a_norm_config.lookup(M)` maps each shipped
signature to one of them (exact rows first, then the profile's range rows;
anything else raises KeyError).

The regime copies keep their absolute imports of the shared device helpers
`xbench.s1f32.qkv_a_norm._decode` / `._decode_tiled_projection`
(listed as source_dependencies in the templated manifests).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import torch

_HERE = pathlib.Path(__file__).resolve().parent


def _load(filename: str, name: str):
    # The Artemis plugin loads candidates with spec_from_file_location, so the
    # siblings are not importable as a package; load them by path under stable
    # unique names (Triton keys its cache on the source, not the module name).
    full = f"_s1f32_glm52_fp8_qkv_a_norm_r3_{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, (_HERE / filename).resolve())
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


_config = _load("qkv_a_norm_config.py", "config")
lookup = _config.lookup
_MODULES = {regime: _load(filename, regime) for regime, filename in _config.REGIMES.items()}
_ENTRIES = {regime: module.mla_qkv_a_norm for regime, module in _MODULES.items()}
# For tools/kernel-templating/verify.py (T1/T2 AMDGCN capture).
__kernel_modules__ = tuple(_MODULES.values())


def _bind_constexpr_helpers():
    # Every manifest row sets `module_local_constexpr_helper`, so the runtime
    # (artemis_runtime/profile.py) calls bind_prefill_constexpr_helpers on the
    # module it loads -- this one.  It has no `triton` global, so that call is
    # a no-op here; the kernels that use `triton.next_power_of_2` inside jitted
    # bodies live in the regime modules.  Forward the same binding to each of
    # them (no-op unless SGLang has replaced the helper).
    try:
        from artemis_runtime.triton_compat import bind_prefill_constexpr_helpers
    except ImportError:
        return
    for module in _MODULES.values():
        bind_prefill_constexpr_helpers(module)


_bind_constexpr_helpers()


def mla_qkv_a_norm(hidden: torch.Tensor, weight: torch.Tensor, q_gamma: torch.Tensor,
                   kv_gamma: torch.Tensor, weight_scales: torch.Tensor | None = None, *,
                   rope_dim: int = 64, eps: float = 1e-05, weight_preshuffled: bool = False):
    entry = _ENTRIES[lookup(hidden.shape[0]).regime]
    return entry(hidden, weight, q_gamma, kv_gamma, weight_scales, rope_dim=rope_dim,
                 eps=eps, weight_preshuffled=weight_preshuffled)
