"""S1 GLM-5.2 FP8 qkv_a_norm with fp32 split-K partials (precision fix experiment).

T  : served template glm52_fp8/qkv_a_norm_r3 (bf16/fp16 split-K partials, as in S1)
Tf : the same kernels copied to xbench/s1f32 with every split-K partial buffer (and the M=8
     intra-CTA wave exchange) in fp32; nothing else changed.
B, B1 : as in S1 (SGLang/AITER path).
"""
import importlib.util
import pathlib
import sys

from xbench.cands import S1 as s1

SHAPES = s1.SHAPES
MODE = "graph"
SEEDS = (0, 100)
AA = ["Tf", "B1"]
gen = s1.gen
check = s1.check

_TF = pathlib.Path(__file__).resolve().parent.parent / "s1f32/qkv_a_norm_r3/qkv_a_norm_template.py"


def _tf_entry():
    name = "_xb_s1f32_template"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _TF)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name].mla_qkv_a_norm


def _Tf(inp):
    fn = _tf_entry()
    kw = dict(rope_dim=s1.NR, eps=s1.EPS, weight_preshuffled=True)

    def call():
        q, kv, kr = fn(**{k: inp[k] for k in ("hidden", "weight", "q_gamma", "kv_gamma", "weight_scales")}, **kw)
        return dict(q=q, kv=kv, kr=kr)
    return call


def arms(shape):
    return {"T": s1._T, "Tf": _Tf, "B": s1._B, "B1": s1._B1}
