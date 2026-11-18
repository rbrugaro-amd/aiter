"""Shared harness core: inputs with sha256, rings, graph timing (ABBA + A/A), calibration,
numerics helpers, environment capture.  Runs unchanged in both images (n38 and n37)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import subprocess
import time
from statistics import median

import torch

ROOT = pathlib.Path(os.environ.get("XB_ROOT", "/scratch/xb"))
INPUTS = ROOT / "inputs"
RESULTS = ROOT / "results"
IMG = os.environ.get("XB_IMG", "n38")
DEV = "cuda"


# ----------------------------------------------------------------------------- inputs
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest(cand):
    p = INPUTS / cand / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_blob(cand, name, obj):
    """Save a dict of tensors (moved to CPU) and record its sha256 in the manifest."""
    d = INPUTS / cand
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.pt"
    torch.save(_to(obj, "cpu"), p)
    man = _manifest(cand)
    man[p.name] = sha256_file(p)
    (d / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    return man[p.name]


def load_blob(cand, name, device=DEV):
    p = INPUTS / cand / f"{name}.pt"
    want = _manifest(cand).get(p.name)
    got = sha256_file(p)
    if want != got:
        raise RuntimeError(f"sha256 mismatch for {p}: manifest {want} file {got}")
    return _to(torch.load(p, map_location="cpu", weights_only=False), device), got


def have_blob(cand, name):
    return (INPUTS / cand / f"{name}.pt").exists()


def _to(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to(v, device) for v in obj)
    return obj


def clone(obj):
    if isinstance(obj, torch.Tensor):
        if obj.is_contiguous():
            return obj.clone()
        out = torch.empty_strided(obj.shape, obj.stride(), dtype=obj.dtype, device=obj.device)
        out.copy_(obj)
        return out
    if isinstance(obj, dict):
        return {k: clone(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(clone(v) for v in obj)
    return obj


def nbytes(obj):
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(nbytes(v) for v in obj)
    return 0


def ring_size(set_bytes, budget=512 << 20, cap=256):
    """src/bench/gemm.py:weight_copies() sizing: power of two, 512 MiB budget, <= cap."""
    n = max(1, budget // max(1, set_bytes))
    n = 1 << int(math.floor(math.log2(n)))
    return max(2, min(cap, n))


# ----------------------------------------------------------------------------- timing
def _capture(calls, count):
    for c in calls:
        c()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(count):
            calls[i % len(calls)]()
    g.replay()  # discarded warm replay
    torch.cuda.synchronize()
    return g


def _time_graph(g, count):
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(100_000)
    s.record()
    g.replay()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1000.0 / count


def probe_us(calls):
    """Rough per-call us from a 20-call graph (used to pick count)."""
    g = _capture(calls, 20)
    t = min(_time_graph(g, 20) for _ in range(3))
    del g
    return t


def pick_count(us):
    if us < 3:
        return 2000
    if us < 20:
        return 1000
    if us < 200:
        return 200
    return 20


def measure(arms, count, rounds=15, reset=None, aa=None, halves=2):
    """arms: {name: [callables]} (ring members cycled).  `aa`: arm names that get a second,
    independently captured graph '<name>#2' (A/A control).  Rounds are split in `halves`
    with fresh graphs each half (plan A2: loop measure twice and pool the rounds).
    ABBA: arm order alternates forward/reverse per round.
    Returns {name: [per-round us]}."""
    aa = list(aa or [])
    names = list(arms) + [f"{a}#2" for a in aa if a in arms]
    times = {n: [] for n in names}
    per_half = [rounds // halves + (1 if i < rounds % halves else 0) for i in range(halves)]
    for nh in per_half:
        graphs = {}
        for n in names:
            graphs[n] = _capture(arms[n.split("#")[0]], count)
        for r in range(nh):
            order = names if r % 2 == 0 else names[::-1]
            for n in order:
                if reset is not None:
                    reset()
                times[n].append(_time_graph(graphs[n], count))
        del graphs
        torch.cuda.synchronize()
    return times


def measure_eager(arms, calls_per_round=20, rounds=7, reset=None, aa=None):
    """For non-graph-safe ops (S12): events around `calls_per_round` eager calls, ABBA."""
    aa = list(aa or [])
    names = list(arms) + [f"{a}#2" for a in aa if a in arms]
    for n in arms:
        for c in arms[n]:
            c()
    torch.cuda.synchronize()
    times = {n: [] for n in names}
    for r in range(rounds):
        order = names if r % 2 == 0 else names[::-1]
        for n in order:
            calls = arms[n.split("#")[0]]
            if reset is not None:
                reset()
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for i in range(calls_per_round):
                calls[i % len(calls)]()
            e.record()
            e.synchronize()
            times[n].append(s.elapsed_time(e) * 1000.0 / calls_per_round)
    return times


def summarize(ts):
    ts = sorted(ts)
    med = median(ts)
    k = len(ts)
    p10 = ts[max(0, int(round(0.1 * (k - 1))))]
    p90 = ts[min(k - 1, int(round(0.9 * (k - 1))))]
    return dict(median=med, p10=p10, p90=p90, spread_pct=100.0 * (p90 - p10) / med, n=k)


# ----------------------------------------------------------------------------- controls
def calibration():
    """Reference kernels per block: 1 GiB bf16 copy_ (bandwidth) and 8192^3 bf16 mm."""
    out = {}
    a = torch.empty(512 << 20, dtype=torch.bfloat16, device=DEV).normal_()
    b = torch.empty_like(a)
    t = measure({"copy": [lambda: b.copy_(a)]}, count=10, rounds=7, halves=1)["copy"]
    out["copy_1GiB_us"] = summarize(t)
    out["copy_GBps"] = 2 * (1 << 30) / (out["copy_1GiB_us"]["median"] * 1e-6) / 1e9
    del a, b
    x = torch.randn(8192, 8192, dtype=torch.bfloat16, device=DEV)
    y = torch.randn(8192, 8192, dtype=torch.bfloat16, device=DEV)
    t = measure({"mm": [lambda: torch.mm(x, y)]}, count=10, rounds=7, halves=1)["mm"]
    out["mm_8192_us"] = summarize(t)
    out["mm_TFLOPs"] = 2 * 8192 ** 3 / (out["mm_8192_us"]["median"] * 1e-6) / 1e12
    del x, y
    torch.cuda.empty_cache()
    return out


def smi():
    gpu = os.environ.get("HIP_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        r = subprocess.run(["rocm-smi", "-d", gpu, "--showclocks", "--showpower", "--showtemp",
                            "--showperflevel", "--json"], capture_output=True, text=True, timeout=60)
        return json.loads(r.stdout) if r.stdout.strip().startswith("{") else r.stdout[-2000:]
    except Exception as e:  # noqa: BLE001
        return f"rocm-smi failed: {e!r}"


# ----------------------------------------------------------------------------- numerics
def _bf16_ulp_hist(out, ref):
    o = out.float().to(torch.bfloat16).view(torch.int16).int()
    r = ref.float().to(torch.bfloat16).view(torch.int16).int()
    # ordered int mapping for sign-magnitude
    o = torch.where(o < 0, -32768 - o, o)
    r = torch.where(r < 0, -32768 - r, r)
    d = (o - r).abs()
    n = d.numel()
    return {"0": (d == 0).sum().item() / n, "1": (d == 1).sum().item() / n,
            "2": (d == 2).sum().item() / n, ">2": (d > 2).sum().item() / n}


def cmp_close(out, ref, atol=2 ** -5, rtol=1e-2, ulp=True):
    """bf16/fp32 criterion: assert_close semantics + stats."""
    o = out.float()
    r = ref.float().to(o.device)
    if o.shape != r.shape:
        return dict(pass_=False, err=f"shape {tuple(o.shape)} vs {tuple(r.shape)}")
    fin = torch.isfinite(o)
    diff = (o - r).abs()
    tol = atol + rtol * r.abs()
    bad = (diff > tol) | ~fin
    big = r.abs() > 1e-3
    d = dict(
        max_abs=diff[fin].max().item() if fin.any() else float("inf"),
        med_abs=diff[fin].median().item() if fin.any() else float("inf"),
        max_rel=(diff[big] / r[big].abs()).max().item() if big.any() else 0.0,
        frac_fail=bad.float().mean().item(),
        nonfinite=int((~fin).sum().item()),
        pass_=bool(not bad.any()),
    )
    if ulp:
        d["ulp_bf16"] = _bf16_ulp_hist(o, r)
    return d


def cmp_fp8(codes, scales, ref_codes, ref_scales, scale_rtol=2 ** -10, max_frac=0.005):
    """fp8 e4m3 codes + fp32 group scales (A5)."""
    c = codes.view(torch.uint8).to(torch.int16)
    rc = ref_codes.to(codes.device).view(torch.uint8).to(torch.int16)

    def ordered(x):
        mag = x & 0x7F
        return torch.where((x & 0x80) != 0, -mag, mag)

    dc = (ordered(c) - ordered(rc)).abs()
    srel = ((scales.float() - ref_scales.float().to(scales.device)).abs()
            / ref_scales.float().abs().clamp_min(1e-30).to(scales.device)).max().item()
    mism = (dc != 0).float().mean().item()
    d = dict(code_mismatch=mism, code_maxdiff=int(dc.max().item()),
             frac_gt1=(dc > 1).float().mean().item(), scale_max_rel=srel)
    d["pass_"] = bool(srel <= scale_rtol and d["code_maxdiff"] <= 1 and mism <= max_frac) or bool(
        srel <= scale_rtol and d["frac_gt1"] == 0 and mism <= max_frac)
    return d


def cmp_bytes(a, b):
    a8 = a.contiguous().view(torch.uint8)
    b8 = b.contiguous().view(torch.uint8).to(a8.device)
    nd = (a8 != b8).sum().item()
    return dict(bytes_differ=nd, frac=nd / max(1, a8.numel()), pass_=nd == 0)


# ----------------------------------------------------------------------------- env
def env():
    from xbench.envinfo import info
    d = info()
    d["img"] = IMG
    d["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return d


class Unavailable(Exception):
    """Raised by an arm builder when the arm cannot run in this image (reason recorded)."""


# ----------------------------------------------------------------------------- sglang/triton
# SGLang 703e763d5f (sglang/srt/utils/triton_utils.py:49) does
#   setattr(triton, "next_power_of_2", <plain python fn>)
# at import time.  Under Triton 3.8 that breaks any Gluon/Triton kernel that calls
# triton.next_power_of_2 inside @jit code ("Unsupported function referenced"), which includes
# several of our templates.  The original is a constexpr_function and also works host-side, so
# the harness restores it after every SGLang import (sglang_import()).
import triton as _triton  # noqa: E402

_TRITON_NP2 = _triton.next_power_of_2


def restore_triton():
    _triton.next_power_of_2 = _TRITON_NP2


def sglang_import(modname):
    import importlib
    m = importlib.import_module(modname)
    restore_triton()
    return m
