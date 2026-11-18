"""One block of one candidate in one image.

  python -m xbench.run --cand S1 --gen                      # once, in n38: inputs + fp32 refs
  python -m xbench.run --cand S1 --block 1 [--shapes a,b] [--cache hot,rotating] [--numerics]

Candidate module xbench/cands/<cand>.py provides:
  SHAPES: list[dict] each with 'label'       MODE: 'graph' | 'eager'
  gen(shape, seed) -> (inputs: dict, ref: dict)          (run once, saved with sha256)
  arms(shape) -> {arm_name: builder}; builder(inp) -> callable() -> {out_name: tensor}
               (builders raise core.Unavailable(reason) when the arm cannot run here)
  check(arm, outs, ref, inp, shape) -> {out_name: stats with 'pass_'}
  optional: rotate(shape) -> list of input names to rotate (default: all tensors)
            reset(shape, ring) -> callable restoring mutated state (called before every timed round)
            AA (default ['T', first B arm]), B37_ARMS (arms timed in the n37 image), SEEDS
            count(shape) -> fixed graph count (default: auto by probe)
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
import traceback

import torch

from xbench import core


def log(f, rec):
    rec["t"] = time.time()
    f.write(json.dumps(rec, default=str) + "\n")
    f.flush()


def build_ring(cand, shape, inp, cache):
    if cache == "hot":
        return [inp], 1
    names = getattr(cand, "rotate", None)
    names = names(shape) if names else [k for k, v in inp.items() if isinstance(v, torch.Tensor)]
    set_bytes = sum(core.nbytes(inp[k]) for k in names)
    cap = getattr(cand, "RING_CAP", 256)
    n = core.ring_size(set_bytes, cap=cap)
    ring = [inp]
    for _ in range(n - 1):
        m = dict(inp)
        for k in names:
            m[k] = core.clone(inp[k])
        ring.append(m)
    return ring, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--cache", default="hot,rotating")
    ap.add_argument("--numerics", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--trace", action="store_true", help="profile kernel names per arm once (no timing)")
    ap.add_argument("--arms", default="", help="restrict to these arms")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cand = importlib.import_module(f"xbench.cands.{args.cand}")
    shapes = cand.SHAPES
    if args.shapes:
        want = args.shapes.split(",")
        shapes = [s for s in shapes if s["label"] in want]
    seeds = getattr(cand, "SEEDS", (0, 100))

    if args.gen:
        for s in shapes:
            for seed in seeds:
                name = f"{s['label']}_s{seed}"
                if core.have_blob(args.cand, name) and core.have_blob(args.cand, "ref_" + name):
                    continue
                inp, ref = cand.gen(s, seed)
                h1 = core.save_blob(args.cand, name, inp)
                h2 = core.save_blob(args.cand, "ref_" + name, ref)
                print(f"gen {name} {h1[:12]} ref {h2[:12]}", flush=True)
                del inp, ref
                torch.cuda.empty_cache()
        return

    out_dir = core.RESULTS / args.cand
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{core.IMG}_b{args.block}{args.tag}.jsonl"
    f = open(path, "a")
    log(f, dict(type="env", block=args.block, env=core.env()))
    log(f, dict(type="smi", when="before", smi=core.smi()))
    torch.manual_seed(0)
    if not args.no_timing:
        log(f, dict(type="calib", calib=core.calibration()))
    arm_filter = set(args.arms.split(",")) if args.arms else None
    b37 = getattr(cand, "B37_ARMS", None)
    order = shapes if args.block % 2 == 1 else shapes[::-1]
    caches = args.cache.split(",") if cand.MODE == "graph" else ["eager"]
    if getattr(cand, "CACHES", None):
        caches = [c for c in caches if c in cand.CACHES]

    def arm_set(s):
        a = cand.arms(s)
        if core.IMG == "n37":
            a = {k: v for k, v in a.items() if b37 and k in b37}
        if arm_filter:
            a = {k: v for k, v in a.items() if k in arm_filter}
        return a

    for s in (order if not args.no_timing else []):
        label = s["label"]
        inp0, sha = core.load_blob(args.cand, f"{label}_s0")
        for cache in [c for c in caches if c in s.get("caches", caches) or c == "eager"]:
            ring, nring = build_ring(cand, s, inp0, "hot" if cache == "eager" else cache)
            calls = {}
            for name, builder in arm_set(s).items():
                try:
                    calls[name] = [builder(m) for m in ring]
                    calls[name][0]()
                    torch.cuda.synchronize()
                except core.Unavailable as e:
                    log(f, dict(type="unavailable", shape=label, cache=cache, arm=name, reason=str(e)))
                    calls.pop(name, None)
                except Exception as e:  # noqa: BLE001
                    log(f, dict(type="error", shape=label, cache=cache, arm=name,
                                reason=repr(e), tb=traceback.format_exc()[-3000:]))
                    calls.pop(name, None)
                    torch.cuda.synchronize()
            if not calls:
                continue
            reset = cand.reset(s, ring) if hasattr(cand, "reset") else None
            aa = getattr(cand, "AA", None) or ["T"] + [k for k in calls if k.startswith("B")][:1]
            try:
                if cache == "eager":
                    times = core.measure_eager(calls, rounds=max(7, args.rounds // 2), reset=reset, aa=aa)
                    count = 20
                else:
                    count = cand.count(s) if hasattr(cand, "count") else None
                    if count is None:
                        count = core.pick_count(max(core.probe_us(c) for c in calls.values()))
                    times = core.measure(calls, count, rounds=args.rounds, reset=reset, aa=aa)
            except Exception as e:  # noqa: BLE001
                log(f, dict(type="error", shape=label, cache=cache, arm="*measure*",
                            reason=repr(e), tb=traceback.format_exc()[-3000:]))
                continue
            for name, ts in times.items():
                log(f, dict(type="timing", shape=label, cache=cache, arm=name, count=count,
                            ring=nring, input_sha=sha, raw=ts, **core.summarize(ts)))
            print(f"[{core.IMG} b{args.block}] {label} {cache} " + "  ".join(
                f"{n}={core.summarize(t)['median']:.2f}" for n, t in times.items()), flush=True)
            del calls, ring, times
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        del inp0

    if args.trace:
        from torch.profiler import ProfilerActivity, profile
        for s in shapes:
            label = s["label"]
            inp, sha = core.load_blob(args.cand, f"{label}_s0")
            for name, builder in arm_set(s).items():
                try:
                    fn = builder(core.clone(inp))
                    fn()
                    torch.cuda.synchronize()
                    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                        fn()
                        torch.cuda.synchronize()
                    kern = [(e.name, round(e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total, 2))
                            for e in prof.events() if getattr(e, "device_type", None) is not None
                            and str(e.device_type).endswith("CUDA")]
                    log(f, dict(type="trace", shape=label, arm=name, kernels=kern))
                    print(f"[{core.IMG}] trace {label} {name}: " + " | ".join(k for k, _ in kern), flush=True)
                except core.Unavailable as e:
                    log(f, dict(type="unavailable", shape=label, arm=name, reason=str(e)))
                    print(f"[{core.IMG}] trace {label} {name}: UNAVAILABLE {e}", flush=True)
                except Exception as e:  # noqa: BLE001
                    log(f, dict(type="error", shape=label, arm=name, phase="trace", reason=repr(e),
                                tb=traceback.format_exc()[-3000:]))
                    print(f"[{core.IMG}] trace {label} {name}: ERROR {e!r}", flush=True)
            del inp
    if args.numerics:
        for s in shapes:
            label = s["label"]
            for seed in seeds:
                inp, sha = core.load_blob(args.cand, f"{label}_s{seed}")
                ref, _ = core.load_blob(args.cand, f"ref_{label}_s{seed}")
                for name, builder in arm_set(s).items():
                    try:
                        m = core.clone(inp)
                        outs = builder(m)()
                        torch.cuda.synchronize()
                        st = cand.check(name, outs, ref, m, s)
                        ok = all(v.get("pass_", False) for v in st.values())
                        log(f, dict(type="numerics", shape=label, seed=seed, arm=name, input_sha=sha,
                                    stats=st, pass_=ok))
                        print(f"[{core.IMG}] numerics {label} s{seed} {name}: {'PASS' if ok else 'FAIL'} "
                              + "; ".join(f"{k}:max_abs={v.get('max_abs', v.get('code_mismatch', v.get('bytes_differ')))}"
                                          for k, v in st.items()), flush=True)
                    except core.Unavailable as e:
                        log(f, dict(type="unavailable", shape=label, seed=seed, arm=name, reason=str(e)))
                    except Exception as e:  # noqa: BLE001
                        log(f, dict(type="error", shape=label, seed=seed, arm=name, phase="numerics",
                                    reason=repr(e), tb=traceback.format_exc()[-3000:]))
                        print(f"[{core.IMG}] numerics {label} s{seed} {name}: ERROR {e!r}", flush=True)
                    torch.cuda.synchronize()
                del inp, ref
                torch.cuda.empty_cache()
    log(f, dict(type="smi", when="after", smi=core.smi()))
    f.close()
    print("wrote", path)


if __name__ == "__main__":
    main()
