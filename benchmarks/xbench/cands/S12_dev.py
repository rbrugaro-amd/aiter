"""Supplementary device-only time for S12 arms: sum of GPU kernel durations per call
(torch.profiler, 5 calls after warmup, median over 3 repeats).  Eager headline includes host
launch gaps; this isolates kernel time.  Usage: python -m xbench.cands.S12_dev [shapes...]"""
import json, sys, statistics
import torch
from torch.profiler import ProfilerActivity, profile
from xbench import core
import importlib, os
CAND = os.environ.get('XB_CAND', 'S12')
S12 = importlib.import_module(f'xbench.cands.{CAND}')

out = []
shapes = [s for s in S12.SHAPES if len(sys.argv) < 2 or s["label"] in sys.argv[1:]]
for s in shapes:
    inp, _ = core.load_blob(CAND, f"{s['label']}_s0")
    for name, b in S12.arms(s).items():
        try:
            fn = b(inp)
        except core.Unavailable:
            continue
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        reps = []
        for _ in range(3):
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
            tot = sum(e.device_time for e in prof.events()
                      if str(getattr(e, "device_type", "")).endswith("CUDA"))
            reps.append(tot / 5)
        us = statistics.median(reps)
        out.append(dict(shape=s["label"], arm=name, device_us=us, img=core.IMG))
        print(f"[{core.IMG}] dev {s['label']} {name} {us:.1f}", flush=True)
    del inp
    torch.cuda.empty_cache()
p = core.RESULTS / CAND / f"{core.IMG}_devtime.json"
p.write_text(json.dumps(out, indent=1))
print("wrote", p)
