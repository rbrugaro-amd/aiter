"""S10 eager wall time (host overhead; plan S10 T4-style): per-call wall us over 300 eager calls."""
import json, time, torch
from xbench import core
from xbench.cands import S10

out = []
for label in ("m1_v4_md5", "m8_v8_md4", "m64_v4_md6", "m256_v8_md4"):
    s = next(x for x in S10.SHAPES if x["label"] == label)
    inp, _ = core.load_blob("S10", f"{label}_s0")
    for name, b in S10.arms(s).items():
        fn = b(core.clone(inp))
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        res = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(300):
                fn()
            torch.cuda.synchronize()
            res.append((time.perf_counter() - t0) / 300 * 1e6)
        res.sort()
        out.append(dict(shape=label, arm=name, eager_wall_us=res[2]))
        print(label, name, round(res[2], 2), flush=True)
p = core.RESULTS / "S10" / "eager_wall.json"
p.write_text(json.dumps(out, indent=1))
