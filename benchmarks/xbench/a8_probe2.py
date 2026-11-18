"""A8 probe via candidate modules: T arm only, one shape, n37 vs n38 bitwise."""
import importlib, sys, torch, triton
from xbench import core
res = {}
for cand, label in [a.split(":") for a in sys.argv[1:]]:
    tag = f"{cand}:{label}"
    try:
        mod = importlib.import_module(f"xbench.cands.{cand}")
        shape = next(s for s in mod.SHAPES if s["label"] == label)
        inp, _ = core.load_blob(cand, f"{label}_s0")
        outs = mod.arms(shape)["T"](core.clone(inp))()
        torch.cuda.synchronize()
        res[tag] = {k: v.detach().cpu() for k, v in outs.items() if isinstance(v, torch.Tensor)}
        print("A8", core.IMG, triton.__version__, tag, "OK", flush=True)
    except Exception as e:  # noqa: BLE001
        print("A8", core.IMG, triton.__version__, tag, "FAIL", repr(e)[:400], flush=True)
torch.save(res, f"/scratch/xb/results/a8b_{core.IMG}.pt")
