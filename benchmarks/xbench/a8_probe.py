"""A8 optional probe: can a template compile/run unchanged on Triton 3.7 (image n37)?
Saves outputs so they can be compared bitwise with the same template on n38."""
import sys, traceback, torch, triton
from xbench import tload, core
import kimi_k3__attention_residual_norm as sp10

out = {}
cases = [("S10", "kimi_k3/attention_residual_norm_xl/arn_wide_xl.py", "attention_residual_norm_wide_xl",
          "kimi_k3_tp8_gfx950/attention_residual_norm.json", (64, 4, 4)),
         ("S10", "kimi_k3/attention_residual_norm_xl/arn_fused_xl.py", "attention_residual_norm_fused_xl",
          "kimi_k3_tp8_gfx950/attention_residual_norm.json", (8, 4, 4))]
for cand, path, entry, prof, key in cases:
    tag = f"{cand}:{path.split('/')[-1]}:{key}"
    try:
        fn = tload.load(path, entry)
        inp = sp10._make_inputs(key, 0, "cuda")
        kw = sp10._call_kwargs(key)
        res = fn(inp["prefix"], inp["addend"], inp["bank"], inp["score_weight"], inp["output_weight"], **kw)
        torch.cuda.synchronize()
        res = res if isinstance(res, (tuple, list)) else (res,)
        out[tag] = [r.cpu() for r in res if isinstance(r, torch.Tensor)] + [inp["bank"].cpu()]
        print("A8", core.IMG, triton.__version__, tag, "OK", flush=True)
    except Exception as e:  # noqa: BLE001
        print("A8", core.IMG, triton.__version__, tag, "FAIL", repr(e)[:300], flush=True)
torch.save(out, f"/scratch/xb/results/a8_{core.IMG}.pt")
