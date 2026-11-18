"""S3 64-step chained drift check: every arm runs 64 decode steps on its own copy of the
same initial state with the same per-step projections; the fp32 torch recurrence
(S3.reference applied step by step, fp32 state) is the reference.  Reports the error of the
final recurrent state and of the last step's output, and the arm/B_port error ratio.

  python -m xbench.cands.S3_chain --shapes nx_m8,q5x_m32 [--steps 64]
"""
import argparse
import json

import torch

from xbench import core
from xbench.cands import S3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="nx_m8,nx_m64,q5f_m2,q5f_m128,q5x_m8,q5x_m32")
    ap.add_argument("--steps", type=int, default=64)
    args = ap.parse_args()
    out = open(core.RESULTS / "S3" / f"{core.IMG}_chain.jsonl", "a")
    for label in args.shapes.split(","):
        shape = next(s for s in S3.SHAPES if s["label"] == label)
        model = shape["model"]
        inp0, sha = core.load_blob("S3", f"{label}_s0")
        g = torch.Generator(device="cuda").manual_seed(1234)
        steps = [((torch.randn(inp0["projected_qkvz"].shape, generator=g, device="cuda")).to(torch.bfloat16),
                  (torch.randn(inp0["projected_ba"].shape, generator=g, device="cuda")).to(torch.bfloat16))
                 for _ in range(args.steps)]
        # reference recurrence
        ref_in = core.clone(inp0)
        idx = ref_in["cache_indices"].long()
        for qkvz, ba in steps:
            ref_in["projected_qkvz"], ref_in["projected_ba"] = qkvz, ba
            r = S3.reference(model, ref_in)
            ref_in["conv_state"][idx] = r["conv_touched"]
            ref_in["delta_state"][idx] = r["delta_touched"]
        ref_state = ref_in["delta_state"][idx].clone()
        ref_last = r["normalized"]
        res = {}
        for name, builder in S3.arms(shape).items():
            try:
                m = core.clone(inp0)
                fn = builder(m)
                for qkvz, ba in steps:
                    m["projected_qkvz"].copy_(qkvz)
                    m["projected_ba"].copy_(ba)
                    o = fn()
                torch.cuda.synchronize()
                st = core.cmp_close(m["delta_state"][idx], ref_state, atol=1e-3, rtol=1e-3, ulp=False)
                if "normalized" in o:
                    last = core.cmp_close(o["normalized"], ref_last)
                else:
                    sc = o["scales"]
                    deq = (o["quantized"].float().reshape(sc.shape[0], -1, 128) * sc[..., None]).reshape(ref_last.shape)
                    last = core.cmp_close(deq, ref_last, atol=2 ** -4, rtol=2 ** -3, ulp=False)
                res[name] = dict(state_max_abs=st["max_abs"], state_med_abs=st["med_abs"],
                                 state_frac_fail_1e3=st["frac_fail"], state_nonfinite=st["nonfinite"],
                                 out_max_abs=last["max_abs"], out_med_abs=last["med_abs"],
                                 out_frac_fail=last["frac_fail"])
            except core.Unavailable as e:
                res[name] = dict(unavailable=str(e))
            except Exception as e:  # noqa: BLE001
                res[name] = dict(error=repr(e))
        if "B_port" in res and "state_med_abs" in res.get("B_port", {}):
            b = res["B_port"]
            for name, v in res.items():
                if "state_med_abs" in v:
                    v["state_med_ratio_vs_Bport"] = v["state_med_abs"] / max(b["state_med_abs"], 1e-30)
                    v["state_max_ratio_vs_Bport"] = v["state_max_abs"] / max(b["state_max_abs"], 1e-30)
        rec = dict(type="chain", shape=label, steps=args.steps, input_sha=sha, results=res)
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
