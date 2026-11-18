"""Merge block jsonl files of one candidate into results/<cand>.csv and a markdown table.

  python3 -m xbench.merge S1 [--root results]     (pure python, runs on the host)

Rules (bench_plan A0/A1 adapted to one container):
  per arm, per block: median of 15 rounds; across blocks: median of block medians.
  best AITER arm = fastest arm whose name starts with 'B' (n38).  speedup = best_B / T.
  A/A = |X / X#2 - 1| per block (separately captured graph of the same arm), max over blocks
  and over A/A arms; block-ratio spread = (max - min) / 2 of per-block speedups.
  noise floor = max(A/A, block spread); parity band = max(2 x noise floor, 2%).
  cell: win if speedup - 1 > band and numerics pass for T in every seed; loss if
  1 - speedup > band; else parity.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from statistics import median


def load(root, cand):
    recs = []
    for p in sorted(glob.glob(os.path.join(root, cand, "*.jsonl"))):
        base = os.path.basename(p)
        img = base.split("_")[0]
        for line in open(p):
            r = json.loads(line)
            r["_img"] = img
            r["_file"] = base
            recs.append(r)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand")
    ap.add_argument("--root", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--weights", default="", help="json {shape_label: weight} for the summary")
    args = ap.parse_args()
    recs = load(args.root, args.cand)
    blocks = set()
    tim = defaultdict(dict)        # (img, shape, cache, arm) -> {block: median}
    for r in recs:
        if r.get("type") != "timing" or "_trace" in r["_file"] or "_xcheck" in r["_file"]:
            continue
        blk = r["_file"]
        blocks.add((r["_img"], blk))
        tim[(r["_img"], r["shape"], r["cache"], r["arm"])][blk] = r["median"]
    num = defaultdict(list)        # (img, shape, arm) -> [(seed, pass, stats)]
    for r in recs:
        if r.get("type") == "numerics" and "_trace" not in r["_file"]:
            num[(r["_img"], r["shape"], r["arm"])].append((r["seed"], r["pass_"], r["stats"]))
    calib = [(r["_img"], r["_file"], r["calib"]["copy_GBps"], r["calib"]["mm_TFLOPs"])
             for r in recs if r.get("type") == "calib"]
    errs = [r for r in recs if r.get("type") in ("error", "unavailable") and "_trace" not in r["_file"]]

    shapes, caches = [], []
    for (img, s, c, a) in tim:
        if s not in shapes:
            shapes.append(s)
        if c not in caches:
            caches.append(c)

    def key(s):
        digits = "".join(ch if ch.isdigit() else " " for ch in s).split()
        return (s.rstrip("0123456789_"), tuple(int(d) for d in digits), s)
    shapes.sort(key=key)

    rows = []
    for s in shapes:
        for c in caches:
            arms = {a: v for (img, ss, cc, a), v in tim.items() if img == "n38" and ss == s and cc == c}
            if "T" not in arms:
                continue
            t_med = median(arms["T"].values())
            bs = {a: median(v.values()) for a, v in arms.items() if a.startswith("B") and "#" not in a}
            xs = {a: median(v.values()) for a, v in arms.items() if a.startswith("X") and "#" not in a}
            if not bs:
                continue
            best = min(bs, key=bs.get)
            per_block = [arms[best][b] / arms["T"][b] for b in arms["T"] if b in arms[best]]
            aa = []
            for a, v in arms.items():
                if a.endswith("#2"):
                    base = arms.get(a[:-2], {})
                    aa += [abs(v[b] / base[b] - 1) for b in v if b in base]
            aa_max = max(aa) if aa else float("nan")
            spread = (max(per_block) - min(per_block)) / 2 if len(per_block) > 1 else 0.0
            noise = max(aa_max if aa else 0.0, spread)
            band = max(2 * noise, 0.02)
            sp = bs[best] / t_med
            b37 = {a: median(v.values()) for (img, ss, cc, a), v in tim.items()
                   if img == "n37" and ss == s and cc == c and "#" not in a}
            b37_best = min(b37.values()) if b37 else None

            def nstat(arm):
                L = num.get(("n38", s, arm), [])
                if not L:
                    return None, None, None
                ok = all(p for _, p, _ in L)
                mx = max(max((v.get("max_abs", 0) or 0) for v in st.values()) for _, _, st in L)
                ff = max(max((v.get("frac_fail", 0) or 0) for v in st.values()) for _, _, st in L)
                return ok, mx, ff
            tp, tmx, tff = nstat("T")
            bp, bmx, bff = nstat(best)
            passing = {a: v for a, v in bs.items() if nstat(a)[0] is not False}
            best_pass = min(passing, key=passing.get) if passing else None
            if sp - 1 > band:
                cell = "win"
            elif 1 - sp > band:
                cell = "loss"
            else:
                cell = "parity"
            if tp is False and bp is not False:
                cell += ", T numerics FAIL (B pass)"
            elif tp is False:
                cell += ", T numerics FAIL"
            rows.append(dict(
                candidate=args.cand, shape=s, cache=c, T_us=round(t_med, 3), best_B_arm=best,
                best_B_us=round(bs[best], 3),
                other_arms=json.dumps({a: round(v, 3) for a, v in {**bs, **xs}.items() if a != best}),
                B37_us=round(b37_best, 3) if b37_best else "",
                speedup_vs_best_B=round(sp, 3),
                speedup_min=round(min(per_block), 3), speedup_max=round(max(per_block), 3),
                speedup_vs_B37=round(b37_best / t_med, 3) if b37_best else "",
                aa_pct=round(100 * aa_max, 2) if aa else "", noise_floor_pct=round(100 * noise, 2),
                parity_band_pct=round(100 * band, 2), n_blocks=len(arms["T"]), cell=cell,
                best_passing_B_arm=best_pass or "",
                speedup_vs_best_passing_B=round(passing[best_pass] / t_med, 3) if best_pass else "",
                T_pass=tp, B_pass=bp, T_maxabs=tmx, B_maxabs=bmx, T_frac_fail=tff, B_frac_fail=bff,
            ))
    out_csv = os.path.join(args.root, f"{args.cand}.csv")
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    # markdown
    md = [f"### {args.cand}", "",
          "| shape | cache | T us | best AITER arm | B us | speedup (min-max) | vs best numerically-passing AITER arm | B37 us | noise % | cell | T num | B num |",
          "|---|---|---:|---|---:|---|---|---:|---:|---|---|---|"]
    for r in rows:
        md.append(f"| {r['shape']} | {r['cache']} | {r['T_us']:.2f} | {r['best_B_arm']} | {r['best_B_us']:.2f} | "
                  f"{r['speedup_vs_best_B']:.2f} ({r['speedup_min']:.2f}-{r['speedup_max']:.2f}) | "
                  f"{(r['best_passing_B_arm'] + ' ' + format(r['speedup_vs_best_passing_B'], '.2f')) if r['best_passing_B_arm'] else 'none passes'} | "
                  f"{r['B37_us'] if r['B37_us'] == '' else format(r['B37_us'], '.2f')} | {r['noise_floor_pct']} | "
                  f"{r['cell']} | {fmt(r['T_pass'])} | {fmt(r['B_pass'])} |")
    weights = json.loads(args.weights) if args.weights else {}
    for c in caches:
        sel = [r for r in rows if r["cache"] == c]
        if not sel:
            continue
        ws = [weights.get(r["shape"], 1.0 if not weights else 0.0) for r in sel]
        if sum(ws) <= 0:
            continue
        g = math.exp(sum(w * math.log(r["speedup_vs_best_B"]) for w, r in zip(ws, sel)) / sum(ws))
        md.append(f"\n{c}: {'weighted' if weights else 'uniform'} geomean speedup vs best AITER arm = {g:.3f} over {sum(1 for w in ws if w > 0)} shapes")
    md.append("\ncalibration (copy GB/s, mm TFLOP/s) per block: " + "; ".join(
        f"{i}/{f.split('.')[0]}: {cp:.0f}, {mm:.0f}" for i, f, cp, mm in calib))
    if errs:
        seen = set()
        md.append("\nerrors/unavailable:")
        for e in errs:
            k = (e["_img"], e.get("arm"), e.get("reason", "")[:200])
            if k in seen:
                continue
            seen.add(k)
            md.append(f"- {e['_img']} {e.get('shape')} {e.get('arm')}: {e['type']}: {e.get('reason', '')[:300]}")
    out_md = os.path.join(args.root, f"{args.cand}.md")
    open(out_md, "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def fmt(x):
    return "-" if x is None else ("pass" if x else "FAIL")


if __name__ == "__main__":
    main()
