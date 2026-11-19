"""Markdown tables from results_074/S4pr/*.jsonl: small, 2.4 GB (_L) and 4.3 GB (_XL) pools."""
import json, pathlib, statistics, sys
R = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).parent / "results_074" / "S4pr")
ORDER = [1, 2, 4, 8, 12, 16, 24, 32]
POOLS = [("", "262144 rows, 151 MB"), ("_L", "4194304 rows, 2.42 GB (> 2^31)"), ("_XL", "7456541 rows, 4.29 GB (> 2^32, served)")]
t, n = {}, {}
for p in sorted(R.glob("n38_b[0-9].jsonl")):
    for line in open(p):
        x = json.loads(line)
        if x.get("type") == "timing":
            t.setdefault((x["shape"], x["cache"]), {}).setdefault(x["arm"], []).extend(x["raw"])
        elif x.get("type") == "numerics":
            n[(x["shape"], x["seed"], x["arm"])] = x
        elif x.get("type") == "error":
            print("ERROR", x["shape"], x.get("arm"), x["reason"][:200])
med = statistics.median
for sfx, desc in POOLS:
    print(f"\n#### device time, graph replay, pool {desc} (median of pooled rounds, us)\n")
    print("| rows | cache | T (branch) | T#2 (A/A) | Bm2 main fp8 dots | Bm2#2 | Bm1 main bf16 dots | Tpk (pack) | T vs Bm2 | T vs Bm1 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    sp2, sp1 = [], []
    for m in ORDER:
        for cache in ("hot", "rotating"):
            a = t.get((f"fp8ep8_tp8_h64_m{m}{sfx}", cache))
            if not a:
                continue
            f = lambda k: f"{med(a[k]):.2f}" if k in a else "-"
            r = lambda k: f"{med(a[k]) / med(a['T']):.2f}x" if k in a and "T" in a else "-"
            print(f"| {m} | {cache} | {f('T')} | {f('T#2')} | {f('Bm2')} | {f('Bm2#2')} | {f('Bm1')} | {f('Tpk')} | {r('Bm2')} | {r('Bm1')} |")
            if cache == "rotating" and "T" in a:
                if "Bm2" in a: sp2.append(med(a["Bm2"]) / med(a["T"]))
                if "Bm1" in a: sp1.append(med(a["Bm1"]) / med(a["T"]))
    if sp2: print(f"\nrotating geomean T vs Bm2: {statistics.geometric_mean(sp2):.3f}x  (min {min(sp2):.2f}x, n={len(sp2)})")
    if sp1: print(f"rotating geomean T vs Bm1: {statistics.geometric_mean(sp1):.3f}x  (min {min(sp1):.2f}x)")
print("\n#### numerics vs fp32 reference (bf16 criterion atol 2^-5, rtol 1e-2): pass / max_abs / frac out of tol\n")
print("| rows | pool | seed | T | Tpk | Bm2 | Bm1 |")
print("|---|---|---|---|---|---|---|")
def cell(x):
    if x is None: return "-"
    c = x["stats"]["out"]
    return f"{'PASS' if c['pass_'] else 'FAIL'} {c['max_abs']:.4f} {c['frac_fail']:.2e}"
for sfx, desc in POOLS:
    for m in ORDER:
        for seed in (0, 100):
            row = [cell(n.get((f"fp8ep8_tp8_h64_m{m}{sfx}", seed, arm))) for arm in ("T", "Tpk", "Bm2", "Bm1")]
            if all(r == "-" for r in row): continue
            print(f"| {m} | {desc.split(',')[0]} | {seed} | " + " | ".join(row) + " |")
