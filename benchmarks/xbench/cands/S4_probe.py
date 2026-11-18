import sys, torch
from xbench import core, tload
import glm52_fp8__sparse_paged_mla as sp
tpl = {64: "glm52_fp8/sparse_paged_mla_r3/spm_tp8_split_template.py", 128: "glm52_fp8/sparse_paged_mla_r3/spm_tp8_policy_template.py",
       8: "glm52_fp8/sparse_paged_mla_xl/decode_tp8_m8_x.py", 12: "glm52_fp8/sparse_paged_mla_r3/spm_tp8_policy_template.py"}
for m in (8, 12, 64, 128):
    fn = tload.load(tpl[m], "sparse_paged_mla", "glm52_fp8_tp8_gfx950/sparse_paged_mla_templated.json")
    for pool in (65536, 262144, 4194304):
        sp._POOL = pool
        d = sp.make_inputs(f"tp8m{m}", 0, "cuda")
        call = lambda: fn(d["query"], d["cache"], d["slots"], softmax_scale=0.0625, value_dim=512, quantize_query=True)
        call(); torch.cuda.synchronize()
        e = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        e[0].record()
        for _ in range(20): call()
        e[1].record(); torch.cuda.synchronize()
        eager = e[0].elapsed_time(e[1]) * 50
        g = core.measure({"T": [call]}, 200, rounds=5, halves=1)["T"]
        print(f"m{m} pool {pool}: eager {eager:.1f} us  graph {sorted(g)[2]:.1f} us", flush=True)
