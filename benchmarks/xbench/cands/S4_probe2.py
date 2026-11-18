"""Is the fp8 split/policy slowdown pool-size or scatter driven?  T and AITER B2 at
pool 65536 / 262144 / 262144-with-slots-confined-to-a-65536-row-window / 4194304-window."""
import torch
from xbench import core, tload
import glm52_fp8__sparse_paged_mla as sp
from aiter.ops.triton.attention.sparse_mla import sparse_mla_fwd
tpl = {64: "glm52_fp8/sparse_paged_mla_r3/spm_tp8_split_template.py", 128: "glm52_fp8/sparse_paged_mla_r3/spm_tp8_policy_template.py",
       256: "glm52_fp8/sparse_paged_mla_r3/spm_tp8_policy_template.py", 8: "glm52_fp8/sparse_paged_mla_xl/decode_tp8_m8_x.py"}
for m in (8, 64, 128, 256):
    fn = tload.load(tpl[m], "sparse_paged_mla", "glm52_fp8_tp8_gfx950/sparse_paged_mla_templated.json")
    for pool, win in ((65536, None), (262144, None), (262144, 65536), (4194304, 65536), (131072, None)):
        sp._POOL = win or pool
        d = sp.make_inputs(f"tp8m{m}", 0, "cuda")
        cache = d["cache"]
        if win:
            big = torch.zeros(pool, 576, dtype=cache.dtype, device="cuda")
            big[pool // 2: pool // 2 + win] = cache
            cache = big
            slots = torch.where(d["slots"] >= 0, d["slots"] + pool // 2, -1).to(torch.int32)
        else:
            slots = d["slots"]
        q = d["query"]
        ip = torch.arange(0, (m + 1) * 2048, 2048, dtype=torch.int32, device="cuda")
        ks = torch.ones(1, device="cuda")
        t = lambda: fn(q, cache, slots, softmax_scale=0.0625, value_dim=512, quantize_query=True)
        b = lambda: sparse_mla_fwd(q, cache, ip, slots.reshape(-1), 0.0625, kv_scale=ks, has_invalid=True, dot_precision="fp8")
        r = core.measure({"T": [t], "B2": [b]}, 200, rounds=5, halves=1)
        print(f"m{m} pool {pool} window {win}: T {sorted(r['T'])[2]:.1f} us  B2 {sorted(r['B2'])[2]:.1f} us", flush=True)
