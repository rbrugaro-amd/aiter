"""A2 cross-check: AITER's own perftest (rotating args) on the S1 GEMM leg."""
import json, torch, aiter
from aiter.test_common import perftest
from aiter.ops.shuffle import shuffle_weight
from aiter import get_hip_quant

q = get_hip_quant(aiter.QuantType.per_1x128)


@perftest()
def run(x, w, xs, ws):
    return aiter.gemm_a8w8_blockscale_bpreshuffle(x, w, xs, ws, torch.bfloat16)


out = {}
for m in (1, 8, 64, 256, 1024, 16384):
    x = torch.randn(m, 6144, device="cuda", dtype=torch.bfloat16)
    w = shuffle_weight((torch.randn(2624, 6144, device="cuda") * 0.25).to(torch.float8_e4m3fn), (16, 16))
    ws = torch.rand(21, 48, device="cuda") * 2 ** -6
    xq, xs = q(x, quant_dtype=aiter.dtypes.fp8)
    xs = xs.t().contiguous().t()
    _, us = run(xq, w, xs, ws)
    out[m] = us
    print(f"aiter perftest m{m}: {us:.2f} us", flush=True)
json.dump(out, open("/scratch/xb/results/S1/xcheck_aiter_perftest.json", "w"))
