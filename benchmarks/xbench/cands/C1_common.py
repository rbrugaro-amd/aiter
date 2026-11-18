"""C1 shared pieces: the FP8 KV pool (4.5 M slots x 2 heads x 128, saved once per draw class,
shared by decode and prefill), template loading, SGLang imports, fp32 reference attention."""
import torch

from xbench import core

HQ, HK, D = 12, 2, 128
POOL = 4_500_000
SCALE = D ** -0.5
K_SCALE, V_SCALE = 0.6, 0.8
POOL_CAND = "C1pool"
_POOLS = {}


def pool_seed(seed):
    return 0 if seed < 100 else 100


def make_pool(pseed):
    """seed 0: K, V ~ 0.5 N(0,1) -> fp8.  seed 100 (cancellation, spec draw): slot p + P/2
    holds -V[p] and K[p] + 0.05 sigma."""
    g = torch.Generator(device="cuda").manual_seed(7_000_003 + pseed)
    rnd = lambda *s, sc=1.0: torch.randn(*s, generator=g, device="cuda", dtype=torch.float32) * sc  # noqa: E731
    f8 = torch.float8_e4m3fn
    if pseed == 0:
        kc = rnd(POOL, HK, D, sc=0.5).to(f8)
        vc = rnd(POOL, HK, D, sc=0.5).to(f8)
    else:
        half = POOL // 2
        k0 = rnd(half, HK, D, sc=0.5)
        k1 = k0 + rnd(half, HK, D, sc=0.05)
        kc = torch.cat([k0.to(f8), k1.to(f8)])
        del k0, k1
        v0 = rnd(half, HK, D, sc=2.0).to(f8)
        vc = torch.cat([v0, (-v0.float()).to(f8)]).contiguous()
    return dict(key_cache=kc, value_cache=vc)


def get_pool(pseed):
    if pseed not in _POOLS:
        name = f"pool_s{pseed}"
        if not core.have_blob(POOL_CAND, name):
            p = make_pool(pseed)
            core.save_blob(POOL_CAND, name, p)
            _POOLS[pseed] = p
        else:
            _POOLS[pseed], _ = core.load_blob(POOL_CAND, name)
    return _POOLS[pseed]


def with_pool(inp):
    p = get_pool(int(inp["pool_seed"].item()))
    m = dict(inp)
    m["key_cache"], m["value_cache"] = p["key_cache"], p["value_cache"]
    return m


def tpl(rel, entry, profile, source_file):
    from xbench import tload
    return tload.load(rel, entry, profile, source_file)


_AB = None


def ab():
    global _AB
    if _AB is None:
        _AB = core.sglang_import("sglang.srt.layers.attention.aiter_backend")
    return _AB


def ref_attention(q, kc, vc, slots, causal_prefix=None, ks=K_SCALE, vs=V_SCALE):
    """fp32 attention of q [T, HQ, D] over the slot list (GQA HQ/HK).  causal_prefix: if set,
    query t sees keys [0, causal_prefix + t]."""
    k = kc[slots].float() * ks          # [L, HK, D]
    v = vc[slots].float() * vs
    g = HQ // HK
    qf = q.float().view(q.shape[0], HK, g, D)
    out = torch.empty(q.shape[0], HK, g, D, device=q.device, dtype=torch.float32)
    for t0 in range(0, q.shape[0], 512):
        qq = qf[t0:t0 + 512]                                   # [t, HK, g, D]
        s = torch.einsum("thgd,lhd->thgl", qq, k) * SCALE
        if causal_prefix is not None:
            pos = torch.arange(t0, t0 + qq.shape[0], device=q.device) + causal_prefix
            mask = torch.arange(k.shape[0], device=q.device)[None, :] > pos[:, None]
            s = s.masked_fill(mask[:, None, None, :], float("-inf"))
        p = torch.softmax(s, -1)
        out[t0:t0 + 512] = torch.einsum("thgl,lhd->thgd", p, v)
    return out.view(q.shape[0], HQ, D)
