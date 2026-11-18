import torch
from xbench import core
from xbench.cands import S12
from aiter.ops.triton.gated_delta_net import causal_conv1d_prefill as cp
from aiter.ops.triton.gated_delta_net import gated_delta_rule as gd
core.restore_triton()
cc = core.sglang_import("sglang.kernels.ops.mamba.causal_conv1d_triton")
fp = core.sglang_import("sglang.kernels.ops.attention.triton_gdn_fused_proj")
ch = core.sglang_import("sglang.kernels.ops.attention.fla.chunk")
gt = core.sglang_import("sglang.kernels.ops.attention.linear.fused_gdn_gating")
for seed in (1, 0):
    inp, _ = core.load_blob("S12", f"s1024_i1_s{seed}")
    T = 1024; D = 128
    qkvz = inp["projected_qkvz"]; x = qkvz[:, :3072].t()
    i1 = core.clone(inp); i2 = core.clone(inp)
    q, k, v = cp.causal_conv1d_split_qkv_triton_fn(x, i1["conv_weight"], None, i1["conv_state"], i1["cu_seqlens"], [T], 512, 2048,
                                                  cache_indices=i1["cache_indices"], has_initial_state=i1["has_initial_state"])
    y = cc.causal_conv1d_fn(qkvz[:, :3072].transpose(0, 1), i2["conv_weight"], None, activation="silu", conv_states=i2["conv_state"],
                            has_initial_state=i2["has_initial_state"], cache_indices=i2["cache_indices"], query_start_loc=i2["cu_seqlens"],
                            seq_lens_cpu=[T]).transpose(0, 1)[:T]
    print("seed", seed, "conv diff q,k,v", [(a - b).abs().max().item() for a, b in ((q, y[:, :512]), (k, y[:, 512:1024]), (v, y[:, 1024:]))])
    # gates
    ba = inp["projected_ba"]
    g1 = -torch.exp(inp["a_log"].float()) * torch.nn.functional.softplus(ba[:, 16:].float() + inp["dt_bias"].float())
    g2, b2 = gt.fused_gdn_gating(inp["a_log"], ba[:, 16:], ba[:, :16], inp["dt_bias"])
    print(" gate diff", (g1 - g2.view(T, 16)).abs().max().item(), "g min", g1.min().item())
    qs, ks, vs = fp.fused_qkv_split_gdn_prefill(y, 4, 4, 16, D, D, D)
    beta = torch.sigmoid(ba[:, :16].float())
    idx = inp["cache_indices"].long()
    h0 = inp["delta_state"][idx].transpose(-1, -2).contiguous()
    o1, _ = gd.chunk_gated_delta_rule_opt(qs, ks, vs, g1.view(1, T, 16), beta.view(1, T, 16), scale=D**-0.5, initial_state=h0,
                                          output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=inp["cu_seqlens"].long())
    st = inp["delta_state"].clone()
    o2 = ch.chunk_gated_delta_rule(q=qs, k=ks, v=vs, g=g2, beta=b2, initial_state=st, initial_state_indices=inp["cache_indices"],
                                   cu_seqlens=inp["cu_seqlens"], head_first=False, use_qk_l2norm_in_kernel=True, inplace_update=True)[0]
    d = (o1.float() - o2.float()).abs()
    print(" gdr o diff max", d.max().item(), "rows>1e-2:", (d.amax((0, 2, 3)) > 1e-2).sum().item(), "first bad t", (d.amax((0, 2, 3)) > 1e-2).nonzero()[:5].flatten().tolist())
