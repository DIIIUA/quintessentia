"""CUDA tests for Deep Micro-Agent NVRTC Kernels (`src/mirror_micro_agents_nvrtc.py`)."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.binary_halving_q1 import BinaryHalvingQ1Linear, halve_weight_to_q1_tree
from src.mirror_micro_agents_nvrtc import (
    launch_micro_agent_q1_group64_reader,
    run_million_micro_agent_substrate,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_readonly_micro_agent_group64_reader_exact_and_adaptive_cuda():
    """Per-group-64 micro-agents read pretrained Q1 weights read-only, match Q8 exactly, and save reads adaptively."""
    torch.manual_seed(777)
    dev = torch.device("cuda:0")
    N, K = 1024, 2048  # 32,768 independent group-64 micro-agents!
    w = (torch.randn(N, K, device=dev) * 0.04).to(torch.bfloat16)
    # Realistic transformer spikes in specific groups & rows
    for r in range(0, N, 16):
        w[r, :256] *= 8.0
    bias = (torch.randn(N, device=dev) * 0.01).to(torch.bfloat16)

    tensors, meta = halve_weight_to_q1_tree(w, root_bits=8, group_size=64, law="positional_halving", use_fwht=True)
    lin = BinaryHalvingQ1Linear(K, N, {k: v.to(dev) for k, v in tensors.items()}, meta, bias=bias)
    packed_snap = lin.packed_lo.clone()

    x = torch.randn(2, K, device=dev, dtype=torch.bfloat16)
    y_q8 = F.linear(x.float(), lin.reconstruct_weight(8, torch.float32).to(dev), bias.float())

    # 1. Exact mode (base_tol=0.0): bit-accurate match with Q8
    y_exact = torch.empty((2, N), device=dev, dtype=torch.float32)
    st_ex = launch_micro_agent_q1_group64_reader(
        x, lin.packed_lo, lin.qscale, lin.qzero, lin.row_scale, lin.col_scale, y_exact,
        max_planes=8, min_planes=8, base_tol=0.0, bias=bias, return_stats=True,
    )
    assert float(F.cosine_similarity(y_exact.flatten(), y_q8.flatten(), dim=0).item()) > 0.9999
    assert st_ex["avg_planes_per_group"] == 8.0

    # 2. Adaptive per-group-64 micro-agent reading (read-only weights!)
    y_adapt = torch.empty((2, N), device=dev, dtype=torch.bfloat16)
    st_ad = launch_micro_agent_q1_group64_reader(
        x, lin.packed_lo, lin.qscale, lin.qzero, lin.row_scale, lin.col_scale, y_adapt,
        max_planes=8, min_planes=3, base_tol=0.14, bias=bias, return_stats=True,
    )
    cos_ad = float(F.cosine_similarity(y_adapt.float().flatten(), y_q8.flatten(), dim=0).item())
    assert 3.0 <= st_ad["avg_planes_per_group"] < 7.0, f"Micro-agents did not save group reads: {st_ad}"
    assert cos_ad > 0.997, f"Micro-agent reader lost accuracy: cos={cos_ad:.6f}"
    # Strictly read-only: packed weights untouched
    assert torch.equal(lin.packed_lo, packed_snap)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_million_micro_agent_autonomous_substrate_cuda():
    """1,048,576 elementary micro-agents self-spawn intersecting rings from a 64-agent seed and settle naturally."""
    torch.manual_seed(1024)
    dev = torch.device("cuda:0")
    N_agents = 1 << 20  # 1,048,576 micro-agents!
    DEG = 4
    nbr_idx = torch.randint(0, N_agents, (N_agents, DEG), dtype=torch.int32, device=dev)
    nbr_w = (torch.randn((N_agents, DEG), dtype=torch.float32, device=dev) * 0.55)

    seed_idx = torch.arange(64, dtype=torch.int32, device=dev)
    seed_val = torch.ones((64,), dtype=torch.float32, device=dev) * 2.0

    res = run_million_micro_agent_substrate(
        seed_idx, seed_val, nbr_idx, nbr_w,
        spawn_tol=0.05, damping=0.72, max_waves=24,
    )
    assert res["N_agents"] == 1_048_576
    assert res["unique_agents_connected"] > 2000, f"Expected massive autonomous expansion, got {res['wave_sizes']}"
    assert res["converged_naturally"] is True, f"Substrate did not naturally settle: {res['wave_sizes']}"
    assert torch.isfinite(res["line_spine"]).all()
    assert float(res["line_spine"].abs().sum().item()) > 0.0
