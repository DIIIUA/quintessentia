"""CUDA tests for the Unified 3D Fractal Cube Fabric (`src/fractal_cube_fabric_nvrtc.py`)."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.fractal_cube_fabric_nvrtc import (
    CONTACT_DIAG,
    CONTACT_HORIZ,
    CONTACT_VERT,
    Fractal3DCubeSwiGLUMLP,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_fractal_3d_cube_fabric_exact_and_rest_mode_cuda():
    """Verify real HORIZ/VERT/DIAG/CORNER90 contact graph, exact Q8 parity, and autonomous rest."""
    torch.manual_seed(2026)
    dev = torch.device("cuda:0")
    H, I = 512, 1024  # 3D cubes: gate (128x8=1024), up (1024), down (64x16=1024) -> 3,072 cubes
    w_g = (torch.randn(I, H, device=dev) * 0.04).to(torch.bfloat16)
    w_u = (torch.randn(I, H, device=dev) * 0.04).to(torch.bfloat16)
    w_d = (torch.randn(H, I, device=dev) * 0.04).to(torch.bfloat16)
    # Create realistic sparse MLP activation peaks + deadzone channels
    w_g[:128, :] *= 4.0
    w_g[768:, :] *= 0.15

    mlp = Fractal3DCubeSwiGLUMLP.from_weights(w_g, w_u, w_d, device=dev, root_bits=8)
    # Verify real physical contacts (HORIZ, VERT, DIAG) are populated — zero random edges
    ct = mlp.stage_gu.contact_type
    assert int((ct == CONTACT_HORIZ).sum().item()) > 0
    assert int((ct == CONTACT_VERT).sum().item()) > 0
    assert int((ct == CONTACT_DIAG).sum().item()) > 0

    g_snap = mlp.gate_lin.packed_lo.clone()
    d_snap = mlp.down_lin.packed_lo.clone()

    x = torch.randn(H, device=dev, dtype=torch.float32)
    y_q8_ref = F.linear(
        F.silu(F.linear(x, mlp.gate_lin.reconstruct_weight(8, torch.float32).to(dev)))
        * F.linear(x, mlp.up_lin.reconstruct_weight(8, torch.float32).to(dev)),
        mlp.down_lin.reconstruct_weight(8, torch.float32).to(dev),
    )

    # 1. Exact mode (base_tol=0, full coverage): bit-accurate parity with Q8 SwiGLU MLP!
    y_exact, st_ex = mlp.forward_fabric(x, max_planes=8, min_planes=8, base_tol=0.0, contact_tol=0.0)
    cos_ex = float(F.cosine_similarity(y_exact, y_q8_ref, dim=0).item())
    assert cos_ex > 0.9999, f"Exact 3D cube fabric mismatch vs Q8 MLP: cos={cos_ex:.6f}"
    assert st_ex["total_ops_vs_full_q8_pct"] == 100.0

    # 2. Autonomous rest + 90-degree corner deadzone skip + partial seed growth via HORIZ/VERT/DIAG!
    y_rest, st_rest = mlp.forward_fabric(
        x,
        max_planes=8,
        min_planes=3,
        base_tol=0.14,
        contact_tol=0.18,
        seed_input_quantile=0.25,
        corner90_deadzone_rel=0.20,
    )
    cos_rest = float(F.cosine_similarity(y_rest, y_q8_ref, dim=0).item())
    assert cos_rest > 0.99, f"Autonomous 3D fabric lost accuracy: cos={cos_rest:.6f}"
    assert st_rest["total_ops_vs_full_q8_pct"] < 80.0, f"Expected real op savings, got {st_rest}"
    assert st_rest["corner90_deadzone_skipped_pct"] > 0.0, "Expected CORNER90 deadzone skipping"
    sp = st_rest["stage_gate_up"]["spawns_by_contact"]
    assert sp["horiz_and_paired"] > 0 and sp["vert"] > 0 and sp["diag"] > 0 and sp["depth"] > 0
    assert st_rest["stage_gate_up"]["converged_naturally"] is True
    assert st_rest["stage_down"]["converged_naturally"] is True
    # Strictly read-only pretrained weights
    assert torch.equal(mlp.gate_lin.packed_lo, g_snap)
    assert torch.equal(mlp.down_lin.packed_lo, d_snap)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_moe_expert_relay_abc_invariants_and_parity_cuda():
    import threading
    from src.binary_halving_q1 import LazyMoEBinaryHalvingCache

    torch.manual_seed(77)
    dev = torch.device("cuda:0")
    h, inter, n_exp = 256, 128, 8
    gu_bank = [torch.randn(2 * inter, h, dtype=torch.bfloat16) * 0.04 for _ in range(n_exp)]
    dn_bank = [torch.randn(h, inter, dtype=torch.bfloat16) * 0.04 for _ in range(n_exp)]
    blueprint = {(0, e): (6 + (e % 4), 7 + (e % 3)) for e in range(n_exp)}

    x = torch.randn(1, h, device=dev, dtype=torch.bfloat16)
    idx = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], device=dev, dtype=torch.long)
    w = torch.tensor([[0.25, 0.18, 0.14, 0.12, 0.10, 0.08, 0.07, 0.06]], device=dev, dtype=torch.bfloat16)

    threads_seen = []

    def reader_ref(li, eid):
        return gu_bank[eid], dn_bank[eid]

    def reader_relay(li, eid):
        threads_seen.append(threading.current_thread().name)
        return gu_bank[eid], dn_bank[eid]

    c_ref = LazyMoEBinaryHalvingCache(
        reader_ref, num_layers=1, num_experts=n_exp, device=dev,
        store_cubes=16, root_bits=16, expert_cubes=blueprint,
        enable_expert_relay=False,
    )
    c_rel = LazyMoEBinaryHalvingCache(
        reader_relay, num_layers=1, num_experts=n_exp, device=dev,
        store_cubes=16, root_bits=16, expert_cubes=blueprint,
        enable_expert_relay=True,
    )
    try:
        y_ref = c_ref.forward_layer_moe(0, x, idx, w, F.silu)
        y_rel = c_rel.forward_layer_moe(0, x, idx, w, F.silu)
        torch.cuda.synchronize()
        assert torch.allclose(y_ref.float(), y_rel.float(), atol=1e-4, rtol=1e-4)
        st = c_rel.stats()["expert_relay"]
        assert st is not None
        assert st["rotations"] == n_exp
        assert st["zero_copy_adoptions"] == n_exp
        assert st["handoffs"] == n_exp
        assert st["transitions"]["B->A"] == 0
        assert st["standby_bytes"] == 0
        assert st["gpu_prepares"] == n_exp
        assert len(threads_seen) == n_exp
        assert all(t.startswith("aircolibri-moe-B") for t in threads_seen)
    finally:
        c_ref.close()
        c_rel.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_moe_expert_relay_plus_mirror_uni_network_rest_skips_cold_tail_cuda():
    from src.binary_halving_q1 import LazyMoEBinaryHalvingCache

    torch.manual_seed(109)
    dev = torch.device("cuda:0")
    h, inter, n_exp = 256, 128, 8
    gu_bank = [torch.randn(2 * inter, h, dtype=torch.bfloat16) * 0.04 for _ in range(n_exp)]
    dn_bank = [torch.randn(h, inter, dtype=torch.bfloat16) * 0.04 for _ in range(n_exp)]
    blueprint = {(0, e): (7, 7) for e in range(n_exp)}

    x = torch.randn(1, h, device=dev, dtype=torch.bfloat16)
    idx = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], device=dev, dtype=torch.long)

    # Case 1: Concentrated router -> Mirror x UNI rest MUST avoid waking cold disk tail experts
    w_sharp = torch.tensor([[0.58, 0.14, 0.09, 0.05, 0.045, 0.04, 0.03, 0.025]], device=dev, dtype=torch.bfloat16)
    disk_reads_sharp = []
    c_full = LazyMoEBinaryHalvingCache(
        lambda li, e: (gu_bank[e], dn_bank[e]), num_layers=1, num_experts=n_exp,
        device=dev, store_cubes=16, root_bits=16, expert_cubes=blueprint,
    )
    c_rest = LazyMoEBinaryHalvingCache(
        lambda li, e: (disk_reads_sharp.append(e) or gu_bank[e], dn_bank[e]),
        num_layers=1, num_experts=n_exp, device=dev, store_cubes=16, root_bits=16,
        expert_cubes=blueprint, enable_expert_relay=True, expert_rest_tol=0.28, expert_rest_min=3,
    )
    try:
        y_full = c_full.forward_layer_moe(0, x, idx, w_sharp, F.silu)
        y_rest = c_rest.forward_layer_moe(0, x, idx, w_sharp, F.silu)
        torch.cuda.synchronize()
        cos = float(F.cosine_similarity(y_rest.float(), y_full.float()).item())
        st = c_rest.stats()
        assert len(disk_reads_sharp) < n_exp, f"Expected cold disk experts to rest, read={disk_reads_sharp}"
        assert st["cold_experts_rested"] > 0 and st["cold_disk_reads_avoided"] > 0
        assert cos > 0.96, f"Unexpected quality drop under expert rest: cos={cos}"
        assert st["expert_relay"]["transitions"]["B->A"] == 0
        assert st["expert_relay"]["standby_bytes"] == 0
    finally:
        c_full.close()
        c_rest.close()

    # Case 2: Flat router -> network detects lack of rest and wakes all needed experts
    w_flat = torch.full((1, n_exp), 1.0 / n_exp, device=dev, dtype=torch.bfloat16)
    disk_reads_flat = []
    c_flat = LazyMoEBinaryHalvingCache(
        lambda li, e: (disk_reads_flat.append(e) or gu_bank[e], dn_bank[e]),
        num_layers=1, num_experts=n_exp, device=dev, store_cubes=16, root_bits=16,
        expert_cubes=blueprint, enable_expert_relay=True, expert_rest_tol=0.24, expert_rest_min=3,
    )
    try:
        y_flat = c_flat.forward_layer_moe(0, x, idx, w_flat, F.silu)
        torch.cuda.synchronize()
        assert len(disk_reads_flat) >= 6, f"Flat router should wake most/all experts, woke={len(disk_reads_flat)}"
        assert c_flat.stats()["expert_relay"]["transitions"]["B->A"] == 0
        assert c_flat.stats()["expert_relay"]["standby_bytes"] == 0
    finally:
        c_flat.close()

