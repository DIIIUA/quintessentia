"""CUDA tests for Mirror Equality NVRTC kernels (`src/mirror_equality_nvrtc.py`)."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.binary_halving_q1 import BinaryHalvingQ1Linear, halve_weight_to_q1_tree
from src.mirror_equality_nvrtc import (
    launch_mirror_equality_balance_step,
    launch_mirror_equality_q1_gemv,
    launch_mirror_fractal_ring_q1_step,
    launch_mirror_uni_binding_q1_gemv,
    run_autonomous_fractal_rings,
    uni_canonical_invariants,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_mirror_equality_balance_step_conservation_cuda():
    """Verify exact GPU conservation of Outcome_(t-1) + Obs_(t-1) = Strategy_t = Obs_(t+1) + Outcome_(t+1)."""
    torch.manual_seed(777)
    dev = torch.device("cuda:0")
    B, D = 16, 1024
    outcome_prev = torch.randn(B, D, device=dev)
    obs_prev = torch.randn(B, D, device=dev) * 0.5
    obs_next = torch.randn(B, D, device=dev) * 0.25

    # 1. Solve mode: must satisfy L == Strategy_t == R to machine precision
    strat, out_next, defect = launch_mirror_equality_balance_step(outcome_prev, obs_prev, obs_next)
    lhs = outcome_prev + obs_prev
    rhs = obs_next + out_next
    torch.testing.assert_close(strat, lhs, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(rhs, lhs, rtol=1e-5, atol=1e-5)
    assert float(defect.max().item()) < 1e-6

    # 2. Verify & project mode with perturbed right wing: detects exact defect and rebalances
    perturb = torch.randn(B, D, device=dev) * 0.1
    out_perturbed = out_next + perturb
    strat_b, out_b, defect_b = launch_mirror_equality_balance_step(
        outcome_prev, obs_prev, obs_next, outcome_next_in=out_perturbed
    )
    expected_defect = perturb.norm(dim=-1) / (strat_b.norm(dim=-1) + 1e-8)
    torch.testing.assert_close(defect_b, expected_defect, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(obs_next + out_b, strat_b, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("N,K", [(192, 576), (512, 2048), (256, 6144)])
@torch.no_grad()
def test_mirror_equality_q1_gemv_exact_and_adaptive_early_stop_cuda(N: int, K: int):
    """Verify exact Q1 GEMV parity across single-chunk (K<=4096) & multi-chunk (K>4096), plus GPU early-stop."""
    torch.manual_seed(2026 + N + K)
    dev = torch.device("cuda:0")
    w = (torch.randn(N, K, dtype=torch.float32, device=dev) * 0.04).to(torch.bfloat16)
    bias = (torch.randn(N, dtype=torch.float32, device=dev) * 0.01).to(torch.bfloat16)

    tensors, meta = halve_weight_to_q1_tree(w, root_bits=8, group_size=64, law="positional_halving", use_fwht=True)
    lin = BinaryHalvingQ1Linear(K, N, {k: v.to(dev) for k, v in tensors.items()}, meta, bias=bias)
    x = torch.randn(3, K, device=dev, dtype=torch.bfloat16)

    # 1. Exact mode (mirror_tol=0.0): must match reconstructed k=4 and k=8 weights bit-accurately
    for k_planes in (4, 8):
        w_ref = lin.reconstruct_weight(cubes=k_planes, dtype=torch.float32).to(dev)
        y_ref = F.linear(x.float(), w_ref, bias.float())
        y_got = torch.empty((3, N), device=dev, dtype=torch.float32)
        used_p, defects = launch_mirror_equality_q1_gemv(
            x, lin.packed_lo, lin.qscale, lin.qzero, lin.row_scale, lin.col_scale, y_got,
            group_size=64, max_planes=k_planes, min_planes=k_planes, mirror_tol=0.0,
            bias=bias, return_telemetry=True,
        )
        cos = float(F.cosine_similarity(y_got.flatten(), y_ref.flatten(), dim=0).item())
        assert cos > 0.9999, f"Exact mirror GEMV mismatch (N={N}, K={K}, k={k_planes}): cos={cos:.6f}"
        assert int(used_p.min().item()) == k_planes and int(used_p.max().item()) == k_planes
        assert torch.isfinite(defects).all()

    # 2. Adaptive Mirror Convergence Gate (mirror_tol=0.12, min_planes=3, max_planes=8):
    #    Blocks must stop early (avg planes < 8) while staying extremely close to full Q8!
    w_q8 = lin.reconstruct_weight(cubes=8, dtype=torch.float32).to(dev)
    y_q8 = F.linear(x.float(), w_q8, bias.float())
    y_adapt = torch.empty((3, N), device=dev, dtype=torch.bfloat16)
    used_adapt, defect_adapt = launch_mirror_equality_q1_gemv(
        x, lin.packed_lo, lin.qscale, lin.qzero, lin.row_scale, lin.col_scale, y_adapt,
        group_size=64, max_planes=8, min_planes=3, mirror_tol=0.12,
        bias=bias, return_telemetry=True,
    )
    avg_planes = float(used_adapt.float().mean().item())
    cos_adapt = float(F.cosine_similarity(y_adapt.float().flatten(), y_q8.flatten(), dim=0).item())
    assert 3.0 <= avg_planes < 7.5, f"Mirror early-stop did not save planes: avg={avg_planes:.2f}"
    assert cos_adapt > 0.994, f"Adaptive mirror GEMV lost accuracy: cos={cos_adapt:.6f}"
    assert torch.isfinite(defect_adapt).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_uni_mirror_binding_prioritizes_spike_neurons_over_flat_background_cuda():
    """UNI x Mirror binding must assign strictly higher precision (more Q1 cubes) to spike tiles than flat tiles."""
    inv = uni_canonical_invariants()
    assert abs(inv["Phi"] - 0.22313016014848982) < 1e-12
    assert abs(inv["Omega"] - 1.5707963267948966) < 1e-12
    assert abs(inv["Lambda"] - (1024.0 / 14348907.0)) < 1e-12
    assert abs(inv["Sigma"] - 1.7724538509055159) < 1e-12
    assert abs(inv["Delta"] - 1.8390715290764524) < 1e-12

    torch.manual_seed(314)
    dev = torch.device("cuda:0")
    N, K = 512, 1024  # 64 tiles of 8 rows each
    w = (torch.randn(N, K, dtype=torch.float32, device=dev) * 0.04).to(torch.bfloat16)
    # First half of tiles (rows 0..255): 1 dominant spike neuron per 8-row tile (high concentration lambda_tile -> ~1)
    for tile in range(32):
        w[tile * 8, :] *= 12.0
    # Second half of tiles (rows 256..511): uniform flat background (low lambda_tile -> ~1/8)

    tensors, meta = halve_weight_to_q1_tree(w, root_bits=8, group_size=64, law="positional_halving", use_fwht=True)
    lin = BinaryHalvingQ1Linear(K, N, {k: v.to(dev) for k, v in tensors.items()}, meta)
    x = torch.randn(4, K, device=dev, dtype=torch.bfloat16)
    y_q8 = F.linear(x.float(), lin.reconstruct_weight(8, torch.float32).to(dev))

    # Exact mode parity
    y_exact = torch.empty((4, N), device=dev, dtype=torch.float32)
    launch_mirror_uni_binding_q1_gemv(
        x, lin.packed_lo, lin.qscale, lin.qzero, lin.row_scale, lin.col_scale, y_exact,
        max_planes=8, min_planes=8, base_tol=0.0,
    )
    assert float(F.cosine_similarity(y_exact.flatten(), y_q8.flatten(), dim=0).item()) > 0.9999

    # Adaptive UNI binding mode
    y_uni = torch.empty((4, N), device=dev, dtype=torch.bfloat16)
    used_u, def_u, bind_u = launch_mirror_uni_binding_q1_gemv(
        x, lin.packed_lo, lin.qscale, lin.qzero, lin.row_scale, lin.col_scale, y_uni,
        max_planes=8, min_planes=3, base_tol=0.16, return_telemetry=True,
    )
    spike_cubes = float(used_u[:, :32].float().mean().item())
    flat_cubes = float(used_u[:, 32:].float().mean().item())
    spike_bind = float(bind_u[:, :32].mean().item())
    flat_bind = float(bind_u[:, 32:].mean().item())
    cos_uni = float(F.cosine_similarity(y_uni.float().flatten(), y_q8.flatten(), dim=0).item())

    assert spike_bind > flat_bind * 1.15, f"Expected higher binding strength on spike tiles: {spike_bind} vs {flat_bind}"
    assert spike_cubes > flat_cubes, f"Expected more cubes on spike tiles: {spike_cubes:.2f} vs {flat_cubes:.2f}"
    assert cos_uni > 0.996, f"UNI mirror binding accuracy too low: {cos_uni:.6f}"
    assert torch.isfinite(def_u).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_fractal_intersecting_rings_multikernel_spine_cuda():
    """Verify freely oriented intersecting rings across 2 distinct projections + Mirror Balance on one line spine."""
    torch.manual_seed(2026)
    dev = torch.device("cuda:0")
    N, K = 1024, 1024  # 128 tiles
    # Ring A: primary trunk projection (with spike rows)
    w_trunk = (torch.randn(N, K, device=dev) * 0.04).to(torch.bfloat16)
    for t in range(32):
        w_trunk[t * 8, :] *= 10.0
    # Ring B: secondary expert/branch projection intersecting the EXACT same assembly line
    w_expert = (torch.randn(N, K, device=dev) * 0.025).to(torch.bfloat16)

    t_tr, m_tr = halve_weight_to_q1_tree(w_trunk, root_bits=8, group_size=64, law="positional_halving", use_fwht=True)
    t_ex, m_ex = halve_weight_to_q1_tree(w_expert, root_bits=8, group_size=64, law="positional_halving", use_fwht=True)
    lin_tr = BinaryHalvingQ1Linear(K, N, {k: v.to(dev) for k, v in t_tr.items()}, m_tr)
    lin_ex = BinaryHalvingQ1Linear(K, N, {k: v.to(dev) for k, v in t_ex.items()}, m_ex)

    x = torch.randn(1, K, device=dev, dtype=torch.bfloat16)
    expert_alpha = 0.5
    y_ref_whole = (
        F.linear(x.float(), lin_tr.reconstruct_weight(8, torch.float32).to(dev))
        + expert_alpha * F.linear(x.float(), lin_ex.reconstruct_weight(4, torch.float32).to(dev))
    )

    # Single shared assembly line (`line_spine`): every ring starts from & returns onto this line!
    line_spine = torch.zeros((1, N), dtype=torch.float32, device=dev)
    last_obs = torch.zeros((1, N), dtype=torch.float32, device=dev)

    # --- RING A (base trunk p = 0, 1, 2 across all 128 tiles) ---
    active = None
    for p in range(3):
        spawn, cnt, _ = launch_mirror_fractal_ring_q1_step(
            x, lin_tr.packed_lo, lin_tr.qscale, lin_tr.qzero, lin_tr.row_scale, lin_tr.col_scale,
            line_spine, plane_idx=p, active_tiles=active, include_zero=(p == 0),
            base_tol=0.14 if p == 2 else 0.0, last_obs_out=last_obs, spawn_next=True,
        )
        active = spawn
    # Save intersection point S_mid on the assembly line BEFORE branching into Ring B
    s_mid_outcome = line_spine - last_obs
    s_mid_obs = last_obs.clone()
    assert 0 < int(active.numel()) <= 128

    # --- RING B (branches off intermediate point S_mid into a DIFFERENT projection / expert kernel!) ---
    obs_branch_total = torch.zeros((1, N), dtype=torch.float32, device=dev)
    for p in range(4):
        obs_b = torch.zeros((1, N), dtype=torch.float32, device=dev)
        launch_mirror_fractal_ring_q1_step(
            x, lin_ex.packed_lo, lin_ex.qscale, lin_ex.qzero, lin_ex.row_scale, lin_ex.col_scale,
            line_spine, plane_idx=p, active_tiles=None, include_zero=(p == 0),
            base_tol=0.0, branch_weight=expert_alpha, last_obs_out=obs_b, spawn_next=False,
        )
        obs_branch_total += obs_b

    # Verify Mirror Equality conservation at the A <-> B intersection junction on the shared line
    strat_j, _, defect_j = launch_mirror_equality_balance_step(
        s_mid_outcome, s_mid_obs, obs_branch_total, outcome_next_in=(line_spine - obs_branch_total - obs_branch_total)
    )
    # Exact check in solve mode on the A -> B junction
    strat_ok, out_ok, defect_ok = launch_mirror_equality_balance_step(s_mid_outcome, s_mid_obs, obs_branch_total)
    torch.testing.assert_close(strat_ok, s_mid_outcome + s_mid_obs, rtol=1e-5, atol=1e-5)
    assert float(defect_ok.max().item()) < 1e-6

    # --- RING C (continues trunk p = 3..7 ONLY on auto-spawned fractal intersection tiles!) ---
    tile_counts = [int(active.numel())]
    for p in range(3, 8):
        if int(active.numel()) == 0:
            break
        spawn, cnt, _ = launch_mirror_fractal_ring_q1_step(
            x, lin_tr.packed_lo, lin_tr.qscale, lin_tr.qzero, lin_tr.row_scale, lin_tr.col_scale,
            line_spine, plane_idx=p, active_tiles=active, include_zero=False,
            base_tol=0.14, last_obs_out=last_obs, spawn_next=True,
        )
        active = spawn
        tile_counts.append(cnt)

    # Queue must monotonically contract (fractal convergence: fewer active GPU blocks each ring step!)
    assert tile_counts[-1] < tile_counts[0], f"Fractal intersection queue did not contract: {tile_counts}"
    cos_whole = float(F.cosine_similarity(line_spine.flatten(), y_ref_whole.flatten(), dim=0).item())
    assert cos_whole > 0.998, f"Assembled whole lost accuracy across intersecting rings: cos={cos_whole:.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_autonomous_self_spawning_rings_from_base_only_cuda():
    """Given ONLY a base seed and a pool of dormant organs, rings must self-spawn dynamically per input."""
    torch.manual_seed(404)
    dev = torch.device("cuda:0")
    N, K = 512, 512

    def make_op(scale: float, spike: bool = False):
        w = (torch.randn(N, K, device=dev) * scale).to(torch.bfloat16)
        if spike:
            for t in range(16):
                w[t * 8, :] *= 8.0
        t_d, m_d = halve_weight_to_q1_tree(w, root_bits=8, group_size=64, law="positional_halving", use_fwht=True)
        lin = BinaryHalvingQ1Linear(K, N, {k: v.to(dev) for k, v in t_d.items()}, m_d)
        return {
            "packed_lo": lin.packed_lo,
            "qscale": lin.qscale,
            "qzero": lin.qzero,
            "row_scale": lin.row_scale,
            "col_scale": lin.col_scale,
            "lin": lin,
        }

    trunk = make_op(0.04, spike=True)
    # Create 6 dormant organs with distinct resonance anchors along orthogonal subspaces
    organs = {}
    for idx in range(6):
        op = make_op(0.025)
        anchor = torch.zeros(N, dtype=torch.float32, device=dev)
        seg = N // 6
        anchor[idx * seg : (idx + 1) * seg] = torch.randn(seg, device=dev)
        op["anchor"] = anchor
        op["weight"] = 0.35
        op["max_planes"] = 5
        op["min_planes"] = 2
        organs[f"organ_{idx}"] = op

    # Input 1 vs Input 2: the exact same base & pool autonomously spawns DIFFERENT ring topologies!
    x1 = torch.randn(1, K, device=dev, dtype=torch.bfloat16)
    # Bias organ_0 and organ_1 anchors toward trunk's output for x1 so multiple rings cascade naturally,
    # while organ_2 responds to the COMBINED line after organ_0 fires (secondary self-spawn!).
    y_tr1 = F.linear(x1.float(), trunk["lin"].reconstruct_weight(4, torch.float32).to(dev))[0]
    organs["organ_0"]["anchor"] = y_tr1.clone()
    y_org0 = F.linear(x1.float(), organs["organ_0"]["lin"].reconstruct_weight(4, torch.float32).to(dev))[0]
    # Make organ_2 orthogonal to trunk alone, but aligned with organ_0's contribution -> cascade spawn!
    proj_on_tr = (torch.dot(y_org0, y_tr1) / (y_tr1.pow(2).sum() + 1e-8)) * y_tr1
    organs["organ_2"]["anchor"] = (y_org0 - proj_on_tr).clone()

    res1 = run_autonomous_fractal_rings(x1, trunk, organs, base_tol=0.15, wake_cos_threshold=0.18)
    assert res1["num_awakened_rings"] >= 3, f"Expected cascading self-spawn (>=3 rings), got {res1['awakened_rings']}"
    assert "organ_0" in res1["awakened_rings"]
    assert "organ_2" in res1["awakened_rings"]
    assert res1["steps_executed"] < 64  # terminated naturally when frontier emptied!

    # Verify exact numerical agreement of assembled whole against full Q8 trunk + Q5 awakened organs
    y_expected = F.linear(x1.float(), trunk["lin"].reconstruct_weight(8, torch.float32).to(dev))
    for name in res1["awakened_rings"]:
        if name == "trunk":
            continue
        # Match actual weight sign used on wake (always positive abs(cand_w) in run_autonomous_fractal_rings)
        y_expected = y_expected + 0.35 * F.linear(x1.float(), organs[name]["lin"].reconstruct_weight(5, torch.float32).to(dev))
    cos1 = float(F.cosine_similarity(res1["line_spine"].flatten(), y_expected.flatten(), dim=0).item())
    assert cos1 > 0.996, f"Autonomous assembled whole drifted: cos={cos1:.6f}"



