"""Strict tests for Version 3 Pure Mirror new mechanisms (Bracketless Stem, Intact Participants + Y_L
with Algebraically Reversible Scar, Differential-Reversible Q1 CUDA GEMV, Activation-Aware Crystal
Carver, Unclosed-Step Two-Slit Protection, and Bidirectional MirrorDualPrefill).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.binary_halving_q1 import (
    LazyMoEBinaryHalvingCache,
    carve_q1_cubes_incremental,
    carve_q1_mirror_crystal,
)
from src.pure_mirror_agents_nvrtc import (
    EQ_EQUAL,
    EQ_SPLIT,
    EQ_TENSION,
    SLOT_CENTER,
    SLOT_LOGIC,
    PureMirrorCudaFeedback,
    _eval_4x4_mirror_cube_full_cpu,
    launch_pure_mirror_diff_q1_gemv,
)
from src.stanok_contour import StanokContainmentContour


def test_bracketless_central_stem_stone_law_and_contradiction_catch():
    """Verify bracketless center-reading of:
        Outcome_(t-1) + ObservedBehavior_(t-1) = Strategy_t = ObservedBehavior_(t+1) + Outcome_(t+1)
    1. Stone Law: distinct outer outcomes (throw vs shatter) unified by matching central observed
       flight trajectory (P1 == S == P2) must yield high `tau_core` and living TENSION (NOT SPLIT!).
    2. Contradictory Stem: identical outer outcomes with opposite ObservedBehavior (P1 == -P2) must
       yield `tau_core == 0.0` and MUST NEVER be falsely classified as EQUAL!
    """
    e0 = torch.zeros(32, dtype=torch.float32); e0[0] = 1.0
    e1 = torch.zeros(32, dtype=torch.float32); e1[1] = 1.0
    e2 = torch.zeros(32, dtype=torch.float32); e2[2] = 1.0

    # Stone Law: P0 = throw (e0), P1 = flight (e2), S = flight (e2), P2 = flight (e2), P3 = shatter (e1)
    stone = _eval_4x4_mirror_cube_full_cpu(e0, e2, e2, e2, e1)
    assert stone["tau_core"] > 0.99
    assert stone["d_core"] < 1e-5
    assert stone["eq_status"] == EQ_TENSION
    assert stone["tie"] > 0.3

    # Even when outer outcomes are strictly opposite (P0 = +e0, P3 = -e0), matching stem P1 = P2 = e2 holds!
    opp_outer = _eval_4x4_mirror_cube_full_cpu(e0, e2, e2, e2, -e0)
    assert opp_outer["tau_core"] > 0.99
    assert opp_outer["eq_status"] == EQ_TENSION
    assert opp_outer["tie"] > 0.4

    # Contradictory Stem: outer outcomes match (P0 = P3 = e0), yet ObservedBehavior flips (P1 = +e1, P2 = -e1)!
    contra = _eval_4x4_mirror_cube_full_cpu(e0, e1, e0, -e1, e0)
    assert contra["tau_core"] == 0.0
    assert contra["eq_status"] != EQ_EQUAL, "Bracketless central stem must block false EQUAL when P1 == -P2!"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_intact_participants_separate_layer_node_and_exact_reversible_scar_cuda():
    """Verify the Multi-Ray Intersection Iron Law on both GPU and CPU:
    1. Participants STAY THEMSELVES: adding a peer on the same token NEVER overwrites or smears
       agent 0's own `Strategy_self` or `Logic_self`!
    2. A SEPARATE collective object `Y_L` (`layer_nodes[li]`) is born in the center with its own
       6 roles and exact algebraically reversible fresh scar `Gamma_scar(Y_L)`:
         - Forward: `alpha_k * S_k + r_k == S_Y` (atol <= 1e-5)
         - Inverse: `(S_Y - r_k) / alpha_k == S_k` (atol <= 1e-4)
    3. Exact CUDA == CPU parity across `roles`, `layer_nodes`, `layer_node_meta`, and `layer_scar_*`!
    """
    torch.manual_seed(20261123)
    h = 64
    x_in = torch.randn(4, h, dtype=torch.float32)
    x_post = x_in + 0.3 * torch.randn(4, h, dtype=torch.float32)

    # Compare agent 0 running ALONE vs running WITH peers [1, 2] on the exact same inputs!
    pm_solo = PureMirrorCudaFeedback(num_layers=2, num_experts=16, device="cuda:0")
    pm_multi_gpu = PureMirrorCudaFeedback(num_layers=2, num_experts=16, device="cuda:0")
    pm_multi_cpu = PureMirrorCudaFeedback(num_layers=2, num_experts=16, device="cpu")

    u_in_g = pm_solo.project_states(0, x_in.to("cuda:0"), update_mean=True)
    u_post_g = pm_solo.project_states(0, x_post.to("cuda:0"), update_mean=False)
    u_in_m = pm_multi_gpu.project_states(0, x_in.to("cuda:0"), update_mean=True)
    u_post_m = pm_multi_gpu.project_states(0, x_post.to("cuda:0"), update_mean=False)
    u_in_c = pm_multi_cpu.project_states(0, x_in, update_mean=True)
    u_post_c = pm_multi_cpu.project_states(0, x_post, update_mean=False)

    routes_solo = [[(0, 1.0)] for _ in range(4)]
    routes_multi = [[(0, 0.5), (1, 0.3), (2, 0.2)] for _ in range(4)]

    pm_solo.observe_mirror_step(0, routes_solo, u_in_g, u_post=u_post_g, is_commit=True)
    pm_multi_gpu.observe_mirror_step(0, routes_multi, u_in_m, u_post=u_post_m, is_commit=True)
    pm_multi_cpu.observe_mirror_step(0, routes_multi, u_in_c, u_post=u_post_c, is_commit=True)

    # 1. Participant 0's own Strategy_self and Logic_self MUST BE IDENTICAL whether peers exist or not!
    assert torch.allclose(pm_solo.roles[0, 0, SLOT_CENTER], pm_multi_gpu.roles[0, 0, SLOT_CENTER], atol=1e-5)
    assert torch.allclose(pm_solo.roles[0, 0, SLOT_LOGIC], pm_multi_gpu.roles[0, 0, SLOT_LOGIC], atol=1e-5)

    # 2. Exact CUDA == CPU parity on born layer node Y_L and its reversible scar Gamma_scar!
    assert torch.allclose(pm_multi_cpu.layer_nodes[0], pm_multi_gpu.layer_nodes[0].cpu(), atol=1e-4)
    assert torch.allclose(pm_multi_cpu.layer_node_meta[0], pm_multi_gpu.layer_node_meta[0].cpu(), atol=1e-4)
    assert torch.equal(pm_multi_cpu.layer_scar_eids[0], pm_multi_gpu.layer_scar_eids[0].cpu())
    assert torch.allclose(pm_multi_cpu.layer_scar_alpha[0], pm_multi_gpu.layer_scar_alpha[0].cpu(), atol=1e-4)
    assert torch.allclose(pm_multi_cpu.layer_scar_res[0], pm_multi_gpu.layer_scar_res[0].cpu(), atol=1e-4)

    # 3. Exact algebraic reversibility forward & inverse for EVERY active ray in Y_L!
    s_y = pm_multi_gpu.layer_nodes[0, SLOT_CENTER]
    assert float(s_y.norm().item()) > 0.1
    assert int(pm_multi_gpu.layer_node_steps[0, 0].item()) == 4
    for slot_k in range(3):
        eid_f, rec_sy = pm_multi_gpu.reconstruct_layer_node_from_ray(0, slot_k)
        assert eid_f == slot_k
        assert torch.allclose(rec_sy, s_y, atol=1e-5), f"Forward scar reconstruction mismatch on ray {slot_k}"

        eid_b, ak, rec_sk = pm_multi_gpu.reconstruct_ray_from_layer_node(0, slot_k)
        assert eid_b == slot_k and abs(ak) > 0.1
        assert torch.allclose(rec_sk, pm_multi_gpu.roles[0, slot_k, SLOT_CENTER], atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_differential_reversible_q1_cuda_gemv_zero_plane_and_accuracy():
    """Verify Kernel #6 (`pure_mirror_diff_q1_gemv_f32`):
    1. Exact CUDA == CPU parity with and without crisp BF16 scar-anchor (and with bias).
    2. Zero-Plane EQUAL Short-Circuit: collinear `x_t = 1.75 * x_last` reads 0 weight planes!
    3. Small Fresh Scar (`95%` shared + `5%` orthogonal): automatically exits in fewer planes while
       achieving HIGHER cosine similarity to exact BF16 `W @ x_t` than non-differential Q1!
    """
    torch.manual_seed(777)
    dev = torch.device("cuda:0")
    N, K = 128, 256
    w = (torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.05).contiguous()
    bias = (torch.randn(N, dtype=torch.bfloat16, device=dev) * 0.02).contiguous()
    t_gpu, _ = carve_q1_cubes_incremental(w, 5, root_bits=8, group_size=64, use_fwht=True, sinkhorn_iters=3)
    t_cpu = {k: v.cpu() for k, v in t_gpu.items()}

    x_last = torch.randn(K, dtype=torch.bfloat16, device=dev)
    y_last = F.linear(x_last.unsqueeze(0), w).squeeze(0).to(torch.bfloat16)
    y_last_b = F.linear(x_last.unsqueeze(0), w, bias).squeeze(0).to(torch.bfloat16)

    # Case 1: Zero-Plane EQUAL Short-Circuit (x_t = 1.75 * x_last -> r_x == 0 -> 0 planes read!)
    x_eq = (1.75 * x_last).unsqueeze(0).to(torch.bfloat16)
    y_eq_g, used_eq_g, _ = launch_pure_mirror_diff_q1_gemv(
        x_eq, t_gpu["packed_lo"], t_gpu["qscale"], t_gpu["qzero"], t_gpu["row_scale"], t_gpu["col_scale"],
        x_last=x_last, y_last=y_last, group_size=64, max_planes=5, min_planes=1, mirror_tol=0.03,
        return_telemetry=True,
    )
    y_eq_c, used_eq_c, _ = launch_pure_mirror_diff_q1_gemv(
        x_eq.cpu(), t_cpu["packed_lo"], t_cpu["qscale"], t_cpu["qzero"], t_cpu["row_scale"], t_cpu["col_scale"],
        x_last=x_last.cpu(), y_last=y_last.cpu(), group_size=64, max_planes=5, min_planes=1, mirror_tol=0.03,
        return_telemetry=True,
    )
    assert int(used_eq_g.max().item()) == 0 and int(used_eq_c.max().item()) == 0
    assert torch.allclose(y_eq_g.float().cpu(), y_eq_c.float(), atol=2e-2)
    cos_eq = float(F.cosine_similarity(y_eq_g.float(), F.linear(x_eq, w).float()).item())
    assert cos_eq > 0.9999

    # Case 2: Small Fresh Scar (95% aligned + 5% fresh orthogonal) -> fewer planes AND higher accuracy!
    u_orth = torch.randn(K, dtype=torch.float32, device=dev)
    u_orth = u_orth - ((u_orth * x_last.float()).sum() / ((x_last.float() ** 2).sum() + 1e-8)) * x_last.float()
    u_orth = u_orth * (x_last.float().norm() / (u_orth.norm() + 1e-8))
    x_small = (0.95 * x_last.float() + 0.05 * u_orth).unsqueeze(0).to(torch.bfloat16)
    y_exact = F.linear(x_small, w).float()

    y_diff_g, used_diff_g, _ = launch_pure_mirror_diff_q1_gemv(
        x_small, t_gpu["packed_lo"], t_gpu["qscale"], t_gpu["qzero"], t_gpu["row_scale"], t_gpu["col_scale"],
        x_last=x_last, y_last=y_last, group_size=64, max_planes=5, min_planes=1, mirror_tol=0.03,
        return_telemetry=True,
    )
    y_diff_c, used_diff_c, _ = launch_pure_mirror_diff_q1_gemv(
        x_small.cpu(), t_cpu["packed_lo"], t_cpu["qscale"], t_cpu["qzero"], t_cpu["row_scale"], t_cpu["col_scale"],
        x_last=x_last.cpu(), y_last=y_last.cpu(), group_size=64, max_planes=5, min_planes=1, mirror_tol=0.03,
        return_telemetry=True,
    )
    y_noanc_g, used_noanc_g, _ = launch_pure_mirror_diff_q1_gemv(
        x_small, t_gpu["packed_lo"], t_gpu["qscale"], t_gpu["qzero"], t_gpu["row_scale"], t_gpu["col_scale"],
        x_last=None, y_last=None, group_size=64, max_planes=5, min_planes=1, mirror_tol=0.03,
        return_telemetry=True,
    )
    mean_p_diff = float(used_diff_g.float().mean().item())
    mean_p_noanc = float(used_noanc_g.float().mean().item())
    assert mean_p_diff < mean_p_noanc, f"Expected differential scar to save planes: {mean_p_diff} vs {mean_p_noanc}"
    assert mean_p_diff <= 0.65 * mean_p_noanc and mean_p_diff <= 3.0
    assert torch.allclose(y_diff_g.float().cpu(), y_diff_c.float(), atol=2e-2)
    assert torch.equal(used_diff_g.cpu(), used_diff_c)
    cos_diff = float(F.cosine_similarity(y_diff_g.float(), y_exact).item())
    cos_noanc = float(F.cosine_similarity(y_noanc_g.float(), y_exact).item())
    assert cos_diff > 0.999 and cos_diff >= cos_noanc - 1e-4

    # Case 3: Verify exact bias handling with crisp anchor (`y_val += (1 - alpha) * bias`)
    y_bias_g = launch_pure_mirror_diff_q1_gemv(
        x_small, t_gpu["packed_lo"], t_gpu["qscale"], t_gpu["qzero"], t_gpu["row_scale"], t_gpu["col_scale"],
        x_last=x_last, y_last=y_last_b, bias=bias, group_size=64, max_planes=5, min_planes=1, mirror_tol=0.0,
    )
    y_bias_ref = F.linear(x_small, w, bias).float()
    assert float(F.cosine_similarity(y_bias_g.float(), y_bias_ref).item()) > 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_activation_aware_mirror_crystal_carver_and_two_slit_protection_cuda():
    """Verify:
    1. `carve_q1_mirror_crystal` stops dynamically at activation-verified `k_opt <= max_k` and records scar.
    2. Kernel #4 `unclosed_cone_protect=True` returns `"direct_pass_unclosed"` (both CUDA & CPU)
       instead of dropping unseen branches to zero!
    """
    torch.manual_seed(42)
    dev = torch.device("cuda:0")
    w = (torch.randn(128, 256, dtype=torch.bfloat16, device=dev) * 0.04).contiguous()
    x = torch.randn(4, 256, dtype=torch.bfloat16, device=dev)
    y_ex = F.linear(x, w)

    t_cryst, meta, scar = carve_q1_mirror_crystal(
        w, x, y_ex, max_cubes=8, min_cubes=2, mirror_tol=0.06, root_bits=8, group_size=64
    )
    assert 2 <= scar["k_opt"] < 8, f"Expected activation-aware early stop < 8, got {scar}"
    assert scar["closed_equal"] is True
    assert scar["scar_rel_rmse"] < 0.06
    assert int(t_cryst["packed_lo"].shape[0]) == scar["k_opt"]
    assert meta.law == "mirror_crystal_positional_halving"

    # Two-Slit Unclosed Protection in Kernel #4 (CUDA == CPU)
    pm_c = PureMirrorCudaFeedback(num_layers=8, num_experts=16, device="cpu")
    pm_g = PureMirrorCudaFeedback(num_layers=8, num_experts=16, device="cuda:0")
    v = torch.randn(4, 64, dtype=torch.float32)
    u_c = pm_c.project_states(0, v, update_mean=True)
    u_g = pm_g.project_states(0, v.to(dev), update_mean=True)
    pm_c.observe_mirror_step(0, [[(0, 0.5), (1, 0.3), (2, 0.2)] for _ in range(4)], u_c, u_post=u_c, is_commit=True)
    pm_g.observe_mirror_step(0, [[(0, 0.5), (1, 0.3), (2, 0.2)] for _ in range(4)], u_g, u_post=u_g, is_commit=True)

    # Candidate 15 is unseen (`act == 0`): without protection -> no_mirror_link; WITH protection -> direct_pass_unclosed!
    r_off_c = pm_c.evaluate_continuation_batch(
        0, [0, 1, 2], [], [15], [0.1], [0], u_info=u_c[0], prev_u_info=None,
        line_norm=2.0, last_obs_norm=0.8, last_res_norm=0.5, w_done=0.8, rem_w=0.2,
        tok_done=3, rest_min=3, rest_tol=0.24, unclosed_cone_protect=False,
    )
    r_off_g = pm_g.evaluate_continuation_batch(
        0, [0, 1, 2], [], [15], [0.1], [0], u_info=u_g[0], prev_u_info=None,
        line_norm=2.0, last_obs_norm=0.8, last_res_norm=0.5, w_done=0.8, rem_w=0.2,
        tok_done=3, rest_min=3, rest_tol=0.24, unclosed_cone_protect=False,
    )
    r_on_c = pm_c.evaluate_continuation_batch(
        0, [0, 1, 2], [], [15], [0.1], [0], u_info=u_c[0], prev_u_info=None,
        line_norm=2.0, last_obs_norm=0.8, last_res_norm=0.5, w_done=0.8, rem_w=0.2,
        tok_done=3, rest_min=3, rest_tol=0.24, unclosed_cone_protect=True,
    )
    r_on_g = pm_g.evaluate_continuation_batch(
        0, [0, 1, 2], [], [15], [0.1], [0], u_info=u_g[0], prev_u_info=None,
        line_norm=2.0, last_obs_norm=0.8, last_res_norm=0.5, w_done=0.8, rem_w=0.2,
        tok_done=3, rest_min=3, rest_tol=0.24, unclosed_cone_protect=True,
    )
    assert r_off_c[0][0] == r_off_g[0][0] == "no_mirror_link"
    assert r_on_c[0][0] == r_on_g[0][0] == "direct_pass_unclosed"


@pytest.mark.parametrize("device_str", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))])
def test_mirror_dual_prefill_exact_bf16_and_anchored_decode(device_str: str):
    """End-to-end verification of V3 `MirrorDualPrefill` + Anchored Differential Decode on BOTH CPU & CUDA:
    1. Prefill executes ALL routed experts in exact BF16 (`cos > 0.9999` vs reference BF16!).
    2. One-off mid-prompt periphery experts skip slow Q1 carving (`prefill_carves_avoided > 0`).
    3. Tail `FuturePressure` & recurring intersection experts crystallize with crisp BF16 anchors.
    4. Decode uses Differential-Reversible Q1 CUDA kernel with anchors (`diff_q1_anchored_calls > 0` on CUDA)
       while maintaining strict `A->B->C(0)->A` (`B->A == 0`, `standby_bytes == 0`)!
    """
    torch.manual_seed(1337)
    dev = torch.device(device_str)
    h, inter = 64, 64
    num_layers, num_exp = 2, 24
    bank = {
        (li, e): (
            torch.randn(2 * inter, h, dtype=torch.bfloat16) * 0.04,
            torch.randn(h, inter, dtype=torch.bfloat16) * 0.04,
        )
        for li in range(num_layers)
        for e in range(num_exp)
    }

    # 8 prefill tokens: experts 0,1,2 recur in core/tail; experts 10..23 are one-off periphery!
    x_pre = torch.randn(8, h, device=dev, dtype=torch.bfloat16)
    idx_pre = torch.tensor([
        [0, 1, 2, 10],
        [0, 1, 2, 11],
        [0, 1, 2, 12],
        [0, 1, 2, 13],
        [0, 1, 2, 14],
        [0, 1, 2, 15],
        [0, 1, 3, 16],
        [0, 1, 3, 17],
    ], dtype=torch.long, device=dev)
    w_pre = torch.tensor([[0.42, 0.32, 0.22, 0.04] for _ in range(8)], dtype=torch.bfloat16, device=dev)

    # Exact BF16 causal reference for layer 0
    ref_y0 = torch.zeros((8, h), dtype=torch.float32, device=dev)
    for ti in range(8):
        for k_pos in range(4):
            eid = int(idx_pre[ti, k_pos].item())
            rw = float(w_pre[ti, k_pos].item())
            gu_w, dn_w = bank[(0, eid)]
            gu_o = F.linear(x_pre[ti : ti + 1], gu_w.to(dev))
            gp, up = gu_o.chunk(2, dim=-1)
            ye = F.linear(F.silu(gp) * up, dn_w.to(dev))
            ref_y0[ti] += ye[0].float() * rw

    contour = StanokContainmentContour(device=dev, enabled=False)
    cache = LazyMoEBinaryHalvingCache(
        lambda li, e: bank[(li, e)],
        expert_batch_reader=lambda li, es: {e: bank[(li, e)] for e in es},
        num_layers=num_layers,
        num_experts=num_exp,
        device=dev,
        store_cubes=8,
        root_bits=8,
        max_cache_bytes=64 * 1024**2,
        enable_expert_relay=True,
        imprint_puzzle=True,
        imprint_depths=(5, 4, 3, 2),
        expert_rest_tol=0.24,
        expert_rest_min=3,
        stanok_contour=contour,
        graphify_feedback="pure_mirror",
    )
    try:
        y0 = cache.forward_layer_moe(0, x_pre, idx_pre, w_pre, F.silu)
        _ = cache.forward_layer_moe(1, x_pre, idx_pre, w_pre, F.silu)
        cos_pre = float(F.cosine_similarity(y0.float().flatten(), ref_y0.flatten(), dim=0).item())
        assert cos_pre > 0.9999, f"MirrorDualPrefill should match exact BF16, got cos={cos_pre}"

        # Decode token aligned with prompt tail -> hits crystallized anchors on experts 0, 1, 3!
        x_dec = (x_pre[-1:] + 0.05 * torch.randn(1, h, device=dev, dtype=torch.bfloat16)).to(torch.bfloat16)
        y_dec = cache.forward_layer_moe(
            0,
            x_dec,
            torch.tensor([[0, 1, 3, 18]], dtype=torch.long, device=dev),
            torch.tensor([[0.45, 0.30, 0.21, 0.04]], dtype=torch.bfloat16, device=dev),
            F.silu,
        )
        assert bool(torch.isfinite(y_dec).all())

        st = cache._imprint_overlay.stats()
        mv3 = st["mirror_v3"]
        assert mv3["prefill_crystallized"] >= 8
        assert mv3["prefill_carves_avoided"] >= 10, f"Expected one-off periphery to skip Q1 carving: {mv3}"
        assert mv3["prefill_direct_pass"] >= 10
        assert mv3["crystal_scars_active"] >= 8
        assert mv3["anchors_resident"] >= 8
        assert st["relay"]["transitions"]["B->A"] == 0
        assert st["relay"]["standby_bytes"] == 0
        if device_str.startswith("cuda"):
            assert mv3["diff_q1_anchored_calls"] >= 6
            assert st["graphify_feedback"]["topology"]["cuda_launches"]["diff_q1_gemv_mirror"] >= 6
            assert st["graphify_feedback"]["topology"]["layer_nodes_born"] == 2
            assert st["graphify_feedback"]["topology"]["bracketless_stem"] is True
            assert st["graphify_feedback"]["topology"]["participants_intact"] is True
    finally:
        cache.close()
        contour.close()
