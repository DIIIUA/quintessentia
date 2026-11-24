"""Strict tests for Version 4 (Final — Conveyor-Free): Isolated Multi-Scale Mirror Equality.

Verifies:
  1. ZERO Conveyor (`Q1RoleCycleArena` / `A->B->C(0)->A` removed in V4!).
  2. ZERO Graphify & ZERO Cross-Scale Bus (`UnifiedProbabilityBus` / `StanokContainmentContour`).
  3. ZERO Cross-Scale Coupling between M1, M2, M3, M4, and M5.
  4. Genuine mirror-born mechanisms in M1 (`MirrorLoadDispatcher`), M2 (`MirrorRegionBreathingPool`),
     M3 (`MirrorColdWarmupScale`), M4 (`MirrorPostPrefillRegionWarmupScale`), and M5 (end-to-end).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.binary_halving_q1 import LazyMoEBinaryHalvingCache
from src.mirror_scales_v4 import (
    MirrorColdWarmupScale,
    MirrorIsolatedMoEOverlayV4,
    MirrorLoadDispatcher,
    MirrorPostPrefillRegionWarmupScale,
    MirrorRegionBreathingPool,
)
from src.stanok_contour import normalize_feedback_mode


def _make_synthetic_experts(num_layers: int, num_experts: int, h_dim: int, i_dim: int, seed: int = 17):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    weights = {}
    for li in range(num_layers):
        for eid in range(num_experts):
            gu = (torch.randn((2 * i_dim, h_dim), generator=gen, dtype=torch.float32) * 0.08).to(torch.bfloat16)
            dn = (torch.randn((h_dim, i_dim), generator=gen, dtype=torch.float32) * 0.08).to(torch.bfloat16)
            weights[(li, eid)] = (gu, dn)
    return weights


def _exact_moe_bf16(weights, li, x, top_k_idx, top_k_w):
    m, h = x.shape
    out = torch.zeros((m, h), dtype=torch.float32, device=x.device)
    for ti in range(m):
        for ki in range(top_k_idx.shape[1]):
            eid = int(top_k_idx[ti, ki].item())
            rw = float(top_k_w[ti, ki].item())
            gu, dn = weights[(li, eid)]
            gu_d = gu.to(device=x.device, dtype=torch.bfloat16)
            dn_d = dn.to(device=x.device, dtype=torch.bfloat16)
            g_out = F.linear(x[ti : ti + 1].to(torch.bfloat16), gu_d)
            g_p, u_p = g_out.chunk(2, dim=-1)
            act = F.silu(g_p) * u_p
            y = F.linear(act, dn_d).float()
            out[ti : ti + 1] += rw * y
    return out.to(x.dtype)


def test_v4_zero_conveyor_zero_graphify_zero_cross_scale_coupling():
    """HARD LAW: V4 has ZERO Conveyor, ZERO Graphify, ZERO Bus, ZERO cross-scale references."""
    assert normalize_feedback_mode("v4") == "mirror_scales"
    assert normalize_feedback_mode("mirror_scales") == "mirror_scales"
    assert normalize_feedback_mode("mirror_v4") == "mirror_scales"
    assert normalize_feedback_mode("isolated_mirror") == "mirror_scales"

    # AST purity check: mirror_scales_v4.py must NEVER import or reference Q1RoleCycleArena, Graphify, Bus, or UNI_PHI
    src_text = Path(__file__).resolve().parents[1].joinpath("src", "mirror_scales_v4.py").read_text(encoding="utf-8")
    tree = ast.parse(src_text)
    code_symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                code_symbols.add(alias.name)
                if alias.asname:
                    code_symbols.add(alias.asname)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                code_symbols.add(alias.name)
        elif isinstance(node, ast.Name):
            code_symbols.add(node.id)
        elif isinstance(node, ast.Attribute):
            code_symbols.add(node.attr)

    for forbidden in (
        "Q1RoleCycleArena",
        "OnlineGraphifyTopology",
        "SpiralSphereMutualFeedback",
        "UnifiedProbabilityBus",
        "StanokContainmentContour",
        "UNI_PHI",
    ):
        assert forbidden not in code_symbols, f"Forbidden conveyor/Graphify/Bus symbol {forbidden!r} used in V4!"

    weights = _make_synthetic_experts(2, 8, 64, 64)
    cache = LazyMoEBinaryHalvingCache(
        lambda l, e: weights[(l, e)],
        num_layers=2,
        num_experts=8,
        device=torch.device("cpu"),
        store_cubes=5,
        root_bits=8,
        group_size=64,
        imprint_puzzle=True,
        graphify_feedback="v4",
    )
    try:
        ov = cache._imprint_overlay
        assert isinstance(ov, MirrorIsolatedMoEOverlayV4)
        assert ov.relay is None
        assert ov.graph is None
        assert ov.graphify_feedback is False
        assert ov.probability_bus is None

        m1 = ov.load_dispatcher
        m2 = ov.region_pool
        m3 = MirrorColdWarmupScale()
        m4 = ov.post_prefill_warmup_scale
        mechanisms = [m1, m2, m3, m4]

        # None of M1, M2, M3, M4 may hold any reference to another mechanism, overlay, cache, or conveyor
        forbidden_ids = {id(ov), id(cache)}
        for i, mech in enumerate(mechanisms):
            other_ids = forbidden_ids | {id(mechanisms[j]) for j in range(len(mechanisms)) if j != i}
            for attr_name, attr_val in vars(mech).items():
                assert id(attr_val) not in other_ids, (
                    f"{type(mech).__name__}.{attr_name} holds forbidden cross-scale reference!"
                )

        st = ov.stats()
        assert st["conveyor_removed"] is True
        assert st["graphify_removed"] is True
        assert st["cross_scale_coupling"] is False
        assert "relay" not in st
    finally:
        cache.close()


def test_m1_mirror_load_dispatcher_regimes_damping_and_burst_adaptation():
    """Mechanism M1: DIRECT_BURST, OVERLAP_ASYNC, preventive THROTTLE_DAMP, SHED_PROTECT + burst span."""
    GiB = 1024**3
    total_b = 8 * GiB
    reserve_b = int(0.10 * total_b)

    readings = [
        # 1. Plenty of headroom + calm GPU/bus -> DIRECT_BURST + expand
        (5 * GiB, total_b, 20.0, 25.0, 0, "mock_nvml"),
        # 2. Same steady state -> DIRECT_BURST + EQUAL hold
        (5 * GiB, total_b, 20.0, 25.0, 0, "mock_nvml"),
        # 3. Higher GPU/bus load with safe headroom -> OVERLAP_ASYNC
        (4.8 * GiB, total_b, 78.0, 74.0, 0, "mock_nvml"),
        # 4. Sharp drop 4.8 -> 1.8 GiB (drop 3.0 GiB => projected -1.2 GiB < reserve) BEFORE floor -> THROTTLE_DAMP!
        (int(1.8 * GiB), total_b, 80.0, 82.0, 0, "mock_nvml"),
        # 5. Actual physical floor breach -> SHED_PROTECT with exact deficit!
        (int(0.5 * GiB), total_b, 92.0, 95.0, 0, "mock_nvml"),
    ]
    idx = {"i": 0}

    def _sensor():
        i = min(idx["i"], len(readings) - 1)
        idx["i"] += 1
        return readings[i]

    m1 = MirrorLoadDispatcher(
        device="cpu",
        enabled=True,
        safety_reserve_bytes=reserve_b,
        initial_batch_span=16,
        min_batch_span=4,
        max_batch_span=32,
        sensor_fn=_sensor,
    )
    budget = 2 * GiB

    d1 = m1.evaluate_load(cached_bytes=1 * GiB, current_budget_bytes=budget)
    assert d1.dispatch_mode == "DIRECT_BURST"
    assert d1.budget_action == "expand"
    budget = d1.target_budget_bytes

    d2 = m1.evaluate_load(cached_bytes=1 * GiB, current_budget_bytes=budget)
    assert d2.dispatch_mode == "DIRECT_BURST"
    assert d2.budget_action == "hold"
    assert d2.mirror_state == "EQUAL"

    d3 = m1.evaluate_load(cached_bytes=1 * GiB, current_budget_bytes=budget)
    assert d3.dispatch_mode == "OVERLAP_ASYNC"

    d4 = m1.evaluate_load(cached_bytes=2 * GiB, current_budget_bytes=budget)
    assert d4.dispatch_mode == "THROTTLE_DAMP"
    assert d4.budget_action == "damp"
    assert d4.in_flight_cap == 1
    assert d4.shed_bytes_needed == 0
    assert d4.target_budget_bytes < budget

    d5 = m1.evaluate_load(cached_bytes=2 * GiB, current_budget_bytes=d4.target_budget_bytes)
    assert d5.dispatch_mode == "SHED_PROTECT"
    assert d5.budget_action == "protect"
    assert d5.shed_bytes_needed == reserve_b - int(0.5 * GiB)

    # Timing-based burst span adaptation (independent of conveyor!)
    m1.batch_span = 16
    span_after_slow_io, _, _ = m1.observe_dispatch_timing(t_gpu_ms=4.0, t_io_ms=14.0, t_wait_ms=10.0, items_dispatched=16)
    assert span_after_slow_io < 16
    span_after_fast_io, _, _ = m1.observe_dispatch_timing(
        t_gpu_ms=22.0, t_io_ms=4.0, t_wait_ms=0.0, items_dispatched=span_after_slow_io
    )
    assert span_after_fast_io > span_after_slow_io


def test_m2_mirror_region_breathing_pool_reuse_hold_and_exhale():
    """Mechanism M2: direct admit, PIN_REUSE, HOLD_WARM_REGION, immediate EXHALE_RECLAIM (NO A/B/C ring!)."""
    m2 = MirrorRegionBreathingPool(
        device=torch.device("cpu"),
        max_pool_bytes=64 * 1024**2,
        max_warm_regions=3,
    )

    # 1. One-off region -> DIRECT_ADMIT then immediate EXHALE_RECLAIM (0 leaked bytes!)
    p1 = m2.admit_region(("oneoff", 0), {"x": 1}, gpu_bytes=4096, recurring_hint=False)
    assert p1 == {"x": 1}
    assert m2.live_bytes == 4096
    act1 = m2.finish_region(("oneoff", 0), keep_if_reusable=False)
    assert act1 == "EXHALE_RECLAIM"
    assert m2.live_bytes == 0
    assert m2.exhales_reaped >= 1

    # 2. Recurring region -> DIRECT_ADMIT -> HOLD_WARM_REGION -> instant PIN_REUSE (0 copy, 0 IO!)
    m2.admit_region(("warm_L0", 0), {"w": 42}, gpu_bytes=8192, recurring_hint=True)
    act2 = m2.finish_region(("warm_L0", 0), keep_if_reusable=True)
    assert act2 == "HOLD_WARM_REGION"
    assert m2.live_bytes == 8192

    reused = m2.lookup_reuse(("warm_L0", 0))
    assert reused == {"w": 42}
    assert m2.pin_reuses == 1
    assert m2.stats()["conveyor_removed"] is True
    m2.close()
    assert m2.live_bytes == 0


def test_m3_mirror_cold_warmup_scale_plateau_stop():
    """Scale M3: stops startup warmup automatically once latency reaches warm mirror plateau."""
    m3 = MirrorColdWarmupScale(mode="plateau", min_layers_warm=2, plateau_confirm=2, max_warmup_seconds=10.0)
    fake_ms = [60.0, 28.0, 12.0, 11.9, 11.9, 11.8, 11.8, 11.8]
    stopped = None
    for i, dt in enumerate(fake_ms, start=1):
        dec = m3.evaluate_step(dt, warmed_so_far=i, total_targets=40, elapsed_seconds=0.5)
        if not dec.continue_warmup:
            stopped = (i, dec.reason)
            break
    assert stopped is not None
    assert stopped[1] == "mirror_plateau_equal"
    assert stopped[0] < 40


def test_m4_mirror_post_prefill_region_warmup_and_scar_preservation():
    """Scale M4: warms known post-prefill regions without Graphify and preserves map on SPLIT scars."""
    m4 = MirrorPostPrefillRegionWarmupScale(num_layers=4, max_region_experts=8, initial_width=2, max_width=3)
    m4.record_prefill_layer(0, body_weights={1: 3.0, 2: 2.0, 5: 1.0}, tail_weights={2: 1.5})
    m4.record_prefill_layer(1, body_weights={3: 4.0, 4: 2.0, 6: 1.0}, tail_weights={4: 2.0})
    m4.record_prefill_layer(2, body_weights={7: 3.0, 8: 2.0}, tail_weights={8: 1.0})

    orig_l0 = set(m4._prefill_sets[0])
    resident = {(0, 1), (0, 2)}

    p1 = m4.observe_and_plan_next(0, [2, 1], [0.6, 0.4], lambda l, e: (l, e) in resident)
    assert p1.target_layer == 1
    assert p1.eids_to_warm == (4, 3)

    resident.update({(1, 3), (1, 4), (1, 6)})
    p2 = m4.observe_and_plan_next(0, [2, 1], [0.6, 0.4], lambda l, e: (l, e) in resident)
    assert p2.target_layer == 2
    assert p2.eids_to_warm == (8, 7)
    assert m4.lookahead_l2_plans == 1

    resident.update({(2, 7), (2, 8)})
    p3 = m4.observe_and_plan_next(0, [2, 1], [0.6, 0.4], lambda l, e: (l, e) in resident)
    assert p3.eids_to_warm == ()
    assert p3.mirror_state == "SATURATED"

    m4.observe_and_plan_next(1, [14, 15], [0.7, 0.3], lambda l, e: (l, e) in resident)
    assert set(m4._prefill_sets[0]) == orig_l0
    assert 14 in m4._scar_weights[1] and 15 in m4._scar_weights[1]
    p_scar = m4.observe_and_plan_next(0, [2, 1], [0.6, 0.4], lambda l, e: (l, e) in resident)
    assert p_scar.target_layer == 1
    assert 14 in p_scar.eids_to_warm


@pytest.mark.parametrize(
    "dev_str",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))],
)
def test_v4_conveyor_free_end_to_end_cpu_and_cuda(dev_str: str):
    """End-to-end conveyor-free V4 test: M1 async bursts + M2 breathing pool + M4 warmup + M5 Kernel #6."""
    device = torch.device(dev_str)
    num_layers, num_experts, h_dim, i_dim = 4, 12, 128, 128
    weights = _make_synthetic_experts(num_layers, num_experts, h_dim, i_dim, seed=42)

    def _read_single(li, eid):
        g, d = weights[(li, eid)]
        return g.to(device=device), d.to(device=device)

    def _read_batch(li, eids):
        return {eid: _read_single(li, eid) for eid in eids}

    cache = LazyMoEBinaryHalvingCache(
        _read_single,
        num_layers=num_layers,
        num_experts=num_experts,
        device=device,
        store_cubes=5,
        root_bits=8,
        group_size=64,
        use_fwht=True,
        max_cache_bytes=128 * 1024**2,
        host_cache_bytes=64 * 1024**2,
        cache_on_gpu=(device.type == "cuda"),
        expert_batch_reader=_read_batch,
        expert_rest_tol=0.18,
        expert_rest_min=2,
        imprint_puzzle=True,
        imprint_depths=(5, 4, 3, 2),
        imprint_auto_vram=False,
        graphify_feedback="v4",
    )
    try:
        ov = cache._imprint_overlay
        assert isinstance(ov, MirrorIsolatedMoEOverlayV4)
        assert ov.relay is None
        gen = torch.Generator(device="cpu").manual_seed(99)

        # 1. Prefill (M = 10 tokens across 4 layers) — exact BF16 via M1 + M2
        x_pre = (torch.randn((10, h_dim), generator=gen, dtype=torch.float32) * 0.5).to(device=device, dtype=torch.bfloat16)
        idx_pre = torch.tensor([[0, 1, 8, 9]] * 7 + [[0, 2, 3, 10]] * 3, dtype=torch.long, device=device)
        w_pre = torch.tensor([[0.45, 0.30, 0.15, 0.10]] * 10, dtype=torch.float32, device=device)

        for li in range(num_layers):
            y_v4 = cache.forward_layer_moe(li, x_pre, idx_pre, w_pre, F.silu)
            y_ref = _exact_moe_bf16(weights, li, x_pre, idx_pre, w_pre)
            cos_pre = float(F.cosine_similarity(y_v4.float().reshape(-1), y_ref.float().reshape(-1), dim=0).item())
            assert cos_pre > 0.9995, f"Prefill layer {li} must be exact BF16 (got cos={cos_pre:.5f})"

        assert ov.post_prefill_warmup_scale.prefill_layers_mapped == num_layers
        assert ov.prefill_crystallized >= 4
        assert ov.prefill_carves_avoided >= 4

        # 2. Decode (3 steps x 4 layers) — M4 background warmup + M2 direct admit/exhale + M5 Kernel #6
        for step in range(3):
            x_dec = x_pre[-1:] + 0.04 * torch.randn((1, h_dim), generator=gen).to(device=device, dtype=torch.bfloat16)
            idx_dec = torch.tensor([[0, 2, 1, 8]], dtype=torch.long, device=device)
            w_dec = torch.tensor([[0.45, 0.30, 0.18, 0.07]], dtype=torch.float32, device=device)
            for li in range(num_layers):
                y_dec = cache.forward_layer_moe(li, x_dec, idx_dec, w_dec, F.silu)
                y_ref = _exact_moe_bf16(weights, li, x_dec, idx_dec, w_dec)
                cos_d = float(F.cosine_similarity(y_dec.float().reshape(-1), y_ref.float().reshape(-1), dim=0).item())
                assert cos_d > 0.97, f"Decode step {step} layer {li} cos too low: {cos_d:.4f}"

        st = ov.stats()
        assert st["conveyor_removed"] is True
        assert st["graphify_removed"] is True
        assert st["cross_scale_coupling"] is False
        assert "relay" not in st
        assert st["m1_load_dispatcher"]["timing_updates"] >= 4
        assert st["m2_region_breathing_pool"]["admits_direct"] >= 4
        assert st["m2_region_breathing_pool"]["exhales_reaped"] >= 4
        assert st["m2_region_breathing_pool"]["live_regions"] == 0
        assert st["m5_layer_execution"]["diff_q1_anchored_calls"] >= 4
        assert st["m4_post_prefill_warmup"]["plans_emitted"] >= 1
        assert st["m5_layer_execution"]["post_prefill_bg_adoptions"] >= 1
        assert "zero_disk_spike_layers" in st["m5_layer_execution"]
        assert "shed_scars_saved" in st["m5_layer_execution"]
    finally:
        cache.close()


def test_m5b_pure_math_cube_depth_and_reversible_shed_scar():
    """M5b: pure-math cube depth (no 0.5/0.25 step-if) + bit-exact reversible shed scar."""
    from src.binary_halving_q1 import carve_q1_cubes_incremental
    from src.mirror_scales_v4 import (
        _reattach_q1_scar,
        _trim_q1_planes_with_scar,
        mirror_cube_depth_math,
    )

    # Weak body + zero tail -> rim_k (3); strong rising tail -> core_k (5)
    assert mirror_cube_depth_math(0.05, 0.0, 3, 5) == 3
    assert mirror_cube_depth_math(0.20, 0.85, 3, 5) == 5
    assert mirror_cube_depth_math(0.90, 0.90, 3, 5) == 5
    assert 3 <= mirror_cube_depth_math(0.40, 0.30, 3, 5) <= 5

    # Bit-exact reversible Depth-Shedding: trim 5 -> 4 -> reattach scar -> bit-identical 5-plane tensor!
    gen = torch.Generator(device="cpu").manual_seed(7)
    w = (torch.randn((64, 128), generator=gen, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    orig, _ = carve_q1_cubes_incremental(w, 5, root_bits=8, group_size=64, use_fwht=True, sinkhorn_iters=2)
    trimmed, scar = _trim_q1_planes_with_scar(orig, 4)
    assert int(trimmed["packed_lo"].shape[0]) == 4
    assert scar is not None and int(scar["packed_lo"].shape[0]) == 1
    restored = _reattach_q1_scar(trimmed, scar, torch.device("cpu"))
    assert int(restored["packed_lo"].shape[0]) == 5
    assert torch.equal(restored["packed_lo"], orig["packed_lo"])


def test_m0_trunk_mirror_carve_kernel6_and_mirror_survey():
    """M0: pure-math trunk carving + Kernel #6 on BinaryHalvingQ1Linear + mirror StreamingCubeSurvey."""
    from src.binary_halving_q1 import BinaryHalvingQ1Linear, carve_trunk_mirror_q1
    from src.streaming_cube_survey import StreamingCubeSurvey

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device="cpu").manual_seed(11)
    w = (torch.randn((128, 128), generator=gen, dtype=torch.float32) * 0.08).to(device=dev, dtype=torch.bfloat16)
    b = (torch.randn((128,), generator=gen, dtype=torch.float32) * 0.02).to(device=dev, dtype=torch.bfloat16)

    tensors, meta, k_opt, rem_rel = carve_trunk_mirror_q1(
        w, 5, min_cubes=3, mirror_tol=0.05, root_bits=8, group_size=64, use_fwht=True, sinkhorn_iters=2
    )
    assert 3 <= k_opt <= 5
    assert rem_rel < 0.25

    mod = BinaryHalvingQ1Linear(128, 128, tensors, meta, bias=b, layer_idx=6, proj_name="q_proj")
    mod.active_cubes = k_opt
    mod.mirror_diff_tol = 0.015

    # Prefill (M = 4) captures crisp BF16 anchor; decode (M = 1) runs CUDA Kernel #6 with exact bias!
    x_pre = (torch.randn((4, 128), generator=gen, dtype=torch.float32) * 0.4).to(device=dev, dtype=torch.bfloat16)
    y_pre = mod(x_pre)
    y_pre_ref = F.linear(x_pre, w, b)
    cos_pre = float(F.cosine_similarity(y_pre.float().reshape(-1), y_pre_ref.float().reshape(-1), dim=0).item())
    assert cos_pre > 0.98

    x_dec1 = x_pre[-1:] + 0.03 * torch.randn((1, 128), generator=gen).to(device=dev, dtype=torch.bfloat16)
    x_dec2 = x_pre[-1:] * 1.15  # collinear -> zero-plane EQUAL path in CUDA Kernel #6!
    y_d1 = mod(x_dec1)
    y_d2 = mod(x_dec2)
    cos_d1 = float(F.cosine_similarity(y_d1.float().reshape(-1), F.linear(x_dec1, w, b).float().reshape(-1), dim=0).item())
    cos_d2 = float(F.cosine_similarity(y_d2.float().reshape(-1), F.linear(x_dec2, w, b).float().reshape(-1), dim=0).item())
    assert cos_d1 > 0.98 and cos_d2 > 0.98
    if dev.type == "cuda":
        assert mod.mirror_diff_calls == 2
        assert mod._mirror_x_last is not None

    # Method 1 StreamingCubeSurvey with pure-math `selection_policy='mirror'`
    survey = StreamingCubeSurvey([], root_bits=8)
    survey.module_specs = [(0, "q_proj", 128 * 128), (0, "k_proj", 128 * 128)]
    # Still halving sharply at k=4 (0.04 -> 0.008 -> 0.004): mirror holds until k=5 where d_step <= 0.5
    survey.errors[(0, "q_proj")] = [0.25, 0.11, 0.04, 0.008, 0.004, 0.002, 0.001, 0.0005]
    # True plateau at k=4 (0.008 -> 0.007 -> 0.0065): d_next <= 0.25 -> converges at k=4
    survey.errors[(0, "k_proj")] = [0.25, 0.11, 0.04, 0.008, 0.007, 0.0065, 0.006, 0.0058]
    bp = survey.blueprint({"layers": [{"layer": 0}]}, relative_error=0.01, selection_policy="mirror")
    assert bp["selection_policy"] == "mirror"
    assert bp["layer_proj_cubes"][0]["q_proj"] == 5
    assert bp["layer_proj_cubes"][0]["k_proj"] == 4


def test_m6_mirror_probability_path_split_protection_and_entropy_floor():
    """M6: pure-math tensor MirrorProbabilityPath (ZERO 0.65/0.35, ZERO -12.0) protects late SPLIT tokens."""
    from src.survivor_probability_path import SurvivorProbabilityTracker

    class _FakeHead:
        vocab_size = 64

        def project_candidates(self, h_norm, cand_ids):
            # Pass checkpoint logits directly through identity-encoded hidden vector (dim == 64)
            return h_norm.reshape(-1)[cand_ids]

    class _FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(5)])
            self.lm_head = _FakeHead()

    class _FakeEngine:
        def __init__(self):
            self.model = _FakeModel()
            self.config = type("Cfg", (), {"vocab_size": 64})()

        def _decoder_layers(self):
            return list(self.model.layers)

    eng = _FakeEngine()
    tracker = SurvivorProbabilityTracker(eng, top_survivors=8, candidate_cap=64, mirror_mode=True)
    # Bypass RMS norm so checkpoint vector == raw logits for clean deterministic check
    tracker._apply_norm = staticmethod(lambda _norm, x: x.float())

    # Construct 5 depth checkpoints over 64 tokens:
    #   token 0: steady strong leader (+6.0 across all checkpoints)
    #   token 1: dead floor (-30.0 across all checkpoints -> pruned by entropy floor!)
    #   token 2: late SPLIT turn (-22.0 at L0..L3, shoots up to -11.0 at L4 -> protected!)
    base = torch.full((64,), -2.0, dtype=torch.float32)
    base[0] = 6.0
    base[1] = -30.0
    for ck in range(5):
        v = base.clone()
        v[2] = -22.0 if ck < 4 else -11.0
        tracker._checkpoint_hidden[ck] = v.unsqueeze(0)

    tracker._recompute_survivors()
    assert tracker.survivor_ids is not None
    surv = set(tracker.survivor_ids.tolist())
    pruned = set(tracker.early_pruned_ids.tolist())
    assert 0 in surv, "Steady leader must survive"
    assert 1 in pruned, "Dead floor token must be pruned by entropy floor"
    assert tracker.last_entropy_floor is not None
    assert tracker.split_protected_count >= 1, "Late rising SPLIT token must trigger mirror split protection!"
    tracker.close()


def test_approach1_zero_sync_hot_decode_and_m0_m5_tails():
    """Verify Approach 1: zero-sync hot decode, coalesced D2H, M0 plateau stop, and M5 mirror cubes_for_weight."""
    from src.binary_halving_q1 import LazyMoEBinaryHalvingCache, carve_trunk_mirror_q1

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights = _make_synthetic_experts(2, 8, 128, 128, seed=77)

    def _read_single(li, eid):
        g, d = weights[(li, eid)]
        return g.to(device=dev), d.to(device=dev)

    def _read_batch(li, eids):
        return {eid: _read_single(li, eid) for eid in eids}

    cache = LazyMoEBinaryHalvingCache(
        _read_single,
        num_layers=2,
        num_experts=8,
        device=dev,
        store_cubes=6,
        root_bits=8,
        group_size=64,
        use_fwht=True,
        max_cache_bytes=128 * 1024**2,
        host_cache_bytes=64 * 1024**2,
        cache_on_gpu=(dev.type == "cuda"),
        expert_batch_reader=_read_batch,
        imprint_puzzle=True,
        imprint_depths=(6, 5, 4, 3),
        imprint_auto_vram=False,
        graphify_feedback="v4",
    )
    try:
        # 1. M5 cubes_for_weight uses continuous mirror algebra in V4 (NOT constant store_cubes)
        k_weak = cache.cubes_for_weight(0.05, max_router_weight=0.70, total_router_weight=1.0)
        k_strong = cache.cubes_for_weight(0.70, max_router_weight=0.70, total_router_weight=1.0)
        assert 3 <= k_weak < k_strong <= 6

        # 2. Prefill crystallizes experts 0, 1, 2, 3 so decode on those experts is 100% zero-disk
        gen = torch.Generator(device="cpu").manual_seed(123)
        x_pre = (torch.randn((6, 128), generator=gen, dtype=torch.float32) * 0.4).to(device=dev, dtype=torch.bfloat16)
        idx_pre = torch.tensor([[0, 1, 2, 3]] * 6, dtype=torch.long, device=dev)
        w_pre = torch.tensor([[0.45, 0.30, 0.15, 0.10]] * 6, dtype=torch.float32, device=dev)
        cache.forward_layer_moe(0, x_pre, idx_pre, w_pre, F.silu)

        ov = cache._imprint_overlay
        # Pick only experts actually crystallized into L1/L2 by prefill for the 100% zero-disk hot test
        hot_eids = sorted(e for (li, e) in cache._lru.keys() if li == 0)
        assert len(hot_eids) >= 2
        syncs_before = ov.decode_norm_syncs_performed
        avoided_before = ov.decode_norm_syncs_avoided
        x_dec = x_pre[-1:]
        idx_hot = torch.tensor([hot_eids], dtype=torch.long, device=dev)
        w_vals = [1.0 / len(hot_eids)] * len(hot_eids)
        w_hot = torch.tensor([w_vals], dtype=torch.float32, device=dev)
        y_hot = cache.forward_layer_moe(0, x_dec, idx_hot, w_hot, F.silu)
        assert torch.isfinite(y_hot).all()
        # Zero-disk decode must perform ZERO norm syncs and avoid per-expert + spine syncs!
        assert ov.decode_norm_syncs_performed == syncs_before
        assert ov.decode_norm_syncs_avoided >= avoided_before + len(hot_eids) + 1
        if dev.type == "cuda":
            assert ov.decode_d2h_coalesced >= 2

        # Cold-tail candidate layer performs AT MOST 1 batched norm sync for the entire layer
        cold_eids = [e for e in range(8) if e not in hot_eids][:2]
        idx_mix = torch.tensor([hot_eids[:2] + cold_eids], dtype=torch.long, device=dev)
        w_mix = torch.tensor([[0.55, 0.35, 0.06, 0.04]], dtype=torch.float32, device=dev)
        syncs_mid = ov.decode_norm_syncs_performed
        y_mix = cache.forward_layer_moe(0, x_dec, idx_mix, w_mix, F.silu)
        assert torch.isfinite(y_mix).all()
        assert (ov.decode_norm_syncs_performed - syncs_mid) <= 1

        # 3. M0 carve_trunk_mirror_q1 stops early cleanly on low-rank/structured weight plateau
        u = torch.randn((128, 4), generator=gen, dtype=torch.float32) * 0.25
        v = torch.randn((4, 128), generator=gen, dtype=torch.float32) * 0.25
        w_struct = (u @ v).to(device=dev, dtype=torch.bfloat16)
        _t, _m, k_opt, rem_rel = carve_trunk_mirror_q1(
            w_struct, 7, min_cubes=3, mirror_tol=0.05, root_bits=8, group_size=64, use_fwht=True, sinkhorn_iters=2
        )
        assert 3 <= k_opt < 7
        assert rem_rel <= 0.125
    finally:
        cache.close()
