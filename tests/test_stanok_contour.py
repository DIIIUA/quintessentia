"""Tests for the Unified Stanok Containment Contour (`src/stanok_contour.py`) and
consolidation of IC-01..IC-07 (single NVML sensor, single A->B->C(0)->A relay,
shared Mirror x UNI stabilizer, snapshot admission guard, UnifiedProbabilityBus,
and Method 1 -> Method 2 + Silicon bridges).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from quintessentia_cli import gpu_memory
from src.binary_halving_q1 import LazyMoEBinaryHalvingCache
from src.pure_mirror_agents_nvrtc import (
    EQ_EQUAL,
    EQ_SPLIT,
    EQ_TENSION,
    EQ_UNKNOWN,
    PureMirrorCudaFeedback,
    _eval_4x4_mirror_cube_cpu,
    launch_pure_mirror_line_step,
)
from src.stanok_contour import (
    AgentRingMutualFeedback,
    MirrorUniStabilizer,
    OnlineGraphifyTopology,
    SpiralSphereMutualFeedback,
    StanokContainmentContour,
    UNI_DELTA,
    UNI_LAMBDA,
    UNI_OMEGA,
    UNI_PHI,
    UNI_PSI,
    UNI_SIGMA,
    UnifiedProbabilityBus,
    get_method1_survey_tools,
    get_unified_silicon_registry,
    load_cube_blueprint,
)


def test_mirror_uni_stabilizer_canonical_constants_and_rest_law():
    assert abs(UNI_PSI - 16.49539308) < 1e-6
    assert abs(UNI_PHI - 0.22313016) < 1e-6
    assert abs(UNI_OMEGA - 1.57079633) < 1e-6
    assert abs(UNI_LAMBDA - 0.00518744) < 1e-6
    assert abs(UNI_SIGMA - 1.77245385) < 1e-6
    assert abs(UNI_DELTA - 1.83907153) < 1e-6

    line = torch.randn(3, 64, dtype=torch.float32)
    invs = MirrorUniStabilizer.compute_line_invariants(line, [0, 2])
    assert len(invs) == 2
    assert all(norm > 0.0 and 0.0 < lam <= 1.0 for norm, lam in invs)

    # Converged tiny defect after rest_min -> rests!
    rested, defect, eff_tol = MirrorUniStabilizer.evaluate_token_rest(
        line_norm=10.0, lam_v=0.25, last_obs_norm=0.2, prev_obs_norm=0.3,
        last_w=0.20, next_w=0.05, rem_w=0.10, last_ortho=0.1,
        tok_done=3, rest_min=3, rest_tol=0.24,
    )
    assert rested is True and defect < eff_tol

    # High orthogonal novelty + large step -> wakes!
    rested2, defect2, eff_tol2 = MirrorUniStabilizer.evaluate_token_rest(
        line_norm=2.0, lam_v=0.80, last_obs_norm=1.5, prev_obs_norm=1.4,
        last_w=0.20, next_w=0.18, rem_w=0.45, last_ortho=0.95,
        tok_done=3, rest_min=3, rest_tol=0.24,
    )
    assert rested2 is False and defect2 >= eff_tol2


def test_stanok_contour_deadband_and_snapshot_admission_guard(monkeypatch):
    GiB = 1024**3
    MiB = 1024**2
    contour = StanokContainmentContour(
        device="cpu",
        enabled=False,
        safety_reserve_bytes=768 * MiB,
        expand_margin_bytes=320 * MiB,
    )
    # Simulate CUDA containment logic deterministically without needing external VRAM pressure
    contour.device = torch.device("cuda:0")
    contour.enabled = True
    state = {"free": 1000 * MiB, "pool": 0}
    monkeypatch.setattr(
        contour,
        "query_memory_snapshot",
        lambda: {
            "source": "nvml",
            "free_bytes": state["free"],
            "total_bytes": 8 * GiB,
            "torch_reserved_bytes": 2 * GiB + state["pool"],
            "torch_allocated_bytes": 2 * GiB,
            "torch_pool_free_bytes": state["pool"],
            "effective_free_bytes": state["free"] + state["pool"],
        },
    )
    monkeypatch.setattr(contour, "query_physical_vram", lambda: (state["free"], 8 * GiB, "nvml"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    # 1. Inside deadband (768 MiB <= 1000 MiB <= 1088 MiB): HANDS-OFF HOLD (0 ping-pong!)
    action, target, emerg = contour.evaluate_budget(cached_bytes=1500 * MiB, current_budget_bytes=1600 * MiB)
    assert action == "hold" and target == 1600 * MiB and emerg is False
    assert contour.holds_in_deadband == 1

    # 2. Genuine headroom above deadband (free = 3000 MiB): smooth EXPAND
    state["free"] = 3000 * MiB
    action, target, emerg = contour.evaluate_budget(cached_bytes=1500 * MiB, current_budget_bytes=1600 * MiB)
    assert action == "expand" and target > 1600 * MiB and emerg is False
    assert contour.expansions == 1

    # 3. Sudden pressure below outer wall (free = 300 MiB < 768 MiB reserve): EMERGENCY CONTRACT!
    state["free"] = 300 * MiB
    action, target, emerg = contour.evaluate_budget(cached_bytes=1500 * MiB, current_budget_bytes=2200 * MiB)
    assert action == "contract" and target < 1500 * MiB and emerg is True
    assert contour.contractions == 1

    # 4. Snapshot Admission Guard (IC-07): admit 200 MiB when 2 GiB free; DENY 2.63 GiB snapshot!
    state["free"] = 2048 * MiB
    state["pool"] = 0
    assert contour.can_admit_bytes(200 * MiB) is True
    assert contour.can_admit_bytes(int(2.63 * GiB)) is False
    st = contour.stats()
    assert st["snapshot_admissions"] == 1
    assert st["snapshot_denials"] == 1

    # 5. Phase tracking
    for _ in range(10):
        contour.record_layer_phase(fixated=True, skipped_experts=3)
    assert contour.current_phase == "FIXATION_SPIKE"
    contour.close()


def test_unified_probability_bus_propagates_topic_shift():
    contour = StanokContainmentContour(device="cpu", enabled=False)
    bus = UnifiedProbabilityBus(contour)

    pulses = []

    class _FakeOverlay:
        def on_topic_shift(self, layer=None):
            pulses.append(layer)

    class _FakeTracker:
        def __init__(self):
            self.probability_bus = None

        def trigger(self):
            if self.probability_bus is not None:
                self.probability_bus.notify_topic_shift(layer=12, reason="route_revision")

    tr = _FakeTracker()
    ov = _FakeOverlay()
    bus.bind(initial_anchor=object(), survivor_tracker=tr, route_exclusion=object(), imprint_overlay=ov)
    tr.trigger()

    assert pulses == [12]
    assert contour.topic_shift_pulses == 1
    bst = bus.stats()
    assert all(bst["wired_organs"].values())
    assert bst["topic_shifts_propagated"] == 1
    contour.close()


def test_method1_bridge_and_unified_silicon_registry(tmp_path):
    bp_file = tmp_path / "bp.json"
    bp_file.write_text(
        json.dumps({
            "root_bits": 16,
            "layer_proj_cubes": {"0": {"q_proj": 6, "down_proj": 5}},
            "layer_expert_cubes": {"0": {"7": {"gate_up": 5, "down": 4}}},
        }),
        encoding="utf-8",
    )
    proj_bp, exp_bp, max_d = load_cube_blueprint(bp_file, expected_root_bits=16)
    assert proj_bp == {0: {"q_proj": 6, "down_proj": 5}}
    assert exp_bp == {0: {7: {"gate_up": 5, "down": 4}}}
    assert max_d == 6

    with pytest.raises(ValueError, match="complete tree"):
        load_cube_blueprint(bp_file, expected_root_bits=8)

    survey_tools = get_method1_survey_tools()
    assert {"QuintessentiaBlueprint", "QuintessentiaCubeComparator", "StreamingCubeSurvey", "projection_prefix_errors"} <= set(survey_tools)

    reg = get_unified_silicon_registry()
    assert {
        "q1_multiplane_gemv",
        "mirror_equality_q1_gemv",
        "mirror_uni_binding_q1_gemv",
        "autonomous_fractal_rings",
        "micro_agent_group64_reader",
        "fractal_3d_swiglu_mlp",
        "qwen_moe_fabric",
    } <= set(reg)


def test_single_relay_no_duplicate_threads_when_imprint_and_relay_both_enabled():
    """IC-01B regression: enabling both `enable_expert_relay=True` and `imprint_puzzle=True`
    MUST create ONLY ONE `Q1RoleCycleArena` and ONE worker pool (no duplicate leaked relay!).
    """
    h, inter = 64, 64
    bank = {
        (0, e): (
            torch.randn(2 * inter, h, dtype=torch.bfloat16) * 0.04,
            torch.randn(h, inter, dtype=torch.bfloat16) * 0.04,
        )
        for e in range(4)
    }
    contour = StanokContainmentContour(device="cpu", enabled=False)
    cache = LazyMoEBinaryHalvingCache(
        lambda li, e: bank[(li, e)],
        num_layers=1,
        num_experts=4,
        device=torch.device("cpu"),
        store_cubes=8,
        root_bits=8,
        max_cache_bytes=16 * 1024**2,
        enable_expert_relay=True,
        imprint_puzzle=True,
        imprint_depths=(4, 3, 2, 2),
        stanok_contour=contour,
        fixation_cov_thresh=0.72,
    )
    try:
        # MUST NOT double-allocate `_expert_relay` or `_expert_relay_pool`!
        assert cache._expert_relay is None
        assert cache._expert_relay_pool is None
        assert cache._imprint_overlay is not None
        assert cache._imprint_overlay.contour is contour
        assert cache._imprint_overlay.fixation_cov_thresh == pytest.approx(0.72)

        x = torch.randn(1, h, dtype=torch.bfloat16)
        idx = torch.tensor([[0, 1, 2]], dtype=torch.long)
        w = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.bfloat16)
        y = cache.forward_layer_moe(0, x, idx, w, F.silu)
        assert y.shape == (1, h) and bool(torch.isfinite(y).all())

        st = cache.stats()
        # Unified stats report the single active relay under both `imprint_puzzle.relay` and `expert_relay`
        assert st["expert_relay"] is not None
        assert st["expert_relay"]["transitions"]["B->A"] == 0
        assert st["expert_relay"]["standby_bytes"] == 0
        assert st["expert_relay"]["rotations"] == st["imprint_puzzle"]["relay"]["rotations"] > 0
    finally:
        cache.close()
        contour.close()


def test_physical_vram_monitor_uuid_matching(monkeypatch):
    """IC-01A regression: `PhysicalVRAMMonitor` matches NVML by CUDA UUID, never bare ordinal."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (6 * 1024**3, 8 * 1024**3))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: SimpleNamespace(uuid="GPU-TEST-UUID"))
    seen_uuids = []

    def _fake_open(uuid_str):
        seen_uuids.append(uuid_str)
        return None  # simulate NVML dll absent -> clean CUDA fallback

    monkeypatch.setattr(gpu_memory.PhysicalVRAMMonitor, "_try_open_nvml_by_uuid", staticmethod(_fake_open))
    mon = gpu_memory.PhysicalVRAMMonitor("cuda:0")
    try:
        assert seen_uuids == ["GPU-TEST-UUID"]
        fb, tb, src = mon.query()
        assert fb == 6 * 1024**3 and tb == 8 * 1024**3 and src == "cuda"
    finally:
        mon.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stanok_contour_live_cuda_nvml_uuid_and_topic_shift_equilibration():
    """End-to-end CUDA test of UUID-matched PhysicalVRAMMonitor + shared StanokContainmentContour
    + single A->B->C(0)->A relay + UnifiedProbabilityBus topic-shift equilibration pulse.
    """
    dev = torch.device("cuda:0")
    contour = StanokContainmentContour(device=dev, enabled=True)
    bus = UnifiedProbabilityBus(contour)
    h, inter = 256, 128
    bank = {
        (0, e): (
            torch.randn(2 * inter, h, dtype=torch.bfloat16) * 0.04,
            torch.randn(h, inter, dtype=torch.bfloat16) * 0.04,
        )
        for e in range(8)
    }
    disk_calls = []

    def reader(li, eid):
        disk_calls.append((li, eid))
        return bank[(li, eid)]

    cache = LazyMoEBinaryHalvingCache(
        reader,
        num_layers=1,
        num_experts=8,
        device=dev,
        store_cubes=16,
        root_bits=16,
        max_cache_bytes=16 * 1024**2,
        enable_expert_relay=True,
        imprint_puzzle=True,
        imprint_depths=(5, 4, 3, 2),
        expert_rest_tol=0.0,
        imprint_auto_vram=True,
        stanok_contour=contour,
        fixation_cov_thresh=0.75,
    )
    try:
        # Real NVML UUID sensor check
        fb, tb, src = contour.query_physical_vram()
        assert tb > 0 and 0 <= fb <= tb and src in ("nvml", "cuda")
        # Zero duplicate relays when both flags are set!
        assert cache._expert_relay is None and cache._expert_relay_pool is None
        ov = cache._imprint_overlay
        assert ov is not None and ov.contour is contour
        bus.bind(imprint_overlay=ov)

        x = torch.randn(1, h, device=dev, dtype=torch.bfloat16)
        w = torch.tensor([[0.40, 0.30, 0.20, 0.10]], device=dev, dtype=torch.bfloat16)

        # Warm experts 0, 1, 2, 3
        cache.forward_layer_moe(0, x, torch.tensor([[0, 1, 2, 3]], device=dev), w, F.silu)
        torch.cuda.synchronize()
        reads_warm = len(disk_calls)
        assert reads_warm == 4

        # Step 1: Fixated (0,1,2 resident = 90% >= 75%, expert 6 cold) -> 0 disk reads!
        cache.forward_layer_moe(0, x, torch.tensor([[0, 1, 2, 6]], device=dev), w, F.silu)
        assert len(disk_calls) == reads_warm
        assert contour.fixation_layers >= 1

        # Step 2: UnifiedProbabilityBus signals topic shift -> fixation steps aside and equilibrates expert 6!
        bus.notify_topic_shift(layer=0, reason="topic_change_test")
        assert contour.topic_shift_pulses == 1
        assert ov.topic_shift_equilibrations == 1
        cache.forward_layer_moe(0, x, torch.tensor([[0, 1, 2, 6]], device=dev), w, F.silu)
        torch.cuda.synchronize()
        assert len(disk_calls) == reads_warm + 1, "Topic shift must trigger equilibration read of new boundary piece!"
        st = cache.stats()
        assert st["expert_relay"]["transitions"]["B->A"] == 0
        assert st["expert_relay"]["standby_bytes"] == 0
    finally:
        cache.close()
        contour.close()
        torch.cuda.empty_cache()


def test_self_collapsing_lattice_is_model_agnostic_across_top2_and_top8():
    """Verify ZERO per-model tuning across wildly different MoE geometries (`4-exp top-2` vs
    `64-exp top-8`): with default `fixation_cov_thresh=1.0` (NO hardcoded 0.72 switch!) and
    plentiful memory (never hitting VRAM wall), the coupled lattice:
      1. expands on demand when chaos/novelty arrives,
      2. naturally settles into zero-disk spikes via Mirror x UNI on resident intersections,
      3. self-collapses fading uncoupled nodes from inside (`lattice_self_collapses > 0`).
    """
    torch.manual_seed(777)
    h, inter = 64, 64
    for num_exp, top_k in ((4, 2), (64, 8)):
        bank = {
            (0, e): (
                torch.randn(2 * inter, h, dtype=torch.bfloat16) * 0.04,
                torch.randn(h, inter, dtype=torch.bfloat16) * 0.04,
            )
            for e in range(num_exp)
        }
        disk_calls = []

        def reader(li, eid, _bank=bank, _calls=disk_calls):
            _calls.append((li, eid))
            return _bank[(li, eid)]

        contour = StanokContainmentContour(device="cpu", enabled=False)
        cache = LazyMoEBinaryHalvingCache(
            reader,
            num_layers=1,
            num_experts=num_exp,
            device=torch.device("cpu"),
            store_cubes=8,
            root_bits=8,
            max_cache_bytes=256 * 1024**2,  # plentiful memory: outer wall is NEVER touched!
            enable_expert_relay=True,
            imprint_puzzle=True,
            imprint_depths=(5, 4, 3, 2),
            expert_rest_tol=0.24,
            stanok_contour=contour,
            fixation_cov_thresh=1.0,  # NO central 0.72 switch!
        )
        try:
            x = torch.randn(1, h, dtype=torch.bfloat16)
            # Phase 1 (Topic A): warm up expert set A with strong weights -> gets depth > min_shed_k
            eids_a = list(range(top_k))
            w_a = torch.softmax(torch.linspace(3.0, 0.2, top_k), dim=0).unsqueeze(0).to(torch.bfloat16)
            for _ in range(3):
                cache.forward_layer_moe(0, x, torch.tensor([eids_a], dtype=torch.long), w_a, F.silu)

            # Phase 2 (Topic B): shift to completely disjoint active lattice set B (leaves set A fading!)
            eids_b = list(range(top_k, min(num_exp, 2 * top_k)))
            if len(eids_b) < top_k:
                eids_b = list(range(num_exp - top_k, num_exp))
            # First step on B expands naturally; subsequent steps settle on resident B (0 disk wakes!)
            for _ in range(12):
                cache.forward_layer_moe(0, x, torch.tensor([eids_b], dtype=torch.long), w_a, F.silu)

            ov = cache._imprint_overlay
            st = ov.stats()
            # Outer wall never touched, yet fading Topic A nodes collapsed from INSIDE the lattice!
            assert st["depth_sheds"] == 0
            assert st["lattice_self_collapses"] > 0, f"Expected internal lattice self-collapse for {(num_exp, top_k)}: {st}"
            assert contour.lattice_self_collapses == st["lattice_self_collapses"]
            assert st["fixation_layer_skips"] > 0, f"Expected emergent zero-disk fixation for {(num_exp, top_k)}: {st}"
            assert st["relay"]["transitions"]["B->A"] == 0
            assert st["relay"]["standby_bytes"] == 0
        finally:
            cache.close()
            contour.close()


def test_loop_centric_gating_skips_dangling_non_loop_agents_and_wakes_closed_loops():
    """Verify Anton's loop-centric principle:
      1. When resident nodes already close a dominant loop on `line_spine`, a dangling cold agent
         with < 2 loop contacts CANNOT create a new loop/intersection -> skipped (`0` disk reads,
         `non_loop_branches_skipped > 0`), invariant to expert ID ordering!
      2. When a cold candidate DOES close a real loop (>= 2 active face/90°-corner contacts) AND
         breaks loop balance, it wakes via A->B->C(0)->A.
      3. When the resident loop is genuinely broken/open (`w_done < rem_w`), cold nodes always wake.
    """
    torch.manual_seed(20261122)
    h, inter = 64, 64
    num_layers, num_exp = 2, 12
    bank = {
        (li, e): (
            torch.randn(2 * inter, h, dtype=torch.bfloat16) * 0.04,
            torch.randn(h, inter, dtype=torch.bfloat16) * 0.04,
        )
        for li in range(num_layers)
        for e in range(num_exp)
    }
    disk_calls = []

    def reader(li, eid):
        disk_calls.append((li, eid))
        return bank[(li, eid)]

    contour = StanokContainmentContour(device="cpu", enabled=False)
    cache = LazyMoEBinaryHalvingCache(
        reader,
        num_layers=num_layers,
        num_experts=num_exp,
        device=torch.device("cpu"),
        store_cubes=8,
        root_bits=8,
        max_cache_bytes=64 * 1024**2,
        enable_expert_relay=True,
        imprint_puzzle=True,
        imprint_depths=(4, 3, 2, 2),
        expert_rest_tol=0.24,
        expert_rest_min=3,
        stanok_contour=contour,
        fixation_cov_thresh=1.0,  # Pure loop-centric gating — ZERO hardcoded 0.72 switch!
    )
    try:
        x = torch.randn(1, h, dtype=torch.bfloat16)
        w_base = torch.tensor([[0.35, 0.25, 0.18, 0.12, 0.06, 0.04]], dtype=torch.bfloat16)

        # Warm a real closed 3D loop across layers 0 -> 1 on experts [0, 1, 2, 3, 4, 5]
        for li in range(num_layers):
            cache.forward_layer_moe(li, x, torch.tensor([[0, 1, 2, 3, 4, 5]]), w_base, F.silu)
        reads_after_warm = len(disk_calls)
        ov = cache._imprint_overlay

        # Case 1: Resident loop [0, 1, 2] carries 0.78 (>= 0.50), cold dangling agents [9, 10, 11]
        # have ZERO face/corner90 contacts -> CANNOT create a new loop -> 0 disk reads!
        # Notice we put a tiny resident expert `2` with highest ID last to prove order-invariance too!
        w_dangling = torch.tensor([[0.40, 0.28, 0.10, 0.09, 0.07, 0.06]], dtype=torch.bfloat16)
        cache.forward_layer_moe(0, x, torch.tensor([[0, 1, 2, 9, 10, 11]]), w_dangling, F.silu)
        assert len(disk_calls) == reads_after_warm, "Dangling non-loop cold agents must NOT wake disk!"
        assert ov.non_loop_branches_skipped >= 3

        # Case 2: Broken/open loop (resident expert `0` has only 0.15 < remaining 0.85) ->
        # MUST wake dominant cold agents so a genuinely new topic can spawn new loops!
        w_new_topic = torch.tensor([[0.45, 0.25, 0.15, 0.07, 0.05, 0.03]], dtype=torch.bfloat16)
        cache.forward_layer_moe(1, x, torch.tensor([[9, 10, 0, 11, 7, 8]]), w_new_topic, F.silu)
        assert len(disk_calls) > reads_after_warm, "Broken/open loop must wake dominant cold nodes!"
        st = ov.stats()
        assert st["relay"]["transitions"]["B->A"] == 0
        assert st["relay"]["standby_bytes"] == 0
    finally:
        cache.close()
        contour.close()


def test_online_graphify_topology_communities_bridges_and_speed():
    """Verify fast deterministic OnlineGraphifyTopology:
      1. Cleanly splits two dense cliques into separate communities (Q > 0.25);
      2. Identifies the modest-weight bridge node linking both communities (`role == "bridge"`);
      3. Classifies weak dangling tail as high-redundancy periphery (`bridge_strength == 0.0`);
      4. Scales to a 40-layer x 20-active-expert (800-node) Qwen3.6-35B-sized lattice in milliseconds!
    """
    topo = OnlineGraphifyTopology(gamma=1.0, max_iters=6)
    adj = {}

    def _add(u, v, w):
        adj.setdefault(u, {})[v] = adj.get(u, {}).get(v, 0.0) + w
        adj.setdefault(v, {})[u] = adj.get(v, {}).get(u, 0.0) + w

    # Dense Community A: (0,0)..(0,3)
    for a in range(4):
        for b in range(a + 1, 4):
            _add((0, a), (0, b), 1.0)
    # Dense Community B: (0,4)..(0,7)
    for a in range(4, 8):
        for b in range(a + 1, 8):
            _add((0, a), (0, b), 1.0)
    # Inter-community bridge (0, 8) with modest edges to A0 and B0
    _add((0, 8), (0, 0), 0.4)
    _add((0, 8), (0, 4), 0.4)
    # Weak dangling periphery tail (0, 9) attached only to A1
    _add((0, 9), (0, 1), 0.05)

    st = topo.refresh(adj, cold_start=True)
    assert st["communities"] == 2
    assert st["modularity_q"] > 0.25
    assert topo.community_of[(0, 0)] != topo.community_of[(0, 4)]

    role_m, _core_m, bridge_m, _red_m, _cid_m = topo.node_feedback((0, 8))
    assert role_m == "bridge" and bridge_m >= UNI_PHI
    assert topo.bridges_active_community((0, 8), [(0, 0), (0, 1)]) is True

    role_a2, core_a2, bridge_a2, _red_a2, _cid_a2 = topo.node_feedback((0, 2))
    assert role_a2 == "core" and core_a2 >= UNI_PHI and bridge_a2 == 0.0

    role_x, _core_x, bridge_x, red_x, _cid_x = topo.node_feedback((0, 9))
    assert role_x == "periphery" and bridge_x == 0.0 and red_x > 0.5

    snap = topo.export_graph_snapshot(adj)
    assert len(snap["nodes"]) == 10
    assert any(edge["inter_community"] for edge in snap["links"])

    # Speed benchmark on a realistic 40-layer x 20-node (800-node, ~2400-edge) 35B lattice
    big_adj = {}

    def _add_big(u, v, w):
        big_adj.setdefault(u, {})[v] = big_adj.get(u, {}).get(v, 0.0) + w
        big_adj.setdefault(v, {})[u] = big_adj.get(v, {}).get(u, 0.0) + w

    for li in range(40):
        # Two 10-node communities per layer connected across layers + 1 bridge
        for base in (0, 10):
            for i in range(4):
                _add_big((li, base + i), (li, base + (i + 1) % 10), 0.8)
                _add_big((li, base + i), (li, base + (i + 2) % 10), 0.5)
            for i in range(4, 10):
                _add_big((li, base + i), (li, base + (i + 1) % 10), 0.4)
        _add_big((li, 0), (li, 10), 0.25)
        if li > 0:
            for e in range(0, 20, 2):
                _add_big((li - 1, e), (li, e), 0.6)

    topo_big = OnlineGraphifyTopology(gamma=1.0, max_iters=6)
    st_cold = topo_big.refresh(big_adj, cold_start=True)
    st_warm = topo_big.refresh(big_adj, cold_start=False)
    assert st_cold["nodes"] == 800
    assert st_cold["communities"] >= 2
    assert st_cold["last_refresh_ms"] < 100.0
    assert st_warm["last_refresh_ms"] <= st_cold["last_refresh_ms"] + 10.0


def test_ab_v1_loop_local_vs_v2_graphify_feedback_on_prefill_and_decode():
    """Direct side-by-side A/B verification of:
      - Version 1 (`graphify_feedback=False`): local 1-hop loop-centric lattice (clean baseline)
      - Version 2 (`graphify_feedback=True`): embedded OnlineGraphifyTopology feedback inside
        prefill, probabilistic cube depth allocation, VRAM shedding protection, and decode loop!
    """
    torch.manual_seed(20261122)
    h, inter = 64, 64
    num_layers, num_exp = 2, 14
    bank = {
        (li, e): (
            torch.randn(2 * inter, h, dtype=torch.bfloat16) * 0.04,
            torch.randn(h, inter, dtype=torch.bfloat16) * 0.04,
        )
        for li in range(num_layers)
        for e in range(num_exp)
    }

    m_pre = 10
    # Construct two distinct semantic topic directions (Topic A vs orthogonal Topic B) plus bridge tokens
    dir_a = torch.zeros(h, dtype=torch.float32)
    dir_a[: h // 2] = 1.0
    dir_b = torch.zeros(h, dtype=torch.float32)
    dir_b[h // 2 :] = 1.0
    noise = torch.randn(m_pre, h, dtype=torch.float32) * 0.12
    x_pre_f = torch.empty((m_pre, h), dtype=torch.float32)
    for ti in range(4):
        x_pre_f[ti] = dir_a + noise[ti]
    x_pre_f[4] = 0.75 * dir_a + 0.25 * dir_b + noise[4]
    x_pre_f[5] = 0.25 * dir_a + 0.75 * dir_b + noise[5]
    for ti in range(6, 10):
        x_pre_f[ti] = dir_b + noise[ti]
    x_pre = x_pre_f.to(torch.bfloat16)
    x_dec = (dir_a + torch.randn(1, h, dtype=torch.float32) * 0.10).to(torch.bfloat16)

    def _run_version(graphify_on: bool):
        disk_calls = []

        def reader(li, eid):
            disk_calls.append((li, eid))
            return bank[(li, eid)]

        contour = StanokContainmentContour(device="cpu", enabled=False)
        bus = UnifiedProbabilityBus(contour)
        cache = LazyMoEBinaryHalvingCache(
            reader,
            num_layers=num_layers,
            num_experts=num_exp,
            device=torch.device("cpu"),
            store_cubes=8,
            root_bits=8,
            max_cache_bytes=64 * 1024**2,
            enable_expert_relay=True,
            imprint_puzzle=True,
            imprint_depths=(5, 4, 3, 2),
            expert_rest_tol=0.24,
            expert_rest_min=3,
            stanok_contour=contour,
            fixation_cov_thresh=1.0,
            graphify_feedback=graphify_on,
        )
        try:
            ov = cache._imprint_overlay
            bus.bind(imprint_overlay=ov)

            # Prefill prompt spanning two distinct communities (A: 0..3, B: 4..7) stitched by
            # modest-weight bridge expert `8`, plus weak isolated/tail experts `10..13`.
            idx_rows = [
                # 4 tokens solidly in Community A (0..3) + weak noise tails 10, 11
                [0, 1, 2, 3, 10, 11],
                [1, 0, 3, 2, 10, 12],
                [2, 3, 0, 1, 11, 13],
                [3, 2, 1, 0, 12, 13],
                # 2 bridge tokens stitching Community A and Community B via modest-weight bridge `8`
                [0, 1, 8, 2, 10, 11],
                [4, 5, 8, 6, 12, 13],
                # 4 tokens solidly in Community B (4..7) + weak noise tails
                [4, 5, 6, 7, 10, 11],
                [5, 4, 7, 6, 11, 12],
                [6, 7, 4, 5, 12, 13],
                [7, 6, 5, 4, 10, 13],
            ]
            w_rows = [[0.34, 0.28, 0.16, 0.14, 0.05, 0.03] for _ in range(m_pre)]
            idx_t = torch.tensor(idx_rows, dtype=torch.long)
            w_t = torch.tensor(w_rows, dtype=torch.bfloat16)

            outs_pre = []
            for li in range(num_layers):
                y = cache.forward_layer_moe(li, x_pre, idx_t, w_t, F.silu)
                assert bool(torch.isfinite(y).all())
                outs_pre.append(y)

            # Check probabilistic depth choice for the modest-weight bridge `(0, 8)` vs weak tail `(0, 10)`
            bridge_depth = ov.graph.target_depth(
                (0, 8), batch_weight_sum=0.14, batch_token_count=1, batch_max_weight=0.40, root_bits=8
            )
            # VRAM shedding score MUST remain pure local physics (Graphify never meddles outside agents/coalitions!)
            assert ov.graph.score((0, 8)) == pytest.approx(ov.graph.relative_binding((0, 8)))
            bridge_role = ov.graph.topology.role_of.get((0, 8)) if ov.graph.topology is not None else None

            # Trigger topic shift via UnifiedProbabilityBus and run decode steps
            bus.notify_topic_shift(layer=0, reason="ab_topic_shift")
            y_dec = cache.forward_layer_moe(
                0,
                x_dec,
                torch.tensor([[0, 1, 2, 8, 10, 11]], dtype=torch.long),
                torch.tensor([[0.36, 0.28, 0.18, 0.10, 0.05, 0.03]], dtype=torch.bfloat16),
                F.silu,
            )
            assert bool(torch.isfinite(y_dec).all())
            st = ov.stats()
            assert st["relay"]["transitions"]["B->A"] == 0
            assert st["relay"]["standby_bytes"] == 0
            return {
                "stats": st,
                "disk_calls": len(disk_calls),
                "bridge_depth": bridge_depth,
                "bridge_role": bridge_role,
                "out_pre": outs_pre[0],
                "out_dec": y_dec,
            }
        finally:
            cache.close()
            contour.close()

    res_v1 = _run_version(False)
    res_v2 = _run_version(True)

    # Version 1: clean baseline without Graphify topology
    assert res_v1["stats"]["graphify_feedback"]["enabled"] is False
    assert res_v1["stats"]["graphify_feedback"]["topology"] is None

    # Version 2: zero-delay 3D Spiral-Sphere Overlap Lenses & continuous topic-shift coils!
    gf2 = res_v2["stats"]["graphify_feedback"]
    assert gf2["enabled"] is True
    assert gf2["topology"] is not None
    assert gf2["topology"]["mode"] == "spiral_sphere_mutual"
    assert gf2["topology"]["total_lenses_formed"] > 0
    assert gf2["topology"]["topic_shift_continuations"] >= 1
    assert gf2["topology"]["mutual_steps"] >= 3
    assert gf2["topology"]["avg_refresh_ms"] < 5.0
    assert res_v2["bridge_role"] == "sphere_intersection"
    assert gf2["bridges_protected"] > 0
    # Sphere-intersection agent `(0, 8)` bridging disjoint Topics A & B gets protected >= mid_k (4)!
    assert res_v2["bridge_depth"] >= 4
    assert res_v2["bridge_depth"] >= res_v1["bridge_depth"]
    # Outputs remain closely aligned in direction (no destabilization!)
    cos_pre = float(F.cosine_similarity(res_v1["out_pre"].float().flatten(), res_v2["out_pre"].float().flatten(), dim=0).item())
    assert cos_pre > 0.97, f"Unexpected divergence between V1 and V2 prefill outputs: cos={cos_pre}"


def test_agent_ring_mutual_feedback_minimum_rings_and_continuation():
    """Verify pure Agent <-> Ring mutual feedback (NO coalitions):
      1. Closed 3-agent minimum `(0, 1, 2)` generates real flat rings (`min_rc == 1`, `min_re > 0`);
      2. A cold candidate attached to unlinked nodes (no closed triangle!) yields `delta_rc == 0` -> `"no_new_loop"`;
      3. A cold candidate closing a real flat or 90° ring yields `delta_rc > 0` and scales continuation
         against `delta_re / min_re` (strong new ring expands; tiny ripple settles);
      4. An agent sitting at the intersection of two disjoint rings `(8, 0, 1)` & `(8, 4, 5)` is recognized
         purely as `"ring_intersection"` in microseconds!
    """
    rf = AgentRingMutualFeedback(num_layers=2)
    face_edges = {}
    corner90_edges = {}

    def _face(li, a, b, w):
        face_edges.setdefault((li, a), {})[b] = w
        face_edges.setdefault((li, b), {})[a] = w

    def _c90(li_prev, p, c, w):
        corner90_edges.setdefault((li_prev, p), {})[c] = w

    # Ring 1 in layer 0: closed triangle (0, 1, 2) + ring (8, 0, 1)
    _face(0, 0, 1, 0.8)
    _face(0, 1, 2, 0.8)
    _face(0, 0, 2, 0.8)
    _face(0, 8, 0, 0.6)
    _face(0, 8, 1, 0.6)
    # Ring 2 in layer 0: disjoint ring (8, 4, 5) touching ring 1 ONLY at intersection agent `8`
    _face(0, 4, 5, 0.8)
    _face(0, 8, 4, 0.6)
    _face(0, 8, 5, 0.6)
    # Dangling candidate `9` linked ONLY to unlinked nodes (2 and 4 — no edge 2-4 -> NO closed ring!)
    _face(0, 9, 2, 0.5)
    _face(0, 9, 4, 0.5)
    # Inter-layer 90° ring from layer 0 (0, 1) -> layer 1 (0, 1)
    _face(1, 0, 1, 0.7)
    _c90(0, 0, 0, 0.7)
    _c90(0, 0, 1, 0.7)
    _c90(0, 1, 0, 0.7)

    n0 = rf.observe_executed_rings(
        0, [[(0, 0.3), (1, 0.3), (2, 0.2), (8, 0.2)], [(8, 0.35), (4, 0.35), (5, 0.3)]], None, face_edges, corner90_edges
    )
    n1 = rf.observe_executed_rings(
        1, [[(0, 0.5), (1, 0.5)]], [[(0, 0.5), (1, 0.5)]], face_edges, corner90_edges
    )
    assert n0 >= 3 and n1 >= 2
    assert rf.stats()["last_refresh_ms"] < 2.0

    # Agent `8` sits on two disjoint rings -> pure ring_intersection!
    role_8, rel_8, inter_8, has_r0 = rf.agent_ring_feedback((0, 8))
    assert has_r0 is True and role_8 == "ring_intersection" and rel_8 >= UNI_PHI and inter_8 >= UNI_PHI

    # Compare minimum (0, 1, 2) against dangling `9` (no edge 2-4 -> ZERO new rings!)
    min_rc, min_re, d_rc9, d_re9 = rf.compare_minimum_and_continuation(0, [0, 1, 2], [], 9, face_edges, corner90_edges)
    assert min_rc == 1 and min_re > 0.0
    assert d_rc9 == 0 and d_re9 == 0.0
    rest9, _, _, reason9 = MirrorUniStabilizer.evaluate_loop_continuation(
        line_norm=5.0, lam_v=0.3, last_obs_norm=0.5, prev_obs_norm=0.5,
        w_done=0.75, next_w=0.15, rem_w=0.25, last_ortho=0.5,
        tok_done=3, rest_min=3, rest_tol=0.24, closes_new_loop=True,
        min_ring_count=min_rc, min_ring_energy=min_re, delta_ring_count=d_rc9, delta_ring_energy=d_re9,
    )
    assert rest9 is True and reason9 == "no_new_loop"

    # Compare minimum (0, 1, 2) against real ring-closing agent `8` (closes (8, 0, 1)!)
    min_rc2, min_re2, d_rc8, d_re8 = rf.compare_minimum_and_continuation(0, [0, 1, 2], [], 8, face_edges, corner90_edges)
    assert min_rc2 == 1 and d_rc8 == 1 and d_re8 > 0.0
    # Strong new ring (d_re8 / min_re2 ~= 0.83 >= UNI_PHI) tightens eff_tol vs tiny ripple!
    _, _, tol_strong, _ = MirrorUniStabilizer.evaluate_loop_continuation(
        line_norm=5.0, lam_v=0.3, last_obs_norm=0.5, prev_obs_norm=0.5,
        w_done=0.75, next_w=0.15, rem_w=0.25, last_ortho=0.5,
        tok_done=3, rest_min=3, rest_tol=0.24, closes_new_loop=True,
        min_ring_count=min_rc2, min_ring_energy=min_re2, delta_ring_count=d_rc8, delta_ring_energy=d_re8,
    )
    _, _, tol_ripple, _ = MirrorUniStabilizer.evaluate_loop_continuation(
        line_norm=5.0, lam_v=0.3, last_obs_norm=0.5, prev_obs_norm=0.5,
        w_done=0.75, next_w=0.15, rem_w=0.25, last_ortho=0.5,
        tok_done=3, rest_min=3, rest_tol=0.24, closes_new_loop=True,
        min_ring_count=min_rc2, min_ring_energy=min_re2, delta_ring_count=1, delta_ring_energy=0.01 * min_re2,
    )
    assert tol_strong < tol_ripple


def test_spiral_sphere_mutual_feedback_overlap_lens_and_topic_shift_continuation():
    """Verify Anton's 3D Spiral-to-Sphere Overlap & Continuous Topic-Shift Coil physics:
      1. Agents sharing Topic A form real Sphere Overlap Lenses (`d < r_a + r_b`); orthogonal
         Topic B spheres pass cleanly by (`d >= r_a + r_b` -> `"no_new_loop"`);
      2. ONLY a true bridge agent spanning disjoint Topic A and Topic B spheres receives
         `"sphere_intersection"`, while intra-topic hubs remain `"sphere_core"` (no 86% bloat!);
      3. On an abrupt topic shift (`on_topic_shift` + orthogonal Topic B input), active agents
         NEVER reset or stop: `coil_steps` increments monotonically, center shifts smoothly
         bridging old and new knowledge, radius expands naturally during the transition to catch
         Topic B overlap lenses, and re-tightens once Topic B stabilizes;
      4. Sub-millisecond speed on full 40-layer x 256-expert Qwen3.6-35B geometry.
    """
    import numpy as np

    torch.manual_seed(20261122)
    h_dim = 128
    sf = SpiralSphereMutualFeedback(num_layers=40, num_experts=256, proj_dim=32)

    # Two orthogonal semantic topics in hidden space
    v_a = torch.zeros(h_dim, dtype=torch.float32)
    v_a[: h_dim // 2] = 1.0
    v_b = torch.zeros(h_dim, dtype=torch.float32)
    v_b[h_dim // 2 :] = 1.0

    # Batch spanning Topic A (0..3), bridge (4..5), and orthogonal Topic B (6..9)
    xs = []
    for _ in range(4):
        xs.append(v_a + 0.10 * torch.randn(h_dim))
    xs.append(0.75 * v_a + 0.25 * v_b + 0.08 * torch.randn(h_dim))
    xs.append(0.25 * v_a + 0.75 * v_b + 0.08 * torch.randn(h_dim))
    for _ in range(4):
        xs.append(v_b + 0.10 * torch.randn(h_dim))
    x_batch = torch.stack(xs, dim=0)
    u_batch = sf.project_states(0, x_batch, update_mean=True)

    routes = [
        [(0, 0.40), (1, 0.35), (2, 0.25)],
        [(1, 0.40), (0, 0.35), (2, 0.25)],
        [(2, 0.40), (0, 0.35), (1, 0.25)],
        [(0, 0.40), (2, 0.35), (1, 0.25)],
        [(0, 0.38), (1, 0.32), (8, 0.30)],  # bridge 8 meets Topic A
        [(4, 0.38), (5, 0.32), (8, 0.30)],  # bridge 8 meets Topic B
        [(4, 0.40), (5, 0.35), (6, 0.25)],
        [(5, 0.40), (4, 0.35), (6, 0.25)],
        [(6, 0.40), (4, 0.35), (5, 0.25)],
        [(4, 0.40), (6, 0.35), (5, 0.25)],
    ]
    lenses_n = sf.observe_spiral_step(0, routes, u_batch)
    assert lenses_n > 0

    # 1. Within-topic spheres overlap cleanly; orthogonal topic spheres do NOT overlap!
    assert sf.sphere_overlap_lens(0, 0, 1) > 0.2
    assert sf.sphere_overlap_lens(0, 4, 5) > 0.2
    assert sf.sphere_overlap_lens(0, 0, 6) == 0.0, "Orthogonal topic spheres must pass by without false overlap!"

    # 2. Bridge `8` gets `sphere_intersection`, while intra-topic hubs `0`, `4` stay `sphere_core`!
    role_8, lrel_8, brat_8, has_l = sf.agent_sphere_feedback((0, 8))
    role_0, lrel_0, brat_0, _ = sf.agent_sphere_feedback((0, 0))
    role_4, lrel_4, brat_4, _ = sf.agent_sphere_feedback((0, 4))
    assert has_l is True
    assert role_8 == "sphere_intersection" and brat_8 >= UNI_PHI
    assert role_0 == "sphere_core" and brat_0 < UNI_PHI
    assert role_4 == "sphere_core" and brat_4 < UNI_PHI

    # Comparing Topic A minimum `[0, 1, 2]` on Topic A input against disjoint Topic B sphere `6`:
    # ZERO overlap -> `"no_new_loop"`!
    u_a_query = sf.project_states(0, v_a.unsqueeze(0), update_mean=False)[0]
    min_c, min_e, d_c6, d_e6 = sf.compare_minimum_and_continuation(0, [0, 1, 2], [], 6, u_info=u_a_query)
    assert min_c >= 3 and min_e > 0.0 and d_c6 == 0 and d_e6 == 0.0
    rest6, _, _, reason6 = MirrorUniStabilizer.evaluate_loop_continuation(
        line_norm=5.0, lam_v=0.3, last_obs_norm=0.5, prev_obs_norm=0.5,
        w_done=0.75, next_w=0.15, rem_w=0.25, last_ortho=0.5,
        tok_done=3, rest_min=3, rest_tol=0.24, closes_new_loop=True,
        min_ring_count=min_c, min_ring_energy=min_e, delta_ring_count=d_c6, delta_ring_energy=d_e6,
    )
    assert rest6 is True and reason6 == "no_new_loop"

    # 3. CONTINUOUS COIL CONTINUATION ON TOPIC SHIFT (NO reset, NO stop!):
    steps_before = int(sf.coil_steps[0, 0])
    center_before = sf.centers[0, 0].copy()
    radius_before = float(sf.radii[0, 0])
    sf.on_topic_shift(layer=0)
    assert int(sf.coil_steps[0, 0]) == steps_before, "Topic shift MUST NOT reset coil_steps!"
    assert np.allclose(sf.centers[0, 0], center_before), "Topic shift MUST NOT wipe prior center!"
    assert float(sf.radii[0, 0]) == pytest.approx(radius_before), "Topic shift MUST NOT wipe radius!"

    # Abrupt Topic B token arrives while agent `0` is still active:
    u_b_query = sf.project_states(0, v_b.unsqueeze(0), update_mean=False)[0]
    # Before execution, projected next coil of `[0, 1]` reaches bridge `8`!
    _, _, d_c8, d_e8 = sf.compare_minimum_and_continuation(0, [0, 1], [], 8, u_info=u_b_query)
    assert d_c8 > 0 and d_e8 > 0.0

    # Agent `0` executes its next spiral coil directly onto Topic B knowledge:
    sf.observe_spiral_step(0, [[(0, 0.5), (8, 0.5)]], u_b_query[None, :])
    assert int(sf.coil_steps[0, 0]) == steps_before + 1
    center_mid = sf.centers[0, 0].copy()
    radius_mid = float(sf.radii[0, 0])
    # Smooth bridge between old (A) and new (B) knowledge + natural transition radius expansion!
    assert float(np.dot(center_mid, center_before)) > 0.0
    assert float(np.dot(center_mid, u_b_query)) > float(np.dot(center_before, u_b_query))
    assert radius_mid > radius_before

    # After settling for several coils in stable Topic B, center converges to B & radius re-tightens!
    for _ in range(14):
        u_b_jit = sf.project_states(0, (v_b + 0.05 * torch.randn(h_dim)).unsqueeze(0), update_mean=False)
        sf.observe_spiral_step(0, [[(0, 0.5), (4, 0.5)]], u_b_jit)
    center_settled = sf.centers[0, 0]
    cos_to_b = float(np.dot(center_settled, u_b_query) / (np.linalg.norm(center_settled) + 1e-8))
    assert cos_to_b > 0.90
    assert float(sf.radii[0, 0]) < radius_mid

    # 4. Speed check across all 40 layers of a 256-expert geometry (< 1.0 ms/step!)
    u_one = sf.project_states(1, torch.randn(1, h_dim), update_mean=False)
    for li in range(1, 40):
        sf.observe_spiral_step(li, [[(10, 0.35), (20, 0.30), (30, 0.20), (40, 0.15)]], u_one)
    st = sf.stats()
    assert st["mode"] == "spiral_sphere_mutual"
    assert st["avg_refresh_ms"] < 1.0
    assert st["topic_shift_continuations"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Pure Mirror NVRTC kernels")
def test_pure_mirror_cuda_kernels_exact_parity_and_topic_shift():
    """Verify Version 3 specialized NVRTC CUDA kernels built STRICTLY from the Irreducible
    4x4 Mirror Equality law + D15 Logic<->Strategy topology (ZERO UNI constants, ZERO reduction!):
      1. Full 4x4 cube non-commutativity (`L02`), reverse `FuturePressure` (`F14`), and all 4
         `KA-F01` statuses (`UNKNOWN`, `EQUAL`, `TENSION`, `SPLIT`);
      2. Exact parity (atol <= 1e-4) between all 4 CUDA kernels and CPU reference across ALL 6
         irreducible role slots (`roles: [L, E, 6, 32]`), `link_energy`, `bridge_ratio`, `role_ids`;
      3. Immutability of past surfaces (`left_outer=Outcome_(t-1)`, `left_inner=ObservedBehavior_(t-1)`)
         on `is_commit=False` vs materialization + D15 `Logic_t` formation on `is_commit=True`;
      4. `C03`/`L04` Law of New Observation: duplicate candidate -> `mirror_maintained` (0 B disk),
         disjoint Topic B -> `no_mirror_link` (0 B disk), living bridge -> `mirror_expand`, and
         topic shift WITHOUT reset bridges cleanly through right wing `u_B`!
      5. All 4 specialized CUDA kernels actually execute on GPU (`cube_bind_4x4_d15`,
         `eval_bridges_4x4`, `line_step_residual`, `continuation_gate_4x4`).
    """
    torch.manual_seed(20261122)

    # 1. Direct verification of the irreducible 4x4 cube (non-commutativity, FuturePressure, KA-F01)
    e0 = torch.zeros(32, dtype=torch.float32); e0[0] = 1.0
    e1 = torch.zeros(32, dtype=torch.float32); e1[1] = 1.0
    e2 = torch.zeros(32, dtype=torch.float32); e2[2] = 1.0
    zero32 = torch.zeros(32, dtype=torch.float32)

    # UNKNOWN when right wing is empty
    _, _, _, t_unk, _, _, st_unk = _eval_4x4_mirror_cube_cpu(e0, e0, e0, zero32, zero32)
    assert st_unk == EQ_UNKNOWN and t_unk == 0.0
    # EQUAL when left and right reconstruct the same center with small residual
    _, _, _, t_eq, r_eq, fp_eq, st_eq = _eval_4x4_mirror_cube_cpu(e0, e0, e0, e0, e0)
    assert st_eq == EQ_EQUAL and t_eq > 0.99 and r_eq < 1e-4 and fp_eq > 0.0
    # TENSION when shared center exists WITH genuine orthogonal RelationResidual (45 deg)
    e01 = (e0 + e1) / (2.0 ** 0.5)
    _, _, _, t_ten, r_ten, _, st_ten = _eval_4x4_mirror_cube_cpu(e0, e0, e0, e01, e01)
    assert st_ten == EQ_TENSION and t_ten > 0.3 and r_ten > 0.5
    # SPLIT when orthogonal / opposite (topic shift!)
    _, _, _, t_spl, _, _, st_spl = _eval_4x4_mirror_cube_cpu(e0, e0, e0, e1, e1)
    assert st_spl == EQ_SPLIT and t_spl == 0.0
    # FuturePressure (F14): future right wing (e1, e1) reweights mixed past (p0=e0, p1=e1) toward e1 WITHOUT mutating p0, p1!
    sl_f0, _, _, _, _, fp_f0, _ = _eval_4x4_mirror_cube_cpu(e0, e1, zero32, e0, e0)
    sl_f1, _, _, _, _, fp_f1, _ = _eval_4x4_mirror_cube_cpu(e0, e1, zero32, e1, e1)
    assert fp_f0 > 0.0 and fp_f1 > 0.0
    assert float(sl_f0[0]) > float(sl_f0[1]) and float(sl_f1[1]) > float(sl_f1[0])
    # Non-commutativity (L02): swapping asymmetric left & right wings changes directed RelationResidual!
    _, _, _, _, r_ab, _, _ = _eval_4x4_mirror_cube_cpu(e0, e0 + 0.5 * e2, zero32, e01, e01 + 2.0 * e1)
    _, _, _, _, r_ba, _, _ = _eval_4x4_mirror_cube_cpu(e01, e01 + 2.0 * e1, zero32, e0, e0 + 0.5 * e2)
    assert abs(r_ab - r_ba) > 1e-3, "4x4 cube must remain strictly non-commutative!"

    # 2. Multi-agent two-topic + bridge cohort with DISTINCT input (u_in) and outcome (u_post)
    h_dim = 128
    v_a = torch.zeros(h_dim, dtype=torch.float32)
    v_a[: h_dim // 2] = 1.0
    v_b = torch.zeros(h_dim, dtype=torch.float32)
    v_b[h_dim // 2 :] = 1.0
    # Distinct transformation law shift for u_post != u_in so Logic_t != 0
    v_shift = torch.zeros(h_dim, dtype=torch.float32)
    v_shift[::2] = 0.25

    xs_in = []
    for _ in range(4):
        xs_in.append(v_a + 0.08 * torch.randn(h_dim))
    xs_in.append(0.75 * v_a + 0.25 * v_b + 0.06 * torch.randn(h_dim))
    xs_in.append(0.25 * v_a + 0.75 * v_b + 0.06 * torch.randn(h_dim))
    for _ in range(4):
        xs_in.append(v_b + 0.08 * torch.randn(h_dim))
    x_in_cpu = torch.stack(xs_in, dim=0)
    x_post_cpu = x_in_cpu + v_shift.unsqueeze(0)
    x_in_gpu = x_in_cpu.to("cuda:0")
    x_post_gpu = x_post_cpu.to("cuda:0")

    routes = [
        [(0, 0.40), (1, 0.35), (2, 0.25)],
        [(1, 0.40), (0, 0.35), (2, 0.25)],
        [(2, 0.40), (0, 0.35), (3, 0.25)],  # agent 3 = pure Topic A duplicate for C03 test!
        [(0, 0.40), (2, 0.35), (3, 0.25)],
        [(0, 0.38), (1, 0.32), (8, 0.30)],  # bridge 8 meets Topic A
        [(4, 0.38), (5, 0.32), (8, 0.30)],  # bridge 8 meets Topic B
        [(4, 0.40), (5, 0.35), (6, 0.25)],
        [(5, 0.40), (4, 0.35), (6, 0.25)],
        [(6, 0.40), (4, 0.35), (5, 0.25)],
        [(4, 0.40), (6, 0.35), (5, 0.25)],
    ]

    pm_cpu = PureMirrorCudaFeedback(num_layers=4, num_experts=32, device="cpu")
    pm_gpu = PureMirrorCudaFeedback(num_layers=4, num_experts=32, device="cuda:0")
    assert pm_cpu.use_cuda_kernels is False
    assert pm_gpu.use_cuda_kernels is True

    u_in_cpu = pm_cpu.project_states(0, x_in_cpu, update_mean=True)
    u_post_cpu = pm_cpu.project_states(0, x_post_cpu, update_mean=False)
    u_in_gpu = pm_gpu.project_states(0, x_in_gpu, update_mean=True)
    u_post_gpu = pm_gpu.project_states(0, x_post_gpu, update_mean=False)
    assert torch.allclose(u_in_cpu, u_in_gpu.cpu(), atol=1e-4)
    assert torch.allclose(u_post_cpu, u_post_gpu.cpu(), atol=1e-4)

    # First: pre-execution candidate bind (`is_commit=False`), then CompletedPass commit (`is_commit=True`)!
    n_c0 = pm_cpu.observe_mirror_step(0, routes, u_in_cpu, u_post=None, is_commit=False)
    n_g0 = pm_gpu.observe_mirror_step(0, routes, u_in_gpu, u_post=None, is_commit=False)
    assert n_c0 == n_g0 and n_g0 > 0
    n_c1 = pm_cpu.observe_mirror_step(0, routes, u_in_cpu, u_post=u_post_cpu, is_commit=True)
    n_g1 = pm_gpu.observe_mirror_step(0, routes, u_in_gpu, u_post=u_post_gpu, is_commit=True)
    assert n_c1 == n_g1

    # Exact CUDA == CPU parity across ALL 6 irreducible role slots, link_energy, and bridge_ratio!
    assert torch.allclose(pm_cpu.roles[0], pm_gpu.roles[0].cpu(), atol=1e-4)
    assert torch.allclose(pm_cpu.link_energy[0], pm_gpu.link_energy[0].cpu(), rtol=1e-3, atol=5e-4)
    assert torch.allclose(pm_cpu.bridge_ratio[0], pm_gpu.bridge_ratio[0].cpu(), atol=1e-4)

    # Verify distinct irreducible surfaces: left_outer != left_inner AND D15 Logic_t != 0!
    assert float((pm_gpu.left_outer[0, 0] - pm_gpu.left_inner[0, 0]).norm().item()) > 1e-3
    assert float(pm_gpu.Logic[0, 0].norm().item()) > 1e-3

    # Verify IMMUTABILITY of past surfaces (left_outer, left_inner) during `is_commit=False`!
    p0_saved = pm_gpu.left_outer[0, 0].clone()
    p1_saved = pm_gpu.left_inner[0, 0].clone()
    act_saved = int(pm_gpu.active_steps[0, 0].item())
    ub_probe = pm_gpu.project_states(0, v_b.unsqueeze(0).to("cuda:0"), update_mean=False)
    pm_gpu.observe_mirror_step(0, [[(0, 0.6), (1, 0.4)]], ub_probe, u_post=None, is_commit=False)
    assert torch.equal(pm_gpu.left_outer[0, 0], p0_saved), "Past Outcome_(t-1) must NEVER be overwritten when is_commit=False!"
    assert torch.equal(pm_gpu.left_inner[0, 0], p1_saved), "Past ObservedBehavior_(t-1) must NEVER be overwritten when is_commit=False!"
    assert int(pm_gpu.active_steps[0, 0].item()) == act_saved
    assert torch.allclose(pm_gpu.right_inner[0, 0], ub_probe[0], atol=1e-5)

    # Restore exact CPU parity by running the same probe on CPU before role assertions
    ub_probe_c = pm_cpu.project_states(0, v_b.unsqueeze(0), update_mean=False)
    pm_cpu.observe_mirror_step(0, [[(0, 0.6), (1, 0.4)]], ub_probe_c, u_post=None, is_commit=False)
    assert torch.allclose(pm_cpu.roles[0], pm_gpu.roles[0].cpu(), atol=1e-4)

    # Re-commit Topic A on [0, 1] so [0, 1, 2, 3] sit cleanly in Topic A for Bridge & C03 checks
    pm_cpu.observe_mirror_step(0, routes, u_in_cpu, u_post=u_post_cpu, is_commit=True)
    pm_gpu.observe_mirror_step(0, routes, u_in_gpu, u_post=u_post_gpu, is_commit=True)

    # Bridge `8` connects disjoint mirror peers (A vs B) -> mirror_bridge; hubs `0`, `4` -> mirror_core!
    r8, _lr8, br8, hl = pm_gpu.agent_mirror_feedback((0, 8))
    r0, _lr0, br0, _ = pm_gpu.agent_mirror_feedback((0, 0))
    r4, _lr4, br4, _ = pm_gpu.agent_mirror_feedback((0, 4))
    assert hl is True
    assert r8 == "mirror_bridge" and br8 >= 0.5
    assert r0 == "mirror_core" and br0 < 0.5
    assert r4 == "mirror_core" and br4 < 0.5
    pm_cpu._sync_all_dirty()
    assert pm_cpu.role_of == pm_gpu.role_of

    # Kernel #3 parity: fused 4-column Mirror Equality line_spine step with exact RelationResidual!
    line_c = torch.randn(4, h_dim, dtype=torch.float32)
    line_g = line_c.to("cuda:0")
    ye_c = torch.randn(2, h_dim, dtype=torch.bfloat16)
    ye_g = ye_c.to("cuda:0")
    toks_c = torch.tensor([1, 3], dtype=torch.long)
    ws_c = torch.tensor([0.4, 0.6], dtype=torch.float32)
    met_c, used_c = launch_pure_mirror_line_step(line_c, ye_c, toks_c, ws_c)
    met_g, used_g = launch_pure_mirror_line_step(line_g, ye_g, toks_c.to("cuda:0"), ws_c.to("cuda:0"))
    assert used_c is False and used_g is True
    assert met_g.shape == (2, 4)
    assert torch.allclose(line_c, line_g.cpu(), atol=1e-4)
    assert torch.allclose(met_c, met_g.cpu(), atol=1e-4)

    # Kernel #4 parity + C03/L04 Law of New Observation:
    #   - candidate `3` (pure Topic A duplicate of [0, 1, 2]) -> `mirror_maintained` (0 B disk!)
    #   - candidate `6` (orthogonal Topic B while on Topic A) -> `no_mirror_link` (0 B disk!)
    #   - candidate `8` (bridge with real TENSION + RelationResidual) -> `mirror_expand`!
    ua_c = pm_cpu.project_states(0, v_a.unsqueeze(0), update_mean=False)[0]
    ua_g = pm_gpu.project_states(0, v_a.unsqueeze(0).to("cuda:0"), update_mean=False)[0]
    res_a_c = pm_cpu.evaluate_continuation_batch(
        0, [0, 1, 2], [], [3, 6, 8], [0.15, 0.15, 0.25], [1, 1, 1],
        u_info=ua_c, prev_u_info=None, line_norm=2.0, last_obs_norm=1.2, last_res_norm=1.0,
        w_done=0.60, rem_w=0.40, tok_done=3, rest_min=3, rest_tol=0.24,
    )
    res_a_g = pm_gpu.evaluate_continuation_batch(
        0, [0, 1, 2], [], [3, 6, 8], [0.15, 0.15, 0.25], [1, 1, 1],
        u_info=ua_g, prev_u_info=None, line_norm=2.0, last_obs_norm=1.2, last_res_norm=1.0,
        w_done=0.60, rem_w=0.40, tok_done=3, rest_min=3, rest_tol=0.24,
    )
    assert [r[0] for r in res_a_c] == [r[0] for r in res_a_g]
    assert res_a_g[0][0] == "mirror_maintained", "C03: pure EQUAL duplicate must return mirror_maintained!"
    assert res_a_g[1][0] == "no_mirror_link", "Orthogonal Topic B on Topic A must return no_mirror_link!"
    assert res_a_g[2][0] == "mirror_expand" and res_a_g[2][2] > 0.0, "Bridge TENSION must expand!"

    # Kernel #5 parity: Autonomous 4x4 Mirror Self-Collapse (1-to-1 with rings, ZERO UNI!):
    #   On live Topic A flow (active=[0, 1]):
    #     - `6` (orthogonal Topic B, SPLIT)   -> 1 (`collapse_split`, halves stale link_energy!)
    #     - `3` (Topic A duplicate, EQUAL/C03)-> 2 (`collapse_equal`)
    #     - `8` (living mirror_bridge/TENSION)-> 0 (`hold_tension`)
    ua_post_c = pm_cpu.project_states(0, (v_a + v_shift).unsqueeze(0), update_mean=False)[0]
    ua_post_g = pm_gpu.project_states(0, (v_a + v_shift).unsqueeze(0).to("cuda:0"), update_mean=False)[0]
    le6_before = float(pm_gpu.link_energy[0, 6].item())
    col_c = pm_cpu.evaluate_self_collapse_batch(0, [6, 3, 8], [0, 1], ua_c, ua_post_c)
    col_g = pm_gpu.evaluate_self_collapse_batch(0, [6, 3, 8], [0, 1], ua_g, ua_post_g)
    assert col_c == col_g == [1, 2, 0], f"Expected [collapse_split=1, collapse_equal=2, hold=0], got {col_g}"
    assert float(pm_gpu.link_energy[0, 6].item()) == pytest.approx(0.5 * le6_before, rel=1e-4)

    # Topic shift WITHOUT reset: agent `0` keeps its past surfaces & Strategy, and links smoothly to Topic B `6`!
    steps_0_before = int(pm_gpu.active_steps[0, 0].item())
    roles_0_before = pm_gpu.roles[0, 0].clone()
    pm_gpu.on_topic_shift(layer=0)
    assert int(pm_gpu.active_steps[0, 0].item()) == steps_0_before
    assert torch.equal(pm_gpu.roles[0, 0], roles_0_before)

    ub_g = pm_gpu.project_states(0, v_b.unsqueeze(0).to("cuda:0"), update_mean=False)[0]
    res_b_g = pm_gpu.evaluate_continuation_batch(
        0, [0, 1, 2], [], [6], [0.25], [0],
        u_info=ub_g, prev_u_info=None, line_norm=2.0, last_obs_norm=1.2, last_res_norm=1.0,
        w_done=0.60, rem_w=0.40, tok_done=3, rest_min=3, rest_tol=0.24,
    )
    assert res_b_g[0][0] == "mirror_expand" and res_b_g[0][2] > 0.0

    # Advancing agent `0` on Topic B with `is_commit=True` updates active_steps monotonically & shifts S toward B
    ub_post_g = pm_gpu.project_states(0, (v_b + v_shift).unsqueeze(0).to("cuda:0"), update_mean=False)
    pm_gpu.observe_mirror_step(0, [[(0, 0.5), (6, 0.5)]], ub_g.unsqueeze(0), u_post=ub_post_g, is_commit=True)
    assert int(pm_gpu.active_steps[0, 0].item()) == steps_0_before + 1

    st = pm_gpu.stats()
    assert st["mode"] == "pure_mirror_4x4_d15_cuda"
    assert st["cuda_kernels_active"] is True
    assert st["cuda_launches"]["cube_bind_4x4_d15"] >= 4
    assert st["cuda_launches"]["eval_bridges_4x4"] >= 4
    assert st["cuda_launches"]["continuation_gate_4x4"] >= 2
    assert st["cuda_launches"]["self_collapse_4x4"] >= 1
    assert st["self_collapse"]["split"] >= 1
    assert st["self_collapse"]["equal_c03"] >= 1
    assert st["self_collapse"]["held_tension"] >= 1
    assert st["continuation_maintained"] >= 1
    assert st["continuation_no_link"] >= 1
    assert st["topic_shift_continuations"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_three_way_v1_v2_v3_imprint_overlay_cuda():
    """End-to-end CUDA verification of all 3 versions side-by-side:
      - V1 (`False`)           : Loop-Local baseline
      - V2 (`"spiral_sphere"`) : 3D Spiral-Sphere Overlap Lenses
      - V3 (`"pure_mirror"`)   : Irreducible 4x4 + D15 Pure Mirror Equality on NVRTC CUDA kernels
    """
    torch.manual_seed(20261122)
    dev = torch.device("cuda:0")
    h, inter = 64, 64
    num_layers, num_exp = 2, 14
    bank = {
        (li, e): (
            torch.randn(2 * inter, h, dtype=torch.bfloat16) * 0.04,
            torch.randn(h, inter, dtype=torch.bfloat16) * 0.04,
        )
        for li in range(num_layers)
        for e in range(num_exp)
    }

    dir_a = torch.zeros(h, dtype=torch.float32)
    dir_a[: h // 2] = 1.0
    dir_b = torch.zeros(h, dtype=torch.float32)
    dir_b[h // 2 :] = 1.0
    noise = torch.randn(10, h, dtype=torch.float32) * 0.12
    x_pre_f = torch.empty((10, h), dtype=torch.float32)
    for ti in range(4):
        x_pre_f[ti] = dir_a + noise[ti]
    x_pre_f[4] = 0.75 * dir_a + 0.25 * dir_b + noise[4]
    x_pre_f[5] = 0.25 * dir_a + 0.75 * dir_b + noise[5]
    for ti in range(6, 10):
        x_pre_f[ti] = dir_b + noise[ti]
    x_pre = x_pre_f.to(device=dev, dtype=torch.bfloat16)
    x_dec = (dir_a + torch.randn(1, h, dtype=torch.float32) * 0.10).to(device=dev, dtype=torch.bfloat16)

    idx_t = torch.tensor([
        [0, 1, 2, 3, 10, 11],
        [1, 0, 3, 2, 10, 12],
        [2, 3, 0, 1, 11, 13],
        [3, 2, 1, 0, 12, 13],
        [0, 1, 8, 2, 10, 11],
        [4, 5, 8, 6, 12, 13],
        [4, 5, 6, 7, 10, 11],
        [5, 4, 7, 6, 11, 12],
        [6, 7, 4, 5, 12, 13],
        [7, 6, 5, 4, 10, 13],
    ], dtype=torch.long, device=dev)
    w_t = torch.tensor([[0.34, 0.28, 0.16, 0.14, 0.05, 0.03] for _ in range(10)], dtype=torch.bfloat16, device=dev)

    outs = {}
    stats = {}
    for label, fb_mode in (("v1", False), ("v2", "spiral_sphere"), ("v3", "pure_mirror")):
        contour = StanokContainmentContour(device=dev, enabled=False)
        bus = UnifiedProbabilityBus(contour)
        cache = LazyMoEBinaryHalvingCache(
            lambda li, e: bank[(li, e)],
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
            fixation_cov_thresh=1.0,
            graphify_feedback=fb_mode,
        )
        try:
            ov = cache._imprint_overlay
            bus.bind(imprint_overlay=ov)
            y0 = cache.forward_layer_moe(0, x_pre, idx_t, w_t, F.silu)
            _ = cache.forward_layer_moe(1, x_pre, idx_t, w_t, F.silu)
            bus.notify_topic_shift(layer=0, reason=f"{label}_shift")
            y_d = cache.forward_layer_moe(
                0,
                x_dec,
                torch.tensor([[0, 1, 2, 8, 12, 13]], dtype=torch.long, device=dev),
                torch.tensor([[0.36, 0.28, 0.18, 0.10, 0.05, 0.03]], dtype=torch.bfloat16, device=dev),
                F.silu,
            )
            assert bool(torch.isfinite(y0).all()) and bool(torch.isfinite(y_d).all())
            assert ov.graph.score((0, 8)) == pytest.approx(ov.graph.relative_binding((0, 8)))
            outs[label] = y0.float().cpu()
            stats[label] = ov.stats()
            assert stats[label]["relay"]["transitions"]["B->A"] == 0
            assert stats[label]["relay"]["standby_bytes"] == 0
        finally:
            cache.close()
            contour.close()

    gf3 = stats["v3"]["graphify_feedback"]
    assert gf3["enabled"] is True and gf3["mode"] == "pure_mirror"
    t3 = gf3["topology"]
    assert t3["mode"] == "pure_mirror_4x4_d15_cuda"
    assert t3["cuda_kernels_active"] is True
    assert t3["cuda_launches"]["cube_bind_4x4_d15"] >= 3
    assert t3["cuda_launches"]["eval_bridges_4x4"] >= 3
    assert t3["cuda_launches"]["line_step_residual"] > 0
    assert t3["cuda_launches"]["continuation_gate_4x4"] > 0
    assert t3["cuda_launches"]["self_collapse_4x4"] >= 1
    assert stats["v3"]["lattice_self_collapses"] >= 1
    assert t3["topic_shift_continuations"] >= 1
    assert t3["total_mirror_links"] > 0
    cos_13 = float(F.cosine_similarity(outs["v1"].flatten(), outs["v3"].flatten(), dim=0).item())
    cos_23 = float(F.cosine_similarity(outs["v2"].flatten(), outs["v3"].flatten(), dim=0).item())
    assert cos_13 > 0.97 and cos_23 > 0.97






