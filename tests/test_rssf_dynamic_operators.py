"""End-to-end tests for Anton's 4-layer solver-free dynamic balance integrated across Quintessentia:
  - Micro scale (Q1 weight cube) & Multi-scale balance observer (`observe_dynamic_balance`)
  - Token scale (`M7` MirrorSpeculationScale in `src/general_speculation.py`)
  - Device scale (`M1` MirrorLoadDispatcher in `src/mirror_scales_v4.py`)
  - Turn / KV-archive / Prefill scale (`GenerationBalanceTracker` wired into `unified_runtime.py`).
"""
from __future__ import annotations

import math

import pytest

from src.general_speculation import MirrorSpeculationScale
from src.mirror_scales_v4 import MirrorLoadDispatcher
from src.rssf_dynamic_operators import (
    GenerationBalanceTracker,
    RSSFCollapseDetector,
    VolterraFDSMonitor,
    observe_dynamic_balance,
    observe_q1_cube_balance,
)


def test_solver_free_eight_force_balance_across_all_scales_of_information():
    """Same 8-term numerator/denominator equality (0 solver iterations) works identically
    whether Information `I` is one Q1 weight cube, a dialogue KV archive, a whole model, or a swarm."""
    # Exact algebraic cancellation in Term 3: (I+L+ab+c^2)^2 - c^4 = (I+L+ab)^2 + 2*c^2*(I+L+ab)
    snap = observe_dynamic_balance(U=1.5, G=0.5, Q=0.6, S=0.3, I=0.8, L=0.9, C_L=1.2, a=0.25, b=0.25, c=0.50)
    il_ab = 0.8 + 0.9 + 0.25 * 0.25
    expected_d3 = 1.2 * 1.5 - 0.6 + (0.5 + 0.25**2) * (0.3 + 0.25**2) + il_ab**2 + 2.0 * (0.5**2) * il_ab
    assert abs(snap.d3_margin - expected_d3) < 1e-12
    assert snap.regime in ("BALANCED", "EXPAND", "COMPRESS")

    # Micro scale: hot Q1-cube vs redundant tail bits vs over-compressed cube under shock Q
    hot_cube = observe_q1_cube_balance(info_density=0.95, weight_mass_frac=0.3, input_impulse=0.9, residual_entropy=0.1)
    tail_cube = observe_q1_cube_balance(info_density=0.05, weight_mass_frac=0.9, input_impulse=0.2, residual_entropy=0.2)
    shock_cube = observe_q1_cube_balance(
        info_density=0.05, weight_mass_frac=0.5, input_impulse=2.6, residual_entropy=0.8, output_coherence=0.15, bandwidth_factor=0.8
    )
    assert hot_cube.info_efficiency > tail_cube.info_efficiency * 10.0
    assert hot_cube.d3_margin > 0.0
    assert shock_cube.regime == "SCALE_BREAK" and shock_cube.d3_margin <= 0.0

    # Macro scale: right-sized 9B vs overkill 120B vs underpowered 1B on hard chaotic task
    m9b = observe_dynamic_balance(U=1.5, G=0.35, Q=0.6, S=0.3, I=0.75, L=0.90, C_L=1.3)
    m120b = observe_dynamic_balance(U=1.5, G=0.95, Q=0.15, S=0.1, I=0.98, L=0.90, C_L=0.5)
    m1b_hard = observe_dynamic_balance(U=1.0, G=0.15, Q=2.2, S=0.9, I=0.10, L=0.15, C_L=1.0)
    assert m9b.info_efficiency > m120b.info_efficiency * 2.5
    assert m1b_hard.regime == "SCALE_BREAK" and m1b_hard.d3_margin <= 0.0


def test_volterra_fds_scale_relativity_and_zero_false_alarms():
    """Exact scale invariance across 4..80 GB GPUs, equilibrium detection, and 0 false alarms on breathing."""
    warn_steps = []
    t_ends = []
    for gb in (4, 6, 8, 12, 16, 24, 80):
        total = gb * (1024 ** 3)
        reserve = int(0.10 * total)
        leak = int(0.0075 * total)
        mon = VolterraFDSMonitor()
        det = RSSFCollapseDetector(drift_horizon_steps=4.0)
        first_warn = None
        for t in range(30):
            free = int(0.30 * total) - leak * t
            diag = mon.update(free_bytes=free, total_bytes=total, reserve_bytes=reserve)
            rep = det.evaluate(free_bytes=free, total_bytes=total, ec_fds=diag)
            if first_warn is None and t >= 2 and rep.unbounded_result_drift:
                first_warn = (t, round(diag.fds_t_end_steps, 4))
        assert first_warn is not None
        warn_steps.append(first_warn[0])
        t_ends.append(first_warn[1])

    assert len(set(warn_steps)) == 1, f"Not scale-invariant across GPUs: {warn_steps}"
    assert max(t_ends) - min(t_ends) < 1e-3
    assert warn_steps[0] <= 19  # 8+ steps before wall breach at step 27!

    # Healthy breathing (+/-120 MiB) -> 0 drift warnings
    MiB = 1024 ** 2
    mon_b = VolterraFDSMonitor()
    det_b = RSSFCollapseDetector(drift_horizon_steps=4.0)
    for t in range(40):
        free = int((2000 + 120 * math.sin(1.3 * t)) * MiB)
        diag = mon_b.update(free_bytes=free, total_bytes=8192 * MiB, reserve_bytes=819 * MiB)
        rep = det_b.evaluate(free_bytes=free, total_bytes=8192 * MiB, ec_fds=diag)
        assert not rep.unbounded_result_drift

    # Equilibrium plateau detection
    mon_eq = VolterraFDSMonitor(equilibrium_window=3)
    eq_reached = False
    for free_mib in (4000, 3500, 3100, 3095, 3094, 3094, 3094):
        diag = mon_eq.update(free_bytes=free_mib * MiB, total_bytes=8192 * MiB, reserve_bytes=819 * MiB)
        if diag.at_equilibrium:
            eq_reached = True
    assert eq_reached


def test_production_m1_early_leak_detection_2d_scars_and_scale_relativity():
    """Live production M1 (`MirrorLoadDispatcher`) catches quiet shmem leaks 8-9 steps before wall,
    preserves non-degenerate 2D scars, triggers 0 false damps on breathing, and matches across 8/24 GB."""
    MiB = 1024 ** 2
    GiB = 1024 ** 3

    # 1. Quiet -60 MiB/step shmem creep on 8 GB GPU (wall at step 27) -> M1 catches by step 18 (9 steps lead!)
    s_leak = [(int((2400 - 60 * t) * MiB), 8 * GiB, 15.0, 18.0, 0, "mock") for t in range(30)]
    i = {"k": 0}
    m1 = MirrorLoadDispatcher(
        device="cpu", enabled=True, safety_reserve_bytes=int(0.10 * 8 * GiB),
        sensor_fn=lambda: s_leak[min(i.__setitem__("k", i["k"] + 1) or i["k"] - 1, len(s_leak) - 1)],
    )
    b = 2 * GiB
    first_damp = None
    max_scar = 0.0
    for step in range(30):
        d = m1.evaluate_load(cached_bytes=1 * GiB, current_budget_bytes=b)
        b = d.target_budget_bytes
        max_scar = max(max_scar, m1.last_scar_norm)
        if first_damp is None and d.dispatch_mode == "THROTTLE_DAMP":
            first_damp = step
    assert first_damp is not None and first_damp <= 19, f"Expected early leak catch, got step {first_damp}"
    assert max_scar > 1e-4, "Restored 2D M1 must produce non-zero reversible scars!"
    st = m1.stats()
    assert st["rssf_balance"]["ec_fds_damp_steps"] >= 1
    assert st["rssf_balance"]["last_balance"] is not None

    # 2. Healthy +/-120 MiB breathing -> ZERO false THROTTLE_DAMP!
    s_breath = [(int((2200 + 120 * math.sin(1.3 * t)) * MiB), 8 * GiB, 40.0, 42.0, 0, "mock") for t in range(40)]
    j = {"k": 0}
    m1_b = MirrorLoadDispatcher(
        device="cpu", enabled=True, safety_reserve_bytes=int(0.10 * 8 * GiB),
        sensor_fn=lambda: s_breath[min(j.__setitem__("k", j["k"] + 1) or j["k"] - 1, len(s_breath) - 1)],
    )
    b2 = 2 * GiB
    for _ in range(40):
        d = m1_b.evaluate_load(cached_bytes=1 * GiB, current_budget_bytes=b2)
        b2 = d.target_budget_bytes
        assert d.dispatch_mode != "THROTTLE_DAMP"


def test_production_m7_speculation_balance_adapts_depth_on_scale_break():
    """Live production M7 (`MirrorSpeculationScale`) exposes 8-force balance and clamps depth to 1
    when draft confidence & acceptance collapse into SCALE_BREAK."""
    m7 = MirrorSpeculationScale(max_drafts=3)
    # High confidence -> full depth allowed, healthy positive d3_margin
    k_ok = m7.pick_depth([0.96, 0.94, 0.92], max_allowed=3)
    assert k_ok == 3
    assert m7.stats()["rssf_balance"]["d3_margin"] > 0.0

    # Simulate repeated severe rejections (huge logp_gap, 0 accepted) -> drives balance into SCALE_BREAK
    for step in range(4):
        m7.record_verification(proposed=3, accepted=0, position=10 + step, drafts=[1, 2, 3], verified_token=99)
    m7.last_logp_gap = 3.5
    k_bad = m7.pick_depth([0.20, 0.15, 0.10], max_allowed=3)
    assert k_bad == 1
    assert m7.stats()["balance_scale_breaks"] >= 1


def test_generation_balance_tracker_full_turn_lifecycle():
    """End-to-end turn tracker: detects ForgeMind LossOfArchive, WDDM LossOfObserverCoherence,
    and preemptively damps prefill chunk size during quiet VRAM creep."""
    GiB = 1024 ** 3
    MiB = 1024 ** 2
    total_b = 8 * GiB
    reserve_b = int(0.05 * total_b)
    gb = GenerationBalanceTracker(fds_chunk_warn_steps=3.0)

    # Turn 1: cold start, healthy prefill (no leak) -> keeps full 512 chunk
    bal1, col1 = gb.begin_turn(
        prompt_tokens=2048, cached_prefix_tokens=0, had_prior_archive=False,
        free_bytes=3 * GiB, total_bytes=total_b, reserve_bytes=reserve_b,
        secondary_free_bytes=3 * GiB, pcie_gb_s=23.6, offload_ratio=0.0,
    )
    assert not col1.loss_of_archive
    assert not col1.loss_of_observer_coherence
    assert bal1.d3_margin > 0.0
    for _ in range(4):
        ch, _, _ = gb.advise_prefill_chunk(
            base_chunk=512, remaining_tokens=2048, free_bytes=3 * GiB, total_bytes=total_b, reserve_bytes=reserve_b
        )
        assert ch == 512
    summary1 = gb.finish_turn(
        prompt_tokens=2048, cached_prefix_tokens=0, generated_tokens=128,
        free_bytes=3 * GiB, total_bytes=total_b, kv_offloaded_layers=0, kv_gpu_layers=8,
        decode_tok_per_s=17.0, calibrated_decode_tok_per_s=18.0,
    )
    assert summary1["prefill_damped_chunks"] == 0
    assert summary1["turn_end_balance"]["d3_margin"] > 0.0

    # Turn 2: ForgeMind-style archive wipe (had_prior_archive=True, cached=0) + WDDM +2 GiB phantom divergence
    #         + quiet VRAM creep during prefill -> flags BOTH collapses AND damps chunk preemptively!
    bal2, col2 = gb.begin_turn(
        prompt_tokens=20000, cached_prefix_tokens=0, had_prior_archive=True,
        free_bytes=1200 * MiB, total_bytes=total_b, reserve_bytes=reserve_b,
        secondary_free_bytes=3200 * MiB, pcie_gb_s=23.6, offload_ratio=0.5,
    )
    assert col2.loss_of_archive is True
    assert col2.loss_of_observer_coherence is True
    assert bal2.regime == "SCALE_BREAK"

    damped_seen = False
    for c_idx in range(10):
        free_now = int((1150 - 75 * c_idx) * MiB)
        ch, _, _ = gb.advise_prefill_chunk(
            base_chunk=512, remaining_tokens=10000, free_bytes=free_now, total_bytes=total_b, reserve_bytes=reserve_b
        )
        if ch < 512:
            damped_seen = True
            break
    assert damped_seen, "Prefill balance tracker must preemptively shrink chunk on sustained VRAM creep!"


def test_live_stream_chat_end_to_end_rssf_balance_archive_and_prefill_damping(tmp_path, monkeypatch):
    """Live end-to-end `QuintessentiaUnifiedEngine.stream_chat` test WITHOUT launching LAN server:
      Turn 1: Cold start -> produces valid `rssf_generation_balance` & parks KV archive.
      Turn 2: Normal continuation -> reuses KV prefix (`cached_prefix_tokens > 0`, `loss_of_archive=False`).
      Turn 3: Client history truncation (ForgeMind 21-msg pattern) -> live `stream_chat` flags `loss_of_archive=True`!
      Turn 4: Multi-chunk prefill under simulated quiet VRAM creep -> live `_effective_prefill_chunk` preemptively damps!
    """
    import json
    import torch
    from transformers import AutoModelForCausalLM, Qwen2Config
    from quintessentia_cli.unified_runtime import QuintessentiaUnifiedEngine
    import quintessentia_cli.runtime_config as rc_mod

    torch.manual_seed(2026)
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
    )
    AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).save_pretrained(tmp_path)
    # ByteLevel BPE with single-character ASCII alphabet (native to Qwen2TokenizerFast!)
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    chars = list("abcdefghijklmnopqrstuvwxyz0123456789 .,:-_!?/\n")
    vocab = {"<eos>": 0, "Ġ": 1}
    for ch in chars:
        key = "Ġ" if ch == " " else ("Ċ" if ch == "\n" else ch)
        if key not in vocab and len(vocab) < 64:
            vocab[key] = len(vocab)
    while len(vocab) < 64:
        vocab[f"x{len(vocab)}"] = len(vocab)
    tok_obj = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok_obj.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok_obj.decoder = decoders.ByteLevel()
    tok_obj.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({
            "tokenizer_class": "PreTrainedTokenizerFast",
            "eos_token": "<eos>",
            "chat_template": "{% for m in messages %}{{ m['role'] }}:{{ m['content'] }}<eos>{% endfor %}assistant:",
        }),
        encoding="utf-8",
    )

    has_cuda = torch.cuda.is_available()

    def _run_turn(eng, messages, max_new=6, use_cache=True):
        done = None
        for ev in eng.stream_chat(
            messages,
            max_new_tokens=max_new,
            enable_thinking=False,
            use_session_cache=use_cache,
            stop_on_eos=False,
        ):
            if ev.get("type") == "done":
                done = ev
        assert done is not None
        return done

    with QuintessentiaUnifiedEngine(
        tmp_path,
        device="auto" if has_cuda else "cpu",
        weight_format="bf16",
        speculation="lookup",
        prefill_chunk_tokens=256,
        progressive_start=False,
        verbose=False,
    ) as eng:
        # Turn 1: Cold start with valid session cache
        m1 = [{"role": "user", "content": "hello world 1234"}]
        d1 = _run_turn(eng, m1, max_new=6, use_cache=True)
        rb1 = d1["metrics"]["rssf_generation_balance"]
        assert d1["metrics"]["prompt_tokens"] >= 10
        assert rb1["entry_collapse"]["loss_of_archive"] is False
        assert rb1["entry_balance"]["d3_margin"] > 0.0
        assert rb1["turn_end_balance"]["d3_margin"] > 0.0
        assert eng.session.parked_bytes > 0

        # Turn 2: Direct continuation -> real KV prefix hit, NO archive collapse
        m2 = [*m1, {"role": "assistant", "content": d1["answer_text"]}, {"role": "user", "content": "next turn 5678"}]
        d2 = _run_turn(eng, m2, max_new=6, use_cache=True)
        rb2 = d2["metrics"]["rssf_generation_balance"]
        assert d2["metrics"]["cached_prefix_tokens"] > 0
        assert rb2["entry_collapse"]["loss_of_archive"] is False
        assert rb2["turn_end_balance"]["d3_margin"] > 0.0

        # Turn 3: Simulated client history truncation (ForgeMind 21-msg wipe) while prior archive is parked!
        m3 = [{"role": "user", "content": "truncated history reset 9999"}]
        d3 = _run_turn(eng, m3, max_new=4, use_cache=True)
        rb3 = d3["metrics"]["rssf_generation_balance"]
        assert d3["metrics"]["cached_prefix_tokens"] == 0
        assert rb3["entry_collapse"]["loss_of_archive"] is True
        assert any("LossOfArchive" in r for r in rb3["entry_collapse"]["reasons"])

        # Turn 4 (CUDA): Multi-chunk prefill (~1200 tokens) under simulated quiet VRAM creep (-110 MiB/chunk)
        # verifies live `_effective_prefill_chunk` preemptively shrinks chunk via EC Volterra + FDS!
        if has_cuda and eng.engine.device.type == "cuda":
            MiB = 1024 ** 2
            GiB = 1024 ** 3
            creep_step = {"k": 0}

            def _fake_phys_vram(_device, _plan=None):
                # Starts at 1200 MiB free (well above ~256 MiB workspace), drops 110 MiB/chunk!
                free_sim = max(420 * MiB, int((1200 - 110 * creep_step["k"]) * MiB))
                creep_step["k"] += 1
                return free_sim, 8 * GiB

            monkeypatch.setattr(rc_mod, "_query_physical_avail_vram", _fake_phys_vram)
            eng.prefill_chunk_tokens = 256
            long_text = "abcd efgh 1234 " * 80
            d4 = _run_turn(eng, [{"role": "user", "content": long_text}], max_new=4, use_cache=False)
            assert d4["metrics"]["prompt_tokens"] >= 1000
            rb4 = d4["metrics"]["rssf_generation_balance"]
            assert rb4["prefill_chunks_observed"] >= 4
            assert rb4["prefill_damped_chunks"] >= 1, f"Expected EC+FDS prefill damping, got {rb4}"
            assert rb4["prefill_min_fds_t_end_steps"] is not None and rb4["prefill_min_fds_t_end_steps"] <= 3.0


def test_prefill_initial_kv_allocation_no_false_damping():
    """Initial one-time KV allocation with ample free headroom (>=30%) must NOT falsely damp 23K prefill,
    while sustained creep toward the reserve wall still damps preemptively."""
    GiB = 1024 ** 3
    MiB = 1024 ** 2
    total_b = 8 * GiB
    reserve_b = int(0.05 * total_b)
    gb = GenerationBalanceTracker(fds_chunk_warn_steps=3.0)
    gb.begin_turn(
        prompt_tokens=23000, cached_prefix_tokens=0, had_prior_archive=False,
        free_bytes=3600 * MiB, total_bytes=total_b, reserve_bytes=reserve_b,
    )
    # One-time initial KV buffer allocation while headroom stays well above 30% (~3.0 GiB free on 8 GiB GPU)
    for free_mib in (3400, 3150, 2950, 2850, 2850, 2850):
        ch, _, _ = gb.advise_prefill_chunk(
            base_chunk=512, remaining_tokens=16000, free_bytes=free_mib * MiB,
            total_bytes=total_b, reserve_bytes=reserve_b,
        )
        assert ch == 512, f"Healthy initial KV alloc at {free_mib} MiB must not false-damp chunk (got {ch})"

    # Now simulate genuine creep approaching reserve (<30% headroom) -> must damp!
    gb2 = GenerationBalanceTracker(fds_chunk_warn_steps=3.0)
    gb2.begin_turn(
        prompt_tokens=60000, cached_prefix_tokens=0, had_prior_archive=False,
        free_bytes=1300 * MiB, total_bytes=total_b, reserve_bytes=reserve_b,
    )
    damped = False
    for step in range(8):
        ch, _, _ = gb2.advise_prefill_chunk(
            base_chunk=512, remaining_tokens=30000, free_bytes=int((1220 - 80 * step) * MiB),
            total_bytes=total_b, reserve_bytes=reserve_b,
        )
        if ch < 512:
            damped = True
            break
    assert damped, "Genuine sustained creep near reserve must still damp prefill chunk!"


def test_kv_placement_balance_scale_invariance_and_safety():
    """KV placement 8-pair balance is scale-invariant across 4..80 GB GPUs and detects VRAM wall breach."""
    from src.rssf_dynamic_operators import observe_kv_placement_balance

    d3_vals = []
    for gb in (4, 8, 16, 24, 80):
        tot = gb * (1024 ** 3)
        snap = observe_kv_placement_balance(
            n_gpu=8, n_full=8,
            kv_gpu_bytes=int(0.18 * tot),
            staging_bytes=0,
            avail_vram_bytes=int(0.45 * tot),
            total_vram_bytes=tot,
            reserve_bytes=int(0.05 * tot),
            workspace_bytes=int(0.04 * tot),
            pcie_gb_s=23.6,
            is_decode=False,
        )
        assert snap.regime != "SCALE_BREAK" and snap.d3_margin > 0.0
        d3_vals.append(snap.d3_margin)
    assert max(d3_vals) - min(d3_vals) < 1e-6, f"KV placement balance not scale-invariant: {d3_vals}"

    # Over-allocating past the reserve wall must trip SCALE_BREAK immediately!
    tot8 = 8 * (1024 ** 3)
    snap_bad = observe_kv_placement_balance(
        n_gpu=8, n_full=8,
        kv_gpu_bytes=int(0.48 * tot8),
        staging_bytes=0,
        avail_vram_bytes=int(0.45 * tot8),
        total_vram_bytes=tot8,
        reserve_bytes=int(0.05 * tot8),
        workspace_bytes=int(0.04 * tot8),
        is_decode=False,
    )
    assert snap_bad.regime == "SCALE_BREAK" and snap_bad.d3_margin <= 0.0


def test_all_mirror_scales_m0_m2_m3_m4_m5b_m6_m8_rssf_balance():
    """Verify live 8-pair balance + EC/FDS integration across ALL remaining scales: M0, M2, M3, M4, M5b, M6, M8."""
    import torch
    from src.binary_halving_q1 import carve_trunk_mirror_q1
    from src.mirror_scales_v4 import (
        MirrorColdWarmupScale,
        MirrorPostPrefillRegionWarmupScale,
        MirrorRegionBreathingPool,
        mirror_cube_depth_math,
    )
    from src.survivor_probability_path import SurvivorProbabilityTracker
    from quintessentia_cli.thought_buffer import InitialProbabilityAnchor

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device="cpu").manual_seed(2027)

    # 1. M0: carve_trunk_mirror_q1 fills balance_out cleanly on CPU & CUDA
    w = (torch.randn((128, 128), generator=gen, dtype=torch.float32) * 0.08).to(device=dev, dtype=torch.bfloat16)
    m0_bal = {}
    _t, _m, k_opt, _rem = carve_trunk_mirror_q1(
        w, 5, min_cubes=3, mirror_tol=0.05, root_bits=8, group_size=64,
        use_fwht=True, sinkhorn_iters=2, balance_out=m0_bal,
    )
    assert 3 <= k_opt <= 5
    assert "d3_margin" in m0_bal and "info_efficiency_t4" in m0_bal and m0_bal["k_opt"] == k_opt

    # 2. M5b: pure-math + Q1 cube balance preserves boundary contracts
    assert mirror_cube_depth_math(0.05, 0.0, 3, 5) == 3
    assert mirror_cube_depth_math(0.20, 0.85, 3, 5) == 5
    assert mirror_cube_depth_math(0.90, 0.90, 3, 5) == 5

    # 3. M2: MirrorRegionBreathingPool exposes rssf_balance and exhales under pool pressure
    m2 = MirrorRegionBreathingPool(device=torch.device("cpu"), max_pool_bytes=1000, max_warm_regions=4)
    m2.admit_region(("a", 0), {"v": 1}, gpu_bytes=100, recurring_hint=True)
    assert m2.finish_region(("a", 0), keep_if_reusable=True) == "HOLD_WARM_REGION"
    assert m2.stats()["rssf_balance"] is not None
    # Pushing occupancy near pool ceiling -> must exhale cleanly rather than hoard
    m2.admit_region(("big", 1), {"v": 2}, gpu_bytes=850, recurring_hint=True)
    assert m2.finish_region(("big", 1), keep_if_reusable=True) == "EXHALE_RECLAIM"
    m2.close()

    # 4. M3: MirrorColdWarmupScale confirms equilibrium via EC/FDS + exposes rssf_balance
    m3 = MirrorColdWarmupScale(mode="plateau", min_layers_warm=2, plateau_confirm=2, max_warmup_seconds=10.0)
    d0 = m3.evaluate_step(65.0, warmed_so_far=1, total_targets=20, elapsed_seconds=0.1)
    assert d0.mirror_state == "UNKNOWN"
    for i, dt in enumerate((25.0, 12.0, 11.9, 11.9, 11.8), start=2):
        dec = m3.evaluate_step(dt, warmed_so_far=i, total_targets=20, elapsed_seconds=0.3)
        if not dec.continue_warmup:
            assert dec.reason == "mirror_plateau_equal"
            break
    st3 = m3.stats()
    assert st3["rssf_balance"] is not None
    assert st3["rssf_balance"]["fds_equilibrium_confirmations"] >= 1

    # 5. M4: MirrorPostPrefillRegionWarmupScale records rssf_balance & protects scars on sudden phase shift
    m4 = MirrorPostPrefillRegionWarmupScale(num_layers=3, max_region_experts=6, initial_width=2, max_width=2)
    m4.record_prefill_layer(0, body_weights={1: 3.0, 2: 2.0}, tail_weights={1: 1.5})
    m4.record_prefill_layer(1, body_weights={3: 3.0, 4: 2.0}, tail_weights={3: 1.5})
    m4.observe_and_plan_next(0, [1, 2], [0.6, 0.4], lambda l, e: False)
    assert m4.stats()["rssf_balance"] is not None and m4.stats()["rssf_balance"]["d3_margin"] > 0.0
    m4.observe_and_plan_next(0, [20, 21], [0.7, 0.3], lambda l, e: False)
    assert 20 in m4._scar_weights[0]

    # 6. M6: SurvivorProbabilityTracker exposes rssf_balance & protects late SPLIT surges
    class _FakeHead:
        vocab_size = 32
        def project_candidates(self, h_norm, cand_ids):
            return h_norm.reshape(-1)[cand_ids]

    class _FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
            self.lm_head = _FakeHead()

    class _FakeEngine:
        def __init__(self):
            self.model = _FakeModel()
            self.config = type("Cfg", (), {"vocab_size": 32})()
        def _decoder_layers(self):
            return list(self.model.layers)

    tr = SurvivorProbabilityTracker(_FakeEngine(), top_survivors=8, candidate_cap=32, mirror_mode=True)
    tr._apply_norm = staticmethod(lambda _norm, x: x.float())
    base = torch.full((32,), -2.0, dtype=torch.float32)
    base[0] = 5.5
    base[1] = -30.0
    for ck in range(4):
        v = base.clone()
        v[2] = -20.0 if ck < 3 else -9.0
        tr._checkpoint_hidden[ck] = v.unsqueeze(0)
    tr._recompute_survivors()
    st6 = tr.stats()
    assert st6["rssf_balance"] is not None
    assert 0 in set(tr.survivor_ids.tolist()) and 1 in set(tr.early_pruned_ids.tolist())
    assert tr.split_protected_count >= 1
    tr.close()

    # 7. M8: InitialProbabilityAnchor pure-tensor zero-sync 8-pair balance (GPU & CPU!)
    anc = InitialProbabilityAnchor(mirror_mode=True)
    phi0 = torch.zeros(32, dtype=torch.float32, device=dev)
    phi0[5] = 2.0
    phi0[6] = 1.5
    phi0[9] = -1.2
    anc.phi0 = phi0.clone()
    anc.phi_buf = phi0.clone()
    fp0 = anc.initial_fingerprint()

    # Opposing logits -> strictly zero-force (delta == 0)
    z_opp = torch.zeros((1, 32), dtype=torch.float32, device=dev)
    z_opp[0, 9] = 4.5
    out_opp = anc.transform_logits(z_opp, None, step_index=1)
    assert float((out_opp - z_opp).abs().max().item()) < 1e-6

    # Balanced uncertain fork aligned with Phi0 -> gentle modulated resolution + rssf_balance telemetry
    z_fork = torch.full((1, 32), -3.0, dtype=torch.float32, device=dev)
    z_fork[0, 5] = 1.018
    z_fork[0, 6] = 1.00
    z_fork[0, 7] = 1.020
    z_fork[0, 8] = 0.85
    out_fork = anc.transform_logits(z_fork, None, step_index=2)
    assert float((out_fork[0, 5] - z_fork[0, 5]).item()) > 0.0
    assert int(out_fork.argmax(dim=-1).item()) == 5
    assert anc.initial_fingerprint() == fp0
    st8 = anc.mirror_stats()
    assert st8["rssf_balance"] is not None
    assert st8["rssf_balance"]["steps"] == 2
    assert st8["rssf_balance"]["mean_d3_margin"] > 0.0


