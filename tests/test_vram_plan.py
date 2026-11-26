"""CPU checks for the VRAM planner (`normal` / `turbo`, no fixed 4gb/8gb profiles)."""
import pytest

from quintessentia_cli.unified_runtime import GiB, MiB, plan_vram, q4_bytes_for_shape

# Qwen3.5-9B-like numbers: 32 layers of ~122 MiB in Q4, LM head 572 MiB, MTP ~130 MiB.
LAYERS = [122 * MiB] * 32
COMMON = dict(
    layer_bytes=LAYERS,
    lm_head_bytes=q4_bytes_for_shape((248320, 4096)),
    mtp_bytes=130 * MiB,
    kv_bytes_per_token=8 * 2 * 4 * 256 * 2,
    recurrent_state_bytes=24 * 32 * 128 * 128 * 4,
    hidden_size=4096,
    intermediate_size=12288,
)


def test_lm_head_q4_size():
    assert q4_bytes_for_shape((248320, 4096)) == 248320 * 4096 // 2 + 2 * 248320 * 64 * 2


def test_8gb_card_keeps_everything_resident_with_a_reserve():
    p = plan_vram("normal", total_bytes=8 * GiB, free_bytes=int(6.9 * GiB), **COMMON)
    assert p.full_residency and p.enable_mtp and p.ring_slots == 0
    assert p.reserve_bytes > 1 * GiB  # OS/WDDM guard + KV for 8k context + prefill workspace
    assert p.budget_bytes + p.reserve_bytes == int(6.9 * GiB)
    assert p.prefill_chunk_tokens == 512


def test_turbo_uses_more_vram_than_normal():
    n = plan_vram("normal", total_bytes=4 * GiB, free_bytes=int(3.8 * GiB), **COMMON)
    t = plan_vram("turbo", total_bytes=4 * GiB, free_bytes=int(3.8 * GiB), **COMMON)
    assert t.reserve_bytes < n.reserve_bytes
    assert t.budget_bytes > n.budget_bytes
    assert t.resident_layers >= n.resident_layers


def test_4gb_card_streams_through_equal_ring_slots():
    p = plan_vram("normal", total_bytes=4 * GiB, free_bytes=int(3.8 * GiB), **COMMON)
    assert not p.full_residency
    assert p.ring_slots >= 3 and p.slot_bytes == max(LAYERS)
    assert 0 < p.resident_layers < 32
    assert p.weights_bytes <= p.budget_bytes
    # normal mode leaves roughly 0.3-0.8 GiB free on a 4 GB card instead of taking everything
    assert 0.3 * GiB < p.reserve_bytes < 1.0 * GiB
    assert p.prefill_chunk_tokens == 256 and p.max_context_tokens == 4096


def test_fixed_profiles_are_rejected():
    for bad in ("4gb", "8gb", "auto"):
        with pytest.raises(ValueError):
            plan_vram(bad, total_bytes=8 * GiB, free_bytes=7 * GiB, **COMMON)


def test_too_little_vram_is_a_clear_error():
    with pytest.raises(RuntimeError, match="Not enough free VRAM"):
        plan_vram("turbo", total_bytes=2 * GiB, free_bytes=int(1.0 * GiB), **COMMON)


def test_q1_cubes_auto_halves_planes_to_preserve_full_residency(tmp_path):
    """When k=5 exceeds free VRAM, planner steps down by whole halving planes (5 -> 4 -> 3)
    instead of falling back to slow layer streaming, and keeps single-slot ring when streaming.
    Conversely, when a compact dense model fits comfortably inside <= 35% of the VRAM budget,
    auto_fit_q1_planes steps up (refines) up to k=8 while preserving >= 65% headroom."""
    import sys
    from pathlib import Path
    tests_dir = str(Path(__file__).resolve().parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from quintessentia_cli.cli import resolve_auto_v4_plan
    from quintessentia_cli.unified_runtime import plan_for_checkpoint
    from test_ideal_q1_and_modes import _make_tiny_llama

    d = tmp_path / "llama_auto_halve"
    d.mkdir()
    _make_tiny_llama(d, hidden_size=256, intermediate_size=512, num_layers=4, vocab_size=512)

    # Probe exact byte requirements at k=5, k=4, k=3
    _, pl_k5, _ = plan_for_checkpoint(
        d, mode="turbo", total_bytes=8 * GiB, free_bytes=6 * GiB,
        weight_format="q1_cubes", q1_planes=5, max_context_tokens=128, prefill_chunk_tokens=16,
    )
    _, pl_k4, _ = plan_for_checkpoint(
        d, mode="turbo", total_bytes=8 * GiB, free_bytes=6 * GiB,
        weight_format="q1_cubes", q1_planes=4, max_context_tokens=128, prefill_chunk_tokens=16,
    )
    _, pl_k3, _ = plan_for_checkpoint(
        d, mode="turbo", total_bytes=8 * GiB, free_bytes=6 * GiB,
        weight_format="q1_cubes", q1_planes=3, max_context_tokens=128, prefill_chunk_tokens=16,
    )
    assert pl_k3.weights_bytes < pl_k4.weights_bytes < pl_k5.weights_bytes
    reserve = pl_k5.reserve_bytes

    # Case 1: budget sits strictly between k=4 and k=5 -> auto-halves 5 -> 4 with 100% residency!
    free_for_k4 = reserve + pl_k4.weights_bytes + 4096
    assert free_for_k4 - reserve < pl_k5.weights_bytes
    prof4, plan4, _ = plan_for_checkpoint(
        d, mode="turbo", total_bytes=8 * GiB, free_bytes=free_for_k4,
        weight_format="q1_cubes", q1_planes=5, auto_fit_q1_planes=True,
        max_context_tokens=128, prefill_chunk_tokens=16,
    )
    assert plan4.full_residency is True
    assert plan4.resident_layers == 4
    assert prof4["q1_planes"] == 4
    assert prof4["q1_planes_requested"] == 5
    assert prof4["q1_planes_auto_halved"] is True

    # Case 2: budget sits strictly between k=3 and k=4 -> auto-halves 5 -> 3 with 100% residency!
    free_for_k3 = reserve + pl_k3.weights_bytes + 4096
    assert free_for_k3 - reserve < pl_k4.weights_bytes
    prof3, plan3, _ = plan_for_checkpoint(
        d, mode="turbo", total_bytes=8 * GiB, free_bytes=free_for_k3,
        weight_format="q1_cubes", q1_planes=5, auto_fit_q1_planes=True,
        max_context_tokens=128, prefill_chunk_tokens=16,
    )
    assert plan3.full_residency is True
    assert plan3.resident_layers == 4
    assert prof3["q1_planes"] == 3
    assert prof3["q1_planes_auto_halved"] is True

    # Case 3: explicit auto_fit_q1_planes=False preserves requested k=5 and uses honest 1-slot streaming
    prof_fix, plan_fix, _ = plan_for_checkpoint(
        d, mode="turbo", total_bytes=8 * GiB, free_bytes=free_for_k4,
        weight_format="q1_cubes", q1_planes=5, auto_fit_q1_planes=False,
        max_context_tokens=128, prefill_chunk_tokens=16,
    )
    assert prof_fix["q1_planes"] == 5
    assert prof_fix["q1_planes_auto_halved"] is False
    assert plan_fix.full_residency is False
    assert plan_fix.ring_slots == 1

    # Case 4: CLI auto enables auto_fit_q1_planes, while explicit --q1-planes disables it
    auto_plan = resolve_auto_v4_plan(model_dir=d, weight_format="auto", feedback_mode="auto", q1_planes=None, cuda_available=True)
    explicit_plan = resolve_auto_v4_plan(model_dir=d, weight_format="auto", feedback_mode="auto", q1_planes=5, cuda_available=True)
    assert auto_plan["auto_fit_q1_planes"] is True
    assert explicit_plan["auto_fit_q1_planes"] is False

    # Case 5: Abundant VRAM headroom (<= 35% budget used even at k=8) -> mirror-balance refines 5 -> 8
    prof_up, plan_up, _ = plan_for_checkpoint(
        d, mode="turbo", total_bytes=8 * GiB, free_bytes=6 * GiB,
        weight_format="q1_cubes", q1_planes=5, auto_fit_q1_planes=True,
        max_context_tokens=128, prefill_chunk_tokens=16,
    )
    assert plan_up.full_residency is True
    assert prof_up["q1_planes"] == 8
    assert prof_up["q1_planes_auto_halved"] is False
    assert prof_up["q1_planes_auto_refined"] is True


