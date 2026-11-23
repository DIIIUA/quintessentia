"""End-to-end CPU and CUDA acceptance tests for the Wave 2 Unified Primary Engine."""
from __future__ import annotations

from dataclasses import replace
import torch
import pytest

from aircolibri_cli.cli import build_parser
from aircolibri_cli.unified_runtime import (
    AirColibriUnifiedEngine,
    ColibriSpecGate,
    GiB,
    MiB,
    plan_vram,
    q4_bytes_for_shape,
)


def _make_tiny_llama(path, *, hidden_size=64, intermediate_size=128, num_layers=4, vocab_size=259):
    import json
    from transformers import AutoModelForCausalLM, LlamaConfig
    from transformers.convert_slow_tokenizer import bytes_to_unicode
    from test_model_source import _write_tiny_hf_tokenizer

    torch.manual_seed(2026)
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    model.save_pretrained(path)
    _write_tiny_hf_tokenizer(path, vocab_size=vocab_size)
    tok_path = path / "tokenizer.json"
    tok = json.loads(tok_path.read_text(encoding="utf-8"))
    tok["model"]["vocab"] = {"<eos>": 0, **{c: i + 1 for i, c in enumerate(bytes_to_unicode().values())}}
    tok_path.write_text(json.dumps(tok), encoding="utf-8")
    return model


def test_cli_new_flags_and_vram_cube_subdivision_cpu():
    parser = build_parser()
    ask_args = parser.parse_args(
        ["ask", "--speculation", "q1", "--thought-anchor", "--max-slot-mib", "1.5", "Hello"]
    )
    assert ask_args.speculation == "q1"
    assert ask_args.thought_anchor is True
    assert ask_args.max_slot_mib == 1.5

    doc_args = parser.parse_args(["doctor", "--max-slot-mib", "2.0"])
    assert doc_args.max_slot_mib == 2.0

    gate = ColibriSpecGate(max_drafts=2)
    assert ColibriSpecGate.SRC_Q1 == 2
    assert gate.expected_gain(ColibriSpecGate.SRC_Q1, [0.75, 0.55]) > 1.02
    gate.record_draft_cost(ColibriSpecGate.SRC_Q1, 0.001)
    gate.record_result(ColibriSpecGate.SRC_Q1, judged=2, accepted=1, probs=[0.75, 0.55])
    summary = gate.summary()
    assert summary["q1_verifies"] == 1
    assert summary["q1_proposed"] == 2
    assert summary["q1_accepted"] == 1
    assert summary["q1_accept_rate_pct"] == 50.0

    common = dict(
        layer_bytes=[120 * MiB] * 24,
        lm_head_bytes=q4_bytes_for_shape((128000, 4096)),
        mtp_bytes=0,
        kv_bytes_per_token=32768,
        recurrent_state_bytes=0,
        hidden_size=4096,
        intermediate_size=11008,
    )
    # Default allow_cube_subdivision=False preserves legacy RuntimeError
    with pytest.raises(RuntimeError, match="Not enough free VRAM"):
        plan_vram("turbo", total_bytes=2 * GiB, free_bytes=int(0.9 * GiB), **common)

    # Sub-layer Q1 cubes with resident LM head
    p_sub_layer = plan_vram(
        "turbo",
        total_bytes=2 * GiB,
        free_bytes=int(0.9 * GiB),
        allow_cube_subdivision=True,
        **common,
    )
    assert p_sub_layer.sub_layer_cubes is True
    assert p_sub_layer.streamed_lm_head is False
    assert 2 * MiB <= p_sub_layer.slot_bytes < 120 * MiB

    # Sub-layer + streamed LM head when budget cannot hold full LM head
    p_sub_both = plan_vram(
        "turbo",
        total_bytes=2 * GiB,
        free_bytes=int(0.45 * GiB),
        allow_cube_subdivision=True,
        max_context_tokens=1024,
        prefill_chunk_tokens=64,
        **common,
    )
    assert p_sub_both.sub_layer_cubes is True
    assert p_sub_both.streamed_lm_head is True
    assert p_sub_both.slot_bytes >= 2 * MiB


@torch.no_grad()
def test_cpu_only_survivor_tracker_zero_cuda_zero_disk_and_thought_anchor(tmp_path, monkeypatch):
    from test_cpu_independence import forbid_cuda
    from aircolibri_cli.model_source import PreadSafetensorsReader

    _make_tiny_llama(tmp_path)
    forbid_cuda(monkeypatch)

    with AirColibriUnifiedEngine(
        tmp_path,
        device="cpu",
        weight_format="q6",
        thought_anchor=True,
        speculation="off",
        verbose=False,
        max_context_tokens=128,
        prefill_chunk_tokens=4,
    ) as u:
        assert u.survivor_tracker is not None
        assert getattr(u.engine, "probability_warmup", None) is None  # No GPU arena fabricated on CPU-only

        orig_read = PreadSafetensorsReader.read_tensor
        disk_reads_during_survivor = [0]
        orig_recompute = u.survivor_tracker._recompute_survivors

        def wrapped_read(self_r, *a, **kw):
            if getattr(u.survivor_tracker, "_in_recompute_check", False):
                disk_reads_during_survivor[0] += 1
            return orig_read(self_r, *a, **kw)

        def wrapped_recompute():
            u.survivor_tracker._in_recompute_check = True
            try:
                return orig_recompute()
            finally:
                u.survivor_tracker._in_recompute_check = False

        monkeypatch.setattr(PreadSafetensorsReader, "read_tensor", wrapped_read)
        u.survivor_tracker._recompute_survivors = wrapped_recompute

        events = list(
            u.stream_chat(
                [{"role": "user", "content": "Hello world"}],
                max_new_tokens=4,
                enable_thinking=False,
                stop_on_eos=False,
                use_session_cache=False,
            )
        )
        done = events[-1]
        assert done["type"] == "done"
        assert disk_reads_during_survivor[0] == 0
        surv_metrics = done["metrics"]["survivor_probability_path"]
        assert surv_metrics["updates"] >= 1
        assert surv_metrics["survivors"] > 0
        assert surv_metrics["anchor_applications"] >= 1
        assert u.survivor_tracker.survivor_log_prior is not None
        assert float(u.survivor_tracker.survivor_log_prior.abs().max().item()) <= 0.125 + 1e-6
        assert "thought_anchor" in done["metrics"]
        snap = u.status_snapshot()
        assert snap["unified_primary_engine"] is True
        assert snap["thought_anchor_enabled"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_default_unified_primary_dense_and_moe_auto_attach_and_manual_override_cuda(tmp_path, monkeypatch):
    import aircolibri_cli.unified_runtime as runtime
    from test_probability_warmup import make_model

    dense_dir = tmp_path / "dense_llama"
    dense_dir.mkdir()
    _make_tiny_llama(dense_dir, num_layers=5)

    orig_plan = runtime.plan_for_checkpoint

    def force_streaming(*args, **kwargs):
        profile, plan, mtp = orig_plan(*args, **kwargs)
        return profile, replace(plan, full_residency=False, resident_layers=1, ring_slots=3), mtp

    monkeypatch.setattr(runtime, "plan_for_checkpoint", force_streaming)

    # 1. Default Dense GPU-first auto-attaches DynamicWorkingMemory(cache=None)
    with AirColibriUnifiedEngine(
        dense_dir,
        cpu_first=False,
        weight_format="q6",
        speculation="off",
        verbose=False,
        max_context_tokens=128,
        prefill_chunk_tokens=8,
    ) as u_dense:
        warm = u_dense.engine.probability_warmup
        assert warm is not None and warm._auto_unified_attached is True
        done = list(
            u_dense.stream_chat(
                [{"role": "user", "content": "Hi"}],
                max_new_tokens=4,
                enable_thinking=False,
                stop_on_eos=False,
            )
        )[-1]
        assert done["type"] == "done"
        sm = warm.stats()["shared_memory"]
        assert sm["area_transitions"]["B->A"] == 0
        assert sm["area_transitions"]["A->B"] > 0
        assert sm["roles"]["C"]["used_bytes"] == 0
        assert sm["peak_dense_blocks"] <= 2

    # 2. Default MoE GPU-first auto-attaches DynamicWorkingMemory + allows manual override before step 0
    moe_dir = tmp_path / "moe_granite"
    moe_dir.mkdir()
    make_model(moe_dir, "granitemoe")

    def force_moe_streaming(*args, **kwargs):
        profile, plan, mtp = orig_plan(*args, **kwargs)
        expert_bytes = int(3 * 128 * 128 * (4 / 8 + 4 / 64))
        temp_reserve = 8 * 6 * 128 * 128 + 32 * MiB
        budget = plan.weights_bytes + 256 * MiB + temp_reserve + 24 * expert_bytes
        return profile, replace(plan, full_residency=False, resident_layers=2, ring_slots=3, budget_bytes=budget), mtp

    monkeypatch.setattr(runtime, "plan_for_checkpoint", force_moe_streaming)
    with AirColibriUnifiedEngine(
        moe_dir,
        cpu_first=False,
        weight_format="q4",
        enable_mtp=False,
        verbose=False,
        max_context_tokens=128,
        prefill_chunk_tokens=8,
    ) as u_moe:
        assert u_moe.engine.probability_warmup is not None
        assert u_moe.engine.probability_warmup._auto_unified_attached is True
        # Explicit manual override before generation replaces auto-attached cleanly
        manual = u_moe.enable_unified_memory(region_experts=4, host_cache_bytes=2 * MiB)
        assert manual._auto_unified_attached is False
        # Duplicate manual call still raises RuntimeError
        with pytest.raises(RuntimeError, match="already enabled"):
            u_moe.enable_unified_memory(region_experts=4, host_cache_bytes=2 * MiB)
        for tokens in ([1, 3, 5, 7], [9], [11], [13]):
            logits, past = u_moe.engine._forward_step(
                torch.tensor([tokens]),
                past_key_values=None if len(tokens) > 1 else past,
                prefetch_next=True,
            )
            assert torch.isfinite(logits).all()
        sm_moe = manual.stats()["shared_memory"]
        assert sm_moe["area_transitions"]["B->A"] == 0
        assert sm_moe["roles"]["C"]["used_bytes"] == 0
        assert len(manual.cache.shared_memory._pinned_resident_pos) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_progressive_no_pingpong_and_auto_unified_takeover_cuda(tmp_path, monkeypatch):
    import threading
    import aircolibri_cli.unified_runtime as runtime

    _make_tiny_llama(tmp_path, hidden_size=128, intermediate_size=256, num_layers=5)
    orig_plan = runtime.plan_for_checkpoint
    hold_ring = threading.Event()

    def streaming_plan(*args, **kwargs):
        profile, plan, mtp = orig_plan(*args, **kwargs)
        return profile, replace(plan, full_residency=False, resident_layers=2, ring_slots=3), mtp

    monkeypatch.setattr(runtime, "plan_for_checkpoint", streaming_plan)
    with AirColibriUnifiedEngine(
        tmp_path,
        progressive_start=True,
        weight_format="q6",
        speculation="off",
        verbose=False,
        max_context_tokens=128,
        prefill_chunk_tokens=4,
    ) as u:
        assert u.engine.weight_manager.ram_cache.max_cached_layers == 5
        assert u.engine.probability_warmup is not None
        assert u.engine.progressive.backend_ready.wait(30)

        # Verify progressive _after(0) keeps activation on GPU when layer 1 is also GPU-ready
        p = u.engine.progressive
        layer0_out_devices = []
        orig_before_1 = u.engine._decoder_layers()[1].register_forward_pre_hook(
            lambda mod, args: layer0_out_devices.append(args[0].device.type)
        )
        try:
            with p.lock:
                for idx in (0, 1):
                    if idx not in p.ready:
                        p.ready[idx] = p._copy_payload(
                            u.engine.weight_manager.read_layer_tensors(idx, target_device=torch.device("cpu"))
                        )
            # Clear bundle_ready if the worker raced ahead, so this step runs through per-layer hooks
            p.bundle_ready.clear()
            _, past = u.engine._forward_step(torch.tensor([[3, 7, 4]]), None)
            assert layer0_out_devices == ["cuda"], f"Expected GPU->GPU handoff without CPU ping-pong, got {layer0_out_devices}"
            # Now let bundle_ready finish (or set it if worker already exited after seeing len(ready)==2)
            if not p.bundle_ready.wait(5) and len(p.ready) == p.plan.resident_layers:
                from src.bounded_memory_manager import BoundedPinnedBufferPool
                from src.gpu_ring_pipeline import GPURingPipeline
                if p.gpu_pipeline is None:
                    p.gpu_pool = BoundedPinnedBufferPool(
                        arena_layout=u.engine.weight_manager.pinned_pool.layout,
                        num_slots=p.plan.ring_slots, dtype=u.engine.dtype, device=p.gpu
                    )
                    p.gpu_pipeline = GPURingPipeline(
                        num_slots=p.plan.ring_slots, device=p.gpu, dtype=u.engine.dtype,
                        tail_to_head_prefetch=True, arena_layout=p.gpu_pool.layout
                    )
                p.bundle_ready.set()
            assert p.bundle_ready.is_set()
            _, past = u.engine._forward_step(torch.tensor([[9]]), past)
            assert p.complete is True
            assert u.engine.probability_warmup.shared_memory is not None
            _, past = u.engine._forward_step(torch.tensor([[11]]), past)
            sm = u.engine.probability_warmup.stats()["shared_memory"]
            assert sm["area_transitions"]["B->A"] == 0
            assert sm["roles"]["C"]["used_bytes"] == 0
        finally:
            orig_before_1.remove()
            hold_ring.set()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_sub_layer_and_sub_lm_head_q1_batched_streaming_and_q1_speculation_cuda(tmp_path):
    from src.q1_cube_engine import Q1StreamedLMHead, Q1StreamedLinear

    _make_tiny_llama(tmp_path, hidden_size=128, intermediate_size=512, num_layers=3, vocab_size=512)
    prompt = [{"role": "user", "content": "1 2 3 1 2 3"}]

    # Reference output with speculation="off" under tight 0.125 MiB (128 KiB) per-slot cap
    # (128 KiB < single layer and < lm_head 512*128*2 = 128 KiB BF16 -> triggers both!)
    with AirColibriUnifiedEngine(
        tmp_path,
        cpu_first=False,
        weight_format="q6",
        speculation="off",
        max_slot_bytes=96 * 1024,
        verbose=False,
        max_context_tokens=128,
        prefill_chunk_tokens=8,
    ) as u_ref:
        assert u_ref.plan.sub_layer_cubes is True
        assert u_ref.plan.streamed_lm_head is True
        assert isinstance(u_ref.engine.model.lm_head, Q1StreamedLMHead)
        assert any(isinstance(m, Q1StreamedLinear) for m in u_ref.engine.model.modules())
        done_ref = list(
            u_ref.stream_chat(prompt, max_new_tokens=6, enable_thinking=False, stop_on_eos=False)
        )[-1]
        ref_text = done_ref["answer_text"]
        arena_stats = done_ref["metrics"]["q1_role_arena"]
        assert arena_stats["peak_slot_bytes"] <= 96 * 1024
        assert arena_stats["transitions"]["B->A"] == 0
        assert arena_stats["area_used_bytes"][arena_stats["standby_area"]] == 0
        assert arena_stats["nvrtc_batch_launches"] > 0
        assert arena_stats["async_h2d_transfers"] > 0
        assert u_ref.engine.model.lm_head.candidate_calls >= 1

    # Q1 self-speculation (SRC_Q1) on the same streamed Q1 engine preserves exact lossless output!
    with AirColibriUnifiedEngine(
        tmp_path,
        cpu_first=False,
        weight_format="q6",
        speculation="q1",
        max_slot_bytes=96 * 1024,
        verbose=False,
        max_context_tokens=128,
        prefill_chunk_tokens=8,
    ) as u_q1:
        # Force gate to accept testing depth >= 1 so Q1SkeletonDrafter + RowwiseVerification execute
        orig_pick = ColibriSpecGate.pick_depth
        ColibriSpecGate.pick_depth = lambda self_g, src, probs: min(len(probs), self_g.max_drafts)
        try:
            done_q1 = list(
                u_q1.stream_chat(prompt, max_new_tokens=6, enable_thinking=False, stop_on_eos=False)
            )[-1]
        finally:
            ColibriSpecGate.pick_depth = orig_pick
        assert done_q1["answer_text"] == ref_text
        sg = done_q1["metrics"]["spec_gate"]
        assert sg["q1_verifies"] > 0
        assert sg["source"] == "q1_skeleton"
        assert all(
            getattr(m, "active_planes", None) is None
            for m in u_q1.engine.model.modules()
            if isinstance(m, (Q1StreamedLinear, Q1StreamedLMHead))
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_dense_three_page_retiring_reuse_waits_on_gpu_fence_cuda():
    from src.bounded_memory_manager import SlotArenaLayout
    from src.dynamic_working_memory import DynamicWorkingMemory
    from src.gpu_ring_pipeline import GPURingPipeline

    layout = SlotArenaLayout.from_templates(
        {"weight": torch.empty(256, 256, dtype=torch.bfloat16)},
        slot_alignment=256,
        tensor_alignment=256,
    )
    ring = GPURingPipeline(num_slots=3, device=torch.device("cuda"), dtype=torch.bfloat16, arena_layout=layout)
    memory = DynamicWorkingMemory(ring, None)
    slow = torch.cuda.Stream()
    observed = []
    expected = [torch.full((256, 256), float(i + 1), dtype=torch.bfloat16).pin_memory() for i in range(4)]
    try:
        assert memory.num_slots == 3
        memory.begin_layer(0)
        s0 = ring.get_slot_for_loading(0)
        ring.load_tensors_async(s0, {"weight": expected[0]})
        s1 = ring.get_slot_for_loading(1)
        ring.load_tensors_async(s1, {"weight": expected[1]})

        with torch.cuda.stream(slow):
            ring.acquire_slot_for_compute(0)
            torch.cuda._sleep(150_000_000)
            observed.append(s0.active_tensors["weight"].clone())
            ring.release_slot_after_compute(s0)

            memory.begin_layer(1)
            s2 = ring.get_slot_for_loading(2)
            ring.load_tensors_async(s2, {"weight": expected[2]})
            ring.acquire_slot_for_compute(1)
            torch.cuda._sleep(150_000_000)
            observed.append(s1.active_tensors["weight"].clone())
            ring.release_slot_after_compute(s1)

            # At layer 2: page for s2 is active, and BOTH s0 and s1 pages are still retiring on `slow`!
            # bind_dense for layer 3 must enqueue transfer_stream.wait_event without raising RuntimeError.
            memory.begin_layer(2)
            s3 = ring.get_slot_for_loading(3)
            ring.load_tensors_async(s3, {"weight": expected[3]})
            ring.acquire_slot_for_compute(2)
            observed.append(s2.active_tensors["weight"].clone())
            ring.release_slot_after_compute(s2)

            memory.begin_layer(3)
            ring.acquire_slot_for_compute(3)
            observed.append(s3.active_tensors["weight"].clone())
            ring.release_slot_after_compute(s3)

        slow.synchronize()
        torch.cuda.synchronize()
        for obs, exp in zip(observed, expected):
            assert torch.equal(obs.cpu(), exp)
        assert memory.stats()["area_transitions"]["B->A"] == 0
        assert memory.stats()["roles"]["C"]["used_bytes"] == 0
    finally:
        slow.synchronize()
        ring.close()
