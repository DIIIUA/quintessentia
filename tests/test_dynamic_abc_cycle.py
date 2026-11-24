"""Strict monotonic A -> B -> C(0) -> A cycle, lookahead, retirement, and parity checks."""
from types import SimpleNamespace
import threading

import pytest
import torch

from src.dynamic_working_memory import DynamicWorkingMemory
from src.gpu_ring_pipeline import GPURingPipeline
from src.quantized_projection import pack_weight, packed_specs
from src.unified_prefetch import UnifiedExpertPrefetch


def _make_cpu_memory(bits: int = 4, extra_slots: int = 24, warm_cap: int = 8):
    torch.manual_seed(91)
    weights = torch.randn(48, 3, 128, 128, dtype=torch.bfloat16) * 0.04
    specs_gu = packed_specs("weight", (256, 128), bits, 64)
    specs_dn = packed_specs("weight", (128, 128), bits, 64)
    expert_bytes = sum(
        torch.empty((), dtype=dt).element_size() * sh[0] * sh[1]
        for sh, dt in list(specs_gu.values()) + list(specs_dn.values())
    )
    device = torch.device("cpu")

    def _load_packed(layer: int, eid: int):
        w = weights[eid % len(weights)]
        return (pack_weight(torch.cat((w[0], w[1])), bits, 64), pack_weight(w[2], bits, 64))

    cache = SimpleNamespace(
        device=device,
        bits=bits,
        group_size=64,
        I=128,
        H=128,
        expert_bytes=expert_bytes,
        temporary_reserve_bytes=32 * 1024**2,
        num_slots=max(8, warm_cap),
        _lru={},
        shared_memory=None,
        _load_packed=_load_packed,
    )
    ring = GPURingPipeline({"weight": ((128, 128), torch.bfloat16)}, device=device)
    memory = DynamicWorkingMemory(
        ring,
        cache,
        extra_slots * expert_bytes,
        min_region_experts=4,
        allow_cpu_mock=True,
        enable_prefetch_timing=False,
    )
    return weights, cache, ring, memory


def test_strict_monotonic_abc_cycle_and_empty_standby_cpu():
    _, _, ring, memory = _make_cpu_memory(extra_slots=24)
    try:
        # Include non-sequential layer indices (0 -> 1 -> 0 -> 2 -> 0) to catch any % 3 inversion.
        sequence = [0, 1, 0, 2, 0, 1, 2, 3, 1, 0, 2, 4, 5, 6]
        prev_roles = None
        for layer in sequence:
            memory.begin_layer(layer)
            stats = memory.stats()
            roles = stats["physical_area_roles"]
            assert set(roles.values()) == {"A", "B", "C"}
            assert stats["roles"]["C"]["used_bytes"] == 0
            assert stats["roles"]["C"]["reserved_bytes"] == 0
            if prev_roles is not None:
                for area_id in (0, 1, 2):
                    expected_next = {"A": "B", "B": "C", "C": "A"}[prev_roles[area_id]]
                    assert roles[area_id] == expected_next
            prev_roles = dict(roles)

        transitions = memory.stats()["area_transitions"]
        assert transitions["B->A"] == 0
        assert transitions["A->C"] == 0
        assert transitions["C->B"] == 0
        assert transitions["A->B"] > 0 and transitions["B->C"] > 0 and transitions["C->A"] > 0
    finally:
        ring.close()


def test_lookahead_zero_obsolete_after_pack_and_reset_cpu():
    weights, cache, ring, memory = _make_cpu_memory(extra_slots=24)
    entered, release = threading.Event(), threading.Event()
    loads = []

    def load_cpu(layer: int, eid: int):
        loads.append((layer, eid))
        if (layer, eid) == (0, 1):
            entered.set()
            assert release.wait(5)
        w = weights[eid]
        return tuple(w[i] for i in range(3))

    worker = UnifiedExpertPrefetch(memory, cache, capacity=4, load_cpu=load_cpu, close_cpu=lambda: None, host_cache_bytes=2**20)
    try:
        memory.begin_layer(0)
        worker.submit(0, [0, 1], lane=0, region=(0, 1), step=1)
        assert entered.wait(5)

        # Advance to layer 1 while (0, 1) is still packing; lookahead keeps lane 0 alive in A.
        memory.begin_layer(1)
        worker.submit(1, [2, 3], lane=1, region=(1, 1), step=1)
        assert len(memory.desired) <= memory.warm_capacity
        assert (0, 1) in memory.active_protected

        release.set()
        worker.drain()
        assert worker.counters["obsolete_after_pack"] == 0
        assert (0, 1) in memory.directory
        assert (1, 2) in memory.directory and (1, 3) in memory.directory

        # Verify reset clears active protection/desired and keeps C empty.
        worker.reset()
        stats = memory.stats()
        assert not memory.desired and not memory.active_protected
        assert stats["roles"]["C"]["used_bytes"] == 0 and stats["roles"]["C"]["reserved_bytes"] == 0
        assert worker.stats()["shared_gpu_bytes"] == memory.gpu_bytes
    finally:
        release.set()
        worker.close()
        ring.close()


def test_dense_reclaims_expert_zero_and_demand_yield_cpu():
    weights, cache, ring, memory = _make_cpu_memory(extra_slots=32)
    payload = {"weight": torch.randn(128, 128, dtype=torch.bfloat16)}
    worker = UnifiedExpertPrefetch(
        memory,
        cache,
        capacity=4,
        load_cpu=lambda layer, eid: tuple(weights[eid % len(weights)]),
        close_cpu=lambda: None,
        host_cache_bytes=2**20,
    )
    try:
        for i in range(12):
            with memory.lease(i // 4, [i % 4], cache._load_packed):
                pass
        for layer in range(6):
            memory.begin_layer(layer)
            slot = ring.get_slot_for_loading(layer)
            ring.load_tensors_async(slot, payload)
            ring.acquire_slot_for_compute(layer)
            ring.release_slot_after_compute(slot)
        assert memory.stats()["dense_reclaims_expert"] == 0

        # Simulate active cold demand reads; worker yields briefly without starving steady-state.
        memory.demand_inflight = 1
        worker.submit(2, [5, 6], lane=2, region=(2, 1), step=2)
        memory.demand_inflight = 0
        worker.drain()
        st = worker.stats()
        assert st["published"] >= 2
        assert st["timeline"] == []
    finally:
        memory.demand_inflight = 0
        worker.close()
        ring.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_retiring_pages_isolate_live_cuda_readers_and_strict_abc_gpu():
    from test_unified_working_memory import make_memory

    weights, cache, ring, _ = make_memory(4, unify=False)
    memory = DynamicWorkingMemory(ring, cache, 24 * cache.expert_bytes, min_region_experts=4)
    slow = torch.cuda.Stream()
    try:
        memory.begin_layer(0)
        packed = (
            pack_weight(torch.cat((weights[5, 0], weights[5, 1])), 4),
            pack_weight(weights[5, 2], 4),
        )
        memory.set_desired({(0, 5)})
        status, record = memory.prepare((0, 5), packed, torch.cuda.current_stream(), memory.epoch)
        assert status == "published"
        torch.cuda.synchronize()
        snapshot = memory._views(record)[0]["packed_lo"].clone()

        # Launch a slow CUDA reader while record is still in B, then drop from desired and rotate B -> C.
        with torch.cuda.stream(slow):
            with memory.lease(0, [5], cache._load_packed) as entries:
                torch.cuda._sleep(250_000_000)
                observed = entries[0]["gu"]["packed_lo"].clone()
        memory.set_desired(set())
        memory.begin_layer(1)

        mid_stats = memory.stats()
        assert mid_stats["retiring_pages"] >= 1
        assert mid_stats["roles"]["C"]["used_bytes"] == 0
        assert mid_stats["roles"]["C"]["reserved_bytes"] == 0
        assert mid_stats["area_transitions"]["B->A"] == 0

        slow.synchronize()
        assert torch.equal(observed, snapshot)
        memory.begin_layer(2)
        assert memory.stats()["retiring_pages"] == 0
    finally:
        slow.synchronize()
        ring.close()
        cache.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["granitemoe", "olmoe"])
@torch.no_grad()
def test_granite_olmoe_dynamic_abc_parity_and_invariants_gpu(tmp_path, kind):
    from quintessentia_cli.model_source import open_model_source
    from src.quintessentia_integrated import QuintessentiaModel
    from test_probability_warmup import make_model

    make_model(tmp_path, kind)
    expert_bytes = int(3 * 128 * 128 * (4 / 8 + 4 / 64))
    budget = 8 * 6 * 128 * 128 + 32 * 1024**2 + 8 * expert_bytes
    kwargs = dict(
        weight_format="q4",
        universal_quantization=True,
        vram_resident_layers=1,
        num_slots=3,
        expert_cache_bytes=budget,
    )
    steps = ([1, 4, 6, 8], [7], [9], [11], [13], [15], [17], [19])
    expected = []
    with QuintessentiaModel(str(tmp_path), model_source=open_model_source(tmp_path), **kwargs) as base:
        past = None
        for tokens in steps:
            logits, past = base._forward_step(torch.tensor([tokens]), past_key_values=past)
            expected.append((logits.cpu(), [(l.keys.cpu(), l.values.cpu()) for l in past.layers]))

    with QuintessentiaModel(str(tmp_path), model_source=open_model_source(tmp_path), **kwargs) as engine:
        warm = engine.enable_probability_warmup(
            region_experts=4,
            gpu_budget_bytes=2**20,
            host_cache_bytes=2**20,
            unified_memory=True,
            dynamic_regions=True,
        )
        past = None
        for tokens, (ref_logits, ref_kv) in zip(steps, expected):
            logits, past = engine._forward_step(torch.tensor([tokens]), past_key_values=past)
            assert torch.equal(logits.cpu(), ref_logits)
            assert all(
                torch.equal(l.keys.cpu(), rk) and torch.equal(l.values.cpu(), rv)
                for l, (rk, rv) in zip(past.layers, ref_kv)
            )
        warm.prefetch.drain()
        mem_stats = warm.cache.shared_memory.stats()
        pref_stats = warm.prefetch.stats()
        assert mem_stats["area_transitions"]["B->A"] == 0
        assert mem_stats["area_transitions"]["A->B"] > 0
        assert mem_stats["area_transitions"]["B->C"] > 0
        assert mem_stats["area_transitions"]["C->A"] > 0
        assert mem_stats["roles"]["C"]["used_bytes"] == 0
        assert mem_stats["roles"]["C"]["reserved_bytes"] == 0
        assert pref_stats["obsolete_after_pack"] == 0

        engine.reset_after_cancel()
        assert not warm.region_map.initial
        post_reset = warm.cache.shared_memory.stats()
        assert post_reset["roles"]["C"]["used_bytes"] == 0
        assert post_reset["roles"]["C"]["reserved_bytes"] == 0
        assert post_reset["dense_blocks"] == 0
