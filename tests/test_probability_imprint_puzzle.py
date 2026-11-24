"""Tests for ProbabilityImprintGraph + ImprintPuzzleMoEOverlay (`src/probability_imprint_puzzle.py`)."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.binary_halving_q1 import LazyMoEBinaryHalvingCache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_imprint_depth_shedding_eliminates_lru_thrashing_cuda():
    """When budget holds all experts at k=2..3 but NOT at k=5, naive LRU thrashes on every
    multi-layer pass while ImprintPuzzleMoEOverlay sheds trailing Q1 planes in VRAM and hits 0 recarves!
    """
    torch.manual_seed(2026)
    dev = torch.device("cuda:0")
    num_layers, num_exp = 8, 8
    h, inter = 256, 128

    # Fixed read-only synthetic expert bank
    bank = {
        (l, e): (
            torch.randn(2 * inter, h, dtype=torch.bfloat16) * 0.04,
            torch.randn(h, inter, dtype=torch.bfloat16) * 0.04,
        )
        for l in range(num_layers)
        for e in range(num_exp)
    }

    # Expert size: base ~12 KB, each plane = 12 KB -> k=5 ~72 KB, k=2 ~36 KB.
    # 48 experts (8 layers x 6 active) at k=5 = ~3.45 MB; at k=2..3 = ~1.9 MB.
    # Choose budget = 2.2 MB: naive LRU at k=5 can hold only ~30 experts -> thrashes on EVERY pass!
    budget = int(2.2 * 1024 * 1024)
    active_eids = [0, 1, 2, 3, 4, 5]
    idx = torch.tensor([active_eids], device=dev, dtype=torch.long)
    w = torch.tensor([[0.32, 0.22, 0.16, 0.12, 0.10, 0.08]], device=dev, dtype=torch.bfloat16)
    x = torch.randn(1, h, device=dev, dtype=torch.bfloat16)

    c_lru = LazyMoEBinaryHalvingCache(
        lambda li, eid: bank[(li, eid)],
        num_layers=num_layers, num_experts=num_exp, device=dev,
        store_cubes=5, root_bits=16, max_cache_bytes=budget, host_cache_bytes=0,
        imprint_puzzle=False,
    )
    c_imp = LazyMoEBinaryHalvingCache(
        lambda li, eid: bank[(li, eid)],
        num_layers=num_layers, num_experts=num_exp, device=dev,
        store_cubes=16, root_bits=16, max_cache_bytes=budget, host_cache_bytes=0,
        imprint_puzzle=True, imprint_depths=(5, 4, 3, 2), expert_rest_tol=0.0,
    )
    try:
        for _pass in range(3):
            for li in range(num_layers):
                y_lru = c_lru.forward_layer_moe(li, x, idx, w, F.silu)
                y_imp = c_imp.forward_layer_moe(li, x, idx, w, F.silu)
                cos = float(F.cosine_similarity(y_imp.float(), y_lru.float()).item())
                assert cos > 0.94, f"Unexpected quality drop at layer {li}: cos={cos}"
        torch.cuda.synchronize()
        st_lru = c_lru.stats()
        st_imp = c_imp.stats()
        imp_recarves = st_imp["misses"] - st_imp["unique_experts_carved"]
        lru_recarves = st_lru["misses"] - st_lru["unique_experts_carved"]
        assert lru_recarves >= 48, f"Expected naive LRU to thrash, got lru_recarves={lru_recarves}"
        assert imp_recarves == 0, f"Imprint Depth-Shedding should eliminate recarves, got {imp_recarves}"
        assert st_imp["imprint_puzzle"]["depth_sheds"] > 0
        rel = st_imp["imprint_puzzle"]["relay"]
        assert rel["transitions"]["B->A"] == 0
        assert rel["standby_bytes"] == 0
        assert rel["rotations"] == rel["zero_copy_adoptions"] > 0
    finally:
        c_lru.close()
        c_imp.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_imprint_puzzle_coalesced_abc_relay_and_accuracy_cuda():
    """Verify coalesced prefill chunk delivery via B, 3D face + 90° corner graph growth,
    Mirror x UNI rest on decode, and strict A->B->C(0)->A invariants.
    """
    torch.manual_seed(314)
    dev = torch.device("cuda:0")
    num_layers, num_exp = 4, 16
    h, inter = 256, 128

    bank = {
        (l, e): (
            torch.randn(2 * inter, h, dtype=torch.bfloat16) * 0.04,
            torch.randn(h, inter, dtype=torch.bfloat16) * 0.04,
        )
        for l in range(num_layers)
        for e in range(num_exp)
    }

    batch_calls = []

    def batch_reader(li, eids):
        batch_calls.append((li, list(eids)))
        return {e: bank[(li, e)] for e in eids}

    c_imp = LazyMoEBinaryHalvingCache(
        lambda li, e: bank[(li, e)],
        expert_batch_reader=batch_reader,
        num_layers=num_layers, num_experts=num_exp, device=dev,
        store_cubes=16, root_bits=16, max_cache_bytes=64 * 1024**2,
        imprint_puzzle=True, imprint_depths=(5, 4, 3, 2), expert_rest_tol=0.24, expert_rest_min=3,
    )
    try:
        # Prefill with M=12 tokens across 4 layers
        x_pre = torch.randn(12, h, device=dev, dtype=torch.bfloat16)
        idx_pre = torch.randint(0, num_exp, (12, 8), device=dev)
        w_pre = F.softmax(torch.randn(12, 8, device=dev) * 2.0, dim=-1).to(torch.bfloat16)
        for li in range(num_layers):
            y = c_imp.forward_layer_moe(li, x_pre, idx_pre, w_pre, F.silu)
            assert y.shape == (12, h) and bool(torch.isfinite(y).all())

        # Decode with M=1 token
        x_dec = torch.randn(1, h, device=dev, dtype=torch.bfloat16)
        idx_dec = idx_pre[:1]
        w_dec = torch.tensor([[0.55, 0.15, 0.10, 0.05, 0.05, 0.04, 0.03, 0.03]], device=dev, dtype=torch.bfloat16)
        for li in range(num_layers):
            y_d = c_imp.forward_layer_moe(li, x_dec, idx_dec, w_dec, F.silu)
            assert y_d.shape == (1, h) and bool(torch.isfinite(y_d).all())

        torch.cuda.synchronize()
        st = c_imp.stats()
        imp = st["imprint_puzzle"]
        assert imp["graph"]["total_face_links"] > 0
        assert imp["graph"]["total_corner90_links"] > 0
        assert imp["coalesced_batches_loaded"] > 0
        assert any(len(eids) > 1 for _, eids in batch_calls), "Expected coalesced multi-expert reads in B"
        rel = imp["relay"]
        assert rel["transitions"]["B->A"] == 0
        assert rel["standby_bytes"] == 0
        assert rel["rotations"] == rel["zero_copy_adoptions"] > 0
    finally:
        c_imp.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_imprint_oom_proof_elastic_vram_regulator_and_core_protection_cuda():
    """Verify the elastic OOM-proof VRAM regulator breathes up into free VRAM, protects the hot
    imprint core at k>=4 under normal shedding, and contracts safely without OOM under sudden
    external VRAM pressure.
    """
    torch.manual_seed(404)
    dev = torch.device("cuda:0")
    num_layers, num_exp = 6, 8
    h, inter = 512, 256  # larger experts (~250 KB at k=5) so shedding & budget breathing are clearly visible

    bank = {
        (l, e): (
            torch.randn(2 * inter, h, dtype=torch.bfloat16) * 0.03,
            torch.randn(h, inter, dtype=torch.bfloat16) * 0.03,
        )
        for l in range(num_layers)
        for e in range(num_exp)
    }

    # Start with a small 1 MiB budget and auto_vram_adapt=True -> must expand automatically!
    c_imp = LazyMoEBinaryHalvingCache(
        lambda li, e: bank[(li, e)],
        num_layers=num_layers, num_experts=num_exp, device=dev,
        store_cubes=16, root_bits=16, max_cache_bytes=1 * 1024**2, host_cache_bytes=16 * 1024**2,
        imprint_puzzle=True, imprint_depths=(5, 4, 3, 2), expert_rest_tol=0.0,
        imprint_auto_vram=True,
    )
    pressure_blocks = []
    try:
        x = torch.randn(1, h, device=dev, dtype=torch.bfloat16)
        idx = torch.tensor([[0, 1, 2, 3, 4, 5]], device=dev, dtype=torch.long)
        w = torch.tensor([[0.40, 0.25, 0.14, 0.09, 0.07, 0.05]], device=dev, dtype=torch.bfloat16)

        for li in range(num_layers):
            y = c_imp.forward_layer_moe(li, x, idx, w, F.silu)
            assert bool(torch.isfinite(y).all())
        torch.cuda.synchronize()
        st1 = c_imp.stats()["imprint_puzzle"]
        assert st1["auto_vram_adapt"] is True
        assert st1["vram_expansions"] >= 1, f"Expected elastic expansion into free VRAM: {st1}"

        # Simulate sudden external VRAM pressure leaving less than safety_reserve free (physical NVML!)
        reserve = c_imp._imprint_overlay.vram_safety_reserve_bytes
        for _ in range(32):
            free_now, _tot, _src = c_imp._imprint_overlay._query_physical_vram()
            if free_now < int(reserve * 0.85):
                break
            chunk_b = min(256 * 1024**2, max(64 * 1024**2, int(free_now - reserve * 0.50)))
            try:
                blk = torch.zeros((chunk_b // 4,), dtype=torch.float32, device=dev)
                pressure_blocks.append(blk)
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                break

        # Run again under tight VRAM pressure -> MUST contract automatically and NOT OOM!
        for li in range(num_layers):
            y = c_imp.forward_layer_moe(li, x, idx, w, F.silu)
            assert bool(torch.isfinite(y).all())
        torch.cuda.synchronize()
        st2 = c_imp.stats()["imprint_puzzle"]
        assert st2["vram_contractions"] >= 1, f"Expected elastic contraction under VRAM pressure: {st2}"
        assert st2["relay"]["transitions"]["B->A"] == 0
        assert st2["relay"]["standby_bytes"] == 0

        # Release external VRAM pressure -> must breathe back up!
        pressure_blocks.clear()
        torch.cuda.empty_cache()
        exp_before = st2["vram_expansions"]
        for li in range(num_layers):
            y = c_imp.forward_layer_moe(li, x, idx, w, F.silu)
            assert bool(torch.isfinite(y).all())
        torch.cuda.synchronize()
        st3 = c_imp.stats()["imprint_puzzle"]
        assert st3["vram_expansions"] > exp_before, f"Expected elastic re-expansion after pressure lifted: {st3}"
    finally:
        pressure_blocks.clear()
        c_imp.close()
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_imprint_nvml_and_resident_draft_zero_disk_cuda():
    """Verify NVML physical VRAM sensor and zero-disk `draft_only_resident` burst drafting."""
    torch.manual_seed(909)
    dev = torch.device("cuda:0")
    num_layers, num_exp = 2, 8
    h, inter = 256, 128

    bank = {
        (l, e): (
            torch.randn(2 * inter, h, dtype=torch.bfloat16) * 0.04,
            torch.randn(h, inter, dtype=torch.bfloat16) * 0.04,
        )
        for l in range(num_layers)
        for e in range(num_exp)
    }
    disk_calls = []

    def reader(li, eid):
        disk_calls.append((li, eid))
        return bank[(li, eid)]

    c_imp = LazyMoEBinaryHalvingCache(
        reader, num_layers=num_layers, num_experts=num_exp, device=dev,
        store_cubes=16, root_bits=16, max_cache_bytes=16 * 1024**2,
        imprint_puzzle=True, imprint_depths=(5, 4, 3, 2), expert_rest_tol=0.0,
        imprint_auto_vram=True,
    )
    try:
        x = torch.randn(1, h, device=dev, dtype=torch.bfloat16)
        idx_warm = torch.tensor([[0, 1, 2, 3]], device=dev, dtype=torch.long)
        w_warm = torch.tensor([[0.4, 0.3, 0.2, 0.1]], device=dev, dtype=torch.bfloat16)
        for li in range(num_layers):
            c_imp.forward_layer_moe(li, x, idx_warm, w_warm, F.silu)
        torch.cuda.synchronize()
        reads_after_warm = len(disk_calls)
        assert reads_after_warm == 8
        ov = c_imp._imprint_overlay
        assert ov.stats()["vram_sensor"] in ("nvml", "cuda")

        # Draft mode touching 2 warm experts (0,1) + 2 cold experts (6,7): MUST do 0 disk reads!
        ov.draft_only_resident = True
        ov.draft_step_hits = ov.draft_step_total = 0
        idx_mix = torch.tensor([[0, 1, 6, 7]], device=dev, dtype=torch.long)
        for li in range(num_layers):
            y_d = c_imp.forward_layer_moe(li, x, idx_mix, w_warm, F.silu)
            assert bool(torch.isfinite(y_d).all())
        ov.draft_only_resident = False
        assert len(disk_calls) == reads_after_warm, "draft_only_resident MUST NEVER touch disk!"
        assert ov.draft_step_total == 8
        assert ov.draft_step_hits == 4

        # Direct per-layer Imprint Fixation (`fixation_cov_thresh = 0.75`):
        # Case A (Fixated): experts 0,1,2 are resident (weight 0.4+0.3+0.2 = 0.90 >= 0.75), expert 6 is cold -> 0 disk reads!
        ov.fixation_cov_thresh = 0.75
        ov.rest_min = 3
        idx_fix = torch.tensor([[0, 1, 2, 6]], device=dev, dtype=torch.long)
        y_fix = c_imp.forward_layer_moe(0, x, idx_fix, w_warm, F.silu)
        assert bool(torch.isfinite(y_fix).all())
        assert len(disk_calls) == reads_after_warm, "Fixated layer (>=75% resident mass) must do 0 disk reads!"
        assert ov.fixation_layer_skips == 1
        assert ov.fixation_experts_skipped == 1

        # Case B (Equilibration): experts 0,6,7,5 have only 0.40 < 0.75 resident mass -> MUST wake A->B->C(0)->A and read disk!
        idx_eq = torch.tensor([[0, 6, 7, 5]], device=dev, dtype=torch.long)
        y_eq = c_imp.forward_layer_moe(0, x, idx_eq, w_warm, F.silu)
        assert bool(torch.isfinite(y_eq).all())
        assert len(disk_calls) > reads_after_warm, "Low-coverage layer must wake disk relay for equilibration!"
        assert ov.equilibration_layer_wakes >= 1
        st = ov.stats()
        assert st["relay"]["transitions"]["B->A"] == 0
        assert st["relay"]["standby_bytes"] == 0
    finally:
        c_imp.close()


