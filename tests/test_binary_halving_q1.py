"""Unit and CUDA tests for Binary Halving Q1 Tree (`Q16 -> 2xQ8 -> 4xQ4 -> 8xQ2 -> 16xQ1`)."""
import pytest
import torch
import torch.nn.functional as F

from src.binary_halving_q1 import (
    AirLLMLazyBinaryHalvingRunner,
    BinaryHalvingModelSlicer,
    BinaryHalvingQ1Linear,
    IncrementalCubeCarver,
    LazyMoEBinaryHalvingCache,
    carve_q1_cubes_incremental,
    halve_weight_to_q1_tree,
)


@pytest.mark.parametrize("law", ["residual_halving", "positional_halving"])
@pytest.mark.parametrize("root_bits", [4, 8, 16])
def test_binary_halving_tree_progression_and_monotonic_convergence_cpu(law, root_bits):
    torch.manual_seed(123)
    w = torch.randn(128, 256, dtype=torch.bfloat16) * 0.05
    # Add realistic channel outliers to verify Sinkhorn + FWHT robustness
    w[:, :4] *= 10.0

    tensors, meta = halve_weight_to_q1_tree(w, root_bits=root_bits, group_size=64, law=law, use_fwht=True)
    expected_levels = tuple(1 << d for d in range(meta.depth + 1))
    assert meta.level_node_counts == expected_levels
    assert len(meta.leaf_paths) == root_bits
    assert len(set(meta.leaf_paths)) == root_bits
    assert all(len(p) == meta.depth and set(p) <= {"L", "R"} for p in meta.leaf_paths)
    assert meta.leaf_paths[0] == "L" * meta.depth
    assert meta.leaf_paths[-1] == "R" * meta.depth

    mod = BinaryHalvingQ1Linear(256, 128, tensors, meta)
    cos_ladder = []
    for k in (1, 2, 4, root_bits):
        if k > root_bits:
            continue
        wk = mod.reconstruct_weight(cubes=k, dtype=torch.float32)
        c = F.cosine_similarity(wk.reshape(-1), w.float().reshape(-1), dim=0).item()
        cos_ladder.append((k, c))

    # Strict quality growth along the geometric halving levels
    for i in range(len(cos_ladder) - 1):
        assert cos_ladder[i + 1][1] >= cos_ladder[i][1] - 1e-4, f"{law} root={root_bits}: {cos_ladder}"
    if law == "positional_halving" and root_bits >= 8:
        assert cos_ladder[-1][1] > 0.9999, f"{law} root={root_bits}: {cos_ladder}"
    elif root_bits >= 8:
        assert cos_ladder[-1][1] > 0.995, f"{law} root={root_bits}: {cos_ladder}"
    else:
        assert cos_ladder[-1][1] > 0.985, f"{law} root={root_bits}: {cos_ladder}"

    # Verify step-wise CPU path matches direct reconstruction
    x = torch.randn(3, 256, dtype=torch.bfloat16)
    mod.active_cubes = min(4, root_bits)
    y_direct = mod(x)
    mod.record_geometry = True
    y_step = mod(x)
    mod.record_geometry = False
    torch.testing.assert_close(y_step.float(), y_direct.float(), rtol=2e-2, atol=2e-2)
    assert mod.calls == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("root_bits", [4, 8, 16])
def test_binary_halving_nvrtc_gemv_and_b1_mma_parity_cuda(root_bits):
    torch.manual_seed(456)
    dev = torch.device("cuda:0")
    w = torch.randn(192, 256, dtype=torch.bfloat16, device=dev) * 0.04
    bias = torch.randn(192, dtype=torch.bfloat16, device=dev) * 0.01
    tensors, meta = halve_weight_to_q1_tree(w, root_bits=root_bits, group_size=64, law="residual_halving", use_fwht=True)
    mod = BinaryHalvingQ1Linear(256, 192, tensors, meta, bias=bias)

    for m_batch in (1, 12):
        x = torch.randn(m_batch, 256, dtype=torch.bfloat16, device=dev)
        for p_use in (1, 2, 4, root_bits):
            mod.active_cubes = p_use
            y_nvrtc = mod(x)
            w_ref = mod.reconstruct_weight(cubes=p_use, dtype=torch.float32)
            y_ref = F.linear(x.float(), w_ref, bias.float())
            cos = F.cosine_similarity(y_nvrtc.float().reshape(-1), y_ref.reshape(-1), dim=0).item()
            assert cos > 0.999, f"CUDA parity drop root={root_bits} M={m_batch} p={p_use}: cos={cos}"

    # Check dynamic branch early-stop on CUDA
    mod.active_cubes = root_bits
    mod.branch_stop_tol = 0.05
    mod.reset_telemetry()
    y_es = mod(torch.randn(1, 256, dtype=torch.bfloat16, device=dev))
    assert torch.isfinite(y_es).all()
    assert 2 <= mod.last_used_cubes <= root_bits


def test_model_slicer_geometry_discovery_and_moe_budget_cpu():
    from transformers import AutoModelForCausalLM, GraniteMoeConfig

    torch.manual_seed(77)
    cfg = GraniteMoeConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    x_ids = torch.tensor([[1, 5, 9, 13]], dtype=torch.long)
    with torch.no_grad():
        ref_logits = model(input_ids=x_ids).logits.float()

    slicer = BinaryHalvingModelSlicer(model, root_bits=8, group_size=64, law="positional_halving", use_fwht=True)
    summary = slicer.slice_all(compute_diagnostics=True)
    assert summary["sliced_linear_modules"] == 8  # 2 layers x (q, k, v, o)
    assert summary["sliced_moe_blocks"] == 2
    assert summary["level_progression"] == [1, 2, 4, 8]
    # Compact leaf count: 8 linear * 8 + 2 layers * 2 banks * 4 experts * 8 = 192 total Q1 cubes (NOT millions!)
    assert summary["total_q1_leaf_cubes"] == 8 * 8 + 2 * 2 * 4 * 8

    with torch.no_grad():
        q8_logits = model(input_ids=x_ids).logits.float()
    cos_full = F.cosine_similarity(q8_logits.reshape(-1), ref_logits.reshape(-1), dim=0).item()
    assert cos_full > 0.998, f"Full 8-cube tree should nearly match BF16 oracle, got cos={cos_full}"

    geom = slicer.discover_inter_cube_geometry(lambda: model(input_ids=x_ids))
    assert len(geom["cube_mean_rel_contrib"]) == 8
    assert geom["cube_mean_rel_contrib"][0] > geom["cube_mean_rel_contrib"][-1]

    alloc = slicer.apply_geometry_budget(target_avg_cubes=3.0)
    assert alloc["actual_param_weighted_avg_cubes"] <= 3.05

    # Router-weight-adaptive MoE depth
    slicer.set_moe_router_schedule(((0.5, 4), (0.0, 2)))
    slicer.reset_telemetry()
    with torch.no_grad():
        out = model(input_ids=x_ids).logits
    assert torch.isfinite(out).all()
    total_exp_calls = sum(in_b.expert_calls for _, _, in_b, _ in slicer.moe_blocks)
    assert total_exp_calls > 0


def test_incremental_cube_carver_matches_method1_cpu():
    torch.manual_seed(314)
    w = torch.randn(128, 256, dtype=torch.bfloat16) * 0.05
    w[:, :4] *= 8.0

    t_m1, meta_m1 = halve_weight_to_q1_tree(w, root_bits=8, group_size=64, law="positional_halving", use_fwht=True)
    m1_lin = BinaryHalvingQ1Linear(256, 128, t_m1, meta_m1)

    carver = IncrementalCubeCarver(w, root_bits=8, group_size=64, use_fwht=True)
    rmses = []
    for step in range(8):
        info = carver.carve_next_cube()
        assert info["cube_index"] == step
        assert info["path"] == meta_m1.leaf_paths[step]
        rmses.append(info["residual_rel_rmse"])
    # Each carved Q1 cube strictly reduces residual RMSE (~halves every step)
    for a, b in zip(rmses, rmses[1:]):
        assert b < a

    for k in (1, 2, 4, 8):
        t_m2, meta_m2 = carve_q1_cubes_incremental(w, k, root_bits=8, group_size=64, use_fwht=True)
        m2_lin = BinaryHalvingQ1Linear(256, 128, t_m2, meta_m2)
        w1 = m1_lin.reconstruct_weight(cubes=k, dtype=torch.float32)
        w2 = m2_lin.reconstruct_weight(cubes=k, dtype=torch.float32)
        cos = F.cosine_similarity(w1.reshape(-1), w2.reshape(-1), dim=0).item()
        assert cos > 0.9999, f"Method 2 vs Method 1 mismatch at k={k}: cos={cos}"


@pytest.mark.parametrize("root_bits", [4, 8, 16])
@pytest.mark.parametrize("use_fwht", [False, True])
def test_positional_stored_grid_matches_all_prefixes(root_bits, use_fwht):
    torch.manual_seed(520)
    w = (torch.randn(32, 128) * .17 + .013).to(torch.bfloat16)
    w[:, :3] *= 7
    bulk, bulk_meta = halve_weight_to_q1_tree(w, root_bits=root_bits,
        use_fwht=use_fwht, compute_diagnostics=False)
    lazy, lazy_meta = carve_q1_cubes_incremental(w, root_bits,
        root_bits=root_bits, use_fwht=use_fwht)
    bulk_mod = BinaryHalvingQ1Linear(128, 32, bulk, bulk_meta)
    lazy_mod = BinaryHalvingQ1Linear(128, 32, lazy, lazy_meta)
    torch.testing.assert_close(lazy['qscale'], bulk['qscale'], rtol=0, atol=0)
    for prefix in range(1, root_bits + 1):
        torch.testing.assert_close(lazy_mod.reconstruct_weight(cubes=prefix),
            bulk_mod.reconstruct_weight(cubes=prefix), rtol=0, atol=1e-6)
    reconstructed = lazy_mod.reconstruct_weight()
    if not use_fwht:
        half_step = lazy['qscale'][0].float() * 2.0 ** (1-root_bits)
        errors = (reconstructed - w.float()).abs().reshape(32, 2, 64)
        assert bool((errors <= half_step.unsqueeze(-1)*1.02 + 1e-6).all())
    if root_bits == 16:
        relative_error = (reconstructed-w.float()).norm() / w.float().norm()
        assert relative_error < 5e-5


def test_positional_grid_constant_and_zero_groups():
    w = torch.zeros(3, 128, dtype=torch.bfloat16)
    w[1] = .125
    w[2, :64] = -.75
    for method in ('bulk', 'incremental'):
        if method == 'bulk':
            tree, meta = halve_weight_to_q1_tree(w, root_bits=16,
                use_fwht=False, compute_diagnostics=False)
        else:
            tree, meta = carve_q1_cubes_incremental(w, 16, root_bits=16, use_fwht=False)
        reconstructed = BinaryHalvingQ1Linear(128, 3, tree, meta).reconstruct_weight()
        assert torch.isfinite(reconstructed).all()
        assert torch.equal(reconstructed, w.float())


def test_runtime_carving_skips_unused_residual_statistics():
    torch.manual_seed(37)
    w = torch.randn(8, 64, dtype=torch.bfloat16)
    diagnostic = IncrementalCubeCarver(w, root_bits=16)
    runtime = IncrementalCubeCarver(w, root_bits=16, compute_diagnostics=False)
    assert runtime._ref_norm is None
    assert diagnostic.carve_next_cube()['residual_rel_rmse'] is not None
    assert runtime.carve_next_cube()['residual_rel_rmse'] is None
    diagnostic_tree, _ = diagnostic.carve_until(16)
    runtime_tree, _ = runtime.carve_until(16)
    packaged_tree, _ = carve_q1_cubes_incremental(w, 16, root_bits=16)
    for name in diagnostic_tree:
        assert torch.equal(diagnostic_tree[name], runtime_tree[name])
        assert torch.equal(diagnostic_tree[name], packaged_tree[name])


@pytest.mark.parametrize("use_cuda", [False, True])
def test_method2_airllm_lazy_runner_parity_and_abc_invariants(use_cuda):
    if use_cuda and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from transformers import AutoModelForCausalLM, GraniteMoeConfig

    torch.manual_seed(99)
    dev = torch.device("cuda:0" if use_cuda else "cpu")
    cfg = GraniteMoeConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=8,
        num_experts_per_tok=2,
    )
    base_model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    # Snapshot raw weights on CPU to emulate disk reader
    raw_layers = {}
    raw_experts = {}
    for l_idx, layer in enumerate(base_model.model.layers):
        raw_layers[l_idx] = {k: v.detach().cpu().clone() for k, v in layer.named_parameters()}
        raw_experts[l_idx] = (
            layer.block_sparse_moe.experts.gate_up_proj.detach().cpu().clone(),
            layer.block_sparse_moe.experts.down_proj.detach().cpu().clone(),
        )

    # Method 1 reference (sliced upfront at k=4)
    m1_model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    m1_model.load_state_dict(base_model.state_dict())
    m1_model = m1_model.to(dev)
    slicer = BinaryHalvingModelSlicer(m1_model, root_bits=8, group_size=64, law="positional_halving", use_fwht=True)
    slicer.slice_all(compute_diagnostics=False)
    slicer.set_uniform_cubes(4, moe_cubes=4)
    x_ids = torch.tensor([[1, 5, 9, 13]], dtype=torch.long, device=dev)
    with torch.no_grad():
        m1_logits = m1_model(input_ids=x_ids).logits[0, -1].float().cpu()
    del m1_model

    # Method 2: AirLLM lazy layer-by-layer streaming with host_cache_bytes=0 (pure disk) and cached mode
    for cache_bytes in (0, 64 * 1024**2):
        m2_model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
        m2_model.load_state_dict(base_model.state_dict())
        m2_model = m2_model.to(dev)

        def expert_reader(li, eid):
            gu_all, dn_all = raw_experts[li]
            return gu_all[eid], dn_all[eid]

        moe_cache = LazyMoEBinaryHalvingCache(
            expert_reader,
            num_layers=3,
            num_experts=8,
            device=dev,
            store_cubes=4,
            root_bits=8,
            group_size=64,
            max_cache_bytes=cache_bytes,
            cache_on_gpu=use_cuda,
        )
        runner = AirLLMLazyBinaryHalvingRunner(
            m2_model,
            lambda li: raw_layers[li],
            device=dev,
            cubes_per_proj=4,
            root_bits=8,
            group_size=64,
            host_cache_bytes=cache_bytes,
            moe_cache=moe_cache,
        ).install()
        try:
            # All decoder layers must be meta (0 B) when idle
            for layer in m2_model.model.layers:
                assert all(p.is_meta for p in layer.parameters())

            with torch.no_grad():
                m2_logits_1 = m2_model(input_ids=x_ids).logits[0, -1].float().cpu()
                m2_logits_2 = m2_model(input_ids=x_ids).logits[0, -1].float().cpu()

            cos = F.cosine_similarity(m2_logits_1, m1_logits, dim=0).item()
            assert cos > 0.9995, f"Method 2 vs Method 1 logit parity drop (cuda={use_cuda}, cache={cache_bytes}): {cos}"
            st = runner.stats()
            assert st["transitions"]["B->A"] == 0
            assert st["transitions"]["A->B"] == st["transitions"]["B->C"] == st["transitions"]["C->A"] == 6
            assert st["standby_bytes"] == 0
            if cache_bytes == 0:
                assert st["disk_layer_reads"] == 6 and st["host_cache_hits"] == 0
            else:
                assert st["disk_layer_reads"] == 3 and st["host_cache_hits"] == 3
                assert st["moe_cache"]["hits"] > 0
            # Verify all decoder layers returned to meta after forward
            for layer in m2_model.model.layers:
                assert all(p.is_meta for p in layer.parameters())
        finally:
            runner.close()
            del m2_model



def test_lazy_moe_two_tier_gpu_to_host_lru_prevents_recarve():
    """Evicting an expert from L1 cache must demote it to L2 Host LRU so it is never recarved from disk."""
    torch.manual_seed(17)
    H, I = 128, 64
    num_experts = 6
    bank_gu = torch.randn(num_experts, 2 * I, H, dtype=torch.bfloat16) * 0.05
    bank_dn = torch.randn(num_experts, H, I, dtype=torch.bfloat16) * 0.05
    disk_calls = {"count": 0}

    def reader(layer_idx: int, eid: int):
        disk_calls["count"] += 1
        return bank_gu[eid].clone(), bank_dn[eid].clone()

    # Probe exact byte size of one carved Q1 expert so L1 holds exactly 2 experts, L2 holds all 6.
    probe = LazyMoEBinaryHalvingCache(
        reader, num_layers=1, num_experts=num_experts, device=torch.device("cpu"),
        store_cubes=4, root_bits=8, group_size=64, max_cache_bytes=10 * 1024**2, cache_on_gpu=False,
    )
    probe._get_or_carve(0, 0)
    one_expert_bytes = probe._cached_bytes
    probe.close()
    disk_calls["count"] = 0

    cache = LazyMoEBinaryHalvingCache(
        reader,
        num_layers=1,
        num_experts=num_experts,
        device=torch.device("cpu"),
        store_cubes=4,
        root_bits=8,
        group_size=64,
        max_cache_bytes=2 * one_expert_bytes,
        host_cache_bytes=6 * one_expert_bytes,
        cache_on_gpu=False,
    )
    try:
        # Pass 1: touch experts 0..4 (exceeds L1 capacity=2 -> evictions demote to L2 Host LRU)
        for eid in range(5):
            cache._get_or_carve(0, eid)
        assert disk_calls["count"] == 5
        assert cache.misses == 5
        assert cache.evictions == 3
        assert len(cache._host_lru) == 3

        # Pass 2: re-access evicted experts 0, 1, 2 -> MUST hit L2 Host LRU with ZERO new disk reads!
        for eid in (0, 1, 2):
            cache._get_or_carve(0, eid)
        st = cache.stats()
        assert disk_calls["count"] == 5, f"Expected 0 disk recarves, got {disk_calls['count']}"
        assert st["misses"] == 5
        assert st["host_hits"] == 3
        assert st["unique_experts_carved"] == 5
    finally:
        cache.close()


@torch.no_grad()
def test_shared_qscale_storage_and_bpw_parity():
    """Shared qscale[0] halves scale memory (~3.5 bpw at k=3, ~4.5 bpw at k=4) and matches reference on CPU/CUDA."""
    torch.manual_seed(42)
    N, K = 512, 1024
    w = (torch.randn(N, K, dtype=torch.float32) * 0.05).to(torch.bfloat16)

    tensors, meta = halve_weight_to_q1_tree(
        w, root_bits=8, group_size=64, use_fwht=True, law="positional_halving"
    )
    assert tuple(tensors["qscale"].shape) == (1, N, K // 64)

    lin = BinaryHalvingQ1Linear(K, N, tensors, meta)
    bpw3 = lin.effective_bits_per_weight(3)
    bpw4 = lin.effective_bits_per_weight(4)
    assert 3.50 <= bpw3 <= 3.56, f"Unexpected k=3 bpw: {bpw3}"
    assert 4.50 <= bpw4 <= 4.56, f"Unexpected k=4 bpw: {bpw4}"

    carved_t, carved_m = carve_q1_cubes_incremental(w, num_cubes=4, root_bits=8, group_size=64, use_fwht=True)
    assert tuple(carved_t["qscale"].shape) == (1, N, K // 64)
    assert torch.equal(carved_t["packed_lo"], tensors["packed_lo"][:4])
    assert torch.equal(carved_t["qscale"], tensors["qscale"])

    # Verify numerical accuracy against explicit dequantization on CPU and CUDA (GEMV M=1 + b1.mma M=12)
    w_rec4 = lin.reconstruct_weight(cubes=4, dtype=torch.float32)
    for dev_str in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        dev = torch.device(dev_str)
        mod = BinaryHalvingQ1Linear(K, N, {k: v.to(dev) for k, v in carved_t.items()}, carved_m)
        mod.active_cubes = 4
        for m_rows in (1, 12):
            x = torch.randn(m_rows, K, device=dev, dtype=torch.bfloat16)
            y_got = mod(x).float()
            y_ref = F.linear(x.float(), w_rec4.to(dev))
            cos = float(F.cosine_similarity(y_got.flatten(), y_ref.flatten(), dim=0).item())
            assert cos > 0.9985, f"shared qscale mismatch on {dev_str} M={m_rows}: cos={cos:.6f}"


@torch.no_grad()
def test_runner_resident_head_plus_streamed_tail_parity():
    """Resident head stays in VRAM/RAM (Method 1 speed) while tail streams A->B->C(0)->A with exact logit parity."""
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM, LlamaConfig

    torch.manual_seed(99)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    src_model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    sd = {k: v.detach().cpu().clone() for k, v in src_model.state_dict().items()}
    del src_model

    def make_reader():
        def reader(l_idx: int):
            pref = f"model.layers.{l_idx}."
            return {k[len(pref):]: v.clone() for k, v in sd.items() if k.startswith(pref)}
        return reader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_ids = torch.tensor([[1, 5, 11, 23]], dtype=torch.long, device=device)

    def run_with_resident(res_layers: int):
        with init_empty_weights():
            m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
        from accelerate.utils import set_module_tensor_to_device
        for k, v in sd.items():
            if not k.startswith("model.layers."):
                set_module_tensor_to_device(m, k, device, value=v.to(device), dtype=v.dtype, clear_cache=False)
        runner = AirLLMLazyBinaryHalvingRunner(
            m, make_reader(), device=device, cubes_per_proj=4, root_bits=8,
            resident_layers=res_layers, host_cache_bytes=64 * 1024**2,
        ).install()
        try:
            out1 = m(input_ids=x_ids).logits[0, -1].float().cpu()
            out2 = m(input_ids=x_ids).logits[0, -1].float().cpu()
            st = runner.stats()
            meta_states = [all(p.is_meta for p in lyr.parameters()) for lyr in m.model.layers]
            return out1, out2, st, meta_states
        finally:
            runner.close()
            del m

    ref1, ref2, st_stream, meta_stream = run_with_resident(0)
    hyb1, hyb2, st_hyb, meta_hyb = run_with_resident(2)

    assert F.cosine_similarity(ref1, hyb1, dim=0).item() > 0.9999
    assert F.cosine_similarity(ref2, hyb2, dim=0).item() > 0.9999
    assert meta_stream == [True, True, True]
    assert meta_hyb == [False, False, True]
    assert st_hyb["resident_layers"] == 2
    assert st_hyb["resident_ready"] == 2
    assert st_hyb["resident_layer_hits"] == 2
    assert st_hyb["transitions"]["B->A"] == 0
    assert st_hyb["transitions"]["A->B"] == st_hyb["transitions"]["B->C"] == st_hyb["transitions"]["C->A"] == 2
    assert st_hyb["standby_bytes"] == 0


@torch.no_grad()
def test_unified_engine_q1_cubes_cpu_only_and_cli(tmp_path, monkeypatch):
    """Unified q1_cubes mode works end-to-end via CLI, planner, Dense and MoE on CPU-only (zero CUDA)."""
    from quintessentia_cli.cli import build_parser
    from quintessentia_cli.unified_runtime import QuintessentiaUnifiedEngine, GiB, plan_for_checkpoint
    from test_cpu_independence import forbid_cuda
    from test_ideal_q1_and_modes import _make_tiny_llama, _make_tiny_olmoe

    parser = build_parser()
    args = parser.parse_args(["ask", "--model", "1", "--weight-format", "q1_cubes", "--q1-planes", "3", "Hi"])
    assert args.weight_format == "q1_cubes"
    assert args.q1_planes == 3

    llama_dir = tmp_path / "tiny_llama"
    llama_dir.mkdir()
    _make_tiny_llama(llama_dir, hidden_size=128, intermediate_size=256, num_layers=2, vocab_size=384)

    moe_dir = tmp_path / "tiny_olmoe"
    moe_dir.mkdir()
    _make_tiny_olmoe(moe_dir, hidden_size=128, intermediate_size=128, num_layers=2, num_experts=6)

    # Planner monotonicity across cube depths 1 -> 2 -> 4 -> 8
    sizes = []
    for p in (1, 2, 4, 8):
        prof, pl, _ = plan_for_checkpoint(
            llama_dir, mode="normal", total_bytes=8 * GiB, free_bytes=6 * GiB,
            weight_format="q1_cubes", q1_planes=p, max_context_tokens=128, prefill_chunk_tokens=16,
        )
        assert prof["weight_format"] == "q1_cubes"
        assert prof["backend"] == "unified-q1-cubes"
        sizes.append(pl.weights_bytes)
    assert sizes[0] < sizes[1] < sizes[2] < sizes[3]

    forbid_cuda(monkeypatch)
    for model_path, expect_moe in ((llama_dir, False), (moe_dir, True)):
        with QuintessentiaUnifiedEngine(
            model_path,
            device="cpu",
            weight_format="q1_cubes",
            q1_planes=4,
            speculation="off",
            verbose=False,
            max_context_tokens=128,
            prefill_chunk_tokens=8,
        ) as u:
            done = list(
                u.stream_chat(
                    [{"role": "user", "content": "Hello"}],
                    max_new_tokens=2,
                    enable_thinking=False,
                    stop_on_eos=False,
                    use_session_cache=False,
                )
            )[-1]
            assert done["type"] == "done"
            assert done["metrics"]["generated_tokens"] == 2
            snap = u.status_snapshot()
            assert "q1_cubes" in snap
            assert snap["q1_cubes"]["transitions"]["B->A"] == 0
            assert snap["q1_cubes"]["standby_bytes"] == 0
            if expect_moe:
                assert snap["q1_cubes"]["moe_cache"] is not None
                assert snap["q1_cubes"]["moe_cache"]["unique_experts_carved"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_unified_engine_q1_cubes_gpu_dense_and_moe_cuda(tmp_path):
    """Unified q1_cubes mode on GPU runs full-resident Method-1 speed for small models and L1/L2 MoE."""
    from quintessentia_cli.unified_runtime import QuintessentiaUnifiedEngine
    from test_ideal_q1_and_modes import _make_tiny_llama, _make_tiny_olmoe

    llama_dir = tmp_path / "tiny_llama_gpu"
    llama_dir.mkdir()
    _make_tiny_llama(llama_dir, hidden_size=128, intermediate_size=256, num_layers=3, vocab_size=384)

    moe_dir = tmp_path / "tiny_olmoe_gpu"
    moe_dir.mkdir()
    _make_tiny_olmoe(moe_dir, hidden_size=128, intermediate_size=128, num_layers=2, num_experts=8)

    with QuintessentiaUnifiedEngine(
        llama_dir,
        weight_format="q1_cubes",
        q1_planes=4,
        speculation="off",
        verbose=False,
        max_context_tokens=128,
        prefill_chunk_tokens=8,
    ) as u:
        done = list(
            u.stream_chat(
                [{"role": "user", "content": "1 2 3"}],
                max_new_tokens=3,
                enable_thinking=False,
                stop_on_eos=False,
                use_session_cache=False,
            )
        )[-1]
        assert done["type"] == "done"
        assert done["metrics"]["generated_tokens"] == 3
        st = u.status_snapshot()["q1_cubes"]
        assert st["resident_layers"] == 3
        assert st["resident_ready"] == 3
        assert st["resident_layer_hits"] >= 6
        assert st["rotations"] == 0

    with QuintessentiaUnifiedEngine(
        moe_dir,
        weight_format="q1_cubes",
        q1_planes=4,
        speculation="off",
        verbose=False,
        max_context_tokens=128,
        prefill_chunk_tokens=8,
    ) as u:
        done = list(
            u.stream_chat(
                [{"role": "user", "content": "1 2 3"}],
                max_new_tokens=3,
                enable_thinking=False,
                stop_on_eos=False,
                use_session_cache=False,
            )
        )[-1]
        assert done["type"] == "done"
        assert done["metrics"]["generated_tokens"] == 3
        st = u.status_snapshot()["q1_cubes"]
        assert st["moe_cache"] is not None
        assert st["moe_cache"]["unique_experts_carved"] > 0
        assert st["moe_cache"]["cache_on_gpu"] is True
        assert u.expert_cache_stats()["unique_experts_carved"] > 0
