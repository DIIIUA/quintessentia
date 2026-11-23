"""Tests for Spatial-Bitwise Q1 Cube Engine and NVRTC Kernels (CPU & CUDA)."""
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.q1_cube_engine import (
    Q1RoleCycleArena,
    Q1StreamedLMHead,
    Q1StreamedLinear,
    build_gate_radar_depths,
    dequantize_scale_laws,
    pack_for_radar_kernel,
    pack_q1_cubes,
    q1_cubes_from_packed,
    q1_linear_streamed,
    q1_radar_linear,
    radar_pack_from_packed,
    streamed_checkpoint_mlp_pass,
    unpack_q1_cubes,
)
from src.quantized_projection import PackedLinear, pack_weight, unpack_weight


SMOLLM2_CKPT = Path(
    r"D:\Air-Colibri Storage\release_validation_1_2_0\downloaded\SmolLM2-135M-Instruct\model.safetensors"
)


@pytest.mark.parametrize("bits", [4, 6])
def test_q1_positional_equivalence_to_colibri_q4_q6_exact(bits):
    torch.manual_seed(123)
    w = torch.randn(192, 576, dtype=torch.bfloat16) * 0.05
    ref_unpacked = unpack_weight(pack_weight(w, bits=bits, group_size=64), group_size=64, dtype=torch.bfloat16)
    q1_mat = pack_q1_cubes(w, planes=bits, r_tile=64, c_tile=128, group_size=64)
    q1_unpacked = unpack_q1_cubes(q1_mat, dtype=torch.bfloat16)
    assert torch.equal(q1_unpacked, ref_unpacked)


def test_q1_8_planes_matches_affine_q8_exact():
    torch.manual_seed(7)
    w = torch.randn(128, 256, dtype=torch.bfloat16) * 0.05
    w_grp = w.float().reshape(128, 4, 64)
    zero = w_grp.amin(-1).to(torch.bfloat16)
    scale = ((w_grp.amax(-1).double() - zero.double()) / 255.0).clamp_min_(1e-8).to(torch.bfloat16)
    q = ((w_grp.double() - zero.double().unsqueeze(-1)) / scale.double().unsqueeze(-1)).round_().clamp_(0, 255)
    ref_q8 = (q.float() * scale.float().unsqueeze(-1) + zero.float().unsqueeze(-1)).reshape(128, 256).to(torch.bfloat16)

    q1_mat = pack_q1_cubes(w, planes=8, r_tile=64, c_tile=128, group_size=64)
    q1_unpacked = unpack_q1_cubes(q1_mat, dtype=torch.bfloat16)
    assert torch.equal(q1_unpacked, ref_q8)


def test_midpoint_corrected_prefix_equals_sign_geom_q1():
    torch.manual_seed(11)
    w = torch.randn(64, 128, dtype=torch.float32) * 0.05
    K, P, GS = 8, 2, 64
    q1 = pack_q1_cubes(w, planes=K, r_tile=64, c_tile=128, group_size=GS)
    w_mid = unpack_q1_cubes(q1, dtype=torch.float32, active_planes=P, midpoint_correct=True)

    meta = q1.spatial_tiles[(0, 0)]
    z = meta.qzero.float().unsqueeze(-1)
    s = meta.qscale.float().unsqueeze(-1)
    midpoint = z + s * ((1 << K) - 1) / 2.0
    bit_shifts = torch.arange(32, dtype=torch.int64)
    w_sign_geom = midpoint.expand(64, 2, 64).clone()
    for p_idx, cube in enumerate(q1.cubes[:P]):
        b_p = (((cube.bits_i32.to(torch.int64) & 0xFFFFFFFF).unsqueeze(-1) >> bit_shifts) & 1).reshape(64, 2, 64).float()
        s_p = 2.0 * b_p - 1.0
        w_sign_geom += (s * float(1 << (K - 2 - p_idx))) * s_p
    assert torch.allclose(w_mid, w_sign_geom.reshape(64, 128), atol=1e-6, rtol=0.0)


def test_scale_laws_comparison_and_2d_linear_accumulation_cpu():
    torch.manual_seed(42)
    w = torch.randn(192, 256, dtype=torch.bfloat16) * 0.05
    laws = dequantize_scale_laws(w, planes=4, group_size=64)
    cos_pos = F.cosine_similarity(laws["positional"].flatten(), w.float().flatten(), dim=0).item()
    cos_geom = F.cosine_similarity(laws["sign_geom"].flatten(), w.float().flatten(), dim=0).item()
    cos_arith = F.cosine_similarity(laws["sign_arith"].flatten(), w.float().flatten(), dim=0).item()
    cos_res = F.cosine_similarity(laws["sign_residual"].flatten(), w.float().flatten(), dim=0).item()
    assert cos_pos > 0.99 and cos_geom > 0.99
    assert cos_pos >= cos_arith and cos_pos >= cos_res

    q1 = pack_q1_cubes(w, planes=4, r_tile=64, c_tile=128, group_size=64)
    slot_budget = max(c.nbytes + q1.spatial_tiles[(c.tile_r, c.tile_c)].nbytes for c in q1.cubes) + 512
    assert slot_budget < q1.total_bytes
    x = torch.randn(2, 256, dtype=torch.bfloat16)
    y_str, st = q1_linear_streamed(x, q1, slot_budget, out_dtype=torch.float32, use_nvrtc=False)
    y_ref = F.linear(x.float(), unpack_q1_cubes(q1, dtype=torch.float32))
    assert (y_str - y_ref).abs().max().item() < 1e-5
    assert st["transitions"]["B->A"] == 0
    assert st["transitions"]["A->B"] == st["rotations"]


def _load_mlp_weights_for_test(device):
    if SMOLLM2_CKPT.is_file():
        from safetensors import safe_open

        with safe_open(str(SMOLLM2_CKPT), framework="pt", device="cpu") as f:
            w_g = f.get_tensor("model.layers.0.mlp.gate_proj.weight").to(device)
            w_u = f.get_tensor("model.layers.0.mlp.up_proj.weight").to(device)
            w_d = f.get_tensor("model.layers.0.mlp.down_proj.weight").to(device)
            lm_shape = f.get_slice("model.embed_tokens.weight").get_shape()
            lm_head_bytes = int(lm_shape[0]) * int(lm_shape[1]) * 2
            return w_g, w_u, w_d, lm_head_bytes
    torch.manual_seed(55)
    w_g = (torch.randn(256, 128, dtype=torch.bfloat16) * 0.04).to(device)
    w_u = (torch.randn(256, 128, dtype=torch.bfloat16) * 0.04).to(device)
    w_d = (torch.randn(128, 256, dtype=torch.bfloat16) * 0.04).to(device)
    return w_g, w_u, w_d, 256 * 512 * 2


def test_checkpoint_mlp_streaming_sub_matrix_sub_lm_head_cpu():
    dev = torch.device("cpu")
    w_g, w_u, w_d, lm_head_bytes = _load_mlp_weights_for_test(dev)
    torch.manual_seed(99)
    x = (torch.randn(1, w_g.shape[1], dtype=torch.bfloat16, device=dev) * 0.5)
    out, info = streamed_checkpoint_mlp_pass(
        x, w_g, w_u, w_d, planes=4, r_tile=64, c_tile=64, group_size=64, lm_head_bytes=lm_head_bytes
    )
    assert info["slot_budget_bytes"] < info["min_matrix_bytes"]
    assert info["slot_budget_bytes"] < lm_head_bytes

    y_ref = F.linear((F.silu(F.linear(x.float(), w_g.float())) * F.linear(x.float(), w_u.float())), w_d.float())
    cos = F.cosine_similarity(out.float().flatten(), y_ref.flatten(), dim=0).item()
    assert cos > 0.985
    assert info["down_stats"]["transitions"]["B->A"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_checkpoint_mlp_and_1launch_radar_nvrtc_cuda():
    dev = torch.device("cuda:0")
    w_g, w_u, w_d, lm_head_bytes = _load_mlp_weights_for_test(dev)
    torch.manual_seed(99)
    x = (torch.randn(1, w_g.shape[1], dtype=torch.bfloat16, device=dev) * 0.5).contiguous()

    out, info = streamed_checkpoint_mlp_pass(
        x, w_g, w_u, w_d, planes=4, r_tile=64, c_tile=64, group_size=64, lm_head_bytes=lm_head_bytes
    )
    assert info["slot_budget_bytes"] < info["min_matrix_bytes"]
    assert info["slot_budget_bytes"] < lm_head_bytes

    y_ref = F.linear((F.silu(F.linear(x.float(), w_g.float())) * F.linear(x.float(), w_u.float())), w_d.float())
    assert F.cosine_similarity(out.float().flatten(), y_ref.flatten(), dim=0).item() > 0.985

    # 1-launch NVRTC adaptive radar check
    c_tile = w_g.shape[1]
    gp = pack_for_radar_kernel(w_g, k_max=6, r_tile=16, c_tile=c_tile, group_size=64, device=dev)
    up = pack_for_radar_kernel(w_u, k_max=6, r_tile=16, c_tile=c_tile, group_size=64, device=dev)
    depths = build_gate_radar_depths(x, gp, quiet_depth=4, hot_depth=6)
    assert (depths == 0).any() and (depths == 4).any() and (depths == 6).any()
    y_u_cuda = q1_radar_linear(x, up, depths)
    y_u_cpu = q1_radar_linear(
        x.cpu(),
        Q1RadarPack_cpu := pack_for_radar_kernel(w_u.cpu(), k_max=6, r_tile=16, c_tile=c_tile, group_size=64, device="cpu"),
        depths.cpu(),
    )
    assert F.cosine_similarity(y_u_cuda.cpu().float().flatten(), y_u_cpu.float().flatten(), dim=0).item() > 0.9999


@pytest.mark.parametrize("bits", [4, 6])
@pytest.mark.parametrize("group_size", [16, 32, 64])
def test_q1_cubes_and_radar_from_packed_zero_roundtrip_error(bits, group_size):
    torch.manual_seed(303 + bits + group_size)
    w = torch.randn(64, 192, dtype=torch.bfloat16) * 0.05
    packed = pack_weight(w, bits=bits, group_size=group_size)
    ref = unpack_weight(packed, group_size=group_size, dtype=torch.bfloat16)

    q1_mat = q1_cubes_from_packed(packed, bits=bits, group_size=group_size, r_tile=32, c_tile=64)
    rec = unpack_q1_cubes(q1_mat, dtype=torch.bfloat16)
    assert (rec.float() - ref.float()).abs().max().item() == 0.0

    # Unbiased truncated prefix P=2 matches z_eff(2) identity
    w_p2 = unpack_q1_cubes(q1_mat, dtype=torch.float32, active_planes=2, midpoint_correct=True)
    x = torch.randn(2, 192, dtype=torch.bfloat16)
    y_stream, st = q1_linear_streamed(x, q1_mat, slot_budget_bytes=q1_mat.total_bytes - 1, active_planes=2, out_dtype=torch.float32, use_nvrtc=False)
    assert (y_stream - F.linear(x.float(), w_p2)).abs().max().item() < 1e-4
    assert st["transitions"]["B->A"] == 0

    radar = radar_pack_from_packed(packed, bits=bits, group_size=group_size, r_tile=16, c_tile=64)
    depths = torch.full((4, 3), bits, dtype=torch.int32)
    y_radar = q1_radar_linear(x, radar, depths)
    assert (y_radar.float() - F.linear(x.float(), ref.float())).abs().max().item() < 0.02


def test_q1_streamed_linear_and_lm_head_candidates_cpu():
    torch.manual_seed(404)
    w = torch.randn(96, 128, dtype=torch.bfloat16) * 0.04
    bias = torch.randn(96, dtype=torch.bfloat16) * 0.01
    packed = pack_weight(w, bits=4, group_size=64)
    pl = PackedLinear(128, 96, bits=4, bias=True, group_size=64).to_empty(device="cpu")
    with torch.no_grad():
        for k_name, v_tensor in packed.items():
            getattr(pl, k_name).copy_(v_tensor)
        pl.bias.copy_(bias)

    shared_arena = Q1RoleCycleArena(slot_budget_bytes=16384, device=torch.device("cpu"))
    q1_lin = Q1StreamedLinear.from_packed_linear(pl, slot_budget_bytes=16384, arena=shared_arena, r_tile=32, c_tile=128)
    x = torch.randn(2, 128, dtype=torch.bfloat16)
    y_lin = q1_lin(x)
    y_ref = pl(x)
    assert (y_lin.float() - y_ref.float()).abs().max().item() < 0.02
    assert q1_lin.stats()["batch_count"] == 1
    assert q1_lin.stats()["arena"]["transitions"]["B->A"] == 0

    # LMHead backed by row reader & candidate subset projection without touching all row tiles
    vocab_w = torch.randn(512, 128, dtype=torch.bfloat16) * 0.03
    lm_bias = torch.randn(512, dtype=torch.bfloat16) * 0.01
    reads = []

    def reader(r0, r1):
        reads.append((r0, r1))
        return vocab_w[r0:r1]

    head = Q1StreamedLMHead(
        128, 512, read_rows=reader, bits=4, group_size=64, r_tile=64, c_tile=128,
        bias=lm_bias, slot_budget_bytes=32768, arena=shared_arena,
    )
    assert all(r1 - r0 <= 64 for r0, r1 in reads)
    head_reads_before = len(reads)
    y_full = head(x)
    cands = torch.tensor([[5, 19, 450]], dtype=torch.long)
    y_cands = head.project_candidates(x, cands)
    assert len(reads) == head_reads_before  # zero extra source reads during project_candidates
    assert y_cands.dtype == torch.float32
    assert y_cands.shape == (2, 3)
    assert (y_cands.to(torch.bfloat16) == y_full[:, [5, 19, 450]]).all()
    assert (y_cands - y_full.float()[:, [5, 19, 450]]).abs().max().item() < 2e-3
    h_stats = head.stats()
    assert h_stats["row_tiles_skipped"] >= 6
    assert h_stats["arena"]["transitions"]["B->A"] == 0
    assert h_stats["arena"]["area_used_bytes"][h_stats["arena"]["standby_area"]] == 0


@pytest.mark.parametrize("bits", [4, 6])
def test_packed_linear_active_planes_cpu_matches_q1_unbiased(bits):
    torch.manual_seed(505 + bits)
    w = torch.randn(64, 128, dtype=torch.bfloat16) * 0.05
    bias = torch.randn(64, dtype=torch.bfloat16) * 0.01
    packed = pack_weight(w, bits=bits, group_size=16)
    pl = PackedLinear(128, 64, bits=bits, bias=True, group_size=16).to_empty(device="cpu")
    with torch.no_grad():
        for k_name, v_tensor in packed.items():
            getattr(pl, k_name).copy_(v_tensor)
        pl.bias.copy_(bias)

    x = torch.randn(2, 128, dtype=torch.bfloat16)
    y_exact = pl(x)

    q1_mat = q1_cubes_from_packed(packed, bits=bits, group_size=16, r_tile=32, c_tile=64)
    w_p2 = unpack_q1_cubes(q1_mat, dtype=torch.bfloat16, active_planes=2, midpoint_correct=True)
    y_p2_ref = F.linear(x, w_p2, bias)

    pl.active_planes = 2
    y_p2 = pl(x)
    assert (y_p2.float() - y_p2_ref.float()).abs().max().item() == 0.0

    pl.active_planes = None
    assert torch.equal(pl(x), y_exact)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("K,GS", [(256, 16), (16384, 64), (16384, 16)])
def test_nvrtc_all_5_kernels_gs16_and_large_k_smem_cuda(K, GS):
    from src.colibri_q1_cube_nvrtc import (
        launch_q1_adaptive_radar_gemv,
        launch_q1_multiplane_gemv,
        launch_q1_packed_trunc_gemv,
        launch_q1_plane_gemv,
    )

    dev = torch.device("cuda:0")
    torch.manual_seed(606 + K + GS)
    N = 64
    w = (torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.02).contiguous()
    x = (torch.randn(1, K, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()

    q1 = pack_q1_cubes(w, planes=4, r_tile=N, c_tile=K, group_size=GS, device=dev)
    meta = q1.spatial_tiles[(0, 0)]
    cube_msb = q1.cubes[0]
    planes_i32 = torch.stack([c.bits_i32 for c in q1.cubes[:2]], dim=0).contiguous()

    # 1 & 2: single-plane __ffs and __popc
    for use_popc in (False, True):
        y_acc = torch.zeros((1, N), dtype=torch.float32, device=dev)
        launch_q1_plane_gemv(
            x, cube_msb.bits_i32, meta.qscale, meta.qzero, y_acc,
            r0=0, c0=0, c_actual=K, group_size=GS, alpha=cube_msb.weight_pow2,
            add_zero=True, use_popc=use_popc,
        )
        assert torch.isfinite(y_acc).all()

    # 3 & 4: multiplane __ffs and __popc
    w_p2 = unpack_q1_cubes(q1, dtype=torch.bfloat16, active_planes=2, midpoint_correct=True)
    y_ref_p2 = F.linear(x.float(), w_p2.float())
    for use_popc, tol_cos in ((False, 0.9999), (True, 0.999)):
        y_acc = torch.zeros((1, N), dtype=torch.float32, device=dev)
        launch_q1_multiplane_gemv(
            x, planes_i32, meta.qscale, meta.qzero, y_acc,
            r0=0, c0=0, c_actual=K, group_size=GS, total_planes=4,
            add_zero=True, use_popc=use_popc, midpoint_correct=True,
        )
        assert F.cosine_similarity(y_acc.flatten(), y_ref_p2.flatten(), dim=0).item() > tol_cos

    # 5: adaptive radar (d=2 active + d=0 skip)
    c_tile = 256 if K >= 1024 else 64
    radar = pack_for_radar_kernel(w, k_max=4, r_tile=16, c_tile=c_tile, group_size=GS, device=dev)
    n_rt, n_ct = radar.planes_u32.shape[0], radar.planes_u32.shape[1]
    depths = torch.full((n_rt, n_ct), 2, dtype=torch.int32, device=dev)
    depths[:, ::2] = 0
    y_cuda = torch.empty((1, N), dtype=torch.bfloat16, device=dev)
    launch_q1_adaptive_radar_gemv(
        x, radar.planes_u32, radar.qscale, radar.qzero, depths, y_cuda,
        r_tile=16, c_tile=c_tile, group_size=GS, k_max=4,
    )
    y_cpu = q1_radar_linear(x.cpu(), Q1RadarPack_cpu := pack_for_radar_kernel(w.cpu(), k_max=4, r_tile=16, c_tile=c_tile, group_size=GS, device="cpu"), depths.cpu())
    assert F.cosine_similarity(y_cuda.cpu().float().flatten(), y_cpu.float().flatten(), dim=0).item() > 0.9999

    # 6: direct packed trunc GEMV (Q4 P=2, and Q6 P=2 reading packed_hi ONLY with packed_lo=None)
    p4 = pack_weight(w, bits=4, group_size=GS, device=dev)
    q1_4 = q1_cubes_from_packed(p4, bits=4, group_size=GS, r_tile=N, c_tile=K, device=dev)
    y4_ref = F.linear(x.float(), unpack_q1_cubes(q1_4, dtype=torch.float32, active_planes=2, midpoint_correct=True))
    y4_out = torch.empty((1, N), dtype=torch.float32, device=dev)
    launch_q1_packed_trunc_gemv(
        x, p4["packed_lo"], None, p4["qscale"], p4["qzero"], y4_out,
        group_size=GS, bits=4, p_active=2, midpoint_correct=True,
    )
    assert F.cosine_similarity(y4_out.flatten(), y4_ref.flatten(), dim=0).item() > 0.9999

    p6 = pack_weight(w, bits=6, group_size=GS, device=dev)
    q1_6 = q1_cubes_from_packed(p6, bits=6, group_size=GS, r_tile=N, c_tile=K, device=dev)
    y6_ref = F.linear(x.float(), unpack_q1_cubes(q1_6, dtype=torch.float32, active_planes=2, midpoint_correct=True))
    y6_out = torch.empty((1, N), dtype=torch.float32, device=dev)
    # Pass packed_lo=None explicitly to prove Q6 P<=2 never accesses packed_lo in GDDR6
    launch_q1_packed_trunc_gemv(
        x, None, p6["packed_hi"], p6["qscale"], p6["qzero"], y6_out,
        group_size=GS, bits=6, p_active=2, midpoint_correct=True,
    )
    assert F.cosine_similarity(y6_out.flatten(), y6_ref.flatten(), dim=0).item() > 0.9999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_q1_streamed_modules_and_packed_active_planes_cuda():
    dev = torch.device("cuda:0")
    torch.manual_seed(707)
    w = torch.randn(128, 256, dtype=torch.bfloat16, device=dev) * 0.04
    packed = pack_weight(w, bits=6, group_size=16)
    pl = PackedLinear(256, 128, bits=6, bias=False, group_size=16).to_empty(device=dev)
    with torch.no_grad():
        for k_name, v_tensor in packed.items():
            getattr(pl, k_name).copy_(v_tensor)
    x = torch.randn(1, 256, dtype=torch.bfloat16, device=dev)

    pl.active_planes = 2
    y_cuda_p2 = pl(x)
    q1 = q1_cubes_from_packed({k: v.cpu() for k, v in packed.items()}, bits=6, group_size=16)
    y_cpu_p2 = F.linear(x.cpu().float(), unpack_q1_cubes(q1, dtype=torch.float32, active_planes=2, midpoint_correct=True))
    assert F.cosine_similarity(y_cuda_p2.cpu().float().flatten(), y_cpu_p2.flatten(), dim=0).item() > 0.9999
    pl.active_planes = None

    # CPU-packed LMHead streamed to CUDA via 2-stream pin_memory + transfer_stream + 1 launch/batch
    packed_cpu = {k: v.cpu() for k, v in packed.items()}
    lm = Q1StreamedLMHead(256, 128, packed_dict=packed_cpu, bits=6, group_size=16, r_tile=32, c_tile=128, slot_budget_bytes=20480)
    y_full = lm(x)
    y_cands = lm.project_candidates(x, torch.tensor([1, 65, 120], device=dev))
    assert F.cosine_similarity(y_cands.cpu().flatten(), y_full.cpu().float()[:, [1, 65, 120]].flatten(), dim=0).item() > 0.9999
    st = lm.stats()
    assert st["arena"]["nvrtc_batch_launches"] == st["batches_executed"]
    assert st["arena"]["async_h2d_transfers"] >= 1
    assert st["arena"]["transitions"]["B->A"] == 0
    assert st["arena"]["area_used_bytes"][st["arena"]["standby_area"]] == 0

    # Q1StreamedLinear with host-packed Q6 on CUDA via 2-stream arena
    lin_arena = Q1RoleCycleArena(slot_budget_bytes=24576, device=dev)
    qlin = Q1StreamedLinear(256, 128, packed_dict=packed_cpu, bits=6, group_size=16, r_tile=64, c_tile=128, arena=lin_arena)
    y_qlin = qlin(x)
    qlin.active_planes = 2
    y_qlin_p2 = qlin(x)
    assert F.cosine_similarity(y_qlin_p2.cpu().float().flatten(), y_cpu_p2.flatten(), dim=0).item() > 0.9999
    qlin.active_planes = None
    st_l = qlin.stats()
    assert st_l["arena"]["nvrtc_batch_launches"] == st_l["batches_executed"]
    assert st_l["arena"]["transitions"]["B->A"] == 0


@torch.no_grad()
def test_q1_speculation_preserves_tokens_and_restores_planes(tmp_path, monkeypatch):
    from test_cpu_independence import tiny_checkpoint
    from aircolibri_cli.unified_runtime import AirColibriUnifiedEngine
    from src.general_speculation import GeneralSpeculation, Q1SkeletonDrafter

    tiny_checkpoint(tmp_path)
    prompt = [3, 7, 4, 9]
    with AirColibriUnifiedEngine(tmp_path, device="cpu", weight_format="q4", speculation="off", verbose=False) as base:
        events_base = list(base._stream_token_ids(prompt, 8, False, 0, None))
        expected = [t for ev in events_base if ev["kind"] == "tokens" for t in ev["new_ids"]]

    with AirColibriUnifiedEngine(tmp_path, device="cpu", weight_format="q4", speculation="lookup", verbose=False) as u:
        u.speculator.close()
        u.speculator = GeneralSpeculation(u, mode="q1")
        events = list(u._stream_token_ids(prompt, 8, False, 0, None))
        actual = [t for ev in events if ev["kind"] == "tokens" for t in ev["new_ids"]]
        assert actual == expected
        for mod in u.engine.model.modules():
            if isinstance(mod, (PackedLinear, Q1StreamedLinear)):
                assert mod.active_planes is None
        stats = u.speculator.stats()
        assert stats["source"] == "q1_skeleton"
        assert stats["q1_draft_supported"] is True

        # Exception during Q1SkeletonDrafter.draft must still restore active_planes=None & cache state
        last_state = events[-1]
        past_before = last_state["past"]
        seq_before = past_before.get_seq_length()
        drafter = Q1SkeletonDrafter(u, active_planes=2)
        orig_step = u.engine._forward_step

        def fail_step(*args, **kwargs):
            raise RuntimeError("injected draft failure")

        monkeypatch.setattr(u.engine, "_forward_step", fail_step)
        with pytest.raises(RuntimeError, match="injected draft failure"):
            drafter.draft(actual[-1], past_before, seq_before, 2)
        monkeypatch.setattr(u.engine, "_forward_step", orig_step)
        for mod in u.engine.model.modules():
            if isinstance(mod, (PackedLinear, Q1StreamedLinear)):
                assert mod.active_planes is None
        assert past_before.get_seq_length() == seq_before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_q1_speculation_cuda_q1_and_auto_preserves_tokens(tmp_path):
    from test_cpu_independence import tiny_checkpoint
    from aircolibri_cli.unified_runtime import AirColibriUnifiedEngine
    from src.general_speculation import GeneralSpeculation

    tiny_checkpoint(tmp_path)
    prompt = [3, 7, 4, 9]
    with AirColibriUnifiedEngine(tmp_path, device="auto", cpu_first=False, weight_format="q4", speculation="off", verbose=False) as base:
        events_base = list(base._stream_token_ids(prompt, 8, False, 0, None))
        expected = [t for ev in events_base if ev["kind"] == "tokens" for t in ev["new_ids"]]

    for mode in ("q1", "auto"):
        with AirColibriUnifiedEngine(tmp_path, device="auto", cpu_first=False, weight_format="q4", speculation="lookup", verbose=False) as u:
            u.speculator.close()
            u.speculator = GeneralSpeculation(u, mode=mode)
            assert u.engine.device.type == "cuda"
            events = list(u._stream_token_ids(prompt, 8, False, 0, None))
            actual = [t for ev in events if ev["kind"] == "tokens" for t in ev["new_ids"]]
            assert actual == expected
            for mod in u.engine.model.modules():
                if isinstance(mod, (PackedLinear, Q1StreamedLinear)):
                    assert mod.active_planes is None
            if mode == "q1":
                assert u.speculator.stats()["source"] == "q1_skeleton"
