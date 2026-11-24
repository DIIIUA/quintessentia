"""Wave 4 tests: Dynamic 3D Q1PuzzleOverlay (horiz / vert / 90-deg diag_corner contacts), Q2 depth fix & MoE Q1/Q2 LRU."""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from quintessentia_cli.cli import build_parser
from quintessentia_cli.unified_runtime import QuintessentiaUnifiedEngine, MiB
from src.quantized_projection import pack_weight, unpack_weight


def _make_tiny_llama(folder: Path, *, hidden_size: int = 128, intermediate_size: int = 256, num_layers: int = 3, vocab_size: int = 512):
    from transformers import AutoModelForCausalLM, LlamaConfig
    from transformers.convert_slow_tokenizer import bytes_to_unicode
    from test_model_source import _write_tiny_hf_tokenizer

    torch.manual_seed(40404)
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    model.save_pretrained(folder)
    _write_tiny_hf_tokenizer(folder, vocab_size=vocab_size)
    tok_path = folder / "tokenizer.json"
    tok = json.loads(tok_path.read_text(encoding="utf-8"))
    tok["model"]["vocab"] = {"<eos>": 0, **{c: i + 1 for i, c in enumerate(bytes_to_unicode().values())}}
    tok_path.write_text(json.dumps(tok), encoding="utf-8")
    del model


@torch.no_grad()
def test_cli_puzzle_flags_and_q2_30layer_norm_stability_cpu():
    parser = build_parser()
    args = parser.parse_args(
        ["ask", "--model", "1", "--puzzle-overlay", "--puzzle-slice-ratio", "0.25", "Hello"]
    )
    assert args.puzzle_overlay is True
    assert args.puzzle_slice_ratio == 0.25

    args_def = parser.parse_args(["ask", "--model", "1", "Hello"])
    assert args_def.puzzle_overlay is False
    assert args_def.puzzle_slice_ratio == 0.35

    with pytest.raises(ValueError, match="puzzle_slice_ratio"):
        QuintessentiaUnifiedEngine(device="cpu", puzzle_overlay=True, puzzle_slice_ratio=1.5)

    # Verify Wave 3/4 Q2 depth fix: 30-layer transformer residual + RMSNorm chain with 8x channel outliers preserves norm & cosine
    torch.manual_seed(2026)
    dim = 128
    x_ref = torch.randn(8, dim, dtype=torch.float32)
    x_q2 = x_ref.clone()
    for l_idx in range(30):
        w = torch.randn(dim, dim, dtype=torch.float32) * 0.03
        w[:, l_idx % dim] *= 8.0
        w_bf16 = w.to(torch.bfloat16)
        pk2 = pack_weight(w_bf16, bits=2, group_size=64, device="cpu")
        wq2 = unpack_weight(pk2, group_size=64, dtype=torch.float32)
        xn_ref = x_ref * torch.rsqrt(x_ref.pow(2).mean(-1, keepdim=True) + 1e-6)
        xn_q2 = x_q2 * torch.rsqrt(x_q2.pow(2).mean(-1, keepdim=True) + 1e-6)
        x_ref = x_ref + F.linear(xn_ref, w_bf16.float())
        x_q2 = x_q2 + F.linear(xn_q2, wq2)

    norm_ratio = float((x_q2.norm() / x_ref.norm().clamp_min(1e-8)).item())
    cos_29 = float(F.cosine_similarity(x_ref.flatten(), x_q2.flatten(), dim=0).item())
    assert 0.92 <= norm_ratio <= 1.08, f"Expected bounded norm ratio after 30 Q2 residual layers, got {norm_ratio:.4f}"
    assert cos_29 > 0.90, f"Expected high 30-layer residual Q2 cosine, got {cos_29:.4f}"


@torch.no_grad()
def test_q1_puzzle_overlay_contacts_uniqueness_and_ladder_cpu(tmp_path):
    from transformers import AutoModelForCausalLM
    from src.q1_puzzle_overlay import Q1PuzzleLinear, install_q1_puzzle_overlay

    _make_tiny_llama(tmp_path, hidden_size=128, intermediate_size=256, num_layers=2, vocab_size=320)
    ref_model = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16).eval()
    puz_model = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16).eval()

    ids_seq = torch.tensor([[2, 19, 44, 77, 105, 13, 88, 201]], dtype=torch.long)
    logits_ref = ref_model(ids_seq).logits.float()

    ctrl = install_q1_puzzle_overlay(
        puz_model,
        slice_ratio=0.0,
        r_tile=32,
        c_tile=64,
        group_size=64,
        slot_budget_bytes=256 * 1024,
        device="cpu",
    )
    assert any(isinstance(m, Q1PuzzleLinear) for m in puz_model.modules())

    try:
        cos_by_ratio = {}
        for r in (0.0, 0.30, 0.65, 1.0):
            ctrl.set_slice_ratio(r)
            ctrl.reset_stats()
            if r == 0.30:
                # Feed distinct single-token steps so each step assembles its own unique slice S_t
                logits_steps = []
                past = None
                for t in range(ids_seq.shape[1]):
                    out = puz_model(ids_seq[:, t : t + 1], past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    logits_steps.append(out.logits.float())
                lg = torch.cat(logits_steps, dim=1)
                st_30 = ctrl.stats()
            else:
                lg = puz_model(ids_seq).logits.float()
            cos_by_ratio[r] = float(F.cosine_similarity(logits_ref.flatten(), lg.flatten(), dim=0).item())

        # Cold cubes at ratio=0.0 are non-zero Ideal Q1 (P=1); ratio=1.0 recovers full main weights
        assert cos_by_ratio[1.0] > 0.999, f"Expected near-exact reconstruction at ratio=1.0, got {cos_by_ratio[1.0]:.6f}"
        assert cos_by_ratio[0.0] < cos_by_ratio[0.65] < cos_by_ratio[1.0]

        # Dynamic contact slice uniqueness & 3 physical contacts
        assert st_30["calls"] > 0
        assert st_30["unique_slice_signatures"] >= 2
        assert st_30["avg_slice_jaccard"] < 1.0
        cc = st_30["contact_counts"]
        assert cc["seed"] > 0
        assert (cc["horiz"] + cc["vert"] + cc["diag_corner"]) > 0
        assert cc["diag_corner"] > 0  # 90-deg output r == input c' & corner contacts active!
    finally:
        ctrl.close()


@torch.no_grad()
def test_q1_puzzle_overlay_cpu_only_engine_zero_cuda(tmp_path, monkeypatch):
    from test_cpu_independence import forbid_cuda

    _make_tiny_llama(tmp_path, hidden_size=128, intermediate_size=256, num_layers=2, vocab_size=320)
    forbid_cuda(monkeypatch)

    with QuintessentiaUnifiedEngine(
        tmp_path,
        device="cpu",
        weight_format="q6",
        puzzle_overlay=True,
        puzzle_slice_ratio=0.4,
        speculation="off",
        verbose=False,
        max_context_tokens=128,
        prefill_chunk_tokens=8,
    ) as u:
        done = list(
            u.stream_chat(
                [{"role": "user", "content": "Hello world"}],
                max_new_tokens=4,
                enable_thinking=False,
                stop_on_eos=False,
                use_session_cache=False,
            )
        )[-1]
        assert done["type"] == "done"
        assert done["metrics"]["generated_tokens"] == 4
        puz = done["metrics"].get("q1_puzzle_overlay")
        assert puz is not None
        assert puz["calls"] > 0
        assert abs(puz["slice_ratio"] - 0.4) < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_nvrtc_puzzle_kernel_abc_arena_and_unified_engine_cuda(tmp_path):
    from src.q1_cube_engine import Q1RoleCycleArena
    from src.q1_puzzle_overlay import Q1PuzzleLinear

    device = torch.device("cuda")
    torch.manual_seed(909)

    # 1. Direct Q1PuzzleLinear CUDA kernel #11 vs full weight at slice_ratio=1.0 + strict A->B->C(0)->A
    w_main = (torch.randn(192, 256, dtype=torch.float32) * 0.04).to(torch.bfloat16)
    bias = (torch.randn(192, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    arena = Q1RoleCycleArena(slot_budget_bytes=256 * 1024, device=device)
    puz_lin = Q1PuzzleLinear(
        256,
        192,
        weight=w_main,
        bias=bias,
        r_tile=64,
        c_tile=128,
        group_size=64,
        slice_ratio=1.0,
        arena=arena,
        device=device,
    )
    try:
        for m_rows in (1, 8):
            x = torch.randn(m_rows, 256, device=device, dtype=torch.bfloat16)
            y_puz = puz_lin(x).float()
            y_ref = F.linear(x.float(), w_main.to(device).float(), bias.to(device).float())
            cos = float(F.cosine_similarity(y_puz.flatten(), y_ref.flatten(), dim=0).item())
            assert cos > 0.9995, f"M={m_rows}: cos={cos:.6f}"
        ast = arena.stats()
        assert ast["transitions"]["B->A"] == 0
        assert ast["area_used_bytes"][ast["standby_area"]] == 0
    finally:
        del puz_lin, arena

    # 2. Full QuintessentiaUnifiedEngine on CUDA (GPU-first & Progressive takeover) with puzzle_overlay=True
    _make_tiny_llama(tmp_path, hidden_size=128, intermediate_size=256, num_layers=3, vocab_size=320)
    prompt = [{"role": "user", "content": "1 2 3 4 5"}]

    with QuintessentiaUnifiedEngine(
        tmp_path,
        cpu_first=False,
        weight_format="q8",
        puzzle_overlay=True,
        puzzle_slice_ratio=0.45,
        max_slot_mib=1.0,
        speculation="off",
        verbose=False,
        max_context_tokens=128,
        prefill_chunk_tokens=8,
    ) as u_gpu:
        done_g = list(u_gpu.stream_chat(prompt, max_new_tokens=5, enable_thinking=False, stop_on_eos=False))[-1]
        puz_g = done_g["metrics"]["q1_puzzle_overlay"]
        assert puz_g["calls"] > 0
        assert puz_g["unique_slice_signatures"] >= 2
        assert puz_g["arena"]["transitions"]["B->A"] == 0
        assert puz_g["arena"]["area_used_bytes"][puz_g["arena"]["standby_area"]] == 0

    # Tight-slot regression: max_slot_mib smaller than one Q8 layer triggers plan.sub_layer_cubes=True,
    # yet puzzle_overlay MUST install Q1PuzzleLinear (not Q1StreamedLinear) with calls > 0!
    with QuintessentiaUnifiedEngine(
        tmp_path,
        cpu_first=False,
        weight_format="q8",
        puzzle_overlay=True,
        puzzle_slice_ratio=0.40,
        max_slot_mib=0.06,
        speculation="off",
        verbose=False,
        max_context_tokens=128,
        prefill_chunk_tokens=8,
    ) as u_tight:
        assert u_tight.plan.sub_layer_cubes is True
        assert u_tight.engine.puzzle_controller is not None
        assert len(u_tight.engine.puzzle_controller.modules) > 0
        done_t = list(u_tight.stream_chat(prompt, max_new_tokens=4, enable_thinking=False, stop_on_eos=False))[-1]
        puz_t = done_t["metrics"]["q1_puzzle_overlay"]
        assert puz_t["calls"] > 0
        assert puz_t["unique_slice_signatures"] >= 2
        assert puz_t["arena"]["transitions"]["B->A"] == 0
        assert puz_t["arena"]["area_used_bytes"][puz_t["arena"]["standby_area"]] == 0

    with QuintessentiaUnifiedEngine(
        tmp_path,
        progressive_start=True,
        weight_format="q6",
        puzzle_overlay=True,
        puzzle_slice_ratio=0.35,
        speculation="off",
        verbose=False,
        max_context_tokens=128,
        prefill_chunk_tokens=8,
    ) as u_prog:
        assert u_prog.engine.progressive.backend_ready.wait(30)
        _ = list(u_prog.stream_chat(prompt, max_new_tokens=3, enable_thinking=False, stop_on_eos=False))
        assert u_prog.engine.progressive.bundle_ready.wait(30)
        done_p = list(u_prog.stream_chat(prompt, max_new_tokens=4, enable_thinking=False, stop_on_eos=False))[-1]
        assert u_prog.engine.progressive.complete is True
        assert done_p["metrics"]["q1_puzzle_overlay"]["calls"] > 0
