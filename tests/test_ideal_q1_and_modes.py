"""Wave 3 tests: Ideal Q1 (Sinkhorn + FWHT + Hessian LDLQ + Early-Stop + sm_86 b1.mma) & unified 6-mode ladder."""
from __future__ import annotations

import math
from pathlib import Path
import pytest
import torch

from quintessentia_cli.cli import build_parser
from quintessentia_cli.unified_runtime import (
    QuintessentiaUnifiedEngine,
    QuintessentiaSpecGate,
    GiB,
    MiB,
    plan_for_checkpoint,
)
from src.quantized_projection import PackedLinear, pack_weight, packed_specs, unpack_weight


def _make_tiny_llama(folder: Path, *, hidden_size: int = 128, intermediate_size: int = 256, num_layers: int = 3, vocab_size: int = 512):
    import json
    from transformers import AutoModelForCausalLM, LlamaConfig
    from transformers.convert_slow_tokenizer import bytes_to_unicode
    from test_model_source import _write_tiny_hf_tokenizer

    torch.manual_seed(31415)
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


def _make_tiny_olmoe(folder: Path, *, hidden_size: int = 128, intermediate_size: int = 128, num_layers: int = 3, num_experts: int = 8):
    import json
    from transformers import AutoModelForCausalLM, OlmoeConfig
    from transformers.convert_slow_tokenizer import bytes_to_unicode
    from test_model_source import _write_tiny_hf_tokenizer

    torch.manual_seed(27182)
    cfg = OlmoeConfig(
        vocab_size=260,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=num_experts,
        num_experts_per_tok=2,
        max_position_embeddings=128,
    )
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    model.save_pretrained(folder)
    _write_tiny_hf_tokenizer(folder, vocab_size=260)
    tok_path = folder / "tokenizer.json"
    tok = json.loads(tok_path.read_text(encoding="utf-8"))
    tok["model"]["vocab"] = {"<eos>": 0, **{c: i + 1 for i, c in enumerate(bytes_to_unicode().values())}}
    tok_path.write_text(json.dumps(tok), encoding="utf-8")
    del model


def test_cli_6modes_and_checkpoint_vram_ladder_cpu(tmp_path):
    parser = build_parser()
    for fmt in ("auto", "q1", "q2", "q4", "q6", "q8", "bf16"):
        args = parser.parse_args(
            [
                "ask",
                "--model",
                "1",
                "--weight-format",
                fmt,
                "--q1-planes",
                "3",
                "--no-q1-early-stop",
                "Hello",
            ]
        )
        assert args.weight_format == fmt
        assert args.q1_planes == 3
        assert args.q1_early_stop is False

    args_def = parser.parse_args(["ask", "--model", "1", "Hello"])
    # An omitted prefix is resolved by the selected representation: legacy
    # q1 uses four, while q1_cubes retains its complete tree before surveying.
    assert args_def.q1_planes is None
    assert args_def.q1_early_stop is True

    _make_tiny_llama(tmp_path, hidden_size=256, intermediate_size=512, num_layers=2, vocab_size=1024)
    plans = {}
    for fmt in ("q1", "q2", "q4", "q6", "q8", "bf16"):
        _, pl, _ = plan_for_checkpoint(
            tmp_path,
            mode="normal",
            total_bytes=8 * GiB,
            free_bytes=6 * GiB,
            weight_format=fmt,
            q1_planes=1 if fmt == "q1" else 4,
            max_context_tokens=256,
            prefill_chunk_tokens=32,
        )
        plans[fmt] = pl.weights_bytes

    assert plans["q1"] < plans["q2"] < plans["q4"] < plans["q6"] < plans["q8"] < plans["bf16"]

    q1_by_planes = []
    for p in (1, 2, 4, 8):
        _, pl_p, _ = plan_for_checkpoint(
            tmp_path,
            mode="normal",
            total_bytes=8 * GiB,
            free_bytes=6 * GiB,
            weight_format="q1",
            q1_planes=p,
            max_context_tokens=256,
            prefill_chunk_tokens=32,
        )
        q1_by_planes.append(pl_p.weights_bytes)
    assert q1_by_planes[0] < q1_by_planes[1] < q1_by_planes[2] < q1_by_planes[3]


@torch.no_grad()
def test_ideal_q1_math_beats_naive_q1_and_preserves_q4_q6_exactness_cpu():
    from src.ideal_q1_engine import fwht_block, sinkhorn_equilibrate

    torch.manual_seed(101)
    # 1. Orthonormal FWHT involution across GS in {16, 32, 64}
    x = torch.randn(7, 128, dtype=torch.float32)
    for gs in (16, 32, 64):
        y = fwht_block(x, gs)
        x_rec = fwht_block(y, gs)
        assert torch.allclose(x, x_rec, atol=1e-5)
        assert torch.allclose(x.norm(), y.norm(), rtol=1e-5)

    # 2. Sinkhorn equilibration exact reconstruction up to bf16 scale rounding
    w = torch.randn(128, 256, dtype=torch.float32) * 0.03
    # Inject severe channel & row outliers typical of transformer activations/weights
    w[:, 7] *= 25.0
    w[:, 113] *= 18.0
    w[19, :] *= 14.0
    r_sc, c_sc, w_eq = sinkhorn_equilibrate(w, iters=3)
    w_back = w_eq * (r_sc.float()[:, None] * c_sc.float()[None, :])
    assert torch.allclose(w, w_back, rtol=1e-4, atol=1e-5)

    w_bf16 = w.to(torch.bfloat16)
    x_probe = torch.randn(16, 256, dtype=torch.bfloat16)
    y_ref = torch.nn.functional.linear(x_probe.float(), w_bf16.float())

    # Naive 1-bit per-group sign quantization (without Sinkhorn + FWHT + LDLQ)
    wg = w_bf16.float().reshape(128, -1, 64)
    naive_alpha = wg.abs().mean(dim=-1, keepdim=True)
    w_naive_q1 = (torch.where(wg >= 0, 1.0, -1.0) * naive_alpha).reshape(128, 256)
    y_naive = torch.nn.functional.linear(x_probe.float(), w_naive_q1)
    cos_naive = float(torch.nn.functional.cosine_similarity(y_ref.flatten(), y_naive.flatten(), dim=0).item())

    # Ideal Q1 ladder P=1, 2, 4
    cos_ladder = []
    for p in (1, 2, 4):
        pk = pack_weight(w_bf16, bits=1, group_size=64, device="cpu", q1_planes=p)
        w_ideal = unpack_weight(pk, group_size=64, dtype=torch.float32)
        y_ideal = torch.nn.functional.linear(x_probe.float(), w_ideal)
        cos_p = float(torch.nn.functional.cosine_similarity(y_ref.flatten(), y_ideal.flatten(), dim=0).item())
        cos_ladder.append(cos_p)

    # Ideal Q1 (P=1) dramatically outperforms naive sign Q1 on outlier-heavy weights, and P=1->2->4 climbs monotonically
    assert cos_ladder[0] > cos_naive + 0.10, f"Ideal Q1 P=1 ({cos_ladder[0]:.4f}) should beat naive ({cos_naive:.4f})"
    assert cos_ladder[0] < cos_ladder[1] < cos_ladder[2]
    assert cos_ladder[2] > 0.99

    # Ideal Q2 & Quintessentia Q8
    pk_q2 = pack_weight(w_bf16, bits=2, group_size=64, device="cpu")
    w_q2 = unpack_weight(pk_q2, group_size=64, dtype=torch.float32)
    cos_q2 = float(
        torch.nn.functional.cosine_similarity(
            y_ref.flatten(), torch.nn.functional.linear(x_probe.float(), w_q2).flatten(), dim=0
        ).item()
    )
    assert cos_q2 > 0.94

    pk_q8 = pack_weight(w_bf16, bits=8, group_size=64, device="cpu")
    w_q8 = unpack_weight(pk_q8, group_size=64, dtype=torch.float32)
    cos_q8 = float(
        torch.nn.functional.cosine_similarity(
            y_ref.flatten(), torch.nn.functional.linear(x_probe.float(), w_q8).flatten(), dim=0
        ).item()
    )
    assert cos_q8 > 0.9999


@torch.no_grad()
def test_all_6_modes_cpu_only_end_to_end_zero_cuda(tmp_path, monkeypatch):
    from test_cpu_independence import forbid_cuda

    _make_tiny_llama(tmp_path, hidden_size=128, intermediate_size=256, num_layers=2, vocab_size=384)
    forbid_cuda(monkeypatch)

    for fmt in ("q1", "q2", "q4", "q6", "q8", "bf16"):
        with QuintessentiaUnifiedEngine(
            tmp_path,
            device="cpu",
            weight_format=fmt,
            q1_planes=3,
            q1_early_stop=True,
            thought_anchor=(fmt == "q1"),
            speculation="off",
            verbose=False,
            max_context_tokens=128,
            prefill_chunk_tokens=8,
        ) as u:
            done = list(
                u.stream_chat(
                    [{"role": "user", "content": "Hello"}],
                    max_new_tokens=3,
                    enable_thinking=False,
                    stop_on_eos=False,
                    use_session_cache=False,
                )
            )[-1]
            assert done["type"] == "done"
            assert done["metrics"]["generated_tokens"] == 3
            assert done["metrics"]["survivor_probability_path"]["updates"] >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_nvrtc_ideal_q1_b1_mma_q2_q8_and_multistratum_cuda():
    from src.ideal_q1_engine import IdealQ1LMHead, MultiStratumLinear

    torch.manual_seed(2026)
    device = torch.device("cuda")

    # 1. PackedLinear CUDA vs CPU reference across bits in {1, 2, 8} and M in {1, 3, 12 (b1.mma!)}
    for gs in (16, 32, 64):
        w = (torch.randn(96, 256, dtype=torch.float32) * 0.05).to(torch.bfloat16)
        bias = (torch.randn(96, dtype=torch.float32) * 0.01).to(torch.bfloat16)
        for bits in (1, 2, 8):
            mod_gpu = PackedLinear(256, 96, bits, bias=True, group_size=gs, q1_planes=4)
            pk = pack_weight(w, bits=bits, group_size=gs, device="cpu", q1_planes=4)
            for k, v in pk.items():
                mod_gpu._buffers[k] = v.to(device)
            mod_gpu.bias = torch.nn.Parameter(bias.to(device), requires_grad=False)

            w_deq = unpack_weight(pk, group_size=gs, dtype=torch.float32).to(device)
            b_f32 = bias.to(device=device, dtype=torch.float32)

            for m_rows in (1, 3, 12):
                x = torch.randn(m_rows, 256, device=device, dtype=torch.bfloat16)
                y_gpu = mod_gpu(x).float()
                y_ref = torch.nn.functional.linear(x.float(), w_deq, b_f32)
                cos = float(torch.nn.functional.cosine_similarity(y_gpu.flatten(), y_ref.flatten(), dim=0).item())
                assert cos > (0.998 if bits == 1 and m_rows >= 8 else 0.9999), (
                    f"bits={bits}, gs={gs}, M={m_rows}: cos={cos:.6f}"
                )

            # Truncated active_planes on bits=1 and bits=8
            if bits in (1, 8):
                mod_gpu.active_planes = 2
                x1 = torch.randn(2, 256, device=device, dtype=torch.bfloat16)
                y_trunc = mod_gpu(x1)
                assert torch.isfinite(y_trunc).all()
                mod_gpu.active_planes = None

    # 2. IdealQ1LMHead dynamic Plane-0 Early-Stop & project_candidates
    vocab_w = (torch.randn(2048, 256, dtype=torch.float32) * 0.04).to(torch.bfloat16)
    head_full = IdealQ1LMHead(
        256, 2048, read_rows=lambda s, e: vocab_w[s:e], planes=4, group_size=64, early_stop=False, device=device
    )
    head_es_exact = IdealQ1LMHead(
        256,
        2048,
        packed_dict=head_full.packed(),
        planes=4,
        group_size=64,
        early_stop=True,
        early_stop_z=None,  # Strict Cauchy-Schwarz: 100% mathematically guaranteed argmax
        min_keep_rows=128,
        device=device,
    )
    head_es_subg = IdealQ1LMHead(
        256,
        2048,
        packed_dict=head_full.packed(),
        planes=4,
        group_size=64,
        early_stop=True,
        early_stop_z=6.0,
        min_keep_rows=128,
        device=device,
    )
    # Realistic transformer hidden vector aligned with a semantic target token + noise
    x_tok = (vocab_w[42].to(device=device, dtype=torch.float32) * 20.0 + torch.randn(256, device=device) * 0.25).reshape(1, 1, 256).to(torch.bfloat16)
    logits_full = head_full(x_tok)
    logits_exact = head_es_exact(x_tok)
    logits_subg = head_es_subg(x_tok)
    assert int(logits_exact.argmax().item()) == int(logits_full.argmax().item())
    assert int(logits_subg.argmax().item()) == int(logits_full.argmax().item())
    st = head_es_subg.stats()
    assert st["vocab_rows_pruned_pct"] > 50.0
    assert st["effective_bits"] < 4.0

    cands = torch.tensor([0, 7, 42, 511, 2047], device=device, dtype=torch.long)
    cand_scores = head_es_subg.project_candidates(x_tok.reshape(1, 256), cands)
    assert cand_scores.shape[-1] == 5
    assert torch.allclose(cand_scores.reshape(-1).float(), logits_full.reshape(-1)[cands].float(), atol=5e-2, rtol=2e-2)

    # 3. MultiStratumLinear (2xQ2 and 2xQ4) over Q1RoleCycleArena A->B->C(0)->A
    w_ms = (torch.randn(128, 256, dtype=torch.float32) * 0.05).to(torch.bfloat16)
    x_ms = torch.randn(4, 256, device=device, dtype=torch.bfloat16)
    y_ms_ref = torch.nn.functional.linear(x_ms.float(), w_ms.to(device).float())
    for base_bits in (2, 4):
        ms_1 = MultiStratumLinear(256, 128, w_ms.to(device), stratum_bits=base_bits, strata=1, group_size=64)
        ms_lin = MultiStratumLinear(256, 128, w_ms.to(device), stratum_bits=base_bits, strata=2, group_size=64)
        cos_1 = float(torch.nn.functional.cosine_similarity(ms_1(x_ms).float().flatten(), y_ms_ref.flatten(), dim=0).item())
        y_ms = ms_lin(x_ms).float()
        cos_ms = float(torch.nn.functional.cosine_similarity(y_ms.flatten(), y_ms_ref.flatten(), dim=0).item())
        assert cos_ms > cos_1
        assert cos_ms > (0.985 if base_bits == 2 else 0.999)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_unified_engine_all_6_modes_progressive_moe_and_q1_speculation_cuda(tmp_path):
    lm_dir = tmp_path / "llama"
    lm_dir.mkdir()
    _make_tiny_llama(lm_dir, hidden_size=128, intermediate_size=256, num_layers=3, vocab_size=512)
    prompt = [{"role": "user", "content": "1 2 3 1 2 3"}]

    # 1. All 6 weight_formats on GPU-first + Q1 LM head telemetry + progressive Q1 takeover
    q1_ref_answer = None
    for fmt in ("q1", "q2", "q4", "q6", "q8", "bf16"):
        with QuintessentiaUnifiedEngine(
            lm_dir,
            cpu_first=False,
            weight_format=fmt,
            q1_planes=4,
            q1_early_stop=True,
            speculation="off",
            verbose=False,
            max_context_tokens=128,
            prefill_chunk_tokens=8,
        ) as u:
            done = list(u.stream_chat(prompt, max_new_tokens=5, enable_thinking=False, stop_on_eos=False))[-1]
            assert done["metrics"]["generated_tokens"] == 5
            if fmt == "q1":
                q1_ref_answer = done["answer_text"]
                q1_lh = done["metrics"].get("q1_lm_head")
                assert q1_lh is not None
                assert q1_lh["q1_planes"] == 4
                assert q1_lh["early_stop_enabled"] is True
                assert "effective_bits" in q1_lh and "vocab_rows_pruned_pct" in q1_lh

    # 2. Lossless Q1 self-speculation on weight_format="q1" (planes=4 -> draft active_planes=2 -> full verify planes=4)
    with QuintessentiaUnifiedEngine(
        lm_dir,
        cpu_first=False,
        weight_format="q1",
        q1_planes=4,
        q1_early_stop=True,
        speculation="q1",
        verbose=False,
        max_context_tokens=128,
        prefill_chunk_tokens=8,
    ) as u_spec:
        orig_pick = QuintessentiaSpecGate.pick_depth
        QuintessentiaSpecGate.pick_depth = lambda self_g, src, probs: min(len(probs), self_g.max_drafts)
        try:
            done_spec = list(u_spec.stream_chat(prompt, max_new_tokens=5, enable_thinking=False, stop_on_eos=False))[-1]
        finally:
            QuintessentiaSpecGate.pick_depth = orig_pick
        assert done_spec["answer_text"] == q1_ref_answer
        assert done_spec["metrics"]["spec_gate"]["q1_verifies"] > 0

    # 3. Progressive CPU->GPU takeover with weight_format="q1" adopts IdealQ1LMHead
    with QuintessentiaUnifiedEngine(
        lm_dir,
        progressive_start=True,
        weight_format="q1",
        q1_planes=3,
        q1_early_stop=True,
        speculation="off",
        verbose=False,
        max_context_tokens=128,
        prefill_chunk_tokens=8,
    ) as u_prog:
        assert u_prog.engine.progressive.backend_ready.wait(30)
        # First turn offers CPU layer payloads to background GPU worker
        _ = list(u_prog.stream_chat(prompt, max_new_tokens=3, enable_thinking=False, stop_on_eos=False))
        assert u_prog.engine.progressive.bundle_ready.wait(30)
        done_p2 = list(u_prog.stream_chat(prompt, max_new_tokens=4, enable_thinking=False, stop_on_eos=False))[-1]
        assert u_prog.engine.progressive.complete is True
        assert done_p2["metrics"]["q1_lm_head"]["q1_planes"] == 3

    # 4. MoE across q1, q2, q8
    moe_dir = tmp_path / "moe"
    moe_dir.mkdir()
    _make_tiny_olmoe(moe_dir, hidden_size=128, intermediate_size=128, num_layers=3, num_experts=8)
    for fmt in ("q1", "q2", "q8"):
        with QuintessentiaUnifiedEngine(
            moe_dir,
            cpu_first=False,
            weight_format=fmt,
            q1_planes=3,
            enable_mtp=False,
            speculation="off",
            verbose=False,
            max_context_tokens=128,
            prefill_chunk_tokens=8,
        ) as u_moe:
            done_m = list(u_moe.stream_chat(prompt, max_new_tokens=4, enable_thinking=False, stop_on_eos=False))[-1]
            assert done_m["metrics"]["generated_tokens"] == 4

    # 5. Progressive MoE CPU -> GPU takeover with weight_format="q1", q1_planes=4 (exercises _adopt_all + adopt_cpu_packed)
    with QuintessentiaUnifiedEngine(
        moe_dir,
        progressive_start=True,
        weight_format="q1",
        q1_planes=4,
        q1_early_stop=True,
        enable_mtp=False,
        speculation="off",
        verbose=False,
        max_context_tokens=128,
        prefill_chunk_tokens=8,
    ) as u_moe_prog:
        p_moe = u_moe_prog.engine.progressive
        assert p_moe.backend_ready.wait(30)
        # Force one CPU step first so QuantizedExpertCache populates CPU seeds at q1_planes=4 and offers layers
        p_moe.bundle_ready.clear()
        _, past_m = u_moe_prog.engine._forward_step(torch.tensor([[1, 2, 3]]), None)
        assert p_moe.bundle_ready.wait(30)
        done_mp = list(u_moe_prog.stream_chat(prompt, max_new_tokens=4, enable_thinking=False, stop_on_eos=False))[-1]
        assert p_moe.complete is True
        assert done_mp["metrics"]["generated_tokens"] == 4
        assert getattr(u_moe_prog.engine.model_source._expert_loader, "q1_planes", None) == 4
