"""Regression tests for Quintessentia 2.1.0 universal model & runtime fixes."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def test_effective_eos_ids_collects_chat_template_eot_without_breaking_harmony():
    from quintessentia_cli.unified_runtime import _resolve_effective_eos_ids

    class _FakeTok:
        def __init__(self, eos_id, tmpl, vocab):
            self.eos_token_id = eos_id
            self.chat_template = tmpl
            self.unk_token_id = 0
            self._v = vocab

        def convert_tokens_to_ids(self, tok):
            return self._v.get(tok, self.unk_token_id)

    # Phi-4-mini: eos is <|endoftext|>=199999, template ends turns with <|end|>=200020
    phi_tok = _FakeTok(199999, "{% for m in messages %}<|user|>{{m}}<|end|>{% endfor %}", {"<|end|>": 200020})
    assert _resolve_effective_eos_ids(phi_tok) == {199999, 200020}

    # Gemma-2: eos is <eos>=1, template ends turns with <end_of_turn>=107
    gemma_tok = _FakeTok(1, "<start_of_turn>user\n{{m}}<end_of_turn>\n", {"<end_of_turn>": 107})
    assert _resolve_effective_eos_ids(gemma_tok) == {1, 107}

    # Command-R7B: eos is <|END_OF_TURN_TOKEN|>=255001, template uses <|END_RESPONSE|>=255022
    cmdr_tok = _FakeTok(
        255001,
        "<|START_RESPONSE|>{{m}}<|END_RESPONSE|><|END_OF_TURN_TOKEN|>",
        {"<|END_RESPONSE|>": 255022, "<|END_OF_TURN_TOKEN|>": 255001},
    )
    assert _resolve_effective_eos_ids(cmdr_tok) == {255001, 255022}

    # GPT-OSS Harmony: <|end|> separates analysis/final channels and MUST NOT stop generation!
    harmony_tok = _FakeTok(
        200002,
        "<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...<|return|>",
        {"<|end|>": 200007, "<|return|>": 200002},
    )
    eos_h = _resolve_effective_eos_ids(harmony_tok)
    assert 200002 in eos_h
    assert 200007 not in eos_h


def test_ggml_type_i2_s_size_and_dequantization():
    from quintessentia_cli.gguf_reader import GGML_TYPE_I2_S, _tensor_bytes, ggml_type_name
    from quintessentia_cli.model_source import dequantize_gguf_tensor

    assert ggml_type_name(GGML_TYPE_I2_S) == "I2_S"
    nbytes, numel = _tensor_bytes([128, 2], GGML_TYPE_I2_S)
    assert numel == 256
    assert nbytes == 64 + 32

    # Build a 128-element interleaved I2_S block with known ternary values and scale = 1.5
    # Byte j packs elements [j, j+32, j+64, j+96] at shifts [6, 4, 2, 0] with encoding (val + 1) in {0,1,2}.
    trits = torch.zeros(128, dtype=torch.int8)
    trits[0] = 1     # +1 -> code 2 at shift 6 of byte 0
    trits[32] = -1   # -1 -> code 0 at shift 4 of byte 0
    trits[64] = 0    #  0 -> code 1 at shift 2 of byte 0
    trits[96] = 1    # +1 -> code 2 at shift 0 of byte 0
    trits[15] = -1   # -1 -> code 0 at shift 6 of byte 15
    trits[127] = 1   # +1 -> code 2 at shift 0 of byte 31

    codes = (trits + 1).to(torch.uint8)
    packed = (
        (codes[0:32] << 6)
        | (codes[32:64] << 4)
        | (codes[64:96] << 2)
        | codes[96:128]
    )
    scale = 1.5
    trailer = bytearray(32)
    trailer[0:4] = struct.pack("<f", scale)
    raw = torch.cat([packed, torch.tensor(list(trailer), dtype=torch.uint8)])
    out = dequantize_gguf_tensor(raw, GGML_TYPE_I2_S, dtype=torch.float32)
    expected = trits.float() * scale
    assert out.shape == (128,)
    assert torch.allclose(out, expected, atol=1e-6)


def test_gguf_bitnet_minicpm3_granite_smollm3_mappings():
    from transformers import AutoConfig
    from quintessentia_cli.gguf_arch_registry import (
        LlamaRopeReversePermute,
        build_gguf_config_dict,
        get_extended_gguf_mapping,
        is_extended_gguf_arch_supported,
    )

    for arch in ("bitnet", "minicpm3", "granite", "smollm3"):
        assert is_extended_gguf_arch_supported(arch)

    # BitNet config + sub_norm + RoPE permute
    bn_meta = {
        "general.architecture": "bitnet",
        "bitnet.block_count": 2,
        "bitnet.context_length": 4096,
        "bitnet.embedding_length": 256,
        "bitnet.feed_forward_length": 512,
        "bitnet.attention.head_count": 4,
        "bitnet.attention.head_count_kv": 2,
        "bitnet.rope.freq_base": 500000.0,
        "bitnet.attention.layer_norm_rms_epsilon": 1e-5,
        "tokenizer.ggml.tokens": 1000,
    }
    bn_cfg_d = build_gguf_config_dict(bn_meta, ("token_embd.weight", "blk.0.attn_sub_norm.weight"))
    bn_mt = bn_cfg_d.pop("model_type")
    bn_cfg = AutoConfig.for_model(bn_mt, **bn_cfg_d)
    bn_rules = get_extended_gguf_mapping("bitnet", bn_cfg)
    assert any(getattr(r, "target_patterns", None) == [".self_attn.attn_sub_norm."] for r in bn_rules)
    assert any(getattr(r, "target_patterns", None) == [".mlp.ffn_sub_norm."] for r in bn_rules)
    assert any(
        any(isinstance(op, LlamaRopeReversePermute) for op in getattr(r, "operations", ()))
        for r in bn_rules
    )

    # MiniCPM3 MLA config fields
    mc_meta = {
        "general.architecture": "minicpm3",
        "minicpm3.block_count": 2,
        "minicpm3.context_length": 32768,
        "minicpm3.embedding_length": 256,
        "minicpm3.feed_forward_length": 512,
        "minicpm3.attention.head_count": 4,
        "minicpm3.attention.head_count_kv": 4,
        "minicpm3.attention.q_lora_rank": 64,
        "minicpm3.attention.kv_lora_rank": 32,
        "minicpm3.attention.key_length": 96,
        "minicpm3.rope.dimension_count": 32,
        "minicpm3.attention.layer_norm_rms_epsilon": 1e-5,
        "tokenizer.ggml.tokens": 1000,
    }
    mc_cfg_d = build_gguf_config_dict(
        mc_meta,
        ("token_embd.weight", "blk.0.attn_kv_b.weight"),
        tensor_shapes={"blk.0.attn_kv_b.weight": (512, 32)},
    )
    assert mc_cfg_d["qk_rope_head_dim"] == 32
    assert mc_cfg_d["qk_nope_head_dim"] == 64
    assert mc_cfg_d["v_head_dim"] == 64
    mc_mt = mc_cfg_d.pop("model_type")
    mc_cfg = AutoConfig.for_model(mc_mt, **mc_cfg_d)
    assert mc_cfg.q_lora_rank == 64 and mc_cfg.kv_lora_rank == 32

    # Granite & SmolLM3 get LlamaRopeReversePermute
    for arch in ("granite", "smollm3"):
        dummy_cfg = SimpleNamespace(model_type=arch, num_attention_heads=16, num_key_value_heads=4)
        rules = get_extended_gguf_mapping(arch, dummy_cfg)
        assert any(
            any(isinstance(op, LlamaRopeReversePermute) for op in getattr(r, "operations", ()))
            for r in rules
        ), f"{arch} must include LlamaRopeReversePermute"


def test_auto_map_remote_code_raises_unsupported_instead_of_prompting(tmp_path):
    from safetensors.torch import save_file
    from quintessentia_cli.model_source import UnsupportedModelError, check_supported, load_runtime_config

    d = tmp_path / "custom_remote_model"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps(
            {
                "model_type": "totally_custom_unregistered_arch_xyz",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "auto_map": {"AutoConfig": "configuration_custom.CustomConfig"},
            }
        ),
        encoding="utf-8",
    )
    save_file({"model.embed_tokens.weight": torch.zeros(16, 64)}, str(d / "model.safetensors"))

    with pytest.raises(UnsupportedModelError, match="auto_map|custom remote code|cannot load"):
        check_supported(d)
    with pytest.raises(UnsupportedModelError):
        load_runtime_config(d)


def test_qwen2_vl_config_unwraps_to_qwen2_text_causal_lm(tmp_path):
    from safetensors.torch import save_file
    from quintessentia_cli.model_source import load_runtime_config

    d = tmp_path / "qwen2_vl_tiny"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2_vl",
                "vocab_size": 512,
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "max_position_embeddings": 2048,
                "tie_word_embeddings": True,
            }
        ),
        encoding="utf-8",
    )
    save_file({"model.embed_tokens.weight": torch.zeros(512, 128)}, str(d / "model.safetensors"))
    cfg = load_runtime_config(d)
    assert cfg.model_type == "qwen2"
    assert cfg.hidden_size == 128 and cfg.num_hidden_layers == 2


def test_session_unpark_handles_none_conv_and_recurrent_slots():
    """Regression test for Zamba2HybridDynamicCache where lazy_initialization leaves None slots."""
    from quintessentia_cli.unified_runtime import QuintessentiaKVPrefixSession

    class _MockHybridLayer:
        def __init__(self):
            self.conv_states = {0: None}
            self.recurrent_states = {0: None}
            self.is_conv_states_initialized = {0: False}
            self.is_recurrent_states_initialized = {0: False}
            self.conv_kernel_size = {0: None}
            self.has_previous_state = {0: False}

        def lazy_initialization(self, conv_states=None, recurrent_states=None, state_idx=0, conv_kernel_size=None):
            # Simulate Zamba2HybridDynamicCache leaving slot and conv_kernel_size as None
            pass

    class _MockConfig:
        pass

    sc = QuintessentiaKVPrefixSession()
    sc.config = _MockConfig()
    sc.fed = [1, 2, 3]
    c_t = torch.ones(1, 8, 4)
    r_t = torch.full((1, 4, 8), 2.0)
    sc.parked_backbone = {
        0: {
            "linear_states": {
                0: {
                    "conv_states": c_t,
                    "recurrent_states": r_t,
                    "conv_kernel_size": 4,
                    "has_previous_state": True,
                }
            }
        }
    }

    # Monkeypatch DynamicCache to yield our mock hybrid layer
    import transformers.cache_utils as cu

    orig_dc = cu.DynamicCache

    class _MockDC:
        def __init__(self, config=None):
            self.layers = [_MockHybridLayer()]

    cu.DynamicCache = _MockDC
    try:
        past, _ = sc.unpark_to_gpu(torch.device("cpu"), None, first_new_token=4)
    finally:
        cu.DynamicCache = orig_dc

    l0 = past.layers[0]
    assert torch.equal(l0.conv_states[0], c_t)
    assert torch.equal(l0.recurrent_states[0], r_t)
    assert l0.has_previous_state[0] is True
    assert l0.is_conv_states_initialized[0] is True
    assert l0.is_recurrent_states_initialized[0] is True
    assert l0.conv_kernel_size[0] == 4


def test_kv_offload_supports_asymmetric_mla_k_v_dims():
    """MLA caches (MiniCPM3, DeepSeek-V2/V3) store K with dim=kv_lora_rank and V with dim=qk_rope_head_dim."""
    from quintessentia_cli.kv_offload import _build_kv_classes

    PreallocatedGPUDynamicLayer, PinnedOffloadedDynamicLayer, _ = _build_kv_classes()

    k1 = torch.randn(1, 1, 5, 256)
    v1 = torch.randn(1, 1, 5, 32)
    k2 = torch.randn(1, 1, 2, 256)
    v2 = torch.randn(1, 1, 2, 32)

    for layer in (
        PreallocatedGPUDynamicLayer(initial_capacity=64, target_device=torch.device("cpu")),
        PinnedOffloadedDynamicLayer(initial_capacity=64, pin_memory=False, target_device=torch.device("cpu")),
    ):
        layer.seed_from_cpu_tensors(k1, v1, torch.device("cpu"), is_bshd=False)
        ok, ov = layer.update(k2, v2)
        assert ok.shape == (1, 1, 7, 256)
        assert ov.shape == (1, 1, 7, 32)
        assert torch.allclose(ok[:, :, :5, :], k1)
        assert torch.allclose(ov[:, :, :5, :], v1)
        assert torch.allclose(ok[:, :, 5:, :], k2)
        assert torch.allclose(ov[:, :, 5:, :], v2)

