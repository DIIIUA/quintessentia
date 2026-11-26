"""Comprehensive CPU tests for universal multi-protocol tool calling and auto runtime limits."""
from __future__ import annotations

import http.client
import json
from types import SimpleNamespace

import pytest

from quintessentia_cli.lan_server import QuintessentiaLANServer
from quintessentia_cli.runtime_config import compute_runtime_limits
from quintessentia_cli.tool_calling import (
    StreamingToolCallGuard,
    apply_chat_template_with_tools,
    normalize_inline_tool_calls,
    parse_tool_calls,
)


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "metric": {"type": "boolean"},
                    "tags": {"type": "array"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_list",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "target_env": {"type": "string"},
                },
            },
        },
    },
]


def _args(tc):
    return json.loads(tc["function"]["arguments"])


# ---------------------------------------------------------------------------
# 1. All 12+ LLM wire protocols
# ---------------------------------------------------------------------------

def test_parse_qwen35_xml_with_typed_coercion():
    text = (
        "Проверю погоду.\n"
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=city>Kyiv</parameter>\n"
        "<parameter=days>3</parameter>\n"
        "<parameter=metric>true</parameter>\n"
        "<parameter=tags>[\"rain\", \"wind\"]</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    clean, calls = parse_tool_calls(text, tools=SAMPLE_TOOLS)
    assert clean == "Проверю погоду."
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert _args(calls[0]) == {"city": "Kyiv", "days": 3, "metric": True, "tags": ["rain", "wind"]}


def test_parse_qwen25_qwen3_smollm3_hermes_json_and_flat_json():
    # Canonical JSON inside <tool_call>
    clean, calls = parse_tool_calls(
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Odesa", "days": 2}}</tool_call>',
        tools=SAMPLE_TOOLS,
    )
    assert clean == "" and len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert _args(calls[0]) == {"city": "Odesa", "days": 2}

    # Real Qwythos-9B flat JSON bug (missing "arguments" wrapper)
    flat = '<tool_call>\n{"name": "file_list", "path": "", "target_env": "internal"}\n</tool_call>'
    clean2, calls2 = parse_tool_calls(flat, tools=SAMPLE_TOOLS)
    assert clean2 == "" and len(calls2) == 1
    assert calls2[0]["function"]["name"] == "file_list"
    assert _args(calls2[0]) == {"path": "", "target_env": "internal"}

    # Prompt-only inline normalization for clients that do not pass OpenAI `tools`
    norm = normalize_inline_tool_calls(flat)
    assert "<tool_call>" in norm and '"arguments"' in norm
    inner_json = norm.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
    assert json.loads(inner_json) == {
        "name": "file_list",
        "arguments": {"path": "", "target_env": "internal"},
    }


def test_parse_granite_tool_call():
    text = '<|tool_call|>[{"name": "get_weather", "arguments": {"city": "Lviv", "days": 1}}]'
    clean, calls = parse_tool_calls(text, tools=SAMPLE_TOOLS)
    assert clean == "" and len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert _args(calls[0]) == {"city": "Lviv", "days": 1}


def test_parse_gpt_oss_harmony():
    text = (
        '<|channel|>commentary to=functions.get_weather <|constrain|>json'
        '<|message|>{"city": "Warsaw", "days": 5}<|call|>'
    )
    clean, calls = parse_tool_calls(text, tools=SAMPLE_TOOLS)
    assert clean == "" and len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert _args(calls[0]) == {"city": "Warsaw", "days": 5}


def test_parse_cohere_command_r_tags_and_action_block():
    text1 = '<|START_ACTION|>[{"tool_name": "file_list", "parameters": {"path": "/tmp"}}]<|END_ACTION|>'
    _, calls1 = parse_tool_calls(text1, tools=SAMPLE_TOOLS)
    assert len(calls1) == 1 and calls1[0]["function"]["name"] == "file_list"
    assert _args(calls1[0]) == {"path": "/tmp"}

    text2 = 'Action: ```json\n[{"tool_name": "get_weather", "parameters": {"city": "Prague"}}]\n```'
    _, calls2 = parse_tool_calls(text2, tools=SAMPLE_TOOLS)
    assert len(calls2) == 1 and calls2[0]["function"]["name"] == "get_weather"
    assert _args(calls2[0]) == {"city": "Prague"}


def test_parse_phi4_tags_and_functools():
    text1 = '<|tool_call|>[{"name": "file_list", "arguments": {"path": "docs"}}]<|/tool_call|>'
    _, calls1 = parse_tool_calls(text1, tools=SAMPLE_TOOLS)
    assert len(calls1) == 1 and calls1[0]["function"]["name"] == "file_list"
    assert _args(calls1[0]) == {"path": "docs"}

    text2 = 'functools[{"name": "get_weather", "arguments": {"city": "Berlin"}}]'
    _, calls2 = parse_tool_calls(text2, tools=SAMPLE_TOOLS)
    assert len(calls2) == 1 and calls2[0]["function"]["name"] == "get_weather"
    assert _args(calls2[0]) == {"city": "Berlin"}


def test_parse_llama_python_tag_bare_json_and_pythonic():
    _, c1 = parse_tool_calls('<|python_tag|>{"name": "get_weather", "parameters": {"city": "Rome"}}', tools=SAMPLE_TOOLS)
    assert len(c1) == 1 and c1[0]["function"]["name"] == "get_weather" and _args(c1[0]) == {"city": "Rome"}

    _, c2 = parse_tool_calls('{"name": "file_list", "parameters": {"path": "src"}}', tools=SAMPLE_TOOLS, allow_bare_json=True)
    assert len(c2) == 1 and c2[0]["function"]["name"] == "file_list" and _args(c2[0]) == {"path": "src"}

    _, c3 = parse_tool_calls('[get_weather(city="Paris", days=4)]', tools=SAMPLE_TOOLS)
    assert len(c3) == 1 and c3[0]["function"]["name"] == "get_weather" and _args(c3[0]) == {"city": "Paris", "days": 4}


def test_parse_mistral_json_and_args_format():
    _, c1 = parse_tool_calls('[TOOL_CALLS][{"name": "file_list", "arguments": {"path": "."}}]', tools=SAMPLE_TOOLS)
    assert len(c1) == 1 and c1[0]["function"]["name"] == "file_list" and _args(c1[0]) == {"path": "."}

    _, c2 = parse_tool_calls('[TOOL_CALLS]get_weather[ARGS]{"city": "Madrid"}', tools=SAMPLE_TOOLS)
    assert len(c2) == 1 and c2[0]["function"]["name"] == "get_weather" and _args(c2[0]) == {"city": "Madrid"}


def test_parse_deepseek_v3_r1():
    text = (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
        '```json\n{"city": "Tokyo", "days": 7}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
    )
    _, calls = parse_tool_calls(text, tools=SAMPLE_TOOLS)
    assert len(calls) == 1 and calls[0]["function"]["name"] == "get_weather"
    assert _args(calls[0]) == {"city": "Tokyo", "days": 7}


def test_parse_lfm2_and_internlm():
    _, c1 = parse_tool_calls('<|tool_call_start|>[get_weather(city="Vienna", days=2)]<|tool_call_end|>', tools=SAMPLE_TOOLS)
    assert len(c1) == 1 and c1[0]["function"]["name"] == "get_weather" and _args(c1[0]) == {"city": "Vienna", "days": 2}

    _, c2 = parse_tool_calls('<|action_start|><|plugin|>{"name": "file_list", "parameters": {"path": "data"}}<|action_end|>', tools=SAMPLE_TOOLS)
    assert len(c2) == 1 and c2[0]["function"]["name"] == "file_list" and _args(c2[0]) == {"path": "data"}


# ---------------------------------------------------------------------------
# 2. Universal prompt rendering (`apply_chat_template_with_tools`)
# ---------------------------------------------------------------------------

def test_apply_chat_template_detects_silent_tools_drop_and_injects_fallback():
    class IgnoringTemplateTokenizer:
        chat_template = "dummy"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
            # Intentionally ignores kw["tools"] the way many older HF templates do
            return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    prompt = apply_chat_template_with_tools(
        IgnoringTemplateTokenizer(),
        [{"role": "user", "content": "What is the weather?"}],
        tools=SAMPLE_TOOLS,
        tool_choice="auto",
    )
    assert "# Tools" in prompt and "get_weather" in prompt and "file_list" in prompt


def test_apply_chat_template_falls_back_when_role_tool_rejected():
    class StrictAlternationTokenizer:
        chat_template = "strict"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
            for m in messages:
                if m["role"] == "tool" or "tool_calls" in m:
                    raise ValueError("Unsupported role: tool")
            return "\n".join(f"[{m['role']}] {m['content']}" for m in messages) + "\n[assistant]"

    history = [
        {"role": "user", "content": "List files"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "file_list", "arguments": '{"path": "."}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "file_list", "content": '["a.py", "b.py"]'},
    ]
    prompt = apply_chat_template_with_tools(StrictAlternationTokenizer(), history, tools=SAMPLE_TOOLS)
    assert "<tool_call>" in prompt and "<tool_response" in prompt and '["a.py", "b.py"]' in prompt


# ---------------------------------------------------------------------------
# 3. Streaming guard
# ---------------------------------------------------------------------------

def test_streaming_guard_withholds_split_tool_tags_and_emits_prose():
    guard = StreamingToolCallGuard(tools=SAMPLE_TOOLS, structured_mode=True)
    chunks = [
        "Секунду, проверю. ",
        "<to",
        "ol_ca",
        "ll>\n<function=get_weather>\n<parameter=city>Kyiv</parameter>\n</function>\n</tool_call>",
    ]
    emitted = []
    for c in chunks:
        emitted.extend(guard.push(c))
    rem, calls = guard.finish()
    prose = "".join(emitted) + rem
    assert prose.strip() == "Секунду, проверю."
    assert "<tool_call>" not in prose
    assert len(calls) == 1 and calls[0]["function"]["name"] == "get_weather"
    assert _args(calls[0]) == {"city": "Kyiv"}


def test_streaming_guard_prompt_only_normalizes_inline_flat_json():
    guard = StreamingToolCallGuard(tools=None, structured_mode=False)
    chunks = [
        "Смотрю файлы:\n<tool_",
        'call>{"name": "file_list", "path": "", "target_env": "internal"}</tool',
        "_call>",
    ]
    out = []
    for c in chunks:
        out.extend(guard.push(c))
    rem, calls = guard.finish()
    full = "".join(out) + rem
    assert calls == []
    assert '"arguments": {"path": "", "target_env": "internal"}' in full


# ---------------------------------------------------------------------------
# 4. Auto runtime limits calculation across hardware & architectures
# ---------------------------------------------------------------------------

def test_compute_runtime_limits_for_qwen35_hybrid_and_ssm_and_small_gpu():
    GiB = 1024 ** 3
    MiB = 1024 ** 2

    # Qwythos-9B / Qwen3.5 hybrid: 8 full_attention + 24 linear_attention -> 32 KiB/token
    qwen35_cfg = {
        "model_type": "qwen3_5_text",
        "max_position_embeddings": 262144,
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "layer_types": ["linear_attention"] * 24 + ["full_attention"] * 8,
        "linear_num_value_heads": 32,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
    }
    # 8 GB RTX 3060 Ti with ~2.2 GiB free after weights
    eng_8gb = SimpleNamespace(
        config=qwen35_cfg,
        device=SimpleNamespace(type="cuda"),
        vram_mode="normal",
        prefill_chunk_tokens=512,
        plan=SimpleNamespace(total_bytes=8 * GiB, free_bytes=int(7.4 * GiB), weights_bytes=int(5.2 * GiB), max_context_tokens=8192),
        _idle_free_vram_bytes=int(2.2 * GiB),
    )
    # Use busy=True so test uses deterministic _idle_free_vram_bytes without touching host CUDA
    lim8 = compute_runtime_limits(eng_8gb, busy=True)
    assert lim8["kv_bytes_per_token"] == 32768
    assert lim8["native_context_length"] == 262144
    assert 32768 <= lim8["effective_context_length"] <= 65536
    assert 16384 <= lim8["max_output_tokens"] <= min(32768, lim8["effective_context_length"] // 2)
    assert lim8["max_input_tokens"] < lim8["effective_context_length"]
    assert lim8["context_limit_enforced"] is False

    # 24 GB GPU with 17 GiB free -> fits entire 262144 native context!
    eng_24gb = SimpleNamespace(
        config=qwen35_cfg,
        device=SimpleNamespace(type="cuda"),
        vram_mode="normal",
        prefill_chunk_tokens=512,
        plan=SimpleNamespace(total_bytes=24 * GiB, free_bytes=22 * GiB, weights_bytes=5 * GiB, max_context_tokens=8192),
        _idle_free_vram_bytes=17 * GiB,
    )
    lim24 = compute_runtime_limits(eng_24gb, busy=True)
    assert lim24["effective_context_length"] == 262144
    assert lim24["max_output_tokens"] == 32768
    assert lim24["max_input_tokens"] == 262144 - 4096

    # Pure recurrent Mamba -> zero KV growth -> full native context
    mamba_cfg = {
        "model_type": "mamba2",
        "max_position_embeddings": 131072,
        "num_hidden_layers": 24,
        "hidden_size": 2048,
        "state_size": 64,
    }
    eng_ssm = SimpleNamespace(config=mamba_cfg, device=SimpleNamespace(type="cuda"), prefill_chunk_tokens=512)
    lim_ssm = compute_runtime_limits(eng_ssm, busy=True)
    assert lim_ssm["kv_bytes_per_token"] == 0
    assert lim_ssm["effective_context_length"] == 131072
    assert lim_ssm["context_limit_source"] == "recurrent_constant_kv"

    # Standard Llama-3.2 GQA on tight 4 GB card (~1.1 GiB free after weights) -> modest safe context
    llama_cfg = {
        "model_type": "llama",
        "max_position_embeddings": 131072,
        "num_hidden_layers": 28,
        "hidden_size": 3072,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    eng_4gb = SimpleNamespace(
        config=llama_cfg,
        device=SimpleNamespace(type="cuda"),
        vram_mode="normal",
        prefill_chunk_tokens=256,
        plan=SimpleNamespace(total_bytes=4 * GiB, free_bytes=int(3.6 * GiB), weights_bytes=int(2.5 * GiB), max_context_tokens=4096),
        _idle_free_vram_bytes=int(1.1 * GiB),
    )
    lim4 = compute_runtime_limits(eng_4gb, busy=True)
    assert lim4["kv_bytes_per_token"] == 28 * 2 * 8 * 128 * 2
    assert 4096 <= lim4["effective_context_length"] <= 12288


# ---------------------------------------------------------------------------
# 5. End-to-end LAN server HTTP & SSE tool-calling tests
# ---------------------------------------------------------------------------

class ToolFakeEngine:
    model_id = "qwythos-tool-fake"

    def __init__(self, chunks, thinking=""):
        self.chunks = list(chunks)
        self.thinking = thinking
        self.last_call = {}
        self.config = SimpleNamespace(max_position_embeddings=65536)

    def status_snapshot(self):
        return {"model_id": self.model_id, "gpu_name": "fake", "vram_mode": "normal", "mtp_enabled": False}

    def reset_session(self):
        pass

    def stream_chat(
        self,
        messages,
        max_new_tokens=16,
        enable_thinking=True,
        use_session_cache=False,
        stop_on_eos=True,
        cancel_check=None,
        tools=None,
        tool_choice=None,
        **extra,
    ):
        self.last_call = {
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "extra": extra,
        }
        if self.thinking:
            yield {"type": "delta", "phase": "thinking", "text": self.thinking}
        for c in self.chunks:
            yield {"type": "delta", "phase": "answer", "text": c}
        full_ans = "".join(self.chunks)
        yield {
            "type": "done",
            "thinking_text": self.thinking,
            "answer_text": full_ans,
            "metrics": {"finish_reason": "stop", "prompt_tokens": 42, "generated_tokens": 18},
        }


def _http_post(port, path, body):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = json.dumps(body).encode("utf-8")
    conn.request("POST", path, body=data, headers={"Content-Type": "application/json", "Content-Length": str(len(data))})
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp.status, raw


def test_lan_server_tool_calling_json_and_sse_and_tool_role(tmp_path):
    eng = ToolFakeEngine(
        thinking="Нужно узнать погоду.",
        chunks=[
            "Проверяю... ",
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>Kyiv</parameter>\n",
            "<parameter=days>2</parameter>\n",
            "</function>\n</tool_call>",
        ],
    )
    srv = QuintessentiaLANServer(eng, host="127.0.0.1", http_port=0, enable_beacon=False, log_file=str(tmp_path / "srv.log"))
    srv.start_background()
    try:
        # Non-streaming JSON response
        st, raw = _http_post(
            srv.http_port,
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Weather in Kyiv for 2 days?"}],
                "tools": SAMPLE_TOOLS,
                "tool_choice": "auto",
                "temperature": 0.3,
                "stream": False,
            },
        )
        assert st == 200
        obj = json.loads(raw)
        choice = obj["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] == "Проверяю..."
        assert choice["message"]["reasoning_content"] == "Нужно узнать погоду."
        tcs = choice["message"]["tool_calls"]
        assert len(tcs) == 1 and tcs[0]["function"]["name"] == "get_weather"
        assert _args(tcs[0]) == {"city": "Kyiv", "days": 2}
        assert eng.last_call["tools"] == SAMPLE_TOOLS
        assert eng.last_call["extra"]["temperature"] == pytest.approx(0.3)

        # Streaming SSE response
        st, raw_sse = _http_post(
            srv.http_port,
            "/v1/chat/completions",
            {
                "messages": [
                    {"role": "user", "content": "Weather?"},
                    {"role": "assistant", "content": None, "tool_calls": tcs},
                    {"role": "tool", "tool_call_id": tcs[0]["id"], "name": "get_weather", "content": '{"temp": 12}'},
                ],
                "tools": SAMPLE_TOOLS,
                "stream": True,
            },
        )
        assert st == 200
        events = []
        for line in raw_sse.decode("utf-8").splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))
        content_acc = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
        reasoning_acc = "".join(e["choices"][0]["delta"].get("reasoning_content", "") for e in events)
        tool_deltas = [tc for e in events for tc in (e["choices"][0]["delta"].get("tool_calls") or [])]
        finish_reasons = [e["choices"][0]["finish_reason"] for e in events if e["choices"][0]["finish_reason"]]
        assert "<tool_call>" not in content_acc
        assert content_acc.strip() == "Проверяю..."
        assert reasoning_acc == "Нужно узнать погоду."
        assert len(tool_deltas) == 1 and tool_deltas[0]["index"] == 0
        assert tool_deltas[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_deltas[0]["function"]["arguments"]) == {"city": "Kyiv", "days": 2}
        assert finish_reasons == ["tool_calls"]
        assert eng.last_call["messages"][1]["content"] is None
        assert eng.last_call["messages"][2]["role"] == "tool"
    finally:
        srv.stop()


def test_render_incremental_turn_suffix_and_tool_calls_match():
    from quintessentia_cli.tool_calling import render_incremental_turn_suffix, tool_calls_match_signature

    class ChatMLTokenizer:
        chat_template = "chatml"
        eos_token = "<|im_end|>"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True, **kw):
            parts = []
            for m in messages:
                role = m["role"]
                if role == "tool":
                    parts.append(f"<|im_start|>user\n<tool_response>\n{m['content']}\n</tool_response><|im_end|>\n")
                else:
                    parts.append(f"<|im_start|>{role}\n{m['content']}<|im_end|>\n")
            if add_generation_prompt:
                parts.append("<|im_start|>assistant\n<think>\n" if enable_thinking else "<|im_start|>assistant\n")
            return "".join(parts)

    suffix = render_incremental_turn_suffix(
        ChatMLTokenizer(),
        [{"role": "tool", "name": "get_weather", "tool_call_id": "c1", "content": '{"temp": 14}'}],
        enable_thinking=True,
        prev_ended_with_eos=True,
    )
    assert suffix is not None
    assert not suffix.startswith("<|im_end|>")
    assert "<tool_response>\n{\"temp\": 14}\n</tool_response>" in suffix
    assert suffix.endswith("<|im_start|>assistant\n<think>\n")

    a = [{"id": "call_aaa", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Kyiv", "days": 2}'}}]
    b = [{"id": "call_zzz", "type": "function", "function": {"name": "get_weather", "arguments": {"days": 2, "city": "Kyiv"}}}]
    assert tool_calls_match_signature(a, b) is True
    assert tool_calls_match_signature(a, [{"function": {"name": "file_list", "arguments": "{}"}}]) is False


def test_pinned_offloaded_kv_cache_parity_and_crop():
    import torch
    from transformers import DynamicCache
    from quintessentia_cli.kv_offload import make_offloaded_kv_cache, release_kv_cache_gpu_buffers, set_kv_cache_chunk_pad

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))

    for dev in devices:
        ref = DynamicCache()
        off = make_offloaded_kv_cache(initial_capacity=64, chunk_pad=16, device=dev)
        gen = torch.Generator(device="cpu").manual_seed(123)
        # 2 prefill chunks (16 + 12 tokens) + 3 decode tokens across 3 layers
        for step_len in (16, 12, 1, 1, 1):
            if step_len == 1:
                set_kv_cache_chunk_pad(off, 1)
            for li in range(3):
                k = torch.randn((1, 4, step_len, 32), generator=gen, dtype=torch.bfloat16).to(dev)
                v = torch.randn((1, 4, step_len, 32), generator=gen, dtype=torch.bfloat16).to(dev)
                kr, vr = ref.update(k, v, li)
                ko, vo = off.update(k, v, li)
                assert torch.equal(kr, ko), f"K mismatch on {dev} layer {li} step {step_len}"
                assert torch.equal(vr, vo), f"V mismatch on {dev} layer {li} step {step_len}"

        assert off.get_seq_length() == 31
        release_kv_cache_gpu_buffers(off)
        for li in range(3):
            assert off.layers[li].keys.device.type == "cpu"
            assert off.layers[li].keys.shape == (1, 4, 31, 32)
        off.crop(-2)
        assert off.get_seq_length() == 29

        # Seed a fresh offloaded cache from parked CPU tensors (simulates unpark_to_gpu)
        off2 = make_offloaded_kv_cache(initial_capacity=64, chunk_pad=8, device=dev)
        for li in range(3):
            off2.layers.append(off2.layer_class_to_replicate())
            off2.layers[li].seed_from_cpu_tensors(off.layers[li].keys, off.layers[li].values, dev)
        assert off2.get_seq_length() == 29
        k_next = torch.randn((1, 4, 1, 32), generator=gen, dtype=torch.bfloat16).to(dev)
        v_next = torch.randn((1, 4, 1, 32), generator=gen, dtype=torch.bfloat16).to(dev)
        ko2, vo2 = off2.update(k_next, v_next, 0)
        assert ko2.shape == (1, 4, 30, 32) and ko2.device == dev
        release_kv_cache_gpu_buffers(off2)

        # O(1) zero-copy snapshot/restore compatibility with thought_buffer & GeneralSpeculation
        from quintessentia_cli.thought_buffer import capture_cache_snapshot, restore_cache_snapshot
        from src.general_speculation import GeneralSpeculation
        snap = capture_cache_snapshot(off)
        assert snap.vram_bytes == 0
        assert GeneralSpeculation._snapshot_bytes(off) == 0
        assert GeneralSpeculation._crop_append_only(off, 29) is True
        k_tmp = torch.randn((1, 4, 2, 32), generator=gen, dtype=torch.bfloat16).to(dev)
        v_tmp = torch.randn((1, 4, 2, 32), generator=gen, dtype=torch.bfloat16).to(dev)
        for li in range(3):
            off.update(k_tmp, v_tmp, li)
        assert off.get_seq_length() == 31
        restore_cache_snapshot(off, snap)
        assert off.get_seq_length() == 29
        for li in range(3):
            assert off.layers[li].keys.shape == (1, 4, 29, 32)

        # Hybrid partial layer offload (e.g. 2 GPU preallocated + 1 pinned RAM, and 1 GPU + 2 RAM) + promote
        if dev.type == "cuda":
            from types import SimpleNamespace as _SN
            from transformers import PretrainedConfig
            hyb_cfg = PretrainedConfig(num_hidden_layers=3, layer_types=["full_attention"] * 3)
            for n_off in (0, 1, 2):
                ref_h = DynamicCache(config=hyb_cfg)
                hyb = make_offloaded_kv_cache(
                    hyb_cfg,
                    initial_capacity=64,
                    chunk_pad=16,
                    device=dev,
                    offload_layer_count=n_off,
                )
                assert hyb.gpu_kv_layer_count == 3 - n_off
                assert hyb.offloaded_kv_layer_count == n_off
                assert hyb.is_quintessentia_offloaded is (n_off > 0)
                for step_len in (16, 12, 1, 1, 1):
                    if step_len == 1:
                        set_kv_cache_chunk_pad(hyb, 1)
                    for li in range(3):
                        k = torch.randn((1, 4, step_len, 32), generator=gen, dtype=torch.bfloat16).to(dev)
                        v = torch.randn((1, 4, step_len, 32), generator=gen, dtype=torch.bfloat16).to(dev)
                        kr, vr = ref_h.update(k, v, li)
                        kh, vh = hyb.update(k, v, li)
                        assert torch.equal(kr, kh), f"Hybrid n_off={n_off} K mismatch L{li} step={step_len}"
                        assert torch.equal(vr, vh), f"Hybrid n_off={n_off} V mismatch L{li} step={step_len}"
                # Promote offloaded layers into freed VRAM at prefill->decode boundary
                if n_off > 0:
                    promoted = hyb.promote_offloaded_layers(avail_vram_bytes=256 * 1024**2, reserve_bytes=16 * 1024**2)
                    assert promoted == n_off
                    assert hyb.offloaded_kv_layer_count == 0
                    assert hyb.gpu_kv_layer_count == 3
                    assert hyb.offloading is False
                    k1 = torch.randn((1, 4, 1, 32), generator=gen, dtype=torch.bfloat16).to(dev)
                    v1 = torch.randn((1, 4, 1, 32), generator=gen, dtype=torch.bfloat16).to(dev)
                    for li in range(3):
                        kr, vr = ref_h.update(k1, v1, li)
                        kh, vh = hyb.update(k1, v1, li)
                        assert torch.equal(kr, kh) and torch.equal(vr, vh)
                # O(1) zero-copy snapshot/restore works on PreallocatedGPUDynamicLayer too
                snap_h = capture_cache_snapshot(hyb)
                assert snap_h.vram_bytes == 0
                assert GeneralSpeculation._snapshot_bytes(hyb) == 0
                release_kv_cache_gpu_buffers(hyb)
