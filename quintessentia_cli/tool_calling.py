"""Universal multi-protocol OpenAI Tool Calling layer for Quintessentia.

Supports every major open-weight LLM tool-calling wire format across the entire
Quintessentia model zoo (Qwen 2.5 / 3 / 3.5 / 3.6, Llama 3.x / 4, IBM Granite 3/4,
OpenAI GPT-OSS Harmony, Cohere Command-R, Microsoft Phi-4, Mistral / Devstral,
DeepSeek V3 / R1, Liquid LFM2, NVIDIA Nemotron, SmolLM3 / Hermes, Gemma, InternLM,
GLM, and any model whose chat_template does not natively accept `tools`):

1. Input side (`prepare_messages_and_tools`, `apply_chat_template_with_tools`):
   - Preserves full OpenAI message structures (`role="tool"`, `tool_call_id`, `name`,
     `assistant.tool_calls`, `reasoning_content`, multipart `content`).
   - Passes `tools` to `tokenizer.apply_chat_template(..., tools=tools)` when supported,
     AND verifies that the template actually rendered the tool declarations (many Jinja
     templates silently ignore an unknown `tools` kwarg).
   - If the model's `chat_template` ignores or lacks `tools` support, injects a compact,
     model-agnostic `# Tools` block using canonical `<tool_call>` JSON format.
   - Honors `tool_choice` (`"none"`, `"auto"`, `"required"`, or `{"type": "function", "function": {"name": "..."}}`).

2. Output side (`parse_tool_calls`, `StreamingToolCallGuard`, `normalize_inline_tool_calls`):
   - Parses all 12+ native model formats into canonical OpenAI `tool_calls` objects:
       * Qwen 3.5 / 3.6 XML: `<tool_call><function=NAME><parameter=k>v</parameter></function></tool_call>`
       * XML/tag JSON: `<tool_call>...</tool_call>`, `<toolcall>...</toolcall>`,
         `<|tool_call|>...<|/tool_call|>`, `<|tool_call|>[...]` (Granite / Phi-4),
         `<|action_start|>...<|action_end|>` (InternLM), `<|START_ACTION|>...<|END_ACTION|>` (Command-R)
       * OpenAI GPT-OSS Harmony: `<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{...}<|call|>`
       * Mistral: `[TOOL_CALLS][{...}]` and `[TOOL_CALLS]name[ARGS]{...}`
       * DeepSeek V3/R1: `<｜tool▁calls▁begin｜>...<｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME...`
       * Llama 3.x / Phi-4 / LFM2 Pythonic & tag calls: `<|python_tag|>...`, `functools[...]`,
         `<|tool_call_start|>[func(a="b")]<|tool_call_end|>` (safe `ast.parse`, never `eval`)
       * Cohere `Action: ```json [...]````
       * Bare JSON / fenced ```json``` fallback (when `tools` are active or name matches known tools),
         including automatic normalization of flat JSON `{"name": "fn", "a": 1}` -> `{"name": "fn", "arguments": {"a": 1}}`.
   - Provides `StreamingToolCallGuard` so SSE streams emit normal prose tokens immediately
     with zero latency while cleanly withholding tool-call markup for structured `delta.tool_calls`.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import uuid
from typing import Any


# ---------------------------------------------------------------------------
# Helpers: tool schema inspection & stable cache signatures
# ---------------------------------------------------------------------------

def extract_tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    if not tools or not isinstance(tools, list):
        return names
    for item in tools:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name") if isinstance(fn, dict) else None
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return names


def extract_tool_param_schemas(tools: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Map tool_name -> {param_name: schema_dict} for typed coercion of XML parameters."""
    schemas: dict[str, dict[str, Any]] = {}
    if not tools or not isinstance(tools, list):
        return schemas
    for item in tools:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name") if isinstance(fn, dict) else None
        params = fn.get("parameters") if isinstance(fn, dict) else None
        if isinstance(name, str) and name.strip() and isinstance(params, dict):
            props = params.get("properties")
            if isinstance(props, dict):
                schemas[name.strip()] = {str(k): v for k, v in props.items() if isinstance(v, dict)}
    return schemas


def tools_signature(tools: list[dict[str, Any]] | None, tool_choice: Any = None) -> str:
    if not tools and tool_choice is None:
        return ""
    blob = json.dumps({"tools": tools or [], "tool_choice": tool_choice}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def is_tool_choice_none(tool_choice: Any) -> bool:
    return isinstance(tool_choice, str) and tool_choice.strip().lower() == "none"


def forced_tool_name(tool_choice: Any) -> str | None:
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str) and fn["name"].strip():
            return fn["name"].strip()
        if isinstance(tool_choice.get("name"), str) and tool_choice["name"].strip():
            return tool_choice["name"].strip()
    return None


# ---------------------------------------------------------------------------
# Message normalization & universal chat-template application
# ---------------------------------------------------------------------------

def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if "text" in part and part["text"] is not None:
                    parts.append(str(part["text"]))
                elif part.get("type") == "text" and "content" in part:
                    parts.append(str(part["content"]))
        return "".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def normalize_messages(
    messages: list[dict[str, Any]],
    *,
    parse_tool_args_as_dict: bool = True,
) -> list[dict[str, Any]]:
    """Normalize OpenAI chat messages while preserving tool_calls, tool_call_id, name, and reasoning."""
    out: list[dict[str, Any]] = []
    for raw in messages:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role") or "user")
        if role == "developer":
            role = "system"
        msg: dict[str, Any] = {"role": role, "content": _flatten_content(raw.get("content"))}
        if raw.get("name") is not None:
            msg["name"] = str(raw["name"])
        if raw.get("tool_call_id") is not None:
            msg["tool_call_id"] = str(raw["tool_call_id"])
        if raw.get("reasoning_content") is not None:
            msg["reasoning_content"] = str(raw["reasoning_content"])

        raw_tcs = raw.get("tool_calls")
        if isinstance(raw_tcs, list) and raw_tcs:
            norm_tcs: list[dict[str, Any]] = []
            for idx, tc in enumerate(raw_tcs):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                fn_name = str(fn.get("name") or "")
                raw_args = fn.get("arguments", {})
                if parse_tool_args_as_dict:
                    if isinstance(raw_args, str):
                        try:
                            parsed = json.loads(raw_args) if raw_args.strip() else {}
                            args_val: Any = parsed if isinstance(parsed, dict) else {"value": parsed}
                        except json.JSONDecodeError:
                            args_val = {"raw": raw_args}
                    elif isinstance(raw_args, dict):
                        args_val = raw_args
                    else:
                        args_val = {}
                else:
                    args_val = raw_args if isinstance(raw_args, str) else json.dumps(raw_args or {}, ensure_ascii=False)
                norm_tcs.append(
                    {
                        "id": str(tc.get("id") or f"call_{idx}"),
                        "type": "function",
                        "function": {"name": fn_name, "arguments": args_val},
                    }
                )
            if norm_tcs:
                msg["tool_calls"] = norm_tcs
        out.append(msg)
    return out


def _render_fallback_tools_instruction(tools: list[dict[str, Any]], tool_choice: Any = None) -> str:
    lines = [
        "# Tools",
        "",
        "You have access to the following functions:",
        "<tools>",
    ]
    for tool in tools:
        lines.append(json.dumps(tool, ensure_ascii=False))
    lines.extend(
        [
            "</tools>",
            "",
            "To call a function, reply with a <tool_call> block containing valid JSON with \"name\" and \"arguments\":",
            "<tool_call>",
            '{"name": "function_name", "arguments": {"param": "value"}}',
            "</tool_call>",
        ]
    )
    forced = forced_tool_name(tool_choice)
    if forced:
        lines.append(f'You MUST call the function "{forced}" using the <tool_call> format above.')
    elif isinstance(tool_choice, str) and tool_choice.strip().lower() == "required":
        lines.append("You MUST call at least one of the available functions using the <tool_call> format above.")
    return "\n".join(lines)


def _inject_system_block(messages: list[dict[str, Any]], block: str) -> list[dict[str, Any]]:
    if not block.strip():
        return messages
    cloned = [dict(m) for m in messages]
    if cloned and cloned[0].get("role") == "system":
        existing = str(cloned[0].get("content") or "").strip()
        cloned[0]["content"] = f"{existing}\n\n{block}" if existing else block
    else:
        cloned.insert(0, {"role": "system", "content": block})
    return cloned


def _flatten_tool_history_for_plain_template(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert assistant.tool_calls and role='tool' into clean chat turns when a template rejects them."""
    plain: list[dict[str, str]] = []
    for m in messages:
        role = str(m.get("role") or "user")
        content = str(m.get("content") or "")
        if role == "assistant" and m.get("tool_calls"):
            call_blocks = []
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                call_blocks.append(
                    "<tool_call>\n"
                    + json.dumps({"name": fn.get("name", ""), "arguments": args if args is not None else {}}, ensure_ascii=False)
                    + "\n</tool_call>"
                )
            joined = "\n".join(call_blocks)
            text = f"{content}\n{joined}".strip() if content.strip() else joined
            plain.append({"role": "assistant", "content": text})
        elif role == "tool":
            tname = m.get("name") or ""
            header = f' name="{tname}"' if tname else ""
            plain.append({"role": "user", "content": f"<tool_response{header}>\n{content}\n</tool_response>"})
        else:
            plain.append({"role": role, "content": content})
    return plain


def _try_apply_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    enable_thinking: bool,
) -> str | None:
    if not getattr(tokenizer, "chat_template", None):
        return None
    attempts: list[dict[str, Any]] = []
    base = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        attempts.append({**base, "tools": tools, "enable_thinking": enable_thinking})
        attempts.append({**base, "tools": tools})
    attempts.append({**base, "enable_thinking": enable_thinking})
    attempts.append(dict(base))

    for kw in attempts:
        try:
            res = tokenizer.apply_chat_template(messages, **kw)
            if isinstance(res, str) and res:
                return res
        except Exception:
            continue
    return None


def apply_chat_template_with_tools(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    enable_thinking: bool = True,
) -> str:
    """Render prompt string for ANY model architecture, guaranteeing tools and tool turns are never dropped."""
    active_tools = None if (not tools or is_tool_choice_none(tool_choice)) else list(tools)
    tool_names = extract_tool_names(active_tools)
    forced = forced_tool_name(tool_choice)

    norm_dict = normalize_messages(messages, parse_tool_args_as_dict=True)
    norm_str = normalize_messages(messages, parse_tool_args_as_dict=False)

    # First try the model's native chat_template with full structured messages + tools
    prompt_str: str | None = None
    if getattr(tokenizer, "chat_template", None):
        for candidate_msgs in (norm_dict, norm_str):
            if forced:
                dir_msg = _inject_system_block(
                    candidate_msgs, f'You MUST call the function "{forced}" for this response.'
                )
            elif isinstance(tool_choice, str) and tool_choice.strip().lower() == "required":
                dir_msg = _inject_system_block(
                    candidate_msgs, "You MUST call at least one available tool for this response."
                )
            else:
                dir_msg = candidate_msgs

            cand = _try_apply_template(
                tokenizer, dir_msg, tools=active_tools, enable_thinking=enable_thinking
            )
            if cand is not None:
                # Verify that the template actually rendered the declared tools (some templates silently ignore `tools=`)
                if active_tools and tool_names and not any(name in cand for name in tool_names):
                    injected = _inject_system_block(
                        _flatten_tool_history_for_plain_template(candidate_msgs),
                        _render_fallback_tools_instruction(active_tools, tool_choice),
                    )
                    cand2 = _try_apply_template(
                        tokenizer, injected, tools=None, enable_thinking=enable_thinking
                    )
                    if cand2 is not None:
                        prompt_str = cand2
                        break
                prompt_str = cand
                break

        # If native structured messages failed (e.g. strict Gemma/Mistral role alternation rejecting role="tool"),
        # fall back cleanly to flattened tool turns + fallback tool prompt if needed.
        if prompt_str is None:
            flat_msgs = _flatten_tool_history_for_plain_template(norm_dict)
            if active_tools:
                flat_msgs = _inject_system_block(
                    flat_msgs, _render_fallback_tools_instruction(active_tools, tool_choice)
                )
            prompt_str = _try_apply_template(
                tokenizer, flat_msgs, tools=None, enable_thinking=enable_thinking
            )

    if prompt_str is not None:
        return prompt_str

    # Generic fallback for base models without any chat_template at all
    flat_msgs = _flatten_tool_history_for_plain_template(norm_dict)
    if active_tools:
        flat_msgs = _inject_system_block(
            flat_msgs, _render_fallback_tools_instruction(active_tools, tool_choice)
        )
    parts: list[str] = []
    for m in flat_msgs:
        role = m["role"]
        content = m["content"]
        if role == "system":
            parts.append(f"System: {content}\n\n")
        elif role == "assistant":
            parts.append(f"Assistant:{content}\n\n")
        else:
            parts.append(f"User: {content}\n\n")
    parts.append("Assistant:")
    return "".join(parts)


def render_incremental_turn_suffix(
    tokenizer: Any,
    new_messages: list[dict[str, Any]],
    *,
    enable_thinking: bool = True,
    prev_ended_with_eos: bool = True,
) -> str | None:
    """Render ONLY the continuation suffix (e.g. `<tool_response>` or next `user` turn + generation prompt)
    using the model's own `chat_template`, so parked multi-turn KV caches can be reused without
    re-prefilling the entire conversation history (even when `<think>` from the prior turn was stripped).
    """
    if not new_messages or not getattr(tokenizer, "chat_template", None):
        return None
    marker_a = "__QUINT_INC_ASST_9c2b__"
    norm_new = normalize_messages(new_messages, parse_tool_args_as_dict=True)
    dummy = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": marker_a},
        *norm_new,
    ]
    rendered = _try_apply_template(tokenizer, dummy, tools=None, enable_thinking=enable_thinking)
    if rendered is None:
        dummy_plain = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": marker_a},
            *_flatten_tool_history_for_plain_template(norm_new),
        ]
        rendered = _try_apply_template(tokenizer, dummy_plain, tools=None, enable_thinking=enable_thinking)
    if not rendered or marker_a not in rendered:
        return None
    suffix = rendered[rendered.index(marker_a) + len(marker_a) :]
    if not suffix:
        return None
    if prev_ended_with_eos:
        eos_candidates: list[str] = []
        eos_tok = getattr(tokenizer, "eos_token", None)
        if isinstance(eos_tok, str) and eos_tok:
            eos_candidates.append(eos_tok)
        for common in ("<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>", "</s>"):
            if common not in eos_candidates:
                eos_candidates.append(common)
        for tok_s in eos_candidates:
            if suffix.startswith(tok_s):
                suffix = suffix[len(tok_s) :]
                break
    return suffix or None


def tool_calls_match_signature(
    a_calls: list[dict[str, Any]] | None, b_calls: list[dict[str, Any]] | None
) -> bool:
    """Compare two OpenAI `tool_calls` lists by `(name, canonical_args)` ignoring generated IDs."""
    if not a_calls or not b_calls or len(a_calls) != len(b_calls):
        return False

    def _canon(tc: dict[str, Any]) -> tuple[str, str]:
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        name = str(fn.get("name") or "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                pass
        return name, json.dumps(args, sort_keys=True, ensure_ascii=False)

    return [_canon(x) for x in a_calls] == [_canon(x) for x in b_calls]


# ---------------------------------------------------------------------------
# Universal multi-protocol tool-call parser (all 12+ LLM families)
# ---------------------------------------------------------------------------

def _coerce_xml_param_value(raw_val: str, schema: dict[str, Any] | None = None) -> Any:
    val = raw_val
    if val.startswith("\n"):
        val = val[1:]
    if val.endswith("\n"):
        val = val[:-1]
    stripped = val.strip()
    expected_type = str(schema.get("type", "")).lower() if isinstance(schema, dict) else ""

    if expected_type == "string":
        return val
    if expected_type in ("integer", "number", "boolean", "array", "object") or not expected_type:
        if stripped in ("true", "false", "null"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return val
        if stripped and (
            stripped[0] in ("{", "[")
            or (expected_type in ("integer", "number") and re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", stripped))
            or (not expected_type and re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?", stripped))
        ):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
    return val


def _normalize_call_dict(
    obj: Any,
    *,
    allowed_names: set[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Normalize a single or list of parsed JSON call objects into [(name, arguments_dict), ...]."""
    if isinstance(obj, list):
        out: list[tuple[str, dict[str, Any]]] = []
        for el in obj:
            out.extend(_normalize_call_dict(el, allowed_names=allowed_names))
        return out
    if not isinstance(obj, dict):
        return []

    # Unwrap OpenAI / nested wrapper {"type": "function", "function": {"name": ..., "arguments": ...}}
    if isinstance(obj.get("function"), dict):
        inner = obj["function"]
        name = inner.get("name")
        raw_args = inner.get("arguments", inner.get("parameters", {}))
    else:
        name = obj.get("name") or obj.get("tool_name") or obj.get("function_name") or obj.get("tool")
        if "arguments" in obj:
            raw_args = obj["arguments"]
        elif "parameters" in obj:
            raw_args = obj["parameters"]
        elif "args" in obj:
            raw_args = obj["args"]
        elif "input" in obj and isinstance(obj["input"], dict):
            raw_args = obj["input"]
        else:
            # Flat JSON fallback: {"name": "file_list", "path": "", "target_env": "internal"}
            meta_keys = {"name", "tool_name", "function_name", "tool", "type", "id", "tool_call_id"}
            raw_args = {k: v for k, v in obj.items() if k not in meta_keys}

    if not isinstance(name, str) or not name.strip():
        return []
    clean_name = name.strip()
    if allowed_names and clean_name not in allowed_names:
        # Allow dotted prefix stripping (e.g. "functions.file_list" -> "file_list")
        short = clean_name.rsplit(".", 1)[-1]
        if short in allowed_names:
            clean_name = short
        else:
            return []

    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args) if raw_args.strip() else {}
            args_dict = parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            args_dict = {"raw": raw_args}
    elif isinstance(raw_args, dict):
        args_dict = raw_args
    elif raw_args is None:
        args_dict = {}
    else:
        args_dict = {"value": raw_args}

    return [(clean_name, args_dict)]


def _extract_json_objects(blob: str) -> list[Any]:
    """Extract one or more JSON values (objects or arrays) from a potentially noisy string."""
    dec = json.JSONDecoder()
    results: list[Any] = []
    s = blob.strip()
    # Strip optional markdown code fence
    m_fence = re.match(r"^```(?:json|JSON)?\s*([\s\S]*?)\s*```$", s)
    if m_fence:
        s = m_fence.group(1).strip()

    pos = 0
    n = len(s)
    while pos < n:
        while pos < n and s[pos] not in "{[":
            pos += 1
        if pos >= n:
            break
        try:
            val, end = dec.raw_decode(s, pos)
            results.append(val)
            pos = end
        except json.JSONDecodeError:
            # Try tolerating trailing commas before } or ]
            cleaned = re.sub(r",\s*([}\]])", r"\1", s[pos:])
            try:
                val, end = dec.raw_decode(cleaned, 0)
                results.append(val)
                break
            except json.JSONDecodeError:
                pos += 1
    return results


def _parse_pythonic_calls(
    blob: str,
    *,
    allowed_names: set[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Safely parse LFM2 / Llama Pythonic calls like `[get_weather(city='Paris', days=2)]` via `ast.parse`."""
    s = blob.strip()
    if not s:
        return []
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        try:
            tree = ast.parse(s, mode="exec")
        except SyntaxError:
            return []

    nodes: list[ast.Call] = []

    def _collect(node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            _collect(node.body)
        elif isinstance(node, ast.Module):
            for st in node.body:
                if isinstance(st, ast.Expr):
                    _collect(st.value)
        elif isinstance(node, (ast.List, ast.Tuple)):
            for el in node.elts:
                _collect(el)
        elif isinstance(node, ast.Call):
            nodes.append(node)

    _collect(tree)
    calls: list[tuple[str, dict[str, Any]]] = []
    for call in nodes:
        if isinstance(call.func, ast.Name):
            fname = call.func.id
        elif isinstance(call.func, ast.Attribute):
            fname = call.func.attr
        else:
            continue
        if allowed_names and fname not in allowed_names:
            continue
        args_dict: dict[str, Any] = {}
        ok = True
        for i, pos_arg in enumerate(call.args):
            try:
                args_dict[f"arg{i}"] = ast.literal_eval(pos_arg)
            except Exception:
                ok = False
                break
        for kw in call.keywords:
            if kw.arg is None:
                continue
            try:
                args_dict[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                ok = False
                break
        if ok:
            calls.append((fname, args_dict))
    return calls


# Precompiled patterns for all supported model families
_RE_QWEN35_XML_BLOCK = re.compile(
    r"<tool_call>\s*([\s\S]*?)\s*</tool_call>",
    re.IGNORECASE,
)
_RE_QWEN35_FUNC = re.compile(
    r"<function=([^>\s]+)>([\s\S]*?)</function>",
    re.IGNORECASE,
)
_RE_QWEN35_PARAM = re.compile(
    r"<parameter=([^>\s]+)>([\s\S]*?)</parameter>",
    re.IGNORECASE,
)
_RE_TAGGED_CALL_BLOCKS = [
    # Qwen 2.5/3/3.5/3.6, SmolLM3, Hermes, Nemotron, Phi-4, Granite, Command-R, InternLM, LFM2
    re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", re.IGNORECASE),
    re.compile(r"<toolcall>\s*([\s\S]*?)\s*</toolcall>", re.IGNORECASE),
    re.compile(r"<\|tool_call\|>\s*([\s\S]*?)\s*<\|/tool_call\|>", re.IGNORECASE),
    re.compile(r"<\|tool_call_start\|>\s*([\s\S]*?)\s*<\|tool_call_end\|>", re.IGNORECASE),
    re.compile(r"<\|START_ACTION\|>\s*([\s\S]*?)\s*<\|END_ACTION\|>", re.IGNORECASE),
    re.compile(r"<\|action_start\|>(?:\s*<\|plugin\|>)?\s*([\s\S]*?)\s*<\|action_end\|>", re.IGNORECASE),
]
# OpenAI GPT-OSS Harmony: <|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{...}<|call|>
_RE_HARMONY_CALL = re.compile(
    r"(?:<\|start\|>assistant)?\s*<\|channel\|>commentary\s+to=(?:functions\.)?([A-Za-z0-9_.\-]+)"
    r"(?:\s*<\|constrain\|>[A-Za-z0-9_\-]+)?\s*<\|message\|>([\s\S]*?)(?:<\|call\|>|<\|return\|>|<\|end\|>|$)",
    re.IGNORECASE,
)
# DeepSeek V3 / R1 tool calls
_RE_DEEPSEEK_SECTION = re.compile(
    r"<[｜|]tool[▁_]calls[▁_]begin[｜|]>([\s\S]*?)<[｜|]tool[▁_]calls[▁_]end[｜|]>",
    re.IGNORECASE,
)
_RE_DEEPSEEK_ITEM = re.compile(
    r"<[｜|]tool[▁_]call[▁_]begin[｜|]>\s*function\s*<[｜|]tool[▁_]sep[｜|]>\s*([^\n<]+?)\s*\n([\s\S]*?)<[｜|]tool[▁_]call[▁_]end[｜|]>",
    re.IGNORECASE,
)
# Mistral [TOOL_CALLS]
_RE_MISTRAL_ARGS = re.compile(
    r"\[TOOL_CALLS\]\s*([A-Za-z0-9_.\-]+)\s*\[ARGS\]\s*(\{[\s\S]*?\})(?=\s*\[TOOL_CALLS\]|\s*$)",
)
_RE_MISTRAL_JSON = re.compile(r"\[TOOL_CALLS\]\s*(\[[\s\S]*?\]|\{[\s\S]*?\})")
# Granite bare <|tool_call|> JSON without closing tag
_RE_GRANITE_OPEN = re.compile(r"<\|tool_call\|>\s*(\[[\s\S]*?\]|\{[\s\S]*?\})")
# Llama 3.x <|python_tag|> ...
_RE_LLAMA_PYTHON_TAG = re.compile(r"<\|python_tag\|>\s*([\s\S]+)$")
# Phi-4 functools[...]
_RE_PHI4_FUNCTOOLS = re.compile(r"functools(\[[\s\S]*?\])")
# Cohere Action: ```json [...] ```
_RE_COHERE_ACTION = re.compile(r"Action:\s*```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
# Fenced JSON block (used only when tools are declared)
_RE_FENCED_JSON = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```")


def _parse_qwen35_xml_inner(
    inner: str,
    *,
    allowed_names: set[str] | None = None,
    param_schemas: dict[str, dict[str, Any]] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    for fm in _RE_QWEN35_FUNC.finditer(inner):
        fname = fm.group(1).strip()
        if allowed_names and fname not in allowed_names:
            short = fname.rsplit(".", 1)[-1]
            if short in allowed_names:
                fname = short
            else:
                continue
        body = fm.group(2)
        f_schema = (param_schemas or {}).get(fname, {})
        args_dict: dict[str, Any] = {}
        for pm in _RE_QWEN35_PARAM.finditer(body):
            pname = pm.group(1).strip()
            pval = _coerce_xml_param_value(pm.group(2), f_schema.get(pname))
            args_dict[pname] = pval
        calls.append((fname, args_dict))
    return calls


def _build_openai_tool_calls(raw_calls: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, (name, args) in enumerate(raw_calls):
        out.append(
            {
                "index": idx,
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        )
    return out


def parse_tool_calls(
    text: str,
    *,
    tools: list[dict[str, Any]] | None = None,
    allow_bare_json: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Extract structured OpenAI `tool_calls` from ANY supported model wire format.

    Returns `(remaining_clean_content, openai_tool_calls)`.
    """
    if not text:
        return "", []

    allowed_names = extract_tool_names(tools) if tools else None
    param_schemas = extract_tool_param_schemas(tools) if tools else None
    found: list[tuple[str, dict[str, Any]]] = []
    spans_to_remove: list[tuple[int, int]] = []

    # 1. Tagged blocks (<tool_call>, <toolcall>, <|tool_call|>...</|/tool_call|>, <|tool_call_start|>, <|START_ACTION|>, <|action_start|>)
    for rx in _RE_TAGGED_CALL_BLOCKS:
        for m in rx.finditer(text):
            inner = m.group(1).strip()
            block_calls: list[tuple[str, dict[str, Any]]] = []
            if "<function=" in inner.lower():
                block_calls = _parse_qwen35_xml_inner(
                    inner, allowed_names=allowed_names, param_schemas=param_schemas
                )
            if not block_calls:
                for obj in _extract_json_objects(inner):
                    block_calls.extend(_normalize_call_dict(obj, allowed_names=allowed_names))
            if not block_calls:
                block_calls = _parse_pythonic_calls(inner, allowed_names=allowed_names)
            if block_calls:
                found.extend(block_calls)
                spans_to_remove.append(m.span())

    # 2. Standalone Qwen 3.5/3.6 <function=...> without outer <tool_call> wrapper
    if not found and "<function=" in text.lower():
        for fm in _RE_QWEN35_FUNC.finditer(text):
            c = _parse_qwen35_xml_inner(fm.group(0), allowed_names=allowed_names, param_schemas=param_schemas)
            if c:
                found.extend(c)
                spans_to_remove.append(fm.span())

    # 3. OpenAI GPT-OSS Harmony (<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{...}<|call|>)
    if not found and "<|channel|>" in text:
        for m in _RE_HARMONY_CALL.finditer(text):
            fname = m.group(1).strip()
            body = m.group(2).strip()
            if allowed_names and fname not in allowed_names:
                continue
            objs = _extract_json_objects(body)
            args_dict = objs[0] if (objs and isinstance(objs[0], dict)) else {}
            found.append((fname, args_dict))
            spans_to_remove.append(m.span())

    # 4. DeepSeek V3 / R1 (<｜tool▁calls▁begin｜>...)
    if not found and ("tool▁call" in text or "tool_call_begin" in text):
        for sm in _RE_DEEPSEEK_SECTION.finditer(text):
            sec_calls: list[tuple[str, dict[str, Any]]] = []
            for im in _RE_DEEPSEEK_ITEM.finditer(sm.group(1)):
                fname = im.group(1).strip()
                if allowed_names and fname not in allowed_names:
                    continue
                objs = _extract_json_objects(im.group(2))
                args_dict = objs[0] if (objs and isinstance(objs[0], dict)) else {}
                sec_calls.append((fname, args_dict))
            if sec_calls:
                found.extend(sec_calls)
                spans_to_remove.append(sm.span())

    # 5. Mistral [TOOL_CALLS] (both v3 JSON array and v11 name[ARGS]{...})
    if not found and "[TOOL_CALLS]" in text:
        for m in _RE_MISTRAL_ARGS.finditer(text):
            fname = m.group(1).strip()
            if allowed_names and fname not in allowed_names:
                continue
            objs = _extract_json_objects(m.group(2))
            args_dict = objs[0] if (objs and isinstance(objs[0], dict)) else {}
            found.append((fname, args_dict))
            spans_to_remove.append(m.span())
        if not found:
            for m in _RE_MISTRAL_JSON.finditer(text):
                block_calls = []
                for obj in _extract_json_objects(m.group(1)):
                    block_calls.extend(_normalize_call_dict(obj, allowed_names=allowed_names))
                if block_calls:
                    found.extend(block_calls)
                    spans_to_remove.append(m.span())

    # 6. Granite open <|tool_call|> [...] (without closing tag)
    if not found and "<|tool_call|>" in text:
        for m in _RE_GRANITE_OPEN.finditer(text):
            block_calls = []
            for obj in _extract_json_objects(m.group(1)):
                block_calls.extend(_normalize_call_dict(obj, allowed_names=allowed_names))
            if block_calls:
                found.extend(block_calls)
                spans_to_remove.append(m.span())

    # 7. Llama 3.x <|python_tag|>
    if not found and "<|python_tag|>" in text:
        m = _RE_LLAMA_PYTHON_TAG.search(text)
        if m:
            inner = m.group(1).strip()
            block_calls = []
            for obj in _extract_json_objects(inner):
                block_calls.extend(_normalize_call_dict(obj, allowed_names=allowed_names))
            if not block_calls:
                block_calls = _parse_pythonic_calls(inner, allowed_names=allowed_names)
            if block_calls:
                found.extend(block_calls)
                spans_to_remove.append(m.span())

    # 8. Phi-4 functools[...]
    if not found and "functools[" in text:
        for m in _RE_PHI4_FUNCTOOLS.finditer(text):
            block_calls = []
            for obj in _extract_json_objects(m.group(1)):
                block_calls.extend(_normalize_call_dict(obj, allowed_names=allowed_names))
            if not block_calls:
                block_calls = _parse_pythonic_calls(m.group(1), allowed_names=allowed_names)
            if block_calls:
                found.extend(block_calls)
                spans_to_remove.append(m.span())

    # 9. Cohere Action: ```json [...] ```
    if not found and "Action:" in text:
        for m in _RE_COHERE_ACTION.finditer(text):
            block_calls = []
            for obj in _extract_json_objects(m.group(1)):
                block_calls.extend(_normalize_call_dict(obj, allowed_names=allowed_names))
            if block_calls:
                found.extend(block_calls)
                spans_to_remove.append(m.span())

    # 10. Bare JSON or fenced ```json``` (ONLY when tools were explicitly provided or allow_bare_json=True)
    if not found and (allow_bare_json or (allowed_names and len(allowed_names) > 0)):
        # Fenced JSON blocks first
        for m in _RE_FENCED_JSON.finditer(text):
            block_calls = []
            for obj in _extract_json_objects(m.group(1)):
                block_calls.extend(_normalize_call_dict(obj, allowed_names=allowed_names))
            if block_calls:
                found.extend(block_calls)
                spans_to_remove.append(m.span())
        # Bare top-level JSON object/array or Llama 3.2/4 pythonic `[fn(k=v)]`
        if not found:
            stripped = text.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                block_calls = []
                for obj in _extract_json_objects(stripped):
                    block_calls.extend(_normalize_call_dict(obj, allowed_names=allowed_names))
                if not block_calls and stripped.startswith("[") and allowed_names:
                    block_calls = _parse_pythonic_calls(stripped, allowed_names=allowed_names)
                if block_calls:
                    found.extend(block_calls)
                    spans_to_remove.append((0, len(text)))

    if not found:
        return text, []

    # Remove matched spans cleanly from prose content
    spans_to_remove.sort()
    kept_parts: list[str] = []
    cursor = 0
    for s_idx, e_idx in spans_to_remove:
        if s_idx > cursor:
            kept_parts.append(text[cursor:s_idx])
        cursor = max(cursor, e_idx)
    if cursor < len(text):
        kept_parts.append(text[cursor:])
    clean_text = "".join(kept_parts).strip()

    # Strip leftover Harmony final-channel framing if present
    clean_text = re.sub(r"<\|channel\|>final\s*<\|message\|>", "", clean_text).strip()
    return clean_text, _build_openai_tool_calls(found)


def normalize_inline_tool_calls(text: str) -> str:
    """For prompt-only clients (no `tools` field in request) that parse `<tool_call>` from content:

    Rewrites Qwen 3.5 `<function=...><parameter=...>` XML or flat JSON `{"name": "fn", "k": "v"}`
    inside `<tool_call>...</tool_call>` into canonical `{"name": "fn", "arguments": {"k": "v"}}`,
    while leaving already-canonical JSON blocks untouched.
    """
    if not text or "<tool_call>" not in text.lower():
        return text

    def _repl(m: re.Match[str]) -> str:
        inner = m.group(1).strip()
        if not inner:
            return m.group(0)
        # Already canonical JSON with explicit "name" and "arguments" dict -> keep verbatim
        if inner.startswith("{") and "<function=" not in inner.lower():
            try:
                obj = json.loads(inner)
                if isinstance(obj, dict) and isinstance(obj.get("name"), str) and isinstance(obj.get("arguments"), dict):
                    return m.group(0)
            except json.JSONDecodeError:
                pass
        _, parsed = parse_tool_calls(m.group(0), tools=None, allow_bare_json=False)
        if not parsed:
            return m.group(0)
        blocks = []
        for tc in parsed:
            fn = tc["function"]
            try:
                args_obj = json.loads(fn["arguments"])
            except json.JSONDecodeError:
                args_obj = {}
            blocks.append(
                "<tool_call>\n"
                + json.dumps({"name": fn["name"], "arguments": args_obj}, ensure_ascii=False)
                + "\n</tool_call>"
            )
        return "\n".join(blocks)

    return _RE_QWEN35_XML_BLOCK.sub(_repl, text)


# ---------------------------------------------------------------------------
# Streaming guard for SSE responses (zero-latency prose, zero raw-tag leaks)
# ---------------------------------------------------------------------------

_STREAM_START_TRIGGERS = (
    "<tool_call>",
    "<toolcall>",
    "<|tool_call|>",
    "<|tool_call_start|>",
    "<|START_ACTION|>",
    "<|action_start|>",
    "<|python_tag|>",
    "<|channel|>commentary",
    "[TOOL_CALLS]",
    "<｜tool▁calls▁begin｜>",
    "<|tool_calls_begin|>",
    "<function=",
    "functools[",
    "Action:",
)
_MAX_TRIGGER_LEN = max(len(t) for t in _STREAM_START_TRIGGERS)


class StreamingToolCallGuard:
    """Incrementally separates normal answer prose from tool-call markup during SSE streaming.

    - When `structured_mode=True` (client passed OpenAI `tools`):
      streams prose before any tool-call trigger immediately, buffers once a trigger (or
      leading bare JSON `{`/`[`/` ``` `) appears, and at `finish()` emits parsed OpenAI
      `tool_calls` (or flushes plain text if no valid tool call was formed).
    - When `structured_mode=False` (client did NOT pass `tools`, e.g. system-prompt tool mode):
      streams all normal text immediately, only briefly buffering `<tool_call>...</tool_call>`
      blocks so XML / flat-JSON tool calls can be normalized cleanly before sending to the client.
    """

    def __init__(
        self,
        *,
        tools: list[dict[str, Any]] | None = None,
        structured_mode: bool = False,
    ) -> None:
        self.tools = tools
        self.structured_mode = bool(structured_mode and tools)
        self._buf = ""
        self._buffering_call = False
        self._emitted_any_prose = False

    def push(self, chunk_text: str) -> list[str]:
        if not chunk_text:
            return []
        self._buf += chunk_text

        if not self.structured_mode:
            # Prompt-only mode: only buffer inside `<tool_call>...</tool_call>` so we can normalize XML/flat JSON
            out: list[str] = []
            while self._buf:
                low = self._buf.lower()
                if not self._buffering_call:
                    idx = low.find("<tool_call>")
                    if idx != -1:
                        if idx > 0:
                            out.append(self._buf[:idx])
                            self._emitted_any_prose = True
                        self._buf = self._buf[idx:]
                        self._buffering_call = True
                        continue
                    # Hold only a tiny tail if it could be the start of "<tool_call>"
                    keep = 0
                    prefix = "<tool_call>"
                    max_k = min(len(self._buf), len(prefix) - 1)
                    for k in range(max_k, 0, -1):
                        if low.endswith(prefix[:k]):
                            keep = k
                            break
                    emit_len = len(self._buf) - keep
                    if emit_len > 0:
                        out.append(self._buf[:emit_len])
                        self._emitted_any_prose = True
                        self._buf = self._buf[emit_len:]
                    break
                else:
                    end_idx = low.find("</tool_call>")
                    if end_idx == -1:
                        break
                    close_end = end_idx + len("</tool_call>")
                    block = self._buf[:close_end]
                    self._buf = self._buf[close_end:]
                    self._buffering_call = False
                    out.append(normalize_inline_tool_calls(block))
            return out

        # Structured OpenAI tools mode:
        if self._buffering_call:
            return []

        # If nothing emitted yet and text starts with whitespace + `{` / `[` / ```` (Llama 3.2 bare JSON), buffer it
        if not self._emitted_any_prose:
            s_lead = self._buf.lstrip()
            if not s_lead:
                return []
            if s_lead[0] in ("{", "[", "`"):
                self._buffering_call = True
                return []

        # Check for any explicit tool-call trigger
        low = self._buf.lower()
        earliest = -1
        for trig in _STREAM_START_TRIGGERS:
            pos = low.find(trig.lower())
            if pos != -1 and (earliest == -1 or pos < earliest):
                earliest = pos
        if earliest != -1:
            emit: list[str] = []
            if earliest > 0:
                emit.append(self._buf[:earliest])
                self._emitted_any_prose = True
            self._buf = self._buf[earliest:]
            self._buffering_call = True
            return emit

        # Hold back only the shortest possible suffix that matches a prefix of any trigger
        keep = 0
        max_k = min(len(self._buf), _MAX_TRIGGER_LEN - 1)
        for k in range(max_k, 0, -1):
            tail = low[-k:]
            if any(t.lower().startswith(tail) for t in _STREAM_START_TRIGGERS):
                keep = k
                break
        emit_len = len(self._buf) - keep
        if emit_len > 0:
            piece = self._buf[:emit_len]
            self._buf = self._buf[emit_len:]
            self._emitted_any_prose = True
            return [piece]
        return []

    def finish(self) -> tuple[str, list[dict[str, Any]]]:
        rem = self._buf
        self._buf = ""
        self._buffering_call = False
        if not rem:
            return "", []
        if not self.structured_mode:
            return normalize_inline_tool_calls(rem), []
        clean_text, calls = parse_tool_calls(rem, tools=self.tools, allow_bare_json=True)
        if calls:
            return clean_text, calls
        return rem, []
