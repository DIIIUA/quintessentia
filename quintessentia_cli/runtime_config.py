"""Lightweight CLI contracts; listing/help must not import Torch or start CUDA."""
import ctypes
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VRAM_MODES = ("normal", "turbo")
_MIB = 1024 ** 2
_GIB = 1024 ** 3


def find_quintessentia_root(explicit_root=None):
    candidates = [Path(explicit_root)] if explicit_root else []
    if os.environ.get("QUINTESSENTIA_ROOT"):
        candidates.append(Path(os.environ["QUINTESSENTIA_ROOT"]))
    candidates.append(PROJECT_ROOT)
    for candidate in candidates:
        if (candidate/"src"/"quintessentia_integrated.py").exists():
            resolved = candidate.resolve()
            if str(resolved) not in sys.path:
                sys.path.insert(0, str(resolved))
            return resolved
    raise FileNotFoundError("Could not locate the Quintessentia project root. Pass --quintessentia-root or set QUINTESSENTIA_ROOT.")


def checkpoint_manifest_path(root):
    return root/"data"/"trusted-checkpoint-manifest.json"


class ContextBudgetExceeded(ValueError):
    """Request exceeds the context capacity reserved by the current VRAM plan."""


class GenerationCancelled(RuntimeError):
    _is_cooperative_cancel = True

    def __init__(self, phase, reason="cancelled"):
        super().__init__(f"generation cancelled during {phase}: {reason}")
        self.phase, self.reason = phase, reason


def model_context_limit(engine):
    """Architectural limit only; the KV planning reservation is not a request ceiling."""
    config = getattr(engine, "config", None)
    for cfg in (getattr(config, "text_config", None), config):
        for name in ("max_position_embeddings", "n_positions", "seq_length", "max_sequence_length", "max_seq_len"):
            value = cfg.get(name) if isinstance(cfg, dict) else getattr(cfg, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return None


def _extract_text_config_dict(obj: Any) -> dict[str, Any]:
    """Return the model's text config as a plain dict without importing torch/transformers."""
    cfg = getattr(obj, "config", None)
    if cfg is None:
        return {}
    if hasattr(cfg, "get_text_config") and callable(cfg.get_text_config):
        try:
            cfg = cfg.get_text_config()
        except Exception:
            pass
    elif getattr(cfg, "text_config", None) is not None:
        cfg = getattr(cfg, "text_config")
    elif isinstance(cfg, dict) and isinstance(cfg.get("text_config"), dict):
        cfg = cfg["text_config"]
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "to_dict") and callable(cfg.to_dict):
        try:
            d = cfg.to_dict()
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    if hasattr(cfg, "__dict__"):
        return dict(vars(cfg))
    return {}


def _kv_geometry_from_config_dict(cfg: dict[str, Any]) -> tuple[int | None, int, int]:
    """Compute `(kv_bytes_per_token, recurrent_state_bytes, hidden_size)` from a HF-style config dict.

    Returns `(None, 0, 0)` when the config lacks layer/hidden geometry (e.g. minimal mocks).
    Returns `(0, rec_bytes, hidden)` for pure recurrent SSM/RWKV architectures with zero KV growth.
    """
    if not cfg:
        return None, 0, 0
    mtype = str(cfg.get("model_type") or "").lower()
    raw_layers = (
        cfg.get("num_hidden_layers")
        or cfg.get("n_layers")
        or cfg.get("n_layer")
        or cfg.get("num_layers")
    )
    raw_hidden = cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model")
    if not raw_layers or not raw_hidden:
        return None, 0, 0

    n = int(raw_layers)
    hidden = int(raw_hidden)
    n_heads = int(cfg.get("num_attention_heads") or cfg.get("n_head") or cfg.get("n_heads") or 0)
    if mtype in ("mamba", "mamba2", "falcon_mamba", "rwkv"):
        full = 0
        lin = n
    else:
        layer_types = cfg.get("layer_types") or []
        if layer_types:
            full = sum(1 for t in layer_types if t in ("full_attention", "attention", "sliding_attention"))
            lin = max(0, n - full)
        else:
            if n_heads <= 0 and not cfg.get("head_dim"):
                return None, 0, hidden
            full = n
            lin = 0

    if full > 0:
        eff_heads = max(1, n_heads)
        n_kv = int(
            cfg.get("num_key_value_heads")
            or cfg.get("num_kv_heads")
            or (1 if cfg.get("multi_query") else eff_heads)
        )
        head = int(cfg.get("head_dim") or max(1, hidden // eff_heads))
        if mtype == "minicpm3" and cfg.get("kv_lora_rank"):
            kv = full * (int(cfg["kv_lora_rank"]) + int(cfg.get("qk_rope_head_dim") or 0)) * 2
        elif cfg.get("qk_nope_head_dim") is not None or cfg.get("v_head_dim") is not None:
            qk_d = int(cfg.get("qk_nope_head_dim") or head) + int(cfg.get("qk_rope_head_dim") or 0)
            v_d = int(cfg.get("v_head_dim") or head)
            kv = full * n_kv * (qk_d + v_d) * 2
        else:
            kv = full * 2 * n_kv * head * 2
    else:
        kv = 0

    rec = 0
    if lin and cfg.get("linear_num_value_heads"):
        rec = lin * (
            int(cfg.get("linear_num_value_heads", 32))
            * int(cfg.get("linear_key_head_dim", 128))
            * int(cfg.get("linear_value_head_dim", 128))
            * 4
            + 2 * int(cfg.get("linear_num_key_heads", 16)) * int(cfg.get("linear_key_head_dim", 128)) * 4 * 2
        )
    elif lin and (cfg.get("state_size") or cfg.get("mamba_d_state") or cfg.get("ssm_state_size")):
        d_inner = int(cfg.get("intermediate_size") or hidden * int(cfg.get("expand") or 2))
        d_state = int(cfg.get("state_size") or cfg.get("mamba_d_state") or cfg.get("ssm_state_size") or 16)
        d_conv = int(cfg.get("conv_kernel") or cfg.get("mamba_d_conv") or 4)
        rec = lin * (d_inner * d_state * 4 + d_inner * d_conv * 2)
    elif mtype == "rwkv":
        rec = n * hidden * 5 * 4

    return kv, rec, hidden


def _available_host_ram_bytes() -> int:
    try:
        if os.name == "nt":
            class _MemStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            st = _MemStatusEx()
            st.dwLength = ctypes.sizeof(_MemStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullAvailPhys)
        else:
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def _read_generation_max_new_tokens(wrapper: Any, inner: Any) -> int | None:
    for obj in (inner, wrapper):
        gcfg = getattr(obj, "generation_config", None)
        val = getattr(gcfg, "max_new_tokens", None) if gcfg is not None else None
        if isinstance(val, int) and not isinstance(val, bool) and val >= 256:
            return val
    model_dir = getattr(wrapper, "model_dir", None) or getattr(inner, "model_dir", None)
    if model_dir:
        gpath = Path(model_dir) / "generation_config.json"
        if gpath.is_file():
            try:
                data = json.loads(gpath.read_text(encoding="utf-8"))
                val = data.get("max_new_tokens")
                if isinstance(val, int) and not isinstance(val, bool) and val >= 256:
                    return val
            except Exception:
                pass
    return None


def _round_down_tokens(tokens: int) -> int:
    if tokens >= 16384:
        return (tokens // 4096) * 4096
    if tokens >= 2048:
        return (tokens // 1024) * 1024
    if tokens >= 256:
        return (tokens // 256) * 256
    return max(1, tokens)


def _query_physical_avail_vram(device: Any, plan: Any) -> tuple[int | None, int]:
    """Conservative physical free VRAM using min(NVML, CUDA) capped by `total - allocated - os_ctx`.

    Prevents WSL2/WDDM shared-memory overcommit in `torch.cuda.mem_get_info` from reporting
    phantom free VRAM above the physical GPU size.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None, 0
        cuda_free, cuda_tot = torch.cuda.mem_get_info(device)
        free_b, tot_b = int(cuda_free), int(cuda_tot)
        try:
            from .gpu_memory import measure_gpu_memory
            probe = measure_gpu_memory(device)
            p_free = int(probe.get("free_bytes", 0) or 0)
            p_tot = int(probe.get("total_bytes", 0) or 0)
            if p_free > 0:
                free_b = min(free_b, p_free)
            if p_tot > 0:
                tot_b = min(tot_b, p_tot)
        except Exception:
            pass
        reserved_b = int(torch.cuda.memory_reserved(device))
        allocated_b = int(torch.cuda.memory_allocated(device))
        planned_weights = int(getattr(plan, "weights_bytes", 0) or 0) if plan is not None else 0
        pending_weights = max(0, planned_weights - allocated_b) if planned_weights > 0 else 0
        os_ctx_floor = (768 * _MIB) if tot_b >= 6 * _GIB else (384 * _MIB)
        phys_cap = max(0, tot_b - allocated_b - pending_weights - os_ctx_floor)
        avail_vram = min(
            phys_cap,
            max(0, free_b + max(0, reserved_b - allocated_b) - pending_weights),
        )
        return avail_vram, tot_b
    except Exception:
        return None, 0


def compute_runtime_limits(wrapper: Any, *, busy: bool = False) -> dict[str, Any]:
    """Automatically compute hardware-aware context, max output, and max single-request input limits.

    - Uses the loaded model's native context + KV-bytes-per-token geometry + live available GPU VRAM
      (or Host RAM on CPU) after weights are loaded.
    - Purely advisory (`context_limit_enforced: False`) so explicit caller `max_tokens` is never silently capped,
      while OpenAI / LM Studio / llama.cpp / KoboldCpp / Ollama clients automatically detect real capacity.
    """
    inner = getattr(wrapper, "engine", wrapper)
    native_ctx = model_context_limit(inner)
    if native_ctx is None and inner is not wrapper:
        native_ctx = model_context_limit(wrapper)

    cfg = _extract_text_config_dict(inner) or _extract_text_config_dict(wrapper)
    arch = str(cfg.get("model_type") or "llm").lower()
    kv_per_tok, rec_bytes, hidden_size = _kv_geometry_from_config_dict(cfg)

    plan = getattr(wrapper, "plan", None)
    plan_ctx = int(getattr(plan, "max_context_tokens", 0) or 0) if plan is not None else 0
    prefill_chunk = int(
        getattr(wrapper, "prefill_chunk_tokens", None)
        or (getattr(plan, "prefill_chunk_tokens", None) if plan is not None else None)
        or 512
    )
    vram_mode = str(
        getattr(wrapper, "vram_mode", None)
        or (getattr(plan, "mode", None) if plan is not None else None)
        or "normal"
    ).lower()

    device = getattr(wrapper, "device", None) or getattr(inner, "device", None)
    if callable(getattr(wrapper, "_gpu_metric_device", None)):
        try:
            mdev = wrapper._gpu_metric_device()
            if mdev is not None:
                device = mdev
        except Exception:
            pass
    dev_type = str(getattr(device, "type", device or "")).lower()

    hw_ctx_raw: int | None = None
    fast_vram_ctx_raw: int | None = None
    avail_kv_bytes: int | None = None
    source = "unknown"
    cal = getattr(wrapper, "calibration", None)
    is_calibrated = isinstance(cal, dict) and bool(cal.get("ok"))

    if kv_per_tok == 0:
        hw_ctx_raw = native_ctx or 131072
        fast_vram_ctx_raw = hw_ctx_raw
        source = "recurrent_constant_kv"
    elif kv_per_tok is not None and kv_per_tok > 0 and dev_type.startswith("cuda"):
        avail_vram: int | None = None
        total_vram = int(getattr(plan, "total_bytes", 0) or 0) if plan is not None else 0
        cached_idle = getattr(wrapper, "_idle_free_vram_bytes", None)
        if busy and isinstance(cached_idle, int) and cached_idle > 0:
            avail_vram = cached_idle
        else:
            q_avail, q_tot = _query_physical_avail_vram(device, plan)
            if q_avail is not None:
                avail_vram = q_avail
                total_vram = max(total_vram, q_tot)
                try:
                    setattr(wrapper, "_idle_free_vram_bytes", avail_vram)
                except Exception:
                    pass
        if avail_vram is None and plan is not None and int(getattr(plan, "free_bytes", 0) or 0) > 0:
            avail_vram = max(0, int(plan.free_bytes) - int(getattr(plan, "weights_bytes", 0) or 0))
            total_vram = max(total_vram, int(getattr(plan, "total_bytes", 0) or 0))

        if avail_vram is not None:
            # Prefer live-calibrated chunk activation peak when available; otherwise conservative estimate.
            if is_calibrated and isinstance(cal.get("chunk_workspace_mib"), (int, float)) and cal["chunk_workspace_mib"] > 0:
                prefill_ws = max(64 * _MIB, int(float(cal["chunk_workspace_mib"]) * 1.25 * _MIB))
            else:
                prefill_ws = max(160 * _MIB, min(256 * _MIB, prefill_chunk * max(hidden_size, 1024) * 16))
            if total_vram >= 12 * _GIB:
                base_guard = max(int(total_vram * 0.04), 384 * _MIB)
            elif total_vram >= 6 * _GIB:
                base_guard = max(int(total_vram * 0.05), 352 * _MIB)
            else:
                base_guard = max(int(total_vram * 0.05), 192 * _MIB)
            os_guard = int(base_guard * 0.7) if vram_mode == "turbo" else base_guard
            avail_kv_bytes = max(0, avail_vram - rec_bytes - prefill_ws - os_guard)
            # PreallocatedGPUDynamicLayer needs ~1.06x memory (zero torch.cat 2x growth).
            fast_vram_ctx_raw = int(avail_kv_bytes // max(1, int(kv_per_tok * 1.06)))
            hw_ctx_raw = fast_vram_ctx_raw
            source = "auto_vram"

            # When automatic layer-wise pinned-RAM KV offloading is enabled, only 2 active attention
            # layers (current + async prefetch) need VRAM at once; total KV lives in Host RAM.
            kv_offload_policy = os.environ.get("QUINTESSENTIA_KV_OFFLOAD", "auto").strip().lower()
            if getattr(wrapper, "kv_offload_enabled", False) and kv_offload_policy not in ("0", "false", "never", "off"):
                layer_types = cfg.get("layer_types") or []
                if layer_types:
                    n_full = sum(1 for t in layer_types if t in ("full_attention", "attention"))
                else:
                    n_full = int(
                        cfg.get("num_hidden_layers")
                        or cfg.get("n_layers")
                        or cfg.get("n_layer")
                        or cfg.get("num_layers")
                        or 1
                    )
                n_full = max(1, n_full)
                avail_ram = _available_host_ram_bytes()
                if n_full > 2 and avail_ram > 0:
                    ram_kv_budget = max(0, int(avail_ram * 0.35))
                    ram_ctx = ram_kv_budget // kv_per_tok
                    per_layer_kv = max(1, kv_per_tok // n_full)
                    vram_two_layer_ctx = avail_kv_bytes // (2 * per_layer_kv)
                    offload_ctx = min(ram_ctx, vram_two_layer_ctx)
                    if offload_ctx > hw_ctx_raw:
                        hw_ctx_raw = offload_ctx
                        source = "auto_vram_offload"
    elif kv_per_tok is not None and kv_per_tok > 0 and dev_type == "cpu":
        avail_ram = _available_host_ram_bytes()
        if avail_ram > 0:
            avail_kv_bytes = max(0, int(avail_ram * 0.35) - rec_bytes)
            hw_ctx_raw = avail_kv_bytes // kv_per_tok
            source = "auto_ram"

    eff_ctx: int | None
    if hw_ctx_raw is not None:
        floor_ctx = max(2048, plan_ctx)
        if native_ctx is not None:
            floor_ctx = min(native_ctx, floor_ctx)
        if native_ctx is not None and hw_ctx_raw >= native_ctx:
            eff_ctx = native_ctx
            if source == "auto_vram":
                source = "native_fits_vram"
            elif source == "auto_ram":
                source = "native_fits_ram"
        else:
            rounded = _round_down_tokens(max(floor_ctx, hw_ctx_raw))
            eff_ctx = min(native_ctx, rounded) if native_ctx is not None else rounded
    elif native_ctx is not None:
        eff_ctx = native_ctx
        source = "native_config"
    elif plan_ctx > 0:
        eff_ctx = plan_ctx
        source = "vram_plan"
    else:
        eff_ctx = None

    max_output: int | None = None
    max_input: int | None = None
    if eff_ctx is not None:
        gen_max = _read_generation_max_new_tokens(wrapper, inner)
        if gen_max is not None:
            max_output = min(eff_ctx, gen_max)
        else:
            raw_out = min(32768, max(2048, eff_ctx // 2))
            if raw_out >= eff_ctx:
                max_output = max(256, eff_ctx // 2) if eff_ctx >= 512 else eff_ctx
            else:
                max_output = _round_down_tokens(raw_out)

        reply_reserve = (eff_ctx // 2) if eff_ctx <= 1024 else min(4096, max(512, eff_ctx // 8))
        raw_in = max(256, eff_ctx - reply_reserve)
        if raw_in >= 2048:
            max_input = (raw_in // 1024) * 1024
        elif raw_in >= 256:
            max_input = (raw_in // 256) * 256
        else:
            max_input = raw_in
        max_input = min(eff_ctx, max_input)

    fast_vram_ctx: int | None = None
    if fast_vram_ctx_raw is not None:
        if native_ctx is not None and fast_vram_ctx_raw >= native_ctx:
            fast_vram_ctx = native_ctx
        elif fast_vram_ctx_raw > 0:
            r_fast = _round_down_tokens(fast_vram_ctx_raw)
            fast_vram_ctx = min(native_ctx, r_fast) if native_ctx is not None else r_fast

    return {
        "effective_context_length": eff_ctx,
        "fast_vram_context_length": fast_vram_ctx,
        "native_context_length": native_ctx,
        "max_output_tokens": max_output,
        "max_input_tokens": max_input,
        "prefill_chunk_tokens": prefill_chunk,
        "kv_bytes_per_token": kv_per_tok,
        "recurrent_state_mib": round(rec_bytes / _MIB, 2) if rec_bytes else 0.0,
        "available_kv_mib": round(avail_kv_bytes / _MIB, 1) if avail_kv_bytes is not None else None,
        "architecture": arch,
        "context_limit_source": source,
        "context_limit_enforced": False,
        "calibrated": is_calibrated,
    }
