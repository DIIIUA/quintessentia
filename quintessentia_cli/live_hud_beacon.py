"""Zero-overhead asynchronous live telemetry beacon for the Quintessentia Desktop HUD.

Writes compact JSON snapshots in a background daemon thread (throttled to ~6-8 Hz) so the
token generation hot path only updates an in-memory dict (~2 us, zero impact on tok/s).
Automatically bridges WSL 2 Linux runs to the host Windows profile `.quintessentia/live_telemetry.json`
when present, and silently disables itself during pytest runs unless explicitly enabled.
"""
from __future__ import annotations

import datetime
import json
import os
import platform
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any


def _detect_is_wsl() -> bool:
    if os.name == "nt":
        return False
    try:
        rel = platform.release().lower()
        if "microsoft" in rel or "wsl" in rel:
            return True
    except Exception:
        pass
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False


def _resolve_telemetry_paths() -> list[Path]:
    explicit = os.environ.get("QUINTESSENTIA_HUD_TELEMETRY", "").strip()
    if explicit:
        return [Path(explicit)]

    paths: list[Path] = []
    local_home = Path.home() / ".quintessentia" / "live_telemetry.json"
    paths.append(local_home)

    if _detect_is_wsl():
        users_root = Path("/mnt/c/Users")
        if users_root.is_dir():
            skip = {"public", "default", "default user", "all users", "defaultapppool", "glcache"}
            win_candidates: list[tuple[int, Path]] = []
            try:
                for entry in sorted(users_root.iterdir()):
                    el = entry.name.lower()
                    if not entry.is_dir() or el in skip or "sandbox" in el:
                        continue
                    has_q = (entry / ".quintessentia").is_dir()
                    has_wslcfg = (entry / ".wslconfig").is_file()
                    has_desk = (entry / "Desktop").is_dir()
                    if has_q or has_wslcfg or has_desk:
                        score = (0 if has_q else 1) * 10 + (0 if has_wslcfg else 1) * 2 + (0 if has_desk else 1)
                        win_candidates.append((score, entry / ".quintessentia" / "live_telemetry.json"))
            except OSError:
                pass
            win_candidates.sort(key=lambda item: (item[0], item[1].as_posix()))
            if win_candidates:
                paths.insert(0, win_candidates[0][1])
    return paths


def _atomic_write_json(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


class LiveTelemetryBeacon:
    """Non-blocking background telemetry publisher attached to `QuintessentiaUnifiedEngine`."""

    def __init__(self, *, enabled: bool | None = None, min_interval_s: float = 0.14) -> None:
        env_flag = os.environ.get("QUINTESSENTIA_LIVE_HUD", "").strip().lower()
        if enabled is None:
            if env_flag in ("0", "false", "off", "no"):
                enabled = False
            elif env_flag in ("1", "true", "on", "yes"):
                enabled = True
            else:
                enabled = "PYTEST_CURRENT_TEST" not in os.environ
        self.enabled = bool(enabled)
        self.min_interval_s = max(0.05, float(min_interval_s))
        self.paths = _resolve_telemetry_paths() if self.enabled else []
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop = False
        self._dirty = False
        self._state: dict[str, Any] = {
            "pid": os.getpid(),
            "env": "WSL2 Ubuntu" if _detect_is_wsl() else ("Windows" if os.name == "nt" else platform.system()),
            "active": False,
            "phase": "IDLE",
        }
        self._tok_times: deque[float] = deque(maxlen=24)
        self._turn_start_t = 0.0
        self._first_tok_t = 0.0
        self._turn_count = 0
        self._peak_tok_s = 0.0
        self._thread: threading.Thread | None = None
        if self.enabled and self.paths:
            self._thread = threading.Thread(
                target=self._writer_loop, name="quintessentia-hud-beacon", daemon=True
            )
            self._thread.start()

    def _writer_loop(self) -> None:
        last_write = 0.0
        while not self._stop:
            self._event.wait(timeout=0.5)
            self._event.clear()
            if self._stop:
                break
            now = time.monotonic()
            wait_rem = self.min_interval_s - (now - last_write)
            if wait_rem > 0.005:
                time.sleep(min(wait_rem, self.min_interval_s))
            self._flush_once()
            last_write = time.monotonic()

    def _flush_once(self) -> None:
        if not self.enabled or not self.paths:
            return
        with self._lock:
            if not self._dirty:
                return
            snap = dict(self._state)
            self._dirty = False
        snap["updated_ts"] = time.time()
        snap["updated_at"] = datetime.datetime.now().strftime("%H:%M:%S")
        payload = json.dumps(snap, ensure_ascii=False)
        for p in self.paths:
            _atomic_write_json(p, payload)

    def publish_engine_ready(self, engine: Any) -> None:
        if not self.enabled:
            return
        # Protect Anton's live Desktop HUD from ad-hoc synthetic test models created in system temp dirs
        # unless QUINTESSENTIA_LIVE_HUD is explicitly forced to 1/true.
        env_flag = os.environ.get("QUINTESSENTIA_LIVE_HUD", "").strip().lower()
        if env_flag not in ("1", "true", "on", "yes"):
            raw_dir = str(getattr(engine, "model_dir", "") or getattr(engine, "model_id", "") or "")
            if raw_dir:
                try:
                    tmp_root = os.path.normcase(os.path.realpath(tempfile.gettempdir()))
                    cand_norm = os.path.normcase(os.path.realpath(raw_dir))
                    if cand_norm == tmp_root or cand_norm.startswith(tmp_root + os.sep) or raw_dir.startswith("/tmp/"):
                        self.enabled = False
                        self.paths = []
                        return
                except Exception:
                    pass
        prof = getattr(engine, "profile", None) or {}
        plan = getattr(engine, "plan", None)
        runner = getattr(engine, "_q1_cubes_runner", None)
        rstats = {}
        try:
            if runner is not None and hasattr(runner, "stats"):
                rstats = runner.stats() or {}
        except Exception:
            rstats = {}
        tm = rstats.get("trunk_mirror_v4") or {}
        layers_total = int(getattr(plan, "layers_total", 0) or prof.get("num_hidden_layers") or 0)
        full_res = bool(getattr(plan, "full_residency", False))
        resident = layers_total if full_res else int(getattr(plan, "resident_layers", 0) or 0)
        if runner is not None and hasattr(runner, "resident_layers"):
            resident = int(runner.resident_layers)
            full_res = layers_total > 0 and resident >= layers_total

        self._peak_tok_s = 0.0
        with self._lock:
            self._state.update(
                {
                    "active": True,
                    "phase": "READY",
                    "model": str(getattr(engine, "model_id", "Quintessentia")),
                    "model_type": str(prof.get("model_type") or ""),
                    "format": str(getattr(engine, "model_format", "") or prof.get("checkpoint_format") or ""),
                    "backend": str(prof.get("backend") or ""),
                    "weight_format": str(prof.get("weight_format") or ""),
                    "feedback_mode": str(getattr(engine, "q1_graphify_feedback", "") or "off"),
                    "q1_planes": int(getattr(engine, "q1_planes", prof.get("q1_planes", 0)) or 0),
                    "q1_planes_requested": int(
                        getattr(engine, "q1_planes_requested", prof.get("q1_planes_requested", 0)) or 0
                    ),
                    "q1_planes_auto_halved": bool(prof.get("q1_planes_auto_halved", False)),
                    "lm_head_planes": int(prof.get("q1_lm_head_planes") or 0),
                    "root_bits": int(getattr(engine, "cube_root_bits", 8) or 8),
                    "layers_total": layers_total,
                    "resident_layers": resident,
                    "full_residency": full_res,
                    "ring_slots": int(getattr(plan, "ring_slots", 0) or 0),
                    "budget_mib": round(float(getattr(plan, "budget_bytes", 0) or 0) / (1024**2), 1),
                    "weights_mib": round(float(getattr(plan, "weights_bytes", 0) or 0) / (1024**2), 1),
                    "q1_projections": int(tm.get("projections_carved", prof.get("packed_projections", 0)) or 0),
                    "is_moe": bool(int(prof.get("num_experts") or 0) >= 2),
                    "num_experts": int(prof.get("num_experts") or 0),
                    "vram_mode": str(getattr(engine, "vram_mode", "normal")),
                    "turn": self._turn_count,
                    "peak_tok_s": 0.0,
                }
            )
            self._dirty = True
        self._flush_once()

    def begin_turn(self, prompt_tokens: int, cached_prefix: int = 0) -> None:
        if not self.enabled:
            return
        self._turn_count += 1
        self._turn_start_t = time.perf_counter()
        self._first_tok_t = 0.0
        self._tok_times.clear()
        with self._lock:
            self._state.update(
                {
                    "active": True,
                    "phase": "PREFILL",
                    "turn": self._turn_count,
                    "prompt_tokens": int(prompt_tokens),
                    "prefill_done": int(cached_prefix),
                    "prefill_total": int(prompt_tokens),
                    "cached_prefix_tokens": int(cached_prefix),
                    "tokens": 0,
                    "tok_s": 0.0,
                    "ttft_ms": 0.0,
                    "answer_preview": "",
                    "thinking_preview": "",
                }
            )
            self._dirty = True
        self._event.set()

    def on_prefill_progress(self, done_tokens: int, total_tokens: int) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        elapsed = max(1e-4, now - self._turn_start_t) if self._turn_start_t > 0.0 else 0.0
        done_i = max(0, int(done_tokens))
        tot_i = max(1, int(total_tokens))
        cached_i = int(self._state.get("cached_prefix_tokens", 0) or 0)
        newly_done = max(0, done_i - cached_i)
        pp_tok_s = (newly_done / elapsed) if elapsed > 1e-3 else 0.0
        with self._lock:
            self._state.update(
                {
                    "active": True,
                    "phase": "PREFILL",
                    "prefill_done": done_i,
                    "prefill_total": tot_i,
                    "prefill_tok_s": round(pp_tok_s, 1),
                    "tok_s": 0.0,
                }
            )
            self._dirty = True
        self._event.set()

    def on_tokens(
        self,
        total_tokens: int,
        answer_tail: str,
        thinking_tail: str = "",
        phase: str = "GENERATING",
    ) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if self._first_tok_t <= 0.0:
            self._first_tok_t = now
        self._tok_times.append(now)

        # Sliding instant tok/s over recent tokens + average decode tok/s
        inst_tok_s = 0.0
        span = 0.0
        if len(self._tok_times) >= 2:
            span = self._tok_times[-1] - self._tok_times[0]
            if span > 1e-4:
                inst_tok_s = (len(self._tok_times) - 1) / span
        dec_span = now - self._first_tok_t
        avg_tok_s = (max(0, total_tokens - 1) / dec_span) if (total_tokens > 1 and dec_span > 1e-4) else inst_tok_s
        # Filter micro-burst artifacts (< 80 ms window or < 5 tokens) so session peak stays truthful
        if total_tokens >= 5 and span >= 0.08 and inst_tok_s <= 600.0 and inst_tok_s > self._peak_tok_s:
            self._peak_tok_s = inst_tok_s
        elif total_tokens >= 3 and dec_span >= 0.08 and avg_tok_s <= 600.0 and avg_tok_s > self._peak_tok_s:
            self._peak_tok_s = avg_tok_s
        ttft_ms = (self._first_tok_t - self._turn_start_t) * 1000.0 if self._turn_start_t > 0.0 else 0.0

        with self._lock:
            self._state.update(
                {
                    "active": True,
                    "phase": "THINKING" if phase == "thinking" else "GENERATING",
                    "tokens": int(total_tokens),
                    "tok_s": round(inst_tok_s, 2),
                    "avg_tok_s": round(avg_tok_s, 2),
                    "peak_tok_s": round(max(self._peak_tok_s, avg_tok_s), 2),
                    "ttft_ms": round(ttft_ms, 1),
                    "answer_preview": answer_tail[-8000:],
                    "thinking_preview": thinking_tail[-4000:],
                }
            )
            self._dirty = True
        self._event.set()

    def finish_turn(self, engine: Any, metrics: dict[str, Any], answer_text: str, thinking_text: str = "") -> None:
        if not self.enabled:
            return
        tok_s = float(metrics.get("decode_tok_per_s", 0.0) or 0.0)
        gen_toks = int(metrics.get("generated_tokens", 0) or 0)
        if gen_toks >= 3 and tok_s <= 600.0 and tok_s > self._peak_tok_s:
            self._peak_tok_s = tok_s
        runner = getattr(engine, "_q1_cubes_runner", None)
        diff_calls = 0
        q1_projs = int(self._state.get("q1_projections", 0) or 0)
        moe_hits = 0
        moe_host_hits = 0
        moe_misses = 0
        try:
            if runner is not None and hasattr(runner, "stats"):
                rs = runner.stats() or {}
                tm = rs.get("trunk_mirror_v4") or {}
                diff_calls = int(tm.get("diff_q1_calls", 0) or 0)
                q1_projs = int(tm.get("projections_carved", q1_projs) or q1_projs)
                mc = rs.get("moe_cache") or {}
                if isinstance(mc, dict):
                    moe_hits = int(mc.get("hits", 0) or 0)
                    moe_host_hits = int(mc.get("host_hits", 0) or 0)
                    moe_misses = int(mc.get("misses", 0) or 0)
        except Exception:
            pass

        rb = metrics.get("rssf_generation_balance") or {}
        tb = rb.get("turn_end_balance") or {}
        eb = rb.get("entry_balance") or {}
        ec = rb.get("entry_collapse") or {}
        asc = rb.get("active_scales") or {}
        scale_tags: list[str] = []
        _tag_map = (
            ("m0_trunk_q1", "M0"),
            ("m1_trunk_load", "M1"),
            ("m1_moe_load", "M1"),
            ("m2_trunk_pool", "M2"),
            ("m2_moe_pool", "M2"),
            ("m3_cold_warmup", "M3"),
            ("m4_moe_warmup", "M4"),
            ("m6_probability_path", "M6"),
            ("m7_speculation", "M7"),
            ("m8_thought_buffer", "M8"),
            ("kv_placement", "KV"),
        )
        for k_src, lbl in _tag_map:
            if k_src in asc and lbl not in scale_tags:
                scale_tags.append(lbl)

        with self._lock:
            self._state.update(
                {
                    "active": False,
                    "phase": "DONE",
                    "tokens": gen_toks,
                    "tok_s": round(tok_s, 2),
                    "avg_tok_s": round(tok_s, 2),
                    "peak_tok_s": round(self._peak_tok_s, 2),
                    "ttft_ms": round(float(metrics.get("prefill_ms", 0.0) or 0.0), 1),
                    "prefill_ms": round(float(metrics.get("prefill_ms", 0.0) or 0.0), 1),
                    "decode_s": round(float(metrics.get("decode_s", 0.0) or 0.0), 3),
                    "peak_vram_mib": round(float(metrics.get("peak_vram_mib", 0.0) or 0.0), 1),
                    "kv_gpu_layers": int(metrics.get("kv_gpu_layers", 0) or 0),
                    "kv_offloaded_layers": int(metrics.get("kv_offloaded_layers", 0) or 0),
                    "kv_promoted_layers": int(metrics.get("kv_promoted_layers", 0) or 0),
                    "parked_session_ram_mib": round(float(metrics.get("parked_session_ram_mib", 0.0) or 0.0), 1),
                    "rssf_d3_margin": tb.get("d3_margin"),
                    "rssf_regime": str(tb.get("regime") or ""),
                    "rssf_entry_d3": eb.get("d3_margin"),
                    "rssf_damped_chunks": int(rb.get("prefill_damped_chunks", 0) or 0),
                    "rssf_scale_breaks": int(rb.get("prefill_scale_breaks", 0) or 0),
                    "rssf_loss_of_archive": bool(ec.get("loss_of_archive", False)),
                    "rssf_active_scales": scale_tags,
                    "q1_projections": q1_projs,
                    "diff_calls": diff_calls,
                    "moe_hits": moe_hits,
                    "moe_host_hits": moe_host_hits,
                    "moe_misses": moe_misses,
                    "finish_reason": str(metrics.get("finish_reason", "stop")),
                    "answer_preview": answer_text[-8000:],
                    "thinking_preview": thinking_text[-4000:],
                }
            )
            self._dirty = True
        self._flush_once()

    def cancel_turn(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._state.update({"active": False, "phase": "READY", "tok_s": 0.0})
            self._dirty = True
        self._flush_once()

    def close(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._state["active"] = False
            if self._state.get("phase") in ("PREFILL", "GENERATING", "THINKING", "READY", "LOADING"):
                self._state["phase"] = "DONE"
            self._dirty = True
        self._stop = True
        self._event.set()
        self._flush_once()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.3)
