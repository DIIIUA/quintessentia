"""Ultra-lightweight zero-dependency telemetry collector for the Quintessentia Live HUD.

Reads GPU metrics directly from `nvml.dll` via ctypes (~0.05 ms, ZERO subprocess spawning),
Host RAM, Host CPU load (`GetSystemTimes`), and WSL2 `vmmemWSL` RAM/Priority directly via
Win32 ctypes (~0.3 ms), live engine telemetry from `~/.quintessentia/live_telemetry.json`
(written asynchronously by Quintessentia from Windows or WSL2), and optional local LAN
server status (`http://127.0.0.1:8787/api/info`).

Works cleanly on native Windows (with or without NVIDIA GPU, with or without WSL2) as well as
WSL2 Ubuntu bridged to Windows. Runs at ~4 Hz (250 ms) with <0.1% CPU and 0 MiB GPU VRAM;
exits automatically when its parent UI process closes.
"""
from __future__ import annotations

import ctypes
import datetime
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent


def _resolve_cache_dir() -> Path:
    env_cache = os.environ.get("QUINTESSENTIA_HUD_CACHE", "").strip()
    if env_cache:
        p = Path(env_cache)
    elif BASE_DIR.name == "hud" and BASE_DIR.parent.name in ("quintessentia_cli", ".quintessentia"):
        p = Path.home() / ".quintessentia" / "hud_cache"
    else:
        p = BASE_DIR / ".cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


CACHE_DIR = _resolve_cache_dir()
CONFIG_PATH = Path(os.environ.get("QUINTESSENTIA_HUD_CONFIG", str(BASE_DIR / "config.json")))
HUD_OUT = CACHE_DIR / "quintessentia_hud.json"
CLEAR_FLAG = CACHE_DIR / "clear_history.flag"


# ---------------------------------------------------------------------------
# Direct Win32 structures (Host RAM, Host CPU, vmmemWSL WorkingSet & Priority)
# ---------------------------------------------------------------------------
class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("load", ctypes.c_ulong),
        ("total_phys", ctypes.c_ulonglong),
        ("avail_phys", ctypes.c_ulonglong),
        ("total_page", ctypes.c_ulonglong),
        ("avail_page", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("avail_virtual", ctypes.c_ulonglong),
        ("extended", ctypes.c_ulonglong),
    ]


class _FileTime(ctypes.Structure):
    _fields_ = [
        ("dwLowDateTime", ctypes.c_uint32),
        ("dwHighDateTime", ctypes.c_uint32),
    ]


class _WinCpuTracker:
    """Zero-subprocess Win32 host CPU utilization tracker using GetSystemTimes (~0.02 ms)."""

    def __init__(self) -> None:
        self._prev_idle: Optional[int] = None
        self._prev_tot: Optional[int] = None

    @staticmethod
    def _ft_to_int(ft: _FileTime) -> int:
        return (int(ft.dwHighDateTime) << 32) | int(ft.dwLowDateTime)

    def poll_pct(self) -> float:
        if os.name != "nt":
            return 0.0
        try:
            idle, kern, user = _FileTime(), _FileTime(), _FileTime()
            if not ctypes.windll.kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)
            ):
                return 0.0
            i_v = self._ft_to_int(idle)
            t_v = self._ft_to_int(kern) + self._ft_to_int(user)
            if self._prev_idle is None or self._prev_tot is None:
                self._prev_idle, self._prev_tot = i_v, t_v
                return 0.0
            di = max(0, i_v - self._prev_idle)
            dt = max(1, t_v - self._prev_tot)
            self._prev_idle, self._prev_tot = i_v, t_v
            busy = max(0.0, min(100.0, 100.0 * (1.0 - float(di) / float(dt))))
            return round(busy, 1)
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Direct NVML reader (zero subprocess, ~0.05 ms per poll, outlier-filtered)
# ---------------------------------------------------------------------------
class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


# Physical ceiling filter: PCIe 4.0 x16 max bidirectional is ~31.5 GB/s (~32000 MB/s).
# NVML occasionally emits transient counter wrap spikes (>32000 MB/s) on P-state wakeups.
_MAX_VALID_PCIE_MBS = 32000.0


class DirectNVMLReader:
    def __init__(self) -> None:
        self._lib = None
        self._handle = ctypes.c_void_p()
        self.gpu_name = "CPU / No NVIDIA GPU"
        self.power_cap_w = 0.0
        self.ok = False
        self._init()

    def _init(self) -> None:
        candidates = [
            str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvml.dll"),
            "nvml.dll",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll",
        ]
        lib = None
        for path in candidates:
            try:
                lib = ctypes.CDLL(path)
                break
            except OSError:
                continue
        if lib is None:
            return
        try:
            if lib.nvmlInit_v2() != 0:
                return
            get_h = lib.nvmlDeviceGetHandleByIndex_v2
            get_h.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
            get_h.restype = ctypes.c_int
            if get_h(0, ctypes.byref(self._handle)) != 0:
                lib.nvmlShutdown()
                return

            lib.nvmlDeviceGetName.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
            lib.nvmlDeviceGetName.restype = ctypes.c_int
            lib.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NvmlMemory)]
            lib.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
            lib.nvmlDeviceGetUtilizationRates.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NvmlUtilization)]
            lib.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
            lib.nvmlDeviceGetTemperature.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
            lib.nvmlDeviceGetTemperature.restype = ctypes.c_int
            lib.nvmlDeviceGetPowerUsage.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
            lib.nvmlDeviceGetPowerUsage.restype = ctypes.c_int
            lib.nvmlDeviceGetPowerManagementLimit.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
            lib.nvmlDeviceGetPowerManagementLimit.restype = ctypes.c_int
            lib.nvmlDeviceGetClockInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
            lib.nvmlDeviceGetClockInfo.restype = ctypes.c_int
            try:
                lib.nvmlDeviceGetFanSpeed.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
                lib.nvmlDeviceGetFanSpeed.restype = ctypes.c_int
            except AttributeError:
                pass
            try:
                lib.nvmlDeviceGetPcieThroughput.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
                lib.nvmlDeviceGetPcieThroughput.restype = ctypes.c_int
            except AttributeError:
                pass

            name_buf = ctypes.create_string_buffer(96)
            if lib.nvmlDeviceGetName(self._handle, name_buf, 96) == 0:
                raw = name_buf.value.decode("utf-8", errors="ignore").strip()
                self.gpu_name = raw.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")

            lim = ctypes.c_uint(0)
            if lib.nvmlDeviceGetPowerManagementLimit(self._handle, ctypes.byref(lim)) == 0 and lim.value > 0:
                self.power_cap_w = round(lim.value / 1000.0, 1)

            self._lib = lib
            self.ok = True
        except Exception:
            try:
                lib.nvmlShutdown()
            except Exception:
                pass
            self._lib = None
            self.ok = False

    def poll(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.gpu_name,
            "vram_used_mib": 0.0,
            "vram_free_mib": 0.0,
            "vram_total_mib": 0.0,
            "power_w": 0.0,
            "power_cap_w": self.power_cap_w,
            "temp_c": 0.0,
            "util_pct": 0.0,
            "mem_util_pct": 0.0,
            "clock_mhz": 0.0,
            "mem_clock_mhz": 0.0,
            "fan_pct": 0,
            "pcie_tx_mbs": 0.0,
            "pcie_rx_mbs": 0.0,
            "pcie_total_mbs": 0.0,
            "nvml_direct": self.ok,
        }
        if not self.ok or self._lib is None:
            return out

        lib = self._lib
        h = self._handle

        mem = _NvmlMemory()
        if lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(mem)) == 0:
            out["vram_used_mib"] = round(mem.used / (1024**2), 1)
            out["vram_free_mib"] = round(mem.free / (1024**2), 1)
            out["vram_total_mib"] = round(mem.total / (1024**2), 1)

        util = _NvmlUtilization()
        if lib.nvmlDeviceGetUtilizationRates(h, ctypes.byref(util)) == 0:
            out["util_pct"] = float(util.gpu)
            out["mem_util_pct"] = float(util.memory)

        temp = ctypes.c_uint(0)
        if lib.nvmlDeviceGetTemperature(h, 0, ctypes.byref(temp)) == 0:
            out["temp_c"] = float(temp.value)

        pwr = ctypes.c_uint(0)
        if lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(pwr)) == 0:
            out["power_w"] = round(pwr.value / 1000.0, 1)

        gclk = ctypes.c_uint(0)
        if lib.nvmlDeviceGetClockInfo(h, 0, ctypes.byref(gclk)) == 0:
            out["clock_mhz"] = float(gclk.value)

        mclk = ctypes.c_uint(0)
        if lib.nvmlDeviceGetClockInfo(h, 2, ctypes.byref(mclk)) == 0:
            out["mem_clock_mhz"] = float(mclk.value)

        try:
            fan = ctypes.c_uint(0)
            if lib.nvmlDeviceGetFanSpeed(h, ctypes.byref(fan)) == 0:
                out["fan_pct"] = int(fan.value)
        except Exception:
            pass

        try:
            tx = ctypes.c_uint(0)
            rx = ctypes.c_uint(0)
            tx_ok = lib.nvmlDeviceGetPcieThroughput(h, 0, ctypes.byref(tx)) == 0
            rx_ok = lib.nvmlDeviceGetPcieThroughput(h, 1, ctypes.byref(rx)) == 0
            tx_mbs = round(tx.value / 1024.0, 1) if tx_ok else 0.0
            rx_mbs = round(rx.value / 1024.0, 1) if rx_ok else 0.0
            # Drop physically impossible NVML counter wrap glitches
            if tx_mbs < 0.0 or tx_mbs > _MAX_VALID_PCIE_MBS:
                tx_mbs = 0.0
            if rx_mbs < 0.0 or rx_mbs > _MAX_VALID_PCIE_MBS:
                rx_mbs = 0.0
            out["pcie_tx_mbs"] = tx_mbs
            out["pcie_rx_mbs"] = rx_mbs
            out["pcie_total_mbs"] = round(tx_mbs + rx_mbs, 1)
        except Exception:
            pass

        return out

    def close(self) -> None:
        if self.ok and self._lib is not None:
            try:
                self._lib.nvmlShutdown()
            except Exception:
                pass
            self._lib = None
            self.ok = False


def poll_host_and_wsl(cpu_tracker: Optional[_WinCpuTracker] = None) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "ram_used_gib": 0.0,
        "ram_avail_gib": 0.0,
        "ram_total_gib": 0.0,
        "ram_pct": 0.0,
        "cpu_pct": cpu_tracker.poll_pct() if cpu_tracker is not None else 0.0,
        "wsl_running": False,
        "wsl_ram_used_gib": 0.0,
        "wsl_ram_limit_gib": 0.0,
        "wsl_swap_limit_gib": 0.0,
        "wsl_priority": "-",
    }
    if os.name != "nt":
        return info

    k32 = ctypes.windll.kernel32
    ms = _MemoryStatusEx()
    ms.length = ctypes.sizeof(_MemoryStatusEx)
    if k32.GlobalMemoryStatusEx(ctypes.byref(ms)):
        tot = ms.total_phys / (1024**3)
        avail = ms.avail_phys / (1024**3)
        used = max(0.0, tot - avail)
        info["ram_total_gib"] = round(tot, 2)
        info["ram_avail_gib"] = round(avail, 2)
        info["ram_used_gib"] = round(used, 2)
        info["ram_pct"] = round((used / max(0.01, tot)) * 100.0, 1)

    wslcfg = Path.home() / ".wslconfig"
    if wslcfg.is_file():
        try:
            txt = wslcfg.read_text(encoding="utf-8", errors="ignore")
            m_mem = re.search(r"^\s*memory\s*=\s*([0-9.]+)\s*(gb|g|mb|m)?", txt, re.I | re.M)
            if m_mem:
                val = float(m_mem.group(1))
                unit = (m_mem.group(2) or "gb").lower()
                info["wsl_ram_limit_gib"] = round(val / 1024.0 if unit.startswith("m") else val, 2)
            m_sw = re.search(r"^\s*swap\s*=\s*([0-9.]+)\s*(gb|g|mb|m)?", txt, re.I | re.M)
            if m_sw:
                val = float(m_sw.group(1))
                unit = (m_sw.group(2) or "gb").lower()
                info["wsl_swap_limit_gib"] = round(val / 1024.0 if unit.startswith("m") else val, 2)
        except OSError:
            pass

    try:
        ntdll = ctypes.windll.ntdll
        buf_size = 512 * 1024
        ret_len = ctypes.c_ulong(0)
        buf = None
        for _ in range(5):
            buf = (ctypes.c_ubyte * buf_size)()
            status = ntdll.NtQuerySystemInformation(5, ctypes.byref(buf), buf_size, ctypes.byref(ret_len))
            if status == 0:
                break
            if (status & 0xFFFFFFFF) == 0xC0000004:
                buf_size = max(buf_size * 2, ret_len.value + 65536)
            else:
                buf = None
                break
        if buf is not None:
            raw = memoryview(buf)
            off = 0
            while off + 200 <= len(raw):
                next_off = int.from_bytes(raw[off : off + 4], "little")
                img_len = int.from_bytes(raw[off + 56 : off + 58], "little")
                img_ptr = int.from_bytes(raw[off + 64 : off + 72], "little")
                if img_len > 0 and img_ptr:
                    try:
                        name = ctypes.wstring_at(img_ptr, img_len // 2).lower()
                    except Exception:
                        name = ""
                    if name in ("vmmemwsl", "vmmemwsl.exe", "vmmem", "vmmem.exe"):
                        info["wsl_running"] = True
                        base_prio = int.from_bytes(raw[off + 72 : off + 76], "little", signed=True)
                        ws_bytes = int.from_bytes(raw[off + 144 : off + 152], "little")
                        info["wsl_ram_used_gib"] = round(ws_bytes / (1024**3), 2)
                        if base_prio >= 24:
                            info["wsl_priority"] = "Realtime"
                        elif base_prio >= 13:
                            info["wsl_priority"] = "High"
                        elif base_prio >= 10:
                            info["wsl_priority"] = "AboveNorm"
                        elif base_prio >= 8:
                            info["wsl_priority"] = "Normal"
                        elif base_prio >= 6:
                            info["wsl_priority"] = "BelowNorm"
                        else:
                            info["wsl_priority"] = "Idle"
                        break
                if next_off == 0:
                    break
                off += next_off
    except Exception:
        pass
    return info


def read_config() -> Dict[str, Any]:
    default_tf = str(Path.home() / ".quintessentia" / "live_telemetry.json")
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
        if isinstance(cfg, dict):
            cfg.setdefault("telemetry_file", default_tf)
            cfg.setdefault("refresh_ms", 250)
            cfg.setdefault("lan_url", "http://127.0.0.1:8787/api/info")
            return cfg
    except Exception:
        pass
    return {
        "telemetry_file": default_tf,
        "refresh_ms": 250,
        "lan_url": "http://127.0.0.1:8787/api/info",
    }


def _extract_rssf_fields_from_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    rb = metrics.get("rssf_generation_balance") or {}
    tb = rb.get("turn_end_balance") or {}
    eb = rb.get("entry_balance") or {}
    ec = rb.get("entry_collapse") or {}
    asc = rb.get("active_scales") or {}
    tags: List[str] = []
    for k_src, lbl in (
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
    ):
        if k_src in asc and lbl not in tags:
            tags.append(lbl)
    return {
        "kv_gpu_layers": int(metrics.get("kv_gpu_layers", 0) or 0),
        "kv_offloaded_layers": int(metrics.get("kv_offloaded_layers", 0) or 0),
        "kv_promoted_layers": int(metrics.get("kv_promoted_layers", 0) or 0),
        "rssf_d3_margin": tb.get("d3_margin"),
        "rssf_regime": str(tb.get("regime") or ""),
        "rssf_entry_d3": eb.get("d3_margin"),
        "rssf_damped_chunks": int(rb.get("prefill_damped_chunks", 0) or 0),
        "rssf_scale_breaks": int(rb.get("prefill_scale_breaks", 0) or 0),
        "rssf_loss_of_archive": bool(ec.get("loss_of_archive", False)),
        "rssf_active_scales": tags,
    }


def read_live_telemetry(cfg: Dict[str, Any], *, wsl_running: bool = False) -> Dict[str, Any]:
    candidates: List[Path] = []
    tf = str(cfg.get("telemetry_file") or "").strip()
    if tf:
        candidates.append(Path(tf))
    default_p = Path.home() / ".quintessentia" / "live_telemetry.json"
    if default_p not in candidates:
        candidates.append(default_p)
    for extra in cfg.get("extra_telemetry_files") or []:
        p = Path(str(extra))
        if p not in candidates:
            candidates.append(p)

    best_data: Optional[Dict[str, Any]] = None
    best_ts = -1.0
    for path in candidates:
        try:
            if not path.is_file():
                continue
            raw = path.read_text(encoding="utf-8")
            if not raw.strip():
                continue
            data = json.loads(raw)
            if isinstance(data, dict):
                ts = float(data.get("updated_ts", path.stat().st_mtime))
                if ts > best_ts:
                    best_ts = ts
                    best_data = data
        except Exception:
            continue

    now = time.time()
    if best_data is None:
        return {
            "source": "standby",
            "active": False,
            "phase": "IDLE",
            "model": "Quintessentia (standby)",
            "model_type": "",
            "format": "",
            "backend": "unified-q1-cubes",
            "weight_format": "q1_cubes",
            "feedback_mode": "mirror_scales",
            "q1_planes": 5,
            "q1_planes_requested": 5,
            "q1_planes_auto_halved": False,
            "lm_head_planes": 5,
            "root_bits": 8,
            "layers_total": 0,
            "resident_layers": 0,
            "full_residency": True,
            "ring_slots": 0,
            "q1_projections": 0,
            "is_moe": False,
            "num_experts": 0,
            "turn": 0,
            "prompt_tokens": 0,
            "cached_prefix_tokens": 0,
            "tokens": 0,
            "tok_s": 0.0,
            "avg_tok_s": 0.0,
            "peak_tok_s": 0.0,
            "ttft_ms": 0.0,
            "prefill_ms": 0.0,
            "decode_s": 0.0,
            "peak_vram_mib": 0.0,
            "kv_gpu_layers": 0,
            "kv_offloaded_layers": 0,
            "kv_promoted_layers": 0,
            "rssf_d3_margin": None,
            "rssf_regime": "",
            "rssf_entry_d3": None,
            "rssf_damped_chunks": 0,
            "rssf_scale_breaks": 0,
            "rssf_loss_of_archive": False,
            "rssf_active_scales": [],
            "diff_calls": 0,
            "moe_hits": 0,
            "moe_host_hits": 0,
            "moe_misses": 0,
            "env": "WSL2 Ubuntu" if wsl_running else "Windows",
            "age_s": 0.0,
            "answer_preview": "Run `quint ask`, `quint chat`, or `quint serve` (Windows or WSL2) — live speed, layers, balance, PCIe flow and text will appear here.",
            "thinking_preview": "",
        }

    age = max(0.0, now - best_ts)
    phase = str(best_data.get("phase") or "IDLE").upper()
    active = bool(best_data.get("active", False))

    if active and phase in ("GENERATING", "THINKING") and age > 12.0:
        active = False
        phase = "DONE"
    elif active and phase in ("PREFILL", "LOADING", "READY") and age > 180.0:
        active = False
        phase = "IDLE"

    out = dict(best_data)
    out["source"] = "live_json"
    out["active"] = active
    out["phase"] = phase
    out["age_s"] = round(age, 1)
    if not active and phase in ("DONE", "IDLE"):
        out["live_tok_s"] = 0.0
    else:
        out["live_tok_s"] = float(out.get("tok_s", 0.0) or 0.0)
    return out


def poll_lan_server(lan_url: str) -> Dict[str, Any]:
    urls: List[str] = []
    if lan_url:
        urls.append(lan_url)
    for fallback in ("http://127.0.0.1:8787/api/info", "http://127.0.0.1:8080/api/info"):
        if fallback not in urls:
            urls.append(fallback)

    for u in urls:
        try:
            req = urllib.request.Request(u, method="GET")
            with urllib.request.urlopen(req, timeout=0.18) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                    parsed = urllib.parse.urlparse(u)
                    port = parsed.port or 80
                    eng = data.get("engine") if isinstance(data.get("engine"), dict) else {}
                    lm = eng.get("last_metrics") if isinstance(eng.get("last_metrics"), dict) else {}
                    return {
                        "online": True,
                        "url": u.replace("/api/info", ""),
                        "port": port,
                        "model": str(data.get("model") or data.get("model_id") or eng.get("model_id") or ""),
                        "backend": str(data.get("backend") or eng.get("backend") or ""),
                        "last_metrics": lm,
                    }
        except Exception:
            continue
    base = (lan_url or "http://127.0.0.1:8787/api/info").replace("/api/info", "")
    return {"online": False, "url": base, "port": 8787, "model": "", "backend": "", "last_metrics": {}}


def is_pid_alive(pid: int) -> bool:
    if pid <= 0 or os.name != "nt":
        return True
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not h:
        return False
    code = ctypes.c_ulong(0)
    ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h)
    return bool(ok and code.value == 259)


def atomic_write_json(path: Path, data: Any) -> bool:
    raw = json.dumps(data, ensure_ascii=False)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    for attempt in range(5):
        try:
            tmp.write_text(raw, encoding="utf-8")
            os.replace(tmp, path)
            return True
        except OSError:
            time.sleep(0.015 * (attempt + 1))
    try:
        path.write_text(raw, encoding="utf-8")
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        return True
    except OSError:
        return False


def _log_backend_error(exc: BaseException) -> None:
    try:
        err_path = CACHE_DIR / "backend_err.log"
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with err_path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] {type(exc).__name__}: {exc}\n")
    except OSError:
        pass


def main() -> None:
    parent_pid = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    once_mode = "--once" in sys.argv

    nvml = DirectNVMLReader()
    cpu_tracker = _WinCpuTracker()
    hist_len = 80
    tok_history: List[float] = [0.0] * hist_len
    gpu_history: List[float] = [0.0] * hist_len
    vram_history: List[float] = [0.0] * hist_len
    pcie_history: List[float] = [0.0] * hist_len

    last_lan_check = 0.0
    lan_info: Dict[str, Any] = {"online": False, "url": "", "port": 8787, "model": "", "backend": "", "last_metrics": {}}
    session_peak_tok = 0.0
    session_peak_pcie = 0.0
    ema_tok = 0.0
    ema_gpu = 0.0
    last_model_seen = ""

    try:
        while True:
            if parent_pid > 0 and not is_pid_alive(parent_pid):
                break

            sleep_s = 0.25
            try:
                cfg = read_config()
                sleep_s = max(0.18, min(1.0, float(cfg.get("refresh_ms", 250)) / 1000.0))
                if CLEAR_FLAG.exists():
                    try:
                        CLEAR_FLAG.unlink()
                    except OSError:
                        pass
                    tok_history = [0.0] * hist_len
                    gpu_history = [0.0] * hist_len
                    vram_history = [0.0] * hist_len
                    pcie_history = [0.0] * hist_len
                    session_peak_tok = 0.0
                    session_peak_pcie = 0.0
                    ema_tok = 0.0

                gpu = nvml.poll()
                sys_mem = poll_host_and_wsl(cpu_tracker)
                run = read_live_telemetry(cfg, wsl_running=bool(sys_mem.get("wsl_running")))

                now = time.time()
                if now - last_lan_check >= 3.0:
                    lan_info = poll_lan_server(str(cfg.get("lan_url") or ""))
                    last_lan_check = now

                # Enrich RSSF / KV fields from live LAN server if telemetry file omitted them
                lan_lm = lan_info.get("last_metrics") if isinstance(lan_info.get("last_metrics"), dict) else {}
                if lan_info.get("online"):
                    if run.get("source") == "standby" and lan_info.get("model"):
                        run["model"] = lan_info["model"]
                        run["phase"] = "READY"
                        run["backend"] = lan_info.get("backend") or run.get("backend")
                    if lan_lm and run.get("rssf_d3_margin") is None:
                        run.update(_extract_rssf_fields_from_metrics(lan_lm))

                # Auto-reset speed peaks when switching to a different model
                cur_model = str(run.get("model") or "")
                if cur_model and last_model_seen and cur_model != last_model_seen:
                    tok_history = [0.0] * hist_len
                    session_peak_tok = 0.0
                    ema_tok = 0.0
                if cur_model:
                    last_model_seen = cur_model

                raw_live_tok = float(run.get("live_tok_s", 0.0) or 0.0)
                if raw_live_tok > 600.0:
                    raw_live_tok = 0.0
                if raw_live_tok > 0.0:
                    ema_tok = raw_live_tok if ema_tok <= 0.01 else (0.55 * raw_live_tok + 0.45 * ema_tok)
                else:
                    ema_tok = ema_tok * 0.45 if ema_tok > 0.08 else 0.0

                raw_gpu = float(gpu.get("util_pct", 0.0) or 0.0) if gpu.get("nvml_direct") else float(sys_mem.get("cpu_pct", 0.0) or 0.0)
                ema_gpu = 0.65 * raw_gpu + 0.35 * ema_gpu

                pcie_val = float(gpu.get("pcie_total_mbs", 0.0) or 0.0)
                vram_val = float(gpu.get("vram_used_mib", 0.0) or 0.0)

                peak_reported = float(run.get("peak_tok_s", 0.0) or 0.0)
                avg_reported = float(run.get("avg_tok_s", 0.0) or 0.0)
                toks_cnt = int(run.get("tokens", 0) or 0)
                if toks_cnt >= 4 and 0.0 < raw_live_tok <= 600.0:
                    session_peak_tok = max(session_peak_tok, raw_live_tok)
                if 0.0 < peak_reported <= 600.0:
                    session_peak_tok = max(session_peak_tok, peak_reported)
                if 0.0 < avg_reported <= 600.0:
                    session_peak_tok = max(session_peak_tok, avg_reported)
                if 0.0 <= pcie_val <= _MAX_VALID_PCIE_MBS:
                    session_peak_pcie = max(session_peak_pcie, pcie_val)

                tok_history.append(round(ema_tok, 2))
                gpu_history.append(round(ema_gpu, 1))
                vram_history.append(round(vram_val, 1))
                pcie_history.append(round(pcie_val, 1))

                if len(tok_history) > hist_len:
                    tok_history = tok_history[-hist_len:]
                    gpu_history = gpu_history[-hist_len:]
                    vram_history = vram_history[-hist_len:]
                    pcie_history = pcie_history[-hist_len:]

                lan_pub = {
                    "online": bool(lan_info.get("online", False)),
                    "url": str(lan_info.get("url") or ""),
                    "port": int(lan_info.get("port") or 8787),
                    "model": str(lan_info.get("model") or ""),
                    "backend": str(lan_info.get("backend") or ""),
                }
                payload = {
                    "updated_at": datetime.datetime.now().strftime("%H:%M:%S"),
                    "updated_ts": now,
                    "gpu": gpu,
                    "sys": sys_mem,
                    "lan": lan_pub,
                    "run": run,
                    "session_peak_tok_s": round(session_peak_tok, 2),
                    "session_peak_pcie_mbs": round(session_peak_pcie, 1),
                    "tok_history": tok_history,
                    "gpu_history": gpu_history,
                    "vram_history": vram_history,
                    "pcie_history": pcie_history,
                }
                atomic_write_json(HUD_OUT, payload)
            except Exception as exc:
                _log_backend_error(exc)

            if once_mode:
                break
            time.sleep(sleep_s)
    finally:
        nvml.close()


if __name__ == "__main__":
    main()
