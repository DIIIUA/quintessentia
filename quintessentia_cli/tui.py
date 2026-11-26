"""Cross-platform button-driven interactive terminal UI (TUI) for Quintessentia.

Zero external dependencies:
  - Windows (PowerShell / Windows Terminal / cmd): `msvcrt.getwch()` + VT100 console mode.
  - Linux / WSL2 / macOS: `termios` + `tty.setraw()` + ANSI escape sequences.

Allows configuring EVERY key runtime option with arrow keys (`↑↓←→`, `Space`, `Enter`) without
typing CLI flags: model picker (with live search filter), weight format (`auto`, `q1_cubes`,
`bf16`, `q8`, `q6`, `q4`, `q2`, `q1`), Q1 depth mode (`auto_vram`, `auto_kl`, `manual`),
explicit `K` (1..16), `cube_root_bits` (2/4/8/16), `lm_head_planes`, `min/max` bounds,
`RSSF`, `Token Ledger`, `speculation`, `<think>`, Desktop HUD, and direct launch of
Interactive Chat, LAN Server, Benchmark, Doctor, or Hugging Face Download.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
from typing import Any


# ANSI palette
_RST = "\x1b[0m"
_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"
_CYAN = "\x1b[96m"
_BLUE = "\x1b[94m"
_GREEN = "\x1b[92m"
_YELLOW = "\x1b[93m"
_RED = "\x1b[91m"
_WHITE = "\x1b[97m"
_BG_SEL = "\x1b[48;5;24m"
_CLEAR = "\x1b[2J\x1b[H"
_HIDE_CUR = "\x1b[?25l"
_SHOW_CUR = "\x1b[?25h"

from .i18n import SUPPORTED_LANGS, normalize_lang, resolve_cli_lang, set_lang, tr

WEIGHT_FORMATS = ("auto", "q1_cubes", "bf16", "q8", "q6", "q4", "q2", "q1")
K_MODES = ("auto_vram", "auto_kl", "manual")
PRESETS = ("auto", "auto_kl", "quality", "balanced", "speed", "bf16", "custom")
ROOT_BITS_CHOICES: tuple[int | None, ...] = (None, 2, 4, 8, 16)
LM_HEAD_CHOICES: tuple[int | None, ...] = (None, 4, 5, 6, 8, 12, 16)
KL_TARGET_CHOICES = (0.01, 0.02, 0.05, 0.10, 0.20)
SPEC_CHOICES = ("auto", "off", "lookup", "native", "q1")
VRAM_CHOICES = ("normal", "turbo")
DEVICE_CHOICES = ("auto", "cpu")


@dataclass
class TuiConfig:
    """Persistent configuration backing the button-driven Quintessentia CLI menu."""

    language: str = "en"                # "en" | "ru"
    model: str | None = None
    preset: str = "auto"
    weight_format: str = "auto"
    k_mode: str = "auto_vram"           # "auto_vram" | "auto_kl" | "manual"
    q1_planes: int = 5                  # Used when k_mode == "manual"
    q1_kl_target: float = 0.05          # Used when k_mode == "auto_kl"
    cube_root_bits: int | None = None   # None = auto
    lm_head_planes: int | None = None   # None = auto
    min_q1_planes: int = 3
    max_q1_planes: int = 8
    device: str = "auto"
    vram_mode: str = "normal"
    speculation: str = "auto"
    rssf: str = "on"
    token_ledger: bool = True
    session_id: str = "default"
    thinking: bool = True
    hud: bool = False
    port: int = 8787
    host: str = "0.0.0.0"

    def apply_preset(self, name: str) -> None:
        p = str(name or "auto").strip().lower()
        if p not in PRESETS:
            return
        self.preset = p
        if p == "auto":
            self.weight_format = "auto"
            self.k_mode = "auto_vram"
            self.q1_planes = 5
            self.cube_root_bits = None
            self.lm_head_planes = None
            self.min_q1_planes = 3
            self.max_q1_planes = 8
        elif p == "auto_kl":
            self.weight_format = "q1_cubes"
            self.k_mode = "auto_kl"
            self.q1_kl_target = 0.05
            self.cube_root_bits = 8
            self.lm_head_planes = None
            self.min_q1_planes = 3
            self.max_q1_planes = 8
        elif p == "quality":
            self.weight_format = "q1_cubes"
            self.k_mode = "manual"
            self.q1_planes = 8
            self.cube_root_bits = 8
            self.lm_head_planes = 8
        elif p == "balanced":
            self.weight_format = "q1_cubes"
            self.k_mode = "manual"
            self.q1_planes = 5
            self.cube_root_bits = 8
            self.lm_head_planes = 6
        elif p == "speed":
            self.weight_format = "q1_cubes"
            self.k_mode = "manual"
            self.q1_planes = 4
            self.cube_root_bits = 8
            self.lm_head_planes = 5
        elif p == "bf16":
            self.weight_format = "bf16"
            self.k_mode = "auto_vram"
            self.cube_root_bits = None
            self.lm_head_planes = None

    def build_argv(self, action: str, *, quintessentia_root: str | Path | None = None) -> list[str]:
        """Build an exact, argparse-valid CLI argument vector for `chat`, `serve`, `bench`, or `doctor`."""
        act = str(action).strip().lower()
        argv: list[str] = [act]
        norm_lang = normalize_lang(self.language)
        if norm_lang:
            argv += ["--lang", norm_lang]
        if quintessentia_root is not None:
            argv += ["--quintessentia-root", str(quintessentia_root)]
        if self.model:
            argv += ["--model", str(self.model)]

        argv += ["--device", self.device, "--vram-mode", self.vram_mode, "--weight-format", self.weight_format]

        is_q1_capable = self.weight_format in ("auto", "q1_cubes", "q1")
        if act == "doctor":
            if is_q1_capable and self.k_mode == "manual":
                argv += ["--q1-planes", str(int(self.q1_planes))]
            return argv

        if is_q1_capable:
            if self.k_mode == "manual":
                argv += ["--q1-planes", str(int(self.q1_planes)), "--no-auto-fit-q1"]
            elif self.k_mode == "auto_kl":
                argv += [
                    "--auto-fit-q1",
                    "--min-q1-planes", str(int(self.min_q1_planes)),
                    "--max-q1-planes", str(int(self.max_q1_planes)),
                    "--q1-kl-target", str(float(self.q1_kl_target)),
                ]
            else:
                argv += [
                    "--auto-fit-q1",
                    "--min-q1-planes", str(int(self.min_q1_planes)),
                    "--max-q1-planes", str(int(self.max_q1_planes)),
                ]
            if self.cube_root_bits is not None:
                eff_root = int(self.cube_root_bits)
                if self.k_mode == "manual" and int(self.q1_planes) > eff_root:
                    eff_root = 16 if int(self.q1_planes) > 8 else 8
                argv += ["--cube-root-bits", str(eff_root)]
            if self.lm_head_planes is not None:
                argv += ["--lm-head-planes", str(int(self.lm_head_planes))]

        argv += ["--speculation", self.speculation]
        argv += ["--rssf", self.rssf]
        argv += ["--token-ledger" if self.token_ledger else "--no-token-ledger"]
        if self.token_ledger and self.session_id and self.session_id != "default":
            argv += ["--session-id", self.session_id]
        if self.hud:
            argv.append("--hud")

        if act == "chat":
            if not self.thinking:
                argv.append("--no-think")
        elif act == "serve":
            argv += ["--host", str(self.host), "--port", str(int(self.port))]
        elif act == "bench":
            argv += ["--runs", "3", "--max-tokens", "64"]
            if not self.thinking:
                argv.append("--no-think")
        return argv

    def preview_command(self, action: str = "serve") -> str:
        parts = ["quintessentia"] + self.build_argv(action)
        return " ".join(shlex.quote(p) if (" " in p) else p for p in parts)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "TuiConfig":
        cfg = cls()
        has_saved_lang = False
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        if hasattr(cfg, k):
                            setattr(cfg, k, v)
                    if normalize_lang(raw.get("language")):
                        cfg.language = normalize_lang(raw.get("language")) or "en"
                        has_saved_lang = True
            except Exception:
                pass
        if not has_saved_lang:
            cfg.language = resolve_cli_lang(config_path=path)
        return cfg


# ---------------------------------------------------------------------------
# Cross-platform VT100 + raw single-key reader
# ---------------------------------------------------------------------------

def _enable_windows_vt100() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def _read_posix_key_from_fd(fd: int, *, esc_timeout: float = 0.12) -> str:
    """Read one logical key press directly from an unbuffered POSIX TTY file descriptor.

    Uses raw `os.read(fd, 1)` (never buffered `sys.stdin.read`) so `select.select` always
    agrees with kernel PTY readiness under WSL2 / ConPTY / Windows Terminal.
    """
    import select

    def _read_byte(timeout: float | None) -> bytes:
        if timeout is not None:
            r, _, _ = select.select([fd], [], [], timeout)
            if not r:
                return b""
        return os.read(fd, 1)

    b0 = _read_byte(None)
    if not b0:
        return "CTRL_C"

    byte0 = b0[0]
    if byte0 == 0x03:
        return "CTRL_C"
    if byte0 in (0x0D, 0x0A):
        return "ENTER"
    if byte0 == 0x20:
        return "SPACE"
    if byte0 == 0x09:
        return "TAB"
    if byte0 in (0x7F, 0x08):
        return "BACKSPACE"

    if byte0 == 0x1B:
        b1 = _read_byte(esc_timeout)
        if not b1:
            return "ESC"
        if b1 == b"O":
            # SS3 application-cursor keys: ESC O A/B/C/D/H/F
            b2 = _read_byte(0.08)
            if not b2:
                return ""
            return {
                b"A": "UP",
                b"B": "DOWN",
                b"C": "RIGHT",
                b"D": "LEFT",
                b"H": "HOME",
                b"F": "END",
            }.get(b2, "")
        if b1 == b"[":
            # ECMA-48 CSI sequence: parameter bytes 0x20..0x3F terminated by final byte 0x40..0x7E
            params = bytearray()
            final = b""
            for _ in range(24):
                bi = _read_byte(0.08)
                if not bi:
                    break
                val = bi[0]
                if 0x40 <= val <= 0x7E:
                    final = bi
                    break
                params.append(val)
            if not final:
                return ""
            if final == b"A":
                return "UP"
            if final == b"B":
                return "DOWN"
            if final == b"C":
                return "RIGHT"
            if final == b"D":
                return "LEFT"
            if final == b"H":
                return "HOME"
            if final == b"F":
                return "END"
            if final == b"~":
                p_str = params.decode("ascii", errors="ignore").split(";")[0]
                return {"1": "HOME", "4": "END", "5": "PGUP", "6": "PGDN", "7": "HOME", "8": "END"}.get(p_str, "")
            # Unknown CSI (mouse reports, focus events ESC[I/O, F-keys): ignore instead of exiting!
            return ""
        # Unknown ESC + char (e.g. Alt+key): ignore instead of exiting!
        return ""

    # Multi-byte UTF-8 character (e.g. Cyrillic search input)
    if byte0 >= 0x80:
        need = 1 if (byte0 & 0xE0) == 0xC0 else (2 if (byte0 & 0xF0) == 0xE0 else (3 if (byte0 & 0xF8) == 0xF0 else 0))
        buf = bytearray(b0)
        for _ in range(need):
            bi = _read_byte(0.05)
            if not bi:
                break
            buf.extend(bi)
        return buf.decode("utf-8", errors="ignore")

    return b0.decode("ascii", errors="ignore")


def _read_key() -> str:
    """Read one logical key press: UP, DOWN, LEFT, RIGHT, PGUP, PGDN, ENTER, SPACE, TAB, ESC, BACKSPACE, or char."""
    if os.name == "nt":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            code = msvcrt.getwch()
            return {
                "H": "UP",
                "P": "DOWN",
                "K": "LEFT",
                "M": "RIGHT",
                "I": "PGUP",
                "Q": "PGDN",
                "G": "HOME",
                "O": "END",
            }.get(code, "")
        if ch in ("\r", "\n"):
            return "ENTER"
        if ch == " ":
            return "SPACE"
        if ch == "\t":
            return "TAB"
        if ch == "\x1b":
            return "ESC"
        if ch in ("\x08", "\x7f"):
            return "BACKSPACE"
        if ch == "\x03":
            return "CTRL_C"
        return ch

    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return _read_posix_key_from_fd(fd)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _cycle(seq: tuple[Any, ...], current: Any, step: int) -> Any:
    try:
        idx = seq.index(current)
    except ValueError:
        idx = 0
    return seq[(idx + step) % len(seq)]


# ---------------------------------------------------------------------------
# Environment & local model discovery
# ---------------------------------------------------------------------------

def _discover_models(root: Path) -> list[dict[str, Any]]:
    models_dir = root / "models"
    items: list[dict[str, Any]] = []
    if not models_dir.is_dir():
        return items
    for entry in sorted(models_dir.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_dir() or entry.name.startswith("_") or entry.name.startswith("."):
            continue
        files = [f for f in entry.iterdir() if f.is_file()]
        ggufs = [f for f in files if f.suffix.lower() == ".gguf"]
        safes = [f for f in files if f.suffix.lower() == ".safetensors"]
        bins = [f for f in files if f.suffix.lower() == ".bin"]
        w_bytes = sum(f.stat().st_size for f in (ggufs + safes + bins))
        if w_bytes <= 0:
            continue
        fmt = "GGUF" if ggufs else ("HF" if safes else "BIN")
        items.append({
            "name": entry.name,
            "path": str(entry),
            "format": fmt,
            "size_gib": round(w_bytes / (1024 ** 3), 2),
        })
    return items


def _gpu_banner() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            free_b, tot_b = torch.cuda.mem_get_info(0)
            return f"{name}  ({free_b / 2**30:.2f} / {tot_b / 2**30:.2f} GiB free)"
    except Exception:
        pass
    return "CPU mode (CUDA unavailable)"


# ---------------------------------------------------------------------------
# Sub-screens (Model Picker + Inline Text Prompt)
# ---------------------------------------------------------------------------

def _prompt_line(title: str, current: str = "") -> str | None:
    sys.stdout.write(_SHOW_CUR + "\n")
    sys.stdout.flush()
    try:
        hint = f" [{current}]" if current else ""
        val = input(f"  {_CYAN}{title}{hint}:{_RST} ").strip()
        return val if val else current
    except (EOFError, KeyboardInterrupt):
        return None
    finally:
        sys.stdout.write(_HIDE_CUR)
        sys.stdout.flush()


def _model_picker_screen(models: list[dict[str, Any]], current_model: str | None, *, lang: str = "en") -> str | None:
    if not models:
        return current_model
    query = ""
    cursor = 0
    for i, m in enumerate(models):
        if m["name"] == current_model:
            cursor = i
            break

    while True:
        filtered = (
            [m for m in models if query.lower() in m["name"].lower() or query.lower() in m["format"].lower()]
            if query
            else models
        )
        if not filtered:
            cursor = 0
        else:
            cursor = max(0, min(cursor, len(filtered) - 1))

        term_w, term_h = shutil.get_terminal_size((100, 30))
        view_rows = max(6, min(18, term_h - 10))
        start = max(0, min(cursor - view_rows // 2, max(0, len(filtered) - view_rows)))
        visible = filtered[start : start + view_rows]

        buf = [_CLEAR]
        hdr_title = tr(
            "Select Local Model ({shown}/{total})",
            "Выбор локальной модели ({shown}/{total})",
            lang=lang,
            shown=len(filtered),
            total=len(models),
        )
        buf.append(f"{_BOLD}{_CYAN}╭─ {hdr_title} ─{'─' * max(4, term_w - len(hdr_title) - 8)}╮{_RST}")
        search_lbl = tr("Search (type to filter / Backspace):", "Поиск (просто печатай / Backspace):", lang=lang)
        nav_lbl = tr("[↑↓ / PgUp PgDn] navigate  [Enter] select  [Esc] back", "[↑↓ / PgUp PgDn] навигация  [Enter] выбрать  [Esc] назад", lang=lang)
        buf.append(
            f"  {_DIM}{search_lbl}{_RST} {_YELLOW}{query or '—'}{_RST}   "
            f"{_DIM}{nav_lbl}{_RST}"
        )
        buf.append(f"{_DIM}{'─' * min(term_w - 1, 96)}{_RST}")
        if not visible:
            no_match = tr("No models match filter '{q}'", "Ничего не найдено по фильтру '{q}'", lang=lang, q=query)
            buf.append(f"  {_RED}{no_match}{_RST}")
        else:
            for offset, m in enumerate(visible):
                real_idx = start + offset
                selected = real_idx == cursor
                mark = "▸" if selected else " "
                badge = f"{_GREEN if m['format']=='GGUF' else _BLUE}[{m['format']:4s}]{_RST}"
                sz = f"{m['size_gib']:6.2f} GiB"
                name_str = m["name"][: max(20, min(term_w - 28, 62))]
                if selected:
                    buf.append(f"{_BG_SEL}{_BOLD}{_WHITE} {mark} {real_idx + 1:2d}. {name_str:<54s} {m['format']:4s} {sz} {_RST}")
                else:
                    buf.append(f" {mark} {_DIM}{real_idx + 1:2d}.{_RST} {name_str:<54s} {badge} {_DIM}{sz}{_RST}")
        buf.append(f"{_DIM}{'─' * min(term_w - 1, 96)}{_RST}")
        sys.stdout.write("\n".join(buf) + "\n")
        sys.stdout.flush()

        key = _read_key()
        if key in ("ESC", "CTRL_C"):
            return current_model
        if key == "UP":
            cursor = (cursor - 1) % max(1, len(filtered))
        elif key == "DOWN":
            cursor = (cursor + 1) % max(1, len(filtered))
        elif key == "PGUP":
            cursor = max(0, cursor - view_rows)
        elif key == "PGDN":
            cursor = min(max(0, len(filtered) - 1), cursor + view_rows)
        elif key == "ENTER":
            if filtered:
                return filtered[cursor]["name"]
            return current_model
        elif key == "BACKSPACE":
            query = query[:-1]
            cursor = 0
        elif len(key) == 1 and key.isprintable() and key != "/":
            query += key
            cursor = 0


# ---------------------------------------------------------------------------
# Main interactive menu screen
# ---------------------------------------------------------------------------

def _fmt_on_off(flag: bool, *, lang: str = "en") -> str:
    on_txt = "[ ВКЛ ]" if lang == "ru" else "[ ON ]"
    off_txt = "[ ВЫКЛ ]" if lang == "ru" else "[ OFF ]"
    return f"{_GREEN}{_BOLD}{on_txt}{_RST}" if flag else f"{_DIM}{off_txt}{_RST}"


def _build_menu_rows(cfg: TuiConfig, models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lang = normalize_lang(cfg.language) or "en"
    cur_m = next((m for m in models if m["name"] == cfg.model), None)
    model_val = (
        f"{cfg.model} ({cur_m['format']}, {cur_m['size_gib']:.2f} GiB)"
        if cur_m is not None
        else (cfg.model or tr("(not selected — press Enter)", "(не выбрана — нажми Enter)", lang=lang))
    )
    k_mode_labels = {
        "auto_vram": tr("auto (fit K to VRAM)", "auto (подбор K под VRAM)", lang=lang),
        "auto_kl": tr(
            "auto-KL (calibrate K to KL <= {t})",
            "auto-KL (подбор K по точности KL <= {t})",
            lang=lang,
            t=cfg.q1_kl_target,
        ),
        "manual": tr(
            "manual (fixed K = {k})",
            "ручной (фиксированное K = {k})",
            lang=lang,
            k=cfg.q1_planes,
        ),
    }
    preset_labels = {
        "auto": tr("Auto (smart VRAM)", "Auto (умный VRAM)", lang=lang),
        "auto_kl": tr("Auto-KL (smart accuracy)", "Auto-KL (умная точность)", lang=lang),
        "quality": "Max Quality (Q1 k=8, lm=8)",
        "balanced": "Balanced (Q1 k=5, lm=6)",
        "speed": "Fast / Low-VRAM (Q1 k=4)",
        "bf16": tr("BF16 Native (no Q1)", "BF16 Native (без Q1)", lang=lang),
        "custom": tr("Custom (manual tuning)", "Custom (ручная настройка)", lang=lang),
    }
    lang_val = "English" if lang == "en" else "Русский"
    is_q1 = cfg.weight_format in ("auto", "q1_cubes", "q1")
    rows: list[dict[str, Any]] = [
        {"id": "section", "label": tr("─── MODEL & PRESET ───", "─── МОДЕЛЬ И ПРЕСЕТ ───", lang=lang)},
        {
            "id": "language",
            "label": tr("Language / Язык", "Язык / Language", lang=lang),
            "value": f"◂ {lang_val} ▸",
            "hint": tr("← → or [L]: EN / RU", "← → или [L]: EN / RU", lang=lang),
        },
        {
            "id": "model",
            "label": tr("Model", "Модель", lang=lang),
            "value": model_val,
            "hint": tr("Enter: model picker", "Enter: список моделей", lang=lang),
        },
        {
            "id": "preset",
            "label": tr("Quick Preset", "Быстрый пресет", lang=lang),
            "value": f"◂ {preset_labels.get(cfg.preset, cfg.preset)} ▸",
            "hint": tr("← → switch", "← → переключить", lang=lang),
        },
        {"id": "section", "label": tr("─── FORMAT, Q1 CUBES & DEPTH K ───", "─── ФОРМАТ, КУБЫ Q1 И ГЛУБИНА K ───", lang=lang)},
        {
            "id": "weight_format",
            "label": tr("Weight Format", "Формат весов", lang=lang),
            "value": f"◂ {cfg.weight_format} ▸",
            "hint": tr("auto = Q1 cubes on GPU", "auto = Q1-кубы на GPU", lang=lang),
        },
    ]
    if is_q1:
        rows.append({
            "id": "k_mode",
            "label": tr("K Depth Mode", "Режим глубины K", lang=lang),
            "value": f"◂ {k_mode_labels[cfg.k_mode]} ▸",
            "hint": tr("auto / auto-KL / manual", "auto / auto-KL / ручной", lang=lang),
        })
        if cfg.k_mode == "manual":
            rows.append({
                "id": "q1_planes",
                "label": tr("Depth K (planes)", "Глубина K (плоскостей)", lang=lang),
                "value": f"◂ {cfg.q1_planes} / 16 ▸",
                "hint": "← → 1..16",
            })
        elif cfg.k_mode == "auto_kl":
            rows.append({
                "id": "q1_kl_target",
                "label": tr("KL Accuracy Target", "Порог точности KL", lang=lang),
                "value": f"◂ <= {cfg.q1_kl_target:.2f} nats ▸",
                "hint": tr("← → KL threshold", "← → порог KL", lang=lang),
            })
            rows.append({
                "id": "min_q1_planes",
                "label": tr("Auto-K Min", "Авто-K минимум", lang=lang),
                "value": f"◂ {cfg.min_q1_planes} ▸",
                "hint": tr("lower bound", "нижняя граница", lang=lang),
            })
            rows.append({
                "id": "max_q1_planes",
                "label": tr("Auto-K Max", "Авто-K максимум", lang=lang),
                "value": f"◂ {cfg.max_q1_planes} ▸",
                "hint": tr("upper bound", "верхняя граница", lang=lang),
            })
        else:
            rows.append({
                "id": "min_q1_planes",
                "label": tr("Auto-K Min", "Авто-K минимум", lang=lang),
                "value": f"◂ {cfg.min_q1_planes} ▸",
                "hint": tr("step down floor", "шаг вниз до", lang=lang),
            })
            rows.append({
                "id": "max_q1_planes",
                "label": tr("Auto-K Max", "Авто-K максимум", lang=lang),
                "value": f"◂ {cfg.max_q1_planes} ▸",
                "hint": tr("step up ceiling", "шаг вверх до", lang=lang),
            })
        root_lbl = "auto" if cfg.cube_root_bits is None else tr("{b}-bit cube", "{b}-bit куб", lang=lang, b=cfg.cube_root_bits)
        lm_lbl = "auto" if cfg.lm_head_planes is None else tr("{p} planes", "{p} плоскостей", lang=lang, p=cfg.lm_head_planes)
        rows.append({
            "id": "cube_root_bits",
            "label": tr("Cube Root (root_bits)", "Корень куба (root_bits)", lang=lang),
            "value": f"◂ {root_lbl} ▸",
            "hint": "auto / 2 / 4 / 8 / 16",
        })
        rows.append({
            "id": "lm_head_planes",
            "label": tr("lm_head Precision", "Точность lm_head", lang=lang),
            "value": f"◂ {lm_lbl} ▸",
            "hint": tr("independent head depth", "отдельная глубина головы", lang=lang),
        })

    rows += [
        {"id": "section", "label": tr("─── ENGINE & SERVER SETTINGS ───", "─── ПАРАМЕТРЫ ДВИЖКА И СЕРВЕРА ───", lang=lang)},
        {
            "id": "device",
            "label": tr("Device", "Устройство", lang=lang),
            "value": f"◂ {cfg.device} ▸",
            "hint": "auto (GPU) / cpu",
        },
        {
            "id": "vram_mode",
            "label": tr("VRAM Mode", "Режим VRAM", lang=lang),
            "value": f"◂ {cfg.vram_mode} ▸",
            "hint": "normal / turbo",
        },
        {
            "id": "speculation",
            "label": tr("Speculation / MTP", "Спекуляция / MTP", lang=lang),
            "value": f"◂ {cfg.speculation} ▸",
            "hint": "auto / off / lookup / native / q1",
        },
        {
            "id": "rssf",
            "label": tr("RSSF Memory Guard", "Защита памяти RSSF", lang=lang),
            "value": _fmt_on_off(cfg.rssf == "on", lang=lang),
            "hint": tr("Volterra/FDS balance guard", "Volterra/FDS страховка", lang=lang),
        },
        {
            "id": "token_ledger",
            "label": tr("Disk Token Ledger", "Дисковый Token Ledger", lang=lang),
            "value": _fmt_on_off(cfg.token_ledger, lang=lang),
            "hint": tr("100% retokenization + archive", "100% перетокенизация + архив", lang=lang),
        },
        {
            "id": "thinking",
            "label": tr("<think> Reasoning", "Режим <think>", lang=lang),
            "value": _fmt_on_off(cfg.thinking, lang=lang),
            "hint": tr("chain-of-thought block", "цепочка рассуждений", lang=lang),
        },
        {
            "id": "hud",
            "label": "Desktop Live HUD",
            "value": _fmt_on_off(cfg.hud, lang=lang),
            "hint": tr("telemetry widget", "окно телеметрии", lang=lang),
        },
        {
            "id": "port",
            "label": tr("LAN Server Port", "Порт LAN-сервера", lang=lang),
            "value": f"{cfg.host}:{cfg.port}",
            "hint": tr("Enter: change port", "Enter: изменить порт", lang=lang),
        },
        {"id": "section", "label": tr("─── ACTIONS (PRESS ENTER) ───", "─── ДЕЙСТВИЯ (НАЖМИ ENTER) ───", lang=lang)},
        {
            "id": "act_chat",
            "label": tr("▶  Launch Interactive Chat", "▶  Запустить интерактивный Чат", lang=lang),
            "value": tr("(inside: /k, /compact, /think, /lang)", "(внутри: /k, /compact, /think, /lang)", lang=lang),
            "action": True,
        },
        {
            "id": "act_serve",
            "label": tr("🌐 Launch LAN Server + Web UI", "🌐 Запустить LAN-сервер + Web UI", lang=lang),
            "value": f"http://localhost:{cfg.port}",
            "action": True,
        },
        {
            "id": "act_bench",
            "label": tr("⚡ Run Quick Benchmark (3 turns)", "⚡ Прогнать быстрый бенчмарк (3 хода)", lang=lang),
            "value": tr("speed + VRAM + KV", "скорость + VRAM + KV", lang=lang),
            "action": True,
        },
        {
            "id": "act_doctor",
            "label": tr("🩺 GPU & VRAM Diagnostics (Doctor)", "🩺 Диагностика GPU и плана VRAM (Doctor)", lang=lang),
            "value": tr("budget check without startup", "проверка бюджета без старта", lang=lang),
            "action": True,
        },
        {
            "id": "act_download",
            "label": tr("⬇  Download Model from Hugging Face", "⬇  Скачать модель с Hugging Face", lang=lang),
            "value": tr("enter repo or URL", "ввод repo/URL", lang=lang),
            "action": True,
        },
        {
            "id": "act_save",
            "label": tr("💾 Save Settings as Default", "💾 Сохранить настройки как умолчание", lang=lang),
            "value": "cli_config.json",
            "action": True,
        },
        {
            "id": "act_exit",
            "label": tr("✕  Exit", "✕  Выход", lang=lang),
            "value": "",
            "action": True,
        },
    ]
    return rows


def run_tui(*, quintessentia_root: str | Path | None = None, lang: str | None = None) -> int:
    """Launch the interactive button-driven Quintessentia CLI menu."""
    from . import __version__
    from .cli import main as cli_main
    from .runtime_config import find_quintessentia_root

    root = find_quintessentia_root(quintessentia_root)
    cfg_path = root / "cli_config.json"
    cfg = TuiConfig.load(cfg_path)
    explicit_lang = normalize_lang(lang)
    if explicit_lang:
        cfg.language = explicit_lang
    else:
        cfg.language = normalize_lang(cfg.language) or resolve_cli_lang(config_path=cfg_path)
    set_lang(cfg.language)

    models = _discover_models(root)
    if (not cfg.model or not any(m["name"] == cfg.model for m in models)) and models:
        # Prefer Qwythos if present, else first local model
        pref = next((m["name"] for m in models if "qwythos" in m["name"].lower()), models[0]["name"])
        cfg.model = pref

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(
            tr(
                "Quintessentia Interactive Menu requires a TTY. Use `quintessentia --help` for CLI flags.",
                "Интерактивное меню Quintessentia требует TTY. Используйте `quintessentia --help` для флагов CLI.",
                lang=cfg.language,
            )
        )
        return 0

    _enable_windows_vt100()
    gpu_info = _gpu_banner()
    status_msg = tr(
        "Ready. Navigate with ↑↓ ←→, Space, Enter, or press [L] to switch EN/RU.",
        "Готово. Управляй стрелками ↑↓ ←→, Space, Enter или нажми [L] для смены EN/RU.",
        lang=cfg.language,
    )
    if cfg.hud:
        try:
            from .cli import launch_desktop_hud
            if launch_desktop_hud(quiet=True, bind_to_console=True):
                status_msg = tr(
                    "Ready (Desktop HUD opened automatically). Navigate with ↑↓ ←→, Space, Enter.",
                    "Готово (Desktop HUD открыт автоматически). Управляй стрелками ↑↓ ←→, Space и Enter.",
                    lang=cfg.language,
                )
        except Exception:
            pass
    cursor_id = "act_chat"

    sys.stdout.write(_HIDE_CUR)
    sys.stdout.flush()
    try:
        while True:
            cur_lang = normalize_lang(cfg.language) or "en"
            rows = _build_menu_rows(cfg, models)
            selectable = [i for i, r in enumerate(rows) if r["id"] != "section"]
            cur_pos = next((idx for idx, r_i in enumerate(selectable) if rows[r_i]["id"] == cursor_id), 0)
            row_idx = selectable[cur_pos]
            cursor_id = rows[row_idx]["id"]

            term_w, term_h = shutil.get_terminal_size((104, 34))
            box_w = min(max(72, term_w - 2), 104)
            view_h = max(10, term_h - 11)
            start = max(0, min(row_idx - view_h // 2, max(0, len(rows) - view_h)))
            vis = rows[start : start + view_h]

            lines = [_CLEAR]
            title = tr(
                " QUINTESSENTIA v{ver} — Interactive Control Center ",
                " QUINTESSENTIA v{ver} — Кнопочный центр управления ",
                lang=cur_lang,
                ver=__version__,
            )
            lines.append(f"{_BOLD}{_CYAN}╭─{title}{'─' * max(2, box_w - len(title) - 3)}╮{_RST}")
            models_lbl = tr("| Models:", "| Моделей:", lang=cur_lang)
            lang_badge = "EN" if cur_lang == "en" else "RU"
            lines.append(
                f"  {_DIM}GPU:{_RST} {_WHITE}{gpu_info}{_RST}   "
                f"{_DIM}{models_lbl}{_RST} {_CYAN}{len(models)}{_RST}   "
                f"{_DIM}| Lang:{_RST} {_GREEN}{lang_badge}{_RST}"
            )
            lines.append(f"{_DIM}{'─' * box_w}{_RST}")

            for off, r in enumerate(vis):
                abs_i = start + off
                if r["id"] == "section":
                    lines.append(f"  {_BOLD}{_YELLOW}{r['label']}{_RST}")
                    continue
                is_sel = abs_i == row_idx
                lbl = r["label"]
                val = r.get("value", "")
                hint = r.get("hint", "")
                if is_sel:
                    plain_val = val
                    for code in (_RST, _BOLD, _DIM, _CYAN, _BLUE, _GREEN, _YELLOW, _RED, _WHITE):
                        plain_val = plain_val.replace(code, "")
                    extra = f"  ({hint})" if hint else ""
                    line_txt = f" ▸ {lbl:<34s} {plain_val}{extra}"
                    lines.append(f"{_BG_SEL}{_BOLD}{_WHITE}{line_txt:<{box_w}s}{_RST}")
                else:
                    if r.get("action"):
                        lines.append(f"   {_BOLD}{_CYAN}{lbl:<34s}{_RST} {_DIM}{val}{_RST}")
                    else:
                        lines.append(f"   {lbl:<34s} {val}  {_DIM}{hint}{_RST}")

            lines.append(f"{_DIM}{'─' * box_w}{_RST}")
            preview_act = "chat" if cursor_id == "act_chat" else ("bench" if cursor_id == "act_bench" else ("doctor" if cursor_id == "act_doctor" else "serve"))
            cmd_prev = cfg.preview_command(preview_act)
            if len(cmd_prev) > box_w - 6:
                cmd_prev = cmd_prev[: box_w - 9] + "..."
            cmd_lbl = tr("Command:", "Команда:", lang=cur_lang)
            lines.append(f"  {_DIM}{cmd_lbl}{_RST} {_GREEN}{cmd_prev}{_RST}")
            lines.append(f"  {_YELLOW}{status_msg}{_RST}")
            footer_hint = tr(
                "[↑↓] select  [←→ / Space] change  [Enter] open / run  [L] EN/RU  [S] save  [Q / Esc] exit",
                "[↑↓] выбор  [←→ / Space] изменить  [Enter] открыть / запуск  [L] EN/RU  [S] сохранить  [Q / Esc] выход",
                lang=cur_lang,
            )
            lines.append(f"  {_DIM}{footer_hint}{_RST}")
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()

            key = _read_key()
            if not key:
                continue
            if key in ("ESC", "CTRL_C", "q", "Q"):
                return 0
            if key == "UP":
                cur_pos = (cur_pos - 1) % len(selectable)
                cursor_id = rows[selectable[cur_pos]]["id"]
                continue
            if key == "DOWN":
                cur_pos = (cur_pos + 1) % len(selectable)
                cursor_id = rows[selectable[cur_pos]]["id"]
                continue
            if key == "HOME":
                cursor_id = rows[selectable[0]]["id"]
                continue
            if key == "END":
                cursor_id = rows[selectable[-1]]["id"]
                continue
            if key == "PGUP":
                cur_pos = max(0, cur_pos - 6)
                cursor_id = rows[selectable[cur_pos]]["id"]
                continue
            if key == "PGDN":
                cur_pos = min(len(selectable) - 1, cur_pos + 6)
                cursor_id = rows[selectable[cur_pos]]["id"]
                continue
            if key in ("l", "L"):
                cfg.language = _cycle(SUPPORTED_LANGS, cur_lang, 1)
                set_lang(cfg.language)
                status_msg = tr("Language switched to English.", "Язык переключён на русский.", lang=cfg.language)
                continue
            if key in ("s", "S"):
                cfg.save(cfg_path)
                status_msg = tr("Settings saved to {p}", "Настройки сохранены в {p}", lang=cur_lang, p=cfg_path)
                continue

            step = 1 if key in ("RIGHT", "SPACE", "ENTER") else (-1 if key == "LEFT" else 0)
            rid = cursor_id

            if rid == "language" and step:
                cfg.language = _cycle(SUPPORTED_LANGS, cur_lang, step)
                set_lang(cfg.language)
                status_msg = tr("Language switched to English.", "Язык переключён на русский.", lang=cfg.language)
            elif rid == "model" and key in ("ENTER", "SPACE", "RIGHT", "LEFT"):
                if key in ("LEFT", "RIGHT") and models:
                    names = tuple(m["name"] for m in models)
                    cfg.model = _cycle(names, cfg.model, 1 if key == "RIGHT" else -1)
                    status_msg = tr("Selected model: {m}", "Выбрана модель: {m}", lang=cur_lang, m=cfg.model)
                else:
                    picked = _model_picker_screen(models, cfg.model, lang=cur_lang)
                    if picked:
                        cfg.model = picked
                        status_msg = tr("Selected model: {m}", "Выбрана модель: {m}", lang=cur_lang, m=cfg.model)
            elif rid == "preset" and step:
                nxt = _cycle(PRESETS[:-1], cfg.preset if cfg.preset != "custom" else "auto", step)
                cfg.apply_preset(nxt)
                status_msg = tr("Applied preset: {p}", "Применён пресет: {p}", lang=cur_lang, p=nxt)
            elif rid == "weight_format" and step:
                cfg.weight_format = _cycle(WEIGHT_FORMATS, cfg.weight_format, step)
                cfg.preset = "custom"
                status_msg = tr("Weight format: {wf}", "Формат весов: {wf}", lang=cur_lang, wf=cfg.weight_format)
            elif rid == "k_mode" and step:
                cfg.k_mode = _cycle(K_MODES, cfg.k_mode, step)
                cfg.preset = "custom"
                status_msg = tr("K depth mode: {km}", "Режим подбора K: {km}", lang=cur_lang, km=cfg.k_mode)
            elif rid == "q1_planes" and step:
                cfg.q1_planes = max(1, min(16, cfg.q1_planes + step))
                cfg.preset = "custom"
                status_msg = tr("Manual depth K = {k}", "Ручная глубина K = {k}", lang=cur_lang, k=cfg.q1_planes)
            elif rid == "q1_kl_target" and step:
                cfg.q1_kl_target = _cycle(KL_TARGET_CHOICES, cfg.q1_kl_target, step)
                cfg.preset = "custom"
                status_msg = tr(
                    "Auto-KL target <= {t:.2f} nats",
                    "Порог авто-KL <= {t:.2f} nats",
                    lang=cur_lang,
                    t=cfg.q1_kl_target,
                )
            elif rid == "min_q1_planes" and step:
                cfg.min_q1_planes = max(1, min(cfg.max_q1_planes, cfg.min_q1_planes + step))
                cfg.preset = "custom"
            elif rid == "max_q1_planes" and step:
                cfg.max_q1_planes = max(cfg.min_q1_planes, min(16, cfg.max_q1_planes + step))
                cfg.preset = "custom"
            elif rid == "cube_root_bits" and step:
                cfg.cube_root_bits = _cycle(ROOT_BITS_CHOICES, cfg.cube_root_bits, step)
                cfg.preset = "custom"
            elif rid == "lm_head_planes" and step:
                cfg.lm_head_planes = _cycle(LM_HEAD_CHOICES, cfg.lm_head_planes, step)
                cfg.preset = "custom"
            elif rid == "device" and step:
                cfg.device = _cycle(DEVICE_CHOICES, cfg.device, step)
            elif rid == "vram_mode" and step:
                cfg.vram_mode = _cycle(VRAM_CHOICES, cfg.vram_mode, step)
            elif rid == "speculation" and step:
                cfg.speculation = _cycle(SPEC_CHOICES, cfg.speculation, step)
            elif rid == "rssf" and step:
                cfg.rssf = "off" if cfg.rssf == "on" else "on"
            elif rid == "token_ledger" and step:
                cfg.token_ledger = not cfg.token_ledger
            elif rid == "thinking" and step:
                cfg.thinking = not cfg.thinking
            elif rid == "hud" and step:
                cfg.hud = not cfg.hud
                try:
                    from .cli import launch_desktop_hud, stop_desktop_hud, stop_hud_console_heartbeat
                    if cfg.hud:
                        ok_hud = launch_desktop_hud(quiet=True, bind_to_console=True)
                        status_msg = (
                            tr("Desktop HUD enabled and opened on screen.", "Desktop HUD включён и открыт на экране.", lang=cur_lang)
                            if ok_hud
                            else tr("Failed to launch Desktop HUD (check Windows host).", "Desktop HUD не удалось запустить (проверь Windows host).", lang=cur_lang)
                        )
                    else:
                        stop_hud_console_heartbeat(signal_stop=True)
                        stop_desktop_hud(quiet=True)
                        status_msg = tr("Desktop HUD disabled.", "Desktop HUD выключен.", lang=cur_lang)
                except Exception as exc:
                    status_msg = f"Desktop HUD: {exc}"
            elif rid == "port" and key == "ENTER":
                val = _prompt_line(tr("LAN Server Port", "Порт LAN-сервера", lang=cur_lang), str(cfg.port))
                if val and val.isdigit() and 1 <= int(val) <= 65535:
                    cfg.port = int(val)
                    status_msg = tr("Port updated: {p}", "Порт обновлён: {p}", lang=cur_lang, p=cfg.port)
            elif rid == "act_save" and key in ("ENTER", "SPACE"):
                cfg.save(cfg_path)
                status_msg = tr("Saved to {p}", "Сохранено в {p}", lang=cur_lang, p=cfg_path)
            elif rid == "act_exit" and key in ("ENTER", "SPACE"):
                return 0
            elif rid in ("act_chat", "act_serve", "act_bench", "act_doctor") and key == "ENTER":
                act = rid.replace("act_", "")
                argv = cfg.build_argv(act, quintessentia_root=root)
                sys.stdout.write(_SHOW_CUR + _CLEAR)
                sys.stdout.flush()
                print(
                    tr(
                        "[Quintessentia] Running: {cmd}\n",
                        "[Quintessentia] Запуск: {cmd}\n",
                        lang=cur_lang,
                        cmd=cfg.preview_command(act),
                    ),
                    flush=True,
                )
                rc_code = 0
                had_error = False
                try:
                    rc_code = int(cli_main(argv) or 0)
                except SystemExit as exc:
                    rc_code = int(exc.code) if isinstance(exc.code, int) else 0
                except KeyboardInterrupt:
                    print(tr("\n[Stopped by user]", "\n[Остановлено пользователем]", lang=cur_lang), flush=True)
                except Exception as exc:
                    had_error = True
                    print(tr("\n[Launch error] {e}", "\n[Ошибка запуска] {e}", lang=cur_lang, e=exc), flush=True)
                # After clean server/chat return (M/Q/Esc/Ctrl+C), jump straight back to menu!
                # Pause only for bench/doctor (so user can read results) or on errors.
                if had_error or rc_code != 0 or act in ("bench", "doctor"):
                    try:
                        input(
                            tr(
                                "\nPress Enter to return to Quintessentia menu...",
                                "\nНажми Enter, чтобы вернуться в меню Quintessentia...",
                                lang=cur_lang,
                            )
                        )
                    except (EOFError, KeyboardInterrupt):
                        return 0
                models = _discover_models(root)
                gpu_info = _gpu_banner()
                if act == "serve" and not had_error and rc_code == 0:
                    status_msg = tr(
                        "LAN server stopped, VRAM released.",
                        "LAN-сервер остановлен, видеопамять освобождена.",
                        lang=cur_lang,
                    )
                elif act == "chat" and not had_error and rc_code == 0:
                    status_msg = tr(
                        "Chat finished, VRAM released.",
                        "Чат завершён, видеопамять освобождена.",
                        lang=cur_lang,
                    )
                sys.stdout.write(_HIDE_CUR)
                sys.stdout.flush()
            elif rid == "act_download" and key == "ENTER":
                ref = _prompt_line(
                    tr(
                        "Hugging Face repo or direct GGUF URL (blank = cancel)",
                        "Hugging Face repo или прямая ссылка на GGUF (пусто = отмена)",
                        lang=cur_lang,
                    ),
                    "",
                )
                if ref:
                    sys.stdout.write(_SHOW_CUR + _CLEAR)
                    sys.stdout.flush()
                    try:
                        cli_main(["download", "--lang", cur_lang, "--quintessentia-root", str(root), ref])
                    except SystemExit:
                        pass
                    except KeyboardInterrupt:
                        print(tr("\n[Download cancelled]", "\n[Скачивание прервано]", lang=cur_lang), flush=True)
                    try:
                        input(
                            tr(
                                "\nPress Enter to return to Quintessentia menu...",
                                "\nНажми Enter, чтобы вернуться в меню Quintessentia...",
                                lang=cur_lang,
                            )
                        )
                    except (EOFError, KeyboardInterrupt):
                        return 0
                    models = _discover_models(root)
                    status_msg = tr(
                        "Model list updated ({n})",
                        "Список моделей обновлён ({n})",
                        lang=cur_lang,
                        n=len(models),
                    )
                    sys.stdout.write(_HIDE_CUR)
                    sys.stdout.flush()
    finally:
        try:
            from .cli import stop_hud_console_heartbeat
            stop_hud_console_heartbeat(signal_stop=True)
        except Exception:
            pass
        sys.stdout.write(_SHOW_CUR + _RST)
        sys.stdout.flush()
