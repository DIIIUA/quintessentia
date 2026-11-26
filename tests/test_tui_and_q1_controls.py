"""Tests for the button-driven TUI launcher, fine-grained Q1 CLI controls, live K switching, KL calibration, and LAN API."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

import pytest
import torch
import torch.nn as nn

from quintessentia_cli.cli import build_parser
from quintessentia_cli.lan_server import QuintessentiaLANServer
from quintessentia_cli.tui import PRESETS, TuiConfig, _build_menu_rows
from quintessentia_cli.unified_runtime import QuintessentiaUnifiedEngine
from src.binary_halving_q1 import BinaryHalvingQ1Linear, carve_trunk_mirror_q1


def test_tui_config_presets_save_load_and_build_valid_cli_argv(tmp_path: Path):
    """Every TUI preset and action must produce an exact CLI argv accepted by `build_parser()`."""
    parser = build_parser()
    cfg_path = tmp_path / "cli_config.json"

    for preset in PRESETS:
        if preset == "custom":
            continue
        cfg = TuiConfig(model="Qwythos-9B-v2-MTP-BF16")
        cfg.apply_preset(preset)
        cfg.save(cfg_path)
        reloaded = TuiConfig.load(cfg_path)
        assert reloaded.preset == preset
        assert reloaded.weight_format == cfg.weight_format
        assert reloaded.k_mode == cfg.k_mode

        for act in ("chat", "serve", "bench", "doctor"):
            argv = reloaded.build_argv(act)
            ns = parser.parse_args(argv)
            assert ns.command == act
            assert ns.model == "Qwythos-9B-v2-MTP-BF16"

    # Menu / UI aliases parse cleanly
    for alias in ("menu", "ui", "tui"):
        ns = parser.parse_args([alias])
        assert ns.command == alias

    # Menu row builder works cleanly across all modes without error
    rows = _build_menu_rows(TuiConfig(model="demo", k_mode="auto_kl"), [{"name": "demo", "format": "GGUF", "size_gib": 4.2}])
    assert any(r["id"] == "q1_kl_target" for r in rows)
    assert any(r["id"] == "act_chat" for r in rows)


def test_fine_grained_q1_cli_flags_parse():
    parser = build_parser()
    ns = parser.parse_args([
        "serve",
        "--model", "demo",
        "--weight-format", "q1_cubes",
        "--q1-planes", "6",
        "--cube-root-bits", "8",
        "--lm-head-planes", "8",
        "--min-q1-planes", "4",
        "--max-q1-planes", "8",
        "--no-auto-fit-q1",
        "--q1-kl-target", "0.05",
        "--rssf", "off",
        "--no-token-ledger",
        "--session-id", "work",
    ])
    assert ns.q1_planes == 6
    assert ns.cube_root_bits == 8
    assert ns.lm_head_planes == 8
    assert ns.min_q1_planes == 4
    assert ns.max_q1_planes == 8
    assert ns.auto_fit_q1 is False
    assert ns.q1_kl_target == pytest.approx(0.05)
    assert ns.rssf == "off"
    assert ns.token_ledger is False
    assert ns.session_id == "work"


def test_live_set_active_q1_planes_and_kl_calibration_never_touches_embeddings():
    """Verify 0-ms live `set_active_q1_planes` and `calibrate_q1_planes_by_kl` on real `BinaryHalvingQ1Linear`."""
    torch.manual_seed(1234)
    vocab, hidden = 64, 64

    class _MiniQ1Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab, hidden)
            w1 = torch.randn((hidden, hidden), dtype=torch.bfloat16) * 0.15
            t1, m1, _, _ = carve_trunk_mirror_q1(w1, 8, min_cubes=8, mirror_tol=0.0, root_bits=8, group_size=64, use_fwht=True)
            self.proj = BinaryHalvingQ1Linear(hidden, hidden, t1, m1)
            w2 = torch.randn((vocab, hidden), dtype=torch.bfloat16) * 0.15
            t2, m2, _, _ = carve_trunk_mirror_q1(w2, 8, min_cubes=8, mirror_tol=0.0, root_bits=8, group_size=64, use_fwht=True)
            self.lm_head = BinaryHalvingQ1Linear(hidden, vocab, t2, m2)

        def forward(self, input_ids, use_cache=False):
            x = self.embed_tokens(input_ids).to(torch.bfloat16)
            h = self.proj(x)
            logits = self.lm_head(h)
            return SimpleNamespace(logits=logits)

    model = _MiniQ1Model().eval()
    emb_snapshot = model.embed_tokens.weight.detach().clone()

    # Bind methods directly onto a lightweight mock engine instance
    eng = object.__new__(QuintessentiaUnifiedEngine)
    eng.profile = {"weight_format": "q1_cubes", "q1_planes": 8}
    eng.q1_planes = 8
    eng.q1_planes_loaded = 8
    eng.cube_root_bits = 8
    eng.q1_lm_head_planes = 8
    eng.min_q1_planes = 3
    eng._q1_cubes_runner = None
    eng.engine = SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        tokenizer=lambda text, return_tensors=None: SimpleNamespace(
            input_ids=torch.tensor([[i % vocab for i in range(24)]], dtype=torch.long)
        ),
    )

    # 1. Live 0-ms switching
    res = eng.set_active_q1_planes(4, lm_head_k=6)
    assert res["q1_planes"] == 4
    assert res["lm_head_planes"] == 6
    assert model.proj.active_cubes == 4
    assert model.lm_head.active_cubes == 6

    with pytest.raises(ValueError):
        eng.set_active_q1_planes(9)  # exceeds loaded=8

    # 2. KL calibration picks minimal k <= target and leaves embed_tokens 100% untouched
    cal = eng.calibrate_q1_planes_by_kl(kl_target=0.05, min_k=3)
    assert cal["ok"] is True
    assert 3 <= cal["chosen_k"] <= 8
    assert cal["chosen_kl"] <= 0.05 or cal["chosen_k"] == 8
    assert eng.q1_planes == cal["chosen_k"]
    assert torch.equal(model.embed_tokens.weight, emb_snapshot)


def test_lan_server_api_q1_and_compact_endpoints_live():
    """Verify POST /api/q1, POST /api/compact, and GET /api/info Q1 fields on a live LAN HTTP server."""

    class _FakeEngine:
        model_id = "demo-q1"
        prefill_chunk_tokens = 512
        q1_planes = 8
        q1_planes_loaded = 8
        cube_root_bits = 8
        enable_rssf_balance = True
        calibration = None

        def status_snapshot(self):
            return {
                "model_id": self.model_id,
                "backend": "unified-q1-cubes",
                "weight_format": "q1_cubes",
                "q1_planes": self.q1_planes,
                "q1_planes_loaded": self.q1_planes_loaded,
                "cube_root_bits": self.cube_root_bits,
                "rssf_balance_enabled": self.enable_rssf_balance,
                "gpu_name": "MockGPU",
                "vram_mode": "normal",
                "mtp_enabled": False,
                "allocated_vram_mib": 1234.0,
                "last_metrics": {},
            }

        def set_active_q1_planes(self, k: int, *, lm_head_k=None):
            if not 1 <= k <= self.q1_planes_loaded:
                raise ValueError("out of range")
            self.q1_planes = k
            return {"q1_planes": k, "q1_planes_loaded": self.q1_planes_loaded, "cube_root_bits": 8}

        def compact_session(self, summary: str, **kwargs):
            return {"closed_epoch": 0, "new_epoch": 1, "old_tokens": 120, "new_tokens": 24}

        def reset_session(self):
            pass

    srv = QuintessentiaLANServer(_FakeEngine(), host="127.0.0.1", http_port=0, enable_beacon=False, log_file=None)
    srv.start_background()
    try:
        base = f"http://127.0.0.1:{srv.http_port}"
        with urlopen(f"{base}/api/info", timeout=5) as resp:
            info = json.loads(resp.read().decode("utf-8"))
        assert info["q1_planes"] == 8
        assert info["q1_planes_loaded"] == 8
        assert info["cube_root_bits"] == 8

        # Switch live K via POST /api/q1
        req1 = Request(f"{base}/api/q1", data=json.dumps({"q1_planes": 4}).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req1, timeout=5) as resp:
            q1_res = json.loads(resp.read().decode("utf-8"))
        assert q1_res["ok"] is True and q1_res["q1_planes"] == 4

        # Compact session via POST /api/compact
        req2 = Request(f"{base}/api/compact", data=json.dumps({"summary": "Short summary"}).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req2, timeout=5) as resp:
            c_res = json.loads(resp.read().decode("utf-8"))
        assert c_res["ok"] is True and c_res["new_epoch"] == 1
    finally:
        srv.stop()


def test_posix_pty_arrow_keys_never_trigger_false_esc():
    """Regression test: buffered sys.stdin.read(1)+select caused arrow keys to return ESC and exit under WSL2/ConPTY."""
    import os
    import threading
    import time

    if os.name == "nt":
        pytest.skip("POSIX PTY test runs on Linux/WSL2")

    import pty
    import tty

    from quintessentia_cli.tui import _read_posix_key_from_fd

    master_fd, slave_fd = pty.openpty()
    try:
        tty.setraw(slave_fd)

        # 1. Burst CSI arrows (exact ConPTY/WSL2 pattern that broke buffered sys.stdin)
        os.write(master_fd, b"\x1b[A\x1b[B\x1b[C\x1b[D")
        assert [_read_posix_key_from_fd(slave_fd, esc_timeout=0.08) for _ in range(4)] == [
            "UP", "DOWN", "RIGHT", "LEFT"
        ]

        # 2. SS3 application-cursor arrows + modified arrow + PgUp/PgDn
        os.write(master_fd, b"\x1bOA\x1bOB\x1bOC\x1bOD\x1b[1;5C\x1b[5~\x1b[6~")
        assert [_read_posix_key_from_fd(slave_fd, esc_timeout=0.08) for _ in range(7)] == [
            "UP", "DOWN", "RIGHT", "LEFT", "RIGHT", "PGUP", "PGDN"
        ]

        # 3. Unknown CSI (focus event + SGR mouse) must return "" (ignored), NEVER "ESC"
        os.write(master_fd, b"\x1b[I\x1b[<0;12;5M")
        assert _read_posix_key_from_fd(slave_fd, esc_timeout=0.08) == ""
        assert _read_posix_key_from_fd(slave_fd, esc_timeout=0.08) == ""

        # 4. Split arrow across ConPTY boundary (ESC arrives 30ms before "[B")
        def _send_split():
            os.write(master_fd, b"\x1b")
            time.sleep(0.03)
            os.write(master_fd, b"[B")

        t = threading.Thread(target=_send_split)
        t.start()
        assert _read_posix_key_from_fd(slave_fd, esc_timeout=0.12) == "DOWN"
        t.join(timeout=1.0)

        # 5. Standalone ESC returns "ESC"
        os.write(master_fd, b"\x1b")
        assert _read_posix_key_from_fd(slave_fd, esc_timeout=0.05) == "ESC"

        # 6. Multi-byte UTF-8 Cyrillic input
        os.write(master_fd, "П".encode("utf-8"))
        assert _read_posix_key_from_fd(slave_fd, esc_timeout=0.05) == "П"
    finally:
        os.close(master_fd)
        os.close(slave_fd)


def test_server_console_logger_info_and_console_hotkeys():
    """Verify server console logger emits INFO+ and console hotkeys (M/Q/Esc stop, 1..9 switch Q1 K)."""
    import logging
    from quintessentia_cli.lan_server import setup_server_logger

    log = setup_server_logger(None)
    stream_handlers = [h for h in log.handlers if isinstance(h, logging.StreamHandler)]
    assert stream_handlers and any(h.level <= logging.INFO for h in stream_handlers)

    class _FakeEng:
        q1_planes = 5
        q1_planes_loaded = 8

        def set_active_q1_planes(self, k: int, *, lm_head_k=None):
            self.q1_planes = k
            return {"q1_planes": k, "q1_planes_loaded": self.q1_planes_loaded, "updated_modules": 10}

    eng = _FakeEng()
    srv = QuintessentiaLANServer(eng, host="127.0.0.1", http_port=0, enable_beacon=False, log_file=None)
    assert srv._handle_console_key("4") is False
    assert eng.q1_planes == 4
    assert srv._handle_console_key("m") is True
    assert srv._handle_console_key("Q") is True
    assert srv._handle_console_key("\x1b") is True


def test_bilingual_en_ru_cli_tui_and_i18n(tmp_path: Path, monkeypatch):
    """Verify full English and Russian CLI/TUI localization, `--lang` parsing, and priority resolution."""
    from quintessentia_cli.i18n import (
        extract_lang_from_argv,
        normalize_lang,
        resolve_cli_lang,
        set_lang,
    )
    from quintessentia_cli.lan_server import WEB_UI_HTML

    assert normalize_lang("en") == "en"
    assert normalize_lang("en_US.UTF-8") == "en"
    assert normalize_lang("ru") == "ru"
    assert normalize_lang("ru_RU.UTF-8") == "ru"
    assert normalize_lang("русский") == "ru"
    assert normalize_lang("auto") is None

    assert extract_lang_from_argv(["--lang", "ru", "chat"]) == "ru"
    assert extract_lang_from_argv(["chat", "--lang=en"]) == "en"
    assert extract_lang_from_argv(["--lang", "ru", "chat", "--lang", "en"]) == "en"
    assert extract_lang_from_argv(["ask", "--", "--lang", "ru"]) is None

    # Resolution priority: argv > env > cli_config.json
    cfg_path = tmp_path / "cli_config.json"
    cfg_path.write_text(json.dumps({"language": "ru"}), encoding="utf-8")
    monkeypatch.delenv("QUINTESSENTIA_LANG", raising=False)
    assert resolve_cli_lang([], config_path=cfg_path) == "ru"
    monkeypatch.setenv("QUINTESSENTIA_LANG", "en")
    assert resolve_cli_lang([], config_path=cfg_path) == "en"
    assert resolve_cli_lang(["--lang", "ru", "doctor"], config_path=cfg_path) == "ru"

    # Parser builds clean EN and RU help and accepts `--lang` before or after subcommands
    p_en = build_parser(lang="en")
    p_ru = build_parser(lang="ru")
    help_en = p_en.format_help()
    help_ru = p_ru.format_help()
    assert "Interactive button-driven terminal menu" in help_en
    assert "Интерактивное кнопочное меню" in help_ru
    assert "Язык интерфейса" in help_ru

    for argv in (
        ["--lang", "ru", "chat"],
        ["chat", "--lang", "ru"],
        ["--lang", "en", "menu"],
        ["menu", "--lang", "en"],
        ["serve", "--lang", "ru"],
        ["doctor", "--lang", "en"],
        ["models", "--lang", "ru"],
        ["hud", "--lang", "en", "--status"],
    ):
        ns = p_en.parse_args(argv)
        assert ns.lang in ("en", "ru")

    # TUI rows render cleanly in both EN and RU and persist language in cli_config.json
    demo_models = [{"name": "demo", "format": "GGUF", "size_gib": 4.2}]
    cfg_en = TuiConfig(language="en", model="demo")
    rows_en = _build_menu_rows(cfg_en, demo_models)
    labels_en = {r["id"]: r["label"] for r in rows_en if r["id"] != "section"}
    assert labels_en["model"] == "Model"
    assert "Launch Interactive Chat" in labels_en["act_chat"]
    assert "[ ON ]" in next(r["value"] for r in rows_en if r["id"] == "token_ledger")
    assert "--lang" in cfg_en.build_argv("chat") and "en" in cfg_en.build_argv("chat")

    cfg_ru = TuiConfig(language="ru", model="demo")
    rows_ru = _build_menu_rows(cfg_ru, demo_models)
    labels_ru = {r["id"]: r["label"] for r in rows_ru if r["id"] != "section"}
    assert labels_ru["model"] == "Модель"
    assert "Запустить интерактивный Чат" in labels_ru["act_chat"]
    assert "[ ВКЛ ]" in next(r["value"] for r in rows_ru if r["id"] == "token_ledger")
    assert "--lang" in cfg_ru.build_argv("serve") and "ru" in cfg_ru.build_argv("serve")

    cfg_ru.save(cfg_path)
    assert TuiConfig.load(cfg_path).language == "ru"

    # Built-in LAN Web UI includes EN/RU switcher and both dictionaries
    assert 'id="langBtn"' in WEB_UI_HTML
    assert "Ask the local LAN model..." in WEB_UI_HTML
    assert "Напиши запрос модели в локальной сети..." in WEB_UI_HTML
    set_lang("en")
