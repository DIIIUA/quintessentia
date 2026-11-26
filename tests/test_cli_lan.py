"""CPU-only checks for the CLI streaming splitter and LAN helpers (no GPU, no model)."""
import json
import socket

from quintessentia_cli.cli import build_parser
from quintessentia_cli.lan_server import (
    DISCOVER_MAGIC,
    UDPLANBeacon,
    _subnet_broadcast_targets,
    detect_lan_ips,
    format_connection_passport,
)
from quintessentia_cli.unified_runtime import _IncrementalThinkSplitter, quintessentia_spec_lookup


def _run_splitter(chunks, start_in_thinking=True):
    sp = _IncrementalThinkSplitter(start_in_thinking=start_in_thinking)
    out = []
    for c in chunks:
        out.extend(sp.push(c))
    out.extend(sp.push("", final=True))
    thinking = "".join(t for p, t in out if p == "thinking")
    answer = "".join(t for p, t in out if p == "answer")
    return thinking, answer


def test_think_splitter_handles_tag_split_across_chunks():
    thinking, answer = _run_splitter(["Reason", "ing </th", "ink>\n\nFinal", " answer"])
    assert thinking == "Reasoning "
    assert answer == "Final answer"


def test_think_splitter_without_thinking_phase():
    thinking, answer = _run_splitter(["Direct ", "answer"], start_in_thinking=False)
    assert thinking == ""
    assert answer == "Direct answer"


def test_spec_lookup_proposes_tokens_after_repeated_suffix():
    history = [1, 2, 3, 4, 9, 9, 1, 2, 3]
    assert quintessentia_spec_lookup(history, max_drafts=2) == [4, 9]
    assert quintessentia_spec_lookup([1, 2], max_drafts=2) == []


def test_lan_helpers_and_passport():
    primary, all_ips = detect_lan_ips()
    assert primary in all_ips
    targets = _subnet_broadcast_targets(["192.168.1.219"])
    assert "255.255.255.255" in targets and "192.168.1.255" in targets
    text = format_connection_passport(
        {
            "hostname": "node",
            "gpu_name": "GPU",
            "vram_mode": "8gb",
            "model_id": "model",
            "local_web_url": "http://127.0.0.1:8787",
            "lan_web_url": "http://192.168.1.2:8787",
            "openai_base_url": "http://192.168.1.2:8787/v1",
            "api_key": "quintessentia-local",
            "require_auth": False,
            "lan_ip": "192.168.1.2",
            "all_lan_ips": ["192.168.1.2"],
            "http_port": 8787,
            "beacon_port": 8788,
        }
    )
    assert "http://192.168.1.2:8787/v1" in text and "quintessentia-local" in text


def test_udp_beacon_answers_discovery_probe():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tmp:
        tmp.bind(("127.0.0.1", 0))
        port = tmp.getsockname()[1]
    beacon = UDPLANBeacon(lambda: {"service": "quintessentia", "http_port": 8787}, ["127.0.0.1"], beacon_port=port, interval_s=60)
    beacon.start()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as cli:
            cli.settimeout(3.0)
            cli.sendto(DISCOVER_MAGIC, ("127.0.0.1", port))
            while True:
                data, _ = cli.recvfrom(4096)
                info = json.loads(data.decode("utf-8"))
                if info.get("service") == "quintessentia":
                    break
        assert info["http_port"] == 8787
    finally:
        beacon.stop()


def test_cli_parser_commands():
    parser = build_parser()
    args = parser.parse_args(["serve", "--port", "9000", "--require-auth"])
    assert args.command == "serve" and args.port == 9000 and args.require_auth
    args = parser.parse_args(["ask", "--no-think", "hello", "world"])
    assert args.prompt == ["hello", "world"] and args.no_think
    args = parser.parse_args(["bench", "--model", "1", "--runs", "3", "--max-tokens", "64", "--no-think"])
    assert args.command == "bench" and args.runs == 3 and args.max_tokens == 64 and args.no_think
    args = parser.parse_args(["hud", "--status"])
    assert args.command == "hud" and args.status is True
    args = parser.parse_args(["hud", "--stop"])
    assert args.command == "hud" and args.stop is True
    args = parser.parse_args(["hud", "--install-desktop"])
    assert args.command == "hud" and args.install_desktop is True


def test_hud_backend_rssf_extract_and_standby_env():
    from quintessentia_cli.hud import hud_backend

    extracted = hud_backend._extract_rssf_fields_from_metrics({
        "kv_gpu_layers": 2,
        "kv_offloaded_layers": 6,
        "kv_promoted_layers": 1,
        "rssf_generation_balance": {
            "entry_balance": {"d3_margin": 2.22},
            "entry_collapse": {"loss_of_archive": False},
            "turn_end_balance": {"d3_margin": 3.75, "regime": "EXPAND"},
            "prefill_damped_chunks": 2,
            "prefill_scale_breaks": 0,
            "active_scales": {
                "m0_trunk_q1": {},
                "m3_cold_warmup": {},
                "m6_probability_path": {},
                "kv_placement": {},
            },
        },
    })
    assert extracted["kv_gpu_layers"] == 2 and extracted["kv_offloaded_layers"] == 6
    assert extracted["rssf_d3_margin"] == 3.75 and extracted["rssf_regime"] == "EXPAND"
    assert extracted["rssf_active_scales"] == ["M0", "M3", "M6", "KV"]

    standby_win = hud_backend.read_live_telemetry(
        {"telemetry_file": "/nonexistent/quint_missing_12345.json"}, wsl_running=False
    )
    assert standby_win["env"] == "Windows" or standby_win["source"] == "live_json"

