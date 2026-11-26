"""Quintessentia command-line interface.

Commands:
  models    List local models; select by number, name or path
  download  Download a Hugging Face URL/repository and verify adaptation
  prepare   Verify automatic runtime adaptation (Q1 cubes V4 on CUDA) with a first forward
  doctor    Check GPU, VRAM plan, model files, caches and LAN addresses
  ask       Stream a single prompt (`<think>...</think>` -> answer -> `<|im_end|>`)
  chat      Interactive multi-turn chat (KV-prefix reuse + pinned host-RAM parking)
  serve     Web chat + OpenAI-compatible API (TCP :8787) and UDP LAN beacon (:8788)
  discover  Find Quintessentia nodes on the local network

Exit codes: 0 OK, 1 runtime error, 2 usage error, 3 model not found / unsupported.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import sys

# Ensure UTF-8 stdout/stderr on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from quintessentia_cli.i18n import (
    SUPPORTED_LANGS,
    get_lang,
    normalize_lang,
    resolve_cli_lang,
    set_lang,
    tr,
)
from quintessentia_cli.lan_server import (
    DEFAULT_BEACON_PORT,
    DEFAULT_HTTP_PORT,
    DEFAULT_LOG_FILE,
    DEFAULT_MAX_REQUEST_BYTES,
    QuintessentiaLANServer,
    detect_lan_ips,
    discover_lan_servers,
    format_connection_passport,
)
from quintessentia_cli.model_source import (
    DEFAULT_MODEL_ID,
    ModelNotFoundError,
    UnsupportedModelError,
    detect_format,
    resolve_model_dir,
)
from quintessentia_cli.runtime_config import (
    VRAM_MODES,
    GenerationCancelled,
    checkpoint_manifest_path,
    find_quintessentia_root,
)

EXIT_OK, EXIT_RUNTIME, EXIT_USAGE, EXIT_MODEL = 0, 1, 2, 3
WEIGHT_FORMAT_CHOICES = ("auto", "q1_cubes", "q1", "q2", "q4", "q6", "q8", "bf16")
FEEDBACK_MODE_CHOICES = ("auto", "off", "spiral_sphere", "pure_mirror", "mirror_scales", "v4")
PROFILE_MODE_CHOICES = ("default", "custom")


def resolve_generation_profile(
    mode: str = "default",
    *,
    thinking: bool = True,
    max_tokens: int = 0,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    presence_penalty: float | None = None,
    seed: int | None = None,
) -> dict:
    """Resolve user-facing `default` vs `custom` generation & anti-loop sampling profile.

    - `default`: out-of-the-box user mode — unlimited tokens (`0` -> natural EOS) and
      anti-loop sampling tuned separately for `<think>` (stops greedy draft-rewriting loops)
      and `--no-think` (crisp, direct answers). Any explicitly passed CLI flag overrides.
    - `custom`: full manual control — zero hidden sampling overrides (deterministic greedy
      unless the user passes `--temperature`, `--top-p`, `--top-k`, `--presence-penalty`, etc.).
    """
    prof = str(mode or "default").strip().lower()
    if prof not in PROFILE_MODE_CHOICES:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {PROFILE_MODE_CHOICES}")

    if prof == "default":
        if thinking:
            base = dict(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                repetition_penalty=1.05,
                presence_penalty=1.2,
            )
        else:
            base = dict(
                temperature=0.35,
                top_p=0.9,
                top_k=20,
                repetition_penalty=1.03,
                presence_penalty=0.5,
            )
    else:
        base = dict(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            presence_penalty=0.0,
        )

    return {
        "mode": prof,
        "max_new_tokens": int(max_tokens),
        "temperature": float(base["temperature"] if temperature is None else temperature),
        "top_p": float(base["top_p"] if top_p is None else top_p),
        "top_k": int(base["top_k"] if top_k is None else top_k),
        "repetition_penalty": float(base["repetition_penalty"] if repetition_penalty is None else repetition_penalty),
        "presence_penalty": float(base["presence_penalty"] if presence_penalty is None else presence_penalty),
        "seed": seed,
    }


def _model_weight_bytes(folder: Path) -> int:
    from .model_source import runtime_model_dir

    target = runtime_model_dir(folder)
    return sum(
        f.stat().st_size
        for f in target.iterdir()
        if f.suffix in (".gguf", ".safetensors", ".bin") and f.is_file()
    )


def resolve_auto_v4_plan(
    *,
    model_dir: Path | None,
    weight_format: str = "auto",
    feedback_mode: str = "auto",
    q1_planes: int | None = None,
    device: str = "auto",
    cuda_available: bool | None = None,
    free_vram_bytes: int | None = None,
    profile: dict | None = None,
    weight_bytes: int | None = None,
) -> dict:
    """Resolve runtime weight format and V4 isolated mirror-scales activation for CLI/doctor.

    Rules:
      - Explicit `--feedback-mode off`: never forces V4 or `q1_cubes` (keeps legacy `auto` -> `q4`/`q6`).
      - Explicit `--feedback-mode v4|mirror_scales|pure_mirror|spiral_sphere`: promotes
        `weight_format=auto` -> `q1_cubes`, enables `q1_imprint_puzzle` for MoE/unknown, and picks
        `q1_planes=5`, `cube_root_bits=8` unless overridden.
      - `--feedback-mode auto` + `--weight-format q1_cubes`: enables V4 (`mirror_scales`,
        `M0-M8`) automatically.
      - `--feedback-mode auto` + `--weight-format auto` on CUDA: promotes ALL CUDA models
        (dense and MoE alike) to pure Q1 cubes V4 (`q1_cubes` + `mirror_scales` + `q1_planes=5`
        + `cube_root_bits=8`; enables `q1_imprint_puzzle=True` when `num_experts >= 2` or model
        metadata is not yet local). CPU execution (`--device cpu` or no CUDA) keeps standard `auto`.
    """
    from src.stanok_contour import normalize_feedback_mode

    wf = str(weight_format or "auto").strip().lower()
    fb_raw = str(feedback_mode or "auto").strip().lower()

    if fb_raw == "off":
        return {
            "weight_format": wf,
            "feedback_mode": "off",
            "q1_imprint_puzzle": False,
            "q1_planes": q1_planes,
            "cube_root_bits": 16,
            "auto_v4_selected": False,
            "auto_fit_q1_planes": (wf == "q1_cubes" and q1_planes is None),
            "reason": "explicit_off",
        }

    if fb_raw != "auto":
        fb_norm = normalize_feedback_mode(fb_raw)
        if wf == "auto":
            wf = "q1_cubes"
        eff_planes = 5 if (q1_planes is None and wf == "q1_cubes") else q1_planes
        eff_root = 16 if (eff_planes is not None and int(eff_planes) > 8) else 8
        return {
            "weight_format": wf,
            "feedback_mode": fb_norm,
            "q1_imprint_puzzle": (wf == "q1_cubes"),
            "q1_planes": eff_planes,
            "cube_root_bits": eff_root if wf == "q1_cubes" else 16,
            "auto_v4_selected": (fb_norm == "mirror_scales"),
            "auto_fit_q1_planes": (wf == "q1_cubes" and q1_planes is None),
            "reason": f"explicit_{fb_norm}",
        }

    # feedback_mode == "auto"
    if wf == "q1_cubes":
        eff_planes = 5 if q1_planes is None else int(q1_planes)
        eff_root = 16 if eff_planes > 8 else 8
        num_exp = int((profile or {}).get("num_experts") or 0)
        if profile is None and model_dir is not None and model_dir.exists():
            try:
                from .model_source import runtime_profile
                num_exp = int((runtime_profile(model_dir) or {}).get("num_experts") or 0)
            except Exception:
                num_exp = 0
        return {
            "weight_format": "q1_cubes",
            "feedback_mode": "mirror_scales",
            "q1_imprint_puzzle": (num_exp >= 2 or model_dir is None),
            "q1_planes": eff_planes,
            "cube_root_bits": eff_root,
            "auto_v4_selected": True,
            "auto_fit_q1_planes": (q1_planes is None),
            "reason": "auto_q1_cubes_v4",
        }

    if wf != "auto" or device == "cpu":
        return {
            "weight_format": wf,
            "feedback_mode": "off",
            "q1_imprint_puzzle": False,
            "q1_planes": q1_planes,
            "cube_root_bits": 16,
            "auto_v4_selected": False,
            "auto_fit_q1_planes": False,
            "reason": "explicit_non_q1_or_cpu",
        }

    # Default CUDA auto: promote all CUDA checkpoints to pure Q1 cubes V4 (k=5, root_bits=8)
    if cuda_available is None:
        try:
            import torch
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
    if not cuda_available:
        return {
            "weight_format": wf,
            "feedback_mode": "off",
            "q1_imprint_puzzle": False,
            "q1_planes": q1_planes,
            "cube_root_bits": 16,
            "auto_v4_selected": False,
            "reason": "no_cuda",
        }

    eff_planes = 5 if q1_planes is None else int(q1_planes)
    eff_root = 16 if eff_planes > 8 else 8
    has_local_model = model_dir is not None and Path(model_dir).exists()

    prof = profile
    if prof is None and has_local_model:
        try:
            from .model_source import runtime_profile
            prof = runtime_profile(model_dir)
        except Exception:
            prof = {}
    num_experts = int((prof or {}).get("num_experts") or 0)

    if num_experts >= 8 and has_local_model:
        wb = weight_bytes
        if wb is None:
            try:
                wb = _model_weight_bytes(Path(model_dir))
            except Exception:
                wb = 0

        free_b = free_vram_bytes
        if free_b is None:
            try:
                from .gpu_memory import measure_gpu_memory
                free_b = int(measure_gpu_memory(0).get("free_bytes") or 0)
            except Exception:
                free_b = 0

        if free_b > 0 and wb > int(1.25 * free_b):
            return {
                "weight_format": "q1_cubes",
                "feedback_mode": "mirror_scales",
                "q1_imprint_puzzle": True,
                "q1_planes": eff_planes,
                "cube_root_bits": eff_root,
                "auto_v4_selected": True,
                "auto_fit_q1_planes": (q1_planes is None),
                "reason": "auto_large_moe_exceeds_vram",
                "num_experts": num_experts,
                "weight_gib": round(wb / 2**30, 2),
                "free_vram_gib": round(free_b / 2**30, 2),
            }

    return {
        "weight_format": "q1_cubes",
        "feedback_mode": "mirror_scales",
        "q1_imprint_puzzle": (num_experts >= 2 or not has_local_model),
        "q1_planes": eff_planes,
        "cube_root_bits": eff_root,
        "auto_v4_selected": True,
        "auto_fit_q1_planes": (q1_planes is None),
        "reason": "auto_q1_cubes_v4",
    }


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        "--model-dir",
        dest="model",
        type=str,
        default=None,
        help=f"Catalog number/name, local HF/GGUF/Quintessentia folder, or Hugging Face URL/repository "
        f"such as {DEFAULT_MODEL_ID} (downloaded once). Default: a locally recognized folder",
    )
    p.add_argument(
        "--quintessentia-root",
        dest="quintessentia_root",
        type=str,
        default=None,
        help="Path to the Quintessentia program folder",
    )
    p.add_argument("--download-dir", type=str, default=None, help="Save a Hugging Face model directly into this directory")
    p.add_argument("--device", choices=("auto", "cpu"), default="auto",
                   help="auto: CPU execution with optional GPU adoption; cpu: disk/CPU execution only")
    p.add_argument(
        "--vram-mode",
        choices=list(VRAM_MODES),
        default="normal",
        help="normal: almost all free VRAM with a safety reserve (default); turbo: all free VRAM",
    )
    p.add_argument("--max-context", type=int, default=None, help="KV memory-planning reservation; does not cap requests. Model architecture and available memory determine actual capacity.")
    p.add_argument("--prefill-chunk", type=int, default=None, help="Prompt tokens per prefill chunk (default 512, 256 on <=5 GB cards)")
    p.add_argument("--prefill-wavefront", action="store_true",
                   help="Experimental Llama prefill: overlap prompt chunks across three CUDA stages")
    p.add_argument("--no-mtp", action="store_true", help="Disable the speculative MTP head")
    p.add_argument("--speculation", choices=("auto", "off", "lookup", "native", "q1"), default="auto",
                   help="Shared streamed verification: native MTP head, prompt lookup, or Q1 2-plane self-speculation")
    p.add_argument("--mtp-head", type=str, default=None, help="Separate trained MTP safetensors for the selected model")
    p.add_argument("--mtp-head-budget-mib", type=int, default=512)
    p.add_argument("--region-pruning", action=argparse.BooleanOptionalAction, default=True,
                   help="Exclude proven zero-weight routed experts and cancel obsolete current-scope preparation")
    p.add_argument("--thought-anchor", action=argparse.BooleanOptionalAction, default=False,
                   help="Retain initial probability field Phi_0 with bounded log-odds steering across generation")
    p.add_argument("--max-slot-mib", type=float, default=None,
                   help="Cap per-slot VRAM budget in MiB (enables Q1 sub-layer / sub-lm_head cube streaming when below a layer or lm_head)")
    p.add_argument("--progressive-start", "--cpu-first", action=argparse.BooleanOptionalAction, default=True,
                   help="Start CPU and GPU preparation together; devices take work when ready (default); --no-progressive-start selects the GPU-first reference")
    p.add_argument("--weight-format", choices=WEIGHT_FORMAT_CHOICES, default="auto",
                   help="Runtime weights: auto selects pure Q1 cubes V4 (k=5) on CUDA or standard CPU path; explicit q1_cubes/q1/q2/q4/q6/q8/bf16")
    p.add_argument("--q1-planes", type=int, choices=tuple(range(1, 17)), default=None,
                   help="Explicit prefix depth k (1..16); auto V4 / q1_cubes defaults to auto-fit from 5 planes")
    p.add_argument("--cube-root-bits", type=int, choices=(2, 4, 8, 16), default=None,
                   help="Explicit Binary Halving Q1 tree root width (2, 4, 8, or 16; default: auto from k)")
    p.add_argument("--lm-head-planes", type=int, choices=tuple(range(1, 17)), default=None,
                   help="Explicit Q1 bit-plane depth for lm_head independently of trunk layers (1..16)")
    p.add_argument("--min-q1-planes", type=int, choices=tuple(range(1, 17)), default=3,
                   help="Lower bound for auto-fit Q1 depth step-down (default: 3)")
    p.add_argument("--max-q1-planes", type=int, choices=tuple(range(1, 17)), default=8,
                   help="Upper bound for auto-fit Q1 depth step-up (default: 8)")
    p.add_argument("--auto-fit-q1", action=argparse.BooleanOptionalAction, default=None,
                   help="Enable/disable automatic VRAM-based Q1 depth fitting (default: auto when --q1-planes is omitted)")
    p.add_argument("--q1-kl-target", type=float, default=None,
                   help="Auto-calibrate smallest Q1 k whose mean KL vs loaded max planes stays <= target (e.g. 0.05)")
    p.add_argument("--cube-blueprint", help="Method 1 JSON map selecting Method 2 cube prefixes per projection")
    p.add_argument("--q1-early-stop", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable dynamic Plane-0 vocabulary early-stop on Ideal Q1 LM head (default: on)")
    p.add_argument("--puzzle-overlay", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable dynamic 3D Q1PuzzleOverlay (Ideal Q1 P=1 skeleton + contact-assembled delta cubes)")
    p.add_argument("--puzzle-slice-ratio", type=float, default=0.35,
                   help="Fraction of delta-cubes selected per dynamic slice S_t when --puzzle-overlay is enabled (0.0..1.0, default: 0.35)")
    p.add_argument(
        "--mode",
        choices=PROFILE_MODE_CHOICES,
        default="default",
        help="User profile: default (auto V4 + unlimited EOS + anti-loop sampling) or custom (manual control over all sampling & engine flags)",
    )
    p.add_argument(
        "--feedback-mode",
        choices=FEEDBACK_MODE_CHOICES,
        default="auto",
        help="Q1 mirror feedback mode: auto (pure Q1 cubes V4 M0-M8 on CUDA), off (opt out to legacy q4/q6), spiral_sphere (V2), pure_mirror (V3), or mirror_scales/v4 (V4 isolated pure-math mirror scales)",
    )
    p.add_argument(
        "--rest-tol",
        type=float,
        default=None,
        help="Custom MoE cold-expert mirror rest tolerance (requires q1_cubes / V4; default: 0.24)",
    )
    p.add_argument("--max-drafts", type=int, default=3, help="Max speculative draft depth (1..3, default: 3)")
    p.add_argument(
        "--fast-vocab",
        action="store_true",
        help="Experimental vocabulary pruner (off by default: it drops rare tokens on long outputs)",
    )
    p.add_argument(
        "--hud",
        action="store_true",
        help="Open the live visual Quintessentia Desktop HUD window alongside CLI execution",
    )
    p.add_argument(
        "--rssf",
        choices=("on", "off"),
        default="on",
        help="Enable or disable the RSSF/EC/FDS dynamic balance operators for clean ablation (default: on)",
    )
    p.add_argument(
        "--token-ledger",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist disk-backed Token Ledger (archive + active window) for deterministic retokenization (default: enabled)",
    )
    p.add_argument(
        "--session-id",
        type=str,
        default="default",
        help="Disk Token Ledger session identifier (default: default)",
    )


def _add_generation_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Max new tokens (0 = unlimited until natural EOS, default: 0)",
    )
    p.add_argument("--no-think", action="store_true", help="Disable `<think>` reasoning block")
    p.add_argument("--system", type=str, default=None, help="Optional system prompt")
    p.add_argument("--temperature", type=float, default=None, help="Sampling temperature (overrides mode default; 0 = greedy)")
    p.add_argument("--top-p", dest="top_p", type=float, default=None, help="Nucleus top-p sampling cutoff (0..1)")
    p.add_argument("--top-k", dest="top_k", type=int, default=None, help="Top-k sampling cutoff (0 = disabled)")
    p.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=None, help="Repetition penalty on generated tokens (>=1.0)")
    p.add_argument("--presence-penalty", dest="presence_penalty", type=float, default=None, help="Presence penalty on generated tokens (>=0.0)")
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible sampling")


def _engine_from_args(args: argparse.Namespace) -> QuintessentiaUnifiedEngine:
    from quintessentia_cli.unified_runtime import QuintessentiaUnifiedEngine
    if getattr(args, "hud", False):
        launch_desktop_hud(quiet=True, bind_to_console=True)
    model = args.model
    root = find_quintessentia_root(args.quintessentia_root)
    if getattr(args, "download_dir", None):
        if not model:
            raise ModelNotFoundError("--download-dir requires a Hugging Face model reference")
        from .model_hub import download_model
        model, _, _ = download_model(model, root, directory=args.download_dir)
    explicit_repo = bool(model) and not Path(model).exists()
    wf_arg = str(getattr(args, "weight_format", "auto") or "auto").strip().lower()
    fb_arg = str(getattr(args, "feedback_mode", "auto") or "auto").strip().lower()
    dev_arg = str(getattr(args, "device", "auto") or "auto").strip().lower()
    is_local_ref = (
        (model is None)
        or Path(model).exists()
        or (isinstance(model, str) and ("/" not in model) and ("\\" not in model))
    )
    needs_local_probe = (
        is_local_ref
        and (dev_arg != "cpu")
        and (fb_arg != "off")
        and (wf_arg in ("auto", "q1_cubes"))
    )
    resolved_dir: Path | None = None
    if needs_local_probe:
        try:
            resolved_dir = resolve_model_dir(model, root, allow_download=False)
        except (ModelNotFoundError, UnsupportedModelError, OSError, ValueError):
            resolved_dir = None

    v4_plan = resolve_auto_v4_plan(
        model_dir=resolved_dir,
        weight_format=wf_arg,
        feedback_mode=fb_arg,
        q1_planes=getattr(args, "q1_planes", None),
        device=dev_arg,
    )
    eff_q1_planes = v4_plan["q1_planes"]
    eff_root_bits = int(v4_plan["cube_root_bits"])
    user_root = getattr(args, "cube_root_bits", None)
    if user_root is not None:
        eff_root_bits = int(user_root)
        if eff_q1_planes is not None and int(eff_q1_planes) > eff_root_bits:
            eff_q1_planes = eff_root_bits
    user_auto_fit = getattr(args, "auto_fit_q1", None)
    eff_auto_fit = bool(v4_plan.get("auto_fit_q1_planes", False)) if user_auto_fit is None else bool(user_auto_fit)
    kl_target = getattr(args, "q1_kl_target", None)
    max_q1_p = int(getattr(args, "max_q1_planes", 8) or 8)
    min_q1_p = int(getattr(args, "min_q1_planes", 3) or 3)
    if kl_target is not None and v4_plan["weight_format"] == "q1_cubes" and getattr(args, "q1_planes", None) is None:
        # Load at `max_q1_planes` (stepped down if needed by auto_fit) so KL calibration can pick minimal k!
        eff_q1_planes = max_q1_p
        eff_root_bits = max(eff_root_bits, 16 if max_q1_p > 8 else 8)
        eff_auto_fit = True
    if v4_plan.get("auto_v4_selected") and v4_plan.get("reason") == "auto_large_moe_exceeds_vram":
        print(
            f"[Quintessentia Auto-V4] Large MoE ({v4_plan.get('num_experts')} experts, "
            f"{v4_plan.get('weight_gib')} GiB > 1.25x {v4_plan.get('free_vram_gib')} GiB free VRAM) "
            f"-> weight_format={v4_plan['weight_format']}, feedback_mode={v4_plan['feedback_mode']} (M0-M8), "
            f"q1_planes={eff_q1_planes}, cube_root_bits={eff_root_bits}",
            file=sys.stderr,
            flush=True,
        )

    return QuintessentiaUnifiedEngine(
        model_dir=model,
        quintessentia_root=args.quintessentia_root,
        vram_mode=args.vram_mode,
        enable_mtp=not args.no_mtp,
        enable_vocab_pruner=args.fast_vocab,
        max_drafts=args.max_drafts,
        max_context_tokens=args.max_context,
        prefill_chunk_tokens=args.prefill_chunk,
        allow_download=explicit_repo,
        weight_format=v4_plan["weight_format"],
        prefill_wavefront=getattr(args, "prefill_wavefront", False),
        progressive_start=getattr(args, "progressive_start", getattr(args, "cpu_first", True)),
        device=getattr(args, "device", "auto"),
        speculation=getattr(args, "speculation", "auto"),
        mtp_head_path=getattr(args, "mtp_head", None),
        mtp_head_budget_mib=getattr(args, "mtp_head_budget_mib", 512),
        prune_regions=getattr(args, "region_pruning", True),
        thought_anchor=getattr(args, "thought_anchor", False),
        max_slot_mib=getattr(args, "max_slot_mib", None),
        q1_planes=eff_q1_planes,
        cube_root_bits=eff_root_bits,
        cube_blueprint=getattr(args, "cube_blueprint", None),
        q1_early_stop=getattr(args, "q1_early_stop", True),
        puzzle_overlay=getattr(args, "puzzle_overlay", False),
        puzzle_slice_ratio=getattr(args, "puzzle_slice_ratio", 0.35),
        q1_expert_rest_tol=(
            getattr(args, "rest_tol", None)
            if v4_plan["weight_format"] == "q1_cubes"
            else None
        ),
        q1_imprint_puzzle=bool(v4_plan["q1_imprint_puzzle"]),
        q1_graphify_feedback=v4_plan["feedback_mode"],
        auto_fit_q1_planes=eff_auto_fit,
        min_q1_planes=min_q1_p,
        max_q1_planes=max_q1_p,
        lm_head_planes=getattr(args, "lm_head_planes", None),
        q1_kl_target=kl_target,
        enable_rssf_balance=(str(getattr(args, "rssf", "on") or "on").strip().lower() != "off"),
        enable_token_ledger=bool(getattr(args, "token_ledger", True)),
        session_id=str(getattr(args, "session_id", "default") or "default"),
    )


def cmd_doctor(args: argparse.Namespace) -> int:
    import torch

    from quintessentia_cli.model_source import runtime_profile
    from quintessentia_cli.unified_runtime import plan_for_checkpoint

    root = find_quintessentia_root(args.quintessentia_root)
    lan_ip, all_ips = detect_lan_ips()
    cuda_ok = torch.cuda.is_available()
    cpu_execution = getattr(args, "device", "auto") == "cpu" or not cuda_ok
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok and not cpu_execution else "NONE"
    report: dict = {
        "cuda_available": cuda_ok,
        "gpu_name": gpu_name,
        "execution_device": "cpu" if cpu_execution else "auto",
        "quintessentia_root": str(root),
        "primary_lan_ip": lan_ip,
        "all_lan_ips": all_ips,
    }
    model_dir = None
    model_error = None
    try:
        explicit_repo = bool(args.model) and not Path(args.model).exists()
        if getattr(args, "download_dir", None):
            from .model_hub import download_model
            model_dir, _, _ = download_model(args.model, root, directory=args.download_dir)
        else:
            model_dir = resolve_model_dir(args.model, root, allow_download=explicit_repo)
    except (ModelNotFoundError, UnsupportedModelError, OSError, ValueError) as exc:
        model_error = str(exc)
    report["model_dir"] = str(model_dir) if model_dir else None
    report["model_error"] = model_error

    plan = None
    if model_dir is not None:
        fmt = detect_format(model_dir)
        report["model_format"] = fmt
        profile = runtime_profile(model_dir)
        report["runtime_profile"] = profile
        if fmt == "quintessentia" and profile.get("fast_qwen"):
            report["q4_cache_ready"] = (root / "data" / "q4-g64-cache").exists() and checkpoint_manifest_path(root).exists()
        if cuda_ok and not cpu_execution:
            from .gpu_memory import measure_gpu_memory
            report["startup_memory_probe"] = measure_gpu_memory(0)
            free_b, total_b = (report["startup_memory_probe"][k] for k in ("free_bytes", "total_bytes"))
            try:
                max_slot_mib = getattr(args, "max_slot_mib", None)
                extra_plan = {"max_slot_bytes": int(float(max_slot_mib) * 1024**2)} if max_slot_mib is not None else {}
                v4_doc = resolve_auto_v4_plan(
                    model_dir=model_dir,
                    weight_format=getattr(args, "weight_format", "auto"),
                    feedback_mode=getattr(args, "feedback_mode", "auto"),
                    q1_planes=getattr(args, "q1_planes", None),
                    device=getattr(args, "device", "auto"),
                    cuda_available=cuda_ok and not cpu_execution,
                    free_vram_bytes=free_b,
                    profile=profile,
                )
                report["auto_v4_plan"] = v4_doc
                wf_doc = v4_doc["weight_format"]
                q1_p_doc = v4_doc["q1_planes"] or (16 if wf_doc == "q1_cubes" else 4)
                if wf_doc in ("q1", "q1_cubes") or q1_p_doc != 4:
                    extra_plan["q1_planes"] = q1_p_doc
                if wf_doc == "q1_cubes":
                    extra_plan["auto_fit_q1_planes"] = bool(v4_doc.get("auto_fit_q1_planes", False))
                profile, plan, _has_mtp = plan_for_checkpoint(
                    model_dir,
                    mode=args.vram_mode,
                    total_bytes=total_b,
                    free_bytes=free_b,
                    enable_mtp=not getattr(args, "no_mtp", False),
                    weight_format=wf_doc,
                    prefill_wavefront=getattr(args, "prefill_wavefront", False),
                    **extra_plan,
                )
                report["runtime_profile"] = profile
                report["mtp_weights_present"] = bool(profile.get("mtp_weights"))
                report["vram_plan"] = plan.as_dict()
            except (RuntimeError, ValueError) as exc:
                report["vram_plan_error"] = str(exc)
                cpu_execution = True
                report["execution_device"] = "cpu"

    print("=" * 76)
    print("  QUINTESSENTIA DOCTOR — ENVIRONMENT & LAN READINESS")
    print("=" * 76)
    print(f"  CUDA / GPU         : {'OK' if cuda_ok else 'MISSING'} — {gpu_name}")
    print(f"  Execution          : {report['execution_device']}")
    print(f"  Program folder     : {root}")
    print(f"  Model              : {model_dir or 'NOT FOUND'} ({report.get('model_format', '-')})")
    if model_error:
        print(f"                       {model_error}")
    profile = report.get("runtime_profile") or {}
    v4_doc = report.get("auto_v4_plan")
    if profile:
        if profile.get("weight_format") == "q1_cubes":
            q1_p = profile.get("q1_planes") or (v4_doc or {}).get("q1_planes") or 5
            root_b = (v4_doc or {}).get("cube_root_bits") or (16 if int(q1_p) > 8 else 8)
            halved_note = (
                f", auto-halved from k={profile.get('q1_planes_requested')}"
                if profile.get("q1_planes_auto_halved")
                else ""
            )
            print(f"  Backend            : {profile.get('backend')} / q1_cubes (k={q1_p}{halved_note}, root_bits={root_b})")
        else:
            why = "; ".join(profile.get("why_not_fast") or [])
            fast_line = "on" if profile.get("fast_qwen") else f"off ({why})" if why else "off"
            print(f"  Backend            : {profile.get('backend')} / {profile.get('weight_format')} / fast {fast_line}")
    if v4_doc and v4_doc.get("auto_v4_selected"):
        print(
            f"  V4 Q1 Cubes        : ON ({v4_doc['feedback_mode']}, "
            f"q1_planes={v4_doc['q1_planes']}, root_bits={v4_doc['cube_root_bits']}, "
            f"reason={v4_doc['reason']})"
        )
    if "mtp_weights_present" in report:
        mtp_on = bool(profile.get("mtp_enabled"))
        present = "present" if report["mtp_weights_present"] else "absent"
        print(f"  MTP head           : weights {present}, {'on' if mtp_on else 'off'}")
    if profile.get("num_experts"):
        print(
            f"  Experts            : {profile.get('num_experts')} total, "
            f"{profile.get('num_experts_per_tok')} per token, "
            f"{profile.get('num_hidden_layers')} layers, hidden {profile.get('hidden_size')}"
        )
    if plan is not None:
        if plan.full_residency:
            layout = "all layers resident"
        elif profile.get("weight_format") == "q1_cubes" and v4_doc and v4_doc.get("auto_v4_selected"):
            layout = f"{plan.resident_layers}/{plan.layers_total} resident + V4 streamed tail"
        else:
            layout = f"{plan.resident_layers}/{plan.layers_total} resident + {plan.ring_slots}-slot ring"
        print(
            f"  VRAM plan ({plan.mode:6s}): total {plan.total_bytes / 2**30:.2f} GiB, free {plan.free_bytes / 2**30:.2f} GiB, "
            f"reserve {plan.reserve_bytes / 2**30:.2f} GiB -> {layout}, MTP {'on' if (plan.enable_mtp or profile.get('mtp_enabled')) else 'off'}"
        )
    elif "vram_plan_error" in report:
        print(f"  VRAM plan          : {report['vram_plan_error']}")
    print(f"  RSSF Balance       : ON (8-pair force balance + EC Volterra + FDS + Collapse)")
    hud_ps1 = Path(__file__).resolve().parent / "hud" / "quintessentia_hud.ps1"
    print(f"  Desktop HUD        : {'ready' if hud_ps1.is_file() else 'missing'} (`quint hud` / `--hud`)")
    print(f"  Detected LAN IPv4  : {lan_ip} (all: {', '.join(all_ips)})")
    print("=" * 76)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if model_dir is None:
        return EXIT_MODEL
    return EXIT_OK if cpu_execution or "vram_plan_error" not in report else EXIT_RUNTIME


def cmd_models(args):
    from .model_source import local_model_candidates, runtime_model_dir, is_supported
    root = find_quintessentia_root(args.quintessentia_root)
    rows = []
    for i, folder in enumerate(local_model_candidates(root), 1):
        row = dict(index=i, name=folder.name, path=str(folder.resolve()))
        try:
            target = runtime_model_dir(folder)
            row["runtime_path"] = str(target.resolve())
            row["format"] = detect_format(target) or "unadapted"
            row["weight_bytes"] = sum(f.stat().st_size for f in target.iterdir()
                                      if f.suffix in (".gguf", ".safetensors", ".bin") and f.is_file())
            row["status"] = "available" if row["weight_bytes"] else "incomplete"
            if args.check:
                row["metadata_supported"] = is_supported(target)
        except (OSError, ValueError) as exc:
            row.update(status="invalid", error=str(exc))
        rows.append(row)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(f"Models: {root/'models'}")
        for row in rows:
            checked = f" | metadata {'OK' if row['metadata_supported'] else 'needs adapter'}" if "metadata_supported" in row else ""
            print(f"{row['index']:>3}  {row['name']}  [{row.get('format','?')}, {row.get('weight_bytes',0)/2**30:.2f} GiB]{checked}")
        print('Select by number, folder name or full path: quint ask --model 1 "Hi"')
        print('Available files and metadata checks do not certify runtime support; see MODEL_SUPPORT.md.')
    return EXIT_OK


def _verify_preparation(folder, *, root, device="auto", weight_format="auto", vram_mode="normal",
                        feedback_mode="auto", q1_planes=None, q1_early_stop=True):
    import torch
    from .unified_runtime import QuintessentiaUnifiedEngine
    v4_plan = resolve_auto_v4_plan(
        model_dir=Path(folder),
        weight_format=weight_format,
        feedback_mode=feedback_mode,
        q1_planes=q1_planes,
        device=device,
    )
    with QuintessentiaUnifiedEngine(
        folder, quintessentia_root=root, device=device, weight_format=v4_plan["weight_format"],
        vram_mode=vram_mode, enable_mtp=False, speculation="off", verbose=False,
        max_context_tokens=512, prefill_chunk_tokens=8, q1_planes=v4_plan["q1_planes"],
        cube_root_bits=v4_plan["cube_root_bits"], q1_imprint_puzzle=bool(v4_plan["q1_imprint_puzzle"]),
        q1_graphify_feedback=v4_plan["feedback_mode"], q1_early_stop=q1_early_stop,
        auto_fit_q1_planes=bool(v4_plan.get("auto_fit_q1_planes", False)),
    ) as engine:
        ids = engine.tokenizer("Hi", return_tensors="pt").input_ids
        with torch.inference_mode():
            logits, cache = engine.engine._forward_step(ids, prefill=True, prefill_end=True, prefetch_next=False)
            finite = bool(torch.isfinite(logits).all())
        if not finite:
            raise RuntimeError("Preparation produced nonfinite logits")
        del logits, cache
        snapshot = engine.status_snapshot()
        return dict(status="ready", model_dir=str(folder), source_format=engine.model_format,
                    weight_format=engine.profile["weight_format"], adaptation="on_demand",
                    finite_first_forward=True, packed_projections=snapshot["packed_projections"],
                    execution_device=snapshot["execution_device"], startup_memory_probe=snapshot.get("startup_memory_probe"))


def cmd_prepare(args):
    root = find_quintessentia_root(args.quintessentia_root)
    folder = resolve_model_dir(args.model, root, allow_download=bool(args.model) and not Path(args.model).exists())
    report = _verify_preparation(folder, root=root, device=args.device, weight_format=args.weight_format,
                                 vram_mode=args.vram_mode, feedback_mode=getattr(args, "feedback_mode", "auto"),
                                 q1_planes=getattr(args, "q1_planes", None),
                                 q1_early_stop=getattr(args, "q1_early_stop", True))
    from .model_hub import register_model, save_json
    register_model(root, folder)
    if args.json_out:
        save_json(args.json_out, report)
    print(f"Ready: {folder} | {report['weight_format']} | first forward OK")
    return EXIT_OK


def cmd_download(args):
    from .model_hub import download_model, save_json
    root = find_quintessentia_root(args.quintessentia_root)
    folder, marker, receipt = download_model(args.reference, root, directory=args.directory,
                                             filename=args.file, revision=args.revision)
    try:
        if not args.no_verify:
            receipt["preparation"] = _verify_preparation(
                folder, root=root, device=args.device, weight_format=args.weight_format,
                vram_mode=args.vram_mode, feedback_mode=getattr(args, "feedback_mode", "auto"),
                q1_planes=getattr(args, "q1_planes", None),
                q1_early_stop=getattr(args, "q1_early_stop", True))
            receipt["status"] = "ready"
        save_json(marker, receipt)
        if args.json_out:
            save_json(args.json_out, receipt)
    except Exception as exc:
        receipt.update(status="downloaded_unverified", preparation_error=str(exc))
        save_json(marker, receipt)
        if args.json_out:
            save_json(args.json_out, receipt)
        raise
    print(f"{'Ready' if receipt['status']=='ready' else 'Downloaded (verification skipped)'}: {folder}")
    print(f'Next: quintessentia ask --model "{folder}" "Hi"')
    return EXIT_OK


def _stream_to_console(
    engine: QuintessentiaUnifiedEngine,
    messages: list[dict[str, str]],
    max_tokens: int,
    thinking: bool,
    use_session: bool,
    *,
    gen_profile: dict | None = None,
) -> dict:
    in_think_printed = False
    in_ans_printed = False
    done_packet: dict = {}
    gp = gen_profile or {}

    gen = engine.stream_chat(
        messages,
        max_new_tokens=int(gp.get("max_new_tokens", max_tokens)),
        enable_thinking=thinking,
        use_session_cache=use_session,
        stop_on_eos=True,
        temperature=float(gp.get("temperature", 0.0)),
        top_p=float(gp.get("top_p", 1.0)),
        top_k=int(gp.get("top_k", 0)),
        repetition_penalty=float(gp.get("repetition_penalty", 1.0)),
        presence_penalty=float(gp.get("presence_penalty", 0.0)),
        seed=gp.get("seed"),
    )
    try:
        for ev in gen:
            if ev["type"] == "delta":
                if ev["phase"] == "thinking":
                    if not in_think_printed:
                        print(tr("\n[Thinking <think>]\n", "\n[Размышление <think>]\n"), end="", flush=True)
                        in_think_printed = True
                    print(ev["text"], end="", flush=True)
                else:
                    if in_think_printed and not in_ans_printed:
                        print(tr("\n\n[/think -> Answer]\n", "\n\n[/think -> Ответ]\n"), end="", flush=True)
                    elif not in_ans_printed:
                        print(tr("\n[Answer]\n", "\n[Ответ]\n"), end="", flush=True)
                    in_ans_printed = True
                    print(ev["text"], end="", flush=True)
            elif ev["type"] == "done":
                done_packet = ev
    except KeyboardInterrupt:
        gen.close()
        print(tr("\n\n[Stopped]", "\n\n[Остановлено]"), flush=True)
        return {}
    finally:
        gen.close()

    m = done_packet.get("metrics", {})
    sg = m.get("spec_gate", {})
    ec = m.get("expert_cache") or {}
    if engine.mtp_enabled:
        acc_pct = sg.get("overall_accept_rate_pct", 0.0)
        accel_str = f" | MTP accept: {acc_pct}% ({m.get('avg_tokens_per_backbone_pass', 1.0):.2f} tok/pass)"
    elif ec.get("slots", 0) > 0:
        accel_str = (
            f" | Expert cache: {ec.get('hit_rate_pct', 0.0)}% "
            f"({ec.get('used_slots', 0)}/{ec.get('slots', 0)} slots)"
        )
    else:
        accel_str = ""
    reuse_str = f" | Prefix reused: {m.get('cached_prefix_tokens', 0)} tok" if m.get("cached_prefix_tokens") else ""
    park_str = f" | Parked RAM: {m.get('parked_session_ram_mib', 0.0)} MiB" if use_session else ""
    kv_g = int(m.get("kv_gpu_layers", 0) or 0)
    kv_o = int(m.get("kv_offloaded_layers", 0) or 0)
    kv_str = f" | KV: {kv_g}G/{kv_o}R" if (kv_g + kv_o) > 0 else ""
    rb = m.get("rssf_generation_balance") or {}
    tb = rb.get("turn_end_balance") or {}
    ec_rep = rb.get("entry_collapse") or {}
    bal_str = ""
    if tb.get("d3_margin") is not None:
        bal_str = f" | Balance: d3={tb.get('d3_margin')} ({tb.get('regime')})"
        if ec_rep.get("loss_of_archive"):
            bal_str += " [ARCHIVE_WIPE]"
    print(
        f"\n\n--- [ one answer: {m.get('decode_tok_per_s', 0.0):.2f} tok/s | "
        f"{m.get('generated_tokens', 0)} tok (EOS={m.get('stopped_by_eos')}) | "
        f"Prefill: {m.get('prefill_ms', 0.0):.1f} ms{reuse_str}{accel_str}{kv_str}{bal_str} | "
        f"Peak VRAM: {m.get('peak_vram_mib', 0.0):.1f} MiB{park_str} ] ---",
        flush=True,
    )
    return done_packet


def cmd_ask(args: argparse.Namespace) -> int:
    raw_parts = list(args.prompt or [])
    if raw_parts == ["-"] or (not raw_parts and not sys.stdin.isatty()):
        prompt_text = sys.stdin.read().strip()
    else:
        prompt_text = " ".join(raw_parts).strip()
    if not prompt_text:
        print(
            tr(
                'Error: please provide a prompt, e.g. `quint ask "Hello!"` or pipe via stdin (`-`)',
                'Ошибка: укажите запрос, например `quint ask "Привет!"` или передайте через stdin (`-`)',
            ),
            file=sys.stderr,
        )
        return EXIT_USAGE

    thinking = not args.no_think
    gen_profile = resolve_generation_profile(
        getattr(args, "mode", "default"),
        thinking=thinking,
        max_tokens=args.max_tokens,
        temperature=getattr(args, "temperature", None),
        top_p=getattr(args, "top_p", None),
        top_k=getattr(args, "top_k", None),
        repetition_penalty=getattr(args, "repetition_penalty", None),
        presence_penalty=getattr(args, "presence_penalty", None),
        seed=getattr(args, "seed", None),
    )
    tok_label = "unlimited->EOS" if gen_profile["max_new_tokens"] == 0 else f"max={gen_profile['max_new_tokens']}"
    print(
        f"[Quintessentia Mode: {gen_profile['mode']} | {tok_label} | think={'ON' if thinking else 'OFF'} | "
        f"temp={gen_profile['temperature']}, top_p={gen_profile['top_p']}, top_k={gen_profile['top_k']}, "
        f"presence={gen_profile['presence_penalty']}, rep={gen_profile['repetition_penalty']}]",
        file=sys.stderr,
        flush=True,
    )

    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt_text})

    with _engine_from_args(args) as eng:
        done_pkt = _stream_to_console(
            eng,
            messages=messages,
            max_tokens=args.max_tokens,
            thinking=thinking,
            use_session=False,
            gen_profile=gen_profile,
        )
        if args.json_out and done_pkt:
            Path(args.json_out).write_text(json.dumps(done_pkt, ensure_ascii=False, indent=2), encoding="utf-8")
    return EXIT_OK


def cmd_chat(args: argparse.Namespace) -> int:
    thinking = not args.no_think
    cur_mode = getattr(args, "mode", "default")
    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    with _engine_from_args(args) as eng:
        import os as _os
        if _os.environ.get("QUINTESSENTIA_SKIP_CALIBRATION", "").strip().lower() not in ("1", "true", "yes"):
            if hasattr(eng, "calibrate_hardware_limits"):
                eng.calibrate_hardware_limits(verbose=True)
        b_lbl = eng.profile.get("backend", "auto")
        wf_lbl = eng.profile.get("weight_format", "auto")
        k_lbl = (
            f" k={eng.q1_planes}/{getattr(eng, 'q1_planes_loaded', eng.q1_planes)} root={getattr(eng, 'cube_root_bits', 8)}"
            if wf_lbl == "q1_cubes"
            else ""
        )
        print(
            tr(
                "\n[Quintessentia Interactive Chat | mode={mode} | {backend} ({wf}{k})] "
                "Commands: `/k <1..max>`, `/compact <summary>`, `/reset`, `/think on|off`, `/mode default|custom`, "
                "`/lang en|ru`, `/balance`, `/hud`, `/status`, `/exit`. Ctrl+C stops the current answer.\n",
                "\n[Quintessentia Интерактивный чат | режим={mode} | {backend} ({wf}{k})] "
                "Команды: `/k <1..max>`, `/compact <сводка>`, `/reset`, `/think on|off`, `/mode default|custom`, "
                "`/lang en|ru`, `/balance`, `/hud`, `/status`, `/exit`. Ctrl+C останавливает текущий ответ.\n",
                mode=cur_mode,
                backend=b_lbl,
                wf=wf_lbl,
                k=k_lbl,
            ),
            flush=True,
        )
        while True:
            try:
                user_in = input(tr("\nYou > ", "\nВы > ")).strip()
            except (EOFError, KeyboardInterrupt):
                print(tr("\nExiting.", "\nВыход."), flush=True)
                break
            if not user_in:
                continue
            if user_in.lower() in ("/exit", "/quit", "exit", "quit"):
                break
            if user_in.lower().startswith("/lang"):
                parts = user_in.split()
                if len(parts) >= 2 and normalize_lang(parts[1]):
                    new_l = set_lang(parts[1])
                    print(
                        tr(
                            "[Language switched to English (en)]",
                            "[Язык переключён на русский (ru)]",
                            lang=new_l,
                        ),
                        flush=True,
                    )
                else:
                    print(
                        tr(
                            "[Current language: {l}. Usage: /lang en|ru]",
                            "[Текущий язык: {l}. Использование: /lang en|ru]",
                            l=get_lang(),
                        ),
                        flush=True,
                    )
                continue
            if user_in.lower().startswith("/k"):
                parts = user_in.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    try:
                        res_k = eng.set_active_q1_planes(int(parts[1]))
                        print(
                            tr(
                                "[Active K switched in 0 ms: k={k}/{loaded}, root_bits={rb}, modules={mods}]",
                                "[Активное K переключено за 0 мс: k={k}/{loaded}, root_bits={rb}, модулей={mods}]",
                                k=res_k["q1_planes"],
                                loaded=res_k["q1_planes_loaded"],
                                rb=res_k["cube_root_bits"],
                                mods=res_k["updated_modules"],
                            ),
                            flush=True,
                        )
                    except Exception as exc:
                        print(tr("[Failed to switch K: {e}]", "[Ошибка смены K: {e}]", e=exc), flush=True)
                else:
                    print(
                        tr(
                            "[Current K={k}/{loaded}, root_bits={rb}. Usage: /k <1..loaded>]",
                            "[Текущее K={k}/{loaded}, root_bits={rb}. Использование: /k <1..loaded>]",
                            k=getattr(eng, "q1_planes", "?"),
                            loaded=getattr(eng, "q1_planes_loaded", "?"),
                            rb=getattr(eng, "cube_root_bits", "?"),
                        ),
                        flush=True,
                    )
                continue
            if user_in.lower().startswith("/compact"):
                summary = user_in[len("/compact"):].strip()
                if not summary:
                    print(
                        tr(
                            "[Usage: /compact <brief summary of prior context>]",
                            "[Использование: /compact <краткая сводка прошлого контекста>]",
                        ),
                        flush=True,
                    )
                else:
                    rep = eng.compact_session(summary, keep_last_turns=1)
                    messages = list(rep.get("compacted_messages") or [])
                    print(
                        tr(
                            "[Epoch {old_e} -> {new_e} compacted: {old_t} -> {new_t} tokens; archive preserved in ledger.jsonl]",
                            "[Эпоха {old_e} -> {new_e} сжата: {old_t} -> {new_t} токенов; архив сохранён в ledger.jsonl]",
                            old_e=rep["closed_epoch"],
                            new_e=rep["new_epoch"],
                            old_t=rep["old_tokens"],
                            new_t=rep["new_tokens"],
                        ),
                        flush=True,
                    )
                continue
            if user_in.lower() == "/reset":
                eng.reset_session()
                messages = [{"role": "system", "content": args.system}] if args.system else []
                print(tr("[Session and KV cache cleared]", "[Сессия и KV-кэш очищены]"), flush=True)
                continue
            if user_in.lower().startswith("/think"):
                parts = user_in.split()
                thinking = (parts[1].lower() == "on") if len(parts) > 1 else (not thinking)
                print(
                    tr(
                        "[<think> mode: {st}]",
                        "[Режим <think>: {st}]",
                        st="ON" if thinking else "OFF",
                    ),
                    flush=True,
                )
                continue
            if user_in.lower().startswith("/mode"):
                parts = user_in.split()
                if len(parts) > 1 and parts[1].lower() in PROFILE_MODE_CHOICES:
                    cur_mode = parts[1].lower()
                    print(tr("[Profile mode: {m}]", "[Режим профиля: {m}]", m=cur_mode), flush=True)
                else:
                    print(
                        tr(
                            "[Current mode: {m}. Usage: /mode default|custom]",
                            "[Текущий режим: {m}. Использование: /mode default|custom]",
                            m=cur_mode,
                        ),
                        flush=True,
                    )
                continue
            if user_in.lower() == "/balance":
                lm = getattr(eng, "last_metrics", None) or {}
                rb = lm.get("rssf_generation_balance")
                print(json.dumps(rb or {"status": "no_turn_yet"}, ensure_ascii=False, indent=2), flush=True)
                continue
            if user_in.lower() == "/hud":
                launch_desktop_hud(quiet=False)
                continue
            if user_in.lower() == "/status":
                print(json.dumps(eng.status_snapshot(), ensure_ascii=False, indent=2), flush=True)
                continue

            gen_profile = resolve_generation_profile(
                cur_mode,
                thinking=thinking,
                max_tokens=args.max_tokens,
                temperature=getattr(args, "temperature", None),
                top_p=getattr(args, "top_p", None),
                top_k=getattr(args, "top_k", None),
                repetition_penalty=getattr(args, "repetition_penalty", None),
                presence_penalty=getattr(args, "presence_penalty", None),
                seed=getattr(args, "seed", None),
            )
            messages.append({"role": "user", "content": user_in})
            done_pkt = _stream_to_console(
                eng,
                messages=messages,
                max_tokens=args.max_tokens,
                thinking=thinking,
                use_session=True,
                gen_profile=gen_profile,
            )
            if done_pkt:
                messages.append({"role": "assistant", "content": done_pkt.get("answer_text", "")})
            else:
                messages.pop()
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    api_key = args.api_key
    if not api_key:
        api_key = f"quintessentia-{secrets.token_hex(8)}" if args.require_auth else "quintessentia-local"
    root = find_quintessentia_root(args.quintessentia_root)
    log_file = None if args.no_log_file else (args.log_file or str(root / DEFAULT_LOG_FILE))

    with _engine_from_args(args) as eng:
        import os as _os
        if _os.environ.get("QUINTESSENTIA_SKIP_CALIBRATION", "").strip().lower() not in ("1", "true", "yes"):
            if hasattr(eng, "calibrate_hardware_limits"):
                eng.calibrate_hardware_limits(verbose=True)
        srv = QuintessentiaLANServer(
            engine=eng,
            host=args.host,
            http_port=args.port,
            beacon_port=args.beacon_port,
            api_key=api_key,
            require_auth=args.require_auth,
            enable_beacon=not args.no_beacon,
            max_request_bytes=args.max_request_bytes,
            log_file=log_file,
            preempt=not args.no_preempt,
        )
        srv.serve_forever()
    return EXIT_OK


def _resolve_win_userprofile_from_wsl() -> Path | None:
    """Resolve the active Windows user profile directory (`/mnt/c/Users/<name>`) when running under WSL2."""
    import shutil
    import subprocess

    users_root = Path("/mnt/c/Users")
    if not users_root.is_dir():
        return None
    # Prefer cmd.exe /c echo %USERPROFILE% + wslpath when available
    cmd_exe = shutil.which("cmd.exe")
    wslpath = shutil.which("wslpath")
    if cmd_exe and wslpath:
        try:
            win_prof = subprocess.check_output(
                [cmd_exe, "/c", "echo %USERPROFILE%"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3.0,
            ).strip()
            if win_prof and "%" not in win_prof:
                wsl_p = Path(subprocess.check_output([wslpath, "-u", win_prof], text=True, timeout=3.0).strip())
                if wsl_p.is_dir():
                    return wsl_p
        except Exception:
            pass
    skip = {"public", "default", "default user", "all users", "defaultapppool", "glcache"}
    best: list[tuple[int, Path]] = []
    for entry in sorted(users_root.iterdir()):
        el = entry.name.lower()
        if not entry.is_dir() or el in skip or "sandbox" in el:
            continue
        has_q = (entry / ".quintessentia").is_dir()
        has_desk = (entry / "Desktop").is_dir()
        if has_q or has_desk:
            best.append(((0 if has_q else 1) * 10 + (0 if has_desk else 1), entry))
    best.sort(key=lambda item: (item[0], item[1].as_posix()))
    return best[0][1] if best else None


def _sync_hud_bundle_to(target_dir: Path, *, include_launchers: bool = False) -> Path:
    import shutil

    src_dir = Path(__file__).resolve().parent / "hud"
    target_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("quintessentia_hud.ps1", "hud_backend.py"):
        src = src_dir / fname
        if src.is_file():
            shutil.copy2(src, target_dir / fname)
    if include_launchers:
        for fname in ("launch.vbs", "quintessentia_hud.ico"):
            src = src_dir / fname
            if src.is_file():
                shutil.copy2(src, target_dir / fname)
        if not (target_dir / "launch.vbs").is_file():
            (target_dir / "launch.vbs").write_text(
                'Set fso = CreateObject("Scripting.FileSystemObject")\r\n'
                'Set sh = CreateObject("WScript.Shell")\r\n'
                'dir = fso.GetParentFolderName(WScript.ScriptFullName)\r\n'
                'cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -STA -WindowStyle Hidden -File """ & dir & "\\quintessentia_hud.ps1"""\r\n'
                'sh.Run cmd, 0, False\r\n',
                encoding="utf-8",
            )
        (target_dir / "Start Quintessentia HUD.cmd").write_text(
            '@echo off\r\nstart "" wscript.exe "%~dp0launch.vbs"\r\n',
            encoding="utf-8",
        )
        (target_dir / "Stop Quintessentia HUD.cmd").write_text(
            '@echo off\r\n'
            'if not exist "%~dp0.cache" mkdir "%~dp0.cache"\r\n'
            'echo 1 > "%~dp0.cache\\stop_hud.flag"\r\n'
            'if exist "%USERPROFILE%\\.quintessentia\\hud_cache" echo 1 > "%USERPROFILE%\\.quintessentia\\hud_cache\\stop_hud.flag"\r\n'
            'timeout /t 1 /nobreak >nul\r\n'
            'if exist "%~dp0.cache\\hud_ui.pid" (\r\n'
            '  set /p HPID=<"%~dp0.cache\\hud_ui.pid"\r\n'
            '  if defined HPID taskkill /F /T /PID %HPID% >nul 2>&1\r\n'
            ')\r\n'
            'echo Quintessentia HUD stopped.\r\n',
            encoding="utf-8",
        )
    return target_dir / "quintessentia_hud.ps1"


def stop_desktop_hud(*, quiet: bool = False) -> bool:
    """Stop any running Quintessentia Live HUD window and its collector (fast flag + PID, zero WMI lag)."""
    import shutil
    import subprocess
    import time

    cache_dirs: list[Path] = []
    if os.name == "nt":
        cache_dirs.append(Path.home() / ".quintessentia" / "hud_cache")
        cache_dirs.append(Path.home() / "Desktop" / "Quintessentia HUD" / ".cache")
    else:
        wp = _resolve_win_userprofile_from_wsl()
        if wp is not None:
            cache_dirs.append(wp / ".quintessentia" / "hud_cache")
            cache_dirs.append(wp / "Desktop" / "Quintessentia HUD" / ".cache")

    pids: list[int] = []
    for cd in cache_dirs:
        if cd.is_dir():
            try:
                (cd / "stop_hud.flag").write_text("1", encoding="ascii")
            except OSError:
                pass
            pid_f = cd / "hud_ui.pid"
            if pid_f.is_file():
                try:
                    p = int(pid_f.read_text(encoding="ascii").strip())
                    if p > 0 and p not in pids:
                        pids.append(p)
                except Exception:
                    pass

    time.sleep(0.5)
    # Fallback force kill by exact PID (no slow WMI scan!) if process did not exit cleanly
    tk_exe = shutil.which("taskkill.exe") or (r"C:\Windows\System32\taskkill.exe" if os.name == "nt" else None)
    if tk_exe and pids:
        for pid in pids:
            try:
                kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "timeout": 4.0}
                if os.name == "nt":
                    kwargs["creationflags"] = 0x08000000
                subprocess.run([tk_exe, "/F", "/T", "/PID", str(pid)], **kwargs)
            except Exception:
                pass
    if not quiet:
        print("[Quintessentia HUD] Stopped.", flush=True)
    return True


def open_url_in_browser(url: str) -> bool:
    """Open a URL in the host default browser (native Windows or WSL2 -> Windows host)."""
    import shutil
    import subprocess
    import webbrowser

    if os.name == "nt":
        try:
            return bool(webbrowser.open(url))
        except Exception:
            return False
    cmd_exe = shutil.which("cmd.exe")
    if cmd_exe:
        try:
            subprocess.Popen(
                [cmd_exe, "/c", "start", "", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except Exception:
            return False
    try:
        return bool(webbrowser.open(url))
    except Exception:
        return False


def _all_hud_cache_dirs() -> list[Path]:
    """Return every known HUD cache directory (both `.quintessentia/hud_cache` and `Desktop/Quintessentia HUD/.cache`)."""
    dirs: list[Path] = []
    if os.name == "nt":
        roots = [Path.home()]
    else:
        wp = _resolve_win_userprofile_from_wsl()
        roots = [wp] if wp is not None else []
    for base in roots:
        dirs.append(base / ".quintessentia" / "hud_cache")
        dirs.append(base / "Desktop" / "Quintessentia HUD" / ".cache")
    return dirs


_HUD_HEARTBEAT_LOCK = None
_HUD_HEARTBEAT_STOP = None
_HUD_HEARTBEAT_THREAD = None
_HUD_BOUND_TO_CONSOLE = False


def _touch_hud_owner_flags(cache_dirs: list[Path]) -> None:
    import time
    payload = f"{os.getpid()}:{time.time():.3f}"
    for cd in cache_dirs:
        if cd.is_dir():
            try:
                (cd / "console_owner_alive.flag").write_text(payload, encoding="ascii")
            except OSError:
                pass


def stop_hud_console_heartbeat(*, signal_stop: bool = False) -> None:
    """Stop console-bound HUD heartbeat and optionally signal the HUD to close immediately."""
    global _HUD_HEARTBEAT_STOP, _HUD_HEARTBEAT_THREAD, _HUD_BOUND_TO_CONSOLE
    was_bound = _HUD_BOUND_TO_CONSOLE
    _HUD_BOUND_TO_CONSOLE = False
    ev = _HUD_HEARTBEAT_STOP
    if ev is not None:
        ev.set()
    _HUD_HEARTBEAT_STOP = None
    _HUD_HEARTBEAT_THREAD = None
    cache_dirs = _all_hud_cache_dirs()
    for cd in cache_dirs:
        if cd.is_dir():
            try:
                of = cd / "console_owner_alive.flag"
                if of.is_file():
                    of.unlink()
            except OSError:
                pass
            if signal_stop and was_bound:
                try:
                    (cd / "stop_hud.flag").write_text("1", encoding="ascii")
                except OSError:
                    pass


def start_hud_console_heartbeat() -> None:
    """Bind the Desktop HUD lifetime to the current console process.

    Updates `console_owner_alive.flag` every 0.8s and registers atexit + SIGHUP/SIGTERM
    handlers so closing the console (via Q, Ctrl+C, or window [X]) closes the HUD cleanly.
    """
    global _HUD_HEARTBEAT_LOCK, _HUD_HEARTBEAT_STOP, _HUD_HEARTBEAT_THREAD, _HUD_BOUND_TO_CONSOLE
    import atexit
    import signal
    import threading

    if _HUD_HEARTBEAT_LOCK is None:
        _HUD_HEARTBEAT_LOCK = threading.Lock()
    with _HUD_HEARTBEAT_LOCK:
        cache_dirs = _all_hud_cache_dirs()
        _HUD_BOUND_TO_CONSOLE = True
        _touch_hud_owner_flags(cache_dirs)
        if _HUD_HEARTBEAT_THREAD is not None and _HUD_HEARTBEAT_THREAD.is_alive():
            return
        stop_ev = threading.Event()
        _HUD_HEARTBEAT_STOP = stop_ev

        def _beat() -> None:
            while not stop_ev.wait(0.8):
                _touch_hud_owner_flags(cache_dirs)

        t = threading.Thread(target=_beat, name="quint-hud-owner-heartbeat", daemon=True)
        _HUD_HEARTBEAT_THREAD = t
        t.start()

        atexit.register(stop_hud_console_heartbeat, signal_stop=True)

        def _on_term(signum, frame):
            stop_hud_console_heartbeat(signal_stop=True)
            raise SystemExit(0)

        for sig_name in ("SIGHUP", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                try:
                    signal.signal(sig, _on_term)
                except Exception:
                    pass


def launch_desktop_hud(*, quiet: bool = False, bind_to_console: bool = False) -> bool:
    """Launch (or wake/focus if already running) the native Quintessentia Live HUD window.

    Works both natively on Windows and transparently from WSL 2 Linux via Windows host.
    Resolves the Desktop HUD vs `.quintessentia/hud` single-instance mutex cache mismatch
    by clearing stale `stop_hud.flag` and writing `show_window.flag` across all cache dirs.
    When `bind_to_console=True`, closing this console automatically closes the HUD as well.
    """
    import shutil
    import subprocess

    hud_script = Path(__file__).resolve().parent / "hud" / "quintessentia_hud.ps1"
    if not hud_script.is_file():
        if not quiet:
            print(f"[Quintessentia HUD] Script not found: {hud_script}", file=sys.stderr)
        return False

    # 1. Clear stale stop flags and signal any already-running HUD (regardless of which folder launched it)
    cache_dirs = _all_hud_cache_dirs()
    for cd in cache_dirs:
        if cd.is_dir():
            try:
                sf = cd / "stop_hud.flag"
                if sf.is_file():
                    sf.unlink()
            except OSError:
                pass
            try:
                (cd / "show_window.flag").write_text("1", encoding="ascii")
            except OSError:
                pass
    if bind_to_console:
        start_hud_console_heartbeat()

    if os.name == "nt":
        desk_dir = Path.home() / "Desktop" / "Quintessentia HUD"
        target_script = hud_script
        if desk_dir.is_dir():
            try:
                target_script = _sync_hud_bundle_to(desk_dir, include_launchers=False)
            except Exception:
                target_script = desk_dir / "quintessentia_hud.ps1"
        ps_exe = shutil.which("powershell.exe") or r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
        cmd = [
            ps_exe, "-NoProfile", "-ExecutionPolicy", "Bypass", "-STA",
            "-WindowStyle", "Hidden", "-File", str(target_script),
        ]
        try:
            subprocess.Popen(
                cmd,
                creationflags=0x08000000 | 0x00000008,  # CREATE_NO_WINDOW | DETACHED_PROCESS
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if not quiet:
                print("[Quintessentia HUD] Live visual HUD opened.", flush=True)
            return True
        except Exception as exc:
            if not quiet:
                print(f"[Quintessentia HUD] Failed to start: {exc}", file=sys.stderr)
            return False

    # WSL 2 Linux -> prefer installed Desktop HUD folder (or stage into `%USERPROFILE%\.quintessentia\hud`)
    wscript_exe = shutil.which("wscript.exe")
    ps_exe = shutil.which("powershell.exe")
    cmd_exe = shutil.which("cmd.exe")
    wslpath = shutil.which("wslpath")
    if wslpath and (wscript_exe or ps_exe):
        try:
            win_prof = _resolve_win_userprofile_from_wsl()
            target_dir = None
            if win_prof is not None:
                desk_dir = win_prof / "Desktop" / "Quintessentia HUD"
                if desk_dir.is_dir():
                    target_dir = desk_dir
                    staged = _sync_hud_bundle_to(desk_dir, include_launchers=False)
                else:
                    target_dir = win_prof / ".quintessentia" / "hud"
                    staged = _sync_hud_bundle_to(target_dir, include_launchers=False)
                win_script = subprocess.check_output([wslpath, "-w", str(staged)], text=True).strip()
            else:
                win_script = subprocess.check_output([wslpath, "-w", str(hud_script)], text=True).strip()

            # Prefer `wscript.exe launch.vbs` if present (100% detached Windows GUI process, zero console flash)
            vbs_path = (target_dir / "launch.vbs") if target_dir is not None else None
            if wscript_exe and vbs_path is not None and vbs_path.is_file():
                win_vbs = subprocess.check_output([wslpath, "-w", str(vbs_path)], text=True).strip()
                subprocess.Popen(
                    [wscript_exe, win_vbs],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            elif cmd_exe and ps_exe:
                subprocess.Popen(
                    [
                        cmd_exe, "/c", "start", "", "/B",
                        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-STA",
                        "-WindowStyle", "Hidden", "-File", win_script,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            else:
                subprocess.Popen(
                    [
                        ps_exe, "-NoProfile", "-ExecutionPolicy", "Bypass", "-STA",
                        "-WindowStyle", "Hidden", "-File", win_script,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            if not quiet:
                print("[Quintessentia HUD] Live visual HUD opened on Windows host.", flush=True)
            return True
        except Exception as exc:
            if not quiet:
                print(f"[Quintessentia HUD] Failed to launch via WSL bridge: {exc}", file=sys.stderr)
            return False

    if not quiet:
        print(
            "[Quintessentia HUD] Native WPF HUD requires Windows (or WSL 2 with Windows host). "
            f"Live JSON telemetry is available at: {Path.home() / '.quintessentia' / 'live_telemetry.json'}",
            flush=True,
        )
    return False


def cmd_hud(args: argparse.Namespace) -> int:
    if getattr(args, "stop", False):
        return EXIT_OK if stop_desktop_hud(quiet=False) else EXIT_RUNTIME
    if getattr(args, "install_desktop", False):
        if os.name == "nt":
            desk = Path.home() / "Desktop" / "Quintessentia HUD"
        else:
            wp = _resolve_win_userprofile_from_wsl()
            desk = (wp / "Desktop" / "Quintessentia HUD") if wp is not None else None
        if desk is None:
            print("[Quintessentia HUD] Windows Desktop folder not found.", file=sys.stderr)
            return EXIT_RUNTIME
        _sync_hud_bundle_to(desk, include_launchers=True)
        print(f"[Quintessentia HUD] Installed/updated Desktop widget at: {desk}", flush=True)
        return EXIT_OK
    if getattr(args, "status", False):
        tel_p = Path.home() / ".quintessentia" / "live_telemetry.json"
        if os.name != "nt":
            wp = _resolve_win_userprofile_from_wsl()
            if wp is not None and (wp / ".quintessentia" / "live_telemetry.json").is_file():
                tel_p = wp / ".quintessentia" / "live_telemetry.json"
        info: dict = {"telemetry_file": str(tel_p), "telemetry_exists": tel_p.is_file()}
        if tel_p.is_file():
            try:
                d = json.loads(tel_p.read_text(encoding="utf-8"))
                info.update({
                    "phase": d.get("phase"),
                    "model": d.get("model"),
                    "env": d.get("env"),
                    "updated_at": d.get("updated_at"),
                    "rssf_d3_margin": d.get("rssf_d3_margin"),
                    "rssf_regime": d.get("rssf_regime"),
                })
            except Exception:
                pass
        print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
        return EXIT_OK
    ok = launch_desktop_hud(quiet=False)
    return EXIT_OK if ok else EXIT_RUNTIME


def cmd_bench(args: argparse.Namespace) -> int:
    """Run a deterministic multi-turn/multi-run benchmark on a local model with full balance metrics."""
    import statistics

    runs = max(1, int(getattr(args, "runs", 3) or 3))
    max_toks = max(8, int(getattr(args, "max_tokens", 64) or 64))
    thinking = not bool(getattr(args, "no_think", False))
    prompts = [
        "Explain in two concise sentences why fast memory bandwidth improves language model inference.",
        "Continue from your previous answer and give one short Python formula for memory transfer time.",
        "List three numbered checks for verifying model stability on an 8 GB GPU.",
    ]
    results: list[dict] = []
    with _engine_from_args(args) as eng:
        history: list[dict[str, str]] = []
        for idx in range(runs):
            use_cache = (idx == 1)
            if not use_cache:
                eng.reset_session()
                history = []
            prompt = prompts[idx % len(prompts)]
            history.append({"role": "user", "content": prompt})
            done_pkt: dict = {}
            for ev in eng.stream_chat(
                history,
                max_new_tokens=max_toks,
                enable_thinking=thinking,
                use_session_cache=use_cache,
                stop_on_eos=True,
                temperature=0.0,
            ):
                if ev.get("type") == "done":
                    done_pkt = ev
            ans = (done_pkt.get("answer_text") or "").strip()
            history.append({"role": "assistant", "content": ans})
            m = done_pkt.get("metrics") or {}
            rb = m.get("rssf_generation_balance") or {}
            tb = rb.get("turn_end_balance") or {}
            row = {
                "run": idx + 1,
                "kv_reuse": use_cache,
                "prompt_tokens": int(m.get("prompt_tokens", 0) or 0),
                "cached_prefix_tokens": int(m.get("cached_prefix_tokens", 0) or 0),
                "generated_tokens": int(m.get("generated_tokens", 0) or 0),
                "prefill_ms": round(float(m.get("prefill_ms", 0.0) or 0.0), 1),
                "decode_tok_per_s": round(float(m.get("decode_tok_per_s", 0.0) or 0.0), 2),
                "peak_vram_mib": round(float(m.get("peak_vram_mib", 0.0) or 0.0), 1),
                "kv_gpu_layers": int(m.get("kv_gpu_layers", 0) or 0),
                "kv_offloaded_layers": int(m.get("kv_offloaded_layers", 0) or 0),
                "rssf_d3_margin": tb.get("d3_margin"),
                "rssf_regime": tb.get("regime"),
                "active_scales": sorted((rb.get("active_scales") or {}).keys()),
                "preview": ans[:120],
            }
            results.append(row)
            print(
                f"  Run {row['run']}/{runs}: {row['decode_tok_per_s']:6.2f} tok/s | "
                f"gen={row['generated_tokens']} | prefill={row['prefill_ms']:.0f}ms "
                f"(hit={row['cached_prefix_tokens']}) | peak={row['peak_vram_mib']:.0f} MiB | "
                f"d3={row['rssf_d3_margin']} ({row['rssf_regime']})",
                flush=True,
            )

    med_tps = round(statistics.median([r["decode_tok_per_s"] for r in results]), 2) if results else 0.0
    max_vram = max((r["peak_vram_mib"] for r in results), default=0.0)
    summary = {
        "model": args.model or "default",
        "runs": runs,
        "median_decode_tok_per_s": med_tps,
        "max_peak_vram_mib": max_vram,
        "results": results,
    }
    print(f"Summary: median {med_tps:.2f} tok/s | max peak VRAM {max_vram:.1f} MiB | runs={runs}", flush=True)
    if getattr(args, "json_out", None):
        Path(args.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return EXIT_OK


def cmd_discover(args: argparse.Namespace) -> int:
    print(
        tr(
            "[Quintessentia] Scanning LAN via UDP port {port} ({t:.1f}s)...",
            "[Quintessentia] Сканирование LAN по UDP-порту {port} ({t:.1f} с)...",
            port=args.beacon_port,
            t=args.timeout,
        ),
        flush=True,
    )
    nodes = discover_lan_servers(timeout_s=args.timeout, beacon_port=args.beacon_port)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(nodes, ensure_ascii=False, indent=2), encoding="utf-8")

    if not nodes:
        print(
            tr(
                "No active Quintessentia LAN servers responded on this subnet.",
                "В этой подсети не обнаружено активных LAN-серверов Quintessentia.",
            ),
            flush=True,
        )
        return EXIT_RUNTIME

    print(
        tr(
            "Found {n} active Quintessentia node(s) on the LAN:",
            "Найдено активных узлов Quintessentia в LAN: {n}",
            n=len(nodes),
        )
    )
    for n in nodes:
        print(format_connection_passport(n))
    return EXIT_OK


def _add_lang_arg(p: argparse.ArgumentParser, *, lang: str, is_root: bool = False) -> None:
    p.add_argument(
        "--lang",
        "--language",
        dest="lang",
        choices=SUPPORTED_LANGS,
        default=None if is_root else argparse.SUPPRESS,
        help=tr(
            "Interface language: en (English) or ru (Russian)",
            "Язык интерфейса: en (английский) или ru (русский)",
            lang=lang,
        ),
    )


def build_parser(lang: str | None = None) -> argparse.ArgumentParser:
    from . import __version__

    L = normalize_lang(lang) if lang is not None else get_lang()
    parser = argparse.ArgumentParser(
        prog="quintessentia",
        description=tr(
            "Quintessentia: Q1-cubes V4 / Q4 / Q6 / BF16 CUDA & CPU streaming inference engine and LAN OpenAI-compatible server",
            "Quintessentia: движок локального инференса (Q1-кубы V4 / Q4 / Q6 / BF16, CUDA и CPU) и OpenAI-совместимый LAN-сервер",
            lang=L,
        ),
    )
    parser.add_argument("--version", action="version", version=f"Quintessentia {__version__}")
    _add_lang_arg(parser, lang=L, is_root=True)
    sub = parser.add_subparsers(dest="command", required=True)

    p_models = sub.add_parser(
        "models",
        help=tr(
            "List local model files; select by number, folder name or path",
            "Список локальных моделей; выбор по номеру, имени папки или пути",
            lang=L,
        ),
    )
    _add_lang_arg(p_models, lang=L)
    p_models.add_argument("--quintessentia-root", dest="quintessentia_root", default=None)
    p_models.add_argument("--json", action="store_true")
    p_models.add_argument(
        "--check",
        action="store_true",
        help=tr(
            "Check adaptation metadata; does not run generation",
            "Проверить метаданные адаптации без запуска генерации",
            lang=L,
        ),
    )
    p_models.set_defaults(func=cmd_models)

    for command, callback in (("download", cmd_download), ("prepare", cmd_prepare)):
        p_model = sub.add_parser(
            command,
            help=(
                tr(
                    "Download a Hugging Face model into a directory and verify automatic adaptation",
                    "Скачать модель с Hugging Face в папку и проверить автоматическую адаптацию",
                    lang=L,
                )
                if command == "download"
                else tr(
                    "Verify a model's first forward and automatic runtime adaptation",
                    "Проверить первый проход модели и автоматическую адаптацию движка",
                    lang=L,
                )
            ),
        )
        _add_lang_arg(p_model, lang=L)
        p_model.add_argument("--quintessentia-root", dest="quintessentia_root", default=None)
        p_model.add_argument("--device", choices=("auto", "cpu"), default="auto")
        p_model.add_argument("--weight-format", choices=WEIGHT_FORMAT_CHOICES, default="auto")
        p_model.add_argument("--feedback-mode", choices=FEEDBACK_MODE_CHOICES, default="auto")
        p_model.add_argument("--q1-planes", type=int, choices=tuple(range(1, 17)), default=None)
        p_model.add_argument("--q1-early-stop", action=argparse.BooleanOptionalAction, default=True)
        p_model.add_argument("--vram-mode", choices=list(VRAM_MODES), default="normal")
        p_model.add_argument("--json-out", default=None)
        if command == "download":
            p_model.add_argument(
                "reference",
                help=tr(
                    "owner/repo, Hugging Face model page or GGUF file URL",
                    "owner/repo, страница модели Hugging Face или прямая ссылка на GGUF",
                    lang=L,
                ),
            )
            p_model.add_argument("--directory", "--output-dir", dest="directory", default=None)
            p_model.add_argument(
                "--file",
                default=None,
                help=tr(
                    "One GGUF variant; split parts are fetched together",
                    "Конкретный вариант GGUF; части split-GGUF скачиваются вместе",
                    lang=L,
                ),
            )
            p_model.add_argument("--revision", default=None)
            p_model.add_argument(
                "--no-verify",
                action="store_true",
                help=tr(
                    "Download without the first-forward preparation check",
                    "Скачать без проверки первого прохода",
                    lang=L,
                ),
            )
        else:
            p_model.add_argument("--model", default=None)
        p_model.set_defaults(func=callback)

    p_doc = sub.add_parser(
        "doctor",
        help=tr(
            "Check GPU, VRAM plan, model files, and LAN interfaces",
            "Проверить GPU, план VRAM, файлы модели, MTP и сетевые интерфейсы",
            lang=L,
        ),
    )
    _add_lang_arg(p_doc, lang=L)
    p_doc.add_argument("--model", "--model-dir", dest="model", type=str, default=None)
    p_doc.add_argument("--quintessentia-root", dest="quintessentia_root", type=str, default=None)
    p_doc.add_argument("--download-dir", default=None)
    p_doc.add_argument("--device", choices=("auto", "cpu"), default="auto")
    p_doc.add_argument("--vram-mode", choices=list(VRAM_MODES), default="normal")
    p_doc.add_argument("--weight-format", choices=WEIGHT_FORMAT_CHOICES, default="auto")
    p_doc.add_argument("--feedback-mode", choices=FEEDBACK_MODE_CHOICES, default="auto")
    p_doc.add_argument("--q1-planes", type=int, choices=tuple(range(1, 17)), default=None)
    p_doc.add_argument("--q1-early-stop", action=argparse.BooleanOptionalAction, default=True)
    p_doc.add_argument("--max-slot-mib", type=float, default=None)
    p_doc.add_argument("--json-out", type=str, default=None)
    p_doc.set_defaults(func=cmd_doctor)

    p_ask = sub.add_parser(
        "ask",
        help=tr(
            "Stream a single prompt (`<think>...</think>` -> answer)",
            "Одноразовый потоковый запрос (`<think>...</think>` -> ответ)",
            lang=L,
        ),
    )
    _add_lang_arg(p_ask, lang=L)
    _add_engine_args(p_ask)
    _add_generation_args(p_ask)
    p_ask.add_argument("--json-out", type=str, default=None, help=tr("Save full response + telemetry JSON", "Сохранить ответ и телеметрию в JSON", lang=L))
    p_ask.add_argument("prompt", nargs="*", help=tr("Prompt text (or `-` / piped stdin)", "Текст запроса (или `-` / stdin)", lang=L))
    p_ask.set_defaults(func=cmd_ask)

    p_bench = sub.add_parser(
        "bench",
        help=tr(
            "Run a multi-run model stability & speed benchmark with balance telemetry",
            "Прогнать многоходовой бенчмарк скорости, стабильности, KV и баланса",
            lang=L,
        ),
    )
    _add_lang_arg(p_bench, lang=L)
    _add_engine_args(p_bench)
    p_bench.add_argument("--runs", type=int, default=3, help=tr("Number of benchmark runs (default: 3, run 2 tests KV reuse)", "Число прогонов (по умолчанию 3; 2-й проверяет KV-префикс)", lang=L))
    p_bench.add_argument("--max-tokens", type=int, default=64, help=tr("Max new tokens per run (default: 64)", "Максимум новых токенов за прогон (по умолчанию 64)", lang=L))
    p_bench.add_argument("--no-think", action="store_true", help=tr("Disable `<think>` reasoning block", "Отключить блок рассуждений `<think>`", lang=L))
    p_bench.add_argument("--json-out", type=str, default=None, help=tr("Save benchmark JSON report", "Сохранить отчёт бенчмарка в JSON", lang=L))
    p_bench.set_defaults(func=cmd_bench)

    p_chat = sub.add_parser(
        "chat",
        help=tr(
            "Interactive terminal chat with KV-prefix reuse, Token Ledger & live /k switching",
            "Интерактивный чат в терминале с KV-префиксом, Token Ledger и живым переключением /k",
            lang=L,
        ),
    )
    _add_lang_arg(p_chat, lang=L)
    _add_engine_args(p_chat)
    _add_generation_args(p_chat)
    p_chat.set_defaults(func=cmd_chat)

    p_srv = sub.add_parser(
        "serve",
        help=tr(
            "Start LAN HTTP/SSE server + built-in Web UI + UDP LAN self-broadcaster",
            "Запустить LAN HTTP/SSE сервер + встроенный Web UI + UDP-маяк обнаружения",
            lang=L,
        ),
    )
    _add_lang_arg(p_srv, lang=L)
    _add_engine_args(p_srv)
    p_srv.add_argument("--host", type=str, default="0.0.0.0", help=tr("Bind address (default: 0.0.0.0 for LAN)", "Адрес привязки (по умолчанию 0.0.0.0 для LAN)", lang=L))
    p_srv.add_argument("--port", type=int, default=DEFAULT_HTTP_PORT, help=tr(f"HTTP/SSE port (default: {DEFAULT_HTTP_PORT})", f"Порт HTTP/SSE (по умолчанию {DEFAULT_HTTP_PORT})", lang=L))
    p_srv.add_argument(
        "--beacon-port", type=int, default=DEFAULT_BEACON_PORT, help=tr(f"UDP LAN beacon port (default: {DEFAULT_BEACON_PORT})", f"Порт UDP-маяка LAN (по умолчанию {DEFAULT_BEACON_PORT})", lang=L)
    )
    p_srv.add_argument("--api-key", type=str, default=None, help=tr("API key shown in the Connection Passport", "API-ключ, отображаемый в паспорте подключения", lang=L))
    p_srv.add_argument("--require-auth", action="store_true", help=tr("Enforce `Authorization: Bearer <api-key>`", "Требовать `Authorization: Bearer <api-key>`", lang=L))
    p_srv.add_argument("--no-beacon", action="store_true", help=tr("Disable UDP LAN broadcast beacon", "Отключить UDP-рассылку обнаружения в LAN", lang=L))
    p_srv.add_argument(
        "--max-request-bytes", type=int, default=DEFAULT_MAX_REQUEST_BYTES,
        help=tr(f"Reject larger request bodies with 413 (default: {DEFAULT_MAX_REQUEST_BYTES})", f"Отклонять тела запросов крупнее лимита с 413 (по умолчанию {DEFAULT_MAX_REQUEST_BYTES})", lang=L),
    )
    p_srv.add_argument("--log-file", type=str, default=None, help=tr(f"Server log path (default: <root>/{DEFAULT_LOG_FILE})", f"Путь к логу сервера (по умолчанию <root>/{DEFAULT_LOG_FILE})", lang=L))
    p_srv.add_argument("--no-log-file", action="store_true", help=tr("Do not write a server log file", "Не записывать файл лога сервера", lang=L))
    p_srv.add_argument(
        "--no-preempt", action="store_true",
        help=tr("Queue concurrent requests instead of cancelling the running generation", "Ставить параллельные запросы в очередь вместо отмены текущей генерации", lang=L),
    )
    p_srv.set_defaults(func=cmd_serve)

    p_disc = sub.add_parser(
        "discover",
        help=tr(
            "Discover Quintessentia servers broadcasting on the LAN",
            "Найти серверы Quintessentia в локальной сети",
            lang=L,
        ),
    )
    _add_lang_arg(p_disc, lang=L)
    p_disc.add_argument("--beacon-port", type=int, default=DEFAULT_BEACON_PORT)
    p_disc.add_argument("--timeout", type=float, default=2.0, help=tr("Scan duration in seconds (default: 2.0)", "Длительность сканирования в секундах (по умолчанию 2.0)", lang=L))
    p_disc.add_argument("--json-out", type=str, default=None)
    p_disc.set_defaults(func=cmd_discover)

    p_hud = sub.add_parser(
        "hud",
        help=tr(
            "Open, inspect, stop, or install the native Quintessentia Live HUD widget",
            "Открыть, проверить, остановить или установить виджет Quintessentia Live HUD",
            lang=L,
        ),
    )
    _add_lang_arg(p_hud, lang=L)
    p_hud.add_argument("--stop", action="store_true", help=tr("Stop any running HUD window and collector", "Остановить запущенное окно HUD и сборщик телеметрии", lang=L))
    p_hud.add_argument("--status", action="store_true", help=tr("Print live telemetry & HUD status JSON", "Вывести JSON со статусом HUD и живой телеметрией", lang=L))
    p_hud.add_argument("--install-desktop", action="store_true", help=tr("Install/update the Windows Desktop HUD launcher folder", "Установить/обновить папку запуска HUD на рабочем столе Windows", lang=L))
    p_hud.set_defaults(func=cmd_hud)

    p_menu = sub.add_parser(
        "menu",
        aliases=["ui", "tui"],
        help=tr(
            "Interactive button-driven terminal menu (arrow keys, EN/RU switcher, model picker, Q1 K, auto-KL, chat/server)",
            "Интерактивное кнопочное меню в терминале (стрелки, EN/RU, выбор модели, Q1 K, auto-KL, чат/сервер)",
            lang=L,
        ),
    )
    _add_lang_arg(p_menu, lang=L)
    p_menu.add_argument("--quintessentia-root", dest="quintessentia_root", type=str, default=None)
    p_menu.set_defaults(func=cmd_menu)

    return parser


def cmd_menu(args: argparse.Namespace) -> int:
    from .tui import run_tui
    return int(
        run_tui(
            quintessentia_root=getattr(args, "quintessentia_root", None),
            lang=getattr(args, "lang", None),
        )
    )


def _is_bare_lang_only_argv(raw_argv: list[str]) -> str | None:
    """Return normalized lang if `raw_argv` contains ONLY `--lang <val>` (e.g. `quint --lang ru`)."""
    if len(raw_argv) == 1:
        for pref in ("--lang=", "--language="):
            if raw_argv[0].startswith(pref):
                return normalize_lang(raw_argv[0][len(pref):])
    elif len(raw_argv) == 2 and raw_argv[0] in ("--lang", "--language"):
        return normalize_lang(raw_argv[1])
    return None


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:]) if argv is None else list(argv)
    try:
        cfg_path = find_quintessentia_root(None) / "cli_config.json"
    except Exception:
        cfg_path = None
    active_lang = resolve_cli_lang(raw_argv, config_path=cfg_path)
    set_lang(active_lang)

    if sys.stdin.isatty() and sys.stdout.isatty():
        if not raw_argv:
            from .tui import run_tui
            return int(run_tui())
        bare_lang = _is_bare_lang_only_argv(raw_argv)
        if bare_lang is not None:
            from .tui import run_tui
            return int(run_tui(lang=bare_lang))

    parser = build_parser(lang=active_lang)
    args = parser.parse_args(raw_argv)
    if getattr(args, "lang", None):
        set_lang(args.lang)
    try:
        return int(args.func(args))
    except (ModelNotFoundError, UnsupportedModelError) as exc:
        print(tr("Error: {e}", "Ошибка: {e}", e=exc), file=sys.stderr)
        return EXIT_MODEL
    except GenerationCancelled:
        print(tr("\n[Stopped]", "\n[Остановлено]"), file=sys.stderr)
        return EXIT_OK
    except KeyboardInterrupt:
        print(tr("\n[Interrupted]", "\n[Прервано]"), file=sys.stderr)
        return EXIT_RUNTIME
    except (RuntimeError, ValueError, OSError) as exc:
        print(tr("Error: {e}", "Ошибка: {e}", e=exc), file=sys.stderr)
        return EXIT_RUNTIME


if __name__ == "__main__":
    raise SystemExit(main())
