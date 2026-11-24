"""Quintessentia command-line interface.

Commands:
  models    List local models; select by number, name or path
  download  Download a Hugging Face URL/repository and verify preparation
  prepare   Verify automatic runtime packing with a first forward
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
      - Explicit `--feedback-mode off`: never forces V4 or `q1_cubes`.
      - Explicit `--feedback-mode v4|mirror_scales|pure_mirror|spiral_sphere`: promotes
        `weight_format=auto` -> `q1_cubes`, enables `q1_imprint_puzzle=True`, and picks
        `q1_planes=5`, `cube_root_bits=8` unless overridden.
      - `--feedback-mode auto` + `--weight-format q1_cubes`: enables V4 (`mirror_scales`,
        `M0-M8`) automatically.
      - `--feedback-mode auto` + `--weight-format auto` on CUDA: inspects model metadata and
        live free VRAM; when a large MoE (`num_experts >= 8`) exceeds `1.25 * free_vram`,
        automatically enables full V4 (`q1_cubes` + `mirror_scales` + `q1_imprint_puzzle=True`
        + `q1_planes=5` + `cube_root_bits=8`). Smaller / dense models keep standard `auto`.
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
            "reason": "explicit_non_q1_or_cpu",
        }

    # Check if CUDA + large MoE exceeding 1.25x free VRAM -> auto-select full V4 (M0-M8)
    if cuda_available is None:
        try:
            import torch
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
    if not cuda_available or model_dir is None or not Path(model_dir).exists():
        return {
            "weight_format": wf,
            "feedback_mode": "off",
            "q1_imprint_puzzle": False,
            "q1_planes": q1_planes,
            "cube_root_bits": 16,
            "auto_v4_selected": False,
            "reason": "no_cuda_or_model",
        }

    prof = profile
    if prof is None:
        try:
            from .model_source import runtime_profile
            prof = runtime_profile(model_dir)
        except Exception:
            prof = {}
    num_experts = int((prof or {}).get("num_experts") or 0)
    if num_experts < 8:
        return {
            "weight_format": wf,
            "feedback_mode": "off",
            "q1_imprint_puzzle": False,
            "q1_planes": q1_planes,
            "cube_root_bits": 16,
            "auto_v4_selected": False,
            "reason": "dense_or_small_moe",
        }

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
        eff_planes = 5 if q1_planes is None else int(q1_planes)
        eff_root = 16 if eff_planes > 8 else 8
        return {
            "weight_format": "q1_cubes",
            "feedback_mode": "mirror_scales",
            "q1_imprint_puzzle": True,
            "q1_planes": eff_planes,
            "cube_root_bits": eff_root,
            "auto_v4_selected": True,
            "reason": "auto_large_moe_exceeds_vram",
            "num_experts": num_experts,
            "weight_gib": round(wb / 2**30, 2),
            "free_vram_gib": round(free_b / 2**30, 2),
        }

    return {
        "weight_format": wf,
        "feedback_mode": "off",
        "q1_imprint_puzzle": False,
        "q1_planes": q1_planes,
        "cube_root_bits": 16,
        "auto_v4_selected": False,
        "reason": "fits_vram_standard_auto",
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
    p.add_argument("--mtp-head-budget-mib", type=int, default=128)
    p.add_argument("--region-pruning", action=argparse.BooleanOptionalAction, default=True,
                   help="Exclude proven zero-weight routed experts and cancel obsolete current-scope preparation")
    p.add_argument("--thought-anchor", action=argparse.BooleanOptionalAction, default=False,
                   help="Retain initial probability field Phi_0 with bounded log-odds steering across generation")
    p.add_argument("--max-slot-mib", type=float, default=None,
                   help="Cap per-slot VRAM budget in MiB (enables Q1 sub-layer / sub-lm_head cube streaming when below a layer or lm_head)")
    p.add_argument("--progressive-start", "--cpu-first", action=argparse.BooleanOptionalAction, default=True,
                   help="Start CPU and GPU preparation together; devices take work when ready (default); --no-progressive-start selects the GPU-first reference")
    p.add_argument("--weight-format", choices=WEIGHT_FORMAT_CHOICES, default="auto",
                   help="Runtime weights: auto keeps native Qwen Q4 or universal Q6; explicit q1_cubes/q1/q2/q4/q6/q8/bf16")
    p.add_argument("--q1-planes", type=int, choices=tuple(range(1, 17)), default=None,
                   help="Explicit prefix depth; q1_cubes defaults to the complete 16-leaf tree, legacy q1 to 4")
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
        help="Q1 mirror feedback mode: auto (auto-selects V4 M0-M8 on large MoE exceeding VRAM), off, spiral_sphere (V2), pure_mirror (V3), or mirror_scales/v4 (V4 isolated pure-math mirror scales)",
    )
    p.add_argument(
        "--rest-tol",
        type=float,
        default=None,
        help="Custom MoE cold-expert mirror rest tolerance (requires q1_cubes / V4; default: 0.24)",
    )
    p.add_argument("--max-drafts", type=int, default=2, help="Max speculative draft depth (1..3, default: 2)")
    p.add_argument(
        "--fast-vocab",
        action="store_true",
        help="Experimental vocabulary pruner (off by default: it drops rare tokens on long outputs)",
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
    if v4_plan.get("auto_v4_selected") and v4_plan.get("reason") == "auto_large_moe_exceeds_vram":
        print(
            f"[Quintessentia Auto-V4] Large MoE ({v4_plan.get('num_experts')} experts, "
            f"{v4_plan.get('weight_gib')} GiB > 1.25x {v4_plan.get('free_vram_gib')} GiB free VRAM) "
            f"-> weight_format={v4_plan['weight_format']}, feedback_mode={v4_plan['feedback_mode']} (M0-M8), "
            f"q1_planes={v4_plan['q1_planes']}, cube_root_bits={v4_plan['cube_root_bits']}",
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
        mtp_head_budget_mib=getattr(args, "mtp_head_budget_mib", 128),
        prune_regions=getattr(args, "region_pruning", True),
        thought_anchor=getattr(args, "thought_anchor", False),
        max_slot_mib=getattr(args, "max_slot_mib", None),
        q1_planes=v4_plan["q1_planes"],
        cube_root_bits=v4_plan["cube_root_bits"],
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
    if profile:
        why = "; ".join(profile.get("why_not_fast") or [])
        fast_line = "on" if profile.get("fast_qwen") else f"off ({why})" if why else "off"
        print(f"  Backend            : {profile.get('backend')} / {profile.get('weight_format')} / fast {fast_line}")
    v4_doc = report.get("auto_v4_plan")
    if v4_doc and v4_doc.get("auto_v4_selected"):
        print(
            f"  Mirror Scales V4   : ON ({v4_doc['feedback_mode']} M0-M8, "
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
        layout = "all layers resident" if plan.full_residency else (
            f"{plan.resident_layers}/{plan.layers_total} resident + {plan.ring_slots}-slot ring"
        )
        print(
            f"  VRAM plan ({plan.mode:6s}): total {plan.total_bytes / 2**30:.2f} GiB, free {plan.free_bytes / 2**30:.2f} GiB, "
            f"reserve {plan.reserve_bytes / 2**30:.2f} GiB -> {layout}, MTP {'on' if plan.enable_mtp else 'off'}"
        )
    elif "vram_plan_error" in report:
        print(f"  VRAM plan          : {report['vram_plan_error']}")
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
                        q1_planes=4, q1_early_stop=True):
    import torch
    from .unified_runtime import QuintessentiaUnifiedEngine
    with QuintessentiaUnifiedEngine(folder, quintessentia_root=root, device=device, weight_format=weight_format,
        vram_mode=vram_mode, enable_mtp=False, speculation="off", verbose=False,
        max_context_tokens=512, prefill_chunk_tokens=8, q1_planes=q1_planes,
        q1_early_stop=q1_early_stop) as engine:
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
                                 vram_mode=args.vram_mode, q1_planes=getattr(args, "q1_planes", 4),
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
                vram_mode=args.vram_mode, q1_planes=getattr(args, "q1_planes", 4),
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
                        print("\n[Размышление <think>]\n", end="", flush=True)
                        in_think_printed = True
                    print(ev["text"], end="", flush=True)
                else:
                    if in_think_printed and not in_ans_printed:
                        print("\n\n[/think -> Ответ]\n", end="", flush=True)
                    elif not in_ans_printed:
                        print("\n[Ответ]\n", end="", flush=True)
                    in_ans_printed = True
                    print(ev["text"], end="", flush=True)
            elif ev["type"] == "done":
                done_packet = ev
    except KeyboardInterrupt:
        gen.close()
        print("\n\n[Остановлено]", flush=True)
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
    print(
        f"\n\n--- [ one answer: {m.get('decode_tok_per_s', 0.0):.2f} tok/s | "
        f"{m.get('generated_tokens', 0)} tok (EOS={m.get('stopped_by_eos')}) | "
        f"Prefill: {m.get('prefill_ms', 0.0):.1f} ms{reuse_str}{accel_str} | "
        f"Peak VRAM: {m.get('peak_vram_mib', 0.0):.1f} MiB{park_str} ] ---",
        flush=True,
    )
    return done_packet


def cmd_ask(args: argparse.Namespace) -> int:
    prompt_text = " ".join(args.prompt).strip()
    if not prompt_text:
        print("Error: please provide a prompt, e.g. `quint ask \"Привет!\"`", file=sys.stderr)
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
        print(
            f"\n[Quintessentia Interactive Chat | mode={cur_mode}] "
            "Commands: `/reset`, `/think on|off`, `/mode default|custom`, `/status`, `/exit`. "
            "Ctrl+C stops the current answer.\n",
            flush=True,
        )
        while True:
            try:
                user_in = input("\nВы > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nВыход.", flush=True)
                break
            if not user_in:
                continue
            if user_in.lower() in ("/exit", "/quit", "exit", "quit"):
                break
            if user_in.lower() == "/reset":
                eng.reset_session()
                messages = [{"role": "system", "content": args.system}] if args.system else []
                print("[Сессия и KV-кэш очищены]", flush=True)
                continue
            if user_in.lower().startswith("/think"):
                parts = user_in.split()
                thinking = (parts[1].lower() == "on") if len(parts) > 1 else (not thinking)
                print(f"[Режим <think>: {'ON' if thinking else 'OFF'}]", flush=True)
                continue
            if user_in.lower().startswith("/mode"):
                parts = user_in.split()
                if len(parts) > 1 and parts[1].lower() in PROFILE_MODE_CHOICES:
                    cur_mode = parts[1].lower()
                    print(f"[Режим профиля: {cur_mode}]", flush=True)
                else:
                    print(f"[Текущий режим: {cur_mode}. Использование: /mode default|custom]", flush=True)
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


def cmd_discover(args: argparse.Namespace) -> int:
    print(f"[Quintessentia] Scanning LAN via UDP port {args.beacon_port} ({args.timeout:.1f}s)...", flush=True)
    nodes = discover_lan_servers(timeout_s=args.timeout, beacon_port=args.beacon_port)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(nodes, ensure_ascii=False, indent=2), encoding="utf-8")

    if not nodes:
        print("No active Quintessentia LAN servers responded on this subnet.", flush=True)
        return EXIT_RUNTIME

    print(f"Found {len(nodes)} active Quintessentia node(s) on the LAN:")
    for n in nodes:
        print(format_connection_passport(n))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    from . import __version__
    parser = argparse.ArgumentParser(
        prog="quintessentia",
        description="Quintessentia: mirror-equality Q1/Q4/Q6 CUDA & CPU streaming inference engine and LAN OpenAI-compatible server",
    )
    parser.add_argument("--version", action="version", version=f"Quintessentia {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_models = sub.add_parser("models", help="List local model files; select by number, folder name or path")
    p_models.add_argument("--quintessentia-root", dest="quintessentia_root", default=None)
    p_models.add_argument("--json", action="store_true")
    p_models.add_argument("--check", action="store_true", help="Check adaptation metadata; does not run generation")
    p_models.set_defaults(func=cmd_models)

    for command, callback in (("download",cmd_download),("prepare",cmd_prepare)):
        p_model = sub.add_parser(command, help=("Download a Hugging Face model into a directory and verify automatic adaptation"
                                               if command == "download" else "Verify a model's first forward and automatic runtime packing"))
        p_model.add_argument("--quintessentia-root", dest="quintessentia_root", default=None)
        p_model.add_argument("--device", choices=("auto","cpu"), default="auto")
        p_model.add_argument("--weight-format", choices=WEIGHT_FORMAT_CHOICES, default="auto")
        p_model.add_argument("--q1-planes", type=int, choices=tuple(range(1, 9)), default=4)
        p_model.add_argument("--q1-early-stop", action=argparse.BooleanOptionalAction, default=True)
        p_model.add_argument("--vram-mode", choices=list(VRAM_MODES), default="normal")
        p_model.add_argument("--json-out", default=None)
        if command == "download":
            p_model.add_argument("reference", help="owner/repo, Hugging Face model page or GGUF file URL")
            p_model.add_argument("--directory", "--output-dir", dest="directory", default=None)
            p_model.add_argument("--file", default=None, help="One GGUF variant; split parts are fetched together")
            p_model.add_argument("--revision", default=None)
            p_model.add_argument("--no-verify", action="store_true", help="Download without the first-forward preparation check")
        else:
            p_model.add_argument("--model", default=None)
        p_model.set_defaults(func=callback)

    p_doc = sub.add_parser("doctor", help="Check GPU, VRAM plan, model files, and LAN interfaces")
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

    p_ask = sub.add_parser("ask", help="Stream a single prompt (`<think>...</think>` -> `<|im_end|>`)")
    _add_engine_args(p_ask)
    _add_generation_args(p_ask)
    p_ask.add_argument("--json-out", type=str, default=None, help="Save full response + telemetry JSON")
    p_ask.add_argument("prompt", nargs="+", help="Prompt text")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="Interactive terminal chat with `kv_prefix` reuse & RAM parking")
    _add_engine_args(p_chat)
    _add_generation_args(p_chat)
    p_chat.set_defaults(func=cmd_chat)

    p_srv = sub.add_parser("serve", help="Start LAN HTTP/SSE server + UDP LAN self-broadcaster")
    _add_engine_args(p_srv)
    p_srv.add_argument("--host", type=str, default="0.0.0.0", help="Bind address (default: 0.0.0.0 for LAN)")
    p_srv.add_argument("--port", type=int, default=DEFAULT_HTTP_PORT, help=f"HTTP/SSE port (default: {DEFAULT_HTTP_PORT})")
    p_srv.add_argument(
        "--beacon-port", type=int, default=DEFAULT_BEACON_PORT, help=f"UDP LAN beacon port (default: {DEFAULT_BEACON_PORT})"
    )
    p_srv.add_argument("--api-key", type=str, default=None, help="API key shown in the Connection Passport")
    p_srv.add_argument("--require-auth", action="store_true", help="Enforce `Authorization: Bearer <api-key>`")
    p_srv.add_argument("--no-beacon", action="store_true", help="Disable UDP LAN broadcast beacon")
    p_srv.add_argument(
        "--max-request-bytes", type=int, default=DEFAULT_MAX_REQUEST_BYTES,
        help=f"Reject larger request bodies with 413 (default: {DEFAULT_MAX_REQUEST_BYTES} = 8 MiB)",
    )
    p_srv.add_argument("--log-file", type=str, default=None, help=f"Server log path (default: <root>/{DEFAULT_LOG_FILE})")
    p_srv.add_argument("--no-log-file", action="store_true", help="Do not write a server log file")
    p_srv.add_argument(
        "--no-preempt", action="store_true",
        help="Queue concurrent requests instead of cancelling the running generation",
    )
    p_srv.set_defaults(func=cmd_serve)

    p_disc = sub.add_parser("discover", help="Discover Quintessentia servers broadcasting on the LAN")
    p_disc.add_argument("--beacon-port", type=int, default=DEFAULT_BEACON_PORT)
    p_disc.add_argument("--timeout", type=float, default=2.0, help="Scan duration in seconds (default: 2.0)")
    p_disc.add_argument("--json-out", type=str, default=None)
    p_disc.set_defaults(func=cmd_discover)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ModelNotFoundError, UnsupportedModelError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_MODEL
    except GenerationCancelled:
        print("\n[Остановлено]", file=sys.stderr)
        return EXIT_OK
    except KeyboardInterrupt:
        print("\n[Прервано]", file=sys.stderr)
        return EXIT_RUNTIME
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME


if __name__ == "__main__":
    raise SystemExit(main())
