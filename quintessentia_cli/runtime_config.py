"""Lightweight CLI contracts; listing/help must not import Torch or start CUDA."""
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VRAM_MODES = ("normal", "turbo")


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
