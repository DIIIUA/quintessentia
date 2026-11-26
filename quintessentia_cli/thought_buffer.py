"""Retained initial probabilities, bounded lexical steering and reversible probes.

The complete initial log-probability distribution is immutable. A separate
contrastive Phi0 prior participates in an entropy-gated, bounded log-odds update;
the current input can supersede contradicted prior entries without deleting them.
This is not a semantic verifier. Derived notes remain unverified, and full cache
snapshots protect task state from auxiliary probes. Direct recurrent-state mixing
is an explicit research option, disabled by default because it changes the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager, nullcontext
import copy
import hashlib
import time
from typing import Any

import torch


@contextmanager
def _auxiliary_forwards(u_eng: Any):
    """Isolate probe routing and UI capture state from the committed task.

    This does not roll back cache tensors: probes that borrow a task cache must
    also restore their own snapshot in a finally block.
    """
    engine = u_eng.engine
    warmup = getattr(engine, "probability_warmup", None)
    missing = object()
    fields = ("_last_prompt_opened_think", "_last_prompt_chatml_think")
    metadata = {key: getattr(u_eng, key, missing) for key in fields}
    prefetch = getattr(engine, "_prefetch_next_step", missing)
    captured = getattr(u_eng, "captured", None)
    progressive = getattr(engine, "progressive", None)
    saved = dict(captured) if captured is not None else None
    engine._auxiliary_depth = int(getattr(engine, "_auxiliary_depth", 0)) + 1
    try:
        with progressive.auxiliary_forwards() if progressive is not None else nullcontext():
            with warmup.auxiliary_forwards() if warmup is not None else nullcontext():
                yield
    finally:
        engine._auxiliary_depth = max(0, int(getattr(engine, "_auxiliary_depth", 1)) - 1)
        if saved is not None:
            captured.clear()
            captured.update(saved)
        for key, value in metadata.items():
            if value is missing:
                u_eng.__dict__.pop(key, None)
            else:
                setattr(u_eng, key, value)
        if prefetch is not missing:
            engine._prefetch_next_step = prefetch


@dataclass
class CacheStateSnapshot:
    """Lightweight GPU snapshot of Full-Attention KV lengths + hybrid recurrent/conv states."""

    kv_lengths: dict[int, int] = field(default_factory=dict)
    conv_states: dict[int, torch.Tensor] = field(default_factory=dict)
    rec_states: dict[int, torch.Tensor] = field(default_factory=dict)
    layer_states: list[dict[str, Any]] = field(default_factory=list)
    append_only_kv: set[int] = field(default_factory=set)
    vram_bytes: int = 0


def capture_cache_snapshot(past_key_values: Any) -> CacheStateSnapshot:
    """Preserve window contents, positions, initialization flags and linear states.

    Only the exact DynamicLayer implementation is known to append by replacing
    tensors; retaining its old KV references needs no copy. Sliding and recurrent
    layers need independent copies, including metadata, even when shapes do not
    change. Unknown top-level state formats fail before a probe can mutate them.
    """
    from transformers.cache_utils import DynamicLayer

    snap = CacheStateSnapshot()
    if past_key_values is None:
        return snap
    if not hasattr(past_key_values, "layers"):
        raise NotImplementedError("Thought-buffer probes require a supported layered cache; this state format cannot be rolled back")

    def tensor_bytes(value):
        if isinstance(value, torch.Tensor):
            return value.nbytes
        if isinstance(value, dict):
            return sum(tensor_bytes(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return sum(tensor_bytes(v) for v in value)
        return 0

    total_b = 0
    for l_idx, l_cache in enumerate(past_key_values.layers):
        if not (hasattr(l_cache, "keys") or hasattr(l_cache, "conv_states")):
            raise NotImplementedError(f"Unsupported thought-buffer cache layer: {type(l_cache).__name__}")
        if getattr(l_cache, "is_quintessentia_pinned_layer", False) or getattr(
            l_cache, "is_quintessentia_prealloc_layer", False
        ):
            seq_len = int(l_cache.get_seq_length())
            if seq_len > 0:
                snap.kv_lengths[l_idx] = seq_len
            snap.append_only_kv.add(l_idx)
            snap.layer_states.append({"_quintessentia_pinned_seq_len": seq_len})
            continue
        keys = getattr(l_cache, "keys", None)
        if keys is not None and keys.numel() > 0:
            snap.kv_lengths[l_idx] = int(keys.shape[-2])
        memo = {}
        if type(l_cache) is DynamicLayer:
            snap.append_only_kv.add(l_idx)
            memo = {id(t): t for t in (keys, getattr(l_cache, "values", None)) if t is not None}
        state = copy.deepcopy(vars(l_cache), memo)
        snap.layer_states.append(state)
        total_b += tensor_bytes({k: v for k, v in state.items()
                                 if l_idx not in snap.append_only_kv or k not in ("keys", "values")})
        for name, dest in (("conv_states", snap.conv_states), ("recurrent_states", snap.rec_states)):
            states = state.get(name)
            value = states.get(0) if isinstance(states, dict) else (states[0] if states else None)
            if value is not None:
                dest[l_idx] = value
    snap.vram_bytes = total_b
    return snap


def restore_cache_snapshot(past_key_values: Any, snap: CacheStateSnapshot) -> None:
    """Restore complete layer state, including saturated sliding windows."""
    if past_key_values is None:
        return
    if not hasattr(past_key_values, "layers") or len(past_key_values.layers) < len(snap.layer_states):
        raise NotImplementedError("Cannot restore this thought-buffer state format")
    if getattr(past_key_values, "is_quintessentia_offloaded", False):
        from .kv_offload import release_kv_cache_gpu_buffers
        release_kv_cache_gpu_buffers(past_key_values)
    for l_idx, state in enumerate(snap.layer_states):
        if "_quintessentia_pinned_seq_len" in state:
            layer = past_key_values.layers[l_idx]
            target_len = int(state["_quintessentia_pinned_seq_len"])
            cur_len = int(layer.get_seq_length())
            if target_len < cur_len:
                layer.crop(-(cur_len - target_len))
            else:
                layer.offload()
            continue
        memo = {}
        if l_idx in snap.append_only_kv:
            memo = {id(state[k]): state[k] for k in ("keys", "values") if state.get(k) is not None}
        restored = copy.deepcopy(state, memo)
        vars(past_key_values.layers[l_idx]).clear()
        vars(past_key_values.layers[l_idx]).update(restored)
    del past_key_values.layers[len(snap.layer_states):]


def blend_recurrent_anchor(past_key_values: Any, snap: CacheStateSnapshot, lam: float) -> None:
    """Restores non-decaying initial state participation `S_t <- (1 - lam) * S_t + lam * S_0` in GatedDeltaNet."""
    if lam <= 0.0 or past_key_values is None or not snap.rec_states:
        return
    w = float(min(max(lam, 0.0), 1.0))
    for l_idx, r0 in snap.rec_states.items():
        l_cache = past_key_values.layers[l_idx]
        current = l_cache.recurrent_states[0]
        current.lerp_(r0.to(device=current.device, dtype=current.dtype), w)


def _control_token_ids(tokenizer: Any) -> list[int]:
    ids: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    elif isinstance(eos, (list, tuple, set)):
        ids.update(int(x) for x in eos if x is not None)
    for attr in ("bos_token_id", "pad_token_id"):
        v = getattr(tokenizer, attr, None)
        if isinstance(v, int):
            ids.add(v)
    for tok_str in ("<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        try:
            enc = tokenizer.encode(tok_str, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(int(enc[0]))
        except Exception:
            pass
    return sorted(ids)


@dataclass
class AnchorStepTelemetry:
    """Diagnostic counters for initial-prior participation; no quality certification."""

    total_rows_scored: int = 0
    fork_rows_gated: int = 0
    argmax_corrections: int = 0
    recurrent_reanchors: int = 0
    snapshot_setup_ms: float = 0.0
    entropy_sum: float = 0.0
    initial_rows_active: int = 0
    initial_logit_l1: float = 0.0
    phi0_chosen_early: list[float] = field(default_factory=list)
    phi0_chosen_late: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        n = max(1, self.total_rows_scored)
        # Accumulate on the device during decode; one transfer when reporting.
        device = self.entropy_sum.device if isinstance(self.entropy_sum, torch.Tensor) else 'cpu'
        scalar = lambda value: torch.as_tensor(value, device=device, dtype=torch.float32)
        average = lambda values: torch.stack([scalar(v) for v in values]).mean() if values else scalar(0)
        values = torch.stack([scalar(self.fork_rows_gated), scalar(self.argmax_corrections),
            scalar(self.entropy_sum)/n, scalar(self.initial_rows_active), scalar(self.initial_logit_l1),
            average(self.phi0_chosen_early), average(self.phi0_chosen_late)]).detach().cpu().tolist()
        return {
            "total_rows_scored": self.total_rows_scored,
            "fork_rows_gated": int(values[0]),
            "argmax_corrections": int(values[1]),
            "argmax_changes": int(values[1]),
            "recurrent_reanchors": self.recurrent_reanchors,
            "snapshot_setup_ms": round(self.snapshot_setup_ms, 2),
            "mean_step_entropy": round(values[2], 4),
            "initial_rows_active": int(values[3]),
            "initial_logit_l1": round(values[4], 6),
            "mean_phi0_early": round(values[5], 4),
            "mean_phi0_late": round(values[6], 4),
        }


@dataclass
class ThoughtSplitLeaf:
    """Reversible M8 SPLIT/TENSION leaf storing the exact non-commutative contact Gamma_scar = (alpha, r_scar)."""

    revision_index: int
    state: str
    alpha: float
    tie: float
    delta: float
    r_scar: torch.Tensor
    phi_branch: torch.Tensor


def _tensor_directed_contact(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-tensor non-commutative contact u -> v (Section 2 of pure-math pack, zero CPU syncs)."""
    u_f = u.float()
    v_f = v.float()
    alpha = (u_f * v_f).sum(dim=-1, keepdim=True) / ((u_f * u_f).sum(dim=-1, keepdim=True) + eps)
    r_scar = v_f - alpha * u_f
    delta = r_scar.norm(dim=-1, keepdim=True) / (v_f.norm(dim=-1, keepdim=True) + eps)
    return alpha, r_scar, delta


def _tensor_tie(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Pure-tensor symmetric convergence measure Tie(a, b) in [0, 1] (Section 2 of pure-math pack)."""
    a_f = a.float()
    b_f = b.float()
    n_sum = (a_f + b_f).norm(dim=-1, keepdim=True)
    n_diff = (a_f - b_f).norm(dim=-1, keepdim=True)
    return ((n_sum - n_diff) / (n_sum + n_diff + eps)).clamp(0.0, 1.0)


class InitialProbabilityAnchor:
    """Holds `(S_0, KV_0)` and contrastive initial log-probabilities `Phi_0` across generation.

    When `mirror_mode=True` (V4 isolated scale M8), replaces fixed thresholds (0.125, 0.42, 5.5,
    0.25/1.85, 0.75) with pure-tensor canonical L4 mirror equality (natural threshold 1/2),
    immutable Phi_0 preservation, reversible SPLIT leaves Gamma_scar, and zero cross-scale coupling.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.42,
        syntax_margin: float = 5.5,
        entropy_low: float = 0.25,
        entropy_high: float = 1.85,
        rec_anchor_lambda: float = 0.0,
        rec_reanchor_every: int = 16,
        phi_clip_neg: float = -2.0,
        phi_clip_pos: float = 4.0,
        use_mid_clause_probe: bool = True,
        initial_weight: float = 0.75,
        max_log_odds_shift: float = 0.125,
        mirror_mode: bool = False,
    ):
        self.alpha = float(alpha)
        self.syntax_margin = float(syntax_margin)
        self.entropy_low = float(entropy_low)
        self.entropy_high = float(entropy_high)
        self.rec_anchor_lambda = float(rec_anchor_lambda)
        self.rec_reanchor_every = max(1, int(rec_reanchor_every))
        self.phi_clip_neg = float(phi_clip_neg)
        self.phi_clip_pos = float(phi_clip_pos)
        self.use_mid_clause_probe = bool(use_mid_clause_probe)
        if not 0.0 < initial_weight <= 1.0:
            raise ValueError("initial_weight must retain a nonzero initial channel")
        self.initial_weight = float(initial_weight)
        if not 0.0 < max_log_odds_shift < float('inf'):
            raise ValueError('max_log_odds_shift must be finite and positive')
        self.max_log_odds_shift = float(max_log_odds_shift)
        self.mirror_mode = bool(mirror_mode)

        self.snap0: CacheStateSnapshot | None = None
        self.phi0: torch.Tensor | None = None
        self.phi_buf: torch.Tensor | None = None
        self.initial_log_probs: torch.Tensor | None = None
        self.input_field: torch.Tensor | None = None
        self.input_revision_enabled = True
        self.telemetry = AnchorStepTelemetry()
        self._steps_since_reanchor = 0

        # M8 isolated canonical L4 state (used only when mirror_mode=True)
        self.mirror_split_leaves: list[ThoughtSplitLeaf] = []
        self._mirror_last_state: str = "EQUAL"
        self._mirror_last_tie: float = 1.0
        self._mirror_last_delta: float = 0.0
        self._mirror_revisions: int = 0
        self._mirror_split_events: int = 0
        self._mirror_equal_revisions: int = 0
        self._mirror_tension_revisions: int = 0
        self._mirror_zero_force_steps: int = 0
        # M8 pure-tensor 8-pair balance accumulators (zero CPU sync on hot decode path!)
        self._mirror_rssf_steps: int = 0
        self._mirror_rssf_scale_breaks: Any = 0
        self._mirror_rssf_d3_sum: Any = 0.0
        self._mirror_rssf_ie_sum: Any = 0.0

    def initial_fingerprint(self) -> str | None:
        if self.phi0 is None:
            return None
        data = self.phi0.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    def initial_probability_fingerprint(self) -> str | None:
        if self.initial_log_probs is None:
            return None
        data = self.initial_log_probs.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    def revise_input(self, field: torch.Tensor) -> None:
        """A new input may contradict the old prior; preserve Phi0 as immutable evidence.

        In M8 `mirror_mode=True`, evaluates exact non-commutative contact Phi0 -> field_new
        under canonical L4 logic (natural threshold 1/2) and records a reversible SPLIT/TENSION
        leaf Gamma_scar = (alpha, r_scar) without overwriting Phi0.
        """
        new_f = field.detach().float().clone()
        self.input_field = new_f
        clamped = new_f.clamp(self.phi_clip_neg, self.phi_clip_pos).clone()
        if not self.mirror_mode or self.phi0 is None:
            self.phi_buf = clamped
            return

        phi0_dev = self.phi0.detach().to(device=new_f.device, dtype=torch.float32)
        n_v = min(phi0_dev.numel(), new_f.numel())
        u = phi0_dev[:n_v]
        v = new_f[:n_v]
        alpha_t, r_scar, delta_t = _tensor_directed_contact(u, v)
        tie_t = _tensor_tie(u, v)
        alpha_val, tie_val, delta_val = torch.stack(
            [alpha_t.reshape(()), tie_t.reshape(()), delta_t.reshape(())]
        ).tolist()

        self._mirror_revisions += 1
        self._mirror_last_tie = float(tie_val)
        self._mirror_last_delta = float(delta_val)

        if tie_val <= 0.0 or alpha_val <= 0.0:
            state = "SPLIT"
            self._mirror_split_events += 1
            # Birth of new branch: Phi0 stays untouched in archive; working branch adopts new condition
            self.phi_buf = clamped
            self.mirror_split_leaves.append(
                ThoughtSplitLeaf(
                    revision_index=self._mirror_revisions,
                    state=state,
                    alpha=float(alpha_val),
                    tie=float(tie_val),
                    delta=float(delta_val),
                    r_scar=r_scar.detach().clone(),
                    phi_branch=clamped.detach().clone(),
                )
            )
            if len(self.mirror_split_leaves) > 16:
                self.mirror_split_leaves.pop(0)
        elif tie_val >= 0.5 and delta_val < 0.5:
            state = "EQUAL"
            self._mirror_equal_revisions += 1
            # Equilibrium node without overwriting immutable Phi0
            self.phi_buf = (0.5 * (u + clamped[:n_v])).clamp(self.phi_clip_neg, self.phi_clip_pos).clone()
        else:
            state = "TENSION"
            self._mirror_tension_revisions += 1
            # Exact reversible decomposition v = alpha * Phi0 + r_scar
            self.phi_buf = (alpha_t * u + r_scar).clamp(self.phi_clip_neg, self.phi_clip_pos).clone()
            self.mirror_split_leaves.append(
                ThoughtSplitLeaf(
                    revision_index=self._mirror_revisions,
                    state=state,
                    alpha=float(alpha_val),
                    tie=float(tie_val),
                    delta=float(delta_val),
                    r_scar=r_scar.detach().clone(),
                    phi_branch=self.phi_buf.detach().clone(),
                )
            )
            if len(self.mirror_split_leaves) > 16:
                self.mirror_split_leaves.pop(0)
        self._mirror_last_state = state

    @staticmethod
    def _ensure_neutral_baseline(u_eng: Any) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        with _auxiliary_forwards(u_eng):
            return InitialProbabilityAnchor._build_neutral_baseline(u_eng)

    @staticmethod
    def _build_neutral_baseline(u_eng: Any) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        cached = getattr(u_eng, "_neutral_logprob_cache", None)
        if cached is not None:
            return cached

        from .unified_runtime import forward_step_all_logits

        tok = u_eng.tokenizer
        probe_ids = tok.encode("\nИтог: ", add_special_tokens=False) or [tok.eos_token_id or 0]
        neutral_msgs = [{"role": "user", "content": "."}]
        n_ids, _, _, _ = u_eng._prepare_prompt_and_cache(
            neutral_msgs, enable_thinking=False, use_session_cache=False
        )
        with torch.no_grad():
            if u_eng.fast_qwen:
                lg0, past_n = forward_step_all_logits(
                    u_eng.engine,
                    torch.tensor([n_ids], dtype=torch.long),
                    past_key_values=None,
                    prefetch_next=False,
                    logits_to_keep=1,
                )
                lp0 = torch.log_softmax(lg0[0, -1].float(), dim=-1)
                lg1, past_n = forward_step_all_logits(
                    u_eng.engine,
                    torch.tensor([probe_ids], dtype=torch.long),
                    past_key_values=past_n,
                    prefetch_next=False,
                    logits_to_keep=1,
                )
                lp1 = torch.log_softmax(lg1[0, -1].float(), dim=-1)
                u_eng.captured.pop("post_norm", None)
            else:
                lg0, past_n = u_eng.engine._forward_step(
                    torch.tensor([n_ids], dtype=torch.long),
                    past_key_values=None,
                    prefetch_next=False,
                    cache_position=torch.arange(0, len(n_ids), dtype=torch.long),
                )
                lp0 = torch.log_softmax(lg0[0].float(), dim=-1)
                lg1, past_n = u_eng.engine._forward_step(
                    torch.tensor([probe_ids], dtype=torch.long),
                    past_key_values=past_n,
                    prefetch_next=False,
                    cache_position=torch.arange(len(n_ids), len(n_ids) + len(probe_ids), dtype=torch.long),
                )
                lp1 = torch.log_softmax(lg1[0].float(), dim=-1)
            del past_n
        out = (lp0.to(torch.bfloat16), lp1.to(torch.bfloat16), probe_ids)
        u_eng._neutral_logprob_cache = out
        return out

    def on_prefill_end(self, u_eng: Any, past_key_values: Any, logits_last: torch.Tensor) -> torch.Tensor:
        """Establish Phi_0 once; later input updates only the working field."""
        with _auxiliary_forwards(u_eng):
            return self._on_prefill_end(u_eng, past_key_values, logits_last)

    def _on_prefill_end(self, u_eng: Any, past_key_values: Any, logits_last: torch.Tensor) -> torch.Tensor:
        from .unified_runtime import _cache_seq_len, forward_step_all_logits

        t0 = time.perf_counter()
        lp_n0, lp_n1, probe_ids = self._ensure_neutral_baseline(u_eng)
        probe_snapshot = capture_cache_snapshot(past_key_values)

        z0 = logits_last[0, -1].float() if logits_last.ndim == 3 else logits_last[0].float()
        lp_t0 = torch.log_softmax(z0, dim=-1)
        if self.initial_log_probs is None:
            self.initial_log_probs = lp_t0.detach().clone()
        pmi = lp_t0 - lp_n0.to(device=z0.device, dtype=torch.float32)

        if self.use_mid_clause_probe and past_key_values is not None and hasattr(past_key_values, "layers"):
            cur_len = _cache_seq_len(past_key_values)
            try:
                if u_eng.fast_qwen:
                    lg_p, _ = forward_step_all_logits(
                        u_eng.engine,
                        torch.tensor([probe_ids], dtype=torch.long),
                        past_key_values=past_key_values,
                        prefetch_next=True,
                        logits_to_keep=1,
                    )
                    z_p = lg_p[0, -1].float()
                else:
                    lg_p, _ = u_eng.engine._forward_step(
                        torch.tensor([probe_ids], dtype=torch.long),
                        past_key_values=past_key_values,
                        prefetch_next=True,
                        cache_position=torch.arange(cur_len, cur_len + len(probe_ids), dtype=torch.long),
                    )
                    z_p = lg_p[0].float()
            finally:
                restore_cache_snapshot(past_key_values, probe_snapshot)
            lp_tp = torch.log_softmax(z_p, dim=-1)
            pmi_p = lp_tp - lp_n1.to(device=z0.device, dtype=torch.float32)
            pmi = torch.maximum(pmi, pmi_p)

        ctrl_ids = _control_token_ids(u_eng.tokenizer)
        if ctrl_ids:
            valid_ctrl = [i for i in ctrl_ids if 0 <= i < pmi.numel()]
            if valid_ctrl:
                pmi[torch.tensor(valid_ctrl, device=pmi.device, dtype=torch.long)] = 0.0

        pmi = pmi.clamp(self.phi_clip_neg, self.phi_clip_pos)
        if self.snap0 is None:
            self.snap0 = probe_snapshot
        if self.phi0 is None:
            self.phi0 = pmi.clone()
            self.phi_buf = pmi.clone()
            if not self.mirror_mode:
                survivor_tracker = getattr(u_eng, "survivor_tracker", None) or getattr(
                    getattr(u_eng, "engine", None), "survivor_tracker", None
                )
                if survivor_tracker is not None and hasattr(survivor_tracker, "apply_to_anchor"):
                    survivor_tracker.apply_to_anchor(self)
        else:
            self.phi0 = self.phi0.to(device=pmi.device)
            if self.phi_buf is not None:
                self.phi_buf = self.phi_buf.to(device=pmi.device)
            if self.input_revision_enabled:
                self.revise_input(pmi)
            else:
                self._merge_working_field(pmi, weight=0.35)
            if not self.mirror_mode:
                survivor_tracker = getattr(u_eng, "survivor_tracker", None) or getattr(
                    getattr(u_eng, "engine", None), "survivor_tracker", None
                )
                if survivor_tracker is not None and hasattr(survivor_tracker, "apply_to_anchor"):
                    survivor_tracker.apply_to_anchor(self)
        self._steps_since_reanchor = 0
        if torch.cuda.is_available() and z0.is_cuda:
            torch.cuda.synchronize(z0.device)
        self.telemetry.snapshot_setup_ms = (time.perf_counter() - t0) * 1000.0
        return self.transform_logits(logits_last, past_key_values, step_index=0)

    def layer_conclusion_into_buffer(self, step_logits_last: torch.Tensor, lp_neutral: torch.Tensor, weight: float = 0.35) -> None:
        """Update the working prior from a derived distribution, preserving Phi0."""
        if self.phi0 is None:
            return
        z = step_logits_last[0, -1].float() if step_logits_last.ndim == 3 else step_logits_last[0].float()
        pmi_step = (torch.log_softmax(z, dim=-1) - lp_neutral.to(z.device, torch.float32)).clamp(
            self.phi_clip_neg, self.phi_clip_pos
        )
        self._merge_working_field(pmi_step, weight)

    def _merge_working_field(self, field: torch.Tensor, weight: float) -> None:
        self._move_to_device(field.device)
        base = self.phi_buf.float() if self.phi_buf is not None else self.phi0.float()
        if not 0.0 <= weight <= 1.0:
            raise ValueError('Working-field update weight must be between zero and one')
        self.phi_buf = ((1.0-weight)*base + weight*field.float()).clamp(self.phi_clip_neg, self.phi_clip_pos)

    def _move_to_device(self, device) -> None:
        """Follow a progressive device handoff without regenerating original evidence."""
        for name in ("phi0", "phi_buf", "initial_log_probs", "input_field"):
            value = getattr(self, name)
            if value is not None and value.device != device:
                setattr(self, name, value.to(device))
        for name, value in vars(self.telemetry).items():
            if isinstance(value, torch.Tensor):
                setattr(self.telemetry, name, value.to(device))
            elif isinstance(value, list):
                value[:] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
        for m_attr in (
            "_mirror_zero_force_steps",
            "_mirror_rssf_scale_breaks",
            "_mirror_rssf_d3_sum",
            "_mirror_rssf_ie_sum",
        ):
            m_val = getattr(self, m_attr, None)
            if isinstance(m_val, torch.Tensor) and m_val.device != device:
                setattr(self, m_attr, m_val.to(device))

    def _transform_logits_mirror_m8(self, z2d: torch.Tensor, is_3d: bool, step_index: int) -> torch.Tensor:
        """Pure-tensor M8 canonical L4 logit steering (zero CPU syncs, zero 0.125/0.42/5.5/0.25/1.85 constants)."""
        M, V = z2d.shape
        phi0_v = self.phi0[:V].float() if self.phi0 is not None else self.phi_buf[:V].float()
        buf_v = self.phi_buf[:V].float()

        # Under active SPLIT leaf, steer ONLY along the new branch (never pull back to superseded branch);
        # otherwise blend strictly by canonical Tie(Phi0, phi_buf) in [0, 1] while keeping Phi0 immutable.
        if self._mirror_last_state == "SPLIT" and self.mirror_split_leaves:
            y_node = buf_v
            branch_tie_t = torch.zeros((), device=z2d.device, dtype=torch.float32)
        else:
            branch_tie_t = _tensor_tie(phi0_v, buf_v).reshape(())
            y_node = branch_tie_t * phi0_v + (1.0 - branch_tie_t) * buf_v

        logp = torch.log_softmax(z2d, dim=-1)
        probs = logp.exp()
        entropy = -(probs * logp).sum(dim=-1, keepdim=True)  # [M, 1]
        # Natural uncertainty gate g_H = H / (H + 1) in [0, 1): 0 at confident syntax, >= 0.5 at forks
        g_h = entropy / (entropy + 1.0)
        peak_lp = logp.amax(dim=-1, keepdim=True)
        # Dynamic entropy candidate window (replaces hardcoded syntax_margin=5.5)
        cand_mask = (logp >= (peak_lp - 2.0 * entropy.clamp_min(1.0))).to(torch.float32)

        # Center logits over candidate window and compute pure-tensor Tie(z_cand, y_cand) in [0, 1]
        mask_cnt = cand_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        z_mean = (z2d * cand_mask).sum(dim=-1, keepdim=True) / mask_cnt
        z_cand = (z2d - z_mean) * cand_mask
        y_cand = y_node.unsqueeze(0) * cand_mask
        tau_step = _tensor_tie(z_cand, y_cand)  # [M, 1], == 0 on orthogonal/opposite SPLIT -> zero force!

        # Pure-tensor zero-sync 8-pair force balance at thought/logit scale (I = tau_step, S = g_H, L = branch_tie)
        from src.rssf_dynamic_operators import observe_thought_balance_tensor

        d3_m8, ie_m8, _br_m8 = observe_thought_balance_tensor(
            tau_step=tau_step,
            entropy=entropy,
            g_h=g_h,
            branch_tie=branch_tie_t,
            step_index=step_index,
        )
        # Modulate cap_t smoothly by live 8-force balance (strictly zero-force when tau_step == 0!)
        bal_scale = torch.where(
            d3_m8 <= 0.0,
            torch.full_like(d3_m8, 0.5),
            (0.75 + 1.25 * ie_m8).clamp(0.5, 1.25),
        )
        # Dynamic mirror log-odds cap (replaces hardcoded max_log_odds_shift=0.125):
        # cap_t = g_H * tau_step * 0.5 * min(H, 1.0) * bal_scale
        cap_t = g_h * tau_step * (0.5 * entropy.clamp_max(1.0)) * bal_scale
        span = (y_cand.amax(dim=-1, keepdim=True) - y_cand.amin(dim=-1, keepdim=True)).clamp_min(1e-12)
        delta = y_cand * (cap_t / span)

        raw_top = z2d.argmax(dim=-1)
        anchored = z2d + delta
        new_top = anchored.argmax(dim=-1)

        self.telemetry.total_rows_scored += int(M)
        self.telemetry.fork_rows_gated += (tau_step.squeeze(-1) > 0.0).sum().detach()
        self.telemetry.argmax_corrections += (raw_top != new_top).sum().detach()
        self.telemetry.entropy_sum += entropy.sum().detach()
        self.telemetry.initial_rows_active += (delta.abs().sum(-1) > 0).sum().detach()
        self.telemetry.initial_logit_l1 += delta.abs().sum().detach()
        self._mirror_zero_force_steps += (tau_step.squeeze(-1) <= 0.0).sum().detach()
        self._mirror_rssf_steps += int(M)
        self._mirror_rssf_scale_breaks = self._mirror_rssf_scale_breaks + (d3_m8.squeeze(-1) <= 0.0).sum().detach()
        self._mirror_rssf_d3_sum = self._mirror_rssf_d3_sum + d3_m8.sum().detach()
        self._mirror_rssf_ie_sum = self._mirror_rssf_ie_sum + ie_m8.sum().detach()

        if self.phi0 is not None:
            chosen_phi = self.phi0.gather(0, new_top[:1])[0].detach()
            if step_index < 32:
                self.telemetry.phi0_chosen_early.append(chosen_phi)
            else:
                self.telemetry.phi0_chosen_late.append(chosen_phi)

        return anchored.unsqueeze(0) if is_3d else anchored

    def mirror_stats(self) -> dict[str, Any]:
        zero_force = int(self._mirror_zero_force_steps.item()) if isinstance(self._mirror_zero_force_steps, torch.Tensor) else int(self._mirror_zero_force_steps)
        rssf_steps = int(self._mirror_rssf_steps)
        rssf_sb = int(self._mirror_rssf_scale_breaks.item()) if isinstance(self._mirror_rssf_scale_breaks, torch.Tensor) else int(self._mirror_rssf_scale_breaks)
        rssf_d3 = float(self._mirror_rssf_d3_sum.item()) if isinstance(self._mirror_rssf_d3_sum, torch.Tensor) else float(self._mirror_rssf_d3_sum)
        rssf_ie = float(self._mirror_rssf_ie_sum.item()) if isinstance(self._mirror_rssf_ie_sum, torch.Tensor) else float(self._mirror_rssf_ie_sum)
        return {
            "scale": "M8_mirror_thought_buffer",
            "mirror_mode": self.mirror_mode,
            "last_mirror_state": self._mirror_last_state,
            "revisions": self._mirror_revisions,
            "equal_revisions": self._mirror_equal_revisions,
            "tension_revisions": self._mirror_tension_revisions,
            "split_events": self._mirror_split_events,
            "split_leaves_active": len(self.mirror_split_leaves),
            "zero_force_steps": zero_force,
            "last_tie": round(self._mirror_last_tie, 4),
            "last_delta": round(self._mirror_last_delta, 4),
            "phi0_immutable_sha256": self.initial_fingerprint(),
            "rssf_balance": (
                {
                    "steps": rssf_steps,
                    "scale_breaks": rssf_sb,
                    "mean_d3_margin": round(rssf_d3 / max(1, rssf_steps), 4),
                    "mean_info_efficiency_t4": round(rssf_ie / max(1, rssf_steps), 4),
                }
                if rssf_steps > 0
                else None
            ),
        }

    def transform_logits(self, logits: torch.Tensor, past_key_values: Any, step_index: int) -> torch.Tensor:
        """Apply a bounded prior on current model candidates, retaining original evidence."""
        if self.phi_buf is None or (not self.mirror_mode and self.alpha <= 0.0):
            return logits
        self._move_to_device(logits.device)

        is_3d = logits.ndim == 3
        z2d = logits[0].float() if is_3d else logits.float()  # [M, V]
        if self.mirror_mode:
            return self._transform_logits_mirror_m8(z2d, is_3d, step_index)
        M, V = z2d.shape
        initial = self.phi0[:V].float() if self.phi0 is not None else self.phi_buf[:V].float()
        if self.input_field is not None:
            current_input = self.input_field[:V].to(initial.device)
            initial = torch.where(initial*current_input < 0, torch.zeros_like(initial), initial)
        # A fixed original channel survives even if the working field is replaced.
        phi = self.initial_weight * initial + (1.0 - self.initial_weight) * self.phi_buf[:V].float()

        probs = torch.softmax(z2d, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1, keepdim=True)  # [M, 1]
        gate = ((entropy - self.entropy_low) / max(1e-6, self.entropy_high - self.entropy_low)).clamp(0.0, 1.0)

        z_max = z2d.max(dim=-1, keepdim=True).values
        syntax_mask = (z2d >= (z_max - self.syntax_margin)).to(torch.float32)

        raw_top = z2d.argmax(dim=-1)
        delta = (self.alpha * gate) * (syntax_mask * phi.unsqueeze(0))
        # Bound every pairwise log-odds change. A static vocabulary prior may
        # resolve a close alternative, but cannot apply the old 1-2 logit shove
        # to each new position. Preserve FP32 through sampling to avoid BF16 ties.
        span = delta.amax(dim=-1,keepdim=True)-delta.amin(dim=-1,keepdim=True)
        scale = (self.max_log_odds_shift/span.clamp_min(1e-12)).clamp(max=1.0)
        delta = delta*scale
        anchored = z2d + delta
        new_top = anchored.argmax(dim=-1)
        initial_delta = scale * (self.alpha * self.initial_weight * gate) * (syntax_mask * initial.unsqueeze(0))
        self.telemetry.initial_rows_active += (initial_delta.abs().sum(-1) > 0).sum().detach()
        self.telemetry.initial_logit_l1 += initial_delta.abs().sum().detach()

        self.telemetry.total_rows_scored += int(M)
        self.telemetry.fork_rows_gated += (gate.squeeze(-1) > 0.0).sum().detach()
        self.telemetry.argmax_corrections += (raw_top != new_top).sum().detach()
        self.telemetry.entropy_sum += entropy.sum().detach()

        if self.phi0 is not None:
            chosen_phi = self.phi0.gather(0,new_top[:1])[0].detach()
            if step_index < 32:
                self.telemetry.phi0_chosen_early.append(chosen_phi)
            else:
                self.telemetry.phi0_chosen_late.append(chosen_phi)

        out = anchored
        return out.unsqueeze(0) if is_3d else out

    def on_step_committed(self, past_key_values: Any, committed_tokens: int = 1) -> None:
        """Apply explicitly enabled recurrent-state mixing after speculative rollback."""
        if self.snap0 is not None and self.snap0.rec_states and self.rec_anchor_lambda > 0.0 and committed_tokens > 0:
            self._steps_since_reanchor += int(committed_tokens)
            if self._steps_since_reanchor >= self.rec_reanchor_every:
                blend_recurrent_anchor(past_key_values, self.snap0, self.rec_anchor_lambda)
                self._steps_since_reanchor = 0
                self.telemetry.recurrent_reanchors += 1

    def score_text_alignment(self, tokenizer: Any, text: str) -> float:
        """Lexical prior-support heuristic; this is not a correctness verifier."""
        if self.phi0 is None or not text.strip():
            return 0.0
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return 0.0
        idx = torch.tensor([i for i in ids if 0 <= i < self.phi0.numel()], device=self.phi0.device, dtype=torch.long)
        if idx.numel() == 0:
            return 0.0
        vals = self.phi0.float().index_select(0, idx)
        # A lexical-support diagnostic only; high scores can also reward copying.
        k = max(1, idx.numel() // 2)
        return float(torch.topk(vals, k=k).values.mean().item())


class QuintessentiaThoughtBuffer:
    """Retained initial evidence with optional draft notes and bounded steering.

    Both public paths retain the original distribution. Draft notes are explicitly
    unverified, and lexical alignment never authorizes stronger steering or retries.
    """

    def __init__(self, unified_engine: Any):
        self.u_eng = unified_engine

    def ask_anchored(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 512,
        enable_thinking: bool = False,
        alpha: float = 0.42,
        rec_anchor_lambda: float = 0.0,
        rec_reanchor_every: int = 16,
        initial_phi0: torch.Tensor | None = None,
        initial_log_probs: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        anchor = InitialProbabilityAnchor(
            alpha=alpha,
            rec_anchor_lambda=rec_anchor_lambda,
            rec_reanchor_every=rec_reanchor_every,
            mirror_mode=(getattr(self.u_eng, "q1_graphify_feedback", False) == "mirror_scales"),
        )
        if initial_phi0 is not None:
            anchor.phi0 = initial_phi0.detach().clone()
            anchor.phi_buf = anchor.phi0.clone()
            anchor.input_revision_enabled = False
        if initial_log_probs is not None:
            anchor.initial_log_probs = initial_log_probs.detach().clone()
        prev = getattr(self.u_eng, "_active_thought_anchor", None)
        self.u_eng._active_thought_anchor = anchor
        t0 = time.perf_counter()
        thinking_parts: list[str] = []
        answer_parts: list[str] = []
        metrics: dict[str, Any] = {}
        try:
            for ev in self.u_eng.stream_chat(
                messages,
                max_new_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
                use_session_cache=False,
            ):
                etype = ev.get("type")
                if etype == "delta":
                    if ev.get("phase") == "thinking":
                        thinking_parts.append(ev.get("text", ""))
                    else:
                        answer_parts.append(ev.get("text", ""))
                elif etype == "done":
                    thinking_parts = [str(ev.get("thinking_text", ""))]
                    answer_parts = [str(ev.get("answer_text", ""))]
                    metrics = dict(ev.get("metrics") or {})
        finally:
            self.u_eng._active_thought_anchor = prev

        answer_txt = "".join(answer_parts).strip()
        thinking_txt = "".join(thinking_parts).strip()
        return {
            "answer": answer_txt,
            "thinking": thinking_txt,
            "wall_s": round(time.perf_counter() - t0, 3),
            "metrics": metrics,
            "anchor_telemetry": anchor.telemetry.summary(),
            "initial_phi0_sha256": anchor.initial_fingerprint(),
            "initial_probability_sha256": anchor.initial_probability_fingerprint(),
            "phi0_alignment": round(anchor.score_text_alignment(self.u_eng.tokenizer, answer_txt), 4),
        }

    def ask_wave_buffered(
        self,
        messages: list[dict[str, str]],
        *,
        buffer_tokens: int = 260,
        answer_tokens: int = 384,
        alpha_wave: float = 0.38,
        alpha_final: float = 0.48,
        rec_anchor_lambda: float = 0.0,
        min_buffer_alignment: float = 0.25,
    ) -> dict[str, Any]:
        """Generate working notes and a final response with the original evidence.

        min_buffer_alignment is retained for call compatibility; lexical scores
        are diagnostic and no longer trigger retries or certify the notes.
        """
        t_start = time.perf_counter()
        norm_msgs = [{"role": str(m["role"]), "content": str(m.get("content", ""))} for m in messages]

        # Capture the original task BEFORE any draft. Every wave, retry and final
        # synthesis receives this same initial field rather than its own replacement.
        orig_anchor = InitialProbabilityAnchor(alpha=alpha_final, rec_anchor_lambda=rec_anchor_lambda,
                                              rec_reanchor_every=12)
        from .unified_runtime import forward_step_all_logits
        with _auxiliary_forwards(self.u_eng), torch.no_grad():
            orig_ids, _, _, _ = self.u_eng._prepare_prompt_and_cache(
                norm_msgs, enable_thinking=False, use_session_cache=False
            )
            if self.u_eng.fast_qwen:
                lg_orig, past_orig = forward_step_all_logits(
                    self.u_eng.engine, torch.tensor([orig_ids], dtype=torch.long),
                    past_key_values=None, prefetch_next=False, logits_to_keep=1,
                )
            else:
                lg_orig, past_orig = self.u_eng.engine._forward_step(
                    torch.tensor([orig_ids], dtype=torch.long), past_key_values=None,
                    prefetch_next=False, cache_position=torch.arange(0, len(orig_ids), dtype=torch.long),
                )
            orig_anchor.on_prefill_end(self.u_eng, past_orig, lg_orig)
            saved_orig_phi0 = orig_anchor.phi0.clone()
            saved_orig_log_probs = orig_anchor.initial_log_probs.clone()
            del past_orig
            self.u_eng._release_transient_state()

        # Wave 1: Assemble compact Delta-Only Meaning Buffer (never re-copying prompt text already held in Phi_0).
        wave_prompt = (
            norm_msgs[-1]["content"]
            + "\n\n---\n"
            "[СИСТЕМНЫЙ БУФЕР СМЫСЛА — НЕ переписывай условие и правила (они уже в памяти) и НЕ пиши эссе! "
            "Запиши ТОЛЬКО сжатую цепочку вычисленных переходов/дельт по одной короткой строке "
            "(где текущее значение расходится с требуемым порогом/предусловием -> какой шаг/значение получается) "
            "до самого последнего шага включительно:]"
        )
        wave_msgs = norm_msgs[:-1] + [{"role": "user", "content": wave_prompt}]
        w1 = self.ask_anchored(
            wave_msgs,
            max_new_tokens=buffer_tokens,
            enable_thinking=False,
            alpha=alpha_wave,
            rec_anchor_lambda=rec_anchor_lambda,
            initial_phi0=saved_orig_phi0,
            initial_log_probs=saved_orig_log_probs,
        )
        buf_text = w1["answer"]
        buf_align = w1["phi0_alignment"]
        retried_wave = False

        # Lexical support cannot verify facts. Do not retry with a stronger bias
        # merely because a draft uses different words from the original prompt.

        # Wave 2: Keep the original task authoritative over unverified notes.
        synth_msgs = norm_msgs + [
            {"role": "assistant", "content": f"[Буфер смысла]\n{buf_text}"},
            {
                "role": "user",
                "content": (
                    "Дай итоговый ответ в формате исходного запроса. Исходные условия имеют приоритет. "
                    "Буфер выше — непроверенные рабочие заметки: проверь их по условиям и исправь ошибки, "
                    "не принимай их утверждения за доказанные факты."
                ),
            },
        ]
        # Run synthesis where `on_prefill_end` layers the buffer's distribution ON TOP OF `saved_orig_phi0`!
        lp_n0, _, _ = InitialProbabilityAnchor._ensure_neutral_baseline(self.u_eng)
        synth_anchor = InitialProbabilityAnchor(
            alpha=alpha_final,
            rec_anchor_lambda=rec_anchor_lambda,
            rec_reanchor_every=12,
        )

        def _merged_prefill(u_e: Any, past_kv: Any, lg_last: torch.Tensor) -> torch.Tensor:
            t_s0 = time.perf_counter()
            synth_anchor.snap0 = capture_cache_snapshot(past_kv)
            synth_anchor.phi0 = saved_orig_phi0.to(device=lg_last.device)
            synth_anchor.phi_buf = saved_orig_phi0.to(device=lg_last.device).clone()
            synth_anchor.initial_log_probs = saved_orig_log_probs.to(device=lg_last.device)
            synth_anchor.layer_conclusion_into_buffer(lg_last, lp_n0, weight=0.40)
            synth_anchor._steps_since_reanchor = 0
            synth_anchor.telemetry.snapshot_setup_ms = (time.perf_counter() - t_s0) * 1000.0
            return synth_anchor.transform_logits(lg_last, past_kv, step_index=0)

        synth_anchor.on_prefill_end = _merged_prefill  # type: ignore[method-assign]

        prev = getattr(self.u_eng, "_active_thought_anchor", None)
        self.u_eng._active_thought_anchor = synth_anchor
        ans_parts: list[str] = []
        final_metrics: dict[str, Any] = {}
        try:
            for ev in self.u_eng.stream_chat(
                synth_msgs,
                max_new_tokens=answer_tokens,
                enable_thinking=False,
                use_session_cache=False,
            ):
                etype = ev.get("type")
                if etype == "delta" and ev.get("phase") != "thinking":
                    ans_parts.append(ev.get("text", ""))
                elif etype == "done":
                    ans_parts = [str(ev.get("answer_text", ""))]
                    final_metrics = dict(ev.get("metrics") or {})
        finally:
            self.u_eng._active_thought_anchor = prev

        final_answer = "".join(ans_parts).strip()
        return {
            "thought_buffer": buf_text,
            "buffer_phi0_alignment": buf_align,
            "retried_drifted_wave": retried_wave,
            "buffer_verified": False,
            "alignment_is_diagnostic_only": True,
            "answer": final_answer,
            "wall_s": round(time.perf_counter() - t_start, 3),
            "wave1_telemetry": w1["anchor_telemetry"],
            "synth_telemetry": synth_anchor.telemetry.summary(),
            "initial_phi0_sha256": orig_anchor.initial_fingerprint(),
            "wave1_initial_phi0_sha256": w1["initial_phi0_sha256"],
            "synth_initial_phi0_sha256": synth_anchor.initial_fingerprint(),
            "initial_probability_sha256": orig_anchor.initial_probability_fingerprint(),
            "wave1_initial_probability_sha256": w1['initial_probability_sha256'],
            "synth_initial_probability_sha256": synth_anchor.initial_probability_fingerprint(),
            "answer_phi0_alignment": round(
                synth_anchor.score_text_alignment(self.u_eng.tokenizer, final_answer), 4
            ),
            "metrics": final_metrics,
        }
