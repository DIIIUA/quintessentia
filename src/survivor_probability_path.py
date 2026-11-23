"""Read-only depth-trajectory survivor probability tracker (`L0 -> L_last`)."""
from __future__ import annotations

from functools import partial
from typing import Any

import torch
import torch.nn.functional as F


def _resolve_layers(engine: Any) -> list[torch.nn.Module]:
    if hasattr(engine, "_decoder_layers"):
        return list(engine._decoder_layers())
    model = getattr(engine, "model", None)
    if model is not None and hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    return []


def _resolve_final_norm(model: Any) -> torch.nn.Module | None:
    if model is None:
        return None
    for parent in (
        model,
        getattr(model, "model", None),
        getattr(model, "transformer", None),
        getattr(getattr(model, "model", None), "language_model", None),
    ):
        if parent is None:
            continue
        for attr in ("norm", "final_layernorm", "ln_f"):
            mod = getattr(parent, attr, None)
            if isinstance(mod, torch.nn.Module):
                return mod
    return None


def _resolve_lm_head(model: Any) -> Any:
    if model is None:
        return None
    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
        if head is not None:
            return head
    return getattr(model, "lm_head", None)


def _has_meta_state(mod: torch.nn.Module | None) -> bool:
    if mod is None:
        return True
    for p in mod.parameters(recurse=True):
        if getattr(p, "is_meta", False):
            return True
    for b in mod.buffers(recurse=True):
        if getattr(b, "is_meta", False):
            return True
    return False


class SurvivorProbabilityTracker:
    """Capture depth checkpoints during the final prefill chunk without disk I/O."""

    def __init__(
        self,
        engine: Any,
        *,
        draft_vocab: Any = None,
        top_survivors: int = 256,
        candidate_cap: int = 2048,
        prune_floor_logp: float = -12.0,
        max_log_odds_shift: float = 0.125,
        mirror_mode: bool = False,
    ):
        self.engine = engine
        self.draft_vocab = draft_vocab
        self.top_survivors = max(1, int(top_survivors))
        self.candidate_cap = max(self.top_survivors, int(candidate_cap))
        self.prune_floor_logp = float(prune_floor_logp)
        if not 0.0 < float(max_log_odds_shift) < float("inf"):
            raise ValueError("max_log_odds_shift must be finite and positive")
        self.max_log_odds_shift = float(max_log_odds_shift)
        self.mirror_mode = bool(mirror_mode)
        self.split_protected_count: int = 0
        self.last_entropy_floor: float | None = None

        layers = _resolve_layers(engine)
        n_layers = len(layers)
        self.checkpoints = (
            sorted({0, n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers - 1})
            if n_layers > 0
            else []
        )

        self._handles: list[Any] = []
        self._active_capture = False
        self._capture_next_final = False
        self._dirty = False
        self._checkpoint_hidden: dict[int, torch.Tensor] = {}

        self.survivor_ids: torch.Tensor | None = None
        self.early_pruned_ids: torch.Tensor | None = None
        self.survivor_log_prior: torch.Tensor | None = None

        self.updates = 0
        self.route_revisions = 0
        self.skipped_meta = 0
        self.anchor_applications = 0
        self.draft_vocab_syncs = 0
        self.probability_bus: Any = None

        for idx in self.checkpoints:
            self._handles.append(layers[idx].register_forward_hook(partial(self._on_checkpoint, idx)))

    def _on_checkpoint(self, layer_idx: int, module: torch.nn.Module, inputs: Any, output: Any) -> None:
        capture_final = self._capture_next_final and bool(self.checkpoints) and layer_idx == self.checkpoints[-1]
        if not self._active_capture and not capture_final:
            return
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(hidden, torch.Tensor) or hidden.numel() == 0:
            return
        if hidden.ndim >= 3:
            last = hidden[0, -1, :].reshape(1, -1)
        elif hidden.ndim == 2:
            last = hidden[-1:, :]
        else:
            last = hidden.reshape(1, -1)
        self._checkpoint_hidden[layer_idx] = last.detach()
        self._dirty = True
        if capture_final:
            self._capture_next_final = False

    def begin_prefill_chunk(self, *, is_last_chunk: bool, auxiliary: bool = False) -> None:
        self._active_capture = bool(is_last_chunk and not auxiliary)
        if self._active_capture:
            self._checkpoint_hidden.clear()
            self._dirty = False

    def end_prefill_chunk(self) -> None:
        if not self._active_capture:
            return
        try:
            if self._checkpoint_hidden:
                self._recompute_survivors()
                self._dirty = False
        finally:
            self._active_capture = False

    def on_route_revision(self, layer: int | None = None) -> None:
        self.route_revisions += 1
        # Mark final checkpoint for zero-sync refresh on this step; recompute lazily
        # when a consumer (draft_vocab / thought_anchor) requests updated survivors,
        # avoiding redundant 5-checkpoint LM-head projections inside hot decode hooks.
        self._capture_next_final = True
        bus = getattr(self, "probability_bus", None)
        if bus is not None and hasattr(bus, "notify_topic_shift"):
            bus.notify_topic_shift(layer=layer, reason="route_revision")

    def _candidate_ids(self, head: Any, device: torch.device) -> torch.Tensor | None:
        dv = self.draft_vocab
        cand = None
        if dv is not None:
            gpu_list = getattr(dv, "gpu_list", None)
            in_list = getattr(dv, "in_list", None)
            if isinstance(gpu_list, torch.Tensor) and gpu_list.numel() > 0:
                cand = gpu_list.reshape(-1)
            elif isinstance(in_list, torch.Tensor) and in_list.any():
                cand = in_list.nonzero(as_tuple=False).reshape(-1)
        vocab_size = (
            getattr(head, "vocab_size", None)
            or getattr(head, "out_features", None)
            or getattr(getattr(self.engine, "config", None), "vocab_size", 0)
            or 0
        )
        if cand is None:
            if vocab_size <= 0:
                return None
            cand = torch.arange(min(int(vocab_size), self.candidate_cap), dtype=torch.long, device=device)
        else:
            cand = cand.to(device=device, dtype=torch.long)
            if vocab_size > 0:
                cand = cand[(cand >= 0) & (cand < int(vocab_size))]
            if cand.numel() > self.candidate_cap:
                cand = cand[: self.candidate_cap]
        return cand if cand.numel() > 0 else None

    @staticmethod
    def _apply_norm(norm: torch.nn.Module | None, x: torch.Tensor) -> torch.Tensor:
        if norm is not None and not _has_meta_state(norm):
            param = next(norm.parameters(), None)
            if param is None:
                param = next(norm.buffers(), None)
            target_dev = param.device if param is not None else x.device
            target_dt = param.dtype if param is not None else x.dtype
            # Bypass nn.Module.__call__ so read-only survivor tracking never fires
            # external forward hooks (such as GeneralSpeculation._capture) on norm.
            return norm.forward(x.to(device=target_dev, dtype=target_dt)).float()
        xf = x.float()
        return xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    @staticmethod
    def _project_candidates(head: Any, h_norm: torch.Tensor, cand_ids: torch.Tensor) -> torch.Tensor | None:
        if hasattr(head, "project_candidates"):
            return head.project_candidates(h_norm, cand_ids.to(device=h_norm.device)).reshape(-1).float()

        raw = getattr(head, "q4_head", head)
        if _has_meta_state(raw):
            return None

        if hasattr(raw, "qweight") and hasattr(raw, "scales") and hasattr(raw, "zeros"):
            c_dev = cand_ids.to(device=raw.qweight.device, dtype=torch.long)
            qw = raw.qweight[c_dev]
            sc = raw.scales[c_dev].float()
            zr = raw.zeros[c_dev].float()
            gs = int(getattr(raw, "group_size", 64))
            lo = (qw & 0x0F).float()
            hi = (qw >> 4).float()
            w = (
                torch.stack((lo, hi), dim=-1).reshape(c_dev.numel(), -1, gs) * sc.unsqueeze(-1)
                + zr.unsqueeze(-1)
            ).reshape(c_dev.numel(), -1)
            logits = torch.matmul(h_norm.to(device=w.device, dtype=torch.float32).reshape(1, -1), w.t()).reshape(-1)
            bias = getattr(raw, "bias", None)
            if bias is not None and not getattr(bias, "is_meta", False):
                logits = logits + bias[c_dev].float()
            return logits

        if hasattr(raw, "packed_lo") and hasattr(raw, "qscale") and hasattr(raw, "qzero"):
            from src.quantized_projection import PACKED_FIELDS, unpack_weight

            c_dev = cand_ids.to(device=raw.packed_lo.device, dtype=torch.long)
            sub = {}
            for k in PACKED_FIELDS:
                if not hasattr(raw, k):
                    continue
                v = getattr(raw, k)
                if v is None:
                    continue
                if k == "col_scale":
                    sub[k] = v
                elif v.ndim == 3:
                    sub[k] = v[:, c_dev]
                else:
                    sub[k] = v[c_dev]
            w = unpack_weight(
                sub,
                int(getattr(raw, "group_size", 64)),
                dtype=torch.float32,
                active_planes=getattr(raw, "active_planes", None),
            )
            bias = getattr(raw, "bias", None)
            bias_sub = bias[c_dev].float() if (bias is not None and not getattr(bias, "is_meta", False)) else None
            return F.linear(h_norm.to(device=w.device, dtype=torch.float32).reshape(1, -1), w, bias_sub).reshape(-1)

        weight = getattr(raw, "weight", None)
        if isinstance(weight, torch.Tensor) and not weight.is_meta:
            c_dev = cand_ids.to(device=weight.device, dtype=torch.long)
            w = weight[c_dev].float()
            bias = getattr(raw, "bias", None)
            bias_sub = bias[c_dev].float() if (bias is not None and not getattr(bias, "is_meta", False)) else None
            return F.linear(h_norm.to(device=w.device, dtype=torch.float32).reshape(1, -1), w, bias_sub).reshape(-1)
        return None

    @torch.no_grad()
    def _recompute_survivors(self) -> None:
        model = getattr(self.engine, "model", None)
        head = _resolve_lm_head(model)
        if head is None:
            return
        if not hasattr(head, "project_candidates") and _has_meta_state(getattr(head, "q4_head", head)):
            self.skipped_meta += 1
            return

        ordered = [self._checkpoint_hidden[k] for k in sorted(self._checkpoint_hidden)]
        if not ordered:
            return
        cand = self._candidate_ids(head, ordered[-1].device)
        if cand is None:
            return

        norm = _resolve_final_norm(model)
        lp_list: list[torch.Tensor] = []
        target_dev = ordered[-1].device
        for h in ordered:
            h_norm = self._apply_norm(norm, h)
            logits = self._project_candidates(head, h_norm, cand)
            if logits is None:
                self.skipped_meta += 1
                return
            lp_list.append(F.log_softmax(logits.to(device=target_dev, dtype=torch.float32), dim=-1))

        lp_stack = torch.stack(lp_list, dim=0)
        lp_last = lp_stack[-1]
        lp_max = lp_stack.max(dim=0).values

        if self.mirror_mode:
            # Pure-math M6 MirrorProbabilityPath (ZERO 0.65/0.35 heuristics, ZERO -12.0 cutoff!):
            #   Delta_c = L_c - L_{c-1}
            #   S_prob  = 0.5*(L_{-2} + L_{-1}) + 0.5*(mean(Delta_{:-1}) + Delta_{-1})
            #   tau_spl = |Delta_{-1} - mean(Delta_{:-1})| / (|Delta_{-1}| + |mean(Delta_{:-1})| + eps) * I[Delta_{-1} > 0]
            #   floor   = max(L_{-1}) - 2 * max(H(p_{-1}), 1.0)
            c_cnt = int(lp_stack.shape[0])
            if c_cnt >= 3:
                deltas = lp_stack[1:] - lp_stack[:-1]
                d_last = deltas[-1]
                d_prev = deltas[:-1].mean(dim=0)
                s_prob = 0.5 * (lp_stack[-2] + lp_last) + 0.5 * (d_prev + d_last)
                tau_split = (
                    (d_last - d_prev).abs() / (d_last.abs() + d_prev.abs() + 1e-8)
                ) * (d_last > 0.0).to(lp_last.dtype)
            elif c_cnt == 2:
                d_last = lp_last - lp_stack[0]
                s_prob = 0.5 * (lp_stack[0] + lp_last) + 0.5 * d_last
                tau_split = (
                    d_last.abs() / (lp_stack[0].abs() + lp_last.abs() + 1e-8)
                ) * (d_last > 0.0).to(lp_last.dtype)
            else:
                d_last = torch.zeros_like(lp_last)
                s_prob = lp_last
                tau_split = torch.zeros_like(lp_last)

            p_last = lp_last.exp()
            entropy = float((-(p_last * lp_last).sum()).clamp_min(1.0).item())
            peak_lp = float(lp_last.max().item())
            dyn_floor = peak_lp - 2.0 * entropy
            self.last_entropy_floor = round(dyn_floor, 4)

            split_rising = (tau_split > 0.5) & (d_last > 0.0)
            base_alive = lp_max >= dyn_floor
            self.split_protected_count = int((split_rising & (~base_alive)).sum().item())
            alive = base_alive | split_rising
            score = s_prob + 0.5 * tau_split * d_last.clamp_min(0.0)
        else:
            lp_mean = lp_stack.mean(dim=0)
            score = 0.65 * lp_last + 0.35 * lp_mean
            alive = lp_max >= self.prune_floor_logp
        if not alive.any():
            alive = torch.ones_like(alive)
        ranked_score = torch.where(alive, score, score - 1e4)
        k = min(self.top_survivors, cand.numel())
        keep_order = torch.topk(ranked_score, k=k, largest=True, sorted=True).indices
        keep_mask = torch.zeros(cand.numel(), dtype=torch.bool, device=cand.device)
        keep_mask[keep_order] = True
        keep_mask = keep_mask & alive
        if not keep_mask.any():
            keep_mask[keep_order] = True

        keep_idx = keep_order[keep_mask[keep_order]]
        prune_idx = (~keep_mask).nonzero(as_tuple=False).reshape(-1)

        self.survivor_ids = cand[keep_idx].detach().clone()
        self.early_pruned_ids = cand[prune_idx].detach().clone()
        centered = lp_last[keep_idx] - lp_last[keep_idx].mean()
        self.survivor_log_prior = centered.clamp(-self.max_log_odds_shift, self.max_log_odds_shift).detach().clone()
        self.updates += 1

    def sync_draft_vocab(self, draft_vocab: Any = None) -> None:
        dv = draft_vocab if draft_vocab is not None else self.draft_vocab
        if dv is None:
            return
        if self._dirty and self._checkpoint_hidden:
            self._recompute_survivors()
            self._dirty = False
        if self.survivor_ids is None or self.survivor_ids.numel() == 0:
            return
        ids_cpu = self.survivor_ids.detach().to(device="cpu", dtype=torch.long)
        vocab_size = int(getattr(dv, "vocab_size", 0))
        if vocab_size > 0:
            ids_cpu = ids_cpu[(ids_cpu >= 0) & (ids_cpu < vocab_size)]
        if ids_cpu.numel() == 0:
            return
        if hasattr(dv, "in_conversation") and isinstance(dv.in_conversation, torch.Tensor):
            dv.in_conversation[ids_cpu] = True
        if hasattr(dv, "counts") and isinstance(dv.counts, torch.Tensor):
            dv.counts[ids_cpu] = torch.maximum(dv.counts[ids_cpu], torch.ones_like(dv.counts[ids_cpu]))
        if hasattr(dv, "rebuild"):
            dv.rebuild()
        self.draft_vocab_syncs += 1

    @torch.no_grad()
    def apply_to_anchor(self, anchor: Any) -> None:
        if anchor is None:
            return
        if self._dirty and self._checkpoint_hidden:
            self._recompute_survivors()
            self._dirty = False
        if self.survivor_ids is None or self.survivor_log_prior is None:
            return
        buf = getattr(anchor, "phi_buf", None)
        if not isinstance(buf, torch.Tensor) or buf.numel() == 0:
            return
        limit = min(self.max_log_odds_shift, float(getattr(anchor, "max_log_odds_shift", self.max_log_odds_shift)))
        ids = self.survivor_ids.to(device=buf.device, dtype=torch.long)
        valid = (ids >= 0) & (ids < buf.numel())
        if not valid.any():
            return
        ids = ids[valid]
        prior = self.survivor_log_prior.to(device=buf.device, dtype=buf.dtype)[valid].clamp(-limit, limit)
        updated = buf.clone()
        clip_neg = float(getattr(anchor, "phi_clip_neg", -2.0))
        clip_pos = float(getattr(anchor, "phi_clip_pos", 4.0))
        updated[ids] = (updated[ids] + prior).clamp(clip_neg, clip_pos)
        anchor.phi_buf = updated
        self.anchor_applications += 1

    def reset(self) -> None:
        self._active_capture = False
        self._capture_next_final = False
        self._dirty = False
        self._checkpoint_hidden.clear()
        self.survivor_ids = None
        self.early_pruned_ids = None
        self.survivor_log_prior = None

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.reset()

    def stats(self) -> dict[str, Any]:
        return {
            "checkpoints": list(self.checkpoints),
            "mirror_mode": self.mirror_mode,
            "survivors": int(self.survivor_ids.numel()) if self.survivor_ids is not None else 0,
            "early_pruned": int(self.early_pruned_ids.numel()) if self.early_pruned_ids is not None else 0,
            "split_protected": self.split_protected_count,
            "last_entropy_floor": self.last_entropy_floor,
            "updates": self.updates,
            "route_revisions": self.route_revisions,
            "skipped_meta": self.skipped_meta,
            "anchor_applications": self.anchor_applications,
            "draft_vocab_syncs": self.draft_vocab_syncs,
            "max_log_odds_shift": self.max_log_odds_shift,
        }
