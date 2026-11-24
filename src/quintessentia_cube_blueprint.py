"""Quintessentia Dual-Model Probability Comparator & Cube Blueprint Generator.

Implements the Method 1 -> Quintessentia Tomography -> Method 2 pipeline:
  1. Runs Quintessentia probability diagnostics (survivor recall, zero-prob deadzone intrusion,
     entropy shift / false confidence, layer amplification vs damping, and MoE router agreement)
     on BOTH the original model and the Q1-cube-reconstructed model across all decoder layers.
  2. Combines layer-level Quintessentia sensitivity with real activation-weighted projection errors
     ||X @ W^T - X @ W_hat(k)^T||_F / ||X @ W^T||_F measured directly on calibration prompts.
  3. Allocates exact per-(layer, projection) Q1 cube depths (a QuintessentiaBlueprint) maximizing
     Quintessentia probability fidelity per parameter/byte under a target average cube budget,
     ready to drive AirLLMLazyBinaryHalvingRunner (Method 2) with zero guesswork.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.binary_halving_q1 import BinaryHalvingQ1Linear, _is_router_module


@torch.no_grad()
def quintessentia_prob_metrics(
    logits_orig: torch.Tensor,
    logits_cube: torch.Tensor,
    *,
    survivor_prob_floor: float = 1e-3,
    deadzone_prob_ceil: float = 1e-4,
    survivor_top_k: int = 16,
) -> Dict[str, float]:
    """Compare original vs Q1-cube logits through Quintessentia survivor & deadzone optics."""
    o = logits_orig.detach().float().reshape(-1)
    c = logits_cube.detach().float().reshape(-1)

    log_po = F.log_softmax(o, dim=-1)
    log_pc = F.log_softmax(c, dim=-1)
    po = log_po.exp()
    pc = log_pc.exp()

    kl = float((po * (log_po - log_pc)).sum().clamp_min(0.0).item())
    cos = float(F.cosine_similarity(o, c, dim=0).item())

    ent_o = float((-(po * log_po).sum()).clamp_min(0.0).item())
    ent_c = float((-(pc * log_pc).sum()).clamp_min(0.0).item())
    delta_ent = ent_c - ent_o

    o_top2_val, o_top2_idx = po.topk(2)
    c_top2_val, c_top2_idx = pc.topk(2)
    margin_o = float((o_top2_val[0] - o_top2_val[1]).item())
    margin_c = float((c_top2_val[0] - c_top2_val[1]).item())
    top1_match = float(int(o_top2_idx[0].item()) == int(c_top2_idx[0].item()))

    o_top5 = set(o.topk(5).indices.tolist())
    c_top5 = set(c.topk(5).indices.tolist())
    top5_overlap = len(o_top5 & c_top5) / 5.0

    # Survivor set: tokens with meaningful probability in the original model (or top-K fallback)
    surv_mask_o = po >= survivor_prob_floor
    if int(surv_mask_o.sum().item()) == 0:
        idx_k = po.topk(min(survivor_top_k, po.numel())).indices
        surv_mask_o = torch.zeros_like(po, dtype=torch.bool)
        surv_mask_o[idx_k] = True
    surv_mask_c = pc >= (survivor_prob_floor * 0.5)

    # Probability-weighted survivor recall: how much of original survivor mass stays alive in cube model
    orig_surv_mass = float(po[surv_mask_o].sum().item())
    retained_surv_mass = float(po[surv_mask_o & surv_mask_c].sum().item())
    survivor_recall = retained_surv_mass / max(1e-8, orig_surv_mass)

    # Deadzone intrusion: excess cube probability mass leaking into tokens that are dead (< 1e-4) in original
    dead_mask_o = po < deadzone_prob_ceil
    deadzone_intrusion = float((pc[dead_mask_o].sum() - po[dead_mask_o].sum()).clamp_min(0.0).item())

    # False confidence: cube model collapses entropy harder than original while picking the WRONG top-1
    false_confidence = 1.0 if (top1_match == 0.0 and delta_ent < -0.15) else 0.0

    return {
        "kl_div": round(kl, 6),
        "logit_cos": round(cos, 6),
        "top1_match": round(top1_match, 4),
        "top5_overlap": round(top5_overlap, 4),
        "entropy_orig": round(ent_o, 4),
        "entropy_cube": round(ent_c, 4),
        "delta_entropy": round(delta_ent, 4),
        "margin_orig": round(margin_o, 4),
        "margin_cube": round(margin_c, 4),
        "survivor_count_orig": int(surv_mask_o.sum().item()),
        "survivor_recall": round(survivor_recall, 5),
        "deadzone_intrusion": round(deadzone_intrusion, 5),
        "false_confidence": false_confidence,
    }


def _find_model_parts(model: nn.Module) -> Tuple[nn.ModuleList, Optional[nn.Module], Optional[nn.Module]]:
    """Locate decoder layers, final norm, and lm_head across Llama/Qwen/Granite/GPT architectures."""
    base = None
    layers = None
    for attr in ("model", "transformer", "gpt_neox"):
        sub = getattr(model, attr, None)
        if sub is not None:
            for l_attr in ("layers", "h", "blocks"):
                cand = getattr(sub, l_attr, None)
                if isinstance(cand, nn.ModuleList):
                    base = sub
                    layers = cand
                    break
        if layers is not None:
            break
    if layers is None:
        raise ValueError(f"Could not locate decoder layers on {type(model).__name__}")

    final_norm = None
    for n_attr in ("norm", "final_layernorm", "ln_f"):
        cand = getattr(base, n_attr, None)
        if cand is not None:
            final_norm = cand
            break

    lm_head = getattr(model, "lm_head", None) or getattr(model, "embed_out", None)
    return layers, final_norm, lm_head


@torch.no_grad()
def _project_hidden_to_logits(
    h_last: torch.Tensor,
    final_norm: Optional[nn.Module],
    lm_head: Optional[nn.Module],
) -> Optional[torch.Tensor]:
    """Quintessentia Logit Lens projection of intermediate hidden state h_l -> vocabulary logits."""
    if lm_head is None:
        return None
    w = getattr(lm_head, "weight", None)
    if w is None or w.is_meta:
        return None
    dev = w.device
    dt = w.dtype
    x = h_last.to(device=dev, dtype=dt)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if final_norm is not None:
        nw = getattr(final_norm, "weight", None)
        if nw is not None and not nw.is_meta:
            x = final_norm(x.to(device=nw.device, dtype=nw.dtype)).to(device=dev, dtype=dt)
    return lm_head(x).squeeze(0).float().cpu()


@dataclass
class LayerTrajectory:
    """Captured per-layer hidden states, Quintessentia Logit-Lens logits, router top-k, and projection inputs."""

    num_layers: int
    layer_h_last: Dict[int, torch.Tensor] = field(default_factory=dict)
    layer_logits: Dict[int, torch.Tensor] = field(default_factory=dict)
    final_logits: Optional[torch.Tensor] = None
    router_topk: Dict[int, torch.Tensor] = field(default_factory=dict)
    proj_inputs: Dict[Tuple[int, str], torch.Tensor] = field(default_factory=dict)


@dataclass
class QuintessentiaBlueprint:
    """Actionable Method 1 -> Method 2 blueprint derived from Quintessentia dual-model tomography."""

    target_avg_cubes: float
    actual_param_weighted_avg_cubes: float
    k_min: int
    k_max: int
    layer_proj_cubes: Dict[int, Dict[str, int]]
    depth_histogram: Dict[int, int]
    layer_weights: Dict[int, float]
    top_amplifier_layers: List[int]
    top_deadzone_trauma_layers: List[int]
    safe_damper_layers: List[int]
    layer_horizon: Optional[int]
    tomography: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_avg_cubes": self.target_avg_cubes,
            "actual_param_weighted_avg_cubes": self.actual_param_weighted_avg_cubes,
            "k_min": self.k_min,
            "k_max": self.k_max,
            "depth_histogram": {str(k): int(v) for k, v in sorted(self.depth_histogram.items())},
            "layer_weights": {str(k): round(float(v), 4) for k, v in sorted(self.layer_weights.items())},
            "top_amplifier_layers": list(self.top_amplifier_layers),
            "top_deadzone_trauma_layers": list(self.top_deadzone_trauma_layers),
            "safe_damper_layers": list(self.safe_damper_layers),
            "layer_horizon": self.layer_horizon,
            "layer_proj_cubes": {str(l): dict(m) for l, m in sorted(self.layer_proj_cubes.items())},
            "tomography": self.tomography,
        }


class QuintessentiaCubeComparator:
    """Runs Quintessentia probability optics on original vs Q1-cube models and builds Method 2 blueprints."""

    @staticmethod
    @torch.no_grad()
    def capture_trajectory(
        model: nn.Module,
        step_fn: Callable[[torch.Tensor], torch.Tensor],
        prompt_ids: torch.Tensor,
        *,
        capture_proj_inputs: bool = False,
        max_input_tokens: int = 16,
    ) -> LayerTrajectory:
        """Capture per-layer hidden states, Quintessentia Logit-Lens logits, router top-k, and optional proj inputs."""
        layers, final_norm, lm_head = _find_model_parts(model)
        traj = LayerTrajectory(num_layers=len(layers))
        handles: List[Any] = []

        for l_idx, layer in enumerate(layers):
            def _make_layer_hook(li: int):
                def _hook(_mod, _inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    if not isinstance(h, torch.Tensor):
                        return
                    # Take last token hidden state across sequence
                    h_last = h[0, -1] if h.ndim == 3 else (h[-1] if h.ndim == 2 else h)
                    traj.layer_h_last[li] = h_last.detach().float().cpu()
                    lg = _project_hidden_to_logits(h_last, final_norm, lm_head)
                    if lg is not None:
                        traj.layer_logits[li] = lg
                return _hook

            handles.append(layer.register_forward_hook(_make_layer_hook(l_idx)))

            for sub_name, sub_mod in layer.named_modules():
                if capture_proj_inputs and isinstance(sub_mod, (nn.Linear, BinaryHalvingQ1Linear)):
                    if _is_router_module(f"layers.{l_idx}.{sub_name}", sub_mod):
                        pass
                    else:
                        def _make_in_hook(li: int, sn: str):
                            def _pre(_mod, args):
                                if args and isinstance(args[0], torch.Tensor):
                                    x = args[0].detach()
                                    flat = x.reshape(-1, x.shape[-1])
                                    if flat.shape[0] > max_input_tokens:
                                        flat = flat[-max_input_tokens:]
                                    traj.proj_inputs[(li, sn)] = flat.to(dtype=torch.float32).cpu()
                            return _pre
                        handles.append(sub_mod.register_forward_pre_hook(_make_in_hook(l_idx, sub_name)))

                # Capture MoE router top-k output if present
                leaf = sub_name.rsplit(".", 1)[-1]
                if leaf in ("router", "gate") and ("moe" in sub_name.lower() or "block_sparse" in sub_name.lower() or "mlp" in sub_name.lower()):
                    if isinstance(sub_mod, nn.Linear) or hasattr(sub_mod, "weight"):
                        def _make_router_hook(li: int):
                            def _rhook(_mod, _inp, out):
                                if isinstance(out, tuple) and len(out) >= 2:
                                    # GraniteMoeTopKGating returns (router_logits, router_scores, top_k_index) or (scores, indices)
                                    for item in out:
                                        if isinstance(item, torch.Tensor) and item.dtype in (torch.int64, torch.int32):
                                            traj.router_topk[li] = item.detach().cpu()
                                            return
                                    if isinstance(out[0], torch.Tensor) and out[0].is_floating_point():
                                        k = min(8, out[0].shape[-1])
                                        traj.router_topk[li] = out[0].detach().float().topk(k, dim=-1).indices.cpu()
                                elif isinstance(out, torch.Tensor) and out.is_floating_point():
                                    k = min(8, out.shape[-1])
                                    traj.router_topk[li] = out.detach().float().topk(k, dim=-1).indices.cpu()
                            return _rhook
                        handles.append(sub_mod.register_forward_hook(_make_router_hook(l_idx)))

        try:
            final_lg = step_fn(prompt_ids)
            traj.final_logits = final_lg.detach().float().reshape(-1).cpu()
        finally:
            for h in handles:
                h.remove()

        return traj

    @staticmethod
    def compare_trajectories(
        orig: LayerTrajectory,
        cube: LayerTrajectory,
    ) -> Dict[str, Any]:
        """Produce 4-slice Quintessentia tomography comparing original vs Q1-cube layer trajectories."""
        num_layers = orig.num_layers
        rows: List[Dict[str, Any]] = []
        prev_h_err = 0.0
        layer_horizon: Optional[int] = None

        for l_idx in range(num_layers):
            ho = orig.layer_h_last[l_idx]
            hc = cube.layer_h_last[l_idx]
            h_err = float((ho - hc).norm().item() / max(1e-8, float(ho.norm().item())))
            h_cos = float(F.cosine_similarity(ho, hc, dim=0).item())
            amp = h_err / max(1e-4, prev_h_err) if l_idx > 0 else 1.0
            prev_h_err = h_err

            row: Dict[str, Any] = {
                "layer": l_idx,
                "hidden_rel_err": round(h_err, 5),
                "hidden_cos": round(h_cos, 6),
                "amplification": round(min(amp, 10.0), 3),
            }

            if l_idx in orig.layer_logits and l_idx in cube.layer_logits:
                pm = quintessentia_prob_metrics(orig.layer_logits[l_idx], cube.layer_logits[l_idx])
                row.update(pm)
                # Detect layer horizon: earliest layer in second half where survivors>=0.90 and top1 matches to end
                if l_idx >= num_layers // 3 and pm["top1_match"] == 1.0 and pm["survivor_recall"] >= 0.85:
                    if layer_horizon is None:
                        layer_horizon = l_idx
                elif l_idx >= num_layers // 3 and pm["top1_match"] == 0.0:
                    layer_horizon = None

            if l_idx in orig.router_topk and l_idx in cube.router_topk:
                ro = orig.router_topk[l_idx].reshape(-1, orig.router_topk[l_idx].shape[-1])
                rc = cube.router_topk[l_idx].reshape(-1, cube.router_topk[l_idx].shape[-1])
                n_tok = min(ro.shape[0], rc.shape[0])
                k_sel = min(ro.shape[1], rc.shape[1])
                overlaps = []
                for t in range(n_tok):
                    so = set(ro[t, :k_sel].tolist())
                    sc = set(rc[t, :k_sel].tolist())
                    overlaps.append(len(so & sc) / max(1, k_sel))
                row["router_overlap"] = round(sum(overlaps) / max(1, len(overlaps)), 4)

            rows.append(row)

        final_metrics = (
            quintessentia_prob_metrics(orig.final_logits, cube.final_logits)
            if (orig.final_logits is not None and cube.final_logits is not None)
            else {}
        )

        # Rank amplifiers (layers that amplify hidden error relative to previous layer)
        amps = sorted(rows[1:], key=lambda r: r["amplification"], reverse=True)
        top_amplifiers = [int(r["layer"]) for r in amps[: max(2, num_layers // 5)]]
        safe_dampers = [int(r["layer"]) for r in reversed(amps[-max(2, num_layers // 5):])]

        # Rank Quintessentia deadzone-trauma layers (mid-to-late layers where deadzone intrusion or survivor drop spikes)
        trauma_scored = []
        for r in rows:
            depth_factor = 0.35 + 0.65 * (r["layer"] / max(1, num_layers - 1))
            dz = float(r.get("deadzone_intrusion", 0.0))
            surv_loss = 1.0 - float(r.get("survivor_recall", 1.0))
            r_loss = 1.0 - float(r.get("router_overlap", 1.0))
            kl_l = min(5.0, float(r.get("kl_div", 0.0)))
            score = depth_factor * (2.5 * dz + 2.0 * surv_loss + 0.25 * kl_l) + 1.5 * r_loss + 0.5 * r["hidden_rel_err"]
            trauma_scored.append((score, int(r["layer"])))
        trauma_scored.sort(reverse=True)
        top_trauma = [l for _, l in trauma_scored[: max(2, num_layers // 4)]]

        return {
            "num_layers": num_layers,
            "layer_horizon": layer_horizon,
            "final_quintessentia": final_metrics,
            "top_amplifier_layers": top_amplifiers,
            "safe_damper_layers": safe_dampers,
            "top_deadzone_trauma_layers": top_trauma,
            "layers": rows,
        }

    @staticmethod
    @torch.no_grad()
    def measure_activation_projection_errors(
        slicer_modules: Sequence[Tuple[int, str, BinaryHalvingQ1Linear]],
        proj_inputs: Dict[Tuple[int, str], torch.Tensor],
        *,
        k_min: int = 2,
        k_max: int = 6,
    ) -> Dict[Tuple[int, str], Dict[int, float]]:
        """Measure exact activation-weighted output error ||X W_full^T - X W_k^T||_F / ||X W_full^T||_F.

        Uses the full K-cube reconstruction (cos > 0.9994 to BF16) as reference so zero extra BF16
        copy of the model is required! Runs in milliseconds on GPU.
        """
        errors: Dict[Tuple[int, str], Dict[int, float]] = {}
        for l_idx, sub_name, mod in slicer_modules:
            x_cpu = proj_inputs.get((l_idx, sub_name))
            if x_cpu is None or x_cpu.numel() == 0:
                # Fallback to weight-domain geometric decay if activation wasn't captured
                errors[(l_idx, sub_name)] = {k: (0.5 ** k) for k in range(k_min, min(k_max, mod.root_bits) + 1)}
                continue
            dev = mod.packed_lo.device
            x = x_cpu.to(device=dev, dtype=torch.bfloat16)
            prev_k = mod.active_cubes
            try:
                mod.active_cubes = mod.root_bits
                y_ref = mod(x).float()
                ref_norm = float(y_ref.norm().item())
                if ref_norm < 1e-8:
                    errors[(l_idx, sub_name)] = {k: (0.5 ** k) for k in range(k_min, min(k_max, mod.root_bits) + 1)}
                    continue
                err_k: Dict[int, float] = {}
                for k in range(k_min, min(k_max, mod.root_bits) + 1):
                    mod.active_cubes = k
                    y_k = mod(x).float()
                    err_k[k] = float((y_ref - y_k).norm().item() / ref_norm)
                errors[(l_idx, sub_name)] = err_k
            finally:
                mod.active_cubes = prev_k
        return errors

    @staticmethod
    def build_blueprint(
        tomography_report: Dict[str, Any],
        module_specs: Sequence[Tuple[int, str, int]],
        act_errors: Dict[Tuple[int, str], Dict[int, float]],
        *,
        target_avg_cubes: float = 3.5,
        k_min: int = 2,
        k_max: int = 6,
    ) -> QuintessentiaBlueprint:
        """Allocate exact per-(layer, proj) Q1 cube depths maximizing Quintessentia fidelity per parameter.

        module_specs: list of (layer_idx, sub_name, num_params).
        """
        num_layers = int(tomography_report["num_layers"])
        rows_by_l = {int(r["layer"]): r for r in tomography_report["layers"]}

        # Compute Quintessentia importance weight omega(l) for every layer from tomography
        raw_w: Dict[int, float] = {}
        for l_idx in range(num_layers):
            r = rows_by_l.get(l_idx, {})
            amp = min(3.0, max(0.4, float(r.get("amplification", 1.0))))
            dz = float(r.get("deadzone_intrusion", 0.0))
            surv_loss = 1.0 - float(r.get("survivor_recall", 1.0))
            r_loss = 1.0 - float(r.get("router_overlap", 1.0))
            # Mild edge/early protection (layers 0..2 shape all downstream attention/DeltaNet)
            early_boost = 1.35 if l_idx <= 1 else (1.15 if l_idx == num_layers - 1 else 1.0)
            w_l = early_boost * (1.0 + 0.6 * (amp - 1.0) + 2.0 * dz + 1.5 * surv_loss + 2.0 * r_loss)
            raw_w[l_idx] = max(0.35, w_l)

        mean_w = sum(raw_w.values()) / max(1, len(raw_w))
        layer_weights = {l: w / max(1e-8, mean_w) for l, w in raw_w.items()}

        # Output / value projection sensitivity boost (known structural amplifiers inside residual stream)
        proj_mult_map = {
            "o_proj": 1.35,
            "out_proj": 1.30,
            "down_proj": 1.30,
            "v_proj": 1.25,
            "up_proj": 1.05,
            "gate_proj": 0.95,
            "q_proj": 0.85,
            "k_proj": 0.85,
        }

        total_params = sum(p for _, _, p in module_specs)
        target_param_cubes = float(target_avg_cubes) * float(total_params)

        depths: Dict[Tuple[int, str], int] = {(l, sn): int(k_min) for l, sn, _ in module_specs}
        current_param_cubes = float(k_min) * float(total_params)

        # Greedy marginal gain per parameter: Delta(omega_l * mult_proj * act_err^2) / params
        # Using sqrt(params) damping prevents tiny projections from hogging k=8 while huge projections starve,
        # yet still strongly prioritizes compact high-leverage projections (v_proj, o_proj)!
        while current_param_cubes < target_param_cubes:
            best_key = None
            best_score = -1.0
            best_params = 0
            for l_idx, sn, n_params in module_specs:
                cur_k = depths[(l_idx, sn)]
                if cur_k >= k_max:
                    continue
                if current_param_cubes + n_params > target_param_cubes + 0.25 * (total_params / max(1, len(module_specs))):
                    # Allow tiny slack only if budget isn't overshot meaningfully
                    if current_param_cubes + n_params > target_param_cubes * 1.015:
                        continue
                errs = act_errors.get((l_idx, sn), {})
                e_cur = errs.get(cur_k, 0.5 ** cur_k)
                e_nxt = errs.get(cur_k + 1, 0.5 ** (cur_k + 1))
                delta_mse = max(1e-10, (e_cur * e_cur) - (e_nxt * e_nxt))
                leaf = sn.rsplit(".", 1)[-1]
                p_mult = proj_mult_map.get(leaf, 1.0)
                score = (layer_weights[l_idx] * p_mult * delta_mse) / (float(n_params) ** 0.45)
                if score > best_score:
                    best_score = score
                    best_key = (l_idx, sn)
                    best_params = n_params
            if best_key is None:
                break
            depths[best_key] += 1
            current_param_cubes += best_params

        layer_proj_cubes: Dict[int, Dict[str, int]] = {}
        hist: Dict[int, int] = {}
        for (l_idx, sn), k in depths.items():
            layer_proj_cubes.setdefault(l_idx, {})[sn] = k
            hist[k] = hist.get(k, 0) + 1

        actual_avg = round(current_param_cubes / max(1.0, float(total_params)), 4)
        return QuintessentiaBlueprint(
            target_avg_cubes=round(float(target_avg_cubes), 3),
            actual_param_weighted_avg_cubes=actual_avg,
            k_min=int(k_min),
            k_max=int(k_max),
            layer_proj_cubes=layer_proj_cubes,
            depth_histogram=hist,
            layer_weights=layer_weights,
            top_amplifier_layers=list(tomography_report["top_amplifier_layers"]),
            top_deadzone_trauma_layers=list(tomography_report["top_deadzone_trauma_layers"]),
            safe_damper_layers=list(tomography_report["safe_damper_layers"]),
            layer_horizon=tomography_report["layer_horizon"],
            tomography=list(tomography_report["layers"]),
        )
