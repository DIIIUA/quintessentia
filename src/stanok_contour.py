"""Unified Stanok Containment Contour (`src/stanok_contour.py`).

Single architectural spine unifying all 6 contours of Air-Colibri without
duplicate VRAM limiters, duplicate A->B->C(0)->A relays, copy-pasted Mirror x UNI
math, disconnected probability objects, or orphan AST islands:

  1. `StanokContainmentContour` — ONE unbreakable physical VRAM outer wall
     backed by UUID-matched `PhysicalVRAMMonitor` (`aircolibri_cli.gpu_memory`):
       - Hard Floor (`0 OOM`, `0 WDDM Shared-RAM spill`) via proportional
         peripheral `Depth-Shedding` while protecting the hot core (`k >= 4`).
       - Hands-off Deadband Equilibrium ("Просто дай ему быть" inside the wall,
         eliminating manual `exp/ctr` ping-pong jitter).
       - Unified Snapshot Admission Guard (`can_admit_bytes`) shared with
         `GeneralSpeculation` / `RowwiseVerification` so optional recurrent
         snapshots never crush the active 3D imprint (`IC-07`).
       - Two-Phase Chaos -> Stabilization telemetry (`EQUILIBRATION` vs
         `FIXATION_SPIKE`).

  2. `MirrorUniStabilizer` — ONE canonical implementation of Mirror Equality x
     UNI (all 6 invariants: Psi, Phi, Omega, Lambda, Sigma, Delta) for network
     rest and chaos decay on `line_spine`.

  3. `UnifiedProbabilityBus` — ONE clean bus connecting the 4 distinct
     probability objects (`Phi_0` initial anchor, `SurvivorProbabilityTracker`,
     `PrefillRegionMap`/`RouteExclusion`, and `ProbabilityImprintGraph`) so
     topic shifts gently trigger equilibration without conflating their roles.

  4. `load_cube_blueprint` / `run_method1_survey` / `get_unified_silicon_registry`
     — Official Method 1 -> Method 2 bridge and unified NVRTC silicon capability
     facade uniting all Q1, Mirror, Micro-Agent, and 3D Fabric modules.
"""
from __future__ import annotations

from collections import defaultdict, deque
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 1. Canonical UNI Mathematical Invariants & Mirror Equality x UNI Stabilizer
# ---------------------------------------------------------------------------
UNI_PSI = 16.49539308107775     # piecewise exponential + integral binding energy
UNI_PHI = 0.22313016014848982   # exp(-1.5) damping / coupling factor
UNI_OMEGA = 1.5707963267948966  # pi / 2 orthogonality angle
UNI_LAMBDA = 0.005187440409386348  # canonical concentration constant
UNI_SIGMA = 1.7724538509055159  # sqrt(pi) Gaussian integral norm
UNI_DELTA = 1.8390715290764524  # integral_0^10 sin(t) dt oscillation phase


class MirrorUniStabilizer:
    """Single source of truth for Mirror Equality x UNI rest on `line_spine`."""

    @staticmethod
    def update_ortho_angle(
        dot_v: float,
        prev_norm: float,
        obs_norm: float,
        default: float = 0.5,
    ) -> float:
        if prev_norm > 1e-8 and obs_norm > 1e-8:
            ca = min(1.0, abs(dot_v) / (prev_norm * obs_norm + 1e-8))
            return math.acos(ca) / UNI_OMEGA
        return default

    @staticmethod
    @torch.no_grad()
    def compute_line_invariants(
        line_spine: torch.Tensor,
        cand_toks: Sequence[int],
    ) -> List[Tuple[float, float]]:
        """Compute `(line_norm, spectral_lambda)` on device in one batched transfer."""
        if not cand_toks:
            return []
        dev = line_spine.device
        h_dim = int(line_spine.shape[-1])
        cand_t = torch.tensor(list(cand_toks), dtype=torch.long, device=dev)
        ls_c = line_spine.index_select(0, cand_t)
        ls_norms = ls_c.norm(dim=-1)
        n_buckets = 8 if (h_dim % 8 == 0) else 1
        buckets = ls_c.pow(2).reshape(len(cand_toks), n_buckets, -1).sum(dim=-1)
        probs = buckets / (buckets.sum(dim=-1, keepdim=True) + 1e-12)
        p_cl = probs.clamp_min(1e-7)
        ent = -(p_cl * p_cl.log()).sum(dim=-1)
        lams = torch.exp(-ent)
        return [(float(a), float(b)) for a, b in torch.stack([ls_norms, lams], dim=-1).tolist()]

    @staticmethod
    def evaluate_token_rest(
        *,
        line_norm: float,
        lam_v: float,
        last_obs_norm: float,
        prev_obs_norm: float,
        last_w: float,
        next_w: float,
        rem_w: float,
        last_ortho: float,
        tok_done: int,
        rest_min: int,
        rest_tol: float,
        ref_w: Optional[float] = None,
    ) -> Tuple[bool, float, float]:
        """Return `(should_rest, defect, eff_tol)` for one token on `line_spine`.

        When `ref_w` (order-invariant mean loop weight `w_done / tok_done`) is provided,
        uses it instead of arbitrary iteration-order `last_w`.
        """
        smooth_obs = (
            last_obs_norm
            if prev_obs_norm <= 0.0
            else max(last_obs_norm, 0.5 * prev_obs_norm)
        )
        scale_ref = last_w if ref_w is None else max(ref_w, 1e-6)
        scale_next = (next_w / max(scale_ref, 1e-6)) * (1.0 + 0.5 * max(0.0, rem_w))
        defect = (smooth_obs / (line_norm + 1e-8)) * scale_next
        bind_factor = max(0.25, (1.0 - UNI_PHI * last_ortho) * (1.0 - lam_v / UNI_SIGMA))
        eff_tol = float(rest_tol) * bind_factor
        return (tok_done >= rest_min and defect < eff_tol), defect, eff_tol

    @staticmethod
    def evaluate_loop_continuation(
        *,
        line_norm: float,
        lam_v: float,
        last_obs_norm: float,
        prev_obs_norm: float,
        w_done: float,
        next_w: float,
        rem_w: float,
        last_ortho: float,
        tok_done: int,
        rest_min: int,
        rest_tol: float,
        closes_new_loop: bool,
        min_ring_count: Optional[int] = None,
        min_ring_energy: Optional[float] = None,
        delta_ring_count: Optional[int] = None,
        delta_ring_energy: Optional[float] = None,
    ) -> Tuple[bool, float, float, str]:
        """Loop/Ring-centric continuation law (orient around MINIMUM AGENTS & GENERATED RINGS).

        1. If the active minimum is NOT yet established/dominant (`tok_done < rest_min` or
           `w_done < rem_w`) -> `"loop_open"` (MUST wake next agent to form minimum/ring).
        2. When explicit Ring feedback `(min_ring_*, delta_ring_*)` is supplied:
           - If the minimum ALREADY generated closed rings (`min_ring_count > 0`) and adding
             the continuation generates ZERO new closed rings (`delta_ring_count == 0`) ->
             `"no_new_loop"` (leave asleep, 0 B disk).
           - If the continuation DOES generate new closed rings (`delta_ring_count > 0`),
             compare the new ring energy BACK against the minimum's generated rings:
             `ring_ratio = delta_ring_energy / min_ring_energy` (or `1.0` if this continuation
             closes the minimum's very first ring!). Scale `eff_tol` strictly by `ring_ratio`
             against `UNI_PHI` so strong new rings expand cleanly while tiny ripples settle.
        3. Otherwise (V1 fallback without ring counts): use `closes_new_loop` + order-invariant
           Mirror x UNI referenced to `w_done / tok_done`.
        """
        if tok_done < rest_min or w_done < rem_w:
            return False, float("inf"), float(rest_tol), "loop_open"

        has_ring_cmp = (min_ring_count is not None and delta_ring_count is not None)
        if has_ring_cmp:
            if min_ring_count > 0 and delta_ring_count == 0:
                return True, 0.0, 0.0, "no_new_loop"
            if min_ring_count == 0 and delta_ring_count == 0 and not closes_new_loop:
                return True, 0.0, 0.0, "no_new_loop"
        elif not closes_new_loop:
            return True, 0.0, 0.0, "no_new_loop"

        mean_loop_w = w_done / max(1, tok_done)
        should_rest, defect, eff_tol = MirrorUniStabilizer.evaluate_token_rest(
            line_norm=line_norm,
            lam_v=lam_v,
            last_obs_norm=last_obs_norm,
            prev_obs_norm=prev_obs_norm,
            last_w=mean_loop_w,
            ref_w=mean_loop_w,
            next_w=next_w,
            rem_w=rem_w,
            last_ortho=last_ortho,
            tok_done=tok_done,
            rest_min=rest_min,
            rest_tol=rest_tol,
        )
        if has_ring_cmp and delta_ring_count > 0:
            m_e = float(min_ring_energy or 0.0)
            d_e = float(delta_ring_energy or 0.0)
            if min_ring_count == 0 or m_e <= 1e-12:
                # Continuation closes the very first ring of an open minimum -> strong expansion signal
                ring_ratio = 1.0
            else:
                ring_ratio = d_e / m_e
            if ring_ratio >= UNI_PHI:
                # Significant new ring compared to minimum -> tighten rest tolerance (easier to expand)
                eff_tol = eff_tol / (1.0 + min(2.0, ring_ratio))
            else:
                # Tiny ring ripple compared to already-strong minimum rings -> easier to settle
                eff_tol = eff_tol * (1.0 + (UNI_PHI - ring_ratio) / UNI_PHI)
            should_rest = (tok_done >= rest_min and defect < eff_tol)
        return should_rest, defect, eff_tol, ("loop_settled" if should_rest else "loop_expand")


# ---------------------------------------------------------------------------
# 2. StanokContainmentContour — ONE Unbreakable Outer Wall + Deadband Law
# ---------------------------------------------------------------------------
class StanokContainmentContour:
    """Single model-agnostic physical VRAM containment contour shared across the entire engine.

    Pure hardware conservation law (ZERO model-specific multipliers, tick timers, or
    allocator-pool fantasy inflation):
      - `safety_reserve_bytes`: scale-free physical floor (`reserve_ratio * total_vram`, default 10%).
      - `deadband_bytes`: scale-free equilibrium band (`safety_reserve_bytes // 4`, ~2.5% of GPU).
      - Below floor (`free_vram < safety_reserve`): reclaim idle PyTorch pool, then shed ONLY the
        exact physical deficit (`safety_reserve - free_vram`) -> 0 OOM, 0 WDDM Shared-RAM spill.
      - Above floor (`free_vram >= safety_reserve`): true physical ceiling is simply
        `cached_bytes + (free_vram - safety_reserve)`; open cleanly when headroom exceeds deadband,
        otherwise hold hands-off so the coupled lattice inside expands and self-collapses freely.
    """

    def __init__(
        self,
        device: Any = None,
        *,
        enabled: bool = True,
        safety_reserve_bytes: Optional[int] = None,
        reserve_ratio: float = 0.10,
        expand_margin_bytes: Optional[int] = None,
    ) -> None:
        self.device = torch.device("cpu" if device is None else device)
        self.enabled = bool(enabled and self.device.type == "cuda" and torch.cuda.is_available())
        self.reserve_ratio = max(0.02, min(0.40, float(reserve_ratio)))
        self._monitor = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            from aircolibri_cli.gpu_memory import PhysicalVRAMMonitor
            self._monitor = PhysicalVRAMMonitor(self.device)

        if safety_reserve_bytes is not None:
            self.safety_reserve_bytes = max(1, int(safety_reserve_bytes))
        elif self.device.type == "cuda" and torch.cuda.is_available():
            _fb, total_vram, _src = self.query_physical_vram()
            self.safety_reserve_bytes = max(1, int(total_vram * self.reserve_ratio))
        else:
            self.safety_reserve_bytes = 0

        # Scale-free deadband (25% of safety_reserve ~= 2.5% of total GPU VRAM)
        self.deadband_bytes = (
            max(1, int(expand_margin_bytes))
            if expand_margin_bytes is not None
            else max(1, self.safety_reserve_bytes // 4)
        )

        # Telemetry
        self.expansions: int = 0
        self.contractions: int = 0
        self.holds_in_deadband: int = 0
        self.oom_recoveries: int = 0
        self.snapshot_admissions: int = 0
        self.snapshot_denials: int = 0
        self.min_observed_free_vram_bytes: Optional[int] = None
        self.max_observed_budget_bytes: int = 0

        # Chaos -> Stabilization phase tracking
        self.fixation_layers: int = 0
        self.equilibration_layers: int = 0
        self.fixation_experts_skipped: int = 0
        self.topic_shift_pulses: int = 0
        self.lattice_self_collapses: int = 0
        self._recent_fixation_window: List[bool] = []

    @property
    def sensor_source(self) -> str:
        if self._monitor is not None and self._monitor.nvml_available:
            return "nvml"
        return "cuda" if self.device.type == "cuda" else "cpu"

    def query_physical_vram(self) -> Tuple[int, int, str]:
        """Return `(free_bytes, total_bytes, source)` via UUID-matched NVML or CUDA."""
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return 0, 0, "cpu"
        if self._monitor is not None:
            return self._monitor.query()
        fb, tb = torch.cuda.mem_get_info(self.device)
        return int(fb), int(tb), "cuda"

    def query_memory_snapshot(self) -> Dict[str, Any]:
        """Return real physical free VRAM + PyTorch allocator pool stats."""
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return {
                "source": "cpu",
                "free_bytes": 0,
                "total_bytes": 0,
                "torch_reserved_bytes": 0,
                "torch_allocated_bytes": 0,
                "torch_pool_free_bytes": 0,
                "effective_free_bytes": 0,
            }
        free_b, total_b, src = self.query_physical_vram()
        reserved_b = int(torch.cuda.memory_reserved(self.device))
        allocated_b = int(torch.cuda.memory_allocated(self.device))
        pool_free = max(0, reserved_b - allocated_b)
        return {
            "source": src,
            "free_bytes": free_b,
            "total_bytes": total_b,
            "torch_reserved_bytes": reserved_b,
            "torch_allocated_bytes": allocated_b,
            "torch_pool_free_bytes": pool_free,
            "effective_free_bytes": free_b,
        }

    def evaluate_budget(
        self,
        *,
        cached_bytes: int,
        current_budget_bytes: int,
    ) -> Tuple[str, int, bool]:
        """Evaluate pure physical VRAM conservation (ZERO model-specific tuning).

        Returns `(action, target_budget_bytes, emergency)`:
          - `"contract"` : real physical free VRAM fell below `safety_reserve_bytes` ->
                           reclaim idle PyTorch pool first, then shed ONLY exact deficit.
          - `"expand"`   : real physical ceiling `cached + (free - reserve)` exceeds
                           `current_budget + deadband` -> open cleanly to real ceiling.
          - `"hold"`     : inside equilibrium deadband -> hands-off.
        """
        if not self.enabled or self.device.type != "cuda":
            return "hold", int(current_budget_bytes), False

        snap = self.query_memory_snapshot()
        free_b = int(snap["free_bytes"])
        pool_free = int(snap["torch_pool_free_bytes"])
        if self.min_observed_free_vram_bytes is None or free_b < self.min_observed_free_vram_bytes:
            self.min_observed_free_vram_bytes = free_b

        reserve = self.safety_reserve_bytes

        # Zone 1: OUTER WALL CONTACT (pure physical conservation — 0 OOM, 0 WDDM spill)
        if free_b < reserve:
            if pool_free > 0:
                torch.cuda.empty_cache()
                free_b, _tb, _src = self.query_physical_vram()
            if free_b < reserve:
                deficit = reserve - free_b
                emergency = free_b < (reserve // 2)
                new_budget = max(0, int(cached_bytes) - deficit)
                self.contractions += 1
                return "contract", new_budget, emergency

        # Zone 3: GENUINE PHYSICAL HEADROOM ABOVE DEADBAND (real NVML free only — no pool inflation!)
        headroom = free_b - reserve
        physical_ceiling = int(cached_bytes) + headroom
        if physical_ceiling > int(current_budget_bytes) + self.deadband_bytes:
            self.expansions += 1
            if physical_ceiling > self.max_observed_budget_bytes:
                self.max_observed_budget_bytes = physical_ceiling
            return "expand", physical_ceiling, False

        # Zone 2: HANDS-OFF EQUILIBRIUM DEADBAND
        self.holds_in_deadband += 1
        return "hold", int(current_budget_bytes), False

    def can_admit_bytes(self, needed_bytes: int, *, reserve_margin_bytes: int = 0) -> bool:
        """Unified admission guard for optional buffers (e.g. RowwiseVerification snapshots)."""
        needed = max(0, int(needed_bytes))
        if needed == 0 or not self.enabled or self.device.type != "cuda":
            self.snapshot_admissions += 1
            return True
        snap = self.query_memory_snapshot()
        free_b = int(snap["free_bytes"])
        required_floor = self.safety_reserve_bytes + max(0, int(reserve_margin_bytes))
        if free_b - needed >= required_floor:
            self.snapshot_admissions += 1
            return True
        self.snapshot_denials += 1
        return False

    def record_oom_recovery(self) -> None:
        self.oom_recoveries += 1

    def record_lattice_self_collapse(self, count: int = 1) -> None:
        self.lattice_self_collapses += max(0, int(count))

    def record_layer_phase(self, *, fixated: bool, skipped_experts: int = 0) -> None:
        if fixated:
            self.fixation_layers += 1
            self.fixation_experts_skipped += max(0, int(skipped_experts))
        else:
            self.equilibration_layers += 1
        self._recent_fixation_window.append(bool(fixated))
        if len(self._recent_fixation_window) > 64:
            self._recent_fixation_window.pop(0)

    def record_topic_shift(self) -> None:
        self.topic_shift_pulses += 1

    @property
    def current_phase(self) -> str:
        if not self._recent_fixation_window:
            return "IDLE"
        fixed_frac = sum(self._recent_fixation_window) / len(self._recent_fixation_window)
        return "FIXATION_SPIKE" if fixed_frac >= 0.65 else "EQUILIBRATION"

    def stats(self) -> Dict[str, Any]:
        total_layers = self.fixation_layers + self.equilibration_layers
        return {
            "enabled": self.enabled,
            "device": str(self.device),
            "vram_sensor": self.sensor_source,
            "safety_reserve_mib": round(self.safety_reserve_bytes / 1024**2, 1),
            "deadband_mib": round(self.deadband_bytes / 1024**2, 1),
            "vram_expansions": self.expansions,
            "vram_contractions": self.contractions,
            "deadband_holds": self.holds_in_deadband,
            "lattice_self_collapses": self.lattice_self_collapses,
            "oom_recoveries": self.oom_recoveries,
            "snapshot_admissions": self.snapshot_admissions,
            "snapshot_denials": self.snapshot_denials,
            "min_observed_free_vram_mib": (
                None if self.min_observed_free_vram_bytes is None
                else round(self.min_observed_free_vram_bytes / 1024**2, 1)
            ),
            "max_observed_budget_mib": round(self.max_observed_budget_bytes / 1024**2, 1),
            "phase": self.current_phase,
            "fixation_layers": self.fixation_layers,
            "equilibration_layers": self.equilibration_layers,
            "fixation_ratio_pct": round(100.0 * self.fixation_layers / max(1, total_layers), 2),
            "fixation_experts_skipped": self.fixation_experts_skipped,
            "topic_shift_pulses": self.topic_shift_pulses,
        }

    def close(self) -> None:
        if self._monitor is not None:
            self._monitor.close()
            self._monitor = None


# ---------------------------------------------------------------------------
# 3. UnifiedProbabilityBus — Cleanly Connects All 4 Probability Objects (IC-05)
# ---------------------------------------------------------------------------
class UnifiedProbabilityBus:
    """Typed coordination bus keeping the 4 probability organs distinct yet connected:
      1. `initial_anchor`   : Phi_0 / ThoughtBuffer (immutable task anchor + user edits)
      2. `survivor_tracker` : SurvivorProbabilityTracker (L_0 -> L_last surviving path)
      3. `route_exclusion`  : PrefillRegionMap / RouteExclusion (soil + proven P=0)
      4. `imprint_overlay`  : ImprintPuzzleMoEOverlay / ProbabilityImprintGraph (3D puzzle)
    """

    def __init__(self, contour: Optional[StanokContainmentContour] = None) -> None:
        self.contour = contour
        self.initial_anchor: Any = None
        self.survivor_tracker: Any = None
        self.route_exclusion: Any = None
        self.imprint_overlay: Any = None
        self.topic_shifts_propagated: int = 0
        self.last_topic_shift_reason: Optional[str] = None

    def bind(
        self,
        *,
        initial_anchor: Any = None,
        survivor_tracker: Any = None,
        route_exclusion: Any = None,
        imprint_overlay: Any = None,
    ) -> "UnifiedProbabilityBus":
        if initial_anchor is not None:
            self.initial_anchor = initial_anchor
        if survivor_tracker is not None:
            self.survivor_tracker = survivor_tracker
            setattr(survivor_tracker, "probability_bus", self)
        if route_exclusion is not None:
            self.route_exclusion = route_exclusion
        if imprint_overlay is not None:
            self.imprint_overlay = imprint_overlay
            setattr(imprint_overlay, "probability_bus", self)
        return self

    def notify_topic_shift(self, layer: Optional[int] = None, *, reason: str = "route_revision") -> None:
        """Propagate a semantic/topic shift from SurvivorTracker/Phi_0 into the 3D Imprint."""
        self.topic_shifts_propagated += 1
        self.last_topic_shift_reason = str(reason)
        if self.contour is not None:
            self.contour.record_topic_shift()
        if self.imprint_overlay is not None and hasattr(self.imprint_overlay, "on_topic_shift"):
            self.imprint_overlay.on_topic_shift(layer=layer)

    def stats(self) -> Dict[str, Any]:
        graphify_on = bool(
            self.imprint_overlay is not None
            and getattr(self.imprint_overlay, "graphify_feedback", False)
        )
        return {
            "wired_organs": {
                "initial_anchor_phi0": self.initial_anchor is not None,
                "survivor_path": self.survivor_tracker is not None,
                "route_exclusion_p0": self.route_exclusion is not None,
                "imprint_3d_puzzle": self.imprint_overlay is not None,
            },
            "online_graphify_feedback": graphify_on,
            "topic_shifts_propagated": self.topic_shifts_propagated,
            "last_topic_shift_reason": self.last_topic_shift_reason,
        }


# ---------------------------------------------------------------------------
# 3.5 OnlineGraphifyTopology — Real-Time Modularity, Cores & Inter-Community
#     Bridges Embedded Inside Prefill, Probabilistic Cube Choice & Decode Loop
# ---------------------------------------------------------------------------
GraphNode = Tuple[int, int]


class OnlineGraphifyTopology:
    """Fast (~0.5-3 ms), zero-VRAM, model-agnostic internal Graphify feedback engine.

    Runs deterministic modularity maximization (Louvain/Leiden-style local moving +
    connected-component refinement) directly on the live weighted 3D intersection graph
    (intra-layer faces + interlayer 90° corners) and feeds structural roles back into:
      1. Prefill cube depth & wave selection (protecting inter-community bridges even
         when their scalar router weight is modest, while damping isolated noise);
      2. Probabilistic Q1 cube depth allocation (`core`, `bridge`, `periphery`, `isolated`);
      3. Decode loop-closure gating, bridge-aware VRAM shedding protection, and
         community-aware lattice self-collapse.

    Uses ONLY the canonical scale-free `UNI_PHI = exp(-1.5)` threshold and standard
    resolution `gamma = 1.0` — zero model-specific knobs.
    """

    def __init__(self, *, gamma: float = 1.0, max_iters: int = 6) -> None:
        self.gamma: float = max(0.1, float(gamma))
        self.max_iters: int = max(1, int(max_iters))

        self.community_of: Dict[GraphNode, int] = {}
        self.community_sizes: Dict[int, int] = {}
        self.core_cohesion: Dict[GraphNode, float] = {}
        self.bridge_strength: Dict[GraphNode, float] = {}
        self.bridge_targets: Dict[GraphNode, Tuple[int, ...]] = {}
        self.redundancy: Dict[GraphNode, float] = {}
        self.role_of: Dict[GraphNode, str] = {}
        self.layer_node_counts: Dict[int, int] = {}

        self.num_nodes: int = 0
        self.num_edges: int = 0
        self.num_communities: int = 0
        self.modularity_q: float = 0.0
        self.refreshes: int = 0
        self.total_refresh_ms: float = 0.0
        self.last_refresh_ms: float = 0.0
        self._last_edge_count: int = 0
        self._last_node_count: int = 0

        # Zero-delay incremental state for mutual Coalition <-> Agent generation (`mutual_step`)
        self._adj: Dict[GraphNode, Dict[GraphNode, float]] = defaultdict(dict)
        self._deg: Dict[GraphNode, float] = defaultdict(float)
        self._sigma_tot: Dict[int, float] = defaultdict(float)
        self._two_m: float = 0.0
        self._next_cid: int = 0
        self.mutual_steps: int = 0
        self.total_mutual_ms: float = 0.0
        self.last_mutual_ms: float = 0.0

    def mutual_step(
        self,
        active_nodes: Iterable[GraphNode],
        edge_updates: Sequence[Tuple[GraphNode, GraphNode, float]],
        *,
        local_iters: int = 2,
    ) -> Dict[GraphNode, Tuple[str, float, float, float, int]]:
        """Zero-delay mutual generation step strictly between active Agents and their Coalitions.

        Runs in O(|active_nodes| * avg_deg) — NEVER scans sleeping nodes and NEVER touches
        external subsystems (VRAM contour, shedding, prefetch). Updates local coalition
        membership synchronously and returns immediate structural feedback ONLY to the
        participating agents and their coalitions.
        """
        t0 = time.perf_counter()
        active_set: set[GraphNode] = set(active_nodes)

        def _ensure_node(u: GraphNode) -> None:
            if u not in self.community_of:
                cid = self._next_cid
                self._next_cid += 1
                self.community_of[u] = cid
                self.community_sizes[cid] = 1
                self._sigma_tot[cid] = 0.0
                self.role_of[u] = "isolated"
                self.core_cohesion[u] = 0.0
                self.bridge_strength[u] = 0.0
                self.bridge_targets[u] = ()
                self.redundancy[u] = 0.0
                li = u[0]
                self.layer_node_counts[li] = self.layer_node_counts.get(li, 0) + 1
                self.num_nodes += 1

        # 1. Incremental O(1)-per-edge ingestion (preserves exact `two_m` & `sigma_tot`)
        for u, v, dw in edge_updates:
            if u == v or dw <= 1e-12:
                continue
            _ensure_node(u)
            _ensure_node(v)
            row_u = self._adj[u]
            if v not in row_u:
                self.num_edges += 1
            row_u[v] = row_u.get(v, 0.0) + dw
            row_v = self._adj[v]
            row_v[u] = row_v.get(u, 0.0) + dw
            self._deg[u] += dw
            self._deg[v] += dw
            self._two_m += 2.0 * dw
            self._sigma_tot[self.community_of[u]] += dw
            self._sigma_tot[self.community_of[v]] += dw
            active_set.add(u)
            active_set.add(v)

        for u in list(active_set):
            if u in self._deg and self._deg[u] > 1e-12:
                _ensure_node(u)

        live_active = [u for u in active_set if self._deg.get(u, 0.0) > 1e-12]
        two_m = self._two_m
        if not live_active or two_m <= 1e-12:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.mutual_steps += 1
            self.last_mutual_ms = dt_ms
            self.total_mutual_ms += dt_ms
            return {}

        # 2. Local modularity update ONLY on currently active agents (Agents -> Coalitions)
        live_active.sort(key=lambda u: (-self._deg[u], u[0], u[1]))
        gamma = self.gamma
        comm = self.community_of
        sizes = self.community_sizes
        sigma = self._sigma_tot
        adj = self._adj
        deg = self._deg

        for _ in range(max(1, int(local_iters))):
            moved = 0
            for u in live_active:
                k_u = deg[u]
                c_old = comm[u]
                old_sz = sizes.get(c_old, 1)
                sigma[c_old] = max(0.0, sigma[c_old] - k_u)
                sizes[c_old] = old_sz - 1
                if old_sz == 2:
                    self.num_communities -= 1

                k_to_c: Dict[int, float] = {}
                for v, w in adj[u].items():
                    cv = comm[v]
                    k_to_c[cv] = k_to_c.get(cv, 0.0) + w

                best_c = -1
                best_gain = 0.0
                if sizes[c_old] > 0:
                    g_old = two_m * k_to_c.get(c_old, 0.0) - gamma * k_u * sigma[c_old]
                    if g_old > best_gain:
                        best_gain = g_old
                        best_c = c_old

                for cand_c, k_u_c in k_to_c.items():
                    gain = two_m * k_u_c - gamma * k_u * sigma[cand_c]
                    if gain > best_gain + 1e-12 or (
                        abs(gain - best_gain) <= 1e-12
                        and best_c != -1
                        and cand_c < best_c
                    ):
                        best_gain = gain
                        best_c = cand_c

                if best_c == -1:
                    if sizes[c_old] == 0:
                        best_c = c_old
                    else:
                        best_c = self._next_cid
                        self._next_cid += 1
                        sizes[best_c] = 0
                        sigma[best_c] = 0.0

                comm[u] = best_c
                sigma[best_c] += k_u
                new_sz = sizes.get(best_c, 0) + 1
                sizes[best_c] = new_sz
                if new_sz == 2:
                    self.num_communities += 1
                if best_c != c_old:
                    moved += 1
            if moved == 0:
                break

        # 3. Immediate local feedback ONLY back to participating agents (Coalitions -> Agents)
        feedback_out: Dict[GraphNode, Tuple[str, float, float, float, int]] = {}
        for u in live_active:
            cu = comm[u]
            c_sz = sizes.get(cu, 1)
            k_u = deg[u]
            nbrs = adj[u]
            kin = 0.0
            kout = 0.0
            ext_c: Dict[int, float] = {}
            nbr_deg_sum = 0.0
            for v, w in nbrs.items():
                nbr_deg_sum += deg[v]
                cv = comm[v]
                if cv == cu:
                    kin += w
                else:
                    kout += w
                    ext_c[cv] = ext_c.get(cv, 0.0) + w
            nbr_mean_deg = nbr_deg_sum / max(1, len(nbrs))
            rel_deg = min(1.0, k_u / max(1e-12, 0.5 * nbr_mean_deg))

            if c_sz >= 2 and k_u > 1e-12:
                core_v = min(1.0, (kin / k_u) * rel_deg)
            else:
                core_v = 0.0

            if kin > 1e-12 and kout > 1e-12:
                balance = 4.0 * kin * kout / ((kin + kout) ** 2)
                multi = min(1.35, 1.0 + 0.175 * max(0, len(ext_c) - 1))
                bridge_v = min(1.0, balance * multi * rel_deg)
            elif kin <= 1e-12 and len(ext_c) >= 2 and kout > 1e-12:
                max_share = max(ext_c.values()) / kout
                spread = min(1.0, 2.0 * (1.0 - max_share))
                bridge_v = min(1.0, spread * rel_deg)
            else:
                bridge_v = 0.0

            if c_sz >= 3 and bridge_v < UNI_PHI and core_v < UNI_PHI:
                red_v = min(1.0, max(0.0, 1.0 - max(core_v, bridge_v)))
            else:
                red_v = 0.0

            if bridge_v >= UNI_PHI:
                role = "bridge"
            elif core_v >= UNI_PHI:
                role = "core"
            elif c_sz >= 2:
                role = "periphery"
            else:
                role = "isolated"

            self.core_cohesion[u] = core_v
            self.bridge_strength[u] = bridge_v
            self.bridge_targets[u] = tuple(sorted(ext_c.keys(), key=lambda c_id: -ext_c[c_id]))
            self.redundancy[u] = red_v
            self.role_of[u] = role
            feedback_out[u] = (role, core_v, bridge_v, red_v, cu)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.mutual_steps += 1
        self.last_mutual_ms = dt_ms
        self.total_mutual_ms += dt_ms
        return feedback_out

    def compute_current_modularity(self) -> float:
        """Compute exact Newman-Girvan modularity Q on demand (used only for export/reporting)."""
        if self._two_m <= 1e-12 or not self._adj:
            return self.modularity_q
        e_in: Dict[int, float] = defaultdict(float)
        for u, nbrs in self._adj.items():
            cu = self.community_of.get(u)
            if cu is None:
                continue
            for v, w in nbrs.items():
                if self.community_of.get(v) == cu:
                    e_in[cu] += w
        two_m = self._two_m
        mod_q = 0.0
        for cid, s_tot in self._sigma_tot.items():
            if s_tot > 0.0:
                mod_q += (e_in.get(cid, 0.0) / two_m) - self.gamma * ((s_tot / two_m) ** 2)
        self.modularity_q = float(mod_q)
        return self.modularity_q

    def refresh(
        self,
        adj: Dict[GraphNode, Dict[GraphNode, float]],
        *,
        cold_start: bool = False,
        fast_warm: bool = False,
    ) -> Dict[str, Any]:
        """Recompute communities, modularity Q, cores, bridges, and redundancies."""
        t0 = time.perf_counter()

        # Filter to live nodes with positive incident edges
        nodes: List[GraphNode] = []
        deg: Dict[GraphNode, float] = {}
        edge_slots = 0
        for u, nbrs in adj.items():
            if not nbrs:
                continue
            d_u = 0.0
            for v, w in nbrs.items():
                if v != u and w > 1e-12:
                    d_u += w
                    edge_slots += 1
            if d_u > 1e-12:
                nodes.append(u)
                deg[u] = d_u

        two_m = sum(deg.values())
        if len(nodes) < 2 or two_m <= 1e-12:
            self.community_of.clear()
            self.community_sizes.clear()
            self.core_cohesion.clear()
            self.bridge_strength.clear()
            self.bridge_targets.clear()
            self.redundancy.clear()
            self.role_of.clear()
            self.layer_node_counts.clear()
            self.num_nodes = len(nodes)
            self.num_edges = edge_slots // 2
            self.num_communities = 0
            self.modularity_q = 0.0
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.refreshes += 1
            self.last_refresh_ms = dt_ms
            self.total_refresh_ms += dt_ms
            self._last_node_count = self.num_nodes
            self._last_edge_count = self.num_edges
            return self.stats()

        # Deterministic hub-first node order: stabilizes cores in 1-2 sweeps
        nodes.sort(key=lambda u: (-deg[u], u[0], u[1]))
        n_nodes = len(nodes)
        node_idx: Dict[GraphNode, int] = {u: i for i, u in enumerate(nodes)}
        base_adj: List[Dict[int, float]] = [{} for _ in range(n_nodes)]
        base_deg: List[float] = [deg[u] for u in nodes]
        for i, u in enumerate(nodes):
            row = base_adj[i]
            for v, w in adj[u].items():
                if v != u and w > 1e-12:
                    j = node_idx.get(v)
                    if j is not None:
                        row[j] = row.get(j, 0.0) + w

        gamma = self.gamma

        def _local_move(
            adj_lvl: List[Dict[int, float]],
            deg_lvl: List[float],
            init_c: Optional[List[int]] = None,
            iters: int = 6,
        ) -> Tuple[List[int], int]:
            n_lvl = len(adj_lvl)
            if init_c is None:
                c_arr = list(range(n_lvl))
                next_id = n_lvl
            else:
                c_arr = list(init_c)
                next_id = (max(c_arr) + 1) if c_arr else 0
            sig: Dict[int, float] = defaultdict(float)
            cnt: Dict[int, int] = defaultdict(int)
            for idx_u, c_u in enumerate(c_arr):
                sig[c_u] += deg_lvl[idx_u]
                cnt[c_u] += 1

            order = sorted(range(n_lvl), key=lambda idx_u: (-deg_lvl[idx_u], idx_u))
            total_moves = 0
            for _ in range(iters):
                moved = 0
                for idx_u in order:
                    k_u = deg_lvl[idx_u]
                    c_old = c_arr[idx_u]
                    sig[c_old] -= k_u
                    cnt[c_old] -= 1

                    k_to_c: Dict[int, float] = {}
                    for idx_v, w_uv in adj_lvl[idx_u].items():
                        cv = c_arr[idx_v]
                        k_to_c[cv] = k_to_c.get(cv, 0.0) + w_uv

                    best_c = -1
                    best_gain = 0.0
                    if cnt[c_old] > 0:
                        g_old = two_m * k_to_c.get(c_old, 0.0) - gamma * k_u * sig[c_old]
                        if g_old > best_gain:
                            best_gain = g_old
                            best_c = c_old

                    for cand_c, k_u_c in k_to_c.items():
                        gain = two_m * k_u_c - gamma * k_u * sig[cand_c]
                        if gain > best_gain + 1e-12 or (
                            abs(gain - best_gain) <= 1e-12
                            and best_c != -1
                            and cand_c < best_c
                        ):
                            best_gain = gain
                            best_c = cand_c

                    if best_c == -1:
                        if cnt[c_old] == 0:
                            best_c = c_old
                        else:
                            best_c = next_id
                            next_id += 1

                    c_arr[idx_u] = best_c
                    sig[best_c] += k_u
                    cnt[best_c] += 1
                    if best_c != c_old:
                        moved += 1
                total_moves += moved
                if moved == 0:
                    break
            return c_arr, total_moves

        # Optional warm-start community labels on incremental decode
        init_level0: Optional[List[int]] = None
        if not cold_start and self.community_of:
            used_c = set()
            tmp_c: List[int] = [-1] * n_nodes
            for i, u in enumerate(nodes):
                pc = self.community_of.get(u)
                if pc is not None:
                    tmp_c[i] = pc
                    used_c.add(pc)
            nxt = (max(used_c) + 1) if used_c else 0
            for i in range(n_nodes):
                if tmp_c[i] == -1:
                    tmp_c[i] = nxt
                    nxt += 1
            init_level0 = tmp_c

        # Multi-level Louvain/Leiden coarsening (or fast level-0 warm polish on incremental decode)
        if fast_warm and init_level0 is not None:
            final_c_idx, _ = _local_move(base_adj, base_deg, init_c=init_level0, iters=2)
        else:
            node_to_super = list(range(n_nodes))
            cur_adj = base_adj
            cur_deg = base_deg
            cur_self = [0.0] * n_nodes
            for lvl in range(4):
                c_lvl, _moves = _local_move(
                    cur_adj,
                    cur_deg,
                    init_c=(init_level0 if lvl == 0 else None),
                    iters=self.max_iters,
                )
                remap: Dict[int, int] = {}
                for c_val in c_lvl:
                    if c_val not in remap:
                        remap[c_val] = len(remap)
                k_super = len(remap)
                mapped_lvl = [remap[c_val] for c_val in c_lvl]
                node_to_super = [mapped_lvl[s] for s in node_to_super]
                if k_super >= len(cur_adj) or k_super <= 1:
                    break
                # Aggregate into super-graph preserving exact `two_m` invariant
                next_adj: List[Dict[int, float]] = [{} for _ in range(k_super)]
                next_self: List[float] = [0.0] * k_super
                for idx_u, nbrs_u in enumerate(cur_adj):
                    su = mapped_lvl[idx_u]
                    next_self[su] += cur_self[idx_u]
                    row_su = next_adj[su]
                    for idx_v, w_uv in nbrs_u.items():
                        sv = mapped_lvl[idx_v]
                        if su == sv:
                            next_self[su] += w_uv
                        else:
                            row_su[sv] = row_su.get(sv, 0.0) + w_uv
                next_deg = [next_self[s] + sum(next_adj[s].values()) for s in range(k_super)]
                cur_adj = next_adj
                cur_self = next_self
                cur_deg = next_deg

            # Final level-0 polish so inter-community bridge boundaries sit cleanly
            final_c_idx, _ = _local_move(base_adj, base_deg, init_c=node_to_super, iters=2)
        comm: Dict[GraphNode, int] = {nodes[i]: final_c_idx[i] for i in range(n_nodes)}
        node_set = set(nodes)

        # Phase 2: Leiden-style connected-component refinement (no disconnected communities)
        by_comm: Dict[int, List[GraphNode]] = defaultdict(list)
        for u in nodes:
            by_comm[comm[u]].append(u)

        refined_groups: List[List[GraphNode]] = []
        for _cid, grp in by_comm.items():
            if len(grp) <= 2:
                refined_groups.append(grp)
                continue
            grp_set = set(grp)
            visited: set[GraphNode] = set()
            for start in grp:
                if start in visited:
                    continue
                comp: List[GraphNode] = []
                q = deque([start])
                visited.add(start)
                while q:
                    cur = q.popleft()
                    comp.append(cur)
                    for nb, w in adj[cur].items():
                        if w > 1e-12 and nb in grp_set and nb not in visited:
                            visited.add(nb)
                            q.append(nb)
                refined_groups.append(comp)

        # Deterministic compact re-indexing by descending total community weight
        refined_groups.sort(key=lambda g: (-sum(deg[x] for x in g), min(g)))
        new_comm: Dict[GraphNode, int] = {}
        new_sizes: Dict[int, int] = {}
        new_sigma: Dict[int, float] = {}
        for new_id, grp in enumerate(refined_groups):
            new_sizes[new_id] = len(grp)
            s_w = 0.0
            for u in grp:
                new_comm[u] = new_id
                s_w += deg[u]
            new_sigma[new_id] = s_w

        # Per-node internal / external weights, modularity Q, and per-layer scale normalization
        k_in_map: Dict[GraphNode, float] = {}
        k_out_map: Dict[GraphNode, float] = {}
        ext_comms_map: Dict[GraphNode, Dict[int, float]] = {}
        comm_peak_in: Dict[int, float] = defaultdict(float)
        layer_comm_peak_in: Dict[Tuple[int, int], float] = defaultdict(float)
        comm_e_in: Dict[int, float] = defaultdict(float)
        layer_deg_sum: Dict[int, float] = defaultdict(float)
        layer_cnt: Dict[int, int] = defaultdict(int)

        for u in nodes:
            cu = new_comm[u]
            kin = 0.0
            kout = 0.0
            ext_c: Dict[int, float] = {}
            for v, w in adj[u].items():
                if v != u and w > 1e-12 and v in node_set:
                    cv = new_comm[v]
                    if cv == cu:
                        kin += w
                    else:
                        kout += w
                        ext_c[cv] = ext_c.get(cv, 0.0) + w
            k_in_map[u] = kin
            k_out_map[u] = kout
            ext_comms_map[u] = ext_c
            comm_e_in[cu] += kin
            if kin > comm_peak_in[cu]:
                comm_peak_in[cu] = kin
            li = u[0]
            if kin > layer_comm_peak_in[(li, cu)]:
                layer_comm_peak_in[(li, cu)] = kin
            layer_deg_sum[li] += deg[u]
            layer_cnt[li] += 1

        mod_q = 0.0
        for cid, s_tot in new_sigma.items():
            mod_q += (comm_e_in[cid] / two_m) - gamma * ((s_tot / two_m) ** 2)

        # Compute dimensionless structural roles in [0, 1]
        new_core: Dict[GraphNode, float] = {}
        new_bridge: Dict[GraphNode, float] = {}
        new_targets: Dict[GraphNode, Tuple[int, ...]] = {}
        new_red: Dict[GraphNode, float] = {}
        new_role: Dict[GraphNode, str] = {}

        for u in nodes:
            cu = new_comm[u]
            c_sz = new_sizes[cu]
            k_u = deg[u]
            kin = k_in_map[u]
            kout = k_out_map[u]
            ext_c = ext_comms_map[u]
            li = u[0]
            l_avg = layer_deg_sum[li] / max(1, layer_cnt[li])
            rel_deg = min(1.0, k_u / max(1e-12, 0.5 * l_avg))

            # 1. Core cohesion normalized to node's OWN (layer, community) scale so layer 0
            #    (no incoming l-1 corners) and deeper layers are judged fairly on their own scale!
            peak_in = layer_comm_peak_in.get((li, cu), 0.0) or comm_peak_in[cu]
            if c_sz >= 2 and peak_in > 1e-12:
                core_v = min(1.0, (kin / peak_in) * (kin / max(1e-12, k_u)))
            else:
                core_v = 0.0

            # 2. Inter-community bridge strength (high ONLY when genuinely stitching communities)
            if kin > 1e-12 and kout > 1e-12:
                balance = 4.0 * kin * kout / ((kin + kout) ** 2)
                multi = min(1.35, 1.0 + 0.175 * max(0, len(ext_c) - 1))
                bridge_v = min(1.0, balance * multi * rel_deg)
            elif kin <= 1e-12 and len(ext_c) >= 2 and kout > 1e-12:
                max_share = max(ext_c.values()) / kout
                spread = min(1.0, 2.0 * (1.0 - max_share))
                bridge_v = min(1.0, spread * rel_deg)
            else:
                bridge_v = 0.0

            # 3. Redundancy inside an already multi-node community
            if c_sz >= 3 and bridge_v < UNI_PHI and core_v < UNI_PHI:
                red_v = min(1.0, max(0.0, 1.0 - max(core_v, bridge_v)))
            else:
                red_v = 0.0

            # 4. Canonical role governed purely by scale-free UNI_PHI
            if bridge_v >= UNI_PHI:
                role = "bridge"
            elif core_v >= UNI_PHI:
                role = "core"
            elif c_sz >= 2:
                role = "periphery"
            else:
                role = "isolated"

            new_core[u] = core_v
            new_bridge[u] = bridge_v
            new_targets[u] = tuple(sorted(ext_c.keys(), key=lambda c_id: -ext_c[c_id]))
            new_red[u] = red_v
            new_role[u] = role

        self.community_of = new_comm
        self.community_sizes = new_sizes
        self.core_cohesion = new_core
        self.bridge_strength = new_bridge
        self.bridge_targets = new_targets
        self.redundancy = new_red
        self.role_of = new_role
        self.layer_node_counts = dict(layer_cnt)
        self.num_nodes = len(nodes)
        self.num_edges = edge_slots // 2
        self.num_communities = sum(1 for sz in new_sizes.values() if sz >= 2)
        self.modularity_q = float(mod_q)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.refreshes += 1
        self.last_refresh_ms = dt_ms
        self.total_refresh_ms += dt_ms
        self._last_node_count = self.num_nodes
        self._last_edge_count = self.num_edges
        return self.stats()

    def node_feedback(self, key: GraphNode) -> Tuple[str, float, float, float, int]:
        """Return `(role, core_cohesion, bridge_strength, redundancy, community_id)`."""
        role = self.role_of.get(key)
        if role is None:
            return "cold", 0.0, 0.0, 0.0, -1
        return (
            role,
            self.core_cohesion.get(key, 0.0),
            self.bridge_strength.get(key, 0.0),
            self.redundancy.get(key, 0.0),
            self.community_of.get(key, -1),
        )

    def active_communities(self, keys: Iterable[GraphNode]) -> set[int]:
        out: set[int] = set()
        for k in keys:
            cid = self.community_of.get(k)
            if cid is not None and self.community_sizes.get(cid, 0) >= 2:
                out.add(cid)
        return out

    def bridges_active_community(
        self,
        key: GraphNode,
        active_keys: Iterable[GraphNode],
    ) -> bool:
        """True if `key` is a verified inter-community bridge touching an active community."""
        if self.bridge_strength.get(key, 0.0) < UNI_PHI:
            return False
        act_comms = self.active_communities(active_keys)
        if not act_comms:
            return False
        own_c = self.community_of.get(key, -1)
        targets = set(self.bridge_targets.get(key, ()))
        if own_c in act_comms and bool(targets - act_comms):
            return True
        if bool(targets & act_comms) and (own_c not in act_comms or bool(targets - act_comms)):
            return True
        return False

    def export_graph_snapshot(
        self,
        adj: Optional[Dict[GraphNode, Dict[GraphNode, float]]] = None,
        *,
        max_edges: int = 4000,
    ) -> Dict[str, Any]:
        """Export a Graphify-compatible JSON snapshot of coalitions, cores, bridges, and edges."""
        src_adj = adj if adj is not None else (self._adj if self._adj else None)
        if self.mutual_steps > 0 and self.refreshes == 0:
            self.compute_current_modularity()
        nodes_out = []
        for u in sorted(self.community_of.keys()):
            nodes_out.append({
                "id": f"L{u[0]}_E{u[1]}",
                "layer": int(u[0]),
                "expert": int(u[1]),
                "community": int(self.community_of[u]),
                "role": self.role_of.get(u, "isolated"),
                "core_cohesion": round(self.core_cohesion.get(u, 0.0), 4),
                "bridge_strength": round(self.bridge_strength.get(u, 0.0), 4),
                "redundancy": round(self.redundancy.get(u, 0.0), 4),
            })
        edges_out = []
        if src_adj is not None:
            raw_edges = []
            for u, nbrs in src_adj.items():
                if u not in self.community_of:
                    continue
                for v, w in nbrs.items():
                    if v > u and v in self.community_of and w > 1e-12:
                        raw_edges.append((w, u, v))
            raw_edges.sort(key=lambda it: -it[0])
            for w, u, v in raw_edges[: max(0, int(max_edges))]:
                edges_out.append({
                    "source": f"L{u[0]}_E{u[1]}",
                    "target": f"L{v[0]}_E{v[1]}",
                    "weight": round(float(w), 5),
                    "kind": "face" if u[0] == v[0] else "corner90",
                    "inter_community": self.community_of[u] != self.community_of[v],
                })
        return {
            "summary": self.stats(),
            "nodes": nodes_out,
            "links": edges_out,
        }

    def stats(self) -> Dict[str, Any]:
        role_counts = {"bridge": 0, "core": 0, "periphery": 0, "isolated": 0}
        for r in self.role_of.values():
            if r in role_counts:
                role_counts[r] += 1
        if self.mutual_steps > 0 and self.refreshes == 0:
            self.compute_current_modularity()
            avg_ms = self.total_mutual_ms / max(1, self.mutual_steps)
            last_ms = self.last_mutual_ms
            tot_ms = self.total_mutual_ms
            ops = self.mutual_steps
            mode = "mutual_coalition_agent"
        else:
            avg_ms = self.total_refresh_ms / max(1, self.refreshes)
            last_ms = self.last_refresh_ms
            tot_ms = self.total_refresh_ms
            ops = self.refreshes
            mode = "global_refresh"
        return {
            "mode": mode,
            "nodes": self.num_nodes,
            "edges": self.num_edges,
            "communities": self.num_communities,
            "modularity_q": round(self.modularity_q, 4),
            "roles": role_counts,
            "refreshes": ops,
            "mutual_steps": self.mutual_steps,
            "last_refresh_ms": round(last_ms, 4),
            "avg_refresh_ms": round(avg_ms, 4),
            "total_refresh_ms": round(tot_ms, 3),
        }


# ---------------------------------------------------------------------------
# 3.6 AgentRingMutualFeedback — Pure Zero-Delay Mutual Function of Agents
#     (Minimum) and Generated Closed Rings Compared Back Against Minimum &
#     Continuation (NO abstract coalitions, NO global graph scans!)
# ---------------------------------------------------------------------------
class AgentRingMutualFeedback:
    """Zero-delay mutual feedback strictly between Agents (the minimum) and the real
    closed Rings generated by their use — compared back against BOTH the minimum and
    each candidate continuation.

    Counts ONLY two real physical objects:
      1. Active Agents (`(layer, eid)` forming the minimum and its continuations),
      2. Real Closed 3D Rings generated by using them:
           - Flat intra-layer rings    : `(a, b, c)` where `a-b`, `b-c`, `c-a` all exist;
           - Interlayer 90° rings      : `(p in l-1, a in l, b in l)` where `p->a`, `p->b`, `a-b` exist;
           - Interlayer 90° crossroads : `(p1 in l-1, p2 in l-1, a in l)` where `p1-p2`, `p1->a`, `p2->a` exist.

    Geometric mean `(w1 * w2 * w3)**(1/3)` guarantees zero weight for any open arc and
    strict scale-invariance for dimensionless ratios `delta_ring_energy / min_ring_energy`.
    Feedback returns ONLY to the participating agents and rings (0 external interference,
    ~0.01-0.08 ms per step).
    """

    def __init__(self, num_layers: int) -> None:
        self.num_layers: int = max(1, int(num_layers))
        self._decay: float = 1.0 - UNI_PHI
        self._layer_step: List[int] = [0] * self.num_layers
        self._layer_peak_ring: List[float] = [1e-6] * self.num_layers
        self._layer_ring_events: List[int] = [0] * self.num_layers

        # Per-agent ring state (lazy decay by `1 - UNI_PHI`)
        self._last_seen: Dict[GraphNode, int] = {}
        self.ring_energy: Dict[GraphNode, float] = defaultdict(float)
        self.flat_ring_energy: Dict[GraphNode, float] = defaultdict(float)
        self.corner90_ring_energy: Dict[GraphNode, float] = defaultdict(float)
        self.agent_ring_count: Dict[GraphNode, int] = defaultdict(int)
        # Compact set of distinct ring-partner agents per node (capped to keep O(1) memory/speed)
        self._ring_partners: Dict[GraphNode, set[GraphNode]] = defaultdict(set)
        self.intersection_score: Dict[GraphNode, float] = {}
        self.role_of: Dict[GraphNode, str] = {}

        # Live closed rings observed (compact counters)
        self.total_flat_rings: int = 0
        self.total_corner90_rings: int = 0
        self.mutual_steps: int = 0
        self.total_step_ms: float = 0.0
        self.last_step_ms: float = 0.0
        self.continuation_comparisons: int = 0
        self.continuation_new_rings: int = 0
        self.continuation_no_new_ring: int = 0

    def _decayed_ring_energy(self, key: GraphNode) -> float:
        li = key[0]
        cur_step = self._layer_step[li] if 0 <= li < self.num_layers else 0
        age = max(0, cur_step - self._last_seen.get(key, cur_step))
        factor = (self._decay ** min(age, 32)) if age > 0 else 1.0
        return self.ring_energy.get(key, 0.0) * factor

    def _add_ring_energy(
        self,
        key: GraphNode,
        w_ring: float,
        cur_step: int,
        *,
        is_corner90: bool,
        partners: Tuple[GraphNode, GraphNode],
    ) -> None:
        li = key[0]
        prev = self._decayed_ring_energy(key)
        e_new = prev + UNI_PHI * w_ring
        self.ring_energy[key] = e_new
        if is_corner90:
            self.corner90_ring_energy[key] = self.corner90_ring_energy.get(key, 0.0) * self._decay + UNI_PHI * w_ring
        else:
            self.flat_ring_energy[key] = self.flat_ring_energy.get(key, 0.0) * self._decay + UNI_PHI * w_ring
        self._last_seen[key] = cur_step
        self.agent_ring_count[key] += 1
        if 0 <= li < self.num_layers and e_new > self._layer_peak_ring[li]:
            self._layer_peak_ring[li] = e_new
        pset = self._ring_partners[key]
        if len(pset) < 16:
            pset.add(partners[0])
            pset.add(partners[1])

    def observe_executed_rings(
        self,
        layer_idx: int,
        token_routes: List[List[Tuple[int, float]]],
        prev_routes: Optional[List[List[Tuple[int, float]]]],
        face_edges: Dict[Tuple[int, int], Dict[int, float]],
        corner90_edges: Dict[Tuple[int, int], Dict[int, float]],
    ) -> int:
        """Synchronously generate closed 3D Rings from the active executed Agents (zero delay!).

        Inspects ONLY the active top agents per token (`<= 4` current, `<= 3` previous) —
        at most 34 triangle checks per token (~10-40 microseconds), zero scan of sleeping nodes.
        Updates ring energy & ring-intersection role ONLY for participating agents.
        """
        t0 = time.perf_counter()
        li = int(layer_idx)
        if 0 <= li < self.num_layers:
            self._layer_step[li] += 1
            cur_step = self._layer_step[li]
            self._layer_peak_ring[li] = max(1e-6, self._layer_peak_ring[li] * self._decay)
        else:
            cur_step = 0

        has_prev = (li > 0 and prev_routes is not None and len(prev_routes) == len(token_routes))
        touched: set[GraphNode] = set()
        rings_found = 0

        for tok_i, pairs in enumerate(token_routes):
            if not pairs:
                continue
            cur_eids = [int(e) for e, _ in pairs[:4]]
            for e in cur_eids:
                touched.add((li, e))

            # 1. Flat intra-layer closed rings `(a, b, c)`
            n_cur = len(cur_eids)
            if n_cur >= 3:
                for i in range(n_cur):
                    ea = cur_eids[i]
                    fa = face_edges.get((li, ea))
                    if not fa:
                        continue
                    for j in range(i + 1, n_cur):
                        eb = cur_eids[j]
                        w_ab = fa.get(eb, 0.0)
                        if w_ab <= 1e-12:
                            continue
                        fb = face_edges.get((li, eb))
                        if not fb:
                            continue
                        for k in range(j + 1, n_cur):
                            ec = cur_eids[k]
                            w_ac = fa.get(ec, 0.0)
                            if w_ac <= 1e-12:
                                continue
                            w_bc = fb.get(ec, 0.0)
                            if w_bc <= 1e-12:
                                continue
                            w_ring = (w_ab * w_ac * w_bc) ** (1.0 / 3.0)
                            ka, kb, kc = (li, ea), (li, eb), (li, ec)
                            self._add_ring_energy(ka, w_ring, cur_step, is_corner90=False, partners=(kb, kc))
                            self._add_ring_energy(kb, w_ring, cur_step, is_corner90=False, partners=(ka, kc))
                            self._add_ring_energy(kc, w_ring, cur_step, is_corner90=False, partners=(ka, kb))
                            self.total_flat_rings += 1
                            rings_found += 1

            # 2. Inter-layer 90° closed rings `(p in l-1, a in l, b in l)` & `(p1, p2 in l-1, a in l)`
            if has_prev:
                prev_eids = [int(e) for e, _ in prev_routes[tok_i][:3]]
                # Type 2A: 1 parent p(l-1) + 2 current agents (a, b) with face a-b
                if n_cur >= 2:
                    for ep in prev_eids:
                        c_map = corner90_edges.get((li - 1, ep))
                        if not c_map:
                            continue
                        kp = (li - 1, ep)
                        for i in range(n_cur):
                            ea = cur_eids[i]
                            w_pa = c_map.get(ea, 0.0)
                            if w_pa <= 1e-12:
                                continue
                            fa = face_edges.get((li, ea))
                            if not fa:
                                continue
                            for j in range(i + 1, n_cur):
                                eb = cur_eids[j]
                                w_pb = c_map.get(eb, 0.0)
                                if w_pb <= 1e-12:
                                    continue
                                w_ab = fa.get(eb, 0.0)
                                if w_ab <= 1e-12:
                                    continue
                                w_ring = (w_pa * w_pb * w_ab) ** (1.0 / 3.0)
                                ka, kb = (li, ea), (li, eb)
                                self._add_ring_energy(ka, w_ring, cur_step, is_corner90=True, partners=(kp, kb))
                                self._add_ring_energy(kb, w_ring, cur_step, is_corner90=True, partners=(kp, ka))
                                self.total_corner90_rings += 1
                                rings_found += 1

                # Type 2B: 2 linked parents (p1, p2 in l-1) converging on 1 agent a(l)
                n_prev = len(prev_eids)
                if n_prev >= 2:
                    for i in range(n_prev):
                        ep1 = prev_eids[i]
                        fp1 = face_edges.get((li - 1, ep1))
                        c1 = corner90_edges.get((li - 1, ep1))
                        if not fp1 or not c1:
                            continue
                        for j in range(i + 1, n_prev):
                            ep2 = prev_eids[j]
                            w_p12 = fp1.get(ep2, 0.0)
                            if w_p12 <= 1e-12:
                                continue
                            c2 = corner90_edges.get((li - 1, ep2))
                            if not c2:
                                continue
                            kp1, kp2 = (li - 1, ep1), (li - 1, ep2)
                            for ea in cur_eids:
                                w_1a = c1.get(ea, 0.0)
                                w_2a = c2.get(ea, 0.0)
                                if w_1a > 1e-12 and w_2a > 1e-12:
                                    w_ring = (w_p12 * w_1a * w_2a) ** (1.0 / 3.0)
                                    self._add_ring_energy((li, ea), w_ring, cur_step, is_corner90=True, partners=(kp1, kp2))
                                    self.total_corner90_rings += 1
                                    rings_found += 1

        if rings_found > 0 and 0 <= li < self.num_layers:
            self._layer_ring_events[li] += rings_found

        # Update local ring-intersection score & role ONLY for touched agents (Rings -> Agents)
        peak_l = self._layer_peak_ring[li] if 0 <= li < self.num_layers else 1e-6
        for key in touched:
            r_now = self._decayed_ring_energy(key)
            r_rel = min(1.0, r_now / max(1e-6, peak_l))

            # Genuine ring intersection: agent bridges BOTH flat & 90° rings OR participates in
            # multiple distinct rings whose partner agents are NOT directly connected!
            f_e = self.flat_ring_energy.get(key, 0.0)
            c_e = self.corner90_ring_energy.get(key, 0.0)
            cross_3d = (4.0 * f_e * c_e / ((f_e + c_e) ** 2)) if (f_e > 1e-12 and c_e > 1e-12) else 0.0

            # Check non-connected partner pairs among this agent's intra-layer ring partners
            same_layer_partners = [p[1] for p in self._ring_partners.get(key, ()) if p[0] == li]
            non_adj_ratio = 0.0
            n_p = len(same_layer_partners)
            if n_p >= 3:
                pairs_tot = 0
                pairs_disjoint = 0
                cap_p = same_layer_partners[:8]
                for a_i in range(len(cap_p)):
                    pa_edges = face_edges.get((li, cap_p[a_i]), {})
                    for b_i in range(a_i + 1, len(cap_p)):
                        pairs_tot += 1
                        if pa_edges.get(cap_p[b_i], 0.0) <= 1e-12:
                            pairs_disjoint += 1
                if pairs_tot > 0:
                    non_adj_ratio = pairs_disjoint / pairs_tot

            inter_score = min(1.0, max(cross_3d, non_adj_ratio))
            self.intersection_score[key] = inter_score
            # A genuine ring intersection stitches >= 2 distinct closed rings even when it fires
            # less frequently than an intra-ring hub (so do NOT gate it on high scalar r_rel!).
            if self.agent_ring_count.get(key, 0) >= 2 and r_now > 1e-12 and inter_score >= UNI_PHI:
                self.role_of[key] = "ring_intersection"
            elif r_rel >= UNI_PHI:
                self.role_of[key] = "ring_core"
            else:
                self.role_of[key] = "non_ring"

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.mutual_steps += 1
        self.last_step_ms = dt_ms
        self.total_step_ms += dt_ms
        return rings_found

    def compare_minimum_and_continuation(
        self,
        layer_idx: int,
        minimum_eids: Sequence[int],
        prev_loop_eids: Sequence[int],
        cand_eid: int,
        face_edges: Dict[Tuple[int, int], Dict[int, float]],
        corner90_edges: Dict[Tuple[int, int], Dict[int, float]],
    ) -> Tuple[int, float, int, float]:
        """Directly compare closed Rings ALREADY generated by the active Minimum (`minimum_eids`)
        against NEW closed Rings that adding `cand_eid` would generate.

        Returns `(min_ring_count, min_ring_energy, delta_ring_count, delta_ring_energy)` in ~10 us.
        """
        li = int(layer_idx)
        ce = int(cand_eid)
        min_list = [int(e) for e in minimum_eids]
        prev_list = [int(e) for e in prev_loop_eids] if li > 0 else []

        # 1. Closed rings ALREADY generated by the minimum M (and M x prev_loop)
        min_cnt = 0
        min_e = 0.0
        n_m = len(min_list)
        if n_m >= 3:
            for i in range(n_m):
                ea = min_list[i]
                fa = face_edges.get((li, ea))
                if not fa:
                    continue
                for j in range(i + 1, n_m):
                    eb = min_list[j]
                    w_ab = fa.get(eb, 0.0)
                    if w_ab <= 1e-12:
                        continue
                    fb = face_edges.get((li, eb))
                    if not fb:
                        continue
                    for k in range(j + 1, n_m):
                        ec = min_list[k]
                        w_ac = fa.get(ec, 0.0)
                        w_bc = fb.get(ec, 0.0)
                        if w_ac > 1e-12 and w_bc > 1e-12:
                            min_cnt += 1
                            min_e += (w_ab * w_ac * w_bc) ** (1.0 / 3.0)

        if prev_list:
            if n_m >= 2:
                for ep in prev_list:
                    c_map = corner90_edges.get((li - 1, ep))
                    if not c_map:
                        continue
                    for i in range(n_m):
                        ea = min_list[i]
                        w_pa = c_map.get(ea, 0.0)
                        if w_pa <= 1e-12:
                            continue
                        fa = face_edges.get((li, ea))
                        if not fa:
                            continue
                        for j in range(i + 1, n_m):
                            eb = min_list[j]
                            w_pb = c_map.get(eb, 0.0)
                            w_ab = fa.get(eb, 0.0)
                            if w_pb > 1e-12 and w_ab > 1e-12:
                                min_cnt += 1
                                min_e += (w_pa * w_pb * w_ab) ** (1.0 / 3.0)
            n_p = len(prev_list)
            if n_p >= 2 and n_m >= 1:
                for i in range(n_p):
                    ep1 = prev_list[i]
                    fp1 = face_edges.get((li - 1, ep1))
                    c1 = corner90_edges.get((li - 1, ep1))
                    if not fp1 or not c1:
                        continue
                    for j in range(i + 1, n_p):
                        ep2 = prev_list[j]
                        w_p12 = fp1.get(ep2, 0.0)
                        if w_p12 <= 1e-12:
                            continue
                        c2 = corner90_edges.get((li - 1, ep2))
                        if not c2:
                            continue
                        for ea in min_list:
                            w_1a = c1.get(ea, 0.0)
                            w_2a = c2.get(ea, 0.0)
                            if w_1a > 1e-12 and w_2a > 1e-12:
                                min_cnt += 1
                                min_e += (w_p12 * w_1a * w_2a) ** (1.0 / 3.0)

        # 2. NEW closed rings generated by adding continuation `ce` to minimum M
        delta_cnt = 0
        delta_e = 0.0
        fc = face_edges.get((li, ce))
        if fc and n_m >= 2:
            for i in range(n_m):
                ea = min_list[i]
                w_ca = fc.get(ea, 0.0)
                if w_ca <= 1e-12:
                    continue
                fa = face_edges.get((li, ea))
                if not fa:
                    continue
                for j in range(i + 1, n_m):
                    eb = min_list[j]
                    w_cb = fc.get(eb, 0.0)
                    if w_cb <= 1e-12:
                        continue
                    w_ab = fa.get(eb, 0.0)
                    if w_ab > 1e-12:
                        delta_cnt += 1
                        delta_e += (w_ca * w_cb * w_ab) ** (1.0 / 3.0)

        if prev_list:
            # 90° rings `(p, a in M, ce)`
            if fc and n_m >= 1:
                for ep in prev_list:
                    c_map = corner90_edges.get((li - 1, ep))
                    if not c_map:
                        continue
                    w_pc = c_map.get(ce, 0.0)
                    if w_pc <= 1e-12:
                        continue
                    for ea in min_list:
                        w_pa = c_map.get(ea, 0.0)
                        w_ac = fc.get(ea, 0.0)
                        if w_pa > 1e-12 and w_ac > 1e-12:
                            delta_cnt += 1
                            delta_e += (w_pc * w_pa * w_ac) ** (1.0 / 3.0)
            # 90° crossroads `(p1, p2 in prev, ce)`
            n_p = len(prev_list)
            if n_p >= 2:
                for i in range(n_p):
                    ep1 = prev_list[i]
                    fp1 = face_edges.get((li - 1, ep1))
                    c1 = corner90_edges.get((li - 1, ep1))
                    if not fp1 or not c1:
                        continue
                    w_1c = c1.get(ce, 0.0)
                    if w_1c <= 1e-12:
                        continue
                    for j in range(i + 1, n_p):
                        ep2 = prev_list[j]
                        w_p12 = fp1.get(ep2, 0.0)
                        if w_p12 <= 1e-12:
                            continue
                        c2 = corner90_edges.get((li - 1, ep2))
                        if not c2:
                            continue
                        w_2c = c2.get(ce, 0.0)
                        if w_2c > 1e-12:
                            delta_cnt += 1
                            delta_e += (w_p12 * w_1c * w_2c) ** (1.0 / 3.0)

        self.continuation_comparisons += 1
        if delta_cnt > 0:
            self.continuation_new_rings += 1
        else:
            self.continuation_no_new_ring += 1
        return min_cnt, min_e, delta_cnt, delta_e

    def agent_ring_feedback(self, key: GraphNode) -> Tuple[str, float, float, bool]:
        """Return `(role, ring_rel_in_layer, intersection_score, layer_has_rings)` in O(1)."""
        li = key[0]
        if not (0 <= li < self.num_layers):
            return "cold", 0.0, 0.0, False
        layer_has_rings = self._layer_ring_events[li] > 0
        role = self.role_of.get(key)
        if role is None:
            return "cold", 0.0, 0.0, layer_has_rings
        r_rel = min(1.0, self._decayed_ring_energy(key) / max(1e-6, self._layer_peak_ring[li]))
        inter_s = self.intersection_score.get(key, 0.0)
        return role, r_rel, inter_s, layer_has_rings

    def export_ring_snapshot(
        self,
        face_edges: Dict[Tuple[int, int], Dict[int, float]],
        corner90_edges: Dict[Tuple[int, int], Dict[int, float]],
        *,
        max_edges: int = 4000,
    ) -> Dict[str, Any]:
        """Export a clean JSON snapshot of active Agents and their Ring intersections."""
        nodes_out = []
        for key in sorted(self.role_of.keys()):
            role, r_rel, inter_s, _ = self.agent_ring_feedback(key)
            nodes_out.append({
                "id": f"L{key[0]}_E{key[1]}",
                "layer": int(key[0]),
                "expert": int(key[1]),
                "role": role,
                "ring_rel": round(r_rel, 4),
                "ring_count": int(self.agent_ring_count.get(key, 0)),
                "intersection_score": round(inter_s, 4),
            })
        raw_edges = []
        node_set = set(self.role_of.keys())
        for ka, nbrs in face_edges.items():
            if ka not in node_set:
                continue
            li, ea = ka
            for eb, w in nbrs.items():
                if eb > ea and (li, eb) in node_set and w > 1e-9:
                    raw_edges.append((w, ka, (li, eb), "face"))
        for kp, nbrs in corner90_edges.items():
            if kp not in node_set:
                continue
            li_next = kp[0] + 1
            for ec, w in nbrs.items():
                kc = (li_next, ec)
                if kc in node_set and w > 1e-9:
                    raw_edges.append((w, kp, kc, "corner90"))
        raw_edges.sort(key=lambda it: -it[0])
        links_out = [
            {
                "source": f"L{u[0]}_E{u[1]}",
                "target": f"L{v[0]}_E{v[1]}",
                "weight": round(float(w), 5),
                "kind": kind,
            }
            for w, u, v, kind in raw_edges[: max(0, int(max_edges))]
        ]
        return {"summary": self.stats(), "nodes": nodes_out, "links": links_out}

    def stats(self) -> Dict[str, Any]:
        role_counts = {"ring_intersection": 0, "ring_core": 0, "non_ring": 0}
        for r in self.role_of.values():
            if r in role_counts:
                role_counts[r] += 1
        avg_ms = self.total_step_ms / max(1, self.mutual_steps)
        return {
            "mode": "agent_ring_mutual",
            "nodes": len(self.role_of),
            "total_rings_formed": self.total_flat_rings + self.total_corner90_rings,
            "flat_rings": self.total_flat_rings,
            "corner90_rings": self.total_corner90_rings,
            "roles": role_counts,
            "mutual_steps": self.mutual_steps,
            "continuation_comparisons": self.continuation_comparisons,
            "continuation_new_rings": self.continuation_new_rings,
            "continuation_no_new_ring": self.continuation_no_new_ring,
            "last_refresh_ms": round(self.last_step_ms, 4),
            "avg_refresh_ms": round(avg_ms, 4),
            "total_refresh_ms": round(self.total_step_ms, 3),
            # Compatible fields for unified A/B summary reporting
            "communities": sum(1 for c in self._layer_ring_events if c > 0),
            "modularity_q": round(
                self.continuation_new_rings / max(1, self.continuation_comparisons), 4
            ),
        }


# ---------------------------------------------------------------------------
# 3.7 SpiralSphereMutualFeedback — 3D Spiral-to-Sphere Overlap Lenses &
#     Continuous Coil Continuation on Topic Shift (NO 2D hex grids, NO reset!)
# ---------------------------------------------------------------------------
class SpiralSphereMutualFeedback:
    """Zero-delay 3D Spiral-Sphere feedback engine derived directly from Anton's geometry:

      1. Every active agent `(layer, eid)` traces a 3D spiral of computation coils that never
         returns to the exact same point (`coil_step` monotonically increases) and may tilt
         freely in 3D space without breaking -> simplified to a living **Sphere `(c_e, r_e)`**
         in a compact, scale-free 32D orthonormal projection of layer hidden-state space.
      2. Agents activated by shared information pull symmetrically toward that information
         (shrinking pairwise distance by `(1 - UNI_PHI)` per shared coil); sphere radius `r_e`
         tracks the physical half-span (`0.5 * step_dist`) of the agent's spiral coils.
      3. Shared generated data & unified knowledge emerge STRICTLY inside genuine **Sphere
         Overlap Lenses** (`d(c_a, c_b) < r_a + r_b`, depth `(1 - d / (r_a + r_b)) * coh`).
         Non-overlapping spheres (`d >= r_a + r_b`) pass by cleanly (`"no_new_loop"`).
      4. True knowledge-unifying bridges (`"sphere_intersection"`) are ONLY agents whose sphere
         overlaps partner spheres that do NOT directly overlap each other (eliminating the
         86% over-tagging flaw of flat-vs-90° `cross_3d` heuristics).
      5. **Continuous Coil Continuation on Topic Shift**: when the topic shifts, active agents
         NEVER stop, reset, or discard prior knowledge (`centers`, `radii`, `coil_steps` are
         preserved). Each agent smoothly continues its next spiral coil toward the new
         information, naturally expanding its coil span to bridge old and new knowledge in
         the overlap lens, then re-tightening once the new topic stabilizes.
    """

    def __init__(self, num_layers: int, num_experts: int, *, proj_dim: int = 32) -> None:
        self.num_layers: int = max(1, int(num_layers))
        self.num_experts: int = max(1, int(num_experts))
        self.proj_dim: int = max(8, int(proj_dim))
        self._decay: float = 1.0 - UNI_PHI

        # Compact dense CPU state (~1.3 MiB total for 40x256 Qwen3.6-35B, 0 extra VRAM for spheres)
        self.centers: np.ndarray = np.zeros((self.num_layers, self.num_experts, self.proj_dim), dtype=np.float32)
        self.radii: np.ndarray = np.zeros((self.num_layers, self.num_experts), dtype=np.float32)
        self.coil_steps: np.ndarray = np.zeros((self.num_layers, self.num_experts), dtype=np.int32)
        self.lens_energy: np.ndarray = np.zeros((self.num_layers, self.num_experts), dtype=np.float32)
        self.last_seen: np.ndarray = np.zeros((self.num_layers, self.num_experts), dtype=np.int32)
        self.bridge_ratio: np.ndarray = np.zeros((self.num_layers, self.num_experts), dtype=np.float32)

        # Per-layer running mean for anisotropy removal (unpacks narrow LLM residual cones cleanly)
        self._layer_mean: np.ndarray = np.zeros((self.num_layers, self.proj_dim), dtype=np.float32)
        self._layer_mean_init: np.ndarray = np.zeros((self.num_layers,), dtype=bool)
        self._layer_step: np.ndarray = np.zeros((self.num_layers,), dtype=np.int32)
        self._layer_peak_lens: np.ndarray = np.full((self.num_layers,), 1e-6, dtype=np.float32)
        self._layer_lens_events: np.ndarray = np.zeros((self.num_layers,), dtype=np.int32)

        # Deterministic orthonormal projection matrices cached per `(h_dim, device_str)`
        self._proj_cache: Dict[Tuple[int, str], torch.Tensor] = {}

        # Top-8 strongest intra-layer overlap partners per active agent (for true bridge detection)
        self._overlap_partners: Dict[GraphNode, Dict[int, float]] = defaultdict(dict)
        self.role_of: Dict[GraphNode, str] = {}

        # Telemetry
        self.total_lenses_formed: int = 0
        self.total_coils_advanced: int = 0
        self.topic_shift_continuations: int = 0
        self.mutual_steps: int = 0
        self.total_step_ms: float = 0.0
        self.last_step_ms: float = 0.0
        self.continuation_comparisons: int = 0
        self.continuation_new_lenses: int = 0
        self.continuation_no_overlap: int = 0

    def _get_projection(self, h_dim: int, device: torch.device) -> torch.Tensor:
        key = (int(h_dim), str(device))
        proj = self._proj_cache.get(key)
        if proj is None:
            d = self.proj_dim
            gen = torch.Generator(device="cpu")
            gen.manual_seed(20261122 ^ (int(h_dim) * 1315423911 & 0x7FFFFFFF))
            if h_dim >= d:
                raw = torch.randn((int(h_dim), d), generator=gen, dtype=torch.float32)
                q_mat, _ = torch.linalg.qr(raw, mode="reduced")
                proj_cpu = q_mat.contiguous()
            else:
                proj_cpu = torch.zeros((int(h_dim), d), dtype=torch.float32)
                proj_cpu[:h_dim, :h_dim] = torch.eye(int(h_dim), dtype=torch.float32)
            proj = proj_cpu.to(device=device, dtype=torch.float32)
            self._proj_cache[key] = proj
        return proj

    @torch.no_grad()
    def project_states(
        self,
        layer_idx: int,
        states: torch.Tensor,
        *,
        update_mean: bool = True,
    ) -> np.ndarray:
        """Project `[M, H]` hidden/residual vectors into unit-sphere `R^{32}` with per-layer
        anisotropy centering (`z_raw - mu_l`) so semantic separation is invariant across models.
        """
        if states.ndim == 1:
            states = states.unsqueeze(0)
        h_dim = int(states.shape[-1])
        proj = self._get_projection(h_dim, states.device)
        z_np = (states.detach().to(torch.float32) @ proj).cpu().numpy().astype(np.float32, copy=False)

        li = int(layer_idx)
        if 0 <= li < self.num_layers:
            if update_mean and z_np.shape[0] > 0:
                b_mean = z_np.mean(axis=0)
                if not self._layer_mean_init[li]:
                    if z_np.shape[0] >= 4:
                        spread = float(np.linalg.norm(z_np - b_mean, axis=-1).mean())
                        raw_n = float(np.linalg.norm(z_np, axis=-1).mean()) + 1e-8
                        if spread > 1e-3 * raw_n:
                            self._layer_mean[li] = b_mean
                            self._layer_mean_init[li] = True
                else:
                    beta = 0.1 * UNI_PHI
                    self._layer_mean[li] = (1.0 - beta) * self._layer_mean[li] + beta * b_mean

            if self._layer_mean_init[li]:
                zc = z_np - self._layer_mean[li]
                nc = np.linalg.norm(zc, axis=-1, keepdims=True)
                nr = np.linalg.norm(z_np, axis=-1, keepdims=True) + 1e-8
                use_c = nc > (1e-3 * nr)
                u = np.where(use_c, zc / np.maximum(nc, 1e-8), z_np / nr)
                return u.astype(np.float32, copy=False)

        nr = np.linalg.norm(z_np, axis=-1, keepdims=True) + 1e-8
        return (z_np / nr).astype(np.float32, copy=False)

    def on_topic_shift(self, layer: Optional[int] = None) -> None:
        """Record a topic-shift transition WITHOUT resetting or stopping any spiral sphere.

        Active agents keep their current `(c_e, r_e, coil_step)` and smoothly continue their
        next spiral coil directly into the new topic's information.
        """
        self.topic_shift_continuations += 1

    def _decayed_lens_energy(self, li: int, eid: int) -> float:
        cur_step = int(self._layer_step[li])
        age = max(0, cur_step - int(self.last_seen[li, eid]))
        factor = (self._decay ** min(age, 32)) if age > 0 else 1.0
        return float(self.lens_energy[li, eid]) * factor

    @staticmethod
    def _vec_norm(v: np.ndarray) -> float:
        return math.sqrt(float(np.dot(v, v)))

    def _advance_coil(self, li: int, eid: int, u_target: np.ndarray, d_io: float = 0.0) -> None:
        """Advance agent `(li, eid)` by one continuous spiral coil toward `u_target` (NO reset!)."""
        step_n = int(self.coil_steps[li, eid])
        if step_n == 0:
            self.centers[li, eid] = u_target
            self.radii[li, eid] = float(0.5 * d_io) if d_io > 1e-6 else 0.0
            self.coil_steps[li, eid] = 1
        else:
            c_prev = self.centers[li, eid]
            diff = u_target - c_prev
            d_step = self._vec_norm(diff)
            r_coil = 0.5 * max(d_step, d_io)
            # True running mean for initial 2..4 coils (order-invariant), then steady UNI_PHI EMA
            alpha = max(UNI_PHI, 1.0 / (step_n + 1))
            self.centers[li, eid] = c_prev + alpha * diff
            r_prev = float(self.radii[li, eid])
            if r_prev <= 1e-9:
                self.radii[li, eid] = r_coil
            else:
                self.radii[li, eid] = (1.0 - alpha) * r_prev + alpha * r_coil
            self.coil_steps[li, eid] = step_n + 1
        self.total_coils_advanced += 1

    def sphere_overlap_lens(self, li: int, ea: int, eb: int) -> float:
        """Return dimensionless overlap-lens depth in [0, 1] between spheres `(li, ea)` and `(li, eb)`."""
        if (
            not (0 <= li < self.num_layers)
            or not (0 <= ea < self.num_experts)
            or not (0 <= eb < self.num_experts)
        ):
            return 0.0
        if self.coil_steps[li, ea] == 0 or self.coil_steps[li, eb] == 0:
            return 0.0
        r_sum = float(self.radii[li, ea] + self.radii[li, eb])
        if r_sum <= 1e-9:
            return 0.0
        diff = self.centers[li, ea] - self.centers[li, eb]
        d = self._vec_norm(diff)
        if d >= r_sum:
            return 0.0
        return 1.0 - (d / r_sum)

    @classmethod
    def _pair_lens(cls, ca: np.ndarray, ra: float, cb: np.ndarray, rb: float) -> float:
        r_sum = ra + rb
        if r_sum <= 1e-9:
            return 0.0
        diff = ca - cb
        d = cls._vec_norm(diff)
        if d >= r_sum:
            return 0.0
        return 1.0 - (d / r_sum)

    def _record_overlap_partner(self, key: GraphNode, partner_eid: int, weight: float) -> None:
        pmap = self._overlap_partners[key]
        prev = pmap.get(partner_eid)
        if prev is not None:
            pmap[partner_eid] = prev * self._decay + weight
        elif len(pmap) < 8:
            pmap[partner_eid] = weight
        else:
            weakest_e = min(pmap, key=pmap.get)
            if weight > pmap[weakest_e]:
                del pmap[weakest_e]
                pmap[partner_eid] = weight

    def _refresh_touched_roles(self, li: int, touched: Iterable[int]) -> None:
        """Update role & true non-overlapping bridge ratio ONLY for currently touched agents."""
        peak_l = max(1e-6, float(self._layer_peak_lens[li]))
        for eid in touched:
            key = (li, eid)
            l_now = self._decayed_lens_energy(li, eid)
            l_rel = min(1.0, l_now / peak_l)

            pmap = self._overlap_partners.get(key)
            b_ratio = 0.0
            if pmap and len(pmap) >= 2:
                plist = list(pmap.keys())
                n_p = len(plist)
                tot_pairs = 0
                disjoint_pairs = 0
                for i in range(n_p):
                    pa = plist[i]
                    ca = self.centers[li, pa]
                    ra = float(self.radii[li, pa])
                    for j in range(i + 1, n_p):
                        pb = plist[j]
                        tot_pairs += 1
                        r_sum = ra + float(self.radii[li, pb])
                        if r_sum <= 1e-9 or self._vec_norm(ca - self.centers[li, pb]) >= r_sum:
                            disjoint_pairs += 1
                if tot_pairs > 0:
                    b_ratio = disjoint_pairs / tot_pairs

            self.bridge_ratio[li, eid] = b_ratio
            # Genuine sphere_intersection: bridges >= 2 significant partner spheres that do NOT overlap!
            if l_now > 1e-9 and b_ratio >= UNI_PHI:
                self.role_of[key] = "sphere_intersection"
            elif l_rel >= UNI_PHI:
                self.role_of[key] = "sphere_core"
            else:
                self.role_of[key] = "sphere_periphery"

    def observe_spiral_step(
        self,
        layer_idx: int,
        token_routes: List[List[Tuple[int, float]]],
        u_in: Optional[np.ndarray],
        u_post: Optional[np.ndarray] = None,
    ) -> int:
        """Advance active agents' spiral spheres synchronously and score real overlap lenses."""
        t0 = time.perf_counter()
        li = int(layer_idx)
        if not (0 <= li < self.num_layers) or u_in is None or len(token_routes) == 0:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.mutual_steps += 1
            self.last_step_ms = dt_ms
            self.total_step_ms += dt_ms
            return 0

        self._layer_step[li] += 1
        cur_step = int(self._layer_step[li])
        self._layer_peak_lens[li] = max(1e-6, float(self._layer_peak_lens[li]) * self._decay)

        has_post = (u_post is not None and len(u_post) == len(u_in))
        touched: set[int] = set()
        norm_batch: List[Tuple[int, List[Tuple[int, float]], float]] = []

        # Pass 1: Continuous spiral coil advance toward current/generated information (NO reset!)
        n_vecs = int(u_in.shape[0])
        for tok_i, pairs in enumerate(token_routes):
            if not pairs or tok_i >= n_vecs:
                continue
            tot_w = sum(rw for _, rw in pairs) or 1.0
            norm_p = [(int(e), float(rw) / tot_w) for e, rw in pairs[:6] if 0 <= int(e) < self.num_experts]
            if not norm_p:
                continue
            max_p = max(p for _, p in norm_p)
            norm_batch.append((tok_i, norm_p, max_p))
            u_target = u_post[tok_i] if has_post else u_in[tok_i]
            d_io = self._vec_norm(u_post[tok_i] - u_in[tok_i]) if has_post else 0.0
            for eid, p_share in norm_p:
                if p_share >= UNI_PHI * max_p or self.coil_steps[li, eid] == 0:
                    self._advance_coil(li, eid, u_target, d_io)
                touched.add(eid)

        # Pass 2: Evaluate real Sphere Overlap Lenses strictly among active cohort
        lenses_found = 0
        for _tok_i, norm_p, max_p in norm_batch:
            top_a = norm_p[:4]
            n_a = len(top_a)
            for i in range(n_a):
                ea, pa = top_a[i]
                for j in range(i + 1, n_a):
                    eb, pb = top_a[j]
                    if ea == eb:
                        continue
                    lens_ab = self.sphere_overlap_lens(li, ea, eb)
                    if lens_ab <= 1e-9:
                        continue
                    w_lens = lens_ab * math.sqrt(pa * pb)
                    if w_lens <= 1e-9:
                        continue
                    e_a = self._decayed_lens_energy(li, ea) + UNI_PHI * w_lens
                    e_b = self._decayed_lens_energy(li, eb) + UNI_PHI * w_lens
                    self.lens_energy[li, ea] = e_a
                    self.lens_energy[li, eb] = e_b
                    self.last_seen[li, ea] = cur_step
                    self.last_seen[li, eb] = cur_step
                    if e_a > self._layer_peak_lens[li]:
                        self._layer_peak_lens[li] = e_a
                    if e_b > self._layer_peak_lens[li]:
                        self._layer_peak_lens[li] = e_b
                    if pa >= UNI_PHI * max_p and pb >= UNI_PHI * max_p:
                        self._record_overlap_partner((li, ea), eb, w_lens)
                        self._record_overlap_partner((li, eb), ea, w_lens)
                    self.total_lenses_formed += 1
                    lenses_found += 1

        if lenses_found > 0:
            self._layer_lens_events[li] += lenses_found

        if touched:
            self._refresh_touched_roles(li, touched)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.mutual_steps += 1
        self.last_step_ms = dt_ms
        self.total_step_ms += dt_ms
        return lenses_found

    def observe_post_coil(
        self,
        layer_idx: int,
        executed_routes: List[List[Tuple[int, float]]],
        u_in: Optional[np.ndarray],
        u_post: Optional[np.ndarray],
    ) -> None:
        """Complete prefill coil with real generated displacement `||u_post - u_in||` so even
        single-token experts acquire an authentic physical coil radius without arbitrary constants.
        """
        li = int(layer_idx)
        if not (0 <= li < self.num_layers) or u_in is None or u_post is None:
            return
        n_vecs = min(int(u_in.shape[0]), int(u_post.shape[0]))
        touched: set[int] = set()
        for tok_i, pairs in enumerate(executed_routes):
            if not pairs or tok_i >= n_vecs:
                continue
            d_io = self._vec_norm(u_post[tok_i] - u_in[tok_i])
            if d_io <= 1e-6:
                continue
            r_io = 0.5 * d_io
            u_p = u_post[tok_i]
            for e, _ in pairs[:4]:
                eid = int(e)
                if not (0 <= eid < self.num_experts):
                    continue
                if self.coil_steps[li, eid] == 0:
                    self._advance_coil(li, eid, u_p, d_io)
                else:
                    if float(self.radii[li, eid]) <= 1e-9:
                        self.radii[li, eid] = r_io
                    else:
                        self.radii[li, eid] = (1.0 - UNI_PHI) * float(self.radii[li, eid]) + UNI_PHI * max(float(self.radii[li, eid]), r_io)
                    self.centers[li, eid] = (1.0 - 0.5 * UNI_PHI) * self.centers[li, eid] + (0.5 * UNI_PHI) * u_p
                touched.add(eid)
        if touched:
            self._refresh_touched_roles(li, touched)

    def compare_minimum_and_continuation(
        self,
        layer_idx: int,
        minimum_eids: Sequence[int],
        prev_loop_eids: Sequence[int],
        cand_eid: int,
        *,
        u_info: Optional[np.ndarray] = None,
        prev_u_info: Optional[np.ndarray] = None,
        corner90_edges: Optional[Dict[Tuple[int, int], Dict[int, float]]] = None,
    ) -> Tuple[int, float, int, float]:
        """Compare overlap lenses ALREADY formed by the active minimum against NEW overlap lenses
        formed by continuing with `cand_eid` (including continuous coil shift toward `u_info`!).

        Returns `(min_lens_count, min_lens_energy, delta_lens_count, delta_lens_energy)`.
        """
        li = int(layer_idx)
        ce = int(cand_eid)
        if not (0 <= li < self.num_layers) or not (0 <= ce < self.num_experts):
            return 0, 0.0, 0, 0.0

        # 1. Active minimum spheres (with continuous next-coil projection toward `u_info` if shifted!)
        min_spheres: List[Tuple[int, np.ndarray, float]] = []
        for raw_e in minimum_eids:
            em = int(raw_e)
            if not (0 <= em < self.num_experts) or self.coil_steps[li, em] == 0:
                continue
            cm = self.centers[li, em]
            rm = float(self.radii[li, em])
            if u_info is not None:
                diff = u_info - cm
                d_info = self._vec_norm(diff)
                if d_info > rm:
                    # Smooth coil continuation toward new information (no reset, no stop!)
                    cm = cm + UNI_PHI * diff
                    rm = max(rm, (1.0 - UNI_PHI) * rm + UNI_PHI * (0.5 * d_info))
            min_spheres.append((em, cm, rm))

        # 2. Overlap lenses ALREADY formed inside the active minimum
        min_cnt = 0
        min_e = 0.0
        n_m = len(min_spheres)
        for i in range(n_m):
            _, ca, ra = min_spheres[i]
            for j in range(i + 1, n_m):
                _, cb, rb = min_spheres[j]
                lv = self._pair_lens(ca, ra, cb, rb)
                if lv > 1e-9:
                    min_cnt += 1
                    min_e += lv

        # 3. NEW overlap lenses generated by adding candidate `ce`
        delta_cnt = 0
        delta_e = 0.0
        if self.coil_steps[li, ce] > 0:
            cc = self.centers[li, ce]
            rc = float(self.radii[li, ce])
            if rc > 1e-9:
                for _, cm, rm in min_spheres:
                    lv = self._pair_lens(cm, rm, cc, rc)
                    if lv > 1e-9:
                        delta_cnt += 1
                        delta_e += lv

                # Inter-layer 3D continuity: if `cand` overlaps incoming `u_info` and connects via
                # 90° corner to an active sphere in layer `li - 1` covering `prev_u_info`.
                if (
                    li > 0
                    and corner90_edges is not None
                    and u_info is not None
                    and prev_u_info is not None
                    and prev_loop_eids
                ):
                    d_c_info = self._vec_norm(cc - u_info)
                    if d_c_info < rc:
                        c_cov = 1.0 - (d_c_info / rc)
                        for raw_ep in prev_loop_eids:
                            ep = int(raw_ep)
                            if not (0 <= ep < self.num_experts) or self.coil_steps[li - 1, ep] == 0:
                                continue
                            c_map = corner90_edges.get((li - 1, ep))
                            if not c_map or c_map.get(ce, 0.0) <= 1e-12:
                                continue
                            rp = float(self.radii[li - 1, ep])
                            if rp <= 1e-9:
                                continue
                            d_p_info = self._vec_norm(self.centers[li - 1, ep] - prev_u_info)
                            if d_p_info < rp:
                                p_cov = 1.0 - (d_p_info / rp)
                                delta_cnt += 1
                                delta_e += c_cov * p_cov

        self.continuation_comparisons += 1
        if delta_cnt > 0:
            self.continuation_new_lenses += 1
        else:
            self.continuation_no_overlap += 1
        return min_cnt, min_e, delta_cnt, delta_e

    def agent_sphere_feedback(self, key: GraphNode) -> Tuple[str, float, float, bool]:
        """Return `(role, lens_rel_in_layer, bridge_overlap_ratio, layer_has_lenses)` in O(1)."""
        li, eid = int(key[0]), int(key[1])
        if not (0 <= li < self.num_layers) or not (0 <= eid < self.num_experts):
            return "cold", 0.0, 0.0, False
        layer_has_lenses = int(self._layer_lens_events[li]) > 0
        role = self.role_of.get((li, eid))
        if role is None:
            return "cold", 0.0, 0.0, layer_has_lenses
        l_rel = min(1.0, self._decayed_lens_energy(li, eid) / max(1e-6, float(self._layer_peak_lens[li])))
        b_rat = float(self.bridge_ratio[li, eid])
        return role, l_rel, b_rat, layer_has_lenses

    def export_sphere_snapshot(
        self,
        face_edges: Dict[Tuple[int, int], Dict[int, float]],
        corner90_edges: Dict[Tuple[int, int], Dict[int, float]],
        *,
        max_edges: int = 4000,
    ) -> Dict[str, Any]:
        """Export a clean JSON snapshot of active Spiral Spheres and their overlap lenses."""
        nodes_out = []
        for key in sorted(self.role_of.keys()):
            li, eid = key
            role, l_rel, b_rat, _ = self.agent_sphere_feedback(key)
            nodes_out.append({
                "id": f"L{li}_E{eid}",
                "layer": int(li),
                "expert": int(eid),
                "role": role,
                "coil_steps": int(self.coil_steps[li, eid]),
                "radius": round(float(self.radii[li, eid]), 4),
                "center_coherence": round(float(np.linalg.norm(self.centers[li, eid])), 4),
                "lens_rel": round(l_rel, 4),
                "bridge_overlap_ratio": round(b_rat, 4),
                "pos3": [round(float(v), 4) for v in self.centers[li, eid, :3]],
            })
        raw_edges = []
        node_set = set(self.role_of.keys())
        for ka, nbrs in face_edges.items():
            if ka not in node_set:
                continue
            li, ea = ka
            for eb, w in nbrs.items():
                if eb > ea and (li, eb) in node_set and w > 1e-9:
                    lens_v = self.sphere_overlap_lens(li, ea, eb)
                    if lens_v > 1e-6:
                        raw_edges.append((lens_v * w, ka, (li, eb), "sphere_lens"))
        for kp, nbrs in corner90_edges.items():
            if kp not in node_set:
                continue
            li_next = kp[0] + 1
            for ec, w in nbrs.items():
                kc = (li_next, ec)
                if kc in node_set and w > 1e-9:
                    raw_edges.append((w, kp, kc, "corner90"))
        raw_edges.sort(key=lambda it: -it[0])
        links_out = [
            {
                "source": f"L{u[0]}_E{u[1]}",
                "target": f"L{v[0]}_E{v[1]}",
                "weight": round(float(w), 5),
                "kind": kind,
            }
            for w, u, v, kind in raw_edges[: max(0, int(max_edges))]
        ]
        return {"summary": self.stats(), "nodes": nodes_out, "links": links_out}

    def stats(self) -> Dict[str, Any]:
        role_counts = {"sphere_intersection": 0, "sphere_core": 0, "sphere_periphery": 0}
        for r in self.role_of.values():
            if r in role_counts:
                role_counts[r] += 1
        avg_ms = self.total_step_ms / max(1, self.mutual_steps)
        active_mask = self.coil_steps > 0
        mean_r = float(self.radii[active_mask].mean()) if bool(active_mask.any()) else 0.0
        max_coil = int(self.coil_steps.max()) if self.coil_steps.size > 0 else 0
        return {
            "mode": "spiral_sphere_mutual",
            "nodes": len(self.role_of),
            "total_lenses_formed": self.total_lenses_formed,
            "total_coils_advanced": self.total_coils_advanced,
            "max_coil_step": max_coil,
            "mean_sphere_radius": round(mean_r, 4),
            "topic_shift_continuations": self.topic_shift_continuations,
            "roles": role_counts,
            "mutual_steps": self.mutual_steps,
            "continuation_comparisons": self.continuation_comparisons,
            "continuation_new_lenses": self.continuation_new_lenses,
            "continuation_no_overlap": self.continuation_no_overlap,
            "last_refresh_ms": round(self.last_step_ms, 4),
            "avg_refresh_ms": round(avg_ms, 4),
            "total_refresh_ms": round(self.total_step_ms, 3),
            # Compatible fields for unified A/B summary reporting
            "communities": int((self._layer_lens_events > 0).sum()),
            "modularity_q": round(
                self.continuation_new_lenses / max(1, self.continuation_comparisons), 4
            ),
        }


def normalize_feedback_mode(value: Any) -> str:
    """Normalize feedback selector into `'off'` (V1), `'spiral_sphere'` (V2), `'pure_mirror'` (V3), or `'mirror_scales'` (V4)."""
    if value is None or value is False:
        return "off"
    if value is True:
        return "spiral_sphere"
    s = str(value).strip().lower()
    if s in ("", "0", "false", "off", "none", "v1", "loop_local"):
        return "off"
    if s in ("mirror_scales", "v4", "mirror_v4", "isolated_mirror"):
        return "mirror_scales"
    if s in ("pure_mirror", "pure_mirror_cuda", "mirror", "v3"):
        return "pure_mirror"
    return "spiral_sphere"


# ---------------------------------------------------------------------------
# 4. Method 1 -> Method 2 Bridge & Unified Silicon Registry (IC-04 & IC-06)
# ---------------------------------------------------------------------------
def load_cube_blueprint(
    blueprint_source: Any,
    *,
    expected_root_bits: int = 16,
) -> Tuple[Dict[int, Dict[str, int]], Dict[int, Dict[int, Dict[str, int]]], int]:
    """Canonical loader bridging Method 1 (`ColibriBlueprint` / `StreamingCubeSurvey`) -> Method 2.

    Accepts a file path (`str`/`Path`), a `ColibriBlueprint` instance, or a raw `dict`.
    Returns `(layer_proj_cubes, layer_expert_cubes, max_depth)`.
    """
    if blueprint_source is None:
        return {}, {}, 0

    if isinstance(blueprint_source, (str, Path)):
        raw_obj: Any = json.loads(Path(blueprint_source).read_text(encoding="utf-8-sig"))
    else:
        raw_obj = blueprint_source

    if isinstance(raw_obj, dict) and "root_bits" in raw_obj:
        if int(raw_obj["root_bits"]) != int(expected_root_bits):
            raise ValueError("Blueprint and executor must use the same complete tree")

    raw_cube_map = getattr(raw_obj, "layer_proj_cubes", raw_obj) or {}
    raw_expert_bp = raw_cube_map.get("layer_expert_cubes", {}) if isinstance(raw_cube_map, dict) else {}
    if not raw_expert_bp and hasattr(raw_obj, "layer_expert_cubes"):
        raw_expert_bp = getattr(raw_obj, "layer_expert_cubes") or {}

    expert_bp: Dict[int, Dict[int, Dict[str, int]]] = {
        int(li): {
            int(eid): {str(sn): int(k) for sn, k in ds.items()}
            for eid, ds in es.items()
        }
        for li, es in (raw_expert_bp.items() if isinstance(raw_expert_bp, dict) else ())
    }
    expert_depths = [
        k
        for es in expert_bp.values()
        for ds in es.values()
        for k in ds.values()
    ]
    if any(not 1 <= k <= int(expected_root_bits) for k in expert_depths):
        raise ValueError("Expert blueprint depths must fit the complete tree")

    if isinstance(raw_cube_map, dict) and "layer_proj_cubes" in raw_cube_map:
        raw_cube_map = raw_cube_map["layer_proj_cubes"]

    proj_bp: Dict[int, Dict[str, int]] = {
        int(li): {str(sn): int(k) for sn, k in projs.items()}
        for li, projs in (raw_cube_map.items() if isinstance(raw_cube_map, dict) else ())
    }
    proj_depths = [k for projs in proj_bp.values() for k in projs.values()]
    if any(not 1 <= k <= int(expected_root_bits) for k in proj_depths):
        raise ValueError("Blueprint cube depths must fit the complete tree")

    max_depth = max([0, *proj_depths, *expert_depths])
    return proj_bp, expert_bp, max_depth


def get_method1_survey_tools() -> Dict[str, Any]:
    """Official bridge to Method 1 cartographer tools (`ColibriBlueprint` & `StreamingCubeSurvey`)."""
    from src.colibri_cube_blueprint import ColibriBlueprint, ColibriCubeComparator
    from src.streaming_cube_survey import StreamingCubeSurvey, projection_prefix_errors

    return {
        "ColibriBlueprint": ColibriBlueprint,
        "ColibriCubeComparator": ColibriCubeComparator,
        "StreamingCubeSurvey": StreamingCubeSurvey,
        "projection_prefix_errors": projection_prefix_errors,
    }


def get_unified_silicon_registry() -> Dict[str, Callable[..., Any]]:
    """Unified capability facade over all NVRTC Q1 / Mirror / Micro-Agent / 3D Fabric kernels."""
    from src.colibri_q1_cube_nvrtc import (
        launch_ideal_q1_b1_mma,
        launch_ideal_q1_fwht_gemv,
        launch_q1_multiplane_gemv,
    )
    from src.fractal_cube_fabric_nvrtc import Fractal3DCubeSwiGLUMLP, Projection3DCubeStage
    from src.mirror_equality_nvrtc import (
        launch_mirror_equality_q1_gemv,
        launch_mirror_fractal_ring_q1_step,
        launch_mirror_uni_binding_q1_gemv,
        run_autonomous_fractal_rings,
    )
    from src.mirror_micro_agents_nvrtc import (
        launch_coupled_micro_agent_q1_reader,
        launch_micro_agent_q1_group64_reader,
        run_million_micro_agent_substrate,
    )
    from src.pure_mirror_agents_nvrtc import (
        PureMirrorCudaFeedback,
        launch_pure_mirror_line_step,
    )
    from src.qwen_micro_fabric import launch_q1_moe_fabric

    return {
        "q1_multiplane_gemv": launch_q1_multiplane_gemv,
        "ideal_q1_fwht_gemv": launch_ideal_q1_fwht_gemv,
        "ideal_q1_b1_mma": launch_ideal_q1_b1_mma,
        "mirror_equality_q1_gemv": launch_mirror_equality_q1_gemv,
        "mirror_uni_binding_q1_gemv": launch_mirror_uni_binding_q1_gemv,
        "mirror_fractal_ring_step": launch_mirror_fractal_ring_q1_step,
        "autonomous_fractal_rings": run_autonomous_fractal_rings,
        "pure_mirror_line_step": launch_pure_mirror_line_step,
        "pure_mirror_cuda_feedback": PureMirrorCudaFeedback,
        "micro_agent_group64_reader": launch_micro_agent_q1_group64_reader,
        "coupled_micro_agent_reader": launch_coupled_micro_agent_q1_reader,
        "million_micro_agent_substrate": run_million_micro_agent_substrate,
        "fractal_3d_cube_stage": Projection3DCubeStage,
        "fractal_3d_swiglu_mlp": Fractal3DCubeSwiGLUMLP,
        "qwen_moe_fabric": launch_q1_moe_fabric,
    }
