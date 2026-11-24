"""Probability Imprint Puzzle Overlay (`ProbabilityImprintGraph` + `ImprintPuzzleOverlay`).

Instead of fetching one isolated expert/cube at a time from disk and letting LRU thrash
across 40 layers, builds a connected **Probability Imprint Puzzle** overlaid on top of the
read-only original model:

  1. Probability Imprint Graph across ALL layers:
       - Vertex mass      : cumulative router probability mass + anchored prompt support
       - Intra-layer faces: HORIZ/VERT/DIAG co-occurrence on the shared `line_spine`
       - Inter-layer 90°  : CORNER90 transitions `(layer l-1, e_prev) -> (layer l, e_cur)`
     Each expert's `imprint_score` determines its multi-resolution Q1 cube depth in the
     overlaid dismantled model (`core=5, mid=4, rim=3, shell=2` planes instead of bloating
     every unseen expert to 16 planes!).

  2. Depth-Shedding (`k -> k-1` in VRAM) instead of destructive LRU eviction:
     Because positional binary-halving Q1 trees share a single `qscale[0]`, trimming trailing
     bit-planes (`packed_lo[:new_k]`) on cooled peripheral experts takes ~0.01 ms in VRAM and
     frees space for new boundary puzzle pieces WITHOUT evicting the expert or triggering a
     50 ms HDD re-read on the next token!

  3. Strict `A -> B -> C(0) -> A` Relay (`Q1RoleCycleArena`, `B->A == 0`, `standby == 0 B`):
     - On prefill (`M > 1`), area `B` reads and carves coalesced contiguous chunks of puzzle
       experts (up to 8 per disk pass) on a dedicated CUDA stream and hands them zero-copy to
       `old_C(0) -> new_A`.
     - On decode (`M == 1`), active area `A` executes directly over the pinned multi-resolution
       imprint already resident in VRAM; `Mirror Equality x UNI` network rest on `line_spine`
       keeps cold exterior tails asleep on disk; only genuine boundary crossings wake `B` to
       pull the next connected puzzle piece through `A -> B -> C(0) -> A`.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.pure_mirror_agents_nvrtc import (
    EQ_SPLIT,
    PureMirrorCudaFeedback,
    launch_pure_mirror_diff_q1_gemv,
    launch_pure_mirror_line_step,
)
from src.q1_cube_engine import Q1RoleCycleArena
from src.stanok_contour import (
    MirrorUniStabilizer,
    SpiralSphereMutualFeedback,
    StanokContainmentContour,
    UNI_PHI,
    normalize_feedback_mode,
)


def _tensor_dict_bytes(d: Dict[str, torch.Tensor]) -> int:
    return sum(int(t.numel()) * int(t.element_size()) for t in d.values() if isinstance(t, torch.Tensor))


def _trim_q1_planes(t: Dict[str, torch.Tensor], new_k: int) -> Dict[str, torch.Tensor]:
    """Shrink a Q1 tensor dict in-place-safe fashion from P planes down to `new_k` planes.

    Valid because `positional_halving` stores shared `qscale[0]` (shape `[1, N, NG]`) and
    `packed_lo` ordered MSB-first `[P, N, K // 8]`.
    """
    cur_k = int(t["packed_lo"].shape[0])
    if new_k >= cur_k:
        return t
    out = dict(t)
    out["packed_lo"] = t["packed_lo"][:new_k].contiguous().clone()
    if int(t["qscale"].shape[0]) > 1:
        out["qscale"] = t["qscale"][:new_k].contiguous().clone()
    return out


class ProbabilityImprintGraph:
    """Decentralized, scale-free 3D lattice of probability intersections (vertices + faces + 90° corners).

    Uses dimensionless local EMA energy & lattice coupling damped by canonical `UNI_PHI = exp(-1.5)`
    so depth, lattice backbone retention, and natural self-collapse are invariant across model sizes,
    expert counts, top-K routing widths, and conversation lengths (ZERO model-specific cutoffs).
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        *,
        core_k: int = 5,
        mid_k: int = 4,
        rim_k: int = 3,
        shell_k: int = 2,
        min_shed_k: int = 2,
        graphify_feedback: Any = False,
        device: Any = None,
    ) -> None:
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.core_k = int(core_k)
        self.mid_k = int(mid_k)
        self.rim_k = int(rim_k)
        self.shell_k = int(shell_k)
        self.min_shed_k = max(1, min(int(min_shed_k), self.shell_k))
        self.feedback_mode: str = normalize_feedback_mode(graphify_feedback)
        self.graphify_feedback: bool = (self.feedback_mode != "off")
        self.sphere_feedback: Optional[SpiralSphereMutualFeedback] = (
            SpiralSphereMutualFeedback(self.num_layers, self.num_experts)
            if self.feedback_mode == "spiral_sphere"
            else None
        )
        self.pure_mirror_feedback: Optional[PureMirrorCudaFeedback] = (
            PureMirrorCudaFeedback(self.num_layers, self.num_experts, device=device)
            if self.feedback_mode == "pure_mirror"
            else None
        )
        # Backward-compatible attribute aliases pointing to the active feedback organ
        self.topology: Any = self.pure_mirror_feedback if self.pure_mirror_feedback is not None else self.sphere_feedback
        self.ring_feedback: Any = self.topology
        self._last_layer_u_in: Optional[Any] = None

        # Telemetry / cumulative counters
        self.mass: Dict[Tuple[int, int], float] = defaultdict(float)
        self.anchor_mass: Dict[Tuple[int, int], float] = defaultdict(float)
        self.hit_count: Dict[Tuple[int, int], int] = defaultdict(int)
        self.face_support: Dict[Tuple[int, int], float] = defaultdict(float)
        # Actual pairwise intra-layer face intersections: (l, ea) -> {eb: weight}
        self.face_edges: Dict[Tuple[int, int], Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.corner90_support: Dict[Tuple[int, int], float] = defaultdict(float)
        self.corner90_edges: Dict[Tuple[int, int], Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        # Scale-free dimensionless EMA local lattice state:
        #   - V3 Pure Mirror: pure 1/2 mirror halving (ZERO UNI_PHI!)
        #   - V1/V2 Lattice : canonical 1 - UNI_PHI
        self._decay: float = 0.5 if self.pure_mirror_feedback is not None else (1.0 - UNI_PHI)
        self._layer_step: List[int] = [0] * max(1, self.num_layers)
        self._last_seen: Dict[Tuple[int, int], int] = {}
        self.local_energy: Dict[Tuple[int, int], float] = defaultdict(float)
        self.lattice_coupling: Dict[Tuple[int, int], float] = defaultdict(float)
        self._layer_peak_energy: List[float] = [1e-6] * max(1, self.num_layers)
        self._layer_peak_coupling: List[float] = [1e-6] * max(1, self.num_layers)

        # Transient per-step last layer routing for building real l-1 -> l CORNER90 edges
        self._last_layer_idx: int = -1
        self._last_layer_routes: List[List[Tuple[int, float]]] = []
        self.total_face_links: int = 0
        self.total_corner90_links: int = 0

        # Online Graphify internal feedback telemetry
        self.graphify_bridges_protected: int = 0
        self.graphify_cores_reinforced: int = 0
        self.graphify_noise_damped: int = 0
        self.graphify_bridge_loop_wakes: int = 0

    def _node_liveness(self, key: Tuple[int, int]) -> float:
        li = key[0]
        cur_step = self._layer_step[li] if 0 <= li < len(self._layer_step) else 0
        age = max(0, cur_step - self._last_seen.get(key, cur_step))
        return (self._decay ** min(age, 32)) if age > 0 else 1.0

    def _decayed_pair(self, key: Tuple[int, int]) -> Tuple[float, float]:
        factor = self._node_liveness(key)
        return self.local_energy.get(key, 0.0) * factor, self.lattice_coupling.get(key, 0.0) * factor

    def build_live_adjacency(self) -> Dict[Tuple[int, int], Dict[Tuple[int, int], float]]:
        """Build symmetric, liveness-weighted, backbone-sparsified adjacency over face & 90° links.

        Drops ephemeral micro-noise edges that sit below `UNI_PHI * row_peak` on BOTH endpoints,
        while preserving every genuine core link and every inter-community bridge link.
        """
        adj: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = defaultdict(dict)
        live_cache: Dict[Tuple[int, int], float] = {}

        def _live(k: Tuple[int, int]) -> float:
            v = live_cache.get(k)
            if v is None:
                v = self._node_liveness(k)
                live_cache[k] = v
            return v

        # Precompute per-node peak face weight for scale-free UNI_PHI backbone sparsification
        face_peak: Dict[Tuple[int, int], float] = {
            ka: max(nbrs.values(), default=0.0)
            for ka, nbrs in self.face_edges.items()
            if nbrs
        }

        # Intra-layer face edges (stored symmetrically; iterate half via ea < eb)
        for ka, nbrs in self.face_edges.items():
            li, ea = ka
            la = _live(ka)
            cut_a = face_peak.get(ka, 0.0) * UNI_PHI
            for eb, w in nbrs.items():
                if eb <= ea or w <= 1e-9:
                    continue
                kb = (li, eb)
                if w < cut_a and w < face_peak.get(kb, 0.0) * UNI_PHI:
                    continue
                w_eff = w * math.sqrt(la * _live(kb))
                if w_eff > 1e-6:
                    adj[ka][kb] = adj[ka].get(kb, 0.0) + w_eff
                    adj[kb][ka] = adj[kb].get(ka, 0.0) + w_eff

        # Inter-layer 90° corner edges (layer l-1 -> layer l)
        c90_in_peak: Dict[Tuple[int, int], float] = defaultdict(float)
        for kp, nbrs in self.corner90_edges.items():
            li_next = kp[0] + 1
            for ec, w in nbrs.items():
                kc = (li_next, ec)
                if w > c90_in_peak[kc]:
                    c90_in_peak[kc] = w

        for kp, nbrs in self.corner90_edges.items():
            if not nbrs:
                continue
            li_prev = kp[0]
            li_next = li_prev + 1
            lp = _live(kp)
            cut_p = max(nbrs.values()) * UNI_PHI
            for ec, w in nbrs.items():
                if w <= 1e-9:
                    continue
                kc = (li_next, ec)
                if w < cut_p and w < c90_in_peak.get(kc, 0.0) * UNI_PHI:
                    continue
                w_eff = w * math.sqrt(lp * _live(kc))
                if w_eff > 1e-6:
                    adj[kp][kc] = adj[kp].get(kc, 0.0) + w_eff
                    adj[kc][kp] = adj[kc].get(kp, 0.0) + w_eff

        return adj

    def refresh_topology(
        self,
        *,
        cold_start: bool = False,
        fast_warm: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """No-op global scan: active V2 Spiral-Sphere or V3 Pure Mirror feedback runs synchronously."""
        if self.topology is None:
            return None
        return self.topology.stats()

    def export_graphify_snapshot(self, *, max_edges: int = 4000) -> Optional[Dict[str, Any]]:
        if self.pure_mirror_feedback is not None:
            return self.pure_mirror_feedback.export_mirror_snapshot(
                self.face_edges, self.corner90_edges, max_edges=max_edges
            )
        if self.sphere_feedback is not None:
            return self.sphere_feedback.export_sphere_snapshot(
                self.face_edges, self.corner90_edges, max_edges=max_edges
            )
        return None

    def observe_routing(
        self,
        layer_idx: int,
        token_routes: List[List[Tuple[int, float]]],
        *,
        is_prefill: bool = False,
        u_in: Optional[Any] = None,
        u_post: Optional[Any] = None,
    ) -> None:
        """Update dimensionless local vertex energy, intra-layer faces, and 90° interlayer corners.

        When `graphify_feedback=True`, synchronously advances each active agent's 3D spiral sphere
        and evaluates real Sphere Overlap Lenses via `SpiralSphereMutualFeedback` (zero delay,
        continuous coil progression with NO topic-shift reset).
        """
        li = int(layer_idx)
        if 0 <= li < len(self._layer_step):
            self._layer_step[li] += 1
            cur_step = self._layer_step[li]
            self._layer_peak_energy[li] = max(1e-6, self._layer_peak_energy[li] * self._decay)
            self._layer_peak_coupling[li] = max(1e-6, self._layer_peak_coupling[li] * self._decay)
        else:
            cur_step = 0

        has_prev = (
            self._last_layer_idx == li - 1
            and len(self._last_layer_routes) == len(token_routes)
        )
        prev_routes_ref = self._last_layer_routes if has_prev else None
        prev_norm_routes: List[List[Tuple[int, float]]] = []
        if has_prev:
            for p_pairs in self._last_layer_routes:
                ps = sum(w for _, w in p_pairs) or 1.0
                prev_norm_routes.append([(e, w / ps) for e, w in p_pairs[:3]])

        step_gain = 0.5 if self.pure_mirror_feedback is not None else UNI_PHI
        cur_norm_routes: List[List[Tuple[int, float]]] = []
        for tok_i, pairs in enumerate(token_routes):
            if not pairs:
                cur_norm_routes.append([])
                continue
            tot_w = sum(rw for _, rw in pairs) or 1.0
            norm_pairs = [(eid, rw / tot_w) for eid, rw in pairs]
            cur_norm_routes.append(norm_pairs)

            # 1. Dimensionless local energy + telemetry mass
            for (eid, rw), (_, pn) in zip(pairs, norm_pairs):
                key = (li, eid)
                e_old, c_old = self._decayed_pair(key)
                e_new = e_old + step_gain * pn
                self.local_energy[key] = e_new
                self.lattice_coupling[key] = c_old
                self._last_seen[key] = cur_step
                if 0 <= li < len(self._layer_peak_energy) and e_new > self._layer_peak_energy[li]:
                    self._layer_peak_energy[li] = e_new

                self.mass[key] += rw
                self.hit_count[key] += 1
                if is_prefill:
                    self.anchor_mass[key] += rw

            # 2. Intra-layer faces (co-occurring intersections sharing this token's line_spine)
            top_f = norm_pairs[:4]
            for a_i in range(len(top_f)):
                ea, pa = top_f[a_i]
                for b_i in range(a_i + 1, len(top_f)):
                    eb, pb = top_f[b_i]
                    w_face = pa * pb
                    ka, kb = (li, ea), (li, eb)
                    self.face_edges[ka][eb] += w_face
                    self.face_edges[kb][ea] += w_face
                    self.face_support[ka] += w_face
                    self.face_support[kb] += w_face
                    self.lattice_coupling[ka] += step_gain * w_face
                    self.lattice_coupling[kb] += step_gain * w_face
                    if 0 <= li < len(self._layer_peak_coupling):
                        self._layer_peak_coupling[li] = max(
                            self._layer_peak_coupling[li],
                            self.lattice_coupling[ka],
                            self.lattice_coupling[kb],
                        )
                    self.total_face_links += 1

            # 3. Inter-layer 90° corner links (layer l-1 -> layer l)
            if has_prev and tok_i < len(prev_norm_routes):
                prev_top = prev_norm_routes[tok_i]
                cur_top = norm_pairs[:3]
                for ep, pp in prev_top:
                    edge_map = self.corner90_edges[(li - 1, ep)]
                    for ec, pc in cur_top:
                        w_c90 = pp * pc
                        edge_map[ec] += w_c90
                        kc = (li, ec)
                        kp = (li - 1, ep)
                        self.corner90_support[kc] += w_c90
                        self.corner90_support[kp] += w_c90 * 0.5
                        self.lattice_coupling[kc] += step_gain * w_c90
                        if 0 <= li < len(self._layer_peak_coupling):
                            if self.lattice_coupling[kc] > self._layer_peak_coupling[li]:
                                self._layer_peak_coupling[li] = self.lattice_coupling[kc]
                        self.total_corner90_links += 1

        self._last_layer_idx = li
        self._last_layer_routes = token_routes
        self._last_layer_u_in = u_in

        # Zero-delay feedback strictly on active cohort (V3 Irreducible 4x4 Pure Mirror CUDA or V2 Spiral-Sphere)
        if self.pure_mirror_feedback is not None and u_in is not None:
            if is_prefill:
                # Candidate pre-execution binding: keep past P0/P1 immutable (`is_commit=False`)
                self.pure_mirror_feedback.observe_mirror_step(
                    li, token_routes, u_in, u_post=None, is_commit=False
                )
            else:
                # Decode CompletedPass commit: materialize real `(Observed=u_in, Outcome=u_post)` & refine Logic
                self.pure_mirror_feedback.observe_mirror_step(
                    li, token_routes, u_in, u_post=u_post, is_commit=True
                )
        elif self.sphere_feedback is not None and u_in is not None:
            self.sphere_feedback.observe_spiral_step(li, token_routes, u_in, u_post)

    def observe_post_coil(
        self,
        layer_idx: int,
        executed_routes: List[List[Tuple[int, float]]],
        u_in: Optional[Any],
        u_post: Optional[Any],
    ) -> None:
        """Complete prefill step with generated residual state `u_post` (`CompletedPass` commit)."""
        if self.pure_mirror_feedback is not None and u_in is not None and u_post is not None:
            self.pure_mirror_feedback.observe_mirror_step(
                layer_idx, executed_routes, u_in, u_post=u_post, is_commit=True
            )
        elif self.sphere_feedback is not None and u_in is not None and u_post is not None:
            self.sphere_feedback.observe_post_coil(layer_idx, executed_routes, u_in, u_post)

    def compare_minimum_rings_with_continuation(
        self,
        layer_idx: int,
        active_minimum_eids: Sequence[int],
        prev_layer_loop_eids: Sequence[int],
        cand_eid: int,
        *,
        u_info: Optional[Any] = None,
        prev_u_info: Optional[Any] = None,
    ) -> Tuple[int, float, int, float]:
        """Compare overlap lenses formed by the active minimum against NEW lenses from `cand_eid`."""
        if self.sphere_feedback is None:
            return 0, 0.0, 0, 0.0
        return self.sphere_feedback.compare_minimum_and_continuation(
            layer_idx,
            active_minimum_eids,
            prev_layer_loop_eids,
            cand_eid,
            u_info=u_info,
            prev_u_info=prev_u_info,
            corner90_edges=self.corner90_edges,
        )

    def closes_active_loop(
        self,
        layer_idx: int,
        cand_eid: int,
        active_resident_eids: Sequence[int],
        prev_layer_loop_eids: Sequence[int],
        *,
        prefer_bridges: bool = False,
    ) -> bool:
        """Topological loop-closure test: can connecting `cand_eid` close a real loop/intersection?

        Requires at least TWO established contacts with the live active loop — either
        intra-layer face intersections with already-executed nodes on `line_spine` and/or
        incoming 90° corners from layer `layer_idx - 1`'s active executed loop.
        A dangling 0- or 1-contact cold agent cannot close another loop or intersection.
        """
        li = int(layer_idx)
        ce = int(cand_eid)
        contacts = 0
        f_edges = self.face_edges.get((li, ce))
        if f_edges:
            for ea in active_resident_eids:
                if f_edges.get(int(ea), 0.0) > 0.0:
                    contacts += 1
                    if contacts >= 2:
                        return True
        if li > 0 and prev_layer_loop_eids:
            for ep in prev_layer_loop_eids:
                c_edges = self.corner90_edges.get((li - 1, int(ep)))
                if c_edges and c_edges.get(ce, 0.0) > 0.0:
                    contacts += 1
                    if contacts >= 2:
                        return True
        return False

    def relative_binding(self, key: Tuple[int, int]) -> float:
        """Dimensionless local lattice binding in [0, 1], normalized to its own layer's live scale."""
        li = key[0]
        if not (0 <= li < len(self._layer_peak_energy)):
            return 0.0
        e_now, c_now = self._decayed_pair(key)
        e_rel = min(1.0, e_now / max(1e-6, self._layer_peak_energy[li]))
        c_rel = min(1.0, c_now / max(1e-6, self._layer_peak_coupling[li]))
        return min(1.0, 0.5 * e_rel + 0.5 * c_rel)

    def score(self, key: Tuple[int, int]) -> float:
        """Dimensionless live lattice binding strength (pure local physics; VRAM contour untouched)."""
        return self.relative_binding(key)

    def target_depth(
        self,
        key: Tuple[int, int],
        *,
        batch_weight_sum: float = 0.0,
        batch_token_count: int = 1,
        batch_max_weight: float = 0.0,
        blueprint_depth: Optional[int] = None,
        root_bits: int = 16,
    ) -> int:
        """Derive Q1 cube depth purely from dimensionless local lattice binding [0, 1].

        Normalizes current-step participation against the layer's own strongest active weight
        (`mean_w / batch_max_weight`) so even on the very first visit to a cold layer, depth
        assignment is 100% invariant across top-2, top-4, top-8, and top-16 MoE architectures.

        When `graphify_feedback=True`, internal Graphify roles modulate probabilistic cube depth:
          - inter-community bridges get structural protection (at least `mid_k`),
          - community cores receive cohesive reinforcement,
          - redundant periphery / isolated tails in settled layers shed excess cube depth.
        """
        rel = self.relative_binding(key)
        mean_w = max(0.0, batch_weight_sum / max(1, batch_token_count))
        if batch_max_weight > 1e-6:
            batch_share = min(1.0, mean_w / batch_max_weight)
        else:
            batch_share = min(1.0, mean_w)
        strength = max(rel, batch_share)

        fb_floor = 0
        if self.pure_mirror_feedback is not None:
            # Version 3: 100% Pure Mirror Equality half-symmetry (ZERO UNI constants!)
            role, l_rel, b_rat, layer_has = self.pure_mirror_feedback.agent_mirror_feedback(key)
            if role == "mirror_bridge":
                strength = max(strength, l_rel, b_rat)
                fb_floor = min(self.mid_k, self.core_k)
                self.graphify_bridges_protected += 1
            elif role == "mirror_core":
                strength = max(strength, l_rel)
                self.graphify_cores_reinforced += 1
            elif role in ("mirror_periphery", "cold") and layer_has and batch_share < 0.5:
                strength = 0.5 * strength
                self.graphify_noise_damped += 1
        elif self.sphere_feedback is not None:
            role, l_rel, b_rat, layer_has_lenses = self.sphere_feedback.agent_sphere_feedback(key)
            if role == "sphere_intersection":
                strength = max(strength, l_rel, b_rat)
                fb_floor = min(self.mid_k, self.core_k)
                self.graphify_bridges_protected += 1
            elif role == "sphere_core":
                strength = max(strength, 0.5 * rel + 0.5 * l_rel)
                self.graphify_cores_reinforced += 1
            elif role in ("sphere_periphery", "cold") and layer_has_lenses and batch_share < (1.0 - UNI_PHI):
                strength = strength * (1.0 - UNI_PHI)
                self.graphify_noise_damped += 1

        span = max(0, self.core_k - self.shell_k)
        depth = max(fb_floor, int(round(self.shell_k + strength * span)))

        if blueprint_depth is not None:
            depth = min(depth, int(blueprint_depth))
        return max(self.min_shed_k, min(int(root_bits), int(depth)))

    def predict_next_layer_eids(
        self,
        layer_idx: int,
        active_eids: Sequence[int],
        *,
        top_n: int = 4,
    ) -> List[int]:
        """Predict strongly-coupled experts in `layer_idx + 1` via dimensionless corner90 share >= UNI_PHI."""
        if layer_idx + 1 >= self.num_layers:
            return []
        agg: Dict[int, float] = defaultdict(float)
        for ep in active_eids:
            for en, w in self.corner90_edges.get((int(layer_idx), int(ep)), {}).items():
                agg[en] += w
        if not agg:
            return []
        total = sum(agg.values())
        if total <= 1e-12:
            return []
        cut_share = 0.25 if self.pure_mirror_feedback is not None else UNI_PHI
        ranked = sorted(
            ((eid, w / total) for eid, w in agg.items() if (w / total) >= cut_share),
            key=lambda kv: -kv[1],
        )
        return [eid for eid, _ in ranked[:top_n]]

    def stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "tracked_vertices": len(self.mass),
            "anchored_vertices": len(self.anchor_mass),
            "total_face_links": self.total_face_links,
            "total_corner90_links": self.total_corner90_links,
            "depth_tiers": [self.core_k, self.mid_k, self.rim_k, self.shell_k],
            "min_shed_k": self.min_shed_k,
            "graphify_feedback": self.graphify_feedback,
        }
        if self.topology is not None:
            out["graphify_topology"] = self.topology.stats()
            out["graphify_bridges_protected"] = self.graphify_bridges_protected
            out["graphify_cores_reinforced"] = self.graphify_cores_reinforced
            out["graphify_noise_damped"] = self.graphify_noise_damped
            out["graphify_bridge_loop_wakes"] = self.graphify_bridge_loop_wakes
        return out


class ImprintPuzzleMoEOverlay:
    """Multi-resolution Probability Imprint Overlay with Depth-Shedding and strict A->B->C(0)->A relay.

    Attached to `LazyMoEBinaryHalvingCache` when `imprint_puzzle=True` (opt-in).
    """

    def __init__(
        self,
        cache: Any,
        *,
        core_k: int = 5,
        mid_k: int = 4,
        rim_k: int = 3,
        shell_k: int = 2,
        min_shed_k: int = 2,
        core_keep_frac: float = 0.18,
        core_min_k: int = 4,
        rest_tol: float = 0.24,
        rest_min: int = 3,
        prefill_coalesce_chunk: int = 8,
        auto_vram_adapt: bool = False,
        vram_safety_reserve_bytes: Optional[int] = None,
        fixation_cov_thresh: float = 1.0,
        stanok_contour: Optional[StanokContainmentContour] = None,
        graphify_feedback: Any = False,
    ) -> None:
        self.cache = cache
        self.device = cache.device
        self.feedback_mode: str = normalize_feedback_mode(graphify_feedback)
        self.graphify_feedback: bool = (self.feedback_mode != "off")
        self.graph = ProbabilityImprintGraph(
            cache.num_layers,
            cache.num_experts,
            core_k=core_k,
            mid_k=mid_k,
            rim_k=rim_k,
            shell_k=shell_k,
            min_shed_k=min_shed_k,
            graphify_feedback=self.feedback_mode,
            device=self.device,
        )
        self.core_keep_frac = max(0.0, min(0.5, float(core_keep_frac)))
        self.core_min_k = max(min_shed_k, min(core_k, int(core_min_k)))
        self.rest_tol = float(rest_tol)
        self.rest_min = max(1, int(rest_min))
        self.prefill_coalesce_chunk = max(1, int(prefill_coalesce_chunk))

        # Single shared outer Stanok Containment Contour (UUID-matched NVML + deadband)
        if stanok_contour is not None:
            self.contour = stanok_contour
            self._owns_contour = False
        else:
            self.contour = StanokContainmentContour(
                device=self.device,
                enabled=bool(auto_vram_adapt),
                safety_reserve_bytes=vram_safety_reserve_bytes,
            )
            self._owns_contour = True
        self.auto_vram_adapt = bool(self.contour.enabled)
        self.vram_safety_reserve_bytes = int(self.contour.safety_reserve_bytes)

        # Strict A -> B -> C(0) -> A relay arena for puzzle chunk/piece delivery
        self.relay = Q1RoleCycleArena(512 * 1024**2, self.device)
        self._stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quintessentia-puzzle-B")
        self._future = None
        self._pending_tag: Optional[Tuple[int, Tuple[int, ...]]] = None

        # Telemetry
        self.depth_sheds: int = 0
        self.bytes_freed_by_shedding: int = 0
        self.lattice_self_collapses: int = 0
        self.bytes_freed_by_self_collapse: int = 0
        self.imprint_evictions: int = 0
        self.coalesced_batches_loaded: int = 0
        self.coalesced_experts_loaded: int = 0
        self.prefill_chunks_relayed: int = 0
        self.decode_pieces_relayed: int = 0
        self.corner90_prefetches_submitted: int = 0
        self.corner90_prefetch_hits: int = 0
        self.relay_wait_seconds: float = 0.0
        self.vram_expansions: int = 0
        self.vram_contractions: int = 0
        self.oom_recoveries: int = 0
        self.min_observed_free_vram_bytes: Optional[int] = None
        self.max_observed_budget_bytes: int = int(cache.max_cache_bytes)
        self.last_trace: Dict[str, Any] = {}

        # ProbabilityBus hook for topic-shift equilibration pulses
        self.probability_bus: Any = None
        self.topic_shift_equilibrations: int = 0
        self._topic_shift_equilibrate_layers: int = 0

        # Spike/Equilibration resident-only drafting mode (ZERO disk reads when True!)
        self.draft_only_resident: bool = False
        self.draft_step_hits: int = 0
        self.draft_step_total: int = 0
        self.draft_step_weight_cov: float = 0.0
        self.draft_step_weight_tot: float = 0.0

        # Direct per-layer Imprint Fixation (opt-in; 1.0 = disabled):
        self.fixation_cov_thresh: float = max(0.0, min(1.0, float(fixation_cov_thresh)))
        self.fixation_layer_skips: int = 0
        self.fixation_experts_skipped: int = 0
        self.equilibration_layer_wakes: int = 0
        self.non_loop_branches_skipped: int = 0
        self.loop_closures_checked: int = 0

        # Optional interlayer B-prefetch state (layer l+1 connected puzzle piece prepared in B)
        self._interlayer_future = None
        self._interlayer_key: Optional[Tuple[int, int]] = None

        # Mutual Coalition <-> Agent Graphify feedback state & telemetry (zero global refresh!)
        self._bridge_open_layers: int = 0
        self.graphify_prefill_bridge_promotions: int = 0
        self.graphify_prefill_relax_collapses: int = 0
        self.graphify_exec_depth_trimmed: int = 0

        # Version 3 Pure Mirror Dual Prefill, Activation-Aware Crystallizer, Inner Metric g_Omega & Diff Q1
        self.entry_cone_layers: int = 4 if int(cache.num_layers) >= 8 else 0
        self.mirror_carve_tol: float = 0.06
        self.mirror_diff_tol: float = 0.03
        self._crystal_scars: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.mirror_prefill_crystallized: int = 0
        self.mirror_prefill_direct_pass: int = 0
        self.mirror_prefill_carves_avoided: int = 0
        self.mirror_crystal_k_opt_sum: int = 0
        self.mirror_crystal_k_max_sum: int = 0
        self.mirror_crystal_early_stops: int = 0
        self.mirror_decode_direct_unclosed: int = 0
        self.mirror_decode_direct_oneoff: int = 0
        self.mirror_diff_q1_calls: int = 0
        self.mirror_diff_q1_anchored_calls: int = 0

    def on_topic_shift(self, layer: Optional[int] = None) -> None:
        """Signal from UnifiedProbabilityBus that trajectory shifted -> continue seamlessly (NO reset!)."""
        self.topic_shift_equilibrations += 1
        self._topic_shift_equilibrate_layers = max(self._topic_shift_equilibrate_layers, 1)
        if self.graphify_feedback:
            self._bridge_open_layers = max(self._bridge_open_layers, 2)
            if self.graph.pure_mirror_feedback is not None:
                self.graph.pure_mirror_feedback.on_topic_shift(layer=layer)
            elif self.graph.sphere_feedback is not None:
                self.graph.sphere_feedback.on_topic_shift(layer=layer)

    def _query_physical_vram(self) -> Tuple[int, int, str]:
        """Return `(free_bytes, total_bytes, source)` via the single shared StanokContainmentContour."""
        return self.contour.query_physical_vram()

    def _rebalance_budget_to_free_vram(
        self,
        protected_layer: int,
        protected_eids: set[int],
        *,
        force_empty_cache: bool = False,
    ) -> None:
        """Delegate outer VRAM containment to the single shared StanokContainmentContour."""
        if not self.auto_vram_adapt or not self.contour.enabled or self.device.type != "cuda":
            return
        c = self.cache
        action, target_budget, emergency = self.contour.evaluate_budget(
            cached_bytes=c._cached_bytes,
            current_budget_bytes=c.max_cache_bytes,
        )
        self.min_observed_free_vram_bytes = self.contour.min_observed_free_vram_bytes
        if action == "contract":
            c.max_cache_bytes = target_budget
            self._make_room_by_shedding_or_evicting(0, protected_layer, protected_eids, emergency=emergency)
            torch.cuda.empty_cache()
            self.vram_contractions += 1
        elif action == "expand":
            c.max_cache_bytes = target_budget
            self.vram_expansions += 1
            if target_budget > self.max_observed_budget_bytes:
                self.max_observed_budget_bytes = target_budget

    def _carve_expert_tensor_pair(
        self,
        gu_raw: torch.Tensor,
        dn_raw: torch.Tensor,
        gu_depth: int,
        dn_depth: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int]:
        from src.binary_halving_q1 import carve_q1_cubes_incremental

        c = self.cache
        try:
            gu_gpu = gu_raw.to(device=self.device, dtype=torch.bfloat16)
            dn_gpu = dn_raw.to(device=self.device, dtype=torch.bfloat16)
            gu_t, _ = carve_q1_cubes_incremental(
                gu_gpu, gu_depth, root_bits=c.root_bits, group_size=c.group_size,
                use_fwht=c.use_fwht, sinkhorn_iters=c.sinkhorn_iters,
            )
            dn_t, _ = carve_q1_cubes_incremental(
                dn_gpu, dn_depth, root_bits=c.root_bits, group_size=c.group_size,
                use_fwht=c.use_fwht, sinkhorn_iters=c.sinkhorn_iters,
            )
            del gu_gpu, dn_gpu
            entry_b = _tensor_dict_bytes(gu_t) + _tensor_dict_bytes(dn_t)
            return gu_t, dn_t, entry_b
        except torch.cuda.OutOfMemoryError:
            # Self-healing OOM recovery: shed/shrink imprint to free VRAM and retry at safe minimal depth
            self.oom_recoveries += 1
            self.contour.record_oom_recovery()
            c.max_cache_bytes = max(0, c._cached_bytes // 2)
            self._make_room_by_shedding_or_evicting(0, -1, set(), emergency=True)
            torch.cuda.empty_cache()
            safe_k = max(1, min(self.graph.min_shed_k, gu_depth, dn_depth))
            gu_gpu = gu_raw.to(device=self.device, dtype=torch.bfloat16)
            dn_gpu = dn_raw.to(device=self.device, dtype=torch.bfloat16)
            gu_t, _ = carve_q1_cubes_incremental(
                gu_gpu, safe_k, root_bits=c.root_bits, group_size=c.group_size,
                use_fwht=c.use_fwht, sinkhorn_iters=c.sinkhorn_iters,
            )
            dn_t, _ = carve_q1_cubes_incremental(
                dn_gpu, safe_k, root_bits=c.root_bits, group_size=c.group_size,
                use_fwht=c.use_fwht, sinkhorn_iters=c.sinkhorn_iters,
            )
            del gu_gpu, dn_gpu
            entry_b = _tensor_dict_bytes(gu_t) + _tensor_dict_bytes(dn_t)
            return gu_t, dn_t, entry_b

    def _make_room_by_shedding_or_evicting(
        self,
        needed_bytes: int,
        protected_layer: int,
        protected_eids: set[int],
        *,
        emergency: bool = False,
    ) -> None:
        """Free `needed_bytes` in GPU imprint VRAM with Core Protection + OOM-proof fallback:
        Phase 1A (Peripheral Depth-Shedding): trim trailing Q1 planes (`k -> min_shed_k`) ONLY on
                 the lower `(1 - core_keep_frac)` peripheral experts, keeping the hot imprint core
                 at `core_min_k` (e.g. `k=4`, `cos > 0.996`)!
        Phase 1B (Peripheral Host Demotion): move coldest peripheral experts to Host L2.
        Phase 2  (Core Emergency Shedding / Demotion): ONLY if still over budget (or `emergency=True`
                 under tight physical VRAM), trim core down to `1` plane and demote if needed so
                 the engine NEVER hits OutOfMemoryError!
        """
        c = self.cache
        budget = c.max_cache_bytes
        if c._cached_bytes + needed_bytes <= budget:
            return

        # Gather non-protected candidates sorted by ascending imprint score (coldest first)
        cands = [
            (self.graph.score(k), k)
            for k in c._lru.keys()
            if not (k[0] == protected_layer and k[1] in protected_eids)
        ]
        cands.sort(key=lambda item: item[0])
        if not cands:
            return

        # Natural decentralized partition:
        #   - V3 Pure Mirror: pure mirror roles (`mirror_core`/`mirror_bridge` vs `mirror_periphery`/`cold`, ZERO UNI!)
        #   - V1/V2 Lattice : dimensionless lattice binding (`UNI_PHI = exp(-1.5)`)
        pm_fb = self.graph.pure_mirror_feedback if self.graphify_feedback else None
        if pm_fb is not None:
            periph = []
            core = []
            for item in cands:
                role, _, _, _ = pm_fb.agent_mirror_feedback(item[1])
                if role in ("mirror_core", "mirror_bridge"):
                    core.append(item)
                else:
                    periph.append(item)
        else:
            periph = [item for item in cands if item[0] < UNI_PHI]
            core = [item for item in cands if item[0] >= UNI_PHI]
        if not periph:
            split = max(1, len(cands) // 2)
            periph = cands[:split]
            core = cands[split:]

        def _shed_list(items: List[Tuple[float, Tuple[int, int]]], floor_k: int, step: int = 2) -> None:
            for _sc, key in items:
                if c._cached_bytes + needed_bytes <= budget:
                    return
                entry = c._lru.get(key)
                if entry is None:
                    continue
                gu_t, dn_t, old_b = entry
                cur_g = int(gu_t["packed_lo"].shape[0])
                cur_d = int(dn_t["packed_lo"].shape[0])
                if cur_g <= floor_k and cur_d <= floor_k:
                    continue
                new_g = max(floor_k, cur_g - step) if cur_g > floor_k else cur_g
                new_d = max(floor_k, cur_d - step) if cur_d > floor_k else cur_d
                new_gu = _trim_q1_planes(gu_t, new_g)
                new_dn = _trim_q1_planes(dn_t, new_d)
                new_b = _tensor_dict_bytes(new_gu) + _tensor_dict_bytes(new_dn)
                freed = max(0, old_b - new_b)
                if freed > 0:
                    c._lru[key] = (new_gu, new_dn, new_b)
                    c._cached_bytes -= freed
                    self.depth_sheds += 1
                    self.bytes_freed_by_shedding += freed
                    sc_rec = self._crystal_scars.get(key)
                    if sc_rec is None:
                        self._crystal_scars[key] = {
                            "original_k": max(cur_g, cur_d),
                            "current_k": max(new_g, new_d),
                            "shed_planes": max(0, max(cur_g, cur_d) - max(new_g, new_d)),
                            "source": "depth_shed",
                        }
                    else:
                        sc_rec["current_k"] = max(new_g, new_d)
                        sc_rec["shed_planes"] = max(0, int(sc_rec.get("original_k", max(cur_g, cur_d))) - max(new_g, new_d))
                        sc_rec["source"] = "depth_shed"

        def _demote_list(items: List[Tuple[float, Tuple[int, int]]]) -> None:
            for _sc, key in items:
                if c._cached_bytes + needed_bytes <= budget:
                    return
                item = c._lru.pop(key, None)
                if item is None:
                    continue
                old_gu, old_dn, old_b = item
                c._cached_bytes -= old_b
                c._mirror_anchors.pop(key, None)
                c.evictions += 1
                self.imprint_evictions += 1
                if c.host_cache_bytes > 0:
                    c._put_host(key, old_gu, old_dn, old_b)

        # Phase 1A: Shed peripheral experts down to `min_shed_k` (protecting core at `core_min_k`!)
        _shed_list(periph, self.graph.min_shed_k, step=2)
        if c._cached_bytes + needed_bytes <= budget:
            return
        # Gentle trim of core above `core_min_k` (e.g. k=5 -> k=4) before any eviction
        _shed_list(core, self.core_min_k, step=1)
        if c._cached_bytes + needed_bytes <= budget:
            return

        # Phase 1B: Demote coldest peripheral experts (NEVER hot core yet) to Host L2
        _demote_list(periph)
        if c._cached_bytes + needed_bytes <= budget:
            return

        # Phase 2 (Emergency / tiny VRAM): shed core down to `min_shed_k` -> `1`, then demote if required
        _shed_list(core, self.graph.min_shed_k, step=2)
        if c._cached_bytes + needed_bytes <= budget:
            return
        if emergency or budget < c._cached_bytes + needed_bytes:
            _shed_list(core, 1, step=2)
            if c._cached_bytes + needed_bytes <= budget:
                return
            _demote_list(core)

    def _relax_settled_layer_lattice(
        self,
        layer_idx: int,
        active_eids: set[int],
        *,
        u_in: Optional[torch.Tensor] = None,
        u_post: Optional[torch.Tensor] = None,
    ) -> int:
        """Internal self-collapse:
          - V3 Pure Mirror: autonomous GPU Kernel #5 (`pure_mirror_self_collapse_f32`) evaluates
            every collapsible non-active resident agent via the irreducible 4x4 Mirror Cube against
            live `(u_in, u_post)` and `active_eids` — collapsing `SPLIT` (uncoupled) and `EQUAL`
            (C03 duplicate) agents by 1 trailing Q1 bit-plane while HOLDING living `TENSION` /
            `mirror_bridge` agents crisp (ZERO UNI_PHI!).
          - V1/V2 Lattice : fading uncoupled nodes (`relative_binding < UNI_PHI`) collapse 1 plane.
        """
        c = self.cache
        li = int(layer_idx)
        pm_fb = self.graph.pure_mirror_feedback if self.graphify_feedback else None
        # Inner Space Metric g_Omega: in V3 Pure Mirror, keep crystals crisp (>= rim_k) while
        # containing VRAM has ample headroom (g_Omega < 0.75); only collapse down to min_shed_k under pressure!
        if pm_fb is not None:
            g_om = self._inner_space_metric_g_omega()
            floor_k = self.graph.min_shed_k if g_om >= 0.75 else max(self.graph.min_shed_k, self.graph.rim_k)
        else:
            floor_k = self.graph.min_shed_k
        collapsed = 0

        if pm_fb is not None and u_in is not None and u_post is not None:
            collapsible_keys: List[Tuple[int, int]] = []
            for key, entry in c._lru.items():
                if key[0] != li or key[1] in active_eids:
                    continue
                gu_t, dn_t, _old_b = entry
                if int(gu_t["packed_lo"].shape[0]) > floor_k or int(dn_t["packed_lo"].shape[0]) > floor_k:
                    collapsible_keys.append(key)
            if not collapsible_keys:
                return 0
            codes = pm_fb.evaluate_self_collapse_batch(
                li,
                [k[1] for k in collapsible_keys],
                sorted(active_eids),
                u_in,
                u_post,
            )
            keys_to_collapse = [k for k, code in zip(collapsible_keys, codes) if code != 0]
        else:
            keys_to_collapse = [
                k
                for k in c._lru.keys()
                if k[0] == li and k[1] not in active_eids and self.graph.relative_binding(k) < UNI_PHI
            ]

        for key in keys_to_collapse:
            entry = c._lru.get(key)
            if entry is None:
                continue
            gu_t, dn_t, old_b = entry
            cur_g = int(gu_t["packed_lo"].shape[0])
            cur_d = int(dn_t["packed_lo"].shape[0])
            if cur_g <= floor_k and cur_d <= floor_k:
                continue
            new_g = max(floor_k, cur_g - 1) if cur_g > floor_k else cur_g
            new_d = max(floor_k, cur_d - 1) if cur_d > floor_k else cur_d
            new_gu = _trim_q1_planes(gu_t, new_g)
            new_dn = _trim_q1_planes(dn_t, new_d)
            new_b = _tensor_dict_bytes(new_gu) + _tensor_dict_bytes(new_dn)
            freed = max(0, old_b - new_b)
            if freed > 0:
                c._lru[key] = (new_gu, new_dn, new_b)
                c._cached_bytes -= freed
                self.lattice_self_collapses += 1
                self.bytes_freed_by_self_collapse += freed
                collapsed += 1
                sc_rec = self._crystal_scars.get(key)
                if sc_rec is None:
                    self._crystal_scars[key] = {
                        "original_k": max(cur_g, cur_d),
                        "current_k": max(new_g, new_d),
                        "shed_planes": max(0, max(cur_g, cur_d) - max(new_g, new_d)),
                        "source": "self_collapse",
                    }
                else:
                    sc_rec["current_k"] = max(new_g, new_d)
                    sc_rec["shed_planes"] = max(0, int(sc_rec.get("original_k", max(cur_g, cur_d))) - max(new_g, new_d))
                    sc_rec["source"] = "self_collapse"
        if collapsed > 0:
            self.contour.record_lattice_self_collapse(collapsed)
        return collapsed

    def _store_in_imprint(
        self,
        key: Tuple[int, int],
        gu_t: Dict[str, torch.Tensor],
        dn_t: Dict[str, torch.Tensor],
        entry_b: int,
        protected_layer: int,
        protected_eids: set[int],
    ) -> None:
        c = self.cache
        if key in c._lru:
            _, _, prev_b = c._lru.pop(key)
            c._cached_bytes -= prev_b
        if c.max_cache_bytes > 0 and entry_b <= c.max_cache_bytes:
            self._make_room_by_shedding_or_evicting(entry_b, protected_layer, protected_eids)
            if c._cached_bytes + entry_b <= c.max_cache_bytes:
                store_gu = gu_t if c.cache_on_gpu else {k: v.detach().cpu() for k, v in gu_t.items()}
                store_dn = dn_t if c.cache_on_gpu else {k: v.detach().cpu() for k, v in dn_t.items()}
                c._lru[key] = (store_gu, store_dn, entry_b)
                c._cached_bytes += entry_b
                return
        if c.host_cache_bytes > 0:
            c._put_host(key, gu_t, dn_t, entry_b)

    @torch.no_grad()
    def _prepare_chunk_on_B(
        self,
        layer_idx: int,
        eids: List[int],
        depths: Dict[int, Tuple[int, int]],
    ) -> None:
        """Background area B: read a coalesced chunk of puzzle experts from disk and carve Q1 cubes."""
        c = self.cache
        if self.device.type == "cuda":
            with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
                if c.expert_batch_reader is not None and len(eids) > 1:
                    raw_map = c.expert_batch_reader(int(layer_idx), list(eids))
                else:
                    raw_map = {eid: c.expert_reader(int(layer_idx), int(eid)) for eid in eids}
                carved_map: Dict[int, Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, int, int]] = {}
                tot_bytes = 0
                for eid in eids:
                    gu_raw, dn_raw = raw_map.pop(eid)
                    gd, dd = depths[eid]
                    gu_t, dn_t, eb = self._carve_expert_tensor_pair(gu_raw, dn_raw, gd, dd)
                    del gu_raw, dn_raw
                    carved_map[eid] = (gu_t, dn_t, eb, gd, dd)
                    tot_bytes += eb
                ready = torch.cuda.Event(enable_timing=False)
                ready.record(self._stream)
            self.relay.publish_prepared(
                ("chunk", carved_map),
                gpu_bytes=min(tot_bytes, self.relay.slot_budget_bytes),
                ready_event=ready,
            )
        else:
            if c.expert_batch_reader is not None and len(eids) > 1:
                raw_map = c.expert_batch_reader(int(layer_idx), list(eids))
            else:
                raw_map = {eid: c.expert_reader(int(layer_idx), int(eid)) for eid in eids}
            carved_map = {}
            for eid in eids:
                gu_raw, dn_raw = raw_map.pop(eid)
                gd, dd = depths[eid]
                gu_t, dn_t, eb = self._carve_expert_tensor_pair(gu_raw, dn_raw, gd, dd)
                del gu_raw, dn_raw
                carved_map[eid] = (gu_t, dn_t, eb, gd, dd)
            self.relay.publish_prepared(("chunk", carved_map), gpu_bytes=0, ready_event=None)

    def _submit_chunk_to_B(
        self,
        layer_idx: int,
        eids: List[int],
        depths: Dict[int, Tuple[int, int]],
    ) -> None:
        self._pending_tag = (int(layer_idx), tuple(eids))
        self._future = self._pool.submit(self._prepare_chunk_on_B, int(layer_idx), list(eids), dict(depths))

    def _adopt_chunk_from_B(
        self,
        layer_idx: int,
        protected_eids: set[int],
    ) -> Dict[int, Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """Wait for B, rotate strict A->B->C(0)->A wheel (old_C->new_A adopts zero-copy, old_B->C(0))."""
        t0 = time.perf_counter()
        if self._future is not None:
            self._future.result()
        self.relay.rotate()
        payload = self.relay.active_payload()
        self.relay_wait_seconds += time.perf_counter() - t0
        self._future = None
        self._pending_tag = None

        _, carved_map = payload
        c = self.cache
        out: Dict[int, Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = {}
        self.coalesced_batches_loaded += 1
        self.coalesced_experts_loaded += len(carved_map)
        for eid, (gu_t, dn_t, eb, gd, dd) in carved_map.items():
            key = (int(layer_idx), int(eid))
            c.misses += 1
            c.unique_experts_carved.add(key)
            c.total_cubes_carved += int(gd) + int(dd)
            self._store_in_imprint(key, gu_t, dn_t, eb, int(layer_idx), protected_eids)
            out[eid] = (gu_t, dn_t)
        return out

    def _drain_interlayer_prefetch(self, protected_layer: int, protected_eids: set[int]) -> None:
        """If B finished an interlayer 90°-corner prefetch, adopt it cleanly via A->B->C(0)->A."""
        if self._interlayer_future is None:
            return
        li_pf, eid_pf = self._interlayer_key
        t0 = time.perf_counter()
        self._interlayer_future.result()
        self.relay.rotate()
        payload = self.relay.active_payload()
        self.relay_wait_seconds += time.perf_counter() - t0
        self._interlayer_future = None
        self._interlayer_key = None

        _, carved_map = payload
        c = self.cache
        for eid, (gu_t, dn_t, eb, gd, dd) in carved_map.items():
            key = (int(li_pf), int(eid))
            c.misses += 1
            c.unique_experts_carved.add(key)
            c.total_cubes_carved += int(gd) + int(dd)
            self._store_in_imprint(key, gu_t, dn_t, eb, protected_layer, protected_eids)
        self.relay.release_active_A()

    def _inner_space_metric_g_omega(self) -> float:
        """Inner containing-space pressure metric `g_Omega in [0, 1]` (acts from inside, not as an outer wall)."""
        c = self.cache
        if c.max_cache_bytes <= 0:
            return 1.0
        return min(1.0, max(0.0, float(c._cached_bytes) / float(c.max_cache_bytes)))

    @torch.no_grad()
    def _prepare_raw_chunk_on_B(
        self,
        layer_idx: int,
        eids: List[int],
    ) -> None:
        """Background area B: stream coalesced raw BF16 expert chunk via 4-worker NVRTC-GGUF (0 carving in B!)."""
        c = self.cache
        if self.device.type == "cuda":
            with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
                if c.expert_batch_reader is not None and len(eids) > 1:
                    raw_in = c.expert_batch_reader(int(layer_idx), list(eids))
                else:
                    raw_in = {eid: c.expert_reader(int(layer_idx), int(eid)) for eid in eids}
                raw_map: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
                tot_bytes = 0
                for eid in eids:
                    gu_r, dn_r = raw_in.pop(eid)
                    if gu_r.device != self.device or gu_r.dtype != torch.bfloat16:
                        gu_r = gu_r.to(device=self.device, dtype=torch.bfloat16, non_blocking=True)
                    if dn_r.device != self.device or dn_r.dtype != torch.bfloat16:
                        dn_r = dn_r.to(device=self.device, dtype=torch.bfloat16, non_blocking=True)
                    raw_map[eid] = (gu_r, dn_r)
                    tot_bytes += (
                        int(gu_r.numel()) * int(gu_r.element_size())
                        + int(dn_r.numel()) * int(dn_r.element_size())
                    )
                ready = torch.cuda.Event(enable_timing=False)
                ready.record(self._stream)
            self.relay.publish_prepared(
                ("raw_chunk", raw_map),
                gpu_bytes=min(tot_bytes, self.relay.slot_budget_bytes),
                ready_event=ready,
            )
        else:
            if c.expert_batch_reader is not None and len(eids) > 1:
                raw_in = c.expert_batch_reader(int(layer_idx), list(eids))
            else:
                raw_in = {eid: c.expert_reader(int(layer_idx), int(eid)) for eid in eids}
            raw_map = {}
            for eid in eids:
                gu_r, dn_r = raw_in.pop(eid)
                raw_map[eid] = (
                    gu_r.to(device=self.device, dtype=torch.bfloat16),
                    dn_r.to(device=self.device, dtype=torch.bfloat16),
                )
            self.relay.publish_prepared(("raw_chunk", raw_map), gpu_bytes=0, ready_event=None)

    def _submit_raw_chunk_to_B(self, layer_idx: int, eids: List[int]) -> None:
        self._pending_tag = (int(layer_idx), tuple(eids))
        self._future = self._pool.submit(self._prepare_raw_chunk_on_B, int(layer_idx), list(eids))

    def _adopt_raw_chunk_from_B(self) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Wait for B and rotate strict A->B->C(0)->A wheel (B->A == 0, zero-copy adoption into A)."""
        t0 = time.perf_counter()
        if self._future is not None:
            self._future.result()
        self.relay.rotate()
        payload = self.relay.active_payload()
        self.relay_wait_seconds += time.perf_counter() - t0
        self._future = None
        self._pending_tag = None
        _, raw_map = payload
        self.coalesced_batches_loaded += 1
        self.coalesced_experts_loaded += len(raw_map)
        return raw_map

    @torch.no_grad()
    def _crystallize_expert_with_probe(
        self,
        layer_idx: int,
        eid: int,
        gu_raw: torch.Tensor,
        dn_raw: torch.Tensor,
        x_probe: torch.Tensor,
        gu_exact: torch.Tensor,
        act_probe: torch.Tensor,
        dn_exact: torch.Tensor,
        gu_max_k: int,
        dn_max_k: int,
        protected_eids: set[int],
    ) -> None:
        """Crystallize a Left x Right intersection expert into Q1 up to activation-verified `k_opt`,
        recording reversible crystal scar `Gamma_scar` and crisp BF16 scar-anchor `(x_last, y_last)`.
        """
        from src.binary_halving_q1 import carve_q1_mirror_crystal

        c = self.cache
        key = (int(layer_idx), int(eid))
        g_om = self._inner_space_metric_g_omega()
        eff_gu_max = max(self.graph.min_shed_k, gu_max_k - (1 if g_om >= 0.875 else 0))
        eff_dn_max = max(self.graph.min_shed_k, dn_max_k - (1 if g_om >= 0.875 else 0))
        try:
            gu_t, _gu_meta, gu_scar = carve_q1_mirror_crystal(
                gu_raw, x_probe, gu_exact, eff_gu_max,
                min_cubes=self.graph.min_shed_k, mirror_tol=self.mirror_carve_tol,
                root_bits=c.root_bits, group_size=c.group_size,
                use_fwht=c.use_fwht, sinkhorn_iters=c.sinkhorn_iters,
            )
            dn_t, _dn_meta, dn_scar = carve_q1_mirror_crystal(
                dn_raw, act_probe, dn_exact, eff_dn_max,
                min_cubes=self.graph.min_shed_k, mirror_tol=self.mirror_carve_tol,
                root_bits=c.root_bits, group_size=c.group_size,
                use_fwht=c.use_fwht, sinkhorn_iters=c.sinkhorn_iters,
            )
            entry_b = _tensor_dict_bytes(gu_t) + _tensor_dict_bytes(dn_t)
        except torch.cuda.OutOfMemoryError:
            self.oom_recoveries += 1
            self.contour.record_oom_recovery()
            c.max_cache_bytes = max(0, c._cached_bytes // 2)
            self._make_room_by_shedding_or_evicting(0, -1, set(), emergency=True)
            torch.cuda.empty_cache()
            safe_k = max(1, min(self.graph.min_shed_k, eff_gu_max, eff_dn_max))
            gu_t, _gu_meta, gu_scar = carve_q1_mirror_crystal(
                gu_raw, x_probe, gu_exact, safe_k,
                min_cubes=1, mirror_tol=self.mirror_carve_tol,
                root_bits=c.root_bits, group_size=c.group_size,
                use_fwht=c.use_fwht, sinkhorn_iters=c.sinkhorn_iters,
            )
            dn_t, _dn_meta, dn_scar = carve_q1_mirror_crystal(
                dn_raw, act_probe, dn_exact, safe_k,
                min_cubes=1, mirror_tol=self.mirror_carve_tol,
                root_bits=c.root_bits, group_size=c.group_size,
                use_fwht=c.use_fwht, sinkhorn_iters=c.sinkhorn_iters,
            )
            entry_b = _tensor_dict_bytes(gu_t) + _tensor_dict_bytes(dn_t)

        k_opt_g = int(gu_scar["k_opt"])
        k_opt_d = int(dn_scar["k_opt"])
        self.mirror_crystal_k_opt_sum += k_opt_g + k_opt_d
        self.mirror_crystal_k_max_sum += eff_gu_max + eff_dn_max
        if k_opt_g < eff_gu_max:
            self.mirror_crystal_early_stops += 1
        if k_opt_d < eff_dn_max:
            self.mirror_crystal_early_stops += 1

        c.unique_experts_carved.add(key)
        c.total_cubes_carved += k_opt_g + k_opt_d
        self._crystal_scars[key] = {
            "original_k": max(k_opt_g, k_opt_d),
            "current_k": max(k_opt_g, k_opt_d),
            "shed_planes": 0,
            "gu_scar": gu_scar,
            "dn_scar": dn_scar,
            "source": "mirror_crystal",
        }
        self._store_in_imprint(key, gu_t, dn_t, entry_b, int(layer_idx), protected_eids)
        if key in c._lru:
            c._mirror_anchors[key] = (
                x_probe[-1].detach().to(device=self.device, dtype=torch.bfloat16).contiguous(),
                gu_exact[-1].detach().to(device=self.device, dtype=torch.bfloat16).contiguous(),
                act_probe[-1].detach().to(device=self.device, dtype=torch.bfloat16).contiguous(),
                dn_exact[-1].detach().to(device=self.device, dtype=torch.bfloat16).contiguous(),
            )

    @torch.no_grad()
    def _forward_prefill_pure_mirror(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        top_k_weights: torch.Tensor,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        per_tok_full: List[List[Tuple[int, int, float]]],
        token_routes: List[List[Tuple[int, float]]],
        all_eids_in_layer: set[int],
        layer_u_in: torch.Tensor,
        pm_fb: PureMirrorCudaFeedback,
    ) -> torch.Tensor:
        """Version 3 Mirror Dual Prefill (`MirrorDualPrefill`):
        1. Exact causal forward for every token (ZERO dropped prefill experts, ZERO KV-cache error!).
        2. Bidirectional mirror analysis of the layer's condition space (Left prompt body x Right
           `FuturePressure` from prompt tail `[-tail_n:]`) separates:
             - Living Intersection (`crystallize_set`) -> exact BF16 forward + activation-verified
               Q1 crystallization (`k_opt`) + crisp BF16 scar-anchor `(x_last, y_last)` for decode!
             - One-Off Periphery (`direct_pass`) -> exact BF16 forward directly from 4-worker
               NVRTC-GGUF chunk via strict `A->B->C(0)->A` (0 ms Sinkhorn+FWHT carving, 0 B bloat!).
        3. Commits the complete multi-ray layer object `Y_L` (`Y_prompt`) + reversible scar `Gamma_scar`.
        """
        c = self.cache
        li = int(layer_idx)
        m_tok, h_dim = int(hidden_states.shape[0]), int(hidden_states.shape[1])
        dev = hidden_states.device

        self.graph.observe_routing(li, token_routes, is_prefill=True, u_in=layer_u_in)
        self._rebalance_budget_to_free_vram(li, all_eids_in_layer)
        if self._interlayer_future is not None:
            pf_li, pf_eid = self._interlayer_key
            if pf_li == li and pf_eid in all_eids_in_layer:
                self.corner90_prefetch_hits += 1
            self._drain_interlayer_prefetch(li, all_eids_in_layer)

        # Group all routed slots across the M tokens by expert
        by_exp: Dict[int, Tuple[List[int], List[int], List[float]]] = {}
        for tok_i, valid in enumerate(per_tok_full):
            for e_int, k_pos, rw in valid:
                b = by_exp.get(e_int)
                if b is None:
                    b = ([], [], [])
                    by_exp[e_int] = b
                b[0].append(tok_i)
                b[1].append(k_pos)
                b[2].append(rw)

        if not by_exp:
            return torch.zeros_like(hidden_states)

        sorted_eids = sorted(by_exp)
        hot_eids: List[int] = []
        cold_eids: List[int] = []
        for eid in sorted_eids:
            if ((li, eid) in c._lru) or ((li, eid) in c._host_lru):
                hot_eids.append(eid)
            else:
                cold_eids.append(eid)

        # Bidirectional Mirror Analysis: Left prompt body x Right FuturePressure from prompt tail
        tail_n = max(1, min(4, (m_tok + 3) // 4))
        tail_start = max(0, m_tok - tail_n)
        tail_core_eids: set[int] = set()
        for tok_i in range(tail_start, m_tok):
            valid_t = per_tok_full[tok_i]
            n_vt = len(valid_t)
            eff_min = min(self.rest_min, max(1, n_vt - 1)) if n_vt > 1 else n_vt
            for rank, (e_int, _kp, _rw) in enumerate(valid_t):
                if rank < eff_min:
                    tail_core_eids.add(e_int)

        crystallize_set: set[int] = set()
        if cold_eids:
            u_tail = layer_u_in[tail_start:].mean(dim=0)
            u_tail = u_tail / u_tail.norm().clamp_min(1e-8)
            u_bodies = []
            for eid in cold_eids:
                t_idx = torch.tensor(by_exp[eid][0], dtype=torch.long, device=layer_u_in.device)
                ub = layer_u_in.index_select(0, t_idx).mean(dim=0)
                ub = ub / ub.norm().clamp_min(1e-8)
                u_bodies.append(ub)
            u_mat = torch.stack(u_bodies, dim=0)  # [U, 32]
            n_sum = (u_mat + u_tail.unsqueeze(0)).norm(dim=-1)
            n_dif = (u_mat - u_tail.unsqueeze(0)).norm(dim=-1)
            tie_fut = torch.where(
                n_sum > n_dif,
                (n_sum - n_dif) / (n_sum + n_dif + 1e-8),
                torch.zeros_like(n_sum),
            ).detach().cpu().tolist()

            w_sums = {eid: sum(by_exp[eid][2]) for eid in sorted_eids}
            total_slots = sum(len(by_exp[eid][0]) for eid in sorted_eids)
            mean_slot_w = sum(w_sums.values()) / max(1, total_slots)
            core_hit_bar = max(3, m_tok // 6)
            scored_cryst: List[Tuple[float, int]] = []
            for eid, tf in zip(cold_eids, tie_fut):
                hit_cnt = len(by_exp[eid][0])
                ws = w_sums[eid]
                mean_e_w = ws / max(1, hit_cnt)
                in_tail = (eid in tail_core_eids)
                # Left x Right intersection (scale-free 1/2 mirror balance, ZERO magic thresholds!):
                #   - Direct prompt tail core facing decode (`in_tail`), OR
                #   - Recurring prompt core/bridge (`hit_cnt >= 2` with `mean_e_w >= 0.5 * mean_slot_w`
                #     AND genuine mirror alignment to FuturePressure `tf >= 0.25` or strong recurrence `hit_cnt >= core_hit_bar`)
                # Weak bottom-tier noise tails and one-off experts stream directly in exact BF16 (0 ms carving!).
                if in_tail or (hit_cnt >= 2 and mean_e_w >= 0.5 * mean_slot_w and (tf >= 0.25 or hit_cnt >= core_hit_bar)):
                    prio = (2.0 if in_tail else 0.0) + ws * (1.0 + tf)
                    scored_cryst.append((prio, eid))

            # Inner-space fair-share cap prevents giant prompts from thrashing VRAM during prefill
            if scored_cryst:
                scored_cryst.sort(key=lambda kv: -kv[0])
                max_layer_cryst = max(16, min(64, len(scored_cryst)))
                crystallize_set = {eid for _, eid in scored_cryst[:max_layer_cryst]}

        cold_depths = self._compute_depths_for_eids(li, by_exp, sorted(crystallize_set)) if crystallize_set else {}
        line_spine = torch.zeros((m_tok, h_dim), dtype=torch.float32, device=dev)
        executed_routes_per_tok: List[List[Tuple[int, float]]] = [[] for _ in range(m_tok)]

        chunk_sz = max(self.prefill_coalesce_chunk, 16)
        cold_chunks: List[List[int]] = [
            cold_eids[i : i + chunk_sz] for i in range(0, len(cold_eids), chunk_sz)
        ]

        # Overlap: while A executes already-resident crystals, B streams cold_chunks[0] (raw BF16, 0 carving!)
        if cold_chunks:
            self._submit_raw_chunk_to_B(li, cold_chunks[0])

        for eid in hot_eids:
            pair = self._get_hot_or_host(li, eid, all_eids_in_layer)
            if pair is None:
                continue
            gu_t, dn_t = pair
            toks, kpos, rws = by_exp[eid]
            tok_t = torch.tensor(toks, dtype=torch.long, device=dev)
            kpos_t = torch.tensor(kpos, dtype=torch.long, device=dev)
            mean_rw = sum(rws) / max(1, len(rws))
            p_use = c.cubes_for_weight(mean_rw)
            x_e = hidden_states.index_select(0, tok_t)
            gu = c._run_proj(x_e, gu_t, p_use)
            g_part, u_part = gu.chunk(2, dim=-1)
            act = act_fn(g_part) * u_part
            y_e = c._run_proj(act, dn_t, p_use)
            _, used_cuda = launch_pure_mirror_line_step(line_spine, y_e, tok_t, top_k_weights[tok_t, kpos_t])
            if used_cuda:
                pm_fb.total_line_launches += 1
            for tok_i, rw in zip(toks, rws):
                executed_routes_per_tok[tok_i].append((eid, rw))
                c.routed_slots_executed += 1
            c.expert_calls += 1
            c.expert_cubes_executed += (
                min(p_use, gu_t["packed_lo"].shape[0]) + min(p_use, dn_t["packed_lo"].shape[0])
            ) / 2

        # Single-pass pipelined cold execution through strict A -> B -> C(0) -> A (ZERO per-token wave loops!)
        for ch_i, ch_eids in enumerate(cold_chunks):
            self._rebalance_budget_to_free_vram(li, all_eids_in_layer)
            raw_map = self._adopt_raw_chunk_from_B()
            self.prefill_chunks_relayed += 1
            if ch_i + 1 < len(cold_chunks):
                self._submit_raw_chunk_to_B(li, cold_chunks[ch_i + 1])

            for eid in ch_eids:
                gu_raw, dn_raw = raw_map.pop(eid)
                c.misses += 1
                toks, kpos, rws = by_exp[eid]
                tok_t = torch.tensor(toks, dtype=torch.long, device=dev)
                kpos_t = torch.tensor(kpos, dtype=torch.long, device=dev)
                x_e = hidden_states.index_select(0, tok_t).to(dtype=gu_raw.dtype)

                # Exact BF16 GPU Tensor Core forward (10x faster than carving + 0 Q1 error on prefill KV-cache!)
                gu_out = F.linear(x_e, gu_raw)
                g_part, u_part = gu_out.chunk(2, dim=-1)
                act = act_fn(g_part) * u_part
                y_e = F.linear(act, dn_raw)

                _, used_cuda = launch_pure_mirror_line_step(line_spine, y_e, tok_t, top_k_weights[tok_t, kpos_t])
                if used_cuda:
                    pm_fb.total_line_launches += 1
                for tok_i, rw in zip(toks, rws):
                    executed_routes_per_tok[tok_i].append((eid, rw))
                    c.routed_slots_executed += 1
                c.expert_calls += 1

                if eid in crystallize_set:
                    gd, dd = cold_depths[eid]
                    self._crystallize_expert_with_probe(
                        li, eid, gu_raw, dn_raw, x_e, gu_out, act, y_e, gd, dd, all_eids_in_layer
                    )
                    self.mirror_prefill_crystallized += 1
                else:
                    self.mirror_prefill_direct_pass += 1
                    self.mirror_prefill_carves_avoided += 1
                del gu_raw, dn_raw

            self.last_trace = {
                "layer": li,
                "chunk_eids": list(ch_eids),
                "is_prefill": True,
                "mirror_dual_prefill": True,
                "imprint_cached": len(c._lru),
            }
            self.relay.release_active_A()

        # Commit complete layer pass -> births multi-ray layer node Y_L (Y_prompt) + reversible scar Gamma_scar!
        for r_list in executed_routes_per_tok:
            r_list.sort(key=lambda item: -item[1])
        layer_u_post = pm_fb.project_states(li, hidden_states.float() + line_spine, update_mean=False)
        self.graph.observe_post_coil(li, executed_routes_per_tok, layer_u_in, layer_u_post)
        return line_spine.to(hidden_states.dtype)

    def _get_hot_or_host(
        self,
        layer_idx: int,
        eid: int,
        protected_eids: set[int],
    ) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """Return `(gu_t, dn_t)` if already resident in GPU imprint L1 or Host L2 (0 disk I/O)."""
        c = self.cache
        key = (int(layer_idx), int(eid))
        hit = c._lru.get(key)
        if hit is not None:
            c.hits += 1
            gu_t, dn_t, _ = hit
            if c.cache_on_gpu or self.device.type != "cuda":
                return gu_t, dn_t
            return (
                {k: v.to(self.device, non_blocking=True) for k, v in gu_t.items()},
                {k: v.to(self.device, non_blocking=True) for k, v in dn_t.items()},
            )

        host_hit = c._host_lru.pop(key, None)
        if host_hit is not None:
            c.host_hits += 1
            h_gu, h_dn, eb = host_hit
            c._host_cached_bytes -= eb
            gu_t = {k: v.to(self.device, non_blocking=True) for k, v in h_gu.items()}
            dn_t = {k: v.to(self.device, non_blocking=True) for k, v in h_dn.items()}
            self._store_in_imprint(key, gu_t, dn_t, eb, int(layer_idx), protected_eids)
            return gu_t, dn_t

        return None

    def _compute_depths_for_eids(
        self,
        layer_idx: int,
        by_exp: Dict[int, Tuple[List[int], List[int], List[float]]],
        eids: List[int],
    ) -> Dict[int, Tuple[int, int]]:
        c = self.cache
        depths: Dict[int, Tuple[int, int]] = {}
        max_mean_w = max(
            ((sum(rws) / max(1, len(toks))) for toks, _, rws in by_exp.values() if rws),
            default=0.0,
        )
        for eid in eids:
            key = (int(layer_idx), int(eid))
            toks, _, rws = by_exp.get(eid, ([], [], []))
            bp = c.expert_cubes.get(key)
            bp_g = bp[0] if bp is not None else None
            bp_d = bp[1] if bp is not None else None
            gd = self.graph.target_depth(
                key, batch_weight_sum=sum(rws), batch_token_count=len(toks),
                batch_max_weight=max_mean_w, blueprint_depth=bp_g, root_bits=c.root_bits,
            )
            dd = self.graph.target_depth(
                key, batch_weight_sum=sum(rws), batch_token_count=len(toks),
                batch_max_weight=max_mean_w, blueprint_depth=bp_d, root_bits=c.root_bits,
            )
            depths[eid] = (gd, dd)
        return depths

    @torch.no_grad()
    def forward_layer_moe(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Execute MoE layer over the Probability Imprint Overlay with strict A->B->C(0)->A relay."""
        c = self.cache
        li = int(layer_idx)
        m_tok, h_dim = int(hidden_states.shape[0]), int(hidden_states.shape[1])
        dev = hidden_states.device

        # FAST ZERO-DISK RESIDENT-IMPRINT DRAFT PATH (for fixed-stability burst generation!)
        if self.draft_only_resident:
            out_flat = torch.zeros_like(hidden_states)
            idx_cpu = top_k_index.detach().cpu().tolist()
            w_cpu = top_k_weights.detach().float().cpu().tolist()
            by_hot: Dict[int, Tuple[List[int], List[int], List[float]]] = {}
            for tok_i, (row_e, row_w) in enumerate(zip(idx_cpu, w_cpu)):
                for k_pos, (eid, rw) in enumerate(zip(row_e, row_w)):
                    e_int = int(eid)
                    rw_f = float(rw)
                    self.draft_step_total += 1
                    self.draft_step_weight_tot += rw_f
                    if (li, e_int) in c._lru:
                        self.draft_step_hits += 1
                        self.draft_step_weight_cov += rw_f
                        b = by_hot.get(e_int)
                        if b is None:
                            b = ([], [], [])
                            by_hot[e_int] = b
                        b[0].append(tok_i)
                        b[1].append(k_pos)
                        b[2].append(rw_f)
            for eid, (toks, kpos, rws) in by_hot.items():
                gu_t, dn_t, _ = c._lru[(li, eid)]
                tok_t = torch.tensor(toks, dtype=torch.long, device=dev)
                kpos_t = torch.tensor(kpos, dtype=torch.long, device=dev)
                p_use = int(gu_t["packed_lo"].shape[0])
                x_e = hidden_states.index_select(0, tok_t)
                gu = c._run_proj(x_e, gu_t, p_use)
                g_part, u_part = gu.chunk(2, dim=-1)
                act = act_fn(g_part) * u_part
                y_e = c._run_proj(act, dn_t, int(dn_t["packed_lo"].shape[0]))
                y_e = y_e * top_k_weights[tok_t, kpos_t, None]
                out_flat.index_add_(0, tok_t, y_e.to(out_flat.dtype))
            return out_flat

        is_prefill = (m_tok > 1)

        idx_cpu = top_k_index.detach().cpu().tolist()
        w_cpu = top_k_weights.detach().float().cpu().tolist()

        # 1. Build per-token sorted routing & update connected 3D Probability Imprint Graph
        token_routes: List[List[Tuple[int, float]]] = []
        per_tok_full: List[List[Tuple[int, int, float]]] = []
        all_eids_in_layer: set[int] = set()
        for tok_i, (row_e, row_w) in enumerate(zip(idx_cpu, w_cpu)):
            valid: List[Tuple[int, int, float]] = []
            for k_pos, (eid, rw) in enumerate(zip(row_e, row_w)):
                e_int = int(eid)
                if 0 <= e_int < c.num_experts:
                    valid.append((e_int, k_pos, float(rw)))
                    all_eids_in_layer.add(e_int)
            valid.sort(key=lambda it: -it[2])
            c.routed_slots_total += len(valid)
            per_tok_full.append(valid)
            token_routes.append([(e, rw) for e, _, rw in valid])

        # Capture previous layer's REAL executed loops & projected 32D info vectors BEFORE running this layer
        prev_layer_loops: List[List[int]] = (
            [[e for e, _ in pairs] for pairs in self.graph._last_layer_routes]
            if (self.graph._last_layer_idx == li - 1 and len(self.graph._last_layer_routes) == m_tok)
            else [[] for _ in range(m_tok)]
        )
        prev_layer_u_in = (
            self.graph._last_layer_u_in
            if (
                self.graph._last_layer_idx == li - 1
                and self.graph._last_layer_u_in is not None
                and len(self.graph._last_layer_u_in) == m_tok
            )
            else None
        )
        pm_fb = self.graph.pure_mirror_feedback if self.graphify_feedback else None
        sf_fb = self.graph.sphere_feedback if self.graphify_feedback else None
        if pm_fb is not None:
            layer_u_in = pm_fb.project_states(li, hidden_states, update_mean=True)
        elif sf_fb is not None:
            layer_u_in = sf_fb.project_states(li, hidden_states, update_mean=True)
        else:
            layer_u_in = None
        if is_prefill and pm_fb is not None and layer_u_in is not None:
            # Version 3 Mirror Dual Prefill: bidirectional Left prompt x Right FuturePressure!
            return self._forward_prefill_pure_mirror(
                li,
                hidden_states,
                top_k_weights,
                act_fn,
                per_tok_full,
                token_routes,
                all_eids_in_layer,
                layer_u_in,
                pm_fb,
            )
        if is_prefill:
            # Synchronously updates V1/V2 Spiral-Sphere feedback (zero delay!)
            self.graph.observe_routing(li, token_routes, is_prefill=True, u_in=layer_u_in)

        # Elastic OOM-proof VRAM regulator: breathe down if free VRAM dropped, expand if free grew!
        self._rebalance_budget_to_free_vram(li, all_eids_in_layer)

        # If B was prefetching an interlayer 90° corner expert for this layer, adopt it first!
        if self._interlayer_future is not None:
            pf_li, pf_eid = self._interlayer_key
            if pf_li == li and pf_eid in all_eids_in_layer:
                self.corner90_prefetch_hits += 1
            self._drain_interlayer_prefetch(li, all_eids_in_layer)

        # 2. Resident-First Lattice Execution (ZERO central creation centers!):
        #    - On decode with `rest_tol > 0`: run ALL already-resident lattice nodes FIRST (0 disk reads!),
        #      then let dimensionless `MirrorUniStabilizer` on `line_spine` decide naturally whether
        #      remaining cold nodes are still needed or whether the coupled lattice already settled!
        #    - Explicit `fixation_cov_thresh < 1.0` remains supported for backward compatibility.
        wave0_items: List[Tuple[int, int, int, float]] = []
        cold_queues: List[List[Tuple[int, int, float]]] = [[] for _ in range(m_tok)]
        tok_total_w: List[float] = [0.0] * m_tok
        layer_fixed_zero_disk = False
        phase_recorded = False
        misses_before_layer = c.misses
        rested_before_layer = c.cold_experts_rested

        for tok_i, valid in enumerate(per_tok_full):
            tot_w = sum(rw for _, _, rw in valid)
            tok_total_w[tok_i] = tot_w
            if (
                (not is_prefill)
                and (0.0 < self.fixation_cov_thresh < 1.0)
                and (self._topic_shift_equilibrate_layers <= 0)
                and tot_w > 1e-8
            ):
                res_items = [
                    (e_int, k_pos, rw)
                    for e_int, k_pos, rw in valid
                    if ((li, e_int) in c._lru) or ((li, e_int) in c._host_lru)
                ]
                res_w = sum(rw for _, _, rw in res_items)
                eff_min = min(self.rest_min, max(1, len(valid) - 1)) if len(valid) > 1 else len(valid)
                if len(res_items) >= eff_min and (res_w / tot_w) >= self.fixation_cov_thresh:
                    layer_fixed_zero_disk = True
                    phase_recorded = True
                    skipped = len(valid) - len(res_items)
                    self.fixation_layer_skips += 1
                    self.contour.record_layer_phase(fixated=True, skipped_experts=skipped)
                    if skipped > 0:
                        self.fixation_experts_skipped += skipped
                        c.cold_experts_rested += skipped
                        c.cold_disk_reads_avoided += skipped
                    for e_int, k_pos, rw in res_items:
                        wave0_items.append((tok_i, e_int, k_pos, rw))
                    continue
                else:
                    phase_recorded = True
                    self.equilibration_layer_wakes += 1
                    self.contour.record_layer_phase(fixated=False)
            elif (not is_prefill) and self._topic_shift_equilibrate_layers > 0:
                self._topic_shift_equilibrate_layers -= 1
                phase_recorded = True
                self.equilibration_layer_wakes += 1
                self.contour.record_layer_phase(fixated=False)
            for rank, (e_int, k_pos, rw) in enumerate(valid):
                in_imprint = ((li, e_int) in c._lru) or ((li, e_int) in c._host_lru)
                if self.rest_tol <= 0.0:
                    wave0_items.append((tok_i, e_int, k_pos, rw))
                elif is_prefill:
                    # On prefill, seed mandatory base minimum + resident lattice cleanly (no wave0 bloat)
                    eff_min = min(self.rest_min, max(1, len(valid) - 1)) if len(valid) > 1 else len(valid)
                    if rank < eff_min or in_imprint:
                        wave0_items.append((tok_i, e_int, k_pos, rw))
                    else:
                        cold_queues[tok_i].append((e_int, k_pos, rw))
                else:
                    # On decode, run ALL resident lattice nodes FIRST (0 B disk);
                    # cold nodes wake ONLY if line_spine tension stays unresolved!
                    if in_imprint:
                        wave0_items.append((tok_i, e_int, k_pos, rw))
                    else:
                        cold_queues[tok_i].append((e_int, k_pos, rw))

        line_spine = torch.zeros((m_tok, h_dim), dtype=torch.float32, device=dev)
        last_line_norm = [0.0] * m_tok
        last_obs_norm = [0.0] * m_tok
        prev_obs_norm = [0.0] * m_tok
        last_res_norm = [0.0] * m_tok
        last_w = [1e-6] * m_tok
        w_done = [0.0] * m_tok
        tok_done = [0] * m_tok
        last_ortho = [0.5] * m_tok
        rested = [False] * m_tok
        executed_routes_per_tok: List[List[Tuple[int, float]]] = [[] for _ in range(m_tok)]
        executed_eids_per_tok: List[List[int]] = [[] for _ in range(m_tok)]

        def _run_expert_on_A(
            eid: int,
            gu_t: Dict[str, torch.Tensor],
            dn_t: Dict[str, torch.Tensor],
            toks: List[int],
            kpos: List[int],
            rws: List[float],
        ) -> None:
            tok_t = torch.tensor(toks, dtype=torch.long, device=dev)
            kpos_t = torch.tensor(kpos, dtype=torch.long, device=dev)
            mean_rw = sum(rws) / max(1, len(rws))
            p_use = c.cubes_for_weight(mean_rw)
            x_e = hidden_states.index_select(0, tok_t)
            if (
                pm_fb is not None
                and m_tok == 1
                and c.use_fwht
                and c.micro_agent_tol is None
                and dev.type == "cuda"
                and x_e.dtype == torch.bfloat16
            ):
                # Version 3 Kernel #6: Differential-Reversible Pure Mirror Q1 CUDA GEMV!
                # Uses crisp BF16 scar-anchor (x_last, y_last) -> y_t = alpha * y_last + W_Q1(r_x)
                anc = c._mirror_anchors.get((li, eid))
                p_gu = min(p_use, int(gu_t["packed_lo"].shape[0]))
                p_dn = min(p_use, int(dn_t["packed_lo"].shape[0]))
                gu = launch_pure_mirror_diff_q1_gemv(
                    x_e,
                    gu_t["packed_lo"], gu_t["qscale"], gu_t["qzero"], gu_t["row_scale"], gu_t["col_scale"],
                    x_last=anc[0] if anc is not None else None,
                    y_last=anc[1] if anc is not None else None,
                    group_size=c.group_size,
                    max_planes=p_gu,
                    min_planes=1,
                    mirror_tol=self.mirror_diff_tol,
                )
                g_part, u_part = gu.chunk(2, dim=-1)
                act = act_fn(g_part) * u_part
                y_e = launch_pure_mirror_diff_q1_gemv(
                    act,
                    dn_t["packed_lo"], dn_t["qscale"], dn_t["qzero"], dn_t["row_scale"], dn_t["col_scale"],
                    x_last=anc[2] if anc is not None else None,
                    y_last=anc[3] if anc is not None else None,
                    group_size=c.group_size,
                    max_planes=p_dn,
                    min_planes=1,
                    mirror_tol=self.mirror_diff_tol,
                )
                pm_fb.total_diff_q1_launches += 2
                self.mirror_diff_q1_calls += 2
                if anc is not None:
                    self.mirror_diff_q1_anchored_calls += 2
            else:
                gu = c._run_proj(x_e, gu_t, p_use)
                g_part, u_part = gu.chunk(2, dim=-1)
                act = act_fn(g_part) * u_part
                y_e = c._run_proj(act, dn_t, p_use)

            if pm_fb is not None:
                # Version 3: Fused Irreducible Mirror Equality CUDA kernel on `line_spine` (with RelationResidual!)
                metrics, used_cuda = launch_pure_mirror_line_step(
                    line_spine, y_e, tok_t, top_k_weights[tok_t, kpos_t]
                )
                if used_cuda:
                    pm_fb.total_line_launches += 1
                m_rows = metrics.tolist()
                for local_i, tok_i in enumerate(toks):
                    sn_v, on_v, res_v, tie_v = m_rows[local_i]
                    last_line_norm[tok_i] = sn_v
                    prev_obs_norm[tok_i] = last_obs_norm[tok_i]
                    last_obs_norm[tok_i] = on_v
                    last_res_norm[tok_i] = res_v
                    last_w[tok_i] = rws[local_i]
                    w_done[tok_i] += rws[local_i]
                    tok_done[tok_i] += 1
                    executed_eids_per_tok[tok_i].append(eid)
                    executed_routes_per_tok[tok_i].append((eid, rws[local_i]))
                    last_ortho[tok_i] = 1.0 - tie_v
                    c.routed_slots_executed += 1
            else:
                obs_f = (y_e * top_k_weights[tok_t, kpos_t, None]).to(torch.float32)
                prev_lines = line_spine.index_select(0, tok_t)
                dots = (prev_lines * obs_f).sum(dim=-1)
                prev_norms = prev_lines.norm(dim=-1)
                obs_norms = obs_f.norm(dim=-1)
                line_spine.index_add_(0, tok_t, obs_f)

                triples = torch.stack([dots, prev_norms, obs_norms], dim=-1).tolist()
                for local_i, tok_i in enumerate(toks):
                    dot_v, pn_v, on_v = triples[local_i]
                    prev_obs_norm[tok_i] = last_obs_norm[tok_i]
                    last_obs_norm[tok_i] = on_v
                    last_w[tok_i] = rws[local_i]
                    w_done[tok_i] += rws[local_i]
                    tok_done[tok_i] += 1
                    executed_eids_per_tok[tok_i].append(eid)
                    executed_routes_per_tok[tok_i].append((eid, rws[local_i]))
                    last_ortho[tok_i] = MirrorUniStabilizer.update_ortho_angle(
                        dot_v, pn_v, on_v, default=last_ortho[tok_i]
                    )
                    c.routed_slots_executed += 1

            c.expert_calls += 1
            c.expert_cubes_executed += (
                min(p_use, gu_t["packed_lo"].shape[0]) + min(p_use, dn_t["packed_lo"].shape[0])
            ) / 2

        def _record_line_metrics(
            eid: int,
            y_e: torch.Tensor,
            toks: List[int],
            kpos: List[int],
            rws: List[float],
            tok_t: torch.Tensor,
            kpos_t: torch.Tensor,
        ) -> None:
            metrics, used_cuda = launch_pure_mirror_line_step(
                line_spine, y_e, tok_t, top_k_weights[tok_t, kpos_t]
            )
            if used_cuda and pm_fb is not None:
                pm_fb.total_line_launches += 1
            m_rows = metrics.tolist()
            for local_i, tok_i in enumerate(toks):
                sn_v, on_v, res_v, tie_v = m_rows[local_i]
                last_line_norm[tok_i] = sn_v
                prev_obs_norm[tok_i] = last_obs_norm[tok_i]
                last_obs_norm[tok_i] = on_v
                last_res_norm[tok_i] = res_v
                last_w[tok_i] = rws[local_i]
                w_done[tok_i] += rws[local_i]
                tok_done[tok_i] += 1
                executed_eids_per_tok[tok_i].append(eid)
                executed_routes_per_tok[tok_i].append((eid, rws[local_i]))
                last_ortho[tok_i] = 1.0 - tie_v
                c.routed_slots_executed += 1

        def _execute_wave(
            items: List[Tuple[int, int, int, float]],
            *,
            allow_crystallize: bool = True,
        ) -> None:
            if not items:
                return
            by_exp: Dict[int, Tuple[List[int], List[int], List[float]]] = {}
            for tok_i, e_int, k_pos, rw in items:
                b = by_exp.get(e_int)
                if b is None:
                    b = ([], [], [])
                    by_exp[e_int] = b
                b[0].append(tok_i)
                b[1].append(k_pos)
                b[2].append(rw)

            sorted_eids = sorted(by_exp)
            hot_eids: List[int] = []
            cold_eids: List[int] = []
            for eid in sorted_eids:
                if ((li, eid) in c._lru) or ((li, eid) in c._host_lru):
                    hot_eids.append(eid)
                else:
                    cold_eids.append(eid)

            # Group cold experts into coalesced disk chunks (up to 8 contiguous/nearby experts on prefill)
            chunk_sz = self.prefill_coalesce_chunk if is_prefill else 4
            cold_chunks: List[List[int]] = [
                cold_eids[i : i + chunk_sz] for i in range(0, len(cold_eids), chunk_sz)
            ]
            cold_depths = self._compute_depths_for_eids(li, by_exp, cold_eids) if cold_eids else {}

            # Version 3 Pure Mirror Decode Cold Path: stream raw BF16 via strict A->B->C(0)->A, compute
            # exact BF16 F.linear FIRST, and crystallize ONLY when Left x Right recurrence meets!
            if pm_fb is not None:
                if cold_chunks:
                    self._submit_raw_chunk_to_B(li, cold_chunks[0])
                for eid in hot_eids:
                    pair = self._get_hot_or_host(li, eid, all_eids_in_layer)
                    if pair is not None:
                        toks, kpos, rws = by_exp[eid]
                        _run_expert_on_A(eid, pair[0], pair[1], toks, kpos, rws)
                for ch_i, ch_eids in enumerate(cold_chunks):
                    self._rebalance_budget_to_free_vram(li, all_eids_in_layer)
                    raw_map = self._adopt_raw_chunk_from_B()
                    self.decode_pieces_relayed += 1
                    if ch_i + 1 < len(cold_chunks):
                        self._submit_raw_chunk_to_B(li, cold_chunks[ch_i + 1])
                    for eid in ch_eids:
                        gu_raw, dn_raw = raw_map.pop(eid)
                        c.misses += 1
                        toks, kpos, rws = by_exp[eid]
                        tok_t = torch.tensor(toks, dtype=torch.long, device=dev)
                        kpos_t = torch.tensor(kpos, dtype=torch.long, device=dev)
                        x_e = hidden_states.index_select(0, tok_t).to(dtype=gu_raw.dtype)
                        gu_out = F.linear(x_e, gu_raw)
                        g_part, u_part = gu_out.chunk(2, dim=-1)
                        act = act_fn(g_part) * u_part
                        y_e = F.linear(act, dn_raw)
                        _record_line_metrics(eid, y_e, toks, kpos, rws, tok_t, kpos_t)
                        c.expert_calls += 1
                        should_cryst = allow_crystallize and (self.graph.hit_count.get((li, eid), 0) >= 1)
                        if should_cryst:
                            gd, dd = cold_depths[eid]
                            self._crystallize_expert_with_probe(
                                li, eid, gu_raw, dn_raw, x_e, gu_out, act, y_e, gd, dd, all_eids_in_layer
                            )
                        else:
                            self.mirror_decode_direct_oneoff += 1
                        del gu_raw, dn_raw
                    self.last_trace = {
                        "layer": li,
                        "chunk_eids": list(ch_eids),
                        "is_prefill": False,
                        "imprint_cached": len(c._lru),
                    }
                    self.relay.release_active_A()
                return

            # Overlap! While A runs all already-resident imprint experts in VRAM, B prepares cold_chunks[0]!
            if cold_chunks:
                self._submit_chunk_to_B(li, cold_chunks[0], cold_depths)

            for eid in hot_eids:
                pair = self._get_hot_or_host(li, eid, all_eids_in_layer)
                if pair is not None:
                    toks, kpos, rws = by_exp[eid]
                    _run_expert_on_A(eid, pair[0], pair[1], toks, kpos, rws)

            # Pipeline cold chunks through strict A -> B -> C(0) -> A wheel!
            for ch_i, ch_eids in enumerate(cold_chunks):
                self._rebalance_budget_to_free_vram(li, all_eids_in_layer)
                loaded_map = self._adopt_chunk_from_B(li, all_eids_in_layer)
                if is_prefill:
                    self.prefill_chunks_relayed += 1
                else:
                    self.decode_pieces_relayed += 1
                # Immediately kick off B for the next coalesced chunk while A computes current chunk!
                if ch_i + 1 < len(cold_chunks):
                    self._submit_chunk_to_B(li, cold_chunks[ch_i + 1], cold_depths)
                for eid in ch_eids:
                    gu_t, dn_t = loaded_map[eid]
                    toks, kpos, rws = by_exp[eid]
                    _run_expert_on_A(eid, gu_t, dn_t, toks, kpos, rws)
                self.last_trace = {
                    "layer": li,
                    "chunk_eids": list(ch_eids),
                    "is_prefill": is_prefill,
                    "imprint_cached": len(c._lru),
                }
                self.relay.release_active_A()

        # Execute Wave 0 (imprint hits + mandatory core)
        _execute_wave(wave0_items)

        # 3. Loop-centric continuation check: orient around LOOPS & INTERSECTIONS, not isolated agents!
        #    When the resident loop is already closed and dominant on `line_spine`, any dangling cold
        #    agent that cannot close a new loop/intersection (`< 2` active face/corner contacts) is
        #    left asleep as non-generating (`"no_new_loop"`). Loop-closing candidates are checked via
        #    order-invariant Mirror Equality x UNI on `line_spine`.
        while self.rest_tol > 0.0:
            cand_toks = [t for t in range(m_tok) if (not rested[t]) and cold_queues[t]]
            if not cand_toks:
                break

            use_pure_mirror_gate = (not is_prefill) and (pm_fb is not None) and (layer_u_in is not None)
            inv_pairs = (
                []
                if use_pure_mirror_gate
                else MirrorUniStabilizer.compute_line_invariants(line_spine, cand_toks)
            )
            next_wave: List[Tuple[int, int, int, float]] = []
            direct_unclosed_wave: List[Tuple[int, int, int, float]] = []
            avoided_eids: set[int] = set()
            woken_eids: set[int] = set()
            progressed_any = False
            unclosed_protect = False
            if use_pure_mirror_gate:
                l_code, _, _, _, _ = pm_fb.layer_node_state(li)
                unclosed_protect = (li < self.entry_cone_layers) or (l_code == EQ_SPLIT)

            for ci, tok_i in enumerate(cand_toks):
                n_valid_tok = len(per_tok_full[tok_i])
                eff_min_tok = min(self.rest_min, max(1, n_valid_tok - 1)) if n_valid_tok > 1 else n_valid_tok
                prev_loop_tok = prev_layer_loops[tok_i] if tok_i < len(prev_layer_loops) else []

                if use_pure_mirror_gate:
                    # Version 3: Evaluate ALL queued cold candidates on GPU in ONE Bracketless Mirror launch!
                    # First drain any head candidate that already became resident in VRAM/Host
                    if cold_queues[tok_i]:
                        e_head, kp_head, nw_head = cold_queues[tok_i][0]
                        if ((li, e_head) in c._lru) or ((li, e_head) in c._host_lru):
                            cold_queues[tok_i].pop(0)
                            next_wave.append((tok_i, e_head, kp_head, nw_head))
                            woken_eids.add(e_head)
                            progressed_any = True
                            continue
                    q_items = list(cold_queues[tok_i])
                    c_eids = [it[0] for it in q_items]
                    c_ws = [it[2] for it in q_items]
                    c_gloop = [
                        1 if self.graph.closes_active_loop(li, ce, executed_eids_per_tok[tok_i], prev_loop_tok) else 0
                        for ce in c_eids
                    ]
                    u_tok = layer_u_in[tok_i]
                    prev_u_tok = (
                        prev_layer_u_in[tok_i]
                        if (prev_layer_u_in is not None and tok_i < len(prev_layer_u_in))
                        else None
                    )
                    batch_res = pm_fb.evaluate_continuation_batch(
                        li,
                        executed_eids_per_tok[tok_i],
                        prev_loop_tok,
                        c_eids,
                        c_ws,
                        c_gloop,
                        u_info=u_tok,
                        prev_u_info=prev_u_tok,
                        line_norm=last_line_norm[tok_i],
                        last_obs_norm=last_obs_norm[tok_i],
                        last_res_norm=last_res_norm[tok_i],
                        w_done=w_done[tok_i],
                        rem_w=tok_total_w[tok_i] - w_done[tok_i],
                        tok_done=tok_done[tok_i],
                        rest_min=eff_min_tok,
                        rest_tol=self.rest_tol,
                        unclosed_cone_protect=unclosed_protect,
                    )
                    for idx_q, (reason, _min_e, delta_e, defect_v, _tol) in enumerate(batch_res):
                        e_int, k_pos, next_w = cold_queues[tok_i][0]
                        self.loop_closures_checked += 1
                        if reason == "no_mirror_link":
                            cold_queues[tok_i].pop(0)
                            c.cold_experts_rested += 1
                            self.non_loop_branches_skipped += 1
                            avoided_eids.add(e_int)
                            progressed_any = True
                            continue
                        if reason == "direct_pass_unclosed":
                            # Law of Two Slits: NEVER drop an unclosed entry-cone / SPLIT branch to zero!
                            cold_queues[tok_i].pop(0)
                            direct_unclosed_wave.append((tok_i, e_int, k_pos, next_w))
                            woken_eids.add(e_int)
                            self.mirror_decode_direct_unclosed += 1
                            progressed_any = True
                            break
                        if reason == "mirror_maintained":
                            if delta_e <= 1e-8 and defect_v <= 1e-8:
                                # Candidate-level C03/L04 empty repetition (EQ_EQUAL): skip just this candidate
                                # so a genuine TENSION bridge behind it in the queue is still checked!
                                cold_queues[tok_i].pop(0)
                                c.cold_experts_rested += 1
                                self.non_loop_branches_skipped += 1
                                avoided_eids.add(e_int)
                                progressed_any = True
                                continue
                            # Line-level mirror balance (`defect < eff_tol`): rest entire remaining tail
                            rested[tok_i] = True
                            for rem_e, _, _ in cold_queues[tok_i]:
                                c.cold_experts_rested += 1
                                avoided_eids.add(rem_e)
                            cold_queues[tok_i].clear()
                            progressed_any = True
                            break
                        if reason == "mirror_expand" and delta_e > 1e-7:
                            self.graph.graphify_bridge_loop_wakes += 1
                        cold_queues[tok_i].pop(0)
                        next_wave.append((tok_i, e_int, k_pos, next_w))
                        woken_eids.add(e_int)
                        progressed_any = True
                        break
                    continue

                next_l_norm, lam_v = inv_pairs[ci]
                while cold_queues[tok_i]:
                    e_int, k_pos, next_w = cold_queues[tok_i][0]
                    if ((li, e_int) in c._lru) or ((li, e_int) in c._host_lru):
                        cold_queues[tok_i].pop(0)
                        next_wave.append((tok_i, e_int, k_pos, next_w))
                        woken_eids.add(e_int)
                        progressed_any = True
                        break

                    self.loop_closures_checked += 1
                    min_rc: Optional[int] = None
                    min_re: Optional[float] = None
                    delta_rc: Optional[int] = None
                    delta_re: Optional[float] = None
                    if is_prefill:
                        closes_loop = True
                    elif self.graphify_feedback and self.graph.sphere_feedback is not None:
                        u_tok = layer_u_in[tok_i] if (layer_u_in is not None and tok_i < len(layer_u_in)) else None
                        prev_u_tok = (
                            prev_layer_u_in[tok_i]
                            if (prev_layer_u_in is not None and tok_i < len(prev_layer_u_in))
                            else None
                        )
                        min_rc, min_re, delta_rc, delta_re = (
                            self.graph.compare_minimum_rings_with_continuation(
                                li,
                                executed_eids_per_tok[tok_i],
                                prev_loop_tok,
                                e_int,
                                u_info=u_tok,
                                prev_u_info=prev_u_tok,
                            )
                        )
                        closes_loop = (delta_rc > 0) or (
                            min_rc == 0
                            and self.graph.closes_active_loop(
                                li, e_int, executed_eids_per_tok[tok_i], prev_loop_tok
                            )
                        )
                    else:
                        closes_loop = self.graph.closes_active_loop(
                            li, e_int, executed_eids_per_tok[tok_i], prev_loop_tok
                        )
                    should_rest, _defect, _eff_tol, reason = MirrorUniStabilizer.evaluate_loop_continuation(
                        line_norm=next_l_norm,
                        lam_v=lam_v,
                        last_obs_norm=last_obs_norm[tok_i],
                        prev_obs_norm=prev_obs_norm[tok_i],
                        w_done=w_done[tok_i],
                        next_w=next_w,
                        rem_w=tok_total_w[tok_i] - w_done[tok_i],
                        last_ortho=last_ortho[tok_i],
                        tok_done=tok_done[tok_i],
                        rest_min=eff_min_tok,
                        rest_tol=self.rest_tol,
                        closes_new_loop=closes_loop,
                        min_ring_count=min_rc,
                        min_ring_energy=min_re,
                        delta_ring_count=delta_rc,
                        delta_ring_energy=delta_re,
                    )
                    if reason == "loop_expand" and delta_rc is not None and delta_rc > 0:
                        self.graph.graphify_bridge_loop_wakes += 1
                    if reason == "no_new_loop":
                        # Dangling isolated agent cannot close a new loop/intersection -> skip it!
                        cold_queues[tok_i].pop(0)
                        c.cold_experts_rested += 1
                        self.non_loop_branches_skipped += 1
                        avoided_eids.add(e_int)
                        progressed_any = True
                        continue
                    if should_rest:
                        rested[tok_i] = True
                        for rem_e, _, _ in cold_queues[tok_i]:
                            c.cold_experts_rested += 1
                            avoided_eids.add(rem_e)
                        cold_queues[tok_i].clear()
                        progressed_any = True
                        break
                    cold_queues[tok_i].pop(0)
                    next_wave.append((tok_i, e_int, k_pos, next_w))
                    woken_eids.add(e_int)
                    progressed_any = True
                    break

            c.cold_disk_reads_avoided += len(avoided_eids - woken_eids)
            if direct_unclosed_wave:
                _execute_wave(direct_unclosed_wave, allow_crystallize=False)
            if not next_wave:
                if not progressed_any and not direct_unclosed_wave:
                    break
                continue
            _execute_wave(next_wave)

        # Complete active step with actual generated residual `hidden_states + line_spine`
        if pm_fb is not None:
            layer_u_post = pm_fb.project_states(li, hidden_states.float() + line_spine, update_mean=False)
        elif sf_fb is not None:
            layer_u_post = sf_fb.project_states(li, hidden_states.float() + line_spine, update_mean=False)
        else:
            layer_u_post = None
        if is_prefill:
            if layer_u_post is not None:
                self.graph.observe_post_coil(li, executed_routes_per_tok, layer_u_in, layer_u_post)
        else:
            # On decode, record ONLY the real executed loops & continuous spiral coils (NO reset!)
            for r_list in executed_routes_per_tok:
                r_list.sort(key=lambda item: -item[1])
            self.graph.observe_routing(
                li,
                executed_routes_per_tok,
                is_prefill=False,
                u_in=layer_u_in,
                u_post=layer_u_post,
            )

            layer_cold_wakes = c.misses - misses_before_layer
            layer_rested_delta = c.cold_experts_rested - rested_before_layer
            active_executed = {e for lst in executed_eids_per_tok for e in lst}
            if layer_cold_wakes == 0:
                layer_fixed_zero_disk = True
                if not phase_recorded:
                    self.fixation_layer_skips += 1
                    self.fixation_experts_skipped += layer_rested_delta
                    self.contour.record_layer_phase(fixated=True, skipped_experts=layer_rested_delta)
                if pm_fb is None:
                    # V1/V2: coupled lattice resolved cleanly without disk chaos -> relax uncoupled fading nodes
                    self._relax_settled_layer_lattice(li, active_executed)
            elif not phase_recorded:
                self.equilibration_layer_wakes += 1
                self.contour.record_layer_phase(fixated=False)

            if pm_fb is not None and layer_u_in is not None and layer_u_post is not None:
                # V3 Pure Mirror: autonomous 4x4 self-collapse on EVERY decode step (1-to-1 with rings!)
                self._relax_settled_layer_lattice(
                    li, active_executed, u_in=layer_u_in, u_post=layer_u_post
                )

        # 4. On decode (M == 1): use 90° interlayer corner graph to proactively prefetch 1 cold
        #    connected puzzle piece of layer `l+1` into B while next layer's attention runs!
        #    Skip disk prefetch if THIS layer was locked in zero-disk fixation so pure spikes
        #    stay 100% disk-free!
        if (
            (not is_prefill)
            and (not layer_fixed_zero_disk)
            and (li + 1 < c.num_layers)
            and self._interlayer_future is None
        ):
            top_active = [e for e, _ in token_routes[0][:3]] if token_routes else []
            if pm_fb is not None and layer_u_post is not None:
                pred_next = pm_fb.predict_next_layer_mirror(
                    li, top_active, layer_u_post, self.graph.corner90_edges, top_n=4
                )
            else:
                pred_next = self.graph.predict_next_layer_eids(li, top_active, top_n=4)
            for cand_e in pred_next:
                if ((li + 1, cand_e) not in c._lru) and ((li + 1, cand_e) not in c._host_lru):
                    bp = c.expert_cubes.get((li + 1, cand_e))
                    gd = self.graph.target_depth(
                        (li + 1, cand_e), blueprint_depth=bp[0] if bp else None, root_bits=c.root_bits
                    )
                    dd = self.graph.target_depth(
                        (li + 1, cand_e), blueprint_depth=bp[1] if bp else None, root_bits=c.root_bits
                    )
                    self._interlayer_key = (li + 1, cand_e)
                    self._interlayer_future = self._pool.submit(
                        self._prepare_chunk_on_B, li + 1, [cand_e], {cand_e: (gd, dd)}
                    )
                    self.corner90_prefetches_submitted += 1
                    break

        if (not is_prefill) and self._bridge_open_layers > 0:
            self._bridge_open_layers -= 1

        return line_spine.to(hidden_states.dtype)

    def stats(self) -> Dict[str, Any]:
        c = self.cache
        depth_hist: Dict[int, int] = defaultdict(int)
        for gu_t, _dn_t, _b in c._lru.values():
            depth_hist[int(gu_t["packed_lo"].shape[0])] += 1
        rel_st = self.relay.stats()
        rel_st.pop("role_history", None)  # keep stats JSON compact
        return {
            "graph": self.graph.stats(),
            "vram_depth_histogram": {str(k): v for k, v in sorted(depth_hist.items())},
            "depth_sheds": self.depth_sheds,
            "mib_freed_by_shedding": round(self.bytes_freed_by_shedding / 1024**2, 2),
            "lattice_self_collapses": self.lattice_self_collapses,
            "mib_freed_by_self_collapse": round(self.bytes_freed_by_self_collapse / 1024**2, 2),
            "imprint_evictions": self.imprint_evictions,
            "coalesced_batches_loaded": self.coalesced_batches_loaded,
            "coalesced_experts_loaded": self.coalesced_experts_loaded,
            "prefill_chunks_relayed": self.prefill_chunks_relayed,
            "decode_pieces_relayed": self.decode_pieces_relayed,
            "corner90_prefetches_submitted": self.corner90_prefetches_submitted,
            "corner90_prefetch_hits": self.corner90_prefetch_hits,
            "relay_wait_seconds": round(self.relay_wait_seconds, 4),
            "auto_vram_adapt": self.auto_vram_adapt,
            "vram_sensor": self.contour.sensor_source,
            "vram_safety_reserve_mib": round(self.vram_safety_reserve_bytes / 1024**2, 1),
            "core_keep_frac": self.core_keep_frac,
            "core_min_k": self.core_min_k,
            "vram_expansions": self.vram_expansions,
            "vram_contractions": self.vram_contractions,
            "oom_recoveries": self.oom_recoveries,
            "fixation_cov_thresh": self.fixation_cov_thresh,
            "fixation_layer_skips": self.fixation_layer_skips,
            "fixation_experts_skipped": self.fixation_experts_skipped,
            "equilibration_layer_wakes": self.equilibration_layer_wakes,
            "non_loop_branches_skipped": self.non_loop_branches_skipped,
            "loop_closures_checked": self.loop_closures_checked,
            "topic_shift_equilibrations": self.topic_shift_equilibrations,
            "graphify_feedback": {
                "enabled": self.graphify_feedback,
                "mode": self.feedback_mode,
                "prefill_bridge_promotions": self.graphify_prefill_bridge_promotions,
                "prefill_relax_collapses": self.graphify_prefill_relax_collapses,
                "exec_depth_trimmed": self.graphify_exec_depth_trimmed,
                "bridges_protected": self.graph.graphify_bridges_protected,
                "cores_reinforced": self.graph.graphify_cores_reinforced,
                "noise_damped": self.graph.graphify_noise_damped,
                "bridge_loop_wakes": self.graph.graphify_bridge_loop_wakes,
                "topology": (
                    self.graph.topology.stats() if self.graph.topology is not None else None
                ),
            },
            "mirror_v3": {
                "prefill_crystallized": self.mirror_prefill_crystallized,
                "prefill_direct_pass": self.mirror_prefill_direct_pass,
                "prefill_carves_avoided": self.mirror_prefill_carves_avoided,
                "crystal_k_opt_avg": round(
                    self.mirror_crystal_k_opt_sum / max(1, 2 * self.mirror_prefill_crystallized), 2
                ) if self.mirror_prefill_crystallized > 0 else 0.0,
                "crystal_early_stops": self.mirror_crystal_early_stops,
                "crystal_scars_active": len(self._crystal_scars),
                "decode_direct_unclosed": self.mirror_decode_direct_unclosed,
                "decode_direct_oneoff": self.mirror_decode_direct_oneoff,
                "diff_q1_calls": self.mirror_diff_q1_calls,
                "diff_q1_anchored_calls": self.mirror_diff_q1_anchored_calls,
                "anchors_resident": len(c._mirror_anchors),
            },
            "min_observed_free_vram_mib": (
                None if self.min_observed_free_vram_bytes is None
                else round(self.min_observed_free_vram_bytes / 1024**2, 1)
            ),
            "max_observed_budget_mib": round(self.max_observed_budget_bytes / 1024**2, 1),
            "relay": {
                **rel_st,
                "standby_bytes": self.relay.area_used_bytes[self.relay.standby_area],
                "last_trace": dict(self.last_trace),
            },
        }

    def close(self) -> None:
        if self._interlayer_future is not None:
            try:
                self._interlayer_future.result()
            except Exception:
                pass
            self._interlayer_future = None
            self._interlayer_key = None
        if self._future is not None:
            try:
                self._future.result()
            except Exception:
                pass
            self._future = None
            self._pending_tag = None
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        if self._owns_contour and self.contour is not None:
            self.contour.close()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.relay.buffers[:] = [None, None, None]
        self.relay._pinned_host_scratch[:] = [None, None, None]
