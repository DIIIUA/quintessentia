"""Q1PuzzleOverlay: Base Ideal Q1 (P=1) + Dynamic 3D Contact-Graph Sparse Q8 Delta Cubes.

Implements Wave 4 dynamic sparse refinement over a full-coverage Ideal Q1 (P=1) base:
  Y[:, r0:r1] = Y^(Q1)[:, r0:r1] + sum_{(r, c) in S_t} X[:, c0:c1] @ (DeltaW[r, c])^T

where S_t is a per-forward connected puzzle slice built from:
  - 40% seed cubes ranked by e_y(r) * e_x(c) * ||DeltaW[r, c]||_F
  - 60% frontier expansion across three physical contact types:
      * horiz       : (r, c +/- 1) and paired matrices (q<->k<->v with GQA, gate<->up)
      * vert        : (r +/- 1, c)
      * diag_corner : (r +/- 1, c +/- 1) and 90-deg corner rotations (output r == input c')
                      between gate/up -> down, q/v -> o, layer l -> l+1, plus lm_head feedback.

Selected Q8 delta cubes reside in Host RAM and stream strictly through
Q1RoleCycleArena (A -> B -> C(0) -> A, B->A == 0, C == 0 B) into NVRTC kernel #11
`puzzle_overlay_delta_cubes_f32` (on CUDA) or an exact FP32 accumulation fallback (on CPU).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from src.ideal_q1_engine import IdealQ1Linear, unpack_ideal_q1
from src.q1_cube_engine import Q1RoleCycleArena


_CONTACT_SEED = "seed"
_CONTACT_HORIZ = "horiz"
_CONTACT_VERT = "vert"
_CONTACT_DIAG_CORNER = "diag_corner"
_CONTACT_NAMES = (_CONTACT_SEED, _CONTACT_HORIZ, _CONTACT_VERT, _CONTACT_DIAG_CORNER)

_SKIP_ROUTER_EXACT = frozenset({"gate", "router", "router_proj"})


def _infer_proj_role(rel_name: str, leaf_name: str) -> str:
    """Classify a decoder linear projection role from its relative/leaf module name."""
    lower = leaf_name.lower()
    rel_lower = rel_name.lower()
    for role in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if lower == role or rel_lower.endswith(f".{role}") or rel_lower == role:
            return role
    return "other"


def _is_router_module(rel_name: str, leaf_name: str) -> bool:
    """Return True for MoE router projections that must remain un-quantized/un-overlaid."""
    lower = leaf_name.lower()
    if lower in _SKIP_ROUTER_EXACT or lower.endswith("_gate"):
        return True
    if "router" in rel_name.lower() and not lower.endswith("_proj"):
        return True
    return False


def _slice_signature(sorted_indices: torch.Tensor | List[int]) -> bytes:
    """Compact deterministic hash digest representing a cube slice S_t."""
    if isinstance(sorted_indices, torch.Tensor):
        if sorted_indices.numel() == 0:
            return b"\x00" * 16
        raw = sorted_indices.to(device="cpu", dtype=torch.int32).numpy().tobytes()
        return hashlib.blake2b(raw, digest_size=16).digest()
    if not sorted_indices:
        return b"\x00" * 16
    raw = torch.tensor(sorted_indices, dtype=torch.int32).numpy().tobytes()
    return hashlib.blake2b(raw, digest_size=16).digest()


def _quantize_delta_cubes_q8(
    delta_w: torch.Tensor,
    r_tile: int,
    c_tile: int,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Tile delta_w [N, K] into edge-padded Q8 delta cubes stored in Host RAM.

    Returns:
        delta_q8:    uint8  [n_cubes, r_tile, c_tile] on CPU
        delta_scale: bf16   [n_cubes, r_tile, c_tile // group_size] on CPU
        delta_zero:  bf16   [n_cubes, r_tile, c_tile // group_size] on CPU
        cube_coords: int32  [n_cubes, 4] -> (r0, r_len, c0, c_len) on CPU
        cube_frob:   float32 [n_cubes] on CPU
        n_rt:        number of row tiles
        n_ct:        number of col tiles
    """
    w_cpu = delta_w.detach().to(device="cpu", dtype=torch.float32).contiguous()
    n, k = int(w_cpu.shape[0]), int(w_cpu.shape[1])
    if c_tile % group_size != 0:
        raise ValueError(f"c_tile ({c_tile}) must be divisible by group_size ({group_size})")
    n_rt = (n + r_tile - 1) // r_tile
    n_ct = (k + c_tile - 1) // c_tile
    n_cubes = n_rt * n_ct
    ng_tile = c_tile // group_size

    if n % r_tile == 0 and k % c_tile == 0 and n_cubes > 0:
        w_blocks = (
            w_cpu.view(n_rt, r_tile, n_ct, c_tile)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(n_cubes, r_tile, ng_tile, group_size)
        )
        cube_frob = w_blocks.reshape(n_cubes, r_tile * c_tile).norm(dim=1).contiguous()
        w_min = w_blocks.amin(dim=-1, keepdim=True)
        w_max = w_blocks.amax(dim=-1, keepdim=True)
        scale_f = ((w_max - w_min) / 255.0).clamp_min(1e-8)
        zero_bf16 = w_min.to(torch.bfloat16)
        scale_bf16 = scale_f.to(torch.bfloat16)
        scale_q = scale_bf16.float().clamp_min(1e-8)
        delta_q8 = (
            torch.round((w_blocks - zero_bf16.float()) / scale_q)
            .clamp(0.0, 255.0)
            .to(torch.uint8)
            .reshape(n_cubes, r_tile, c_tile)
            .contiguous()
        )
        delta_scale = scale_bf16.squeeze(-1).contiguous()
        delta_zero = zero_bf16.squeeze(-1).contiguous()
        r_grid = (torch.arange(n_rt, dtype=torch.int32) * r_tile).unsqueeze(1).expand(n_rt, n_ct)
        c_grid = (torch.arange(n_ct, dtype=torch.int32) * c_tile).unsqueeze(0).expand(n_rt, n_ct)
        cube_coords = (
            torch.stack(
                [
                    r_grid,
                    torch.full_like(r_grid, r_tile),
                    c_grid,
                    torch.full_like(c_grid, c_tile),
                ],
                dim=-1,
            )
            .reshape(n_cubes, 4)
            .contiguous()
        )
        return delta_q8, delta_scale, delta_zero, cube_coords, cube_frob, n_rt, n_ct

    delta_q8 = torch.zeros((n_cubes, r_tile, c_tile), dtype=torch.uint8)
    delta_scale = torch.zeros((n_cubes, r_tile, ng_tile), dtype=torch.bfloat16)
    delta_zero = torch.zeros((n_cubes, r_tile, ng_tile), dtype=torch.bfloat16)
    cube_coords = torch.zeros((n_cubes, 4), dtype=torch.int32)
    cube_frob = torch.zeros((n_cubes,), dtype=torch.float32)

    idx = 0
    for tr in range(n_rt):
        r0 = tr * r_tile
        r1 = min(r0 + r_tile, n)
        r_len = r1 - r0
        for tc in range(n_ct):
            c0 = tc * c_tile
            c1 = min(c0 + c_tile, k)
            c_len = c1 - c0
            sub = w_cpu[r0:r1, c0:c1]
            cube_coords[idx, 0] = r0
            cube_coords[idx, 1] = r_len
            cube_coords[idx, 2] = c0
            cube_coords[idx, 3] = c_len
            cube_frob[idx] = float(torch.linalg.vector_norm(sub))

            # Quantize group-by-group so edge zero-padding never distorts group min/max.
            for g in range(ng_tile):
                gc0 = g * group_size
                if gc0 >= c_len:
                    continue
                valid_c = min(group_size, c_len - gc0)
                w_grp = sub[:, gc0 : gc0 + valid_c]
                w_min = w_grp.min(dim=-1, keepdim=True).values
                w_max = w_grp.max(dim=-1, keepdim=True).values
                scale_f = ((w_max - w_min) / 255.0).clamp_min(1e-8)
                zero_bf16 = w_min.to(torch.bfloat16)
                scale_bf16 = scale_f.to(torch.bfloat16)
                # Ensure scale_bf16 stays strictly positive if w_max > w_min
                scale_q = scale_bf16.float().clamp_min(1e-8)
                q_vals = torch.round((w_grp - zero_bf16.float()) / scale_q).clamp(0.0, 255.0).to(torch.uint8)

                delta_q8[idx, :r_len, gc0 : gc0 + valid_c] = q_vals
                delta_scale[idx, :r_len, g] = scale_bf16.squeeze(-1)
                delta_zero[idx, :r_len, g] = zero_bf16.squeeze(-1)

            idx += 1

    return delta_q8, delta_scale, delta_zero, cube_coords, cube_frob, n_rt, n_ct


class Q1PuzzleContactGraph:
    """Tracks cross-matrix 3D puzzle contacts (`horiz` twins + `diag_corner` 90-deg turns + `lm_head`)."""

    def __init__(self) -> None:
        self._meta: Dict[str, Dict[str, Any]] = {}
        self._layer_role_to_keys: Dict[Tuple[int, str], List[str]] = defaultdict(list)
        self._twin_horiz_hints: Dict[str, torch.Tensor] = {}
        self._corner_col_hints: Dict[str, torch.Tensor] = {}
        self._corner_row_hints: Dict[str, torch.Tensor] = {}
        self._lm_head_prior: Optional[torch.Tensor] = None
        self._lm_head_dynamic: Optional[torch.Tensor] = None

    def register_module(
        self,
        mod_key: str,
        *,
        layer_idx: Optional[int],
        proj_role: str,
        out_features: int,
        in_features: int,
        r_tile: int,
        c_tile: int,
    ) -> None:
        n_rt = (out_features + r_tile - 1) // r_tile
        n_ct = (in_features + c_tile - 1) // c_tile
        self._meta[mod_key] = {
            "layer_idx": layer_idx,
            "proj_role": proj_role,
            "out_features": out_features,
            "in_features": in_features,
            "r_tile": r_tile,
            "c_tile": c_tile,
            "n_rt": n_rt,
            "n_ct": n_ct,
        }
        if layer_idx is not None:
            self._layer_role_to_keys[(layer_idx, proj_role)].append(mod_key)

    def set_lm_head_prior(self, col_energy: torch.Tensor) -> None:
        vec = col_energy.detach().to(device="cpu", dtype=torch.float32).flatten()
        mx = float(vec.max()) if vec.numel() > 0 else 0.0
        self._lm_head_prior = vec / mx if mx > 0.0 else vec

    def update_lm_head_feedback(self, col_sensitivity: torch.Tensor) -> None:
        vec = col_sensitivity.detach().to(device="cpu", dtype=torch.float32).flatten()
        mx = float(vec.max()) if vec.numel() > 0 else 0.0
        self._lm_head_dynamic = vec / mx if mx > 0.0 else vec

    def clear_transient_hints(self) -> None:
        self._twin_horiz_hints.clear()
        self._corner_col_hints.clear()
        self._corner_row_hints.clear()
        self._lm_head_dynamic = None

    def consume_hints(
        self, mod_key: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Consume accumulated twin-horizontal and 90-deg corner hints for `mod_key`."""
        horiz = self._twin_horiz_hints.pop(mod_key, None)
        col_h = self._corner_col_hints.pop(mod_key, None)
        row_h = self._corner_row_hints.pop(mod_key, None)

        meta = self._meta.get(mod_key)
        if meta is not None:
            lm_vec = self._lm_head_dynamic if self._lm_head_dynamic is not None else self._lm_head_prior
            if lm_vec is not None:
                c_tile = int(meta["c_tile"])
                n_ct = int(meta["n_ct"])
                in_f = int(meta["in_features"])
                r_tile = int(meta["r_tile"])
                n_rt = int(meta["n_rt"])
                out_f = int(meta["out_features"])
                role = meta["proj_role"]
                # Residual-stream input projections (q/k/v/gate/up) align directly with hidden dim -> col hint
                if lm_vec.numel() == in_f and role in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
                    tile_scores = self._sparse_top_tiles(self._bin_vec_to_tiles(lm_vec, n_ct, c_tile))
                    col_h = tile_scores if col_h is None else (col_h + 0.35 * tile_scores)
                # Residual-stream output projections (o_proj/down_proj) align with hidden dim -> row hint
                if lm_vec.numel() == out_f and role in ("o_proj", "down_proj"):
                    tile_scores = self._sparse_top_tiles(self._bin_vec_to_tiles(lm_vec, n_rt, r_tile))
                    row_h = tile_scores if row_h is None else (row_h + 0.35 * tile_scores)
        return horiz, col_h, row_h

    @staticmethod
    def _sparse_top_tiles(scores: torch.Tensor) -> torch.Tensor:
        """Retain only top-quartile sensitive tiles so priors remain sparse and directional."""
        n = int(scores.numel())
        if n <= 1:
            return scores
        k = max(1, n // 4)
        top_vals, top_idx = torch.topk(scores, k=k)
        out = torch.zeros_like(scores)
        out[top_idx] = top_vals
        return out

    @staticmethod
    def _bin_vec_to_tiles(vec: torch.Tensor, n_tiles: int, tile_size: int) -> torch.Tensor:
        padded_len = n_tiles * tile_size
        if vec.numel() < padded_len:
            padded = torch.zeros((padded_len,), dtype=torch.float32)
            padded[: vec.numel()] = vec
        else:
            padded = vec[:padded_len]
        return padded.view(n_tiles, tile_size).mean(dim=1)

    def _get_twin_buf(self, dst_key: str) -> Optional[torch.Tensor]:
        meta = self._meta.get(dst_key)
        if meta is None:
            return None
        buf = self._twin_horiz_hints.get(dst_key)
        if buf is None:
            buf = torch.zeros((int(meta["n_rt"]), int(meta["n_ct"])), dtype=torch.float32)
            self._twin_horiz_hints[dst_key] = buf
        return buf

    def _add_twin_hint(self, dst_key: str, r_dst: int, c_dst: int, val: float = 0.5) -> None:
        buf = self._get_twin_buf(dst_key)
        if buf is None:
            return
        if not (0 <= r_dst < buf.shape[0] and 0 <= c_dst < buf.shape[1]):
            return
        buf[r_dst, c_dst] = min(2.0, float(buf[r_dst, c_dst]) + val)

    def _add_corner_col_hint(self, dst_key: str, r_src_idx: int, src_r_tile: int, val: float = 0.6) -> None:
        meta = self._meta.get(dst_key)
        if meta is None:
            return
        n_ct, dst_c_tile = int(meta["n_ct"]), int(meta["c_tile"])
        r0 = r_src_idx * src_r_tile
        r1 = r0 + src_r_tile - 1
        c_start = max(0, r0 // dst_c_tile)
        c_end = min(n_ct - 1, r1 // dst_c_tile)
        if c_start > c_end:
            return
        buf = self._corner_col_hints.get(dst_key)
        if buf is None:
            buf = torch.zeros((n_ct,), dtype=torch.float32)
            self._corner_col_hints[dst_key] = buf
        buf[c_start : c_end + 1] = torch.clamp(buf[c_start : c_end + 1] + val, max=2.0)

    def _add_corner_mask_hint(
        self, dst_key: str, active_rows: torch.Tensor, src_r_tile: int, val: float = 0.6
    ) -> None:
        meta = self._meta.get(dst_key)
        if meta is None:
            return
        r_idx = torch.nonzero(active_rows, as_tuple=False).squeeze(-1)
        if r_idx.numel() == 0:
            return
        n_ct, dst_c_tile = int(meta["n_ct"]), int(meta["c_tile"])
        buf = self._corner_col_hints.get(dst_key)
        if buf is None:
            buf = torch.zeros((n_ct,), dtype=torch.float32)
            self._corner_col_hints[dst_key] = buf
        if src_r_tile <= dst_c_tile and (dst_c_tile % src_r_tile) == 0:
            c_hits = torch.div(r_idx * src_r_tile, dst_c_tile, rounding_mode="floor")
            c_hits = c_hits[c_hits < n_ct]
            if c_hits.numel() > 0:
                buf.add_(torch.bincount(c_hits, minlength=n_ct).to(torch.float32) * val)
                buf.clamp_(max=2.0)
            return
        for r_val in r_idx.tolist():
            r0 = r_val * src_r_tile
            r1 = r0 + src_r_tile - 1
            c_start = max(0, r0 // dst_c_tile)
            c_end = min(n_ct - 1, r1 // dst_c_tile)
            if c_start <= c_end:
                buf[c_start : c_end + 1] = torch.clamp(buf[c_start : c_end + 1] + val, max=2.0)

    def emit_contacts(self, mod_key: str, selected: torch.Tensor | List[Tuple[int, int]]) -> None:
        """Propagate horizontal twin and 90-degree corner contacts from selected cubes."""
        meta = self._meta.get(mod_key)
        if meta is None:
            return
        layer_idx = meta["layer_idx"]
        if layer_idx is None:
            return
        role = meta["proj_role"]
        n_rt = int(meta["n_rt"])
        n_ct = int(meta["n_ct"])
        r_tile = int(meta["r_tile"])

        if isinstance(selected, torch.Tensor):
            if not bool(selected.any().item()):
                return
            sel_f = selected.to(dtype=torch.float32) * 0.5
            active_rows_mask = selected.any(dim=1)

            # 1) Twin horizontal contacts (same layer, matching input column c, GQA-aware row mapping)
            if role in ("q_proj", "k_proj", "v_proj"):
                active_r_list = torch.nonzero(active_rows_mask, as_tuple=False).squeeze(-1).tolist()
                for p_role in ("q_proj", "k_proj", "v_proj"):
                    if p_role == role:
                        continue
                    for dst_key in self._layer_role_to_keys.get((layer_idx, p_role), ()):
                        buf = self._get_twin_buf(dst_key)
                        if buf is None:
                            continue
                        dst_rt, dst_ct = int(buf.shape[0]), int(buf.shape[1])
                        c_lim = min(n_ct, dst_ct)
                        if dst_rt == n_rt:
                            buf[:, :c_lim].add_(sel_f[:, :c_lim]).clamp_(max=2.0)
                        else:
                            for r in active_r_list:
                                r_lo = (r * dst_rt) // max(1, n_rt)
                                r_hi = min(dst_rt - 1, max(r_lo, ((r + 1) * dst_rt - 1) // max(1, n_rt)))
                                if r_lo <= r_hi:
                                    buf[r_lo : r_hi + 1, :c_lim].add_(sel_f[r, :c_lim].unsqueeze(0)).clamp_(max=2.0)
            elif role in ("gate_proj", "up_proj"):
                peer = "up_proj" if role == "gate_proj" else "gate_proj"
                for dst_key in self._layer_role_to_keys.get((layer_idx, peer), ()):
                    buf = self._get_twin_buf(dst_key)
                    if buf is None:
                        continue
                    rr = min(n_rt, int(buf.shape[0]))
                    cc = min(n_ct, int(buf.shape[1]))
                    buf[:rr, :cc].add_(sel_f[:rr, :cc]).clamp_(max=2.0)

            # 2) Structural 90-degree corner contacts: active output row r -> destination input column c' == r
            if role in ("gate_proj", "up_proj"):
                for dst_key in self._layer_role_to_keys.get((layer_idx, "down_proj"), ()):
                    self._add_corner_mask_hint(dst_key, active_rows_mask, r_tile)
            elif role in ("q_proj", "v_proj"):
                for dst_key in self._layer_role_to_keys.get((layer_idx, "o_proj"), ()):
                    self._add_corner_mask_hint(dst_key, active_rows_mask, r_tile)
            elif role in ("o_proj", "down_proj"):
                next_layer = layer_idx + 1
                for n_role in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
                    for dst_key in self._layer_role_to_keys.get((next_layer, n_role), ()):
                        self._add_corner_mask_hint(dst_key, active_rows_mask, r_tile)
            return

        selected_rc = selected
        if not selected_rc:
            return
        active_rows = {r for r, _ in selected_rc}

        # 1) Twin horizontal contacts (same layer, matching input column c, GQA-aware row mapping)
        if role in ("q_proj", "k_proj", "v_proj"):
            peers = ("q_proj", "k_proj", "v_proj")
            for p_role in peers:
                if p_role == role:
                    continue
                for dst_key in self._layer_role_to_keys.get((layer_idx, p_role), ()):
                    dst_meta = self._meta[dst_key]
                    dst_rt = int(dst_meta["n_rt"])
                    dst_ct = int(dst_meta["n_ct"])
                    for r, c in selected_rc:
                        if c >= dst_ct:
                            continue
                        r_lo = (r * dst_rt) // max(1, n_rt)
                        r_hi = max(r_lo, ((r + 1) * dst_rt - 1) // max(1, n_rt))
                        for r_dst in range(r_lo, min(dst_rt, r_hi + 1)):
                            self._add_twin_hint(dst_key, r_dst, c)
        elif role in ("gate_proj", "up_proj"):
            peer = "up_proj" if role == "gate_proj" else "gate_proj"
            for dst_key in self._layer_role_to_keys.get((layer_idx, peer), ()):
                for r, c in selected_rc:
                    self._add_twin_hint(dst_key, r, c)

        # 2) Structural 90-degree corner contacts: active output row r -> destination input column c' == r
        if role in ("gate_proj", "up_proj"):
            for dst_key in self._layer_role_to_keys.get((layer_idx, "down_proj"), ()):
                for r in active_rows:
                    self._add_corner_col_hint(dst_key, r, r_tile)
        elif role in ("q_proj", "v_proj"):
            for dst_key in self._layer_role_to_keys.get((layer_idx, "o_proj"), ()):
                for r in active_rows:
                    self._add_corner_col_hint(dst_key, r, r_tile)
        elif role in ("o_proj", "down_proj"):
            next_layer = layer_idx + 1
            next_roles = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
            for n_role in next_roles:
                for dst_key in self._layer_role_to_keys.get((next_layer, n_role), ()):
                    for r in active_rows:
                        self._add_corner_col_hint(dst_key, r, r_tile)


class Q1PuzzleController:
    """Runtime controller for Q1PuzzleOverlay statistics, dynamic slice_ratio, and cleanup."""

    def __init__(
        self,
        *,
        slice_ratio: float,
        arena: Optional[Q1RoleCycleArena],
        contact_graph: Q1PuzzleContactGraph,
    ) -> None:
        self.slice_ratio = max(0.0, min(1.0, float(slice_ratio)))
        self.arena = arena
        self.contact_graph = contact_graph
        self.modules: Dict[str, "Q1PuzzleLinear"] = {}
        self._hook_handles: List[Any] = []
        self._calls = 0
        self._pct_sum = 0.0
        self._contact_counts: Dict[str, int] = {name: 0 for name in _CONTACT_NAMES}
        self._unique_signatures: set[Tuple[str, bytes]] = set()
        self._last_slice_sets: Dict[str, set[int]] = {}
        self._last_slice_masks: Dict[str, torch.Tensor] = {}
        self._jaccard_sum = 0.0
        self._jaccard_count = 0

    def register_linear(self, mod_key: str, linear: "Q1PuzzleLinear") -> None:
        self.modules[mod_key] = linear

    def register_hook_handle(self, handle: Any) -> None:
        self._hook_handles.append(handle)

    def set_slice_ratio(self, ratio: float) -> None:
        """Dynamically change slice_ratio in [0.0, 1.0] without repacking matrices and reset stats."""
        clamped = max(0.0, min(1.0, float(ratio)))
        self.slice_ratio = clamped
        for mod in self.modules.values():
            mod.slice_ratio = clamped
        self.contact_graph.clear_transient_hints()
        self.reset_stats()

    def reset_stats(self) -> None:
        """Reset accumulated forward statistics without clearing installed modules."""
        self._calls = 0
        self._pct_sum = 0.0
        for name in _CONTACT_NAMES:
            self._contact_counts[name] = 0
        self._unique_signatures.clear()
        self._last_slice_sets.clear()
        self._last_slice_masks.clear()
        self._jaccard_sum = 0.0
        self._jaccard_count = 0

    def record_forward(
        self,
        mod_key: str,
        total_cubes: int,
        selected_indices: torch.Tensor | List[int],
        contact_delta: Dict[str, int],
        sel_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self._calls += 1
        n_sel = int(selected_indices.numel()) if isinstance(selected_indices, torch.Tensor) else len(selected_indices)
        pct = (100.0 * n_sel / float(total_cubes)) if total_cubes > 0 else 0.0
        self._pct_sum += pct
        for name in _CONTACT_NAMES:
            self._contact_counts[name] += int(contact_delta.get(name, 0))

        if sel_mask is not None and isinstance(selected_indices, torch.Tensor):
            sig = _slice_signature(selected_indices)
            self._unique_signatures.add((mod_key, sig))
            prev_mask = self._last_slice_masks.get(mod_key)
            if prev_mask is not None and prev_mask.shape == sel_mask.shape:
                union_sz = int((sel_mask | prev_mask).sum().item())
                inter_sz = int((sel_mask & prev_mask).sum().item()) if union_sz > 0 else 0
                jacc = (inter_sz / float(union_sz)) if union_sz > 0 else 1.0
                self._jaccard_sum += jacc
                self._jaccard_count += 1
            self._last_slice_masks[mod_key] = sel_mask
            return

        idx_list = selected_indices.tolist() if isinstance(selected_indices, torch.Tensor) else selected_indices
        sorted_idx = sorted(idx_list)
        sig = _slice_signature(sorted_idx)
        self._unique_signatures.add((mod_key, sig))

        curr_set = set(sorted_idx)
        prev_set = self._last_slice_sets.get(mod_key)
        if prev_set is not None:
            union_sz = len(curr_set | prev_set)
            jacc = (len(curr_set & prev_set) / float(union_sz)) if union_sz > 0 else 1.0
            self._jaccard_sum += jacc
            self._jaccard_count += 1
        self._last_slice_sets[mod_key] = curr_set

    def stats(self) -> Dict[str, Any]:
        avg_pct = (self._pct_sum / float(self._calls)) if self._calls > 0 else 0.0
        avg_jaccard = (self._jaccard_sum / float(self._jaccard_count)) if self._jaccard_count > 0 else 1.0
        return {
            "calls": int(self._calls),
            "slice_ratio": float(self.slice_ratio),
            "avg_cubes_selected_pct": float(avg_pct),
            "unique_slice_signatures": int(len(self._unique_signatures)),
            "avg_slice_jaccard": float(avg_jaccard),
            "contact_counts": {k: int(self._contact_counts[k]) for k in _CONTACT_NAMES},
            "arena": self.arena.stats() if self.arena is not None else None,
        }

    def close(self) -> None:
        """Detach registered hooks and clear transient hint buffers."""
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()
        self.contact_graph.clear_transient_hints()


class Q1PuzzleLinear(nn.Module):
    """Linear layer executing Ideal Q1 (P=1) base + dynamic streamed Q8 delta-cube slice S_t."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        slice_ratio: float = 0.35,
        r_tile: int = 64,
        c_tile: int = 128,
        group_size: int = 64,
        arena: Optional[Q1RoleCycleArena] = None,
        controller: Optional[Q1PuzzleController] = None,
        mod_key: str = "linear",
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.slice_ratio = max(0.0, min(1.0, float(slice_ratio)))
        self.r_tile = int(r_tile)
        self.c_tile = int(c_tile)
        self.group_size = int(group_size)
        self.mod_key = str(mod_key)
        self.controller = controller

        target_dev = torch.device(device) if device is not None else weight.device
        if target_dev.type == "meta":
            target_dev = torch.device("cpu")

        # Keep bias on Q1PuzzleLinear so base IdealQ1Linear computes raw linear projection once.
        w_dev = weight.detach().to(device=target_dev)
        self.base_q1 = IdealQ1Linear(
            self.in_features,
            self.out_features,
            planes=1,
            group_size=self.group_size,
            bias=None,
            weight=w_dev,
            device=target_dev,
        )
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone().to(device=target_dev, dtype=torch.bfloat16))
        else:
            self.bias = None

        w_q1 = unpack_ideal_q1(
            self.base_q1.packed(),
            group_size=self.group_size,
            dtype=torch.float32,
        )[: self.out_features, : self.in_features]
        delta_w = w_dev.to(torch.float32) - w_q1.to(device=w_dev.device, dtype=torch.float32)

        (
            delta_q8,
            delta_scale,
            delta_zero,
            cube_coords,
            cube_frob,
            n_rt,
            n_ct,
        ) = _quantize_delta_cubes_q8(delta_w, self.r_tile, self.c_tile, self.group_size)

        self.n_rt = n_rt
        self.n_ct = n_ct
        self.n_cubes = n_rt * n_ct
        # Keep Host RAM buffers cleanly outside CUDA parameters/buffers
        self.delta_q8_host = delta_q8.contiguous()
        self.delta_scale_host = delta_scale.contiguous()
        self.delta_zero_host = delta_zero.contiguous()
        self.cube_coords_host = cube_coords.contiguous()
        self.cube_frob_host = cube_frob.contiguous()
        self.cube_frob_2d = self.cube_frob_host.view(n_rt, n_ct).contiguous()
        self._ones_2d = torch.ones((n_rt, n_ct), dtype=torch.float32)
        self._all_indices_host = torch.arange(self.n_cubes, dtype=torch.long)
        self._empty_indices_host = torch.empty((0,), dtype=torch.long)

        ng_tile = self.c_tile // self.group_size
        self.bytes_per_cube = (
            self.r_tile * self.c_tile * 1
            + self.r_tile * ng_tile * 2 * 2
            + 4 * 4
        )
        if arena is None:
            default_budget = max(self.bytes_per_cube * 4, self.bytes_per_cube * max(1, self.n_cubes // 2))
            self.arena = Q1RoleCycleArena(default_budget, target_dev)
        else:
            self.arena = arena

    def _fused_xy_profiles(self, flat_x: torch.Tensor, y_base: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-tile L1 activation profiles [n_ct] and [n_rt] in one D2H sync."""
        rx = flat_x.float().abs().mean(dim=0)
        pad_x = self.n_ct * self.c_tile
        if rx.numel() < pad_x:
            rx = torch.nn.functional.pad(rx, (0, pad_x - rx.numel()))
        elif rx.numel() > pad_x:
            rx = rx[:pad_x]
        ex = rx.view(self.n_ct, self.c_tile).mean(dim=1)

        ry = y_base.float().abs().mean(dim=0)
        pad_y = self.n_rt * self.r_tile
        if ry.numel() < pad_y:
            ry = torch.nn.functional.pad(ry, (0, pad_y - ry.numel()))
        elif ry.numel() > pad_y:
            ry = ry[:pad_y]
        ey = ry.view(self.n_rt, self.r_tile).mean(dim=1)

        both = torch.cat((ex, ey), dim=0).cpu()
        e_x = both[: self.n_ct]
        e_y = both[self.n_ct :]
        mx_x = float(e_x.max().item()) if e_x.numel() > 0 else 0.0
        if mx_x > 0.0:
            e_x = e_x / mx_x
        mx_y = float(e_y.max().item()) if e_y.numel() > 0 else 0.0
        if mx_y > 0.0:
            e_y = e_y / mx_y
        return e_x, e_y

    def _select_slice(
        self,
        flat_x: torch.Tensor,
        y_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, int]]:
        """Build unique connected slice S_t via vectorized 40% seed + 60% {horiz, vert, diag_corner} wavefront."""
        n_cubes = self.n_cubes
        counts = {name: 0 for name in _CONTACT_NAMES}
        ratio = self.slice_ratio
        if ratio <= 0.0 or n_cubes == 0:
            return self._empty_indices_host, None, counts

        if ratio >= 1.0:
            counts[_CONTACT_SEED] = n_cubes
            return self._all_indices_host, None, counts

        n_rt, n_ct = self.n_rt, self.n_ct
        target_k = max(1, min(n_cubes, int(round(ratio * n_cubes))))
        e_x, e_y = self._fused_xy_profiles(flat_x, y_base)

        horiz_hint_t: Optional[torch.Tensor] = None
        col_hint_t: Optional[torch.Tensor] = None
        row_hint_t: Optional[torch.Tensor] = None
        if self.controller is not None:
            horiz_hint_t, col_hint_t, row_hint_t = self.controller.contact_graph.consume_hints(self.mod_key)

        base_scores = (e_y + 0.05).unsqueeze(1) * (e_x + 0.05).unsqueeze(0) * self.cube_frob_2d
        boost = self._ones_2d.clone()
        if horiz_hint_t is not None:
            boost.add_(horiz_hint_t)
        if col_hint_t is not None:
            boost.add_(col_hint_t.unsqueeze(0))
        if row_hint_t is not None:
            boost.add_(row_hint_t.unsqueeze(1))
        seed_scores = base_scores * boost

        # 40% seed budget (at least 1 seed, leaving up to 60% for contact expansion when target_k > 1)
        k_seed = 1 if target_k == 1 else max(1, min(target_k - 1, int(round(0.40 * target_k))))
        seed_flat = torch.topk(seed_scores.view(-1), k=k_seed).indices
        sel_mask = torch.zeros((n_rt, n_ct), dtype=torch.bool)
        sel_mask.view(-1)[seed_flat] = True
        counts[_CONTACT_SEED] = k_seed
        selected_count = k_seed

        # Inject sparse top cross-matrix twin horizontal and 90-degree corner hints into hop-0 frontier
        ext_cap = max(1, target_k // 4)
        ext_h_mask: Optional[torch.Tensor] = None
        if selected_count < target_k and horiz_hint_t is not None:
            m_h = (horiz_hint_t > 0.25) & (~sel_mask)
            idx_h = torch.nonzero(m_h.view(-1), as_tuple=False).squeeze(-1)
            if idx_h.numel() > 0:
                if int(idx_h.numel()) > ext_cap:
                    sc_h = base_scores.view(-1)[idx_h] * (1.0 + 0.15 * horiz_hint_t.view(-1)[idx_h])
                    idx_h = idx_h[torch.topk(sc_h, k=ext_cap).indices]
                ext_h_mask = torch.zeros((n_rt, n_ct), dtype=torch.bool)
                ext_h_mask.view(-1)[idx_h] = True

        ch_2d: Optional[torch.Tensor] = None
        rh_2d: Optional[torch.Tensor] = None
        ext_d_mask: Optional[torch.Tensor] = None
        if col_hint_t is not None or row_hint_t is not None:
            ch_2d = col_hint_t.unsqueeze(0).expand(n_rt, n_ct) if col_hint_t is not None else torch.zeros_like(base_scores)
            rh_2d = row_hint_t.unsqueeze(1).expand(n_rt, n_ct) if row_hint_t is not None else torch.zeros_like(base_scores)
            if selected_count < target_k:
                m_d = ((ch_2d > 0.25) | (rh_2d > 0.25)) & (~sel_mask)
                idx_d = torch.nonzero(m_d.view(-1), as_tuple=False).squeeze(-1)
                if idx_d.numel() > 0:
                    if int(idx_d.numel()) > ext_cap:
                        sc_d = base_scores.view(-1)[idx_d] * (1.0 + 0.10 * (ch_2d + rh_2d).view(-1)[idx_d])
                        idx_d = idx_d[torch.topk(sc_d, k=ext_cap).indices]
                    ext_d_mask = torch.zeros((n_rt, n_ct), dtype=torch.bool)
                    ext_d_mask.view(-1)[idx_d] = True

        sh_raw = base_scores * (1.0 + 0.15 * horiz_hint_t) if horiz_hint_t is not None else base_scores
        sv_raw = base_scores * (1.0 + 0.15 * row_hint_t.unsqueeze(1)) if row_hint_t is not None else base_scores
        sd_raw = (
            base_scores * (1.0 + 0.10 * (ch_2d + rh_2d))
            if (ch_2d is not None and rh_2d is not None)
            else base_scores
        )
        neg_one = torch.full_like(base_scores, -1.0)

        hop = 0
        while selected_count < target_k:
            not_sel = ~sel_mask
            h_nbr = torch.zeros_like(sel_mask)
            if n_ct > 1:
                h_nbr[:, :-1] |= sel_mask[:, 1:]
                h_nbr[:, 1:] |= sel_mask[:, :-1]
            v_nbr = torch.zeros_like(sel_mask)
            if n_rt > 1:
                v_nbr[:-1, :] |= sel_mask[1:, :]
                v_nbr[1:, :] |= sel_mask[:-1, :]
            d_nbr = torch.zeros_like(sel_mask)
            if n_rt > 1 and n_ct > 1:
                d_nbr[:-1, :-1] |= sel_mask[1:, 1:]
                d_nbr[:-1, 1:] |= sel_mask[1:, :-1]
                d_nbr[1:, :-1] |= sel_mask[:-1, 1:]
                d_nbr[1:, 1:] |= sel_mask[:-1, :-1]
            if hop == 0:
                if ext_h_mask is not None:
                    h_nbr |= ext_h_mask
                if ext_d_mask is not None:
                    d_nbr |= ext_d_mask

            sh = torch.where(h_nbr & not_sel, sh_raw, neg_one)
            sv = torch.where(v_nbr & not_sel, sv_raw, neg_one)
            sd = torch.where(d_nbr & not_sel, sd_raw, neg_one)
            max_vals, best_kind = torch.stack((sh, sv, sd), dim=0).max(dim=0)

            avail_idx = torch.nonzero((max_vals >= 0.0).view(-1), as_tuple=False).squeeze(-1)
            if avail_idx.numel() == 0:
                break
            rem = target_k - selected_count
            if int(avail_idx.numel()) > rem:
                top_pos = torch.topk(max_vals.view(-1)[avail_idx], k=rem).indices
                chosen_idx = avail_idx[top_pos]
            else:
                chosen_idx = avail_idx

            sel_mask.view(-1)[chosen_idx] = True
            kinds = best_kind.view(-1)[chosen_idx]
            counts[_CONTACT_HORIZ] += int((kinds == 0).sum().item())
            counts[_CONTACT_VERT] += int((kinds == 1).sum().item())
            counts[_CONTACT_DIAG_CORNER] += int((kinds == 2).sum().item())
            selected_count += int(chosen_idx.numel())
            hop += 1

        if selected_count < target_k:
            rem = target_k - selected_count
            rem_scores = seed_scores.masked_fill(sel_mask, -1.0)
            extra_idx = torch.topk(rem_scores.view(-1), k=rem).indices
            sel_mask.view(-1)[extra_idx] = True
            counts[_CONTACT_SEED] += rem

        if self.controller is not None:
            self.controller.contact_graph.emit_contacts(self.mod_key, sel_mask)

        selected_idx = torch.nonzero(sel_mask.view(-1), as_tuple=False).squeeze(-1)
        return selected_idx, sel_mask, counts

    def _compute_chunk(
        self,
        flat_x: torch.Tensor,
        payload: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        y_acc_f32: torch.Tensor,
    ) -> None:
        """Accumulate a staged batch of Q8 delta cubes into y_acc_f32."""
        dq, ds, dz, coords = payload
        if flat_x.device.type == "cuda" and flat_x.dtype == torch.bfloat16:
            from src.colibri_q1_cube_nvrtc import launch_puzzle_overlay_delta_cubes

            launch_puzzle_overlay_delta_cubes(
                flat_x,
                dq,
                ds,
                dz,
                coords,
                y_acc_f32,
                group_size=self.group_size,
            )
            self.arena.nvrtc_batch_launches += 1
            return

        # Exact CPU / general dtype accumulation matching kernel #11 group affine math
        x_f = flat_x.to(torch.float32)
        m_sel = int(dq.shape[0])
        ng_tile = self.c_tile // self.group_size
        coords_list = coords.cpu().tolist()
        dq_f = dq.to(dtype=torch.float32)
        ds_f = ds.to(dtype=torch.float32)
        dz_f = dz.to(dtype=torch.float32)
        for s_idx in range(m_sel):
            r0, r_len, c0, c_len = coords_list[s_idx]
            if r_len <= 0 or c_len <= 0:
                continue
            w_tile = (
                dq_f[s_idx, :r_len].view(r_len, ng_tile, self.group_size) * ds_f[s_idx, :r_len].unsqueeze(-1)
                + dz_f[s_idx, :r_len].unsqueeze(-1)
            ).reshape(r_len, self.c_tile)[:, :c_len]
            y_acc_f32[:, r0 : r0 + r_len] += x_f[:, c0 : c0 + c_len] @ w_tile.t()

    def _apply_delta_cubes(
        self,
        flat_x: torch.Tensor,
        selected_indices: torch.Tensor | List[int],
        y_acc_f32: torch.Tensor,
    ) -> None:
        """Stream selected delta cubes strictly via A -> B -> C(0) -> A."""
        idx_all = selected_indices if isinstance(selected_indices, torch.Tensor) else torch.tensor(selected_indices, dtype=torch.long)
        m_total = int(idx_all.numel())
        if m_total == 0:
            return
        max_per_slot = max(1, self.arena.slot_budget_bytes // self.bytes_per_cube)

        if m_total == self.n_cubes and m_total <= max_per_slot and int(idx_all[0].item()) == 0 and int(idx_all[-1].item()) == self.n_cubes - 1:
            chunks = [((self.delta_q8_host, self.delta_scale_host, self.delta_zero_host, self.cube_coords_host), self.n_cubes * self.bytes_per_cube)]
        else:
            chunks = []
            for start in range(0, m_total, max_per_slot):
                idx_t = idx_all[start : start + max_per_slot]
                dq = self.delta_q8_host.index_select(0, idx_t).contiguous()
                ds = self.delta_scale_host.index_select(0, idx_t).contiguous()
                dz = self.delta_zero_host.index_select(0, idx_t).contiguous()
                coords = self.cube_coords_host.index_select(0, idx_t).contiguous()
                nbytes = int(idx_t.numel()) * self.bytes_per_cube
                chunks.append(((dq, ds, dz, coords), nbytes))

        # Strict physical cycle: stage chunk_0 into B -> rotate (B->A, C->A) -> ping-pong B->A->B->C(0)
        self.arena.stage_payload_into_B(chunks[0][0], chunks[0][1])
        self.arena.rotate()

        for ci in range(len(chunks)):
            active = self.arena.active_payload()
            if ci + 1 < len(chunks):
                self.arena.stage_payload_into_B(chunks[ci + 1][0], chunks[ci + 1][1])
            self._compute_chunk(flat_x, active, y_acc_f32)
            self.arena.release_active_A()
            self.arena.rotate()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        flat_x = x.reshape(-1, self.in_features)
        m = int(flat_x.shape[0])
        if m == 0:
            return torch.empty((*orig_shape[:-1], self.out_features), dtype=x.dtype, device=x.device)

        if x.device.type == "cuda" and flat_x.dtype != torch.bfloat16:
            flat_x_compute = flat_x.to(torch.bfloat16).contiguous()
        else:
            flat_x_compute = flat_x.contiguous()

        # Always compute full Ideal Q1 (P=1) base so cold cubes stay active and non-zero.
        y_base = self.base_q1(flat_x_compute)

        selected_idx, sel_mask, counts = self._select_slice(flat_x_compute, y_base)
        if self.controller is not None:
            self.controller.record_forward(self.mod_key, self.n_cubes, selected_idx, counts, sel_mask=sel_mask)

        if selected_idx.numel() == 0:
            if self.bias is not None:
                y_out = y_base.to(torch.float32) + self.bias.to(device=y_base.device, dtype=torch.float32).unsqueeze(0)
                return y_out.to(x.dtype).view(*orig_shape[:-1], self.out_features)
            return y_base.to(x.dtype).view(*orig_shape[:-1], self.out_features)

        y_acc_f32 = y_base.to(dtype=torch.float32).contiguous()
        self._apply_delta_cubes(flat_x_compute, selected_idx, y_acc_f32)
        if self.bias is not None:
            y_acc_f32 += self.bias.to(device=y_acc_f32.device, dtype=torch.float32).unsqueeze(0)
        return y_acc_f32.to(x.dtype).view(*orig_shape[:-1], self.out_features)


def _find_decoder_layers(model: nn.Module) -> List[Tuple[int, nn.Module]]:
    """Locate decoder layer sequence across Llama/SmolLM/Qwen/Granite/GPT-style modules."""
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    ]
    for seq in candidates:
        if seq is not None and isinstance(seq, (nn.ModuleList, list, tuple)):
            return list(enumerate(seq))
    return []


def _lookup_provider_tensor(
    layer_weights: Dict[str, torch.Tensor],
    rel_name: str,
    leaf_name: str,
    suffix: str,
) -> Optional[torch.Tensor]:
    candidates = (
        f"{rel_name}.{suffix}",
        f"{leaf_name}.{suffix}",
    )
    for key in candidates:
        if key in layer_weights:
            return layer_weights[key]
    if suffix == "weight":
        for key in (rel_name, leaf_name):
            if key in layer_weights:
                return layer_weights[key]
    tail_exact = f".{rel_name}.{suffix}"
    for key, val in layer_weights.items():
        if key.endswith(tail_exact):
            return val
    return None


def _extract_module_weight_and_bias(
    mod: nn.Module,
    *,
    rel_name: str,
    leaf_name: str,
    group_size: int,
    layer_weights: Optional[Dict[str, torch.Tensor]],
) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor], int, int]]:
    """Extract materialized W_main [out, in], optional bias, in_features, out_features."""
    if isinstance(mod, Q1PuzzleLinear):
        return None
    if isinstance(mod, IdealQ1Linear):
        w = unpack_ideal_q1(mod.packed(), group_size=mod.group_size, dtype=torch.bfloat16)[
            : mod.out_features, : mod.in_features
        ]
        bias = mod.bias.detach() if getattr(mod, "bias", None) is not None else None
        return w, bias, mod.in_features, mod.out_features

    in_f = getattr(mod, "in_features", None)
    out_f = getattr(mod, "out_features", None)
    if in_f is None or out_f is None:
        return None
    in_f, out_f = int(in_f), int(out_f)

    # 1) Direct nn.Linear weight
    w_attr = getattr(mod, "weight", None)
    if isinstance(w_attr, torch.Tensor) and w_attr.device.type != "meta":
        b_attr = getattr(mod, "bias", None)
        bias = b_attr.detach() if isinstance(b_attr, torch.Tensor) and b_attr.device.type != "meta" else None
        return w_attr.detach(), bias, in_f, out_f

    # 2) PackedLinear (Q1/Q2/Q4/Q6/Q8)
    if hasattr(mod, "packed") and callable(mod.packed):
        packed = mod.packed()
        if packed and all(isinstance(v, torch.Tensor) and v.device.type != "meta" for v in packed.values()):
            from src.quantized_projection import unpack_weight

            w = unpack_weight(packed, group_size=int(getattr(mod, "group_size", group_size)), dtype=torch.bfloat16)
            b_attr = getattr(mod, "bias", None)
            bias = b_attr.detach() if isinstance(b_attr, torch.Tensor) and b_attr.device.type != "meta" else None
            return w, bias, in_f, out_f

    # 3) ColibriQ4Linear
    if hasattr(mod, "qweight") and hasattr(mod, "scales") and hasattr(mod, "zeros"):
        qw, sc, zr = mod.qweight, mod.scales, mod.zeros
        if (
            isinstance(qw, torch.Tensor)
            and qw.device.type != "meta"
            and sc.device.type != "meta"
            and zr.device.type != "meta"
        ):
            from src.quantization_q4 import dequantize_q4_tensor

            w = dequantize_q4_tensor(qw, sc, zr, group_size=int(getattr(mod, "group_size", group_size)), dtype=torch.bfloat16)
            b_attr = getattr(mod, "bias", None)
            bias = b_attr.detach() if isinstance(b_attr, torch.Tensor) and b_attr.device.type != "meta" else None
            return w, bias, in_f, out_f

    # 4) Q1StreamedLinear (defense-in-depth if installed prior to puzzle overlay)
    if hasattr(mod, "q1_mat"):
        from src.q1_cube_engine import unpack_q1_cubes

        w = unpack_q1_cubes(
            mod.q1_mat,
            dtype=torch.bfloat16,
            active_planes=getattr(mod, "active_planes", None),
            midpoint_correct=True,
        )[:out_f, :in_f]
        b_attr = getattr(mod, "bias", None)
        bias = b_attr.detach() if isinstance(b_attr, torch.Tensor) and b_attr.device.type != "meta" else None
        return w, bias, in_f, out_f

    # 5) Fallback to layer_weight_provider (for meta modules)
    if layer_weights is not None:
        w = _lookup_provider_tensor(layer_weights, rel_name, leaf_name, "weight")
        if w is not None and w.device.type != "meta":
            bias = _lookup_provider_tensor(layer_weights, rel_name, leaf_name, "bias")
            if bias is not None and bias.device.type == "meta":
                bias = None
            return w.detach(), (bias.detach() if bias is not None else None), in_f, out_f

    return None


def _collect_replaceable_children(
    parent: nn.Module,
    prefix: str = "",
) -> List[Tuple[nn.Module, str, str, nn.Module]]:
    """Collect leaf linear-like modules (parent, attr_name, rel_path, child) skipping MoE routers."""
    items: List[Tuple[nn.Module, str, str, nn.Module]] = []
    for attr_name, child in list(parent.named_children()):
        rel_path = f"{prefix}.{attr_name}" if prefix else attr_name
        if _is_router_module(rel_path, attr_name):
            continue
        is_linear_like = (
            isinstance(child, (nn.Linear, IdealQ1Linear))
            or (hasattr(child, "packed") and callable(child.packed) and hasattr(child, "in_features"))
            or (hasattr(child, "qweight") and hasattr(child, "scales") and hasattr(child, "zeros"))
            or (hasattr(child, "q1_mat") and hasattr(child, "in_features") and hasattr(child, "out_features"))
        )
        if is_linear_like and not isinstance(child, Q1PuzzleLinear):
            items.append((parent, attr_name, rel_path, child))
        else:
            items.extend(_collect_replaceable_children(child, rel_path))
    return items


def _attach_lm_head_feedback(
    model: nn.Module,
    controller: Q1PuzzleController,
) -> None:
    """Extract static lm_head prior (without replacing lm_head) and install top-k backward hint hook."""
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None and hasattr(model, "get_output_embeddings"):
        lm_head = model.get_output_embeddings()
    if lm_head is None:
        return

    w_lm = getattr(lm_head, "weight", None)
    has_tensor_w = isinstance(w_lm, torch.Tensor) and w_lm.device.type != "meta"
    if not has_tensor_w and hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        emb_w = getattr(emb, "weight", None)
        out_f = int(getattr(lm_head, "vocab_size", getattr(lm_head, "out_features", 0)) or 0)
        in_f = int(getattr(lm_head, "in_features", 0) or 0)
        if (
            isinstance(emb_w, torch.Tensor)
            and emb_w.device.type != "meta"
            and emb_w.ndim == 2
            and (out_f == 0 or (emb_w.shape[0] == out_f and emb_w.shape[1] == in_f))
        ):
            w_lm = emb_w
            has_tensor_w = True

    if has_tensor_w and w_lm is not None:
        with torch.no_grad():
            n_rows, n_cols = int(w_lm.shape[0]), int(w_lm.shape[1])
            sq_sum = torch.zeros((n_cols,), dtype=torch.float32, device=w_lm.device)
            chunk = 4096
            for r0 in range(0, n_rows, chunk):
                part = w_lm[r0 : min(n_rows, r0 + chunk)].detach().float()
                sq_sum.add_((part * part).sum(dim=0))
            col_e = (sq_sum / max(1, n_rows)).sqrt()
            controller.contact_graph.set_lm_head_prior(col_e)

        def _post_hook(_mod: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
            logits = output[0] if isinstance(output, tuple) else output
            if not isinstance(logits, torch.Tensor) or logits.numel() == 0:
                return
            with torch.no_grad():
                last_logits = logits.reshape(-1, logits.shape[-1])[-1]
                k = min(8, int(last_logits.shape[0]))
                topk_idx = torch.topk(last_logits.float(), k=k).indices.to(device=w_lm.device)
                top_rows = w_lm.index_select(0, topk_idx).float()
                probs = torch.softmax(last_logits.index_select(0, topk_idx).float(), dim=0)
                sens = (probs.unsqueeze(-1) * top_rows.abs()).sum(dim=0)
                controller.contact_graph.update_lm_head_feedback(sens)

        controller.register_hook_handle(lm_head.register_forward_hook(_post_hook))
    elif hasattr(lm_head, "q1_mat") and getattr(lm_head.q1_mat, "spatial_tiles", None):
        with torch.no_grad():
            q1_mat = lm_head.q1_mat
            in_f = int(q1_mat.shape[1])
            gs = int(q1_mat.group_size)
            c_tile = int(q1_mat.c_tile)
            col_acc = torch.zeros((in_f,), dtype=torch.float32)
            col_cnt = torch.zeros((in_f,), dtype=torch.float32)
            for (_tr, tc), meta in q1_mat.spatial_tiles.items():
                c0 = int(meta.c0)
                c1 = int(meta.c1)
                c_len = c1 - c0
                if c_len <= 0:
                    continue
                g_scales = meta.qscale.float().mean(dim=0).repeat_interleave(gs)[:c_len].cpu()
                col_acc[c0:c1].add_(g_scales)
                col_cnt[c0:c1].add_(1.0)
            col_e = col_acc / col_cnt.clamp_min(1.0)
            controller.contact_graph.set_lm_head_prior(col_e)


def install_q1_puzzle_overlay(
    model: nn.Module,
    *,
    slice_ratio: float = 0.35,
    r_tile: int = 64,
    c_tile: int = 128,
    group_size: int = 64,
    slot_budget_bytes: int | None = None,
    arena: Q1RoleCycleArena | None = None,
    device: torch.device | str | None = None,
    layer_weight_provider: Callable[[int], dict[str, torch.Tensor]] | None = None,
) -> Q1PuzzleController:
    """Replace decoder-layer projections (excluding routers and lm_head) with Q1PuzzleLinear."""
    if device is not None:
        target_dev = torch.device(device)
    elif arena is not None:
        target_dev = arena.device
    else:
        target_dev = torch.device("cpu")
        for t in list(model.parameters()) + list(model.buffers()):
            if t.device.type != "meta":
                target_dev = t.device
                break

    layers = _find_decoder_layers(model)
    ng_tile = c_tile // group_size
    bytes_per_cube = r_tile * c_tile + r_tile * ng_tile * 4 + 16

    # Pass 1: collect lightweight structural metadata so arena budget can be sized without holding weights.
    meta_plan: List[Tuple[int, nn.Module, str, str, nn.Module, int, int]] = []
    max_q8_matrix_bytes = 0
    for layer_idx, layer_mod in layers:
        candidates = _collect_replaceable_children(layer_mod)
        for parent, attr_name, rel_path, child in candidates:
            in_f = int(getattr(child, "in_features", 0) or 0)
            out_f = int(getattr(child, "out_features", 0) or 0)
            if in_f <= 0 or out_f <= 0 or (in_f % group_size) != 0:
                continue
            n_rt = (out_f + r_tile - 1) // r_tile
            n_ct = (in_f + c_tile - 1) // c_tile
            max_q8_matrix_bytes = max(max_q8_matrix_bytes, n_rt * n_ct * bytes_per_cube)
            meta_plan.append((layer_idx, parent, attr_name, rel_path, child, in_f, out_f))

    if arena is None:
        if slot_budget_bytes is not None:
            resolved_budget = max(bytes_per_cube, int(slot_budget_bytes))
        else:
            half_mat = max_q8_matrix_bytes // 2 if max_q8_matrix_bytes > bytes_per_cube else bytes_per_cube
            resolved_budget = max(bytes_per_cube, min(4 * 1024 * 1024, half_mat))
        arena = Q1RoleCycleArena(resolved_budget, target_dev)

    graph = Q1PuzzleContactGraph()
    controller = Q1PuzzleController(
        slice_ratio=slice_ratio,
        arena=arena,
        contact_graph=graph,
    )

    # Pass 2: convert projection-by-projection, caching provider dict only within one decoder layer.
    current_layer_idx: Optional[int] = None
    layer_weights: Optional[Dict[str, torch.Tensor]] = None
    for layer_idx, parent, attr_name, rel_path, child, in_f, out_f in meta_plan:
        if layer_idx != current_layer_idx:
            current_layer_idx = layer_idx
            layer_weights = None

        extracted = _extract_module_weight_and_bias(
            child,
            rel_name=rel_path,
            leaf_name=attr_name,
            group_size=group_size,
            layer_weights=None,
        )
        if extracted is None and layer_weight_provider is not None:
            if layer_weights is None:
                layer_weights = layer_weight_provider(layer_idx)
            extracted = _extract_module_weight_and_bias(
                child,
                rel_name=rel_path,
                leaf_name=attr_name,
                group_size=group_size,
                layer_weights=layer_weights,
            )
        if extracted is None:
            continue

        w_main, bias, in_f, out_f = extracted
        mod_key = f"layer.{layer_idx}.{rel_path}"
        proj_role = _infer_proj_role(rel_path, attr_name)
        graph.register_module(
            mod_key,
            layer_idx=layer_idx,
            proj_role=proj_role,
            out_features=out_f,
            in_features=in_f,
            r_tile=r_tile,
            c_tile=c_tile,
        )
        puzzle_lin = Q1PuzzleLinear(
            in_f,
            out_f,
            weight=w_main,
            bias=bias,
            slice_ratio=slice_ratio,
            r_tile=r_tile,
            c_tile=c_tile,
            group_size=group_size,
            arena=arena,
            controller=controller,
            mod_key=mod_key,
            device=target_dev,
        )
        setattr(parent, attr_name, puzzle_lin)
        controller.register_linear(mod_key, puzzle_lin)
        del w_main, bias, extracted
    layer_weights = None

    _attach_lm_head_feedback(model, controller)
    return controller
