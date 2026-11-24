"""Spatial-Bitwise Q1 Cube Engine with A -> B -> C(0) -> A Sub-Matrix Streaming.

Implements:
- 2D spatial tiling (R_tile x C_tile) along rows and columns.
- Positional bit-plane packing B_k in {0, 1} (32 bits per uint32/int32 word)
  with FP64-calibrated BF16 group scale/zero (exact Q4/Q6/Q8 equivalence).
- Unbiased truncation identity z_eff(P) = z + s * (2^(K-P) - 1) / 2, making
  truncated positional bit-planes mathematically identical to Sign Geometric Q1.
- Scale comparison utilities (positional_z_eff, sign_geom, sign_arith, sign_residual).
- 1-bit gate_proj radar depth selection and 1-launch NVRTC / CPU execution.
- Bounded-slot sub-matrix streaming through strict A -> B -> C(0) -> A cycle
  with FP32 partial-sum accumulation prior to nonlinearities on both CPU and CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class Q1Cube:
    """One 2D spatial bit-plane cube (tile_r, tile_c, plane_idx)."""

    tile_r: int
    tile_c: int
    r0: int
    r1: int
    c0: int
    c1: int
    plane_idx: int
    weight_pow2: float
    bits_i32: torch.Tensor  # [r_len, c_len // 32] int32 (bit-exact uint32 layout)

    @property
    def nbytes(self) -> int:
        return self.bits_i32.numel() * self.bits_i32.element_size()


@dataclass
class Q1SpatialTileMeta:
    """Group affine scale/zero metadata shared by all bit-planes of a 2D tile."""

    tile_r: int
    tile_c: int
    r0: int
    r1: int
    c0: int
    c1: int
    qscale: torch.Tensor  # [r_len, c_len // group_size] bfloat16
    qzero: torch.Tensor   # [r_len, c_len // group_size] bfloat16

    @property
    def nbytes(self) -> int:
        return (self.qscale.numel() + self.qzero.numel()) * 2


@dataclass
class Q1CubeMatrix:
    """2D-tiled positional Q1 bit-plane matrix [N, K]."""

    shape: Tuple[int, int]
    planes: int
    r_tile: int
    c_tile: int
    group_size: int
    padded_k: int
    spatial_tiles: Dict[Tuple[int, int], Q1SpatialTileMeta]
    cubes: List[Q1Cube]

    @property
    def total_bytes(self) -> int:
        return sum(t.nbytes for t in self.spatial_tiles.values()) + sum(c.nbytes for c in self.cubes)

    @property
    def bf16_bytes(self) -> int:
        return self.shape[0] * self.shape[1] * 2

    def cubes_for_tile(self, tile_r: int, tile_c: int) -> List[Q1Cube]:
        return [c for c in self.cubes if c.tile_r == tile_r and c.tile_c == tile_c]


@torch.no_grad()
def pack_q1_cubes(
    weight: torch.Tensor,
    planes: int = 4,
    r_tile: int = 64,
    c_tile: int = 128,
    group_size: int = 64,
    device: Optional[torch.device | str] = None,
) -> Q1CubeMatrix:
    """Pack a 2D weight matrix [N, K] into MSB-first 2D positional Q1 cubes."""
    if planes < 1 or planes > 8:
        raise ValueError(f"planes must be in [1, 8], got {planes}")
    if group_size not in (16, 32, 64):
        raise ValueError(f"group_size must be in (16, 32, 64), got {group_size}")
    if c_tile % 32 != 0 or c_tile % group_size != 0:
        raise ValueError(f"c_tile ({c_tile}) must be divisible by 32 and group_size ({group_size})")
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight tensor, got {tuple(weight.shape)}")

    n, k = weight.shape
    if k % group_size != 0:
        raise ValueError(f"Input width K={k} must be divisible by group_size={group_size}")

    dst = torch.device(device) if device is not None else weight.device
    padded_k = ((k + c_tile - 1) // c_tile) * c_tile
    qmax = (1 << planes) - 1
    bit_shifts = torch.arange(32, dtype=torch.int64, device=dst)

    spatial_tiles: Dict[Tuple[int, int], Q1SpatialTileMeta] = {}
    cubes: List[Q1Cube] = []

    n_rtiles = math.ceil(n / r_tile)
    n_ctiles = math.ceil(k / c_tile)

    for tr in range(n_rtiles):
        r0 = tr * r_tile
        r1 = min(n, r0 + r_tile)
        r_len = r1 - r0
        for tc in range(n_ctiles):
            c0 = tc * c_tile
            c1_actual = min(k, c0 + c_tile)
            c_len_actual = c1_actual - c0
            c_len_pad = ((c_len_actual + 31) // 32) * 32
            if c_len_pad % group_size != 0:
                c_len_pad = ((c_len_pad + group_size - 1) // group_size) * group_size

            tile = weight[r0:r1, c0:c1_actual].to(device=dst, dtype=torch.float32)
            num_groups_actual = c_len_actual // group_size
            w_grp = tile.reshape(r_len, num_groups_actual, group_size)

            zero = w_grp.amin(-1).to(torch.bfloat16)
            scale = (
                ((w_grp.amax(-1).double() - zero.double()) / qmax)
                .clamp_min_(1e-8)
                .to(torch.bfloat16)
            )
            q = (
                ((w_grp.double() - zero.double().unsqueeze(-1)) / scale.double().unsqueeze(-1))
                .round_()
                .clamp_(0, qmax)
                .to(torch.int32)
                .reshape(r_len, c_len_actual)
            )
            if c_len_pad > c_len_actual:
                q_padded = torch.zeros((r_len, c_len_pad), dtype=torch.int32, device=dst)
                q_padded[:, :c_len_actual] = q
                q = q_padded
                pad_groups = c_len_pad // group_size
                if pad_groups > num_groups_actual:
                    z_pad = torch.zeros((r_len, pad_groups), dtype=torch.bfloat16, device=dst)
                    s_pad = torch.ones((r_len, pad_groups), dtype=torch.bfloat16, device=dst)
                    z_pad[:, :num_groups_actual] = zero
                    s_pad[:, :num_groups_actual] = scale
                    zero, scale = z_pad, s_pad

            spatial_tiles[(tr, tc)] = Q1SpatialTileMeta(
                tile_r=tr, tile_c=tc, r0=r0, r1=r1, c0=c0, c1=c1_actual,
                qscale=scale.contiguous(), qzero=zero.contiguous(),
            )

            c_words = c_len_pad // 32
            q_words = q.reshape(r_len, c_words, 32)
            for k_plane in range(planes - 1, -1, -1):
                b = ((q_words >> k_plane) & 1).to(torch.int64)
                bits_i32 = (b << bit_shifts).sum(dim=-1).to(torch.int32).contiguous()
                cubes.append(
                    Q1Cube(
                        tile_r=tr, tile_c=tc, r0=r0, r1=r1, c0=c0, c1=c1_actual,
                        plane_idx=k_plane, weight_pow2=float(1 << k_plane), bits_i32=bits_i32,
                    )
                )

    return Q1CubeMatrix(
        shape=(n, k), planes=planes, r_tile=r_tile, c_tile=c_tile,
        group_size=group_size, padded_k=padded_k,
        spatial_tiles=spatial_tiles, cubes=cubes,
    )


@torch.no_grad()
def unpack_q1_cubes(
    q1_mat: Q1CubeMatrix,
    dtype: torch.dtype = torch.bfloat16,
    active_planes: Optional[int] = None,
    midpoint_correct: bool = False,
) -> torch.Tensor:
    """Reconstruct [N, K] from Q1 cubes with optional z_eff(P) midpoint correction."""
    n, k = q1_mat.shape
    device = next(iter(q1_mat.spatial_tiles.values())).qscale.device
    out = torch.empty((n, k), dtype=dtype, device=device)
    gs = q1_mat.group_size
    bit_shifts = torch.arange(32, dtype=torch.int64, device=device)

    p_act = q1_mat.planes if active_planes is None else int(active_planes)
    min_plane = max(0, q1_mat.planes - p_act)
    tail_mid = 0.5 * ((1 << min_plane) - 1) if (midpoint_correct and min_plane > 0) else 0.0

    for (tr, tc), meta in q1_mat.spatial_tiles.items():
        r_len = meta.r1 - meta.r0
        c_actual = meta.c1 - meta.c0
        tile_cubes = [c for c in q1_mat.cubes_for_tile(tr, tc) if c.plane_idx >= min_plane]
        c_words = meta.qscale.shape[1] * gs // 32
        q_acc = torch.zeros((r_len, c_words * 32), dtype=torch.float32, device=device)
        for cube in tile_cubes:
            bits_i64 = cube.bits_i32.to(torch.int64) & 0xFFFFFFFF
            unpacked = ((bits_i64.unsqueeze(-1) >> bit_shifts) & 1).reshape(r_len, c_words * 32)
            q_acc += cube.weight_pow2 * unpacked.float()

        w_grp = q_acc.reshape(r_len, -1, gs)
        qscale_f = meta.qscale.float().unsqueeze(-1)
        qzero_f = meta.qzero.float().unsqueeze(-1) + (qscale_f * tail_mid if tail_mid != 0.0 else 0.0)
        w_deq = (w_grp * qscale_f + qzero_f).reshape(r_len, -1)[:, :c_actual]
        out[meta.r0 : meta.r1, meta.c0 : meta.c1] = w_deq.to(dtype)
    return out


@torch.no_grad()
def dequantize_scale_laws(
    weight: torch.Tensor,
    planes: int = 4,
    group_size: int = 64,
) -> Dict[str, torch.Tensor]:
    """Compare Q1 scale laws on [N, K]: positional_z_eff, sign_geom, sign_arith, sign_residual."""
    n, k = weight.shape
    w_grp = weight.float().reshape(n, k // group_size, group_size)

    # 1. Positional (with z_eff midpoint when truncated; exact affine here at full planes)
    q1 = pack_q1_cubes(weight, planes=planes, r_tile=min(64, n), c_tile=max(64, group_size), group_size=group_size)
    w_pos = unpack_q1_cubes(q1, dtype=torch.float32)

    # 2. Sign Geometric 2^(-(p+1)) around group midpoint
    abs_max = w_grp.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale_g = abs_max / max(1e-8, sum(2.0 ** (-(p + 1)) for p in range(planes)))
    rem_g = w_grp.clone()
    rec_g = torch.zeros_like(w_grp)
    for p in range(planes):
        s_p = torch.where(rem_g >= 0, 1.0, -1.0)
        step = scale_g * (2.0 ** (-(p + 1)))
        rec_g += step * s_p
        rem_g -= step * s_p

    # 3. Sign Arithmetic (K - p) / sum(1..K)
    denom_a = planes * (planes + 1) / 2.0
    rem_a = w_grp.clone()
    rec_a = torch.zeros_like(w_grp)
    for p in range(planes):
        s_p = torch.where(rem_a >= 0, 1.0, -1.0)
        step = abs_max * ((planes - p) / denom_a)
        rec_a += step * s_p
        rem_a -= step * s_p

    # 4. Residual Sign Q1 (alpha_p = mean(|R_{p-1}|) per group)
    rem_r = w_grp.clone()
    rec_r = torch.zeros_like(w_grp)
    for _ in range(planes):
        s_p = torch.where(rem_r >= 0, 1.0, -1.0)
        alpha_p = rem_r.abs().mean(dim=-1, keepdim=True)
        delta = alpha_p * s_p
        rec_r += delta
        rem_r -= delta

    return {
        "positional": w_pos,
        "sign_geom": rec_g.reshape(n, k),
        "sign_arith": rec_a.reshape(n, k),
        "sign_residual": rec_r.reshape(n, k),
    }


@dataclass
class Q1PackedBatch:
    """Contiguous packed sub-matrix batch [P_max, r_len, c_words] for 1-launch NVRTC."""

    r0: int
    r1: int
    c0: int
    c1: int
    c_actual: int
    planes_u32: torch.Tensor  # [P_max, r_len, c_words] int32
    qscale: torch.Tensor      # [r_len, num_groups] bfloat16
    qzero: torch.Tensor       # [r_len, num_groups] bfloat16

    def nbytes(self, p_active: Optional[int] = None) -> int:
        p = int(self.planes_u32.shape[0]) if p_active is None else int(p_active)
        p = max(1, min(p, int(self.planes_u32.shape[0])))
        plane_bytes = p * int(self.planes_u32.shape[1]) * int(self.planes_u32.shape[2]) * 4
        meta_bytes = (self.qscale.numel() + self.qzero.numel()) * 2
        return plane_bytes + meta_bytes

    def pin_inplace(self) -> None:
        if self.planes_u32.device.type == "cpu" and not self.planes_u32.is_pinned():
            self.planes_u32 = self.planes_u32.pin_memory()
            self.qscale = self.qscale.pin_memory()
            self.qzero = self.qzero.pin_memory()


class Q1RoleCycleArena:
    """Strict physical A -> B -> C(0) -> A sub-matrix streaming buffer with 2-stream CUDA overlap."""

    def __init__(self, slot_budget_bytes: int, device: torch.device):
        if slot_budget_bytes <= 0:
            raise ValueError("slot_budget_bytes must be positive")
        self.slot_budget_bytes = int(slot_budget_bytes)
        self.device = torch.device(device)
        self.active_area = 0
        self.warm_area = 1
        self.standby_area = 2
        self.buffers: List[Optional[Tuple[Any, Any]]] = [None, None, None]
        self._slot_compute_events: List[Any] = [None, None, None]
        self._slot_h2d_events: List[Any] = [None, None, None]
        self._pinned_host_scratch: List[Optional[torch.Tensor]] = [None, None, None]
        self._scratch_offset: int = 0
        self._transfer_stream: Any = None
        self.area_used_bytes = [0, 0, 0]
        self.role_history: List[List[str]] = [["A"], ["B"], ["C"]]
        self.transitions: Dict[str, int] = {"A->B": 0, "B->C": 0, "C->A": 0, "B->A": 0}
        self.peak_slot_bytes = 0
        self.rotations = 0
        self.zero_copy_adoptions = 0
        self.batches_staged = 0
        self.nvrtc_batch_launches = 0
        self.async_h2d_transfers = 0

    def _get_transfer_stream(self):
        if self.device.type != "cuda":
            return None
        if self._transfer_stream is None:
            self._transfer_stream = torch.cuda.Stream(device=self.device)
        return self._transfer_stream

    def stage_into_B(self, cube: Q1Cube, meta: Q1SpatialTileMeta) -> int:
        needed = cube.nbytes + meta.nbytes
        if needed > self.slot_budget_bytes:
            raise ValueError(
                f"Single Q1 cube ({needed} B) exceeds slot_budget_bytes ({self.slot_budget_bytes} B)"
            )
        if cube.bits_i32.device != self.device:
            cube = Q1Cube(
                cube.tile_r, cube.tile_c, cube.r0, cube.r1, cube.c0, cube.c1,
                cube.plane_idx, cube.weight_pow2, cube.bits_i32.to(self.device),
            )
            meta = Q1SpatialTileMeta(
                meta.tile_r, meta.tile_c, meta.r0, meta.r1, meta.c0, meta.c1,
                meta.qscale.to(self.device), meta.qzero.to(self.device),
            )
        self.buffers[self.warm_area] = (cube, meta)
        self.area_used_bytes[self.warm_area] = needed
        self.peak_slot_bytes = max(self.peak_slot_bytes, needed)
        return self.warm_area

    def stage_batch_into_B(self, batch: Q1PackedBatch, p_active: int) -> int:
        """Stage a full slot-budgeted Q1PackedBatch [P_active, r_batch, c_words] into B."""
        needed = batch.nbytes(p_active)
        if needed > self.slot_budget_bytes:
            raise ValueError(
                f"Q1PackedBatch ({needed} B) exceeds slot_budget_bytes ({self.slot_budget_bytes} B)"
            )
        planes_src = batch.planes_u32[:p_active]
        done_ev = None
        if self.device.type == "cuda":
            t_stream = self._get_transfer_stream()
            if planes_src.device.type == "cpu":
                batch.pin_inplace()
                planes_src = batch.planes_u32[:p_active]
            with torch.cuda.stream(t_stream):
                prev_comp = self._slot_compute_events[self.warm_area]
                if prev_comp is not None:
                    t_stream.wait_event(prev_comp)
                    self._slot_compute_events[self.warm_area] = None
                if planes_src.device != self.device:
                    p_dev = planes_src.to(self.device, non_blocking=True)
                    s_dev = batch.qscale.to(self.device, non_blocking=True)
                    z_dev = batch.qzero.to(self.device, non_blocking=True)
                    self.async_h2d_transfers += 1
                else:
                    p_dev = planes_src
                    s_dev = batch.qscale
                    z_dev = batch.qzero
                done_ev = torch.cuda.Event(enable_timing=False)
                done_ev.record(t_stream)
            dev_batch = Q1PackedBatch(
                batch.r0, batch.r1, batch.c0, batch.c1, batch.c_actual, p_dev, s_dev, z_dev,
            )
        else:
            dev_batch = Q1PackedBatch(
                batch.r0, batch.r1, batch.c0, batch.c1, batch.c_actual,
                planes_src if planes_src.device == self.device else planes_src.to(self.device),
                batch.qscale if batch.qscale.device == self.device else batch.qscale.to(self.device),
                batch.qzero if batch.qzero.device == self.device else batch.qzero.to(self.device),
            )
        self.buffers[self.warm_area] = (dev_batch, done_ev)
        self.area_used_bytes[self.warm_area] = needed
        self.peak_slot_bytes = max(self.peak_slot_bytes, needed)
        self.batches_staged += 1
        return self.warm_area

    def _move_payload_to_device(self, payload: Any) -> Tuple[Any, bool]:
        """Recursively stage CPU tensors onto self.device (reused pinned scratch + non_blocking on CUDA)."""
        moved = False
        if isinstance(payload, torch.Tensor):
            if payload.device == self.device:
                return payload, False
            src = payload
            if self.device.type == "cuda" and src.device.type == "cpu":
                if src.is_pinned():
                    return src.to(self.device, non_blocking=True), True
                src_c = src.contiguous()
                raw_bytes = int(src_c.numel()) * int(src_c.element_size())
                if raw_bytes == 0:
                    return src_c.to(self.device), True
                buf = self._pinned_host_scratch[self.warm_area]
                off = (self._scratch_offset + 63) & ~63
                if buf is not None and off + raw_bytes <= int(buf.numel()):
                    dst_view = buf[off : off + raw_bytes].view(src_c.dtype).reshape(src_c.shape)
                    dst_view.copy_(src_c)
                    self._scratch_offset = off + raw_bytes
                    return dst_view.to(self.device, non_blocking=True), True
                return src_c.pin_memory().to(self.device, non_blocking=True), True
            return src.to(self.device), True
        if isinstance(payload, tuple):
            items = []
            for item in payload:
                out_item, item_moved = self._move_payload_to_device(item)
                items.append(out_item)
                moved = moved or item_moved
            return tuple(items), moved
        if isinstance(payload, list):
            items = []
            for item in payload:
                out_item, item_moved = self._move_payload_to_device(item)
                items.append(out_item)
                moved = moved or item_moved
            return items, moved
        if isinstance(payload, dict):
            out_dict: Dict[Any, Any] = {}
            for k, v in payload.items():
                out_v, v_moved = self._move_payload_to_device(v)
                out_dict[k] = out_v
                moved = moved or v_moved
            return out_dict, moved
        return payload, False

    def stage_payload_into_B(self, payload: Any, nbytes: int) -> int:
        """Stage a generic payload (e.g. Q1PuzzleOverlay delta batch) into warm area B."""
        needed = int(nbytes)
        if needed < 0:
            raise ValueError(f"nbytes must be non-negative, got {needed}")
        if needed > self.slot_budget_bytes:
            raise ValueError(
                f"Payload ({needed} B) exceeds slot_budget_bytes ({self.slot_budget_bytes} B)"
            )
        done_ev = None
        if self.device.type == "cuda":
            prev_h2d = self._slot_h2d_events[self.warm_area]
            if prev_h2d is not None:
                prev_h2d.synchronize()
                self._slot_h2d_events[self.warm_area] = None
            # B stages the actual next working set, not the maximum possible
            # A allocation. In particular a short surveyed prefix must not
            # pin three full-depth layers in host RAM.
            scratch_cap = needed + 4096
            buf = self._pinned_host_scratch[self.warm_area]
            if buf is None or int(buf.numel()) < scratch_cap:
                self._pinned_host_scratch[self.warm_area] = torch.empty(
                    (scratch_cap,), dtype=torch.uint8, pin_memory=True
                )
            self._scratch_offset = 0
            t_stream = self._get_transfer_stream()
            with torch.cuda.stream(t_stream):
                prev_comp = self._slot_compute_events[self.warm_area]
                if prev_comp is not None:
                    t_stream.wait_event(prev_comp)
                    self._slot_compute_events[self.warm_area] = None
                dev_payload, moved = self._move_payload_to_device(payload)
                if moved:
                    self.async_h2d_transfers += 1
                done_ev = torch.cuda.Event(enable_timing=False)
                done_ev.record(t_stream)
            self._slot_h2d_events[self.warm_area] = done_ev
        else:
            dev_payload, _ = self._move_payload_to_device(payload)

        self.buffers[self.warm_area] = (dev_payload, done_ev)
        self.area_used_bytes[self.warm_area] = needed
        self.peak_slot_bytes = max(self.peak_slot_bytes, needed)
        self.batches_staged += 1
        return self.warm_area

    def active_payload(self) -> Any:
        """Return the payload currently in active area A, waiting on its H2D event on CUDA."""
        entry = self.buffers[self.active_area]
        if entry is None:
            return None
        payload, done_ev = entry
        if done_ev is not None and self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(done_ev)
            # The allocation was made on the preparation stream. Its lifetime
            # must include the consumer, even after B has gone through C(0).
            self._record_consumer(payload, stream)
        return payload

    @staticmethod
    def _record_consumer(payload: Any, stream: Any) -> None:
        if isinstance(payload, torch.Tensor):
            if payload.device.type == "cuda":
                payload.record_stream(stream)
        elif isinstance(payload, dict):
            for value in payload.values():
                Q1RoleCycleArena._record_consumer(value, stream)
        elif isinstance(payload, (tuple, list)):
            for value in payload:
                Q1RoleCycleArena._record_consumer(value, stream)

    def publish_prepared(self, payload: Any, *, gpu_bytes: int = 0, ready_event: Any = None) -> None:
        """Publish already prepared work; the turn adopts it without copying.

        Host-only disk preparation occupies no VRAM. Its owner must account
        for host bytes separately. A GPU payload remains charged to B until
        rotate() transfers that same object to old C, the new A.
        """
        if not 0 <= int(gpu_bytes) <= self.slot_budget_bytes:
            raise ValueError("Prepared work exceeds the relay slot budget")
        if self.buffers[self.warm_area] is not None:
            raise RuntimeError("B already contains unconsumed prepared work")
        self.buffers[self.warm_area] = (payload, ready_event)
        self.area_used_bytes[self.warm_area] = int(gpu_bytes)
        self.peak_slot_bytes = max(self.peak_slot_bytes, int(gpu_bytes))

    def release_active_A(self) -> None:
        """Record active compute completion on A and release A's payload reference cleanly."""
        if self.device.type == "cuda":
            comp_ev = torch.cuda.Event(enable_timing=False)
            comp_ev.record(torch.cuda.current_stream(self.device))
            self._slot_compute_events[self.active_area] = comp_ev
        self.buffers[self.active_area] = None
        self.area_used_bytes[self.active_area] = 0

    def rotate(self) -> None:
        """Advance physical areas strictly: old_C(0) -> new_A, old_A -> new_B, old_B -> new_C(0)."""
        old_a, old_b, old_c = self.active_area, self.warm_area, self.standby_area
        new_a, new_b, new_c = old_c, old_a, old_b

        adopted = self.buffers[old_b]
        adopted_bytes = self.area_used_bytes[old_b]
        old_a_comp_ev = self._slot_compute_events[old_a]

        self.buffers[old_b] = None
        self.area_used_bytes[old_b] = 0
        self.buffers[old_a] = None
        self.area_used_bytes[old_a] = 0

        self.buffers[new_a] = adopted
        self.area_used_bytes[new_a] = adopted_bytes
        if adopted is not None:
            self.zero_copy_adoptions += 1

        self._slot_compute_events[new_a] = None
        self._slot_compute_events[new_b] = old_a_comp_ev
        self._slot_compute_events[new_c] = None

        self.active_area, self.warm_area, self.standby_area = new_a, new_b, new_c
        assert self.area_used_bytes[new_c] == 0
        self.role_history[new_a].append("A")
        self.role_history[new_b].append("B")
        self.role_history[new_c].append("C")
        self.transitions["A->B"] += 1
        self.transitions["B->C"] += 1
        self.transitions["C->A"] += 1
        self.rotations += 1

    def stats(self) -> Dict[str, object]:
        return {
            "slot_budget_bytes": self.slot_budget_bytes,
            "peak_slot_bytes": self.peak_slot_bytes,
            "rotations": self.rotations,
            "zero_copy_adoptions": self.zero_copy_adoptions,
            "batches_staged": self.batches_staged,
            "nvrtc_batch_launches": self.nvrtc_batch_launches,
            "async_h2d_transfers": self.async_h2d_transfers,
            "transitions": dict(self.transitions),
            "area_used_bytes": list(self.area_used_bytes),
            "active_area": self.active_area,
            "warm_area": self.warm_area,
            "standby_area": self.standby_area,
            "role_history": [list(h) for h in self.role_history],
        }


@torch.no_grad()
def q1_linear_streamed(
    x: torch.Tensor,
    q1_mat: Q1CubeMatrix,
    slot_budget_bytes: int,
    *,
    active_planes: Optional[int] = None,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    midpoint_correct: bool = True,
    use_nvrtc: bool = True,
    use_popc: bool = False,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Stream 2D Q1 cubes via A -> B -> C(0) -> A under slot_budget_bytes < 1 matrix."""
    if slot_budget_bytes >= q1_mat.total_bytes:
        raise ValueError(
            f"slot_budget_bytes ({slot_budget_bytes}) must be strictly < matrix bytes ({q1_mat.total_bytes})"
        )
    orig_shape = x.shape
    device = x.device
    flat_x = x.reshape(-1, q1_mat.shape[1])
    m, k = flat_x.shape
    n, k_mat = q1_mat.shape
    assert k == k_mat

    p_act = q1_mat.planes if active_planes is None else int(active_planes)
    min_plane = max(0, q1_mat.planes - p_act)
    tail_mid = 0.5 * ((1 << min_plane) - 1) if (midpoint_correct and min_plane > 0) else 0.0
    selected_cubes = [c for c in q1_mat.cubes if c.plane_idx >= min_plane]

    arena = Q1RoleCycleArena(slot_budget_bytes, device)
    y_acc = torch.zeros((m, n), dtype=torch.float32, device=device)
    gs = q1_mat.group_size
    zero_applied: set[Tuple[int, int]] = set()
    use_cuda_nvrtc = use_nvrtc and device.type == "cuda" and flat_x.dtype == torch.bfloat16

    if use_cuda_nvrtc:
        from src.quintessentia_q1_cube_nvrtc import launch_q1_plane_gemv

        x_c = flat_x.contiguous()
        for cube in selected_cubes:
            meta = q1_mat.spatial_tiles[(cube.tile_r, cube.tile_c)]
            arena.stage_into_B(cube, meta)
            arena.rotate()
            cur_cube, cur_meta = arena.buffers[arena.active_area]
            tile_key = (cur_cube.tile_r, cur_cube.tile_c)
            add_zero = tile_key not in zero_applied
            zero_applied.add(tile_key)
            eff_zero = (
                (cur_meta.qzero.float() + cur_meta.qscale.float() * tail_mid).to(torch.bfloat16)
                if (add_zero and tail_mid != 0.0)
                else cur_meta.qzero
            )
            launch_q1_plane_gemv(
                x_c, cur_cube.bits_i32, cur_meta.qscale, eff_zero, y_acc,
                r0=cur_cube.r0, c0=cur_cube.c0, c_actual=cur_cube.c1 - cur_cube.c0,
                group_size=gs, alpha=cur_cube.weight_pow2, add_zero=add_zero, use_popc=use_popc,
            )
    else:
        flat_x_f = flat_x.float()
        bit_shifts = torch.arange(32, dtype=torch.int64, device=device)
        for cube in selected_cubes:
            meta = q1_mat.spatial_tiles[(cube.tile_r, cube.tile_c)]
            arena.stage_into_B(cube, meta)
            arena.rotate()
            cur_cube, cur_meta = arena.buffers[arena.active_area]
            r0, r1, c0, c1 = cur_cube.r0, cur_cube.r1, cur_cube.c0, cur_cube.c1
            r_len, c_actual = r1 - r0, c1 - c0
            c_words = cur_cube.bits_i32.shape[1]

            x_tile = flat_x_f[:, c0:c1]
            if c_words * 32 > c_actual:
                x_pad = torch.zeros((m, c_words * 32), dtype=torch.float32, device=device)
                x_pad[:, :c_actual] = x_tile
                x_tile_grp = x_pad.reshape(m, -1, gs)
            else:
                x_tile_grp = x_tile.reshape(m, -1, gs)

            tile_key = (cur_cube.tile_r, cur_cube.tile_c)
            if tile_key not in zero_applied:
                zero_applied.add(tile_key)
                z_eff = cur_meta.qzero.float() + (cur_meta.qscale.float() * tail_mid if tail_mid != 0.0 else 0.0)
                y_acc[:, r0:r1] += x_tile_grp.sum(dim=-1) @ z_eff.t()

            bits_i64 = cur_cube.bits_i32.to(torch.int64) & 0xFFFFFFFF
            b_k = ((bits_i64.unsqueeze(-1) >> bit_shifts) & 1).reshape(r_len, -1, gs).float()
            grp_dots = torch.einsum("mgs,rgs->mrg", x_tile_grp, b_k)
            y_acc[:, r0:r1] += cur_cube.weight_pow2 * (
                grp_dots * cur_meta.qscale.float().unsqueeze(0)
            ).sum(dim=-1)

    if bias is not None:
        y_acc += bias.to(device=device, dtype=torch.float32).unsqueeze(0)

    target_dtype = x.dtype if out_dtype is None else out_dtype
    out = y_acc.to(target_dtype).reshape(*orig_shape[:-1], n)
    stats = dict(
        matrix_total_bytes=q1_mat.total_bytes,
        matrix_bf16_bytes=q1_mat.bf16_bytes,
        slot_budget_bytes=slot_budget_bytes,
        peak_slot_bytes=arena.peak_slot_bytes,
        budget_ratio=arena.peak_slot_bytes / max(1, q1_mat.total_bytes),
        cubes_executed=len(selected_cubes),
        rotations=arena.rotations,
        zero_copy_adoptions=arena.zero_copy_adoptions,
        transitions=dict(arena.transitions),
        role_history=arena.role_history,
    )
    return out, stats


@dataclass
class Q1RadarPack:
    """Contiguous 2D-tiled layout for 1-launch adaptive radar GEMV (CPU or NVRTC)."""

    shape: Tuple[int, int]
    k_max: int
    r_tile: int
    c_tile: int
    group_size: int
    planes_u32: torch.Tensor  # [n_rtiles, n_ctiles, k_max, r_tile, c_words]
    qscale: torch.Tensor      # [n_rtiles, n_ctiles, r_tile, groups_tile]
    qzero: torch.Tensor       # [n_rtiles, n_ctiles, r_tile, groups_tile]


@torch.no_grad()
def pack_for_radar_kernel(
    weight: torch.Tensor,
    k_max: int = 6,
    r_tile: int = 16,
    c_tile: int = 64,
    group_size: int = 64,
    device: Optional[torch.device | str] = None,
) -> Q1RadarPack:
    """Pack [N, K] (N % r_tile == 0, K % c_tile == 0) into contiguous 2D radar tiles."""
    dst = torch.device(device) if device is not None else weight.device
    n, k = weight.shape
    if n % r_tile != 0 or k % c_tile != 0:
        raise ValueError(f"Shape {(n, k)} must be divisible by tile {(r_tile, c_tile)}")
    q1 = pack_q1_cubes(weight, planes=k_max, r_tile=r_tile, c_tile=c_tile, group_size=group_size, device=dst)
    n_rt, n_ct = n // r_tile, k // c_tile
    c_words, groups_tile = c_tile // 32, c_tile // group_size
    planes_u32 = torch.empty((n_rt, n_ct, k_max, r_tile, c_words), dtype=torch.int32, device=dst)
    qscale = torch.empty((n_rt, n_ct, r_tile, groups_tile), dtype=torch.bfloat16, device=dst)
    qzero = torch.empty((n_rt, n_ct, r_tile, groups_tile), dtype=torch.bfloat16, device=dst)
    for tr in range(n_rt):
        for tc in range(n_ct):
            meta = q1.spatial_tiles[(tr, tc)]
            qscale[tr, tc].copy_(meta.qscale)
            qzero[tr, tc].copy_(meta.qzero)
            for p_idx, cube in enumerate(q1.cubes_for_tile(tr, tc)):
                planes_u32[tr, tc, p_idx].copy_(cube.bits_i32)
    return Q1RadarPack(
        shape=(n, k), k_max=k_max, r_tile=r_tile, c_tile=c_tile, group_size=group_size,
        planes_u32=planes_u32.contiguous(), qscale=qscale.contiguous(), qzero=qzero.contiguous(),
    )


@torch.no_grad()
def q1_radar_linear(
    x: torch.Tensor,
    radar_pack: Q1RadarPack,
    tile_depths: torch.Tensor,
) -> torch.Tensor:
    """Execute adaptive-depth 2D-tiled Q1 linear on CUDA (1-launch NVRTC) or CPU."""
    flat_x = x.reshape(-1, radar_pack.shape[1]).contiguous()
    m, _ = flat_x.shape
    n, _ = radar_pack.shape
    if x.device.type == "cuda" and flat_x.dtype == torch.bfloat16:
        from src.quintessentia_q1_cube_nvrtc import launch_q1_adaptive_radar_gemv

        y_out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        launch_q1_adaptive_radar_gemv(
            flat_x, radar_pack.planes_u32, radar_pack.qscale, radar_pack.qzero,
            tile_depths.to(device=x.device, dtype=torch.int32).contiguous(), y_out,
            r_tile=radar_pack.r_tile, c_tile=radar_pack.c_tile,
            group_size=radar_pack.group_size, k_max=radar_pack.k_max,
        )
        return y_out.reshape(*x.shape[:-1], n)

    # Exact CPU / general fallback matching the kernel's d==0 skip and z_eff(d)
    bit_shifts = torch.arange(32, dtype=torch.int64, device=x.device)
    n_rt, n_ct = radar_pack.planes_u32.shape[0], radar_pack.planes_u32.shape[1]
    gs = radar_pack.group_size
    y_acc = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    x_f = flat_x.float()
    depths = tile_depths.cpu()
    for tr in range(n_rt):
        r0, r1 = tr * radar_pack.r_tile, (tr + 1) * radar_pack.r_tile
        for tc in range(n_ct):
            d = int(depths[tr, tc].item())
            if d <= 0:
                continue
            d = min(d, radar_pack.k_max)
            c0, c1 = tc * radar_pack.c_tile, (tc + 1) * radar_pack.c_tile
            x_grp = x_f[:, c0:c1].reshape(m, -1, gs)
            tail_mid = 0.5 * ((1 << (radar_pack.k_max - d)) - 1) if d < radar_pack.k_max else 0.0
            sc = radar_pack.qscale[tr, tc].float()
            zr = radar_pack.qzero[tr, tc].float() + (sc * tail_mid if tail_mid != 0.0 else 0.0)
            y_acc[:, r0:r1] += x_grp.sum(dim=-1) @ zr.t()
            for p in range(d):
                alpha = float(1 << (radar_pack.k_max - 1 - p))
                bits_i64 = radar_pack.planes_u32[tr, tc, p].to(torch.int64) & 0xFFFFFFFF
                b_p = ((bits_i64.unsqueeze(-1) >> bit_shifts) & 1).reshape(radar_pack.r_tile, -1, gs).float()
                y_acc[:, r0:r1] += alpha * (torch.einsum("mgs,rgs->mrg", x_grp, b_p) * sc.unsqueeze(0)).sum(dim=-1)
    return y_acc.to(x.dtype).reshape(*x.shape[:-1], n)


@torch.no_grad()
def build_gate_radar_depths(
    x: torch.Tensor,
    gate_radar: Q1RadarPack,
    *,
    quiet_depth: int = 4,
    hot_depth: int = 6,
    closed_quantile: float = 0.12,
    hot_quantile: float = 0.60,
) -> torch.Tensor:
    """Compute per-tile depths using 1-bit (P=1) gate_proj radar."""
    n_rt, n_ct = gate_radar.planes_u32.shape[0], gate_radar.planes_u32.shape[1]
    d_p1 = torch.ones((n_rt, n_ct), dtype=torch.int32, device=x.device)
    y_radar = q1_radar_linear(x, gate_radar, d_p1)
    row_act = F.silu(y_radar.reshape(-1, gate_radar.shape[0])[0].float()).abs().reshape(n_rt, gate_radar.r_tile).mean(dim=-1)
    q_c = torch.quantile(row_act, closed_quantile)
    q_h = torch.quantile(row_act, hot_quantile)
    row_depths = torch.where(
        row_act <= q_c,
        torch.zeros_like(row_act, dtype=torch.int32),
        torch.where(
            row_act <= q_h,
            torch.full_like(row_act, quiet_depth, dtype=torch.int32),
            torch.full_like(row_act, hot_depth, dtype=torch.int32),
        ),
    )
    return row_depths.unsqueeze(1).expand(n_rt, n_ct).contiguous()


@torch.no_grad()
def streamed_checkpoint_mlp_pass(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    planes: int = 4,
    r_tile: int = 64,
    c_tile: int = 128,
    group_size: int = 64,
    slot_budget_bytes: Optional[int] = None,
    lm_head_bytes: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Run SwiGLU MLP (gate, up -> SiLU*up -> down) streaming 2D Q1 cubes under <1 matrix & <lm_head."""
    q1_g = pack_q1_cubes(w_gate, planes=planes, r_tile=r_tile, c_tile=c_tile, group_size=group_size, device=x.device)
    q1_u = pack_q1_cubes(w_up, planes=planes, r_tile=r_tile, c_tile=c_tile, group_size=group_size, device=x.device)
    q1_d = pack_q1_cubes(w_down, planes=planes, r_tile=r_tile, c_tile=c_tile, group_size=group_size, device=x.device)

    min_mat_bytes = min(q1_g.total_bytes, q1_u.total_bytes, q1_d.total_bytes)
    max_cube = max(
        c.nbytes + q1.spatial_tiles[(c.tile_r, c.tile_c)].nbytes
        for q1 in (q1_g, q1_u, q1_d)
        for c in q1.cubes
    )
    if slot_budget_bytes is None:
        slot_budget_bytes = max_cube + 1024
    if slot_budget_bytes >= min_mat_bytes:
        raise ValueError("slot_budget_bytes must be strictly less than one matrix")
    if lm_head_bytes is not None and slot_budget_bytes >= lm_head_bytes:
        raise ValueError("slot_budget_bytes must be strictly less than lm_head_bytes")

    g_f32, st_g = q1_linear_streamed(x, q1_g, slot_budget_bytes, out_dtype=torch.float32)
    u_f32, st_u = q1_linear_streamed(x, q1_u, slot_budget_bytes, out_dtype=torch.float32)
    inter = (F.silu(g_f32) * u_f32).to(x.dtype)
    out, st_d = q1_linear_streamed(inter, q1_d, slot_budget_bytes, out_dtype=x.dtype)

    return out, {
        "slot_budget_bytes": slot_budget_bytes,
        "min_matrix_bytes": min_mat_bytes,
        "lm_head_bytes": lm_head_bytes,
        "gate_stats": st_g,
        "up_stats": st_u,
        "down_stats": st_d,
    }


@torch.no_grad()
def _extract_q_tile_from_packed(
    packed: Dict[str, torch.Tensor],
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    bits: int,
    dst: torch.device,
) -> torch.Tensor:
    if bits == 8:
        return packed["packed_lo"][r0:r1, c0:c1].to(device=dst, dtype=torch.int32)
    r_len = r1 - r0
    c_len = c1 - c0
    lo_tile = packed["packed_lo"][r0:r1, c0 // 2 : c1 // 2].to(device=dst)
    q = torch.stack((lo_tile & 15, lo_tile >> 4), dim=-1).reshape(r_len, c_len).to(torch.int32)
    if bits == 6:
        hi = packed.get("packed_hi")
        if hi is None:
            raise ValueError("packed_hi is required when bits == 6")
        hi_tile = hi[r0:r1, c0 // 4 : c1 // 4].to(device=dst)
        upper = torch.stack(tuple((hi_tile >> shift) & 3 for shift in (0, 2, 4, 6)), dim=-1).reshape(r_len, c_len).to(torch.int32)
        q = q | (upper << 4)
    return q


@torch.no_grad()
def q1_cubes_from_packed(
    packed: Dict[str, torch.Tensor],
    bits: int,
    group_size: int = 64,
    r_tile: int = 64,
    c_tile: int = 128,
    device: Optional[torch.device | str] = None,
) -> Q1CubeMatrix:
    """Bitwise-convert Quintessentia Q4/Q6/Q8 packed tensors to 2D Q1CubeMatrix without float round-trip."""
    if bits not in (4, 6, 8):
        raise ValueError(f"bits must be 4, 6, or 8, got {bits}")
    if group_size not in (16, 32, 64):
        raise ValueError(f"group_size must be in (16, 32, 64), got {group_size}")
    if c_tile % 32 != 0 or c_tile % group_size != 0:
        raise ValueError(f"c_tile ({c_tile}) must be divisible by 32 and group_size ({group_size})")

    lo = packed["packed_lo"]
    n = int(lo.shape[0])
    k = int(lo.shape[1]) if bits == 8 else int(lo.shape[1]) * 2
    if k % group_size != 0:
        raise ValueError(f"Input width K={k} must be divisible by group_size={group_size}")

    dst = torch.device(device) if device is not None else lo.device
    padded_k = ((k + c_tile - 1) // c_tile) * c_tile
    bit_shifts = torch.arange(32, dtype=torch.int64, device=dst)

    spatial_tiles: Dict[Tuple[int, int], Q1SpatialTileMeta] = {}
    cubes: List[Q1Cube] = []

    n_rtiles = math.ceil(n / r_tile)
    n_ctiles = math.ceil(k / c_tile)

    for tr in range(n_rtiles):
        r0 = tr * r_tile
        r1 = min(n, r0 + r_tile)
        r_len = r1 - r0
        for tc in range(n_ctiles):
            c0 = tc * c_tile
            c1_actual = min(k, c0 + c_tile)
            c_len_actual = c1_actual - c0
            c_len_pad = ((c_len_actual + 31) // 32) * 32
            if c_len_pad % group_size != 0:
                c_len_pad = ((c_len_pad + group_size - 1) // group_size) * group_size

            q = _extract_q_tile_from_packed(packed, r0, r1, c0, c1_actual, bits, dst)
            g0 = c0 // group_size
            g1 = c1_actual // group_size
            scale = packed["qscale"][r0:r1, g0:g1].to(device=dst, dtype=torch.bfloat16)
            zero = packed["qzero"][r0:r1, g0:g1].to(device=dst, dtype=torch.bfloat16)

            if c_len_pad > c_len_actual:
                q_pad = torch.zeros((r_len, c_len_pad), dtype=torch.int32, device=dst)
                q_pad[:, :c_len_actual] = q
                q = q_pad
                pad_groups = c_len_pad // group_size
                actual_groups = g1 - g0
                if pad_groups > actual_groups:
                    z_pad = torch.zeros((r_len, pad_groups), dtype=torch.bfloat16, device=dst)
                    s_pad = torch.ones((r_len, pad_groups), dtype=torch.bfloat16, device=dst)
                    z_pad[:, :actual_groups] = zero
                    s_pad[:, :actual_groups] = scale
                    zero, scale = z_pad, s_pad

            spatial_tiles[(tr, tc)] = Q1SpatialTileMeta(
                tile_r=tr, tile_c=tc, r0=r0, r1=r1, c0=c0, c1=c1_actual,
                qscale=scale.contiguous(), qzero=zero.contiguous(),
            )

            c_words = c_len_pad // 32
            q_words = q.reshape(r_len, c_words, 32)
            for k_plane in range(bits - 1, -1, -1):
                b = ((q_words >> k_plane) & 1).to(torch.int64)
                bits_i32 = (b << bit_shifts).sum(dim=-1).to(torch.int32).contiguous()
                cubes.append(
                    Q1Cube(
                        tile_r=tr, tile_c=tc, r0=r0, r1=r1, c0=c0, c1=c1_actual,
                        plane_idx=k_plane, weight_pow2=float(1 << k_plane), bits_i32=bits_i32,
                    )
                )

    return Q1CubeMatrix(
        shape=(n, k), planes=bits, r_tile=r_tile, c_tile=c_tile,
        group_size=group_size, padded_k=padded_k,
        spatial_tiles=spatial_tiles, cubes=cubes,
    )


@torch.no_grad()
def radar_pack_from_packed(
    packed: Dict[str, torch.Tensor],
    bits: int,
    group_size: int = 64,
    r_tile: int = 16,
    c_tile: int = 64,
    device: Optional[torch.device | str] = None,
) -> Q1RadarPack:
    """Bitwise-convert Quintessentia Q4/Q6 packed tensors directly into contiguous Q1RadarPack."""
    lo = packed["packed_lo"]
    n = int(lo.shape[0])
    k = int(lo.shape[1]) if bits == 8 else int(lo.shape[1]) * 2
    if n % r_tile != 0 or k % c_tile != 0:
        raise ValueError(f"Shape {(n, k)} must be divisible by tile {(r_tile, c_tile)}")
    dst = torch.device(device) if device is not None else lo.device
    q1 = q1_cubes_from_packed(packed, bits=bits, group_size=group_size, r_tile=r_tile, c_tile=c_tile, device=dst)
    n_rt, n_ct = n // r_tile, k // c_tile
    c_words, groups_tile = c_tile // 32, c_tile // group_size
    planes_u32 = torch.empty((n_rt, n_ct, bits, r_tile, c_words), dtype=torch.int32, device=dst)
    qscale = torch.empty((n_rt, n_ct, r_tile, groups_tile), dtype=torch.bfloat16, device=dst)
    qzero = torch.empty((n_rt, n_ct, r_tile, groups_tile), dtype=torch.bfloat16, device=dst)
    for tr in range(n_rt):
        for tc in range(n_ct):
            meta = q1.spatial_tiles[(tr, tc)]
            qscale[tr, tc].copy_(meta.qscale)
            qzero[tr, tc].copy_(meta.qzero)
            for p_idx, cube in enumerate(q1.cubes_for_tile(tr, tc)):
                planes_u32[tr, tc, p_idx].copy_(cube.bits_i32)
    return Q1RadarPack(
        shape=(n, k), k_max=bits, r_tile=r_tile, c_tile=c_tile, group_size=group_size,
        planes_u32=planes_u32.contiguous(), qscale=qscale.contiguous(), qzero=qzero.contiguous(),
    )


@torch.no_grad()
def _pack_q1_matrix_from_row_reader(
    n: int,
    k: int,
    read_rows: Callable[[int, int], torch.Tensor],
    *,
    planes: int,
    r_tile: int,
    c_tile: int,
    group_size: int,
    device: torch.device | str = "cpu",
) -> Q1CubeMatrix:
    """Pack [n, k] into Q1CubeMatrix reading only bounded [r_tile, k] row bands."""
    dst = torch.device(device)
    spatial_tiles: Dict[Tuple[int, int], Q1SpatialTileMeta] = {}
    cubes: List[Q1Cube] = []
    padded_k = ((k + c_tile - 1) // c_tile) * c_tile
    n_rtiles = math.ceil(n / r_tile)

    for tr in range(n_rtiles):
        r0 = tr * r_tile
        r1 = min(n, r0 + r_tile)
        band = read_rows(r0, r1)
        sub = pack_q1_cubes(
            band, planes=planes, r_tile=r1 - r0, c_tile=c_tile, group_size=group_size, device=dst,
        )
        for (_, tc), meta in sub.spatial_tiles.items():
            spatial_tiles[(tr, tc)] = Q1SpatialTileMeta(
                tile_r=tr, tile_c=tc, r0=r0, r1=r1, c0=meta.c0, c1=meta.c1,
                qscale=meta.qscale, qzero=meta.qzero,
            )
        for c in sub.cubes:
            cubes.append(
                Q1Cube(
                    tile_r=tr, tile_c=c.tile_c, r0=r0, r1=r1, c0=c.c0, c1=c.c1,
                    plane_idx=c.plane_idx, weight_pow2=c.weight_pow2, bits_i32=c.bits_i32,
                )
            )
        del band, sub

    return Q1CubeMatrix(
        shape=(n, k), planes=planes, r_tile=r_tile, c_tile=c_tile,
        group_size=group_size, padded_k=padded_k,
        spatial_tiles=spatial_tiles, cubes=cubes,
    )


def build_q1_packed_batches(
    q1_mat: Q1CubeMatrix,
    slot_budget_bytes: int,
    *,
    tile_r_filter: Optional[int] = None,
) -> List[Q1PackedBatch]:
    """Pack Q1CubeMatrix into maximal contiguous batches [P, r_batch, c_words] <= slot_budget_bytes."""
    n, _k = q1_mat.shape
    n_rt = math.ceil(n / q1_mat.r_tile)
    n_ct = math.ceil(_k / q1_mat.c_tile)
    by_tile_cubes: Dict[Tuple[int, int], List[Q1Cube]] = {}
    for c in q1_mat.cubes:
        by_tile_cubes.setdefault((c.tile_r, c.tile_c), []).append(c)
    for k_t, lst in by_tile_cubes.items():
        lst.sort(key=lambda cube: cube.plane_idx, reverse=True)

    tr_list = [tile_r_filter] if tile_r_filter is not None else list(range(n_rt))
    if not tr_list:
        return []

    # Group contiguous row-bands of contiguous column tiles under slot_budget_bytes
    # 1. Partition column tiles into maximal spans where >=1 row fits inside slot_budget_bytes
    def _col_span_row_bytes(tc0: int, tc1: int) -> Tuple[int, int, int]:
        cw = sum(by_tile_cubes[(tr_list[0], tc)][0].bits_i32.shape[1] for tc in range(tc0, tc1))
        ng = sum(q1_mat.spatial_tiles[(tr_list[0], tc)].qscale.shape[1] for tc in range(tc0, tc1))
        return q1_mat.planes * cw * 4 + 2 * ng * 2, cw, ng

    col_spans: List[Tuple[int, int, int, int, int]] = []
    tc_start = 0
    while tc_start < n_ct:
        tc_end = n_ct
        row_bytes, cw_span, ng_span = _col_span_row_bytes(tc_start, tc_end)
        while tc_end - tc_start > 1 and row_bytes > slot_budget_bytes:
            tc_end -= 1
            row_bytes, cw_span, ng_span = _col_span_row_bytes(tc_start, tc_end)
        if row_bytes > slot_budget_bytes:
            raise ValueError(
                f"Single row of column tile ({row_bytes} B) exceeds slot_budget_bytes ({slot_budget_bytes} B)"
            )
        col_spans.append((tc_start, tc_end, row_bytes, cw_span, ng_span))
        tc_start = tc_end

    # 2. Concatenate column span per row tile first, then merge consecutive row bands up to budget
    batches: List[Q1PackedBatch] = []
    for tc0, tc1, row_bytes, cw_span, ng_span in col_spans:
        max_rows = max(1, slot_budget_bytes // row_bytes)
        c0 = q1_mat.spatial_tiles[(tr_list[0], tc0)].c0
        c1 = q1_mat.spatial_tiles[(tr_list[0], tc1 - 1)].c1
        c_actual = c1 - c0

        pending_planes: List[torch.Tensor] = []
        pending_scales: List[torch.Tensor] = []
        pending_zeros: List[torch.Tensor] = []
        pending_rows = 0
        batch_r0: Optional[int] = None
        batch_r1: int = 0

        def _flush_pending() -> None:
            nonlocal pending_rows, batch_r0, batch_r1
            if not pending_planes:
                return
            p_cat = pending_planes[0] if len(pending_planes) == 1 else torch.cat(pending_planes, dim=1)
            s_cat = pending_scales[0] if len(pending_scales) == 1 else torch.cat(pending_scales, dim=0)
            z_cat = pending_zeros[0] if len(pending_zeros) == 1 else torch.cat(pending_zeros, dim=0)
            batches.append(
                Q1PackedBatch(
                    r0=int(batch_r0), r1=int(batch_r1), c0=c0, c1=c1, c_actual=c_actual,
                    planes_u32=p_cat.contiguous(), qscale=s_cat.contiguous(), qzero=z_cat.contiguous(),
                )
            )
            pending_planes.clear()
            pending_scales.clear()
            pending_zeros.clear()
            pending_rows = 0
            batch_r0 = None

        for tr in tr_list:
            meta0 = q1_mat.spatial_tiles[(tr, tc0)]
            r0_tr, r1_tr = meta0.r0, meta0.r1
            r_len_tr = r1_tr - r0_tr

            # Concatenate column span for this row tile into [P, r_len_tr, cw_span]
            if tc1 - tc0 == 1:
                tc_cubes = by_tile_cubes[(tr, tc0)]
                tr_planes = torch.stack([cb.bits_i32 for cb in tc_cubes], dim=0)
                tr_scale = meta0.qscale
                tr_zero = meta0.qzero
            else:
                src_dev = meta0.qscale.device
                tr_planes = torch.empty((q1_mat.planes, r_len_tr, cw_span), dtype=torch.int32, device=src_dev)
                tr_scale = torch.empty((r_len_tr, ng_span), dtype=torch.bfloat16, device=src_dev)
                tr_zero = torch.empty((r_len_tr, ng_span), dtype=torch.bfloat16, device=src_dev)
                cw_off = 0
                ng_off = 0
                for tc in range(tc0, tc1):
                    t_cubes = by_tile_cubes[(tr, tc)]
                    t_meta = q1_mat.spatial_tiles[(tr, tc)]
                    cw_t = t_cubes[0].bits_i32.shape[1]
                    ng_t = t_meta.qscale.shape[1]
                    for p_idx, cb in enumerate(t_cubes):
                        tr_planes[p_idx, :, cw_off : cw_off + cw_t].copy_(cb.bits_i32)
                    tr_scale[:, ng_off : ng_off + ng_t].copy_(t_meta.qscale)
                    tr_zero[:, ng_off : ng_off + ng_t].copy_(t_meta.qzero)
                    cw_off += cw_t
                    ng_off += ng_t

            # Split row tile into sub-chunks if r_len_tr > max_rows, or merge when room remains
            local_off = 0
            while local_off < r_len_tr:
                if pending_rows == max_rows:
                    _flush_pending()
                room = max_rows - pending_rows
                take = min(room, r_len_tr - local_off)
                if batch_r0 is None:
                    batch_r0 = r0_tr + local_off
                batch_r1 = r0_tr + local_off + take
                pending_planes.append(tr_planes[:, local_off : local_off + take, :])
                pending_scales.append(tr_scale[local_off : local_off + take, :])
                pending_zeros.append(tr_zero[local_off : local_off + take, :])
                pending_rows += take
                local_off += take

        _flush_pending()
    return batches


def _execute_q1_batches_on_arena(
    flat_x: torch.Tensor,
    arena: Q1RoleCycleArena,
    batches: List[Q1PackedBatch],
    y_acc: torch.Tensor,
    *,
    p_act: int,
    total_planes: int,
    group_size: int,
    row_offset: int = 0,
    midpoint_correct: bool = True,
    use_nvrtc: bool = True,
    use_popc: bool = False,
) -> int:
    """Stream maximal Q1PackedBatch chunks with 2-stream overlap & 1 fused NVRTC launch per batch."""
    if not batches or flat_x.shape[0] == 0:
        return 0
    device = flat_x.device
    m = flat_x.shape[0]
    use_cuda_nvrtc = use_nvrtc and device.type == "cuda" and flat_x.dtype == torch.bfloat16
    x_c = flat_x.contiguous() if use_cuda_nvrtc else flat_x.float()
    bit_shifts = None if use_cuda_nvrtc else torch.arange(32, dtype=torch.int64, device=device)
    tail_mid = (
        0.5 * ((1 << (total_planes - p_act)) - 1)
        if (midpoint_correct and p_act < total_planes)
        else 0.0
    )

    if use_cuda_nvrtc:
        from src.quintessentia_q1_cube_nvrtc import launch_q1_multiplane_gemv

    arena.stage_batch_into_B(batches[0], p_act)
    n_batches = len(batches)
    for idx in range(n_batches):
        arena.rotate()
        cur_batch, transfer_ev = arena.buffers[arena.active_area]
        if idx + 1 < n_batches:
            arena.stage_batch_into_B(batches[idx + 1], p_act)

        if use_cuda_nvrtc:
            compute_stream = torch.cuda.current_stream(device)
            if transfer_ev is not None:
                compute_stream.wait_event(transfer_ev)
                cur_batch.planes_u32.record_stream(compute_stream)
                cur_batch.qscale.record_stream(compute_stream)
                cur_batch.qzero.record_stream(compute_stream)
            launch_q1_multiplane_gemv(
                x_c,
                cur_batch.planes_u32,
                cur_batch.qscale,
                cur_batch.qzero,
                y_acc,
                r0=cur_batch.r0 - row_offset,
                c0=cur_batch.c0,
                c_actual=cur_batch.c_actual,
                group_size=group_size,
                total_planes=total_planes,
                add_zero=True,
                use_popc=use_popc,
                midpoint_correct=midpoint_correct,
            )
            comp_ev = torch.cuda.Event(enable_timing=False)
            comp_ev.record(compute_stream)
            arena._slot_compute_events[arena.active_area] = comp_ev
            arena.nvrtc_batch_launches += 1
        else:
            r0 = cur_batch.r0 - row_offset
            r1 = cur_batch.r1 - row_offset
            r_len = r1 - r0
            c0, c1, c_actual = cur_batch.c0, cur_batch.c1, cur_batch.c_actual
            c_words = int(cur_batch.planes_u32.shape[2])
            x_tile = x_c[:, c0:c1]
            if c_words * 32 > c_actual:
                x_pad = torch.zeros((m, c_words * 32), dtype=torch.float32, device=device)
                x_pad[:, :c_actual] = x_tile
                x_grp = x_pad.reshape(m, -1, group_size)
            else:
                x_grp = x_tile.reshape(m, -1, group_size)

            sc_f = cur_batch.qscale.float()
            z_eff = cur_batch.qzero.float() + (sc_f * tail_mid if tail_mid != 0.0 else 0.0)
            y_acc[:, r0:r1] += x_grp.sum(dim=-1) @ z_eff.t()
            for p_idx in range(p_act):
                alpha = float(1 << (total_planes - 1 - p_idx))
                bits_i64 = cur_batch.planes_u32[p_idx].to(torch.int64) & 0xFFFFFFFF
                b_k = ((bits_i64.unsqueeze(-1) >> bit_shifts) & 1).reshape(r_len, -1, group_size).float()
                y_acc[:, r0:r1] += alpha * (torch.einsum("mgs,rgs->mrg", x_grp, b_k) * sc_f.unsqueeze(0)).sum(dim=-1)
    return n_batches


class Q1StreamedLinear(nn.Module):
    """Streamed 2D Q1 linear module executing maximal batches under Q1RoleCycleArena."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        q1_mat: Optional[Q1CubeMatrix] = None,
        packed_dict: Optional[Dict[str, torch.Tensor]] = None,
        read_rows: Optional[Callable[[int, int], torch.Tensor]] = None,
        bits: int = 4,
        group_size: int = 64,
        r_tile: int = 64,
        c_tile: int = 128,
        bias: Optional[torch.Tensor] = None,
        slot_budget_bytes: int = 2 * 1024**2,
        arena: Optional[Q1RoleCycleArena] = None,
        active_planes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.r_tile = int(r_tile)
        self.c_tile = int(c_tile)
        self.slot_budget_bytes = int(slot_budget_bytes)
        self.active_planes: Optional[int] = active_planes

        if q1_mat is not None:
            self.q1_mat = q1_mat
            self.bits = q1_mat.planes
            self.group_size = q1_mat.group_size
            self.r_tile = q1_mat.r_tile
            self.c_tile = q1_mat.c_tile
        elif packed_dict is not None:
            self.q1_mat = q1_cubes_from_packed(
                packed_dict, bits=self.bits, group_size=self.group_size,
                r_tile=self.r_tile, c_tile=self.c_tile,
            )
        elif read_rows is not None:
            self.q1_mat = _pack_q1_matrix_from_row_reader(
                self.out_features, self.in_features, read_rows,
                planes=self.bits, r_tile=self.r_tile, c_tile=self.c_tile,
                group_size=self.group_size, device="cpu",
            )
        else:
            raise ValueError("Provide one of q1_mat, packed_dict, or read_rows")

        if tuple(self.q1_mat.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"Q1CubeMatrix shape {self.q1_mat.shape} != expected {(self.out_features, self.in_features)}"
            )

        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

        src_dev = next(iter(self.q1_mat.spatial_tiles.values())).qscale.device if self.q1_mat.spatial_tiles else torch.device("cpu")
        self._owns_arena = arena is None
        self.arena = arena if arena is not None else Q1RoleCycleArena(self.slot_budget_bytes, src_dev)
        self._batch_budget = min(self.slot_budget_bytes, self.arena.slot_budget_bytes)
        self._packed_batches = build_q1_packed_batches(self.q1_mat, self._batch_budget)
        self.calls = 0
        self.cubes_streamed = 0
        self.batches_executed = 0

    @classmethod
    def from_packed_linear(
        cls,
        mod: Any,
        *,
        slot_budget_bytes: int = 2 * 1024**2,
        arena: Optional[Q1RoleCycleArena] = None,
        r_tile: int = 64,
        c_tile: int = 128,
    ) -> "Q1StreamedLinear":
        packed = mod.packed()
        if any(t.device.type == "meta" for t in packed.values()):
            raise ValueError("Cannot build Q1StreamedLinear from meta-device PackedLinear")
        bias = mod.bias.detach() if getattr(mod, "bias", None) is not None else None
        return cls(
            mod.in_features,
            mod.out_features,
            packed_dict=packed,
            bits=mod.bits,
            group_size=mod.group_size,
            r_tile=r_tile,
            c_tile=c_tile,
            bias=bias,
            slot_budget_bytes=slot_budget_bytes,
            arena=arena,
            active_planes=getattr(mod, "active_planes", None),
        )

    def _ensure_arena_device(self, device: torch.device) -> Q1RoleCycleArena:
        if self.arena.device != device and self._owns_arena:
            self.arena = Q1RoleCycleArena(self.slot_budget_bytes, device)
        return self.arena

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        flat_x = x.reshape(-1, self.in_features)
        m = flat_x.shape[0]
        device = x.device
        arena = self._ensure_arena_device(device)

        p_act = self.q1_mat.planes if self.active_planes is None else int(self.active_planes)
        if p_act < 1 or p_act > self.q1_mat.planes:
            raise ValueError(f"active_planes ({p_act}) must be in [1, {self.q1_mat.planes}]")

        y_acc = torch.zeros((m, self.out_features), dtype=torch.float32, device=device)
        n_batches = _execute_q1_batches_on_arena(
            flat_x, arena, self._packed_batches, y_acc,
            p_act=p_act, total_planes=self.q1_mat.planes, group_size=self.group_size,
            midpoint_correct=True, use_nvrtc=True, use_popc=False,
        )
        if self.bias is not None:
            y_acc += self.bias.to(device=device, dtype=torch.float32).unsqueeze(0)

        self.calls += 1
        self.batches_executed += n_batches
        self.cubes_streamed += n_batches
        return y_acc.to(x.dtype).reshape(*orig_shape[:-1], self.out_features)

    def stats(self) -> Dict[str, object]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bits": self.bits,
            "group_size": self.group_size,
            "active_planes": self.active_planes,
            "calls": self.calls,
            "batch_count": len(self._packed_batches),
            "batches_executed": self.batches_executed,
            "cubes_streamed": self.cubes_streamed,
            "q1_total_bytes": self.q1_mat.total_bytes,
            "bf16_bytes": self.q1_mat.bf16_bytes,
            "arena": self.arena.stats(),
        }


class Q1StreamedLMHead(nn.Module):
    """Sub-vocabulary streamed LM head with bounded Q1 arena, 2-stream overlap & candidate projection."""

    def __init__(
        self,
        in_features: int,
        vocab_size: int,
        *,
        q1_mat: Optional[Q1CubeMatrix] = None,
        packed_dict: Optional[Dict[str, torch.Tensor]] = None,
        read_rows: Optional[Callable[[int, int], torch.Tensor]] = None,
        weight_slice: Any = None,
        bits: int = 4,
        group_size: int = 64,
        r_tile: int = 256,
        c_tile: int = 128,
        bias: Optional[torch.Tensor] = None,
        slot_budget_bytes: int = 4 * 1024**2,
        arena: Optional[Q1RoleCycleArena] = None,
        active_planes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.vocab_size = int(vocab_size)
        self.out_features = self.vocab_size
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.r_tile = int(r_tile)
        self.c_tile = int(c_tile)
        self.slot_budget_bytes = int(slot_budget_bytes)
        self.active_planes: Optional[int] = active_planes

        if q1_mat is not None:
            self.q1_mat = q1_mat
            self.bits = q1_mat.planes
            self.group_size = q1_mat.group_size
            self.r_tile = q1_mat.r_tile
            self.c_tile = q1_mat.c_tile
        elif packed_dict is not None:
            self.q1_mat = q1_cubes_from_packed(
                packed_dict, bits=self.bits, group_size=self.group_size,
                r_tile=self.r_tile, c_tile=self.c_tile,
            )
        elif read_rows is not None or weight_slice is not None:
            reader = read_rows if read_rows is not None else (lambda s, e: weight_slice[s:e])
            self.q1_mat = _pack_q1_matrix_from_row_reader(
                self.vocab_size, self.in_features, reader,
                planes=self.bits, r_tile=self.r_tile, c_tile=self.c_tile,
                group_size=self.group_size, device="cpu",
            )
        else:
            raise ValueError("Provide one of q1_mat, packed_dict, read_rows, or weight_slice")

        if tuple(self.q1_mat.shape) != (self.vocab_size, self.in_features):
            raise ValueError(
                f"Q1CubeMatrix shape {self.q1_mat.shape} != expected {(self.vocab_size, self.in_features)}"
            )

        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

        src_dev = next(iter(self.q1_mat.spatial_tiles.values())).qscale.device if self.q1_mat.spatial_tiles else torch.device("cpu")
        self._owns_arena = arena is None
        self.arena = arena if arena is not None else Q1RoleCycleArena(self.slot_budget_bytes, src_dev)
        self._batch_budget = min(self.slot_budget_bytes, self.arena.slot_budget_bytes)
        self._packed_batches = build_q1_packed_batches(self.q1_mat, self._batch_budget)
        self._cand_tr_batches: Dict[int, List[Q1PackedBatch]] = {}
        self._n_rtiles = math.ceil(self.vocab_size / self.r_tile)
        self.calls = 0
        self.candidate_calls = 0
        self.batches_executed = 0
        self.cubes_streamed = 0
        self.row_tiles_skipped = 0

    def _ensure_arena_device(self, device: torch.device) -> Q1RoleCycleArena:
        if self.arena.device != device and self._owns_arena:
            self.arena = Q1RoleCycleArena(self.slot_budget_bytes, device)
        return self.arena

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        flat_x = x.reshape(-1, self.in_features)
        m = flat_x.shape[0]
        device = x.device
        arena = self._ensure_arena_device(device)

        p_act = self.q1_mat.planes if self.active_planes is None else int(self.active_planes)
        if p_act < 1 or p_act > self.q1_mat.planes:
            raise ValueError(f"active_planes ({p_act}) must be in [1, {self.q1_mat.planes}]")

        y_acc = torch.zeros((m, self.vocab_size), dtype=torch.float32, device=device)
        n_batches = _execute_q1_batches_on_arena(
            flat_x, arena, self._packed_batches, y_acc,
            p_act=p_act, total_planes=self.q1_mat.planes, group_size=self.group_size,
            midpoint_correct=True, use_nvrtc=True, use_popc=False,
        )
        if self.bias is not None:
            y_acc += self.bias.to(device=device, dtype=torch.float32).unsqueeze(0)

        self.calls += 1
        self.batches_executed += n_batches
        self.cubes_streamed += n_batches
        return y_acc.to(x.dtype).reshape(*orig_shape[:-1], self.vocab_size)

    @torch.no_grad()
    def project_candidates(self, x: torch.Tensor, cand_ids: torch.Tensor) -> torch.Tensor:
        """Compute FP32 logits [..., C] for cand_ids ([C] or [1, C]) touching only needed row tiles."""
        orig_shape = x.shape
        flat_x = x.reshape(-1, self.in_features)
        m = flat_x.shape[0]
        device = x.device
        ids = cand_ids.reshape(-1).to(device=device, dtype=torch.long)
        num_cands = int(ids.numel())
        if num_cands == 0:
            return torch.empty((*orig_shape[:-1], 0), dtype=torch.float32, device=device)
        if bool((ids < 0).any().item()) or bool((ids >= self.vocab_size).any().item()):
            raise IndexError("Candidate token id out of LM head vocabulary bounds")

        arena = self._ensure_arena_device(device)
        p_act = self.q1_mat.planes if self.active_planes is None else int(self.active_planes)
        if p_act < 1 or p_act > self.q1_mat.planes:
            raise ValueError(f"active_planes ({p_act}) must be in [1, {self.q1_mat.planes}]")

        ids_cpu = ids.tolist()
        tr_to_positions: Dict[int, Tuple[List[int], List[int]]] = {}
        for pos, cid in enumerate(ids_cpu):
            tr = cid // self.r_tile
            local_r = cid - tr * self.r_tile
            bucket = tr_to_positions.get(tr)
            if bucket is None:
                tr_to_positions[tr] = ([pos], [local_r])
            else:
                bucket[0].append(pos)
                bucket[1].append(local_r)

        self.row_tiles_skipped += max(0, self._n_rtiles - len(tr_to_positions))
        out_f32 = torch.empty((m, num_cands), dtype=torch.float32, device=device)

        for tr in sorted(tr_to_positions):
            positions, local_rows = tr_to_positions[tr]
            tr_batches = self._cand_tr_batches.get(tr)
            if tr_batches is None:
                tr_batches = build_q1_packed_batches(
                    self.q1_mat, self._batch_budget, tile_r_filter=tr,
                )
                self._cand_tr_batches[tr] = tr_batches
            if not tr_batches:
                continue
            r0_tr = tr * self.r_tile
            r_len = min(self.r_tile, self.vocab_size - r0_tr)
            y_tile = torch.zeros((m, r_len), dtype=torch.float32, device=device)
            n_b = _execute_q1_batches_on_arena(
                flat_x, arena, tr_batches, y_tile,
                p_act=p_act, total_planes=self.q1_mat.planes, group_size=self.group_size,
                row_offset=r0_tr, midpoint_correct=True, use_nvrtc=True, use_popc=False,
            )
            self.batches_executed += n_b
            self.cubes_streamed += n_b
            idx_local = torch.tensor(local_rows, dtype=torch.long, device=device)
            idx_pos = torch.tensor(positions, dtype=torch.long, device=device)
            out_f32[:, idx_pos] = y_tile.index_select(1, idx_local)

        if self.bias is not None:
            out_f32 += self.bias.to(device=device, dtype=torch.float32).index_select(0, ids).unsqueeze(0)

        self.candidate_calls += 1
        return out_f32.reshape(*orig_shape[:-1], num_cands)

    def stats(self) -> Dict[str, object]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "vocab_size": self.vocab_size,
            "bits": self.bits,
            "group_size": self.group_size,
            "active_planes": self.active_planes,
            "calls": self.calls,
            "candidate_calls": self.candidate_calls,
            "batch_count": len(self._packed_batches),
            "batches_executed": self.batches_executed,
            "cubes_streamed": self.cubes_streamed,
            "row_tiles_skipped": self.row_tiles_skipped,
            "q1_total_bytes": self.q1_mat.total_bytes,
            "bf16_bytes": self.q1_mat.bf16_bytes,
            "arena": self.arena.stats(),
        }
