"""Binary Halving Q1 Tree (`Q16 -> 2xQ8 -> 4xQ4 -> 8xQ2 -> 16xQ1`) over AirLLM layers.

Method 1 ("slice upfront"): recursively splits each layer's weight matrices in half by
bit-width until every leaf is a coarse, dense 1-bit `Q1` cube. Leaf count grows in a
compact geometric progression (`2^d`: 4 cubes for Q4, 8 for Q8, 16 for Q16) rather than
millions of tiny spatial tiles, enabling fast inter-cube geometry discovery and direct
execution on our Ampere 1-bit Tensor Core (`b1.mma.and.popc`) and FWHT GEMV NVRTC kernels.
"""
from __future__ import annotations

import math
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from src.ideal_q1_engine import fwht_block, sinkhorn_equilibrate


_PACK_WEIGHTS_CACHE: Dict[str, torch.Tensor] = {}


def _bit_pack_weights(device: torch.device) -> torch.Tensor:
    key = str(device)
    t = _PACK_WEIGHTS_CACHE.get(key)
    if t is None:
        t = (1 << torch.arange(8, device=device, dtype=torch.int32)).to(torch.uint8)
        _PACK_WEIGHTS_CACHE[key] = t
    return t


def _pack_sign_bits_u8(signs_bool: torch.Tensor) -> torch.Tensor:
    """Pack boolean sign matrix [N, K] (K % 8 == 0) into uint8 [N, K // 8] (LSB-first)."""
    n, k = signs_bool.shape
    bw = _bit_pack_weights(signs_bool.device)
    bits = signs_bool.reshape(n, k // 8, 8).to(torch.uint8)
    return (bits * bw).sum(dim=-1, dtype=torch.int32).to(torch.uint8)


def _plane_scale_f32(qscale: torch.Tensor, p: int) -> torch.Tensor:
    """Return float32 scale for bit-plane `p`, supporting shared [1, N, NG] and per-plane [P, N, NG]."""
    if int(qscale.shape[0]) == 1 and p > 0:
        return qscale[0].to(torch.float32) * float(2.0 ** (-p))
    return qscale[p].to(torch.float32)


def _unpack_sign_bits_f32(packed_u8: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 [N, K // 8] into {-1.0, +1.0} float32 [N, K]."""
    n, k8 = packed_u8.shape
    shifts = torch.arange(8, device=packed_u8.device, dtype=torch.uint8)
    bits = ((packed_u8.unsqueeze(-1) >> shifts) & 1).reshape(n, k8 * 8)
    return bits.to(torch.float32).mul_(2.0).sub_(1.0)


def _positional_grid(u_g: torch.Tensor, root_bits: int):
    """Choose the grid in the same BF16 metadata used by every consumer.

    Round the positive shared scale upward so all source values fit around
    the *stored* origin. Lower plane scales are exact powers of two of this
    shared scale; rounding a different grid after digit selection leaves an
    error floor even when every plane is retained.
    """
    qzero = ((u_g.amin(-1) + u_g.amax(-1)) * 0.5).to(torch.bfloat16)
    centered = u_g - qzero.float().unsqueeze(-1)
    radius = centered.abs().amax(-1)
    desired = radius / (2.0 * (1.0 - 2.0 ** (-root_bits)))
    base = desired.to(torch.bfloat16)
    # Positive BF16 encodings are ordered integers, including zero/subnormals.
    rounded_down = base.float() < desired
    bits = base.view(torch.int16).to(torch.int32) + rounded_down.to(torch.int32)
    base = bits.to(torch.int16).view(torch.bfloat16).contiguous()
    step = base.float() * float(2.0 ** (2 - root_bits))
    return qzero.contiguous(), base, step, centered


@dataclass
class BinaryHalvingTreeMeta:
    root_bits: int
    depth: int
    law: str
    group_size: int
    use_fwht: bool
    level_node_counts: Tuple[int, ...]
    leaf_paths: Tuple[str, ...]
    leaf_energy_frac: Tuple[float, ...]
    prefix_rel_rmse: Tuple[float, ...]
    prefix_cos: Tuple[float, ...]
    source_root_bits: Optional[int] = None


def _halve_residual_recursive(
    r_cur: torch.Tensor,
    cur_bits: int,
    path: str,
    group_size: int,
) -> Tuple[List[Tuple[str, torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Recursively split residual tensor `r_cur` [N, K] in half: cur_bits -> 2 x (cur_bits // 2) -> ... -> Q1."""
    if cur_bits == 1:
        n, k = r_cur.shape
        r_g = r_cur.reshape(n, k // group_size, group_size)
        sign_bool = r_cur >= 0
        alpha = r_g.abs().mean(dim=-1).to(torch.bfloat16)
        packed = _pack_sign_bits_u8(sign_bool)
        sign_f = sign_bool.to(torch.float32).mul_(2.0).sub_(1.0)
        recon = (sign_f.reshape(n, k // group_size, group_size) * alpha.to(torch.float32).unsqueeze(-1)).reshape(n, k)
        return [(path, packed, alpha)], recon

    half = cur_bits // 2
    left_leaves, recon_left = _halve_residual_recursive(r_cur, half, path + "L", group_size)
    # Lossless tree carry: pass the exact residual after left half into right half.
    r_right = r_cur - recon_left
    right_leaves, recon_right = _halve_residual_recursive(r_right, half, path + "R", group_size)
    return left_leaves + right_leaves, recon_left + recon_right


def _halve_positional_recursive(
    q_int: torch.Tensor,
    cur_bits: int,
    bit_shift_base: int,
    base_scale: torch.Tensor,
    path: str,
) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
    """Recursively split integer codes `q_int` in [0, 2^cur_bits - 1] in half until Q1 leaves.

    Uses signed binary basis `q = sum_{p} (2 * b_p - 1) * 2^(B - 1 - p) * (base_scale / 2) + center`,
    so every leaf has exact bit plane `b_p in {-1, +1}` and geometric scale `base_scale * 2^(shift - 1)`.
    """
    if cur_bits == 1:
        sign_bool = (q_int & 1) != 0
        packed = _pack_sign_bits_u8(sign_bool)
        scale_p = (base_scale * float(2.0 ** (bit_shift_base - 1))).to(torch.bfloat16)
        return [(path, packed, scale_p)]

    half = cur_bits // 2
    hi = q_int >> half
    lo = q_int & ((1 << half) - 1)
    left = _halve_positional_recursive(hi, half, bit_shift_base + half, base_scale, path + "L")
    right = _halve_positional_recursive(lo, half, bit_shift_base, base_scale, path + "R")
    return left + right


@torch.no_grad()
def halve_weight_to_q1_tree(
    weight: torch.Tensor,
    *,
    root_bits: int = 8,
    group_size: int = 64,
    law: str = "positional_halving",
    use_fwht: bool = True,
    sinkhorn_iters: int = 3,
    compute_diagnostics: bool = True,
) -> Tuple[Dict[str, torch.Tensor], BinaryHalvingTreeMeta]:
    """Slice 2D weight matrix [N, K] into a binary halving tree of `root_bits` coarse Q1 cubes.

    Supported `root_bits`: 2, 4, 8, 16 (levels: 1 -> 2 -> 4 -> 8 -> 16 Q1 leaves).
    Supported `law`:
      - `"residual_halving"`: recursive residual halving `W^(2m) = W_L^(m) + W_R^(m)` down to Q1.
      - `"positional_halving"`: recursive bit-half splitting `q^(2m) = 2^m * q_hi + q_lo` down to Q1.
    """
    if root_bits not in (2, 4, 8, 16):
        raise ValueError(f"root_bits must be a power of 2 in {{2, 4, 8, 16}}, got {root_bits}")
    if group_size not in (16, 32, 64):
        raise ValueError(f"group_size must be in {{16, 32, 64}}, got {group_size}")
    if law not in ("residual_halving", "positional_halving"):
        raise ValueError(f"Unsupported halving law {law!r}")
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got {tuple(weight.shape)}")
    n, k = int(weight.shape[0]), int(weight.shape[1])
    if k % group_size != 0:
        raise ValueError(f"in_features ({k}) must be divisible by group_size ({group_size})")

    dev = weight.device
    depth = int(math.log2(root_bits))
    level_counts = tuple(1 << d for d in range(depth + 1))

    if use_fwht:
        row_scale, col_scale, w_eq = sinkhorn_equilibrate(weight, iters=sinkhorn_iters)
        u = fwht_block(w_eq, group_size).to(torch.float32)
    else:
        row_scale = torch.ones((n,), dtype=torch.bfloat16, device=dev)
        col_scale = torch.ones((k,), dtype=torch.bfloat16, device=dev)
        u = weight.to(torch.float32)

    ng = k // group_size
    u_g = u.reshape(n, ng, group_size)

    if law == "residual_halving":
        qzero = u_g.mean(dim=-1).to(torch.bfloat16)
        r0 = (u_g - qzero.to(torch.float32).unsqueeze(-1)).reshape(n, k)
        leaves, _ = _halve_residual_recursive(r0, root_bits, "", group_size)
    else:
        # Positional binary halving over symmetric [-max_abs, +max_abs] centered at group mean
        # mapped onto exact signed-binary code q in [0, 2^B - 1]:
        # u_approx = z_mid + step * (q - (2^B - 1)/2) = z_mid + sum_p (2*b_p - 1) * (step/2) * 2^(B-1-p)
        qzero, base_scale, step, centered = _positional_grid(u_g, root_bits)
        # Form the complete integer code with the same residual decisions as
        # Method 2. A single float32 division/offset rounds near cell boundaries
        # at B=16 and can choose a different last digit after FWHT.
        remaining = centered.clone()
        q_int = torch.zeros_like(centered, dtype=torch.int32)
        for p in range(root_bits):
            positive = remaining >= 0
            q_int = (q_int << 1) | positive.to(torch.int32)
            sign = positive.to(torch.float32).mul_(2.0).sub_(1.0)
            remaining.sub_(sign * (base_scale.float() * float(2.0 ** (-p))).unsqueeze(-1))
        q_int = q_int.reshape(n, k)
        leaves = _halve_positional_recursive(q_int, root_bits, 0, step, "")

    leaf_paths = tuple(p for p, _, _ in leaves)
    packed_lo = torch.stack([pk for _, pk, _ in leaves], dim=0).contiguous()  # [P, N, K // 8]
    if law == "positional_halving":
        # Shared geometric scale qscale[0] -> plane p uses qscale[0] * 2^(-p) (saves (P-1)*0.25 bpw!)
        qscale = leaves[0][2].unsqueeze(0).contiguous()                      # [1, N, K // GS]
    else:
        qscale = torch.stack([sc for _, _, sc in leaves], dim=0).contiguous()  # [P, N, K // GS]

    leaf_energy_frac: List[float] = []
    prefix_rel_rmse: List[float] = []
    prefix_cos: List[float] = []
    if compute_diagnostics:
        r_ref = u
        ref_norm2 = float(r_ref.pow(2).sum().item()) + 1e-12
        ref_norm = math.sqrt(ref_norm2)
        acc = qzero.to(torch.float32).unsqueeze(-1).expand(n, ng, group_size).reshape(n, k).clone()
        for p in range(root_bits):
            s_f = _unpack_sign_bits_f32(packed_lo[p]).reshape(n, ng, group_size)
            contrib = (s_f * _plane_scale_f32(qscale, p).unsqueeze(-1)).reshape(n, k)
            leaf_energy_frac.append(round(float(contrib.pow(2).sum().item()) / ref_norm2, 6))
            acc.add_(contrib)
            diff_norm = float((r_ref - acc).pow(2).sum().sqrt().item())
            prefix_rel_rmse.append(round(diff_norm / ref_norm, 6))
            dot = float((r_ref * acc).sum().item())
            acc_norm = float(acc.pow(2).sum().sqrt().item()) + 1e-12
            prefix_cos.append(round(dot / (ref_norm * acc_norm), 6))

    tensors = {
        "packed_lo": packed_lo,
        "qscale": qscale,
        "qzero": qzero.contiguous(),
        "row_scale": row_scale.contiguous(),
        "col_scale": col_scale.contiguous(),
    }
    meta = BinaryHalvingTreeMeta(
        root_bits=root_bits,
        depth=depth,
        law=law,
        group_size=group_size,
        use_fwht=use_fwht,
        level_node_counts=level_counts,
        leaf_paths=leaf_paths,
        leaf_energy_frac=tuple(leaf_energy_frac),
        prefix_rel_rmse=tuple(prefix_rel_rmse),
        prefix_cos=tuple(prefix_cos),
    )
    return tensors, meta


class BinaryHalvingQ1Linear(nn.Module):
    """Linear operator backed by a Binary Halving Q1 Tree (`root_bits` coarse Q1 cubes)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tensors: Dict[str, torch.Tensor],
        meta: BinaryHalvingTreeMeta,
        *,
        bias: Optional[torch.Tensor] = None,
        layer_idx: int = -1,
        proj_name: str = "",
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.root_bits = int(meta.root_bits)
        self.group_size = int(meta.group_size)
        self.use_fwht = bool(meta.use_fwht)
        self.law = str(meta.law)
        self.layer_idx = int(layer_idx)
        self.proj_name = str(proj_name)
        self.meta = meta
        self.micro_agent_tol: Optional[float] = None
        self.micro_agent_calls = 0
        # M0 Pure-Math Differential Mirror Q1 GEMV (CUDA Kernel #6) on dense trunk projections
        self.mirror_diff_tol: Optional[float] = None
        self.mirror_diff_calls: int = 0
        self._mirror_x_last: Optional[torch.Tensor] = None
        self._mirror_y_last: Optional[torch.Tensor] = None

        self.register_buffer("packed_lo", tensors["packed_lo"])
        self.register_buffer("qscale", tensors["qscale"])
        self.register_buffer("qzero", tensors["qzero"])
        self.register_buffer("row_scale", tensors["row_scale"])
        self.register_buffer("col_scale", tensors["col_scale"])
        if bias is not None:
            self.register_buffer("bias", bias.detach().to(device=self.packed_lo.device, dtype=torch.bfloat16).contiguous())
        else:
            self.bias = None

        self.active_cubes: int = self.root_bits
        self.branch_stop_tol: Optional[float] = None
        self.record_geometry: bool = False

        # Runtime geometry telemetry
        self.calls: int = 0
        self.total_cubes_executed: int = 0
        self.cube_contrib_energy: List[float] = [0.0] * self.root_bits
        self.cube_rel_contrib_sum: List[float] = [0.0] * self.root_bits
        self.last_used_cubes: int = self.root_bits

    def reset_telemetry(self) -> None:
        self.calls = 0
        self.total_cubes_executed = 0
        self.cube_contrib_energy = [0.0] * self.root_bits
        self.cube_rel_contrib_sum = [0.0] * self.root_bits
        self.last_used_cubes = self.active_cubes

    def weight_bytes(self, cubes: Optional[int] = None) -> int:
        p = self.active_cubes if cubes is None else int(cubes)
        lo_per_cube = self.out_features * (self.in_features // 8)
        sc_per_plane = self.out_features * (self.in_features // self.group_size) * 2
        n_sc_planes = 1 if int(self.qscale.shape[0]) == 1 else p
        shared = (
            self.out_features * (self.in_features // self.group_size) * 2  # qzero
            + self.out_features * 2                                        # row_scale
            + self.in_features * 2                                         # col_scale
        )
        return p * lo_per_cube + n_sc_planes * sc_per_plane + shared

    def effective_bits_per_weight(self, cubes: Optional[int] = None) -> float:
        return (self.weight_bytes(cubes) * 8.0) / float(self.in_features * self.out_features)

    @torch.no_grad()
    def reconstruct_weight(self, cubes: Optional[int] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        p_use = self.active_cubes if cubes is None else max(1, min(self.root_bits, int(cubes)))
        n, k = self.out_features, self.in_features
        ng = k // self.group_size
        u = self.qzero.to(torch.float32).unsqueeze(-1).expand(n, ng, self.group_size).reshape(n, k).clone()
        for p in range(p_use):
            s_f = _unpack_sign_bits_f32(self.packed_lo[p]).reshape(n, ng, self.group_size)
            u.add_((s_f * _plane_scale_f32(self.qscale, p).unsqueeze(-1)).reshape(n, k))
        if self.use_fwht:
            w = fwht_block(u, self.group_size)
            w = w * self.row_scale.to(torch.float32).unsqueeze(1) * self.col_scale.to(torch.float32).unsqueeze(0)
        else:
            w = u
        return w.to(dtype)

    def _forward_cuda_fast(self, flat: torch.Tensor, p_use: int) -> torch.Tensor:
        from src.quintessentia_q1_cube_nvrtc import launch_ideal_q1_b1_mma, launch_ideal_q1_fwht_gemv
        y_out = torch.empty((flat.shape[0], self.out_features), dtype=torch.bfloat16, device=flat.device)
        if flat.shape[0] == 0:
            return y_out
        if (self.micro_agent_tol is not None and flat.shape[0] == 1 and self.group_size == 64
                and self.in_features >= 1024 and self.out_features >= 512):
            from src.mirror_micro_agents_nvrtc import launch_coupled_micro_agent_q1_reader
            launch_coupled_micro_agent_q1_reader(
                flat.contiguous(), self.packed_lo, self.qscale, self.qzero,
                self.row_scale, self.col_scale, y_out, max_planes=p_use,
                base_tol=self.micro_agent_tol, bias=self.bias)
            self.micro_agent_calls += 1
            return y_out
        if flat.shape[0] >= 8:
            launch_ideal_q1_b1_mma(
                flat.contiguous(),
                self.packed_lo,
                self.qscale,
                self.qzero,
                self.row_scale,
                self.col_scale,
                y_out,
                group_size=self.group_size,
                total_planes=self.root_bits,
                active_planes=p_use,
                bias=self.bias,
            )
        else:
            launch_ideal_q1_fwht_gemv(
                flat.contiguous(),
                self.packed_lo,
                self.qscale,
                self.qzero,
                self.row_scale,
                self.col_scale,
                y_out,
                group_size=self.group_size,
                total_planes=self.root_bits,
                plane_start=0,
                plane_count=p_use,
                bias=self.bias,
            )
        return y_out

    def _forward_stepwise_cubes(self, flat: torch.Tensor, p_max: int) -> torch.Tensor:
        """Compute outputs incrementally cube-by-cube to record geometry or early-stop along tree branches."""
        dev = flat.device
        m = flat.shape[0]
        if m == 0:
            return torch.empty((0, self.out_features), dtype=flat.dtype, device=dev)

        if self.use_fwht and dev.type == "cuda" and flat.dtype == torch.bfloat16:
            from src.quintessentia_q1_cube_nvrtc import launch_ideal_q1_fwht_gemv
            y_acc = torch.zeros((m, self.out_features), dtype=torch.float32, device=dev)
            y_delta = torch.empty((m, self.out_features), dtype=torch.float32, device=dev)
            used = 0
            for p in range(p_max):
                launch_ideal_q1_fwht_gemv(
                    flat.contiguous(),
                    self.packed_lo,
                    self.qscale,
                    self.qzero,
                    self.row_scale,
                    self.col_scale,
                    y_delta,
                    group_size=self.group_size,
                    total_planes=self.root_bits,
                    plane_start=p,
                    plane_count=1,
                    bias=self.bias if p == 0 else None,
                    accumulate=False,
                )
                y_acc.add_(y_delta)
                used = p + 1
                delta_e = float(y_delta.pow(2).sum().item())
                acc_e = float(y_acc.pow(2).sum().item()) + 1e-12
                rel = math.sqrt(delta_e / acc_e)
                if self.record_geometry:
                    self.cube_contrib_energy[p] += delta_e
                    self.cube_rel_contrib_sum[p] += rel
                if self.branch_stop_tol is not None and used >= 2 and rel < self.branch_stop_tol:
                    break
            self.last_used_cubes = used
            self.total_cubes_executed += used
            return y_acc.to(flat.dtype)

        # General CPU / non-FWHT stepwise path
        n, k = self.out_features, self.in_features
        ng = k // self.group_size
        x_f = flat.to(torch.float32)
        if self.use_fwht:
            x_rot = fwht_block(x_f * self.col_scale.to(torch.float32), self.group_size)
        else:
            x_rot = x_f
        r_scale = self.row_scale.to(torch.float32)
        x_gsum = x_rot.reshape(m, ng, self.group_size).sum(dim=-1)
        y_acc = F.linear(x_gsum, self.qzero.to(torch.float32)) * r_scale
        if self.bias is not None:
            y_acc = y_acc + self.bias.to(torch.float32)
        used = 0
        for p in range(p_max):
            s_f = _unpack_sign_bits_f32(self.packed_lo[p]).reshape(n, ng, self.group_size)
            w_p = (s_f * _plane_scale_f32(self.qscale, p).unsqueeze(-1)).reshape(n, k)
            y_delta = F.linear(x_rot, w_p) * r_scale
            y_acc = y_acc + y_delta
            used = p + 1
            delta_e = float(y_delta.pow(2).sum().item())
            acc_e = float(y_acc.pow(2).sum().item()) + 1e-12
            rel = math.sqrt(delta_e / acc_e)
            if self.record_geometry:
                self.cube_contrib_energy[p] += delta_e
                self.cube_rel_contrib_sum[p] += rel
            if self.branch_stop_tol is not None and used >= 2 and rel < self.branch_stop_tol:
                break
        self.last_used_cubes = used
        self.total_cubes_executed += used
        return y_acc.to(flat.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        p_use = max(1, min(self.root_bits, int(self.active_cubes)))
        flat = x.reshape(-1, self.in_features)
        if self.record_geometry or self.branch_stop_tol is not None:
            y = self._forward_stepwise_cubes(flat, p_use)
            return y.view(*x.shape[:-1], self.out_features)
        self.last_used_cubes = p_use
        self.total_cubes_executed += p_use
        if self.use_fwht and flat.device.type == "cuda" and flat.dtype == torch.bfloat16:
            if self.mirror_diff_tol is not None and self.group_size == 64:
                if flat.shape[0] == 1:
                    from src.pure_mirror_agents_nvrtc import launch_pure_mirror_diff_q1_gemv
                    y = launch_pure_mirror_diff_q1_gemv(
                        flat.contiguous(),
                        self.packed_lo,
                        self.qscale,
                        self.qzero,
                        self.row_scale,
                        self.col_scale,
                        x_last=self._mirror_x_last,
                        y_last=self._mirror_y_last,
                        bias=self.bias,
                        group_size=self.group_size,
                        max_planes=p_use,
                        min_planes=1,
                        mirror_tol=float(self.mirror_diff_tol),
                    )
                    self.mirror_diff_calls += 1
                    if self._mirror_x_last is None:
                        self._mirror_x_last = flat[0].detach().contiguous()
                        self._mirror_y_last = y[0].detach().contiguous()
                    return y.view(*x.shape[:-1], self.out_features)
                y = self._forward_cuda_fast(flat, p_use)
                if flat.shape[0] > 0:
                    self._mirror_x_last = flat[-1].detach().contiguous()
                    self._mirror_y_last = y[-1].detach().contiguous()
                return y.view(*x.shape[:-1], self.out_features)
            y = self._forward_cuda_fast(flat, p_use)
            return y.view(*x.shape[:-1], self.out_features)
        w = self.reconstruct_weight(p_use, dtype=x.dtype)
        b = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, w, b)


class BinaryHalvingMoEExperts(nn.Module):
    """MoE routed experts sliced into a Binary Halving Q1 Tree (`root_bits` Q1 cubes per expert).

    Supports both uniform `active_cubes` and router-weight-adaptive cube depth (giving top
    routing-weight experts deeper tree precision and weak tail experts fewer Q1 cubes).
    """

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        packed_lo: torch.Tensor,     # [E, P, N, K // 8]
        qscale: torch.Tensor,        # [E, P, N, K // GS]
        qzero: torch.Tensor,         # [E, N, K // GS]
        row_scale: torch.Tensor,     # [E, N]
        col_scale: torch.Tensor,     # [E, K]
        *,
        root_bits: int,
        group_size: int,
        use_fwht: bool = True,
        layer_idx: int = -1,
        proj_name: str = "",
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.root_bits = int(root_bits)
        self.group_size = int(group_size)
        self.use_fwht = bool(use_fwht)
        self.layer_idx = int(layer_idx)
        self.proj_name = str(proj_name)

        self.register_buffer("packed_lo", packed_lo)
        self.register_buffer("qscale", qscale)
        self.register_buffer("qzero", qzero)
        self.register_buffer("row_scale", row_scale)
        self.register_buffer("col_scale", col_scale)

        self.active_cubes: int = self.root_bits
        # Optional router-weight adaptive schedule: tuple of (min_weight_threshold, cubes)
        self.router_depth_schedule: Optional[Tuple[Tuple[float, int], ...]] = None
        self.expert_calls: int = 0
        self.expert_cubes_executed: int = 0

    def reset_telemetry(self) -> None:
        self.expert_calls = 0
        self.expert_cubes_executed = 0

    def cubes_for_weight(self, router_weight: float) -> int:
        if self.router_depth_schedule is None:
            return max(1, min(self.root_bits, int(self.active_cubes)))
        for thresh, cubes in self.router_depth_schedule:
            if router_weight >= thresh:
                return max(1, min(self.root_bits, int(cubes)))
        return 1

    def forward_expert(self, expert_idx: int, x: torch.Tensor, router_weight: float = 1.0) -> torch.Tensor:
        p_use = self.cubes_for_weight(router_weight)
        self.expert_calls += 1
        self.expert_cubes_executed += p_use
        flat = x.reshape(-1, self.in_features)
        if flat.shape[0] == 0:
            return torch.empty((*x.shape[:-1], self.out_features), dtype=x.dtype, device=x.device)
        if self.use_fwht and flat.device.type == "cuda" and flat.dtype == torch.bfloat16:
            from src.quintessentia_q1_cube_nvrtc import launch_ideal_q1_b1_mma, launch_ideal_q1_fwht_gemv
            y_out = torch.empty((flat.shape[0], self.out_features), dtype=torch.bfloat16, device=flat.device)
            if flat.shape[0] >= 8:
                launch_ideal_q1_b1_mma(
                    flat.contiguous(),
                    self.packed_lo[expert_idx],
                    self.qscale[expert_idx],
                    self.qzero[expert_idx],
                    self.row_scale[expert_idx],
                    self.col_scale[expert_idx],
                    y_out,
                    group_size=self.group_size,
                    total_planes=self.root_bits,
                    active_planes=p_use,
                    bias=None,
                )
            else:
                launch_ideal_q1_fwht_gemv(
                    flat.contiguous(),
                    self.packed_lo[expert_idx],
                    self.qscale[expert_idx],
                    self.qzero[expert_idx],
                    self.row_scale[expert_idx],
                    self.col_scale[expert_idx],
                    y_out,
                    group_size=self.group_size,
                    total_planes=self.root_bits,
                    plane_start=0,
                    plane_count=p_use,
                    bias=None,
                )
            return y_out.view(*x.shape[:-1], self.out_features)
        n, k = self.out_features, self.in_features
        ng = k // self.group_size
        u = self.qzero[expert_idx].to(torch.float32).unsqueeze(-1).expand(n, ng, self.group_size).reshape(n, k).clone()
        for p in range(p_use):
            s_f = _unpack_sign_bits_f32(self.packed_lo[expert_idx, p]).reshape(n, ng, self.group_size)
            u.add_((s_f * _plane_scale_f32(self.qscale[expert_idx], p).unsqueeze(-1)).reshape(n, k))
        w = fwht_block(u, self.group_size) * self.row_scale[expert_idx].float().unsqueeze(1) * self.col_scale[expert_idx].float().unsqueeze(0) if self.use_fwht else u
        return F.linear(x, w.to(x.dtype))


def _set_submodule(root: nn.Module, dotted: str, new_mod: nn.Module) -> None:
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


def _infer_layer_and_proj(full_name: str) -> Tuple[int, str]:
    parts = full_name.split(".")
    layer_idx = -1
    for i, tok in enumerate(parts):
        if tok == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            layer_idx = int(parts[i + 1])
            break
    return layer_idx, parts[-1]


def _is_router_module(full_name: str, mod: nn.Module) -> bool:
    leaf = full_name.rsplit(".", 1)[-1]
    if leaf in ("router", "gate", "shared_expert_gate") and ("moe" in full_name.lower() or "block_sparse" in full_name.lower() or "mlp" in full_name.lower()):
        # Distinguish MoE router (small out_features <= 512) from MLP gate_proj
        if getattr(mod, "out_features", 0) <= 512:
            return True
    return False


class BinaryHalvingModelSlicer:
    """Method 1 ("slice upfront"): slices all transformer layers into a compact binary tree of Q1 cubes."""

    def __init__(
        self,
        model: nn.Module,
        *,
        root_bits: int = 8,
        group_size: int = 64,
        law: str = "positional_halving",
        use_fwht: bool = True,
        sinkhorn_iters: int = 3,
        include_lm_head: bool = False,
        include_router: bool = False,
        include_moe_experts: bool = True,
    ):
        self.model = model
        self.root_bits = int(root_bits)
        self.group_size = int(group_size)
        self.law = str(law)
        self.use_fwht = bool(use_fwht)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.include_lm_head = bool(include_lm_head)
        self.include_router = bool(include_router)
        self.include_moe_experts = bool(include_moe_experts)

        self.modules: Dict[str, BinaryHalvingQ1Linear] = {}
        self.moe_blocks: List[Any] = []
        self.skipped: List[str] = []
        self.total_bf16_bytes: int = 0
        self.total_q1_leaf_cubes: int = 0

    @torch.no_grad()
    def slice_all(self, *, compute_diagnostics: bool = True) -> Dict[str, Any]:
        """Slice every eligible Linear layer (and Granite/HF MoE expert banks) upfront into Q1 trees."""
        candidates: List[Tuple[str, nn.Linear]] = []
        for name, mod in list(self.model.named_modules()):
            if not isinstance(mod, nn.Linear):
                continue
            if not self.include_lm_head and name.endswith("lm_head"):
                self.skipped.append(name)
                continue
            if not self.include_router and _is_router_module(name, mod):
                self.skipped.append(name)
                continue
            if mod.weight.is_meta or mod.in_features % self.group_size != 0:
                self.skipped.append(name)
                continue
            candidates.append((name, mod))

        for name, mod in candidates:
            layer_idx, proj_name = _infer_layer_and_proj(name)
            w = mod.weight.detach()
            self.total_bf16_bytes += w.numel() * 2
            tensors, meta = halve_weight_to_q1_tree(
                w,
                root_bits=self.root_bits,
                group_size=self.group_size,
                law=self.law,
                use_fwht=self.use_fwht,
                sinkhorn_iters=self.sinkhorn_iters,
                compute_diagnostics=compute_diagnostics,
            )
            q1_mod = BinaryHalvingQ1Linear(
                mod.in_features,
                mod.out_features,
                tensors,
                meta,
                bias=mod.bias,
                layer_idx=layer_idx,
                proj_name=proj_name,
            )
            _set_submodule(self.model, name, q1_mod)
            self.modules[name] = q1_mod
            self.total_q1_leaf_cubes += self.root_bits

        # Optional: 3D fused MoE expert banks (`gate_up_proj` [E, 2*I, H], `down_proj` [E, H, I])
        if self.include_moe_experts:
            for name, mod in list(self.model.named_modules()):
                gu_w = getattr(mod, "gate_up_proj", None)
                dn_w = getattr(mod, "down_proj", None)
                if (
                    isinstance(gu_w, torch.Tensor)
                    and isinstance(dn_w, torch.Tensor)
                    and gu_w.ndim == 3
                    and dn_w.ndim == 3
                    and not gu_w.is_meta
                    and gu_w.shape[2] % self.group_size == 0
                    and dn_w.shape[2] % self.group_size == 0
                ):
                    self._slice_3d_moe_experts(name, mod)

        return self.summary()

    @torch.no_grad()
    def _slice_3d_moe_experts(self, name: str, exp_mod: nn.Module) -> None:
        layer_idx, _ = _infer_layer_and_proj(name)
        gu_w = exp_mod.gate_up_proj.detach()   # [E, 2*I, H]
        dn_w = exp_mod.down_proj.detach()      # [E, H, I]
        num_e = int(gu_w.shape[0])
        self.total_bf16_bytes += (gu_w.numel() + dn_w.numel()) * 2

        def _pack_3d(w3: torch.Tensor, proj_label: str) -> BinaryHalvingMoEExperts:
            n_e, n_out, k_in = int(w3.shape[0]), int(w3.shape[1]), int(w3.shape[2])
            lo_list, sc_list, z_list, rs_list, cs_list = [], [], [], [], []
            for e in range(n_e):
                t, _ = halve_weight_to_q1_tree(
                    w3[e],
                    root_bits=self.root_bits,
                    group_size=self.group_size,
                    law=self.law,
                    use_fwht=self.use_fwht,
                    sinkhorn_iters=self.sinkhorn_iters,
                    compute_diagnostics=False,
                )
                lo_list.append(t["packed_lo"])
                sc_list.append(t["qscale"])
                z_list.append(t["qzero"])
                rs_list.append(t["row_scale"])
                cs_list.append(t["col_scale"])
            return BinaryHalvingMoEExperts(
                n_e, k_in, n_out,
                torch.stack(lo_list, dim=0).contiguous(),
                torch.stack(sc_list, dim=0).contiguous(),
                torch.stack(z_list, dim=0).contiguous(),
                torch.stack(rs_list, dim=0).contiguous(),
                torch.stack(cs_list, dim=0).contiguous(),
                root_bits=self.root_bits,
                group_size=self.group_size,
                use_fwht=self.use_fwht,
                layer_idx=layer_idx,
                proj_name=proj_label,
            )

        gu_bank = _pack_3d(gu_w, "moe_gate_up_proj")
        dn_bank = _pack_3d(dn_w, "moe_down_proj")
        self.total_q1_leaf_cubes += 2 * num_e * self.root_bits

        # Release original BF16 3D parameters and bind binary-halving Q1 forward
        del exp_mod.gate_up_proj
        del exp_mod.down_proj
        exp_mod.gu_bank = gu_bank
        exp_mod.dn_bank = dn_bank
        act_fn = getattr(exp_mod, "act_fn", F.silu)

        def _halving_experts_forward(
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            out_flat = torch.zeros_like(hidden_states)
            idx_cpu = top_k_index.detach().cpu().tolist()
            w_cpu = top_k_weights.detach().float().cpu().tolist()
            by_exp: Dict[int, Tuple[List[int], List[int], List[float]]] = {}
            for tok_i, (row_e, row_w) in enumerate(zip(idx_cpu, w_cpu)):
                for k_pos, (eid, rw) in enumerate(zip(row_e, row_w)):
                    e_int = int(eid)
                    if 0 <= e_int < num_e:
                        bucket = by_exp.get(e_int)
                        if bucket is None:
                            bucket = ([], [], [])
                            by_exp[e_int] = bucket
                        bucket[0].append(tok_i)
                        bucket[1].append(k_pos)
                        bucket[2].append(float(rw))
            for eid, (toks, kpos, rws) in by_exp.items():
                tok_t = torch.tensor(toks, dtype=torch.long, device=hidden_states.device)
                kpos_t = torch.tensor(kpos, dtype=torch.long, device=hidden_states.device)
                mean_rw = sum(rws) / max(1, len(rws))
                x_e = hidden_states.index_select(0, tok_t)
                gu = gu_bank.forward_expert(eid, x_e, router_weight=mean_rw)
                g_part, u_part = gu.chunk(2, dim=-1)
                act = act_fn(g_part) * u_part
                y_e = dn_bank.forward_expert(eid, act, router_weight=mean_rw)
                y_e = y_e * top_k_weights[tok_t, kpos_t, None]
                out_flat.index_add_(0, tok_t, y_e.to(out_flat.dtype))
            return out_flat

        exp_mod.forward = _halving_experts_forward
        self.moe_blocks.append((name, exp_mod, gu_bank, dn_bank))

    def set_uniform_cubes(self, cubes: int, *, moe_cubes: Optional[int] = None) -> None:
        c = max(1, min(self.root_bits, int(cubes)))
        for mod in self.modules.values():
            mod.active_cubes = c
            mod.branch_stop_tol = None
        mc = c if moe_cubes is None else max(1, min(self.root_bits, int(moe_cubes)))
        for _, _, in_b, out_b in self.moe_blocks:
            in_b.active_cubes = mc
            out_b.active_cubes = mc
            in_b.router_depth_schedule = None
            out_b.router_depth_schedule = None

    def set_branch_stop_tol(self, tol: Optional[float], *, max_cubes: Optional[int] = None) -> None:
        c = self.root_bits if max_cubes is None else max(1, min(self.root_bits, int(max_cubes)))
        for mod in self.modules.values():
            mod.active_cubes = c
            mod.branch_stop_tol = tol

    def set_moe_router_schedule(self, schedule: Optional[Tuple[Tuple[float, int], ...]]) -> None:
        for _, _, in_b, out_b in self.moe_blocks:
            in_b.router_depth_schedule = schedule
            out_b.router_depth_schedule = schedule

    def reset_telemetry(self) -> None:
        for mod in self.modules.values():
            mod.reset_telemetry()
        for _, _, in_b, out_b in self.moe_blocks:
            in_b.reset_telemetry()
            out_b.reset_telemetry()

    def summary(self) -> Dict[str, Any]:
        active_bytes = sum(m.weight_bytes() for m in self.modules.values())
        moe_active_bytes = 0
        for _, _, in_b, out_b in self.moe_blocks:
            for b in (in_b, out_b):
                lo_b = b.num_experts * b.active_cubes * b.out_features * (b.in_features // 8)
                sc_b = b.num_experts * b.active_cubes * b.out_features * (b.in_features // b.group_size) * 2
                sh_b = b.num_experts * (b.out_features * (b.in_features // b.group_size) * 2 + (b.out_features + b.in_features) * 2)
                moe_active_bytes += lo_b + sc_b + sh_b
        total_active = active_bytes + moe_active_bytes
        return {
            "root_bits": self.root_bits,
            "depth": int(math.log2(self.root_bits)),
            "law": self.law,
            "level_progression": [1 << d for d in range(int(math.log2(self.root_bits)) + 1)],
            "sliced_linear_modules": len(self.modules),
            "sliced_moe_blocks": len(self.moe_blocks),
            "total_q1_leaf_cubes": self.total_q1_leaf_cubes,
            "bf16_weight_mib": round(self.total_bf16_bytes / 1024**2, 2),
            "active_tree_mib": round(total_active / 1024**2, 2),
            "compression_vs_bf16": round(self.total_bf16_bytes / max(1, total_active), 2),
            "skipped_modules": len(self.skipped),
        }

    @torch.no_grad()
    def discover_inter_cube_geometry(self, run_forward_fn) -> Dict[str, Any]:
        """Run `run_forward_fn()` with cube-by-cube telemetry to map geometry across layers & projections."""
        self.reset_telemetry()
        for mod in self.modules.values():
            mod.active_cubes = self.root_bits
            mod.branch_stop_tol = None
            mod.record_geometry = True
        try:
            run_forward_fn()
        finally:
            for mod in self.modules.values():
                mod.record_geometry = False

        # Aggregate energy decay per cube index, per layer, and per projection family
        decay_sum = [0.0] * self.root_bits
        decay_cnt = 0
        by_layer: Dict[int, List[float]] = {}
        by_proj: Dict[str, List[float]] = {}
        module_tail_sensitivity: Dict[str, float] = {}

        for name, mod in self.modules.items():
            if mod.calls == 0:
                continue
            avg_rel = [v / mod.calls for v in mod.cube_rel_contrib_sum]
            # Tail sensitivity = residual relative energy still carried by cubes 1..root_bits-1
            tail_sens = sum(avg_rel[1:])
            module_tail_sensitivity[name] = tail_sens
            decay_cnt += 1
            for p in range(self.root_bits):
                decay_sum[p] += avg_rel[p]
            if mod.layer_idx >= 0:
                by_layer.setdefault(mod.layer_idx, []).append(tail_sens)
            by_proj.setdefault(mod.proj_name, []).append(tail_sens)

        mean_decay = [round(v / max(1, decay_cnt), 5) for v in decay_sum]
        layer_sens = {str(l): round(sum(vs) / len(vs), 5) for l, vs in sorted(by_layer.items())}
        proj_sens = {p: round(sum(vs) / len(vs), 5) for p, vs in sorted(by_proj.items())}

        # Measure inter-layer correlation of tail sensitivity (layer l vs l+1)
        sorted_layers = sorted(by_layer)
        adj_diffs = []
        for a, b in zip(sorted_layers, sorted_layers[1:]):
            va = sum(by_layer[a]) / len(by_layer[a])
            vb = sum(by_layer[b]) / len(by_layer[b])
            adj_diffs.append(abs(vb - va))

        # Rank projections & layers from highest to lowest sensitivity
        ranked_layers = sorted(((sum(vs) / len(vs), l) for l, vs in by_layer.items()), reverse=True)
        ranked_projs = sorted(((sum(vs) / len(vs), p) for p, vs in by_proj.items()), reverse=True)

        self._last_module_sensitivity = module_tail_sensitivity
        return {
            "cube_mean_rel_contrib": mean_decay,
            "layer_tail_sensitivity": layer_sens,
            "proj_tail_sensitivity": proj_sens,
            "top_sensitive_layers": [l for _, l in ranked_layers[:5]],
            "least_sensitive_layers": [l for _, l in ranked_layers[-5:]],
            "top_sensitive_projs": [p for _, p in ranked_projs[:4]],
            "mean_adjacent_layer_delta": round(sum(adj_diffs) / max(1, len(adj_diffs)), 5),
        }

    def apply_geometry_budget(self, target_avg_cubes: float, *, min_cubes: int = 1, max_cubes: Optional[int] = None) -> Dict[str, Any]:
        """Allocate Q1 cube depths per module using discovered inter-cube sensitivity geometry.

        Greedily invests extra Q1 cubes in modules with highest marginal error-reduction per parameter
        (measured from each module's live tail sensitivity * leaf energy fraction).
        """
        hi = self.root_bits if max_cubes is None else min(self.root_bits, int(max_cubes))
        lo = max(1, min(hi, int(min_cubes)))
        sens_map = getattr(self, "_last_module_sensitivity", {})
        total_params = sum(m.in_features * m.out_features for m in self.modules.values())
        if total_params == 0:
            return {"avg_cubes": 0.0, "histogram": {}}

        depths = {name: lo for name in self.modules}
        cur_budget_params = lo * total_params
        target_budget_params = float(target_avg_cubes) * total_params

        # Priority score per marginal step (d -> d+1): sensitivity * leaf_energy[d]
        while True:
            best_name = None
            best_score = -1.0
            for name, mod in self.modules.items():
                d = depths[name]
                if d >= hi:
                    continue
                n_params = mod.in_features * mod.out_features
                if cur_budget_params + n_params > target_budget_params + 1e-6:
                    continue
                leaf_e = mod.meta.leaf_energy_frac[d] if d < len(mod.meta.leaf_energy_frac) else (0.5 ** d)
                live_s = sens_map.get(name, 1.0)
                # First/last layers anchor representation geometry strongly
                layer_bonus = 1.35 if mod.layer_idx in (0, max((m.layer_idx for m in self.modules.values()), default=0)) else 1.0
                score = live_s * (leaf_e + 1e-6) * layer_bonus
                if score > best_score:
                    best_score = score
                    best_name = name
            if best_name is None:
                break
            depths[best_name] += 1
            cur_budget_params += self.modules[best_name].in_features * self.modules[best_name].out_features

        hist: Dict[int, int] = {}
        for name, d in depths.items():
            self.modules[name].active_cubes = d
            self.modules[name].branch_stop_tol = None
            hist[d] = hist.get(d, 0) + 1

        avg_actual = cur_budget_params / total_params
        return {
            "target_avg_cubes": round(float(target_avg_cubes), 3),
            "actual_param_weighted_avg_cubes": round(avg_actual, 3),
            "depth_histogram": {str(k): v for k, v in sorted(hist.items())},
        }


# ======================================================================================
# Method 2: AirLLM-style layer-by-layer streaming, carving Q1 cubes on the fly
# ======================================================================================


def _binary_tree_path(cube_idx: int, depth: int) -> str:
    return "".join("R" if ((cube_idx >> (depth - 1 - b)) & 1) else "L" for b in range(depth))


class IncrementalCubeCarver:
    """Carves one coarse Q1 cube at a time from a weight matrix along the binary halving tree.

    Mathematically equivalent to `positional_halving` (Method 1), but never computes or stores
    lower tree branches beyond the requested number of carved cubes `k <= root_bits`.
    """

    @torch.no_grad()
    def __init__(
        self,
        weight: torch.Tensor,
        *,
        root_bits: int = 8,
        group_size: int = 64,
        use_fwht: bool = True,
        sinkhorn_iters: int = 3,
        compute_diagnostics: bool = True,
    ):
        if root_bits not in (2, 4, 8, 16):
            raise ValueError(f"root_bits must be in {{2, 4, 8, 16}}, got {root_bits}")
        if group_size not in (16, 32, 64):
            raise ValueError(f"group_size must be in {{16, 32, 64}}, got {group_size}")
        if weight.ndim != 2 or int(weight.shape[1]) % group_size != 0:
            raise ValueError(f"Invalid weight shape {tuple(weight.shape)} for group_size={group_size}")

        self.n, self.k = int(weight.shape[0]), int(weight.shape[1])
        self.ng = self.k // group_size
        self.root_bits = int(root_bits)
        self.depth = int(math.log2(root_bits))
        self.group_size = int(group_size)
        self.use_fwht = bool(use_fwht)
        self.compute_diagnostics = bool(compute_diagnostics)

        dev = weight.device
        if use_fwht:
            self.row_scale, self.col_scale, w_eq = sinkhorn_equilibrate(weight, iters=sinkhorn_iters)
            u = fwht_block(w_eq, group_size).to(torch.float32)
        else:
            self.row_scale = torch.ones((self.n,), dtype=torch.bfloat16, device=dev)
            self.col_scale = torch.ones((self.k,), dtype=torch.bfloat16, device=dev)
            u = weight.to(torch.float32)

        u_g = u.reshape(self.n, self.ng, group_size)
        self.qzero, self._base_scale, self._step, self._rem = _positional_grid(u_g, self.root_bits)
        self._ref_norm = (float(u.pow(2).sum().sqrt().item()) + 1e-12
                          if self.compute_diagnostics else None)
        self._carved_lo: List[torch.Tensor] = []
        self._carved_sc: List[torch.Tensor] = []
        self._carved_paths: List[str] = []

    @property
    def carved_count(self) -> int:
        return len(self._carved_lo)

    @torch.no_grad()
    def carve_next_cube(self) -> Dict[str, Any]:
        """Carve exactly one more Q1 cube from the remaining residual, halving the error bound."""
        p = len(self._carved_lo)
        if p >= self.root_bits:
            raise StopIteration(f"All {self.root_bits} Q1 cubes have already been carved")
        scale_f = self._base_scale.float() * float(2.0 ** (-p))
        sign_bool = (self._rem >= 0).reshape(self.n, self.k)
        packed = _pack_sign_bits_u8(sign_bool)
        scale_bf16 = scale_f.to(torch.bfloat16)
        sign_f = sign_bool.reshape(self.n, self.ng, self.group_size).to(torch.float32).mul_(2.0).sub_(1.0)
        self._rem.sub_(sign_f * scale_f.unsqueeze(-1))
        path = _binary_tree_path(p, self.depth)
        self._carved_lo.append(packed)
        self._carved_sc.append(scale_bf16)
        self._carved_paths.append(path)
        rem_rel = (float(self._rem.pow(2).sum().sqrt().item()) / self._ref_norm
                   if self.compute_diagnostics else None)
        return {
            "cube_index": p,
            "path": path,
            "packed_lo": packed,
            "qscale": scale_bf16,
            "residual_rel_rmse": None if rem_rel is None else round(rem_rel, 6),
        }

    @torch.no_grad()
    def carve_until(self, num_cubes: int) -> Tuple[Dict[str, torch.Tensor], BinaryHalvingTreeMeta]:
        """Carve cube-by-cube until `num_cubes` Q1 cubes are materialized, then package into a tensor dict."""
        target = max(1, min(self.root_bits, int(num_cubes)))
        while len(self._carved_lo) < target:
            self.carve_next_cube()
        tensors = {
            "packed_lo": torch.stack(self._carved_lo[:target], dim=0).contiguous(),
            "qscale": self._carved_sc[0].unsqueeze(0).contiguous(),
            "qzero": self.qzero,
            "row_scale": self.row_scale.contiguous(),
            "col_scale": self.col_scale.contiguous(),
        }
        meta = BinaryHalvingTreeMeta(
            root_bits=target,
            depth=self.depth,
            law="incremental_positional_halving",
            group_size=self.group_size,
            use_fwht=self.use_fwht,
            level_node_counts=tuple(1 << d for d in range(self.depth + 1)),
            leaf_paths=tuple(self._carved_paths[:target]),
            leaf_energy_frac=(),
            prefix_rel_rmse=(),
            prefix_cos=(),
            source_root_bits=self.root_bits,
        )
        return tensors, meta

    @torch.no_grad()
    def carve_until_mirror_equal(
        self,
        x_probe: torch.Tensor,
        y_exact: torch.Tensor,
        max_cubes: int,
        *,
        min_cubes: int = 2,
        mirror_tol: float = 0.05,
    ) -> Tuple[Dict[str, torch.Tensor], BinaryHalvingTreeMeta, Dict[str, Any]]:
        """Activation-aware Mirror Crystal Carver (Left weight recursion x Right live activation recursion).

        Carves Q1 bit-planes one by one and tests convergence of the bracketless central stem
        `ObservedBehavior_(p-1) = Strategy_carve = ObservedBehavior_(p+1)` directly on the live
        activation subspace `x_probe` against the intact BF16 reference `y_exact`. Stops cleanly
        at `k_opt` once EQUAL is reached, returning the reversible crystal scar `Gamma_scar`.
        """
        target_max = max(1, min(self.root_bits, int(max_cubes)))
        target_min = max(1, min(target_max, int(min_cubes)))
        dev = self.row_scale.device
        x_1d = (x_probe[-1] if x_probe.ndim > 1 else x_probe).detach().to(device=dev, dtype=torch.float32)
        y_ref = (y_exact[-1] if y_exact.ndim > 1 else y_exact).detach().to(device=dev, dtype=torch.float32)

        x_sc = x_1d * self.col_scale.float()
        x_rot = fwht_block(x_sc.unsqueeze(0), self.group_size).squeeze(0).to(torch.float32) if self.use_fwht else x_sc
        x_rot_g = x_rot.reshape(self.ng, self.group_size)
        x_sum = x_rot_g.sum(dim=-1)
        rs_f = self.row_scale.float()
        y_cur = (self.qzero.float() * x_sum.unsqueeze(0)).sum(dim=-1) * rs_f
        y_ref_norm = float(y_ref.norm().item()) + 1e-8

        step_rel = 1.0
        scar_rel = 1.0
        closed_equal = False

        while len(self._carved_lo) < target_max:
            p = len(self._carved_lo)
            scale_f = self._base_scale.float() * float(2.0 ** (-p))
            sign_bool = (self._rem >= 0).reshape(self.n, self.k)
            packed = _pack_sign_bits_u8(sign_bool)
            scale_bf16 = scale_f.to(torch.bfloat16)
            sign_f = sign_bool.reshape(self.n, self.ng, self.group_size).to(torch.float32).mul_(2.0).sub_(1.0)
            self._rem.sub_(sign_f * scale_f.unsqueeze(-1))
            path = _binary_tree_path(p, self.depth)
            self._carved_lo.append(packed)
            self._carved_sc.append(scale_bf16)
            self._carved_paths.append(path)

            delta_y = ((sign_f * x_rot_g.unsqueeze(0)).sum(dim=-1) * scale_f).sum(dim=-1) * rs_f
            y_cur = y_cur + delta_y
            # Only evaluate convergence once `target_min` planes exist (1 batched GPU->CPU transfer, not 3/plane!)
            if (p + 1) >= target_min and (mirror_tol > 0.0 or (p + 1) == target_max):
                n_cur, n_step, n_scar = torch.stack(
                    [y_cur.norm(), delta_y.norm(), (y_ref - y_cur).norm()]
                ).tolist()
                step_rel = n_step / (n_cur + 1e-8)
                scar_rel = n_scar / y_ref_norm
                if mirror_tol > 0.0 and step_rel < mirror_tol and scar_rel < mirror_tol:
                    closed_equal = True
                    break

        k_opt = len(self._carved_lo)
        tensors = {
            "packed_lo": torch.stack(self._carved_lo[:k_opt], dim=0).contiguous(),
            "qscale": self._carved_sc[0].unsqueeze(0).contiguous(),
            "qzero": self.qzero,
            "row_scale": self.row_scale.contiguous(),
            "col_scale": self.col_scale.contiguous(),
        }
        meta = BinaryHalvingTreeMeta(
            root_bits=k_opt,
            depth=self.depth,
            law="mirror_crystal_positional_halving",
            group_size=self.group_size,
            use_fwht=self.use_fwht,
            level_node_counts=tuple(1 << d for d in range(self.depth + 1)),
            leaf_paths=tuple(self._carved_paths[:k_opt]),
            leaf_energy_frac=(),
            prefix_rel_rmse=(),
            prefix_cos=(),
            source_root_bits=self.root_bits,
        )
        crystal_scar = {
            "k_opt": k_opt,
            "max_k": target_max,
            "scar_rel_rmse": round(scar_rel, 6),
            "step_rel_delta": round(step_rel, 6),
            "closed_equal": closed_equal,
        }
        return tensors, meta, crystal_scar

    @torch.no_grad()
    def carve_trunk_mirror(
        self,
        max_cubes: int,
        *,
        min_cubes: int = 3,
        mirror_tol: float = 0.045,
    ) -> Tuple[Dict[str, torch.Tensor], BinaryHalvingTreeMeta, int, float]:
        """Pure-math M0 Mirror Equality trunk carver (zero words, tensor norm invariants only).

        Evaluates bracketless convergence directly on the rotated weight residual:
            eps_p = ||W - W_p||_F / ||W||_F
            delta_p = ||Delta_p||_F / (||W||_F + eps)
            tau_p = |delta_p - delta_{p-1}| / (delta_p + delta_{p-1} + eps)
        Stopping cleanly at `k_opt` once `eps_p <= mirror_tol and delta_p <= mirror_tol`.
        """
        target_max = max(1, min(self.root_bits, int(max_cubes)))
        target_min = max(1, min(target_max, int(min_cubes)))
        ref_n = (
            float(self._ref_norm)
            if self._ref_norm is not None
            else float(self._rem.pow(2).sum().sqrt().item())
        ) + 1e-8
        rem_rel = 1.0

        while len(self._carved_lo) < target_max:
            p = len(self._carved_lo)
            scale_f = self._base_scale.float() * float(2.0 ** (-p))
            sign_bool = (self._rem >= 0).reshape(self.n, self.k)
            packed = _pack_sign_bits_u8(sign_bool)
            scale_bf16 = scale_f.to(torch.bfloat16)
            sign_f = sign_bool.reshape(self.n, self.ng, self.group_size).to(torch.float32).mul_(2.0).sub_(1.0)
            step_t = sign_f * scale_f.unsqueeze(-1)
            self._rem.sub_(step_t)
            path = _binary_tree_path(p, self.depth)
            self._carved_lo.append(packed)
            self._carved_sc.append(scale_bf16)
            self._carved_paths.append(path)

            if (p + 1) >= target_min and (mirror_tol > 0.0 or (p + 1) == target_max):
                n_rem, n_step = torch.stack(
                    [self._rem.pow(2).sum().sqrt(), step_t.pow(2).sum().sqrt()]
                ).tolist()
                rem_rel = n_rem / ref_n
                step_rel = n_step / ref_n
                if mirror_tol > 0.0 and rem_rel <= mirror_tol and step_rel <= mirror_tol:
                    break

        k_opt = len(self._carved_lo)
        tensors = {
            "packed_lo": torch.stack(self._carved_lo[:k_opt], dim=0).contiguous(),
            "qscale": self._carved_sc[0].unsqueeze(0).contiguous(),
            "qzero": self.qzero,
            "row_scale": self.row_scale.contiguous(),
            "col_scale": self.col_scale.contiguous(),
        }
        meta = BinaryHalvingTreeMeta(
            root_bits=k_opt,
            depth=self.depth,
            law="mirror_trunk_positional_halving",
            group_size=self.group_size,
            use_fwht=self.use_fwht,
            level_node_counts=tuple(1 << d for d in range(self.depth + 1)),
            leaf_paths=tuple(self._carved_paths[:k_opt]),
            leaf_energy_frac=(),
            prefix_rel_rmse=(),
            prefix_cos=(),
            source_root_bits=self.root_bits,
        )
        return tensors, meta, k_opt, rem_rel


@torch.no_grad()
def carve_trunk_mirror_q1(
    weight: torch.Tensor,
    max_cubes: int,
    *,
    min_cubes: int = 3,
    mirror_tol: float = 0.045,
    root_bits: int = 8,
    group_size: int = 64,
    use_fwht: bool = True,
    sinkhorn_iters: int = 3,
) -> Tuple[Dict[str, torch.Tensor], BinaryHalvingTreeMeta, int, float]:
    """Carve a dense trunk projection under pure-math M0 Mirror Equality."""
    carver = IncrementalCubeCarver(
        weight,
        root_bits=root_bits,
        group_size=group_size,
        use_fwht=use_fwht,
        sinkhorn_iters=sinkhorn_iters,
        compute_diagnostics=False,
    )
    return carver.carve_trunk_mirror(max_cubes, min_cubes=min_cubes, mirror_tol=mirror_tol)


@torch.no_grad()
def carve_q1_mirror_crystal(
    weight: torch.Tensor,
    x_probe: torch.Tensor,
    y_exact: torch.Tensor,
    max_cubes: int,
    *,
    min_cubes: int = 2,
    mirror_tol: float = 0.05,
    root_bits: int = 8,
    group_size: int = 64,
    use_fwht: bool = True,
    sinkhorn_iters: int = 3,
) -> Tuple[Dict[str, torch.Tensor], BinaryHalvingTreeMeta, Dict[str, Any]]:
    """Carve a Q1 crystal on the Left x Right intersection, stopping at activation-verified `k_opt`."""
    carver = IncrementalCubeCarver(
        weight,
        root_bits=root_bits,
        group_size=group_size,
        use_fwht=use_fwht,
        sinkhorn_iters=sinkhorn_iters,
        compute_diagnostics=False,
    )
    return carver.carve_until_mirror_equal(
        x_probe,
        y_exact,
        max_cubes,
        min_cubes=min_cubes,
        mirror_tol=mirror_tol,
    )


@torch.no_grad()
def carve_q1_cubes_incremental(
    weight: torch.Tensor,
    num_cubes: int,
    *,
    root_bits: int = 8,
    group_size: int = 64,
    use_fwht: bool = True,
    sinkhorn_iters: int = 3,
) -> Tuple[Dict[str, torch.Tensor], BinaryHalvingTreeMeta]:
    """Carve only the leading `num_cubes` Q1 cubes cube-by-cube without materializing lower branches."""
    carver = IncrementalCubeCarver(
        weight,
        root_bits=root_bits,
        group_size=group_size,
        use_fwht=use_fwht,
        sinkhorn_iters=sinkhorn_iters,
        # Packaging consumes no per-plane statistics. Avoid a CUDA .item()
        # synchronization and full residual reduction for every carved plane.
        compute_diagnostics=False,
    )
    return carver.carve_until(num_cubes)


def _tensor_dict_bytes(d: Dict[str, torch.Tensor]) -> int:
    return sum(int(t.numel()) * int(t.element_size()) for t in d.values() if isinstance(t, torch.Tensor))


class LazyMoEBinaryHalvingCache:
    """Demand-driven AirLLM Q1 cube carver & bounded cache for MoE experts read from disk.

    Never slices unvisited experts upfront: reads an expert's raw weights only when selected by the
    router, carves `store_cubes` Q1 cubes on GPU, caches the compact Q1 representation in a bounded
    LRU arena (GPU or Host), and executes `cubes_for_weight(router_weight)` cubes per token via NVRTC.
    """

    def __init__(
        self,
        expert_reader: Callable[[int, int], Tuple[torch.Tensor, torch.Tensor]],
        *,
        num_layers: int,
        num_experts: int,
        device: torch.device,
        store_cubes: int = 4,
        root_bits: int = 8,
        group_size: int = 64,
        use_fwht: bool = True,
        sinkhorn_iters: int = 3,
        max_cache_bytes: int = 512 * 1024**2,
        host_cache_bytes: int = 0,
        cache_on_gpu: bool = True,
        router_depth_schedule: Optional[Tuple[Tuple[float, int], ...]] = None,
        expert_batch_reader: Optional[Callable[[int, List[int]], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]] = None,
        expert_cubes: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None,
        expert_rest_tol: Optional[float] = None,
        expert_rest_min: int = 3,
        enable_expert_relay: bool = False,
        imprint_puzzle: bool = False,
        imprint_depths: Tuple[int, int, int, int] = (5, 4, 3, 2),
        imprint_auto_vram: bool = False,
        stanok_contour: Any = None,
        fixation_cov_thresh: float = 1.0,
        graphify_feedback: Any = False,
    ):
        self.expert_reader = expert_reader
        self.expert_batch_reader = expert_batch_reader
        self.expert_cubes = dict(expert_cubes or {})
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.device = torch.device(device)
        self.store_cubes = max(1, min(int(root_bits), int(store_cubes)))
        if any(not 1 <= int(depth) <= self.store_cubes for ds in self.expert_cubes.values() for depth in ds):
            raise ValueError("Expert blueprint depth exceeds the stored tree")
        self.root_bits = int(root_bits)
        self.group_size = int(group_size)
        self.use_fwht = bool(use_fwht)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.max_cache_bytes = max(0, int(max_cache_bytes))
        self.host_cache_bytes = max(0, int(host_cache_bytes))
        self.cache_on_gpu = bool(cache_on_gpu and self.device.type == "cuda")
        self.router_depth_schedule = router_depth_schedule
        self.expert_rest_tol: Optional[float] = None if expert_rest_tol is None else float(expert_rest_tol)
        self.expert_rest_min: int = max(1, int(expert_rest_min))
        self.enable_expert_relay: bool = bool(enable_expert_relay)
        self.imprint_puzzle_enabled: bool = bool(imprint_puzzle)
        self.imprint_depths: Tuple[int, int, int, int] = tuple(int(x) for x in imprint_depths)
        self.imprint_auto_vram: bool = bool(imprint_auto_vram)
        from src.stanok_contour import normalize_feedback_mode
        self.feedback_mode: str = normalize_feedback_mode(graphify_feedback)
        self.graphify_feedback: bool = (self.feedback_mode != "off")
        self._imprint_overlay = None

        self._lru: "OrderedDict[Tuple[int, int], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int]]" = OrderedDict()
        self._cached_bytes: int = 0
        self._host_lru: "OrderedDict[Tuple[int, int], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int]]" = OrderedDict()
        self._host_cached_bytes: int = 0
        # Crisp BF16 scar-anchors `(x_gu_last, y_gu_last, x_dn_last, y_dn_last)` for V3 differential Q1 GEMV
        self._mirror_anchors: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # Optional strict A -> B -> C(0) -> A Q1RoleCycleArena relay for MoE experts
        self._expert_relay = None
        self._expert_relay_pool = None
        self._expert_relay_stream = None
        self._expert_relay_future = None
        self._expert_relay_key: Optional[Tuple[int, int]] = None
        self._expert_relay_staged: bool = False
        self.expert_relay_disk_prepares: int = 0
        self.expert_relay_cache_prepares: int = 0
        self.expert_relay_gpu_prepares: int = 0
        self.expert_relay_handoffs: int = 0
        self.expert_relay_wait_seconds: float = 0.0
        self.expert_relay_trace: Dict[str, Any] = {}
        # IC-01B consolidation: when `imprint_puzzle=True`, `ImprintPuzzleMoEOverlay` owns the
        # SINGLE `Q1RoleCycleArena` + CUDA stream + worker pool; never double-allocate a second relay!
        if self.enable_expert_relay and not self.imprint_puzzle_enabled:
            from src.q1_cube_engine import Q1RoleCycleArena
            # Bounded per-expert relay slot (accounting bound only; no static VRAM pre-allocation)
            self._expert_relay = Q1RoleCycleArena(256 * 1024**2, self.device)
            if self.device.type == "cuda":
                self._expert_relay_stream = torch.cuda.Stream(device=self.device)
            self._expert_relay_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quintessentia-moe-B")

        # Telemetry
        self.hits: int = 0
        self.host_hits: int = 0
        self.misses: int = 0
        self.unique_experts_carved: set[Tuple[int, int]] = set()
        self.total_cubes_carved: int = 0
        self.expert_calls: int = 0
        self.expert_cubes_executed: int = 0
        self.evictions: int = 0
        self.host_evictions: int = 0
        self.micro_agent_tol: Optional[float] = None
        self.micro_agent_calls = 0
        self.moe_fabric = False
        self.fabric_calls = 0
        self.routed_slots_total: int = 0
        self.routed_slots_executed: int = 0
        self.cold_experts_rested: int = 0
        self.cold_disk_reads_avoided: int = 0
        if self.imprint_puzzle_enabled:
            ck, mk, rk, sk = self.imprint_depths
            if self.feedback_mode == "mirror_scales":
                from src.mirror_scales_v4 import MirrorIsolatedMoEOverlayV4
                self._imprint_overlay = MirrorIsolatedMoEOverlayV4(
                    self,
                    core_k=ck,
                    mid_k=mk,
                    rim_k=rk,
                    shell_k=sk,
                    min_shed_k=sk,
                    rest_tol=0.24 if self.expert_rest_tol is None else float(self.expert_rest_tol),
                    rest_min=self.expert_rest_min,
                    auto_vram_adapt=self.imprint_auto_vram,
                )
            else:
                from src.probability_imprint_puzzle import ImprintPuzzleMoEOverlay
                self._imprint_overlay = ImprintPuzzleMoEOverlay(
                    self,
                    core_k=ck,
                    mid_k=mk,
                    rim_k=rk,
                    shell_k=sk,
                    min_shed_k=sk,
                    rest_tol=0.24 if self.expert_rest_tol is None else float(self.expert_rest_tol),
                    rest_min=self.expert_rest_min,
                    auto_vram_adapt=self.imprint_auto_vram,
                    fixation_cov_thresh=fixation_cov_thresh,
                    stanok_contour=stanok_contour,
                    graphify_feedback=self.feedback_mode,
                )

    def reset_telemetry(self) -> None:
        self.hits = 0
        self.host_hits = 0
        self.misses = 0
        self.expert_calls = 0
        self.expert_cubes_executed = 0
        self.routed_slots_total = 0
        self.routed_slots_executed = 0
        self.cold_experts_rested = 0
        self.cold_disk_reads_avoided = 0

    def cubes_for_weight(self, router_weight: float) -> int:
        if self.router_depth_schedule is None:
            return self.store_cubes
        for thresh, cubes in self.router_depth_schedule:
            if router_weight >= thresh:
                return max(1, min(self.store_cubes, int(cubes)))
        return 1

    def _put_host(
        self,
        key: Tuple[int, int],
        gu_t: Dict[str, torch.Tensor],
        dn_t: Dict[str, torch.Tensor],
        entry_bytes: int,
    ) -> None:
        if self.host_cache_bytes <= 0 or entry_bytes > self.host_cache_bytes:
            return
        if key in self._host_lru:
            _, _, prev_b = self._host_lru.pop(key)
            self._host_cached_bytes -= prev_b
        while self._host_lru and (self._host_cached_bytes + entry_bytes > self.host_cache_bytes):
            _, (_, _, old_b) = self._host_lru.popitem(last=False)
            self._host_cached_bytes -= old_b
            self.host_evictions += 1
        cpu_gu = {k: (v if v.device.type == "cpu" else v.detach().cpu()) for k, v in gu_t.items()}
        cpu_dn = {k: (v if v.device.type == "cpu" else v.detach().cpu()) for k, v in dn_t.items()}
        self._host_lru[key] = (cpu_gu, cpu_dn, entry_bytes)
        self._host_cached_bytes += entry_bytes

    def _put_primary(
        self,
        key: Tuple[int, int],
        gu_t: Dict[str, torch.Tensor],
        dn_t: Dict[str, torch.Tensor],
        entry_bytes: int,
    ) -> None:
        if self.max_cache_bytes > 0 and entry_bytes <= self.max_cache_bytes:
            while self._lru and (self._cached_bytes + entry_bytes > self.max_cache_bytes):
                old_key, (old_gu, old_dn, old_b) = self._lru.popitem(last=False)
                self._cached_bytes -= old_b
                self.evictions += 1
                if self.host_cache_bytes > 0:
                    self._put_host(old_key, old_gu, old_dn, old_b)
            if self.cache_on_gpu:
                store_gu, store_dn = gu_t, dn_t
            else:
                store_gu = {k: v.detach().cpu() for k, v in gu_t.items()}
                store_dn = {k: v.detach().cpu() for k, v in dn_t.items()}
            self._lru[key] = (store_gu, store_dn, entry_bytes)
            self._cached_bytes += entry_bytes
        elif self.host_cache_bytes > 0:
            self._put_host(key, gu_t, dn_t, entry_bytes)

    @torch.no_grad()
    def _get_or_carve(
        self,
        layer_idx: int,
        eid: int,
        prefetched: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        key = (int(layer_idx), int(eid))
        hit = self._lru.get(key)
        if hit is not None:
            self.hits += 1
            self._lru.move_to_end(key)
            gu_t, dn_t, _ = hit
            if self.cache_on_gpu:
                return gu_t, dn_t
            return (
                {k: v.to(self.device, non_blocking=True) for k, v in gu_t.items()},
                {k: v.to(self.device, non_blocking=True) for k, v in dn_t.items()},
            )

        host_hit = self._host_lru.pop(key, None)
        if host_hit is not None:
            self.host_hits += 1
            h_gu, h_dn, entry_bytes = host_hit
            self._host_cached_bytes -= entry_bytes
            gu_t = {k: v.to(self.device, non_blocking=True) for k, v in h_gu.items()}
            dn_t = {k: v.to(self.device, non_blocking=True) for k, v in h_dn.items()}
            self._put_primary(key, gu_t, dn_t, entry_bytes)
            return gu_t, dn_t

        self.misses += 1
        gu_raw, dn_raw = prefetched if prefetched is not None else self.expert_reader(layer_idx, eid)
        gu_gpu = gu_raw.to(device=self.device, dtype=torch.bfloat16)
        dn_gpu = dn_raw.to(device=self.device, dtype=torch.bfloat16)
        gu_depth, dn_depth = self.expert_cubes.get(key, (self.store_cubes, self.store_cubes))
        gu_t, _ = carve_q1_cubes_incremental(
            gu_gpu,
            gu_depth,
            root_bits=self.root_bits,
            group_size=self.group_size,
            use_fwht=self.use_fwht,
            sinkhorn_iters=self.sinkhorn_iters,
        )
        dn_t, _ = carve_q1_cubes_incremental(
            dn_gpu,
            dn_depth,
            root_bits=self.root_bits,
            group_size=self.group_size,
            use_fwht=self.use_fwht,
            sinkhorn_iters=self.sinkhorn_iters,
        )
        del gu_raw, dn_raw, gu_gpu, dn_gpu
        self.unique_experts_carved.add(key)
        self.total_cubes_carved += gu_depth + dn_depth

        entry_bytes = _tensor_dict_bytes(gu_t) + _tensor_dict_bytes(dn_t)
        self._put_primary(key, gu_t, dn_t, entry_bytes)

        return gu_t, dn_t

    @torch.no_grad()
    def _run_proj(self, x: torch.Tensor, t: Dict[str, torch.Tensor], p_use: int) -> torch.Tensor:
        stored = int(t["packed_lo"].shape[0])
        p_use = min(p_use, stored)
        out_features = int(t["packed_lo"].shape[1])
        in_features = int(t["packed_lo"].shape[2]) * 8
        flat = x.reshape(-1, in_features)
        if flat.shape[0] == 0:
            return torch.empty((*x.shape[:-1], out_features), dtype=x.dtype, device=x.device)
        if self.use_fwht and flat.device.type == "cuda" and flat.dtype == torch.bfloat16:
            from src.quintessentia_q1_cube_nvrtc import launch_ideal_q1_b1_mma, launch_ideal_q1_fwht_gemv
            y_out = torch.empty((flat.shape[0], out_features), dtype=torch.bfloat16, device=flat.device)
            if (self.micro_agent_tol is not None and flat.shape[0] == 1 and self.group_size == 64
                    and in_features >= 1024 and out_features >= 512):
                from src.mirror_micro_agents_nvrtc import launch_coupled_micro_agent_q1_reader
                launch_coupled_micro_agent_q1_reader(
                    flat.contiguous(), t['packed_lo'], t['qscale'], t['qzero'],
                    t['row_scale'], t['col_scale'], y_out, max_planes=p_use,
                    base_tol=self.micro_agent_tol)
                self.micro_agent_calls += 1
                return y_out.view(*x.shape[:-1], out_features)
            if flat.shape[0] >= 8:
                launch_ideal_q1_b1_mma(
                    flat.contiguous(),
                    t["packed_lo"], t["qscale"], t["qzero"], t["row_scale"], t["col_scale"], y_out,
                    group_size=self.group_size, total_planes=stored, active_planes=p_use, bias=None,
                )
            else:
                launch_ideal_q1_fwht_gemv(
                    flat.contiguous(),
                    t["packed_lo"], t["qscale"], t["qzero"], t["row_scale"], t["col_scale"], y_out,
                    group_size=self.group_size, total_planes=stored, plane_start=0, plane_count=p_use, bias=None,
                )
            return y_out.view(*x.shape[:-1], out_features)
        ng = in_features // self.group_size
        u = t["qzero"].to(torch.float32).unsqueeze(-1).expand(out_features, ng, self.group_size).reshape(out_features, in_features).clone()
        for p in range(p_use):
            s_f = _unpack_sign_bits_f32(t["packed_lo"][p]).reshape(out_features, ng, self.group_size)
            u.add_((s_f * _plane_scale_f32(t["qscale"], p).unsqueeze(-1)).reshape(out_features, in_features))
        w = fwht_block(u, self.group_size) * t["row_scale"].float().unsqueeze(1) * t["col_scale"].float().unsqueeze(0) if self.use_fwht else u
        return F.linear(x, w.to(x.dtype))

    @torch.no_grad()
    def _carve_expert_on_B(self, layer_idx: int, eid: int) -> None:
        """Background area B: read cold expert from disk & carve blueprint Q1 cubes on dedicated CUDA stream."""
        key = (int(layer_idx), int(eid))
        gu_depth, dn_depth = self.expert_cubes.get(key, (self.store_cubes, self.store_cubes))
        if self.device.type == "cuda":
            with torch.cuda.device(self.device), torch.cuda.stream(self._expert_relay_stream):
                gu_raw, dn_raw = self.expert_reader(layer_idx, eid)
                gu_gpu = gu_raw.to(device=self.device, dtype=torch.bfloat16)
                dn_gpu = dn_raw.to(device=self.device, dtype=torch.bfloat16)
                gu_t, _ = carve_q1_cubes_incremental(
                    gu_gpu, gu_depth, root_bits=self.root_bits, group_size=self.group_size,
                    use_fwht=self.use_fwht, sinkhorn_iters=self.sinkhorn_iters,
                )
                dn_t, _ = carve_q1_cubes_incremental(
                    dn_gpu, dn_depth, root_bits=self.root_bits, group_size=self.group_size,
                    use_fwht=self.use_fwht, sinkhorn_iters=self.sinkhorn_iters,
                )
                del gu_raw, dn_raw, gu_gpu, dn_gpu
                entry_b = _tensor_dict_bytes(gu_t) + _tensor_dict_bytes(dn_t)
                ready = torch.cuda.Event(enable_timing=False)
                ready.record(self._expert_relay_stream)
            self.expert_relay_gpu_prepares += 1
            self._expert_relay.publish_prepared(
                ("carved", gu_t, dn_t, entry_b, gu_depth, dn_depth),
                gpu_bytes=entry_b,
                ready_event=ready,
            )
        else:
            gu_raw, dn_raw = self.expert_reader(layer_idx, eid)
            gu_t, _ = carve_q1_cubes_incremental(
                gu_raw.to(device=self.device, dtype=torch.bfloat16),
                gu_depth, root_bits=self.root_bits, group_size=self.group_size,
                use_fwht=self.use_fwht, sinkhorn_iters=self.sinkhorn_iters,
            )
            dn_t, _ = carve_q1_cubes_incremental(
                dn_raw.to(device=self.device, dtype=torch.bfloat16),
                dn_depth, root_bits=self.root_bits, group_size=self.group_size,
                use_fwht=self.use_fwht, sinkhorn_iters=self.sinkhorn_iters,
            )
            del gu_raw, dn_raw
            entry_b = _tensor_dict_bytes(gu_t) + _tensor_dict_bytes(dn_t)
            self._expert_relay.publish_prepared(
                ("carved", gu_t, dn_t, entry_b, gu_depth, dn_depth),
                gpu_bytes=0,
                ready_event=None,
            )

    def _prepare_expert_relay(self, layer_idx: int, eid: int) -> None:
        """Stage/prepare one next MoE expert into warm area B while active area A computes."""
        if self._expert_relay is None or self._expert_relay_key is not None:
            return
        key = (int(layer_idx), int(eid))
        self._expert_relay_key = key

        hit = self._lru.get(key)
        if hit is not None:
            self.hits += 1
            self._lru.move_to_end(key)
            gu_t, dn_t, entry_b = hit
            if self.cache_on_gpu or self.device.type != "cuda":
                self._expert_relay.publish_prepared(("gpu_hit", gu_t, dn_t), gpu_bytes=entry_b if self.cache_on_gpu else 0)
            else:
                with torch.cuda.device(self.device), torch.cuda.stream(self._expert_relay_stream):
                    g_dev = {k: v.to(self.device, non_blocking=True) for k, v in gu_t.items()}
                    d_dev = {k: v.to(self.device, non_blocking=True) for k, v in dn_t.items()}
                    ready = torch.cuda.Event(enable_timing=False)
                    ready.record(self._expert_relay_stream)
                self._expert_relay.publish_prepared(("gpu_hit", g_dev, d_dev), gpu_bytes=entry_b, ready_event=ready)
            self._expert_relay_staged = True
            self.expert_relay_cache_prepares += 1
            return

        host_hit = self._host_lru.pop(key, None)
        if host_hit is not None:
            self.host_hits += 1
            h_gu, h_dn, entry_b = host_hit
            self._host_cached_bytes -= entry_b
            if self.device.type == "cuda":
                with torch.cuda.device(self.device), torch.cuda.stream(self._expert_relay_stream):
                    gu_t = {k: v.to(self.device, non_blocking=True) for k, v in h_gu.items()}
                    dn_t = {k: v.to(self.device, non_blocking=True) for k, v in h_dn.items()}
                    ready = torch.cuda.Event(enable_timing=False)
                    ready.record(self._expert_relay_stream)
                self._expert_relay.publish_prepared(
                    ("host_hit", gu_t, dn_t, entry_b), gpu_bytes=entry_b, ready_event=ready
                )
            else:
                self._expert_relay.publish_prepared(("host_hit", h_gu, h_dn, entry_b), gpu_bytes=0)
            self._expert_relay_staged = True
            self.expert_relay_cache_prepares += 1
            return

        self._expert_relay_future = self._expert_relay_pool.submit(self._carve_expert_on_B, int(layer_idx), int(eid))
        self._expert_relay_staged = False
        self.expert_relay_disk_prepares += 1

    def _take_expert_relay(self, layer_idx: int, eid: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Rotate strict A -> B -> C(0) -> A wheel: old_C(0) -> new_A adopts B's prepared expert zero-copy."""
        import time
        key = (int(layer_idx), int(eid))
        if self._expert_relay_key is not None and self._expert_relay_key != key:
            if self._expert_relay_future is not None:
                self._expert_relay_future.result()
            self._expert_relay.buffers[self._expert_relay.warm_area] = None
            self._expert_relay.area_used_bytes[self._expert_relay.warm_area] = 0
            self._expert_relay_key = self._expert_relay_future = None
            self._expert_relay_staged = False
        if self._expert_relay_key is None:
            self._prepare_expert_relay(layer_idx, eid)
        t0 = time.perf_counter()
        if not self._expert_relay_staged and self._expert_relay_future is not None:
            self._expert_relay_future.result()
        self._expert_relay.rotate()
        payload = self._expert_relay.active_payload()
        self.expert_relay_wait_seconds += time.perf_counter() - t0
        self.expert_relay_handoffs += 1
        self._expert_relay_key = self._expert_relay_future = None
        self._expert_relay_staged = False

        kind = payload[0]
        if kind == "gpu_hit":
            return payload[1], payload[2]
        if kind == "host_hit":
            _, gu_t, dn_t, entry_b = payload
            self._put_primary(key, gu_t, dn_t, entry_b)
            return gu_t, dn_t
        # "carved"
        _, gu_t, dn_t, entry_b, gu_depth, dn_depth = payload
        self.misses += 1
        self.unique_experts_carved.add(key)
        self.total_cubes_carved += int(gu_depth) + int(dn_depth)
        self._put_primary(key, gu_t, dn_t, entry_b)
        return gu_t, dn_t

    @torch.no_grad()
    def _forward_layer_moe_with_rest(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Mirror Equality x UNI Network Rest across MoE experts on the shared `line_spine`.

        Always evaluates the leading `expert_rest_min` router experts PLUS any already-hot
        (GPU L1 / Host L2) experts onto `line_spine`, then wakes remaining cold disk-resident
        tail experts wave-by-wave ONLY for tokens whose `line_spine` has NOT reached Mirror x UNI rest.
        """
        from src.stanok_contour import MirrorUniStabilizer
        tol = float(self.expert_rest_tol or 0.0)
        min_exp = self.expert_rest_min

        m_tok, h_dim = int(hidden_states.shape[0]), int(hidden_states.shape[1])
        dev = hidden_states.device
        idx_cpu = top_k_index.detach().cpu().tolist()
        w_cpu = top_k_weights.detach().float().cpu().tolist()

        # Per-token sorted expert plans: descending router weight
        wave0_items: List[Tuple[int, int, int, float]] = []     # (tok_i, eid, k_pos, rw)
        cold_queues: List[List[Tuple[int, int, float]]] = [[] for _ in range(m_tok)]
        tok_total_w: List[float] = [0.0] * m_tok

        for tok_i, (row_e, row_w) in enumerate(zip(idx_cpu, w_cpu)):
            valid: List[Tuple[int, int, float]] = []
            for k_pos, (eid, rw) in enumerate(zip(row_e, row_w)):
                e_int = int(eid)
                if 0 <= e_int < self.num_experts:
                    valid.append((e_int, k_pos, float(rw)))
            valid.sort(key=lambda item: -item[2])
            self.routed_slots_total += len(valid)
            tok_total_w[tok_i] = sum(item[2] for item in valid)
            for rank, (e_int, k_pos, rw) in enumerate(valid):
                is_hot = ((int(layer_idx), e_int) in self._lru) or ((int(layer_idx), e_int) in self._host_lru)
                if rank < min_exp or is_hot:
                    wave0_items.append((tok_i, e_int, k_pos, rw))
                else:
                    cold_queues[tok_i].append((e_int, k_pos, rw))

        # Shared assembly line (`line_spine`) & per-token Mirror x UNI state (all float32 on device)
        line_spine = torch.zeros((m_tok, h_dim), dtype=torch.float32, device=dev)
        last_obs_norm = [0.0] * m_tok
        prev_obs_norm = [0.0] * m_tok
        last_w = [1e-6] * m_tok
        w_done = [0.0] * m_tok
        tok_done = [0] * m_tok
        last_ortho = [0.5] * m_tok  # default neutral angle before 2nd step
        rested = [False] * m_tok

        def _execute_wave(items: List[Tuple[int, int, int, float]]) -> None:
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
            use_relay = self._expert_relay is not None
            prefetched_map: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
            if (not use_relay) and self.expert_batch_reader is not None:
                missing_eids = [
                    eid
                    for eid in sorted_eids
                    if (int(layer_idx), int(eid)) not in self._lru and (int(layer_idx), int(eid)) not in self._host_lru
                ]
                if missing_eids:
                    prefetched_map = self.expert_batch_reader(int(layer_idx), missing_eids)

            if use_relay and sorted_eids:
                self._prepare_expert_relay(layer_idx, sorted_eids[0])

            # Group execution by expert (one Q1 load/carve per unique expert in this wave)
            for pos_e, eid in enumerate(sorted_eids):
                toks, kpos, rws = by_exp[eid]
                tok_t = torch.tensor(toks, dtype=torch.long, device=dev)
                kpos_t = torch.tensor(kpos, dtype=torch.long, device=dev)
                mean_rw = sum(rws) / max(1, len(rws))
                p_use = self.cubes_for_weight(mean_rw)
                if use_relay:
                    gu_t, dn_t = self._take_expert_relay(layer_idx, eid)
                    # Overlap: while A computes `eid`, B prepares the next expert in background!
                    if pos_e + 1 < len(sorted_eids):
                        self._prepare_expert_relay(layer_idx, sorted_eids[pos_e + 1])
                else:
                    gu_t, dn_t = self._get_or_carve(layer_idx, eid, prefetched=prefetched_map.pop(eid, None))
                x_e = hidden_states.index_select(0, tok_t)
                gu = self._run_proj(x_e, gu_t, p_use)
                g_part, u_part = gu.chunk(2, dim=-1)
                act = act_fn(g_part) * u_part
                y_e = self._run_proj(act, dn_t, p_use)
                obs_f = (y_e * top_k_weights[tok_t, kpos_t, None]).to(torch.float32)

                prev_lines = line_spine.index_select(0, tok_t)
                dots = (prev_lines * obs_f).sum(dim=-1)
                prev_norms = prev_lines.norm(dim=-1)
                obs_norms = obs_f.norm(dim=-1)
                line_spine.index_add_(0, tok_t, obs_f)

                # Single batched device->host transfer for this expert's scalar invariants
                triples = torch.stack([dots, prev_norms, obs_norms], dim=-1).tolist()
                for local_i, tok_i in enumerate(toks):
                    dot_v, pn_v, on_v = triples[local_i]
                    prev_obs_norm[tok_i] = last_obs_norm[tok_i]
                    last_obs_norm[tok_i] = on_v
                    last_w[tok_i] = rws[local_i]
                    w_done[tok_i] += rws[local_i]
                    tok_done[tok_i] += 1
                    last_ortho[tok_i] = MirrorUniStabilizer.update_ortho_angle(
                        dot_v, pn_v, on_v, default=last_ortho[tok_i]
                    )
                    self.routed_slots_executed += 1

                self.expert_calls += 1
                self.expert_cubes_executed += (
                    min(p_use, gu_t["packed_lo"].shape[0]) + min(p_use, dn_t["packed_lo"].shape[0])
                ) / 2
                if use_relay:
                    self.expert_relay_trace = {
                        "layer": int(layer_idx),
                        "completed_expert": int(eid),
                        "tokens": len(toks),
                        "mean_rw": round(mean_rw, 4),
                    }
                    self._expert_relay.release_active_A()

        # Wave 0: mandatory core + already-hot L1/L2 experts
        _execute_wave(wave0_items)

        # Evaluate Mirror Equality x UNI rest wave-by-wave before waking cold disk tail experts
        while True:
            # Find tokens that still have cold tail experts pending
            cand_toks = [t for t in range(m_tok) if (not rested[t]) and cold_queues[t]]
            if not cand_toks:
                break

            # Batch-compute line_spine norm & 8-bucket UNI spectral entropy on GPU via shared stabilizer
            inv_pairs = MirrorUniStabilizer.compute_line_invariants(line_spine, cand_toks)

            next_wave: List[Tuple[int, int, int, float]] = []
            avoided_eids_this_check: set[int] = set()
            woken_eids_this_wave: set[int] = set()

            for ci, tok_i in enumerate(cand_toks):
                next_l_norm, lam_v = inv_pairs[ci]
                # Pop next cold expert; if it became hot in an earlier wave, run it freely!
                e_int, k_pos, next_w = cold_queues[tok_i][0]
                is_now_hot = ((int(layer_idx), e_int) in self._lru) or ((int(layer_idx), e_int) in self._host_lru)
                if is_now_hot:
                    cold_queues[tok_i].pop(0)
                    next_wave.append((tok_i, e_int, k_pos, next_w))
                    woken_eids_this_wave.add(e_int)
                    continue

                should_rest, _defect, _eff_tol = MirrorUniStabilizer.evaluate_token_rest(
                    line_norm=next_l_norm,
                    lam_v=lam_v,
                    last_obs_norm=last_obs_norm[tok_i],
                    prev_obs_norm=prev_obs_norm[tok_i],
                    last_w=last_w[tok_i],
                    next_w=next_w,
                    rem_w=tok_total_w[tok_i] - w_done[tok_i],
                    last_ortho=last_ortho[tok_i],
                    tok_done=tok_done[tok_i],
                    rest_min=min_exp,
                    rest_tol=tol,
                )
                if should_rest:
                    # Network rest achieved: skip all remaining cold disk experts for this token!
                    rested[tok_i] = True
                    for rem_e, _, _ in cold_queues[tok_i]:
                        self.cold_experts_rested += 1
                        avoided_eids_this_check.add(rem_e)
                    cold_queues[tok_i].clear()
                else:
                    cold_queues[tok_i].pop(0)
                    next_wave.append((tok_i, e_int, k_pos, next_w))
                    woken_eids_this_wave.add(e_int)

            # Count unique cold experts avoided in this layer call that were not woken by another token
            self.cold_disk_reads_avoided += len(avoided_eids_this_check - woken_eids_this_wave)
            if not next_wave:
                break
            _execute_wave(next_wave)

        return line_spine.to(hidden_states.dtype)

    @torch.no_grad()
    def forward_layer_moe(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if self._imprint_overlay is not None:
            return self._imprint_overlay.forward_layer_moe(layer_idx, hidden_states, top_k_index, top_k_weights, act_fn)
        if self.expert_rest_tol is not None and self.expert_rest_tol > 0.0:
            return self._forward_layer_moe_with_rest(layer_idx, hidden_states, top_k_index, top_k_weights, act_fn)
        out_flat = torch.zeros_like(hidden_states)
        idx_cpu = top_k_index.detach().cpu().tolist()
        w_cpu = top_k_weights.detach().float().cpu().tolist()
        by_exp: Dict[int, Tuple[List[int], List[int], List[float]]] = {}
        for tok_i, (row_e, row_w) in enumerate(zip(idx_cpu, w_cpu)):
            for k_pos, (eid, rw) in enumerate(zip(row_e, row_w)):
                e_int = int(eid)
                if 0 <= e_int < self.num_experts:
                    bucket = by_exp.get(e_int)
                    if bucket is None:
                        bucket = ([], [], [])
                        by_exp[e_int] = bucket
                    bucket[0].append(tok_i)
                    bucket[1].append(k_pos)
                    bucket[2].append(float(rw))
                    self.routed_slots_total += 1
                    self.routed_slots_executed += 1
        sorted_eids = sorted(by_exp)
        use_relay = (self._expert_relay is not None) and (not self.moe_fabric)
        prefetched_map: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        if (not use_relay) and self.expert_batch_reader is not None:
            missing_eids = [
                eid
                for eid in sorted_eids
                if (int(layer_idx), int(eid)) not in self._lru and (int(layer_idx), int(eid)) not in self._host_lru
            ]
            if missing_eids:
                prefetched_map = self.expert_batch_reader(int(layer_idx), missing_eids)
        activation_name = getattr(act_fn, '__name__', act_fn.__class__.__name__).lower()
        if (self.moe_fabric and hidden_states.device.type == 'cuda'
                and hidden_states.dtype == torch.bfloat16 and hidden_states.shape[0] == 1
                and self.group_size == 64 and self.use_fwht
                and hidden_states.shape[1] <= 4096 and activation_name in ('silu', 'siluactivation')
                and sorted_eids and all(len(by_exp[e][0]) == 1 for e in sorted_eids)):
            from src.qwen_micro_fabric import launch_q1_moe_fabric
            packages = []
            weights = []
            for eid in sorted_eids:
                _, kpos, rws = by_exp[eid]
                p_use = self.cubes_for_weight(rws[0])
                gu_t, dn_t = self._get_or_carve(layer_idx, eid, prefetched=prefetched_map.pop(eid, None))
                packages.append((gu_t, dn_t, p_use))
                weights.append(top_k_weights[0, kpos[0]])
            intermediate = int(packages[0][1]['packed_lo'].shape[2]) * 8
            if intermediate <= 4096:
                result, _ = launch_q1_moe_fabric(
                    hidden_states.contiguous(), packages, torch.stack(weights),
                    base_tol=0.0 if self.micro_agent_tol is None else self.micro_agent_tol)
                self.fabric_calls += 1
                self.expert_calls += len(packages)
                self.expert_cubes_executed += sum(
                    (min(p, gu['packed_lo'].shape[0]) + min(p, dn['packed_lo'].shape[0])) / 2
                    for gu, dn, p in packages)
                return result
            # An optional unsupported shape retains the ordinary operator path.
            # Hold the already loaded packages so this does not reload the disk.
            for (gu, dn, p), rw in zip(packages, weights):
                gate, up = self._run_proj(hidden_states, gu, p).chunk(2, dim=-1)
                y = self._run_proj(act_fn(gate) * up, dn, p)
                out_flat.add_((y * rw.reshape(1, 1)).to(out_flat.dtype))
                self.expert_calls += 1
                self.expert_cubes_executed += (min(p,gu['packed_lo'].shape[0])+min(p,dn['packed_lo'].shape[0]))/2
            return out_flat
        if use_relay and sorted_eids:
            self._prepare_expert_relay(layer_idx, sorted_eids[0])
        for pos_e, eid in enumerate(sorted_eids):
            toks, kpos, rws = by_exp[eid]
            tok_t = torch.tensor(toks, dtype=torch.long, device=hidden_states.device)
            kpos_t = torch.tensor(kpos, dtype=torch.long, device=hidden_states.device)
            mean_rw = sum(rws) / max(1, len(rws))
            p_use = self.cubes_for_weight(mean_rw)
            if use_relay:
                gu_t, dn_t = self._take_expert_relay(layer_idx, eid)
                if pos_e + 1 < len(sorted_eids):
                    self._prepare_expert_relay(layer_idx, sorted_eids[pos_e + 1])
            else:
                gu_t, dn_t = self._get_or_carve(layer_idx, eid, prefetched=prefetched_map.pop(eid, None))
            x_e = hidden_states.index_select(0, tok_t)
            gu = self._run_proj(x_e, gu_t, p_use)
            g_part, u_part = gu.chunk(2, dim=-1)
            act = act_fn(g_part) * u_part
            y_e = self._run_proj(act, dn_t, p_use)
            y_e = y_e * top_k_weights[tok_t, kpos_t, None]
            out_flat.index_add_(0, tok_t, y_e.to(out_flat.dtype))
            self.expert_calls += 1
            self.expert_cubes_executed += (min(p_use, gu_t['packed_lo'].shape[0]) + min(p_use, dn_t['packed_lo'].shape[0])) / 2
            if use_relay:
                self.expert_relay_trace = {"layer": int(layer_idx), "completed_expert": int(eid), "tokens": len(toks)}
                self._expert_relay.release_active_A()
        return out_flat

    def stats(self) -> Dict[str, Any]:
        total_possible = self.num_layers * self.num_experts
        imprint_stats = None if self._imprint_overlay is None else self._imprint_overlay.stats()
        if self._expert_relay is not None:
            relay_stats: Any = {
                **self._expert_relay.stats(),
                "standby_bytes": self._expert_relay.area_used_bytes[self._expert_relay.standby_area],
                "handoffs": self.expert_relay_handoffs,
                "disk_prepares": self.expert_relay_disk_prepares,
                "cache_prepares": self.expert_relay_cache_prepares,
                "gpu_prepares": self.expert_relay_gpu_prepares,
                "wait_seconds": round(self.expert_relay_wait_seconds, 6),
                "last_trace": dict(self.expert_relay_trace),
            }
        elif self.enable_expert_relay and imprint_stats is not None and "relay" in imprint_stats:
            relay_stats = dict(imprint_stats["relay"])
        else:
            relay_stats = None
        return {
            "store_cubes": self.store_cubes,
            "cached_experts": len(self._lru),
            "cached_mib": round(self._cached_bytes / 1024**2, 2),
            "host_cached_experts": len(self._host_lru),
            "host_cached_mib": round(self._host_cached_bytes / 1024**2, 2),
            "cache_on_gpu": self.cache_on_gpu,
            "gpu_budget_mib": self.max_cache_bytes / 1024**2,
            "host_budget_mib": self.host_cache_bytes / 1024**2,
            "planned_experts": len(self.expert_cubes),
            "unique_experts_carved": len(self.unique_experts_carved),
            "total_model_experts": total_possible,
            "carved_fraction_pct": round(100.0 * len(self.unique_experts_carved) / max(1, total_possible), 2),
            "total_cubes_carved": self.total_cubes_carved,
            "hits": self.hits,
            "host_hits": self.host_hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "host_evictions": self.host_evictions,
            "micro_agent_tol": self.micro_agent_tol,
            "micro_agent_calls": self.micro_agent_calls,
            "moe_fabric": self.moe_fabric,
            "fabric_calls": self.fabric_calls,
            "expert_rest_tol": self.expert_rest_tol,
            "expert_rest_min": self.expert_rest_min,
            "routed_slots_total": self.routed_slots_total,
            "routed_slots_executed": self.routed_slots_executed,
            "cold_experts_rested": self.cold_experts_rested,
            "cold_disk_reads_avoided": self.cold_disk_reads_avoided,
            "cold_rest_pct": round(100.0 * self.cold_experts_rested / max(1, self.routed_slots_total), 2),
            "avg_cubes_executed": round(self.expert_cubes_executed / max(1, self.expert_calls), 3),
            "expert_relay": relay_stats,
            "imprint_puzzle": imprint_stats,
        }

    def close(self) -> None:
        if self._imprint_overlay is not None:
            self._imprint_overlay.close()
            self._imprint_overlay = None
        if self._expert_relay_pool is not None:
            self._expert_relay_pool.shutdown(wait=True, cancel_futures=True)
            self._expert_relay_pool = None
        self._expert_relay_future = self._expert_relay_key = None
        if self._expert_relay is not None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self._expert_relay.buffers[:] = [None, None, None]
            self._expert_relay._pinned_host_scratch[:] = [None, None, None]
            self._expert_relay = None
        self._lru.clear()
        self._cached_bytes = 0
        self._host_lru.clear()
        self._host_cached_bytes = 0
        self._mirror_anchors.clear()


class AirLLMLazyBinaryHalvingRunner:
    """Method 2: AirLLM-style layer-by-layer streaming with on-the-fly Q1 cube carving.

    Keeps all decoder layers on `meta` (0 B VRAM when inactive) and streams exactly one layer at a
    time along the strict monotonic role cycle `A -> B -> C(0) -> A` (`standby_bytes == 0`, `B->A == 0`):
      - Pre-hook (C -> A): loads layer `l` from bounded Host Q1 cube cache or reads raw layer `l`
        from disk via `layer_reader(l)` and carves `cubes_for_proj` Q1 cubes on GPU cube-by-cube.
      - Compute (A): executes layer `l` on GPU via NVRTC `b1.mma` / `fwht_gemv`.
      - Post-hook (A -> B -> C(0)): parks compact Q1 cubes into bounded Host LRU (if budget > 0)
        and immediately strips every parameter/buffer of layer `l` back to `meta` (`standby_bytes = 0`).
    """

    def __init__(
        self,
        model: nn.Module,
        layer_reader: Callable[[int], Dict[str, torch.Tensor]],
        *,
        device: torch.device | str = "cuda:0",
        cubes_per_proj: int = 4,
        proj_cubes_override: Optional[Dict[str, int]] = None,
        layer_proj_cubes: Optional[Dict[int, Dict[str, int]]] = None,
        root_bits: int = 8,
        group_size: int = 64,
        use_fwht: bool = True,
        sinkhorn_iters: int = 3,
        resident_layers: int = 0,
        host_cache_bytes: int = 512 * 1024**2,
        moe_cache: Optional[LazyMoEBinaryHalvingCache] = None,
        moe_module_paths: Optional[Dict[int, str]] = None,
        cancel_check: Optional[Callable[[], None]] = None,
        enable_relay: bool = True,
        micro_agent_tol: Optional[float] = None,
        mirror_v4_trunk: bool = False,
        trunk_mirror_tol: float = 0.015,
        trunk_carve_tol: float = 0.045,
    ):
        self.model = model
        self.layer_reader = layer_reader
        self.device = torch.device(device)
        self.cubes_per_proj = max(1, min(int(root_bits), int(cubes_per_proj)))
        self.proj_cubes_override = dict(proj_cubes_override or {})
        self.layer_proj_cubes: Dict[int, Dict[str, int]] = {
            int(k): dict(v) for k, v in (layer_proj_cubes or {}).items()
        }
        self.root_bits = int(root_bits)
        self.group_size = int(group_size)
        self.use_fwht = bool(use_fwht)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.host_cache_bytes = max(0, int(host_cache_bytes))
        self.moe_cache = moe_cache
        self.moe_module_paths = dict(moe_module_paths or {})
        self.cancel_check = cancel_check
        self.relay_enabled = bool(enable_relay)
        self.micro_agent_tol = micro_agent_tol
        self.mirror_v4_trunk = bool(mirror_v4_trunk)
        self.trunk_mirror_tol = float(trunk_mirror_tol)
        self.trunk_carve_tol = float(trunk_carve_tol)
        self.trunk_projections_carved: int = 0
        self.trunk_early_stops: int = 0
        self._trunk_q1_modules: List[BinaryHalvingQ1Linear] = []
        self._relay = None
        self._relay_pool = None
        self._relay_future = None
        self._relay_layer = None
        self._relay_staged = False
        self.relay_disk_prepares = 0
        self.relay_cache_prepares = 0
        self.relay_wait_seconds = 0.0
        self.relay_handoffs = 0
        self.relay_trace = {}
        self._relay_prepare_stream = None
        self.relay_gpu_prepares = 0

        self.layers = self._find_decoder_layers()
        self.num_layers = len(self.layers)
        for li, depths in self.layer_proj_cubes.items():
            if not 0 <= li < self.num_layers:
                raise ValueError(f"Blueprint layer {li} is outside this model")
            supported = [name for name, mod in self.layers[li].named_modules()
                         if isinstance(mod, nn.Linear) and mod.in_features % self.group_size == 0
                         and not _is_router_module(f"layers.{li}.{name}", mod)]
            accepted = set(supported) | {name.rsplit('.', 1)[-1] for name in supported}
            for name, depth in depths.items():
                if name not in accepted or not 1 <= int(depth) <= self.root_bits:
                    raise ValueError(f"Invalid blueprint projection/depth: layer={li}, {name}={depth}")
        self.resident_layers = max(0, min(self.num_layers, int(resident_layers)))
        self._resident_ready: set[int] = set()
        self.resident_layer_hits: int = 0
        self.resident_bytes: int = 0
        self._handles: List[Any] = []
        self._layer_specs: List[Dict[str, Any]] = []
        self._host_cache: "OrderedDict[int, Tuple[Dict[str, Any], int]]" = OrderedDict()
        self._host_cached_bytes: int = 0
        self._active_layer: Optional[int] = None
        self._installed: bool = False

        # Strict A -> B -> C(0) -> A role telemetry
        self.rotations: int = 0
        self.transitions: Dict[str, int] = {"A->B": 0, "B->C": 0, "C->A": 0, "B->A": 0}
        self.standby_bytes: int = 0
        self.disk_layer_reads: int = 0
        self.host_cache_hits: int = 0
        self.host_cache_evictions: int = 0
        self.total_cubes_carved: int = 0
        self.peak_active_layer_bytes: int = 0
        self.peak_vram_bytes: int = 0

    def _find_decoder_layers(self) -> nn.ModuleList:
        for attr in ("model", "transformer", "gpt_neox"):
            sub = getattr(self.model, attr, None)
            if sub is not None:
                for l_attr in ("layers", "h", "blocks"):
                    layers = getattr(sub, l_attr, None)
                    if isinstance(layers, nn.ModuleList):
                        return layers
        raise ValueError("Could not locate decoder ModuleList on model")

    def _cubes_for_sub(self, layer_idx: int, sub_name: str) -> int:
        leaf = sub_name.rsplit(".", 1)[-1]
        l_map = self.layer_proj_cubes.get(int(layer_idx))
        if l_map is not None:
            if sub_name in l_map:
                return max(1, min(self.root_bits, int(l_map[sub_name])))
            if leaf in l_map:
                return max(1, min(self.root_bits, int(l_map[leaf])))
        return max(1, min(self.root_bits, int(self.proj_cubes_override.get(leaf, self.cubes_per_proj))))

    @torch.no_grad()
    def install(self) -> "AirLLMLazyBinaryHalvingRunner":
        """Strip all decoder layers to `meta` (0 B VRAM) and install AirLLM layer hooks."""
        if self._installed:
            return self
        from accelerate.utils import set_module_tensor_to_device

        self._layer_specs.clear()
        for l_idx, layer in enumerate(self.layers):
            linear_map: Dict[str, Tuple[int, int, bool, int]] = {}
            other_params: List[Tuple[str, torch.dtype]] = []
            other_buffers: List[Tuple[str, torch.dtype]] = []
            moe_mod_name: Optional[str] = self.moe_module_paths.get(l_idx)

            for sub_name, sub_mod in list(layer.named_modules()):
                if isinstance(sub_mod, nn.Linear):
                    if _is_router_module(f"layers.{l_idx}.{sub_name}", sub_mod):
                        continue
                    if sub_mod.in_features % self.group_size == 0:
                        linear_map[sub_name] = (
                            int(sub_mod.in_features),
                            int(sub_mod.out_features),
                            sub_mod.bias is not None,
                            self._cubes_for_sub(l_idx, sub_name),
                        )
                elif (
                    moe_mod_name is None
                    and self.moe_cache is not None
                    and hasattr(sub_mod, "gate_up_proj")
                    and hasattr(sub_mod, "down_proj")
                ):
                    moe_mod_name = sub_name

            # Record non-Q1 parameters (norms, routers, conv1d, A_log, dt_bias, etc.) before stripping to meta
            linear_param_names = set()
            for sn in linear_map:
                linear_param_names.add(f"{sn}.weight")
                linear_param_names.add(f"{sn}.bias")
            moe_param_names = set()
            if moe_mod_name is not None:
                moe_param_names.add(f"{moe_mod_name}.gate_up_proj")
                moe_param_names.add(f"{moe_mod_name}.down_proj")

            for pname, p in list(layer.named_parameters()):
                if pname not in linear_param_names and pname not in moe_param_names:
                    other_params.append((pname, p.dtype))
                if not p.is_meta:
                    set_module_tensor_to_device(layer, pname, "meta", clear_cache=False)

            for bname, b in list(layer.named_buffers()):
                if b is not None and not b.is_meta:
                    # Keep tiny constant buffers like rotary inv_freq or mark to move
                    other_buffers.append((bname, b.dtype))

            # If MoE lazy cache is provided, bind demand-driven Q1 forward directly on experts module
            if moe_mod_name is not None and self.moe_cache is not None:
                exp_mod = layer.get_submodule(moe_mod_name)
                act_fn = getattr(exp_mod, "act_fn", F.silu)
                moe_cache_ref = self.moe_cache
                l_fixed = l_idx

                def _make_moe_fwd(li, af):
                    def _fwd(hidden_states, top_k_index, top_k_weights):
                        return moe_cache_ref.forward_layer_moe(li, hidden_states, top_k_index, top_k_weights, af)
                    return _fwd

                exp_mod.forward = _make_moe_fwd(l_fixed, act_fn)

            self._layer_specs.append({
                "linear_map": linear_map,
                "other_params": other_params,
                "other_buffers": other_buffers,
            })

            h_pre = layer.register_forward_pre_hook(self._make_pre_hook(l_idx), with_kwargs=True)
            h_post = layer.register_forward_hook(self._make_post_hook(l_idx))
            self._handles.extend([h_pre, h_post])

        self._installed = True
        self.standby_bytes = 0
        if self.relay_enabled and self.resident_layers < self.num_layers:
            from src.q1_cube_engine import Q1RoleCycleArena
            # Only a bound: Q1RoleCycleArena does not allocate three full slots.
            max_bytes = max(
                sum(
                    p * n * (k // 8) + 4 * n * (k // self.group_size) + 2 * (n + k) + (2 * n if _bias else 0)
                    for k, n, _bias, p in spec["linear_map"].values()
                ) + sum(
                    int(self.layers[li].get_parameter(name).numel()) * 4
                    for name, _dt in spec["other_params"]
                )
                for li, spec in enumerate(self._layer_specs)
            )
            self._relay = Q1RoleCycleArena(max(1, max_bytes), self.device)
            if self.device.type == "cuda":
                self._relay_prepare_stream = torch.cuda.Stream(device=self.device)
            self._relay_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quintessentia-cube-B")
        return self

    def _prepare_relay(self, layer_idx: int) -> None:
        """B prepares one actual next streamed layer while A is computing."""
        if self._relay is None or self._relay_layer is not None:
            return
        self._relay_layer = int(layer_idx)
        cached = self._host_cache.get(layer_idx)
        if cached is not None:
            pkg, nbytes = cached
            self._relay.stage_payload_into_B(("cached", pkg), nbytes)
            self._relay_staged = True
            self.relay_cache_prepares += 1
        else:
            self._relay_future = self._relay_pool.submit(self._prepare_layer_on_B, layer_idx)
            self._relay_staged = False
            self.relay_disk_prepares += 1

    @torch.no_grad()
    def _prepare_layer_on_B(self, layer_idx: int) -> None:
        raw = self.layer_reader(layer_idx)
        if self.device.type != "cuda":
            self._relay.publish_prepared(("raw", raw))
            return
        spec = self._layer_specs[layer_idx]
        with torch.cuda.device(self.device), torch.cuda.stream(self._relay_prepare_stream):
            pkg = {"q1": {}, "other": {}}
            for name, (in_f, out_f, bias, depth) in spec['linear_map'].items():
                w = raw[name+'.weight'].to(device=self.device, dtype=torch.bfloat16)
                b = raw.get(name+'.bias')
                b = None if b is None else b.to(device=self.device, dtype=torch.bfloat16)
                tree, meta = carve_q1_cubes_incremental(w, depth, root_bits=self.root_bits,
                    group_size=self.group_size, use_fwht=self.use_fwht, sinkhorn_iters=self.sinkhorn_iters)
                pkg['q1'][name] = dict(tensors=tree, bias=b, meta=meta)
                del w
            for name, dtype in spec['other_params']:
                if name in raw:
                    pkg['other'][name] = raw[name].to(device=self.device, dtype=dtype)
            nbytes = sum(_tensor_dict_bytes(v['tensors']) +
                         (0 if v['bias'] is None else v['bias'].numel()*v['bias'].element_size())
                         for v in pkg['q1'].values()) + _tensor_dict_bytes(pkg['other'])
            host = None
            if nbytes <= self.host_cache_bytes:
                host = {'q1': {name: dict(tensors={k:v.detach().cpu() for k,v in ent['tensors'].items()},
                         bias=None if ent['bias'] is None else ent['bias'].detach().cpu(), meta=ent['meta'])
                         for name,ent in pkg['q1'].items()},
                        'other': {k:v.detach().cpu() for k,v in pkg['other'].items()}}
            ready = torch.cuda.Event(enable_timing=False)
            ready.record(self._relay_prepare_stream)
        self.relay_gpu_prepares += 1
        self._relay.publish_prepared(("prepared", pkg, host, nbytes), gpu_bytes=nbytes, ready_event=ready)

    def _take_relay(self, layer_idx: int):
        import time
        if self._relay_layer is not None and self._relay_layer != layer_idx:
            # A restarted/aborted forward must never consume another layer's
            # prepared work. Finish the single bounded read before reusing B.
            if self._relay_future is not None:
                self._relay_future.result()
            self._relay.buffers[self._relay.warm_area] = None
            self._relay.area_used_bytes[self._relay.warm_area] = 0
            self._relay_layer = self._relay_future = None
        if self._relay_layer is None:
            self._prepare_relay(layer_idx)
        started = time.perf_counter()
        if not self._relay_staged:
            self._relay_future.result()  # B has published its complete work and ready event.
        self._relay.rotate()
        payload = self._relay.active_payload()
        self.relay_wait_seconds += time.perf_counter() - started
        self.relay_handoffs += 1
        self._relay_layer = self._relay_future = None
        self._relay_staged = False
        return payload

    def _make_pre_hook(self, layer_idx: int):
        from accelerate.utils import set_module_tensor_to_device

        @torch.no_grad()
        def _pre(module, args, kwargs):
            if self.cancel_check is not None:
                self.cancel_check()
            if layer_idx in self._resident_ready:
                self.resident_layer_hits += 1
                return args, kwargs

            spec = self._layer_specs[layer_idx]
            linear_map = spec["linear_map"]
            other_params = spec["other_params"]
            other_buffers = spec["other_buffers"]
            is_resident_target = layer_idx < self.resident_layers

            # C(0) -> A transition (for streamed layers)
            if not is_resident_target:
                self.transitions["C->A"] += 1
            self._active_layer = layer_idx
            active_bytes = 0

            prepared = self._take_relay(layer_idx) if self._relay is not None and not is_resident_target else None
            cached = self._host_cache.get(layer_idx)
            if prepared is not None and prepared[0] == "prepared":
                self.disk_layer_reads += 1
                self.total_cubes_carved += sum(v[3] for v in linear_map.values())
                host, entry_b = prepared[2], prepared[3]
                if host is not None:
                    while self._host_cache and self._host_cached_bytes + entry_b > self.host_cache_bytes:
                        _, (_, old_b) = self._host_cache.popitem(last=False)
                        self._host_cached_bytes -= old_b
                        self.host_cache_evictions += 1
                    self._host_cache[layer_idx] = (host, entry_b)
                    self._host_cached_bytes += entry_b
            if prepared is not None and prepared[0] in ("cached", "prepared"):
                cached = (prepared[1], 0)
            if cached is not None:
                if prepared is None or prepared[0] != "prepared":
                    self.host_cache_hits += 1
                if layer_idx in self._host_cache:
                    self._host_cache.move_to_end(layer_idx)
                pkg, _ = cached
                q1_host = pkg["q1"]
                other_host = pkg["other"]
                for sub_name, (in_f, out_f, has_bias, n_cubes) in linear_map.items():
                    h_entry = q1_host[sub_name]
                    t_gpu = {k: v.to(self.device, non_blocking=True) for k, v in h_entry["tensors"].items()}
                    b_gpu = None if h_entry["bias"] is None else h_entry["bias"].to(self.device, non_blocking=True)
                    active_bytes += _tensor_dict_bytes(t_gpu) + (0 if b_gpu is None else b_gpu.numel() * b_gpu.element_size())
                    q1_mod = BinaryHalvingQ1Linear(
                        in_f, out_f, t_gpu, h_entry["meta"],
                        bias=b_gpu, layer_idx=layer_idx, proj_name=sub_name.rsplit(".", 1)[-1],
                    )
                    q1_mod.active_cubes = n_cubes
                    q1_mod.micro_agent_tol = self.micro_agent_tol
                    _set_submodule(module, sub_name, q1_mod)
                for pname, dt in other_params:
                    if pname in other_host:
                        t_g = other_host[pname].to(self.device, dtype=dt, non_blocking=True)
                        active_bytes += t_g.numel() * t_g.element_size()
                        set_module_tensor_to_device(module, pname, self.device, value=t_g, dtype=dt, clear_cache=False)
            else:
                self.disk_layer_reads += 1
                raw_tensors = prepared[1] if prepared is not None else self.layer_reader(layer_idx)
                q1_for_host: Dict[str, Any] = {}
                other_for_host: Dict[str, torch.Tensor] = {}
                keep_in_host = (not is_resident_target) and (self.host_cache_bytes > 0)

                for sub_name, (in_f, out_f, has_bias, n_cubes) in linear_map.items():
                    w_raw = raw_tensors[f"{sub_name}.weight"].to(device=self.device, dtype=torch.bfloat16)
                    b_raw = raw_tensors.get(f"{sub_name}.bias")
                    b_gpu = None if b_raw is None else b_raw.to(device=self.device, dtype=torch.bfloat16).contiguous()
                    in_cone = (self.num_layers >= 8 and layer_idx < 4)
                    if self.mirror_v4_trunk:
                        min_k = n_cubes if in_cone else max(3, n_cubes - 1)
                        t_gpu, meta, k_used, _ = carve_trunk_mirror_q1(
                            w_raw,
                            n_cubes,
                            min_cubes=min_k,
                            mirror_tol=self.trunk_carve_tol,
                            root_bits=self.root_bits,
                            group_size=self.group_size,
                            use_fwht=self.use_fwht,
                            sinkhorn_iters=self.sinkhorn_iters,
                        )
                        self.trunk_projections_carved += 1
                        if k_used < n_cubes:
                            self.trunk_early_stops += 1
                    else:
                        t_gpu, meta = carve_q1_cubes_incremental(
                            w_raw,
                            n_cubes,
                            root_bits=self.root_bits,
                            group_size=self.group_size,
                            use_fwht=self.use_fwht,
                            sinkhorn_iters=self.sinkhorn_iters,
                        )
                        k_used = n_cubes
                    del w_raw
                    self.total_cubes_carved += k_used
                    active_bytes += _tensor_dict_bytes(t_gpu) + (0 if b_gpu is None else b_gpu.numel() * b_gpu.element_size())
                    q1_mod = BinaryHalvingQ1Linear(
                        in_f, out_f, t_gpu, meta,
                        bias=b_gpu, layer_idx=layer_idx, proj_name=sub_name.rsplit(".", 1)[-1],
                    )
                    q1_mod.active_cubes = k_used
                    q1_mod.micro_agent_tol = self.micro_agent_tol
                    if self.mirror_v4_trunk:
                        q1_mod.mirror_diff_tol = (0.5 * self.trunk_mirror_tol) if in_cone else self.trunk_mirror_tol
                        if is_resident_target:
                            self._trunk_q1_modules.append(q1_mod)
                    _set_submodule(module, sub_name, q1_mod)
                    if keep_in_host:
                        q1_for_host[sub_name] = {
                            "tensors": {k: v.detach().cpu() for k, v in t_gpu.items()},
                            "bias": None if b_gpu is None else b_gpu.detach().cpu(),
                            "meta": meta,
                        }

                for pname, dt in other_params:
                    if pname in raw_tensors:
                        t_raw = raw_tensors[pname]
                        t_g = t_raw.to(device=self.device, dtype=dt)
                        active_bytes += t_g.numel() * t_g.element_size()
                        set_module_tensor_to_device(module, pname, self.device, value=t_g, dtype=dt, clear_cache=False)
                        if keep_in_host:
                            other_for_host[pname] = t_raw.detach().cpu()

                for bname, bdt in other_buffers:
                    if bname in raw_tensors:
                        b_raw = raw_tensors[bname]
                        b_g = b_raw.to(device=self.device, dtype=bdt)
                        set_module_tensor_to_device(module, bname, self.device, value=b_g, dtype=bdt, clear_cache=False)

                del raw_tensors
                if keep_in_host:
                    entry_b = sum(
                        _tensor_dict_bytes(v["tensors"]) + (0 if v["bias"] is None else v["bias"].numel() * v["bias"].element_size())
                        for v in q1_for_host.values()
                    ) + _tensor_dict_bytes(other_for_host)
                    if entry_b <= self.host_cache_bytes:
                        while self._host_cache and (self._host_cached_bytes + entry_b > self.host_cache_bytes):
                            _, (_, old_b) = self._host_cache.popitem(last=False)
                            self._host_cached_bytes -= old_b
                            self.host_cache_evictions += 1
                        self._host_cache[layer_idx] = ({"q1": q1_for_host, "other": other_for_host}, entry_b)
                        self._host_cached_bytes += entry_b

            if is_resident_target:
                self._resident_ready.add(layer_idx)
                self.resident_bytes += active_bytes
            if active_bytes > self.peak_active_layer_bytes:
                self.peak_active_layer_bytes = active_bytes
            if self.device.type == "cuda":
                cur_v = int(torch.cuda.memory_allocated(self.device))
                if cur_v > self.peak_vram_bytes:
                    self.peak_vram_bytes = cur_v
            if self._relay is not None:
                if not is_resident_target:
                    # Retain the prepared device objects until A's consumer
                    # event is recorded. The raw disk package can now go away.
                    self._relay.buffers[self._relay.active_area] = (
                        tuple(m for m in module.modules() if isinstance(m, BinaryHalvingQ1Linear)), None
                    )
                    self._relay.area_used_bytes[self._relay.active_area] = active_bytes
                next_layer = layer_idx + 1
                if next_layer >= self.num_layers:
                    next_layer = self.resident_layers
                if next_layer >= self.resident_layers and next_layer < self.num_layers:
                    self._prepare_relay(next_layer)
            return args, kwargs

        return _pre

    def _make_post_hook(self, layer_idx: int):
        from accelerate.utils import set_module_tensor_to_device

        @torch.no_grad()
        def _post(module, args, output):
            if layer_idx in self._resident_ready:
                self._active_layer = None
                return output

            spec = self._layer_specs[layer_idx]
            linear_map = spec["linear_map"]
            other_params = spec["other_params"]

            if self._relay is not None:
                # The prefill-derived selection remains the preparation basis;
                # releasing A does not erase that retained plan.
                self.relay_trace = {"completed_layer": layer_idx,
                                    "cube_depths": dict(self.layer_proj_cubes.get(layer_idx, {}))}
                self._relay.release_active_A()

            # A -> B (retire active layer) -> C(0) (zero standby bytes in VRAM)
            self.transitions["A->B"] += 1
            for sub_name, (in_f, out_f, has_bias, _) in linear_map.items():
                # Replace active BinaryHalvingQ1Linear with zero-byte meta nn.Linear shell
                shell = nn.Linear(in_f, out_f, bias=has_bias, device="meta", dtype=torch.bfloat16)
                _set_submodule(module, sub_name, shell)
            for pname, _ in other_params:
                set_module_tensor_to_device(module, pname, "meta", clear_cache=False)

            self.transitions["B->C"] += 1
            self.rotations += 1
            self._active_layer = None
            self.standby_bytes = 0
            return output

        return _post

    @torch.no_grad()
    def warmup_resident_layer(self, layer_idx: int) -> bool:
        """Warm one cold resident trunk layer once (for Scale S3 MirrorColdWarmupScale)."""
        li = int(layer_idx)
        if (not self._installed) or li < 0 or li >= self.resident_layers or li >= len(self.layers):
            return False
        if li in self._resident_ready:
            return False
        from accelerate.utils import set_module_tensor_to_device

        module = self.layers[li]
        spec = self._layer_specs[li]
        linear_map = spec["linear_map"]
        other_params = spec["other_params"]
        other_buffers = spec["other_buffers"]
        active_bytes = 0

        self.disk_layer_reads += 1
        raw_tensors = self.layer_reader(li)
        in_cone = (self.num_layers >= 8 and li < 4)
        for sub_name, (in_f, out_f, has_bias, n_cubes) in linear_map.items():
            w_raw = raw_tensors[f"{sub_name}.weight"].to(device=self.device, dtype=torch.bfloat16)
            b_raw = raw_tensors.get(f"{sub_name}.bias")
            b_gpu = None if b_raw is None else b_raw.to(device=self.device, dtype=torch.bfloat16).contiguous()
            if self.mirror_v4_trunk:
                min_k = n_cubes if in_cone else max(3, n_cubes - 1)
                t_gpu, meta, k_used, _ = carve_trunk_mirror_q1(
                    w_raw,
                    n_cubes,
                    min_cubes=min_k,
                    mirror_tol=self.trunk_carve_tol,
                    root_bits=self.root_bits,
                    group_size=self.group_size,
                    use_fwht=self.use_fwht,
                    sinkhorn_iters=self.sinkhorn_iters,
                )
                self.trunk_projections_carved += 1
                if k_used < n_cubes:
                    self.trunk_early_stops += 1
            else:
                t_gpu, meta = carve_q1_cubes_incremental(
                    w_raw,
                    n_cubes,
                    root_bits=self.root_bits,
                    group_size=self.group_size,
                    use_fwht=self.use_fwht,
                    sinkhorn_iters=self.sinkhorn_iters,
                )
                k_used = n_cubes
            del w_raw
            self.total_cubes_carved += k_used
            active_bytes += _tensor_dict_bytes(t_gpu) + (0 if b_gpu is None else b_gpu.numel() * b_gpu.element_size())
            q1_mod = BinaryHalvingQ1Linear(
                in_f, out_f, t_gpu, meta,
                bias=b_gpu, layer_idx=li, proj_name=sub_name.rsplit(".", 1)[-1],
            )
            q1_mod.active_cubes = k_used
            q1_mod.micro_agent_tol = self.micro_agent_tol
            if self.mirror_v4_trunk:
                q1_mod.mirror_diff_tol = (0.5 * self.trunk_mirror_tol) if in_cone else self.trunk_mirror_tol
                self._trunk_q1_modules.append(q1_mod)
            _set_submodule(module, sub_name, q1_mod)

        for pname, dt in other_params:
            if pname in raw_tensors:
                t_g = raw_tensors[pname].to(device=self.device, dtype=dt)
                active_bytes += t_g.numel() * t_g.element_size()
                set_module_tensor_to_device(module, pname, self.device, value=t_g, dtype=dt, clear_cache=False)

        for bname, bdt in other_buffers:
            if bname in raw_tensors:
                b_g = raw_tensors[bname].to(device=self.device, dtype=bdt)
                set_module_tensor_to_device(module, bname, self.device, value=b_g, dtype=bdt, clear_cache=False)

        del raw_tensors
        self._resident_ready.add(li)
        self.resident_bytes += active_bytes
        if active_bytes > self.peak_active_layer_bytes:
            self.peak_active_layer_bytes = active_bytes
        if self.device.type == "cuda":
            cur_v = int(torch.cuda.memory_allocated(self.device))
            if cur_v > self.peak_vram_bytes:
                self.peak_vram_bytes = cur_v
        return True

    def stats(self) -> Dict[str, Any]:
        return {
            "mode": "unified_q1_cubes" if self.resident_layers > 0 else "method_2_airllm_lazy_binary_halving",
            "num_layers": self.num_layers,
            "resident_layers": self.resident_layers,
            "resident_ready": len(self._resident_ready),
            "resident_layer_hits": self.resident_layer_hits,
            "resident_mib": round(self.resident_bytes / 1024**2, 2),
            "streamed_layers": max(0, self.num_layers - self.resident_layers),
            "cubes_per_proj": self.cubes_per_proj,
            "micro_agent_tol": self.micro_agent_tol,
            "selected_depth_histogram": dict(Counter(v[3] for spec in self._layer_specs for v in spec['linear_map'].values())),
            "root_bits": self.root_bits,
            "rotations": self._relay.rotations if self._relay is not None else self.rotations,
            "transitions": dict(self._relay.transitions if self._relay is not None else self.transitions),
            "standby_bytes": (self._relay.area_used_bytes[self._relay.standby_area]
                              if self._relay is not None else self.standby_bytes),
            "disk_layer_reads": self.disk_layer_reads,
            "host_cache_hits": self.host_cache_hits,
            "host_cache_evictions": self.host_cache_evictions,
            "host_cached_layers": len(self._host_cache),
            "host_cached_mib": round(self._host_cached_bytes / 1024**2, 2),
            "total_cubes_carved": self.total_cubes_carved,
            "peak_active_layer_mib": round(self.peak_active_layer_bytes / 1024**2, 2),
            "peak_vram_mib": round(self.peak_vram_bytes / 1024**2, 2),
            "trunk_mirror_v4": {
                "enabled": self.mirror_v4_trunk,
                "projections_carved": self.trunk_projections_carved,
                "early_stops": self.trunk_early_stops,
                "diff_q1_calls": sum(int(m.mirror_diff_calls) for m in self._trunk_q1_modules),
                "anchored_projections": sum(1 for m in self._trunk_q1_modules if m._mirror_x_last is not None),
            } if self.mirror_v4_trunk else None,
            "moe_cache": self.moe_cache.stats() if self.moe_cache is not None else None,
            "relay": None if self._relay is None else {
                **self._relay.stats(), "handoffs": self.relay_handoffs,
                "disk_prepares": self.relay_disk_prepares,
                "cache_prepares": self.relay_cache_prepares,
                "gpu_prepares": self.relay_gpu_prepares,
                "wait_seconds": round(self.relay_wait_seconds, 6),
                "retained_plan_layers": len(self.layer_proj_cubes),
                "last_trace": dict(self.relay_trace),
            },
        }

    def close(self) -> None:
        if self._relay_pool is not None:
            self._relay_pool.shutdown(wait=True, cancel_futures=True)
            self._relay_pool = None
        self._relay_future = self._relay_layer = None
        if self._relay is not None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self._relay.buffers[:] = [None, None, None]
            self._relay._pinned_host_scratch[:] = [None, None, None]
            self._relay = None
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._resident_ready.clear()
        self.resident_bytes = 0
        self._host_cache.clear()
        self._host_cached_bytes = 0
        self._trunk_q1_modules.clear()
        if self.moe_cache is not None:
            self.moe_cache.close()
        self._installed = False


@torch.no_grad()
def attach_q1_cubes_to_engine(
    eng: Any,
    *,
    cubes_per_proj: int = 4,
    root_bits: int = 8,
    group_size: int = 64,
    resident_layers: int = 0,
    host_cache_bytes: int = 512 * 1024**2,
    moe_gpu_cache_bytes: int = 512 * 1024**2,
    moe_host_cache_bytes: int = 2048 * 1024**2,
    enable_relay: bool = True,
    moe_router_schedule: Optional[Tuple[Tuple[float, int], ...]] = None,
    layer_proj_cubes: Optional[Dict[int, Dict[str, int]]] = None,
    layer_expert_cubes: Optional[Dict[int, Dict[int, Dict[str, int]]]] = None,
    micro_agent_tol: Optional[float] = None,
    moe_fabric: bool = False,
    expert_rest_tol: Optional[float] = None,
    expert_rest_min: int = 3,
    enable_expert_relay: bool = False,
    imprint_puzzle: bool = False,
    imprint_depths: Tuple[int, int, int, int] = (5, 4, 3, 2),
    fixation_cov_thresh: float = 1.0,
    stanok_contour: Any = None,
    graphify_feedback: Any = False,
    mirror_v4_trunk: bool = False,
) -> AirLLMLazyBinaryHalvingRunner:
    """Attach the unified Binary Halving Q1 Cube runner (resident head + AirLLM streamed tail + L1/L2 MoE)
    onto an existing `QuintessentiaModel` instance without altering disk streaming invariants.
    """
    device = torch.device(getattr(eng, "device", "cuda:0"))
    # Detach existing QuintessentiaModel pipeline hooks & release unused BF16 ring arena
    for attr in ("_hook_handles", "_pipeline_hooks"):
        handles = getattr(eng, attr, None)
        if handles:
            for h in list(handles):
                h.remove()
            handles.clear()
    pipe = getattr(eng, "pipeline", None)
    if pipe is not None and getattr(pipe, "num_slots", 0) > 0:
        try:
            pipe.close()
        except Exception:
            pass
        from src.quintessentia_integrated import _NoStreamingPipeline
        eng.pipeline = _NoStreamingPipeline()
    wm = getattr(eng, "weight_manager", None)
    if wm is not None:
        if getattr(wm, "ram_cache", None) is not None:
            wm.ram_cache.clear()
        if getattr(wm, "pinned_pool", None) is not None:
            try:
                wm.pinned_pool.close()
            except Exception:
                pass
    eng.vram_resident_layers = 0

    src = getattr(wm, "model_source", None) if wm is not None else None
    moe_cache: Optional[LazyMoEBinaryHalvingCache] = None
    moe_paths: Dict[int, str] = {}

    if src is not None and getattr(src, "_expert_loader", None) is not None:
        base_loader = src._expert_loader
        if hasattr(base_loader, "base"):
            base_loader = base_loader.base
        # Free the original BF16/uint8 expert arena so VRAM belongs solely to our compact Q1 expert cache
        if getattr(base_loader, "arena", None) is not None:
            base_loader.arena = None
            if hasattr(base_loader, "_lru"):
                base_loader._lru.clear()
            if hasattr(base_loader, "num_slots"):
                base_loader.num_slots = 0

        # Check shape compatibility (gated SwiGLU experts with hidden & inter dims divisible by group_size)
        compatible_layers: List[int] = []
        num_experts_max = 0
        has_biases = getattr(base_loader, "_biases", {}) or {}
        if hasattr(base_loader, "_layer_meta"):
            # GGUFMoE loader
            for l_idx, meta in base_loader._layer_meta.items():
                if l_idx in has_biases:
                    continue
                g_sh = tuple(meta["gate"]["torch_shape"][-2:])
                u_sh = tuple(meta["up"]["torch_shape"][-2:])
                d_sh = tuple(meta["down"]["torch_shape"][-2:])
                if g_sh == u_sh and d_sh == (g_sh[1], g_sh[0]) and g_sh[0] % group_size == 0 and g_sh[1] % group_size == 0:
                    compatible_layers.append(int(l_idx))
                    num_experts_max = max(num_experts_max, int(meta["num_experts"]))
        elif getattr(base_loader, "uniform_shapes", False) and getattr(base_loader, "has_gate", False):
            # HFShardedSource MoE loader
            I_dim, H_dim = int(base_loader.inter_dim), int(base_loader.hidden_dim)
            if I_dim % group_size == 0 and H_dim % group_size == 0:
                for l_idx, n_e in getattr(base_loader, "num_experts_by_layer", {}).items():
                    if l_idx in has_biases:
                        continue
                    compatible_layers.append(int(l_idx))
                    num_experts_max = max(num_experts_max, int(n_e))

        if compatible_layers and num_experts_max > 0:
            container_map = getattr(src, "_routed_expert_container", {}) or {}
            moe_paths = {l_idx: container_map.get(l_idx, "mlp.experts") for l_idx in compatible_layers}
            num_layers_cfg = int(getattr(eng.model.config, "num_hidden_layers", max(compatible_layers) + 1))

            def _single_expert_reader(l_idx: int, eid: int):
                g, up, dn = base_loader.load_batch(l_idx, [eid], device, torch.bfloat16, allow_evict=False)
                return torch.cat([g[0], up[0]], dim=0).contiguous(), dn[0].contiguous()

            def _batch_expert_reader(l_idx: int, eids: List[int]):
                out: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
                for i in range(0, len(eids), 8):
                    chunk = eids[i : i + 8]
                    g, up, dn = base_loader.load_batch(l_idx, chunk, device, torch.bfloat16, allow_evict=False)
                    for j, eid in enumerate(chunk):
                        out[eid] = (torch.cat([g[j], up[j]], dim=0).contiguous(), dn[j].contiguous())
                    del g, up, dn
                return out

            moe_cache = LazyMoEBinaryHalvingCache(
                _single_expert_reader,
                num_layers=num_layers_cfg,
                num_experts=num_experts_max,
                device=device,
                store_cubes=cubes_per_proj,
                root_bits=root_bits,
                group_size=group_size,
                use_fwht=True,
                max_cache_bytes=moe_gpu_cache_bytes if device.type == "cuda" else 0,
                host_cache_bytes=moe_host_cache_bytes,
                cache_on_gpu=(device.type == "cuda"),
                router_depth_schedule=moe_router_schedule,
                expert_batch_reader=_batch_expert_reader,
                expert_cubes={(int(li), int(eid)): (int(ds['gate_up']), int(ds['down']))
                              for li, es in (layer_expert_cubes or {}).items() for eid, ds in es.items()},
                expert_rest_tol=expert_rest_tol,
                expert_rest_min=expert_rest_min,
                enable_expert_relay=enable_expert_relay,
                imprint_puzzle=imprint_puzzle,
                imprint_depths=imprint_depths,
                imprint_auto_vram=bool(imprint_puzzle and device.type == "cuda"),
                fixation_cov_thresh=fixation_cov_thresh,
                stanok_contour=stanok_contour,
                graphify_feedback=graphify_feedback,
            )
            moe_cache.micro_agent_tol = micro_agent_tol
            moe_cache.moe_fabric = bool(moe_fabric)

    def _disk_layer_reader(l_idx: int) -> Dict[str, torch.Tensor]:
        return wm.read_layer_tensors(l_idx, target_device=torch.device("cpu"))

    def _cancel_check() -> None:
        fn = getattr(eng, "_step_cancel_check", None)
        if fn is not None:
            fn()

    runner = AirLLMLazyBinaryHalvingRunner(
        eng.model,
        _disk_layer_reader,
        device=device,
        cubes_per_proj=cubes_per_proj,
        layer_proj_cubes=layer_proj_cubes,
        root_bits=root_bits,
        group_size=group_size,
        use_fwht=True,
        resident_layers=resident_layers,
        host_cache_bytes=host_cache_bytes,
        moe_cache=moe_cache,
        moe_module_paths=moe_paths,
        cancel_check=_cancel_check,
        enable_relay=enable_relay,
        micro_agent_tol=micro_agent_tol,
        mirror_v4_trunk=bool(mirror_v4_trunk),
    ).install()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return runner

