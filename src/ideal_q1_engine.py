"""Ideal Q1 (FWHT + Dual Sinkhorn L2 + Output-Hessian LDLQ + Early-Stop LMHead) & Ideal Q2 Engine.

Implements:
- Pillar 1: Dual Sinkhorn L2 equilibration + orthogonal block Walsh-Hadamard transform (FWHT).
- Pillar 2: Zero-shot output-space Hessian GPTQ / LDLQ column pursuit for multi-plane Q1
  and 2-bit Q2 (`pack_ideal_q1_rows`, `unpack_ideal_q1`, `pack_ideal_q2_rows`, `unpack_ideal_q2`).
- Pillar 3: `IdealQ1Linear`, `IdealQ1LMHead` (exact Sub-Gaussian / Cauchy-Schwarz Early-Stop
  over candidate vocabulary rows + `project_candidates` for SurvivorProbabilityTracker),
  `IdealQ1SwiGLUMLP`, and `MultiStratumLinear` (`2xQ4`, `2xQ2`).
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from src.q1_cube_engine import Q1RoleCycleArena


TILE_ELEMENTS_IDEAL = 1024 * 1024
_FWHT_MATRIX_CACHE: Dict[Tuple[int, str, int], torch.Tensor] = {}


def _fwht_block_butterfly(x: torch.Tensor, group_size: int) -> torch.Tensor:
    orig_shape = x.shape
    k = orig_shape[-1]
    orig_dtype = x.dtype
    n_groups = k // group_size
    y = x.to(torch.float32).reshape(*orig_shape[:-1], n_groups, group_size)
    h = 1
    while h < group_size:
        pairs = group_size // (2 * h)
        y_p = y.reshape(*orig_shape[:-1], n_groups, pairs, 2, h)
        a = y_p[..., 0, :]
        b = y_p[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(*orig_shape[:-1], n_groups, group_size)
        h <<= 1
    y = y * (1.0 / math.sqrt(group_size))
    out = y.reshape(orig_shape)
    return out if orig_dtype == torch.float32 else out.to(orig_dtype)


def _fwht_orth_matrix(group_size: int, device: torch.device, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    key = (int(group_size), str(device), 64 if dtype == torch.float64 else 32)
    cached = _FWHT_MATRIX_CACHE.get(key)
    if cached is not None:
        return cached
    eye = torch.eye(group_size, dtype=torch.float32 if dtype == torch.float32 else torch.float64, device=device)
    mat = _fwht_block_butterfly(eye, group_size).to(dtype=dtype).contiguous()
    _FWHT_MATRIX_CACHE[key] = mat
    return mat


def fwht_block(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Orthogonal block Walsh-Hadamard transform H_bar = H / sqrt(GS) on the last dimension."""
    if group_size <= 0 or (group_size & (group_size - 1)) != 0:
        raise ValueError(f"group_size must be a positive power of 2, got {group_size}")
    orig_shape = x.shape
    k = orig_shape[-1]
    if k % group_size != 0:
        raise ValueError(f"Last dimension {k} must be divisible by group_size={group_size}")
    # Fast CUDA cuBLAS GEMM against cached [GS, GS] Hadamard matrix (1 kernel vs 24 butterfly passes)
    if x.is_cuda and group_size <= 128 and x.numel() >= group_size:
        orig_dtype = x.dtype
        y = x if orig_dtype == torch.float32 else x.to(torch.float32)
        h_mat = _fwht_orth_matrix(group_size, x.device, torch.float32)
        out = torch.matmul(y.reshape(-1, group_size), h_mat).reshape(orig_shape)
        return out if orig_dtype == torch.float32 else out.to(orig_dtype)
    return _fwht_block_butterfly(x, group_size)


@torch.no_grad()
def sinkhorn_equilibrate(
    w: torch.Tensor,
    iters: int = 3,
    *,
    col_scale_fixed: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dual L2 Sinkhorn equilibration returning (row_scale bf16, col_scale bf16, w_eq fp32).

    Divides explicitly by BF16-rounded scales so `row_scale[:, None] * w_eq * col_scale[None, :] == w`
    holds cleanly at BF16 precision. Uses fused allocation-free NVRTC CUDA kernels on BF16 CUDA
    weights (with FP32 PyTorch fallback on non-BF16 CUDA and FP64 on CPU).
    """
    if w.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {tuple(w.shape)}")
    if col_scale_fixed is None and w.is_cuda and w.dtype == torch.bfloat16 and w.shape[0] >= 32 and w.shape[1] >= 32:
        from src.pure_mirror_agents_nvrtc import launch_pure_mirror_sinkhorn_l2
        return launch_pure_mirror_sinkhorn_l2(w, iters=iters)
    work_dtype = torch.float32 if w.is_cuda else torch.float64
    w_d = w if w.dtype == work_dtype else w.to(work_dtype)
    n, k = w_d.shape

    if col_scale_fixed is not None:
        col_scale = col_scale_fixed.to(device=w.device, dtype=torch.bfloat16)
        w_f = w_d if work_dtype == torch.float32 else w.to(torch.float32)
        w_c = w_f / col_scale.to(torch.float32).unsqueeze(0).clamp_min(1e-8)
        r_rms = (w_c.to(work_dtype).pow(2).mean(dim=1) + 1e-12).sqrt().clamp_min(1e-6)
        row_scale = r_rms.to(torch.bfloat16)
        denom = row_scale.to(torch.float32).unsqueeze(1).clamp_min(1e-8) * col_scale.to(torch.float32).unsqueeze(0).clamp_min(1e-8)
        w_eq = w_f / denom
        return row_scale, col_scale, w_eq

    dr = torch.ones((n,), dtype=work_dtype, device=w.device)
    dc = torch.ones((k,), dtype=work_dtype, device=w.device)
    for _ in range(max(1, int(iters))):
        cur = w_d / (dr.unsqueeze(1) * dc.unsqueeze(0))
        r_rms = (cur.pow(2).mean(dim=1) + 1e-12).sqrt()
        dr = (dr * r_rms).clamp_min(1e-6)
        cur = w_d / (dr.unsqueeze(1) * dc.unsqueeze(0))
        c_rms = (cur.pow(2).mean(dim=0) + 1e-12).sqrt()
        dc = (dc * c_rms).clamp_min(1e-6)

    norm_c = dc.mean().clamp_min(1e-8)
    dc = (dc / norm_c).clamp_min(1e-6)
    dr = (dr * norm_c).clamp_min(1e-6)

    row_scale = dr.to(torch.bfloat16)
    col_scale = dc.to(torch.bfloat16)
    denom = row_scale.to(torch.float32).unsqueeze(1).clamp_min(1e-8) * col_scale.to(torch.float32).unsqueeze(0).clamp_min(1e-8)
    w_f = w_d if work_dtype == torch.float32 else w.to(torch.float32)
    w_eq = w_f / denom
    return row_scale, col_scale, w_eq


def ideal_q1_packed_specs(
    name: str,
    shape: Tuple[int, int],
    planes: int = 1,
    group_size: int = 64,
) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
    """Return buffer specifications matching PACKED_FIELDS for bits == 1 (Ideal Q1)."""
    n, k = shape
    prefix = name.removesuffix("weight")
    p = int(planes)
    return {
        prefix + "packed_lo": ((p, n, k // 8), torch.uint8),
        prefix + "qscale": ((p, n, k // group_size), torch.bfloat16),
        prefix + "qzero": ((n, k // group_size), torch.bfloat16),
        prefix + "row_scale": ((n,), torch.bfloat16),
        prefix + "col_scale": ((k,), torch.bfloat16),
        prefix + "res_norm_p1": ((n,), torch.float32),
    }


def ideal_q2_packed_specs(
    name: str,
    shape: Tuple[int, int],
    group_size: int = 64,
) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
    """Return buffer specifications matching PACKED_FIELDS for bits == 2 (Ideal Q2)."""
    n, k = shape
    prefix = name.removesuffix("weight")
    return {
        prefix + "packed_lo": ((n, k // 4), torch.uint8),
        prefix + "qscale": ((n, k // group_size), torch.bfloat16),
        prefix + "qzero": ((n, k // group_size), torch.bfloat16),
        prefix + "row_scale": ((n,), torch.bfloat16),
        prefix + "col_scale": ((k,), torch.bfloat16),
    }


def _build_group_hessian_and_uinv(
    col_scale: torch.Tensor,
    group_size: int,
    damping: float = 0.02,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute output-space Hessian H_eff = H_bar diag(w_c) H_bar + 0.02 I and upper Cholesky(H_eff^-1)."""
    dev = col_scale.device
    k = int(col_scale.numel())
    n_groups = k // group_size
    d_cg = col_scale.to(torch.float64).reshape(n_groups, group_size)
    d2 = d_cg.pow(2)
    w_cg = d2 / d2.mean(dim=-1, keepdim=True).clamp_min(1e-12)

    h_bar = _fwht_orth_matrix(group_size, dev, torch.float64)
    eye = torch.eye(group_size, dtype=torch.float64, device=dev).unsqueeze(0)
    h_eff_d = (h_bar.unsqueeze(0) * w_cg.unsqueeze(1)) @ h_bar.unsqueeze(0) + damping * eye
    h_inv = torch.linalg.inv(h_eff_d)
    h_inv = 0.5 * (h_inv + h_inv.transpose(-1, -2))
    u_inv = torch.linalg.cholesky(h_inv, upper=True).to(torch.float32)
    return h_eff_d.to(torch.float32), u_inv


def _pack_signs_le_u8(signs_pm1: torch.Tensor) -> torch.Tensor:
    """Pack {-1, +1} tensor [rows, K] into little-endian bits [rows, K // 8] uint8."""
    rows, k = signs_pm1.shape
    bits01 = (signs_pm1 >= 0).to(torch.uint8).reshape(rows, k // 8, 8)
    shifts = torch.arange(8, dtype=torch.uint8, device=signs_pm1.device)
    return ((bits01 << shifts).sum(dim=-1)).to(torch.uint8)


def _unpack_signs_le_u8(packed_plane: torch.Tensor) -> torch.Tensor:
    """Unpack little-endian uint8 [rows, K // 8] into {-1.0, +1.0} float32 [rows, K]."""
    rows, k8 = packed_plane.shape
    shifts = torch.arange(8, dtype=torch.uint8, device=packed_plane.device)
    bits01 = ((packed_plane.unsqueeze(-1) >> shifts) & 1).reshape(rows, k8 * 8)
    return torch.where(bits01 > 0, 1.0, -1.0)


@torch.no_grad()
def pack_ideal_q1_rows(
    shape: Tuple[int, int],
    read_rows: Callable[[int, int], torch.Tensor],
    *,
    planes: int = 1,
    q1_planes: Optional[int] = None,
    group_size: int = 64,
    device: torch.device | str = "cpu",
    tile_elements: Optional[int] = None,
    row_alignment: int = 1,
    sinkhorn_iters: int = 3,
    calib_rows: int = 512,
) -> Dict[str, torch.Tensor]:
    """Pack [N, K] into multi-plane Ideal Q1 (Sinkhorn L2 + FWHT + Output-Hessian LDLQ) row-by-row."""
    p_total = int(planes if q1_planes is None else q1_planes)
    if p_total < 1 or p_total > 8:
        raise ValueError(f"planes must be in [1, 8], got {p_total}")
    if group_size not in (16, 32, 64):
        raise ValueError(f"group_size must be in (16, 32, 64), got {group_size}")
    n, k = int(shape[0]), int(shape[1])
    if n < 0 or k <= 0 or k % group_size != 0 or k % 8 != 0:
        raise ValueError(f"Invalid shape {(n, k)} for group_size={group_size}")

    limit = TILE_ELEMENTS_IDEAL if tile_elements is None else int(tile_elements)
    if limit <= 0:
        raise ValueError("tile_elements must be positive")
    alignment = int(row_alignment)
    if alignment <= 0 or limit < min(n, alignment) * k:
        raise ValueError("tile_elements cannot hold one aligned group of weight rows")

    dst = torch.device(device)
    result = {
        key: torch.empty(sp_shape, dtype=dt, device=dst)
        for key, (sp_shape, dt) in ideal_q1_packed_specs("weight", (n, k), p_total, group_size).items()
    }
    if n == 0:
        return result

    n_groups = k // group_size
    rows_per_tile = max(alignment, (limit // k // alignment) * alignment)
    single_band = rows_per_tile >= n

    col_scale_fixed: Optional[torch.Tensor] = None
    if not single_band:
        c_end = min(n, max(alignment, int(calib_rows)))
        calib = read_rows(0, c_end).to(device=dst, dtype=torch.float32)
        _, col_scale_fixed, _ = sinkhorn_equilibrate(calib, iters=sinkhorn_iters)
        del calib

    h_eff: Optional[torch.Tensor] = None
    u_inv: Optional[torch.Tensor] = None

    for start in range(0, n, rows_per_tile):
        end = min(n, start + rows_per_tile)
        r_len = end - start
        tile = read_rows(start, end)
        if tuple(tile.shape) != (r_len, k):
            raise ValueError(f"Weight reader returned {tuple(tile.shape)} for {start}:{end}, expected {(r_len, k)}")
        tile_dev = tile.to(device=dst, dtype=torch.float32)
        row_scale_b, col_scale_b, w_eq = sinkhorn_equilibrate(
            tile_dev, iters=sinkhorn_iters, col_scale_fixed=col_scale_fixed,
        )
        if col_scale_fixed is None:
            col_scale_fixed = col_scale_b
            result["col_scale"].copy_(col_scale_b)
        elif start == 0:
            result["col_scale"].copy_(col_scale_fixed)

        if h_eff is None:
            h_eff, u_inv = _build_group_hessian_and_uinv(col_scale_b, group_size)

        result["row_scale"][start:end].copy_(row_scale_b)
        w_rot = fwht_block(w_eq, group_size).reshape(r_len, n_groups, group_size).to(torch.float32)

        residual = w_rot.clone()
        tail_accum = torch.zeros_like(w_rot) if p_total > 1 else None

        for p_idx in range(p_total):
            alpha_init = 0.85 * (residual.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
            w_cur = residual.clone()
            s_p = torch.empty_like(w_cur)
            for j in range(group_size):
                w_j = w_cur[:, :, j]
                s_j = torch.where(w_j >= 0.0, 1.0, -1.0)
                s_p[:, :, j] = s_j
                err_j = (w_j - alpha_init[:, :, 0] * s_j) / u_inv[:, j, j].unsqueeze(0)
                if j + 1 < group_size:
                    w_cur[:, :, j + 1 :] -= err_j.unsqueeze(-1) * u_inv[:, j, j + 1 :].unsqueeze(0)

            h_s = torch.einsum("gij,rgj->rgi", h_eff, s_p)
            num = (residual * h_s).sum(dim=-1)
            den = (s_p * h_s).sum(dim=-1).clamp_min(1e-12)
            alpha_p = (num / den).clamp_min(0.0).to(torch.bfloat16)

            result["qscale"][p_idx, start:end].copy_(alpha_p)
            result["packed_lo"][p_idx, start:end].copy_(_pack_signs_le_u8(s_p.reshape(r_len, k)))

            step_recon = alpha_p.to(torch.float32).unsqueeze(-1) * s_p
            residual = residual - step_recon
            if p_idx >= 1 and tail_accum is not None:
                tail_accum = tail_accum + step_recon

        qzero_b = residual.mean(dim=-1).to(torch.bfloat16)
        result["qzero"][start:end].copy_(qzero_b)
        if tail_accum is not None:
            res_l2 = tail_accum.reshape(r_len, k).norm(dim=-1)
            result["res_norm_p1"][start:end].copy_(row_scale_b.to(torch.float32).abs() * res_l2)
        else:
            result["res_norm_p1"][start:end].zero_()

        del tile, tile_dev, w_eq, w_rot, residual, tail_accum

    if col_scale_fixed is not None:
        result["col_scale"].copy_(col_scale_fixed)
    return result


@torch.no_grad()
def unpack_ideal_q1(
    packed: Dict[str, torch.Tensor],
    group_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    *,
    active_planes: Optional[int] = None,
    apply_fwht_and_scales: bool = True,
) -> torch.Tensor:
    """Reconstruct Ideal Q1 weight matrix [N, K] from packed dict."""
    lo = packed["packed_lo"]
    if lo.ndim != 3:
        raise ValueError(f"Expected 3D packed_lo [P, N, K//8], got {tuple(lo.shape)}")
    p_total, n, k8 = int(lo.shape[0]), int(lo.shape[1]), int(lo.shape[2])
    k = k8 * 8
    n_groups = k // group_size
    p_act = p_total if active_planes is None else int(active_planes)
    if p_act < 1 or p_act > p_total:
        raise ValueError(f"active_planes ({p_act}) must be in [1, {p_total}]")

    dev = lo.device
    w_rot = packed["qzero"].to(device=dev, dtype=torch.float32).unsqueeze(-1).expand(n, n_groups, group_size).clone()
    for p_idx in range(p_act):
        s_p = _unpack_signs_le_u8(lo[p_idx]).reshape(n, n_groups, group_size)
        alpha = packed["qscale"][p_idx].to(device=dev, dtype=torch.float32).unsqueeze(-1)
        w_rot = w_rot + alpha * s_p

    w_rot_2d = w_rot.reshape(n, k)
    if not apply_fwht_and_scales:
        return w_rot_2d.to(dtype)
    w_spat = fwht_block(w_rot_2d, group_size).to(torch.float32)
    w_spat = w_spat * packed["row_scale"].to(device=dev, dtype=torch.float32).unsqueeze(1)
    w_spat = w_spat * packed["col_scale"].to(device=dev, dtype=torch.float32).unsqueeze(0)
    return w_spat.to(dtype)


@torch.no_grad()
def pack_ideal_q2_rows(
    shape: Tuple[int, int],
    read_rows: Callable[[int, int], torch.Tensor],
    *,
    group_size: int = 64,
    device: torch.device | str = "cpu",
    tile_elements: Optional[int] = None,
    row_alignment: int = 1,
    sinkhorn_iters: int = 3,
    calib_rows: int = 512,
) -> Dict[str, torch.Tensor]:
    """Pack [N, K] into 2-bit Ideal Q2 (Sinkhorn L2 + FWHT + Output-Hessian LDLQ)."""
    if group_size not in (16, 32, 64):
        raise ValueError(f"group_size must be in (16, 32, 64), got {group_size}")
    n, k = int(shape[0]), int(shape[1])
    if n < 0 or k <= 0 or k % group_size != 0 or k % 4 != 0:
        raise ValueError(f"Invalid shape {(n, k)} for group_size={group_size}")

    limit = TILE_ELEMENTS_IDEAL if tile_elements is None else int(tile_elements)
    if limit <= 0:
        raise ValueError("tile_elements must be positive")
    alignment = int(row_alignment)
    if alignment <= 0 or limit < min(n, alignment) * k:
        raise ValueError("tile_elements cannot hold one aligned group of weight rows")

    dst = torch.device(device)
    result = {
        key: torch.empty(sp_shape, dtype=dt, device=dst)
        for key, (sp_shape, dt) in ideal_q2_packed_specs("weight", (n, k), group_size).items()
    }
    if n == 0:
        return result

    n_groups = k // group_size
    rows_per_tile = max(alignment, (limit // k // alignment) * alignment)
    single_band = rows_per_tile >= n

    col_scale_fixed: Optional[torch.Tensor] = None
    if not single_band:
        c_end = min(n, max(alignment, int(calib_rows)))
        calib = read_rows(0, c_end).to(device=dst, dtype=torch.float32)
        _, col_scale_fixed, _ = sinkhorn_equilibrate(calib, iters=sinkhorn_iters)
        del calib

    u_inv: Optional[torch.Tensor] = None

    for start in range(0, n, rows_per_tile):
        end = min(n, start + rows_per_tile)
        r_len = end - start
        tile = read_rows(start, end)
        if tuple(tile.shape) != (r_len, k):
            raise ValueError(f"Weight reader returned {tuple(tile.shape)} for {start}:{end}, expected {(r_len, k)}")
        tile_dev = tile.to(device=dst, dtype=torch.float32)
        row_scale_b, col_scale_b, w_eq = sinkhorn_equilibrate(
            tile_dev, iters=sinkhorn_iters, col_scale_fixed=col_scale_fixed,
        )
        if col_scale_fixed is None:
            col_scale_fixed = col_scale_b
            result["col_scale"].copy_(col_scale_b)
        elif start == 0:
            result["col_scale"].copy_(col_scale_fixed)

        if u_inv is None:
            h_eff, u_inv = _build_group_hessian_and_uinv(col_scale_b, group_size)

        result["row_scale"][start:end].copy_(row_scale_b)
        w_rot = fwht_block(w_eq, group_size).reshape(r_len, n_groups, group_size).to(torch.float32)

        mu = w_rot.mean(dim=-1, keepdim=True)
        sigma = (w_rot - mu).pow(2).mean(dim=-1, keepdim=True).add(1e-12).sqrt()
        clip_c = 1.65
        zero_f = (mu - clip_c * sigma).squeeze(-1)
        scale_f = (2.0 * clip_c * sigma / 3.0).clamp_min(1e-8).squeeze(-1)

        w_cur = w_rot.clone()
        q_int = torch.empty((r_len, n_groups, group_size), dtype=torch.uint8, device=dst)
        for j in range(group_size):
            w_j = w_cur[:, :, j]
            q_j = ((w_j - zero_f) / scale_f).round().clamp(0, 3)
            q_int[:, :, j] = q_j.to(torch.uint8)
            w_hat_j = q_j * scale_f + zero_f
            err_j = (w_j - w_hat_j) / u_inv[:, j, j].unsqueeze(0)
            if j + 1 < group_size:
                w_cur[:, :, j + 1 :] -= err_j.unsqueeze(-1) * u_inv[:, j, j + 1 :].unsqueeze(0)

        # Post-LDLQ closed-form H_eff-optimal affine recalibration (prevents depth norm drift)
        q_f = q_int.to(torch.float32)
        q_mean = q_f.mean(dim=-1)
        w_mean = w_rot.mean(dim=-1)
        dq = q_f - q_mean.unsqueeze(-1)
        dw = w_rot - w_mean.unsqueeze(-1)
        h_dq = torch.einsum("gij,rgj->rgi", h_eff, dq)
        scale_b = ((dw * h_dq).sum(dim=-1) / (dq * h_dq).sum(dim=-1).clamp_min(1e-12)).clamp_min(1e-8).to(torch.bfloat16)
        zero_b = (w_mean - scale_b.to(torch.float32) * q_mean).to(torch.bfloat16)

        q_flat = q_int.reshape(r_len, k)
        packed_b = q_flat[:, 0::4] | (q_flat[:, 1::4] << 2) | (q_flat[:, 2::4] << 4) | (q_flat[:, 3::4] << 6)
        result["packed_lo"][start:end].copy_(packed_b)
        result["qscale"][start:end].copy_(scale_b)
        result["qzero"][start:end].copy_(zero_b)
        del tile, tile_dev, w_eq, w_rot, w_cur, q_int, q_flat, q_f, dq, dw, h_dq

    if col_scale_fixed is not None:
        result["col_scale"].copy_(col_scale_fixed)
    return result


@torch.no_grad()
def unpack_ideal_q2(
    packed: Dict[str, torch.Tensor],
    group_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    *,
    apply_fwht_and_scales: bool = True,
) -> torch.Tensor:
    """Reconstruct Ideal Q2 weight matrix [N, K] from packed dict."""
    lo = packed["packed_lo"]
    n, quarter_k = int(lo.shape[0]), int(lo.shape[1])
    k = quarter_k * 4
    n_groups = k // group_size
    dev = lo.device

    q = torch.stack(tuple((lo >> sh) & 3 for sh in (0, 2, 4, 6)), dim=-1).reshape(n, n_groups, group_size)
    w_rot = (
        q.to(torch.float32) * packed["qscale"].to(device=dev, dtype=torch.float32).unsqueeze(-1)
        + packed["qzero"].to(device=dev, dtype=torch.float32).unsqueeze(-1)
    ).reshape(n, k)
    if not apply_fwht_and_scales:
        return w_rot.to(dtype)
    w_spat = fwht_block(w_rot, group_size).to(torch.float32)
    w_spat = w_spat * packed["row_scale"].to(device=dev, dtype=torch.float32).unsqueeze(1)
    w_spat = w_spat * packed["col_scale"].to(device=dev, dtype=torch.float32).unsqueeze(0)
    return w_spat.to(dtype)


def _cpu_ideal_q1_forward_band(
    x_rot_f: torch.Tensor,
    packed_lo: torch.Tensor,
    qscale: torch.Tensor,
    qzero: torch.Tensor,
    row_scale: torch.Tensor,
    *,
    group_size: int,
    plane_start: int,
    plane_count: int,
    row_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exact tiled CPU / fallback linear forward over rotated activations `x_rot_f = FWHT(x * col_scale)`."""
    dev = x_rot_f.device
    m, k = x_rot_f.shape
    n_groups = k // group_size
    x_grp = x_rot_f.reshape(m, n_groups, group_size)
    x_sum = x_grp.sum(dim=-1)  # [M, G]

    n_full = packed_lo.shape[1]
    rows_total = n_full if row_indices is None else int(row_indices.numel())
    out = torch.empty((m, rows_total), dtype=torch.float32, device=dev)
    band_rows = max(1, TILE_ELEMENTS_IDEAL // k)

    for b0 in range(0, rows_total, band_rows):
        b1 = min(rows_total, b0 + band_rows)
        if row_indices is None:
            sl = slice(b0, b1)
            r_sc = row_scale[sl].to(torch.float32)
            acc = (x_sum @ qzero[sl].to(torch.float32).t()) if plane_start == 0 else torch.zeros((m, b1 - b0), dtype=torch.float32, device=dev)
            for p_idx in range(plane_start, plane_start + plane_count):
                s_p = _unpack_signs_le_u8(packed_lo[p_idx, sl]).reshape(b1 - b0, n_groups, group_size)
                dots = torch.einsum("mgs,rgs->mrg", x_grp, s_p)
                acc += (dots * qscale[p_idx, sl].to(torch.float32).unsqueeze(0)).sum(dim=-1)
        else:
            r_idx = row_indices[b0:b1]
            r_sc = row_scale.index_select(0, r_idx).to(torch.float32)
            acc = (
                (x_sum @ qzero.index_select(0, r_idx).to(torch.float32).t())
                if plane_start == 0
                else torch.zeros((m, b1 - b0), dtype=torch.float32, device=dev)
            )
            for p_idx in range(plane_start, plane_start + plane_count):
                s_p = _unpack_signs_le_u8(packed_lo[p_idx].index_select(0, r_idx)).reshape(b1 - b0, n_groups, group_size)
                dots = torch.einsum("mgs,rgs->mrg", x_grp, s_p)
                acc += (dots * qscale[p_idx].index_select(0, r_idx).to(torch.float32).unsqueeze(0)).sum(dim=-1)
        out[:, b0:b1] = acc * r_sc.unsqueeze(0)
    return out


class IdealQ1Linear(nn.Module):
    """Multi-plane Ideal Q1 linear layer supporting NVRTC FWHT/b1.mma on CUDA and tiled FWHT on CPU."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        planes: int = 4,
        group_size: int = 64,
        bias: Optional[torch.Tensor] = None,
        packed_dict: Optional[Dict[str, torch.Tensor]] = None,
        read_rows: Optional[Callable[[int, int], torch.Tensor]] = None,
        weight: Optional[torch.Tensor] = None,
        active_planes: Optional[int] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bits = 1
        self.group_size = int(group_size)
        self.active_planes: Optional[int] = active_planes

        if packed_dict is not None:
            p_dict = {k: v.detach().clone() for k, v in packed_dict.items()}
            self.planes = int(p_dict["packed_lo"].shape[0])
        elif read_rows is not None or weight is not None:
            reader = read_rows if read_rows is not None else (lambda s, e: weight[s:e])
            dst = device if device is not None else ("cpu" if weight is None else weight.device)
            p_dict = pack_ideal_q1_rows(
                (self.out_features, self.in_features), reader,
                planes=planes, group_size=self.group_size, device=dst,
            )
            self.planes = int(planes)
        else:
            raise ValueError("Provide packed_dict, read_rows, or weight")

        self.q1_planes = self.planes
        for name in ("packed_lo", "qscale", "qzero", "row_scale", "col_scale", "res_norm_p1"):
            self.register_buffer(name, p_dict[name])
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone().to(self.row_scale.device))
        else:
            self.bias = None

        self.calls = 0

    def packed(self) -> Dict[str, torch.Tensor]:
        return {
            name: getattr(self, name)
            for name in ("packed_lo", "qscale", "qzero", "row_scale", "col_scale", "res_norm_p1")
            if hasattr(self, name)
        }

    @classmethod
    def from_linear(
        cls,
        mod: nn.Module,
        *,
        planes: int = 4,
        group_size: int = 64,
    ) -> "IdealQ1Linear":
        bias = mod.bias.detach() if getattr(mod, "bias", None) is not None else None
        if hasattr(mod, "weight") and mod.weight is not None and mod.weight.device.type != "meta":
            return cls(
                mod.in_features, mod.out_features, planes=planes, group_size=group_size,
                bias=bias, weight=mod.weight.detach(),
            )
        from src.quantized_projection import unpack_weight
        if hasattr(mod, "packed"):
            w = unpack_weight(mod.packed(), getattr(mod, "group_size", group_size), dtype=torch.bfloat16)
            return cls(
                mod.in_features, mod.out_features, planes=planes, group_size=group_size,
                bias=bias, weight=w,
            )
        raise ValueError("Cannot construct IdealQ1Linear from meta-device module")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        flat_x = x.reshape(-1, self.in_features)
        m = int(flat_x.shape[0])
        p_act = self.planes if self.active_planes is None else int(self.active_planes)
        if p_act < 1 or p_act > self.planes:
            raise ValueError(f"active_planes ({p_act}) must be in [1, {self.planes}]")

        self.calls += 1
        if m == 0:
            return torch.empty((*orig_shape[:-1], self.out_features), dtype=x.dtype, device=x.device)

        if x.device.type == "cuda" and x.dtype == torch.bfloat16:
            from src.quintessentia_q1_cube_nvrtc import launch_ideal_q1_b1_mma, launch_ideal_q1_fwht_gemv

            y_out = torch.empty((m, self.out_features), dtype=torch.bfloat16, device=x.device)
            if m >= 8:
                launch_ideal_q1_b1_mma(
                    flat_x.contiguous(),
                    self.packed_lo, self.qscale, self.qzero,
                    self.row_scale, self.col_scale, y_out,
                    group_size=self.group_size, total_planes=self.planes,
                    active_planes=p_act, bias=self.bias,
                )
            else:
                launch_ideal_q1_fwht_gemv(
                    flat_x.contiguous(),
                    self.packed_lo, self.qscale, self.qzero,
                    self.row_scale, self.col_scale, y_out,
                    group_size=self.group_size, total_planes=self.planes,
                    plane_start=0, plane_count=p_act, bias=self.bias, accumulate=False,
                )
            return y_out.view(*orig_shape[:-1], self.out_features)

        x_rot = fwht_block(flat_x.to(torch.float32) * self.col_scale.to(torch.float32).unsqueeze(0), self.group_size)
        y_f32 = _cpu_ideal_q1_forward_band(
            x_rot, self.packed_lo, self.qscale, self.qzero, self.row_scale,
            group_size=self.group_size, plane_start=0, plane_count=p_act,
        )
        if self.bias is not None:
            y_f32 += self.bias.to(device=x.device, dtype=torch.float32).unsqueeze(0)
        return y_f32.to(x.dtype).view(*orig_shape[:-1], self.out_features)


class IdealQ1LMHead(nn.Module):
    """Ideal Q1 Vocabulary Head with Exact Sub-Gaussian / Cauchy-Schwarz Early-Stop & Candidate Support."""

    def __init__(
        self,
        in_features: int,
        vocab_size: int,
        *,
        planes: int = 4,
        group_size: int = 64,
        early_stop: bool = True,
        early_stop_z: Optional[float] = 6.0,
        min_keep_rows: int = 128,
        bias: Optional[torch.Tensor] = None,
        packed_dict: Optional[Dict[str, torch.Tensor]] = None,
        read_rows: Optional[Callable[[int, int], torch.Tensor]] = None,
        weight_slice: Any = None,
        weight: Optional[torch.Tensor] = None,
        slot_budget_bytes: Optional[int] = None,
        arena: Optional[Q1RoleCycleArena] = None,
        active_planes: Optional[int] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.vocab_size = int(vocab_size)
        self.out_features = self.vocab_size
        self.bits = 1
        self.group_size = int(group_size)
        self.early_stop = bool(early_stop)
        self.early_stop_z = early_stop_z
        self.min_keep_rows = max(1, int(min_keep_rows))
        self.slot_budget_bytes = slot_budget_bytes
        self.arena = arena
        self.active_planes: Optional[int] = active_planes

        if packed_dict is not None:
            p_dict = {k: v.detach().clone() for k, v in packed_dict.items()}
            self.planes = int(p_dict["packed_lo"].shape[0])
        elif read_rows is not None or weight_slice is not None or weight is not None:
            if read_rows is not None:
                reader = read_rows
            elif weight is not None:
                reader = lambda s, e: weight[s:e]
            else:
                reader = lambda s, e: weight_slice[s:e]
            dst = device if device is not None else ("cpu" if weight is None else weight.device)
            p_dict = pack_ideal_q1_rows(
                (self.vocab_size, self.in_features), reader,
                planes=planes, group_size=self.group_size, device=dst,
            )
            self.planes = int(planes)
        else:
            raise ValueError("Provide packed_dict, read_rows, weight_slice, or weight")

        self.q1_planes = self.planes
        for name in ("packed_lo", "qscale", "qzero", "row_scale", "col_scale", "res_norm_p1"):
            self.register_buffer(name, p_dict[name])
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone().to(self.row_scale.device))
        else:
            self.bias = None

        self.calls = 0
        self.early_stop_calls = 0
        self.candidate_calls = 0
        self._total_vocab_rows_evaluated = 0
        self._total_plane_rows_evaluated = 0

    def packed(self) -> Dict[str, torch.Tensor]:
        return {
            name: getattr(self, name)
            for name in ("packed_lo", "qscale", "qzero", "row_scale", "col_scale", "res_norm_p1")
            if hasattr(self, name)
        }

    @classmethod
    def from_packed_linear(
        cls,
        mod: nn.Module,
        *,
        planes: int = 4,
        group_size: int = 64,
        early_stop: bool = True,
        early_stop_z: Optional[float] = 6.0,
        min_keep_rows: int = 128,
        slot_budget_bytes: Optional[int] = None,
        arena: Optional[Q1RoleCycleArena] = None,
    ) -> "IdealQ1LMHead":
        bias = mod.bias.detach() if getattr(mod, "bias", None) is not None else None
        if getattr(mod, "bits", None) == 1 and hasattr(mod, "res_norm_p1"):
            return cls(
                mod.in_features, mod.out_features, planes=getattr(mod, "q1_planes", planes),
                group_size=getattr(mod, "group_size", group_size),
                early_stop=early_stop, early_stop_z=early_stop_z,
                min_keep_rows=min_keep_rows, bias=bias, packed_dict=mod.packed(),
                slot_budget_bytes=slot_budget_bytes, arena=arena,
                active_planes=getattr(mod, "active_planes", None),
            )
        if hasattr(mod, "weight") and mod.weight is not None and mod.weight.device.type != "meta":
            w = mod.weight.detach()
            return cls(
                mod.in_features, mod.out_features, planes=planes, group_size=group_size,
                early_stop=early_stop, early_stop_z=early_stop_z,
                min_keep_rows=min_keep_rows, bias=bias, weight=w,
                slot_budget_bytes=slot_budget_bytes, arena=arena,
            )
        from src.quantized_projection import unpack_weight
        w = unpack_weight(mod.packed(), getattr(mod, "group_size", group_size), dtype=torch.bfloat16)
        return cls(
            mod.in_features, mod.out_features, planes=planes, group_size=group_size,
            early_stop=early_stop, early_stop_z=early_stop_z,
            min_keep_rows=min_keep_rows, bias=bias, weight=w,
            slot_budget_bytes=slot_budget_bytes, arena=arena,
        )

    def _select_survivor_rows(self, y1_row: torch.Tensor, x_norm_factor: float) -> torch.Tensor:
        """Select surviving candidate row indices via Sub-Gaussian/Cauchy-Schwarz margin + top min_keep_rows."""
        top1_score = y1_row.max()
        if self.early_stop_z is None:
            margin = self.res_norm_p1 * x_norm_factor
        else:
            margin = (float(self.early_stop_z) / math.sqrt(self.in_features)) * x_norm_factor * self.res_norm_p1
        threshold = top1_score - margin[int(y1_row.argmax().item())]
        survivor_mask = (y1_row + margin) >= threshold
        k_keep = min(self.min_keep_rows, self.vocab_size)
        if k_keep > 0:
            _, topk_idx = torch.topk(y1_row, k_keep, sorted=False)
            survivor_mask[topk_idx] = True
        return torch.nonzero(survivor_mask, as_tuple=False).reshape(-1).to(torch.int32)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        flat_x = x.reshape(-1, self.in_features)
        m = int(flat_x.shape[0])
        p_act = self.planes if self.active_planes is None else int(self.active_planes)
        if p_act < 1 or p_act > self.planes:
            raise ValueError(f"active_planes ({p_act}) must be in [1, {self.planes}]")

        self.calls += 1
        if m == 0:
            return torch.empty((*orig_shape[:-1], self.vocab_size), dtype=x.dtype, device=x.device)

        use_early_stop = self.early_stop and p_act > 1 and m == 1
        is_cuda_bf16 = x.device.type == "cuda" and x.dtype == torch.bfloat16

        if not use_early_stop:
            self._total_vocab_rows_evaluated += m * self.vocab_size
            self._total_plane_rows_evaluated += m * self.vocab_size * p_act
            if is_cuda_bf16:
                from src.quintessentia_q1_cube_nvrtc import launch_ideal_q1_b1_mma, launch_ideal_q1_fwht_gemv
                y_out = torch.empty((m, self.vocab_size), dtype=torch.bfloat16, device=x.device)
                if m >= 8:
                    launch_ideal_q1_b1_mma(
                        flat_x.contiguous(),
                        self.packed_lo, self.qscale, self.qzero,
                        self.row_scale, self.col_scale, y_out,
                        group_size=self.group_size, total_planes=self.planes,
                        active_planes=p_act, bias=self.bias,
                    )
                else:
                    launch_ideal_q1_fwht_gemv(
                        flat_x.contiguous(),
                        self.packed_lo, self.qscale, self.qzero,
                        self.row_scale, self.col_scale, y_out,
                        group_size=self.group_size, total_planes=self.planes,
                        plane_start=0, plane_count=p_act, bias=self.bias, accumulate=False,
                    )
                return y_out.view(*orig_shape[:-1], self.vocab_size)

            x_rot = fwht_block(flat_x.to(torch.float32) * self.col_scale.to(torch.float32).unsqueeze(0), self.group_size)
            y_f32 = _cpu_ideal_q1_forward_band(
                x_rot, self.packed_lo, self.qscale, self.qzero, self.row_scale,
                group_size=self.group_size, plane_start=0, plane_count=p_act,
            )
            if self.bias is not None:
                y_f32 += self.bias.to(device=x.device, dtype=torch.float32).unsqueeze(0)
            return y_f32.to(x.dtype).view(*orig_shape[:-1], self.vocab_size)

        # Dynamic Exact / Sub-Gaussian Early-Stop (M == 1, p_act > 1):
        # Step 1: Compute Plane 0 across full vocabulary
        self.early_stop_calls += 1
        x_scaled = flat_x.to(torch.float32) * self.col_scale.to(torch.float32).unsqueeze(0)
        x_norm = float(x_scaled.norm(dim=-1)[0].item())

        if is_cuda_bf16:
            from src.quintessentia_q1_cube_nvrtc import launch_ideal_q1_fwht_gemv
            y_acc = torch.empty((1, self.vocab_size), dtype=torch.float32, device=x.device)
            launch_ideal_q1_fwht_gemv(
                flat_x.contiguous(),
                self.packed_lo, self.qscale, self.qzero,
                self.row_scale, self.col_scale, y_acc,
                group_size=self.group_size, total_planes=self.planes,
                plane_start=0, plane_count=1, bias=self.bias, accumulate=False,
            )
            cand_rows = self._select_survivor_rows(y_acc[0], x_norm)
            if int(cand_rows.numel()) > 0:
                launch_ideal_q1_fwht_gemv(
                    flat_x.contiguous(),
                    self.packed_lo, self.qscale, self.qzero,
                    self.row_scale, self.col_scale, y_acc,
                    group_size=self.group_size, total_planes=self.planes,
                    plane_start=1, plane_count=p_act - 1, cand_rows=cand_rows,
                    bias=None, accumulate=True,
                )
            self._total_vocab_rows_evaluated += self.vocab_size
            self._total_plane_rows_evaluated += self.vocab_size + int(cand_rows.numel()) * (p_act - 1)
            return y_acc.to(x.dtype).view(*orig_shape[:-1], self.vocab_size)

        x_rot = fwht_block(x_scaled, self.group_size)
        y_acc = _cpu_ideal_q1_forward_band(
            x_rot, self.packed_lo, self.qscale, self.qzero, self.row_scale,
            group_size=self.group_size, plane_start=0, plane_count=1,
        )
        if self.bias is not None:
            y_acc += self.bias.to(device=x.device, dtype=torch.float32).unsqueeze(0)
        cand_rows = self._select_survivor_rows(y_acc[0], x_norm)
        if int(cand_rows.numel()) > 0:
            cand_l = cand_rows.to(torch.long)
            y_tail = _cpu_ideal_q1_forward_band(
                x_rot, self.packed_lo, self.qscale, self.qzero, self.row_scale,
                group_size=self.group_size, plane_start=1, plane_count=p_act - 1, row_indices=cand_l,
            )
            y_acc.index_add_(1, cand_l, y_tail)
        self._total_vocab_rows_evaluated += self.vocab_size
        self._total_plane_rows_evaluated += self.vocab_size + int(cand_rows.numel()) * (p_act - 1)
        return y_acc.to(x.dtype).view(*orig_shape[:-1], self.vocab_size)

    @torch.no_grad()
    def project_candidates(self, x: torch.Tensor, cand_ids: torch.Tensor) -> torch.Tensor:
        """Compute FP32 logits [..., C] for candidate token ids without touching non-candidate rows or disk."""
        orig_shape = x.shape
        flat_x = x.reshape(-1, self.in_features)
        m = int(flat_x.shape[0])
        ids = cand_ids.reshape(-1).to(device=x.device, dtype=torch.long)
        num_cands = int(ids.numel())
        if num_cands == 0 or m == 0:
            return torch.empty((*orig_shape[:-1], num_cands), dtype=torch.float32, device=x.device)
        if bool((ids < 0).any().item()) or bool((ids >= self.vocab_size).any().item()):
            raise IndexError("Candidate token id out of LM head vocabulary bounds")

        p_act = self.planes if self.active_planes is None else int(self.active_planes)
        if p_act < 1 or p_act > self.planes:
            raise ValueError(f"active_planes ({p_act}) must be in [1, {self.planes}]")

        x_rot = fwht_block(flat_x.to(torch.float32) * self.col_scale.to(torch.float32).unsqueeze(0), self.group_size)
        out_f32 = _cpu_ideal_q1_forward_band(
            x_rot, self.packed_lo, self.qscale, self.qzero, self.row_scale,
            group_size=self.group_size, plane_start=0, plane_count=p_act, row_indices=ids,
        )
        if self.bias is not None:
            out_f32 += self.bias.to(device=x.device, dtype=torch.float32).index_select(0, ids).unsqueeze(0)
        self.candidate_calls += 1
        return out_f32.reshape(*orig_shape[:-1], num_cands)

    def stats(self) -> Dict[str, object]:
        p_act = self.planes if self.active_planes is None else int(self.active_planes)
        if self._total_vocab_rows_evaluated > 0:
            avg_eff = self._total_plane_rows_evaluated / float(self._total_vocab_rows_evaluated)
            full_plane_rows = self._total_vocab_rows_evaluated * max(1, p_act)
            pruned_pct = 100.0 * max(0.0, 1.0 - self._total_plane_rows_evaluated / float(full_plane_rows))
        else:
            avg_eff = float(p_act)
            pruned_pct = 0.0
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "vocab_size": self.vocab_size,
            "bits": self.bits,
            "q1_planes": self.planes,
            "planes": self.planes,
            "active_planes": self.active_planes,
            "early_stop_enabled": self.early_stop,
            "effective_bits": avg_eff,
            "avg_effective_bits": avg_eff,
            "vocab_rows_pruned_pct": pruned_pct,
            "calls": self.calls,
            "early_stop_calls": self.early_stop_calls,
            "candidate_calls": self.candidate_calls,
        }


class IdealQ1SwiGLUMLP(nn.Module):
    """Ideal Q1 SwiGLU MLP block (`gate`, `up` -> SiLU * up -> `down`) with FP32 intermediate accumulation."""

    def __init__(
        self,
        gate_proj: IdealQ1Linear,
        up_proj: IdealQ1Linear,
        down_proj: IdealQ1Linear,
    ) -> None:
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    @classmethod
    def from_weights(
        cls,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        *,
        planes: int = 4,
        group_size: int = 64,
    ) -> "IdealQ1SwiGLUMLP":
        return cls(
            IdealQ1Linear(w_gate.shape[1], w_gate.shape[0], planes=planes, group_size=group_size, weight=w_gate),
            IdealQ1Linear(w_up.shape[1], w_up.shape[0], planes=planes, group_size=group_size, weight=w_up),
            IdealQ1Linear(w_down.shape[1], w_down.shape[0], planes=planes, group_size=group_size, weight=w_down),
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate_proj(x).float()
        u = self.up_proj(x).float()
        inter = (F.silu(g) * u).to(x.dtype)
        return self.down_proj(inter)


class MultiStratumLinear(nn.Module):
    """Residual multi-stratum quantization (`2xQ4` = 8-bit or `2xQ2` = 4-bit) over Q1RoleCycleArena."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        *,
        stratum_bits: int = 4,
        strata: int = 2,
        group_size: int = 64,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if stratum_bits not in (2, 4):
            raise ValueError("stratum_bits must be 2 (2xQ2) or 4 (2xQ4)")
        if strata < 1:
            raise ValueError("strata must be >= 1")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.stratum_bits = int(stratum_bits)
        self.strata = int(strata)
        self.group_size = int(group_size)
        self.mode_name = f"{self.strata}xQ{self.stratum_bits}"

        from src.quantized_projection import pack_weight, unpack_weight

        dev = weight.device
        residual = weight.detach().to(torch.float32).clone()
        self.stratum_dicts: List[Dict[str, torch.Tensor]] = []
        for _ in range(self.strata):
            if self.stratum_bits == 4:
                p = pack_weight(residual, bits=4, group_size=self.group_size, device=dev)
                recon = unpack_weight(p, group_size=self.group_size, dtype=torch.float32)
            else:
                p = pack_ideal_q2_rows(
                    (self.out_features, self.in_features),
                    lambda r0, r1, res=residual: res[r0:r1],
                    group_size=self.group_size,
                    device=dev,
                )
                recon = unpack_ideal_q2(p, group_size=self.group_size, dtype=torch.float32)
            self.stratum_dicts.append(p)
            residual = residual - recon

        if bias is not None:
            self.register_buffer("bias", bias.detach().clone().to(dev))
        else:
            self.bias = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        flat_x = x.reshape(-1, self.in_features)
        y_acc = torch.zeros((flat_x.shape[0], self.out_features), dtype=torch.float32, device=x.device)
        if self.stratum_bits == 4:
            from src.quantized_projection import unpack_weight
            for p in self.stratum_dicts:
                if x.device.type == "cuda" and flat_x.dtype == torch.bfloat16:
                    from src.quintessentia_nvrtc import can_use_fused_dense_q4, fused_dense_linear_q4
                    if can_use_fused_dense_q4(flat_x, p["packed_lo"], p["qscale"], p["qzero"], self.group_size):
                        y_acc += fused_dense_linear_q4(flat_x, p["packed_lo"], p["qscale"], p["qzero"], None, self.group_size).float()
                        continue
                w = unpack_weight({k: v.to(x.device) for k, v in p.items()}, self.group_size, torch.float32)
                y_acc += F.linear(flat_x.float(), w)
        else:
            for p in self.stratum_dicts:
                if x.device.type == "cuda" and flat_x.dtype == torch.bfloat16:
                    from src.quintessentia_q1_cube_nvrtc import launch_quintessentia_q2_fwht_gemv
                    y_s = torch.empty((flat_x.shape[0], self.out_features), dtype=torch.float32, device=x.device)
                    launch_quintessentia_q2_fwht_gemv(
                        flat_x.contiguous(),
                        p["packed_lo"].to(x.device), p["qscale"].to(x.device), p["qzero"].to(x.device),
                        p["row_scale"].to(x.device), p["col_scale"].to(x.device), y_s,
                        group_size=self.group_size,
                    )
                    y_acc += y_s
                else:
                    w = unpack_ideal_q2({k: v.to(x.device) for k, v in p.items()}, self.group_size, torch.float32)
                    y_acc += F.linear(flat_x.float(), w)

        if self.bias is not None:
            y_acc += self.bias.to(device=x.device, dtype=torch.float32).unsqueeze(0)
        return y_acc.to(x.dtype).reshape(*orig_shape[:-1], self.out_features)
