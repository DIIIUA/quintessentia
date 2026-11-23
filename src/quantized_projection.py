"""Bounded Colibri affine Q4/Q6 storage shared by streaming and compute.

This is a runtime representation, not a GGUF block format. Source quantization
is decoded by the source adapter before conversion. Routers and parameters
consumed directly by architecture-specific kernels are deliberately preserved.
"""
from __future__ import annotations

import inspect
import math
import torch
from torch import nn
from torch.nn import functional as F

PACKED_FIELDS = ("packed_lo", "packed_hi", "qscale", "qzero", "row_scale", "col_scale", "res_norm_p1")
TILE_ELEMENTS = 1024 * 1024


def packed_specs(name, shape, bits, group_size=64, *, q1_planes=1):
    n, k = shape
    prefix = name.removesuffix("weight")
    if bits == 1:
        from src.ideal_q1_engine import ideal_q1_packed_specs
        return ideal_q1_packed_specs(name, shape, planes=max(1, int(q1_planes)), group_size=group_size)
    if bits == 2:
        from src.ideal_q1_engine import ideal_q2_packed_specs
        return ideal_q2_packed_specs(name, shape, group_size=group_size)
    if bits == 8:
        return {
            prefix + "packed_lo": ((n, k), torch.uint8),
            prefix + "qscale": ((n, k // group_size), torch.bfloat16),
            prefix + "qzero": ((n, k // group_size), torch.bfloat16),
        }
    result = {
        prefix + "packed_lo": ((n, k // 2), torch.uint8),
        prefix + "qscale": ((n, k // group_size), torch.bfloat16),
        prefix + "qzero": ((n, k // group_size), torch.bfloat16),
    }
    if bits == 6:
        result[prefix + "packed_hi"] = ((n, k // 4), torch.uint8)
    return result


@torch.no_grad()
def pack_weight(weight, bits, group_size=64, device=None, *, q1_planes=1):
    """Quantize in bounded row tiles; never allocate a full FP32 weight copy."""
    dst = torch.device(device) if device is not None else weight.device
    return pack_weight_rows(weight.shape, lambda start, end: weight[start:end],
                            bits, group_size, dst, q1_planes=q1_planes)


@torch.no_grad()
def pack_weight_rows(shape, read_rows, bits, group_size=64, device="cpu", tile_elements=None,
                     row_alignment=1, *, q1_planes=1):
    """Pack successive source tiles directly into the final Q1/Q2/Q4/Q6/Q8 allocation.

    read_rows(start, end) returns [end-start, in_features]. No full uncompressed
    source matrix is needed. The final packed matrix still must fit the caller's
    resident/streaming slot budget.
    """
    if bits not in (1, 2, 4, 6, 8) or group_size not in (16, 32, 64):
        raise ValueError("Colibri supports Q1/Q2/Q4/Q6/Q8 with groups of 16, 32 or 64")
    if bits == 1:
        from src.ideal_q1_engine import pack_ideal_q1_rows
        return pack_ideal_q1_rows(
            shape,
            read_rows,
            planes=max(1, int(q1_planes)),
            group_size=group_size,
            device=device,
            tile_elements=tile_elements,
            row_alignment=row_alignment,
        )
    if bits == 2:
        from src.ideal_q1_engine import pack_ideal_q2_rows
        return pack_ideal_q2_rows(
            shape,
            read_rows,
            group_size=group_size,
            device=device,
            tile_elements=tile_elements,
            row_alignment=row_alignment,
        )
    n, k = shape
    if n < 0 or k <= 0 or k % group_size:
        raise ValueError("Input width must be divisible by group size")
    limit = TILE_ELEMENTS if tile_elements is None else int(tile_elements)
    if limit <= 0:
        raise ValueError("tile_elements must be positive")
    alignment = int(row_alignment)
    if alignment <= 0 or limit < min(n, alignment) * k:
        raise ValueError("tile_elements cannot hold one aligned group of weight rows")
    dst = torch.device(device)
    result = {key: torch.empty(shape, dtype=dt, device=dst)
              for key, (shape, dt) in packed_specs("weight", (n, k), bits, group_size).items()}
    rows = max(alignment, (limit // k // alignment) * alignment)
    use_nvrtc_pack = dst.type == "cuda" and bits in (4, 6, 8)
    if use_nvrtc_pack:
        from src.colibri_q1_cube_nvrtc import launch_colibri_pack_affine
    for start in range(0, n, rows):
        end = min(n, start + rows)
        tile = read_rows(start, end)
        if tuple(tile.shape) != (end-start, k):
            raise ValueError(f"Weight reader returned {tuple(tile.shape)} for rows {start}:{end}, expected {(end-start, k)}")
        if use_nvrtc_pack:
            t_in = tile if tile.dtype in (torch.bfloat16, torch.float32) else tile.to(dtype=torch.float32)
            t_in = t_in.to(device=dst).contiguous()
            launch_colibri_pack_affine(
                t_in,
                result["packed_lo"][start:end],
                result["packed_hi"][start:end] if bits == 6 else None,
                result["qscale"][start:end],
                result["qzero"][start:end],
                bits=bits,
                group_size=group_size,
            )
            del tile, t_in
            continue
        w = tile.to(device=dst, dtype=torch.float32).reshape(-1, k // group_size, group_size)
        zero = w.amin(-1).to(torch.bfloat16)
        # CPU and CUDA FP32 division can land on different sides of a BF16
        # midpoint. Keep rounding decisions in FP64 so placement does not alter
        # the packed weights. Conversion still operates on a bounded row tile.
        scale = ((w.amax(-1).double() - zero.double()) / ((1 << bits) - 1)).clamp_min_(1e-8).to(torch.bfloat16)
        q = ((w.double() - zero.double().unsqueeze(-1)) / scale.double().unsqueeze(-1)).round_().clamp_(0, (1 << bits) - 1).to(torch.uint8).reshape(-1, k)
        if bits == 8:
            result["packed_lo"][start:end] = q
        else:
            result["packed_lo"][start:end] = (q[:, 0::2] & 15) | ((q[:, 1::2] & 15) << 4)
            if bits == 6:
                hi = q >> 4
                result["packed_hi"][start:end] = hi[:, 0::4] | (hi[:, 1::4] << 2) | (hi[:, 2::4] << 4) | (hi[:, 3::4] << 6)
                del hi
        result["qscale"][start:end] = scale
        result["qzero"][start:end] = zero
        del tile, w, zero, scale, q
    return result


def unpack_weight(packed, group_size=64, dtype=torch.bfloat16, *, active_planes=None, apply_fwht_and_scales=True):
    """Decode a caller-bounded tile (also used as the numerical reference)."""
    lo = packed["packed_lo"]
    if "res_norm_p1" in packed or lo.ndim == 3:
        from src.ideal_q1_engine import unpack_ideal_q1
        return unpack_ideal_q1(
            packed,
            group_size=group_size,
            dtype=dtype,
            active_planes=active_planes,
            apply_fwht_and_scales=apply_fwht_and_scales,
        )
    if "row_scale" in packed and "col_scale" in packed:
        from src.ideal_q1_engine import unpack_ideal_q2
        return unpack_ideal_q2(
            packed,
            group_size=group_size,
            dtype=dtype,
            apply_fwht_and_scales=apply_fwht_and_scales,
        )
    n, half = lo.shape
    qs = packed["qscale"]
    hi = packed.get("packed_hi")
    k_full = qs.shape[1] * group_size
    if lo.is_cuda and dtype in (torch.bfloat16, torch.float32) and group_size in (16, 32, 64):
        from src.colibri_q1_cube_nvrtc import launch_colibri_unpack_affine
        bits = 8 if (hi is None and half == k_full) else (6 if hi is not None else 4)
        out = torch.empty((n, k_full), dtype=dtype, device=lo.device)
        launch_colibri_unpack_affine(lo, hi, qs, packed["qzero"], out, bits=bits, group_size=group_size)
        return out
    if hi is None and half == k_full:
        w = lo.float().reshape(n, -1, group_size)
        return (w * qs.float().unsqueeze(-1) + packed["qzero"].float().unsqueeze(-1)).reshape(n, -1).to(dtype)
    q = torch.stack((lo & 15, lo >> 4), dim=-1).reshape(n, half * 2)
    if hi is not None:
        upper = torch.stack(tuple((hi >> shift) & 3 for shift in (0, 2, 4, 6)), dim=-1).reshape_as(q)
        q = q | (upper << 4)
    w = q.float().reshape(n, -1, group_size)
    return (w * qs.float().unsqueeze(-1) + packed["qzero"].float().unsqueeze(-1)).reshape(n, -1).to(dtype)


def _slice_packed_rows(packed, start, end):
    sub = {}
    for k, v in packed.items():
        if k == "col_scale":
            sub[k] = v
        elif v.ndim == 3:
            sub[k] = v[:, start:end]
        else:
            sub[k] = v[start:end]
    return sub


class PackedLinear(nn.Module):
    def __init__(self, in_features, out_features, bits, bias=False, group_size=64, *, q1_planes=1):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.bits, self.group_size = int(bits), int(group_size)
        self.q1_planes = max(1, int(q1_planes)) if self.bits == 1 else self.bits
        for name, (shape, dt) in packed_specs(
            "weight", (out_features, in_features), self.bits, self.group_size, q1_planes=self.q1_planes
        ).items():
            self.register_buffer(name, torch.empty(shape, dtype=dt, device="meta"))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16, device="meta"), requires_grad=False) if bias else None
        self.active_planes: int | None = None
        self.fused_calls = 0
        self.tiled_calls = 0

    def packed(self):
        return {name: getattr(self, name) for name in PACKED_FIELDS if hasattr(self, name)}

    def forward(self, x):
        if self.bits == 1:
            total_p = max(1, int(getattr(self, "q1_planes", 1)))
            p_act = total_p if self.active_planes is None else int(self.active_planes)
            if p_act < 1 or p_act > total_p:
                raise ValueError(f"active_planes ({p_act}) must be in [1, {total_p}]")
            flat = x.reshape(-1, self.in_features)
            if x.device.type == "cuda" and x.dtype == torch.bfloat16:
                from src.colibri_q1_cube_nvrtc import launch_ideal_q1_b1_mma, launch_ideal_q1_fwht_gemv
                self.fused_calls += 1
                y_out = torch.empty((flat.shape[0], self.out_features), dtype=torch.bfloat16, device=x.device)
                if flat.shape[0] >= 8:
                    launch_ideal_q1_b1_mma(
                        flat.contiguous(), self.packed_lo, self.qscale, self.qzero,
                        self.row_scale, self.col_scale, y_out,
                        group_size=self.group_size, total_planes=total_p, active_planes=p_act, bias=self.bias,
                    )
                elif flat.shape[0] > 0:
                    launch_ideal_q1_fwht_gemv(
                        flat.contiguous(), self.packed_lo, self.qscale, self.qzero,
                        self.row_scale, self.col_scale, y_out,
                        group_size=self.group_size, total_planes=total_p, plane_start=0, plane_count=p_act, bias=self.bias,
                    )
                return y_out.view(*x.shape[:-1], self.out_features)
            from src.ideal_q1_engine import fwht_block
            self.tiled_calls += 1
            x_rot = fwht_block(flat.float() * self.col_scale.float(), self.group_size)
            out = torch.empty((flat.shape[0], self.out_features), dtype=x.dtype, device=x.device)
            rows = max(1, TILE_ELEMENTS // self.in_features)
            packed = self.packed()
            for start in range(0, self.out_features, rows):
                end = min(self.out_features, start + rows)
                w_rot = unpack_weight(
                    _slice_packed_rows(packed, start, end),
                    self.group_size, torch.float32, active_planes=p_act, apply_fwht_and_scales=False,
                )
                y_t = F.linear(x_rot, w_rot) * self.row_scale[start:end].float()
                if self.bias is not None:
                    y_t = y_t + self.bias[start:end].float()
                out[:, start:end] = y_t.to(x.dtype)
            return out.view(*x.shape[:-1], self.out_features)

        if self.bits == 2:
            if self.active_planes is not None:
                p_act = int(self.active_planes)
                if p_act < 1 or p_act > 2:
                    raise ValueError(f"active_planes ({p_act}) must be in [1, 2]")
            flat = x.reshape(-1, self.in_features)
            if x.device.type == "cuda" and x.dtype == torch.bfloat16:
                from src.colibri_q1_cube_nvrtc import launch_colibri_q2_fwht_gemv
                self.fused_calls += 1
                y_out = torch.empty((flat.shape[0], self.out_features), dtype=torch.bfloat16, device=x.device)
                if flat.shape[0] > 0:
                    launch_colibri_q2_fwht_gemv(
                        flat.contiguous(), self.packed_lo, self.qscale, self.qzero,
                        self.row_scale, self.col_scale, y_out,
                        group_size=self.group_size, bias=self.bias,
                    )
                return y_out.view(*x.shape[:-1], self.out_features)
            from src.ideal_q1_engine import fwht_block
            self.tiled_calls += 1
            x_rot = fwht_block(flat.float() * self.col_scale.float(), self.group_size)
            out = torch.empty((flat.shape[0], self.out_features), dtype=x.dtype, device=x.device)
            rows = max(1, TILE_ELEMENTS // self.in_features)
            packed = self.packed()
            for start in range(0, self.out_features, rows):
                end = min(self.out_features, start + rows)
                w_rot = unpack_weight(
                    _slice_packed_rows(packed, start, end),
                    self.group_size, torch.float32, apply_fwht_and_scales=False,
                )
                y_t = F.linear(x_rot, w_rot) * self.row_scale[start:end].float()
                if self.bias is not None:
                    y_t = y_t + self.bias[start:end].float()
                out[:, start:end] = y_t.to(x.dtype)
            return out.view(*x.shape[:-1], self.out_features)

        if self.bits == 8:
            flat = x.reshape(-1, self.in_features)
            p_act = self.bits if self.active_planes is None else int(self.active_planes)
            if p_act < 1 or p_act > 8:
                raise ValueError(f"active_planes ({p_act}) must be in [1, 8]")
            if x.device.type == "cuda" and x.dtype == torch.bfloat16:
                self.fused_calls += 1
                y_out = torch.empty((flat.shape[0], self.out_features), dtype=torch.bfloat16, device=x.device)
                if flat.shape[0] > 0:
                    if p_act < 8:
                        from src.colibri_q1_cube_nvrtc import launch_q1_packed_trunc_gemv
                        launch_q1_packed_trunc_gemv(
                            flat.contiguous(), self.packed_lo, None, self.qscale, self.qzero, y_out,
                            group_size=self.group_size, bits=8, p_active=p_act, bias=self.bias, midpoint_correct=True,
                        )
                    else:
                        from src.colibri_q1_cube_nvrtc import launch_colibri_q8_gemv
                        launch_colibri_q8_gemv(
                            flat.contiguous(), self.packed_lo, self.qscale, self.qzero, y_out,
                            group_size=self.group_size, bias=self.bias,
                        )
                return y_out.view(*x.shape[:-1], self.out_features)
            self.tiled_calls += 1
            out = torch.empty((flat.shape[0], self.out_features), dtype=x.dtype, device=x.device)
            rows = max(1, TILE_ELEMENTS // self.in_features)
            packed = self.packed()
            shift = 8 - p_act
            mask = 0xFF ^ ((1 << shift) - 1) if p_act < 8 else 0xFF
            tail_mid = 0.5 * ((1 << shift) - 1) if p_act < 8 else 0.0
            for start in range(0, self.out_features, rows):
                end = min(self.out_features, start + rows)
                if p_act < 8:
                    q_trunc = (packed["packed_lo"][start:end] & mask).float().reshape(end - start, -1, self.group_size)
                    sc_s = packed["qscale"][start:end].float().unsqueeze(-1)
                    zr_s = packed["qzero"][start:end].float().unsqueeze(-1) + sc_s * tail_mid
                    w = (q_trunc * sc_s + zr_s).reshape(end - start, self.in_features).to(x.dtype)
                else:
                    w = unpack_weight(_slice_packed_rows(packed, start, end), self.group_size, x.dtype)
                out[:, start:end] = F.linear(flat, w, None if self.bias is None else self.bias[start:end])
            return out.view(*x.shape[:-1], self.out_features)

        if self.active_planes is not None:
            p_act = int(self.active_planes)
            if p_act < 1 or p_act > self.bits:
                raise ValueError(f"active_planes ({p_act}) must be in [1, {self.bits}]")
            if p_act < self.bits:
                flat = x.reshape(-1, self.in_features)
                if x.device.type == "cuda" and x.dtype == torch.bfloat16:
                    from src.colibri_q1_cube_nvrtc import launch_q1_packed_trunc_gemv
                    self.fused_calls += 1
                    y_out = torch.empty((flat.shape[0], self.out_features), dtype=torch.bfloat16, device=x.device)
                    if flat.shape[0] > 0:
                        launch_q1_packed_trunc_gemv(
                            flat.contiguous(),
                            self.packed_lo,
                            getattr(self, "packed_hi", None) if self.bits == 6 else None,
                            self.qscale,
                            self.qzero,
                            y_out,
                            group_size=self.group_size,
                            bits=self.bits,
                            p_active=p_act,
                            bias=self.bias,
                            midpoint_correct=True,
                        )
                    return y_out.view(*x.shape[:-1], self.out_features)
                self.tiled_calls += 1
                out = torch.empty((flat.shape[0], self.out_features), dtype=x.dtype, device=x.device)
                rows = max(1, TILE_ELEMENTS // self.in_features)
                packed = self.packed()
                shift = self.bits - p_act
                mask = ((1 << self.bits) - 1) ^ ((1 << shift) - 1)
                tail_mid = 0.5 * ((1 << shift) - 1)
                for start in range(0, self.out_features, rows):
                    end = min(self.out_features, start + rows)
                    lo_s = packed["packed_lo"][start:end]
                    q_s = torch.stack((lo_s & 15, lo_s >> 4), dim=-1).reshape(end - start, self.in_features)
                    if self.bits == 6 and "packed_hi" in packed:
                        hi_s = packed["packed_hi"][start:end]
                        upper = torch.stack(tuple((hi_s >> sh) & 3 for sh in (0, 2, 4, 6)), dim=-1).reshape_as(q_s)
                        q_s = q_s | (upper << 4)
                    q_trunc = (q_s & mask).float().reshape(end - start, -1, self.group_size)
                    sc_s = packed["qscale"][start:end].float().unsqueeze(-1)
                    zr_s = packed["qzero"][start:end].float().unsqueeze(-1) + sc_s * tail_mid
                    w = (q_trunc * sc_s + zr_s).reshape(end - start, self.in_features).to(x.dtype)
                    out[:, start:end] = F.linear(flat, w, None if self.bias is None else self.bias[start:end])
                return out.view(*x.shape[:-1], self.out_features)
        from src.colibri_nvrtc import can_use_fused_dense_q4, fused_dense_linear_q4, fused_dense_linear_q6
        if x.dtype == torch.bfloat16 and can_use_fused_dense_q4(x, self.packed_lo, self.qscale, self.qzero, self.group_size):
            self.fused_calls += 1
            if self.bits == 6:
                return fused_dense_linear_q6(x, self.packed_lo, self.packed_hi, self.qscale, self.qzero, self.bias, self.group_size)
            return fused_dense_linear_q4(x, self.packed_lo, self.qscale, self.qzero, self.bias, self.group_size)
        self.tiled_calls += 1
        flat = x.reshape(-1, self.in_features)
        out = torch.empty((flat.shape[0], self.out_features), dtype=x.dtype, device=x.device)
        rows = max(1, TILE_ELEMENTS // self.in_features)
        packed = self.packed()
        for start in range(0, self.out_features, rows):
            end = min(self.out_features, start + rows)
            w = unpack_weight({key: value[start:end] for key, value in packed.items()}, self.group_size, x.dtype)
            out[:, start:end] = F.linear(flat, w, None if self.bias is None else self.bias[start:end])
        return out.view(*x.shape[:-1], self.out_features)


def projection_map(model, layer_shapes, bits, group_size=64, replace=False, *, q1_planes=1):
    """Select actual module calls, never infer execution semantics from shapes alone."""
    from aircolibri_cli.model_source import resolve_decoder_layers
    from transformers.pytorch_utils import Conv1D
    result = {}
    for idx, layer in enumerate(resolve_decoder_layers(model)):
        specs = {}
        shapes = {key.removeprefix(f"model.layers.{idx}."): value for key, value in layer_shapes[idx].items()}
        modules = dict(layer.named_modules())
        for name, mod in list(modules.items()):
            if type(mod) not in (nn.Linear, Conv1D):
                continue
            key = name + ".weight"
            shape = shapes.get(key)
            if shape is None or len(shape) != 2 or tuple(mod.weight.shape) != tuple(shape):
                continue
            # Keep routing logits exact and bypass fused recurrent-weight consumers.
            if any(part in ("gate", "router", "router_proj") or part.endswith("_gate") for part in name.split(".")):
                continue
            parent_name, _, leaf = name.rpartition(".")
            parent = modules[parent_name]
            try:
                direct_weight_use = f".{leaf}.weight" in inspect.getsource(type(parent))
            except (OSError, TypeError):
                direct_weight_use = True
            if direct_weight_use:
                continue
            transpose = type(mod) is Conv1D
            canonical = tuple(reversed(shape)) if transpose else tuple(shape)
            n, k = canonical
            if k % group_size or math.prod(canonical) < 4096:
                continue
            specs[key] = {"shape": canonical, "transpose": transpose}
            if replace:
                setattr(parent, leaf, PackedLinear(k, n, bits, mod.bias is not None, group_size, q1_planes=q1_planes))
        result[idx] = specs
    return result


def projection_stats(model):
    mods = [m for m in model.modules() if isinstance(m, PackedLinear)]
    return {"modules": len(mods), "bits": sorted({m.bits for m in mods}),
            "fused_calls": sum(m.fused_calls for m in mods),
            "tiled_calls": sum(m.tiled_calls for m in mods)}
