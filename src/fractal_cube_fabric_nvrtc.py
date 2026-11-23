"""Unified 3D Fractal Cube Fabric NVRTC Engine (Read-Only over Pretrained LLM Weights).

Unifies ALL elements of the author's complete architecture into a single GPU-native fabric:
  1. Binary-Halving Q1 3D Cube (`positional_halving`, shared `qscale[0]`, `Sinkhorn + FWHT`):
     each projection `[N, K]` is partitioned into `(N/8) x (K/64)` elementary 3D cube tiles
     (`8 rows x 64 cols x P_max bit-planes`). Pretrained weights are STRICTLY READ-ONLY.
  2. Real Physical 3D Contact Graph (ZERO random edges):
     every elementary 3D cube `u = (proj, r_tile, c_tile)` is wired to its real physical
     neighbors across all four contact types:
       - HORIZ    : `(r, c +/- 1)` + paired projection sharing the same input (`gate <-> up`, `q <-> k <-> v`)
       - VERT     : `(r +/- 1, c)`
       - DIAG     : `(r +/- 1, c +/- 1)`
       - CORNER90 : 90-degree angle turn where output row-tile `r` becomes input col-tile `c'`
                    of the downstream projection (`gate/up -> down`, `q/v -> o`, layer `l -> l+1`).
  3. Mirror Equality x UNI Binding (`Phi=e^-1.5, Omega=pi/2, Sigma=sqrt(pi), lambda=exp(-H)`):
     each active 3D cube executes inside a single 32-thread warp (4 threads/row x 8 rows, ZERO
     shared memory, ZERO `__syncthreads()`, 5-cycle register shuffle), adds its update directly
     onto the shared assembly line (`line_spine`), and autonomously decides whether to:
       - deepen itself (`DEPTH: p -> p+1`),
       - wake contacting neighbors (`HORIZ / VERT / DIAG`) via GPU `atomicExch` + `atomicAdd`,
       - turn 90 degrees (`CORNER90`) into live downstream columns while skipping deadzone (`P ~ 0`),
       - or enter network rest (`defect < eff_tol`) with zero further reads from VRAM/disk!
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import os
import threading
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.binary_halving_q1 import BinaryHalvingQ1Linear, halve_weight_to_q1_tree
from src.colibri_nvrtc import PTX_CACHE_DIR, _load_driver_and_nvrtc
from src.ideal_q1_engine import fwht_block

CONTACT_DEPTH = 0
CONTACT_HORIZ = 1
CONTACT_VERT = 2
CONTACT_DIAG = 3
CONTACT_CORNER90 = 4

_FABRIC_CUDA_SOURCE = r"""
__device__ __forceinline__ float bf16_to_f32(unsigned short b) {
    return __uint_as_float(((unsigned int)b) << 16);
}

/* Kernel: Warp-Level 3D Puzzle Cube Mirror x UNI Fabric Step
 *
 * 1 warp (32 lanes) = 1 active elementary 3D cube `u = (proj_id, r_tile, c_tile)` of size (8 x 64).
 *   r_local = lane >> 2  (0..7)
 *   chunk4  = lane & 3   (0..3 -> 16 columns each)
 *
 * ZERO shared memory, ZERO __syncthreads(), pure warp-register Mirror x UNI computation.
 * Multiple projections (e.g., gate_proj + up_proj simultaneously) share a single unified node space
 * and contact graph so HORIZ/VERT/DIAG/PAIRED-PROJ contacts cross between matrices in GPU silicon!
 *
 * Grid:  ((num_active_warps + 7) / 8, 1, 1)
 * Block: (256, 1, 1) -> 8 warps = 8 active 3D cubes per block
 */
extern "C" __global__ void cube_puzzle_mirror_fabric_step_f32(
    const int* __restrict__ frontier_in,
    int num_active,
    /* Per-node descriptor (length = total_nodes across participating projections) */
    const int* __restrict__ node_proj,
    const int* __restrict__ node_rtile,
    const int* __restrict__ node_ctile,
    /* Per-projection metadata arrays (length = num_projs) */
    const unsigned long long* __restrict__ proj_packed_ptrs,
    const unsigned long long* __restrict__ proj_qscale_ptrs,
    const unsigned long long* __restrict__ proj_qzero_ptrs,
    const unsigned long long* __restrict__ proj_rscale_ptrs,
    const unsigned long long* __restrict__ proj_bias_ptrs,
    const unsigned long long* __restrict__ proj_xrot_ptrs,
    const unsigned long long* __restrict__ proj_xsum_ptrs,
    const unsigned long long* __restrict__ proj_line_ptrs,
    const int* __restrict__ proj_N,
    const int* __restrict__ proj_K,
    const int* __restrict__ proj_shared_scale,
    /* Contact graph: real physical faces & angles (HORIZ / VERT / DIAG / PAIRED) */
    const int* __restrict__ contact_idx,    /* [total_nodes, MAX_C] */
    const unsigned char* __restrict__ contact_type, /* [total_nodes, MAX_C] */
    int MAX_C,
    /* Mutable fabric state (read-only weights; only assembly state updates) */
    int* __restrict__ cube_plane_done,      /* [total_nodes]: planes already read (0..P_max) */
    float* __restrict__ cube_prev_obs_norm, /* [total_nodes]: two-turn smooth mirror envelope */
    int* __restrict__ enqueued_gen,         /* [total_nodes]: dedup tag per wave */
    int* __restrict__ frontier_out,         /* [total_nodes] */
    int* __restrict__ next_count_out,       /* [1] */
    unsigned long long* __restrict__ stats_out, /* [8]: 0=executed, 1=depth, 2=horiz, 3=vert, 4=diag, 5=rested */
    int total_nodes,
    int P_max,
    int P_min,
    int gen_id,
    float base_tol,
    float contact_tol
) {
    const float UNI_PHI   = 0.22313016f; /* exp(-1.5)  */
    const float UNI_OMEGA = 1.57079633f; /* pi / 2     */
    const float UNI_SIGMA = 1.77245385f; /* sqrt(pi)   */

    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_global >= num_active) return;

    int u = frontier_in[warp_global];
    if (u < 0 || u >= total_nodes) return;

    /* Claim the next Q1 plane `p` for this exact 3D cube atomically (never double-counted!) */
    int p = 0;
    if (lane == 0) {
        p = atomicAdd(&cube_plane_done[u], 1);
    }
    p = __shfl_sync(0xffffffffu, p, 0);
    if (p >= P_max) return;

    int pid = node_proj[u];
    int rt  = node_rtile[u];
    int ct  = node_ctile[u];
    int N   = proj_N[pid];
    int K   = proj_K[pid];
    int ng  = K >> 6; /* K / 64 */
    int k8  = K >> 3; /* K / 8  */
    int shared_sc = proj_shared_scale[pid];

    const unsigned char*  packed_lo = reinterpret_cast<const unsigned char*>(proj_packed_ptrs[pid]);
    const unsigned short* qscale    = reinterpret_cast<const unsigned short*>(proj_qscale_ptrs[pid]);
    const unsigned short* qzero     = reinterpret_cast<const unsigned short*>(proj_qzero_ptrs[pid]);
    const unsigned short* row_scale = reinterpret_cast<const unsigned short*>(proj_rscale_ptrs[pid]);
    const unsigned short* bias_bf16 = reinterpret_cast<const unsigned short*>(proj_bias_ptrs[pid]);
    const float* x_rot  = reinterpret_cast<const float*>(proj_xrot_ptrs[pid]);
    const float* x_sum  = reinterpret_cast<const float*>(proj_xsum_ptrs[pid]);
    float* line_spine   = reinterpret_cast<float*>(proj_line_ptrs[pid]);

    int r_local = lane >> 2;  /* 0..7 */
    int chunk4  = lane & 3;   /* 0..3 -> 16 columns */
    int row     = rt * 8 + r_local;
    bool valid_row = (row < N);

    /* 1. Evaluate 16 bits of this (8 x 64) Q1 cube per lane (read-only pretrained weights) */
    float d1 = 0.0f;
    if (valid_row) {
        long long plane_byte_stride = (long long)N * k8;
        const unsigned short* b_u16 = reinterpret_cast<const unsigned short*>(
            packed_lo + (long long)p * plane_byte_stride + (long long)row * k8 + (ct * 8)
        );
        unsigned int bits16 = (unsigned int)b_u16[chunk4];
        const float* x_seg = x_rot + (ct * 64 + chunk4 * 16);
        while (bits16) {
            int b = __ffs(bits16) - 1;
            d1 += x_seg[b];
            bits16 &= (bits16 - 1u);
        }
    }

    /* Reduce 4 chunks -> full 64-col dot per row in 2 register hops */
    d1 += __shfl_xor_sync(0xffffffffu, d1, 1);
    d1 += __shfl_xor_sync(0xffffffffu, d1, 2);

    float prev_line = 0.0f;
    float next_line = 0.0f;
    float obs_step  = 0.0f;

    /* 2. Intersect with shared assembly line (`line_spine`): start and next step meet on one line */
    if (chunk4 == 0 && valid_row) {
        long long grp_idx = (long long)row * ng + ct;
        long long plane_grp_stride = (long long)N * ng;
        float sv = shared_sc
            ? (bf16_to_f32(qscale[grp_idx]) * ldexpf(1.0f, -p))
            : bf16_to_f32(qscale[(long long)p * plane_grp_stride + grp_idx]);
        float xsum_g = x_sum[ct];
        float raw_delta = sv * (2.0f * d1 - xsum_g);
        if (p == 0) {
            raw_delta += bf16_to_f32(qzero[grp_idx]) * xsum_g;
        }
        obs_step = raw_delta * bf16_to_f32(row_scale[row]);
        if (p == 0 && ct == 0 && bias_bf16 != nullptr) {
            obs_step += bf16_to_f32(bias_bf16[row]);
        }
        prev_line = atomicAdd(&line_spine[row], obs_step);
        next_line = prev_line + obs_step;
    }

    /* 3. Compute Mirror Equality x UNI binding invariants across the 8 rows in 3 register hops */
    float e_prev = (chunk4 == 0 && valid_row) ? (prev_line * prev_line) : 0.0f;
    float e_next = (chunk4 == 0 && valid_row) ? (next_line * next_line) : 0.0f;
    float e_obs  = (chunk4 == 0 && valid_row) ? (obs_step * obs_step)   : 0.0f;
    float v_dot  = (chunk4 == 0 && valid_row) ? (prev_line * obs_step)  : 0.0f;

    float tot_prev_e = e_prev, tot_next_e = e_next, tot_obs_e = e_obs, tot_dot = v_dot;
    #pragma unroll
    for (int d = 4; d <= 16; d <<= 1) {
        tot_prev_e += __shfl_xor_sync(0xffffffffu, tot_prev_e, d);
        tot_next_e += __shfl_xor_sync(0xffffffffu, tot_next_e, d);
        tot_obs_e  += __shfl_xor_sync(0xffffffffu, tot_obs_e, d);
        tot_dot    += __shfl_xor_sync(0xffffffffu, tot_dot, d);
    }

    float pr = (chunk4 == 0 && valid_row) ? (e_next / (tot_next_e + 1e-12f)) : 0.0f;
    float h_part = (pr > 1e-7f) ? (-pr * logf(pr)) : 0.0f;
    #pragma unroll
    for (int d = 4; d <= 16; d <<= 1) {
        h_part += __shfl_xor_sync(0xffffffffu, h_part, d);
    }

    /* 4. Lane 0 evaluates Mirror x UNI rest gate and autonomously spawns DEPTH + HORIZ/VERT/DIAG */
    if (lane == 0) {
        if (stats_out != nullptr) atomicAdd(&stats_out[0], 1ULL);

        float cur_obs_norm = sqrtf(tot_obs_e);
        float prev_obs_norm = cube_prev_obs_norm[u];
        cube_prev_obs_norm[u] = cur_obs_norm;
        float smooth_obs = (p == 0) ? cur_obs_norm : fmaxf(cur_obs_norm, 0.5f * prev_obs_norm);

        /* Scale-consistent denominator across c_tiles: RMS per-cube contribution on the shared line */
        float line_scale = sqrtf(tot_next_e / (float)ng) + 1e-6f;
        float defect = (1.15470054f * smooth_obs) / (line_scale + cur_obs_norm * 0.5f);

        float cos_abs = fabsf(tot_dot) / (sqrtf(tot_prev_e * tot_obs_e) + 1e-8f);
        if (cos_abs > 1.0f) cos_abs = 1.0f;
        float ortho_ratio = acosf(cos_abs) / UNI_OMEGA;
        float lambda_tile = expf(-h_part);
        float bind_factor = (1.0f - UNI_PHI * ortho_ratio) * (1.0f - lambda_tile / UNI_SIGMA);
        if (bind_factor < 0.25f) bind_factor = 0.25f;

        float eff_depth_tol   = base_tol * bind_factor;
        float eff_contact_tol = contact_tol * bind_factor;

        bool wake_depth = ((p + 1) < P_max) && (((p + 1) < P_min) || (base_tol <= 0.0f) || (defect >= eff_depth_tol));
        bool wake_faces = (base_tol <= 0.0f) || (defect >= eff_contact_tol);

        if (wake_depth) {
            if (atomicExch(&enqueued_gen[u], gen_id) != gen_id) {
                int pos = atomicAdd(next_count_out, 1);
                if (pos < total_nodes) frontier_out[pos] = u;
                if (stats_out != nullptr) atomicAdd(&stats_out[1], 1ULL);
            }
        } else {
            if (stats_out != nullptr) atomicAdd(&stats_out[5], 1ULL);
        }

        /* Propagate along real 3D cube faces & angles (HORIZ / VERT / DIAG / PAIRED) */
        if (wake_faces && MAX_C > 0 && contact_idx != nullptr) {
            long long c_base = (long long)u * MAX_C;
            for (int k = 0; k < MAX_C; ++k) {
                int v = contact_idx[c_base + k];
                if (v < 0 || v >= total_nodes) continue;
                if (cube_plane_done[v] >= P_max) continue;
                /* Only wake untouched or still-shallow neighbors (<= current depth) */
                if (cube_plane_done[v] > p + 1) continue;
                if (atomicExch(&enqueued_gen[v], gen_id) != gen_id) {
                    int pos = atomicAdd(next_count_out, 1);
                    if (pos < total_nodes) frontier_out[pos] = v;
                    if (stats_out != nullptr && contact_type != nullptr) {
                        unsigned char ct_k = contact_type[c_base + k];
                        if (ct_k >= 1 && ct_k <= 3) {
                            atomicAdd(&stats_out[ct_k + 1], 1ULL);
                        }
                    }
                }
            }
        }
    }
}
"""

_FABRIC_KERNEL_NAMES = (b"cube_puzzle_mirror_fabric_step_f32",)
_LOCK = threading.Lock()
_DEVICE_MODULES: Dict[int, Dict[bytes, ctypes.c_void_p]] = {}


def _compile_or_load_fabric_ptx(major: int, minor: int, nvrtc) -> bytes:
    src_bytes = _FABRIC_CUDA_SOURCE.encode("utf-8")
    digest = hashlib.sha256(src_bytes).hexdigest()[:16]
    PTX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = PTX_CACHE_DIR / f"fractal_cube_fabric_sm{major}{minor}_{digest}.ptx"
    if cache_file.is_file():
        data = cache_file.read_bytes()
        return data if data.endswith(b"\x00") else data + b"\x00"

    prog = ctypes.c_void_p()
    rc = nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src_bytes, b"fractal_cube_fabric.cu", 0, None, None)
    if rc != 0:
        raise RuntimeError(f"nvrtcCreateProgram(fractal_cube_fabric) failed ({rc})")
    try:
        arch_flag = f"--gpu-architecture=compute_{major}{minor}".encode("utf-8")
        opts = (ctypes.c_char_p * 2)(arch_flag, b"--use_fast_math")
        rc = nvrtc.nvrtcCompileProgram(prog, 2, opts)
        if rc != 0:
            log_sz = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_sz))
            log_buf = ctypes.create_string_buffer(log_sz.value)
            nvrtc.nvrtcGetProgramLog(prog, log_buf)
            raise RuntimeError(
                f"NVRTC fractal_cube_fabric compile failed ({rc}):\n{log_buf.value.decode('utf-8', errors='replace')}"
            )
        ptx_sz = ctypes.c_size_t()
        nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_sz))
        ptx_buf = ctypes.create_string_buffer(ptx_sz.value)
        rc = nvrtc.nvrtcGetPTX(prog, ptx_buf)
        if rc != 0:
            raise RuntimeError(f"nvrtcGetPTX(fractal_cube_fabric) failed ({rc})")
        raw_ptx = ptx_buf.raw
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    tmp = cache_file.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(raw_ptx)
    os.replace(tmp, cache_file)
    return raw_ptx


def get_fabric_kernels(device: torch.device):
    dev_idx = 0 if device.index is None else int(device.index)
    nvcuda, nvrtc = _load_driver_and_nvrtc()
    cached = _DEVICE_MODULES.get(dev_idx)
    if cached is not None:
        return nvcuda, cached
    with _LOCK:
        cached = _DEVICE_MODULES.get(dev_idx)
        if cached is not None:
            return nvcuda, cached
        with torch.cuda.device(dev_idx):
            _ = torch.empty(1, device=f"cuda:{dev_idx}")
            major, minor = torch.cuda.get_device_capability(dev_idx)
            ptx = _compile_or_load_fabric_ptx(major, minor, nvrtc)
            ptx_buf = ctypes.create_string_buffer(ptx)
            mod = ctypes.c_void_p()
            rc = nvcuda.cuModuleLoadData(ctypes.byref(mod), ctypes.cast(ptx_buf, ctypes.c_void_p))
            if rc != 0:
                raise RuntimeError(f"cuModuleLoadData(fractal_cube_fabric) failed ({rc})")
            fn = ctypes.c_void_p()
            rc = nvcuda.cuModuleGetFunction(ctypes.byref(fn), mod, b"cube_puzzle_mirror_fabric_step_f32")
            if rc != 0:
                raise RuntimeError(f"cuModuleGetFunction failed ({rc})")
            funcs = {"_module": mod, b"cube_puzzle_mirror_fabric_step_f32": fn}
            _DEVICE_MODULES[dev_idx] = funcs
            return nvcuda, funcs


class Projection3DCubeStage:
    """One or more paired projections sharing an input vector, represented as a unified 3D Q1-cube graph.

    For example, `gate_proj` and `up_proj` share the exact same input `x`, so every `(r_tile, c_tile)`
    in `gate_proj` has real HORIZ contacts to `(r, c +/- 1)` AND paired HORIZ contact to `(r, c)` in
    `up_proj`, VERT contacts to `(r +/- 1, c)`, and DIAG contacts to `(r +/- 1, c +/- 1)` — ZERO randint!
    """

    MAX_CONTACTS = 9  # 2 horiz + 2 vert + 4 diag + 1 paired-projection

    def __init__(self, projs: List[BinaryHalvingQ1Linear], *, paired: bool = False):
        if not projs:
            raise ValueError("Need at least one BinaryHalvingQ1Linear")
        self.projs = projs
        self.device = projs[0].packed_lo.device
        self.num_projs = len(projs)
        self.paired = bool(paired and len(projs) == 2)
        self.root_bits = int(projs[0].root_bits)

        # Static 1-time layout of all 3D cubes `(pid, r_tile, c_tile)` and their real contact graph
        proj_ids: List[int] = []
        r_tiles: List[int] = []
        c_tiles: List[int] = []
        self.proj_offsets: List[int] = []
        self.proj_grid: List[Tuple[int, int]] = []  # (n_rt, n_ct)

        offset = 0
        for pid, lin in enumerate(projs):
            n_rt = (lin.out_features + 7) // 8
            n_ct = lin.in_features // 64
            self.proj_offsets.append(offset)
            self.proj_grid.append((n_rt, n_ct))
            for rt in range(n_rt):
                for ct in range(n_ct):
                    proj_ids.append(pid)
                    r_tiles.append(rt)
                    c_tiles.append(ct)
            offset += n_rt * n_ct

        self.total_nodes = offset
        c_idx = torch.full((self.total_nodes, self.MAX_CONTACTS), -1, dtype=torch.int32)
        c_typ = torch.zeros((self.total_nodes, self.MAX_CONTACTS), dtype=torch.uint8)

        # Build exact physical contact topology on CPU once at init, then upload to GPU
        for pid, (n_rt, n_ct) in enumerate(self.proj_grid):
            base = self.proj_offsets[pid]
            other_pid = (1 - pid) if self.paired else -1
            other_base = self.proj_offsets[other_pid] if other_pid >= 0 else -1
            other_same = (other_pid >= 0 and self.proj_grid[other_pid] == (n_rt, n_ct))

            rt_arr = torch.arange(n_rt, dtype=torch.int32).unsqueeze(1).expand(n_rt, n_ct)
            ct_arr = torch.arange(n_ct, dtype=torch.int32).unsqueeze(0).expand(n_rt, n_ct)
            u_grid = base + rt_arr * n_ct + ct_arr

            slot = 0
            # HORIZ: (r, c - 1), (r, c + 1)
            for dc in (-1, 1):
                nc = ct_arr + dc
                valid = (nc >= 0) & (nc < n_ct)
                v = torch.where(valid, base + rt_arr * n_ct + nc.clamp(0, n_ct - 1), torch.tensor(-1, dtype=torch.int32))
                c_idx[u_grid.reshape(-1), slot] = v.reshape(-1)
                c_typ[u_grid.reshape(-1), slot] = torch.where(valid, CONTACT_HORIZ, 0).to(torch.uint8).reshape(-1)
                slot += 1
            # Paired HORIZ contact between gate_proj <-> up_proj at exact same (r_tile, c_tile)
            if other_same:
                v_pair = other_base + rt_arr * n_ct + ct_arr
                c_idx[u_grid.reshape(-1), slot] = v_pair.reshape(-1)
                c_typ[u_grid.reshape(-1), slot] = CONTACT_HORIZ
            slot += 1
            # VERT: (r - 1, c), (r + 1, c)
            for dr in (-1, 1):
                nr = rt_arr + dr
                valid = (nr >= 0) & (nr < n_rt)
                v = torch.where(valid, base + nr.clamp(0, n_rt - 1) * n_ct + ct_arr, torch.tensor(-1, dtype=torch.int32))
                c_idx[u_grid.reshape(-1), slot] = v.reshape(-1)
                c_typ[u_grid.reshape(-1), slot] = torch.where(valid, CONTACT_VERT, 0).to(torch.uint8).reshape(-1)
                slot += 1
            # DIAG: (r +/- 1, c +/- 1)
            for dr in (-1, 1):
                for dc in (-1, 1):
                    nr = rt_arr + dr
                    nc = ct_arr + dc
                    valid = (nr >= 0) & (nr < n_rt) & (nc >= 0) & (nc < n_ct)
                    v = torch.where(
                        valid,
                        base + nr.clamp(0, n_rt - 1) * n_ct + nc.clamp(0, n_ct - 1),
                        torch.tensor(-1, dtype=torch.int32),
                    )
                    c_idx[u_grid.reshape(-1), slot] = v.reshape(-1)
                    c_typ[u_grid.reshape(-1), slot] = torch.where(valid, CONTACT_DIAG, 0).to(torch.uint8).reshape(-1)
                    slot += 1

        dev = self.device
        self.node_proj = torch.tensor(proj_ids, dtype=torch.int32, device=dev)
        self.node_rtile = torch.tensor(r_tiles, dtype=torch.int32, device=dev)
        self.node_ctile = torch.tensor(c_tiles, dtype=torch.int32, device=dev)
        self.contact_idx = c_idx.to(dev)
        self.contact_type = c_typ.to(dev)

        self.proj_packed_ptrs = torch.tensor([int(l.packed_lo.data_ptr()) for l in projs], dtype=torch.int64, device=dev)
        self.proj_qscale_ptrs = torch.tensor([int(l.qscale.data_ptr()) for l in projs], dtype=torch.int64, device=dev)
        self.proj_qzero_ptrs = torch.tensor([int(l.qzero.data_ptr()) for l in projs], dtype=torch.int64, device=dev)
        self.proj_rscale_ptrs = torch.tensor([int(l.row_scale.data_ptr()) for l in projs], dtype=torch.int64, device=dev)
        self.proj_bias_ptrs = torch.tensor(
            [0 if l.bias is None else int(l.bias.data_ptr()) for l in projs], dtype=torch.int64, device=dev
        )
        self.proj_N = torch.tensor([int(l.out_features) for l in projs], dtype=torch.int32, device=dev)
        self.proj_K = torch.tensor([int(l.in_features) for l in projs], dtype=torch.int32, device=dev)
        self.proj_shared_scale = torch.tensor(
            [1 if int(l.qscale.shape[0]) == 1 else 0 for l in projs], dtype=torch.int32, device=dev
        )

    @torch.no_grad()
    def run(
        self,
        x_1d: torch.Tensor,
        *,
        seed_nodes: Optional[torch.Tensor] = None,
        max_planes: int = 8,
        min_planes: int = 3,
        base_tol: float = 0.14,
        contact_tol: float = 0.12,
        max_waves: int = 32,
    ) -> Tuple[List[torch.Tensor], Dict[str, object]]:
        """Execute autonomous GPU self-spawning across all 3D cubes in this stage until rest."""
        dev = self.device
        nvcuda, fns = get_fabric_kernels(dev)
        x_f = x_1d.reshape(-1).to(device=dev, dtype=torch.float32)

        # Precompute FWHT-rotated x once per projection
        x_rots: List[torch.Tensor] = []
        x_sums: List[torch.Tensor] = []
        lines: List[torch.Tensor] = []
        for lin in self.projs:
            xr = fwht_block((x_f * lin.col_scale.float()).unsqueeze(0), 64).squeeze(0).contiguous()
            xs = xr.reshape(-1, 64).sum(dim=-1).contiguous()
            x_rots.append(xr)
            x_sums.append(xs)
            lines.append(torch.zeros((lin.out_features,), dtype=torch.float32, device=dev))

        xrot_ptrs = torch.tensor([int(t.data_ptr()) for t in x_rots], dtype=torch.int64, device=dev)
        xsum_ptrs = torch.tensor([int(t.data_ptr()) for t in x_sums], dtype=torch.int64, device=dev)
        line_ptrs = torch.tensor([int(t.data_ptr()) for t in lines], dtype=torch.int64, device=dev)

        cube_plane_done = torch.zeros((self.total_nodes,), dtype=torch.int32, device=dev)
        cube_prev_obs = torch.zeros((self.total_nodes,), dtype=torch.float32, device=dev)
        enqueued_gen = torch.zeros((self.total_nodes,), dtype=torch.int32, device=dev)
        frontier_a = torch.empty((self.total_nodes,), dtype=torch.int32, device=dev)
        frontier_b = torch.empty((self.total_nodes,), dtype=torch.int32, device=dev)
        next_cnt = torch.zeros((1,), dtype=torch.int32, device=dev)
        stats_buf = torch.zeros((8,), dtype=torch.int64, device=dev)

        if seed_nodes is None:
            num_active = self.total_nodes
            frontier_a.copy_(torch.arange(self.total_nodes, dtype=torch.int32, device=dev))
        else:
            s_c = seed_nodes.to(device=dev, dtype=torch.int32).contiguous()
            num_active = int(s_c.numel())
            frontier_a[:num_active].copy_(s_c)

        stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
        wave_sizes: List[int] = [num_active]

        for wave in range(1, max_waves + 1):
            if num_active <= 0:
                break
            next_cnt.zero_()

            p_fin = ctypes.c_void_p(frontier_a.data_ptr())
            c_na = ctypes.c_int(num_active)
            p_np = ctypes.c_void_p(self.node_proj.data_ptr())
            p_nr = ctypes.c_void_p(self.node_rtile.data_ptr())
            p_nc = ctypes.c_void_p(self.node_ctile.data_ptr())

            p_pk = ctypes.c_void_p(self.proj_packed_ptrs.data_ptr())
            p_qs = ctypes.c_void_p(self.proj_qscale_ptrs.data_ptr())
            p_qz = ctypes.c_void_p(self.proj_qzero_ptrs.data_ptr())
            p_rs = ctypes.c_void_p(self.proj_rscale_ptrs.data_ptr())
            p_bi = ctypes.c_void_p(self.proj_bias_ptrs.data_ptr())
            p_xr = ctypes.c_void_p(xrot_ptrs.data_ptr())
            p_xs = ctypes.c_void_p(xsum_ptrs.data_ptr())
            p_ln = ctypes.c_void_p(line_ptrs.data_ptr())
            p_pN = ctypes.c_void_p(self.proj_N.data_ptr())
            p_pK = ctypes.c_void_p(self.proj_K.data_ptr())
            p_ss = ctypes.c_void_p(self.proj_shared_scale.data_ptr())

            p_ci = ctypes.c_void_p(self.contact_idx.data_ptr())
            p_ct = ctypes.c_void_p(self.contact_type.data_ptr())
            c_mc = ctypes.c_int(self.MAX_CONTACTS)

            p_pd = ctypes.c_void_p(cube_plane_done.data_ptr())
            p_po = ctypes.c_void_p(cube_prev_obs.data_ptr())
            p_eg = ctypes.c_void_p(enqueued_gen.data_ptr())
            p_fo = ctypes.c_void_p(frontier_b.data_ptr())
            p_no = ctypes.c_void_p(next_cnt.data_ptr())
            p_st = ctypes.c_void_p(stats_buf.data_ptr())

            c_tn = ctypes.c_int(self.total_nodes)
            c_pmax, c_pmin = ctypes.c_int(int(max_planes)), ctypes.c_int(int(min_planes))
            c_gen = ctypes.c_int(wave)
            c_bt, c_ctol = ctypes.c_float(float(base_tol)), ctypes.c_float(float(contact_tol))

            args = (ctypes.c_void_p * 31)(
                ctypes.addressof(p_fin), ctypes.addressof(c_na),
                ctypes.addressof(p_np), ctypes.addressof(p_nr), ctypes.addressof(p_nc),
                ctypes.addressof(p_pk), ctypes.addressof(p_qs), ctypes.addressof(p_qz),
                ctypes.addressof(p_rs), ctypes.addressof(p_bi),
                ctypes.addressof(p_xr), ctypes.addressof(p_xs), ctypes.addressof(p_ln),
                ctypes.addressof(p_pN), ctypes.addressof(p_pK), ctypes.addressof(p_ss),
                ctypes.addressof(p_ci), ctypes.addressof(p_ct), ctypes.addressof(c_mc),
                ctypes.addressof(p_pd), ctypes.addressof(p_po), ctypes.addressof(p_eg),
                ctypes.addressof(p_fo), ctypes.addressof(p_no), ctypes.addressof(p_st),
                ctypes.addressof(c_tn), ctypes.addressof(c_pmax), ctypes.addressof(c_pmin),
                ctypes.addressof(c_gen), ctypes.addressof(c_bt), ctypes.addressof(c_ctol),
            )
            grid_x = (num_active + 7) // 8
            rc = nvcuda.cuLaunchKernel(
                fns[b"cube_puzzle_mirror_fabric_step_f32"],
                grid_x, 1, 1, 256, 1, 1, 0, stream, args, None,
            )
            if rc != 0:
                raise RuntimeError(f"cuLaunchKernel(cube_puzzle_mirror_fabric_step_f32) failed ({rc})")

            num_active = min(self.total_nodes, int(next_cnt.item()))
            wave_sizes.append(num_active)
            frontier_a, frontier_b = frontier_b, frontier_a

        st_vals = stats_buf.tolist()
        touched_cubes = int((cube_plane_done > 0).sum().item())
        total_possible_ops = self.total_nodes * int(max_planes)
        return lines, {
            "total_3d_cubes": self.total_nodes,
            "touched_3d_cubes": touched_cubes,
            "touched_cubes_pct": round(100.0 * touched_cubes / max(1, self.total_nodes), 2),
            "executed_cube_plane_ops": int(st_vals[0]),
            "ops_vs_full_q8_pct": round(100.0 * st_vals[0] / max(1, total_possible_ops), 2),
            "avg_planes_per_cube": round(st_vals[0] / max(1, self.total_nodes), 3),
            "spawns_by_contact": {
                "depth": int(st_vals[1]),
                "horiz_and_paired": int(st_vals[2]),
                "vert": int(st_vals[3]),
                "diag": int(st_vals[4]),
                "rested_cubes": int(st_vals[5]),
            },
            "waves": len(wave_sizes) - 1,
            "wave_sizes": wave_sizes,
            "converged_naturally": (wave_sizes[-1] == 0),
        }


class Fractal3DCubeSwiGLUMLP:
    """Full SwiGLU MLP block (`gate_proj + up_proj --[CORNER90 + deadzone]--> down_proj`)
    running entirely as a unified 3D Q1-Cube Puzzle x Mirror x UNI Fabric (STRICTLY READ-ONLY).

    Stage 1 (`gate_proj <-> up_proj`): paired HORIZ + VERT + DIAG + DEPTH self-spawning to rest.
    90-degree Corner Bridge (`CORNER90`): computes `h = silu(line_gate) * line_up`; only live
    64-channel groups (`||h[c_tile]||_2 > deadzone_rel * rms`) turn 90 degrees into `down_proj`
    columns `c_tile`, automatically skipping deadzone (`P ~ 0`) columns in `down_proj`!
    Stage 2 (`down_proj`): HORIZ + VERT + DIAG + DEPTH self-spawning to rest on `line_down`.
    """

    def __init__(
        self,
        gate_lin: BinaryHalvingQ1Linear,
        up_lin: BinaryHalvingQ1Linear,
        down_lin: BinaryHalvingQ1Linear,
    ):
        self.gate_lin = gate_lin
        self.up_lin = up_lin
        self.down_lin = down_lin
        self.stage_gu = Projection3DCubeStage([gate_lin, up_lin], paired=True)
        self.stage_dn = Projection3DCubeStage([down_lin], paired=False)

    @classmethod
    def from_weights(
        cls,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        *,
        device: torch.device | str = "cuda:0",
        root_bits: int = 8,
    ) -> "Fractal3DCubeSwiGLUMLP":
        dev = torch.device(device)

        def _slice(w: torch.Tensor) -> BinaryHalvingQ1Linear:
            N, K = int(w.shape[0]), int(w.shape[1])
            td, md = halve_weight_to_q1_tree(
                w.to(device=dev, dtype=torch.bfloat16), root_bits=root_bits, group_size=64,
                law="positional_halving", use_fwht=True,
            )
            return BinaryHalvingQ1Linear(K, N, {k: v.to(dev) for k, v in td.items()}, md)

        return cls(_slice(w_gate), _slice(w_up), _slice(w_down))

    @torch.no_grad()
    def forward_fabric(
        self,
        x_1d: torch.Tensor,
        *,
        max_planes: int = 8,
        min_planes: int = 3,
        base_tol: float = 0.14,
        contact_tol: float = 0.14,
        seed_input_quantile: float = 0.0,
        corner90_deadzone_rel: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Run the full connected 3D cube MLP fabric from base seed to network rest.

        - `seed_input_quantile = 0.0` starts Stage 1 on all input column tiles (exact base coverage);
          `> 0.0` seeds ONLY the top energetic input columns and lets HORIZ/VERT/DIAG contacts
          organically grow the connected puzzle slice!
        - `corner90_deadzone_rel > 0.0` skips `down_proj` column tiles whose SwiGLU activation
          energy `||h[c_tile]||_2` falls into the deadzone (`< corner90_deadzone_rel * rms`).
        """
        dev = self.stage_gu.device
        x_f = x_1d.reshape(-1).to(device=dev, dtype=torch.float32)

        # Build Stage 1 base seed (either full base or energetic input seed columns)
        seed_gu: Optional[torch.Tensor] = None
        if seed_input_quantile > 0.0:
            x_grp_e = x_f.reshape(-1, 64).pow(2).sum(dim=-1).sqrt()
            thresh = float(torch.quantile(x_grp_e, seed_input_quantile).item())
            hot_cols = (x_grp_e >= thresh).nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
            if hot_cols.numel() == 0:
                hot_cols = torch.tensor([int(x_grp_e.argmax().item())], dtype=torch.int32, device=dev)
            seed_parts = []
            for pid, (n_rt, n_ct) in enumerate(self.stage_gu.proj_grid):
                base = self.stage_gu.proj_offsets[pid]
                rt_all = torch.arange(n_rt, dtype=torch.int32, device=dev).unsqueeze(1)
                nodes_p = (base + rt_all * n_ct + hot_cols.unsqueeze(0)).reshape(-1)
                seed_parts.append(nodes_p)
            seed_gu = torch.cat(seed_parts, dim=0)

        (line_gate, line_up), st_gu = self.stage_gu.run(
            x_f, seed_nodes=seed_gu,
            max_planes=max_planes, min_planes=min_planes,
            base_tol=base_tol, contact_tol=contact_tol,
        )

        # 90-degree Corner Bridge (CORNER90): output rows of gate/up -> input columns of down_proj!
        h_act = F.silu(line_gate) * line_up
        h_grp_e = h_act.reshape(-1, 64).pow(2).sum(dim=-1).sqrt()
        h_rms = float(h_grp_e.pow(2).mean().sqrt().item()) + 1e-8
        n_rt_dn, n_ct_dn = self.stage_dn.proj_grid[0]

        if corner90_deadzone_rel > 0.0:
            live_cols = (h_grp_e >= corner90_deadzone_rel * h_rms).nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
            if live_cols.numel() == 0:
                live_cols = torch.tensor([int(h_grp_e.argmax().item())], dtype=torch.int32, device=dev)
            rt_all = torch.arange(n_rt_dn, dtype=torch.int32, device=dev).unsqueeze(1)
            seed_dn = (rt_all * n_ct_dn + live_cols.unsqueeze(0)).reshape(-1)
            live_col_cnt = int(live_cols.numel())
        else:
            seed_dn = None
            live_col_cnt = n_ct_dn

        (line_down,), st_dn = self.stage_dn.run(
            h_act, seed_nodes=seed_dn,
            max_planes=max_planes, min_planes=min_planes,
            base_tol=base_tol, contact_tol=contact_tol,
        )

        tot_cubes = st_gu["total_3d_cubes"] + st_dn["total_3d_cubes"]
        tot_ops = st_gu["executed_cube_plane_ops"] + st_dn["executed_cube_plane_ops"]
        return line_down, {
            "total_3d_cubes": tot_cubes,
            "total_executed_ops": tot_ops,
            "total_ops_vs_full_q8_pct": round(100.0 * tot_ops / max(1, tot_cubes * int(max_planes)), 2),
            "corner90_live_cols": live_col_cnt,
            "corner90_total_cols": n_ct_dn,
            "corner90_deadzone_skipped_pct": round(100.0 * (n_ct_dn - live_col_cnt) / max(1, n_ct_dn), 2),
            "stage_gate_up": st_gu,
            "stage_down": st_dn,
        }
