"""Native NVRTC CUDA kernels built directly on the Mirror Equality principle:

    Outcome_(t-1) + ObservedBehavior_(t-1) = Strategy_t = ObservedBehavior_(t+1) + Outcome_(t+1)

Self-contained NVRTC + CUDA Driver API runtime (compiles once, caches PTX under
`data/native/ptx-cache/`, dispatches directly onto `torch.cuda.current_stream(device)`).

Kernels:
1. `mirror_equality_q1_gemv_f32`:
   Single-launch Binary-Halving Q1 FWHT GEMV whose inner plane loop IS the mirror equality:
   - Left wing:  Strategy_t = Outcome_(t-1) + ObservedBehavior_(t-1)
   - Right wing: computes next Q1 cube ObservedBehavior_(t+1) = Delta_y_t and bounds the
     unobserved geometric tail Outcome_(t+1) ~= sqrt(4/3) * ||Delta_y_t||_2 across an
     8-row warp tile in shared memory.
   - Mirror Convergence Gate: once (t + 1) >= min_planes and the tile mirror defect
     1.1547 * ||Delta_y_t||_2 / (||Strategy_t + Delta_y_t||_2 + eps) < mirror_tol,
     the entire block exits early and skips all remaining GDDR6 weight-plane reads.
   - Supports both shared `qscale[1, N, NG]` (positional_halving) and per-plane `qscale[P, N, NG]`,
     and works for arbitrary K (single-pass FWHT in shared memory for K <= 4096, chunked for K > 4096).

2. `mirror_equality_balance_step_f32`:
   Direct GPU operator for the Mirror Equality conservation law on state batches `[B, D]`:
   computes `Strategy_t = Outcome_(t-1) + ObservedBehavior_(t-1)`, solves or verifies the
   right wing `ObservedBehavior_(t+1) + Outcome_(t+1)`, and emits per-row L2 mirror balance
   defect `||L - R||_2 / (||Strategy_t||_2 + eps)` via warp/block reductions.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import threading
from typing import Dict, Optional, Tuple

import torch

from src.colibri_nvrtc import PTX_CACHE_DIR, _load_driver_and_nvrtc

_MIRROR_CUDA_SOURCE = r"""
__device__ __forceinline__ float bf16_to_f32(unsigned short b) {
    return __uint_as_float(((unsigned int)b) << 16);
}

__device__ __forceinline__ unsigned short f32_to_bf16_rne(float val) {
    unsigned int bits = __float_as_uint(val);
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u) {
        return 0x7fc0u;
    }
    unsigned int lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return (unsigned short)(bits >> 16);
}

/* Orthonormal Block FWHT H_bar = H / sqrt(GS) + per-group sum (zero butterfly multiplies) */
__device__ __forceinline__ void mirror_block_fwht_and_sum(
    float* s_x,
    float* s_xsum,
    int ng_cur,
    int GS,
    int warp_id,
    int lane,
    int warps_per_block
) {
    if (GS == 16) {
        for (int g_base = warp_id * 2; g_base < ng_cur; g_base += warps_per_block * 2) {
            int g = g_base + (lane >> 4);
            int sub = lane & 15;
            float v = (g < ng_cur) ? s_x[g * 16 + sub] : 0.0f;
            #pragma unroll
            for (int d = 1; d < 16; d <<= 1) {
                float p = __shfl_xor_sync(0xffffffffu, v, d);
                v = (lane & d) ? (p - v) : (v + p);
            }
            v *= 0.25f;
            if (g < ng_cur) s_x[g * 16 + sub] = v;
            float s = v;
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            s += __shfl_xor_sync(0xffffffffu, s, 4);
            s += __shfl_xor_sync(0xffffffffu, s, 8);
            if (sub == 0 && g < ng_cur) s_xsum[g] = s;
        }
    } else if (GS == 32) {
        const float inv_sq32 = 0.176776695f;
        for (int g = warp_id; g < ng_cur; g += warps_per_block) {
            float v = s_x[g * 32 + lane];
            #pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                float p = __shfl_xor_sync(0xffffffffu, v, d);
                v = (lane & d) ? (p - v) : (v + p);
            }
            v *= inv_sq32;
            s_x[g * 32 + lane] = v;
            float s = v;
            #pragma unroll
            for (int d = 16; d > 0; d >>= 1) {
                s += __shfl_xor_sync(0xffffffffu, s, d);
            }
            if (lane == 0) s_xsum[g] = s;
        }
    } else {
        /* GS == 64: two 32-element halves per warp + register butterfly d=32 */
        for (int g = warp_id; g < ng_cur; g += warps_per_block) {
            int base = g * 64;
            float v0 = s_x[base + lane];
            float v1 = s_x[base + 32 + lane];
            #pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                float p0 = __shfl_xor_sync(0xffffffffu, v0, d);
                float p1 = __shfl_xor_sync(0xffffffffu, v1, d);
                v0 = (lane & d) ? (p0 - v0) : (v0 + p0);
                v1 = (lane & d) ? (p1 - v1) : (v1 + p1);
            }
            float y0 = (v0 + v1) * 0.125f;
            float y1 = (v0 - v1) * 0.125f;
            s_x[base + lane] = y0;
            s_x[base + 32 + lane] = y1;
            float s = y0 + y1;
            #pragma unroll
            for (int d = 16; d > 0; d >>= 1) {
                s += __shfl_xor_sync(0xffffffffu, s, d);
            }
            if (lane == 0) s_xsum[g] = s;
        }
    }
}

/* Helper: compute one Q1 cube signed dot contribution for a chunk of rotated x in shared memory */
__device__ __forceinline__ float eval_q1_plane_chunk(
    const float* s_x,
    const float* s_xsum,
    const unsigned char* b_chunk,
    const unsigned short* s_row,
    float p_mult,
    int c_cur,
    int ng_cur,
    int g_start,
    int GS,
    int lane
) {
    float partial = 0.0f;
    const unsigned short* b_u16 = reinterpret_cast<const unsigned short*>(b_chunk);
    for (int c_off = lane * 16; c_off < c_cur; c_off += 512) {
        unsigned int bits16 = (unsigned int)b_u16[c_off >> 4];
        float sv = bf16_to_f32(s_row[g_start + (c_off / GS)]) * p_mult;
        float d1 = 0.0f;
        while (bits16) {
            int b = __ffs(bits16) - 1;
            d1 += s_x[c_off + b];
            bits16 &= (bits16 - 1u);
        }
        partial += 2.0f * sv * d1;
    }
    for (int g = lane; g < ng_cur; g += 32) {
        partial -= (bf16_to_f32(s_row[g_start + g]) * p_mult) * s_xsum[g];
    }
    return partial;
}

/* Kernel #1: Mirror Equality Q1-Cube Adaptive GEMV
 *
 * Implements the exact Mirror Equality inside a single GPU kernel launch:
 *   Outcome_(t-1) + ObservedBehavior_(t-1) = Strategy_t = ObservedBehavior_(t+1) + Outcome_(t+1)
 *
 * Grid:  ((N + 7) / 8, M, 1)
 * Block: (256, 1, 1) -> 8 warps = 8 output rows (1 tile = 1 mirror convergence unit)
 */
extern "C" __global__ void mirror_equality_q1_gemv_f32(
    const unsigned short* __restrict__ x_bf16,
    const unsigned char* __restrict__ packed_lo,
    const unsigned short* __restrict__ qscale,
    const unsigned short* __restrict__ qzero,
    const unsigned short* __restrict__ row_scale,
    const unsigned short* __restrict__ col_scale,
    const unsigned short* __restrict__ bias_bf16,
    void* __restrict__ y_out,
    int* __restrict__ used_planes_out,
    float* __restrict__ mirror_defect_out,
    int M, int N, int K, int GS,
    int P_max, int P_min,
    float mirror_tol,
    int out_is_bf16,
    int shared_scale
) {
    extern __shared__ char smem_raw[];
    int chunk_cap = (K < 4096) ? K : 4096;
    float* s_x = reinterpret_cast<float*>(smem_raw);
    float* s_xsum = s_x + chunk_cap;
    float* sm_strat_e = s_xsum + (chunk_cap / GS);
    float* sm_obs_e = sm_strat_e + 8;

    int tid = threadIdx.x;
    int m = blockIdx.y;
    if (m >= M) return;

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5; /* 8 */
    int row = blockIdx.x * warps_per_block + warp_id;
    bool valid_row = (row < N);

    const unsigned short* x_row = x_bf16 + (long long)m * K;
    int num_groups_full = K / GS;
    int k8 = K >> 3;
    long long plane_byte_stride = (long long)N * k8;
    long long plane_grp_stride = (long long)N * num_groups_full;
    int single_chunk = (K <= chunk_cap);

    /* Initial Mirror State (t = 0):
     *   Outcome_(-1) = qzero projection
     *   ObservedBehavior_(-1) = 0
     */
    float zero_acc = 0.0f;
    if (single_chunk) {
        int ng_cur = K / GS;
        for (int c = tid; c < K; c += blockDim.x) {
            s_x[c] = bf16_to_f32(x_row[c]) * bf16_to_f32(col_scale[c]);
        }
        __syncthreads();
        mirror_block_fwht_and_sum(s_x, s_xsum, ng_cur, GS, warp_id, lane, warps_per_block);
        __syncthreads();
        if (valid_row) {
            const unsigned short* z_row = qzero + (long long)row * num_groups_full;
            for (int g = lane; g < ng_cur; g += 32) {
                zero_acc += bf16_to_f32(z_row[g]) * s_xsum[g];
            }
        }
    } else {
        for (int k_start = 0; k_start < K; k_start += chunk_cap) {
            int c_cur = K - k_start;
            if (c_cur > chunk_cap) c_cur = chunk_cap;
            int ng_cur = c_cur / GS;
            int g_start = k_start / GS;
            for (int c = tid; c < c_cur; c += blockDim.x) {
                int ka = k_start + c;
                s_x[c] = bf16_to_f32(x_row[ka]) * bf16_to_f32(col_scale[ka]);
            }
            __syncthreads();
            mirror_block_fwht_and_sum(s_x, s_xsum, ng_cur, GS, warp_id, lane, warps_per_block);
            __syncthreads();
            if (valid_row) {
                const unsigned short* z_row = qzero + (long long)row * num_groups_full;
                for (int g = lane; g < ng_cur; g += 32) {
                    zero_acc += bf16_to_f32(z_row[g_start + g]) * s_xsum[g];
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        zero_acc += __shfl_down_sync(0xffffffffu, zero_acc, offset);
    }

    float outcome_prev = (lane == 0 && valid_row) ? zero_acc : 0.0f;
    float obs_prev = 0.0f;
    int used_planes = 0;
    float last_defect = 0.0f;

    /* Mirror Equality Recurrence over Q1 Cubes t = 0 .. P_max - 1 */
    for (int t = 0; t < P_max; ++t) {
        /* 1. Left Wing: Strategy_t = Outcome_(t-1) + ObservedBehavior_(t-1) */
        float strategy_t = outcome_prev + obs_prev;

        /* 2. Right Wing Action: compute ObservedBehavior_(t+1) = Delta_y_t from Q1 cube t */
        float obs_partial = 0.0f;
        float p_mult = shared_scale ? ldexpf(1.0f, -t) : 1.0f;
        const unsigned short* s_row = valid_row
            ? (shared_scale
                ? (qscale + (long long)row * num_groups_full)
                : (qscale + (long long)t * plane_grp_stride + (long long)row * num_groups_full))
            : nullptr;

        if (single_chunk) {
            if (valid_row) {
                const unsigned char* b_row = packed_lo + (long long)t * plane_byte_stride + (long long)row * k8;
                obs_partial = eval_q1_plane_chunk(
                    s_x, s_xsum, b_row, s_row, p_mult, K, num_groups_full, 0, GS, lane
                );
            }
        } else {
            for (int k_start = 0; k_start < K; k_start += chunk_cap) {
                int c_cur = K - k_start;
                if (c_cur > chunk_cap) c_cur = chunk_cap;
                int ng_cur = c_cur / GS;
                int g_start = k_start / GS;
                for (int c = tid; c < c_cur; c += blockDim.x) {
                    int ka = k_start + c;
                    s_x[c] = bf16_to_f32(x_row[ka]) * bf16_to_f32(col_scale[ka]);
                }
                __syncthreads();
                mirror_block_fwht_and_sum(s_x, s_xsum, ng_cur, GS, warp_id, lane, warps_per_block);
                __syncthreads();
                if (valid_row) {
                    const unsigned char* b_chunk =
                        packed_lo + (long long)t * plane_byte_stride + (long long)row * k8 + (k_start >> 3);
                    obs_partial += eval_q1_plane_chunk(
                        s_x, s_xsum, b_chunk, s_row, p_mult, c_cur, ng_cur, g_start, GS, lane
                    );
                }
                __syncthreads();
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            obs_partial += __shfl_down_sync(0xffffffffu, obs_partial, offset);
        }

        float obs_next = (lane == 0 && valid_row) ? obs_partial : 0.0f;

        /* 3. Advance Mirror Equality State:
         *    Outcome_t = Strategy_t
         *    ObservedBehavior_t = obs_next
         *    (so next Strategy_(t+1) = Outcome_t + ObservedBehavior_t)
         */
        float next_strategy = strategy_t + obs_next;
        if (lane == 0) {
            float rs = valid_row ? bf16_to_f32(row_scale[row]) : 0.0f;
            float scaled_strat = next_strategy * rs;
            float scaled_obs = obs_next * rs;
            sm_strat_e[warp_id] = valid_row ? (scaled_strat * scaled_strat) : 0.0f;
            sm_obs_e[warp_id] = valid_row ? (scaled_obs * scaled_obs) : 0.0f;
        }
        __syncthreads();

        /* 4. Evaluate Right-Wing Geometric Tail Bound Outcome_(t+1) ~= sqrt(4/3) * ||obs_next||_2 */
        float tile_strat_e = sm_strat_e[0] + sm_strat_e[1] + sm_strat_e[2] + sm_strat_e[3]
                           + sm_strat_e[4] + sm_strat_e[5] + sm_strat_e[6] + sm_strat_e[7];
        float tile_obs_e = sm_obs_e[0] + sm_obs_e[1] + sm_obs_e[2] + sm_obs_e[3]
                         + sm_obs_e[4] + sm_obs_e[5] + sm_obs_e[6] + sm_obs_e[7];
        float defect = (1.15470054f * sqrtf(tile_obs_e)) / (sqrtf(tile_strat_e) + 1e-8f);

        outcome_prev = strategy_t;
        obs_prev = obs_next;
        used_planes = t + 1;
        last_defect = defect;

        /* 5. Mirror Convergence Gate: stop reading deeper Q1 planes from VRAM once balanced */
        bool converged = (used_planes >= P_min) && (mirror_tol > 0.0f) && (defect < mirror_tol);
        if (converged) {
            break;
        }
        __syncthreads();
    }

    /* Emit final Strategy = Outcome_(T-1) + ObservedBehavior_(T-1) */
    if (lane == 0 && valid_row) {
        float y_val = (outcome_prev + obs_prev) * bf16_to_f32(row_scale[row]);
        if (bias_bf16 != nullptr) {
            y_val += bf16_to_f32(bias_bf16[row]);
        }
        long long idx = (long long)m * N + row;
        if (out_is_bf16) {
            reinterpret_cast<unsigned short*>(y_out)[idx] = f32_to_bf16_rne(y_val);
        } else {
            reinterpret_cast<float*>(y_out)[idx] = y_val;
        }
    }

    /* Emit per-tile telemetry: planes read & final mirror defect */
    if (tid == 0) {
        int tiles_per_row = (N + 7) >> 3;
        long long tile_idx = (long long)m * tiles_per_row + blockIdx.x;
        if (used_planes_out != nullptr) {
            used_planes_out[tile_idx] = used_planes;
        }
        if (mirror_defect_out != nullptr) {
            mirror_defect_out[tile_idx] = last_defect;
        }
    }
}

/* Kernel #2: Direct Mirror Equality Conservation Operator on State Batches [B, D]
 *
 * Mode 0 (solve right outcome):
 *   strategy_t[i]   = outcome_prev[i] + obs_prev[i]
 *   outcome_next[i] = strategy_t[i]   - obs_next[i]
 *   balance_defect  = ||(outcome_prev + obs_prev) - (obs_next + outcome_next)||_2 / (||strategy_t||_2 + 1e-8)
 *
 * Mode 1 (verify & mirror-project given both wings):
 *   L[i]            = outcome_prev[i] + obs_prev[i]
 *   R[i]            = obs_next[i]     + outcome_next_in[i]
 *   strategy_t[i]   = 0.5f * (L[i] + R[i])
 *   outcome_next[i] = strategy_t[i] - obs_next[i]
 *   balance_defect  = ||L - R||_2 / (||strategy_t||_2 + 1e-8)
 */
extern "C" __global__ void mirror_equality_balance_step_f32(
    const float* __restrict__ outcome_prev,
    const float* __restrict__ obs_prev,
    const float* __restrict__ obs_next,
    const float* __restrict__ outcome_next_in,
    float* __restrict__ strategy_out,
    float* __restrict__ outcome_next_out,
    float* __restrict__ defect_out,
    int B, int D, int mode
) {
    __shared__ float sm_diff[8];
    __shared__ float sm_norm[8];

    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    long long base = (long long)b * D;
    float local_diff_sq = 0.0f;
    float local_norm_sq = 0.0f;

    for (int i = tid; i < D; i += blockDim.x) {
        float L = outcome_prev[base + i] + obs_prev[base + i];
        float onext = obs_next[base + i];
        float strat;
        float out_n;
        float diff;
        if (mode == 0 || outcome_next_in == nullptr) {
            strat = L;
            out_n = strat - onext;
            float R = onext + out_n;
            diff = L - R;
        } else {
            float R = onext + outcome_next_in[base + i];
            strat = 0.5f * (L + R);
            out_n = strat - onext;
            diff = L - R;
        }
        strategy_out[base + i] = strat;
        outcome_next_out[base + i] = out_n;
        local_diff_sq += diff * diff;
        local_norm_sq += strat * strat;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_diff_sq += __shfl_down_sync(0xffffffffu, local_diff_sq, off);
        local_norm_sq += __shfl_down_sync(0xffffffffu, local_norm_sq, off);
    }
    if (lane == 0) {
        sm_diff[warp_id] = local_diff_sq;
        sm_norm[warp_id] = local_norm_sq;
    }
    __syncthreads();

    if (tid == 0 && defect_out != nullptr) {
        float tot_d = sm_diff[0] + sm_diff[1] + sm_diff[2] + sm_diff[3]
                    + sm_diff[4] + sm_diff[5] + sm_diff[6] + sm_diff[7];
        float tot_n = sm_norm[0] + sm_norm[1] + sm_norm[2] + sm_norm[3]
                    + sm_norm[4] + sm_norm[5] + sm_norm[6] + sm_norm[7];
        defect_out[b] = sqrtf(tot_d) / (sqrtf(tot_n) + 1e-8f);
    }
}

/* Kernel #3: UNI x Mirror Equality Recursive Binding Q1-Cube GEMV
 *
 * Connects the linear base (t=0) and recursive Q1 cube composition f(f(...(x))) inside a single
 * CUDA launch using Mirror Equality + UNI binding invariants:
 *   Phi   = exp(-1.5)         = 0.22313016f  (orthogonal novelty damping weight)
 *   Omega = pi / 2            = 1.57079633f  (orthogonal angle scale)
 *   Sigma = sqrt(pi)          = 1.77245385f  (Gaussian concentration scale after Sinkhorn+FWHT)
 *   Delta = 1 - cos(10)       = 1.83907153f  (phase oscillation normalization)
 *
 * For each 8-row tile at recursion step t:
 *   - Mirror defect:       defect = 1.15470054 * ||Delta_y_t||_2 / (||Strategy_t + Delta_y_t||_2 + eps)
 *   - Orthogonal novelty:  theta / Omega = acos(|<Strategy_t, Delta_y_t>| / (||S||*||d|| + eps)) / (pi/2)
 *   - Tile concentration:  lambda_tile = exp(-H(p_0..p_7)) in [1/8, 1]  (UNI Lambda entropy analogue)
 *   - Effective bind tol:  tol_eff = tol_base * (1 - Phi * (theta / Omega)) * (1 - lambda_tile / Sigma)
 *
 * High-energy spike neurons (high lambda_tile) and orthogonal updates (theta -> Omega) receive
 * strictly tighter convergence gates (more Q1 cubes), while flat background tiles stop early and
 * skip remaining GDDR6 reads.
 */
extern "C" __global__ void mirror_uni_binding_q1_gemv_f32(
    const unsigned short* __restrict__ x_bf16,
    const unsigned char* __restrict__ packed_lo,
    const unsigned short* __restrict__ qscale,
    const unsigned short* __restrict__ qzero,
    const unsigned short* __restrict__ row_scale,
    const unsigned short* __restrict__ col_scale,
    const unsigned short* __restrict__ bias_bf16,
    void* __restrict__ y_out,
    int* __restrict__ used_planes_out,
    float* __restrict__ mirror_defect_out,
    float* __restrict__ bind_strength_out,
    int M, int N, int K, int GS,
    int P_max, int P_min,
    float base_tol,
    int out_is_bf16,
    int shared_scale
) {
    const float UNI_PHI   = 0.22313016f; /* exp(-1.5)  */
    const float UNI_OMEGA = 1.57079633f; /* pi / 2     */
    const float UNI_SIGMA = 1.77245385f; /* sqrt(pi)   */

    extern __shared__ char smem_raw[];
    int chunk_cap = (K < 4096) ? K : 4096;
    float* s_x = reinterpret_cast<float*>(smem_raw);
    float* s_xsum = s_x + chunk_cap;
    float* sm_next_e = s_xsum + (chunk_cap / GS); /* 8 */
    float* sm_prev_e = sm_next_e + 8;             /* 8 */
    float* sm_obs_e  = sm_prev_e + 8;             /* 8 */
    float* sm_dot    = sm_obs_e + 8;              /* 8 */

    int tid = threadIdx.x;
    int m = blockIdx.y;
    if (m >= M) return;

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5; /* 8 */
    int row = blockIdx.x * warps_per_block + warp_id;
    bool valid_row = (row < N);

    const unsigned short* x_row = x_bf16 + (long long)m * K;
    int num_groups_full = K / GS;
    int k8 = K >> 3;
    long long plane_byte_stride = (long long)N * k8;
    long long plane_grp_stride = (long long)N * num_groups_full;
    int single_chunk = (K <= chunk_cap);

    float zero_acc = 0.0f;
    if (single_chunk) {
        int ng_cur = K / GS;
        for (int c = tid; c < K; c += blockDim.x) {
            s_x[c] = bf16_to_f32(x_row[c]) * bf16_to_f32(col_scale[c]);
        }
        __syncthreads();
        mirror_block_fwht_and_sum(s_x, s_xsum, ng_cur, GS, warp_id, lane, warps_per_block);
        __syncthreads();
        if (valid_row) {
            const unsigned short* z_row = qzero + (long long)row * num_groups_full;
            for (int g = lane; g < ng_cur; g += 32) {
                zero_acc += bf16_to_f32(z_row[g]) * s_xsum[g];
            }
        }
    } else {
        for (int k_start = 0; k_start < K; k_start += chunk_cap) {
            int c_cur = K - k_start;
            if (c_cur > chunk_cap) c_cur = chunk_cap;
            int ng_cur = c_cur / GS;
            int g_start = k_start / GS;
            for (int c = tid; c < c_cur; c += blockDim.x) {
                int ka = k_start + c;
                s_x[c] = bf16_to_f32(x_row[ka]) * bf16_to_f32(col_scale[ka]);
            }
            __syncthreads();
            mirror_block_fwht_and_sum(s_x, s_xsum, ng_cur, GS, warp_id, lane, warps_per_block);
            __syncthreads();
            if (valid_row) {
                const unsigned short* z_row = qzero + (long long)row * num_groups_full;
                for (int g = lane; g < ng_cur; g += 32) {
                    zero_acc += bf16_to_f32(z_row[g_start + g]) * s_xsum[g];
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        zero_acc += __shfl_down_sync(0xffffffffu, zero_acc, offset);
    }

    float outcome_prev = (lane == 0 && valid_row) ? zero_acc : 0.0f;
    float obs_prev = 0.0f;
    int used_planes = 0;
    float last_defect = 0.0f;
    float last_bind_strength = 1.0f;

    for (int t = 0; t < P_max; ++t) {
        /* Linear base (t=0) -> Recursive Mirror composition (t>=1) */
        float strategy_t = outcome_prev + obs_prev;
        float obs_partial = 0.0f;
        float p_mult = shared_scale ? ldexpf(1.0f, -t) : 1.0f;
        const unsigned short* s_row = valid_row
            ? (shared_scale
                ? (qscale + (long long)row * num_groups_full)
                : (qscale + (long long)t * plane_grp_stride + (long long)row * num_groups_full))
            : nullptr;

        if (single_chunk) {
            if (valid_row) {
                const unsigned char* b_row = packed_lo + (long long)t * plane_byte_stride + (long long)row * k8;
                obs_partial = eval_q1_plane_chunk(
                    s_x, s_xsum, b_row, s_row, p_mult, K, num_groups_full, 0, GS, lane
                );
            }
        } else {
            for (int k_start = 0; k_start < K; k_start += chunk_cap) {
                int c_cur = K - k_start;
                if (c_cur > chunk_cap) c_cur = chunk_cap;
                int ng_cur = c_cur / GS;
                int g_start = k_start / GS;
                for (int c = tid; c < c_cur; c += blockDim.x) {
                    int ka = k_start + c;
                    s_x[c] = bf16_to_f32(x_row[ka]) * bf16_to_f32(col_scale[ka]);
                }
                __syncthreads();
                mirror_block_fwht_and_sum(s_x, s_xsum, ng_cur, GS, warp_id, lane, warps_per_block);
                __syncthreads();
                if (valid_row) {
                    const unsigned char* b_chunk =
                        packed_lo + (long long)t * plane_byte_stride + (long long)row * k8 + (k_start >> 3);
                    obs_partial += eval_q1_plane_chunk(
                        s_x, s_xsum, b_chunk, s_row, p_mult, c_cur, ng_cur, g_start, GS, lane
                    );
                }
                __syncthreads();
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            obs_partial += __shfl_down_sync(0xffffffffu, obs_partial, offset);
        }

        float obs_next = (lane == 0 && valid_row) ? obs_partial : 0.0f;
        float next_strategy = strategy_t + obs_next;

        if (lane == 0) {
            float rs = valid_row ? bf16_to_f32(row_scale[row]) : 0.0f;
            float s_prev = strategy_t * rs;
            float s_next = next_strategy * rs;
            float d_cur  = obs_next * rs;
            sm_prev_e[warp_id] = valid_row ? (s_prev * s_prev) : 0.0f;
            sm_next_e[warp_id] = valid_row ? (s_next * s_next) : 0.0f;
            sm_obs_e[warp_id]  = valid_row ? (d_cur * d_cur) : 0.0f;
            sm_dot[warp_id]    = valid_row ? (s_prev * d_cur) : 0.0f;
        }
        __syncthreads();

        /* Compute tile-level Mirror + UNI binding invariants */
        float tot_next_e = 0.0f, tot_prev_e = 0.0f, tot_obs_e = 0.0f, tot_dot = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            tot_next_e += sm_next_e[r];
            tot_prev_e += sm_prev_e[r];
            tot_obs_e  += sm_obs_e[r];
            tot_dot    += sm_dot[r];
        }

        float defect = (1.15470054f * sqrtf(tot_obs_e)) / (sqrtf(tot_next_e) + 1e-8f);

        /* Orthogonal novelty ratio: theta / Omega in [0, 1] */
        float cos_abs = fabsf(tot_dot) / (sqrtf(tot_prev_e * tot_obs_e) + 1e-8f);
        if (cos_abs > 1.0f) cos_abs = 1.0f;
        float ortho_ratio = acosf(cos_abs) / UNI_OMEGA;

        /* Tile energy concentration: lambda_tile = exp(-H(p)) in [1/8, 1] */
        float inv_e = 1.0f / (tot_next_e + 1e-12f);
        float entropy = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            float pr = sm_next_e[r] * inv_e;
            if (pr > 1e-7f) {
                entropy -= pr * logf(pr);
            }
        }
        float lambda_tile = expf(-entropy);

        /* UNI Binding Gate factor: tighter tol on spike/orthogonal tiles, relaxed on flat noise */
        float bind_factor = (1.0f - UNI_PHI * ortho_ratio) * (1.0f - lambda_tile / UNI_SIGMA);
        if (bind_factor < 0.25f) bind_factor = 0.25f;
        float eff_tol = base_tol * bind_factor;

        outcome_prev = strategy_t;
        obs_prev = obs_next;
        used_planes = t + 1;
        last_defect = defect;
        last_bind_strength = 1.0f / bind_factor;

        if ((used_planes >= P_min) && (base_tol > 0.0f) && (defect < eff_tol)) {
            break;
        }
        __syncthreads();
    }

    if (lane == 0 && valid_row) {
        float y_val = (outcome_prev + obs_prev) * bf16_to_f32(row_scale[row]);
        if (bias_bf16 != nullptr) {
            y_val += bf16_to_f32(bias_bf16[row]);
        }
        long long idx = (long long)m * N + row;
        if (out_is_bf16) {
            reinterpret_cast<unsigned short*>(y_out)[idx] = f32_to_bf16_rne(y_val);
        } else {
            reinterpret_cast<float*>(y_out)[idx] = y_val;
        }
    }

    if (tid == 0) {
        int tiles_per_row = (N + 7) >> 3;
        long long tile_idx = (long long)m * tiles_per_row + blockIdx.x;
        if (used_planes_out != nullptr) used_planes_out[tile_idx] = used_planes;
        if (mirror_defect_out != nullptr) mirror_defect_out[tile_idx] = last_defect;
        if (bind_strength_out != nullptr) bind_strength_out[tile_idx] = last_bind_strength;
    }
}

/* Kernel #4: Fractal Intersecting-Ring Q1 Step (Free-Orientation Multi-Kernel Node)
 *
 * Implements the author's fractal intersecting-circle topology directly in GPU silicon:
 * - Every call executes ONE arbitrary step on a selected subset of 8-row tiles (`active_tiles[]`),
 *   starting from the shared assembly line (`line_spine[m, row]` = current Strategy point) and
 *   writing its mirror update BACK onto the exact same line (`line_spine[m, row] += obs_next`),
 *   so the start point of this step and the entry point of the next step ALWAYS intersect on one line.
 * - Automatically turns EVERY active point on the ring into a potential new intersection node:
 *   computes the Mirror Equality + UNI binding defect (Phi, Omega, Sigma, lambda_tile) on the tile,
 *   and if `defect >= eff_tol` atomically appends the tile index into `spawn_tiles[]` (`atomicAdd`),
 *   allowing the next ring to branch freely to another plane, another projection, an MoE expert,
 *   or another kernel without wasting a single GPU block on already-converged tiles.
 *
 * Grid:  (num_active_tiles, M, 1)  [indirect sparse dispatch!]
 * Block: (256, 1, 1) -> 8 warps = 8 rows per tile
 */
extern "C" __global__ void mirror_fractal_ring_q1_step_f32(
    const unsigned short* __restrict__ x_bf16,
    const unsigned char* __restrict__ packed_lo,
    const unsigned short* __restrict__ qscale,
    const unsigned short* __restrict__ qzero,
    const unsigned short* __restrict__ row_scale,
    const unsigned short* __restrict__ col_scale,
    const unsigned short* __restrict__ bias_bf16,
    const int* __restrict__ active_tiles,
    float* __restrict__ line_spine,
    float* __restrict__ last_obs_out,
    int* __restrict__ spawn_tiles_out,
    int* __restrict__ spawn_count_out,
    float* __restrict__ tile_defect_out,
    int M, int N, int K, int GS,
    int num_active_tiles,
    int plane_idx,
    int include_zero,
    int shared_scale,
    float base_tol,
    float branch_weight
) {
    const float UNI_PHI   = 0.22313016f; /* exp(-1.5)  */
    const float UNI_OMEGA = 1.57079633f; /* pi / 2     */
    const float UNI_SIGMA = 1.77245385f; /* sqrt(pi)   */

    extern __shared__ char smem_raw[];
    int chunk_cap = (K < 4096) ? K : 4096;
    float* s_x = reinterpret_cast<float*>(smem_raw);
    float* s_xsum = s_x + chunk_cap;
    float* sm_prev_e = s_xsum + (chunk_cap / GS); /* 8 */
    float* sm_next_e = sm_prev_e + 8;             /* 8 */
    float* sm_obs_e  = sm_next_e + 8;             /* 8 */
    float* sm_dot    = sm_obs_e + 8;              /* 8 */

    int tid = threadIdx.x;
    int m = blockIdx.y;
    int a_idx = blockIdx.x;
    if (m >= M || a_idx >= num_active_tiles) return;

    int tile_id = (active_tiles != nullptr) ? active_tiles[a_idx] : a_idx;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = tile_id * warps_per_block + warp_id;
    bool valid_row = (row >= 0 && row < N);

    const unsigned short* x_row = x_bf16 + (long long)m * K;
    int num_groups_full = K / GS;
    int k8 = K >> 3;
    long long plane_byte_stride = (long long)N * k8;
    long long plane_grp_stride = (long long)N * num_groups_full;
    float p_mult = shared_scale ? ldexpf(1.0f, -plane_idx) : 1.0f;

    const unsigned short* s_row = valid_row
        ? (shared_scale
            ? (qscale + (long long)row * num_groups_full)
            : (qscale + (long long)plane_idx * plane_grp_stride + (long long)row * num_groups_full))
        : nullptr;
    const unsigned short* z_row = valid_row ? (qzero + (long long)row * num_groups_full) : nullptr;

    float partial = 0.0f;
    for (int k_start = 0; k_start < K; k_start += chunk_cap) {
        int c_cur = K - k_start;
        if (c_cur > chunk_cap) c_cur = chunk_cap;
        int ng_cur = c_cur / GS;
        int g_start = k_start / GS;

        for (int c = tid; c < c_cur; c += blockDim.x) {
            int ka = k_start + c;
            s_x[c] = bf16_to_f32(x_row[ka]) * bf16_to_f32(col_scale[ka]);
        }
        __syncthreads();
        mirror_block_fwht_and_sum(s_x, s_xsum, ng_cur, GS, warp_id, lane, warps_per_block);
        __syncthreads();

        if (valid_row) {
            if (include_zero) {
                for (int g = lane; g < ng_cur; g += 32) {
                    partial += bf16_to_f32(z_row[g_start + g]) * s_xsum[g];
                }
            }
            const unsigned char* b_chunk =
                packed_lo + (long long)plane_idx * plane_byte_stride + (long long)row * k8 + (k_start >> 3);
            partial += eval_q1_plane_chunk(
                s_x, s_xsum, b_chunk, s_row, p_mult, c_cur, ng_cur, g_start, GS, lane
            );
        }
        if (k_start + chunk_cap < K) {
            __syncthreads();
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffffu, partial, offset);
    }

    /* Intersect with the shared assembly line (Strategy_t -> Strategy_(t+1)) */
    if (lane == 0) {
        if (valid_row) {
            long long idx = (long long)m * N + row;
            float prev_line = line_spine[idx];
            float obs_step = partial * bf16_to_f32(row_scale[row]) * branch_weight;
            if (include_zero && bias_bf16 != nullptr) {
                obs_step += bf16_to_f32(bias_bf16[row]) * branch_weight;
            }
            float next_line = prev_line + obs_step;
            line_spine[idx] = next_line;
            if (last_obs_out != nullptr) {
                last_obs_out[idx] = obs_step;
            }
            sm_prev_e[warp_id] = prev_line * prev_line;
            sm_next_e[warp_id] = next_line * next_line;
            sm_obs_e[warp_id]  = obs_step * obs_step;
            sm_dot[warp_id]    = prev_line * obs_step;
        } else {
            sm_prev_e[warp_id] = 0.0f;
            sm_next_e[warp_id] = 0.0f;
            sm_obs_e[warp_id]  = 0.0f;
            sm_dot[warp_id]    = 0.0f;
        }
    }
    __syncthreads();

    /* Automatically evaluate whether this ring point spawns a new intersection node */
    if (tid == 0) {
        float tot_next_e = 0.0f, tot_prev_e = 0.0f, tot_obs_e = 0.0f, tot_dot = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            tot_next_e += sm_next_e[r];
            tot_prev_e += sm_prev_e[r];
            tot_obs_e  += sm_obs_e[r];
            tot_dot    += sm_dot[r];
        }
        float defect = (1.15470054f * sqrtf(tot_obs_e)) / (sqrtf(tot_next_e) + 1e-8f);
        float cos_abs = fabsf(tot_dot) / (sqrtf(tot_prev_e * tot_obs_e) + 1e-8f);
        if (cos_abs > 1.0f) cos_abs = 1.0f;
        float ortho_ratio = acosf(cos_abs) / UNI_OMEGA;

        float inv_e = 1.0f / (tot_next_e + 1e-12f);
        float entropy = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            float pr = sm_next_e[r] * inv_e;
            if (pr > 1e-7f) entropy -= pr * logf(pr);
        }
        float lambda_tile = expf(-entropy);
        float bind_factor = (1.0f - UNI_PHI * ortho_ratio) * (1.0f - lambda_tile / UNI_SIGMA);
        if (bind_factor < 0.25f) bind_factor = 0.25f;
        float eff_tol = base_tol * bind_factor;

        int tiles_per_row = (N + 7) >> 3;
        if (tile_defect_out != nullptr) {
            tile_defect_out[(long long)m * tiles_per_row + tile_id] = defect;
        }
        if (spawn_tiles_out != nullptr && spawn_count_out != nullptr && m == 0) {
            if (base_tol <= 0.0f || defect >= eff_tol) {
                int pos = atomicAdd(spawn_count_out, 1);
                spawn_tiles_out[pos] = tile_id;
            }
        }
    }
}
"""

_MIRROR_KERNEL_NAMES = (
    b"mirror_equality_q1_gemv_f32",
    b"mirror_equality_balance_step_f32",
    b"mirror_uni_binding_q1_gemv_f32",
    b"mirror_fractal_ring_q1_step_f32",
)

_LOCK = threading.Lock()
_DEVICE_MODULES: Dict[int, Dict[bytes, ctypes.c_void_p]] = {}


def _compile_or_load_mirror_ptx(major: int, minor: int, nvrtc) -> bytes:
    src_bytes = _MIRROR_CUDA_SOURCE.encode("utf-8")
    digest = hashlib.sha256(src_bytes).hexdigest()[:16]
    PTX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = PTX_CACHE_DIR / f"mirror_equality_sm{major}{minor}_{digest}.ptx"
    if cache_file.is_file():
        data = cache_file.read_bytes()
        return data if data.endswith(b"\x00") else data + b"\x00"

    prog = ctypes.c_void_p()
    rc = nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src_bytes, b"mirror_equality.cu", 0, None, None)
    if rc != 0:
        raise RuntimeError(f"nvrtcCreateProgram(mirror_equality) failed with error {rc}")
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
                f"NVRTC mirror_equality compile failed ({rc}):\n{log_buf.value.decode('utf-8', errors='replace')}"
            )
        ptx_sz = ctypes.c_size_t()
        nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_sz))
        ptx_buf = ctypes.create_string_buffer(ptx_sz.value)
        rc = nvrtc.nvrtcGetPTX(prog, ptx_buf)
        if rc != 0:
            raise RuntimeError(f"nvrtcGetPTX(mirror_equality) failed with error {rc}")
        raw_ptx = ptx_buf.raw
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    tmp = cache_file.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(raw_ptx)
    os.replace(tmp, cache_file)
    return raw_ptx


def get_mirror_kernels(device: torch.device):
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
            ptx = _compile_or_load_mirror_ptx(major, minor, nvrtc)
            ptx_buf = ctypes.create_string_buffer(ptx)
            mod = ctypes.c_void_p()
            rc = nvcuda.cuModuleLoadData(ctypes.byref(mod), ctypes.cast(ptx_buf, ctypes.c_void_p))
            if rc != 0:
                raise RuntimeError(f"cuModuleLoadData(mirror_equality) failed with CUresult={rc}")
            funcs: Dict[bytes, ctypes.c_void_p] = {"_module": mod}
            for name in _MIRROR_KERNEL_NAMES:
                fn = ctypes.c_void_p()
                rc = nvcuda.cuModuleGetFunction(ctypes.byref(fn), mod, name)
                if rc != 0:
                    raise RuntimeError(f"cuModuleGetFunction({name!r}) failed with CUresult={rc}")
                funcs[name] = fn
            _DEVICE_MODULES[dev_idx] = funcs
            return nvcuda, funcs


def launch_mirror_equality_q1_gemv(
    x_bf16: torch.Tensor,
    packed_lo: torch.Tensor,
    qscale: torch.Tensor,
    qzero: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor,
    y_out: torch.Tensor,
    *,
    group_size: int = 64,
    max_planes: int | None = None,
    min_planes: int = 3,
    mirror_tol: float = 0.0,
    bias: Optional[torch.Tensor] = None,
    return_telemetry: bool = False,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Launch `mirror_equality_q1_gemv_f32` onto the current CUDA stream.

    When `mirror_tol > 0`, each 8-row tile evaluates the Mirror Equality Cauchy defect
    `1.1547 * ||Delta_y_t||_2 / ||Strategy_t + Delta_y_t||_2` directly in shared memory
    and stops reading deeper Q1 planes from VRAM once `defect < mirror_tol` (and `t+1 >= min_planes`).
    """
    dev = x_bf16.device
    if dev.type != "cuda":
        raise ValueError("launch_mirror_equality_q1_gemv requires CUDA tensors")
    nvcuda, fns = get_mirror_kernels(dev)

    x_c = x_bf16.contiguous()
    M, K = int(x_c.shape[0]), int(x_c.shape[1])
    N = int(y_out.shape[1])
    if M == 0 or N == 0:
        return (None, None) if return_telemetry else None

    p_avail = int(packed_lo.shape[0])
    p_max = p_avail if max_planes is None else max(1, min(p_avail, int(max_planes)))
    p_min = max(1, min(p_max, int(min_planes)))
    shared_sc = 1 if (qscale.ndim == 2 or (qscale.ndim == 3 and int(qscale.shape[0]) == 1 and p_max > 1)) else 0
    out_is_bf16 = 1 if y_out.dtype == torch.bfloat16 else 0

    grid_x = (N + 7) // 8
    used_planes_t = torch.empty((M, grid_x), dtype=torch.int32, device=dev) if return_telemetry else None
    defect_t = torch.empty((M, grid_x), dtype=torch.float32, device=dev) if return_telemetry else None

    p_x = ctypes.c_void_p(x_c.data_ptr())
    p_lo = ctypes.c_void_p(packed_lo.data_ptr())
    p_s = ctypes.c_void_p(qscale.data_ptr())
    p_z = ctypes.c_void_p(qzero.data_ptr())
    p_rs = ctypes.c_void_p(row_scale.data_ptr())
    p_cs = ctypes.c_void_p(col_scale.data_ptr())
    p_b = ctypes.c_void_p(0 if bias is None else bias.data_ptr())
    p_y = ctypes.c_void_p(y_out.data_ptr())
    p_up = ctypes.c_void_p(0 if used_planes_t is None else used_planes_t.data_ptr())
    p_df = ctypes.c_void_p(0 if defect_t is None else defect_t.data_ptr())

    c_M, c_N, c_K, c_gs = ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(group_size)
    c_pmax, c_pmin = ctypes.c_int(p_max), ctypes.c_int(p_min)
    c_tol = ctypes.c_float(float(mirror_tol))
    c_ob, c_ss = ctypes.c_int(out_is_bf16), ctypes.c_int(shared_sc)

    args = (ctypes.c_void_p * 19)(
        ctypes.addressof(p_x), ctypes.addressof(p_lo),
        ctypes.addressof(p_s), ctypes.addressof(p_z),
        ctypes.addressof(p_rs), ctypes.addressof(p_cs),
        ctypes.addressof(p_b), ctypes.addressof(p_y),
        ctypes.addressof(p_up), ctypes.addressof(p_df),
        ctypes.addressof(c_M), ctypes.addressof(c_N), ctypes.addressof(c_K), ctypes.addressof(c_gs),
        ctypes.addressof(c_pmax), ctypes.addressof(c_pmin),
        ctypes.addressof(c_tol), ctypes.addressof(c_ob), ctypes.addressof(c_ss),
    )

    chunk_cap = min(K, 4096)
    smem = (chunk_cap + chunk_cap // group_size + 16) * 4
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    rc = nvcuda.cuLaunchKernel(
        fns[b"mirror_equality_q1_gemv_f32"],
        grid_x, M, 1, 256, 1, 1, smem, stream, args, None,
    )
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel(mirror_equality_q1_gemv_f32) failed with CUresult={rc}")

    if return_telemetry:
        return used_planes_t, defect_t
    return None


def launch_mirror_equality_balance_step(
    outcome_prev: torch.Tensor,
    obs_prev: torch.Tensor,
    obs_next: torch.Tensor,
    outcome_next_in: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute the Mirror Equality conservation law directly on CUDA state batches `[B, D]`.

    Returns `(strategy_t, outcome_next, balance_defect)`:
    - If `outcome_next_in` is None (solve mode): solves `Outcome_(t+1) = Strategy_t - ObservedBehavior_(t+1)`
      with machine-zero balance defect (`~0`).
    - If `outcome_next_in` is provided (verify & project mode): computes the L2 conservation defect
      `||L - R||_2 / (||Strategy_t||_2 + eps)` and returns the mirror-balanced `Strategy_t = 0.5*(L + R)`
      together with its consistent `Outcome_(t+1) = Strategy_t - ObservedBehavior_(t+1)`.
    """
    dev = outcome_prev.device
    if dev.type != "cuda":
        raise ValueError("launch_mirror_equality_balance_step requires CUDA tensors")
    nvcuda, fns = get_mirror_kernels(dev)

    op = outcome_prev.float().contiguous()
    obp = obs_prev.float().contiguous()
    obn = obs_next.float().contiguous()
    oni = None if outcome_next_in is None else outcome_next_in.float().contiguous()

    if op.ndim == 1:
        op, obp, obn = op.unsqueeze(0), obp.unsqueeze(0), obn.unsqueeze(0)
        if oni is not None:
            oni = oni.unsqueeze(0)
        squeeze_out = True
    else:
        squeeze_out = False

    B, D = int(op.shape[0]), int(op.shape[1])
    strategy_out = torch.empty((B, D), dtype=torch.float32, device=dev)
    outcome_next_out = torch.empty((B, D), dtype=torch.float32, device=dev)
    defect_out = torch.empty((B,), dtype=torch.float32, device=dev)

    p_op = ctypes.c_void_p(op.data_ptr())
    p_obp = ctypes.c_void_p(obp.data_ptr())
    p_obn = ctypes.c_void_p(obn.data_ptr())
    p_oni = ctypes.c_void_p(0 if oni is None else oni.data_ptr())
    p_st = ctypes.c_void_p(strategy_out.data_ptr())
    p_ono = ctypes.c_void_p(outcome_next_out.data_ptr())
    p_df = ctypes.c_void_p(defect_out.data_ptr())
    c_B, c_D, c_mode = ctypes.c_int(B), ctypes.c_int(D), ctypes.c_int(0 if oni is None else 1)

    args = (ctypes.c_void_p * 10)(
        ctypes.addressof(p_op), ctypes.addressof(p_obp),
        ctypes.addressof(p_obn), ctypes.addressof(p_oni),
        ctypes.addressof(p_st), ctypes.addressof(p_ono), ctypes.addressof(p_df),
        ctypes.addressof(c_B), ctypes.addressof(c_D), ctypes.addressof(c_mode),
    )
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    rc = nvcuda.cuLaunchKernel(
        fns[b"mirror_equality_balance_step_f32"],
        B, 1, 1, 256, 1, 1, 0, stream, args, None,
    )
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel(mirror_equality_balance_step_f32) failed with CUresult={rc}")

    if squeeze_out:
        return strategy_out.squeeze(0), outcome_next_out.squeeze(0), defect_out.squeeze(0)
    return strategy_out, outcome_next_out, defect_out


def launch_mirror_uni_binding_q1_gemv(
    x_bf16: torch.Tensor,
    packed_lo: torch.Tensor,
    qscale: torch.Tensor,
    qzero: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor,
    y_out: torch.Tensor,
    *,
    group_size: int = 64,
    max_planes: int | None = None,
    min_planes: int = 3,
    base_tol: float = 0.12,
    bias: Optional[torch.Tensor] = None,
    return_telemetry: bool = False,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Launch `mirror_uni_binding_q1_gemv_f32` (UNI x Mirror Equality Recursive Binding Q1 GEMV).

    Returns `(used_planes, mirror_defect, bind_strength)` when `return_telemetry=True`.
    """
    dev = x_bf16.device
    if dev.type != "cuda":
        raise ValueError("launch_mirror_uni_binding_q1_gemv requires CUDA tensors")
    nvcuda, fns = get_mirror_kernels(dev)

    x_c = x_bf16.contiguous()
    M, K = int(x_c.shape[0]), int(x_c.shape[1])
    N = int(y_out.shape[1])
    if M == 0 or N == 0:
        return (None, None, None) if return_telemetry else None

    p_avail = int(packed_lo.shape[0])
    p_max = p_avail if max_planes is None else max(1, min(p_avail, int(max_planes)))
    p_min = max(1, min(p_max, int(min_planes)))
    shared_sc = 1 if (qscale.ndim == 2 or (qscale.ndim == 3 and int(qscale.shape[0]) == 1 and p_max > 1)) else 0
    out_is_bf16 = 1 if y_out.dtype == torch.bfloat16 else 0

    grid_x = (N + 7) // 8
    used_planes_t = torch.empty((M, grid_x), dtype=torch.int32, device=dev) if return_telemetry else None
    defect_t = torch.empty((M, grid_x), dtype=torch.float32, device=dev) if return_telemetry else None
    bind_t = torch.empty((M, grid_x), dtype=torch.float32, device=dev) if return_telemetry else None

    p_x = ctypes.c_void_p(x_c.data_ptr())
    p_lo = ctypes.c_void_p(packed_lo.data_ptr())
    p_s = ctypes.c_void_p(qscale.data_ptr())
    p_z = ctypes.c_void_p(qzero.data_ptr())
    p_rs = ctypes.c_void_p(row_scale.data_ptr())
    p_cs = ctypes.c_void_p(col_scale.data_ptr())
    p_b = ctypes.c_void_p(0 if bias is None else bias.data_ptr())
    p_y = ctypes.c_void_p(y_out.data_ptr())
    p_up = ctypes.c_void_p(0 if used_planes_t is None else used_planes_t.data_ptr())
    p_df = ctypes.c_void_p(0 if defect_t is None else defect_t.data_ptr())
    p_bs = ctypes.c_void_p(0 if bind_t is None else bind_t.data_ptr())

    c_M, c_N, c_K, c_gs = ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(group_size)
    c_pmax, c_pmin = ctypes.c_int(p_max), ctypes.c_int(p_min)
    c_tol = ctypes.c_float(float(base_tol))
    c_ob, c_ss = ctypes.c_int(out_is_bf16), ctypes.c_int(shared_sc)

    args = (ctypes.c_void_p * 20)(
        ctypes.addressof(p_x), ctypes.addressof(p_lo),
        ctypes.addressof(p_s), ctypes.addressof(p_z),
        ctypes.addressof(p_rs), ctypes.addressof(p_cs),
        ctypes.addressof(p_b), ctypes.addressof(p_y),
        ctypes.addressof(p_up), ctypes.addressof(p_df), ctypes.addressof(p_bs),
        ctypes.addressof(c_M), ctypes.addressof(c_N), ctypes.addressof(c_K), ctypes.addressof(c_gs),
        ctypes.addressof(c_pmax), ctypes.addressof(c_pmin),
        ctypes.addressof(c_tol), ctypes.addressof(c_ob), ctypes.addressof(c_ss),
    )

    chunk_cap = min(K, 4096)
    smem = (chunk_cap + chunk_cap // group_size + 32) * 4
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    rc = nvcuda.cuLaunchKernel(
        fns[b"mirror_uni_binding_q1_gemv_f32"],
        grid_x, M, 1, 256, 1, 1, smem, stream, args, None,
    )
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel(mirror_uni_binding_q1_gemv_f32) failed with CUresult={rc}")

    if return_telemetry:
        return used_planes_t, defect_t, bind_t
    return None


def uni_canonical_invariants(d_counts: Tuple[int, ...] = (3, 4, 2)) -> Dict[str, float]:
    """Return the exact closed-form UNI binding invariants (Phi, Omega, Lambda, Sigma, Delta)."""
    import math
    s_d = sum(d_counts)
    lam = 1.0
    for d in d_counts:
        lam *= (d / s_d) ** d
    return {
        "Phi": math.exp(-1.5),
        "Omega": math.pi * 0.5,
        "Lambda": lam,
        "Sigma": math.sqrt(math.pi),
        "Delta": 1.0 - math.cos(10.0),
    }


def launch_mirror_fractal_ring_q1_step(
    x_bf16: torch.Tensor,
    packed_lo: torch.Tensor,
    qscale: torch.Tensor,
    qzero: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor,
    line_spine: torch.Tensor,
    *,
    plane_idx: int,
    active_tiles: Optional[torch.Tensor] = None,
    include_zero: bool = False,
    group_size: int = 64,
    base_tol: float = 0.14,
    branch_weight: float = 1.0,
    bias: Optional[torch.Tensor] = None,
    last_obs_out: Optional[torch.Tensor] = None,
    spawn_next: bool = True,
) -> Tuple[Optional[torch.Tensor], int, torch.Tensor]:
    """Execute one freely oriented ring step on `line_spine` and auto-spawn new intersection nodes.

    Every executed tile reads the current point on `line_spine`, adds its Mirror step back onto
    `line_spine` (start and next step always intersect on one assembly line), and atomically
    appends tiles that still exceed the UNI x Mirror binding threshold into `spawn_tiles`.

    Returns `(spawn_tiles_1d, spawn_count, tile_defects)`.
    """
    dev = x_bf16.device
    if dev.type != "cuda":
        raise ValueError("launch_mirror_fractal_ring_q1_step requires CUDA tensors")
    nvcuda, fns = get_mirror_kernels(dev)

    x_c = x_bf16.contiguous()
    M, K = int(x_c.shape[0]), int(x_c.shape[1])
    N = int(line_spine.shape[1])
    tiles_total = (N + 7) // 8

    if active_tiles is not None:
        act_c = active_tiles.to(device=dev, dtype=torch.int32).contiguous()
        num_act = int(act_c.numel())
    else:
        act_c = None
        num_act = tiles_total

    tile_defects = torch.zeros((M, tiles_total), dtype=torch.float32, device=dev)
    if M == 0 or N == 0 or num_act == 0:
        empty_t = torch.empty((0,), dtype=torch.int32, device=dev) if spawn_next else None
        return empty_t, 0, tile_defects

    spawn_buf = torch.empty((tiles_total,), dtype=torch.int32, device=dev) if spawn_next else None
    spawn_cnt = torch.zeros((1,), dtype=torch.int32, device=dev) if spawn_next else None
    shared_sc = 1 if (qscale.ndim == 2 or (qscale.ndim == 3 and int(qscale.shape[0]) == 1 and int(packed_lo.shape[0]) > 1)) else 0

    p_x = ctypes.c_void_p(x_c.data_ptr())
    p_lo = ctypes.c_void_p(packed_lo.data_ptr())
    p_s = ctypes.c_void_p(qscale.data_ptr())
    p_z = ctypes.c_void_p(qzero.data_ptr())
    p_rs = ctypes.c_void_p(row_scale.data_ptr())
    p_cs = ctypes.c_void_p(col_scale.data_ptr())
    p_b = ctypes.c_void_p(0 if bias is None else bias.data_ptr())
    p_act = ctypes.c_void_p(0 if act_c is None else act_c.data_ptr())
    p_line = ctypes.c_void_p(line_spine.data_ptr())
    p_obs = ctypes.c_void_p(0 if last_obs_out is None else last_obs_out.data_ptr())
    p_sp = ctypes.c_void_p(0 if spawn_buf is None else spawn_buf.data_ptr())
    p_sc = ctypes.c_void_p(0 if spawn_cnt is None else spawn_cnt.data_ptr())
    p_df = ctypes.c_void_p(tile_defects.data_ptr())

    c_M, c_N, c_K, c_gs = ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(group_size)
    c_na, c_pi = ctypes.c_int(num_act), ctypes.c_int(int(plane_idx))
    c_iz, c_ss = ctypes.c_int(1 if include_zero else 0), ctypes.c_int(shared_sc)
    c_tol, c_bw = ctypes.c_float(float(base_tol)), ctypes.c_float(float(branch_weight))

    args = (ctypes.c_void_p * 23)(
        ctypes.addressof(p_x), ctypes.addressof(p_lo),
        ctypes.addressof(p_s), ctypes.addressof(p_z),
        ctypes.addressof(p_rs), ctypes.addressof(p_cs),
        ctypes.addressof(p_b), ctypes.addressof(p_act),
        ctypes.addressof(p_line), ctypes.addressof(p_obs),
        ctypes.addressof(p_sp), ctypes.addressof(p_sc), ctypes.addressof(p_df),
        ctypes.addressof(c_M), ctypes.addressof(c_N), ctypes.addressof(c_K), ctypes.addressof(c_gs),
        ctypes.addressof(c_na), ctypes.addressof(c_pi),
        ctypes.addressof(c_iz), ctypes.addressof(c_ss),
        ctypes.addressof(c_tol), ctypes.addressof(c_bw),
    )

    chunk_cap = min(K, 4096)
    smem = (chunk_cap + chunk_cap // group_size + 32) * 4
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    rc = nvcuda.cuLaunchKernel(
        fns[b"mirror_fractal_ring_q1_step_f32"],
        num_act, M, 1, 256, 1, 1, smem, stream, args, None,
    )
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel(mirror_fractal_ring_q1_step_f32) failed with CUresult={rc}")

    if not spawn_next:
        return None, 0, tile_defects
    cnt = int(spawn_cnt.item())
    return spawn_buf[:cnt], cnt, tile_defects


def run_autonomous_fractal_rings(
    x_bf16: torch.Tensor,
    base_trunk: Dict[str, torch.Tensor],
    organ_pool: Dict[str, Dict[str, torch.Tensor]],
    *,
    group_size: int = 64,
    max_planes: int = 8,
    min_planes: int = 3,
    base_tol: float = 0.14,
    wake_cos_threshold: float = 0.08,
    max_ring_steps: int = 64,
) -> Dict[str, object]:
    """Autonomous self-spawning fractal ring execution given ONLY the base seed and an organ pool.

    No fixed number of rings is prescribed. Starting from a single seed `(trunk, p=0)` on the
    shared assembly line (`line_spine`), EVERY executed ring point automatically:
      1. Intersects and updates `line_spine` via `mirror_fractal_ring_q1_step_f32` on GPU.
      2. Spawns its own deeper refinement ring `(op, p+1, spawn_tiles)` if GPU `atomicAdd`
         found tiles still above the Mirror x UNI binding gate.
      3. Tests resonance of the updated `line_spine` against dormant organs in `organ_pool`;
         any organ whose anchor resonates above `wake_cos_threshold` autonomously spawns a NEW
         intersecting ring rooted at this exact intermediate point on `line_spine`!
    Terminates naturally when the frontier of spawned intersections becomes empty.
    """
    dev = x_bf16.device
    N = int(base_trunk["row_scale"].shape[0])
    line_spine = torch.zeros((int(x_bf16.shape[0]), N), dtype=torch.float32, device=dev)
    last_obs = torch.zeros_like(line_spine)

    # All available operators keyed by name (base trunk + dormant organs)
    ops: Dict[str, Dict[str, torch.Tensor]] = {"trunk": base_trunk, **organ_pool}
    awakened = {"trunk"}
    # Priority queue (list popped by highest urgency / breadth):
    # each entry = (op_name, plane_idx, active_tiles_or_None, weight, parent_tag)
    frontier = [("trunk", 0, None, 1.0, "BASE_SEED")]
    events = []
    rings_spawned = {"trunk": 1}

    step_idx = 0
    while frontier and step_idx < max_ring_steps:
        op_name, p, act_tiles, weight, parent_tag = frontier.pop(0)
        op = ops[op_name]
        p_cap = int(op.get("max_planes", max_planes))
        p_floor = int(op.get("min_planes", min_planes))
        tol_here = base_tol if (p + 1 >= p_floor) else 0.0

        n_tiles_in = (N + 7) // 8 if act_tiles is None else int(act_tiles.numel())
        if n_tiles_in == 0:
            continue

        spawn_tiles, spawn_cnt, tile_defects = launch_mirror_fractal_ring_q1_step(
            x_bf16,
            op["packed_lo"],
            op["qscale"],
            op["qzero"],
            op["row_scale"],
            op["col_scale"],
            line_spine,
            plane_idx=p,
            active_tiles=act_tiles,
            include_zero=(p == 0),
            group_size=group_size,
            base_tol=tol_here,
            branch_weight=weight,
            bias=op.get("bias"),
            last_obs_out=last_obs,
            spawn_next=True,
        )
        step_idx += 1
        mean_def = float(tile_defects.mean().item())
        spawned_children = []

        # 1. Self-continuation along the same organ if GPU spawned unresolved tiles
        if spawn_cnt > 0 and (p + 1) < p_cap:
            frontier.append((op_name, p + 1, spawn_tiles, weight, f"{op_name}@p{p}"))
            spawned_children.append(f"{op_name}[p={p+1}:{spawn_cnt}t]")

        # 2. Autonomous cross-organ spawning from THIS intermediate ring point:
        #    both the assembled whole (`line_spine`) AND this ring's own step (`last_obs`)
        #    can ignite new intersecting rings!
        line_norm = float(line_spine.norm().item()) + 1e-8
        obs_norm = float(last_obs.norm().item()) + 1e-8
        for cand_name, cand in organ_pool.items():
            if cand_name in awakened:
                continue
            anchor = cand["anchor"]  # [N] f32 resonance vector on the shared line
            a_norm = float(anchor.norm().item()) + 1e-8
            res_line = float(torch.dot(line_spine[0], anchor).item()) / (line_norm * a_norm)
            res_obs = float(torch.dot(last_obs[0], anchor).item()) / (obs_norm * a_norm)
            res_score = res_line if abs(res_line) >= abs(res_obs) else res_obs
            if abs(res_score) >= wake_cos_threshold:
                awakened.add(cand_name)
                rings_spawned[cand_name] = 1
                cand_w = float(cand.get("weight", 0.5)) * (1.0 if res_score >= 0 else -1.0)
                # Newly woken ring starts at p=0 right from THIS point on `line_spine`
                hot_tiles = spawn_tiles if (spawn_cnt > 0 and cand.get("sparse_inherit", False)) else None
                src_wing = "line" if abs(res_line) >= abs(res_obs) else "obs"
                frontier.insert(0, (cand_name, 0, hot_tiles, abs(cand_w), f"SPAWNED@{op_name}:p{p}({src_wing}={res_score:+.3f})"))
                spawned_children.append(f"NEW_RING:{cand_name}({src_wing}={res_score:+.3f})")

        events.append({
            "step": step_idx,
            "op": op_name,
            "plane": p,
            "tiles": n_tiles_in,
            "spawned_tiles": spawn_cnt,
            "mean_defect": round(mean_def, 4),
            "parent": parent_tag,
            "children": spawned_children,
        })

    return {
        "line_spine": line_spine,
        "steps_executed": step_idx,
        "awakened_rings": sorted(awakened),
        "num_awakened_rings": len(awakened),
        "events": events,
    }



