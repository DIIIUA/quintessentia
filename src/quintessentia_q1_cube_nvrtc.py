"""Self-contained NVRTC + CUDA Driver API runtime for Q1 Bit-Plane Cube kernels.

Compiles Q1 CUDA kernels once via PyTorch's bundled NVRTC, caches PTX keyed by
source SHA-256 and SM capability under `data/native/ptx-cache/`, and dispatches
via `nvcuda.dll` / `libcuda.so.1` onto `torch.cuda.current_stream(device)`.

Kernels:
- `q1_cube_plane_gemv_f32`: exact BF16 activation bit-plane GEMV (__ffs skip).
- `q1_cube_popc_gemv_f32`: 8-bit sliced activation __popc GEMV.
- `q1_cube_multiplane_gemv_f32`: fused P-plane GEMV with optional z_eff(P).
- `q1_cube_popc_multiplane_gemv_f32`: fused P-plane __popc GEMV with z_eff(P).
- `q1_cube_adaptive_radar_gemv_f32`: 1-launch per-tile adaptive depth radar GEMV
  (skips all weight/meta GDDR6 reads when d == 0; applies z_eff(d) for 0 < d < K_max).
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import threading

import torch

from src.quintessentia_nvrtc import PTX_CACHE_DIR, _load_driver_and_nvrtc

_Q1_CUDA_SOURCE = r"""
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

/* Warp-cooperative Block FWHT H_bar = H / sqrt(GS) with zero multiplications inside butterflies */
__device__ __forceinline__ void block_fwht_and_sum(
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
            if (g < ng_cur) {
                s_x[g * 16 + sub] = v;
            }
            float s = v;
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            s += __shfl_xor_sync(0xffffffffu, s, 4);
            s += __shfl_xor_sync(0xffffffffu, s, 8);
            if (sub == 0 && g < ng_cur) {
                s_xsum[g] = s;
            }
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
            if (lane == 0) {
                s_xsum[g] = s;
            }
        }
    } else {
        /* GS == 64: two 32-element halves per warp + register d=32 stage */
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
            float a = (v0 + v1) * 0.125f;
            float b = (v0 - v1) * 0.125f;
            s_x[base + lane] = a;
            s_x[base + 32 + lane] = b;
            float s = a + b;
            #pragma unroll
            for (int d = 16; d > 0; d >>= 1) {
                s += __shfl_xor_sync(0xffffffffu, s, d);
            }
            if (lane == 0) {
                s_xsum[g] = s;
            }
        }
    }
}

/* 1. Single Q1 Bit-Plane GEMV (Exact BF16 activations, __ffs register skip) */
extern "C" __global__ void q1_cube_plane_gemv_f32(
    const unsigned short* __restrict__ x_bf16,
    const unsigned int* __restrict__ bits_u32,
    const unsigned short* __restrict__ qscale,
    const unsigned short* __restrict__ qzero,
    float* __restrict__ y_acc,
    int M, int N_full, int K_full,
    int r0, int r_len, int c0, int c_actual, int c_words, int GS,
    float alpha, int add_zero
) {
    extern __shared__ char smem_raw[];
    int chunk_w_cap = (c_words < 128) ? c_words : 128;
    int cap_pad = chunk_w_cap * 32;
    float* s_x = reinterpret_cast<float*>(smem_raw);
    float* s_xsum = s_x + cap_pad;

    int tid = threadIdx.x;
    int m = blockIdx.y;
    if (m >= M) return;

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int local_r = blockIdx.x * warps_per_block + warp_id;

    const unsigned short* x_row = x_bf16 + (long long)m * K_full + c0;
    const unsigned int* w_row = (local_r < r_len) ? (bits_u32 + (long long)local_r * c_words) : nullptr;
    const unsigned short* s_row = (local_r < r_len) ? (qscale + (long long)local_r * (c_words * 32 / GS)) : nullptr;
    const unsigned short* z_row = (local_r < r_len) ? (qzero  + (long long)local_r * (c_words * 32 / GS)) : nullptr;

    float acc = 0.0f;
    for (int w_start = 0; w_start < c_words; w_start += chunk_w_cap) {
        int cw_cur = c_words - w_start;
        if (cw_cur > chunk_w_cap) cw_cur = chunk_w_cap;
        int cpad_cur = cw_cur * 32;
        int ng_cur = cpad_cur / GS;
        int c_base_col = w_start * 32;
        int g_start = c_base_col / GS;

        for (int c = tid; c < cpad_cur; c += blockDim.x) {
            int col_t = c_base_col + c;
            s_x[c] = (col_t < c_actual) ? bf16_to_f32(x_row[col_t]) : 0.0f;
        }
        __syncthreads();

        if (add_zero) {
            for (int g = tid; g < ng_cur; g += blockDim.x) {
                float sum = 0.0f;
                int base = g * GS;
                for (int j = 0; j < GS; ++j) sum += s_x[base + j];
                s_xsum[g] = sum;
            }
            __syncthreads();
        }

        if (local_r < r_len) {
            for (int lw = lane; lw < cw_cur; lw += 32) {
                int w_idx = w_start + lw;
                unsigned int w = w_row[w_idx];
                if (w != 0u) {
                    int col_local = lw * 32;
                    if (GS >= 32) {
                        int g = g_start + (col_local / GS);
                        float sv = bf16_to_f32(s_row[g]);
                        float dot = 0.0f;
                        while (w) {
                            int b = __ffs(w) - 1;
                            dot += s_x[col_local + b];
                            w &= (w - 1u);
                        }
                        acc += alpha * sv * dot;
                    } else {
                        int g0 = g_start + (col_local >> 4);
                        unsigned int lo16 = w & 0xFFFFu;
                        unsigned int hi16 = w >> 16;
                        float dot0 = 0.0f;
                        while (lo16) {
                            int b = __ffs(lo16) - 1;
                            dot0 += s_x[col_local + b];
                            lo16 &= (lo16 - 1u);
                        }
                        float dot1 = 0.0f;
                        while (hi16) {
                            int b = __ffs(hi16) - 1;
                            dot1 += s_x[col_local + 16 + b];
                            hi16 &= (hi16 - 1u);
                        }
                        acc += alpha * (bf16_to_f32(s_row[g0]) * dot0 + bf16_to_f32(s_row[g0 + 1]) * dot1);
                    }
                }
            }

            if (add_zero) {
                for (int g = lane; g < ng_cur; g += 32) {
                    acc += bf16_to_f32(z_row[g_start + g]) * s_xsum[g];
                }
            }
        }
        if (w_start + chunk_w_cap < c_words) {
            __syncthreads();
        }
    }

    if (local_r >= r_len) return;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (lane == 0) {
        y_acc[(long long)m * N_full + r0 + local_r] += acc;
    }
}

/* 2. True __popc Bit-Sliced Q1 GEMV (8-bit sliced activations in shared mem) */
extern "C" __global__ void q1_cube_popc_gemv_f32(
    const unsigned short* __restrict__ x_bf16,
    const unsigned int* __restrict__ bits_u32,
    const unsigned short* __restrict__ qscale,
    const unsigned short* __restrict__ qzero,
    float* __restrict__ y_acc,
    int M, int N_full, int K_full,
    int r0, int r_len, int c0, int c_actual, int c_words, int GS,
    float alpha, int add_zero
) {
    extern __shared__ char smem_raw[];
    int chunk_w_cap = (c_words < 64) ? c_words : 64;
    int cap_pad = chunk_w_cap * 32;
    int cap_groups = cap_pad / GS;
    float* s_x = reinterpret_cast<float*>(smem_raw);
    float* s_xsum = s_x + cap_pad;
    float* s_xzero = s_xsum + cap_groups;
    float* s_xscale = s_xzero + cap_groups;
    unsigned int* s_xplanes = reinterpret_cast<unsigned int*>(s_xscale + cap_groups);
    unsigned char* s_qx = reinterpret_cast<unsigned char*>(s_xplanes + 8 * chunk_w_cap);

    int tid = threadIdx.x;
    int m = blockIdx.y;
    if (m >= M) return;

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int local_r = blockIdx.x * warps_per_block + warp_id;

    const unsigned short* x_row = x_bf16 + (long long)m * K_full + c0;
    const unsigned int* w_row = (local_r < r_len) ? (bits_u32 + (long long)local_r * c_words) : nullptr;
    const unsigned short* s_row = (local_r < r_len) ? (qscale + (long long)local_r * (c_words * 32 / GS)) : nullptr;
    const unsigned short* z_row = (local_r < r_len) ? (qzero  + (long long)local_r * (c_words * 32 / GS)) : nullptr;

    float acc = 0.0f;
    for (int w_start = 0; w_start < c_words; w_start += chunk_w_cap) {
        int cw_cur = c_words - w_start;
        if (cw_cur > chunk_w_cap) cw_cur = chunk_w_cap;
        int cpad_cur = cw_cur * 32;
        int ng_cur = cpad_cur / GS;
        int c_base_col = w_start * 32;
        int g_start = c_base_col / GS;

        for (int c = tid; c < cpad_cur; c += blockDim.x) {
            int col_t = c_base_col + c;
            s_x[c] = (col_t < c_actual) ? bf16_to_f32(x_row[col_t]) : 0.0f;
        }
        __syncthreads();

        for (int g = tid; g < ng_cur; g += blockDim.x) {
            int base = g * GS;
            float mn = s_x[base];
            float mx = mn;
            float sum = 0.0f;
            for (int j = 0; j < GS; ++j) {
                float v = s_x[base + j];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
                sum += v;
            }
            float sc = (mx - mn) * (1.0f / 255.0f);
            if (sc < 1e-8f) sc = 1e-8f;
            s_xsum[g] = sum;
            s_xzero[g] = mn;
            s_xscale[g] = sc;
            float inv_sc = 1.0f / sc;
            for (int j = 0; j < GS; ++j) {
                int qv = __float2int_rn((s_x[base + j] - mn) * inv_sc);
                if (qv < 0) qv = 0;
                if (qv > 255) qv = 255;
                s_qx[base + j] = (unsigned char)qv;
            }
        }
        __syncthreads();

        for (int idx = tid; idx < 8 * cw_cur; idx += blockDim.x) {
            int a = idx / cw_cur;
            int lw = idx - a * cw_cur;
            int base = lw * 32;
            unsigned int word = 0u;
            #pragma unroll
            for (int b = 0; b < 32; ++b) {
                word |= ((unsigned int)((s_qx[base + b] >> a) & 1u)) << b;
            }
            s_xplanes[idx] = word;
        }
        __syncthreads();

        if (local_r < r_len) {
            for (int lw = lane; lw < cw_cur; lw += 32) {
                int w_idx = w_start + lw;
                unsigned int w = w_row[w_idx];
                if (w != 0u) {
                    int col_local = lw * 32;
                    if (GS >= 32) {
                        int g_local = col_local / GS;
                        int g_glob = g_start + g_local;
                        float sv = bf16_to_f32(s_row[g_glob]);
                        float xz = s_xzero[g_local];
                        float xs = s_xscale[g_local];
                        int pop_w = __popc(w);
                        int bit_acc = 0;
                        #pragma unroll
                        for (int a = 0; a < 8; ++a) {
                            bit_acc += __popc(w & s_xplanes[a * cw_cur + lw]) << a;
                        }
                        acc += alpha * sv * (xz * (float)pop_w + xs * (float)bit_acc);
                    } else {
                        int g0_local = col_local >> 4;
                        int g0_glob = g_start + g0_local;
                        unsigned int w0 = w & 0xFFFFu;
                        unsigned int w1 = w >> 16;
                        int pop0 = __popc(w0);
                        int pop1 = __popc(w1);
                        int ba0 = 0, ba1 = 0;
                        #pragma unroll
                        for (int a = 0; a < 8; ++a) {
                            unsigned int xp = s_xplanes[a * cw_cur + lw];
                            ba0 += __popc(w0 & xp) << a;
                            ba1 += __popc(w1 & (xp >> 16)) << a;
                        }
                        float dot0 = s_xzero[g0_local] * (float)pop0 + s_xscale[g0_local] * (float)ba0;
                        float dot1 = s_xzero[g0_local + 1] * (float)pop1 + s_xscale[g0_local + 1] * (float)ba1;
                        acc += alpha * (bf16_to_f32(s_row[g0_glob]) * dot0 + bf16_to_f32(s_row[g0_glob + 1]) * dot1);
                    }
                }
            }

            if (add_zero) {
                for (int g = lane; g < ng_cur; g += 32) {
                    acc += bf16_to_f32(z_row[g_start + g]) * s_xsum[g];
                }
            }
        }
        if (w_start + chunk_w_cap < c_words) {
            __syncthreads();
        }
    }

    if (local_r >= r_len) return;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (lane == 0) {
        y_acc[(long long)m * N_full + r0 + local_r] += acc;
    }
}

/* 3. Fused Multi-Plane Q1 Tile GEMV with optional z_eff(P) midpoint correction */
extern "C" __global__ void q1_cube_multiplane_gemv_f32(
    const unsigned short* __restrict__ x_bf16,
    const unsigned int* __restrict__ planes_u32,
    const unsigned short* __restrict__ qscale,
    const unsigned short* __restrict__ qzero,
    float* __restrict__ y_acc,
    int M, int N_full, int K_full,
    int r0, int r_len, int c0, int c_actual, int c_words, int GS,
    int P_active, int total_planes, int add_zero, int midpoint_correct
) {
    extern __shared__ char smem_raw[];
    int chunk_w_cap = (c_words < 128) ? c_words : 128;
    int cap_pad = chunk_w_cap * 32;
    float* s_x = reinterpret_cast<float*>(smem_raw);
    float* s_xsum = s_x + cap_pad;

    int tid = threadIdx.x;
    int m = blockIdx.y;
    if (m >= M) return;

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int local_r = blockIdx.x * warps_per_block + warp_id;

    const unsigned short* x_row = x_bf16 + (long long)m * K_full + c0;
    long long plane_stride = (long long)r_len * c_words;
    int num_groups_full = (c_words * 32) / GS;
    const unsigned short* s_row = (local_r < r_len) ? (qscale + (long long)local_r * num_groups_full) : nullptr;
    const unsigned short* z_row = (local_r < r_len) ? (qzero  + (long long)local_r * num_groups_full) : nullptr;
    float tail_mid = (add_zero && midpoint_correct && P_active < total_planes)
        ? 0.5f * (float)((1u << (total_planes - P_active)) - 1u)
        : 0.0f;

    float acc = 0.0f;
    for (int w_start = 0; w_start < c_words; w_start += chunk_w_cap) {
        int cw_cur = c_words - w_start;
        if (cw_cur > chunk_w_cap) cw_cur = chunk_w_cap;
        int cpad_cur = cw_cur * 32;
        int ng_cur = cpad_cur / GS;
        int c_base_col = w_start * 32;
        int g_start = c_base_col / GS;

        for (int c = tid; c < cpad_cur; c += blockDim.x) {
            int col_t = c_base_col + c;
            s_x[c] = (col_t < c_actual) ? bf16_to_f32(x_row[col_t]) : 0.0f;
        }
        __syncthreads();

        if (add_zero) {
            for (int g = tid; g < ng_cur; g += blockDim.x) {
                float sum = 0.0f;
                int base = g * GS;
                for (int j = 0; j < GS; ++j) sum += s_x[base + j];
                s_xsum[g] = sum;
            }
            __syncthreads();
        }

        if (local_r < r_len) {
            for (int lw = lane; lw < cw_cur; lw += 32) {
                int w_idx = w_start + lw;
                int col_local = lw * 32;
                if (GS >= 32) {
                    int g = g_start + (col_local / GS);
                    float sv = bf16_to_f32(s_row[g]);
                    float wdot = 0.0f;
                    for (int p = 0; p < P_active; ++p) {
                        unsigned int w = planes_u32[p * plane_stride + (long long)local_r * c_words + w_idx];
                        if (w != 0u) {
                            float alpha = (float)(1u << (total_planes - 1 - p));
                            float p_dot = 0.0f;
                            while (w) {
                                int b = __ffs(w) - 1;
                                p_dot += s_x[col_local + b];
                                w &= (w - 1u);
                            }
                            wdot += alpha * p_dot;
                        }
                    }
                    acc += sv * wdot;
                } else {
                    int g0 = g_start + (col_local >> 4);
                    float sv0 = bf16_to_f32(s_row[g0]);
                    float sv1 = bf16_to_f32(s_row[g0 + 1]);
                    float wdot0 = 0.0f, wdot1 = 0.0f;
                    for (int p = 0; p < P_active; ++p) {
                        unsigned int w = planes_u32[p * plane_stride + (long long)local_r * c_words + w_idx];
                        if (w != 0u) {
                            float alpha = (float)(1u << (total_planes - 1 - p));
                            unsigned int lo16 = w & 0xFFFFu;
                            unsigned int hi16 = w >> 16;
                            float d0 = 0.0f, d1 = 0.0f;
                            while (lo16) { int b = __ffs(lo16) - 1; d0 += s_x[col_local + b]; lo16 &= (lo16 - 1u); }
                            while (hi16) { int b = __ffs(hi16) - 1; d1 += s_x[col_local + 16 + b]; hi16 &= (hi16 - 1u); }
                            wdot0 += alpha * d0;
                            wdot1 += alpha * d1;
                        }
                    }
                    acc += sv0 * wdot0 + sv1 * wdot1;
                }
            }

            if (add_zero) {
                for (int g = lane; g < ng_cur; g += 32) {
                    int gg = g_start + g;
                    float z_eff = bf16_to_f32(z_row[gg]) + bf16_to_f32(s_row[gg]) * tail_mid;
                    acc += z_eff * s_xsum[g];
                }
            }
        }
        if (w_start + chunk_w_cap < c_words) {
            __syncthreads();
        }
    }

    if (local_r >= r_len) return;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (lane == 0) {
        y_acc[(long long)m * N_full + r0 + local_r] += acc;
    }
}

/* 4. Fused Multi-Plane __popc Q1 Tile GEMV with optional z_eff(P) */
extern "C" __global__ void q1_cube_popc_multiplane_gemv_f32(
    const unsigned short* __restrict__ x_bf16,
    const unsigned int* __restrict__ planes_u32,
    const unsigned short* __restrict__ qscale,
    const unsigned short* __restrict__ qzero,
    float* __restrict__ y_acc,
    int M, int N_full, int K_full,
    int r0, int r_len, int c0, int c_actual, int c_words, int GS,
    int P_active, int total_planes, int add_zero, int midpoint_correct
) {
    extern __shared__ char smem_raw[];
    int chunk_w_cap = (c_words < 64) ? c_words : 64;
    int cap_pad = chunk_w_cap * 32;
    int cap_groups = cap_pad / GS;
    float* s_x = reinterpret_cast<float*>(smem_raw);
    float* s_xsum = s_x + cap_pad;
    float* s_xzero = s_xsum + cap_groups;
    float* s_xscale = s_xzero + cap_groups;
    unsigned int* s_xplanes = reinterpret_cast<unsigned int*>(s_xscale + cap_groups);
    unsigned char* s_qx = reinterpret_cast<unsigned char*>(s_xplanes + 8 * chunk_w_cap);

    int tid = threadIdx.x;
    int m = blockIdx.y;
    if (m >= M) return;

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int local_r = blockIdx.x * warps_per_block + warp_id;

    const unsigned short* x_row = x_bf16 + (long long)m * K_full + c0;
    long long plane_stride = (long long)r_len * c_words;
    int num_groups_full = (c_words * 32) / GS;
    const unsigned short* s_row = (local_r < r_len) ? (qscale + (long long)local_r * num_groups_full) : nullptr;
    const unsigned short* z_row = (local_r < r_len) ? (qzero  + (long long)local_r * num_groups_full) : nullptr;
    float tail_mid = (add_zero && midpoint_correct && P_active < total_planes)
        ? 0.5f * (float)((1u << (total_planes - P_active)) - 1u)
        : 0.0f;

    float acc = 0.0f;
    for (int w_start = 0; w_start < c_words; w_start += chunk_w_cap) {
        int cw_cur = c_words - w_start;
        if (cw_cur > chunk_w_cap) cw_cur = chunk_w_cap;
        int cpad_cur = cw_cur * 32;
        int ng_cur = cpad_cur / GS;
        int c_base_col = w_start * 32;
        int g_start = c_base_col / GS;

        for (int c = tid; c < cpad_cur; c += blockDim.x) {
            int col_t = c_base_col + c;
            s_x[c] = (col_t < c_actual) ? bf16_to_f32(x_row[col_t]) : 0.0f;
        }
        __syncthreads();

        for (int g = tid; g < ng_cur; g += blockDim.x) {
            int base = g * GS;
            float mn = s_x[base];
            float mx = mn;
            float sum = 0.0f;
            for (int j = 0; j < GS; ++j) {
                float v = s_x[base + j];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
                sum += v;
            }
            float sc = (mx - mn) * (1.0f / 255.0f);
            if (sc < 1e-8f) sc = 1e-8f;
            s_xsum[g] = sum;
            s_xzero[g] = mn;
            s_xscale[g] = sc;
            float inv_sc = 1.0f / sc;
            for (int j = 0; j < GS; ++j) {
                int qv = __float2int_rn((s_x[base + j] - mn) * inv_sc);
                if (qv < 0) qv = 0;
                if (qv > 255) qv = 255;
                s_qx[base + j] = (unsigned char)qv;
            }
        }
        __syncthreads();

        for (int idx = tid; idx < 8 * cw_cur; idx += blockDim.x) {
            int a = idx / cw_cur;
            int lw = idx - a * cw_cur;
            int base = lw * 32;
            unsigned int word = 0u;
            #pragma unroll
            for (int b = 0; b < 32; ++b) {
                word |= ((unsigned int)((s_qx[base + b] >> a) & 1u)) << b;
            }
            s_xplanes[idx] = word;
        }
        __syncthreads();

        if (local_r < r_len) {
            for (int lw = lane; lw < cw_cur; lw += 32) {
                int w_idx = w_start + lw;
                int col_local = lw * 32;
                if (GS >= 32) {
                    int g_local = col_local / GS;
                    int g_glob = g_start + g_local;
                    float sv = bf16_to_f32(s_row[g_glob]);
                    float xz = s_xzero[g_local];
                    float xs = s_xscale[g_local];
                    unsigned int xp0 = s_xplanes[0 * cw_cur + lw];
                    unsigned int xp1 = s_xplanes[1 * cw_cur + lw];
                    unsigned int xp2 = s_xplanes[2 * cw_cur + lw];
                    unsigned int xp3 = s_xplanes[3 * cw_cur + lw];
                    unsigned int xp4 = s_xplanes[4 * cw_cur + lw];
                    unsigned int xp5 = s_xplanes[5 * cw_cur + lw];
                    unsigned int xp6 = s_xplanes[6 * cw_cur + lw];
                    unsigned int xp7 = s_xplanes[7 * cw_cur + lw];
                    float wdot = 0.0f;
                    for (int p = 0; p < P_active; ++p) {
                        unsigned int w = planes_u32[p * plane_stride + (long long)local_r * c_words + w_idx];
                        if (w != 0u) {
                            float alpha = (float)(1u << (total_planes - 1 - p));
                            int pop_w = __popc(w);
                            int bit_acc = (__popc(w & xp0))
                                        + (__popc(w & xp1) << 1)
                                        + (__popc(w & xp2) << 2)
                                        + (__popc(w & xp3) << 3)
                                        + (__popc(w & xp4) << 4)
                                        + (__popc(w & xp5) << 5)
                                        + (__popc(w & xp6) << 6)
                                        + (__popc(w & xp7) << 7);
                            wdot += alpha * (xz * (float)pop_w + xs * (float)bit_acc);
                        }
                    }
                    acc += sv * wdot;
                } else {
                    int g0_local = col_local >> 4;
                    int g0_glob = g_start + g0_local;
                    float sv0 = bf16_to_f32(s_row[g0_glob]);
                    float sv1 = bf16_to_f32(s_row[g0_glob + 1]);
                    float xz0 = s_xzero[g0_local], xs0 = s_xscale[g0_local];
                    float xz1 = s_xzero[g0_local + 1], xs1 = s_xscale[g0_local + 1];
                    float wdot0 = 0.0f, wdot1 = 0.0f;
                    for (int p = 0; p < P_active; ++p) {
                        unsigned int w = planes_u32[p * plane_stride + (long long)local_r * c_words + w_idx];
                        if (w != 0u) {
                            float alpha = (float)(1u << (total_planes - 1 - p));
                            unsigned int w0 = w & 0xFFFFu;
                            unsigned int w1 = w >> 16;
                            int ba0 = 0, ba1 = 0;
                            #pragma unroll
                            for (int a = 0; a < 8; ++a) {
                                unsigned int xp = s_xplanes[a * cw_cur + lw];
                                ba0 += __popc(w0 & xp) << a;
                                ba1 += __popc(w1 & (xp >> 16)) << a;
                            }
                            wdot0 += alpha * (xz0 * (float)__popc(w0) + xs0 * (float)ba0);
                            wdot1 += alpha * (xz1 * (float)__popc(w1) + xs1 * (float)ba1);
                        }
                    }
                    acc += sv0 * wdot0 + sv1 * wdot1;
                }
            }

            if (add_zero) {
                for (int g = lane; g < ng_cur; g += 32) {
                    int gg = g_start + g;
                    float z_eff = bf16_to_f32(z_row[gg]) + bf16_to_f32(s_row[gg]) * tail_mid;
                    acc += z_eff * s_xsum[g];
                }
            }
        }
        if (w_start + chunk_w_cap < c_words) {
            __syncthreads();
        }
    }

    if (local_r >= r_len) return;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (lane == 0) {
        y_acc[(long long)m * N_full + r0 + local_r] += acc;
    }
}

/* 5. 1-Launch Adaptive Depth Radar Q1 GEMV (skips d == 0, z_eff(d) for 0 < d < K_max) */
extern "C" __global__ void q1_cube_adaptive_radar_gemv_f32(
    const unsigned short* __restrict__ x_bf16,
    const unsigned int* __restrict__ planes_u32,
    const unsigned short* __restrict__ qscale,
    const unsigned short* __restrict__ qzero,
    const int* __restrict__ tile_depths,
    unsigned short* __restrict__ y_bf16,
    int M, int N, int K,
    int R_tile, int C_tile, int GS, int K_max
) {
    extern __shared__ char smem_raw[];
    int n_ctiles = K / C_tile;
    int tiles_per_chunk = 4096 / C_tile;
    if (tiles_per_chunk < 1) tiles_per_chunk = 1;
    if (tiles_per_chunk > n_ctiles) tiles_per_chunk = n_ctiles;
    int cap_cols = tiles_per_chunk * C_tile;
    float* s_x = reinterpret_cast<float*>(smem_raw);
    float* s_xsum = s_x + cap_cols;

    int tid = threadIdx.x;
    int m = blockIdx.y;
    if (m >= M) return;

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp_id;

    int tr = (row < N) ? (row / R_tile) : 0;
    int local_r = (row < N) ? (row - tr * R_tile) : 0;
    int c_words_tile = C_tile >> 5;
    int groups_tile = C_tile / GS;

    long long plane_elems = (long long)R_tile * c_words_tile;
    long long ctile_plane_stride = (long long)K_max * plane_elems;
    long long ctile_meta_stride = (long long)R_tile * groups_tile;

    const unsigned short* x_row = x_bf16 + (long long)m * K;
    float acc = 0.0f;

    for (int tc_start = 0; tc_start < n_ctiles; tc_start += tiles_per_chunk) {
        int tc_end = tc_start + tiles_per_chunk;
        if (tc_end > n_ctiles) tc_end = n_ctiles;
        int c_chunk_start = tc_start * C_tile;
        int chunk_cols = (tc_end - tc_start) * C_tile;
        int chunk_groups = chunk_cols / GS;

        for (int c = tid; c < chunk_cols; c += blockDim.x) {
            s_x[c] = bf16_to_f32(x_row[c_chunk_start + c]);
        }
        __syncthreads();

        for (int g = tid; g < chunk_groups; g += blockDim.x) {
            float sum = 0.0f;
            int base = g * GS;
            for (int j = 0; j < GS; ++j) sum += s_x[base + j];
            s_xsum[g] = sum;
        }
        __syncthreads();

        if (row < N) {
            for (int tc = tc_start; tc < tc_end; ++tc) {
                int tile_idx = tr * n_ctiles + tc;
                int d = tile_depths[tile_idx];
                if (d <= 0) continue;
                if (d > K_max) d = K_max;

                int c_local = (tc - tc_start) * C_tile;
                int g0_local = c_local / GS;
                const unsigned int* t_planes = planes_u32 + (long long)tile_idx * ctile_plane_stride;
                const unsigned short* s_row = qscale + (long long)tile_idx * ctile_meta_stride + (long long)local_r * groups_tile;
                const unsigned short* z_row = qzero  + (long long)tile_idx * ctile_meta_stride + (long long)local_r * groups_tile;

                for (int w_idx = lane; w_idx < c_words_tile; w_idx += 32) {
                    int col_base = c_local + w_idx * 32;
                    if (GS >= 32) {
                        int g_tile = (w_idx * 32) / GS;
                        float sv = bf16_to_f32(s_row[g_tile]);
                        float wdot = 0.0f;
                        for (int p = 0; p < d; ++p) {
                            unsigned int w = t_planes[(long long)p * plane_elems + (long long)local_r * c_words_tile + w_idx];
                            if (w != 0u) {
                                float alpha = (float)(1u << (K_max - 1 - p));
                                float pd = 0.0f;
                                while (w) {
                                    int b = __ffs(w) - 1;
                                    pd += s_x[col_base + b];
                                    w &= (w - 1u);
                                }
                                wdot += alpha * pd;
                            }
                        }
                        acc += sv * wdot;
                    } else {
                        int g0_tile = (w_idx * 32) >> 4;
                        float sv0 = bf16_to_f32(s_row[g0_tile]);
                        float sv1 = bf16_to_f32(s_row[g0_tile + 1]);
                        float wdot0 = 0.0f, wdot1 = 0.0f;
                        for (int p = 0; p < d; ++p) {
                            unsigned int w = t_planes[(long long)p * plane_elems + (long long)local_r * c_words_tile + w_idx];
                            if (w != 0u) {
                                float alpha = (float)(1u << (K_max - 1 - p));
                                unsigned int lo16 = w & 0xFFFFu;
                                unsigned int hi16 = w >> 16;
                                float d0 = 0.0f, d1 = 0.0f;
                                while (lo16) { int b = __ffs(lo16) - 1; d0 += s_x[col_base + b]; lo16 &= (lo16 - 1u); }
                                while (hi16) { int b = __ffs(hi16) - 1; d1 += s_x[col_base + 16 + b]; hi16 &= (hi16 - 1u); }
                                wdot0 += alpha * d0;
                                wdot1 += alpha * d1;
                            }
                        }
                        acc += sv0 * wdot0 + sv1 * wdot1;
                    }
                }

                float tail_mid = (d < K_max) ? 0.5f * (float)((1u << (K_max - d)) - 1u) : 0.0f;
                for (int gl = lane; gl < groups_tile; gl += 32) {
                    float z_eff = bf16_to_f32(z_row[gl]) + bf16_to_f32(s_row[gl]) * tail_mid;
                    acc += z_eff * s_xsum[g0_local + gl];
                }
            }
        }
        if (tc_end < n_ctiles) {
            __syncthreads();
        }
    }

    if (row >= N) return;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (lane == 0) {
        y_bf16[(long long)m * N + row] = f32_to_bf16_rne(acc);
    }
}

/* 6. Zero-Extra-VRAM Direct Packed Q4/Q6 MSB-Truncated GEMV with z_eff(P_active).
 * For bits == 6 and P_active <= 2, reads ONLY packed_hi (N*K/4 bytes -> 3x less DRAM traffic). */
extern "C" __global__ void q1_packed_trunc_gemv_f32(
    const unsigned short* __restrict__ x_bf16,
    const unsigned char* __restrict__ packed_lo,
    const unsigned char* __restrict__ packed_hi,
    const unsigned short* __restrict__ qscale,
    const unsigned short* __restrict__ qzero,
    const unsigned short* __restrict__ row_scale,
    const unsigned short* __restrict__ col_scale,
    const unsigned short* __restrict__ bias_bf16,
    void* __restrict__ y_out,
    int M, int N, int K, int GS,
    int bits, int P_active, int midpoint_correct, int out_is_bf16
) {
    extern __shared__ char smem_raw[];
    int chunk_cap = (K < 4096) ? K : 4096;
    float* s_x = reinterpret_cast<float*>(smem_raw);
    float* s_xsum = s_x + chunk_cap;

    int tid = threadIdx.x;
    int m = blockIdx.y;
    if (m >= M) return;

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp_id;

    const unsigned short* x_row = x_bf16 + (long long)m * K;
    int half_k = K >> 1;
    int quarter_k = K >> 2;
    int num_groups_full = K / GS;
    const unsigned char* lo_row = (row < N && packed_lo != nullptr) ? (packed_lo + (long long)row * half_k) : nullptr;
    const unsigned char* hi_row = (row < N && packed_hi != nullptr) ? (packed_hi + (long long)row * quarter_k) : nullptr;
    const unsigned short* s_row = (row < N) ? (qscale + (long long)row * num_groups_full) : nullptr;
    const unsigned short* z_row = (row < N) ? (qzero  + (long long)row * num_groups_full) : nullptr;

    int shift = bits - P_active;
    unsigned int mask = ~((1u << shift) - 1u);
    float tail_mid = (midpoint_correct && shift > 0)
        ? 0.5f * (float)((1u << shift) - 1u)
        : 0.0f;
    int hi_only = (bits == 6 && P_active <= 2);
    int use_fwht = (bits == 2 && col_scale != nullptr && row_scale != nullptr);

    float acc = 0.0f;
    for (int k_start = 0; k_start < K; k_start += chunk_cap) {
        int c_cur = K - k_start;
        if (c_cur > chunk_cap) c_cur = chunk_cap;
        int ng_cur = c_cur / GS;
        int g_start = k_start / GS;

        for (int c = tid; c < c_cur; c += blockDim.x) {
            int ka = k_start + c;
            float xv = bf16_to_f32(x_row[ka]);
            s_x[c] = use_fwht ? (xv * bf16_to_f32(col_scale[ka])) : xv;
        }
        __syncthreads();

        if (use_fwht) {
            block_fwht_and_sum(s_x, s_xsum, ng_cur, GS, warp_id, lane, warps_per_block);
        } else {
            for (int g = tid; g < ng_cur; g += blockDim.x) {
                float sum = 0.0f;
                int base = g * GS;
                for (int j = 0; j < GS; ++j) sum += s_x[base + j];
                s_xsum[g] = sum;
            }
        }
        __syncthreads();

        if (row < N) {
            if (hi_only) {
                const unsigned int* hi_u32 = reinterpret_cast<const unsigned int*>(hi_row + (k_start >> 2));
                unsigned int mask2 = (P_active == 1) ? 2u : 3u;
                float scale_shift = 16.0f;
                for (int c_off = lane * 16; c_off < c_cur; c_off += 512) {
                    unsigned int hw = hi_u32[c_off >> 4];
                    float sv = bf16_to_f32(s_row[g_start + (c_off / GS)]) * scale_shift;
                    float dot16 = 0.0f;
                    #pragma unroll
                    for (int t = 0; t < 16; ++t) {
                        unsigned int qv = (hw >> (t * 2)) & mask2;
                        dot16 += (float)qv * s_x[c_off + t];
                    }
                    acc += sv * dot16;
                }
            } else if (bits == 8) {
                const unsigned char* lo8_row = packed_lo + (long long)row * K;
                for (int c_off = lane * 4; c_off < c_cur; c_off += 128) {
                    int k_abs = k_start + c_off;
                    unsigned int q0 = (unsigned int)lo8_row[k_abs + 0] & mask;
                    unsigned int q1 = (unsigned int)lo8_row[k_abs + 1] & mask;
                    unsigned int q2 = (unsigned int)lo8_row[k_abs + 2] & mask;
                    unsigned int q3 = (unsigned int)lo8_row[k_abs + 3] & mask;
                    float sv = bf16_to_f32(s_row[g_start + (c_off / GS)]);
                    float dot4 = (float)q0 * s_x[c_off]
                               + (float)q1 * s_x[c_off + 1]
                               + (float)q2 * s_x[c_off + 2]
                               + (float)q3 * s_x[c_off + 3];
                    acc += sv * dot4;
                }
            } else if (bits == 2) {
                const unsigned char* lo2_row = packed_lo + (long long)row * (K >> 2);
                for (int c_off = lane * 4; c_off < c_cur; c_off += 128) {
                    unsigned char b = lo2_row[(k_start + c_off) >> 2];
                    unsigned int q0 = ((unsigned int)((b >> 0) & 3u)) & mask;
                    unsigned int q1 = ((unsigned int)((b >> 2) & 3u)) & mask;
                    unsigned int q2 = ((unsigned int)((b >> 4) & 3u)) & mask;
                    unsigned int q3 = ((unsigned int)((b >> 6) & 3u)) & mask;
                    float sv = bf16_to_f32(s_row[g_start + (c_off / GS)]);
                    float dot4 = (float)q0 * s_x[c_off]
                               + (float)q1 * s_x[c_off + 1]
                               + (float)q2 * s_x[c_off + 2]
                               + (float)q3 * s_x[c_off + 3];
                    acc += sv * dot4;
                }
            } else {
                for (int c_off = lane * 4; c_off < c_cur; c_off += 128) {
                    int k_abs = k_start + c_off;
                    int lo_idx = k_abs >> 1;
                    unsigned char b0 = lo_row[lo_idx];
                    unsigned char b1 = lo_row[lo_idx + 1];
                    unsigned int q0 = (unsigned int)(b0 & 0x0Fu);
                    unsigned int q1 = (unsigned int)(b0 >> 4);
                    unsigned int q2 = (unsigned int)(b1 & 0x0Fu);
                    unsigned int q3 = (unsigned int)(b1 >> 4);
                    if (bits == 6) {
                        unsigned char h = hi_row[k_abs >> 2];
                        q0 |= ((unsigned int)(h & 3u)) << 4;
                        q1 |= ((unsigned int)((h >> 2) & 3u)) << 4;
                        q2 |= ((unsigned int)((h >> 4) & 3u)) << 4;
                        q3 |= ((unsigned int)((h >> 6) & 3u)) << 4;
                    }
                    q0 &= mask; q1 &= mask; q2 &= mask; q3 &= mask;
                    float sv = bf16_to_f32(s_row[g_start + (c_off / GS)]);
                    float dot4 = (float)q0 * s_x[c_off]
                               + (float)q1 * s_x[c_off + 1]
                               + (float)q2 * s_x[c_off + 2]
                               + (float)q3 * s_x[c_off + 3];
                    acc += sv * dot4;
                }
            }

            for (int g = lane; g < ng_cur; g += 32) {
                int gg = g_start + g;
                float z_eff = bf16_to_f32(z_row[gg]) + bf16_to_f32(s_row[gg]) * tail_mid;
                acc += z_eff * s_xsum[g];
            }
        }
        if (k_start + chunk_cap < K) {
            __syncthreads();
        }
    }

    if (row >= N) return;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (lane == 0) {
        if (use_fwht) {
            acc *= bf16_to_f32(row_scale[row]);
        }
        if (bias_bf16 != nullptr) {
            acc += bf16_to_f32(bias_bf16[row]);
        }
        long long idx = (long long)m * N + row;
        if (out_is_bf16) {
            reinterpret_cast<unsigned short*>(y_out)[idx] = f32_to_bf16_rne(acc);
        } else {
            reinterpret_cast<float*>(y_out)[idx] = acc;
        }
    }
}

/* 7. Ideal Q1 FWHT + Sign Planes GEMV with optional cand_rows & plane_start > 0 accumulate */
extern "C" __global__ void ideal_q1_fwht_gemv_f32(
    const unsigned short* __restrict__ x_bf16,
    const unsigned char* __restrict__ packed_lo,
    const unsigned short* __restrict__ qscale,
    const unsigned short* __restrict__ qzero,
    const unsigned short* __restrict__ row_scale,
    const unsigned short* __restrict__ col_scale,
    const int* __restrict__ cand_rows,
    const unsigned short* __restrict__ bias_bf16,
    void* __restrict__ y_out,
    int M, int N, int K, int GS,
    int P_total, int plane_start, int plane_count,
    int cand_count, int accumulate, int out_is_bf16, int shared_scale
) {
    extern __shared__ char smem_raw[];
    int chunk_cap = (K < 4096) ? K : 4096;
    float* s_x = reinterpret_cast<float*>(smem_raw);
    float* s_xsum = s_x + chunk_cap;

    int tid = threadIdx.x;
    int m = blockIdx.y;
    if (m >= M) return;

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int out_idx = blockIdx.x * warps_per_block + warp_id;
    int num_rows = (cand_count > 0) ? cand_count : N;
    int row = (out_idx < num_rows) ? ((cand_count > 0) ? cand_rows[out_idx] : out_idx) : -1;

    const unsigned short* x_row = x_bf16 + (long long)m * K;
    int num_groups_full = K / GS;
    int k8 = K >> 3;
    long long plane_byte_stride = (long long)N * k8;
    long long plane_grp_stride = (long long)N * num_groups_full;

    float acc = 0.0f;
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

        block_fwht_and_sum(s_x, s_xsum, ng_cur, GS, warp_id, lane, warps_per_block);
        __syncthreads();

        if (row >= 0 && row < N) {
            if (plane_start == 0) {
                const unsigned short* z_row = qzero + (long long)row * num_groups_full;
                for (int g = lane; g < ng_cur; g += 32) {
                    acc += bf16_to_f32(z_row[g_start + g]) * s_xsum[g];
                }
            }
            for (int p = plane_start; p < plane_start + plane_count; ++p) {
                const unsigned short* s_row = shared_scale
                    ? (qscale + (long long)row * num_groups_full)
                    : (qscale + p * plane_grp_stride + (long long)row * num_groups_full);
                float p_mult = shared_scale ? ldexpf(1.0f, -p) : 1.0f;
                const unsigned char* b_row = packed_lo + p * plane_byte_stride + (long long)row * k8 + (k_start >> 3);
                const unsigned short* b_u16 = reinterpret_cast<const unsigned short*>(b_row);

                for (int c_off = lane * 16; c_off < c_cur; c_off += 512) {
                    unsigned int bits16 = (unsigned int)b_u16[c_off >> 4];
                    float sv = bf16_to_f32(s_row[g_start + (c_off / GS)]) * p_mult;
                    float d1 = 0.0f;
                    while (bits16) {
                        int b = __ffs(bits16) - 1;
                        d1 += s_x[c_off + b];
                        bits16 &= (bits16 - 1u);
                    }
                    acc += 2.0f * sv * d1;
                }
                for (int g = lane; g < ng_cur; g += 32) {
                    acc -= (bf16_to_f32(s_row[g_start + g]) * p_mult) * s_xsum[g];
                }
            }
        }
        if (k_start + chunk_cap < K) {
            __syncthreads();
        }
    }

    if (row < 0 || row >= N) return;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (lane == 0) {
        float val = acc * bf16_to_f32(row_scale[row]);
        if (plane_start == 0 && bias_bf16 != nullptr) {
            val += bf16_to_f32(bias_bf16[row]);
        }
        long long idx = (long long)m * N + row;
        if (accumulate) {
            if (out_is_bf16) {
                unsigned short* out_p = reinterpret_cast<unsigned short*>(y_out);
                out_p[idx] = f32_to_bf16_rne(bf16_to_f32(out_p[idx]) + val);
            } else {
                reinterpret_cast<float*>(y_out)[idx] += val;
            }
        } else {
            if (out_is_bf16) {
                reinterpret_cast<unsigned short*>(y_out)[idx] = f32_to_bf16_rne(val);
            } else {
                reinterpret_cast<float*>(y_out)[idx] = val;
            }
        }
    }
}

/* Verification kernel for Ampere PTX mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc */
extern "C" __global__ void b1_mma_m8n8k128_verify_u32(
    const unsigned int* __restrict__ a_bits,  /* [8, 4] uint32 */
    const unsigned int* __restrict__ b_bits,  /* [8, 4] uint32 */
    int* __restrict__ c_out                   /* [8, 8] int32 */
) {
    int lane = threadIdx.x & 31;
    int groupID = lane >> 2;
    int tid_g = lane & 3;
    unsigned int a0 = a_bits[groupID * 4 + tid_g];
    unsigned int b0 = b_bits[groupID * 4 + tid_g];
    int d0 = 0, d1 = 0;
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc "
        "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(b0), "r"(0), "r"(0)
    );
#else
    int n0 = tid_g * 2;
    int n1 = n0 + 1;
    for (int k_word = 0; k_word < 4; ++k_word) {
        unsigned int aw = __shfl_sync(0xffffffffu, a0, groupID * 4 + k_word);
        unsigned int bw0 = __shfl_sync(0xffffffffu, b0, n0 * 4 + k_word);
        unsigned int bw1 = __shfl_sync(0xffffffffu, b0, n1 * 4 + k_word);
        d0 += __popc(aw & bw0);
        d1 += __popc(aw & bw1);
    }
#endif
    c_out[groupID * 8 + tid_g * 2] = d0;
    c_out[groupID * 8 + tid_g * 2 + 1] = d1;
}

/* 8. Ideal Q1 Batched 1-bit Tensor Core MMA GEMM (sm_80+ b1.b1.s32.and.popc + __popc fallback) */
extern "C" __global__ void ideal_q1_b1_mma_gemm_f32(
    const unsigned short* __restrict__ x_bf16,
    const unsigned char* __restrict__ packed_lo,
    const unsigned short* __restrict__ qscale,
    const unsigned short* __restrict__ qzero,
    const unsigned short* __restrict__ row_scale,
    const unsigned short* __restrict__ col_scale,
    const unsigned short* __restrict__ bias_bf16,
    void* __restrict__ y_out,
    int M, int N, int K, int GS,
    int P_total, int P_active, int out_is_bf16, int shared_scale
) {
    /* Tile: M_tile = 8 rows per blockIdx.y, N_tile = 64 rows per blockIdx.x (8 warps x 8 rows) */
    const int K_CAP = 256;
    extern __shared__ char smem_raw[];
    float* s_x = reinterpret_cast<float*>(smem_raw);                         /* [8 * K_CAP] */
    float* s_xsum = s_x + 8 * K_CAP;                                         /* [8 * (K_CAP / GS)] */
    float* s_xzero = s_xsum + 8 * (K_CAP / GS);                              /* [8 * (K_CAP / GS)] */
    float* s_xscale = s_xzero + 8 * (K_CAP / GS);                            /* [8 * (K_CAP / GS)] */
    unsigned int* s_xplanes = reinterpret_cast<unsigned int*>(s_xscale + 8 * (K_CAP / GS)); /* [8 * 8 * 8] */
    unsigned char* s_qx = reinterpret_cast<unsigned char*>(s_xplanes + 512); /* [8 * K_CAP] */

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;

    int m_base = blockIdx.y * 8;
    int r_base = blockIdx.x * 64 + warp_id * 8;

    int groupID = lane >> 2;
    int tid_g = lane & 3;
    int n_local0 = tid_g * 2;
    int n_local1 = n_local0 + 1;
    int row0 = r_base + n_local0;
    int row1 = r_base + n_local1;
    int row_b = r_base + groupID;
    int m_out = m_base + groupID;

    int num_groups_full = K / GS;
    int k8 = K >> 3;
    long long plane_byte_stride = (long long)N * k8;
    long long plane_grp_stride = (long long)N * num_groups_full;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int k_start = 0; k_start < K; k_start += K_CAP) {
        int c_cur = K - k_start;
        if (c_cur > K_CAP) c_cur = K_CAP;
        int ng_cur = c_cur / GS;
        int g_start = k_start / GS;
        int c_words = (c_cur + 31) >> 5;
        int k128_steps = (c_cur + 127) >> 7;

        /* 1. Load & scale activations [8, c_cur] */
        for (int idx = tid; idx < 8 * c_cur; idx += blockDim.x) {
            int ml = idx / c_cur;
            int c = idx - ml * c_cur;
            int mg = m_base + ml;
            int ka = k_start + c;
            s_x[idx] = (mg < M) ? (bf16_to_f32(x_bf16[(long long)mg * K + ka]) * bf16_to_f32(col_scale[ka])) : 0.0f;
        }
        __syncthreads();

        /* 2. Cooperative FWHT per row m_local = 0..7 & 8-bit group quantization */
        block_fwht_and_sum(s_x, s_xsum, 8 * ng_cur, GS, warp_id, lane, warps_per_block);
        __syncthreads();

        for (int g_all = tid; g_all < 8 * ng_cur; g_all += blockDim.x) {
            int base = g_all * GS;
            float mn = s_x[base];
            float mx = mn;
            for (int j = 1; j < GS; ++j) {
                float v = s_x[base + j];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            float sc = (mx - mn) * (1.0f / 255.0f);
            if (sc < 1e-8f) sc = 1e-8f;
            s_xzero[g_all] = mn;
            s_xscale[g_all] = sc;
            float inv_sc = 1.0f / sc;
            for (int j = 0; j < GS; ++j) {
                int qv = __float2int_rn((s_x[base + j] - mn) * inv_sc);
                if (qv < 0) qv = 0;
                if (qv > 255) qv = 255;
                s_qx[base + j] = (unsigned char)qv;
            }
        }
        __syncthreads();

        /* 3. Bit-slice quantized activations into s_xplanes[ml * 64 + a * 8 + w] */
        for (int idx = tid; idx < 8 * 8 * 8; idx += blockDim.x) {
            int ml = idx >> 6;
            int rem = idx & 63;
            int a = rem >> 3;
            int w = rem & 7;
            unsigned int word = 0u;
            if (w < c_words && (m_base + ml) < M) {
                int base = ml * c_cur + w * 32;
                int avail = c_cur - w * 32;
                if (avail > 32) avail = 32;
                for (int b = 0; b < avail; ++b) {
                    word |= ((unsigned int)((s_qx[base + b] >> a) & 1u)) << b;
                }
            }
            s_xplanes[idx] = word;
        }
        __syncthreads();

        /* 4. Per-warp 8x8 MMA accumulation */
        if (r_base < N) {
            if (m_out < M) {
                if (row0 < N) {
                    const unsigned short* z0 = qzero + (long long)row0 * num_groups_full + g_start;
                    for (int g = 0; g < ng_cur; ++g) {
                        acc0 += bf16_to_f32(z0[g]) * s_xsum[groupID * ng_cur + g];
                    }
                }
                if (row1 < N) {
                    const unsigned short* z1 = qzero + (long long)row1 * num_groups_full + g_start;
                    for (int g = 0; g < ng_cur; ++g) {
                        acc1 += bf16_to_f32(z1[g]) * s_xsum[groupID * ng_cur + g];
                    }
                }
            }

            int g_per_128 = 128 / GS;
            int max_byte = c_cur >> 3;
            for (int p = 0; p < P_active; ++p) {
                const unsigned char* b_plane_base = packed_lo + p * plane_byte_stride + (long long)row_b * k8 + (k_start >> 3);
                long long s_p_off = shared_scale ? 0LL : (p * plane_grp_stride);
                float p_mult = shared_scale ? ldexpf(1.0f, -p) : 1.0f;
                const unsigned short* s0_row = (row0 < N) ? (qscale + s_p_off + (long long)row0 * num_groups_full + g_start) : nullptr;
                const unsigned short* s1_row = (row1 < N) ? (qscale + s_p_off + (long long)row1 * num_groups_full + g_start) : nullptr;

                for (int k128 = 0; k128 < k128_steps; ++k128) {
                    int w_abs = k128 * 4 + tid_g;
                    unsigned int b_word = 0u;
                    if (row_b < N && w_abs < c_words) {
                        int b_off = w_abs * 4;
                        #pragma unroll
                        for (int bi = 0; bi < 4; ++bi) {
                            if (b_off + bi < max_byte) {
                                b_word |= ((unsigned int)b_plane_base[b_off + bi]) << (bi * 8);
                            }
                        }
                    }

                    for (int sg = 0; sg < g_per_128; ++sg) {
                        int g_loc = k128 * g_per_128 + sg;
                        if (g_loc >= ng_cur) break;
                        unsigned int w_mask = 0u;
                        if (GS == 64) {
                            w_mask = ((tid_g >> 1) == sg) ? 0xFFFFFFFFu : 0u;
                        } else if (GS == 32) {
                            w_mask = (tid_g == sg) ? 0xFFFFFFFFu : 0u;
                        } else {
                            if (tid_g == (sg >> 1)) {
                                w_mask = (sg & 1) ? 0xFFFF0000u : 0x0000FFFFu;
                            }
                        }

                        unsigned int b_sub = b_word & w_mask;
                        int pop_row = __popc(b_sub);
                        pop_row += __shfl_xor_sync(0xffffffffu, pop_row, 1);
                        pop_row += __shfl_xor_sync(0xffffffffu, pop_row, 2);
                        int pop0 = __shfl_sync(0xffffffffu, pop_row, n_local0 * 4);
                        int pop1 = __shfl_sync(0xffffffffu, pop_row, n_local1 * 4);

                        int dot_acc0 = 0;
                        int dot_acc1 = 0;
                        #pragma unroll
                        for (int a = 0; a < 8; ++a) {
                            unsigned int a_sub = s_xplanes[((groupID * 8 + a) * 8) + k128 * 4 + tid_g] & w_mask;
                            int d0 = 0, d1 = 0;
#if __CUDA_ARCH__ >= 800
                            asm volatile(
                                "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
                                : "=r"(d0), "=r"(d1)
                                : "r"(a_sub), "r"(b_sub), "r"(0), "r"(0)
                            );
#else
                            for (int kw = 0; kw < 4; ++kw) {
                                unsigned int aw = __shfl_sync(0xffffffffu, a_sub, groupID * 4 + kw);
                                unsigned int bw0 = __shfl_sync(0xffffffffu, b_sub, n_local0 * 4 + kw);
                                unsigned int bw1 = __shfl_sync(0xffffffffu, b_sub, n_local1 * 4 + kw);
                                d0 += __popc(aw & bw0);
                                d1 += __popc(aw & bw1);
                            }
#endif
                            dot_acc0 += (d0 << a);
                            dot_acc1 += (d1 << a);
                        }

                        if (m_out < M) {
                            int x_idx = groupID * ng_cur + g_loc;
                            float xz = s_xzero[x_idx];
                            float xs = s_xscale[x_idx];
                            float xsm = s_xsum[x_idx];
                            if (row0 < N) {
                                float xdb0 = xz * (float)pop0 + xs * (float)dot_acc0;
                                acc0 += (bf16_to_f32(s0_row[g_loc]) * p_mult) * (2.0f * xdb0 - xsm);
                            }
                            if (row1 < N) {
                                float xdb1 = xz * (float)pop1 + xs * (float)dot_acc1;
                                acc1 += (bf16_to_f32(s1_row[g_loc]) * p_mult) * (2.0f * xdb1 - xsm);
                            }
                        }
                    }
                }
            }
        }
        if (k_start + K_CAP < K) {
            __syncthreads();
        }
    }

    if (m_out >= M || r_base >= N) return;
    if (row0 < N) {
        float v0 = acc0 * bf16_to_f32(row_scale[row0]);
        if (bias_bf16 != nullptr) v0 += bf16_to_f32(bias_bf16[row0]);
        long long idx0 = (long long)m_out * N + row0;
        if (out_is_bf16) reinterpret_cast<unsigned short*>(y_out)[idx0] = f32_to_bf16_rne(v0);
        else reinterpret_cast<float*>(y_out)[idx0] = v0;
    }
    if (row1 < N) {
        float v1 = acc1 * bf16_to_f32(row_scale[row1]);
        if (bias_bf16 != nullptr) v1 += bf16_to_f32(bias_bf16[row1]);
        long long idx1 = (long long)m_out * N + row1;
        if (out_is_bf16) reinterpret_cast<unsigned short*>(y_out)[idx1] = f32_to_bf16_rne(v1);
        else reinterpret_cast<float*>(y_out)[idx1] = v1;
    }
}

/* 9. Ideal Q2 (2-bit + FWHT + Sinkhorn) GEMV */
extern "C" __global__ void quintessentia_q2_fwht_gemv_f32(
    const unsigned short* __restrict__ x_bf16,
    const unsigned char* __restrict__ packed_lo,
    const unsigned short* __restrict__ qscale,
    const unsigned short* __restrict__ qzero,
    const unsigned short* __restrict__ row_scale,
    const unsigned short* __restrict__ col_scale,
    const unsigned short* __restrict__ bias_bf16,
    void* __restrict__ y_out,
    int M, int N, int K, int GS, int out_is_bf16
) {
    extern __shared__ char smem_raw[];
    int chunk_cap = (K < 4096) ? K : 4096;
    float* s_x = reinterpret_cast<float*>(smem_raw);
    float* s_xsum = s_x + chunk_cap;

    int tid = threadIdx.x;
    int m = blockIdx.y;
    if (m >= M) return;

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp_id;

    const unsigned short* x_row = x_bf16 + (long long)m * K;
    int num_groups_full = K / GS;
    const unsigned char* lo_row = (row < N) ? (packed_lo + (long long)row * (K >> 2)) : nullptr;
    const unsigned short* s_row = (row < N) ? (qscale + (long long)row * num_groups_full) : nullptr;
    const unsigned short* z_row = (row < N) ? (qzero  + (long long)row * num_groups_full) : nullptr;

    float acc = 0.0f;
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

        block_fwht_and_sum(s_x, s_xsum, ng_cur, GS, warp_id, lane, warps_per_block);
        __syncthreads();

        if (row < N) {
            for (int c_off = lane * 4; c_off < c_cur; c_off += 128) {
                unsigned char b = lo_row[(k_start + c_off) >> 2];
                float sv = bf16_to_f32(s_row[g_start + (c_off / GS)]);
                float dot4 = (float)(b & 3u) * s_x[c_off]
                           + (float)((b >> 2) & 3u) * s_x[c_off + 1]
                           + (float)((b >> 4) & 3u) * s_x[c_off + 2]
                           + (float)((b >> 6) & 3u) * s_x[c_off + 3];
                acc += sv * dot4;
            }
            for (int g = lane; g < ng_cur; g += 32) {
                acc += bf16_to_f32(z_row[g_start + g]) * s_xsum[g];
            }
        }
        if (k_start + chunk_cap < K) {
            __syncthreads();
        }
    }

    if (row >= N) return;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (lane == 0) {
        float val = acc * bf16_to_f32(row_scale[row]);
        if (bias_bf16 != nullptr) val += bf16_to_f32(bias_bf16[row]);
        long long idx = (long long)m * N + row;
        if (out_is_bf16) reinterpret_cast<unsigned short*>(y_out)[idx] = f32_to_bf16_rne(val);
        else reinterpret_cast<float*>(y_out)[idx] = val;
    }
}

/* 10. Quintessentia Q8 (8-bit group affine) GEMV */
extern "C" __global__ void quintessentia_q8_gemv_f32(
    const unsigned short* __restrict__ x_bf16,
    const unsigned char* __restrict__ packed_lo,
    const unsigned short* __restrict__ qscale,
    const unsigned short* __restrict__ qzero,
    const unsigned short* __restrict__ bias_bf16,
    void* __restrict__ y_out,
    int M, int N, int K, int GS, int out_is_bf16
) {
    extern __shared__ char smem_raw[];
    int chunk_cap = (K < 4096) ? K : 4096;
    float* s_x = reinterpret_cast<float*>(smem_raw);
    float* s_xsum = s_x + chunk_cap;

    int tid = threadIdx.x;
    int m = blockIdx.y;
    if (m >= M) return;

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp_id;

    const unsigned short* x_row = x_bf16 + (long long)m * K;
    int num_groups_full = K / GS;
    const unsigned char* lo_row = (row < N) ? (packed_lo + (long long)row * K) : nullptr;
    const unsigned short* s_row = (row < N) ? (qscale + (long long)row * num_groups_full) : nullptr;
    const unsigned short* z_row = (row < N) ? (qzero  + (long long)row * num_groups_full) : nullptr;

    float acc = 0.0f;
    for (int k_start = 0; k_start < K; k_start += chunk_cap) {
        int c_cur = K - k_start;
        if (c_cur > chunk_cap) c_cur = chunk_cap;
        int ng_cur = c_cur / GS;
        int g_start = k_start / GS;

        for (int c = tid; c < c_cur; c += blockDim.x) {
            s_x[c] = bf16_to_f32(x_row[k_start + c]);
        }
        __syncthreads();

        for (int g = tid; g < ng_cur; g += blockDim.x) {
            float sum = 0.0f;
            int base = g * GS;
            for (int j = 0; j < GS; ++j) sum += s_x[base + j];
            s_xsum[g] = sum;
        }
        __syncthreads();

        if (row < N) {
            const unsigned int* lo_u32 = reinterpret_cast<const unsigned int*>(lo_row + k_start);
            for (int c_off = lane * 4; c_off < c_cur; c_off += 128) {
                unsigned int w4 = lo_u32[c_off >> 2];
                float sv = bf16_to_f32(s_row[g_start + (c_off / GS)]);
                float dot4 = (float)(w4 & 0xFFu) * s_x[c_off]
                           + (float)((w4 >> 8) & 0xFFu) * s_x[c_off + 1]
                           + (float)((w4 >> 16) & 0xFFu) * s_x[c_off + 2]
                           + (float)(w4 >> 24) * s_x[c_off + 3];
                acc += sv * dot4;
            }
            for (int g = lane; g < ng_cur; g += 32) {
                acc += bf16_to_f32(z_row[g_start + g]) * s_xsum[g];
            }
        }
        if (k_start + chunk_cap < K) {
            __syncthreads();
        }
    }

    if (row >= N) return;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (lane == 0) {
        if (bias_bf16 != nullptr) acc += bf16_to_f32(bias_bf16[row]);
        long long idx = (long long)m * N + row;
        if (out_is_bf16) reinterpret_cast<unsigned short*>(y_out)[idx] = f32_to_bf16_rne(acc);
        else reinterpret_cast<float*>(y_out)[idx] = acc;
    }
}

// Kernel #11: Q1PuzzleOverlay sparse Q8 delta-cube accumulation (atomicAdd into y_acc_f32).
// Grid: (M_sel * ((R_tile + 7) / 8), M, 1), Block: (256, 1, 1) -> 8 warps = 8 cube rows per block.
extern "C" __global__ void puzzle_overlay_delta_cubes_f32(
    const unsigned short* __restrict__ x_bf16,
    const unsigned char* __restrict__ delta_q8,
    const unsigned short* __restrict__ delta_scale,
    const unsigned short* __restrict__ delta_zero,
    const int* __restrict__ cube_coords_i32,
    float* __restrict__ y_acc_f32,
    int M,
    int N,
    int K,
    int M_sel,
    int R_tile,
    int C_tile,
    int GS
) {
    int row_blocks_per_cube = (R_tile + 7) >> 3;
    int sel = blockIdx.x / row_blocks_per_cube;
    int blk_r = blockIdx.x - sel * row_blocks_per_cube;
    int m = blockIdx.y;
    if (sel >= M_sel || m >= M) return;

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int local_r = blk_r * 8 + warp;

    int r0 = cube_coords_i32[sel * 4 + 0];
    int r_len = cube_coords_i32[sel * 4 + 1];
    int c0 = cube_coords_i32[sel * 4 + 2];
    int c_len = cube_coords_i32[sel * 4 + 3];

    const unsigned short* x_row = x_bf16 + (long long)m * K;
    int n_groups_tile = C_tile / GS;
    int chunk_cap = C_tile < 4096 ? C_tile : 4096;

    extern __shared__ float smem[];
    float* s_x = smem;
    float* s_xsum = smem + chunk_cap;

    bool row_valid = (local_r < r_len) && (local_r < R_tile) && ((r0 + local_r) < N);
    long long row_idx = (long long)sel * R_tile + local_r;
    float acc = 0.0f;

    for (int c_start = 0; c_start < C_tile; c_start += chunk_cap) {
        int c_cur = C_tile - c_start;
        if (c_cur > chunk_cap) c_cur = chunk_cap;
        int ng_cur = c_cur / GS;

        for (int c = tid; c < c_cur; c += blockDim.x) {
            int local_c = c_start + c;
            int global_c = c0 + local_c;
            s_x[c] = (local_c < c_len && global_c < K) ? bf16_to_f32(x_row[global_c]) : 0.0f;
        }
        __syncthreads();

        for (int g = tid; g < ng_cur; g += blockDim.x) {
            float sum = 0.0f;
            int base = g * GS;
            for (int j = 0; j < GS; ++j) sum += s_x[base + j];
            s_xsum[g] = sum;
        }
        __syncthreads();

        if (row_valid) {
            int g_start = c_start / GS;
            const unsigned char* q_row = delta_q8 + row_idx * C_tile + c_start;
            const unsigned short* s_row = delta_scale + row_idx * n_groups_tile + g_start;
            const unsigned short* z_row = delta_zero + row_idx * n_groups_tile + g_start;

            const unsigned int* q_u32 = reinterpret_cast<const unsigned int*>(q_row);
            for (int c_off = lane * 4; c_off < c_cur; c_off += 128) {
                unsigned int w4 = q_u32[c_off >> 2];
                float sv = bf16_to_f32(s_row[c_off / GS]);
                float dot4 = (float)(w4 & 0xFFu) * s_x[c_off]
                           + (float)((w4 >> 8) & 0xFFu) * s_x[c_off + 1]
                           + (float)((w4 >> 16) & 0xFFu) * s_x[c_off + 2]
                           + (float)(w4 >> 24) * s_x[c_off + 3];
                acc += sv * dot4;
            }
            for (int g = lane; g < ng_cur; g += 32) {
                acc += bf16_to_f32(z_row[g]) * s_xsum[g];
            }
        }
        if (c_start + chunk_cap < C_tile) {
            __syncthreads();
        }
    }

    if (!row_valid) return;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (lane == 0 && acc != 0.0f) {
        atomicAdd(&y_acc_f32[(long long)m * N + r0 + local_r], acc);
    }
}

/* 12. Exact IEEE-754 FP64-decision affine Q4/Q6/Q8 weight packer (1 launch, zero temp VRAM) */
extern "C" __global__ void quintessentia_pack_affine_q46(
    const void* __restrict__ src_ptr,
    unsigned char* __restrict__ packed_lo,
    unsigned char* __restrict__ packed_hi,
    unsigned short* __restrict__ qscale,
    unsigned short* __restrict__ qzero,
    long long total_groups,
    int GS,
    int bits,
    int is_bf16
) {
    __shared__ float sm_min[8];
    __shared__ float sm_max[8];

    int t = threadIdx.x;
    int warp_id = t >> 5;
    int lane = t & 31;
    int groups_per_block = 256 / GS;
    int local_g = t / GS;
    int in_g = t - local_g * GS;
    long long global_g = (long long)blockIdx.x * groups_per_block + local_g;
    bool valid = (global_g < total_groups);
    long long elem_idx = global_g * (long long)GS + in_g;

    float w_f = 0.0f;
    if (valid) {
        if (is_bf16) {
            w_f = bf16_to_f32(reinterpret_cast<const unsigned short*>(src_ptr)[elem_idx]);
        } else {
            w_f = reinterpret_cast<const float*>(src_ptr)[elem_idx];
        }
    }

    float w_min = w_f;
    float w_max = w_f;
    if (GS == 16) {
        #pragma unroll
        for (int d = 8; d > 0; d >>= 1) {
            w_min = fminf(w_min, __shfl_xor_sync(0xffffffffu, w_min, d));
            w_max = fmaxf(w_max, __shfl_xor_sync(0xffffffffu, w_max, d));
        }
    } else if (GS == 32) {
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            w_min = fminf(w_min, __shfl_xor_sync(0xffffffffu, w_min, d));
            w_max = fmaxf(w_max, __shfl_xor_sync(0xffffffffu, w_max, d));
        }
    } else {
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            w_min = fminf(w_min, __shfl_xor_sync(0xffffffffu, w_min, d));
            w_max = fmaxf(w_max, __shfl_xor_sync(0xffffffffu, w_max, d));
        }
        if (lane == 0) {
            sm_min[warp_id] = w_min;
            sm_max[warp_id] = w_max;
        }
        __syncthreads();
        int pair = warp_id & ~1;
        w_min = fminf(sm_min[pair], sm_min[pair + 1]);
        w_max = fmaxf(sm_max[pair], sm_max[pair + 1]);
    }

    int max_qi = (1 << bits) - 1;
    int qi = 0;
    unsigned short zero_bf16 = 0;
    unsigned short scale_bf16 = 0;
    if (valid) {
        zero_bf16 = f32_to_bf16_rne(w_min);
        double zero_d = (double)bf16_to_f32(zero_bf16);
        double scale_raw = __ddiv_rn((double)w_max - zero_d, (double)max_qi);
        if (scale_raw < 1e-8) scale_raw = 1e-8;
        scale_bf16 = f32_to_bf16_rne((float)scale_raw);
        double scale_d = (double)bf16_to_f32(scale_bf16);
        double qd = rint(__ddiv_rn((double)w_f - zero_d, scale_d));
        qi = (int)qd;
        if (qi < 0) qi = 0;
        if (qi > max_qi) qi = max_qi;
        if (in_g == 0) {
            qscale[global_g] = scale_bf16;
            qzero[global_g] = zero_bf16;
        }
    }

    int q1 = __shfl_down_sync(0xffffffffu, qi, 1);
    int q2 = __shfl_down_sync(0xffffffffu, qi, 2);
    int q3 = __shfl_down_sync(0xffffffffu, qi, 3);
    if (!valid) return;

    if (bits == 8) {
        packed_lo[elem_idx] = (unsigned char)qi;
    } else {
        if ((in_g & 1) == 0) {
            packed_lo[elem_idx >> 1] = (unsigned char)((qi & 15) | ((q1 & 15) << 4));
        }
        if (bits == 6 && (in_g & 3) == 0) {
            packed_hi[elem_idx >> 2] = (unsigned char)(
                ((qi >> 4) & 3)
                | (((q1 >> 4) & 3) << 2)
                | (((q2 >> 4) & 3) << 4)
                | (((q3 >> 4) & 3) << 6)
            );
        }
    }
}

/* 13. Bit-exact affine Q4/Q6/Q8 weight unpacker (matches q.float()*qs.float()+qz.float()) */
extern "C" __global__ void quintessentia_unpack_affine_q46(
    const unsigned char* __restrict__ packed_lo,
    const unsigned char* __restrict__ packed_hi,
    const unsigned short* __restrict__ qscale,
    const unsigned short* __restrict__ qzero,
    void* __restrict__ dst_ptr,
    long long total_elems,
    int gs_shift,
    int bits,
    int dst_is_bf16
) {
    long long idx = (long long)blockIdx.x * 256 + threadIdx.x;
    if (idx >= total_elems) return;
    long long g_idx = idx >> gs_shift;
    unsigned int q;
    if (bits == 8) {
        q = (unsigned int)packed_lo[idx];
    } else {
        unsigned char b = packed_lo[idx >> 1];
        q = (idx & 1LL) ? (unsigned int)(b >> 4) : (unsigned int)(b & 15u);
        if (bits == 6) {
            unsigned char h = packed_hi[idx >> 2];
            q |= (((unsigned int)(h >> ((idx & 3LL) * 2))) & 3u) << 4;
        }
    }
    float out_f = __fadd_rn(__fmul_rn((float)q, bf16_to_f32(qscale[g_idx])), bf16_to_f32(qzero[g_idx]));
    if (dst_is_bf16) {
        reinterpret_cast<unsigned short*>(dst_ptr)[idx] = f32_to_bf16_rne(out_f);
    } else {
        reinterpret_cast<float*>(dst_ptr)[idx] = out_f;
    }
}

/* 14. Zero-alloc fused quant+dequant affine Q4/Q6/Q8 (bit-exact with unpack(pack(w))) */
extern "C" __global__ void quintessentia_quant_dequant_affine_q46(
    const void* __restrict__ src_ptr,
    void* __restrict__ dst_ptr,
    long long total_groups,
    int GS,
    int bits,
    int src_is_bf16,
    int dst_is_bf16
) {
    __shared__ float sm_min[8];
    __shared__ float sm_max[8];

    int t = threadIdx.x;
    int warp_id = t >> 5;
    int lane = t & 31;
    int groups_per_block = 256 / GS;
    int local_g = t / GS;
    int in_g = t - local_g * GS;
    long long global_g = (long long)blockIdx.x * groups_per_block + local_g;
    bool valid = (global_g < total_groups);
    long long elem_idx = global_g * (long long)GS + in_g;

    float w_f = 0.0f;
    if (valid) {
        if (src_is_bf16) {
            w_f = bf16_to_f32(reinterpret_cast<const unsigned short*>(src_ptr)[elem_idx]);
        } else {
            w_f = reinterpret_cast<const float*>(src_ptr)[elem_idx];
        }
    }

    float w_min = w_f;
    float w_max = w_f;
    if (GS == 16) {
        #pragma unroll
        for (int d = 8; d > 0; d >>= 1) {
            w_min = fminf(w_min, __shfl_xor_sync(0xffffffffu, w_min, d));
            w_max = fmaxf(w_max, __shfl_xor_sync(0xffffffffu, w_max, d));
        }
    } else if (GS == 32) {
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            w_min = fminf(w_min, __shfl_xor_sync(0xffffffffu, w_min, d));
            w_max = fmaxf(w_max, __shfl_xor_sync(0xffffffffu, w_max, d));
        }
    } else {
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            w_min = fminf(w_min, __shfl_xor_sync(0xffffffffu, w_min, d));
            w_max = fmaxf(w_max, __shfl_xor_sync(0xffffffffu, w_max, d));
        }
        if (lane == 0) {
            sm_min[warp_id] = w_min;
            sm_max[warp_id] = w_max;
        }
        __syncthreads();
        int pair = warp_id & ~1;
        w_min = fminf(sm_min[pair], sm_min[pair + 1]);
        w_max = fmaxf(sm_max[pair], sm_max[pair + 1]);
    }

    if (!valid) return;

    int max_qi = (1 << bits) - 1;
    unsigned short zero_bf16 = f32_to_bf16_rne(w_min);
    float zero_f = bf16_to_f32(zero_bf16);
    double zero_d = (double)zero_f;
    double scale_raw = __ddiv_rn((double)w_max - zero_d, (double)max_qi);
    if (scale_raw < 1e-8) scale_raw = 1e-8;
    unsigned short scale_bf16 = f32_to_bf16_rne((float)scale_raw);
    float scale_f = bf16_to_f32(scale_bf16);
    double qd = rint(__ddiv_rn((double)w_f - zero_d, (double)scale_f));
    int qi = (int)qd;
    if (qi < 0) qi = 0;
    if (qi > max_qi) qi = max_qi;

    float out_f = __fadd_rn(__fmul_rn((float)qi, scale_f), zero_f);
    if (dst_is_bf16) {
        reinterpret_cast<unsigned short*>(dst_ptr)[elem_idx] = f32_to_bf16_rne(out_f);
    } else {
        reinterpret_cast<float*>(dst_ptr)[elem_idx] = out_f;
    }
}
"""

_KERNEL_NAMES = (
    b"q1_cube_plane_gemv_f32",
    b"q1_cube_popc_gemv_f32",
    b"q1_cube_multiplane_gemv_f32",
    b"q1_cube_popc_multiplane_gemv_f32",
    b"q1_cube_adaptive_radar_gemv_f32",
    b"q1_packed_trunc_gemv_f32",
    b"ideal_q1_fwht_gemv_f32",
    b"b1_mma_m8n8k128_verify_u32",
    b"ideal_q1_b1_mma_gemm_f32",
    b"quintessentia_q2_fwht_gemv_f32",
    b"quintessentia_q8_gemv_f32",
    b"puzzle_overlay_delta_cubes_f32",
    b"quintessentia_pack_affine_q46",
    b"quintessentia_unpack_affine_q46",
    b"quintessentia_quant_dequant_affine_q46",
)

_LOCK = threading.Lock()
_DEVICE_MODULES: dict[int, dict[bytes, ctypes.c_void_p]] = {}


def _compile_or_load_ptx(major: int, minor: int, nvrtc) -> bytes:
    src_bytes = _Q1_CUDA_SOURCE.encode("utf-8")
    digest = hashlib.sha256(src_bytes).hexdigest()[:16]
    PTX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = PTX_CACHE_DIR / f"quintessentia_q1_cubes_sm{major}{minor}_{digest}.ptx"
    if cache_file.is_file():
        data = cache_file.read_bytes()
        return data if data.endswith(b"\x00") else data + b"\x00"

    prog = ctypes.c_void_p()
    rc = nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src_bytes, b"quintessentia_q1_cubes.cu", 0, None, None)
    if rc != 0:
        raise RuntimeError(f"nvrtcCreateProgram failed with error {rc}")
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
                f"NVRTC Q1 compile failed ({rc}):\n{log_buf.value.decode('utf-8', errors='replace')}"
            )
        ptx_sz = ctypes.c_size_t()
        nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_sz))
        ptx_buf = ctypes.create_string_buffer(ptx_sz.value)
        rc = nvrtc.nvrtcGetPTX(prog, ptx_buf)
        if rc != 0:
            raise RuntimeError(f"nvrtcGetPTX failed with error {rc}")
        raw_ptx = ptx_buf.raw
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    tmp = cache_file.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(raw_ptx)
    os.replace(tmp, cache_file)
    return raw_ptx


def get_q1_kernels(device: torch.device):
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
            ptx = _compile_or_load_ptx(major, minor, nvrtc)
            ptx_buf = ctypes.create_string_buffer(ptx)
            mod = ctypes.c_void_p()
            rc = nvcuda.cuModuleLoadData(ctypes.byref(mod), ctypes.cast(ptx_buf, ctypes.c_void_p))
            if rc != 0:
                raise RuntimeError(f"cuModuleLoadData failed with CUresult={rc}")
            funcs: dict[bytes, ctypes.c_void_p] = {"_module": mod}
            for name in _KERNEL_NAMES:
                fn = ctypes.c_void_p()
                rc = nvcuda.cuModuleGetFunction(ctypes.byref(fn), mod, name)
                if rc != 0:
                    raise RuntimeError(f"cuModuleGetFunction({name.decode()}) failed with CUresult={rc}")
                funcs[name] = fn
            _DEVICE_MODULES[dev_idx] = funcs
            return nvcuda, funcs


def launch_q1_plane_gemv(
    x_bf16: torch.Tensor,
    bits_i32: torch.Tensor,
    qscale: torch.Tensor,
    qzero: torch.Tensor,
    y_acc: torch.Tensor,
    *,
    r0: int,
    c0: int,
    c_actual: int,
    group_size: int,
    alpha: float,
    add_zero: bool,
    use_popc: bool = False,
) -> None:
    """Launch either exact __ffs or bit-sliced __popc Q1 kernel for one 2D cube."""
    dev = x_bf16.device
    nvcuda, fns = get_q1_kernels(dev)
    M, K_full = int(x_bf16.shape[0]), int(x_bf16.shape[1])
    N_full = int(y_acc.shape[1])
    r_len, c_words = int(bits_i32.shape[0]), int(bits_i32.shape[1])
    c_pad = c_words * 32
    num_groups = c_pad // group_size

    p_x = ctypes.c_void_p(x_bf16.data_ptr())
    p_b = ctypes.c_void_p(bits_i32.data_ptr())
    p_s = ctypes.c_void_p(qscale.data_ptr())
    p_z = ctypes.c_void_p(qzero.data_ptr())
    p_y = ctypes.c_void_p(y_acc.data_ptr())

    c_M, c_N, c_K = ctypes.c_int(M), ctypes.c_int(N_full), ctypes.c_int(K_full)
    c_r0, c_rl, c_c0 = ctypes.c_int(r0), ctypes.c_int(r_len), ctypes.c_int(c0)
    c_ca, c_cw, c_gs = ctypes.c_int(c_actual), ctypes.c_int(c_words), ctypes.c_int(group_size)
    c_al, c_az = ctypes.c_float(float(alpha)), ctypes.c_int(1 if add_zero else 0)

    args = (ctypes.c_void_p * 16)(
        ctypes.addressof(p_x), ctypes.addressof(p_b),
        ctypes.addressof(p_s), ctypes.addressof(p_z), ctypes.addressof(p_y),
        ctypes.addressof(c_M), ctypes.addressof(c_N), ctypes.addressof(c_K),
        ctypes.addressof(c_r0), ctypes.addressof(c_rl), ctypes.addressof(c_c0),
        ctypes.addressof(c_ca), ctypes.addressof(c_cw), ctypes.addressof(c_gs),
        ctypes.addressof(c_al), ctypes.addressof(c_az),
    )

    if use_popc:
        eff_cw = min(c_words, 64)
        eff_cpad = eff_cw * 32
        eff_ng = eff_cpad // group_size
        smem = eff_cpad * 4 + eff_ng * 12 + 8 * eff_cw * 4 + eff_cpad
        fn_key = b"q1_cube_popc_gemv_f32"
    else:
        eff_cw = min(c_words, 128)
        eff_cpad = eff_cw * 32
        eff_ng = eff_cpad // group_size
        smem = (eff_cpad + eff_ng) * 4
        fn_key = b"q1_cube_plane_gemv_f32"

    grid_x = (r_len + 7) // 8
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    rc = nvcuda.cuLaunchKernel(fns[fn_key], grid_x, M, 1, 256, 1, 1, smem, stream, args, None)
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel({fn_key.decode()}) failed with CUresult={rc}")


def launch_q1_multiplane_gemv(
    x_bf16: torch.Tensor,
    planes_i32: torch.Tensor,
    qscale: torch.Tensor,
    qzero: torch.Tensor,
    y_acc: torch.Tensor,
    *,
    r0: int,
    c0: int,
    c_actual: int,
    group_size: int,
    total_planes: int,
    add_zero: bool,
    use_popc: bool = False,
    midpoint_correct: bool = True,
) -> None:
    """Launch fused multi-plane Q1 kernel for [P_active, r_len, c_words] on GPU."""
    dev = x_bf16.device
    nvcuda, fns = get_q1_kernels(dev)
    M, K_full = int(x_bf16.shape[0]), int(x_bf16.shape[1])
    N_full = int(y_acc.shape[1])
    P_active, r_len, c_words = (
        int(planes_i32.shape[0]),
        int(planes_i32.shape[1]),
        int(planes_i32.shape[2]),
    )
    c_pad = c_words * 32
    num_groups = c_pad // group_size

    p_x = ctypes.c_void_p(x_bf16.data_ptr())
    p_p = ctypes.c_void_p(planes_i32.data_ptr())
    p_s = ctypes.c_void_p(qscale.data_ptr())
    p_z = ctypes.c_void_p(qzero.data_ptr())
    p_y = ctypes.c_void_p(y_acc.data_ptr())

    c_M, c_N, c_K = ctypes.c_int(M), ctypes.c_int(N_full), ctypes.c_int(K_full)
    c_r0, c_rl, c_c0 = ctypes.c_int(r0), ctypes.c_int(r_len), ctypes.c_int(c0)
    c_ca, c_cw, c_gs = ctypes.c_int(c_actual), ctypes.c_int(c_words), ctypes.c_int(group_size)
    c_pa, c_tp = ctypes.c_int(P_active), ctypes.c_int(total_planes)
    c_az, c_mc = ctypes.c_int(1 if add_zero else 0), ctypes.c_int(1 if midpoint_correct else 0)

    args = (ctypes.c_void_p * 18)(
        ctypes.addressof(p_x), ctypes.addressof(p_p),
        ctypes.addressof(p_s), ctypes.addressof(p_z), ctypes.addressof(p_y),
        ctypes.addressof(c_M), ctypes.addressof(c_N), ctypes.addressof(c_K),
        ctypes.addressof(c_r0), ctypes.addressof(c_rl), ctypes.addressof(c_c0),
        ctypes.addressof(c_ca), ctypes.addressof(c_cw), ctypes.addressof(c_gs),
        ctypes.addressof(c_pa), ctypes.addressof(c_tp),
        ctypes.addressof(c_az), ctypes.addressof(c_mc),
    )

    if use_popc:
        eff_cw = min(c_words, 64)
        eff_cpad = eff_cw * 32
        eff_ng = eff_cpad // group_size
        smem = eff_cpad * 4 + eff_ng * 12 + 8 * eff_cw * 4 + eff_cpad
        fn_key = b"q1_cube_popc_multiplane_gemv_f32"
    else:
        eff_cw = min(c_words, 128)
        eff_cpad = eff_cw * 32
        eff_ng = eff_cpad // group_size
        smem = (eff_cpad + eff_ng) * 4
        fn_key = b"q1_cube_multiplane_gemv_f32"

    grid_x = (r_len + 7) // 8
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    rc = nvcuda.cuLaunchKernel(fns[fn_key], grid_x, M, 1, 256, 1, 1, smem, stream, args, None)
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel({fn_key.decode()}) failed with CUresult={rc}")


def launch_q1_adaptive_radar_gemv(
    x_bf16: torch.Tensor,
    planes_u32: torch.Tensor,
    qscale: torch.Tensor,
    qzero: torch.Tensor,
    tile_depths: torch.Tensor,
    y_bf16: torch.Tensor,
    *,
    r_tile: int,
    c_tile: int,
    group_size: int,
    k_max: int,
) -> None:
    """Single-launch adaptive-depth 2D-tiled Q1 radar GEMV kernel."""
    dev = x_bf16.device
    nvcuda, fns = get_q1_kernels(dev)
    M, K = int(x_bf16.shape[0]), int(x_bf16.shape[1])
    N = int(y_bf16.shape[1])
    num_groups = K // group_size

    p_x = ctypes.c_void_p(x_bf16.data_ptr())
    p_p = ctypes.c_void_p(planes_u32.data_ptr())
    p_s = ctypes.c_void_p(qscale.data_ptr())
    p_z = ctypes.c_void_p(qzero.data_ptr())
    p_d = ctypes.c_void_p(tile_depths.data_ptr())
    p_y = ctypes.c_void_p(y_bf16.data_ptr())

    c_M, c_N, c_K = ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)
    c_rt, c_ct = ctypes.c_int(r_tile), ctypes.c_int(c_tile)
    c_gs, c_km = ctypes.c_int(group_size), ctypes.c_int(k_max)

    args = (ctypes.c_void_p * 13)(
        ctypes.addressof(p_x), ctypes.addressof(p_p),
        ctypes.addressof(p_s), ctypes.addressof(p_z),
        ctypes.addressof(p_d), ctypes.addressof(p_y),
        ctypes.addressof(c_M), ctypes.addressof(c_N), ctypes.addressof(c_K),
        ctypes.addressof(c_rt), ctypes.addressof(c_ct),
        ctypes.addressof(c_gs), ctypes.addressof(c_km),
    )

    n_ctiles = K // c_tile
    tiles_per_chunk = max(1, min(n_ctiles, 4096 // c_tile))
    cap_cols = tiles_per_chunk * c_tile
    cap_groups = cap_cols // group_size
    smem = (cap_cols + cap_groups) * 4
    grid_x = (N + 7) // 8
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    rc = nvcuda.cuLaunchKernel(
        fns[b"q1_cube_adaptive_radar_gemv_f32"],
        grid_x, M, 1, 256, 1, 1, smem, stream, args, None,
    )
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel(q1_cube_adaptive_radar_gemv_f32) failed with CUresult={rc}")


def launch_q1_packed_trunc_gemv(
    x_bf16: torch.Tensor,
    packed_lo: torch.Tensor | None,
    packed_hi: torch.Tensor | None,
    qscale: torch.Tensor,
    qzero: torch.Tensor,
    y_out: torch.Tensor,
    *,
    group_size: int,
    bits: int,
    p_active: int,
    row_scale: torch.Tensor | None = None,
    col_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    midpoint_correct: bool = True,
) -> None:
    """Direct zero-extra-VRAM NVRTC GEMV reading packed_lo/packed_hi (packed_hi-only when bits==6, P<=2)."""
    dev = x_bf16.device
    nvcuda, fns = get_q1_kernels(dev)
    M, K = int(x_bf16.shape[0]), int(x_bf16.shape[1])
    N = int(y_out.shape[1])

    hi_only = (bits == 6 and p_active <= 2)
    out_is_bf16 = 1 if y_out.dtype == torch.bfloat16 else 0
    p_x = ctypes.c_void_p(x_bf16.data_ptr())
    p_lo = ctypes.c_void_p(0 if (hi_only or packed_lo is None) else packed_lo.data_ptr())
    p_hi = ctypes.c_void_p(0 if packed_hi is None else packed_hi.data_ptr())
    p_s = ctypes.c_void_p(qscale.data_ptr())
    p_z = ctypes.c_void_p(qzero.data_ptr())
    p_rs = ctypes.c_void_p(0 if row_scale is None else row_scale.data_ptr())
    p_cs = ctypes.c_void_p(0 if col_scale is None else col_scale.data_ptr())
    p_bias = ctypes.c_void_p(0 if bias is None else bias.data_ptr())
    p_y = ctypes.c_void_p(y_out.data_ptr())

    c_M, c_N, c_K, c_gs = ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(group_size)
    c_b, c_pa = ctypes.c_int(bits), ctypes.c_int(p_active)
    c_mc, c_ob = ctypes.c_int(1 if midpoint_correct else 0), ctypes.c_int(out_is_bf16)

    args = (ctypes.c_void_p * 17)(
        ctypes.addressof(p_x), ctypes.addressof(p_lo), ctypes.addressof(p_hi),
        ctypes.addressof(p_s), ctypes.addressof(p_z),
        ctypes.addressof(p_rs), ctypes.addressof(p_cs),
        ctypes.addressof(p_bias), ctypes.addressof(p_y),
        ctypes.addressof(c_M), ctypes.addressof(c_N), ctypes.addressof(c_K), ctypes.addressof(c_gs),
        ctypes.addressof(c_b), ctypes.addressof(c_pa), ctypes.addressof(c_mc), ctypes.addressof(c_ob),
    )

    chunk_cap = min(K, 4096)
    smem = (chunk_cap + chunk_cap // group_size) * 4
    grid_x = (N + 7) // 8
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    rc = nvcuda.cuLaunchKernel(
        fns[b"q1_packed_trunc_gemv_f32"],
        grid_x, M, 1, 256, 1, 1, smem, stream, args, None,
    )
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel(q1_packed_trunc_gemv_f32) failed with CUresult={rc}")


def launch_ideal_q1_fwht_gemv(
    x_bf16: torch.Tensor,
    packed_lo: torch.Tensor,
    qscale: torch.Tensor,
    qzero: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor,
    y_out: torch.Tensor,
    *,
    group_size: int,
    total_planes: int,
    plane_start: int = 0,
    plane_count: int | None = None,
    cand_rows: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    accumulate: bool = False,
) -> None:
    """Launch Ideal Q1 FWHT + Sign Planes GEMV (#7), supporting cand_rows Early-Stop."""
    dev = x_bf16.device
    nvcuda, fns = get_q1_kernels(dev)
    M, K = int(x_bf16.shape[0]), int(x_bf16.shape[1])
    N = int(y_out.shape[1])
    if M == 0:
        return
    p_cnt = (total_planes - plane_start) if plane_count is None else int(plane_count)
    if p_cnt <= 0:
        return
    cand_cnt = int(cand_rows.numel()) if cand_rows is not None else 0
    out_rows = cand_cnt if cand_cnt > 0 else N
    out_is_bf16 = 1 if y_out.dtype == torch.bfloat16 else 0

    p_x = ctypes.c_void_p(x_bf16.data_ptr())
    p_lo = ctypes.c_void_p(packed_lo.data_ptr())
    p_s = ctypes.c_void_p(qscale.data_ptr())
    p_z = ctypes.c_void_p(qzero.data_ptr())
    p_rs = ctypes.c_void_p(row_scale.data_ptr())
    p_cs = ctypes.c_void_p(col_scale.data_ptr())
    p_cand = ctypes.c_void_p(0 if cand_rows is None else cand_rows.data_ptr())
    p_bias = ctypes.c_void_p(0 if bias is None else bias.data_ptr())
    p_y = ctypes.c_void_p(y_out.data_ptr())

    c_M, c_N, c_K, c_gs = ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(group_size)
    c_pt, c_ps, c_pc = ctypes.c_int(total_planes), ctypes.c_int(plane_start), ctypes.c_int(p_cnt)
    c_cc, c_acc, c_ob = ctypes.c_int(cand_cnt), ctypes.c_int(1 if accumulate else 0), ctypes.c_int(out_is_bf16)
    shared_sc = 1 if (qscale.ndim == 2 or (qscale.ndim == 3 and int(qscale.shape[0]) == 1 and (total_planes > 1 or plane_start > 0))) else 0
    c_ss = ctypes.c_int(shared_sc)

    args = (ctypes.c_void_p * 20)(
        ctypes.addressof(p_x), ctypes.addressof(p_lo),
        ctypes.addressof(p_s), ctypes.addressof(p_z),
        ctypes.addressof(p_rs), ctypes.addressof(p_cs),
        ctypes.addressof(p_cand), ctypes.addressof(p_bias), ctypes.addressof(p_y),
        ctypes.addressof(c_M), ctypes.addressof(c_N), ctypes.addressof(c_K), ctypes.addressof(c_gs),
        ctypes.addressof(c_pt), ctypes.addressof(c_ps), ctypes.addressof(c_pc),
        ctypes.addressof(c_cc), ctypes.addressof(c_acc), ctypes.addressof(c_ob), ctypes.addressof(c_ss),
    )

    chunk_cap = min(K, 4096)
    smem = (chunk_cap + chunk_cap // group_size) * 4
    grid_x = (out_rows + 7) // 8
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    rc = nvcuda.cuLaunchKernel(
        fns[b"ideal_q1_fwht_gemv_f32"],
        grid_x, M, 1, 256, 1, 1, smem, stream, args, None,
    )
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel(ideal_q1_fwht_gemv_f32) failed with CUresult={rc}")


def launch_b1_mma_raw_verify(a_bits: torch.Tensor, b_bits: torch.Tensor, c_out: torch.Tensor) -> None:
    """Verify exact Ampere PTX mma.sync.m8n8k128.b1.b1.s32.and.popc fragment layout on [8, 4] u32."""
    dev = a_bits.device
    nvcuda, fns = get_q1_kernels(dev)
    p_a = ctypes.c_void_p(a_bits.data_ptr())
    p_b = ctypes.c_void_p(b_bits.data_ptr())
    p_c = ctypes.c_void_p(c_out.data_ptr())
    args = (ctypes.c_void_p * 3)(ctypes.addressof(p_a), ctypes.addressof(p_b), ctypes.addressof(p_c))
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    rc = nvcuda.cuLaunchKernel(
        fns[b"b1_mma_m8n8k128_verify_u32"],
        1, 1, 1, 32, 1, 1, 0, stream, args, None,
    )
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel(b1_mma_m8n8k128_verify_u32) failed with CUresult={rc}")


def launch_ideal_q1_b1_mma(
    x_bf16: torch.Tensor,
    packed_lo: torch.Tensor,
    qscale: torch.Tensor,
    qzero: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor,
    y_out: torch.Tensor,
    *,
    group_size: int,
    total_planes: int,
    active_planes: int | None = None,
    bias: torch.Tensor | None = None,
) -> None:
    """Launch Ideal Q1 1-bit Tensor Core MMA GEMM (#8) for batched M >= 1."""
    dev = x_bf16.device
    nvcuda, fns = get_q1_kernels(dev)
    M, K = int(x_bf16.shape[0]), int(x_bf16.shape[1])
    N = int(y_out.shape[1])
    if M == 0:
        return
    p_act = total_planes if active_planes is None else int(active_planes)
    out_is_bf16 = 1 if y_out.dtype == torch.bfloat16 else 0

    p_x = ctypes.c_void_p(x_bf16.data_ptr())
    p_lo = ctypes.c_void_p(packed_lo.data_ptr())
    p_s = ctypes.c_void_p(qscale.data_ptr())
    p_z = ctypes.c_void_p(qzero.data_ptr())
    p_rs = ctypes.c_void_p(row_scale.data_ptr())
    p_cs = ctypes.c_void_p(col_scale.data_ptr())
    p_bias = ctypes.c_void_p(0 if bias is None else bias.data_ptr())
    p_y = ctypes.c_void_p(y_out.data_ptr())

    c_M, c_N, c_K, c_gs = ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(group_size)
    c_pt, c_pa, c_ob = ctypes.c_int(total_planes), ctypes.c_int(p_act), ctypes.c_int(out_is_bf16)
    shared_sc = 1 if (qscale.ndim == 2 or (qscale.ndim == 3 and int(qscale.shape[0]) == 1 and total_planes > 1)) else 0
    c_ss = ctypes.c_int(shared_sc)

    args = (ctypes.c_void_p * 16)(
        ctypes.addressof(p_x), ctypes.addressof(p_lo),
        ctypes.addressof(p_s), ctypes.addressof(p_z),
        ctypes.addressof(p_rs), ctypes.addressof(p_cs),
        ctypes.addressof(p_bias), ctypes.addressof(p_y),
        ctypes.addressof(c_M), ctypes.addressof(c_N), ctypes.addressof(c_K), ctypes.addressof(c_gs),
        ctypes.addressof(c_pt), ctypes.addressof(c_pa), ctypes.addressof(c_ob), ctypes.addressof(c_ss),
    )

    k_cap = 256
    ng_cap = k_cap // group_size
    smem = (8 * k_cap) * 4 + 3 * (8 * ng_cap) * 4 + 512 * 4 + (8 * k_cap)
    grid_x = (N + 63) // 64
    grid_y = (M + 7) // 8
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    rc = nvcuda.cuLaunchKernel(
        fns[b"ideal_q1_b1_mma_gemm_f32"],
        grid_x, grid_y, 1, 256, 1, 1, smem, stream, args, None,
    )
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel(ideal_q1_b1_mma_gemm_f32) failed with CUresult={rc}")


def launch_quintessentia_q2_fwht_gemv(
    x_bf16: torch.Tensor,
    packed_lo: torch.Tensor,
    qscale: torch.Tensor,
    qzero: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor,
    y_out: torch.Tensor,
    *,
    group_size: int,
    bias: torch.Tensor | None = None,
) -> None:
    """Launch Ideal Q2 (2-bit + FWHT + Sinkhorn) GEMV (#9)."""
    dev = x_bf16.device
    nvcuda, fns = get_q1_kernels(dev)
    M, K = int(x_bf16.shape[0]), int(x_bf16.shape[1])
    N = int(y_out.shape[1])
    if M == 0:
        return
    out_is_bf16 = 1 if y_out.dtype == torch.bfloat16 else 0

    p_x = ctypes.c_void_p(x_bf16.data_ptr())
    p_lo = ctypes.c_void_p(packed_lo.data_ptr())
    p_s = ctypes.c_void_p(qscale.data_ptr())
    p_z = ctypes.c_void_p(qzero.data_ptr())
    p_rs = ctypes.c_void_p(row_scale.data_ptr())
    p_cs = ctypes.c_void_p(col_scale.data_ptr())
    p_bias = ctypes.c_void_p(0 if bias is None else bias.data_ptr())
    p_y = ctypes.c_void_p(y_out.data_ptr())

    c_M, c_N, c_K, c_gs = ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(group_size)
    c_ob = ctypes.c_int(out_is_bf16)

    args = (ctypes.c_void_p * 13)(
        ctypes.addressof(p_x), ctypes.addressof(p_lo),
        ctypes.addressof(p_s), ctypes.addressof(p_z),
        ctypes.addressof(p_rs), ctypes.addressof(p_cs),
        ctypes.addressof(p_bias), ctypes.addressof(p_y),
        ctypes.addressof(c_M), ctypes.addressof(c_N), ctypes.addressof(c_K), ctypes.addressof(c_gs),
        ctypes.addressof(c_ob),
    )

    chunk_cap = min(K, 4096)
    smem = (chunk_cap + chunk_cap // group_size) * 4
    grid_x = (N + 7) // 8
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    rc = nvcuda.cuLaunchKernel(
        fns[b"quintessentia_q2_fwht_gemv_f32"],
        grid_x, M, 1, 256, 1, 1, smem, stream, args, None,
    )
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel(quintessentia_q2_fwht_gemv_f32) failed with CUresult={rc}")


def launch_quintessentia_q8_gemv(
    x_bf16: torch.Tensor,
    packed_lo: torch.Tensor,
    qscale: torch.Tensor,
    qzero: torch.Tensor,
    y_out: torch.Tensor,
    *,
    group_size: int,
    bias: torch.Tensor | None = None,
) -> None:
    """Launch Quintessentia Q8 (8-bit group affine) GEMV (#10)."""
    dev = x_bf16.device
    nvcuda, fns = get_q1_kernels(dev)
    M, K = int(x_bf16.shape[0]), int(x_bf16.shape[1])
    N = int(y_out.shape[1])
    if M == 0:
        return
    out_is_bf16 = 1 if y_out.dtype == torch.bfloat16 else 0

    p_x = ctypes.c_void_p(x_bf16.data_ptr())
    p_lo = ctypes.c_void_p(packed_lo.data_ptr())
    p_s = ctypes.c_void_p(qscale.data_ptr())
    p_z = ctypes.c_void_p(qzero.data_ptr())
    p_bias = ctypes.c_void_p(0 if bias is None else bias.data_ptr())
    p_y = ctypes.c_void_p(y_out.data_ptr())

    c_M, c_N, c_K, c_gs = ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(group_size)
    c_ob = ctypes.c_int(out_is_bf16)

    args = (ctypes.c_void_p * 11)(
        ctypes.addressof(p_x), ctypes.addressof(p_lo),
        ctypes.addressof(p_s), ctypes.addressof(p_z),
        ctypes.addressof(p_bias), ctypes.addressof(p_y),
        ctypes.addressof(c_M), ctypes.addressof(c_N), ctypes.addressof(c_K), ctypes.addressof(c_gs),
        ctypes.addressof(c_ob),
    )

    chunk_cap = min(K, 4096)
    smem = (chunk_cap + chunk_cap // group_size) * 4
    grid_x = (N + 7) // 8
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    rc = nvcuda.cuLaunchKernel(
        fns[b"quintessentia_q8_gemv_f32"],
        grid_x, M, 1, 256, 1, 1, smem, stream, args, None,
    )
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel(quintessentia_q8_gemv_f32) failed with CUresult={rc}")


def launch_puzzle_overlay_delta_cubes(
    x_bf16: torch.Tensor,
    delta_q8: torch.Tensor,
    delta_scale: torch.Tensor,
    delta_zero: torch.Tensor,
    cube_coords_i32: torch.Tensor,
    y_acc_f32: torch.Tensor,
    *,
    group_size: int = 64,
) -> None:
    """Launch sparse Q8 delta-cube accumulation (#11) into y_acc_f32 [M, N] via atomicAdd."""
    M, K = int(x_bf16.shape[0]), int(x_bf16.shape[1])
    N = int(y_acc_f32.shape[1])
    M_sel = int(delta_q8.shape[0])
    if M == 0 or M_sel == 0:
        return
    R_tile, C_tile = int(delta_q8.shape[1]), int(delta_q8.shape[2])
    if group_size not in (16, 32, 64):
        raise ValueError(f"group_size must be in {{16, 32, 64}}, got {group_size}")
    if C_tile % group_size != 0:
        raise ValueError(f"C_tile ({C_tile}) must be divisible by group_size ({group_size})")
    if not y_acc_f32.is_contiguous() or y_acc_f32.dtype != torch.float32:
        raise ValueError("y_acc_f32 must be contiguous float32")

    x_bf16 = x_bf16.contiguous()
    delta_q8 = delta_q8.contiguous()
    delta_scale = delta_scale.contiguous()
    delta_zero = delta_zero.contiguous()
    cube_coords_i32 = cube_coords_i32.to(dtype=torch.int32).contiguous()

    dev = x_bf16.device
    nvcuda, fns = get_q1_kernels(dev)

    p_x = ctypes.c_void_p(x_bf16.data_ptr())
    p_q = ctypes.c_void_p(delta_q8.data_ptr())
    p_s = ctypes.c_void_p(delta_scale.data_ptr())
    p_z = ctypes.c_void_p(delta_zero.data_ptr())
    p_coords = ctypes.c_void_p(cube_coords_i32.data_ptr())
    p_y = ctypes.c_void_p(y_acc_f32.data_ptr())

    c_M = ctypes.c_int(M)
    c_N = ctypes.c_int(N)
    c_K = ctypes.c_int(K)
    c_Msel = ctypes.c_int(M_sel)
    c_Rt = ctypes.c_int(R_tile)
    c_Ct = ctypes.c_int(C_tile)
    c_gs = ctypes.c_int(group_size)

    args = (ctypes.c_void_p * 13)(
        ctypes.addressof(p_x),
        ctypes.addressof(p_q),
        ctypes.addressof(p_s),
        ctypes.addressof(p_z),
        ctypes.addressof(p_coords),
        ctypes.addressof(p_y),
        ctypes.addressof(c_M),
        ctypes.addressof(c_N),
        ctypes.addressof(c_K),
        ctypes.addressof(c_Msel),
        ctypes.addressof(c_Rt),
        ctypes.addressof(c_Ct),
        ctypes.addressof(c_gs),
    )

    chunk_cap = min(C_tile, 4096)
    smem = (chunk_cap + chunk_cap // group_size) * 4
    grid_x = M_sel * ((R_tile + 7) // 8)
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    rc = nvcuda.cuLaunchKernel(
        fns[b"puzzle_overlay_delta_cubes_f32"],
        grid_x, M, 1, 256, 1, 1, smem, stream, args, None,
    )
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel(puzzle_overlay_delta_cubes_f32) failed with CUresult={rc}")


def launch_quintessentia_pack_affine(
    src: torch.Tensor,
    packed_lo: torch.Tensor,
    packed_hi: torch.Tensor | None,
    qscale: torch.Tensor,
    qzero: torch.Tensor,
    *,
    bits: int,
    group_size: int = 64,
) -> None:
    """Fused NVRTC affine Q4/Q6/Q8 weight packer (#12), bit-exact with CPU FP64 pack_weight."""
    if bits not in (4, 6, 8) or group_size not in (16, 32, 64):
        raise ValueError(f"Unsupported bits={bits}, group_size={group_size}")
    if src.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"src dtype must be bfloat16 or float32, got {src.dtype}")
    total_elems = int(src.numel())
    if total_elems == 0:
        return
    if total_elems % group_size != 0:
        raise ValueError(f"src numel ({total_elems}) not divisible by group_size ({group_size})")
    src = src.contiguous()
    if not (packed_lo.is_contiguous() and qscale.is_contiguous() and qzero.is_contiguous()):
        raise ValueError("Output buffers must be contiguous")
    if bits == 6 and (packed_hi is None or not packed_hi.is_contiguous()):
        raise ValueError("packed_hi must be a contiguous CUDA tensor for bits=6")

    total_groups = total_elems // group_size
    groups_per_block = 256 // group_size
    grid_x = (total_groups + groups_per_block - 1) // groups_per_block

    dev = src.device
    nvcuda, fns = get_q1_kernels(dev)

    p_src = ctypes.c_void_p(src.data_ptr())
    p_lo = ctypes.c_void_p(packed_lo.data_ptr())
    p_hi = ctypes.c_void_p(packed_hi.data_ptr() if packed_hi is not None else 0)
    p_s = ctypes.c_void_p(qscale.data_ptr())
    p_z = ctypes.c_void_p(qzero.data_ptr())
    c_tg = ctypes.c_longlong(total_groups)
    c_gs = ctypes.c_int(group_size)
    c_bits = ctypes.c_int(bits)
    c_bf16 = ctypes.c_int(1 if src.dtype == torch.bfloat16 else 0)

    args = (ctypes.c_void_p * 9)(
        ctypes.addressof(p_src),
        ctypes.addressof(p_lo),
        ctypes.addressof(p_hi),
        ctypes.addressof(p_s),
        ctypes.addressof(p_z),
        ctypes.addressof(c_tg),
        ctypes.addressof(c_gs),
        ctypes.addressof(c_bits),
        ctypes.addressof(c_bf16),
    )
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    rc = nvcuda.cuLaunchKernel(
        fns[b"quintessentia_pack_affine_q46"],
        grid_x, 1, 1, 256, 1, 1, 0, stream, args, None,
    )
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel(quintessentia_pack_affine_q46) failed with CUresult={rc}")


def launch_quintessentia_unpack_affine(
    packed_lo: torch.Tensor,
    packed_hi: torch.Tensor | None,
    qscale: torch.Tensor,
    qzero: torch.Tensor,
    dst: torch.Tensor,
    *,
    bits: int,
    group_size: int = 64,
) -> None:
    """Fused NVRTC affine Q4/Q6/Q8 weight unpacker (#13), bit-exact with scalar reference."""
    if bits not in (4, 6, 8) or group_size not in (16, 32, 64):
        raise ValueError(f"Unsupported bits={bits}, group_size={group_size}")
    if dst.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"dst dtype must be bfloat16 or float32, got {dst.dtype}")
    total_elems = int(dst.numel())
    if total_elems == 0:
        return
    packed_lo = packed_lo.contiguous()
    if packed_hi is not None:
        packed_hi = packed_hi.contiguous()
    qscale = qscale.contiguous()
    qzero = qzero.contiguous()
    if not dst.is_contiguous():
        raise ValueError("dst must be contiguous")

    gs_shift = {16: 4, 32: 5, 64: 6}[group_size]
    grid_x = (total_elems + 255) // 256

    dev = dst.device
    nvcuda, fns = get_q1_kernels(dev)

    p_lo = ctypes.c_void_p(packed_lo.data_ptr())
    p_hi = ctypes.c_void_p(packed_hi.data_ptr() if packed_hi is not None else 0)
    p_s = ctypes.c_void_p(qscale.data_ptr())
    p_z = ctypes.c_void_p(qzero.data_ptr())
    p_dst = ctypes.c_void_p(dst.data_ptr())
    c_te = ctypes.c_longlong(total_elems)
    c_shift = ctypes.c_int(gs_shift)
    c_bits = ctypes.c_int(bits)
    c_bf16 = ctypes.c_int(1 if dst.dtype == torch.bfloat16 else 0)

    args = (ctypes.c_void_p * 9)(
        ctypes.addressof(p_lo),
        ctypes.addressof(p_hi),
        ctypes.addressof(p_s),
        ctypes.addressof(p_z),
        ctypes.addressof(p_dst),
        ctypes.addressof(c_te),
        ctypes.addressof(c_shift),
        ctypes.addressof(c_bits),
        ctypes.addressof(c_bf16),
    )
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    rc = nvcuda.cuLaunchKernel(
        fns[b"quintessentia_unpack_affine_q46"],
        grid_x, 1, 1, 256, 1, 1, 0, stream, args, None,
    )
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel(quintessentia_unpack_affine_q46) failed with CUresult={rc}")


def launch_quintessentia_quant_dequant_affine(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    bits: int,
    group_size: int = 64,
) -> None:
    """Zero-alloc fused NVRTC quant+dequant (#14, safe in-place when src is dst)."""
    if bits not in (4, 6, 8) or group_size not in (16, 32, 64):
        raise ValueError(f"Unsupported bits={bits}, group_size={group_size}")
    if src.dtype not in (torch.bfloat16, torch.float32) or dst.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported dtypes src={src.dtype}, dst={dst.dtype}")
    total_elems = int(src.numel())
    if total_elems == 0:
        return
    if total_elems % group_size != 0 or int(dst.numel()) != total_elems:
        raise ValueError("Mismatched or unaligned tensor sizes")
    src = src.contiguous()
    if not dst.is_contiguous():
        raise ValueError("dst must be contiguous")

    total_groups = total_elems // group_size
    groups_per_block = 256 // group_size
    grid_x = (total_groups + groups_per_block - 1) // groups_per_block

    dev = src.device
    nvcuda, fns = get_q1_kernels(dev)

    p_src = ctypes.c_void_p(src.data_ptr())
    p_dst = ctypes.c_void_p(dst.data_ptr())
    c_tg = ctypes.c_longlong(total_groups)
    c_gs = ctypes.c_int(group_size)
    c_bits = ctypes.c_int(bits)
    c_sbf16 = ctypes.c_int(1 if src.dtype == torch.bfloat16 else 0)
    c_dbf16 = ctypes.c_int(1 if dst.dtype == torch.bfloat16 else 0)

    args = (ctypes.c_void_p * 7)(
        ctypes.addressof(p_src),
        ctypes.addressof(p_dst),
        ctypes.addressof(c_tg),
        ctypes.addressof(c_gs),
        ctypes.addressof(c_bits),
        ctypes.addressof(c_sbf16),
        ctypes.addressof(c_dbf16),
    )
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    rc = nvcuda.cuLaunchKernel(
        fns[b"quintessentia_quant_dequant_affine_q46"],
        grid_x, 1, 1, 256, 1, 1, 0, stream, args, None,
    )
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel(quintessentia_quant_dequant_affine_q46) failed with CUresult={rc}")

