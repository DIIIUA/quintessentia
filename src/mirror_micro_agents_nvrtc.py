"""Deep Micro-Agent NVRTC CUDA Kernels (Mirror Equality x UNI Fractal Topology).

Two strictly complementary GPU engines operating at the micro-agent scale (tens of thousands
to 1,000,000+ elementary agents inside pure GPU silicon, zero Python micro-loop):

1. `micro_agent_q1_group64_reader_f32` (STRICTLY READ-ONLY over pretrained LLM weights):
   - Never alters model weights. Changes ONLY how Q1 bit-planes are read on GPU.
   - Every single 64-weight group `(row, g)` is an independent micro-agent (`N * (K / 64)` agents
     per projection -> hundreds of millions / billions across an LLM).
   - Inside a warp (32 lanes = 1 row), each group-64 agent maintains its own Mirror Equality
     (`g_strategy = g_outcome + g_obs`), shares row-level energy across its warp micro-ring via
     1-cycle register `__shfl_xor_sync`, applies UNI binding (`Phi, Omega, Sigma`), and decides
     locally whether to stop reading deeper Q1 planes for THAT specific group-64 while hot
     sibling groups in the same row continue refining!
   - Uses `__ballot_sync` to exit the entire row immediately once all 64-weight agents converge.

2. `micro_agent_million_substrate_step_f32` (Autonomous 1,000,000+ Micro-Agent Substrate):
   - 1 CUDA thread = 1 elementary agent on the shared assembly line (`line_spine[N_agents]`).
   - Given ONLY a tiny base seed (e.g., 64 agents out of 1,048,576), active agents freely turn
     along arbitrary neighbor links, intersect `line_spine[v]` via `atomicAdd`, evaluate Mirror x
     UNI binding defect, and use warp-aggregated `__ballot_sync` + `__popc` + `atomicCAS` to
     self-spawn new micro-rings directly in GPU memory until the whole million-agent fabric settles.
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import os
import threading
from typing import Dict, List, Optional, Tuple

import torch

from src.quintessentia_nvrtc import PTX_CACHE_DIR, _load_driver_and_nvrtc

_MICRO_CUDA_SOURCE = r"""
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

/* Warp-cooperative Block FWHT H_bar = H / sqrt(64) = H / 8 + group sum */
__device__ __forceinline__ void micro_fwht64_and_sum(
    float* s_x,
    float* s_xsum,
    int ng_cur,
    int warp_id,
    int lane,
    int warps_per_block
) {
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

/* Evaluate one 64-weight group's exact Q1 cube dot product from 64 packed bits (8 bytes = 2 u32) */
__device__ __forceinline__ float eval_group64_bits(
    const float* s_x_grp,
    float xsum_g,
    unsigned int lo32,
    unsigned int hi32,
    float sv
) {
    float d1 = 0.0f;
    unsigned int b0 = lo32;
    while (b0) {
        int b = __ffs(b0) - 1;
        d1 += s_x_grp[b];
        b0 &= (b0 - 1u);
    }
    unsigned int b1 = hi32;
    while (b1) {
        int b = __ffs(b1) - 1;
        d1 += s_x_grp[32 + b];
        b1 &= (b1 - 1u);
    }
    return sv * (2.0f * d1 - xsum_g);
}

__device__ __forceinline__ bool micro_substrate_wants_more(
    float state, float obs, float previous, float energy, int ng,
    int cursor, int maximum, int minimum, float tol
) {
    if (cursor >= maximum) return false;
    if (cursor < minimum || tol <= 0.0f) return true;
    float rms = sqrtf(energy / (float)ng) + 1e-8f;
    float share = state * state / (energy + 1e-12f);
    float factor = (1.0f - 0.22313016f) *
        (1.0f - fminf(1.0f, share * 4.0f) / 1.77245385f);
    factor = fmaxf(factor, 0.25f);
    float envelope = fmaxf(fabsf(obs), 0.5f * fabsf(previous));
    return 1.15470054f * envelope / (fabsf(state) + 0.5f * rms) >= tol * factor;
}

// Fusion of the contribution reader and the same substrate decision used by
// kernel B below. A warp is a dependency component (one learned output row).
// All its groups, including paused ones, reconsider the updated shared state.
__device__ void micro_coupled_warp_row(
    const unsigned short* x, const unsigned char* packed, const unsigned short* scales,
    const unsigned short* zeros, const unsigned short* rows, const unsigned short* cols,
    const unsigned short* bias, void* output, unsigned long long* reads_out,
    int* rounds_out, int N, int K, int maximum, int minimum, float tol, int bf16, int shared_scale
) {
    extern __shared__ char coupled_smem[];
    float* tx = reinterpret_cast<float*>(coupled_smem);
    int ng = K >> 6;
    float* sums = tx + K;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int c = tid; c < K; c += blockDim.x) tx[c] = bf16_to_f32(x[c]) * bf16_to_f32(cols[c]);
    __syncthreads();
    micro_fwht64_and_sum(tx, sums, ng, warp, lane, blockDim.x >> 5);
    __syncthreads();
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;
    float state[2], obs[2], previous[2];
    int cursor[2];
    bool active[2];
    unsigned int reads = 0;
    for (int a = 0; a < 2; ++a) {
        int g = lane + a * 32;
        state[a] = g < ng ? bf16_to_f32(zeros[(long long)row * ng + g]) * sums[g] : 0.0f;
        obs[a] = previous[a] = 0.0f;
        cursor[a] = 0;
        active[a] = g < ng;
    }
    int rounds = 0;
    // Each nonempty round advances at least one cursor; this is a finite
    // work bound, not a fixed cadence or a precision cutoff.
    while (__ballot_sync(0xffffffffu, active[0] || active[1])) {
        for (int a = 0; a < 2; ++a) {
            if (!active[a]) continue;
            int g = lane + a * 32, p = cursor[a];
            long long si = shared_scale ? (long long)row * ng + g : ((long long)p * N + row) * ng + g;
            float scale = bf16_to_f32(scales[si]);
            if (shared_scale) scale = ldexpf(scale, -p);
            const unsigned int* bits = reinterpret_cast<const unsigned int*>(
                packed + ((long long)p * N + row) * (K >> 3));
            float delta = eval_group64_bits(tx + g * 64, sums[g], bits[g * 2], bits[g * 2 + 1], scale);
            previous[a] = obs[a];
            obs[a] = delta;
            state[a] += delta;
            cursor[a]++;
            reads++;
        }
        float energy = state[0] * state[0] + state[1] * state[1];
        for (int d = 16; d > 0; d >>= 1) energy += __shfl_xor_sync(0xffffffffu, energy, d);
        for (int a = 0; a < 2; ++a) {
            int g = lane + a * 32;
            active[a] = g < ng && micro_substrate_wants_more(state[a],obs[a],previous[a],energy,
                ng,cursor[a],maximum,minimum,tol);
        }
        rounds++;
    }
    float result = state[0] + state[1];
    for (int d = 16; d > 0; d >>= 1) {
        result += __shfl_down_sync(0xffffffffu,result,d);
        reads += __shfl_down_sync(0xffffffffu,reads,d);
    }
    if (lane == 0) {
        result *= bf16_to_f32(rows[row]);
        if (bias != nullptr) result += bf16_to_f32(bias[row]);
        if (bf16) reinterpret_cast<unsigned short*>(output)[row] = f32_to_bf16_rne(result);
        else reinterpret_cast<float*>(output)[row] = result;
        if (reads_out != nullptr) atomicAdd(reads_out,(unsigned long long)reads);
        if (rounds_out != nullptr) rounds_out[row] = rounds;
    }
}

/* Kernel A: Read-Only Micro-Agent Group-64 Q1 Reader (1 lane = 1 group-64 micro-agent for K <= 2048,
 * or up to 4 groups per lane for K <= 8192).
 *
 * Pretrained model weights are NEVER modified. Every 64-weight group independently tracks its own
 * Mirror Equality state, communicates across its 32-agent warp micro-ring via 1-cycle __shfl_xor_sync,
 * and stops reading deeper Q1 planes from GDDR6 as soon as its own Mirror x UNI gate converges!
 *
 * Grid:  ((N + 7) / 8, M, 1)
 * Block: (256, 1, 1) -> 8 warps = 8 output rows
 */
extern "C" __global__ void micro_agent_q1_group64_reader_f32(
    const unsigned short* __restrict__ x_bf16,
    const unsigned char* __restrict__ packed_lo,
    const unsigned short* __restrict__ qscale,
    const unsigned short* __restrict__ qzero,
    const unsigned short* __restrict__ row_scale,
    const unsigned short* __restrict__ col_scale,
    const unsigned short* __restrict__ bias_bf16,
    void* __restrict__ y_out,
    unsigned long long* __restrict__ group_reads_out,
    int M, int N, int K,
    int P_max, int P_min,
    float base_tol,
    int out_is_bf16,
    int shared_scale,
    const int* __restrict__ frontier_in,
    const int* __restrict__ frontier_count,
    float* __restrict__ line_spine,
    float* __restrict__ group_obs,
    float* __restrict__ group_prev_obs,
    int* __restrict__ plane_cursor,
    const float* __restrict__ input_fwht,
    const float* __restrict__ input_group_sum,
    int chunk_planes
) {
    const float UNI_PHI   = 0.22313016f; /* exp(-1.5)  */
    const float UNI_SIGMA = 1.77245385f; /* sqrt(pi)   */

    if (chunk_planes < 0) {
        micro_coupled_warp_row(x_bf16,packed_lo,qscale,qzero,row_scale,col_scale,bias_bf16,
            y_out,group_reads_out,plane_cursor,N,K,P_max,P_min,base_tol,out_is_bf16,shared_scale);
        return;
    }

    // Coupled mode: consume the substrate's live worklist. A group owns its
    // cursor and accumulated contribution; revisiting it never adds old bits.
    if (frontier_in != nullptr) {
        int job = blockIdx.x * blockDim.x + threadIdx.x;
        if (job >= frontier_count[0]) return;
        int ng = K >> 6;
        int agent = frontier_in[job];
        int r = agent / ng;
        int g = agent % ng;
        int p = plane_cursor[agent];
        float total = line_spine[agent];
        float obs = group_obs[agent];
        float prev = group_prev_obs[agent];
        if (p == 0) total = bf16_to_f32(qzero[agent]) * input_group_sum[g];
        int end = p + chunk_planes;
        if (end > P_max) end = P_max;
        for (; p < end; ++p) {
            long long si = shared_scale ? agent : (long long)p * N * ng + agent;
            float scale = bf16_to_f32(qscale[si]);
            if (shared_scale) scale = ldexpf(scale, -p);
            const unsigned int* bits = reinterpret_cast<const unsigned int*>(
                packed_lo + (long long)p * N * (K >> 3) + (long long)r * (K >> 3));
            float delta = eval_group64_bits(input_fwht + g * 64,
                input_group_sum[g], bits[g * 2], bits[g * 2 + 1], scale);
            prev = obs;
            obs = delta;
            total += delta;
            if (group_reads_out != nullptr) atomicAdd(group_reads_out, 1ull);
        }
        plane_cursor[agent] = p;
        line_spine[agent] = total;
        group_obs[agent] = obs;
        group_prev_obs[agent] = prev;
        return;
    }

    extern __shared__ char smem_raw[];
    int ng_full = K >> 6; /* K / 64, supports K <= 4096 (ng_full <= 64 -> max 2 groups per lane!) */
    float* s_x = reinterpret_cast<float*>(smem_raw);
    float* s_xsum = s_x + K;

    int tid = threadIdx.x;
    int m = blockIdx.y;
    if (m >= M) return;

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp_id;
    bool valid_row = (row < N);

    const unsigned short* x_row = x_bf16 + (long long)m * K;
    for (int c = tid; c < K; c += blockDim.x) {
        s_x[c] = bf16_to_f32(x_row[c]) * bf16_to_f32(col_scale[c]);
    }
    __syncthreads();
    micro_fwht64_and_sum(s_x, s_xsum, ng_full, warp_id, lane, warps_per_block);
    __syncthreads();

    if (!valid_row) return;

    int k8 = K >> 3;
    long long plane_byte_stride = (long long)N * k8;
    long long plane_grp_stride = (long long)N * ng_full;
    const unsigned short* z_row = qzero + (long long)row * ng_full;

    /* Each lane owns up to 2 elementary group-64 micro-agents (supports up to K=4096 with 32 lanes) */
    int g0 = lane;
    int g1 = lane + 32;
    bool has_g0 = (g0 < ng_full);
    bool has_g1 = (g1 < ng_full);

    float out_g0 = has_g0 ? (bf16_to_f32(z_row[g0]) * s_xsum[g0]) : 0.0f;
    float out_g1 = has_g1 ? (bf16_to_f32(z_row[g1]) * s_xsum[g1]) : 0.0f;
    float obs_g0 = 0.0f;
    float obs_g1 = 0.0f;
    bool alive_g0 = has_g0;
    bool alive_g1 = has_g1;
    unsigned int my_reads = 0u;

    for (int p = 0; p < P_max; ++p) {
        /* 1-cycle warp-ring check: if ALL 64-weight micro-agents in this row converged, exit! */
        unsigned int any_alive = __ballot_sync(0xffffffffu, alive_g0 || alive_g1);
        if (any_alive == 0u) break;

        float strat_g0 = out_g0 + obs_g0;
        float strat_g1 = out_g1 + obs_g1;

        const unsigned short* s_row = shared_scale
            ? (qscale + (long long)row * ng_full)
            : (qscale + (long long)p * plane_grp_stride + (long long)row * ng_full);
        float p_mult = shared_scale ? ldexpf(1.0f, -p) : 1.0f;
        const unsigned int* b_u32 = reinterpret_cast<const unsigned int*>(
            packed_lo + (long long)p * plane_byte_stride + (long long)row * k8
        );

        float next_obs0 = 0.0f;
        float next_obs1 = 0.0f;

        if (alive_g0) {
            float sv0 = bf16_to_f32(s_row[g0]) * p_mult;
            next_obs0 = eval_group64_bits(s_x + (g0 << 6), s_xsum[g0], b_u32[g0 * 2], b_u32[g0 * 2 + 1], sv0);
            my_reads += 1u;
        }
        if (alive_g1) {
            float sv1 = bf16_to_f32(s_row[g1]) * p_mult;
            next_obs1 = eval_group64_bits(s_x + (g1 << 6), s_xsum[g1], b_u32[g1 * 2], b_u32[g1 * 2 + 1], sv1);
            my_reads += 1u;
        }

        float new_s0 = strat_g0 + next_obs0;
        float new_s1 = strat_g1 + next_obs1;
        /* Two-turn smooth mirror envelope prevents single-plane scalar zero-crossing false stops */
        float env0 = fmaxf(fabsf(next_obs0), 0.5f * fabsf(obs_g0));
        float env1 = fmaxf(fabsf(next_obs1), 0.5f * fabsf(obs_g1));

        /* Advance per-agent Mirror Equality state */
        out_g0 = strat_g0; obs_g0 = next_obs0;
        out_g1 = strat_g1; obs_g1 = next_obs1;

        /* Warp micro-ring exchange: share total row energy across all 32 lanes in 5 register hops */
        float lane_e = (has_g0 ? new_s0 * new_s0 : 0.0f) + (has_g1 ? new_s1 * new_s1 : 0.0f);
        float row_e = lane_e;
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            row_e += __shfl_xor_sync(0xffffffffu, row_e, d);
        }
        float row_rms = sqrtf(row_e / (float)ng_full) + 1e-8f;

        /* Each 64-weight micro-agent autonomously decides whether to converge or keep refining */
        if ((p + 1) >= P_min && base_tol > 0.0f) {
            if (alive_g0) {
                float share0 = (new_s0 * new_s0) / (row_e + 1e-12f);
                /* High-energy micro-agent gets tighter UNI protection; quiet micro-agent exits early */
                float factor0 = (1.0f - UNI_PHI) * (1.0f - fminf(1.0f, share0 * 4.0f) / UNI_SIGMA);
                if (factor0 < 0.25f) factor0 = 0.25f;
                float defect0 = (1.15470054f * env0) / (fabsf(new_s0) + row_rms * 0.5f);
                if (defect0 < base_tol * factor0) {
                    alive_g0 = false;
                }
            }
            if (alive_g1) {
                float share1 = (new_s1 * new_s1) / (row_e + 1e-12f);
                float factor1 = (1.0f - UNI_PHI) * (1.0f - fminf(1.0f, share1 * 4.0f) / UNI_SIGMA);
                if (factor1 < 0.25f) factor1 = 0.25f;
                float defect1 = (1.15470054f * env1) / (fabsf(new_s1) + row_rms * 0.5f);
                if (defect1 < base_tol * factor1) {
                    alive_g1 = false;
                }
            }
        }
    }

    /* Assemble final row from all micro-agents on the shared line */
    float acc = (out_g0 + obs_g0) + (out_g1 + obs_g1);
    unsigned int reads_sum = my_reads;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
        reads_sum += __shfl_down_sync(0xffffffffu, reads_sum, off);
    }

    if (lane == 0) {
        float y_val = acc * bf16_to_f32(row_scale[row]);
        if (bias_bf16 != nullptr) {
            y_val += bf16_to_f32(bias_bf16[row]);
        }
        long long idx = (long long)m * N + row;
        if (out_is_bf16) {
            reinterpret_cast<unsigned short*>(y_out)[idx] = f32_to_bf16_rne(y_val);
        } else {
            reinterpret_cast<float*>(y_out)[idx] = y_val;
        }
        if (group_reads_out != nullptr) {
            atomicAdd(group_reads_out, (unsigned long long)reads_sum);
        }
    }
}

/* Kernel B: Autonomous 1,000,000+ Elementary Micro-Agent Substrate Step
 *
 * 1 CUDA thread = 1 active elementary micro-agent `u` from `frontier_in[]`.
 * Every agent:
 *   1. Reads its current state on the shared assembly line (`line_spine[u]`, `obs_in[u]`).
 *   2. Turns along its `DEG` arbitrary spatial neighbors `nbr_idx[u * DEG + d]` with local
 *      couplings `nbr_w[u * DEG + d]`, computing Mirror x UNI propagation `delta`.
 *   3. Intersects the shared line at each neighbor (`atomicAdd(&line_spine[v], delta)`) so
 *      start and next step ALWAYS meet on one assembly line.
 *   4. Evaluates Mirror defect at `v` and uses warp-aggregated `__ballot_sync` + `atomicCAS`
 *      to self-spawn `v` into `frontier_out[]` without CPU intervention.
 *
 * Grid:  ((num_active + 255) / 256, 1, 1)
 * Block: (256, 1, 1)
 */
extern "C" __global__ void micro_agent_million_substrate_step_f32(
    const int* __restrict__ frontier_in,
    int num_active,
    const int* __restrict__ nbr_idx,
    const float* __restrict__ nbr_w,
    float* __restrict__ line_spine,
    float* __restrict__ obs_curr,
    float* __restrict__ obs_next,
    int* __restrict__ visited_gen,
    int* __restrict__ frontier_out,
    int* __restrict__ next_count_out,
    int N_agents,
    int DEG,
    int gen_id,
    float spawn_tol,
    float damping,
    int coupling_mode,
    const int* __restrict__ plane_cursor,
    const float* __restrict__ previous_obs,
    const float* __restrict__ row_energy,
    int groups_per_row,
    int max_planes,
    int min_planes
) {
    const float UNI_PHI   = 0.22313016f; /* exp(-1.5)  */
    const float UNI_SIGMA = 1.77245385f; /* sqrt(pi)   */

    int tid_g = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid_g >= num_active) return;

    // Trained-weight mode transports requests, never artificial activations.
    // All groups sharing a real output row see its updated state. Previously
    // quiet groups can request more bits when that shared context changes.
    if (coupling_mode) {
        int p = plane_cursor[tid_g];
        if (micro_substrate_wants_more(line_spine[tid_g],obs_curr[tid_g],previous_obs[tid_g],
                row_energy[tid_g / groups_per_row],groups_per_row,p,max_planes,min_planes,spawn_tol)) {
            int pos = atomicAdd(next_count_out, 1);
            frontier_out[pos] = tid_g;
        }
        return;
    }

    int u = frontier_in[tid_g];
    if (u < 0 || u >= N_agents) return;

    /* Left Wing at micro-agent u: Strategy_u = Outcome_(t-1) + ObservedBehavior_(t-1) */
    float obs_u = obs_curr[u];
    float line_u = line_spine[u];
    float mag_u = fabsf(obs_u);
    if (mag_u < 1e-9f) return;

    /* Propagate freely to DEG arbitrary neighbors in the micro-agent fabric */
    long long base_e = (long long)u * DEG;
    for (int d = 0; d < DEG; ++d) {
        int v = nbr_idx[base_e + d];
        if (v < 0 || v >= N_agents) continue;
        float w_uv = nbr_w[base_e + d];

        /* Mirror x UNI scaled step: Phi dampens recursive hop, Sigma normalizes Gaussian spread */
        float delta = obs_u * w_uv * damping * (1.0f - UNI_PHI / UNI_SIGMA);

        /* Start point (u) and next step (v) intersect directly on the shared assembly line */
        float prev_v = atomicAdd(&line_spine[v], delta);
        atomicAdd(&obs_next[v], delta);

        float next_v = prev_v + delta;
        float defect_v = (1.15470054f * fabsf(delta)) / (fabsf(next_v) + 1.0f);

        /* Autonomous self-spawning: if this point on the micro-ring still has open mirror energy,
         * claim neighbor v once for the next generation via atomicCAS + append to frontier_out */
        if (defect_v >= spawn_tol && fabsf(delta) >= spawn_tol * 0.25f) {
            if (atomicExch(&visited_gen[v], gen_id) != gen_id) {
                int pos = atomicAdd(next_count_out, 1);
                if (pos < N_agents) {
                    frontier_out[pos] = v;
                }
            }
        }
    }
}

extern "C" __global__ void micro_agent_prepare_input_f32(
    const unsigned short* x, const unsigned short* col_scale,
    float* transformed, float* group_sum, int K
) {
    int g = blockIdx.x;
    int lane = threadIdx.x;
    if (g * 64 >= K) return;
    int a = g * 64 + lane;
    int b = a + 32;
    float v0 = bf16_to_f32(x[a]) * bf16_to_f32(col_scale[a]);
    float v1 = bf16_to_f32(x[b]) * bf16_to_f32(col_scale[b]);
    for (int d = 1; d < 32; d <<= 1) {
        float u0 = __shfl_xor_sync(0xffffffffu, v0, d);
        float u1 = __shfl_xor_sync(0xffffffffu, v1, d);
        v0 = (lane & d) ? (u0 - v0) : (v0 + u0);
        v1 = (lane & d) ? (u1 - v1) : (v1 + u1);
    }
    float y0 = (v0 + v1) * 0.125f;
    float y1 = (v0 - v1) * 0.125f;
    transformed[a] = y0;
    transformed[b] = y1;
    float sum = y0 + y1;
    for (int d = 16; d > 0; d >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, d);
    if (lane == 0) group_sum[g] = sum;
}

extern "C" __global__ void micro_agent_assemble_rows_f32(
    const float* line_spine, float* row_energy, const unsigned short* row_scale,
    const unsigned short* bias, void* output, int N, int groups_per_row, int bf16
) {
    int row = blockIdx.x;
    int lane = threadIdx.x;
    if (row >= N) return;
    float sum = 0.0f;
    float energy = 0.0f;
    for (int g = lane; g < groups_per_row; g += 32) {
        float s = line_spine[(long long)row * groups_per_row + g];
        sum += s;
        energy += s * s;
    }
    for (int d = 16; d > 0; d >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, d);
        energy += __shfl_down_sync(0xffffffffu, energy, d);
    }
    if (lane == 0) {
        row_energy[row] = energy;
        float value = sum * bf16_to_f32(row_scale[row]);
        if (bias != nullptr) value += bf16_to_f32(bias[row]);
        if (bf16) reinterpret_cast<unsigned short*>(output)[row] = f32_to_bf16_rne(value);
        else reinterpret_cast<float*>(output)[row] = value;
    }
}
"""

_MICRO_KERNEL_NAMES = (
    b"micro_agent_q1_group64_reader_f32",
    b"micro_agent_million_substrate_step_f32",
    b"micro_agent_prepare_input_f32",
    b"micro_agent_assemble_rows_f32",
)

_LOCK = threading.Lock()
_DEVICE_MODULES: Dict[int, Dict[bytes, ctypes.c_void_p]] = {}


def _compile_or_load_micro_ptx(major: int, minor: int, nvrtc) -> bytes:
    src_bytes = _MICRO_CUDA_SOURCE.encode("utf-8")
    digest = hashlib.sha256(src_bytes).hexdigest()[:16]
    PTX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = PTX_CACHE_DIR / f"mirror_micro_agents_sm{major}{minor}_{digest}.ptx"
    if cache_file.is_file():
        data = cache_file.read_bytes()
        return data if data.endswith(b"\x00") else data + b"\x00"

    prog = ctypes.c_void_p()
    rc = nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src_bytes, b"mirror_micro_agents.cu", 0, None, None)
    if rc != 0:
        raise RuntimeError(f"nvrtcCreateProgram(mirror_micro_agents) failed with error {rc}")
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
                f"NVRTC mirror_micro_agents compile failed ({rc}):\n{log_buf.value.decode('utf-8', errors='replace')}"
            )
        ptx_sz = ctypes.c_size_t()
        nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_sz))
        ptx_buf = ctypes.create_string_buffer(ptx_sz.value)
        rc = nvrtc.nvrtcGetPTX(prog, ptx_buf)
        if rc != 0:
            raise RuntimeError(f"nvrtcGetPTX(mirror_micro_agents) failed with error {rc}")
        raw_ptx = ptx_buf.raw
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    tmp = cache_file.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(raw_ptx)
    os.replace(tmp, cache_file)
    return raw_ptx


def get_micro_agent_kernels(device: torch.device):
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
            ptx = _compile_or_load_micro_ptx(major, minor, nvrtc)
            ptx_buf = ctypes.create_string_buffer(ptx)
            mod = ctypes.c_void_p()
            rc = nvcuda.cuModuleLoadData(ctypes.byref(mod), ctypes.cast(ptx_buf, ctypes.c_void_p))
            if rc != 0:
                raise RuntimeError(f"cuModuleLoadData(mirror_micro_agents) failed with CUresult={rc}")
            funcs: Dict[bytes, ctypes.c_void_p] = {"_module": mod}
            for name in _MICRO_KERNEL_NAMES:
                fn = ctypes.c_void_p()
                rc = nvcuda.cuModuleGetFunction(ctypes.byref(fn), mod, name)
                if rc != 0:
                    raise RuntimeError(f"cuModuleGetFunction({name!r}) failed with CUresult={rc}")
                funcs[name] = fn
            _DEVICE_MODULES[dev_idx] = funcs
            return nvcuda, funcs


def launch_micro_agent_q1_group64_reader(
    x_bf16: torch.Tensor,
    packed_lo: torch.Tensor,
    qscale: torch.Tensor,
    qzero: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor,
    y_out: torch.Tensor,
    *,
    max_planes: int | None = None,
    min_planes: int = 3,
    base_tol: float = 0.14,
    bias: Optional[torch.Tensor] = None,
    return_stats: bool = False,
) -> Optional[Dict[str, float]]:
    """Read-only per-group-64 micro-agent Q1 reader (`K <= 4096`, `K % 64 == 0`).

    Returns telemetry dict (`total_group_agents`, `group_plane_reads`, `avg_planes_per_group`,
    `read_fraction_vs_max`) when `return_stats=True`.
    """
    dev = x_bf16.device
    if dev.type != "cuda":
        raise ValueError("launch_micro_agent_q1_group64_reader requires CUDA tensors")
    x_c = x_bf16.contiguous()
    M, K = int(x_c.shape[0]), int(x_c.shape[1])
    N = int(y_out.shape[1])
    if K % 64 != 0 or K > 4096:
        raise ValueError(f"micro_agent_q1_group64_reader requires K % 64 == 0 and K <= 4096 (got K={K})")
    if M == 0 or N == 0:
        return {} if return_stats else None

    nvcuda, fns = get_micro_agent_kernels(dev)
    p_avail = int(packed_lo.shape[0])
    p_max = p_avail if max_planes is None else max(1, min(p_avail, int(max_planes)))
    p_min = max(1, min(p_max, int(min_planes)))
    shared_sc = 1 if (qscale.ndim == 2 or (qscale.ndim == 3 and int(qscale.shape[0]) == 1 and p_max > 1)) else 0
    out_is_bf16 = 1 if y_out.dtype == torch.bfloat16 else 0

    reads_counter = torch.zeros((1,), dtype=torch.int64, device=dev) if return_stats else None

    p_x = ctypes.c_void_p(x_c.data_ptr())
    p_lo = ctypes.c_void_p(packed_lo.data_ptr())
    p_s = ctypes.c_void_p(qscale.data_ptr())
    p_z = ctypes.c_void_p(qzero.data_ptr())
    p_rs = ctypes.c_void_p(row_scale.data_ptr())
    p_cs = ctypes.c_void_p(col_scale.data_ptr())
    p_b = ctypes.c_void_p(0 if bias is None else bias.data_ptr())
    p_y = ctypes.c_void_p(y_out.data_ptr())
    p_rc = ctypes.c_void_p(0 if reads_counter is None else reads_counter.data_ptr())

    c_M, c_N, c_K = ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)
    c_pmax, c_pmin = ctypes.c_int(p_max), ctypes.c_int(p_min)
    c_tol = ctypes.c_float(float(base_tol))
    c_ob, c_ss = ctypes.c_int(out_is_bf16), ctypes.c_int(shared_sc)

    null_args = [ctypes.c_void_p() for _ in range(8)]
    c_chunk = ctypes.c_int(0)
    args = (ctypes.c_void_p * 26)(
        ctypes.addressof(p_x), ctypes.addressof(p_lo),
        ctypes.addressof(p_s), ctypes.addressof(p_z),
        ctypes.addressof(p_rs), ctypes.addressof(p_cs),
        ctypes.addressof(p_b), ctypes.addressof(p_y), ctypes.addressof(p_rc),
        ctypes.addressof(c_M), ctypes.addressof(c_N), ctypes.addressof(c_K),
        ctypes.addressof(c_pmax), ctypes.addressof(c_pmin),
        ctypes.addressof(c_tol), ctypes.addressof(c_ob), ctypes.addressof(c_ss),
        *[ctypes.addressof(p) for p in null_args], ctypes.addressof(c_chunk),
    )

    smem = (K + K // 64) * 4
    grid_x = (N + 7) // 8
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    rc = nvcuda.cuLaunchKernel(
        fns[b"micro_agent_q1_group64_reader_f32"],
        grid_x, M, 1, 256, 1, 1, smem, stream, args, None,
    )
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel(micro_agent_q1_group64_reader_f32) failed with CUresult={rc}")

    if not return_stats:
        return None
    total_agents = M * N * (K // 64)
    reads = int(reads_counter.item())
    return {
        "total_group_agents": total_agents,
        "group_plane_reads": reads,
        "avg_planes_per_group": round(reads / max(1, total_agents), 3),
        "read_fraction_vs_max_pct": round(100.0 * reads / max(1, total_agents * p_max), 2),
    }


def launch_coupled_micro_agent_q1_reader(
    x_bf16: torch.Tensor,
    packed_lo: torch.Tensor,
    qscale: torch.Tensor,
    qzero: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor,
    y_out: torch.Tensor,
    *,
    max_planes: int | None = None,
    min_planes: int = 3,
    base_tol: float = 0.14,
    bias: Optional[torch.Tensor] = None,
    return_stats: bool = False,
    backend: str = "auto",
) -> Optional[Dict[str, object]]:
    """Couple the actual reader and substrate kernels over trained weight groups.

    Each node is one real (output row, input group of 64) contribution. The
    substrate queues necessary deeper reads using updated shared row energy.
    Cursors prevent duplicate accumulation and permit quiet nodes to reactivate.
    No random graph or damping is applied to the learned operator. This first
    implementation supports decode M=1, including K > 4096. For K <= 4096 the
    reader and substrate decision are fused in a warp with no CPU wave waits.
    Wider projections use a global worklist with GPU work-count waits. The stopping rule is a local approximation,
    not a certified bound on final transformer logits. base_tol=0 reads all bits.
    """
    dev = x_bf16.device
    if dev.type != "cuda" or x_bf16.ndim != 2 or x_bf16.shape[0] != 1:
        raise ValueError("coupled reader requires a CUDA input with shape [1, K]")
    K = int(x_bf16.shape[1])
    if K <= 0 or K % 64 or packed_lo.ndim != 3:
        raise ValueError("coupled reader requires K % 64 == 0 and packed [P,N,K/8]")
    P, N, k8 = map(int, packed_lo.shape)
    ng = K // 64
    if P < 1 or N < 1 or k8 != K // 8 or tuple(y_out.shape) != (1, N):
        raise ValueError("coupled reader input, packed weights and output shapes disagree")
    if not math.isfinite(base_tol) or base_tol < 0:
        raise ValueError("base_tol must be finite and nonnegative")
    if backend not in ("auto", "fused", "worklist") or (backend == "fused" and K > 4096):
        raise ValueError("backend must be auto, worklist, or fused with K <= 4096")
    expected = [(x_bf16, torch.bfloat16), (packed_lo, torch.uint8),
                (qscale, torch.bfloat16), (qzero, torch.bfloat16),
                (row_scale, torch.bfloat16), (col_scale, torch.bfloat16)]
    if bias is not None:
        expected.append((bias, torch.bfloat16))
    for tensor, dtype in expected:
        if tensor.device != dev or tensor.dtype != dtype or not tensor.is_contiguous():
            raise ValueError("coupled reader tensors must be contiguous, on the input device and have the expected dtype")
    if y_out.device != dev or y_out.dtype not in (torch.float32, torch.bfloat16) or not y_out.is_contiguous():
        raise ValueError("output must be contiguous CUDA float32 or bfloat16")
    if tuple(qzero.shape) != (N, ng) or row_scale.numel() != N or col_scale.numel() != K:
        raise ValueError("coupled reader metadata shape mismatch")
    if bias is not None and bias.numel() != N:
        raise ValueError("coupled reader bias shape mismatch")
    if tuple(qscale.shape) not in ((N, ng), (1, N, ng), (P, N, ng)):
        raise ValueError("coupled reader scale shape mismatch")
    agents = N * ng
    if agents >= 2**31:
        raise ValueError("one coupled projection must fit int32 group indices")
    pmax = P if max_planes is None else max(1, min(P, int(max_planes)))
    pmin = max(1, min(pmax, int(min_planes)))
    shared_scale = int(qscale.numel() == agents)
    nvcuda, fns = get_micro_agent_kernels(dev)
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)

    def ptr(t):
        return ctypes.c_void_p(0 if t is None else t.data_ptr())

    def integer(v):
        return ctypes.c_int(int(v))

    def launch(name, grid, block, values, smem=0):
        args = (ctypes.c_void_p * len(values))(*[ctypes.addressof(v) for v in values])
        rc = nvcuda.cuLaunchKernel(fns[name], grid, 1, 1, block, 1, 1, smem, stream, args, None)
        if rc:
            raise RuntimeError(f"cuLaunchKernel({name!r}) failed with CUresult={rc}")

    if backend == "fused" or (backend == "auto" and K <= 4096):
        reads_counter = torch.zeros(1, dtype=torch.int64, device=dev) if return_stats else None
        rounds = torch.empty(N, dtype=torch.int32, device=dev) if return_stats else None
        launch(b"micro_agent_q1_group64_reader_f32", (N + 7) // 8, 256, [
            ptr(x_bf16), ptr(packed_lo), ptr(qscale), ptr(qzero), ptr(row_scale), ptr(col_scale),
            ptr(bias), ptr(y_out), ptr(reads_counter), integer(1), integer(N), integer(K), integer(pmax),
            integer(pmin), ctypes.c_float(base_tol), integer(y_out.dtype == torch.bfloat16), integer(shared_scale),
            ptr(None), ptr(None), ptr(None), ptr(None), ptr(None), ptr(rounds),
            ptr(None), ptr(None), integer(-1)], smem=(K + ng) * 4)
        if not return_stats:
            return None
        reads = int(reads_counter.item())
        return dict(backend="fused", total_group_agents=agents, group_plane_reads=reads,
                    avg_planes_per_group=reads / agents,
                    read_fraction_vs_max_pct=100.0 * reads / (agents * pmax),
                    waves_executed=int(rounds.max().item()), wave_sizes=None,
                    reader_launches=1, substrate_launches=0, fused_substrate=True,
                    remaining_agents=0, converged_naturally=True, exact_drain=False,
                    dependency_graph="groups sharing the same learned output row", max_planes=pmax)

    transformed = torch.empty(K, dtype=torch.float32, device=dev)
    group_sum = torch.empty(ng, dtype=torch.float32, device=dev)
    state = torch.zeros((3, agents), dtype=torch.float32, device=dev)
    line, obs, previous = state.unbind(0)
    cursor = torch.zeros(agents, dtype=torch.int32, device=dev)
    energy = torch.empty(N, dtype=torch.float32, device=dev)
    frontier = torch.arange(agents, dtype=torch.int32, device=dev)
    following = torch.empty_like(frontier)
    active_count = torch.full((1,), agents, dtype=torch.int32, device=dev)
    next_count = torch.zeros_like(active_count)
    launch(b"micro_agent_prepare_input_f32", ng, 32,
           [ptr(x_bf16), ptr(col_scale), ptr(transformed), ptr(group_sum), integer(K)])
    wave_sizes = [agents]
    activations = 0
    waves = 0
    # A bounded drain switches to all remaining bits rather than silently
    # accepting a nonempty queue. Usually the queue empties in <= P waves.
    drain = False
    while wave_sizes[-1]:
        activations += wave_sizes[-1]
        chunk = pmax if drain else (pmin if waves == 0 else 1)
        launch(b"micro_agent_q1_group64_reader_f32", (wave_sizes[-1] + 255) // 256, 256, [
            ptr(x_bf16), ptr(packed_lo), ptr(qscale), ptr(qzero), ptr(row_scale), ptr(col_scale),
            ptr(bias), ptr(y_out), ptr(None), integer(1), integer(N), integer(K), integer(pmax),
            integer(pmin), ctypes.c_float(base_tol), integer(y_out.dtype == torch.bfloat16), integer(shared_scale),
            ptr(frontier), ptr(active_count), ptr(line), ptr(obs), ptr(previous), ptr(cursor),
            ptr(transformed), ptr(group_sum), integer(chunk)])
        launch(b"micro_agent_assemble_rows_f32", N, 32, [ptr(line), ptr(energy), ptr(row_scale),
            ptr(bias), ptr(y_out), integer(N), integer(ng), integer(y_out.dtype == torch.bfloat16)])
        next_count.zero_()
        launch(b"micro_agent_million_substrate_step_f32", (agents + 255) // 256, 256, [
            ptr(frontier), integer(agents), ptr(None), ptr(None), ptr(line), ptr(obs), ptr(None),
            ptr(None), ptr(following), ptr(next_count), integer(agents), integer(0), integer(waves + 1),
            ctypes.c_float(base_tol), ctypes.c_float(1.0), integer(1), ptr(cursor), ptr(previous),
            ptr(energy), integer(ng), integer(pmax), integer(pmin)])
        waves += 1
        count = int(next_count.item())
        wave_sizes.append(count)
        frontier, following = following, frontier
        active_count, next_count = next_count, active_count
        if waves >= pmax * 2 and not drain and count:
            # Every node occurs once; finishing all nodes is the exact escape.
            frontier.copy_(torch.arange(agents, dtype=torch.int32, device=dev))
            active_count.fill_(agents)
            wave_sizes[-1] = agents
            drain = True
        elif drain and count:
            raise RuntimeError("coupled reader still has work after exact drain")
    if not return_stats:
        return None
    reads = int(cursor.sum().item())
    return dict(backend="worklist", total_group_agents=agents, group_plane_reads=reads,
                avg_planes_per_group=reads / agents,
                read_fraction_vs_max_pct=100.0 * reads / (agents * pmax),
                waves_executed=waves, wave_sizes=wave_sizes, total_agent_activations=activations,
                reader_launches=waves, substrate_launches=waves, converged_naturally=not drain,
                remaining_agents=0, fused_substrate=False,
                exact_drain=drain, dependency_graph="groups sharing the same learned output row",
                max_planes=pmax)


def run_million_micro_agent_substrate(
    seed_indices: torch.Tensor,
    seed_values: torch.Tensor,
    nbr_idx: torch.Tensor,
    nbr_w: torch.Tensor,
    *,
    spawn_tol: float = 0.02,
    damping: float = 0.62,
    max_waves: int = 32,
) -> Dict[str, object]:
    """Run an autonomous self-spawning micro-agent fabric directly on GPU (scales to 1M+ agents)."""
    dev = nbr_idx.device
    if dev.type != "cuda":
        raise ValueError("run_million_micro_agent_substrate requires CUDA tensors")
    nvcuda, fns = get_micro_agent_kernels(dev)

    N_agents, DEG = int(nbr_idx.shape[0]), int(nbr_idx.shape[1])
    nbr_i = nbr_idx.to(torch.int32).contiguous()
    nbr_f = nbr_w.to(torch.float32).contiguous()

    line_spine = torch.zeros((N_agents,), dtype=torch.float32, device=dev)
    obs_curr = torch.zeros((N_agents,), dtype=torch.float32, device=dev)
    obs_next = torch.zeros((N_agents,), dtype=torch.float32, device=dev)
    visited_gen = torch.zeros((N_agents,), dtype=torch.int32, device=dev)

    frontier_a = torch.empty((N_agents,), dtype=torch.int32, device=dev)
    frontier_b = torch.empty((N_agents,), dtype=torch.int32, device=dev)
    next_cnt = torch.zeros((1,), dtype=torch.int32, device=dev)

    s_idx = seed_indices.to(device=dev, dtype=torch.int32).contiguous()
    s_val = seed_values.to(device=dev, dtype=torch.float32).contiguous()
    num_active = int(s_idx.numel())
    frontier_a[:num_active].copy_(s_idx)
    line_spine.index_copy_(0, s_idx.long(), s_val)
    obs_curr.index_copy_(0, s_idx.long(), s_val)

    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    wave_sizes: List[int] = [num_active]
    total_activations = 0

    for wave in range(1, max_waves + 1):
        if num_active <= 0:
            break
        total_activations += num_active
        next_cnt.zero_()
        obs_next.zero_()

        p_fin = ctypes.c_void_p(frontier_a.data_ptr())
        p_ni = ctypes.c_void_p(nbr_i.data_ptr())
        p_nw = ctypes.c_void_p(nbr_f.data_ptr())
        p_ls = ctypes.c_void_p(line_spine.data_ptr())
        p_oc = ctypes.c_void_p(obs_curr.data_ptr())
        p_on = ctypes.c_void_p(obs_next.data_ptr())
        p_vg = ctypes.c_void_p(visited_gen.data_ptr())
        p_fout = ctypes.c_void_p(frontier_b.data_ptr())
        p_nc = ctypes.c_void_p(next_cnt.data_ptr())

        c_na = ctypes.c_int(num_active)
        c_N, c_D, c_g = ctypes.c_int(N_agents), ctypes.c_int(DEG), ctypes.c_int(wave)
        c_tol, c_damp = ctypes.c_float(float(spawn_tol)), ctypes.c_float(float(damping))

        c_mode = ctypes.c_int(0)
        extra_ptrs = [ctypes.c_void_p() for _ in range(3)]
        extra_ints = [ctypes.c_int(0) for _ in range(3)]
        args = (ctypes.c_void_p * 22)(
            ctypes.addressof(p_fin), ctypes.addressof(c_na),
            ctypes.addressof(p_ni), ctypes.addressof(p_nw),
            ctypes.addressof(p_ls), ctypes.addressof(p_oc), ctypes.addressof(p_on),
            ctypes.addressof(p_vg), ctypes.addressof(p_fout), ctypes.addressof(p_nc),
            ctypes.addressof(c_N), ctypes.addressof(c_D), ctypes.addressof(c_g),
            ctypes.addressof(c_tol), ctypes.addressof(c_damp),
            ctypes.addressof(c_mode),
            *[ctypes.addressof(p) for p in extra_ptrs],
            *[ctypes.addressof(v) for v in extra_ints],
        )
        grid_x = (num_active + 255) // 256
        rc = nvcuda.cuLaunchKernel(
            fns[b"micro_agent_million_substrate_step_f32"],
            grid_x, 1, 1, 256, 1, 1, 0, stream, args, None,
        )
        if rc != 0:
            raise RuntimeError(f"cuLaunchKernel(micro_agent_million_substrate_step_f32) failed with CUresult={rc}")

        num_active = min(N_agents, int(next_cnt.item()))
        wave_sizes.append(num_active)
        frontier_a, frontier_b = frontier_b, frontier_a
        obs_curr, obs_next = obs_next, obs_curr

    unique_touched = int((visited_gen > 0).sum().item()) + int(s_idx.numel())
    return {
        "line_spine": line_spine,
        "N_agents": N_agents,
        "waves_executed": len(wave_sizes) - 1,
        "wave_sizes": wave_sizes,
        "total_agent_activations": total_activations,
        "unique_agents_connected": min(N_agents, unique_touched),
        "converged_naturally": (wave_sizes[-1] == 0),
    }
