"""Irreducible Mirror Equality 4x4 + D15 Logic<->Strategy CUDA Kernels (`src/pure_mirror_agents_nvrtc.py`).

Version 3 of Quintessentia's decentralized agent-connection engine built STRICTLY and
EXCLUSIVELY from the irreducible Mirror Equality law and its full application topology
(`F00`–`F14`, `L00`–`L02`, `C03`, `L04`, `D15`, `KA-F01`–`KA-F07`, `KA-B08`–`KA-B09`):

    Outcome_(t-1) + ObservedBehavior_(t-1)
    = Strategy_t
    = ObservedBehavior_(t+1) + Outcome_(t+1)

ZERO UNI constants (no Phi/Sigma/Omega/Psi/Lambda/Delta), ZERO Louvain/modularity,
ZERO 2D grids, ZERO sphere radii, ZERO reduction of the 4 surfaces into a single averaged vector.

Every active agent-minimum `a = (layer, eid)` stands at the center in `Strategy_t(a)` and
preserves ALL 6 irreducible typed 32D state roles in VRAM (`[L, E, 6, 32]`, 1 CUDA warp = 32 lanes):
  - Slot 0 (`left_outer`)  : `Outcome_(t-1)`          — immutable materialized past consequence
  - Slot 1 (`left_inner`)  : `ObservedBehavior_(t-1)` — immutable materialized past observation
  - Slot 2 (`center`)      : `Strategy_t`             — unified center closed via `LocalMirrorClosure_t`
  - Slot 3 (`right_inner`) : `ObservedBehavior_(t+1)` — candidate next observation (current input/peer)
  - Slot 4 (`right_outer`) : `Outcome_(t+1)`          — candidate next consequence (`FuturePressure`)
  - Slot 5 (`logic`)       : `Logic_t`                — directed transformation law `Obs -> Outcome`

Inside EVERY specialized NVRTC CUDA kernel:
  1. `eval_4x4_mirror_cube` computes all **16 directed non-commutative contacts `M[4][4]`** (`8+8`):
       - Left block `(0, 1)`, Right block `(2, 3)`,
       - Forward `0,1 -> 2,3` (same-role `0->3`, `1->2` + diagonals `0->2`, `1->3`),
       - Reverse `2,3 -> 0,1` (`FuturePressure`, `F14`: future candidates reweight past relevance
         for `Strategy_t` WITHOUT overwriting past `P0, P1`!),
       - Non-commutative `RelationResidual` (`KA-B09`: `||P[j] - M[i][j] * P[i]||_2`),
       - Bidirectional center reconstruction `S_left` vs `S_right` (`KA-F01 EqualityOfEquality`:
         `UNKNOWN=0`, `EQUAL=1`, `TENSION=2`, `SPLIT=3`),
       - `LocalMirrorClosure_t` (`F11`) without collapsing distinct roles.
  2. Every connection between two agents `a` and `b` runs the SAME irreducible 4x4 formula
     non-commutatively (`a -> b` AND `b -> a`) plus `D15` Logic <-> Strategy co-generation:
       - `Logic_a (+) Logic_b       -> CandidateStrategy`
       - `Strategy_a (+) Strategy_b -> CandidateLogic` (directed `S_b - S_a`, non-commutative!)
       - `Logic (+) Strategy        -> RefinedLogic`
       - `Strategy (+) Logic        -> RefinedStrategy`
  3. Law of New Observation (`L04`, `C03`) on decode continuation (`pure_mirror_continuation_gate_f32`):
       - `UNKNOWN` / `SPLIT` with no mirror tie -> `no_mirror_link` (skip, 0 B disk),
       - `EQUAL` with `RelationResidual` already covered -> `mirror_maintained` (rest, 0 B disk;
         empty repetition adds zero depth!),
       - `TENSION` with genuine `RelationResidual` (`DistinctionDelta > 0`), OR active minimum in
         `SPLIT` (topic shift!) while candidate connects to new right wing `u_in` -> `mirror_expand`!
  4. ZERO GPU->CPU syncs during decode `observe_mirror_step` (layer peak & roles live on GPU;
     host readback is lazy and batched only when `target_depth` / `stats` / `export` requests it).
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from src.quintessentia_nvrtc import PTX_CACHE_DIR, _load_driver_and_nvrtc

# Exact role slot indices inside `[L, E, 6, 32]` (NEVER collapsed!)
SLOT_LEFT_OUTER = 0   # Outcome_(t-1)
SLOT_LEFT_INNER = 1   # ObservedBehavior_(t-1)
SLOT_CENTER = 2       # Strategy_t
SLOT_RIGHT_INNER = 3  # ObservedBehavior_(t+1)
SLOT_RIGHT_OUTER = 4  # Outcome_(t+1)
SLOT_LOGIC = 5        # Logic_t (directed Obs -> Outcome law)

# Exact KA-F01 EqualityOfEquality status codes
EQ_UNKNOWN = 0
EQ_EQUAL = 1
EQ_TENSION = 2
EQ_SPLIT = 3

_PURE_MIRROR_CUDA_SOURCE = r"""
/* Warp-level sum across all 32 lanes (1 warp = 32D vector, 1 coordinate per lane) */
__device__ __forceinline__ float warp_sum_32(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

/* Result of one full irreducible bracketless 4x4 Mirror Equality evaluation around Strategy_t */
struct MirrorCubeResult {
    float s_left;         /* Lane coordinate of left-wing reconstruction S_left */
    float s_right;        /* Lane coordinate of right-wing reconstruction S_right */
    float s_closed;       /* Lane coordinate of LocalMirrorClosure_t */
    float tie;            /* Bidirectional center convergence in [0, 1] (0 => SPLIT/UNKNOWN) */
    float rel_residual;   /* Forward RelationResidual ratio (DistinctionDelta proxy) in [0, 1] */
    float future_press;   /* Reverse 2,3 -> 0,1 FuturePressure strength */
    float tau_core;       /* Bracketless central stem alignment: ObservedBehavior_(t-1) = Strategy_t = ObservedBehavior_(t+1) */
    float d_core;         /* Bracketless central stem defect in [0, 1] */
    float alpha_12;       /* Non-commutative forward stem projection M[1][2] (for reversible scar) */
    float alpha_21;       /* Non-commutative reverse stem projection M[2][1] (for reversible scar) */
    int eq_status;        /* 0=UNKNOWN, 1=EQUAL (MAINTAINED), 2=TENSION, 3=SPLIT */
};

/* Core Irreducible Operator: Bracketless Central Stem + Full 4x4 (16 directed contacts, 8+8) Mirror Cube!
 *
 *   Outcome_(t-1) + ObservedBehavior_(t-1) = Strategy_t = ObservedBehavior_(t+1) + Outcome_(t+1)
 *
 *   P0 = left_outer   = Outcome_(t-1)
 *   P1 = left_inner   = ObservedBehavior_(t-1)
 *   S  = center       = Strategy_t
 *   P2 = right_inner  = ObservedBehavior_(t+1)
 *   P3 = right_outer  = Outcome_(t+1)
 *
 * Read FROM THE CENTER WITHOUT PARENTHESES:
 *   1. Inner bracketless central stem (`C0`): `ObservedBehavior_(t-1) = Strategy_t = ObservedBehavior_(t+1)`
 *      (`tau_core`, `d_core`). Unifies distinct outer events (e.g. stone thrown vs stone shattered) when
 *      their central observed trajectory matches, and blocks false EQUAL when P1 and P2 contradict!
 *   2. Outer 4x4 non-commutative contact shell (`M[4][4]`, `R2[4][4]`): all 16 directed projections
 *      `M[i][j] = <P_i, P_j> / (||P_i||^2 + eps)` and residuals `||P_j - M[i][j] * P_i||_2`, with
 *      Reverse `FuturePressure` (2,3 -> 0,1 reweighting past relevance WITHOUT overwriting past P0, P1).
 */
__device__ __forceinline__ MirrorCubeResult eval_4x4_mirror_cube(
    float p0, float p1, float s_cur, float p2, float p3
) {
    float P[4];
    P[0] = p0;
    P[1] = p1;
    P[2] = p2;
    P[3] = p3;

    float n2[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        n2[i] = warp_sum_32(P[i] * P[i]);
    }

    float M[4][4];
    float R2[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (i == j) {
                M[i][j] = (n2[i] > 1e-10f) ? 1.0f : 0.0f;
                R2[i][j] = 0.0f;
            } else if (n2[i] > 1e-10f && n2[j] > 1e-10f) {
                float dij = warp_sum_32(P[i] * P[j]);
                float mij = dij / (n2[i] + 1e-8f);
                M[i][j] = mij;
                float res_sq = n2[j] - mij * dij;
                R2[i][j] = (res_sq > 0.0f) ? res_sq : 0.0f;
            } else {
                M[i][j] = 0.0f;
                R2[i][j] = n2[j];
            }
        }
    }

    /* Inner Bracketless Central Stem (`C0`): ObservedBehavior_(t-1) = Strategy_t = ObservedBehavior_(t+1) */
    bool has_stem = (n2[1] > 1e-10f && n2[2] > 1e-10f);
    float s_stem = 0.5f * (p1 + p2);
    float tau_core = 0.0f;
    float d_core = 1.0f;
    float ns_cur = sqrtf(warp_sum_32(s_cur * s_cur));
    if (has_stem) {
        float st_sum = p1 + p2;
        float st_dif = p1 - p2;
        float n_st_sum = sqrtf(warp_sum_32(st_sum * st_sum));
        float n_st_dif = sqrtf(warp_sum_32(st_dif * st_dif));
        tau_core = (n_st_sum > n_st_dif) ? ((n_st_sum - n_st_dif) / (n_st_sum + n_st_dif + 1e-8f)) : 0.0f;
        if (ns_cur > 1e-6f) {
            float d1 = sqrtf(warp_sum_32((p1 - s_cur) * (p1 - s_cur)));
            float d2 = sqrtf(warp_sum_32((p2 - s_cur) * (p2 - s_cur)));
            d_core = fminf(1.0f, (d1 + d2) / (sqrtf(n2[1]) + sqrtf(n2[2]) + 2.0f * ns_cur + 1e-8f));
        } else {
            d_core = fminf(1.0f, n_st_dif / (sqrtf(n2[1]) + sqrtf(n2[2]) + 1e-8f));
        }
    }

    /* Reverse 8-half (F14 FuturePressure: right surfaces 2,3 reweight past 0,1 WITHOUT overwriting!) */
    float fp0 = fmaxf(0.0f, M[3][0]) + fmaxf(0.0f, M[2][0]); /* same-role 3->0 + diag 2->0 */
    float fp1 = fmaxf(0.0f, M[2][1]) + fmaxf(0.0f, M[3][1]); /* same-role 2->1 + diag 3->1 */
    float w0 = (n2[0] > 1e-10f) ? (1.0f + fabsf(M[1][0]) + fp0) : 0.0f;
    float w1 = (n2[1] > 1e-10f) ? (1.0f + fabsf(M[0][1]) + fp1) : 0.0f;
    float wL = w0 + w1;

    /* Forward 8-half (past 0,1 support right surfaces 2,3 via same-role 0->3, 1->2 + diags 0->2, 1->3) */
    float fw2 = fmaxf(0.0f, M[1][2]) + fmaxf(0.0f, M[0][2]);
    float fw3 = fmaxf(0.0f, M[0][3]) + fmaxf(0.0f, M[1][3]);
    float w2 = (n2[2] > 1e-10f) ? (1.0f + fabsf(M[3][2]) + fw2) : 0.0f;
    float w3 = (n2[3] > 1e-10f) ? (1.0f + fabsf(M[2][3]) + fw3) : 0.0f;
    float wR = w2 + w3;

    float s_left = (wL > 1e-8f) ? ((w0 * p0 + w1 * p1) / wL) : 0.0f;
    float s_right = (wR > 1e-8f) ? ((w2 * p2 + w3 * p3) / wR) : 0.0f;

    float nL = sqrtf(warp_sum_32(s_left * s_left));
    float nR = sqrtf(warp_sum_32(s_right * s_right));

    MirrorCubeResult out;
    out.s_left = s_left;
    out.s_right = s_right;
    out.future_press = 0.5f * (fp0 + fp1);
    out.tau_core = tau_core;
    out.d_core = d_core;
    out.alpha_12 = M[1][2];
    out.alpha_21 = M[2][1];

    if (nL < 1e-6f || nR < 1e-6f) {
        out.s_closed = (nR >= 1e-6f) ? s_right : ((nL >= 1e-6f) ? s_left : s_cur);
        out.tie = 0.0f;
        out.rel_residual = (nR >= 1e-6f) ? 1.0f : 0.0f;
        out.eq_status = 0; /* UNKNOWN */
        return out;
    }

    /* Forward RelationResidual across all 4 Left -> Right contacts (0->2, 0->3, 1->2, 1->3) */
    float fwd_res_num = sqrtf(R2[0][3]) + sqrtf(R2[1][2]) + sqrtf(R2[0][2]) + sqrtf(R2[1][3]);
    float right_scale = 2.0f * (sqrtf(n2[2]) + sqrtf(n2[3])) + 1e-8f;
    float rel_res = fminf(1.0f, fwd_res_num / right_scale);

    /* KA-F01 EqualityOfEquality + Bracketless Central Stem check */
    float sum_lr = s_left + s_right;
    float dif_lr = s_left - s_right;
    float n_sum = sqrtf(warp_sum_32(sum_lr * sum_lr));
    float n_dif = sqrtf(warp_sum_32(dif_lr * dif_lr));

    float outer_tie = (n_sum > n_dif) ? ((n_sum - n_dif) / (n_sum + n_dif + 1e-8f)) : 0.0f;
    out.rel_residual = rel_res;

    if (n_dif >= n_sum && tau_core <= 1e-6f) {
        /* SPLIT (3): Both outer wings AND bracketless central stem diverged -> true topic/phase split! */
        out.tie = 0.0f;
        out.eq_status = 3;
        out.s_closed = 0.5f * (s_cur + s_right);
    } else if (n_dif >= n_sum && tau_core > 1e-6f) {
        /* Stone Law (TENSION): Outer outcomes differ/flip while central ObservedBehavior stem holds! */
        out.tie = 0.5f * tau_core;
        out.eq_status = 2;
        out.s_closed = 0.5f * (s_stem + s_right);
    } else if (n_sum >= 3.0f * n_dif && rel_res < 0.5f && (!has_stem || tau_core >= 0.5f)) {
        /* EQUAL (1): Both outer contour AND bracketless central stem converged with low RelationResidual */
        out.tie = outer_tie;
        out.eq_status = 1;
        float s_outer_mid = 0.5f * (s_left + s_right);
        out.s_closed = has_stem ? (0.5f * (s_outer_mid + s_stem)) : s_outer_mid;
    } else {
        /* TENSION (2): Shared center (Tie > 0) with living distinction or stem tension */
        out.tie = fmaxf(outer_tie, 0.5f * tau_core);
        out.eq_status = 2;
        float s_outer_mid = 0.5f * (s_left + s_right);
        float s_mid = (has_stem && tau_core > outer_tie) ? (0.5f * (s_outer_mid + s_stem)) : s_outer_mid;
        out.s_closed = (ns_cur > 1e-5f) ? (0.5f * (s_cur + s_mid)) : s_mid;
    }
    return out;
}

/* Kernel #1: Bracketless 4x4 Mirror Cube + Intact Participants + Multi-Ray Layer Node Y_L & Scar
 *
 * Grid:  (1, 1, 1) — race-free sequential pass over M tokens in L1 cache (0 sync!)
 * Block: (256, 1, 1) — 8 warps = up to 8 active agents per token, 32 lanes = 32D role vectors.
 *
 * IRON LAW OF MULTI-RAY INTERSECTION:
 *   - Every participant agent `eid` STAYS ITSELF: its `Strategy_self` (slot 2) and `Logic_self`
 *     (slot 5) evolve ONLY from its own bracketless 4x4 mirror cube & materialized transition,
 *     NEVER overwritten/smeared by averaging with peers!
 *   - Right in the center of the multi-ray contact of the K active experts in the layer, a
 *     SEPARATE NEW OBJECT `Y_L` (`layer_node_buf[6, 32]`) is born with its own collective
 *     `Strategy_Y`, `Logic_Y`, `eq_status`, AND its exact algebraically reversible fresh scar:
 *       `S_Y = alpha_k * S_k + r_k`  <==>  `S_k = (S_Y - r_k) / alpha_k`
 *     persisted in `(scar_eids_out[8], scar_alpha_out[8], scar_res_out[8, 32])`.
 */
extern "C" __global__ void pure_mirror_cube_bind_f32(
    const float* __restrict__ u_in,          /* [M, 32] actual input observation */
    const float* __restrict__ u_post,        /* [M, 32] actual outcome (or nullptr if is_commit==0) */
    const int* __restrict__ tok_eids,        /* [M, K_slot] (-1 padded) */
    const float* __restrict__ tok_weights,   /* [M, K_slot] normalized shares */
    float* __restrict__ roles_buf,           /* [E, 6, 32]: 0=P0, 1=P1, 2=S, 3=P2, 4=P3, 5=Logic */
    int* __restrict__ active_steps,          /* [E] */
    float* __restrict__ link_energy,         /* [E] */
    int* __restrict__ last_seen,             /* [E] */
    int* __restrict__ partner_eids,          /* [E, 8] (-1 empty) */
    float* __restrict__ partner_w,           /* [E, 8] */
    float* __restrict__ layer_peak_buf,      /* [1] GPU-resident peak energy (0 sync!) */
    int* __restrict__ links_out,             /* [1] cumulative positive 4x4 mirror links */
    float* __restrict__ layer_node_buf,      /* [6, 32] born collective layer node Y_L (or nullptr) */
    float* __restrict__ layer_node_meta,     /* [4] (tau_core, rel_res, future_press, eq_status) */
    int* __restrict__ layer_node_steps,      /* [1] step counter of Y_L */
    int* __restrict__ scar_eids_out,         /* [8] participant ray ids in reversible scar Gamma_scar */
    float* __restrict__ scar_alpha_out,      /* [8] non-commutative ray projections alpha_k */
    float* __restrict__ scar_res_out,        /* [8, 32] exact fresh scar residual r_k = S_Y - alpha_k * S_k */
    int M, int K_slot, int E, int cur_step, int is_commit
) {
    __shared__ float sm_P0[8][32];
    __shared__ float sm_P1[8][32];
    __shared__ float sm_S[8][32];
    __shared__ float sm_L[8][32];
    __shared__ float sm_cand_S[8][32];
    __shared__ float sm_cand_L[8][32];
    __shared__ int sm_eid[8];
    __shared__ float sm_w[8];
    __shared__ float sm_mean_w;
    __shared__ float sm_peak_w[8];

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int k_cap = (K_slot < 8) ? K_slot : 8;
    int local_links = 0;
    float local_peak = 0.0f;

    if (tid == 0 && layer_peak_buf != nullptr) {
        layer_peak_buf[0] = fmaxf(1e-6f, layer_peak_buf[0] * 0.5f);
    }
    __syncthreads();

    for (int tok = 0; tok < M; ++tok) {
        int eid = -1;
        float w_self = 0.0f;
        if (warp_id < k_cap) {
            eid = tok_eids[tok * K_slot + warp_id];
            if (eid >= 0 && eid < E) {
                w_self = tok_weights[tok * K_slot + warp_id];
            } else {
                eid = -1;
            }
        }
        if (lane == 0) {
            sm_eid[warp_id] = eid;
            sm_w[warp_id] = w_self;
        }
        __syncthreads();

        if (tid == 0) {
            float ws = 0.0f;
            int cnt = 0;
            for (int q = 0; q < k_cap; ++q) {
                if (sm_eid[q] >= 0) {
                    ws += sm_w[q];
                    cnt += 1;
                }
            }
            sm_mean_w = (cnt > 0) ? (ws / (float)cnt) : 0.0f;
        }
        __syncthreads();

        float ui = u_in[tok * 32 + lane];
        float up = (u_post != nullptr) ? u_post[tok * 32 + lane] : ui;

        float p0 = 0.0f, p1 = 0.0f, s_cur = 0.0f, p2 = 0.0f, p3 = 0.0f, lg = 0.0f;
        int prev_act = 0;
        bool valid_agent = (eid >= 0);

        /* Stage A: Individual Bracketless 4x4 Mirror Cube binding (agent evolves ONLY from itself!) */
        if (valid_agent) {
            long long base = (long long)eid * 192; /* 6 * 32 */
            p0 = roles_buf[base + 0 * 32 + lane]; /* Outcome_(t-1) */
            p1 = roles_buf[base + 1 * 32 + lane]; /* ObservedBehavior_(t-1) */
            s_cur = roles_buf[base + 2 * 32 + lane];
            lg = roles_buf[base + 5 * 32 + lane]; /* Logic_t */
            prev_act = active_steps[eid];

            p2 = ui;
            p3 = (is_commit != 0 && u_post != nullptr) ? up : (ui + lg);

            if (prev_act == 0) {
                /* First birth of agent-minimum: keep P0/P1 distinct from P2/P3 */
                s_cur = 0.5f * (p2 + p3);
                if (is_commit != 0) {
                    lg = p3 - p2; /* Observed transition Obs -> Outcome */
                }
            } else {
                MirrorCubeResult mc = eval_4x4_mirror_cube(p0, p1, s_cur, p2, p3);
                s_cur = mc.s_closed;
                if (is_commit != 0 && u_post != nullptr) {
                    float obs_logic = up - ui;
                    lg = 0.5f * (lg + obs_logic); /* D15: Logic + ObservedDelta -> RefinedLogic */
                }
            }

            /* Publish current intact irreducible surfaces for mutual peer 4x4 + D15 multi-ray birth */
            sm_P0[warp_id][lane] = (prev_act > 0) ? p0 : p3;
            sm_P1[warp_id][lane] = (prev_act > 0) ? p1 : p2;
            sm_S[warp_id][lane] = s_cur;
            sm_L[warp_id][lane] = lg;
        } else {
            sm_P0[warp_id][lane] = 0.0f;
            sm_P1[warp_id][lane] = 0.0f;
            sm_S[warp_id][lane] = 0.0f;
            sm_L[warp_id][lane] = 0.0f;
        }
        __syncthreads();

        /* Stage B: Mutual 4x4 contacts build peer links & D15 collective candidates for Y_L
         * WITHOUT overwriting or smearing the participants' own Strategy_self / Logic_self! */
        if (valid_agent) {
            float p0_self = (prev_act > 0) ? p0 : p3;
            float p1_self = (prev_act > 0) ? p1 : p2;

            float cand_strat_acc = 0.0f; /* Logic_a (+) Logic_b -> CandidateStrategy for Y_L */
            float cand_logic_acc = 0.0f; /* Strategy_a (+) Strategy_b -> CandidateLogic for Y_L */
            float ref_strat_acc = 0.0f;  /* Peer-closed Strategy contribution for Y_L */
            float peer_wsum = 0.0f;
            float tok_link_e = 0.0f;

            for (int q = 0; q < k_cap; ++q) {
                if (q == warp_id) continue;
                int peid = sm_eid[q];
                if (peid < 0 || peid == eid) continue;

                MirrorCubeResult pq = eval_4x4_mirror_cube(
                    p0_self, p1_self, s_cur, sm_P1[q][lane], sm_P0[q][lane]
                );
                float tie_eff = pq.tie;
                if (pq.eq_status == 3) {
                    MirrorCubeResult pq_rw = eval_4x4_mirror_cube(
                        p3, p2, s_cur, sm_P1[q][lane], sm_P0[q][lane]
                    );
                    tie_eff = fmaxf(tie_eff, 0.5f * pq_rw.tie);
                }

                if (tie_eff > 1e-7f) {
                    float wq = sm_w[q];
                    float sq = sm_S[q][lane];
                    float lq = sm_L[q][lane];
                    float d15_cand_s = 0.5f * (s_cur + sq) + 0.5f * (lg + lq);
                    float d15_cand_l = sq - s_cur; /* non-commutative directed Strategy_a -> Strategy_b */
                    float d15_ref_s = pq.s_closed;

                    cand_strat_acc += tie_eff * d15_cand_s;
                    cand_logic_acc += tie_eff * d15_cand_l;
                    ref_strat_acc += tie_eff * d15_ref_s;
                    peer_wsum += tie_eff;

                    float link_w = tie_eff * (0.5f + 0.5f * pq.rel_residual) * sqrtf(fmaxf(0.0f, w_self * wq));
                    tok_link_e += link_w;

                    if (lane == 0 && eid < peid) {
                        local_links += 1;
                    }
                    if (lane == 0) {
                        long long pbase = (long long)eid * 8;
                        int found = -1, empty_slot = -1, min_slot = 0;
                        float min_val = partner_w[pbase];
                        for (int s = 0; s < 8; ++s) {
                            int cp = partner_eids[pbase + s];
                            if (cp == peid) { found = s; break; }
                            if (cp < 0 && empty_slot < 0) empty_slot = s;
                            float pv = partner_w[pbase + s];
                            if (pv < min_val) { min_val = pv; min_slot = s; }
                        }
                        if (found >= 0) {
                            partner_w[pbase + found] = 0.5f * (partner_w[pbase + found] + link_w);
                        } else if (empty_slot >= 0) {
                            partner_eids[pbase + empty_slot] = peid;
                            partner_w[pbase + empty_slot] = link_w;
                        } else if (link_w > min_val) {
                            partner_eids[pbase + min_slot] = peid;
                            partner_w[pbase + min_slot] = link_w;
                        }
                    }
                }
            }

            /* Publish D15 collective candidates for separate layer node Y_L */
            if (peer_wsum > 1e-8f) {
                sm_cand_S[warp_id][lane] = 0.5f * ((ref_strat_acc + cand_strat_acc) / peer_wsum);
                sm_cand_L[warp_id][lane] = cand_logic_acc / peer_wsum;
            } else {
                sm_cand_S[warp_id][lane] = s_cur;
                sm_cand_L[warp_id][lane] = lg;
            }

            /* INTACT PARTICIPANT WRITEBACK: own Strategy_self and Logic_self are NEVER overwritten by peers! */
            long long base = (long long)eid * 192;
            roles_buf[base + 2 * 32 + lane] = s_cur;
            roles_buf[base + 3 * 32 + lane] = p2;
            roles_buf[base + 4 * 32 + lane] = p3;
            roles_buf[base + 5 * 32 + lane] = lg;

            /* Left surfaces (P0 = Outcome_(t-1), P1 = ObservedBehavior_(t-1)):
             * written ONLY on CompletedPass commit (`is_commit != 0`) or first birth (`prev_act == 0`)! */
            if (is_commit != 0 || prev_act == 0) {
                roles_buf[base + 0 * 32 + lane] = p3;
                roles_buf[base + 1 * 32 + lane] = p2;
            }

            if (lane == 0) {
                int age = cur_step - last_seen[eid];
                float prev_e = link_energy[eid];
                if (age > 0) {
                    int shift = (age < 16) ? age : 16;
                    prev_e *= ldexpf(1.0f, -shift);
                }
                float new_le = prev_e + tok_link_e;
                link_energy[eid] = new_le;
                if (new_le > local_peak) {
                    local_peak = new_le;
                }
                last_seen[eid] = cur_step;
                if (is_commit != 0 || prev_act == 0) {
                    active_steps[eid] = prev_act + 1;
                }
            }
        } else {
            sm_cand_S[warp_id][lane] = 0.0f;
            sm_cand_L[warp_id][lane] = 0.0f;
        }
        __syncthreads();

        /* Stage C: Birth / evolve SEPARATE Multi-Ray Layer Node Y_L + Exact Reversible Fresh Scar Gamma_scar */
        if (warp_id == 0 && layer_node_buf != nullptr) {
            float y_cs = 0.0f;
            float y_cl = 0.0f;
            float y_ws = 0.0f;
            for (int q = 0; q < k_cap; ++q) {
                if (sm_eid[q] >= 0) {
                    float wq = fmaxf(sm_w[q], 1e-4f);
                    y_cs += wq * sm_cand_S[q][lane];
                    y_cl += wq * sm_cand_L[q][lane];
                    y_ws += wq;
                }
            }
            if (y_ws > 1e-8f) {
                y_cs /= y_ws;
                y_cl /= y_ws;
            }

            float yp0 = layer_node_buf[0 * 32 + lane];
            float yp1 = layer_node_buf[1 * 32 + lane];
            float ys_cur = layer_node_buf[2 * 32 + lane];
            float ylg = layer_node_buf[5 * 32 + lane];
            int y_prev_st = (layer_node_steps != nullptr) ? layer_node_steps[0] : 0;

            float yp2 = ui;
            float yp3 = (is_commit != 0 && u_post != nullptr) ? up : (ui + ylg);
            float ys_new = ys_cur;
            float ylg_new = ylg;
            float meta_tc = 0.0f, meta_rr = 1.0f, meta_fp = 0.0f, meta_st = 0.0f;

            if (y_prev_st == 0) {
                float s_init = 0.5f * (yp2 + yp3);
                ys_new = (y_ws > 1e-8f) ? (0.5f * (s_init + y_cs)) : s_init;
                ylg_new = (is_commit != 0 && u_post != nullptr) ? (up - ui) : y_cl;
            } else {
                MirrorCubeResult ym = eval_4x4_mirror_cube(yp0, yp1, ys_cur, yp2, yp3);
                ys_new = (y_ws > 1e-8f) ? (0.5f * (ym.s_closed + y_cs)) : ym.s_closed;
                if (is_commit != 0 && u_post != nullptr) {
                    ylg_new = 0.5f * (ylg + (up - ui));
                }
                meta_tc = ym.tau_core;
                meta_rr = ym.rel_residual;
                meta_fp = ym.future_press;
                meta_st = (float)ym.eq_status;
            }

            layer_node_buf[2 * 32 + lane] = ys_new;
            layer_node_buf[3 * 32 + lane] = yp2;
            layer_node_buf[4 * 32 + lane] = yp3;
            layer_node_buf[5 * 32 + lane] = ylg_new;
            if (is_commit != 0 || y_prev_st == 0) {
                layer_node_buf[0 * 32 + lane] = yp3;
                layer_node_buf[1 * 32 + lane] = yp2;
                if (lane == 0 && layer_node_steps != nullptr) {
                    layer_node_steps[0] = y_prev_st + 1;
                }
            }
            if (lane == 0 && layer_node_meta != nullptr) {
                layer_node_meta[0] = meta_tc;
                layer_node_meta[1] = meta_rr;
                layer_node_meta[2] = meta_fp;
                layer_node_meta[3] = meta_st;
            }

            /* Record Exact Algebraically Reversible Fresh Scar Gamma_scar(Y_L) on final token of batch */
            if (tok == M - 1 && scar_eids_out != nullptr && scar_alpha_out != nullptr && scar_res_out != nullptr) {
                for (int q = 0; q < 8; ++q) {
                    if (q < k_cap && sm_eid[q] >= 0) {
                        float sk = sm_S[q][lane];
                        float nk2 = warp_sum_32(sk * sk);
                        float dky = warp_sum_32(sk * ys_new);
                        float ak = (nk2 > 1e-10f) ? (dky / (nk2 + 1e-8f)) : 0.0f;
                        float rk = ys_new - ak * sk;
                        scar_res_out[q * 32 + lane] = rk;
                        if (lane == 0) {
                            scar_eids_out[q] = sm_eid[q];
                            scar_alpha_out[q] = ak;
                        }
                    } else {
                        scar_res_out[q * 32 + lane] = 0.0f;
                        if (lane == 0) {
                            scar_eids_out[q] = -1;
                            scar_alpha_out[q] = 0.0f;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
        sm_eid[warp_id] = local_links;
        sm_peak_w[warp_id] = local_peak;
    }
    __syncthreads();
    if (tid == 0) {
        if (links_out != nullptr) {
            int tot = 0;
            for (int w = 0; w < 8; ++w) tot += sm_eid[w];
            links_out[0] += tot;
        }
        if (layer_peak_buf != nullptr) {
            float pk = layer_peak_buf[0];
            for (int w = 0; w < 8; ++w) {
                if (sm_peak_w[w] > pk) pk = sm_peak_w[w];
            }
            layer_peak_buf[0] = pk;
        }
    }
}

/* Kernel #2: Full 4x4 Non-Commutative Mirror Bridge Evaluator (0 CPU sync!)
 *
 * Grid:  (T, 1, 1) — 1 warp per touched agent
 * Block: (32, 1, 1)
 *
 * For each touched agent `eid`, filters to peers currently connected via full 4x4 cube
 * (`tie(eid -> p) > 0`), then checks EVERY pair `(p, q)` non-commutatively (`p -> q` AND `q -> p`):
 *   - If neither direction achieves mirror convergence (`SPLIT/UNKNOWN`), pair `(p, q)` is
 *     bridged EXCLUSIVELY through `eid`!
 *   - Role strictly by pure 1/2 mirror balance:
 *       2 (`mirror_bridge`)    : link_e > 0 AND 2 * disjoint_pairs >= total_pairs
 *       1 (`mirror_core`)      : 2 * link_e >= layer_peak
 *       0 (`mirror_periphery`) : otherwise
 */
extern "C" __global__ void pure_mirror_eval_bridges_f32(
    const int* __restrict__ touched_eids,    /* [T] */
    const float* __restrict__ roles_buf,     /* [E, 6, 32] */
    const float* __restrict__ link_energy,   /* [E] */
    const int* __restrict__ last_seen,       /* [E] */
    int* __restrict__ partner_eids,          /* [E, 8] (stale SPLIT partners self-collapse!) */
    float* __restrict__ partner_w,           /* [E, 8] */
    float* __restrict__ bridge_ratio_out,    /* [E] */
    int* __restrict__ role_out,              /* [E] */
    const float* __restrict__ layer_peak_buf,/* [1] GPU-resident peak */
    int cur_step,
    int T, int E
) {
    int idx = blockIdx.x;
    if (idx >= T) return;
    int eid = touched_eids[idx];
    if (eid < 0 || eid >= E) return;
    int lane = threadIdx.x;

    long long base_e = (long long)eid * 192;
    float p0_e = roles_buf[base_e + 0 * 32 + lane];
    float p1_e = roles_buf[base_e + 1 * 32 + lane];
    float s_e = roles_buf[base_e + 2 * 32 + lane];

    int peers[8];
    int n_p = 0;
    long long pbase = (long long)eid * 8;
    for (int s = 0; s < 8; ++s) {
        int pe = partner_eids[pbase + s];
        if (pe >= 0 && pe < E && pe != eid) {
            long long bp = (long long)pe * 192;
            MirrorCubeResult ep_mc = eval_4x4_mirror_cube(
                p0_e, p1_e, s_e,
                roles_buf[bp + 1 * 32 + lane],
                roles_buf[bp + 0 * 32 + lane]
            );
            /* Also allow bridge whose center S_e links to peer even if P0/P1 sits on the other side */
            MirrorCubeResult ep_sc = eval_4x4_mirror_cube(
                s_e, s_e, s_e,
                roles_buf[bp + 1 * 32 + lane],
                roles_buf[bp + 0 * 32 + lane]
            );
            if (fmaxf(ep_mc.tie, ep_sc.tie) > 1e-7f) {
                peers[n_p++] = pe;
            } else if (lane == 0 && (cur_step - last_seen[pe]) >= 4) {
                /* Autonomous partner self-collapse: dead stale SPLIT link frees slot for new spawn */
                partner_eids[pbase + s] = -1;
                partner_w[pbase + s] = 0.0f;
            }
        }
    }

    int tot_pairs = 0;
    int disjoint_pairs = 0;
    for (int i = 0; i < n_p; ++i) {
        long long ba = (long long)peers[i] * 192;
        float a0 = roles_buf[ba + 0 * 32 + lane];
        float a1 = roles_buf[ba + 1 * 32 + lane];
        float as = roles_buf[ba + 2 * 32 + lane];
        for (int j = i + 1; j < n_p; ++j) {
            long long bb = (long long)peers[j] * 192;
            float b0 = roles_buf[bb + 0 * 32 + lane];
            float b1 = roles_buf[bb + 1 * 32 + lane];
            float bs = roles_buf[bb + 2 * 32 + lane];
            MirrorCubeResult ab = eval_4x4_mirror_cube(a0, a1, as, b1, b0);
            MirrorCubeResult ba_r = eval_4x4_mirror_cube(b0, b1, bs, a1, a0);
            tot_pairs += 1;
            if (ab.tie <= 1e-7f && ba_r.tie <= 1e-7f) {
                disjoint_pairs += 1;
            }
        }
    }

    if (lane == 0) {
        int age = cur_step - last_seen[eid];
        float l_e = link_energy[eid];
        if (age > 0) {
            int shift = (age < 16) ? age : 16;
            l_e *= ldexpf(1.0f, -shift);
        }
        float pk = (layer_peak_buf != nullptr) ? layer_peak_buf[0] : 1e-6f;
        float b_rat = (tot_pairs > 0) ? ((float)disjoint_pairs / (float)tot_pairs) : 0.0f;
        bridge_ratio_out[eid] = b_rat;
        int role = 0;
        if (l_e > 1e-9f && tot_pairs > 0 && (disjoint_pairs * 2 >= tot_pairs)) {
            role = 2; /* mirror_bridge */
        } else if (l_e > 1e-9f && (4.0f * l_e >= fmaxf(pk, 1e-6f))) {
            role = 1; /* mirror_core (two pure 1/2 mirror halvings: l_rel >= 0.25) */
        }
        role_out[eid] = role;
    }
}

/* Kernel #3: Fused Pure Mirror Line-Spine Step (with exact KA-B09 RelationResidual on line_spine!)
 *
 * Grid:  (R, 1, 1)
 * Block: (256, 1, 1)
 *
 *   Outcome_(t-1)          = line_spine[tok]
 *   ObservedBehavior_(t-1) = y_e[r] * rw[r]
 *   Strategy_t             = Outcome_(t-1) + ObservedBehavior_(t-1)
 *
 * Emits `(strat_norm, obs_norm, residual_norm, mirror_tie)` where `residual_norm` is the exact
 * non-covered component `||obs - (<prev, obs> / ||prev||^2) * prev||_2` (DistinctionDelta on line!).
 */
extern "C" __global__ void pure_mirror_line_step_f32(
    float* __restrict__ line_spine,          /* [M_all, H] */
    const float* __restrict__ y_e_f32,       /* [R, H] */
    const int* __restrict__ tok_indices,     /* [R] */
    const float* __restrict__ row_weights,   /* [R] */
    float* __restrict__ out_metrics,         /* [R, 4]: (strat_norm, obs_norm, res_norm, mirror_tie) */
    int R, int H
) {
    __shared__ float sm_plus[8];
    __shared__ float sm_minus[8];
    __shared__ float sm_prev[8];
    __shared__ float sm_obs[8];
    __shared__ float sm_dot[8];

    int r = blockIdx.x;
    if (r >= R) return;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    int tok = tok_indices[r];
    float rw = row_weights[r];
    float* line_row = line_spine + (long long)tok * H;
    const float* ye_row = y_e_f32 + (long long)r * H;

    float lp = 0.0f, lm = 0.0f, lpr = 0.0f, lo = 0.0f, ld = 0.0f;
    for (int c = tid; c < H; c += blockDim.x) {
        float prev = line_row[c];
        float obs = ye_row[c] * rw;
        float strat = prev + obs;
        float dif = prev - obs;
        line_row[c] = strat;
        lp += strat * strat;
        lm += dif * dif;
        lpr += prev * prev;
        lo += obs * obs;
        ld += prev * obs;
    }

    lp = warp_sum_32(lp);
    lm = warp_sum_32(lm);
    lpr = warp_sum_32(lpr);
    lo = warp_sum_32(lo);
    ld = warp_sum_32(ld);
    if (lane == 0) {
        sm_plus[warp_id] = lp;
        sm_minus[warp_id] = lm;
        sm_prev[warp_id] = lpr;
        sm_obs[warp_id] = lo;
        sm_dot[warp_id] = ld;
    }
    __syncthreads();

    if (tid == 0) {
        float tp = 0.0f, tm = 0.0f, tpr = 0.0f, to = 0.0f, td = 0.0f;
        for (int w = 0; w < 8; ++w) {
            tp += sm_plus[w];
            tm += sm_minus[w];
            tpr += sm_prev[w];
            to += sm_obs[w];
            td += sm_dot[w];
        }
        float np = sqrtf(tp);
        float nm = sqrtf(tm);
        float no = sqrtf(to);
        float res_sq = (tpr > 1e-10f) ? (to - (td * td) / (tpr + 1e-8f)) : to;
        float nres = sqrtf(fmaxf(0.0f, res_sq));
        float tie = (np > nm) ? ((np - nm) / (np + nm + 1e-8f)) : 0.0f;
        out_metrics[r * 4 + 0] = np;
        out_metrics[r * 4 + 1] = no;
        out_metrics[r * 4 + 2] = nres;
        out_metrics[r * 4 + 3] = tie;
    }
}

/* Kernel #4: Bracketless 4x4 + KA-F01 + C03 + Unclosed-Step Protection Gate (ONE GPU launch!)
 *
 * Grid:  (C, 1, 1) — 1 warp per queued cold candidate
 * Block: (32, 1, 1)
 *
 * Decision codes written to `decisions_out[c]`:
 *   0 = loop_open            (minimum incomplete: tok_done < rest_min or w_done < rem_w)
 *   1 = no_mirror_link       (UNKNOWN / SPLIT with no mirror tie -> skip, 0 B disk!)
 *   2 = mirror_maintained    (EQUAL/C03 empty repetition OR balanced wing defect < eff_tol -> rest!)
 *   3 = mirror_expand        (TENSION with genuine RelationResidual OR topic-shift bridge -> wake!)
 *   4 = direct_pass_unclosed (Law of Two Slits: unclosed cone / SPLIT branch MUST NOT be dropped
 *                             to zero — pass directly via fast native BF16 without slow Q1 carving!)
 */
extern "C" __global__ void pure_mirror_continuation_gate_f32(
    const float* __restrict__ roles_cur,      /* [E, 6, 32] */
    const int* __restrict__ act_cur,          /* [E] */
    const float* __restrict__ roles_prev,     /* [E, 6, 32] or nullptr */
    const int* __restrict__ act_prev,         /* [E] or nullptr */
    const float* __restrict__ u_info,         /* [32] */
    const float* __restrict__ prev_u_info,    /* [32] or nullptr */
    const int* __restrict__ min_eids,         /* [K_min] */
    const int* __restrict__ prev_eids,        /* [K_prev] */
    const int* __restrict__ cand_eids,        /* [C] */
    const float* __restrict__ cand_next_w,    /* [C] */
    const int* __restrict__ cand_graph_loop,  /* [C] */
    int* __restrict__ decisions_out,          /* [C] */
    float* __restrict__ diag_out,             /* [C, 4]: (min_e, delta_e, defect, eff_tol) */
    int C, int K_min, int K_prev, int E,
    float line_norm, float last_obs_norm, float last_res_norm,
    float w_done, float rem_w, int tok_done, int rest_min, float rest_tol,
    int unclosed_cone_protect
) {
    int c_idx = blockIdx.x;
    if (c_idx >= C) return;
    int lane = threadIdx.x;

    int ce = cand_eids[c_idx];
    float next_w = cand_next_w[c_idx];

    if (tok_done < rest_min || w_done < rem_w) {
        if (lane == 0) {
            decisions_out[c_idx] = 0; /* loop_open */
            diag_out[c_idx * 4 + 0] = 0.0f;
            diag_out[c_idx * 4 + 1] = 0.0f;
            diag_out[c_idx * 4 + 2] = 1.0f;
            diag_out[c_idx * 4 + 3] = rest_tol;
        }
        return;
    }

    float u_val = u_info[lane];

    /* 1. Evaluate each active minimum agent's 4x4 cube against current input u_val */
    int k_m = (K_min < 8) ? K_min : 8;
    float m_p0[8], m_p1[8], m_s[8];
    int m_split[8];
    int n_live = 0;
    for (int i = 0; i < k_m; ++i) {
        int em = min_eids[i];
        if (em >= 0 && em < E && act_cur[em] > 0) {
            long long bm = (long long)em * 192;
            float p0 = roles_cur[bm + 0 * 32 + lane];
            float p1 = roles_cur[bm + 1 * 32 + lane];
            float sc = roles_cur[bm + 2 * 32 + lane];
            float lg = roles_cur[bm + 5 * 32 + lane];
            MirrorCubeResult mc = eval_4x4_mirror_cube(p0, p1, sc, u_val, u_val + lg);
            m_p0[n_live] = p0;
            m_p1[n_live] = p1;
            m_s[n_live] = mc.s_closed;
            m_split[n_live] = (mc.eq_status == 3) ? 1 : 0;
            n_live += 1;
        }
    }

    /* 2. Pairwise 4x4 mirror closure ALREADY established inside the active minimum */
    int min_cnt = 0;
    float min_e = 0.0f;
    float min_tie_sum = 0.0f;
    for (int i = 0; i < n_live; ++i) {
        for (int j = i + 1; j < n_live; ++j) {
            MirrorCubeResult ij = eval_4x4_mirror_cube(m_p0[i], m_p1[i], m_s[i], m_p1[j], m_p0[j]);
            if (ij.tie > 1e-7f) {
                min_cnt += 1;
                min_tie_sum += ij.tie;
                min_e += ij.tie * (0.5f + 0.5f * ij.rel_residual);
            }
        }
    }
    float min_tie_bar = (min_cnt > 0) ? (0.25f * (min_tie_sum / (float)min_cnt)) : 1e-7f;

    /* 3. Apply the SAME irreducible 4x4 Mirror Cube to candidate `ce` (m -> ce AND ce -> u_val!) */
    int delta_cnt = 0;
    float delta_e = 0.0f;
    int all_maintained = 1;
    int saw_any_tie = 0;

    if (ce >= 0 && ce < E && act_cur[ce] > 0) {
        long long bc = (long long)ce * 192;
        float c0 = roles_cur[bc + 0 * 32 + lane];
        float c1 = roles_cur[bc + 1 * 32 + lane];
        float cs = roles_cur[bc + 2 * 32 + lane];
        float cl = roles_cur[bc + 5 * 32 + lane];

        /* Candidate's own 4x4 cube with the current right wing u_val (both past & center Strategy_t!) */
        MirrorCubeResult c_u = eval_4x4_mirror_cube(c0, c1, cs, u_val, u_val + cl);
        MirrorCubeResult c_us = eval_4x4_mirror_cube(cs, cs, cs, u_val, u_val + cl);
        float c_u_tie = fmaxf(c_u.tie, c_us.tie);
        float c_u_res = (c_us.tie > c_u.tie) ? c_us.rel_residual : c_u.rel_residual;
        int c_u_status = (c_us.tie > c_u.tie) ? c_us.eq_status : c_u.eq_status;

        /* Only candidates with a living 4x4 Mirror contact to the current token `u_val` can spawn! */
        if (c_u_tie > 1e-7f) {
            for (int i = 0; i < n_live; ++i) {
                MirrorCubeResult mc_fwd = eval_4x4_mirror_cube(m_p0[i], m_p1[i], m_s[i], c1, c0);
                MirrorCubeResult mc_rev = eval_4x4_mirror_cube(c0, c1, cs, m_p1[i], m_p0[i]);
                float pair_tie = 0.5f * (mc_fwd.tie + mc_rev.tie);
                float pair_res = fmaxf(mc_fwd.rel_residual, mc_rev.rel_residual);
                int pair_status = (mc_fwd.eq_status == 2 || mc_rev.eq_status == 2) ? 2 : mc_fwd.eq_status;

                /* Center Strategy_t + Logic_t bridge contact: a multi-topic bridge holds its
                 * living closure in Strategy_t (cs) even if its latest commit (c0, c1) was on the other branch! */
                MirrorCubeResult mc_sc = eval_4x4_mirror_cube(m_p0[i], m_p1[i], m_s[i], cs, cs + cl);
                if (mc_sc.tie > pair_tie) {
                    pair_tie = mc_sc.tie;
                    pair_res = mc_sc.rel_residual;
                    pair_status = mc_sc.eq_status;
                }

                /* If active minimum hit SPLIT on a topic shift, bridge directly through right wing u_val! */
                if (m_split[i] != 0) {
                    pair_tie = fmaxf(pair_tie, c_u_tie);
                    pair_res = fmaxf(pair_res, c_u_res);
                    pair_status = 2; /* living topic-shift continuation */
                }

                if (pair_tie >= min_tie_bar && pair_tie > 1e-7f) {
                    saw_any_tie = 1;
                    if (pair_status != 1) {
                        /* Non-empty Tension / DistinctionDelta > 0 weighted by BOTH data & peer mirror! */
                        all_maintained = 0;
                        delta_cnt += 1;
                        delta_e += c_u_tie * pair_tie * (0.5f + 0.5f * pair_res);
                    }
                }
            }

            /* Inter-layer 4x4 continuity when candidate is linked in graph & meets mirror bar on u_val */
            if (
                roles_prev != nullptr && act_prev != nullptr && prev_u_info != nullptr
                && K_prev > 0 && cand_graph_loop[c_idx] != 0
                && c_u_tie >= min_tie_bar
            ) {
                float pu_val = prev_u_info[lane];
                int k_p = (K_prev < 8) ? K_prev : 8;
                for (int p = 0; p < k_p; ++p) {
                    int ep = prev_eids[p];
                    if (ep >= 0 && ep < E && act_prev[ep] > 0) {
                        long long bp = (long long)ep * 192;
                        MirrorCubeResult p_u = eval_4x4_mirror_cube(
                            roles_prev[bp + 0 * 32 + lane],
                            roles_prev[bp + 1 * 32 + lane],
                            roles_prev[bp + 2 * 32 + lane],
                            pu_val,
                            pu_val + roles_prev[bp + 5 * 32 + lane]
                        );
                        if (p_u.tie >= min_tie_bar && p_u.tie > 1e-7f) {
                            saw_any_tie = 1;
                            if (c_u_status != 1 || c_u_res >= 0.5f) {
                                all_maintained = 0;
                                delta_cnt += 1;
                                delta_e += c_u_tie * p_u.tie;
                            }
                        }
                    }
                }
            }
        }
    }

    if (lane == 0) {
        int graph_ok = cand_graph_loop[c_idx];
        /* C03 / L04: Pure EQUAL repetition (saw_any_tie && all_maintained) adds ZERO distinction -> MAINTAINED! */
        if (saw_any_tie != 0 && all_maintained != 0 && delta_cnt == 0) {
            decisions_out[c_idx] = 2; /* mirror_maintained (rest, 0 B disk!) */
            diag_out[c_idx * 4 + 0] = min_e;
            diag_out[c_idx * 4 + 1] = 0.0f;
            diag_out[c_idx * 4 + 2] = 0.0f;
            diag_out[c_idx * 4 + 3] = rest_tol;
            return;
        }
        if ((min_cnt > 0 && delta_cnt == 0) || (min_cnt == 0 && delta_cnt == 0 && graph_ok == 0)) {
            if (unclosed_cone_protect != 0) {
                /* Law of Two Slits: NEVER drop an unseen/unclosed branch to zero inside entry cone or SPLIT! */
                decisions_out[c_idx] = 4; /* direct_pass_unclosed */
                diag_out[c_idx * 4 + 0] = min_e;
                diag_out[c_idx * 4 + 1] = 0.0f;
                diag_out[c_idx * 4 + 2] = 1.0f;
                diag_out[c_idx * 4 + 3] = rest_tol;
                return;
            }
            decisions_out[c_idx] = 1; /* no_mirror_link (skip, 0 B disk!) */
            diag_out[c_idx * 4 + 0] = min_e;
            diag_out[c_idx * 4 + 1] = 0.0f;
            diag_out[c_idx * 4 + 2] = 0.0f;
            diag_out[c_idx * 4 + 3] = 0.0f;
            return;
        }

        /* 4. Pure Mirror Equality balance on `line_spine` using exact RelationResidual (ZERO UNI!) */
        float obs_sig = fmaxf(last_res_norm, 0.5f * last_obs_norm);
        float mean_w = w_done / fmaxf(1.0f, (float)tok_done);
        float scale_next = (next_w / fmaxf(mean_w, 1e-6f)) * (1.0f + 0.5f * fmaxf(0.0f, rem_w));
        float est_delta = obs_sig * scale_next;
        float defect = est_delta / (line_norm + est_delta + 1e-8f);

        float eff_tol = rest_tol;
        if (delta_cnt > 0) {
            if (min_cnt == 0 || min_e <= 1e-8f) {
                eff_tol = 0.5f * rest_tol;
            } else {
                eff_tol = rest_tol * (min_e / (min_e + delta_e + 1e-8f));
            }
        }
        bool should_rest = (tok_done >= rest_min && defect < eff_tol);
        decisions_out[c_idx] = should_rest ? 2 : 3;
        diag_out[c_idx * 4 + 0] = min_e;
        diag_out[c_idx * 4 + 1] = delta_e;
        diag_out[c_idx * 4 + 2] = defect;
        diag_out[c_idx * 4 + 3] = eff_tol;
    }
}

/* Kernel #5: Autonomous 4x4 Mirror Self-Collapse (`pure_mirror_self_collapse_f32`)
 *
 * Grid:  (N_res, 1, 1) — 1 warp per collapsible non-active resident agent in VRAM
 * Block: (32, 1, 1)
 *
 * 1-to-1 with ring self-collapse, driven EXCLUSIVELY by the irreducible 4x4 Mirror Cube (ZERO UNI!):
 *   0 = hold_tension   : living EQ_TENSION with live (u_in, u_post) or with an active agent,
 *                        OR active mirror_bridge linked to current flow -> KEEP CRISP!
 *   1 = collapse_split : SPLIT/UNKNOWN against live flow & no active TENSION link ->
 *                        SELF-COLLAPSE 1 trailing Q1 bit-plane in VRAM + halve stale link_energy!
 *   2 = collapse_equal : EQ_EQUAL (C03 already covered by current step) ->
 *                        SELF-COLLAPSE 1 trailing Q1 bit-plane in VRAM (shed redundant excess)!
 */
extern "C" __global__ void pure_mirror_self_collapse_f32(
    const int* __restrict__ res_eids,        /* [N_res] collapsible non-active resident agents */
    const int* __restrict__ active_eids,     /* [K_act] agents executed on this step */
    const float* __restrict__ roles_buf,     /* [E, 6, 32] */
    float* __restrict__ link_energy,         /* [E] */
    const int* __restrict__ role_ids,        /* [E] */
    const float* __restrict__ u_in,          /* [32] */
    const float* __restrict__ u_post,        /* [32] */
    int* __restrict__ collapse_out,          /* [N_res]: 0=hold, 1=collapse_split, 2=collapse_equal */
    int N_res, int K_act, int E
) {
    int idx = blockIdx.x;
    if (idx >= N_res) return;
    int lane = threadIdx.x;
    int eid = res_eids[idx];
    if (eid < 0 || eid >= E) {
        if (lane == 0) collapse_out[idx] = 0;
        return;
    }

    long long base_e = (long long)eid * 192;
    float p0 = roles_buf[base_e + 0 * 32 + lane];
    float p1 = roles_buf[base_e + 1 * 32 + lane];
    float sc = roles_buf[base_e + 2 * 32 + lane];
    float ui = u_in[lane];
    float up = u_post[lane];

    MirrorCubeResult cu = eval_4x4_mirror_cube(p0, p1, sc, ui, up);
    MirrorCubeResult cs = eval_4x4_mirror_cube(sc, sc, sc, ui, up);
    float u_tie = fmaxf(cu.tie, cs.tie);
    int u_stat = (cs.tie > cu.tie) ? cs.eq_status : cu.eq_status;

    int hold = 0;
    if (u_tie > 1e-7f && (u_stat == 2 || role_ids[eid] == 2)) {
        hold = 1;
    }

    if (hold == 0 && u_tie > 1e-7f) {
        int k_a = (K_act < 8) ? K_act : 8;
        for (int i = 0; i < k_a; ++i) {
            int ae = active_eids[i];
            if (ae >= 0 && ae < E && ae != eid) {
                long long ba = (long long)ae * 192;
                MirrorCubeResult am = eval_4x4_mirror_cube(
                    p0, p1, sc,
                    roles_buf[ba + 1 * 32 + lane],
                    roles_buf[ba + 0 * 32 + lane]
                );
                if (am.tie > 1e-7f && am.eq_status == 2) {
                    hold = 1;
                    break;
                }
            }
        }
    }

    if (lane == 0) {
        if (hold != 0) {
            collapse_out[idx] = 0; /* hold_tension */
        } else if (u_tie <= 1e-7f || u_stat == 0 || u_stat == 3) {
            collapse_out[idx] = 1; /* collapse_split */
            link_energy[eid] *= 0.5f;
        } else {
            collapse_out[idx] = 2; /* collapse_equal (C03 redundant plane trim) */
        }
    }
}

/* Helpers for Kernel #6: Differential-Reversible Pure Mirror Q1 GEMV */
__device__ __forceinline__ float pm_bf16_to_f32(unsigned short b) {
    return __uint_as_float(((unsigned int)b) << 16);
}

__device__ __forceinline__ unsigned short pm_f32_to_bf16_rne(float val) {
    unsigned int bits = __float_as_uint(val);
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u) {
        return 0x7fc0u;
    }
    unsigned int lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return (unsigned short)(bits >> 16);
}

__device__ __forceinline__ void pm_block_fwht_and_sum(
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

__device__ __forceinline__ float pm_eval_q1_plane_chunk(
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
        float sv = pm_bf16_to_f32(s_row[g_start + (c_off / GS)]) * p_mult;
        float d1 = 0.0f;
        while (bits16) {
            int b = __ffs(bits16) - 1;
            d1 += s_x[c_off + b];
            bits16 &= (bits16 - 1u);
        }
        partial += 2.0f * sv * d1;
    }
    for (int g = lane; g < ng_cur; g += 32) {
        partial -= (pm_bf16_to_f32(s_row[g_start + g]) * p_mult) * s_xsum[g];
    }
    return partial;
}

/* Kernel #6: Differential-Reversible Pure Mirror Q1 GEMV (`pure_mirror_diff_q1_gemv_f32`)
 *
 * A genuinely NEW CUDA mechanism derived directly from the Bracketless Mirror Equality:
 *   1. Non-commutative mirror contact on the incoming activation vector `x_t` relative to the
 *      projection's crisp scar-anchor `(x_last, y_last)`:
 *         `alpha = <x_last, x_t> / (||x_last||^2 + eps)`
 *         `r_x   = x_t - alpha * x_last`    (the fresh scar of `ObservedBehavior`!)
 *      By exact algebraic linearity of W:
 *         `y_t = W * x_t = alpha * y_last + W * r_x`
 *   2. Zero-Plane `EQUAL` Short-Circuit: if `||r_x|| / ||x_t|| < 1e-4`, the central stem is
 *      already closed (`EQUAL`) -> emits `y_t = alpha * y_last` with ZERO GDDR6 weight reads!
 *   3. Otherwise, applies Block-FWHT ONLY to `r_x * col_scale` and runs Q1 bit-planes
 *      `p = 0 .. P_max - 1` ONLY on the fresh scar `r_x`, checking convergence of `Delta_y_p`
 *      against the TOTAL inherited+fresh trajectory `y_total = alpha * y_last + scar_y` so
 *      quiet/aligned layers automatically stop in 1-2 planes without manual heuristics!
 *
 * Grid:  ((N + 7) / 8, M, 1)
 * Block: (256, 1, 1) — 8 warps = 8 output rows per block
 */
extern "C" __global__ void pure_mirror_diff_q1_gemv_f32(
    const unsigned short* __restrict__ x_bf16,      /* [M, K] */
    const unsigned short* __restrict__ x_last_bf16, /* [K] crisp anchor input or nullptr */
    const unsigned short* __restrict__ y_last_bf16, /* [N] crisp anchor output or nullptr */
    const unsigned char* __restrict__ packed_lo,    /* [P_stored, N, K // 8] */
    const unsigned short* __restrict__ qscale,      /* [1 or P_stored, N, K // GS] */
    const unsigned short* __restrict__ qzero,       /* [N, K // GS] */
    const unsigned short* __restrict__ row_scale,   /* [N] */
    const unsigned short* __restrict__ col_scale,   /* [K] */
    const unsigned short* __restrict__ bias_bf16,   /* [N] or nullptr */
    void* __restrict__ y_out,                       /* [M, N] bf16 or f32 */
    int* __restrict__ used_planes_out,              /* [M, (N+7)/8] or nullptr */
    float* __restrict__ mirror_defect_out,          /* [M, (N+7)/8] or nullptr */
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
    float* sm_tot_e = s_xsum + (chunk_cap / GS);
    float* sm_step_e = sm_tot_e + 8;
    float* sm_red0 = sm_step_e + 8;
    float* sm_red1 = sm_red0 + 8;
    float* sm_red2 = sm_red1 + 8;

    int tid = threadIdx.x;
    int m = blockIdx.y;
    if (m >= M) return;

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp_id;
    bool valid_row = (row < N);

    const unsigned short* x_row = x_bf16 + (long long)m * K;
    int num_groups_full = K / GS;
    int k8 = K >> 3;
    long long plane_byte_stride = (long long)N * k8;
    long long plane_grp_stride = (long long)N * num_groups_full;
    int single_chunk = (K <= chunk_cap);
    bool has_anchor = (x_last_bf16 != nullptr && y_last_bf16 != nullptr);

    /* Step 0: Block-wide non-commutative mirror contact x_t = alpha * x_last + r_x (0 CPU sync!) */
    float alpha = 0.0f;
    float scar_ratio = 1.0f;
    if (has_anchor) {
        float l_dot = 0.0f, l_n2l = 0.0f, l_n2x = 0.0f;
        for (int c = tid; c < K; c += blockDim.x) {
            float xt = pm_bf16_to_f32(x_row[c]);
            float xl = pm_bf16_to_f32(x_last_bf16[c]);
            l_dot += xl * xt;
            l_n2l += xl * xl;
            l_n2x += xt * xt;
        }
        l_dot = warp_sum_32(l_dot);
        l_n2l = warp_sum_32(l_n2l);
        l_n2x = warp_sum_32(l_n2x);
        if (lane == 0) {
            sm_red0[warp_id] = l_dot;
            sm_red1[warp_id] = l_n2l;
            sm_red2[warp_id] = l_n2x;
        }
        __syncthreads();
        float t_dot = sm_red0[0] + sm_red0[1] + sm_red0[2] + sm_red0[3]
                    + sm_red0[4] + sm_red0[5] + sm_red0[6] + sm_red0[7];
        float t_n2l = sm_red1[0] + sm_red1[1] + sm_red1[2] + sm_red1[3]
                    + sm_red1[4] + sm_red1[5] + sm_red1[6] + sm_red1[7];
        float t_n2x = sm_red2[0] + sm_red2[1] + sm_red2[2] + sm_red2[3]
                    + sm_red2[4] + sm_red2[5] + sm_red2[6] + sm_red2[7];
        if (t_n2l > 1e-10f) {
            alpha = t_dot / (t_n2l + 1e-8f);
            float scar_n2 = fmaxf(0.0f, t_n2x - alpha * t_dot);
            scar_ratio = sqrtf(scar_n2 / (t_n2x + 1e-8f));
        } else {
            has_anchor = false;
            alpha = 0.0f;
            scar_ratio = 1.0f;
        }
        __syncthreads();

        /* Zero-Plane EQUAL Short-Circuit: fresh scar already inside EQUAL basin -> 0 B GDDR6 weights read! */
        float zero_cut = (mirror_tol > 0.0f) ? fmaxf(1e-4f, 0.25f * mirror_tol) : 1e-4f;
        if (has_anchor && scar_ratio < zero_cut) {
            if (lane == 0 && valid_row) {
                float y_val = alpha * pm_bf16_to_f32(y_last_bf16[row]);
                if (bias_bf16 != nullptr) {
                    y_val += (1.0f - alpha) * pm_bf16_to_f32(bias_bf16[row]);
                }
                long long idx = (long long)m * N + row;
                if (out_is_bf16) {
                    reinterpret_cast<unsigned short*>(y_out)[idx] = pm_f32_to_bf16_rne(y_val);
                } else {
                    reinterpret_cast<float*>(y_out)[idx] = y_val;
                }
            }
            if (tid == 0) {
                int tiles_per_row = (N + 7) >> 3;
                long long tile_idx = (long long)m * tiles_per_row + blockIdx.x;
                if (used_planes_out != nullptr) used_planes_out[tile_idx] = 0;
                if (mirror_defect_out != nullptr) mirror_defect_out[tile_idx] = scar_ratio;
            }
            return;
        }
    }

    /* Step 1: Load fresh scar r_x = x_t - alpha * x_last into s_x, apply Block-FWHT, compute qzero */
    float zero_acc = 0.0f;
    if (single_chunk) {
        int ng_cur = K / GS;
        for (int c = tid; c < K; c += blockDim.x) {
            float rx = pm_bf16_to_f32(x_row[c]) - (has_anchor ? alpha * pm_bf16_to_f32(x_last_bf16[c]) : 0.0f);
            s_x[c] = rx * pm_bf16_to_f32(col_scale[c]);
        }
        __syncthreads();
        pm_block_fwht_and_sum(s_x, s_xsum, ng_cur, GS, warp_id, lane, warps_per_block);
        __syncthreads();
        if (valid_row) {
            const unsigned short* z_row = qzero + (long long)row * num_groups_full;
            for (int g = lane; g < ng_cur; g += 32) {
                zero_acc += pm_bf16_to_f32(z_row[g]) * s_xsum[g];
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
                float rx = pm_bf16_to_f32(x_row[ka]) - (has_anchor ? alpha * pm_bf16_to_f32(x_last_bf16[ka]) : 0.0f);
                s_x[c] = rx * pm_bf16_to_f32(col_scale[ka]);
            }
            __syncthreads();
            pm_block_fwht_and_sum(s_x, s_xsum, ng_cur, GS, warp_id, lane, warps_per_block);
            __syncthreads();
            if (valid_row) {
                const unsigned short* z_row = qzero + (long long)row * num_groups_full;
                for (int g = lane; g < ng_cur; g += 32) {
                    zero_acc += pm_bf16_to_f32(z_row[g_start + g]) * s_xsum[g];
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        zero_acc += __shfl_down_sync(0xffffffffu, zero_acc, offset);
    }

    float y_anchor = (lane == 0 && valid_row && has_anchor) ? (alpha * pm_bf16_to_f32(y_last_bf16[row])) : 0.0f;
    float scar_acc = (lane == 0 && valid_row) ? zero_acc : 0.0f;
    float rs = (lane == 0 && valid_row) ? pm_bf16_to_f32(row_scale[row]) : 0.0f;
    int used_planes = 0;
    float last_defect = scar_ratio;

    /* Step 2: Q1 Bit-Plane Loop ONLY on fresh scar r_x with Total-Trajectory Mirror Self-Stop */
    for (int p = 0; p < P_max; ++p) {
        float obs_partial = 0.0f;
        float p_mult = shared_scale ? ldexpf(1.0f, -p) : 1.0f;
        const unsigned short* s_row = valid_row
            ? (shared_scale
                ? (qscale + (long long)row * num_groups_full)
                : (qscale + (long long)p * plane_grp_stride + (long long)row * num_groups_full))
            : nullptr;

        if (single_chunk) {
            if (valid_row) {
                const unsigned char* b_row = packed_lo + (long long)p * plane_byte_stride + (long long)row * k8;
                obs_partial = pm_eval_q1_plane_chunk(
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
                    float rx = pm_bf16_to_f32(x_row[ka]) - (has_anchor ? alpha * pm_bf16_to_f32(x_last_bf16[ka]) : 0.0f);
                    s_x[c] = rx * pm_bf16_to_f32(col_scale[ka]);
                }
                __syncthreads();
                pm_block_fwht_and_sum(s_x, s_xsum, ng_cur, GS, warp_id, lane, warps_per_block);
                __syncthreads();
                if (valid_row) {
                    const unsigned char* b_chunk =
                        packed_lo + (long long)p * plane_byte_stride + (long long)row * k8 + (k_start >> 3);
                    obs_partial += pm_eval_q1_plane_chunk(
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

        float obs_p = (lane == 0 && valid_row) ? obs_partial : 0.0f;
        scar_acc += obs_p;

        if (lane == 0) {
            float y_tot = valid_row ? (y_anchor + scar_acc * rs) : 0.0f;
            float y_stp = valid_row ? (obs_p * rs) : 0.0f;
            sm_tot_e[warp_id] = y_tot * y_tot;
            sm_step_e[warp_id] = y_stp * y_stp;
        }
        __syncthreads();

        float tile_tot_e = sm_tot_e[0] + sm_tot_e[1] + sm_tot_e[2] + sm_tot_e[3]
                         + sm_tot_e[4] + sm_tot_e[5] + sm_tot_e[6] + sm_tot_e[7];
        float tile_stp_e = sm_step_e[0] + sm_step_e[1] + sm_step_e[2] + sm_step_e[3]
                         + sm_step_e[4] + sm_step_e[5] + sm_step_e[6] + sm_step_e[7];
        float defect = (1.15470054f * sqrtf(tile_stp_e)) / (sqrtf(tile_tot_e) + 1e-8f);

        used_planes = p + 1;
        last_defect = defect;

        /* Self-stop as soon as fresh scar closes to EQUAL relative to FULL inherited trajectory! */
        if ((used_planes >= P_min) && (mirror_tol > 0.0f) && (defect < mirror_tol)) {
            break;
        }
        __syncthreads();
    }

    if (lane == 0 && valid_row) {
        float y_val = y_anchor + scar_acc * rs;
        if (bias_bf16 != nullptr) {
            y_val += (has_anchor ? (1.0f - alpha) : 1.0f) * pm_bf16_to_f32(bias_bf16[row]);
        }
        long long idx = (long long)m * N + row;
        if (out_is_bf16) {
            reinterpret_cast<unsigned short*>(y_out)[idx] = pm_f32_to_bf16_rne(y_val);
        } else {
            reinterpret_cast<float*>(y_out)[idx] = y_val;
        }
    }
    if (tid == 0) {
        int tiles_per_row = (N + 7) >> 3;
        long long tile_idx = (long long)m * tiles_per_row + blockIdx.x;
        if (used_planes_out != nullptr) used_planes_out[tile_idx] = used_planes;
        if (mirror_defect_out != nullptr) mirror_defect_out[tile_idx] = last_defect;
    }
}
"""

_PM_LOCK = threading.Lock()
_PM_MODULE_CACHE: Dict[Tuple[int, int], Tuple[Any, Dict[bytes, ctypes.c_void_p]]] = {}

_PM_KERNELS = (
    b"pure_mirror_cube_bind_f32",
    b"pure_mirror_eval_bridges_f32",
    b"pure_mirror_line_step_f32",
    b"pure_mirror_continuation_gate_f32",
    b"pure_mirror_self_collapse_f32",
    b"pure_mirror_diff_q1_gemv_f32",
)


def _compile_pure_mirror_ptx(major: int, minor: int) -> bytes:
    src_hash = hashlib.sha256(_PURE_MIRROR_CUDA_SOURCE.encode("utf-8")).hexdigest()[:16]
    PTX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ptx_path = PTX_CACHE_DIR / f"pure_mirror_cube4x4_sm{major}{minor}_{src_hash}.ptx"
    if ptx_path.exists():
        return ptx_path.read_bytes()

    _, nvrtc = _load_driver_and_nvrtc()
    prog = ctypes.c_void_p()
    src_b = _PURE_MIRROR_CUDA_SOURCE.encode("utf-8")
    rc = nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src_b, b"pure_mirror_cube4x4.cu", 0, None, None)
    if rc != 0:
        raise RuntimeError(f"nvrtcCreateProgram failed: {rc}")

    try:
        opts = [
            f"--gpu-architecture=compute_{major}{minor}".encode("utf-8"),
            b"--use_fast_math",
            b"-lineinfo",
        ]
        c_opts = (ctypes.c_char_p * len(opts))(*opts)
        rc = nvrtc.nvrtcCompileProgram(prog, len(opts), c_opts)
        if rc != 0:
            log_sz = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_sz))
            log_buf = ctypes.create_string_buffer(max(1, log_sz.value))
            nvrtc.nvrtcGetProgramLog(prog, log_buf)
            raise RuntimeError(
                f"NVRTC compile failed for pure_mirror_cube4x4 (rc={rc}):\n"
                f"{log_buf.value.decode('utf-8', errors='replace')}"
            )
        ptx_sz = ctypes.c_size_t()
        nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_sz))
        ptx_buf = ctypes.create_string_buffer(ptx_sz.value)
        nvrtc.nvrtcGetPTX(prog, ptx_buf)
        ptx_bytes = ptx_buf.raw
        tmp_path = ptx_path.with_suffix(".ptx.tmp")
        tmp_path.write_bytes(ptx_bytes)
        os.replace(tmp_path, ptx_path)
        return ptx_bytes
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))


def get_pure_mirror_kernels(device: torch.device) -> Tuple[Any, Dict[bytes, ctypes.c_void_p]]:
    """Compile/load cached PTX for all 4 Irreducible 4x4 Pure Mirror Equality CUDA kernels."""
    dev_idx = 0 if device.index is None else int(device.index)
    major, minor = torch.cuda.get_device_capability(dev_idx)
    cache_key = (dev_idx, major * 10 + minor)
    with _PM_LOCK:
        hit = _PM_MODULE_CACHE.get(cache_key)
        if hit is not None:
            return hit

        nvcuda, _ = _load_driver_and_nvrtc()
        torch.cuda.init()
        torch.cuda.set_device(dev_idx)
        ptx_bytes = _compile_pure_mirror_ptx(major, minor)
        mod = ctypes.c_void_p()
        rc = nvcuda.cuModuleLoadData(ctypes.byref(mod), ctypes.c_char_p(ptx_bytes))
        if rc != 0:
            raise RuntimeError(f"cuModuleLoadData(pure_mirror_cube4x4) failed: {rc}")
        fns: Dict[bytes, ctypes.c_void_p] = {}
        for kname in _PM_KERNELS:
            fn = ctypes.c_void_p()
            rc = nvcuda.cuModuleGetFunction(ctypes.byref(fn), mod, kname)
            if rc != 0:
                raise RuntimeError(f"cuModuleGetFunction({kname!r}) failed: {rc}")
            fns[kname] = fn
        _PM_MODULE_CACHE[cache_key] = (nvcuda, fns)
        return nvcuda, fns


def _eval_4x4_mirror_cube_full_cpu(
    p0: torch.Tensor, p1: torch.Tensor, s_cur: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor
) -> Dict[str, Any]:
    """Exact bit-faithful CPU PyTorch reference of bracketless `eval_4x4_mirror_cube` (with full diagnostics)."""
    P = (p0, p1, p2, p3)
    n2 = [float((v * v).sum().item()) for v in P]
    M = [[0.0] * 4 for _ in range(4)]
    R2 = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            if i == j:
                M[i][j] = 1.0 if n2[i] > 1e-10 else 0.0
                R2[i][j] = 0.0
            elif n2[i] > 1e-10 and n2[j] > 1e-10:
                dij = float((P[i] * P[j]).sum().item())
                mij = dij / (n2[i] + 1e-8)
                M[i][j] = mij
                R2[i][j] = max(0.0, n2[j] - mij * dij)
            else:
                M[i][j] = 0.0
                R2[i][j] = n2[j]

    # Inner Bracketless Central Stem (C0): ObservedBehavior_(t-1) = Strategy_t = ObservedBehavior_(t+1)
    has_stem = (n2[1] > 1e-10 and n2[2] > 1e-10)
    s_stem = 0.5 * (p1 + p2)
    tau_core = 0.0
    d_core = 1.0
    ns_cur = float(s_cur.norm().item())
    if has_stem:
        n_st_sum = float((p1 + p2).norm().item())
        n_st_dif = float((p1 - p2).norm().item())
        tau_core = ((n_st_sum - n_st_dif) / (n_st_sum + n_st_dif + 1e-8)) if n_st_sum > n_st_dif else 0.0
        if ns_cur > 1e-6:
            d1 = float((p1 - s_cur).norm().item())
            d2 = float((p2 - s_cur).norm().item())
            d_core = min(1.0, (d1 + d2) / (math.sqrt(n2[1]) + math.sqrt(n2[2]) + 2.0 * ns_cur + 1e-8))
        else:
            d_core = min(1.0, n_st_dif / (math.sqrt(n2[1]) + math.sqrt(n2[2]) + 1e-8))

    fp0 = max(0.0, M[3][0]) + max(0.0, M[2][0])
    fp1 = max(0.0, M[2][1]) + max(0.0, M[3][1])
    w0 = (1.0 + abs(M[1][0]) + fp0) if n2[0] > 1e-10 else 0.0
    w1 = (1.0 + abs(M[0][1]) + fp1) if n2[1] > 1e-10 else 0.0
    wL = w0 + w1

    fw2 = max(0.0, M[1][2]) + max(0.0, M[0][2])
    fw3 = max(0.0, M[0][3]) + max(0.0, M[1][3])
    w2 = (1.0 + abs(M[3][2]) + fw2) if n2[2] > 1e-10 else 0.0
    w3 = (1.0 + abs(M[2][3]) + fw3) if n2[3] > 1e-10 else 0.0
    wR = w2 + w3

    z = torch.zeros_like(s_cur)
    s_left = ((w0 * p0 + w1 * p1) / wL) if wL > 1e-8 else z
    s_right = ((w2 * p2 + w3 * p3) / wR) if wR > 1e-8 else z
    nL = float(s_left.norm().item())
    nR = float(s_right.norm().item())
    fp = 0.5 * (fp0 + fp1)

    if nL < 1e-6 or nR < 1e-6:
        s_cl = s_right if nR >= 1e-6 else (s_left if nL >= 1e-6 else s_cur)
        return {
            "s_left": s_left, "s_right": s_right, "s_closed": s_cl,
            "tie": 0.0, "rel_residual": (1.0 if nR >= 1e-6 else 0.0),
            "future_press": fp, "tau_core": tau_core, "d_core": d_core,
            "alpha_12": M[1][2], "alpha_21": M[2][1], "M": M, "R2": R2,
            "eq_status": EQ_UNKNOWN,
        }

    fwd_res_num = math.sqrt(R2[0][3]) + math.sqrt(R2[1][2]) + math.sqrt(R2[0][2]) + math.sqrt(R2[1][3])
    right_scale = 2.0 * (math.sqrt(n2[2]) + math.sqrt(n2[3])) + 1e-8
    rel_res = min(1.0, fwd_res_num / right_scale)

    n_sum = float((s_left + s_right).norm().item())
    n_dif = float((s_left - s_right).norm().item())
    outer_tie = ((n_sum - n_dif) / (n_sum + n_dif + 1e-8)) if n_sum > n_dif else 0.0

    if n_dif >= n_sum and tau_core <= 1e-6:
        tie, eq_st, s_cl = 0.0, EQ_SPLIT, 0.5 * (s_cur + s_right)
    elif n_dif >= n_sum and tau_core > 1e-6:
        tie, eq_st, s_cl = 0.5 * tau_core, EQ_TENSION, 0.5 * (s_stem + s_right)
    elif n_sum >= 3.0 * n_dif and rel_res < 0.5 and ((not has_stem) or tau_core >= 0.5):
        tie, eq_st = outer_tie, EQ_EQUAL
        s_outer_mid = 0.5 * (s_left + s_right)
        s_cl = (0.5 * (s_outer_mid + s_stem)) if has_stem else s_outer_mid
    else:
        tie, eq_st = max(outer_tie, 0.5 * tau_core), EQ_TENSION
        s_outer_mid = 0.5 * (s_left + s_right)
        s_mid = (0.5 * (s_outer_mid + s_stem)) if (has_stem and tau_core > outer_tie) else s_outer_mid
        s_cl = (0.5 * (s_cur + s_mid)) if ns_cur > 1e-5 else s_mid

    return {
        "s_left": s_left, "s_right": s_right, "s_closed": s_cl,
        "tie": tie, "rel_residual": rel_res, "future_press": fp,
        "tau_core": tau_core, "d_core": d_core,
        "alpha_12": M[1][2], "alpha_21": M[2][1], "M": M, "R2": R2,
        "eq_status": eq_st,
    }


def _eval_4x4_mirror_cube_cpu(
    p0: torch.Tensor, p1: torch.Tensor, s_cur: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, int]:
    """Exact bit-faithful CPU PyTorch reference of `eval_4x4_mirror_cube` (7-tuple view)."""
    r = _eval_4x4_mirror_cube_full_cpu(p0, p1, s_cur, p2, p3)
    return (
        r["s_left"], r["s_right"], r["s_closed"],
        r["tie"], r["rel_residual"], r["future_press"], r["eq_status"],
    )


@torch.no_grad()
def launch_pure_mirror_diff_q1_gemv(
    x: torch.Tensor,
    packed_lo: torch.Tensor,
    qscale: torch.Tensor,
    qzero: torch.Tensor,
    row_scale: torch.Tensor,
    col_scale: torch.Tensor,
    *,
    x_last: Optional[torch.Tensor] = None,
    y_last: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    group_size: int = 64,
    max_planes: Optional[int] = None,
    min_planes: int = 1,
    mirror_tol: float = 0.03,
    return_telemetry: bool = False,
) -> Any:
    """Execute Kernel #6 (`pure_mirror_diff_q1_gemv_f32`) with exact CPU fallback.

    Decomposes `x_t = alpha * x_last + r_x`, inherits `alpha * y_last` at O(N), short-circuits
    to 0 GDDR6 weight planes when `||r_x|| / ||x_t|| < 1e-4` (EQUAL), and otherwise evaluates
    Q1 bit-planes ONLY on fresh scar `r_x` stopping early when total trajectory defect < `mirror_tol`.
    """
    dev = x.device
    orig_shape = x.shape[:-1]
    in_features = int(packed_lo.shape[2]) * 8
    out_features = int(packed_lo.shape[1])
    stored = int(packed_lo.shape[0])
    p_max = stored if max_planes is None else max(1, min(stored, int(max_planes)))
    p_min = max(1, min(p_max, int(min_planes)))
    flat = x.reshape(-1, in_features)
    M = int(flat.shape[0])
    if M == 0:
        empty_y = torch.empty((*orig_shape, out_features), dtype=x.dtype, device=dev)
        if return_telemetry:
            return empty_y, torch.empty((0,), dtype=torch.int32, device=dev), torch.empty((0,), dtype=torch.float32, device=dev)
        return empty_y

    shared_scale = 1 if int(qscale.shape[0]) == 1 else 0
    tiles_per_row = (out_features + 7) // 8

    if dev.type == "cuda" and torch.cuda.is_available():
        x_bf16 = flat if (flat.dtype == torch.bfloat16 and flat.is_contiguous()) else flat.to(device=dev, dtype=torch.bfloat16).contiguous()
        if x_last is not None:
            xl_bf16 = x_last if (x_last.ndim == 1 and x_last.dtype == torch.bfloat16 and x_last.device == dev and x_last.is_contiguous()) else x_last.reshape(-1).to(device=dev, dtype=torch.bfloat16).contiguous()
        else:
            xl_bf16 = None
        if y_last is not None:
            yl_bf16 = y_last if (y_last.ndim == 1 and y_last.dtype == torch.bfloat16 and y_last.device == dev and y_last.is_contiguous()) else y_last.reshape(-1).to(device=dev, dtype=torch.bfloat16).contiguous()
        else:
            yl_bf16 = None
        if bias is not None:
            b_bf16 = bias if (bias.dtype == torch.bfloat16 and bias.device == dev and bias.is_contiguous()) else bias.to(device=dev, dtype=torch.bfloat16).contiguous()
        else:
            b_bf16 = None
        out_is_bf16 = 1 if x.dtype == torch.bfloat16 else 0
        y_out = torch.empty((M, out_features), dtype=torch.bfloat16 if out_is_bf16 else torch.float32, device=dev)
        used_t = torch.empty((M, tiles_per_row), dtype=torch.int32, device=dev) if return_telemetry else None
        def_t = torch.empty((M, tiles_per_row), dtype=torch.float32, device=dev) if return_telemetry else None

        nvcuda, fns = get_pure_mirror_kernels(dev)
        stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)

        p_x = ctypes.c_void_p(x_bf16.data_ptr())
        p_xl = ctypes.c_void_p(0 if xl_bf16 is None else xl_bf16.data_ptr())
        p_yl = ctypes.c_void_p(0 if yl_bf16 is None else yl_bf16.data_ptr())
        p_pk = ctypes.c_void_p((packed_lo if packed_lo.is_contiguous() else packed_lo.contiguous()).data_ptr())
        p_qs = ctypes.c_void_p((qscale if qscale.is_contiguous() else qscale.contiguous()).data_ptr())
        p_qz = ctypes.c_void_p((qzero if qzero.is_contiguous() else qzero.contiguous()).data_ptr())
        p_rs = ctypes.c_void_p((row_scale if row_scale.is_contiguous() else row_scale.contiguous()).data_ptr())
        p_cs = ctypes.c_void_p((col_scale if col_scale.is_contiguous() else col_scale.contiguous()).data_ptr())
        p_bs = ctypes.c_void_p(0 if b_bf16 is None else b_bf16.data_ptr())
        p_yo = ctypes.c_void_p(y_out.data_ptr())
        p_up = ctypes.c_void_p(0 if used_t is None else used_t.data_ptr())
        p_df = ctypes.c_void_p(0 if def_t is None else def_t.data_ptr())

        c_M, c_N, c_K, c_GS = ctypes.c_int(M), ctypes.c_int(out_features), ctypes.c_int(in_features), ctypes.c_int(int(group_size))
        c_Pm, c_Pn = ctypes.c_int(p_max), ctypes.c_int(p_min)
        c_tol = ctypes.c_float(float(mirror_tol))
        c_ob, c_ss = ctypes.c_int(out_is_bf16), ctypes.c_int(shared_scale)

        args = (ctypes.c_void_p * 21)(
            ctypes.addressof(p_x), ctypes.addressof(p_xl), ctypes.addressof(p_yl),
            ctypes.addressof(p_pk), ctypes.addressof(p_qs), ctypes.addressof(p_qz),
            ctypes.addressof(p_rs), ctypes.addressof(p_cs), ctypes.addressof(p_bs),
            ctypes.addressof(p_yo), ctypes.addressof(p_up), ctypes.addressof(p_df),
            ctypes.addressof(c_M), ctypes.addressof(c_N), ctypes.addressof(c_K), ctypes.addressof(c_GS),
            ctypes.addressof(c_Pm), ctypes.addressof(c_Pn), ctypes.addressof(c_tol),
            ctypes.addressof(c_ob), ctypes.addressof(c_ss),
        )
        chunk_cap = min(in_features, 4096)
        smem_bytes = (chunk_cap + (chunk_cap // int(group_size)) + 40) * 4
        rc = nvcuda.cuLaunchKernel(
            fns[b"pure_mirror_diff_q1_gemv_f32"],
            tiles_per_row, M, 1, 256, 1, 1, smem_bytes, stream, args, None,
        )
        if rc != 0:
            raise RuntimeError(f"cuLaunchKernel(pure_mirror_diff_q1_gemv_f32) failed: {rc}")
        y_res = (y_out if y_out.dtype == x.dtype else y_out.to(dtype=x.dtype)).view(*orig_shape, out_features)
        if return_telemetry:
            return y_res, used_t, def_t
        return y_res

    # Exact bit-faithful CPU PyTorch reference of Kernel #6
    from src.binary_halving_q1 import _plane_scale_f32, _unpack_sign_bits_f32
    from src.ideal_q1_engine import fwht_block

    x_f = flat.to(dtype=torch.float32)
    cs_f = col_scale.to(device=dev, dtype=torch.float32)
    rs_f = row_scale.to(device=dev, dtype=torch.float32)
    qz_f = qzero.to(device=dev, dtype=torch.float32)
    ng = in_features // int(group_size)
    has_anchor = (x_last is not None and y_last is not None)
    xl_f = x_last.reshape(-1).to(device=dev, dtype=torch.float32) if has_anchor else None
    yl_f = y_last.reshape(-1).to(device=dev, dtype=torch.float32) if has_anchor else None
    n2_l = float((xl_f * xl_f).sum().item()) if has_anchor else 0.0
    if n2_l <= 1e-10:
        has_anchor = False

    y_rows = []
    used_rows = []
    def_rows = []
    for mi in range(M):
        xm = x_f[mi]
        alpha = 0.0
        scar_ratio = 1.0
        if has_anchor:
            dot_lx = float((xl_f * xm).sum().item())
            n2_x = float((xm * xm).sum().item())
            alpha = dot_lx / (n2_l + 1e-8)
            scar_n2 = max(0.0, n2_x - alpha * dot_lx)
            scar_ratio = math.sqrt(scar_n2 / (n2_x + 1e-8))
            zero_cut = max(1e-4, 0.25 * mirror_tol) if mirror_tol > 0.0 else 1e-4
            if scar_ratio < zero_cut:
                yv = alpha * yl_f
                if bias is not None:
                    yv = yv + (1.0 - alpha) * bias.to(device=dev, dtype=torch.float32)
                y_rows.append(yv)
                used_rows.append(torch.zeros((tiles_per_row,), dtype=torch.int32, device=dev))
                def_rows.append(torch.full((tiles_per_row,), scar_ratio, dtype=torch.float32, device=dev))
                continue

        rx = (xm - alpha * xl_f) if has_anchor else xm
        x_rot = fwht_block((rx * cs_f).unsqueeze(0), int(group_size)).squeeze(0).to(torch.float32)
        x_sum = x_rot.reshape(ng, int(group_size)).sum(dim=-1)
        y_anc = (alpha * yl_f) if has_anchor else torch.zeros((out_features,), dtype=torch.float32, device=dev)
        scar_acc = (qz_f * x_sum.unsqueeze(0)).sum(dim=-1)

        # Evaluate plane-by-plane with per-8-row tile convergence matching CUDA blocks!
        active_tile = [True] * tiles_per_row
        tile_used = [0] * tiles_per_row
        tile_def = [scar_ratio] * tiles_per_row
        y_final = y_anc + scar_acc * rs_f
        for p in range(p_max):
            if not any(active_tile):
                break
            s_f = _unpack_sign_bits_f32(packed_lo[p]).reshape(out_features, ng, int(group_size))
            sc_p = _plane_scale_f32(qscale, p).unsqueeze(-1)
            obs_p = (s_f * sc_p * x_rot.reshape(1, ng, int(group_size))).sum(dim=(-1, -2))
            for ti in range(tiles_per_row):
                if not active_tile[ti]:
                    continue
                r0 = ti * 8
                r1 = min(out_features, r0 + 8)
                scar_acc[r0:r1] = scar_acc[r0:r1] + obs_p[r0:r1]
                y_tot = y_anc[r0:r1] + scar_acc[r0:r1] * rs_f[r0:r1]
                y_stp = obs_p[r0:r1] * rs_f[r0:r1]
                y_final[r0:r1] = y_tot
                te = float((y_tot * y_tot).sum().item())
                se = float((y_stp * y_stp).sum().item())
                df = (1.15470054 * math.sqrt(se)) / (math.sqrt(te) + 1e-8)
                tile_used[ti] = p + 1
                tile_def[ti] = df
                if (p + 1) >= p_min and mirror_tol > 0.0 and df < mirror_tol:
                    active_tile[ti] = False
        if bias is not None:
            y_final = y_final + ((1.0 - alpha) if has_anchor else 1.0) * bias.to(device=dev, dtype=torch.float32)
        y_rows.append(y_final)
        used_rows.append(torch.tensor(tile_used, dtype=torch.int32, device=dev))
        def_rows.append(torch.tensor(tile_def, dtype=torch.float32, device=dev))

    y_res = torch.stack(y_rows, dim=0).to(dtype=x.dtype).view(*orig_shape, out_features)
    if return_telemetry:
        return y_res, torch.stack(used_rows, dim=0), torch.stack(def_rows, dim=0)
    return y_res


@torch.no_grad()
def launch_pure_mirror_line_step(
    line_spine: torch.Tensor,
    y_e: torch.Tensor,
    tok_indices: torch.Tensor,
    row_weights: torch.Tensor,
    *,
    out_metrics: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, bool]:
    """Execute fused Mirror Equality update on `line_spine`:
         Strategy_t = Outcome_(t-1) + ObservedBehavior_(t-1)
    Returns `(metrics [R, 4], used_cuda)` where columns are
    `(strat_norm, obs_norm, residual_norm, mirror_tie)`.
    """
    dev = line_spine.device
    R = int(tok_indices.shape[0])
    H = int(line_spine.shape[1])
    if R == 0:
        return torch.empty((0, 4), dtype=torch.float32, device=dev), False

    y_f32 = y_e if (y_e.dtype == torch.float32 and y_e.is_contiguous()) else y_e.to(dtype=torch.float32).contiguous()
    w_f32 = row_weights if (row_weights.dtype == torch.float32 and row_weights.device == dev and row_weights.is_contiguous()) else row_weights.to(device=dev, dtype=torch.float32).contiguous()

    if dev.type == "cuda" and torch.cuda.is_available():
        tok_i32 = tok_indices if (tok_indices.dtype == torch.int32 and tok_indices.device == dev and tok_indices.is_contiguous()) else tok_indices.to(device=dev, dtype=torch.int32).contiguous()
        metrics = out_metrics if (out_metrics is not None and out_metrics.shape == (R, 4) and out_metrics.device == dev) else torch.empty((R, 4), dtype=torch.float32, device=dev)
        nvcuda, fns = get_pure_mirror_kernels(dev)
        p_line = ctypes.c_void_p(line_spine.data_ptr())
        p_y = ctypes.c_void_p(y_f32.data_ptr())
        p_tok = ctypes.c_void_p(tok_i32.data_ptr())
        p_w = ctypes.c_void_p(w_f32.data_ptr())
        p_out = ctypes.c_void_p(metrics.data_ptr())
        c_R, c_H = ctypes.c_int(R), ctypes.c_int(H)
        args = (ctypes.c_void_p * 7)(
            ctypes.addressof(p_line), ctypes.addressof(p_y),
            ctypes.addressof(p_tok), ctypes.addressof(p_w),
            ctypes.addressof(p_out), ctypes.addressof(c_R), ctypes.addressof(c_H),
        )
        stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
        rc = nvcuda.cuLaunchKernel(
            fns[b"pure_mirror_line_step_f32"],
            R, 1, 1, 256, 1, 1, 0, stream, args, None,
        )
        if rc != 0:
            raise RuntimeError(f"cuLaunchKernel(pure_mirror_line_step_f32) failed: {rc}")
        return metrics, True

    tok_l = tok_indices.to(device=dev, dtype=torch.long)
    obs = y_f32 * w_f32.unsqueeze(-1)
    prev = line_spine.index_select(0, tok_l)
    strat = prev + obs
    dif = prev - obs
    line_spine.index_add_(0, tok_l, obs)
    np_v = strat.norm(dim=-1)
    nm_v = dif.norm(dim=-1)
    no_v = obs.norm(dim=-1)
    prev_sq = (prev * prev).sum(dim=-1)
    obs_sq = (obs * obs).sum(dim=-1)
    dots = (prev * obs).sum(dim=-1)
    res_sq = torch.where(prev_sq > 1e-10, obs_sq - (dots * dots) / (prev_sq + 1e-8), obs_sq).clamp_min(0.0)
    nres = res_sq.sqrt()
    tie = torch.where(np_v > nm_v, (np_v - nm_v) / (np_v + nm_v + 1e-8), torch.zeros_like(np_v))
    return torch.stack([np_v, no_v, nres, tie], dim=-1), False


class PureMirrorCudaFeedback:
    """Version 3 Engine: Irreducible 4x4 Mirror Equality + D15 Logic<->Strategy on NVRTC CUDA.

    Preserves all 4 contact surfaces (`Outcome_(t-1)`, `ObservedBehavior_(t-1)`,
    `ObservedBehavior_(t+1)`, `Outcome_(t+1)`), center `Strategy_t`, and `Logic_t` separately
    per agent in VRAM (`[L, E, 6, 32]`). ZERO GPU->CPU syncs on decode `observe_mirror_step`.
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        *,
        device: Any = None,
        proj_dim: int = 32,
    ) -> None:
        self.num_layers: int = max(1, int(num_layers))
        self.num_experts: int = max(1, int(num_experts))
        self.proj_dim: int = 32  # Fixed to 1 CUDA warp (32 lanes)
        self.device: torch.device = torch.device("cpu" if device is None else device)
        self.use_cuda_kernels: bool = bool(self.device.type == "cuda" and torch.cuda.is_available())

        L, E, D = self.num_layers, self.num_experts, self.proj_dim
        # Irreducible 6-role tensor per agent: [L, E, 6, 32] (~7.5 MiB VRAM for 40x256 35B!)
        self.roles: torch.Tensor = torch.zeros((L, E, 6, D), dtype=torch.float32, device=self.device)
        self.active_steps: torch.Tensor = torch.zeros((L, E), dtype=torch.int32, device=self.device)
        self.link_energy: torch.Tensor = torch.zeros((L, E), dtype=torch.float32, device=self.device)
        self.last_seen: torch.Tensor = torch.zeros((L, E), dtype=torch.int32, device=self.device)
        self.partner_eids: torch.Tensor = torch.full((L, E, 8), -1, dtype=torch.int32, device=self.device)
        self.partner_w: torch.Tensor = torch.zeros((L, E, 8), dtype=torch.float32, device=self.device)
        self.bridge_ratio: torch.Tensor = torch.zeros((L, E), dtype=torch.float32, device=self.device)
        self.role_ids: torch.Tensor = torch.zeros((L, E), dtype=torch.int32, device=self.device)

        # GPU-resident peak & link counters (eliminates per-layer decode GPU->CPU sync!)
        self.layer_peak_energy: torch.Tensor = torch.full((L, 1), 1e-6, dtype=torch.float32, device=self.device)
        self.layer_links_gpu: torch.Tensor = torch.zeros((L, 1), dtype=torch.int32, device=self.device)

        # Born Separate Multi-Ray Layer Node Y_L & Algebraically Reversible Fresh Scar Gamma_scar(Y_L)
        # Participants (`self.roles`) remain INTACT; collective path object lives here (`~48 KB` for 40L!).
        self.layer_nodes: torch.Tensor = torch.zeros((L, 6, D), dtype=torch.float32, device=self.device)
        self.layer_node_meta: torch.Tensor = torch.zeros((L, 4), dtype=torch.float32, device=self.device)
        self.layer_node_steps: torch.Tensor = torch.zeros((L, 1), dtype=torch.int32, device=self.device)
        self.layer_scar_eids: torch.Tensor = torch.full((L, 8), -1, dtype=torch.int32, device=self.device)
        self.layer_scar_alpha: torch.Tensor = torch.zeros((L, 8), dtype=torch.float32, device=self.device)
        self.layer_scar_res: torch.Tensor = torch.zeros((L, 8, D), dtype=torch.float32, device=self.device)

        # Per-layer anisotropy-aware mean
        self._layer_mean: torch.Tensor = torch.zeros((L, D), dtype=torch.float32, device=self.device)
        self._layer_mean_init: List[bool] = [False] * L
        self._layer_step: List[int] = [0] * L

        # Cached orthonormal projection matrices `R^H -> R^32`
        self._proj_cache: Dict[int, torch.Tensor] = {}

        # Lazy host mirror synced ONLY when a dirty layer is actually queried (`target_depth`/`stats`/`export`)
        self._dirty_layers: Dict[int, set[int]] = {}
        self.role_of: Dict[Tuple[int, int], str] = {}
        self._host_link_rel: Dict[Tuple[int, int], float] = {}
        self._host_bridge_ratio: Dict[Tuple[int, int], float] = {}
        self._layer_link_events_host: List[int] = [0] * L

        # Telemetry
        self.total_bind_launches: int = 0
        self.total_role_launches: int = 0
        self.total_line_launches: int = 0
        self.total_gate_launches: int = 0
        self.total_collapse_launches: int = 0
        self.total_diff_q1_launches: int = 0
        self.self_collapse_split: int = 0
        self.self_collapse_equal: int = 0
        self.self_collapse_held: int = 0
        self.topic_shift_continuations: int = 0
        self.continuation_comparisons: int = 0
        self.continuation_new_links: int = 0
        self.continuation_maintained: int = 0
        self.continuation_no_link: int = 0
        self.continuation_direct_unclosed: int = 0
        self.mutual_steps: int = 0
        self.total_step_ms: float = 0.0
        self.last_step_ms: float = 0.0

        if self.use_cuda_kernels:
            get_pure_mirror_kernels(self.device)

    # Direct read-only views into the 6 irreducible role tensors (NEVER collapsed)
    @property
    def left_outer(self) -> torch.Tensor:
        return self.roles[:, :, SLOT_LEFT_OUTER, :]

    @property
    def left_inner(self) -> torch.Tensor:
        return self.roles[:, :, SLOT_LEFT_INNER, :]

    @property
    def S(self) -> torch.Tensor:
        return self.roles[:, :, SLOT_CENTER, :]

    @property
    def right_inner(self) -> torch.Tensor:
        return self.roles[:, :, SLOT_RIGHT_INNER, :]

    @property
    def right_outer(self) -> torch.Tensor:
        return self.roles[:, :, SLOT_RIGHT_OUTER, :]

    @property
    def Logic(self) -> torch.Tensor:
        return self.roles[:, :, SLOT_LOGIC, :]

    def _get_projection(self, h_dim: int) -> torch.Tensor:
        proj = self._proj_cache.get(int(h_dim))
        if proj is None:
            d = self.proj_dim
            gen = torch.Generator(device="cpu")
            gen.manual_seed(20261122 ^ (int(h_dim) * 1315423911 & 0x7FFFFFFF))
            if h_dim >= d:
                raw = torch.randn((int(h_dim), d), generator=gen, dtype=torch.float32)
                q_mat, _ = torch.linalg.qr(raw, mode="reduced")
                proj_cpu = q_mat.contiguous()
            else:
                proj_cpu = torch.zeros((int(h_dim), d), dtype=torch.float32)
                proj_cpu[:h_dim, :h_dim] = torch.eye(int(h_dim), dtype=torch.float32)
            proj = proj_cpu.to(device=self.device, dtype=torch.float32)
            self._proj_cache[int(h_dim)] = proj
        return proj

    @torch.no_grad()
    def project_states(
        self,
        layer_idx: int,
        states: torch.Tensor,
        *,
        update_mean: bool = True,
    ) -> torch.Tensor:
        """Project `[M, H]` states into unit-sphere `R^{32}` on `self.device` (0 CPU copy on decode)."""
        if states.ndim == 1:
            states = states.unsqueeze(0)
        st_dev = states.detach().to(device=self.device, dtype=torch.float32)
        z = st_dev @ self._get_projection(int(st_dev.shape[-1]))
        li = int(layer_idx)
        if 0 <= li < self.num_layers:
            if update_mean and int(z.shape[0]) >= 4:
                b_mean = z.mean(dim=0)
                mean_sq = float((b_mean * b_mean).sum().item())
                avg_sq = float((z * z).sum(dim=-1).mean().item()) + 1e-8
                rho = min(0.95, mean_sq / avg_sq)
                scaled_mean = b_mean * rho
                if not self._layer_mean_init[li]:
                    spread = (z - b_mean).norm(dim=-1).mean()
                    raw_n = z.norm(dim=-1).mean() + 1e-8
                    if float(spread.item()) > 1e-3 * float(raw_n.item()):
                        self._layer_mean[li].copy_(scaled_mean)
                        self._layer_mean_init[li] = True
                else:
                    self._layer_mean[li].add_(scaled_mean).mul_(0.5)

            if self._layer_mean_init[li]:
                zc = z - self._layer_mean[li]
                nc = zc.norm(dim=-1, keepdim=True)
                nr = z.norm(dim=-1, keepdim=True) + 1e-8
                u = torch.where(nc > (1e-3 * nr), zc / nc.clamp_min(1e-8), z / nr)
                return u.contiguous()

        nr = z.norm(dim=-1, keepdim=True) + 1e-8
        return (z / nr).contiguous()

    def on_topic_shift(self, layer: Optional[int] = None) -> None:
        """Record topic shift WITHOUT resetting past surfaces P0/P1, Strategy S, or Logic."""
        self.topic_shift_continuations += 1

    def _sync_layer_to_host(self, li: int) -> None:
        """Single batched GPU->CPU readback for dirty agents of layer `li` (lazy, never per-decode-step!)."""
        dirty = self._dirty_layers.pop(li, None)
        if not dirty:
            return
        eids = sorted(dirty)
        idx_t = torch.tensor(eids, dtype=torch.long, device=self.device)
        packed = torch.stack(
            [
                self.role_ids[li].index_select(0, idx_t).to(torch.float32),
                self.bridge_ratio[li].index_select(0, idx_t),
                self.link_energy[li].index_select(0, idx_t),
            ],
            dim=-1,
        ).cpu().tolist()
        pk = max(1e-6, float(self.layer_peak_energy[li, 0].item()))
        self._layer_link_events_host[li] = int(self.layer_links_gpu[li, 0].item())
        role_names = ("mirror_periphery", "mirror_core", "mirror_bridge")
        for eid, (rf, br, ev) in zip(eids, packed):
            rid = int(round(rf))
            key = (li, eid)
            self.role_of[key] = role_names[rid] if 0 <= rid <= 2 else "mirror_periphery"
            self._host_link_rel[key] = min(1.0, ev / pk)
            self._host_bridge_ratio[key] = br

    def _sync_all_dirty(self) -> None:
        for li in list(self._dirty_layers.keys()):
            self._sync_layer_to_host(li)

    def _bind_and_eval_cpu(
        self,
        li: int,
        u_in: torch.Tensor,
        u_post: Optional[torch.Tensor],
        tok_eids: List[List[int]],
        tok_ws: List[List[float]],
        cur_step: int,
        is_commit: bool,
    ) -> List[int]:
        """Exact bit-faithful CPU PyTorch reference of Kernels #1 and #2."""
        E = self.num_experts
        R_l = self.roles[li]
        act_l = self.active_steps[li]
        le_l = self.link_energy[li]
        ls_l = self.last_seen[li]
        pe_l = self.partner_eids[li]
        pw_l = self.partner_w[li]
        pk = max(1e-6, float(self.layer_peak_energy[li, 0].item()) * 0.5)
        touched_set: set[int] = set()
        links_added = 0

        for tok_i, (eids_row, ws_row) in enumerate(zip(tok_eids, tok_ws)):
            valid = [(e, w) for e, w in zip(eids_row, ws_row) if 0 <= e < E]
            if not valid:
                continue
            ui = u_in[tok_i]
            up = u_post[tok_i] if u_post is not None else ui
            mean_w = sum(w for _, w in valid) / len(valid)
            half_mean = 0.5 * mean_w

            # Stage A: 4x4 cube with incoming/materialized data
            stage_rows = []
            for eid, w_s in valid:
                touched_set.add(eid)
                p0 = R_l[eid, SLOT_LEFT_OUTER].clone()
                p1 = R_l[eid, SLOT_LEFT_INNER].clone()
                s_cur = R_l[eid, SLOT_CENTER].clone()
                lg = R_l[eid, SLOT_LOGIC].clone()
                prev_act = int(act_l[eid].item())

                p2 = ui
                p3 = up if (is_commit and u_post is not None) else (ui + lg)
                if prev_act == 0:
                    s_cur = 0.5 * (p2 + p3)
                    if is_commit:
                        lg = p3 - p2
                else:
                    _, _, s_cl, _, _, _, _ = _eval_4x4_mirror_cube_cpu(p0, p1, s_cur, p2, p3)
                    s_cur = s_cl
                    if is_commit and u_post is not None:
                        lg = 0.5 * (lg + (up - ui))

                p0_pub = p0 if prev_act > 0 else p3
                p1_pub = p1 if prev_act > 0 else p2
                stage_rows.append((eid, w_s, prev_act, p0_pub, p1_pub, s_cur, p2, p3, lg))

            # Stage B: Mutual 4x4 contacts build peer links & D15 collective candidates for Y_L
            # WITHOUT overwriting or smearing the participants' own Strategy_self / Logic_self!
            n_v = len(stage_rows)
            cand_s_list: List[torch.Tensor] = []
            cand_l_list: List[torch.Tensor] = []
            for i in range(n_v):
                eid, w_s, prev_act, p0_s, p1_s, s_cur, p2, p3, lg = stage_rows[i]
                cand_s_acc = torch.zeros_like(ui)
                cand_l_acc = torch.zeros_like(ui)
                ref_s_acc = torch.zeros_like(ui)
                peer_wsum = 0.0
                tok_le = 0.0

                for j in range(n_v):
                    if i == j:
                        continue
                    peid, w_q, _, p0_q, p1_q, s_q, _, _, l_q = stage_rows[j]
                    if peid == eid:
                        continue
                    _, _, s_cl_pq, tie_pq, res_pq, _, eq_pq = _eval_4x4_mirror_cube_cpu(
                        p0_s, p1_s, s_cur, p1_q, p0_q
                    )
                    tie_eff = tie_pq
                    if eq_pq == EQ_SPLIT:
                        _, _, _, tie_rw, _, _, _ = _eval_4x4_mirror_cube_cpu(p3, p2, s_cur, p1_q, p0_q)
                        tie_eff = max(tie_eff, 0.5 * tie_rw)

                    if tie_eff > 1e-7:
                        d15_cand_s = 0.5 * (s_cur + s_q) + 0.5 * (lg + l_q)
                        d15_cand_l = s_q - s_cur
                        cand_s_acc = cand_s_acc + tie_eff * d15_cand_s
                        cand_l_acc = cand_l_acc + tie_eff * d15_cand_l
                        ref_s_acc = ref_s_acc + tie_eff * s_cl_pq
                        peer_wsum += tie_eff
                        lw = tie_eff * (0.5 + 0.5 * res_pq) * math.sqrt(max(0.0, w_s * w_q))
                        tok_le += lw
                        if eid < peid:
                            links_added += 1
                        cur_ps = pe_l[eid].tolist()
                        cur_ws = pw_l[eid].tolist()
                        if peid in cur_ps:
                            idx_s = cur_ps.index(peid)
                            pw_l[eid, idx_s] = 0.5 * (cur_ws[idx_s] + lw)
                        elif -1 in cur_ps:
                            idx_s = cur_ps.index(-1)
                            pe_l[eid, idx_s] = peid
                            pw_l[eid, idx_s] = lw
                        else:
                            min_s = min(range(8), key=lambda k: cur_ws[k])
                            if lw > cur_ws[min_s]:
                                pe_l[eid, min_s] = peid
                                pw_l[eid, min_s] = lw

                if peer_wsum > 1e-8:
                    cand_s_list.append(0.5 * ((ref_s_acc + cand_s_acc) / peer_wsum))
                    cand_l_list.append(cand_l_acc / peer_wsum)
                else:
                    cand_s_list.append(s_cur)
                    cand_l_list.append(lg)

                # INTACT PARTICIPANT WRITEBACK: own Strategy_self and Logic_self NEVER overwritten by peers!
                R_l[eid, SLOT_CENTER].copy_(s_cur)
                R_l[eid, SLOT_RIGHT_INNER].copy_(p2)
                R_l[eid, SLOT_RIGHT_OUTER].copy_(p3)
                R_l[eid, SLOT_LOGIC].copy_(lg)
                if is_commit or prev_act == 0:
                    R_l[eid, SLOT_LEFT_OUTER].copy_(p3)
                    R_l[eid, SLOT_LEFT_INNER].copy_(p2)

                age = cur_step - int(ls_l[eid].item())
                prev_e = float(le_l[eid].item())
                if age > 0:
                    prev_e *= math.ldexp(1.0, -min(age, 16))
                new_le = prev_e + tok_le
                le_l[eid] = new_le
                if new_le > pk:
                    pk = new_le
                ls_l[eid] = cur_step
                if is_commit or prev_act == 0:
                    act_l[eid] = prev_act + 1

            # Stage C: Birth / evolve SEPARATE Multi-Ray Layer Node Y_L + Exact Reversible Fresh Scar Gamma_scar
            y_cs = torch.zeros_like(ui)
            y_cl = torch.zeros_like(ui)
            y_ws = 0.0
            for q_i, (_, w_q, *_rest) in enumerate(stage_rows):
                wq = max(w_q, 1e-4)
                y_cs = y_cs + wq * cand_s_list[q_i]
                y_cl = y_cl + wq * cand_l_list[q_i]
                y_ws += wq
            if y_ws > 1e-8:
                y_cs = y_cs / y_ws
                y_cl = y_cl / y_ws

            Y_l = self.layer_nodes[li]
            yp0 = Y_l[SLOT_LEFT_OUTER].clone()
            yp1 = Y_l[SLOT_LEFT_INNER].clone()
            ys_cur = Y_l[SLOT_CENTER].clone()
            ylg = Y_l[SLOT_LOGIC].clone()
            y_prev_st = int(self.layer_node_steps[li, 0].item())

            yp2 = ui
            yp3 = up if (is_commit and u_post is not None) else (ui + ylg)
            meta_tc, meta_rr, meta_fp, meta_st = 0.0, 1.0, 0.0, 0.0
            if y_prev_st == 0:
                s_init = 0.5 * (yp2 + yp3)
                ys_new = (0.5 * (s_init + y_cs)) if y_ws > 1e-8 else s_init
                ylg_new = (up - ui) if (is_commit and u_post is not None) else y_cl
            else:
                ym = _eval_4x4_mirror_cube_full_cpu(yp0, yp1, ys_cur, yp2, yp3)
                ys_new = (0.5 * (ym["s_closed"] + y_cs)) if y_ws > 1e-8 else ym["s_closed"]
                ylg_new = (0.5 * (ylg + (up - ui))) if (is_commit and u_post is not None) else ylg
                meta_tc = ym["tau_core"]
                meta_rr = ym["rel_residual"]
                meta_fp = ym["future_press"]
                meta_st = float(ym["eq_status"])

            Y_l[SLOT_CENTER].copy_(ys_new)
            Y_l[SLOT_RIGHT_INNER].copy_(yp2)
            Y_l[SLOT_RIGHT_OUTER].copy_(yp3)
            Y_l[SLOT_LOGIC].copy_(ylg_new)
            if is_commit or y_prev_st == 0:
                Y_l[SLOT_LEFT_OUTER].copy_(yp3)
                Y_l[SLOT_LEFT_INNER].copy_(yp2)
                self.layer_node_steps[li, 0] = y_prev_st + 1
            self.layer_node_meta[li, 0] = meta_tc
            self.layer_node_meta[li, 1] = meta_rr
            self.layer_node_meta[li, 2] = meta_fp
            self.layer_node_meta[li, 3] = meta_st

            if tok_i == len(tok_eids) - 1:
                for q_i in range(8):
                    if q_i < n_v:
                        q_eid = stage_rows[q_i][0]
                        sk = stage_rows[q_i][5]
                        nk2 = float((sk * sk).sum().item())
                        dky = float((sk * ys_new).sum().item())
                        ak = (dky / (nk2 + 1e-8)) if nk2 > 1e-10 else 0.0
                        rk = ys_new - ak * sk
                        self.layer_scar_eids[li, q_i] = q_eid
                        self.layer_scar_alpha[li, q_i] = ak
                        self.layer_scar_res[li, q_i].copy_(rk)
                    else:
                        self.layer_scar_eids[li, q_i] = -1
                        self.layer_scar_alpha[li, q_i] = 0.0
                        self.layer_scar_res[li, q_i].zero_()

        self.layer_peak_energy[li, 0] = pk
        self.layer_links_gpu[li, 0] = int(self.layer_links_gpu[li, 0].item()) + links_added
        touched = sorted(touched_set)

        # Kernel #2 reference: Full 4x4 non-commutative bridge check + stale SPLIT link self-collapse
        for eid in touched:
            p0_e, p1_e, s_e = R_l[eid, 0], R_l[eid, 1], R_l[eid, 2]
            peers = []
            for s_idx, p in enumerate(pe_l[eid].tolist()):
                if 0 <= p < E and p != eid:
                    _, _, _, t1, _, _, _ = _eval_4x4_mirror_cube_cpu(p0_e, p1_e, s_e, R_l[p, 1], R_l[p, 0])
                    _, _, _, t2, _, _, _ = _eval_4x4_mirror_cube_cpu(s_e, s_e, s_e, R_l[p, 1], R_l[p, 0])
                    if max(t1, t2) > 1e-7:
                        peers.append(p)
                    elif (cur_step - int(ls_l[p].item())) >= 4:
                        pe_l[eid, s_idx] = -1
                        pw_l[eid, s_idx] = 0.0
            tot_p = 0
            disj_p = 0
            for i in range(len(peers)):
                pa = peers[i]
                for j in range(i + 1, len(peers)):
                    pb = peers[j]
                    _, _, _, tab, _, _, _ = _eval_4x4_mirror_cube_cpu(R_l[pa, 0], R_l[pa, 1], R_l[pa, 2], R_l[pb, 1], R_l[pb, 0])
                    _, _, _, tba, _, _, _ = _eval_4x4_mirror_cube_cpu(R_l[pb, 0], R_l[pb, 1], R_l[pb, 2], R_l[pa, 1], R_l[pa, 0])
                    tot_p += 1
                    if tab <= 1e-7 and tba <= 1e-7:
                        disj_p += 1
            b_rat = (disj_p / tot_p) if tot_p > 0 else 0.0
            self.bridge_ratio[li, eid] = b_rat
            ev = float(le_l[eid].item())
            if ev > 1e-9 and tot_p > 0 and (disj_p * 2 >= tot_p):
                rid = 2
            elif ev > 1e-9 and (4.0 * ev >= max(pk, 1e-6)):
                rid = 1
            else:
                rid = 0
            self.role_ids[li, eid] = rid
        return touched

    @torch.no_grad()
    def observe_mirror_step(
        self,
        layer_idx: int,
        token_routes: List[List[Tuple[int, float]]],
        u_in: Optional[torch.Tensor],
        u_post: Optional[torch.Tensor] = None,
        *,
        is_commit: bool = True,
    ) -> int:
        """Run full 4x4 Mirror Cube binding + mutual D15 co-generation (0 GPU->CPU syncs on decode!)."""
        t0 = time.perf_counter()
        li = int(layer_idx)
        if not (0 <= li < self.num_layers) or u_in is None or not token_routes:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.mutual_steps += 1
            self.last_step_ms = dt_ms
            self.total_step_ms += dt_ms
            return 0

        self._layer_step[li] += 1
        cur_step = self._layer_step[li]

        M = min(len(token_routes), int(u_in.shape[0]))
        K_slot = 8
        eids_rows: List[List[int]] = []
        ws_rows: List[List[float]] = []
        touched_set: set[int] = set()
        for tok_i in range(M):
            pairs = token_routes[tok_i]
            tot_w = sum(rw for _, rw in pairs) or 1.0
            er = [-1] * K_slot
            wr = [0.0] * K_slot
            pos = 0
            for e, rw in pairs[:K_slot]:
                ei = int(e)
                if 0 <= ei < self.num_experts:
                    er[pos] = ei
                    wr[pos] = float(rw) / tot_w
                    touched_set.add(ei)
                    pos += 1
            eids_rows.append(er)
            ws_rows.append(wr)

        if not touched_set:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.mutual_steps += 1
            self.last_step_ms = dt_ms
            self.total_step_ms += dt_ms
            return 0

        touched = sorted(touched_set)
        if self.use_cuda_kernels:
            ui_c = u_in[:M].to(device=self.device, dtype=torch.float32).contiguous()
            up_c = (
                u_post[:M].to(device=self.device, dtype=torch.float32).contiguous()
                if u_post is not None
                else None
            )
            eids_t = torch.tensor(eids_rows, dtype=torch.int32, device=self.device)
            ws_t = torch.tensor(ws_rows, dtype=torch.float32, device=self.device)
            touched_t = torch.tensor(touched, dtype=torch.int32, device=self.device)

            nvcuda, fns = get_pure_mirror_kernels(self.device)
            stream = ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)

            # Launch Kernel #1: pure_mirror_cube_bind_f32 (intact participants + born Y_L + scar Gamma_scar!)
            p_ui = ctypes.c_void_p(ui_c.data_ptr())
            p_up = ctypes.c_void_p(0 if up_c is None else up_c.data_ptr())
            p_eids = ctypes.c_void_p(eids_t.data_ptr())
            p_ws = ctypes.c_void_p(ws_t.data_ptr())
            p_R = ctypes.c_void_p(self.roles[li].data_ptr())
            p_act = ctypes.c_void_p(self.active_steps[li].data_ptr())
            p_le = ctypes.c_void_p(self.link_energy[li].data_ptr())
            p_ls = ctypes.c_void_p(self.last_seen[li].data_ptr())
            p_pe = ctypes.c_void_p(self.partner_eids[li].data_ptr())
            p_pw = ctypes.c_void_p(self.partner_w[li].data_ptr())
            p_pk = ctypes.c_void_p(self.layer_peak_energy[li].data_ptr())
            p_lout = ctypes.c_void_p(self.layer_links_gpu[li].data_ptr())
            p_yn = ctypes.c_void_p(self.layer_nodes[li].data_ptr())
            p_ym = ctypes.c_void_p(self.layer_node_meta[li].data_ptr())
            p_ys = ctypes.c_void_p(self.layer_node_steps[li].data_ptr())
            p_se = ctypes.c_void_p(self.layer_scar_eids[li].data_ptr())
            p_sa = ctypes.c_void_p(self.layer_scar_alpha[li].data_ptr())
            p_sr = ctypes.c_void_p(self.layer_scar_res[li].data_ptr())
            c_M, c_K, c_E, c_st, c_cm = (
                ctypes.c_int(M), ctypes.c_int(K_slot), ctypes.c_int(self.num_experts),
                ctypes.c_int(cur_step), ctypes.c_int(1 if is_commit else 0),
            )
            args1 = (ctypes.c_void_p * 23)(
                ctypes.addressof(p_ui), ctypes.addressof(p_up),
                ctypes.addressof(p_eids), ctypes.addressof(p_ws),
                ctypes.addressof(p_R), ctypes.addressof(p_act), ctypes.addressof(p_le),
                ctypes.addressof(p_ls), ctypes.addressof(p_pe), ctypes.addressof(p_pw),
                ctypes.addressof(p_pk), ctypes.addressof(p_lout),
                ctypes.addressof(p_yn), ctypes.addressof(p_ym), ctypes.addressof(p_ys),
                ctypes.addressof(p_se), ctypes.addressof(p_sa), ctypes.addressof(p_sr),
                ctypes.addressof(c_M), ctypes.addressof(c_K), ctypes.addressof(c_E),
                ctypes.addressof(c_st), ctypes.addressof(c_cm),
            )
            rc = nvcuda.cuLaunchKernel(
                fns[b"pure_mirror_cube_bind_f32"],
                1, 1, 1, 256, 1, 1, 0, stream, args1, None,
            )
            if rc != 0:
                raise RuntimeError(f"cuLaunchKernel(pure_mirror_cube_bind_f32) failed: {rc}")
            self.total_bind_launches += 1

            # Launch Kernel #2: pure_mirror_eval_bridges_f32 (reads layer_peak_buf on GPU -> 0 CPU sync!)
            T = len(touched)
            p_te = ctypes.c_void_p(touched_t.data_ptr())
            p_br = ctypes.c_void_p(self.bridge_ratio[li].data_ptr())
            p_ro = ctypes.c_void_p(self.role_ids[li].data_ptr())
            c_T = ctypes.c_int(T)
            args2 = (ctypes.c_void_p * 12)(
                ctypes.addressof(p_te), ctypes.addressof(p_R), ctypes.addressof(p_le),
                ctypes.addressof(p_ls), ctypes.addressof(p_pe), ctypes.addressof(p_pw),
                ctypes.addressof(p_br), ctypes.addressof(p_ro),
                ctypes.addressof(p_pk), ctypes.addressof(c_st), ctypes.addressof(c_T), ctypes.addressof(c_E),
            )
            rc = nvcuda.cuLaunchKernel(
                fns[b"pure_mirror_eval_bridges_f32"],
                T, 1, 1, 32, 1, 1, 0, stream, args2, None,
            )
            if rc != 0:
                raise RuntimeError(f"cuLaunchKernel(pure_mirror_eval_bridges_f32) failed: {rc}")
            self.total_role_launches += 1
        else:
            touched = self._bind_and_eval_cpu(
                li,
                u_in[:M].to(dtype=torch.float32),
                u_post[:M].to(dtype=torch.float32) if u_post is not None else None,
                eids_rows,
                ws_rows,
                cur_step,
                is_commit,
            )

        # Mark dirty on host — NO synchronous GPU->CPU readback during decode!
        dset = self._dirty_layers.get(li)
        if dset is None:
            self._dirty_layers[li] = set(touched)
        else:
            dset.update(touched)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.mutual_steps += 1
        self.last_step_ms = dt_ms
        self.total_step_ms += dt_ms
        return len(touched)

    @torch.no_grad()
    def evaluate_continuation_batch(
        self,
        layer_idx: int,
        minimum_eids: Sequence[int],
        prev_loop_eids: Sequence[int],
        cand_eids: Sequence[int],
        cand_next_w: Sequence[float],
        cand_graph_loop: Sequence[int],
        *,
        u_info: torch.Tensor,
        prev_u_info: Optional[torch.Tensor],
        line_norm: float,
        last_obs_norm: float,
        last_res_norm: float,
        w_done: float,
        rem_w: float,
        tok_done: int,
        rest_min: int,
        rest_tol: float,
        unclosed_cone_protect: bool = False,
    ) -> List[Tuple[str, float, float, float, float]]:
        """Evaluate ALL queued cold candidates on GPU in ONE bracketless 4x4 Mirror launch."""
        C = len(cand_eids)
        if C == 0:
            return []
        li = int(layer_idx)
        names = ("loop_open", "no_mirror_link", "mirror_maintained", "mirror_expand", "direct_pass_unclosed")

        if self.use_cuda_kernels:
            dev = self.device
            u_c = u_info.to(device=dev, dtype=torch.float32).contiguous()
            pu_c = (
                prev_u_info.to(device=dev, dtype=torch.float32).contiguous()
                if (prev_u_info is not None and li > 0)
                else None
            )
            min_t = torch.tensor(list(minimum_eids[:8]), dtype=torch.int32, device=dev) if minimum_eids else torch.empty((0,), dtype=torch.int32, device=dev)
            prev_t = torch.tensor(list(prev_loop_eids[:8]), dtype=torch.int32, device=dev) if (prev_loop_eids and li > 0) else torch.empty((0,), dtype=torch.int32, device=dev)
            cand_t = torch.tensor(list(cand_eids), dtype=torch.int32, device=dev)
            cw_t = torch.tensor(list(cand_next_w), dtype=torch.float32, device=dev)
            cg_t = torch.tensor(list(cand_graph_loop), dtype=torch.int32, device=dev)
            dec_t = torch.empty((C,), dtype=torch.int32, device=dev)
            diag_t = torch.empty((C, 4), dtype=torch.float32, device=dev)

            nvcuda, fns = get_pure_mirror_kernels(dev)
            stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)

            p_rc = ctypes.c_void_p(self.roles[li].data_ptr())
            p_ac = ctypes.c_void_p(self.active_steps[li].data_ptr())
            p_rp = ctypes.c_void_p(self.roles[li - 1].data_ptr() if (li > 0 and pu_c is not None) else 0)
            p_ap = ctypes.c_void_p(self.active_steps[li - 1].data_ptr() if (li > 0 and pu_c is not None) else 0)
            p_ui = ctypes.c_void_p(u_c.data_ptr())
            p_pui = ctypes.c_void_p(pu_c.data_ptr() if pu_c is not None else 0)
            p_min = ctypes.c_void_p(min_t.data_ptr() if min_t.numel() > 0 else 0)
            p_prv = ctypes.c_void_p(prev_t.data_ptr() if prev_t.numel() > 0 else 0)
            p_cnd = ctypes.c_void_p(cand_t.data_ptr())
            p_cw = ctypes.c_void_p(cw_t.data_ptr())
            p_cg = ctypes.c_void_p(cg_t.data_ptr())
            p_dec = ctypes.c_void_p(dec_t.data_ptr())
            p_dg = ctypes.c_void_p(diag_t.data_ptr())

            c_C = ctypes.c_int(C)
            c_Km = ctypes.c_int(int(min_t.numel()))
            c_Kp = ctypes.c_int(int(prev_t.numel()))
            c_E = ctypes.c_int(self.num_experts)
            c_ln = ctypes.c_float(float(line_norm))
            c_lo = ctypes.c_float(float(last_obs_norm))
            c_lr = ctypes.c_float(float(last_res_norm))
            c_wd = ctypes.c_float(float(w_done))
            c_rw = ctypes.c_float(float(rem_w))
            c_td = ctypes.c_int(int(tok_done))
            c_rm = ctypes.c_int(int(rest_min))
            c_rt = ctypes.c_float(float(rest_tol))
            c_up = ctypes.c_int(1 if unclosed_cone_protect else 0)

            args = (ctypes.c_void_p * 26)(
                ctypes.addressof(p_rc), ctypes.addressof(p_ac),
                ctypes.addressof(p_rp), ctypes.addressof(p_ap),
                ctypes.addressof(p_ui), ctypes.addressof(p_pui),
                ctypes.addressof(p_min), ctypes.addressof(p_prv),
                ctypes.addressof(p_cnd), ctypes.addressof(p_cw), ctypes.addressof(p_cg),
                ctypes.addressof(p_dec), ctypes.addressof(p_dg),
                ctypes.addressof(c_C), ctypes.addressof(c_Km), ctypes.addressof(c_Kp), ctypes.addressof(c_E),
                ctypes.addressof(c_ln), ctypes.addressof(c_lo), ctypes.addressof(c_lr),
                ctypes.addressof(c_wd), ctypes.addressof(c_rw),
                ctypes.addressof(c_td), ctypes.addressof(c_rm), ctypes.addressof(c_rt),
                ctypes.addressof(c_up),
            )
            rc = nvcuda.cuLaunchKernel(
                fns[b"pure_mirror_continuation_gate_f32"],
                C, 1, 1, 32, 1, 1, 0, stream, args, None,
            )
            if rc != 0:
                raise RuntimeError(f"cuLaunchKernel(pure_mirror_continuation_gate_f32) failed: {rc}")
            self.total_gate_launches += 1

            decs = dec_t.cpu().tolist()
            diags = diag_t.cpu().tolist()
            out: List[Tuple[str, float, float, float, float]] = []
            for d_code, dg in zip(decs, diags):
                reason = names[d_code] if 0 <= d_code <= 4 else "mirror_expand"
                self.continuation_comparisons += 1
                if reason == "mirror_expand" and dg[1] > 1e-7:
                    self.continuation_new_links += 1
                elif reason == "mirror_maintained":
                    self.continuation_maintained += 1
                elif reason == "no_mirror_link":
                    self.continuation_no_link += 1
                elif reason == "direct_pass_unclosed":
                    self.continuation_direct_unclosed += 1
                out.append((reason, dg[0], dg[1], dg[2], dg[3]))
            return out

        # Exact bit-faithful CPU PyTorch reference of Kernel #4
        R_l = self.roles[li]
        act_l = self.active_steps[li]
        u_c = u_info.to(device=self.device, dtype=torch.float32)

        m_rows = []
        for em in minimum_eids[:8]:
            emi = int(em)
            if 0 <= emi < self.num_experts and int(act_l[emi].item()) > 0:
                p0, p1, sc, lg = R_l[emi, 0], R_l[emi, 1], R_l[emi, 2], R_l[emi, 5]
                _, _, s_cl, _, _, _, eq_m = _eval_4x4_mirror_cube_cpu(p0, p1, sc, u_c, u_c + lg)
                m_rows.append((p0, p1, s_cl, eq_m == EQ_SPLIT))

        min_cnt = 0
        min_e = 0.0
        min_tie_sum = 0.0
        for i in range(len(m_rows)):
            for j in range(i + 1, len(m_rows)):
                _, _, _, tie_ij, res_ij, _, _ = _eval_4x4_mirror_cube_cpu(
                    m_rows[i][0], m_rows[i][1], m_rows[i][2], m_rows[j][1], m_rows[j][0]
                )
                if tie_ij > 1e-7:
                    min_cnt += 1
                    min_tie_sum += tie_ij
                    min_e += tie_ij * (0.5 + 0.5 * res_ij)
        min_tie_bar = (0.25 * (min_tie_sum / min_cnt)) if min_cnt > 0 else 1e-7

        out = []
        for ce, nw, g_ok in zip(cand_eids, cand_next_w, cand_graph_loop):
            if tok_done < rest_min or w_done < rem_w:
                self.continuation_comparisons += 1
                out.append(("loop_open", 0.0, 0.0, 1.0, rest_tol))
                continue
            delta_cnt = 0
            delta_e = 0.0
            all_maintained = True
            saw_any_tie = False
            cei = int(ce)
            if 0 <= cei < self.num_experts and int(act_l[cei].item()) > 0:
                c0, c1, cs, cl = R_l[cei, 0], R_l[cei, 1], R_l[cei, 2], R_l[cei, 5]
                _, _, _, cu_t1, cu_r1, _, cu_eq1 = _eval_4x4_mirror_cube_cpu(c0, c1, cs, u_c, u_c + cl)
                _, _, _, cu_t2, cu_r2, _, cu_eq2 = _eval_4x4_mirror_cube_cpu(cs, cs, cs, u_c, u_c + cl)
                if cu_t2 > cu_t1:
                    cu_tie, cu_res, cu_eq = cu_t2, cu_r2, cu_eq2
                else:
                    cu_tie, cu_res, cu_eq = cu_t1, cu_r1, cu_eq1

                if cu_tie > 1e-7:
                    for p0_m, p1_m, sm_m, is_sp in m_rows:
                        _, _, _, tf, rf, _, eqf = _eval_4x4_mirror_cube_cpu(p0_m, p1_m, sm_m, c1, c0)
                        _, _, _, tr, rr, _, eqr = _eval_4x4_mirror_cube_cpu(c0, c1, cs, p1_m, p0_m)
                        pt = 0.5 * (tf + tr)
                        pr = max(rf, rr)
                        ps = EQ_TENSION if (eqf == EQ_TENSION or eqr == EQ_TENSION) else eqf
                        _, _, _, tsc, rsc, _, eqsc = _eval_4x4_mirror_cube_cpu(p0_m, p1_m, sm_m, cs, cs + cl)
                        if tsc > pt:
                            pt = tsc
                            pr = rsc
                            ps = eqsc
                        if is_sp:
                            pt = max(pt, cu_tie)
                            pr = max(pr, cu_res)
                            ps = EQ_TENSION
                        if pt >= min_tie_bar and pt > 1e-7:
                            saw_any_tie = True
                            if ps != EQ_EQUAL:
                                all_maintained = False
                                delta_cnt += 1
                                delta_e += cu_tie * pt * (0.5 + 0.5 * pr)

                    if li > 0 and prev_u_info is not None and prev_loop_eids and g_ok and cu_tie >= min_tie_bar:
                        pu_c = prev_u_info.to(device=self.device, dtype=torch.float32)
                        R_prev = self.roles[li - 1]
                        for ep in prev_loop_eids[:8]:
                            epi = int(ep)
                            if 0 <= epi < self.num_experts and int(self.active_steps[li - 1, epi].item()) > 0:
                                _, _, _, pt_p, _, _, _ = _eval_4x4_mirror_cube_cpu(
                                    R_prev[epi, 0], R_prev[epi, 1], R_prev[epi, 2], pu_c, pu_c + R_prev[epi, 5]
                                )
                                if pt_p >= min_tie_bar and pt_p > 1e-7:
                                    saw_any_tie = True
                                    if cu_eq != EQ_EQUAL or cu_res >= 0.5:
                                        all_maintained = False
                                        delta_cnt += 1
                                        delta_e += cu_tie * pt_p

            self.continuation_comparisons += 1
            if saw_any_tie and all_maintained and delta_cnt == 0:
                self.continuation_maintained += 1
                out.append(("mirror_maintained", min_e, 0.0, 0.0, rest_tol))
                continue
            if (min_cnt > 0 and delta_cnt == 0) or (min_cnt == 0 and delta_cnt == 0 and not g_ok):
                if unclosed_cone_protect:
                    self.continuation_direct_unclosed += 1
                    out.append(("direct_pass_unclosed", min_e, 0.0, 1.0, rest_tol))
                    continue
                self.continuation_no_link += 1
                out.append(("no_mirror_link", min_e, 0.0, 0.0, 0.0))
                continue

            obs_sig = max(last_res_norm, 0.5 * last_obs_norm)
            mean_w = w_done / max(1.0, float(tok_done))
            scale_next = (nw / max(mean_w, 1e-6)) * (1.0 + 0.5 * max(0.0, rem_w))
            est_delta = obs_sig * scale_next
            defect = est_delta / (line_norm + est_delta + 1e-8)
            eff_tol = rest_tol
            if delta_cnt > 0:
                eff_tol = (0.5 * rest_tol) if (min_cnt == 0 or min_e <= 1e-8) else (rest_tol * (min_e / (min_e + delta_e + 1e-8)))
            if tok_done >= rest_min and defect < eff_tol:
                self.continuation_maintained += 1
                out.append(("mirror_maintained", min_e, delta_e, defect, eff_tol))
            else:
                if delta_cnt > 0:
                    self.continuation_new_links += 1
                out.append(("mirror_expand", min_e, delta_e, defect, eff_tol))
        return out

    @torch.no_grad()
    def evaluate_self_collapse_batch(
        self,
        layer_idx: int,
        res_eids: Sequence[int],
        active_eids: Sequence[int],
        u_in: torch.Tensor,
        u_post: torch.Tensor,
    ) -> List[int]:
        """Run Kernel #5 (`pure_mirror_self_collapse_f32`) over collapsible resident agents in VRAM.

        Returns list of ints aligned with `res_eids`:
          0 = hold_tension   (living EQ_TENSION or active mirror_bridge -> keep crisp!)
          1 = collapse_split (SPLIT/UNKNOWN against live flow -> self-collapse 1 Q1 bit-plane!)
          2 = collapse_equal (EQ_EQUAL / C03 duplicate -> self-collapse 1 redundant Q1 bit-plane!)
        """
        N_res = len(res_eids)
        li = int(layer_idx)
        if N_res == 0 or not (0 <= li < self.num_layers):
            return []

        if self.use_cuda_kernels:
            dev = self.device
            ui_c = (u_in[0] if u_in.ndim > 1 else u_in).to(device=dev, dtype=torch.float32).contiguous()
            up_c = (u_post[0] if u_post.ndim > 1 else u_post).to(device=dev, dtype=torch.float32).contiguous()
            res_t = torch.tensor(list(res_eids), dtype=torch.int32, device=dev)
            act_list = list(active_eids[:8])
            act_t = torch.tensor(act_list, dtype=torch.int32, device=dev) if act_list else torch.empty((0,), dtype=torch.int32, device=dev)
            out_t = torch.empty((N_res,), dtype=torch.int32, device=dev)

            nvcuda, fns = get_pure_mirror_kernels(dev)
            stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)

            p_res = ctypes.c_void_p(res_t.data_ptr())
            p_act = ctypes.c_void_p(act_t.data_ptr() if act_t.numel() > 0 else 0)
            p_R = ctypes.c_void_p(self.roles[li].data_ptr())
            p_le = ctypes.c_void_p(self.link_energy[li].data_ptr())
            p_ro = ctypes.c_void_p(self.role_ids[li].data_ptr())
            p_ui = ctypes.c_void_p(ui_c.data_ptr())
            p_up = ctypes.c_void_p(up_c.data_ptr())
            p_out = ctypes.c_void_p(out_t.data_ptr())
            c_N = ctypes.c_int(N_res)
            c_Ka = ctypes.c_int(int(act_t.numel()))
            c_E = ctypes.c_int(self.num_experts)

            args = (ctypes.c_void_p * 11)(
                ctypes.addressof(p_res), ctypes.addressof(p_act),
                ctypes.addressof(p_R), ctypes.addressof(p_le), ctypes.addressof(p_ro),
                ctypes.addressof(p_ui), ctypes.addressof(p_up), ctypes.addressof(p_out),
                ctypes.addressof(c_N), ctypes.addressof(c_Ka), ctypes.addressof(c_E),
            )
            rc = nvcuda.cuLaunchKernel(
                fns[b"pure_mirror_self_collapse_f32"],
                N_res, 1, 1, 32, 1, 1, 0, stream, args, None,
            )
            if rc != 0:
                raise RuntimeError(f"cuLaunchKernel(pure_mirror_self_collapse_f32) failed: {rc}")
            self.total_collapse_launches += 1
            codes = out_t.cpu().tolist()
        else:
            # Exact bit-faithful CPU PyTorch reference of Kernel #5
            ui_c = (u_in[0] if u_in.ndim > 1 else u_in).to(device=self.device, dtype=torch.float32)
            up_c = (u_post[0] if u_post.ndim > 1 else u_post).to(device=self.device, dtype=torch.float32)
            R_l = self.roles[li]
            le_l = self.link_energy[li]
            ro_l = self.role_ids[li]
            act_list = [int(a) for a in active_eids[:8] if 0 <= int(a) < self.num_experts]
            codes = []
            for eid_raw in res_eids:
                eid = int(eid_raw)
                if not (0 <= eid < self.num_experts):
                    codes.append(0)
                    continue
                p0, p1, sc = R_l[eid, 0], R_l[eid, 1], R_l[eid, 2]
                _, _, _, t1, _, _, st1 = _eval_4x4_mirror_cube_cpu(p0, p1, sc, ui_c, up_c)
                _, _, _, t2, _, _, st2 = _eval_4x4_mirror_cube_cpu(sc, sc, sc, ui_c, up_c)
                u_tie, u_stat = (t2, st2) if t2 > t1 else (t1, st1)
                hold = bool(u_tie > 1e-7 and (u_stat == EQ_TENSION or int(ro_l[eid].item()) == 2))
                if (not hold) and u_tie > 1e-7:
                    for ae in act_list:
                        if ae != eid:
                            _, _, _, ta, _, _, sta = _eval_4x4_mirror_cube_cpu(
                                p0, p1, sc, R_l[ae, 1], R_l[ae, 0]
                            )
                            if ta > 1e-7 and sta == EQ_TENSION:
                                hold = True
                                break
                if hold:
                    codes.append(0)
                elif u_tie <= 1e-7 or u_stat == EQ_UNKNOWN or u_stat == EQ_SPLIT:
                    codes.append(1)
                    le_l[eid] = float(le_l[eid].item()) * 0.5
                else:
                    codes.append(2)

        dirty_set = self._dirty_layers.setdefault(li, set())
        for eid_raw, code in zip(res_eids, codes):
            if code == 0:
                self.self_collapse_held += 1
            elif code == 1:
                self.self_collapse_split += 1
                dirty_set.add(int(eid_raw))
            else:
                self.self_collapse_equal += 1
        return codes

    @torch.no_grad()
    def predict_next_layer_mirror(
        self,
        layer_idx: int,
        active_eids: Sequence[int],
        u_post: Optional[torch.Tensor],
        corner90_edges: Dict[Tuple[int, int], Dict[int, float]],
        *,
        top_n: int = 4,
    ) -> List[int]:
        """Autonomous next-layer `l+1` Mirror spawning prediction (ZERO UNI_PHI!).

        Ranks candidate agents in `layer_idx + 1` linked via 90-deg transitions strictly by
        their living 4x4 Mirror contact (`tie * (0.5 + 0.5 * rel_residual)`) with outgoing `u_post`.
        """
        li_next = int(layer_idx) + 1
        if li_next >= self.num_layers or u_post is None:
            return []
        cand_w: Dict[int, float] = {}
        for ep in active_eids:
            for en, w in corner90_edges.get((int(layer_idx), int(ep)), {}).items():
                if w > 0.0:
                    cand_w[en] = cand_w.get(en, 0.0) + w
        if not cand_w:
            return []
        # Keep top 8 graph-linked candidates to evaluate with 4x4 cube (microseconds)
        top_cands = [e for e, _ in sorted(cand_w.items(), key=lambda kv: -kv[1])[:8]]
        u_vec = (u_post[0] if u_post.ndim > 1 else u_post).detach().cpu().to(torch.float32)
        R_next = self.roles[li_next]
        act_next = self.active_steps[li_next]
        scored: List[Tuple[float, int]] = []
        for en in top_cands:
            if not (0 <= en < self.num_experts):
                continue
            if int(act_next[en].item()) <= 0:
                scored.append((cand_w[en] * 0.5, en))
                continue
            row = R_next[en].detach().cpu()
            p0, p1, sc, lg = row[0], row[1], row[2], row[5]
            _, _, _, t1, r1, _, st1 = _eval_4x4_mirror_cube_cpu(p0, p1, sc, u_vec, u_vec + lg)
            _, _, _, t2, r2, _, st2 = _eval_4x4_mirror_cube_cpu(sc, sc, sc, u_vec, u_vec + lg)
            tie, res, st = (t2, r2, st2) if t2 > t1 else (t1, r1, st1)
            if tie > 1e-7 and st != EQ_EQUAL:
                scored.append((tie * (0.5 + 0.5 * res) * cand_w[en], en))
        scored.sort(key=lambda kv: -kv[0])
        return [en for _, en in scored[:top_n]]

    def agent_mirror_feedback(self, key: Tuple[int, int]) -> Tuple[str, float, float, bool]:
        """Return `(role, link_rel, bridge_ratio, layer_has_links)` (lazy-syncs only if dirty)."""
        li, eid = int(key[0]), int(key[1])
        if not (0 <= li < self.num_layers) or not (0 <= eid < self.num_experts):
            return "cold", 0.0, 0.0, False
        if li in self._dirty_layers:
            self._sync_layer_to_host(li)
        layer_has = self._layer_link_events_host[li] > 0
        role = self.role_of.get((li, eid))
        if role is None:
            return "cold", 0.0, 0.0, layer_has
        return role, self._host_link_rel.get((li, eid), 0.0), self._host_bridge_ratio.get((li, eid), 0.0), layer_has

    def export_mirror_snapshot(
        self,
        face_edges: Dict[Tuple[int, int], Dict[int, float]],
        corner90_edges: Dict[Tuple[int, int], Dict[int, float]],
        *,
        max_edges: int = 4000,
    ) -> Dict[str, Any]:
        """Export a JSON snapshot of active Irreducible 4x4 Mirror agents & their links."""
        self._sync_all_dirty()
        R_cpu = self.roles.detach().cpu()
        act_cpu = self.active_steps.detach().cpu()
        nodes_out = []
        for key in sorted(self.role_of.keys()):
            li, eid = key
            role, l_rel, b_rat, _ = self.agent_mirror_feedback(key)
            sv = R_cpu[li, eid, SLOT_CENTER]
            lv = R_cpu[li, eid, SLOT_LOGIC]
            nodes_out.append({
                "id": f"L{li}_E{eid}",
                "layer": int(li),
                "expert": int(eid),
                "role": role,
                "mirror_steps": int(act_cpu[li, eid].item()),
                "strategy_norm": round(float(sv.norm().item()), 4),
                "logic_norm": round(float(lv.norm().item()), 4),
                "link_rel": round(l_rel, 4),
                "bridge_ratio": round(b_rat, 4),
                "pos3": [round(float(v), 4) for v in sv[:3].tolist()],
            })
        raw_edges = []
        node_set = set(self.role_of.keys())
        for ka, nbrs in face_edges.items():
            if ka not in node_set:
                continue
            li, ea = ka
            for eb, w in nbrs.items():
                if eb > ea and (li, eb) in node_set and w > 1e-9:
                    _, _, _, t_ab, _, _, _ = _eval_4x4_mirror_cube_cpu(
                        R_cpu[li, ea, 0], R_cpu[li, ea, 1], R_cpu[li, ea, 2],
                        R_cpu[li, eb, 1], R_cpu[li, eb, 0],
                    )
                    if t_ab > 1e-6:
                        raw_edges.append((t_ab * w, ka, (li, eb), "mirror_4x4_link"))
        for kp, nbrs in corner90_edges.items():
            if kp not in node_set:
                continue
            li_next = kp[0] + 1
            for ec, w in nbrs.items():
                kc = (li_next, ec)
                if kc in node_set and w > 1e-9:
                    raw_edges.append((w, kp, kc, "corner90"))
        raw_edges.sort(key=lambda it: -it[0])
        links_out = [
            {
                "source": f"L{u[0]}_E{u[1]}",
                "target": f"L{v[0]}_E{v[1]}",
                "weight": round(float(w), 5),
                "kind": kind,
            }
            for w, u, v, kind in raw_edges[: max(0, int(max_edges))]
        ]
        return {"summary": self.stats(), "nodes": nodes_out, "links": links_out}

    @torch.no_grad()
    def layer_node_state(self, layer_idx: int) -> Tuple[int, str, float, float, float]:
        """Return `(status_code, status_name, tau_core, rel_residual, future_press)` of born layer node Y_L."""
        li = int(layer_idx)
        if not (0 <= li < self.num_layers):
            return EQ_UNKNOWN, "UNKNOWN", 0.0, 1.0, 0.0
        m = self.layer_node_meta[li].detach().cpu().tolist()
        code = int(round(m[3]))
        names = ("UNKNOWN", "EQUAL", "TENSION", "SPLIT")
        return code, (names[code] if 0 <= code <= 3 else "UNKNOWN"), float(m[0]), float(m[1]), float(m[2])

    @torch.no_grad()
    def reconstruct_layer_node_from_ray(self, layer_idx: int, slot_k: int) -> Tuple[int, torch.Tensor]:
        """Forward reversible scar check: recover `S_Y = alpha_k * S_k + r_k` from intact participant ray `k`."""
        li, sk = int(layer_idx), int(slot_k)
        eid = int(self.layer_scar_eids[li, sk].item())
        if eid < 0:
            return -1, torch.zeros((self.proj_dim,), dtype=torch.float32, device=self.device)
        ak = self.layer_scar_alpha[li, sk]
        rk = self.layer_scar_res[li, sk]
        s_k = self.roles[li, eid, SLOT_CENTER]
        return eid, ak * s_k + rk

    @torch.no_grad()
    def reconstruct_ray_from_layer_node(self, layer_idx: int, slot_k: int) -> Tuple[int, float, torch.Tensor]:
        """Inverse reversible scar check: recover participant `S_k = (S_Y - r_k) / alpha_k` from born `Y_L`."""
        li, sk = int(layer_idx), int(slot_k)
        eid = int(self.layer_scar_eids[li, sk].item())
        ak = float(self.layer_scar_alpha[li, sk].item())
        if eid < 0 or abs(ak) <= 1e-6:
            return eid, ak, torch.zeros((self.proj_dim,), dtype=torch.float32, device=self.device)
        s_y = self.layer_nodes[li, SLOT_CENTER]
        rk = self.layer_scar_res[li, sk]
        return eid, ak, (s_y - rk) / ak

    def stats(self) -> Dict[str, Any]:
        self._sync_all_dirty()
        role_counts = {"mirror_bridge": 0, "mirror_core": 0, "mirror_periphery": 0}
        for r in self.role_of.values():
            if r in role_counts:
                role_counts[r] += 1
        avg_ms = self.total_step_ms / max(1, self.mutual_steps)
        max_steps = int(self.active_steps.max().item()) if self.active_steps.numel() > 0 else 0
        total_links = int(self.layer_links_gpu.sum().item())
        born_layers = int((self.layer_node_steps > 0).sum().item()) if self.layer_node_steps.numel() > 0 else 0
        return {
            "mode": "pure_mirror_4x4_d15_cuda",
            "bracketless_stem": True,
            "participants_intact": True,
            "layer_nodes_born": born_layers,
            "cuda_kernels_active": self.use_cuda_kernels,
            "nodes": len(self.role_of),
            "total_mirror_links": total_links,
            "max_mirror_steps": max_steps,
            "topic_shift_continuations": self.topic_shift_continuations,
            "cuda_launches": {
                "cube_bind_4x4_d15": self.total_bind_launches,
                "eval_bridges_4x4": self.total_role_launches,
                "line_step_residual": self.total_line_launches,
                "continuation_gate_4x4": self.total_gate_launches,
                "self_collapse_4x4": self.total_collapse_launches,
                "diff_q1_gemv_mirror": self.total_diff_q1_launches,
            },
            "self_collapse": {
                "split": self.self_collapse_split,
                "equal_c03": self.self_collapse_equal,
                "held_tension": self.self_collapse_held,
            },
            "roles": role_counts,
            "mutual_steps": self.mutual_steps,
            "continuation_comparisons": self.continuation_comparisons,
            "continuation_new_links": self.continuation_new_links,
            "continuation_maintained": self.continuation_maintained,
            "continuation_no_link": self.continuation_no_link,
            "continuation_direct_unclosed": self.continuation_direct_unclosed,
            "last_refresh_ms": round(self.last_step_ms, 4),
            "avg_refresh_ms": round(avg_ms, 4),
            "total_refresh_ms": round(self.total_step_ms, 3),
            # Compatible fields for unified A/B/C summary reporting
            "communities": sum(1 for c in self._layer_link_events_host if c > 0),
            "modularity_q": round(
                self.continuation_new_links / max(1, self.continuation_comparisons), 4
            ),
        }
