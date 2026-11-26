"""Quintessentia runtime: W4A16 + NVRTC kernels + native MTP speculative decoding + budgeted VRAM.

`QuintessentiaUnifiedEngine` combines:
- NVRTC CUDA kernels, including single-weight-read multi-row kernels (`M in {1,2,3,4}`)
- Native MTP head (`QuintessentiaNativeMTPHead`, ~130 MiB Q4)
- `QuintessentiaDraftVocab`, `QuintessentiaSpecGate`, `quintessentia_spec_lookup`, `HybridSpecSnapshotManager`
- VRAM modes `normal` (almost all free VRAM, with a safety reserve) and `turbo` (all of it):
  all layers resident when they fit, otherwise resident layers + an equal-slot streaming ring
- Chunked prefill that projects only the last position onto the vocabulary
- Cooperative cancellation and guaranteed VRAM cleanup (`try/finally`) for every generation
- `QuintessentiaKVPrefixSession`: universal multi-turn KV / GatedDeltaNet prefix reuse with host-RAM parking
- Quintessentia containers, standard Hugging Face shards, and sliced GGUF checkpoints (see `model_source`)
- Optional `ZeroProbSubspaceVocabHead` (`--fast-vocab`, off by default)
"""
from __future__ import annotations

import copy
import ctypes
from dataclasses import asdict, dataclass
import gc
import math
import os
from pathlib import Path
import sys
import time
import traceback
import types
from typing import Any, Callable, Iterator

import torch


from .runtime_config import (compute_runtime_limits, model_context_limit, PROJECT_ROOT, VRAM_MODES, ContextBudgetExceeded, GenerationCancelled,
                             find_quintessentia_root, checkpoint_manifest_path)


DEFAULT_DRAFT_SEED_CORPUS = (
    "Quintessentia combines 4-bit group-64 quantization, NVRTC CUDA kernel fusion, "
    "Multi-Token Prediction (MTP), GatedDeltaNet recurrent state snapshots, and "
    "PCIe triple-ring asynchronous DMA prefetching.\n"
    "В физике шин памяти видеокарты главная задержка при генерации по одному токену "
    "возникает из-за повторного чтения весов слоёв из GDDR6. Спекулятивная проверка "
    "нескольких кандидатов за один проход делит чтение весов на 2-3 токена без потери точности.\n"
    "def verify_ring_slot_lease(slot_id: int, layer_idx: int) -> bool:\n"
    "    return slot_id >= 0 and layer_idx >= 0\n"
)


class QuintessentiaDraftVocab:
    """Port of `upstream_colibri/c/draft_vocab.h` scored via `indexed_vocab_gemv_q4_f32`."""

    NEAR_NUM = 3
    NEAR_DEN = 4
    HALVE_EVERY = 1024

    def __init__(self, vocab_size: int, top_k: int = 16000, device: torch.device = torch.device("cuda")):
        self.vocab_size = int(vocab_size)
        self.top_k = min(int(top_k), self.vocab_size)
        self.device = device
        self.counts = torch.zeros(self.vocab_size, dtype=torch.float32)
        self.in_conversation = torch.zeros(self.vocab_size, dtype=torch.bool)
        self.in_list = torch.zeros(self.vocab_size, dtype=torch.bool)
        self.counted = 0
        self.gpu_list: torch.Tensor | None = None
        self.stats = {"picks": 0, "picks_inside": 0, "appended": 0, "rebuilds": 0, "halvings": 0}

    def seed_from_corpus_tokens(self, token_ids: list[int]) -> None:
        for tid in token_ids:
            if 0 <= tid < self.vocab_size:
                self.counts[tid] += 1.0

    def _count_one(self, tok: int) -> bool:
        self.counts[tok] += 1.0
        self.counted += 1
        if self.counted >= self.HALVE_EVERY:
            self.counts.mul_(0.5)
            self.counted = 0
            self.stats["halvings"] += 1
            return True
        return False

    def rebuild(self) -> None:
        pos_mask = self.counts > 0
        n_pos = int(pos_mask.sum().item())
        if n_pos >= self.top_k:
            kth_val = float(torch.topk(self.counts, k=self.top_k).values[-1].item())
            cutoff = (self.NEAR_NUM / self.NEAR_DEN) * kth_val
            selected = (self.counts >= cutoff) | self.in_conversation
        else:
            selected = pos_mask | self.in_conversation
            rem = self.top_k - int(selected.sum().item())
            if rem > 0:
                not_sel = (~selected).nonzero(as_tuple=False).squeeze(-1)[:rem]
                selected[not_sel] = True

        self.in_list = selected
        idx_cpu = selected.nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
        self.gpu_list = idx_cpu.to(self.device, non_blocking=True).unsqueeze(0).contiguous()
        self.stats["rebuilds"] += 1

    def begin_request(self, prompt_ids: list[int], cached_prompt_tokens: int = 0) -> None:
        self.in_conversation.zero_()
        for tid in prompt_ids:
            if 0 <= tid < self.vocab_size:
                self.in_conversation[tid] = True
        for tid in prompt_ids[cached_prompt_tokens:]:
            if 0 <= tid < self.vocab_size:
                self._count_one(tid)
        self.rebuild()

    def observe_picked_token(self, tok: int) -> None:
        if not (0 <= tok < self.vocab_size):
            return
        self.stats["picks"] += 1
        was_inside = bool(self.in_list[tok].item())
        if was_inside:
            self.stats["picks_inside"] += 1
        self.in_conversation[tok] = True
        halved = self._count_one(tok)
        if halved:
            self.rebuild()
        elif not was_inside:
            self.in_list[tok] = True
            new_id = torch.tensor([[tok]], dtype=torch.int32, device=self.device)
            self.gpu_list = torch.cat([self.gpu_list, new_id], dim=1).contiguous()
            self.stats["appended"] += 1


def quintessentia_spec_lookup(history: list[int], max_drafts: int = 2, max_n: int = 5, min_n: int = 2) -> list[int]:
    """Port of `spec_lookup` from `upstream_colibri/c/spec_draft.h`."""
    H = len(history)
    if H < min_n + 1 or max_drafts <= 0:
        return []
    search_start = max(0, H - 1024)
    for n in range(min(max_n, H - 1), min_n - 1, -1):
        pat = history[H - n :]
        last_tok = pat[-1]
        for pos in range(H - n - 1, search_start + n - 2, -1):
            if history[pos] == last_tok and history[pos - n + 1 : pos + 1] == pat:
                nxt_start = pos + 1
                nxt_end = min(H - 1, nxt_start + max_drafts)
                if nxt_end > nxt_start:
                    return history[nxt_start:nxt_end]
    return []


class QuintessentiaSpecGate:
    """Port of confidence-scheduled `SpecGate` from `upstream_colibri/c/spec_draft.h`."""

    SRC_MTP = 0
    SRC_LOOKUP = 1
    SRC_Q1 = 2
    NUM_SOURCES = 3
    PBINS = 10

    def __init__(self, max_drafts: int = 2, force_off: bool = False):
        self.max_drafts = max_drafts
        self.off = force_off
        self.acc = [
            [0.80, 0.58, 0.45, 0.40],
            [0.65, 0.52, 0.45, 0.40],
            [0.72, 0.50, 0.38, 0.30],
        ]
        self.acc_w = [[2.0] * 4 for _ in range(self.NUM_SOURCES)]
        self.t = [0.0] * 6
        self.t_w = [0.0] * 6
        self.rho = 0.12
        self.rho_w = 1.0
        self.d = [0.0] * self.NUM_SOURCES
        self.d_w = [0.0] * self.NUM_SOURCES
        self.pb = [[(b + 0.5) / self.PBINS for b in range(self.PBINS)] for _ in range(self.NUM_SOURCES)]
        self.pb_w = [[2.0] * self.PBINS for _ in range(self.NUM_SOURCES)]
        self.prop = [[0, 0, 0, 0] for _ in range(self.NUM_SOURCES)]
        self.hit = [[0, 0, 0, 0] for _ in range(self.NUM_SOURCES)]
        self.verifies = [0] * self.NUM_SOURCES
        self.declined = [0] * self.NUM_SOURCES

    @staticmethod
    def _ema(val: float, weight: float, sample: float, memory: float) -> tuple[float, float]:
        if weight < memory:
            weight += 1.0
        val += (sample - val) / weight
        return val, weight

    def record_forward(self, rows: int, seconds: float) -> None:
        if 1 <= rows < len(self.t) and seconds > 0.0:
            self.t[rows], self.t_w[rows] = self._ema(self.t[rows], self.t_w[rows], seconds, 8.0)
            if rows > 1 and self.t_w[1] > 0.0 and self.t[1] > 0.0:
                r = max(0.0, min(4.0, (seconds / self.t[1] - 1.0) / (rows - 1)))
                self.rho, self.rho_w = self._ema(self.rho, self.rho_w, r, 8.0)

    def record_draft_cost(self, src: int, seconds: float) -> None:
        if 0 <= src < self.NUM_SOURCES:
            self.d[src], self.d_w[src] = self._ema(self.d[src], self.d_w[src], seconds, 8.0)

    def rows_cost(self, rows: int) -> float:
        if rows <= 1:
            return 1.0
        if rows < len(self.t) and self.t_w[rows] > 0.0 and self.t_w[1] > 0.0 and self.t[1] > 0.0:
            return self.t[rows] / self.t[1]
        return 1.0 + (rows - 1) * self.rho

    def _pbin(self, p: float) -> int:
        b = int(p * self.PBINS) if p >= 0.0 else 0
        return max(0, min(self.PBINS - 1, b))

    def calibrated_p(self, src: int, p_est: float, pos_j: int) -> float:
        bin_rate = self.pb[src][self._pbin(p_est)]
        return 0.5 * bin_rate + 0.5 * max(p_est, self.acc[src][pos_j] * 0.85)

    def should_draft_second_mtp(self, p1: float) -> bool:
        if self.max_drafts < 2:
            return False
        if self.off:
            return p1 >= 0.55
        a1 = self.calibrated_p(self.SRC_MTP, p1, 0)
        if a1 < 0.55:
            return False
        a2 = self.acc[self.SRC_MTP][1]
        v1 = (1.0 + a1) / self.rows_cost(2)
        extra_d = (self.d[self.SRC_MTP] / self.t[1]) if (self.d_w[self.SRC_MTP] > 0 and self.t[1] > 0) else 0.03
        v2 = (1.0 + a1 + a1 * a2) / (self.rows_cost(3) + extra_d)
        return v2 > v1 and v2 > 1.02

    def expected_gain(self, src: int, probs: list[float]) -> float:
        k_avail = min(len(probs), self.max_drafts)
        if k_avail <= 0:
            return 1.0
        if self.off:
            return 1.5
        best_val = 1.0
        e = 1.0
        prod = 1.0
        measured_d = (
            self.d[src] / self.t[1]
            if (0 <= src < self.NUM_SOURCES and self.d_w[src] > 0.0 and self.t_w[1] > 0.0 and self.t[1] > 0.0)
            else None
        )
        for j in range(k_avail):
            aj = self.calibrated_p(src, probs[j], j)
            prod *= aj
            e += prod
            if measured_d is not None:
                d_rel = measured_d
            elif src == self.SRC_Q1:
                d_rel = 0.25 * (j + 1)
            elif src == self.SRC_MTP:
                d_rel = 0.03 * (j + 1)
            else:
                d_rel = 0.0
            val = e / (self.rows_cost(j + 2) + d_rel)
            if val > best_val:
                best_val = val
        return best_val

    def pick_depth(self, src: int, probs: list[float]) -> int:
        k_avail = min(len(probs), self.max_drafts)
        if k_avail == 0:
            return 0
        if self.off:
            return k_avail
        best_k = 0
        best_val = 1.0
        e = 1.0
        prod = 1.0
        for j in range(k_avail):
            aj = self.calibrated_p(src, probs[j], j)
            prod *= aj
            e += prod
            val = e / self.rows_cost(j + 2)
            if val > best_val:
                best_val = val
                best_k = j + 1
        if best_k == 0:
            self.declined[src] += 1
        return best_k

    def record_result(self, src: int, judged: int, accepted: int, probs: list[float] | None = None) -> None:
        if judged < 1 or not (0 <= src < self.NUM_SOURCES):
            return
        self.verifies[src] += 1
        for j in range(min(judged, 4)):
            self.prop[src][j] += 1
            ok = 1.0 if j < accepted else 0.0
            if j < accepted:
                self.hit[src][j] += 1
            self.acc[src][j], self.acc_w[src][j] = self._ema(self.acc[src][j], self.acc_w[src][j], ok, 16.0)
            if probs is not None and j < len(probs):
                b = self._pbin(probs[j])
                self.pb[src][b], self.pb_w[src][b] = self._ema(self.pb[src][b], self.pb_w[src][b], ok, 64.0)

    def summary(self) -> dict[str, Any]:
        mtp_prop = sum(self.prop[0])
        mtp_hit = sum(self.hit[0])
        lkp_prop = sum(self.prop[1])
        lkp_hit = sum(self.hit[1])
        q1_prop = sum(self.prop[2])
        q1_hit = sum(self.hit[2])
        tot_prop = mtp_prop + lkp_prop + q1_prop
        tot_hit = mtp_hit + lkp_hit + q1_hit
        return {
            "mtp_verifies": self.verifies[0],
            "mtp_proposed": mtp_prop,
            "mtp_accepted": mtp_hit,
            "mtp_accept_rate_pct": round(100.0 * mtp_hit / max(1, mtp_prop), 2),
            "lookup_verifies": self.verifies[1],
            "lookup_proposed": lkp_prop,
            "lookup_accepted": lkp_hit,
            "lookup_accept_rate_pct": round(100.0 * lkp_hit / max(1, lkp_prop), 2),
            "q1_verifies": self.verifies[2],
            "q1_proposed": q1_prop,
            "q1_accepted": q1_hit,
            "q1_accept_rate_pct": round(100.0 * q1_hit / max(1, q1_prop), 2),
            "overall_accept_rate_pct": round(100.0 * tot_hit / max(1, tot_prop), 2),
            "declined_low_confidence": sum(self.declined),
            "measured_rho_extra_row_cost": round(self.rho, 4),
        }


class HybridSpecSnapshotManager:
    """GPU-resident snapshot slots for Qwen3.5's 24 GatedDeltaNet + 8 Full-Attention layers."""

    def __init__(self, config, device: torch.device, max_drafts: int = 3):
        self.device = device
        self.max_slots = max_drafts
        self.linear_layers = [i for i, lt in enumerate(config.layer_types) if lt == "linear_attention"]
        self.full_layers = [i for i, lt in enumerate(config.layer_types) if lt == "full_attention"]
        self.layer_to_lin_slot = {l_idx: pos for pos, l_idx in enumerate(self.linear_layers)}
        n_lin = len(self.linear_layers)
        conv_dim = (
            config.linear_key_head_dim * config.linear_num_key_heads * 2
            + config.linear_value_head_dim * config.linear_num_value_heads
        )
        conv_k = config.linear_conv_kernel_dim
        self.snap_conv = torch.empty((n_lin, self.max_slots, 1, conv_dim, conv_k), dtype=torch.bfloat16, device=device)
        self.snap_rec = torch.empty(
            (n_lin, self.max_slots, 1, config.linear_num_value_heads, config.linear_key_head_dim, config.linear_value_head_dim),
            dtype=torch.float32,
            device=device,
        )
        self.active_capture = False
        self.rollbacks = 0
        self._captured_row0_count = 0

    def begin_capture(self) -> None:
        self._captured_row0_count = 0
        self.active_capture = True

    def end_capture(self) -> None:
        self.active_capture = False

    def save_row(self, layer_idx: int, row_r: int, conv_state: torch.Tensor, rec_state: torch.Tensor) -> None:
        if not self.active_capture or row_r >= self.max_slots:
            return
        lin_i = self.layer_to_lin_slot[layer_idx]
        c_len = conv_state.shape[-1]
        self.snap_conv[lin_i, row_r, :, :, :c_len].copy_(conv_state)
        self.snap_rec[lin_i, row_r].copy_(rec_state)
        if row_r == 0:
            self._captured_row0_count += 1

    def rollback_to_accepted(self, past_key_values: Any, accepted_drafts: int, total_drafts: int) -> bool:
        rejected = total_drafts - accepted_drafts
        if rejected <= 0:
            return True
        if accepted_drafts < 0 or accepted_drafts >= self.max_slots:
            return False
        if self._captured_row0_count < len(self.linear_layers):
            return False
        for l_idx in self.linear_layers:
            lin_i = self.layer_to_lin_slot[l_idx]
            cs = past_key_values.get_conv_state(l_idx, state_idx=0)
            rs = past_key_values.get_recurrent_state(l_idx, state_idx=0)
            c_len = cs.shape[-1]
            cs.copy_(self.snap_conv[lin_i, accepted_drafts, :, :, :c_len])
            rs.copy_(self.snap_rec[lin_i, accepted_drafts])
        for l_idx in self.full_layers:
            if l_idx < len(past_key_values.layers):
                past_key_values.layers[l_idx].crop(-rejected)
        self.rollbacks += 1
        return True


_DECODE_NO_MASK = {"full_attention": None, "linear_attention": None}


def fused_deltanet_multirow(
    gk: Any,
    lat_mod: Any,
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    snap_mgr: HybridSpecSnapshotManager | None = None,
) -> torch.Tensor:
    from .kernels import get_mr_kernels

    B, S, _ = hidden_states.shape
    if S == 1 and B <= 8:
        return gk.fused_deltanet_full_step(lat_mod, hidden_states, conv_state, recurrent_state)

    assert B == 1 and 2 <= S <= 4
    mrk = get_mr_kernels(gk.device)
    x_c = hidden_states.reshape(S, 4096)
    if not x_c.is_contiguous():
        x_c = x_c.contiguous()
    stream_ptr = ctypes.c_void_p(torch.cuda.current_stream(gk.device).cuda_stream)

    c_x = ctypes.c_uint64(x_c.data_ptr())
    c_qkv_q = ctypes.c_uint64(lat_mod.in_proj_qkv.qweight.data_ptr())
    c_qkv_s = ctypes.c_uint64(lat_mod.in_proj_qkv.scales.data_ptr())
    c_qkv_z = ctypes.c_uint64(lat_mod.in_proj_qkv.zeros.data_ptr())
    c_z_q = ctypes.c_uint64(lat_mod.in_proj_z.qweight.data_ptr())
    c_z_s = ctypes.c_uint64(lat_mod.in_proj_z.scales.data_ptr())
    c_z_z = ctypes.c_uint64(lat_mod.in_proj_z.zeros.data_ptr())
    c_b_q = ctypes.c_uint64(lat_mod.in_proj_b.qweight.data_ptr())
    c_b_s = ctypes.c_uint64(lat_mod.in_proj_b.scales.data_ptr())
    c_b_z = ctypes.c_uint64(lat_mod.in_proj_b.zeros.data_ptr())
    c_a_q = ctypes.c_uint64(lat_mod.in_proj_a.qweight.data_ptr())
    c_a_s = ctypes.c_uint64(lat_mod.in_proj_a.scales.data_ptr())
    c_a_z = ctypes.c_uint64(lat_mod.in_proj_a.zeros.data_ptr())

    c_oqkv = ctypes.c_uint64(gk.ws_qkv.data_ptr())
    c_oz = ctypes.c_uint64(gk.ws_z.data_ptr())
    c_ob = ctypes.c_uint64(gk.ws_b.data_ptr())
    c_oa = ctypes.c_uint64(gk.ws_a.data_ptr())

    p_proj = (ctypes.c_void_p * 17)(
        ctypes.cast(ctypes.byref(c_x), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_qkv_q), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_qkv_s), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_qkv_z), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_z_q), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_z_s), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_z_z), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_b_q), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_b_s), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_b_z), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_a_q), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_a_s), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_a_z), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_oqkv), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_oz), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_ob), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_oa), ctypes.c_void_p),
    )
    rc0 = mrk.nvcuda.cuLaunchKernel(mrk.qkv_fns[S], 1544, 1, 1, 256, 1, 1, S * 8448, stream_ptr, p_proj, None)
    if rc0 != 0:
        raise RuntimeError(f"multirow_qkvzba_m{S} failed: {rc0}")

    w2d = lat_mod.conv1d.weight.squeeze(1)
    if not w2d.is_contiguous():
        w2d = w2d.contiguous()
    c_st = ctypes.c_uint64(conv_state.data_ptr())
    c_w = ctypes.c_uint64(w2d.data_ptr())
    c_slen = ctypes.c_int(int(conv_state.shape[-1]))
    c_alog = ctypes.c_uint64(lat_mod.A_log.data_ptr())
    c_dtb = ctypes.c_uint64(lat_mod.dt_bias.data_ptr())
    c_nw = ctypes.c_uint64(lat_mod.norm.weight.data_ptr())
    c_rec = ctypes.c_uint64(recurrent_state.data_ptr())
    c_abf = ctypes.c_int(1 if lat_mod.A_log.dtype == torch.bfloat16 else 0)
    c_dbf = ctypes.c_int(1 if lat_mod.dt_bias.dtype == torch.bfloat16 else 0)

    for r in range(S):
        c_rqkv = ctypes.c_uint64(gk.ws_qkv[r].data_ptr())
        c_rqkvc = ctypes.c_uint64(gk.ws_qkv_conv[r].data_ptr())
        c_rz = ctypes.c_uint64(gk.ws_z[r].data_ptr())
        c_rb = ctypes.c_uint64(gk.ws_b[r].data_ptr())
        c_ra = ctypes.c_uint64(gk.ws_a[r].data_ptr())
        c_rcore = ctypes.c_uint64(gk.ws_core[r].data_ptr())

        p_conv = (ctypes.c_void_p * 5)(
            ctypes.cast(ctypes.byref(c_rqkv), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_st), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_w), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_rqkvc), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_slen), ctypes.c_void_p),
        )
        rc1 = gk.nvcuda.cuLaunchKernel(
            gk.fns[b"fused_causal_conv1d_silu_batched"], 32, 1, 1, 256, 1, 1, 0, stream_ptr, p_conv, None
        )
        if rc1 != 0:
            raise RuntimeError(f"conv row {r} failed: {rc1}")

        p_rec = (ctypes.c_void_p * 11)(
            ctypes.cast(ctypes.byref(c_rqkvc), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_rz), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_rb), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_ra), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_alog), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_dtb), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_nw), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_rec), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_rcore), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_abf), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_dbf), ctypes.c_void_p),
        )
        rc2 = gk.nvcuda.cuLaunchKernel(
            gk.fns[b"fused_gated_delta_recurrent_batched"], 32, 1, 1, 128, 1, 1, 0, stream_ptr, p_rec, None
        )
        if rc2 != 0:
            raise RuntimeError(f"delta row {r} failed: {rc2}")

        if snap_mgr is not None and r < S - 1:
            snap_mgr.save_row(lat_mod.layer_idx, r, conv_state, recurrent_state)

    return lat_mod.out_proj(gk.ws_core[:S].view(1, S, 4096))


def _cache_seq_len(past: Any) -> int:
    if past is None:
        return 0
    fn = getattr(past, "get_seq_length", None)
    if not callable(fn):
        return 0
    try:
        return int(fn())
    except (AttributeError, ValueError, TypeError):
        return 0


def forward_step_all_logits(
    engine: Any,
    step_tokens: torch.Tensor,
    past_key_values: Any = None,
    prefetch_next: bool = True,
    logits_to_keep: int = 0,
) -> tuple[torch.Tensor, Any]:
    """One backbone pass. ``logits_to_keep=1`` projects only the last position onto the vocabulary.

    Prefill must always pass ``logits_to_keep=1``: projecting every prompt position materialises
    ``[1, S, 248320]`` logits (about 0.5 MiB per token in BF16, 1-2 GiB for a multi-thousand-token prompt).
    """
    extra: dict[str, Any] = (
        {"logits_to_keep": int(logits_to_keep)}
        if (logits_to_keep and getattr(engine, "_supports_logits_to_keep", True))
        else {}
    )
    if hasattr(engine, "_anchor_ring_prepare"):
        return engine._anchor_ring_forward(
            step_tokens, past_key_values=past_key_values, prefetch_next=prefetch_next, **extra
        )

    engine._prefetch_next_step = prefetch_next
    num_layers = getattr(engine.config, "num_hidden_layers", None) or len(engine._decoder_layers())
    inputs_embeds = engine._embed_step_tokens(step_tokens)
    ple_fn = getattr(engine, "_compute_per_layer_inputs", None)
    ple_inputs = ple_fn(step_tokens) if callable(ple_fn) else None
    if ple_inputs is not None:
        extra["per_layer_inputs"] = ple_inputs
    if engine.num_slots > 1 or engine.pipeline_mode != "sequential":
        first_s = engine.vram_resident_layers
        for s in range(first_s, min(first_s + engine.num_slots, num_layers)):
            if s == first_s and engine.tail_prefetched:
                engine.tail_prefetched = False
                continue
            pinned_t = engine.weight_manager.load_layer_to_pinned_slot(s, target_slot_idx=s % engine.num_slots)
            slot = engine.pipeline.get_slot_for_loading(s)
            engine.pipeline.load_tensors_async(slot, pinned_t, pinned_pool=engine.weight_manager.pinned_pool)

    B, S = step_tokens.shape
    cache_kw = getattr(engine, "_cache_kwarg", "past_key_values")
    mask_arg = (
        _DECODE_NO_MASK
        if (B == 1 and 1 <= S <= 4 and past_key_values is not None and _cache_seq_len(past_key_values) > 0)
        else None
    )
    if cache_kw is not None:
        extra[cache_kw] = past_key_values
    if getattr(engine, "_supports_use_cache", True):
        extra["use_cache"] = True
    if cache_kw == "past_key_values" or mask_arg is not None:
        extra["attention_mask"] = mask_arg
    tracker = getattr(engine, "survivor_tracker", None)
    is_prefill_call = bool(logits_to_keep == 1 and (past_key_values is None or step_tokens.shape[-1] > 1))
    is_aux_call = bool(getattr(engine, "_auxiliary_depth", 0))
    if tracker is not None:
        tracker.begin_prefill_chunk(is_last_chunk=is_prefill_call, auxiliary=is_aux_call)
    try:
        outputs = engine.model(
            inputs_embeds=inputs_embeds,
            input_ids=None,
            **extra,
        )
    finally:
        if tracker is not None:
            tracker.end_prefill_chunk()
            if is_prefill_call and not is_aux_call:
                tracker.sync_draft_vocab()
    next_cache = getattr(outputs, "past_key_values", None)
    if next_cache is None and cache_kw and cache_kw != "past_key_values":
        next_cache = getattr(outputs, cache_kw, None)
    return outputs.logits, next_cache


def install_quintessentia_multirow_unifications(
    engine: Any,
    gk: Any,
    snap_mgr: HybridSpecSnapshotManager | None = None,
    enable_vocab_pruner: bool = False,
    captured: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )
    from src.quantization_q4 import QuintessentiaQ4Linear
    from .kernels import ZeroProbSubspaceVocabHead, get_mr_kernels

    mrk = get_mr_kernels(gk.device)
    if captured is None:
        captured = {}

    if not getattr(engine, "_quintessentia_mr_installed", False):
        for mod in engine.model.modules():
            if type(mod).__name__ == "Qwen3_5RMSNorm":
                eps_val = float(mod.eps)

                def make_rn_fwd(e_v):
                    def _rnf(self, x):
                        if x.is_cuda and x.dtype == torch.bfloat16 and self.weight.dtype == torch.bfloat16:
                            return gk.rmsnorm_zero_centered(x, self.weight, e_v)
                        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + e_v)
                        return (out * (1.0 + self.weight.float())).type_as(x)

                    return _rnf

                mod.forward = types.MethodType(make_rn_fwd(eps_val), mod)

        for submod in engine.model.model.layers.modules():
            if isinstance(submod, QuintessentiaQ4Linear):
                orig_qf = submod.forward

                def _mr_qfwd(x, _m=submod, _of=orig_qf):
                    M = x.numel() // x.shape[-1]
                    if x.is_cuda and x.dtype == torch.bfloat16 and _m.qweight.is_cuda and 1 <= M <= 4:
                        return mrk.linear_q4(x, _m)
                    return _of(x)

                submod.forward = _mr_qfwd

        orig_final_norm = engine.model.model.norm.forward

        def _capturing_final_norm(x):
            out = orig_final_norm(x)
            captured["post_norm"] = out
            return out

        engine.model.model.norm.forward = _capturing_final_norm

        for layer in engine.model.model.layers:
            mlp = layer.mlp
            orig_mlp_fwd = mlp.forward

            def make_mlp_fwd(orig_f):
                def _mfwd(self, x):
                    if x.is_cuda and x.dtype == torch.bfloat16 and hasattr(self.gate_proj, "qweight"):
                        M = x.numel() // x.shape[-1]
                        if x.ndim == 3 and x.shape[1] == 1 and x.shape[0] <= 8:
                            act = gk.fused_mlp_gate_up_swiglu(x, self.gate_proj, self.up_proj)
                            return self.down_proj(act)
                        if 2 <= M <= 4:
                            act = mrk.gate_up_swiglu(x, self.gate_proj, self.up_proj)
                            return self.down_proj(act)
                    return orig_f(x)

                return _mfwd

            mlp.forward = types.MethodType(make_mlp_fwd(orig_mlp_fwd), mlp)

            lat = getattr(layer, "linear_attn", None)
            if lat is not None:
                orig_lat_fwd = lat.forward

                def make_lat_fwd(orig_lf):
                    def _lfwd(self, hidden_states, cache_params=None, attention_mask=None, **kw):
                        B, S, _ = hidden_states.shape
                        if (
                            hidden_states.is_cuda
                            and ((S == 1 and B <= 8) or (B == 1 and 2 <= S <= 4))
                            and attention_mask is None
                            and cache_params is not None
                            and not cache_params.is_recording_past(self.layer_idx)
                            and hasattr(self.in_proj_qkv, "qweight")
                        ):
                            cs = cache_params.get_conv_state(self.layer_idx, state_idx=0)
                            rs = cache_params.get_recurrent_state(self.layer_idx, state_idx=0)
                            if (
                                cs is not None
                                and rs is not None
                                and cs.is_contiguous()
                                and rs.is_contiguous()
                                and rs.dtype == torch.float32
                            ):
                                return fused_deltanet_multirow(gk, self, hidden_states, cs, rs, snap_mgr=snap_mgr)
                        return orig_lf(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kw)

                    return _lfwd

                lat.forward = types.MethodType(make_lat_fwd(orig_lat_fwd), lat)

            s_attn = getattr(layer, "self_attn", None)
            if s_attn is not None:
                orig_sa_fwd = s_attn.forward

                def make_sa_fwd(orig_sf):
                    def _safwd(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
                        B, S, _ = hidden_states.shape
                        if (
                            B == 1
                            and 2 <= S <= 4
                            and past_key_values is not None
                            and past_key_values.get_seq_length() > 0
                        ):
                            input_shape = hidden_states.shape[:-1]
                            hidden_shape = (*input_shape, -1, self.head_dim)
                            query_states, gate = torch.chunk(
                                self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
                            )
                            gate = gate.reshape(*input_shape, -1)
                            query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
                            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                            cos, sin = position_embeddings
                            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
                            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                                self.config._attn_implementation, eager_attention_forward
                            )
                            base_kv = key_states.shape[2] - S
                            row_outs = []
                            for r in range(S):
                                ao_r, _ = attention_interface(
                                    self,
                                    query_states[:, :, r : r + 1, :],
                                    key_states[:, :, : base_kv + r + 1, :],
                                    value_states[:, :, : base_kv + r + 1, :],
                                    None,
                                    dropout=0.0,
                                    scaling=self.scaling,
                                    **kwargs,
                                )
                                row_outs.append(ao_r)
                            attn_output = torch.cat(row_outs, dim=1).reshape(*input_shape, -1).contiguous()
                            attn_output = attn_output * torch.sigmoid(gate)
                            return self.o_proj(attn_output), None
                        return orig_sf(
                            hidden_states, position_embeddings, attention_mask, past_key_values=past_key_values, **kwargs
                        )

                    return _safwd

                s_attn.forward = types.MethodType(make_sa_fwd(orig_sa_fwd), s_attn)

        engine._quintessentia_mr_installed = True

    raw_q4_head = engine.model.lm_head.q4_head if hasattr(engine.model.lm_head, "q4_head") else engine.model.lm_head
    if hasattr(raw_q4_head, "qweight") and raw_q4_head.qweight.is_cuda:
        if enable_vocab_pruner:
            pin_set = set(getattr(engine.tokenizer, "all_special_ids", []) or [])
            for tok_str in ("<think>", "</think>", "<|im_end|>", "<|im_start|>", "<|endoftext|>", "\n", "\n\n"):
                for tid in engine.tokenizer.encode(tok_str, add_special_tokens=False):
                    pin_set.add(int(tid))
            pruner = ZeroProbSubspaceVocabHead(
                raw_q4_head,
                gk,
                norm_weight=engine.model.model.norm.weight,
                zero_centered_norm=True,
                rank_r=512,
                survivors=12288,
                pinned_token_ids=sorted(pin_set),
            )
            engine.model.lm_head = pruner
        else:
            orig_h_fwd = raw_q4_head.forward

            def _phfwd(x, _q=raw_q4_head, _of=orig_h_fwd):
                if x.is_cuda and x.dtype == torch.bfloat16 and x.ndim == 3:
                    M = x.numel() // x.shape[-1]
                    if x.shape[1] == 1 and x.shape[0] <= 8:
                        return gk.persistent_gemv_q4(x, _q)
                    if 2 <= M <= 4:
                        return mrk.linear_q4(x, _q)
                return _of(x)

            raw_q4_head.forward = _phfwd
            engine.model.lm_head = raw_q4_head

    return captured


def fused_deltanet_q1_multirow(
    gk: Any,
    lat_mod: Any,
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    snap_mgr: HybridSpecSnapshotManager | None = None,
) -> torch.Tensor:
    """Fused CUDA causal_conv1d + gated_delta_recurrent + RMSNormGated over Q1-Cubes (or BF16) projections."""
    B, S, _ = hidden_states.shape
    assert B == 1 and 1 <= S <= 4
    qkv = lat_mod.in_proj_qkv(hidden_states).reshape(S, 8192).contiguous()
    z = lat_mod.in_proj_z(hidden_states).reshape(S, 4096).contiguous()
    b = lat_mod.in_proj_b(hidden_states).reshape(S, 32).contiguous()
    a = lat_mod.in_proj_a(hidden_states).reshape(S, 32).contiguous()

    buf = getattr(gk, "_q1_dn_argbuf", None)
    if buf is None:
        c_rqkv = ctypes.c_uint64(0)
        c_st = ctypes.c_uint64(0)
        c_w = ctypes.c_uint64(0)
        c_rqkvc = ctypes.c_uint64(0)
        c_slen = ctypes.c_int(0)
        c_rz = ctypes.c_uint64(0)
        c_rb = ctypes.c_uint64(0)
        c_ra = ctypes.c_uint64(0)
        c_alog = ctypes.c_uint64(0)
        c_dtb = ctypes.c_uint64(0)
        c_nw = ctypes.c_uint64(0)
        c_rec = ctypes.c_uint64(0)
        c_rcore = ctypes.c_uint64(0)
        c_abf = ctypes.c_int(0)
        c_dbf = ctypes.c_int(0)
        c_stream = ctypes.c_void_p(0)
        p_conv = (ctypes.c_void_p * 5)(
            ctypes.addressof(c_rqkv), ctypes.addressof(c_st), ctypes.addressof(c_w),
            ctypes.addressof(c_rqkvc), ctypes.addressof(c_slen),
        )
        p_rec = (ctypes.c_void_p * 11)(
            ctypes.addressof(c_rqkvc), ctypes.addressof(c_rz), ctypes.addressof(c_rb),
            ctypes.addressof(c_ra), ctypes.addressof(c_alog), ctypes.addressof(c_dtb),
            ctypes.addressof(c_nw), ctypes.addressof(c_rec), ctypes.addressof(c_rcore),
            ctypes.addressof(c_abf), ctypes.addressof(c_dbf),
        )
        buf = (
            c_rqkv, c_st, c_w, c_rqkvc, c_slen, c_rz, c_rb, c_ra,
            c_alog, c_dtb, c_nw, c_rec, c_rcore, c_abf, c_dbf, c_stream,
            p_conv, p_rec, gk.fns[b"fused_causal_conv1d_silu_batched"], gk.fns[b"fused_gated_delta_recurrent_batched"],
        )
        gk._q1_dn_argbuf = buf
    (
        c_rqkv, c_st, c_w, c_rqkvc, c_slen, c_rz, c_rb, c_ra,
        c_alog, c_dtb, c_nw, c_rec, c_rcore, c_abf, c_dbf, c_stream,
        p_conv, p_rec, fn_conv, fn_rec,
    ) = buf

    w2d = lat_mod.conv1d.weight.squeeze(1)
    if not w2d.is_contiguous():
        w2d = w2d.contiguous()
    c_stream.value = torch.cuda.current_stream(gk.device).cuda_stream
    c_st.value = conv_state.data_ptr()
    c_w.value = w2d.data_ptr()
    c_slen.value = int(conv_state.shape[-1])
    c_alog.value = lat_mod.A_log.data_ptr()
    c_dtb.value = lat_mod.dt_bias.data_ptr()
    c_nw.value = lat_mod.norm.weight.data_ptr()
    c_rec.value = recurrent_state.data_ptr()
    c_abf.value = 1 if lat_mod.A_log.dtype == torch.bfloat16 else 0
    c_dbf.value = 1 if lat_mod.dt_bias.dtype == torch.bfloat16 else 0

    qkv_base = qkv.data_ptr()
    qkvc_base = gk.ws_qkv_conv.data_ptr()
    z_base = z.data_ptr()
    b_base = b.data_ptr()
    a_base = a.data_ptr()
    core_base = gk.ws_core.data_ptr()
    launch = gk.nvcuda.cuLaunchKernel

    for r in range(S):
        c_rqkv.value = qkv_base + r * (8192 * 2)
        c_rqkvc.value = qkvc_base + r * (8192 * 2)
        c_rz.value = z_base + r * (4096 * 2)
        c_rb.value = b_base + r * (32 * 2)
        c_ra.value = a_base + r * (32 * 2)
        c_rcore.value = core_base + r * (4096 * 2)

        rc1 = launch(fn_conv, 32, 1, 1, 256, 1, 1, 0, c_stream, p_conv, None)
        if rc1 != 0:
            raise RuntimeError(f"q1_deltanet conv row {r} failed: {rc1}")

        rc2 = launch(fn_rec, 32, 1, 1, 128, 1, 1, 0, c_stream, p_rec, None)
        if rc2 != 0:
            raise RuntimeError(f"q1_deltanet delta row {r} failed: {rc2}")

        if snap_mgr is not None and r < S - 1:
            snap_mgr.save_row(lat_mod.layer_idx, r, conv_state, recurrent_state)

    return lat_mod.out_proj(gk.ws_core[:S].view(1, S, 4096))


def install_q1_qwen35_fast_decode_unifications(
    engine: Any,
    *,
    max_drafts: int = 0,
    full_residency: bool = False,
) -> HybridSpecSnapshotManager | None:
    """Attach fused Qwen3.5 RMSNorm + GatedDeltaNet CUDA kernels and zero-replay snapshot slots to Q1-Cubes."""
    cfg = getattr(engine, "config", None)
    dev = getattr(engine, "device", None)
    if (
        cfg is None
        or dev is None
        or torch.device(dev).type != "cuda"
        or not torch.cuda.is_available()
        or getattr(cfg, "model_type", None) != "qwen3_5_text"
        or int(getattr(cfg, "hidden_size", 0)) != 4096
        or int(getattr(cfg, "linear_num_key_heads", 0)) != 16
        or int(getattr(cfg, "linear_num_value_heads", 0)) != 32
        or int(getattr(cfg, "linear_key_head_dim", 0)) != 128
        or int(getattr(cfg, "linear_value_head_dim", 0)) != 128
        or int(getattr(cfg, "linear_conv_kernel_dim", 0)) != 4
    ):
        return None
    if getattr(engine, "_q1_qwen35_fast_installed", False):
        return getattr(engine, "_q1_snap_mgr", None)

    from .kernels import GrandUnifiedKernels

    gk = getattr(engine, "_q1_gk", None)
    if gk is None:
        gk = GrandUnifiedKernels(torch.device(dev), max_batch=8)
        engine._q1_gk = gk
    snap_mgr: HybridSpecSnapshotManager | None = None
    if full_residency and max_drafts > 0:
        try:
            free_b, _ = torch.cuda.mem_get_info(dev)
            slots = min(3, max(1, int(max_drafts)))
            # 24 layers * (64 KiB + 2 MiB) * slots ~= 50 MiB * slots
            need_b = slots * 54 * MiB
            if free_b >= need_b + 384 * MiB:
                snap_mgr = HybridSpecSnapshotManager(cfg, torch.device(dev), max_drafts=slots)
        except Exception:
            snap_mgr = None

    for mod in engine.model.modules():
        if type(mod).__name__ == "Qwen3_5RMSNorm":
            eps_val = float(mod.eps)

            def make_rn_fwd(e_v):
                def _rnf(self, x):
                    if x.is_cuda and x.dtype == torch.bfloat16 and self.weight.dtype == torch.bfloat16:
                        return gk.rmsnorm_zero_centered(x, self.weight, e_v)
                    out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + e_v)
                    return (out * (1.0 + self.weight.float())).type_as(x)

                return _rnf

            mod.forward = types.MethodType(make_rn_fwd(eps_val), mod)

    for layer in engine._decoder_layers():
        lat = getattr(layer, "linear_attn", None)
        if lat is None:
            continue
        orig_lat_fwd = lat.forward

        def make_q1_lat_fwd(orig_lf):
            def _q1_lfwd(self, hidden_states, cache=None, attention_mask=None, **kw):
                c_obj = cache if cache is not None else kw.get("cache_params")
                B, S, _ = hidden_states.shape
                if (
                    hidden_states.is_cuda
                    and hidden_states.dtype == torch.bfloat16
                    and B == 1
                    and 1 <= S <= 4
                    and attention_mask is None
                    and c_obj is not None
                    and getattr(self.conv1d, "bias", None) is None
                    and not c_obj.is_recording_past(self.layer_idx)
                ):
                    cs = c_obj.get_conv_state(self.layer_idx, state_idx=0)
                    rs = c_obj.get_recurrent_state(self.layer_idx, state_idx=0)
                    if (
                        cs is not None
                        and rs is not None
                        and cs.is_contiguous()
                        and rs.is_contiguous()
                        and cs.shape == (1, 8192, 4)
                        and rs.shape == (1, 32, 128, 128)
                        and cs.dtype == torch.bfloat16
                        and rs.dtype == torch.float32
                    ):
                        return fused_deltanet_q1_multirow(gk, self, hidden_states, cs, rs, snap_mgr=snap_mgr)
                if cache is not None:
                    return orig_lf(hidden_states, cache=cache, attention_mask=attention_mask, **kw)
                return orig_lf(hidden_states, attention_mask=attention_mask, **kw)

            return _q1_lfwd

        lat.forward = types.MethodType(make_q1_lat_fwd(orig_lat_fwd), lat)

    engine._q1_snap_mgr = snap_mgr
    engine._q1_qwen35_fast_installed = True
    return snap_mgr


class QuintessentiaNativeMTPHead:
    """Loads `model-mtp.safetensors` as Q4 (130.5 MiB VRAM) and scores over `QuintessentiaDraftVocab`."""

    def __init__(self, engine: Any, gk: Any, draft_vocab: QuintessentiaDraftVocab):
        from safetensors import safe_open
        from transformers.cache_utils import DynamicCache
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm
        from src.quintessentia_integrated import set_module_tensor_to_device
        from src.quintessentia_nvrtc import can_use_fused_dense_q4, fused_dense_linear_q4
        from src.quantization_q4 import QuintessentiaQ4Linear, quantize_tensor_q4
        from .kernels import get_mr_kernels

        self._DynamicCache = DynamicCache
        self.engine = engine
        self.gk = gk
        self.draft_vocab = draft_vocab
        self.device = engine.device
        self.dtype = engine.dtype
        cfg = engine.config

        mtp_cfg = copy.deepcopy(cfg)
        mtp_cfg.num_hidden_layers = 1
        mtp_cfg.layer_types = ["full_attention"]
        mtp_cfg._attn_implementation = getattr(cfg, "_attn_implementation", "sdpa")
        self.mtp_cfg = mtp_cfg

        mtp_raw = {}
        source = getattr(engine, "model_source", None)
        if source is not None:
            for k, (path, stored) in source.mtp_entries().items():
                with safe_open(path, framework="pt", device="cpu") as f:
                    mtp_raw[k] = f.get_tensor(stored)
        else:
            with safe_open(str(Path(engine.model_dir) / "model-mtp.safetensors"), framework="pt", device="cpu") as f:
                for k in f.keys():
                    mtp_raw[k] = f.get_tensor(k)

        layer = Qwen3_5DecoderLayer(mtp_cfg, layer_idx=0).to(dtype=self.dtype)
        for parent in (layer.self_attn, layer.mlp):
            for ch_name, ch in list(parent.named_children()):
                if isinstance(ch, torch.nn.Linear):
                    setattr(
                        parent,
                        ch_name,
                        QuintessentiaQ4Linear(
                            out_features=ch.out_features,
                            in_features=ch.in_features,
                            has_bias=(ch.bias is not None),
                            group_size=64,
                            compute_dtype=self.dtype,
                            scratch_provider=engine._get_q4_scratch,
                        ),
                    )
        self.fc_q4 = QuintessentiaQ4Linear(
            out_features=cfg.hidden_size,
            in_features=cfg.hidden_size * 2,
            has_bias=False,
            scratch_provider=engine._get_q4_scratch,
            group_size=64,
            compute_dtype=self.dtype,
        )
        p_fc, s_fc, z_fc = quantize_tensor_q4(
            mtp_raw.pop("mtp.fc.weight"),
            group_size=64,
            compute_dtype=self.dtype,
            quant_device=self.device,
            target_device=self.device,
        )
        self.fc_q4.bind_q4_tensors(p_fc, s_fc, z_fc)

        for mod_name, submod in layer.named_modules():
            if isinstance(submod, QuintessentiaQ4Linear):
                w_cpu = mtp_raw.pop(f"mtp.layers.0.{mod_name}.weight")
                pq, sq, zq = quantize_tensor_q4(
                    w_cpu,
                    group_size=64,
                    compute_dtype=self.dtype,
                    quant_device=self.device,
                    target_device=self.device,
                )
                submod.bind_q4_tensors(pq, sq, zq)

        for p_name, _ in list(layer.named_parameters()):
            w_cpu = mtp_raw.pop(f"mtp.layers.0.{p_name}")
            set_module_tensor_to_device(
                layer, p_name, self.device, value=w_cpu.to(device=self.device, dtype=self.dtype), clear_cache=False
            )

        self.layer = layer.eval()
        self.enorm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(device=self.device, dtype=self.dtype)
        self.hnorm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(device=self.device, dtype=self.dtype)
        self.head_norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(device=self.device, dtype=self.dtype)
        self.enorm.weight.data.copy_(mtp_raw.pop("mtp.pre_fc_norm_embedding.weight").to(device=self.device, dtype=self.dtype))
        self.hnorm.weight.data.copy_(mtp_raw.pop("mtp.pre_fc_norm_hidden.weight").to(device=self.device, dtype=self.dtype))
        self.head_norm.weight.data.copy_(mtp_raw.pop("mtp.norm.weight").to(device=self.device, dtype=self.dtype))
        assert len(mtp_raw) == 0, f"Unconsumed MTP tensors: {list(mtp_raw.keys())}"

        mrk = get_mr_kernels(self.device)
        for mod in [self.fc_q4, self.enorm, self.hnorm, self.head_norm, *self.layer.modules()]:
            if type(mod).__name__ == "Qwen3_5RMSNorm":
                eps_v = float(mod.eps)

                def make_rn(ev):
                    def _rf(s_mod, x):
                        if x.is_cuda and x.dtype == torch.bfloat16:
                            return gk.rmsnorm_zero_centered(x, s_mod.weight, ev)
                        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + ev)
                        return (out * (1.0 + s_mod.weight.float())).type_as(x)

                    return _rf

                mod.forward = types.MethodType(make_rn(eps_v), mod)
            elif isinstance(mod, QuintessentiaQ4Linear):
                orig_f = mod.forward

                def _qfwd(x, _m=mod, _of=orig_f):
                    if x.is_cuda and x.dtype == torch.bfloat16:
                        M = x.numel() // x.shape[-1]
                        if M == 1 and can_use_fused_dense_q4(x, _m.qweight, _m.scales, _m.zeros, _m.group_size):
                            return fused_dense_linear_q4(
                                x, _m.qweight, _m.scales, _m.zeros, bias=_m._buffers["bias"], group_size=_m.group_size
                            )
                        if 2 <= M <= 4:
                            return mrk.linear_q4(x, _m)
                    return _of(x)

                mod.forward = _qfwd

        orig_mtp_mlp = self.layer.mlp.forward

        def _mtp_mlp_fwd(s_mlp, x):
            if x.is_cuda and x.dtype == torch.bfloat16:
                M = x.numel() // x.shape[-1]
                if x.ndim == 3 and x.shape[1] == 1 and x.shape[0] == 1:
                    act = gk.fused_mlp_gate_up_swiglu(x, s_mlp.gate_proj, s_mlp.up_proj)
                    return s_mlp.down_proj(act)
                if 2 <= M <= 4:
                    act = mrk.gate_up_swiglu(x, s_mlp.gate_proj, s_mlp.up_proj)
                    return s_mlp.down_proj(act)
            return orig_mtp_mlp(x)

        self.layer.mlp.forward = types.MethodType(_mtp_mlp_fwd, self.layer.mlp)
        self._QuintessentiaQ4Linear = QuintessentiaQ4Linear
        self.kv_cache = DynamicCache(config=self.mtp_cfg)
        self.committed_len = 0
        self.mtp_step_times_ms: list[float] = []

    def reset_cache(self) -> None:
        self.kv_cache = self._DynamicCache(config=self.mtp_cfg)
        self.committed_len = 0
        self.mtp_step_times_ms.clear()

    def q4_vram_bytes(self) -> int:
        total = 0
        for m in [self.fc_q4, *self.layer.modules(), self.enorm, self.hnorm, self.head_norm]:
            if isinstance(m, self._QuintessentiaQ4Linear):
                total += m.qweight.nbytes + m.scales.nbytes + m.zeros.nbytes
        for p in [*self.layer.parameters(), *self.enorm.parameters(), *self.hnorm.parameters(), *self.head_norm.parameters()]:
            total += p.numel() * p.element_size()
        return total

    def _raw_lm_head(self):
        h = self.engine.model.lm_head
        return h.q4_head if hasattr(h, "q4_head") else h

    def _score_draft(self, normed_last_1x1xd: torch.Tensor) -> tuple[int, float]:
        x2d = normed_last_1x1xd.reshape(1, -1)
        cand = self.draft_vocab.gpu_list
        scores = self.gk.indexed_vocab_scores(x2d, cand, self._raw_lm_head())[0]
        top_vals, top_pos = torch.topk(scores, k=min(64, scores.numel()), dim=-1)
        probs_top = torch.softmax(top_vals, dim=-1)
        best_tok = cand[0, top_pos[0]]
        tok_f, conf = torch.stack([best_tok.to(torch.float32), probs_top[0]]).tolist()
        return int(tok_f), float(conf)

    def _forward_pairs(self, h_post_seq: torch.Tensor, next_token_ids: list[int], start_pos: int) -> torch.Tensor:
        L = len(next_token_ids)
        tok_t = torch.tensor([next_token_ids], dtype=torch.long)
        emb = self.engine._embed_step_tokens(tok_t)
        en = self.enorm(emb)
        hn = self.hnorm(h_post_seq)
        x = self.fc_q4(torch.cat([en, hn], dim=-1))

        pos_1d = torch.arange(start_pos, start_pos + L, device=self.device, dtype=torch.long)
        pos_3d = pos_1d.view(1, 1, L).expand(3, 1, L)
        pos_emb = self.engine.model.model.rotary_emb(x, pos_3d)

        attn_mask = None
        if L > 1:
            total_kv = start_pos + L
            mask = torch.zeros((L, total_kv), device=self.device, dtype=self.dtype)
            causal_part = torch.triu(torch.full((L, L), float("-inf"), device=self.device, dtype=self.dtype), diagonal=1)
            mask[:, start_pos:] = causal_part
            attn_mask = mask.view(1, 1, L, total_kv)

        out = self.layer(
            x,
            attention_mask=attn_mask,
            position_ids=pos_1d.unsqueeze(0),
            past_key_values=self.kv_cache,
            use_cache=True,
            position_embeddings=pos_emb,
        )
        if isinstance(out, tuple):
            out = out[0]
        return self.head_norm(out)

    def commit_and_draft_first(self, h_post_seq: torch.Tensor, next_token_ids: list[int]) -> tuple[int, float, torch.Tensor]:
        t0 = time.perf_counter()
        cur_kv = self.kv_cache.get_seq_length()
        if cur_kv > self.committed_len:
            self.kv_cache.layers[0].crop(-(cur_kv - self.committed_len))

        L = len(next_token_ids)
        normed = self._forward_pairs(h_post_seq, next_token_ids, start_pos=self.committed_len)
        self.committed_len += L
        last_normed = normed[:, -1:, :]
        d1, p1 = self._score_draft(last_normed)
        if L <= 4:
            self.mtp_step_times_ms.append((time.perf_counter() - t0) * 1000.0)
        return d1, p1, last_normed

    def draft_more(self, prev_mtp_normed: torch.Tensor, prev_draft_tok: int) -> tuple[int, float, torch.Tensor]:
        t0 = time.perf_counter()
        cur_kv = self.kv_cache.get_seq_length()
        normed = self._forward_pairs(prev_mtp_normed, [prev_draft_tok], start_pos=cur_kv)
        last_normed = normed[:, -1:, :]
        d2, p2 = self._score_draft(last_normed)
        self.mtp_step_times_ms.append((time.perf_counter() - t0) * 1000.0)
        return d2, p2, last_normed


def install_quintessentia_anchor_ring_schedule(engine: Any, resident_set: set[int]) -> dict[str, Any]:
    """Installs `.coli_analysis.json` / `PriorityRanker` non-contiguous Anchor Residency + 3-slot queue."""
    from src.quintessentia_integrated import set_module_tensor_to_device

    for h in engine._hook_handles:
        h.remove()
    engine._hook_handles.clear()

    for r_idx in sorted(resident_set):
        engine._load_resident_layer(r_idx)

    num_layers = engine.config.num_hidden_layers
    stream_layers = [i for i in range(num_layers) if i not in resident_set]
    next_stream_layer: dict[int, int | None] = {
        l_id: (stream_layers[pos + 1] if pos + 1 < len(stream_layers) else None)
        for pos, l_id in enumerate(stream_layers)
    }
    slot_queues: list[list[int]] = [[] for _ in range(engine.num_slots)]
    next_in_slot_queue: dict[int, int | None] = {}
    for s_idx in stream_layers:
        slot_queues[s_idx % engine.num_slots].append(s_idx)
    for q in slot_queues:
        for pos, l_id in enumerate(q):
            next_in_slot_queue[l_id] = q[pos + 1] if (pos + 1 < len(q)) else None

    tail_prefetched_slot = [False] * engine.num_slots
    dynamic_tail_prefetched = [False]
    layer_q4_bindings = []
    layer_param_names = []
    for layer_mod in engine.model.model.layers:
        qb, rp = engine._collect_layer_bindings(layer_mod)
        layer_q4_bindings.append(qb)
        layer_param_names.append(rp)

    engine.active_slot_per_layer = [None] * num_layers

    for idx, layer_module in enumerate(engine.model.model.layers):
        layer_module._air_layer_idx = idx

        def _pre_hook(module, _args):
            cancel_fn = getattr(engine, "_step_cancel_check", None)
            if cancel_fn is not None:
                cancel_fn()
            i = module._air_layer_idx
            warm = getattr(engine, "probability_warmup", None)
            dynamic = bool(warm is not None and warm.dynamic_regions)
            if dynamic:
                warm.before_layer(i)
            if i in resident_set:
                return
            slot = engine.pipeline.acquire_slot_for_compute(i)
            engine.active_slot_per_layer[i] = slot
            slot_map = getattr(slot, "active_tensors", None) or slot.tensors
            for submod, q_k, s_k, z_k in layer_q4_bindings[i]:
                submod.bind_q4_tensors(slot_map[q_k], slot_map[s_k], slot_map[z_k])
            for k in layer_param_names[i]:
                if k in slot_map:
                    set_module_tensor_to_device(module, k, engine.device, value=slot_map[k], clear_cache=False)
            if dynamic:
                nxt_s = next_stream_layer.get(i)
                if nxt_s is not None:
                    nxt_sid = nxt_s % engine.num_slots
                    nxt_slot = engine.pipeline.slots[nxt_sid]
                    if nxt_slot.current_layer_idx != nxt_s or not nxt_slot.loaded_keys:
                        pinned_t = engine.weight_manager.load_layer_to_pinned_slot(nxt_s, target_slot_idx=nxt_sid)
                        ready_slot = engine.pipeline.get_slot_for_loading(nxt_s)
                        engine.pipeline.load_tensors_async(ready_slot, pinned_t, pinned_pool=engine.weight_manager.pinned_pool)

        def _post_hook(module, _args, output):
            i = module._air_layer_idx
            if i in resident_set:
                return output
            slot = engine.active_slot_per_layer[i]
            engine.pipeline.release_slot_after_compute(slot)
            engine.active_slot_per_layer[i] = None

            warm = getattr(engine, "probability_warmup", None)
            if warm is not None and warm.dynamic_regions:
                if next_stream_layer.get(i) is None and engine.tail_to_head_prefetch and engine._prefetch_next_step and stream_layers:
                    first_l = stream_layers[0]
                    first_sid = first_l % engine.num_slots
                    first_slot = engine.pipeline.slots[first_sid]
                    if first_slot.state.value == "FREE" and not dynamic_tail_prefetched[0]:
                        pinned_t = engine.weight_manager.load_layer_to_pinned_slot(first_l, target_slot_idx=first_sid)
                        next_slot = engine.pipeline.get_slot_for_loading(first_l)
                        engine.pipeline.load_tensors_async(next_slot, pinned_t, pinned_pool=engine.weight_manager.pinned_pool)
                        dynamic_tail_prefetched[0] = True
                return output

            sid = slot.slot_id
            nxt_l = next_in_slot_queue[i]
            if nxt_l is not None:
                pinned_t = engine.weight_manager.load_layer_to_pinned_slot(nxt_l, target_slot_idx=sid)
                next_slot = engine.pipeline.get_slot_for_loading(nxt_l)
                engine.pipeline.load_tensors_async(next_slot, pinned_t, pinned_pool=engine.weight_manager.pinned_pool)
            elif engine.tail_to_head_prefetch and engine._prefetch_next_step and slot_queues[sid]:
                first_l = slot_queues[sid][0]
                pinned_t = engine.weight_manager.load_layer_to_pinned_slot(first_l, target_slot_idx=sid)
                next_slot = engine.pipeline.get_slot_for_loading(first_l)
                engine.pipeline.load_tensors_async(next_slot, pinned_t, pinned_pool=engine.weight_manager.pinned_pool)
                tail_prefetched_slot[sid] = True
            return output

        engine._hook_handles.append(layer_module.register_forward_pre_hook(_pre_hook))
        engine._hook_handles.append(layer_module.register_forward_hook(_post_hook))

    def _anchor_ring_forward(
        step_tokens: torch.Tensor, past_key_values: Any = None, prefetch_next: bool = True, **model_kwargs
    ):
        engine._prefetch_next_step = prefetch_next
        warm = getattr(engine, "probability_warmup", None)
        is_prefill = past_key_values is None or step_tokens.shape[-1] > 1
        if warm is not None:
            warm.begin_step(step_tokens, past_key_values, prefill=is_prefill, prefill_end=True)
        inputs_embeds = engine._embed_step_tokens(step_tokens)
        if warm is not None and warm.dynamic_regions:
            if stream_layers:
                if dynamic_tail_prefetched[0]:
                    dynamic_tail_prefetched[0] = False
                else:
                    first_l = stream_layers[0]
                    first_sid = first_l % engine.num_slots
                    pinned_t = engine.weight_manager.load_layer_to_pinned_slot(first_l, target_slot_idx=first_sid)
                    slot = engine.pipeline.get_slot_for_loading(first_l)
                    engine.pipeline.load_tensors_async(slot, pinned_t, pinned_pool=engine.weight_manager.pinned_pool)
        else:
            for sid, q in enumerate(slot_queues):
                if not q:
                    continue
                if tail_prefetched_slot[sid]:
                    tail_prefetched_slot[sid] = False
                    continue
                first_l = q[0]
                pinned_t = engine.weight_manager.load_layer_to_pinned_slot(first_l, target_slot_idx=sid)
                slot = engine.pipeline.get_slot_for_loading(first_l)
                engine.pipeline.load_tensors_async(slot, pinned_t, pinned_pool=engine.weight_manager.pinned_pool)
        B, S = step_tokens.shape
        mask_arg = (
            _DECODE_NO_MASK
            if (B == 1 and 1 <= S <= 4 and past_key_values is not None and _cache_seq_len(past_key_values) > 0)
            else None
        )
        tracker = getattr(engine, "survivor_tracker", None)
        is_aux_ring = bool(getattr(engine, "_auxiliary_depth", 0) or (warm is not None and getattr(warm, "_auxiliary_depth", 0)))
        if tracker is not None:
            tracker.begin_prefill_chunk(is_last_chunk=bool(is_prefill), auxiliary=is_aux_ring)
        try:
            outputs = engine.model(
                inputs_embeds=inputs_embeds,
                input_ids=None,
                attention_mask=mask_arg,
                past_key_values=past_key_values,
                use_cache=True,
                **model_kwargs,
            )
        finally:
            if tracker is not None:
                tracker.end_prefill_chunk()
                if is_prefill and not is_aux_ring:
                    tracker.sync_draft_vocab()
        if warm is not None:
            warm.end_step(outputs.logits, outputs.past_key_values)
        return outputs.logits, outputs.past_key_values

    def _reset_anchor_ring() -> None:
        for idx_s in range(len(tail_prefetched_slot)):
            tail_prefetched_slot[idx_s] = False
        dynamic_tail_prefetched[0] = False

    engine._anchor_ring_prepare = True
    engine._anchor_ring_forward = _anchor_ring_forward
    engine._anchor_ring_reset = _reset_anchor_ring
    return {"resident_layers": sorted(resident_set), "streamed_layers": stream_layers, "slot_queues": slot_queues}


def _detach_park_cpu(tensor: torch.Tensor, allow_pin: bool) -> torch.Tensor:
    cpu_t = tensor.detach().to("cpu", non_blocking=False)
    if allow_pin:
        try:
            return cpu_t.pin_memory()
        except RuntimeError:
            return cpu_t
    return cpu_t


class QuintessentiaKVPrefixSession:
    """Multi-turn KV + GatedDeltaNet + MTP session ledger with Host-RAM parking (`0 MiB` idle VRAM)."""

    def __init__(self):
        self.fed: list[int] = []
        self.unfed_tail: list[int] = []
        self.chatml_think_format: bool = False
        self.tainted: bool = False
        self.parked_backbone: dict[int, dict[str, Any]] | None = None
        self.parked_mtp: dict[str, torch.Tensor] | None = None
        self.parked_mtp_committed_len: int = 0
        self.config = None
        self.park_ms: float = 0.0
        self.parked_bytes: int = 0
        self._pending_bridge_h: torch.Tensor | None = None

    def reset(self) -> None:
        self.fed = []
        self.unfed_tail = []
        self.chatml_think_format = False
        self.tainted = False
        self.parked_backbone = None
        self.parked_mtp = None
        self.parked_mtp_committed_len = 0
        self.parked_bytes = 0
        self._pending_bridge_h = None

    def record_turn_end(
        self,
        config: Any,
        all_ids_full: list[int],
        past_key_values: Any,
        last_fed_h_post: torch.Tensor | None,
        engine: Any,
        mtp_head: QuintessentiaNativeMTPHead | None,
        captured: dict[str, torch.Tensor],
        chatml_think_format: bool = False,
        consumed_tokens: int | None = None,
    ) -> float:
        self.config = config
        # Cache-only callers need no engine. Their GPU-to-CPU copies below are
        # blocking, so only an actual CUDA executor needs allocator/sync work.
        execution_device = torch.device(getattr(engine, "device", "cpu"))
        using_cuda = execution_device.type == "cuda"
        self.chatml_think_format = bool(chatml_think_format)
        all_ids = list(all_ids_full)
        need_bridge = (mtp_head is not None) or (last_fed_h_post is not None) or ("post_norm" in captured)
        if not hasattr(past_key_values, "layers"):
            del past_key_values
            self.reset()
            self.tainted = True
            gc.collect()
            if using_cuda:
                torch.cuda.empty_cache()
            return 0.0
        # Pure recurrent caches have no KV sequence axis. Generation owns the
        # number of committed inputs; a state tensor's shape cannot supply it.
        past_len = _cache_seq_len(past_key_values) if consumed_tokens is None else int(consumed_tokens)
        if not 0 <= past_len <= len(all_ids):
            raise ValueError("consumed_tokens must index the committed input prefix")
        if past_len <= 0 and not need_bridge:
            del past_key_values
            self.reset()
            self.tainted = True
            gc.collect()
            if using_cuda:
                torch.cuda.empty_cache()
            return 0.0
        if need_bridge:
            if past_len < len(all_ids):
                unfed = all_ids[past_len:]
                with torch.no_grad():
                    st = torch.tensor([unfed], dtype=torch.long)
                    _, past_key_values = forward_step_all_logits(
                        engine, st, past_key_values=past_key_values, prefetch_next=False
                    )
                    last_h = captured["post_norm"][:, -1:, :].clone()
            else:
                last_h = last_fed_h_post
            self.fed = all_ids
            self.unfed_tail = []
        else:
            last_h = None
            self.fed = all_ids[:past_len]
            self.unfed_tail = all_ids[past_len:]

        ram_budget = _streaming_ram_budget_bytes()
        from .kv_offload import empty_host_pinned_cache, release_kv_cache_gpu_buffers
        from .runtime_config import _available_host_ram_bytes
        est_kv_bytes = sum(
            (int(lc.keys.nbytes + lc.values.nbytes) if getattr(lc, "keys", None) is not None else 0)
            for lc in getattr(past_key_values, "layers", ())
        )
        avail_ram_b = _available_host_ram_bytes()
        allow_pin = bool(
            using_cuda
            and (ram_budget is None or ram_budget >= 512 * MiB)
            and est_kv_bytes <= 1024 * MiB
            and (avail_ram_b <= 0 or (avail_ram_b - est_kv_bytes) >= int(1.5 * GiB))
        )
        self._pending_bridge_h = _detach_park_cpu(last_h, allow_pin) if last_h is not None else None

        t0 = time.perf_counter()
        release_kv_cache_gpu_buffers(past_key_values)
        if using_cuda:
            torch.cuda.synchronize(execution_device)
        self.parked_backbone = {}
        total_bytes = 0
        for l_idx, l_cache in enumerate(past_key_values.layers):
            entry: dict[str, Any] = {}
            bshd_pair = l_cache.export_bshd_slices() if hasattr(l_cache, "export_bshd_slices") else None
            if bshd_pair is not None:
                entry["keys"] = _detach_park_cpu(bshd_pair[0], allow_pin)
                entry["values"] = _detach_park_cpu(bshd_pair[1], allow_pin)
                entry["layout"] = "bshd"
                total_bytes += entry["keys"].nbytes + entry["values"].nbytes
                l_cache.reset()
            elif getattr(l_cache, "keys", None) is not None and l_cache.keys.numel() > 0:
                entry["keys"] = _detach_park_cpu(l_cache.keys, allow_pin)
                entry["values"] = _detach_park_cpu(l_cache.values, allow_pin)
                total_bytes += entry["keys"].nbytes + entry["values"].nbytes
                if getattr(l_cache, "is_quintessentia_pinned_layer", False) or getattr(
                    l_cache, "is_quintessentia_prealloc_layer", False
                ):
                    l_cache.reset()
            # Sliding KV tensors only retain a window, not the absolute position.
            if hasattr(l_cache, "cumulative_length"):
                entry["cumulative_length"] = int(l_cache.cumulative_length)
            if hasattr(l_cache, "conv_states"):
                states = {}
                for state_idx in l_cache.has_previous_state:
                    state = {"has_previous_state": bool(l_cache.has_previous_state[state_idx])}
                    for name, flags in (("conv_states", "is_conv_states_initialized"),
                                        ("recurrent_states", "is_recurrent_states_initialized")):
                        tensor = getattr(l_cache, name).get(state_idx)
                        if getattr(l_cache, flags).get(state_idx, False) and tensor is not None:
                            state[name] = _detach_park_cpu(tensor, allow_pin)
                            total_bytes += tensor.nbytes
                    if "conv_states" in state:
                        state["conv_kernel_size"] = l_cache.conv_kernel_size[state_idx]
                    states[state_idx] = state
                entry["linear_states"] = states
            self.parked_backbone[l_idx] = entry

        if mtp_head is not None and mtp_head.kv_cache.get_seq_length() > 0:
            mtp_l0 = mtp_head.kv_cache.layers[0]
            self.parked_mtp = {
                "keys": _detach_park_cpu(mtp_l0.keys, allow_pin),
                "values": _detach_park_cpu(mtp_l0.values, allow_pin),
            }
            total_bytes += self.parked_mtp["keys"].nbytes + self.parked_mtp["values"].nbytes
            self.parked_mtp_committed_len = mtp_head.committed_len
            mtp_head.reset_cache()
        else:
            self.parked_mtp = None
            self.parked_mtp_committed_len = 0

        del past_key_values
        gc.collect()
        if using_cuda:
            torch.cuda.empty_cache()
            from .kv_offload import empty_host_pinned_cache
            empty_host_pinned_cache()
            torch.cuda.synchronize(execution_device)
        self.park_ms = (time.perf_counter() - t0) * 1000.0
        self.parked_bytes = total_bytes
        return self.park_ms

    def can_truncate_parked_prefix(self) -> bool:
        """True iff parked KV layers are pure full-attention tensors that can be truncated to an earlier prefix.

        Recurrent layers (`linear_states`, e.g. GatedDeltaNet/Mamba), sliding-window caches
        (`cumulative_length`), and bridged MTP caches cannot be rewound cleanly without full replay.
        """
        if self.tainted or self.parked_backbone is None or self.parked_mtp is not None:
            return False
        if not self.parked_backbone:
            return False
        for entry in self.parked_backbone.values():
            if entry.get("linear_states") or ("cumulative_length" in entry):
                return False
        return True

    def truncate_parked_prefix(self, new_len: int) -> bool:
        """Trim parked pure-attention KV tensors in-place to `new_len` tokens."""
        n = int(new_len)
        if not self.can_truncate_parked_prefix() or n <= 0 or n > len(self.fed):
            return False
        if n == len(self.fed):
            self.unfed_tail = []
            return True
        total_bytes = 0
        assert self.parked_backbone is not None
        for entry in self.parked_backbone.values():
            if "keys" in entry and "values" in entry:
                is_bshd = entry.get("layout") == "bshd"
                k_t = entry["keys"]
                v_t = entry["values"]
                seq_dim = 1 if is_bshd else 2
                if k_t.shape[seq_dim] < n or v_t.shape[seq_dim] < n:
                    return False
                if is_bshd:
                    entry["keys"] = k_t[:, :n, :, :].contiguous()
                    entry["values"] = v_t[:, :n, :, :].contiguous()
                else:
                    entry["keys"] = k_t[:, :, :n, :].contiguous()
                    entry["values"] = v_t[:, :, :n, :].contiguous()
                total_bytes += entry["keys"].nbytes + entry["values"].nbytes
        self.fed = list(self.fed[:n])
        self.unfed_tail = []
        self.parked_bytes = total_bytes
        return True

    def kv_prefix_reuse(self, new_prompt_ids: list[int], *, allow_truncate: bool = True) -> int:
        L = len(self.fed)
        if self.tainted or L <= 0 or self.parked_backbone is None:
            return 0
        if L < len(new_prompt_ids) and new_prompt_ids[:L] == self.fed:
            return L
        if allow_truncate and self.can_truncate_parked_prefix():
            # Find exact longest common token prefix, keeping >= 1 new token for prefill
            limit = min(L, len(new_prompt_ids) - 1)
            common = 0
            while common < limit and new_prompt_ids[common] == self.fed[common]:
                common += 1
            if common > 0 and self.truncate_parked_prefix(common):
                return common
        return 0

    def unpark_to_gpu(
        self,
        device: torch.device,
        mtp_head: QuintessentiaNativeMTPHead | None,
        first_new_token: int,
        *,
        offload_kv: bool = False,
        offload_layer_count: int | None = None,
        disable_prefetch: bool = False,
        initial_capacity: int = 0,
        chunk_pad: int = 512,
    ) -> tuple[Any, float]:
        from transformers.cache_utils import DynamicCache

        t0 = time.perf_counter()
        tcfg = self.config.get_text_config() if hasattr(self.config, "get_text_config") else self.config
        if offload_kv and device.type == "cuda":
            from .kv_offload import make_offloaded_kv_cache
            past = make_offloaded_kv_cache(
                tcfg,
                initial_capacity=max(len(self.fed) + chunk_pad, int(initial_capacity)),
                chunk_pad=chunk_pad,
                device=device,
                disable_prefetch=disable_prefetch,
                offload_layer_count=offload_layer_count,
            )
        else:
            past = DynamicCache(config=tcfg)
        parked_map = self.parked_backbone or {}
        self.parked_backbone = None
        self.parked_bytes = 0
        for l_idx in list(parked_map.keys()):
            entry = parked_map.pop(l_idx)
            if l_idx < len(past.layers):
                l_cache = past.layers[l_idx]
                if "keys" in entry:
                    is_bshd = entry.get("layout") == "bshd"
                    if hasattr(l_cache, "seed_from_cpu_tensors"):
                        l_cache.seed_from_cpu_tensors(entry["keys"], entry["values"], device, is_bshd=is_bshd)
                    else:
                        k_src = entry["keys"].transpose(1, 2).contiguous() if is_bshd else entry["keys"]
                        v_src = entry["values"].transpose(1, 2).contiguous() if is_bshd else entry["values"]
                        k_gpu = k_src.to(device, non_blocking=True)
                        v_gpu = v_src.to(device, non_blocking=True)
                        l_cache.update(k_gpu, v_gpu)
                    entry.pop("keys", None)
                    entry.pop("values", None)
                if "cumulative_length" in entry:
                    l_cache.cumulative_length = entry["cumulative_length"]
                for state_idx, state in entry.get("linear_states", {}).items():
                    c_gpu = state.get("conv_states")
                    r_gpu = state.get("recurrent_states")
                    c_gpu = c_gpu.to(device, non_blocking=True) if c_gpu is not None else None
                    r_gpu = r_gpu.to(device, non_blocking=True) if r_gpu is not None else None
                    l_cache.lazy_initialization(conv_states=c_gpu, recurrent_states=r_gpu,
                                                state_idx=state_idx,
                                                conv_kernel_size=state.get("conv_kernel_size"))
                    if c_gpu is not None:
                        dst_c = (
                            l_cache.conv_states.get(state_idx)
                            if isinstance(l_cache.conv_states, dict)
                            else l_cache.conv_states[state_idx]
                        )
                        if dst_c is None:
                            l_cache.conv_states[state_idx] = c_gpu.clone()
                        else:
                            dst_c.copy_(c_gpu)
                        if hasattr(l_cache, "is_conv_states_initialized"):
                            l_cache.is_conv_states_initialized[state_idx] = True
                        if hasattr(l_cache, "conv_kernel_size"):
                            cks = state.get("conv_kernel_size")
                            if cks is None and c_gpu.ndim >= 1:
                                cks = int(c_gpu.shape[-1])
                            if cks is not None:
                                l_cache.conv_kernel_size[state_idx] = int(cks)
                    if r_gpu is not None:
                        dst_r = (
                            l_cache.recurrent_states.get(state_idx)
                            if isinstance(l_cache.recurrent_states, dict)
                            else l_cache.recurrent_states[state_idx]
                        )
                        if dst_r is None:
                            l_cache.recurrent_states[state_idx] = r_gpu.clone()
                        else:
                            dst_r.copy_(r_gpu)
                        if hasattr(l_cache, "is_recurrent_states_initialized"):
                            l_cache.is_recurrent_states_initialized[state_idx] = True
                    l_cache.has_previous_state[state_idx] = state["has_previous_state"]
            elif "keys" in entry:
                k_gpu = entry["keys"].to(device, non_blocking=True)
                v_gpu = entry["values"].to(device, non_blocking=True)
                past.update(k_gpu, v_gpu, l_idx)

        if mtp_head is not None and self.parked_mtp is not None and self._pending_bridge_h is not None:
            mtp_head.reset_cache()
            mk = self.parked_mtp["keys"].to(device, non_blocking=True)
            mv = self.parked_mtp["values"].to(device, non_blocking=True)
            mtp_head.kv_cache.layers[0].update(mk, mv)
            mtp_head.committed_len = self.parked_mtp_committed_len
            with torch.no_grad():
                h_bridge = self._pending_bridge_h.to(device, non_blocking=True)
                mtp_head._forward_pairs(h_bridge, [first_new_token], start_pos=mtp_head.committed_len)
                mtp_head.committed_len += 1

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            from .kv_offload import empty_host_pinned_cache
            empty_host_pinned_cache()
        return past, (time.perf_counter() - t0) * 1000.0


class _IncrementalThinkSplitter:
    """UTF-8 & tag-boundary-safe splitter emitting `("thinking"|"answer", text_delta)`."""

    CLOSE_TAG = "</think>"

    def __init__(self, start_in_thinking: bool):
        self.phase = "thinking" if start_in_thinking else "answer"
        self.buf = ""
        self._strip_leading_newlines = False

    def push(self, new_text: str, final: bool = False) -> list[tuple[str, str]]:
        if not new_text and not final:
            return []
        self.buf += new_text
        out: list[tuple[str, str]] = []

        if self.phase == "thinking":
            idx = self.buf.find(self.CLOSE_TAG)
            if idx != -1:
                before = self.buf[:idx]
                if before:
                    out.append(("thinking", before))
                self.buf = self.buf[idx + len(self.CLOSE_TAG) :]
                self.phase = "answer"
                self._strip_leading_newlines = True
            else:
                if final:
                    if self.buf:
                        out.append(("thinking", self.buf))
                        self.buf = ""
                    return out
                # Hold back up to 7 chars if suffix matches a prefix of `</think>`
                hold = 0
                max_k = min(len(self.CLOSE_TAG) - 1, len(self.buf))
                for k in range(max_k, 0, -1):
                    if self.buf.endswith(self.CLOSE_TAG[:k]):
                        hold = k
                        break
                emit_n = len(self.buf) - hold
                if emit_n > 0:
                    out.append(("thinking", self.buf[:emit_n]))
                    self.buf = self.buf[emit_n:]
                return out

        if self.phase == "answer" and self.buf:
            if self._strip_leading_newlines:
                stripped = self.buf.lstrip("\r\n")
                if not stripped and not final:
                    self.buf = ""
                    return out
                self.buf = stripped
                self._strip_leading_newlines = False
            if self.buf:
                out.append(("answer", self.buf))
                self.buf = ""
        return out


MiB = 1024 ** 2
GiB = 1024 ** 3
def _align(value: int, alignment: int = 2 * MiB) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


def q4_bytes_for_shape(shape: tuple[int, ...], group_size: int = 64) -> int:
    """VRAM of one W4A16 matrix: packed nibbles + BF16 scales and zeros per group."""
    out_f, in_f = int(shape[0]), int(shape[1])
    return out_f * in_f // 2 + 2 * out_f * (in_f // group_size) * 2


def estimate_layer_vram_bytes(layer_shapes: dict[str, tuple[int, ...]], group_size: int = 64) -> int:
    """Resident VRAM of one decoder layer in the engine's Q4 format (from tensor shapes only)."""
    from src.quantization_q4 import is_q4_quantizable

    total = 0
    for name, shape in layer_shapes.items():
        short = name.split(".", 3)[-1] if name.startswith("model.layers.") else name
        if is_q4_quantizable(short, tuple(shape), group_size=group_size):
            total += q4_bytes_for_shape(shape, group_size)
        else:
            n = 1
            for s in shape:
                n *= int(s)
            total += 2 * n
    return total


def bf16_bytes_for_shape(shape: tuple[int, ...]) -> int:
    """Uncompressed BF16 storage of one tensor."""
    n = 1
    for s in shape:
        n *= int(s)
    return 2 * n


def state_bytes_from_config(cfg: dict[str, Any]) -> tuple[int, int, int, int]:
    """KV bytes per token, recurrent-state bytes, hidden size, intermediate size.

    Qwen3.5 counts only ``full_attention`` layers toward KV and the GatedDeltaNet
    layers toward a fixed recurrent state. Pure SSM/RWKV models allocate fixed recurrent
    state with zero KV growth, and MLA models account for QK/V head dimensions.
    """
    n = int(
        cfg.get("num_hidden_layers")
        or cfg.get("n_layers")
        or cfg.get("n_layer")
        or cfg.get("num_layers")
        or 1
    )
    hidden = int(cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model") or 1)
    n_heads = int(cfg.get("num_attention_heads") or cfg.get("n_head") or cfg.get("n_heads") or 1)
    n_kv = int(
        cfg.get("num_key_value_heads")
        or cfg.get("num_kv_heads")
        or (1 if cfg.get("multi_query") else n_heads)
    )
    head = int(cfg.get("head_dim") or max(1, hidden // max(1, n_heads)))
    layer_types = cfg.get("layer_types") or []
    mtype = str(cfg.get("model_type") or "").lower()
    if layer_types:
        full = sum(1 for t in layer_types if t in ("full_attention", "attention", "sliding_attention"))
        lin = max(0, n - full)
    elif mtype in ("mamba", "mamba2", "falcon_mamba", "rwkv"):
        full = 0
        lin = n
    else:
        full = n
        lin = 0
    if mtype == "minicpm3" and cfg.get("kv_lora_rank"):
        # MiniCPM3 caches compressed MLA latent (kv_lora_rank) + RoPE key (qk_rope_head_dim) as 1 head.
        kv = full * (int(cfg["kv_lora_rank"]) + int(cfg.get("qk_rope_head_dim") or 0)) * 2
    elif cfg.get("qk_nope_head_dim") is not None or cfg.get("v_head_dim") is not None:
        qk_d = int(cfg.get("qk_nope_head_dim") or head) + int(cfg.get("qk_rope_head_dim") or 0)
        v_d = int(cfg.get("v_head_dim") or head)
        kv = full * n_kv * (qk_d + v_d) * 2
    else:
        kv = full * 2 * n_kv * head * 2
    rec = 0
    if lin and cfg.get("linear_num_value_heads"):
        rec = lin * (
            int(cfg.get("linear_num_value_heads", 32)) * int(cfg.get("linear_key_head_dim", 128))
            * int(cfg.get("linear_value_head_dim", 128)) * 4
            + 2 * int(cfg.get("linear_num_key_heads", 16)) * int(cfg.get("linear_key_head_dim", 128)) * 4 * 2
        )
    elif lin and (cfg.get("state_size") or cfg.get("mamba_d_state") or cfg.get("ssm_state_size")):
        d_inner = int(cfg.get("intermediate_size") or hidden * int(cfg.get("expand") or 2))
        d_state = int(cfg.get("state_size") or cfg.get("mamba_d_state") or cfg.get("ssm_state_size") or 16)
        d_conv = int(cfg.get("conv_kernel") or cfg.get("mamba_d_conv") or 4)
        rec = lin * (d_inner * d_state * 4 + d_inner * d_conv * 2)
    elif mtype == "rwkv":
        rec = n * hidden * 5 * 4
    moe_each = int(cfg.get("moe_intermediate_size") or 0)
    top_k = max(1, int(cfg.get("num_experts_per_tok") or cfg.get("num_selected_experts") or 1))
    moe_inter = moe_each * top_k
    moe_inter += int(
        cfg.get("shared_expert_intermediate_size")
        or (moe_each * int(cfg.get("n_shared_experts") or 0))
    )
    ffn_cfg = cfg.get("ffn_config") if isinstance(cfg.get("ffn_config"), dict) else {}
    inter = int(
        cfg.get("intermediate_size")
        or cfg.get("ffn_dim")
        or cfg.get("n_inner")
        or ffn_cfg.get("ffn_hidden_size")
        or moe_inter
        or hidden * 4
    )
    return kv, rec, hidden, inter


@dataclass
class VRAMPlan:
    """How the engine spends the VRAM budget for one run."""

    mode: str
    total_bytes: int
    free_bytes: int
    reserve_bytes: int
    budget_bytes: int
    layers_total: int
    resident_layers: int
    ring_slots: int
    slot_bytes: int
    full_residency: bool
    enable_mtp: bool
    prefill_chunk_tokens: int
    max_context_tokens: int
    weights_bytes: int
    sub_layer_cubes: bool = False
    streamed_lm_head: bool = False

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("total_bytes", "free_bytes", "reserve_bytes", "budget_bytes", "slot_bytes", "weights_bytes"):
            d[k.replace("_bytes", "_mib")] = round(d.pop(k) / MiB, 1)
        return d


def plan_vram(
    mode: str,
    total_bytes: int,
    free_bytes: int,
    layer_bytes: list[int],
    lm_head_bytes: int,
    mtp_bytes: int,
    kv_bytes_per_token: int,
    recurrent_state_bytes: int,
    hidden_size: int,
    intermediate_size: int,
    max_context_tokens: int | None = None,
    prefill_chunk_tokens: int | None = None,
    ring_slots: int = 3,
    prefill_wavefront: bool = False,
    allow_cube_subdivision: bool = False,
    max_slot_bytes: int | None = None,
    min_cube_slot_bytes: int = 2 * MiB,
    allow_single_slot: bool = False,
) -> VRAMPlan:
    """Splits the *available* VRAM between weights and a guaranteed working reserve.

    ``normal`` takes almost all free VRAM but keeps a safety margin for the OS/WDDM, the KV and
    GatedDeltaNet states, the chunked prefill workspace and CUDA library workspaces, so a long
    prompt cannot run the card out of memory. ``turbo`` keeps only a minimal margin.

    When the weights do not fit, the remaining budget is cut into ``ring_slots`` equal slots that
    rotate roles (computing -> clean -> receiving the next weights) so the next layer is already
    in VRAM when the current one finishes; as many layers as fit stay resident. When
    ``allow_cube_subdivision=True`` (or ``max_slot_bytes`` caps per-slot VRAM below a full layer
    or ``lm_head``), layers and/or ``lm_head`` subdivide into 2D Q1 cubes along ``A -> B -> C(0) -> A``.
    """
    if mode not in VRAM_MODES:
        raise ValueError(
            f"vram_mode must be one of {VRAM_MODES} (got {mode!r}); fixed `4gb`/`8gb` profiles were removed"
        )
    if max_slot_bytes is not None and int(max_slot_bytes) <= 0:
        raise ValueError("max_slot_bytes must be positive")
    low_vram = total_bytes <= 5 * GiB
    chunk = int(prefill_chunk_tokens or (256 if low_vram else 512))
    max_ctx = int(max_context_tokens or (4096 if low_vram else 8192))
    context_bytes = kv_bytes_per_token * max_ctx + 2 * recurrent_state_bytes
    prefill_bytes = _align(int(chunk * (4 * hidden_size + 3 * intermediate_size) * 2 * 1.5))
    if prefill_wavefront:
        # Three simultaneous layer workspaces plus four chunks of activations.
        prefill_bytes = 3 * prefill_bytes + _align(4 * chunk * hidden_size * 2) + 96 * MiB
    logits_bytes = 4 * MiB
    if mode == "normal":
        guard = _align(max(160 * MiB, int(0.045 * total_bytes))) if low_vram else _align(max(384 * MiB, int(0.055 * total_bytes)))
        slack = 256 * MiB
    else:
        guard = 84 * MiB if low_vram else 100 * MiB
        slack = 128 * MiB
    reserve = guard + slack + _align(context_bytes) + prefill_bytes + logits_bytes
    budget = max(0, int(free_bytes) - reserve)

    n = len(layer_bytes)
    max_layer = max(layer_bytes) if layer_bytes else 0
    explicit_sub_layer = max_slot_bytes is not None and max_layer > 0 and int(max_slot_bytes) < max_layer
    explicit_sub_head = max_slot_bytes is not None and lm_head_bytes > 0 and int(max_slot_bytes) < lm_head_bytes

    weights_all = sum(layer_bytes) + lm_head_bytes
    if not explicit_sub_layer and not explicit_sub_head and budget >= weights_all + mtp_bytes:
        return VRAMPlan(mode, total_bytes, free_bytes, reserve, budget, n, n, 0, 0, True, mtp_bytes > 0,
                        chunk, max_ctx, weights_all + mtp_bytes, False, False)

    min_slots = 1 if allow_single_slot else 3
    ring_slots = max(min_slots, min(4, int(ring_slots)))
    if explicit_sub_layer or explicit_sub_head:
        if not allow_cube_subdivision:
            raise RuntimeError(
                f"Not enough per-slot VRAM ({int(max_slot_bytes) / MiB:.1f} MiB) without Q1 cube subdivision"
            )
        cap = int(max_slot_bytes)
        slot = min(max_layer, cap) if max_layer > 0 else cap
        stream_head = bool(explicit_sub_head)
        eff_head = 0 if stream_head else lm_head_bytes
        if budget < eff_head + ring_slots * slot:
            if not stream_head and budget >= ring_slots * min(slot, max(1, budget // ring_slots)) and budget >= ring_slots * min(cap, min_cube_slot_bytes):
                stream_head = True
                eff_head = 0
                slot = min(slot, budget // ring_slots)
            elif stream_head and budget >= ring_slots * min(cap, min_cube_slot_bytes):
                slot = min(slot, budget // ring_slots)
            else:
                raise RuntimeError(
                    f"Not enough free VRAM for mode={mode}: budget {budget / MiB:.0f} MiB < "
                    f"{(eff_head + ring_slots * slot) / MiB:.0f} MiB"
                )
        weights = eff_head + ring_slots * slot
        return VRAMPlan(mode, total_bytes, free_bytes, reserve, budget, n, 0, ring_slots, slot, False,
                        False, chunk, max_ctx, weights,
                        sub_layer_cubes=bool(explicit_sub_layer or (max_layer > 0 and slot < max_layer)),
                        streamed_lm_head=stream_head)

    slot = max_layer
    base = lm_head_bytes + ring_slots * slot
    if budget < base:
        if not allow_cube_subdivision:
            raise RuntimeError(
                f"Not enough free VRAM for mode={mode}: budget {budget / MiB:.0f} MiB < "
                f"{base / MiB:.0f} MiB (LM head + {ring_slots} ring slots). Close other GPU programs or use --vram-mode turbo."
            )
        min_ring = ring_slots * int(min_cube_slot_bytes)
        if budget < min_ring:
            raise RuntimeError(
                f"Not enough free VRAM for mode={mode}: budget {budget / MiB:.0f} MiB < "
                f"{min_ring / MiB:.0f} MiB minimum Q1 cube streaming ring. Close other GPU programs or use --device cpu."
            )
        if budget >= lm_head_bytes + min_ring:
            slot_cap = (budget - lm_head_bytes) // ring_slots
            if max_slot_bytes is not None:
                slot_cap = min(slot_cap, int(max_slot_bytes))
            sub_slot = max(int(min_cube_slot_bytes), min(max_layer, slot_cap)) if max_layer > 0 else slot_cap
            weights = lm_head_bytes + ring_slots * sub_slot
            return VRAMPlan(mode, total_bytes, free_bytes, reserve, budget, n, 0, ring_slots, sub_slot, False,
                            False, chunk, max_ctx, weights, sub_layer_cubes=True, streamed_lm_head=False)
        slot_cap = budget // ring_slots
        if max_slot_bytes is not None:
            slot_cap = min(slot_cap, int(max_slot_bytes))
        sub_slot = max(int(min_cube_slot_bytes), slot_cap)
        weights = ring_slots * sub_slot
        return VRAMPlan(mode, total_bytes, free_bytes, reserve, budget, n, 0, ring_slots, sub_slot, False,
                        False, chunk, max_ctx, weights, sub_layer_cubes=True, streamed_lm_head=True)

    enable_mtp = mtp_bytes > 0 and budget - base - mtp_bytes >= min(layer_bytes or [0]) * 4
    room = budget - base - (mtp_bytes if enable_mtp else 0)
    resident = 0
    used = 0
    # The scheduler may select non-adjacent priority anchors. Budget the largest
    # layers so any selected resident set of this size remains within the limit.
    for b in sorted(layer_bytes, reverse=True):
        if used + b > room:
            break
        used += b
        resident += 1
    resident = min(resident, n - 1)
    weights = lm_head_bytes + used + ring_slots * slot + (mtp_bytes if enable_mtp else 0)
    return VRAMPlan(mode, total_bytes, free_bytes, reserve, budget, n, resident, ring_slots, slot, False,
                    enable_mtp, chunk, max_ctx, weights, False, False)


def _model_shape_inventory(model_dir: Path, source: Any, num_layers: int) -> tuple[list[dict[str, tuple]], dict[str, tuple], dict[str, tuple]]:
    """Reads tensor shapes (headers only) of every layer, the globals and the MTP head."""
    from .model_source import read_safetensors_header

    layers: list[dict[str, tuple]] = []
    if source is not None:
        for i in range(num_layers):
            layers.append({k: source.tensor_shape(e) for k, e in source.layer_entries(i).items()})
        globals_ = {k: source.tensor_shape(e) for k, e in source.globals_entries().items()}
        mtp = {k: source.tensor_shape(e) for k, e in source.mtp_entries().items()}
        return layers, globals_, mtp

    def _shapes(path: Path) -> dict[str, tuple]:
        if not path.is_file():
            return {}
        hdr, _ = read_safetensors_header(path)
        return {k: tuple(v["shape"]) for k, v in hdr.items() if k != "__metadata__"}

    for i in range(num_layers):
        layers.append(_shapes(model_dir / f"model-{i:05d}.safetensors"))
    return layers, _shapes(model_dir / "model-globals.safetensors"), _shapes(model_dir / "model-mtp.safetensors")


SUPPORTED_RUNTIME_WEIGHT_FORMATS = ("auto", "q1_cubes", "q1", "q2", "q4", "q6", "q8", "bf16")


def _q1_lm_head_telemetry(model: Any) -> dict[str, Any] | None:
    if model is None:
        return None
    out_emb = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else getattr(model, "lm_head", None)
    if out_emb is None or (type(out_emb).__name__ != "IdealQ1LMHead" and not hasattr(out_emb, "vocab_rows_pruned_pct")):
        return None
    raw = out_emb.stats() if hasattr(out_emb, "stats") and callable(out_emb.stats) else {}
    planes = int(raw.get("q1_planes", getattr(out_emb, "q1_planes", getattr(out_emb, "planes", 1))))
    eff = float(raw.get("effective_bits", raw.get("avg_effective_bits", float(planes))))
    pruned = float(raw.get("vocab_rows_pruned_pct", 0.0))
    early = bool(raw.get("early_stop_enabled", getattr(out_emb, "early_stop", False)))
    return {
        **raw,
        "effective_bits": round(eff, 4),
        "vocab_rows_pruned_pct": round(pruned, 2),
        "early_stop_enabled": early,
        "q1_planes": planes,
    }


def _q1_cubes_storage_bytes(
    layers: list[dict[str, tuple]],
    globals_sh: dict[str, tuple],
    cfg: dict[str, Any],
    cfg_hidden: int,
    q1_p: int,
    lm_head_planes: int | None = None,
) -> tuple[list[int], int, int, int]:
    """Compute exact Binary Halving Q1 storage bytes per layer and for `lm_head` at plane depth `q1_p`."""
    layer_bytes: list[int] = []
    packed_count = 0
    for shapes in layers:
        size = 0
        for name, shape in shapes.items():
            sh = tuple(shape)
            lname = str(name).lower()
            is_router = (
                len(sh) == 2
                and int(sh[0]) <= 512
                and (
                    lname.endswith(".gate.weight")
                    or lname.endswith(".router.weight")
                    or lname.endswith(".router.layer.weight")
                    or ".router." in lname
                )
            )
            if len(sh) == 2 and lname.endswith(".weight") and not is_router and int(sh[1]) >= 64 and int(sh[1]) % 64 == 0:
                n_out, k_in = int(sh[0]), int(sh[1])
                # Shared qscale[0] Binary Halving Q1 storage: packed_lo + qscale[0] + qzero + Sinkhorn row/col scales
                size += int(q1_p) * n_out * (k_in // 8) + 2 * n_out * (k_in // 64) * 2 + 2 * (n_out + k_in)
                packed_count += 1
            else:
                size += bf16_bytes_for_shape(sh)
        layer_bytes.append(size)

    shape = tuple(globals_sh.get("lm_head.weight", (int(cfg.get("vocab_size") or 1), cfg_hidden)))
    bf16_lm_b = bf16_bytes_for_shape(shape)
    lm_root = 16 if int(q1_p) > 8 else 8
    if lm_head_planes is not None:
        lm_cubes = max(1, min(lm_root, int(lm_head_planes)))
    else:
        lm_cubes = max(1, min(lm_root, max(5, int(q1_p))))
    if len(shape) == 2 and bf16_lm_b >= 256 * MiB and int(shape[1]) >= 64 and int(shape[1]) % 64 == 0:
        v_sz, h_sz = int(shape[0]), int(shape[1])
        lm_head_bytes = lm_cubes * v_sz * (h_sz // 8) + 2 * v_sz * (h_sz // 64) * 2 + 2 * (v_sz + h_sz)
    else:
        lm_head_bytes = bf16_lm_b
    return layer_bytes, packed_count, int(lm_head_bytes), int(lm_cubes)


def plan_for_checkpoint(
    model_dir: str | Path,
    *,
    mode: str,
    total_bytes: int,
    free_bytes: int,
    enable_mtp: bool = True,
    max_context_tokens: int | None = None,
    prefill_chunk_tokens: int | None = None,
    weight_format: str = "auto",
    prefill_wavefront: bool = False,
    force_universal: bool = False,
    projection_metadata: dict | None = None,
    allow_cube_subdivision: bool = True,
    max_slot_bytes: int | None = None,
    q1_planes: int | None = None,
    auto_fit_q1_planes: bool = False,
    min_q1_planes: int = 3,
    max_q1_planes: int = 8,
    lm_head_planes: int | None = None,
) -> tuple[dict[str, Any], VRAMPlan, bool]:
    """Plan from headers and meta modules, including the actual packed projections.

    Native MTP remains guarded by the specialized Qwen geometry. Generic Q1/Q2/Q4/Q6/Q8
    uses architecture modules and bounded dense/expert kernels.
    """
    from .model_source import open_model_source, raw_text_config_dict, runtime_profile

    folder = Path(model_dir)
    profile = runtime_profile(folder)
    if prefill_wavefront and profile.get("model_type") != "llama":
        raise ValueError("Experimental wavefront prefill currently supports Llama models only")
    if weight_format not in SUPPORTED_RUNTIME_WEIGHT_FORMATS:
        raise ValueError(f"weight_format must be one of {SUPPORTED_RUNTIME_WEIGHT_FORMATS}")
    q1_p = int(q1_planes if q1_planes is not None else (16 if weight_format == "q1_cubes" else 4))
    if not 1 <= q1_p <= (16 if weight_format == "q1_cubes" else 8):
        raise ValueError("q1_planes exceeds the representation depth (q1_cubes supports 16)")
    profile = dict(profile)
    fast = bool(profile["fast_qwen"]) and weight_format in ("auto", "q4") and not force_universal
    selected_format = ("q4" if fast else "q6") if weight_format == "auto" else weight_format
    profile.update(
        fast_qwen=fast,
        weight_format=selected_format,
        backend=(
            "qwen35-q4"
            if fast
            else "unified-q1-cubes"
            if selected_format == "q1_cubes"
            else f"universal-{selected_format}"
        ),
    )
    if selected_format in ("q1", "q1_cubes"):
        profile["q1_planes"] = q1_p
    cfg = raw_text_config_dict(folder)
    num_layers = int(
        cfg.get("num_hidden_layers")
        or cfg.get("n_layers")
        or cfg.get("n_layer")
        or cfg.get("num_layers")
    )
    cfg_hidden = int(cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model"))
    tmp_source = open_model_source(folder)
    try:
        layers, globals_sh, mtp_sh = _model_shape_inventory(folder, tmp_source, num_layers)
    finally:
        if tmp_source is not None and hasattr(tmp_source, "close"):
            tmp_source.close()
    kv, rec, hidden_size, intermediate = state_bytes_from_config(cfg)
    fast = bool(profile["fast_qwen"])
    if fast:
        layer_bytes = [estimate_layer_vram_bytes(sh) for sh in layers]
        vocab, hidden = globals_sh.get("lm_head.weight", (cfg.get("vocab_size", 0), cfg_hidden))
        lm_head_bytes = q4_bytes_for_shape((int(vocab), int(hidden)))
        mtp_bytes = int(estimate_layer_vram_bytes({k: v for k, v in mtp_sh.items() if str(k).startswith("mtp.layers.")}))
        if "mtp.fc.weight" in mtp_sh:
            mtp_bytes += q4_bytes_for_shape(tuple(mtp_sh["mtp.fc.weight"]))
        has_mtp = "mtp.fc.weight" in mtp_sh
    elif selected_format == "q1_cubes":
        q1_p_req = int(q1_p)
        lm_req = int(lm_head_planes) if lm_head_planes is not None else None
        k_ceil = max(q1_p_req, min(16, int(max_q1_planes)))
        layer_bytes, packed_count, lm_head_bytes, lm_cubes = _q1_cubes_storage_bytes(
            layers, globals_sh, cfg, cfg_hidden, q1_p_req, lm_head_planes=lm_req
        )
        # Adaptive Binary Halving under VRAM constraints:
        # When the full model does not fit at `q1_p_req`, step down strictly by whole halving
        # planes (`k -> k-1 -> ... -> min_q1_planes`) so every layer stays resident at peak speed
        # instead of falling back to slow PCIe layer streaming.
        if auto_fit_q1_planes and q1_p_req > 1 and max_slot_bytes is None and free_bytes > 0:
            probe_plan = plan_vram(
                mode,
                total_bytes=total_bytes,
                free_bytes=free_bytes,
                layer_bytes=[0] * len(layers),
                lm_head_bytes=0,
                mtp_bytes=0,
                kv_bytes_per_token=kv,
                recurrent_state_bytes=rec,
                hidden_size=hidden_size,
                intermediate_size=intermediate,
                max_context_tokens=max_context_tokens,
                prefill_chunk_tokens=prefill_chunk_tokens,
                prefill_wavefront=False,
                allow_cube_subdivision=True,
                max_slot_bytes=None,
            )
            avail_budget = int(probe_plan.budget_bytes)
            is_moe_model = int(profile.get("num_experts") or 0) >= 2
            if is_moe_model:
                moe_reserve = 512 * MiB if avail_budget >= 768 * MiB else int(avail_budget * 0.35)
                target_budget = max(0, avail_budget - moe_reserve)
            else:
                target_budget = avail_budget

            if avail_budget > 0 and (sum(layer_bytes) + lm_head_bytes) > target_budget:
                k_floor = max(1, min(q1_p_req, int(min_q1_planes)))
                chosen = None
                for k_cand in range(q1_p_req - 1, k_floor - 1, -1):
                    # First try preserving requested/high-precision lm_head, then halve lm_head to match k_cand.
                    if lm_req is not None:
                        lm_candidates = [lm_req, min(lm_req, k_cand)] if k_cand < lm_req else [lm_req]
                    else:
                        lm_candidates = [max(5, k_cand), k_cand] if q1_p_req >= 5 and k_cand < 5 else [k_cand]
                    for lm_cand in lm_candidates:
                        c_layers, c_packed, c_lm_b, c_lm_k = _q1_cubes_storage_bytes(
                            layers, globals_sh, cfg, cfg_hidden, k_cand, lm_head_planes=lm_cand
                        )
                        if (sum(c_layers) + c_lm_b) <= target_budget:
                            chosen = (k_cand, c_layers, c_packed, c_lm_b, c_lm_k)
                            break
                    if chosen is not None:
                        break
                if chosen is not None:
                    q1_p, layer_bytes, packed_count, lm_head_bytes, lm_cubes = chosen
                elif not is_moe_model and k_floor < q1_p_req:
                    # Even at k_floor the dense model exceeds VRAM; still halve to k_floor so
                    # maximum layers remain resident and streamed layers transfer faster over PCIe.
                    q1_p = k_floor
                    layer_bytes, packed_count, lm_head_bytes, lm_cubes = _q1_cubes_storage_bytes(
                        layers, globals_sh, cfg, cfg_hidden, k_floor, lm_head_planes=(lm_req if lm_req is not None else k_floor)
                    )
            elif avail_budget > 0 and not is_moe_model and q1_p_req < k_ceil:
                # Auto depth refinement (step-up): when a compact dense checkpoint fits
                # comfortably inside <= 35% of the available VRAM budget even at higher bit-plane
                # depth (up to `max_q1_planes`, default 8), step up automatically so outlier-heavy
                # small models keep BF16-level fidelity while leaving >= 65% VRAM free.
                refine_cap = int(target_budget * 0.35)
                for k_cand in range(k_ceil, q1_p_req, -1):
                    c_layers, c_packed, c_lm_b, c_lm_k = _q1_cubes_storage_bytes(
                        layers, globals_sh, cfg, cfg_hidden, k_cand, lm_head_planes=(lm_req if lm_req is not None else k_cand)
                    )
                    if (sum(c_layers) + c_lm_b) <= refine_cap:
                        q1_p, layer_bytes, packed_count, lm_head_bytes, lm_cubes = (
                            k_cand, c_layers, c_packed, c_lm_b, c_lm_k
                        )
                        break
        profile["q1_planes"] = int(q1_p)
        profile["q1_planes_requested"] = int(q1_p_req)
        profile["q1_planes_auto_halved"] = bool(int(q1_p) < int(q1_p_req))
        profile["q1_planes_auto_refined"] = bool(int(q1_p) > int(q1_p_req))
        profile["q1_lm_head_planes"] = int(lm_cubes)
        profile["packed_projections"] = packed_count
        profile["q1_cubes_layer_bytes"] = list(layer_bytes)
        profile["q1_cubes_lm_head_bytes"] = int(lm_head_bytes)
        mtp_bytes = 0
        has_mtp = "mtp.fc.weight" in mtp_sh
    else:
        quant_map = {}
        sel_bits = 0
        if selected_format != "bf16":
            sel_bits = int(selected_format[1:])
            from accelerate import init_empty_weights
            from transformers import AutoModelForCausalLM
            from .model_source import load_runtime_config
            from src.quantized_projection import projection_map, packed_specs
            if projection_metadata is not None:
                quant_map = projection_metadata
            else:
                with init_empty_weights():
                    probe_model = AutoModelForCausalLM.from_config(load_runtime_config(folder), dtype=torch.bfloat16)
                quant_map = projection_map(probe_model, layers, sel_bits, q1_planes=q1_p)
                del probe_model
        layer_bytes = []
        for idx, shapes in enumerate(layers):
            size = 0
            for name, shape in shapes.items():
                key = name.removeprefix(f"model.layers.{idx}.")
                spec = quant_map.get(idx, {}).get(key)
                if spec is None:
                    size += bf16_bytes_for_shape(tuple(shape))
                else:
                    size += sum(
                        math.prod(sh) * torch.empty((), dtype=dt).element_size()
                        for sh, dt in packed_specs(key, spec["shape"], sel_bits, q1_planes=q1_p).values()
                    )
            layer_bytes.append(size)
        profile["packed_projections"] = sum(len(v) for v in quant_map.values())
        shape = tuple(globals_sh.get("lm_head.weight", (int(cfg.get("vocab_size") or 1), cfg_hidden)))
        if selected_format == "q1" and len(shape) == 2 and shape[1] >= 16 and shape[1] % 16 == 0:
            v_sz, h_sz = int(shape[0]), int(shape[1])
            gs = 64 if h_sz % 64 == 0 else (32 if h_sz % 32 == 0 else 16)
            lm_head_bytes = sum(
                math.prod(sh) * torch.empty((), dtype=dt).element_size()
                for sh, dt in packed_specs("lm_head.weight", (v_sz, h_sz), 1, gs, q1_planes=q1_p).values()
            )
        else:
            lm_head_bytes = bf16_bytes_for_shape(shape)
        mtp_bytes = 0
        has_mtp = "mtp.fc.weight" in mtp_sh
    eff_allow_cube_sub = bool(allow_cube_subdivision and not prefill_wavefront and selected_format != "q1_cubes")
    try:
        plan = plan_vram(
            mode,
            total_bytes=total_bytes,
            free_bytes=free_bytes,
            layer_bytes=layer_bytes,
            lm_head_bytes=lm_head_bytes,
            mtp_bytes=mtp_bytes if (enable_mtp and has_mtp and fast) else 0,
            kv_bytes_per_token=kv,
            recurrent_state_bytes=rec,
            hidden_size=hidden_size,
            intermediate_size=intermediate,
            max_context_tokens=max_context_tokens,
            prefill_chunk_tokens=prefill_chunk_tokens,
            ring_slots=1 if selected_format == "q1_cubes" else 3,
            prefill_wavefront=prefill_wavefront,
            allow_cube_subdivision=eff_allow_cube_sub,
            max_slot_bytes=max_slot_bytes,
            allow_single_slot=(selected_format == "q1_cubes"),
        )
    except RuntimeError:
        if selected_format != "q1_cubes":
            raise
        # Binary Halving Q1 Cube mode streams strictly 1 active layer at a time (A->B->C(0)->A, 0 B standby),
        # so it succeeds whenever free VRAM fits lm_head + 1 active layer even if a 3-slot ring would not.
        max_layer = max(layer_bytes) if layer_bytes else 0
        probe_plan = plan_vram(
            mode,
            total_bytes=total_bytes,
            free_bytes=free_bytes,
            layer_bytes=[0] * len(layer_bytes),
            lm_head_bytes=0,
            mtp_bytes=0,
            kv_bytes_per_token=kv,
            recurrent_state_bytes=rec,
            hidden_size=hidden_size,
            intermediate_size=intermediate,
            max_context_tokens=max_context_tokens,
            prefill_chunk_tokens=prefill_chunk_tokens,
            prefill_wavefront=False,
            allow_cube_subdivision=True,
            max_slot_bytes=None,
        )
        if probe_plan.budget_bytes < lm_head_bytes + max_layer:
            raise
        plan = VRAMPlan(
            mode,
            total_bytes,
            free_bytes,
            probe_plan.reserve_bytes,
            probe_plan.budget_bytes,
            len(layer_bytes),
            0,
            1,
            max_layer,
            False,
            False,
            probe_plan.prefill_chunk_tokens,
            probe_plan.max_context_tokens,
            lm_head_bytes + max_layer,
            False,
            False,
        )
    profile = dict(profile)
    profile["mtp_weights"] = has_mtp
    profile["mtp_enabled"] = bool(has_mtp and enable_mtp and (plan.enable_mtp if fast else True))
    profile["sub_layer_cubes"] = bool(plan.sub_layer_cubes)
    profile["streamed_lm_head"] = bool(plan.streamed_lm_head)
    return profile, plan, has_mtp and fast


def choose_resident_layers(count: int, num_layers: int, priority_summary: dict[str, list[int]] | None) -> set[int]:
    """Picks which layers stay resident when the model is streamed: Colibrì priority ranks first."""
    order: list[int] = []
    if priority_summary:
        for rank in ("rank_1_critical", "rank_2_medium", "rank_3_standard"):
            for i in priority_summary.get(rank, []):
                if 0 <= int(i) < num_layers and int(i) not in order:
                    order.append(int(i))
    for i in range(num_layers):
        if i not in order:
            order.append(i)
    return set(order[: max(0, min(count, num_layers))])


def _streaming_ram_budget_bytes() -> int | None:
    """Bounds host-RAM layer caching to 35% of currently free RAM so streaming never swaps."""
    try:
        if os.name == "nt":
            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return max(256 * MiB, int(stat.ullAvailPhys * 0.35))
        elif os.path.isfile("/proc/meminfo"):
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return max(256 * MiB, int(kb * 1024 * 0.35))
    except Exception:
        pass
    return None


def _eos_token_set(eos_token_id: Any) -> set[int]:
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    if isinstance(eos_token_id, (list, tuple, set)):
        return {int(x) for x in eos_token_id if x is not None}
    return set()


_CHAT_TEMPLATE_TURN_STOP_TOKENS: tuple[str, ...] = (
    "<|im_end|>",
    "<|eot_id|>",
    "<end_of_turn>",
    "<|END_OF_TURN_TOKEN|>",
    "<|END_RESPONSE|>",
    "<|end_of_text|>",
    "<|return|>",
    "<|assistant_end|>",
    "<|end|>",
)


def _resolve_effective_eos_ids(tokenizer: Any, config: Any = None, model_dir: Path | None = None) -> set[int]:
    """Collects primary EOS plus chat-template turn-end tokens (e.g. Phi `<|end|>`, Gemma `<end_of_turn>`).

    Avoids false-positive truncation on multi-channel formats (e.g. Harmony `<|channel|>` where
    `<|end|>` separates reasoning channels and `<|return|>` terminates the turn).
    """
    eos = _eos_token_set(getattr(tokenizer, "eos_token_id", None))
    if config is not None:
        tcfg = config.get_text_config() if hasattr(config, "get_text_config") else config
        eos |= _eos_token_set(getattr(tcfg, "eos_token_id", None))
        eos |= _eos_token_set(getattr(config, "eos_token_id", None))
    if model_dir is not None:
        gen_cfg_path = Path(model_dir) / "generation_config.json"
        if gen_cfg_path.is_file():
            try:
                gdata = json.loads(gen_cfg_path.read_text(encoding="utf-8"))
                eos |= _eos_token_set(gdata.get("eos_token_id"))
            except Exception:
                pass
    tmpl = getattr(tokenizer, "chat_template", None)
    if isinstance(tmpl, str) and tmpl:
        is_harmony = "<|channel|>" in tmpl and "<|return|>" in tmpl
        unk_id = getattr(tokenizer, "unk_token_id", None)
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            for tok_str in _CHAT_TEMPLATE_TURN_STOP_TOKENS:
                if tok_str == "<|end|>" and is_harmony:
                    continue
                if tok_str in tmpl:
                    try:
                        tid = convert(tok_str)
                    except Exception:
                        tid = None
                    if isinstance(tid, int) and tid >= 0 and (unk_id is None or tid != unk_id):
                        eos.add(tid)
    return eos


@dataclass
class SamplingConfig:
    """Fast GPU token sampling + anti-loop penalty parameters."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    penalty_window: int = 256
    seed: int | None = None

    @property
    def is_greedy_plain(self) -> bool:
        return (
            float(self.temperature) <= 0.0
            and float(self.repetition_penalty) == 1.0
            and float(self.presence_penalty) == 0.0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "temperature": round(float(self.temperature), 4),
            "top_p": round(float(self.top_p), 4),
            "top_k": int(self.top_k),
            "repetition_penalty": round(float(self.repetition_penalty), 4),
            "presence_penalty": round(float(self.presence_penalty), 4),
            "penalty_window": int(self.penalty_window),
            "seed": self.seed,
        }


@torch.no_grad()
def sample_next_token(
    logits_row: torch.Tensor,
    gen_history: list[int] | None,
    cfg: SamplingConfig | None,
    *,
    rng: torch.Generator | None = None,
) -> int:
    """Selects next token id on GPU/CPU with optional anti-loop penalties and top-k/top-p sampling.

    When `cfg is None` or `cfg.is_greedy_plain`, runs zero-overhead exact `argmax`.
    Penalties apply strictly to `gen_history` (model-generated tokens, never the user prompt).
    """
    row = logits_row.reshape(-1)
    if cfg is None or cfg.is_greedy_plain:
        return int(row.argmax(dim=-1).item())

    scores = row.float().clone()
    vocab = int(scores.numel())

    rp = float(cfg.repetition_penalty)
    pp = float(cfg.presence_penalty)
    if (rp != 1.0 or pp != 0.0) and gen_history:
        win = int(cfg.penalty_window)
        tail = gen_history[-win:] if win > 0 else gen_history
        uniq = sorted({int(t) for t in tail if 0 <= int(t) < vocab})
        if uniq:
            idx = torch.tensor(uniq, dtype=torch.long, device=scores.device)
            vals = scores.index_select(0, idx)
            if rp != 1.0:
                vals = torch.where(vals < 0.0, vals * rp, vals / rp)
            if pp != 0.0:
                vals = vals - pp
            scores.index_copy_(0, idx, vals)

    temp = float(cfg.temperature)
    if temp <= 0.0:
        return int(scores.argmax(dim=-1).item())

    scores = scores / temp
    top_k = int(cfg.top_k)
    top_p = float(cfg.top_p)

    def _multinomial_pick(p: torch.Tensor) -> int:
        if rng is not None and rng.device != p.device:
            p = p.to(rng.device)
        return int(torch.multinomial(p, num_samples=1, generator=rng).item())

    if 0 < top_k < vocab:
        top_v, top_i = torch.topk(scores, k=top_k)
        probs = torch.softmax(top_v, dim=-1)
        if 0.0 < top_p < 1.0:
            cdf = torch.cumsum(probs, dim=-1)
            keep = (cdf - probs) <= top_p
            keep[0] = True
            probs = torch.where(keep, probs, torch.zeros_like(probs))
            probs = probs / probs.sum().clamp_min(1e-12)
        pos = _multinomial_pick(probs)
        return int(top_i[pos].item())

    if 0.0 < top_p < 1.0:
        sorted_v, sorted_i = torch.sort(scores, descending=True)
        probs = torch.softmax(sorted_v, dim=-1)
        cdf = torch.cumsum(probs, dim=-1)
        keep = (cdf - probs) <= top_p
        keep[0] = True
        probs = torch.where(keep, probs, torch.zeros_like(probs))
        probs = probs / probs.sum().clamp_min(1e-12)
        pos = _multinomial_pick(probs)
        return int(sorted_i[pos].item())

    probs = torch.softmax(scores, dim=-1)
    return _multinomial_pick(probs)


class QuintessentiaUnifiedEngine:
    """High-level self-contained unified engine for CLI `ask`, `chat`, and LAN `serve`."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        quintessentia_root: str | Path | None = None,
        vram_mode: str = "normal",
        enable_mtp: bool = True,
        enable_vocab_pruner: bool = False,
        max_drafts: int = 3,
        verbose: bool = True,
        max_context_tokens: int | None = None,
        prefill_chunk_tokens: int | None = None,
        allow_download: bool = False,
        weight_format: str = "auto",
        prefill_wavefront: bool = False,
        cpu_first: bool | None = None,
        progressive_start: bool = True,
        device: str = "auto",
        speculation: str = "auto",
        mtp_head_path: str | Path | None = None,
        mtp_head_budget_mib: int = 512,
        prune_regions: bool = True,
        thought_anchor: bool = False,
        max_slot_mib: float | int | None = None,
        max_slot_bytes: int | None = None,
        q1_planes: int | None = None,
        q1_early_stop: bool = True,
        puzzle_overlay: bool = False,
        puzzle_slice_ratio: float = 0.35,
        cube_root_bits: int = 16,
        cube_blueprint: Any = None,
        cube_relay: bool = True,
        q1_micro_agent_tol: float | None = None,
        q1_moe_fabric: bool = False,
        q1_expert_relay: bool = False,
        q1_expert_rest_tol: float | None = None,
        q1_expert_rest_min: int = 3,
        q1_imprint_puzzle: bool = False,
        q1_imprint_depths: tuple[int, int, int, int] = (5, 4, 3, 2),
        q1_fixation_cov_thresh: float = 1.0,
        q1_graphify_feedback: Any = False,
        auto_fit_q1_planes: bool | None = None,
        min_q1_planes: int = 3,
        max_q1_planes: int = 8,
        lm_head_planes: int | None = None,
        q1_kl_target: float | None = None,
        enable_rssf_balance: bool = True,
        enable_token_ledger: bool = True,
        token_ledger_dir: str | Path | None = None,
        session_id: str = "default",
    ):
        if device not in ("auto", "cpu"):
            raise ValueError("device must be auto or cpu")
        if q1_micro_agent_tol is not None and (
                not math.isfinite(float(q1_micro_agent_tol)) or float(q1_micro_agent_tol) < 0):
            raise ValueError("q1_micro_agent_tol must be finite and nonnegative")
        if q1_micro_agent_tol is not None and weight_format != "q1_cubes":
            raise ValueError("q1_micro_agent_tol requires weight_format=q1_cubes")
        self.q1_micro_agent_tol = None if q1_micro_agent_tol is None else float(q1_micro_agent_tol)
        if q1_moe_fabric and weight_format != "q1_cubes":
            raise ValueError("q1_moe_fabric requires weight_format=q1_cubes")
        self.q1_moe_fabric = bool(q1_moe_fabric)
        if q1_expert_rest_tol is not None and (
                not math.isfinite(float(q1_expert_rest_tol)) or float(q1_expert_rest_tol) < 0):
            raise ValueError("q1_expert_rest_tol must be finite and nonnegative")
        if not math.isfinite(float(q1_fixation_cov_thresh)) or not (0.0 <= float(q1_fixation_cov_thresh) <= 1.0):
            raise ValueError("q1_fixation_cov_thresh must be between 0.0 and 1.0")
        from src.stanok_contour import normalize_feedback_mode
        _fb_mode = normalize_feedback_mode(q1_graphify_feedback)
        if (
            q1_expert_relay
            or q1_expert_rest_tol is not None
            or q1_imprint_puzzle
            or float(q1_fixation_cov_thresh) < 1.0
            or _fb_mode != "off"
        ) and weight_format != "q1_cubes":
            raise ValueError(
                "q1_expert_relay, q1_expert_rest_tol, q1_imprint_puzzle, q1_fixation_cov_thresh, and q1_graphify_feedback require weight_format=q1_cubes"
            )
        self.q1_expert_relay = bool(q1_expert_relay)
        self.q1_expert_rest_tol = None if q1_expert_rest_tol is None else float(q1_expert_rest_tol)
        self.q1_expert_rest_min = max(1, int(q1_expert_rest_min))
        self.q1_imprint_puzzle = bool(q1_imprint_puzzle)
        self.q1_imprint_depths = tuple(int(x) for x in q1_imprint_depths)
        self.q1_fixation_cov_thresh = float(q1_fixation_cov_thresh)
        self.q1_graphify_feedback = _fb_mode if _fb_mode != "off" else False
        if speculation not in ("auto", "off", "lookup", "native", "q1") or mtp_head_budget_mib <= 0:
            raise ValueError("Invalid speculation mode or native head budget")
        if weight_format not in SUPPORTED_RUNTIME_WEIGHT_FORMATS:
            raise ValueError("Unsupported runtime weight format")
        if q1_planes is None:
            q1_planes = cube_root_bits if weight_format == "q1_cubes" else 4
        if not 1 <= int(q1_planes) <= (16 if weight_format == "q1_cubes" else 8):
            raise ValueError("q1_planes exceeds the representation depth (q1_cubes supports 16)")
        if cube_root_bits not in (2, 4, 8, 16):
            raise ValueError("cube_root_bits must be 2, 4, 8 or 16")
        if weight_format == "q1_cubes" and int(q1_planes) > cube_root_bits:
            raise ValueError("q1_planes cannot exceed cube_root_bits")
        if not (0.0 <= float(puzzle_slice_ratio) <= 1.0) or math.isnan(float(puzzle_slice_ratio)):
            raise ValueError("puzzle_slice_ratio must be between 0.0 and 1.0")
        if max_slot_bytes is not None:
            resolved_slot_bytes = int(max_slot_bytes)
        elif max_slot_mib is not None:
            resolved_slot_bytes = int(float(max_slot_mib) * MiB)
        else:
            resolved_slot_bytes = None
        if resolved_slot_bytes is not None and resolved_slot_bytes <= 0:
            raise ValueError("max_slot_mib / max_slot_bytes must be positive")
        if speculation == "auto" and not enable_mtp:
            speculation = "off"
        if speculation == "off":
            enable_mtp = False
        self.speculator = self._region_controller = None
        self.survivor_tracker = None
        self.stanok_contour = None
        self.probability_bus = None
        self.cold_warmup_scale = None
        self._q1_cubes_runner = None
        self.thought_anchor_enabled = bool(thought_anchor)
        self._session_thought_anchor = None
        self.enable_rssf_balance = bool(enable_rssf_balance)
        self.enable_token_ledger = bool(enable_token_ledger)
        self._token_ledger_dir = token_ledger_dir
        self.session_id = str(session_id or "default")
        self.token_ledger: Any = None
        self._last_ledger_recovery: dict[str, Any] = {
            "kind": "fresh",
            "matched_turn_index": -1,
            "disk_prefix_tokens": 0,
            "kv_reused_tokens": 0,
        }
        from src.rssf_dynamic_operators import GenerationBalanceTracker
        self.generation_balance = GenerationBalanceTracker() if self.enable_rssf_balance else None
        self._active_sampling: SamplingConfig | None = None
        self._sampling_rng: torch.Generator | None = None
        self.max_slot_bytes = resolved_slot_bytes
        self.q1_planes = int(q1_planes)
        self.q1_planes_loaded = int(q1_planes)
        self.q1_planes_requested = int(q1_planes)
        self.q1_lm_head_planes: int | None = (int(lm_head_planes) if lm_head_planes is not None else None)
        self.cube_root_bits = int(cube_root_bits)
        self.cube_relay = bool(cube_relay)
        from src.stanok_contour import load_cube_blueprint
        self.cube_blueprint, self.cube_expert_blueprint, bp_max_depth = load_cube_blueprint(
            cube_blueprint, expected_root_bits=self.cube_root_bits
        )
        if self.cube_blueprint and weight_format != "q1_cubes":
            raise ValueError("A cube blueprint requires weight_format=q1_cubes")
        if bp_max_depth > 0:
            self.q1_planes = max(self.q1_planes, bp_max_depth)
            self.q1_planes_loaded = self.q1_planes
        self.min_q1_planes = max(1, int(min_q1_planes))
        self.max_q1_planes = max(self.min_q1_planes, min(16, int(max_q1_planes)))
        self.q1_kl_target: float | None = (
            max(1e-4, float(q1_kl_target)) if (q1_kl_target is not None and float(q1_kl_target) > 0) else None
        )
        self.q1_kl_calibration: dict[str, Any] | None = None
        if self.max_q1_planes > self.cube_root_bits and weight_format == "q1_cubes":
            self.cube_root_bits = 16 if self.max_q1_planes > 8 else 8
        self.auto_fit_q1_planes = (
            bool(weight_format == "q1_cubes" and self.cube_blueprint is None)
            if auto_fit_q1_planes is None
            else bool(auto_fit_q1_planes)
        )
        self.q1_early_stop = bool(q1_early_stop)
        self.puzzle_overlay = bool(puzzle_overlay)
        self.puzzle_slice_ratio = float(puzzle_slice_ratio)
        self.root = find_quintessentia_root(quintessentia_root)
        # Historical spelling remains accepted by local experiments; it no
        # longer means that GPU startup waits for CPU work.
        if cpu_first is not None:
            progressive_start = bool(cpu_first)
        from .live_hud_beacon import LiveTelemetryBeacon
        self._hud_beacon = LiveTelemetryBeacon()
        use_direct_q1_cubes_gpu = bool(weight_format == "q1_cubes" and device != "cpu" and torch.cuda.is_available())
        if (progressive_start and not use_direct_q1_cubes_gpu) or device == "cpu":
            self._init_progressive(model_dir, vram_mode, enable_mtp, enable_vocab_pruner,
                                 max_drafts, verbose, max_context_tokens,
                                 prefill_chunk_tokens, allow_download, weight_format,
                                 prefill_wavefront, allow_gpu=device != "cpu")
            self._configure_general_stream(
                speculation,
                mtp_head_path,
                mtp_head_budget_mib,
                bool(prune_regions and self.profile.get("weight_format") != "q1_cubes"),
            )
            self._hud_beacon.publish_engine_ready(self)
            return
        from src.quintessentia_engine import PriorityRanker
        from src.quintessentia_integrated import QuintessentiaModel
        from src.catalog_bridge import ExistingCrossLayerCatalog
        from src.existing_hooks import ExistingForwardHookAdapter
        from src.lease_ack_bridge import ExistingRingAcknowledgementAdapter
        from src.upstream_bridge import NativeTierPolicy
        from .model_source import (
            PreadSafetensorsReader,
            open_model_source,
            resolve_model_dir,
        )

        if vram_mode not in VRAM_MODES:
            raise ValueError(
                f"vram_mode must be one of {VRAM_MODES} (got {vram_mode!r}); fixed `4gb`/`8gb` profiles were removed"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required to run QuintessentiaUnifiedEngine.")

        self.engine = None
        self.pread_reader = None
        self.native_policy = None
        self.device = torch.device("cuda:0")
        self.gpu_name = torch.cuda.get_device_name(self.device)
        self.model_dir = resolve_model_dir(model_dir, self.root, allow_download=allow_download)
        self.model_id = self.model_dir.name
        self.model_source = open_model_source(self.model_dir)
        self.model_format = getattr(self.model_source, "kind", "hf") if self.model_source is not None else "quintessentia"

        from .gpu_memory import measure_gpu_memory
        self.memory_probe = measure_gpu_memory(self.device)
        free_b, total_b = self.memory_probe["free_bytes"], self.memory_probe["total_bytes"]
        self.total_vram_gib = round(total_b / GiB, 2)
        extra_plan_kw = {"max_slot_bytes": self.max_slot_bytes} if self.max_slot_bytes is not None else {}
        if weight_format in ("q1", "q1_cubes") or self.q1_planes != 4:
            extra_plan_kw["q1_planes"] = self.q1_planes
        if weight_format == "q1_cubes":
            extra_plan_kw["auto_fit_q1_planes"] = self.auto_fit_q1_planes
            extra_plan_kw["min_q1_planes"] = self.min_q1_planes
            extra_plan_kw["max_q1_planes"] = self.max_q1_planes
            if self.q1_lm_head_planes is not None:
                extra_plan_kw["lm_head_planes"] = self.q1_lm_head_planes
        self.profile, self.plan, has_mtp = plan_for_checkpoint(
            self.model_dir,
            mode=vram_mode,
            total_bytes=total_b,
            free_bytes=free_b,
            enable_mtp=enable_mtp,
            max_context_tokens=max_context_tokens,
            prefill_chunk_tokens=prefill_chunk_tokens,
            weight_format=weight_format,
            prefill_wavefront=prefill_wavefront,
            force_universal=(
                mtp_head_path is not None
                or speculation in ("native", "lookup", "q1")
                or self.max_slot_bytes is not None
                or self.puzzle_overlay
            ),
            **extra_plan_kw,
        )
        if self.profile.get("weight_format") == "q1_cubes":
            self.q1_planes = int(self.profile.get("q1_planes", self.q1_planes))
            self.q1_planes_loaded = int(self.q1_planes)
            self.q1_planes_requested = int(self.profile.get("q1_planes_requested", self.q1_planes_requested))
            self.q1_lm_head_planes = self.profile.get("q1_lm_head_planes")
            if self.q1_planes_loaded > self.cube_root_bits:
                self.cube_root_bits = 16 if self.q1_planes_loaded > 8 else 8
        self.fast_qwen = bool(self.profile["fast_qwen"])
        self.vram_mode = vram_mode
        self.prefill_chunk_tokens = self.plan.prefill_chunk_tokens
        self.prefill_wavefront = bool(prefill_wavefront)
        self._wavefront_windows: list[dict[str, Any]] = []
        num_layers = int(self.plan.layers_total)

        dll_path = self.root / "data" / "native" / "colibri_tier.dll"
        self.has_native_dll = dll_path.exists() and os.name == "nt"
        self.pread_reader = PreadSafetensorsReader(max_handles=8)

        t_boot0 = time.perf_counter()
        if verbose:
            p = self.plan
            layout = (
                f"all {p.layers_total} layers resident"
                if p.full_residency
                else f"{p.resident_layers}/{p.layers_total} resident + {p.ring_slots}-slot ring ({p.slot_bytes / MiB:.0f} MiB each)"
            )
            planes_tag = ""
            if self.profile.get("weight_format") == "q1_cubes":
                if self.profile.get("q1_planes_auto_halved"):
                    planes_tag = f" (k={self.q1_planes}, auto-halved from k={self.q1_planes_requested})"
                else:
                    planes_tag = f" (k={self.q1_planes})"
            print(
                f"[Quintessentia] GPU: {self.gpu_name} ({self.total_vram_gib} GiB, free {free_b / GiB:.2f} GiB) | "
                f"Mode: {vram_mode} | Backend: {self.profile['backend']}{planes_tag} | "
                f"Budget: {p.budget_bytes / GiB:.2f} GiB, reserve {p.reserve_bytes / GiB:.2f} GiB | "
                f"{layout} | Model: {self.model_id} ({self.model_format})",
                flush=True,
            )

        self.max_drafts = max(0, min(3, int(max_drafts)))
        self.gk = None
        self.snap_mgr = None
        self.captured = {}
        self.draft_vocab = None
        self.mtp_head = None
        self.mtp_vram_mib = 0.0
        self.enable_vocab_pruner = False
        self.anchor_schedule = {"resident_layers": [], "streamed_layers": []}

        if self.fast_qwen:
            from .kernels import GrandUnifiedKernels

            q4_cache = Path(os.environ.get("QUINTESSENTIA_CACHE_DIR") or (self.root / "data" / "q4-g64-cache"))
            if self.model_source is not None:
                use_cache_dir, use_manifest = str(q4_cache), None
            else:
                manifest = checkpoint_manifest_path(self.root)
                have = q4_cache.exists() and manifest.exists()
                use_cache_dir = str(q4_cache) if have else None
                use_manifest = str(manifest) if have else None
            self.gk = GrandUnifiedKernels(self.device)
            common = dict(
                pipeline_mode="triple_ring_prefetch",
                weight_format="q4",
                quantize_lm_head=True,
                q4_group_size=64,
                q4_cache_dir=use_cache_dir,
                checkpoint_manifest=use_manifest,
                model_source=self.model_source,
                weight_reader=self.pread_reader,
            )
            if self.plan.full_residency:
                self.engine = QuintessentiaModel(str(self.model_dir), vram_resident_layers=num_layers, **common)
                self.anchor_schedule = {"resident_layers": list(range(num_layers)), "streamed_layers": []}
            else:
                prio = None
                if self.model_source is None:
                    try:
                        prio = PriorityRanker(str(self.model_dir)).get_priority_summary()
                    except Exception:
                        prio = None
                resident_set = choose_resident_layers(self.plan.resident_layers, num_layers, prio)
                max_ram_layers = max(4, min(num_layers, 16))
                if self.has_native_dll:
                    try:
                        self.native_policy = NativeTierPolicy(
                            str(dll_path), num_layers=num_layers, capacity=max_ram_layers, mode="apply", decay_interval=16
                        )
                    except Exception:
                        self.native_policy = None
                self.engine = QuintessentiaModel(
                    str(self.model_dir),
                    vram_resident_layers=0,
                    max_ram_layers=max_ram_layers,
                    max_ram_bytes=_streaming_ram_budget_bytes(),
                    cache_policy="lru",
                    num_slots=self.plan.ring_slots,
                    control_mode="off",
                    native_policy=self.native_policy,
                    **common,
                )
                self.engine.tail_to_head_prefetch = True
                self.anchor_schedule = install_quintessentia_anchor_ring_schedule(self.engine, resident_set)
                if self.model_source is None:
                    catalog = ExistingCrossLayerCatalog(str(self.model_dir))
                    self.engine.trace_adapter = ExistingForwardHookAdapter(self.engine, catalog).install()
                    self.engine.ack_adapter = ExistingRingAcknowledgementAdapter(self.engine.pipeline).install()

            self.snap_mgr = HybridSpecSnapshotManager(self.engine.config, self.device, max_drafts=max(2, self.max_drafts))
            self.captured = install_quintessentia_multirow_unifications(
                self.engine,
                self.gk,
                snap_mgr=self.snap_mgr,
                enable_vocab_pruner=enable_vocab_pruner,
            )
            self.enable_vocab_pruner = enable_vocab_pruner
            self.mtp_enabled = bool(enable_mtp and has_mtp and self.plan.enable_mtp and self.max_drafts >= 1)
            if self.mtp_enabled:
                self.draft_vocab = QuintessentiaDraftVocab(self.engine.config.vocab_size, top_k=16000, device=self.device)
                seed_ids = self.engine.tokenizer(DEFAULT_DRAFT_SEED_CORPUS, add_special_tokens=False).input_ids
                self.draft_vocab.seed_from_corpus_tokens(seed_ids)
                self.mtp_head = QuintessentiaNativeMTPHead(self.engine, self.gk, self.draft_vocab)
                self.mtp_vram_mib = round(self.mtp_head.q4_vram_bytes() / MiB, 2)
        elif self.profile["weight_format"] == "q1_cubes":
            from src.binary_halving_q1 import attach_q1_cubes_to_engine

            self.engine = QuintessentiaModel(
                str(self.model_dir),
                pipeline_mode="sequential",
                vram_resident_layers=0,
                max_ram_layers=0,
                max_ram_bytes=0,
                cache_policy="lru",
                num_slots=1,
                weight_format="bf16",
                universal_quantization=False,
                quantize_lm_head=False,
                use_fused_q4_gemv=False,
                model_source=self.model_source,
                weight_reader=self.pread_reader,
                expert_cache_bytes=0,
            )
            has_moe = bool(
                self.model_source is not None and getattr(self.model_source, "_expert_loader", None) is not None
            )
            layer_b_list = list(self.profile.get("q1_cubes_layer_bytes") or [])
            lm_head_b = int(self.profile.get("q1_cubes_lm_head_bytes", 0))
            avail_after_head = max(0, int(self.plan.budget_bytes) - lm_head_b)
            if has_moe:
                if avail_after_head >= 512 * MiB:
                    moe_gpu_bytes = min(1536 * MiB, max(256 * MiB, int(avail_after_head * 0.45)))
                else:
                    moe_gpu_bytes = max(0, int(avail_after_head * 0.35))
                trunk_room = max(0, avail_after_head - moe_gpu_bytes - 256 * MiB)
                if layer_b_list and sum(layer_b_list) <= trunk_room:
                    resident = num_layers
                elif layer_b_list:
                    max_l = max(layer_b_list)
                    res_room = max(0, trunk_room - max_l)
                    resident = 0
                    used_res = 0
                    for lb in layer_b_list:
                        if used_res + lb > res_room:
                            break
                        used_res += lb
                        resident += 1
                    resident = min(resident, max(0, num_layers - 1))
                else:
                    resident = 0
            else:
                moe_gpu_bytes = 0
                resident = num_layers if self.plan.full_residency else int(self.plan.resident_layers)

            ram_budget = _streaming_ram_budget_bytes() or (2048 * MiB)
            if has_moe:
                layer_host_bytes = min(1024 * MiB, ram_budget // 3)
                moe_host_bytes = min(8192 * MiB, max(0, ram_budget - layer_host_bytes))
            else:
                layer_host_bytes = min(8192 * MiB, ram_budget)
                moe_host_bytes = 0
            is_v4_isolated = (self.q1_graphify_feedback == "mirror_scales")
            if not is_v4_isolated:
                from src.stanok_contour import StanokContainmentContour, UnifiedProbabilityBus
                if self.stanok_contour is None:
                    self.stanok_contour = StanokContainmentContour(
                        device=self.device, enabled=(self.device.type == "cuda")
                    )
                if self.probability_bus is None:
                    self.probability_bus = UnifiedProbabilityBus(self.stanok_contour)
            self._q1_cubes_runner = attach_q1_cubes_to_engine(
                self.engine,
                cubes_per_proj=self.q1_planes,
                root_bits=self.cube_root_bits,
                layer_proj_cubes=self.cube_blueprint,
                layer_expert_cubes=self.cube_expert_blueprint,
                enable_relay=False if is_v4_isolated else self.cube_relay,
                micro_agent_tol=self.q1_micro_agent_tol,
                moe_fabric=self.q1_moe_fabric,
                enable_expert_relay=False if is_v4_isolated else self.q1_expert_relay,
                expert_rest_tol=self.q1_expert_rest_tol,
                expert_rest_min=self.q1_expert_rest_min,
                imprint_puzzle=self.q1_imprint_puzzle,
                imprint_depths=self.q1_imprint_depths,
                fixation_cov_thresh=self.q1_fixation_cov_thresh,
                stanok_contour=None if is_v4_isolated else self.stanok_contour,
                graphify_feedback=self.q1_graphify_feedback,
                mirror_v4_trunk=is_v4_isolated,
                lm_head_cubes=self.q1_lm_head_planes,
                resident_layers=resident,
                host_cache_bytes=layer_host_bytes,
                moe_gpu_cache_bytes=moe_gpu_bytes,
                moe_host_cache_bytes=moe_host_bytes,
            )
            if is_v4_isolated:
                from src.mirror_scales_v4 import MirrorColdWarmupScale, warmup_v4_cuda_kernels
                self.cold_warmup_scale = MirrorColdWarmupScale(
                    mode="plateau", min_layers_warm=2, plateau_confirm=2, max_warmup_seconds=6.0
                )
                self.cold_warmup_scale.run_warmup(
                    kernel_warmup_fn=lambda: warmup_v4_cuda_kernels(self.device),
                    layer_warmup_fn=self._q1_cubes_runner.warmup_resident_layer,
                    total_resident_layers=resident,
                )
            self.anchor_schedule = {
                "resident_layers": list(range(resident)),
                "streamed_layers": list(range(resident, num_layers)),
            }
            self.mtp_enabled = False
            self.enable_vocab_pruner = False
        else:
            sub_cubes = bool(getattr(self.plan, "sub_layer_cubes", False))
            sub_head = bool(getattr(self.plan, "streamed_lm_head", False))
            resident = num_layers if self.plan.full_residency else int(self.plan.resident_layers)
            expert_cache_bytes = max(0, int(self.plan.budget_bytes - self.plan.weights_bytes - 256 * MiB))
            self.engine = QuintessentiaModel(
                str(self.model_dir),
                pipeline_mode="triple_ring_prefetch",
                vram_resident_layers=resident,
                max_ram_layers=max(4, min(num_layers, 16)),
                max_ram_bytes=None if self.plan.full_residency else _streaming_ram_budget_bytes(),
                cache_policy="lru",
                num_slots=None if self.plan.full_residency else self.plan.ring_slots,
                weight_format=self.profile["weight_format"],
                universal_quantization=True,
                quantize_lm_head=False,
                use_fused_q4_gemv=False,
                model_source=self.model_source,
                weight_reader=self.pread_reader,
                expert_cache_bytes=expert_cache_bytes,
                sub_layer_cubes=sub_cubes,
                streamed_lm_head=sub_head,
                q1_slot_budget_bytes=self.plan.slot_bytes if (sub_cubes or sub_head) else None,
                q1_planes=self.q1_planes,
                q1_early_stop=self.q1_early_stop,
                puzzle_overlay=self.puzzle_overlay,
                puzzle_slice_ratio=self.puzzle_slice_ratio,
                puzzle_slot_budget_bytes=(
                    self.max_slot_bytes or (self.plan.slot_bytes if self.plan.slot_bytes > 0 else 16 * MiB)
                ) if self.puzzle_overlay else None,
            )
            self.anchor_schedule = {
                "resident_layers": list(range(resident)),
                "streamed_layers": list(range(resident, num_layers)),
            }
            self.mtp_enabled = False
            self.enable_vocab_pruner = False

        from .kv_offload import enable_zero_copy_gqa_sdpa
        enable_zero_copy_gqa_sdpa()
        self.kv_offload_enabled = True
        self.session = QuintessentiaKVPrefixSession()
        self._session_messages: list[dict[str, Any]] = []
        self._session_prompt_messages: list[dict[str, Any]] = []
        self._session_last_tool_calls: list[dict[str, Any]] = []
        self._session_tools_sig: str = ""
        self._idle_free_vram_bytes: int | None = None
        self.calibration: dict[str, Any] | None = None
        self.last_metrics: dict[str, Any] = {}
        self.generations_started = 0
        self.generations_cancelled = 0

        is_q1_cubes = self.profile.get("weight_format") == "q1_cubes"
        self._attach_unified_primary_components(
            allow_dynamic_memory=bool(not self.prefill_wavefront and not is_q1_cubes)
        )

        if self.fast_qwen:
            warm_ids = self.engine.tokenizer("Привет!", return_tensors="pt").input_ids[0].tolist()
            for _ in self._stream_token_ids(warm_ids, max_new_tokens=4, stop_on_eos=True, cached_prefix_len=0, past_kv=None):
                pass
            self.reset_session(clear_ledger=False)
        self._init_token_ledger()
        if self.q1_kl_target is not None and self.profile.get("weight_format") == "q1_cubes":
            try:
                self.q1_kl_calibration = self.calibrate_q1_planes_by_kl(
                    kl_target=self.q1_kl_target,
                    min_k=self.min_q1_planes,
                )
                if verbose and self.q1_kl_calibration.get("ok"):
                    print(
                        f"[Quintessentia] KL-calibrated Q1 depth: k={self.q1_kl_calibration['chosen_k']}/"
                        f"{self.q1_kl_calibration['reference_k']} (target KL<={self.q1_kl_target}, "
                        f"actual KL={self.q1_kl_calibration['chosen_kl']})",
                        flush=True,
                    )
            except Exception:
                pass

        if not self.fast_qwen:
            self._configure_general_stream(
                speculation, mtp_head_path, mtp_head_budget_mib, bool(prune_regions and not is_q1_cubes)
            )

        self.boot_seconds = round(time.perf_counter() - t_boot0, 2)
        self.resident_vram_mib = round(torch.cuda.memory_allocated(self.device) / MiB, 2)
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                _fb, _ = torch.cuda.mem_get_info(self.device)
                _alloc = int(torch.cuda.memory_allocated(self.device))
                _pending = max(0, int(getattr(self.plan, "weights_bytes", 0) or 0) - _alloc)
                self._idle_free_vram_bytes = max(
                    0, int(_fb) + max(0, int(torch.cuda.memory_reserved(self.device)) - _alloc) - _pending
                )
            except Exception:
                pass
        if verbose:
            ec = self.expert_cache_stats()
            if ec is not None and ec.get("slots", 0) > 0:
                ec_str = f", ExpertCache: {ec['slots']} slots ({ec['arena_mib']} MiB)"
            elif ec is not None and "total_model_experts" in ec:
                ec_str = f", Q1MoECache: L1/L2 ({ec['total_model_experts']} experts)"
            else:
                ec_str = ""
            mtp_str = "off"
            if self.mtp_enabled:
                nat = getattr(self.speculator, "native", None) if self.speculator is not None else None
                if nat is not None:
                    mtp_str = f"on ({nat.device.type}, {nat.weight_bytes / MiB:.0f} MiB)"
                else:
                    mtp_str = "on"
            print(
                f"[Quintessentia] Ready in {self.boot_seconds}s | Resident VRAM: {self.resident_vram_mib} MiB "
                f"(backend {self.profile['backend']}, MTP: {mtp_str}, "
                f"FastVocab: {self.enable_vocab_pruner}{ec_str})",
                flush=True,
            )

        if getattr(self, "_hud_beacon", None) is not None:
            self._hud_beacon.publish_engine_ready(self)

    def _attach_unified_primary_components(self, *, allow_dynamic_memory: bool = True) -> None:
        """Attach the Unified Primary Engine mechanisms by default (A->B->C(0)->A + Survivor Tracker)."""
        eng = self.engine
        if eng is None:
            return
        src = getattr(eng, "model_source", None)
        exp_cache = getattr(src, "_expert_loader", None) if src is not None else None
        moe_q1_q2 = bool(exp_cache is not None and getattr(exp_cache, "bits", None) in (1, 2))
        if allow_dynamic_memory and not moe_q1_q2 and not getattr(self, "puzzle_overlay", False):
            prog = getattr(eng, "progressive", None)
            if prog is not None and not prog.complete:
                if prog.gpu_enabled:
                    try:
                        eng.enable_unified_memory(_auto_unified=True, gpu_budget_bytes=0, host_cache_bytes=64 * MiB)
                    except ValueError:
                        pass
            elif (
                not getattr(self.plan, "full_residency", False)
                and not getattr(self.plan, "sub_layer_cubes", False)
                and getattr(getattr(eng, "pipeline", None), "num_slots", 0) == 3
            ):
                try:
                    eng.enable_unified_memory(_auto_unified=True, gpu_budget_bytes=0, host_cache_bytes=64 * MiB)
                except ValueError:
                    pass
        if self.survivor_tracker is None:
            from src.survivor_probability_path import SurvivorProbabilityTracker
            self.survivor_tracker = SurvivorProbabilityTracker(
                eng,
                draft_vocab=self.draft_vocab,
                mirror_mode=(self.q1_graphify_feedback == "mirror_scales"),
            )
            eng.survivor_tracker = self.survivor_tracker
        if self.q1_graphify_feedback == "mirror_scales":
            # V4 Isolated Mirror Scales: NO UnifiedProbabilityBus, NO StanokContainmentContour coupling!
            eng.stanok_contour = None
            eng.probability_bus = None
            return
        from src.stanok_contour import StanokContainmentContour, UnifiedProbabilityBus
        if self.stanok_contour is None:
            dev = getattr(eng, "device", self.device)
            self.stanok_contour = StanokContainmentContour(
                device=dev, enabled=(torch.device(dev).type == "cuda")
            )
        if self.probability_bus is None:
            self.probability_bus = UnifiedProbabilityBus(self.stanok_contour)
        imprint_ov = None
        q1_runner = getattr(self, "_q1_cubes_runner", None)
        if q1_runner is not None and getattr(q1_runner, "moe_cache", None) is not None:
            imprint_ov = getattr(q1_runner.moe_cache, "_imprint_overlay", None)
        self.probability_bus.bind(
            survivor_tracker=self.survivor_tracker,
            route_exclusion=self._region_controller,
            imprint_overlay=imprint_ov,
        )
        eng.stanok_contour = self.stanok_contour
        eng.probability_bus = self.probability_bus

    def _configure_general_stream(self, mode, head_path, head_budget_mib, prune_regions):
        try:
            if prune_regions:
                from src.route_exclusion import RouteExclusion
                self._region_controller = RouteExclusion(self.engine)
                self.engine.route_exclusion = self._region_controller
                if self.probability_bus is not None:
                    self.probability_bus.bind(route_exclusion=self._region_controller)
            if mode != "off" and self.max_drafts:
                from src.general_speculation import GeneralSpeculation
                self.speculator = GeneralSpeculation(
                    self,
                    mode,
                    head_path,
                    head_budget_mib*MiB,
                    mirror_mode=(self.q1_graphify_feedback == "mirror_scales"),
                )
                self.mtp_enabled = self.speculator.native is not None
            if (
                self.profile.get("weight_format") == "q1_cubes"
                and self.device.type == "cuda"
                and self.engine is not None
            ):
                snap_slots = max(3, int(self.max_drafts or 0)) if self.speculator is not None else 0
                self.snap_mgr = install_q1_qwen35_fast_decode_unifications(
                    self.engine,
                    max_drafts=snap_slots,
                    full_residency=bool(getattr(self.plan, "full_residency", False)),
                )
        except BaseException:
            self.close()
            raise

    def _init_progressive(self, model_dir, mode, enable_mtp, enable_vocab_pruner,
                        max_drafts, verbose, max_context, prefill_chunk,
                        allow_download, weight_format, prefill_wavefront, allow_gpu=True):
        from .model_source import (PreadSafetensorsReader, open_model_source,
                                   resolve_model_dir, runtime_profile, load_runtime_config)
        from src.quintessentia_integrated import QuintessentiaModel
        if mode not in VRAM_MODES:
            raise ValueError(f"vram_mode must be one of {VRAM_MODES}")
        if weight_format not in SUPPORTED_RUNTIME_WEIGHT_FORMATS:
            raise ValueError("Unsupported runtime weight format")
        self.engine = self.pread_reader = self.model_source = self.native_policy = None
        self.device = torch.device("cpu")
        self.gpu_name = "preparing in parallel" if allow_gpu and torch.cuda.is_available() else "NONE"
        self.total_vram_gib = 0.0
        self.model_dir = resolve_model_dir(model_dir, self.root, allow_download=allow_download)
        self.model_id = self.model_dir.name
        self.model_source = open_model_source(self.model_dir)
        self.model_format = getattr(self.model_source, "kind", "quintessentia")
        self.profile = runtime_profile(self.model_dir)
        fmt = ("q4" if self.profile["fast_qwen"] else "q6") if weight_format == "auto" else weight_format
        self.profile = dict(
            self.profile,
            fast_qwen=False,
            backend="unified-q1-cubes-cpu" if fmt == "q1_cubes" else "progressive-hf",
            weight_format=fmt,
        )
        if fmt in ("q1", "q1_cubes"):
            self.profile["q1_planes"] = self.q1_planes
        config = load_runtime_config(self.model_dir)
        layers = int(getattr(config, "num_hidden_layers", 0) or getattr(config, "n_layer", 0)
                     or getattr(config, "n_layers", 0))
        self.vram_mode = mode
        self.prefill_chunk_tokens = int(prefill_chunk or 512)
        context = int(max_context or 8192)
        if context < 1 or self.prefill_chunk_tokens < 1:
            raise ValueError("Context and prefill chunk must be positive")
        self.plan = VRAMPlan(mode, 0, 0, 0, 0, layers, 0, 3, 0, False, False,
                             self.prefill_chunk_tokens, context, 0)
        self.fast_qwen = False
        self.prefill_wavefront = bool(prefill_wavefront)
        self._wavefront_windows = []
        self.max_drafts = max(0, min(3, int(max_drafts)))
        self.gk = self.snap_mgr = self.draft_vocab = self.mtp_head = None
        self.captured = {}
        self.mtp_enabled = False
        self.mtp_vram_mib = 0.0
        self.enable_vocab_pruner = False
        self._requested_native_mtp = bool(enable_mtp)
        self.has_native_dll = False
        self.anchor_schedule = {"resident_layers": [], "streamed_layers": list(range(layers))}
        self.pread_reader = PreadSafetensorsReader(max_handles=8)
        from .kv_offload import enable_zero_copy_gqa_sdpa
        enable_zero_copy_gqa_sdpa()
        self.kv_offload_enabled = True
        self.session = QuintessentiaKVPrefixSession()
        self._session_messages = []
        self._session_prompt_messages = []
        self._session_last_tool_calls = []
        self._session_tools_sig = ""
        self._idle_free_vram_bytes = None
        self.calibration = None
        self.last_metrics = {}
        self.generations_started = self.generations_cancelled = 0

        def prepare_gpu_plan(initializing_engine):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is required for progressive GPU takeover")
            gpu = torch.device("cuda:0")
            from .gpu_memory import measure_gpu_memory
            self.memory_probe = measure_gpu_memory(gpu)
            free, total = self.memory_probe["free_bytes"], self.memory_probe["total_bytes"]
            extra_plan_kw = {"max_slot_bytes": self.max_slot_bytes} if self.max_slot_bytes is not None else {}
            q1_p = getattr(self, "q1_planes", 4)
            if fmt in ("q1", "q1_cubes") or q1_p != 4:
                extra_plan_kw["q1_planes"] = q1_p
            _, plan, _ = plan_for_checkpoint(
                self.model_dir, mode=mode, total_bytes=total, free_bytes=free,
                enable_mtp=False, max_context_tokens=context,
                prefill_chunk_tokens=self.prefill_chunk_tokens, weight_format=fmt,
                prefill_wavefront=prefill_wavefront, force_universal=True,
                projection_metadata=initializing_engine.weight_manager.quantized_projections,
                **extra_plan_kw)
            self.plan = plan
            self.gpu_name = torch.cuda.get_device_name(gpu)
            self.total_vram_gib = round(total / GiB, 2)
            self.anchor_schedule = {"resident_layers": list(range(plan.resident_layers)),
                                    "streamed_layers": list(range(plan.resident_layers, layers))}
            return plan

        boot = time.perf_counter()
        try:
            if fmt == "q1_cubes":
                from src.binary_halving_q1 import attach_q1_cubes_to_engine

                self.engine = QuintessentiaModel(
                    str(self.model_dir),
                    device=torch.device("cpu"),
                    pipeline_mode="sequential",
                    vram_resident_layers=0,
                    max_ram_layers=0,
                    max_ram_bytes=0,
                    cache_policy="lru",
                    num_slots=1,
                    weight_format="bf16",
                    universal_quantization=False,
                    use_fused_q4_gemv=False,
                    model_source=self.model_source,
                    weight_reader=self.pread_reader,
                    cpu_first_plan=None,
                    expert_cache_bytes=0,
                    progressive_gpu=False,
                )
                ram_b = _streaming_ram_budget_bytes() or (512 * MiB)
                is_v4_isolated = (self.q1_graphify_feedback == "mirror_scales")
                if not is_v4_isolated:
                    from src.stanok_contour import StanokContainmentContour, UnifiedProbabilityBus
                    if self.stanok_contour is None:
                        self.stanok_contour = StanokContainmentContour(device=torch.device("cpu"), enabled=False)
                    if self.probability_bus is None:
                        self.probability_bus = UnifiedProbabilityBus(self.stanok_contour)
                self._q1_cubes_runner = attach_q1_cubes_to_engine(
                    self.engine,
                    cubes_per_proj=self.q1_planes,
                    root_bits=self.cube_root_bits,
                    layer_proj_cubes=self.cube_blueprint,
                    layer_expert_cubes=self.cube_expert_blueprint,
                    enable_relay=False if is_v4_isolated else self.cube_relay,
                    micro_agent_tol=self.q1_micro_agent_tol,
                    moe_fabric=self.q1_moe_fabric,
                    enable_expert_relay=False if is_v4_isolated else self.q1_expert_relay,
                    expert_rest_tol=self.q1_expert_rest_tol,
                    expert_rest_min=self.q1_expert_rest_min,
                    imprint_puzzle=self.q1_imprint_puzzle,
                    imprint_depths=self.q1_imprint_depths,
                    fixation_cov_thresh=self.q1_fixation_cov_thresh,
                    stanok_contour=None if is_v4_isolated else self.stanok_contour,
                    graphify_feedback=self.q1_graphify_feedback,
                    mirror_v4_trunk=is_v4_isolated,
                    resident_layers=0,
                    host_cache_bytes=min(512 * MiB, ram_b // 3),
                    moe_gpu_cache_bytes=0,
                    moe_host_cache_bytes=min(1024 * MiB, ram_b - min(512 * MiB, ram_b // 3)),
                )
                self._attach_unified_primary_components(allow_dynamic_memory=False)
            else:
                direct_cpu_puzzle = bool(self.puzzle_overlay and not allow_gpu)
                self.engine = QuintessentiaModel(
                    str(self.model_dir), device=torch.device("cpu"),
                    vram_resident_layers=layers if direct_cpu_puzzle else 0,
                    max_ram_layers=max(4, min(max(1, layers), 16)),
                    max_ram_bytes=_streaming_ram_budget_bytes(), cache_policy="lru",
                    num_slots=1 if direct_cpu_puzzle else 3,
                    weight_format=fmt, universal_quantization=True,
                    use_fused_q4_gemv=False, model_source=self.model_source,
                    weight_reader=self.pread_reader,
                    cpu_first_plan=None if direct_cpu_puzzle else prepare_gpu_plan,
                    expert_cache_bytes=64*MiB, progressive_gpu=allow_gpu,
                    q1_planes=self.q1_planes, q1_early_stop=self.q1_early_stop,
                    puzzle_overlay=self.puzzle_overlay, puzzle_slice_ratio=self.puzzle_slice_ratio,
                    puzzle_slot_budget_bytes=self.max_slot_bytes)
                self._attach_unified_primary_components(allow_dynamic_memory=bool(allow_gpu and not prefill_wavefront))
        except BaseException:
            self.close()
            raise
        self._init_token_ledger()
        self.boot_seconds = round(time.perf_counter() - boot, 2)
        metric_device = self._gpu_metric_device()
        self.resident_vram_mib = round(torch.cuda.memory_allocated(metric_device)/MiB, 2) if metric_device is not None else 0.0
        if metric_device is not None and torch.cuda.is_available():
            try:
                from .runtime_config import _query_physical_avail_vram
                q_avail, _ = _query_physical_avail_vram(metric_device, self.plan)
                if q_avail is not None:
                    self._idle_free_vram_bytes = q_avail
            except Exception:
                pass
        if verbose:
            policy = "Common CPU/GPU start" if getattr(self.engine.progressive, "gpu_enabled", False) else "CPU streaming"
            print(f"[Quintessentia] {policy} | {self.model_id} ({fmt}); work follows device readiness", flush=True)

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    def expert_cache_stats(self) -> dict[str, Any] | None:
        q1_runner = getattr(self, "_q1_cubes_runner", None)
        if q1_runner is not None and q1_runner.moe_cache is not None:
            return q1_runner.moe_cache.stats()
        src = getattr(self, "model_source", None)
        if src is not None and hasattr(src, "expert_cache_stats"):
            return src.expert_cache_stats()
        return None

    # ------------------------------------------------------------------ state management

    def _gpu_metric_device(self):
        if self.engine.device.type == "cuda":
            return self.engine.device
        progressive = getattr(self.engine, "progressive", None)
        return (progressive.gpu if progressive is not None and progressive.gpu_enabled and progressive.error is None
                and progressive.plan is not None else None)

    def _sync_compute(self):
        self.device = self.engine.device
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _reset_compute_peaks(self):
        self.device = self.engine.device
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _release_transient_state(self) -> None:
        """Drops per-generation tensors (MTP cache, captured hidden states, snapshots) and returns VRAM."""
        if self.mtp_head is not None:
            self.mtp_head.reset_cache()
        if self.speculator is not None:
            self.speculator.reset()
        if self.snap_mgr is not None:
            self.snap_mgr.active_capture = False
        if self.captured:
            self.captured.pop("post_norm", None)
        q1_runner = getattr(self, "_q1_cubes_runner", None)
        if q1_runner is not None and hasattr(q1_runner, "clear_mirror_anchors"):
            q1_runner.clear_mirror_anchors()
        gc.collect()
        if self.engine is not None and self.engine.device.type == "cuda":
            torch.cuda.empty_cache()
            from .kv_offload import empty_host_pinned_cache
            empty_host_pinned_cache()
            try:
                from .runtime_config import _query_physical_avail_vram
                q_avail, _ = _query_physical_avail_vram(self.engine.device, getattr(self, "plan", None))
                if q_avail is not None:
                    self._idle_free_vram_bytes = q_avail
            except Exception:
                pass

    def _effective_prefill_chunk(self, remaining_tokens: int) -> int:
        """Adapts the prefill chunk size via calibrated workspace need + RSSF EC/FDS/8-force balance."""
        base = max(1, int(self.prefill_chunk_tokens))
        if remaining_tokens <= 128 or self.engine is None or getattr(self.engine, "device", None) is None:
            return base
        gpu_dev = self._gpu_metric_device()
        is_cuda = bool(gpu_dev is not None and gpu_dev.type == "cuda" and torch.cuda.is_available())
        try:
            advised = base
            free_b: int
            total_b: int
            sec_free_b: int | None = None
            if is_cuda:
                from .runtime_config import _query_physical_avail_vram
                cuda_free_b, total_b = torch.cuda.mem_get_info(gpu_dev)
                phys_free_b, phys_total_b = _query_physical_avail_vram(gpu_dev, getattr(self, "plan", None))
                free_b = int(phys_free_b if phys_free_b is not None else cuda_free_b)
                if phys_total_b and phys_total_b > 0:
                    total_b = int(phys_total_b)
                sec_free_b = int(cuda_free_b)
                cal = getattr(self, "calibration", None)
                if isinstance(cal, dict) and cal.get("ok") and isinstance(cal.get("chunk_workspace_mib"), (int, float)):
                    # Keep full calibrated chunk whenever physical free VRAM covers measured workspace + safety guard.
                    ws_b = max(64 * MiB, int(float(cal["chunk_workspace_mib"]) * MiB))
                    need_full = int(ws_b * 1.5) + 96 * MiB
                    need_half = int(ws_b * 0.9) + 64 * MiB
                    if free_b < need_half:
                        advised = min(base, 128)
                    elif free_b < need_full:
                        advised = min(base, 256)
                else:
                    if free_b < 256 * MiB:
                        advised = min(base, 128)
                    elif free_b < 512 * MiB:
                        advised = min(base, 256)
            else:
                import psutil
                vm = psutil.virtual_memory()
                free_b, total_b = int(vm.available), int(vm.total)
            # Overlay scale-invariant EC Volterra + FDS T_end + 8-force balance (GPU & CPU alike!)
            gb = getattr(self, "generation_balance", None)
            if gb is not None and total_b > 0:
                reserve_b = max(192 * MiB, int(total_b * 0.05))
                rssf_chunk, _bal, _ec = gb.advise_prefill_chunk(
                    base_chunk=advised,
                    remaining_tokens=remaining_tokens,
                    free_bytes=free_b,
                    total_bytes=total_b,
                    reserve_bytes=reserve_b,
                    secondary_free_bytes=sec_free_b,
                )
                advised = min(advised, max(128, rssf_chunk))
            return advised
        except Exception:
            pass
        return base

    def _abort_cleanup(self) -> None:
        """Restores a clean engine after a cancelled or failed generation (no VRAM or busy ring slot may stay behind)."""
        self.session.reset()
        self._session_messages.clear()
        self._session_prompt_messages.clear()
        self._session_last_tool_calls.clear()
        self._session_tools_sig = ""
        eng = self.engine
        if eng is not None:
            if hasattr(eng, "reset_after_cancel"):
                eng.reset_after_cancel()
            elif getattr(eng, "active_slot_per_layer", None):
                for i, slot in enumerate(eng.active_slot_per_layer):
                    if slot is not None:
                        try:
                            eng.pipeline.release_slot_after_compute(slot)
                        except Exception:
                            pass
                        eng.active_slot_per_layer[i] = None
            if hasattr(eng, "_anchor_ring_reset"):
                eng._anchor_ring_reset()
        self._release_transient_state()

    def enable_unified_memory(self, **kwargs):
        """Configure or override the unified working-memory arena before generation starts."""
        if self.generations_started:
            raise ValueError('Enable unified memory before generation')
        return self.engine.enable_unified_memory(**kwargs)

    def _init_token_ledger(self) -> None:
        if not getattr(self, "enable_token_ledger", True):
            self.token_ledger = None
            return
        try:
            tok = self.tokenizer
        except Exception:
            tok = None
        if tok is None:
            self.token_ledger = None
            return
        from .token_ledger import DiskTokenLedger, compute_tokenizer_fingerprint

        base_dir = (
            Path(self._token_ledger_dir)
            if getattr(self, "_token_ledger_dir", None) is not None
            else (self.root / "sessions")
        )
        max_ctx = int(getattr(getattr(self, "plan", None), "max_context_tokens", 0) or 0) or None
        self.token_ledger = DiskTokenLedger(
            storage_dir=base_dir,
            model_id=getattr(self, "model_id", "model"),
            tokenizer_fingerprint=compute_tokenizer_fingerprint(tok),
            session_id=getattr(self, "session_id", "default"),
            enabled=True,
            max_active_tokens=max_ctx,
        )

    def reset_session(self, *, clear_ledger: bool = True) -> None:
        self.session.reset()
        self._session_messages.clear()
        self._session_prompt_messages.clear()
        self._session_last_tool_calls.clear()
        self._session_tools_sig = ""
        self._session_thought_anchor = None
        self._last_ledger_recovery = {
            "kind": "fresh",
            "matched_turn_index": -1,
            "disk_prefix_tokens": 0,
            "kv_reused_tokens": 0,
        }
        warmup = getattr(self.engine, "probability_warmup", None)
        if warmup is not None:
            warmup.reset()
        if getattr(self, "survivor_tracker", None) is not None:
            self.survivor_tracker.reset()
        if getattr(self, "generation_balance", None) is not None:
            self.generation_balance.reset_session()
        if clear_ledger and getattr(self, "token_ledger", None) is not None:
            self.token_ledger.clear_active_window(reason="session_reset")
        self._release_transient_state()

    def compact_session(
        self,
        summary_text: str,
        *,
        keep_last_turns: int = 1,
        system_prompt: str | None = None,
        max_active_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Archive the current epoch in `ledger.jsonl` and rewrite `active_window.json` with a compact epoch.

        Stores the exact token sequence (`[system] + [summary] + [last K turns]`) sized to fit the
        active context so subsequent turns (and any server restart) retokenize 100% deterministically.
        """
        summary = str(summary_text or "").strip()
        if not summary:
            raise ValueError("summary_text must be non-empty")
        from .tool_calling import normalize_messages

        existing_msgs = list(getattr(self, "_session_messages", []) or [])
        if not existing_msgs and getattr(self, "token_ledger", None) is not None and self.token_ledger.turns:
            existing_msgs = list(self.token_ledger.turns[-1].get("messages") or [])
        existing_msgs = normalize_messages(existing_msgs, parse_tool_args_as_dict=False)

        compacted_msgs: list[dict[str, Any]] = []
        if system_prompt is not None and str(system_prompt).strip():
            compacted_msgs.append({"role": "system", "content": str(system_prompt).strip()})
        else:
            for m in existing_msgs:
                if m.get("role") == "system":
                    compacted_msgs.append(m)
                    break

        compacted_msgs.append({"role": "user", "content": f"[Context Summary]\n{summary}"})
        compacted_msgs.append({"role": "assistant", "content": "Understood. Continuing from the compacted context."})

        keep_k = max(0, int(keep_last_turns))
        if keep_k > 0:
            non_sys = [m for m in existing_msgs if m.get("role") != "system"]
            # Collect complete trailing (user .. assistant) turn pairs
            tail_pairs: list[dict[str, Any]] = []
            i = len(non_sys) - 1
            pairs = 0
            while i >= 1 and pairs < keep_k:
                if non_sys[i].get("role") == "assistant" and non_sys[i - 1].get("role") == "user":
                    tail_pairs = [non_sys[i - 1], non_sys[i]] + tail_pairs
                    pairs += 1
                    i -= 2
                else:
                    i -= 1
            compacted_msgs.extend(tail_pairs)

        # Render exact closed assistant prefix (no trailing generation prompt) for bit-exact ledger storage
        tok = self.tokenizer
        rendered: str | None = None
        if getattr(tok, "chat_template", None):
            try:
                rendered = tok.apply_chat_template(
                    compacted_msgs, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                rendered = None
        if rendered is None:
            parts: list[str] = []
            for m in compacted_msgs:
                r = m["role"]
                c = m.get("content", "")
                if r == "system":
                    parts.append(f"System: {c}\n\n")
                elif r == "assistant":
                    parts.append(f"Assistant:{c}")
                else:
                    parts.append(f"User: {c}\n\n")
            rendered = "".join(parts)

        compacted_ids = tok(rendered, return_tensors="pt").input_ids[0].tolist()
        eos_ids = self._effective_eos_token_ids()
        eos_ended = bool(compacted_ids and compacted_ids[-1] in eos_ids)

        # Drop old parked RAM KV (prefix changed to the new compact epoch), keep disk archive intact
        self.session.reset()
        self._session_messages = list(compacted_msgs)
        self._session_prompt_messages = list(compacted_msgs[:-1]) if compacted_msgs else []
        self._session_last_tool_calls.clear()
        self._session_thought_anchor = None
        self._release_transient_state()

        if getattr(self, "token_ledger", None) is not None:
            report = self.token_ledger.compact_active_window(
                compacted_messages=compacted_msgs,
                compacted_token_ids=compacted_ids,
                eos_ended=eos_ended,
                tools_sig=getattr(self, "_session_tools_sig", ""),
                reason="compact_session",
                max_active_tokens=max_active_tokens,
            )
        else:
            report = {
                "closed_epoch": 0,
                "new_epoch": 1,
                "old_tokens": 0,
                "new_tokens": len(compacted_ids),
                "trimmed_to_cap": False,
                "active_path": None,
                "archive_path": None,
            }
        report["compacted_messages"] = compacted_msgs
        return report

    def set_active_q1_planes(self, k: int, *, lm_head_k: int | None = None) -> dict[str, Any]:
        """Instant 0-ms live switching of active Q1 bit-plane depth (`1 <= k <= q1_planes_loaded`) without reloading."""
        if self.profile.get("weight_format") != "q1_cubes":
            raise ValueError(
                f"Live Q1 plane switching requires weight_format='q1_cubes' (current: {self.profile.get('weight_format')!r})"
            )
        loaded = max(1, int(getattr(self, "q1_planes_loaded", getattr(self, "q1_planes", 5))))
        target_k = int(k)
        if not 1 <= target_k <= loaded:
            raise ValueError(f"k must be between 1 and {loaded} (loaded planes), got {target_k}")

        from src.binary_halving_q1 import BinaryHalvingQ1Linear, BinaryHalvingMoEExperts

        updated = 0
        lm_updated = False
        model = getattr(self.engine, "model", None)
        if model is not None:
            lm_ref = getattr(model, "lm_head", None)
            for mod in model.modules():
                is_q1 = isinstance(mod, (BinaryHalvingQ1Linear, BinaryHalvingMoEExperts)) or (
                    hasattr(mod, "active_cubes")
                    and hasattr(mod, "root_bits")
                    and getattr(mod, "packed_lo", None) is not None
                )
                if is_q1:
                    if mod is lm_ref:
                        if lm_head_k is not None:
                            lk = max(1, min(int(mod.root_bits), int(lm_head_k)))
                            mod.active_cubes = lk
                            self.q1_lm_head_planes = lk
                            lm_updated = True
                        else:
                            mod.active_cubes = min(int(mod.root_bits), max(target_k, int(self.q1_lm_head_planes or target_k)))
                    else:
                        mod.active_cubes = min(int(mod.root_bits), target_k)
                    updated += 1

        runner = getattr(self, "_q1_cubes_runner", None)
        if runner is not None:
            for attr in ("active_cubes", "n_cubes"):
                if hasattr(runner, attr):
                    try:
                        setattr(runner, attr, target_k)
                    except Exception:
                        pass
            moe_cache = getattr(runner, "moe_cache", None)
            if moe_cache is not None:
                for attr in ("active_cubes", "n_cubes", "moe_cubes"):
                    if hasattr(moe_cache, attr):
                        try:
                            setattr(moe_cache, attr, target_k)
                        except Exception:
                            pass

        self.q1_planes = target_k
        self.profile["q1_planes"] = target_k
        return {
            "q1_planes": target_k,
            "q1_planes_loaded": loaded,
            "cube_root_bits": int(getattr(self, "cube_root_bits", 8)),
            "lm_head_planes": self.q1_lm_head_planes,
            "lm_head_updated": lm_updated,
            "updated_modules": updated,
        }

    def calibrate_q1_planes_by_kl(
        self,
        *,
        kl_target: float = 0.05,
        min_k: int | None = None,
        probe_text: str | None = None,
    ) -> dict[str, Any]:
        """Select the smallest active Q1 plane depth `k` whose mean KL divergence vs the full
        loaded depth (`q1_planes_loaded`) stays at or below `kl_target` on a calibration passage.

        Never touches `embed_tokens` (stays native BF16) and switches `active_cubes` in-place in O(1).
        """
        if self.profile.get("weight_format") != "q1_cubes":
            return {"ok": False, "reason": "not_q1_cubes"}
        k_ref = max(1, int(getattr(self, "q1_planes_loaded", getattr(self, "q1_planes", 5))))
        k_lo = max(1, min(k_ref, int(min_k if min_k is not None else getattr(self, "min_q1_planes", 3))))
        if k_lo >= k_ref:
            return {"ok": True, "reference_k": k_ref, "chosen_k": k_ref, "chosen_kl": 0.0, "kl_target": kl_target, "curve": {}}

        text = probe_text or (
            "The capital of France is Paris, renowned for the Eiffel Tower and the Louvre Museum. "
            "In computer architecture, memory bandwidth often limits autoregressive decoding because "
            "each generated token reads billions of parameters from graphics memory. Binary halving "
            "bit-planes allow selecting the minimal precision depth k that preserves model logits "
            "without distorting outlier channels or degrading downstream reasoning accuracy. "
            "Meanwhile, exact disk-backed token ledgers eliminate BPE whitespace drift across turns."
        )
        tok = self.tokenizer
        ids = tok(text, return_tensors="pt").input_ids
        if ids.shape[1] < 8:
            return {"ok": False, "reason": "probe_too_short"}
        dev = self.engine.device
        ids = ids[:, : min(256, int(ids.shape[1]))].to(dev)

        import torch.nn.functional as F

        self.set_active_q1_planes(k_ref)
        with torch.no_grad():
            out_ref = self.engine.model(input_ids=ids, use_cache=False)
            lg_ref = out_ref.logits[0, :-1].float()
            lp_ref = F.log_softmax(lg_ref, dim=-1)
            p_ref = lp_ref.exp()
            top1_ref = lg_ref.argmax(dim=-1)
            del out_ref, lg_ref

        curve: dict[str, dict[str, float]] = {}
        chosen_k = k_ref
        chosen_kl = 0.0
        for k_cand in range(k_lo, k_ref):
            self.set_active_q1_planes(k_cand)
            with torch.no_grad():
                out_c = self.engine.model(input_ids=ids, use_cache=False)
                lg_c = out_c.logits[0, :-1].float()
                lp_c = F.log_softmax(lg_c, dim=-1)
                mean_kl = float((p_ref * (lp_ref - lp_c)).sum(dim=-1).mean().item())
                top1_pct = float((lg_c.argmax(dim=-1) == top1_ref).float().mean().item()) * 100.0
                del out_c, lg_c, lp_c
            curve[f"k={k_cand}"] = {"mean_kl": round(mean_kl, 5), "top1_match_pct": round(top1_pct, 2)}
            if mean_kl <= float(kl_target):
                chosen_k = k_cand
                chosen_kl = mean_kl
                break

        self.set_active_q1_planes(chosen_k)
        del lp_ref, p_ref, top1_ref, ids
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "ok": True,
            "kl_target": float(kl_target),
            "reference_k": k_ref,
            "chosen_k": chosen_k,
            "chosen_kl": round(chosen_kl, 5),
            "curve": curve,
        }

    def runtime_limits(self, *, busy: bool = False) -> dict[str, Any]:
        return compute_runtime_limits(self, busy=busy)

    def calibrate_hardware_limits(
        self,
        *,
        probe_tokens: int = 512,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run a short 1-chunk prefill + decode hardware probe (never fills memory to OOM).

        Measures real post-weights VRAM/RAM headroom, real chunk activation workspace (`M_chunk`),
        real prefill/decode throughput, and RAM->GPU PCIe bandwidth on the user's machine, then
        releases all temporary buffers so the GPU stays clean for user requests.
        """
        if self.engine is None or self.tokenizer is None:
            return {"ok": False, "reason": "engine_not_ready"}

        t_all = time.perf_counter()
        dev = self.engine.device
        is_cuda = bool(dev.type == "cuda" and torch.cuda.is_available())
        pcie_gb_s: float | None = None
        if is_cuda:
            try:
                buf_cpu = torch.empty((4 * 1024 * 1024,), dtype=torch.float32, pin_memory=True)
                buf_gpu = torch.empty_like(buf_cpu, device=dev)
                buf_gpu.copy_(buf_cpu, non_blocking=True)
                torch.cuda.synchronize(dev)
                t_c0 = time.perf_counter()
                for _ in range(4):
                    buf_gpu.copy_(buf_cpu, non_blocking=True)
                torch.cuda.synchronize(dev)
                dt_c = max(1e-6, time.perf_counter() - t_c0)
                pcie_gb_s = round((4 * buf_cpu.nbytes / 1e9) / dt_c, 2)
                del buf_gpu, buf_cpu
                torch.cuda.empty_cache()
            except Exception:
                pcie_gb_s = None

        # Warm up lazy Q1 cube carving / CUDA kernels with a tiny 12-token pass so the 512-token
        # measurement reflects steady-state chunk workspace and speed (and keeps first user turn instant).
        try:
            for _ in self.stream_chat(
                [{"role": "user", "content": "OK"}],
                max_new_tokens=2,
                enable_thinking=False,
                use_session_cache=False,
                stop_on_eos=True,
                temperature=0.0,
            ):
                pass
            self.reset_session(clear_ledger=False)
            if is_cuda:
                torch.cuda.synchronize(dev)
                torch.cuda.empty_cache()

            base_alloc_mib = (
                round(torch.cuda.memory_allocated(dev) / MiB, 2) if is_cuda else 0.0
            )
            target_probe = max(64, min(int(probe_tokens), int(self.prefill_chunk_tokens or 512)))
            # Build a deterministic ~target_probe-token prompt without exceeding 1 chunk.
            unit = "Quintessentia hardware probe: measure chunk workspace, prefill throughput, and memory headroom. "
            u_ids = self.tokenizer.encode(unit, add_special_tokens=False) or [1]
            reps = max(1, (target_probe - 24) // max(1, len(u_ids)))
            probe_text = (unit * reps).strip()

            done_metrics: dict[str, Any] = {}
            for ev in self.stream_chat(
                [{"role": "user", "content": probe_text}],
                max_new_tokens=6,
                enable_thinking=False,
                use_session_cache=False,
                stop_on_eos=False,
                temperature=0.0,
            ):
                if ev.get("type") == "done":
                    done_metrics = dict(ev.get("metrics") or {})

            self.reset_session(clear_ledger=False)
            if is_cuda:
                torch.cuda.synchronize(dev)
                torch.cuda.empty_cache()

            post_alloc_mib = round(torch.cuda.memory_allocated(dev) / MiB, 2) if is_cuda else 0.0
            peak_mib = float(done_metrics.get("peak_vram_mib") or post_alloc_mib)
            n_probe = int(done_metrics.get("prompt_tokens") or target_probe)
            prefill_ms = float(done_metrics.get("prefill_ms") or 0.0)
            dec_tok_s = float(done_metrics.get("decode_tok_per_s") or 0.0)
            pre_tok_s = round(n_probe / max(1e-3, prefill_ms / 1000.0), 1) if prefill_ms > 0 else 0.0

            from .runtime_config import (
                _extract_text_config_dict,
                _kv_geometry_from_config_dict,
                _query_physical_avail_vram,
            )
            cfg = _extract_text_config_dict(self.engine) or _extract_text_config_dict(self)
            kv_per_tok, rec_b, _ = _kv_geometry_from_config_dict(cfg)
            probe_kv_mib = ((n_probe * (kv_per_tok or 0)) + rec_b) / MiB
            chunk_ws_mib = round(max(48.0, peak_mib - max(base_alloc_mib, post_alloc_mib) - probe_kv_mib), 1)

            phys_free_mib: float | None = None
            if is_cuda:
                q_avail, _ = _query_physical_avail_vram(dev, getattr(self, "plan", None))
                if q_avail is not None:
                    self._idle_free_vram_bytes = q_avail
                    phys_free_mib = round(q_avail / MiB, 1)

            self.calibration = {
                "ok": True,
                "probe_tokens": n_probe,
                "base_vram_mib": post_alloc_mib,
                "physical_free_vram_mib": phys_free_mib,
                "chunk_workspace_mib": chunk_ws_mib,
                "prefill_tok_per_s": pre_tok_s,
                "decode_tok_per_s": round(dec_tok_s, 2),
                "pcie_h2d_gb_s": pcie_gb_s,
                "elapsed_s": round(time.perf_counter() - t_all, 2),
            }
            lim = self.runtime_limits(busy=False)
            self.calibration["fast_vram_context_length"] = lim.get("fast_vram_context_length")
            self.calibration["effective_context_length"] = lim.get("effective_context_length")
            if verbose:
                pcie_str = f" | PCIe H2D: {pcie_gb_s} GB/s" if pcie_gb_s else ""
                print(
                    f"[Quintessentia] Calibrated ({n_probe} tok in {self.calibration['elapsed_s']}s): "
                    f"chunk_ws={chunk_ws_mib} MiB | prefill={pre_tok_s} tok/s | decode={round(dec_tok_s, 2)} tok/s"
                    f"{pcie_str} | fast_vram_ctx={lim.get('fast_vram_context_length')} | hybrid_ctx={lim.get('effective_context_length')}",
                    flush=True,
                )
            return dict(self.calibration)
        except Exception as exc:
            self.reset_session(clear_ledger=False)
            self.calibration = {"ok": False, "error": str(exc)}
            return dict(self.calibration)

    def status_snapshot(self) -> dict[str, Any]:
        self.device = self.engine.device
        metric_device = self._gpu_metric_device()
        alloc_mib = round(torch.cuda.memory_allocated(metric_device) / MiB, 2) if metric_device is not None else 0.0
        snap = {
            "model_id": self.model_id,
            "model_dir": str(self.model_dir),
            "model_format": self.model_format,
            "backend": self.profile.get("backend"),
            "weight_format": self.profile.get("weight_format"),
            "execution_device": str(self.device),
            "fast_qwen": self.fast_qwen,
            "gpu_name": self.gpu_name,
            "total_vram_gib": self.total_vram_gib,
            "vram_mode": self.vram_mode,
            "vram_plan": self.plan.as_dict(),
            "allocated_vram_mib": alloc_mib,
            "startup_memory_probe": getattr(self, "memory_probe", None),
            "mtp_enabled": self.mtp_enabled,
            "mtp_vram_mib": self.mtp_vram_mib,
            "vocab_pruner_enabled": self.enable_vocab_pruner,
            "prefill_wavefront_enabled": self.prefill_wavefront,
            "parked_session_tokens": len(self.session.fed),
            "parked_session_ram_mib": round(self.session.parked_bytes / MiB, 2),
            "generations_started": self.generations_started,
            "generations_cancelled": self.generations_cancelled,
            "last_metrics": self.last_metrics,
        }
        ec = self.expert_cache_stats()
        if ec is not None:
            snap["expert_cache"] = ec
        snap["unified_primary_engine"] = True
        snap["thought_anchor_enabled"] = bool(
            self.thought_anchor_enabled or getattr(self, "_active_thought_anchor", None) is not None
        )
        progressive = getattr(self.engine, "progressive", None)
        if progressive is not None:
            snap["progressive_execution"] = progressive.stats()
            spec_src = self.speculator.stats().get("source") if self.speculator is not None else None
            snap["mtp_status"] = (
                "native head on CPU; shared streamed verification" if self.mtp_enabled
                else "q1 self-speculation" if spec_src == "q1_skeleton"
                else "prompt lookup" if self.speculator is not None else "disabled"
            )
        if self.speculator is not None:
            snap["speculation"] = self.speculator.stats()
        if self._region_controller is not None:
            snap["route_exclusion"] = self._region_controller.stats()
        if getattr(self, "survivor_tracker", None) is not None:
            snap["survivor_probability_path"] = self.survivor_tracker.stats()
        if getattr(self, "stanok_contour", None) is not None:
            snap["stanok_contour"] = self.stanok_contour.stats()
        if getattr(self, "probability_bus", None) is not None:
            snap["probability_bus"] = self.probability_bus.stats()
        if getattr(self, "cold_warmup_scale", None) is not None:
            snap["cold_warmup_v4"] = self.cold_warmup_scale.stats()
        q1_arena = getattr(self.engine, "q1_arena", None)
        if q1_arena is not None:
            snap["q1_role_arena"] = q1_arena.stats()
        q1_lh = _q1_lm_head_telemetry(self.engine.model)
        if q1_lh is not None:
            snap["q1_lm_head"] = q1_lh
        puz_ctrl = getattr(self.engine, "puzzle_controller", None)
        if puz_ctrl is not None:
            snap["q1_puzzle_overlay"] = puz_ctrl.stats()
        q1_cubes_runner = getattr(self, "_q1_cubes_runner", None)
        if q1_cubes_runner is not None:
            snap["q1_cubes"] = q1_cubes_runner.stats()
            snap["q1_planes"] = getattr(self, "q1_planes", 4)
        if self.profile.get("weight_format") == "q1_cubes":
            snap["q1_planes"] = int(getattr(self, "q1_planes", 5))
            snap["q1_planes_loaded"] = int(getattr(self, "q1_planes_loaded", snap["q1_planes"]))
            snap["cube_root_bits"] = int(getattr(self, "cube_root_bits", 8))
            snap["q1_lm_head_planes"] = getattr(self, "q1_lm_head_planes", None)
            snap["auto_fit_q1_planes"] = bool(getattr(self, "auto_fit_q1_planes", False))
            if getattr(self, "q1_kl_calibration", None) is not None:
                snap["q1_kl_calibration"] = self.q1_kl_calibration
        if self.profile.get("weight_format") == "q1":
            snap["q1_planes"] = getattr(self, "q1_planes", 4)
            snap["q1_early_stop_enabled"] = getattr(self, "q1_early_stop", True)
        from src.quantized_projection import projection_stats
        snap["packed_projections"] = projection_stats(self.engine.model)
        projection_io = getattr(self.model_source, "projection_io_stats", None)
        if callable(projection_io):
            snap["projection_io"] = projection_io()
        warmup = getattr(self.engine, "probability_warmup", None)
        if warmup is not None:
            snap["probability_warmup"] = warmup.stats()
        if getattr(self, "calibration", None) is not None:
            snap["calibration"] = self.calibration
        if getattr(self, "token_ledger", None) is not None:
            snap["token_ledger"] = self.token_ledger.stats()
        snap["rssf_balance_enabled"] = bool(getattr(self, "enable_rssf_balance", True))
        snap["limits"] = self.runtime_limits()
        return snap

    # ------------------------------------------------------------------ prompt preparation

    def _effective_eos_token_ids(self) -> set[int]:
        cached = getattr(self, "_cached_effective_eos_ids", None)
        if cached is not None:
            return cached
        cfg = getattr(getattr(self, "engine", None), "config", None)
        mdir = getattr(self, "model_dir", None)
        eos = _resolve_effective_eos_ids(self.tokenizer, config=cfg, model_dir=mdir)
        self._cached_effective_eos_ids = eos
        return eos

    def _prepare_prompt_and_cache(
        self,
        messages: list[dict[str, Any]],
        enable_thinking: bool,
        use_session_cache: bool,
        max_new_tokens: int = 0,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> tuple[list[int], int, Any, float]:
        from .kv_offload import plan_kv_cache
        from .token_ledger import canonical_message_sig, canonical_message_sigs
        from .tool_calling import (
            apply_chat_template_with_tools,
            normalize_messages,
            render_incremental_turn_suffix,
            tool_calls_match_signature,
            tools_signature,
        )

        eos_ids = self._effective_eos_token_ids()
        norm_msgs = normalize_messages(messages, parse_tool_args_as_dict=False)
        cur_tools_sig = tools_signature(tools, tool_choice)
        ledger = getattr(self, "token_ledger", None)
        self._last_ledger_recovery = {
            "kind": "fresh",
            "matched_turn_index": -1,
            "disk_prefix_tokens": 0,
            "kv_reused_tokens": 0,
            "epoch": int(ledger.epoch) if ledger is not None else 0,
        }
        has_parked = (
            use_session_cache
            and cur_tools_sig == getattr(self, "_session_tools_sig", "")
            and len(self.session.fed) > 0
            and self.session.parked_backbone is not None
        )

        def _unpark_for(full_ids: list[int], reused: int) -> tuple[Any, float]:
            target_len = len(full_ids) + min(2048, max(0, int(max_new_tokens)))
            use_managed, n_offload, disable_pf = plan_kv_cache(self, target_len, device=self.device)
            return self.session.unpark_to_gpu(
                self.device,
                self.mtp_head,
                first_new_token=full_ids[reused],
                offload_kv=use_managed,
                offload_layer_count=n_offload,
                disable_prefetch=disable_pf,
                initial_capacity=target_len,
                chunk_pad=max(1, int(self.prefill_chunk_tokens)),
            )

        def _render_suffix(tail_msgs: list[dict[str, Any]], prev_ended_eos: bool, chatml_fmt: bool) -> str | None:
            s = render_incremental_turn_suffix(
                self.tokenizer,
                tail_msgs,
                enable_thinking=enable_thinking,
                prev_ended_with_eos=prev_ended_eos,
            )
            if s is None and chatml_fmt and len(tail_msgs) == 1 and tail_msgs[0]["role"] == "user":
                lead = "" if prev_ended_eos else "<|im_end|>"
                think_tail = "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
                s = (
                    f"{lead}\n<|im_start|>user\n{tail_msgs[0]['content']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{think_tail}"
                )
            return s

        # Path 1: Fast RAM continuation (whitespace- & `<think>`-tolerant canonical comparison!)
        if has_parked:
            prev_prompt_msgs = getattr(self, "_session_prompt_messages", [])
            prev_n = len(prev_prompt_msgs)
            prev_tool_calls = getattr(self, "_session_last_tool_calls", [])
            prev_asst_msg = (
                self._session_messages[-1]
                if (self._session_messages and self._session_messages[-1].get("role") == "assistant")
                else None
            )
            is_direct_continuation = False
            tail_msgs: list[dict[str, Any]] = []
            if (
                prev_n > 0
                and len(norm_msgs) >= prev_n + 2
                and canonical_message_sigs(norm_msgs[:prev_n]) == canonical_message_sigs(prev_prompt_msgs)
            ):
                asst_msg = norm_msgs[prev_n]
                cand_tail = norm_msgs[prev_n + 1 :]
                if asst_msg.get("role") == "assistant" and cand_tail and all(
                    m.get("role") in ("tool", "user") for m in cand_tail
                ):
                    if asst_msg.get("tool_calls") and prev_tool_calls:
                        is_direct_continuation = tool_calls_match_signature(
                            asst_msg.get("tool_calls"), prev_tool_calls
                        )
                    elif prev_asst_msg is not None and canonical_message_sig(asst_msg) == canonical_message_sig(prev_asst_msg):
                        is_direct_continuation = True
                    if is_direct_continuation:
                        tail_msgs = cand_tail
            elif (
                not tools
                and len(norm_msgs) >= 3
                and norm_msgs[-1]["role"] == "user"
                and canonical_message_sigs(norm_msgs[:-1]) == canonical_message_sigs(self._session_messages)
            ):
                is_direct_continuation = True
                tail_msgs = [norm_msgs[-1]]

            if is_direct_continuation and tail_msgs:
                prev_full = self.session.fed + self.session.unfed_tail
                prev_ended_eos = bool(prev_full and prev_full[-1] in eos_ids)
                suffix_str = _render_suffix(tail_msgs, prev_ended_eos, self.session.chatml_think_format)
                if suffix_str:
                    suffix_ids = self.tokenizer(suffix_str, add_special_tokens=False).input_ids
                    full_ids = prev_full + suffix_ids
                    self._check_context_budget(len(full_ids), max_new_tokens)
                    reused = self.session.kv_prefix_reuse(full_ids)
                    if reused > 0:
                        self._last_prompt_opened_think = suffix_str.rstrip().endswith("<think>")
                        self._last_prompt_chatml_think = (
                            self.session.chatml_think_format or "<|im_start|>assistant\n<think>" in suffix_str
                        )
                        past_kv, unpark_ms = _unpark_for(full_ids, reused)
                        self._last_ledger_recovery = {
                            "kind": "exact_continuation",
                            "matched_turn_index": (len(ledger.turns) - 1) if (ledger and ledger.turns) else -1,
                            "disk_prefix_tokens": len(prev_full),
                            "kv_reused_tokens": reused,
                            "epoch": int(ledger.epoch) if ledger is not None else 0,
                        }
                        if ledger is not None:
                            ledger.last_recovery_kind = "exact_kv_reuse"
                            ledger.last_recovered_tokens = reused
                        return full_ids, reused, past_kv, unpark_ms

        # Path 2: Disk Token Ledger recovery (handles server restarts, compacted epochs, and
        # mid-conversation edits / ordering disruptions with 100% bit-exact historical token IDs
        # + partial pure-attention KV prefix truncation when available!).
        if use_session_cache and ledger is not None and ledger.enabled:
            lm = ledger.match_messages(norm_msgs, tools_sig=cur_tools_sig)
            if lm.kind in ("exact_continuation", "partial_turn_prefix") and lm.tail_messages:
                suffix_str = _render_suffix(
                    lm.tail_messages, lm.prev_ended_with_eos, ledger.chatml_think_format
                )
                if suffix_str:
                    suffix_ids = self.tokenizer(suffix_str, add_special_tokens=False).input_ids
                    full_ids = list(lm.prefix_token_ids) + list(suffix_ids)
                    self._check_context_budget(len(full_ids), max_new_tokens)
                    self._last_prompt_opened_think = suffix_str.rstrip().endswith("<think>")
                    self._last_prompt_chatml_think = (
                        ledger.chatml_think_format or "<|im_start|>assistant\n<think>" in suffix_str
                    )
                    reused = 0
                    if len(self.session.fed) > 0 and self.session.parked_backbone is not None:
                        reused = self.session.kv_prefix_reuse(full_ids, allow_truncate=True)
                    if reused > 0:
                        past_kv, unpark_ms = _unpark_for(full_ids, reused)
                        rec_label = (
                            "partial_kv_truncate"
                            if lm.kind == "partial_turn_prefix"
                            else "exact_kv_reuse"
                        )
                        self._last_ledger_recovery = {
                            "kind": lm.kind,
                            "recovery_mode": rec_label,
                            "matched_turn_index": lm.matched_turn_index,
                            "disk_prefix_tokens": len(lm.prefix_token_ids),
                            "kv_reused_tokens": reused,
                            "epoch": lm.epoch,
                        }
                        ledger.last_recovery_kind = rec_label
                        ledger.last_recovered_tokens = len(lm.prefix_token_ids)
                        return full_ids, reused, past_kv, unpark_ms
                    # RAM KV unavailable (restart / compacted epoch / recurrent layer rewind):
                    # still replay 100% bit-exact historical token IDs from disk with zero BPE drift!
                    self.session.reset()
                    self._last_ledger_recovery = {
                        "kind": lm.kind,
                        "recovery_mode": "ledger_exact_retokenize",
                        "matched_turn_index": lm.matched_turn_index,
                        "disk_prefix_tokens": len(lm.prefix_token_ids),
                        "kv_reused_tokens": 0,
                        "epoch": lm.epoch,
                    }
                    ledger.last_recovery_kind = "ledger_exact_retokenize"
                    ledger.last_recovered_tokens = len(lm.prefix_token_ids)
                    return full_ids, 0, None, 0.0

        prompt_str = apply_chat_template_with_tools(
            self.tokenizer,
            messages,
            tools=tools,
            tool_choice=tool_choice,
            enable_thinking=enable_thinking,
        )
        self._last_prompt_opened_think = prompt_str.rstrip().endswith("<think>")
        self._last_prompt_chatml_think = "<|im_start|>assistant\n<think>" in prompt_str
        prompt_ids = self.tokenizer(prompt_str, return_tensors="pt").input_ids[0].tolist()
        self._check_context_budget(len(prompt_ids), max_new_tokens)

        can_reuse_prefix = (
            use_session_cache
            and len(self.session.fed) > 0
            and self.session.parked_backbone is not None
        )
        if can_reuse_prefix:
            reused = self.session.kv_prefix_reuse(prompt_ids, allow_truncate=False)
            if reused > 0:
                past_kv, unpark_ms = _unpark_for(prompt_ids, reused)
                self._last_ledger_recovery = {
                    "kind": "exact_continuation",
                    "recovery_mode": "exact_kv_reuse",
                    "matched_turn_index": -1,
                    "disk_prefix_tokens": 0,
                    "kv_reused_tokens": reused,
                    "epoch": int(ledger.epoch) if ledger is not None else 0,
                }
                if ledger is not None:
                    ledger.last_recovery_kind = "exact_kv_reuse"
                    ledger.last_recovered_tokens = reused
                return prompt_ids, reused, past_kv, unpark_ms

        if use_session_cache:
            self.session.reset()
        return prompt_ids, 0, None, 0.0

    # ------------------------------------------------------------------ generation core

    def _check_context_budget(self, prompt_tokens: int, max_new_tokens: int) -> None:
        # Context and generation length are caller policy, not a server admission ceiling.
        # Allocation and the model implementation determine actual executable capacity.
        if prompt_tokens < 0 or max_new_tokens < 0:
            raise ValueError("Token counts must be non-negative")

    @staticmethod
    def _make_cancel_checker(cancel_check: Callable[[], bool] | None) -> Callable[[str], None]:
        def _check(phase: str) -> None:
            if cancel_check is not None and cancel_check():
                raise GenerationCancelled(phase)

        return _check

    def _prefill(
        self,
        tail_ids: list[int],
        past_kv: Any,
        use_mtp: bool,
        check: Callable[[str], None],
        prefetch_next: bool,
    ) -> tuple[int, Any, torch.Tensor, tuple[int, float, torch.Tensor] | None]:
        """Feeds the prompt tail in chunks; only the last position is projected to the vocabulary.

        Returns ``(first_token, past, last_hidden[1,1,H], mtp_draft or None)``. The MTP head is
        committed chunk by chunk so its attention never sees the whole prompt at once either.
        """
        n = len(tail_ids)
        past = past_kv
        picked = -1
        last_h = None
        mtp_draft = None
        a = 0
        while a < n:
            check("prefill")
            cur_chunk = self._effective_prefill_chunk(n - a)
            b = min(n, a + cur_chunk)
            is_last = b == n
            lg, past = forward_step_all_logits(
                self.engine,
                torch.tensor([tail_ids[a:b]], dtype=torch.long),
                past_key_values=past,
                prefetch_next=prefetch_next if is_last else True,
                logits_to_keep=1,
            )
            h_chunk = self.captured["post_norm"].clone()
            if is_last:
                t_anchor = getattr(self, "_active_thought_anchor", None)
                if t_anchor is not None:
                    lg = t_anchor.on_prefill_end(self, past, lg)
                picked = int(lg[0, -1].argmax(dim=-1).item())
                last_h = h_chunk[:, -1:, :].clone()
            del lg
            if use_mtp:
                next_toks = tail_ids[a + 1 : b + 1] if not is_last else tail_ids[a + 1 :] + [picked]
                mtp_draft = self.mtp_head.commit_and_draft_first(h_chunk, next_toks)
            del h_chunk
            self.captured.pop("post_norm", None)
            a = b
        return picked, past, last_h, mtp_draft

    def _sample_next_token(self, logits_row: torch.Tensor, gen_history: list[int] | None = None) -> int:
        return sample_next_token(
            logits_row,
            gen_history,
            getattr(self, "_active_sampling", None),
            rng=getattr(self, "_sampling_rng", None),
        )

    def _stream_bf16(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        stop_on_eos: bool,
        cancel_check: Callable[[], bool] | None,
        cached_prefix_len: int = 0,
        past_kv: Any = None,
    ) -> Iterator[dict[str, Any]]:
        """Greedy BF16 generation for HF and GGUF checkpoints on the ring.

        Prefill chunks pass ``cache_position`` starting at ``cached_prefix_len`` so unparked
        multi-turn KV + GatedDeltaNet caches stay aligned, and every decoder layer's pre-hook
        polls ``cancel_check`` for sub-layer stop latency.
        """
        eos_ids = self._effective_eos_token_ids()
        check = self._make_cancel_checker(cancel_check)
        if not prompt_ids:
            raise ValueError("empty prompt")
        chunk = max(1, int(self.prefill_chunk_tokens))
        n = len(prompt_ids)
        start = max(0, min(int(cached_prefix_len), n - 1))
        all_ids = list(prompt_ids)
        gen_history: list[int] = []
        past = past_kv if start > 0 else None
        if past is None:
            from .kv_offload import make_offloaded_kv_cache, plan_kv_cache
            target_len = n + min(2048, max_new_tokens)
            use_managed, n_offload, disable_pf = plan_kv_cache(
                self, target_len, device=self.engine.device
            )
            if use_managed:
                past = make_offloaded_kv_cache(
                    self.engine.config,
                    initial_capacity=target_len,
                    chunk_pad=chunk,
                    device=self.engine.device,
                    disable_prefetch=disable_pf,
                    offload_layer_count=n_offload,
                )
        picked = -1
        phase_ref = ["prefill"]
        self._wavefront_windows = []
        stride = chunk * (4 if self.prefill_wavefront else 1)
        if cancel_check is not None:
            self.engine._step_cancel_check = lambda: check(phase_ref[0])
        try:
            with torch.no_grad():
                self._reset_compute_peaks()
                self._sync_compute()
                t_pre0 = time.perf_counter()
                prog = getattr(self.engine, "progressive", None)
                has_moe_loader = getattr(self.model_source, "_expert_loader", None) is not None
                a = start
                hud_beacon = getattr(self, "_hud_beacon", None)
                while a < n:
                    check("prefill")
                    cur_chunk = self._effective_prefill_chunk(n - a)
                    cur_stride = cur_chunk * (4 if self.prefill_wavefront else 1)
                    step_stride = (
                        min(cur_stride, 2)
                        if (
                            prog is not None
                            and prog.gpu_enabled
                            and prog.error is None
                            and not prog.complete
                            and not prog.bundle_ready.is_set()
                            and not self.prefill_wavefront
                            and has_moe_loader
                        )
                        else cur_stride
                    )
                    b = min(n, a + step_stride)
                    step = torch.tensor([prompt_ids[a:b]], dtype=torch.long)
                    cache_pos = torch.arange(a, b, dtype=torch.long)
                    if self.prefill_wavefront and b - a > cur_chunk and self.engine.device.type == "cuda":
                        logits, past = self.engine.prefill_wavefront(step, cur_chunk, past)
                        self._wavefront_windows.append(dict(self.engine.last_wavefront_stats))
                    else:
                        logits, past = self.engine._forward_step(
                            step,
                            past_key_values=past,
                            prefetch_next=(b < n or max_new_tokens > 1),
                            cache_position=cache_pos,
                            prefill=True, prefill_end=(b == n),
                        )
                    if b == n:
                        t_anchor = getattr(self, "_active_thought_anchor", None)
                        if t_anchor is not None:
                            logits = t_anchor.on_prefill_end(self, past, logits)
                        picked = self._sample_next_token(logits[0], gen_history)
                    del logits
                    a = b
                    if hud_beacon is not None and b < n:
                        hud_beacon.on_prefill_progress(b, n)
                all_ids.append(picked)
                gen_history.append(picked)
                self._sync_compute()
                prefill_ms = (time.perf_counter() - t_pre0) * 1000.0
                yield {"kind": "tokens", "new_ids": [picked]}

                generated = 1
                passes = 0
                cursor = n
                phase_ref[0] = "decode"
                from .kv_offload import transition_kv_cache_to_decode
                transition_kv_cache_to_decode(
                    past,
                    self,
                    decode_pad=1,
                    extra_decode_tokens=min(2048, max(256, max_new_tokens)),
                )
                self._sync_compute()
                t_dec0 = time.perf_counter()
                while generated < max_new_tokens:
                    if stop_on_eos and picked in eos_ids:
                        break
                    check("decode")
                    logits, past = self.engine._forward_step(
                        torch.tensor([[picked]], dtype=torch.long),
                        past_key_values=past,
                        prefetch_next=(generated + 1 < max_new_tokens),
                        cache_position=torch.tensor([cursor], dtype=torch.long),
                        prefill=False,
                    )
                    cursor += 1
                    t_anchor = getattr(self, "_active_thought_anchor", None)
                    if t_anchor is not None:
                        logits = t_anchor.transform_logits(logits, past, generated)
                        t_anchor.on_step_committed(past, 1)
                    picked = self._sample_next_token(logits[0], gen_history)
                    del logits
                    all_ids.append(picked)
                    gen_history.append(picked)
                    generated += 1
                    passes += 1
                    yield {"kind": "tokens", "new_ids": [picked]}
                self._sync_compute()
                dec_s = time.perf_counter() - t_dec0
        finally:
            self.engine._step_cancel_check = None

        yield {
            "kind": "final_state",
            "all_ids": all_ids,
            "past": past,
            "last_fed_h": None,
            "consumed_tokens": cursor,
            "prefill_ms": prefill_ms,
            "decode_s": dec_s,
            "decode_tokens": max(0, generated - 1),
            "backbone_passes": passes,
            "spec_gate": {"overall_accept_rate_pct": 0.0, "mtp_accept_rate_pct": 0.0},
            "kv_offloaded": bool(getattr(past, "is_quintessentia_offloaded", False)),
            "kv_gpu_layers": int(getattr(past, "gpu_kv_layer_count", 0)),
            "kv_offloaded_layers": int(getattr(past, "offloaded_kv_layer_count", 0)),
            "kv_promoted_layers": int(getattr(past, "promoted_layers_count", 0)),
        }

    def _stream_token_ids(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        stop_on_eos: bool,
        cached_prefix_len: int,
        past_kv: Any,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Iterator[dict[str, Any]]:
        if not self.fast_qwen:
            if self.speculator is not None:
                yield from self.speculator.stream(prompt_ids, max_new_tokens, stop_on_eos,
                    cached_prefix_len, past_kv, cancel_check)
                return
            yield from self._stream_bf16(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                stop_on_eos=stop_on_eos,
                cancel_check=cancel_check,
                cached_prefix_len=cached_prefix_len,
                past_kv=past_kv,
            )
            return
        eos_ids = self._effective_eos_token_ids()
        check = self._make_cancel_checker(cancel_check)
        all_ids = list(prompt_ids)
        gen_history_fq: list[int] = []
        tail_ids = prompt_ids[cached_prefix_len:]
        assert len(tail_ids) >= 1
        past = None

        if not self.mtp_enabled or self.mtp_head is None or self.draft_vocab is None:
            with torch.no_grad():
                self._reset_compute_peaks()
                self._sync_compute()
                t_pre0 = time.perf_counter()
                picked, past, last_fed_h, _ = self._prefill(
                    tail_ids, past_kv, use_mtp=False, check=check, prefetch_next=max_new_tokens > 1
                )
                all_ids.append(picked)
                gen_history_fq.append(picked)
                self._sync_compute()
                prefill_ms = (time.perf_counter() - t_pre0) * 1000.0
                yield {"kind": "tokens", "new_ids": [picked]}

                generated_count = 1
                passes = 0
                self._sync_compute()
                t_dec0 = time.perf_counter()
                while generated_count < max_new_tokens:
                    if stop_on_eos and picked in eos_ids:
                        break
                    check("decode")
                    lg, past = forward_step_all_logits(
                        self.engine,
                        torch.tensor([[picked]], dtype=torch.long),
                        past_key_values=past,
                        prefetch_next=(generated_count + 1 < max_new_tokens),
                    )
                    last_fed_h = self.captured["post_norm"][:, -1:, :].clone()
                    t_anchor = getattr(self, "_active_thought_anchor", None)
                    if t_anchor is not None:
                        lg = t_anchor.transform_logits(lg, past, generated_count)
                        t_anchor.on_step_committed(past, 1)
                    picked = self._sample_next_token(lg[0, -1], gen_history_fq)
                    del lg
                    all_ids.append(picked)
                    gen_history_fq.append(picked)
                    generated_count += 1
                    passes += 1
                    yield {"kind": "tokens", "new_ids": [picked]}
                self._sync_compute()
                dec_s = time.perf_counter() - t_dec0

            yield {
                "kind": "final_state",
                "all_ids": all_ids,
                "past": past,
                "last_fed_h": last_fed_h,
                "consumed_tokens": len(all_ids) - 1,
                "prefill_ms": prefill_ms,
                "decode_s": dec_s,
                "decode_tokens": max(0, generated_count - 1),
                "backbone_passes": passes,
                "spec_gate": {"overall_accept_rate_pct": 0.0, "mtp_accept_rate_pct": 0.0},
            }
            return

        gate = QuintessentiaSpecGate(max_drafts=self.max_drafts, force_off=False)
        self.draft_vocab.begin_request(prompt_ids, cached_prompt_tokens=cached_prefix_len)
        if cached_prefix_len == 0:
            self.mtp_head.reset_cache()

        with torch.no_grad():
            self._reset_compute_peaks()
            self._sync_compute()
            t_pre0 = time.perf_counter()
            picked, past, last_fed_h_post, mtp_draft = self._prefill(
                tail_ids, past_kv, use_mtp=True, check=check, prefetch_next=True
            )
            d1, p1, last_mtp_norm = mtp_draft
            all_ids.append(picked)
            self.draft_vocab.observe_picked_token(picked)
            self._sync_compute()
            prefill_ms = (time.perf_counter() - t_pre0) * 1000.0
            yield {"kind": "tokens", "new_ids": [picked]}

            generated_count = 1
            backbone_passes = 0
            self._sync_compute()
            t_dec0 = time.perf_counter()

            while generated_count < max_new_tokens:
                if stop_on_eos and picked in eos_ids:
                    break
                check("decode")
                rem = max_new_tokens - generated_count

                lkp = quintessentia_spec_lookup(all_ids, max_drafts=min(self.max_drafts, rem))
                if len(lkp) >= 2 or (len(lkp) == 1 and p1 < 0.45):
                    src = QuintessentiaSpecGate.SRC_LOOKUP
                    cand_drafts = lkp[: min(self.max_drafts, rem)]
                    cand_probs = [0.75] * len(cand_drafts)
                else:
                    src = QuintessentiaSpecGate.SRC_MTP
                    cand_drafts = [d1]
                    cand_probs = [p1]
                    if rem >= 2 and gate.should_draft_second_mtp(p1):
                        t_m2 = time.perf_counter()
                        d2, p2, _ = self.mtp_head.draft_more(last_mtp_norm, d1)
                        gate.record_draft_cost(src, time.perf_counter() - t_m2)
                        cand_drafts.append(d2)
                        cand_probs.append(p2)

                k_use = min(rem - 1, gate.pick_depth(src, cand_probs)) if rem > 1 else 0
                drafts = cand_drafts[:k_use]
                probs_used = cand_probs[:k_use]

                batch_tokens = [picked] + drafts
                step_t = torch.tensor([batch_tokens], dtype=torch.long)
                self.snap_mgr.active_capture = k_use > 0
                t_f0 = time.perf_counter()
                try:
                    logits_b, past = forward_step_all_logits(
                        self.engine,
                        step_t,
                        past_key_values=past,
                        prefetch_next=(generated_count + 1 < max_new_tokens),
                    )
                finally:
                    self.snap_mgr.active_capture = False
                h_post_b = self.captured["post_norm"]
                t_anchor = getattr(self, "_active_thought_anchor", None)
                if t_anchor is not None:
                    logits_b = t_anchor.transform_logits(logits_b, past, generated_count)

                row_preds = logits_b[0].argmax(dim=-1).tolist()
                del logits_b
                gate.record_forward(1 + k_use, time.perf_counter() - t_f0)
                backbone_passes += 1

                step_emitted: list[int] = []
                accepted = 0
                hit_eos = False
                for j in range(k_use):
                    if drafts[j] == row_preds[j]:
                        accepted += 1
                        all_ids.append(drafts[j])
                        step_emitted.append(drafts[j])
                        self.draft_vocab.observe_picked_token(drafts[j])
                        generated_count += 1
                        if stop_on_eos and drafts[j] in eos_ids:
                            hit_eos = True
                            break
                    else:
                        break

                judged = accepted if (accepted == k_use or hit_eos) else (accepted + 1)
                if k_use > 0:
                    gate.record_result(src, judged=judged, accepted=accepted, probs=probs_used)

                self.snap_mgr.rollback_to_accepted(past, accepted_drafts=accepted, total_drafts=k_use)
                if t_anchor is not None:
                    t_anchor.on_step_committed(past, accepted + (0 if (hit_eos or generated_count >= max_new_tokens) else 1))

                if hit_eos or generated_count >= max_new_tokens:
                    if accepted > 0:
                        self.mtp_head.commit_and_draft_first(h_post_b[:, :accepted, :].clone(), drafts[:accepted])
                    last_fed_h_post = h_post_b[:, accepted : accepted + 1, :].clone()
                    picked = all_ids[-1]
                    if step_emitted:
                        yield {"kind": "tokens", "new_ids": step_emitted}
                    break

                bonus_tok = row_preds[accepted]
                all_ids.append(bonus_tok)
                step_emitted.append(bonus_tok)
                self.draft_vocab.observe_picked_token(bonus_tok)
                generated_count += 1
                picked = bonus_tok
                last_fed_h_post = None
                yield {"kind": "tokens", "new_ids": step_emitted}

                mtp_tokens = drafts[:accepted] + [bonus_tok]
                if stop_on_eos and bonus_tok in eos_ids:
                    self.mtp_head.commit_and_draft_first(h_post_b[:, : accepted + 1, :].clone(), mtp_tokens)
                    break

                if generated_count < max_new_tokens:
                    d1, p1, last_mtp_norm = self.mtp_head.commit_and_draft_first(
                        h_post_b[:, : accepted + 1, :].clone(), mtp_tokens
                    )
                else:
                    self.mtp_head.commit_and_draft_first(h_post_b[:, : accepted + 1, :].clone(), mtp_tokens)
                del h_post_b

            self._sync_compute()
            dec_s = time.perf_counter() - t_dec0

        yield {
            "kind": "final_state",
            "all_ids": all_ids,
            "past": past,
            "last_fed_h": last_fed_h_post,
            "prefill_ms": prefill_ms,
            "decode_s": dec_s,
            "decode_tokens": max(0, generated_count - 1),
            "backbone_passes": backbone_passes,
            "spec_gate": gate.summary(),
        }

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int | None = 1024,
        enable_thinking: bool = True,
        use_session_cache: bool = False,
        stop_on_eos: bool = True,
        cancel_check: Callable[[], bool] | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        penalty_window: int = 256,
        seed: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> Iterator[dict[str, Any]]:
        """Streams UTF-8-safe `thinking` and `answer` deltas, followed by a `done` telemetry packet.

        Pass ``max_new_tokens=0`` (or ``None``) for unlimited generation up to natural EOS or the
        model's context window. ``cancel_check`` is polled before every prefill chunk, every decode
        step, and every streamed layer pre-hook; when it returns True the generation stops with
        ``GenerationCancelled``. On cancellation, errors and early close (``generator.close()``)
        every per-request tensor and ring slot is released in ``finally``.
        """
        eos_ids = self._effective_eos_token_ids()
        check = self._make_cancel_checker(cancel_check)
        self.generations_started += 1
        completed = False
        inner = None
        final_info: dict[str, Any] = {}
        auto_anchor_installed = False
        active_anchor_ref = getattr(self, "_active_thought_anchor", None)
        sampling_cfg = SamplingConfig(
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            presence_penalty=float(presence_penalty),
            penalty_window=int(penalty_window),
            seed=seed,
        )
        from src.rssf_dynamic_operators import _RSSF_ENABLED_OVERRIDE as _prev_rssf_override, set_rssf_balance_enabled
        _rssf_scoped = not bool(getattr(self, "enable_rssf_balance", True))
        if _rssf_scoped:
            set_rssf_balance_enabled(False)
        try:
            check("queue")
            if max_new_tokens is not None and int(max_new_tokens) < 0:
                raise ValueError("max_new_tokens must be >= 0 (0 = unlimited until EOS)")
            unlimited_tokens = (max_new_tokens is None) or (int(max_new_tokens) == 0)
            plan_reserve_tokens = 4096 if unlimited_tokens else int(max_new_tokens)
            had_prior_archive = bool(
                use_session_cache and len(self.session.fed) > 0 and self.session.parked_backbone is not None
            )
            prompt_ids, cached_len, past_kv, unpark_ms = self._prepare_prompt_and_cache(
                messages, enable_thinking=enable_thinking, use_session_cache=use_session_cache,
                max_new_tokens=plan_reserve_tokens, tools=tools, tool_choice=tool_choice,
            )
            gb = getattr(self, "generation_balance", None)
            if gb is not None:
                try:
                    entry_free_b, entry_total_b, entry_sec_b = 0, 1, None
                    if self.engine is not None and getattr(self.engine, "device", None) is not None and self.engine.device.type == "cuda" and torch.cuda.is_available():
                        from .runtime_config import _query_physical_avail_vram
                        c_free, c_tot = torch.cuda.mem_get_info(self.engine.device)
                        p_free, p_tot = _query_physical_avail_vram(self.engine.device, getattr(self, "plan", None))
                        entry_free_b = int(p_free if p_free is not None else c_free)
                        entry_total_b = int(p_tot if (p_tot and p_tot > 0) else c_tot)
                        entry_sec_b = int(c_free)
                    else:
                        import psutil
                        vm = psutil.virtual_memory()
                        entry_free_b, entry_total_b = int(vm.available), int(vm.total)
                    cal = getattr(self, "calibration", None)
                    pcie_v = float(cal["pcie_h2d_gb_s"]) if (isinstance(cal, dict) and isinstance(cal.get("pcie_h2d_gb_s"), (int, float))) else None
                    n_off = int(getattr(past_kv, "offloaded_kv_layer_count", 0)) if past_kv is not None else 0
                    n_gpu = int(getattr(past_kv, "gpu_kv_layer_count", 0)) if past_kv is not None else 0
                    off_ratio = (float(n_off) / float(n_off + n_gpu)) if (n_off + n_gpu) > 0 else 0.0
                    gb.begin_turn(
                        prompt_tokens=len(prompt_ids),
                        cached_prefix_tokens=cached_len,
                        had_prior_archive=had_prior_archive,
                        free_bytes=entry_free_b,
                        total_bytes=max(1, entry_total_b),
                        reserve_bytes=max(192 * MiB, int(entry_total_b * 0.05)),
                        secondary_free_bytes=entry_sec_b,
                        pcie_gb_s=pcie_v,
                        offload_ratio=off_ratio,
                    )
                except Exception:
                    pass
            if unlimited_tokens:
                lim = self.runtime_limits(busy=True)
                model_ctx = int(lim.get("effective_context_length") or lim.get("native_context_length") or 32768)
                eff_max_new = max(1, min(32768, max(2048, model_ctx - len(prompt_ids))))
            else:
                eff_max_new = int(max_new_tokens)

            self._active_sampling = sampling_cfg
            if sampling_cfg.seed is not None and sampling_cfg.temperature > 0.0:
                rng_dev = self.engine.device if getattr(self.engine, "device", None) is not None else torch.device("cpu")
                gen_rng = torch.Generator(device=rng_dev)
                gen_rng.manual_seed(int(sampling_cfg.seed))
                self._sampling_rng = gen_rng
            else:
                self._sampling_rng = None
            if self.thought_anchor_enabled and active_anchor_ref is None:
                from .thought_buffer import InitialProbabilityAnchor
                if use_session_cache and cached_len > 0 and self._session_thought_anchor is not None:
                    active_anchor_ref = self._session_thought_anchor
                else:
                    active_anchor_ref = InitialProbabilityAnchor(
                        mirror_mode=(self.q1_graphify_feedback == "mirror_scales"),
                    )
                self._active_thought_anchor = active_anchor_ref
                auto_anchor_installed = True
            if self.probability_bus is not None and active_anchor_ref is not None:
                self.probability_bus.bind(initial_anchor=active_anchor_ref)

            started_in_think = bool(getattr(self, "_last_prompt_opened_think", enable_thinking))
            splitter = _IncrementalThinkSplitter(start_in_thinking=started_in_think)
            clean_gen_ids: list[int] = []
            raw_gen_ids: list[int] = []
            committed_tok_idx = 0
            thinking_acc: list[str] = []
            answer_acc: list[str] = []
            hud_beacon = getattr(self, "_hud_beacon", None)
            if hud_beacon is not None:
                hud_beacon.begin_turn(len(prompt_ids), cached_len)

            inner = self._stream_token_ids(
                prompt_ids,
                max_new_tokens=eff_max_new,
                stop_on_eos=stop_on_eos,
                cached_prefix_len=cached_len,
                past_kv=past_kv,
                cancel_check=cancel_check,
            )
            del past_kv
            for ev in inner:
                if ev["kind"] == "tokens":
                    new_ids = ev["new_ids"]
                    raw_gen_ids.extend(new_ids)
                    for tid in new_ids:
                        if tid not in eos_ids:
                            clean_gen_ids.append(tid)
                    if len(clean_gen_ids) > committed_tok_idx:
                        if committed_tok_idx >= 4:
                            ctx_ids = clean_gen_ids[committed_tok_idx - 4 : committed_tok_idx]
                            win_ids = clean_gen_ids[committed_tok_idx - 4 :]
                            ctx_s = self.tokenizer.decode(ctx_ids, skip_special_tokens=False)
                            win_s = self.tokenizer.decode(win_ids, skip_special_tokens=False)
                            delta_cand = win_s[len(ctx_s) :] if win_s.startswith(ctx_s) else ""
                        else:
                            prev_s = (
                                self.tokenizer.decode(clean_gen_ids[:committed_tok_idx], skip_special_tokens=False)
                                if committed_tok_idx > 0
                                else ""
                            )
                            win_s = self.tokenizer.decode(clean_gen_ids, skip_special_tokens=False)
                            delta_cand = win_s[len(prev_s) :] if win_s.startswith(prev_s) else ""

                        if delta_cand and not delta_cand.endswith("�"):
                            committed_tok_idx = len(clean_gen_ids)
                            for phase, chunk in splitter.push(delta_cand, final=False):
                                if phase == "thinking":
                                    thinking_acc.append(chunk)
                                else:
                                    answer_acc.append(chunk)
                                if hud_beacon is not None:
                                    hud_beacon.on_tokens(
                                        len(raw_gen_ids),
                                        "".join(answer_acc),
                                        "".join(thinking_acc),
                                        phase=phase,
                                    )
                                yield {"type": "delta", "phase": phase, "text": chunk}
                elif ev["kind"] == "final_state":
                    final_info = ev
            inner = None

            # Flush any remaining uncommitted tokens & buffered tail characters
            rem_raw = (
                self.tokenizer.decode(clean_gen_ids[committed_tok_idx:], skip_special_tokens=False)
                if len(clean_gen_ids) > committed_tok_idx
                else ""
            )
            for phase, chunk in splitter.push(rem_raw, final=True):
                if phase == "thinking":
                    thinking_acc.append(chunk)
                else:
                    answer_acc.append(chunk)
                yield {"type": "delta", "phase": phase, "text": chunk}

            metric_device = self._gpu_metric_device()
            peak_vram_mib = round(torch.cuda.max_memory_allocated(metric_device) / MiB, 2) if metric_device is not None else 0.0
            dec_tokens = int(final_info.get("decode_tokens", 0))
            dec_s = float(final_info.get("decode_s", 0.0))
            tok_per_s = round(dec_tokens / max(1e-6, dec_s), 2) if dec_tokens > 0 else 0.0
            passes = max(1, int(final_info.get("backbone_passes", 1)))
            answer_text = "".join(answer_acc)
            thinking_text = "".join(thinking_acc)

            park_ms = 0.0
            if use_session_cache and final_info.get("past") is not None:
                park_ms = self.session.record_turn_end(
                    config=self.engine.config,
                    all_ids_full=final_info["all_ids"],
                    past_key_values=final_info["past"],
                    last_fed_h_post=final_info.get("last_fed_h"),
                    engine=self.engine,
                    mtp_head=self.mtp_head,
                    captured=self.captured,
                    chatml_think_format=bool(getattr(self, "_last_prompt_chatml_think", False)),
                    consumed_tokens=final_info.get("consumed_tokens"),
                )
                from .tool_calling import normalize_messages, parse_tool_calls, tools_signature
                norm_prompt_msgs = normalize_messages(messages, parse_tool_args_as_dict=False)
                self._session_prompt_messages = norm_prompt_msgs
                _, self._session_last_tool_calls = parse_tool_calls(
                    answer_text, tools=tools, allow_bare_json=bool(tools)
                )
                self._session_messages = [
                    *norm_prompt_msgs,
                    {"role": "assistant", "content": answer_text},
                ]
                self._session_tools_sig = tools_signature(tools, tool_choice)
                ledger_ref = getattr(self, "token_ledger", None)
                if ledger_ref is not None and ledger_ref.enabled:
                    rec_info = getattr(self, "_last_ledger_recovery", {}) or {}
                    ledger_ref.record_turn(
                        full_messages=self._session_messages,
                        all_token_ids=final_info["all_ids"],
                        prompt_token_len=len(prompt_ids),
                        committed_len=len(self.session.fed),
                        eos_ended=bool(raw_gen_ids and raw_gen_ids[-1] in eos_ids),
                        tools_sig=self._session_tools_sig,
                        chatml_think_format=bool(getattr(self, "_last_prompt_chatml_think", False)),
                        recovery_kind=str(rec_info.get("kind", "fresh")),
                        matched_turn_index=int(rec_info.get("matched_turn_index", -1)),
                    )
            final_info.pop("past", None)
            final_info.pop("last_fed_h", None)
            if not use_session_cache:
                self._release_transient_state()
            else:
                gc.collect()
                if self.engine.device.type == "cuda":
                    torch.cuda.empty_cache()
                    try:
                        from .runtime_config import _query_physical_avail_vram
                        q_avail, _ = _query_physical_avail_vram(self.engine.device, getattr(self, "plan", None))
                        if q_avail is not None:
                            self._idle_free_vram_bytes = q_avail
                    except Exception:
                        pass

            stopped_by_eos = bool(raw_gen_ids and raw_gen_ids[-1] in eos_ids)
            metrics = {
                "prompt_tokens": len(prompt_ids),
                "cached_prefix_tokens": cached_len,
                "generated_tokens": len(raw_gen_ids),
                "stopped_by_eos": stopped_by_eos,
                "finish_reason": "stop" if stopped_by_eos else "length",
                "unpark_ms": round(unpark_ms, 2),
                "prefill_ms": round(float(final_info.get("prefill_ms", 0.0)), 2),
                "prefill_chunk_tokens": self.prefill_chunk_tokens,
                "decode_s": round(dec_s, 3),
                "decode_tok_per_s": tok_per_s,
                "avg_tokens_per_backbone_pass": round(dec_tokens / passes, 3) if dec_tokens > 0 else 1.0,
                "peak_vram_mib": peak_vram_mib,
                "park_to_ram_ms": round(park_ms, 2),
                "parked_session_ram_mib": round(self.session.parked_bytes / MiB, 2),
                "spec_gate": final_info.get("spec_gate", {}),
                "sampling": sampling_cfg.as_dict(),
                "unlimited_tokens": unlimited_tokens,
                "kv_offloaded": bool(final_info.get("kv_offloaded", False)),
                "kv_gpu_layers": int(final_info.get("kv_gpu_layers", 0)),
                "kv_offloaded_layers": int(final_info.get("kv_offloaded_layers", 0)),
                "kv_promoted_layers": int(final_info.get("kv_promoted_layers", 0)),
            }
            progressive = getattr(self.engine, "progressive", None)
            if progressive is not None:
                metrics["progressive_execution"] = progressive.stats()
            ec = self.expert_cache_stats()
            if ec is not None:
                metrics["expert_cache"] = ec
            from src.quantized_projection import projection_stats
            metrics["packed_projections"] = projection_stats(self.engine.model)
            if getattr(self, "survivor_tracker", None) is not None:
                metrics["survivor_probability_path"] = self.survivor_tracker.stats()
            if getattr(self, "stanok_contour", None) is not None:
                metrics["stanok_contour"] = self.stanok_contour.stats()
            if getattr(self, "probability_bus", None) is not None:
                metrics["probability_bus"] = self.probability_bus.stats()
            if getattr(self, "cold_warmup_scale", None) is not None:
                metrics["cold_warmup_v4"] = self.cold_warmup_scale.stats()
            if getattr(self.engine, "q1_arena", None) is not None:
                metrics["q1_role_arena"] = self.engine.q1_arena.stats()
            q1_lh = _q1_lm_head_telemetry(self.engine.model)
            if q1_lh is not None:
                metrics["q1_lm_head"] = q1_lh
            puz_ctrl = getattr(self.engine, "puzzle_controller", None)
            if puz_ctrl is not None:
                metrics["q1_puzzle_overlay"] = puz_ctrl.stats()
            if active_anchor_ref is not None:
                metrics["thought_anchor"] = active_anchor_ref.telemetry.summary()
            if getattr(self, "token_ledger", None) is not None:
                metrics["token_ledger"] = {
                    **self.token_ledger.stats(),
                    **dict(getattr(self, "_last_ledger_recovery", {}) or {}),
                }
            if getattr(self, "generation_balance", None) is not None:
                try:
                    end_free_b, end_total_b = 0, 1
                    if self.engine is not None and getattr(self.engine, "device", None) is not None and self.engine.device.type == "cuda" and torch.cuda.is_available():
                        from .runtime_config import _query_physical_avail_vram
                        c_free, c_tot = torch.cuda.mem_get_info(self.engine.device)
                        p_free, p_tot = _query_physical_avail_vram(self.engine.device, getattr(self, "plan", None))
                        end_free_b = int(p_free if p_free is not None else c_free)
                        end_total_b = int(p_tot if (p_tot and p_tot > 0) else c_tot)
                    else:
                        import psutil
                        vm = psutil.virtual_memory()
                        end_free_b, end_total_b = int(vm.available), int(vm.total)
                    cal = getattr(self, "calibration", None)
                    cal_dec = float(cal["decode_tok_per_s"]) if (isinstance(cal, dict) and isinstance(cal.get("decode_tok_per_s"), (int, float))) else None
                    gb_summary = self.generation_balance.finish_turn(
                        prompt_tokens=len(prompt_ids),
                        cached_prefix_tokens=cached_len,
                        generated_tokens=len(raw_gen_ids),
                        free_bytes=end_free_b,
                        total_bytes=max(1, end_total_b),
                        kv_offloaded_layers=int(final_info.get("kv_offloaded_layers", 0)),
                        kv_gpu_layers=int(final_info.get("kv_gpu_layers", 0)),
                        decode_tok_per_s=float(tok_per_s),
                        calibrated_decode_tok_per_s=cal_dec,
                    )
                    scales_bal: dict[str, Any] = {}
                    kv_bal = getattr(self, "_last_kv_placement_balance", None)
                    if kv_bal is not None:
                        scales_bal["kv_placement"] = kv_bal
                    q1_runner = getattr(self, "_q1_cubes_runner", None)
                    if q1_runner is not None and hasattr(q1_runner, "stats"):
                        tm_v4 = (q1_runner.stats() or {}).get("trunk_mirror_v4") or {}
                        if tm_v4.get("rssf_balance") is not None:
                            scales_bal["m0_trunk_q1"] = tm_v4["rssf_balance"]
                        tb = tm_v4.get("v4_trunk_breather") or {}
                        if isinstance(tb.get("m1"), dict) and tb["m1"].get("rssf_balance") is not None:
                            scales_bal["m1_trunk_load"] = tb["m1"]["rssf_balance"]
                        if isinstance(tb.get("m2"), dict) and tb["m2"].get("rssf_balance") is not None:
                            scales_bal["m2_trunk_pool"] = tb["m2"]["rssf_balance"]
                    if getattr(self, "cold_warmup_scale", None) is not None:
                        m3_bal = (self.cold_warmup_scale.stats() or {}).get("rssf_balance")
                        if m3_bal is not None:
                            scales_bal["m3_cold_warmup"] = m3_bal
                    if getattr(self, "survivor_tracker", None) is not None:
                        m6_bal = (self.survivor_tracker.stats() or {}).get("rssf_balance")
                        if m6_bal is not None:
                            scales_bal["m6_probability_path"] = m6_bal
                    if active_anchor_ref is not None and hasattr(active_anchor_ref, "mirror_stats"):
                        m8_bal = (active_anchor_ref.mirror_stats() or {}).get("rssf_balance")
                        if m8_bal is not None:
                            scales_bal["m8_thought_buffer"] = m8_bal
                    if isinstance(metrics.get("expert_cache"), dict):
                        imp = metrics["expert_cache"].get("imprint_overlay") or {}
                        for k_src, k_dst in (
                            ("m1_load_dispatcher", "m1_moe_load"),
                            ("m2_region_breathing_pool", "m2_moe_pool"),
                            ("m4_post_prefill_warmup", "m4_moe_warmup"),
                        ):
                            sub = imp.get(k_src)
                            if isinstance(sub, dict) and sub.get("rssf_balance") is not None:
                                scales_bal[k_dst] = sub["rssf_balance"]
                    if isinstance(metrics.get("speculative_decoding"), dict):
                        ms7 = metrics["speculative_decoding"].get("mirror_speculation")
                        if isinstance(ms7, dict) and ms7.get("rssf_balance") is not None:
                            scales_bal["m7_speculation"] = ms7["rssf_balance"]
                    gb_summary["active_scales"] = scales_bal
                    metrics["rssf_generation_balance"] = gb_summary
                except Exception:
                    pass
            if self.prefill_wavefront:
                metrics["prefill_wavefront"] = {"enabled": True, "windows": list(self._wavefront_windows)}
            if auto_anchor_installed and use_session_cache:
                self._session_thought_anchor = active_anchor_ref
            self.last_metrics = metrics
            completed = True
            if hud_beacon is not None:
                hud_beacon.finish_turn(self, metrics, answer_text, thinking_text)
            yield {
                "type": "done",
                "thinking_text": thinking_text,
                "answer_text": answer_text,
                "metrics": metrics,
            }
        except BaseException as exc:
            if isinstance(exc, GenerationCancelled):
                self.generations_cancelled += 1
            # The traceback keeps the finished generator frames alive, and with them the partial
            # KV cache / GatedDeltaNet states of the aborted request. Drop those frame locals now.
            traceback.clear_frames(exc.__traceback__)
            raise
        finally:
            if _rssf_scoped:
                set_rssf_balance_enabled(_prev_rssf_override)
            self._active_sampling = None
            self._sampling_rng = None
            if auto_anchor_installed:
                self._active_thought_anchor = None
                if self.probability_bus is not None:
                    self.probability_bus.initial_anchor = None
            if inner is not None:
                inner.close()
                inner = None
            final_info.clear()
            if not completed:
                hb = getattr(self, "_hud_beacon", None)
                if hb is not None and hasattr(hb, "cancel_turn"):
                    hb.cancel_turn()
                self._abort_cleanup()

    def close(self) -> None:
        if getattr(self, "_hud_beacon", None) is not None:
            try:
                self._hud_beacon.close()
            except Exception:
                pass
            self._hud_beacon = None
        # Optional ThoughtBuffer baselines belong to this engine's device lifetime.
        self._neutral_logprob_cache = None
        self._session_thought_anchor = None
        if getattr(self, "session", None) is not None:
            # Preserve disk Token Ledger across clean engine shutdown / server restart!
            self.reset_session(clear_ledger=False)
        if getattr(self, "survivor_tracker", None) is not None:
            self.survivor_tracker.close()
            self.survivor_tracker = None
        if getattr(self, "stanok_contour", None) is not None:
            try:
                self.stanok_contour.close()
            except Exception:
                pass
            self.stanok_contour = None
        self.probability_bus = None
        if getattr(self, "speculator", None) is not None:
            self.speculator.close()
            self.speculator = None
        if getattr(self, "_region_controller", None) is not None:
            self._region_controller.close()
            self._region_controller = None
        if getattr(self, "_q1_cubes_runner", None) is not None:
            try:
                self._q1_cubes_runner.close()
            except Exception:
                pass
            self._q1_cubes_runner = None
        if self.pread_reader is not None:
            try:
                self.pread_reader.close()
            except Exception:
                pass
            self.pread_reader = None
        if getattr(self, "engine", None) is not None:
            try:
                self.engine.close()
            except Exception:
                pass
            self.engine = None
        if getattr(self, "model_source", None) is not None and hasattr(self.model_source, "close"):
            try:
                self.model_source.close()
            except Exception:
                pass
            self.model_source = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    def __enter__(self) -> "QuintessentiaUnifiedEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
