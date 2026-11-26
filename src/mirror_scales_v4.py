"""Version 4 (Final — Conveyor-Free): Strictly Isolated Multi-Scale Mirror Equality Architecture.

ZERO Conveyor (`Q1RoleCycleArena` / `A -> B -> C(0) -> A` REMOVED in V4!).
ZERO Graphify (`OnlineGraphifyTopology` / `SpiralSphereMutualFeedback` / `face_edges` / `corner90_edges` REMOVED!).
ZERO Cross-Scale Bus (`UnifiedProbabilityBus` / `StanokContainmentContour` REMOVED!).

Every scale runs its OWN self-contained bracketless Mirror Equality read from the center:

    Outcome_(t-1) + ObservedBehavior_(t-1) = Strategy_t = ObservedBehavior_(t+1) + Outcome_(t+1)
                    ObservedBehavior_(t-1) = Strategy_t = ObservedBehavior_(t+1)

HARD ARCHITECTURAL LAW (ZERO CROSS-SCALE COUPLING):
  - NO mirror formula on one scale may read the Strategy_t, tolerances, tension, or commands of
    another scale's mirror formula.
  - Scales touch ONLY through physical machine reality (real GPU sensors, real compute/IO timings,
    real CUDA events, real resident cache presence, real layer tensors).

The legacy conveyor is replaced by TWO DISTINCT, INDEPENDENT mirror-born mechanisms (M1 & M2),
alongside the isolated warmup & layer execution scales (M3, M4, M5):

  1. M1 — `MirrorLoadDispatcher` (Replaces conveyor pacing + static VRAM thermostat):
     Reads ONLY hardware load sensors (NVML/CUDA free/total VRAM + GPU/memory-bus utilization)
     and physical compute-vs-IO timings (`t_gpu_ms`, `t_io_ms`). Decides by center-read Mirror
     Equality:
       - `dispatch_mode`: `"DIRECT_BURST"` | `"OVERLAP_ASYNC"` | `"THROTTLE_DAMP"` | `"SHED_PROTECT"`
       - `batch_span` (adaptive burst width) and `in_flight_cap` (1..2 async streams)
       - `target_budget_bytes` & `shed_bytes_needed` (preventive damping BEFORE floor contact!).

  2. M2 — `MirrorRegionBreathingPool` (Replaces rigid 3-slot A/B/C(0) conveyor ring):
     Manages live GPU memory regions directly via hardware CUDA Events (`ready_event`,
     `consumer_event`, `record_stream`) with NO `A -> B -> C(0) -> A` ring, NO `rotate()`, and
     NO forced empty standby slot. Decides each region's lifecycle by its own Mirror Equality:
       - `"PIN_REUSE"`        : region already live in pool -> instant zero-copy pointer reuse
       - `"DIRECT_ADMIT"`     : admit freshly prepared GPU region directly to compute stream
       - `"HOLD_WARM_REGION"` : keep recurring region warm in pool across steps
       - `"EXHALE_RECLAIM"`   : exhale one-off / excess finished region as soon as its CUDA
                                consumer event completes, returning bytes to GPU immediately.

  3. M3 — `MirrorColdWarmupScale`:
     Startup warmup (NVRTC CUDA JIT + cold resident trunk layers), stopping on warm plateau `EQUAL`.

  4. M4 — `MirrorPostPrefillRegionWarmupScale`:
     Targeted post-prefill region warmup using immutable per-layer prefill maps + fresh decode scars.

  5. M5 — `MirrorIsolatedMoEOverlayV4`:
     Layer-local exact BF16 prefill + activation-verified Q1 crystallization + Kernel #6 anchored
     differential Q1 decode + strict local `line_spine` Mirror Equality (Two-Slit entry cone).
"""
from __future__ import annotations

import atexit
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import uuid

import torch
import torch.nn.functional as F

from src.pure_mirror_agents_nvrtc import (
    launch_pure_mirror_diff_q1_gemv,
    launch_pure_mirror_diff_q1_moe_batch,
    launch_pure_mirror_line_step,
)
from src.rssf_dynamic_operators import (
    DynamicBalanceSnapshot,
    RSSFCollapseDetector,
    StrategyCollapseReport,
    VolterraFDSMonitor,
    is_rssf_balance_enabled,
    observe_hardware_load_balance,
    observe_post_prefill_region_balance,
    observe_region_breathing_balance,
    observe_warmup_fds_balance,
)


# ---------------------------------------------------------------------------
# Canonical Stateless Bracketless Mirror Equality Primitive (L4 Pure Math)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MirrorBalanceResult:
    """Result of one canonical 2-contour bracketless Mirror Equality evaluation (L4).

    Zero hand-tuned magic thresholds — uses ONLY the natural mirror threshold 1/2 (0.5)
    and sign of Tie(a, b) with non-commutative directed contacts alpha_{u->v}, r_{u->v}:
      - UNKNOWN : any wing role or center has zero norm (unclosed time step)
      - SPLIT   : tau_core <= 0 or tau_wing <= 0 (phase break / leaf branching)
      - EQUAL   : tau_core > 0 and tau_wing >= 0.5 and delta_res < 0.5 (stable rest)
      - TENSION : otherwise (live non-equilibrium -> targeted step along scar r)
    """

    state: str                        # "UNKNOWN" | "EQUAL" | "TENSION" | "SPLIT"
    strategy: Tuple[float, ...]       # S_t = (w1 * P1 + w2 * P2) / (w1 + w2) (non-commutative center)
    tau_core: float                   # min(Tie(P1, S_t), Tie(S_t, P2)) in [0, 1]
    tau_wing: float                   # Tie(S_left, S_right) in [0, 1]
    d_core: float                     # D_core = delta(P1->S_t)+delta(S_t->P2)+delta(P2->S_t)+delta(S_t->P1)
    delta_res: float                  # Non-commutative cross-wing residual (past -> future)
    action_energy: float              # Variational functional E_t(Sigma_t)
    scar_alpha: float                 # Directed contact alpha_{P1 -> P2}
    scar_residual: Tuple[float, ...]  # Exact reversible scar r_{P1->P2} = P2 - alpha * P1

    @property
    def d_wings(self) -> float:
        return self.delta_res

    @property
    def tension(self) -> float:
        return self.action_energy

    @property
    def contra_core(self) -> bool:
        return self.state == "SPLIT"


def _vec_norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def _directed_contact(
    u: Sequence[float], v: Sequence[float], eps: float = 1e-12
) -> Tuple[float, Tuple[float, ...], float]:
    """Non-commutative directed contact u -> v: alpha = <u,v>/(||u||^2+eps), r = v - alpha*u."""
    uu = sum(float(a) * float(a) for a in u)
    uv = sum(float(a) * float(b) for a, b in zip(u, v))
    alpha = uv / (uu + eps)
    res = tuple(float(b) - alpha * float(a) for a, b in zip(u, v))
    delta = _vec_norm(res) / (_vec_norm(v) + eps)
    return alpha, res, delta


def _mirror_tie(a: Sequence[float], b: Sequence[float], eps: float = 1e-12) -> float:
    """Canonical symmetric convergence measure Tie(a, b) in [0, 1]."""
    n_sum = math.sqrt(sum((float(x) + float(y)) ** 2 for x, y in zip(a, b)))
    n_diff = math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
    val = (n_sum - n_diff) / (n_sum + n_diff + eps)
    return val if val > 0.0 else 0.0


def eval_mirror_equality_step(
    outcome_prev: Sequence[float],
    obs_prev: Sequence[float],
    obs_next: Sequence[float],
    outcome_next: Sequence[float],
    *,
    strategy: Optional[Sequence[float]] = None,
    eps: float = 1e-12,
    equal_tol: Optional[float] = None,   # deprecated compatibility arg (canonical threshold is always 0.5)
    split_tol: Optional[float] = None,   # deprecated compatibility arg (canonical SPLIT is tau <= 0.0)
) -> MirrorBalanceResult:
    """Evaluate canonical 2-contour bracketless Mirror Equality from the center (L4).

    Ordered non-commutative 4-wing tuple:
        P = (P0, P1, P2, P3) = (Outcome_(t-1), ObservedBehavior_(t-1), ObservedBehavior_(t+1), Outcome_(t+1))
    Contour A (inner bracketless Stone-Law stem):
        ObservedBehavior_(t-1) = Strategy_t = ObservedBehavior_(t+1)
    Contour B (outer 4x4 non-commutative contact shell with FuturePressure):
        M_ij = alpha_{P_i -> P_j}, R_ij = r_{P_i -> P_j}
        w0, w1 reweight immutable past via FuturePressure (2,3 -> 0,1) without mutating P0, P1
        w2, w3 ground future wing via past support (0,1 -> 2,3)
    Center S_t (non-commutative equilibrium, NOT 0.5 average):
        S_t = (w1 * P1 + w2 * P2) / (w1 + w2) when not explicitly supplied.
    4-valued logic L4 (ZERO magic constants — only natural 1/2 threshold and Tie sign):
        UNKNOWN | SPLIT | EQUAL | TENSION.
    """
    del equal_tol, split_tol
    n = len(obs_prev)
    if not (len(outcome_prev) == n == len(obs_next) == len(outcome_next)) or n == 0:
        raise ValueError("All four mirror vectors must share the same positive length")
    if strategy is not None and len(strategy) != n:
        raise ValueError("Explicit strategy vector must share the same length as wing vectors")

    p0 = tuple(float(x) for x in outcome_prev)
    p1 = tuple(float(x) for x in obs_prev)
    p2 = tuple(float(x) for x in obs_next)
    p3 = tuple(float(x) for x in outcome_next)
    roles = (p0, p1, p2, p3)
    norms = tuple(_vec_norm(r) for r in roles)
    zero_vec = tuple(0.0 for _ in range(n))

    # UNKNOWN: unclosed step whenever any role (or explicit strategy) has zero norm
    if any(nm <= eps for nm in norms) or (strategy is not None and _vec_norm(strategy) <= eps):
        return MirrorBalanceResult(
            state="UNKNOWN",
            strategy=tuple(float(x) for x in strategy) if strategy is not None else zero_vec,
            tau_core=0.0,
            tau_wing=0.0,
            d_core=1.0,
            delta_res=1.0,
            action_energy=float("inf"),
            scar_alpha=0.0,
            scar_residual=p2,
        )

    # Contour B: full 4x4 non-commutative directed contact matrix M_ij and residual norms ||R_ij||
    m = [[0.0] * 4 for _ in range(4)]
    r_vecs: List[List[Optional[Tuple[float, ...]]]] = [[None] * 4 for _ in range(4)]
    r_norms = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            a_ij, r_ij, _ = _directed_contact(roles[i], roles[j], eps)
            m[i][j] = a_ij
            r_vecs[i][j] = r_ij
            r_norms[i][j] = _vec_norm(r_ij)

    # FuturePressure weights on immutable past wing (P0, P1)
    w0 = 1.0 + abs(m[1][0]) + max(0.0, m[3][0]) + max(0.0, m[2][0])
    w1 = 1.0 + abs(m[0][1]) + max(0.0, m[2][1]) + max(0.0, m[3][1])
    s_left = tuple((w0 * a + w1 * b) / (w0 + w1) for a, b in zip(p0, p1))

    # Past-grounded weights on future wing (P2, P3)
    w2 = 1.0 + abs(m[3][2]) + max(0.0, m[1][2]) + max(0.0, m[0][2])
    w3 = 1.0 + abs(m[2][3]) + max(0.0, m[0][3]) + max(0.0, m[1][3])
    s_right = tuple((w2 * a + w3 * b) / (w2 + w3) for a, b in zip(p2, p3))

    # Born center object S_t (non-commutative equilibrium weighted by FuturePressure vs past support)
    if strategy is not None:
        s_center = tuple(float(x) for x in strategy)
    else:
        s_center = tuple((w1 * a + w2 * b) / (w1 + w2) for a, b in zip(p1, p2))

    # Exact reversible scar Gamma_scar for P1 -> P2: alpha * P1 + r == P2 identically
    scar_alpha = m[1][2]
    scar_res = r_vecs[1][2] if r_vecs[1][2] is not None else zero_vec

    # Contour A: inner bracketless central stem (P1 ≍ S_t ≍ P2)
    if _vec_norm(s_center) <= eps:
        # Exact opposite cancellation of P1 and P2 in center -> SPLIT
        tau_core = 0.0
        d_core = 4.0
        d_sl_s = 1.0
        d_sr_s = 1.0
    else:
        tau_core = min(
            _mirror_tie(p1, s_center, eps),
            _mirror_tie(s_center, p2, eps),
            _mirror_tie(p1, p2, eps),
        )
        d_core = (
            _directed_contact(p1, s_center, eps)[2]
            + _directed_contact(s_center, p2, eps)[2]
            + _directed_contact(p2, s_center, eps)[2]
            + _directed_contact(s_center, p1, eps)[2]
        )
        d_sl_s = _directed_contact(s_left, s_center, eps)[2]
        d_sr_s = _directed_contact(s_right, s_center, eps)[2]

    tau_wing = _mirror_tie(s_left, s_right, eps)
    delta_res = (r_norms[0][3] + r_norms[1][2] + r_norms[0][2] + r_norms[1][3]) / (
        2.0 * (norms[2] + norms[3]) + eps
    )
    action_energy = d_core + (1.0 - tau_wing) + d_sl_s + d_sr_s + delta_res

    if tau_core <= 0.0 or tau_wing <= 0.0:
        state = "SPLIT"
    elif tau_wing >= 0.5 and delta_res < 0.5:
        state = "EQUAL"
    else:
        state = "TENSION"

    return MirrorBalanceResult(
        state=state,
        strategy=s_center,
        tau_core=tau_core,
        tau_wing=tau_wing,
        d_core=d_core,
        delta_res=delta_res,
        action_energy=action_energy,
        scar_alpha=scar_alpha,
        scar_residual=scar_res,
    )


# ---------------------------------------------------------------------------
# Mechanism M1: MirrorLoadDispatcher (Replaces Conveyor Pacing + VRAM Thermostat)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadDispatchDecision:
    dispatch_mode: str            # "DIRECT_BURST" | "OVERLAP_ASYNC" | "THROTTLE_DAMP" | "SHED_PROTECT"
    budget_action: str            # "expand" | "hold" | "damp" | "protect"
    target_budget_bytes: int
    shed_bytes_needed: int
    emergency: bool
    batch_span: int               # adaptive expert burst width
    in_flight_cap: int            # 1..2 async dispatch windows (NO 3-slot conveyor!)
    mirror_state: str
    free_bytes: int
    total_bytes: int
    gpu_util_pct: Optional[float]
    mem_util_pct: Optional[float]


class MirrorLoadDispatcher:
    """Mechanism M1 (Conveyor Replacement #1): Self-contained Mirror Equality GPU Load Dispatcher.

    Directly governs BOTH GPU memory headroom AND compute/IO dispatch pressure without any rigid
    A->B->C(0)->A conveyor:
      - Reads ONLY hardware sensors (NVML/CUDA free/total bytes + GPU/memory-bus utilization) and
        physical step timings (`t_gpu_ms`, `t_io_ms`, `t_wait_ms`).
      - Chooses live dispatch regime (`DIRECT_BURST`, `OVERLAP_ASYNC`, `THROTTLE_DAMP`,
        `SHED_PROTECT`), adaptive `batch_span`, and async window depth `in_flight_cap` (1..2).
      - Predictively damps memory & dispatch pressure BEFORE hitting the physical VRAM floor.

    Holds ZERO references to M2 (region breathing pool), M3/M4 (warmup), or M5 (MoE layer).
    """

    def __init__(
        self,
        device: Any = None,
        *,
        enabled: bool = True,
        safety_reserve_bytes: Optional[int] = None,
        reserve_ratio: float = 0.10,
        deadband_bytes: Optional[int] = None,
        initial_batch_span: int = 16,
        min_batch_span: int = 4,
        max_batch_span: int = 32,
        max_in_flight: int = 2,
        sensor_fn: Optional[Callable[[], Tuple[int, int, Optional[float], Optional[float], int, str]]] = None,
    ) -> None:
        self.device = torch.device("cpu" if device is None else device)
        self._sensor_fn = sensor_fn
        self.enabled = bool(
            enabled and (sensor_fn is not None or (self.device.type == "cuda" and torch.cuda.is_available()))
        )
        self.reserve_ratio = max(0.02, min(0.40, float(reserve_ratio)))
        self.min_batch_span = max(1, int(min_batch_span))
        self.max_batch_span = max(self.min_batch_span, int(max_batch_span))
        self.batch_span = max(self.min_batch_span, min(self.max_batch_span, int(initial_batch_span)))
        self.max_in_flight = max(1, min(2, int(max_in_flight)))
        self.in_flight_cap: int = self.max_in_flight
        self.dispatch_mode: str = "OVERLAP_ASYNC"

        self._monitor = None
        if self._sensor_fn is None and self.device.type == "cuda" and torch.cuda.is_available():
            from quintessentia_cli.gpu_memory import PhysicalVRAMMonitor
            self._monitor = PhysicalVRAMMonitor(self.device)

        # Short hardware sensor cache avoids 40 NVML + memory_stats calls/token:
        # - NVML utilization (`gpu_util`, `mem_util`) is ALWAYS cached for 0.25s (driver updates every ~50ms).
        # - VRAM free/total is cached 0.20s in calm headroom zone and 0.03s near reserve (fast mem_get_info only).
        self._hw_cache: Optional[Tuple[int, int, Optional[float], Optional[float], int, str]] = None
        self._hw_cache_ts: float = 0.0
        self._hw_poll_interval_s: float = 0.20
        self._hw_tight_poll_interval_s: float = 0.03
        self._util_cache: Tuple[Optional[float], Optional[float]] = (None, None)
        self._util_cache_ts: float = 0.0
        self._util_poll_interval_s: float = 0.25
        self.safety_reserve_bytes: int = 0

        if safety_reserve_bytes is not None:
            self.safety_reserve_bytes = max(1, int(safety_reserve_bytes))
        elif self.enabled:
            _fb, total_b, _gu, _mu, _pf, _src = self._read_hardware(force=True)
            self.safety_reserve_bytes = max(1, int(total_b * self.reserve_ratio)) if total_b > 0 else 0
        else:
            self.safety_reserve_bytes = 0

        self.deadband_bytes = (
            max(1, int(deadband_bytes))
            if deadband_bytes is not None
            else max(1, self.safety_reserve_bytes // 4)
        )

        # Own physical history (Mechanism M1 ONLY)
        self._prev_free_bytes: Optional[int] = None
        self._prev_used_frac: Optional[float] = None
        self._prev_load_frac: Optional[float] = None
        self._prev_headroom_frac: Optional[float] = None
        self._prev_calm_frac: Optional[float] = None
        self._prev_gpu_share: float = 0.5
        self._prev_io_share: float = 0.5
        self._prev_wait_share: float = 0.0
        self._last_source: str = "cpu"
        # Own scale-invariant RSSF / EC-Volterra / FDS / 8-force balance state (M1 ONLY)
        self._ec_fds = VolterraFDSMonitor(archive_decay=0.85)
        self._collapse_det = RSSFCollapseDetector(drift_horizon_steps=4.0)

        # Own telemetry
        self.evaluations: int = 0
        self.timing_updates: int = 0
        self.direct_burst_steps: int = 0
        self.overlap_async_steps: int = 0
        self.throttle_damp_steps: int = 0
        self.shed_protect_steps: int = 0
        self.ec_fds_damp_steps: int = 0
        self.balance_scale_break_steps: int = 0
        self.collapse_events: int = 0
        self.equal_holds: int = 0
        self.expansions: int = 0
        self.span_shrinks: int = 0
        self.span_expands: int = 0
        self.oom_recoveries: int = 0
        self.admissions_allowed: int = 0
        self.admissions_denied: int = 0
        self.total_gpu_ms: float = 0.0
        self.total_io_ms: float = 0.0
        self.total_wait_ms: float = 0.0
        self.min_observed_free_bytes: Optional[int] = None
        self.max_observed_budget_bytes: int = 0
        self.max_observed_gpu_util: Optional[float] = None
        self.max_observed_mem_util: Optional[float] = None
        self.last_mirror_state: str = "IDLE"
        self.last_scar_norm: float = 0.0
        self.last_fds_t_end_steps: Optional[float] = None
        self.last_volterra_archive: float = 0.0
        self.last_balance: Optional[DynamicBalanceSnapshot] = None
        self.last_collapse: Optional[StrategyCollapseReport] = None

    def _read_hardware(self, *, force: bool = False) -> Tuple[int, int, Optional[float], Optional[float], int, str]:
        """Return `(free_bytes, total_bytes, gpu_util_pct, mem_util_pct, torch_pool_free, source)`."""
        if self._sensor_fn is not None:
            return self._sensor_fn()
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return 0, 0, None, None, 0, "cpu"
        now = time.perf_counter()
        if not force and self._hw_cache is not None:
            age = now - self._hw_cache_ts
            is_calm = self._hw_cache[0] >= int(1.5 * self.safety_reserve_bytes)
            if (is_calm and age < self._hw_poll_interval_s) or (not is_calm and age < self._hw_tight_poll_interval_s):
                return self._hw_cache
        if self._monitor is not None:
            free_b, total_b, src = self._monitor.query()
            if force or (now - self._util_cache_ts) >= self._util_poll_interval_s:
                gpu_u, mem_u, _usrc = self._monitor.query_utilization()
                self._util_cache = (gpu_u, mem_u)
                self._util_cache_ts = now
            else:
                gpu_u, mem_u = self._util_cache
        else:
            fb, tb = torch.cuda.mem_get_info(self.device)
            free_b, total_b, src = int(fb), int(tb), "cuda"
            gpu_u, mem_u = None, None
        reserved_b = int(torch.cuda.memory_reserved(self.device))
        allocated_b = int(torch.cuda.memory_allocated(self.device))
        pool_free = max(0, reserved_b - allocated_b)
        snap = (free_b, total_b, gpu_u, mem_u, pool_free, src)
        self._hw_cache = snap
        self._hw_cache_ts = now
        return snap

    @property
    def sensor_source(self) -> str:
        if self._sensor_fn is not None:
            return self._last_source
        if self._monitor is not None and self._monitor.nvml_available:
            return "nvml"
        return "cuda" if self.device.type == "cuda" else "cpu"

    @staticmethod
    def _combine_load_frac(gpu_u: Optional[float], mem_u: Optional[float], fallback_used: float) -> float:
        if gpu_u is not None and mem_u is not None:
            return max(0.0, min(1.0, 0.5 * (gpu_u / 100.0) + 0.5 * (mem_u / 100.0)))
        if mem_u is not None:
            return max(0.0, min(1.0, mem_u / 100.0))
        if gpu_u is not None:
            return max(0.0, min(1.0, gpu_u / 100.0))
        return max(0.0, min(1.0, fallback_used))

    def evaluate_load(self, *, cached_bytes: int, current_budget_bytes: int) -> LoadDispatchDecision:
        """Evaluate physical GPU load & VRAM mirror balance -> emit load & dispatch regime."""
        cached_b = max(0, int(cached_bytes))
        cur_budget = max(0, int(current_budget_bytes))
        if not self.enabled:
            return LoadDispatchDecision(
                self.dispatch_mode, "hold", cur_budget, 0, False,
                self.batch_span, self.in_flight_cap, "EQUAL", 0, 0, None, None,
            )

        self.evaluations += 1
        free_b, total_b, gpu_u, mem_u, pool_free, src = self._read_hardware()
        self._last_source = src
        if total_b <= 0:
            return LoadDispatchDecision(
                self.dispatch_mode, "hold", cur_budget, 0, False,
                self.batch_span, self.in_flight_cap, "EQUAL", free_b, total_b, gpu_u, mem_u,
            )

        if self.min_observed_free_bytes is None or free_b < self.min_observed_free_bytes:
            self.min_observed_free_bytes = free_b
        if gpu_u is not None and (self.max_observed_gpu_util is None or gpu_u > self.max_observed_gpu_util):
            self.max_observed_gpu_util = gpu_u
        if mem_u is not None and (self.max_observed_mem_util is None or mem_u > self.max_observed_mem_util):
            self.max_observed_mem_util = mem_u

        reserve = self.safety_reserve_bytes
        rssf_on = is_rssf_balance_enabled()
        ec_diag = None
        collapse = None
        if rssf_on:
            ec_diag = self._ec_fds.update(free_bytes=free_b, total_bytes=total_b, reserve_bytes=reserve)
            collapse = self._collapse_det.evaluate(free_bytes=free_b, total_bytes=total_b, ec_fds=ec_diag)
            self.last_fds_t_end_steps = (
                ec_diag.fds_t_end_steps if math.isfinite(ec_diag.fds_t_end_steps) else None
            )
            self.last_volterra_archive = ec_diag.volterra_archive
            self.last_collapse = collapse
            if collapse.any_collapse:
                self.collapse_events += 1

        # 1. Hard physical wall contact -> SHED_PROTECT (0 OOM, 0 WDDM spill)
        if free_b < reserve:
            if pool_free > 0 and self._sensor_fn is None and self.device.type == "cuda":
                torch.cuda.empty_cache()
                free_b, total_b, gpu_u, mem_u, pool_free, src = self._read_hardware(force=True)
                if self.min_observed_free_bytes is None or free_b < self.min_observed_free_bytes:
                    self.min_observed_free_bytes = free_b
            if free_b < reserve:
                deficit = reserve - free_b
                emergency = free_b < (reserve // 2)
                target = max(0, cached_b - deficit)
                self.dispatch_mode = "SHED_PROTECT"
                self.in_flight_cap = 1
                self.batch_span = max(self.min_batch_span, self.batch_span // 2)
                self.shed_protect_steps += 1
                self.last_mirror_state = "PROTECT"
                self._prev_free_bytes = free_b
                self._prev_used_frac = max(0.0, min(1.0, 1.0 - free_b / max(1, total_b)))
                self._prev_load_frac = self._combine_load_frac(gpu_u, mem_u, self._prev_used_frac)
                self._prev_headroom_frac = 1e-4
                self._prev_calm_frac = max(1e-4, 1.0 - self._prev_load_frac)
                return LoadDispatchDecision(
                    "SHED_PROTECT", "protect", target, deficit, emergency,
                    self.batch_span, self.in_flight_cap, "PROTECT", free_b, total_b, gpu_u, mem_u,
                )

        # 2. Restored non-degenerate 2D canonical L4 contour `(headroom_frac, calm_frac)` with
        #    Enhanced Calculus Volterra projection + FDS T_end horizon + solver-free 8-force balance.
        used_frac = max(0.0, min(1.0, 1.0 - free_b / max(1, total_b)))
        load_frac = self._combine_load_frac(gpu_u, mem_u, used_frac)
        usable_span = max(1, total_b - reserve)
        headroom = free_b - reserve
        headroom_frac = max(1e-4, headroom / usable_span)
        calm_frac = max(1e-4, 1.0 - load_frac)

        is_first_step = self._prev_free_bytes is None
        prev_free = free_b if is_first_step else self._prev_free_bytes
        prev_load = load_frac if self._prev_load_frac is None else self._prev_load_frac
        prev_headroom_frac = headroom_frac if self._prev_headroom_frac is None else self._prev_headroom_frac
        prev_calm_frac = calm_frac if self._prev_calm_frac is None else self._prev_calm_frac

        delta_free_drop = max(0, prev_free - free_b)
        projected_free_next = free_b - delta_free_drop
        load_rise = max(0.0, load_frac - prev_load)
        proj_calm = calm_frac - load_rise

        if rssf_on and ec_diag is not None and collapse is not None:
            proj_headroom = ec_diag.projected_headroom_ec
            bal8 = observe_hardware_load_balance(
                used_frac=used_frac,
                headroom_frac=headroom_frac,
                load_frac=load_frac,
                calm_frac=calm_frac,
                drop_frac=delta_free_drop / float(usable_span),
                load_rise=load_rise,
                sustained_drain=ec_diag.sustained_drain_rate,
            )
            self.last_balance = bal8
            ec_fds_drift_breach = (self.evaluations >= 3) and collapse.unbounded_result_drift
            scale_break = bal8.regime == "SCALE_BREAK"
            sustained_drain_cur = ec_diag.sustained_drain_rate
        else:
            proj_headroom = (projected_free_next - reserve) / usable_span
            bal8 = None
            self.last_balance = None
            ec_fds_drift_breach = False
            scale_break = False
            sustained_drain_cur = 0.0

        # Feed EC/FDS horizon, 8-force scale break, and physical wall projection INTO the future wing P2
        # so canonical 2D L4 itself evaluates the true coupled trajectory (an idle bus cannot stay "calm"
        # when VRAM is projected to breach reserve or enter sustained FDS/Term-3 scale break).
        physical_pred_breach = (not is_first_step) and (projected_free_next <= reserve)

        p2_h = proj_headroom
        p2_c = proj_calm
        if physical_pred_breach:
            p2_h = min(p2_h, (projected_free_next - reserve) / usable_span, -1e-3)
            p2_c = min(p2_c, p2_h)
        if ec_fds_drift_breach and ec_diag is not None:
            fds_pull = -max(1e-3, headroom_frac * (4.0 / max(0.1, ec_diag.fds_t_end_steps) - 1.0) + 1e-3)
            p2_h = min(p2_h, fds_pull)
            p2_c = min(p2_c, p2_h)
            self.ec_fds_damp_steps += 1
        if scale_break and bal8 is not None:
            p2_h = min(p2_h, bal8.d3_margin - 1e-3)
            p2_c = min(p2_c, p2_h)
            self.balance_scale_break_steps += 1

        # Non-degenerate 2D L4 Mirror Equality (genuine non-zero reversible scar r_{P1->P2}!)
        p0 = (0.0, 0.0) if is_first_step else (prev_headroom_frac, prev_calm_frac)
        bal_2d = eval_mirror_equality_step(p0, (headroom_frac, calm_frac), (p2_h, p2_c), (1.0, 1.0))
        self.last_scar_norm = _vec_norm(bal_2d.scar_residual)
        # Strict 2-contour rest requires BOTH inner stem (tau_core >= 0.5) AND outer wings (tau_wing >= 0.5)
        bal_2d_state = "TENSION" if (bal_2d.state == "EQUAL" and bal_2d.tau_core < 0.5) else bal_2d.state

        if bal_2d_state == "SPLIT":
            # Ample VRAM headroom + no memory breach/drift: pure GPU/bus burst is OVERLAP_ASYNC
            if (
                not physical_pred_breach
                and not ec_fds_drift_breach
                and not scale_break
                and headroom_frac >= 0.45
                and sustained_drain_cur < 0.005
            ):
                joint_state = "TENSION"
            else:
                joint_state = "SPLIT"
        else:
            joint_state = bal_2d_state
        joint_tau_wing = bal_2d.tau_wing

        self.last_mirror_state = joint_state
        self._prev_free_bytes = free_b
        self._prev_used_frac = used_frac
        self._prev_load_frac = load_frac
        self._prev_headroom_frac = headroom_frac
        self._prev_calm_frac = calm_frac

        physical_ceiling = cached_b + headroom

        # 3. Canonical L4 SPLIT on either physical contour -> preventive Mirror Damping BEFORE floor impact
        if joint_state == "SPLIT":
            damp_target = max(cached_b, cached_b + (headroom // 2))
            target = damp_target if cur_budget > damp_target else cur_budget
            self.dispatch_mode = "THROTTLE_DAMP"
            self.in_flight_cap = 1
            self.throttle_damp_steps += 1
            return LoadDispatchDecision(
                "THROTTLE_DAMP", "damp", target, 0, False,
                self.batch_span, self.in_flight_cap, joint_state, free_b, total_b, gpu_u, mem_u,
            )

        # 4. Canonical L4 TENSION -> OVERLAP_ASYNC; EQUAL / UNKNOWN -> DIRECT_BURST
        self.in_flight_cap = self.max_in_flight
        if joint_state == "TENSION":
            self.dispatch_mode = "OVERLAP_ASYNC"
            self.overlap_async_steps += 1
        else:
            self.dispatch_mode = "DIRECT_BURST"
            self.direct_burst_steps += 1

        if physical_ceiling > cur_budget + self.deadband_bytes:
            if joint_state == "TENSION":
                step_scale = max(0.25, min(1.0, joint_tau_wing))
                target = cur_budget + max(self.deadband_bytes, int((physical_ceiling - cur_budget) * step_scale))
            else:
                target = physical_ceiling
            self.expansions += 1
            if target > self.max_observed_budget_bytes:
                self.max_observed_budget_bytes = target
            return LoadDispatchDecision(
                self.dispatch_mode, "expand", target, 0, False,
                self.batch_span, self.in_flight_cap, joint_state, free_b, total_b, gpu_u, mem_u,
            )

        self.equal_holds += 1
        return LoadDispatchDecision(
            self.dispatch_mode, "hold", cur_budget, 0, False,
            self.batch_span, self.in_flight_cap, joint_state, free_b, total_b, gpu_u, mem_u,
        )

    def observe_dispatch_timing(
        self,
        *,
        t_gpu_ms: float,
        t_io_ms: float,
        t_wait_ms: float,
        items_dispatched: int,
    ) -> Tuple[int, int, str]:
        """Update `(batch_span, in_flight_cap, mirror_state)` purely from physical GPU-vs-IO timings via canonical L4."""
        tg = max(0.0, float(t_gpu_ms))
        ti = max(0.0, float(t_io_ms))
        tw = max(0.0, float(t_wait_ms))
        self.timing_updates += 1
        self.total_gpu_ms += tg
        self.total_io_ms += ti
        self.total_wait_ms += tw

        denom = tg + ti + 1e-6
        cur_g = tg / denom
        cur_i = ti / denom
        cur_w = min(1.0, tw / max(tg + tw, ti, 1e-6))

        bal = eval_mirror_equality_step(
            (self._prev_gpu_share - self._prev_io_share, 0.5 - self._prev_wait_share),
            (cur_g - cur_i, 0.5 - cur_w),
            (0.0, 0.5),
            (0.0, 0.5),
        )
        self._prev_gpu_share = cur_g
        self._prev_io_share = cur_i
        self._prev_wait_share = cur_w

        # Adapt burst span strictly when canonical L4 reports non-rest (TENSION / SPLIT)
        if items_dispatched >= self.min_batch_span and bal.state in ("TENSION", "SPLIT"):
            old_span = self.batch_span
            if cur_i > cur_g and cur_w > 0.0:
                ratio = max(0.5, min(0.875, tg / max(ti, 1e-6)))
                new_span = max(self.min_batch_span, min(old_span - 1, int(round(old_span * ratio))))
                if new_span < old_span:
                    self.batch_span = new_span
                    self.span_shrinks += 1
                if self.dispatch_mode != "THROTTLE_DAMP" and self.dispatch_mode != "SHED_PROTECT":
                    self.in_flight_cap = self.max_in_flight
            elif cur_g > cur_i and cur_w <= 0.1 and items_dispatched >= old_span:
                ratio = min(1.5, 0.5 * (1.0 + tg / max(ti, 1e-6)))
                new_span = min(self.max_batch_span, max(old_span + 1, int(round(old_span * ratio))))
                if new_span > old_span:
                    self.batch_span = new_span
                    self.span_expands += 1

        return self.batch_span, self.in_flight_cap, bal.state

    def can_admit_bytes(self, needed_bytes: int, *, margin_bytes: int = 0) -> bool:
        """Pure physical admission check on GPU free memory (Mechanism M1 ONLY)."""
        needed = max(0, int(needed_bytes))
        if needed == 0 or not self.enabled:
            self.admissions_allowed += 1
            return True
        free_b, total_b, _gu, _mu, _pf, _src = self._read_hardware()
        if total_b <= 0:
            self.admissions_allowed += 1
            return True
        floor_b = self.safety_reserve_bytes + max(0, int(margin_bytes))
        if free_b - needed >= floor_b:
            self.admissions_allowed += 1
            return True
        self.admissions_denied += 1
        return False

    def record_oom_recovery(self) -> None:
        self.oom_recoveries += 1

    def stats(self) -> Dict[str, Any]:
        return {
            "mechanism": "M1_mirror_load_dispatcher",
            "conveyor_removed": True,
            "enabled": self.enabled,
            "sensor": self.sensor_source,
            "dispatch_mode": self.dispatch_mode,
            "batch_span": self.batch_span,
            "batch_span_bounds": [self.min_batch_span, self.max_batch_span],
            "in_flight_cap": self.in_flight_cap,
            "safety_reserve_mib": round(self.safety_reserve_bytes / 1024**2, 1),
            "deadband_mib": round(self.deadband_bytes / 1024**2, 1),
            "evaluations": self.evaluations,
            "timing_updates": self.timing_updates,
            "direct_burst_steps": self.direct_burst_steps,
            "overlap_async_steps": self.overlap_async_steps,
            "throttle_damp_steps": self.throttle_damp_steps,
            "shed_protect_steps": self.shed_protect_steps,
            "equal_holds": self.equal_holds,
            "expansions": self.expansions,
            "span_shrinks": self.span_shrinks,
            "span_expands": self.span_expands,
            "oom_recoveries": self.oom_recoveries,
            "admissions_allowed": self.admissions_allowed,
            "admissions_denied": self.admissions_denied,
            "total_gpu_ms": round(self.total_gpu_ms, 2),
            "total_io_ms": round(self.total_io_ms, 2),
            "total_wait_ms": round(self.total_wait_ms, 2),
            "min_observed_free_vram_mib": (
                None if self.min_observed_free_bytes is None
                else round(self.min_observed_free_bytes / 1024**2, 1)
            ),
            "max_observed_budget_mib": round(self.max_observed_budget_bytes / 1024**2, 1),
            "max_observed_gpu_util_pct": self.max_observed_gpu_util,
            "max_observed_mem_util_pct": self.max_observed_mem_util,
            "last_mirror_state": self.last_mirror_state,
            "rssf_balance": {
                "last_scar_norm": round(self.last_scar_norm, 5),
                "fds_t_end_steps": (
                    None if self.last_fds_t_end_steps is None else round(self.last_fds_t_end_steps, 3)
                ),
                "volterra_archive": round(self.last_volterra_archive, 5),
                "ec_fds_damp_steps": self.ec_fds_damp_steps,
                "balance_scale_break_steps": self.balance_scale_break_steps,
                "collapse_events": self.collapse_events,
                "last_balance": self.last_balance.as_dict() if self.last_balance is not None else None,
                "last_collapse": self.last_collapse.as_dict() if self.last_collapse is not None else None,
            },
        }

    def close(self) -> None:
        if self._monitor is not None:
            self._monitor.close()
            self._monitor = None


# ---------------------------------------------------------------------------
# Mechanism M2: MirrorRegionBreathingPool (Replaces Rigid A/B/C(0) Ring!)
# ---------------------------------------------------------------------------
@dataclass
class _BreathingRegion:
    region_key: Tuple[Any, ...]
    payload: Any
    gpu_bytes: int
    ready_event: Any
    consumer_event: Any = None
    reuse_count: int = 0
    recurring_hint: bool = False
    last_step: int = 0
    state: str = "READY"          # "READY" | "ACTIVE" | "WARM" | "EXHALING"


class MirrorRegionBreathingPool:
    """Mechanism M2 (Conveyor Replacement #2): Self-contained Mirror Equality Region Breathing Pool.

    COMPLETELY REPLACES the rigid 3-slot `A -> B -> C(0) -> A` conveyor (`Q1RoleCycleArena`):
      - NO 3-area ring, NO `rotate()`, NO forced empty standby slot `C(0)`.
      - Regions live by hardware CUDA Events (`ready_event` -> `consumer_stream.wait_event` +
        `record_stream`, then `consumer_event` fence before reclamation).
      - Uses its OWN center-read Mirror Equality on region reuse density & occupancy to decide:
          1. `PIN_REUSE`        : requested region already live -> instant zero-copy reuse!
          2. `DIRECT_ADMIT`     : newly prepared region enters compute stream directly!
          3. `HOLD_WARM_REGION` : recurring region stays warm in VRAM (up to adaptive breathing cap)
                                  without being wiped through a conveyor ring!
          4. `EXHALE_RECLAIM`   : one-off or excess finished regions are exhaled immediately once
                                  their CUDA consumer event finishes, keeping transient VRAM flat!

    Holds ZERO references to M1 (load dispatcher), M3/M4 (warmup), or M5 (MoE layer).
    """

    def __init__(
        self,
        device: torch.device,
        *,
        max_pool_bytes: int = 512 * 1024**2,
        max_warm_regions: int = 4,
    ) -> None:
        self.device = torch.device(device)
        self.max_pool_bytes = max(1, int(max_pool_bytes))
        self.max_warm_regions = max(1, int(max_warm_regions))

        self._regions: "OrderedDict[Tuple[Any, ...], _BreathingRegion]" = OrderedDict()
        self._exhaling: List[_BreathingRegion] = []
        self._live_bytes: int = 0
        self._step_clock: int = 0

        # Own physical breathing history (Mechanism M2 ONLY)
        self._prev_reuse_rate: float = 0.0
        self._prev_occ_frac: float = 0.0
        self._recent_outcomes: deque[int] = deque(maxlen=16)
        self._volterra_fds = VolterraFDSMonitor(archive_decay=0.80)
        self._last_rssf_balance: Optional[DynamicBalanceSnapshot] = None
        self.rssf_scale_breaks: int = 0

        # Own telemetry
        self.admits_direct: int = 0
        self.pin_reuses: int = 0
        self.warm_holds: int = 0
        self.exhales_immediate: int = 0
        self.exhales_reaped: int = 0
        self.peak_live_bytes: int = 0
        self.peak_live_regions: int = 0
        self.last_mirror_state: str = "IDLE"

    @staticmethod
    def _record_stream_recursive(payload: Any, stream: Any) -> None:
        if isinstance(payload, torch.Tensor):
            if payload.device.type == "cuda":
                payload.record_stream(stream)
        elif isinstance(payload, dict):
            for v in payload.values():
                MirrorRegionBreathingPool._record_stream_recursive(v, stream)
        elif isinstance(payload, (tuple, list)):
            for v in payload:
                MirrorRegionBreathingPool._record_stream_recursive(v, stream)

    def reap_exhaled(self) -> int:
        """Reclaim any exhaling regions whose CUDA consumer event has physically completed."""
        if not self._exhaling:
            return 0
        still: List[_BreathingRegion] = []
        reaped = 0
        for reg in self._exhaling:
            ev = reg.consumer_event
            if ev is None or self.device.type != "cuda" or ev.query():
                reg.payload = None
                reg.ready_event = None
                reg.consumer_event = None
                reaped += 1
            else:
                still.append(reg)
        self._exhaling = still
        self.exhales_reaped += reaped
        return reaped

    def lookup_reuse(self, region_key: Tuple[Any, ...]) -> Optional[Any]:
        """PIN_REUSE: return live region payload immediately with zero copy if present in pool."""
        self.reap_exhaled()
        reg = self._regions.get(region_key)
        if reg is None:
            return None
        self._step_clock += 1
        reg.reuse_count += 1
        reg.last_step = self._step_clock
        reg.state = "ACTIVE"
        self._regions.move_to_end(region_key)
        if reg.ready_event is not None and self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(reg.ready_event)
            self._record_stream_recursive(reg.payload, stream)
        self.pin_reuses += 1
        self._recent_outcomes.append(1)
        return reg.payload

    def admit_region(
        self,
        region_key: Tuple[Any, ...],
        payload: Any,
        *,
        gpu_bytes: int = 0,
        ready_event: Any = None,
        recurring_hint: bool = False,
    ) -> Any:
        """DIRECT_ADMIT: register freshly prepared region directly onto compute stream (NO ring!)."""
        self.reap_exhaled()
        g_bytes = max(0, int(gpu_bytes))
        if g_bytes > self.max_pool_bytes:
            raise ValueError(f"Region payload ({g_bytes} B) exceeds pool limit ({self.max_pool_bytes} B)")

        if region_key in self._regions:
            old = self._regions.pop(region_key)
            self._live_bytes = max(0, self._live_bytes - old.gpu_bytes)
            self._queue_exhale(old)

        # Ensure breathing headroom by exhaling oldest non-ACTIVE warm regions if needed
        while self._regions and (
            self._live_bytes + g_bytes > self.max_pool_bytes
            or len(self._regions) >= self.max_warm_regions + 2
        ):
            victim_key = None
            for k, r in self._regions.items():
                if r.state != "ACTIVE":
                    victim_key = k
                    break
            if victim_key is None:
                break
            victim = self._regions.pop(victim_key)
            self._live_bytes = max(0, self._live_bytes - victim.gpu_bytes)
            self._queue_exhale(victim)

        self._step_clock += 1
        reg = _BreathingRegion(
            region_key=region_key,
            payload=payload,
            gpu_bytes=g_bytes,
            ready_event=ready_event,
            recurring_hint=bool(recurring_hint),
            last_step=self._step_clock,
            state="ACTIVE",
        )
        self._regions[region_key] = reg
        self._live_bytes += g_bytes
        if self._live_bytes > self.peak_live_bytes:
            self.peak_live_bytes = self._live_bytes
        if len(self._regions) > self.peak_live_regions:
            self.peak_live_regions = len(self._regions)

        if ready_event is not None and self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(ready_event)
            self._record_stream_recursive(payload, stream)

        self.admits_direct += 1
        self._recent_outcomes.append(0)
        return payload

    def finish_region(self, region_key: Tuple[Any, ...], *, keep_if_reusable: bool = False) -> str:
        """Mirror Equality lifecycle decision after compute finishes: HOLD_WARM_REGION vs EXHALE_RECLAIM."""
        reg = self._regions.get(region_key)
        if reg is None:
            return "EXHALE_RECLAIM"

        if self.device.type == "cuda" and torch.cuda.is_available():
            ev = torch.cuda.Event(enable_timing=False)
            ev.record(torch.cuda.current_stream(self.device))
            reg.consumer_event = ev

        occ_frac = min(1.0, self._live_bytes / max(1, self.max_pool_bytes))
        reuse_rate = (
            sum(self._recent_outcomes) / len(self._recent_outcomes)
            if self._recent_outcomes
            else 0.0
        )
        want_warm = bool(keep_if_reusable or reg.recurring_hint or reg.reuse_count > 0)

        # Canonical L4 center-read Mirror Equality on real reuse density vs breathing headroom (natural 1/2 threshold)
        # coupled with local EC Volterra archive + FDS horizon + 8-pair force balance (ZERO cross-scale refs!).
        free_frac = max(0.05, 1.0 - occ_frac)
        prev_free_frac = max(0.05, 1.0 - self._prev_occ_frac)
        reuse_obs = max(0.05, reuse_rate if reg.reuse_count == 0 else max(0.5, reuse_rate))
        reuse_prev = max(0.05, self._prev_reuse_rate)
        reg_frac = reg.gpu_bytes / max(1, self.max_pool_bytes)
        warm_count = sum(1 for r in self._regions.values() if r.state == "WARM")

        proj_free = (free_frac - reg_frac) if occ_frac <= 0.5 else -2.0 * (occ_frac - 0.5)
        if is_rssf_balance_enabled():
            ec_m2 = self._volterra_fds.update(
                free_bytes=max(0, self.max_pool_bytes - self._live_bytes),
                total_bytes=self.max_pool_bytes,
                reserve_bytes=max(1, int(self.max_pool_bytes * 0.15)),
            )
            rssf_m2 = observe_region_breathing_balance(
                occ_frac=occ_frac,
                reuse_rate=reuse_rate,
                reg_frac=reg_frac,
                sustained_drain=ec_m2.sustained_drain_rate,
                warm_fill_frac=float(warm_count) / float(max(1, self.max_warm_regions)),
            )
            self._last_rssf_balance = rssf_m2
            if rssf_m2.regime == "SCALE_BREAK":
                self.rssf_scale_breaks += 1
                proj_free = min(proj_free, -max(0.25, occ_frac))
            elif ec_m2.sustained_drain_rate >= 0.015 and ec_m2.fds_t_end_steps <= 3.0:
                proj_free = min(proj_free, -max(0.15, ec_m2.sustained_drain_rate * 4.0))
        else:
            self._last_rssf_balance = None

        out_prev = (0.0, 0.0) if self._step_clock <= 1 else (reuse_prev, prev_free_frac)
        obs_prev = (reuse_obs, free_frac)
        obs_next = (reuse_obs if want_warm else -reuse_obs, proj_free)
        out_next = (max(0.5, reuse_obs) if want_warm else 0.05, 1.0)
        bal = eval_mirror_equality_step(out_prev, obs_prev, obs_next, out_next)

        self.last_mirror_state = bal.state
        self._prev_reuse_rate = reuse_obs
        self._prev_occ_frac = occ_frac

        if want_warm and bal.state != "SPLIT" and warm_count < self.max_warm_regions:
            reg.state = "WARM"
            self.warm_holds += 1
            return "HOLD_WARM_REGION"

        # Immediate exhale: remove from active pool and release after CUDA consumer fence
        self._regions.pop(region_key, None)
        self._live_bytes = max(0, self._live_bytes - reg.gpu_bytes)
        self._queue_exhale(reg)
        self.reap_exhaled()
        return "EXHALE_RECLAIM"

    def _queue_exhale(self, reg: _BreathingRegion) -> None:
        reg.state = "EXHALING"
        self.exhales_immediate += 1
        if reg.consumer_event is None or self.device.type != "cuda" or reg.consumer_event.query():
            reg.payload = None
            reg.ready_event = None
            reg.consumer_event = None
            self.exhales_reaped += 1
        else:
            self._exhaling.append(reg)

    @property
    def live_bytes(self) -> int:
        return self._live_bytes

    @property
    def live_regions(self) -> int:
        return len(self._regions)

    def stats(self) -> Dict[str, Any]:
        self.reap_exhaled()
        return {
            "mechanism": "M2_mirror_region_breathing_pool",
            "conveyor_removed": True,
            "live_regions": len(self._regions),
            "live_mib": round(self._live_bytes / 1024**2, 2),
            "peak_live_regions": self.peak_live_regions,
            "peak_live_mib": round(self.peak_live_bytes / 1024**2, 2),
            "exhaling_pending": len(self._exhaling),
            "admits_direct": self.admits_direct,
            "pin_reuses": self.pin_reuses,
            "warm_holds": self.warm_holds,
            "exhales_immediate": self.exhales_immediate,
            "exhales_reaped": self.exhales_reaped,
            "last_mirror_state": self.last_mirror_state,
            "rssf_balance": (
                {
                    **self._last_rssf_balance.as_dict(),
                    "scale_breaks": self.rssf_scale_breaks,
                }
                if self._last_rssf_balance is not None
                else None
            ),
        }

    def close(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        for reg in list(self._regions.values()):
            reg.payload = None
            reg.ready_event = None
            reg.consumer_event = None
        self._regions.clear()
        for reg in self._exhaling:
            reg.payload = None
            reg.ready_event = None
            reg.consumer_event = None
        self._exhaling.clear()
        self._live_bytes = 0


# ---------------------------------------------------------------------------
# Scale M3: MirrorColdWarmupScale (Startup Kernel JIT & Cold Trunk Warmup)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ColdWarmupStepDecision:
    continue_warmup: bool
    mirror_state: str
    step_ms: float
    plateau_streak: int
    reason: str


class MirrorColdWarmupScale:
    """Scale M3: Self-contained Mirror Equality controller for cold startup warmup.

    Holds ZERO references to M1, M2, M4, or M5.
    """

    def __init__(
        self,
        *,
        mode: str = "plateau",
        min_layers_warm: int = 2,
        plateau_confirm: int = 2,
        max_warmup_seconds: float = 6.0,
    ) -> None:
        if mode not in ("plateau", "full_resident"):
            raise ValueError("MirrorColdWarmupScale mode must be 'plateau' or 'full_resident'")
        self.mode = mode
        self.min_layers_warm = max(1, int(min_layers_warm))
        self.plateau_confirm = max(1, int(plateau_confirm))
        self.max_warmup_seconds = max(0.0, float(max_warmup_seconds))

        self._prev_ms: Optional[float] = None
        self._ema_ms: Optional[float] = None
        self._volterra_latency_drift: float = 0.0
        self._last_rssf_balance: Optional[DynamicBalanceSnapshot] = None
        self.fds_equilibrium_confirmations: int = 0
        self.rssf_scale_breaks: int = 0
        self._plateau_streak: int = 0
        self.kernels_warmed: int = 0
        self.kernel_warmup_ms: float = 0.0
        self.layers_warmed: int = 0
        self.total_warmup_ms: float = 0.0
        self.step_timings_ms: List[float] = []
        self.stopped_reason: str = "not_run"
        self.last_mirror_state: str = "IDLE"

    def evaluate_step(
        self,
        step_ms: float,
        *,
        warmed_so_far: int,
        total_targets: int,
        elapsed_seconds: float,
    ) -> ColdWarmupStepDecision:
        dt = max(1e-4, float(step_ms))
        self.step_timings_ms.append(round(dt, 3))
        done_frac = min(1.0, float(warmed_so_far) / max(1, int(total_targets)))

        if self._prev_ms is None:
            self._prev_ms = dt
            self._ema_ms = dt
            self._volterra_latency_drift = 0.0
            bal0 = eval_mirror_equality_step((0.0, 0.0), (1.0, max(1e-4, done_frac)), (1.0, max(1e-4, done_frac)), (1.0, 1.0))
            self.last_mirror_state = bal0.state  # canonical UNKNOWN on unclosed first step
            if warmed_so_far >= total_targets:
                return ColdWarmupStepDecision(False, "EQUAL", dt, 1, "all_warm")
            if self.max_warmup_seconds > 0.0 and elapsed_seconds >= self.max_warmup_seconds:
                return ColdWarmupStepDecision(False, bal0.state, dt, 0, "time_budget")
            return ColdWarmupStepDecision(True, bal0.state, dt, 0, "warming")

        scale_ref = max(self._prev_ms, dt, 1e-4)
        p_norm = self._prev_ms / scale_ref
        c_norm = dt / scale_ref
        ratio = dt / max(self._prev_ms, 1e-4)
        rssf_m3 = None
        if is_rssf_balance_enabled():
            inst_rel = (dt - self._prev_ms) / scale_ref
            self._ema_ms = dt if self._ema_ms is None else (0.65 * self._ema_ms + 0.35 * dt)
            self._volterra_latency_drift = 0.75 * self._volterra_latency_drift + inst_rel
            fds_power = abs(inst_rel) * (1.0 + 0.25 / max(0.05, done_frac))
            rssf_m3 = observe_warmup_fds_balance(
                step_ms=dt,
                ema_ms=self._ema_ms,
                prev_ms=self._prev_ms,
                done_frac=done_frac,
                fds_power=fds_power,
            )
        self._last_rssf_balance = rssf_m3

        # Signed speed delta along natural 1/2 mirror scale: sharp drop (<0.5x or >1.5x) pushes into TENSION/SPLIT; plateau -> EQUAL
        out_prev = (p_norm, 1.0)
        obs_prev = (c_norm, ratio if abs(ratio - 1.0) < 0.5 else -(abs(ratio - 1.0)))
        obs_next = (c_norm, 1.0)
        out_next = (c_norm, 1.0)

        bal = eval_mirror_equality_step(out_prev, obs_prev, obs_next, out_next)
        self._prev_ms = dt
        self.last_mirror_state = bal.state

        if rssf_m3 is not None and rssf_m3.regime == "SCALE_BREAK":
            self.rssf_scale_breaks += 1
            self._plateau_streak = 0
        elif bal.state == "EQUAL":
            self._plateau_streak += 1
            if rssf_m3 is not None:
                sustained_rel_drift = 0.25 * self._volterra_latency_drift
                if rssf_m3.d3_margin > 0.0 and abs(sustained_rel_drift) <= 0.20:
                    self.fds_equilibrium_confirmations += 1
        else:
            self._plateau_streak = 0

        if warmed_so_far >= total_targets:
            return ColdWarmupStepDecision(False, bal.state, dt, self._plateau_streak, "all_warm")
        if self.max_warmup_seconds > 0.0 and elapsed_seconds >= self.max_warmup_seconds:
            return ColdWarmupStepDecision(False, bal.state, dt, self._plateau_streak, "time_budget")
        if (
            self.mode == "plateau"
            and warmed_so_far >= self.min_layers_warm
            and self._plateau_streak >= self.plateau_confirm
        ):
            return ColdWarmupStepDecision(False, bal.state, dt, self._plateau_streak, "mirror_plateau_equal")
        return ColdWarmupStepDecision(True, bal.state, dt, self._plateau_streak, "warming")

    def run_warmup(
        self,
        *,
        kernel_warmup_fn: Optional[Callable[[], int]] = None,
        layer_warmup_fn: Optional[Callable[[int], bool]] = None,
        total_resident_layers: int = 0,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if kernel_warmup_fn is not None:
            tk0 = time.perf_counter()
            self.kernels_warmed += int(kernel_warmup_fn() or 0)
            self.kernel_warmup_ms = (time.perf_counter() - tk0) * 1000.0

        total_res = max(0, int(total_resident_layers))
        if layer_warmup_fn is None or total_res == 0:
            self.total_warmup_ms = (time.perf_counter() - t0) * 1000.0
            self.stopped_reason = "kernels_only"
            return self.stats()

        self.stopped_reason = "all_warm"
        for li in range(total_res):
            elapsed = time.perf_counter() - t0
            if self.max_warmup_seconds > 0.0 and elapsed >= self.max_warmup_seconds:
                self.stopped_reason = "time_budget"
                break
            ts = time.perf_counter()
            did_work = bool(layer_warmup_fn(li))
            step_ms = (time.perf_counter() - ts) * 1000.0
            self.layers_warmed += 1
            if not did_work:
                continue
            dec = self.evaluate_step(
                step_ms,
                warmed_so_far=self.layers_warmed,
                total_targets=total_res,
                elapsed_seconds=time.perf_counter() - t0,
            )
            if not dec.continue_warmup:
                self.stopped_reason = dec.reason
                break

        self.total_warmup_ms = (time.perf_counter() - t0) * 1000.0
        return self.stats()

    def stats(self) -> Dict[str, Any]:
        return {
            "scale": "M3_cold_warmup",
            "mode": self.mode,
            "kernels_warmed": self.kernels_warmed,
            "kernel_warmup_ms": round(self.kernel_warmup_ms, 2),
            "layers_warmed": self.layers_warmed,
            "total_warmup_ms": round(self.total_warmup_ms, 2),
            "plateau_streak": self._plateau_streak,
            "stopped_reason": self.stopped_reason,
            "last_mirror_state": self.last_mirror_state,
            "rssf_balance": (
                {
                    **self._last_rssf_balance.as_dict(),
                    "fds_equilibrium_confirmations": self.fds_equilibrium_confirmations,
                    "scale_breaks": self.rssf_scale_breaks,
                }
                if self._last_rssf_balance is not None
                else None
            ),
        }


# ---------------------------------------------------------------------------
# Scale M4: MirrorPostPrefillRegionWarmupScale (Post-Prefill Known Areas)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PostPrefillWarmupPlan:
    target_layer: int
    eids_to_warm: Tuple[int, ...]
    mirror_state: str             # "EQUAL" | "ALIGN" | "SPLIT" | "PROTECT" | "SATURATED" | "NO_MAP"
    prefill_coverage: float
    scar_fraction: float


class MirrorPostPrefillRegionWarmupScale:
    """Scale M4: Self-contained Mirror Equality controller for warming KNOWN post-prefill areas.

    NO Graphify! NO cross-scale links! Holds ZERO references to M1, M2, M3, or M5.
    """

    def __init__(
        self,
        num_layers: int,
        *,
        max_region_experts: int = 10,
        initial_width: int = 1,
        max_width: int = 2,
    ) -> None:
        self.num_layers = max(1, int(num_layers))
        self.max_region_experts = max(4, int(max_region_experts))
        self.max_width = max(1, int(max_width))
        init_w = max(1, min(self.max_width, int(initial_width)))

        self._prefill_ranked: Dict[int, Tuple[Tuple[int, float, bool], ...]] = {}
        self._prefill_sets: Dict[int, frozenset[int]] = {}
        self._scar_weights: Dict[int, Dict[int, float]] = defaultdict(dict)
        self._scar_hits: Dict[int, Dict[int, int]] = defaultdict(dict)
        self._recent_decode: Dict[int, Dict[int, float]] = defaultdict(dict)

        self._prev_cov: List[float] = [1.0] * self.num_layers
        self._prev_scar: List[float] = [0.0] * self.num_layers
        self._width: List[int] = [init_w] * self.num_layers
        self._prev_hit_rate: List[float] = [0.5] * self.num_layers
        self._miss_streak: List[int] = [0] * self.num_layers

        self.prefill_layers_mapped: int = 0
        self.plans_emitted: int = 0
        self.plans_saturated: int = 0
        self.split_scar_updates: int = 0
        self.lookahead_l2_plans: int = 0
        self.warmed_experts_submitted: int = 0
        self.warmed_experts_hit: int = 0
        self.warmed_experts_checked: int = 0
        self.last_mirror_state: str = "IDLE"
        self._last_rssf_balance: Optional[DynamicBalanceSnapshot] = None
        self.rssf_scale_breaks: int = 0

    def record_prefill_layer(
        self,
        layer_idx: int,
        body_weights: Dict[int, float],
        tail_weights: Dict[int, float],
    ) -> None:
        li = int(layer_idx)
        all_eids = set(body_weights.keys()) | set(tail_weights.keys())
        if not all_eids:
            return
        max_body = max(body_weights.values(), default=1.0) or 1.0
        max_tail = max(tail_weights.values(), default=1.0) or 1.0

        scored: List[Tuple[float, bool, int]] = []
        for eid in all_eids:
            tw = float(tail_weights.get(eid, 0.0))
            bw = float(body_weights.get(eid, 0.0))
            is_tail = tw > 0.0
            prio = (2.0 * (tw / max_tail) if is_tail else 0.0) + (bw / max_body)
            scored.append((prio, is_tail, int(eid)))

        scored.sort(key=lambda item: (-item[0], item[2]))
        top = scored[: self.max_region_experts]
        first = li not in self._prefill_ranked
        self._prefill_ranked[li] = tuple((eid, prio, is_tail) for prio, is_tail, eid in top)
        self._prefill_sets[li] = frozenset(eid for _p, _t, eid in top)
        if first:
            self.prefill_layers_mapped += 1
        self._scar_weights[li].clear()
        self._scar_hits[li].clear()
        self._recent_decode[li].clear()
        if 0 <= li < self.num_layers:
            self._prev_cov[li] = 1.0
            self._prev_scar[li] = 0.0
            self._miss_streak[li] = 0

    def observe_and_plan_next(
        self,
        layer_idx: int,
        actual_eids: Sequence[int],
        actual_weights: Sequence[float],
        is_physically_resident_fn: Callable[[int, int], bool],
    ) -> PostPrefillWarmupPlan:
        li = int(layer_idx)
        prefill_set = self._prefill_sets.get(li, frozenset())
        tot_w = sum(max(0.0, float(w)) for w in actual_weights) or 1.0

        cov_w = 0.0
        scar_eids: List[Tuple[int, float]] = []
        rec_map = self._recent_decode[li]
        for rk in list(rec_map.keys()):
            rec_map[rk] *= 0.72
            if rec_map[rk] < 1e-3:
                del rec_map[rk]
        for eid, rw in zip(actual_eids, actual_weights):
            e_int = int(eid)
            w_val = max(0.0, float(rw))
            rec_map[e_int] = rec_map.get(e_int, 0.0) + w_val
            if e_int in prefill_set:
                cov_w += w_val
            else:
                scar_eids.append((e_int, w_val))

        cur_cov = min(1.0, cov_w / tot_w) if prefill_set else 0.0
        cur_scar = max(0.0, 1.0 - cur_cov)

        prev_cov = self._prev_cov[li] if 0 <= li < self.num_layers else 1.0
        prev_scar = self._prev_scar[li] if 0 <= li < self.num_layers else 0.0

        bal = eval_mirror_equality_step(
            (max(1e-4, prev_cov), max(1e-4, 1.0 - prev_scar)),
            (max(1e-4, cur_cov), max(1e-4, 1.0 - cur_scar)),
            (cur_cov if cur_cov >= 0.5 else -max(0.1, cur_scar), (1.0 - cur_scar) if cur_cov >= 0.5 else -max(0.1, cur_scar)),
            (1.0, 1.0),
        )
        self.last_mirror_state = bal.state
        if 0 <= li < self.num_layers:
            self._prev_cov[li] = cur_cov
            self._prev_scar[li] = cur_scar

        rssf_m4 = None
        if is_rssf_balance_enabled():
            rssf_m4 = observe_post_prefill_region_balance(
                cur_cov=cur_cov,
                cur_scar=cur_scar,
                hit_rate=(self._prev_hit_rate[li] if 0 <= li < self.num_layers else 0.5),
                miss_streak=(self._miss_streak[li] if 0 <= li < self.num_layers else 0),
                scar_ratio=float(len(self._scar_weights[li])) / float(max(1, self.max_region_experts)),
            )
            if rssf_m4.regime == "SCALE_BREAK":
                self.rssf_scale_breaks += 1
        self._last_rssf_balance = rssf_m4

        if scar_eids and (
            bal.state in ("TENSION", "SPLIT")
            or (rssf_m4 is not None and rssf_m4.regime == "SCALE_BREAK")
        ):
            sc_map = self._scar_weights[li]
            sh_map = self._scar_hits[li]
            for k in list(sc_map.keys()):
                sc_map[k] *= 0.5
                if sc_map[k] < 1e-4:
                    del sc_map[k]
                    sh_map.pop(k, None)
            mean_slot_w = tot_w / max(1, len(actual_weights))
            for e_int, w_val in scar_eids:
                if w_val >= 0.5 * mean_slot_w:
                    sc_map[e_int] = sc_map.get(e_int, 0.0) + w_val
                    sh_map[e_int] = sh_map.get(e_int, 0) + 1
            self.split_scar_updates += 1

        for offset in (1, 2):
            target_l = li + offset
            if target_l >= self.num_layers:
                break
            if self._miss_streak[target_l] >= 3 and bal.state == "EQUAL":
                continue
            ranked_prefill = self._prefill_ranked.get(target_l)
            sc_target = self._scar_weights.get(target_l)
            sh_target = self._scar_hits.get(target_l)
            if not ranked_prefill and not sc_target:
                continue

            ordered_cands: List[int] = []
            seen: set[int] = set()
            if sc_target:
                for e_int, _sw in sorted(
                    sc_target.items(),
                    key=lambda kv: (-sh_target.get(kv[0], 0), -kv[1], kv[0]),
                )[:4]:
                    if e_int not in seen:
                        ordered_cands.append(e_int)
                        seen.add(e_int)
            if ranked_prefill:
                for e_int, _prio, _is_tail in ranked_prefill:
                    if e_int not in seen:
                        ordered_cands.append(e_int)
                        seen.add(e_int)

            cold_cands = [e for e in ordered_cands if not is_physically_resident_fn(target_l, e)]
            if cold_cands:
                w_cap = self._width[target_l] if 0 <= target_l < self.num_layers else 1
                if rssf_m4 is not None and rssf_m4.regime == "SCALE_BREAK":
                    w_cap = 1
                chosen = tuple(cold_cands[:w_cap])
                self.plans_emitted += 1
                self.warmed_experts_submitted += len(chosen)
                if offset == 2:
                    self.lookahead_l2_plans += 1
                return PostPrefillWarmupPlan(
                    target_layer=target_l,
                    eids_to_warm=chosen,
                    mirror_state=bal.state,
                    prefill_coverage=round(cur_cov, 4),
                    scar_fraction=round(cur_scar, 4),
                )

        self.plans_saturated += 1
        return PostPrefillWarmupPlan(
            target_layer=min(self.num_layers - 1, li + 1),
            eids_to_warm=(),
            mirror_state="SATURATED" if self._prefill_ranked else "NO_MAP",
            prefill_coverage=round(cur_cov, 4),
            scar_fraction=round(cur_scar, 4),
        )

    def plan_wide_layer_candidates(
        self,
        target_layer: int,
        is_physically_resident_fn: Callable[[int, int], bool],
        *,
        width: int = 8,
    ) -> List[int]:
        """Select a wide cold corridor for Linear CPU->GPU Raw Byte Injection on `target_layer`.

        Priority: live recent decode trajectory -> SPLIT scars -> immutable prefill map.
        CPU never computes or dequantizes these experts; it only streams raw bytes ahead of GPU.
        """
        tl = int(target_layer)
        if tl < 0 or tl >= self.num_layers:
            return []
        out: List[int] = []
        seen: set[int] = set()
        lim = max(1, int(width))

        def _try_add(eid: int) -> bool:
            if eid in seen or is_physically_resident_fn(tl, eid):
                return False
            seen.add(eid)
            out.append(eid)
            return len(out) >= lim

        rec = self._recent_decode.get(tl)
        if rec:
            for e_int, _w in sorted(rec.items(), key=lambda kv: (-kv[1], kv[0])):
                if _try_add(int(e_int)):
                    return out

        sc_t = self._scar_weights.get(tl)
        sh_t = self._scar_hits.get(tl)
        if sc_t:
            for e_int, _w in sorted(sc_t.items(), key=lambda kv: (-sh_t.get(kv[0], 0), -kv[1], kv[0])):
                if _try_add(int(e_int)):
                    return out

        ranked = self._prefill_ranked.get(tl)
        if ranked:
            for e_int, _prio, _is_tail in ranked:
                if _try_add(int(e_int)):
                    return out

        return out

    def record_warmup_outcome(self, layer_idx: int, warmed_eids: Sequence[int], actual_eids: Sequence[int]) -> None:
        if not warmed_eids:
            return
        li = int(layer_idx)
        actual_set = set(int(e) for e in actual_eids)
        hits = sum(1 for e in warmed_eids if int(e) in actual_set)
        self.warmed_experts_checked += len(warmed_eids)
        self.warmed_experts_hit += hits

        if not (0 <= li < self.num_layers):
            return
        hit_rate = hits / max(1, len(warmed_eids))
        prev_hr = self._prev_hit_rate[li]
        bal = eval_mirror_equality_step(
            (max(1e-4, prev_hr), max(1e-4, 0.5 + 0.5 * prev_hr)),
            (max(1e-4, hit_rate), max(1e-4, 0.5 + 0.5 * hit_rate)),
            (hit_rate if hit_rate >= 0.5 else -max(0.1, 1.0 - hit_rate), 1.0 if hit_rate >= 0.5 else -max(0.1, 1.0 - hit_rate)),
            (1.0, 1.0),
        )
        self._prev_hit_rate[li] = hit_rate
        if bal.state == "EQUAL":
            self._miss_streak[li] = 0
            self._width[li] = min(self.max_width, self._width[li] + (1 if hit_rate >= 0.99 else 0))
        elif bal.state == "SPLIT":
            self._miss_streak[li] += 1
            self._width[li] = max(1, self._width[li] - 1)
        else:
            self._miss_streak[li] = 0

    def stats(self) -> Dict[str, Any]:
        return {
            "scale": "M4_post_prefill_region_warmup",
            "prefill_layers_mapped": self.prefill_layers_mapped,
            "plans_emitted": self.plans_emitted,
            "plans_saturated": self.plans_saturated,
            "lookahead_l2_plans": self.lookahead_l2_plans,
            "split_scar_updates": self.split_scar_updates,
            "warmed_experts_submitted": self.warmed_experts_submitted,
            "warmed_experts_checked": self.warmed_experts_checked,
            "warmed_experts_hit": self.warmed_experts_hit,
            "hit_rate_pct": round(
                100.0 * self.warmed_experts_hit / max(1, self.warmed_experts_checked), 2
            ),
            "last_mirror_state": self.last_mirror_state,
            "rssf_balance": (
                {
                    **self._last_rssf_balance.as_dict(),
                    "scale_breaks": self.rssf_scale_breaks,
                }
                if self._last_rssf_balance is not None
                else None
            ),
        }


# ---------------------------------------------------------------------------
# Scale M5 + Host Coordinator: MirrorIsolatedMoEOverlayV4 (CONVEYOR-FREE!)
# ---------------------------------------------------------------------------
def _tensor_dict_bytes(d: Dict[str, torch.Tensor]) -> int:
    return sum(int(t.numel()) * int(t.element_size()) for t in d.values() if isinstance(t, torch.Tensor))


def _trim_q1_planes(t: Dict[str, torch.Tensor], new_k: int) -> Dict[str, torch.Tensor]:
    cur_k = int(t["packed_lo"].shape[0])
    if new_k >= cur_k:
        return t
    out = dict(t)
    out["packed_lo"] = t["packed_lo"][:new_k].contiguous().clone()
    if int(t["qscale"].shape[0]) > 1:
        out["qscale"] = t["qscale"][:new_k].contiguous().clone()
    return out


def _trim_q1_planes_with_scar(
    t: Dict[str, torch.Tensor], new_k: int
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    """Trim trailing Q1 planes and return `(trimmed_gpu_dict, reversible_cpu_scar_planes)`."""
    cur_k = int(t["packed_lo"].shape[0])
    if new_k >= cur_k:
        return t, None
    scar: Dict[str, torch.Tensor] = {
        "packed_lo": t["packed_lo"][new_k:cur_k].detach().cpu().contiguous(),
    }
    if int(t["qscale"].shape[0]) > 1:
        scar["qscale"] = t["qscale"][new_k:cur_k].detach().cpu().contiguous()
    return _trim_q1_planes(t, new_k), scar


def _reattach_q1_scar(
    t: Dict[str, torch.Tensor], scar: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    """Reversibly reattach a previously shed Q1 bit-plane scar (`Expand o Compress`, 0 disk I/O)."""
    out = dict(t)
    s_lo = scar["packed_lo"].to(device=device, non_blocking=True)
    out["packed_lo"] = torch.cat([t["packed_lo"], s_lo], dim=0).contiguous()
    if "qscale" in scar and int(t["qscale"].shape[0]) > 1:
        s_sc = scar["qscale"].to(device=device, non_blocking=True)
        out["qscale"] = torch.cat([t["qscale"], s_sc], dim=0).contiguous()
    return out


def mirror_cube_depth_math(p_body: float, p_tail: float, min_k: int, max_k: int) -> int:
    """Pure-math M5b cube depth from bracketless center-read probability balance.

    (Dead M5b `observe_q1_cube_balance` branch removed after audit: 0/100k SCALE_BREAK reachability.)
    Given past body density `p_body in [0, 1]` and future tail density `p_tail in [0, 1]`:
        s = 0.5 * (p_body + p_tail)
        d = |p_body - p_tail| / (p_body + p_tail + 1e-8)
        rho = min(1.0, max(s, p_tail) * (1.0 + 0.5 * d * I[p_tail > p_body]))
        k = clamp(floor(min_k + rho * (max_k - min_k) + 0.5), min_k, max_k)
    """
    pb = max(0.0, min(1.0, float(p_body)))
    pt = max(0.0, min(1.0, float(p_tail)))
    s = 0.5 * (pb + pt)
    d = abs(pb - pt) / (pb + pt + 1e-8)
    growth_boost = (1.0 + 0.5 * d) if pt > pb else 1.0
    rho = min(1.0, max(s, pt) * growth_boost)
    k = int(math.floor(min_k + rho * max(0, max_k - min_k) + 0.5))
    return max(int(min_k), min(int(max_k), k))


class MirrorAcceleratorRouter:
    """Self-organizing physical accelerator router driven purely by canonical L4 Mirror Equality.

    ZERO hardcoded hardware thresholds (e.g., no fixed `T >= 384`): for each physical regime
    bucket (such as `seq_len.bit_length()`), measures real local latency & finite stability of
    candidate accelerators (e.g., native NVRTC vs Linux Triton FLA) and lets `eval_canonical_l4_mirror`
    (natural threshold 1/2, reversible failure scars) place the right accelerator automatically
    for THIS exact GPU / driver / input geometry.
    """

    def __init__(self, name: str = "accelerator_router") -> None:
        self.name = str(name)
        # bucket -> chosen backend name ("primary" | "candidate")
        self._route: Dict[Any, str] = {}
        # bucket -> telemetry record (ms_primary, ms_candidate, state, tie, delta, winner)
        self._records: Dict[Any, Dict[str, Any]] = {}
        # bucket -> reversible failure/NaN scar for candidate
        self._scars: Dict[Any, str] = {}

    def select_and_run(
        self,
        bucket: Any,
        primary_name: str,
        primary_fn: Callable[[], Any],
        candidate_name: Optional[str],
        candidate_fn: Optional[Callable[[], Any]],
        *,
        is_cuda: bool = False,
        validate_fn: Optional[Callable[[Any], bool]] = None,
    ) -> Tuple[Any, str]:
        # Fast steady-state path: 0 overhead once mirror equality has placed this bucket
        chosen = self._route.get(bucket)
        if chosen == primary_name or candidate_fn is None or candidate_name is None:
            return primary_fn(), primary_name
        if chosen == candidate_name:
            try:
                out = candidate_fn()
                if validate_fn is None or validate_fn(out):
                    return out, candidate_name
                self._scars[bucket] = f"{candidate_name}:nonfinite"
            except Exception as exc:
                self._scars[bucket] = f"{candidate_name}:{type(exc).__name__}"
            self._route[bucket] = primary_name
            return primary_fn(), primary_name

        # First contact for this physical bucket: measure real latency & stability of both sides
        from src.general_speculation import eval_canonical_l4_mirror

        def _timed_run(fn: Callable[[], Any]) -> Tuple[Any, float, float, Optional[str]]:
            try:
                if is_cuda and torch.cuda.is_available():
                    # One quick warmup run for candidate JIT if needed is folded naturally when
                    # primary runs first; time with CUDA events for sub-millisecond accuracy.
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev1 = torch.cuda.Event(enable_timing=True)
                    ev0.record()
                    res = fn()
                    ev1.record()
                    ev1.synchronize()
                    ms = max(1e-4, float(ev0.elapsed_time(ev1)))
                else:
                    t0 = time.perf_counter()
                    res = fn()
                    ms = max(1e-4, (time.perf_counter() - t0) * 1000.0)
                ok = True if validate_fn is None else bool(validate_fn(res))
                return res, ms, (1.0 if ok else 0.0), (None if ok else "nonfinite")
            except Exception as exc:
                return None, 1e9, 0.0, type(exc).__name__

        # Probe primary (baseline)
        out_p, ms_p, rel_p, err_p = _timed_run(primary_fn)

        # Probe candidate (warm once on CUDA so one-time Triton JIT compile does not distort steady speed)
        out_c, ms_c, rel_c, err_c = _timed_run(candidate_fn)
        if rel_c > 0.0 and is_cuda and torch.cuda.is_available():
            out_c, ms_c, rel_c, err_c = _timed_run(candidate_fn)
            # Re-time primary once warm as well for symmetric comparison
            out_p2, ms_p2, rel_p2, err_p2 = _timed_run(primary_fn)
            if rel_p2 > 0.0:
                out_p, ms_p, rel_p, err_p = out_p2, ms_p2, rel_p2, err_p2

        if rel_c <= 0.0:
            self._scars[bucket] = f"{candidate_name}:{err_c or 'failed'}"
            self._route[bucket] = primary_name
            self._records[bucket] = {
                "winner": primary_name,
                "ms_primary": round(ms_p, 4),
                "ms_candidate": None,
                "mirror_state": "SPLIT",
                "scar": self._scars[bucket],
            }
            return out_p, primary_name

        if rel_p <= 0.0:
            self._route[bucket] = candidate_name
            self._records[bucket] = {
                "winner": candidate_name,
                "ms_primary": None,
                "ms_candidate": round(ms_c, 4),
                "mirror_state": "SPLIT",
                "scar": f"{primary_name}:{err_p or 'failed'}",
            }
            return out_c, candidate_name

        # Canonical L4 Mirror Equality on physical throughput (1/ms) and stability (rel)
        sp_p = 1.0 / ms_p
        sp_c = 1.0 / ms_c
        s_ref = max(sp_p, sp_c, 1e-12)
        v_p = (sp_p / s_ref, rel_p)
        v_c = (sp_c / s_ref, rel_c)
        strat = (0.5 * (v_p[0] + v_c[0]), 0.5 * (v_p[1] + v_c[1]))
        state, tau_core, tau_wing, delta_res = eval_canonical_l4_mirror(v_p, v_p, strat, v_c, v_c)

        # Pure canonical L4 decision (ONLY natural threshold 1/2, zero magic constants):
        # Under EQUAL, keep zero-churn primary; under TENSION/SPLIT, place the faster hardware path.
        winner = primary_name if state == "EQUAL" else (candidate_name if sp_c > sp_p else primary_name)

        self._route[bucket] = winner
        self._records[bucket] = {
            "winner": winner,
            "ms_primary": round(ms_p, 4),
            "ms_candidate": round(ms_c, 4),
            "mirror_state": state,
            "tau_core": round(tau_core, 4),
            "tau_wing": round(tau_wing, 4),
            "delta_res": round(delta_res, 4),
        }
        return (out_c, candidate_name) if winner == candidate_name else (out_p, primary_name)

    def stats(self) -> Dict[str, Any]:
        return {
            "router": self.name,
            "routes": {str(k): v for k, v in sorted(self._route.items(), key=lambda kv: str(kv[0]))},
            "records": {str(k): v for k, v in sorted(self._records.items(), key=lambda kv: str(kv[0]))},
            "scars": {str(k): v for k, v in sorted(self._scars.items(), key=lambda kv: str(kv[0]))},
        }


class Q1DiskL3Cache:
    """Isolated physical L3 SSD cache for Q1 crystals evicted from L2 Host RAM.

    Pure memory-hierarchy tier (L1 VRAM -> L2 Host RAM -> L3 native ext4 SSD):
      - Stores compact Q1 safetensors crystals (`~0.5-1.5 MiB` vs `~6-12 MiB` raw BF16) so
        returning experts skip both raw BF16 disk reads and Q1 re-carving.
      - Uses native `ext4` (`~/q1_l3_cache/session_<pid>_<id>` on POSIX by default, NEVER `/tmp`
        tmpfs!) with bounded LRU eviction (`max_bytes`, default 2 GiB).
      - Single-thread async spill writer with in-memory `_pending` handoff (zero race conditions).
      - Strictly physical tiering — ZERO cross-scale mirror coupling.
    """

    def __init__(
        self,
        cache_dir: Optional[str | Path] = None,
        *,
        max_bytes: int = 2 * 1024**3,
    ) -> None:
        if cache_dir is not None:
            base_dir = Path(cache_dir)
        else:
            env_dir = os.environ.get("QUINTESSENTIA_L3_CACHE_DIR")
            base_dir = Path(env_dir) if env_dir else (Path.home() / "q1_l3_cache")
        self.base_dir = base_dir
        self.session_dir = base_dir / f"session_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max(0, int(max_bytes))

        self._lock = threading.Lock()
        self._pending: Dict[
            Tuple[int, int], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, Tuple[int, int]]
        ] = {}
        # key -> (safetensors_path, disk_bytes, logical_entry_bytes, (k_gu, k_dn))
        self._disk_lru: "OrderedDict[Tuple[int, int], Tuple[Path, int, int, Tuple[int, int]]]" = OrderedDict()
        self._disk_bytes: int = 0
        self._pool: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="quintessentia-l3-spill"
        )
        self._closed: bool = False

        # Telemetry
        self.spills: int = 0
        self.spills_deduped: int = 0
        self.hits: int = 0
        self.evictions: int = 0
        self.write_errors: int = 0

        atexit.register(self.close)

    def _entry_path(self, key: Tuple[int, int]) -> Path:
        return self.session_dir / f"L{int(key[0]):04d}_E{int(key[1]):05d}.safetensors"

    @staticmethod
    def _depth_sig(gu_t: Dict[str, torch.Tensor], dn_t: Dict[str, torch.Tensor]) -> Tuple[int, int]:
        kg = int(gu_t["packed_lo"].shape[0]) if "packed_lo" in gu_t else 0
        kd = int(dn_t["packed_lo"].shape[0]) if "packed_lo" in dn_t else 0
        return (kg, kd)

    def spill_async(
        self,
        key: Tuple[int, int],
        gu_t: Dict[str, torch.Tensor],
        dn_t: Dict[str, torch.Tensor],
        entry_bytes: int,
    ) -> None:
        if self._closed or self.max_bytes <= 0 or int(entry_bytes) > self.max_bytes:
            return
        sig = self._depth_sig(gu_t, dn_t)
        eb_i = int(entry_bytes)
        with self._lock:
            if self._closed or self._pool is None:
                return
            # Read-keep dedup: if an identical (entry_bytes, depth_sig) crystal is already
            # resident on ext4, refresh its LRU position with 0 disk writes!
            existing = self._disk_lru.get(key)
            if existing is not None and existing[2] == eb_i and existing[3] == sig:
                self._disk_lru.move_to_end(key)
                self.spills_deduped += 1
                return
        cpu_gu = {
            k: (v.detach().contiguous() if v.device.type == "cpu" else v.detach().cpu().contiguous())
            for k, v in gu_t.items()
        }
        cpu_dn = {
            k: (v.detach().contiguous() if v.device.type == "cpu" else v.detach().cpu().contiguous())
            for k, v in dn_t.items()
        }
        with self._lock:
            if self._closed or self._pool is None:
                return
            existing = self._disk_lru.get(key)
            if existing is not None and existing[2] == eb_i and existing[3] == sig:
                self._disk_lru.move_to_end(key)
                self.spills_deduped += 1
                return
            self._pending[key] = (cpu_gu, cpu_dn, eb_i, sig)
            pool = self._pool
        try:
            pool.submit(self._write_worker, key)
        except RuntimeError:
            with self._lock:
                self._pending.pop(key, None)

    def _write_worker(self, key: Tuple[int, int]) -> None:
        from safetensors.torch import save_file

        with self._lock:
            if self._closed:
                return
            pending_ent = self._pending.get(key)
            if pending_ent is None:
                return
            cpu_gu, cpu_dn, entry_bytes, sig = pending_ent

        flat: Dict[str, torch.Tensor] = {}
        for k, v in cpu_gu.items():
            flat[f"gu.{k}"] = v
        for k, v in cpu_dn.items():
            flat[f"dn.{k}"] = v

        dst_path = self._entry_path(key)
        tmp_path = self.session_dir / f"L{int(key[0]):04d}_E{int(key[1]):05d}.{uuid.uuid4().hex[:6]}.tmp"
        try:
            save_file(flat, str(tmp_path))
            os.replace(str(tmp_path), str(dst_path))
            disk_b = int(dst_path.stat().st_size)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            with self._lock:
                if self._pending.get(key) is pending_ent:
                    self._pending.pop(key, None)
                self.write_errors += 1
            return

        evict_paths: List[Path] = []
        discard_written = False
        with self._lock:
            if self._closed or (self._pending.get(key) is not pending_ent):
                discard_written = True
            else:
                self._pending.pop(key, None)
                prev = self._disk_lru.pop(key, None)
                if prev is not None:
                    self._disk_bytes -= prev[1]
                while self._disk_lru and (self._disk_bytes + disk_b > self.max_bytes):
                    _, (ev_path, ev_db, _ev_eb, _ev_sig) = self._disk_lru.popitem(last=False)
                    self._disk_bytes -= ev_db
                    self.evictions += 1
                    evict_paths.append(ev_path)
                if disk_b <= self.max_bytes:
                    self._disk_lru[key] = (dst_path, disk_b, entry_bytes, sig)
                    self._disk_bytes += disk_b
                    self.spills += 1
                else:
                    discard_written = True

        if discard_written:
            try:
                dst_path.unlink(missing_ok=True)
            except OSError:
                pass
        for ep in evict_paths:
            try:
                ep.unlink(missing_ok=True)
            except OSError:
                pass

    def contains(self, key: Tuple[int, int]) -> bool:
        with self._lock:
            return (not self._closed) and ((key in self._pending) or (key in self._disk_lru))

    def flush(self) -> None:
        with self._lock:
            if self._closed or self._pool is None:
                return
            pool = self._pool
        try:
            pool.submit(lambda: None).result()
        except Exception:
            pass

    def load(
        self, key: Tuple[int, int]
    ) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int]]:
        from safetensors.torch import load_file

        with self._lock:
            if self._closed:
                return None
            pend = self._pending.get(key)
            if pend is not None:
                self.hits += 1
                return pend[0], pend[1], pend[2]
            disk_ent = self._disk_lru.get(key)
            if disk_ent is None:
                return None
            # Read-keep LRU: keep crystal on ext4 so future L2 evictions deduplicate in 0 ms
            self._disk_lru.move_to_end(key)
            path, _disk_b, entry_bytes, _sig = disk_ent

        try:
            flat = load_file(str(path), device="cpu")
        except Exception:
            with self._lock:
                stale = self._disk_lru.pop(key, None)
                if stale is not None:
                    self._disk_bytes -= stale[1]
            return None

        gu_t = {k[3:]: v for k, v in flat.items() if k.startswith("gu.")}
        dn_t = {k[3:]: v for k, v in flat.items() if k.startswith("dn.")}
        if not gu_t or not dn_t:
            return None
        with self._lock:
            self.hits += 1
        return gu_t, dn_t, entry_bytes

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": not self._closed,
                "dir": str(self.session_dir),
                "spills": self.spills,
                "spills_deduped": self.spills_deduped,
                "hits": self.hits,
                "evictions": self.evictions,
                "write_errors": self.write_errors,
                "entries": len(self._disk_lru) + len(self._pending),
                "disk_entries": len(self._disk_lru),
                "pending_entries": len(self._pending),
                "mib_used": round(self._disk_bytes / 1024**2, 2),
                "max_mib": round(self.max_bytes / 1024**2, 2),
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._pending.clear()
            self._disk_lru.clear()
            self._disk_bytes = 0
            pool = self._pool
            self._pool = None
        if pool is not None:
            try:
                pool.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass
        try:
            shutil.rmtree(self.session_dir, ignore_errors=True)
        except Exception:
            pass
        try:
            atexit.unregister(self.close)
        except Exception:
            pass


class MirrorIsolatedMoEOverlayV4:
    """Version 4 (Final — Conveyor-Free) MoE Overlay running strictly isolated mirror mechanisms:
      - M1 `MirrorLoadDispatcher`      (GPU load, VRAM budget, dispatch regime, burst span, async cap)
      - M2 `MirrorRegionBreathingPool` (live GPU region reuse & event-fenced exhale, NO A/B/C ring!)
      - M4 `MirrorPostPrefillRegionWarmupScale` (post-prefill known region warmup, NO Graphify!)
      - M5 Layer-local BF16 prefill + activation-verified Q1 + Kernel #6 anchored decode + line_spine

    ZERO Conveyor (`self.relay = None`), ZERO Graphify (`self.graph = None`), ZERO Bus (`self.probability_bus = None`).
    """

    def __init__(
        self,
        cache: Any,
        *,
        core_k: int = 5,
        mid_k: int = 4,
        rim_k: int = 3,
        shell_k: int = 2,
        min_shed_k: int = 2,
        rest_tol: float = 0.0,
        rest_min: int = 3,
        auto_vram_adapt: bool = False,
        vram_safety_reserve_bytes: Optional[int] = None,
        sensor_fn: Optional[Callable[[], Tuple[int, int, Optional[float], Optional[float], int, str]]] = None,
        l3_cache_dir: Optional[str | Path] = None,
        l3_cache_max_bytes: int = 2 * 1024**3,
        enable_l3_cache: Optional[bool] = None,
    ) -> None:
        self.cache = cache
        self.device = cache.device
        self.feedback_mode: str = "mirror_scales"
        # Explicitly NO Conveyor, NO Graphify, NO Cross-Scale Bus!
        self.relay: Any = None
        self.graphify_feedback: bool = False
        self.graph: Any = None
        self.probability_bus: Any = None

        self.core_k = int(core_k)
        self.mid_k = int(mid_k)
        self.rim_k = max(2, int(rim_k))
        self.shell_k = int(shell_k)
        self.min_shed_k = max(1, min(int(min_shed_k), self.shell_k))
        self.rest_tol = float(rest_tol)
        self.rest_min = max(1, int(rest_min))

        # M1: Isolated GPU Load & Dispatch Dispatcher (Conveyor Replacement #1)
        self.load_dispatcher = MirrorLoadDispatcher(
            device=self.device,
            enabled=bool(auto_vram_adapt or sensor_fn is not None),
            safety_reserve_bytes=vram_safety_reserve_bytes,
            initial_batch_span=16,
            min_batch_span=4,
            max_batch_span=32,
            max_in_flight=2,
            sensor_fn=sensor_fn,
        )
        # M2: Isolated GPU Region Breathing Pool (Conveyor Replacement #2 — NO A/B/C(0) ring!)
        self.region_pool = MirrorRegionBreathingPool(
            device=self.device,
            max_pool_bytes=512 * 1024**2,
            max_warm_regions=4,
        )
        # M4: Isolated Post-Prefill Known-Region Warmup Scale (expanded map for wide linear CPU->GPU injection)
        self.post_prefill_warmup_scale = MirrorPostPrefillRegionWarmupScale(
            num_layers=int(cache.num_layers),
            max_region_experts=24,
            initial_width=1,
            max_width=2,
        )

        # Async IO helper (2 dedicated CUDA transfer streams, 2 concurrent IO workers so eager decode & M4 never block each other!)
        self._io_streams = (
            [torch.cuda.Stream(device=self.device), torch.cuda.Stream(device=self.device)]
            if self.device.type == "cuda"
            else [None, None]
        )
        self._stream_cursor: int = 0
        self._io_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="quintessentia-v4-io")
        self._warm_future = None
        self._warm_target_layer: Optional[int] = None
        self._warm_eids: Tuple[int, ...] = ()

        # Non-blocking background Q1 crystallization stream (runs concurrently with next layer's NVMe IO; max 1 in flight!)
        self._crystal_stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
        self._crystal_pool = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="quintessentia-v4-cryst")
            if self.device.type == "cuda"
            else None
        )
        self._crystal_future = None
        self._bg_crystals_completed: int = 0

        # Non-blocking async line_spine mirror norm channel (zero per-layer GPU .tolist() stalls!)
        self._async_norm_cpu = (
            torch.zeros((3,), dtype=torch.float32, pin_memory=True)
            if (self.device.type == "cuda" and torch.cuda.is_available())
            else None
        )
        self._async_norm_ev = None
        self._async_norm_meta: Optional[Tuple[int, float, float]] = None
        self._layer_norm_trace: Dict[int, Tuple[float, float, float]] = {}

        # Linear CPU->GPU Raw Byte Lookahead Injector (zero CPU compute / zero dequant / zero carve!):
        # CPU linearly streams wide raw quantized GGUF byte corridors `lookahead=3` layers ahead of GPU;
        # GPU selectively dequantizes ONLY router-matched experts and discards the rest at 0 FLOPs.
        self._raw_injector_stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
        self._raw_injector_pool = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="quintessentia-v4-raw-inj")
            if self.device.type == "cuda"
            else None
        )
        self._raw_lookahead: int = 1
        self._raw_width: int = 2
        self._raw_window: Dict[int, Tuple[Dict[int, Tuple[Any, Any, Any]], Any, int]] = {}
        self._raw_inflight: Dict[int, Tuple[Any, Tuple[int, ...]]] = {}
        self.raw_injector_layers_pushed: int = 0
        self.raw_injector_experts_pushed: int = 0
        self.raw_injector_hits: int = 0
        self.raw_injector_misses: int = 0
        self.raw_injector_bytes_pushed: int = 0
        self._raw_recent_checked: int = 0
        self._raw_recent_hits: int = 0
        self._raw_cooldown: int = 0

        # M5 Layer-local MoE execution state ONLY
        self.entry_cone_layers: int = 4 if int(cache.num_layers) >= 8 else 0
        self.mirror_carve_tol: float = 0.05
        self.mirror_diff_tol: float = 0.02
        self._access_tick: int = 0
        self._expert_last_tick: Dict[Tuple[int, int], int] = {}
        self._expert_hits: Dict[Tuple[int, int], int] = defaultdict(int)
        self._decode_seen_cold: set[Tuple[int, int]] = set()
        self._crystal_scars: Dict[Tuple[int, int], Dict[str, Any]] = {}
        # Reversible host bit-plane scars saved during Depth-Shedding (`k <-> k+-1`, 0 disk I/O!)
        self._shed_scars: "OrderedDict[Tuple[int, int], Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]], int]]" = OrderedDict()
        self._shed_scar_bytes: int = 0
        self._max_shed_scar_bytes: int = 128 * 1024**2

        # M5 telemetry
        self.prefill_crystallized: int = 0
        self.total_crystallized: int = 0
        self.prefill_direct_pass: int = 0
        self.prefill_carves_avoided: int = 0
        self.crystal_k_opt_sum: int = 0
        self.crystal_early_stops: int = 0
        self.decode_direct_unclosed: int = 0
        self.decode_direct_oneoff: int = 0
        self.decode_cold_rested_equal: int = 0
        self.decode_cold_woken_split: int = 0
        self.zero_disk_spike_layers: int = 0
        self.decode_layers_total: int = 0
        self.diff_q1_calls: int = 0
        self.diff_q1_anchored_calls: int = 0
        self.line_cuda_launches: int = 0
        self.decode_norm_syncs_performed: int = 0
        self.decode_norm_syncs_avoided: int = 0
        self.decode_d2h_coalesced: int = 0
        self._decode_tok_zero = torch.zeros(
            (1,),
            dtype=(torch.int32 if self.device.type == "cuda" else torch.long),
            device=self.device,
        )
        self._decode_line_metrics = torch.empty((1, 4), dtype=torch.float32, device=self.device)
        self.decode_eager_overlaps: int = 0
        self.depth_sheds: int = 0
        self.shed_scars_saved: int = 0
        self.shed_scars_restored: int = 0
        self.bytes_freed_by_shedding: int = 0
        self.evictions: int = 0
        self.post_prefill_bg_adoptions: int = 0
        self.last_trace: Dict[str, Any] = {}

        # Physical L3 SSD Q1 Crystal Cache (ext4 on POSIX by default; NEVER /tmp tmpfs!)
        self.l3_cache: Optional[Q1DiskL3Cache] = None
        if enable_l3_cache is None:
            use_l3 = (
                (l3_cache_dir is not None)
                or (os.name == "posix" and int(getattr(cache, "host_cache_bytes", 0)) > 0)
            )
        else:
            use_l3 = bool(enable_l3_cache)
        if use_l3 and int(l3_cache_max_bytes) > 0:
            try:
                self.l3_cache = Q1DiskL3Cache(
                    cache_dir=l3_cache_dir,
                    max_bytes=int(l3_cache_max_bytes),
                )
                cache._host_eviction_sink = self.l3_cache.spill_async
            except Exception:
                self.l3_cache = None

    def _in_l3(self, layer_idx: int, eid: int) -> bool:
        return self.l3_cache is not None and self.l3_cache.contains((int(layer_idx), int(eid)))

    # ------------------------------------------------------------------ M1 Physical Load Application
    def _apply_m1_load(self, protected_layer: int, protected_eids: set[int]) -> LoadDispatchDecision:
        """Query M1 on physical hardware and apply ONLY the physical VRAM budget limit."""
        c = self.cache
        dec = self.load_dispatcher.evaluate_load(
            cached_bytes=c._cached_bytes,
            current_budget_bytes=c.max_cache_bytes,
        )
        c.max_cache_bytes = dec.target_budget_bytes
        if dec.budget_action == "protect" and dec.shed_bytes_needed > 0:
            self._make_room_physical(
                needed_bytes=0,
                protected_layer=protected_layer,
                protected_eids=protected_eids,
                emergency=dec.emergency,
            )
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        return dec

    def _make_room_physical(
        self,
        needed_bytes: int,
        protected_layer: int,
        protected_eids: set[int],
        *,
        emergency: bool = False,
    ) -> None:
        c = self.cache
        budget = c.max_cache_bytes
        if c._cached_bytes + needed_bytes <= budget:
            return

        cands = [
            (self._expert_last_tick.get(k, 0), self._expert_hits.get(k, 0), k)
            for k in c._lru.keys()
            if not (k[0] == protected_layer and k[1] in protected_eids)
        ]
        cands.sort(key=lambda item: (item[0], item[1]))
        if not cands:
            return

        def _trim_pass(floor_k: int) -> None:
            for _tk, _ht, key in cands:
                if c._cached_bytes + needed_bytes <= budget:
                    return
                ent = c._lru.get(key)
                if ent is None:
                    continue
                gu_t, dn_t, old_b = ent
                cg = int(gu_t["packed_lo"].shape[0])
                cd = int(dn_t["packed_lo"].shape[0])
                if cg <= floor_k and cd <= floor_k:
                    continue
                ng = max(floor_k, cg - 1) if cg > floor_k else cg
                nd = max(floor_k, cd - 1) if cd > floor_k else cd
                new_gu, scar_gu = _trim_q1_planes_with_scar(gu_t, ng)
                new_dn, scar_dn = _trim_q1_planes_with_scar(dn_t, nd)
                new_b = _tensor_dict_bytes(new_gu) + _tensor_dict_bytes(new_dn)
                freed = max(0, old_b - new_b)
                c._lru[key] = (new_gu, new_dn, new_b)
                c._cached_bytes -= freed
                self.depth_sheds += 1
                self.bytes_freed_by_shedding += freed
                # Save reversible shed scar in host RAM (bounded) for instant zero-disk restore
                if (scar_gu is not None or scar_dn is not None) and self._max_shed_scar_bytes > 0:
                    sb = (_tensor_dict_bytes(scar_gu) if scar_gu else 0) + (_tensor_dict_bytes(scar_dn) if scar_dn else 0)
                    if sb <= self._max_shed_scar_bytes:
                        if key in self._shed_scars:
                            _, _, old_sb = self._shed_scars.pop(key)
                            self._shed_scar_bytes -= old_sb
                        while self._shed_scars and (self._shed_scar_bytes + sb > self._max_shed_scar_bytes):
                            _, (_, _, ev_sb) = self._shed_scars.popitem(last=False)
                            self._shed_scar_bytes -= ev_sb
                        self._shed_scars[key] = (scar_gu, scar_dn, sb)
                        self._shed_scar_bytes += sb
                        self.shed_scars_saved += 1

        if self.rim_k > self.min_shed_k:
            _trim_pass(self.rim_k)
        if c._cached_bytes + needed_bytes > budget:
            _trim_pass(self.min_shed_k)

        for _tk, _ht, key in cands:
            if c._cached_bytes + needed_bytes <= budget:
                return
            ent = c._lru.pop(key, None)
            if ent is None:
                continue
            gu_t, dn_t, old_b = ent
            c._cached_bytes -= old_b
            c._mirror_anchors.pop(key, None)
            self.evictions += 1
            c.evictions += 1
            if c.host_cache_bytes > 0:
                c._put_host(key, gu_t, dn_t, old_b)

        if emergency and c._cached_bytes + needed_bytes > budget:
            _trim_pass(1)

    def _store_crystal(
        self,
        key: Tuple[int, int],
        gu_t: Dict[str, torch.Tensor],
        dn_t: Dict[str, torch.Tensor],
        entry_b: int,
        protected_layer: int,
        protected_eids: set[int],
    ) -> bool:
        c = self.cache
        if key in c._lru:
            _, _, old_b = c._lru.pop(key)
            c._cached_bytes -= old_b
        if c.max_cache_bytes > 0 and entry_b <= c.max_cache_bytes:
            self._make_room_physical(entry_b, protected_layer, protected_eids)
            if c._cached_bytes + entry_b <= c.max_cache_bytes:
                if c.cache_on_gpu:
                    s_gu, s_dn = gu_t, dn_t
                else:
                    s_gu = {k: v.detach().cpu() for k, v in gu_t.items()}
                    s_dn = {k: v.detach().cpu() for k, v in dn_t.items()}
                c._lru[key] = (s_gu, s_dn, entry_b)
                c._cached_bytes += entry_b
                self._access_tick += 1
                self._expert_last_tick[key] = self._access_tick
                return True
        if c.host_cache_bytes > 0:
            c._put_host(key, gu_t, dn_t, entry_b)
        return False

    def _get_hot_or_host(
        self,
        layer_idx: int,
        eid: int,
        protected_eids: set[int],
    ) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        c = self.cache
        key = (int(layer_idx), int(eid))
        hit = c._lru.get(key)
        if hit is not None:
            c.hits += 1
            self._access_tick += 1
            self._expert_last_tick[key] = self._access_tick
            self._expert_hits[key] += 1
            gu_t, dn_t, _ = hit
            if c.cache_on_gpu or self.device.type != "cuda":
                return gu_t, dn_t
            return (
                {k: v.to(self.device, non_blocking=True) for k, v in gu_t.items()},
                {k: v.to(self.device, non_blocking=True) for k, v in dn_t.items()},
            )

        host_hit = c._host_lru.pop(key, None)
        if host_hit is not None:
            c.host_hits += 1
            h_gu, h_dn, eb = host_hit
            c._host_cached_bytes -= eb
            gu_t = {k: v.to(self.device, non_blocking=True) for k, v in h_gu.items()}
            dn_t = {k: v.to(self.device, non_blocking=True) for k, v in h_dn.items()}
            self._store_crystal(key, gu_t, dn_t, eb, int(layer_idx), protected_eids)
            return gu_t, dn_t

        if self.l3_cache is not None:
            l3_hit = self.l3_cache.load(key)
            if l3_hit is not None:
                c.host_hits += 1
                l3_gu, l3_dn, eb = l3_hit
                gu_t = {k: v.to(self.device, non_blocking=True) for k, v in l3_gu.items()}
                dn_t = {k: v.to(self.device, non_blocking=True) for k, v in l3_dn.items()}
                self._store_crystal(key, gu_t, dn_t, eb, int(layer_idx), protected_eids)
                self._access_tick += 1
                self._expert_last_tick[key] = self._access_tick
                self._expert_hits[key] += 1
                return gu_t, dn_t

        return None

    # ------------------------------------------------------------------ Conveyor-Free Async Fetch -> M2 Pool
    def _next_io_stream(self):
        st = self._io_streams[self._stream_cursor % len(self._io_streams)]
        self._stream_cursor += 1
        return st

    def _fetch_raw_region_async(
        self,
        layer_idx: int,
        eids: List[int],
        io_stream: Any,
    ) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], int, Any, float]:
        """Background IO task: stream raw BF16 experts directly onto GPU stream (NO conveyor slot!)."""
        t0 = time.perf_counter()
        c = self.cache
        if self.device.type == "cuda" and io_stream is not None:
            with torch.cuda.device(self.device), torch.cuda.stream(io_stream):
                if c.expert_batch_reader is not None and len(eids) > 1:
                    raw_in = c.expert_batch_reader(int(layer_idx), list(eids))
                else:
                    raw_in = {eid: c.expert_reader(int(layer_idx), int(eid)) for eid in eids}
                raw_map: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
                tot_bytes = 0
                for eid in eids:
                    gu_r, dn_r = raw_in.pop(eid)
                    if gu_r.device != self.device or gu_r.dtype != torch.bfloat16:
                        gu_r = gu_r.to(device=self.device, dtype=torch.bfloat16, non_blocking=True)
                    if dn_r.device != self.device or dn_r.dtype != torch.bfloat16:
                        dn_r = dn_r.to(device=self.device, dtype=torch.bfloat16, non_blocking=True)
                    raw_map[eid] = (gu_r, dn_r)
                    tot_bytes += (
                        int(gu_r.numel()) * int(gu_r.element_size())
                        + int(dn_r.numel()) * int(dn_r.element_size())
                    )
                ready = torch.cuda.Event(enable_timing=False)
                ready.record(io_stream)
            return raw_map, min(tot_bytes, self.region_pool.max_pool_bytes), ready, (time.perf_counter() - t0) * 1000.0

        if c.expert_batch_reader is not None and len(eids) > 1:
            raw_in = c.expert_batch_reader(int(layer_idx), list(eids))
        else:
            raw_in = {eid: c.expert_reader(int(layer_idx), int(eid)) for eid in eids}
        raw_map = {}
        for eid in eids:
            gu_r, dn_r = raw_in.pop(eid)
            raw_map[eid] = (
                gu_r.to(device=self.device, dtype=torch.bfloat16),
                dn_r.to(device=self.device, dtype=torch.bfloat16),
            )
        return raw_map, 0, None, (time.perf_counter() - t0) * 1000.0

    def _fetch_post_prefill_warm_async(
        self,
        target_layer: int,
        eids: List[int],
        io_stream: Any,
    ) -> Tuple[int, Dict[int, Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int]], int, Any]:
        """Background IO task for M4 post-prefill region warmup (NO conveyor slot!)."""
        from src.binary_halving_q1 import carve_q1_cubes_incremental

        c = self.cache
        depth = self.rim_k
        if self.device.type == "cuda" and io_stream is not None:
            with torch.cuda.device(self.device), torch.cuda.stream(io_stream):
                if c.expert_batch_reader is not None and len(eids) > 1:
                    raw_in = c.expert_batch_reader(int(target_layer), list(eids))
                else:
                    raw_in = {eid: c.expert_reader(int(target_layer), int(eid)) for eid in eids}
                carved: Dict[int, Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int]] = {}
                for eid in eids:
                    gu_r, dn_r = raw_in.pop(eid)
                    gu_g = gu_r.to(device=self.device, dtype=torch.bfloat16, non_blocking=True)
                    dn_g = dn_r.to(device=self.device, dtype=torch.bfloat16, non_blocking=True)
                    gu_t, _ = carve_q1_cubes_incremental(
                        gu_g, depth, root_bits=c.root_bits, group_size=c.group_size,
                        use_fwht=c.use_fwht, sinkhorn_iters=c.sinkhorn_iters,
                    )
                    dn_t, _ = carve_q1_cubes_incremental(
                        dn_g, depth, root_bits=c.root_bits, group_size=c.group_size,
                        use_fwht=c.use_fwht, sinkhorn_iters=c.sinkhorn_iters,
                    )
                    del gu_g, dn_g
                    eb = _tensor_dict_bytes(gu_t) + _tensor_dict_bytes(dn_t)
                    carved[eid] = (gu_t, dn_t, eb)
                ready = torch.cuda.Event(enable_timing=False)
                ready.record(io_stream)
            return int(target_layer), carved, depth, ready

        if c.expert_batch_reader is not None and len(eids) > 1:
            raw_in = c.expert_batch_reader(int(target_layer), list(eids))
        else:
            raw_in = {eid: c.expert_reader(int(target_layer), int(eid)) for eid in eids}
        carved = {}
        for eid in eids:
            gu_r, dn_r = raw_in.pop(eid)
            gu_t, _ = carve_q1_cubes_incremental(
                gu_r.to(device=self.device, dtype=torch.bfloat16), depth,
                root_bits=c.root_bits, group_size=c.group_size,
                use_fwht=c.use_fwht, sinkhorn_iters=c.sinkhorn_iters,
            )
            dn_t, _ = carve_q1_cubes_incremental(
                dn_r.to(device=self.device, dtype=torch.bfloat16), depth,
                root_bits=c.root_bits, group_size=c.group_size,
                use_fwht=c.use_fwht, sinkhorn_iters=c.sinkhorn_iters,
            )
            eb = _tensor_dict_bytes(gu_t) + _tensor_dict_bytes(dn_t)
            carved[eid] = (gu_t, dn_t, eb)
        return int(target_layer), carved, depth, None

    def _adopt_post_prefill_warm(
        self,
        current_layer: int,
        protected_eids: set[int],
        *,
        force: bool = False,
    ) -> Tuple[int, Tuple[int, ...]]:
        """Adopt finished background M4 post-prefill region directly via M2 breathing pool (NO rotate!)."""
        if self._warm_future is None:
            return -1, ()
        if (not force) and (self._warm_target_layer is not None) and (self._warm_target_layer != current_layer):
            if not self._warm_future.done():
                return -1, ()

        warm_layer = int(self._warm_target_layer) if self._warm_target_layer is not None else -1
        warm_eids = self._warm_eids
        try:
            t_li, carved_map, depth, ready_ev = self._warm_future.result()
            reg_key = ("warm", t_li, warm_eids)
            tot_b = sum(eb for _, _, eb in carved_map.values())
            admitted = self.region_pool.admit_region(
                reg_key,
                carved_map,
                gpu_bytes=min(tot_b, self.region_pool.max_pool_bytes),
                ready_event=ready_ev,
                recurring_hint=False,
            )
            c = self.cache
            for eid, (gu_t, dn_t, eb) in admitted.items():
                key = (int(t_li), int(eid))
                if key in c._lru:
                    continue
                if (
                    c._cached_bytes + eb <= c.max_cache_bytes
                    and self.load_dispatcher.can_admit_bytes(eb, margin_bytes=256 * 1024**2)
                ):
                    c.unique_experts_carved.add(key)
                    c.total_cubes_carved += 2 * int(depth)
                    self._store_crystal(key, gu_t, dn_t, eb, int(current_layer), protected_eids)
                    self.post_prefill_bg_adoptions += 1
                elif c.host_cache_bytes > 0:
                    c._put_host(key, gu_t, dn_t, eb)
            self.region_pool.finish_region(reg_key, keep_if_reusable=False)
        finally:
            self._warm_future = None
            self._warm_target_layer = None
            self._warm_eids = ()
        return warm_layer, warm_eids

    # ------------------------------------------------------------------ M5 Activation-Aware Crystal Carve
    @torch.no_grad()
    def _crystallize_with_probe(
        self,
        layer_idx: int,
        eid: int,
        gu_raw: torch.Tensor,
        dn_raw: torch.Tensor,
        x_probe: torch.Tensor,
        gu_exact: torch.Tensor,
        act_probe: torch.Tensor,
        dn_exact: torch.Tensor,
        max_k: int,
        protected_eids: set[int],
    ) -> None:
        from src.binary_halving_q1 import carve_q1_mirror_crystal

        c = self.cache
        key = (int(layer_idx), int(eid))
        min_k = min(self.rim_k, max_k)
        try:
            gu_t, _, gu_scar = carve_q1_mirror_crystal(
                gu_raw, x_probe, gu_exact, max_k,
                min_cubes=min_k, mirror_tol=self.mirror_carve_tol,
                root_bits=c.root_bits, group_size=c.group_size,
                use_fwht=c.use_fwht, sinkhorn_iters=c.sinkhorn_iters,
            )
            dn_t, _, dn_scar = carve_q1_mirror_crystal(
                dn_raw, act_probe, dn_exact, max_k,
                min_cubes=min_k, mirror_tol=self.mirror_carve_tol,
                root_bits=c.root_bits, group_size=c.group_size,
                use_fwht=c.use_fwht, sinkhorn_iters=c.sinkhorn_iters,
            )
            entry_b = _tensor_dict_bytes(gu_t) + _tensor_dict_bytes(dn_t)
        except torch.cuda.OutOfMemoryError:
            self.load_dispatcher.record_oom_recovery()
            c.max_cache_bytes = max(0, c._cached_bytes // 2)
            self._make_room_physical(0, -1, set(), emergency=True)
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            safe_k = max(1, min(self.min_shed_k, max_k))
            gu_t, _, gu_scar = carve_q1_mirror_crystal(
                gu_raw, x_probe, gu_exact, safe_k,
                min_cubes=1, mirror_tol=self.mirror_carve_tol,
                root_bits=c.root_bits, group_size=c.group_size,
                use_fwht=c.use_fwht, sinkhorn_iters=c.sinkhorn_iters,
            )
            dn_t, _, dn_scar = carve_q1_mirror_crystal(
                dn_raw, act_probe, dn_exact, safe_k,
                min_cubes=1, mirror_tol=self.mirror_carve_tol,
                root_bits=c.root_bits, group_size=c.group_size,
                use_fwht=c.use_fwht, sinkhorn_iters=c.sinkhorn_iters,
            )
            entry_b = _tensor_dict_bytes(gu_t) + _tensor_dict_bytes(dn_t)

        kg = int(gu_scar["k_opt"])
        kd = int(dn_scar["k_opt"])
        self.crystal_k_opt_sum += kg + kd
        self.total_crystallized += 1
        if kg < max_k:
            self.crystal_early_stops += 1
        if kd < max_k:
            self.crystal_early_stops += 1

        c.unique_experts_carved.add(key)
        c.total_cubes_carved += kg + kd
        self._crystal_scars[key] = {"k_opt": max(kg, kd), "gu_scar": gu_scar, "dn_scar": dn_scar}
        stored = self._store_crystal(key, gu_t, dn_t, entry_b, int(layer_idx), protected_eids)
        if stored and key in c._lru:
            c._mirror_anchors[key] = (
                x_probe[-1].detach().to(device=self.device, dtype=torch.bfloat16).contiguous(),
                gu_exact[-1].detach().to(device=self.device, dtype=torch.bfloat16).contiguous(),
                act_probe[-1].detach().to(device=self.device, dtype=torch.bfloat16).contiguous(),
                dn_exact[-1].detach().to(device=self.device, dtype=torch.bfloat16).contiguous(),
            )

    def _bg_crystal_worker(
        self,
        layer_idx: int,
        eid: int,
        gu_raw: torch.Tensor,
        dn_raw: torch.Tensor,
        x_p: torch.Tensor,
        gu_ex: torch.Tensor,
        act_p: torch.Tensor,
        dn_ex: torch.Tensor,
        max_k: int,
        start_ev: Any,
    ) -> Optional[Tuple[Tuple[int, int], Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, Dict[str, Any], Dict[str, Any], Tuple[torch.Tensor, ...], Any]]:
        """Run Q1 crystal carving on dedicated `_crystal_stream` concurrently with main decode."""
        from src.binary_halving_q1 import carve_q1_mirror_crystal

        c = self.cache
        key = (int(layer_idx), int(eid))
        min_k = min(self.rim_k, max_k)
        cs = self._crystal_stream
        if cs is None:
            return None
        try:
            with torch.no_grad(), torch.cuda.device(self.device), torch.cuda.stream(cs):
                if start_ev is not None:
                    cs.wait_event(start_ev)
                gu_t, _, gu_scar = carve_q1_mirror_crystal(
                    gu_raw, x_p, gu_ex, max_k,
                    min_cubes=min_k, mirror_tol=self.mirror_carve_tol,
                    root_bits=c.root_bits, group_size=c.group_size,
                    use_fwht=c.use_fwht, sinkhorn_iters=c.sinkhorn_iters,
                )
                dn_t, _, dn_scar = carve_q1_mirror_crystal(
                    dn_raw, act_p, dn_ex, max_k,
                    min_cubes=min_k, mirror_tol=self.mirror_carve_tol,
                    root_bits=c.root_bits, group_size=c.group_size,
                    use_fwht=c.use_fwht, sinkhorn_iters=c.sinkhorn_iters,
                )
                entry_b = _tensor_dict_bytes(gu_t) + _tensor_dict_bytes(dn_t)
                anc = (
                    x_p.detach().to(device=self.device, dtype=torch.bfloat16).contiguous(),
                    gu_ex.detach().to(device=self.device, dtype=torch.bfloat16).contiguous(),
                    act_p.detach().to(device=self.device, dtype=torch.bfloat16).contiguous(),
                    dn_ex.detach().to(device=self.device, dtype=torch.bfloat16).contiguous(),
                )
                ready_ev = torch.cuda.Event(enable_timing=False)
                ready_ev.record(cs)
                return (key, gu_t, dn_t, entry_b, gu_scar, dn_scar, anc, ready_ev)
        except Exception:
            return None

    def _drain_bg_crystal(
        self,
        protected_layer: int,
        protected_eids: set[int],
        *,
        force: bool = False,
    ) -> None:
        """Adopt finished background Q1 crystal on the main thread (ZERO lock contention / ZERO sync if not ready!)."""
        fut = self._crystal_future
        if fut is None:
            return
        if not force and not fut.done():
            return
        self._crystal_future = None
        try:
            res = fut.result()
        except Exception:
            return
        if res is None:
            return
        key, gu_t, dn_t, entry_b, gu_scar, dn_scar, anc, ready_ev = res
        if ready_ev is not None and self.device.type == "cuda":
            torch.cuda.current_stream(self.device).wait_event(ready_ev)
        c = self.cache
        if key in c._lru:
            return
        kg = int(gu_scar["k_opt"])
        kd = int(dn_scar["k_opt"])
        max_k = int(gu_scar.get("max_k", max(kg, kd)))
        self.crystal_k_opt_sum += kg + kd
        self.total_crystallized += 1
        self._bg_crystals_completed += 1
        if kg < max_k:
            self.crystal_early_stops += 1
        if kd < max_k:
            self.crystal_early_stops += 1
        c.unique_experts_carved.add(key)
        c.total_cubes_carved += kg + kd
        self._crystal_scars[key] = {"k_opt": max(kg, kd), "gu_scar": gu_scar, "dn_scar": dn_scar}
        stored = self._store_crystal(key, gu_t, dn_t, entry_b, int(protected_layer), protected_eids)
        if stored and key in c._lru:
            c._mirror_anchors[key] = anc

    def _submit_bg_crystal(
        self,
        layer_idx: int,
        eid: int,
        gu_raw: torch.Tensor,
        dn_raw: torch.Tensor,
        x_probe: torch.Tensor,
        gu_exact: torch.Tensor,
        act_probe: torch.Tensor,
        dn_exact: torch.Tensor,
        max_k: int,
        protected_eids: set[int],
    ) -> bool:
        """Non-blocking background crystallization on CUDA; falls back to sync on CPU."""
        if self._crystal_pool is None or self._crystal_stream is None:
            self._crystallize_with_probe(
                layer_idx, eid, gu_raw, dn_raw, x_probe, gu_exact, act_probe, dn_exact, max_k, protected_eids
            )
            return True
        self._drain_bg_crystal(layer_idx, protected_eids, force=False)
        if self._crystal_future is not None:
            return False
        start_ev = torch.cuda.Event(enable_timing=False)
        start_ev.record(torch.cuda.current_stream(self.device))
        # Mark raw tensors as consumed on _crystal_stream so caching allocator keeps them intact until carve ends
        gu_raw.record_stream(self._crystal_stream)
        dn_raw.record_stream(self._crystal_stream)
        xp = x_probe[-1].detach().clone()
        ge = gu_exact[-1].detach().clone()
        ap = act_probe[-1].detach().clone()
        de = dn_exact[-1].detach().clone()
        xp.record_stream(self._crystal_stream)
        ge.record_stream(self._crystal_stream)
        ap.record_stream(self._crystal_stream)
        de.record_stream(self._crystal_stream)
        self._crystal_future = self._crystal_pool.submit(
            self._bg_crystal_worker,
            int(layer_idx), int(eid), gu_raw, dn_raw, xp, ge, ap, de, int(max_k), start_ev,
        )
        return True

    # ------------------------------------------------------------------ Linear CPU->GPU Raw Byte Injector
    def _has_raw_injector(self) -> bool:
        return (
            self._raw_injector_pool is not None
            and self._raw_injector_stream is not None
            and getattr(self.cache, "raw_u8_fetcher", None) is not None
            and getattr(self.cache, "raw_u8_dequantizer", None) is not None
        )

    def _fetch_raw_u8_async(
        self,
        layer_idx: int,
        eids: List[int],
    ) -> Tuple[Dict[int, Tuple[Any, Any, Any]], Any, int]:
        """Worker: streams raw quantized GGUF bytes directly to GPU (0 CPU compute / 0 dequant / 0 Q1 carve!)."""
        fetcher = self.cache.raw_u8_fetcher
        st = self._raw_injector_stream
        with torch.no_grad(), torch.cuda.device(self.device), torch.cuda.stream(st):
            raw_map, tot_b = fetcher(int(layer_idx), list(eids))
            ev = torch.cuda.Event(enable_timing=False)
            ev.record(st)
        return raw_map, ev, int(tot_b)

    def _harvest_raw_inflight(self, *, wait_layer: Optional[int] = None) -> None:
        for tl in list(self._raw_inflight.keys()):
            fut, _eids = self._raw_inflight[tl]
            if tl == wait_layer or fut.done():
                self._raw_inflight.pop(tl, None)
                try:
                    raw_map, ev, tot_b = fut.result()
                    if raw_map:
                        self._raw_window[tl] = (raw_map, ev, tot_b)
                except Exception:
                    pass

    def _pump_raw_injector(
        self,
        current_layer: int,
        is_resident_fn: Callable[[int, int], bool],
    ) -> None:
        """Linear CPU stream running `lookahead` layers ahead of GPU — feeds wide raw byte corridors into VRAM."""
        if not self._has_raw_injector():
            return
        num_l = int(self.cache.num_layers)
        keep_layers = {(int(current_layer) + off) % num_l for off in range(self._raw_lookahead + 1)}
        # Discard untouched stale raw corridors behind GPU at 0 FLOPs
        for stale_l in list(self._raw_window.keys()):
            if stale_l not in keep_layers:
                self._raw_window.pop(stale_l, None)

        self._harvest_raw_inflight(wait_layer=None)

        # Mirror self-regulation: pause speculative SSD reads when cold-tail hit-rate falls below 25%
        # (prevents NVMe queue saturation on scattered 256-expert tails; re-probes periodically).
        if self._raw_cooldown > 0:
            self._raw_cooldown -= 1
            return
        if self._raw_recent_checked >= 16:
            hr = self._raw_recent_hits / max(1, self._raw_recent_checked)
            self._raw_recent_checked = 0
            self._raw_recent_hits = 0
            if hr < 0.25:
                self._raw_cooldown = num_l  # rest 1 full token before re-probing
                return

        for off in range(1, self._raw_lookahead + 1):
            if len(self._raw_window) + len(self._raw_inflight) >= self._raw_lookahead:
                break
            tl = (int(current_layer) + off) % num_l
            if tl in self._raw_window or tl in self._raw_inflight:
                continue
            if not self.load_dispatcher.can_admit_bytes(12 * 1024**2, margin_bytes=128 * 1024**2):
                break
            wide_eids = self.post_prefill_warmup_scale.plan_wide_layer_candidates(
                tl, is_resident_fn, width=self._raw_width
            )
            if not wide_eids:
                continue
            fut = self._raw_injector_pool.submit(self._fetch_raw_u8_async, tl, wide_eids)
            self._raw_inflight[tl] = (fut, tuple(wide_eids))
            self.raw_injector_layers_pushed += 1
            self.raw_injector_experts_pushed += len(wide_eids)

    def _consume_raw_window_for_layer(
        self,
        layer_idx: int,
        want_eids: List[int],
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """GPU selectively dequantizes ONLY router-matched experts from pre-injected raw VRAM bytes."""
        if not want_eids or not self._has_raw_injector():
            return {}
        li = int(layer_idx)
        self._harvest_raw_inflight(wait_layer=li)
        win = self._raw_window.pop(li, None)
        if win is None:
            self.raw_injector_misses += len(want_eids)
            return {}
        raw_map, ready_ev, tot_b = win
        self.raw_injector_bytes_pushed += int(tot_b)
        hits = [int(e) for e in want_eids if int(e) in raw_map]
        misses = len(want_eids) - len(hits)
        self.raw_injector_hits += len(hits)
        self.raw_injector_misses += misses
        self._raw_recent_checked += len(raw_map)
        self._raw_recent_hits += len(hits)
        if not hits:
            return {}
        cur_st = torch.cuda.current_stream(self.device)
        if ready_ev is not None:
            cur_st.wait_event(ready_ev)
        # Keep raw tensors alive on compute stream while GPU selectively dequantizes matched experts
        for e in hits:
            g_r, u_r, d_r = raw_map[e]
            g_r.record_stream(cur_st)
            u_r.record_stream(cur_st)
            d_r.record_stream(cur_st)
        deq = self.cache.raw_u8_dequantizer
        return deq(li, raw_map, hits, torch.bfloat16)

    def _clear_raw_injector(self) -> None:
        for tl in list(self._raw_inflight.keys()):
            fut, _eids = self._raw_inflight.pop(tl)
            try:
                fut.result()
            except Exception:
                pass
        self._raw_window.clear()

    @torch.no_grad()
    def _run_proj_diff_anchored(
        self,
        key: Tuple[int, int],
        x_e: torch.Tensor,
        t: Dict[str, torch.Tensor],
        p_use: int,
        x_anc: Optional[torch.Tensor],
        y_anc: Optional[torch.Tensor],
    ) -> torch.Tensor:
        c = self.cache
        if int(x_e.shape[0]) == 1 and c.use_fwht and int(c.group_size) == 64:
            stored = int(t["packed_lo"].shape[0])
            p_eff = max(1, min(int(p_use), stored))
            y_out = launch_pure_mirror_diff_q1_gemv(
                x_e,
                t["packed_lo"],
                t["qscale"],
                t["qzero"],
                t["row_scale"],
                t["col_scale"],
                x_last=x_anc,
                y_last=y_anc,
                group_size=int(c.group_size),
                max_planes=p_eff,
                min_planes=1,
                mirror_tol=self.mirror_diff_tol,
            )
            self.diff_q1_calls += 1
            if x_anc is not None and y_anc is not None:
                self.diff_q1_anchored_calls += 1
            return y_out if y_out.dtype == x_e.dtype else y_out.to(dtype=x_e.dtype)
        return c._run_proj(x_e, t, p_use)

    # ------------------------------------------------------------------ Forward Entry Point
    @torch.no_grad()
    def forward_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        li = int(layer_idx)
        m_tok = int(hidden_states.shape[0])
        if m_tok == 0:
            return torch.zeros_like(hidden_states)

        # Coalesce top_k_index + top_k_weights into a SINGLE D2H transfer per layer (halves router syncs)
        if top_k_index.device.type == "cuda" and top_k_weights.device == top_k_index.device:
            k_width = int(top_k_index.shape[-1])
            packed_cpu = torch.cat(
                [top_k_index.detach().to(torch.float32), top_k_weights.detach().to(torch.float32)],
                dim=-1,
            ).cpu().tolist()
            idx_cpu = [[int(v) for v in row[:k_width]] for row in packed_cpu]
            w_cpu = [row[k_width:] for row in packed_cpu]
            self.decode_d2h_coalesced += 1
        else:
            idx_cpu = top_k_index.detach().cpu().tolist()
            w_cpu = top_k_weights.detach().float().cpu().tolist()
        num_exp = self.cache.num_experts

        per_tok_full: List[List[Tuple[int, int, float]]] = []
        all_eids_in_layer: set[int] = set()
        for row_e, row_w in zip(idx_cpu, w_cpu):
            valid: List[Tuple[int, int, float]] = []
            for k_pos, (eid, rw) in enumerate(zip(row_e, row_w)):
                e_int = int(eid)
                if 0 <= e_int < num_exp:
                    valid.append((e_int, k_pos, float(rw)))
                    all_eids_in_layer.add(e_int)
            valid.sort(key=lambda item: -item[2])
            self.cache.routed_slots_total += len(valid)
            per_tok_full.append(valid)

        if m_tok > 1:
            return self._forward_prefill_v4(
                li, hidden_states, top_k_weights, act_fn, per_tok_full, all_eids_in_layer
            )
        return self._forward_decode_v4(
            li, hidden_states, top_k_weights, act_fn, per_tok_full[0], all_eids_in_layer
        )

    forward_layer_moe = forward_layer

    @torch.no_grad()
    def _forward_prefill_v4(
        self,
        li: int,
        hidden_states: torch.Tensor,
        top_k_weights: torch.Tensor,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        per_tok_full: List[List[Tuple[int, int, float]]],
        all_eids_in_layer: set[int],
    ) -> torch.Tensor:
        """Conveyor-Free Prefill: M1 LoadDispatcher (async window 1..2) + M2 RegionBreathingPool + M4 map."""
        c = self.cache
        m_tok, h_dim = int(hidden_states.shape[0]), int(hidden_states.shape[1])
        dev = hidden_states.device

        if self._warm_future is not None:
            self._adopt_post_prefill_warm(li, all_eids_in_layer, force=True)
        if self._crystal_future is not None:
            self._drain_bg_crystal(li, all_eids_in_layer, force=True)
        if self._raw_window or self._raw_inflight:
            self._clear_raw_injector()

        load_dec = self._apply_m1_load(li, all_eids_in_layer)

        by_exp: Dict[int, Tuple[List[int], List[int], List[float]]] = {}
        for tok_i, valid in enumerate(per_tok_full):
            for e_int, k_pos, rw in valid:
                b = by_exp.get(e_int)
                if b is None:
                    b = ([], [], [])
                    by_exp[e_int] = b
                b[0].append(tok_i)
                b[1].append(k_pos)
                b[2].append(rw)
        if not by_exp:
            return torch.zeros_like(hidden_states)

        sorted_eids = sorted(by_exp)
        hot_eids: List[int] = []
        cold_eids: List[int] = []
        for eid in sorted_eids:
            if ((li, eid) in c._lru) or ((li, eid) in c._host_lru) or self._in_l3(li, eid):
                hot_eids.append(eid)
            else:
                cold_eids.append(eid)

        # Capture pure layer-local Left (body) x Right (tail) weights for M4 & local crystallization
        tail_n = max(1, min(4, (m_tok + 3) // 4))
        tail_start = max(0, m_tok - tail_n)
        body_weights: Dict[int, float] = {eid: sum(by_exp[eid][2]) for eid in sorted_eids}
        tail_weights: Dict[int, float] = defaultdict(float)
        tail_core_eids: set[int] = set()
        for tok_i in range(tail_start, m_tok):
            valid_t = per_tok_full[tok_i]
            eff_min = min(self.rest_min, max(1, len(valid_t) - 1)) if len(valid_t) > 1 else len(valid_t)
            for rank, (e_int, _kp, rw) in enumerate(valid_t):
                tail_weights[e_int] += rw
                if rank < eff_min:
                    tail_core_eids.add(e_int)

        self.post_prefill_warmup_scale.record_prefill_layer(li, body_weights, dict(tail_weights))

        crystallize_set: set[int] = set()
        if cold_eids:
            total_slots = sum(len(by_exp[eid][0]) for eid in sorted_eids)
            mean_slot_w = sum(body_weights.values()) / max(1, total_slots)
            core_hit_bar = max(3, m_tok // 6)
            scored_cryst: List[Tuple[float, int]] = []
            for eid in cold_eids:
                hit_cnt = len(by_exp[eid][0])
                ws = body_weights[eid]
                mean_e_w = ws / max(1, hit_cnt)
                in_tail = eid in tail_core_eids
                if in_tail or (hit_cnt >= core_hit_bar and mean_e_w >= 0.5 * mean_slot_w):
                    prio = (2.0 if in_tail else 0.0) + ws
                    scored_cryst.append((prio, eid))
            if scored_cryst:
                scored_cryst.sort(key=lambda kv: -kv[0])
                max_layer_cryst = max(16, min(48, len(scored_cryst)))
                crystallize_set = {eid for _, eid in scored_cryst[:max_layer_cryst]}

        max_body = max(body_weights.values(), default=1.0) or 1.0
        max_tail = max(tail_weights.values(), default=1.0) or 1.0
        cold_depths: Dict[int, int] = {}
        for eid in crystallize_set:
            bp = c.expert_cubes.get((li, eid))
            if bp is not None:
                cold_depths[eid] = max(self.rim_k, min(c.root_bits, max(int(bp[0]), int(bp[1]))))
            else:
                # Pure-math M5b depth from body vs tail probability density (ZERO 0.5/0.25 step-if thresholds!)
                p_b = body_weights.get(eid, 0.0) / max_body
                p_t = tail_weights.get(eid, 0.0) / max_tail
                cold_depths[eid] = mirror_cube_depth_math(p_b, p_t, self.rim_k, self.core_k)

        line_spine = torch.zeros((m_tok, h_dim), dtype=torch.float32, device=dev)

        # M1 controls burst width and async in-flight window depth (1..2, NO conveyor ring!)
        span = load_dec.batch_span
        in_flight_cap = load_dec.in_flight_cap
        cold_bursts: List[List[int]] = [
            cold_eids[i : i + span] for i in range(0, len(cold_eids), span)
        ]

        # Seed async dispatch queue up to M1's in_flight_cap (0..2 live futures)
        pending: deque[Tuple[List[int], Any]] = deque()
        next_b_idx = 0
        while next_b_idx < len(cold_bursts) and len(pending) < in_flight_cap:
            b_eids = cold_bursts[next_b_idx]
            fut = self._io_pool.submit(self._fetch_raw_region_async, li, b_eids, self._next_io_stream())
            pending.append((b_eids, fut))
            next_b_idx += 1

        # Compute already-resident hot experts while M1's async IO bursts stream in parallel
        t_hot0 = time.perf_counter()
        for eid in hot_eids:
            pair = self._get_hot_or_host(li, eid, all_eids_in_layer)
            if pair is None:
                continue
            gu_t, dn_t = pair
            toks, kpos, rws = by_exp[eid]
            tok_t = torch.tensor(toks, dtype=torch.long, device=dev)
            kpos_t = torch.tensor(kpos, dtype=torch.long, device=dev)
            mean_rw = sum(rws) / max(1, len(rws))
            p_use = c.cubes_for_weight(mean_rw)
            x_e = hidden_states.index_select(0, tok_t)
            gu = c._run_proj(x_e, gu_t, p_use)
            g_part, u_part = gu.chunk(2, dim=-1)
            act = act_fn(g_part) * u_part
            y_e = c._run_proj(act, dn_t, p_use)
            _, used_cuda = launch_pure_mirror_line_step(line_spine, y_e, tok_t, top_k_weights[tok_t, kpos_t])
            if used_cuda:
                self.line_cuda_launches += 1
            c.routed_slots_executed += len(toks)
            c.expert_calls += 1
            c.expert_cubes_executed += (
                min(p_use, gu_t["packed_lo"].shape[0]) + min(p_use, dn_t["packed_lo"].shape[0])
            ) / 2
        prev_gpu_ms = (time.perf_counter() - t_hot0) * 1000.0

        # Execute cold bursts directly through M2 MirrorRegionBreathingPool (NO rotate(), NO C(0)!)
        burst_i = 0
        while pending:
            load_dec = self._apply_m1_load(li, all_eids_in_layer)
            b_eids, fut = pending.popleft()
            reg_key = ("prefill", li, tuple(b_eids))

            tw0 = time.perf_counter()
            raw_map_in, gpu_b, ready_ev, io_ms = fut.result()
            wait_ms = (time.perf_counter() - tw0) * 1000.0

            # Keep M1's async window filled (up to live `load_dec.in_flight_cap`)
            while next_b_idx < len(cold_bursts) and len(pending) < load_dec.in_flight_cap:
                nb = cold_bursts[next_b_idx]
                nfut = self._io_pool.submit(self._fetch_raw_region_async, li, nb, self._next_io_stream())
                pending.append((nb, nfut))
                next_b_idx += 1

            # M2 admits region directly onto compute stream (event-fenced, zero conveyor!)
            raw_map = self.region_pool.admit_region(
                reg_key, raw_map_in, gpu_bytes=gpu_b, ready_event=ready_ev, recurring_hint=False
            )

            t_g0 = time.perf_counter()
            for eid in b_eids:
                gu_raw, dn_raw = raw_map.pop(eid)
                c.misses += 1
                toks, kpos, _rws = by_exp[eid]
                tok_t = torch.tensor(toks, dtype=torch.long, device=dev)
                kpos_t = torch.tensor(kpos, dtype=torch.long, device=dev)
                x_e = hidden_states.index_select(0, tok_t).to(dtype=gu_raw.dtype)

                gu_out = F.linear(x_e, gu_raw)
                g_part, u_part = gu_out.chunk(2, dim=-1)
                act = act_fn(g_part) * u_part
                y_e = F.linear(act, dn_raw)

                _, used_cuda = launch_pure_mirror_line_step(line_spine, y_e, tok_t, top_k_weights[tok_t, kpos_t])
                if used_cuda:
                    self.line_cuda_launches += 1
                c.routed_slots_executed += len(toks)
                c.expert_calls += 1

                if eid in crystallize_set:
                    self._crystallize_with_probe(
                        li, eid, gu_raw, dn_raw, x_e, gu_out, act, y_e, cold_depths[eid], all_eids_in_layer
                    )
                    self.prefill_crystallized += 1
                else:
                    self.prefill_direct_pass += 1
                    self.prefill_carves_avoided += 1
                del gu_raw, dn_raw

            gpu_ms = (time.perf_counter() - t_g0) * 1000.0
            # M2 exhales finished one-off burst immediately after CUDA consumer event
            self.region_pool.finish_region(reg_key, keep_if_reusable=False)
            # M1 observes physical GPU-vs-IO timing to adapt next layer's dispatch burst span
            self.load_dispatcher.observe_dispatch_timing(
                t_gpu_ms=gpu_ms if burst_i > 0 else max(gpu_ms, prev_gpu_ms),
                t_io_ms=io_ms,
                t_wait_ms=wait_ms,
                items_dispatched=len(b_eids),
            )
            self.last_trace = {
                "layer": li,
                "burst_eids": list(b_eids),
                "is_prefill": True,
                "m1_mode": self.load_dispatcher.dispatch_mode,
                "m1_span": self.load_dispatcher.batch_span,
                "m2_live_regions": self.region_pool.live_regions,
                "imprint_cached": len(c._lru),
            }
            burst_i += 1

        # Prime Linear CPU->GPU Raw Byte Injector right at the end of prefill so L0..L2 are ready for token #1!
        if li == int(c.num_layers) - 1 and self._has_raw_injector():
            def _is_res_pf(target_l: int, eid: int) -> bool:
                return ((target_l, eid) in c._lru) or ((target_l, eid) in c._host_lru) or self._in_l3(target_l, eid)
            self._pump_raw_injector(li, _is_res_pf)

        return line_spine.to(hidden_states.dtype)

    @torch.no_grad()
    def _forward_decode_v4(
        self,
        li: int,
        hidden_states: torch.Tensor,
        top_k_weights: torch.Tensor,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        valid: List[Tuple[int, int, float]],
        all_eids_in_layer: set[int],
    ) -> torch.Tensor:
        """Conveyor-Free Decode: M2 direct adoption + Kernel #6 anchored Q1 + strict local line_spine mirror."""
        c = self.cache
        h_dim = int(hidden_states.shape[1])
        dev = hidden_states.device
        if not valid:
            return torch.zeros_like(hidden_states)
        self.decode_layers_total += 1
        misses_before = int(c.misses)

        # 1. Adopt background M4 post-prefill region & any finished async Q1 crystal (ZERO blocking!)
        self._drain_bg_crystal(li, all_eids_in_layer, force=False)
        warm_layer, warm_eids = self._adopt_post_prefill_warm(li, all_eids_in_layer, force=False)
        if warm_layer == li and warm_eids:
            self.post_prefill_warmup_scale.record_warmup_outcome(
                li, warm_eids, [e for e, _, _ in valid]
            )

        # 2. Independent M1 physical GPU load check
        self._apply_m1_load(li, all_eids_in_layer)

        hot_items: List[Tuple[int, int, float]] = []
        cold_queue: List[Tuple[int, int, float]] = []
        tok_total_w = sum(rw for _, _, rw in valid) or 1.0
        eff_min = min(self.rest_min, max(1, len(valid) - 1)) if len(valid) > 1 else len(valid)
        for _rank, (e_int, k_pos, rw) in enumerate(valid):
            is_hot = ((li, e_int) in c._lru) or ((li, e_int) in c._host_lru) or self._in_l3(li, e_int)
            if is_hot:
                hot_items.append((e_int, k_pos, rw))
            else:
                cold_queue.append((e_int, k_pos, rw))

        line_spine = torch.zeros((1, h_dim), dtype=torch.float32, device=dev)
        tok_t = (
            self._decode_tok_zero
            if self._decode_tok_zero.device == dev
            else torch.zeros((1,), dtype=(torch.int32 if dev.type == "cuda" else torch.long), device=dev)
        )
        line_metrics = (
            self._decode_line_metrics
            if self._decode_line_metrics.device == dev
            else torch.empty((1, 4), dtype=torch.float32, device=dev)
        )
        done_count = 0
        w_done = 0.0
        prev_y_ref: Optional[torch.Tensor] = None
        prev_y_rw: float = 0.0
        last_y_ref: Optional[torch.Tensor] = None
        last_y_rw: float = 0.0

        # 2b. True GPU+NVMe Overlap: launch guaranteed-mandatory cold experts onto async O_DIRECT+DMA
        #     BEFORE running hot_items — skipping any expert already injected into VRAM by the Linear CPU->GPU stream!
        in_entry_cone = li < self.entry_cone_layers
        pre_injected_eids: set[int] = set()
        if li in self._raw_window:
            pre_injected_eids.update(self._raw_window[li][0].keys())
        if li in self._raw_inflight:
            pre_injected_eids.update(self._raw_inflight[li][1])
        eager_fut = None
        eager_eids: List[int] = []
        if cold_queue and hot_items:
            sim_done_e = len(hot_items)
            sim_w_e = sum(rw for _, _, rw in hot_items)
            for e_int, _k_pos, rw in cold_queue:
                rem_w_e = max(0.0, tok_total_w - sim_w_e)
                if in_entry_cone or sim_done_e < eff_min or sim_w_e < rem_w_e:
                    if e_int not in pre_injected_eids and not (
                        in_entry_cone and ("cone_bf16", li, e_int) in self.region_pool._regions
                    ):
                        eager_eids.append(e_int)
                    sim_done_e += 1
                    sim_w_e += rw
                else:
                    break
            if eager_eids:
                eager_fut = self._io_pool.submit(
                    self._fetch_raw_region_async, li, eager_eids, self._next_io_stream()
                )
                self.decode_eager_overlaps += 1

        # 3. Execute resident hot experts via Kernel #6 / Batched Kernel #6c (ZERO per-expert .item() syncs!)
        max_rw = max((rw for _, _, rw in valid), default=1.0) or 1.0
        w_f32_row = top_k_weights if top_k_weights.dtype == torch.float32 else top_k_weights.float()

        ready_hot: List[Tuple[int, int, float, Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, Optional[Tuple[Any, ...]]]] = []
        for e_int, k_pos, rw in hot_items:
            pair = self._get_hot_or_host(li, e_int, all_eids_in_layer)
            if pair is None:
                cold_queue.append((e_int, k_pos, rw))
                continue
            gu_t, dn_t = pair
            key = (li, e_int)
            # Reversible Expand o Compress: reattach shed Q1 scar plane if this expert is strong & room exists
            if key in self._shed_scars and rw >= 0.5 * max_rw:
                s_gu, s_dn, sb = self._shed_scars[key]
                if c._cached_bytes + sb <= c.max_cache_bytes and self.load_dispatcher.can_admit_bytes(sb):
                    self._shed_scars.pop(key, None)
                    self._shed_scar_bytes -= sb
                    if s_gu is not None:
                        gu_t = _reattach_q1_scar(gu_t, s_gu, self.device)
                    if s_dn is not None:
                        dn_t = _reattach_q1_scar(dn_t, s_dn, self.device)
                    new_b = _tensor_dict_bytes(gu_t) + _tensor_dict_bytes(dn_t)
                    old_ent = c._lru.get(key)
                    if old_ent is not None:
                        c._cached_bytes += new_b - old_ent[2]
                        c._lru[key] = (gu_t, dn_t, new_b)
                    self.shed_scars_restored += 1
            stored_k = min(int(gu_t["packed_lo"].shape[0]), int(dn_t["packed_lo"].shape[0]))
            p_use = mirror_cube_depth_math(rw / max_rw, rw / tok_total_w, min(self.rim_k, stored_k), stored_k)
            anc = c._mirror_anchors.get(key)
            ready_hot.append((e_int, k_pos, rw, gu_t, dn_t, p_use, anc))

        batched_ok = False
        if (
            len(ready_hot) >= 2
            and len(ready_hot) <= 16
            and dev.type == "cuda"
            and c.use_fwht
            and int(c.group_size) == 64
        ):
            gu_entries = []
            dn_entries = []
            kpos_list: List[int] = []
            for _e_int, k_pos, _rw, gu_t, dn_t, p_use, anc in ready_hot:
                x_gu_a, y_gu_a, x_dn_a, y_dn_a = anc if anc is not None else (None, None, None, None)
                gu_entries.append((gu_t, p_use, x_gu_a, y_gu_a))
                dn_entries.append((dn_t, p_use, x_dn_a, y_dn_a))
                kpos_list.append(k_pos)
            gu_all = launch_pure_mirror_diff_q1_moe_batch(
                hidden_states,
                gu_entries,
                shared_x=True,
                group_size=64,
                min_planes=1,
                mirror_tol=self.mirror_diff_tol,
            )
            if gu_all is not None:
                g_all, u_all = gu_all.chunk(2, dim=-1)
                act_all = act_fn(g_all) * u_all
                y_all = launch_pure_mirror_diff_q1_moe_batch(
                    act_all,
                    dn_entries,
                    shared_x=False,
                    group_size=64,
                    min_planes=1,
                    mirror_tol=self.mirror_diff_tol,
                )
                if y_all is not None:
                    w_vec = w_f32_row[0, kpos_list]
                    line_spine.add_((y_all.float() * w_vec.unsqueeze(-1)).sum(dim=0, keepdim=True))
                    batched_ok = True
                    h_cnt = len(ready_hot)
                    self.diff_q1_calls += 2 * h_cnt
                    for _e_int, _kp, rw, gu_t, dn_t, p_use, anc in ready_hot:
                        if anc is not None:
                            if anc[0] is not None and anc[1] is not None:
                                self.diff_q1_anchored_calls += 1
                            if anc[2] is not None and anc[3] is not None:
                                self.diff_q1_anchored_calls += 1
                        self.decode_norm_syncs_avoided += 1
                        done_count += 1
                        w_done += rw
                        c.routed_slots_executed += 1
                        c.expert_calls += 1
                        c.expert_cubes_executed += (
                            min(p_use, gu_t["packed_lo"].shape[0]) + min(p_use, dn_t["packed_lo"].shape[0])
                        ) / 2
                    prev_y_ref, prev_y_rw = y_all[-2:-1], ready_hot[-2][2]
                    last_y_ref, last_y_rw = y_all[-1:], ready_hot[-1][2]

        if not batched_ok:
            for e_int, k_pos, rw, gu_t, dn_t, p_use, anc in ready_hot:
                key = (li, e_int)
                x_gu_a, y_gu_a, x_dn_a, y_dn_a = anc if anc is not None else (None, None, None, None)

                gu = self._run_proj_diff_anchored(key, hidden_states, gu_t, p_use, x_gu_a, y_gu_a)
                g_part, u_part = gu.chunk(2, dim=-1)
                act = act_fn(g_part) * u_part
                y_e = self._run_proj_diff_anchored(key, act, dn_t, p_use, x_dn_a, y_dn_a)

                w_t = w_f32_row[0:1, k_pos]
                _, used_cuda = launch_pure_mirror_line_step(line_spine, y_e, tok_t, w_t, out_metrics=line_metrics)
                if used_cuda:
                    self.line_cuda_launches += 1

                # Defer norm evaluation: keep only last two GPU tensor refs; avoid per-expert CPU-GPU sync!
                prev_y_ref, prev_y_rw = last_y_ref, last_y_rw
                last_y_ref, last_y_rw = y_e, rw
                self.decode_norm_syncs_avoided += 1
                done_count += 1
                w_done += rw
                c.routed_slots_executed += 1
                c.expert_calls += 1
                c.expert_cubes_executed += (
                    min(p_use, gu_t["packed_lo"].shape[0]) + min(p_use, dn_t["packed_lo"].shape[0])
                ) / 2

        # 4. Local M5 Mirror Equality on line_spine for remaining cold experts (strict `and` -> crisp spelling!)
        cold_to_run: List[Tuple[int, int, float, bool]] = []
        line_norm = 0.0
        last_delta_norm = 0.0
        prev_delta_norm = 0.0

        # Compute & sync [line_norm, last_norm, prev_norm] ONCE per layer ONLY if a cold candidate will enter mirror rest check
        needs_mirror_norms = False
        if cold_queue and not in_entry_cone:
            sim_done = done_count
            sim_w = w_done
            for _e_int, _k_pos, rw in cold_queue:
                if sim_done < eff_min or sim_w < max(0.0, tok_total_w - sim_w):
                    sim_done += 1
                    sim_w += rw
                else:
                    needs_mirror_norms = True
                    break

        if needs_mirror_norms:
            if self._async_norm_cpu is not None and dev.type == "cuda":
                # Non-blocking async mirror-norm harvesting: NEVER stall the GPU stream with `.tolist()`!
                # First harvest any previously completed async norm copy from an earlier layer/token.
                if self._async_norm_ev is not None and self._async_norm_meta is not None and self._async_norm_ev.query():
                    p_li, p_lrw, p_prw = self._async_norm_meta
                    self._layer_norm_trace[p_li] = (
                        float(self._async_norm_cpu[0]),
                        float(self._async_norm_cpu[1]) * p_lrw,
                        float(self._async_norm_cpu[2]) * p_prw,
                    )
                    self._async_norm_ev = None
                    self._async_norm_meta = None
                # Queue a fresh non-blocking copy if slot is free
                if self._async_norm_ev is None:
                    z0 = line_spine.new_zeros(())
                    norm_vec = torch.stack([
                        line_spine.norm(),
                        last_y_ref.float().norm() if last_y_ref is not None else z0,
                        prev_y_ref.float().norm() if prev_y_ref is not None else z0,
                    ])
                    self._async_norm_cpu.copy_(norm_vec, non_blocking=True)
                    ev = torch.cuda.Event(enable_timing=False)
                    ev.record(torch.cuda.current_stream(dev))
                    self._async_norm_ev = ev
                    self._async_norm_meta = (li, last_y_rw, prev_y_rw)
                prev_trace = self._layer_norm_trace.get(li)
                if prev_trace is not None:
                    line_norm, last_delta_norm, prev_delta_norm = prev_trace
                self.decode_norm_syncs_avoided += 1
            else:
                norm_tensors = [line_spine.norm()]
                if last_y_ref is not None:
                    norm_tensors.append(last_y_ref.float().norm())
                if prev_y_ref is not None:
                    norm_tensors.append(prev_y_ref.float().norm())
                norm_vals = torch.stack(norm_tensors).tolist()
                self.decode_norm_syncs_performed += 1
                line_norm = float(norm_vals[0])
                if last_y_ref is not None:
                    last_delta_norm = float(norm_vals[1]) * last_y_rw
                if prev_y_ref is not None:
                    prev_delta_norm = float(norm_vals[2 if last_y_ref is not None else 1]) * prev_y_rw
        else:
            self.decode_norm_syncs_avoided += 1
        prev_y_ref = None
        last_y_ref = None

        for e_int, k_pos, rw in cold_queue:
            rem_w = max(0.0, tok_total_w - w_done)
            if done_count < eff_min or w_done < rem_w:
                cold_to_run.append((e_int, k_pos, rw, False))
                w_done += rw
                done_count += 1
                self.decode_cold_woken_split += 1
                continue

            if in_entry_cone:
                cold_to_run.append((e_int, k_pos, rw, True))
                w_done += rw
                done_count += 1
                self.decode_direct_unclosed += 1
                continue

            if self.rest_tol <= 0.0:
                cold_to_run.append((e_int, k_pos, rw, False))
                w_done += rw
                done_count += 1
                self.decode_cold_woken_split += 1
                continue

            mean_w = w_done / max(1, done_count)
            est_next_delta = max(last_delta_norm, 0.5 * prev_delta_norm) * (rw / max(mean_w, 1e-6))
            scale_ref = max(line_norm, est_next_delta, 1e-6)
            bal = eval_mirror_equality_step(
                (line_norm / scale_ref, w_done / tok_total_w),
                (max(1e-6, last_delta_norm / scale_ref), mean_w / tok_total_w),
                (max(1e-6, est_next_delta / scale_ref), rw / tok_total_w),
                ((line_norm + est_next_delta) / scale_ref, min(1.0, (w_done + rw) / tok_total_w)),
            )
            rel_tail_impact = est_next_delta / (line_norm + 1e-6)
            # Rest ONLY when BOTH canonical L4 mirror converged (EQUAL) AND explicit caller error bound < rest_tol
            if bal.state == "EQUAL" and rel_tail_impact < self.rest_tol:
                c.cold_experts_rested += 1
                c.cold_disk_reads_avoided += 1
                self.decode_cold_rested_equal += 1
                continue

            cold_to_run.append((e_int, k_pos, rw, False))
            w_done += rw
            done_count += 1
            self.decode_cold_woken_split += 1

        # 5. Execute woken cold experts directly through M2 MirrorRegionBreathingPool (NO conveyor!)
        #    Recurring entry-cone experts (`L0..L3`) stay warm in M2 (`HOLD_WARM_REGION -> PIN_REUSE`)!
        if cold_to_run:
            if self._warm_future is not None and self._warm_target_layer == li:
                self._adopt_post_prefill_warm(li, all_eids_in_layer, force=True)

            cone_reused: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
            need_fetch: List[Tuple[int, int, float, bool]] = []
            for item in cold_to_run:
                e_int = item[0]
                cone_key = ("cone_bf16", li, e_int)
                hit_reg = self.region_pool.lookup_reuse(cone_key) if in_entry_cone else None
                if hit_reg is not None:
                    cone_reused[e_int] = hit_reg
                else:
                    need_fetch.append(item)

            raw_map: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
            reg_keys_to_finish: List[Tuple[Any, ...]] = []
            io_ms = 0.0
            wait_ms = 0.0
            fetched_cnt = 0
            if eager_fut is not None:
                tw0 = time.perf_counter()
                e_map_in, e_gpu_b, e_ev, e_io_ms = eager_fut.result()
                wait_ms += (time.perf_counter() - tw0) * 1000.0
                io_ms += e_io_ms
                eager_fut = None
                e_reg_key = ("decode_eager", li, tuple(eager_eids))
                admitted_e = self.region_pool.admit_region(
                    e_reg_key, e_map_in, gpu_bytes=e_gpu_b, ready_event=e_ev, recurring_hint=False
                )
                raw_map.update(admitted_e)
                reg_keys_to_finish.append(e_reg_key)
                fetched_cnt += len(eager_eids)

            if need_fetch and self._has_raw_injector():
                want_raw = [e for e, _, _, _ in need_fetch if e not in raw_map]
                if want_raw:
                    pre_deq = self._consume_raw_window_for_layer(li, want_raw)
                    if pre_deq:
                        raw_map.update(pre_deq)

            if need_fetch:
                extra_eids = [e for e, _, _, _ in need_fetch if e not in raw_map]
                if extra_eids:
                    reg_key = ("decode", li, tuple(extra_eids))
                    tw0 = time.perf_counter()
                    fut = self._io_pool.submit(self._fetch_raw_region_async, li, extra_eids, self._next_io_stream())
                    raw_map_in, gpu_b, ready_ev, x_io_ms = fut.result()
                    wait_ms += (time.perf_counter() - tw0) * 1000.0
                    io_ms += x_io_ms
                    admitted_x = self.region_pool.admit_region(
                        reg_key, raw_map_in, gpu_bytes=gpu_b, ready_event=ready_ev, recurring_hint=False
                    )
                    raw_map.update(admitted_x)
                    reg_keys_to_finish.append(reg_key)
                    fetched_cnt += len(extra_eids)

            t_g0 = time.perf_counter()
            for e_int, k_pos, _rw, is_unclosed in cold_to_run:
                if e_int in cone_reused:
                    gu_raw, dn_raw = cone_reused[e_int]
                    from_cone_pool = True
                else:
                    gu_raw, dn_raw = raw_map[e_int]
                    from_cone_pool = False
                    c.misses += 1
                x_e = hidden_states if hidden_states.dtype == gu_raw.dtype else hidden_states.to(dtype=gu_raw.dtype)
                gu_out = F.linear(x_e, gu_raw)
                g_part, u_part = gu_out.chunk(2, dim=-1)
                act = act_fn(g_part) * u_part
                y_e = F.linear(act, dn_raw)
                w_t = w_f32_row[0:1, k_pos]
                _, used_cuda = launch_pure_mirror_line_step(line_spine, y_e, tok_t, w_t, out_metrics=line_metrics)
                if used_cuda:
                    self.line_cuda_launches += 1
                c.routed_slots_executed += 1
                c.expert_calls += 1

                key = (li, e_int)
                if from_cone_pool:
                    self.region_pool.finish_region(("cone_bf16", li, e_int), keep_if_reusable=True)
                elif is_unclosed and key in self._decode_seen_cold:
                    eb = (
                        int(gu_raw.numel()) * int(gu_raw.element_size())
                        + int(dn_raw.numel()) * int(dn_raw.element_size())
                    )
                    if eb <= self.region_pool.max_pool_bytes // 4 and self.load_dispatcher.can_admit_bytes(eb, margin_bytes=256 * 1024**2):
                        ckey = ("cone_bf16", li, e_int)
                        self.region_pool.admit_region(ckey, (gu_raw, dn_raw), gpu_bytes=eb, recurring_hint=True)
                        self.region_pool.finish_region(ckey, keep_if_reusable=True)
                elif (not is_unclosed) and (key in self._decode_seen_cold):
                    self._submit_bg_crystal(
                        li, e_int, gu_raw, dn_raw, x_e, gu_out, act, y_e, self.mid_k, all_eids_in_layer
                    )
                else:
                    self._decode_seen_cold.add(key)
                    if not is_unclosed:
                        self.decode_direct_oneoff += 1

            gpu_ms = (time.perf_counter() - t_g0) * 1000.0
            for rk in reg_keys_to_finish:
                self.region_pool.finish_region(rk, keep_if_reusable=False)
            if fetched_cnt > 0:
                self.load_dispatcher.observe_dispatch_timing(
                    t_gpu_ms=gpu_ms, t_io_ms=io_ms, t_wait_ms=wait_ms, items_dispatched=fetched_cnt
                )
        elif eager_fut is not None:
            try:
                eager_fut.result()
            except Exception:
                pass

        if int(c.misses) == misses_before:
            self.zero_disk_spike_layers += 1

        # 6. M4 Post-Prefill Known-Region Warmup (background fetch without any conveyor slot!)
        def _is_resident(target_l: int, eid: int) -> bool:
            return (
                ((target_l, eid) in c._lru)
                or ((target_l, eid) in c._host_lru)
                or self._in_l3(target_l, eid)
            )

        plan = self.post_prefill_warmup_scale.observe_and_plan_next(
            li,
            [e for e, _, _ in valid],
            [rw for _, _, rw in valid],
            _is_resident,
        )
        if self._has_raw_injector():
            # Linear CPU->GPU Raw Byte Injector replaces legacy narrow M4 Q1-carve warmup:
            # CPU streams wide raw quantized byte corridors ahead of GPU without computing/carving!
            self._pump_raw_injector(li, _is_resident)
        elif plan.eids_to_warm and self._warm_future is None:
            self._warm_target_layer = int(plan.target_layer)
            self._warm_eids = tuple(plan.eids_to_warm)
            self._warm_future = self._io_pool.submit(
                self._fetch_post_prefill_warm_async,
                int(plan.target_layer),
                list(plan.eids_to_warm),
                self._next_io_stream(),
            )

        self.last_trace = {
            "layer": li,
            "is_prefill": False,
            "hot_count": len(hot_items),
            "cold_executed": len(cold_to_run),
            "m1_mode": self.load_dispatcher.dispatch_mode,
            "m2_live_regions": self.region_pool.live_regions,
            "m4_next_plan": list(plan.eids_to_warm),
            "m4_state": plan.mirror_state,
            "imprint_cached": len(c._lru),
        }
        return line_spine.to(hidden_states.dtype)

    def stats(self) -> Dict[str, Any]:
        c = self.cache
        depth_hist: Dict[int, int] = defaultdict(int)
        for gu_t, _dn_t, _b in c._lru.values():
            depth_hist[int(gu_t["packed_lo"].shape[0])] += 1
        return {
            "version": "v4_conveyor_free_isolated_mirror",
            "conveyor_removed": True,
            "graphify_removed": True,
            "cross_scale_coupling": False,
            "vram_depth_histogram": {str(k): v for k, v in sorted(depth_hist.items())},
            "depth_sheds": self.depth_sheds,
            "mib_freed_by_shedding": round(self.bytes_freed_by_shedding / 1024**2, 2),
            "evictions": self.evictions,
            "m1_load_dispatcher": self.load_dispatcher.stats(),
            "m2_region_breathing_pool": self.region_pool.stats(),
            "m4_post_prefill_warmup": self.post_prefill_warmup_scale.stats(),
            "m5_layer_execution": {
                "scale": "M5_layer_moe_execution",
                "prefill_crystallized": self.prefill_crystallized,
                "total_crystallized": self.total_crystallized,
                "prefill_direct_pass": self.prefill_direct_pass,
                "prefill_carves_avoided": self.prefill_carves_avoided,
                "crystal_k_opt_avg": round(
                    self.crystal_k_opt_sum / max(1, 2 * max(self.total_crystallized, self.prefill_crystallized)), 2
                ) if max(self.total_crystallized, self.prefill_crystallized) > 0 else 0.0,
                "crystal_early_stops": self.crystal_early_stops,
                "crystal_scars_active": len(self._crystal_scars),
                "decode_direct_unclosed": self.decode_direct_unclosed,
                "decode_direct_oneoff": self.decode_direct_oneoff,
                "decode_cold_rested_equal": self.decode_cold_rested_equal,
                "decode_cold_woken_split": self.decode_cold_woken_split,
                "zero_disk_spike_layers": self.zero_disk_spike_layers,
                "zero_disk_spike_pct": round(
                    100.0 * self.zero_disk_spike_layers / max(1, self.decode_layers_total), 2
                ),
                "shed_scars_saved": self.shed_scars_saved,
                "shed_scars_restored": self.shed_scars_restored,
                "diff_q1_calls": self.diff_q1_calls,
                "diff_q1_anchored_calls": self.diff_q1_anchored_calls,
                "anchors_resident": len(c._mirror_anchors),
                "line_cuda_launches": self.line_cuda_launches,
                "decode_norm_syncs_performed": self.decode_norm_syncs_performed,
                "decode_norm_syncs_avoided": self.decode_norm_syncs_avoided,
                "decode_d2h_coalesced": self.decode_d2h_coalesced,
                "decode_eager_overlaps": self.decode_eager_overlaps,
                "bg_crystals_completed": self._bg_crystals_completed,
                "raw_injector_layers_pushed": self.raw_injector_layers_pushed,
                "raw_injector_experts_pushed": self.raw_injector_experts_pushed,
                "raw_injector_hits": self.raw_injector_hits,
                "raw_injector_misses": self.raw_injector_misses,
                "raw_injector_hit_rate_pct": round(
                    100.0 * self.raw_injector_hits / max(1, self.raw_injector_hits + self.raw_injector_misses), 1
                ),
                "post_prefill_bg_adoptions": self.post_prefill_bg_adoptions,
                "l3_enabled": self.l3_cache is not None,
                "l3_spills": self.l3_cache.spills if self.l3_cache is not None else 0,
                "l3_spills_deduped": self.l3_cache.spills_deduped if self.l3_cache is not None else 0,
                "l3_hits": self.l3_cache.hits if self.l3_cache is not None else 0,
                "l3_evictions": self.l3_cache.evictions if self.l3_cache is not None else 0,
                "l3_mib_used": self.l3_cache.stats()["mib_used"] if self.l3_cache is not None else 0.0,
            },
            "l3_disk_cache": self.l3_cache.stats() if self.l3_cache is not None else {"enabled": False},
            "last_trace": dict(self.last_trace),
        }

    def close(self) -> None:
        self._clear_raw_injector()
        if self._raw_injector_pool is not None:
            self._raw_injector_pool.shutdown(wait=True, cancel_futures=True)
            self._raw_injector_pool = None
        if self._crystal_future is not None:
            try:
                self._drain_bg_crystal(-1, set(), force=True)
            except Exception:
                self._crystal_future = None
        if self._crystal_pool is not None:
            self._crystal_pool.shutdown(wait=True, cancel_futures=True)
            self._crystal_pool = None
        if self._warm_future is not None:
            try:
                self._warm_future.result()
            except Exception:
                pass
            self._warm_future = None
            self._warm_target_layer = None
            self._warm_eids = ()
        if self._io_pool is not None:
            self._io_pool.shutdown(wait=True, cancel_futures=True)
            self._io_pool = None
        if self.l3_cache is not None:
            if getattr(self.cache, "_host_eviction_sink", None) == self.l3_cache.spill_async:
                self.cache._host_eviction_sink = None
            self.l3_cache.close()
            self.l3_cache = None
        self._shed_scars.clear()
        self._shed_scar_bytes = 0
        self.load_dispatcher.close()
        self.region_pool.close()


@torch.no_grad()
def warmup_v4_cuda_kernels(device: torch.device) -> int:
    """Compile & warm critical V4 NVRTC CUDA kernels once on tiny synthetic tensors (Scale M3 helper)."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0
    warmed = 0
    try:
        from src.binary_halving_q1 import carve_q1_cubes_incremental
        from src.quintessentia_q1_cube_nvrtc import launch_ideal_q1_fwht_gemv

        w = torch.randn((64, 64), dtype=torch.bfloat16, device=device)
        x = torch.randn((1, 64), dtype=torch.bfloat16, device=device)
        t, _ = carve_q1_cubes_incremental(w, 3, root_bits=8, group_size=64, use_fwht=True, sinkhorn_iters=1)
        y_out = torch.empty((1, 64), dtype=torch.bfloat16, device=device)
        launch_ideal_q1_fwht_gemv(
            x, t["packed_lo"], t["qscale"], t["qzero"], t["row_scale"], t["col_scale"], y_out,
            group_size=64, total_planes=3, plane_start=0, plane_count=3, bias=None,
        )
        warmed += 1
        y_d = launch_pure_mirror_diff_q1_gemv(
            x, t["packed_lo"], t["qscale"], t["qzero"], t["row_scale"], t["col_scale"],
            x_last=x.reshape(-1), y_last=y_out.reshape(-1), group_size=64,
            max_planes=3, min_planes=1, mirror_tol=0.02,
        )
        warmed += 1
        line = torch.zeros((1, 64), dtype=torch.float32, device=device)
        tok_idx = torch.zeros((1,), dtype=torch.long, device=device)
        rw = torch.ones((1,), dtype=torch.float32, device=device)
        _, used_l = launch_pure_mirror_line_step(line, y_d, tok_idx, rw)
        if used_l:
            warmed += 1
        torch.cuda.synchronize(device)
        del w, x, t, y_out, y_d, line, tok_idx, rw
    except Exception:
        pass
    return warmed
