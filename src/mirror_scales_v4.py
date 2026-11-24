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

from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.pure_mirror_agents_nvrtc import (
    launch_pure_mirror_diff_q1_gemv,
    launch_pure_mirror_line_step,
)


# ---------------------------------------------------------------------------
# Pure Stateless Bracketless Mirror Equality Primitive (Read from the Center)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MirrorBalanceResult:
    """Result of one stateless center-read Mirror Equality evaluation."""

    state: str                    # "EQUAL" | "ALIGN" | "SPLIT" | "PROTECT"
    strategy: Tuple[float, ...]   # S_t = 0.5 * (ObservedBehavior_(t-1) + ObservedBehavior_(t+1))
    d_core: float                 # ||Obs_(t-1) - Obs_(t+1)|| / (||Obs_(t-1)|| + ||Obs_(t+1)|| + eps)
    d_wings: float                # ||(Out_(t-1)+Obs_(t-1)) - (Obs_(t+1)+Out_(t+1))|| / denom
    tension: float                # max(d_core, 0.5 * d_wings)
    contra_core: bool             # True when past & future ObservedBehavior oppose each other


def _vec_norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def eval_mirror_equality_step(
    outcome_prev: Sequence[float],
    obs_prev: Sequence[float],
    obs_next: Sequence[float],
    outcome_next: Sequence[float],
    *,
    equal_tol: float = 0.125,
    split_tol: float = 0.375,
) -> MirrorBalanceResult:
    """Evaluate bracketless Mirror Equality from the center on local physical observables.

    Central stem (Stone Law):
        ObservedBehavior_(t-1) = Strategy_t = ObservedBehavior_(t+1)
    Full mirror wings:
        Outcome_(t-1) + ObservedBehavior_(t-1) = Strategy_t = ObservedBehavior_(t+1) + Outcome_(t+1)

    Pure function: holds ZERO state and ZERO references to any scale.
    """
    n = len(obs_prev)
    if not (len(outcome_prev) == n == len(obs_next) == len(outcome_next)) or n == 0:
        raise ValueError("All four mirror vectors must share the same positive length")

    eps = 1e-8
    strategy = tuple(0.5 * (float(a) + float(b)) for a, b in zip(obs_prev, obs_next))

    norm_p = _vec_norm(obs_prev)
    norm_n = _vec_norm(obs_next)
    diff_core = _vec_norm([float(a) - float(b) for a, b in zip(obs_prev, obs_next)])
    d_core = diff_core / (norm_p + norm_n + eps)

    dot_core = sum(float(a) * float(b) for a, b in zip(obs_prev, obs_next))
    contra_core = bool(norm_p > 1e-6 and norm_n > 1e-6 and dot_core < -1e-6)

    left_wing = [float(o) + float(b) for o, b in zip(outcome_prev, obs_prev)]
    right_wing = [float(b) + float(o) for b, o in zip(obs_next, outcome_next)]
    diff_wings = _vec_norm([a - b for a, b in zip(left_wing, right_wing)])
    d_wings = diff_wings / (_vec_norm(left_wing) + _vec_norm(right_wing) + eps)

    tension = max(d_core, 0.5 * d_wings)
    if contra_core:
        state = "PROTECT"
    elif tension <= float(equal_tol):
        state = "EQUAL"
    elif tension <= float(split_tol):
        state = "ALIGN"
    else:
        state = "SPLIT"

    return MirrorBalanceResult(
        state=state,
        strategy=strategy,
        d_core=d_core,
        d_wings=d_wings,
        tension=tension,
        contra_core=contra_core,
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

        if safety_reserve_bytes is not None:
            self.safety_reserve_bytes = max(1, int(safety_reserve_bytes))
        elif self.enabled:
            _fb, total_b, _gu, _mu, _pf, _src = self._read_hardware()
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
        self._prev_gpu_share: float = 0.5
        self._prev_io_share: float = 0.5
        self._prev_wait_share: float = 0.0
        self._last_source: str = "cpu"

        # Own telemetry
        self.evaluations: int = 0
        self.timing_updates: int = 0
        self.direct_burst_steps: int = 0
        self.overlap_async_steps: int = 0
        self.throttle_damp_steps: int = 0
        self.shed_protect_steps: int = 0
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

    def _read_hardware(self) -> Tuple[int, int, Optional[float], Optional[float], int, str]:
        """Return `(free_bytes, total_bytes, gpu_util_pct, mem_util_pct, torch_pool_free, source)`."""
        if self._sensor_fn is not None:
            return self._sensor_fn()
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return 0, 0, None, None, 0, "cpu"
        if self._monitor is not None:
            free_b, total_b, src = self._monitor.query()
            gpu_u, mem_u, _usrc = self._monitor.query_utilization()
        else:
            fb, tb = torch.cuda.mem_get_info(self.device)
            free_b, total_b, src = int(fb), int(tb), "cuda"
            gpu_u, mem_u = None, None
        reserved_b = int(torch.cuda.memory_reserved(self.device))
        allocated_b = int(torch.cuda.memory_allocated(self.device))
        pool_free = max(0, reserved_b - allocated_b)
        return free_b, total_b, gpu_u, mem_u, pool_free, src

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
        # 1. Hard physical wall contact -> SHED_PROTECT (0 OOM, 0 WDDM spill)
        if free_b < reserve:
            if pool_free > 0 and self._sensor_fn is None and self.device.type == "cuda":
                torch.cuda.empty_cache()
                free_b, total_b, gpu_u, mem_u, pool_free, src = self._read_hardware()
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
                return LoadDispatchDecision(
                    "SHED_PROTECT", "protect", target, deficit, emergency,
                    self.batch_span, self.in_flight_cap, "PROTECT", free_b, total_b, gpu_u, mem_u,
                )

        # 2. Local physical wings for center-read Mirror Equality
        used_frac = max(0.0, min(1.0, 1.0 - free_b / max(1, total_b)))
        load_frac = self._combine_load_frac(gpu_u, mem_u, used_frac)
        prev_free = free_b if self._prev_free_bytes is None else self._prev_free_bytes
        prev_used = used_frac if self._prev_used_frac is None else self._prev_used_frac
        prev_load = load_frac if self._prev_load_frac is None else self._prev_load_frac

        delta_free_drop = max(0, prev_free - free_b)
        projected_free_next = free_b - delta_free_drop
        safe_used_ceiling = max(0.0, 1.0 - self.reserve_ratio)

        out_prev = (prev_used, prev_load)
        obs_prev = (used_frac - prev_used, load_frac - prev_load)
        if projected_free_next < reserve:
            safe_next_used = min(used_frac, safe_used_ceiling)
            obs_next = (safe_next_used - used_frac, min(0.85, load_frac) - load_frac)
        else:
            obs_next = (0.0, 0.0)
        out_next = (min(used_frac, safe_used_ceiling), min(load_frac, 0.85))

        bal = eval_mirror_equality_step(out_prev, obs_prev, obs_next, out_next)
        self.last_mirror_state = bal.state
        self._prev_free_bytes = free_b
        self._prev_used_frac = used_frac
        self._prev_load_frac = load_frac

        headroom = free_b - reserve
        physical_ceiling = cached_b + headroom

        # 3. Preventive Mirror Damping BEFORE floor impact -> THROTTLE_DAMP
        bus_saturated = bool(mem_u is not None and mem_u >= 90.0 and delta_free_drop > 0)
        if projected_free_next < reserve or bus_saturated or bal.state == "PROTECT":
            damp_target = max(cached_b, cached_b + (headroom // 2))
            target = damp_target if cur_budget > damp_target else cur_budget
            self.dispatch_mode = "THROTTLE_DAMP"
            self.in_flight_cap = 1
            self.throttle_damp_steps += 1
            return LoadDispatchDecision(
                "THROTTLE_DAMP", "damp", target, 0, False,
                self.batch_span, self.in_flight_cap, bal.state, free_b, total_b, gpu_u, mem_u,
            )

        # 4. Ample headroom & calm bus -> DIRECT_BURST or OVERLAP_ASYNC
        self.in_flight_cap = self.max_in_flight
        if headroom >= 2 * reserve and (gpu_u is None or gpu_u < 65.0) and (mem_u is None or mem_u < 65.0):
            self.dispatch_mode = "DIRECT_BURST"
            self.direct_burst_steps += 1
        else:
            self.dispatch_mode = "OVERLAP_ASYNC"
            self.overlap_async_steps += 1

        if physical_ceiling > cur_budget + self.deadband_bytes:
            if mem_u is not None and mem_u >= 80.0:
                target = cur_budget + max(self.deadband_bytes, (physical_ceiling - cur_budget) // 2)
            else:
                target = physical_ceiling
            self.expansions += 1
            if target > self.max_observed_budget_bytes:
                self.max_observed_budget_bytes = target
            return LoadDispatchDecision(
                self.dispatch_mode, "expand", target, 0, False,
                self.batch_span, self.in_flight_cap, bal.state, free_b, total_b, gpu_u, mem_u,
            )

        self.equal_holds += 1
        return LoadDispatchDecision(
            self.dispatch_mode, "hold", cur_budget, 0, False,
            self.batch_span, self.in_flight_cap, bal.state, free_b, total_b, gpu_u, mem_u,
        )

    def observe_dispatch_timing(
        self,
        *,
        t_gpu_ms: float,
        t_io_ms: float,
        t_wait_ms: float,
        items_dispatched: int,
    ) -> Tuple[int, int, str]:
        """Update `(batch_span, in_flight_cap, mirror_state)` purely from physical GPU-vs-IO timings."""
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
            (self._prev_gpu_share, self._prev_io_share, self._prev_wait_share),
            (cur_i - cur_g, cur_w, cur_g - self._prev_gpu_share),
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            equal_tol=0.15,
            split_tol=0.40,
        )
        self._prev_gpu_share = cur_g
        self._prev_io_share = cur_i
        self._prev_wait_share = cur_w

        # Only adapt burst span on multi-item prefill bursts (single-token decode misses stay untouched)
        if items_dispatched >= self.min_batch_span and bal.state != "EQUAL":
            old_span = self.batch_span
            if ti > tg * 1.15 and tw > 0.25:
                # IO lagged behind GPU compute -> tighten burst span and deepen async overlap (up to 2)
                ratio = max(0.5, min(0.875, tg / max(ti, 1e-6)))
                new_span = max(self.min_batch_span, min(old_span - 1, int(round(old_span * ratio))))
                if new_span < old_span:
                    self.batch_span = new_span
                    self.span_shrinks += 1
                if self.dispatch_mode != "THROTTLE_DAMP" and self.dispatch_mode != "SHED_PROTECT":
                    self.in_flight_cap = self.max_in_flight
            elif tg > ti * 1.25 and tw <= 0.25 and items_dispatched >= old_span:
                # GPU compute dominated while IO finished with ~0 wait -> widen direct burst span
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

        # Center-read Mirror Equality on region persistence vs memory breathing headroom
        out_prev = (self._prev_reuse_rate, self._prev_occ_frac)
        obs_prev = (reuse_rate - self._prev_reuse_rate, occ_frac - self._prev_occ_frac)
        obs_next = (1.0 - reuse_rate if want_warm else -reuse_rate, -occ_frac)
        out_next = (1.0 if want_warm else 0.0, 0.5 * occ_frac)
        bal = eval_mirror_equality_step(out_prev, obs_prev, obs_next, out_next, equal_tol=0.18, split_tol=0.45)

        self.last_mirror_state = bal.state
        self._prev_reuse_rate = reuse_rate
        self._prev_occ_frac = occ_frac

        warm_count = sum(1 for r in self._regions.values() if r.state == "WARM")
        if (
            want_warm
            and bal.state != "PROTECT"
            and occ_frac < 0.75
            and warm_count < self.max_warm_regions
        ):
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
            self.last_mirror_state = "ALIGN"
            if warmed_so_far >= total_targets:
                return ColdWarmupStepDecision(False, "EQUAL", dt, 1, "all_warm")
            if self.max_warmup_seconds > 0.0 and elapsed_seconds >= self.max_warmup_seconds:
                return ColdWarmupStepDecision(False, "ALIGN", dt, 0, "time_budget")
            return ColdWarmupStepDecision(True, "ALIGN", dt, 0, "warming")

        scale_ref = max(self._prev_ms, dt, 1e-4)
        p_norm = self._prev_ms / scale_ref
        c_norm = dt / scale_ref

        out_prev = (p_norm, max(0.0, done_frac - 1.0 / max(1, total_targets)))
        obs_prev = (c_norm - p_norm, 1.0 / max(1, total_targets))
        obs_next = (0.0, 1.0 / max(1, total_targets))
        out_next = (c_norm, min(1.0, done_frac + 1.0 / max(1, total_targets)))

        bal = eval_mirror_equality_step(out_prev, obs_prev, obs_next, out_next, equal_tol=0.15, split_tol=0.35)
        self._prev_ms = dt
        self.last_mirror_state = bal.state

        if bal.state == "EQUAL":
            self._plateau_streak += 1
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
        for eid, rw in zip(actual_eids, actual_weights):
            e_int = int(eid)
            w_val = max(0.0, float(rw))
            if e_int in prefill_set:
                cov_w += w_val
            else:
                scar_eids.append((e_int, w_val))

        cur_cov = min(1.0, cov_w / tot_w) if prefill_set else 0.0
        cur_scar = max(0.0, 1.0 - cur_cov)

        prev_cov = self._prev_cov[li] if 0 <= li < self.num_layers else 1.0
        prev_scar = self._prev_scar[li] if 0 <= li < self.num_layers else 0.0

        bal = eval_mirror_equality_step(
            (prev_cov, prev_scar),
            (cur_cov - prev_cov, cur_scar - prev_scar),
            (1.0 - cur_cov, -cur_scar) if cur_cov >= 0.5 else (0.0, 0.0),
            (1.0, 0.0),
            equal_tol=0.15,
            split_tol=0.40,
        )
        self.last_mirror_state = bal.state
        if 0 <= li < self.num_layers:
            self._prev_cov[li] = cur_cov
            self._prev_scar[li] = cur_scar

        if scar_eids and (bal.state in ("SPLIT", "ALIGN", "PROTECT") or cur_scar > 0.25):
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
            (prev_hr, 1.0 - prev_hr),
            (hit_rate - prev_hr, prev_hr - hit_rate),
            (1.0 - hit_rate, hit_rate - 1.0),
            (1.0, 0.0),
            equal_tol=0.15,
            split_tol=0.40,
        )
        self._prev_hit_rate[li] = hit_rate
        if hit_rate >= 0.5 and bal.state in ("EQUAL", "ALIGN"):
            self._miss_streak[li] = 0
            self._width[li] = min(self.max_width, self._width[li] + (1 if hit_rate >= 0.99 else 0))
        elif hit_rate == 0.0:
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
    """Pure-math M5b cube depth from bracketless center-read probability balance (ZERO step-if thresholds!).

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
        rest_tol: float = 0.24,
        rest_min: int = 3,
        auto_vram_adapt: bool = False,
        vram_safety_reserve_bytes: Optional[int] = None,
        sensor_fn: Optional[Callable[[], Tuple[int, int, Optional[float], Optional[float], int, str]]] = None,
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
        # M4: Isolated Post-Prefill Known-Region Warmup Scale
        self.post_prefill_warmup_scale = MirrorPostPrefillRegionWarmupScale(
            num_layers=int(cache.num_layers),
            max_region_experts=10,
            initial_width=1,
            max_width=2,
        )

        # Async IO helper (2 dedicated CUDA transfer streams, 1 serialized GGUF/HF reader thread — NO ring!)
        self._io_streams = (
            [torch.cuda.Stream(device=self.device), torch.cuda.Stream(device=self.device)]
            if self.device.type == "cuda"
            else [None, None]
        )
        self._stream_cursor: int = 0
        self._io_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quintessentia-v4-io")
        self._warm_future = None
        self._warm_target_layer: Optional[int] = None
        self._warm_eids: Tuple[int, ...] = ()

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
        self.depth_sheds: int = 0
        self.shed_scars_saved: int = 0
        self.shed_scars_restored: int = 0
        self.bytes_freed_by_shedding: int = 0
        self.evictions: int = 0
        self.post_prefill_bg_adoptions: int = 0
        self.last_trace: Dict[str, Any] = {}

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
            return y_out.to(dtype=x_e.dtype)
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
            if ((li, eid) in c._lru) or ((li, eid) in c._host_lru):
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

        # 1. Adopt background M4 post-prefill region directly via M2 (if ready for this layer)
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
            is_hot = ((li, e_int) in c._lru) or ((li, e_int) in c._host_lru)
            if is_hot:
                hot_items.append((e_int, k_pos, rw))
            else:
                cold_queue.append((e_int, k_pos, rw))

        line_spine = torch.zeros((1, h_dim), dtype=torch.float32, device=dev)
        tok_t = torch.zeros((1,), dtype=torch.long, device=dev)
        done_count = 0
        w_done = 0.0
        prev_delta_norm = 0.0
        last_delta_norm = 0.0

        # 3. Execute resident hot experts via Kernel #6 Differential-Reversible Q1 GEMV
        max_rw = max((rw for _, _, rw in valid), default=1.0) or 1.0
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
            x_gu_a, y_gu_a, x_dn_a, y_dn_a = anc if anc is not None else (None, None, None, None)

            gu = self._run_proj_diff_anchored(key, hidden_states, gu_t, p_use, x_gu_a, y_gu_a)
            g_part, u_part = gu.chunk(2, dim=-1)
            act = act_fn(g_part) * u_part
            y_e = self._run_proj_diff_anchored(key, act, dn_t, p_use, x_dn_a, y_dn_a)

            w_t = top_k_weights[0:1, k_pos]
            _, used_cuda = launch_pure_mirror_line_step(line_spine, y_e, tok_t, w_t)
            if used_cuda:
                self.line_cuda_launches += 1

            prev_delta_norm = last_delta_norm
            last_delta_norm = float(y_e.float().norm().item()) * rw
            done_count += 1
            w_done += rw
            c.routed_slots_executed += 1
            c.expert_calls += 1
            c.expert_cubes_executed += (
                min(p_use, gu_t["packed_lo"].shape[0]) + min(p_use, dn_t["packed_lo"].shape[0])
            ) / 2

        # 4. Local M5 Mirror Equality on line_spine for remaining cold experts (strict `and` -> crisp spelling!)
        cold_to_run: List[Tuple[int, int, float, bool]] = []
        in_entry_cone = li < self.entry_cone_layers
        line_norm = float(line_spine.norm().item())

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

            mean_w = w_done / max(1, done_count)
            est_next_delta = max(last_delta_norm, 0.5 * prev_delta_norm) * (rw / max(mean_w, 1e-6))
            scale_ref = max(line_norm, est_next_delta, 1e-6)
            bal = eval_mirror_equality_step(
                (line_norm / scale_ref, w_done / tok_total_w),
                (last_delta_norm / scale_ref, mean_w / tok_total_w),
                (est_next_delta / scale_ref, rw / tok_total_w),
                ((line_norm + est_next_delta) / scale_ref, min(1.0, (w_done + rw) / tok_total_w)),
                equal_tol=self.rest_tol,
                split_tol=min(0.65, self.rest_tol * 2.0),
            )
            rel_tail_impact = est_next_delta / (line_norm + 1e-6)
            # Rest ONLY when BOTH center-read mirror converged (EQUAL) AND relative tail impact < rest_tol
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
            if self._warm_future is not None:
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
            reg_key = None
            io_ms = 0.0
            wait_ms = 0.0
            if need_fetch:
                eids_batch = [e for e, _, _, _ in need_fetch]
                reg_key = ("decode", li, tuple(eids_batch))
                tw0 = time.perf_counter()
                fut = self._io_pool.submit(self._fetch_raw_region_async, li, eids_batch, self._next_io_stream())
                raw_map_in, gpu_b, ready_ev, io_ms = fut.result()
                wait_ms = (time.perf_counter() - tw0) * 1000.0
                raw_map = self.region_pool.admit_region(
                    reg_key, raw_map_in, gpu_bytes=gpu_b, ready_event=ready_ev, recurring_hint=False
                )

            t_g0 = time.perf_counter()
            for e_int, k_pos, _rw, is_unclosed in cold_to_run:
                if e_int in cone_reused:
                    gu_raw, dn_raw = cone_reused[e_int]
                    from_cone_pool = True
                else:
                    gu_raw, dn_raw = raw_map[e_int]
                    from_cone_pool = False
                    c.misses += 1
                x_e = hidden_states.to(dtype=gu_raw.dtype)
                gu_out = F.linear(x_e, gu_raw)
                g_part, u_part = gu_out.chunk(2, dim=-1)
                act = act_fn(g_part) * u_part
                y_e = F.linear(act, dn_raw)
                w_t = top_k_weights[0:1, k_pos]
                _, used_cuda = launch_pure_mirror_line_step(line_spine, y_e, tok_t, w_t)
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
                    self._crystallize_with_probe(
                        li, e_int, gu_raw, dn_raw, x_e, gu_out, act, y_e, self.mid_k, all_eids_in_layer
                    )
                else:
                    self._decode_seen_cold.add(key)
                    if not is_unclosed:
                        self.decode_direct_oneoff += 1

            gpu_ms = (time.perf_counter() - t_g0) * 1000.0
            if reg_key is not None:
                self.region_pool.finish_region(reg_key, keep_if_reusable=False)
                self.load_dispatcher.observe_dispatch_timing(
                    t_gpu_ms=gpu_ms, t_io_ms=io_ms, t_wait_ms=wait_ms, items_dispatched=len(need_fetch)
                )

        if int(c.misses) == misses_before:
            self.zero_disk_spike_layers += 1

        # 6. M4 Post-Prefill Known-Region Warmup (background fetch without any conveyor slot!)
        def _is_resident(target_l: int, eid: int) -> bool:
            return ((target_l, eid) in c._lru) or ((target_l, eid) in c._host_lru)

        plan = self.post_prefill_warmup_scale.observe_and_plan_next(
            li,
            [e for e, _, _ in valid],
            [rw for _, _, rw in valid],
            _is_resident,
        )
        if plan.eids_to_warm and self._warm_future is None:
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
                "prefill_direct_pass": self.prefill_direct_pass,
                "prefill_carves_avoided": self.prefill_carves_avoided,
                "crystal_k_opt_avg": round(
                    self.crystal_k_opt_sum / max(1, 2 * self.prefill_crystallized), 2
                ) if self.prefill_crystallized > 0 else 0.0,
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
                "post_prefill_bg_adoptions": self.post_prefill_bg_adoptions,
            },
            "last_trace": dict(self.last_trace),
        }

    def close(self) -> None:
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
