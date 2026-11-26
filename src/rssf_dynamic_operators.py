"""Scale-Invariant Dynamic Balance Operators (RSSF + FDS + Enhanced Calculus + Dynamic Universe).

Implements Anton's unified four-layer framework as a **solver-free, O(1) (~2 us), scale-invariant
balance observer** across EVERY level of the Quintessentia generation pipeline — from a single
Q1 weight cube and speculative token step to KV-cache archive, prefill chunking, and hardware M1.

Core principle (zero hardcodes for any GPU size or model family):
  1. Dynamic Universe Equality (observed as a live balance of 8 numerator/denominator force pairs,
     NOT solved as a frozen static equation):
       U <-> T1(G*Q) + T2(G*S) + T3(Phase) + T4(I*Q) + T5(L*S) + T6(All5) + T7(Floor) + T8(Lin)
     where algebraic identity in Term 3 cancels +c^4 and -c^4 cleanly:
       D3 = C_L*U - Q + (G + b^2)*(S + a^2) + (I + L + a*b)^2 + 2*c^2*(I + L + a*b)
     revealing:
       - `d3_margin` <= 0   -> scale transition / phase break (impulse Q breached support pad)
       - `info_efficiency`  -> T4 / sum(T), useful information*impulse yield on this scale
       - `balance_ratio`    -> sum(T) / U (>1 expansion, ~1 live balance, <1 compression)
       - `dominant_term`    -> which force pair currently drives the step.
  2. Formula for Dynamic Systems (FDS):
       dF/dt = dE/dt - lambda(t) * dS/dt -> dimensionless `fds_t_end_steps` horizon & plateau.
  3. Enhanced Calculus (EC):
       dX/dt = F(X(t), dX/dt, integral_0^t K(t-tau) X(tau) dtau) -> discounted Volterra archive
       that cancels symmetric breathing (+/-) while catching quiet sustained drift early.
  4. RSSF v0.3 Section 16 (`StrategyCollapse`):
       LossOfArchive v LossOfObserverCoherence v LossOfScaleCoherence v UnboundedResultDrift.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Layer 3 (Core Perspective): Solver-Free 8-Term Balance Observer
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DynamicBalanceSnapshot:
    """Single O(1) direct observation of the 8-pair numerator/denominator force balance."""

    u_observed: float             # Measured live whole state U_t on this scale
    u_projected_sum: float        # Sum_{k=1..8} T_k(U_t)
    balance_ratio: float          # sum(T_k) / U_t  (>1 expand, ~1 balanced, <1 compress)
    d3_margin: float              # Term-3 denominator reserve (>0 scale holds, <=0 scale break!)
    info_efficiency: float        # Share of T4 = I*Q / (C_L*U + G + S) in total balance
    phase_share: float            # Share of T3 (phase transition channel) in total balance
    dominant_term: str            # Name of currently dominating force pair
    regime: str                   # "SCALE_BREAK" | "COMPRESS" | "BALANCED" | "EXPAND"
    terms: Tuple[Tuple[str, float], ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "u_observed": round(self.u_observed, 4),
            "u_projected_sum": round(self.u_projected_sum, 4),
            "balance_ratio": round(self.balance_ratio, 4),
            "d3_margin": round(self.d3_margin, 4),
            "info_efficiency_t4": round(self.info_efficiency, 4),
            "phase_share_t3": round(self.phase_share, 4),
            "dominant_term": self.dominant_term,
            "regime": self.regime,
        }


def observe_dynamic_balance(
    *,
    U: float,
    G: float,
    Q: float,
    S: float,
    I: float,
    L: float,
    C_L: float,
    a: float = 0.25,
    b: float = 0.25,
    c: float = 0.50,
) -> DynamicBalanceSnapshot:
    """Directly observe the 8-pair force balance at ANY scale of Information `I` (0 solver iterations!).

    Works identically whether `I` represents:
      - a single Q1 weight cube / bit-plane significance (micro scale),
      - a speculative draft token's confidence/acceptance archive (token scale),
      - a dialogue's KV-cache prefix archive (meso scale),
      - a hardware VRAM/bus execution balance (device scale),
      - a whole model or multi-agent swarm archive (macro/mega scale).
    """
    u_v = max(1e-4, float(U))
    g_v = max(1e-4, float(G))
    q_v = max(1e-4, float(Q))
    s_v = max(1e-4, float(S))
    i_v = max(1e-4, float(I))
    l_v = max(1e-4, float(L))
    cl_v = max(1e-3, float(C_L))

    a_v = max(1e-3, float(a))
    b_v = max(1e-3, float(b))
    c_v = max(1e-3, float(c))
    c2 = c_v * c_v
    c4 = c2 * c2
    il_ab = i_v + l_v + a_v * b_v
    quad_pad = (il_ab + c2) ** 2

    d12 = max(1e-9, cl_v * u_v + i_v + l_v)
    t1 = (g_v * q_v) / d12
    t2 = (g_v * s_v) / d12

    # Exact algebraic form of Term 3 denominator (note: +c^4 from quad_pad cancels -c^4 identically):
    # D3 = C_L*U - Q - c^4 + (G + b^2)*(S + a^2) + (I + L + a*b + c^2)^2
    d3_margin = cl_v * u_v - q_v - c4 + (g_v + b_v * b_v) * (s_v + a_v * a_v) + quad_pad
    t3 = (g_v * (i_v + l_v + q_v + s_v)) / max(1e-4, d3_margin)

    t4 = (i_v * q_v) / max(1e-9, cl_v * u_v + g_v + s_v)
    t5 = (l_v * s_v) / max(1e-9, cl_v * u_v + g_v + q_v)

    geom5 = (max(0.0, g_v * i_v * l_v * q_v * s_v)) ** 0.2
    t6 = ((g_v + i_v) * (l_v + q_v + s_v)) / max(1e-9, cl_v * u_v + geom5)

    t7 = ((g_v + s_v) * (a_v * a_v + b_v * b_v)) / max(1e-9, q_v + c4 + quad_pad)
    t8 = ((g_v + q_v + s_v) * (a_v + b_v + c_v)) / max(1e-9, cl_v + i_v + l_v + u_v)

    named = (
        ("T1_GQ", t1),
        ("T2_GS", t2),
        ("T3_Phase", t3),
        ("T4_IQ", t4),
        ("T5_LS", t5),
        ("T6_All5", t6),
        ("T7_Floor", t7),
        ("T8_Linear", t8),
    )
    total = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8
    ratio = total / u_v
    inv_tot = 1.0 / max(1e-12, total)
    info_eff = t4 * inv_tot
    phase_share = t3 * inv_tot
    dominant = max(named, key=lambda kv: kv[1])[0]

    if d3_margin <= 0.0:
        regime = "SCALE_BREAK"
    elif ratio < 0.85:
        regime = "COMPRESS"
    elif ratio > 1.25:
        regime = "EXPAND"
    else:
        regime = "BALANCED"

    return DynamicBalanceSnapshot(
        u_observed=u_v,
        u_projected_sum=total,
        balance_ratio=ratio,
        d3_margin=d3_margin,
        info_efficiency=info_eff,
        phase_share=phase_share,
        dominant_term=dominant,
        regime=regime,
        terms=named,
    )


# ---------------------------------------------------------------------------
# Layers 1 & 2: Enhanced Calculus (Volterra Archive) + FDS (Free-Energy Drift)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ECFDSDiagnostics:
    headroom_frac: float          # h_t = (free - reserve) / (total - reserve)
    inst_drain_rate: float        # -dh/dt (> 0 when draining toward wall)
    drain_accel: float            # d(-dh/dt)/dt
    volterra_archive: float       # discounted integral of net drain (cancels symmetric breathing!)
    sustained_drain_rate: float   # (1 - gamma) * volterra_archive (equals steady leak rate)
    fds_drift_power: float        # dF/dt = dE/dt - lambda(t) * dS/dt (free-energy rate)
    fds_t_end_steps: float        # estimated steps to wall under sustained + entropic barrier
    projected_headroom_ec: float  # 2nd-order EC-projected next headroom_frac
    at_equilibrium: bool          # |dF/dt| small over equilibrium window


class VolterraFDSMonitor:
    """Dimensionless Enhanced-Calculus (Volterra archive) + FDS (dE/dt - lambda dS/dt) tracker.

    Completely scale-invariant across any GPU VRAM capacity (4 GB .. 80+ GB) or host RAM:
    all internal quantities are normalized by `(total_bytes - reserve_bytes)` and `reserve / total`.
    """

    def __init__(
        self,
        *,
        archive_decay: float = 0.85,
        equilibrium_tol: float = 0.005,
        equilibrium_window: int = 3,
    ) -> None:
        self.archive_decay = max(0.5, min(0.98, float(archive_decay)))
        self.equilibrium_tol = max(1e-5, float(equilibrium_tol))
        self.equilibrium_window = max(1, int(equilibrium_window))

        self._prev_headroom: Optional[float] = None
        self._prev_drain_rate: float = 0.0
        self._volterra_archive: float = 0.0
        self._calm_streak: int = 0
        self._steps: int = 0

    def reset(self) -> None:
        self._prev_headroom = None
        self._prev_drain_rate = 0.0
        self._volterra_archive = 0.0
        self._calm_streak = 0
        self._steps = 0

    def update(self, *, free_bytes: int, total_bytes: int, reserve_bytes: int) -> ECFDSDiagnostics:
        self._steps += 1
        total_f = float(max(1, total_bytes))
        reserve_f = float(max(1, min(reserve_bytes, max(1, total_bytes - 1))))
        usable_f = max(1.0, total_f - reserve_f)
        rho = reserve_f / total_f  # scale-free entropic coupling lambda = reserve / total

        headroom_frac = (float(free_bytes) - reserve_f) / usable_f
        h_pos = max(1e-4, headroom_frac)

        if self._prev_headroom is None:
            inst_drain = 0.0
            accel = 0.0
            self._volterra_archive = 0.0
        else:
            inst_drain = self._prev_headroom - headroom_frac
            accel = inst_drain - self._prev_drain_rate
            # Discounted Volterra integral: symmetric +/- breathing cancels to ~0, steady leak converges
            self._volterra_archive = self.archive_decay * self._volterra_archive + inst_drain

        sustained_drain = max(0.0, self._volterra_archive * (1.0 - self.archive_decay))
        eff_inst = max(0.0, inst_drain + 0.5 * max(0.0, accel))
        eff_drain = max(eff_inst, sustained_drain)

        # FDS: E = 1 - h, S = ln(h_pos), lambda = rho
        # dF/dt = dE/dt - lambda * dS/dt = eff_drain * (1 + rho / h_pos)
        entropic_barrier = 1.0 + (rho / h_pos)
        fds_power = eff_drain * entropic_barrier

        if headroom_frac <= 0.0:
            t_end = 0.0
        elif fds_power > 1e-7:
            t_end = headroom_frac / fds_power
        else:
            t_end = float("inf")

        step_drop = max(0.0, inst_drain)
        if sustained_drain > 0.0:
            step_drop = max(step_drop, sustained_drain + 0.5 * max(0.0, accel))
        proj_headroom = headroom_frac - step_drop

        signed_fds = abs(inst_drain) * entropic_barrier
        if self._steps >= 2 and signed_fds <= self.equilibrium_tol:
            self._calm_streak += 1
        else:
            self._calm_streak = 0
        at_eq = self._calm_streak >= self.equilibrium_window

        self._prev_drain_rate = inst_drain
        self._prev_headroom = headroom_frac

        return ECFDSDiagnostics(
            headroom_frac=headroom_frac,
            inst_drain_rate=inst_drain,
            drain_accel=accel,
            volterra_archive=self._volterra_archive,
            sustained_drain_rate=sustained_drain,
            fds_drift_power=fds_power,
            fds_t_end_steps=t_end,
            projected_headroom_ec=proj_headroom,
            at_equilibrium=at_eq,
        )


# ---------------------------------------------------------------------------
# Layer 4: RSSF v0.3 Section 16 StrategyCollapse Detector
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyCollapseReport:
    loss_of_archive: bool             # KV prefix/cache hit collapse (e.g. client history truncation)
    loss_of_observer_coherence: bool  # Primary vs secondary memory sensor divergence (NVML vs WDDM)
    loss_of_scale_coherence: bool     # Physical scale violation (free > total, total jump)
    unbounded_result_drift: bool      # Sustained EC+FDS drift approaching wall
    reasons: Tuple[str, ...]

    @property
    def any_collapse(self) -> bool:
        return (
            self.loss_of_archive
            or self.loss_of_observer_coherence
            or self.loss_of_scale_coherence
            or self.unbounded_result_drift
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "any_collapse": self.any_collapse,
            "loss_of_archive": self.loss_of_archive,
            "loss_of_observer_coherence": self.loss_of_observer_coherence,
            "loss_of_scale_coherence": self.loss_of_scale_coherence,
            "unbounded_result_drift": self.unbounded_result_drift,
            "reasons": list(self.reasons),
        }


class RSSFCollapseDetector:
    """Explicit realization of RSSF v0.3 Section 16:
        StrategyCollapse = LossOfArchive v LossOfObserverCoherence v LossOfScaleCoherence v UnboundedResultDrift
    """

    def __init__(
        self,
        *,
        archive_hit_floor: float = 0.25,
        observer_divergence_frac: float = 0.15,
        drift_horizon_steps: float = 4.0,
        min_sustained_drift_frac: float = 0.004,
    ) -> None:
        self.archive_hit_floor = float(archive_hit_floor)
        self.observer_divergence_frac = float(observer_divergence_frac)
        self.drift_horizon_steps = float(drift_horizon_steps)
        self.min_sustained_drift_frac = float(min_sustained_drift_frac)
        self._baseline_total_bytes: Optional[int] = None
        self._prev_kv_hit_rate: Optional[float] = None

    def reset_session(self) -> None:
        self._prev_kv_hit_rate = None

    def evaluate(
        self,
        *,
        free_bytes: int,
        total_bytes: int,
        secondary_free_bytes: Optional[int] = None,
        kv_prefix_hit_rate: Optional[float] = None,
        had_prior_archive: bool = False,
        ec_fds: Optional[ECFDSDiagnostics] = None,
    ) -> StrategyCollapseReport:
        reasons = []

        # 1. LossOfArchive: continuing dialogue suffered sudden KV prefix wipe (e.g. client truncation)
        loss_archive = False
        if kv_prefix_hit_rate is not None:
            hr = max(0.0, min(1.0, float(kv_prefix_hit_rate)))
            if (
                (self._prev_kv_hit_rate is not None and self._prev_kv_hit_rate >= 0.70 and hr < self.archive_hit_floor)
                or (had_prior_archive and hr < self.archive_hit_floor)
            ):
                loss_archive = True
                reasons.append(f"LossOfArchive(kv_hit={hr:.2f}<{self.archive_hit_floor:.2f})")
            self._prev_kv_hit_rate = hr

        # 2. LossOfObserverCoherence: e.g. WSL2 WDDM mem_get_info (+2 GiB phantom) vs NVML physical free
        loss_observer = False
        if secondary_free_bytes is not None and total_bytes > 0:
            div = abs(int(free_bytes) - int(secondary_free_bytes)) / float(total_bytes)
            if div >= self.observer_divergence_frac:
                loss_observer = True
                reasons.append(f"LossOfObserverCoherence(sensor_div={div:.2%})")

        # 3. LossOfScaleCoherence: impossible reading or sudden hardware capacity shift
        loss_scale = False
        if total_bytes <= 0 or free_bytes < 0 or free_bytes > total_bytes:
            loss_scale = True
            reasons.append("LossOfScaleCoherence(invalid_bounds)")
        else:
            if self._baseline_total_bytes is None:
                self._baseline_total_bytes = int(total_bytes)
            elif abs(int(total_bytes) - self._baseline_total_bytes) / float(self._baseline_total_bytes) > 0.05:
                loss_scale = True
                reasons.append("LossOfScaleCoherence(total_capacity_shift)")

        # 4. UnboundedResultDrift: sustained Volterra archive drain with FDS horizon <= drift_horizon_steps
        unbounded_drift = False
        if ec_fds is not None:
            if (
                ec_fds.fds_t_end_steps <= self.drift_horizon_steps
                and ec_fds.sustained_drain_rate >= self.min_sustained_drift_frac
            ):
                unbounded_drift = True
                reasons.append(
                    f"UnboundedResultDrift(T_end={ec_fds.fds_t_end_steps:.2f},v={ec_fds.sustained_drain_rate:.4f})"
                )

        return StrategyCollapseReport(
            loss_of_archive=loss_archive,
            loss_of_observer_coherence=loss_observer,
            loss_of_scale_coherence=loss_scale,
            unbounded_result_drift=unbounded_drift,
            reasons=tuple(reasons),
        )


# ---------------------------------------------------------------------------
# Scale-Specific Balance Adapters (Same Formula, Different Scale of `I`!)
# ---------------------------------------------------------------------------
def observe_hardware_load_balance(
    *,
    used_frac: float,
    headroom_frac: float,
    load_frac: float,
    calm_frac: float,
    drop_frac: float,
    load_rise: float,
    sustained_drain: float,
    kv_prefix_hit_rate: Optional[float] = None,
    sensor_divergence: float = 0.0,
) -> DynamicBalanceSnapshot:
    """Device scale (M1): I = archive/cache continuity, G = memory occupancy, Q = coupled surge."""
    u_live = max(0.35, 0.7 + 0.6 * used_frac + 0.5 * load_frac)
    g_v = max(0.05, used_frac)
    # Impulse Q stays bounded under ordinary single-factor jumps so 2D L4 governs normal load,
    # and breaches Term-3 D3 specifically on genuine multi-factor cascades (drop + archive/sensor loss).
    q_v = (
        0.10
        + 2.8 * max(0.0, drop_frac)
        + 1.4 * max(0.0, load_rise)
        + 1.0 * max(0.0, load_frac - 0.5) * max(0.0, 1.0 - headroom_frac)
    )
    s_v = min(2.0, 0.5 * load_frac + 0.5 * max(0.0, 1.0 - headroom_frac) + 0.12 / max(0.05, headroom_frac))
    i_v = (
        max(0.01, min(1.0, float(kv_prefix_hit_rate)))
        if kv_prefix_hit_rate is not None
        else max(0.05, 1.0 - 8.0 * max(0.0, sustained_drain))
    )
    l_v = max(0.02, 1.0 - 3.5 * max(0.0, sensor_divergence) - 0.4 * max(0.0, load_rise))
    c_l = max(0.30, 0.6 + 0.7 * calm_frac + 0.6 * max(0.0, headroom_frac))
    return observe_dynamic_balance(U=u_live, G=g_v, Q=q_v, S=s_v, I=i_v, L=l_v, C_L=c_l)


def observe_speculation_step_balance(
    *,
    depth_ratio: float,
    mean_prob: float,
    accept_ema: float,
    logp_gap: float,
    state_budget_ratio: float = 0.3,
) -> DynamicBalanceSnapshot:
    """Token/Speculation scale (M7): I = draft confidence & acceptance archive, Q = speculative depth."""
    i_v = max(0.02, min(1.0, 0.5 * mean_prob + 0.5 * accept_ema))
    l_v = max(0.02, min(1.0, math.exp(-max(0.0, logp_gap))))
    g_v = max(0.05, min(1.5, 0.25 + 0.6 * depth_ratio + 0.4 * state_budget_ratio))
    q_v = max(0.05, 0.3 + 2.2 * depth_ratio * (1.0 - 0.5 * accept_ema) + 0.8 * max(0.0, logp_gap))
    s_v = max(0.05, min(2.0, (1.0 - mean_prob) + 0.5 * (1.0 - l_v)))
    c_l = max(0.4, 1.3 - 0.5 * state_budget_ratio)
    u_live = max(0.5, 0.9 + 0.6 * depth_ratio)
    return observe_dynamic_balance(U=u_live, G=g_v, Q=q_v, S=s_v, I=i_v, L=l_v, C_L=c_l)


def observe_q1_cube_balance(
    *,
    info_density: float,
    weight_mass_frac: float,
    input_impulse: float,
    residual_entropy: float,
    output_coherence: float = 0.9,
    bandwidth_factor: float = 1.2,
) -> DynamicBalanceSnapshot:
    """Micro weight scale (M0 / M5b Q1-Cube): I = bit-plane information significance of one weight/expert."""
    u_live = max(0.5, 1.0 + 0.5 * info_density + 0.3 * input_impulse)
    return observe_dynamic_balance(
        U=u_live,
        G=max(0.05, weight_mass_frac),
        Q=max(0.02, input_impulse),
        S=max(0.02, residual_entropy),
        I=max(0.01, info_density),
        L=max(0.05, output_coherence),
        C_L=max(0.3, bandwidth_factor),
    )


def observe_kv_placement_balance(
    *,
    n_gpu: int,
    n_full: int,
    kv_gpu_bytes: int,
    staging_bytes: int,
    avail_vram_bytes: int,
    total_vram_bytes: int,
    reserve_bytes: int,
    workspace_bytes: int = 0,
    pcie_gb_s: Optional[float] = None,
    is_decode: bool = False,
) -> DynamicBalanceSnapshot:
    """KV-archive placement scale (`kv_offload.py`): I = fraction of attention archive kept in fast GDDR6.

    Completely scale-invariant across 4 GB .. 80+ GB GPUs: balances fast residency `I` against
    post-placement occupancy `G`, transient allocation impulse `Q`, offload + wall entropy `S`,
    headroom coherence `L`, and channel capacity `C_L`.
    """
    nf = max(1, int(n_full))
    ng = max(0, min(nf, int(n_gpu)))
    tot = float(max(1, int(total_vram_bytes)))
    avail = float(max(0, int(avail_vram_bytes)))
    res = float(max(1, min(int(reserve_bytes), max(1, int(total_vram_bytes) - 1))))
    add_bytes = float(max(0, int(kv_gpu_bytes)) + max(0, int(staging_bytes)) + max(0, int(workspace_bytes)))

    post_free = avail - add_bytes
    usable = max(1.0, tot - res)
    post_headroom = (post_free - res) / usable
    used_after = max(0.05, min(1.0, 1.0 - max(0.0, post_free) / tot))

    gpu_frac = float(ng) / float(nf)
    offload_frac = 1.0 - gpu_frac
    alloc_ratio = add_bytes / max(1.0, avail)

    # Impulse Q reflects the allocation burst relative to available VRAM; stays bounded whenever
    # post_free >= reserve and breaches Term-3 D3 when allocation crowds into/past the reserve wall.
    wall_deficit = max(0.0, (res - post_free) / max(1.0, res))
    q_v = (
        (0.08 if is_decode else 0.14)
        + (0.95 if is_decode else 1.35) * min(2.0, alloc_ratio)
        + 2.8 * wall_deficit
    )
    s_v = min(
        2.5,
        0.15
        + 0.70 * offload_frac
        + 0.45 * used_after
        + (0.08 / max(0.04, max(0.0, post_headroom) + 0.04)),
    )
    i_v = max(0.05, 0.10 + 0.90 * gpu_frac)
    l_v = max(0.05, min(1.0, 1.0 - 0.85 * wall_deficit))
    pcie_norm = (float(pcie_gb_s) / 16.0) if (pcie_gb_s and pcie_gb_s > 0) else 1.0
    c_l = max(0.35, min(2.2, 0.55 + 0.45 * pcie_norm + 0.85 * max(0.0, post_headroom)))
    u_live = max(0.55, 0.75 + 0.55 * used_after + 0.30 * gpu_frac)
    return observe_dynamic_balance(U=u_live, G=used_after, Q=q_v, S=s_v, I=i_v, L=l_v, C_L=c_l)


def observe_region_breathing_balance(
    *,
    occ_frac: float,
    reuse_rate: float,
    reg_frac: float,
    sustained_drain: float,
    warm_fill_frac: float = 0.0,
) -> DynamicBalanceSnapshot:
    """Breathing pool scale (M2 `MirrorRegionBreathingPool`): I = live region reuse archive."""
    occ = max(0.0, min(1.0, float(occ_frac)))
    reuse = max(0.0, min(1.0, float(reuse_rate)))
    rf = max(0.0, min(1.0, float(reg_frac)))
    sd = max(0.0, float(sustained_drain))
    wf = max(0.0, min(1.0, float(warm_fill_frac)))

    i_v = max(0.04, 0.08 + 0.92 * reuse)
    g_v = max(0.05, occ)
    q_v = 0.10 + 2.2 * rf + 2.4 * sd + 1.4 * max(0.0, occ + rf - 0.85)
    s_v = min(2.2, 0.15 + 0.65 * occ + 0.45 * wf + 0.08 / max(0.05, 1.0 - occ))
    l_v = max(0.05, 1.0 - 3.5 * sd)
    c_l = max(0.35, 1.25 - 0.65 * occ)
    u_live = max(0.50, 0.75 + 0.50 * occ + 0.30 * reuse)
    return observe_dynamic_balance(U=u_live, G=g_v, Q=q_v, S=s_v, I=i_v, L=l_v, C_L=c_l)


def observe_warmup_fds_balance(
    *,
    step_ms: float,
    ema_ms: float,
    prev_ms: float,
    done_frac: float,
    fds_power: float = 0.0,
) -> DynamicBalanceSnapshot:
    """Cold startup warmup scale (M3 `MirrorColdWarmupScale`): I = latency convergence archive."""
    dt = max(1e-4, float(step_ms))
    ema = max(1e-4, float(ema_ms))
    prev = max(1e-4, float(prev_ms))
    done = max(0.0, min(1.0, float(done_frac)))

    rel_ema = abs(dt - ema) / max(dt, ema)
    rel_step = abs(dt - prev) / max(dt, prev)
    i_v = max(0.05, 1.0 / (1.0 + 2.0 * rel_ema))
    l_v = max(0.05, min(dt, ema) / max(dt, ema))
    g_v = max(0.05, 0.20 + 0.70 * done)
    q_v = 0.10 + 2.2 * rel_step + 0.8 * max(0.0, float(fds_power))
    s_v = max(0.05, min(2.0, 0.15 + rel_ema + max(0.0, float(fds_power))))
    c_l = max(0.45, 0.90 + 0.50 * done)
    u_live = max(0.55, 0.85 + 0.35 * done + 0.30 * rel_step)
    return observe_dynamic_balance(U=u_live, G=g_v, Q=q_v, S=s_v, I=i_v, L=l_v, C_L=c_l)


def observe_post_prefill_region_balance(
    *,
    cur_cov: float,
    cur_scar: float,
    hit_rate: float,
    miss_streak: int = 0,
    scar_ratio: float = 0.0,
) -> DynamicBalanceSnapshot:
    """Post-prefill known-region warmup scale (M4): I = prefill routing archive coverage."""
    cov = max(0.0, min(1.0, float(cur_cov)))
    scar = max(0.0, min(1.0, float(cur_scar)))
    hr = max(0.0, min(1.0, float(hit_rate)))
    ms = max(0, int(miss_streak))
    sr = max(0.0, min(1.0, float(scar_ratio)))

    i_v = max(0.04, 0.08 + 0.92 * cov)
    l_v = max(0.05, 0.10 + 0.90 * hr)
    g_v = max(0.05, 0.20 + 0.65 * sr + 0.25 * scar)
    q_v = 0.12 + 1.55 * scar + 0.45 * min(4, ms)
    s_v = max(0.05, min(2.0, 0.15 + 0.85 * scar + 0.35 * (1.0 - hr)))
    c_l = max(0.35, 0.75 + 0.65 * hr)
    u_live = max(0.55, 0.80 + 0.40 * cov + 0.35 * scar)
    return observe_dynamic_balance(U=u_live, G=g_v, Q=q_v, S=s_v, I=i_v, L=l_v, C_L=c_l)


def observe_probability_path_balance(
    *,
    top_mass: float,
    entropy: float,
    max_entropy: float,
    late_surge: float,
    checkpoint_agreement: float,
    alive_frac: float,
) -> DynamicBalanceSnapshot:
    """Depth probability-path scale (M6 `SurvivorProbabilityTracker`): I = survivor probability concentration."""
    tm = max(0.0, min(1.0, float(top_mass)))
    h_norm = max(0.0, min(2.0, float(entropy) / max(1.0, float(max_entropy))))
    surge = max(0.0, min(4.0, float(late_surge)))
    agree = max(0.0, min(1.0, float(checkpoint_agreement)))
    af = max(0.0, min(1.0, float(alive_frac)))

    i_v = max(0.04, 0.10 + 0.90 * tm)
    l_v = max(0.05, 0.10 + 0.90 * agree)
    g_v = max(0.05, 0.15 + 0.75 * (1.0 - af))
    q_v = 0.12 + 0.95 * surge + 0.45 * h_norm
    s_v = max(0.05, min(2.2, 0.15 + 0.95 * h_norm))
    c_l = max(0.40, 0.80 + 0.60 * agree)
    u_live = max(0.55, 0.85 + 0.40 * tm + 0.30 * h_norm)
    return observe_dynamic_balance(U=u_live, G=g_v, Q=q_v, S=s_v, I=i_v, L=l_v, C_L=c_l)


def observe_thought_balance_tensor(
    *,
    tau_step: Any,
    entropy: Any,
    g_h: Any,
    branch_tie: Any,
    step_index: int,
    a: float = 0.25,
    b: float = 0.25,
    c: float = 0.50,
) -> Tuple[Any, Any, Any]:
    """Pure-tensor zero-sync 8-pair force balance for M8 (`InitialProbabilityAnchor`).

    Executes 100% on the tensor's native device (GPU or CPU) with ZERO `.item()` / CPU syncs!
    Returns `(d3_margin, info_eff_t4, balance_ratio)` tensors broadcast to `entropy.shape` (`[M, 1]`).
    """
    import torch

    tau = tau_step.float().clamp(0.0, 1.0)
    ent = entropy.float().clamp_min(0.0)
    gh = g_h.float().clamp(0.0, 1.0)
    btie = branch_tie.float().clamp(0.0, 1.0) if isinstance(branch_tie, torch.Tensor) else torch.full_like(ent, float(branch_tie))

    i_v = tau.clamp_min(1e-4)
    s_v = gh.clamp_min(0.02)
    l_v = btie.clamp(0.05, 1.0)
    g_scalar = min(1.0, 0.20 + float(max(0, int(step_index))) / float(max(0, int(step_index)) + 64))
    g_v = torch.full_like(ent, g_scalar)
    q_v = (0.10 + 1.25 * gh * (1.0 - tau) + 0.45 * (1.0 - btie)).clamp_min(0.05)
    c_l = (0.85 + 0.55 * btie + 0.30 * (1.0 - gh)).clamp_min(0.35)
    u_v = (0.80 + 0.45 * gh + 0.35 * tau).clamp_min(0.50)

    c2 = c * c
    c4 = c2 * c2
    il_ab = i_v + l_v + (a * b)
    quad_pad = (il_ab + c2).pow(2)

    d12 = (c_l * u_v + i_v + l_v).clamp_min(1e-9)
    t1 = (g_v * q_v) / d12
    t2 = (g_v * s_v) / d12

    d3_margin = c_l * u_v - q_v - c4 + (g_v + b * b) * (s_v + a * a) + quad_pad
    t3 = (g_v * (i_v + l_v + q_v + s_v)) / d3_margin.clamp_min(1e-4)

    t4 = (i_v * q_v) / (c_l * u_v + g_v + s_v).clamp_min(1e-9)
    t5 = (l_v * s_v) / (c_l * u_v + g_v + q_v).clamp_min(1e-9)

    geom5 = (g_v * i_v * l_v * q_v * s_v).clamp_min(1e-12).pow(0.2)
    t6 = ((g_v + i_v) * (l_v + q_v + s_v)) / (c_l * u_v + geom5).clamp_min(1e-9)

    t7 = ((g_v + s_v) * (a * a + b * b)) / (q_v + c4 + quad_pad).clamp_min(1e-9)
    t8 = ((g_v + q_v + s_v) * (a + b + c)) / (c_l + i_v + l_v + u_v).clamp_min(1e-9)

    total = (t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8).clamp_min(1e-12)
    info_eff_t4 = t4 / total
    balance_ratio = total / u_v
    return d3_margin, info_eff_t4, balance_ratio


# ---------------------------------------------------------------------------
# End-to-End Generation Lifecycle Balance Tracker (Entry -> Prefill -> Decode -> Turn End)
# ---------------------------------------------------------------------------
class GenerationBalanceTracker:
    """Tracks Anton's 4-layer balance across a whole generation turn (Entry -> Prefill -> Decode -> Exit).

    Zero overhead in the hot token decode loop: invoked on turn entry, once per prefill chunk
    (steering adaptive chunk damping via EC+FDS + Term-3 scale break!), and on turn completion.
    """

    def __init__(self, *, fds_chunk_warn_steps: float = 3.0) -> None:
        self.fds_chunk_warn_steps = float(fds_chunk_warn_steps)
        self._prefill_mon = VolterraFDSMonitor(archive_decay=0.80)
        self._collapse_det = RSSFCollapseDetector(drift_horizon_steps=fds_chunk_warn_steps)
        self._entry_balance: Optional[DynamicBalanceSnapshot] = None
        self._entry_collapse: Optional[StrategyCollapseReport] = None
        self._turn_end_balance: Optional[DynamicBalanceSnapshot] = None
        self._prefill_chunks: int = 0
        self._prefill_damped_chunks: int = 0
        self._prefill_scale_breaks: int = 0
        self._prefill_min_t_end: float = float("inf")
        self._last_prefill_balance: Optional[DynamicBalanceSnapshot] = None
        self._last_prefill_collapse: Optional[StrategyCollapseReport] = None

    def reset_session(self) -> None:
        self._prefill_mon.reset()
        self._collapse_det.reset_session()

    def begin_turn(
        self,
        *,
        prompt_tokens: int,
        cached_prefix_tokens: int,
        had_prior_archive: bool,
        free_bytes: int,
        total_bytes: int,
        reserve_bytes: int,
        secondary_free_bytes: Optional[int] = None,
        pcie_gb_s: Optional[float] = None,
        offload_ratio: float = 0.0,
    ) -> Tuple[DynamicBalanceSnapshot, StrategyCollapseReport]:
        """Evaluate meso-scale balance at turn entry (I = KV prefix archive continuity)."""
        self._prefill_mon.reset()
        self._prefill_chunks = 0
        self._prefill_damped_chunks = 0
        self._prefill_scale_breaks = 0
        self._prefill_min_t_end = float("inf")
        self._last_prefill_balance = None
        self._last_prefill_collapse = None
        self._turn_end_balance = None

        n_tok = max(1, int(prompt_tokens))
        kv_hit = max(0.0, min(1.0, float(cached_prefix_tokens) / float(n_tok)))
        new_ratio = max(0.0, min(1.0, float(n_tok - cached_prefix_tokens) / float(n_tok)))
        tot = max(1, int(total_bytes))
        used_frac = max(0.05, min(1.0, 1.0 - float(max(0, free_bytes)) / float(tot)))
        div = (
            abs(int(free_bytes) - int(secondary_free_bytes)) / float(tot)
            if secondary_free_bytes is not None
            else 0.0
        )

        # Seed EC/FDS monitor with turn-entry reading
        ec_diag = self._prefill_mon.update(
            free_bytes=max(0, free_bytes), total_bytes=tot, reserve_bytes=max(1, reserve_bytes)
        )
        collapse = self._collapse_det.evaluate(
            free_bytes=max(0, free_bytes),
            total_bytes=tot,
            secondary_free_bytes=secondary_free_bytes,
            kv_prefix_hit_rate=kv_hit,
            had_prior_archive=had_prior_archive,
            ec_fds=ec_diag,
        )

        # Scale-free channel capacity C_L from calibrated PCIe or headroom
        c_l = max(0.4, min(2.0, (float(pcie_gb_s) / 16.0) if (pcie_gb_s and pcie_gb_s > 0) else 1.1))
        # Impulse Q weights new uncached tail smoothly (tiny 10-tok prompts never fake a 20K prefill shock!)
        new_toks = max(0, n_tok - int(cached_prefix_tokens))
        tail_weight = float(new_toks) / float(new_toks + 512)
        free_frac = max(0.10, 1.0 - used_frac)
        q_val = (
            0.15
            + 1.0 * new_ratio * tail_weight
            + 0.45 * (new_ratio * tail_weight / free_frac)
            + (1.6 if collapse.loss_of_archive else 0.0)
        )
        s_val = max(0.05, 0.25 + 0.65 * max(0.0, offload_ratio) + 0.4 * used_frac)
        i_val = max(0.02, kv_hit if had_prior_archive else max(0.5, kv_hit))
        l_val = max(0.05, 1.0 - 3.0 * div)
        u_live = max(0.6, 0.8 + 0.6 * used_frac + 0.4 * new_ratio * tail_weight)

        bal = observe_dynamic_balance(U=u_live, G=used_frac, Q=q_val, S=s_val, I=i_val, L=l_val, C_L=c_l)
        self._entry_balance = bal
        self._entry_collapse = collapse
        return bal, collapse

    def advise_prefill_chunk(
        self,
        *,
        base_chunk: int,
        remaining_tokens: int,
        free_bytes: int,
        total_bytes: int,
        reserve_bytes: int,
        secondary_free_bytes: Optional[int] = None,
    ) -> Tuple[int, DynamicBalanceSnapshot, ECFDSDiagnostics]:
        """Adaptive scale-invariant prefill chunking steered by EC Volterra + FDS T_end + Term-3 D3."""
        self._prefill_chunks += 1
        tot = max(1, int(total_bytes))
        res = max(1, min(int(reserve_bytes), tot - 1))
        ec = self._prefill_mon.update(free_bytes=max(0, free_bytes), total_bytes=tot, reserve_bytes=res)
        if ec.fds_t_end_steps < self._prefill_min_t_end:
            self._prefill_min_t_end = ec.fds_t_end_steps

        col = self._collapse_det.evaluate(
            free_bytes=max(0, free_bytes),
            total_bytes=tot,
            secondary_free_bytes=secondary_free_bytes,
            ec_fds=ec,
        )
        used_frac = max(0.05, min(1.0, 1.0 - float(max(0, free_bytes)) / float(tot)))
        h_frac = max(1e-4, ec.headroom_frac)
        div = (
            abs(int(free_bytes) - int(secondary_free_bytes)) / float(tot)
            if secondary_free_bytes is not None
            else 0.0
        )
        bal = observe_hardware_load_balance(
            used_frac=used_frac,
            headroom_frac=h_frac,
            load_frac=used_frac,
            calm_frac=max(1e-4, 1.0 - used_frac),
            drop_frac=max(0.0, ec.inst_drain_rate),
            load_rise=max(0.0, ec.inst_drain_rate),
            sustained_drain=ec.sustained_drain_rate,
            sensor_divergence=div,
        )
        self._last_prefill_balance = bal
        self._last_prefill_collapse = col

        advised = max(1, int(base_chunk))
        if bal.regime == "SCALE_BREAK":
            self._prefill_scale_breaks += 1
            advised = min(advised, max(128, advised // 4))
            self._prefill_damped_chunks += 1
        elif (
            self._prefill_chunks >= 3
            and col.unbounded_result_drift
            and (ec.headroom_frac < 0.30 or (ec.fds_t_end_steps <= 1.75 and ec.headroom_frac < 0.45))
        ):
            # Sustained Volterra drain genuinely approaching reserve (not one-time initial KV allocation
            # with ample free headroom!) -> halve prefill chunk smoothly before hitting the wall.
            advised = min(advised, max(128, advised // 2))
            self._prefill_damped_chunks += 1

        advised = min(advised, max(1, int(remaining_tokens)))
        return advised, bal, ec

    def finish_turn(
        self,
        *,
        prompt_tokens: int,
        cached_prefix_tokens: int,
        generated_tokens: int,
        free_bytes: int,
        total_bytes: int,
        kv_offloaded_layers: int,
        kv_gpu_layers: int,
        decode_tok_per_s: float,
        calibrated_decode_tok_per_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Close the turn with a unified multi-scale balance summary for telemetry & HUD."""
        tot = max(1, int(total_bytes))
        used_frac = max(0.05, min(1.0, 1.0 - float(max(0, free_bytes)) / float(tot)))
        tot_kv = max(1, int(kv_offloaded_layers) + int(kv_gpu_layers))
        offload_frac = float(max(0, kv_offloaded_layers)) / float(tot_kv)
        kv_hit = max(0.0, min(1.0, float(cached_prefix_tokens) / float(max(1, prompt_tokens))))
        gen_frac = min(1.5, float(max(1, generated_tokens)) / 512.0)
        speed_ratio = (
            max(0.2, min(2.0, float(decode_tok_per_s) / float(calibrated_decode_tok_per_s)))
            if (calibrated_decode_tok_per_s and calibrated_decode_tok_per_s > 0.1 and decode_tok_per_s > 0.0)
            else 1.0
        )

        bal = observe_dynamic_balance(
            U=max(0.6, 0.8 + 0.5 * used_frac + 0.4 * offload_frac),
            G=used_frac,
            Q=0.15 + 0.6 * gen_frac,
            S=max(0.05, 0.15 + 0.7 * offload_frac + 0.25 * (1.0 - min(1.0, speed_ratio))),
            I=max(0.05, 0.5 * kv_hit + 0.5 * min(1.0, speed_ratio)),
            L=max(0.1, min(1.0, speed_ratio)),
            C_L=max(0.4, speed_ratio),
        )
        self._turn_end_balance = bal

        return {
            "entry_balance": self._entry_balance.as_dict() if self._entry_balance is not None else None,
            "entry_collapse": self._entry_collapse.as_dict() if self._entry_collapse is not None else None,
            "prefill_chunks_observed": self._prefill_chunks,
            "prefill_damped_chunks": self._prefill_damped_chunks,
            "prefill_scale_breaks": self._prefill_scale_breaks,
            "prefill_min_fds_t_end_steps": (
                round(self._prefill_min_t_end, 3) if math.isfinite(self._prefill_min_t_end) else None
            ),
            "last_prefill_balance": (
                self._last_prefill_balance.as_dict() if self._last_prefill_balance is not None else None
            ),
            "turn_end_balance": bal.as_dict(),
        }
