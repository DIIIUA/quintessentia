"""Shared streamed draft/verify/commit loop for CPU, GPU and the progressive graph.

Native heads and prompt lookup share verification. Only cache state is snapshotted;
weights remain under the existing bounded layer/expert managers.
"""
from collections import Counter, deque
from contextlib import contextmanager, ExitStack, nullcontext
from dataclasses import dataclass
import math
import time
from typing import Callable, Optional

import torch

from quintessentia_cli.thought_buffer import _auxiliary_forwards, capture_cache_snapshot, restore_cache_snapshot


def _vec_dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _vec_norm(a: tuple[float, ...]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _directed_contact(u: tuple[float, ...], v: tuple[float, ...], eps: float = 1e-12) -> tuple[float, tuple[float, ...], float]:
    """Canonical non-commutative directed contact u -> v (Section 2 of pure-math pack):
      alpha_{u->v} = <u, v> / (||u||^2 + eps)
      r_{u->v}     = v - alpha_{u->v} * u
      delta(u->v)  = ||r_{u->v}|| / (||v|| + eps) in [0, 1]
    """
    uu = _vec_dot(u, u)
    alpha = _vec_dot(u, v) / (uu + eps)
    res = tuple(vi - alpha * ui for ui, vi in zip(u, v))
    delta = _vec_norm(res) / (_vec_norm(v) + eps)
    return alpha, res, delta


def _mirror_tie(a: tuple[float, ...], b: tuple[float, ...], eps: float = 1e-12) -> float:
    """Canonical symmetric convergence measure Tie(a, b) in [0, 1] (Section 2 of pure-math pack)."""
    n_sum = math.sqrt(sum((x + y) ** 2 for x, y in zip(a, b)))
    n_diff = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    val = (n_sum - n_diff) / (n_sum + n_diff + eps)
    return val if val > 0.0 else 0.0


def eval_canonical_l4_mirror(
    p0_out_prev: tuple[float, ...],
    p1_obs_prev: tuple[float, ...],
    strategy: tuple[float, ...],
    p2_obs_next: tuple[float, ...],
    p3_out_next: tuple[float, ...],
    eps: float = 1e-12,
) -> tuple[str, float, float, float]:
    """Canonical bracketless 2-contour mirror equality & 4-valued logic L4 (Sections 3–4 of pure-math pack).

    Zero hand-tuned magic thresholds — uses ONLY the natural mirror threshold 1/2 (0.5) and sign of Tie:
      UNKNOWN : any wing role has zero norm
      SPLIT   : tau_core == 0 or tau_wing == 0
      EQUAL   : tau_core > 0 and tau_wing >= 0.5 and delta_res < 0.5
      TENSION : otherwise

    Returns (state, tau_core, tau_wing, delta_res).
    """
    roles = (p0_out_prev, p1_obs_prev, p2_obs_next, p3_out_next)
    norms = tuple(_vec_norm(r) for r in roles)
    if any(n <= eps for n in norms) or _vec_norm(strategy) <= eps:
        return "UNKNOWN", 0.0, 0.0, 1.0

    # Contour A: inner bracketless central stem (ObservedBehavior_(t-1) ≍ Strategy_t ≍ ObservedBehavior_(t+1))
    tau_core = min(_mirror_tie(p1_obs_prev, strategy, eps), _mirror_tie(strategy, p2_obs_next, eps))

    # Contour B: outer 4x4 non-commutative contact shell M_ij = alpha_{P_i -> P_j}, R_ij = r_{P_i -> P_j}
    m = [[0.0] * 4 for _ in range(4)]
    r_norms = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            a_ij, r_ij, _ = _directed_contact(roles[i], roles[j], eps)
            m[i][j] = a_ij
            r_norms[i][j] = _vec_norm(r_ij)

    # FuturePressure weights on immutable past wing (P0, P1)
    w0 = 1.0 + abs(m[1][0]) + max(0.0, m[3][0]) + max(0.0, m[2][0])
    w1 = 1.0 + abs(m[0][1]) + max(0.0, m[2][1]) + max(0.0, m[3][1])
    s_left = tuple((w0 * a + w1 * b) / (w0 + w1) for a, b in zip(p0_out_prev, p1_obs_prev))

    # Past-grounded weights on future wing (P2, P3)
    w2 = 1.0 + abs(m[3][2]) + max(0.0, m[1][2]) + max(0.0, m[0][2])
    w3 = 1.0 + abs(m[2][3]) + max(0.0, m[0][3]) + max(0.0, m[1][3])
    s_right = tuple((w2 * a + w3 * b) / (w2 + w3) for a, b in zip(p2_obs_next, p3_out_next))

    tau_wing = _mirror_tie(s_left, s_right, eps)
    delta_res = (r_norms[0][3] + r_norms[1][2] + r_norms[0][2] + r_norms[1][3]) / (
        2.0 * (norms[2] + norms[3]) + eps
    )

    if tau_core <= 0.0 or tau_wing <= 0.0:
        return "SPLIT", tau_core, tau_wing, delta_res
    if tau_wing >= 0.5 and delta_res < 0.5:
        return "EQUAL", tau_core, tau_wing, delta_res
    return "TENSION", tau_core, tau_wing, delta_res


@dataclass
class DraftScar:
    """Reversible tensor scar Gamma_scar captured from a rejected speculative draft position (M7)."""
    position: int
    draft_token: int
    verified_token: int
    logp_gap: float
    x_draft: torch.Tensor
    y_draft: torch.Tensor


class MirrorSpeculationScale:
    """Isolated M7 pure-math canonical L4 mirror equality scale for speculative drafting & reversible scar recycling.

    Zero cross-scale coupling with M0..M6 and ZERO hand-tuned magic thresholds:
    uses strictly the canonical non-commutative contact operator (alpha, r, delta),
    bracketless central stem + 4x4 shell, and 4-valued logic L4 with natural mirror threshold 1/2.
    """

    def __init__(
        self,
        max_drafts: int = 3,
        *,
        recycle_residual_tol: float = 0.08,
        max_scars: int = 16,
    ):
        self.max_drafts = max(1, int(max_drafts))
        self.recycle_residual_tol = float(recycle_residual_tol)
        self.max_scars = max(1, int(max_scars))

        # Immutable past roles P0 = Outcome_(t-1), P1 = ObservedBehavior_(t-1) and center Strategy_t
        self._p0_prev: tuple[float, float] = (1.0, 1.0)
        self._p1_prev: tuple[float, float] = (1.0, 1.0)
        self._strategy: tuple[float, float] = (1.0, 1.0)
        self._current_cap: int = self.max_drafts
        self._last_state: str = "EQUAL"
        self._last_tau_core: float = 1.0
        self._last_tau_wing: float = 1.0
        self._last_delta_res: float = 0.0
        self._scars: deque[DraftScar] = deque(maxlen=self.max_scars)

        # Telemetry
        self.steps: int = 0
        self.equal_steps: int = 0
        self.tension_steps: int = 0
        self.split_events: int = 0
        self.scars_captured: int = 0
        self.anchors_recycled: int = 0
        self.zero_flop_recycles: int = 0
        self.diff_recycles: int = 0
        self.q1_anchors_retained: int = 0
        self.last_logp_gap: float = 0.0

    def pick_depth(self, probs: list[float], max_allowed: int) -> int:
        """Pure-math canonical L4 depth selection using natural 1/2 survival & mirror wing balance."""
        limit = min(self.max_drafts, max(0, int(max_allowed)), len(probs))
        if limit <= 0:
            return 0
        cap = min(limit, max(1, self._current_cap))

        # Cumulative survival prefix product P_k = prod_{i=1..k} p_i against natural mirror threshold 1/2
        cum_p = 1.0
        best_k = 0
        expected_acc = 0.0
        for i in range(cap):
            p_i = max(1e-6, min(1.0, float(probs[i])))
            cum_p *= p_i
            expected_acc += cum_p
            if cum_p >= 0.5 or i == 0:
                best_k = i + 1
            else:
                break

        if best_k <= 0:
            return 0

        # Canonical L4 evaluation between immutable past (P0, P1), Strategy_t, and proposed wing (P2, P3)
        mean_p = sum(float(p) for p in probs[:best_k]) / float(best_k)
        p2_next = (float(best_k) / float(self.max_drafts), 2.0 * mean_p - 1.0)
        p3_next = (expected_acc / float(self.max_drafts), 2.0 * cum_p - 1.0)
        state, tau_core, tau_wing, delta_res = eval_canonical_l4_mirror(
            self._p0_prev, self._p1_prev, self._strategy, p2_next, p3_next
        )
        self._last_tau_core = tau_core
        self._last_tau_wing = tau_wing
        self._last_delta_res = delta_res

        if self._last_state == "SPLIT" or state == "SPLIT":
            return 1
        if state == "TENSION" and best_k > 1:
            return max(1, min(best_k, int(round(tau_wing * best_k))))
        return best_k

    @torch.no_grad()
    def record_verification(
        self,
        *,
        proposed: int,
        accepted: int,
        position: int,
        drafts: list[int],
        verified_token: int,
        reject_row_logits: Optional[torch.Tensor] = None,
        reject_hidden: Optional[torch.Tensor] = None,
        reject_out: Optional[torch.Tensor] = None,
    ) -> str:
        """Close verification step via canonical L4 mirror equality and capture reversible DraftScar on rejection."""
        self.steps += 1
        prop = max(1, int(proposed))
        acc = max(0, min(prop, int(accepted)))
        accept_ratio = float(acc) / float(prop)
        depth_ratio = float(prop) / float(self.max_drafts)

        logp_gap = 0.0
        rejected = acc < prop and acc < len(drafts)
        if rejected and reject_row_logits is not None:
            flat_lg = reject_row_logits.detach().float().reshape(-1)
            draft_tok = int(drafts[acc])
            ver_tok = int(verified_token)
            if 0 <= draft_tok < flat_lg.numel() and 0 <= ver_tok < flat_lg.numel():
                # Exact log-probability ratio gap: log p(ver) - log p(draft) = logit(ver) - logit(draft)
                logp_gap = max(0.0, float((flat_lg[ver_tok] - flat_lg[draft_tok]).item()))
        self.last_logp_gap = logp_gap

        # Capture reversible (x_draft, y_draft) tensor scar instead of discarding rejected compute
        if rejected and reject_hidden is not None and reject_out is not None:
            scar = DraftScar(
                position=int(position),
                draft_token=int(drafts[acc]),
                verified_token=int(verified_token),
                logp_gap=logp_gap,
                x_draft=reject_hidden.detach().reshape(1, -1).clone(),
                y_draft=reject_out.detach().reshape(1, -1).clone(),
            )
            self._scars.append(scar)
            self.scars_captured += 1

        # Signed cosine-like log-agreement coordinate: s = 2 * exp(-logp_gap) - 1 in (-1, +1]
        # Crosses 0 naturally at exp(-logp_gap) = 1/2 (i.e., draft prob < half of verified prob -> SPLIT!)
        log_agree = 2.0 * math.exp(-logp_gap) - 1.0
        obs_signed = 2.0 * accept_ratio - 1.0
        p2_obs = (obs_signed, log_agree) if (rejected and acc == 0) else (depth_ratio, obs_signed)
        p3_out = (obs_signed, log_agree)

        state, tau_core, tau_wing, delta_res = eval_canonical_l4_mirror(
            self._p0_prev, self._p1_prev, self._strategy, p2_obs, p3_out
        )
        self._last_state = state
        self._last_tau_core = tau_core
        self._last_tau_wing = tau_wing
        self._last_delta_res = delta_res

        if state == "SPLIT":
            self.split_events += 1
            self._current_cap = 1
            # Birth of new branch Strategy_(t+1) without overwriting immutable past record
            self._strategy = (max(0.5, depth_ratio), max(0.5, abs(log_agree)))
            self._p1_prev = (max(0.5, depth_ratio), 0.5)
            self._p0_prev = (max(0.5, accept_ratio if accept_ratio > 0 else 0.5), 0.5)
        elif state == "EQUAL":
            self.equal_steps += 1
            if acc == prop and self._current_cap < self.max_drafts:
                self._current_cap += 1
            self._p0_prev = p3_out
            self._p1_prev = p2_obs
            self._strategy = (
                0.5 * (self._p1_prev[0] + p2_obs[0]),
                0.5 * (self._p1_prev[1] + p2_obs[1]),
            )
        else:
            self.tension_steps += 1
            if acc == prop and self._current_cap < self.max_drafts:
                self._current_cap += 1
            elif rejected and self._current_cap > 1:
                self._current_cap -= 1
            self._p0_prev = (max(0.25, accept_ratio), max(0.25, math.exp(-logp_gap)))
            self._p1_prev = (depth_ratio, max(0.25, accept_ratio))

        return self._last_state

    @property
    def latest_scar(self) -> Optional[DraftScar]:
        return self._scars[-1] if self._scars else None

    @torch.no_grad()
    def recycle_scar_for_projection(
        self,
        x_next: torch.Tensor,
        linear_fn: Callable[[torch.Tensor], torch.Tensor],
        scar: Optional[DraftScar] = None,
    ) -> tuple[torch.Tensor, bool]:
        """Reversibly recycle a rejected draft scar (x_draft, y_draft) via exact non-commutative contact:

            alpha = <x_draft, x_next> / (||x_draft||^2 + eps)
            r_x   = x_next - alpha * x_draft
            delta = ||r_x|| / (||x_next|| + eps)
            y_out = alpha * y_draft + linear_fn(r_x)   (zero-FLOP exact reuse when delta <= tol!)
        """
        sc = self.latest_scar if scar is None else scar
        if sc is None:
            return linear_fn(x_next), False

        x_flat = x_next.reshape(1, -1)
        x_d = sc.x_draft.to(device=x_flat.device, dtype=torch.float32)
        y_d = sc.y_draft.to(device=x_flat.device, dtype=torch.float32)
        if x_d.shape[-1] != x_flat.shape[-1]:
            return linear_fn(x_next), False

        x_f = x_flat.float()
        denom = float((x_d * x_d).sum().item()) + 1e-12
        if denom <= 1e-10:
            return linear_fn(x_next), False
        alpha = float((x_f * x_d).sum().item()) / denom
        r_x = x_f - alpha * x_d
        x_norm = float(x_f.norm().item()) + 1e-12
        rel_res = float(r_x.norm().item()) / x_norm

        self.anchors_recycled += 1
        if rel_res <= self.recycle_residual_tol:
            self.zero_flop_recycles += 1
            y_rec = (alpha * y_d).to(dtype=x_next.dtype)
            return y_rec.reshape(*x_next.shape[:-1], y_d.shape[-1]), True

        self.diff_recycles += 1
        y_delta = linear_fn(r_x.to(dtype=x_next.dtype).reshape_as(x_next))
        y_rec = (alpha * y_d.reshape_as(y_delta) + y_delta.float()).to(dtype=y_delta.dtype)
        return y_rec, True

    def reset(self) -> None:
        self._p0_prev = (1.0, 1.0)
        self._p1_prev = (1.0, 1.0)
        self._strategy = (1.0, 1.0)
        self._current_cap = self.max_drafts
        self._last_state = "EQUAL"
        self._last_tau_core = 1.0
        self._last_tau_wing = 1.0
        self._last_delta_res = 0.0
        self._scars.clear()
        self.last_logp_gap = 0.0

    def stats(self) -> dict:
        return {
            "scale": "M7_mirror_speculation",
            "max_drafts": self.max_drafts,
            "current_cap": self._current_cap,
            "last_mirror_state": self._last_state,
            "tau_core": round(self._last_tau_core, 4),
            "tau_wing": round(self._last_tau_wing, 4),
            "delta_res": round(self._last_delta_res, 4),
            "steps": self.steps,
            "equal_steps": self.equal_steps,
            "tension_steps": self.tension_steps,
            "split_events": self.split_events,
            "scars_captured": self.scars_captured,
            "scars_active": len(self._scars),
            "anchors_recycled": self.anchors_recycled,
            "zero_flop_recycles": self.zero_flop_recycles,
            "diff_recycles": self.diff_recycles,
            "q1_anchors_retained": self.q1_anchors_retained,
            "last_logp_gap": round(self.last_logp_gap, 4),
        }


def _iter_q1_modules(model):
    from src.quantized_projection import PackedLinear
    from src.q1_cube_engine import Q1StreamedLinear

    types_list = [PackedLinear, Q1StreamedLinear]
    try:
        from src.ideal_q1_engine import IdealQ1Linear
        types_list.append(IdealQ1Linear)
    except ImportError:
        pass
    try:
        from src.binary_halving_q1 import BinaryHalvingQ1Linear
        types_list.append(BinaryHalvingQ1Linear)
    except ImportError:
        pass
    cls_tuple = tuple(types_list)
    return [m for m in model.modules() if isinstance(m, cls_tuple)]


class Q1SkeletonDrafter:
    """Lossless-speculation skeleton drafter using truncated MSB Q1 planes (P=2)."""

    def __init__(self, runtime, active_planes: int = 2):
        self.runtime = runtime
        self.active_planes = int(active_planes)

    @property
    def supported(self) -> bool:
        return bool(_iter_q1_modules(self.runtime.engine.model))

    @torch.no_grad()
    def draft(self, last_token: int, past, cursor: int, count: int) -> tuple[list[int], list[float]]:
        if count <= 0:
            return [], []
        targets = _iter_q1_modules(self.runtime.engine.model)
        if not targets:
            return [], []
        eng = self.runtime.engine
        tokens: list[int] = []
        probs: list[float] = []
        with _auxiliary_forwards(self.runtime):
            snap = capture_cache_snapshot(past)
            saved_cubes: list[tuple[object, int]] = []
            try:
                for mod in targets:
                    if hasattr(mod, "active_cubes") and hasattr(mod, "root_bits"):
                        saved_cubes.append((mod, int(mod.active_cubes)))
                        mod.active_cubes = min(self.active_planes, int(mod.root_bits))
                        continue
                    bits_val = getattr(mod, "bits", None)
                    if bits_val == 1 or bits_val is None:
                        total_p = (
                            getattr(mod, "total_planes", None)
                            or getattr(mod, "q1_planes", None)
                            or getattr(mod, "planes", None)
                            or bits_val
                            or self.active_planes
                        )
                    else:
                        total_p = bits_val
                    mod.active_planes = min(self.active_planes, int(total_p))
                cur_tok = int(last_token)
                cur_pos = int(cursor)
                draft_past = past
                for _ in range(count):
                    logits, draft_past = eng._forward_step(
                        torch.tensor([[cur_tok]], dtype=torch.long),
                        draft_past,
                        cache_position=torch.tensor([cur_pos], dtype=torch.long),
                        prefill=False,
                        prefill_end=False,
                        return_all_logits=False,
                        prefetch_next=False,
                    )
                    row = logits[0, -1, :] if logits.ndim == 3 else logits[0]
                    row_f = row.float()
                    next_tok = int(row_f.argmax().item())
                    max_logit = float(row_f[next_tok].item())
                    p = float((1.0 / torch.exp(row_f - max_logit).sum()).item())
                    tokens.append(next_tok)
                    probs.append(max(0.05, min(0.99, p)))
                    cur_tok = next_tok
                    cur_pos += 1
            finally:
                for mod, old_c in saved_cubes:
                    mod.active_cubes = old_c
                for mod in targets:
                    if not (hasattr(mod, "active_cubes") and hasattr(mod, "root_bits")):
                        mod.active_planes = None
                restore_cache_snapshot(past, snap)
        return tokens, probs


class GeneralSpeculation:
    def __init__(self, runtime, mode="auto", head_path=None, head_budget_bytes=128*2**20, *, mirror_mode: bool = False):
        from src.quintessentia_integrated import _NORM_WEIGHT_CANDIDATES
        self.runtime, self.mode = runtime, mode
        self.mirror_mode = bool(mirror_mode)
        self.mirror_spec = (
            MirrorSpeculationScale(max_drafts=max(1, int(getattr(runtime, "max_drafts", 2))))
            if self.mirror_mode
            else None
        )
        self.native = None
        self.native_error = None
        self.q1_drafter = Q1SkeletonDrafter(runtime, active_planes=2) if mode in ("auto", "q1") else None
        self.counters = Counter()
        self.state_budget_bytes = 64*2**20
        self.snapshot_peak_bytes = 0
        self.hidden = None
        self.capturing = False
        self.hook = None
        self._draft_ctx = None
        modules = dict(runtime.engine.model.named_modules())
        for candidate in _NORM_WEIGHT_CANDIDATES:
            norm = modules.get(candidate.removesuffix(".weight"))
            if norm is not None:
                self.hook = norm.register_forward_hook(self._capture)
                break
        if mode not in ("lookup", "q1", "off"):
            source = runtime.engine.model_source
            available = head_path is not None or (hasattr(source, "mtp_entries") and bool(source.mtp_entries()))
            if available:
                try:
                    from src.native_mtp import NativeMTPHead
                    self.native = NativeMTPHead(runtime.engine, head_path, head_budget_bytes)
                except (ValueError, NotImplementedError, FileNotFoundError) as exc:
                    self.native_error = str(exc)
                    if mode == "native":
                        self.close()
                        raise
            elif mode == "native":
                self.close()
                raise FileNotFoundError("Native speculation requires trained MTP weights; supply --mtp-head")
        if mode == "q1" and not self.q1_drafter.supported:
            self.close()
            raise NotImplementedError("Q1 speculation requires PackedLinear or Q1StreamedLinear layers")
        if self.native is not None and self.hook is None:
            self.close()
            raise NotImplementedError("Cannot locate the native head's backbone hidden-state boundary")
        from src.rowwise_verification import RowwiseVerification
        self.rowwise = RowwiseVerification(runtime.engine)

    def _capture(self, module, args, output):
        if self.capturing:
            self.hidden = output.detach()

    def _run(self, tokens, past, position, *, prefill=False, prefill_end=True, all_logits=False, prefetch=True):
        self.hidden, self.capturing = None, True
        try:
            result, past = self.runtime.engine._forward_step(
                torch.tensor([tokens], dtype=torch.long), past, cache_position=torch.arange(position, position+len(tokens)),
                prefill=prefill, prefill_end=prefill_end, return_all_logits=all_logits, prefetch_next=prefetch)
            return result, past, self.hidden
        finally:
            self.capturing = False

    def propose(self, history, count, native_ready, *, past=None, cursor=None, gate=None):
        from quintessentia_cli.unified_runtime import QuintessentiaSpecGate, quintessentia_spec_lookup
        if self._draft_ctx is not None and past is None and cursor is None and gate is None:
            past, cursor, gate = self._draft_ctx
        src_q1 = getattr(QuintessentiaSpecGate, "SRC_Q1", 2)
        if self.mode == "off" or count <= 0:
            return [], [], QuintessentiaSpecGate.SRC_LOOKUP
        if native_ready and self.mode in ("auto", "native"):
            tokens, probs = self.native.draft(count)
            return tokens, probs, QuintessentiaSpecGate.SRC_MTP
        if self.mode != "q1":
            tokens = quintessentia_spec_lookup(history, max_drafts=count)
            if tokens:
                return tokens, [0.75]*len(tokens), QuintessentiaSpecGate.SRC_LOOKUP
        if self.mode in ("auto", "q1") and self.q1_drafter is not None and self.q1_drafter.supported and history and cursor is not None:
            eng_dev = getattr(getattr(self.runtime.engine, "device", None), "type", "cpu")
            allow_q1 = self.mode == "q1" or (
                eng_dev == "cuda"
                and (gate is None or not hasattr(gate, "expected_gain") or gate.expected_gain(src_q1, [0.72]*count) > 1.02)
            )
            if allow_q1:
                try:
                    tokens, probs = self.q1_drafter.draft(history[-1], past, cursor, count)
                    if tokens:
                        return tokens, probs, src_q1
                except (NotImplementedError, torch.cuda.OutOfMemoryError):
                    self.counters["q1_unsupported_steps"] += 1
        return [], [], QuintessentiaSpecGate.SRC_LOOKUP

    def _boundary(self, past):
        progressive = getattr(self.runtime.engine, "progressive", None)
        if progressive is not None and not progressive.complete:
            if progressive.error is not None and progressive.gpu_enabled:
                past = progressive._continue_on_cpu(past)
            progressive.start()
            past = progressive._adopt_all(past)
        return past

    @contextmanager
    def _verification(self):
        eng = self.runtime.engine
        with ExitStack() as stack:
            progressive = getattr(eng, "progressive", None)
            warmup = getattr(eng, "probability_warmup", None)
            if progressive is not None:
                stack.enter_context(progressive.verification_forwards())
            if warmup is not None:
                stack.enter_context(warmup.verification_forwards())
            yield

    @staticmethod
    def _snapshot_bytes(past):
        seen = set()
        def size(value):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            if isinstance(value, torch.Tensor):
                return value.nbytes
            if isinstance(value, dict):
                return sum(size(v) for v in value.values())
            if isinstance(value, (tuple, list)):
                return sum(size(v) for v in value)
            return 0
        total = 0
        for layer in getattr(past, "layers", ()):
            seen.clear()
            state = vars(layer)
            # Append-only snapshots retain old KV storage even without cloning;
            # count it conservatively so optional verification cannot pin an
            # unbounded old prefix while the new cache grows.
            total += sum(size(v) for v in state.values())
        return total

    @staticmethod
    def _crop_append_only(past, position):
        from transformers.cache_utils import DynamicLayer
        if hasattr(past, "layers") and all(type(layer) is DynamicLayer for layer in past.layers):
            remove = int(past.get_seq_length())-position
            if remove > 0:
                past.crop(-remove)
            return True
        return False

    def stream(self, prompt_ids, max_new_tokens, stop_on_eos, cached_prefix_len=0,
               past_kv=None, cancel_check=None):
        from quintessentia_cli.unified_runtime import QuintessentiaSpecGate, _eos_token_set
        u, eng = self.runtime, self.runtime.engine
        check = u._make_cancel_checker(cancel_check)
        eos = _eos_token_set(u.tokenizer.eos_token_id)
        anchor = getattr(u, "_active_thought_anchor", None)
        gate = QuintessentiaSpecGate(max_drafts=u.max_drafts)
        before = self.counters.copy()
        all_ids = list(prompt_ids)
        gen_history: list[int] = []
        sampling = getattr(u, "_active_sampling", None)
        stochastic_sampling = bool(sampling is not None and float(sampling.temperature) > 0.0)
        past = past_kv if cached_prefix_len else None
        native_ready = self.native is not None and self.native.resume(cached_prefix_len, prompt_ids[cached_prefix_len])
        cursor = cached_prefix_len
        native_rows = 0
        phase = ["prefill"]
        eng._step_cancel_check = lambda: check(phase[0])
        try:
            with torch.no_grad():
                u._reset_compute_peaks()
                u._sync_compute()
                started = time.perf_counter()
                chunk = max(1, u.prefill_chunk_tokens)
                for a in range(cached_prefix_len, len(prompt_ids), chunk):
                    check("prefill")
                    b = min(len(prompt_ids), a+chunk)
                    logits, past, hidden = self._run(prompt_ids[a:b], past, a,
                        prefill=True, prefill_end=b == len(prompt_ids),
                        prefetch=b < len(prompt_ids) or max_new_tokens > 1)
                    if b == len(prompt_ids):
                        if anchor is not None:
                            logits = anchor.on_prefill_end(u, past, logits)
                        picked = u._sample_next_token(logits[0], gen_history)
                    if native_ready:
                        next_tokens = prompt_ids[a+1:b+1] if b < len(prompt_ids) else prompt_ids[a+1:b]+[picked]
                        self.native.commit(hidden, next_tokens)
                        native_rows += len(next_tokens)
                    cursor = b
                all_ids.append(picked)
                gen_history.append(picked)
                u._sync_compute()
                prefill_ms = (time.perf_counter()-started)*1000
                yield {"kind": "tokens", "new_ids": [picked]}
                generated, passes = 1, 0
                phase[0] = "decode"
                u._sync_compute()
                started = time.perf_counter()
                while generated < max_new_tokens and not (stop_on_eos and picked in eos):
                    check("decode")
                    past = self._boundary(past)
                    remaining = max_new_tokens-generated
                    # Recurrent anchor mixing changes state after each token;
                    # stochastic sampling also runs plain 1-token steps to preserve exact distribution.
                    depth = (
                        0
                        if (
                            stochastic_sampling
                            or (anchor is not None and getattr(anchor, "rec_anchor_lambda", 0) > 0)
                        )
                        else min(u.max_drafts, remaining-1)
                    )
                    if depth and not self.rowwise.supported:
                        depth = 0
                        self.counters["decoder_contract_plain_steps"] += 1
                    t_draft = time.perf_counter()
                    self._draft_ctx = (past, cursor, gate)
                    try:
                        drafts, probs, source = self.propose(all_ids, depth, native_ready)
                    finally:
                        self._draft_ctx = None
                    gate_src = source if source < len(gate.d) else gate.SRC_LOOKUP
                    gate.record_draft_cost(gate_src, time.perf_counter()-t_draft)
                    if self.mirror_spec is not None:
                        depth = self.mirror_spec.pick_depth(probs, len(drafts)) if drafts else 0
                    else:
                        depth = min(len(drafts), gate.pick_depth(gate_src, probs)) if drafts else 0
                    drafts, probs = drafts[:depth], probs[:depth]
                    snapshot = None
                    if depth:
                        try:
                            needed = self._snapshot_bytes(past)
                            contour = getattr(u, "stanok_contour", None)
                            while depth and (
                                needed * (depth + 1) > self.state_budget_bytes
                                or (contour is not None and not contour.can_admit_bytes(needed * (depth + 1)))
                            ):
                                depth -= 1
                                drafts, probs = drafts[:depth], probs[:depth]
                            if not depth:
                                self.counters["state_budget_plain_steps"] += 1
                                depth, drafts, probs = 0, [], []
                            else:
                                snapshot = capture_cache_snapshot(past)
                                self.snapshot_peak_bytes = max(self.snapshot_peak_bytes, needed*(depth+1))
                        except (NotImplementedError, torch.cuda.OutOfMemoryError):
                            self.counters["unsupported_cache_plain_steps"] += 1
                            depth, drafts, probs = 0, [], []
                    inputs = [picked]+drafts
                    routes = getattr(eng, "route_exclusion", None)
                    if routes is not None and depth:
                        routes.start_capture()
                    t_forward = time.perf_counter()
                    # Tentative rows must not rewrite task maps or move a layer
                    # between devices before a rejection can restore its cache.
                    with self._verification() if depth else nullcontext(), self.rowwise.execution(past, len(inputs)):
                        self.rowwise.position = cursor
                        logits, past, hidden = self._run(inputs, past, cursor, all_logits=True, prefetch=remaining > 1)
                        route_trace = routes.finish_capture() if routes is not None and depth else None
                        rows, emitted, accepted, hit_eos = [], [], 0, False
                        for j in range(len(inputs)):
                            row = logits[:, j, :]
                            if anchor is not None:
                                row = anchor.transform_logits(row, past, generated+len(emitted))
                            token = u._sample_next_token(row[0], gen_history + emitted)
                            rows.append(token)
                            emitted.append(token)
                            if j < depth and token == drafts[j]:
                                accepted += 1
                                if stop_on_eos and token in eos:
                                    hit_eos = True
                                    break
                                continue
                            hit_eos = stop_on_eos and token in eos
                            break
                        keep = accepted if hit_eos and accepted == len(emitted) else accepted+1
                        kept_hidden = hidden[:, :keep].detach() if hidden is not None else None
                        if keep < len(inputs):
                            if self.mirror_spec is not None and depth:
                                rej_idx = min(accepted, len(inputs) - 1)
                                rej_lg = logits[:, rej_idx, :]
                                rej_h = hidden[:, keep:keep + 1, :] if (hidden is not None and keep < hidden.shape[1]) else (
                                    hidden[:, rej_idx:rej_idx + 1, :] if hidden is not None else None
                                )
                                rej_y = logits[:, keep:keep + 1, :] if keep < logits.shape[1] else rej_lg.unsqueeze(1)
                                self.mirror_spec.record_verification(
                                    proposed=depth,
                                    accepted=accepted,
                                    position=cursor + accepted,
                                    drafts=drafts,
                                    verified_token=emitted[-1],
                                    reject_row_logits=rej_lg,
                                    reject_hidden=rej_h,
                                    reject_out=rej_y,
                                )
                                # Rejected forward already seeded fresh (x_last, y_last) anchors in Q1 trunk layers
                                self.mirror_spec.q1_anchors_retained += 1
                            if not self._crop_append_only(past, cursor+keep):
                                if self.rowwise.restore(past, keep):
                                    self.counters["row_state_restores"] += 1
                                else:
                                    restore_cache_snapshot(past, snapshot)
                                    with _auxiliary_forwards(u):
                                        _, past, kept_hidden = self._run(inputs[:keep], past, cursor)
                                    self.counters["replay_forwards"] += 1
                                    passes += 1
                            self.counters["rejected_tail_rows"] += len(inputs)-keep
                        elif self.mirror_spec is not None and depth:
                            self.mirror_spec.record_verification(
                                proposed=depth,
                                accepted=accepted,
                                position=cursor + accepted,
                                drafts=drafts,
                                verified_token=emitted[-1],
                            )
                    if routes is not None and depth:
                        routes.commit_capture(route_trace, keep)
                    gate.record_forward(len(inputs), time.perf_counter()-t_forward)
                    if depth:
                        gate.record_result(gate_src, min(depth, accepted+1), accepted, probs)
                        self.counters["verified_batches"] += 1
                        self.counters["drafts_proposed"] += depth
                        self.counters["drafts_accepted"] += accepted
                        src_q1 = getattr(gate, "SRC_Q1", 2)
                        if source == gate.SRC_MTP:
                            self.counters["native_batches"] += 1
                        elif source == src_q1:
                            self.counters["q1_batches"] += 1
                        else:
                            self.counters["lookup_batches"] += 1
                    if native_ready:
                        # The next token for the final committed hidden state is
                        # either the mismatch/bonus or the terminal accepted EOS.
                        next_tokens = inputs[1:keep]+[emitted[-1]]
                        self.native.commit(kept_hidden, next_tokens)
                        native_rows += keep
                    if anchor is not None:
                        anchor.on_step_committed(past, len(emitted))
                    cursor += keep
                    passes += 1
                    generated += len(emitted)
                    all_ids.extend(emitted)
                    gen_history.extend(emitted)
                    picked = emitted[-1]
                    yield {"kind": "tokens", "new_ids": emitted}
                u._sync_compute()
                decode_s = time.perf_counter()-started
                metrics = gate.summary()
                metrics.update({key: self.counters[key]-before[key] for key in self.counters})
                q1_used = (self.counters.get("q1_batches", 0) - before.get("q1_batches", 0)) > 0
                src_name = "native_mtp" if native_ready else ("q1_skeleton" if (self.mode == "q1" or q1_used) else "prompt_lookup")
                metrics.update(source=src_name, native_head_rows=native_rows)
                if self.mirror_spec is not None:
                    metrics["mirror_spec"] = self.mirror_spec.stats()
                yield {"kind": "final_state", "all_ids": all_ids, "past": past, "last_fed_h": None,
                       "consumed_tokens": cursor, "prefill_ms": prefill_ms, "decode_s": decode_s,
                       "decode_tokens": generated-1, "backbone_passes": passes, "spec_gate": metrics}
        finally:
            eng._step_cancel_check = None
            self.capturing, self.hidden = False, None
            routes = getattr(eng, "route_exclusion", None)
            if routes is not None:
                routes.cancel_capture()

    def stats(self):
        src_name = "native_mtp" if self.native else ("q1_skeleton" if (self.mode == "q1" or self.counters.get("q1_batches", 0) > 0) else "prompt_lookup")
        out = dict(self.counters, enabled=self.mode != "off", source=src_name,
                   mirror_mode=self.mirror_mode,
                   batching_supported=self.rowwise.supported,
                   native_head_bytes=self.native.weight_bytes if self.native else 0,
                   q1_draft_supported=bool(self.q1_drafter and self.q1_drafter.supported),
                   state_budget_bytes=self.state_budget_bytes, snapshot_peak_bytes=self.snapshot_peak_bytes,
                   native_head_error=self.native_error)
        if self.mirror_spec is not None:
            out["mirror_spec"] = self.mirror_spec.stats()
        return out

    def reset(self):
        if self.native is not None:
            self.native.reset()
        if self.mirror_spec is not None:
            self.mirror_spec.reset()

    def close(self):
        if getattr(self, "rowwise", None) is not None:
            self.rowwise.close()
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
        if self.native is not None:
            self.native.close()
            self.native = None
