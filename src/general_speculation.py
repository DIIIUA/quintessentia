"""Shared streamed draft/verify/commit loop for CPU, GPU and the progressive graph.

Native heads and prompt lookup share verification. Only cache state is snapshotted;
weights remain under the existing bounded layer/expert managers.
"""
from collections import Counter
from contextlib import contextmanager, ExitStack, nullcontext
import time

import torch

from quintessentia_cli.thought_buffer import _auxiliary_forwards, capture_cache_snapshot, restore_cache_snapshot


def _iter_q1_modules(model):
    from src.quantized_projection import PackedLinear
    from src.q1_cube_engine import Q1StreamedLinear

    types_list = [PackedLinear, Q1StreamedLinear]
    try:
        from src.ideal_q1_engine import IdealQ1Linear
        types_list.append(IdealQ1Linear)
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
            try:
                for mod in targets:
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
                for mod in targets:
                    mod.active_planes = None
                restore_cache_snapshot(past, snap)
        return tokens, probs


class GeneralSpeculation:
    def __init__(self, runtime, mode="auto", head_path=None, head_budget_bytes=128*2**20):
        from src.quintessentia_integrated import _NORM_WEIGHT_CANDIDATES
        self.runtime, self.mode = runtime, mode
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
                        picked = int(logits[0].argmax().item())
                    if native_ready:
                        next_tokens = prompt_ids[a+1:b+1] if b < len(prompt_ids) else prompt_ids[a+1:b]+[picked]
                        self.native.commit(hidden, next_tokens)
                        native_rows += len(next_tokens)
                    cursor = b
                all_ids.append(picked)
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
                    # use ordinary steps while that explicitly enabled option runs.
                    depth = 0 if anchor is not None and getattr(anchor, "rec_anchor_lambda", 0) > 0 else min(u.max_drafts, remaining-1)
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
                            token = int(row[0].argmax().item())
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
                    picked = emitted[-1]
                    yield {"kind": "tokens", "new_ids": emitted}
                u._sync_compute()
                decode_s = time.perf_counter()-started
                metrics = gate.summary()
                metrics.update({key: self.counters[key]-before[key] for key in self.counters})
                q1_used = (self.counters.get("q1_batches", 0) - before.get("q1_batches", 0)) > 0
                src_name = "native_mtp" if native_ready else ("q1_skeleton" if (self.mode == "q1" or q1_used) else "prompt_lookup")
                metrics.update(source=src_name, native_head_rows=native_rows)
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
        return dict(self.counters, enabled=self.mode != "off", source=src_name,
                    batching_supported=self.rowwise.supported,
                    native_head_bytes=self.native.weight_bytes if self.native else 0,
                    q1_draft_supported=bool(self.q1_drafter and self.q1_drafter.supported),
                    state_budget_bytes=self.state_budget_bytes, snapshot_peak_bytes=self.snapshot_peak_bytes,
                    native_head_error=self.native_error)

    def reset(self):
        if self.native is not None:
            self.native.reset()

    def close(self):
        if getattr(self, "rowwise", None) is not None:
            self.rowwise.close()
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
        if self.native is not None:
            self.native.close()
            self.native = None
