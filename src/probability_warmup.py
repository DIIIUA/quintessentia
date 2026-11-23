"""Opt-in prefill-map weight preparation for HF Granite/OLMoE Q4/Q6.

The first prefill activity map is retained separately from its working regions.
Actual router results guard those regions; there is no additional lm_head or
candidate-attention forward. Placement changes never remove actual expert work.
This activity prior is not the semantic Phi_0 anchor from thought_buffer.py.
"""
from collections import Counter, deque
from contextlib import contextmanager
from functools import partial
import hashlib
import json

import torch

from src.progressive_prefetch import ThreeBankExpertPrefetch


class PrefillRegionMap:
    """Immutable initial activity counts and event-revised working regions.

    A sustained route miss is an observable proxy for trajectory change, not a
    claim to detect changes of meaning. Isolated misses leave the plan intact.
    """

    def __init__(self, num_experts, capacity, *, drift_patience=3,
                 drift_miss_fraction=0.5, recent_window=8, initial_weight=0.25):
        if drift_patience < 1 or recent_window < drift_patience or not 0 < drift_miss_fraction <= 1:
            raise ValueError("Invalid trajectory guard settings")
        self.num_experts = dict(num_experts)
        self.capacity = int(capacity)
        self.patience = int(drift_patience)
        self.miss_fraction = float(drift_miss_fraction)
        self.window = int(recent_window)
        if not 0 < initial_weight < 1:
            raise ValueError("initial_weight must be between zero and one")
        self.initial_weight = float(initial_weight)
        self.initial = {}
        self.regions = {}
        self.revisions = {}
        self.recent = {}
        self.outside_streak = {}
        self.events = []
        self.checks = self.misses = self.drift_checks = self.drift_revisions = 0

    def _rank(self, layer, counts):
        prior = self.initial[layer]
        old_total = max(1, sum(prior))
        new_total = max(1, sum(counts[e] for e in range(self.num_experts[layer])))
        def score(e):
            return (self.initial_weight * prior[e] / old_total
                    + (1-self.initial_weight) * counts[e] / new_total)
        return tuple(sorted(range(self.num_experts[layer]),
                            key=lambda e: (-score(e), -prior[e], e))[:self.capacity])

    def _set(self, layer, region, reason):
        old = self.regions.get(layer, ())
        # A priority shuffle within the same retained area needs no new DMA.
        if set(old) == set(region):
            return False
        self.regions[layer] = region
        self.revisions[layer] = self.revisions.get(layer, 0)+1
        self.events.append(dict(layer=layer, reason=reason, revision=self.revisions[layer],
                                before=list(old), after=list(region)))
        self.events = self.events[-128:]
        return True

    def seed(self, layer, counts):
        counts = tuple(int(n) for n in counts)
        if len(counts) != self.num_experts[layer] or not sum(counts) or min(counts) < 0:
            raise ValueError("Invalid prefill activity counts")
        first = layer not in self.initial
        if first:
            self.initial[layer] = counts
        self.recent[layer] = deque(maxlen=self.window)
        self.outside_streak[layer] = 0
        return self._set(layer, self._rank(layer, counts), "initial_prefill" if first else "new_input")

    def observe(self, layer, ids):
        if layer not in self.regions or not ids:
            return False
        self.checks += 1
        missed = sum(e not in self.regions[layer] for e in ids)
        self.misses += missed
        self.recent[layer].append(tuple(ids))
        self.outside_streak[layer] = self.outside_streak[layer]+1 if missed/len(ids) >= self.miss_fraction else 0
        if self.outside_streak[layer] < self.patience:
            return False
        self.outside_streak[layer] = 0
        self.drift_checks += 1
        counts = Counter(e for row in self.recent[layer] for e in row)
        changed = self._set(layer, self._rank(layer, counts), "sustained_route_shift")
        self.drift_revisions += int(changed)
        return changed

    def stats(self):
        original = {str(k): list(v) for k, v in sorted(self.initial.items())}
        digest = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
        return dict(initial_sha256=digest, initial_counts=original,
                    regions={str(k): list(v) for k, v in sorted(self.regions.items())},
                    revisions=dict(self.revisions), checks=self.checks, missed_experts=self.misses,
                    drift_checks=self.drift_checks, drift_revisions=self.drift_revisions,
                    drift_patience=self.patience, drift_miss_fraction=self.miss_fraction,
                    recent_window=self.window, initial_weight=self.initial_weight, events=list(self.events))


class ProbabilityWarmup:
    """Prefill establishes the map; three bank lanes retain its working areas."""

    def __init__(self, engine, region_experts=12, target_layers=None,
                 gpu_budget_bytes=64*1024**2, host_cache_bytes=64*1024**2,
                 drift_patience=3, drift_miss_fraction=0.5, recent_window=8,
                 unified_memory=False, dynamic_regions=False):
        from aircolibri_cli.model_source import HFShardedSource, PreadSafetensorsReader, _HFExpertCacheAndLoader, _GGUFExpertCacheAndLoader
        from src.quantized_experts import QuantizedExpertCache
        source = getattr(engine, "model_source", None)
        cache = getattr(source, "_expert_loader", None) if source is not None else None
        kind = getattr(engine.config, "model_type", "")
        if getattr(engine, "_wavefront_active", False):
            raise ValueError("Disable wavefront for prefill-map warmup trials")
        if dynamic_regions and target_layers is not None:
            raise ValueError('Dynamic preparation follows the next layer from the complete prefill map')
        if dynamic_regions and not unified_memory:
            raise ValueError('Dynamic regions require unified memory')

        has_quant_moe = isinstance(cache, QuantizedExpertCache) and len(cache.enabled_layers) > 0
        if not unified_memory:
            if (kind not in {"granitemoe", "olmoe"} or type(source) is not HFShardedSource
                    or not isinstance(cache, QuantizedExpertCache) or len(cache.enabled_layers) < 3):
                raise ValueError("Prefill-map warmup requires HF Granite/OLMoE Q4/Q6 and three eligible MoE layers")
        elif not has_quant_moe and not dynamic_regions:
            raise ValueError("Dense/Hybrid unified memory requires dynamic_regions=True")

        self.engine = engine
        self.cache = cache if has_quant_moe else None
        self.dynamic_regions = bool(dynamic_regions)
        self._auto_unified_attached = False
        self._next_target = None
        self.step = 0
        self.enabled_step = self.is_prefill = self.collecting = False
        self.prefill_end = True
        self.building_counts = {}
        self.submitted_versions = {}
        self._auxiliary_depth = 0
        self._verification_depth = 0
        self.counters = dict(prefill_chunks=0, prefill_route_batches=0, route_checks=0,
                             map_submissions=0, stable_checks=0, actual_experts=0,
                             map_hits=0, extra_lm_head_calls=0, extra_attention_calls=0,
                             auxiliary_forwards=0)
        self.handles = []
        self.guard_options = dict(drift_patience=drift_patience, drift_miss_fraction=drift_miss_fraction,
                                  recent_window=recent_window)
        resident_limit = int(getattr(engine, 'vram_resident_layers', 0) or 0)

        if not has_quant_moe:
            from src.dynamic_working_memory import DynamicWorkingMemory
            if not hasattr(engine.pipeline, 'slot_strides') or engine.pipeline.num_slots != 3:
                raise ValueError('Unified memory requires three streaming dense slots')
            memory = DynamicWorkingMemory(engine.pipeline, None, gpu_budget_bytes,
                                          min_region_experts=1, resident_layer_limit=resident_limit)
            self.shared_memory = memory
            self.prefetch = None
            self.targets = ()
            self.lanes = {}
            self.num_experts = {}
            self.capacity = 0
            self.region_map = None
            return

        targets = tuple(sorted(cache.enabled_layers)[:3] if target_layers is None else target_layers)
        if not unified_memory:
            if len(targets) != 3 or len(set(targets)) != 3 or any(t not in cache.enabled_layers for t in targets):
                raise ValueError("Select three different eligible target layers")
        else:
            if len(targets) < 1 or len(set(targets)) != len(targets) or any(t not in cache.enabled_layers for t in targets):
                raise ValueError("Select valid eligible target layers")
        self.targets = targets
        self.lanes = {layer: lane for lane, layer in enumerate(targets)}
        self.num_experts = {
            i: int(cache.base._layer_meta[i]["num_experts"] if hasattr(cache.base, "_layer_meta")
                   else cache.base.num_experts_by_layer[i])
            for i in sorted(cache.enabled_layers)
        }
        available_regions = ((cache.num_slots + int(gpu_budget_bytes)//cache.expert_bytes - 8)//3
                             if unified_memory else int(gpu_budget_bytes)//(3*cache.expert_bytes))
        if self.dynamic_regions:
            available_regions = 8  # one bounded B region; actual arena admission follows below
        top_k_tok = int(getattr(engine.config, 'num_experts_per_tok', 2) or 2)
        self.capacity = min(int(region_experts), min(self.num_experts.values()), available_regions)
        if self.capacity < top_k_tok:
            raise ValueError("Each bank must fit at least the actual top-k expert set")
        self.region_map = PrefillRegionMap(self.num_experts, self.capacity, **self.guard_options)
        if host_cache_bytes < cache.expert_bytes:
            raise ValueError('Host preparation budget must fit at least one expert')
        memory = None
        if unified_memory:
            from src.unified_working_memory import UnifiedWorkingMemory
            if not hasattr(engine.pipeline, 'slot_strides') or engine.pipeline.num_slots != 3:
                raise ValueError('Unified memory requires three streaming dense slots')
            # This cap reserves demand/probe room before mutating either backend.
            total_experts = cache.num_slots + int(gpu_budget_bytes)//cache.expert_bytes
            if not self.dynamic_regions and (total_experts-8)//3 < top_k_tok:
                raise ValueError('Unified budget needs warm regions plus eight demand/probe slots')
            if self.dynamic_regions:
                from src.dynamic_working_memory import DynamicWorkingMemory
                memory = DynamicWorkingMemory(engine.pipeline, cache, gpu_budget_bytes,
                                              min_region_experts=top_k_tok,
                                              resident_layer_limit=resident_limit)
            else:
                memory = UnifiedWorkingMemory(engine.pipeline, cache, gpu_budget_bytes,
                                              min_region_experts=top_k_tok)
                memory._resident_layer_limit = resident_limit
            self.capacity = min(self.capacity, memory.warm_capacity if self.dynamic_regions else (memory.num_slots-8)//3)
            self.region_map.capacity = self.capacity
            if self.dynamic_regions:
                self.targets = tuple(self.num_experts)
                self.lanes = {layer: layer % 3 for layer in self.targets}
        self.shared_memory = memory

        if getattr(source, "kind", None) == "gguf" or isinstance(cache.base, _GGUFExpertCacheAndLoader):
            cpu = _GGUFExpertCacheAndLoader(
                source.gguf_path,
                source._routed_experts,
                source._per_expert_gguf,
                source._routed_expert_biases,
                source=source,
                device=torch.device("cpu"),
                dtype=torch.bfloat16,
                budget_bytes=0,
            )
            def load(layer, eid):
                g, u, d = cpu.load_batch(layer, [eid], torch.device("cpu"), torch.bfloat16, False)
                return g[0], u[0], d[0]
            def close():
                cpu.close()
        else:
            reader = PreadSafetensorsReader()
            cpu = _HFExpertCacheAndLoader(reader, source._routed_experts, source._packed_experts,
                                         device=torch.device("cpu"), dtype=torch.bfloat16, budget_bytes=0)
            def load(layer, eid):
                g, u, d = cpu.load_batch(layer, [eid], torch.device("cpu"), torch.bfloat16, False)
                return g[0], u[0], d[0]
            def close():
                cpu.close()
                reader.close()

        if unified_memory:
            from src.unified_prefetch import UnifiedExpertPrefetch
            self.prefetch = UnifiedExpertPrefetch(memory, cache, self.capacity, load, close, host_cache_bytes)
        else:
            self.prefetch = ThreeBankExpertPrefetch(cache.H, cache.I, cache.bits, self.capacity, load, close,
                                                  engine.device, host_cache_bytes, cache.group_size)
        self.cache.warmup = self.prefetch
        layers = engine._decoder_layers() if hasattr(engine, "_decoder_layers") else engine.model.model.layers
        routed_containers = getattr(source, "_routed_expert_container", {}) or {}
        for idx in self.num_experts:
            exp_path = routed_containers.get(idx) or ("block_sparse_moe.experts" if kind == "granitemoe" else "mlp.experts")
            experts_mod = layers[idx].get_submodule(exp_path)
            self.handles.append(experts_mod.register_forward_pre_hook(partial(self._observe, idx)))

    @contextmanager
    def auxiliary_forwards(self):
        """Keep task maps/banks intact while an isolated thought-buffer probe runs.

        The producer can finish already submitted task work. Legacy probes bypass
        warm banks. Unified probes may lease immutable shared weights, but do not
        evict the desired task region or alter its map or task-use telemetry.
        Nested probes and exceptions restore attachment.
        """
        previous = self.cache.warmup if self.cache is not None else None
        self._auxiliary_depth += 1
        if self.cache is not None:
            self.cache.auxiliary_depth += 1
            self.cache.warmup = None
        try:
            yield
        finally:
            self._auxiliary_depth -= 1
            if self.cache is not None:
                self.cache.auxiliary_depth -= 1
                if self.prefetch is not None and not self.prefetch.closed:
                    self.cache.warmup = previous

    @contextmanager
    def verification_forwards(self):
        # Keep the retained-map producer alive while tentative routes are held
        # out of the task history. Actual misses can still use uncached weights.
        self._verification_depth += 1
        if self.cache is not None:
            self.cache.auxiliary_depth += 1
            self.cache.verification_depth += 1
        try:
            yield
        finally:
            if self.cache is not None:
                self.cache.verification_depth -= 1
                self.cache.auxiliary_depth -= 1
            self._verification_depth -= 1

    def begin_step(self, tokens, past, *, prefill=None, prefill_end=True):
        if self.prefetch is not None:
            self.prefetch._check()
        if self._auxiliary_depth:
            self.counters["auxiliary_forwards"] += 1
            return
        if past is None:
            self.reset()
        self.step += 1
        self.enabled_step = tokens.ndim == 2 and tokens.shape[0] == 1 and not getattr(self.engine,"_wavefront_active",False)
        self.is_prefill = bool(past is None or tokens.shape[-1] > 1) if prefill is None else bool(prefill)
        self.prefill_end = bool(prefill_end)
        if self.prefetch is None:
            return
        if self.enabled_step and self.is_prefill:
            self.counters["prefill_chunks"] += 1
            if not self.collecting:
                self.building_counts = {i: [0]*n for i, n in self.num_experts.items()}
                self.collecting = True
        elif self.enabled_step and not self.dynamic_regions:
            for layer in self.targets:
                self._submit(layer, "retained_map")

    def _submit(self, layer, reason):
        if layer not in self.lanes or not self.engine._prefetch_next_step:
            return
        version = self.region_map.revisions.get(layer)
        if self.dynamic_regions:
            if layer != self._next_target or version is None:
                return
            ids = tuple(self.region_map.regions[layer][:self.capacity])
            request = self.prefetch.requests.get(self.lanes[layer])
            if (request is not None and request['layer'] == layer
                    and request['region'] == (layer, version) and request['ids'] == ids):
                return
            self.prefetch.submit(layer, ids,
                region=(layer, version), lane=self.lanes[layer], depth=reason, step=self.step)
            self.counters['map_submissions'] += 1
            return
        request = self.prefetch.requests.get(self.lanes[layer])
        complete_region = request is not None and request['ids'] == tuple(self.region_map.regions[layer][:self.capacity])
        if version is None or (self.submitted_versions.get(layer) == version and complete_region):
            return
        self.prefetch.submit(layer, self.region_map.regions[layer], region=layer, lane=self.lanes[layer],
                             depth=reason, step=self.step)
        self.counters["map_submissions"] += 1
        self.submitted_versions[layer] = version

    def before_layer(self, layer):
        """Advance ownership using the retained map, without rebuilding it."""
        if not self.dynamic_regions:
            return
        mem = self.cache.shared_memory if self.cache is not None else self.shared_memory
        if mem is not None:
            mem.begin_layer(layer)
        if self.prefetch is None or self.region_map is None or self._auxiliary_depth or not self.enabled_step:
            return
        resident = int(getattr(self.engine, 'vram_resident_layers', 0) or 0)
        if layer < resident - 1 and resident in self.lanes:
            self._next_target = resident
        else:
            index = self.targets.index(layer)
            self._next_target = self.targets[(index+1)%len(self.targets)]
        if self._next_target in self.region_map.regions:
            self._submit(self._next_target,'next_region_from_retained_map')
        else:
            # First prefill has not observed the next region yet.
            if self.prefetch.requests or mem.desired or getattr(mem, 'active_protected', None):
                with self.prefetch.cv:
                    self.prefetch.requests.clear()
                    mem.set_desired(set())
                    self.prefetch.cv.notify_all()

    def prepare_next_dense(self, layer):
        engine = self.engine
        layers = engine._decoder_layers() if hasattr(engine, "_decoder_layers") else engine.model.model.layers
        last = len(layers) - 1
        target = layer+1
        if target>last:
            if not engine.tail_to_head_prefetch or not engine._prefetch_next_step:
                return
            target = engine.vram_resident_layers
        if target<engine.vram_resident_layers:
            return
        slot = engine.pipeline.slots[target%engine.num_slots]
        if slot.current_layer_idx==target and slot.loaded_keys:
            return
        if layer==last and slot.state.value!='FREE':
            return  # Same physical role: publish after its current reader is released.
        pinned = engine.weight_manager.load_layer_to_pinned_slot(target,target_slot_idx=target%engine.num_slots)
        ready = engine.pipeline.get_slot_for_loading(target)
        engine.pipeline.load_tensors_async(ready,pinned,pinned_pool=engine.weight_manager.pinned_pool)
        if layer==last:
            engine.tail_prefetched = True
            engine.pipeline.tail_prefetches += 1

    def _observe(self, layer, module, args):
        if len(args) < 2 or self._auxiliary_depth or self._verification_depth or not self.enabled_step:
            return
        indices = args[1].detach()
        # The dispatcher needs the same CPU routes. Share this bounded current
        # chunk copy instead of a histogram kernel plus a second GPU->CPU sync.
        observed = self.cache.observed_route_rows(layer, args[1])
        rows = observed if observed is not None else indices.cpu().tolist()
        self.cache._observed_routes = (layer,args[1],rows)
        ids = [eid for row in rows for eid in row if 0 <= eid < self.num_experts[layer]]
        if self.is_prefill:
            counts = [0]*self.num_experts[layer]
            for eid in ids:
                if 0 <= eid < len(counts):
                    counts[eid] += 1
            self.building_counts[layer] = [a+b for a, b in zip(self.building_counts[layer], counts)]
            self.counters["prefill_route_batches"] += 1
            if self.prefill_end:
                self.region_map.seed(layer, self.building_counts[layer])
                # This layer's map is known while its experts and the rest
                # of the current prefill are still being computed.
                self._submit(layer, "prefill_map")
            return
        if layer not in self.lanes or layer not in self.region_map.regions:
            return
        self.counters["route_checks"] += 1
        self.counters["actual_experts"] += len(ids)
        self.counters["map_hits"] += sum(e in self.region_map.regions[layer] for e in ids)
        if self.region_map.observe(layer, ids):
            self._submit(layer, "sustained_route_shift")
            tracker = getattr(self.engine, "survivor_tracker", None)
            if tracker is not None:
                tracker.on_route_revision(layer)
        else:
            self.counters["stable_checks"] += 1

    def end_step(self, logits, past):
        if self._auxiliary_depth:
            return
        if self.enabled_step and self.prefetch is not None:
            self.prefetch.finish_step(self.step)
        if self.is_prefill and self.prefill_end:
            self.collecting = False
            self.building_counts.clear()

    def reset(self):
        if self.cache is not None:
            self.cache._observed_routes = None
        if self.prefetch is not None:
            self.prefetch.reset()
        elif self.shared_memory is not None:
            self.shared_memory.reset_task()
        if self.num_experts:
            self.region_map = PrefillRegionMap(self.num_experts, self.capacity, **self.guard_options)
        self.collecting = self.enabled_step = False
        self.building_counts.clear()
        self.submitted_versions.clear()
        self._next_target = None

    def detach_and_restore(self):
        if self.step > 0:
            raise RuntimeError('Cannot detach unified memory after forward step 0')
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if getattr(self.engine, 'probability_warmup', None) is self:
            self.engine.probability_warmup = None
        if self.cache is not None:
            self.cache.warmup = None
        if self.prefetch is not None:
            self.prefetch.close()
            self.prefetch = None
        mem = self.shared_memory
        self.shared_memory = None
        if self.cache is not None:
            self.cache.shared_memory = None
        if mem is not None:
            mem.close()
        pipeline = getattr(self.engine, 'pipeline', None)
        if pipeline is not None:
            pipeline.working_memory = None
            if hasattr(pipeline, 'slots') and hasattr(pipeline, 'slot_layouts'):
                from src.bounded_memory_manager import slot_arena_geometry
                _, offsets, total_bytes = slot_arena_geometry(pipeline.layout, pipeline.num_slots, pipeline.slot_layouts)
                pipeline.slot_offsets = offsets
                pipeline.arena = (
                    torch.zeros(total_bytes, dtype=torch.uint8, device=pipeline.device)
                    if total_bytes > 0
                    else torch.empty(0, dtype=torch.uint8, device=pipeline.device)
                )
                for s, slot in enumerate(pipeline.slots):
                    slot.tensors = slot.layout.bind_slot_views(
                        pipeline.arena, s, dtype=pipeline.dtype, base_offset_bytes=offsets[s]
                    )
                    slot.active_tensors = dict(slot.tensors)
        if self.cache is not None:
            from src.quantized_experts import QuantizedExpertCache
            from src.quantized_projection import packed_specs
            if isinstance(self.cache, QuantizedExpertCache):
                slots = getattr(self.cache, '_initial_num_slots', self.cache.num_slots)
                self.cache.num_slots = slots
                if slots > 0 and self.cache.device.type == 'cuda':
                    specs_gu = packed_specs('weight', (2 * self.cache.I, self.cache.H), self.cache.bits, self.cache.group_size)
                    specs_dn = packed_specs('weight', (self.cache.H, self.cache.I), self.cache.bits, self.cache.group_size)
                    self.cache.gu = {k: torch.empty((slots, *sh), dtype=dt, device=self.cache.device) for k, (sh, dt) in specs_gu.items()}
                    self.cache.dn = {k: torch.empty((slots, *sh), dtype=dt, device=self.cache.device) for k, (sh, dt) in specs_dn.items()}

    def stats(self):
        mem = self.cache.shared_memory if self.cache is not None else self.shared_memory
        return {**self.counters,
                "policy": "prefill_map_with_route_guard" if self.region_map is not None else "dense_dynamic_cycle",
                "capacity": self.capacity,
                "target_layers": list(self.targets),
                "map": self.region_map.stats() if self.region_map is not None else None,
                "prefetch": self.prefetch.stats() if self.prefetch is not None else None,
                "shared_memory": mem.stats() if mem is not None else None}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.cache is not None:
            self.cache.warmup = None
        if self.prefetch is not None:
            self.prefetch.close()
        elif self.shared_memory is not None and self.cache is None:
            self.shared_memory.close()
