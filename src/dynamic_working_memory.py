"""Elastic A (current/cache), B (next preparation), C (empty standby).

One fixed-address byte allocation is shared by dense payloads, cyclic layer
working sets (physical_area_id in {0, 1, 2} strictly rotating A -> B -> C(0) -> A),
and a cross-layer LRU expert cache_pool (physical_area_id=None).
Pages change ownership, never addresses. Retired CUDA readers live in
retiring_pages until their actual CUDA events complete, leaving standby C strictly
at used_bytes == 0 and reserved_bytes == 0.
"""
from collections import Counter, OrderedDict
import math
import threading

import torch

from src.bounded_memory_manager import SlotArenaLayout
from src.quantized_projection import packed_specs
from src.unified_working_memory import CudaEventPool, UnifiedWorkingMemory


class DynamicWorkingMemory(UnifiedWorkingMemory):
    dynamic_regions = True

    def __init__(self, pipeline, cache=None, extra_budget_bytes=0, min_region_experts=0, *,
                 allow_cpu_mock=False, enable_prefetch_timing=True, resident_layer_limit=0,
                 gpu_budget_bytes=None):
        if gpu_budget_bytes is not None:
            extra_budget_bytes = gpu_budget_bytes
        if pipeline.num_slots != 3:
            raise ValueError('Dynamic regions require a three-slot CUDA ring')
        if pipeline.device.type != 'cuda' and not allow_cpu_mock:
            raise ValueError('Dynamic regions require a three-slot CUDA ring')
        if pipeline.total_loads or getattr(cache, '_lru', None) or pipeline.working_memory is not None:
            raise ValueError('Enable dynamic regions before the first forward')
        if extra_budget_bytes < 0:
            raise ValueError('Negative preparation budget')
        self.device, self.pipeline = pipeline.device, pipeline
        self._has_expert_cache = cache is not None
        self._resident_layer_limit = max(0, int(resident_layer_limit))
        self._pinned_resident_pos = {}
        self.allow_cpu_mock = bool(allow_cpu_mock)
        self.enable_prefetch_timing = bool(enable_prefetch_timing)
        self.cv = threading.Condition(threading.RLock())
        self._waiters = 0
        self.event_pool = CudaEventPool(self.device, max_pool_size=512, enable_timing=False)
        self.closed = False
        self.epoch = 0
        self.desired = set()
        self.active_protected = set()
        self.demand_inflight = 0
        self.directory = OrderedDict()
        self.access_clock = 0
        self.counts = Counter({'dense_reclaims_expert': 0})
        self.area_reads, self.area_writes = [0] * 3, [0] * 3
        self.area_dense = [Counter() for _ in range(3)]
        self.dense_active = [False] * 3
        self.dense_blocks = {}
        self._dense_layouts = {}
        self._next_layout_id = 1
        self._dense_view_cache = {}

        # Monotonic physical cycle: A -> B -> C(0) -> A (never B -> A)
        self.active_area, self.warm_area, self.standby_area = 0, 1, 2
        self.area_role_history = [['A'], ['B'], ['C']]
        self.area_transitions = Counter({
            'A->B': 0, 'B->C': 0, 'C->A': 0,
            'B->A': 0, 'A->C': 0, 'C->B': 0,
        })
        self._area_pages = [set(), set(), set()]
        self.retiring_pages = []
        self.current_layer = None

        arena_bytes = int(pipeline.arena.numel()) if pipeline.arena is not None else 0
        if cache is None:
            # Dense/Hybrid mode: 0 extra VRAM above pipeline.arena; derive page_bytes from slot geometry.
            self.budget_bytes = arena_bytes
            self.temporary_reserve_bytes = 0
            self.expert_bytes = 0
            self.expert_layout = None
            strides = [int(layout.slot_stride) for layout in pipeline.slot_layouts if getattr(layout, 'slot_stride', 0) > 0]
            if strides and arena_bytes > 0:
                g = math.gcd(*strides, arena_bytes)
                max_s = max(strides)
                page_bytes = g if g >= max(256, max_s // 64) else max(256, arena_bytes // max(1, pipeline.num_slots))
            else:
                page_bytes = max(256, arena_bytes // max(1, pipeline.num_slots)) if arena_bytes > 0 else 256
            self.page_bytes = page_bytes
            self.num_slots = arena_bytes // self.page_bytes if arena_bytes > 0 else pipeline.num_slots
            self.max_dense_pages = max(
                (math.ceil(max((s.offset_bytes + s.nbytes for s in layout.specs.values()), default=0) / self.page_bytes)
                 for layout in pipeline.slot_layouts),
                default=1,
            )
            if self.num_slots < 2 * self.max_dense_pages:
                raise ValueError('Dense dynamic budget cannot fit active and next dense payloads')
            self.warm_capacity = 0
            self.per_area = self.num_slots
            self.dense_zone_pages = min(self.num_slots, 2 * self.max_dense_pages)
            self.gpu_bytes = self.num_slots * self.page_bytes
            self.staging_slots = 0
            self.pinned_bytes = 0
            self._staging_ready = []
            self._staging_next = 0
            pipeline.reset_in_flight()
            for slot in pipeline.slots:
                slot.tensors.clear()
                slot.active_tensors.clear()
            self.gpu = pipeline.arena
            self.host = None
            self.banks, self.host_banks = [], []
        else:
            self.budget_bytes = arena_bytes + cache.num_slots * cache.expert_bytes + int(extra_budget_bytes)
            self.temporary_reserve_bytes = cache.temporary_reserve_bytes
            self.expert_bytes = cache.expert_bytes
            specs = (packed_specs('weight', (2 * cache.I, cache.H), cache.bits, cache.group_size),
                     packed_specs('weight', (cache.H, cache.I), cache.bits, cache.group_size))
            templates = {f'{prefix}.{name}': spec for prefix, fields in zip(('gu', 'dn'), specs)
                         for name, spec in fields.items()}
            self.expert_layout = SlotArenaLayout.from_templates(templates, slot_alignment=256, tensor_alignment=256)
            self.page_bytes = self.expert_layout.slot_stride
            self.num_slots = self.budget_bytes // self.page_bytes
            self.max_dense_pages = max(
                math.ceil(max((s.offset_bytes + s.nbytes for s in layout.specs.values()), default=0) / self.page_bytes)
                for layout in pipeline.slot_layouts
            )
            self.warm_capacity = min(8, self.num_slots - 2 * self.max_dense_pages - 8)
            if self.warm_capacity < 1 or self.num_slots < 2 * self.max_dense_pages + 8 + min(min_region_experts, 8):
                raise ValueError('Dynamic budget cannot fit current, next payload and expert readers')
            self.per_area = self.num_slots  # one strided expert address space, not three fixed banks
            self.dense_zone_pages = min(self.num_slots, 2 * self.max_dense_pages)
            self.gpu_bytes = self.num_slots * self.page_bytes
            # Only in-flight H2D needs pinned backing. The GPU expert cache does not
            # need a second, equally large, pinned-RAM cache.
            self.staging_slots = 2
            self.pinned_bytes = self.staging_slots * self.page_bytes
            self._staging_ready = [None] * self.staging_slots
            self._staging_next = 0
            pipeline.reset_in_flight()
            for slot in pipeline.slots:
                slot.tensors.clear()
                slot.active_tensors.clear()
            pipeline.arena = None
            cache.gu = cache.dn = None
            self.gpu = torch.empty(self.gpu_bytes, dtype=torch.uint8, device=self.device)
            self.host = torch.empty(self.pinned_bytes, dtype=torch.uint8, pin_memory=(self.device.type == 'cuda'))
            self.banks, self.host_banks = [], []
            for arena, banks in ((self.gpu, self.banks), (self.host, self.host_banks)):
                slots = self.num_slots if arena is self.gpu else self.staging_slots
                fields = {}
                for name, spec in self.expert_layout.specs.items():
                    element_bytes = torch.empty((), dtype=spec.dtype).element_size()
                    tail = arena[spec.offset_bytes:].view(spec.dtype)
                    fields[name] = tail.as_strided(
                        (slots, *spec.shape),
                        (self.page_bytes // element_bytes, spec.shape[1], 1),
                    )
                banks.append(tuple({k.removeprefix(prefix + '.'): v for k, v in fields.items()
                                    if k.startswith(prefix + '.')} for prefix in ('gu', 'dn')))
        self.records = [
            dict(
                bank=0, pos=i, physical_area_id=None, retiring=False,
                key=None, generation=0, ready=None, readers=[],
                leases=0, warm_epoch=-1, origin=None, kind='free', last_used=-1,
            )
            for i in range(self.num_slots)
        ]
        self._page_busy = bytearray(self.num_slots)
        self._page_occupied = bytearray(self.num_slots)
        pipeline.arena = self.gpu
        pipeline.working_memory = self
        pipeline.slot_offsets = (None, None, None)
        if cache is not None:
            cache.shared_memory = self
            cache.num_slots = self.num_slots

    def _set_area(self, rec, new_area):
        pos = rec['pos']
        old_area = rec['physical_area_id']
        if old_area == new_area:
            return
        if old_area is not None:
            self._area_pages[old_area].discard(pos)
        if new_area is not None:
            self._area_pages[new_area].add(pos)
        rec['physical_area_id'] = new_area

    def _reap_retiring_pages(self):
        if not self.retiring_pages:
            return
        remaining = []
        for pos in self.retiring_pages:
            rec = self.records[pos]
            if not rec['retiring']:
                continue
            if rec['leases'] == 0 and not self._fences(rec):
                self._set_area(rec, None)
                rec.update(retiring=False, kind='free', key=None, warm_epoch=-1, origin=None)
                self._page_busy[pos] = 0
                self._page_occupied[pos] = 0
                self.counts['retired_reaped'] += 1
            else:
                remaining.append(pos)
        self.retiring_pages = remaining

    def _retire_record(self, rec):
        pos = rec['pos']
        if rec['key'] is not None:
            self._pinned_resident_pos.pop(rec['key'], None)
            self.directory.pop(rec['key'], None)
        rec['key'] = None
        rec['warm_epoch'] = -1
        self._set_area(rec, None)
        self._page_occupied[pos] = 0
        if rec['leases'] > 0 or self._fences(rec):
            if not rec['retiring']:
                rec['retiring'] = True
                rec['kind'] = 'retiring'
                self.retiring_pages.append(pos)
                self.counts['pages_retired'] += 1
            self._page_busy[pos] = 1 if rec['leases'] > 0 else 0
        else:
            rec['retiring'] = False
            rec['kind'] = 'free'
            self._page_busy[pos] = 0

    def area_for(self, record):
        area = record.get('physical_area_id')
        if area is not None:
            return area
        return self.warm_area if record['key'] in self.desired else self.active_area

    def _stage_packed(self, packed, record):
        slot = self._staging_next
        ready = self._staging_ready[slot]
        if ready is not None and not ready.query():
            # The CPU must not overwrite bytes still being read by H2D. This
            # waits for a real completion event, not an estimated transfer time.
            self.counts['staging_waits'] += 1
            ready.synchronize()
        staged = tuple({k: v[slot] for k, v in bank.items()} for bank in self.host_banks[0])
        for dst, src in zip(staged, packed):
            for name in dst:
                dst[name].copy_(src[name])
        return staged

    def _staging_complete(self, event):
        self._staging_ready[self._staging_next] = event
        self._staging_next = (self._staging_next + 1) % self.staging_slots

    def begin_layer(self, layer):
        """Advance physical areas strictly along A -> B -> C(0) -> A."""
        with self.cv:
            self._check()
            self._reap_retiring_pages()
            layer = int(layer)
            if self.current_layer is None:
                self.current_layer = layer
                self.counts['role_rotations'] += 1
                return

            old_a = self.active_area
            old_b = self.warm_area
            old_c = self.standby_area

            new_a = old_c
            new_b = old_a
            new_c = old_b

            # 1. old_B -> new_C(0): zero-copy adopt prepared pages into new_A;
            #    idle valid experts demote to cross-layer cache_pool;
            #    unfinished CUDA readers move to retiring_pages.
            for pos in tuple(self._area_pages[old_b]):
                rec = self.records[pos]
                if rec['kind'] == 'dense':
                    self._set_area(rec, new_a)
                    self.counts['zero_copy_adoptions'] += 1
                elif rec['kind'] == 'expert' and (rec['key'] in self.desired or rec['key'] in self.active_protected):
                    if rec['key'] in self.desired:
                        self._set_area(rec, new_a)
                        self.counts['zero_copy_adoptions'] += 1
                    else:
                        self._set_area(rec, None)
                        self.counts['demoted_to_cache_pool'] += 1
                elif rec['kind'] == 'expert' and rec['key'] is not None:
                    if rec['leases'] > 0 or self._fences(rec):
                        self._retire_record(rec)
                    else:
                        self._set_area(rec, None)
                        rec['warm_epoch'] = -1
                        self.counts['demoted_to_cache_pool'] += 1
                else:
                    self._retire_record(rec)

            for blk in self.dense_blocks.values():
                if blk.get('area') == old_b:
                    blk['area'] = new_a

            # 2. old_A -> new_B: shrink old_A to bounded prep set (<= warm_capacity);
            #    surviving reusable experts demote honestly to cache_pool (None).
            b_kept = 0
            for pos in tuple(self._area_pages[old_a]):
                rec = self.records[pos]
                if rec['kind'] == 'dense':
                    self._set_area(rec, new_a)
                elif rec['kind'] == 'expert':
                    if rec['key'] in self.desired and b_kept < self.warm_capacity:
                        self._set_area(rec, new_b)
                        b_kept += 1
                    else:
                        self._set_area(rec, None)
                        self.counts['demoted_to_cache_pool'] += 1

            for blk in self.dense_blocks.values():
                if blk.get('area') == old_a:
                    blk['area'] = new_a

            self.active_area = new_a
            self.warm_area = new_b
            self.standby_area = new_c
            self.current_layer = layer

            assert not self._area_pages[new_c], 'Standby C must be empty after rotation'
            self.area_role_history[new_a].append('A')
            self.area_role_history[new_b].append('B')
            self.area_role_history[new_c].append('C')
            self.area_transitions['A->B'] += 1
            self.area_transitions['B->C'] += 1
            self.area_transitions['C->A'] += 1
            self.counts['role_rotations'] += 1

    def set_desired(self, keys, active_protected=None):
        keys = set(keys)
        if len(keys) > self.warm_capacity:
            raise ValueError('B exceeds its bounded next-region preparation allowance')
        with self.cv:
            self._check()
            self.desired = keys
            self.active_protected = set(active_protected) if active_protected is not None else set()
            for pos in tuple(self._area_pages[self.warm_area]):
                rec = self.records[pos]
                if rec['kind'] == 'expert' and rec['key'] not in keys:
                    if not rec['leases'] and not self._fences(rec):
                        self._set_area(rec, None)
            b_count = sum(
                1 for pos in self._area_pages[self.warm_area]
                if self.records[pos]['key'] in keys
            )
            for key in keys:
                if b_count >= self.warm_capacity:
                    break
                pos = self.directory.get(key)
                if pos is not None:
                    rec = self.records[pos]
                    if rec['kind'] == 'expert' and rec['physical_area_id'] != self.warm_area:
                        rec['warm_epoch'] = self.epoch
                        self._set_area(rec, self.warm_area)
                        b_count += 1
            for key in self.active_protected:
                pos = self.directory.get(key)
                if pos is not None:
                    self.records[pos]['warm_epoch'] = self.epoch
            self._notify_waiters()

    def set_active_protected(self, keys):
        with self.cv:
            self._check()
            self.active_protected = set(keys)

    def _victim(self, key, protected, optional, auxiliary=False):
        self._reap_retiring_pages()
        dz = self.dense_zone_pages
        act_prot = self.active_protected
        pinned = self._pinned_resident_pos
        lim = self._resident_layer_limit
        empty_expert = (
            r for r in self.records[dz:]
            if r['kind'] == 'free' and not r['retiring'] and not r['leases']
        )
        empty_dense = (
            r for r in self.records[:dz]
            if r['kind'] == 'free' and not r['retiring'] and not r['leases']
        )
        retiring_done = (
            self.records[pos] for pos in self.retiring_pages
            if pos >= dz and not self.records[pos]['leases'] and not self._fences(self.records[pos])
        )
        retiring_fenced = () if optional else (
            self.records[pos] for pos in self.retiring_pages
            if not self.records[pos]['leases']
        )
        cold_expert_nonres = (
            self.records[i] for k, i in self.directory.items()
            if i >= dz and k[0] >= lim and k not in protected and k not in self.desired and k not in act_prot
            and (not optional or k not in pinned) and not self.records[i]['leases']
        )
        cold_expert_res = (
            self.records[i] for k, i in self.directory.items()
            if i >= dz and k[0] < lim and k not in protected and k not in self.desired and k not in act_prot
            and (not optional or k not in pinned) and not self.records[i]['leases']
        )
        cold_dense_nonres = (
            self.records[i] for k, i in self.directory.items()
            if i < dz and k[0] >= lim and k not in protected and k not in self.desired and k not in act_prot
            and (not optional or k not in pinned) and not self.records[i]['leases']
        )
        cold_dense_res = (
            self.records[i] for k, i in self.directory.items()
            if i < dz and k[0] < lim and k not in protected and k not in self.desired and k not in act_prot
            and (not optional or k not in pinned) and not self.records[i]['leases']
        )
        if optional:
            cold_expert = (
                cold_expert_nonres,
                cold_expert_res,
            )
            cold_dense = (
                cold_dense_nonres,
                cold_dense_res,
            )
        else:
            cold_expert = ((
                self.records[i] for k, i in self.directory.items()
                if i >= dz and k not in protected and k not in self.desired and k not in act_prot
                and not self.records[i]['leases']
            ),)
            cold_dense = ((
                self.records[i] for k, i in self.directory.items()
                if i < dz and k not in protected and k not in self.desired and k not in act_prot
                and not self.records[i]['leases']
            ),)
        hot = () if optional or auxiliary else (
            self.records[i] for k, i in self.directory.items()
            if k not in protected and (k in self.desired or k in act_prot) and not self.records[i]['leases']
        )
        # Keep small pools fully usable and protect dense_zone_pages once dense streaming is active.
        protect_dense = self.counts.get('dense_bindings', 0) > 0 and self.num_slots > dz + 8
        groups = (
            (empty_expert, retiring_done, *cold_expert, empty_dense, *cold_dense, retiring_fenced, hot)
            if protect_dense
            else (empty_expert, empty_dense, retiring_done, *cold_expert, *cold_dense, retiring_fenced, hot)
        )
        for group in groups:
            for record in group:
                if not optional or not self._fences(record):
                    return record
        return None

    def _write(self, key, packed, record, stream, origin):
        if record['key'] is not None:
            self._pinned_resident_pos.pop(record['key'], None)
        self._pinned_resident_pos.pop(key, None)
        result = super()._write(key, packed, record, stream, origin)
        pos = record['pos']
        if record['retiring'] and pos in self.retiring_pages:
            self.retiring_pages.remove(pos)
        target_area = self.warm_area if key in self.desired else self.active_area
        record['kind'] = 'expert'
        record['retiring'] = False
        self._set_area(record, target_area)
        self._page_busy[pos] = 0
        self._page_occupied[pos] = 1
        return result

    def prepare(self, key, packed, stream, epoch):
        if not self._has_expert_cache:
            raise RuntimeError('No expert storage in dense-only DynamicWorkingMemory')
        with self.cv:
            self._check()
            if epoch != self.epoch or (key not in self.desired and key not in self.active_protected):
                return 'stale', None
            if key in self.directory:
                r = self.records[self.directory[key]]
                r['warm_epoch'] = self.epoch
                if key in self.desired:
                    self._set_area(r, self.warm_area)
                self.counts['retained_prepared'] += 1
                return 'retained', r
            record = self._victim(key, set(), optional=True)
            if record is None:
                self.counts['producer_busy'] += 1
                return None, None
            return 'published', self._write(key, packed, record, stream, 'prefetch')

    def _take(self, key, stream, allow_evict, auxiliary, hit):
        record = self.records[self.directory[key]]
        if record['kind'] == 'expert' and record['physical_area_id'] is None:
            self._set_area(record, self.active_area)
        self._page_busy[record['pos']] = 1
        entry = super()._take(key, stream, allow_evict, auxiliary, hit)
        if (not auxiliary and key[0] < self._resident_layer_limit
                and record['bank'] == 0 and not record['retiring']
                and record['ready'] is None and record['kind'] == 'expert'):
            self._pinned_resident_pos[key] = record['pos']
        return entry

    def _on_lease_release(self, record):
        if record['leases'] == 0 and record['kind'] != 'dense':
            self._page_busy[record['pos']] = 0

    def wait_for_space(self, cancelled=lambda: False):
        with self.cv:
            self._check()
            self._reap_retiring_pages()
            act_prot = self.active_protected
            eligible = [
                r for r in self.records
                if r['kind'] != 'dense' and not r['leases']
                and r['key'] not in self.desired and r['key'] not in act_prot
            ]
            if not eligible:
                self._waiters += 1
                try:
                    self.cv.wait_for(lambda: self.closed or cancelled() or any(
                        r['kind'] != 'dense' and not r['leases']
                        and r['key'] not in self.desired and r['key'] not in self.active_protected
                        for r in self.records
                    ))
                finally:
                    self._waiters -= 1
                return
            fences = self._fences(eligible[0])
        if fences:
            fences[0].synchronize()

    def _get_or_create_dense_layout(self, payload):
        signature = tuple((name, tuple(src.shape), src.dtype) for name, src in payload.items())
        entry = self._dense_layouts.get(signature)
        if entry is None:
            layout = SlotArenaLayout.from_templates(payload, slot_alignment=256, tensor_alignment=256)
            layout_id = self._next_layout_id
            self._next_layout_id += 1
            entry = (layout_id, layout)
            self._dense_layouts[signature] = entry
        return entry

    def _get_cached_dense_views(self, offset_bytes, layout_id, layout):
        key = (offset_bytes, layout_id)
        views = self._dense_view_cache.get(key)
        if views is not None:
            self.counts['view_cache_hits'] += 1
            return dict(views)
        self.counts['view_cache_misses'] += 1
        views = layout.bind_slot_views(self.gpu, 0, base_offset_bytes=offset_bytes)
        self._dense_view_cache[key] = views
        return dict(views)

    def bind_dense(self, slot, payload, stream):
        # Actual payload geometry, without reserving the old 2 MiB slot padding.
        for name, src in payload.items():
            spec = slot.layout.specs.get(name)
            if spec is None or src.dtype != spec.dtype or tuple(src.shape) not in (spec.allowed_shapes or (spec.shape,)):
                raise ValueError(f'GPU payload shape/dtype/key mismatch: {name}')
        layout_id, layout = self._get_or_create_dense_layout(payload)
        count = math.ceil(layout.slot_stride / self.page_bytes)
        with self.cv:
            self._check()
            self._reap_retiring_pages()
            if slot.slot_id in self.dense_blocks:
                raise RuntimeError('Dense role still owns a payload')
            if len(self.dense_blocks) >= 2:
                raise RuntimeError('Standby must stay empty: at most current and next dense payloads')
            best = None
            busy = warm = occupied = retiring_cnt = heat = 0
            records = self.records
            page_busy = self._page_busy
            page_occ = self._page_occupied
            desired = self.desired
            for end, record in enumerate(records):
                busy += page_busy[end]
                is_occ = page_occ[end]
                occupied += is_occ
                retiring_cnt += int(record['retiring'])
                if is_occ:
                    warm += int(record['key'] in desired)
                    lu = record['last_used']
                    if lu > 0:
                        heat += lu
                if end >= count:
                    old_idx = end - count
                    old = records[old_idx]
                    busy -= page_busy[old_idx]
                    old_occ = page_occ[old_idx]
                    occupied -= old_occ
                    retiring_cnt -= int(old['retiring'])
                    if old_occ:
                        warm -= int(old['key'] in desired)
                        lu = old['last_used']
                        if lu > 0:
                            heat -= lu
                if end + 1 >= count and not busy:
                    candidate = (warm, occupied, retiring_cnt, heat, end - count + 1)
                    if best is None or candidate < best:
                        best = candidate
                    if occupied == 0 and retiring_cnt == 0:
                        break
            if best is None:
                raise RuntimeError('No contiguous dense space without overwriting a live reader')
            start = best[4]
            target_area = self.active_area if self.dense_active[slot.slot_id] else self.warm_area
            for idx in range(start, start + count):
                record = records[idx]
                for event in self._fences(record):
                    if stream is not None:
                        stream.wait_event(event)
                    else:
                        event.synchronize()
                if record['key'] is not None:
                    self._pinned_resident_pos.pop(record['key'], None)
                    self.directory.pop(record['key'])
                    self.counts['dense_reclaims_expert'] += 1
                if record['retiring'] and idx in self.retiring_pages:
                    self.retiring_pages.remove(idx)
                record.update(key=None, kind='dense', retiring=False, warm_epoch=-1)
                self._set_area(record, target_area)
                page_busy[idx] = 1
                page_occ[idx] = 0
            self.dense_blocks[slot.slot_id] = dict(
                start=start, count=count, payload_bytes=layout.slot_stride,
                layer=slot.current_layer_idx, area=target_area,
            )
            slot.tensors = self._get_cached_dense_views(start * self.page_bytes, layout_id, layout)
            self.pipeline.slot_offsets = tuple(
                self.dense_blocks[s]['start'] * self.page_bytes if s in self.dense_blocks else None
                for s in range(3)
            )
            self.counts['dense_bindings'] += 1
            self.counts['peak_dense_blocks'] = max(self.counts['peak_dense_blocks'], len(self.dense_blocks))
            self._notify_waiters()

    def dense_event(self, bank, kind):
        with self.cv:
            super().dense_event(bank, kind)
            if kind == 'acquire':
                blk = self.dense_blocks.get(bank)
                if blk is not None:
                    blk['area'] = self.active_area
                    for idx in range(blk['start'], blk['start'] + blk['count']):
                        self._set_area(self.records[idx], self.active_area)
            elif kind == 'release':
                slot = self.pipeline.slots[bank]
                block = self.dense_blocks.pop(bank)
                for idx in range(block['start'], block['start'] + block['count']):
                    rec = self.records[idx]
                    self._set_area(rec, None)
                    readers = [slot.compute_done_event] if slot.compute_done_event is not None else []
                    rec.update(
                        kind='retiring', retiring=True, key=None,
                        ready=slot.transfer_done_event, readers=readers, leases=0,
                    )
                    if self._fences(rec):
                        if idx not in self.retiring_pages:
                            self.retiring_pages.append(idx)
                    else:
                        rec.update(kind='free', retiring=False)
                    self._page_busy[idx] = 0
                    self._page_occupied[idx] = 0
                slot.tensors.clear()
                slot.active_tensors.clear()
                slot.loaded_keys.clear()
                self.pipeline.slot_offsets = tuple(
                    self.dense_blocks[s]['start'] * self.page_bytes if s in self.dense_blocks else None
                    for s in range(3)
                )
                self.counts['dense_returns_to_pool'] += 1
                self._notify_waiters()

    def reset_dense(self):
        # Caller has synchronized CUDA before clearing cancelled ownership.
        with self.cv:
            for idx, record in enumerate(self.records):
                if record['kind'] == 'dense':
                    self._set_area(record, None)
                    record.update(kind='free', retiring=False, key=None, ready=None, readers=[], leases=0)
                    self._page_busy[idx] = 0
                    self._page_occupied[idx] = 0
            self.dense_blocks.clear()
            self.dense_active = [False] * 3
            self.pipeline.slot_offsets = (None, None, None)
            for slot in self.pipeline.slots:
                slot.tensors.clear()
            self._notify_waiters()

    def reset_task(self):
        with self.cv:
            self.epoch += 1
            self.desired.clear()
            self.active_protected.clear()
            self._pinned_resident_pos.clear()
            for pos in tuple(self._area_pages[self.warm_area]):
                rec = self.records[pos]
                if rec['kind'] == 'expert':
                    self._set_area(rec, None)
            self._notify_waiters()

    def stats(self):
        with self.cv:
            self._reap_retiring_pages()
            a_only_pages = len(self._area_pages[self.active_area])
            b_pages = len(self._area_pages[self.warm_area])
            c_pages = len(self._area_pages[self.standby_area])
            pending_pages = len(self.retiring_pages)
            a_experts = sum(1 for pos in self._area_pages[self.active_area] if self.records[pos]['kind'] == 'expert')
            b_experts = sum(1 for pos in self._area_pages[self.warm_area] if self.records[pos]['kind'] == 'expert')
            cache_pool_pages = len(self.directory) - a_experts - b_experts
            free_pages = self.num_slots - a_only_pages - cache_pool_pages - b_pages - pending_pages
            assert c_pages == 0 and min(free_pages, pending_pages, cache_pool_pages) >= 0

            a_bytes = (a_only_pages + cache_pool_pages) * self.page_bytes
            b_bytes = b_pages * self.page_bytes
            free = free_pages * self.page_bytes
            pending = pending_pages * self.page_bytes
            physical_area_roles = {
                self.active_area: 'A',
                self.warm_area: 'B',
                self.standby_area: 'C',
            }
            roles = dict(
                A=dict(area=self.active_area, used_bytes=a_bytes,
                       active_only_bytes=a_only_pages * self.page_bytes,
                       cache_pool_bytes=cache_pool_pages * self.page_bytes,
                       available_bytes=a_bytes + free),
                B=dict(area=self.warm_area, used_bytes=b_bytes,
                       expert_limit_bytes=self.warm_capacity * self.page_bytes),
                C=dict(area=self.standby_area, used_bytes=0, reserved_bytes=0),
            )
            return dict(
                policy='dynamic_A_current_B_prepare_C_standby', **dict(self.counts),
                budget_bytes=self.budget_bytes, gpu_bytes=self.gpu_bytes, pinned_bytes=self.pinned_bytes,
                temporary_reserve_bytes=self.temporary_reserve_bytes, page_bytes=self.page_bytes,
                num_slots=self.num_slots, live_experts=len(self.directory), roles=roles,
                physical_area_roles=physical_area_roles,
                area_transitions=dict(self.area_transitions),
                cache_pool_experts=cache_pool_pages,
                retiring_pages=pending_pages,
                free_bytes=free, retired_waiting_cuda_bytes=pending,
                dense_blocks=len(self.dense_blocks), dense_events=[dict(c) for c in self.area_dense],
                area_reads=list(self.area_reads), area_writes=list(self.area_writes),
                active_leases=sum(r['leases'] for r in self.records), warm_bank_to_cache_copies=0,
                allocation_ptr=self.gpu.data_ptr() if self.gpu is not None else None,
                events_created=self.event_pool.created, events_reused=self.event_pool.reused,
                epoch=self.epoch, closed=self.closed,
            )

    def close(self):
        self._pinned_resident_pos.clear()
        self._dense_view_cache.clear()
        self._dense_layouts.clear()
        self.retiring_pages.clear()
        super().close()
