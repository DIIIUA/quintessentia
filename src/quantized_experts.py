"""Selective Q4/Q6 expert cache: only routed experts are read and converted.

The byte budget includes space for bounded conversion/dequantization temporaries.
Prefill uses the same packed representation as decode and preserves the existing
anti-thrashing rule. Unsupported expert mathematics stays on the source path.
"""
from collections import OrderedDict
import math
import threading
import torch
from src.quantized_projection import pack_weight, unpack_weight, packed_specs


def _slice_gu_part(gu, r0, r1):
    out = {}
    for k, v in gu.items():
        if k == "col_scale":
            out[k] = v
        elif v.ndim == 3:
            out[k] = v[:, r0:r1]
        else:
            out[k] = v[r0:r1]
    return out


class QuantizedExpertCache:
    def __init__(self, base, bits, budget_bytes, group_size=64, *, q1_planes=1):
        self.base, self.bits, self.group_size = base, bits, group_size
        self.q1_planes = max(1, int(q1_planes)) if bits == 1 else int(bits)
        self._budget_bytes = int(budget_bytes)
        self._total_experts = 0
        self.device = base.device
        self.enabled_layers = set()
        self.cpu_quant_layers = set()
        self._cpu_packed = OrderedDict()
        self._cpu_lock = threading.Lock()
        self.cpu_packed_bytes = 0
        if base.device.type == "cpu":
            self.cpu_packed_budget = max(0, int(budget_bytes))
        elif base.device.type == "cuda" and self.bits in (1, 2):
            self.cpu_packed_budget = min(256 * 1024**2, max(64 * 1024**2, int(budget_bytes) // 2))
        else:
            self.cpu_packed_budget = 0
        self.cpu_startup_reuses = 0
        self._lru = OrderedDict()
        self.hits = self.misses = self.conversions = self.fused_calls = self.prefill_batches = 0
        self.fallback_calls = 0
        self.warmup = None
        self.shared_memory = None
        self._observed_routes = None
        self.auxiliary_depth = 0
        self.verification_depth = 0
        self.warm_hits = 0
        self.gu = self.dn = None
        self.num_slots = self._initial_num_slots = 0
        self.inter_buf = self.part_buf = None
        if hasattr(base, "_layer_meta"):
            shapes = {(tuple(m["gate"]["torch_shape"][-2:]), tuple(m["up"]["torch_shape"][-2:]), tuple(m["down"]["torch_shape"][-2:])) for m in base._layer_meta.values()}
            total = sum(m["num_experts"] for m in base._layer_meta.values())
            if len(shapes) != 1:
                return
            gate, up, down = shapes.pop()
            self.I, self.H = gate
            if up != gate or down != (self.H, self.I):
                return
        else:
            if not getattr(base, "uniform_shapes", False) or not base.has_gate:
                return
            self.I, self.H = base.inter_dim, base.hidden_dim
            total = len(base._spans)
        if self.device.type != "cuda" or self.I % group_size or self.H % group_size:
            return
        self._total_experts = int(total)
        specs_gu = packed_specs("weight", (2*self.I, self.H), bits, group_size, q1_planes=self.q1_planes)
        specs_dn = packed_specs("weight", (self.H, self.I), bits, group_size, q1_planes=self.q1_planes)
        self.expert_bytes = sum(torch.empty((), dtype=dt).element_size() * math.prod(shape)
                                for shape, dt in list(specs_gu.values()) + list(specs_dn.values()))
        # Eight BF16 expert results plus bounded FP32 conversion tiles/workspaces.
        self.temporary_reserve_bytes = 8 * 6 * self.I * self.H + 32 * 1024**2
        available = max(0, int(budget_bytes) - self.temporary_reserve_bytes)
        slots = min(total, available // self.expert_bytes)
        if slots < 8:
            return
        self.num_slots = self._initial_num_slots = slots
        self.gu = {k: torch.empty((slots, *sh), dtype=dt, device=self.device) for k, (sh, dt) in specs_gu.items()}
        self.dn = {k: torch.empty((slots, *sh), dtype=dt, device=self.device) for k, (sh, dt) in specs_dn.items()}
        self.inter_buf = torch.empty((8, self.I), dtype=torch.float32, device=self.device)
        self.part_buf = torch.empty((8, self.H), dtype=torch.float32, device=self.device)

    def set_q1_planes(self, q1_planes):
        if self._lru or self.hits or self.misses or self.shared_memory is not None:
            raise RuntimeError("set_q1_planes must be called before expert cache use")
        self.q1_planes = max(1, int(q1_planes)) if self.bits == 1 else int(self.bits)
        planes = getattr(self, "q1_planes", 1)
        if self.bits != 1 or self.device.type != "cuda" or not hasattr(self, "I") or self.I % self.group_size or self.H % self.group_size:
            return
        specs_gu = packed_specs("weight", (2 * self.I, self.H), 1, self.group_size, q1_planes=planes)
        specs_dn = packed_specs("weight", (self.H, self.I), 1, self.group_size, q1_planes=planes)
        self.expert_bytes = sum(
            torch.empty((), dtype=dt).element_size() * math.prod(shape)
            for shape, dt in list(specs_gu.values()) + list(specs_dn.values())
        )
        available = max(0, self._budget_bytes - self.temporary_reserve_bytes)
        slots = min(self._total_experts, available // self.expert_bytes)
        if slots < 8:
            self.num_slots = self._initial_num_slots = 0
            self.gu = self.dn = None
            self.enabled_layers.clear()
            return
        self.num_slots = self._initial_num_slots = slots
        self.gu = {k: torch.empty((slots, *sh), dtype=dt, device=self.device) for k, (sh, dt) in specs_gu.items()}
        self.dn = {k: torch.empty((slots, *sh), dtype=dt, device=self.device) for k, (sh, dt) in specs_dn.items()}
        if self.inter_buf is None:
            self.inter_buf = torch.empty((8, self.I), dtype=torch.float32, device=self.device)
        if self.part_buf is None:
            self.part_buf = torch.empty((8, self.H), dtype=torch.float32, device=self.device)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def configure_layer(self, layer, module, has_bias=False, input_weighted=False):
        from transformers.activations import SiLUActivation
        act = getattr(module, "act_fn", None)
        silu = type(act) in (torch.nn.SiLU, SiLUActivation) or act is torch.nn.functional.silu
        custom = getattr(type(module), "_apply_gate", None)
        custom_ok = custom is None or getattr(custom, "__name__", "") == "_default_apply_gate"
        if silu and custom_ok and not has_bias and not input_weighted and not hasattr(module, "alpha") and not getattr(module, "has_post_expert_norm", False):
            if self.num_slots:
                self.enabled_layers.add(layer)
            if hasattr(self, "H") and self.H % self.group_size == 0 and self.I % self.group_size == 0:
                self.cpu_quant_layers.add(layer)

    def _take_cpu_seed(self, layer, eid):
        keep_cpu = self.bits in (1, 2)
        with self._cpu_lock:
            if keep_cpu:
                seed = self._cpu_packed.get((layer, eid))
                if seed is not None:
                    self._cpu_packed.move_to_end((layer, eid))
                    self.cpu_startup_reuses += 1
            else:
                seed = self._cpu_packed.pop((layer, eid), None)
                if seed is not None:
                    self.cpu_packed_bytes -= self._packed_bytes(seed)
                    self.cpu_startup_reuses += 1
        if seed is not None:
            return tuple({k: v.to(self.device) for k, v in part.items()} for part in seed)
        return None

    def _admit_packed(self, key, gu, dn, allow_evict, protected):
        slot = None
        if len(self._lru) < self.num_slots:
            slot = len(self._lru)
        elif allow_evict:
            victim = next((k for k in self._lru if k not in protected), None)
            if victim is not None:
                slot = self._lru.pop(victim)
        if slot is not None:
            for k, v in gu.items():
                self.gu[k][slot].copy_(v)
            for k, v in dn.items():
                self.dn[k][slot].copy_(v)
            self._lru[key] = slot
        return gu, dn, slot

    def _ensure_batch(self, layer, ids, allow_evict, protected=frozenset()):
        """Resolve hits/warmup/CPU seeds first, then load & pack all disk misses in one sorted batch."""
        ids = [int(e) for e in ids]
        results = [None] * len(ids)
        disk_misses = {}
        pending_idx = []
        for idx, eid in enumerate(ids):
            key = (layer, eid)
            if key in self._lru:
                self.hits += 1
                if allow_evict:
                    self._lru.move_to_end(key)
                slot = self._lru[key]
                results[idx] = ({k: v[slot] for k, v in self.gu.items()}, {k: v[slot] for k, v in self.dn.items()}, slot)
                continue
            self.misses += 1
            # The producer only writes its own bounded banks. Admission and LRU
            # mutation remain on the normal compute thread, behind a reader fence.
            if self.warmup is not None and allow_evict:
                victim = None
                target = len(self._lru) if len(self._lru) < self.num_slots else None
                if target is None:
                    victim = next((k for k in self._lru if k not in protected), None)
                    target = self._lru.get(victim)
                if target is not None:
                    gu_dst = {k: v[target] for k, v in self.gu.items()}
                    dn_dst = {k: v[target] for k, v in self.dn.items()}
                    if self.warmup.try_copy(layer, eid, (gu_dst, dn_dst)):
                        if victim is not None:
                            self._lru.pop(victim)
                        self._lru[key] = target
                        self.warm_hits += 1
                        results[idx] = (gu_dst, dn_dst, target)
                        continue
            seed = self._take_cpu_seed(layer, eid)
            if seed is not None:
                results[idx] = self._admit_packed(key, seed[0], seed[1], allow_evict, protected)
                continue
            pending_idx.append(idx)
            if eid not in disk_misses:
                disk_misses[eid] = None
        if disk_misses:
            miss_sorted = sorted(disk_misses)
            g, u, d = self.base.load_batch(layer, miss_sorted, self.device, torch.bfloat16, False)
            keep_cpu = self.bits in (1, 2)
            planes = getattr(self, "q1_planes", 1)
            if self.bits in (4, 6, 8) and self.device.type == "cuda":
                from src.colibri_q1_cube_nvrtc import launch_colibri_pack_affine
                m_cnt = len(miss_sorted)
                self.conversions += m_cnt
                gu_cat = torch.cat((g, u), dim=1).contiguous()
                d_c = d.contiguous()
                gu_specs = packed_specs("weight", (2 * self.I, self.H), self.bits, self.group_size)
                dn_specs = packed_specs("weight", (self.H, self.I), self.bits, self.group_size)
                gu_p = {k: torch.empty((m_cnt, *sh), dtype=dt, device=self.device) for k, (sh, dt) in gu_specs.items()}
                dn_p = {k: torch.empty((m_cnt, *sh), dtype=dt, device=self.device) for k, (sh, dt) in dn_specs.items()}
                launch_colibri_pack_affine(
                    gu_cat, gu_p["packed_lo"], gu_p.get("packed_hi"), gu_p["qscale"], gu_p["qzero"],
                    bits=self.bits, group_size=self.group_size,
                )
                launch_colibri_pack_affine(
                    d_c, dn_p["packed_lo"], dn_p.get("packed_hi"), dn_p["qscale"], dn_p["qzero"],
                    bits=self.bits, group_size=self.group_size,
                )
                del gu_cat, d_c
                for j, eid in enumerate(miss_sorted):
                    disk_misses[eid] = (
                        {k: v[j] for k, v in gu_p.items()},
                        {k: v[j] for k, v in dn_p.items()},
                    )
            else:
                for j, eid in enumerate(miss_sorted):
                    self.conversions += 1
                    packed = (
                        pack_weight(torch.cat((g[j], u[j])), self.bits, self.group_size, q1_planes=planes),
                        pack_weight(d[j], self.bits, self.group_size, q1_planes=planes),
                    )
                    if keep_cpu and self.cpu_packed_budget > 0:
                        cpu_packed = tuple({k: v.detach().cpu() for k, v in part.items()} for part in packed)
                        self._remember_cpu((layer, eid), cpu_packed)
                    disk_misses[eid] = packed
            del g, u, d
            for idx in pending_idx:
                eid = ids[idx]
                key = (layer, eid)
                if key in self._lru:
                    if allow_evict:
                        self._lru.move_to_end(key)
                    slot = self._lru[key]
                    results[idx] = ({k: v[slot] for k, v in self.gu.items()}, {k: v[slot] for k, v in self.dn.items()}, slot)
                else:
                    gu, dn = disk_misses[eid]
                    results[idx] = self._admit_packed(key, gu, dn, allow_evict, protected)
        return results

    def _get(self, layer, eid, allow_evict, protected=frozenset()):
        return self._ensure_batch(layer, [eid], allow_evict, protected)[0]

    def load_batch(self, layer, ids, device, dtype, allow_evict):
        if torch.device(device).type == "cpu" and layer in self.cpu_quant_layers:
            g, u, d = self.base.load_batch(layer, ids, device, torch.bfloat16, allow_evict)
            planes = getattr(self, "q1_planes", 1)
            # Same packed representation as the GPU cache, bounded to the
            # dispatch batch; CPU startup must not silently use different weights.
            for i in range(len(ids)):
                packed = (pack_weight(torch.cat((g[i], u[i])), self.bits, self.group_size, q1_planes=planes),
                          pack_weight(d[i], self.bits, self.group_size, q1_planes=planes))
                if not self.auxiliary_depth:
                    self._remember_cpu((layer, ids[i]), packed)
                gu = unpack_weight(packed[0], self.group_size)
                dn = unpack_weight(packed[1], self.group_size)
                g[i].copy_(gu[:self.I]); u[i].copy_(gu[self.I:]); d[i].copy_(dn)
            return g.to(dtype), u.to(dtype), d.to(dtype)
        if layer not in self.enabled_layers or torch.device(device) != self.device:
            self.fallback_calls += 1
            return self.base.load_batch(layer, ids, device, dtype, allow_evict)
        if len(ids) > 8:
            raise ValueError("Expert dequantization batch exceeds the bounded workspace")
        self.prefill_batches += 1
        if self.shared_memory is not None:
            g = torch.empty((len(ids), self.I, self.H), dtype=dtype, device=device)
            u = torch.empty_like(g)
            d = torch.empty((len(ids), self.H, self.I), dtype=dtype, device=device)
            with self.shared_memory.lease(layer, ids, self._load_packed, allow_evict,
                                          auxiliary=bool(self.auxiliary_depth)) as entries:
                for i, entry in enumerate(entries):
                    gu, dn = entry['gu'], entry['dn']
                    g[i].copy_(unpack_weight(_slice_gu_part(gu, 0, self.I), self.group_size, dtype))
                    u[i].copy_(unpack_weight(_slice_gu_part(gu, self.I, 2 * self.I), self.group_size, dtype))
                    d[i].copy_(unpack_weight(dn, self.group_size, dtype))
            return g, u, d
        # Fast uncached prefill path: when LRU is full and eviction is disallowed,
        # quantize+dequantize misses directly via fused NVRTC without packing/unpacking.
        if (
            not allow_evict
            and len(self._lru) >= self.num_slots
            and self.bits in (4, 6, 8)
            and self.device.type == "cuda"
            and dtype in (torch.bfloat16, torch.float32)
        ):
            from src.colibri_q1_cube_nvrtc import launch_colibri_quant_dequant_affine, launch_colibri_unpack_affine
            int_ids = [int(e) for e in ids]
            hit_entries = [None] * len(int_ids)
            miss_order = []
            miss_seen = set()
            pending_idx = []
            for idx, eid in enumerate(int_ids):
                key = (layer, eid)
                if key in self._lru:
                    self.hits += 1
                    slot = self._lru[key]
                    hit_entries[idx] = (
                        {k: v[slot] for k, v in self.gu.items()},
                        {k: v[slot] for k, v in self.dn.items()},
                    )
                    continue
                self.misses += 1
                seed = self._take_cpu_seed(layer, eid)
                if seed is not None:
                    hit_entries[idx] = seed
                    continue
                pending_idx.append(idx)
                if eid not in miss_seen:
                    miss_seen.add(eid)
                    miss_order.append(eid)
            miss_sorted = sorted(miss_order)
            mg_o = mu_o = md_o = None
            if miss_sorted:
                mg, mu, md = self.base.load_batch(layer, miss_sorted, self.device, torch.bfloat16, False)
                self.conversions += len(miss_sorted)
                if dtype == torch.bfloat16:
                    mg_o, mu_o, md_o = mg.contiguous(), mu.contiguous(), md.contiguous()
                    launch_colibri_quant_dequant_affine(mg_o, mg_o, bits=self.bits, group_size=self.group_size)
                    launch_colibri_quant_dequant_affine(mu_o, mu_o, bits=self.bits, group_size=self.group_size)
                    launch_colibri_quant_dequant_affine(md_o, md_o, bits=self.bits, group_size=self.group_size)
                else:
                    mg_o = torch.empty_like(mg, dtype=dtype)
                    mu_o = torch.empty_like(mu, dtype=dtype)
                    md_o = torch.empty_like(md, dtype=dtype)
                    launch_colibri_quant_dequant_affine(mg, mg_o, bits=self.bits, group_size=self.group_size)
                    launch_colibri_quant_dequant_affine(mu, mu_o, bits=self.bits, group_size=self.group_size)
                    launch_colibri_quant_dequant_affine(md, md_o, bits=self.bits, group_size=self.group_size)
                del mg, mu, md
                if len(pending_idx) == len(int_ids) and int_ids == miss_sorted:
                    return mg_o, mu_o, md_o
            g = torch.empty((len(int_ids), self.I, self.H), dtype=dtype, device=device)
            u = torch.empty_like(g)
            d = torch.empty((len(int_ids), self.H, self.I), dtype=dtype, device=device)
            miss_pos = {eid: j for j, eid in enumerate(miss_sorted)}
            for i, eid in enumerate(int_ids):
                hit = hit_entries[i]
                if hit is not None:
                    gu, dn = hit
                    g_part = _slice_gu_part(gu, 0, self.I)
                    u_part = _slice_gu_part(gu, self.I, 2 * self.I)
                    launch_colibri_unpack_affine(
                        g_part["packed_lo"], g_part.get("packed_hi"), g_part["qscale"], g_part["qzero"],
                        g[i], bits=self.bits, group_size=self.group_size,
                    )
                    launch_colibri_unpack_affine(
                        u_part["packed_lo"], u_part.get("packed_hi"), u_part["qscale"], u_part["qzero"],
                        u[i], bits=self.bits, group_size=self.group_size,
                    )
                    launch_colibri_unpack_affine(
                        dn["packed_lo"], dn.get("packed_hi"), dn["qscale"], dn["qzero"],
                        d[i], bits=self.bits, group_size=self.group_size,
                    )
                else:
                    j = miss_pos[eid]
                    g[i].copy_(mg_o[j])
                    u[i].copy_(mu_o[j])
                    d[i].copy_(md_o[j])
            return g, u, d
        protected = {(layer, int(eid)) for eid in ids}
        entries = self._ensure_batch(layer, ids, allow_evict, protected)
        g = torch.empty((len(ids), self.I, self.H), dtype=dtype, device=device)
        u = torch.empty_like(g)
        d = torch.empty((len(ids), self.H, self.I), dtype=dtype, device=device)
        if self.bits in (4, 6, 8) and self.device.type == "cuda" and dtype in (torch.bfloat16, torch.float32):
            from src.colibri_q1_cube_nvrtc import launch_colibri_unpack_affine
            for i, (gu, dn, _) in enumerate(entries):
                g_part = _slice_gu_part(gu, 0, self.I)
                u_part = _slice_gu_part(gu, self.I, 2 * self.I)
                launch_colibri_unpack_affine(
                    g_part["packed_lo"], g_part.get("packed_hi"), g_part["qscale"], g_part["qzero"],
                    g[i], bits=self.bits, group_size=self.group_size,
                )
                launch_colibri_unpack_affine(
                    u_part["packed_lo"], u_part.get("packed_hi"), u_part["qscale"], u_part["qzero"],
                    u[i], bits=self.bits, group_size=self.group_size,
                )
                launch_colibri_unpack_affine(
                    dn["packed_lo"], dn.get("packed_hi"), dn["qscale"], dn["qzero"],
                    d[i], bits=self.bits, group_size=self.group_size,
                )
            return g, u, d
        for i, (gu, dn, _) in enumerate(entries):
            # Decode one projection at a time, not the complete expert batch.
            g[i].copy_(unpack_weight(_slice_gu_part(gu, 0, self.I), self.group_size, dtype))
            u[i].copy_(unpack_weight(_slice_gu_part(gu, self.I, 2 * self.I), self.group_size, dtype))
            d[i].copy_(unpack_weight(dn, self.group_size, dtype))
        return g, u, d

    def _load_packed(self, layer, eid):
        keep_cpu = self.bits in (1, 2)
        with self._cpu_lock:
            if keep_cpu:
                seed = self._cpu_packed.get((layer, eid))
                if seed is not None:
                    self._cpu_packed.move_to_end((layer, eid))
                    self.cpu_startup_reuses += 1
            else:
                seed = self._cpu_packed.pop((layer, eid), None)
                if seed is not None:
                    self.cpu_packed_bytes -= self._packed_bytes(seed)
                    self.cpu_startup_reuses += 1
        if seed is not None:
            return tuple({k: v.to(self.device) for k, v in part.items()} for part in seed)
        g, u, d = self.base.load_batch(layer, [eid], self.device, torch.bfloat16, False)
        self.conversions += 1
        planes = getattr(self, "q1_planes", 1)
        packed = (pack_weight(torch.cat((g[0], u[0])), self.bits, self.group_size, q1_planes=planes),
                  pack_weight(d[0], self.bits, self.group_size, q1_planes=planes))
        if keep_cpu and self.cpu_packed_budget > 0:
            cpu_packed = tuple({k: v.detach().cpu() for k, v in part.items()} for part in packed)
            self._remember_cpu((layer, eid), cpu_packed)
        return packed

    @staticmethod
    def _packed_bytes(entry):
        return sum(t.numel()*t.element_size() for part in entry for t in part.values())

    def _remember_cpu(self, key, entry):
        size = self._packed_bytes(entry)
        if size > self.cpu_packed_budget:
            return
        with self._cpu_lock:
            old = self._cpu_packed.pop(key, None)
            if old is not None:
                self.cpu_packed_bytes -= self._packed_bytes(old)
            while self._cpu_packed and self.cpu_packed_bytes+size > self.cpu_packed_budget:
                self.cpu_packed_bytes -= self._packed_bytes(self._cpu_packed.popitem(last=False)[1])
            self._cpu_packed[key] = entry
            self.cpu_packed_bytes += size

    def detach_cpu_packed(self):
        with self._cpu_lock:
            entries, self._cpu_packed = self._cpu_packed, OrderedDict()
            self.cpu_packed_bytes = 0
        return (self.bits, self.group_size, self.cpu_packed_budget, entries)

    def adopt_cpu_packed(self, packet):
        bits, group, budget, entries = packet
        if bits != self.bits or group != self.group_size:
            raise ValueError("CPU expert seed quantization differs from GPU storage")
        if self.bits == 1 and entries:
            first_entry = next(iter(entries.values()))
            lo = first_entry[0].get("packed_lo") if isinstance(first_entry, (tuple, list)) and first_entry else None
            p_seed = int(lo.shape[0]) if lo is not None and lo.ndim == 3 else 1
            cur_planes = getattr(self, "q1_planes", 1)
            if p_seed != cur_planes:
                if not self._lru and self.hits == 0 and self.misses == 0:
                    self.set_q1_planes(p_seed)
                else:
                    raise ValueError(f"CPU expert seed q1_planes={p_seed} differs from GPU storage q1_planes={cur_planes}")
        with self._cpu_lock:
            self._cpu_packed = entries
            if self.bits in (1, 2) and self.device.type == "cuda":
                self.cpu_packed_budget = max(self.cpu_packed_budget, int(budget))
            else:
                self.cpu_packed_budget = budget
            self.cpu_packed_bytes = sum(self._packed_bytes(v) for v in entries.values())

    def observed_route_rows(self, layer, indices):
        """Consume the warmup observer's CPU route copy exactly once."""
        observed = self._observed_routes
        self._observed_routes = None
        if observed is not None and observed[0] == layer and observed[1] is indices:
            return observed[2]
        return None

    def _fused_shared(self, layer, hidden, valid, weights):
        from src.colibri_nvrtc import fused_routed_moe_q4, fused_routed_moe_q6, reduce_routed_parts
        memory = self.shared_memory
        lim = getattr(memory, '_resident_layer_limit', 0)
        pinned = getattr(memory, '_pinned_resident_pos', None)
        if (lim > 0 and layer < lim and pinned is not None
                and not self.auxiliary_depth and not self.verification_depth):
            positions = []
            for _, eid in valid:
                pos = pinned.get((layer, eid))
                if pos is None:
                    positions = None
                    break
                positions.append(pos)
            if positions is not None:
                gu, dn = memory.banks[0]
                idx = torch.tensor(positions, dtype=torch.int64, device=self.device)
                route = weights[0, [pos for pos, _ in valid]].contiguous()
                args = (hidden, idx, route, gu['packed_lo'])
                if self.bits == 4:
                    out = fused_routed_moe_q4(*args, gu['qscale'], gu['qzero'],
                        dn['packed_lo'], dn['qscale'], dn['qzero'],
                        self.inter_buf, self.part_buf, self.group_size, reduce=True)
                else:
                    out = fused_routed_moe_q6(*args, gu['packed_hi'], gu['qscale'], gu['qzero'],
                        dn['packed_lo'], dn['packed_hi'], dn['qscale'], dn['qzero'],
                        self.inter_buf, self.part_buf, self.group_size, reduce=True)
                memory.counts['hits'] += len(valid)
                memory.counts['leases'] += len(valid)
                memory.counts['fused_groups'] += 1
                return out
        with memory.lease(layer, [eid for _, eid in valid], self._load_packed,
                          auxiliary=bool(self.auxiliary_depth)) as entries:
            if self.verification_depth:
                # Verification really consumes prepared weights, even though
                # its tentative routes must not age or evict the task cache.
                with memory.cv:
                    for entry in entries:
                        record = entry['record']
                        if record is not None and record['warm_epoch'] == memory.epoch:
                            memory.counts['verification_warm_direct_hits'] += 1
                            memory.counts[f'verification_warm_area_{memory.area_for(record)}'] += 1
            # Keep the original routing order and FP32 contributions. Only the
            # addresses change; reduce once, with the original native kernel.
            start = 0
            while start < len(entries):
                bank = entries[start]['bank']
                end = start+1
                while end < len(entries) and entries[end]['bank'] == bank:
                    end += 1
                if bank is None:
                    # Bounded uncached fallback (e.g. a protected auxiliary
                    # working set). A standalone expert is a one-entry bank.
                    end = start+1
                    gu = {k:v.unsqueeze(0) for k,v in entries[start]['gu'].items()}
                    dn = {k:v.unsqueeze(0) for k,v in entries[start]['dn'].items()}
                    positions = [0]
                else:
                    gu, dn = memory.banks[bank]
                    positions = [e['pos'] for e in entries[start:end]]
                single_group = start == 0 and end == len(entries) and bank is not None
                idx = torch.tensor(positions, dtype=torch.int64, device=self.device)
                route = weights[0, [pos for pos,_ in valid[start:end]]].contiguous()
                args = (hidden, idx, route, gu['packed_lo'])
                reduce_arg = True if single_group else None
                inter_slice = self.inter_buf if single_group else self.inter_buf[start:end]
                part_slice = self.part_buf if single_group else self.part_buf[start:end]
                if self.bits == 4:
                    out = fused_routed_moe_q4(*args, gu['qscale'], gu['qzero'],
                        dn['packed_lo'], dn['qscale'], dn['qzero'],
                        inter_slice, part_slice, self.group_size, reduce=reduce_arg)
                else:
                    out = fused_routed_moe_q6(*args, gu['packed_hi'], gu['qscale'], gu['qzero'],
                        dn['packed_lo'], dn['packed_hi'], dn['qscale'], dn['qzero'],
                        inter_slice, part_slice, self.group_size, reduce=reduce_arg)
                memory.counts['fused_groups'] += 1
                if single_group:
                    return out
                start = end
            return reduce_routed_parts(self.part_buf[:len(entries)])

    def try_fused(self, layer, hidden, indices, weights, num_experts):
        if self.bits not in (4, 6) or layer not in self.enabled_layers or hidden.shape[0] != 1 or hidden.dtype != torch.bfloat16:
            return None
        if (self.H//self.group_size)*4 + self.H*2 > 49152 or (self.I+self.I//self.group_size)*4 > 49152:
            return None
        observed = self.observed_route_rows(layer, indices)
        ids = observed[0] if observed is not None else indices[0].tolist()
        valid = [(pos, int(eid)) for pos, eid in enumerate(ids) if 0 <= int(eid) < num_experts]
        if not valid:
            return torch.zeros_like(hidden)
        if len(valid) > 8:
            return None
        if self.shared_memory is not None:
            self.fused_calls += 1
            return self._fused_shared(layer, hidden, valid, weights)
        protected = {(layer, eid) for _, eid in valid}
        slots = [entry[2] for entry in self._ensure_batch(layer, [eid for _, eid in valid], True, protected)]
        if any(s is None for s in slots):
            return None
        slot_idx = torch.tensor(slots, device=self.device, dtype=torch.int64)
        route_weights = weights[0, [pos for pos, _ in valid]].contiguous()
        from src.colibri_nvrtc import fused_routed_moe_q4, fused_routed_moe_q6
        self.fused_calls += 1
        gu, dn = self.gu, self.dn
        if self.bits == 4:
            return fused_routed_moe_q4(hidden, slot_idx, route_weights, gu["packed_lo"], gu["qscale"], gu["qzero"], dn["packed_lo"], dn["qscale"], dn["qzero"], self.inter_buf, self.part_buf, self.group_size)
        return fused_routed_moe_q6(hidden, slot_idx, route_weights, gu["packed_lo"], gu["packed_hi"], gu["qscale"], gu["qzero"], dn["packed_lo"], dn["packed_hi"], dn["qscale"], dn["qzero"], self.inter_buf, self.part_buf, self.group_size)

    def stats(self):
        size = sum(v.numel()*v.element_size() for bank in (self.gu, self.dn) if bank for v in bank.values())
        hits, misses = self.hits, self.misses
        slots = self.num_slots
        sm = self.shared_memory.stats() if self.shared_memory is not None else None
        if sm is not None:
            slots = int(sm.get("num_slots", slots))
            if not size:
                size = int(sm.get("gpu_bytes", 0))
            hits += int(sm.get("hits", 0) or (sm.get("cache_hits", 0) + sm.get("warm_direct_hits", 0)))
            misses += int(sm.get("misses", 0) or sm.get("demand_writes", 0))
        return {"format": f"colibri_q{self.bits}", "slots": slots, "arena_mib": round(size/1024**2, 2),
                "enabled_layers": sorted(self.enabled_layers), "hits": hits, "misses": misses,
                "hit_rate": hits/max(1, hits+misses), "conversions": self.conversions,
                "fused_calls": self.fused_calls, "prefill_batches": self.prefill_batches,
                "fallback_calls": self.fallback_calls, "warm_hits": self.warm_hits,
                "cpu_startup_reuses": self.cpu_startup_reuses, "cpu_packed_bytes": self.cpu_packed_bytes,
                "cpu_packed_budget": self.cpu_packed_budget,
                "shared_memory": sm,
                "source": self.base.stats()}

    def close(self):
        with self._cpu_lock:
            self._cpu_packed.clear()
            self.cpu_packed_bytes = 0
        self._observed_routes = None
        if self.warmup is not None:
            self.warmup.close()
            self.warmup = None
        self._lru.clear()
        self.gu = self.dn = self.inter_buf = self.part_buf = None
        if self.shared_memory is not None:
            self.shared_memory.close()
            self.shared_memory = None
        self.base.close()
