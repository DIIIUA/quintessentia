"""Method 1 survey while the reference engine streams one layer at a time.

No full model copy is retained. Each visited projection is split completely,
and every cumulative prefix is measured against its source operation. The
result is a Method 2 selection, driven by an error target, not a fixed k=4.
The positional representation is a quantizer; root_bits=16 does not encode
IEEE BF16 bit patterns losslessly. Its residual error is measured too.
"""
from __future__ import annotations

from collections import Counter
import math

import torch
from torch import nn
import torch.nn.functional as F

from src.binary_halving_q1 import (
    _is_router_module, _plane_scale_f32, _unpack_sign_bits_f32,
    carve_q1_cubes_incremental,
)
from src.ideal_q1_engine import fwht_block


@torch.no_grad()
def projection_prefix_errors(weight, inputs, bias=None, *, root_bits=16, group_size=64):
    """Measure Q1, Q1+Q1, ... from the same complete tree and the same input."""
    tree, meta = carve_q1_cubes_incremental(weight, root_bits, root_bits=root_bits,
                                           group_size=group_size)
    x = inputs.reshape(-1, inputs.shape[-1]).float()
    reference = F.linear(x, weight.float(), None if bias is None else bias.float())
    denominator = reference.square().sum().clamp_min(1e-20)
    xr = fwht_block(x * tree['col_scale'].float(), group_size)
    rows = tree['row_scale'].float()
    n, k = weight.shape
    grouped = xr.reshape(xr.shape[0], k // group_size, group_size).sum(-1)
    acc = F.linear(grouped, tree['qzero'].float()) * rows
    if bias is not None:
        acc = acc + bias.float()
    errors = []
    for p in range(root_bits):
        signs = _unpack_sign_bits_f32(tree['packed_lo'][p]).reshape(n, k // group_size, group_size)
        wp = (signs * _plane_scale_f32(tree['qscale'], p).unsqueeze(-1)).reshape(n, k)
        acc = acc + F.linear(xr, wp) * rows
        errors.append(((reference - acc).square().sum() / denominator).sqrt())
    measured = torch.stack(errors)
    if not bool(torch.isfinite(measured).all()):
        raise FloatingPointError('Non-finite output in the complete cube survey')
    return measured.cpu().tolist()


class StreamingCubeSurvey:
    """Observe loaded reference projections; release temporary trees per call."""

    def __init__(self, layers, *, root_bits=16, max_input_tokens=6, progress=None):
        if root_bits not in (2, 4, 8, 16):
            raise ValueError('Unsupported complete tree depth')
        self.layers = layers
        self.root_bits = root_bits
        self.max_input_tokens = max_input_tokens
        self.progress = progress
        self.errors = {}
        self.module_specs = []
        self.handles = []
        self.expert_errors = {}
        self._moe_inputs = {}
        self._restore = []

    def observe_moe(self, engine):
        """Inspect the already loaded expert batch; do not reread expert weights."""
        source = getattr(engine, 'model_source', None)
        loader = getattr(source, '_expert_loader', None)
        if loader is None or not hasattr(loader, 'load_batch'):
            return
        containers = getattr(source, '_routed_expert_container', {})
        for li, layer in enumerate(self.layers):
            name = containers.get(li, 'mlp.experts')
            try:
                experts = layer.get_submodule(name)
            except AttributeError:
                continue
            def before(_mod, args, kwargs, layer=li):
                values = list(args) + list(kwargs.values())
                h = next((x for x in values if isinstance(x, torch.Tensor) and x.is_floating_point() and x.ndim >= 2), None)
                idx = next((x for x in values if isinstance(x, torch.Tensor) and x.dtype in (torch.int64, torch.int32) and x.ndim == 2), None)
                if h is not None and idx is not None:
                    self._moe_inputs[layer] = (h.reshape(-1, h.shape[-1]), idx)
            def after(_mod, _args, _out, layer=li):
                self._moe_inputs.pop(layer, None)
            self.handles.append(experts.register_forward_pre_hook(before, with_kwargs=True))
            self.handles.append(experts.register_forward_hook(after))
        original = loader.load_batch
        self._restore.append((loader, 'load_batch', original, 'load_batch' in vars(loader)))
        if callable(getattr(loader, 'try_fused', None)):
            # The reference survey needs the decoded weights and their actual
            # inputs. A native packed fast-path may otherwise skip load_batch.
            fused = loader.try_fused
            self._restore.append((loader, 'try_fused', fused, 'try_fused' in vars(loader)))
            loader.try_fused = lambda *args, **kwargs: None
        @torch.no_grad()
        def loaded(li, eids, *args, **kwargs):
            result = original(li, eids, *args, **kwargs)
            context = self._moe_inputs.get(int(li))
            if context is None:
                return result
            hidden, indices = context
            gate, up, down = result
            last_curve = None
            for j, eid in enumerate(eids):
                key = (int(li), int(eid))
                if key in self.expert_errors:
                    continue
                token_ids = (indices == int(eid)).any(-1).nonzero().flatten()[-self.max_input_tokens:]
                if token_ids.numel() == 0:
                    continue
                x = hidden.index_select(0, token_ids.to(hidden.device)).to(gate.device)
                gu = torch.cat((gate[j], up[j]), dim=0)
                activation = F.silu(F.linear(x, gate[j])) * F.linear(x, up[j])
                self.expert_errors[key] = dict(
                    gate_up=projection_prefix_errors(gu, x, root_bits=self.root_bits),
                    down=projection_prefix_errors(down[j], activation, root_bits=self.root_bits))
                last_curve = self.expert_errors[key]['gate_up']
            if self.progress is not None and last_curve is not None:
                self.progress(int(li), f'expert_survey/{len(self.expert_errors)}', last_curve)
            return result
        loader.load_batch = loaded

    def __enter__(self):
        for li, layer in enumerate(self.layers):
            for name, mod in layer.named_modules():
                if (not isinstance(mod, nn.Linear) or mod.in_features % 64
                        or _is_router_module(f'layers.{li}.{name}', mod)):
                    continue
                self.handles.append(mod.register_forward_pre_hook(self._hook(li, name)))
        return self

    def _hook(self, layer, name):
        @torch.no_grad()
        def observe(mod, args):
            key = (layer, name)
            if key in self.errors or not args or mod.weight.is_meta:
                return
            x = args[0].detach().reshape(-1, args[0].shape[-1])[-self.max_input_tokens:]
            self.errors[key] = projection_prefix_errors(mod.weight.detach(), x, mod.bias,
                                                        root_bits=self.root_bits)
            self.module_specs.append((layer, name, mod.weight.numel()))
            if self.progress is not None:
                self.progress(layer, name, self.errors[key])
        return observe

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for loader, name, original, was_instance_attr in reversed(self._restore):
            if was_instance_attr:
                setattr(loader, name, original)
            else:
                delattr(loader, name)
        self._restore.clear()
        self._moe_inputs.clear()

    def blueprint(self, tomography, *, relative_error=0.01, selection_policy='measured_prefix_convergence'):
        """Choose the first stable adequate prefix, with Colibri sensitivity or pure-math Mirror Equality.

        When `selection_policy='mirror'`, selects `k_opt` purely from bracketless Mirror Equality
        on the measured error sequence `(e_{p-1}, e_p, e_{p+1})`:
            delta_p = |e_p - e_{p-1}| / (e_p + e_{p-1} + 1e-12)
        stopping at the earliest prefix where BOTH residual `e_p <= relative_error` AND mirror step
        `delta_p <= 0.5` (no arbitrary `relative_error / sqrt(sensitivity)` scaling).
        """
        if not math.isfinite(relative_error) or relative_error <= 0:
            raise ValueError('relative_error must be finite and positive')
        if selection_policy not in ('measured_prefix_convergence', 'mirror'):
            raise ValueError(f'Unsupported selection_policy: {selection_policy!r}')
        use_mirror = (selection_policy == 'mirror')

        def _pick_mirror(curve: list[float], tol: float) -> int:
            for i in range(len(curve)):
                e_cur = float(curve[i])
                if e_cur > tol:
                    continue
                if i == 0:
                    return 1
                e_prev = float(curve[i - 1])
                d_step = abs(e_cur - e_prev) / (e_cur + e_prev + 1e-12)
                e_next = float(curve[i + 1]) if (i + 1) < len(curve) else e_cur
                d_next = abs(e_next - e_cur) / (e_next + e_cur + 1e-12)
                # Center-read convergence: residual inside tol and future wing shows no upward split
                if e_next <= tol and (d_step <= 0.5 or d_next <= 0.25):
                    return i + 1
            return len(curve)

        rows = {r['layer']: r for r in tomography['layers']}
        selection, hist, failures, details = {}, Counter(), [], []
        weighted = total = 0
        for li, name, nparams in self.module_specs:
            curve = self.errors[(li, name)]
            if use_mirror:
                tolerance = float(relative_error)
                chosen = _pick_mirror(curve, tolerance)
            else:
                row = rows.get(li, {})
                sensitivity = (1.0 + float(row.get('deadzone_intrusion', 0)) * 2
                               + 1 - float(row.get('survivor_recall', 1))
                               + 1 - float(row.get('router_overlap', 1)))
                sensitivity *= max(1.0, min(3.0, float(row.get('amplification', 1))))
                tolerance = relative_error / math.sqrt(max(1, sensitivity))
                chosen = self.root_bits
                # Inspect the whole suffix to avoid mistaking accidental
                # cancellation at an early prefix for convergence.
                for i in range(self.root_bits):
                    if max(curve[i:]) <= tolerance:
                        chosen = i + 1
                        break
            if curve[-1] > tolerance:
                failures.append([li, name])
            selection.setdefault(li, {})[name] = chosen
            hist[chosen] += 1
            weighted += chosen * nparams
            total += nparams
            details.append(dict(layer=li, projection=name, prefix_relative_errors=curve,
                                selected=chosen, tolerance=tolerance, parameters=nparams))
        expert_selection = {}
        expert_hist = Counter()
        for (li, eid), curves in self.expert_errors.items():
            if use_mirror:
                expert_tolerance = float(relative_error)
            else:
                row = rows.get(li, {})
                sensitivity = (1 + 2 * float(row.get('deadzone_intrusion', 0))
                               + 1 - float(row.get('survivor_recall', 1))
                               + 1 - float(row.get('router_overlap', 1)))
                sensitivity *= max(1., min(3., float(row.get('amplification', 1))))
                expert_tolerance = relative_error / math.sqrt(max(1., sensitivity))
            selected = {}
            for projection, curve in curves.items():
                if use_mirror:
                    depth = _pick_mirror(curve, expert_tolerance)
                else:
                    depth = self.root_bits
                    for i in range(self.root_bits):
                        if max(curve[i:]) <= expert_tolerance:
                            depth = i + 1
                            break
                selected[projection] = depth
                expert_hist[depth] += 1
            expert_selection.setdefault(li, {})[eid] = selected
        return dict(root_bits=self.root_bits, selection_policy=selection_policy,
                    layer_proj_cubes=selection, relative_error=relative_error,
                    layer_expert_cubes=expert_selection, expert_depth_histogram=dict(expert_hist),
                    experts_observed=len(self.expert_errors), unseen_expert_depth=self.root_bits,
                    depth_histogram=dict(sorted(hist.items())),
                    parameter_weighted_depth=weighted / max(1, total),
                    projections=len(details), layers_observed=len(selection),
                    unresolved_at_full_depth=failures, prefix_history=details)
