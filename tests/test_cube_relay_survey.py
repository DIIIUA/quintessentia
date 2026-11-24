"""Reference survey -> variable prefixes -> streamed relay with real payloads."""
import copy
from types import SimpleNamespace

import pytest
import torch

from src.binary_halving_q1 import AirLLMLazyBinaryHalvingRunner, LazyMoEBinaryHalvingCache
from src.q1_cube_engine import Q1RoleCycleArena
from src.streaming_cube_survey import StreamingCubeSurvey, projection_prefix_errors
from test_quintessentia_cube_blueprint import _ToyCausalLM


def test_reference_survey_observes_experts_without_duplicate_reads_and_restores_fused():
    torch.manual_seed(84)
    g, u, d = [torch.randn(2, 64, 64, dtype=torch.bfloat16)*.03 for _ in range(3)]
    calls = []
    def calculate(x, idx, rw, gate, up, down):
        out = torch.zeros_like(x)
        for eid in range(2):
            rows = (idx[:,0] == eid).nonzero().flatten()
            xe = x[rows]
            y = torch.nn.functional.linear(torch.nn.functional.silu(
                torch.nn.functional.linear(xe, gate[eid])) *
                torch.nn.functional.linear(xe, up[eid]), down[eid])
            out[rows] = y * rw[rows]
        return out
    class Loader:
        def try_fused(self, li, x, idx, rw):
            calls.append('fused')
            return calculate(x, idx, rw, g, u, d)
        def load_batch(self, li, eids):
            calls.append('batch')
            return g[eids], u[eids], d[eids]
    loader = Loader()
    class Experts(torch.nn.Module):
        def forward(self, x, idx, rw):
            fused = loader.try_fused(0, x, idx, rw)
            if fused is not None:
                return fused
            gate, up, down = loader.load_batch(0, [0,1])
            return calculate(x, idx, rw, gate, up, down)
    layer = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    layer.mlp.experts = Experts()
    engine = SimpleNamespace(model_source=SimpleNamespace(
        _expert_loader=loader, _routed_expert_container={0:'mlp.experts'}))
    x = torch.randn(3, 64, dtype=torch.bfloat16)
    idx = torch.tensor([[0],[1],[0]])
    rw = torch.ones((3,1), dtype=torch.bfloat16)
    expected = layer.mlp.experts(x, idx, rw)
    with StreamingCubeSurvey([layer]) as survey:
        survey.observe_moe(engine)
        actual = layer.mlp.experts(x, idx, rw)
    assert torch.equal(actual, expected)
    assert torch.equal(layer.mlp.experts(x, idx, rw), expected)
    assert calls == ['fused','batch','fused']
    assert len(survey.expert_errors) == 2
    assert all(len(curve)==16 for curves in survey.expert_errors.values() for curve in curves.values())
    assert 'load_batch' not in vars(loader) and 'try_fused' not in vars(loader)


def test_complete_tree_survey_records_every_prefix_and_drives_lazy_relay():
    torch.manual_seed(420)
    base = _ToyCausalLM(n_layers=3, h=64, vocab=64).to(torch.bfloat16).eval()
    ids = torch.randint(0, 64, (1, 6))
    with torch.inference_mode():
        with StreamingCubeSurvey(base.model.layers, root_bits=16) as survey:
            base(ids)
        assert len(survey.errors) == 15
        assert all(len(curve) == 16 for curve in survey.errors.values())
        assert all(curve[-1] < curve[0] for curve in survey.errors.values())
        blueprint = survey.blueprint(dict(layers=[dict(layer=i) for i in range(3)]), relative_error=.025)
        assert blueprint['root_bits'] == 16
        assert blueprint['projections'] == 15
        assert len(blueprint['prefix_history']) == 15

        # Both executions receive the measured map, rather than a hardcoded 4.
        outputs, reports = [], []
        for relay in (False, True):
            model = copy.deepcopy(base)
            raw = [{k: v.detach().clone() for k, v in l.state_dict().items()} for l in model.model.layers]
            reads = []
            def read(li):
                reads.append(li)
                return {k: v.clone() for k, v in raw[li].items()}
            runner = AirLLMLazyBinaryHalvingRunner(
                model, read, device='cpu', root_bits=16, cubes_per_proj=16,
                layer_proj_cubes=blueprint['layer_proj_cubes'],
                host_cache_bytes=4 * 1024**2, enable_relay=relay,
            ).install()
            try:
                outputs.append(torch.stack([model(ids).float() for _ in range(3)]))
                reports.append(runner.stats())
                assert len(reads) == 3  # warm handoffs do not reread source layers
            finally:
                runner.close()
        assert torch.equal(outputs[0], outputs[1])
        relay = reports[1]['relay']
        assert relay['handoffs'] == 9
        assert relay['cache_prepares'] >= 6
        assert relay['area_used_bytes'][relay['standby_area']] == 0
        assert relay['transitions']['B->A'] == 0
        assert relay['retained_plan_layers'] == 3


def test_relay_turn_preserves_the_prepared_object_and_cleans_old_B():
    arena = Q1RoleCycleArena(4096, torch.device('cpu'))
    histories = []
    for step in range(7):
        prepared = {'layer': step, 'probability_trace': torch.arange(32)}
        old_b, old_c = arena.warm_area, arena.standby_area
        arena.publish_prepared(prepared)
        arena.rotate()
        assert arena.active_area == old_c
        assert arena.standby_area == old_b
        assert arena.active_payload() is prepared
        assert arena.buffers[arena.standby_area] is None
        histories.append(tuple((arena.active_area, arena.warm_area, arena.standby_area)))
        arena.release_active_A()
    assert histories[:3] == [(2, 0, 1), (1, 2, 0), (0, 1, 2)]


def test_measured_expert_prefixes_control_storage_and_execution():
    torch.manual_seed(92)
    gu = torch.randn(128, 64, dtype=torch.bfloat16) * .03
    dn = torch.randn(64, 64, dtype=torch.bfloat16) * .03
    cache = LazyMoEBinaryHalvingCache(
        lambda li, eid: (gu, dn), num_layers=1, num_experts=2,
        device=torch.device('cpu'), store_cubes=16, root_bits=16,
        expert_cubes={(0, 0): (3, 5)}, cache_on_gpu=False, max_cache_bytes=0)
    try:
        gt, dt = cache._get_or_carve(0, 0)
        assert gt['packed_lo'].shape[0] == 3
        assert dt['packed_lo'].shape[0] == 5
        assert cache.total_cubes_carved == 8
        out = cache._run_proj(torch.randn(2, 64, dtype=torch.bfloat16), gt, 16)
        assert out.shape == (2, 128)
        assert bool(torch.isfinite(out).all())
        # A route absent from calibration retains every leaf, rather than
        # being excluded or silently assigned the old fixed depth of four.
        unseen, _ = cache._get_or_carve(0, 1)
        assert unseen['packed_lo'].shape[0] == 16
    finally:
        cache.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cold_B_prepares_cubes_before_handoff_without_recarving_in_A(monkeypatch):
    import threading
    import src.binary_halving_q1 as binary
    torch.manual_seed(24)
    base = _ToyCausalLM(n_layers=2, h=64, vocab=64).to(dtype=torch.bfloat16, device='cuda').eval()
    raw = [{k:v.detach().cpu().clone() for k,v in layer.state_dict().items()} for layer in base.model.layers]
    original_carve = binary.carve_q1_cubes_incremental
    calls = []
    def record(*args, **kwargs):
        calls.append(threading.current_thread().name)
        return original_carve(*args, **kwargs)
    monkeypatch.setattr(binary, 'carve_q1_cubes_incremental', record)
    outputs = []
    with torch.inference_mode():
        for relay in (False, True):
            calls.clear()
            model = copy.deepcopy(base)
            runner = AirLLMLazyBinaryHalvingRunner(model, lambda li: raw[li], device='cuda',
                cubes_per_proj=16, root_bits=16, host_cache_bytes=1024**2, enable_relay=relay).install()
            try:
                outputs.append(model(torch.tensor([[1,2,3,4]],device='cuda')).detach().clone())
                torch.cuda.synchronize()
                if relay:
                    assert runner.stats()['relay']['gpu_prepares'] == 2
                    assert len(calls) == 10
                    assert all(name.startswith('quintessentia-cube-B') for name in calls)
            finally:
                runner.close()
        assert torch.equal(outputs[0], outputs[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_gpu_prepared_payload_survives_relay_and_stream_retirement():
    device = torch.device('cuda')
    arena = Q1RoleCycleArena(64 * 1024, device)
    results = []
    expected = []
    for i in range(12):
        source = torch.full((2048,), float(i), pin_memory=True)
        arena.stage_payload_into_B({'value': source}, source.numel() * source.element_size())
        staged = arena.buffers[arena.warm_area][0]['value']
        pointer = staged.data_ptr()
        arena.rotate()
        payload = arena.active_payload()
        assert payload['value'].data_ptr() == pointer
        results.append(payload['value'].square().sum())
        expected.append(float(i * i * source.numel()))
        arena.release_active_A()
    torch.cuda.synchronize()
    assert torch.equal(torch.stack(results).cpu(), torch.tensor(expected))
    assert arena.area_used_bytes[arena.standby_area] == 0
