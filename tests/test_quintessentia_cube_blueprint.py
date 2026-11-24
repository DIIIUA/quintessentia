"""Unit tests for Quintessentia Dual-Model Probability Comparator & Cube Blueprint -> Method 2 pipeline."""
import copy
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.binary_halving_q1 import (
    AirLLMLazyBinaryHalvingRunner,
    BinaryHalvingModelSlicer,
    BinaryHalvingQ1Linear,
)
from src.quintessentia_cube_blueprint import (
    QuintessentiaCubeComparator,
    quintessentia_prob_metrics,
)


class _ToyLayer(nn.Module):
    def __init__(self, h=128, inter=256):
        super().__init__()
        self.norm = nn.LayerNorm(h)
        self.q_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def forward(self, x):
        xn = self.norm(x)
        attn = self.o_proj(F.silu(self.q_proj(xn)) * self.v_proj(xn))
        mlp = self.down_proj(F.silu(self.up_proj(xn)))
        return x + 0.1 * (attn + mlp)


class _ToyInner(nn.Module):
    def __init__(self, n_layers=4, h=128, vocab=64):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, h)
        self.layers = nn.ModuleList([_ToyLayer(h, h * 2) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(h)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class _ToyCausalLM(nn.Module):
    def __init__(self, n_layers=4, h=128, vocab=64):
        super().__init__()
        self.model = _ToyInner(n_layers, h, vocab)
        self.lm_head = nn.Linear(h, vocab, bias=False)

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids))


def test_quintessentia_prob_metrics_detects_deadzone_and_false_confidence():
    vocab = 100
    orig = torch.full((vocab,), -10.0)
    orig[5] = 4.0
    orig[7] = 3.0  # runner-up survivor

    # Case 1: identical -> zero KL, 1.0 survivor recall, 0 deadzone intrusion
    m_same = quintessentia_prob_metrics(orig, orig.clone())
    assert m_same["kl_div"] == 0.0
    assert m_same["survivor_recall"] == 1.0
    assert m_same["deadzone_intrusion"] == 0.0
    assert m_same["false_confidence"] == 0.0

    # Case 2: cube collapses sharply onto a deadzone token (token 50) -> false confidence + deadzone intrusion!
    bad_cube = torch.full((vocab,), -10.0)
    bad_cube[50] = 8.0
    m_bad = quintessentia_prob_metrics(orig, bad_cube)
    assert m_bad["top1_match"] == 0.0
    assert m_bad["deadzone_intrusion"] > 0.95
    assert m_bad["survivor_recall"] < 0.01
    assert m_bad["false_confidence"] == 1.0


def test_quintessentia_blueprint_drives_method2_airllm_runner_cpu():
    torch.manual_seed(123)
    base_model = _ToyCausalLM(n_layers=4, h=128, vocab=64).to(torch.bfloat16).eval()
    m1_model = copy.deepcopy(base_model).eval()
    m2_model = copy.deepcopy(base_model).eval()
    ids = torch.randint(0, 64, (1, 8))

    # 1. Capture original trajectory + projection activations
    orig_traj = QuintessentiaCubeComparator.capture_trajectory(
        base_model, lambda x: base_model(x)[0, -1], ids, capture_proj_inputs=True
    )
    assert len(orig_traj.layer_logits) == 4
    assert len(orig_traj.proj_inputs) == 4 * 5

    # 2. Slice m1_model (Method 1), probe at k=3, capture cube trajectory
    slicer = BinaryHalvingModelSlicer(m1_model, root_bits=8, group_size=64, use_fwht=True)
    slicer.slice_all()
    slicer.set_uniform_cubes(3)
    cube_traj = QuintessentiaCubeComparator.capture_trajectory(
        m1_model, lambda x: m1_model(x)[0, -1], ids, capture_proj_inputs=False
    )

    tomo = QuintessentiaCubeComparator.compare_trajectories(orig_traj, cube_traj)
    assert tomo["num_layers"] == 4
    assert len(tomo["layers"]) == 4
    assert "survivor_recall" in tomo["layers"][-1]
    assert "deadzone_intrusion" in tomo["layers"][-1]

    # 3. Measure activation-weighted errors & build Quintessentia blueprint for avg_k = 3.5
    slicer_mods = []
    specs = []
    for l_idx, layer in enumerate(m1_model.model.layers):
        for sn, sm in layer.named_modules():
            if isinstance(sm, BinaryHalvingQ1Linear):
                slicer_mods.append((l_idx, sn, sm))
                specs.append((l_idx, sn, sm.in_features * sm.out_features))
    act_errs = QuintessentiaCubeComparator.measure_activation_projection_errors(
        slicer_mods, orig_traj.proj_inputs, k_min=2, k_max=6
    )
    bp = QuintessentiaCubeComparator.build_blueprint(
        tomo, specs, act_errs, target_avg_cubes=3.5, k_min=2, k_max=6
    )
    assert 3.2 <= bp.actual_param_weighted_avg_cubes <= 3.8
    assert len(bp.depth_histogram) >= 2  # non-uniform allocation!

    # 4. Feed blueprint directly into Method 2 AirLLMLazyBinaryHalvingRunner and verify execution + ABC invariants
    saved_layers = [
        {k: v.detach().cpu().clone() for k, v in layer.state_dict().items()}
        for layer in base_model.model.layers
    ]
    runner = AirLLMLazyBinaryHalvingRunner(
        m2_model,
        lambda li: saved_layers[li],
        device="cpu",
        cubes_per_proj=4,
        layer_proj_cubes=bp.layer_proj_cubes,
        root_bits=8,
        group_size=64,
        use_fwht=True,
        host_cache_bytes=64 * 1024**2,
    ).install()
    try:
        with torch.no_grad():
            out_m2 = m2_model(ids)[0, -1]
        st = runner.stats()
        assert st["transitions"]["B->A"] == 0
        assert st["standby_bytes"] == 0
        # Verify carved cubes match exact sum of blueprint depths (not uniform 4!)
        expected_cubes = sum(k for m in bp.layer_proj_cubes.values() for k in m.values())
        assert st["total_cubes_carved"] == expected_cubes
        cos = F.cosine_similarity(out_m2.float(), orig_traj.final_logits.float(), dim=0).item()
        assert cos > 0.98
    finally:
        runner.close()
