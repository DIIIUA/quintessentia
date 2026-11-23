"""The reader/substrate feedback loop must preserve the learned operator."""
import pytest
import torch
import torch.nn.functional as F

from src.binary_halving_q1 import BinaryHalvingQ1Linear, LazyMoEBinaryHalvingCache, halve_weight_to_q1_tree
from src.mirror_micro_agents_nvrtc import launch_coupled_micro_agent_q1_reader

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("n,k,planes", [(96,576,8),(192,2048,16),(64,8192,8)])
@torch.no_grad()
def test_coupled_feedback_preserves_full_weight_operator(n,k,planes):
    torch.manual_seed(680)
    dev=torch.device('cuda:0')
    w=(torch.randn(n,k,device=dev)*.04).bfloat16()
    tensors,meta=halve_weight_to_q1_tree(w,root_bits=planes,group_size=64,
                                       law='positional_halving',use_fwht=True)
    bias=(torch.randn(n,device=dev)*.01).bfloat16()
    lin=BinaryHalvingQ1Linear(k,n,tensors,meta,bias=bias)
    frozen={name:t.clone() for name,t in tensors.items()}
    x=torch.randn(1,k,device=dev,dtype=torch.bfloat16)
    reference=F.linear(x.float(),lin.reconstruct_weight(planes,torch.float32),bias.float())
    y=torch.empty((1,n),device=dev,dtype=torch.float32)
    stats=launch_coupled_micro_agent_q1_reader(x,*[tensors[name] for name in
        ('packed_lo','qscale','qzero','row_scale','col_scale')],y,
        base_tol=0.,max_planes=planes,bias=bias,return_stats=True,backend='worklist')
    torch.testing.assert_close(y,reference,atol=1e-4,rtol=1e-4)
    assert stats['group_plane_reads']==n*(k//64)*planes
    assert stats['reader_launches']==stats['substrate_launches']>1
    assert stats['wave_sizes'][-1]==0 and not stats['exact_drain']
    assert all(torch.equal(t,frozen[name]) for name,t in tensors.items())

    adaptive=launch_coupled_micro_agent_q1_reader(x,*[tensors[name] for name in
        ('packed_lo','qscale','qzero','row_scale','col_scale')],y,
        base_tol=.10,max_planes=planes,bias=bias,return_stats=True,backend='worklist')
    assert adaptive['group_plane_reads']<stats['group_plane_reads']
    assert adaptive['wave_sizes'][-1]==0
    assert F.cosine_similarity(y.flatten(),reference.flatten(),dim=0)>.99
    assert all(torch.equal(t,frozen[name]) for name,t in tensors.items())
    if k<=4096:
        fused=launch_coupled_micro_agent_q1_reader(x,*[tensors[name] for name in
            ('packed_lo','qscale','qzero','row_scale','col_scale')],y,
            base_tol=0.,max_planes=planes,bias=bias,return_stats=True,backend='fused')
        torch.testing.assert_close(y,reference,atol=1e-4,rtol=1e-4)
        assert fused['fused_substrate'] and fused['group_plane_reads']==stats['group_plane_reads']
        assert fused['remaining_agents']==0
        fused=launch_coupled_micro_agent_q1_reader(x,*[tensors[name] for name in
            ('packed_lo','qscale','qzero','row_scale','col_scale')],y,
            base_tol=.10,max_planes=planes,bias=bias,return_stats=True,backend='fused')
        assert fused['group_plane_reads']<stats['group_plane_reads']
        assert F.cosine_similarity(y.flatten(),reference.flatten(),dim=0)>.99


@torch.no_grad()
def test_coupled_substrate_handles_one_million_real_group_indices():
    # Group indices address actual packed weight coordinates, with no random
    # substitute graph. This stresses the queue capacity with 1,048,576 nodes.
    n=k=8192
    ng=k//64
    dev=torch.device('cuda:0')
    packed=torch.zeros((8,n,k//8),device=dev,dtype=torch.uint8)
    scales=torch.full((1,n,ng),.01,device=dev,dtype=torch.bfloat16)
    zeros=torch.zeros((n,ng),device=dev,dtype=torch.bfloat16)
    rows=torch.ones(n,device=dev,dtype=torch.bfloat16)
    cols=torch.ones(k,device=dev,dtype=torch.bfloat16)
    x=torch.zeros((1,k),device=dev,dtype=torch.bfloat16)
    bias=torch.full((n,),.25,device=dev,dtype=torch.bfloat16)
    y=torch.empty((1,n),device=dev,dtype=torch.bfloat16)
    stats=launch_coupled_micro_agent_q1_reader(x,packed,scales,zeros,rows,cols,y,
                                            bias=bias,base_tol=.14,return_stats=True)
    assert stats['total_group_agents']==1_048_576
    assert stats['wave_sizes']==[1_048_576,0]
    assert stats['group_plane_reads']==3*1_048_576
    torch.testing.assert_close(y,bias.unsqueeze(0),atol=0,rtol=0)


@torch.no_grad()
def test_dense_and_lazy_expert_dispatch_reach_the_same_coupled_operator():
    torch.manual_seed(711)
    dev=torch.device('cuda:0')
    w=(torch.randn(512,1024,device=dev)*.03).bfloat16()
    t,meta=halve_weight_to_q1_tree(w,root_bits=8,group_size=64,use_fwht=True)
    x=torch.randn((1,1024),device=dev,dtype=torch.bfloat16)
    expected=torch.empty((1,512),device=dev,dtype=torch.bfloat16)
    launch_coupled_micro_agent_q1_reader(x,*[t[k] for k in
        ('packed_lo','qscale','qzero','row_scale','col_scale')],expected,base_tol=0.)
    dense=BinaryHalvingQ1Linear(1024,512,t,meta)
    dense.micro_agent_tol=0.
    torch.testing.assert_close(dense(x),expected,atol=0,rtol=0)
    assert dense.micro_agent_calls==1
    experts=LazyMoEBinaryHalvingCache(lambda _l,_e:None,num_layers=1,num_experts=1,
                                    device=dev,store_cubes=8,group_size=64,use_fwht=True)
    experts.micro_agent_tol=0.
    torch.testing.assert_close(experts._run_proj(x,t,8),expected,atol=0,rtol=0)
    assert experts.micro_agent_calls==1


@torch.no_grad()
def test_coupled_reader_runs_through_resident_and_streamed_relay(monkeypatch):
    import copy
    import src.mirror_micro_agents_nvrtc as micro
    from src.binary_halving_q1 import AirLLMLazyBinaryHalvingRunner
    from test_colibri_cube_blueprint import _ToyCausalLM
    torch.manual_seed(722)
    base=_ToyCausalLM(n_layers=3,h=1024,vocab=64).bfloat16().eval()
    raw=[{k:v.clone() for k,v in layer.state_dict().items()} for layer in base.model.layers]
    ids=torch.tensor([[7]],device='cuda')
    outputs=[];reports=[];observed=[]
    original=micro.launch_coupled_micro_agent_q1_reader
    def seen(*args,**kwargs):
        observed.append(tuple(args[0].shape))
        return original(*args,**kwargs)
    monkeypatch.setattr(micro,'launch_coupled_micro_agent_q1_reader',seen)
    for enabled in (False,True):
        model=copy.deepcopy(base).cuda()
        def read(li):return {k:v.clone() for k,v in raw[li].items()}
        runner=AirLLMLazyBinaryHalvingRunner(model,read,device='cuda',root_bits=8,cubes_per_proj=8,
            resident_layers=1,host_cache_bytes=64*1024**2,enable_relay=enabled,
            micro_agent_tol=0. if enabled else None).install()
        try:
            outputs.append(torch.stack([model(ids).float() for _ in range(2)]))
            reports.append(runner.stats())
        finally:runner.close()
    assert observed, 'real relay execution never reached the coupled operator'
    torch.testing.assert_close(outputs[1],outputs[0],rtol=.01,atol=.02)
    relay=reports[1]['relay']
    assert relay['handoffs']==relay['zero_copy_adoptions']==4
    assert relay['transitions']['B->A']==0
    assert relay['area_used_bytes'][relay['standby_area']]==0
