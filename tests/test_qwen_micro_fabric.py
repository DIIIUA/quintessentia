"""Real trained operator dependencies, numeric fidelity, autonomous GPU spawning."""
import pytest
import torch
import torch.nn.functional as F
from src.binary_halving_q1 import carve_q1_cubes_incremental,LazyMoEBinaryHalvingCache
from src.qwen_micro_fabric import launch_q1_moe_fabric

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

@pytest.mark.parametrize('H,I,E,P',[(1024,128,2,8),(2048,512,8,16)])
@torch.no_grad()
def test_fabric_preserves_moe_and_spawns_down_before_gu_finishes(H,I,E,P):
    torch.manual_seed(807)
    dev=torch.device('cuda:0');packages=[];frozen=[]
    for _ in range(E):
        gu,_=carve_q1_cubes_incremental((torch.randn(2*I,H,device=dev)*.025).bfloat16(),P,root_bits=P)
        dn,_=carve_q1_cubes_incremental((torch.randn(H,I,device=dev)*.025).bfloat16(),P,root_bits=P)
        packages.append((gu,dn,P));frozen.append(({k:v.clone() for k,v in gu.items()},{k:v.clone() for k,v in dn.items()}))
    x=torch.randn(1,H,device=dev,dtype=torch.bfloat16)
    weights=torch.softmax(torch.arange(E,device=dev,dtype=torch.float32)/E,dim=0)
    cache=LazyMoEBinaryHalvingCache(lambda _l,_e:None,num_layers=1,num_experts=E,device=dev,store_cubes=P,root_bits=P)
    expected=torch.zeros_like(x)
    for (gu,dn,p),rw in zip(packages,weights):
        g,u=cache._run_proj(x,gu,p).chunk(2,dim=-1)
        y=cache._run_proj(F.silu(g)*u,dn,p)
        expected.add_((y*rw.reshape(1,1)).bfloat16())
    actual,stats=launch_q1_moe_fabric(x,packages,weights,base_tol=0.,return_stats=True)
    relative=float(torch.linalg.vector_norm(actual.float()-expected.float())/torch.linalg.vector_norm(expected.float()))
    assert relative<.015,(relative,stats)
    assert F.cosine_similarity(actual.float(),expected.float()).item()>.9999
    assert stats['root_seeds']==1 and stats['remaining_tasks']==0 and stats['queue_empty']
    assert stats['tasks']['activation']==E*(I//64)
    assert stats['tasks']['combine']==H//8
    if I>=512:assert stats['early_down_jobs']>0,stats
    if E==8:
        for _ in range(3):
            repeated,repeated_stats=launch_q1_moe_fabric(x,packages,weights,base_tol=0.,return_stats=True)
            assert torch.equal(repeated,actual), 'output depends on GPU work scheduling'
            assert repeated_stats['spawned_tasks']==stats['spawned_tasks']
    assert stats['gate_up_group_plane_reads']+stats['down_group_plane_reads']==stats['full_group_plane_reads']
    for (gu,dn,_),(g0,d0) in zip(packages,frozen):
        assert all(torch.equal(v,g0[k]) for k,v in gu.items())
        assert all(torch.equal(v,d0[k]) for k,v in dn.items())
    adaptive,adaptive_stats=launch_q1_moe_fabric(x,packages,weights,base_tol=.10,return_stats=True)
    assert torch.isfinite(adaptive).all()
    assert adaptive_stats['gate_up_group_plane_reads']<stats['gate_up_group_plane_reads']
    if I>=512:assert adaptive_stats['early_down_jobs']>0
    assert F.cosine_similarity(adaptive.float(),expected.float()).item()>.99


@torch.no_grad()
def test_lazy_moe_cache_reaches_fabric_and_reuses_real_expert_weights():
    torch.manual_seed(817)
    H,I=1024,128
    gu=[(torch.randn(2*I,H)*.03).bfloat16() for _ in range(2)]
    dn=[(torch.randn(H,I)*.03).bfloat16() for _ in range(2)]
    loads=[]
    def reader(layer,eid):
        loads.append((layer,eid));return gu[eid],dn[eid]
    cache=LazyMoEBinaryHalvingCache(reader,num_layers=1,num_experts=2,device='cuda',
        store_cubes=8,root_bits=8,max_cache_bytes=16*1024**2)
    x=torch.randn(1,H,device='cuda',dtype=torch.bfloat16)
    idx=torch.tensor([[0,1]],device='cuda');rw=torch.tensor([[.6,.4]],device='cuda')
    expected=cache.forward_layer_moe(0,x,idx,rw,F.silu)
    cache.moe_fabric=True
    actual=cache.forward_layer_moe(0,x,idx,rw,F.silu)
    assert cache.fabric_calls==1 and loads==[(0,0),(0,1)]
    assert cache.hits==2
    assert F.cosine_similarity(actual.float(),expected.float()).item()>.9999
    cache.close()


@torch.no_grad()
def test_unified_engine_fabric_generates_tokens(tmp_path):
    from quintessentia_cli.unified_runtime import QuintessentiaUnifiedEngine
    from test_ideal_q1_and_modes import _make_tiny_olmoe
    folder=tmp_path/'moe_fabric';folder.mkdir()
    _make_tiny_olmoe(folder,hidden_size=128,intermediate_size=128,num_layers=2,num_experts=4)
    with QuintessentiaUnifiedEngine(folder,weight_format='q1_cubes',cube_root_bits=8,q1_planes=8,
            q1_moe_fabric=True,enable_mtp=False,speculation='off',progressive_start=False,
            max_context_tokens=128,prefill_chunk_tokens=8,verbose=False) as engine:
        done=list(engine.stream_chat([{'role':'user','content':'1 2 3'}],max_new_tokens=4,
                enable_thinking=False,stop_on_eos=False,use_session_cache=False))[-1]
        assert done['metrics']['generated_tokens']==4
        stats=engine.expert_cache_stats()
        assert stats['moe_fabric'] and stats['fabric_calls']>=6
