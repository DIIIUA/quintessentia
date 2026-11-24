"""A trained SwiGLU/MoE work graph, spawned and executed inside one GPU launch.

The reader supplies real Q1 contributions. The substrate owns a shared work
stack and readiness counters: input -> gate/up rows -> activation groups ->
down contributions -> weighted mixture. No synthetic graph enters activations.
"""
from __future__ import annotations
import ctypes
import hashlib
import math
import os
import threading
from typing import Sequence
import torch
from src.quintessentia_nvrtc import PTX_CACHE_DIR, _load_driver_and_nvrtc
from src.mirror_micro_agents_nvrtc import _MICRO_CUDA_SOURCE

_FABRIC_SOURCE = r'''
// Task kinds: ROOT, INPUT, GU_TILE, ACT_GROUP, DOWN_GROUP, DOWN_FINAL, COMBINE.
// Immutable node IDs prevent ABA. A CAS stack publishes complete task nodes;
// no global lock can starve the producer while other blocks wait for work.
// Control: top node (-1 means empty), outstanding, error, total_spawned.
__device__ void fabric_push(int* c,int* stack,int capacity,int type,int e,int g,int row) {
    atomicAdd(c+1,1);
    int index=atomicAdd(c+3,1);
    if(index>=capacity){atomicExch(c+2,1);return;}
    volatile int* q=stack+index*5;q[0]=type;q[1]=e;q[2]=g;q[3]=row;
    int top;
    do{top=atomicAdd(c,0);q[4]=top;__threadfence();}while(atomicCAS(c,top,index)!=top);
}
__device__ void fabric_fwht_group(const unsigned short* x,const unsigned short* cols,
                                 float* tx,float* sums,int g,int lane) {
    int a=g*64+lane,b=a+32;
    float v0=bf16_to_f32(x[a])*bf16_to_f32(cols[a]);
    float v1=bf16_to_f32(x[b])*bf16_to_f32(cols[b]);
    for(int d=1;d<32;d<<=1) {
        float u0=__shfl_xor_sync(0xffffffffu,v0,d),u1=__shfl_xor_sync(0xffffffffu,v1,d);
        v0=(lane&d)?u0-v0:v0+u0;v1=(lane&d)?u1-v1:v1+u1;
    }
    float y0=(v0+v1)*0.125f,y1=(v0-v1)*0.125f;
    tx[a]=y0;tx[b]=y1;
    float s=y0+y1;
    for(int d=16;d;d>>=1)s+=__shfl_down_sync(0xffffffffu,s,d);
    if(!lane)sums[g]=s;
}
__device__ unsigned short fabric_read_row(
    const float* tx,const float* sums,const unsigned char* packed,
    const unsigned short* scale,const unsigned short* zero,const unsigned short* rs,
    int row,int N,int K,int maximum,int shared_scale,float tol,
    unsigned int* reads_out
) {
    int lane=threadIdx.x&31,ng=K/64;
    float state[2],obs[2],prev[2];int cursor[2];bool alive[2];
    for(int a=0;a<2;++a){int g=lane+32*a;
        state[a]=g<ng?bf16_to_f32(zero[(long long)row*ng+g])*sums[g]:0.0f;
        obs[a]=prev[a]=0;cursor[a]=0;alive[a]=g<ng;}
    unsigned int reads=0;
    while(__ballot_sync(0xffffffffu,alive[0]||alive[1])) {
        for(int a=0;a<2;++a)if(alive[a]){
            int g=lane+32*a,p=cursor[a];
            long long si=shared_scale?(long long)row*ng+g:((long long)p*N+row)*ng+g;
            float s=bf16_to_f32(scale[si]);if(shared_scale)s=ldexpf(s,-p);
            const unsigned int* bits=reinterpret_cast<const unsigned int*>(packed+((long long)p*N+row)*(K/8));
            float delta=eval_group64_bits(tx+g*64,sums[g],bits[g*2],bits[g*2+1],s);
            prev[a]=obs[a];obs[a]=delta;state[a]+=delta;++cursor[a];++reads;
        }
        float energy=state[0]*state[0]+state[1]*state[1];
        for(int d=16;d;d>>=1)energy+=__shfl_xor_sync(0xffffffffu,energy,d);
        for(int a=0;a<2;++a){int g=lane+32*a;alive[a]=g<ng &&
            micro_substrate_wants_more(state[a],obs[a],prev[a],energy,ng,cursor[a],maximum,
                                      maximum<3?maximum:3,tol);}
    }
    float sum=state[0]+state[1];
    for(int d=16;d;d>>=1){sum+=__shfl_down_sync(0xffffffffu,sum,d);reads+=__shfl_down_sync(0xffffffffu,reads,d);}
    *reads_out=reads;
    return f32_to_bf16_rne(sum*bf16_to_f32(rs[row]));
}
extern "C" __global__ void qwen_micro_fabric_f32(
    const unsigned short* x,const unsigned long long* pointers,const int* info,const float* rw,
    float* gu_tx,float* gu_sum,unsigned short* gu_out,unsigned short* act,
    float* dn_tx,float* dn_sum,float* dn_parts,unsigned short* expert_out,unsigned short* output,
    int* c,int* stack,int* act_ready,int* dn_ready,int* mix_ready,int* gu_done,
    unsigned long long* stats,int H,int I,int E,int capacity,float tol
) {
    __shared__ int task[4];
    int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    int ngH=H/64,ngI=I/64,guTiles=2*I/8,dnTiles=H/8;
    while(true){
        if(!tid){
            task[0]=-2;unsigned long long start=clock64();
            while(task[0]==-2){
                if(atomicAdd(c+2,0)){task[0]=-1;break;}
                int top=atomicAdd(c,0);
                if(top>=0){
                    volatile int* q=stack+top*5;int next=q[4];
                    if(atomicCAS(c,top,next)==top)for(int j=0;j<4;++j)task[j]=q[j];
                }
                else if(!atomicAdd(c+1,0))task[0]=-1;
                if(task[0]==-2){
                    if(clock64()-start>500000000ull){atomicExch(c+2,3);task[0]=-1;}
                    __nanosleep(64);
                }
            }
        }
        __syncthreads();
        int kind=task[0],e=task[1],g=task[2],row=task[3];
        if(kind<0)return;
        const unsigned long long* p=pointers+e*10;
        const int* m=info+e*6;
        const unsigned char* gu_lo=reinterpret_cast<const unsigned char*>(p[0]);
        const unsigned short* gu_sc=reinterpret_cast<const unsigned short*>(p[1]);
        const unsigned short* gu_z=reinterpret_cast<const unsigned short*>(p[2]);
        const unsigned short* gu_rs=reinterpret_cast<const unsigned short*>(p[3]);
        const unsigned short* gu_cs=reinterpret_cast<const unsigned short*>(p[4]);
        const unsigned char* dn_lo=reinterpret_cast<const unsigned char*>(p[5]);
        const unsigned short* dn_sc=reinterpret_cast<const unsigned short*>(p[6]);
        const unsigned short* dn_z=reinterpret_cast<const unsigned short*>(p[7]);
        const unsigned short* dn_rs=reinterpret_cast<const unsigned short*>(p[8]);
        const unsigned short* dn_cs=reinterpret_cast<const unsigned short*>(p[9]);
        if(kind==0){if(!tid)for(int j=0;j<E;++j)fabric_push(c,stack,capacity,1,j,0,0);}
        else if(kind==1){
            for(int group=warp;group<ngH;group+=8)
                fabric_fwht_group(x,gu_cs,gu_tx+e*H,gu_sum+e*ngH,group,lane);
            __threadfence();__syncthreads();
            if(!tid)for(int r=0;r<2*I;r+=8)fabric_push(c,stack,capacity,2,e,0,r);
        }
        else if(kind==2){
            int r=row+warp;unsigned int reads=0;
            unsigned short value=fabric_read_row(gu_tx+e*H,gu_sum+e*ngH,gu_lo,gu_sc,gu_z,gu_rs,
                                                r,2*I,H,m[2],m[4],tol,&reads);
            if(!lane){gu_out[e*2*I+r]=value;atomicAdd(stats+7,(unsigned long long)reads);}
            __threadfence();__syncthreads();
            if(!tid){
                atomicAdd(gu_done+e,1);
                int group=(row%I)/64;
                if(atomicAdd(act_ready+e*ngI+group,8)==120)
                    fabric_push(c,stack,capacity,3,e,group,0);
            }
        }
        else if(kind==3){
            if(tid<64){
                int j=g*64+tid;
                float gv=bf16_to_f32(gu_out[e*2*I+j]);
                float uv=bf16_to_f32(gu_out[e*2*I+I+j]);
                float silu=bf16_to_f32(f32_to_bf16_rne(gv/(1.0f+expf(-gv))));
                act[e*I+j]=f32_to_bf16_rne(silu*uv);
            }
            __threadfence();__syncthreads();
            if(warp==0)fabric_fwht_group(act+e*I,dn_cs,dn_tx+e*I,dn_sum+e*ngI,g,lane);
            __threadfence();__syncthreads();
            if(!tid)for(int r=0;r<H;r+=8)fabric_push(c,stack,capacity,4,e,g,r);
        }
        else if(kind==4){
            int r=row+warp;
            float v0=dn_tx[e*I+g*64+lane],v1=dn_tx[e*I+g*64+32+lane];
            float sum=bf16_to_f32(dn_z[r*ngI+g])*(v0+v1);
            for(int bit=0;bit<m[3];++bit){
                const unsigned int* bits=reinterpret_cast<const unsigned int*>(dn_lo+((long long)bit*H+r)*(I/8));
                unsigned int lo=bits[g*2],hi=bits[g*2+1];
                long long si=m[5]?r*ngI+g:((long long)bit*H+r)*ngI+g;
                float s=bf16_to_f32(dn_sc[si]);if(m[5])s=ldexpf(s,-bit);
                sum+=s*(((lo>>lane)&1u)?v0:-v0);
                sum+=s*(((hi>>lane)&1u)?v1:-v1);
            }
            for(int d=16;d;d>>=1)sum+=__shfl_down_sync(0xffffffffu,sum,d);
            if(!lane)dn_parts[((long long)e*H+r)*ngI+g]=sum;
            __threadfence();__syncthreads();
            if(!tid){
                atomicAdd(stats+8,(unsigned long long)(8*m[3]));
                if(atomicAdd(gu_done+e,0)<guTiles)atomicAdd(stats+9,1ull);
                if(atomicAdd(dn_ready+e*dnTiles+row/8,1)==ngI-1)
                    fabric_push(c,stack,capacity,5,e,0,row);
            }
        }
        else if(kind==5){
            if(tid<8){int r=row+tid;float sum=0;
                for(int group=0;group<ngI;++group)sum+=dn_parts[((long long)e*H+r)*ngI+group];
                float y=bf16_to_f32(f32_to_bf16_rne(sum*bf16_to_f32(dn_rs[r])));
                expert_out[e*H+r]=f32_to_bf16_rne(y*rw[e]);
            }
            __threadfence();__syncthreads();
            if(!tid && atomicAdd(mix_ready+row/8,1)==E-1)fabric_push(c,stack,capacity,6,0,0,row);
        }
        else if(kind==6){if(tid<8){int r=row+tid;float sum=0;
            for(int j=0;j<E;++j)sum=bf16_to_f32(f32_to_bf16_rne(sum+bf16_to_f32(expert_out[j*H+r])));
            output[r]=f32_to_bf16_rne(sum);
        }}
        __threadfence();__syncthreads();
        if(!tid){atomicAdd(stats+kind,1ull);atomicSub(c+1,1);}
        __syncthreads();
    }
}
'''

_LOCK=threading.Lock()
_MODULES={}

def _kernel(device):
    index=device.index or 0
    if index in _MODULES:return _MODULES[index]
    with _LOCK:
        if index in _MODULES:return _MODULES[index]
        driver,nvrtc=_load_driver_and_nvrtc()
        with torch.cuda.device(device):
            major,minor=torch.cuda.get_device_capability(device)
            source=(_MICRO_CUDA_SOURCE+_FABRIC_SOURCE).encode()
            digest=hashlib.sha256(source).hexdigest()[:20]
            PTX_CACHE_DIR.mkdir(parents=True,exist_ok=True)
            path=PTX_CACHE_DIR/f'qwen_fabric_sm{major}{minor}_{digest}.ptx'
            if path.exists():ptx=path.read_bytes()
            else:
                prog=ctypes.c_void_p()
                if nvrtc.nvrtcCreateProgram(ctypes.byref(prog),source,b'qwen_fabric.cu',0,None,None):raise RuntimeError('NVRTC create failed')
                try:
                    opts=(ctypes.c_char_p*2)(f'--gpu-architecture=compute_{major}{minor}'.encode(),b'--use_fast_math')
                    rc=nvrtc.nvrtcCompileProgram(prog,2,opts)
                    if rc:
                        n=ctypes.c_size_t();nvrtc.nvrtcGetProgramLogSize(prog,ctypes.byref(n));log=ctypes.create_string_buffer(n.value)
                        nvrtc.nvrtcGetProgramLog(prog,log);raise RuntimeError(log.value.decode(errors='replace'))
                    n=ctypes.c_size_t();nvrtc.nvrtcGetPTXSize(prog,ctypes.byref(n));buffer=ctypes.create_string_buffer(n.value)
                    if nvrtc.nvrtcGetPTX(prog,buffer):raise RuntimeError('NVRTC PTX failed')
                    ptx=buffer.raw
                finally:nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
                tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_bytes(ptx);os.replace(tmp,path)
            mod=ctypes.c_void_p();buffer=ctypes.create_string_buffer(ptx)
            if driver.cuModuleLoadData(ctypes.byref(mod),ctypes.cast(buffer,ctypes.c_void_p)):raise RuntimeError('fabric module load failed')
            fn=ctypes.c_void_p()
            if driver.cuModuleGetFunction(ctypes.byref(fn),mod,b'qwen_micro_fabric_f32'):raise RuntimeError('fabric kernel lookup failed')
            driver.cuLaunchCooperativeKernel.argtypes=[ctypes.c_void_p]+[ctypes.c_uint]*7+[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p)]
            driver.cuLaunchCooperativeKernel.restype=ctypes.c_int
            driver.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_void_p,ctypes.c_int,ctypes.c_size_t]
            driver.cuOccupancyMaxActiveBlocksPerMultiprocessor.restype=ctypes.c_int
            active=ctypes.c_int()
            if driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(active),fn,256,0) or active.value<1:
                raise RuntimeError('fabric occupancy query failed')
            workers=torch.cuda.get_device_properties(device).multi_processor_count
            _MODULES[index]=(driver,fn,workers,mod)
            return _MODULES[index]


@torch.no_grad()
def launch_q1_moe_fabric(x:torch.Tensor,packages:Sequence,router_weights:torch.Tensor,
                         *,base_tol:float=0.10,return_stats:bool=False):
    """Execute the selected experts' real trained graph from one root task.

    packages contains (gate_up tensors, down tensors, active plane count).
    Whole decoder blocks retain the model's normal dependencies. This fabric
    replaces the per-expert GU/activation/down Python loop, not the whole LLM.
    """
    if x.device.type!='cuda' or x.dtype!=torch.bfloat16 or x.ndim!=2 or x.shape[0]!=1 or not x.is_contiguous():
        raise ValueError('fabric input must be contiguous CUDA bfloat16 [1,H]')
    if not math.isfinite(base_tol) or base_tol<0:raise ValueError('invalid fabric tolerance')
    E=len(packages);H=int(x.shape[1])
    if not E or H%64 or H>4096:raise ValueError('fabric requires experts and H<=4096 divisible by 64')
    I=int(packages[0][0]['packed_lo'].shape[1])//2
    if I%64 or not 64<=I<=4096:raise ValueError('fabric intermediate width must be 64..4096 in groups of 64')
    if router_weights.device!=x.device or router_weights.numel()!=E:raise ValueError('router weights mismatch')
    names=('packed_lo','qscale','qzero','row_scale','col_scale')
    pointers=[];info=[];active_gu=active_dn=0
    stream=torch.cuda.current_stream(x.device)
    for gu,dn,puse in packages:
        if tuple(gu['packed_lo'].shape[1:])!=(2*I,H//8) or tuple(dn['packed_lo'].shape[1:])!=(H,I//8):
            raise ValueError('fabric expert shapes differ')
        ptrs=[]
        for table,N,K in ((gu,2*I,H),(dn,H,I)):
            for name in names:
                tensor=table[name]
                expected=torch.uint8 if name=='packed_lo' else torch.bfloat16
                if tensor.device!=x.device or tensor.dtype!=expected or not tensor.is_contiguous():raise ValueError('invalid fabric weight tensor')
                tensor.record_stream(stream);ptrs.append(tensor.data_ptr())
            P=int(table['packed_lo'].shape[0])
            if P<1 or tuple(table['qzero'].shape)!=(N,K//64) or table['row_scale'].numel()!=N or table['col_scale'].numel()!=K:
                raise ValueError('invalid fabric metadata shape')
            if tuple(table['qscale'].shape) not in ((N,K//64),(1,N,K//64),(P,N,K//64)):raise ValueError('invalid fabric scales')
        pg,pd=int(gu['packed_lo'].shape[0]),int(dn['packed_lo'].shape[0])
        ag,ad=min(pg,max(1,int(puse))),min(pd,max(1,int(puse)))
        sg=int(gu['qscale'].numel()==2*I*(H//64));sd=int(dn['qscale'].numel()==H*(I//64))
        pointers.append(ptrs);info.append([pg,pd,ag,ad,sg,sd]);active_gu+=2*I*(H//64)*ag;active_dn+=H*(I//64)*ad
    ptrs=torch.tensor(pointers,dtype=torch.int64,device=x.device)
    metadata=torch.tensor(info,dtype=torch.int32,device=x.device)
    rw=router_weights.float().contiguous()
    ngH,ngI=H//64,I//64;guTiles,dnTiles=2*I//8,H//8
    capacity=1+E*(1+guTiles+ngI+ngI*dnTiles+dnTiles)+dnTiles
    f=lambda *s:torch.empty(s,dtype=torch.float32,device=x.device)
    z=lambda *s:torch.zeros(s,dtype=torch.int32,device=x.device)
    gu_tx=f(E,H);gu_sum=f(E,ngH);gu_out=torch.empty((E,2*I),dtype=torch.bfloat16,device=x.device)
    act=torch.empty((E,I),dtype=torch.bfloat16,device=x.device);dn_tx=f(E,I);dn_sum=f(E,ngI);dn_parts=f(E,H,ngI)
    expert_out=torch.empty((E,H),dtype=torch.bfloat16,device=x.device)
    output=torch.full((1,H),float('nan'),dtype=torch.bfloat16,device=x.device)
    control=torch.tensor([0,1,0,1],dtype=torch.int32,device=x.device)
    stack=z(capacity,5);stack[0,4]=-1
    act_ready=z(E,ngI);dn_ready=z(E,dnTiles);mix_ready=z(dnTiles);gu_done=z(E)
    stats=torch.zeros(10,dtype=torch.int64,device=x.device)
    driver,fn,workers,_=_kernel(x.device)
    tensors=[x,ptrs,metadata,rw,gu_tx,gu_sum,gu_out,act,dn_tx,dn_sum,dn_parts,expert_out,output,
             control,stack,act_ready,dn_ready,mix_ready,gu_done,stats]
    values=[ctypes.c_void_p(t.data_ptr()) for t in tensors]+[ctypes.c_int(v) for v in (H,I,E,capacity)]+[ctypes.c_float(base_tol)]
    params=(ctypes.c_void_p*len(values))(*[ctypes.addressof(v) for v in values])
    rc=driver.cuLaunchCooperativeKernel(fn,workers,1,1,256,1,1,0,ctypes.c_void_p(stream.cuda_stream),params)
    cooperative=True
    if rc in (801,720):
        cooperative=False;workers=1
        rc=driver.cuLaunchKernel(fn,1,1,1,256,1,1,0,ctypes.c_void_p(stream.cuda_stream),params,None)
    if rc:raise RuntimeError(f'fabric launch failed: CUresult={rc}')
    error=int(control[2].item())
    if error:raise RuntimeError(f'fabric work graph failed: error={error}, control={control.tolist()}, stats={stats.tolist()}, gu_done={gu_done.tolist()}')
    if not return_stats:return output,None
    counts=stats.tolist();ctl=control.tolist()
    assert ctl[0]==-1 and ctl[1]==0 and ctl[3]==capacity and sum(counts[:7])==capacity
    return output,dict(worker_blocks=workers,cooperative=cooperative,root_seeds=1,
        tasks=dict(zip(('root','input','gate_up','activation','down_group','down_finalize','combine'),counts[:7])),
        spawned_tasks=ctl[3],remaining_tasks=ctl[1],queue_empty=ctl[0]==-1,
        gate_up_group_plane_reads=counts[7],down_group_plane_reads=counts[8],
        full_group_plane_reads=active_gu+active_dn,
        group_agents=E*(2*I*ngH+H*ngI),early_down_jobs=counts[9],
        graph='trained SwiGLU experts and their actual weighted mixture',base_tol=base_tol)
