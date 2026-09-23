"""Common identity-VF timing stress for causal diagnosis, never production use."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import sys
import traceback

from replay_native import UNIT, REPO, file_sha, load
from replay_statistics import summarize_hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device-label',choices=('control-c','control-d','control-e','cross-env'),required=True)
    parser.add_argument('--block-dim',type=int,choices=(1,2,3,4),required=True)
    parser.add_argument('--trials',type=int,default=50)
    args=parser.parse_args()
    assert args.trials>=50 and os.environ.get('BF08_EXTERNAL_DEVICE_LOCK')=='1'
    args.output.mkdir(parents=True,exist_ok=False)
    source_manifest=json.loads((REPO.parent/'source-manifest.json').read_text())
    native=Path(os.environ['BF08_NATIVE_ROOT'])
    for name,digest in source_manifest.items():assert file_sha(REPO/name)==digest,name
    for name,digest in json.loads((native/'accepted-source-manifest.json').read_text()).items():
        assert file_sha(native/name)==digest,name
    import torch
    import torch_npu
    import ascriptor
    from ascend_fla.ops.kda import chunk,chunk_bwd
    from ascend_fla.runtime.compile import compile_kernel
    assert platform.python_version()==os.environ['BF08_ACCEPTED_PYTHON']
    assert torch.__version__==os.environ['BF08_ACCEPTED_TORCH']
    assert torch_npu.__version__==os.environ['BF08_ACCEPTED_TORCH_NPU']
    assert Path(ascriptor.__file__).resolve().is_relative_to((native/'library').resolve())
    torch.set_num_threads(1)
    check=load('_stress_support',REPO/'kernels/projects/a5/kda_layout/native_support.py')
    environment=load('_stress_env',REPO/'kernels/projects/a5/kda_prep/native_environment.py').collect()
    environment.update(driver_sha256=file_sha(__file__),production_source_sha256=source_manifest)
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
    write(args.output/'environment.json',environment)
    derived=json.loads((UNIT/'derivation.json').read_text())['stress_variants']
    compiled={};builds=[]
    for variant in ('baseline','ring','negative'):
        source=UNIT/'kernels'/f'scan_stress_{variant}.py'
        assert file_sha(source)==derived[variant]['source_sha256']
        entry=getattr(load('_stress_'+variant,source),derived[variant]['entry'])
        print('STRESS_COMPILE',variant,args.block_dim,flush=True)
        op=compile_kernel(entry,device='a5',block_dim=args.block_dim);compiled[variant]=op
        builds.append(dict(variant=variant,signature=op.signature,source_sha256=file_sha(source),
            vendor_files={str(p.relative_to(op.vendor_dir)):file_sha(p) for p in op.vendor_dir.rglob('*')
                          if p.is_file() and p.suffix in ('.so','.o','.json')}))
        write(args.output/'compile-experimental.json',dict(environment=environment,complete=len(builds)==3,
              entries=builds,first_custom_launch_not_started=True))
    verifier=load('_stress_full',REPO/'kernels/projects/a5/kda_prep/verify_backward_native.py')
    previous=sys.argv
    sys.argv=['verify_backward_native.py','--block-dim',str(args.block_dim),'--output',str(args.output/'full')]
    try:verifier.main()
    finally:sys.argv=previous
    original=chunk_bwd._compiled_chain('a5',args.block_dim,'stable')['scan_fused']
    real=load('_stress_generator',REPO/'benchmarks/verify_real_shapes.py')
    seed_manifest=json.loads((REPO/'kernels/projects/a5/kda_layout/evidence/repro-bundle/seed-manifest.json').read_text())
    assert file_sha(REPO/'benchmarks/verify_real_shapes.py')==seed_manifest['generator']['sha256']
    def make_inputs(b,h,hv,c):
        x=real.make_inputs(B=b,H=h,HV=hv,C=c,span=46,seed=2026,want_grads=True)
        dev={n:v.npu() for n,v in x.items()}
        _,_,cache=chunk.chunk_kda_fwd_with_caches(**{n:dev[n] for n in ('q','k','v','g','beta')},
            initial_state=dev['h0'],block_dim=args.block_dim,layout_device='npu')
        t=c*64
        gate=cache['g_cumsum'].view(-1)[63*hv*128:]
        last=chunk._layout_runtime().move(gate,(b,c,hv,1,128),
            (t*hv*128,64*hv*128,128,0,1),block_dim=args.block_dim).view(b,c,hv,128)
        inputs={n:cache[n] for n in ('kg','qg','w','Aqk','v_new')}
        inputs.update(g_last=last,grad_out=dev['do'],dht=dev['dht'].view(b,hv,64,256))
        return inputs,x
    def launch(op,inputs,b,hv,c,spins):
        t=c*64
        shapes=dict(dAqk=(b,t,hv,64),dh=(b,c,hv,128,128),dv=(b,t,hv,128),dh0=(b,hv,128,128))
        outputs={n:torch.full(s,float('nan'),dtype=torch.bfloat16,device='npu') for n,s in shapes.items()}
        scalars={n:dict(B=b,HV=hv,C=c,T=t,SPINS=spins)[n] for n in op.scalar_names}
        op(inputs,scalars,outputs);torch.npu.synchronize()
        return check.cpu(outputs)
    def differences(actual,expected):
        return {n:int((v.view(torch.int16)!=expected[n].view(torch.int16)).sum()) for n,v in actual.items()}
    def hashes(values):return {n:check.digest(v) for n,v in values.items()}
    def physical_cpu_reference(x,b,hv,c):
        # Independent Torch CPU FP32 operations, preserving documented BF16 seams.
        def pack(v):return v.reshape(b,c,64,hv,v.shape[-1]).permute(0,3,1,2,4).float()
        def unpack(v):return v.permute(0,2,3,1,4).reshape(b,c*64,hv,v.shape[-1]).contiguous()
        kg,qg,w,aqk,grad,new=(pack(x[n]) for n in ('kg','qg','w','Aqk','grad_out','v_new'))
        dAqk=((grad@new.transpose(-1,-2))*(128**-.5)).bfloat16()
        dv0=aqk.transpose(-1,-2)@grad
        dv=torch.empty_like(grad,dtype=torch.bfloat16)
        dh=torch.empty((b,hv,c,128,128),dtype=torch.bfloat16)
        state=x['dht'].float().reshape(b,hv,128,128).clone()
        for i in range(c-1,-1,-1):
            dh[:,:,i]=state.bfloat16()
            current=(dv0[:,:,i]+kg[:,:,i]@state.bfloat16().float()).bfloat16()
            dv[:,:,i]=current
            decay=(x['g_last'][:,i].float()*math.log(2.)).exp()[...,None]
            seed=qg[:,:,i].transpose(-1,-2)@grad[:,:,i]
            corr=w[:,:,i].transpose(-1,-2)@current.float()
            state=(seed*(128**-.5)+state*decay)-corr
        return dict(dAqk=unpack(dAqk),dh=dh.permute(0,2,1,3,4).contiguous(),
                    dv=unpack(dv),dh0=state.bfloat16())
    try:
        # Full experimental scan before any reduced stress family, on every card.
        full_inputs,public=make_inputs(1,32,32,64)
        cpu=check.cpu(full_inputs)
        reference=physical_cpu_reference(cpu,1,32,64)
        upstream=launch(original,full_inputs,1,32,64,0)
        contract=Path(os.environ['ASCRIPTOR_WORKSPACE'])/'kernels/projects/a5/kda_bwd/contract.json'
        limits=json.loads(contract.read_text())['comparison']['stage_outputs']
        full_rows=[]
        for variant,op in compiled.items():
            actual=launch(op,full_inputs,1,32,64,4096)
            metrics={}
            for name,value in actual.items():
                golden=reference[name].float();a=value.float();limit=limits['scan.'+name]
                l2=float((a-golden).norm()/golden.norm().clamp_min(1e-30))
                finite=bool(a.isfinite().all())
                close=bool(torch.allclose(a,golden,atol=limit['atol'],rtol=limit['rtol']))
                metrics[name]=dict(relative_l2=l2,finite=finite,allclose=close,
                    passed=finite and close and l2<=limit['max_relative_l2'],existing_budget=limit)
            torch.save(actual,args.output/f'full-stress-{variant}.private.pt')
            row=dict(variant=variant,B=1,HV=32,C=64,T=4096,SPINS=4096,metrics=metrics,
                differences_from_original=differences(actual,upstream),output_hashes=hashes(actual))
            full_rows.append(row)
            write(args.output/'full-experimental.json',dict(environment=environment,complete=len(full_rows)==3,
                cases=full_rows,comparison_contract_sha256=file_sha(contract),reference='Torch CPU FP32, physical BF16 seams'))
            assert all(m['passed'] for m in metrics.values()),'Full experimental scan exceeds existing stage budget'
            assert not any(row['differences_from_original'].values()),'Full even-C experimental scan differs from original'
        del full_inputs,cpu,reference,upstream,actual,public
        inputs,public=make_inputs(2,2,4,3)
        assert all(check.digest(v)==seed_manifest['public_inputs'][n]['sha256'] for n,v in public.items())
        assert all(check.digest(v)==seed_manifest['scan_inputs'][n]['sha256'] for n,v in inputs.items())
        torch.save(dict(public_inputs=public,scan_inputs=check.cpu(inputs)),args.output/'inputs.private.pt')
        unperturbed=launch(original,inputs,2,4,3,0)
        torch.save(unperturbed,args.output/'unperturbed.private.pt')
        families=[]
        for spins in (0,64,256,1024,4096):
            for variant,op in compiled.items():
                folder=args.output/f'spins{spins}'/variant;folder.mkdir(parents=True)
                samples=[];rows=[]
                for index in range(args.trials+1):
                    actual=launch(op,inputs,2,4,3,spins);actual_hashes=hashes(actual)
                    against_unperturbed=differences(actual,unperturbed)
                    if index==0:
                        anchor=actual;anchor_hashes=actual_hashes
                        torch.save(actual,folder/'anchor.private.pt')
                    else:
                        against_anchor=differences(actual,anchor)
                        row=dict(trial=index,differing_elements=against_anchor,
                            differing_from_unperturbed=against_unperturbed,
                            finite={n:bool(v.isfinite().all()) for n,v in actual.items()})
                        if any(against_anchor.values()) or any(against_unperturbed.values()):
                            path=folder/f'actual-{index:04d}.private.pt';torch.save(actual,path)
                            row['capture_sha256']=file_sha(path)
                            row['dh0_different_per_bhv_row']=(actual['dh0'].view(torch.int16)!=unperturbed['dh0'].view(torch.int16)).sum(-1).tolist()
                        samples.append(actual_hashes);rows.append(row)
                    write(folder/'progress.json',dict(completed_trials=index,samples=samples,rows=rows))
                source=UNIT/'kernels'/f'scan_stress_{variant}.py'
                summary=dict(environment=environment,complete=True,variant=variant,SPINS=spins,
                    trials=args.trials,block_dim=args.block_dim,device_label=args.device_label,
                    B=2,HV=4,C=3,T=192,anchor=anchor_hashes,samples=samples,rows=rows,
                    unperturbed_hashes=hashes(unperturbed),repeatability=summarize_hashes(anchor_hashes,samples),
                    source_before=derived[variant]['source_sha256'],source_after=file_sha(source),
                    compiled_signature=op.signature,production_fix=False,
                    scope='Common identity-VF timing stress; distinct from unmodified upstream reproduction')
                assert summary['source_before']==summary['source_after']
                write(folder/'summary.json',summary);families.append(summary)
                print('STRESS_RESULT',variant,spins,
                    summary['repeatability']['outputs']['dh0']['deviations_from_anchor'],
                    sum(bool(r['differing_from_unperturbed']['dh0']) for r in rows),flush=True)
        assert all(file_sha(REPO/n)==d for n,d in source_manifest.items())
        write(args.output/'summary.json',dict(complete=True,environment=environment,families=families,production_fix=False))
    except BaseException:
        (args.output/'failure.private.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
