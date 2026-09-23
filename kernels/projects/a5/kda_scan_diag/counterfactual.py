"""Test a pre-registered seed-overwrite prediction against historical tensors.

CPU FP32 only. This reanalyzes captured device outputs; it does not execute or
qualify a device kernel and introduces no numerical acceptance threshold.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--affected-capture', type=Path, required=True)
    parser.add_argument('--affected-sha256', required=True)
    parser.add_argument('--control-capture', type=Path, required=True)
    parser.add_argument('--control-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert sha(args.affected_capture)==args.affected_sha256
    assert sha(args.control_capture)==args.control_sha256
    import torch
    torch.set_num_threads(1)
    affected=torch.load(args.affected_capture,map_location='cpu',weights_only=True)
    control=torch.load(args.control_capture,map_location='cpu',weights_only=True)
    x=affected['results']['candidate1']['scan_inputs']
    control_inputs=control['results']['candidate1']['scan_inputs']
    assert set(x)==set(control_inputs)
    input_hashes={}
    for name,value in x.items():
        assert value.shape==control_inputs[name].shape and value.dtype==control_inputs[name].dtype
        actual_bytes=value.contiguous().view(torch.uint8).numpy().tobytes()
        control_bytes=control_inputs[name].contiguous().view(torch.uint8).numpy().tobytes()
        assert actual_bytes==control_bytes,name
        input_hashes[name]=hashlib.sha256(actual_bytes).hexdigest()
    reference=control['scan_replays'][0]
    assert x['qg'].shape==(2,192,4,128)
    assert x['dht'].shape==(2,4,64,256)
    state=x['dht'].float().reshape(2,4,128,128).clone()
    seeds={}
    scale=128**-.5
    for chunk in range(2,-1,-1):
        for batch in range(2):
            for head in range(4):
                q=x['qg'][batch,chunk*64:(chunk+1)*64,head].float()
                do=x['grad_out'][batch,chunk*64:(chunk+1)*64,head].float()
                seed=q.T@do
                seeds[batch,head,chunk]=seed
                w=x['w'][batch,chunk*64:(chunk+1)*64,head].float()
                dv=reference['dv'][batch,chunk*64:(chunk+1)*64,head].float()
                corr=w.T@dv
                decay=(x['g_last'][batch,chunk,head].float()*math.log(2.)).exp()[:,None]
                if (batch,head,chunk)==(1,2,0):
                    prior=state[batch,head].clone()
                    correction=corr.clone()
                    final_decay=decay.clone()
                state[batch,head]=(seed*scale+state[batch,head]*decay)-corr
    correct=state.bfloat16()
    candidates={name:((seeds[index]*scale+prior*final_decay)-correction).bfloat16()
                for name,index in [('next_head_last',(1,3,2)),('next_head_first',(1,3,0)),
                                   ('same_head_middle',(1,2,1))]}
    rows=[]
    for index,actual in enumerate(affected['scan_replays']):
        assert all(torch.equal(actual[n],reference[n]) for n in ('dAqk','dh','dv'))
        bad=actual['dh0'][1,2]
        stable=reference['dh0'][1,2]
        changed=bad.view(torch.int16)!=stable.view(torch.int16)
        predictions={}
        for name,predicted in candidates.items():
            difference=(bad.float()-predicted.float()).abs()
            predictions[name]=dict(changed_count=int(changed.sum()),
                exact_bits_on_changed=int(((bad.view(torch.int16)==predicted.view(torch.int16))&changed).sum()))
            if changed.any():
                predictions[name]['relative_l2_changed']=float(difference[changed].norm()/bad.float()[changed].norm())
        rows.append(dict(trial=index,changed_rows=(changed.sum(-1)>0).nonzero().flatten().tolist(),
            changed_per_row=changed.sum(-1).tolist(),
            differing_per_bhv=(actual['dh0'].view(torch.int16)!=reference['dh0'].view(torch.int16)).sum((-1,-2)).tolist(),
            counterfactuals=predictions))
    result=dict(schema='a5k03.historical-counterfactual/1',
        scope='CPU FP32 reanalysis of historical native outputs; not a new hardware run',
        environment=dict(python=platform.python_version(),torch=torch.__version__,device='cpu'),
        source_sha256=sha(__file__),affected_capture_sha256=args.affected_sha256,
        control_capture_sha256=args.control_sha256,
        identical_scan_input_hashes=input_hashes,
        formula='bf16(seed_next_head_last/sqrt(128) + prior_state*exp(g_last*ln2) - current_corr)',
        coefficients_fitted=False,
        correct_reference_relative_l2=float((correct.float()-reference['dh0'].float()).norm()/reference['dh0'].float().norm()),
        rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('HISTORICAL_COUNTERFACTUAL',sum(r['counterfactuals']['next_head_last']['exact_bits_on_changed'] for r in rows),
          sum(r['counterfactuals']['next_head_last']['changed_count'] for r in rows),flush=True)


if __name__=='__main__':
    main()
