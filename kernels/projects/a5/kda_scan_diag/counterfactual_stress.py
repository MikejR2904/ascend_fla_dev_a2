"""CPU FP32 seed-substitution check on captured timing-stress device outputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--labels',nargs='+',default=['control-c','control-d'])
    args=parser.parse_args()
    assert not args.output.exists()
    expected=json.loads(args.manifest.read_text())
    import torch
    torch.set_num_threads(1)
    rows=[]
    for label in args.labels:
        folder=args.captures/f'receipts/stress-v1-{label}-bd4'
        names=('inputs.private.pt','unperturbed.private.pt','spins4096/baseline/anchor.private.pt',
               'spins4096/ring/anchor.private.pt')
        digests={}
        for name in names:
            path=folder/name
            digests[name]=sha(path)
            assert digests[name]==expected[str(path.relative_to(args.captures))]
        x=torch.load(folder/names[0],weights_only=True,map_location='cpu')['scan_inputs']
        good,bad,ring=[torch.load(folder/n,weights_only=True,map_location='cpu') for n in names[1:]]
        assert x['qg'].shape==(2,192,4,128)
        state=x['dht'].float().reshape(2,4,128,128).clone()
        seeds,prior,corrs,decays={},{},{},{}
        for chunk in range(2,-1,-1):
            for batch in range(2):
                for head in range(4):
                    q=x['qg'][batch,chunk*64:(chunk+1)*64,head].float()
                    do=x['grad_out'][batch,chunk*64:(chunk+1)*64,head].float()
                    seed=q.T@do
                    seeds[batch,head,chunk]=seed
                    w=x['w'][batch,chunk*64:(chunk+1)*64,head].float()
                    dv=good['dv'][batch,chunk*64:(chunk+1)*64,head].float()
                    corr=w.T@dv
                    decay=(x['g_last'][batch,chunk,head].float()*math.log(2.)).exp()[:,None]
                    if chunk==0:
                        prior[batch,head]=state[batch,head].clone()
                        corrs[batch,head]=corr
                        decays[batch,head]=decay
                    state[batch,head]=(seed*(128**-.5)+state[batch,head]*decay)-corr
        predictions=[]
        for batch in range(2):
            for head in (0,2):
                predicted=((seeds[batch,head+1,2]*(128**-.5)
                    +prior[batch,head]*decays[batch,head])-corrs[batch,head]).bfloat16()
                actual=bad['dh0'][batch,head]
                changed=actual.view(torch.int16)!=good['dh0'][batch,head].view(torch.int16)
                predictions.append(dict(B=batch,HV=head,changed=int(changed.sum()),
                    predicted_bits_exact=int(((actual.view(torch.int16)==predicted.view(torch.int16))&changed).sum()),
                    relative_l2_changed=float((actual.float()-predicted.float())[changed].norm()
                                             /actual.float()[changed].norm().clamp_min(1e-30))))
        native=json.loads((folder/'environment.json').read_text())
        rows.append(dict(label=label,environment=native,
            scope='CPU FP32 analysis of captured device tensors; no new device execution',
            per_bhv_difference=(bad['dh0'].view(torch.int16)!=good['dh0'].view(torch.int16)).sum((-1,-2)).tolist(),
            counterfactual=predictions,
            ring_matches_unperturbed={n:bool(torch.equal(ring[n],good[n])) for n in good},captures=digests))
    result=dict(schema='a5k03.stress-counterfactual/1',source_sha256=sha(__file__),
        environment=dict(python=platform.python_version(),torch=torch.__version__,device='cpu'),
        coefficients_fitted=False,rows=rows)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('STRESS_COUNTERFACTUAL',[(r['label'],sum(p['predicted_bits_exact'] for p in r['counterfactual']),
                                   sum(p['changed'] for p in r['counterfactual'])) for r in rows])


if __name__=='__main__':main()
