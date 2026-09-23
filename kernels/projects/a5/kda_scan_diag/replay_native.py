"""A5K-03 fixed-seed native scan replay, after the unchanged full workload.

Only this experimental runner calls the scan directly. Public dispatch and the
upstream kernel stay unchanged. Captures are private; JSON records are portable.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import traceback

from replay_statistics import summarize_hashes

UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device-label', required=True, choices=('control-c','control-d','control-e','cross-env'))
    parser.add_argument('--block-dim', type=int, choices=(1,2,3,4), required=True)
    parser.add_argument('--trials', type=int, default=50)
    args = parser.parse_args()
    assert args.trials >= 50
    assert os.environ.get('BF08_EXTERNAL_DEVICE_LOCK') == '1'
    assert not args.output.exists(), 'Retain existing results'
    args.output.mkdir(parents=True)
    sources = json.loads((REPO.parent/'source-manifest.json').read_text())
    for name, digest in sources.items():
        assert file_sha(REPO/name) == digest, name

    def write(name, value):
        path = args.output/(name+'.json')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)

    # This verifier validates accepted dependency/import identities, compiles
    # all 50 vendors including the nine backward entries, and executes full Kimi
    # against independent CPU FP32 references before the diagnostic scan calls.
    verifier = load('_a5k03_full', REPO/'kernels/projects/a5/kda_prep/verify_backward_native.py')
    previous_argv = sys.argv
    sys.argv = ['verify_backward_native.py', '--block-dim', str(args.block_dim),
                '--output', str(args.output/'full')]
    try:
        verifier.main()
    finally:
        sys.argv = previous_argv

    import torch
    from ascend_fla.ops.kda import chunk, chunk_bwd
    check = load('_a5k03_native_support', REPO/'kernels/projects/a5/kda_layout/native_support.py')
    env = load('_a5k03_environment', REPO/'kernels/projects/a5/kda_prep/native_environment.py').collect()
    env['production_source_sha256'] = sources
    env['driver_sha256'] = file_sha(__file__)
    env['statistics_sha256'] = file_sha(UNIT/'replay_statistics.py')
    write('environment', env)
    manifest_path = REPO/'kernels/projects/a5/kda_layout/evidence/repro-bundle/seed-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['schema'] == 'fmt02.scan-seeded-inputs/1'
    generator = manifest['generator']
    assert file_sha(REPO/generator['file']) == generator['sha256']
    real = load('_a5k03_generator', REPO/generator['file'])
    x = getattr(real, generator['entry'])(**generator['parameters'])
    reproduction = {'public_inputs': {n:dict(actual=check.digest(t), expected=manifest['public_inputs'][n]['sha256'])
                                      for n,t in x.items()}}
    write('input-reproduction', reproduction)
    assert all(r['actual'] == r['expected'] for r in reproduction['public_inputs'].values())
    dev = {n:t.npu() for n,t in x.items()}
    _, _, caches = chunk.chunk_kda_fwd_with_caches(
        **{n:dev[n] for n in ('q','k','v','g','beta')}, initial_state=dev['h0'],
        block_dim=args.block_dim, layout_device='npu')
    b,hv,c,k,v,t = (manifest['scalars'][n] for n in ('B','HV','C','K','V','T'))
    gate_source = caches['g_cumsum'].view(-1)[63*hv*k:]
    g_last = chunk._layout_runtime().move(gate_source, (b,c,hv,1,k),
        (t*hv*k,64*hv*k,k,0,1), block_dim=args.block_dim).view(b,c,hv,k)
    inputs = {n:caches[n] for n in ('kg','qg','w','Aqk','v_new')}
    inputs.update(g_last=g_last, grad_out=dev['do'], dht=dev['dht'].view(b,hv,k//2,2*v))
    reproduction['scan_inputs'] = {n:dict(actual=check.digest(tensor), expected=manifest['scan_inputs'][n]['sha256'])
                                   for n,tensor in inputs.items()}
    write('input-reproduction', reproduction)
    assert all(r['actual'] == r['expected'] for r in reproduction['scan_inputs'].values())
    torch.save(dict(public_inputs=x,scan_inputs=check.cpu(inputs)), args.output/'inputs.private.pt')
    op = chunk_bwd._compiled_chain('a5', args.block_dim, 'stable')['scan_fused']
    scalars = {n:manifest['scalars'][n] for n in op.scalar_names}
    entry = chunk_bwd.kda_bwd_kernels()['scan_fused']
    import inspect
    scan_source = Path(inspect.getsourcefile(entry.fn))
    scan_before = file_sha(scan_source)
    hashes, differences, anchor = [], [], None
    try:
        for index in range(args.trials+1):
            outputs = {n:torch.empty(s['shape'], dtype=getattr(torch,s['dtype']), device='npu')
                       for n,s in manifest['scan_outputs'].items()}
            op(inputs, scalars, outputs)
            torch.npu.synchronize()
            actual = check.cpu(outputs)
            row = {n:check.digest(tensor) for n,tensor in actual.items()}
            if index == 0:
                anchor = actual
                anchor_hashes = row
                torch.save(anchor, args.output/'anchor.private.pt')
            else:
                hashes.append(row)
                counts = {n:int((tensor.view(torch.int16) != anchor[n].view(torch.int16)).sum())
                          for n,tensor in actual.items()}
                difference = dict(trial=index, differing_elements=counts,
                                  all_finite={n:bool(tensor.isfinite().all()) for n,tensor in actual.items()})
                if any(counts.values()):
                    capture=args.output/f'deviation-{index:04d}.private.pt'
                    torch.save(actual,capture)
                    difference['private_capture_sha256'] = file_sha(capture)
                    # Per BHV and state row localization, without replacing any raw tensor.
                    dh0_diff=(actual['dh0'].view(torch.int16) != anchor['dh0'].view(torch.int16))
                    difference['dh0_different_per_bhv_row']=dh0_diff.sum(dim=-1).tolist()
                differences.append(difference)
                write('progress',dict(completed_trials=index,requested_trials=args.trials,
                                      device_label=args.device_label,hashes=hashes,differences=differences))
                print('SCAN_REPLAY',args.device_label,args.block_dim,index,counts,flush=True)
        assert all(file_sha(REPO/name)==digest for name,digest in sources.items())
        unchanged={n:check.digest(tensor)==manifest['scan_inputs'][n]['sha256'] for n,tensor in inputs.items()}
        assert all(unchanged.values())
        assert file_sha(scan_source)==scan_before
        summary=dict(schema='a5k03.scan-replay/1',complete=True,device_label=args.device_label,
                     block_dim=args.block_dim,case=manifest['scalars'],trials=args.trials,
                     anchor=anchor_hashes,samples=hashes,differences=differences,
                     repeatability=summarize_hashes(anchor_hashes,hashes),
                     input_unchanged=unchanged,scan_source_before=scan_before,
                     scan_source_after=file_sha(scan_source),seed_manifest_sha256=file_sha(manifest_path),
                     environment=env,compiled_scan_signature=op.signature,
                     synchronization='one synchronize and D2H after each scan; fresh output allocation',
                     capture_policy='private anchor and every deviation tensor; all output hashes retained',
                     mechanism_identified=False,production_fix=False)
        write('summary',summary)
        print('A5K03_REPLAY_COMPLETE',args.device_label,args.trials,flush=True)
    except BaseException:
        (args.output/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
