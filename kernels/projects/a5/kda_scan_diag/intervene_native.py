"""Native diagnostic variants and shape sweeps; never used by public dispatch."""
import argparse
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import traceback

from replay_native import UNIT, REPO, file_sha, load
from replay_statistics import summarize_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device-label', choices=('control-c','control-d','control-e','cross-env'), required=True)
    parser.add_argument('--block-dim', type=int, choices=(1,2,3,4), required=True)
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--poison-outputs', action='store_true',
                        help='Match historical replay output fills before each scan')
    args = parser.parse_args()
    assert args.trials >= 50
    assert os.environ.get('BF08_EXTERNAL_DEVICE_LOCK') == '1'
    args.output.mkdir(parents=True, exist_ok=False)
    source_manifest = json.loads((REPO.parent/'source-manifest.json').read_text())
    native = Path(os.environ['BF08_NATIVE_ROOT'])
    for name,digest in source_manifest.items():
        assert file_sha(REPO/name) == digest, name
    for name,digest in json.loads((native/'accepted-source-manifest.json').read_text()).items():
        assert file_sha(native/name) == digest, name
    import torch
    import torch_npu
    import ascriptor
    from ascend_fla.ops.kda import chunk, chunk_bwd
    from ascend_fla.runtime.compile import compile_kernel
    assert platform.python_version() == os.environ['BF08_ACCEPTED_PYTHON']
    assert torch.__version__ == os.environ['BF08_ACCEPTED_TORCH']
    assert torch_npu.__version__ == os.environ['BF08_ACCEPTED_TORCH_NPU']
    assert Path(ascriptor.__file__).resolve().is_relative_to((native/'library').resolve())
    torch.set_num_threads(1)

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)

    environment = load('_scan_env', REPO/'kernels/projects/a5/kda_prep/native_environment.py').collect()
    environment.update(driver_sha256=file_sha(__file__), production_source_sha256=source_manifest)
    write(args.output/'environment.json', environment)
    entries, compiled, build_records = {}, {}, []
    derivation = json.loads((UNIT/'derivation.json').read_text())
    # Every experimental vendor is registered BEFORE the unchanged full verifier
    # compiles the 50 production vendors and starts the first custom execution.
    for variant in ('baseline','ring','negative'):
        source = UNIT/'kernels'/f'scan_{variant}.py'
        assert file_sha(source) == derivation['variants'][variant]['source_sha256']
        module = load('_scan_'+variant, source)
        entry = getattr(module, derivation['variants'][variant]['entry'])
        entries[variant] = entry
        print('DIAG_COMPILE', variant, args.block_dim, flush=True)
        op = compile_kernel(entry, device='a5', block_dim=args.block_dim)
        compiled[variant] = op
        build_records.append(dict(variant=variant, signature=op.signature, source_sha256=file_sha(source),
                                  vendor_files={str(p.relative_to(op.vendor_dir)):file_sha(p)
                                                for p in op.vendor_dir.rglob('*')
                                                if p.is_file() and p.suffix in ('.so','.o','.json')}))
        write(args.output/'compile-experimental.json', dict(environment=environment,
            complete=len(build_records)==3, entries=build_records, first_custom_launch_not_started=True))
    verifier = load('_scan_full', REPO/'kernels/projects/a5/kda_prep/verify_backward_native.py')
    previous_argv = sys.argv
    sys.argv = ['verify_backward_native.py','--block-dim',str(args.block_dim),'--output',str(args.output/'full')]
    try:
        verifier.main()
    finally:
        sys.argv = previous_argv
    check = load('_scan_support', REPO/'kernels/projects/a5/kda_layout/native_support.py')
    manifest_path = REPO/'kernels/projects/a5/kda_layout/evidence/repro-bundle/seed-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    generator = manifest['generator']
    assert file_sha(REPO/generator['file']) == generator['sha256']
    real = load('_scan_generator', REPO/generator['file'])
    public = getattr(real, generator['entry'])(**generator['parameters'])
    assert all(check.digest(v)==manifest['public_inputs'][n]['sha256'] for n,v in public.items())
    dev = {n:v.npu() for n,v in public.items()}
    _,_,cache = chunk.chunk_kda_fwd_with_caches(
        **{n:dev[n] for n in ('q','k','v','g','beta')},initial_state=dev['h0'],
        block_dim=args.block_dim,layout_device='npu')
    gate = cache['g_cumsum'].view(-1)[63*4*128:]
    g_last = chunk._layout_runtime().move(gate,(2,3,4,1,128),
        (192*4*128,64*4*128,128,0,1),block_dim=args.block_dim).view(2,3,4,128)
    captured = {n:cache[n] for n in ('kg','qg','w','Aqk','v_new')}
    captured.update(g_last=g_last,grad_out=dev['do'],dht=dev['dht'].view(2,4,64,256))
    assert all(check.digest(v)==manifest['scan_inputs'][n]['sha256'] for n,v in captured.items())
    cpu = check.cpu(captured)
    torch.save(dict(public_inputs=public, scan_inputs=cpu), args.output/'inputs.private.pt')
    rows = []
    try:
        for b,hv,c in ((2,4,3),(1,2,3),(1,2,2),(1,1,3),(2,4,1),(2,4,2)):
            t = c*64
            case_id = f'b{b}h{hv}c{c}'
            inputs = {}
            for name,tensor in cpu.items():
                sliced = tensor[:b,:hv] if name=='dht' else (
                    tensor[:b,:c,:hv] if name=='g_last' else tensor[:b,:t,:hv])
                inputs[name] = sliced.contiguous().npu()
            input_hashes = {n:check.digest(v) for n,v in inputs.items()}
            shapes = dict(dAqk=(b,t,hv,64),dh=(b,c,hv,128,128),dv=(b,t,hv,128),dh0=(b,hv,128,128))
            baseline_anchor = None
            for variant,op in compiled.items():
                folder = args.output/case_id/variant
                folder.mkdir(parents=True)
                source = Path(inspect.getsourcefile(entries[variant].fn))
                before = file_sha(source)
                scalars = {n:dict(B=b,HV=hv,C=c,T=t)[n] for n in op.scalar_names}
                samples, differences = [], []
                for index in range(args.trials+1):
                    outputs = {n:torch.empty(shape,dtype=torch.bfloat16,device='npu') for n,shape in shapes.items()}
                    if args.poison_outputs:
                        for tensor in outputs.values():
                            tensor.fill_(float('nan'))
                    op(inputs,scalars,outputs)
                    torch.npu.synchronize()
                    actual = check.cpu(outputs)
                    hashes = {n:check.digest(v) for n,v in actual.items()}
                    if index == 0:
                        anchor, anchor_hashes = actual, hashes
                        torch.save(actual,folder/'anchor.private.pt')
                        if variant=='baseline':baseline_anchor=actual
                        vs_baseline = {n:int((v.view(torch.int16)!=baseline_anchor[n].view(torch.int16)).sum())
                                       for n,v in actual.items()}
                    else:
                        counts = {n:int((v.view(torch.int16)!=anchor[n].view(torch.int16)).sum()) for n,v in actual.items()}
                        row = dict(trial=index,differing_elements=counts,finite={n:bool(v.isfinite().all()) for n,v in actual.items()})
                        if any(counts.values()):
                            p=folder/f'deviation-{index:04d}.private.pt'
                            torch.save(actual,p)
                            row['private_capture_sha256']=file_sha(p)
                            row['dh0_different_per_bhv_row']=(actual['dh0'].view(torch.int16)!=anchor['dh0'].view(torch.int16)).sum(-1).tolist()
                        samples.append(hashes);differences.append(row)
                    write(folder/'progress.json',dict(completed_trials=index,requested_trials=args.trials,
                                                      anchor=anchor_hashes,samples=samples,differences=differences))
                assert file_sha(source)==before
                assert all(check.digest(v)==input_hashes[n] for n,v in inputs.items())
                row = dict(case=case_id,B=b,HV=hv,C=c,T=t,block_dim=args.block_dim,
                           variant=variant,device_label=args.device_label,complete=True,trials=args.trials,
                           repeatability=summarize_hashes(anchor_hashes,samples),samples=samples,
                           anchor=anchor_hashes,anchor_differences_from_baseline=vs_baseline,
                           differences=differences,environment=environment,source_before=before,
                           source_after=file_sha(source),input_hashes=input_hashes,
                           compiled_signature=op.signature,
                           output_initialization='NaN fill' if args.poison_outputs else 'uninitialized empty',
                           input_scope='Runtime fixed-seed public cached forward; reduced cases are contiguous slices of captured scan inputs.',
                           synchronization='Fresh outputs; synchronize and D2H after each scan',
                           production_fix=False)
                write(folder/'summary.json',row);rows.append(row)
                print('INTERVENTION_RESULT',case_id,variant,args.block_dim,
                      {n:v['deviations_from_anchor'] for n,v in row['repeatability']['outputs'].items()},flush=True)
        assert all(file_sha(REPO/n)==d for n,d in source_manifest.items())
        write(args.output/'summary.json',dict(complete=True,environment=environment,cases=rows,
              case_families=len(rows),production_fix=False,
              scope='Repeatability and experimental intervention; existing numerical budgets unchanged.'))
    except BaseException:
        (args.output/'failure.private.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
