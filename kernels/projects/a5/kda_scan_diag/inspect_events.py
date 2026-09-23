"""Bounded scan event diagnostics; native full workload must precede this tool.

Uses a private native input capture. Reduced cases retain complete tile sizes,
both AIV participants, and the selected number of BHV iterations per cube.
Model completion is not hardware qualification or evidence of race absence.
"""
import argparse
import gzip
import hashlib
import inspect
import importlib.util
import json
from pathlib import Path
import platform
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--runtime-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, choices=(1, 2), required=True)
    parser.add_argument('--heads', type=int, choices=(1, 2, 4), required=True)
    parser.add_argument('--chunks', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--block-dim', type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument('--variant', choices=('original', 'baseline', 'ring', 'negative'), default='original')
    parser.add_argument('--stress', action='store_true')
    parser.add_argument('--spins', type=int, choices=(0,64,256,1024,4096), default=0)
    args = parser.parse_args()
    assert not args.stress or args.variant!='original'
    assert args.stress or args.spins==0
    args.output.mkdir(parents=True, exist_ok=False)
    assert sha(args.capture) == args.capture_sha256
    manifest = json.loads(args.runtime_manifest.read_text())
    for name, digest in manifest.items():
        assert sha(args.runtime_root/name) == digest, name

    import torch
    import ascriptor
    from ascriptor.backends.sim.pipesim import simulate
    from ascriptor.passes import PIPELINE, PassManager
    from ascriptor.passes.autosync import check_balance
    from ascend_fla.ops.kda.chunk_bwd import kda_bwd_kernels

    torch.set_num_threads(1)
    assert Path(ascriptor.__file__).resolve().is_relative_to(args.runtime_root.resolve()/'library')
    original = kda_bwd_kernels()['scan_fused']
    if args.variant != 'original':
        variant_name=('stress_' if args.stress else '')+args.variant
        variant_path = Path(__file__).parent/'kernels'/f'scan_{variant_name}.py'
        spec = importlib.util.spec_from_file_location('_scan_event_variant', variant_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original = getattr(module, f'a5k03_scan_{variant_name}_kernel')
    source = Path(inspect.getsourcefile(original.fn))
    environment = dict(python=platform.python_version(), torch=torch.__version__,
                       ascriptor=ascriptor.__version__, stage='pipesim',
                       variant=args.variant,
                       common_identity_stress=args.stress, SPINS=args.spins,
                       capture_sha256=args.capture_sha256,
                       source_before=sha(source), driver_sha256=sha(__file__),
                       runtime_manifest_sha256=sha(args.runtime_manifest),
                       verified_runtime_files=len(manifest),
                       library_commit='90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5',
                       kernels_commit='b3b3f9c16df7c4626ed3c081032a1be5a753d0b1')
    def write(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    write('environment.json', environment)
    try:
        captured = torch.load(args.capture, map_location='cpu', weights_only=True)['scan_inputs']
        b, hv, c = args.batch, args.heads, args.chunks
        t = c*64
        values = {}
        for name, tensor in captured.items():
            if name == 'dht':
                values[name] = tensor[:b, :hv].contiguous()
            elif name == 'g_last':
                values[name] = tensor[:b, :c, :hv].contiguous()
            else:
                values[name] = tensor[:b, :t, :hv].contiguous()
        outputs = dict(dAqk=torch.full((b,t,hv,64), float('nan'), dtype=torch.bfloat16),
                       dh=torch.full((b,c,hv,128,128), float('nan'), dtype=torch.bfloat16),
                       dv=torch.full((b,t,hv,128), float('nan'), dtype=torch.bfloat16),
                       dh0=torch.full((b,hv,128,128), float('nan'), dtype=torch.bfloat16))
        values.update(outputs)
        values.update(B=b, HV=hv, C=c)
        if args.stress:values['SPINS']=args.spins
        ordered = tuple(values[name] for name in inspect.signature(original.fn).parameters)
        lowered = PassManager(PIPELINE).run(original.ir())
        balance = check_balance(lowered)
        write('balance.json', dict(event_balance=balance))
        simulation = simulate(lowered, ordered, block_dim=args.block_dim, timeout=90.,
                              seed_outputs=True, check_gm=True, processes=True)
        with gzip.open(args.output/'trace.json.gz', 'wt') as stream:
            json.dump(simulation.scheduler.chrome_trace(), stream)
        write('pipe-report.json', simulation.report)
        returned = dict(zip(outputs, simulation.outputs))
        hashes = {name:hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                  for name,tensor in returned.items()}
        torch.save(returned, args.output/'actual.private.pt')
        assert sha(source) == environment['source_before']
        write('summary.json', dict(environment=environment, complete=True,
                                  case=dict(B=b,HV=hv,C=c,T=t,block_dim=args.block_dim),
                                  event_balance=balance, hazards=simulation.hazards,
                                  deadlock=simulation.report.get('deadlock'),
                                  cycles=simulation.cycles, output_hashes=hashes,
                                  finite={n:bool(v.isfinite().all()) for n,v in returned.items()},
                                  source_after=sha(source),
                                  scope='Executed-trace event diagnostic; not native validation or a fix claim'))
        print('EVENT_DIAGNOSTIC', b,hv,c,args.block_dim,len(simulation.hazards),flush=True)
    except BaseException:
        (args.output/'failure.private.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
