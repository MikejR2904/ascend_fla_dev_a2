"""Matched public launch graph, distinct native baseline/candidate op types."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import torch

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline',type=Path,required=True,help='GD2-01 or qualified GD2-03 preweight stages.py')
parser.add_argument('--output',type=Path,required=True,help='JSON receipt in an ignored scratch directory')
args=parser.parse_args()
unit_root=Path(__file__).resolve().parent
repo=unit_root.parents[3]
sys.path[:0]=[str(repo),str(unit_root)]
import unit
from _unit_runner import compare_outputs
from ascend_fla.ops import gdn2_chunk_fwd as public
from ascend_fla.runtime.compile import compile_kernel

torch.set_num_threads(4)
source=args.baseline.read_text()
baseline_sha=hashlib.sha256(source.encode()).hexdigest()
baseline_revisions={
    '82dd3fec13edee5c358a336721bd8d45ef7238fb1a62a05eca668b5fbae47e66':'0397aa1',
    'a1f305eb5209420e026a1a6c82d7147f00a00c0588ea7f7f331805993d1cce5b':'690f420',
}
if baseline_sha not in baseline_revisions:
    raise ValueError('baseline must match an identified GD2-01/GD2-03 source')
candidate_source=(unit_root/'kernels/stages.py').read_text()
def stage_blocks(text):
    markers=['@vf()\ndef scores_vf(', '@vf()\ndef wy_vf(',
             '@vf()\ndef scan_vf(', '@vf()\ndef output_vf(', '\nSTAGES =']
    boundaries=[0,*(text.index(marker) for marker in markers)]
    return [text[a:b] for a,b in zip(boundaries,boundaries[1:])]
# Each block contains a VF and its kernel; the first also owns shared constants.
if source.split('@vf()')[0]!=candidate_source.split('@vf()')[0]:
    raise ValueError('baseline and candidate must share module imports/constants')
changed=[a!=b for a,b in zip(stage_blocks(source),stage_blocks(candidate_source))]
candidate_ops=public._compiled(8)
args.output.parent.mkdir(parents=True,exist_ok=True)
p=args.output.with_name(args.output.stem+'-baseline_stages.py')
renamed=source
for stage,differs in zip(('prepare','scores','wy','scan','output'),changed):
    if differs:
        name='gdn2_chunk_'+stage
        renamed=renamed.replace(name,'baseline_'+name)
p.write_text(renamed)
spec=importlib.util.spec_from_file_location('_gdn2_timing_baseline',p)
baseline=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=baseline
spec.loader.exec_module(baseline)
# Identical stage blocks share artifacts. Changed stages have distinct names
# to prevent CANN operator-name collisions within this single process.
baseline_ops=tuple(compile_kernel(e,device='a5',block_dim=8,backend='cce') if differs else op
                   for e,op,differs in zip(baseline.STAGES,candidate_ops,changed))
assert all((a.spec.op!=b.spec.op)==differs for a,b,differs in zip(candidate_ops,baseline_ops,changed))
import torch_npu
torch.npu.set_device(0)
torch.npu.matmul.allow_hf32=False
active={'ops':candidate_ops}
public._compiled=lambda block_dim:active['ops']
contract=json.loads((repo/'kernels/projects/a5/gdn2_chunk_fwd/contract.json').read_text())
report={'golden_device':'cpu','timing_device':'npu','block_dim':8,'warmup':10,'repeat':50,
        'rounds':3,'hf32':False,'cases':[],'passed':False,
        'candidate_sha256':hashlib.sha256((repo/'kernels/projects/a5/gdn2_chunk_fwd/kernels/stages.py').read_bytes()).hexdigest(),
        'baseline_source_sha256':baseline_sha,
        'baseline_revision':baseline_revisions[baseline_sha],
        'changed_stages':[name for name,differs in zip(('prepare','scores','wy','scan','output'),changed) if differs],
        'baseline_timing_source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'artifacts':{label:[{'op':op.spec.op,'signature':op.signature} for op in ops]
                     for label,ops in (('candidate',candidate_ops),('baseline',baseline_ops))}}
out=args.output
def save():out.write_text(json.dumps(report,indent=2))
def measure(fn):
    for _ in range(10):fn()
    torch.npu.synchronize()
    samples=[]
    for _ in range(50):
        start=time.perf_counter_ns()
        result=fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns()-start)/1000)
        del result
    return {'median_us':statistics.median(samples),'mean_us':statistics.mean(samples),'samples_us':samples}
try:
    with torch.no_grad():
        for length in (4096,1024):
            case=next(c for c in contract['cases'] if c['id']==f't{length}_h16_prefill')
            cpu=unit.make_inputs(case)
            golden=unit.reference(cpu)
            inputs={k:v.to('npu:0') for k,v in cpu.items()}
            args=tuple(inputs[k] for k in ('q','k','v','g','erase_gate','w'))
            def call():
                return public.chunk_gdn2(*args,initial_state=inputs['initial_state'],
                                        output_final_state=True,block_dim=8)
            row={'T':length,'H':16,'dtype':'float32','seed':case['seed'],'rounds':[]}
            report['cases'].append(row)
            outputs={}
            for label,ops in (('baseline',baseline_ops),('candidate',candidate_ops)):
                active['ops']=ops
                outputs[label]={k:v.cpu() for k,v in zip(('o','final_state'),call())}
                row[label+'_comparison']=compare_outputs(outputs[label],golden,contract)
                torch.npu.synchronize()
                initial=torch.npu.memory_allocated()
                torch.npu.reset_peak_memory_stats()
                got=call()
                torch.npu.synchronize()
                row[label+'_peak_torch_allocated_bytes']=torch.npu.max_memory_allocated()-initial
                del got
            row['baseline_candidate_bitwise']={k:torch.equal(outputs['baseline'][k].view(torch.uint8),
                                                             outputs['candidate'][k].view(torch.uint8))
                                               for k in golden}
            if not all(row['baseline_candidate_bitwise'].values()):
                raise AssertionError('candidate must preserve baseline FP32 results exactly')
            del outputs
            for index in range(3):
                entry={'round':index+1}
                row['rounds'].append(entry)
                for label,ops in (('baseline_before',baseline_ops),('candidate',candidate_ops),('baseline_after',baseline_ops)):
                    active['ops']=ops
                    entry[label]=measure(call)
                faster=min(entry[k]['median_us'] for k in ('baseline_before','baseline_after'))
                entry['speedup_vs_faster_baseline']=faster/entry['candidate']['median_us']
                print(json.dumps({'T':length,'round':index+1,
                                  **{k:v for k,v in entry.items() if not isinstance(v,dict)},
                                  **{k:v['median_us'] for k,v in entry.items() if isinstance(v,dict)}}),flush=True)
                save()
            del cpu,golden,inputs,args
    report['passed']=all(r['speedup_vs_faster_baseline']>1 for c in report['cases'] for r in c['rounds'])
finally:
    save()
assert report['passed']
