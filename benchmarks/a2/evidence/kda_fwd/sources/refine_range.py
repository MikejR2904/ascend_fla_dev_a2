"""Private adaptive orchestration; every sample runs the frozen public benchmark."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ap=argparse.ArgumentParser()
ap.add_argument('--block-dim',type=int,required=True)
ap.add_argument('--output',required=True)
ap.add_argument('--coarse')
a=ap.parse_args()
root=Path(__file__).resolve().parents[1]
out=Path(a.output).resolve()
out.mkdir(parents=True,exist_ok=True)
benchmark=root/'repo/benchmarks/verify_real_shapes.py'
rows=[]
bounds=[]
def good(row):
    return all(m['ok'] for m in row['comparison'].values()) and all(row['finite_stages'].values())
def sample(t,seed,span):
    case=f't{t}_seed{seed}_span{float(span)}'
    if a.coarse:
        cached=Path(a.coarse)/(case+'.json')
        if cached.exists():
            row=json.loads(cached.read_text())
            return row, 'range-coarse-bd1/'+cached.name
    folder=out/case
    cmd=[sys.executable,'-u',str(benchmark),'--check','a2-unit','--block-dim',str(a.block_dim),'--a2-launcher','bridge','--a2-suite','range','--a2-lengths',str(t),'--a2-seeds',str(seed),'--a2-spans',str(span),'--a2-output',str(folder)]
    subprocess.run(cmd,check=True)
    row=json.loads((folder/'results.json').read_text())[0]
    rows.append(row)
    (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    return row, out.name+'/'+case+'/'+case+'.json'
for t,seed in [(4096,s) for s in range(3)]+[(128,0),(64,0)]:
    lo,hi=160,192
    low,low_path=sample(t,seed,lo)
    high,high_path=sample(t,seed,hi)
    assert good(low) and not good(high), (t,seed,'coarse bracket is not established')
    history=[{'target':lo,'good':True,'record':low_path},{'target':hi,'good':False,'record':high_path}]
    while hi-lo>2:
        mid=((lo+hi)//4)*2
        row,path=sample(t,seed,mid)
        passed=good(row)
        history.append({'target':mid,'good':passed,'record':path})
        if passed:
            lo=mid
        else:
            hi=mid
    bounds.append({'t':t,'seed':seed,'block_dim':a.block_dim,'last_good':lo,'first_bad':hi,'history':history})
    (out/'bounds.json').write_text(json.dumps(bounds,indent=2)+'\n')
    print('REFINED_BOUNDARY',json.dumps(bounds[-1]),flush=True)
