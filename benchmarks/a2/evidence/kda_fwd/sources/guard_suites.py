"""Run the exact predeclared guard only after both native boundary scans finish."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
ap=argparse.ArgumentParser()
ap.add_argument('--block-dim',type=int,required=True,choices=[1,2])
ap.add_argument('--output',required=True)
a=ap.parse_args()
root=Path(__file__).resolve().parents[1]
bounds=[]
for bd in (1,2):
    found=json.loads((root/'native'/f'refine-bd{bd}'/'bounds.json').read_text())
    assert len(found)==5 and all(b['block_dim']==bd and b['first_bad']-b['last_good']==2 for b in found)
    bounds.extend(found)
guard=min(math.floor(.9*min(b['first_bad'] for b in bounds)/2)*2,min(b['last_good'] for b in bounds))
print('EXACT_GUARD_PROTOCOL',json.dumps({'guard':guard,'bounds':bounds}),flush=True)
subprocess.run([sys.executable,'-u',str(root/'repo/benchmarks/verify_real_shapes.py'),'--check','a2-unit','--block-dim',str(a.block_dim),'--a2-launcher','bridge','--a2-suite','real','--a2-output',a.output,'--a2-lengths','4096','128','256','512','64','--a2-seeds',*map(str,range(8)),'--a2-spans',str(guard)],check=True)
