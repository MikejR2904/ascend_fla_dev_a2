import json
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
plan=[('real-bd1',1,'real',[4096,128,256,512,64],[2026]),('real-bd2',2,'real',[4096,128,256,512,64],[2026]),('performance-bd1',1,'performance',[4096,128],[2026]),('performance-bd2',2,'performance',[4096,128],[2026]),('defaults-bd1',1,'defaults',[4096],list(range(8)))]
for name,bd,suite,lengths,seeds in plan:
    print('CONTROL_SUITE_START',name,flush=True)
    subprocess.run([sys.executable,'-u',str(root/'repo/benchmarks/verify_real_shapes.py'),'--check','a2-unit','--block-dim',str(bd),'--a2-launcher','bridge','--a2-suite',suite,'--a2-output',str(root/'native'/name),'--a2-lengths',*map(str,lengths),'--a2-seeds',*map(str,seeds),'--a2-spans','8'],check=True)
    print('CONTROL_SUITE_COMPLETE',name,flush=True)
