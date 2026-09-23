"""The FP32 path must be bitwise identical before and after the host conversions were removed.

Two sibling package trees, identical except for ops/gdn2_chunk_fwd.py, are imported in separate
subprocesses (one vendor-tree resolution per process) and their outputs compared byte for byte.
"""
import hashlib, os, subprocess, sys, json

TD = os.environ["TD"]
RUN = r'''
import hashlib, os, sys, torch, torch_npu
sys.path.insert(0, os.environ["TREE"])
from ascend_fla.ops.gdn2_chunk_fwd import chunk_gdn2, prepare
prepare(block_dim=8) if os.environ.get("PREPARE_BOTH") == "1" else None
g = torch.Generator().manual_seed(20260923)
T, H = 64, 16
mk = lambda s: (torch.randn(1, T, H, 128, generator=g) * s)
q, k, v = (mk(0.5).float().to("npu") for _ in range(3))
gate = (-torch.rand(1, T, H, 128, generator=g) * 3.0).to("npu")
b = torch.rand(1, T, H, 128, generator=g).to("npu")
w = torch.rand(1, T, H, 128, generator=g).to("npu")
st = (torch.randn(1, H, 128, 128, generator=g) * 0.1).to("npu")
o, fs = chunk_gdn2(q, k, v, gate, b, w, initial_state=st, output_final_state=True)
for name, t in (("o", o), ("final_state", fs)):
    a = t.cpu().contiguous().view(torch.uint8).numpy().tobytes()
    print(name, hashlib.sha256(a).hexdigest()[:24], t.dtype, flush=True)
'''
out = {}
for tag in ("before", "after"):
    env = dict(os.environ, TREE=f"{TD}/trees/{tag}", PREPARE_BOTH="1" if tag == "after" else "0")
    r = subprocess.run([sys.executable, "-c", RUN], capture_output=True, text=True, env=env, timeout=3000)
    lines = [l for l in r.stdout.splitlines() if l.startswith(("o ", "final_state "))]
    if not lines:
        print(f"{tag}: FAILED\n{r.stdout[-400:]}\n{r.stderr[-800:]}")
        sys.exit(1)
    out[tag] = dict(l.split()[:2] for l in lines)
    print(tag, out[tag], flush=True)
same = out["before"] == out["after"]
print("FP32 PATH BITWISE IDENTICAL" if same else f"DIFFER: {json.dumps(out)}")
