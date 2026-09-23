"""block_dim must not change the arithmetic: outputs must be bitwise equal across bd values.

One process per bd would be ideal, but CANN resolves the vendor tree once per process, so this
compiles every bd up front via the unit's own launcher and compares within the run.
"""
import hashlib, json, os, sys
U = os.environ["UNIT"]
sys.path[:0] = [U, os.path.join(U, "ref")]
os.chdir(U)
import torch, torch_npu  # noqa: F401  registers the npu device
import unit as UN
from ref.reference import make_inputs

BDS = tuple(int(x) for x in os.environ.get("BDS", "1,2,4,8").split(","))
CASES = os.environ.get("CASES", "t64_h16_real,t65_h16_tail,t192_h16_odd").split(",")
contract = json.loads(open("contract.json").read())
cases = {c["id"]: c for c in contract["cases"]}

def h(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]

ok = True
for cid in CASES:
    x = make_inputs(cases[cid])
    got = {}
    for bd in BDS:
        opts = {"device": "a5", "backend": "cce", "block_dim": bd, "launcher": os.environ.get("LAUNCHER", "aclnn"),
                "timeout": 1800, "board": None, "out_dir": os.environ["OUT"] + f"/{cid}_bd{bd}"}
        got[bd] = UN.execute(x, opts)
    line = [f"{cid:20s}"]
    for name in ("o", "final_state"):
        ref = got[BDS[0]][name]
        same = all(torch.equal(ref, got[bd][name]) for bd in BDS[1:])
        ok &= same
        line.append(f"{name}={'bitwise' if same else 'DIFFER'}({h(ref)})")
    print(" ".join(line), flush=True)
print("ALL BITWISE EQUAL" if ok else "MISMATCH")
