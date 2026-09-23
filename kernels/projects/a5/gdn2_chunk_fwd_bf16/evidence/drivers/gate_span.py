"""Gate span on the BF16 path: report 'finite' and 'accurate' separately, per AGENTS.md section 6.

GD2-01's 1520.9 is an FP32 observation. BF16 changes the precision, so the limit is re-measured
here rather than inherited. For each gate_scale: the span actually reached (max |chunk cumsum|),
whether the outputs are finite, and the relative L2 against the unit's CPU reference.
"""
import json, os, sys
U = os.environ["UNIT"]
sys.path[:0] = [U, os.path.join(U, "ref")]
os.chdir(U)
import torch, torch_npu  # noqa: F401
import unit as UN
from ref.reference import make_inputs, reference

SCALES = [float(x) for x in os.environ.get("SCALES", "3,8,16,24,32,48,64").split(",")]
T, H = int(os.environ.get("T", 128)), int(os.environ.get("H", 16))
BUDGET = 5.2354e-03


def rel_l2(got, exp):
    got, exp = got.float(), exp.float()
    n = exp.pow(2).sum().sqrt()
    return float((got - exp).pow(2).sum().sqrt() / n) if float(n) > 0 else 0.0


print(f"{'gate_scale':>10} {'span':>10} {'finite':>7} {'o relL2':>11} {'state relL2':>12} {'verdict':>9}")
for s in SCALES:
    case = {"id": f"span_{s}", "seed": 20260923, "block_dim": 8,
            "parameters": {"B": 1, "T": T, "H": H, "gate_scale": s}}
    x = make_inputs(case)
    gc = x["g"].reshape(1, T // 64, 64, H, 128).cumsum(2)
    span = float(gc.abs().max())
    ref = reference(x)
    opts = {"device": "a5", "backend": "cce", "block_dim": 8, "launcher": "aclnn",
            "timeout": 1800, "board": None, "out_dir": os.environ["OUT"] + f"/s{s}"}
    got = UN.execute(x, opts)
    finite = bool(torch.isfinite(got["o"].float()).all() and torch.isfinite(got["final_state"]).all())
    lo, ls = rel_l2(got["o"], ref["o"]), rel_l2(got["final_state"], ref["final_state"])
    verdict = "ok" if finite and max(lo, ls) <= BUDGET else ("finite-only" if finite else "non-finite")
    print(f"{s:>10.1f} {span:>10.1f} {str(finite):>7} {lo:>11.3e} {ls:>12.3e} {verdict:>9}", flush=True)
