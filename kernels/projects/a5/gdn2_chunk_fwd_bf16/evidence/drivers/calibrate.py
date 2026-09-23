"""BF16 error floor F for the GDN-2 chunk forward, measured before any kernel is written.

The BF16 path takes BF16 q/k/v and writes BF16 o; g, erase_gate, w and the state stay FP32.
Two quantities per case:
  F(o)      relative L2 between the FP32 reference run on BF16-rounded inputs and its own BF16 rounding.
            This is the floor a correct BF16 kernel cannot beat, because o is stored in BF16.
  D(inputs) relative L2 between the FP32 reference on FP32 inputs and on BF16-rounded inputs.
            Not a floor on the kernel; it is how much rounding the public ABI itself introduces.
final_state stays FP32, so it has no storage floor; D is reported for it as well.
"""
import json, sys, pathlib
U = pathlib.Path(sys.argv[1])
sys.path[:0] = [str(U), str(U / "ref")]
import torch
from ref.reference import make_inputs, reference

BF16_IN = ("q", "k", "v")


def rel_l2(got, exp):
    got, exp = got.float(), exp.float()
    d = (got - exp).pow(2).sum().sqrt()
    n = exp.pow(2).sum().sqrt()
    return float(d / n) if float(n) > 0 else float(d)


cases = json.loads((U / "contract.json").read_text())["cases"]
rows = []
for case in cases:
    x = make_inputs(case)
    xb = {k: (v.bfloat16().float() if k in BF16_IN else v) for k, v in x.items()}
    ref_fp32 = reference(x)
    ref_round = reference(xb)
    p = case["parameters"]
    rows.append({
        "case": case["id"], "T": p["T"], "H": p["H"],
        "F_o": rel_l2(ref_round["o"].bfloat16(), ref_round["o"]),
        "D_o": rel_l2(ref_round["o"], ref_fp32["o"]),
        "D_final_state": rel_l2(ref_round["final_state"], ref_fp32["final_state"]),
    })
    r = rows[-1]
    print(f"{r['case']:28s} T={r['T']:>5} H={r['H']:>3}  F(o)={r['F_o']:.4e}  "
          f"D(o)={r['D_o']:.4e}  D(state)={r['D_final_state']:.4e}", flush=True)

F = max(r["F_o"] for r in rows)
print(f"\nF (worst over {len(rows)} contract cases) = {F:.4e}")
print(f"3F = {3 * F:.4e}   budget = min(1e-2, 3F) = {min(1e-2, 3 * F):.4e}")
print(f"D(o) range   {min(r['D_o'] for r in rows):.4e} .. {max(r['D_o'] for r in rows):.4e}")
print(f"D(state) rng {min(r['D_final_state'] for r in rows):.4e} .. {max(r['D_final_state'] for r in rows):.4e}")
(pathlib.Path(sys.argv[2])).write_text(json.dumps(
    {"cases": rows, "F": F, "budget": min(1e-2, 3 * F)}, indent=1) + "\n")
