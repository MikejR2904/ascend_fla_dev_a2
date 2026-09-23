"""Host operator audit for the GDN-2 public entry, one pass per dtype path.

Records every aten op the entry dispatches. Only allocation and non-copying metadata may appear;
any dtype conversion, copy/reshape or arithmetic that produces data is a failure. Read-only
validation is listed separately and does not fail the audit.
"""
import json, os, sys, collections
sys.path.insert(0, os.environ["REPO"])
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ALLOC = ("empty", "empty_like", "empty_strided", "zeros", "zeros_like", "full", "new_empty", "new_zeros")
META = ("view", "_unsafe_view", "reshape", "unsqueeze", "squeeze", "alias", "expand", "detach",
        "t", "select", "slice", "as_strided")
CHECK = ("isfinite", "all", "any", "gt", "lt", "ge", "le", "eq", "ne", "item", "_local_scalar_dense",
         "max", "min", "sum", "abs", "is_contiguous", "equal")
BAN_PREFIX = ("to", "_to_copy", "copy_", "clone", "cat", "stack", "pad", "repeat", "contiguous",
              "npu_format_cast", "convert_element_type")


class Audit(TorchDispatchMode):
    def __init__(self):
        self.ops = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops[func.overloadpacket.__name__] += 1
        return func(*args, **(kwargs or {}))


def classify(name):
    if name in ALLOC:
        return "alloc"
    if name in META:
        return "meta"
    if name in CHECK:
        return "check"
    if name in BAN_PREFIX:
        return "BANNED"
    return "other"


def main():
    from ascend_fla.ops.gdn2_chunk_fwd import chunk_gdn2
    dev = os.environ.get("AFLA_DEVICE", "npu")
    T, H = int(os.environ.get("T", 64)), int(os.environ.get("H", 16))
    out = {}
    for dtype in (torch.float32, torch.bfloat16):
        g = torch.Generator().manual_seed(7)
        mk = lambda s: (torch.randn(1, T, H, 128, generator=g) * s)
        q, k, v = (mk(0.5).to(dtype).to(dev) for _ in range(3))
        gate = (-torch.rand(1, T, H, 128, generator=g) * 3.0).to(dev)
        b = (torch.rand(1, T, H, 128, generator=g)).to(dev)
        w = (torch.rand(1, T, H, 128, generator=g)).to(dev)
        audit = Audit()
        with audit:
            chunk_gdn2(q, k, v, gate, b, w, launcher=os.environ.get("LAUNCHER", "inprocess"))
        rows = {n: {"count": c, "class": classify(n)} for n, c in sorted(audit.ops.items())}
        banned = [n for n, r in rows.items() if r["class"] in ("BANNED", "other")]
        out[str(dtype).replace("torch.", "")] = {"ops": rows, "failing": banned,
                                                 "verdict": "PASS" if not banned else "FAIL"}
        print(f"{dtype}: {len(rows)} distinct aten ops, verdict "
              f"{out[str(dtype).replace('torch.','')]['verdict']}", flush=True)
        for n, r in rows.items():
            print(f"   {r['class']:7s} {n} x{r['count']}")
    print(json.dumps(out, indent=1))


main()
