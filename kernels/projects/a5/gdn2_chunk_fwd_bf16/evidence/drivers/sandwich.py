"""Same-card three-round sandwich: baseline / candidate / baseline.

baseline is the existing FP32 path of this very operator, labelled as such; candidate is the new
BF16 path. No speed target is set; the numbers are reported as measured.
"""
import json, os, statistics, sys, time
sys.path.insert(0, os.environ["REPO"])
import torch, torch_npu  # noqa: F401
from ascend_fla.ops.gdn2_chunk_fwd import chunk_gdn2, prepare

T, H = int(os.environ.get("T", 1024)), int(os.environ.get("H", 16))
REPS, WARMUP = int(os.environ.get("REPS", 20)), int(os.environ.get("WARMUP", 3))
prepare(block_dim=8)                      # both operator sets before anything executes


def inputs(dtype):
    g = torch.Generator().manual_seed(4242)
    mk = lambda s: (torch.randn(1, T, H, 128, generator=g) * s)
    q, k, v = (mk(0.5).to(dtype).to("npu") for _ in range(3))
    gate = (-torch.rand(1, T, H, 128, generator=g) * 3.0).to("npu")
    b = torch.rand(1, T, H, 128, generator=g).to("npu")
    w = torch.rand(1, T, H, 128, generator=g).to("npu")
    st = (torch.randn(1, H, 128, 128, generator=g) * 0.1).to("npu")
    return q, k, v, gate, b, w, st


def timeit(dtype):
    q, k, v, gate, b, w, st = inputs(dtype)
    for _ in range(WARMUP):
        chunk_gdn2(q, k, v, gate, b, w, initial_state=st)
    torch.npu.synchronize()
    xs = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        chunk_gdn2(q, k, v, gate, b, w, initial_state=st)
        torch.npu.synchronize()
        xs.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": statistics.median(xs), "min_ms": min(xs),
            "p90_ms": sorted(xs)[int(0.9 * len(xs)) - 1], "reps": REPS}


rounds = [("baseline_fp32", torch.float32), ("candidate_bf16", torch.bfloat16),
          ("baseline_fp32", torch.float32)]
out = []
for i, (label, dtype) in enumerate(rounds, 1):
    r = timeit(dtype)
    r.update(round=i, label=label, dtype=str(dtype).replace("torch.", ""), T=T, H=H)
    out.append(r)
    print(f"round {i} {label:15s} median {r['median_ms']:.3f} ms  min {r['min_ms']:.3f}  "
          f"p90 {r['p90_ms']:.3f}", flush=True)
base = statistics.median([r["median_ms"] for r in out if r["label"] == "baseline_fp32"])
cand = [r["median_ms"] for r in out if r["label"] == "candidate_bf16"][0]
print(f"\nbaseline FP32 median of the two rounds: {base:.3f} ms")
print(f"candidate BF16: {cand:.3f} ms   ratio candidate/baseline: {cand / base:.3f}")
print(json.dumps({"rounds": out, "baseline_median_ms": base, "candidate_median_ms": cand}, indent=1))
