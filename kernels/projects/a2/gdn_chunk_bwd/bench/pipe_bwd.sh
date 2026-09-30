#!/bin/bash
# msprof PipeUtilization over the a2 GDN chunk backward -> per-pipe ratios, to
# find the bounding pipe and drive it toward the >90% utilization bar.
#
#   ASCRIPTOR_WORKSPACE=... ASCEND_RT_VISIBLE_DEVICES=<d> SHAPE=s2048 ./pipe_bwd.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../../../.." && pwd)}"
export PYTHONPATH="${ASCRIPTOR_WORKSPACE:?set ASCRIPTOR_WORKSPACE}/library:$REPO"
MSPROF="${MSPROF:-msprof}"
SHAPE="${SHAPE:-s2048}"
OUT="${OUT:-$HERE/pipe_bwd/$SHAPE}"
rm -rf "$OUT"; mkdir -p "$OUT"

echo "=== PipeUtilization a2_bwd $SHAPE ==="
timeout 900 "$MSPROF" --application="python3 $HERE/bench_a2_bwd.py --shape $SHAPE --iters 40 --warmup 8" \
  --output="$OUT" --ai-core=on --aic-metrics=PipeUtilization --task-time=on \
  > "$OUT/msprof.log" 2>&1

python3 - "$OUT" <<'PY'
import sys, glob, csv, statistics
out = sys.argv[1]
cands = glob.glob(out + "/**/op_summary_*.csv", recursive=True)
if not cands:
    print("  no op_summary; log tail:", open(glob.glob(out + "/msprof.log")[0]).read().splitlines()[-4:]); sys.exit(0)
rows = [r for r in csv.DictReader(open(sorted(cands)[-1])) if "gdn_chunk_bwd" in str(r).lower()]
if not rows:
    print("  no gdn_chunk_bwd rows"); sys.exit(0)
namec = next((k for k in rows[0] if "name" in k.lower()), None)
keys = [k for k in rows[0] if any(t in k.lower() for t in
        ("ratio", "vec_", "mac_", "mte", "scalar", "cube", "aic", "aiv", "bound"))]
# per stage: which stage dominates, and its bounding pipe
from collections import defaultdict
by = defaultdict(list)
for r in rows:
    by[str(r.get(namec, "?"))].append(r)
print(f"  {len(rows)} launches across {len(by)} stages; per-pipe mean/max:")
for k in keys:
    vals = []
    for r in rows:
        try: vals.append(float(r[k]))
        except: pass
    if vals:
        print(f"    {k:34s} mean={statistics.mean(vals):.4f} max={max(vals):.4f}")
PY
echo "PIPE_BWD_DONE"
