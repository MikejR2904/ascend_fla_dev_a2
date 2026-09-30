#!/bin/bash
# msprof device-time over the a2 GDN chunk backward and the fla Triton baseline.
# Parses op_summary_*.csv for per-op Task Duration -> per-call device us.
#
#   ASCRIPTOR_WORKSPACE=... FLA_ROOT=... ASCEND_RT_VISIBLE_DEVICES=<d> \
#       ITERS=60 SHAPES="s512 s2048" ./msprof_bwd.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../../../.." && pwd)}"
export PYTHONPATH="${ASCRIPTOR_WORKSPACE:?set ASCRIPTOR_WORKSPACE}/library:$REPO${FLA_ROOT:+:$FLA_ROOT}"
MSPROF="${MSPROF:-msprof}"
ROOT="${ROOT:-$HERE/prof_bwd}"
ITERS="${ITERS:-60}"
SHAPES="${SHAPES:-s512 s2048}"
rm -rf "$ROOT"; mkdir -p "$ROOT"

run_one() {  # $1=tag  $2..=python argv
  local tag="$1"; shift
  local out="$ROOT/$tag"; mkdir -p "$out"
  echo "=== msprof $tag (iters=$ITERS) ==="
  timeout 900 "$MSPROF" --application="python3 $*" \
    --output="$out" --ai-core=on --aicpu=off --runtime-api=off --task-time=on --l2=off \
    > "$out/msprof.log" 2>&1
  python3 - "$out" "$tag" "$ITERS" <<'PY'
import sys, glob, csv, os
out, tag, iters = sys.argv[1], sys.argv[2], int(sys.argv[3])
cands = glob.glob(out + "/**/op_summary_*.csv", recursive=True)
if not cands:
    log = glob.glob(out + "/msprof.log")
    print(f"  [{tag}] no op_summary csv; log tail:", open(log[0]).read().splitlines()[-3:] if log else "")
    sys.exit(0)
rows = list(csv.DictReader(open(sorted(cands)[-1])))
def col(keys):
    for k in rows[0]:
        kl = k.lower().replace(" ", "").replace("(us)", "").replace("(ns)", "")
        if any(t in kl for t in keys): return k
namec = col(["opname", "name"]); durc = col(["taskduration", "aicoretime", "totaltime", "duration"])
ns = durc and "ns" in durc.lower()
mine = [r for r in rows if any(t in str(r.get(namec, "")).lower() for t in ("gdn_chunk_bwd", "gated_delta"))]
tot = sum(float(r[durc]) for r in mine) if mine else 0.0
if ns: tot /= 1000.0
print(f"  [{tag}] launches={len(mine)} device_total={tot:.1f}us per-call={tot/iters:.2f}us (dur={durc})")
PY
}

for SHP in $SHAPES; do
  run_one "a2_bwd_$SHP"      "$HERE/bench_a2_bwd.py --shape $SHP --iters $ITERS --warmup 10"
  run_one "triton_fb_$SHP"   "$HERE/bench_triton_gdr.py --shape $SHP --mode fb --iters $ITERS --warmup 10"
  run_one "triton_fwd_$SHP"  "$HERE/bench_triton_gdr.py --shape $SHP --mode fwd --iters $ITERS --warmup 10"
done
echo "MSPROF_BWD_DONE (triton backward-only = triton_fb - triton_fwd)"
