# GDR forward + backward, A5, PyPTO-Pro

Gated delta rule (GDR — **scalar per-head** forget gate, i.e. the
`fla.ops.gated_delta_rule` / `chunk_gated_delta_rule` family, **not** GDN2's
per-channel gate) training forward and backward, as an eight-kernel pipeline
composed in Python.

## ⚠ This is not an ascriptor unit

**Read this before looking for `unit.py`.** AGENTS.md §9 describes
`kernels/projects/<soc>/` as holding this repo's own *ascriptor* units following the
unit protocol (`unit.py` exporting `make_inputs/reference/execute`, plus
`contract.json` and `run.py`). This directory does **not** follow that protocol,
because it is not ascriptor code:

- it is written in **PyPTO-Pro** (`pypto` 0.2.1), a different kernel DSL;
- it was developed in a different repository (`cann/pypto-gym`) under that repo's
  own four-stage workflow, and its authoritative home remains there;
- there is therefore no `unit.py` / `contract.json` / `run.py` / `validation.json`,
  and `tools/unit_runner.py` cannot execute it.

It is archived here as a **working reference and performance record** for the A5 GDR
family, so the measurements and the kernel sources sit alongside the ascriptor
`gdn_chunk_fwd` / `pgdn_chunk_fwd` / `gdn_chunk_bwd` units they should be compared
against. If the repo owner would rather this lived outside `kernels/projects/a5/`
(for example under `docs/research/` or a sibling `archives/` tree), it moves without
consequence — nothing in this repo imports it.

## What is here

| path | contents |
|---|---|
| `kernels/` | 9 PyPTO-Pro kernel implementations — the 8 training pipeline kernels plus `gated_delta_rule_fwd_impl.py` (the separate inference-only forward) |
| `kal/` | hardware-aware-layer sources for every kernel: the KAL module, the transpiled staged DSL, and per-kernel `NOTES*.md` |
| `tests/` | component tests, the composition test, and both NPU and CPU goldens per kernel |
| `docs/perf/PERF_REPORT.md` | the full performance record, including the refuted approaches |
| `docs/perf/round_concurrency_2026_10_14/` | raw msprof `op_summary` CSVs backing the headline numbers, all seven metric groups |

Deliberately **not** copied from the source repo: roughly 700 further historical
msprof CSVs from per-kernel tuning rounds (~3.5 MB). AGENTS.md §10 forbids
committing profiling traces, so only the curated evidence for the numbers quoted
below is included. The full set remains in the source repository.

## Measured state

Composed pipeline at B=1, T=2048, H=8, K=V=128 on a 28 cube / 56 vector
Ascend950PR (DAV_3510, CANN 9.1.0-beta.1-950). Warm medians of three
single-group `PipeUtilization` passes; all 5 composition cases pass against torch
autograd.

| kernel | µs | blk | `aic_mac` | ΣAIC | note |
|---|--:|--:|--:|--:|---|
| `gdr_fwd_train_wy` | 133.4 | 28 | 0.49 | 1.08 | −33.9% from 200.9 |
| `gdr_fwd_train_scan` | 183.2 | 24 | 0.79 | 1.48 | MAC-floored |
| `gdr_bwd_dv_local` | 37.6 | 28 | 0.69 | 1.28 | −21.7% from 47.9 |
| `gdr_bwd_reverse_scan` | 217.1 | 16 | 0.57 | 0.97 | largest component; proven floor |
| `gdr_bwd_dqkw` | 159.8 | 28 | 0.72 | 1.28 | −20.1% from 201.6 |
| `gdr_bwd_wy_matmuls` | 113.9 | 28 | 0.79 | 1.79 | −6.8% from 122.9 |
| `gdr_bwd_wy_reduce` | 40.9 | 56 | — | — | pure vector |
| `gdr_bwd_finalize` | 25.9 | 56 | — | — | pure vector |
| **total** | **912.1** | | | | from **1048.3**, **−13.0%** |

Every kernel changed during that work is **bit-identical** to its predecessor — no
tolerance was spent. Run-to-run spread on the total is ±0.3% (913.5 / 912.0 /
912.1).

### At a large shape the pipeline gets *more* efficient, and the occupancy problem disappears

B=1, **T=16384, H=16**, K=V=128 — 16× the work of the reference (8× T, 2× H). Two
`PipeUtilization` passes, totals 14246.9 and 14238.5 µs, **median 14.24 ms**.
Timing-only: the CPU oracle was not run at this size.

| kernel | µs | blk | `aic_mac` | ΣAIC | cube % | real MAC % |
|---|--:|--:|--:|--:|--:|--:|
| `gdr_fwd_train_wy` | 2051.4 | 28 | 0.474 | 1.130 | **98.5** | 46.7 |
| `gdr_fwd_train_scan` | 2826.4 | 28 | 0.810 | 1.586 | 57.0 | 46.1 |
| `gdr_bwd_dv_local` | 545.1 | 28 | 0.686 | 1.441 | **99.3** | 68.1 |
| `gdr_bwd_reverse_scan` | 3382.5 | 28 | 0.581 | 1.050 | 57.0 | 33.1 |
| `gdr_bwd_dqkw` | 2381.0 | 28 | 0.696 | 1.284 | **99.2** | 69.0 |
| `gdr_bwd_wy_matmuls` | 1566.9 | 28 | 0.830 | 1.883 | **98.6** | **81.9** |
| `gdr_bwd_wy_reduce` | 982.1 | 56 | — | — | **97.5** (vec) | — |
| `gdr_bwd_finalize` | 503.2 | 56 | — | — | **93.8** (vec) | — |
| **total** | **14238.5** | | | | | |

Three things worth carrying forward:

- **Scaling is slightly sub-linear, i.e. better than linear.** 16× the work costs
  15.6×: 14.24 ms against the 14.59 ms that 16 × 912.1 µs would predict. Per unit of
  work (chunks × heads) the cost falls from 3.563 µs to 3.477 µs, −2.4%. Larger
  shapes amortise fixed cost and fill cores better, so the reference shape is the
  pessimistic one — don't extrapolate the 912 µs figure linearly and expect it to be
  conservative.
- **`reverse_scan`'s lattice limit is a property of H=8, not of the kernel.** At the
  reference shape it launched 16 of 28 cube blocks because work items =
  `B·H·ceil(V/TV)` = 1·8·2 = 16, and the reachable counts {8, 16, 32, …} skip (16,
  28]. At H=16 that becomes 32 items, so it launches all 28 blocks. The 57.0%
  occupancy that remains is **wave quantization, not idle cores**: 32 items over 28
  cores is 2 waves, and 32/(2·28) = 57.1%, which the measurement reproduces to 0.1
  point. `train_scan` sits at 57.0% for exactly the same reason. So the honest
  statement is that these two are launch-quantized, not under-occupied, and the fix
  is a work-item count that is a multiple of 28 — not more parallelism.
- **Six of eight kernels clear the 80% occupancy bar at this shape** (93.8–99.3%),
  versus two of eight at the reference shape. On *real MAC* utilization
  (`cube % × aic_mac`), though, only `wy_matmuls` clears it at 81.9%; `reverse_scan`
  is 33.1% and that is the fp32 MMAD rate below, not a scheduling defect.

`aic_mac` is a pipe-busy ratio, not an efficiency: real MAC utilization is
`cube_utilization × aic_mac_ratio`, which for `train_wy` turns a 90% reading into
31%. ΣAIC is the sum of the cube pipe ratios and acts as an in-core overlap meter —
ΣAIC ≈ 1.0 (as on `reverse_scan`) means the pipes are serialized with no overlap
left to redistribute.

## The one result most worth carrying to the ascriptor GDN/PGDN units

**fp32 cube matmul issues at about 1/16 of peak on this part.** A standalone probe
of N back-to-back `[64,64]×[64,64]` MMADs at `block_dim=24`:

| operand dtype | ns per MMAD | `aic_mac` |
|---|--:|--:|
| `DT_FP32` | **623** | 0.991 |
| `DT_BF16` (fp32 accumulate) | **42** | 0.889 |

`aic_mac` 0.991 on the fp32 row is the point: that is MAC-pipe *throughput*, not a
dependency stall, so no accumulator split, prefetch or barrier move can touch it.
The kernels here confirm it independently — 650 ns/MMAD on `train_scan`, 644 ns on
`reverse_scan`, within 3.4% of the probe. Price any cube pipe before scheduling
against it:

```
ns per MMAD = aic_mac_ratio × Task Duration ÷ MMADs issued per core
```

`pl.matmul` pairs lhs and rhs at the **same** dtype (FP16×FP16, BF16×BF16,
FP32×FP32, HF8, FP8 → FP32); there is no fp32×bf16 pairing and no HF32/TF32 knob
(msprof's "HF32 Eligible = YES" is metadata). So the unit of conversion is a
**matmul closure**, not a tensor.

### bf16 was attempted here and reverted — do not read the A/B numbers as a win

Pointing operands at pre-made bf16 workspaces measured −47.7% on `train_scan` and
−55% on `reverse_scan`. Those A/Bs priced **only the consumer side**. When
`train_wy` + `train_scan` were actually converted, `train_scan`'s MAC collapsed as
predicted (145.1 → 20.1 µs absolute, `aic_mac` 0.794 → 0.117) but **only 12.6 µs of
that 125 µs reached the clock**, and the pipeline went 909.4 → 934.6 µs, i.e. 2.8%
*worse*. It was reverted. Three mechanisms:

1. **Self-produced operands must be materialised.** Three of `train_scan`'s eight
   operands are its own outputs; writing them in bf16 beside the fp32 publics put
   +41.7 µs of vector MTE3 on the per-chunk recurrence — its critical path.
2. **Stores are pass-bound, not byte-bound.** +22% bytes but +75% store passes →
   +75% pipe busy.
3. **Dual-emit taxes untouched kernels.** +28 MB of GM footprint moved `dv_local`
   37.2 → 42.4 µs with its code unchanged — L2 pressure.

Also: a kernel whose clock tracks vector busy time ignores MAC entirely. `train_wy`
had 33 µs of MAC removed (ΣAIC 1.090 → 0.652, the cube going *idle* rather than
fast) and got **slower**.

bf16 is therefore **unproven here, not refuted**: the prize sits in the four
backward consumers (`reverse_scan` alone holds 123.6 µs of absolute MAC at the 1/16
rate), and the producers' bf16 twins only pay for themselves once those consumers
migrate and the fp32 originals are deleted. That migration was not run. Relevant to
this repo because `gdn_chunk_fwd_bf16`, `pgdn_chunk_fwd_bf16` and
`gdn_chunk_bwd_bf16` make the same bet on ascriptor — the lesson is to measure the
producer side, not just the consumer side.

Two further constraints found the hard way: **GM→L1 dtype conversion is forbidden**,
not merely slow ("源与目的数据类型位宽必须相同"), so the cast must live on the
producer's store side; and the one free direction is **L0C→GM, which converts
FP32→BF16/FP16 in fixpipe at no cost**. On precision, rounding an operand that feeds
a matrix inverse is amplified ~10× (the `attn` operand reached relative L2 1.9e-2,
past the 2e-2 bar, and composition failed), while rounding the *result* of the
inverse as a matmul operand is safe (~2e-3).

## Approaches refuted, with the evidence

These cost more than the wins did; `docs/perf/PERF_REPORT.md` §8 and §9 carry the
full workings.

- **Parallelising either scan is dead.** The propagator is a `[K,K]` = `[128,128]`
  matrix, not a scalar, because the chunk update depends on the state through the
  delta-rule correction. A scalar blocked scan gives relative L2 **1.07e-01
  identically at G=2, 4 and 8** — a structural error, against a 2e-2 bar — while the
  matrix version gives 2.35e-16, and `‖A − aI‖ / ‖aI‖` has median 1.0e54. The matrix
  form costs `K²/(C·V)` = 2× the serial body, i.e. **4× baseline device work at every
  G ≥ 2**, and the best critical paths are 1.57×/1.66× worse. The low-rank escape
  fails too (rank ≤ C = 64 exceeds K = 128 at M = 2).
- **`reverse_scan` occupancy is a lattice, not a knob.** Work items =
  `B·H·ceil(V/TV)` = 1·8·2 = 16 of 28 cube blocks. Every refinement doubles a
  power-of-two axis, so reachable counts are {8, 16, 32, …} and nothing lands in
  (16, 28]. A V-split costs +20%, a K-split +13% and is structurally impossible.
- **The UB→L1 bridge regresses 7.5–12%**, built correctly: `pl.load` GM→L1 converts
  ND→NZ free in the DMA descriptor while `pl.move` UB→UB pays on the V pipe, about
  1.9 : 1 against. The only route left is having the producing VFs write NZ directly.
- **`wy_reduce` and `finalize` are not measurable** on this harness: ~40 µs kernels
  swing 24% run to run against ≤0.8% for kernels over 100 µs. Not pursued, by choice.

## Environment

Device details, SSH access and CANN paths are deliberately absent from this tree per
AGENTS.md §5 — see the git-ignored `machine_specs.md`. One finding worth recording
because it contradicts §5's own table: the A5 box used here reports CANN
**9.1.0-beta.1** yet **does** carry the `ascend950` built-in op package, whereas §5
states that 9.1.0 machines carry only the 910 series and that 9.2.0 is needed for
`ascend950`. Treat the opp check as per-machine, not inferable from the CANN version.

## Provenance

Developed in `cann/pypto-gym` (gitcode) as the `gated_delta_rule` PyPTO-Pro
operator; the concurrency work summarised above is dated 2026-10-14. The
methodology was generalised into that repo's performance knowledge-card library as
`metrics-01` (occupancy vs efficiency vs real FLOPs), `cube-08` (prefetch as
hoist + depth + independent mutex), `cube-09` (masked transpose identity),
`cube-10` (MTE2 descriptor granularity), `sched-03` (the occupancy lattice),
`sched-04` (the scan propagator feasibility test), `vec-16` (recurrence per-step
overhead) and `vec-17` (strided-side swap for bank conflicts).
