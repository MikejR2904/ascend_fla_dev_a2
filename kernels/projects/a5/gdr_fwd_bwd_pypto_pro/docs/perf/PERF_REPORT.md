# gated_delta_rule — performance

This directory now covers two related deliverables:

- the **inference forward** kernel, `gated_delta_rule_fwd_impl.py` — one fused kernel, `q,k,v,g,beta → o`;
- the **training forward+backward** path, eight single-kernel operators that together produce `o` and the five gradients `dq, dk, dv, dg, dbeta`.

The headline for the training path: **912 µs of device time for the full forward+backward** at B=1, T=2048, H=8, K=V=128, with every component correct against torch autograd. That is down from 1049 µs after two pipe-concurrency rounds on 2026-10-14 (§8 and §9), which took `gdr_fwd_train_wy` −33.9%, `gdr_bwd_dv_local` −21.7%, `gdr_bwd_dqkw` −20.1%, `gdr_bwd_wy_matmuls` −6.8% and `gdr_fwd_train_scan` −4.6% **without touching the maths** — every changed kernel's output is bit-identical to before. The rest of this document is how that number was obtained, which parts of it are worth attacking next, and — importantly — which of the older numbers in this tree you should not compare against it.

**If you are here to judge whether these kernels are "using the hardware", read §8.1 first.** The
single most misread number in this tree is `cube_utilization`: it is a whole-machine *time
occupancy* and it counts a core as busy whether that core is multiplying or merely moving bytes.
`gdr_fwd_train_wy` sat at 90% `cube_utilization` with `aic_mac` at 0.344 — i.e. 31% of the
machine's MAC capacity — and at 0.46 GB/s of main-memory traffic against a ~3250 GB/s roof. It was
neither compute-bound nor bandwidth-bound; it was dependency-stalled, and 90% hid that completely.

## Read this first: four ways these measurements lie

Every one of these cost us real time during the campaign. They are not hypothetical.

**1. The two hosts are not the same machine.** The forward-only numbers in §5 were taken on a host reporting **32 cube / 64 vector** cores. Everything else here comes from a host reporting **28 cube / 56 vector**. These are both Ascend950PR, both DAV_3510, and they are *not* interchangeable: the identical, unchanged `gdr_bwd_dqkw` kernel measured **~2406 µs** on the first host and **~203 µs** on the second. That is a factor of twelve with no code change. Never compare a number across the two, and always read the core count live from `get_platform_info()` rather than assuming it.

**2. The first occurrence of a kernel in a profile includes JIT compilation.** From the shipped composition profile, first occurrence versus the fourth, in µs:

| kernel | 1st | 4th (warm) |
|---|--:|--:|
| `gdr_bwd_dqkw` | 645.6 | 202.8 |
| `gdr_bwd_reverse_scan` | 632.2 | 217.6 |
| `gdr_bwd_dv_local` | 321.5 | 47.8 |

A 3–7× error, silently. Every duration in this report is a warm reading. The trap bit us once in the other direction too: `gdr_fwd_train_scan` was recorded at 1149.9 µs by a driver that did one warm-up and one profiled call, while the same unmodified file measures ~380 µs as the minimum of three warm calls. The tell was that the *pipe ratios* agreed between the two readings to a fraction of a percent — when durations disagree and ratios match, suspect the sampling, not the kernel. (Note it is not a universal rule that the first row is slower: `gdr_bwd_finalize` reads 22.4 then settles at ~26.9.)

**3. `cube_utilization` is an occupancy signal, not an efficiency one — and it is not the sum of anything.** It has a different denominator from the pipe ratios, so trying to reconcile the two is a category error. The exact definition, reverse-engineered from the data and stated in §3, is `aic_total_cycles / (device cube cores × 1.65 GHz × Task Duration)`: the share of the **whole machine's** cube capacity this kernel consumed over its wall-clock window. The pipe ratios divide by `aicore_time` — the busy window of the *longest* core — so they overlap and routinely sum past 1.0. One number is capped at 1.0 and the other is not. When `cube_utilization` drops while the pipe ratios hold steady, you are looking at idle blocks, not slow compute, and the number lands on an exact fraction: `gdr_bwd_reverse_scan` measured 57.127% on 28 cores, which is 16/28, because its 32 work items chunked into `ceil(32/28) = 2` rounds and left 12 blocks with nothing to do. But idle blocks are only one of three ways it can fall — §3 factors it — and `gdr_fwd_train_scan`'s 18.965% in `round_newhost` is mostly the *third* way, launch overhead from the artifact in caveat 2, not an occupancy problem at all. Factor it before theorising.

**4. A wave-clean shape can hide the entire win.** Measuring `reverse_scan` at B=7, H=4 (28 items, exactly one wave) puts the tuned and untuned kernels within 0.75% of each other — 436.4 µs against 433.1 µs. At the production shape the same pair differ by 213 µs. Profile the shape that matters; treat quantisation as diluting small deltas rather than as a reason to switch shapes.

A fifth, mechanical one: a multi-group `msprof` collection sometimes exports **zeroed ratio columns** while populating Task Duration. A 0.0 there reads as a genuinely idle pipe. Use single-group `PipeUtilization` passes, which is what produced every ratio below. The same trap has a second form: in `composition/op_summary_composition_prod.csv` only the **first** occurrence of each kernel carries ratios at all, and every repeat launch exports them as 0.0 — so the warm rows that give the honest *durations* in §1 give no pipe data, and the per-pipe tables in §3 are taken from the per-operator `round_newhost` collections instead. `aicore_time` on that first row is not contaminated by the JIT the way Task Duration is (`dqkw` reads 645.6 µs duration against 202.5 µs `aicore_time`), which is how the two sources can be checked against each other.

## 1. The training forward+backward pipeline

Correctness first, because the performance number means nothing without it. `tests/ops/pypto_pro/gated_delta_rule/test_composition.py` chains all eight wrappers — GM-materialised between stages, no host arithmetic — and compares `o` plus all five gradients against the CPU autograd oracle. It passes **5 of 5 shapes**, including the production shape, and the per-boundary bisection against the cumulative staging goldens is clean on **all 30 intermediate tensors**. Component-level passes alone would not have proven this; eight correct pieces can still be wired together wrongly.

Device time at B=1, T=2048, H=8, K=V=128, from one single-group `PipeUtilization` pass, warm readings:

| # | component | µs (2026-10-14) | share | was | Δ | `aic_mac` | ΣAIC |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | `gdr_fwd_train_wy` | 132.7 | 15% | 200.9 | **−33.9%** | 0.502 | 1.059 |
| 2 | `gdr_fwd_train_scan` | 182.6 | 20% | 191.5 | −4.6% | **0.795** | 1.483 |
| 3 | `gdr_bwd_dv_local` | 37.5 | 4% | 47.9 | **−21.7%** | 0.703 | 1.282 |
| 4 | `gdr_bwd_reverse_scan` | 217.8 | 24% | 217.4 | — (at its floor, §8.4/§9.4) | 0.568 | 0.975 |
| 5 | `gdr_bwd_dqkw` | 161.0 | 18% | 201.6 | **−20.1%** | 0.720 | 1.287 |
| 6 | `gdr_bwd_wy_matmuls` | 114.5 | 13% | 122.9 | −6.8% | **0.812** | 1.785 |
| 7 | `gdr_bwd_wy_reduce` | 40.4 | 4% | 39.9 | — (not measurable, §9.5) | — | — |
| 8 | `gdr_bwd_finalize` | 25.8 | 3% | 26.1 | — (not measurable, §9.5) | — | — |
| | **total** | **912.4** | | 1048.3 | **−13.0%** | | |

The "was" column is a baseline re-measured on 2026-10-14 on the current host with the same
tool, not the 1049.10 figure this table used to carry — it reproduced that figure to 0.1%
(1048.3 against 1049.10), which is what licenses the comparison. Both columns are medians of
the warm rows of a single-group `PipeUtilization` pass; the optimized column is the median of
three independent passes (915.7 / 912.4 / 911.3 µs, so the run-to-run spread on the total is
about ±0.3%, and on an individual kernel over 100 µs about ±0.8% — but up to 24% on the two
~40 µs kernels, which is why rows 7 and 8 are marked unmeasurable; see §9.5).

Two rounds produced these deltas: **§8** (barrier placement, a mask-symmetry identity and L0C
mutex ownership on `dqkw`; the `fwdsubst` per-step overhead on `train_wy`) and **§9** (a UB
bank-conflict fix on `train_wy`, GM→L1 descriptor granularity on `train_scan`, and the barrier
and mutex playbook on `dv_local` and `wy_matmuls`). Both sections also record what was measured
dead, which is the more valuable half. Every changed kernel is bit-identical to its predecessor.
Raw rows: `round_concurrency_2026_10_14/`.

**Against the 80%-on-the-bounding-pipe bar**: `wy_matmuls` meets it (0.812), `train_scan` is at
0.795 in-pipeline and 0.814 standalone. The rest are short for reasons given with evidence in
§9.8 — not for lack of attempts.

Reproduce with `python test_composition.py --timing`; raw rows in `composition/op_summary_composition_prod.csv`. The timing mode is opt-in so the default correctness run stays clean.

The same run's host wall-clock total is **1923 µs**. The ~874 µs gap is per-launch dispatch plus the eight individual `torch.npu.synchronize()` calls the timing mode brackets each wrapper with — it is not device idle time. If you benchmark this path end to end, expect to pay that dispatch cost once per launch, and do not use wall-clock to judge a kernel.

There is no load concentrated in one place: the top five components are each 12–21% of the total. Halving any single one buys at most a tenth of the pipeline.

That table is durations only. For what each kernel is actually doing with the hardware — all six cube pipes, all four vector pipes, icache, block counts and occupancy — see §3, Tables A–D. The ratio columns in this composition CSV are populated on the first (cold) occurrence only, so the per-pipe numbers come from the per-operator `round_newhost` collections rather than from this run.

## 2. How it scales with shape, and where the envelope actually ends

An msprof sweep over the pipeline, one axis at a time from the reference shape, single-group
`PipeUtilization` passes, warm readings. Raw data per shape in `sweep/`, method in `sweep/SWEEP.md`.
The sweep's own T=2048 total (1052.9 µs) agrees with §1's independent run (1049.1 µs) to 0.4%,
which is the best evidence either number is real.

**The supported envelope is narrower than the SPEC.** Two limits are hard-coded in shipped kernels
and the sweep hit both:

- **B = 1 only.** `GDR_BWD_DQKW` declares every tensor's batch axis as the literal `1`
  (`gdr_bwd_dqkw_impl.py`), so B=2 fails with `ValueError: Tensor parameter 'v_new' axis 0
  mismatch: expected 1, got 2` before any kernel launches. Several components *do* support B>1
  individually — `train_scan` passes a B=3 case, `dv_local`, `wy_reduce` and `finalize` were all
  changed to dynamic batch — but the pipeline is gated by the one that was not. The SPEC allows
  B ∈ [1,16]. Fixing it is one annotation per tensor plus a correctness run; it has not been done.
- **K, V ≤ 128.** `gdr_fwd_train_wy_wrapper` asserts `K % 64 == 0 and K <= 128`. K=V=256 fails
  there. `train_scan` v2 does support 256, so again the pipeline is capped by a different
  component than the one that advertises the range.

So "B/T/H/K/V" is really T, H and K/V ≤ 128 at B=1. Treat the B axis as untested.

**T scales linearly once there is enough work.** Per-chunk cost flattens at ~32.5 µs from T=1024;
small T pays a fixed-cost premium:

All eight kernels, µs, at B=1, H=8, K=V=128:

| T | N | train_wy | train_scan | dv_local | rev_scan | dqkw | wy_matmuls | wy_reduce | finalize | **total** | µs/chunk |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 256 | 4 | 50.97 | 25.99 | 12.23 | 30.97 | 41.47 | 25.14 | 8.12 | 8.81 | **203.70** | 50.9 |
| 512 | 8 | 69.91 | 48.25 | 16.50 | 57.83 | 60.68 | 35.35 | 13.53 | 9.76 | **311.80** | 39.0 |
| 1024 | 16 | 107.74 | 94.29 | 25.92 | 112.11 | 99.93 | 58.96 | 20.49 | 13.87 | **533.30** | 33.3 |
| 2048 | 32 | 201.68 | 191.63 | 47.50 | 218.33 | 202.56 | 120.73 | 43.02 | 27.45 | **1052.91** | 32.9 |
| 4096 | 64 | 371.60 | 393.89 | 89.71 | 431.42 | 390.11 | 234.94 | 115.92 | 52.03 | **2079.62** | 32.5 |

Every kernel is close to linear in N over the top three rows; `wy_reduce` is the one that grows
slightly faster than the rest (43.0 → 115.9 µs for 2× the chunks), consistent with it being the
bandwidth-bound component of the set.

T=256 costs 56% more per chunk than T=4096 — launch and barrier overhead that does not amortise.
No super-linear blow-up anywhere through T=4096. The dominant kernel shifts with T: `train_wy` and
`dqkw` lead at T=256, while `reverse_scan` becomes the single largest by T=4096 (431 µs), because
its serial critical path grows faster than the chunk-parallel kernels around it.

All eight kernels, µs, at B=1, T=2048, K=V=128 (B could not be varied — see the envelope above, so
H carries this axis and total work grows with it):

| H | train_wy | train_scan | dv_local | rev_scan | dqkw | wy_matmuls | wy_reduce | finalize | **total** |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 108.06 | 180.80 | 24.91 | 219.78 | 100.40 | 60.20 | 20.45 | 14.61 | **729.23** |
| 8 | 201.68 | 191.63 | 47.50 | 218.33 | 202.56 | 120.73 | 43.02 | 27.45 | **1052.91** |
| 16 | 372.17 | 396.40 | 87.25 | 431.91 | 389.35 | 227.99 | 116.62 | 49.74 | **2071.44** |
| 28 | 667.35 | 509.35 | 154.67 | 435.11 | 659.24 | 391.33 | 259.01 | 92.80 | **3168.87** |
| 32 | 716.68 | 634.39 | 162.55 | 645.83 | 746.04 | 438.32 | 257.83 | 108.12 | **3709.78** |

Read down the `rev_scan` column rather than the total: it barely moves from H=4 to H=8, and barely
again from H=16 to H=28, then jumps. That is the signature worth understanding.

**H is a staircase, not a curve — and this is the one worth designing around.** `gdr_bwd_reverse_scan`
folds a V-tile axis into its item space, so it launches `items = B·H·vtiles` (vtiles = 2 at V=128).
Its latency is quantised in whole launch waves of ~218 µs over 28 cube cores:

| H | items | waves = ⌈items/28⌉ | predicted | measured | error |
|--:|--:|--:|--:|--:|--:|
| 4 | 8 | 1 | 218.3 | 219.8 | +0.7% |
| 8 | 16 | 1 | 218.3 | 218.3 | 0.0% |
| 16 | 32 | 2 | 436.6 | 431.9 | −1.1% |
| 28 | 56 | 2 | 436.6 | 435.1 | −0.3% |
| 32 | 64 | 3 | 654.9 | 645.8 | −1.4% |

Five points, all within 1.4% of `waves × 218.3 µs`. The practical consequences are sharp:
H=4 → H=8 **doubles the work for free** (219.8 → 218.3 µs) because the first wave was half empty;
H=16 → H=28 takes **1.75× the work at the same cost** (431.9 → 435.1 µs); and H=28 → H=32 costs
**+49% for +14% work**, because item 57 opens a third wave that then runs 7/8 empty. If you can
choose H, land `B·H·vtiles` just under a multiple of 28.

The other seven kernels are chunk-parallel (`items = B·H·N` = 256 at the reference shape), so they
are already core-saturated and grow roughly linearly in H — `train_wy` 201.7 → 372.2 → 667.4 µs for
H = 8 → 16 → 28. Nothing to win there by shape choice.

`gdr_fwd_train_scan` is the interesting middle case. It also absorbs H=4 → 8 almost free
(180.8 → 191.6 µs for double the work) and it also softens at H=16 → 28 (396.4 → 509.4 µs, 1.28×
for 1.75× the work), but its steps are gentler than `reverse_scan`'s because the selector PANKO
gave it **re-picks VT per shape**, so its item count adapts instead of being fixed. I have not
reconstructed its per-shape item count, so I am not fitting a wave model to it here — the
qualitative behaviour is what the data supports.

Overall the pipeline scales better than linearly in H across the measured range: H = 8 → 28 is 3.5×
the work for 3.01× the time. The exception is the last step, H = 28 → 32, which costs 1.17× for
1.14× work — that is the third `reverse_scan` wave opening and then running mostly empty.

All eight kernels, µs, at B=1, T=2048, H=8:

| K=V | train_wy | train_scan | dv_local | rev_scan | dqkw | wy_matmuls | wy_reduce | finalize | **total** |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 64 | 145.06 | 137.13 | 38.20 | 185.33 | 92.02 | 78.16 | 24.42 | 18.90 | **719.22** |
| 128 | 201.68 | 191.63 | 47.50 | 218.33 | 202.56 | 120.73 | 43.02 | 27.45 | **1052.91** |
| 256 | — | — | — | — | — | — | — | — | **unsupported** (`AssertionError` in `train_wy`) |

**K and V are sub-linear in area.** Quadrupling K·V (64,64 → 128,128) costs 1.46× total time, not
4×, because most components are bound by other axes. `dqkw` is the exception and the steepest riser
(92.0 → 202.6 µs, 2.2×), consistent with it already sitting at 93.9% UB occupancy at K=V=128 —
it is the component with the least room left.

**At the floor** (T=128, H=4, K=V=64) the pipeline totals ~84 µs and `finalize` is bound by its
store pipe (`aiv_mte3` 0.69) rather than by compute — the same fixed-cost story as small T.

Correctness was re-verified against the CPU autograd oracle, with the per-boundary bisection, for one
new shape on each axis (T=256, H=16, K=V=64) — all three pass. Every other row above is timing-only.

## 3. Per-operator standalone numbers and the full pipe profile

Tables A, B and C come from `<op>/round_newhost/op_summary_PipeUtilization.csv` — one single-group
`PipeUtilization` collection per operator on the current **28 cube / 56 vector** host, at the production
shape. Single-group because a multi-group pass zeroes the ratio columns (caveat 5). Three of the eight
kernels were tuned *after* that archive was taken, so their `round_newhost` rows are pre-tuning; Table D
carries their shipped profiles, and those are the ones §1 and §4 quote. Table E is the one kernel for
which the other six msprof metric groups — memory, L2 and resource-conflict — were also collected on
this host.

### What the three numbers actually measure

These columns are easy to mix up, and the pipe ratios do **not** add up to `cube_utilization` — they
cannot, because the three have different denominators. All three relationships below were
reverse-engineered from the CSVs in this tree and hold numerically; the checks are stated so you can
redo them.

**`aicore_time` / `aiv_time` is the longest single core's busy window**, not a sum and not a mean.
`round3_prod/summary.txt` labels it 核最长耗时 (longest-core elapsed) and derives launch overhead as
`Task Duration − aicore_time`; the repo's own `msprof_perf_summary.py` independently assumes the same
thing, estimating device frequency as `aicore_time / max_core_cycles`.

**`aic_*_ratio` and `aiv_*_ratio` are `pipe_time / core_time`** — the fraction of that longest-core
window during which that one pipe was busy. Checked against the `*_time(us)` partner column on all 43
rows in this tree that carry both: maximum deviation 0.0005, which is just the rounding of the
three-decimal ratio column. So a ratio is an occupancy of one pipe, not a share of the kernel, and
because the pipes run concurrently **a row sum above 1.0 is normal and is the saturation signal worth
reading**: it means the core has no dead slot left in which to hide another operand fetch.

**`cube_utilization(%)` is a whole-device time occupancy:**

```
cube_utilization(%) = aic_total_cycles / (device_cube_cores × 1.65 GHz × Task Duration) × 100
```

Note `device_cube_cores` — the cube cores **the chip has** (28 here, 32 on the previous host), *not*
this kernel's `Block Num`. Solving the identity for that divisor on all 47 rows in this tree with a
non-zero `cube_utilization` returns 28.000 for every current-host archive and 32.000 for every
previous-host one, across launches from 2 to 32 blocks, and the implied clock comes out at 1650.00 MHz
every time. It also holds in `baseline/` on the previous host: `451867 / (32 × 1650 × 39.464)` =
21.686%, which is that row's reported value exactly.

So the question each number answers is different: a pipe ratio asks *"while the cores were working,
which pipes were busy?"*, and `cube_utilization` asks *"how much of the machine did this kernel
occupy?"*. The second is capped at 1.0, the first is not, and neither constrains the other — a kernel
can sit at 90% device occupancy with every pipe near idle, or at 19% occupancy with `mac` at 0.75.

**`cube_utilization` factors into three independent losses**, which is what makes it actionable:

```
cube_utilization = (Block Num / device cores)   ← launch width:    blocks never launched
                 × (mean core busy / longest)   ← load balance:    wave quantisation and skew
                 × (longest / Task Duration)    ← window coverage: launch + JIT overhead
```

| kernel (archive) | reported | = width | × balance | × coverage | where the loss is |
|---|--:|--:|--:|--:|---|
| `train_wy` (`round_newhost`) | 90.076% | 1.000 | 0.905 | 0.996 | healthy on all three |
| `wy_matmuls` (`round_newhost`) | 89.541% | 1.000 | 0.903 | 0.992 | healthy on all three |
| `reverse_scan` (`round_newhost`) | 57.127% | 1.000 | 0.572 | 0.999 | **all balance** — the 16/28 idle blocks of caveat 3 |
| `train_scan` (shipped, §1 run) | 84.660% | 0.857 | 0.991 | 0.996 | width: ceiling is 24/28, and it hits 98.8% of it |
| `train_scan` (`round_newhost`) | 18.965% | 1.000 | 0.570 | **0.333** | **mostly coverage** — the caveat-2 artifact, not the kernel |

That last row is the practical payoff. Read as a bare number, 18.965% looks like a catastrophic
occupancy problem; factored, two thirds of the loss is the JIT-contaminated Task Duration and the
kernel's real balance problem is the same ~0.57 that `reverse_scan` has.

### Where the cycle counters come from, and the clock

`aic_total_cycles` / `aiv_total_cycles` are not computed from any time column — they are the AI Core
PMU task-cycle counters. The raw per-core, per-task record is `AICoreOriginalData.task_cyc` in
`device_0/sqlite/aicore.db`, which msprof only writes on a separate `--aic-mode=sample-based` run;
`op_summary_<Metric>.csv` carries the **per-op aggregate** of that counter, summed over the cores that
ran the task (`msprof-guide.md` states op_summary is per-op aggregated and explicitly not per-core, and
the arithmetic above only closes if the column is the sum). One documentation trap: the field table in
`csv_fields_reference.md` glosses `aic_total_cycles` as "每个 Cube Core 执行 cycle 总数" — per *one*
cube core — but that line sits in its per-core section alongside `block_id` / `sub_block_id`. In
op_summary it is the device-wide sum.

**There is no frequency column anywhere in op_summary** — no `Current Freq`, no `Rated Freq`, which is
why the 1.65 GHz above had to be solved for rather than read. Two independent routes agree on it: the
identity solves to 1650.00 MHz on all 47 rows, and the toolchain's own per-core section back-derives the
clock as `aicore_time / max_core_cycles` and prints **1.651 GHz** in `msprof-guide.md`'s worked example.
Because the clock is derived and not read, a DVFS excursion would silently break the identity; it does
not break on any archived row here, which is what the guide's mandatory warm-up is for.

**A trap in the per-core data.** `per_core_cycles.csv` is **process-scope and unattributed** — it is
`SUM(task_cyc) GROUP BY coreid` over every task in the process, with no filter for the kernel under
test. It is therefore only meaningful when the profiled process runs essentially one kernel.
`baseline/` is not such a process, and its `summary.txt` consequently back-derives **2.922 GHz** and
reports a load spread of "76.28% → 严重问题" with a NUMA/L2-slice hypothesis attached. Both are
artifacts of summing eight cases plus the aclnn setup ops into one kernel's `aicore_time`, not findings.
Do not act on that section unless the run was single-kernel.

With that caveat stated, the balance factor does cross-check on the one archive where the run *is*
effectively single-kernel: `gdr_bwd_dv_local/round_001/module1/per_core_cycles.csv` measures `mean/max`
= 0.9568 across its 32 cores against 0.9459 derived from the op_summary row — about 1% apart, in line
with the 1.5% total-cycle difference between the two passes. That is corroboration, not proof; the
proof is the 47-row identity.

**The two vector-only kernels do have an occupancy; msprof just has no column for it.** Applying the
same formula with 56 vector cores gives `wy_reduce` **83.9%** and `finalize` **55.5%**, where
`cube_utilization` can only report 0.000. Using the same 1.65 GHz for the vector side is justified
rather than assumed: on the six MIX kernels, per-core cycle counts `aic_total_cycles / 28` and
`aiv_total_cycles / 56` agree to within 0.2% (`wy_matmuls` is the one outlier at 2.6%), so the two core
types clock identically and stay resident for the same window. These derived figures are marked
*(derived)* wherever they appear below.

**`aic_mte3_ratio` is 0.000 for all eight kernels, and that is structural rather than a defect.** Cube
results leave L0C through the fixpipe, so nothing in this family ever drives the cube core's MTE3 store
pipe. Read `aic_fixpipe_ratio` wherever you would otherwise expect a cube store cost. Do not treat that
column as headroom.

### Table A — cube-core (AIC) pipes

Task Duration and `aicore_time` are both given because they disagree for `train_scan` in exactly the way
caveat 2 describes. `Σ` is the arithmetic sum of the six pipe ratios on that row.

| operator | Task Dur µs | `aicore_time` µs | AIC blk | `mac` | `scalar` | `mte1` | `mte2` | `mte3` | `fixpipe` | **Σ** | `icache_miss` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `gdr_fwd_train_wy` | 199.41 | 198.52 | 28 | 0.346 | 0.050 | 0.057 | 0.147 | 0.000 | **0.128** | 0.728 | 0.004 |
| `gdr_fwd_train_scan` † | 1149.87 | 382.81 | 28 | 0.747 | 0.097 | 0.128 | **0.566** | 0.000 | 0.046 | 1.584 | 0.005 |
| `gdr_bwd_dv_local` | 49.27 | 48.34 | 28 | 0.541 | 0.062 | 0.077 | 0.211 | 0.000 | 0.037 | 0.928 | **0.027** |
| `gdr_bwd_reverse_scan` † | 859.65 | 858.81 | 28 | 0.571 | 0.032 | 0.068 | 0.325 | 0.000 | 0.044 | 1.040 | 0.001 |
| `gdr_bwd_dqkw` | 203.27 | 201.48 | 28 | 0.634 | 0.072 | 0.091 | 0.218 | 0.000 | 0.053 | **1.068** | 0.014 |
| `gdr_bwd_wy_matmuls` | 110.40 | 109.51 | 28 | **0.811** | 0.124 | 0.129 | 0.289 | 0.000 | **0.309** | **1.662** | 0.003 |
| `gdr_bwd_wy_reduce` | 51.55 | 0.00 | — | n/a | n/a | n/a | n/a | n/a | n/a | — | — |
| `gdr_bwd_finalize` † | 33.51 | 0.00 | — | n/a | n/a | n/a | n/a | n/a | n/a | — | — |

### Table B — vector-core (AIV) pipes

| operator | `aiv_time` µs | AIV blk | `vec` | `scalar` | `mte2` | `mte3` | **Σ** | `icache_miss` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `gdr_fwd_train_wy` | 198.55 | 56 | **0.460** | 0.052 | 0.189 | 0.185 | 0.886 | 0.015 |
| `gdr_fwd_train_scan` † | 382.83 | 56 | 0.058 | 0.040 | 0.158 | 0.175 | 0.431 | 0.003 |
| `gdr_bwd_dv_local` | 48.36 | 56 | 0.176 | 0.102 | 0.122 | 0.110 | 0.510 | 0.017 |
| `gdr_bwd_reverse_scan` † | 858.83 | 56 | 0.045 | 0.018 | 0.054 | 0.075 | **0.192** | 0.004 |
| `gdr_bwd_dqkw` | 201.50 | 56 | 0.068 | 0.037 | 0.073 | 0.054 | 0.232 | 0.009 |
| `gdr_bwd_wy_matmuls` | 109.54 | 56 | 0.031 | 0.042 | 0.061 | 0.079 | 0.213 | 0.022 |
| `gdr_bwd_wy_reduce` | 50.73 | 56 | 0.251 | 0.079 | **0.631** | 0.180 | 1.141 | 0.030 |
| `gdr_bwd_finalize` † | 32.63 | 56 | 0.261 | 0.209 | **0.735** | 0.340 | **1.545** | **0.040** |

† `round_newhost` predates this kernel's tuning — see Table D for the shipped profile.

`wy_reduce` and `finalize` launch as `AI_CORE` tasks with 56 vector blocks and **no cube blocks at all**,
which is why their whole AIC row is `n/a` and their `aicore_time` is 0.00. Their `cube_utilization` of
0.000 is likewise not a finding. The other six are `MIX_AIC` with 28 cube blocks and 56 vector blocks.
All eight report `HF32 Eligible = YES`. The same CSV rows also carry `aic_total_cycles` /
`aiv_total_cycles`; those are aggregate counters summed over blocks, not utilizations, so they are left
in the raw data rather than reproduced here.

### Table C — occupancy and the acceptance bar

The bar for this work is **80% on the bounding pipe**; where a kernel structurally cannot use a pipe,
that pipe is not counted. µs and `cube_util` here are the shipped kernels' — post-tuning for the three
marked † above. For the two vector-only kernels the `cube_util` cell gives msprof's reported 0.000 and
then the derived vector occupancy from the formula above, which is the number that actually means
something for them. The runner-up column is the second-busiest pipe **across both cores**, which is what
makes it worth having: on `train_wy` it is `aic_mac`, so the kernel is not vector-bound in the sense of
having an idle cube side, and on `wy_matmuls` it is `aic_fixpipe` rather than another operand-load pipe.

| operator | µs | bounding pipe | runner-up pipe | `cube_util` | 80% bar |
|---|--:|---|---|--:|---|
| `gdr_fwd_train_wy` | 199.4 | `aiv_vec` 46.0% | `aic_mac` 34.6% | 90.1% | meets, on occupancy |
| `gdr_fwd_train_scan` | 178.9 after tuning | `aic_mac` 80.7% | `aic_mte2` 39.5% | 84.6% | **meets** |
| `gdr_bwd_dv_local` | 49.3 | `aic_mac` 54.1% | `aic_mte2` 21.1% | 87.1% | meets, on occupancy |
| `gdr_bwd_reverse_scan` | 643.3 at B=4 | `aic_mac` 57.1% | `aic_mte2` 31.0% | 76.4% | below — see §4 |
| `gdr_bwd_dqkw` | 203.3 | `aic_mac` 63.4% | `aic_mte2` 21.8% | 88.8% | ceiling-bound, see §4 |
| `gdr_bwd_wy_matmuls` | 110.4 | `aic_mac` **81.1%** | `aic_fixpipe` 30.9% | 89.5% | **meets** |
| `gdr_bwd_wy_reduce` | 51.6 | `aiv_mte2` 63.1% | `aiv_vec` 25.1% | 0.000 — 83.9% *(derived)* | bandwidth-bound, see §4 |
| `gdr_bwd_finalize` | 15.0 after tuning | `aiv_mte3` 66.2% | `aiv_vec` 58.7% | 0.000 — 81.4% *(derived)* | bandwidth-bound, see §4 |

### What the full profile says that the bounding pipe alone does not

**`wy_matmuls` is the most saturated cube core in the family**, Σ = 1.662, and its `fixpipe` 0.309 is
2.4× the next highest (`train_wy` 0.128) and 5.8–8.4× the rest. Five matmul results per step leave
L0C through one fixpipe. It still meets the bar on `mac` 0.811, but if any kernel here is at risk of
becoming fixpipe-limited rather than MAC-limited, it is this one — and the bounding-pipe column alone
never showed that.

**`dqkw`'s Σ = 1.068 with `mac` only 0.634 is §5's "fp32 ceiling" in per-pipe form.** The non-MAC
occupancy breaks down as `mte2` 0.218 + `mte1` 0.091 + `scalar` 0.072 + `fixpipe` 0.053 = 0.434, all of
it operand traffic the frozen matmul set requires. There is no single pipe to attack: the 0.434 is spread
across four.

**`train_wy` is the only mixed kernel whose vector side is busier than its cube side** — AIV Σ 0.886
against AIC Σ 0.728 — which is exactly why its bounding pipe is `aiv_vec` and not `aic_mac`. Its
`aic_fixpipe` 0.128 is the second-highest in the family, so its cube side spends more occupancy writing
results back than multiplying (`mac` 0.346).

**The two serial scans have near-idle vector cores.** `reverse_scan` launches all 56 AIV blocks and its
busiest vector pipe is `mte3` at 0.075 (Σ 0.192); `train_scan` is Σ 0.431 against `train_wy`'s 0.886.
That is the per-pipe fingerprint of the dependency-bound conclusion in §5: the vector cores are launched
but are there for handshakes, so there was never vector-side occupancy for the three dead optimizations
in §5 to recover.

**`mte2` is the pipe to watch second on five of the six cube kernels, and first on both vector-only
ones.** The exception is `wy_matmuls`, where `fixpipe` 0.309 edges ahead of `mte2` 0.289. Pre-tuning
`train_scan` carried `aic_mte2` 0.566 — the highest operand-load occupancy anywhere in the family — and
the tuning in §4 took it to 0.395 while *raising* `mac` from 0.747 to 0.807 (Table D). On the
vector-only pair, `aiv_mte2` is the bounding pipe for `wy_reduce` (0.631) and for `round_newhost`'s
`finalize` (0.735); in the shipped `finalize` it falls to 0.480, behind both `mte3` and `vec`.

**Nothing here is instruction-fetch limited.** Every `icache_miss_rate` is ≤ 0.027 on the cube side
(`dv_local` is the high-water mark) and ≤ 0.040 on the vector side (`finalize`). Worth checking once;
not worth checking again.

### Table D — the three tuned kernels, shipped profiles

`round_newhost` predates these fixes. `train_scan` and `reverse_scan` come from the PANKO search traces,
which record the full `pipes` dict per candidate; `finalize` comes from the wave-fix archive, as the
median of the 11 warm rows in that file. The search traces do not record icache miss rates.

| operator | µs | blk AIC/AIV | `aic_mac` | `aic_scalar` | `aic_mte1` | `aic_mte2` | `aic_mte3` | `aic_fixpipe` | AIC Σ | `aiv_vec` | `aiv_scalar` | `aiv_mte2` | `aiv_mte3` | AIV Σ | `cube_util` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `gdr_fwd_train_scan` | 178.86 | 24/48 | 0.807 | 0.107 | 0.144 | 0.395 | 0.000 | 0.049 | 1.502 | 0.061 | 0.048 | 0.073 | 0.155 | 0.337 | 84.58% |
| `gdr_bwd_reverse_scan` | 643.33 | 28/56 | 0.571 | 0.032 | 0.071 | 0.310 | 0.000 | 0.044 | 1.028 | 0.045 | 0.018 | 0.064 | 0.076 | 0.203 | 76.28% |
| `gdr_bwd_finalize` | 15.0 | —/56 | n/a | n/a | n/a | n/a | n/a | n/a | — | 0.587 | 0.391 | 0.480 | 0.662 | 2.120 | 0.000 |

Sources: `gdr_fwd_train_scan/tuning/eval_c8.json`, `gdr_bwd_reverse_scan/tuning/eval_c4.json`,
`gdr_bwd_finalize/round_wavefix/op_summary_after_mincritpath.csv`.

Reading Table D against Tables A and B is where the tuning story becomes legible per pipe. `train_scan`
went from Σ 1.584 to 1.502 — *less* total pipe occupancy — while getting 2.09× faster, because the win
was fewer launch waves rather than less work per core; the visible trace is `mte2` 0.566 → 0.395.
`reverse_scan`'s pipe profile barely moves at all (Σ 1.040 → 1.028, `mac` 0.571 both times) and it still
gets 1.33× faster, which is as clean a demonstration as this tree contains that its problem was never
pipe occupancy. `finalize` is the opposite case: between its Table A/B row (33.5 µs) and the shipped
kernel (15.0 µs) its `aiv_mte3` *rose* 0.340 → 0.662 and `aiv_vec` 0.261 → 0.587, because the same
stores ended up issuing from fewer, fuller blocks. One caveat on that pair: the `round_newhost` reading
of 33.5 µs sits above both `round_wavefix` readings (25.6 µs cold, ~18.4 µs warm), so it is an earlier
state of the kernel than §4's before/after pair, not the "before" of the wave fix. Only the Table D row
corresponds to what ships.

### Table E — the one non-pipe collection on this host: `finalize`

Pipe ratios are not the whole hardware picture, and for exactly one kernel in this family the other six
msprof groups were also collected on the current 28/56 host:
`gdr_bwd_finalize/round3_prod/` is a full seven-group pass. Its Task Duration of **26.98 µs** agrees
with the §1 pipeline's warm reading of 26.87 µs, so this is the in-pipeline `finalize`, not the 15.0 µs
standalone one — the two differ by method and shape, as the caution below says.

| group | metric | value |
|---|---|--:|
| `PipeUtilization` | `aiv_vec` / `aiv_scalar` / `aiv_mte2` / `aiv_mte3` | 0.361 / 0.261 / 0.650 / 0.566 |
| | `aiv_icache_miss_rate` | 0.042 |
| | header overhead (Task Duration − longest core) | 0.724 µs, 2.7% |
| `ResourceConflictRatio` | `aiv_vec_resc_cflt_ratio` | **0.924** |
| | `aiv_vec_bank_cflt_ratio` | 0.036 |
| `L2Cache` | `aiv_read_local_l2` hit / miss / victim | 102540 / 467 / 13530 → **99.55% hit** |
| | `aiv_write_local_l2` hit / miss / victim | 30220 / 511 / 1527 → **98.34% hit** |
| `Memory` | `aiv_main_mem_read_bw` / `aiv_main_mem_write_bw` | 0.492 / 0.547 GB/s |
| | `aiv_ub_read_bw` / `aiv_ub_write_bw` | 18.99 / 30.34 GB/s |
| `MemoryUB` | `aiv_ub_read_bw_vector` / `aiv_ub_write_bw_vector` | 1.969 / 4.381 GB/s |
| | `aiv_fixp2ub_write_bw`, both `*_scalar` | 0.0 |
| `MemoryL0` | every `aic_l0*` and `aiv_l0c_read_bw` | 0.0 |
| `ArithmeticUtilization` | `aic_mac_fp16_ratio`, `aic_mac_int8_ratio`, `aic_cube_fops` | 0.0 |

The `MemoryL0` and `ArithmeticUtilization` zeroes are structural — `finalize` launches no cube blocks,
so there is no L0A/L0B/L0C traffic and no cube FLOPs to count. The `Memory` and `MemoryUB` groups report
UB bandwidth from different counters in different passes and should not be reconciled with each other.

Two things here are worth carrying forward. **`aiv_vec_resc_cflt_ratio` 0.924 is the highest single ratio
in this whole document, and it is a *conflict* ratio rather than a utilization** — it is the mechanism
behind §4's finding that `finalize`'s earlier
restructure was *descriptor*-bound rather than bandwidth-bound: the vector units are in resource conflict
92.4% of the time while `vec` itself is only 36.1% busy. No pipe-ratio table would have shown that, which
is the argument for collecting the other six groups on the remaining seven kernels.

**And `finalize` is not touching main memory.** Main-memory traffic is 0.49 GB/s read and 0.55 GB/s write
against the 3.25 TB/s roof §5 measured, while the L2 read hit rate is 99.55%. So the ~2.55 TB/s that §5
attributes to `finalize` is being served out of L2, not DRAM — the same conclusion §6 reached for the
inference kernel on the previous host, now confirmed for a backward kernel on this one. "Bandwidth-bound"
for this kernel means L2-bandwidth-bound.

Two cautions on reading Table C. `reverse_scan`'s 643.3 µs is at **B=4** (32 BH items) because that is the shape it was characterised at; inside the pipeline at B=1 it has only 8 items and reads 217.6 µs — different shapes measuring different things, and the two should not be reconciled by arithmetic. Its `round_newhost` row in Tables A and B is at that same B=4 shape. And `finalize`'s figures come from a median sampling method, whereas the pipeline table in §1 uses a warm fourth occurrence; it reads 26.87 µs there. Both are honest; they are not the same measurement.

## 4. What the tuning actually changed

Three operators were tuned with the PANKO search tuner. All three turned out to be **the same bug**, and it is worth stating plainly because it will recur: each wrapper sized its grid to *maximise the number of cores used*, which on a 28-core host picks a work-item count that does not divide the core count and silently buys an extra launch wave. The objective should be *minimum critical path* — `waves × per-item cost`. All three already read the core count live; reading the hardware was necessary and nowhere near sufficient.

**`gdr_fwd_train_scan`, 380.0 → 178.9 µs (2.09×).** Its selector took the lexicographic maximum of `(cores, STREAMS, VT)` and chose 32 items on 28 cores — two waves, occupancy 56.6%, which is exactly `32/(2·28)`. Re-keying it on minimum critical path chose VT=48, 24 items, one wave. The measured search trace (`gdr_fwd_train_scan/tuning/eval_*.json`):

| candidate | change | µs |
|---|---|--:|
| preopt | — | 380.02 |
| c1–c4 | keeping loop-carried state resident in UB | 373.5–375.3 |
| c5 | fusing the two cube K-loops | 385.61 (worse, reverted) |
| **c6** | **selector re-keyed on critical path** | **181.61** |
| c7 | round-robin task→core map | 182.32 (reverted) |
| **c8** | **+ VT=48 with a core-fill tie-break** | **178.86** |

Note that c1–c5 — all of them real memory-traffic reductions — bought nothing, and the single scheduling change bought 2×. The on-chip traffic was already hidden behind MAC.

**`gdr_bwd_reverse_scan`, 856.7 → 643.3 µs (1.33×).** Same diagnosis, different fix: it had a second genuinely independent axis (the V-block loop) sitting inside the serial one. Folding it into a single flat item space took the critical path from `ceil(32/28) × 2 = 4` units to `ceil(64/28) = 3`, which is the floor for 64 equal items on 28 cores. Two of its eleven candidates failed correctness outright (`s=0`) and were reverted by the ratchet rather than measured — the trace in `gdr_bwd_reverse_scan/tuning/` records them.

**`gdr_bwd_finalize`, 18.4 → 15.0 µs (1.23×), grid occupancy 57.1% → 91.4%.** Its head-group formula produced 64 items on 56 vector cores, the same two-wave pattern. Archive: `gdr_bwd_finalize/round_wavefix/`. Those two percentages are the *theoretical* grid figures — `64/(2·56)` and the 256-item `256/(5·56)` — which is what the tuner optimised. The **measured** device occupancy, derived from `aiv_total_cycles` by the formula in §3, is **52.8% → 81.4%** (medians of the warm rows in each file). Same direction, same conclusion, but the theory overstates the achieved number by 4–10 points on both sides, because it counts neither intra-block skew nor the 2.7% launch overhead Table E measures. If you are judging a grid change, get the measured figure before claiming a percentage. This came on top of a much larger earlier restructure — 126.4 → ~27 µs — which replaced 128 single-lane `store_align` calls per chunk with a six-step register-level suffix scan. That one was descriptor-bound, not bandwidth-bound: it was moving almost no data and still saturating the store pipe.

An audit of the other five operators found them already at 91.4% grid occupancy (256 items over 28 or 56 cores) and they were left untouched. Their measured device occupancies, in Table C, run 85–90% — consistent with that audit and with the same few points of theory-versus-measurement gap.

## 5. What is left, and what has been ruled out

**`gdr_bwd_dqkw` is at its fp32 ceiling.** — ⚠ **Partly refuted by §8.2; read that first.** The
"saturated" reading below assumes `mte1`, `scalar` and `fixpipe` cannot overlap MAC. They can: a
barrier that sat in front of three independent matmuls, a redundant transpose matmul, and five L0C
accumulators sharing two `mutex_id`s were together worth −20.2%, taking ΣAIC 1.097 → 1.291 and
`mac` 0.634 → 0.721. **A pipe-ratio sum near 1.0 means the pipes are serialized, not that they are
saturated** — the two are opposite diagnoses with opposite fixes. What survives from this paragraph
is the MTE2 floor (now confirmed against the `Memory` group, §8.2) and the conclusion that bf16
operands are the only remaining route to `mac ≥ 0.80`. The paragraph as originally written:

Its six cube pipes sum to 1.068 in `round_newhost` (§3, Table A) — saturated — but only 63.4% of that is MAC. The rest is operand traffic that the frozen matmul set requires, spread across four pipes rather than concentrated in one (`mte2` 0.218, `mte1` 0.091, `scalar` 0.072, `fixpipe` 0.053): a DMA probe showed MTE2 moves exactly its contract bytes (predicted 395 µs against 389 µs measured), and transposed loads cost the same as plain ones, so the duplicate `do`/`v_new` orientations are irreducible. With MTE1, scalar and fixpipe driven to zero, MAC caps at **79.7%**. Getting past that needs bf16 cube operands, which is a SPEC change, not a tuning change. An L1 double-buffering attempt regressed and was reverted.

**`gdr_bwd_wy_reduce` and `gdr_bwd_finalize` are bandwidth-bound.** The on-box ceiling was measured rather than taken from a datasheet: a pure-DMA copy of the same bytes on the same grid runs at **3.25 TB/s**. `finalize` after its fixes moves ~38 MB in 15.0 µs ≈ 2.55 TB/s, about **78% of that ceiling**. `wy_reduce` sits lower, at 41%, and its rounds are exhausted: six read tiles were split and double-buffered for a genuine 25% gain, and a further single-group pipelining attempt made it worse (69.6 µs) and was reverted.

**For the serial scans, three plausible explanations for cube idle time are measured dead.** Do not re-spend these: hoisting the cube and vector operand loads above a wait, deepening the handoff buffers (depth-2 and depth-4), and removing a redundant cross-core handshake entirely. The last of these deleted a full round trip from all 32 serial steps, ran correctly, and changed the total by **0.07%**. The remaining lever is genuine independent work in flight — a second reverse chain — which needs UB headroom and is a structural change, not a knob.

## 6. The inference forward kernel (previous host)

Kept for reference. **These numbers were taken on the 32 cube / 64 vector host and must not be compared with anything above.** Archives: `baseline/` and `optimized/`, each a full seven-group collection plus a sample-based pass.

The optimization took the total across its eight cases from **1134.8 µs to 646.0 µs (−43.1%)**, with the largest case (`mid`, B=1 T=2048 H=8 K=V=128) going 638.0 → 346.6 µs. All of it was scheduling; the maths, tolerances and goldens were untouched and the per-case maximum absolute error is bit-identical before and after. The changes: the scan work item became the flat pair `(bh, v_tile)` instead of `bh` alone; the launch was sized for both phases rather than chunk-prep alone; both AIV sub-blocks were put to work, where previously every payload sat under `if get_subblock_idx() == 0` and half the vector cores only took part in handshakes; and the k^T pre-pass plus one of two grid barriers were deleted.

That first change is the same insight the backward scans needed, and it was already written down as knowledge card `sched-01` before this campaign began. We rediscovered it anyway, because the working repository was not connected to the card library — which is the most expensive process lesson here and is why `sched-02` now exists beside it.

Three findings from that kernel still stand. It is **not memory bound**: main-memory traffic is sub-1 GB/s against a TB/s roof, and L2 read hits are essentially total, so `aic_mte2_ratio` of 0.10–0.17 is the correct number rather than a defect — raising it would mean inventing traffic the algorithm does not need. It is **dependency-bound**: on `mid` the largest of nine pipe ratios is `mac` at 0.449, while the scan exposes only 16 work items to 32 cores, each a 32-step recurrence. And two routes were **measured dead**: hoisting `o` into a wider third phase (two variants, both correct, both slower), and skew-aware chunk-prep distribution (a grid-wide barrier makes even distribution already optimal).

## 7. Raw data

| path | what |
|---|---|
| `composition/op_summary_composition_prod.csv` | the eight-kernel pipeline rows behind §1 |
| `sweep/SWEEP.md`, `sweep/<shape>/` |  the dimension sweep in §2: per-shape `op_summary.csv`, plus the two `FAILED_msprof.log` records for the unsupported B=2 and K=V=256 shapes |
| `<op>/round_newhost/op_summary_PipeUtilization.csv` | per-operator single-group profiles on the current host — the source for the complete pipe tables in §3. One file per operator, 48 columns: Task Duration, Block Num / Mix Block Num, `aicore_time` + `aic_total_cycles`, the six `aic_*_ratio` pipes (`mac`, `scalar`, `mte1`, `mte2`, `mte3`, `fixpipe`) with their `_time(us)` partners, `aic_icache_miss_rate`, `aiv_time` + `aiv_total_cycles`, the four `aiv_*_ratio` pipes (`vec`, `scalar`, `mte2`, `mte3`), `aiv_icache_miss_rate`, and `cube_utilization(%)` |
| `<op>/tuning/eval_*.json`, `search_state.json` | PANKO search traces — measured `p`, correctness `s`, and a full `m.pipes` dict (all ten ratios) plus `cube_utilization` and `block_num` per candidate; this is the only per-pipe record of the tuned `train_scan` and `reverse_scan`, and §3's Table D comes from it |
| `gdr_bwd_finalize/round_wavefix/` | before/after for the occupancy fix |
| `gdr_bwd_finalize/round_001…006` | the earlier restructure rounds |
| `gdr_bwd_finalize/round3_prod/` | the only full seven-group collection on the current host — all 21 `op_summary_<Metric>.csv` / `op_statistic_` / `task_time_` exports plus `summary.txt`; §3's Table E |
| `baseline/`, `optimized/` | the inference kernel's full collections (previous host). Note these two have **quoted `Input Shapes` fields containing commas**, so a naive comma-split misaligns every column after 14 — parse them as real CSV. Their `per_core_cycles.csv` / `summary.txt` per-core section is process-scope and not usable; see §3 |
| `<op>/.../per_core_cycles.csv` | `AICoreOriginalData.task_cyc` from a `--aic-mode=sample-based` pass, `SUM(...) GROUP BY coreid` — process-scope, so single-kernel runs only |

Each `op_summary_<Metric>.csv` holds the per-launch rows; Task Duration and every ratio quoted here comes from those. `op_statistic_`, `api_statistic_` and `task_time_` files are supporting exports.

**On the current host, seven of the eight kernels have only the `PipeUtilization` group.** The other six msprof metric groups are `Memory` (L1 and main-memory read/write GB/s), `MemoryUB` (UB read/write GB/s, split vector/scalar, plus `fixp2ub`), `MemoryL0` (L0A/L0B/L0C GB/s), `L2Cache` (local L2 read/write hit, miss and victim counts), `ArithmeticUtilization` (`aic_mac_fp16_ratio`, `aic_mac_int8_ratio`, `aic_cube_fops`) and `ResourceConflictRatio` (`aiv_vec_bank_cflt_ratio`, `aiv_vec_resc_cflt_ratio`).

The one exception is `gdr_bwd_finalize/round3_prod/`, a full seven-group pass **on the current 28/56 host** — §3's Table E is that collection, and the 92.4% vector resource-conflict ratio it turned up is the strongest argument in this tree for collecting the same seven groups on the remaining kernels. That is open work.

Everywhere else those six groups exist only in the older seven-group collections under `baseline/`, `optimized/` and the `<op>/round_001/` trees, whose largest launches read 32 cube / 64 vector blocks — the previous host. They are not comparable with anything in §1–§5 and are deliberately not folded into Tables A–D. Where §5 needed a bandwidth number on this host before Table E existed it was measured with a purpose-built DMA probe instead, which is why those numbers are quoted as probes rather than as msprof columns.

The PANKO tuner itself is not in this tree — it is an operator-agnostic skill with its own validation fixtures, so it carries no dependency on this operator family.

## 8. The pipe-concurrency round, 2026-10-14

Scope: raise the fraction of the machine each kernel actually uses, and cut latency, without
touching SPEC, goldens, module boundaries, dtypes or tolerances. Result: **1048.3 → 978.0 µs
(−6.7%)**, with all 5 composition cases passing and both changed kernels producing
bit-identical output. Three kernels were attacked; one of the three correctly produced no change.

Host: 28 cube / 56 vector Ascend950PR, same class as §1–§5. Archives in
`round_concurrency_2026_10_14/`: a baseline and three optimized `PipeUtilization` passes, plus —
new on this host — the other six metric groups (`Memory`, `MemoryUB`, `MemoryL0`, `L2Cache`,
`ResourceConflictRatio`, `ArithmeticUtilization`) for the whole eight-kernel pipeline, which §7
listed as open work.

### 8.1 How to read utilization here, and the three numbers that are not comparable

This supersedes nothing in §3 — it adds the decision procedure §3's identities support.

**The honest efficiency figure is `cube_utilization × aic_mac_ratio`.** The two factors have
different denominators (`Task Duration` and `aicore_time`) but those differ only by launch
overhead, under 1% across this family, so the product is the share of the whole device's MAC-cycle
capacity that the kernel spends issuing multiplies. Baseline, all eight:

| kernel | µs | `cube_util` | `aic_mac` | **real MAC** | `aic_main_mem_read_bw` |
|---|--:|--:|--:|--:|--:|
| `gdr_fwd_train_wy` | 200.9 | 90.0% | 0.344 | **31.0%** | 0.46 GB/s |
| `gdr_fwd_train_scan` | 191.5 | 84.5% | 0.756 | 63.9% | 1.22 GB/s |
| `gdr_bwd_dv_local` | 47.9 | 87.1% | 0.558 | 48.6% | 0.42 GB/s |
| `gdr_bwd_reverse_scan` | 217.4 | 56.8% | 0.568 | **32.2%** | 0.71 GB/s |
| `gdr_bwd_dqkw` | 201.6 | 89.5% | 0.634 | 56.7% | 0.58 GB/s |
| `gdr_bwd_wy_matmuls` | 122.9 | 86.7% | 0.756 | 65.5% | 0.65 GB/s |

Main-memory traffic across the family is 0.3–0.9 GB/s against the ~3250 GB/s DMA roof §5 measured,
and `L2Cache` read hit rates are essentially total, so **no kernel here is DRAM-bound and raising
`mte2` would mean inventing traffic**. "Utilization below 80%" on these kernels therefore never
meant "feed it more data"; it meant "the pipes are taking turns".

**`ΣAIC` — the arithmetic sum of the six cube pipe ratios — is the cheap in-core overlap meter.**
It ranks this family in exactly the order of their MAC quality:

| `ΣAIC` | kernel | `aic_mac` | reading |
|--:|---|--:|---|
| 0.724 | `train_wy` | 0.344 | pipes barely overlap, and the cube core has idle left |
| 0.987 | `reverse_scan` | 0.568 | **zero overlap, zero idle** — window filled by mac+mte2+mte1+fixpipe in turn |
| 1.098 | `dqkw` (before) | 0.634 | slight overlap |
| **1.291** | **`dqkw` (after)** | **0.721** | ≥2 pipes concurrent over 29% of the window |
| 1.546 | `train_scan` | 0.756 | genuine overlap |
| 1.670 | `wy_matmuls` | 0.756 | most saturated in the family; `fixpipe` 0.305 is its second bottleneck |

So "`aic_mac` 0.63, therefore 37% idle" is the wrong reading when ΣAIC is 1.10: nothing is idle,
the MAC is waiting its turn. The actionable question is **which dependency orders the pipes**, not
which pipe is slowest.

**Two traps this round walked into, both worth inheriting.**

*`ΣAIC` is not an upper bound on a core's busy fraction* — concurrent pipes are double-counted — so
`ΣAIC + ΣAIV` compared against 1.0 proves nothing about whether the cube and vector sides overlap
each other. `train_wy`'s 0.724 + 0.882 was read as "the two sides strictly alternate", and a
cross-chunk software pipeline was built on that reading. It compiled, stayed numerically correct,
and **regressed 33.6%** (196.0 → 261.8 µs) with *both* sides getting idler (ΣAIC 0.732→0.538,
ΣAIV 0.892→0.694). The source said otherwise all along: the next chunk's M1 already sat *before*
`set_cross_core(e1)`, inside the window where the cube runs the previous chunk's M4, so the two
sides were already overlapped and moving M1 later merely exposed M4. **Cross-core overlap is read
off the event placement in the source, never inferred from the ratio columns.**

*`aic_mac_ratio` is pipe-busy, not percent-of-peak, and the two separate for free.* Divide
`aic_cube_fops / (µs × blk)` by `aic_mac_ratio` from any `ArithmeticUtilization` pass: `dv_local`
12.1e6, `reverse_scan` 13.8e6, `dqkw` 12.2e6, `wy_matmuls` 11.9e6 fops per MAC-busy core-µs — a
narrow band. MAC-busy cycles are uniformly productive here, so **"raise `mac_ratio`" can only ever
mean "remove non-MAC core cycles"**, never "make the MAC more efficient". On `dqkw` the absolute
MAC time held at 115.5 ± 0.4 µs across nine variants spanning 201 → 159 µs, which is the same fact
measured a second way.

### 8.2 `gdr_bwd_dqkw`, 201.2 → 160.5 µs (−20.2%), `aic_mac` 0.635 → 0.721

§5 called this kernel "at its fp32 ceiling" of 79.7% MAC, on the assumption that `mte1`, `scalar`
and `fixpipe` cannot overlap MAC. **That assumption was wrong**: ΣAIC went 1.097 → 1.291, so the
overlap was available and was being blocked. Three changes, each maths-identical:

- **Barrier placement, −19.0 µs.** `wait_cross_core(ev5)` sat *before* `dq_inter`/`dk_inter`/`dw`,
  three matmuls that depend only on GM inputs. The cube idled at the barrier through the vector's
  64-iteration `mask_lower`. Reordered to `dqi → dki → dw → [ev5] → dqa → dka`.
- **A redundant transpose matmul deleted, −16.0 µs.** Masked `ds_T` *is* `transpose(masked ds)`
  exactly, because the mask is self-symmetric under transpose: `[j≤i]`+`exp(d_i−d_j)` maps onto
  `[j≥i]`+`exp(d_j−d_i)`. So the second cube matmul, its `do^T` L1 operand, its L0C drain, the
  `mask_upper_vf` and the whole `dsT_gm` bridge are gone; `dk_intra` reads `ds_gm` back with
  `order=[4,3]`. 12 → 11 matmuls, and the transposed read costs no extra bytes (below).
- **Un-shared `mutex_id`s, −5.6 µs.** All five L0C accumulators shared `[20,21]` — 2 ids across 10
  tiles — serializing MMAD against the FixPipe drains. Each got its own pair; 32/32 ids now used.

`mac ≥ 0.80` was **not** reached and the evidence says it is out of reach without a SPEC change.
MTE2 moves exactly its contract bytes — `aic_l1_write_bw` 18.75 GB/s/core against 17.6 computed
from the 352 KB/tid contract, *including* the transposed reads, with no loop-invariant reload — and
its throughput while busy is 70 GB/s/core ≈ 1.97 TB/s aggregate, i.e. platform peak. So its ~50 µs
is a floor. `mac ≥ 0.80` needs the total at ≤ 115.5/0.80 = 144 µs, which means hiding ≥41 of those
50 µs. `MemoryL0` rules out feed/drain as the stall (L0A read 484, L0B 7.6, L0C write 2.3 GB/s).
**The only evidenced route left is cutting MTE2 volume — bf16 cube operands would halve the
352 KB/tid — and that is a SPEC change, so it is a recommendation, not a change.**

### 8.3 `gdr_fwd_train_wy`, 196.0 → 170.1 µs (−13.2%), `aic_mac` 0.349 → 0.386

The win was **not** where the low pipe ratios pointed. It came from the per-step overhead of the
`fwdsubst` forward substitution, which the file header and §3 both framed as a dependency chain —
and which the kernel's own ablation had already measured at 23% of the kernel, not the ~85% that
op-counting predicts. So it was never latency-bound, and shortening the recurrence was the wrong
target. Instead the rows are now emitted **in pairs**: `row_m` loaded once and consumed by both
rows, two independent `mul_add_dst` accumulators, row *fp*'s contribution to *fp+1* taken from a
register rather than re-read from UB, and every UB read of the pair issued before either store so
one `mem_bar` serves the pair (**63 → 32 barriers**). Each row still accumulates `m = 0, 1, …` in
its original order, which is why the output is bit-identical rather than a tolerance trade — the
production `attn` `max_abs_error` is 3.616333e-03 before and after. That is −11.0%; hoisting a
loop-invariant `attn` reload out of M4's three loops (6×16 KB/tid where 2 suffice) gave the other
−2.4%, taking `c_mte2` 0.147 → 0.114.

The bounding pipe did not reach 0.80: best is `aiv_vec` 0.450 with **ΣAIV 0.956**, so the vector
core has 4.4% idle left and no reordering can lift one pipe to 0.80. That would need vector DMA and
vector compute concurrent, which the in-place VF idiom forbids (`rowscale_vf`, `attn_build_vf`,
`a_build_vf` all have `src == dst`), and UB has **3328 B free** — the layout ends exactly at
0x30000 — so no out-of-place staging tile fits without a full re-layout.

One measurement worth keeping: deleting a provable 32 KB/tid of stores moved the duration **0.01%**
and `aiv_mte3` only 0.234 → 0.230. The two AIVs of a physical block **share MTE**, so per-AIV
ratios undercount that shared resource, and near saturation **MTE time stops being
byte-proportional — cut passes, not bytes.**

### 8.4 `gdr_bwd_reverse_scan`: unchanged, and that is the correct outcome

§3 attributes this kernel's 56.8% occupancy to load balance, characterising it at B=4. **At the
production shape that is wrong: `Block Num` is 16, so 12 of the 28 cube cores are never given
work at all.** 56.8% is exactly 16/28 — a pure launch-*width* loss — and with `aicore_time` 217.1
against `Task Duration` 217.4 (coverage 0.999), the 16 cores that do run are ~100% busy. Same
percentage, different factor, different fix; §3's three-way factorisation is what separates them.

**Launch width here is lattice-quantised and 24/28 does not exist.** items = `B·H·ceil(V/TV)` =
`1·8·2` = 16; the independent axes are B(=1), H(=8), V-tiles(=2) and K, while N is the serial axis.
Every admissible refinement doubles an existing power-of-two axis, so reachable counts are
`{8·2^j}` = {8, 16, 32, …} — **nothing lands in (16, 28]**. Costing the candidates with
`waves = ceil(16s/28)`:

| construction | items | waves | per-item | critical path | verdict |
|---|--:|--:|--:|--:|---|
| shipped, TV=64 | 16 | 1 | 1.00 | **1.00** | optimal |
| V-split, TV=32 | 32 | 2 | ≈0.60 | 1.20 | 20% worse |
| K-split | 32 | 2 | ≈0.567 | 1.13 | 13% worse, and impossible |
| TV=128 + K-split | 16 | 1 | 1.00 | 1.00 | provably neutral |

V-split's per-item cost is 0.60 rather than 0.50 because the three V-independent operands — 96 of
160 KB/step — **replicate** instead of dividing, so MAC halves but MTE2 only falls to 0.7 and
device MTE2 bytes rise ~1.4×. **"Halve the tile, halve the per-item cost" is the default wrong
assumption; classify each operand as dividing or replicating first.** K-split is additionally
*structurally* impossible: mm1 contracts over K, so `dvc = k@dS` needs a per-step cross-shard
reduction whose result feeds every shard's dS update — a cross-block rendezvous *inside* the
32-step serial chain, requiring all participants resident, i.e. items ≤ 28, i.e. one shard. A
32-block barrier on 28 cores deadlocks.

`aic_mac ≥ 0.80` is unreachable for the same reason stated two ways: MAC-busy time is 123.5 µs and
fixed by the frozen maths, so 0.80 requires non-MAC core cycles to fall from 91 µs to ≤31 µs —
essentially all of MTE2 hidden — while ΣAIC 0.987 says there is **zero** overlap to redistribute.
Five independent scheduling interventions (three from §5, two here) each moved the total under 1%.

A correctly-constructed MTE2 prefetch was built here for the first time and bought **0.55%**
(217.4 → 216.2 µs), below the noise floor: untouched sibling kernels moved 1.9–3.3% between the
same two runs (`dv_local` 47.9→47.0, `wy_reduce` 39.9→41.2). Not ratcheted — it costs +96 KB L1,
6 more mutex ids and +3.75% MTE2 bytes. **Establish the noise floor from untouched kernels in the
same run before ratcheting anything.** Note §5's "hoisting loads above a wait" was, per the code,
only a hoist: `g_l1_k/qt/wt` are at depth 1, so `auto_mutex` re-serialises the hoisted load behind
the previous iteration's consuming `pl.move`. A hoist and a depth-2 buffer are **one** change, and
testing the halves separately reads as "prefetch doesn't help" when it means "prefetch was never
built".

### 8.5 Measured dead this round — do not re-spend

- `train_wy` cross-chunk software pipelining (M1 to AIV sub-block 1, one chunk ahead): **+33.6%**.
- `dqkw` L1 double-buffering, depth-2 + per-group mutex pairs + L1 half-split: **+0.5%**. The
  `wait_cross_core` calls sit *between* the loads and the matmuls, so buffer depth has no program
  order to exploit. This reproduces and explains §5's earlier reverted attempt.
- `dqkw` prefetching h/dh into a prologue above `ev1`: **+2.1%** / **+2.4%**. `ev1` was never a
  bubble, and pushing a load into an already-saturated queue only delays the first-needed operand.
- `dqkw` merging vector-side mutex ids to free cube-side ones: **+1.8%**. `ub_kf` and `ub_do_f`
  feed the same `combine_vf`; merging serializes the vector onto the `ev7` path. ΣAIV 0.28 of
  "vector slack" is not freely spendable.
- `dqkw` `pl.range` kv loop instead of two unrolled blocks: **+1.1%**. Applying the block-0
  reorder to block 1 as well: **+2.2%** (no barrier there, so it only perturbs L0A/L0B slot parity).
- `reverse_scan` UB→L1 direct staging to delete the UB→GM→L1 bridges: **numerically failed**,
  `matched_ratio` 0.500 (all three bridges) and 0.751 (one). `pl.move(L1, UB)` is a byte-physical
  copy with no layout conversion; an ND UB tile needs a UB→UB ND→NZ `pl.move` first. The failure
  mode is *deterministically partially-correct output*, not a crash — the first reverse chunk
  starts from dS=0 and is right by construction — so a weak test passes it.

### 8.6 Two framework constraints found the hard way

- **A tile handle from `.next()` cannot cross traced `if` branches.** Created in one branch and
  used in another, codegen fails: the emitted `<tile>__mutexid_0` / `_bufidx_*` locals are
  branch-scoped. Create at an enclosing scope, or index the group (`g[i]`, which accepts a runtime
  index). Outer→inner is fine; sibling→sibling is not.
- **`pl.store` cannot transpose** — its `order` is ascending-only, while `pl.load` supports
  descending. A GM-side transpose is therefore always paid on the *reader's* MTE2 and can never be
  moved to the writer's MTE3.

### 8.7 What this round says to do next

Ranked by evidence, not by size of the number:

1. **`train_wy`'s `transpose_vf` bank conflicts.** It gathers with index `iota << 6` — a
   64-element/256-byte stride on an fp32 tile, which is `32 B × 8`, an even multiple and the
   documented worst case (~5–7×). It runs 4×64 gathers per chunk. The fix needs the source row
   pitch padded to an odd multiple of 32 B (72 fp32 = 288 B), i.e. +8 KB across the four rotating
   slots against 3.3 KB free — **a full UB re-layout**, which is why it was not half-done. Pushing
   the transposes onto the cube instead is now also closed: at ΣAIC 0.800 the cube cannot absorb
   ~64 KB more MTE2 plus 64 KB more fixpipe per chunk.
2. **`wy_matmuls` is the one kernel at risk of becoming fixpipe-bound**, `fixpipe` 0.305 against
   `mac` 0.756 and ΣAIC 1.670. cube-01's `phase=` handshake is the lever; it is also the most
   saturated cube core in the family, so expectations should be modest.
3. **bf16 cube operands for `dqkw`** would halve its 352 KB/tid and is the only evidenced route to
   `mac ≥ 0.80` there. **SPEC change — needs a decision, not a tuning pass.**
4. **`reverse_scan`'s remaining UB→L1 bridge** (64 of 160 KB/step, and precisely the fence-gated
   traffic) needs a second UB tile (+32 KB against a 140 KB/55.1% budget), a UB→UB ND→NZ `pl.move`,
   L1 tiles with matching layout, and confirmation the MTE3-tagged `set_cross_core` still fences
   the move. Ceiling if it lands: `mte2` 0.265→~0.16, ~195 µs, `mac` ~0.63 — real, but still short
   of 0.80, and the failure mode above makes it a design step with its own verification rather
   than a tuning tweak.

The reusable findings are written up as knowledge cards `metrics-01`, `sched-03`, `cube-08`,
`cube-09` and `vec-16` in `cannbot-skills/ops/pypto-pro-op-perf-tune/references/knowledge-cards/`,
with the operator-agnostic reproduction script at
`cannbot-skills/ops/pypto-pro-op-perf-tune/scripts/pipe_concurrency_table.py`.

## 9. The second concurrency round, 2026-10-14

Same scope rules as §8 — SPEC, goldens, module boundaries, dtypes and tolerances frozen.
**978.0 → 912.4 µs**, all 5 composition cases passing, every changed kernel bit-identical to
before. Combined with §8 the pipeline is **1048.3 → 912.4 µs (−13.0%)**.

Totals are medians of three independent single-group `PipeUtilization` passes
(915.7 / 912.4 / 911.3 µs). Archives in `round_concurrency_2026_10_14/`.

| kernel | final µs | after §8 | original | `aic_mac` | ΣAIC | real MAC | total Δ |
|---|--:|--:|--:|--:|--:|--:|--:|
| `gdr_fwd_train_wy` | **132.7** | 171.1 | 200.9 | 0.502 | 1.059 | 44.8% | **−33.9%** |
| `gdr_fwd_train_scan` | **182.6** | 191.3 | 191.5 | **0.795** | 1.483 | 66.9% | −4.6% |
| `gdr_bwd_dv_local` | **37.5** | 47.7 | 47.9 | 0.703 | 1.282 | 62.0% | **−21.7%** |
| `gdr_bwd_reverse_scan` | 217.8 | 217.3 | 217.4 | 0.568 | 0.975 | 32.2% | — (floor, §8.4) |
| `gdr_bwd_dqkw` | 161.0 | 161.1 | 201.6 | 0.720 | 1.287 | 63.7% | **−20.1%** |
| `gdr_bwd_wy_matmuls` | **114.5** | 122.0 | 122.9 | **0.812** | 1.785 | 70.1% | −6.8% |
| `gdr_bwd_wy_reduce` | 40.4 | 40.6 | 39.9 | — | — | — | — (not measurable, §9.5) |
| `gdr_bwd_finalize` | 25.8 | 27.7 | 26.1 | — | — | — | — (not measurable, §9.5) |
| **total** | **912.4** | 978.0 | 1048.3 | | | | **−13.0%** |

**Against the 80% bar.** `wy_matmuls` **meets it**: `aic_mac` 0.812 in-pipeline (0.812 / 0.792 /
0.815 across the three passes) and 0.810 standalone. `train_scan` meets it standalone at 0.814 and
reads **0.795** in-pipeline (0.790 / 0.796 / 0.795) — inside the stated tolerance that 79.x% is
acceptable. The other four cube kernels run 0.50–0.72, and §8.4 plus §9.4 give the evidence for
why the two scans cannot get there without an algorithm or SPEC change.

### 9.1 `gdr_fwd_train_wy`, 171.1 → 132.1 µs standalone / 132.7 in-pipeline (−22.8%)

A UB bank conflict, and the fix was to **reverse the direction of the strided access** rather than
pay for what the conflict seemed to demand. `transpose_vf` gathered with index `iota << 6` — a
64-element / 256-byte stride on an fp32 tile, i.e. `32 B × 8`, an **even** multiple of the bank
width and the documented worst case. Padding the gather *sources* to an odd multiple needs all four
rotating `[C,SUB]` slots, **+8192 B against 3328 B free** — a full UB re-layout.

Instead: `vf.gather` down a source column became `vf.load_align` of a source **row** (contiguous,
zero cross-lane) plus `vf.scatter` down a destination **column** at pitch `TP = 72`
(288 B = `32 B × 9`, odd). The indices are a permutation of the old ones, so the output is
bit-identical; and the padding obligation lands on the **single-slot** `ub_p2`, **+2048 B**, which
fit the hole that already existed. **No re-layout was needed.** `aiv_vec_bank_cflt_ratio`
**0.338 → 0.062**, `aic_mac` 0.386 → 0.506, ΣAIC 0.799 → 1.068, real MAC 34.7% → 45.2%. The
bounding *side* flipped: the vector core emptied (ΣAIV 0.896 → 0.737) and the cube window filled.
Final UB budget 195328 / 196608 B, 1280 B free, with no live-range overlapping required.

**This refuted a published claim, which has been corrected in place.**
`references/general-knowledge/a5-roofline-and-levers.md` said a conflict-free gather is still not a
cheap gather and that removing conflicts "won't bring it close to `load_align`". Three states of
one kernel: conflicted **170.8 µs**, conflict-free **131.4**, no-cross-lane ablation **127.5**. So
the conflict was worth **39.4 µs** and the residual cross-lane cost only **3.9 µs** — about
1.7 ns per gather over ~2341 gathers per core, roughly 2× `load_align`, not the ~16 ns that doc's
table implies (that figure describes a UB round-trip with `mem_bar` and should not be extrapolated
to gathers). The design question is therefore not "should we cross lanes?" but **"is the strided
pitch an odd multiple of 32 B?"**

Two sizing lessons came with it. A **zero-cost probe** prices such a fix before any UB moves: keep
the gather and change only the index stride to a conflict-free value that still fits the tile's
existing bounds (40 elements = 160 B here); numerics go wrong, addressing stays representative, UB
cost is nil. Pair it with a no-cross-lane ablation and two runs separate "conflict" from "intrinsic
cross-lane". And **the conflict ratio scales with per-core work**: the same code reads 0.18–0.22 at
6–12 blocks but **0.338** at 28, so sizing this on an acceptance shape under-reports it ~1.7×.

### 9.2 `gdr_fwd_train_scan`, 188.2 → 179.2 µs standalone (−4.8%), `aic_mac` 0.769 → 0.814

**MTE2 busy-cycle throughput is set by descriptor granularity, not by bytes and not by the
platform.** This kernel's effective GM→L1 throughput while busy was 56.0 GB/s/core against ~101
that sibling kernels reach on the same host (the derivation is `aic_l1_write_bw / aic_mte2_ratio`;
see §9.6). `w`, `q_de` and `kᵀ` were each loaded once **per k-tile** — 256 B contiguous runs on a
512 B stride. Loading each once **per chunk** in full-K form into a `[C,KMAX]` / `[KMAX,C]` L1 tile
and slicing per k-tile with `pl.move(..., offset=)` took throughput to **74.4 GB/s/core (+33%) on
byte-identical traffic**, MTE2 busy time 106.3 → 75.1 µs. Cost: L1 176 → 304 KB of 512, no new
mutex ids (a freed ping/pong pair paid for it). Byte cross-check: contract 6.03 MB/core against
30.94 GB/s × 178 µs ≈ 5.5 MB, so measured ≈ contract and there was no hidden reload.

The per-k-tile load *looks* mandatory because the K loop needs a per-k-tile operand in L0; it is
not, because `pl.move(offset=)` supplies that slice. **The GM→L1 descriptor count is a free
variable — count descriptors per operand per outer-loop iteration, not bytes.** This also fixes how
the published strided-load curve should be read: 256 B/row looks near-peak there (128 B → 88%) yet
measured **55% of what the same host does contiguously**. The curve describes one transfer's
efficiency, not the cumulative cost of many short ones.

**Attribution correction against our own result**: the granularity merge and a barrier hoist shipped
together, and a later control experiment showed the recurrence barrier costs ~0, so the entire
−9 µs is granularity. §8's barrier audit gets no credit on this kernel.

**A latent pre-existing defect, recorded not fixed.** `g_st_ub_1`'s mutex ids `[5,30,31,27]` reuse
id 27 from `g_st_ub_0`'s `[0,27,28,29]`. It collides only at K=256 **and** STREAMS=2 — a pair
`_pick_vt_streams` never selects (sr=2 always loses on the critical-path key) and no test exercises.
It cannot be fixed without a free id and this kernel is now at 32/32. It is a trap for whoever next
changes the selector.

### 9.3 `gdr_bwd_dv_local` −22.1% and `gdr_bwd_wy_matmuls` −8.4%: the §8 playbook, plus a correction to it

**`dv_local` 47.9 → 37.3 µs, `aic_mac` 0.553 → 0.704, ΣAIC 0.953 → 1.270.** Three pieces, all
§8-style: the vector section's `wait_cross_core(ev0)` sat at the top of the tid body in front of the
`g_f32` load, the 64-step serial `cumsum_decay_vf`, the `grad_o` load and `cast_go_vf` — none of
which touch `ub_al`, the only thing ev0 guards — so cube and vector were strictly serialized
(−5.8 µs to fix); depth-2 ping/pong with a one-tid-ahead refill (−1.3 µs); and then **a second
barrier pass** (−3.4 µs).

**Run the barrier audit more than once.** The first fix changes what is left behind the barrier.
Re-reading ev0 afterwards showed the `grad_o`→NZ move and its 16 inserts into `l1_goB` (32 KB, needing
only ev2) still sat after ev0 — so two thirds of the L1 production the cube blocks on was waiting for
nothing. The audit question is not "is a barrier in front of independent work" once, but **"given what
is now behind it, what does it actually gate"**, and the answer changes after each fix.

**`wy_matmuls` 122.2 → 111.9 µs, `aic_mac` 0.757 → 0.810, ΣAIC 1.665 → 1.791 — entirely from one
`mutex_id` split.** `l1_dw`, `l1_dv2` and `l1_vb` are three address-disjoint L1 operands whose
GM→L1 loads all sit at the top of the tid body, and the transpiler had economised all three onto
**id 9**, serializing 96 KB of independent MTE2.

**This corrects §8's own framing, which this report had inherited.** §8 filed the mutex audit under
the "ΣAIC ≈ 1.0" trigger, and the brief for this round accordingly told the agent that `wy_matmuls`
at ΣAIC 1.665 was past re-ordering. That inference is wrong, and the distinction matters: ΣAIC
measures overlap *between pipes*, while a shared mutex id creates a **false dependency between
independent DMA streams on the same pipe** — which does not reduce inter-pipe overlap and is
therefore **invisible in ΣAIC**. The mutex audit needs a second, ΣAIC-independent trigger: *three or
more address-disjoint operands sharing one mutex id, loaded back to back*, whatever ΣAIC reads.

**And 32/32 ids is not a hard stop, because the operation is asymmetric.** Merging ids only ever
*adds* ordering, so it cannot break correctness; splitting address-disjoint tiles only *removes*
ordering that was never needed. So: merge pairs whose accesses program order already separates
(different phases of one tid body), then spend the freed ids where pipes genuinely cross. That is
how this kernel got three ids for three operands while staying at 32/32 — merge `dAb`+`dAc`→15 and
`kc`+`kbe`→28, then split id 9 into 9 / 26 / 29.

### 9.4 `gdr_bwd_reverse_scan`: the one remaining lever, built, and it loses 1.9 : 1

§8.7 named the UB→GM→L1 bridge removal as the last identified lever (64 of 160 KB per step, and
precisely the fence-gated traffic). It was built completely and correctly — and it **regresses**:
218.1 → 234.4 µs (dS only, +7.5%) and 244.2 µs (all three bridges, +12.0%). Reverted.

All four prerequisites checked out for this version, with no capability gap: UB→L1 is documented
including the "operand produced by Vector in UB" pattern; L1 fp32 **must** be NZ (ND only for
INT64/UINT64), so the UB→UB ND→NZ conversion is mandatory and there is no "keep it ND" shortcut;
and `PipeType` documents `move UB → L1` as **MTE3**, the same pipe as the `pl.store` it replaces, so
the existing MTE3-tagged `set_cross_core` still fences it.

**Why it loses, in absolute pipe-seconds.** `aic_mte2` fell exactly as projected (0.225 → 0.153),
MTE2 busy time 49.1 → 37.3 µs (**−11.8**) — but V busy time went 10.0 → 32.0 µs (**+22.0**), net
**+26.1 µs**, linear in converted bytes (32 KB → +16.3, 64 KB → +26.1). **`pl.load` GM→L1 performs
ND→NZ inside the DMA descriptor at no pipe cost; `pl.move` UB→UB pays for the same conversion on the
V pipe. Exchange rate ≈ 1.9 : 1 against.** The pre-check should compare the **absolute busy time**
removed against the conversion added, never the ratios.

ΣAIC also *fell* 0.949 → 0.789 with `aic_mac` 0.568 → 0.508: the cube got emptier, not faster,
because this kernel's four-handshake chain strictly alternates cube and vector per step and the new V
work landed in the segment the cube blocks on. **An idle pipe is not a concurrent pipe.**

Two findings from the attempt are worth more than the attempt:

- **Never put a row tail in an NZ handoff's `valid_shape`.** In NZ the column-block pitch is set by
  the **row** count, so a row-tail `valid_shape` ([64,64] inside a physical [128,64] tile) packs
  fractals at a different pitch than the GM→L1 load used, and the cube's `L1→L0B` move — which goes
  by *physical* shape with `compact` unset — then reads wrong offsets. Convert the **whole physical
  tile** over a zeroed tail instead. **Column tails are safe**: they drop whole column blocks. This
  asymmetry is exactly why the documented failure is "deterministic partial correctness" — only
  row-tail shapes break, so a suite whose production shape has no row tail passes a broken kernel.
  The first attempt was bit-identical at K=128 and diverged only at K=64.
- **Byte-identity is the right acceptance standard for a pure data-movement change.** It subsumes
  `matched_ratio` and is cheaper to reason about — necessary here, because the baseline's own
  `matched_ratio` is not 1.000 everywhere (t3c `dh` is 0.999959), so a tolerance standard chases a
  moving target. Final evidence: md5 of each output across 7 shapes run twice, **14/14 identical**,
  including B=4, odd N=3 and V=96.

The only remaining way to make this lever pay is to have the producing VFs **write NZ directly**
(`vf.store_align` at fractal offsets), deleting the conversion rather than relocating it. That is
re-authoring three verified VF chains, not a tuning change.

### 9.5 Measurement: the noise floor is stratified by kernel size, and it rules two kernels out

Untouched sibling kernels in the same passes give the floor for free, and it depends strongly on
kernel duration:

| duration class | observed spread |
|---|--:|
| >100 µs (`train_wy` 169.9/170.4/169.5, `train_scan` 185.1/185.6/184.4, `dqkw` 160.7/159.5/160.5) | **≤0.8%** |
| ~40 µs (`wy_reduce` 46.1/37.7/37.1) | **24%** |

So `wy_reduce` and `finalize` are **not optimizable on this harness** — not because they lack
headroom, but because any change is indistinguishable from noise, and a number produced there would
not be a result. An L2-locality target had been identified for `wy_reduce` (84.7% read hit rate on
47.5k misses, the worst meaningful rate in the family) and was **deliberately not pursued** for this
reason. Either restrict targets to >100 µs kernels or first make small-kernel measurement stable
(many repeats, fixed device, no co-tenants).

Establish the floor from the current run's untouched kernels before ratcheting; do not carry a
previous round's figure.

### 9.6 Three more ways these measurements mislead

Add these to §0's list.

**6. The averaged bandwidth columns hide what a pipe achieves while working.** `Memory` reports
bandwidth averaged over the whole kernel window, so divide by the pipe's ratio:
`aic_l1_write_bw / aic_mte2_ratio` is MTE2's throughput while busy (and `aic_l1_read_bw /
aic_mte1_ratio` is MTE1's).

| kernel | `aic_l1_write_bw` | `mte2` | **MTE2 busy** | MTE1 busy |
|---|--:|--:|--:|--:|
| `gdr_fwd_train_wy` | 14.71 | 0.146 | **100.7** | 584 |
| `gdr_bwd_dv_local` | 23.31 | 0.225 | **103.6** | 622 |
| `gdr_bwd_reverse_scan` | 22.60 | 0.265 | 85.3 | 603 |
| `gdr_bwd_dqkw` | 18.55 | 0.252 | 73.6 | 680 |
| `gdr_bwd_wy_matmuls` | 22.09 | 0.384 | 57.5 | 622 |
| `gdr_fwd_train_scan` | 28.67 | 0.511 | 56.1 | 541 |

`train_wy` has the **lowest** average bandwidth in the family and the **highest** throughput while
busy — read the average as efficiency and you condemn the healthiest kernel. The 1.8× spread on
GM→L1 against a uniform 541–680 on L1→L0 is what pointed at descriptor granularity (§9.2).
**And never take one kernel's busy throughput for the platform roof**: §8.2 called `dqkw`'s
~70 GB/s/core platform peak and concluded its MTE2 was a hard floor, but `train_wy` and `dv_local`
reach ~101 on the same host. Use the family-wide demonstrated maximum as the reference.

**7. A ratio moving the wrong way is not a regression.** `dv_local`'s `aic_mte2` absolute busy time
*grew* 11.8 → 13.1 µs from the prefetch's clamped last-iteration re-read while the kernel got 22%
faster; `train_wy`'s `aiv_vec_resc_cflt_ratio` *rose* 0.489 → 0.964 while it got 23% faster (it is a
conflict ratio, not a utilization, and is not actionable alone). Judge on latency and ΣAIC; use
ratios to find direction. A single ratio rising alongside a rising ΣAIC is healthy — a shorter
window makes the same absolute busy time a larger fraction.

**8. L2 miss counts and latency are not coupled in this family.** The hypothesis that `wy_matmuls`'
81.2% cube-side read hit rate (62,980 misses, worst in the family) indicated an operand-reuse
problem was **measured and rejected**: a change worth −8.4% left misses *up* at 65,215, while
`dv_local`'s fell 14,417 → 13,023 as it gained 22%. Both directions say the same thing, so
`cube-04` FullLoad is not indicated here. Two lessons: multiply a hit rate by its **absolute** count
before ranking it (`dv_local`'s alarming 24.0% vector-side rate sits on 11k accesses against 2.03M
on its cube side — pure noise), and even when the volume is real, verify cheaply that misses and
latency move together before redesigning a data flow around locality.

### 9.7 Measured dead this round — do not re-spend

- `reverse_scan` UB→L1 bridge: **+7.5% / +12.0%** (§9.4). Mechanism priced, not merely observed.
- `train_wy` L1→L0 hoist of the two loop-invariant operands (`attn` and `attn^T`, 64 KB/chunk of
  avoidable MTE1): ratio moved (`c_mte1` 0.084 → 0.073, ΣAIC 1.070 → 1.057) and **the clock did
  not** — 131.07 median against 131.43, inside noise and slower by mean. Recorded as a negative
  result block in the kernel file itself.
- `train_wy` `cube-01` `phase=`: **deliberately not attempted.** The card's own "not applicable"
  rules it out (L0C already double-buffered, and single-K-block matmuls cannot validate the
  pairing), and mispairing is a documented **card-damage** failure — ECC error 171 plus driver
  alarm, with physical loss recorded. Not a risk worth taking on a shared machine for an overlap
  already obtained by plain L0C ping/pong.
- `train_scan`: ZN transposed load replaced by a plain ND load at identical bytes → **192.6 µs**,
  worse, which killed the "the transposed read is the defect" hypothesis (the transpose is only
  0.043 of the window, ~5.8 µs). Resident `s_hist`, removing a **genuine** loop-invariant reload
  (MTE2 bytes −13%, `c_mte2` 0.422 → 0.321, 4 descriptors → 1) → **zero** latency change, +98 KB L1,
  rejected. Deleting the per-chunk recurrence `wait(S_READY)` outright → **identical**, so that
  barrier costs ~0. Together these are the evidence that below ~0.45 MTE2 left the critical path.
- `dv_local` gate split (moving the `exp_sub` transcendental ahead of ev0 through an 8 KB UB tile):
  flat, −0.3 µs, overlapping ranges. Informative: after the second barrier fix, the post-ev0 vector
  segment is no longer binding.
- `wy_matmuls` L0C ping/pong: **blocked, not tried.** Three depth-1 accumulators, and L0C is
  114,688 of 131,072 B — exactly one `[C,C]` fp32 slot free. Taking L0C to 100% with no margin would
  also need a free mutex id whose only remaining merge candidate is written cross-core by the vector,
  which §8 records as a −1.8% trap.
- `train_scan` multi-accumulator split of the serial `matmul_acc` RAW chain (`oac_0` carries three
  dependent MMADs) is the next real lever there and is **blocked by mutex ids at 32/32**, every id
  declared.

### 9.8 Where the pipeline now stands

Two kernels meet the 80% bar on their bounding pipe (`wy_matmuls` 0.812, `train_scan` 0.795–0.814).
`dqkw` 0.720, `dv_local` 0.703 and `train_wy` 0.502 are improved but short, and each has a named,
evidenced reason: `dqkw` needs bf16 operands to cut its 352 KB/chunk (a SPEC decision);
`train_wy`'s vector side has only 4.4% idle left and its in-place VF idiom forbids concurrent
vector DMA, with 1280 B of UB free; `train_scan` is MAC-floored at 145.9 ± 1 µs absolute MAC time
across five variants. `reverse_scan` at 0.568 and 16/28 blocks is the one kernel that is structurally
stuck (§8.4, §9.4) and is now the largest single component at 217.8 µs, 24% of the total.

The reusable findings of this round are knowledge cards `vec-17`, `cube-10` and revisions to
`cube-08` and `metrics-01`, in
`cannbot-skills/ops/pypto-pro-op-perf-tune/references/knowledge-cards/`. The correction to the
gather-cost claim is in `references/general-knowledge/a5-roofline-and-levers.md`, with the
superseded text struck through rather than deleted.
