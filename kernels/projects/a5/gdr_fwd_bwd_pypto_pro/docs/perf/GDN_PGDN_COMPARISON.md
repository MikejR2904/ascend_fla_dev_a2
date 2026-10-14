# A5 GDN forward and PGDN forward — measured, and why the GDR fp32 finding does not transfer

Measured 2026-10-14 on a 28 cube / 56 vector Ascend950PR (revision `957b`, CANN
9.1.0-beta.1, npu-smi 25.7.rc1), one free card, no co-tenants. Units measured as
they stand on branch `task/A2-K1`; **nothing under `kernels/projects/a5/` was
modified**.

Written up here because the GDR work in this directory raised a specific question —
whether the fp32 cube-matmul ceiling that dominates GDR also dominates the ascriptor
GDN/PGDN units, which share the same algorithm family and have `_bf16` siblings
making the same bet. The answer is no, for a structural reason worth recording.

## The headline: these units issue no cube MMADs at all

`aic_mac_time = 0` and `aic_mac_ratio = 0` on **every** kernel of both units, in both
dtypes, at both lengths. `aic_cube_fops = 0`; `cube_utilization ≤ 0.058%`. Task Type
reads `MIX_AIC` but 100% of `aicore_time` is `aiv_time`.

So the GDR result — fp32 cube matmul issuing at ~1/16 of peak, 623 ns per `[64,64]`
MMAD against 42 ns in bf16 — **cannot apply**: `ns per MMAD` has zero in both
numerator and denominator. Real MAC utilization (`cube_utilization × aic_mac_ratio`)
is 0 on all 11 kernels.

Three independent confirmations, in increasing order of authority:

1. **Source.** Both units' `kernels/stages.py` use only `vf(...)` vector-function
   blocks over `Position.UB` tensors. Zero occurrences of `matmul`, `mmad`, `gemm`,
   `L0A`, `L0B` or `L0C` in `gdn_chunk_fwd`, `pgdn_chunk_fwd`, or either `_bf16`
   sibling.
2. **The repo already says so.** `gdn_chunk_fwd/kernels/stages.py` opens "Five-launch
   FP32 **Vector** baseline", and `gdn_chunk_fwd_bf16/evidence/timing-summary.json`
   records `"speed_gate": "none; Vector implementation, no Cube throughput claim"`.
3. **Measured**, as above.

**The ceiling these units actually sit against is core count, not pipe rate.** Every
kernel launches `Block Num = 2` against 56 vector cores. The per-core VEC pipe is
saturated (`aiv_vec_ratio` 0.90–0.99), so each kernel is individually efficient —
but machine-level vector occupancy is `0.988 × 2/56 ≈ 3.5%`. That is the structural
headroom, and it is a different problem from anything in the GDR record.

## Latency

Public entry points, `block_dim` 2, warmup 10 / repeat 50, 3 rounds, synchronized
wall time including dispatch. B=1, H=HV=8, K=V=128.

| unit | T | FP32 | BF16 | BF16/FP32 |
|---|--:|--:|--:|--:|
| GDN forward | 4096 | 19.96 ms | 19.64 ms | 0.984 |
| GDN forward | 1024 | 5.15 ms | 5.00 ms | 0.971 |
| PGDN forward + ATK | 4096 | 38.81 ms | 38.52 ms | 0.993 |
| PGDN forward + ATK | 1024 | 10.12 ms | 10.15 ms | 1.002 |

PGDN against its own GDN-equation cost baseline (the five PGDN-owned chunk kernels
with equal normalized read/write keys and no ATK — a cost baseline, **not** GDN
implementation performance): 1.043–1.057 at T=4096, 1.089–1.106 at T=1024.

Dispatch/host overhead (public call minus kernel sum): PGDN ~1.4 ms of 38.8 (3.7%),
GDN ~0.32 ms of 19.96 (1.6%).

## Per-kernel, T=4096

msprof `PipeUtilization` / `ArithmeticUtilization`, 6 calls per pass, per-call
medians. `aic_mac_time`, `aic_mac_ratio` and `aic_cube_fops` are **0 on every row**
and omitted from the tables.

### PGDN

| kernel | FP32 µs | BF16 µs | `aiv_vec_ratio` | Block Num |
|---|--:|--:|--:|--:|
| Scores | 14050 | 14060 | 0.988 | 2 |
| Output | 12350 | 12340 | 0.978 | 2 |
| Wy | 5535 | 5537 | 0.972 | 2 |
| Scan | 3984 | 3984 | 0.959 | 2 |
| Atk | 1091 | **953** | 0.789 → 0.923 | 2 |
| Prepare | 381 | **297** | 0.170 (mte2 0.653) | 2 |
| **sum** | **37391** | **37171** | | |

The unit's own `--profile` stage timing, independent of msprof, agrees within ~1%
(sum 37669 µs).

### GDN

| kernel | FP32 µs | BF16 µs | `aiv_vec_ratio` | Block Num |
|---|--:|--:|--:|--:|
| Scores | 6935 | 6927 | 0.976 | 2 |
| Wy | 5675 | 5676 | 0.973 | 2 |
| Scan | 3997 | 3988 | 0.957 | 2 |
| Output | 2667 | 2668 | 0.899 | 2 |
| Prepare | 367 | **252** | 0.177 (mte2 0.590) | 2 |
| **sum** | **19641** | **19511** | | |

No `aic_mac_ratio` / `mac_ratio` figure exists anywhere in any a5 `evidence/`
directory, so the per-kernel pipe data above is new rather than a re-measurement.

## Why the BF16 arms could not have shown a cube ceiling even if one existed

The `_bf16` units are **I/O-only conversions**. Per
`pgdn_chunk_fwd_bf16/README.md`: "BF16 q/k/v enter device kernels directly;
**normalization, ATK and all intermediate tensors remain FP32**." Only the q/k/v GM
loads and the `o` store change width — no *operation* is converted, just bytes.

This is the same constraint the GDR work ran into from the other side: `pl.matmul`
pairs lhs and rhs at the same dtype, so the unit of conversion is an operation, not a
tensor. Here the consequence is visible to the microsecond: at T=4096 the only
kernels that move are the two that touch raw q/k/v/o bytes (Atk −138 µs, Prepare
−84 µs = −222 µs of 37391, i.e. −0.6%, which is the measured −0.7% end-to-end), while
Scores/Output/Wy/Scan are duration-identical (14050↔14060, 12350↔12340, 5535↔5537,
3984↔3984).

**If the fp32-cube finding is to bear on this family, the target is a cube-based
rewrite of Scores and Output** — 26.6 ms of PGDN's 38.8 ms at T=4096 — not these
units as they stand.

## Correctness on the timed cases

Nothing above is a timing on a failing case.

- **PGDN `measure.py`**: all 4 workloads `passed: true`, verified before *and* after
  timing against both the pinned CPU recurrence and an independent CPU block solve,
  plus the pinned GDN semantics check on the baseline arm. Post-timing relative L2:
  FP32 `o` 4.96e-7–5.10e-7, `final_state` 6.87e-7–1.13e-6, `final_A_state`
  3.18e-7–3.27e-7; BF16 `o` 1.66e-3 (inside the 5e-3 bf16 budget), states unchanged
  at ~1e-6. Inputs verified unmutated by digest.
- **PGDN `benchmark.py`** native case `pgdn340m_bd2` (the exact `measure.py` shape):
  `passed: true`, all 17 composition stages plus independent leaves and both oracles,
  rel L2 ≤ 8.2e-7.
- **GDN `measure.py`**: `TIMING_PASS`, verification before and after each sandwich,
  `inputs_unchanged: true`, plus byte-equality assertions between the FP32 and BF16
  arms. rel L2: FP32 `o` 6.15e-7, `final_state` 7.7e-7–1.6e-6; BF16 `o` 1.66e-3.
- Device-free: `pgdn_chunk_fwd/run.py reference` → all **60** contract cases pass.

## ⚠ A ~15× gap against the committed `evidence/` receipts

| workload | prior `evidence/` | measured here | factor |
|---|--:|--:|--:|
| PGDN T=4096 fp32 | 566.28–566.42 ms | 38.81 ms | 14.6× |
| PGDN T=1024 fp32 | 143.65 ms | 10.12 ms | 14.2× |
| PGDN GDN-baseline T=4096 | 547.66–547.69 ms | 36.78 ms | 14.9× |
| GDN T=4096 fp32 | 302.36–302.42 ms | 19.96 ms | 15.1× |
| GDN T=1024 fp32 | 76.78–76.84 ms | 5.15 ms | 14.9× |

**What agrees** is the relative structure: the PGDN/GDN-baseline ratio (1.034–1.045
prior vs 1.043–1.106 here), BF16 ≈ FP32 in both, and correctness residuals in the
same decades.

**What differs** is a near-uniform 14.2–15.2× across five independent workloads and
two units, with the kernel sources verified byte-identical to the evidence pins
(`stages.py 6d07abf7`, `atk.py e5643c78`, `pipeline.py e078814a`,
`measure.py 94c80206`). That uniformity points at machine/toolchain, not code. Two
documented differences: the prior evidence records SoC `Ascend950PR_9589 V100` and
CANN 9.2.0, where this box reports revision `957b` and CANN 9.1.0-beta.1.

**Not attributed** — closing it needs the other revision. Flagged because a ~15× gap
against an accepted unit's recorded receipts should be resolved before either number
is quoted as "the" latency for these units.

## Caveats on how GDN was measured — read before quoting the GDN rows

`kernels/projects/a5/gdn_chunk_fwd/benchmark.py` **is dead code on branch
`task/A2-K1`** and could not be used. Both requested cases fail identically before
any kernel launches:

```
  File ".../kernels/projects/a5/gdn_chunk_fwd/benchmark.py", line 106, in launch
    scalars={n:scalars[n] for n in op.scalar_names}
               ~~~~~~~^^^
KeyError: 'HK'
```

Root cause, confirmed statically: on this branch
`ascend_fla/ops/gdn_chunk_fwd.py::_compiled()` compiles
`_native_pipeline().all_entries()` — i.e. only the **`gdn_chunk_fwd_bf16`** unit's
entries — while `benchmark.py:73` still zips those compiled ops against the *legacy*
FP32 pipeline's names. `gdn_chunk_fwd/kernels/pipeline.py:42` supplies
`scalars = dict(B,T,H,N)`, but the bf16 ops additionally require `HK`. GDN forward
has migrated into the `_bf16` unit and the old unit's benchmark harness was not
updated. Left unfixed.

Consequences, stated plainly:

1. GDN was measured through **`gdn_chunk_fwd_bf16/measure.py`** — the live unit's own
   entry point, same sandwich protocol.
2. Its FP32 arm **does** genuinely drive `kernels/projects/a5/gdn_chunk_fwd/kernels`:
   the required immutable baseline checkout was reconstructed read-only from rev
   `bdce6671`, whose `ascend_fla/ops/gdn_chunk_fwd.py` hashes to exactly the
   `baseline_source_sha256` `32f9c19c…` in the prior evidence. So the numbers are
   directly comparable to the prior receipts.
3. **Shape differs from the contract's H=16 cases**: this harness runs B=1,
   **H=HV=8**, T ∈ {1024, 4096}. The H=16 path is what the broken `benchmark.py`
   would have exercised. H=HV=8 does have the advantage of exactly matching PGDN's
   `measure.py` shape, making the GDN-vs-PGDN comparison apples-to-apples.
4. `ascend_fla/ops/pgdn_chunk_fwd.py` on this branch hashes `786005e7` against the
   evidence pin `3ce60fbe` — the wrapper gained BF16 dispatch; the PGDN *kernels* are
   unchanged. The prior evidence therefore predates separate BF16 pipelines, so its
   BF16 arm was a pure widening cast while these runs dispatch to real BF16-input
   kernels — and still gain only 0.7–2.9%.

## Two reproducibility hazards found

- **ascriptor is not anonymously obtainable.** All three gitcode repos
  (`ascriptor`, `ascriptor-kernels`, `ascriptor-agent`) return **HTTP 401** on
  `info/refs`, 403 on the web page, 401 on the v5 API. Not a mirror mismatch —
  simply not anonymously readable. Also ruled out: GitHub (no `ascriptor` under
  `ddddwee1`'s 36 public repos; both `ddddwee1/ascriptor` and `ddddwe/ascriptor`
  404), gitee (empty search), and two pypi mirrors. The measurement was only possible
  because a **prior workspace checkout of the pinned revision survives on the box**,
  matching `docs/matrix/ops.json → ascriptor_pin…library = 90cfcdc720bb…`. Bring-up
  is ~2–4 h *with* credentials and effectively unbounded without them. Worth securing
  a credential or vendoring a tarball: as it stands these units are one `rm -rf` away
  from unbuildable.
- **CRLF silently breaks every sha256 pin.** The Windows clone has
  `core.autocrlf=true`, so the working tree is CRLF while the index is LF.
  `_unit_runner.py` hashed `4a684d3f…` instead of the `runner-source.json` pin
  `8e91d95e…`; `git archive` also emits CRLF. Every transferred text file needed
  `\r` stripping before any pin matched. Worth noting in the unit protocol — a pin
  mismatch here looks like tampering and is actually line endings.

Also: `pgdn_chunk_fwd/measure.py` requires two externally-pinned FLA naive oracles
(`--fla-naive` `3baa67a5…`, `--fla-gdn-naive` `d1cf1799…`) that ship with neither
repo; both were located by hash and verified after copying.

## Provenance

Measured with the units' own entry points (`measure.py`, `benchmark.py`, `run.py`) at
their pinned revisions, in an isolated
`python -m venv --system-site-packages` per AGENTS.md §5 — nothing was installed into
the shared container environment. No git state was modified or committed in either
repo. Device and host identities omitted per AGENTS.md §5.

---

# Forward-only comparison: GDR vs GDN vs PGDN

Added after the above. **Forward only, for GDR, means `train_wy` + `train_scan`** —
the other six kernels in that pipeline are backward.

## GDR forward at B=1, T=16384, H=16, K=V=D=128

Dedicated forward-only run (the two kernels and nothing else), two msprof
`PipeUtilization` passes, warmup 2 / repeats 5:

| kernel | µs | blk | `aic_mac` | ΣAIC | cube % | real MAC % |
|---|--:|--:|--:|--:|--:|--:|
| `gdr_fwd_train_wy` | 2047.4 | 28 | 0.474 | 1.135 | **98.6** | 46.8 |
| `gdr_fwd_train_scan` | 2825.9 | 28 | 0.810 | 1.586 | 56.9 | 46.1 |
| **forward total** | **4873.3** | | | | | |

Totals 4876.1 and 4870.4 µs. Host-side wall time including dispatch is 5391 µs, so
dispatch is ~10% at this shape.

**The forward kernels cost the same alone as inside the full fwd+bwd pipeline**:
2051.1 / 2043.6 here against 2051.4 in-pipeline for `train_wy`, and 2825.0 / 2826.8
against 2826.4 for `train_scan` — 0.09% apart on the total. So there is no L2
interference from the six backward kernels, and forward cost is cleanly separable.

Forward is **34.2% of the 14.24 ms training pipeline**; backward is 9.37 ms, i.e.
**1.92× the forward**.

## Normalised against GDN and PGDN

The shapes cannot be matched directly: PGDN's contract caps T at 4096, and the only
working GDN timing harness runs H=HV=8 (see the dead `benchmark.py` above). So
normalise by work units = tokens × heads — GDR has 8× as many (262144 vs 32768).

| operator | shape | total | ns per token·head | vs GDR |
|---|---|--:|--:|--:|
| **GDR forward** (`train_wy`+`train_scan`) | T=16384, H=16 | 4.87 ms | **18.6** | 1.00× |
| GDN forward, fp32 | T=4096, H=8 | 19.96 ms | 609.1 | 32.8× |
| GDN forward, bf16 | T=4096, H=8 | 19.64 ms | 599.4 | 32.2× |
| PGDN forward + ATK, fp32 | T=4096, H=8 | 38.81 ms | 1184.4 | 63.7× |
| PGDN forward + ATK, bf16 | T=4096, H=8 | 38.52 ms | 1175.5 | 63.2× |

**Most of that 33–64× is parallelism, not kernel quality.** GDN and PGDN launch
`Block Num = 2` against 56 vector cores; GDR launches 28 blocks and uses both the
cube and vector pipes. Giving GDN/PGDN a *perfect* hypothetical 28× scaling from 2 to
56 vector cores — which real code never achieves — the picture becomes:

| operator | ns per token·head, core-normalised | vs GDR |
|---|--:|--:|
| **GDR forward** | 18.6 | 1.00× |
| GDN forward, fp32 | 21.8 | 1.17× |
| PGDN forward + ATK, fp32 | 42.3 | 2.28× |

So on equal-core terms GDR's forward is still ~1.2× faster than GDN's and ~2.3×
faster than PGDN's, and the honest headline is that **the GDN/PGDN gap is dominated
by using 2 of 56 cores**. Closing that is worth far more to them than any dtype or
scheduling change.

### Why this comparison is sound even though the receipts disagree by 15×

The ~15× gap flagged above is between *these GDN/PGDN measurements* and the
**recorded `evidence/` receipts**. It does not affect the comparison in this section,
because the GDR and GDN/PGDN numbers here were taken on the **same machine, same
CANN 9.1.0-beta.1, same SoC revision `957b`, within the same working session**.
Whatever causes the discrepancy against the receipts applies to the GDN/PGDN
*environment*, which both sides of this table share.

### Four caveats on reading the table

1. **GDR's forward is a *training* forward and emits 17 tensors**, including
   `attn`, `w_t`, `kbeta_t`, `q_t` and `a_mat` that exist only to feed the backward.
   GDN/PGDN forward emit `o` plus final state(s). So GDR is being charged for strictly
   more work than an inference forward would do — a separate fused inference-only
   forward exists (`kernels/gated_delta_rule_fwd_impl.py`) and was **not** measured
   here. The GDR row is therefore pessimistic as an inference comparison and
   favourable as a training-forward one.
2. **PGDN computes more than GDN or GDR** — it carries an extra ATK state and an
   asymmetric key transform. Its own GDN-equation cost baseline (36.78 ms fp32 at
   T=4096) is the fairer GDN-side figure, and PGDN is 1.04–1.11× that.
3. **Normalising by tokens × heads assumes linearity in both.** It holds well for
   GDR (measured sub-linear, 16× work for 15.6× cost) but was not verified for
   GDN/PGDN across H.
4. **Different numerics.** GDR has bf16 public inputs/outputs with fp32 cube
   accumulation; the GDN/PGDN fp32 rows are fp32 public, and their bf16 rows convert
   I/O only.
