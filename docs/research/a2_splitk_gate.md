# A2 split-K / accumulate-settle gate (A2-11)

Real-machine reproduction and work-around verification for ascriptor library defect **M10-081** on
the Ascend 910B3 (A2 SoC, c220 family). This document delivers **evidence and a recommendation**; it
does **not** decide that A2 operator conclusions now count — that step is the user's (`AGENTS.md` §2,
D-PM-30). Until the user rules, every A2 number here and elsewhere is observation only.

Everything below was measured on the 910B3 through the ascriptor **aclnn** launcher, on the A2-01
pinned revision (ascriptor library `90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5`, kernels
`b3b3f9c16df7c4626ed3c081032a1be5a753d0b1`) so the split-K FP32 numbers are comparable with A2-01.
The reproducer is `benchmarks/a2/repro_splitk_fp32.py`; raw scrubbed console logs and per-case
receipts are under `benchmarks/a2/evidence/splitk/`.

## 1. The defect

On the c220 A2 family, two short cube `mmad` that update the **same L0C** are not interlocked by the
hardware: a second `is_init=False` MMAD can read the accumulator before the first has settled, giving
a silently wrong result. The functional model and the pipe model execute in program order and are
bitwise-clean, so **only silicon shows it**. It surfaces in `matmul(..., splitk=S)`, which expands to
one MMAD per K fragment into one L0C (first `is_init=True`, the rest `is_init=False`).

Three constructs, three fates in the pinned library:
- **FP32 split-K** — the pinned `desugar.py` inserts `barrier(Pipe.M)` after each MMAD of the
  A2-family FP32 split-K expansion (auto-repaired).
- **BF16 / FP16 split-K** — the settle rule is FP32-only, so no barrier is inserted (`gaps.json`
  `a2-splitk-bf16-fp16-unsettled`).
- **FP32 hand-written accumulate chain** (separate `matmul` calls into one L0C) — only a lowered-IR
  lint trap, no automatic barrier (`gaps.json` `a2-splitk-fp32-cube`).

## 2. Method

Every operand is a bounded dyadic value (eighths in [-1, 1]; bias in quarters), so the product and
every partial sum are exact in FP32 / BF16 / FP16 alike — the device output is compared **bitwise**
against an independent CPU float64 reference rounded once to FP32. Any deviation is a wrong result,
not rounding. Outputs are pre-filled with a poison value so an unwritten tile cannot pass. Each case
runs in its own process (CANN reads `ASCEND_CUSTOM_OPP_PATH` once; one op-name is one build per
process), repeated to characterize a timing-dependent fault, with three perturbations of the
inter-MMAD timing (`--jitter`: independent scratch-L0C MMADs inserted between the two target MMADs)
and across block_dims and cards. A wrong-but-finite output is decomposed as a least-squares fit over
the per-MMAD partial products: integer coefficients near {0,1} with ~0 residual mean whole fragments
were dropped or repeated — the stale-accumulator signature.

## 3. Results

All runs on the 910B3, aclnn launcher, pinned revision; per-case receipts in
`benchmarks/a2/evidence/splitk/*.json`, scrubbed console in the matching `*.log`.

| construct | dtype / shape | runs (bitwise / wrong / device-fault) | verdict |
|---|---|---|---|
| split-K, FP32 (repaired) | M64, K128, split-K64, bd1 + bd2 | **2000 / 2000 / 0 / 0** | **bitwise-stable**; lowered IR shows the settle `barrier(Pipe.M)` (`ir_settle=1`). Fix confirmed on device. |
| split-K, BF16 | M16, K128, split-K16 | **500 / 0 / 500 / 0**, with **500 distinct output hashes** | **wrong**, non-deterministic; decomposition `tile_coef ≈ [0,1,0,1,0,1,0,1]` — every other K-fragment dropped (stale-accumulator signature). |
| split-K, BF16 / FP16 | **M32**, K128, split-K16 | **200 / 0 / 0 / 200** | **AI-Core exception** — `test.cpp:53 aclrtSynchronizeStream(stream) failed: 507015` on every run (`evidence/splitk/m32_aicore_exception.txt`). Device fault, not a wrong number. |
| split-K, BF16 | **M64**, K128, split-K16 | **200 / 199 / 0 / 1** | **bitwise-correct** (the M64 tile does not reuse the two-slot L0 ring); the single fault was one transient device event, not a wrong result. |
| hand chain, FP32, no barrier | M16, K16, 2 & 8 terms (`is_init=False`), + jitter | **2000 / 2000 / 0 / 0** | **not reproduced** — bitwise-correct across all runs incl. inter-MMAD timing perturbation. The lint trap fires (`traps=1`) but the fault did not manifest. *Not reproduced*, **not proven safe**. |
| hand chain, FP32, barrier | same, `barrier(Pipe.M)`, bd1 + bd2 | **2000 / 2000 / 0 / 0** | **bitwise-stable** (work-around control). |

The BF16 M16 result is decisive: **500 different wrong outputs in 500 runs** at fixed input — perfect
non-determinism is proof of a hardware timing race, not a deterministic miscompile. A per-run device
fault (the M32 exception, or a rare transient) is recorded and the sweep continues, so a fault is
never miscounted as a wrong number.

## 4. Work-around verification

The a2 units (A2-03 / A2-09) apply the discipline: every `is_init=False` accumulate is preceded by
an explicit `barrier(Pipe.M)`, and BF16/FP16 split-K with M<64 is never used. This document verifies
both directions on real silicon:
- **With the barrier** (the `chain_*_bar` cases): bitwise-stable across all repetitions and block_dims
  — the settle discipline holds.
- **Without the barrier** (the `chain_*_nobar` negative control): the FP32 chain did not trip here,
  but the BF16/FP16 split-K path (the same missing-settle hazard, one dtype over) is reliably wrong
  or faults. Because the FP32 chain shares the exact hardware mechanism and the fault is timing-rare,
  the barrier is **conservatively retained** — "did not trip in N runs" is not "cannot trip"
  (A2-04's lesson: a timing fault masquerades as a regular symptom).

The static guard `tests/test_a2_accumulate_barriers.py` fixes this discipline over
`kernels/projects/a2/**` (every `is_init=False` matmul is immediately preceded by `barrier(Pipe.M)`;
no BF16/FP16 `splitk` at M<64), so a future edit that drops a barrier turns the suite red.

## 5. Recommendation (not a decision)

- **Proven safe on this A2**: FP32 split-K (the pinned desugar settle fires and holds bitwise);
  BF16 split-K at M64.
- **Not reproduced, keep the guard**: the unbarriered FP32 hand chain did not fault in these runs;
  this is *not* a safety proof. Keep the explicit `barrier(Pipe.M)` and the static guard.
- **Must stay permanently forbidden / hard-errored** (`AGENTS.md` §7): BF16 / FP16 split-K (and the
  equivalent hand chain) at M<64 — reliably wrong at M16, device fault at M32. These must not be
  reachable in an a2 kernel; the lint/guard should keep failing them, not warn.

### Conditions for "A2 conclusions count" (for the user to rule on, D-PM-30 / §2)

1. All A2 numbers of record are produced on the pinned ascriptor revision (`90cfcdc` / `b3b3f9c`),
   each receipt carrying its SoC / CANN / op-package identity.
2. Every dispatched a2 kernel passes the static accumulate-barrier guard, and no path uses BF16/FP16
   split-K at M<64.
3. The FP32 split-K repair is confirmed bitwise on device (this document), not merely modeled.
4. The residual "not reproduced" FP32 hand-chain risk is carried as an explicit guard, not treated
   as absence of the fault.

If the user accepts these, A2 operator numbers can move from observation to conclusions of record.
The decision is the user's; the PM and assignee do not relax it.
