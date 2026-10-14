# Stage 3.5 KAL — OP5 `gdr_bwd_dqkw` (parent module M8, golden_step b3) — THE HEAVIEST OP

Hardware-aware (`pypto-kernel-arch-lang` / `easyasc.a5`) pre-build of OP5: the chunk-parallel
`dqkwg` backward (b3) — 6 matmuls + ds triangular mask + dg reduce, BHN-parallel, no carried
state. Authored against the a5 facade, validated in the Python simulator vs the frozen CPU FP32
golden `gdr_bwd_dqkw_golden_cpu.py`, and transpiled to PyPTO-Pro DSL.
**Simulator pass ≠ Stage-4 acceptance** — the `pypto-pro-op-verifier` on real a5 silicon is the
only authority. No `test_` deliverable was written; Stage 4 is NOT entered (needs a5 hardware).

Target pinned & asserted for every transpile: **arch=a5 (device_type=950), backend=pypto**
(via `transpile_guard.py`); limits in force UB=253,952, mutex=32, intra_flag=8, cross_flag=16,
L0A=L0B=64 KB, L0C=256 KB (a5.py override).

## Deliverables (under `custom/gdr_bwd_dqkw/kal/`)
- `gdr_bwd_dqkw_kal_module1.py` — the `@kernel` source (+ `build_args`, `_SHAPE_BINDINGS`).
- `staged_module1.py` — backend transpile to PyPTO-Pro DSL (1 `@pl.jit`, merge_siblings=True,
  40,028 B; 12 `pl.matmul`, nz=0, src_is_transpose(L0A)=0, advisories=0).
- helpers (not deliverables): `transpile_guard.py`, `measure_budget.py`, `emit_staged.py`,
  `sim_verify.py`.

---

## Simulator verification — PASS on all 4 §8 shapes × 3 seeds (ALL FOUR outputs)

`OpExec(..., simulator=True)` vs `gdr_bwd_dqkw_golden_cpu` (inputs built jointly through the full
upstream pipeline; the same tensors feed kernel and golden). Criterion: elementwise peak-scaled
`allclose(rtol=1e-2, atol=1e-2·max|ref|)` OR relative-L2 ≤ 2e-2, applied to dq_p, dk_p, dw, dg_p.
Run SEQUENTIALLY (single foreground call). Reproduce: `python custom/gdr_bwd_dqkw/kal/sim_verify.py`
→ `SIM_ALL_PASS`.

| case (shape [B,T,H,K,V]) | dq_p rel-L2 | dk_p rel-L2 | dw rel-L2 | dg_p rel-L2 |
|---|---|---|---|---|
| aligned [1,128,4,64,64]   | 0        | 0        | 0 | ≤6.8e-9 |
| ttail   [1,100,4,64,64]   | 0        | 0        | 0 | ≤7.1e-9 |
| t3c K=V=128 [1,192,2,128,128] | ≤4.0e-8 | ≤1.2e-7 | 0 | ≤9.6e-7 |
| tail_vlt V<K [1,200,3,128,96] | ≤3.0e-8 | ≤1.2e-7 | 0 | ≤5.4e-7 |

All 12 (shape×seed) finite, all four outputs within the criterion. K=64 chunks are bit-exact
(rel-L2 0); K=128 chunks carry only fp32 rounding (≤1e-6).

---

## Four-resource curve (C0 → C3), transpiled under the guard header

Reference = probe **206,848 B / 35 mutex default** (DESIGN anchor). Reproduce final:
`python custom/gdr_bwd_dqkw/kal/measure_budget.py a5` (worst-case §8 shape 1,200,3,128,96).

**Methodology (measurement discipline, mirrors OP4).** The b3 arithmetic (ds mask via a
register-resident MaskReg, decay_exp, per-row gate via `expsub`, dg reduce via `Reg.cadd()`) is
entirely REGISTER-RESIDENT — it adds ≈0 UB and no L0A/L0B/L0C. The four resources are fixed by the
declared tile set + section structure, so the C0 (structure, math stubbed) footprint equals C3;
the curve is FLAT by construction. **L0A and L0B are determined solely by the matmul operand tiling
and are identical at every checkpoint.** The single measured C3 is the whole curve.

| checkpoint | structure / state | ub_bytes | ub% | mutex (def/merge) | flags (i/c) | **L0A** | **L0B** | L0C | adv |
|---|---|---:|---:|---|---|---|---|---|---:|
| **C0** | tile + 4-handoff SINGLE structure, output-K static-unrolled, math stubbed | 238,432 | 93.9 | 47 / 22 | 0 / 0 | **64 KB** | **64 KB** | 32 KB | 0 |
| **C1** | + real b3 math (ds mask/exp/scale, decay_exp, gate, dq/dk/dw assemble, dg reduce, dg_last) | 238,432 | 93.9 | 47 / 22 | 0 / 0 | **64 KB** | **64 KB** | 32 KB | 0 |
| **C2** | + GM output materialisation (dq_p, dk_p, dw, dg_p + do/ds/dsT bridges) | 238,432 | 93.9 | 47 / 22 | 0 / 0 | **64 KB** | **64 KB** | 32 KB | 0 |
| **C3 (final)** | delivered, guard-asserted (merge fits; default>32 EXPECTED for OP5) | **238,432** | **93.9** | **47 / 22** | **0 / 0** | **64 KB** | **64 KB** | 32 KB | **0** |

**L0A / L0B outcome — both at the 64 KB cap, fit as-is (no TV-subblock recovery needed).**
The V-contraction is loaded FULL (no splitk — multi-chunk splitk is mis-lowered by the sim, see
findings). With the output-K statically subblocked to N=valid_k≤64, the peak fragments are:
- L0A: `[C,V]=[64,128]` fp32 = 32,768 B (cube-1 grad_o/v_new; cube-2 inter LEFT) → DBuff pair
  `addrs=[0x0,0x8000]` = 65,536 B = **64 KB at cap**. Intra LEFT `[C,C]=[64,64]` off-peak reuses it.
- L0B: `[N,V]=[128,64]` fp32 (inter RIGHT h/dh row-slice; intra RIGHT k/q_s`.T`) = 32,768 B →
  DBuff = 65,536 B = **64 KB at cap**.
This is EXACTLY the DESIGN DECISION-THREE projection and matches OP4's L0A=64 KB. Transpile clean
(0 advisories) ⇒ the L0A/L0B budget passed. **The DESIGN's TV-subblock recovery was NOT triggered
(neither exceeded).**

### GATE outcome (4 resources; reference probe 206,848 B)
- **UB 238,432 B (93.9%) → EXCEEDS the comfort gate (>215,000).**  FITS the hard delivery cap
  (238,432 < 253,952) but is ABOVE the drift-model prediction.
- **mutex default = 47 → EXCEEDS the comfort gate (>44).**  **merge = 22 ≤ 32 (delivered, safe).**
- L0A = L0B = 64 KB (at cap, "within 5%").  L0C = 32 KB.  intra/cross flags 0/0.
→ Per the task's 3-zone gate this lands in **"STOP and report"** on UB and mutex_default. It is a
**boundary finding, not a failure**: the operator is CORRECT and DELIVERABLE (merge fits every hard
budget), but the authored footprint is HEAVIER than the probe/drift model predicted. The cause is
attributed below.

---

## DRIFT FACTORS (mandatory) — 3rd data point — **the SINGLE idiom rule DOES NOT HOLD for OP5**

- **UB drift = 238,432 / 206,848 = 1.153  (+15.3%)**  — ABOVE the predicted SINGLE ×0.86.
- **mutex drift (default) = 47 / 35 = 1.343  (+34%)** ; merge 22 / 20 = 1.10.

**Prediction vs reality.** DESIGN predicted "OP5 is SINGLE ⇒ mutex ×1.0, UB ×0.86 (~178,700 B),
possibly sheds merge". **Both were wrong.** The realized SINGLE idiom for OP5's dataflow required
DBuff-ing the 5 L0C accumulators, the 5 UB landing tiles, AND the 2 L1 h/dh operands to break
cross-iteration hazards across the two static output-K subblocks (see findings). That mandatory
DBuff overhead — NOT present in the probe's single-pass stub — inflates UB by +15% and default
mutex by +34% ABOVE the model, and OP5 firmly **NEEDS merge** (default 47 ≫ 32).

**Generalisation verdict:** the transferable "SINGLE ⇒ mutex≈probe, UB≈probe×0.86" rule from OP3/OP4
**breaks whenever an op tiles its OUTPUT axis in a loop** (OP5 tiles output-K into 2 subblocks). The
per-subblock buffers must be double-buffered for correctness, which the single-pass scans (OP3/OP4,
which tiled only the CONTRACTION/free axis and carried ONE state) never paid. **This is the load-
bearing 3rd data point.**

### Implication for OP6a / OP6b (they have MORE matmuls, likely output-tiled)
- If OP6a/OP6b tile any OUTPUT axis (K or V) into subblocks, expect the SAME DBuff tax: UB ≈
  probe×1.1–1.2 and default mutex ≈ probe×1.3, NOT the ×0.86/×1.0 optimistic model. Budget them
  against the HARD delivery cap with merge, not the drift model.
- OP6a's probe (~higher matmul count) at ×1.15 UB would risk the 253,952 cap. **Re-measure early;
  the L0A/L0B 64 KB cap is also exactly-at-cap here, so any extra L0A operand (OP6a has more) that
  cannot off-peak-reuse would overflow.** This is a genuine campaign risk for OP6a.

---

## MEASURED merge_siblings VERDICT (mandatory)

- **default mutex = 47** (> 32 hard limit).  **merged mutex = 22** (≤ 32, safe).
- **OP5 NEEDS merge — it does NOT shed it.** The DESIGN's "SINGLE default ~35 may fall ≤32 and shed
  merge" is DISPROVEN: default is 47 (the DBuff overhead pushed it far above 32). The
  `MERGE_SIBLINGS_REQUIRED={OP1,OP5,OP6a}` tag is CONFIRMED necessary for OP5 (measured, not
  assumed). `transpile_guard.assert_delivery_fits` passes with merge=22, and reports default=47>32
  as EXPECTED (not a failure) for the merge-required op.

---

## q_raw handling — DELIBERATE DEVIATION from DESIGN (q_raw ELIMINATED)

The Stage-2 requirement is that b3 uses RAW q (= `q_s·√K`, since `q_s = q·scale` upstream) in
`dk_intra = ds^T@q_raw` and the `dg_p` reduce `Σ_K(dq_p·q_raw)`. DESIGN §1 built a `qraw_gm` bridge
(a per-subblock vec `muls` → GM). **This kernel ELIMINATES q_raw entirely.** Because the matmul and
the reduce are linear in q, `√K` is pulled out as a scalar:
- `dk_intra = ds^T@(q_s·√K) = √K·(ds^T@q_s)` — the matmul uses `q_s` directly; `dk_intra` is scaled
  by `√K` in the vec assemble.
- `Σ_K(dq_p·q_raw) = √K·Σ_K(dq_p·q_s)` — the reduce uses `q_s`; the row-sum is scaled by `√K`.
`√K` is computed once at runtime (`sqrt(Kd)` in `consts_vf`, K dynamic). This removes the `qraw_gm`
GM bridge, its MTE3 store, AND the per-subblock qraw build loop (which was a real correctness bug —
see finding #6). Mathematically identical to the golden (which recovers `q_raw = q_s/scale`);
sim-verified bit-consistent for K=64.

---

## Load-bearing A5 findings (VERIFIED; reuse for OP6a / OP6b) — a debugging-heavy op

1. **Cross-region vec-write → MTE3-store on the SAME tile is NOT ordered by the mutex boundary.**
   The ds mask writes `ub_ds` in the `cv_ds` consumer, but storing `ds_gm <<= ub_ds` in the NEXT
   region (`vc_dq` producer) read the RAW cube-moved tile — the vec mask writes never landed
   (`ds_gm == raw ds`, rel-L2 0.0). **Fix = store in the SAME region as the vec write** (auto_sync
   then orders V→MTE3). This is the SAME class as OP3's cross-region hazards; applies to EVERY
   vec-produce → GM-bridge in a fused kernel.
2. **A bridge-dependent cube LOAD must sit AFTER the VcMutex `wait()`, not between `ready()` and
   `wait()`.** `l1_do1 <<= do_gm` placed before `vc_cast.wait()` read do_gm before the cast synced →
   ds = NaN. Move the dependent GM→L1 load after `wait()` (OP4's dS_bridge does exactly this; the
   independent input loads may stay before `wait()`).
3. **A RUNTIME (Var) N-offset on a matmul RIGHT-operand row-slice is mis-read by the simulator, and
   a dynamic L0C column offset is REJECTED outright** (`l0c_to_ub ... offset must be static`). Tile
   the output axis with a STATIC unroll (`for k0 in (0, TK)`, k0 a Python literal) guarded by a
   runtime `if Kd > k0`. Note `for k0 in range(...)` inside `@kernel` has `name=` injected by the
   AST rewriter — use a tuple literal `(0, TK)`, not `range`/`builtins.range`.
4. **Multi-chunk `splitk` (V=128 → 2 K-chunks) is mis-lowered** (dw wrong on BOTH output halves with
   the right magnitude). Do NOT `splitk` the contraction here; load V FULL (which lands L0A/L0B at
   the 64 KB cap) and static-unroll the OUTPUT-K instead.
5. **Two static output-K subblocks sharing L0C / L1 / UB landing tiles corrupt EACH OTHER even with
   `depth=1` mutexes** (iter0 stored iter1's data; verified iter0-alone is bit-exact). **Fix = DBuff
   (2 slots) EVERY tile the two iterations both write:** the 5 L0C accumulators, the 5 UB landings,
   and the 2 L1 h/dh operands, indexed by a rotating `Var cnt`. `depth=1` on the mutex alone is
   necessary but NOT sufficient. This is the OP5-specific DBuff tax (see drift). A Python-int DBuff
   index does not rotate slots reliably — use a `Var cnt` (as OP4 does).
6. **A UB tile reused across two build iterations is clobbered** (the qraw build loop reusing a
   single `ub_qraw` corrupted `qraw_gm[:, 0:64]`). Eliminated by folding `√K` out of q entirely
   (finding above) — the cheapest fix, which ALSO removed a GM bridge + MTE3 store and eased the
   MTE3→V event budget (finding #7).
7. **The 8-flag intra event budget (`MTE3->V: 8/8`) is a REAL SIM wall for DBuff-heavy fused
   kernels.** Building qraw into extra DBuff slots blew it; eliminating qraw (finding #6) brought it
   back under. Watch this for OP6a/OP6b (more bridges + more DBuffs).
8. **ds_T without a vec-transpose primitive:** no A5/pypto vec transpose exists
   (`gm_to_ub_nd_dma_transpose` is not pypto-supported). ds_T is produced by a SECOND cube-1 matmul
   `v_new@do^T` (= transpose(do@v_new^T)) and masked in vec with the transposed factor
   `exp(decay_j-decay_i)` upper-incl-diag — mathematically `transpose(mask(ds))`, and it keeps
   `src_is_transpose into L0A = 0` (confirmed).
9. **matmul is `l1a @ l1b**T`** (l1a=[M,K] LEFT, l1b=[N,K] RIGHT; docstring `easyasc/shortcuts/
   matmul.py`). ds = `matmul(do, v_new)` (NO `.T` — v_new is already [N=C,K=V]); dq_intra =
   `matmul(ds, k.T)` (`.T` because k is [C,K]=[M-orient], needs [N=K,K=C]). Deriving each of the 6
   from the golden against THIS convention (not DESIGN §1's PyPTO-Pro sequence) was load-bearing.

## Deviations from DESIGN (realisation-level, math preserved) — reported for orchestrator judgement
- **q_raw eliminated** (folded `√K`) — no `qraw_gm` bridge. DESIGN built one; the fold is cheaper,
  correct, and removes a clobber + a store + event pressure.
- **ds_T via a 2nd cube matmul**, not a vec transpose (no such primitive exists on A5/pypto).
- **Output-K static-unrolled + all per-subblock tiles DBuff'd** — the DESIGN's single-pass tile
  table did not anticipate the DBuff tax; UB/mutex are correspondingly heavier (drift above).
All Module-0 topology, the 4 cube↔vec handoffs, SINGLE idiom, full-V-at-cap L0A/L0B, and the b3
math are honoured. No `transpile_gap`, no `ub_overflow` (fits 253,952 with merge).

## Handoff
Report to orchestrator: **sim PASS on all 12 (shape×seed), all 4 outputs**; transpile clean (0 gaps,
0 advisories, 1 `@pl.jit`, 12 matmul, nz 0, L0A-transpose 0, host no core compute); final 4-resource
under guard = **UB 238,432 (93.9%) / mutex 47 default·22 merge / flags 0,0 / L0A 64 KB / L0B 64 KB /
L0C 32 KB**; **L0A & L0B fit AS-IS at the 64 KB cap (no TV-subblock recovery)**; drift **UB ×1.153,
mutex ×1.343 (default)** — the SINGLE idiom rule is INVALIDATED by the output-K DBuff tax (3rd data
point); **merge REQUIRED (default 47>32, merge 22≤32) — OP5 does NOT shed merge**. **Boundary note:
UB (238 k > 215 k) and mutex_default (47 > 44) exceed the task's comfort gate; the op nonetheless
FITS every HARD delivery budget** — a heaviness finding, not a failure, with a concrete OP6a/OP6b
risk (they add matmuls to an already-at-cap L0A and an already-heavy UB). STOP after Stage 3.5 — do
not enter Stage 4 (a5 silicon) or any other operator.
