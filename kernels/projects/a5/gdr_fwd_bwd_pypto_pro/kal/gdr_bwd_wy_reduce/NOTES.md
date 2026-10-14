# Stage 3.5 KAL NOTES — gdr_bwd_wy_reduce (OP6b)

Pure-VECTOR half of module M9 (golden_step b4 reductions + b4b_dk) of
`gated_delta_rule_bwd`. BHN-parallel, stateless, single Vector Section, no cube /
no matmul / no L0A/L0B / no cross_core. `is_fusion=false` → single Module (L0),
suffix `1`.

Authored against the easyasc **a5** facade; validated in the Python **OpExec
simulator** against `../gdr_bwd_wy_reduce_golden_cpu.py`; transpiled to PyPTO-Pro
DSL by the backend (`staged_module1.py`). Sim PASS ≠ Stage-4 acceptance (needs a5
real HW; the verifier is the sole judge).

## Deliverables
- `gdr_bwd_wy_reduce_kal_module1.py` — the `@kernel(mode="vec")` source + OpExec sim self-check.
- `staged_module1.py` — backend PyPTO-Pro DSL (exactly one `@pl.jit(auto_mutex=True)`).
- `measure_budget.py`, `_transpile_dump.py`, `transpile_guard.py` — measurement/dump tooling.

## Reproduce
```bash
export PYTHONPATH="/workspace/pypto-gym/cannbot-skills/ops/pypto-kernel-arch-lang/payload:$PYTHONPATH"
cd custom/gdr_bwd_wy_reduce/kal
python gdr_bwd_wy_reduce_kal_module1.py                 # sim vs golden, 4 shapes x 3 seeds
python measure_budget.py gdr_bwd_wy_reduce_kal_module1.py gdr_bwd_wy_reduce_kernel a5
python _transpile_dump.py                               # regenerate staged_module1.py
```

## Sim verdict — PASS (all 4 outputs, 4 §8 shapes × seeds 42/123/456)
Guard header on every measurement:
`arch=a5 (device_type=950) backend=pypto || UB=253952 mutex=32 intra_flag=8 cross_flag=16`.

Oracle = `gdr_bwd_wy_reduce_golden_cpu` on identical `_mk_contract_inputs` (direct
chunk-domain randoms + primary v). Acceptance = per-output mixed
`allclose(1e-2,1e-2)` OR rel-L2 ≤ 2e-2.

| shape | dk |Δ| | dv |Δ| | dbeta_wy |Δ| | dg2 |Δ| |
|---|---|---|---|---|
| aligned_t128_h4_k64_v64 | 0.0 | 0.0 | ≤7.6e-6 | ≤9.5e-7 |
| ttail_t100_h4_k64_v64 | 0.0 | 0.0 | ≤7.6e-6 | ≤9.5e-7 |
| t3c_t192_h2_k128_v128 | 0.0 | 0.0 | ≤3.1e-5 | ≤3.1e-5 |
| tail_vlt_t200_h3_k128_v96 | 0.0 | 0.0 | ≤3.1e-5 | ≤3.1e-5 |

`dk`/`dv` are **bit-exact** (pure elementwise, no reduction reorder). `dbeta_wy`/
`dg2` carry ~1e-5 abs diff at magnitudes ~1e4 (rel ~1e-9) — fp32 reduction-order
only, far inside tolerance. A scale-ledger violation would surface here as a
per-output mismatch; none did.

## Transpile — CLEAN
No unmapped opcode (`NotImplementedError`), no UB overflow (`ValueError`), no
`EventBudgetExceeded`. `staged_module1.py`: **exactly one** `@pl.jit`, two
`@pl.vector_function` (passk/passv). Opcode mapping observed:
`cadd → vf.reduce_sum`, `cast → vf.astype`, `update_mask → vf.update_mask`,
`vf_barrier(STORE,LOAD) → vf.mem_bar(VST_VLD)`, `mul/add/sub → vf.mul/add/sub`,
`.single()/.single_value() → vf.load_align(BRC_B32)/store_align`.

## Budget curve C0..C3 (under the guard header; L0A/L0B = N/A, no cube)

| ckpt | what | UB bytes | UB % | mutex (default / merge) | intra | cross |
|---|---|---|---|---|---|---|
| **C0** | tile+section structure, math stubbed, **fused** single-pass | 247,296 | **97.4%** | 12 / 6 | 0 | 0 |
| **C1** | 13 vector steps landed, **two-pass** ordering | 181,760 | 71.6% | 10 / 5 | 0 | 0 |
| **C2** | 4 GM outputs materialised (dk,dv,dbeta_wy,dg2) | 181,760 | 71.6% | 10 / 5 | 0 | 0 |
| **C3** | final, `vf_barrier` added, guard-asserted, 0 warnings | 181,760 | 71.6% | 10 / 5 | 0 | 0 |

UB is fixed by the tile declarations, so C1=C2=C3 (the tile set is decided at C1).

### The fused-vs-two-pass UB outcome (the load-bearing decision)
- C0 **fused** per-c layout (all inputs live at once, dk/dv written in-place over
  the dkbg/dvb tiles): **UB 97.4%** — fits ≤100% but exceeds the brief's 90% trip
  wire.
- Adopted the DESIGN §3 **two-pass** ordering: Pass-K (dkbg,dkb,dkt,k,dk_p →
  dk,dg2,dbeta_partial) then Pass-V (dvb,v → v_c,dv,db2,dbeta), where **Pass-V
  reuses the Pass-K tile addresses** (`t_dvb` aliases the dkbg tile, `t_vc`
  aliases the dkb tile — same Tensor object = one address). Peak UB collapses to
  the per-pass max: 5×[64,128]fp32 (163,840) + 1×[64,128]bf16 v-staging (16,384)
  + col tiles (1,536) = **181,760 = 71.6%**. Matches DESIGN's ~78% two-pass
  estimate (×0.92 of the 198 KB upper end); comfortably under cap.

### Merge verdict — merge_siblings NOT required (measured)
Default-flag mutex = **10 ≤ 32**. `merge_siblings=True` lowers it to 5, but the
default already fits, so **no merge flag is needed for delivery** — matching the
DESIGN prediction ("merge likely NOT needed — measure"). OP6b is not on the
standing `MERGE_SIBLINGS_REQUIRED` list, and does not need to be.

### Drift data points (measured / probe est)
Probe est (CAMPAIGN_BRIEF §6 card / SPEC kernel-contract): ~130–160 KB / ~8–14
mutex / 0·0 flags (an `est-from-split` **lower bound**, v_c prelude uncosted).
- **UB drift**: C3 = 177.5 KB vs probe est 130–160 KB → **×1.11–1.37** (mid ≈
  ×1.22). The probe was a lower bound (v_c bf16 staging + the 5th K tile were not
  in the split estimate); the measured figure is the honest two-pass peak.
- **mutex drift**: C3 default = 10 vs probe est 8–14 → **in-range** (≈ mid). 0
  intra / 0 cross flags exactly as est.

## Scale-ledger subtleties actually encoded in the kernel
Implemented as a **structural single-apply**: per-token constants (β[c],
decay_exp[c]) are folded OUT of every reduction and applied once to the lane-0
scalar (Σ_K(x·c)=c·Σ_K(x)). Concretely:
- `inner_dg=Σ_K(dkbg·k)` and `db3=Σ_K(dkb·k)` are each reduced **once**; `db3` is
  reused by BOTH `dbeta_wy` (raw) and `dg2` (×β). `db1=decay·inner_dg`;
  `dg2_loop1=β·db1` (β once; decay carried by db1, **not** re-multiplied → guards
  DESIGN traps T1/T4).
- `dkb`/`dkt` are consumed **without any decay/pairwise re-gate** (already
  decay-gated in OP6a glue-2 → trap T2): `db3=Σ(dkb·k)`, `dk2:+dkt`,
  `kdkt=Σ(k·dkt)`, `dk2:dkb·β` — decay appears 0× on all of them.
- `db2=Σ_V(dvb·v_c)` uses **raw v_c** (primary v self-served), no β, no decay →
  trap T3. `dbeta_wy=db1+db2+db3` carries **zero β**.
- β on exactly 5 paths, decay on exactly 3, scale=1/√K on 0 — verified by the
  golden's algebraic-identity checks (bit-exact dk/dv; ≤3e-5 dbeta/dg2).

## v_c self-serve (S0) as built
Primary `v[B,T,H,V]` bf16 is loaded with the **transpose absorbed into the GM
index** — `t_vbf[0:vc,0:Vd] <<= v[b, t0:t0+vc, h, 0:Vd]` gives natural row-stride
H·V, no host movedim. `vf.astype` widens bf16→fp32 (the ONLY cast). T-tail: only
`vc=min(C,T-t0)` rows are cast; pad rows [vc:C] of `v_c` are **zeroed** so
`db2` on pad rows is 0 (matches the golden's `F.pad`; upstream `dvb` pad rows are
random in the test but multiply the zeroed `v_c` → 0, and `0×0` avoids any
NaN-from-stale). Outputs are chunk-domain — **no `[:T]` slice** (that is OP7's b7).

## VF hazard fix (not silenced)
The sim first emitted 102 `VfBarrierWarning: VF RAW hazard: store -> load on
overlapping local buffer` — the Pass-V `v_c` STORE feeding the `db2` LOAD (and the
Pass-K `dbeta_partial` STORE feeding the Pass-V LOAD). Fixed with a single
`vf_barrier(VfPipe.STORE, VfPipe.LOAD)` at the Pass-V phase boundary (after the
v_c cast/pad, before db2). Re-run: **0 warnings**, still ALL PASS. Consistent with
the delivered OP6a modules, which use `vf_barrier` for the same round-trip.

## New A5/KAL findings for OP7
- **a5 whole-register reduce is `cadd` (Reg→lane0), NOT a tile op** — it transpiles
  to `vf.reduce_sum`. Row reduction of a free axis >64 = per-64-lane `mul`+`cadd`
  then `add` the lane-0 partials (rule 15 "two 64-wide halves"), all register-level
  inside `@vf`. K∈{64,128} is always full 64-lane registers; only V=96 needs a
  partial-lane `update_mask` — `Var(valid, dtype=DT.uint32)` feeds `update_mask`.
- **Two-pass tile-address reuse = alias the same `Tensor` object** across passes
  (`t_dvb = t_dkbg`). The allocator counts one address; peak UB = per-pass max.
  This is the cheap lever when a fused vec layout is >90% but each phase's working
  set is small — likely applicable to OP7's per-(b,h) finalize.
- **Any UB store→load round-trip inside a `@vf` needs an explicit
  `vf_barrier(VfPipe.STORE, VfPipe.LOAD)`** (VF store buffer is its own ordering
  domain; auto_sync/auto_mutex do NOT cover it). Watch OP7's `dg` reverse-chunk
  cumsum, which is a recurrence over N.
- No cube ⇒ no L0A/L0B pressure and no cross-core sync; the DESIGN's "UB is the
  only watch" held — the operator cleared with headroom (71.6% UB, 10 mutex). This
  was the easy one and it cleared, exactly as the gate predicted.
