# NOTES — gdr_bwd_finalize (OP7 / M10) — Stage 3.5 KAL pre-build (Module 1, L0)

Single-module, **pure-vector**, L0 (`is_fusion=false`, `module_count=1`). Authored against the
easyasc **a5** facade, sim-verified against the frozen CPU golden, transpiled to PyPTO-Pro DSL under
the pinned `(arch=a5, backend=pypto)` guard. **Sim PASS ≠ Stage-4 acceptance** (真机 a5 needed;
this is a pre-build, not a delivery). This operator's PASS ≠ delivery of `gated_delta_rule_bwd`
(needs all 8 components + the §9 composition validation vs torch autograd).

## Artifacts on disk
- `gdr_bwd_finalize_kal_module1.py` — the `@kernel` source + 3 `@vf` functions + `run_finalize`
  OpExec wrapper + `__main__` sim self-check.
- `staged_module1.py` — backend-transpiled PyPTO-Pro DSL (delivered form = **merge_siblings=False**;
  see merge verdict). Exactly one `@pl.jit`, one `pl.section_vector()`, no `section_cube`/`matmul`.
- `transpile_guard.py` (copied from parent), `measure_budget.py`, `_transpile_dump.py` — tooling.

## Contract realized (frozen SPEC order)
`gdr_bwd_finalize(grad_o, dq_p, dk_in, dv_in, dg_p, dg2, dbeta_wy, sig, beta_act) ->
(dq, dk, dv, dg, dbeta)`. Per DESIGN deviations (inherited): the kernel does **not** load `grad_o`
(wrapper reads only `grad_o.shape[1]=T`) nor `beta_act` (≡ sig, unused). The `@kernel` takes the 7
used DATA inputs + 5 outputs + scalars `(B,T,H,Kd,Vd,N,Cc)`. Outputs: dq/dk/dv `[B,T,H,K|V]` bf16,
dg/dbeta `[B,T,H]` fp32. BHN chunk-parallel grid (`for tid in B*H*N`, GetVecIdx-strided).

Steps: b4b_dg `dg_sum=dg_p+dg2`; b5 pure-vector reverse-cumsum `dg_full[r]=Σ_{c≥r}dg_sum[c]`
(== `triu_incl@dg_sum`, NO scale, reads full C rows incl tail-pad); b6 `dbeta=dbeta_wy·sig·(1−sig)`
(sig only); b7 cast dq_p/dk_in/dv_in fp32→bf16 (rule 6: `cast` no-mask + `reg_to_ub_downsample`) +
reindexed store (token t=n·C+r → `[b,t,h,:]`), tail pad rows dropped by storing only `vc=min(C,T−t0)`
rows (poison-immune); dg/dbeta stored fp32.

## Simulation (Exit criterion 1) — 12/12 PASS
Command (reproducible):
```
export PYTHONPATH="/root/.claude/skills/pypto-kernel-arch-lang/payload:$PYTHONPATH"
python3 custom/gdr_bwd_finalize/kal/gdr_bwd_finalize_kal_module1.py
```
Oracle = `gdr_bwd_finalize_golden_cpu.py` (fp32, `_mk_contract_inputs`). `OpExec(..., simulator=True)`,
SimulatorConfig(core_count=32). SPEC §8 four shapes × seeds 42/123/456 = **12/12 PASS**, per-output
peak-scaled mixed criterion `allclose(1e-2,1e-2) OR peak-scaled≤1e-2 OR rel-L2≤2e-2`:

| shape (B,T,H,K,V) | N | worst dq/dk/dv rpk | worst dg rpk | worst dbeta rpk |
|---|---|---|---|---|
| aligned [1,128,4,64,64]   | 2 | 3.6e-3 | 1.9e-7 | 9.5e-8 |
| ttail [1,100,4,64,64]     | 2 | 3.6e-3 | 2.0e-7 | 8.9e-8 |
| t3c [1,192,2,128,128]     | 3 | 3.4e-3 | 2.8e-7 | 9.5e-8 |
| tail_vlt [1,200,3,128,96] | 4 | 3.4e-3 | 1.6e-7 | 9.6e-8 |

dq/dk/dv ≈ 2–4e-3 (bf16 store rounding, absorbed by the 1e-2 criterion); dg/dbeta ≈ 1e-7 (fp32,
essentially exact). Tail (T=100/200) and V<K (V=96<K=128) both covered; dg reduce reads full C rows,
dq/dk/dv pad rows dropped at store.

## Transpile (Exit criteria 2–4) — clean
Command: `python3 _transpile_dump.py` (writes staged_module1.py); budget via
`python3 measure_budget.py gdr_bwd_finalize_kal_module1.py gdr_bwd_finalize_kernel a5`.
No `NotImplementedError` (no unmapped opcode), no `ValueError` (no UB overflow). Exactly one
`@pl.jit`; host side does only shape/alloc/one dispatch (`run_finalize`). Guard header in force:
`arch=a5 (device_type=950) backend=pypto || UB=253952 mutex=32 intra_flag=8 cross_flag=16`.

### Budget curve (pure-vector → L0A/L0B N/A) and drift reconciliation
Because OP7 is a single-module L0 authored in one pass, the phased C0..C3 curve collapses to the
allocation baseline (C0) and the final arithmetic build (C3) — they coincide (all UB is the fixed
tile set, no arithmetic adds tiles):

| point | UB bytes | UB %cap | mutex(default) | mutex(merge) | intra-flags | cross-flags | L0A/L0B |
|---|---|---|---|---|---|---|---|
| C0 (alloc-only)   | 82,944 | 32.7% | — | — | 0 | 0 | N/A |
| C3 (final, deflt) | 82,944 | 32.7% | **7** (5.5% of 32) | — | 0 (0%) | 0 (0%) | N/A |
| C3 (final, merge) | 82,944 | 32.7% | — | **4** (12.5% of 32) | 0 | 0 | N/A |

- **UB 82,944 B (32.7%)** vs DESIGN projection **101,376 B (39.9%)** → **−18,432 B lower**. Drift cause:
  DESIGN §3 budgeted 8 double-buffered (depth-2) tile groups (16 KB in scalar slots + 2× on the wide
  tiles). The implementation uses **single-instance plain `Tensor` allocations + `with auto_sync()`**
  (see deviation D3) — no depth-2 doubling — so the wide fp32 `in_big`[64,128]=32 KB + fp32
  `col_src`[64,128]=32 KB + bf16 `out_big`[64,128]=16 KB + 4× fp32 `[1,64]` scalar tiles = 82,944 B.
  Comfortably under the DESIGN figure and near the probe's 77,824 B (30.6%); the +5,120 B over the
  probe is the extra aligned `col_src` wide tile forced by deviation D2.
- **mutex 7 default / 4 merged** vs the probe's **18 def · 11 merge**. Far below both — the plain-tensor
  + auto_sync dataflow is lean (only MTE2→V→MTE3 edges on a handful of reused tiles). Well within the
  32-mutex cap either way.
- **intra-flags 0, cross-flags 0** — matches the probe (0 flags); single Vector execution domain,
  no cross-core, no hand-written SEvents (CAMPAIGN rule 9).

### Merge verdict
OP7 is **NOT** in `MERGE_SIBLINGS_REQUIRED`. Measured both: default mutex **7** ≤ 32 (no merge
needed) and merged mutex **4**. Delivery does **not** require merge_siblings; the staged artifact on
disk is the **merge_siblings=False** form (guard `assert_delivery_fits('OP7', mutex_merge=4,
ub_bytes=82944, intra=0, cross=0, mutex_default=7)` → "fits delivery budget"). Merged form also has
exactly one `@pl.jit` if a future budget squeeze wants it.

## Deviations from DESIGN (declared with evidence; none change the math / SPEC §7 acceptance)
- **D1 — b5 reverse-cumsum written as `update_mask` total-minus-prefix, not `arange`+`ge` suffix
  mask.** DESIGN §1 prescribes `idx=arange; mask_r=ge(idx,r); reduce_sum`. The `ge` form needs a
  compile-time-constant threshold `float(r)` (an fp32-register-vs-runtime-Var `compare` is family-
  mismatched / board-limited), which forces a **Python unroll** of the r-loop; a Python unroll over a
  **shared result register** races the per-r store against the next r's reduce (measured scramble:
  `col[1]=2044` vs expected `2079`). The `update_mask` prefix form runs as a **DSL `range` loop** (one
  store instruction, in program order — the OP6b-proven scatter idiom) and is **mathematically
  identical**: `dg_full[r] = total − Σ_{c<r}dg_sum[c] = Σ_{c≥r}dg_sum[c] == triu_incl@dg_sum` (golden
  `_validate`[4]). Load-bearing invariant (pure-vector, no matmul, no scale, reads full C rows)
  preserved. Verified: sim dg rpk ≈ 1e-7.
- **D2 — dg/dbeta per-row scalars land in column 0 of an aligned wide `[C,MAX_K]` tile, not a `[C,1]`
  column tile.** DESIGN §2/§3 list `dg_col`/`dbeta_col` as `[64,1]`. A width-1 `[C,1]` tile
  mis-strides the single-element scatter in the simulator (allocator row-alignment corrupts/drops
  rows — probed: only row 0 correct, tail rows unwritten). Writing to column 0 of an **aligned**
  `[C,MAX_K]` tile and storing `dg[b,t0:t0+vc,h:h+1] <<= col_src[0:vc,0:1]` is poison-free (probed:
  all H columns exact, no cross-H bleed). This is a physical-tile realization detail; the store
  geometry (tile-row→T, col→H) and the reindex semantics are unchanged. Costs +32 KB UB (still 32.7%).
- **D3 — plain single-instance `Tensor` + `with auto_sync()`, not depth-2 `make_tile_group` +
  auto_mutex.** DESIGN §3 planned 8 double-buffered groups (16 mutex). The simpler single-buffer +
  auto_sync form is correctness-complete, transpiles cleanly, and measures **lower** on both UB and
  mutex (see drift). It trades some load/store↔compute pipeline overlap for a smaller footprint;
  Stage-4 may reintroduce double-buffering as a perf lever if profiling on a5 warrants it. No
  semantic change.

All three are implementation-layer choices under the DESIGN's frozen intent; none touches the 5
public outputs' values or the SPEC §7 acceptance criterion. (These are in addition to the three
DESIGN-level deviations already frozen in DESIGN §「设计偏差声明」: BHN grid + vector reverse-cumsum;
reindexed store not tile-transpose; grad_o/beta_act values unused.)

## Handoff
Return to orchestrator. Sim 12/12, transpile clean (1 `@pl.jit`, UB 32.7%, mutex 7/4, 0 flags),
staged + kernel on disk. NOT delivered — Stage 4 proper (a5 hardware) + the pypto-pro-op-verifier
真机 verdict remain the only acceptance authority.
