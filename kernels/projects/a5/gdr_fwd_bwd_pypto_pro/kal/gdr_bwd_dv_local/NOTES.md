# Stage 3.5 KAL — OP3 `gdr_bwd_dv_local` (parent module M6, golden_step b1)

Hardware-aware (`pypto-kernel-arch-lang` / `easyasc.a5`) pre-build of OP3, authored from the
STRUCTURE-ONLY probe `custom/gated_delta_rule_bwd/kal/probes/probe_op3_dv_local.py`, with the
real M6 `b1` arithmetic landed, validated in the Python simulator against the frozen CPU FP32
golden, and transpiled to PyPTO-Pro DSL. **Simulator pass ≠ Stage-4 acceptance** — the
`pypto-pro-op-verifier` on real a5 hardware is the only authority. No `test_` deliverable was
written; Stage 4 is NOT entered (needs a5 silicon).

Target pinned & asserted for every transpile: **arch=a5 (device_type=950), backend=pypto**
(via `transpile_guard.py`); limits in force UB=253,952, mutex=32, intra_flag=8, cross_flag=16.

## Deliverables (under `custom/gdr_bwd_dv_local/kal/`)
- `gdr_bwd_dv_local_kal_module1.py` — the `@kernel` source (+ `build_args`).
- `staged_module1.py` — backend transpile to PyPTO-Pro DSL (1 `@pl.jit`, 21,876 B).
- helpers (not deliverables, kept for reproducibility): `transpile_guard.py`, `measure_budget.py`,
  `emit_staged.py`, `sim_verify.py`.

---

## Simulator verification — PASS on all 4 §8 shapes × 3 seeds

`OpExec(..., simulator=True)` vs `gdr_bwd_dv_local_golden_cpu` at composition tolerance
atol=rtol=0.01. Reproduce: `python custom/gdr_bwd_dv_local/kal/sim_verify.py`

| case | shape (dv_local) | seeds 42/123/456 max\|Δ\| |
|---|---|---|
| aligned_t128_h4_k64_v64 (T=2C, K=V=64) | [1,4,2,64,64]  | 2.08e-4 / 2.37e-4 / 2.86e-4 |
| ttail_t100_h4_k64_v64 (T-tail valid_c=36) | [1,4,2,64,64]  | 1.68e-4 / 1.37e-4 / 2.96e-4 |
| k128v128_t192_h2 (K=V=128 full) | [1,2,3,64,128] | 5.17e-4 / 3.05e-4 / 3.43e-4 |
| tail_vlt_t200_h3_k128_v96 (N=4 chunks, valid_c=8, V<K) | [1,3,4,64,96] | 2.56e-4 / 2.21e-4 / 3.66e-4 |

Worst max\|Δ\| = **5.17e-4 << atol 0.01**. All outputs finite; no VF RAW-hazard / auto_sync
warnings in the sim run. Result token: `SIM_ALL_PASS`.

---

## Three-resource budget curve (C0 → C3), transpiled under the guard header

Reference = probe **116,992 B** (the deviation-gate anchor). Reproduce final:
`python custom/gdr_bwd_dv_local/kal/measure_budget.py a5`

| checkpoint | structure / state | ub_bytes | ub% | mutex (def/merge) | flags (intra/cross) | note |
|---|---|---:|---:|---|---|---|
| **C_probe** | probe seed: WRONG `[C,K]` q_T + stubbed arith | 116,992 | 46.1 | 19 / 18 | 0 / 0 | anchor |
| **C0** | corrected q_t `[K,C]` load + drop scale; probe SINGLE-block structure; arith still stubbed | **116,992** | 46.1 | 19 / 19 | 0 / 0 | **layout-correction drift ≈ 0 (byte-identical)** |
| C1/C2 | probe SINGLE-block, full-width [C,128] vec tiles, REAL b1 arithmetic + real GM chunk-load/tail | 160,256 | 63.1 | 18 / 18 | 0 / 0 | **structure is A5-INCORRECT (sim = all-NaN)** — see §"A5 sub-block" below; measured for the record, superseded |
| **C3 (final)** | **A5 DUAL/SPLITM restructure** (HC=32-row bands per vec sub-block), real arithmetic, GM complete, sim-verified, guard-asserted | **111,104** | **43.8** | **22 / 22** | **0 / 0** | delivered; `assert_delivery_fits` OK |

Transpile of C3 is **clean**: 0 gaps (no unmapped opcode `NotImplementedError`), no UB
`ValueError`, exactly 1 `@pl.jit`, `src_is_transpose into L0A = 0` (drift-gate item), Vec `_nz=0`
(every matmul operand GM→L1 nd2nz, Fix A). `merge_siblings` NOT required (default mutex 22 ≤ 32).

### Deviation gate (reference 116,992 B)
Final **111,104 B = −5.0%** → inside the `≤ +15%` band → **PROCEED, attribution recorded.**
(Not a "under-253,952" pass — it is genuinely below the anchor.)

---

## DRIFT FACTOR (mandatory)

- **UB drift = 111,104 / 116,992 = 0.950  (−5.0%)**
- **mutex drift = 22 / 19 = 1.158  (+15.8%)**

### Layout-correction component vs real-arithmetic/structure component (the split that generalises)

- **Layout correction (q_t `[C,K]`→`[K,C]`, scale drop): ≈ 0 B, 0 mutex, 0 flag.** Proven at C0
  (116,992, byte-identical to the probe): the corrected q_t tile `[KMAX,C]=[128,64]` fp32 = 32,768 B
  is byte-identical to the probe's wrong `[C,KMAX]=[64,128]` tile; only the logical shape and the
  matmul operand orientation change, never the footprint. **This component does NOT generalise** to
  OP4–OP6b (it is a probe-seed defect specific to OP3's q_t).
- **Real-arithmetic + A5-structure component (the part that DOES generalise):**
  - *Arithmetic UB:* ~0. The b1 gate (`exp(decay_j−decay_i)` + upper-tri mask) is **register-
    resident** (MaskReg + `select`, no `[C,C]` scratch); decay cumsum adds only `ub_g`+`ub_decay`
    = 2×[1,C] = 512 B. So the real arithmetic is essentially UB-free — the DESIGN §3 "MaskReg not
    [C,C] scratch" ruling holds.
  - *A5 SPLITM structure:* the dominant, generalisable finding. A5 has **two vector sub-blocks**;
    a `[C,C]`/`[C,V]` cube→vec accumulator must be split DUAL/SPLITM into **HC=C/2-row bands** (one
    per sub-block). This **halves the vec-band tiles** (`ub_al`,`ub_go_*`,`ub_dv` become HC-row),
    which is why the final UB fell BELOW the probe's full-tile SINGLE estimate. It **adds mutex**:
    the DBuff SPLITM L0C/UB pairs + a `bar_all()` per cube↔vec band-assembly push distinct mutex
    ids 19 → 22.

### Calibration guidance for the other probe-seeded operators (OP4/OP5/OP6a/OP6b)
1. The probe **UB** numbers are a good estimate of the delivered UB **for this op** (within ±5%),
   but for a structural reason, not because the model is right: the probe's full-tile SINGLE model
   over-estimates vec UB by ~2×, and A5 SPLITM HC-banding under-uses it by ~½ — the two roughly
   cancel. Do NOT assume this cancellation transfers verbatim; re-measure per op.
2. The probe **mutex** numbers UNDER-count by ~**×1.16**: every A5 cube↔vec fusion handoff that the
   probe modelled as a single SINGLE move actually needs a DBuff SPLITM pair (+ possibly `bar_all`),
   so scale probe mutex up ~1.16× before comparing to the 32-hard-limit / merge gate.
3. The **arithmetic itself is UB-cheap** when kept register-resident (MaskReg masks, register
   cumsum) — budget headroom for OP4–6b should assume the STRUCTURE, not the math, dominates.

---

## A5 sub-block reality — why the probe structure is not deliverable (correctness journey)

The probe is structure-faithful for the *task grid and tile set* but its cube↔vec handoff idiom is
**A5-incorrect and was never sim-verified**. Getting a correct kernel required six fixes; each is a
reusable lesson for OP4–6b:

1. **cube-A matmul (kept):** `matmul(l0c_al, l1_k[C,K], l1_qt[K,C].T)` = k@q_t with q_t loaded
   non-transposed `[K,C]`; `.T` is a cheap L0B (NZ2ZN) right transpose, no L0A transpose. Verified
   in isolation (maxd 0.0).
2. **`.subblk(0)` → SPLITM:** A5's `<<=` L0C→UB defaults to `dual_mode=SPLITM` (M doubled across the
   two sub-blocks). The probe's SINGLE `.subblk(0)` reads a cube tensor and, being cube-side, may
   **not** sit inside a vec `if GetSubBlockIdx()==0` guard (`split_instructions` rejects a cube-only
   slice in a vec `start_if`). Correct A5 pattern (= `kernels/a5/attention/flash_attn_full_fp8_causal.py`):
   DUAL/SPLITM with DBuff L0C/UB pairs, `HC=C/2`-row bands, unguarded moves, row offset
   `row_off = GetSubBlockIdx()*HC`.
3. **Mutex ORDER (the all-NaN root):** the L0C→UB move is cube-side (FIX pipe) and must live on the
   **producer** side, inside `cv.lock()…cv.ready()`; the vec compute/store consumes after `wait()`.
   The probe placed the move after `wait()` → move read poison → all-NaN.
4. **cube-B right operand:** slice do to VALID width **before** `.T` — `l1_goB[0:C, 0:Vd].T`, not
   `[0:VMAX].T`. A full-width `.T` transposes the poison columns `[Vd:VMAX]` into the contraction
   (all-NaN dv). Isolated test: `[0:VMAX].T` → maxd 42.9; `[0:Vd].T` → maxd 0.0.
5. **dv SPLITM move width:** move only `[0:Vd]` (`ub_dv[0:HC,0:Vd] <<= l0c_dv[0:C,0:Vd]`); a VMAX-wide
   SPLITM decode spread poison into the valid V columns.
6. **bf16→fp32 cast (the col-32 boundary):** a plain `reg <<= ub_bf16[...]` reads only half the lanes.
   Use the parent's proven idiom — `ub_to_reg_unpack` + `cast(dst, src, CastConfig(RegLayout.ZERO))`
   + `reg_to_ub_normal`, chunked at **SUB=64** (the fp32 VF-register lane width; VMAX=128 needs 2
   chunks). This flipped case-1 from all/partial-NaN to maxd 1.5e-4.

DESIGN §4's "sub-block-0 guard" intent is realised on A5 not as a single-block guard but as the
SPLITM banding above (each sub-block owns HC rows) — the mask uses the absolute row `row_off+r`.
DESIGN §6 on-chip cube↔vec (no GM bridge) is honoured (l0c→ub / ub→l1, `auto_mutex`, no hand SEvents).

---

## Tile attribution (final UB 111,104 B / 43.8%)

Per-position peak the sim reported: **Vec(UB) ≈ 96.5 KB** live (ub_al DBuff 16K, ub_g/ub_decay 0.25K
each, ub_go_bf 16K, ub_go_f 32K, ub_dv DBuff 32K), **Mat(L1) ≈ 112 KB** (l1_k 32K, l1_qt 32K,
l1_alB 16K, l1_goB 32K). The transpiler's reported `ub_bytes` (111,104) is its peak-reservation
figure and is the number the deviation gate uses (consistent with the probe's 116,992). No single
tile dominates the drift; the −5% vs probe is the SPLITM HC-band halving of `ub_al`/`ub_dv` net of
the wider `ub_go_*` (VMAX=128) tiles.

## Handoff
Report to orchestrator: sim PASS on all 12 (shape×seed) cases; transpile clean (0 gaps, 1 `@pl.jit`,
host no core compute); final 3-resource under guard = **UB 111,104 (43.8%) / mutex 22 / flags 0,0**;
drift 0.950 (UB) & 1.158 (mutex), layout-component ≈0. Boundary: none (no `transpile_gap` /
`ub_overflow`). STOP after Stage 3.5 — do not enter Stage 4 (a5 silicon) or any other operator.
