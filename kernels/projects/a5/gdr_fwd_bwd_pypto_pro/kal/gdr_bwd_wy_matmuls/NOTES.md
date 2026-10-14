# gdr_bwd_wy_matmuls (OP6a) — Stage 3.5 KAL verification record

CUBE half of module M9 (golden step `b4`): 8 fp32 matmuls in 3 cube sections split by
2 vector-glue masks, dA threaded cube-A → glue-1 → cube-B → glue-2 → cube-C. Authored
against the A5 facade (`easyasc.a5`), validated in the Python simulator vs
`gdr_bwd_wy_matmuls_golden_cpu.py`, transpiled to PyPTO-Pro DSL (`staged.py`).

**Sim pass ≠ Stage-4 acceptance.** This is a pre-build; the real-hardware verdict is the
Stage-4 verifier's alone. No `test_*` deliverable is produced here.

## Deliverables
- `gdr_bwd_wy_matmuls_kal.py` — the `@kernel` source (+ `build_args`)
- `staged.py` — backend-transpiled PyPTO-Pro DSL (one `@pl.jit`)
- `NOTES.md` — this file

## Exit criteria — ALL MET
1. **Sim correctness** — `python sim_verify.py` → `ALL SIM CHECKS PASS`.
   All 4 §8 shapes × seeds {42,123,456}, all 5 outputs (dkbg, dvb, dkb, dkt, d_a).
   Errors: K=64 shapes exact (0.00e+00); K=128 shapes ≤ 3.7e-07 rel-L2 (tol 2e-2).
   No inf/nan.
2. **Transpile clean** — `merge_siblings=True`: real gaps `[]`, no `NotImplementedError`,
   no UB/L0 overflow. (`measure.py` default+merge advisories both empty.)
3. `staged.py` on disk (443 lines).
4. Exactly **one `@pl.jit(auto_mutex=True)`**; all matmuls/loads/stores/vfs inside it;
   host side does zero arithmetic.
5. This record; sim + measure commands reproducible below.

## Six-resource curve (arch=a5 device=950, backend=pypto; worst case K=V=128, N=3)
Measured under the asserted `transpile_guard.print_header` header. The simulator does not
model NZ staging, so these transpile-budget numbers are the only budget truth.

| resource | used | cap | %cap | note |
|---|---|---|---|---|
| UB | 172032 B | 253952 B | 67.7% | |
| **L0A** | **65536 B** | 65536 B | **100.0%** | = 64 KB exactly — C0 hard gate ≤64 KB **PASS** (at limit, not over) |
| **L0B** | **65536 B** | 65536 B | **100.0%** | = 64 KB exactly — C0 hard gate ≤64 KB **PASS** |
| L1 | 311296 B | 524288 B | 59.4% | |
| L0C | 114688 B | 131072 B | 87.5% | <90% |
| mutex (default) | 32 | 32 | 100.0% | at limit → merge REQUIRED |
| mutex (merge) | 19 | 32 | 59.4% | delivery config |
| intra_flag | 0 | 8 | 0% | all-auto_mutex, no hand SEvent |
| cross_flag | 12 | 16 | 75% | within budget |

**C0 hard gate** (measured L0A≤64 KB AND L0B≤64 KB): both = 64 KB exactly → PASS (fits,
does not exceed; not a >100% STOP). Every per-mmad L0A/L0B operand is a fp32 `[64,64]`
tile = 16 KB; the group sizing sums to 64 KB. The peak sits at the cap with zero slack —
any wider on-chip tile would overflow, so the `[C,C]=64` literal tiling is load-bearing.

**Drift (4th data point).** DESIGN §2/§3 project L0A/L0B ≈ 32 KB (50%); measured 64 KB
(100%) — a **×2.0** on-chip drift vs the design projection (worse than the predicted
×1.15–1.20 for UB-tiled ops; the driver here is fp32 `[64,64]` operand groups, not UB).
UB 172032 B is comfortably under cap. Mutex default 32 vs the design's ×1.30–1.35 mutex
prediction: default hits exactly the 32 cap; merge collapses it to 19.

**Merge verdict: merge_siblings=True REQUIRED and sufficient.** Default-flag mutex count
= 32 (at the hard limit); merge = 19. OP6a is (correctly) in
`transpile_guard.MERGE_SIBLINGS_REQUIRED`. Delivery config = merge.

**Cross-flag: 12 / 16** (estimate was 5–10; measured 12 — the extra cube↔vec crossings
from the internal-bridge mutex pair push it to 12, still ≤16).

## The cube-B "inf" root cause + fix
The carried lead ("cube-B's matmul chain produces d_a=inf; suspect an uninitialized read
through the `dAmid_gm` GM bridge") was the visible symptom of **four** independent defects,
each found and fixed. In the simulator, non-finite values are 0xFF-**poison** (uninitialized
`output`/`l1`/`l0*`/`ub`/`workspace` memory is bit-filled 0xFF → NaN); the sim is otherwise
deterministic, so any non-determinism is a missing-sync tell.

1. **GM bridges are never synced.** `easyasc/targets/ascendc/asc_autosync.py:250` —
   *"GM tensors are not Tensor instances, so they never produce a sync key and stay out of
   the dependence map."* The prior `dA_gm`/`dAmid_gm` GM hand-offs (cube↔vec AND the cube-B
   internal L0C→GM→L1 self-bridge) therefore raced. **Fix:** replace GM hand-offs with the
   canonical **shared-L1** pattern — the vec writes the masked dA straight into `l1_dAb`
   (`ub_to_l1_nd2nz`), guarded by the `vc_b` mutex; the cube reads that L1 tile. L1 is a
   tracked Tensor, so mutex + auto_sync serialize it. (`dA_gm`/`kbeta_gm` remain only where
   a GM round-trip is unavoidable for the `dn2nz` transpose of the final dA, and are
   likewise mutex-gated + single-sublane stored.)
2. **`l0c_to_l1` is a hard transpile gap.** No pypto handler
   (`transpiler.py:9017` generic `else` → `gaps.append("kernel:l0c_to_l1")`, op silently
   dropped → wrong DSL). The DESIGN's mandated on-chip L0C→L1 internal bridge cannot be
   delivered. **Fix:** route the b4.5→b4.6 bridge **L0C→UB→L1 across the vec lane**
   (`l0c_to_ub` cube-FIX → `ub_bridge` → vec `ub_to_l1_nd2nz` → `l1_m16`), synced by a
   dedicated `cv_mid`/`vc_mid` mutex pair. Both ops have transpiler handlers → gaps `[]`.
3. **TWO vec sublanes tear shared writes** (THE dominant non-determinism source).
   `sync.py:39` — `wait_vec` blocks until *all* vec lanes signal; both sublanes execute
   every `@vf` and vec datamove redundantly. When both DMA the same shared-L1 tile or GM
   output, the NZ write tears → poison, non-deterministically. **Fix:** guard every vec
   write of a shared L1 tile (`l1_dAb`, `l1_m16`) or a GM output (`d_a`, `dA_gm`,
   `kbeta_gm`) to a single sublane `if GetSubBlockIdx() < 1:`. (Idempotent UB writes inside
   the vfs are left redundant.)
4. **cube-operand load ordering + Reg width.**
   - Cube consumers must load their own operands **after** the handoff `.wait()` (loading
     `l1_attn`/`l1_kc`/`l1_kbe` before the wait races). `l1_kbe`←`kbeta_gm` in particular
     is a vec-produced value and MUST be gated by `vc_c.wait()`.
   - A vector Reg holds only **C=64 fp32 lanes**; `form_kbeta` originally loaded
     `k[i,0:KMAX=128]` into one Reg, silently dropping lanes 64:128 → `k_beta[:,64:128]`
     poison. Only visible at K=128 (K=64 never stores/uses those lanes). **Fix:** process K
     in two 64-wide halves.

## New A5/KAL findings for OP6b / OP7 (reuse these)
- **GM is invisible to `auto_sync`.** Never hand a value cube↔vec (or cube↔cube) through a
  GM scratch and expect ordering. Use a **shared-L1 tile** written by the vec
  (`ub_to_l1_nd2nz`) + the cube↔vec mutex. Reserve GM only for a `dn2nz` transpose source,
  and still gate it with a mutex + single-sublane store.
- **`l0c_to_l1` does not transpile.** For an L0C→L1 bridge, bounce L0C→UB→L1 across the vec
  lane with an extra Cv/Vc mutex pair (both `l0c_to_ub` and `ub_to_l1_nd2nz` transpile).
- **Sublane discipline is mandatory.** Every vec write of a shared/output region must be
  single-sublane guarded (`GetSubBlockIdx() < 1`) or split by sublane rows; otherwise torn
  writes appear as flaky NaN. This alone explained ~all of the non-determinism.
- **Load cube operands after the mutex `.wait()`**, and never load a vec-produced GM value
  before its gating wait.
- **Reg = 64 fp32 lanes.** Any per-row op over K≤128 must be two 64-wide halves.
- Budget headline for this op family: fp32 `[64,64]` matmul operands pin **L0A/L0B at 64 KB
  (100%)** — zero slack; keep on-chip tiles at the `C=64` literal. mutex needs merge.

## Reproduce
```bash
export PYTHONPATH="/workspace/pypto-gym/cannbot-skills/ops/pypto-kernel-arch-lang/payload:$PYTHONPATH"
cd custom/gdr_bwd_wy_matmuls/kal
python sim_verify.py          # -> ALL SIM CHECKS PASS  (4 shapes x 3 seeds x 5 outputs)
# six-resource curve + gap check: measure.measure(gdr...build_args) under transpile_guard header
# staged.py regenerated via transpiler.transpile(kernel, used_micros, name, merge_siblings=True)
```

## Scope note
Did NOT proceed to OP6b or Stage 4. Sim pass is not an acceptance certificate; the Stage-4
`pypto-pro-op-verifier` real-hardware verdict is authoritative.
