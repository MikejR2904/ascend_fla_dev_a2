# NOTES — gdr_fwd_train_wy Module 1234 (KAL Stage 3.5 RE-HOME)

> **Measurement provenance:** all transpile budget numbers below are taken under
> **(arch=a5, backend=pypto)** — a5 facade (`ub_cap=256` → UB limit 253,952) +
> `globvars.target="pypto"`, pinned and asserted by `kal/transpile_guard.py`
> (copied from the seed), which prints the mandatory header
> `=== TRANSPILE TARGET arch=a5 (device_type=950) backend=pypto || limits in
> force: UB=253952 mutex=32 intra_flag=8 cross_flag=16 ===`. `merge_siblings=True`
> is the delivery configuration. No env var (incl. `PYPTOPRO_JIT_ARCH`) affects
> this.

This is a **RE-HOMING, not fresh authoring.** `gdr_fwd_train_wy` (OP-A of the
gated_delta_rule training-forward pair, the WY-build front-end M1..M4) is a
verbatim re-home of the banked, sim-verified seed
`custom/gated_delta_rule_bwd/kal/gated_delta_rule_bwd_kal_module1234.py`
(127,744 B / 28-17 mutex / 0 flags, sim 12/12). Only which tensors survive the
call changed; the M1..M4 mathematics is byte-for-byte the seed's.

## Seed choice + equivalence

**Seed = the bwd-side `staged_module1234.py` recompute path** (the SPEC's
preferred candidate: 127,744 B vs the fwd-side seed's 226,304 B). The SPEC left
the choice `TBD at Stage 3.5`, contingent on proving the bwd seed's M1..M4
reproduces the OP-A golden. **This session confirms it:** the re-homed kernel
(whose M1..M4 compute path is the seed's, unchanged) matches
`gdr_fwd_train_wy_golden_cpu.py` on all 4 §8 shapes × 3 seeds, all 17 outputs,
elementwise (see below). The provisional seed-equivalence choice is therefore
**CONFIRMED**; the fwd-side fallback was not needed.

## Simulation verification (all 17 outputs)

Canonical reproducible self-check (from repo root `/workspace/pypto-gym`), the
`__main__` of the deliverable, run **sequentially** (single foreground process,
`SIM_CORE_COUNT=32`):

```
export PYTHONPATH="/root/.claude/skills/pypto-kernel-arch-lang/payload:${PYTHONPATH:-}"
python custom/gdr_fwd_train_wy/kal/gdr_fwd_train_wy_kal_module1234.py
```

Oracle (imported, NOT re-implemented): the Stage-2 CPU FP32 golden
`gdr_fwd_train_wy_golden_cpu.py`, `gdr_fwd_train_wy_golden_cpu(q,k,v,g,beta) ->
17 fp32 tensors` (SPEC output order). Inputs generated identically for kernel and
golden (`_mk_case`: q/k/v/beta bf16 randn, g uniform(-16,0) fp32 — mirrors the
golden's own generator). Mixed criterion `allclose(rtol=1e-2, atol=1e-2)` OR
peak-scaled relative L2 ≤ `2e-2`.

| shape [B,T,H,K,V] | scenario | result (worst of 3 seeds) |
|---|---|---|
| [1,128,4,64,64]  | aligned (T=2C, K=V=64, 1 v-tile) | PASS max_abs 1.0e-4 – 4.2e-4 |
| [1,100,4,64,64]  | T tail (last chunk 36 rows)      | PASS max_abs 9.7e-5 – 3.1e-4 |
| [1,192,2,128,128]| T=3C, K=V=128 (2 k/v sub-tiles)  | PASS max_abs 1.2e-4 – 5.2e-4 |
| [1,200,3,128,96] | T tail, K=128, V=96 (partial V)  | PASS max_abs 2.5e-4 – 4.0e-4 |

All 12 (4 shapes × 3 seeds): **`ALL PASS`**, `rel_l2 = 0` on every case
(elementwise allclose satisfied; the peak-L2 fallback was never needed), every
output finite. The 4 re-home-new outputs (`v_b`, `beta_act`, `sig`, `q_t`) match
the golden with `max_abs = 0.0` (they are direct copies / a transpose of exact
vector-computed quantities). Only advisory: the benign `MicroDivPrecisionWarning`
(A5 1-ULP `MicroAPI::Div` in M1's sigmoid), also present in the seed.

## C0..C3 transpile curve (reference = 127,744 B, `merge_siblings=True`)

Reproducible: `python tmp/kal_fwd_wy/measure_curve.py` (probe K=128,V=128,N=2 →
2 kt/2 vt sub-block loops, both matmul paths). C1/C2 are physical line-strips of
the deliverable (KAL forbids compile-time-constant `if` in a kernel body, so the
additions cannot be toggled by a gate; they are stripped textually instead).

| stage | what it adds | outputs | mutex (default / merge) | ub_bytes (% of 253,952) | peak flags | advisories |
|---|---|---|---|---|---|---|
| **C0** seed (reference) | — | 14 | 28 / 17 | **127,744** (50.3%) | 0 | 0 |
| **C1** +v_b valid store  | relabel 11 workspace → declared outputs | 15 | 28 / 17 | 127,744 (50.3%) | 0 | 0 |
| **C2** +beta_act/sig      | 2 store-side additions | 17* | 28 / **16** | 127,744 (50.3%) | 0 | 0 |
| **C3** +q_t (FINAL)       | q_t = q_s^T transpose+store | **17** | 28 / **16** | **127,744** (50.3%) | 0 | 0 |

(*C2 already declares all 17 output args; q_t is written at C3.)

**Attribution:**
- **UB: flat at 127,744 B across all four stages** — every re-home addition costs
  **0 UB** (no new on-chip tile: beta_act/sig reuse the resident `ub_bact`; q_t
  reuses the `ub_p2` [C,C] scratch and the seed's existing `transpose_vf`; v_b's
  valid store reuses `ub_qs`). ub `jit=1, vfun=12, matmul=6` unchanged from seed.
- **default mutex: 28 throughout** — unchanged from the seed (no new mutex_id; the
  additions are stores/transpose on the vec side, not cross-core handoffs).
- **merge mutex: 17 → 16 at C2** — the two `beta_act`/`sig` stores (both read
  `ub_bact`, siblings) let `merge_siblings` coalesce one additional event group,
  a **reduction**, not growth. q_t (C3) changes nothing (28 / 16).
- **peak pipe-flags: 0** and **advisories: 0** (no unmapped opcode /
  `NotImplementedError`, no `perf:`) throughout.

**Deviation from 127,744 B: 0.0%.** Within the +10% gate (≤140,518 B) with the
full margin to spare. Neither mutex gate is tripped (default 28 ≤ 30; merge
16 ≤ 20).

## Diff summary (re-homed KAL vs seed) — output-promotion + wrapper + naming ONLY

Verified mechanically (AST + literal extraction):

| structural element | seed vs re-home |
|---|---|
| all 13 `@vf` bodies (code, docstrings aside) | **IDENTICAL** |
| all UB/L1/L0C tile declarations | **IDENTICAL** |
| all `matmul(...)` statements + operand routing | **IDENTICAL** |
| all `VcMutex`/`CvMutex` declarations | **IDENTICAL** |
| section ordering (M1→M2→M3→M4, flat mutex chain) | **IDENTICAL** |

The ONLY changes:
1. **Output promotion** — the 11 formerly-workspace SPEC tensors are returned. For
   10 this is return-list-only (no body change). For `v_b` it adds ONE valid-width
   `pl.store` (`v_b_out[.,0:C,v0:v0+vvw] <<= ub_qs[.,0:vvw]`): the seed's `v_b_ws`
   is zero-padded to VT for M4's fixed-64 read, but SPEC `v_b` is `[.,V]`, so the
   declared output takes a valid-width store while `v_b_ws` stays as internal
   padded workspace (needed unchanged by M4). This is the only genuinely-necessary
   structural addition, and it changes no seed logic.
2. **Three store-side additions** — `beta_act` and `sig` (two `pl.store` of the
   resident `ub_bact`, pad rows already 0) and `q_t = q_s^T` (the seed's existing
   `transpose_vf` gather in M1, into the reused `ub_p2`, then `pl.store`). No new
   tile, no new mutex; the `ub_p2` reuse in M1 is ordered before M3's `P2_ws`
   reload by the M1→M2→M3 mutex chain (M1's MTE3 store is fenced by
   `vcmutex.ready()`), the same transitive-ordering argument the seed uses for
   `v_b_ws` (M1) → M4.
3. **OpExec arg-order reshuffle** — OpExec classifies a returned GMTensor as an
   *output* and requires `inputs → outputs → Vars`; the seed returned only
   `u/w/w_T`, so its workspace tensors were all "inputs". The re-home returns 17,
   so the signature is regrouped to `[5 primary inputs] → [8 non-returned
   workspace] → [17 declared outputs in SPEC order] → [9 scalars]`, with
   `_SHAPE_BINDINGS` and the `OpExec(...)` call reordered to match. This is a
   pure declaration reorder — no body statement moved.
4. **Wrapper / entry / shape inference / operator naming** — `run_module1234`
   (grad_o-first, bwd golden) → `run_gdr_fwd_train_wy(q,k,v,g,beta)`; the harness
   imports `gdr_fwd_train_wy_golden_cpu` and compares all 17 outputs.

No tile decl, section order, mutex, operand routing, or vf body changed → this is
a re-home, not a re-author.

## The four preserved seed fixes (carried verbatim from the seed's M4 repair)

All four load-bearing M4 fixes recorded in the seed's `NOTES_module1234.md` are
present unchanged (the M4 Cube body and its buffer set were not touched):

1. **FLAT (non-nested) mutex chain** — `vcmutex`(M1→M2) is `free()`d immediately
   after M2 consumes M1's operands, before `cvmutex.lock()`; the three handoffs
   (`vcmutex`/`cvmutex`/`vcmutex2`) are each fully bracketed lock/ready/wait/free
   and never overlap. Double-nesting deadlocks the depth-1 pipeline.
2. **Module-4 DEDICATED Cube buffers** (`l1_attn`, `l1_op`, `l0c_uw`) — reusing
   M2's single-buffered cube objects across the M2→(vec M3)→M4 cube→vec→cube
   boundary starves the pipeline (never gets a reuse/ready credit). +3 cube objects.
3. **`l1_op` at the EXACT [C,C] width** (NOT `[C,KMAX]`) — `l1_op.T` transposes
   the full declared width, so an over-wide operand truncates the u/w contraction
   to the first fp32 K0 fractal (u rows ≥ 8 garbage; masked for w because kbeta_de
   decays, exposed by non-decaying v_b).
4. **v_b built in Module 1** (`scale_vtile_vf`), not via an M3 GM round-trip — the
   `ubout`(MTE3)→`ubin`(MTE2) cross-tile WAR a round-trip introduces is untracked
   by `auto_sync` and corrupts the first V sub-block for V=128/96. `v_b_ws` (M1)
   is ordered before M4's read by the mutex chain. (`v_f32_ws` remains a dead GM
   arg kept for structural parity; `cast_vtile_vf` is dead code the backend prunes.)

## Terminal-mutex note (carried forward)

The seed's NOTES raised an early warning that the *fully-fused* backward terminal
kernel (M1..M10) is at serious risk of exceeding the 32-mutex hard limit because a
cube section after a vec section cannot reuse the prior cube's buffers. **That
warning does not bind OP-A:** `gdr_fwd_train_wy` is only M1..M4, its own
cumulative mutex is 28 default / 16 merge (comfortably ≤ 32 under
`merge_siblings=True`), and it is a *standalone operator*, not a fused
continuation. The warning remains relevant only to the separate fused-backward
line of work, not to this pair's OP-A.

## Scope / caveats (NOT acceptance)

- **Simulator pass ≠ acceptance.** This is Stage 3.5 pre-build; the real-hardware
  verdict is Stage 4's `pypto-pro-op-verifier`. This carries no certification and
  is not a MEMORY.md pass record. Stage 4 needs a5 hardware (per the operator's
  paused status); this pre-build does not run it.
- The KAL Python simulator is CPU-only; no CANNSIM / real-HW run here.
- The 8 non-returned GM workspace tensors are wrapper-allocated scratch discarded
  after the call — not host compute, not deliverable outputs. `v_b_ws` stays
  padded to VT for M4's fixed-64 read; the declared `v_b` output is valid-V-width.
- Deliverables: `gdr_fwd_train_wy_kal_module1234.py` (KAL @kernel + OpExec sim
  entry), `staged_module1234.py` (backend PyPTO-Pro DSL, exactly one `@pl.jit`,
  no host tensor math in the jit body), this NOTES. The staged file is the Stage-4
  coder's input; it is not a delivered `test_*` file.
