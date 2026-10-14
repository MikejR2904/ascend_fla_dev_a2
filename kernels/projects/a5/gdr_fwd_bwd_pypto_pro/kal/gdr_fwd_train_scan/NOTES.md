# NOTES — gdr_fwd_train_scan (OP-B) Stage 3.5 KAL RE-HOMING

OP-B of the gated_delta_rule TRAINING-FORWARD pair. Single fusion module
(`module_count=1`, `is_fusion=true`; DESIGN §0): the M5 inter-chunk scan, emitting
the public forward `o` [B,T,H,V] bf16 **plus** the per-chunk ENTERING state
history `s_hist` [B,H,N,K,V] fp32. This is a **re-home, not fresh authoring**:
exactly ONE thing is added to the read-only fwd seed — the `s_hist` store.

## Seed choice + why the bwd seed was rejected

- **Chosen seed: `custom/gated_delta_rule_fwd/kal/staged_scan_module5.py`**
  (KAL source `tmp/kal_module12345/scan_probe.py::scan_kernel`) — the read-only
  lean forward's standalone M5 scan. 248,064 B / 25 mutex / 0 flags, sim-VERIFIED
  (produces `o`) and transpile-CLEAN. **It already produces `o`**; OP-B needs only
  the `s_hist` entering-state store added — a store-side snapshot of the resident
  state tile, which the bwd already proved costs **0 UB**.
- **Rejected: `custom/gated_delta_rule_bwd/kal/staged_module12345.py`** (231,680 B).
  The bwd scan DROPPED the entire o-branch (no `q_de@S`, no `A@v_new`, no bf16 o
  cast) and emits `v_new` + `S_hist` instead. Re-homing OP-B from it would require
  **re-adding the whole o-branch = re-authoring**, defeating the tight-graft
  contract. The two seeds are NOT equivalent for OP-B: the fwd seed is the o
  producer that needs one store added; the bwd seed is the state producer that
  dropped o. (This matches the orchestrator's recorded seed-steer analysis.)
- **`s_hist` store technique reused from the delivered bwd**:
  `gated_delta_rule_bwd_kal_module12345.py` L217 / `staged_module12345.py`
  L191-195 — the store-side snapshot of the RESIDENT state tile at the chunk-loop
  top, BEFORE the s15 update, sub-block-0 guarded.

## The graft (the ONLY change vs the fwd seed)

Store-side snapshot of the resident persistent state tile `state_ub`, inserted at
the chunk-loop top inside the existing `state_bridge.lock()`/`ready()` window,
sub-block-0 guarded, BEFORE the s15 state update:

```
s_hist[b_idx, h_idx, n_idx, 0:Kd, 0:Vd] <<= state_ub[0:Kd, 0:Vd]
```

- **Entering-state semantics**: the snapshot is taken at chunk-loop top, before
  `state_update_vf` (which runs at chunk bottom). So `s_hist[:,:,i]` = S entering
  chunk i (= state after chunks 0..i-1). For `ni=0`, `state_ub` is the freshly
  zeroed S (`zero_ub_vf(state_ub, Kd)` before the loop) → **`s_hist[:,:,0] = 0`**.
- **Zero new UB, zero arithmetic**: reuses the resident `state_ub` tile — no new
  tile group, no compute. Just an extra GM store of a tile already in UB.

## DIFF SUMMARY (re-homed KAL vs fwd seed, kernel body, name-normalized)

Name-normalized `@kernel` body diff (`scan_kernel` → `gdr_fwd_train_scan_kernel`):

1. signature: `o: GMTensor,` → `o: GMTensor, s_hist: GMTensor,`  (one GM output arg)
2. one store line added: `s_hist[b_idx, h_idx, n_idx, 0:Kd, 0:Vd] <<= state_ub[0:Kd, 0:Vd]`
3. `return o` → `return o, s_hist`
4. (comment block expanded to document the snapshot — cosmetic, no code)

**Nothing structurally else changed.** The o-branch (v_prime=w@S, v_new=u-v_prime,
o_inter=q_de@S, o=o_inter+A@v_new, bf16 cast+store), all tile declarations, the
4-handoff delta_h section ordering, the cross-core/auto_mutex handoffs, the matmul
routing (Fix A GM-routed operand bridging + Fix B pre-transposed k_f32), and every
vf body are the seed's code verbatim. No genuinely-necessary extra change was
required. In the transpiled DSL the graft is `pl.store(s_hist, t49_state_ub, ...)`
at the chunk-loop top (staged_module1.py L280), guarded by `if _k1 < 64`, before
`state_update_vf` (L311) — the exact bwd technique (its set_validshape sandwich
around the following state_gm store is emitted automatically by the backend).

Host side: added `s_hist` allocation + return-tuple handling in `run_scan`, and a
new golden/lean-forward oracle wiring in `__main__` (host-only; no core compute on
host — only `_pad_kf` layout transpose + zero allocation + one OpExec dispatch).

## C0..C3 incremental budget curve (guard header: arch=a5, backend=pypto)

`=== TRANSPILE TARGET  arch=a5 (device_type=950)  backend=pypto  ||  limits in
force: UB=253952  mutex=32  intra_flag=8  cross_flag=16 ===`

Worst-case shape B,T,H,K,V = 1,200,3,128,96 (K=128 splitn path + V<TILE):

| stage | ub_bytes | ub% | mutex | peak_flags/chan | nz | advisories |
|---|---:|---:|---:|---:|---:|---:|
| C0 baseline seed (o-only)        | 248,064 | 97.7 | 25 | 0 | 0 | 0 |
| C1 grafted (o+s_hist, default)   | **248,064** | 97.7 | 25 | 0 | 0 | 0 |
| C2 wrapper + shape-inference     | — | — | — | — | — | — |
| C3 grafted (o+s_hist, merge=True)| 248,064 | 97.7 | 20 | 0 | 0 | 0 |

- **C0 == seed's published 248,064 B / 25 / 0.** The re-homed seed reproduces it.
- **C2** is host-only (`run_scan` wrapper + shape_bindings); the `@kernel` body is
  unchanged, so the transpiled footprint is identical to C1 by construction — no
  separate transpile is meaningful (the backend never sees the host wrapper).
- **C3** engages `merge_siblings=True` (the delivery config): mutex 25 → 20; UB,
  flags, nz, advisories unchanged. Delivery-fit asserted via `transpile_guard.
  assert_delivery_fits` (merge mutex=20 ≤ 32, ub=248,064 ≤ 253,952). NB: this op
  fits at DEFAULT flags too (mutex 25 ≤ 32); merge is applied as the delivery
  choice, not out of necessity.

### TIGHT 0-UB-DELTA GATE (C1 grafted vs C0 baseline) — **PASS**

- **ub_bytes delta = 0** (C1 248,064 − C0 248,064)
- **mutex delta = 0** (C1 25 − C0 25)

The `s_hist` store of the resident tile cost **exactly 0 UB** (and 0 mutex, 0
flags), as required by the tight gate. No positive delta was absorbed into the
5,888 B headroom. Reproduce: `python tmp/kal_gdr_fwd_train_scan/measure.py`.

## Simulation — HARD acceptance (o reproduction + s_hist), 4 shapes × 3 seeds

Oracle: `gdr_fwd_train_scan_golden_cpu` (Stage-2 CPU golden; its `o[:, :T]` is
bit-identical to the read-only lean forward `o` by construction). Cross-check:
`o` also compared directly against `gated_delta_rule_fwd_golden_cpu` (read-only).
Criterion (fwd DESIGN §8/§9 mixed): elementwise peak-scaled
`allclose(rtol=1e-2, atol=1e-2·max|ref|)` OR relative-L2 `‖Δ‖₂/‖ref‖₂ ≤ 2e-2`,
applied to both `o` and `s_hist`. Reproduce:
`python custom/gdr_fwd_train_scan/kal/gdr_fwd_train_scan_kal_module1.py`.

**12/12 PASS.** Representative rows (rel_l2):

| shape [B,T,H,K,V] | seed | o rel_l2 | o-vs-lean rel_l2 | s_hist rel_l2 | s_hist[:,:,0]==0 |
|---|---|---|---|---|---|
| 1,128,4,64,64    | 42  | 1.683e-3 | 1.683e-3 | 1.32e-8 | True |
| 1,100,4,64,64    | 42  | 1.683e-3 | 1.683e-3 | 3.74e-8 | True |
| 1,192,2,128,128  | 42  | 1.544e-3 | 1.544e-3 | 1.91e-8 | True |
| 1,200,3,128,96   | 42  | 1.673e-3 | 1.673e-3 | 1.94e-8 | True |

- **o unchanged by the graft**: o rel_l2 ∈ {1.54e-3 … 1.73e-3} across all 12 cases
  — **bit-identical to the seed's own published numerics** (seed NOTES: 1.68e-3 /
  1.68e-3 / 1.54e-3 / 1.67e-3 for seed 42). o-vs-lean rel_l2 equals o rel_l2 in
  every case → `o[:, :T]` reproduces the read-only lean forward exactly. The graft
  did **not** alter o semantics; no tolerance was widened.
- **s_hist**: rel_l2 ~1e-8 (fp32 matmul roundoff; the snapshot is the exact fp32
  state used in the o computation), matches the golden's fp32 entering state.
  `s_hist[:,:,0] == 0` (exact) on every case. All outputs finite.

## Scope / caveats (NOT acceptance)

- **Simulator pass ≠ Stage-4 acceptance.** This carries no certification. The only
  verdict is Stage 4's `pypto-pro-op-verifier` on real a5 hardware. Per MEMORY,
  Stage 4 for this pair is paused (needs a5); this pre-build stops after Stage 3.5.
- No `test_` deliverable written; no other operator touched.
- The read-only seeds (`gated_delta_rule_fwd/`, `gated_delta_rule_bwd/`) were
  read only, never mutated. All new artifacts are under
  `custom/gdr_fwd_train_scan/kal/`.

## AMENDMENT (2026-09-02, Path B) — third GM output `v_new`

Authorized amendment: promote the per-chunk delta-corrected value `v_new`
`[B,H,N,C,V]` fp32 = `u_i − w_i@S` (S = entering state = `s_hist[:,:,i]`) to a
public GM output, using the **same store-side promotion pattern as `s_hist`**.
`v_new` is already computed inside the M5 scan (s12: `scan_sub_vf` leaves
`ub_uv = u − v_prime` resident) and was previously discarded. This is the exact
quantity the delivered backward b3/M8 (OP5 `gdr_bwd_dqkw`) consumes.

### The graft (the ONLY change vs the pre-amendment o+s_hist kernel)

Store of the RESIDENT `ub_uv` tile (holding `v_new` after `scan_sub_vf`) inside
the existing `vscaled_bridge.lock()`/`ready()` window, sub-block-0 guarded, at the
point `v_new` is resident and BEFORE it is consumed (cube `o += A@v_new`, s15 state
update):

```
v_new[b_idx, h_idx, n_idx, 0:C, 0:Vd] <<= ub_uv[0:C, 0:Vd]
```

Transpiled DSL (`staged_module1.py` L297): `pl.store(v_new, t52_ub_uv, [(bh//H),
(bh%H), ni, 0, 0])` — same shape as the s_hist store (`pl.store(s_hist,
t49_state_ub, ...)` L280). Reuses `ub_uv` (already declared `[C,TILE]` fp32, 32 KB
in the UB dump) → **no new tile group, no arithmetic, EXACTLY 0 UB**.

**DIFF SUMMARY (vs pre-amendment kernel, exactly as scoped):**
1. signature: `o, s_hist,` → `o, s_hist, v_new,` (one GM output arg)
2. one store line added (the `v_new <<= ub_uv[0:C,0:Vd]` above), sub-block-0 guarded
3. `return o, s_hist` → `return o, s_hist, v_new`
4. `_SHAPE_BINDINGS[12] = [0,2,5,6,4]` (== u binding, `[B,H,N,C,V]`)
5. host `run_scan`: `torch.empty(v_new)` alloc + pass + return-tuple; `__main__`
   golden/kernel unpack fixed to the 3-tuple `(o, s_hist, v_new)` + v_new checks.
   (host-only; no core compute on host)

Nothing structurally larger. o/s_hist code paths untouched.

### C0 → C1 budget delta (guard header: arch=a5, backend=pypto) — 0 on every axis

`=== TRANSPILE TARGET  arch=a5 (device_type=950)  backend=pypto  ||  limits in
force: UB=253952  mutex=32  intra_flag=8  cross_flag=16 ===`

Worst-case shape B,T,H,K,V = 1,200,3,128,96:

| stage | ub_bytes | ub% | mutex (default) | mutex (merge) | peak_flags/chan | L0A |
|---|---:|---:|---:|---:|---:|---:|
| C0 (o + s_hist)          | 248,064 | 97.7 | 25 | 20 | 0 | 65,536 |
| C1 (o + s_hist + v_new)  | 248,064 | 97.7 | 25 | 20 | 0 | 65,536 |
| **delta (C1 − C0)**      | **0**   | 0   | **0** | **0** | **0** | **0** |

- **0-UB-DELTA GATE: PASS.** ub delta = 0; the tight operator's 5,888 B headroom
  (253,952 − 248,064) is fully preserved — the v_new store absorbed nothing.
- L1=245,760, L0B=65,536, L0C=131,072 also identical C0↔C1 (v_new store touches no
  cube/matmul buffer). Transpile clean: 1 `@pl.jit`, 7 `pl.matmul`, 16/16
  set/wait_cross_core, Vec `_nz`=0, non-perf advisories=0.
- Reproduce: `python tmp/kal_gdr_fwd_train_scan/measure_vnew.py c0` (pre-edit) and
  `... c1` (post-edit; writes `staged_module1.py`).

### Simulation — o + s_hist + v_new, 4 shapes × seeds 42/123/456 — 12/12 PASS

Oracle: `gdr_fwd_train_scan_golden_cpu` (now returns the 3-tuple `(o, s_hist,
v_new)`). Same mixed peak-scaled / rel-L2 ≤ 2e-2 criterion, applied to all three.
Reproduce: `python custom/gdr_fwd_train_scan/kal/gdr_fwd_train_scan_kal_module1.py`.

Representative rows (rel_l2), seed 42:

| shape [B,T,H,K,V] | o rel_l2 | o-vs-lean | s_hist rel_l2 | v_new rel_l2 | v_new0==u0 | s_hist[:,:,0]==0 |
|---|---|---|---|---|---|---|
| 1,128,4,64,64   | 1.683e-3 | 1.683e-3 | 1.32e-8 | 3.56e-9 | True | True |
| 1,100,4,64,64   | 1.683e-3 | 1.683e-3 | 3.74e-8 | 1.52e-8 | True | True |
| 1,192,2,128,128 | 1.544e-3 | 1.544e-3 | 1.91e-8 | 8.26e-10 | True | True |
| 1,200,3,128,96  | 1.673e-3 | 1.673e-3 | 1.94e-8 | 8.26e-9 | True | True |

- **o unchanged**: o rel_l2 ∈ {1.54e-3…1.73e-3} across all 12 — bit-identical to the
  pre-amendment numerics; `o[:, :T]` still reproduces the read-only lean forward
  (o-vs-lean rel_l2 == o rel_l2 every case). No tolerance widened.
- **s_hist unchanged**: rel_l2 ~1e-8, `s_hist[:,:,0]==0` exact every case.
- **v_new correct**: rel_l2 ~1e-9…1.5e-8 vs golden; `v_new[:,:,0]==u_0` (entering
  S=0 → v_prime=0) every case. All three outputs finite on all 12 cases.

Simulator pass ≠ Stage-4 acceptance (unchanged caveat; Stage 4 still needs a5).

## Deliverables

- `kal/gdr_fwd_train_scan_kal_module1.py` — the `@kernel` source (easyasc a5
  facade) + OpExec sim self-check entry.
- `kal/staged_module1.py` — backend-transpiled PyPTO-Pro DSL (merge_siblings=True):
  1 `@pl.jit`, 7 `pl.matmul`, 16/16 set/wait_cross_core, ub_bytes 248,064,
  Vec _nz 0, advisories 0.
- `kal/transpile_guard.py` — the (arch=a5, backend=pypto) budget guard used for
  every measurement (copied from the bwd op).
- `kal/NOTES.md` — this file.
- `tmp/kal_gdr_fwd_train_scan/measure.py` — the C0..C3 measurement harness
  (measurement only; writes staged_module1.py).
