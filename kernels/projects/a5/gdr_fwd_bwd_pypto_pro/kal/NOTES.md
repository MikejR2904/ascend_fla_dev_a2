# NOTES — gated_delta_rule_fwd KAL pre-build (hardware-architecture-language stage)

Covers the full operator: the four chunk-prep steps plus the inter-chunk scan,
final output `o [B,T,H,V]` bf16, built on top of the sim-verified chunk-prep
kernel. The scan is the loop-carried recurrence over chunks with persistent
state `S[K,V]` — the only genuine cube+vector fusion in the operator.

## Headline status

- **scan (standalone): sim-VERIFIED AND transpile-CLEAN.**
  `tmp/kal_fused/scan_probe.py` implements the scan (reading chunk-prep
  outputs u,w,A,q_de,decay_exp,dfac,k_f32 as GM inputs, produced here by the
  golden `_state_through_uw`), keeps `S[K,V]` loop-carried in a persistent UB
  tile (the `kernels/a5/gdn_legacy/delta_h_state_bridge_v1_c8.py` template), and
  reproduces `gated_delta_rule_fwd_golden_fused`'s final `o` on **all four
  DESIGN §8 shapes** under the DESIGN §8/§9 mixed peak-scaled + rel-L2 criterion.
  It now also **transpiles clean** (`kal/staged_scan.py`): ub_bytes
  248064 <= 253952, `Vec _nz tiles: 0`, advisories 0. **Boundary 2 is RESOLVED**
  (see the resolution below); the two fixes are operand *bridging* and operand
  *layout* only — the scan algorithm and its 4-mutex / per-(b,h) serial-chunk
  structure are unchanged (sim rel_l2 bit-identical to the pre-fix baseline).
- **Full single-kernel integration (chunk-prep + scan): still does NOT pass** —
  Boundary 1 (below) is a genuine Stage-3.5 boundary. Per the task contract it
  is a boundary finding (not a failure) that keeps the operator's full the full fusion
  fusion on the Stage-4 hand-written path; the standalone scan itself
  is a clean, verified pre-build ready for the Stage-4 coder.

## scan — verified simulation (reproducible)

```
export PYTHONPATH="/root/.claude/skills/pypto-kernel-arch-lang/payload:${PYTHONPATH:-}"
python tmp/kal_fused/scan_test_all.py     # all 4 shapes, seed 42
```

Oracle: `golden_fused.py`
(`gated_delta_rule_fwd_golden_fused(q,k,v,g,beta) -> o`). Criterion
(DESIGN §8/§9): elementwise mixed peak-scaled `allclose(rtol=1e-2,
atol=1e-2·max|o_g|)` OR relative-L2 `‖Δ‖₂/‖o_g‖₂ ≤ 2e-2`. Observed:

| shape [B,T,H,K,V] | scenario | N | rel_l2 | max_abs | peak |
|---|---|---:|---|---|---|
| [1,128,4,64,64]  | aligned (T=2C)        | 2 | 1.68e-3 | 6.2e-2 | 2.86e1 |
| [1,100,4,64,64]  | T tail (last chunk 36)| 2 | 1.68e-3 | 5.8e-2 | 3.57e1 |
| [1,192,2,128,128]| T=3C, K=V=128         | 3 | 1.54e-3 | 1.3e1  | 7.45e3 |
| [1,200,3,128,96] | tail + V<TILE, K=128  | 4 | 1.67e-3 | 7.8e0  | 3.53e3 |

All finite, all within the mixed criterion. rel_l2 ~1.5e-3 (bf16 final-store
roundoff dominates, as DESIGN §9 predicted). The large-magnitude peak (up to
7.4e3, and higher for longer sequences) is the legitimate unnormalised-k output.

### Scan realisation (per (b,h), serial over chunks; delta_h template)

Partition = (b,h) across cube cores; each core scans its N chunks serially.
`S[K,V]` is a persistent UB tile, zeroed per (b,h). Per chunk (4 mutex
handoffs, mirroring delta_h_state_bridge):

- **SA** (state_bridge, vec→cube): vec stages S (UB) → **GM** (`state_gm`); cube
  loads S GM→L1 (nd2nz) then `v_prime = w@S` and (accumulate) `o = q_de@S`
  [`is_init=True`]. (Bridging via GM, not UB→L1 — see the Boundary-2 resolution.)
- **handoff A** (cube→vec): cube copies v_prime L0C→UB; vec
  `v_new = u - v_prime` (into ub_uv), `v_scaled = v_new·factor[c]` (in place
  into ub_vp — v_prime is consumed to form v_new before the overwrite, so no
  separate v_scaled tile). `factor[c] = dfac[c,0]` (dfac broadcast over K).
- **vscaled_bridge** (vec→cube): vec stages v_new, v_scaled → **GM**
  (`vnew_gm`, `vscaled_gm`); cube loads them GM→L1 (nd2nz) then
  `o += A@v_new`  [`is_init=False`, one L0C accumulator gives o=q_de@S+A@v_new],
  `delta = k_f32^T @ v_scaled` (left operand `k_f32^T` loaded from the
  pre-transposed GM input `k_f32_T`, so it fills L0A non-transposed).
- **handoff B** (cube→vec): cube copies o, delta L0C→UB; vec casts o→bf16 and
  stores `o[b, t0:t0+vc, h, 0:V]` (s16: slice [:T] + transpose-back), then
  `S = S·decay_exp[C-1] + delta` in place.

Loop-carry is **vec-side only** (S updated in-place in UB); the cube reads a
fresh L1 copy each chunk (producer-first, no cross-side loop-carried mutex, no
consumer-first deadlock). This is exactly why delta_h keeps state in UB.

## Load-bearing findings (verified; reuse these in Stage 4)

1. **L0C→UB copy defaults to SPLITM (dual sub-block).** A single-sub-block
   full-height copy `ub[0:C,:] <<= l0c` fails (`M must be <= M_src, M=128,
   M_src=64`). Use `l0c.subblk(0)` to force `dual_mode=SINGLE` for
   single-sub-block. (fp32→bf16 downcast ALSO requires SINGLE, so cast in a vf
   and store fp32→fp32.)
2. **Vec work must be sub-block-0-guarded** (`row_start=GetSubBlockIdx()*C; if
   row_start<C:`) — both AIV sub-blocks run the shared loop; unguarded, sub-block
   1's uninitialised UB corrupts half the output.
3. **fp32→bf16 store**: `cast(bf16_reg, fp32_reg, cfg)` with **no mask**
   (mask resolves against the bf16 dst lanes) + `reg_to_ub_downsample(...)` (the
   bf16 pack store). Idiom from `kernels/a5/attention/pfa/pfa_fd.py:output_cast_vf`.
4. **Transposed matmul operands cannot be partially sliced on the transposed
   axis** (a broader form of DESIGN/finding #6):
   - partial-K on a transposed L0B (`state_l1[0:Kd,0:TILE].T` with `k=Kd`,
     ROW-slice) — OK.
   - partial-N (`n=Vd<TILE`) or partial-M (`m=Kd<TILE`) on a transposed operand,
     or a COL-slice before `.T` (`kf_l1[0:C,0:Kd].T`) — WRONG (right magnitude,
     scrambled values). Fix: pad the operand to full TILE (zero-fill) and matmul
     at the compile-time full width; the zero pad preserves the exact result.
     The scan pads k_f32 to `[C,TILE]` and does `m=TILE`.
5. **fp32 transposed operand L0B is capped at 8192 elements** ([128,64]).
   `state.T = [128,128]` (K=V=128) overflows. Fix: `splitn=SUB(=64)` on
   v_prime/o_inter/o+=A matmuls keeps each transposed N-tile ≤ [64,128]=8192.
6. Matmul reads need `with auto_sync()`; without it the GM→L1 load is unordered
   before the matmul and the operand reads NaN (a probe artefact, easy to hit).
7. dfac load: dfac is broadcast over K, so a contiguous `[C,8]` block load and
   reading column 0 gives factor[c] (no fragile strided burst).

## Boundary 1 — full-fusion UB budget + phase-1 integration bug

Fusing chunk-prep (UB-heavy WY construction: q_s/k_f32/k_beta/v_b staging + kk_gram/qk_gram/
L_mask/attn/A [C,C] tiles + the forward-substitution) with scan (persistent
`S[K,V]` + `Sdelta[K,V]`, [TILE,TILE]=64 KB each) in one `@kernel` exceeds the
253952 B UB budget when tiles are declared separately (~287 KB; the simulator
sums declared tiles — no automatic liveness reuse).

The pooling that FITS (255 KB, sim-measured) reuses phase-1 tile memory for
phase-2 (they are separated by `bar_all` per (b,h)): the two `[TILE,TILE]`
tiles (S, Sdelta) double as the four phase-1 `[C,TILE]` staging tiles via
ROW-slices; q and k share one bf16 tile (sequential, fenced `SEvent(MTE3,MTE2)`);
kk_gram and qk_gram share one `[C,C]` tile (sequential). Constraint discovered: **a `[C,C]`
view of a `[C,TILE]` tile is a COL-slice → non-contiguous → the vf reg↔ub ops
fail** ("destination view is empty"); only contiguous ROW-slices of `[TILE,TILE]`
tiles are reusable, so `[C,C]` tiles must stay dedicated.

### UB budget — sim-measured vs transpiler-measured (MEASUREMENT-ONLY, 2026-08-26)

The "255 KB, sim-measured FITS" above is **unreliable**: the easyasc simulator
does NOT model NZ staging (NOTES_module1234), so it undercounts the UB→L1
matmul-operand bridges. A backend **transpile** (which DOES model NZ staging;
static codegen only, does not execute — the chunk-0 kk_gram NaN is a *runtime* issue
and does not block measuring the static footprint) of this exact pooled kernel
`gated_delta_rule_fwd_kal_fused.py` reports the REAL terminal-cumulative
footprint. Harness `tmp/kal_fused/transpile_fusion.py`, staged DSL
`tmp/kal_fused/staged_fusion_m12345.py` (written to tmp only; kal/ kernel
source untouched — nothing here was fixed or changed), representative §8 shape
`B,T,H,K,V = 1,200,3,128,96` (K=128 splitn path + V<TILE, same worst case as the
scan transpile):

```
ub_bytes: 416768   budget: 253952  -> OVERFLOW ( -162816 B )
Vec _nz tiles: 252 (source.count("_nz"))  = 3 distinct NZ-staging bridge groups
                                             + 48 column-view aliases (16/bridge)
advisories: 62 total = 61x "mutex_id exhausted (>32) - needs lifetime reuse"
                     +  1x "l1_to_l0 src_is_transpose into L0A ..." (the delta
                        matmul left operand k_f32^T; same gap Fix B fixed in scan)
@pl.jit: 1  @pl.vector_function: 17  pl.matmul: 12  set_cross_core/wait: 21/21
```

**UB breakdown** (from the transpiler's per-tile-group addresses in the staged
DSL; real region is `[0x0 .. first-NZ-addr)`):

| region | bytes | note |
|---|---|---|
| real working tiles | **261,120** (0x0..0x3fc00) | pooled phase-1/phase-2 tiles; **ALREADY 7,168 B OVER budget on their own** (261,120 − 253,952) — the transpiler does no automatic liveness reuse, it sums all declared tile groups at distinct addresses, so the sim's alias-pooling does NOT shrink the transpiled footprint |
| NZ-staging region | **155,648** (0x3fc00..end) | three phase-2 UB→L1 matmul-operand bridges (`state_ub→state_l1` L775, `ub_uv→vnew_l1` L800, `ub_vp→vscaled_l1` L801) — **identical magnitude to the standalone scan's 155,648 B** (same three bridges; this fusion kernel does NOT have Fix A applied) |
| **total ub_bytes** | **416,768** | overflow by 162,816 B |

So both prior "255 KB FITS" claims are wrong two ways: (a) the transpiler's real
footprint is 416,768 B, not ~255 KB (261,120 B); (b) even the sim's ~255 KB
already exceeds 253,952 B arithmetically. The **real** terminal-cumulative
ub_bytes to design the bwd operator against is **416,768 B** (or 261,120 B real
tiles if all NZ staging is removed — see Fix-A note below).

### Fix A (GM-workspace operand bridging) applicability to the fusion

Fix A (route matmul operands GM→L1 nd2nz through GM workspace instead of on-chip
UB→L1) removed 155,648 B of NZ staging from the standalone scan. **It IS
structurally applicable to this fusion**: the fusion's only three UB→L1 bridges
are the phase-2 scan bridges (`state_l1 <<= state_ub` L775, `vnew_l1 <<= ub_uv`
L800, `vscaled_l1 <<= ub_vp` L801) — byte-for-byte the same three bridges Fix A
already eliminated in the scan probe. Phase-1 (preprocess–u/w) already GM-routes all its
matmul operands (UB→GM store then cube GM→L1 nd2nz, kernel header L60–76), which
is why only 3 NZ groups appear and all are phase-2. GM-routing the three phase-2
bridges would remove the identical **155,648 B** of NZ staging (416,768 → 261,120).

**But Fix A alone does NOT make the fusion fit.** After removing all NZ staging,
the pooled **real working tiles still = 261,120 B > 253,952 B (7,168 B over)**.
The pooled real-tile set overflows on its own; NZ staging is not the only, nor
even the binding, overflow. Two additional structural blockers surface only at
transpile: (1) the 61× `mutex_id exhausted (>32)` advisories — the pooled fusion
needs more than the hardware's 32 mutex ids; (2) the surviving `src_is_transpose
into L0A` gap (fusion lacks Fix B; `kf_l1` L559 is the old [C,TILE] operand).
**Design implication for gated_delta_rule_bwd** (which fuses MORE modules): budget
against the **transpiler's** ub_bytes, not the sim's; a single terminal-cumulative
kernel that pools this many tiles is already ~7 KB over UB before any NZ staging
and blows the 32-mutex budget — the bwd design should not assume alias-pooling
buys transpiled UB, and likely needs GM-routing (Fix A) on every matmul operand
**plus** a genuine reduction in resident-tile count (fewer/narrower simultaneous
tiles, or a split kernel), not just tighter aliasing. This is an ANALYSIS note;
the kernel was not modified.

With the pooling the kernel fits (255 KB) and runs finite, but produces a
**chunk-0 phase-1 `kk_gram = k_beta @ k_f32^T` NaN** (chunk 1 and later are fine;
`k_f32`, `k_beta`, `q_s`, `A`/qk_gram are all verified bit-exact; only the kk_gram matmul
of the FIRST inner-chunk iteration reads uninitialised memory). The nested
`(b,h)+chunk` loop reusing the gram L1 operand tiles + the phase pooling defeats
auto_sync's first-iteration L1 ordering. Tried and did not fix it: `qk_fence`
(fixed k_f32/k_beta but not kk_gram), grouping the gram loads, a chunk-boundary
`bar_all`, full-tile `k=TILE` gram matmuls, disabling phase-2 (kk_gram NaN is in
phase-1 itself). Isolated evidence in `tmp/kal_fused/dbg_*.py`.

## Boundary 2 — scan transpile — RESOLVED (was transpile_gap + ub_overflow)

The FIRST transpile of the clean-simulator scan (S loop-carried in UB, matmul
operands bridged **UB→L1**: `state_l1 <<= state_ub`, `vnew_l1 <<= ub_uv`,
`vscaled_l1 <<= ub_vs`) reported:

```
ub_bytes: 403712  budget: 253952 -> OVERFLOW
advisory: kernel:l1_to_l0 src_is_transpose into L0A from a tile no vector
          insert fills; the layout has to be chosen at the GM->L1 load instead
```

- **ub_overflow**: each UB→L1 bridge materialises a same-size NZ-staging copy in
  UB (chunk-prep finding #5) — three bridges added 155,648 B of NZ region on top
  of the 248,064 B of real working tiles → 403,712 > 253,952 B.
- **transpile_gap**: the delta matmul's LEFT operand `k_f32^T` transposed INTO
  L0A; the transpiler cannot fill that layout from a plain L1 tile.

### Resolution — two changes, no algorithm change, no net UB

**Fix A (bridging → removes ub_overflow, `Vec _nz tiles` 3-region → 0).** Route
the three vec→cube matmul operands through **GM workspace** exactly as
chunk-prep's u/w does (NOTES_module1234 finding #5): the vec side stores UB→GM
(MTE3), the cube side loads GM→L1 via the MTE2 nd2nz path — never UB→L1, so no
NZ-staging copy is materialised in UB. New per-(b,h) GM scratch `state_gm`
[B,H,TILE,TILE], `vnew_gm` / `vscaled_gm` [B,H,C,TILE] (distinct regions, no
cross-core aliasing; single-buffered, reused each serial chunk). **S's loop-carry
stays vec-side UB** (`state_ub`, updated in place by `state_update_vf`);
`state_gm` is only a per-chunk copy to supply the matmul operand, so the
"loop-carry is vec-side only / no cross-side loop-carried mutex / no
consumer-first deadlock" property is preserved. The VcMutex end-pipes move to the
proven chunk-prep GM-routing config `src_end_pipe=MTE3, dst_end_pipe=FIX` (was
`dst_end_pipe=MTE1`, appropriate for the old direct UB→L1 bridge).

**Fix B (layout → removes the L0A-transpose gap).** Store `k_f32` PRE-TRANSPOSED
in the GM input `k_f32_T` [B,H,N,TILE,C] (= padded `k_f32`^T; pad+transpose are
pure layout, no arithmetic — in the fused kernel the preprocess step would store it
transposed). The delta matmul's LEFT operand is then loaded NON-transposed
(`kfT_l1` [TILE,C] filled by a plain `pl.move` into L0A), and the RIGHT operand
carries the L1-side `.T` (L0B transpose, clean — the chunk-prep u/w idiom). Under
the easyasc convention `matmul(dst, l1a, l1b) = l1a @ l1b**T` (l1a read [M,K],
l1b read [N,K]): `matmul(delta_l0c, kfT_l1[0:TILE,0:C], vscaled_l1[0:C,0:TILE].T,
m=TILE, n=TILE, k=C)` gives `delta[i,j] = Σ_c k_f32[c,i]·v_scaled[c,j] =
(k_f32^T @ v_scaled)[i,j]`. finding #4 still holds: the transposed axis is not
partially sliced — the full `[TILE,C]` k_f32^T is used at `m=TILE`, its pad rows
[Kd:TILE] are 0 (from the k_f32 col pad) so `delta` rows [Kd:TILE] = 0.

`kfT_l1` [TILE,C] replaces the old `kf_l1` [C,TILE] at the same byte size, and the
GM scratch is in GM, not UB — so the fix adds **no net UB**: the surviving real
working tiles (248,064 B) are exactly the transpile budget, leaving 5,888 B
headroom. The three vprime/o/`o+=A` matmuls keep `splitn=SUB` (the fused KAL
finding #5: the transposed S / v_new L0B tile stays ≤ [64,128] = 8192 fp32).

## Transpile evidence (scan, standalone) — CLEAN

```
python tmp/kal_fused/transpile_scan.py   # writes kal/staged_scan.py
```
Representative shape `B,T,H,K,V,N = 1,200,3,128,96,4` (K=128 splitn path + V<TILE;
worst case for the compile-time TILE=128 tiles). Reported:

```
C1 ub_bytes: 248064  budget: 253952 -> PASS        (5,888 B headroom)
C2 Vec _nz tiles: 0 -> PASS
C3 advisories: 0  non-perf: 0  L0A-transpose advisories: 0 -> PASS
```

Structure preserved vs. the pre-fix DSL: `@pl.jit: 1`, `@pl.vector_function: 4`,
`pl.matmul: 7`, `set_cross_core/wait_cross_core: 16/16` (the backend produces the
DESIGN-§6 INTRA_BLOCK cross-core for the 4 scan handoffs). In the emitted DSL the
delta L0A tile is filled by a plain `pl.move(...__l0a, ...kfT_l1, [0,0])` (no
`src_is_transpose`), confirming Fix B. The DSL is structurally correct (one jit,
correct 4-handoff fusion) and now free of ub_overflow and the transposed-L0A gap.

## Deliverables / artefacts

- `kal/gated_delta_rule_fwd_kal_fused.py` — the full the full fusion integration
  attempt (documented header; does not pass — Boundary 1).
- `kal/staged_scan.py` — transpiled DSL of the sim-VERIFIED scan,
  now **CLEAN** (ub_bytes 248064, `Vec _nz tiles: 0`, advisories 0). Boundary 2
  resolved (Fix A GM-routed bridging + Fix B pre-transposed k_f32).
- `tmp/kal_fused/scan_probe.py` — the sim-verified + transpile-clean
  standalone scan (Fix A/B applied to the kernel, GM scratch + k_f32_T in
  `run_scan`).
- `tmp/kal_fused/scan_test_all.py` (sim, 4 shapes),
  `tmp/kal_fused/transpile_scan.py` (transpile, per-criterion PASS/FAIL),
  `dbg_*.py` — reproduction + isolation scripts.

## Scope / caveats (not acceptance)

- Simulator pass ≠ acceptance. The scan algorithm is sim-verified; the real-
  hardware verdict is Stage 4's `pypto-pro-op-verifier`. This pre-build carries
  no certification.
- Recommendation to the orchestrator (updated): the **standalone scan
  is now both sim-verified and transpile-clean** — Boundary 2 is resolved, so the
  ledger's `sim_verified_through` can move **4 → 5** and `kal.status` can clear
  from `failed` for the standalone-scan scope on the strength of the four criteria
  (C1 ub_bytes 248064 ≤ 253952, C2 `Vec _nz tiles: 0`, C3 advisories 0, C4 sim
  rel_l2 {1.68e-3, 1.68e-3, 1.54e-3, 1.67e-3} on the four §8 shapes). The
  orchestrator owns the ledger; this is a recommendation with the evidence above.
  **Boundary 1 (the chunk-prep + scan single-kernel fusion) is NOT resolved** and stays
  the Stage-4 hand-written path — the full-fusion UB budget + the chunk-0 phase-1
  kk_gram NaN are out of scope here and untouched. The scan algorithm + findings 1–7 +
  the Fix A/B bridging-and-layout recipe are a complete, reusable head start for
  the Stage-4 coder.
