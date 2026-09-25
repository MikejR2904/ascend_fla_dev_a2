# A2-K1 — GDN chunk fwd+bwd move-to-a2 (design / plan)

Batch A of `docs/research/a2_gdn_abi.md` §5. Move the a5 derived units
`kernels/projects/a5/gdn_chunk_{fwd,bwd}` to a2, close six ABI gaps, add the
L0C-settle barriers. Batch B (gate-span calibration) and Batch C (decode /
GDA-04) are explicitly out of scope.

## Port sources (verified present on main)

- `kernels/projects/a5/gdn_chunk_fwd/` — `kernels/{pipeline.py,stages.py}`, `contract.json`,
  `unit.py`, `ref/`. GDA-02 added the GQA/GVA in-kernel head indexing here.
- `kernels/projects/a5/gdn_chunk_bwd/` — reverse recurrence + saved histories.
- Public entries `ascend_fla/ops/gdn_chunk_{fwd,bwd}.py`.

## Iron rules that shape the port (AGENTS.md)

- **One operator name per process per build** (§6 rule 2): the new a2 units get
  **distinct suffixed operator names** — do not reuse the a5 unit names (BF-05
  hit a silent wrong-binary on exactly this).
- **C++-keyword-safe kernel param names**: no `do` / `in` / `class` (flagged 3×
  in the spec — GDA-03, PK-05, and A2-K1).
- **fp32 dual-oracle** correctness (§6): fla `naive`/`naive_recurrent` + repo CPU
  ref; judge in fp32 by rel-L2 / max_abs_diff, never bitwise, never bf16-elementwise.
- **Gate-on-unsupported, no silent fallback** (§7).
- **A2 is 20 cube / 40 vector**, `block_dim` ceiling untested — assert no number.

## Six ABI gaps → approach (from a2_gdn_abi.md §1.1–1.7)

1. **§1.1 GQA/GVA**: carry `HV` on v/state/output/saved histories, keep qk-side on
   `H`; in-kernel `i_h = i_hv // (HV//H)` (GDA-02 template), broadcast q/k/g/b across
   the `HV/H` group. `domain.shape` gains `HV: HV % H == 0`.
2. **§1.2 token-major**: public token-major, permute to kernel layout **inside** the
   launcher as a real materialization (not a strided view — A2 strided Slice is
   unavailable, AGENTS §5). Assert contiguity at the kernel boundary.
3. **§1.3 nonzero initial_state**: accept fp32 `[B,HV,128,128]` (KDA precedent).
4. **§1.4 dh0**: backward emits `dh0`, accepts `dht` (kda_bwd precedent).
5. **§1.5 fp32 final_state**: the only numeric-risk item. Build bf16-state and
   fp32-state variants; **report** o/final_state rel-L2 divergence vs the fp32
   recurrent oracle at long C as *evidence* (not a threshold). **Open decision (PM):**
   whether the shipped default is fp32-state only, or both retained.
6. **§1.7 scale + no-tail-path**: add `scale: f32` absorbed at q's read; require
   `T % 64 == 0`, else reject with the nearest legal T (no real tail path).

## L0C settle (a2_gdn_abi.md §2, A2-01 hit-table)

`barrier(Pipe.M)` after **every** L0C-accumulate MMAD, dtype-agnostic, at:
- fwd **inverse** stage (a5 `gdn_fwd/inverse.py:306/312/317-318` → the corresponding
  inverse block in `gdn_chunk_fwd/kernels/{stages,pipeline}.py`), FP32 M16 hand-chain.
- bwd **finalize** stage (`finalize.py:205/217/226/235/244`), FP32 chain.
- bwd **wu** stage (`wu.py:206`), BF16 M64 — barrier added for uniform discipline,
  not because M64 is unsafe.
Keep the pin's FP32 split-K rule; **reject M<64 bf16/fp16 splitk** per AGENTS §7
(Qwen3-Next is bf16, so this exposure is live). The existing
`tests/test_a2_accumulate_barriers.py` static guard auto-covers the new files.

## Test plan (`tests/test_a2_gdn_chunk.py`, new)

- fp32 dual-oracle on o / final_state / 7 grads.
- **Asymmetric `HV≠H`** (H=2, HV=4) with per-group-distinct values so a wrong
  `i_h` cannot pass.
- Two-segment chaining (nonzero initial_state + `dh0`) vs single call.
- `bd1 == bd4` bitwise where `B*HV` is large enough to exercise core splitting.
- Barrier negative control: delete one `barrier(Pipe.M)` → reproduce the M16 bf16
  failure (A2-11 method); accumulate-barrier count == accumulate-MMAD count.
- Small, safe gate span only (≤10, far from the 88.7 line) — span calibration is Batch B.

## Resolved decisions (PM NOTE, 2026-09-25, issue #44)

- **§1.5 shipped default = fp32-state.** The bf16-state build exists **only** to
  report the divergence evidence vs the fp32 recurrent oracle (into evidence, not a
  threshold, not a switchable shipping option).
- **a2 `MAX_GATE_SPAN` GDN field = blank / "untested (Batch B)".** Span calibration
  is Batch B (separate task). This batch's kernel tests use only small, safe gate
  span (≤10, far from the 88.7 line); no boundary testing, no specific number.

## Status

Onboarding + branch + this plan done. Next: read the a5 `gdn_chunk_fwd` stage
structure (`stages.py`/`pipeline.py`) and contract, then scaffold the a2 unit
(suffixed op name, HV axis, barrier sites) and stand up the dual-oracle test.
