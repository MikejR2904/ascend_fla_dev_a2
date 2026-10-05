# A2-K1 GDN chunk backward — hardware baseline (910B3, device 2)

Measured 2026-10-05 on Ascend 910B3, CANN 9.2.0-beta.1, torch_npu 2.10, ascriptor
library 90cfcdc; fla via triton-ascend (triton 3.2). Shape s512 = B1 T512 H4, HV=H,
fp32, block_dim 40. This is the **pre-optimization baseline** — the shipped 8-kernel
reverse chain (a sim-imposed decomposition), not yet the lean DESIGN.md structure.

## Correctness — a2 kernel vs fla Triton `chunk_gated_delta_rule` backward (rel-L2)

| grad | rel-L2 |
|---|---|
| dq | 2.66e-07 |
| dk | 2.83e-07 |
| dv | 2.84e-07 |
| dg | 4.28e-07 |
| dbeta | 3.07e-07 |

The a2 backward matches fla's Triton backward on-device to ~1e-7 (same op, same NPU).
Confirms the `scale` scalar binding and that fla's dg/dbeta parametrization matches
the a2 adjoint.

## Device time (msprof Task Duration, per-call = total / 48 captures)

a2 backward total: **~9440 µs/call**.

| stage | µs/call |
|---|---|
| ReverseDg | 1850 |
| ReverseDkDdr | 1718 |
| ReverseDq | 1529 |
| ReverseRec | 1384 |
| ReverseDkBz | 1146 |
| Replay | 925 |
| ReverseDbeta | 859 |
| GroupReduce | 28 |

Host wall (npu.sync loop, 50 iters): a2 backward 9507 µs/call; fla Triton **fwd+bwd**
10191 µs/call. Wall ≈ device time, so the chain is device-bound (little dispatch
overhead) — and the a2 backward alone costs about as much as fla's entire fwd+bwd step.

## Utilization — bounding pipe (pure vector; cube 0 / N/A)

Per-pipe ratios, mean / max over 384 kernel launches:

| pipe | mean | max |
|---|---|---|
| aiv_vec_ratio | 0.713 | 0.939 |
| aiv_mte2_ratio (loads) | 0.358 | 0.617 |
| aiv_mte3_ratio (stores) | 0.138 | 0.389 |
| aiv_scalar_ratio | 0.064 | 0.174 |

Bounding pipe is **vector at 0.713 mean — below the 80% bar**. The 0.358 MTE2 (load)
ratio is the cost of re-reading the two `[B,HV,T,128,128]` GM tapes (`tape_d`,
`back_tape`, ~134 MB each at s512) across five of the eight stages. The five costliest
stages are exactly those that reload both tapes.

## v2 — fused reduction kernel (2026-10-05)

Collapsed the five per-token reduction kernels into one `reduce` kernel that loads
each `tape_d[t]`/`back_tape[t]` slice once and derives the shared `sp8`/`(k.d)_K`/
`(back.k)_K` once; dk emitted already combined. Chain 8 kernels -> 4.

| metric | v1 baseline | v2 |
|---|---|---|
| device time / call | 9440 µs | **6643 µs (-30%)** |
| host wall / call | 9507 µs | 6643 µs |
| correctness vs fla Triton | ~1e-7 | identical (bit-for-bit same rel-L2) |

v2 per-kernel device (per-call): reduce 4310, reverse_rec 1381, replay 925,
group_reduce 27. The five reductions (7102 µs total in v1) became one at 4310 µs.

`reduce` kernel pipe ratios (mean/max): aiv_vec 0.450 / 0.870, aiv_mte2 0.354 / 0.600,
aiv_mte3 0.178 / 0.308. Now partly **memory-bound on the tape loads** (128 KB/token),
bounding pipe still under the 80% bar — the remaining cost is the two 134 MB GM tapes
(`tape_d`, `back_tape`) being produced then consumed.

## v3 — tapeless fused reverse scan (2026-10-05)

Folded the reduction work into the reverse recurrence so `back_t` is consumed in-UB;
`back_tape` eliminated. Single runtime loop over t (no C-unroll) keeps compile
bounded. Chain 4 kernels -> 3 (replay, reverse_rec_fused, group_reduce).

| metric | v1 | v2 | v3 |
|---|---|---|---|
| device time / call | 9440 µs | 6643 µs | **5391 µs** |
| vs v1 | — | -30% | **-43%** |
| correctness vs fla Triton | ~1e-7 | ~1e-7 | identical (~1e-7) |

v3 per-kernel device (per-call): reverse_rec_fused 4457, replay 917, group_reduce 18.
`reverse_rec_fused` pipe ratios (mean/max): **aiv_vec 0.956 / 0.959**, aiv_mte2 0.052,
aiv_mte3 0.050, aiv_scalar 0.054. The dominant kernel (83% of device time) is
**vector-bound at 95.6% — above the 80% bar** (mte2 collapsed from v2's 0.354 as the
back_tape round-trip is gone and the remaining tape_d load hides under the vec work).

Host wall: a2 backward 5403 µs/call vs fla Triton fwd+bwd 9440 µs.

## Cross-shape correctness (v3)

| shape | dq | dk | dv | dg | dbeta |
|---|---|---|---|---|---|
| s512  | 2.66e-7 | 2.83e-7 | 2.84e-7 | 4.28e-7 | 3.07e-7 |
| s2048 | 2.64e-7 | 2.82e-7 | 2.81e-7 | 4.80e-7 | 3.12e-7 |

## Verdict vs the fla Triton benchmark (the honest comparison)

fla Triton device (s512): fwd 2354 µs, fwd+bwd 3270 µs -> **backward 915 µs**.
a2 v3 backward device: **5391 µs -> 5.89x slower than fla's Triton backward.**
At s2048 the wall gap widens (a2 21821 µs vs fla's whole fwd+bwd 11122 µs): the a2
cost scales **linearly in T** while fla scales sub-linearly.

**The optimization bar (>=80% utilization) is met — the dominant kernel is 95.6%
vector-bound — and v3 is 43% faster than the a2 baseline. But the "beat the
benchmark" bar is NOT met, and implementation tuning cannot close the gap, because it
is algorithmic:** this a2 `gdn_chunk_bwd` is a **sequential per-token recurrence**
(O(T) full-[128,128] vector reductions per token), whereas fla's Triton
`chunk_gated_delta_rule` backward is **chunk-parallel** (O(T/C) matmul-style work).
v3 is a near-optimal implementation of the wrong-for-this-benchmark algorithm.

To actually beat fla would require re-deriving the backward as a chunk-parallel
(matmul/cube) algorithm — a separate, much larger effort, and constrained on b3 by
A2-01 (no cube L0C DMAs). That is a design-level change, not a kernel-tuning one.

Collapse the 8-kernel decomposition toward the DESIGN.md lean structure (replay/
checkpoints → a single reverse scan holding `back` and `d_t` in UB, two live
`[128,128]` tiles) so the tapes are produced and consumed in-kernel instead of
round-tripped through GM. Expected to cut the MTE2 load ratio and lift vector
utilization past 80%, and to remove most of the ~9.4 ms (the redundant tape I/O and
the per-stage recompute of `dz`/`d_t`). Re-measure device time + pipe ratios after.
