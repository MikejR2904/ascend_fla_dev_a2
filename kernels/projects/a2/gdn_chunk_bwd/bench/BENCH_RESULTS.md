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

## Optimization target

Collapse the 8-kernel decomposition toward the DESIGN.md lean structure (replay/
checkpoints → a single reverse scan holding `back` and `d_t` in UB, two live
`[128,128]` tiles) so the tapes are produced and consumed in-kernel instead of
round-tripped through GM. Expected to cut the MTE2 load ratio and lift vector
utilization past 80%, and to remove most of the ~9.4 ms (the redundant tape I/O and
the per-stage recompute of `dz`/`d_t`). Re-measure device time + pipe ratios after.
