# GDN-2 a2 kernels — full benchmark results

All measured on Ascend **910B3** (a2 / c220; 20 AI cores, 40 vector cores, UB 192KB,
AIV 1800 MHz), CANN 9.2.0-beta.1, torch_npu 2.10. Kernels: `gdn2_fwd_states_kernel`
(training forward + per-step state checkpoints) and `gdn2_recurrent_bwd_kernel`
(reverse recurrence), device=a2, backend=cce.

Shapes use B=batch, T=sequence, H=16 heads, K=V=128. `B*H` = independent recurrence
work items mapped across vector cores (`ceil(B*H/block_dim)` serial rounds).

## 1. block_dim sweep — host wall, fwd-states T=64 (device-bound)

Per-process builds (one block_dim per process). us/call.

| block_dim | B=1 (B*H=16) | B=4 (B*H=64) |
|---|---|---|
| 8  | 349.1 | 1397.4 |
| 16 | 265.6 |  703.7 |
| 20 | 262.6 |  704.5 |
| 32 | 268.1 |  362.4 |
| 40 | 266.1 |  363.9 |

Chosen: **block_dim=40** (fills all 40 vector cores; minimizes serial rounds at every B*H).

## 2. msprof device time per launch — block_dim=8 vs 40 (block_dim snapshot)

Device time = msprof `Task Duration` summed over kernel-op launches / launches. This table is the
**block_dim comparison snapshot** — both columns taken with the then-current *recompute* backward, so
the speedup column isolates the block_dim=8→40 effect. Current shipped numbers are below the table.

| Case (B,T,H) | bd8 device us | bd40 device us | block_dim speedup |
|---|---|---|---|
| fwd decode (1,1,16)  | 11.6  | 11.3  | ~1.0x |
| fwd train  (1,64,16) | 347.6 | 179.1 | 1.94x |
| fwd batch8 (8,64,16) | ~2850* | 713.8 | ~4x |
| bwd decode (1,1,16)  | 27.4  | 19.3  | 1.42x |
| bwd train  (1,64,16) | 1489.0 | 754.8 | 1.97x |
| bwd batch8 (8,64,16) | ~12400* | 3107.3 | ~4x |

*batch8 bd8 estimated from serial-round ratio (directly measured at bd8 for B<=4).

**Current shipped kernels** (delta-checkpoint §5 + muladddst fusion §7), block_dim=40, msprof
Task Duration: **fwd train 175.7, bwd train 658.8, fwd decode 11.3, bwd decode 17.9 µs**. The
delta-checkpoint cut the backward from the 754.8 above to ~668, and the muladddst fusion trimmed it
further to 658.8 (fwd 179.1→175.7); the ≈2× single-sequence / ≈4× batch block_dim structure is
unchanged. (batch8 not re-profiled here — device 0 was memory-contended; it scales with the per-step.)

## 3. Pipe utilization (msprof PipeUtilization, block_dim=40, train T=64, current kernels)

Current shipped kernels (delta-checkpoint §5 + muladddst fusion §7):

| pipe | fwd train | bwd train |
|---|---|---|
| Task Duration (us)   | 175.7 | 658.8 |
| aiv_time (us)        | 71.9  | 265.2 |
| aiv_vec_ratio        | 0.885 | 0.895 |
| aiv_scalar_ratio     | 0.118 | 0.069 |
| aiv_mte2_ratio (DMA in)  | 0.075 | 0.133 |
| aiv_mte3_ratio (DMA out) | 0.203 | 0.071 |
| cube_utilization     | 0 (pure vector) | 0 |

Both kernels are **VEC-bound** (vec_ratio ~0.87) — not DMA-bound. `aiv_time` is only
~40% of Task Duration, the remainder being dependency/sync stalls along the sequential
recurrence; reducing op count cuts both. This motivated the v2 delta-checkpoint (§5).

## 4. Comparison vs the model's shipped baseline (torch / torch_npu recurrence)

`ascend_fla.reference.gdn2` (= `core_backend="torch"`), same NPU, same inputs. us/call.

| shape | torch_npu FWD | a2-kernel FWD | FWD speedup | fwd correctness relL2 | torch_npu FWD+BWD | a2 FWD+BWD | FWD+BWD speedup |
|---|---|---|---|---|---|---|---|
| decode (1,1,16) | 702.3 | 395.1 | 1.78x | 3.9e-6 | (T=1 bwd n/a in training) | | |
| train  (1,64,16) | 39057.5 | 400.8 | **97.5x** | 5.2e-6 | 188761.2 | 27997.6 | **6.7x** |
| batch8 (8,64,16) | 39893.1 | 720.9 | **55.3x** | 5.3e-6 | 342752.4 | 181304.0 | **1.9x** |

The a2 kernels dominate the torch recurrence the model ships (55–97x forward). The
FWD+BWD wall time includes PyTorch autograd + per-iter H2D of fresh inputs (same for
both sides), so the 1.9–6.7x is the honest end-to-end training-step speedup.

## 5. Op-reduction optimization (delta-checkpoint) — now the primary kernels

Both kernels are VEC-bound (§3), so the win is removing vector ops from the critical
path. The backward recomputed `delta` (= erase) every step from `states[t]`: a states
load + eg-rowscale + bk-rowscale + a 7-step row reduction. Since the forward already
computes `delta`, the forward now checkpoints it and the backward reads it directly.

Measured host-wall (block_dim=40), delta-checkpoint vs the earlier recompute version:

| shape | FWD recompute | FWD delta-ckpt | BWD recompute | BWD delta-ckpt | BWD speedup | net fwd+bwd |
|---|---|---|---|---|---|---|
| B1 T64 H16 | 274.7 | 301.2 | 751.7 | 667.9 | **1.13x** | 1.06x |
| B8 T64 H16 | 716.4 | 716.5 | 3112.9 | 2791.6 | **1.12x** | 1.09x |

The forward pays a small delta-write cost (visible only at B=1); the backward — the
heavy VEC-bound kernel — drops ~12-13%, and the net training step (forward+backward,
always run together) is 1.06-1.09x faster. Numerics are identical to the recompute
version (backward vs golden: same relL2 to the digit — 1.2e-5 / 5.5e-6 / 7.0e-6 /
8.1e-6 at T=16/64/128 / B2 T64). This is now the shipped implementation.

Rejected alternatives: caching `states[t]` in UB to cut the 3 reloads is impossible —
3 full [128,128] tiles = 192KB = the entire UB, no room for temps (pipe data also
shows the backward is not DMA-bound: mte2 only 15%). A V-split to fill idle cores at
B*H<40 adds per-segment recompute that lowers large-batch throughput (see PERF_ANALYSIS).

## 6. Correctness summary

- Backward vs analytic golden (autograd-checked): B1 T1/2/4/16/64/128, B2 T64 — worst relL2 ~1e-5, PASS.
- Real 370M weights (layers 0,8): all outputs + 7 grads <= 8.1e-6.
- Real 1.3B weights (layers 0,9,17): all outputs + 7 grads <= 7.0e-6.
- block_dim=8 and block_dim=40 produce identical numerics.

### T=1 backward fix

The earlier (erase-recompute) backward raised a vector-core exception at T=1 — the
single-step reverse loop tripped a fault in the `states[t]` reload + eg/bk rowscale +
row-reduce block. The delta-checkpoint optimization (§5) removes that whole block, so
T=1 now runs cleanly and correctly (autograd fwd+bwd worst grad relL2 2.6e-6 at T=1,
4.2e-6 at T=2). T=1/T=2 are covered by `test_backward.py` and `validation/verify_t1.py`.

## 7. muladddst fusion (op reduction, both kernels)

Still VEC-bound (§3), so the next win is fewer vector ops. Each rank-1 state update
(`state += k⊗delta` in the forward, `dS += q⊗do` and `dS += bk⊗derase` in the backward)
was an outer-product (2 mul) + add (2 add) = 4 ops; `muladddst` (`dst = dst + src1·src2`)
does it in 2. On the a2 vector unit `muladddst` is **bit-identical** to the mul-then-add it
replaces — forward `o`/`final_state` and all 7 gradients unchanged to every digit; real
370M/1.3B checkpoints re-validated (all ≤ 8e-6). Device (msprof Task Duration, block_dim=40,
T=64): forward 179.1→**175.7** (~2%), backward 667.9→**658.8** (~1.4%). This is the shipped
implementation.

## 8. Inference fast-path (no-grad forward)

`gdn2_recurrent` dispatches to a checkpoint-free kernel (`kernels/fwd_infer.py`, same
arithmetic, no `states`/`delta_ckpt`) when no gradient is needed — bit-identical `o`/
`final_state`, but no 68 MB checkpoint allocation and no autograd node, cutting ~120 µs of
host-side per-call overhead. Eager wall (1×64×16): a2 inference forward **301 µs** vs fla's
Triton `fused_recurrent` **328 µs** — the a2 kernel wins the eager wall like-for-like. See
`BENCHMARK_TRITON.md`.
