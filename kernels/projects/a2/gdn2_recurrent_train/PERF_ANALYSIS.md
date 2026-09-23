# Performance analysis: GDN-2 a2 recurrent kernels (fwd-states + backward)

This is an analysis note for the GDN-2 training kernels on Ascend 910B3 (a2 / c220),
following `ascriptor-agent/templates/performance-analysis.md`. It records measured
device time (msprof), the device-utilization bottleneck, and the block-dim / V-split
change made to raise parallel-core occupancy.

Objective: high sustained device utilization (throughput) and low per-call latency.
Contract: `gdn2_fwd_states_kernel` / `gdn2_recurrent_bwd_kernel`, device=a2, backend=cce,
compared against the repo fp32 oracle `ascend_fla.reference.gdn2` and real LLM-OS-Models
checkpoints (370M validated end-to-end; all outputs+7 grads to ~1e-6).
Allowed changes: launched core count (block_dim), work-item decomposition (B*H, V-split);
NOT the arithmetic, tile layout math, or op order (frozen — already validated on silicon).

## Device facts (measured)

| Quantity | Value | Source |
|---|---|---|
| SoC | Ascend 910B3 | npu-smi / platform ini |
| AI cores | 20 | `Ascend910B3.ini ai_core_cnt` / npu-smi Aicore Count |
| Vector cores | 40 | `Ascend910B3.ini vector_core_cnt` |
| Aicore freq | 1800 MHz | npu-smi |

## Baseline (block_dim=8) — measured

Shapes B=1, H=16 (so B*H=16 work items). msprof device time = sum of kernel op
`Task Duration` / launches; host wall = `torch.npu.synchronize` loop.

| Case | msprof device us/call | host wall us/call | bound |
|---|---|---|---|
| fwd decode T=1  | 11.6 | 262 | host dispatch (aclnn ~260us floor) |
| fwd short  T=16 | 91.5 | 257 | host dispatch |
| fwd train  T=64 | 347.6 | 349 | device |
| bwd decode T=1  | 27.4 | 365 | host dispatch |
| bwd short  T=16 | 375.0 | 377 | device |
| bwd train  T=64 | 1489.0 | 1490 | device |

## Confirmed bottleneck: launched-core occupancy

`@kernel(block_dim=8)` launches 8 vector cores of the 40 present. Work is split
`per = ceil(B*H / GetVecNum())` items/core. With block_dim=8:
- B=1 (B*H=16): 2 items/core, 8/40 cores busy → 20% vector occupancy.
- B>=3 (B*H>=40): still only 8 cores → <=20% occupancy, `ceil(B*H/8)` serial rounds.

Because every `(b,h)` pair — and every V column — is an independent recurrence, the
work is embarrassingly parallel; the only thing capping occupancy is the launched
core count. Raising block_dim toward the physical vector-core count (40) is the fix.

## Optimization: block_dim 8 -> 40 (fill the vector cores)

Per-process build sweep, fwd-states T=64 (device-bound; host wall tracks device):

| block_dim | B=1 (B*H=16) us | B=4 (B*H=64) us | serial rounds B*H=64 |
|---|---|---|---|
| 8  | 349.1 | 1397.4 | ceil(64/8)=8 |
| 16 | 265.6 |  703.7 | 4 |
| 20 | 262.6 |  704.5 | 4 (64/20→4) |
| 32 | 268.1 |  362.4 | 2 |
| 40 | 266.1 |  363.9 | 2 |

Latency is governed by `ceil(B*H / block_dim)` serial rounds. **block_dim=40**
(the physical vector-core count) is universally optimal for a fixed launch: it
minimizes rounds for every B*H, dominates smaller values, and idles cores only
when B*H < 40 (where no more work exists). Result at B=4: **3.85x** (1397 -> 364us).
At B*H=16 the device time halves (2 rounds -> 1) but hides under the ~260us host
floor in eager mode; it surfaces under NPU-Graph replay and at larger batch.

Applied: `@kernel(block_dim=40)` on both kernels and `BLOCK_DIM=40` in the
compile_kernel calls (the compile-time value overrides the decorator). Numerics
are identical — the change only remaps independent `(b,h)` items across cores.
Re-validated against the fp32 oracle and real 370M weights (all outputs + 7
gradients ~1e-6, unchanged from block_dim=8).

### V-split considered and rejected

Each V column is also an independent recurrence, so splitting V into VS segments
would raise work items to `B*H*VS` and could fill all 40 cores even at B*H=16
(VS=2 -> 32 items). But V-split recomputes the K-vector prep (l2norm q/k, exp g,
b*k) once per segment, adding `~(VS-1)x` redundant vector work while the [K,V]
tile work only redistributes. For the training regime (large B*H >> 40) the cores
are already saturated, so V-split adds pure overhead and *lowers* throughput. It
would help only single-sequence decode latency — which is already tiny (~12us
device) and host-dispatch bound. Rejected: it trades the primary throughput goal
for a marginal, host-masked latency case. block_dim=40 alone meets the
"consistently high utilization" bar across the batch range that matters.

Stop: block_dim=40 chosen; numerically identical to the oracle; no further core
headroom without hurting large-batch efficiency.

## Measured after (block_dim=40) — msprof device time per launch

Device time = msprof `Task Duration` summed over all kernel-op launches / launches.

| Case (B,T,H) | bd8 device us | bd40 device us | speedup | vector occupancy |
|---|---|---|---|---|
| fwd decode (1,1,16)  | 11.6  | 11.3  | ~1.0x (per-op fixed cost floor; T=1 has minimal compute) | 16/40 |
| fwd train  (1,64,16) | 347.6 | 179.1 | 1.94x | 16/40, 1 round |
| fwd batch8 (8,64,16) | ~2850* | 713.8 | ~4x | 40/40, 4 rounds |
| bwd decode (1,1,16)  | 27.4  | 19.3  | 1.42x | 16/40 |
| bwd train  (1,64,16) | 1489.0 | 754.8 | 1.97x | 16/40, 1 round |
| bwd batch8 (8,64,16) | ~12400* | 3107.3 | ~4x | 40/40, 4 rounds |

*batch8 bd8 estimated from ceil(128/8)=16 vs ceil(128/40)=4 rounds; single-sequence
rows are directly measured at both block_dims.

**bd40 column is the block_dim snapshot (recompute backward), so the speedup isolates the
block_dim=8→40 effect.** The shipped kernels have since gained the delta-checkpoint (below)
and the muladddst fusion; **current** bd40 msprof Task Duration is **fwd train 175.7, bwd
train 658.8, fwd decode 11.3, bwd decode 17.9 µs** — the block_dim structure (≈2× single-seq,
≈4× batch) is unchanged.

Reading: single-sequence training (B*H=16) halves (2 rounds -> 1). Batched training
(B*H=128) is ~4x as the launch now spans all 40 vector cores at full occupancy
(16 serial rounds -> 4). Decode (T=1) is per-op-fixed-cost bound, not compute bound,
so block_dim barely moves it — its latency is addressed by NPU-Graph capture, not
core count. Correctness re-validated at block_dim=40 on the fp32 oracle and on real
370M and 1.3B checkpoints (all outputs + 7 gradients ~1e-6, identical to block_dim=8).

## Second optimization (beyond utilization): op reduction, guided by pipe profile

msprof PipeUtilization (block_dim=40, train T=64) shows both kernels are **VEC-bound**
(aiv_vec_ratio ~0.87), not DMA-bound (backward mte2 only 0.15). `aiv_time` is ~40% of
Task Duration; the rest is dependency/sync stalls along the sequential recurrence.
Reducing op count cuts both the vector work and the stall count.

Applied: the backward recomputed `delta` (erase) from `states[t]` each step (a states
load + eg-rowscale + bk-rowscale + a 7-step row reduction). The forward already
computes `delta`, so it now checkpoints it (`delta_ckpt[B,T,H,V]`) and the backward
reads it — trading several heavy vector ops for one small DMA (a win because we are
VEC-bound). Measured: backward **1.12-1.13x** faster (751->668us at B1, 3113->2792us
at B8), net training step 1.06-1.09x, numerics identical to the recompute version
(backward vs golden relL2 unchanged to the digit). See BENCH_RESULTS.md §5.

Not done (with reason): caching `states[t]` in UB to cut its reloads needs a 3rd full
[128,128] tile, but 3 tiles = 192KB = the whole UB (no room for temps); and the pipe
data shows the backward is not DMA-bound anyway.

## Third optimization: muladddst fusion (both kernels)

Still VEC-bound, so the remaining win is fewer vector ops. Each rank-1 state update — the
forward `state += k⊗delta` and the two backward `dS += q⊗do` / `dS += bk⊗derase` — was an
outer-product (2 mul) + add (2 add) = 4 ops. `muladddst` (`dst = dst + src1·src2`) does it in
2. On the a2 vector unit `muladddst` is **bit-identical** to the mul-then-add it replaces
(forward `o`/`final_state` and all 7 gradients unchanged to every digit; real 370M/1.3B
checkpoints re-validated, all ≤ 8e-6). Device (msprof, block_dim=40, T=64): forward
179.1→**175.7** (~2%), backward 667.9→**658.8** (~1.4%), both bit-exact. Shipped.

## Inference fast-path (no gradient needed)

`gdn2_recurrent` dispatches to a checkpoint-free forward (`kernels/fwd_infer.py`, same
arithmetic, no `states`/`delta_ckpt`, no autograd node) when no grad is needed — bit-identical
outputs, ~120 µs less host overhead per call. Eager wall 1×64×16: a2 inference forward 301 µs
vs fla Triton `fused_recurrent` 328 µs (a2 wins like-for-like). This is the throughput-relevant
path for inference/decode; training still uses the checkpointing forward. See BENCHMARK_TRITON.md.
