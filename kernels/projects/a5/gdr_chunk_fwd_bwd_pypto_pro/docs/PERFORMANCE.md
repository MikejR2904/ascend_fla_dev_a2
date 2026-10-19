# Performance

Measured on an Ascend 950PR, 28 cube and 56 vector cores, DAV_3510, CANN
9.1.0-beta.1. Every number below was produced by the code in this directory.

## How it was measured

One `msprof` pass per figure, single metric group `PipeUtilization`, over a
process that runs three untimed warm-up iterations and then five timed ones. The
warm-up matters: the first launch of a kernel includes its JIT compilation. The
figures are medians of the warm occurrences of each `Op Name`, taken from
`op_summary`'s `Task Duration`, which is device time.

Device time and host wall time are reported separately and never averaged
together. Nothing here is a simulator estimate.

## The composed training pipeline

B=1, T=2048, H=8, K=V=128 — the production shape. `aic_mac` is the fraction of
the kernel's window in which the cube's multiply-accumulate pipe is busy;
`cube %` is `cube_utilization`, which is core occupancy, not efficiency.

| # | kernel | µs | share | blocks | `aic_mac` | cube % |
|--:|---|--:|--:|--:|--:|--:|
| 1 | `gdr_fwd_train_wy` | 132.29 | 15% | 28 | 0.502 | 89.1 |
| 2 | `gdr_fwd_train_scan` | 182.71 | 20% | 24 | 0.796 | 84.1 |
| 3 | `gdr_bwd_dv_local` | 37.35 | 4% | 28 | 0.705 | 87.9 |
| 4 | `gdr_bwd_reverse_scan` | 216.99 | 24% | 16 | 0.570 | 56.7 |
| 5 | `gdr_bwd_dqkw` | 161.02 | 18% | 28 | 0.716 | 89.0 |
| 6 | `gdr_bwd_wy_matmuls` | 111.99 | 12% | 28 | 0.812 | 88.0 |
| 7 | `gdr_bwd_wy_reduce` | 41.64 | 5% | 56 | — | pure vector |
| 8 | `gdr_bwd_finalize` | 27.99 | 3% | 56 | — | pure vector |
| | **total** | **911.98** | | | | |

The load is spread: the top five kernels are each 12–24% of the total, so
halving any single one buys at most an eighth of the pipeline.

`gdr_bwd_reverse_scan` launches 16 of the 28 cube blocks, and that is a property
of this shape rather than of the kernel. Its independent work is
`B * H * ceil(V/64)` items — 16 at `H=8` — and every refinement of the item space
doubles a power-of-two axis, so the reachable counts are 8, 16, 32, … and none
lands in (16, 28]. At `H=16` there are 32 items and it launches all 28 blocks.

The same run's host wall time is **1728 µs**, because the timing mode brackets
each of the eight launches with its own `torch.npu.synchronize()`. The gap is
per-launch dispatch, not device idle. Benchmark the path end to end and the
dispatch is paid once per launch; do not use wall clock to judge a kernel.

## How it scales

One axis at a time from the reference shape, same method:

| T | N | total µs | µs per chunk |
|--:|--:|--:|--:|
| 256 | 4 | 203.7 | 50.9 |
| 512 | 8 | 311.8 | 39.0 |
| 1024 | 16 | 533.3 | 33.3 |
| 2048 | 32 | 1052.9 | 32.9 |
| 4096 | 64 | 2079.6 | 32.5 |

Per-chunk cost flattens from T=1024; below that, launch and barrier overhead does
not amortise. T=256 costs 56% more per chunk than T=4096. There is no
super-linear term anywhere through T=4096.

At **B=1, T=16384, H=16, K=V=128** — sixteen times the work of the reference —
the pipeline takes **14.24 ms**, against the 14.59 ms that scaling the reference
linearly would predict. Scaling is slightly *sub-linear*: per chunk and head the
cost falls from 3.563 to 3.477 µs. Six of the eight kernels clear 93% occupancy
at that shape, against two of eight at the reference shape, because larger shapes
fill the cores better. The reference shape is therefore the pessimistic one; do
not extrapolate from it linearly and assume the result is conservative.

Correctness has not been checked at that shape — see `VALIDATION.md`.

## Forward only

Training forward is kernels 1 and 2. Measured on its own at B=1, T=16384, H=16,
K=V=128: `gdr_fwd_train_wy` 2047.4 µs plus `gdr_fwd_train_scan` 2825.9 µs =
**4873.3 µs** of device time, with a host wall of 5391 µs.

The forward costs the same alone as it does inside the full pipeline (0.09%
apart), so the two halves do not interfere through L2. Forward is 34.2% of
training; backward is 1.92 times forward.

## Against the machine

The chunked forward's arithmetic requirement is countable. Per `(batch, head,
chunk)` at C=64, K=V=128, summing the nine matmuls of the chunk — the two
triangular Gram products, the inverse, the two WY applications, the two output
terms and the state update — gives 5,810,858 multiply-accumulates. At
`B*H*N = 4096` items that is **23.80 GMAC** for the whole forward.

The machine's peaks were measured with a probe of back-to-back
`[64,64] x [64,64]` MMADs: 623 ns per MMAD in float32 and 42 ns in bfloat16,
which is 11.78 and 174.76 TMAC/s across 28 cores. So this shape's floors are
**2020 µs** in float32 and 136 µs in bfloat16.

| | |
|---|--:|
| measured forward | 4873 µs |
| float32 arithmetic floor | 2020 µs |
| **measured / floor** | **2.41x** |
| MACs actually issued | 38.40 GMAC |
| **issued / required** | **1.61x** |
| fraction of the window with the MAC pipe busy | 67% |

The 2.41x decomposes into three independent factors: 1.61x redundant MAC issue
(tile padding, the two triangular products that compute a full square and
discard half, and the transposed duplicates the backward needs), 33% of the
window not MAC-busy, and the dtype itself. The dtype term is the largest by an
order of magnitude: even with perfect scheduling and no redundant work, float32
caps this shape at 2020 µs where bfloat16 caps it at 136 µs.

## Against the framework

`fla`'s `naive_chunk_gated_delta_rule` is pure torch, so it runs under
`torch_npu` on the same part. Synchronised wall time, same session,
float32 inputs, `scale = 1/sqrt(K)`:

| shape | `fla` naive_chunk | ours, wall | speedup | ours, device |
|---|--:|--:|--:|--:|
| B=1, T=2048, H=8, K=V=128 | 9161.3 µs | 695.1 µs | **13.2x** | 316.6 µs |
| B=1, T=16384, H=16, K=V=128 | 32361.7 µs | 5391 µs | **6.0x** | 4873.3 µs |

`fla`'s samples at the large shape were 31422 / 32362 / 33488 µs; ours are
medians of 21 repeats at the production shape and 5 at the large one.

## What these numbers do and do not license

Four ways they can be overread:

1. **The framework reference is a *naive* reference.** Its triangular inverse is
   a 63-iteration Python loop doing in-place slice assignment, so much of its
   cost is eager-mode launch overhead rather than arithmetic. Beating it is
   necessary and nowhere near sufficient; a competently written torch_npu
   composition with a batched inverse would land far below it. The arithmetic
   floor is the stronger comparison precisely because it does not depend on how
   good somebody else's reference is.
2. **The ratio shrinks with size**, 13.2x to 6.0x, as the reference's fixed
   overhead amortises. Quote the shape with the number, never the number alone.
3. **The reference is forward-only and emits no training intermediates.** Our
   forward emits seventeen tensors, several of which exist only to feed the
   backward, so on that axis it is being charged for strictly more work.
4. **Our own host overhead is large at small shapes**: 695.1 µs wall against
   316.6 µs device is 54% dispatch at the production shape, falling to 10% at
   the large shape. Wall-to-wall is therefore the conservative ratio to publish,
   and at small shapes dispatch is a bigger lever than anything inside the
   kernels.

And one about occupancy: `cube_utilization` measures how many cores are busy,
not how well. Multiplying it by `aic_mac` gives the fraction of the machine's
multiply-accumulate capability actually in use, which on this pipeline is 33–82%
depending on the kernel. The two figures answer different questions and neither
alone is "utilisation".

**Defensible from the above**: the pipeline runs at 912 µs at the production
shape; the forward is 2.41 times its float32 arithmetic floor on this part and
issues 1.61 times the algorithmic minimum of multiply-accumulates; it is 6.0 to
13.2 times faster than the framework's own reference implementation at the
shapes stated. **Not defensible**: that it is the fastest implementation of this
operator on this part. That would need a tuned competitor to compare against,
and no such kernel exists for this operator family here.
