# GDN-2 a2 kernel vs fla Triton — benchmark comparison (Ascend 910B3)

Same operation (GDN-2 recurrence, `update = w·v − erase`, per-K-channel gate), same inputs, on the
same NPU. fla's Triton kernels run on the Ascend NPU via the integrated triton backend (no GPU).
Correctness is cross-checked against `fla.ops.gdn2.naive_recurrent_gdn2`; the a2 kernel and fla's
naive are the same op to **8.1e-5** rel-L2 (fp32 recurrence; fla's own chunk/recur match naive to
~2e-7). Conventions aligned: q/k are l2-normalized externally, `scale = 1/√128`; the a2 kernel takes
`q·scale` (it does no preprocessing), fla takes `q,k` with `scale=` (fla scales q internally).

**Fairness note.** The a2 kernel always does the q/k l2-normalization inside the recurrence, so the
comparison must turn fla's in-kernel l2norm **on** (`use_qk_l2norm_in_kernel=True`) with **raw** q/k
fed to both — otherwise the a2 kernel pays for a normalization that fla skips. Both sides run **eager**
(no NPU-Graph on either), so wall and device numbers are like-for-like.

## Device time (µs/call, msprof Task Duration, forward, 1×64×16, both l2-normalizing)

| kernel | device µs/call | note |
|---|---|---|
| **a2 `gdn2_recurrent` (fwd)** | **175.7** | msprof Task Duration, block_dim=40, T=64, post-`muladddst` fusion (76 vec ops/step); **also checkpoints all T states for the backward** (extra work fla's inference kernel skips); VEC-bound aiv_vec_ratio 0.885 |
| fla_recur (Triton `fused_recurrent_gdn2`) | 206.6 | prior run, kernel unchanged; 220 ops/100 calls; l2norm is a separate pass (+43µs over the no-norm 163.7) |
| fla_chunk (Triton `chunk_gdn2`) | ≫ | chunk path far slower on this recurrence shape |

## Wall time (µs/call, host `torch.npu.synchronize` loop, forward, both l2-normalizing)

| shape (B,T,H) | a2 kernel | fla_recur (Triton) | fla_chunk (Triton) |
|---|---|---|---|
| (1, 64, 16)  | 432.1 | 352.1 | 4760.7 |
| (1, 128, 16) | 451.5 | 374.7 | 5169.0 |
| (2, 64, 16)  | 442.6 | 374.3 | 5093.4 |

Correctness (rel-L2 vs `naive_recurrent_gdn2`): a2 **5.3e-6**, fla_recur 2.3e-7, fla_chunk 4.1e-7.

_These a2 wall numbers are the training-forward path (checkpoints + autograd) — measured before the
inference fast-path was added. A no-grad `gdn2_recurrent` call now dispatches to the checkpoint-free
kernel at **301 µs** (1×64×16), beating fla_recur's 328 µs; see the decomposition below._

## NPU-Graph (dispatch removed, fair: both graph-captured), wall µs/call, 1×64×16

| kernel | eager wall | NPU-Graph wall | correctness vs eager |
|---|---|---|---|
| **a2 `gdn2_recurrent`** | 414 | **200.1** | **relL2 0.00e+00 (bit-identical)** |
| fla_recur (Triton) | 349 | 202.0 | relL2 0.00e+00 |

NPU-Graph capture is **bit-for-bit identical** (no arithmetic change) — it only removes the per-call
host overhead. It collapses both kernels to near their device time, so the eager-mode wall gap
(the a2's training-path checkpoint allocation + autograd node, decomposed below) disappears:
graph-vs-graph the a2 kernel is 200.1 vs 202.0 µs — a hair ahead, consistent with its device-time win.

## Full training step (forward + backward) — who wins

The tables above are the **forward**. For training you also pay the backward. Two facts decide it:

1. **fla's `fused_recurrent_gdn2` is not trainable.** Its output comes back with `requires_grad=False`
   / `grad_fn=None` — it is an inference/decode kernel and `.backward()` raises. So fla's *only*
   trainable GDN-2 path is the **chunk** kernel. The a2 op is differentiable end-to-end (its own
   reverse-recurrence backward kernel; all 7 grads match torch autograd to ≤1e-5).

2. **On the trainable comparison, a2 wins ~6×.** Eager wall per fwd+bwd step (grads zeroed each step,
   both l2-normalizing in-kernel, same harness):

   | shape (B,T,H) | a2 fwd+bwd | fla_chunk fwd+bwd | fla_recur |
   |---|---|---|---|
   | (1, 64, 16)  | **1709 µs** | 9906 µs  | untrainable |
   | (1, 128, 16) | **1738 µs** | 10739 µs | untrainable |
   | (2, 64, 16)  | **1592 µs** | 10199 µs | untrainable |

   Host-free (NPU-Graph, driving the a2 kernels directly) the a2 training step is **841 µs** at T=64
   (fwd 182 + bwd 664) and **1659 µs** at T=128 (fwd 349 + bwd 1317) — the backward is ~3.6× the
   forward. Both the forward and the backward fuse their rank-1 state updates into `muladddst`
   (bit-identical to the outer-then-add; validated on synthetic inputs and on the real gdn2-370m /
   gdn2-1.3b checkpoints, all 7 grads ≤ 8e-6). fla's chunk kernel is a *parallel-over-chunks*
   algorithm built for long sequences; on
   these short recurrent shapes its overhead dominates (~10 ms/step, device-bound), so the a2
   sequential recurrence is the right tool and wins decisively.

**Verdict: for full training on a2 at these shapes, the a2 op wins — ~6× on wall — and is the only
recurrent-form option that can train at all.**

## Why fla's eager wall looked lower — decomposed (1×64×16)

The earlier "fla wins on wall" line blamed an "aclnn dispatch floor". That was wrong. Decomposing
the a2 eager wall (each row is the mean of 100 timed calls after 20 warmups) shows the a2 **kernel**
is not the problem — the *training wrapper* is:

| stage | µs/call | what it is |
|---|---|---|
| a2 pure kernel dispatch (`_FWD`, pre-allocated buffers) | **281** | aclnn launch + 197 µs device compute |
| a2 `.contiguous().float()` ×7 inputs | 8 | near no-op (inputs already fp32/contiguous) |
| a2 alloc 4 output buffers | 32 | dominated by the **68 MB** per-step `states` checkpoint |
| a2 **training** forward, full (`gdn2_recurrent`, grad) | **425** | dispatch + alloc + `autograd.Function` + `save_for_backward`(9 tensors) |
| a2 **inference** forward (checkpoint-free, no autograd) | **301** | dispatch + 2 small allocs only |
| fla_recur (Triton `fused_recurrent`, inference) | **328** | its full eager call |

Two facts follow. **(1)** The a2 kernel dispatch alone (281 µs) is *faster* than fla's full eager
call (328 µs) — aclnn is not the bottleneck. **(2)** The a2 op's higher wall came entirely from it
running the **training** path on every call: allocating the 68 MB `states`/`delta` checkpoints and
building an autograd node with a 9-tensor `save_for_backward`, ~120 µs of host-side work fla's
*inference* kernel never does. Comparing like for like — both inference — the a2 checkpoint-free
fast-path **wins the eager wall too: 301 vs 328 µs**, with `o`/`final_state` bit-identical
(`torch.equal`) to the training forward.

`gdn2_recurrent` now dispatches to this checkpoint-free kernel (`kernels/fwd_infer.py`) whenever no
gradient is needed, and to the checkpointing training forward otherwise.

## Reading

- **On device time (the dispatch-agnostic, fair kernel comparison), the a2 kernel WINS: 175.7 vs
  206.6 µs** (a2 forward re-measured post-fusion; fla's kernel unchanged) — while doing *more* work
  (per-step state checkpointing for the backward). fla's fused-recurrent Triton pays ~43µs for its
  separate l2norm pass; the a2 kernel fuses it into the recurrence.
- The a2 kernel **beats fla's chunk Triton ~8-11×** on this recurrent shape.
- **On eager wall, like for like (both inference), the a2 kernel also WINS: 301 vs 328 µs.** The
  training forward's higher wall (425 µs) is host-side checkpoint + autograd overhead, not the
  kernel — and NPU-Graph capture, which elides all host overhead, confirms it (200.1 vs 202.0 µs).
- Remaining stretch goal: push VEC utilization from 0.885 to **> 90%** (needs op-reduction beyond the
  bit-exact op stream) — the `muladddst` fusion already trimmed 2 vec ops/step in the forward and 4 in
  the backward at bit-exact accuracy (~2% device-time win; see STALL_ANALYSIS.md §4).

_Measured on Ascend 910B3, CANN 9.2.0-beta.1, torch_npu 2.10, ascriptor library 90cfcdc / kernels
b3b3f9c; fla pinned commit e52dbc0 (0.6.0) via the integrated triton-ascend backend._
