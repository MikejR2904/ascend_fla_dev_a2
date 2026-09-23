# GDN-2 fused recurrent — training op (a2 / Ascend 910B3)

End-to-end training operator for the GDN-2 (GDN/GDN2-compatible) fused-recurrent
rule, wired as a PyTorch `autograd.Function` over two hand-written Ascend a2
kernels compiled through the `ascend_fla` runtime bridge.

## Recurrence (per token t, per head)

```
q_norm = l2norm(q) * Q_SCALE          k_norm = l2norm(k)
state *= exp(g)[:, None]              # channel-wise (key-side) decay
erase  = (b * k_norm) @ state         # key-side erase gate b
delta  = w * v - erase                # value-side write gate w
state += k_norm ⊗ delta
out    = q_norm @ state
```

Dual gates — `b` (key side) and `w` (value side) — distinguish GDN-2 from KDA.

## Kernels

| file | kernel | role |
|---|---|---|
| `kernels/fwd_states.py` | `gdn2_fwd_states_kernel` | training forward; emits `states[B,H,T+1,K,V]` (per-step checkpoints) and `delta_ckpt[B,T,H,V]` (per-step `delta = w·v − erase`, so the backward need not recompute erase). `states[0]` = initial state, `states[t+1]` = state after step t. block_dim=40. |
| `kernels/bwd_step.py` | `gdn2_recurrent_bwd_kernel` | reverse recurrence consuming `states` + `delta_ckpt`; emits the 7 gradients (dq,dk,dv,dg,db,dw,dh0). block_dim=40. |
| `kernels/fwd_infer.py` | `gdn2_fwd_infer_kernel` | checkpoint-free inference forward — same arithmetic, no `states`/`delta_ckpt` writes; used by `gdn2_recurrent` when no gradient is needed (bit-identical `o`/`final_state`, no 68 MB checkpoint allocation). block_dim=40. |

Both the forward and backward fuse their rank-1 state updates (`state += k⊗delta`,
`dS += q⊗do`, `dS += bk⊗derase`) into `muladddst` (`dst = dst + src1·src2`), which is
bit-identical to the outer-product-then-add on a2 at half the ops.

Both launch `block_dim=40` (all 40 a2 vector cores). Work is `ceil(B*H/40)` items/core,
so `B*H<=40` runs in a single parallel round and larger batches saturate the device.
See `PERF_ANALYSIS.md` (msprof: single-sequence training ~2x, batch-8 ~4x over the
old block_dim=8) and `REAL_WEIGHTS_VALIDATION.md` (validated on real 370M and 1.3B
LLM-OS-Models checkpoints, all outputs + 7 gradients ~1e-6).

A single-step decode kernel lives in the sibling `gdn2_fused_recurrent` project
(`kernels/step.py`); the checkpoint-free inference forward for this op is
`kernels/fwd_infer.py` above.

## Wiring — `autograd.py`

`gdn2_recurrent(...)` dispatches on whether a gradient is needed:
- **no grad** (`torch.is_grad_enabled()` false or no input requires grad): calls
  `gdn2_fwd_infer_kernel` directly (only `o` + `final_state` allocated, no autograd
  node) — bit-identical outputs, ~120 µs less host overhead per call.
- **grad**: runs `GDN2Recurrent(torch.autograd.Function)`:
  - **forward**: calls `gdn2_fwd_states_kernel` → `(o, final_state, states, delta_ckpt)`;
    saves `(q,k,v,g,b,w,S0,states,delta_ckpt)` for backward; returns `(o, final_state)`.
  - **backward**: calls `gdn2_recurrent_bwd_kernel` with the saved `states`/`delta_ckpt`
    and incoming `(do, dfinal)` → the 7 gradients.

All three kernels compile via
`ascend_fla.runtime.compile.compile_kernel(kernel, device="a2", block_dim=40, backend="cce")`
and are called `compiled(inputs_dict, scalars_dict, outputs_dict)` (zero-copy via
`data_ptr`). The required symbolic scalar `TP1 = T + 1` sizes the `states` GM dim.

## Validation

`autograd.py` run as `__main__` compares the `autograd.Function`'s gradients to
torch autograd through the reference recurrence (double precision). All 7
gradients match to relative-L2 < 1e-3 on a2 silicon; the backward kernel alone
matches the analytic golden to ~1e-6.

## Constants

`HEAD_DIM = VALUE_DIM = 128`, `PAD = 192`, `GROUP = 64`, `NGROUP = 2`,
`ROWBLK = 16`, `QK_EPS = 1e-6`, `Q_SCALE = 1/sqrt(128)`.
