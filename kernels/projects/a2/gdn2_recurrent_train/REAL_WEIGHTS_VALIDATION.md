# Real-weights validation — GDN-2 a2 training op vs LLM-OS-Models checkpoints

The a2 GDN-2 training op (`gdn2_recurrent`, forward-with-states + reverse-recurrence
backward, block_dim=40) was validated against real trained weights from the
LLM-OS-Models GDN-2 checkpoints, not just synthetic inputs.

## Method

These repos ship **no `config.json`** — only raw `.pth` state-dicts. The architecture
was therefore confirmed directly from the checkpoint tensor shapes and proven by a
`strict=True` load of the repo's own `GDN2ForCausalLM` (`ascend_fla.models`); a strict
load only succeeds if every parameter name and shape matches, so it is an ABI proof of
the inferred config.

A real forward is then run on each model. The GDN-2 layer's `_run_core` is hooked to
capture the **actual per-head `q,k,v,g,b,w`** that enter the recurrence (post short-conv,
post the `g=-A_log.exp()*softplus(f+dt_bias)`, `b=sigmoid`, `w=sigmoid` gate math). Those
real tensors drive both:
- the repo fp32 oracle `ascend_fla.reference.gdn2.gdn2_recurrent_reference`
  (`use_qk_l2norm=True`, `scale=1/sqrt(128)`, `eps=1e-6`), and
- my deployed `GDN2Recurrent` autograd.Function on the NPU,

and the forward output, final state, and all seven gradients (via a random-cotangent
backward) are compared. This exercises the exact production code path on realistic
trained distributions (strong decay, saturated sigmoids) that random inputs never hit.

## Architecture confirmed from checkpoints (no config.json)

| | 370M (`gdn2-370m-fineweb-edu-5b`) | 1.3B (`gdn2-1.3B-fineweb-edu-100b`, ckpt-95B) |
|---|---|---|
| strict load | OK (380,603,648 params) | OK (1,450,096,416 params) |
| vocab | 32000 | 32000 |
| n_embd | 1024 | 2304 |
| n_layer | 16 | 18 |
| num_heads (QK) | 16 | 16 (checkpoint `A_log`=16; the repo README's "n_head=18" is contradicted by the weights) |
| head_dim (K) | 128 | 128 |
| head_v_dim (V), expand_v | 128, 1.0 | 128, 1.0 |
| gates | dual b (K) / w (V) | dual b (K) / w (V) |

Both are the exact per-head shape my kernel targets: `K=V=128`, dual channel-wise
gates. The 370M `-5b`, `-1b`, `-100b` variants share one architecture at different
training-token budgets (same shapes; the `-5b` checkpoint was used). The kernel is
per-head (it loops `B*H`), so the QK-head count (16) is irrelevant to correctness.

## Results — all layers, all quantities ~1e-6

Captured at first / middle / last layers, B=1, T=64, H=16, on the real hidden states.

| model / layer | decay g range | gate b / w range | max relL2 (o, Sf, dq,dk,dv,dg,db,dw,dh0) |
|---|---|---|---|
| 370M layer 0  | [-18.0, 0] | [0.020,0.996]/[0.016,0.996] | 7.7e-6 |
| 370M layer 8  | [-47.5, 0] | [0.003,0.995]/[0.019,0.967] | 8.1e-6 |
| 1.3B layer 0  | [-20.2, 0] | [0.020,0.998]/[0.002,0.998] | 7.0e-6 |
| 1.3B layer 9  | [-27.8, 0] | [0.000,1.000]/[0.000,0.998] | 6.0e-6 |
| 1.3B layer 17 | [-56.3, 0] | [0.000,1.000]/[0.001,1.000] | 7.0e-6 |

Every output, final state and gradient matches the fp32 oracle to relative-L2 < 1e-5
(worst case 8.1e-6, on `db`), including at layer 17 where the decay reaches exp(-56)
(underflow-adjacent) and the gates saturate the full [0,1] range. Identical numbers at
block_dim=8 and block_dim=40 (the parallelism change does not touch the arithmetic), and
identical again after the `muladddst` fusion of the forward and backward rank-1 state
updates — `muladddst` is bit-exact vs the outer-product-then-add on a2, so these results
hold unchanged for the current shipped kernels (re-run on both checkpoints post-fusion).

## Reproduce

    # 370M (config preset)
    ASCEND_RT_VISIBLE_DEVICES=<d> python real_weights_validate.py <370m.pth> 370m
    # any checkpoint (config auto-inferred from shapes)
    ASCEND_RT_VISIBLE_DEVICES=<d> python real_weights_validate2.py <ckpt.pth>
