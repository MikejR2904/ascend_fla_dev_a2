# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The composed training step: one call for the forward and the backward.

The operator ships as nine single-kernel launches, each with its own host entry
point. This module is the composition of the eight that a training step needs,
and it exists because the way they are chained is itself part of the operator's
cost rather than a detail left to the caller.

Two properties of the dependency graph drive the composition.

**The chain needs one device synchronisation, not eight.** Every component
wrapper synchronises before returning, which is correct when it is called on its
own and is a host-device round trip per launch when eight are chained. The
wrappers therefore take a keyword-only `sync`, whose default preserves their
standalone behaviour, and this entry point passes `sync=False` to all eight and
synchronises once.

**The graph has two independent branches.** `gdr_fwd_train_scan` and the pair
`gdr_bwd_dv_local -> gdr_bwd_reverse_scan` both read only the outputs of
`gdr_fwd_train_wy`, and nothing reads the scan's results until
`gdr_bwd_dqkw`. The two can therefore run concurrently, which this module
expresses by issuing them on separate streams and rejoining before `dqkw`. The
launch bracket is `kernel[stream, block_dim]` and every component passes `None`
there, which the runtime resolves through the current stream, so redirecting a
launch needs no change inside any kernel.

Concurrency alone is not enough, and the reason is the whole of the core
budgeting below: each component's tiling selector maximises its own core fill on
the assumption that it owns the part. Two launches that each claim all 28 cube
cores do not overlap -- they serialise, with the scheduler interleaving them --
so the overlap pays only when the co-running launches are sized so that their
block counts sum to the part rather than oversubscribing it.

Every setting of either budget, and the concurrent and sequential orders alike,
produce bit-identical results: a budget changes only how an independent item
space folds onto cores, never the arithmetic. `tests/test_train_entry.py` is the
acceptance gate for that claim.
"""

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
from pypto_pro.runtime.platform import get_platform_info

from gdr_fwd_train_wy_impl import gdr_fwd_train_wy_wrapper
from gdr_fwd_train_scan_impl import gdr_fwd_train_scan_wrapper
from gdr_bwd_dv_local_impl import gdr_bwd_dv_local_wrapper
from gdr_bwd_reverse_scan_impl import gdr_bwd_reverse_scan_wrapper
from gdr_bwd_dqkw_impl import gdr_bwd_dqkw_wrapper
from gdr_bwd_wy_matmuls_impl import gdr_bwd_wy_matmuls_wrapper
from gdr_bwd_wy_reduce_impl import gdr_bwd_wy_reduce_wrapper
from gdr_bwd_finalize_impl import gdr_bwd_finalize_wrapper

REVERSE_SCAN_TV = 64    # V sub-block width of gdr_bwd_reverse_scan
MIN_SCAN_CORES = 8      # floor on the forward scan's share of the part

_STREAMS = {}


def _streams_for(device):
    """The cached pair of side streams for one device.

    Streams are created once per device and reused. Creating them per call
    would put stream construction on the training step's critical path.
    """
    index = device.index if device.index is not None else torch.npu.current_device()
    pair = _STREAMS.get(index)
    if pair is None:
        pair = (torch.npu.Stream(device=index), torch.npu.Stream(device=index))
        _STREAMS[index] = pair
    return pair


def reverse_scan_core_budget(B, H, V):
    """The blocks to plan `gdr_bwd_reverse_scan` for: the fewest that keep its wave count.

    Its item space is `B * H * ceil(V / TV)` and the kernel strides over it, so
    the block count is free and only balance depends on it. Taking every core is
    right when the items divide evenly over the part. Where they do not, the
    remainder forms a final wave that holds only a few cores while the rest are
    idle for its whole duration, and the launch's own duration is unchanged by
    dividing the same items into the same number of waves over fewer blocks.
    Fewer blocks at an unchanged wave count is therefore free for this launch
    and leaves a constant number of cores free throughout it, which is what a
    concurrent co-runner can use.
    """
    cores = get_platform_info().core_num or 1
    items = B * H * ((V + REVERSE_SCAN_TV - 1) // REVERSE_SCAN_TV)
    if items <= cores:
        return items
    waves = -(-items // cores)
    return -(-items // waves)


def scan_core_budget(B, H, V):
    """The cores to plan `gdr_fwd_train_scan` for while the backward branch runs beside it.

    The branch's resident demand is `gdr_bwd_reverse_scan`'s block count, so
    what is left of the part is what the scan can expect. The floor guards the
    case of a head count large enough that the branch alone would leave nothing.

    Both counts are the balanced ones, which is what makes a single subtraction
    correct: an unbalanced co-runner holds a different number of cores in each
    of its waves, and no single budget fits them all.
    """
    cores = get_platform_info().core_num or 1
    return max(MIN_SCAN_CORES, cores - reverse_scan_core_budget(B, H, V))


def gated_delta_rule_fwd_bwd(q, k, v, g, beta, grad_o, *, concurrent=True,
                             sync=True, scan_core_budget_override=None,
                             reverse_scan_core_budget_override=None):
    """Run the training forward and backward, and return the public results.

    Args:
        q, k: [B, T, H, K] bfloat16
        v: [B, T, H, V] bfloat16
        g: [B, T, H] float32, a log-space decay, non-positive
        beta: [B, T, H] bfloat16, sigmoid logits
        grad_o: [B, T, H, V] bfloat16, the incoming output gradient
        concurrent: issue the two independent branches on separate streams.
            False gives the plain sequential order, which is the reference for
            any comparison: the two agree bit for bit.
        sync: synchronise the device before returning. False hands that
            responsibility to the caller, who must synchronise before reading
            any result on the host; ordering on the device is already
            guaranteed. It also lets successive steps overlap, which is how a
            training loop should call this -- see below.
        scan_core_budget_override: the cores `gdr_fwd_train_scan` plans for
            while it runs concurrently, in place of `scan_core_budget`. Ignored
            when `concurrent` is False.
        reverse_scan_core_budget_override: the blocks
            `gdr_bwd_reverse_scan` plans for, in place of
            `reverse_scan_core_budget`.

    Returns:
        (o, dq, dk, dv, dg, dbeta) -- `o` and the three large gradients in
        bfloat16, `dg` and `dbeta` in float32, all in the public [B, T, H, *]
        layout.

    A training loop should not call this with `sync=True` per step. Queue the
    steps and synchronise once, which lets the host dispatch of one step run
    under the device execution of the one before it. Successive microbatches are
    independent, so issuing them on alternating streams overlaps their device
    work as well:

        step_streams = (torch.npu.Stream(), torch.npu.Stream())
        for i, batch in enumerate(batches):
            with torch.npu.stream(step_streams[i % 2]):
                out = gated_delta_rule_fwd_bwd(*batch, sync=False)
            ...                      # consume `out` on that stream
        torch.npu.synchronize()      # once, at the end
    """
    (q_s, k_f32, v_b, g_f32, beta_act, sig, q_t, decay, decay_exp,
     dfac, q_de, attn, a_mat, kbeta_t, u, w, w_t) = gdr_fwd_train_wy_wrapper(
        q, k, v, g, beta, sync=False)

    def run_scan(budget):
        return gdr_fwd_train_scan_wrapper(u, w, q_de, a_mat, k_f32, dfac, decay_exp,
                                          sync=False, cube_core_budget=budget)

    def run_backward_branch(reverse_budget):
        dv_local = gdr_bwd_dv_local_wrapper(grad_o, k_f32=k_f32, q_t=q_t, g_f32=g_f32,
                                            sync=False)
        return gdr_bwd_reverse_scan_wrapper(
            q_t, w_t, k_f32, grad_o=grad_o, decay_exp=decay_exp, g_f32=g_f32,
            dv_local=dv_local, sync=False, cube_core_budget=reverse_budget)

    if concurrent:
        B, T, H, V = grad_o.shape
        scan_budget = (scan_core_budget_override if scan_core_budget_override is not None
                       else scan_core_budget(B, H, V))
        reverse_budget = (reverse_scan_core_budget_override
                          if reverse_scan_core_budget_override is not None
                          else reverse_scan_core_budget(B, H, V))
        scan_stream, backward_stream = _streams_for(grad_o.device)
        here = torch.npu.current_stream()
        # Both branches read train_wy's outputs, which were produced here.
        scan_stream.wait_stream(here)
        backward_stream.wait_stream(here)
        with torch.npu.stream(scan_stream):
            o, s_hist, v_new = run_scan(scan_budget)
        with torch.npu.stream(backward_stream):
            dh, dv2 = run_backward_branch(reverse_budget)
        # dqkw needs both branches, so rejoin before it.
        here.wait_stream(scan_stream)
        here.wait_stream(backward_stream)
        # These five were allocated on a side stream and are read from this one,
        # so the caching allocator has to know this stream still needs them.
        for tensor in (o, s_hist, v_new, dh, dv2):
            tensor.record_stream(here)
    else:
        o, s_hist, v_new = run_scan(None)
        dh, dv2 = run_backward_branch(None)

    dq_p, dk_p, dw, dg_p = gdr_bwd_dqkw_wrapper(
        grad_o, v_new, k_f32, q_s, s_hist, dh, dv2, g_f32=g_f32, decay=decay,
        sync=False)
    dkbg, dvb, dkb, dkt, _d_a = gdr_bwd_wy_matmuls_wrapper(
        attn, k_f32, v_b, kbeta_t, beta_act, decay, dw, dv2, sync=False)
    dk, dv_chunked, dbeta_wy, dg2 = gdr_bwd_wy_reduce_wrapper(
        dkbg, dvb, dkb, dkt, k_f32, beta_act, decay_exp, dk_p, v, sync=False)
    dq, dk_out, dv_out, dg, dbeta = gdr_bwd_finalize_wrapper(
        grad_o, dq_p, dk, dv_chunked, dg_p, dg2, dbeta_wy, sig, sync=False)

    if sync:
        torch.npu.synchronize()
    return o, dq, dk_out, dv_out, dg, dbeta
