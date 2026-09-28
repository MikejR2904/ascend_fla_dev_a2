"""A2 GDN chunk-backward stage 3: group-reduce dq/dk parts (GVA ratio-sum).

The reverse forms dq/dk per value-head (HV); the qk-side gradients are per qk-head
(H). This sums each qk-head's `ratio = HV//H` value-head parts:

  dq[t, h, :] = sum_{r<ratio} dq_parts[t, h*ratio + r, :]   (and likewise dk)

A no-op copy when HV == H. One core per (b, t, qk-head); a plain accumulation of
`ratio` contiguous [1,128] segments. Low complexity (no reductions across the D
axis), so it stays well within the sim's sync budget. Pure vector.
"""
from ascriptor.a2 import *

D = 128
FULL_MASK = (1 << 64) - 1


@kernel(mode="vec", block_dim=40)
def gdn_chunk_bwd_group_reduce_a2_kernel(
    dq_parts: GM[f32, ("BT", "HVD")],
    dk_parts: GM[f32, ("BT", "HVD")],
    dq: GM[f32, ("BT", "HD")],
    dk: GM[f32, ("BT", "HD")],
    B: i32,
    T: i32,
    H: i32,
    HV: i32,
    N: i32,
):
    accq = Tensor(DT.float, [1, D], Position.UB)
    acck = Tensor(DT.float, [1, D], Position.UB)
    seg = Tensor(DT.float, [1, D], Position.UB)

    work = B * T * H
    per = CeilDiv(work, GetVecNum())
    begin = Var(per * GetVecIdx())
    end = Min(begin + per, work)

    with auto_sync():
        set_mask(FULL_MASK, FULL_MASK)
        ratio = Var(HV // H)
        for item in range(begin, end):
            h = Var(item % H)
            bt = Var(item // H)
            hv0 = Var(h * ratio)

            accq[0:1, 0:D] <<= dq_parts[bt:bt + 1, hv0 * D:hv0 * D + D]
            acck[0:1, 0:D] <<= dk_parts[bt:bt + 1, hv0 * D:hv0 * D + D]
            for r in range(1, ratio):
                hv = Var(hv0 + r)
                seg[0:1, 0:D] <<= dq_parts[bt:bt + 1, hv * D:hv * D + D]
                add(accq[0:1, 0:D], accq[0:1, 0:D], seg[0:1, 0:D], count=D)
                seg[0:1, 0:D] <<= dk_parts[bt:bt + 1, hv * D:hv * D + D]
                add(acck[0:1, 0:D], acck[0:1, 0:D], seg[0:1, 0:D], count=D)
            dq[bt:bt + 1, h * D:h * D + D] <<= accq[0:1, 0:D]
            dk[bt:bt + 1, h * D:h * D + D] <<= acck[0:1, 0:D]

    return dq, dk
