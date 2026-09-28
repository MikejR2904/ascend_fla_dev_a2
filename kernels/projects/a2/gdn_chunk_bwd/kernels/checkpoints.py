"""A2 GDN chunk-backward stage 1: checkpoints (primal forward replay).

Replays the GDN primal recurrence per (b, value-head) and saves the state at each
chunk boundary (every 64 tokens) plus the final state, so the reverse pass can
replay one chunk at a time without inverting the decay. Per token t (K=V=128):

  S    *= exp(g[t])                       # decay (g[t] a per-token scalar)   -> d
  r     = v[t] - sum_K k[t,K] * S[K,:]    # residual over the K rows of d
  S[K,:] += k[t,K] * (beta[t] * r)        # rank-1 delta-rule update

k comes from the qk-head i_h = hv // (HV//H) (GVA §1.1); v/g/beta are HV-indexed.
Pure vector (no cube, b3). Uses the forward `_spread8` + rowscale/tree-reduce and
`muladddst` idioms. Inline reduces (folding into a helper mistraces on a2).
"""
from ascriptor.a2 import *

D = 128
C = 64
GROUP = 64
NGROUP = D // GROUP           # 2
ROWBLK = D // 8              # 16
HALFBLK = GROUP // 8         # 8
FULL_MASK = (1 << 64) - 1


def _spread8(sp8, x_row, width):
    """sp8[j, 0:8] = x_row[0, j] for j in 0..width-1 (aligned offset 0)."""
    brcb(sp8, x_row[0:1, 0:width], repeat=width // 8, dst_blk_stride=1, dst_rep_stride=8)


def _scalar_exp(dst_row, val, tmp):
    """dst scalar-row scratch not needed; return exp(val) via a [1,D] tmp tile."""
    dup(tmp[0:1, 0:D], 0.0, count=D)
    adds(tmp[0:1, 0:D], tmp[0:1, 0:D], val, count=D)
    exp(tmp[0:1, 0:D], tmp[0:1, 0:D], count=D)


@kernel(mode="vec", block_dim=40)
def gdn_chunk_bwd_checkpoints_a2_kernel(
    k: GM[f32, ("BT", "HD")],
    v: GM[f32, ("BT", "HVD")],
    g: GM[f32, ("BT", "HV")],
    beta: GM[f32, ("BT", "HV")],
    initial_state: GM[f32, ("B", "HV", 128, 128)],
    checkpoints: GM[f32, ("B", "HV", "N", 128, 128)],
    final_state: GM[f32, ("B", "HV", 128, 128)],
    B: i32,
    T: i32,
    H: i32,
    HV: i32,
    N: i32,
):
    su = Tensor(DT.float, [D, D], Position.UB)       # running state S[K,V]
    scr = Tensor(DT.float, [D, GROUP], Position.UB)   # rowscale/reduce scratch (half V)
    sp8 = Tensor(DT.float, [D, 8], Position.UB)        # per-K-row scalar spread of k[t]
    krow = Tensor(DT.float, [1, D], Position.UB)
    vrow = Tensor(DT.float, [1, D], Position.UB)
    zrow = Tensor(DT.float, [1, D], Position.UB)       # residual r, then z = beta*r
    tmp = Tensor(DT.float, [1, D], Position.UB)         # scalar-exp scratch

    work = B * HV
    per = CeilDiv(work, GetVecNum())
    begin = Var(per * GetVecIdx())
    end = Min(begin + per, work)

    with auto_sync():
        set_mask(FULL_MASK, FULL_MASK)
        hvh = Var(HV // H)
        for item in range(begin, end):
            hv = Var(item % HV)
            bb = Var(item // HV)
            ih = Var(hv // hvh)

            su[0:D, 0:D] <<= initial_state[bb, hv, 0:D, 0:D]
            for cc in range(N):
                checkpoints[bb, hv, cc, 0:D, 0:D] <<= su[0:D, 0:D]
                r0 = Var(bb * T + cc * C)
                for i in range(C):
                    rr = Var(r0 + i)
                    krow[0:1, 0:D] <<= k[rr:rr + 1, ih * D:ih * D + D]
                    vrow[0:1, 0:D] <<= v[rr:rr + 1, hv * D:hv * D + D]
                    gval = Var(0.0, dtype=DT.float)
                    gval.GetValueFrom(g[rr:rr + 1, hv:hv + 1])
                    bval = Var(0.0, dtype=DT.float)
                    bval.GetValueFrom(beta[rr:rr + 1, hv:hv + 1])

                    # decay: S *= exp(g[t])  (scalar); split full tile (repeatTimes<=255)
                    _scalar_exp(tmp, gval, tmp)
                    egv = Var(0.0, dtype=DT.float)
                    egv.GetValueFrom(tmp[0:1, 0:1])
                    half = D // 2
                    muls(su[0:half, 0:D], su[0:half, 0:D], egv, count=half * D)
                    muls(su[half:D, 0:D], su[half:D, 0:D], egv, count=half * D)

                    # r = v[t] - sum_K k[t,K]*S[K,:]   (matvec over K rows, two V halves)
                    _spread8(sp8, krow, D)
                    for vh in range(NGROUP):
                        vs = vh * GROUP
                        mul(scr[0:D, 0:GROUP], su[0:D, vs:vs + GROUP], sp8,
                            repeat=D, count_per_rep=GROUP,
                            dst_blk_stride=1, dst_rep_stride=HALFBLK,
                            src1_blk_stride=1, src1_rep_stride=ROWBLK,
                            src2_blk_stride=0, src2_rep_stride=1)
                        add(scr[0:64, 0:GROUP], scr[0:64, 0:GROUP], scr[64:128, 0:GROUP])
                        add(scr[0:32, 0:GROUP], scr[0:32, 0:GROUP], scr[32:64, 0:GROUP])
                        add(scr[0:16, 0:GROUP], scr[0:16, 0:GROUP], scr[16:32, 0:GROUP])
                        add(scr[0:8, 0:GROUP], scr[0:8, 0:GROUP], scr[8:16, 0:GROUP])
                        add(scr[0:4, 0:GROUP], scr[0:4, 0:GROUP], scr[4:8, 0:GROUP])
                        add(scr[0:2, 0:GROUP], scr[0:2, 0:GROUP], scr[2:4, 0:GROUP])
                        add(scr[0:1, 0:GROUP], scr[0:1, 0:GROUP], scr[1:2, 0:GROUP])
                        sub(zrow[0:1, vs:vs + GROUP], vrow[0:1, vs:vs + GROUP],
                            scr[0:1, 0:GROUP], count=GROUP)
                    # z = beta[t] * r
                    muls(zrow[0:1, 0:D], zrow[0:1, 0:D], bval, count=D)

                    # S[K,:] += k[t,K] ^ z   (rank-1; sp8 still holds spread of k[t])
                    for vh in range(NGROUP):
                        vs = vh * GROUP
                        muladddst(su[0:D, vs:vs + GROUP], sp8, zrow[0:1, vs:vs + GROUP],
                                  repeat=D, count_per_rep=GROUP,
                                  dst_blk_stride=1, dst_rep_stride=ROWBLK,
                                  src1_blk_stride=0, src1_rep_stride=1,
                                  src2_blk_stride=1, src2_rep_stride=0)
            final_state[bb, hv, 0:D, 0:D] <<= su[0:D, 0:D]

    return checkpoints, final_state
