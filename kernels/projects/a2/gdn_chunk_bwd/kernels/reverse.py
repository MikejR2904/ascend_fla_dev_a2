"""A2 GDN chunk-backward stage 2: reverse (the adjoint scan).

Per (b, value-head), for each chunk in reverse: reload the chunk-start state
(checkpoints), replay the chunk forward taping the decayed state d_t to
`tape_d[B,HV,C,128,128]`, then reverse-scan producing the gradients and threading
the state cotangent `back` (init dht, final -> dh0). Per token (reverse), with
d = tape_d[t], r/z recomputed from d, k, v (analytical adjoint, ref/reference.py):

  dq[t]  = (d . do)_V + k * (z . do)                # row over V, + rank-1 scalar
  back  += scale * q ^ do                            # rank-1 (muladddst)
  dz     = (back . k)_K                              # over K -> row [V]
  dr     = beta * dz ;  dv[t] = dr
  dk[t]  = (back . z)_V - (d . dr)_V                 # two rows over V
  dbeta[t] = (dz . r)_V                              # scalar
  dD     = back - k ^ dr ;  dg[t] = (dD . d)_{K,V}   # full reduce
  back   = exp(g) * dD

dq/dk are rows [1,K] built with `cadd` at dst_rep_stride=1 (contiguous), so no
transpose is needed (b3 has no cube). dq_parts/dk_parts are per value-head; the
GVA ratio-sum to H is `group_reduce`. k/q come from i_h = hv//(HV//H). Pure vector.
The `do` GM edge is named `dout` (C-keyword-safe). Reductions/rank-1s use the
forward idioms; helpers carry no data-dependent loop (that mistraces on a2).
"""
from ascriptor.a2 import *

D = 128
C = 64
GROUP = 64
NGROUP = D // GROUP           # 2
ROWBLK = D // 8              # 16
HALFBLK = GROUP // 8         # 8
FULL_MASK = (1 << 64) - 1


def _spread8(sp8, x_row):
    """sp8[j,0:8] = x_row[0,j], j in 0..D-1 (aligned offset 0)."""
    brcb(sp8, x_row[0:1, 0:D], repeat=D // 8, dst_blk_stride=1, dst_rep_stride=8)


def _tree_k(scr):
    """In-place tree-add of scr[0:D,0:GROUP] over the D rows -> scr[0:1,0:GROUP]."""
    add(scr[0:64, 0:GROUP], scr[0:64, 0:GROUP], scr[64:128, 0:GROUP])
    add(scr[0:32, 0:GROUP], scr[0:32, 0:GROUP], scr[32:64, 0:GROUP])
    add(scr[0:16, 0:GROUP], scr[0:16, 0:GROUP], scr[16:32, 0:GROUP])
    add(scr[0:8, 0:GROUP], scr[0:8, 0:GROUP], scr[8:16, 0:GROUP])
    add(scr[0:4, 0:GROUP], scr[0:4, 0:GROUP], scr[4:8, 0:GROUP])
    add(scr[0:2, 0:GROUP], scr[0:2, 0:GROUP], scr[2:4, 0:GROUP])
    add(scr[0:1, 0:GROUP], scr[0:1, 0:GROUP], scr[1:2, 0:GROUP])


def _kreduce(dstrow, mat, sp8, scr):
    """dstrow[0,0:V] = sum_K mat[K,V] * sp8[K]  (contraction over the K rows)."""
    for vh in range(NGROUP):
        vs = vh * GROUP
        mul(scr[0:D, 0:GROUP], mat[0:D, vs:vs + GROUP], sp8,
            repeat=D, count_per_rep=GROUP,
            dst_blk_stride=1, dst_rep_stride=HALFBLK,
            src1_blk_stride=1, src1_rep_stride=ROWBLK,
            src2_blk_stride=0, src2_rep_stride=1)
        _tree_k(scr)
        adds(dstrow[0:1, vs:vs + GROUP], scr[0:1, 0:GROUP], 0.0, count=GROUP)


def _rowdot(dstrow, dtmp, mat, brow, scr):
    """dstrow[0,0:K] = sum_V mat[K,V] * brow[0,V]  per K row -> contiguous row."""
    dup(dstrow[0:1, 0:D], 0.0, count=D)
    dup(dtmp[0:1, 0:D], 0.0, count=D)
    for gi in range(NGROUP):
        s = gi * GROUP
        mul(scr[0:D, 0:GROUP], mat[0:D, s:s + GROUP], brow[0:1, s:s + GROUP],
            repeat=D, count_per_rep=GROUP,
            dst_blk_stride=1, dst_rep_stride=HALFBLK,
            src1_blk_stride=1, src1_rep_stride=ROWBLK,
            src2_blk_stride=1, src2_rep_stride=0)
        if gi == 0:
            cadd(dstrow[0:1, 0:D], scr[0:D, 0:GROUP],
                 repeat=D, count_per_rep=GROUP,
                 src_blk_stride=1, src_rep_stride=HALFBLK, dst_rep_stride=1)
        else:
            cadd(dtmp[0:1, 0:D], scr[0:D, 0:GROUP],
                 repeat=D, count_per_rep=GROUP,
                 src_blk_stride=1, src_rep_stride=HALFBLK, dst_rep_stride=1)
    add(dstrow[0:1, 0:D], dstrow[0:1, 0:D], dtmp[0:1, 0:D], count=D)


def _dotscalar(out1, prod, red2):
    """out1[0,0] = sum_{0..D-1} prod[0,:]  via two cadd stages."""
    dup(red2[0:1, 0:2], 0.0, count=2)
    cadd(red2[0:1, 0:2], prod[0:1, 0:D], repeat=2, count_per_rep=GROUP,
         src_blk_stride=1, src_rep_stride=HALFBLK, dst_rep_stride=1)
    cadd(out1[0:1, 0:1], red2[0:1, 0:2], repeat=1, count_per_rep=2,
         src_blk_stride=1, src_rep_stride=1, dst_rep_stride=1)


@kernel(mode="vec", block_dim=40)
def gdn_chunk_bwd_reverse_a2_kernel(
    q: GM[f32, ("BT", "HD")],
    k: GM[f32, ("BT", "HD")],
    v: GM[f32, ("BT", "HVD")],
    g: GM[f32, ("BT", "HV")],
    beta: GM[f32, ("BT", "HV")],
    dout: GM[f32, ("BT", "HVD")],
    dht: GM[f32, ("B", "HV", 128, 128)],
    tape_d: GM[f32, ("B", "HV", "T", 128, 128)],
    dq_parts: GM[f32, ("BT", "HVD")],
    dk_parts: GM[f32, ("BT", "HVD")],
    dv: GM[f32, ("BT", "HVD")],
    dg: GM[f32, ("BT", "HV")],
    dbeta: GM[f32, ("BT", "HV")],
    dh0: GM[f32, ("B", "HV", 128, 128)],
    scale: f32,
    B: i32,
    T: i32,
    H: i32,
    HV: i32,
    N: i32,
):
    back = Tensor(DT.float, [D, D], Position.UB)      # state cotangent
    su = Tensor(DT.float, [D, D], Position.UB)         # replay state; reverse d_t
    scr = Tensor(DT.float, [D, GROUP], Position.UB)    # reduction scratch (half V)
    sp8 = Tensor(DT.float, [D, 8], Position.UB)         # per-K spread
    krow = Tensor(DT.float, [1, D], Position.UB)
    qrow = Tensor(DT.float, [1, D], Position.UB)
    vrow = Tensor(DT.float, [1, D], Position.UB)
    dorow = Tensor(DT.float, [1, D], Position.UB)
    zrow = Tensor(DT.float, [1, D], Position.UB)        # z_t
    rrow = Tensor(DT.float, [1, D], Position.UB)        # r_t
    drrow = Tensor(DT.float, [1, D], Position.UB)       # dr / dz
    outrow = Tensor(DT.float, [1, D], Position.UB)      # dq / dk row
    dtmp = Tensor(DT.float, [1, D], Position.UB)        # _rowdot 2nd half
    tmp = Tensor(DT.float, [1, D], Position.UB)         # scalar-exp / products
    red2 = Tensor(DT.float, [1, 2], Position.UB)
    sc1 = Tensor(DT.float, [1, 8], Position.UB)          # scalar reduction target

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

            back[0:D, 0:D] <<= dht[bb, hv, 0:D, 0:D]

            for ccx in range(N):
                cc = Var(N - 1 - ccx)
                r0 = Var(bb * T + cc * C)

                # ---- reverse within chunk (d_t read from the replay tape) ----
                for ix in range(C):
                    i = C - 1 - ix
                    rr = Var(r0 + i)
                    tt = Var(cc * C + i)
                    krow[0:1, 0:D] <<= k[rr:rr + 1, ih * D:ih * D + D]
                    qrow[0:1, 0:D] <<= q[rr:rr + 1, ih * D:ih * D + D]
                    vrow[0:1, 0:D] <<= v[rr:rr + 1, hv * D:hv * D + D]
                    dorow[0:1, 0:D] <<= dout[rr:rr + 1, hv * D:hv * D + D]
                    gval = Var(0.0, dtype=DT.float); gval.GetValueFrom(g[rr:rr + 1, hv:hv + 1])
                    bval = Var(0.0, dtype=DT.float); bval.GetValueFrom(beta[rr:rr + 1, hv:hv + 1])
                    su[0:D, 0:D] <<= tape_d[bb, hv, tt, 0:D, 0:D]    # d_t (from replay)

                    # recompute r,z from d_t
                    _spread8(sp8, krow)
                    _kreduce(rrow, su, sp8, scr)
                    sub(rrow[0:1, 0:D], vrow[0:1, 0:D], rrow[0:1, 0:D], count=D)  # r
                    muls(zrow[0:1, 0:D], rrow[0:1, 0:D], bval, count=D)          # z

                    # dq = (d.do)_V + k*(z.do)
                    _rowdot(outrow, dtmp, su, dorow, scr)                        # (d.do)_V row
                    mul(tmp[0:1, 0:D], zrow[0:1, 0:D], dorow[0:1, 0:D], count=D)
                    _dotscalar(sc1, tmp, red2)
                    zdo = Var(0.0, dtype=DT.float); zdo.GetValueFrom(sc1[0:1, 0:1])
                    muls(tmp[0:1, 0:D], krow[0:1, 0:D], zdo, count=D)
                    add(outrow[0:1, 0:D], outrow[0:1, 0:D], tmp[0:1, 0:D], count=D)
                    dq_parts[rr:rr + 1, hv * D:hv * D + D] <<= outrow[0:1, 0:D]

                    # back += scale*q ^ do
                    muls(tmp[0:1, 0:D], qrow[0:1, 0:D], scale, count=D)
                    _spread8(sp8, tmp)
                    for vh in range(NGROUP):
                        vs = vh * GROUP
                        muladddst(back[0:D, vs:vs + GROUP], sp8, dorow[0:1, vs:vs + GROUP],
                                  repeat=D, count_per_rep=GROUP,
                                  dst_blk_stride=1, dst_rep_stride=ROWBLK,
                                  src1_blk_stride=0, src1_rep_stride=1,
                                  src2_blk_stride=1, src2_rep_stride=0)

                    # dz = (back.k)_K ; dr = beta*dz ; dv = dr
                    _spread8(sp8, krow)
                    _kreduce(drrow, back, sp8, scr)                              # dz
                    mul(tmp[0:1, 0:D], drrow[0:1, 0:D], rrow[0:1, 0:D], count=D)  # dz*r
                    _dotscalar(sc1, tmp, red2)
                    dbeta[rr:rr + 1, hv:hv + 1] <<= sc1[0:1, 0:1]               # dbeta (DMA)
                    muls(drrow[0:1, 0:D], drrow[0:1, 0:D], bval, count=D)        # dr = beta*dz
                    dv[rr:rr + 1, hv * D:hv * D + D] <<= drrow[0:1, 0:D]         # dv

                    # dk = (back.z)_V - (d.dr)_V
                    _rowdot(outrow, dtmp, back, zrow, scr)                       # (back.z)_V
                    _rowdot(tmp, dtmp, su, drrow, scr)                          # (d.dr)_V  (tmp reused ok)
                    sub(outrow[0:1, 0:D], outrow[0:1, 0:D], tmp[0:1, 0:D], count=D)
                    dk_parts[rr:rr + 1, hv * D:hv * D + D] <<= outrow[0:1, 0:D]

                    # dD = back - k ^ dr  (in place: back -= k ^ dr)
                    muls(tmp[0:1, 0:D], drrow[0:1, 0:D], -1.0, count=D)
                    _spread8(sp8, krow)
                    for vh in range(NGROUP):
                        vs = vh * GROUP
                        muladddst(back[0:D, vs:vs + GROUP], sp8, tmp[0:1, vs:vs + GROUP],
                                  repeat=D, count_per_rep=GROUP,
                                  dst_blk_stride=1, dst_rep_stride=ROWBLK,
                                  src1_blk_stride=0, src1_rep_stride=1,
                                  src2_blk_stride=1, src2_rep_stride=0)

                    # dg = sum_{K,V} dD * d
                    dgacc = Var(0.0, dtype=DT.float)
                    for gi in range(NGROUP):
                        s = gi * GROUP
                        mul(scr[0:D, 0:GROUP], back[0:D, s:s + GROUP], su[0:D, s:s + GROUP],
                            repeat=D, count_per_rep=GROUP,
                            dst_blk_stride=1, dst_rep_stride=HALFBLK,
                            src1_blk_stride=1, src1_rep_stride=ROWBLK,
                            src2_blk_stride=1, src2_rep_stride=ROWBLK)
                        _tree_k(scr)                                            # -> scr[0:1,0:GROUP]
                        dup(sc1[0:1, 0:1], 0.0, count=1)                         # cadd accumulates; zero it, consume all
                        cadd(sc1[0:1, 0:1], scr[0:1, 0:GROUP], repeat=1, count_per_rep=GROUP,
                             src_blk_stride=1, src_rep_stride=1, dst_rep_stride=1)
                        gpart = Var(0.0, dtype=DT.float); gpart.GetValueFrom(sc1[0:1, 0:1])
                        dgacc.set(dgacc + gpart)
                    dgacc.SetValueTo(sc1[0:1, 0:1])
                    dg[rr:rr + 1, hv:hv + 1] <<= sc1[0:1, 0:1]                   # dg (DMA)

                    # back = exp(g) * dD
                    dup(tmp[0:1, 0:D], 0.0, count=D)
                    adds(tmp[0:1, 0:D], tmp[0:1, 0:D], gval, count=D)
                    exp(tmp[0:1, 0:D], tmp[0:1, 0:D], count=D)
                    egv2 = Var(0.0, dtype=DT.float); egv2.GetValueFrom(tmp[0:1, 0:1])
                    muls(back[0:64, 0:D], back[0:64, 0:D], egv2, count=64 * D)
                    muls(back[64:128, 0:D], back[64:128, 0:D], egv2, count=64 * D)

            dh0[bb, hv, 0:D, 0:D] <<= back[0:D, 0:D]

    return dq_parts, dk_parts, dv, dg, dbeta, dh0
