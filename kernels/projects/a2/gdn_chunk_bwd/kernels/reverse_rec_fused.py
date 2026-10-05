"""A2 GDN chunk-backward fused reverse scan: recurrence + all reductions, no tape.

Folds the reduction work (dq/dk/dbeta/dg) into the reverse recurrence, so the state
cotangent `back_t` is consumed in-UB the moment it is formed and never written to a
GM tape. Eliminates `back_tape` entirely (vs v2's reverse_rec + reduce). Only
`tape_d` (the decayed forward state d_t, from replay) is still loaded.

Per token t (reverse order), with `back` carrying the state cotangent:

  back += scale * q ^ do                       # back = back_t
  r  = v - (k . d_t)_K ;  z  = beta * r
  dz = (back_t . k)_K   ;  dr = beta * dz
  dq[t]    = scale * ( (d_t . do)_V + k * (z . do) )
  dbeta[t] = (dz . r)_V
  dk[t]    = (back_t . z)_V - (d_t . dr)_V
  dv[t]    = dr
  dD = back_t - k ^ dr ;  dg[t] = (dD . d_t)_{K,V}
  back = exp(g) * dD                            # propagate to t-1

dg reuses the `back - k^dr` intermediate (= dD) that the recurrence forms anyway.
`back` is read as the original back_t (dz, (back.z)_V) before the dD overwrite.
Single runtime loop over t (no C-unroll) to keep the fused body's compile bounded.
dk is emitted already combined; group_reduce does only the GVA ratio-sum. GVA k/q
from i_h. `dh0` = final back. Pure vector.
"""
from ascriptor.a2 import *

D = 128
GROUP = 64
NGROUP = D // GROUP
ROWBLK = D // 8
HALFBLK = GROUP // 8
FULL_MASK = (1 << 64) - 1


def _spread8(sp8, x_row):
    brcb(sp8, x_row[0:1, 0:D], repeat=D // 8, dst_blk_stride=1, dst_rep_stride=8)


def _tree_k(scr):
    add(scr[0:64, 0:GROUP], scr[0:64, 0:GROUP], scr[64:128, 0:GROUP])
    add(scr[0:32, 0:GROUP], scr[0:32, 0:GROUP], scr[32:64, 0:GROUP])
    add(scr[0:16, 0:GROUP], scr[0:16, 0:GROUP], scr[16:32, 0:GROUP])
    add(scr[0:8, 0:GROUP], scr[0:8, 0:GROUP], scr[8:16, 0:GROUP])
    add(scr[0:4, 0:GROUP], scr[0:4, 0:GROUP], scr[4:8, 0:GROUP])
    add(scr[0:2, 0:GROUP], scr[0:2, 0:GROUP], scr[2:4, 0:GROUP])
    add(scr[0:1, 0:GROUP], scr[0:1, 0:GROUP], scr[1:2, 0:GROUP])


def _kreduce(dstrow, mat, sp8, scr):
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
            cadd(dstrow[0:1, 0:D], scr[0:D, 0:GROUP], repeat=D, count_per_rep=GROUP,
                 src_blk_stride=1, src_rep_stride=HALFBLK, dst_rep_stride=1)
        else:
            cadd(dtmp[0:1, 0:D], scr[0:D, 0:GROUP], repeat=D, count_per_rep=GROUP,
                 src_blk_stride=1, src_rep_stride=HALFBLK, dst_rep_stride=1)
    add(dstrow[0:1, 0:D], dstrow[0:1, 0:D], dtmp[0:1, 0:D], count=D)


def _dotscalar(out1, prod, red2):
    dup(red2[0:1, 0:2], 0.0, count=2)
    cadd(red2[0:1, 0:2], prod[0:1, 0:D], repeat=2, count_per_rep=GROUP,
         src_blk_stride=1, src_rep_stride=HALFBLK, dst_rep_stride=1)
    cadd(out1[0:1, 0:1], red2[0:1, 0:2], repeat=1, count_per_rep=2,
         src_blk_stride=1, src_rep_stride=1, dst_rep_stride=1)


@kernel(mode="vec", block_dim=40)
def gdn_chunk_bwd_reverse_rec_fused_a2_kernel(
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
    dbeta: GM[f32, ("BT", "HV")],
    dg: GM[f32, ("BT", "HV")],
    dh0: GM[f32, ("B", "HV", 128, 128)],
    scale: f32,
    B: i32,
    T: i32,
    H: i32,
    HV: i32,
    N: i32,
):
    back = Tensor(DT.float, [D, D], Position.UB)       # back_t, then dD in place
    su = Tensor(DT.float, [D, D], Position.UB)          # d_t
    scr = Tensor(DT.float, [D, GROUP], Position.UB)
    sp8 = Tensor(DT.float, [D, 8], Position.UB)
    krow = Tensor(DT.float, [1, D], Position.UB)
    qrow = Tensor(DT.float, [1, D], Position.UB)
    vrow = Tensor(DT.float, [1, D], Position.UB)
    dorow = Tensor(DT.float, [1, D], Position.UB)
    rrow = Tensor(DT.float, [1, D], Position.UB)
    zrow = Tensor(DT.float, [1, D], Position.UB)
    dzrow = Tensor(DT.float, [1, D], Position.UB)
    drrow = Tensor(DT.float, [1, D], Position.UB)
    outrow = Tensor(DT.float, [1, D], Position.UB)
    dkrow = Tensor(DT.float, [1, D], Position.UB)
    dtmp = Tensor(DT.float, [1, D], Position.UB)
    tmp = Tensor(DT.float, [1, D], Position.UB)
    red2 = Tensor(DT.float, [1, 2], Position.UB)
    sc1 = Tensor(DT.float, [1, 8], Position.UB)

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
            for tx in range(T):
                t = Var(T - 1 - tx)
                rr = Var(bb * T + t)
                krow[0:1, 0:D] <<= k[rr:rr + 1, ih * D:ih * D + D]
                qrow[0:1, 0:D] <<= q[rr:rr + 1, ih * D:ih * D + D]
                vrow[0:1, 0:D] <<= v[rr:rr + 1, hv * D:hv * D + D]
                dorow[0:1, 0:D] <<= dout[rr:rr + 1, hv * D:hv * D + D]
                su[0:D, 0:D] <<= tape_d[bb, hv, t, 0:D, 0:D]
                gval = Var(0.0, dtype=DT.float); gval.GetValueFrom(g[rr:rr + 1, hv:hv + 1])
                bval = Var(0.0, dtype=DT.float); bval.GetValueFrom(beta[rr:rr + 1, hv:hv + 1])

                # back += scale*q ^ do  -> back_t
                muls(tmp[0:1, 0:D], qrow[0:1, 0:D], scale, count=D)
                _spread8(sp8, tmp)
                for vh in range(NGROUP):
                    vs = vh * GROUP
                    muladddst(back[0:D, vs:vs + GROUP], sp8, dorow[0:1, vs:vs + GROUP],
                              repeat=D, count_per_rep=GROUP,
                              dst_blk_stride=1, dst_rep_stride=ROWBLK,
                              src1_blk_stride=0, src1_rep_stride=1,
                              src2_blk_stride=1, src2_rep_stride=0)

                # shared reductions
                _spread8(sp8, krow)
                _kreduce(rrow, su, sp8, scr)                      # (k.d)_K
                sub(rrow[0:1, 0:D], vrow[0:1, 0:D], rrow[0:1, 0:D], count=D)   # r
                muls(zrow[0:1, 0:D], rrow[0:1, 0:D], bval, count=D)            # z
                _kreduce(dzrow, back, sp8, scr)                   # dz = (back.k)_K
                muls(drrow[0:1, 0:D], dzrow[0:1, 0:D], bval, count=D)          # dr

                # dq = scale*((d.do)_V + k*(z.do))
                _rowdot(outrow, dtmp, su, dorow, scr)
                mul(tmp[0:1, 0:D], zrow[0:1, 0:D], dorow[0:1, 0:D], count=D)
                _dotscalar(sc1, tmp, red2)
                zdo = Var(0.0, dtype=DT.float); zdo.GetValueFrom(sc1[0:1, 0:1])
                muls(tmp[0:1, 0:D], krow[0:1, 0:D], zdo, count=D)
                add(outrow[0:1, 0:D], outrow[0:1, 0:D], tmp[0:1, 0:D], count=D)
                muls(outrow[0:1, 0:D], outrow[0:1, 0:D], scale, count=D)
                dq_parts[rr:rr + 1, hv * D:hv * D + D] <<= outrow[0:1, 0:D]

                # dbeta = (dz . r)_V
                mul(tmp[0:1, 0:D], dzrow[0:1, 0:D], rrow[0:1, 0:D], count=D)
                _dotscalar(sc1, tmp, red2)
                dbeta[rr:rr + 1, hv:hv + 1] <<= sc1[0:1, 0:1]

                # dk = (back.z)_V - (d.dr)_V   (original back_t)
                _rowdot(outrow, dtmp, back, zrow, scr)
                _rowdot(dkrow, dtmp, su, drrow, scr)
                sub(dkrow[0:1, 0:D], outrow[0:1, 0:D], dkrow[0:1, 0:D], count=D)
                dk_parts[rr:rr + 1, hv * D:hv * D + D] <<= dkrow[0:1, 0:D]

                # dv = dr
                dv[rr:rr + 1, hv * D:hv * D + D] <<= drrow[0:1, 0:D]

                # dD = back - k ^ dr  (overwrites back);  dg = (dD . d)_{K,V}
                muls(tmp[0:1, 0:D], drrow[0:1, 0:D], -1.0, count=D)
                _spread8(sp8, krow)
                for vh in range(NGROUP):
                    vs = vh * GROUP
                    muladddst(back[0:D, vs:vs + GROUP], sp8, tmp[0:1, vs:vs + GROUP],
                              repeat=D, count_per_rep=GROUP,
                              dst_blk_stride=1, dst_rep_stride=ROWBLK,
                              src1_blk_stride=0, src1_rep_stride=1,
                              src2_blk_stride=1, src2_rep_stride=0)
                dgacc = Var(0.0, dtype=DT.float)
                for gi in range(NGROUP):
                    s = gi * GROUP
                    mul(scr[0:D, 0:GROUP], back[0:D, s:s + GROUP], su[0:D, s:s + GROUP],
                        repeat=D, count_per_rep=GROUP,
                        dst_blk_stride=1, dst_rep_stride=HALFBLK,
                        src1_blk_stride=1, src1_rep_stride=ROWBLK,
                        src2_blk_stride=1, src2_rep_stride=ROWBLK)
                    _tree_k(scr)
                    dup(sc1[0:1, 0:1], 0.0, count=1)
                    cadd(sc1[0:1, 0:1], scr[0:1, 0:GROUP], repeat=1, count_per_rep=GROUP,
                         src_blk_stride=1, src_rep_stride=1, dst_rep_stride=1)
                    gpart = Var(0.0, dtype=DT.float); gpart.GetValueFrom(sc1[0:1, 0:1])
                    dgacc.set(dgacc + gpart)
                dgacc.SetValueTo(sc1[0:1, 0:1])
                dg[rr:rr + 1, hv:hv + 1] <<= sc1[0:1, 0:1]

                # back = exp(g) * dD
                dup(tmp[0:1, 0:D], 0.0, count=D)
                adds(tmp[0:1, 0:D], tmp[0:1, 0:D], gval, count=D)
                exp(tmp[0:1, 0:D], tmp[0:1, 0:D], count=D)
                egv = Var(0.0, dtype=DT.float); egv.GetValueFrom(tmp[0:1, 0:1])
                muls(back[0:64, 0:D], back[0:64, 0:D], egv, count=64 * D)
                muls(back[64:128, 0:D], back[64:128, 0:D], egv, count=64 * D)
            dh0[bb, hv, 0:D, 0:D] <<= back[0:D, 0:D]

    return dq_parts, dk_parts, dv, dbeta, dg, dh0
