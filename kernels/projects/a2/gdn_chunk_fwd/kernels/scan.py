"""A2 GDN chunk-forward stage 4: scan (chunk recurrence).

Tensor-vector port of a5 `gdn_chunk_fwd/kernels/stages.py::scan_vf`. Per (b,
value-head), carry the recurrent state S[K,V] across the N chunks. For each
chunk (S is the state entering the chunk):

  delta[i,:] = u[i,:] - (wy[i,:] @ S)                       (i in 0..C-1)
  k[i,:]    *= exp(gc[C-1] - gc[i])                          (preweight keys)
  S[d,:]     = S[d,:] * exp(gc[C-1]) + sum_i k[i,d]*delta[i,:]   (= S*decay + k^T@delta)

The a5 source is pure vector (`@vf`, no cube MMAD), so this port is pure vector
too — the natural choice on b3, where the cube L0C DMAs (nd2nz / l0c_to_ub) are
unavailable (A2-01). The two matrix contractions are done with the proven
`fwd_states.py` idioms: `wy @ S` as a per-row `_spread8`+rowscale+tree-reduce
matvec, and `k^T @ delta` as `muladddst` rank-1 accumulates (dst = dst +
src1*src2, bit-identical to outer-product-then-add on a2).

UB budget forces a value-half-split reduce scratch: two full [128,128] tiles
(S + scratch) already fill all 192 KiB, leaving no room for `sp8`, so the step-1
reduce works one 64-lane value half at a time (`scr` is [128,64]). GDN's g is a
per-token scalar, so the key preweight and the state decay reduce to per-row
scalars (Var idiom). T % 64 == 0 upstream (gap §1.7): count is always C.
"""
from ascriptor.a2 import *

D = 128
C = 64
GROUP = 64
NGROUP = D // GROUP            # 2
ROWBLK = D // 8               # 16 (row stride of a [.,128] tile, in 32-byte blocks)
HALFBLK = GROUP // 8          # 8  (row stride of a [.,64] tile, in blocks)
FULL_MASK = (1 << 64) - 1


def _spread8(sp8, x_row):
    """sp8[j, 0:8] = x_row[0, j] for j in 0..D-1 (read from aligned offset 0)."""
    brcb(sp8, x_row[0:1, 0:D], repeat=D // 8, dst_blk_stride=1, dst_rep_stride=8)


@kernel(mode="vec", block_dim=40)
def gdn_chunk_scan_a2_kernel(
    kn: GM[f32, ("B", "N", "HV", 64, 128)],
    gc: GM[f32, ("B", "N", "HV", 64, 128)],
    u: GM[f32, ("B", "N", "HV", 64, 128)],
    wy: GM[f32, ("B", "N", "HV", 64, 128)],
    initial_state: GM[f32, ("B", "HV", 128, 128)],
    states: GM[f32, ("B", "N", "HV", 128, 128)],
    delta: GM[f32, ("B", "N", "HV", 64, 128)],
    final_state: GM[f32, ("B", "HV", 128, 128)],
    B: i32,
    T: i32,
    H: i32,
    HV: i32,
    N: i32,
):
    su = Tensor(DT.float, [D, D], Position.UB)      # recurrent state S[K,V]
    uu = Tensor(DT.float, [C, D], Position.UB)       # u -> delta
    wu = Tensor(DT.float, [C, D], Position.UB)       # wy (step1); k (step2/3)
    scr = Tensor(DT.float, [D, GROUP], Position.UB)  # step1 reduce scratch (half V)
    sp8 = Tensor(DT.float, [D, 8], Position.UB)       # per-row scalar spread
    gcol = Tensor(DT.float, [C, 8], Position.UB)      # per-token scalar prefix
    tmp = Tensor(DT.float, [1, D], Position.UB)       # scalar-exp scratch

    work = B * HV
    per = CeilDiv(work, GetVecNum())
    begin = Var(per * GetVecIdx())
    end = Min(begin + per, work)

    with auto_sync():
        set_mask(FULL_MASK, FULL_MASK)
        for item in range(begin, end):
            hv = Var(item % HV)
            bb = Var(item // HV)

            su[0:D, 0:D] <<= initial_state[bb, hv, 0:D, 0:D]

            for cc in range(N):
                states[bb, cc, hv, 0:D, 0:D] <<= su[0:D, 0:D]
                uu[0:C, 0:D] <<= u[bb, cc, hv, 0:C, 0:D]
                wu[0:C, 0:D] <<= wy[bb, cc, hv, 0:C, 0:D]

                # ---- step 1: delta[i,:] = u[i,:] - wy[i,:] @ S ----
                for i in range(C):
                    _spread8(sp8, wu[i:i + 1, 0:D])           # sp8[d] = wy[i,d]
                    for vh in range(NGROUP):
                        vs = vh * GROUP
                        # scr[d, 0:64] = S[d, vs:vs+64] * wy[i,d]
                        mul(scr[0:D, 0:GROUP], su[0:D, vs:vs + GROUP], sp8,
                            repeat=D, count_per_rep=GROUP,
                            dst_blk_stride=1, dst_rep_stride=HALFBLK,
                            src1_blk_stride=1, src1_rep_stride=ROWBLK,
                            src2_blk_stride=0, src2_rep_stride=1)
                        # tree-reduce over the 128 K-rows -> scr[0:1, 0:64]
                        add(scr[0:64, 0:GROUP], scr[0:64, 0:GROUP], scr[64:128, 0:GROUP])
                        add(scr[0:32, 0:GROUP], scr[0:32, 0:GROUP], scr[32:64, 0:GROUP])
                        add(scr[0:16, 0:GROUP], scr[0:16, 0:GROUP], scr[16:32, 0:GROUP])
                        add(scr[0:8, 0:GROUP], scr[0:8, 0:GROUP], scr[8:16, 0:GROUP])
                        add(scr[0:4, 0:GROUP], scr[0:4, 0:GROUP], scr[4:8, 0:GROUP])
                        add(scr[0:2, 0:GROUP], scr[0:2, 0:GROUP], scr[2:4, 0:GROUP])
                        add(scr[0:1, 0:GROUP], scr[0:1, 0:GROUP], scr[1:2, 0:GROUP])
                        sub(uu[i:i + 1, vs:vs + GROUP], uu[i:i + 1, vs:vs + GROUP],
                            scr[0:1, 0:GROUP], count=GROUP)
                # uu now holds delta

                # ---- step 2: preweight keys k[i,:] *= exp(gc[C-1] - gc[i]) ----
                wu[0:C, 0:D] <<= kn[bb, cc, hv, 0:C, 0:D]        # reuse wu as k
                gcol[0:C, 0:1] <<= gc[bb, cc, hv, 0:C, 0:1]      # per-token prefix
                glast = Var(0.0, dtype=DT.float)
                glast.GetValueFrom(gcol[C - 1:C, 0:1])
                for i in range(C):
                    gi = Var(0.0, dtype=DT.float)
                    gi.GetValueFrom(gcol[i:i + 1, 0:1])
                    diff = Var(0.0, dtype=DT.float)
                    diff.set(glast - gi)
                    dup(tmp[0:1, 0:D], 0.0)
                    adds(tmp[0:1, 0:D], tmp[0:1, 0:D], diff, count=D)
                    exp(tmp[0:1, 0:D], tmp[0:1, 0:D], count=D)
                    ev = Var(0.0, dtype=DT.float)
                    ev.GetValueFrom(tmp[0:1, 0:1])
                    muls(wu[i:i + 1, 0:D], wu[i:i + 1, 0:D], ev, count=D)

                # ---- step 3: S = S*exp(gc[C-1]) + k^T @ delta ----
                dup(tmp[0:1, 0:D], 0.0)
                adds(tmp[0:1, 0:D], tmp[0:1, 0:D], glast, count=D)
                exp(tmp[0:1, 0:D], tmp[0:1, 0:D], count=D)
                declast = Var(0.0, dtype=DT.float)
                declast.GetValueFrom(tmp[0:1, 0:1])
                # count/64 is the intrinsic repeatTimes, capped at 255; a full
                # [128,128] tile is 256 repeats, so scale in two row halves.
                half = D // 2
                muls(su[0:half, 0:D], su[0:half, 0:D], declast, count=half * D)
                muls(su[half:D, 0:D], su[half:D, 0:D], declast, count=half * D)
                for i in range(C):
                    _spread8(sp8, wu[i:i + 1, 0:D])              # sp8[d] = k[i,d]
                    for vh in range(NGROUP):
                        vs = vh * GROUP
                        muladddst(su[0:D, vs:vs + GROUP], sp8, uu[i:i + 1, vs:vs + GROUP],
                                  repeat=D, count_per_rep=GROUP,
                                  dst_blk_stride=1, dst_rep_stride=ROWBLK,
                                  src1_blk_stride=0, src1_rep_stride=1,
                                  src2_blk_stride=1, src2_rep_stride=0)

                delta[bb, cc, hv, 0:C, 0:D] <<= uu[0:C, 0:D]

            final_state[bb, hv, 0:D, 0:D] <<= su[0:D, 0:D]

    return states, delta, final_state
