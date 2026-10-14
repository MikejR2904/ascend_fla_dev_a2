# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# -----------------------------------------------------------------------------
# PyPTO-Pro Stage 3.5 (KAL pre-build) -- gated_delta_rule_fwd the four chunk-prep steps
# (cumulative ; outputs the u/w step's results u, w).
#
# Appends the u/w step (uw_cube, golden_step s10, Cube) onto the sim-verified
# the first three chunk-prep steps kernel (the previous rung of the KAL chain).  the first three chunk-prep steps are
# reused VERBATIM (same @vf bodies, same gram Cube matmul path, same WY WY post-
# process) with ONE increment: the preprocess step now additionally materializes v_b = v *
# beta_act into GM workspace (v_b is a preprocess output that the first three chunk-prep steps did not
# need, so the WY rung skipped it; the u/w step consumes it).  the first three chunk-prep steps's seven
# former deliverable outputs (decay_exp/q_de/kbeta_de/dfac/L_mask/attn/A) become
# GM workspace here; the u/w step reads attn + kbeta_de (WY) and v_b (preprocess).
#
# the u/w step (Cube, fp32), per (b,h,n) chunk of C=64:
#   u = attn @ v_b        -> [B,H,N,C,V]   (k_cumsum, i.e. the new v)
#   w = attn @ kbeta_de   -> [B,H,N,C,K]   (k_cumdecay)
# Semantics follow golden_chunk_prep.py EXACTLY.
# Contraction axis is C=64 (a single tile).  Output free dims are V (u) and K
# (w), each up to 128 (a single N tile; TK=128 upper bound).  The tail-chunk
# (vc<C) padding is already baked into attn (padded diagonal = 1, off-diagonal
# padded rows/cols contribute 0) and v_b/kbeta_de (padded rows = 0) by Modules
# 1..3, so the u/w step matmuls the FULL C=64 with NO extra tail masking and
# reproduces the golden's full-C outputs.
#
# MATMUL CONVENTION (easyasc a5 shortcut): matmul(dst, l1a, l1b) computes
#   dst = l1a @ l1b**T   with l1a read [M,K], l1b read [N,K].
# the u/w step needs a STANDARD product attn @ X (X = v_b or kbeta_de, stored [C,V]
# / [C,K] = [K,N]), so the second operand is passed transposed: matmul(l0c_u,
# l1_attn, l1_vb.T) = attn @ (v_b**T)**T = attn @ v_b.  The fp32 L1-side .T
# (NZ2ZN L1->L0B transpose) is the corpus idiom (kernels/a5/attention/mha_ifa.py
# P@V; kernels/a5/matmul/matmul_kmkn_fp32_out.py fp32 .T@.T).
#
# MATMUL OPERAND BRIDGING (load-bearing, finding #5): all three u/w
# operands (attn, v_b, kbeta_de) live in GM workspace and are loaded GM->L1 via
# the MTE2 nd2nz hardware path -- NEVER on-chip UB->L1 (which would materialize
# same-size NZ-staging copies in UB and overflow the 253952 B budget even though
# the easyasc sim, not modelling that staging, would pass).  The L1-side .T is a
# L1->L0B transpose, not a UB->L1 bridge, so it does not add UB NZ-staging.
#
# Dataflow (DESIGN §0/§10): one mixed @kernel, a vec->cube->vec->cube pipeline
# with THREE ownership handoffs (structure mirrors the delivered cube->vec->cube
# kernels/a5/attention/mha_ifa.py QK->softmax->PV, extended with the leading
# vec->cube of the WY rung):
#   [Vec  preprocess] preprocess -> GM ws (q_s/k_f32/k_beta/v_b/g_f32/ltones)
#   --VcMutex(0, vec->cube)-->
#   [Cube gram] load GM ws GM->L1 (nd2nz), 3 fp32 matmuls -> decay/kk_gram/qk_gram GM ws
#   --CvMutex(1, cube->vec)-->
#   [Vec  WY] read decay/kk_gram/qk_gram + q_s/k_beta GM ws -> 7 GM ws outputs (incl attn,
#             kbeta_de)
#   --VcMutex(2, vec->cube)-->
#   [Cube u/w] load attn/v_b/kbeta_de GM->L1 (nd2nz), 2 fp32 matmuls -> u, w
# Authored against the easyasc a5 facade, validated in the OpExec simulator
# against golden_chunk_prep.py, then backend-
# transpiled to PyPTO-Pro DSL.  No PyPTO / PyPTO-Pro API is hand-written.
# -----------------------------------------------------------------------------
import os
import sys

# --- bootstrap easyasc payload onto sys.path (do not modify session env) ---
for _cand in (
    "/workspace/pypto-gym/cannbot-skills/ops/pypto-kernel-arch-lang/payload",
    "/root/.claude/skills/pypto-kernel-arch-lang/payload",
):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

import torch

from easyasc.a5 import *  # noqa: F401,F403
from easyasc.simulator import SimulatorConfig


# ---- operator-frozen constants ----
C = 64            # chunk_size (SPEC frozen internal constant)
TILE = 128        # on-chip K/V tile width (TK=128); compile-time literal
SUB = 64          # fp32 register lane count (one Reg = 64 fp32 lanes)
HMAX = 32         # compile-time head-count bound for the per-token g/beta tile
SIM_CORE_COUNT = 32


# =====================================================================
# Vector functions (@vf) -- the preprocess step preprocessing (reused verbatim from
# the previous rung of the KAL chain; scale_vtile_vf is the preprocess v_b increment)
# =====================================================================
@vf()
def zero_fill_vf(ub_zero: Tensor):
    """Fill the [C, TILE] scratch tile with 0.0 (used as a zero DMA source)."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    z = Reg(DT.float)
    dup(z, 0.0, mask=active)
    for r in range(C, name="zr"):
        for cb in range(TILE // SUB, name="zcb"):
            col0 = cb * SUB
            reg_to_ub_normal(ub_zero[r:r + 1, col0:col0 + SUB], z, mask=active)


@vf()
def build_lower_tri_ones_vf(ub_lt: Tensor):
    """Lower-triangular ones [C,C]: LTones[r,j] = 1.0 if j <= r else 0.0."""
    cols = Reg(DT.int)
    zero = Reg(DT.float)
    row = Reg(DT.float)
    m = MaskReg(DT.int, init_mode=MaskType.NONE)
    zero <<= 0.0
    cols.arange(0)
    for r in range(C, name="ltr"):
        row <<= 1.0
        compare(m, cols, r + 1, CompareMode.LT)   # cols < r+1  <=>  j <= r
        select(row, row, zero, mask=m)
        ub_lt[r:r + 1, 0:C] <<= row


@vf()
def prep_scalars_vf(ub_bh: Tensor, ub_gh: Tensor, ub_bact: Tensor, ub_gf: Tensor,
                    vc: Var, h_idx: Var):
    """beta_act[r] = sigmoid(beta_logit[r,h]); g_f32[r] = g[r,h]."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    to_float = CastConfig(round_mode=RoundMode.NONE, reg_layout=RegLayout.ZERO, name="beta_b2f")
    zero = Reg(DT.float)
    one = Reg(DT.float)
    dup(zero, 0.0, mask=active)
    dup(one, 1.0, mask=active)
    for r in range(vc, name="sr"):
        blog_bf = Reg(DT.bfloat16)
        blog_f = Reg(DT.float)
        den = Reg(DT.float)
        blog_bf <<= ub_bh[r:r + 1, h_idx:h_idx + 1].single()
        cast(blog_f, blog_bf, to_float, mask=active)
        expsub(den, zero, blog_f, mask=active)   # exp(-beta)
        adds(den, den, 1.0, mask=active)
        div(den, one, den, mask=active)          # sigmoid
        ub_bact[0:1, r:r + 1] <<= den.single_value()
        gval = Reg(DT.float)
        gval <<= ub_gh[r:r + 1, h_idx:h_idx + 1].single()
        ub_gf[0:1, r:r + 1] <<= gval.single_value()


@vf()
def scale_ktile_vf(ub_q: Tensor, ub_k: Tensor, ub_bact: Tensor,
                   ub_qs: Tensor, ub_kf: Tensor, ub_kbeta: Tensor,
                   vc: Var, ncsub: Var, scale: Var):
    """For each valid token row r: q_s = q*scale, k_f32 = k, k_beta = k*beta_act[r]."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    to_float = CastConfig(round_mode=RoundMode.NONE, reg_layout=RegLayout.ZERO, name="k_b2f")
    for r in range(vc, name="kr"):
        bs = Reg(DT.float)
        bs <<= ub_bact[0:1, r:r + 1].single()
        for cb in range(ncsub, name="kcb"):
            col0 = Var(cb * SUB)
            qsrc = Reg(DT.bfloat16)
            qf = Reg(DT.float)
            qs = Reg(DT.float)
            ub_to_reg_unpack(qsrc, ub_q[r:r + 1, col0:col0 + SUB])
            cast(qf, qsrc, to_float, mask=active)
            muls(qs, qf, scale, mask=active)
            reg_to_ub_normal(ub_qs[r:r + 1, col0:col0 + SUB], qs, mask=active)

            ksrc = Reg(DT.bfloat16)
            kf = Reg(DT.float)
            kb = Reg(DT.float)
            ub_to_reg_unpack(ksrc, ub_k[r:r + 1, col0:col0 + SUB])
            cast(kf, ksrc, to_float, mask=active)
            reg_to_ub_normal(ub_kf[r:r + 1, col0:col0 + SUB], kf, mask=active)
            mul(kb, kf, bs, mask=active)
            reg_to_ub_normal(ub_kbeta[r:r + 1, col0:col0 + SUB], kb, mask=active)


@vf()
def cast_vtile_vf(ub_v: Tensor, ub_vf: Tensor, vc: Var, nvsub: Var):
    """preprocess: v_f32[r,:] = (bf16->fp32) v[r,:]  (cast only, no scale).

    Mirrors the k_f32 sub-computation of scale_ktile_vf: v_f32_ws is the fp32
    staging of v exactly as k_f32_ws is of k, so the WY step can build v_b from a
    workspace via rowscale_vf (the same GM-read/scale/GM-write shape that
    produces kbeta_de and reaches the u/w step correctly).
    """
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    to_float = CastConfig(round_mode=RoundMode.NONE, reg_layout=RegLayout.ZERO, name="v_b2f")
    for r in range(vc, name="vr"):
        for cb in range(nvsub, name="vcb"):
            col0 = Var(cb * SUB)
            vsrc = Reg(DT.bfloat16)
            vfp = Reg(DT.float)
            ub_to_reg_unpack(vsrc, ub_v[r:r + 1, col0:col0 + SUB])
            cast(vfp, vsrc, to_float, mask=active)
            reg_to_ub_normal(ub_vf[r:r + 1, col0:col0 + SUB], vfp, mask=active)


# =====================================================================
# Vector functions (@vf) -- the WY step (wy_build_vector); all fp32
# (reused verbatim from the previous rung of the KAL chain)
# =====================================================================
@vf()
def decay_exp_vf(ub_dec: Tensor, ub_dexp: Tensor):
    """decay_exp[c] = exp(decay[c]) over the full C=64 row (one fp32 Reg)."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    zero = Reg(DT.float)
    dup(zero, 0.0, mask=active)
    d = Reg(DT.float)
    de = Reg(DT.float)
    d <<= ub_dec[0:1, 0:C]
    expsub(de, d, zero, mask=active)     # exp(decay - 0)
    ub_dexp[0:1, 0:C] <<= de


@vf()
def decay_mask_vf(ub_dec: Tensor, ub_L: Tensor):
    """L_mask[i,j] = exp(decay_i - decay_j) for j<=i else 0 (lower-tri incl diag)."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    cols = Reg(DT.int)
    zero = Reg(DT.float)
    dup(zero, 0.0, mask=active)
    cols.arange(0)
    dvec = Reg(DT.float)
    dvec <<= ub_dec[0:1, 0:C]            # decay_j vector
    for i in range(C, name="li"):
        di = Reg(DT.float)
        di <<= ub_dec[0:1, i:i + 1].single()   # scalar decay_i (broadcast)
        lrow = Reg(DT.float)
        expsub(lrow, di, dvec, mask=active)     # exp(decay_i - decay_j)
        m = MaskReg(DT.int, init_mode=MaskType.NONE)
        compare(m, cols, i + 1, CompareMode.LT)  # cols < i+1  <=>  j <= i
        select(lrow, lrow, zero, mask=m)
        ub_L[i:i + 1, 0:C] <<= lrow


@vf()
def wy_lower_init_vf(ub_kk_gram: Tensor, ub_L: Tensor, ub_attn: Tensor):
    """attn_pre[i,j] = -kk_gram[i,j]*L_mask[i,j] for j<i else 0 (strict lower)."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    cols = Reg(DT.int)
    zero = Reg(DT.float)
    dup(zero, 0.0, mask=active)
    cols.arange(0)
    for i in range(C, name="ai"):
        p = Reg(DT.float)
        l = Reg(DT.float)
        p <<= ub_kk_gram[i:i + 1, 0:C]
        l <<= ub_L[i:i + 1, 0:C]
        r = Reg(DT.float)
        mul(r, p, l, mask=active)
        neg(r, r, mask=active)
        m = MaskReg(DT.int, init_mode=MaskType.NONE)
        compare(m, cols, i, CompareMode.LT)      # cols < i  <=>  j < i (strict lower)
        select(r, r, zero, mask=m)
        ub_attn[i:i + 1, 0:C] <<= r


@vf()
def intra_attn_build_vf(ub_qk_gram: Tensor, ub_L: Tensor, ub_A: Tensor):
    """A[i,j] = qk_gram[i,j]*L_mask[i,j] for j<=i else 0 (lower incl diag)."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    cols = Reg(DT.int)
    zero = Reg(DT.float)
    dup(zero, 0.0, mask=active)
    cols.arange(0)
    for i in range(C, name="Ai"):
        p = Reg(DT.float)
        l = Reg(DT.float)
        p <<= ub_qk_gram[i:i + 1, 0:C]
        l <<= ub_L[i:i + 1, 0:C]
        r = Reg(DT.float)
        mul(r, p, l, mask=active)
        m = MaskReg(DT.int, init_mode=MaskType.NONE)
        compare(m, cols, i + 1, CompareMode.LT)  # cols < i+1  <=>  j <= i
        select(r, r, zero, mask=m)
        ub_A[i:i + 1, 0:C] <<= r


@vf()
def wy_forward_substitution_vf(ub_attn: Tensor):
    """s8 forward substitution + s9 identity, in place on the strict-lower attn.

    For i=1..C-1:  new_row_i = old_row_i + sum_{m<i} old_row_i[m] * attn[m,:]
    (the accumulate is a fp32 MulAddDst FMA over the full 64-lane row; strict-
    lower zeros make column masking unnecessary).  Then add I(C) on the diagonal.
    """
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    one = Reg(DT.float)
    dup(one, 1.0, mask=active)
    for i in range(1, C, name="fi"):
        acc = Reg(DT.float)
        dup(acc, 0.0, mask=active)
        for m in range(i, name="fm"):
            sc = Reg(DT.float)
            row_m = Reg(DT.float)
            sc <<= ub_attn[i:i + 1, m:m + 1].single()   # old_row_i[m] (broadcast)
            row_m <<= ub_attn[m:m + 1, 0:C]              # finalized attn row m
            muladddst(acc, row_m, sc, mask=active)       # acc = row_m*sc + acc
        old_i = Reg(DT.float)
        newi = Reg(DT.float)
        old_i <<= ub_attn[i:i + 1, 0:C]
        add(newi, old_i, acc, mask=active)
        ub_attn[i:i + 1, 0:C] <<= newi
        # loop-carried RMW recurrence: the next row's reads of finalized rows
        # must observe this store (scatter_recurrence_probe rmw_fenced idiom).
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    # order the row writeback (above) before the diagonal +I writes (below);
    # both touch the same attn rows.
    vf_barrier(VfPipe.STORE, VfPipe.STORE)
    for i in range(C, name="fid"):
        ub_attn[i:i + 1, i:i + 1] <<= one.single_value()   # +I(C) diagonal


@vf()
def rowscale_vf(ub_src: Tensor, ub_dexp: Tensor, ub_dst: Tensor, ncsub: Var):
    """dst[c,:] = src[c,:] * decay_exp[c]  (q_de / kbeta_de), full C rows."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    for c in range(C, name="rc"):
        sc = Reg(DT.float)
        sc <<= ub_dexp[0:1, c:c + 1].single()     # decay_exp[c] (broadcast)
        for cb in range(ncsub, name="rcb"):
            col0 = Var(cb * SUB)
            row = Reg(DT.float)
            row <<= ub_src[c:c + 1, col0:col0 + SUB]
            mul(row, row, sc, mask=active)
            ub_dst[c:c + 1, col0:col0 + SUB] <<= row


@vf()
def decay_tail_vf(ub_dec: Tensor, ub_dfac: Tensor, ncsub: Var):
    """dfac[c,k] = exp(decay[C-1] - decay[c]) broadcast over k, full C rows."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    dlast = Reg(DT.float)
    dlast <<= ub_dec[0:1, (C - 1):C].single()     # scalar decay[C-1] (broadcast)
    for c in range(C, name="dc"):
        dc = Reg(DT.float)
        dc <<= ub_dec[0:1, c:c + 1].single()
        s = Reg(DT.float)
        expsub(s, dlast, dc, mask=active)         # exp(decay_last - decay_c)
        for cb in range(ncsub, name="dcb"):
            col0 = Var(cb * SUB)
            ub_dfac[c:c + 1, col0:col0 + SUB] <<= s


# =====================================================================
# Kernel -- mixed Vec(preprocess) -> Cube(gram) -> Vec(WY) -> Cube(u/w)
# =====================================================================
@kernel()
def gated_delta_rule_fwd_chunk_prep_kernel(
    q: GMTensor, k: GMTensor, v: GMTensor, g: GMTensor, beta: GMTensor,
    q_s_ws: GMTensor, k_f32_ws: GMTensor, k_beta_ws: GMTensor,
    v_f32_ws: GMTensor, v_b_ws: GMTensor,
    g_f32_ws: GMTensor, ltones_ws: GMTensor,
    decay_ws: GMTensor, P1_ws: GMTensor, P2_ws: GMTensor,
    decay_exp_ws: GMTensor, q_de_ws: GMTensor, kbeta_de_ws: GMTensor,
    dfac_ws: GMTensor, L_mask_ws: GMTensor, attn_ws: GMTensor, A_ws: GMTensor,
    u: GMTensor, w: GMTensor,
    B: Var, T: Var, H: Var, Kd: Var, Vd: Var, N: Var, Cc: Var, VT: Var, scale: Var,
):
    # preprocess(vec) -> gram(cube): vec ends on MTE3, cube ends on FIX.
    vcmutex = VcMutex(0, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    # gram(cube) -> WY(vec): cube ends on FIX (GM store), vec consumer on V.
    cvmutex = CvMutex(1, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    # WY(vec) -> u/w(cube): vec ends on MTE3 (attn/kbeta_de GM store), cube on FIX.
    vcmutex2 = VcMutex(2, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)

    # ---- Vector-side UB scratch (preprocess) ----
    ub_zero = Tensor(DT.float, [C, TILE], Position.UB)
    ub_q = Tensor(DT.bfloat16, [C, TILE], Position.UB)
    ub_k = Tensor(DT.bfloat16, [C, TILE], Position.UB)
    ub_bh = Tensor(DT.bfloat16, [C, HMAX], Position.UB)
    ub_gh = Tensor(DT.float, [C, HMAX], Position.UB)
    ub_bact = Tensor(DT.float, [1, C], Position.UB)
    ub_gf = Tensor(DT.float, [1, C], Position.UB)
    ub_qs = Tensor(DT.float, [C, TILE], Position.UB)
    ub_kf = Tensor(DT.float, [C, TILE], Position.UB)
    ub_kbeta = Tensor(DT.float, [C, TILE], Position.UB)
    ub_lt = Tensor(DT.float, [C, C], Position.UB)

    # ---- Vector-side UB scratch (WY, dedicated small/[C,C] tiles) ----
    ub_dec = Tensor(DT.float, [1, C], Position.UB)     # decay (read from GM ws)
    ub_dexp = Tensor(DT.float, [1, C], Position.UB)    # decay_exp
    ub_p1 = Tensor(DT.float, [C, C], Position.UB)      # kk_gram -> attn (in place)
    ub_p2 = Tensor(DT.float, [C, C], Position.UB)      # qk_gram -> A    (in place)
    # L_mask reuses ub_lt; q_de/kbeta_de/dfac staging reuse ub_qs/ub_kbeta/ub_kf;
    # preprocess's v_b staging reuses ub_q (bf16 v input) and ub_qs (fp32 v_b output).

    # ---- Cube-side operands (L1) and accumulators (L0C) ----
    l1_qs = Tensor(DT.float, [C, TILE], Position.L1)
    l1_kf = Tensor(DT.float, [C, TILE], Position.L1)
    l1_kbeta = Tensor(DT.float, [C, TILE], Position.L1)
    l1_g = Tensor(DT.float, [16, C], Position.L1)
    l1_lt = Tensor(DT.float, [C, C], Position.L1)
    l0c_decay = Tensor(DT.float, [16, C], Position.L0C)
    l0c_p1 = Tensor(DT.float, [C, C], Position.L0C)
    l0c_p2 = Tensor(DT.float, [C, C], Position.L0C)
    # ---- Cube-side operands/accumulators (u/w) ----
    l1_attn = Tensor(DT.float, [C, C], Position.L1)    # attn [M=C, K=C]
    l1_vb = Tensor(DT.float, [C, TILE], Position.L1)   # v_b [K=C, N=V]; .T at mm
    l1_kbde = Tensor(DT.float, [C, TILE], Position.L1)  # kbeta_de [K=C, N=K]; .T
    l0c_u = Tensor(DT.float, [C, TILE], Position.L0C)  # u = attn @ v_b [C,V]
    l0c_w = Tensor(DT.float, [C, TILE], Position.L0C)  # w = attn @ kbeta_de [C,K]

    k_tiles = CeilDiv(Kd, TILE)
    v_tiles = CeilDiv(Vd, TILE)
    tasks = B * H * N
    per = CeilDiv(tasks, GetCubeNum())
    begin = Var(per * GetCubeIdx())
    end = Min(begin + per, tasks)

    with auto_sync():
        for tid in range(begin, end, name="tid"):
            b_idx = Var(tid // (H * N))
            rem = Var(tid % (H * N))
            h_idx = Var(rem // N)
            n_idx = Var(rem % N)
            t0 = Var(n_idx * C)
            vc = Min(T - t0, C)
            pad = Var(C - vc)

            # =====================================================
            # preprocess (Vector, AIV sub-block 0) -> GM workspace (adds v_b)
            # =====================================================
            vcmutex.lock()
            row_start = Var(GetSubBlockIdx() * C)
            if row_start < C:
                zero_fill_vf(ub_zero)

                ub_bh[0:vc, 0:H] <<= beta[b_idx, t0:t0 + vc, 0:H]
                ub_gh[0:vc, 0:H] <<= g[b_idx, t0:t0 + vc, 0:H]
                prep_scalars_vf(ub_bh, ub_gh, ub_bact, ub_gf, vc, h_idx)
                g_f32_ws[b_idx, h_idx, n_idx, 0:vc] <<= ub_gf[0:1, 0:vc]
                if vc < C:
                    g_f32_ws[b_idx, h_idx, n_idx, vc:C] <<= ub_zero[0:1, 0:pad]

                for kt in range(k_tiles, name="kt"):
                    k0 = Var(kt * TILE)
                    tk = Min(Kd - k0, TILE)
                    ncsub_k = CeilDiv(tk, SUB)
                    ub_q[0:vc, 0:tk] <<= q[b_idx, t0:t0 + vc, h_idx, k0:k0 + tk]
                    ub_k[0:vc, 0:tk] <<= k[b_idx, t0:t0 + vc, h_idx, k0:k0 + tk]
                    scale_ktile_vf(ub_q, ub_k, ub_bact, ub_qs, ub_kf, ub_kbeta,
                                   vc, ncsub_k, scale)
                    q_s_ws[b_idx, h_idx, n_idx, 0:vc, k0:k0 + tk] <<= ub_qs[0:vc, 0:tk]
                    k_f32_ws[b_idx, h_idx, n_idx, 0:vc, k0:k0 + tk] <<= ub_kf[0:vc, 0:tk]
                    k_beta_ws[b_idx, h_idx, n_idx, 0:vc, k0:k0 + tk] <<= ub_kbeta[0:vc, 0:tk]
                    if vc < C:
                        q_s_ws[b_idx, h_idx, n_idx, vc:C, k0:k0 + tk] <<= ub_zero[0:pad, 0:tk]
                        k_f32_ws[b_idx, h_idx, n_idx, vc:C, k0:k0 + tk] <<= ub_zero[0:pad, 0:tk]
                        k_beta_ws[b_idx, h_idx, n_idx, vc:C, k0:k0 + tk] <<= ub_zero[0:pad, 0:tk]

                # v_f32 = (bf16->fp32) v  -- preprocess fp32 staging of v, the exact
                # mirror of k_f32_ws. v_b itself is built in the WY step from this
                # workspace via rowscale_vf (see the WY vt3 loop): that reproduces
                # kbeta_de's proven GM-read -> in-place scale -> GM-write shape,
                # which reaches the u/w step correctly. (Building v_b directly from the
                # primary input v in a bespoke cast+scale vf produced a v_b_ws that
                # the u/w step read stale/misaligned for V<TILE in the simulator; the
                # workspace-sourced rowscale path does not.) Reuse ub_q (bf16 v
                # input) and ub_qs (fp32 output), free after the k-loop.
                for vt in range(v_tiles, name="vt"):
                    v0 = Var(vt * TILE)
                    tv = Min(Vd - v0, TILE)
                    nvsub_v = CeilDiv(tv, SUB)
                    ub_q[0:vc, 0:tv] <<= v[b_idx, t0:t0 + vc, h_idx, v0:v0 + tv]
                    cast_vtile_vf(ub_q, ub_qs, vc, nvsub_v)
                    v_f32_ws[b_idx, h_idx, n_idx, 0:vc, v0:v0 + tv] <<= ub_qs[0:vc, 0:tv]
                    if vc < C:
                        v_f32_ws[b_idx, h_idx, n_idx, vc:C, v0:v0 + tv] <<= ub_zero[0:pad, 0:tv]

                build_lower_tri_ones_vf(ub_lt)
                ltones_ws[0:C, 0:C] <<= ub_lt[0:C, 0:C]
            vcmutex.ready()

            # =====================================================
            # gram (Cube, AIC): 3 fp32 matmuls -> decay/kk_gram/qk_gram GM workspace
            # =====================================================
            vcmutex.wait()
            l1_g[0:1, 0:C] <<= g_f32_ws[b_idx, h_idx, n_idx, 0:C]
            l1_lt[0:C, 0:C] <<= ltones_ws[0:C, 0:C]
            matmul(l0c_decay, l1_g, l1_lt, m=1, n=C, k=C)
            l1_kbeta[0:C, 0:Kd] <<= k_beta_ws[b_idx, h_idx, n_idx, 0:C, 0:Kd]
            l1_kf[0:C, 0:Kd] <<= k_f32_ws[b_idx, h_idx, n_idx, 0:C, 0:Kd]
            matmul(l0c_p1, l1_kbeta, l1_kf, m=C, n=C, k=Kd)
            l1_qs[0:C, 0:Kd] <<= q_s_ws[b_idx, h_idx, n_idx, 0:C, 0:Kd]
            matmul(l0c_p2, l1_qs, l1_kf, m=C, n=C, k=Kd)

            cvmutex.lock()
            decay_ws[b_idx, h_idx, n_idx, 0:C] <<= l0c_decay[0:1, 0:C]
            P1_ws[b_idx, h_idx, n_idx, 0:C, 0:C] <<= l0c_p1[0:C, 0:C]
            P2_ws[b_idx, h_idx, n_idx, 0:C, 0:C] <<= l0c_p2[0:C, 0:C]
            cvmutex.ready()

            # =====================================================
            # WY (Vector, AIV sub-block 0): read GM ws -> 7 GM ws outputs
            # (full C=64; tail padding already baked into decay/kk_gram/qk_gram)
            # =====================================================
            # Flat handoff (mirrors kernels/a5/attention/mha_ifa.py and gdn_legacy/
            # recompute_wu.py): the vec side first CONSUMES the gram step's cube output
            # into UB under cvmutex, releases cvmutex, then PRODUCES the WY step's GM
            # outputs under vcmutex2 -- the two mutex regions are sequential, not
            # nested. (Nesting vcmutex2 inside cvmutex.wait/free left u/w's read of
            # the vec-produced v_b_ws unordered against its store in the simulator.)
            cvmutex.wait()
            if row_start < C:
                ub_dec[0:1, 0:C] <<= decay_ws[b_idx, h_idx, n_idx, 0:C]
                ub_p1[0:C, 0:C] <<= P1_ws[b_idx, h_idx, n_idx, 0:C, 0:C]
                ub_p2[0:C, 0:C] <<= P2_ws[b_idx, h_idx, n_idx, 0:C, 0:C]
            cvmutex.free()

            vcmutex2.lock()
            if row_start < C:
                # decay_exp
                decay_exp_vf(ub_dec, ub_dexp)
                decay_exp_ws[b_idx, h_idx, n_idx, 0:C] <<= ub_dexp[0:1, 0:C]

                # L_mask (reuse ub_lt)
                decay_mask_vf(ub_dec, ub_lt)
                L_mask_ws[b_idx, h_idx, n_idx, 0:C, 0:C] <<= ub_lt[0:C, 0:C]

                # attn = -tril_strict(kk_gram*L) -> forward substitution -> +I
                wy_lower_init_vf(ub_p1, ub_lt, ub_p1)
                wy_forward_substitution_vf(ub_p1)
                attn_ws[b_idx, h_idx, n_idx, 0:C, 0:C] <<= ub_p1[0:C, 0:C]

                # A = tril(qk_gram*L)  (j<=i)
                intra_attn_build_vf(ub_p2, ub_lt, ub_p2)
                A_ws[b_idx, h_idx, n_idx, 0:C, 0:C] <<= ub_p2[0:C, 0:C]

                # q_de / kbeta_de / dfac over K tiles (reuse ub_qs/ub_kbeta/ub_kf)
                for kt in range(k_tiles, name="kt3"):
                    k0 = Var(kt * TILE)
                    tk = Min(Kd - k0, TILE)
                    ncsub_k = CeilDiv(tk, SUB)
                    ub_qs[0:C, 0:tk] <<= q_s_ws[b_idx, h_idx, n_idx, 0:C, k0:k0 + tk]
                    rowscale_vf(ub_qs, ub_dexp, ub_qs, ncsub_k)
                    q_de_ws[b_idx, h_idx, n_idx, 0:C, k0:k0 + tk] <<= ub_qs[0:C, 0:tk]

                    ub_kbeta[0:C, 0:tk] <<= k_beta_ws[b_idx, h_idx, n_idx, 0:C, k0:k0 + tk]
                    rowscale_vf(ub_kbeta, ub_dexp, ub_kbeta, ncsub_k)
                    kbeta_de_ws[b_idx, h_idx, n_idx, 0:C, k0:k0 + tk] <<= ub_kbeta[0:C, 0:tk]

                    decay_tail_vf(ub_dec, ub_kf, ncsub_k)
                    dfac_ws[b_idx, h_idx, n_idx, 0:C, k0:k0 + tk] <<= ub_kf[0:C, 0:tk]

                # v_b = v_f32 * beta_act  (preprocess quantity, built HERE in WY so
                # its GM store is under vcmutex2's vec->cube handoff to u/w). This
                # mirrors the kbeta_de path exactly: load an preprocess fp32 workspace
                # (v_f32_ws, like k_beta_ws), scale IN PLACE by a per-row scalar
                # (beta_act via ub_bact, like decay_exp via ub_dexp), store the
                # result GM. rowscale_vf runs the full C rows; v_f32_ws padding
                # (vc<C rows = 0, zeroed in preprocess) makes v_b padding 0, matching the
                # golden (v and beta padded with 0). beta_act (ub_bact) persists
                # from preprocess on the same vec sub-block.
                nvsub_v = CeilDiv(Vd, SUB)
                ub_qs[0:vc, 0:Vd] <<= v_f32_ws[b_idx, h_idx, n_idx, 0:vc, 0:Vd]
                rowscale_vf(ub_qs, ub_bact, ub_qs, nvsub_v)
                # v_b_ws is padded to the full TILE=VT width: zero the whole tile
                # (all C rows, all TILE cols) then write ONLY the valid tokens
                # [0:vc, 0:Vd]. the u/w step loads the FULL tile and matmuls at the
                # compile-time width n=TILE -- a partial-width (n=Vd<TILE) matmul
                # on the transposed operand mis-reads in the simulator (u wrong for
                # V<TILE while the identical-shape w path is unaffected); the
                # full-width form with zero pad columns/rows yields exactly
                # attn @ v_b in [0:V]. Storing only valid rows also avoids a
                # 0 * (uninitialized beta_act) = NaN in the padded rows.
                v_b_ws[b_idx, h_idx, n_idx, 0:C, 0:TILE] <<= ub_zero[0:C, 0:TILE]
                v_b_ws[b_idx, h_idx, n_idx, 0:vc, 0:Vd] <<= ub_qs[0:vc, 0:Vd]
            vcmutex2.ready()

            # =====================================================
            # u/w (Cube, AIC): u = attn @ v_b, w = attn @ kbeta_de
            # operands GM->L1 nd2nz; standard product via L1-side .T of operand B
            # =====================================================
            vcmutex2.wait()
            l1_attn[0:C, 0:C] <<= attn_ws[b_idx, h_idx, n_idx, 0:C, 0:C]
            l1_vb[0:C, 0:TILE] <<= v_b_ws[b_idx, h_idx, n_idx, 0:C, 0:TILE]
            matmul(l0c_u, l1_attn, l1_vb.T, m=C, n=TILE, k=C)   # attn @ v_b (full width)
            u[b_idx, h_idx, n_idx, 0:C, 0:Vd] <<= l0c_u[0:C, 0:Vd]

            l1_kbde[0:C, 0:Kd] <<= kbeta_de_ws[b_idx, h_idx, n_idx, 0:C, 0:Kd]
            matmul(l0c_w, l1_attn, l1_kbde.T, m=C, n=Kd, k=C)  # attn @ kbeta_de
            w[b_idx, h_idx, n_idx, 0:C, 0:Kd] <<= l0c_w[0:C, 0:Kd]
            vcmutex2.free()

            vcmutex.free()

    return u, w


# =====================================================================
# OpExec simulator self-check entry
# =====================================================================
# axis ids = position in the trailing scalar arg list (B,T,H,Kd,Vd,N,Cc,VT,scale):
#   0=B, 1=T, 2=H, 3=K, 4=V, 5=N, 6=C, 7=VT(=TILE, the padded V tile width)
_SHAPE_BINDINGS = {
    0: [0, 1, 2, 3],          # q            [B,T,H,K]
    1: [0, 1, 2, 3],          # k            [B,T,H,K]
    2: [0, 1, 2, 4],          # v            [B,T,H,V]
    3: [0, 1, 2],             # g            [B,T,H]
    4: [0, 1, 2],             # beta         [B,T,H]
    5: [0, 2, 5, 6, 3],       # q_s_ws       [B,H,N,C,K]
    6: [0, 2, 5, 6, 3],       # k_f32_ws     [B,H,N,C,K]
    7: [0, 2, 5, 6, 3],       # k_beta_ws    [B,H,N,C,K]
    8: [0, 2, 5, 6, 4],       # v_f32_ws     [B,H,N,C,V]
    9: [0, 2, 5, 6, 7],       # v_b_ws       [B,H,N,C,VT]  (V padded to TILE)
    10: [0, 2, 5, 6],         # g_f32_ws     [B,H,N,C]
    11: [6, 6],               # ltones_ws    [C,C]
    12: [0, 2, 5, 6],         # decay_ws     [B,H,N,C]
    13: [0, 2, 5, 6, 6],      # P1_ws        [B,H,N,C,C]
    14: [0, 2, 5, 6, 6],      # P2_ws        [B,H,N,C,C]
    15: [0, 2, 5, 6],         # decay_exp_ws [B,H,N,C]
    16: [0, 2, 5, 6, 3],      # q_de_ws      [B,H,N,C,K]
    17: [0, 2, 5, 6, 3],      # kbeta_de_ws  [B,H,N,C,K]
    18: [0, 2, 5, 6, 3],      # dfac_ws      [B,H,N,C,K]
    19: [0, 2, 5, 6, 6],      # L_mask_ws    [B,H,N,C,C]
    20: [0, 2, 5, 6, 6],      # attn_ws      [B,H,N,C,C]
    21: [0, 2, 5, 6, 6],      # A_ws         [B,H,N,C,C]
    22: [0, 2, 5, 6, 4],      # u            [B,H,N,C,V]
    23: [0, 2, 5, 6, 3],      # w            [B,H,N,C,K]
}


def run_chunk_prep(q, k, v, g, beta, simulator=True, out_dir=""):
    B, T, H, K = q.shape
    V = v.shape[-1]
    N = (T + C - 1) // C
    if H > HMAX:
        raise ValueError(f"this KAL pre-build stages g/beta with HMAX={HMAX}; got H={H}")
    if K > TILE or V > TILE:
        raise ValueError(f"this KAL pre-build uses a single K/V tile TK={TILE}; got K={K}, V={V}")
    device = q.device
    # GM workspace (wrapper-allocated scratch, discarded; not host compute)
    q_s_ws = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    k_f32_ws = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    k_beta_ws = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    v_f32_ws = torch.empty((B, H, N, C, V), dtype=torch.float32, device=device)
    # v_b_ws is padded to the full TILE column width (the u/w step loads the full tile
    # and matmuls at the compile-time width n=TILE; the pad columns are zero).
    v_b_ws = torch.zeros((B, H, N, C, TILE), dtype=torch.float32, device=device)
    g_f32_ws = torch.empty((B, H, N, C), dtype=torch.float32, device=device)
    ltones_ws = torch.empty((C, C), dtype=torch.float32, device=device)
    decay_ws = torch.empty((B, H, N, C), dtype=torch.float32, device=device)
    P1_ws = torch.empty((B, H, N, C, C), dtype=torch.float32, device=device)
    P2_ws = torch.empty((B, H, N, C, C), dtype=torch.float32, device=device)
    decay_exp_ws = torch.empty((B, H, N, C), dtype=torch.float32, device=device)
    q_de_ws = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    kbeta_de_ws = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    dfac_ws = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    L_mask_ws = torch.empty((B, H, N, C, C), dtype=torch.float32, device=device)
    attn_ws = torch.empty((B, H, N, C, C), dtype=torch.float32, device=device)
    A_ws = torch.empty((B, H, N, C, C), dtype=torch.float32, device=device)
    # the u/w step outputs
    u = torch.empty((B, H, N, C, V), dtype=torch.float32, device=device)
    w = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)

    if simulator:
        gated_delta_rule_fwd_chunk_prep_kernel._simulator_config = SimulatorConfig(
            core_count=SIM_CORE_COUNT, execution_timeout_s=1800.0,
        )
    scale = 1.0 / (K ** 0.5)
    return OpExec(gated_delta_rule_fwd_chunk_prep_kernel, out_dir=out_dir, simulator=simulator)(
        q, k, v, g, beta,
        q_s_ws, k_f32_ws, k_beta_ws, v_f32_ws, v_b_ws, g_f32_ws, ltones_ws,
        decay_ws, P1_ws, P2_ws,
        decay_exp_ws, q_de_ws, kbeta_de_ws, dfac_ws, L_mask_ws, attn_ws, A_ws,
        u, w,
        B, T, H, K, V, N, C, TILE, scale,
        shape_bindings=_SHAPE_BINDINGS,
    )


def _load_golden():
    import importlib.util
    gp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "golden_chunk_prep.py")
    spec = importlib.util.spec_from_file_location("gdr_golden_chunk_prep", gp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mk_case(B, T, H, K, V, seed):
    torch.manual_seed(seed)
    q = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    k = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    g = torch.empty((B, T, H), dtype=torch.float32).uniform_(-4.0, 0.0)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16)
    return q, k, v, g, beta


def _mixed_ok(got, exp, atol, rtol, l2_rtol):
    """DESIGN §9 mixed criterion: elementwise allclose OR peak-scaled relative L2.

    u/w inherit the K=128 forward-substitution amplification of fp32 matmul (kk_gram)
    rounding vs. the golden's torch matmul; those elements have large |exp| so
    they clear the elementwise rtol term, but the relative-L2 fallback covers any
    residual peak-magnitude cases without loosening the nominal tolerance.
    """
    gf = got.float()
    ef = exp.float()
    if torch.allclose(gf, ef, rtol=rtol, atol=atol):
        return True, (gf - ef).abs().max().item(), 0.0
    rel_l2 = ((gf - ef).norm() / (ef.norm() + 1e-12)).item()
    return rel_l2 <= l2_rtol, (gf - ef).abs().max().item(), rel_l2


if __name__ == "__main__":
    golden = _load_golden()
    # module_interfaces.yaml composition_verification shapes / seeds.
    shapes = [(1, 128, 4, 64, 64), (1, 100, 4, 64, 64),
              (1, 192, 2, 128, 128), (1, 200, 3, 128, 96)]
    seeds = [42, 123, 456]
    names = ["u", "w"]
    ATOL, RTOL, L2_RTOL = 1e-2, 1e-2, 2e-2
    all_pass = True
    for (B, T, H, K, V) in shapes:
        for seed in seeds:
            q, k, v, g, beta = _mk_case(B, T, H, K, V, seed)
            ref = golden.gated_delta_rule_fwd_golden_chunk_prep(q, k, v, g, beta)
            out = run_chunk_prep(q, k, v, g, beta, simulator=True)
            outs = out if isinstance(out, tuple) else (out,)
            worst = 0.0
            worst_l2 = 0.0
            case_ok = True
            for nm, got, exp in zip(names, outs, ref):
                if tuple(got.shape) != tuple(exp.shape):
                    print(f"  [FAIL] {nm} shape {tuple(got.shape)} != {tuple(exp.shape)}")
                    case_ok = False; all_pass = False; continue
                if not torch.isfinite(got).all():
                    print(f"  [FAIL] {nm} not finite (NaN/Inf present)")
                    case_ok = False; all_pass = False
                ok, d, rel_l2 = _mixed_ok(got, exp, ATOL, RTOL, L2_RTOL)
                worst = max(worst, d)
                worst_l2 = max(worst_l2, rel_l2)
                if not ok:
                    print(f"  [FAIL] {nm} max_abs={d:.3e} rel_l2={rel_l2:.3e} "
                          f"B{B}T{T}H{H}K{K}V{V} seed{seed}")
                    case_ok = False; all_pass = False
            print(f"  [{'PASS' if case_ok else 'FAIL'}] B{B}T{T}H{H}K{K}V{V} seed{seed} "
                  f"max_abs={worst:.3e} rel_l2={worst_l2:.3e}")
    print("ALL PASS" if all_pass else "FAILURES PRESENT")
    if not all_pass:
        sys.exit(1)
