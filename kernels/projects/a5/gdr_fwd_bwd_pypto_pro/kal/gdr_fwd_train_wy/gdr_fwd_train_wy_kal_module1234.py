# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# -----------------------------------------------------------------------------
# PyPTO-Pro Stage 3.5 (KAL pre-build) -- gdr_fwd_train_wy (OP-A of the
# gated_delta_rule TRAINING-FORWARD pair): the WY-build front-end (Modules
# M1..M4, BHN-parallel).  Emits all 17 GM boundary tensors of the SPEC contract.
#
# RE-HOMING, not fresh authoring.  This file is a VERBATIM re-home of the banked,
# sim-verified seed
#     custom/gated_delta_rule_bwd/kal/gated_delta_rule_bwd_kal_module1234.py
# (127,744 B / 28-17 mutex / 0 flags, sim 12/12).  The M1..M4 mathematics is
# byte-for-byte the seed's; only which tensors survive the call changed.  The
# seed already GM-materialises 14 of the 17 SPEC outputs (q_s, k_f32, v_b, g_f32,
# decay, decay_exp, dfac, q_de, attn, a_mat, kbeta_t, u, w, w_t).  The re-home
# delta is exactly:
#   (a) promote the 11 formerly-workspace tensors to declared outputs (return-list
#       only for 10; v_b gets a valid-width store because the seed's v_b_ws is
#       zero-padded to VT for M4's fixed-64 read, but the SPEC output v_b is [.,V]).
#   (b) three store-side additions: beta_act and sig (both == the resident M1
#       ub_bact; two extra pl.store, no new UB tile, no new mutex) and q_t = q_s^T
#       (the seed's existing transpose_vf gather, run in M1 into the reused [C,C]
#       ub_p2 scratch, then stored; no new UB tile, no new mutex).
#   (c) standalone wrapper / entry / shape inference and operator naming.
# NOTHING structural else: tile declarations, section ordering, mutex handoffs,
# matmul operand routing and every @vf body are the seed's, unchanged.
#
# The seed's FOUR load-bearing M4-repair fixes are PRESERVED verbatim (see
# NOTES.md -> "the four preserved fixes"):
#   Fix 1  FLAT (non-nested) mutex chain: vcmutex(M1->M2) is freed right after M2
#          consumes M1's operands, before cvmutex.lock(); the three handoffs never
#          overlap (double-nesting deadlocks the depth-1 pipeline).
#   Fix 2  Module 4 uses DEDICATED Cube buffers (l1_attn, l1_op, l0c_uw): reusing
#          M2's single-buffered cube objects across the M2->(vec M3)->M4
#          cube->vec->cube boundary deadlocks the pipeline.
#   Fix 3  l1_op declared at the EXACT [C,C] width (NOT [C,KMAX]): l1_op.T
#          transposes the full declared width, so an over-wide operand truncates
#          the u/w contraction to the first fp32 K0 fractal (u rows >= 8 garbage).
#   Fix 4  v_b = v * beta_act is built in MODULE 1 (scale_vtile_vf), not via an M3
#          GM round-trip: the ubout(MTE3)->ubin(MTE2) cross-tile WAR that a round-
#          trip introduces is not tracked by auto_sync and corrupts the first V
#          sub-block for V=128/96.
#
# Dataflow (DESIGN sec.0/sec.10): one mixed @kernel, a vec->cube->vec->cube
# pipeline with THREE ownership handoffs, FLAT/sequential (mha_ifa pattern):
#   [Vec  M1] preprocess -> GM (q_s/k_f32/k_beta/v_b/g_f32/ltones + beta_act/sig/q_t)
#   --VcMutex(0, vec->cube)-- freed after M2 --
#   [Cube M2] 3 fp32 matmuls -> decay/P1/P2 GM
#   --CvMutex(1, cube->vec)--
#   [Vec  M3] read decay/P1/P2 -> M3 GM outputs (incl attn, kbeta_de, kbeta_T)
#   --VcMutex(2, vec->cube)--
#   [Cube M4] load attn/v_b/kbeta_de/kbeta_T GM->L1 (nd2nz) -> u, w, w_T (sub-blocked)
# Authored against the easyasc a5 facade, validated in the OpExec simulator
# against gdr_fwd_train_wy_golden_cpu.py (all 17 outputs), then backend-
# transpiled to PyPTO-Pro DSL (merge_siblings=True).  No PyPTO / PyPTO-Pro API is
# hand-written.
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


# ---- operator-frozen constants (DESIGN sec.2.1) ----
C = 64             # chunk_size (SPEC frozen internal constant); [C,C] never sub-blocked
TK = 64            # K/V-direction UB tile (K=128 -> 2 kt loops); on-chip UB tile width
SUB = 64           # fp32 register lane count (one Reg = 64 fp32 lanes) == TK == TV
KMAX = 128         # compile-time L1 operand width bound (holds full K for the matmul)
HMAX = 32          # compile-time head-count bound for the per-token g/beta staging tile
WLOG2 = 6          # log2(TK) -- gather column stride for the transpose
SIM_CORE_COUNT = 32


# =====================================================================
# Vector functions (@vf) -- Module 1 preprocessing (VERBATIM from the seed)
# =====================================================================
@vf()
def prep_scalars_vf(ub_bh: Tensor, ub_gh: Tensor, ub_bact: Tensor, ub_gf: Tensor,
                    vc: Var, h_idx: Var):
    """Extract column h of the contiguous [vc,H] beta/g staging tiles per token r:
    beta_act[r] = sigmoid(beta_logit[r,h]); g_f32[r] = g[r,h]. Pad columns
    [vc:C] are set to 0 (so the cumsum's pad columns equal decay[vc-1])."""
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
    for r in range(vc, C, name="spr"):
        ub_bact[0:1, r:r + 1] <<= zero.single_value()
        ub_gf[0:1, r:r + 1] <<= zero.single_value()


@vf()
def scale_ktile_vf(ub_q: Tensor, ub_k: Tensor, ub_bact: Tensor,
                   ub_qs: Tensor, ub_kf: Tensor, ub_kbeta: Tensor,
                   vc: Var, scale: Var):
    """One [C,TK] sub-block: q_s = q*scale, k_f32 = k, k_beta = k*beta_act[r].
    Pad rows [vc:C] are zeroed so the full-C GM store keeps the padding at 0."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    to_float = CastConfig(round_mode=RoundMode.NONE, reg_layout=RegLayout.ZERO, name="k_b2f")
    z = Reg(DT.float)
    dup(z, 0.0, mask=active)
    for r in range(vc, name="kr"):
        bs = Reg(DT.float)
        bs <<= ub_bact[0:1, r:r + 1].single()
        ksrc = Reg(DT.bfloat16)
        kf = Reg(DT.float)
        kb = Reg(DT.float)
        ub_to_reg_unpack(ksrc, ub_k[r:r + 1, 0:SUB])
        cast(kf, ksrc, to_float, mask=active)
        reg_to_ub_normal(ub_kf[r:r + 1, 0:SUB], kf, mask=active)
        mul(kb, kf, bs, mask=active)
        reg_to_ub_normal(ub_kbeta[r:r + 1, 0:SUB], kb, mask=active)
        qsrc = Reg(DT.bfloat16)
        qf = Reg(DT.float)
        qs = Reg(DT.float)
        ub_to_reg_unpack(qsrc, ub_q[r:r + 1, 0:SUB])
        cast(qf, qsrc, to_float, mask=active)
        muls(qs, qf, scale, mask=active)
        reg_to_ub_normal(ub_qs[r:r + 1, 0:SUB], qs, mask=active)
    for r in range(vc, C, name="kpr"):
        reg_to_ub_normal(ub_qs[r:r + 1, 0:SUB], z, mask=active)
        reg_to_ub_normal(ub_kf[r:r + 1, 0:SUB], z, mask=active)
        reg_to_ub_normal(ub_kbeta[r:r + 1, 0:SUB], z, mask=active)


@vf()
def build_ltones_vf(ub_lt: Tensor):
    """Lower-triangular ones [C,C]: LTones[r,j] = 1.0 if j <= r else 0.0. Loaded
    as matmul l1b, so matmul computes g @ LTones^T = g @ upper-tri = prefix
    cumsum."""
    cols = Reg(DT.int)
    zero = Reg(DT.float)
    row = Reg(DT.float)
    m = MaskReg(DT.int, init_mode=MaskType.NONE)
    zero <<= 0.0
    cols.arange(0)
    for r in range(C, name="ltr"):
        row <<= 1.0
        compare(m, cols, r + 1, CompareMode.LT)   # cols < r+1  <=>  j <= r
        select(row, row, zero, mask=m)            # keep 1 for j<=r, else 0
        ub_lt[r:r + 1, 0:C] <<= row


@vf()
def cast_vtile_vf(ub_v: Tensor, ub_vf: Tensor, vc: Var):
    """Module-1 fp32 staging of v (dead code retained for structural fidelity with
    the seed; the backend prunes it -- v_b is now built by scale_vtile_vf)."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    to_float = CastConfig(round_mode=RoundMode.NONE, reg_layout=RegLayout.ZERO, name="v_b2f")
    for r in range(vc, name="vr"):
        vsrc = Reg(DT.bfloat16)
        vfp = Reg(DT.float)
        ub_to_reg_unpack(vsrc, ub_v[r:r + 1, 0:SUB])
        cast(vfp, vsrc, to_float, mask=active)
        reg_to_ub_normal(ub_vf[r:r + 1, 0:SUB], vfp, mask=active)


@vf()
def scale_vtile_vf(ub_v: Tensor, ub_bact: Tensor, ub_vb: Tensor, vc: Var):
    """v_b = (bf16->fp32) v * beta_act, one [C,TV] sub-block -- the EXACT mirror of
    scale_ktile_vf's k_beta = k * beta_act computation (a single V-pipe write into
    ub_vb per row).  Built in Module 1 (Fix 4): computing v_b via an M3 GM round-
    trip leaves the ubout(MTE3)->ubin(MTE2) cross-tile WAR untracked by auto_sync
    (corrupts the first V-tile for V=128/96).  Pad rows [vc:C] -> 0."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    to_float = CastConfig(round_mode=RoundMode.NONE, reg_layout=RegLayout.ZERO, name="vb_b2f")
    z = Reg(DT.float)
    dup(z, 0.0, mask=active)
    for r in range(vc, name="vbr"):
        bs = Reg(DT.float)
        bs <<= ub_bact[0:1, r:r + 1].single()
        vsrc = Reg(DT.bfloat16)
        vfp = Reg(DT.float)
        vb = Reg(DT.float)
        ub_to_reg_unpack(vsrc, ub_v[r:r + 1, 0:SUB])
        cast(vfp, vsrc, to_float, mask=active)
        mul(vb, vfp, bs, mask=active)
        reg_to_ub_normal(ub_vb[r:r + 1, 0:SUB], vb, mask=active)
    for r in range(vc, C, name="vbpr"):
        reg_to_ub_normal(ub_vb[r:r + 1, 0:SUB], z, mask=active)


# =====================================================================
# Vector functions (@vf) -- transpose (VERBATIM from the seed).  Used for
# kbeta_de -> kbeta_T (M3), w -> w_T (M4b), AND q_s -> q_t (M1, re-home).
# =====================================================================
@vf()
def transpose_vf(ub_src: Tensor, ub_dst: Tensor):
    """Column transpose [C,TK] -> [TK,C] via vf.gather (the A5 idiom): row k of
    dst = column k of src.  Gather index[j] = j*TK (element stride = the declared
    tile width); slicing src to column k keeps its base offset k, so the gather
    reads src[j, k].  Source pad rows must already be 0 so dst pad columns are 0.
    """
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    iota = Reg(DT.int)
    arange(iota, 0)
    idxi = Reg(DT.int)
    shiftls(idxi, iota, WLOG2)
    idx = idxi.reinterpret(DT.uint32)
    for k in range(SUB, name="kg"):
        colreg = Reg(DT.float)
        ub_to_reg_gather(colreg, ub_src[0:C, k:k + 1], idx, mask=active)
        reg_to_ub_normal(ub_dst[k:k + 1, 0:C], colreg, mask=active)


# =====================================================================
# Vector functions (@vf) -- Module 3 (wy_build); all fp32. (VERBATIM from the seed)
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
def lmask_vf(ub_dec: Tensor, ub_L: Tensor):
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
def attn_build_vf(ub_P1: Tensor, ub_L: Tensor, ub_attn: Tensor):
    """attn_pre[i,j] = -P1[i,j]*L_mask[i,j] for j<i else 0 (strict lower)."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    cols = Reg(DT.int)
    zero = Reg(DT.float)
    dup(zero, 0.0, mask=active)
    cols.arange(0)
    for i in range(C, name="ai"):
        p = Reg(DT.float)
        l = Reg(DT.float)
        p <<= ub_P1[i:i + 1, 0:C]
        l <<= ub_L[i:i + 1, 0:C]
        r = Reg(DT.float)
        mul(r, p, l, mask=active)
        neg(r, r, mask=active)
        m = MaskReg(DT.int, init_mode=MaskType.NONE)
        compare(m, cols, i, CompareMode.LT)      # cols < i  <=>  j < i (strict lower)
        select(r, r, zero, mask=m)
        ub_attn[i:i + 1, 0:C] <<= r


@vf()
def a_build_vf(ub_P2: Tensor, ub_L: Tensor, ub_A: Tensor):
    """A[i,j] = P2[i,j]*L_mask[i,j] for j<=i else 0 (lower incl diag)."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    cols = Reg(DT.int)
    zero = Reg(DT.float)
    dup(zero, 0.0, mask=active)
    cols.arange(0)
    for i in range(C, name="Ai"):
        p = Reg(DT.float)
        l = Reg(DT.float)
        p <<= ub_P2[i:i + 1, 0:C]
        l <<= ub_L[i:i + 1, 0:C]
        r = Reg(DT.float)
        mul(r, p, l, mask=active)
        m = MaskReg(DT.int, init_mode=MaskType.NONE)
        compare(m, cols, i + 1, CompareMode.LT)  # cols < i+1  <=>  j <= i
        select(r, r, zero, mask=m)
        ub_A[i:i + 1, 0:C] <<= r


@vf()
def fwdsubst_vf(ub_attn: Tensor):
    """s8 forward substitution + s9 identity, in place on the strict-lower attn."""
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
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    vf_barrier(VfPipe.STORE, VfPipe.STORE)
    for i in range(C, name="fid"):
        ub_attn[i:i + 1, i:i + 1] <<= one.single_value()   # +I(C) diagonal


@vf()
def rowscale_vf(ub_src: Tensor, ub_dexp: Tensor, ub_dst: Tensor):
    """dst[c,0:SUB] = src[c,0:SUB] * scalar[c] (q_de / kbeta_de), one [C,SUB]
    sub-block, full C rows.  scalar is decay_exp[c].  Pad rows scale by 0 -> 0."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    for c in range(C, name="rc"):
        sc = Reg(DT.float)
        sc <<= ub_dexp[0:1, c:c + 1].single()     # scalar[c] (broadcast)
        row = Reg(DT.float)
        row <<= ub_src[c:c + 1, 0:SUB]
        mul(row, row, sc, mask=active)
        ub_dst[c:c + 1, 0:SUB] <<= row


@vf()
def dfac_vf(ub_dec: Tensor, ub_dfac: Tensor):
    """dfac[c,0:SUB] = exp(decay[C-1] - decay[c]) broadcast over one [C,SUB]
    sub-block, full C rows."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    dlast = Reg(DT.float)
    dlast <<= ub_dec[0:1, (C - 1):C].single()     # scalar decay[C-1] (broadcast)
    for c in range(C, name="dc"):
        dc = Reg(DT.float)
        dc <<= ub_dec[0:1, c:c + 1].single()
        s = Reg(DT.float)
        expsub(s, dlast, dc, mask=active)         # exp(decay_last - decay_c)
        ub_dfac[c:c + 1, 0:SUB] <<= s


# =====================================================================
# Kernel -- mixed Vec(M1) -> Cube(M2) -> Vec(M3) -> Cube(M4a) -> Vec(M4b)
# =====================================================================
@kernel()
def gdr_fwd_train_wy_kernel(
    # --- primary inputs ---
    q: GMTensor, k: GMTensor, v: GMTensor, g: GMTensor, beta: GMTensor,
    # --- non-returned GM workspace (OpExec classifies these as inputs) ---
    k_beta_ws: GMTensor, ltones_ws: GMTensor, v_f32_ws: GMTensor,
    P1_ws: GMTensor, P2_ws: GMTensor, kbeta_de_ws: GMTensor,
    L_mask_ws: GMTensor, v_b_ws: GMTensor,
    # --- the 17 declared outputs, in SPEC output order (all returned) ---
    q_s_ws: GMTensor, k_f32_ws: GMTensor, v_b_out: GMTensor, g_f32_ws: GMTensor,
    beta_act_out: GMTensor, sig_out: GMTensor, q_t_out: GMTensor, decay_ws: GMTensor,
    decay_exp_ws: GMTensor, dfac_ws: GMTensor, q_de_ws: GMTensor, attn_ws: GMTensor,
    A_ws: GMTensor, kbeta_T_ws: GMTensor, u: GMTensor, w: GMTensor, w_T: GMTensor,
    # --- scalars ---
    B: Var, T: Var, H: Var, Kd: Var, Vd: Var, N: Var, Cc: Var, VT: Var, scale: Var,
):
    # M1(vec) -> M2(cube): vec ends on MTE3, cube ends on FIX.
    vcmutex = VcMutex(0, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    # M2(cube) -> M3(vec): cube ends on FIX (GM store), vec consumer on V.
    cvmutex = CvMutex(1, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    # M3(vec) -> M4(cube): vec ends on MTE3 (attn/kbeta_de/kbeta_T/v_b GM store), cube FIX.
    vcmutex2 = VcMutex(2, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)

    # ---- Vector-side UB scratch (Module 1 pool objects; compile-time literals) ----
    ub_q = Tensor(DT.bfloat16, [C, TK], Position.UB)     # P_in role (q / v load)
    ub_k = Tensor(DT.bfloat16, [C, TK], Position.UB)     # P_in role (k load)
    ub_bh = Tensor(DT.bfloat16, [C, HMAX], Position.UB)  # M1 beta staging
    ub_gh = Tensor(DT.float, [C, HMAX], Position.UB)     # M1 g staging
    ub_bact = Tensor(DT.float, [1, C], Position.UB)      # beta_act (persists into M3 -> v_b)
    ub_gf = Tensor(DT.float, [1, C], Position.UB)        # g_f32 (M1); M3 reuse -> decay_exp
    ub_qs = Tensor(DT.float, [C, TK], Position.UB)       # P_a role;  M3 reuse -> q_de / v_b
    ub_kf = Tensor(DT.float, [C, TK], Position.UB)       # P_b role;  M3 reuse -> dfac
    ub_kbeta = Tensor(DT.float, [C, TK], Position.UB)    # P_cc1 role; M3 reuse -> kbeta_de
    ub_lt = Tensor(DT.float, [C, C], Position.UB)        # LTones staging; M3 -> L_mask

    # ---- Vector-side UB scratch (Module 3) ----
    ub_p1 = Tensor(DT.float, [C, C], Position.UB)        # P1 -> attn (fwdsubst, dedicated)
    ub_p2 = Tensor(DT.float, [C, C], Position.UB)        # P2 -> A -> kbeta_T; M1 reuse -> q_t transpose dst
    ub_dec = Tensor(DT.float, [1, C], Position.UB)       # decay (frees ub_bact for beta_act)

    # ---- Cube-side operands (L1) and accumulators (L0C) ----
    l1_qs = Tensor(DT.float, [C, KMAX], Position.L1)     # M2 q_s
    l1_kf = Tensor(DT.float, [C, KMAX], Position.L1)
    l1_kbeta = Tensor(DT.float, [C, KMAX], Position.L1)
    l1_g = Tensor(DT.float, [16, C], Position.L1)        # m=1 matmul left operand
    l1_lt = Tensor(DT.float, [C, C], Position.L1)        # lower-tri ones (l1b)
    l0c_decay = Tensor(DT.float, [16, C], Position.L0C)
    l0c_p1 = Tensor(DT.float, [C, C], Position.L0C)      # M2 P1
    l0c_p2 = Tensor(DT.float, [C, C], Position.L0C)
    # ---- M4 DEDICATED Cube objects (Fix 2; correctness-critical) ----
    l1_attn = Tensor(DT.float, [C, C], Position.L1)      # M4 attn operand
    l1_op = Tensor(DT.float, [C, C], Position.L1)        # M4 v_b/kbeta_de/kbeta_T operand (Fix 3: exact [C,C])
    l0c_uw = Tensor(DT.float, [C, C], Position.L0C)      # M4 u/w/w_T accumulator

    k_tiles = CeilDiv(Kd, TK)
    v_tiles = CeilDiv(Vd, TK)
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

            # =====================================================
            # M1 (Vector, AIV sub-block 0) -> GM workspace + outputs
            # =====================================================
            vcmutex.lock()
            row_start = Var(GetSubBlockIdx() * C)
            if row_start < C:
                ub_bh[0:vc, 0:H] <<= beta[b_idx, t0:t0 + vc, 0:H]
                ub_gh[0:vc, 0:H] <<= g[b_idx, t0:t0 + vc, 0:H]
                prep_scalars_vf(ub_bh, ub_gh, ub_bact, ub_gf, vc, h_idx)
                g_f32_ws[b_idx, h_idx, n_idx, 0:C] <<= ub_gf[0:1, 0:C]
                # RE-HOME (b): beta_act and sig are both the resident ub_bact
                # (pad rows already 0).  Two extra pl.store; no new UB, no new mutex.
                beta_act_out[b_idx, h_idx, n_idx, 0:C] <<= ub_bact[0:1, 0:C]
                sig_out[b_idx, h_idx, n_idx, 0:C] <<= ub_bact[0:1, 0:C]

                for kt in range(k_tiles, name="kt"):
                    k0 = Var(kt * TK)
                    ub_q[0:vc, 0:SUB] <<= q[b_idx, t0:t0 + vc, h_idx, k0:k0 + TK]
                    ub_k[0:vc, 0:SUB] <<= k[b_idx, t0:t0 + vc, h_idx, k0:k0 + TK]
                    scale_ktile_vf(ub_q, ub_k, ub_bact, ub_qs, ub_kf, ub_kbeta, vc, scale)
                    q_s_ws[b_idx, h_idx, n_idx, 0:C, k0:k0 + TK] <<= ub_qs[0:C, 0:SUB]
                    k_f32_ws[b_idx, h_idx, n_idx, 0:C, k0:k0 + TK] <<= ub_kf[0:C, 0:SUB]
                    k_beta_ws[b_idx, h_idx, n_idx, 0:C, k0:k0 + TK] <<= ub_kbeta[0:C, 0:SUB]
                    # RE-HOME (b): q_t = q_s^T via the seed's transpose_vf gather,
                    # into the reused [C,C] ub_p2 (free in M1; M3 reloads P2 fresh).
                    # No new UB tile, no new mutex.  The ub_p2 store (MTE3) is fenced
                    # by vcmutex.ready() before M3 reloads P2_ws (mutex chain orders it).
                    transpose_vf(ub_qs, ub_p2)
                    q_t_out[b_idx, h_idx, n_idx, k0:k0 + SUB, 0:C] <<= ub_p2[0:SUB, 0:C]

                # v_b = (bf16->fp32) v * beta_act, built HERE in Module 1 (Fix 4).
                for vt in range(v_tiles, name="vt"):
                    v0 = Var(vt * TK)
                    vvw = Min(Vd - v0, TK)   # valid width for the partial last V-tile
                    ub_q[0:vc, 0:vvw] <<= v[b_idx, t0:t0 + vc, h_idx, v0:v0 + vvw]
                    scale_vtile_vf(ub_q, ub_bact, ub_qs, vc)
                    v_b_ws[b_idx, h_idx, n_idx, 0:C, v0:v0 + TK] <<= ub_qs[0:C, 0:SUB]
                    # RE-HOME (a): declared v_b output at valid V width (v_b_ws stays
                    # padded to VT for M4's fixed-64 read; SPEC v_b is [.,V]).
                    v_b_out[b_idx, h_idx, n_idx, 0:C, v0:v0 + vvw] <<= ub_qs[0:C, 0:vvw]

                build_ltones_vf(ub_lt)
                ltones_ws[0:C, 0:C] <<= ub_lt[0:C, 0:C]
            vcmutex.ready()

            # =====================================================
            # M2 (Cube, AIC): 3 fp32 matmuls -> decay/P1/P2 GM workspace
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
            # FLAT handoff chain (Fix 1): close H1 (vcmutex, M1->M2) HERE, right
            # after M2 has consumed M1's L1 operands -- do NOT hold it open across
            # cvmutex/vcmutex2 (double-nesting deadlocks the depth-1 pipeline).
            vcmutex.free()

            cvmutex.lock()
            decay_ws[b_idx, h_idx, n_idx, 0:C] <<= l0c_decay[0:1, 0:C]
            P1_ws[b_idx, h_idx, n_idx, 0:C, 0:C] <<= l0c_p1[0:C, 0:C]
            P2_ws[b_idx, h_idx, n_idx, 0:C, 0:C] <<= l0c_p2[0:C, 0:C]
            cvmutex.ready()

            # =====================================================
            # M3 (Vector, AIV sub-block 0): FLAT consume-then-produce
            # =====================================================
            cvmutex.wait()
            if row_start < C:
                ub_dec[0:1, 0:C] <<= decay_ws[b_idx, h_idx, n_idx, 0:C]
                ub_p1[0:C, 0:C] <<= P1_ws[b_idx, h_idx, n_idx, 0:C, 0:C]
                ub_p2[0:C, 0:C] <<= P2_ws[b_idx, h_idx, n_idx, 0:C, 0:C]
            cvmutex.free()

            vcmutex2.lock()
            if row_start < C:
                # decay_exp -> ub_gf
                decay_exp_vf(ub_dec, ub_gf)
                decay_exp_ws[b_idx, h_idx, n_idx, 0:C] <<= ub_gf[0:1, 0:C]

                # L_mask (reuse ub_lt)
                lmask_vf(ub_dec, ub_lt)
                L_mask_ws[b_idx, h_idx, n_idx, 0:C, 0:C] <<= ub_lt[0:C, 0:C]

                # attn = -tril_strict(P1*L) -> forward substitution -> +I
                attn_build_vf(ub_p1, ub_lt, ub_p1)
                fwdsubst_vf(ub_p1)
                attn_ws[b_idx, h_idx, n_idx, 0:C, 0:C] <<= ub_p1[0:C, 0:C]

                # A = tril(P2*L)  (j<=i, incl diag)
                a_build_vf(ub_p2, ub_lt, ub_p2)
                A_ws[b_idx, h_idx, n_idx, 0:C, 0:C] <<= ub_p2[0:C, 0:C]

                # q_de / kbeta_de / dfac + kbeta_T over K tiles
                for kt in range(k_tiles, name="kt3"):
                    k0 = Var(kt * TK)
                    ub_qs[0:C, 0:SUB] <<= q_s_ws[b_idx, h_idx, n_idx, 0:C, k0:k0 + TK]
                    rowscale_vf(ub_qs, ub_gf, ub_qs)          # * decay_exp
                    q_de_ws[b_idx, h_idx, n_idx, 0:C, k0:k0 + TK] <<= ub_qs[0:C, 0:SUB]

                    ub_kbeta[0:C, 0:SUB] <<= k_beta_ws[b_idx, h_idx, n_idx, 0:C, k0:k0 + TK]
                    rowscale_vf(ub_kbeta, ub_gf, ub_kbeta)    # * decay_exp
                    kbeta_de_ws[b_idx, h_idx, n_idx, 0:C, k0:k0 + TK] <<= ub_kbeta[0:C, 0:SUB]

                    # kbeta_T = kbeta_de^T (transpose gather); reuse ub_p2 (A tile).
                    transpose_vf(ub_kbeta, ub_p2)
                    kbeta_T_ws[b_idx, h_idx, n_idx, k0:k0 + TK, 0:C] <<= ub_p2[0:SUB, 0:C]

                    dfac_vf(ub_dec, ub_kf)
                    dfac_ws[b_idx, h_idx, n_idx, 0:C, k0:k0 + TK] <<= ub_kf[0:C, 0:SUB]
            vcmutex2.ready()

            # =====================================================
            # M4 (Cube, AIC): u = attn @ v_b, w = attn @ kbeta_de, w_T = w**T
            # =====================================================
            vcmutex2.wait()
            # u sub-blocks (v_b is zero-padded to VT; matmul at compile-time n=SUB)
            for vv in range(v_tiles, name="uv"):
                v0 = Var(vv * TK)
                vvalid = Min(Vd - v0, SUB)
                l1_attn[0:C, 0:C] <<= attn_ws[b_idx, h_idx, n_idx, 0:C, 0:C]
                l1_op[0:C, 0:SUB] <<= v_b_ws[b_idx, h_idx, n_idx, 0:C, v0:v0 + SUB]
                matmul(l0c_uw, l1_attn, l1_op.T, m=C, n=SUB, k=C)   # attn @ v_b sub-block
                u[b_idx, h_idx, n_idx, 0:C, v0:v0 + vvalid] <<= l0c_uw[0:C, 0:vvalid]
            # w sub-blocks (K is a multiple of SUB in-spec -> always full 64)
            for kk in range(k_tiles, name="wk"):
                k0 = Var(kk * TK)
                l1_attn[0:C, 0:C] <<= attn_ws[b_idx, h_idx, n_idx, 0:C, 0:C]
                l1_op[0:C, 0:SUB] <<= kbeta_de_ws[b_idx, h_idx, n_idx, 0:C, k0:k0 + SUB]
                matmul(l0c_uw, l1_attn, l1_op.T, m=C, n=SUB, k=C)   # attn @ kbeta_de sub-block
                w[b_idx, h_idx, n_idx, 0:C, k0:k0 + SUB] <<= l0c_uw[0:C, 0:SUB]
            # w_T sub-blocks: w_T[k,:] = kbeta_T[k,:] @ attn**T
            for kk in range(k_tiles, name="wtk"):
                k0 = Var(kk * TK)
                l1_attn[0:C, 0:C] <<= attn_ws[b_idx, h_idx, n_idx, 0:C, 0:C]
                l1_op[0:SUB, 0:C] <<= kbeta_T_ws[b_idx, h_idx, n_idx, k0:k0 + SUB, 0:C]
                matmul(l0c_uw, l1_op, l1_attn, m=SUB, n=C, k=C)     # kbeta_T @ attn**T
                w_T[b_idx, h_idx, n_idx, k0:k0 + SUB, 0:C] <<= l0c_uw[0:SUB, 0:C]
            vcmutex2.free()   # close H3 (M3->M4); vcmutex (H1) was already freed after M2

    return (q_s_ws, k_f32_ws, v_b_out, g_f32_ws, beta_act_out, sig_out, q_t_out,
            decay_ws, decay_exp_ws, dfac_ws, q_de_ws, attn_ws, A_ws, kbeta_T_ws,
            u, w, w_T)


# =====================================================================
# OpExec simulator self-check entry
# =====================================================================
# axis ids = position in the trailing scalar arg list (B,T,H,Kd,Vd,N,Cc,VT,scale):
#   0=B, 1=T, 2=H, 3=K, 4=V, 5=N, 6=C, 7=VT (the padded V tile width)
_SHAPE_BINDINGS = {
    0: [0, 1, 2, 3],          # q            [B,T,H,K]
    1: [0, 1, 2, 3],          # k            [B,T,H,K]
    2: [0, 1, 2, 4],          # v            [B,T,H,V]
    3: [0, 1, 2],             # g            [B,T,H]
    4: [0, 1, 2],             # beta         [B,T,H]
    # --- non-returned workspace ---
    5: [0, 2, 5, 6, 3],       # k_beta_ws    [B,H,N,C,K]
    6: [6, 6],                # ltones_ws    [C,C]
    7: [0, 2, 5, 6, 7],       # v_f32_ws     [B,H,N,C,VT]
    8: [0, 2, 5, 6, 6],       # P1_ws        [B,H,N,C,C]
    9: [0, 2, 5, 6, 6],       # P2_ws        [B,H,N,C,C]
    10: [0, 2, 5, 6, 3],      # kbeta_de_ws  [B,H,N,C,K]
    11: [0, 2, 5, 6, 6],      # L_mask_ws    [B,H,N,C,C]
    12: [0, 2, 5, 6, 7],      # v_b_ws       [B,H,N,C,VT]
    # --- 17 declared outputs (SPEC order) ---
    13: [0, 2, 5, 6, 3],      # q_s_ws       [B,H,N,C,K]
    14: [0, 2, 5, 6, 3],      # k_f32_ws     [B,H,N,C,K]
    15: [0, 2, 5, 6, 4],      # v_b_out      [B,H,N,C,V]
    16: [0, 2, 5, 6],         # g_f32_ws     [B,H,N,C]
    17: [0, 2, 5, 6],         # beta_act_out [B,H,N,C]
    18: [0, 2, 5, 6],         # sig_out      [B,H,N,C]
    19: [0, 2, 5, 3, 6],      # q_t_out      [B,H,N,K,C]
    20: [0, 2, 5, 6],         # decay_ws     [B,H,N,C]
    21: [0, 2, 5, 6],         # decay_exp_ws [B,H,N,C]
    22: [0, 2, 5, 6, 3],      # dfac_ws      [B,H,N,C,K]
    23: [0, 2, 5, 6, 3],      # q_de_ws      [B,H,N,C,K]
    24: [0, 2, 5, 6, 6],      # attn_ws      [B,H,N,C,C]
    25: [0, 2, 5, 6, 6],      # A_ws         [B,H,N,C,C]
    26: [0, 2, 5, 3, 6],      # kbeta_T_ws   [B,H,N,K,C]
    27: [0, 2, 5, 6, 4],      # u            [B,H,N,C,V]
    28: [0, 2, 5, 6, 3],      # w            [B,H,N,C,K]
    29: [0, 2, 5, 3, 6],      # w_T          [B,H,N,K,C]
}


def run_gdr_fwd_train_wy(q, k, v, g, beta, simulator=True, out_dir=""):
    """Standalone OP-A entry: q,k,v,g,beta -> 17 fp32 outputs (SPEC order).
    Host does only shape inference, workspace/output allocation, one dispatch."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    N = (T + C - 1) // C
    if H > HMAX:
        raise ValueError(f"this KAL pre-build stages g/beta with HMAX={HMAX}; got H={H}")
    if K > KMAX or V > KMAX:
        raise ValueError(f"this KAL pre-build bounds K/V at {KMAX}; got K={K}, V={V}")
    VT = ((V + SUB - 1) // SUB) * SUB   # v_b padded width (multiple of 64 >= V)
    device = q.device
    # GM workspace (wrapper-allocated scratch, discarded; not host compute)
    q_s_ws = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    k_f32_ws = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    k_beta_ws = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    g_f32_ws = torch.empty((B, H, N, C), dtype=torch.float32, device=device)
    ltones_ws = torch.empty((C, C), dtype=torch.float32, device=device)
    v_f32_ws = torch.zeros((B, H, N, C, VT), dtype=torch.float32, device=device)  # dead (seed parity)
    decay_ws = torch.empty((B, H, N, C), dtype=torch.float32, device=device)
    P1_ws = torch.empty((B, H, N, C, C), dtype=torch.float32, device=device)
    P2_ws = torch.empty((B, H, N, C, C), dtype=torch.float32, device=device)
    decay_exp_ws = torch.empty((B, H, N, C), dtype=torch.float32, device=device)
    dfac_ws = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    q_de_ws = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    kbeta_de_ws = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    L_mask_ws = torch.empty((B, H, N, C, C), dtype=torch.float32, device=device)
    attn_ws = torch.empty((B, H, N, C, C), dtype=torch.float32, device=device)
    A_ws = torch.empty((B, H, N, C, C), dtype=torch.float32, device=device)
    kbeta_T_ws = torch.empty((B, H, N, K, C), dtype=torch.float32, device=device)
    v_b_ws = torch.zeros((B, H, N, C, VT), dtype=torch.float32, device=device)   # padded, M4 read
    # Declared outputs
    beta_act_out = torch.empty((B, H, N, C), dtype=torch.float32, device=device)
    sig_out = torch.empty((B, H, N, C), dtype=torch.float32, device=device)
    q_t_out = torch.empty((B, H, N, K, C), dtype=torch.float32, device=device)
    v_b_out = torch.empty((B, H, N, C, V), dtype=torch.float32, device=device)
    u = torch.empty((B, H, N, C, V), dtype=torch.float32, device=device)
    w = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    w_T = torch.empty((B, H, N, K, C), dtype=torch.float32, device=device)

    if simulator:
        gdr_fwd_train_wy_kernel._simulator_config = SimulatorConfig(
            core_count=SIM_CORE_COUNT, execution_timeout_s=1800.0,
        )
    scale = 1.0 / (K ** 0.5)
    return OpExec(gdr_fwd_train_wy_kernel, out_dir=out_dir, simulator=simulator)(
        q, k, v, g, beta,
        k_beta_ws, ltones_ws, v_f32_ws, P1_ws, P2_ws, kbeta_de_ws, L_mask_ws, v_b_ws,
        q_s_ws, k_f32_ws, v_b_out, g_f32_ws, beta_act_out, sig_out, q_t_out, decay_ws,
        decay_exp_ws, dfac_ws, q_de_ws, attn_ws, A_ws, kbeta_T_ws, u, w, w_T,
        B, T, H, K, V, N, C, VT, scale,
        shape_bindings=_SHAPE_BINDINGS,
    )


def _load_golden():
    import importlib.util
    gp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "gdr_fwd_train_wy_golden_cpu.py")
    spec = importlib.util.spec_from_file_location("gdr_fwd_train_wy_golden_cpu", gp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mk_case(B, T, H, K, V, seed, g_lo=-16.0, g_hi=0.0):
    # mirrors gdr_fwd_train_wy_golden_cpu._mk_case exactly (same inputs to both).
    torch.manual_seed(seed)
    q = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    k = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    g = torch.empty((B, T, H), dtype=torch.float32).uniform_(g_lo, g_hi)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16)
    return q, k, v, g, beta


def _mixed_ok(got, exp, atol, rtol, l2_rtol):
    """Mixed criterion: elementwise allclose OR peak-scaled relative L2."""
    gf = got.float()
    ef = exp.float()
    if torch.allclose(gf, ef, rtol=rtol, atol=atol):
        return True, (gf - ef).abs().max().item(), 0.0
    rel_l2 = ((gf - ef).norm() / (ef.norm() + 1e-12)).item()
    return rel_l2 <= l2_rtol, (gf - ef).abs().max().item(), rel_l2


_OUT_NAMES = ["q_s", "k_f32", "v_b", "g_f32", "beta_act", "sig", "q_t", "decay",
              "decay_exp", "dfac", "q_de", "attn", "a_mat", "kbeta_t", "u", "w", "w_t"]


if __name__ == "__main__":
    golden = _load_golden()
    # module_interfaces.yaml composition_verification shapes / seeds (SPEC sec.8).
    shapes = [(1, 128, 4, 64, 64), (1, 100, 4, 64, 64),
              (1, 192, 2, 128, 128), (1, 200, 3, 128, 96)]
    seeds = [42, 123, 456]
    ATOL, RTOL, L2_RTOL = 1e-2, 1e-2, 2e-2
    all_pass = True
    for (B, T, H, K, V) in shapes:
        for seed in seeds:
            q, k, v, g, beta = _mk_case(B, T, H, K, V, seed)
            ref = golden.gdr_fwd_train_wy_golden_cpu(q, k, v, g, beta)
            out = run_gdr_fwd_train_wy(q, k, v, g, beta, simulator=True)
            outs = out if isinstance(out, tuple) else (out,)
            worst = 0.0
            worst_l2 = 0.0
            case_ok = True
            for nm, got, exp in zip(_OUT_NAMES, outs, ref):
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
