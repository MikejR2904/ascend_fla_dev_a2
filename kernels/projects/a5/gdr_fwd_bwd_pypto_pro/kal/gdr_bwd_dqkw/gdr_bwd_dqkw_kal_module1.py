# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# -----------------------------------------------------------------------------
# PyPTO-Pro Stage 3.5 (KAL pre-build) -- gdr_bwd_dqkw  (OP5 of gated_delta_rule_bwd;
# parent module M8, golden_step b3 -- the chunk-parallel dqkwg backward, the heaviest
# operator of the decomposition).  Single fusion module (module_count=1, is_fusion=true;
# DESIGN sec.0): 6 matmuls + ds triangular mask + dg reduce, BHN-parallel, no carried
# state.  Authored against the easyasc a5 facade; validated in the OpExec simulator vs
# gdr_bwd_dqkw_golden_cpu.py; backend-transpiled to PyPTO-Pro DSL (kal/staged_module1.py).
# No PyPTO / PyPTO-Pro API is hand-written.  Simulator pass != Stage-4 acceptance.
#
# =========================== A5 IDIOM (SINGLE) ==============================
# Follows the sim-verified + transpile-clean SINGLE idiom of the sibling scan OP4
# custom/gdr_bwd_reverse_scan/kal/gdr_bwd_reverse_scan_kal_module1.py:
#   * cube<->vec on-chip moves use `.subblk(0)` UNGUARDED on the cube side; the vec
#     compute/load/store is guarded by `if row_start < C` (row_start=GetSubBlockIdx()*C).
#     This is the A5-correct SINGLE idiom (the OP3 all-NaN was `.subblk(0)` placed
#     INSIDE a vec guard -- not done here).
#   * GM-routed operand bridging (Fix A): vec stores UB->GM (MTE3), cube loads GM->L1
#     (nd2nz); never UB->L1 (which would materialise NZ staging in UB and overflow).
#   * auto_mutex ONLY (no hand SEvents); 4 cube<->vec handoffs per chunk.
#
# matmul(dst, l1a, l1b) == l1a @ l1b**T  (l1a=[M,K] LEFT/L0A, l1b=[N,K] RIGHT/L0B;
# .T on l1b appends a cheap L1->L0B NZ2ZN transpose).  Derived from the payload docstring
# easyasc/shortcuts/matmul.py.  The six b3 matmuls (per DESIGN sec.1 / golden_cpu):
#   cube-1 (once):  ds_raw   = grad_o @ v_new**T   -> matmul(do,   v_new)   [do@v_new^T]
#                   dsT_raw  = v_new  @ grad_o**T   -> matmul(v_new,do)      [ds^T, no L0A transpose]
#   cube-2 (per output-K subblock kk):
#      dq_inter = grad_o @ h      -> matmul(do,   h[k],  k=Vd)     h=s_hist [K,V]
#      dq_intra = ds     @ k_f32  -> matmul(ds,   k[:,k].T, k=C)
#      dk_inter = v_new  @ dh     -> matmul(v_new,dh[k], k=Vd)
#      dk_intra = ds^T   @ q_raw  -> matmul(dsT,  q_s[:,k].T, k=C) then *sqrt(K) in vec
#      dw       = dv2    @ h      -> matmul(dv2,  h[k],  k=Vd)   (negated in vec)
# ds_T is produced by a SECOND cube-1 matmul (v_new@do^T) rather than a vec transpose:
# no A5/pypto vec-transpose primitive exists (gm_to_ub_nd_dma_transpose is not pypto-
# supported) and this keeps src_is_transpose into L0A == 0.  ds_T is then masked in vec
# with the transposed factor exp(decay_j-decay_i), upper-incl-diag -- mathematically
# identical to transpose(mask(ds)).  See NOTES.md.
#
# q_raw ELIMINATED (Stage-2 requires RAW q = q_s*sqrt(K); q_s = q*scale upstream).  Since the
# matmul and the reduce are linear in q, sqrt(K) is pulled out as a scalar: dk_intra =
# sqrt(K)*(ds_T@q_s), and sum_K(dq_p*q_raw) = sqrt(K)*sum_K(dq_p*q_s).  So q_s (an input) is used
# directly -- no qraw GM bridge, no per-subblock build loop.  k_f32 is raw; dq_p/dw do not use q.
#
# Output-K (K<=128) is STATIC-unrolled into 2 subblocks (k0 in {0,64}); the 5 L0C accumulators,
# the 5 UB landings, and the 2 L1 h/dh operands are DBuff'd so the two subblock iterations do not
# clobber each other (depth=1 mutex alone is insufficient -- see NOTES findings).
#
# Per-chunk dataflow (BHN-parallel, grid=B*H*N, no carried state):
#   [vc_cast Vc]  vec: grad_o bf16->fp32 -> do_gm ; cube: ds operands (do_gm load AFTER wait)
#   [cube-1]      ds_raw=do@v_new^T ; dsT_raw=v_new@do^T
#   [cv_ds  Cv]   vec: ds mask/exp/scale ; ds_T mask ; decay_exp ; store ds_gm/dsT_gm (same region)
#   [vc_dq  Vc]   vec: reset dg accumulators ; cube: load cube-2 operands (q_s direct)
#   [cube-2 x k0] dq_inter/intra, dk_inter/intra, dw   (5 DBuff L0C, output-K subblocked TK=64)
#   [cv_out Cv]   vec: dq=dqi*dexp*scale+dqa ; dk_inter=dki*gate ; dk_intra=sqrt(K)*(ds_T@q_s) ;
#                      dk=dk_inter+dk_intra ; dw=-dw ; dg_p += sqrt(K)*sum(dq*q_s)-sum(dk*k)
#   [after k0]    dg_last = exp(g_last)*sum_KV(s_hist*dh) + sum_CK(dk_inter*k) ; dg_p[C-1] += dg_last
# -----------------------------------------------------------------------------
"""gdr_bwd_dqkw -- M8 dqkwg backward (b3): dq_p, dk_p, dw, dg_p."""
import os
import sys

for _cand in (
    "/root/.claude/skills/pypto-kernel-arch-lang/payload",
    "/workspace/pypto-gym/cannbot-skills/ops/pypto-kernel-arch-lang/payload",
):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

from easyasc.a5 import *  # noqa

# ---- operator key constants (DESIGN sec.2.1; used verbatim) ----
C = 64        # fixed chunk size (internal constant)
TK = 64       # output-K subblock (K <= 128 -> ceil(K/TK) subblocks)
TV = 64       # V subblock for the cast / dg_last reduction (V <= 128)
KMAX = 128    # K upper bound (physical tile width)
VMAX = 128    # V upper bound (physical tile width)
SUB = 64      # fp32 VF register lane width (bf16->fp32 cast chunk)


# =====================================================================
# Vector functions (all fp32; SINGLE / sub-block-0 executes the full tile).
# =====================================================================
@vf()
def consts_vf(ub_const: Tensor, Kd: Var):
    """ub_const[0,0]=sqrt(Kd) ; ub_const[0,1]=scale=1/sqrt(Kd)  (runtime, K dynamic)."""
    ki = Reg(DT.int)
    dup(ki, Kd)
    kf = Reg(DT.float)
    cast(kf, ki, CastConfig(round_mode=RoundMode.NONE, name="k_i2f"))
    sk = Reg(DT.float)
    sqrt(sk, kf)
    one = Reg(DT.float)
    dup(one, 1.0)
    sc = Reg(DT.float)
    div(sc, one, sk)                                  # scale = 1/sqrt(K)
    ub_const[0:1, 0:1] <<= sk.single_value()
    ub_const[0:1, 1:2] <<= sc.single_value()


@vf()
def cast_go_vf(ub_bf: Tensor, ub_f: Tensor, valid_c: Var, valid_v: Var):
    """grad_o band bf16->fp32 for valid token rows [0:valid_c]; pad rows [valid_c:C]
    zero-filled so ds / dq_inter contributions from padding tokens are exactly 0
    (matches golden's F.pad(do,0)).  bf16->fp32 via the OP3 unpack+cast idiom
    (valid_v <= TV = SUB, single lane chunk)."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    to_f = CastConfig(round_mode=RoundMode.NONE, reg_layout=RegLayout.ZERO, name="go_b2f")
    z = Reg(DT.float)
    dup(z, 0.0, mask=active)
    for r in range(valid_c, name="cgr"):
        src = Reg(DT.bfloat16)
        dst = Reg(DT.float)
        ub_to_reg_unpack(src, ub_bf[r:r + 1, 0:valid_v])
        cast(dst, src, to_f, mask=active)
        reg_to_ub_normal(ub_f[r:r + 1, 0:valid_v], dst, mask=active)
    for r in range(valid_c, C, name="cgpr"):
        reg_to_ub_normal(ub_f[r:r + 1, 0:valid_v], z, mask=active)


@vf()
def dexp_vf(ub_decay: Tensor, ub_dexp: Tensor):
    """decay_exp[c] = exp(decay[c]) over the full C row (decay <= 0 -> in (0,1])."""
    d = Reg(DT.float)
    d <<= ub_decay[0:1, 0:C]
    d <<= d.exp()
    ub_dexp[0:1, 0:C] <<= d


@vf()
def mask_lower_vf(ub_decay: Tensor, ub_ds: Tensor, ub_const: Tensor):
    """ds[i,j] *= exp(decay_i - decay_j) * scale, keep lower incl diag (j<=i) else 0.
    exp(decay_i-decay_j) for j<=i is <= 1 (decay non-increasing); the discarded strict
    upper (j>i) is zeroed BEFORE the multiply so exp(+large) is never formed (no inf*0)."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    cols = Reg(DT.int)
    zero = Reg(DT.float)
    dup(zero, 0.0, mask=active)
    cols.arange(0)
    dvec = Reg(DT.float)
    dvec <<= ub_decay[0:1, 0:C]                        # decay_j vector [1,C]
    sc = Reg(DT.float)
    sc <<= ub_const[0:1, 1:2].single()                # scale scalar
    for r in range(C, name="mli"):
        di = Reg(DT.float)
        di <<= ub_decay[0:1, r:r + 1].single()        # decay_i scalar
        gexp = Reg(DT.float)
        expsub(gexp, di, dvec, mask=active)           # exp(decay_i - decay_j)
        m = MaskReg(DT.int, init_mode=MaskType.NONE)
        compare(m, cols, r, CompareMode.LE)           # cols <= i  (keep lower incl diag)
        select(gexp, gexp, zero, mask=m)
        gexp <<= gexp * sc                            # * scale
        dsrow = Reg(DT.float)
        dsrow <<= ub_ds[r:r + 1, 0:C]
        dsrow <<= dsrow * gexp
        ub_ds[r:r + 1, 0:C] <<= dsrow


@vf()
def mask_upper_vf(ub_decay: Tensor, ub_dsT: Tensor, ub_const: Tensor):
    """ds_T[i,j] *= exp(decay_j - decay_i) * scale, keep upper incl diag (j>=i) else 0.
    ds_T raw = v_new@do^T = transpose(do@v_new^T); masking with the transposed factor
    and the upper triangle yields transpose(mask(ds)) exactly.  exp(decay_j-decay_i) for
    j>=i is <= 1.  Built with the proven negated-operand expsub (scalar src0, vector src1)."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    cols = Reg(DT.int)
    zero = Reg(DT.float)
    dup(zero, 0.0, mask=active)
    cols.arange(0)
    dvec = Reg(DT.float)
    dvec <<= ub_decay[0:1, 0:C]
    ndvec = Reg(DT.float)
    neg(ndvec, dvec)                                  # -decay_j (vector, src1)
    sc = Reg(DT.float)
    sc <<= ub_const[0:1, 1:2].single()
    for r in range(C, name="mui"):
        di = Reg(DT.float)
        di <<= ub_decay[0:1, r:r + 1].single()
        ndi = Reg(DT.float)
        neg(ndi, di)                                  # -decay_i (scalar, src0)
        gexp = Reg(DT.float)
        expsub(gexp, ndi, ndvec, mask=active)         # exp(-decay_i-(-decay_j)) = exp(decay_j-decay_i)
        m = MaskReg(DT.int, init_mode=MaskType.NONE)
        compare(m, cols, r, CompareMode.GE)           # cols >= i  (keep upper incl diag)
        select(gexp, gexp, zero, mask=m)
        gexp <<= gexp * sc
        dsrow = Reg(DT.float)
        dsrow <<= ub_dsT[r:r + 1, 0:C]
        dsrow <<= dsrow * gexp
        ub_dsT[r:r + 1, 0:C] <<= dsrow


@vf()
def zero_row_vf(ub: Tensor, n: Var):
    """Zero ub[0:1, 0:n] (dg_p accumulator init)."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    z = Reg(DT.float)
    dup(z, 0.0, mask=active)
    reg_to_ub_normal(ub[0:1, 0:n], z, mask=active)


@vf()
def combine_vf(ub_dqi: Tensor, ub_dqa: Tensor, ub_dki: Tensor, ub_dka: Tensor,
               ub_dw: Tensor, ub_qraw: Tensor, ub_kf: Tensor, ub_dexp: Tensor,
               ub_decay: Tensor, ub_const: Tensor, ub_dgp: Tensor, ub_dgl: Tensor,
               valid_k: Var):
    """Assemble one output-K subblock and accumulate the dg terms (all C rows):
      dq = dqi*decay_exp[c]*scale + dqa            -> ub_dqa (= dq_p)
      dk_inter = dki*exp(g_last-decay[c])          -> ub_dki (kept for dg_last)
      dk = dk_inter + dka                          -> ub_dka (= dk_p)
      dw = -dw                                     -> ub_dw
      dg_p[c] += sum_K(dq*q_raw) - sum_K(dk*k)     (accumulate over kk)
      dgl     += sum_CK(dk_inter*k)                (accumulate over kk; dg_last term 2)
    Invalid K lanes [valid_k:TK] are masked out of the reductions."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    cols = Reg(DT.int)
    cols.arange(0)
    kmask = MaskReg(DT.int, init_mode=MaskType.NONE)
    compare(kmask, cols, valid_k, CompareMode.LT)     # lanes < valid_k
    zero = Reg(DT.float)
    dup(zero, 0.0, mask=active)
    sc = Reg(DT.float)
    sc <<= ub_const[0:1, 1:2].single()                # scale = 1/sqrt(K)
    sk = Reg(DT.float)
    sk <<= ub_const[0:1, 0:1].single()                # sqrt(K)  (q_raw = q_s*sqrt(K))
    glast = Reg(DT.float)
    glast <<= ub_decay[0:1, (C - 1):C].single()       # g_last = decay[C-1]
    for c in range(C, name="cbc"):
        de = Reg(DT.float)
        de <<= ub_dexp[0:1, c:c + 1].single()         # decay_exp[c]
        de <<= de * sc                                # decay_exp[c]*scale
        dc = Reg(DT.float)
        dc <<= ub_decay[0:1, c:c + 1].single()
        gate = Reg(DT.float)
        expsub(gate, glast, dc)                       # exp(g_last-decay[c])
        dqi = Reg(DT.float); dqi <<= ub_dqi[c:c + 1, 0:valid_k]
        dqa = Reg(DT.float); dqa <<= ub_dqa[c:c + 1, 0:valid_k]
        dki = Reg(DT.float); dki <<= ub_dki[c:c + 1, 0:valid_k]
        dka = Reg(DT.float); dka <<= ub_dka[c:c + 1, 0:valid_k]  # ds_T @ q_s  (needs *sqrt(K))
        dwr = Reg(DT.float); dwr <<= ub_dw[c:c + 1, 0:valid_k]
        qrr = Reg(DT.float); qrr <<= ub_qraw[c:c + 1, 0:valid_k]  # q_s (q_raw = q_s*sqrt(K))
        kfr = Reg(DT.float); kfr <<= ub_kf[c:c + 1, 0:valid_k]
        # dq_p = dqi*decay_exp*scale + dqa
        dq = Reg(DT.float)
        dq <<= dqi * de
        dq <<= dq + dqa
        ub_dqa[c:c + 1, 0:valid_k] <<= dq
        # dk_inter = dki*gate ; dk_intra = sqrt(K)*(ds_T@q_s) ; dk_p = dk_inter + dk_intra
        dki <<= dki * gate
        ub_dki[c:c + 1, 0:valid_k] <<= dki            # keep gated dk_inter for dg_last
        dka <<= dka * sk                              # dk_intra = sqrt(K)*(ds_T@q_s)
        dk = Reg(DT.float)
        dk <<= dki + dka
        ub_dka[c:c + 1, 0:valid_k] <<= dk
        # dw = -dw
        ndw = Reg(DT.float)
        neg(ndw, dwr)
        ub_dw[c:c + 1, 0:valid_k] <<= ndw
        # dg_p[c] += sum_K(dq*q_raw) - sum_K(dk*k)  (q_raw=q_s*sqrt(K); mask invalid K lanes)
        p1v = Reg(DT.float); p1v <<= dq * qrr
        select(p1v, p1v, zero, mask=kmask)
        s1 = Reg(DT.float); s1 <<= p1v.cadd()
        s1 <<= s1 * sk                                # sum_K(dq*q_s)*sqrt(K) = sum_K(dq*q_raw)
        p2v = Reg(DT.float); p2v <<= dk * kfr
        select(p2v, p2v, zero, mask=kmask)
        s2 = Reg(DT.float); s2 <<= p2v.cadd()
        prev = Reg(DT.float); prev <<= ub_dgp[0:1, c:c + 1].single()
        acc = Reg(DT.float)
        add(acc, prev, s1)
        sub(acc, acc, s2)
        ub_dgp[0:1, c:c + 1] <<= acc.single_value()
        # dgl += sum_K(dk_inter*k)
        p3v = Reg(DT.float); p3v <<= dki * kfr
        select(p3v, p3v, zero, mask=kmask)
        s3 = Reg(DT.float); s3 <<= p3v.cadd()
        prevl = Reg(DT.float); prevl <<= ub_dgl[0:1, 0:1].single()
        accl = Reg(DT.float)
        add(accl, prevl, s3)
        ub_dgl[0:1, 0:1] <<= accl.single_value()


@vf()
def shdh_vf(ub_sh: Tensor, ub_dh: Tensor, ub_dgkv: Tensor, valid_kk: Var, valid_v: Var):
    """dgkv += sum_{r<valid_kk, v<valid_v} s_hist[r,v]*dh[r,v]  (dg_last term 1 partial)."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    cols = Reg(DT.int)
    cols.arange(0)
    vmask = MaskReg(DT.int, init_mode=MaskType.NONE)
    compare(vmask, cols, valid_v, CompareMode.LT)
    zero = Reg(DT.float)
    dup(zero, 0.0, mask=active)
    for r in range(valid_kk, name="shr"):
        sh = Reg(DT.float); sh <<= ub_sh[r:r + 1, 0:valid_v]
        dhr = Reg(DT.float); dhr <<= ub_dh[r:r + 1, 0:valid_v]
        pv = Reg(DT.float); pv <<= sh * dhr
        select(pv, pv, zero, mask=vmask)
        s = Reg(DT.float); s <<= pv.cadd()
        prev = Reg(DT.float); prev <<= ub_dgkv[0:1, 0:1].single()
        acc = Reg(DT.float)
        add(acc, prev, s)
        ub_dgkv[0:1, 0:1] <<= acc.single_value()


@vf()
def dglast_fold_vf(ub_dgp: Tensor, ub_dgl: Tensor, ub_dgkv: Tensor, ub_decay: Tensor):
    """dg_last = exp(g_last)*dgkv + dgl ; dg_p[C-1] += dg_last  (fold onto last token)."""
    glast = Reg(DT.float)
    glast <<= ub_decay[0:1, (C - 1):C].single()
    eg = Reg(DT.float)
    eg <<= glast.exp()
    kv = Reg(DT.float); kv <<= ub_dgkv[0:1, 0:1].single()
    dgl = Reg(DT.float); dgl <<= ub_dgl[0:1, 0:1].single()
    dglast = Reg(DT.float)
    mul(dglast, eg, kv)
    add(dglast, dglast, dgl)
    prev = Reg(DT.float); prev <<= ub_dgp[0:1, (C - 1):C].single()
    add(prev, prev, dglast)
    ub_dgp[0:1, (C - 1):C] <<= prev.single_value()


@kernel()
def gdr_bwd_dqkw(
    grad_o: GMTensor, v_new: GMTensor, k_f32: GMTensor, q_s: GMTensor,
    s_hist: GMTensor, dh: GMTensor, dv2: GMTensor, g_f32: GMTensor, decay: GMTensor,
    do_gm: GMTensor, ds_gm: GMTensor, dsT_gm: GMTensor,
    dq_p: GMTensor, dk_p: GMTensor, dw: GMTensor, dg_p: GMTensor,
    B: Var, T: Var, H: Var, Kd: Var, Vd: Var, N: Var, Cc: Var, VT: Var,
):
    # 4 cube<->vec handoffs (Fix A GM-routed; producer MTE3/consumer FIX on vec->cube,
    # FIX/V on cube->vec).
    vc_cast = VcMutex(0, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)  # vec do_gm ; cube ds ops
    cv_ds = CvMutex(1, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)       # cube ds/dsT ; vec mask
    vc_dq = VcMutex(2, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)    # vec ds/dsT/qraw ; cube-2
    cv_out = CvMutex(3, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)      # cube-2 ; vec reduce

    # ---- Vec (UB) tiles ----
    ub_do_bf = Tensor(DT.bfloat16, [C, TV], Position.UB)   # grad_o bf16 band (cast source)
    ub_do_f = Tensor(DT.float, [C, TV], Position.UB)       # grad_o fp32 band ; reused as ub_sh
    ub_ds = Tensor(DT.float, [C, C], Position.UB)          # ds landing / mask
    ub_dsT = Tensor(DT.float, [C, C], Position.UB)         # ds_T landing / mask
    ub_decay = Tensor(DT.float, [1, C], Position.UB)       # decay load
    ub_dexp = Tensor(DT.float, [1, C], Position.UB)        # decay_exp = exp(decay)
    ub_const = Tensor(DT.float, [1, 8], Position.UB)       # [sqrt(K), scale]
    ub_dqi = DBuff(DT.float, [C, TK], Position.UB)        # dq_inter landing
    ub_dqa = DBuff(DT.float, [C, TK], Position.UB)        # dq_intra landing -> dq_p
    ub_dki = DBuff(DT.float, [C, TK], Position.UB)        # dk_inter landing -> gated dk_inter
    ub_dka = DBuff(DT.float, [C, TK], Position.UB)        # dk_intra landing -> dk_p
    ub_dw = DBuff(DT.float, [C, TK], Position.UB)         # dw landing
    ub_kf = Tensor(DT.float, [C, TK], Position.UB)         # k_f32 subblock (dg reduce + dg_last)
    ub_dgp = Tensor(DT.float, [1, C], Position.UB)         # dg_p accumulator
    ub_dgl = Tensor(DT.float, [1, 8], Position.UB)         # sum_CK(dk_inter*k) accumulator
    ub_dgkv = Tensor(DT.float, [1, 8], Position.UB)        # sum_KV(s_hist*dh) accumulator

    # ---- cube-1 (ds_raw / dsT_raw) operands + accumulators (own objects) ----
    l1_do1 = Tensor(DT.float, [C, VMAX], Position.L1)      # grad_o [C,V] (LEFT / RIGHT)
    l1_vn = Tensor(DT.float, [C, VMAX], Position.L1)       # v_new [C,V] (RIGHT / LEFT)
    l0c_ds = Tensor(DT.float, [C, C], Position.L0C)        # ds_raw = do@v_new^T
    l0c_dsT = Tensor(DT.float, [C, C], Position.L0C)       # dsT_raw = v_new@do^T

    # ---- cube-2 operands (DEDICATED -- M4 finding: no reuse across the vec boundary) ----
    l1_do2 = Tensor(DT.float, [C, VMAX], Position.L1)      # grad_o [C,V]
    l1_h = DBuff(DT.float, [C, VMAX], Position.L1)         # s_hist rows [valid_k,V] (per subblk, DBuff)
    l1_k = Tensor(DT.float, [C, KMAX], Position.L1)        # k_f32 [C,K]
    l1_vn2 = Tensor(DT.float, [C, VMAX], Position.L1)      # v_new [C,V]
    l1_dh = DBuff(DT.float, [C, VMAX], Position.L1)        # dh rows [valid_k,V] (per subblk, DBuff)
    l1_dv2 = Tensor(DT.float, [C, VMAX], Position.L1)      # dv2 [C,V]
    l1_dsL = Tensor(DT.float, [C, C], Position.L1)         # ds (from ds_gm)
    l1_dsTL = Tensor(DT.float, [C, C], Position.L1)        # ds_T (from dsT_gm)
    l1_qraw = Tensor(DT.float, [C, KMAX], Position.L1)     # q_raw (from qraw_gm)
    # DBuff (2 slots) so the two output-K subblock iterations use distinct L0C banks:
    # iter1's matmul must not clobber iter0's accumulator before iter0's move reads it
    # (auto_sync does not insert that WAR across the loop/mutex boundary).
    l0c_dqi = DBuff(DT.float, [C, TK], Position.L0C)       # dq_inter (per output-K subblock)
    l0c_dqa = DBuff(DT.float, [C, TK], Position.L0C)       # dq_intra
    l0c_dki = DBuff(DT.float, [C, TK], Position.L0C)       # dk_inter
    l0c_dka = DBuff(DT.float, [C, TK], Position.L0C)       # dk_intra
    l0c_dw = DBuff(DT.float, [C, TK], Position.L0C)        # dw

    tasks = B * H * N
    per = CeilDiv(tasks, GetCubeNum())
    begin = Var(per * GetCubeIdx())
    end = Min(begin + per, tasks)
    v_tiles = CeilDiv(Vd, TV)
    k_tiles = CeilDiv(Kd, TK)

    with auto_sync():
        for tid in range(begin, end, name="tid"):
            b_idx = Var(tid // (H * N))
            rem = Var(tid % (H * N))
            h_idx = Var(rem // N)
            n_idx = Var(rem % N)
            t0 = Var(n_idx * C)
            valid_c = Min(T - t0, C)
            row_start = Var(GetSubBlockIdx() * C)

            # ---- vc_cast (Vc): vec grad_o bf16->fp32 -> do_gm ; cube ds operands ----
            vc_cast.lock()
            if row_start < C:
                for vb in range(v_tiles, name="vb0"):
                    v0 = Var(vb * TV)
                    valid_v = Min(Vd - v0, TV)
                    if valid_c > 0:
                        ub_do_bf[0:valid_c, 0:valid_v] <<= grad_o[b_idx, t0:t0 + valid_c, h_idx, v0:v0 + valid_v]
                    cast_go_vf(ub_do_bf, ub_do_f, valid_c, valid_v)
                    do_gm[b_idx, h_idx, n_idx, 0:C, v0:v0 + valid_v] <<= ub_do_f[0:C, 0:valid_v]
            vc_cast.ready()

            l1_vn[0:C, 0:Vd] <<= v_new[b_idx, h_idx, n_idx, 0:C, 0:Vd]   # independent input
            vc_cast.wait()
            l1_do1[0:C, 0:Vd] <<= do_gm[b_idx, h_idx, n_idx, 0:C, 0:Vd]  # bridge: AFTER wait
            # ds_raw = do @ v_new^T ; dsT_raw = v_new @ do^T  (both k=Vd, m=n=C)
            matmul(l0c_ds, l1_do1[0:C, 0:Vd], l1_vn[0:C, 0:Vd], m=C, n=C, k=Vd)
            matmul(l0c_dsT, l1_vn[0:C, 0:Vd], l1_do1[0:C, 0:Vd], m=C, n=C, k=Vd)
            vc_cast.free()

            # ---- cv_ds (Cv): cube ds/dsT -> UB ; vec mask/exp/scale + decay_exp + q_raw ----
            cv_ds.lock()
            ub_ds[0:C, 0:C] <<= l0c_ds.subblk(0)
            ub_dsT[0:C, 0:C] <<= l0c_dsT.subblk(0)
            cv_ds.ready()
            if row_start < C:
                ub_decay[0:1, 0:C] <<= decay[b_idx, h_idx, n_idx, 0:C]
            cv_ds.wait()
            if row_start < C:
                consts_vf(ub_const, Kd)
                dexp_vf(ub_decay, ub_dexp)
                mask_lower_vf(ub_decay, ub_ds, ub_const)
                mask_upper_vf(ub_decay, ub_dsT, ub_const)
                # store the MASKED ds/ds_T in the same region as the mask so auto_sync
                # orders the V-pipe mask writes before the MTE3 store (a cross-region
                # store reads the raw cube-moved tile -- the vec writes never land).
                ds_gm[b_idx, h_idx, n_idx, 0:C, 0:C] <<= ub_ds[0:C, 0:C]
                dsT_gm[b_idx, h_idx, n_idx, 0:C, 0:C] <<= ub_dsT[0:C, 0:C]
            cv_ds.free()

            # ---- vc_dq (Vc): vec dg accumulators reset ; cube-2 load operands ----
            # q_raw eliminated: dk_intra = ds_T@q_s then *sqrt(K) in vec; the dg reduce
            # uses q_s then *sqrt(K).  So q_s (an input) is loaded directly -- no qraw
            # GM bridge, no per-subblock build loop (which clobbered a shared UB tile).
            vc_dq.lock()
            if row_start < C:
                zero_row_vf(ub_dgp, Cc)
                zero_row_vf(ub_dgl, 1)
                zero_row_vf(ub_dgkv, 1)
            vc_dq.ready()

            l1_do2[0:C, 0:Vd] <<= do_gm[b_idx, h_idx, n_idx, 0:C, 0:Vd]
            l1_k[0:C, 0:Kd] <<= k_f32[b_idx, h_idx, n_idx, 0:C, 0:Kd]
            l1_vn2[0:C, 0:Vd] <<= v_new[b_idx, h_idx, n_idx, 0:C, 0:Vd]
            l1_dv2[0:C, 0:Vd] <<= dv2[b_idx, h_idx, n_idx, 0:C, 0:Vd]
            l1_qraw[0:C, 0:Kd] <<= q_s[b_idx, h_idx, n_idx, 0:C, 0:Kd]   # q_s (input), used as ds_T@q_s
            vc_dq.wait()
            l1_dsL[0:C, 0:C] <<= ds_gm[b_idx, h_idx, n_idx, 0:C, 0:C]
            l1_dsTL[0:C, 0:C] <<= dsT_gm[b_idx, h_idx, n_idx, 0:C, 0:C]
            vc_dq.free()

            # ---- cube-2 + vec reduce, output-K subblocked.  STATIC unroll (k0 in {0,64})
            #      so the l1_h/l1_dh RIGHT-operand ROW slice and the store offset are
            #      compile-time literals (a runtime N-offset is mis-read by the sim).  No
            #      splitk (multi-chunk splitk is mis-lowered): output N=valid_k<=64 keeps
            #      the inter L0B fragment [valid_k,Vd] <= [64,128] = 32 KB (DBuff 64 KB).
            #      Second subblock runtime-guarded by If(Kd > k0). ----------------------
            cnt = Var(0)
            for k0 in (0, TK):                          # static literals 0, 64 (KMAX//TK == 2)
                with If(Kd > k0):
                    valid_k = Min(Kd - k0, TK)
                    # Load this output-K subblock's h/dh rows into SHORT [C,VMAX] tiles.
                    # s_hist/dh are inputs (no bridge).
                    l1_h[cnt][0:valid_k, 0:Vd] <<= s_hist[b_idx, h_idx, n_idx, k0:k0 + valid_k, 0:Vd]
                    l1_dh[cnt][0:valid_k, 0:Vd] <<= dh[b_idx, h_idx, n_idx, k0:k0 + valid_k, 0:Vd]
                    # dq_inter=do@h ; dq_intra=ds@k ; dk_inter=v_new@dh ; dk_intra=ds_T@q_raw ; dw=dv2@h
                    matmul(l0c_dqi[cnt], l1_do2[0:C, 0:Vd], l1_h[cnt][0:valid_k, 0:Vd], m=C, n=valid_k, k=Vd)
                    matmul(l0c_dqa[cnt], l1_dsL[0:C, 0:C], l1_k[0:C, k0:k0 + valid_k].T, m=C, n=valid_k, k=C)
                    matmul(l0c_dki[cnt], l1_vn2[0:C, 0:Vd], l1_dh[cnt][0:valid_k, 0:Vd], m=C, n=valid_k, k=Vd)
                    matmul(l0c_dka[cnt], l1_dsTL[0:C, 0:C], l1_qraw[0:C, k0:k0 + valid_k].T, m=C, n=valid_k, k=C)
                    matmul(l0c_dw[cnt], l1_dv2[0:C, 0:Vd], l1_h[cnt][0:valid_k, 0:Vd], m=C, n=valid_k, k=Vd)

                    cv_out.lock()
                    ub_dqi[cnt][0:C, 0:valid_k] <<= l0c_dqi[cnt][0:C, 0:valid_k].subblk(0)
                    ub_dqa[cnt][0:C, 0:valid_k] <<= l0c_dqa[cnt][0:C, 0:valid_k].subblk(0)
                    ub_dki[cnt][0:C, 0:valid_k] <<= l0c_dki[cnt][0:C, 0:valid_k].subblk(0)
                    ub_dka[cnt][0:C, 0:valid_k] <<= l0c_dka[cnt][0:C, 0:valid_k].subblk(0)
                    ub_dw[cnt][0:C, 0:valid_k] <<= l0c_dw[cnt][0:C, 0:valid_k].subblk(0)
                    cv_out.ready()
                    if row_start < C:
                        ub_do_f[0:C, 0:valid_k] <<= q_s[b_idx, h_idx, n_idx, 0:C, k0:k0 + valid_k]  # reuse cast tile
                        ub_kf[0:C, 0:valid_k] <<= k_f32[b_idx, h_idx, n_idx, 0:C, k0:k0 + valid_k]
                    cv_out.wait()
                    if row_start < C:
                        combine_vf(ub_dqi[cnt], ub_dqa[cnt], ub_dki[cnt], ub_dka[cnt], ub_dw[cnt], ub_do_f, ub_kf,
                                   ub_dexp, ub_decay, ub_const, ub_dgp, ub_dgl, valid_k)
                        dq_p[b_idx, h_idx, n_idx, 0:C, k0:k0 + valid_k] <<= ub_dqa[cnt][0:C, 0:valid_k]
                        dk_p[b_idx, h_idx, n_idx, 0:C, k0:k0 + valid_k] <<= ub_dka[cnt][0:C, 0:valid_k]
                        dw[b_idx, h_idx, n_idx, 0:C, k0:k0 + valid_k] <<= ub_dw[cnt][0:C, 0:valid_k]
                    cv_out.free()
                cnt += 1

            # ---- dg_last: exp(g_last)*sum_KV(s_hist*dh) + dgl ; fold onto dg_p[C-1] ----
            if row_start < C:
                for kb in range(k_tiles, name="sk"):
                    k0 = Var(kb * TK)
                    valid_kk = Min(Kd - k0, TK)
                    for vb in range(v_tiles, name="sv"):
                        v0 = Var(vb * TV)
                        valid_v = Min(Vd - v0, TV)
                        ub_ds[0:valid_kk, 0:valid_v] <<= s_hist[b_idx, h_idx, n_idx, k0:k0 + valid_kk, v0:v0 + valid_v]
                        ub_dsT[0:valid_kk, 0:valid_v] <<= dh[b_idx, h_idx, n_idx, k0:k0 + valid_kk, v0:v0 + valid_v]
                        shdh_vf(ub_ds, ub_dsT, ub_dgkv, valid_kk, valid_v)
                dglast_fold_vf(ub_dgp, ub_dgl, ub_dgkv, ub_decay)
                dg_p[b_idx, h_idx, n_idx, 0:C] <<= ub_dgp[0:1, 0:C]

    return dq_p, dk_p, dw, dg_p


# axis ids from trailing scalars (B,T,H,Kd,Vd,N,Cc,VT):
#   0=B 1=T 2=H 3=Kd 4=Vd 5=N 6=Cc(=C) 7=VT(=TV)
_SHAPE_BINDINGS = {
    0: [0, 1, 2, 4],      # grad_o   [B,T,H,V]
    1: [0, 2, 5, 6, 4],   # v_new    [B,H,N,C,V]
    2: [0, 2, 5, 6, 3],   # k_f32    [B,H,N,C,K]
    3: [0, 2, 5, 6, 3],   # q_s      [B,H,N,C,K]
    4: [0, 2, 5, 3, 4],   # s_hist   [B,H,N,K,V]
    5: [0, 2, 5, 3, 4],   # dh       [B,H,N,K,V]
    6: [0, 2, 5, 6, 4],   # dv2      [B,H,N,C,V]
    7: [0, 2, 5, 6],      # g_f32    [B,H,N,C]
    8: [0, 2, 5, 6],      # decay    [B,H,N,C]
    9: [0, 2, 5, 6, 4],   # do_gm    [B,H,N,C,V]
    10: [0, 2, 5, 6, 6],  # ds_gm    [B,H,N,C,C]
    11: [0, 2, 5, 6, 6],  # dsT_gm   [B,H,N,C,C]
    12: [0, 2, 5, 6, 3],  # dq_p     [B,H,N,C,K]
    13: [0, 2, 5, 6, 3],  # dk_p     [B,H,N,C,K]
    14: [0, 2, 5, 6, 3],  # dw       [B,H,N,C,K]
    15: [0, 2, 5, 6],     # dg_p     [B,H,N,C]
}


def build_args(B=1, T=200, H=3, K=128, V=96):
    """Contract-shape GM args for a transpile trace (worst-case shape by default)."""
    N = (T + C - 1) // C
    f = DT.float
    return gdr_bwd_dqkw, [
        GMTensor(DT.bfloat16, [B, T, H, V], name="grad_o"),
        GMTensor(f, [B, H, N, C, V], name="v_new"),
        GMTensor(f, [B, H, N, C, K], name="k_f32"),
        GMTensor(f, [B, H, N, C, K], name="q_s"),
        GMTensor(f, [B, H, N, K, V], name="s_hist"),
        GMTensor(f, [B, H, N, K, V], name="dh"),
        GMTensor(f, [B, H, N, C, V], name="dv2"),
        GMTensor(f, [B, H, N, C], name="g_f32"),
        GMTensor(f, [B, H, N, C], name="decay"),
        GMTensor(f, [B, H, N, C, V], name="do_gm"),
        GMTensor(f, [B, H, N, C, C], name="ds_gm"),
        GMTensor(f, [B, H, N, C, C], name="dsT_gm"),
        GMTensor(f, [B, H, N, C, K], name="dq_p"),
        GMTensor(f, [B, H, N, C, K], name="dk_p"),
        GMTensor(f, [B, H, N, C, K], name="dw"),
        GMTensor(f, [B, H, N, C], name="dg_p"),
        Var(B, name="B"), Var(T, name="T"), Var(H, name="H"),
        Var(K, name="Kd"), Var(V, name="Vd"), Var(N, name="N"),
        Var(C, name="Cc"), Var(TV, name="VT"),
    ]
