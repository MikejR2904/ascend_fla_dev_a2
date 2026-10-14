# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# -----------------------------------------------------------------------------
# PyPTO-Pro Stage 3.5 (KAL pre-build) -- gdr_bwd_reverse_scan  (OP4 of
# gated_delta_rule_bwd; module M7, golden_step b2).  Single fusion module
# (module_count=1, is_fusion=true; DESIGN sec.0): the REVERSE inter-chunk dS
# scan, emitting dh [B,H,N,K,V] (entering-state snapshot) and dv2 [B,H,N,C,V].
#
# =========================== STRUCTURAL DUAL ================================
# This kernel is the DIRECTION-DUAL of the sim-verified + transpile-clean forward
# scan  custom/gdr_fwd_train_scan/kal/gdr_fwd_train_scan_kal_module1.py
# (itself the re-home of gated_delta_rule_fwd/kal/staged_scan_module5.py).  The
# forward scan realisation is followed VERBATIM in structure:
#   * loop-carried state (here dS[K,TV]) PERSISTENT in a Vec-UB tile, zeroed per
#     (b,h,v-subblock); producer-first (vec produces dS -> GM at chunk top, cube
#     consumes GM->L1) so there is NO cross-side loop-carried mutex / deadlock.
#   * 4 cube<->vec handoffs per chunk, delta_h ordering, Fix A GM-routed matmul
#     operand bridging (UB->GM store on vec, GM->L1 nd2nz on cube; never UB->L1,
#     which would materialise NZ-staging copies in UB and overflow 253952 B).
#   * SINGLE cube<->vec on-chip moves: `.subblk(0)` (forces dual_mode=SINGLE,
#     full-height read on sub-block 0) UNGUARDED on the cube side, with the vec
#     compute/store sub-block-0 guarded (`if GetSubBlockIdx()*C < C`).  This is
#     the forward scan's PROVEN A5 idiom.  (DESIGN sec.4 predicted a SPLITM/DUAL
#     HC-band restructure would be required; the forward-scan dual DISPROVES that
#     for this scan family -- the `.subblk(0)`-UNGUARDED + guarded-vec SINGLE
#     idiom is A5-correct and sim-verified.  See NOTES.md "reverse-direction
#     departure".  The A5-incorrect SINGLE that OP3 hit was `.subblk(0)` placed
#     INSIDE a vec guard; that is not what is done here.)
#
# ============================ A5 IDIOMS (OP3) ===============================
# Migrated from custom/gdr_bwd_dv_local/kal/NOTES.md:
#   * grad_o bf16->fp32 cast: ub_to_reg_unpack + cast(CastConfig ZERO) +
#     reg_to_ub_normal (fp32 lane width SUB=64; the plain `<<=` reads only half
#     the lanes across the bf16/fp32 lane boundary).
#   * register-resident cumsum(g) (no [C,C] scratch); gate = exp(g_last-cumsum(g))
#     computed with the fused `expsub` (scalar g_last src0, vector decay src1) --
#     NEVER decay_exp[C-1]/decay_exp[c] (0/0 -> NaN, DESIGN sec.1).
#   * RIGHT (L0B, transposed) matmul operands sliced to VALID width [0:valid_v]
#     BEFORE .T; a full-width .T transposes poison columns into the contraction.
#
# ============================== TILE SIZING =================================
# V is sub-blocked at TV=64 (DESIGN sec.2; finding #5): every transposed fp32 L0B
# operand (dS.T=[valid_v,Kd]<=[64,128]=8192, do.T/dv2.T=[valid_v,C]<=[64,64]) then
# fits the 8192-fp32 L0B transpose budget WITHOUT splitn.  This also keeps the
# resident tile set to ~140 KB (DESIGN sec.3 baseline 141,568 B), far below the
# forward scan's full-[128,128] 248,064 B.  Each V-column-block is an INDEPENDENT
# dS recurrence (all three matmuls contract over K or C, never V).
#
# Authored against the easyasc a5 facade; validated in the OpExec simulator;
# backend-transpiled to PyPTO-Pro DSL (kal/staged_module1.py).  No PyPTO /
# PyPTO-Pro API is hand-written.  Simulator pass != Stage-4 acceptance; this
# carries no certification.
#
# Reverse per-chunk dataflow (i = N-1 .. 0; loop-carried dS[K,TV] resident UB):
#   [dS_bridge  Vc] vec: dh[.,i] <- dS snapshot (entering; dh[N-1]=0) ; dS->dS_gm
#                   cube: dvc = k_f32_i @ dS            (cube-1, k=Kd)
#   [tdvc_out   Cv] cube: dvc L0C->UB
#                   vec: dv2 = dv_local_i + dvc*exp(g_last-cumsum(g)) ; store dv2
#                        do_scaled = decay_exp[c]*cast(grad_o_i) ; -dv2 in place
#   [dv2_bridge Vc] vec: do_scaled->do_gm ; -dv2->dv2neg_gm
#                   cube: delta = q_t_i @ do_scaled (init) += w_t_i @ (-dv2) (acc)
#   [delta_out  Cv] cube: delta L0C->UB
#                   vec: dS = dS*decay_exp_i[C-1] + delta   (loop-carry, in place)
# -----------------------------------------------------------------------------
"""gdr_bwd_reverse_scan -- M7 reverse dS scan (b2), emitting dh + dv2."""
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
TV = 64       # V-subblock width (keeps transposed L0B <= 8192 fp32 elems, finding #5)
TILE = 128    # K tile upper bound (K <= 128); dS / delta / dh row extent
SUB = 64      # fp32 VF register lane width (bf16->fp32 cast chunk)


# =====================================================================
# Vector functions (all fp32; SINGLE / sub-block-0 executes the full tile).
# =====================================================================
@vf()
def zero_dS_vf(ub_dS: Tensor, nrows: Var):
    """dS[0:nrows, 0:TV] = 0  (per (b,h,v-subblock) state reset; full TV width so
    tail V columns [valid_v:TV] stay 0 and never poison the recurrence)."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    z = Reg(DT.float)
    dup(z, 0.0, mask=active)
    for r in range(nrows, name="zdr"):
        reg_to_ub_normal(ub_dS[r:r + 1, 0:TV], z, mask=active)


@vf()
def cumsum_vf(ub_g: Tensor, ub_decay: Tensor):
    """decay[c] = prefix-sum of g over the full C row (g pad rows are 0). Register-
    resident sequential prefix sum (DESIGN sec.1; OP3 idiom)."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    acc = Reg(DT.float)
    dup(acc, 0.0, mask=active)
    for c in range(C, name="csc"):
        gc = Reg(DT.float)
        gc <<= ub_g[0:1, c:c + 1].single()
        add(acc, acc, gc, mask=active)
        ub_decay[0:1, c:c + 1] <<= acc.single_value()


@vf()
def dv2_gate_vf(ub_decay: Tensor, ub_tdvc: Tensor,
                ub_dvl: Tensor, ub_dv2: Tensor, valid_v: Var):
    """dv2[c,v] = dv_local[c,v] + dvc[c,v]*gate[c], gate[c]=exp(g_last-decay[c]).
    g_last=decay[C-1].  gate is formed PER ROW as a scalar with the fused `expsub`
    (bounded: g<=0 => g_last-decay[c] = sum_{j>c} g_j <= 0 => gate in (0,1]); NEVER
    via decay_exp[C-1]/decay_exp[c] (0/0 NaN).  Computed inline (no UB round-trip
    => no store->load VF RAW hazard)."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    glast = Reg(DT.float)
    glast <<= ub_decay[0:1, (C - 1):C].single()          # g_last scalar
    for c in range(C, name="dgc"):
        dc = Reg(DT.float)
        dc <<= ub_decay[0:1, c:c + 1].single()            # decay[c] scalar
        gate = Reg(DT.float)
        expsub(gate, glast, dc, mask=active)              # exp(g_last - decay[c]) scalar
        dvc = Reg(DT.float)
        dvc <<= ub_tdvc[c:c + 1, 0:valid_v]
        dvl = Reg(DT.float)
        dvl <<= ub_dvl[c:c + 1, 0:valid_v]
        dvc <<= dvc * gate
        dvc <<= dvc + dvl
        ub_dv2[c:c + 1, 0:valid_v] <<= dvc


@vf()
def neg_dv2_vf(ub_dv2: Tensor, valid_v: Var):
    """-dv2 in place (bridge to cube-2 so `+= w_t@(-dv2)` folds the subtraction)."""
    for r in range(C, name="ndr"):
        x = Reg(DT.float)
        x <<= ub_dv2[r:r + 1, 0:valid_v]
        y = Reg(DT.float)
        neg(y, x)
        ub_dv2[r:r + 1, 0:valid_v] <<= y


@vf()
def do_scaled_vf(ub_do_bf: Tensor, ub_do_f: Tensor, ub_dexp: Tensor,
                 valid_c: Var, valid_v: Var):
    """do_scaled[c,v] = decay_exp[c] * fp32(grad_o[c,v]) for valid token rows
    [0:valid_c]; pad rows [valid_c:C] zero-filled (=> term_q excludes padding
    tokens, matching golden's F.pad(do,0)).  bf16->fp32 via the OP3 unpack+cast
    idiom (valid_v <= TV = SUB, single lane chunk).  do_scaled folds decay_exp
    into do so q_t stays a clean non-transposed L0A operand (q_de^T@do == q_t@do_scaled)."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    to_f = CastConfig(round_mode=RoundMode.NONE, reg_layout=RegLayout.ZERO, name="go_b2f")
    z = Reg(DT.float)
    dup(z, 0.0, mask=active)
    for r in range(valid_c, name="dsr"):
        de = Reg(DT.float)
        de <<= ub_dexp[0:1, r:r + 1].single()             # decay_exp[c] scalar
        src = Reg(DT.bfloat16)
        dst = Reg(DT.float)
        ub_to_reg_unpack(src, ub_do_bf[r:r + 1, 0:valid_v])
        cast(dst, src, to_f, mask=active)
        dst <<= dst * de
        reg_to_ub_normal(ub_do_f[r:r + 1, 0:valid_v], dst, mask=active)
    for r in range(valid_c, C, name="dspr"):
        reg_to_ub_normal(ub_do_f[r:r + 1, 0:valid_v], z, mask=active)


@vf()
def dS_update_vf(ub_dS: Tensor, ub_delta: Tensor, ub_dexp: Tensor,
                 ck: Var, valid_v: Var):
    """dS = dS*decay_exp[C-1] + delta  (in-place UB loop-carry; K rows; NO extra
    scale -- scale is folded into q_t upstream)."""
    dreg = Reg(DT.float)
    dreg <<= ub_dexp[0:1, (C - 1):C].single()             # decay_exp[C-1] = exp(g_last)
    for r in range(ck, name="dsur"):
        s = Reg(DT.float)
        s <<= ub_dS[r:r + 1, 0:valid_v]
        d = Reg(DT.float)
        d <<= ub_delta[r:r + 1, 0:valid_v]
        s <<= s * dreg
        s <<= s + d
        ub_dS[r:r + 1, 0:valid_v] <<= s


@kernel()
def gdr_bwd_reverse_scan(
    q_t: GMTensor, w_t: GMTensor, k_f32: GMTensor, grad_o: GMTensor,
    decay_exp: GMTensor, g_f32: GMTensor, dv_local: GMTensor,
    dS_gm: GMTensor, do_gm: GMTensor, dv2neg_gm: GMTensor,
    dh: GMTensor, dv2: GMTensor,
    B: Var, T: Var, H: Var, Kd: Var, Vd: Var, N: Var, Cc: Var, VT: Var,
):
    # 4 cube<->vec handoffs (delta_h GM-routed idiom; producer MTE3 / consumer FIX
    # on vec->cube, FIX / V on cube->vec).
    dS_bridge = VcMutex(0, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)   # vec dS->GM ; cube dvc=k@dS
    tdvc_out = CvMutex(1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)       # cube dvc->UB ; vec dv2
    dv2_bridge = VcMutex(2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)  # vec do/-dv2->GM ; cube delta
    delta_out = CvMutex(3, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)      # cube delta->UB ; vec dS update

    # ---- Vec (UB) resident + scratch tiles ----
    ub_dS = Tensor(DT.float, [TILE, TV], Position.UB)      # loop-carried dS (persistent)
    ub_delta = Tensor(DT.float, [TILE, TV], Position.UB)   # delta landing (cube-2)
    ub_tdvc = Tensor(DT.float, [C, TV], Position.UB)       # dvc landing (cube-1)
    ub_dvl = Tensor(DT.float, [C, TV], Position.UB)        # dv_local load
    ub_dv2 = Tensor(DT.float, [C, TV], Position.UB)        # dv2 result / -dv2 in place
    ub_do_bf = Tensor(DT.bfloat16, [C, TV], Position.UB)   # grad_o bf16 load
    ub_do_f = Tensor(DT.float, [C, TV], Position.UB)       # do_scaled fp32
    ub_g = Tensor(DT.float, [1, C], Position.UB)           # g_f32 load / gate park
    ub_decay = Tensor(DT.float, [1, C], Position.UB)       # cumsum(g)
    ub_dexp = Tensor(DT.float, [1, C], Position.UB)        # decay_exp

    # ---- cube-1 (dvc = k@dS) operands / accumulator (dedicated) ----
    l1_k = Tensor(DT.float, [C, TILE], Position.L1)        # k_f32 [C,K] (LEFT, non-transposed)
    l1_dS = Tensor(DT.float, [TILE, TV], Position.L1)      # dS [K,V] (RIGHT, .T -> L0B)
    tdvc_l0c = Tensor(DT.float, [C, TV], Position.L0C)     # dvc [C,V]

    # ---- cube-2 (delta = q_t@do - w_t@dv2) operands / accumulator (dedicated) ----
    l1_qt = Tensor(DT.float, [TILE, C], Position.L1)       # q_t [K,C] (LEFT, non-transposed)
    l1_wt = Tensor(DT.float, [TILE, C], Position.L1)       # w_t [K,C] (LEFT, non-transposed)
    l1_do = Tensor(DT.float, [C, TV], Position.L1)         # do_scaled [C,V] (RIGHT, .T -> L0B)
    l1_dv2 = Tensor(DT.float, [C, TV], Position.L1)        # -dv2 [C,V] (RIGHT, .T -> L0B)
    delta_l0c = Tensor(DT.float, [TILE, TV], Position.L0C)  # delta [K,V]

    tasks = B * H
    per = CeilDiv(tasks, GetCubeNum())
    begin = Var(per * GetCubeIdx())
    end = Min(begin + per, tasks)
    v_tiles = CeilDiv(Vd, TV)

    with auto_sync():
        for bh in range(begin, end, name="bh"):
            b_idx = Var(bh // H)
            h_idx = Var(bh % H)
            row_start = Var(GetSubBlockIdx() * C)          # 0 on sub-block 0, C on sub-block 1

            for vb in range(v_tiles, name="vb"):
                v0 = Var(vb * TV)
                valid_v = Min(Vd - v0, TV)

                if row_start < C:
                    zero_dS_vf(ub_dS, Kd)                  # dS = 0 (sub-block 0 owns state)
                bar_all()

                for j in range(N, name="rj"):
                    i = Var(N - 1 - j)                     # reverse chunk order
                    t0 = Var(i * C)
                    valid_c = Min(T - t0, C)               # T-tail on the first reverse chunk

                    # ---- dS_bridge (Vc): snapshot dh, stage dS->GM ; cube dvc=k@dS ----
                    dS_bridge.lock()
                    if row_start < C:
                        dh[b_idx, h_idx, i, 0:Kd, v0:v0 + valid_v] <<= ub_dS[0:Kd, 0:valid_v]  # entering (dh[N-1]=0)
                        dS_gm[b_idx, h_idx, 0:Kd, 0:valid_v] <<= ub_dS[0:Kd, 0:valid_v]
                    dS_bridge.ready()

                    l1_k[0:C, 0:Kd] <<= k_f32[b_idx, h_idx, i, 0:C, 0:Kd]
                    dS_bridge.wait()
                    l1_dS[0:Kd, 0:valid_v] <<= dS_gm[b_idx, h_idx, 0:Kd, 0:valid_v]
                    # dvc = k_f32[C,K] @ dS[K,V]; dS is RIGHT/L0B (.T is the L1->L0B
                    # transpose, sliced to valid_v BEFORE .T so pad cols never enter).
                    matmul(tdvc_l0c, l1_k[0:C, 0:Kd], l1_dS[0:Kd, 0:valid_v].T,
                           m=C, n=valid_v, k=Kd)
                    dS_bridge.free()

                    # ---- tdvc_out (Cv): cube dvc->UB ; vec dv2 / do_scaled / -dv2 ----
                    tdvc_out.lock()
                    ub_tdvc[0:C, 0:valid_v] <<= tdvc_l0c[0:C, 0:valid_v].subblk(0)
                    tdvc_out.ready()

                    if row_start < C:
                        ub_dvl[0:C, 0:valid_v] <<= dv_local[b_idx, h_idx, i, 0:C, v0:v0 + valid_v]
                        ub_g[0:1, 0:C] <<= g_f32[b_idx, h_idx, i, 0:C]
                        ub_dexp[0:1, 0:C] <<= decay_exp[b_idx, h_idx, i, 0:C]
                        if valid_c > 0:
                            ub_do_bf[0:valid_c, 0:valid_v] <<= grad_o[b_idx, t0:t0 + valid_c, h_idx, v0:v0 + valid_v]
                    tdvc_out.wait()
                    if row_start < C:
                        cumsum_vf(ub_g, ub_decay)                              # decay = cumsum(g)
                        dv2_gate_vf(ub_decay, ub_tdvc, ub_dvl, ub_dv2, valid_v)
                        dv2[b_idx, h_idx, i, 0:C, v0:v0 + valid_v] <<= ub_dv2[0:C, 0:valid_v]  # output (full C)
                        do_scaled_vf(ub_do_bf, ub_do_f, ub_dexp, valid_c, valid_v)
                        neg_dv2_vf(ub_dv2, valid_v)                            # ub_dv2 now holds -dv2
                    tdvc_out.free()

                    # ---- dv2_bridge (Vc): stage do_scaled/-dv2 ->GM ; cube delta ----
                    dv2_bridge.lock()
                    if row_start < C:
                        do_gm[b_idx, h_idx, 0:C, 0:valid_v] <<= ub_do_f[0:C, 0:valid_v]
                        dv2neg_gm[b_idx, h_idx, 0:C, 0:valid_v] <<= ub_dv2[0:C, 0:valid_v]
                    dv2_bridge.ready()

                    l1_qt[0:Kd, 0:C] <<= q_t[b_idx, h_idx, i, 0:Kd, 0:C]       # non-transposed L0A
                    l1_wt[0:Kd, 0:C] <<= w_t[b_idx, h_idx, i, 0:Kd, 0:C]       # non-transposed L0A
                    dv2_bridge.wait()
                    l1_do[0:C, 0:valid_v] <<= do_gm[b_idx, h_idx, 0:C, 0:valid_v]
                    l1_dv2[0:C, 0:valid_v] <<= dv2neg_gm[b_idx, h_idx, 0:C, 0:valid_v]
                    # delta = q_t@do_scaled (term_q, init) += w_t@(-dv2) (= -term_w, acc).
                    # M=TILE (full) with q_t/w_t loaded [0:Kd]: partial-M (m=Kd<TILE) on
                    # a [TILE,TV] L0C scrambles the result; full-M reads the uninit
                    # rows [Kd:TILE] into ignored output rows only (matmul rows are
                    # independent; those rows are never moved/read).  The RIGHT/L0B
                    # operand is sliced to valid_v BEFORE .T (poison-free), k=C full.
                    matmul(delta_l0c, l1_qt[0:TILE, 0:C], l1_do[0:C, 0:valid_v].T,
                           m=TILE, n=valid_v, k=C, is_init=True)
                    matmul(delta_l0c, l1_wt[0:TILE, 0:C], l1_dv2[0:C, 0:valid_v].T,
                           m=TILE, n=valid_v, k=C, is_init=False)
                    dv2_bridge.free()

                    # ---- delta_out (Cv): cube delta->UB ; vec dS update ----
                    delta_out.lock()
                    ub_delta[0:Kd, 0:valid_v] <<= delta_l0c[0:Kd, 0:valid_v].subblk(0)
                    delta_out.ready()

                    delta_out.wait()
                    if row_start < C:
                        dS_update_vf(ub_dS, ub_delta, ub_dexp, Kd, valid_v)
                    delta_out.free()

    return dh, dv2


# axis ids from trailing scalars (B,T,H,Kd,Vd,N,Cc,VT):
#   0=B 1=T 2=H 3=Kd 4=Vd 5=N 6=Cc(=C) 7=VT(=TV)
_SHAPE_BINDINGS = {
    0: [0, 2, 5, 3, 6],   # q_t        [B,H,N,K,C]
    1: [0, 2, 5, 3, 6],   # w_t        [B,H,N,K,C]
    2: [0, 2, 5, 6, 3],   # k_f32      [B,H,N,C,K]
    3: [0, 1, 2, 4],      # grad_o     [B,T,H,V]
    4: [0, 2, 5, 6],      # decay_exp  [B,H,N,C]
    5: [0, 2, 5, 6],      # g_f32      [B,H,N,C]
    6: [0, 2, 5, 6, 4],   # dv_local   [B,H,N,C,V]
    7: [0, 2, 3, 7],      # dS_gm      [B,H,K,TV]
    8: [0, 2, 6, 7],      # do_gm      [B,H,C,TV]
    9: [0, 2, 6, 7],      # dv2neg_gm  [B,H,C,TV]
    10: [0, 2, 5, 3, 4],  # dh         [B,H,N,K,V]
    11: [0, 2, 5, 6, 4],  # dv2        [B,H,N,C,V]
}


def build_args(B=1, T=200, H=3, K=128, V=96):
    """Contract-shape GM args for a transpile trace (worst-case shape by default)."""
    N = (T + C - 1) // C
    f = DT.float
    return gdr_bwd_reverse_scan, [
        GMTensor(f, [B, H, N, K, C], name="q_t"),
        GMTensor(f, [B, H, N, K, C], name="w_t"),
        GMTensor(f, [B, H, N, C, K], name="k_f32"),
        GMTensor(DT.bfloat16, [B, T, H, V], name="grad_o"),
        GMTensor(f, [B, H, N, C], name="decay_exp"),
        GMTensor(f, [B, H, N, C], name="g_f32"),
        GMTensor(f, [B, H, N, C, V], name="dv_local"),
        GMTensor(f, [B, H, K, TV], name="dS_gm"),
        GMTensor(f, [B, H, C, TV], name="do_gm"),
        GMTensor(f, [B, H, C, TV], name="dv2neg_gm"),
        GMTensor(f, [B, H, N, K, V], name="dh"),
        GMTensor(f, [B, H, N, C, V], name="dv2"),
        Var(B, name="B"), Var(T, name="T"), Var(H, name="H"),
        Var(K, name="Kd"), Var(V, name="Vd"), Var(N, name="N"),
        Var(C, name="Cc"), Var(TV, name="VT"),
    ]
