# KAL kernel -- OP3 gdr_bwd_dv_local (parent module M6, golden_step b1, BHN-parallel).
#
# Authored from the STRUCTURE-ONLY probe
#   custom/gated_delta_rule_bwd/kal/probes/probe_op3_dv_local.py
# with the DESIGN-frozen load-bearing corrections applied AND the real b1 arithmetic
# landed, then adapted to the A5 two-vector-sub-block execution model:
#
#   (1) q_t LAYOUT (DESIGN §9): declare q_t=[B,H,N,K,C] (pre-transposed residual), load
#       NON-transposed [K,C], feed as the RIGHT operand.  Standard k@q_t is expressed in
#       the easyasc facade as matmul(l0c_al, l1_k[C,K], l1_qt[K,C].T): the `.T` is a cheap
#       L0B (NZ2ZN) right-operand transpose, NOT an L0A/LEFT transpose (drift gate:
#       src_is_transpose into L0A = 0).
#   (2) SCALE: none.  scale=1/sqrt(K) is folded into q_t upstream (golden applies none).
#   (3) A5 SUB-BLOCKS: A5 has one VF unit split into two vector sub-blocks that both run
#       this stream.  A cube<->vec on-chip move (l0c->ub / ub->l1) touches a cube tensor
#       and is emitted CUBE-side, so it may NOT sit inside a vec `GetSubBlockIdx()` guard
#       (the simulator's split_instructions rejects a cube-only slice inside a vec
#       start_if).  The correct A5 pattern (mha_ifa) is DUAL/SPLITM: the [C,C]/[C,V] L0C
#       accumulators are split across the two sub-blocks (HC=C/2 rows each), every
#       cube<->vec move is unguarded, and each sub-block owns its HC-row band with a
#       runtime row offset row_off = GetSubBlockIdx()*HC.  The probe's SINGLE .subblk(0)
#       (never sim-verified) is not usable here.
#
# Dataflow (BHN-parallel, per (b,h,n) chunk; each vec sub-block owns rows [row_off,row_off+HC)):
#   [cube A]  A_l[C,C]      = k_f32[C,K] @ q_t[K,C]                (= k@q^T; scale in q_t)
#   --CvMutex cv_a-- SPLITM move A_l L0C->UB (this sub-block's HC rows)
#   [vec]     decay = cumsum(g_f32) (full C) ; A_l[i,:] *= exp(decay_j-decay_i) masked to
#             the upper triangle INCL diagonal (keep i<=j), i absolute = row_off+r ;
#             grad_o chunk bf16->fp32 (this sub-block's HC token rows, tail zero-filled)
#   --VcMutex vc_b-- A_l band -> L1[row_off:] ; grad_o_f32 band -> L1[row_off:]
#   [cube B]  dv_local[C,V] = A_l @ grad_o_chunk                  (dedicated L1/L0C)
#   --CvMutex cv_d-- SPLITM move dv_local L0C->UB (this sub-block's HC rows) ; store -> GM
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

# ---- operator key constants (DESIGN §2.1; used verbatim) ----
C = 64        # fixed chunk size (internal constant)
HC = C // 2   # 32 -- per vector-sub-block row band (A5 SPLITM cube<->vec)
KMAX = 128    # K upper bound (tile declared 128; runtime valid k = Kd)
VMAX = 128    # V upper bound (tile declared 128; runtime valid v = Vd)
SUB = 64      # VF register lane width (fp32): cast/zero the V axis in 64-col chunks


# =====================================================================
# Real b1 vector functions (all fp32; idioms mirror the parent bwd KAL
# decay_exp_vf / lmask_vf / prep_scalars_vf, adapted to M6's UPPER-triangle
# (keep i<=j) gate-decay factor exp(decay_j - decay_i) over an HC-row band).
# =====================================================================
@vf()
def cumsum_decay_vf(ub_g: Tensor, ub_decay: Tensor):
    """decay[c] = prefix-sum of g over the full C=64 row (g pad rows are 0). Register-
    resident sequential prefix sum (DESIGN §1 b1.2: Reg accumulation). Computed in full
    by each sub-block (the mask reads absolute decay_i / decay_j from the full row)."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    acc = Reg(DT.float)
    dup(acc, 0.0, mask=active)
    for c in range(C, name="csc"):
        gc = Reg(DT.float)
        gc <<= ub_g[0:1, c:c + 1].single()
        add(acc, acc, gc, mask=active)
        ub_decay[0:1, c:c + 1] <<= acc.single_value()


@vf()
def gate_mask_vf(ub_decay: Tensor, ub_al: Tensor, row_off: Var):
    """In-place over this sub-block's HC-row band of A_l: for local row r (absolute
    i = row_off + r), A_l[r,j] *= exp(decay_j - decay_i) for j>=i (upper triangle INCL
    diagonal, keep i<=j) else 0.

    Built with the PROVEN lmask_vf operand positions (scalar in expsub src0, vector in
    src1) via negated decay: exp(-decay_i - (-decay_j)) = exp(decay_j - decay_i). `select`
    zeros the discarded strict-lower region BEFORE the multiply, so exp(+large) there is
    never formed into the product (no inf*0 NaN)."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    cols = Reg(DT.int)
    zero = Reg(DT.float)
    dup(zero, 0.0, mask=active)
    cols.arange(0)
    dvec = Reg(DT.float)
    dvec <<= ub_decay[0:1, 0:C]                       # decay_j vector [1,C]
    ndvec = Reg(DT.float)
    neg(ndvec, dvec)                                  # -decay_j (vector, src1)
    for r in range(HC, name="gmi"):
        i = row_off + r                               # absolute row index (Var)
        di = Reg(DT.float)
        di <<= ub_decay[0:1, i:i + 1].single()        # decay_i (scalar broadcast)
        ndi = Reg(DT.float)
        neg(ndi, di)                                  # -decay_i (scalar, src0)
        gexp = Reg(DT.float)
        expsub(gexp, ndi, ndvec, mask=active)         # exp(-decay_i - (-decay_j)) = exp(decay_j-decay_i)
        m = MaskReg(DT.int, init_mode=MaskType.NONE)
        compare(m, cols, i, CompareMode.GE)           # cols >= i  <=>  j >= i (upper incl diag)
        select(gexp, gexp, zero, mask=m)              # keep for j>=i, else exact 0
        alrow = Reg(DT.float)
        alrow <<= ub_al[r:r + 1, 0:C]
        mul(alrow, alrow, gexp, mask=active)
        ub_al[r:r + 1, 0:C] <<= alrow


@vf()
def cast_go_vf(ub_go_bf: Tensor, ub_go_f: Tensor, valid_sb: Var):
    """grad_o band bf16->fp32 for this sub-block's valid token rows [0:valid_sb]; zero-fill
    pad rows [valid_sb:HC] so do padding-token rows contribute 0 to A_l@do (matching
    golden's F.pad(do,0)). Full declared VMAX width is cast; matmul n=Vd ignores columns
    [Vd:VMAX], so uninitialised pad columns are harmless."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    to_f = CastConfig(round_mode=RoundMode.NONE, reg_layout=RegLayout.ZERO, name="go_b2f")
    z = Reg(DT.float)
    dup(z, 0.0, mask=active)
    for r in range(valid_sb, name="gor"):
        for cc in range(0, VMAX, SUB):        # 64-lane fp32 register width -> chunk the V axis
            src = Reg(DT.bfloat16)
            dst = Reg(DT.float)
            ub_to_reg_unpack(src, ub_go_bf[r:r + 1, cc:cc + SUB])   # proven bf16 unpack (parent idiom)
            cast(dst, src, to_f, mask=active)
            reg_to_ub_normal(ub_go_f[r:r + 1, cc:cc + SUB], dst, mask=active)
    for r in range(valid_sb, HC, name="gopr"):  # zero-fill pad rows (this band) in 64-col chunks
        for cc in range(0, VMAX, SUB):
            reg_to_ub_normal(ub_go_f[r:r + 1, cc:cc + SUB], z, mask=active)


@kernel()
def gdr_bwd_dv_local(
    grad_o: GMTensor, k_f32: GMTensor, q_t: GMTensor, g_f32: GMTensor,
    dv_local: GMTensor,
    B: Var, T: Var, H: Var, Kd: Var, Vd: Var, N: Var, Cc: Var,
):
    cv_a = CvMutex(0, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)     # cube A -> vec
    vc_b = VcMutex(1, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)  # vec -> cube B
    cv_d = CvMutex(2, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)     # cube B -> vec

    # ---- cube A operands (own objects) ; A_l accumulator + its SPLITM UB band pair ----
    l1_k = Tensor(DT.float, [C, KMAX], Position.L1)      # k_f32 operand (LEFT, non-transposed)
    l1_qt = Tensor(DT.float, [KMAX, C], Position.L1)     # q_t operand [K,C] (CORRECTED: was [C,KMAX])
    l0c_al = DBuff(DT.float, [C, C], Position.L0C)       # A_l accumulator [C,C] (SPLITM source)
    ub_al = DBuff(DT.float, [HC, C], Position.UB)        # A_l band per sub-block (exp/mask in place)

    # ---- vec-side scratch (full C; each sub-block computes the full decay row) ----
    ub_g = Tensor(DT.float, [1, C], Position.UB)         # g_f32 chunk staging (full C)
    ub_decay = Tensor(DT.float, [1, C], Position.UB)     # decay = cumsum(g) (full C)
    ub_go_bf = Tensor(DT.bfloat16, [HC, VMAX], Position.UB)  # grad_o bf16 band
    ub_go_f = Tensor(DT.float, [HC, VMAX], Position.UB)  # grad_o fp32 band (cast)

    # ---- cube B operands/accumulator (DEDICATED -- M4 finding: no reuse across vec) ----
    l1_alB = Tensor(DT.float, [C, C], Position.L1)       # A_l operand (full, assembled from bands)
    l1_goB = Tensor(DT.float, [C, VMAX], Position.L1)    # grad_o operand (full, assembled from bands)
    l0c_dv = DBuff(DT.float, [C, VMAX], Position.L0C)    # dv_local accumulator [C,V] (SPLITM source)
    ub_dv = DBuff(DT.float, [HC, VMAX], Position.UB)     # dv_local band per sub-block

    tasks = B * H * N
    per = CeilDiv(tasks, GetCubeNum())
    begin = Var(per * GetCubeIdx())
    end = Min(begin + per, tasks)
    cnt = Var(0)                                          # DBuff rotation counter (task pipeline)

    with auto_sync():
        for tid in range(begin, end, name="tid"):
            b_idx = Var(tid // (H * N))
            rem = Var(tid % (H * N))
            h_idx = Var(rem // N)
            n_idx = Var(rem % N)
            t0 = Var(n_idx * C)                     # chunk's first t-row in public [B,T,H,V]
            row_off = Var(GetSubBlockIdx() * HC)    # this sub-block's row band start (0 or HC)
            t_sb = Var(t0 + row_off)                # first token of this sub-block's band
            valid_sb = Min(Max(T - t_sb, 0), HC)    # valid token rows for this band (tail-safe)

            # ---- cube A: A_l = k_f32 @ q_t ; SPLITM move A_l band to UB (producer side) ----
            l1_k[0:C, 0:Kd] <<= k_f32[b_idx, h_idx, n_idx, 0:C, 0:Kd]
            l1_qt[0:Kd, 0:C] <<= q_t[b_idx, h_idx, n_idx, 0:Kd, 0:C]     # CORRECTED [K,C] load
            matmul(l0c_al[cnt], l1_k[0:C, 0:Kd], l1_qt[0:Kd, 0:C].T, m=C, n=C, k=Kd)
            cv_a.lock()
            ub_al[cnt] <<= l0c_al[cnt]                        # SPLITM: this sub-block gets rows [row_off:row_off+HC]
            cv_a.ready()

            # ---- vec (consumer): decay=cumsum(g) ; A_l band *= masked exp(decay_j-decay_i) ----
            cv_a.wait()
            ub_g[0:1, 0:C] <<= g_f32[b_idx, h_idx, n_idx, 0:C]
            cumsum_decay_vf(ub_g, ub_decay)
            gate_mask_vf(ub_decay, ub_al[cnt], row_off)
            cv_a.free()

            # ---- vec: grad_o band bf16->fp32 ; assemble A_l / grad_o bands into L1 (vc_b producer) ----
            if valid_sb > 0:
                ub_go_bf[0:valid_sb, 0:Vd] <<= grad_o[b_idx, t_sb:t_sb + valid_sb, h_idx, 0:Vd]
            cast_go_vf(ub_go_bf, ub_go_f, valid_sb)
            vc_b.lock()
            l1_alB[row_off:row_off + HC, 0:C] <<= ub_al[cnt][0:HC, 0:C]      # A_l band -> L1 (vec->cube bridge)
            l1_goB[row_off:row_off + HC, 0:VMAX] <<= ub_go_f[0:HC, 0:VMAX]   # grad_o band -> L1
            bar_all()                                        # both sub-block bands assembled before cube-B reads
            vc_b.ready()

            # ---- cube B: dv_local = A_l @ grad_o (dedicated objects) ; SPLITM move band to UB ----
            vc_b.wait()
            # RIGHT operand sliced to VALID width [0:Vd] BEFORE .T: a full-width .T would
            # transpose the poison columns [Vd:VMAX] into the contraction (all-NaN dv).
            matmul(l0c_dv[cnt], l1_alB[0:C, 0:C], l1_goB[0:C, 0:Vd].T, m=C, n=Vd, k=C)
            vc_b.free()

            cv_d.lock()
            ub_dv[cnt][0:HC, 0:Vd] <<= l0c_dv[cnt][0:C, 0:Vd]  # SPLITM (valid width Vd only): rows [row_off:row_off+HC]
            cv_d.ready()

            # ---- vec (consumer): store dv_local band -> GM ----
            cv_d.wait()
            dv_local[b_idx, h_idx, n_idx, row_off:row_off + HC, 0:Vd] <<= ub_dv[cnt][0:HC, 0:Vd]
            cv_d.free()
            cnt += 1

    return dv_local


def build_args(B=1, T=200, H=3, K=128, V=96):
    C0 = C
    N = (T + C0 - 1) // C0
    f = DT.float
    return gdr_bwd_dv_local, [
        GMTensor(DT.bfloat16, [B, T, H, V], name="grad_o"),
        GMTensor(f, [B, H, N, C0, K], name="k_f32"),
        GMTensor(f, [B, H, N, K, C0], name="q_t"),           # CORRECTED [B,H,N,K,C]
        GMTensor(f, [B, H, N, C0], name="g_f32"),
        GMTensor(f, [B, H, N, C0, V], name="dv_local"),
        Var(B, name="B"), Var(T, name="T"), Var(H, name="H"),
        Var(K, name="Kd"), Var(V, name="Vd"), Var(N, name="N"), Var(C0, name="Cc"),
    ]
