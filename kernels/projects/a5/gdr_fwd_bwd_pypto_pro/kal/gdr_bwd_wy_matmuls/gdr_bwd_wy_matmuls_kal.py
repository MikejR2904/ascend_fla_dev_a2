"""gdr_bwd_wy_matmuls (OP6a) -- EasyASC / KAL kernel (Stage 3.5 hardware-aware layer).

CUBE half of module M9 (golden_step b4): 8 fp32 matmuls in 3 cube sections split by
2 vector-glue masks. BHN-parallel, stateless. Authored against the A5 facade, validated
in the Python simulator vs gdr_bwd_wy_matmuls_golden_cpu, then transpiled to PyPTO-Pro DSL.

Design intent source: custom/gdr_bwd_wy_matmuls/DESIGN.md (§0, §2-§10) + golden math.
NOT the DESIGN §1 API sequence (that is PyPTO-Pro; this layer uses the easyasc facade).

Key layer facts used here (measured/verified against the pypto transpiler & sim):
  * easyasc matmul(dst, X, Y) computes  X @ Y.T  (MK @ NK -> MN). Every b4.* below is
    derived into this convention (see per-matmul comments).
  * transposed-LEFT operands (A^T, dA^T) are loaded straight from GM with dn2nz (`gm.T`),
    landing a plain NZ L1 tile -- NO on-chip transpose, NO GM pre-transpose scratch.
    (l1_to_l0 `src_is_transpose` into L0A is a transpile gap; the transpose must be
    chosen at the GM->L1 load.)
  * transposed-RIGHT operands use the free L1-side `.T` (into L0B).
  * l0c_to_l1 (Acc->Mat on-chip bridge) does NOT transpile to PyPTO-Pro (no pypto
    handler -> silently dropped) and a same-core GM self-bridge is UNSYNCED (auto_sync
    does not track GM RAW).  The cube-B internal bridge (b4.5 -> b4.6) therefore routes
    L0C -> UB -> L1 across the vec lane: l0c_to_ub (cube FIX) hands ub_bridge to the vec,
    which does ub_to_l1_nd2nz into l1_m16.  Both ops transpile; the cross-lane sync is
    carried by the cv_mid/vc_mid mutex pair.
  * TWO vec sublanes execute every @vf and vec datamove redundantly, so every vec write
    of a SHARED L1 tile or a GM output is guarded to a single sublane (GetSubBlockIdx()<1):
    letting both sublanes DMA the same region tears the write and is non-deterministic.
  * cube consumers load their OWN operands AFTER the handoff mutex .wait(); loading them
    before the wait makes the fused pipe schedule race.
  * a vector Reg holds only C=64 fp32 lanes, so K up to KMAX=128 is processed in two
    64-wide halves (form_kbeta) -- a single 0:KMAX Reg load drops the upper 64 lanes.
  * splitn/splitk keep every per-mmad L0A/L0B operand <= [64,64] = 16 KB -> 32 KB peak.
"""
import os
import sys

for _cand in (
    "/workspace/pypto-gym/cannbot-skills/ops/pypto-kernel-arch-lang/payload",
    "/root/.claude/skills/pypto-kernel-arch-lang/payload",
):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

from easyasc.a5 import *  # noqa: F401,F403

# ── compile-time constants (module-level ints; on-chip tile shapes are literals) ──
C = 64            # fixed chunk size (attn/dA both dims; row dim of every [C,*])
TK = 64           # output-subblock along K (splitn)
TV = 64           # output-subblock along V (splitn)
KL0 = 64          # contraction-axis split step (splitk)
KMAX = 128        # K upper bound (physical L1 tile width; runtime Kd<=KMAX)
VMAX = 128        # V upper bound

# Toggle: when True the two vector-glue phases are identity stubs (C0 structure /
# budget measurement). When False they run the real strict_low / decay masks (C1+).
STUB_GLUE = os.environ.get("KAL_STUB_GLUE", "0") == "1"


# ─────────────────────────── vector-function glue ───────────────────────────

@vf()
def glue1_strict_low(dA: Tensor, n: Var):
    """glue-1: dA = strict_low * dA  (keep row i > col j, zero elsewhere).

    Per row i, keep the lowest i lanes (columns 0..i-1), zero the rest.
    """
    reg = Reg(DT.float)
    out = Reg(DT.float)
    zero = Reg(DT.float)
    mask = MaskReg(DT.float)
    zero <<= 0.0
    for i in range(n, name="g1r"):
        reg <<= dA[i:i + 1, 0:C]
        mask <<= Var(i, dtype=DT.uint32)  # update_mask: lowest i lanes active (j < i)
        out <<= mask.select(reg, zero)
        dA[i:i + 1, 0:C] <<= out


@vf()
def glue1_stub(dA: Tensor, n: Var):
    reg = Reg(DT.float)
    for i in range(n, name="g1s"):
        reg <<= dA[i:i + 1, 0:C]
        dA[i:i + 1, 0:C] <<= reg


@vf()
def glue2_decay_negate(dA: Tensor, decay: Tensor, n: Var):
    """glue-2: decay_fac = exp((decay_i - decay_j) masked to j<i) ; dA = -(dA * decay_fac).

    mask-before-exp: gdiff is zeroed on the upper triangle BEFORE exp (no inf), then the
    final product is re-masked to strict-lower. Result is the FINAL d_a.
    """
    d_row = Reg(DT.float)
    di = Reg(DT.float)
    gdiff = Reg(DT.float)
    ef = Reg(DT.float)
    reg = Reg(DT.float)
    out = Reg(DT.float)
    zero = Reg(DT.float)
    mask = MaskReg(DT.float)
    zero <<= 0.0
    d_row <<= decay[0:1, 0:C]                       # decay_j vector (C lanes)
    for i in range(n, name="g2r"):
        di <<= decay[0:1, i:i + 1].single()         # scalar decay_i broadcast
        gdiff <<= di - d_row                         # decay_i - decay_j (per lane)
        gdiff <<= gdiff.vmins(0.0)                    # cap at 0 -> exp in (0,1], NO overflow
        ef <<= gdiff.exp()                           # exp(min(decay_i-decay_j, 0))
        reg <<= dA[i:i + 1, 0:C]
        reg <<= reg * ef
        reg <<= reg * (-1.0)                         # negate
        mask <<= Var(i, dtype=DT.uint32)             # lowest i lanes active (j < i)
        out <<= mask.select(reg, zero)               # strict-lower: keep j<i, zero j>=i
        dA[i:i + 1, 0:C] <<= out


@vf()
def glue2_stub(dA: Tensor, decay: Tensor, n: Var):
    reg = Reg(DT.float)
    for i in range(n, name="g2s"):
        reg <<= dA[i:i + 1, 0:C]
        dA[i:i + 1, 0:C] <<= reg


@vf()
def form_kbeta(k: Tensor, beta: Tensor, kbeta: Tensor, n: Var, w: Var):
    """k_beta[i, :] = k_f32[i, :] * beta_act[i]  (on-chip, feeds b4.8).

    K spans up to KMAX=128 columns but one vector Reg holds only C=64 fp32 lanes, so the
    row is processed in two 64-wide halves (a single 0:KMAX Reg load silently drops the
    upper 64 lanes -> poison in k_beta[:, 64:128], which only shows up at K=128).
    """
    reg0 = Reg(DT.float)
    reg1 = Reg(DT.float)
    b = Reg(DT.float)
    for i in range(n, name="kbr"):
        b <<= beta[0:1, i:i + 1].single()
        reg0 <<= k[i:i + 1, 0:C]
        reg1 <<= k[i:i + 1, C:KMAX]
        reg0 <<= reg0 * b
        reg1 <<= reg1 * b
        kbeta[i:i + 1, 0:C] <<= reg0
        kbeta[i:i + 1, C:KMAX] <<= reg1


@vf()
def form_kbeta_stub(k: Tensor, beta: Tensor, kbeta: Tensor, n: Var, w: Var):
    reg0 = Reg(DT.float)
    reg1 = Reg(DT.float)
    for i in range(n, name="kbs"):
        reg0 <<= k[i:i + 1, 0:C]
        reg1 <<= k[i:i + 1, C:KMAX]
        kbeta[i:i + 1, 0:C] <<= reg0
        kbeta[i:i + 1, C:KMAX] <<= reg1


# Resolve the glue callables ONCE at import (no Python `if` inside the traced body).
_GLUE1 = glue1_stub if STUB_GLUE else glue1_strict_low
_GLUE2 = glue2_stub if STUB_GLUE else glue2_decay_negate
_KBETA = form_kbeta_stub if STUB_GLUE else form_kbeta


# ─────────────────────────────── the kernel ───────────────────────────────

@kernel()
def gdr_bwd_wy_matmuls_kernel(
    attn: GMTensor, k_f32: GMTensor, v_b: GMTensor, kbeta_t: GMTensor,
    beta_act: GMTensor, decay: GMTensor, dw: GMTensor, dv2: GMTensor,
    dA_gm: GMTensor, kbeta_gm: GMTensor,
    dkbg: GMTensor, dvb: GMTensor, dkb: GMTensor, dkt: GMTensor, d_a: GMTensor,
    B: Var, H: Var, N: Var, Kd: Var, Vd: Var, Cc: Var,
):
    # cube<->vec handoffs (auto_mutex + cross_core). dA crosses cube<->vec 4 times.
    cv_a = CvMutex(0, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    vc_b = VcMutex(1, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    cv_b = CvMutex(2, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    vc_c = VcMutex(3, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    # cube-B internal b4.5->b4.6 bridge: L0C -> UB -> L1 relayout MUST bounce through vec
    # (l0c_to_ub is a cube/FIX op, ub_to_l1_nd2nz is a vec/MTE3 op). GM/UB cross-lane RAW
    # is only synced by a Cv/Vc mutex pair (auto_sync does not track GM; the UB relay op
    # runs on the vec lane), so the vec relayout is guarded by cv_mid + vc_mid.
    cv_mid = CvMutex(4, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE3)
    vc_mid = VcMutex(5, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)

    # ---- UB (vec side) ----
    ub_dA = Tensor(DT.float, [C, C], Position.UB)          # dA landing / glue-1 / glue-2 (d_a)
    ub_decay = Tensor(DT.float, [C, C], Position.UB)        # decay row (uses first row)
    ub_k = Tensor(DT.float, [C, KMAX], Position.UB)         # k_f32 for k_beta
    ub_beta = Tensor(DT.float, [C, C], Position.UB)         # beta_act row (uses first row)
    ub_kbeta = Tensor(DT.float, [C, KMAX], Position.UB)     # k_beta result
    ub_bridge = Tensor(DT.float, [C, C], Position.UB)       # cube-B b4.5->b4.6 on-chip relay

    # ---- cube-A : own L1 / L0C ----
    l1_dw = Tensor(DT.float, [C, KMAX], Position.L1)
    l1_kbT = Tensor(DT.float, [KMAX, C], Position.L1)       # kbeta_t [K,C] (pre-transposed input)
    l1_attnT = Tensor(DT.float, [C, C], Position.L1)        # A^T via dn2nz (LEFT of b4.2/4.4)
    l1_dv2 = Tensor(DT.float, [C, VMAX], Position.L1)
    l1_vb = Tensor(DT.float, [C, VMAX], Position.L1)
    # L0C off-peak reuse: within a cube section a wide [C,KMAX] Acc is reused for the
    # two output products (dkbg then dvb; dkb then dkt), each stored before the next
    # overwrites it (auto_sync WAR). Sections keep DEDICATED L0C (M4 finding): cube-A's
    # tiles are not reused across the vec boundary by cube-B/C.
    l0c_dA = Tensor(DT.float, [C, C], Position.L0C)         # dA (b4.1 seed + b4.3 acc)
    l0c_wideA = Tensor(DT.float, [C, KMAX], Position.L0C)   # cube-A: dkbg, then dvb

    # ---- cube-B : DEDICATED L1 / L0C ----
    l1_dAb = Tensor(DT.float, [C, C], Position.L1)          # masked dA (from dA_gm)
    l1_attn = Tensor(DT.float, [C, C], Position.L1)         # A natural (RIGHT of b4.5)
    l1_attnT2 = Tensor(DT.float, [C, C], Position.L1)       # A^T via dn2nz (LEFT of b4.6)
    l1_m16 = Tensor(DT.float, [C, C], Position.L1)          # bridge T = dA@A^T (from ub_bridge)
    l0c_b5 = Tensor(DT.float, [C, C], Position.L0C)         # cube-B: b4.5  T = dA@A^T
    l0c_bb = Tensor(DT.float, [C, C], Position.L0C)         # cube-B: b4.6  dA = A^T@T

    # ---- cube-C : DEDICATED L1 / L0C ----
    l1_dAc = Tensor(DT.float, [C, C], Position.L1)          # dA final natural (LEFT of b4.7)
    l1_dATc = Tensor(DT.float, [C, C], Position.L1)         # dA^T via dn2nz (LEFT of b4.8)
    l1_kc = Tensor(DT.float, [C, KMAX], Position.L1)        # k_f32
    l1_kbe = Tensor(DT.float, [C, KMAX], Position.L1)       # k_beta (from kbeta_gm)
    l0c_wideC = Tensor(DT.float, [C, KMAX], Position.L0C)   # cube-C: dkb, then dkt

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

            # ============================ cube-A : b4.1-b4.4 ============================
            cv_a.lock()
            l1_dw[0:C, 0:Kd] <<= dw[b_idx, h_idx, n_idx, 0:C, 0:Kd]
            l1_kbT[0:Kd, 0:C] <<= kbeta_t[b_idx, h_idx, n_idx, 0:Kd, 0:C]
            gm_to_l1_dn2nz(l1_attnT, attn[b_idx, h_idx, n_idx, 0:C, 0:C],
                           M=C, N=C, N_src=C, M_dst=C)                     # dn2nz -> A^T
            l1_dv2[0:C, 0:Vd] <<= dv2[b_idx, h_idx, n_idx, 0:C, 0:Vd]
            l1_vb[0:C, 0:Vd] <<= v_b[b_idx, h_idx, n_idx, 0:C, 0:Vd]

            # b4.1  dA = dw @ kbeta_t.  X@Y.T = dw @ (kbT.T).T = dw @ kbeta_t.  splitk over K.
            matmul(l0c_dA, l1_dw[0:C, 0:KMAX], l1_kbT[0:KMAX, 0:C].T,
                   splitk=KL0, m=C, n=C, k=Kd, is_init=True)
            # b4.2  dkbg = A^T @ dw.  X@Y.T = attnT @ (dw.T).T = A^T @ dw.  splitn over K.
            matmul(l0c_wideA, l1_attnT[0:C, 0:C], l1_dw[0:C, 0:KMAX].T,
                   splitn=TK, m=C, n=Kd, k=C, is_init=True)
            dkbg[b_idx, h_idx, n_idx, 0:C, 0:Kd] <<= l0c_wideA[0:C, 0:Kd]     # FINAL dkbg
            # b4.3  dA += dv2 @ v_b.T.  X@Y.T = dv2 @ v_b.T.  splitk over V.  accumulate.
            matmul(l0c_dA, l1_dv2[0:C, 0:VMAX], l1_vb[0:C, 0:VMAX],
                   splitk=KL0, m=C, n=C, k=Vd, is_init=False)
            # b4.4  dvb = A^T @ dv2.  X@Y.T = attnT @ (dv2.T).T = A^T @ dv2.  splitn over V.
            matmul(l0c_wideA, l1_attnT[0:C, 0:C], l1_dv2[0:C, 0:VMAX].T,
                   splitn=TV, m=C, n=Vd, k=C, is_init=True)
            dvb[b_idx, h_idx, n_idx, 0:C, 0:Vd] <<= l0c_wideA[0:C, 0:Vd]      # FINAL dvb
            # dA (raw) -> UB for glue-1
            ub_dA[0:C, 0:C] <<= l0c_dA.subblk(0)
            cv_a.ready()

            # ============================ glue-1 (vec) ============================
            cv_a.wait()
            _GLUE1(ub_dA, Cc)
            cv_a.free()

            # vec -> cube-B handoff via SHARED L1 tile (canonical pattern; a GM bridge is
            # NOT synced -- auto_sync ignores GM RAW).  glue-1's masked dA lands directly in
            # l1_dAb via ub_to_l1_nd2nz; cube-B reads that L1 tile as the b4.5 LEFT operand.
            vc_b.lock()
            if GetSubBlockIdx() < 1:
                l1_dAb[0:C, 0:C] <<= ub_dA[0:C, 0:C]                         # vec ub_to_l1 (single sublane)
            vc_b.ready()

            # ============================ cube-B (dedicated) : b4.5-b4.6 ============================
            # NOTE: cube loads its OWN operands AFTER vc_b.wait() -- loading them before the
            # handoff wait makes the fused pipe schedule non-deterministic (empirically races).
            vc_b.wait()
            l1_attn[0:C, 0:C] <<= attn[b_idx, h_idx, n_idx, 0:C, 0:C]        # A natural
            gm_to_l1_dn2nz(l1_attnT2, attn[b_idx, h_idx, n_idx, 0:C, 0:C],
                           M=C, N=C, N_src=C, M_dst=C)                       # A^T via dn2nz
            # b4.5  T = dA @ A^T.  X@Y.T = dA @ attn.T = dA @ A^T.  (dA = shared l1_dAb)
            matmul(l0c_b5, l1_dAb[0:C, 0:C], l1_attn[0:C, 0:C], m=C, n=C, k=C)
            # b4.5 -> b4.6 bridge: T must leave L0C and re-enter as an L0B operand.  Direct
            # l0c_to_l1 is a transpile gap (no pypto handler -> silently dropped), so route
            # L0C -> UB -> L1 through the vec lane: l0c_to_ub (cube FIX) hands ub_bridge to
            # the vec, which does ub_to_l1_nd2nz into l1_m16 (single sublane -- both sublanes
            # DMAing the same L1 region tears the write).  cv_mid + vc_mid carry the sync.
            cv_mid.lock()                                    # [cube] acquire ub_bridge slot
            ub_bridge[0:C, 0:C] <<= l0c_b5.subblk(0)         # [cube FIX] l0c_to_ub
            cv_mid.ready()
            vc_mid.lock()                                    # [vec] acquire l1_m16 slot
            cv_mid.wait()
            if GetSubBlockIdx() < 1:
                l1_m16[0:C, 0:C] <<= ub_bridge[0:C, 0:C]     # [vec MTE3] ub_to_l1_nd2nz (relayout T)
            cv_mid.free()
            vc_mid.ready()
            vc_mid.wait()                                    # [cube] wait l1_m16
            # b4.6  dA = A^T @ T.  X@Y.T = attnT2 @ (m16.T).T = A^T @ T.  (dedicated l0c_bb)
            matmul(l0c_bb, l1_attnT2[0:C, 0:C], l1_m16[0:C, 0:C].T, m=C, n=C, k=C)
            vc_mid.free()        # [cube] release l1_m16
            vc_b.free()          # cube-B done consuming l1_dAb -> release to glue-1

            cv_b.lock()
            ub_dA[0:C, 0:C] <<= l0c_bb.subblk(0)
            cv_b.ready()

            # ============================ glue-2 (vec) ============================
            cv_b.wait()
            ub_decay[0:1, 0:C] <<= decay[b_idx, h_idx, n_idx, 0:C]
            _GLUE2(ub_dA, ub_decay, Cc)
            # form k_beta = k_f32 * beta_act (on-chip) -> kbeta_gm  (for b4.8)
            ub_k[0:C, 0:Kd] <<= k_f32[b_idx, h_idx, n_idx, 0:C, 0:Kd]
            ub_beta[0:1, 0:C] <<= beta_act[b_idx, h_idx, n_idx, 0:C]
            _KBETA(ub_k, ub_beta, ub_kbeta, Cc, Kd)
            # vec stores are done by a SINGLE sublane (both sublanes run the @vf redundantly;
            # letting both DMA the same GM/L1 region tears the write -> non-deterministic).
            if GetSubBlockIdx() < 1:
                d_a[b_idx, h_idx, n_idx, 0:C, 0:C] <<= ub_dA[0:C, 0:C]           # FINAL d_a
                kbeta_gm[b_idx, h_idx, n_idx, 0:C, 0:Kd] <<= ub_kbeta[0:C, 0:Kd]
            cv_b.free()

            vc_c.lock()
            # dA final -> dA_gm for cube-C (single sublane store).
            if GetSubBlockIdx() < 1:
                dA_gm[b_idx, h_idx, n_idx, 0:C, 0:C] <<= ub_dA[0:C, 0:C]
            vc_c.ready()

            # ============================ cube-C (dedicated) : b4.7-b4.8 ============================
            # ALL cube-C operands load AFTER vc_c.wait(): l1_kbe reads kbeta_gm (a vec-produced
            # GM value) so it must be gated by the mutex; l1_dAc/l1_dATc read dA_gm (vec-produced).
            vc_c.wait()
            l1_kc[0:C, 0:Kd] <<= k_f32[b_idx, h_idx, n_idx, 0:C, 0:Kd]
            l1_kbe[0:C, 0:Kd] <<= kbeta_gm[b_idx, h_idx, n_idx, 0:C, 0:Kd]
            l1_dAc[0:C, 0:C] <<= dA_gm[b_idx, h_idx, n_idx, 0:C, 0:C]        # dA final natural
            gm_to_l1_dn2nz(l1_dATc, dA_gm[b_idx, h_idx, n_idx, 0:C, 0:C],
                           M=C, N=C, N_src=C, M_dst=C)                       # dA^T via dn2nz
            vc_c.free()          # cube-C done consuming dA_gm -> release to glue-2
            # b4.7  dkb = dA @ k_f32.  X@Y.T = dA @ (kc.T).T = dA @ k_f32.  splitn over K.
            matmul(l0c_wideC, l1_dAc[0:C, 0:C], l1_kc[0:C, 0:KMAX].T,
                   splitn=TK, m=C, n=Kd, k=C, is_init=True)
            dkb[b_idx, h_idx, n_idx, 0:C, 0:Kd] <<= l0c_wideC[0:C, 0:Kd]      # FINAL dkb
            # b4.8  dkt = dA^T @ k_beta.  X@Y.T = dATc @ (kbe.T).T = dA^T @ k_beta.  (reuse wideC)
            matmul(l0c_wideC, l1_dATc[0:C, 0:C], l1_kbe[0:C, 0:KMAX].T,
                   splitn=TK, m=C, n=Kd, k=C, is_init=True)
            dkt[b_idx, h_idx, n_idx, 0:C, 0:Kd] <<= l0c_wideC[0:C, 0:Kd]      # FINAL dkt

    return dkbg, dvb, dkb, dkt, d_a


def build_args(B=1, H=2, N=3, K=128, V=128):
    """Worst-case default shape (K=V=128, N=3) for budget measurement."""
    f = DT.float
    return gdr_bwd_wy_matmuls_kernel, [
        GMTensor(f, [B, H, N, C, C], name="attn"),
        GMTensor(f, [B, H, N, C, K], name="k_f32"),
        GMTensor(f, [B, H, N, C, V], name="v_b"),
        GMTensor(f, [B, H, N, K, C], name="kbeta_t"),
        GMTensor(f, [B, H, N, C], name="beta_act"),
        GMTensor(f, [B, H, N, C], name="decay"),
        GMTensor(f, [B, H, N, C, K], name="dw"),
        GMTensor(f, [B, H, N, C, V], name="dv2"),
        GMTensor(f, [B, H, N, C, C], name="dA_gm"),
        GMTensor(f, [B, H, N, C, K], name="kbeta_gm"),
        GMTensor(f, [B, H, N, C, K], name="dkbg"),
        GMTensor(f, [B, H, N, C, V], name="dvb"),
        GMTensor(f, [B, H, N, C, K], name="dkb"),
        GMTensor(f, [B, H, N, C, K], name="dkt"),
        GMTensor(f, [B, H, N, C, C], name="d_a"),
        Var(B, name="B"), Var(H, name="H"), Var(N, name="N"),
        Var(K, name="Kd"), Var(V, name="Vd"), Var(C, name="Cc"),
    ]
