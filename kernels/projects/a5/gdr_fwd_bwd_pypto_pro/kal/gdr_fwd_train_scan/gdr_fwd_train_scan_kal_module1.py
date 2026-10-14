# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# -----------------------------------------------------------------------------
# PyPTO-Pro Stage 3.5 (KAL pre-build) -- gdr_fwd_train_scan  (OP-B of the
# gated_delta_rule TRAINING-FORWARD pair).  Single fusion module (module_count=1,
# is_fusion=true; DESIGN sec.0):  the M5 inter-chunk scan, emitting the public
# forward output `o` PLUS the per-chunk ENTERING state history `s_hist`.
#
# =============================== RE-HOMING ===================================
# This kernel is a VERBATIM re-home of the read-only lean forward's transpile-
# clean standalone M5 scan
#     custom/gated_delta_rule_fwd/kal/staged_scan_module5.py  (KAL source:
#     tmp/kal_module12345/scan_probe.py) -- 248,064 B / 25 mutex / 0 flags,
#     sim-VERIFIED (o) + transpile-CLEAN.
# Exactly ONE thing is ADDED: the `s_hist` entering-state store -- a store-side
# snapshot of the RESIDENT state tile (`state_ub`) at the chunk-loop top, taken
# BEFORE the s15 state update, sub-block-0 guarded.  ni=0 -> state_ub is the
# freshly-zeroed S -> s_hist[:,:,0]=0.  The snapshot introduces NO new tile group
# (it reuses the resident S tile) and NO arithmetic, so it costs EXACTLY 0 UB.
# The store technique is the one proven in the delivered backward's reverse scan
# (custom/gated_delta_rule_bwd/kal/gated_delta_rule_bwd_kal_module12345.py L217 /
# staged_module12345.py L191-195).
#
# Everything else is the seed's code, unchanged:  the o-branch (v_prime=w@S,
# v_new=u-v_prime, o_inter=q_de@S, o=o_inter+A@v_new, bf16 cast+store), the tile
# declarations, the 4-handoff delta_h section ordering, the cross-core /
# auto_mutex handoffs, the matmul routing (Fix A GM-routed operand bridging +
# Fix B pre-transposed k_f32), and the vf bodies.  See NOTES.md for the diff.
#
# Authored against the easyasc a5 facade; validated in the OpExec simulator;
# backend-transpiled to PyPTO-Pro DSL (kal/staged_module1.py).  No PyPTO /
# PyPTO-Pro API is hand-written.  Simulator pass != Stage-4 acceptance; this
# carries no certification.
# -----------------------------------------------------------------------------
"""gdr_fwd_train_scan -- M5 inter-chunk scan emitting o + s_hist.

Reads OP-A's M1-M4 handoffs (u,w,A,q_de,decay_exp,dfac,k_f32) as GM inputs
(computed here by the CPU golden's _project_m1_m4), runs the inter-chunk scan
with loop-carried state S[K,V] persistent in UB, and produces:
  o      [B,T,H,V] bf16   -- the public forward output (identical math to the
                            read-only lean forward; o[:, :T] reproduces it).
  s_hist [B,H,N,K,V] fp32 -- per-chunk ENTERING state S_i (snapshot BEFORE the
                            chunk-i s15 update; s_hist[:,:,0]=0).

o_i = q_de_i @ S + A_i @ v_new  computed as ONE L0C accumulator (is_init then +=).
"""
import os
import sys
import importlib.util

for _cand in (
    "/root/.claude/skills/pypto-kernel-arch-lang/payload",
    "/workspace/pypto-gym/cannbot-skills/ops/pypto-kernel-arch-lang/payload",
):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

import torch
from easyasc.a5 import *  # noqa
from easyasc.simulator import SimulatorConfig

C = 64
TILE = 128
SUB = 64
SIM_CORE_COUNT = 32


@vf()
def scan_sub_vf(ub_vprime: Tensor, ub_uv: Tensor, ub_dfac: Tensor, ub_vscaled: Tensor):
    """v_new = u - v_prime (in place into ub_uv); v_scaled = v_new * factor[r].

    Full C rows, full TILE cols (pad cols of u/v_prime are 0 -> v_new pad 0).
    factor[r] = dfac[r,0].
    """
    u_regs = RegList(DT.float, TILE // SUB)
    vp_regs = RegList(DT.float, TILE // SUB)
    vn_regs = RegList(DT.float, TILE // SUB)
    vs_regs = RegList(DT.float, TILE // SUB)
    fac = Reg(DT.float)
    for r in range(C, name="ssr"):
        fac <<= ub_dfac[r:r + 1, 0:1].single()
        u_regs <<= ub_uv[r:r + 1, 0:TILE]
        vp_regs <<= ub_vprime[r:r + 1, 0:TILE]
        vn_regs <<= u_regs - vp_regs
        vs_regs <<= vn_regs * fac
        ub_uv[r:r + 1, 0:TILE] <<= vn_regs
        ub_vscaled[r:r + 1, 0:TILE] <<= vs_regs


@vf()
def o_cast_vf(ub_o: Tensor, ub_obf: Tensor, vc: Var, nvsub: Var):
    """cast o (fp32) -> bf16 -> ub_obf, valid rows/cols."""
    cfg = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO, reg_layout=RegLayout.ZERO, name="o_bf16")
    for r in range(vc, name="oar"):
        for cb in range(nvsub, name="oacb"):
            col0 = Var(cb * SUB)
            s = Reg(DT.float)
            bf = Reg(DT.bfloat16)
            s <<= ub_o[r:r + 1, col0:col0 + SUB]
            cast(bf, s, cfg)
            reg_to_ub_downsample(ub_obf[r:r + 1, col0:col0 + SUB], bf)


@vf()
def state_update_vf(ub_state: Tensor, ub_delta: Tensor, ub_decay_exp: Tensor, ck: Var):
    """S = S * decay_exp[C-1] + delta  (in-place UB, K rows)."""
    dreg = Reg(DT.float)
    state_regs = RegList(DT.float, TILE // SUB)
    delta_regs = RegList(DT.float, TILE // SUB)
    dreg <<= ub_decay_exp[0:1, (C - 1):C].single()
    for r in range(ck, name="sur"):
        state_regs <<= ub_state[r:r + 1, 0:TILE]
        delta_regs <<= ub_delta[r:r + 1, 0:TILE]
        state_regs <<= state_regs * dreg
        state_regs <<= state_regs + delta_regs
        ub_state[r:r + 1, 0:SUB] <<= state_regs[0]
        vf_barrier(VfPipe.STORE, VfPipe.STORE)
        ub_state[r:r + 1, SUB:TILE] <<= state_regs[1]
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)
        vf_barrier(VfPipe.STORE, VfPipe.STORE)


@vf()
def zero_ub_vf(ub_t: Tensor, nrows: Var):
    """Zero-fill ub_t[0:nrows, 0:TILE]."""
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    z = Reg(DT.float)
    dup(z, 0.0, mask=active)
    for r in range(nrows, name="zr"):
        for cb in range(TILE // SUB, name="zcb"):
            col0 = cb * SUB
            reg_to_ub_normal(ub_t[r:r + 1, col0:col0 + SUB], z, mask=active)


@kernel()
def gdr_fwd_train_scan_kernel(
    u: GMTensor, w: GMTensor, A: GMTensor, q_de: GMTensor,
    decay_exp: GMTensor, dfac: GMTensor, k_f32_T: GMTensor,
    state_gm: GMTensor, vnew_gm: GMTensor, vscaled_gm: GMTensor,
    o: GMTensor, s_hist: GMTensor, v_new: GMTensor,
    B: Var, T: Var, H: Var, Kd: Var, Vd: Var, N: Var, Cc: Var, VT: Var,
):
    # Vec->cube handoffs route their matmul operands through GM workspace
    # (UB->GM store on vec, GM->L1 nd2nz load on cube) instead of an on-chip
    # UB->L1 bridge, which materialises same-size NZ-staging copies in UB and
    # overflows the 253952 B budget (module1234 finding #5). Proven pattern:
    # producer ends on MTE3 (GM store), consumer ends on FIX (GM->L1 + matmul).
    state_bridge = VcMutex(0, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    stage1_out = CvMutex(1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    vscaled_bridge = VcMutex(2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    stage2_out = CvMutex(3, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

    state_ub = Tensor(DT.float, [TILE, TILE], Position.UB)   # persistent S
    ub_delta = Tensor(DT.float, [TILE, TILE], Position.UB)   # Sdelta
    ub_vp = Tensor(DT.float, [C, TILE], Position.UB)         # vprime -> o
    ub_uv = Tensor(DT.float, [C, TILE], Position.UB)         # u -> vnew
    ub_vs = Tensor(DT.float, [C, TILE], Position.UB)         # vscaled
    ub_dfac = Tensor(DT.float, [C, 8], Position.UB)          # factor col
    ub_obf = Tensor(DT.bfloat16, [C, TILE], Position.UB)     # o bf16
    ub_dexp = Tensor(DT.float, [1, C], Position.UB)

    state_l1 = Tensor(DT.float, [TILE, TILE], Position.L1)
    w_l1 = Tensor(DT.float, [C, TILE], Position.L1)
    qde_l1 = Tensor(DT.float, [C, TILE], Position.L1)
    A_l1 = Tensor(DT.float, [C, C], Position.L1)
    vnew_l1 = Tensor(DT.float, [C, TILE], Position.L1)
    vscaled_l1 = Tensor(DT.float, [C, TILE], Position.L1)
    # k_f32 is pre-transposed in GM (k_f32_T [TILE, C] = k_f32^T), so the delta
    # matmul's LEFT operand is loaded NON-transposed into L0A (fixes the
    # `src_is_transpose into L0A` transpile_gap: only the RIGHT/L0B operand may
    # carry an L1-side .T, as in module1234's M4). Same bytes as the old [C,TILE].
    kfT_l1 = Tensor(DT.float, [TILE, C], Position.L1)

    vprime_l0c = Tensor(DT.float, [C, TILE], Position.L0C)
    o_l0c = Tensor(DT.float, [C, TILE], Position.L0C)
    delta_l0c = Tensor(DT.float, [TILE, TILE], Position.L0C)

    tasks = B * H
    per = CeilDiv(tasks, GetCubeNum())
    begin = Var(per * GetCubeIdx())
    end = Min(begin + per, tasks)
    nvsub_v = CeilDiv(Vd, SUB)

    with auto_sync():
        for bh in range(begin, end, name="bh"):
            b_idx = Var(bh // H)
            h_idx = Var(bh % H)
            row_start = Var(GetSubBlockIdx() * C)   # 0 on subblock 0, C on subblock 1
            if row_start < C:
                zero_ub_vf(state_ub, Kd)            # S = 0 (subblock 0 owns state)
            bar_all()

            for n_idx in range(N, name="ni"):
                t0 = Var(n_idx * C)
                vc = Min(T - t0, C)

                # SA: snapshot ENTERING state -> s_hist (store-side, BEFORE the
                # s15 update; ni=0 -> zeroed state_ub -> s_hist[:,:,0]=0). This
                # reuses the RESIDENT S tile (state_ub) -> no new tile group, no
                # arithmetic, EXACTLY 0 UB. Then vec stages S UB->GM; cube loads
                # S GM->L1 (nd2nz), then v_prime=w@S, o=q_de@S (init accumulator).
                # S's loop-carry stays a persistent UB tile (state_ub); state_gm
                # is a per-chunk copy that only supplies the matmul operand (no
                # UB->L1 NZ staging).
                state_bridge.lock()
                if row_start < C:
                    s_hist[b_idx, h_idx, n_idx, 0:Kd, 0:Vd] <<= state_ub[0:Kd, 0:Vd]
                    state_gm[b_idx, h_idx, 0:Kd, 0:TILE] <<= state_ub[0:Kd, 0:TILE]
                state_bridge.ready()

                w_l1[0:C, 0:Kd] <<= w[b_idx, h_idx, n_idx, 0:C, 0:Kd]
                qde_l1[0:C, 0:Kd] <<= q_de[b_idx, h_idx, n_idx, 0:C, 0:Kd]
                state_bridge.wait()
                state_l1[0:Kd, 0:TILE] <<= state_gm[b_idx, h_idx, 0:Kd, 0:TILE]
                # splitn=SUB keeps each transposed S tile <= [SUB, TILE] within the
                # 8192-fp32 L0B transpose budget (S.T is [TILE, Kd], up to [128,128]).
                # S is the RIGHT/L0B operand (.T is an L1->L0B transpose, clean).
                matmul(vprime_l0c, w_l1[0:C, 0:Kd], state_l1[0:Kd, 0:TILE].T, splitn=SUB, k=Kd)
                matmul(o_l0c, qde_l1[0:C, 0:Kd], state_l1[0:Kd, 0:TILE].T, splitn=SUB, k=Kd, is_init=True)
                state_bridge.free()

                # stage1: cube v_prime -> UB; vec v_new, v_scaled
                stage1_out.lock()
                ub_vp[0:C, 0:TILE] <<= vprime_l0c.subblk(0)
                stage1_out.ready()

                if row_start < C:
                    zero_ub_vf(ub_uv, C)
                    ub_uv[0:C, 0:Vd] <<= u[b_idx, h_idx, n_idx, 0:C, 0:Vd]
                    ub_dfac[0:C, 0:8] <<= dfac[b_idx, h_idx, n_idx, 0:C, 0:8]
                stage1_out.wait()
                if row_start < C:
                    scan_sub_vf(ub_vp, ub_uv, ub_dfac, ub_vs)   # ub_uv now holds v_new
                stage1_out.free()

                # vscaled_bridge: vec stages v_new/v_scaled UB->GM; decay S; cube
                # loads them GM->L1 (nd2nz), then o+=A@vnew, delta=k_f32^T@v_scaled.
                vscaled_bridge.lock()
                if row_start < C:
                    # SB: promote the per-chunk delta-corrected value v_new
                    # (= u_i - w_i@S = ub_uv, RESIDENT after scan_sub_vf) to the
                    # public GM output, store-side, sub-block-0 guarded, BEFORE it
                    # is consumed by the cube (o += A@v_new) and the s15 state
                    # update. Mirrors the s_hist store exactly: reuses the resident
                    # ub_uv tile -> no new tile group, no arithmetic, EXACTLY 0 UB.
                    # ni=0 -> S=0 -> v_prime=0 -> v_new[:,:,0]=u_0 (golden Path-A).
                    v_new[b_idx, h_idx, n_idx, 0:C, 0:Vd] <<= ub_uv[0:C, 0:Vd]
                    vnew_gm[b_idx, h_idx, 0:C, 0:TILE] <<= ub_uv[0:C, 0:TILE]
                    vscaled_gm[b_idx, h_idx, 0:C, 0:TILE] <<= ub_vs[0:C, 0:TILE]
                vscaled_bridge.ready()

                if row_start < C:
                    ub_dexp[0:1, 0:C] <<= decay_exp[b_idx, h_idx, n_idx, 0:C]
                A_l1[0:C, 0:C] <<= A[b_idx, h_idx, n_idx, 0:C, 0:C]
                # k_f32_T [TILE, C] = k_f32^T, pre-transposed at the GM->L1 load so
                # the delta matmul's LEFT operand is NON-transposed (fills L0A from a
                # plain L1 tile). k_f32 pad cols [Kd:TILE] are 0 -> k_f32_T pad rows
                # [Kd:TILE] are 0 -> delta rows [Kd:TILE] = 0 (full-tile m=TILE; the
                # transposed axis is not partially sliced, module12345 finding #4).
                kfT_l1[0:TILE, 0:C] <<= k_f32_T[b_idx, h_idx, n_idx, 0:TILE, 0:C]
                vscaled_bridge.wait()
                vnew_l1[0:C, 0:TILE] <<= vnew_gm[b_idx, h_idx, 0:C, 0:TILE]
                vscaled_l1[0:C, 0:TILE] <<= vscaled_gm[b_idx, h_idx, 0:C, 0:TILE]
                matmul(o_l0c, A_l1[0:C, 0:C], vnew_l1[0:C, 0:TILE].T, splitn=SUB, k=C, is_init=False)
                # delta = k_f32^T @ v_scaled: LEFT = kfT_l1 [M=TILE,K=C] non-transposed
                # (L0A), RIGHT = v_scaled with the L1-side .T (L0B transpose, clean).
                matmul(delta_l0c, kfT_l1[0:TILE, 0:C], vscaled_l1[0:C, 0:TILE].T, m=TILE, n=TILE, k=C)
                vscaled_bridge.free()

                # stage2: cube o/delta -> UB; vec o store, S += delta
                stage2_out.lock()
                ub_vp[0:C, 0:TILE] <<= o_l0c.subblk(0)          # o -> ub_vp
                ub_delta[0:Kd, 0:TILE] <<= delta_l0c.subblk(0)
                stage2_out.ready()

                stage2_out.wait()
                if row_start < C:
                    o_cast_vf(ub_vp, ub_obf, vc, nvsub_v)
                    o[b_idx, t0:t0 + vc, h_idx, 0:Vd] <<= ub_obf[0:vc, 0:Vd]
                    state_update_vf(state_ub, ub_delta, ub_dexp, Kd)
                stage2_out.free()

    return o, s_hist, v_new


# axis ids from trailing scalars (B,T,H,Kd,Vd,N,Cc,VT):
#   0=B 1=T 2=H 3=Kd 4=Vd 5=N 6=Cc(=C) 7=VT(=TILE)
_SHAPE_BINDINGS = {
    0: [0, 2, 5, 6, 4],   # u          [B,H,N,C,V]
    1: [0, 2, 5, 6, 3],   # w          [B,H,N,C,K]
    2: [0, 2, 5, 6, 6],   # A          [B,H,N,C,C]
    3: [0, 2, 5, 6, 3],   # q_de       [B,H,N,C,K]
    4: [0, 2, 5, 6],      # decay_exp  [B,H,N,C]
    5: [0, 2, 5, 6, 3],   # dfac       [B,H,N,C,K]
    6: [0, 2, 5, 7, 6],   # k_f32_T    [B,H,N,TILE,C]  (pre-transposed, padded)
    7: [0, 2, 7, 7],      # state_gm   [B,H,TILE,TILE]
    8: [0, 2, 6, 7],      # vnew_gm    [B,H,C,TILE]
    9: [0, 2, 6, 7],      # vscaled_gm [B,H,C,TILE]
    10: [0, 1, 2, 4],     # o          [B,T,H,V]
    11: [0, 2, 5, 3, 4],  # s_hist     [B,H,N,K,V]
    12: [0, 2, 5, 6, 4],  # v_new      [B,H,N,C,V]  (== u binding)
}


def _load_golden():
    gp = ("/workspace/pypto-gym/custom/gdr_fwd_train_scan/"
          "gdr_fwd_train_scan_golden_cpu.py")
    spec = importlib.util.spec_from_file_location("gdr_fwd_train_scan_golden_cpu", gp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_lean_fwd_golden():
    """Read-only lean forward CPU golden (o reference; NOT modified/copied)."""
    gp = ("/workspace/pypto-gym/custom/gated_delta_rule_fwd/"
          "gated_delta_rule_fwd_golden_cpu.py")
    spec = importlib.util.spec_from_file_location("_gdr_lean_fwd_golden_cpu", gp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mk_primary(B, T, H, K, V, seed):
    torch.manual_seed(seed)
    q = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    k = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    g = torch.empty((B, T, H), dtype=torch.float32).uniform_(-4.0, 0.0)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16)
    return q, k, v, g, beta


def _pad_kf(k_f32, K):
    # pad k_f32 [B,H,N,C,K] -> [B,H,N,C,TILE] with zeros beyond K
    B, H, N, Cc, _ = k_f32.shape
    kp = torch.zeros((B, H, N, Cc, TILE), dtype=k_f32.dtype)
    kp[..., :K] = k_f32
    return kp.contiguous()


def run_scan(u, w, A, q_de, decay_exp, dfac, k_f32, B, T, H, K, V, N):
    # k_f32_T = (padded k_f32)^T -> [B,H,N,TILE,C]. Pad+transpose are pure layout
    # (no arithmetic); in the fused kernel Module 1 stores k_f32 transposed. This
    # lets the delta matmul read the left operand non-transposed from L0A.
    k_f32_T = _pad_kf(k_f32, K).transpose(-1, -2).contiguous()
    o = torch.zeros((B, T, H, V), dtype=torch.bfloat16)
    s_hist = torch.zeros((B, H, N, K, V), dtype=torch.float32)
    v_new = torch.zeros((B, H, N, C, V), dtype=torch.float32)   # per-chunk delta-corrected value
    # GM workspace for the vec->cube operand bridges (per-(b,h); distinct regions,
    # no cross-core aliasing; single-buffered, reused each serial chunk).
    state_gm = torch.zeros((B, H, TILE, TILE), dtype=torch.float32)
    vnew_gm = torch.zeros((B, H, C, TILE), dtype=torch.float32)
    vscaled_gm = torch.zeros((B, H, C, TILE), dtype=torch.float32)
    gdr_fwd_train_scan_kernel._simulator_config = SimulatorConfig(
        core_count=SIM_CORE_COUNT, execution_timeout_s=1800.0)
    return OpExec(gdr_fwd_train_scan_kernel, simulator=True)(
        u, w, A, q_de, decay_exp, dfac, k_f32_T,
        state_gm, vnew_gm, vscaled_gm, o, s_hist, v_new,
        B, T, H, K, V, N, C, TILE,
        shape_bindings=_SHAPE_BINDINGS,
    )


def _mixed_ok(got, exp, l2_tol=2e-2):
    gf = got.float()
    ef = exp.float()
    peak = ef.abs().max().item()
    atol = 1e-2 * max(peak, 1e-6)
    ew = torch.allclose(gf, ef, rtol=1e-2, atol=atol)
    rel_l2 = ((gf - ef).norm() / (ef.norm() + 1e-12)).item()
    max_abs = (gf - ef).abs().max().item()
    return (ew or rel_l2 <= l2_tol), max_abs, rel_l2, peak


if __name__ == "__main__":
    golden = _load_golden()
    lean = _load_lean_fwd_golden()
    shapes = [(1, 128, 4, 64, 64), (1, 100, 4, 64, 64),
              (1, 192, 2, 128, 128), (1, 200, 3, 128, 96)]
    seeds = [42, 123, 456]
    all_pass = True
    for (B, T, H, K, V) in shapes:
        for seed in seeds:
            q, k, v, g, beta = _mk_primary(B, T, H, K, V, seed)
            # OP-A M1-M4 projection -> the 7 handoffs (order: u,w,q_de,A,k_f32,dfac,decay_exp)
            u, w, q_de, A, k_f32, dfac, decay_exp = golden._project_m1_m4(q, k, v, g, beta)
            u = u.contiguous().float()
            w = w.contiguous().float()
            q_de = q_de.contiguous().float()
            A = A.contiguous().float()
            k_f32 = k_f32.contiguous().float()
            dfac = dfac.contiguous().float()
            decay_exp = decay_exp.contiguous().float()
            N = u.shape[2]
            # oracle: (o_pad [B,64N,H,V], s_hist [B,H,N,K,V], v_new [B,H,N,C,V])
            o_ref_pad, s_hist_ref, v_new_ref = golden.gdr_fwd_train_scan_golden_cpu(
                u, w, q_de, A, k_f32, dfac, decay_exp)
            o_ref = o_ref_pad[:, :T]                    # public [B,T,H,V]
            # cross-check: read-only lean forward o (bit-identical to o_ref by construction)
            o_lean = lean.gated_delta_rule_fwd_golden_cpu(q, k, v, g, beta)  # [B,T,H,V]
            out = run_scan(u, w, A, q_de, decay_exp, dfac, k_f32, B, T, H, K, V, N)
            o_out, s_hist_out, v_new_out = out
            ok_o, mabs_o, rl2_o, peak_o = _mixed_ok(o_out, o_ref)
            ok_lean, mabs_l, rl2_l, peak_l = _mixed_ok(o_out, o_lean)
            ok_sh, mabs_s, rl2_s, peak_s = _mixed_ok(s_hist_out, s_hist_ref)
            ok_vn, mabs_v, rl2_v, peak_v = _mixed_ok(v_new_out, v_new_ref)
            fin = (torch.isfinite(o_out.float()).all().item()
                   and torch.isfinite(s_hist_out.float()).all().item()
                   and torch.isfinite(v_new_out.float()).all().item())
            sh0_zero = (s_hist_out[:, :, 0].abs().max().item() == 0.0)
            # v_new[:,:,0] == u_0 (entering S=0 -> v_prime=0 -> v_new=u), golden Path-A
            vn0_ok, vn0_max, _, _ = _mixed_ok(v_new_out[:, :, 0], u[:, :, 0, :, :V])
            case_ok = ok_o and ok_lean and ok_sh and ok_vn and fin and sh0_zero and vn0_ok
            print(f"  [{'PASS' if case_ok else 'FAIL'}] B{B}T{T}H{H}K{K}V{V} seed{seed} "
                  f"| o: rl2={rl2_o:.3e} max={mabs_o:.2e} peak={peak_o:.2e} "
                  f"| o-vs-lean: rl2={rl2_l:.3e} "
                  f"| s_hist: rl2={rl2_s:.3e} max={mabs_s:.2e} "
                  f"| v_new: rl2={rl2_v:.3e} max={mabs_v:.2e} peak={peak_v:.2e} "
                  f"| v_new0==u0: {vn0_ok} | s_hist[:,:,0]==0: {sh0_zero} | finite: {fin}")
            if not case_ok:
                all_pass = False
    print("ALL PASS" if all_pass else "FAILURES PRESENT")
    sys.exit(0 if all_pass else 1)
