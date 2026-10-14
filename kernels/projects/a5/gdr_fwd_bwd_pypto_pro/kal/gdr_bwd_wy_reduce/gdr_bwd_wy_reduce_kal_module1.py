# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# -----------------------------------------------------------------------------
# PyPTO-Pro Stage 3.5 (KAL pre-build) -- gdr_bwd_wy_reduce (OP6b), single Module.
#
# OP6b = the PURE-VECTOR half of module M9 (golden_step b4 reductions + b4b_dk)
# of gated_delta_rule_bwd.  BHN-parallel, stateless, single Vector Section, no
# cube / no matmul / no cross_core.  Consumes OP6a's 4 adjoint products
# (dkbg,dvb,dkb,dkt), the train-forward residuals (k_f32,beta_act,decay_exp),
# OP5's dk_p, and the PRIMARY v[B,T,H,V] bf16 (self-served to v_c fp32); emits
# 4 fp32 outputs (dk,dv,dbeta_wy,dg2).
#
# Authored against the easyasc `a5` facade (arch A5), validated in the Python
# OpExec simulator against gdr_bwd_wy_reduce_golden_cpu.py, then transpiled to
# PyPTO-Pro DSL by the backend.  No PyPTO / PyPTO-Pro API is used here; the
# transpile is performed by the backend, not hand-written.
#
# =====================================================================
# SCALE LEDGER (DESIGN "THE SCALE LEDGER"; the golden catches any violation)
#   scale=1/sqrt(K):  applied 0x (folded upstream).  NEVER introduced here.
#   dkb / dkt:        arrive ALREADY decay-gated from OP6a -> NO re-gate here.
#   beta_act (b):     5 paths, each EXACTLY once --
#       dv = dvb*b (S9) ; dk2:dkb*b (S8) ; dk_term1 gate decay*b (S7) ;
#       dg2_loop1 = b*db1 (S3) ; dg2 += b*db3 (S10).  dbeta_wy carries ZERO b.
#   decay_exp (d):    3 paths, each EXACTLY once --
#       dk_term1 gate decay*b (S7) ; db1 = d*inner_dg (S2) ;
#       dg2_loop1 via db1 (S3, decay NOT re-multiplied).
#   inner_dg = sum_K(dkbg*k) and db3 = sum_K(dkb*k) are each reduced ONCE and
#   reused (db3 by BOTH dbeta_wy and dg2).  Per-token constants (b[c],d[c]) are
#   folded OUT of the reductions and applied ONCE to the lane-0 scalar
#   (sum_K(x*c) = c*sum_K(x)) -- a structural single-apply.
#
# TWO-PASS UB ordering (DESIGN §3; adopted because the fused per-c loop measured
# 97.4% UB > 90%):
#   Pass-K: dkbg,dkb,dkt,k,dk_p -> dk (in-place in the dk_p tile), dg2 (final),
#           and dbeta_partial = db1+db3 (K-domain part of dbeta_wy).
#   Pass-V: dvb, v(bf16) -> v_c ; dv (in-place in the dvb tile), db2, and the
#           final dbeta_wy = dbeta_partial + db2.
#   Pass-V REUSES the Pass-K tile addresses (t_dvb aliases the dkbg tile, t_vc
#   aliases the dkb tile) so the peak UB is the per-pass max, not the sum.
#
# TILE POLICY (DESIGN §2): every on-chip tile is a compile-time [64,128] /
# [1,128] / [1,64] literal (C=64, MAX_K=MAX_V=128).  Dynamic extents (K/V/chunk)
# live only in loop bounds and GM offsets, never in a tile shape.
# -----------------------------------------------------------------------------
import os
import sys

for _cand in (
    "/workspace/pypto-gym/cannbot-skills/ops/pypto-kernel-arch-lang/payload",
    "/root/.claude/skills/pypto-kernel-arch-lang/payload",
):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

import torch  # noqa: E402

from easyasc.a5 import *  # noqa: F401,F403,E402
from easyasc.simulator import SimulatorConfig  # noqa: E402


# ---- operator-frozen constants (DESIGN §2.1) ----
C = 64             # fixed chunk size = fp32 Reg lane count
LANES = 64         # fp32 VF register width
MAX_K = 128        # K compile-time bound (SPEC: K<=128)
MAX_V = 128        # V compile-time bound (SPEC: V<=128)
COL_PAD = 128      # per-token column tile over-declaration width
SIM_CORE_COUNT = 32


# =====================================================================
# Pass-K vector function: K-domain reductions + dk assembly + dg2 (final) +
# dbeta_partial (= db1 + db3).  Processes ALL C rows (chunk-domain; padded rows
# use the upstream-padded intermediate values, exactly as the golden does).
# =====================================================================
@vf()
def passk_vf(t_dkbg: Tensor, t_dkb: Tensor, t_dkt: Tensor, t_k: Tensor,
             t_dkp: Tensor, ub_beta: Tensor, ub_decay: Tensor,
             ub_dbeta: Tensor, ub_dg2: Tensor, Kd: Var):
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    n_regs = CeilDiv(Kd, LANES)
    for c in range(C, name="pk_c"):
        # per-token constants (broadcast to all lanes)
        bbc = Reg(DT.float)
        dbc = Reg(DT.float)
        gde = Reg(DT.float)
        bbc <<= ub_beta[0:1, c:c + 1].single()
        dbc <<= ub_decay[0:1, c:c + 1].single()
        mul(gde, dbc, bbc, mask=active)              # gde = decay_exp[c]*beta[c]

        inner = Reg(DT.float)                        # inner_dg = sum_K(dkbg*k)
        db3 = Reg(DT.float)                          # db3      = sum_K(dkb*k)
        kdkt = Reg(DT.float)                         # kdkt     = sum_K(k*dkt)
        dup(inner, 0.0, mask=active)
        dup(db3, 0.0, mask=active)
        dup(kdkt, 0.0, mask=active)

        for r in range(n_regs, name="pk_r"):
            off = Var(r * LANES)
            valid = Min(LANES, Kd - off)
            mred = MaskReg(DT.float, init_mode=MaskType.ALL)
            update_mask(mred, Var(valid, dtype=DT.uint32))

            a = Reg(DT.float)                        # dkbg row-slice
            kk = Reg(DT.float)                       # k_f32 row-slice
            bb = Reg(DT.float)                       # dkb  row-slice
            tt = Reg(DT.float)                       # dkt  row-slice
            a <<= t_dkbg[c:c + 1, off:off + LANES]
            kk <<= t_k[c:c + 1, off:off + LANES]
            bb <<= t_dkb[c:c + 1, off:off + LANES]
            tt <<= t_dkt[c:c + 1, off:off + LANES]

            prod = Reg(DT.float)
            part = Reg(DT.float)
            mul(prod, a, kk, mask=mred)
            cadd(part, prod, mask=mred)
            add(inner, inner, part, mask=active)     # accumulate lane0 across halves
            mul(prod, bb, kk, mask=mred)
            cadd(part, prod, mask=mred)
            add(db3, db3, part, mask=active)
            mul(prod, kk, tt, mask=mred)
            cadd(part, prod, mask=mred)
            add(kdkt, kdkt, part, mask=active)

            # ---- dk assembly for this K register (SCALE LEDGER S7/S8/S12) ----
            dkp = Reg(DT.float)
            dk2 = Reg(DT.float)
            dkt1 = Reg(DT.float)
            dkp <<= t_dkp[c:c + 1, off:off + LANES]
            mul(dkt1, a, gde, mask=active)           # dk_term1 = dkbg*(decay*beta)
            mul(dk2, bb, bbc, mask=active)           # dkb*beta
            add(dk2, dk2, tt, mask=active)           # + dkt (raw, already decay-gated)
            add(dk2, dk2, dkt1, mask=active)         # + dk_term1
            add(dk2, dk2, dkp, mask=active)          # dk = dk_p + dk2
            t_dkp[c:c + 1, off:off + valid] <<= dk2  # in-place store dk

        # ---- per-token scalar combine (lane0) ----
        db1 = Reg(DT.float)
        dg2l = Reg(DT.float)
        bd3 = Reg(DT.float)
        mul(db1, inner, dbc, mask=active)            # db1 = decay*inner_dg (S2)
        mul(dg2l, db1, bbc, mask=active)             # dg2_loop1 = beta*db1 (S3)
        mul(bd3, db3, bbc, mask=active)              # beta*db3 (S10)
        dg2c = Reg(DT.float)
        add(dg2c, dg2l, bd3, mask=active)
        sub(dg2c, dg2c, kdkt, mask=active)           # dg2 = dg2_loop1 + beta*db3 - kdkt
        ub_dg2[0:1, c:c + 1] <<= dg2c.single_value()

        dbp = Reg(DT.float)
        add(dbp, db1, db3, mask=active)              # dbeta_partial = db1 + db3
        ub_dbeta[0:1, c:c + 1] <<= dbp.single_value()


# =====================================================================
# Pass-V vector function: v_c self-serve (bf16->fp32 widen, pad rows zeroed),
# db2 = sum_V(dvb*v_c) (RAW v_c), dv = dvb*beta (in-place), and the final
# dbeta_wy = dbeta_partial + db2.
# =====================================================================
@vf()
def passv_vf(t_dvb: Tensor, t_vc: Tensor, t_vbf: Tensor, ub_beta: Tensor,
             ub_dbeta: Tensor, Vd: Var, vc_rows: Var):
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    to_float = CastConfig(round_mode=RoundMode.NONE, reg_layout=RegLayout.ZERO,
                          name="v_b2f")
    z = Reg(DT.float)
    dup(z, 0.0, mask=active)
    n_regs = CeilDiv(Vd, LANES)

    # ---- S0: v_c = fp32(v) for valid rows (the ONLY cast) ----
    for c in range(vc_rows, name="pv_cast"):
        for r in range(n_regs, name="pv_cr"):
            off = Var(r * LANES)
            valid = Min(LANES, Vd - off)
            vb = Reg(DT.bfloat16)
            vfp = Reg(DT.float)
            ub_to_reg_unpack(vb, t_vbf[c:c + 1, off:off + LANES])
            cast(vfp, vb, to_float, mask=active)
            t_vc[c:c + 1, off:off + valid] <<= vfp
    # ---- pad token rows are inert (db2 contribution 0); zero their v_c ----
    for c in range(vc_rows, C, name="pv_pad"):
        for r in range(n_regs, name="pv_pr"):
            off = Var(r * LANES)
            valid = Min(LANES, Vd - off)
            t_vc[c:c + 1, off:off + valid] <<= z

    # Order the v_c STOREs (and the Pass-K dbeta_partial STOREs) before the
    # db2 phase LOADs: the VF store buffer is its own ordering domain, so a
    # store->load on overlapping UB races on real HW without this barrier.
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

    # ---- db2 + dv per token (all C rows) ----
    for c in range(C, name="pv_c"):
        bbc = Reg(DT.float)
        bbc <<= ub_beta[0:1, c:c + 1].single()
        db2 = Reg(DT.float)
        dup(db2, 0.0, mask=active)
        for r in range(n_regs, name="pv_r"):
            off = Var(r * LANES)
            valid = Min(LANES, Vd - off)
            mred = MaskReg(DT.float, init_mode=MaskType.ALL)
            update_mask(mred, Var(valid, dtype=DT.uint32))
            dvb = Reg(DT.float)
            vcr = Reg(DT.float)
            prod = Reg(DT.float)
            part = Reg(DT.float)
            dvb <<= t_dvb[c:c + 1, off:off + LANES]
            vcr <<= t_vc[c:c + 1, off:off + LANES]
            mul(prod, dvb, vcr, mask=mred)           # RAW v_c; no beta/decay
            cadd(part, prod, mask=mred)
            add(db2, db2, part, mask=active)
            dvout = Reg(DT.float)
            mul(dvout, dvb, bbc, mask=active)        # dv = dvb*beta (S9)
            t_dvb[c:c + 1, off:off + valid] <<= dvout  # in-place store dv

        dbp = Reg(DT.float)
        dbeta = Reg(DT.float)
        dbp <<= ub_dbeta[0:1, c:c + 1].single()      # dbeta_partial (db1+db3)
        add(dbeta, dbp, db2, mask=active)            # dbeta_wy = db1+db3+db2
        ub_dbeta[0:1, c:c + 1] <<= dbeta.single_value()


# =====================================================================
# Kernel (single Vector Section, BHN-parallel flat task loop)
# =====================================================================
@kernel(mode="vec")
def gdr_bwd_wy_reduce_kernel(
    dkbg: GMTensor, dvb: GMTensor, dkb: GMTensor, dkt: GMTensor,
    k_f32: GMTensor, beta_act: GMTensor, decay_exp: GMTensor, dk_p: GMTensor,
    v: GMTensor,
    dk: GMTensor, dv: GMTensor, dbeta_wy: GMTensor, dg2: GMTensor,
    B: Var, T: Var, H: Var, Kd: Var, Vd: Var, N: Var, Cc: Var,
):
    # Pass-K working set (5 K-tiles); the dk_p tile receives dk in-place.
    t_dkbg = Tensor(DT.float, [C, MAX_K], Position.UB)
    t_dkb = Tensor(DT.float, [C, MAX_K], Position.UB)
    t_dkt = Tensor(DT.float, [C, MAX_K], Position.UB)
    t_k = Tensor(DT.float, [C, MAX_K], Position.UB)
    t_dkp = Tensor(DT.float, [C, MAX_K], Position.UB)   # dk output in-place
    t_vbf = Tensor(DT.bfloat16, [C, MAX_V], Position.UB)
    ub_beta = Tensor(DT.float, [1, COL_PAD], Position.UB)
    ub_decay = Tensor(DT.float, [1, COL_PAD], Position.UB)
    ub_dbeta = Tensor(DT.float, [1, C], Position.UB)
    ub_dg2 = Tensor(DT.float, [1, C], Position.UB)
    # Pass-V reuses Pass-K addresses (peak UB = per-pass max, not the sum).
    t_dvb = t_dkbg      # dv output in-place
    t_vc = t_dkb        # v_c fp32

    tasks = B * H * N
    per = CeilDiv(tasks, GetVecNum())
    begin = Var(per * GetVecIdx())
    end = Min(begin + per, tasks)

    with auto_sync():
        for tid in range(begin, end, name="tid"):
            b_idx = Var(tid // (H * N))
            rem = Var(tid % (H * N))
            h_idx = Var(rem // N)
            n_idx = Var(rem % N)
            t0 = Var(n_idx * C)
            vc = Min(T - t0, C)

            # ---- Pass-K: load + compute + store dk, dg2 ----
            t_dkbg[0:C, 0:Kd] <<= dkbg[b_idx, h_idx, n_idx, 0:C, 0:Kd]
            t_dkb[0:C, 0:Kd] <<= dkb[b_idx, h_idx, n_idx, 0:C, 0:Kd]
            t_dkt[0:C, 0:Kd] <<= dkt[b_idx, h_idx, n_idx, 0:C, 0:Kd]
            t_k[0:C, 0:Kd] <<= k_f32[b_idx, h_idx, n_idx, 0:C, 0:Kd]
            t_dkp[0:C, 0:Kd] <<= dk_p[b_idx, h_idx, n_idx, 0:C, 0:Kd]
            ub_beta[0:1, 0:C] <<= beta_act[b_idx, h_idx, n_idx, 0:C]
            ub_decay[0:1, 0:C] <<= decay_exp[b_idx, h_idx, n_idx, 0:C]

            passk_vf(t_dkbg, t_dkb, t_dkt, t_k, t_dkp,
                     ub_beta, ub_decay, ub_dbeta, ub_dg2, Kd)

            dk[b_idx, h_idx, n_idx, 0:C, 0:Kd] <<= t_dkp[0:C, 0:Kd]
            dg2[b_idx, h_idx, n_idx, 0:C] <<= ub_dg2[0:1, 0:C]

            # ---- Pass-V: load (reuse addresses) + compute + store dv, dbeta ----
            t_dvb[0:C, 0:Vd] <<= dvb[b_idx, h_idx, n_idx, 0:C, 0:Vd]
            t_vbf[0:vc, 0:Vd] <<= v[b_idx, t0:t0 + vc, h_idx, 0:Vd]

            passv_vf(t_dvb, t_vc, t_vbf, ub_beta, ub_dbeta, Vd, vc)

            dv[b_idx, h_idx, n_idx, 0:C, 0:Vd] <<= t_dvb[0:C, 0:Vd]
            dbeta_wy[b_idx, h_idx, n_idx, 0:C] <<= ub_dbeta[0:1, 0:C]

    return dk, dv, dbeta_wy, dg2


# =====================================================================
# OpExec simulator self-check entry
# =====================================================================
# kernel arg position -> scalar-arg indices (B=0,T=1,H=2,Kd=3,Vd=4,N=5,Cc=6)
_SHAPE_BINDINGS = {
    0: [0, 2, 5, 6, 3],    # dkbg      [B,H,N,C,K]
    1: [0, 2, 5, 6, 4],    # dvb       [B,H,N,C,V]
    2: [0, 2, 5, 6, 3],    # dkb       [B,H,N,C,K]
    3: [0, 2, 5, 6, 3],    # dkt       [B,H,N,C,K]
    4: [0, 2, 5, 6, 3],    # k_f32     [B,H,N,C,K]
    5: [0, 2, 5, 6],       # beta_act  [B,H,N,C]
    6: [0, 2, 5, 6],       # decay_exp [B,H,N,C]
    7: [0, 2, 5, 6, 3],    # dk_p      [B,H,N,C,K]
    8: [0, 1, 2, 4],       # v         [B,T,H,V]
    9: [0, 2, 5, 6, 3],    # dk        [B,H,N,C,K]
    10: [0, 2, 5, 6, 4],   # dv        [B,H,N,C,V]
    11: [0, 2, 5, 6],      # dbeta_wy  [B,H,N,C]
    12: [0, 2, 5, 6],      # dg2       [B,H,N,C]
}


def run_wy_reduce(dkbg, dvb, dkb, dkt, k_f32, beta_act, decay_exp, dk_p, v,
                  simulator=True, out_dir=""):
    B, H, N, _, K = dkbg.shape
    V = dvb.shape[-1]
    T = v.shape[1]
    device = dkbg.device
    dk = torch.empty((B, H, N, C, K), dtype=torch.float32, device=device)
    dv = torch.empty((B, H, N, C, V), dtype=torch.float32, device=device)
    dbeta_wy = torch.empty((B, H, N, C), dtype=torch.float32, device=device)
    dg2 = torch.empty((B, H, N, C), dtype=torch.float32, device=device)

    if simulator:
        gdr_bwd_wy_reduce_kernel._simulator_config = SimulatorConfig(
            core_count=SIM_CORE_COUNT, execution_timeout_s=600.0,
        )
    return OpExec(gdr_bwd_wy_reduce_kernel, out_dir=out_dir, simulator=simulator)(
        dkbg, dvb, dkb, dkt, k_f32, beta_act, decay_exp, dk_p, v,
        dk, dv, dbeta_wy, dg2,
        B, T, H, K, V, N, C,
        shape_bindings=_SHAPE_BINDINGS,
    )


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_here))
    import importlib.util
    _gp = os.path.join(os.path.dirname(_here), "gdr_bwd_wy_reduce_golden_cpu.py")
    _spec = importlib.util.spec_from_file_location("gdr_bwd_wy_reduce_golden_cpu", _gp)
    _gold = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_gold)

    # (name, B, H, N, K, V, T) -- SPEC §8 four shapes.
    cases = [
        ("aligned_t128_h4_k64_v64", 1, 4, 2, 64, 64, 128),
        ("ttail_t100_h4_k64_v64", 1, 4, 2, 64, 64, 100),
        ("t3c_t192_h2_k128_v128", 1, 2, 3, 128, 128, 192),
        ("tail_vlt_t200_h3_k128_v96", 1, 3, 4, 128, 96, 200),
    ]
    seeds = [42, 123, 456]
    out_names = ("dk", "dv", "dbeta_wy", "dg2")
    ATOL, RTOL = 1e-2, 1e-2
    RELL2 = 2e-2
    all_pass = True

    def _rel_l2(a, b):
        return (torch.linalg.vector_norm(a - b)
                / torch.linalg.vector_norm(b).clamp_min(1e-12)).item()

    for (name, B, H, N, K, V, T) in cases:
        for seed in seeds:
            args = _gold._mk_contract_inputs(B, H, N, K, V, T, seed)
            ref = _gold.gdr_bwd_wy_reduce_golden_cpu(*args)
            out = run_wy_reduce(*args, simulator=True)
            outs = out if isinstance(out, tuple) else (out,)
            case_ok = True
            worst = {}
            for nm, got, exp in zip(out_names, outs, ref):
                if tuple(got.shape) != tuple(exp.shape):
                    print(f"  [FAIL] {name} s{seed} {nm} shape {tuple(got.shape)} != {tuple(exp.shape)}")
                    case_ok = False
                    all_pass = False
                    continue
                gf = got.float()
                ef = exp.float()
                finite = torch.isfinite(gf).all().item()
                mixed = torch.allclose(gf, ef, rtol=RTOL, atol=ATOL)
                rl2 = _rel_l2(gf, ef)
                ok = finite and (mixed or rl2 <= RELL2)
                worst[nm] = (gf - ef).abs().max().item()
                if not ok:
                    print(f"  [FAIL] {name} s{seed} {nm} max_abs={worst[nm]:.3e} relL2={rl2:.3e} finite={finite}")
                    case_ok = False
                    all_pass = False
            print(f"  [{'PASS' if case_ok else 'FAIL'}] {name} s{seed} "
                  + " ".join(f"{nm}|Δ|={worst.get(nm, float('nan')):.2e}" for nm in out_names))
    print("ALL PASS" if all_pass else "FAILURES PRESENT")
    if not all_pass:
        sys.exit(1)
