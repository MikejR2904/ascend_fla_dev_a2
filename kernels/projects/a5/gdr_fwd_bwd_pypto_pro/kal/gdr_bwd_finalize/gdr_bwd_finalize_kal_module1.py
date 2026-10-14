# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# -----------------------------------------------------------------------------
# PyPTO-Pro Stage 3.5 (KAL pre-build) -- gdr_bwd_finalize (OP7), single Module, L0.
#
# OP7 = the LAST operator of gated_delta_rule_bwd (module M10, golden_steps
#   b4b_dg, b5, b6, b7).  Its five public outputs (dq,dk,dv,dg,dbeta) ARE the
#   whole backward deliverable.  PURE VECTOR, BHN-parallel, stateless, single
#   Vector Section, no cube / no matmul / no cross_core (DESIGN deviation 1:
#   dg's reverse-cumsum is per-chunk [C,C], N chunks independent -> chunk-parallel
#   BHN grid + vector suffix-mask reduction, NOT a cube tri-matmul over N).
#
# Authored against the easyasc `a5` facade (arch A5); validated in the Python
# OpExec simulator against gdr_bwd_finalize_golden_cpu.py; then transpiled to
# PyPTO-Pro DSL by the backend.  No PyPTO / PyPTO-Pro API is used here.
#
# MATH (per (b,h,n) chunk, C=64 tokens indexed by r):
#   b4b_dg  dg_sum[c] = dg_p[c] + dg2[c]                              # [1,C] fp32, all C rows
#   b5      dg_full[r] = sum_{c>=r} dg_sum[c]                         # == triu_incl @ dg_sum
#             pure-vector reverse-cumsum via total-minus-prefix: total = cadd(dg_sum);
#             for r, prefix = cadd(dg_sum, update_mask(r)) over lanes [0,r);
#             dg_full[r] = total - prefix.  Mathematically identical to the DESIGN's
#             arange+ge suffix-mask reduce (both == triu_incl@dg_sum); the DSL-range
#             update_mask form is the OP6b-proven per-token scatter idiom (see NOTES,
#             deviation D1).  NO scale.  Reads ALL C rows incl tail-pad (golden L385-388).
#   b6      dbeta[r] = dbeta_wy[r] * sig[r] * (1 - sig[r])            # uses sig ONLY (not beta_act)
#   b7      cast dq_p/dk_in/dv_in fp32->bf16 (rule 6: cast no-mask + reg_to_ub_downsample);
#             REINDEXED STORE token t=n*C+r -> public [B,T,H,*]; tail pad rows dropped by
#             storing only the valid vc = min(C, T-n*C) rows (poison-immune).  dg/dbeta fp32.
#
# TILE POLICY (DESIGN §2/§3): every on-chip tile is a compile-time literal
#   [C,MAX_K]=[64,128] / [1,C]=[1,64] / [C,1]=[64,1] (C=64, MAX_K=MAX_V=128).
#   Dynamic extents (K/V/chunk/token) live only in loop bounds and GM offsets,
#   never in a tile shape.
#
# RULE COMPLIANCE (CAMPAIGN §1):
#   rule 5  : all shared/GM vec writes are sub-block-0-guarded -> here every vec
#             op runs in a single Vector Section with GetVecIdx()-strided task
#             ownership; each (b,h,n) chunk writes DISJOINT public token slices,
#             so no shared-tile tearing (the A2 two-sublane hazard does not arise
#             on this pure-vec A5 single-lane path; task loop already owns slices).
#   rule 6  : fp32->bf16 store = cast() with NO mask + reg_to_ub_downsample().
#   rule 15 : K=128 / V=128 wide reg/vec ops done in ceil(W/64) 64-wide halves.
#   rule 17 : no pure-vector STORE->LOAD across a phase boundary on the SAME tile
#             (dg_col/dbeta_col are written per-row then stored; the wide tiles are
#             load->cast->store with no cross-phase store->load re-read), so no
#             vf_barrier is required; auto_sync owns the MTE2->V->MTE3 edges.
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
C = 64             # fixed chunk size = fp32 Reg lane count (golden _C)
LANES = 64         # fp32 VF register width (CAMPAIGN rule 15)
MAX_K = 128        # K compile-time bound (SPEC: K<=128)
MAX_V = 128        # V compile-time bound (SPEC: V<=128)
SIM_CORE_COUNT = 32


# =====================================================================
# b4b_dg + b5 : dg_sum = dg_p + dg2 ; dg_full[r] = sum_{c>=r} dg_sum[c].
# Writes column 0 (rows 0..C-1) of the aligned wide tile t_wide (row r =
# dg_full[r]); reads ALL C rows of dg_sum incl tail-pad (DESIGN §7 / golden L385).
#
# Reverse cumsum by total-minus-prefix (== triu_incl @ dg_sum, NO scale):
#   total = sum_c dg_sum[c] ; dg_full[r] = total - sum_{c<r} dg_sum[c].
# The prefix mask [0,r) comes from update_mask(Var(r)); the whole loop is a DSL
# range so it lowers to ONE store instruction executed C times in program order
# (the OP6b-proven per-token scatter idiom -- a Python unroll over a shared
# result register instead races the store against the next iteration's reduce).
# =====================================================================
@vf()
def dg_reduce_vf(t_dgp: Tensor, t_dg2: Tensor, t_wide: Tensor):
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    dgp = Reg(DT.float)
    dg2r = Reg(DT.float)
    dgsum = Reg(DT.float)
    dgp <<= t_dgp[0:1, 0:LANES]                       # dg_p[0:C]
    dg2r <<= t_dg2[0:1, 0:LANES]                      # dg2[0:C]
    add(dgsum, dgp, dg2r, mask=active)                # dg_sum = dg_p + dg2  (b4b_dg)

    full = MaskReg(DT.float, init_mode=MaskType.ALL)
    update_mask(full, Var(C, dtype=DT.uint32))        # all C lanes
    total = Reg(DT.float)
    cadd(total, dgsum, mask=full)                     # total = sum_c dg_sum[c]

    mpre = MaskReg(DT.float, init_mode=MaskType.ALL)
    pfx = Reg(DT.float)
    sr = Reg(DT.float)
    for r in range(C, name="dg_r"):                   # DSL loop: r is a Var
        update_mask(mpre, Var(r, dtype=DT.uint32))    # prefix mask lanes [0, r)
        cadd(pfx, dgsum, mask=mpre)                   # sum_{c<r} dg_sum[c]
        sub(sr, total, pfx, mask=active)             # dg_full[r] = sum_{c>=r}
        t_wide[r:r + 1, 0:1] <<= sr.single_value()    # -> column 0, row r


# =====================================================================
# b6 : dbeta[r] = dbeta_wy[r] * sig[r] * (1 - sig[r])  (uses sig ONLY).
# Per-token scalar compute (DSL loop) placed into column 0 of the wide tile;
# sig*(1-sig) in [0,0.25] so no overflow.  Per-token form avoids a row->column
# lane transpose (OP6b-proven `t[0:1,c:c+1] <<= scalar.single_value()` idiom,
# applied to column 0 of an aligned wide tile instead).
# =====================================================================
@vf()
def dbeta_calc_vf(t_dbw: Tensor, t_sig: Tensor, t_wide: Tensor):
    active = MaskReg(DT.float, init_mode=MaskType.ALL)
    one = Reg(DT.float)
    dup(one, 1.0, mask=active)
    sgr = Reg(DT.float)
    dbr = Reg(DT.float)
    omr = Reg(DT.float)
    sdr = Reg(DT.float)
    res = Reg(DT.float)
    for r in range(C, name="db_r"):                   # DSL loop: r is a Var
        sgr <<= t_sig[0:1, r:r + 1].single()          # sig[r]  (broadcast)
        dbr <<= t_dbw[0:1, r:r + 1].single()          # dbeta_wy[r]
        sub(omr, one, sgr, mask=active)              # 1 - sig[r]
        mul(sdr, sgr, omr, mask=active)              # sig[r]*(1-sig[r])
        mul(res, dbr, sdr, mask=active)              # dbeta_wy[r]*sig[r]*(1-sig[r])
        t_wide[r:r + 1, 0:1] <<= res.single_value()   # -> column 0, row r


# =====================================================================
# b7-cast : dq_p/dk_in/dv_in fp32 -> bf16 for ALL C rows (rule 6, rule 15).
# t_out (bf16 [C,MAX_K]) is stored to the public layout by the caller.
# =====================================================================
@vf()
def cast_vf(t_in: Tensor, t_out: Tensor, Wd: Var):
    to_bf16 = CastConfig(round_mode=RoundMode.TO_EVEN, reg_layout=RegLayout.ZERO,
                         name="f2bf16")
    n_regs = CeilDiv(Wd, LANES)
    for r in range(C, name="cast_r"):
        for hh in range(n_regs, name="cast_h"):
            off = Var(hh * LANES)
            valid = Min(LANES, Wd - off)
            fr = Reg(DT.float)
            br = Reg(DT.bfloat16)
            fr <<= t_in[r:r + 1, off:off + LANES]
            cast(br, fr, to_bf16)                     # NO mask (rule 6)
            reg_to_ub_downsample(t_out[r:r + 1, off:off + valid], br)


# =====================================================================
# Kernel (single Vector Section, BHN-parallel flat task loop).
# grad_o value UNUSED (only T scalar); beta_act UNUSED (not passed).
# =====================================================================
@kernel(mode="vec")
def gdr_bwd_finalize_kernel(
    dq_p: GMTensor, dk_in: GMTensor, dv_in: GMTensor,
    dg_p: GMTensor, dg2: GMTensor, dbeta_wy: GMTensor, sig: GMTensor,
    dq: GMTensor, dk: GMTensor, dv: GMTensor, dg: GMTensor, dbeta: GMTensor,
    B: Var, T: Var, H: Var, Kd: Var, Vd: Var, N: Var, Cc: Var,
):
    in_big = Tensor(DT.float, [C, MAX_K], Position.UB)       # wide fp32 load (K or V)
    out_big = Tensor(DT.bfloat16, [C, MAX_K], Position.UB)   # wide bf16 landing
    t_dgp = Tensor(DT.float, [1, C], Position.UB)
    t_dg2 = Tensor(DT.float, [1, C], Position.UB)
    t_dbw = Tensor(DT.float, [1, C], Position.UB)
    t_sig = Tensor(DT.float, [1, C], Position.UB)
    # dg / dbeta store source: per-row scalar lives in column 0 of an ALIGNED wide
    # tile.  A width-1 [C,1] tile mis-strides the single-element scatter (the
    # allocator's row alignment corrupts / drops rows); an aligned [C,MAX_K] tile
    # whose column 0 is stored to dg[b, t0:t0+vc, h:h+1] is poison-free.
    col_src = Tensor(DT.float, [C, MAX_K], Position.UB)

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
            vc = Min(T - t0, C)                              # valid rows in this chunk

            # ---- dg path (b4b_dg + b5) : reads full C rows, stores vc ----
            t_dgp[0:1, 0:C] <<= dg_p[b_idx, h_idx, n_idx, 0:C]
            t_dg2[0:1, 0:C] <<= dg2[b_idx, h_idx, n_idx, 0:C]
            dg_reduce_vf(t_dgp, t_dg2, col_src)
            dg[b_idx, t0:t0 + vc, h_idx:h_idx + 1] <<= col_src[0:vc, 0:1]

            # ---- dbeta path (b6) ----
            t_dbw[0:1, 0:C] <<= dbeta_wy[b_idx, h_idx, n_idx, 0:C]
            t_sig[0:1, 0:C] <<= sig[b_idx, h_idx, n_idx, 0:C]
            dbeta_calc_vf(t_dbw, t_sig, col_src)
            dbeta[b_idx, t0:t0 + vc, h_idx:h_idx + 1] <<= col_src[0:vc, 0:1]

            # ---- dq/dk/dv path (b7 cast + reindexed store), in_big/out_big reused ----
            # dq_p -> dq  (width K)
            in_big[0:C, 0:Kd] <<= dq_p[b_idx, h_idx, n_idx, 0:C, 0:Kd]
            cast_vf(in_big, out_big, Kd)
            dq[b_idx, t0:t0 + vc, h_idx, 0:Kd] <<= out_big[0:vc, 0:Kd]
            # dk_in -> dk  (width K)
            in_big[0:C, 0:Kd] <<= dk_in[b_idx, h_idx, n_idx, 0:C, 0:Kd]
            cast_vf(in_big, out_big, Kd)
            dk[b_idx, t0:t0 + vc, h_idx, 0:Kd] <<= out_big[0:vc, 0:Kd]
            # dv_in -> dv  (width V)
            in_big[0:C, 0:Vd] <<= dv_in[b_idx, h_idx, n_idx, 0:C, 0:Vd]
            cast_vf(in_big, out_big, Vd)
            dv[b_idx, t0:t0 + vc, h_idx, 0:Vd] <<= out_big[0:vc, 0:Vd]

    return dq, dk, dv, dg, dbeta


# =====================================================================
# OpExec simulator self-check entry
# =====================================================================
# kernel scalar-arg order: B=0,T=1,H=2,Kd=3,Vd=4,N=5,Cc=6
_SHAPE_BINDINGS = {
    0: [0, 2, 5, 6, 3],    # dq_p     [B,H,N,C,K]
    1: [0, 2, 5, 6, 3],    # dk_in    [B,H,N,C,K]
    2: [0, 2, 5, 6, 4],    # dv_in    [B,H,N,C,V]
    3: [0, 2, 5, 6],       # dg_p     [B,H,N,C]
    4: [0, 2, 5, 6],       # dg2      [B,H,N,C]
    5: [0, 2, 5, 6],       # dbeta_wy [B,H,N,C]
    6: [0, 2, 5, 6],       # sig      [B,H,N,C]
    7: [0, 1, 2, 3],       # dq       [B,T,H,K]
    8: [0, 1, 2, 3],       # dk       [B,T,H,K]
    9: [0, 1, 2, 4],       # dv       [B,T,H,V]
    10: [0, 1, 2],         # dg       [B,T,H]
    11: [0, 1, 2],         # dbeta    [B,T,H]
}


def run_finalize(grad_o, dq_p, dk_in, dv_in, dg_p, dg2, dbeta_wy, sig, beta_act,
                 simulator=True, out_dir=""):
    """Public OP7 wrapper: reads T from grad_o.shape, allocates the 5 public
    outputs, dispatches the single @kernel once.  grad_o value + beta_act UNUSED."""
    _ = beta_act                                     # contract input; == sig; unused
    B, T, H, _V_pub = grad_o.shape                   # only grad_o.shape is read (T anchor)
    Bd, Hd, N, _C, K = dq_p.shape
    V = dv_in.shape[-1]
    device = dq_p.device

    dq = torch.empty((B, T, H, K), dtype=torch.bfloat16, device=device)
    dk = torch.empty((B, T, H, K), dtype=torch.bfloat16, device=device)
    dv = torch.empty((B, T, H, V), dtype=torch.bfloat16, device=device)
    dg = torch.empty((B, T, H), dtype=torch.float32, device=device)
    dbeta = torch.empty((B, T, H), dtype=torch.float32, device=device)

    if simulator:
        gdr_bwd_finalize_kernel._simulator_config = SimulatorConfig(
            core_count=SIM_CORE_COUNT, execution_timeout_s=600.0,
        )
    return OpExec(gdr_bwd_finalize_kernel, out_dir=out_dir, simulator=simulator)(
        dq_p, dk_in, dv_in, dg_p, dg2, dbeta_wy, sig,
        dq, dk, dv, dg, dbeta,
        B, T, H, K, V, N, C,
        shape_bindings=_SHAPE_BINDINGS,
    )


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    import importlib.util
    _gp = os.path.join(os.path.dirname(_here), "gdr_bwd_finalize_golden_cpu.py")
    _spec = importlib.util.spec_from_file_location("gdr_bwd_finalize_golden_cpu", _gp)
    _gold = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_gold)

    # (name, B, T, H, K, V) -- SPEC §8 four shapes.
    cases = [
        ("aligned_b1_t128_h4_k64_v64", 1, 128, 4, 64, 64),
        ("ttail_b1_t100_h4_k64_v64", 1, 100, 4, 64, 64),
        ("t3c_b1_t192_h2_k128_v128", 1, 192, 2, 128, 128),
        ("tail_vlt_b1_t200_h3_k128_v96", 1, 200, 3, 128, 96),
    ]
    seeds = [42, 123, 456]
    out_names = ("dq", "dk", "dv", "dg", "dbeta")
    ATOL, RTOL, RELL2 = 1e-2, 1e-2, 2e-2
    all_pass = True

    def _rel_l2(a, b):
        return (torch.linalg.vector_norm(a - b)
                / torch.linalg.vector_norm(b).clamp_min(1e-12)).item()

    def _peak_scaled(a, b):
        max_abs = (a - b).abs().max().item()
        peak = b.abs().max().clamp_min(1e-6).item()
        return max_abs, max_abs / peak

    for (name, B, T, H, K, V) in cases:
        for seed in seeds:
            args = _gold._mk_contract_inputs(B, T, H, K, V, seed)
            ref = _gold.gdr_bwd_finalize_golden_cpu(*args)          # all fp32
            out = run_finalize(*args, simulator=True)
            outs = out if isinstance(out, tuple) else (out,)
            case_ok = True
            info = {}
            for nm, got, exp in zip(out_names, outs, ref):
                if tuple(got.shape) != tuple(exp.shape):
                    print(f"  [FAIL] {name} s{seed} {nm} shape {tuple(got.shape)} != {tuple(exp.shape)}")
                    case_ok = False
                    all_pass = False
                    continue
                gf = got.float()
                ef = exp.float()
                finite = torch.isfinite(gf).all().item()
                allc = torch.allclose(gf, ef, rtol=RTOL, atol=ATOL)
                max_abs, rel_peak = _peak_scaled(gf, ef)
                rl2 = _rel_l2(gf, ef)
                ok = finite and (allc or rel_peak <= RTOL or rl2 <= RELL2)
                info[nm] = (max_abs, rel_peak, rl2)
                if not ok:
                    print(f"  [FAIL] {name} s{seed} {nm} |Δ|={max_abs:.3e} "
                          f"rpk={rel_peak:.3e} rL2={rl2:.3e} finite={finite}")
                    case_ok = False
                    all_pass = False
            print(f"  [{'PASS' if case_ok else 'FAIL'}] {name} s{seed} "
                  + " ".join(f"{nm}(rpk={info[nm][1]:.1e},rL2={info[nm][2]:.1e})"
                             for nm in out_names if nm in info))
    print("ALL PASS" if all_pass else "FAILURES PRESENT")
    if not all_pass:
        sys.exit(1)
