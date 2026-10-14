# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO-Pro gdr_fwd_train_wy — L1 STAGED file, Modules 1..3 (cumulative).

Derived by hand from the verified terminal staged file
`modules/test_gdr_fwd_train_wy_module1234.py` (its kernel body is carried over
BYTE-IDENTICALLY, including the kernel's staged name, exactly as
`scripts/gen_cleanup.py` would have done; gen_cleanup refused because the
staged file compares 22 outputs while `golden_cpu` returns the 17 public ones).
Only the wrapper entry name, the public return tuple and the test oracle differ.

Complete, runnable operator implementing Module 1
(preprocess_vector, s0..s3), Module 2 (cumsum_qk_cube, s5/s7_matmul/s11_matmul)
and Module 3 (wy_build_vector, s6/s7_mask/s8/s9/s11_mask), returning the UNION
of all three modules' outputs the 17 PUBLIC outputs of
module_interfaces.yaml `final_outputs`, in `gdr_fwd_train_wy_golden_cpu` order.

  M1 (Vector)  transpose [B,T,H,*]->[B,H,T,*], cast bf16->fp32, pad T to
               Tp=N*C, beta_act=sigmoid(beta), sig=beta_act, q_s=q*scale,
               v_b=v*beta_act, k_beta=k*beta_act, k_f32=cast(k), q_t=q_s^T.
  M2 (Cube)    decay = cumsum(g_f32) via LTones^T matmul
               P1 = k_beta @ k_f32^T ;  P2 = q_s @ k_f32^T
  M3 (Vector)  decay_exp = exp(decay) ; L_mask = tril(exp(decay_i - decay_j))
               attn = (I - tril_strict(P1*L))^-1 by forward substitution, + I
               a_mat = tril_incl_diag(P2*L)
               q_de = q_s*decay_exp ; kbeta_de = k_beta*decay_exp
               kbeta_t = kbeta_de^T ; dfac = exp(decay[C-1] - decay)

  M4 (Cube)    u = attn @ v_b ; w = attn @ kbeta_de ; w_t = kbeta_t @ attn^T

ONE `@pl.jit(auto_mutex=True)` kernel holding one `section_cube` (M2 then M4)
and one `section_vector` (M1 then M3), ONE launch. This is the TERMINAL module:
it implements the whole operator, and `scripts/gen_cleanup.py` derives
`test_gdr_fwd_train_wy.py` from it.

Wrapper entry (HARD contract): gdr_fwd_train_wy_wrapper

ORIGIN — organized from the Stage-3.5 transpiled product
`kal/gdr_fwd_train_wy/staged_module1234.py`, carrying forward the hardware-verified
`test_gdr_fwd_train_wy_module12.py` (module-check PASS; D5 and D7 ratified).
M1's and M2's bodies, tile shapes, addresses and sync are unchanged; M3's VF
bodies and its place in the cross-core chain are the transpiled product's.

Cross-core chain (INTRA_BLOCK), THREE event pairs, flat and non-overlapping
(seed Fix#1: never nested -- double-nesting deadlocks the depth-1 pipeline):
  vcmutex   e1 READY (V->C, M1 produced)  / e0 RELEASE (C->V, M2 consumed)
  cvmutex   e2 READY (C->V, M2 stored)    / e3 RELEASE (V->C, M3 has loaded)
  vcmutex2  e5 READY (V->C, M3 stored)    / e4 RELEASE (C->V, M4 consumed)
Each is bootstrapped once by the section owning its RELEASE half and drained
once at that section's end, so every set has exactly one matching wait.
M4's operands span the D9 banding (attn from sub-block 0, kbeta_de/kbeta_t from
sub-block 1); e5 is set UNGUARDED, so it fires only once BOTH AIVs are done.

=============================================================================
DEVIATIONS
=============================================================================
D1. `order=[1, 3]` on the q / k / v public 4-D reads. Unchanged. All cases run
    H>1 (4/4/2/3/8), so the H=1 blind spot is actively covered.
D2. Mixed-domain contiguous task grid, identical expression in both sections.
    Resolved at Module 12; unchanged.
D3. FULLY RESOLVED THIS ROUND. `ltones_ws` came back at Module 12; the
    VT-padded `v_b_ws` is restored here, which is what Module 4 reads at a
    fixed SUB width while the declared `v_b` output keeps its valid-V width.
    Nothing from the transpiled product remains dropped.
D4. `transpose_vf` index register DT_INT32 -> DT_UINT32; the transpiled
    `vf.xor` form does not compile on a5 (ErrCode F0FFFF).
D5. DESIGN §3 depth 1 -> 2 on M1's five rotating UB tiles (RATIFIED). Seven new
    mutex ids, 16-22.
D7. LTones build hoisted out of the per-tid loop (RATIFIED). DESIGN §1 already
    specified "每核一次"; the transpiled product rebuilt it every tid.
    NOTE for M3: `ub_lt` is reused as the L_mask buffer here, which is safe
    precisely because D7 made the LTones GM copy a write-once artefact that the
    Cube reads from GM, never from UB.

D10. NEW — Module 4's `attn` operand needs TWO L1 tiles, not one, and the L0A/
    L0B views must match the operand's PHYSICAL shape.
    The transpiled product loads `attn` ONCE with `order=[4,3]` into a single
    default-layout [C,C] L1 tile and feeds it to L0A for u/w and to L0B for
    w_t. Two hardware rejections followed, in order:
      (a) `insert: unsupported layout/dtype combination src=Mat(NZ,DT_FP32) ->
          dst=Left(NZ,DT_FP32)` — a transposed load destined for L0A is not a
          supported path. Per the pl.load transpose table, a transposed operand
          needs `layout=ZN`, an untransposed one the default NZ. But u/w want
          attn UNtransposed (GM already holds it that way) while w_t wants
          attn^T, so ONE tile cannot serve both. Split into `l1_attn` (NZ,
          normal load) for u/w and `l1_attn_t` (ZN, `order=[4,3]`) for w_t.
      (b) the same error again, because `pl.move` requires source and
          destination physical [Rows, Cols] to MATCH, and M4's operands are all
          [C,C]=[64,64] while the L0A/L0B views inherited from M2 are
          [C,KMAX]/[KMAX,C]=[64,128]. Added [C,C] Left/Right views aliasing the
          same L0 addresses and mutex ids (the transpiled product's own
          `__l0a__v2` / `__l0b__v0`, which M12 had not needed).
    Math unchanged; `w` and `w_t` verify to 1e-8..1e-11, which is what confirms
    the routing. Cost: one extra [C,C] L1 tile (16 KB, L1 top 0x29000 of 512 KB).

D8. NEW — `dfac_vf` shadows its own loop variable in the transpiled product:
        for dc in pl.range(0, 64, 1):
            dc = vf.load_align(ub_dec, dc, dist=BRC_B32)   # dc is now a REGISTER
            vf.store_align(ub_dfac + (dc) * 64, s, ...)    # register as an offset
    The store offset is therefore computed from a vector register rather than
    the row index. The KAL simulator tolerated it; it cannot mean anything on
    hardware. Repaired by naming the loaded register `dcur`. Values and
    addresses are otherwise exactly what the surrounding code intends.

D9. NEW -- Module 3 vector work BANDED across both AIV sub-blocks.
    a5 gives each block two AIVs and the transpiled guard idiom leaves
    sub-block 1 idle. M3 splits cleanly into two independent GM-to-GM chains:
      sub-block 0: decay_exp, L_mask, attn  (this is the chain holding the
                   sequential `fwdsubst_vf` recurrence)
      sub-block 1: a_mat, q_de, kbeta_de, kbeta_t, dfac
    Both chains need decay, decay_exp and L_mask, so BOTH sub-blocks build them
    (1 + 64 register ops, negligible) and only sub-block 0 stores the two that
    are contract outputs. Every GM region the two sub-blocks write is disjoint,
    and the three tiles they both build (ub_dec / ub_gf / ub_lt) receive
    BIT-IDENTICAL values from either sub-block -- so the split is correct
    whether the two AIVs have private UB or share it.
    Why this split and not another: a `fwdsubst`-ablation probe on hardware
    measured the recurrence at only 47.6 us of the 205.4 us production kernel
    (23%), NOT the ~85% an op-count estimate predicts -- M3 is store-bound, not
    arithmetic-bound, and chain (b) moves 144 KB per tid against chain (a)'s
    32 KB. That makes the two chains nearly balanced, which is what makes
    banding worth taking here. Math, interfaces and the §5 task grid are
    unchanged; only which AIV executes which independent chain changes.

D17. NEW -- `transpose_vf` REVERSED: strided WRITE at pitch 72 instead of
    strided READ at pitch 64, and `ub_p2` re-pitched to [C, TP=72].
    The transpiled form gathered one source COLUMN per output row at a
    64-element fp32 stride = 256 B = 32 B x 8, an EVEN multiple of the UB bank
    width -- all 64 lanes in one bank, the documented worst case. Measured at
    the production shape [1,2048,8,128,128], 28 blocks, warm median of 8:

        form                                    us    bank_cflt  resc_cflt
        gather, stride 64 fp32 (256 B, even) 170.8        0.338      0.489
        gather, conflict-free stride probe   131.4        0.156      0.960
        no cross-lane at all (ablation)      127.5        0.064      1.387
        scatter, stride TP=72 fp32 (288 B)     -- see the perf doc round --

    So 39.4 us (23% of the kernel) was bank conflict, and the cross-lane
    traffic itself costs under 4 us. Re-pitching the gather SOURCES would need
    all four rotating [C,SUB] slots padded (+8 KB) against 3328 B free, i.e. a
    full UB re-layout; moving the stride to the WRITE side needs only the
    single-slot ub_p2 padded (+2048 B) and fits the existing hole. The source
    row is now read with a plain contiguous `load_align`. Index values are a
    permutation of the gather form's, so results are bit-identical.
    Follow-on edits: `a_build_vf` reads/writes ub_p2 at TP; ub_dec moved to
    0x1fa00. Every GM load/store of ub_p2 keeps valid_shape [C, C], so the
    padding columns are never read or written (vec-14).

Precision engine: comparisons route through the workflow's own
`precision_compare.check_precision` (shipped beside the tests), not a bespoke
allclose, and every per-output line prints `matched_ratio=` and
`max_abs_error=`. Thresholds come from that engine's fixed float32 policy. The
oracle is the Stage-2 CPU FP32 golden `gdr_fwd_train_wy_golden_cpu`, unpacked
output by output.

Audited against the preamble's sim/hardware divergence list: every GM store
offset list has exactly `dst.ndim` entries with the column offset on the last
axis; every TileType declares `valid_shape=[-1, -1]`; matmul and fixpipe run on
full tiles with only the contraction axis and the final store narrowed; no
`AccToVecMode` / L0C->UB path exists in M1-M3, so the `SingleModeVec0`
sub-block-guard hazard does not arise here.
"""

import logging
import os
import sys

# dev-only path setup so the parent-dir NPU golden and the per-Module staged
# golden (this dir) resolve when this staged file is run directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ── compile-time constants (DESIGN §2.1; carried from the transpiled product) ──
C = 64        # chunk size (SPEC-frozen internal constant)
SUB = 64      # K/V sub-block width. Also Module 4's strip width, and that is
              # a MEASURED optimum, not a default -- see D21/D22 below.
HMAX = 32     # H staging tile width for the [B,T,H] g/beta loads
KMAX = 128    # max K, the L1/L0 operand column width
MMIN = 16     # minimum physical M rows for an Acc/Left fp32 tile (fractal)
# D17: physical row pitch of the [C,C] transpose/P2 workspace `ub_p2`. 72 fp32 =
# 288 B = 32 B x 9, an ODD multiple of the UB bank width, which is what makes
# the transpose's strided access conflict-free (see D17 in the docstring).
# Valid shape stays [C, C]; columns [C, TP) are physical padding only.
TP = 72

# ── UB PONG addresses (D5, ratified) — all above the fused top 0x1F300 ──
VA_PONG_Q = 0x20000    # [C,SUB] bf16   8192
VA_PONG_K = 0x22000    # [C,SUB] bf16   8192
VA_PONG_QS = 0x24000   # [C,SUB] fp32  16384
VA_PONG_KF = 0x28000   # [C,SUB] fp32  16384
VA_PONG_KB = 0x2c000   # [C,SUB] fp32  16384

# ── L1 / L0A / L0B / L0C address maps (DESIGN §3, verbatim) ──
L1_QS = 0x0        # [C,KMAX]  32768
L1_KF = 0x8000     # [KMAX,C]  32768
L1_KBETA = 0x10000  # [C,KMAX]  32768
L1_G = 0x18000     # [MMIN,C]   4096
L1_LT = 0x19000    # [C,C]     16384
# Module 4 gets DEDICATED L1 / L0C objects (seed Fix#2: reusing M2's single-
# buffered cube tiles across the M2 -> (vec M3) -> M4 cube->vec->cube boundary
# starves the pipeline -- it never earns a reuse credit).
L1_ATTN = 0x1d000  # [C,C]     16384   M4 only
L1_OP = 0x21000    # [C,C]     16384   M4 only (EXACT [C,C], seed Fix#3)
L1_OP2 = 0x29000   # [C,C]     16384   M4 only, D13 ping/pong partner of L1_OP
L1_ATTN_T = 0x25000  # [C,C]   16384   M4 only, ZN layout for the w_t right
                     #                 operand (see D10). L1 top 0x29000.
L0A_0, L0A_1 = 0x0, 0x8000   # decay- and matmul-shaped views alias these
L0B_0, L0B_1 = 0x0, 0x8000
L0C_DECAY = 0x0
L0C_P1 = 0x1000
L0C_P2 = 0x5000
L0C_UW = 0x9000    # M4 only
L0C_UW2 = 0xd000   # M4 only, D13 ping/pong partner (L0C top 0x11000 of 256 KB)

# ── dtype shim, carried VERBATIM from kal/gdr_fwd_train_wy/staged_module1234.py ──
_DTT = type(pl.DT_FP32)


def _dt_fix(v):
    return v if isinstance(v, _DTT) else _DTT(v)


_PL_BF16 = _dt_fix(pl.DT_BF16)
_PL_FP32 = _dt_fix(pl.DT_FP32)
_PL_INT32 = _dt_fix(pl.DT_INT32)
_PL_UINT32 = _dt_fix(pl.DT_UINT32)


# =============================================================================
#   Vector functions — bodies VERBATIM from kal/gdr_fwd_train_wy/staged_module1234.py
# =============================================================================
@pl.vector_function
def prep_scalars_vf(ub_bh, ub_gh, ub_bact, ub_gf, vc, h_idx):
    """beta_act[r] = sigmoid(beta_logit[r,h]) ; g_f32[r] = g[r,h].

    Rows [vc, C) are the T-tail padding and are written 0 for BOTH outputs, so
    every downstream product involving them is exactly the oracle's zero."""
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, active, dtype=_PL_FP32)
    one = vf.full(1.0, active, dtype=_PL_FP32)
    for sr in pl.range(0, vc, 1):
        blog_bf = vf.load_align(ub_bh, (sr) * HMAX + h_idx, dist=pl.LoadDist.BRC_B16)
        blog_f = vf.astype(blog_bf, active, dtype=_PL_FP32, layout=pl.CastLayout.ZERO)
        den = vf.exp_sub(zero, blog_f, active)
        den = vf.adds(den, 1.0, active)
        den = vf.div(one, den, active)
        vf.store_align(ub_bact + sr, den, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
        gval = vf.load_align(ub_gh, (sr) * HMAX + h_idx, dist=pl.LoadDist.BRC_B32)
        vf.store_align(ub_gf + sr, gval, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
    for spr in pl.range(vc, C, 1):
        vf.store_align(ub_bact + spr, zero, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)
        vf.store_align(ub_gf + spr, zero, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)


@pl.vector_function
def scale_ktile_vf(ub_q, ub_k, ub_bact, ub_qs, ub_kf, ub_kbeta, vc, scale):
    """q_s = q*scale ; k_f32 = (fp32)k ; k_beta = k_f32*beta_act, rows [vc,C)=0."""
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    z = vf.full(0.0, active, dtype=_PL_FP32)
    for kr in pl.range(0, vc, 1):
        bs = vf.load_align(ub_bact, kr, dist=pl.LoadDist.BRC_B32)
        ksrc = vf.load_align(ub_k, (kr) * SUB, dist=pl.LoadDist.UNPK_B16)
        kf = vf.astype(ksrc, active, dtype=_PL_FP32, layout=pl.CastLayout.ZERO)
        vf.store_align(ub_kf + (kr) * SUB, kf, active)
        kb = vf.mul(kf, bs, active)
        vf.store_align(ub_kbeta + (kr) * SUB, kb, active)
        qsrc = vf.load_align(ub_q, (kr) * SUB, dist=pl.LoadDist.UNPK_B16)
        qf = vf.astype(qsrc, active, dtype=_PL_FP32, layout=pl.CastLayout.ZERO)
        qs = vf.muls(qf, scale, active)
        vf.store_align(ub_qs + (kr) * SUB, qs, active)
    for kpr in pl.range(vc, C, 1):
        vf.store_align(ub_qs + (kpr) * SUB, z, active)
        vf.store_align(ub_kf + (kpr) * SUB, z, active)
        vf.store_align(ub_kbeta + (kpr) * SUB, z, active)


@pl.vector_function
def transpose_vf(ub_src, ub_dst):
    """dst[j][i] = src[i][j]: src is [C, SUB] fp32, dst is [C, TP] fp32.

    D4 (declared): the transpiled product builds the index register as DT_INT32
    and then launders it through `vf.xor(idxi, zero_u32, ..., dtype=DT_UINT32)`.
    Hardware rejects that: "vf.xor requires dst, src0, src1 to have the same
    type, got dst=DT_UINT32 src0=DT_INT32 src1=DT_UINT32" (ErrCode F0FFFF). The
    xor-with-zero was only the backend's way of retyping the register, because
    the cross-lane index on an FP32 Tile must be DT_UINT32. Building the iota as
    DT_UINT32 from the start removes the retype entirely.

    D17 (declared) -- GATHER-AT-PITCH-64 REPLACED BY SCATTER-AT-PITCH-72.
    The transpiled form read one SOURCE COLUMN per output row:
        colreg = vf.gather(ub_src + kg, iota << 6, active)   # stride 64 fp32
        vf.store_align(ub_dst + kg * C, colreg, active)
    A 64-element fp32 stride is 256 B = 32 B x 8 -- an EVEN multiple of the UB
    bank width, i.e. every one of the 64 lanes lands in the same bank. That is
    the documented worst case, and it was measured as such: at the production
    shape the kernel ran 170.8 us with `aiv_vec_bank_cflt_ratio` 0.338, while a
    probe that kept the gather but moved it to a conflict-free stride ran
    131.4 us at ratio 0.156 -- 39.4 us, 23% of the kernel, was pure bank
    conflict. A no-cross-lane ablation ran 127.5 us, so the cross-lane traffic
    itself costs under 4 us: the conflict was essentially the whole cost.
    The fix needs the strided side's row pitch to be an ODD multiple of 32 B.
    Re-pitching the SOURCES would have to pad all four rotating [C,SUB] slots
    (ub_qs ping/pong + ub_kbeta ping/pong, +8 KB) and UB had only 3328 B free.
    So the strided access is moved to the WRITE side instead: load one source
    ROW contiguously (`load_align`, no cross-lane cost at all) and `vf.scatter`
    it down one destination COLUMN at pitch TP=72 (288 B = 32 B x 9, odd). Only
    ONE tile is re-pitched, the single-slot ub_p2 (+2048 B), which fits the hole
    that already existed. Index values are a permutation of the gather form's,
    so the result is bit-identical."""
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_UINT32)
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    iota = vf.arange(0, dtype=_PL_UINT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    # idxj[j] = j * TP = (j << 6) + (j << 3), TP = 72
    _s64 = vf.shift_left(iota, 6, _tmp_maskreg_0)
    _s8 = vf.shift_left(iota, 3, _tmp_maskreg_0)
    idxj = vf.add(_s64, _s8, _tmp_maskreg_0)
    for kg in pl.range(0, C, 1):
        rowreg = vf.load_align(ub_src, (kg) * SUB, dtype=_PL_FP32)
        vf.scatter(ub_dst + kg, rowreg, idxj, active)


@pl.vector_function
def scale_vtile_vf(ub_v, ub_bact, ub_vb, vc):
    """v_b = (fp32)v * beta_act, rows [vc, C) zeroed."""
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    z = vf.full(0.0, active, dtype=_PL_FP32)
    for vbr in pl.range(0, vc, 1):
        bs = vf.load_align(ub_bact, vbr, dist=pl.LoadDist.BRC_B32)
        vsrc = vf.load_align(ub_v, (vbr) * SUB, dist=pl.LoadDist.UNPK_B16)
        vfp = vf.astype(vsrc, active, dtype=_PL_FP32, layout=pl.CastLayout.ZERO)
        vb = vf.mul(vfp, bs, active)
        vf.store_align(ub_vb + (vbr) * SUB, vb, active)
    for vbpr in pl.range(vc, C, 1):
        vf.store_align(ub_vb + (vbpr) * SUB, z, active)


@pl.vector_function
def build_ltones_vf(ub_lt):
    """LTones[r, j] = 1 if j <= r else 0 (lower-tri incl diag), row pitch C.

    D3 re-add: verbatim from kal/gdr_fwd_train_wy/staged_module1234.py. Module 2 loads this
    transposed (order=[1,0]) to get the UPPER-triangular operand that turns
    g_row[1,C] @ U[C,C] into the running cumsum."""
    _tmp_maskreg_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    m = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=_PL_INT32)
    zero = vf.full(0.0, _tmp_maskreg_0, dtype=_PL_FP32)
    cols = vf.arange(0, dtype=_PL_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    for ltr in pl.range(0, C, 1):
        row = vf.full(1.0, _tmp_maskreg_0, dtype=_PL_FP32)
        m = vf.lt(cols, (ltr + 1), _tmp_maskreg_1)
        row = vf.select(row, zero, m)
        vf.store_align(ub_lt + (ltr) * C, row, _tmp_maskreg_0)


# =============================================================================
#   Module 3 vector functions — bodies VERBATIM from kal/gdr_fwd_train_wy/staged_module1234.py
#   except dfac_vf, whose transpiled body does not survive (see D8).
# =============================================================================
@pl.vector_function
def decay_exp_vf(ub_dec, ub_dexp):
    """decay_exp[j] = exp(decay[j]) for the whole [1,C] row in one register."""
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, active, dtype=_PL_FP32)
    d = vf.load_align(ub_dec, 0, dtype=_PL_FP32)
    de = vf.exp_sub(d, zero, active)
    vf.store_align(ub_dexp + 0, de, _tmp_maskreg_0)


@pl.vector_function
def lmask_vf(ub_dec, ub_L):
    """L_mask[i,j] = exp(decay[i] - decay[j]) for j <= i, else 0."""
    _tmp_maskreg_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, active, dtype=_PL_FP32)
    cols = vf.arange(0, dtype=_PL_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    dvec = vf.load_align(ub_dec, 0, dtype=_PL_FP32)
    for li in pl.range(0, C, 1):
        di = vf.load_align(ub_dec, li, dist=pl.LoadDist.BRC_B32)
        lrow = vf.exp_sub(di, dvec, active)
        m = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=_PL_INT32)
        m = vf.lt(cols, (li + 1), _tmp_maskreg_0)
        lrow = vf.select(lrow, zero, m)
        vf.store_align(ub_L + (li) * C, lrow, _tmp_maskreg_1)


@pl.vector_function
def attn_build_vf(ub_P1, ub_L, ub_attn):
    """attn = -(P1 * L_mask) kept on the STRICT lower triangle (j < i)."""
    _tmp_maskreg_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, active, dtype=_PL_FP32)
    cols = vf.arange(0, dtype=_PL_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    for ai in pl.range(0, C, 1):
        p = vf.load_align(ub_P1, (ai) * C, dtype=_PL_FP32)
        l = vf.load_align(ub_L, (ai) * C, dtype=_PL_FP32)
        r = vf.mul(p, l, active)
        r = vf.neg(r, active)
        m = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=_PL_INT32)
        m = vf.lt(cols, ai, _tmp_maskreg_0)
        r = vf.select(r, zero, m)
        vf.store_align(ub_attn + (ai) * C, r, _tmp_maskreg_1)


@pl.vector_function
def fwdsubst_vf(ub_attn):
    """In-place forward substitution, then set the unit diagonal (the +I).

    Row i is updated from rows 0..i-1, so this is an inherently SEQUENTIAL
    recurrence over i -- it cannot be split across vector sub-blocks (see the
    banding note in the module docstring). `mul_add_dst(a, b, preg)` computes
    dst = a*b + dst on its pre-initialised dst register, per its API doc; the
    `acc = vf.mul_add_dst(...)` rebinding is the documented idiom.
    Columns >= i stay 0 because every row m < i is zero there, so accumulating
    over the full register width is equivalent to the oracle's `[..., :i]`.

    D14 -- ROWS ARE PRODUCED IN PAIRS (fp, fp+1). Bit-identical to the
    one-row-at-a-time form: each row still accumulates m = 0, 1, ... in the
    same order into a single accumulator, so no sum is reassociated. What
    changes is only the loop shape, and it buys three things at once:
      * `row_m` is loaded ONCE and consumed by both rows, so the inner body is
        5 ops per 2 (m, row) pairs instead of 6 (-15% of the ~6k ops per tid);
      * accA and accB are independent dst registers, so the FMA RAW chain that
        a single accumulator creates is broken two ways (vec-08);
      * row fp's contribution to row fp+1 is taken from the `rowA` REGISTER,
        not re-read from UB, and every UB load of the pair is issued before
        either store -- so one `mem_bar` per PAIR suffices (32 instead of 63).
    C-1 = 63 rows need updating, an odd count: the 31 pairs cover rows 1..62
    and row C-1 is finished alone afterwards by the original single-row form.
    """
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    one = vf.full(1.0, active, dtype=_PL_FP32)
    for fp in pl.range(1, C - 1, 2):
        accA = vf.full(0.0, active, dtype=_PL_FP32)
        accB = vf.full(0.0, active, dtype=_PL_FP32)
        for fm in pl.range(0, fp, 1):
            row_m = vf.load_align(ub_attn, (fm) * C, dtype=_PL_FP32)
            scA = vf.load_align(ub_attn, (fp) * C + fm, dist=pl.LoadDist.BRC_B32)
            scB = vf.load_align(ub_attn, (fp + 1) * C + fm, dist=pl.LoadDist.BRC_B32)
            accA = vf.mul_add_dst(row_m, scA, active)
            accB = vf.mul_add_dst(row_m, scB, active)
        # every UB read of this pair happens BEFORE either store
        old_a = vf.load_align(ub_attn, (fp) * C, dtype=_PL_FP32)
        old_b = vf.load_align(ub_attn, (fp + 1) * C, dtype=_PL_FP32)
        scB2 = vf.load_align(ub_attn, (fp + 1) * C + fp, dist=pl.LoadDist.BRC_B32)
        rowA = vf.add(old_a, accA, active)
        vf.store_align(ub_attn + (fp) * C, rowA, _tmp_maskreg_0)
        # row fp is final; its contribution to row fp+1 comes from the register
        accB = vf.mul_add_dst(rowA, scB2, active)
        rowB = vf.add(old_b, accB, active)
        vf.store_align(ub_attn + (fp + 1) * C, rowB, _tmp_maskreg_0)
        vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    # odd leftover row C-1, original single-row body
    accL = vf.full(0.0, active, dtype=_PL_FP32)
    for fl in pl.range(0, C - 1, 1):
        scL = vf.load_align(ub_attn, (C - 1) * C + fl, dist=pl.LoadDist.BRC_B32)
        rowL_m = vf.load_align(ub_attn, (fl) * C, dtype=_PL_FP32)
        accL = vf.mul_add_dst(rowL_m, scL, active)
    old_l = vf.load_align(ub_attn, (C - 1) * C, dtype=_PL_FP32)
    rowL = vf.add(old_l, accL, active)
    vf.store_align(ub_attn + (C - 1) * C, rowL, _tmp_maskreg_0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VST)
    for fid in pl.range(0, C, 1):
        vf.store_align(ub_attn + (fid) * C + fid, one, _tmp_maskreg_0,
                       dist=pl.StoreDist.FIRST_ELEMENT)

@pl.vector_function
def a_build_vf(ub_P2, ub_L, ub_A):
    """a_mat = (P2 * L_mask) kept on the lower triangle INCLUDING the diagonal.

    D17: ub_P2 and ub_A are the SAME tile (`ub_p2`), re-pitched to TP; ub_L is
    the [C, C] L_mask tile and keeps pitch C."""
    _tmp_maskreg_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, active, dtype=_PL_FP32)
    cols = vf.arange(0, dtype=_PL_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    for Ai in pl.range(0, C, 1):
        p = vf.load_align(ub_P2, (Ai) * TP, dtype=_PL_FP32)
        l = vf.load_align(ub_L, (Ai) * C, dtype=_PL_FP32)
        r = vf.mul(p, l, active)
        m = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=_PL_INT32)
        m = vf.lt(cols, (Ai + 1), _tmp_maskreg_0)
        r = vf.select(r, zero, m)
        vf.store_align(ub_A + (Ai) * TP, r, _tmp_maskreg_1)


@pl.vector_function
def rowscale_vf(ub_src, ub_dexp, ub_dst):
    """dst[i,:] = src[i,:] * decay_exp[i] (per-row scalar broadcast)."""
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    for rc in pl.range(0, C, 1):
        sc = vf.load_align(ub_dexp, rc, dist=pl.LoadDist.BRC_B32)
        row = vf.load_align(ub_src, (rc) * SUB, dtype=_PL_FP32)
        row = vf.mul(row, sc, active)
        vf.store_align(ub_dst + (rc) * SUB, row, _tmp_maskreg_0)


@pl.vector_function
def dfac_vf(ub_dec, ub_dfac):
    """dfac[i,:] = exp(decay[C-1] - decay[i]), broadcast along the K axis.

    D8 (declared): the transpiled body SHADOWS its own loop variable --
        for dc in pl.range(0, 64, 1):
            dc = vf.load_align(ub_dec, dc, dist=BRC_B32)   # dc is now a REGISTER
            s  = vf.exp_sub(dlast, dc, active)
            vf.store_align(ub_dfac + (dc) * 64, s, ...)    # register used as an offset
    so the store offset is computed from a vector register instead of the row
    index. The KAL simulator tolerated it; it cannot mean anything on hardware.
    Repaired by giving the loaded register its own name (`dcur`), which is
    plainly what the surrounding code intends. Values and addresses are
    otherwise unchanged."""
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    dlast = vf.load_align(ub_dec, C - 1, dist=pl.LoadDist.BRC_B32)
    for dc in pl.range(0, C, 1):
        dcur = vf.load_align(ub_dec, dc, dist=pl.LoadDist.BRC_B32)
        s = vf.exp_sub(dlast, dcur, active)
        vf.store_align(ub_dfac + (dc) * SUB, s, _tmp_maskreg_0)


# =============================================================================
#   THE single JIT kernel of this file — M1+M3 (Vector) + M2 (Cube), one launch
# =============================================================================
@pl.jit(auto_mutex=True)
def gdr_fwd_train_wy_module1234_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],     # [B,T,H,K]
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],     # [B,T,H,K]
    v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],     # [B,T,H,V]
    g: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],                 # [B,T,H]
    beta: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],              # [B,T,H]
    ltones_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],                     # [C,C] scratch
    v_b_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],  # [B,H,N,C,VT] D3
    q_s_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k_f32_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    v_b_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    g_f32_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    beta_act_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    sig_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    q_t_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k_beta_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    decay_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    p1_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    p2_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    decay_exp_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    q_de_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    dfac_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    attn_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    a_mat_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    kbeta_t_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    kbeta_de_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    l_mask_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    u_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    w_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    w_t_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    B: pl.DT_INT32,
    T: pl.DT_INT32,
    H: pl.DT_INT32,
    Kd: pl.DT_INT32,
    Vd: pl.DT_INT32,
    N: pl.DT_INT32,
    scale: pl.DT_FP32,
):
    # ---- UB tile groups (Module 1). PING addresses verbatim from the
    # transpiled product; PONG slots per the ratified D5. ----
    g_ub_q = pl.make_tile_group(type=pl.TileType(shape=[C, SUB], dtype=pl.DT_BF16,
                                                 target_memory=pl.MemorySpace.Vec,
                                                 valid_shape=[-1, -1]),
                                addrs=[0x0, VA_PONG_Q], mutex_ids=[4, 16])
    g_ub_k = pl.make_tile_group(type=pl.TileType(shape=[C, SUB], dtype=pl.DT_BF16,
                                                 target_memory=pl.MemorySpace.Vec,
                                                 valid_shape=[-1, -1]),
                                addrs=[0x2000, VA_PONG_K], mutex_ids=[17, 18])
    g_ub_bh = pl.make_tile_group(type=pl.TileType(shape=[C, HMAX], dtype=pl.DT_BF16,
                                                  target_memory=pl.MemorySpace.Vec,
                                                  valid_shape=[-1, -1]),
                                 addrs=0x4000, mutex_ids=[5])
    g_ub_gh = pl.make_tile_group(type=pl.TileType(shape=[C, HMAX], dtype=pl.DT_FP32,
                                                  target_memory=pl.MemorySpace.Vec,
                                                  valid_shape=[-1, -1]),
                                 addrs=0x5000, mutex_ids=[6])
    g_ub_bact = pl.make_tile_group(type=pl.TileType(shape=[1, C], dtype=pl.DT_FP32,
                                                    target_memory=pl.MemorySpace.Vec,
                                                    valid_shape=[-1, -1]),
                                   addrs=0x7000, mutex_ids=[7])
    g_ub_gf = pl.make_tile_group(type=pl.TileType(shape=[1, C], dtype=pl.DT_FP32,
                                                  target_memory=pl.MemorySpace.Vec,
                                                  valid_shape=[-1, -1]),
                                 addrs=0x7100, mutex_ids=[7])
    g_ub_qs = pl.make_tile_group(type=pl.TileType(shape=[C, SUB], dtype=pl.DT_FP32,
                                                  target_memory=pl.MemorySpace.Vec,
                                                  valid_shape=[-1, -1]),
                                 addrs=[0x7200, VA_PONG_QS], mutex_ids=[8, 19])
    g_ub_kf = pl.make_tile_group(type=pl.TileType(shape=[C, SUB], dtype=pl.DT_FP32,
                                                  target_memory=pl.MemorySpace.Vec,
                                                  valid_shape=[-1, -1]),
                                 addrs=[0xb200, VA_PONG_KF], mutex_ids=[9, 20])
    g_ub_kbeta = pl.make_tile_group(type=pl.TileType(shape=[C, SUB], dtype=pl.DT_FP32,
                                                     target_memory=pl.MemorySpace.Vec,
                                                     valid_shape=[-1, -1]),
                                    addrs=[0xf200, VA_PONG_KB], mutex_ids=[21, 22])
    # ub_lt: LTones build scratch (transpiled address 0x13200).
    g_ub_lt = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                  target_memory=pl.MemorySpace.Vec,
                                                  valid_shape=[-1, -1]),
                                 addrs=0x13200, mutex_ids=[10])
    # ub_p2: q_t transpose scratch (M1) / P2 -> a_mat / kbeta_t scratch (M3).
    # D17: physical pitch TP=72 (288 B, odd multiple of the bank width) so the
    # transpose's strided WRITE is conflict-free. 64*72*4 = 18432 B at 0x1b200
    # -> ends 0x1fa00, which is why ub_dec moved up to 0x1fa00 (UB top 0x1fb00).
    # Valid shape is still [C, C]; columns [C, TP) are never read or stored.
    g_ub_p2 = pl.make_tile_group(type=pl.TileType(shape=[C, TP], dtype=pl.DT_FP32,
                                                  target_memory=pl.MemorySpace.Vec,
                                                  valid_shape=[-1, -1]),
                                 addrs=0x1b200, mutex_ids=[8])
    # ub_p1: P1 -> attn workspace (M3); transpiled address 0x17200.
    g_ub_p1 = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                  target_memory=pl.MemorySpace.Vec,
                                                  valid_shape=[-1, -1]),
                                 addrs=0x17200, mutex_ids=[23])
    # ub_dec: decay row (M3). D17 moved it 0x1f200 -> 0x1fa00 to make room for
    # ub_p2's TP padding. UB top is now 0x1fb00 (1280 B still free below the
    # 0x20000 PONG region).
    g_ub_dec = pl.make_tile_group(type=pl.TileType(shape=[1, C], dtype=pl.DT_FP32,
                                                   target_memory=pl.MemorySpace.Vec,
                                                   valid_shape=[-1, -1]),
                                  addrs=0x1fa00, mutex_ids=[24])

    # ---- Cube tile groups (Module 2): L1 operands, L0A/L0B, L0C accumulators.
    # The decay-shaped and matmul-shaped L0A/L0B views ALIAS the same buffers
    # (same addrs AND same mutex_ids) so auto_mutex serialises them; decay
    # completes before P1/P2 within each tid. Transpiled-product idiom. ----
    # NEGATIVE RESULT (D19, measured, reverted -- do not re-spend). cube-08
    # group B says three address-disjoint 32 KB L1 operands (q_s, k_f32^T,
    # k_beta) on ONE mutex id, with k_beta's and k_f32^T's loads issued BACK TO
    # BACK, is a false dependency between two independent MTE2 streams; on
    # `gdr_bwd_wy_matmuls` splitting exactly that was worth -8.4%. Here
    # `g_l1_kf` was moved to the one free id (31) and the clock did NOT move:
    # 128.01 us median of 3 against 128.48 median of 5 for this form, i.e.
    # inside the +-1.5 us spread, with `c_mte2` unchanged at 0.183. Reason the
    # sibling result does not carry: both loads sit on the SAME MTE2 pipe and
    # issue in program order anyway, so the removed edge was already free, and
    # MTE2 is not this kernel's binding cube pipe. Id 31 was therefore spent on
    # the L0C accumulator split below instead, where the false edge is
    # cross-pipe (MMAD -> FixPipe).
    g_l1_qs = pl.make_tile_group(type=pl.TileType(shape=[C, KMAX], dtype=pl.DT_FP32,
                                                  target_memory=pl.MemorySpace.Mat,
                                                  valid_shape=[-1, -1]),
                                 addrs=L1_QS, mutex_ids=[11])
    g_l1_kf = pl.make_tile_group(type=pl.TileType(shape=[KMAX, C], dtype=pl.DT_FP32,
                                                  target_memory=pl.MemorySpace.Mat,
                                                  valid_shape=[-1, -1]),
                                 addrs=L1_KF, mutex_ids=[11])
    g_l1_kbeta = pl.make_tile_group(type=pl.TileType(shape=[C, KMAX], dtype=pl.DT_FP32,
                                                     target_memory=pl.MemorySpace.Mat,
                                                     valid_shape=[-1, -1]),
                                    addrs=L1_KBETA, mutex_ids=[11])
    g_l1_g = pl.make_tile_group(type=pl.TileType(shape=[MMIN, C], dtype=pl.DT_FP32,
                                                 target_memory=pl.MemorySpace.Mat,
                                                 valid_shape=[-1, -1]),
                                addrs=L1_G, mutex_ids=[12])
    g_l1_lt = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                  target_memory=pl.MemorySpace.Mat,
                                                  valid_shape=[-1, -1]),
                                 addrs=L1_LT, mutex_ids=[13])

    g_l0a_dec = pl.make_tile_group(type=pl.TileType(shape=[MMIN, C], dtype=pl.DT_FP32,
                                                    target_memory=pl.MemorySpace.Left,
                                                    valid_shape=[-1, -1]),
                                   addrs=[L0A_0, L0A_1], mutex_ids=[0, 1])
    g_l0a_mm = pl.make_tile_group(type=pl.TileType(shape=[C, KMAX], dtype=pl.DT_FP32,
                                                   target_memory=pl.MemorySpace.Left,
                                                   valid_shape=[-1, -1]),
                                  addrs=[L0A_0, L0A_1], mutex_ids=[0, 1])
    g_l0b_dec = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                    target_memory=pl.MemorySpace.Right,
                                                    valid_shape=[-1, -1]),
                                   addrs=[L0B_0, L0B_1], mutex_ids=[2, 3])
    g_l0b_mm = pl.make_tile_group(type=pl.TileType(shape=[KMAX, C], dtype=pl.DT_FP32,
                                                   target_memory=pl.MemorySpace.Right,
                                                   valid_shape=[-1, -1]),
                                  addrs=[L0B_0, L0B_1], mutex_ids=[2, 3])
    # [C,C] views for Module 4, whose operands are all square. `pl.move`
    # requires source and destination PHYSICAL [Rows, Cols] to match, so the
    # [C,KMAX]/[KMAX,C] views above cannot take a [C,C] L1 tile. Same addrs and
    # same mutex_ids, so auto_mutex serialises the alias (transpiled idiom).
    g_l0a_cc = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                   target_memory=pl.MemorySpace.Left,
                                                   valid_shape=[-1, -1]),
                                  addrs=[L0A_0, L0A_1], mutex_ids=[0, 1])
    g_l0b_cc = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                   target_memory=pl.MemorySpace.Right,
                                                   valid_shape=[-1, -1]),
                                  addrs=[L0B_0, L0B_1], mutex_ids=[2, 3])
    g_l0c_decay = pl.make_tile_group(type=pl.TileType(shape=[MMIN, C], dtype=pl.DT_FP32,
                                                      target_memory=pl.MemorySpace.Acc,
                                                      valid_shape=[-1, -1]),
                                     addrs=L0C_DECAY, mutex_ids=[14])
    g_l0c_p1 = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                   target_memory=pl.MemorySpace.Acc,
                                                   valid_shape=[-1, -1]),
                                  addrs=L0C_P1, mutex_ids=[15])
    # D20: P2's accumulator moved OFF P1's id 15 onto the free id 31.
    # cube-08: sharing a mutex id hurts far more on L0C than on L1, because it
    # serialises MMAD against the FixPipe drains. The access order on id 15 was
    #   matmul(P1) -> matmul(P2) -> store(P1) -> store(P2)
    # so `store(P1)` (FIX) had to wait for `matmul(P2)` (MAC) to COMPLETE even
    # though the two touch disjoint L0C addresses. Split, P1 can drain while
    # P2's MMAD is still running -- the one cross-pipe false edge in M2.
    g_l0c_p2 = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                   target_memory=pl.MemorySpace.Acc,
                                                   valid_shape=[-1, -1]),
                                  addrs=L0C_P2, mutex_ids=[31])
    # ---- Module 4 dedicated cube objects ----
    # attn as the LEFT operand of u/w: GM already holds it in the wanted
    # orientation, so this is the plain non-transposed path -- default (NZ)
    # layout and a normal load.
    g_l1_attn = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                    target_memory=pl.MemorySpace.Mat,
                                                    valid_shape=[-1, -1]),
                                   addrs=L1_ATTN, mutex_ids=[25])
    # attn^T as the RIGHT operand of w_t: GM holds its transpose, so per the
    # pl.load transpose table this needs layout=ZN plus an order-reversed load
    # (D10).
    g_l1_attn_t = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                      target_memory=pl.MemorySpace.Mat,
                                                      valid_shape=[-1, -1],
                                                      layout=pl.ZN),
                                     addrs=L1_ATTN_T, mutex_ids=[28])
    g_l1_op = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                  target_memory=pl.MemorySpace.Mat,
                                                  valid_shape=[-1, -1]),
                                 addrs=[L1_OP, L1_OP2], mutex_ids=[26, 29])
    g_l0c_uw = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                                                   target_memory=pl.MemorySpace.Acc,
                                                   valid_shape=[-1, -1]),
                                  addrs=[L0C_UW, L0C_UW2], mutex_ids=[27, 30])

    # =====================================================================
    #   Module 2 — Cube section. Reads M1's GM outputs, writes decay/P1/P2.
    # =====================================================================
    with pl.section_cube():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx() // pl.get_subblock_num()
        total = B * H * N
        per_core = (total + num_cores - 1) // num_cores
        lo = per_core * core_id
        hi = pl.min(lo + per_core, total)

        t_l1_qs = g_l1_qs.next()
        t_l1_kf = g_l1_kf.next()
        t_l1_kbeta = g_l1_kbeta.next()
        t_l1_g = g_l1_g.next()
        t_l1_lt = g_l1_lt.next()
        t_l0c_decay = g_l0c_decay.next()
        t_l0c_p1 = g_l0c_p1.next()
        t_l0c_p2 = g_l0c_p2.next()
        t_l1_attn = g_l1_attn.next()
        t_l1_attn_t = g_l1_attn_t.next()
        v_tiles_c = (Vd + SUB - 1) // SUB
        k_tiles_c = (Kd + SUB - 1) // SUB

        # bootstrap: let the Vector side write the first tid's chunk
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0,
                                 sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        # bootstrap: let Module 3 write the first tid's attn / kbeta_de /
        # kbeta_t (vcmutex2 RELEASE)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=4,
                                 sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

        for tid in pl.range(lo, hi, 1):
            # wait until the Vector side produced this tid's M1 outputs
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=1,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            b = tid // (H * N)
            rem = tid % (H * N)
            h = rem // N
            n = rem % N

            # ---- s5: decay[1,C] = g_row[1,C] @ LTones^T[C,C]  (cumsum) ----
            pl.set_validshape(t_l1_g, [1, C])
            pl.load(t_l1_g, g_f32_out, [b, h, n, 0])
            pl.set_validshape(t_l1_lt, [C, C])
            pl.load(t_l1_lt, ltones_ws, [0, 0], order=[1, 0])
            la0 = g_l0a_dec.next()
            lb0 = g_l0b_dec.next()
            pl.move(la0, t_l1_g, offset=[0, 0])
            pl.move(lb0, t_l1_lt, offset=[0, 0])
            pl.set_validshape(la0, [MMIN, C])
            pl.set_validshape(lb0, [C, C])
            pl.set_validshape(t_l0c_decay, [1, C])
            pl.matmul(t_l0c_decay, la0, lb0)

            # ---- s7_matmul: P1[C,C] = k_beta[C,K] @ k_f32^T[K,C] ----
            pl.set_validshape(t_l1_kbeta, [C, Kd])
            pl.load(t_l1_kbeta, k_beta_out, [b, h, n, 0, 0])
            pl.set_validshape(t_l1_kf, [Kd, C])
            pl.load(t_l1_kf, k_f32_out, [b, h, n, 0, 0], order=[4, 3])
            la1 = g_l0a_mm.next()
            lb1 = g_l0b_mm.next()
            pl.move(la1, t_l1_kbeta, offset=[0, 0])
            pl.move(lb1, t_l1_kf, offset=[0, 0])
            pl.set_validshape(la1, [C, Kd])
            pl.set_validshape(lb1, [Kd, C])
            pl.set_validshape(t_l0c_p1, [C, C])
            pl.matmul(t_l0c_p1, la1, lb1)

            # ---- s11_matmul: P2[C,C] = q_s[C,K] @ k_f32^T[K,C]  (reuse k^T) ----
            pl.set_validshape(t_l1_qs, [C, Kd])
            pl.load(t_l1_qs, q_s_out, [b, h, n, 0, 0])
            la2 = g_l0a_mm.next()
            lb2 = g_l0b_mm.next()
            pl.move(la2, t_l1_qs, offset=[0, 0])
            pl.move(lb2, t_l1_kf, offset=[0, 0])
            pl.set_validshape(la2, [C, Kd])
            pl.set_validshape(lb2, [Kd, C])
            pl.set_validshape(t_l0c_p2, [C, C])
            pl.matmul(t_l0c_p2, la2, lb2)

            # every GM operand of this tid has been consumed -> Vector may
            # overwrite for tid+1 (vcmutex RELEASE)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # wait until Module 3 finished reading the previous tid's
            # decay/P1/P2 out of GM (cvmutex RELEASE) before writing this tid's
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=3,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            pl.set_validshape(t_l0c_decay, [1, C])
            pl.store(decay_out, t_l0c_decay, [b, h, n, 0])
            pl.set_validshape(t_l0c_p1, [C, C])
            pl.store(p1_out, t_l0c_p1, [b, h, n, 0, 0])
            pl.set_validshape(t_l0c_p2, [C, C])
            pl.store(p2_out, t_l0c_p2, [b, h, n, 0, 0])

            # decay/P1/P2 for this tid are in GM -> Module 3 may read them
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=2,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # ===========================================================
            #   Module 4 — uw_cube. Consumes M3's attn / kbeta_de / kbeta_t
            #   and M1's VT-padded v_b_ws.  s10.
            # ===========================================================
            # wait until Module 3 stored this tid's operands (vcmutex2 READY)
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=5,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # D13 -- attn is LOOP-INVARIANT across all three M4 loops, yet the
            # transpiled product reloaded it from GM on every iteration: 4 NZ
            # loads for u/w plus 2 ZN loads for w_t, 6 x 16 KB where 2 x 16 KB
            # is all the maths needs. Hoisted here (cube-04: the small side of
            # the product stays resident in L1 while the other operand
            # streams). Nothing inside the loops writes either tile, so there
            # is no WAR hazard for auto_mutex to serialise.
            pl.set_validshape(t_l1_attn, [C, C])
            pl.load(t_l1_attn, attn_out, [b, h, n, 0, 0])
            pl.set_validshape(t_l1_attn_t, [C, C])
            pl.load(t_l1_attn_t, attn_out, [b, h, n, 0, 0], order=[4, 3])

            # NEGATIVE RESULT (measured, do not re-spend): `attn` is the LEFT
            # operand of all four u/w matmuls and `attn^T` the RIGHT operand of
            # both w_t matmuls, both loop-invariant, so both can be moved
            # L1->L0 ONCE instead of per iteration -- 64 KB/tid less MTE1,
            # cube-04 applied one level down from L1 to L0. It works and it
            # does move the ratio (`aic_mte1` 0.084 -> 0.073, SAIC 1.070 ->
            # 1.057) but NOT the clock: 131.07 us median / 131.24 mean against
            # 131.43 / 131.01 for the form kept below, i.e. inside the +-1.2 us
            # run-to-run spread, and slower by mean. MTE1 is not this kernel's
            # binding cube pipe and those moves were already overlapped, so the
            # reduction buys nothing. Reverted rather than carried: it adds a
            # loop-carried L0 handle whose WAR hazard auto_mutex has to cover,
            # for no measured gain. NOTE this also falsifies the tempting
            # inference that SAIC ~= 1.0 ("pipes serialised") means every
            # removed non-MAC cube cycle is a removed wall-clock cycle.

            # NEGATIVE RESULTS -- Module 4's STRIP WIDTH IS A MEASURED OPTIMUM
            # AT SUB=64, bracketed in both directions. Do not re-spend either.
            #
            # D21 (coarser, 3 passes): collapse each of u / w / w_t to ONE pass
            #   over the whole V or K width ([C,KMAX] and [KMAX,C] tiles, real
            #   extent on set_validshape). 9 L0C drains/tid -> 6, and six 16 KB
            #   GM->L1 descriptors -> three 32 KB ones, per cube-10. EVERY pipe
            #   ratio improved -- c_fixpipe 0.222 -> 0.150, c_mte2 0.184 ->
            #   0.158, c_mte1 0.085 -> 0.075, c_scalar 0.099 -> 0.054 -- and the
            #   clock went the WRONG WAY: 131.5 us against 126.8 (+3.7%), with
            #   SAIC falling 1.106 -> 0.940. The cube got EMPTIER, not faster.
            #   Mechanism: M4's loads, L1->L0 moves and L0C drains were ALREADY
            #   hidden under MAC by the D13 depth-2 rotation, so the only thing
            #   the merge removed was the interleaving that hid them -- 5
            #   overlap seams became 2, and each pass's exposed head and tail
            #   doubled. M4's WALL CLOCK, not its pipe busy time, is what sits
            #   on the loop-carried M2 -> M3 -> M4 -> M2 chain.
            #   It also found an API fact worth keeping: narrowing an NZ
            #   L0B/L0C by valid_shape along N (physical KMAX, valid 64) gives
            #   deterministic PARTIALLY CORRECT results (u matched_ratio 0.125,
            #   w 0.456, w_t 0.734) while the SAME narrowing along K -- what M2
            #   does to the [KMAX,C] k_f32^T operand at Kd=64 -- is exact. So
            #   cube-08's row/column tail asymmetry, which is stated for GM
            #   stores, does NOT carry over to operand extents: there, N is the
            #   unsafe axis. All five cases pass only because K=V=KMAX needs no
            #   narrowing at all.
            #
            # D22 (finer, 12 passes): strip at SUB4=32 instead of 64 (every
            #   supported K/V is a multiple of 32, so no tail anywhere; all 5
            #   cases bit-identical). This is the direction D21 argues for and
            #   it is ALSO slower: 129.9 us (+2.4%). SAIC rose to 1.363 and
            #   c_mac to 0.521 -- the best overlap and the best MAC ratio of any
            #   variant tried -- but the extra seams cost more pipe work than
            #   they recovered: c_fixpipe 0.222 -> 0.339 (28 -> 44 us of busy
            #   time) and c_mte2 0.184 -> 0.257.
            #
            # Three points, 3 / 6 / 12 passes -> 131.5 / 126.8 / 129.9 us. The
            # existing SUB=64 choice is the interior optimum, so sweep this
            # knob, do not extrapolate from either card.

            # ---- u = attn @ v_b, over V sub-blocks ----
            for uv in pl.range(0, v_tiles_c, 1):
                # D13: t_l1_op and t_l0c_uw now ROTATE per matmul over two
                # slots each (cube-05 for the GM->L1 operand, and the same
                # idea on L0C so Fixpipe drains iteration i while MAC runs
                # i+1 -- cube-01 names that serialisation but reaches it with
                # phase=; L0C had 0xf0000 free, so plain double buffering
                # gets it without the unit_flag contract).
                t_l1_op = g_l1_op.next()
                t_l0c_uw = g_l0c_uw.next()
                pl.set_validshape(t_l1_op, [C, C])
                if Vd % SUB == 0:
                    pl.load(t_l1_op, v_b_out, [b, h, n, 0, uv * SUB])
                if Vd % SUB != 0:
                    pl.load(t_l1_op, v_b_ws, [b, h, n, 0, uv * SUB])
                lau = g_l0a_cc.next()
                lbu = g_l0b_cc.next()
                pl.move(lau, t_l1_attn, offset=[0, 0])
                pl.move(lbu, t_l1_op, offset=[0, 0])
                pl.set_validshape(lau, [C, C])
                pl.set_validshape(lbu, [C, C])
                pl.set_validshape(t_l0c_uw, [C, C])
                pl.matmul(t_l0c_uw, lau, lbu)
                pl.set_validshape(t_l0c_uw, [C, pl.min(Vd - uv * SUB, SUB)])
                pl.store(u_out, t_l0c_uw, [b, h, n, 0, uv * SUB])

            # ---- w = attn @ kbeta_de, over K sub-blocks ----
            for wk in pl.range(0, k_tiles_c, 1):
                t_l1_op = g_l1_op.next()
                t_l0c_uw = g_l0c_uw.next()
                pl.set_validshape(t_l1_op, [C, C])
                pl.load(t_l1_op, kbeta_de_out, [b, h, n, 0, wk * SUB])
                law = g_l0a_cc.next()
                lbw = g_l0b_cc.next()
                pl.move(law, t_l1_attn, offset=[0, 0])
                pl.move(lbw, t_l1_op, offset=[0, 0])
                pl.set_validshape(law, [C, C])
                pl.set_validshape(lbw, [C, C])
                pl.set_validshape(t_l0c_uw, [C, C])
                pl.matmul(t_l0c_uw, law, lbw)
                pl.set_validshape(t_l0c_uw, [C, C])
                pl.store(w_out, t_l0c_uw, [b, h, n, 0, wk * SUB])

            # ---- w_t = w^T = kbeta_t @ attn^T (no L1-side .T) ----
            for wtk in pl.range(0, k_tiles_c, 1):
                t_l1_op = g_l1_op.next()
                t_l0c_uw = g_l0c_uw.next()
                pl.set_validshape(t_l1_op, [C, C])
                pl.load(t_l1_op, kbeta_t_out, [b, h, n, wtk * SUB, 0])
                lat = g_l0a_cc.next()
                lbt = g_l0b_cc.next()
                pl.move(lat, t_l1_op, offset=[0, 0])
                pl.move(lbt, t_l1_attn_t, offset=[0, 0])
                pl.set_validshape(lat, [C, C])
                pl.set_validshape(lbt, [C, C])
                pl.set_validshape(t_l0c_uw, [C, C])
                pl.matmul(t_l0c_uw, lat, lbt)
                pl.set_validshape(t_l0c_uw, [C, C])
                pl.store(w_t_out, t_l0c_uw, [b, h, n, wtk * SUB, 0])

            # M3's operands for this tid are consumed -> M3 may overwrite them
            # for tid+1 (vcmutex2 RELEASE)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=4,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

        # drain the bootstrap cvmutex RELEASE the Vector side set before its loop
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=3,
                                  sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

    # =====================================================================
    #   Module 1 — Vector section. Produces the 8 M1 outputs + LTones.
    # =====================================================================
    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx() // pl.get_subblock_num()
        total = B * H * N
        per_core = (total + num_cores - 1) // num_cores
        lo = per_core * core_id
        hi = pl.min(lo + per_core, total)
        k_tiles = (Kd + SUB - 1) // SUB
        v_tiles = (Vd + SUB - 1) // SUB

        # single-slot groups: fetched once, as in the transpiled product
        t_ub_bh = g_ub_bh.next()
        t_ub_gh = g_ub_gh.next()
        t_ub_bact = g_ub_bact.next()
        t_ub_gf = g_ub_gf.next()
        t_ub_lt = g_ub_lt.next()
        t_ub_p2 = g_ub_p2.next()
        t_ub_p1 = g_ub_p1.next()
        t_ub_dec = g_ub_dec.next()

        # bootstrap: give the Cube permission to write the first tid's
        # decay/P1/P2 (cvmutex RELEASE)
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=3,
                                 sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

        # ── LTones: a chunk-INDEPENDENT constant. D7 (declared): the
        # transpiled product rebuilds and re-stores it inside the tid loop; it
        # is hoisted here to once per core. Ordering is preserved because this
        # store precedes the first `set_cross_core(e1)`, which is what the Cube
        # waits on before its first `ltones_ws` read. Every core writes the
        # same bytes to the same address, so the cross-core overlap is benign
        # (identical-value idempotent write), exactly as in the transpiled
        # product where every tid of every core did the same. ──
        if pl.get_subblock_idx() == 0:
            build_ltones_vf(t_ub_lt)
            pl.set_validshape(t_ub_lt, [C, C])
            pl.store(ltones_ws, t_ub_lt, [0, 0])

        for tid in pl.range(lo, hi, 1):
            # wait for permission to overwrite this tid's chunk outputs
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=0,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # chunk indices — pure scalar arithmetic, shared by M1 and M3
            b = tid // (H * N)
            rem = tid % (H * N)
            h = rem // N
            n = rem % N
            t_base = n * C
            vc = pl.min(T - t_base, C)

            # M1 arithmetic runs on AIV sub-block 0; the cross-core set/wait
            # stay UNGUARDED so both AIVs of the block signal (INTRA_BLOCK).
            if pl.get_subblock_idx() == 0:
                # ── beta_act = sigmoid(beta) ; g_f32 = g  (+ T-tail zeroed) ──
                pl.set_validshape(t_ub_bh, [vc, H])
                pl.load(t_ub_bh, beta, [b, t_base, 0])
                pl.set_validshape(t_ub_gh, [vc, H])
                pl.load(t_ub_gh, g, [b, t_base, 0])
                prep_scalars_vf(t_ub_bh, t_ub_gh, t_ub_bact, t_ub_gf, vc, h)

                pl.set_validshape(t_ub_gf, [1, C])
                pl.store(g_f32_out, t_ub_gf, [b, h, n, 0])
                pl.set_validshape(t_ub_bact, [1, C])
                pl.store(beta_act_out, t_ub_bact, [b, h, n, 0])
                pl.set_validshape(t_ub_bact, [1, C])
                pl.store(sig_out, t_ub_bact, [b, h, n, 0])

                # ── q_s / k_f32 / k_beta / q_t, per K sub-block ──
                for kt in pl.range(0, k_tiles, 1):
                    # D5: rotate the double-buffered slots once per iteration
                    t_ub_q = g_ub_q.next()
                    t_ub_k = g_ub_k.next()
                    t_ub_qs = g_ub_qs.next()
                    t_ub_kf = g_ub_kf.next()
                    t_ub_kbeta = g_ub_kbeta.next()
                    pl.set_validshape(t_ub_q, [vc, SUB])
                    pl.load(t_ub_q, q, [b, t_base, h, kt * SUB], order=[1, 3])
                    pl.set_validshape(t_ub_k, [vc, SUB])
                    pl.load(t_ub_k, k, [b, t_base, h, kt * SUB], order=[1, 3])
                    scale_ktile_vf(t_ub_q, t_ub_k, t_ub_bact,
                                   t_ub_qs, t_ub_kf, t_ub_kbeta, vc, scale)
                    pl.set_validshape(t_ub_qs, [C, SUB])
                    pl.store(q_s_out, t_ub_qs, [b, h, n, 0, kt * SUB])
                    pl.set_validshape(t_ub_kf, [C, SUB])
                    pl.store(k_f32_out, t_ub_kf, [b, h, n, 0, kt * SUB])
                    pl.set_validshape(t_ub_kbeta, [C, SUB])
                    pl.store(k_beta_out, t_ub_kbeta, [b, h, n, 0, kt * SUB])
                    transpose_vf(t_ub_qs, t_ub_p2)
                    pl.set_validshape(t_ub_p2, [C, C])
                    pl.store(q_t_out, t_ub_p2, [b, h, n, kt * SUB, 0])

                # ── v_b, per V sub-block (last may be partial: V=96 -> 32) ──
                for vt in pl.range(0, v_tiles, 1):
                    t_ub_q = g_ub_q.next()
                    t_ub_qs = g_ub_qs.next()
                    vw = pl.min(Vd - vt * SUB, SUB)
                    pl.set_validshape(t_ub_q, [vc, vw])
                    pl.load(t_ub_q, v, [b, t_base, h, vt * SUB], order=[1, 3])
                    scale_vtile_vf(t_ub_q, t_ub_bact, t_ub_qs, vc)
                    # D3 re-add: the VT-padded workspace Module 4 reads at a
                    # fixed 64 width, written BEFORE the valid-width output so
                    # the narrower set_validshape does not clip it.
                    # D16: v_b_ws is a VT-padded DUPLICATE of v_b, present
                    # only so Module 4 can read a full SUB-wide tile. When V
                    # is a multiple of SUB both tensors have the same row
                    # pitch and the copy is byte-for-byte redundant -- 32 KB
                    # of dead MTE3 per tid at the production shape. Module 4
                    # selects its source with the matching test.
                    if Vd % SUB != 0:
                        pl.set_validshape(t_ub_qs, [C, SUB])
                        pl.store(v_b_ws, t_ub_qs, [b, h, n, 0, vt * SUB])
                    pl.set_validshape(t_ub_qs, [C, vw])
                    pl.store(v_b_out, t_ub_qs, [b, h, n, 0, vt * SUB])

            # this tid's M1 outputs are in GM -> Cube may read them
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=1,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # ===========================================================
            #   Module 3 — wy_build_vector. Consumes M2's decay/P1/P2.
            # ===========================================================
            # wait until the Cube stored this tid's decay/P1/P2 (cvmutex READY)
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=2,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # D9 BANDING — Module 3 splits into two independent GM-to-GM chains
            # that run CONCURRENTLY on the block's two AIVs:
            #   sub-block 0 : decay_exp, L_mask, attn   (holds fwdsubst)
            #   sub-block 1 : a_mat, q_de, kbeta_de, kbeta_t, dfac
            # Both need `decay`, `decay_exp` and `L_mask` in their own UB, so
            # both compute them (1 + 64 register ops -- negligible) and only
            # sub-block 0 STORES the two that are contract outputs. Every GM
            # region written by the two sub-blocks is disjoint.
            pl.set_validshape(t_ub_dec, [1, C])
            pl.load(t_ub_dec, decay_out, [b, h, n, 0])
            if pl.get_subblock_idx() == 0:
                pl.set_validshape(t_ub_p1, [C, C])
                pl.load(t_ub_p1, p1_out, [b, h, n, 0, 0])
            if pl.get_subblock_idx() == 1:
                pl.set_validshape(t_ub_p2, [C, C])
                pl.load(t_ub_p2, p2_out, [b, h, n, 0, 0])

            # M2's GM outputs are now resident in UB -> the Cube may overwrite
            # them for tid+1 (cvmutex RELEASE). Released BEFORE the M3 compute
            # so the Cube is not held for the whole of Module 3. Unguarded, so
            # it fires only once BOTH AIVs have finished their loads.
            pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=3,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # wait until Module 4 consumed the PREVIOUS tid's attn / kbeta_de
            # / kbeta_t before Module 3 overwrites them (vcmutex2 RELEASE)
            #
            # BARRIER AUDIT (cube-08 precondition zero) -- RUN, AND IT CLOSES
            # EMPTY ON THIS KERNEL. Do not re-spend. A lot of e4-independent
            # work does sit behind this wait (decay_exp_vf, lmask_vf, the whole
            # 64-step fwdsubst_vf recurrence, a_build_vf, dfac_vf, and the q_s /
            # k_beta loads -- none of which touch the three GM regions e4
            # guards), which is exactly the `gdr_bwd_dv_local` shape that paid
            # -12.1% there. Here moving it down cannot pay, because e4 PROVABLY
            # NEVER BLOCKS in steady state: `wait(e2)` for tid j precedes it in
            # program order on this same core, and e2(j) is set by the Cube only
            # after M2(j), which the Cube reaches only after M4(j-1) -> set(e4).
            # By the count, 1 bootstrap + j sets against j prior waits leaves
            # one credit already posted. Same argument retires wait(e0) and the
            # Cube's wait(e3).
            # The three waits that DO block are the Vector's e2 (on M2), the
            # Cube's e5 (on M3) and the Cube's e1 (on M1), and each has almost
            # nothing independent behind it: M3's compute all descends from
            # decay/P1/P2, M4's from attn, and M2's loads are precisely what e1
            # guards -- that last one is a hard floor on cube-08's three-in-one,
            # since the operands do not EXIST until the Vector signals, so no
            # buffer depth can give the load an earlier issue point.
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=4,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # ── s6 / s7_mask: both sub-blocks build their own decay_exp+L_mask
            decay_exp_vf(t_ub_dec, t_ub_gf)
            lmask_vf(t_ub_dec, t_ub_lt)

            if pl.get_subblock_idx() == 0:
                pl.set_validshape(t_ub_gf, [1, C])
                pl.store(decay_exp_out, t_ub_gf, [b, h, n, 0])
                pl.set_validshape(t_ub_lt, [C, C])

                # ── s8: attn = (I - tril_strict(P1*L))^-1, in place on ub_p1 ──
                attn_build_vf(t_ub_p1, t_ub_lt, t_ub_p1)
                fwdsubst_vf(t_ub_p1)
                pl.set_validshape(t_ub_p1, [C, C])
                pl.store(attn_out, t_ub_p1, [b, h, n, 0, 0])

            if pl.get_subblock_idx() == 1:
                # ── s11_mask: a_mat = tril_incl_diag(P2*L), in place on ub_p2 ──
                a_build_vf(t_ub_p2, t_ub_lt, t_ub_p2)
                pl.set_validshape(t_ub_p2, [C, C])
                pl.store(a_mat_out, t_ub_p2, [b, h, n, 0, 0])

                # ── s9: q_de / kbeta_de / kbeta_t / dfac, per K sub-block ──
                # D15: dfac is INDEPENDENT of kt3 -- dfac[i, :] =
                # exp(decay[C-1] - decay[i]) broadcast along K, the SAME tile
                # for every K sub-block. The transpiled form rebuilt it per
                # sub-block (64 exp_sub + 64 register stores wasted per tid).
                t3_kf = g_ub_kf.next()
                dfac_vf(t_ub_dec, t3_kf)

                for kt3 in pl.range(0, k_tiles, 1):
                    t3_qs = g_ub_qs.next()
                    t3_kb = g_ub_kbeta.next()
                    # D15: BOTH GM loads of the iteration are issued before
                    # the first VF, so the k_beta MTE2 overlaps the q_s
                    # rowscale instead of queueing behind its store.
                    pl.set_validshape(t3_qs, [C, SUB])
                    pl.load(t3_qs, q_s_out, [b, h, n, 0, kt3 * SUB])
                    pl.set_validshape(t3_kb, [C, SUB])
                    pl.load(t3_kb, k_beta_out, [b, h, n, 0, kt3 * SUB])

                    rowscale_vf(t3_qs, t_ub_gf, t3_qs)
                    pl.set_validshape(t3_qs, [C, SUB])
                    pl.store(q_de_out, t3_qs, [b, h, n, 0, kt3 * SUB])

                    rowscale_vf(t3_kb, t_ub_gf, t3_kb)
                    pl.set_validshape(t3_kb, [C, SUB])
                    pl.store(kbeta_de_out, t3_kb, [b, h, n, 0, kt3 * SUB])

                    transpose_vf(t3_kb, t_ub_p2)
                    pl.set_validshape(t_ub_p2, [C, C])
                    pl.store(kbeta_t_out, t_ub_p2, [b, h, n, kt3 * SUB, 0])

                    pl.set_validshape(t3_kf, [C, SUB])
                    pl.store(dfac_out, t3_kf, [b, h, n, 0, kt3 * SUB])

            # this tid's attn / kbeta_de / kbeta_t are in GM -> Module 4 may
            # read them (vcmutex2 READY). Unguarded: fires once BOTH AIVs are
            # done, which is what makes the D9 banding safe for M4's operands
            # (attn from sub-block 0, kbeta_de / kbeta_t from sub-block 1).
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=5,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

        # drain the two bootstraps the Cube set after its final reads
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=0,
                                  sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=4,
                                  sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)


# =============================================================================
#   Wrapper — HARD contract name gdr_fwd_train_wy_wrapper
# =============================================================================
def gdr_fwd_train_wy_wrapper(q, k, v, g, beta):
    """Operator entry: validate + pure-Python shape math + alloc + ONE launch.

    Returns the 17 PUBLIC outputs -- module_interfaces.yaml `final_outputs`, in
    the order `gdr_fwd_train_wy_golden_cpu` returns them:
        q_s, k_f32, v_b, g_f32, beta_act, sig, q_t, decay, decay_exp,
        dfac, q_de, attn, a_mat, kbeta_t, u, w, w_t
    The kernel additionally writes 5 internal-only tensors (k_beta, P1, P2,
    kbeta_de, L_mask) that the modules pass between themselves; those are
    allocated here as GM workspace and NOT returned. Note the public order puts
    `dfac` before `q_de`, which is the reverse of the staged module order.
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "q/k/v must be [B,T,H,*]"
    assert g.dim() == 3 and beta.dim() == 3, "g/beta must be [B,T,H]"
    B, T, H, K = q.shape
    V = v.shape[3]
    assert k.shape == q.shape, "k must match q"
    assert v.shape[:3] == q.shape[:3], "v must share [B,T,H] with q"
    assert g.shape == (B, T, H) and beta.shape == (B, T, H)
    assert K % SUB == 0 and K <= KMAX, f"K must be a multiple of {SUB}, <= {KMAX}; got {K}"
    assert H <= HMAX, f"H must be <= {HMAX} (staging tile width), got {H}"

    N = (T + C - 1) // C                 # pure-Python shape arithmetic
    scale = 1.0 / (K ** 0.5)             # pure-Python scalar
    dev = q.device

    # Module 1 outputs (also Module 2's GM operands)
    q_s = torch.empty((B, H, N, C, K), dtype=torch.float32, device=dev)
    k_f32 = torch.empty((B, H, N, C, K), dtype=torch.float32, device=dev)
    v_b = torch.empty((B, H, N, C, V), dtype=torch.float32, device=dev)
    g_f32 = torch.empty((B, H, N, C), dtype=torch.float32, device=dev)
    beta_act = torch.empty((B, H, N, C), dtype=torch.float32, device=dev)
    sig = torch.empty((B, H, N, C), dtype=torch.float32, device=dev)
    q_t = torch.empty((B, H, N, K, C), dtype=torch.float32, device=dev)
    k_beta = torch.empty((B, H, N, C, K), dtype=torch.float32, device=dev)  # internal
    # Module 2 outputs (P1 / P2 internal)
    decay = torch.empty((B, H, N, C), dtype=torch.float32, device=dev)
    p1 = torch.empty((B, H, N, C, C), dtype=torch.float32, device=dev)
    p2 = torch.empty((B, H, N, C, C), dtype=torch.float32, device=dev)
    # Module 3 outputs
    decay_exp = torch.empty((B, H, N, C), dtype=torch.float32, device=dev)
    q_de = torch.empty((B, H, N, C, K), dtype=torch.float32, device=dev)
    dfac = torch.empty((B, H, N, C, K), dtype=torch.float32, device=dev)
    attn = torch.empty((B, H, N, C, C), dtype=torch.float32, device=dev)
    a_mat = torch.empty((B, H, N, C, C), dtype=torch.float32, device=dev)
    kbeta_t = torch.empty((B, H, N, K, C), dtype=torch.float32, device=dev)
    kbeta_de = torch.empty((B, H, N, C, K), dtype=torch.float32, device=dev)  # internal
    l_mask = torch.empty((B, H, N, C, C), dtype=torch.float32, device=dev)    # internal
    # Module 4 outputs
    u = torch.empty((B, H, N, C, V), dtype=torch.float32, device=dev)
    w = torch.empty((B, H, N, C, K), dtype=torch.float32, device=dev)
    w_t = torch.empty((B, H, N, K, C), dtype=torch.float32, device=dev)
    # internal scratch: the lower-triangular ones matrix Module 2 consumes,
    # and the V-padded v_b copy Module 4 reads at a fixed SUB width (D3).
    ltones = torch.empty((C, C), dtype=torch.float32, device=dev)
    VT = ((V + SUB - 1) // SUB) * SUB          # pure-Python shape arithmetic
    v_b_pad = torch.empty((B, H, N, C, VT), dtype=torch.float32, device=dev)

    # DESIGN §5: block_dim = min(max_blocks, B*H*N). Module 12 is a MIXED
    # (AIC + AIV) kernel, so the logical-block count is core_num, read from the
    # runtime platform and never hard-coded. Pure host scalar.
    plat = get_platform_info()
    max_blocks = plat.core_num or 1
    num_cores = max(1, min(B * H * N, max_blocks))

    gdr_fwd_train_wy_module1234_kernel[None, num_cores](
        q, k, v, g, beta, ltones, v_b_pad,
        q_s, k_f32, v_b, g_f32, beta_act, sig, q_t, k_beta,
        decay, p1, p2,
        decay_exp, q_de, dfac, attn, a_mat, kbeta_t, kbeta_de, l_mask,
        u, w, w_t,
        B, T, H, K, V, N, scale)
    torch.npu.synchronize()

    # the 17 `final_outputs`, in golden_cpu order (dfac before q_de)
    return (q_s, k_f32, v_b, g_f32, beta_act, sig, q_t, decay, decay_exp,
            dfac, q_de, attn, a_mat, kbeta_t, u, w, w_t)

