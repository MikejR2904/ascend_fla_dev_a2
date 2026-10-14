# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO-Pro gdr_bwd_wy_matmuls (OP6a) — FINAL DELIVERABLE (all five Modules fused).

Complete, runnable standalone operator implementing Modules 1, 2 and 3
(module_interfaces.yaml modules[0..2], golden steps b4.1-b4.6 + glue-1):

  b4.1   dA   = dw @ kbeta_t                [C,Kd]@[Kd,C] -> [C,C]   (contract Kd)
  b4.2   dkbg = attn^T @ dw                 [C,C]@[C,Kd]  -> [C,Kd]  *** FINAL dkbg ***
  b4.3   dA  += dv2 @ v_b^T                 [C,Vd]@[Vd,C] -> [C,C]   (contract Vd, acc)
  b4.4   dvb  = attn^T @ dv2                [C,C]@[C,Vd]  -> [C,Vd]  *** FINAL dvb ***
  glue-1 dA_g1 = tril(ones[C,C], diag=-1) * dA      (strict lower: keep col < row)
  b4.5   T    = dA_g1 @ attn^T              [C,C]@[C,C]   -> [C,C]
  b4.6   dA_b = attn^T @ T                  [C,C]@[C,C]   -> [C,C]   (chained via bridge)

Outputs (golden_stage123 order): dkbg, dvb, dA_a, dA_g1, dA_b.

This file re-homes `modules/test_gdr_bwd_wy_matmuls_module12.py` — which is
HARDWARE-VERIFIED (5/5 cases, matched_ratio 1.000000 on every output) — VERBATIM: the
M1 cube path, the M2 banded vector section, the L1/L0/UB address maps, the mutex ids,
the grid and the depth-4 dA hand-off are unchanged.  Module 3 is APPENDED to the END of
the SAME cube loop inside the SAME single `@pl.jit` kernel (two matmuls, one GM->L1
read-back, one L0C->L1 bridge, one store), plus ONE new event signalled by the vector
after its existing store.  No new vector work, no second kernel, no restructuring of
M1/M2.

Single `@pl.jit(auto_mutex=True)` kernel, single launch.  MIX kernel:
`pl.get_block_idx() // pl.get_subblock_num()` yields the same physical AI Core id on both
sides (AIC: //1, AIV: //2), so cube and vector walk the identical DESIGN §5 contiguous
BHN task range — the mixed-domain grid.

Wrapper entry (HARD contract): gdr_bwd_wy_matmuls_wrapper
Compared against gdr_bwd_wy_matmuls_golden_cpu.py through the shared engine
`precision_compare.py` (`check_precision`, dtype fp32) — both
dev-only imports kept inside the test function; tolerances come from that engine's
hard-coded table, not from this file.

The 4-variant transposed-operand hardware probe referenced below is archived at
`docs/perf/gdr_bwd_wy_matmuls/probes/` (script + raw run log).

--------------------------------------------------------------------------------
Declared deviations — Module 2 / this round
--------------------------------------------------------------------------------
 W1  **The vec -> cube_B hand-off goes through GM (`dA_g1`), not a shared L1 tile.**
     kal/gdr_bwd_wy_matmuls/staged.py has the vector relayout its masked dA into a shared L1 tile
     (`l1_dAb`) with a UB ND->NZ `pl.move` followed by EIGHT `pl.insert` calls of [64,8]
     column chunks, all under an `if get_subblock_idx() < 1` guard.  That is
     incompatible with the banded DualModeSplitM drain this file already delivers and
     module-check already passed (V1): after banding, neither sub-block holds all 64
     rows, so no single sub-block can build the [64,8] NZ column chunks.  Rather than
     revert V1 to the guarded SingleModeVec0 shape the preamble flags as suspect, cube_B
     reads the masked dA back from `dA_g1` in GM — which glue-1 must write anyway,
     because it is a REQUIRED Module-2 output.  So this adds one GM read per task and no
     GM write at all.  Ordering is carried by a new one-directional event EV_G1_GM,
     signalled by the vector on MTE3 after its store and waited by the cube before its
     load; this is exactly the GM-workspace hand-off idiom of the hardware-verified
     `gated_delta_rule_fwd` (its deviation D1 and its M1->M2 edge).  `dA_g1` is indexed
     by [b,h,n], so distinct tids touch disjoint GM and there is NO write-after-read
     hazard: the event needs no reverse credit and no bootstrap, and sets/waits balance
     1:1 per tid for any per-core task count including zero.
     The KAL note "GM bridges are never synced" is a statement about the KAL layer's
     `auto_sync`, which does not track GM; it does not apply here, where the ordering is
     an explicit cross-core event.

 W2  **The b4.5 -> b4.6 bridge bounces L0C -> UB -> L1 across the vector lane**, which
     is what kal/gdr_bwd_wy_matmuls/staged.py does, and a direct `pl.move` L0C -> L1 is NOT available.
     I first tried the direct move, because the `pl.move` API doc lists
     "L0C Buffer -> L1 Buffer (equal Tile shapes, no scale): NZ -> NZ" as supported for
     DT_FP32 -> DT_FP32, which made kal/gdr_bwd_wy_matmuls/NOTES.md's "l0c_to_l1 is a hard transpile gap"
     look like a KAL-backend limitation that would not bind when writing the DSL by hand.
     On this build it does bind, and the doc is wrong for it -- the parser rejects it:

       ValueError: move: unsupported data path src(Acc)->dst(Mat), supported paths:
       Mat->Left, Mat->Right, Mat->Scaling, Mat->Bias, Mat->ScaleLeft, Mat->ScaleRight,
       Acc->Vec, Vec->Vec, Vec->Mat
       ErrCode: F00002, Failed to parse kernel function
       'gdr_bwd_wy_matmuls_module12345_kernel'

     That enumeration is the authority for this build, and it leaves exactly one route:
     Acc->Vec then Vec->Mat.  Two refinements over staged.py's version: the UB relay tile
     is declared `layout=pl.NZ` so the drain is Acc(NZ)->Vec(NZ) and the relay a legal
     Vec(NZ)->Mat(NZ) byte copy -- staged.py drains into an ND UB tile and needs a
     separate UB ND->NZ conversion, because a UB tile holding ND may not be copied into
     an NZ/ZN L1 tile and an fp32 L1 tile cannot be ND (TileType doc: L1 defaults to NZ,
     may be ZN, ND only for 64-bit dtypes); and the relay is ONE whole-tile `pl.move`
     rather than staged.py's eight `pl.insert` calls of [64,8] fp32-C0 column chunks.
     The bridge drain uses SingleModeVec0 with the copy guarded to sub-block 0 -- the one
     place this file uses that shape.  It is not the pattern the preamble flags: nothing
     is COMPUTED on a half-populated UB tile (sub-block 1 merely sits out a pure byte
     copy), and the guard is what stops two lanes tearing one shared L1 region rather
     than what creates a hazard.  The signal is outside the guard, because a cross-core
     wait is satisfied only once EVERY vector lane has signalled.
     This bridge costs one cube<->vector round trip per task, inside the tid body, and
     depth-1 is provably safe for it: every WAR edge on `ub_mid` and `l1_m16` is already
     implied by program order on the two sides (see the event-order note in the cube
     section), so it needs no bootstrap credit and balances 1:1 per tid at any per-core
     task count including zero.

 W3  **`t_attnT` is reused as both b4.5's RIGHT and b4.6's LEFT operand.**  staged.py
     loads attn twice more (`l1_attn`, `l1_attnT2`), both with the same descending
     `order=[4,3]`, i.e. both are A^T — the same value this file already has in L1 for
     b4.2/b4.4.  It is read-only within the tid, and `pl.move` converts L1 NZ to
     whichever L0 fractal the operand role needs (L0A NZ / L0B ZN), so one L1 tile
     serves both roles.  Saves two GM->L1 loads and 32 KB of L1 per task.  This is also
     the transpose-load pattern the earlier probe validated: A^T reaches L0A from a
     default-layout L1 tile, which is what b4.6 does.

 V1  **L0C->UB drain uses `AccToVecMode.DualModeSplitM`, not the transpiled product's
     `SingleModeVec0`, and the glue-1 mask is BANDED across both AIV sub-blocks.**
     kal/gdr_bwd_wy_matmuls/staged.py drains dA with `SingleModeVec0` (whole matrix to Vec0 only), runs the
     mask redundantly on both sub-blocks — sub-block 1 operating on stale UB — and then
     guards the consumer with `if get_subblock_idx() < 1`.  That guarded-SingleModeVec0
     shape is flagged suspect on a5 (preamble, gdr_bwd_dv_local), and it also leaves half
     the vector hardware idle.  glue-1 is `dA[i,j] = dA[i,j] if j < i else 0`, which is
     purely ROW-LOCAL, so the math admits an exact M-split: with valid_M=64, Vec0 receives
     global rows 0..31 and Vec1 rows 32..63 (AccToVecMode doc: aligned_M/2 rows to Vec0,
     the rest to Vec1).  Each sub-block masks its own 32-row band with the row index
     biased by `row0 = sub_id * 32`, and stores its own disjoint GM band at row offset
     `row0`.  No sub-block guard is needed anywhere, no shared region is written twice
     (so nothing can tear), and both AIV sub-blocks do real work.  Math identical.

 V2  **Two-event cube<->vector handshake on `ub_dA`, DEPTH-4** — the event structure of
     kal/gdr_bwd_wy_matmuls/staged.py's hand-off (e0 = cube "dA drained to UB", e1 = vector "UB free
     again"), but with FOUR rotating UB slots instead of one.  The vector bootstraps e1
     UB_DEPTH times before its loop and the cube drains UB_DEPTH trailing e1 waits after
     its loop, so the counters balance at zero for any per-core task count including 0
     and 1.  Both sections advance the same TileGroup with `.next()` once per tid, so
     their rotation stays in phase.  Both sub-blocks execute the handshake (no guard),
     which is what keeps `wait_vec` balanced.  Two further placement choices: e0 is
     signalled immediately after the drain — before b4.4 — so glue-1 overlaps the last
     cube matmul group; and the cube's e1 wait sits immediately before the drain, NOT at
     the top of the tid loop, so every GM load and three of the four matmul groups run
     before the cube can stall.

     Depth is load-bearing and was measured, not assumed.  Production shape
     (B=1,T=2048,H=8,K=V=128 -> 256 tasks, 8 tids per core):

       | handoff config                        | Task Duration | aic_mac | aic_total_cycles |
       |---------------------------------------|---------------|---------|------------------|
       | depth-1, e1 wait at top of tid loop   |    1447.0 us  |  89.0%  |  2 381 888       |
       | depth-2, e1 wait before the drain     |    1374.9 us  |  94.2%  |  2 230 584       |
       | depth-4, e1 wait before the drain     |     693.5 us  |  94.4%  |  2 224 275       |
       | (M1 cube-only reference, no vector)   |     689.5 us  |  95.3%  |  2 204 236       |

     depth-1 strictly alternates cube and vector and costs 2.1x.  depth-2 removes the
     stall cycles the counters attribute to the cube but still leaves the wall time 2x —
     the cube needs about three tids of slack before the hand-off stops gating it.  At
     depth-4 Module 2 is free: 693.5 vs 689.5 us for the same cube work, i.e. glue-1
     costs +0.6% wall time, and the cube cycle count is back within 0.9% of M1's.
     The DESIGN §8 shapes give only 1 tid per core, so they never showed this at all
     (they sat within 1-4% of M1 at every depth) — it is visible only at production
     scale.  The preamble authorises depth-N double-buffering as an implementation
     lever; UB cost is 4 x 16 KB = 65536 B of 253952 B (25.8%).  Math unchanged.
 V3  `set_cross_core` release pipe on the vector side is `MTE3` (the last op reading
     `ub_dA` is the GM store), not staged.py's `V` (there the release followed the VF
     because a different event carried the later L1 write).

Declared deviations inherited from Module 1 (re-verified, unchanged; full rationale in
modules/test_gdr_bwd_wy_matmuls_module1.py):
 T1  GM store offset lists repaired to `len(offsets) == dst.ndim` with the column offset
     on the last axis — the transpiled product emits 2-entry lists for 5-D tensors and
     puts the +64 column offset on the h axis.  Same defect still present on the dkb/dkt
     stores in staged.py's cube-C portion (Modules 3/5 will need it too).
 T2  L0A/L0B declared at the per-mmad [64,64] shape (32 KB each, 50% of cap) instead of
     the transpiled aliased [64,128]/[128,64]+[64,64] views that pinned them at 100%.
     Restores the DESIGN §5 projection.
 T3  Matmul + fixpipe always on the full [64,64] L0C; only the contraction axis and the
     final store are narrowed (official a5 matmul sample).
 T4  `pl.move(..., offset=[m, n])` as the documented keyword.
 T5  B/H/N/Kd/Vd derived in-kernel from tensor shapes.
"""

import logging
import os
import sys

# dev-only path setup so the parent-dir goldens and this dir's staged golden resolve
# when the file is run directly.
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

# ── compile-time constants (DESIGN §2; kal/gdr_bwd_wy_matmuls/gdr_bwd_wy_matmuls_kal.py) ──
C = 64        # frozen chunk size: row dim of every [C,*] tile, both dims of attn/dA
CH = C // 2   # rows per AIV sub-block after the DualModeSplitM drain (deviation V1)
KL0 = 64      # contraction-axis split step (splitk)
TK = 64       # output-subblock width along K (splitn)
KMAX = 128    # K upper bound = physical L1 tile width (runtime Kd <= KMAX)
VMAX = 128    # V upper bound

# ── L1 address map (verbatim from kal/gdr_bwd_wy_matmuls/staged.py so Modules 3-5 append unchanged) ──
L1_DW = 0x00000      # [C, KMAX] fp32   32768
L1_KBT = 0x08000     # [KMAX, C] fp32   32768
L1_ATTNT = 0x10000   # [C, C]    fp32   16384
L1_DV2 = 0x14000     # [C, VMAX] fp32   32768
L1_VB = 0x1C000      # [VMAX, C] fp32   32768
L1_DAB = 0x24000     # [C, C]    fp32   16384   Module 3: masked dA (cube_B b4.5 LEFT)
L1_M16 = 0x28000     # [C, C]    fp32   16384   Module 3: b4.5->b4.6 bridge tile
L1_DAC = 0x2C000     # [C, C]    fp32   16384   Module 5: d_a natural (b4.7 LEFT)
L1_DATC = 0x30000    # [C, C]    fp32   16384   Module 5: d_a^T  (b4.8 LEFT)
L1_KC = 0x34000      # [C, KMAX] fp32   32768   Module 5: k_f32  (b4.7 RIGHT)
L1_KBE = 0x3C000     # [C, KMAX] fp32   32768   Module 5: k_beta (b4.8 RIGHT)
                     #                          -> L1 high-water 0x44000 = 278528 B
                     #                          -> L1 high-water 0x2C000 = 180224 B
# ── L0 address maps ──
L0A_0, L0A_1 = 0x0000, 0x4000    # 2 × [64,64] fp32 = 32768 B  (50% of 64 KB)
L0B_0, L0B_1 = 0x0000, 0x4000
L0C_DA = 0x0000                  # [C, C] fp32     16384
L0C_WIDE = 0x4000                # 2 × [C, C] fp32 32768
L0C_B5 = 0xC000                  # [C, C] fp32     16384  Module 3: b4.5  T = dA @ A^T
L0C_BB = 0x10000                 # [C, C] fp32     16384  Module 3: b4.6  dA_b = A^T @ T
L0C_WIDEC = 0x14000              # 2 × [C, C] fp32 32768  Module 5: dkb, then dkt
                                 #                        -> L0C high-water 0x1C000 = 114688 B
                                 #                        -> L0C high-water 0x14000 = 81920 B
# ── UB address map (Module 2; kal/gdr_bwd_wy_matmuls/staged.py g_ub_dA base) ──
UB_DA_0 = 0x00000                # [C, C] fp32 slot 0  16384
UB_DA_1 = 0x04000                # [C, C] fp32 slot 1  16384
UB_DA_2 = 0x08000                # [C, C] fp32 slot 2  16384
UB_DA_3 = 0x0C000                # [C, C] fp32 slot 3  16384
UB_DEPTH = 4
UB_MID = 0x10000                 # [C, C] fp32 NZ      16384  Module 3 b4.5->b4.6 bridge
UB_DAB_0 = 0x14000               # [C, C] fp32 slot 0  16384  Module 4 glue-2 input
UB_DAB_1 = 0x18000               # [C, C] fp32 slot 1  16384
UB_DECAY = 0x1C000               # [1, C] fp32           256
UB_BETA = 0x1C100                # [1, C] fp32           256
UB_K = 0x1D000                   # [C, KMAX] fp32      32768
UB_KBETA = 0x25000               # [C, KMAX] fp32      32768
UB_DAB_2 = 0x2D000               # [C, C] fp32 slot 2  16384
UB_DAB_3 = 0x31000               # [C, C] fp32 slot 3  16384
                                 #                     -> UB high-water 0x35000 = 217088 B
                                 #                     -> UB high-water 0x14000 = 81920 B

# ── cross-core event ids for the cube<->vector dA hand-off (deviation V2) ──
EV_DA_READY = 0      # cube -> vector: dA drained into ub_dA
EV_DA_FREE = 1       # vector -> cube: ub_dA consumed, may be overwritten
EV_G1_GM = 2         # vector -> cube: dA_g1[tid] committed to GM, cube_B may read it
EV_MID_READY = 3     # cube -> vector: b4.5 result T drained into ub_mid (NZ)
EV_MID_DONE = 4      # vector -> cube: T relayed into l1_m16, ub_mid free again
EV_DAB_READY = 5     # cube -> vector: dA_b drained into ub_dab (Module 4 input)
EV_DAB_FREE = 6      # vector -> cube: ub_dab consumed, may be overwritten
DAB_DEPTH = 4
EV_DA_GM = 7         # vector -> cube: d_a[tid] committed to GM (gates b4.7)
EV_KB_GM = 8         # vector -> cube: k_beta[tid] committed to GM (gates b4.8)


# =============================================================================
#   Vector function — Module 2 glue-1, banded over one AIV sub-block's rows
# =============================================================================
@pl.vector_function
def glue1_strict_low_banded(dA, n_rows: pl.DT_INT64, row0: pl.DT_INT64):
    """dA[r, j] <- dA[r, j] if j < (row0 + r) else 0, for local rows r in [0, n_rows).

    `row0` biases the local row index to the GLOBAL dA row, so the strict-lower
    boundary is correct in each sub-block's band (deviation V1).  One [C,C] fp32 row is
    exactly one 64-lane register, so the whole row is one load / select / store.

    Loop-variable discipline (preamble, gdr_fwd_train_wy `dfac_vf`): `r` is the loop
    variable and is never reassigned; every address below is built from `r` directly,
    never from a value that a vector register was stored into.
    """
    keep_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, keep_all, dtype=pl.DT_FP32)
    for r in pl.range(0, n_rows, 1):
        reg = vf.load_align(dA, r * C, dtype=pl.DT_FP32)
        keep = vf.update_mask(pl.max(0, row0 + r), dtype=pl.DT_FP32)
        out = vf.select(reg, zero, keep)
        vf.store_align(dA + r * C, out, keep_all)


@pl.vector_function
def glue2_decay_negate_banded(dA, decay, n_rows: pl.DT_INT64, row0: pl.DT_INT64):
    """Module 4 glue-2, banded over one AIV sub-block's rows.

    decay_fac[r, j] = exp(min(decay_{row0+r} - decay_j, 0)) on j < row0+r, else 0
    dA[r, j] <- -(dA[r, j] * decay_fac[r, j])                       *** FINAL d_a ***

    `dA` holds this sub-block's 32 rows at LOCAL offsets; `decay` is the full [1,C] row,
    so decay_i is indexed GLOBALLY at row0+r.  Masking before exp (vf.mins at 0) keeps
    the upper triangle out of exp, so no overflow -- then select re-masks to strict
    lower.  Loop variable `r` is never reassigned; all addresses derive from it."""
    m = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, m, dtype=pl.DT_FP32)
    d_row = vf.load_align(decay, 0, dtype=pl.DT_FP32)
    for r in pl.range(0, n_rows, 1):
        di = vf.load_align(decay, row0 + r, dist=pl.LoadDist.BRC_B32)
        gdiff = vf.sub(di, d_row, m)
        gdiff = vf.mins(gdiff, 0.0, m)
        ef = vf.exp(gdiff, m)
        reg = vf.load_align(dA, r * C, dtype=pl.DT_FP32)
        reg = vf.mul(reg, ef, m)
        reg = vf.muls(reg, -1.0, m)
        keep = vf.update_mask(pl.max(0, row0 + r), dtype=pl.DT_FP32)
        out = vf.select(reg, zero, keep)
        vf.store_align(dA + r * C, out, m)


@pl.vector_function
def form_kbeta_banded(k, beta, kbeta, n_rows: pl.DT_INT64, row0: pl.DT_INT64):
    """k_beta[row0+r, :] = k_f32[row0+r, :] * beta_act[row0+r], written to LOCAL row r.

    `k` / `beta` are whole-tile (both sub-blocks load them redundantly, which is
    idempotent); the result lands at local row r so the sub-block can store its band
    straight from the tile base.  A vector register holds only C=64 fp32 lanes, so a
    K<=KMAX=128 row is processed in two 64-wide halves -- a single 0:KMAX load would
    silently drop lanes 64:128 (kal/gdr_bwd_wy_matmuls/NOTES.md finding 4)."""
    m = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, n_rows, 1):
        bb = vf.load_align(beta, row0 + r, dist=pl.LoadDist.BRC_B32)
        k0 = vf.load_align(k, (row0 + r) * KMAX, dtype=pl.DT_FP32)
        k1 = vf.load_align(k, (row0 + r) * KMAX + C, dtype=pl.DT_FP32)
        k0 = vf.mul(k0, bb, m)
        k1 = vf.mul(k1, bb, m)
        vf.store_align(kbeta + r * KMAX, k0, m)
        vf.store_align(kbeta + r * KMAX + C, k1, m)


# =============================================================================
#   Single @pl.jit kernel — Modules 1+2 (MIX: one cube section + one vector section)
# =============================================================================
@pl.jit(auto_mutex=True)
def gdr_bwd_wy_matmuls_module12345_kernel(
    attn: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k_f32: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    beta_act: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    decay: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    v_b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    kbeta_t: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    dw: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    dv2: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    dkbg: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    dvb: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    dA_a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    dA_g1: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    dA_b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    d_a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k_beta: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    dkb: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    dkt: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    # ---- L1 (Mat) operand tiles; mutex ids kept from kal/gdr_bwd_wy_matmuls/staged.py (merged config) ----
    # PERF (cube-08 mutex audit): dw / dv2 / vb are three disjoint L1 operands
    # (L1_DW, L1_DV2, L1_VB -- no address overlap) whose GM->L1 loads all sit at the
    # top of the tid body, yet the transpiler put all three on mutex id 9 to economise.
    # auto_mutex then serialised 96 KB of independent MTE2 into one chain, on the pipe
    # (L1 MTE2<->MTE1) the card names as the one where sharing actually hurts.  The
    # kernel was at 32/32 ids, so the headroom was made by MERGING two pairs that are
    # phase-disjoint within a tid and already ordered by program order, which adds no
    # effective constraint:  dAb (read at b4.5) + dAc (loaded later, for b4.7) -> id 15;
    # kc (read in the cj loop) + kbe (loaded after it, for the tj loop) -> id 28.
    # Merging can only ADD ordering, so it cannot break correctness; splitting is safe
    # because the three addresses never overlapped and so never needed a shared mutex.
    g_l1_dw = pl.make_tile_group(
        type=pl.TileType(shape=[C, KMAX], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_DW, mutex_ids=[9])
    g_l1_kbT = pl.make_tile_group(
        type=pl.TileType(shape=[KMAX, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_KBT, mutex_ids=[10])
    g_l1_attnT = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_ATTNT, mutex_ids=[11])
    g_l1_dv2 = pl.make_tile_group(
        type=pl.TileType(shape=[C, VMAX], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_DV2, mutex_ids=[26])
    g_l1_vb = pl.make_tile_group(
        type=pl.TileType(shape=[VMAX, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_VB, mutex_ids=[29])

    # ---- L0A / L0B: ping-pong at the per-mmad [64,64] shape (deviation T2) ----
    g_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[C, KL0], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1]),
        addrs=[L0A_0, L0A_1], mutex_ids=[0, 1])
    g_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[KL0, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1]),
        addrs=[L0B_0, L0B_1], mutex_ids=[2, 3])

    # ---- L0C: dA accumulator + a 2-slot rotating output-subblock accumulator ----
    g_l0c_dA = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]),
        addrs=L0C_DA, mutex_ids=[12])
    g_l0c_wide = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]),
        addrs=L0C_WIDE, mutex_ids=[13, 14])

    # ---- UB: the cube->vector dA hand-off tile (Module 2) ----
    g_ub_dA = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=[UB_DA_0, UB_DA_1, UB_DA_2, UB_DA_3], mutex_ids=[4, 5, 6, 7])

    # ---- Module 3 (cube_B): DEDICATED L1 / L0C per the DESIGN M4 finding ----
    g_l1_dAb = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_DAB, mutex_ids=[15])
    g_l1_m16 = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_M16, mutex_ids=[16])
    # UB relay for the b4.5->b4.6 bridge.  Declared NZ so the drain is Acc(NZ)->Vec(NZ)
    # and the relay is Vec(NZ)->Mat(NZ): a UB tile holding ND may NOT be copied into an
    # NZ/ZN L1 tile (move API doc), and an fp32 L1 tile cannot be ND (TileType doc:
    # L1 defaults to NZ, may be ZN, and ND only for 64-bit dtypes).
    g_ub_mid = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                         valid_shape=[-1, -1]),
        addrs=UB_MID, mutex_ids=[19])

    # ---- Module 4 (glue-2) UB tiles ----
    g_ub_dab = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=[UB_DAB_0, UB_DAB_1, UB_DAB_2, UB_DAB_3], mutex_ids=[20, 21, 22, 23])
    g_ub_decay = pl.make_tile_group(
        type=pl.TileType(shape=[1, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=UB_DECAY, mutex_ids=[8])
    g_ub_beta = pl.make_tile_group(
        type=pl.TileType(shape=[1, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=UB_BETA, mutex_ids=[19])
    g_ub_k = pl.make_tile_group(
        type=pl.TileType(shape=[C, KMAX], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=UB_K, mutex_ids=[24])
    g_ub_kbeta = pl.make_tile_group(
        type=pl.TileType(shape=[C, KMAX], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=UB_KBETA, mutex_ids=[25])

    # ---- Module 5 (cube_C): DEDICATED L1 / L0C ----
    g_l1_dAc = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_DAC, mutex_ids=[15])
    g_l1_dATc = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_DATC, mutex_ids=[27])
    g_l1_kc = pl.make_tile_group(
        type=pl.TileType(shape=[C, KMAX], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_KC, mutex_ids=[28])
    g_l1_kbe = pl.make_tile_group(
        type=pl.TileType(shape=[C, KMAX], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=L1_KBE, mutex_ids=[28])
    g_l0c_wideC = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]),
        addrs=L0C_WIDEC, mutex_ids=[30, 31])
    g_l0c_b5 = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]),
        addrs=L0C_B5, mutex_ids=[17])
    g_l0c_bb = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]),
        addrs=L0C_BB, mutex_ids=[18])

    # =====================================================================
    #   Cube section (Module 1, verbatim from module1) + the dA drain
    # =====================================================================
    with pl.section_cube():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx() // pl.get_subblock_num()

        B = dw.shape[0]
        H = dw.shape[1]
        N = dw.shape[2]
        Kd = dw.shape[4]
        Vd = dv2.shape[4]

        HN = H * N
        tasks = B * HN
        n_k = (Kd + KL0 - 1) // KL0      # contraction / output subblocks along K
        n_v = (Vd + KL0 - 1) // KL0      # contraction / output subblocks along V

        # DESIGN §5 flat contiguous BHN grid (verbatim from kal/gdr_bwd_wy_matmuls/staged.py).
        per = (tasks + num_cores - 1) // num_cores
        begin = per * core_id
        end = pl.min(begin + per, tasks)

        t_dw = g_l1_dw.current()
        t_kbT = g_l1_kbT.current()
        t_attnT = g_l1_attnT.current()
        t_dv2 = g_l1_dv2.current()
        t_vb = g_l1_vb.current()
        t_dA = g_l0c_dA.current()
        t_dAb = g_l1_dAb.current()
        t_m16 = g_l1_m16.current()
        t_b5 = g_l0c_b5.current()
        t_ubmid = g_ub_mid.current()
        t_dAc = g_l1_dAc.current()
        t_dATc = g_l1_dATc.current()
        t_kc = g_l1_kc.current()
        t_kbe = g_l1_kbe.current()
        t_bb = g_l0c_bb.current()

        for tid in pl.range(begin, end, 1):
            b = tid // HN
            rem = tid % HN
            h = rem // N
            n = rem % N

            # ---------------- GM -> L1 operands ----------------
            pl.set_validshape(t_dw, [C, Kd])
            pl.load(t_dw, dw, [b, h, n, 0, 0])
            pl.set_validshape(t_kbT, [Kd, C])
            pl.load(t_kbT, kbeta_t, [b, h, n, 0, 0])
            # attn^T via a descending-order (transposing) load; L0A LEFT of b4.2 / b4.4
            pl.set_validshape(t_attnT, [C, C])
            pl.load(t_attnT, attn, [b, h, n, 0, 0], order=[4, 3])
            pl.set_validshape(t_dv2, [C, Vd])
            pl.load(t_dv2, dv2, [b, h, n, 0, 0])
            # v_b^T via a descending-order load; L0B RIGHT of b4.3
            pl.set_validshape(t_vb, [Vd, C])
            pl.load(t_vb, v_b, [b, h, n, 0, 0], order=[4, 3])

            # ---------------- b4.1  dA = dw @ kbeta_t  (splitk over Kd) ----------------
            for ki in pl.range(0, n_k):
                ko = ki * KL0
                sub_k = pl.min(KL0, Kd - ko)
                la = g_l0a.next()
                lb = g_l0b.next()
                pl.set_validshape(t_dw, [C, sub_k])
                pl.move(la, t_dw, offset=[0, ko])
                pl.set_validshape(la, [C, sub_k])
                pl.set_validshape(t_kbT, [sub_k, C])
                pl.move(lb, t_kbT, offset=[ko, 0])
                pl.set_validshape(lb, [sub_k, C])
                pl.set_validshape(t_dA, [C, C])
                if ki == 0:
                    pl.matmul(t_dA, la, lb)
                else:
                    pl.matmul_acc(t_dA, t_dA, la, lb)

            # ------------- b4.3  dA += dv2 @ v_b^T  (splitk over Vd) -------------
            for vi in pl.range(0, n_v):
                vo = vi * KL0
                sub_k = pl.min(KL0, Vd - vo)
                la = g_l0a.next()
                lb = g_l0b.next()
                pl.set_validshape(t_dv2, [C, sub_k])
                pl.move(la, t_dv2, offset=[0, vo])
                pl.set_validshape(la, [C, sub_k])
                pl.set_validshape(t_vb, [sub_k, C])
                pl.move(lb, t_vb, offset=[vo, 0])
                pl.set_validshape(lb, [sub_k, C])
                pl.set_validshape(t_dA, [C, C])
                pl.matmul_acc(t_dA, t_dA, la, lb)

            # dA (raw) is a Module-1 output -> GM
            pl.set_validshape(t_dA, [C, C])
            pl.store(dA_a, t_dA, [b, h, n, 0, 0])

            # [V2] only NOW can the cube stall: every GM load and three of the four
            # matmul groups are already done, so the wait is off the critical path.
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_DA_FREE,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            # [V1/V2] drain dA to BOTH AIV sub-blocks (Vec0 rows 0..31, Vec1 rows
            # 32..63) and release the vector immediately, so glue-1 overlaps b4.2.
            t_ub = g_ub_dA.next()
            pl.set_validshape(t_dA, [C, C])
            pl.move(t_ub, t_dA, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=EV_DA_READY,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # [P1] b4.2 sits HERE, after the dA drain, so its two matmuls and
            # stores cover the EV_G1_GM wait below instead of the cube idling
            # through the vector's glue-1 latency.
            # ------------- b4.2  dkbg = attn^T @ dw  (splitn over Kd) -------------
            for kj in pl.range(0, n_k):
                col_off = kj * TK
                la = g_l0a.next()
                lb = g_l0b.next()
                pl.set_validshape(t_attnT, [C, C])
                pl.move(la, t_attnT, offset=[0, 0])
                pl.set_validshape(la, [C, C])
                # FULL-width N move: the pad columns beyond Kd hold stale L1 and are
                # never stored (deviation T3).
                pl.set_validshape(t_dw, [C, KL0])
                pl.move(lb, t_dw, offset=[0, col_off])
                pl.set_validshape(lb, [C, C])
                wc = g_l0c_wide.next()
                pl.set_validshape(wc, [C, C])
                pl.matmul(wc, la, lb)
                pl.set_validshape(wc, [C, pl.min(TK, Kd - col_off)])
                pl.store(dkbg, wc, [b, h, n, 0, col_off])


            # ================= Module 3 (cube_B): b4.5 -> b4.6 =================
            # [W1] wait until BOTH AIV sub-blocks committed dA_g1[tid] to GM.
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_G1_GM,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            # [W1] masked dA read back from GM (it is a required output anyway) -> L1
            pl.set_validshape(t_dAb, [C, C])
            pl.load(t_dAb, dA_g1, [b, h, n, 0, 0])

            # b4.5  T = dA @ A^T.   L0A = dA (natural), L0B = A^T.
            # [W3] t_attnT already holds A^T for this tid (loaded once, for b4.2/b4.4)
            # and is read-only within the tid, so cube_B reuses it instead of loading
            # attn twice more as kal/gdr_bwd_wy_matmuls/staged.py does.
            la5 = g_l0a.next()
            lb5 = g_l0b.next()
            pl.set_validshape(t_dAb, [C, C])
            pl.move(la5, t_dAb, offset=[0, 0])
            pl.set_validshape(la5, [C, C])
            pl.set_validshape(t_attnT, [C, C])
            pl.move(lb5, t_attnT, offset=[0, 0])
            pl.set_validshape(lb5, [C, C])
            pl.set_validshape(t_b5, [C, C])
            pl.matmul(t_b5, la5, lb5)

            # [W2] b4.5 -> b4.6 bridge: L0C -> UB -> L1 across the vector lane.
            # Acc->Mat does NOT exist on this build (the parser lists the legal paths as
            # Mat->Left/Right/Scaling/Bias/ScaleLeft/ScaleRight, Acc->Vec, Vec->Vec,
            # Vec->Mat), so T leaves L0C into an NZ UB tile and the vector relays it
            # into L1.  EV_MID_READY / EV_MID_DONE carry the two crossings.
            pl.set_validshape(t_b5, [C, C])
            pl.set_validshape(t_ubmid, [C, C])
            pl.move(t_ubmid, t_b5, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=EV_MID_READY,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # [P1] b4.4 sits HERE, after EV_MID_READY is signalled, so its two
            # matmuls and stores cover the EV_MID_DONE wait below instead of the
            # cube idling through the vector's bridge relay.
            # ------------- b4.4  dvb = attn^T @ dv2  (splitn over Vd) -------------
            for vj in pl.range(0, n_v):
                col_off = vj * TK
                la = g_l0a.next()
                lb = g_l0b.next()
                pl.set_validshape(t_attnT, [C, C])
                pl.move(la, t_attnT, offset=[0, 0])
                pl.set_validshape(la, [C, C])
                pl.set_validshape(t_dv2, [C, KL0])
                pl.move(lb, t_dv2, offset=[0, col_off])
                pl.set_validshape(lb, [C, C])
                wc = g_l0c_wide.next()
                pl.set_validshape(wc, [C, C])
                pl.matmul(wc, la, lb)
                pl.set_validshape(wc, [C, pl.min(TK, Vd - col_off)])
                pl.store(dvb, wc, [b, h, n, 0, col_off])

            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_MID_DONE,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # b4.6  dA_b = A^T @ T.  L0A = A^T, L0B = T (the bridged tile).
            la6 = g_l0a.next()
            lb6 = g_l0b.next()
            pl.set_validshape(t_attnT, [C, C])
            pl.move(la6, t_attnT, offset=[0, 0])
            pl.set_validshape(la6, [C, C])
            pl.set_validshape(t_m16, [C, C])
            pl.move(lb6, t_m16, offset=[0, 0])
            pl.set_validshape(lb6, [C, C])
            pl.set_validshape(t_bb, [C, C])
            pl.matmul(t_bb, la6, lb6)

            # [T1] full 5-entry offset list, column offset on the LAST axis
            pl.set_validshape(t_bb, [C, C])
            pl.store(dA_b, t_bb, [b, h, n, 0, 0])

            # ============ Module 4 hand-off: dA_b -> both AIV sub-blocks ============
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_DAB_FREE,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            t_dab = g_ub_dab.next()
            pl.set_validshape(t_bb, [C, C])
            pl.move(t_dab, t_bb, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=EV_DAB_READY,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # ================= Module 5 (cube_C): b4.7 -> b4.8 =================
            # [P2] k_f32 has NO dependency on the vector, so its L1 load is issued
            # BEFORE the EV_G2_GM wait and covers part of that window.
            pl.set_validshape(t_kc, [C, Kd])
            pl.load(t_kc, k_f32, [b, h, n, 0, 0])
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_DA_GM,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            # d_a and k_beta are both REQUIRED outputs, so cube_C reads them straight
            # back from GM -- no scratch workspace (same reasoning as W1).
            pl.set_validshape(t_dAc, [C, C])
            pl.load(t_dAc, d_a, [b, h, n, 0, 0])
            pl.set_validshape(t_dATc, [C, C])
            pl.load(t_dATc, d_a, [b, h, n, 0, 0], order=[4, 3])

            # b4.7  dkb = d_a @ k_f32   (splitn over Kd; full-N matmul, clipped store)
            for cj in pl.range(0, n_k):
                col_off = cj * TK
                la7 = g_l0a.next()
                lb7 = g_l0b.next()
                pl.set_validshape(t_dAc, [C, C])
                pl.move(la7, t_dAc, offset=[0, 0])
                pl.set_validshape(la7, [C, C])
                pl.set_validshape(t_kc, [C, KL0])
                pl.move(lb7, t_kc, offset=[0, col_off])
                pl.set_validshape(lb7, [C, C])
                w7 = g_l0c_wideC.next()
                pl.set_validshape(w7, [C, C])
                pl.matmul(w7, la7, lb7)
                pl.set_validshape(w7, [C, pl.min(TK, Kd - col_off)])
                pl.store(dkb, w7, [b, h, n, 0, col_off])

            # [P3] only the k_beta half of the hand-off is waited on here: b4.7 above
            # ran while the vector was still building k_beta.
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_KB_GM,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.set_validshape(t_kbe, [C, Kd])
            pl.load(t_kbe, k_beta, [b, h, n, 0, 0])

            # b4.8  dkt = d_a^T @ k_beta   (d_a^T reaches L0A by transpose-load)
            for tj in pl.range(0, n_k):
                col_off = tj * TK
                la8 = g_l0a.next()
                lb8 = g_l0b.next()
                pl.set_validshape(t_dATc, [C, C])
                pl.move(la8, t_dATc, offset=[0, 0])
                pl.set_validshape(la8, [C, C])
                pl.set_validshape(t_kbe, [C, KL0])
                pl.move(lb8, t_kbe, offset=[0, col_off])
                pl.set_validshape(lb8, [C, C])
                w8 = g_l0c_wideC.next()
                pl.set_validshape(w8, [C, C])
                pl.matmul(w8, la8, lb8)
                pl.set_validshape(w8, [C, pl.min(TK, Kd - col_off)])
                pl.store(dkt, w8, [b, h, n, 0, col_off])

        # [V2] drain the vector's UB_DEPTH bootstrap credits so EV_DA_FREE ends at zero
        for _d in pl.range(0, UB_DEPTH, 1):
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_DA_FREE,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        for _d2 in pl.range(0, DAB_DEPTH, 1):
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_DAB_FREE,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

    # =====================================================================
    #   Vector section (Module 2, glue-1) — both AIV sub-blocks, row-banded
    # =====================================================================
    with pl.section_vector():
        num_cores2 = pl.get_block_num()
        core_id2 = pl.get_block_idx() // pl.get_subblock_num()
        sub = pl.get_subblock_idx()

        B2 = dw.shape[0]
        H2 = dw.shape[1]
        N2 = dw.shape[2]
        HN2 = H2 * N2
        tasks2 = B2 * HN2
        Kd2 = dw.shape[4]
        per2 = (tasks2 + num_cores2 - 1) // num_cores2
        begin2 = per2 * core_id2
        end2 = pl.min(begin2 + per2, tasks2)

        # this sub-block's band of dA after the DualModeSplitM drain
        row0 = sub * CH

        t_ubmid2 = g_ub_mid.current()
        t_m16v = g_l1_m16.current()
        t_dec = g_ub_decay.current()
        t_bet = g_ub_beta.current()
        t_kk = g_ub_k.current()
        t_kb = g_ub_kbeta.current()

        # [V2] bootstrap: all UB_DEPTH ub_dA slots start free, so the cube
        # may drain tid+1 while the vector is still masking tid.
        for _d in pl.range(0, UB_DEPTH, 1):
            pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=EV_DA_FREE,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        for _d2 in pl.range(0, DAB_DEPTH, 1):
            pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=EV_DAB_FREE,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

        for tid in pl.range(begin2, end2, 1):
            t_ub2 = g_ub_dA.next()
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_DA_READY,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            b = tid // HN2
            rem = tid % HN2
            h = rem // N2
            n = rem % N2

            # glue-1 on this sub-block's own 32 rows, then store its own GM band.
            # Disjoint bands -> no shared write, no sub-block guard, nothing can tear.
            pl.set_validshape(t_ub2, [CH, C])
            glue1_strict_low_banded(t_ub2, CH, row0)
            pl.set_validshape(t_ub2, [CH, C])
            pl.store(dA_g1, t_ub2, [b, h, n, row0, 0])

            # [V3] release on MTE3: the GM store is the last reader of ub_dA
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=EV_DA_FREE,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            # [W1] dA_g1[tid] is now committed to GM -> cube_B may read it back
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=EV_G1_GM,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # ---- [W2] relay the b4.5 result T from UB into L1 for b4.6 ----
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_MID_READY,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            # The drain was SingleModeVec0, so only Vec0 holds T; the copy is guarded to
            # sub-block 0.  Unlike the pattern the preamble flags, NOTHING is computed on
            # a half-populated tile here -- sub-block 1 simply does not participate in a
            # pure byte copy -- and the guard is what prevents two lanes tearing one
            # shared L1 region.  The SIGNAL below is deliberately OUTSIDE the guard:
            # a cross-core wait is satisfied only when EVERY vector lane has signalled,
            # so a guarded signal would hang the cube.
            if pl.get_subblock_idx() < 1:
                pl.set_validshape(t_ubmid2, [C, C])
                pl.set_validshape(t_m16v, [C, C])
                pl.move(t_m16v, t_ubmid2)
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=EV_MID_DONE,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # ================= Module 4 (glue-2), banded like glue-1 =================
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=EV_DAB_READY,
                                      sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            t_dab2 = g_ub_dab.next()
            pl.set_validshape(t_dec, [1, C])
            pl.load(t_dec, decay, [b, h, n, 0])
            pl.set_validshape(t_dab2, [CH, C])
            glue2_decay_negate_banded(t_dab2, t_dec, CH, row0)
            pl.set_validshape(t_dab2, [CH, C])
            pl.store(d_a, t_dab2, [b, h, n, row0, 0])
            # [P3] d_a is complete -> cube_C may start b4.7; and glue-2 is done with
            # ub_dab, so release the slot now rather than after form_kbeta.
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=EV_DA_GM,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=EV_DAB_FREE,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

            # k_beta = k_f32 * beta_act (row-local -> same banding)
            pl.set_validshape(t_kk, [C, Kd2])
            pl.load(t_kk, k_f32, [b, h, n, 0, 0])
            pl.set_validshape(t_bet, [1, C])
            pl.load(t_bet, beta_act, [b, h, n, 0])
            pl.set_validshape(t_kb, [CH, Kd2])
            form_kbeta_banded(t_kk, t_bet, t_kb, CH, row0)
            pl.set_validshape(t_kb, [CH, Kd2])
            pl.store(k_beta, t_kb, [b, h, n, row0, 0])

            # [P3] k_beta for this tid is now in GM -> cube_C may run b4.8
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=EV_KB_GM,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)


# =============================================================================
#   Wrapper — HARD contract name gdr_bwd_wy_matmuls_wrapper
# =============================================================================
def gdr_bwd_wy_matmuls_wrapper(attn, k_f32, v_b, kbeta_t, beta_act, decay, dw, dv2):
    """OP6a entry: allocate outputs + ONE kernel launch.

    Signature mirrors SPEC / module_interfaces.yaml primary_inputs.
    Returns the five OP6a final_outputs (dkbg, dvb, dkb, dkt, d_a), matching
    gdr_bwd_wy_matmuls_golden_cpu's return order.
    """
    B, H, N, _C, K = dw.shape
    V = dv2.shape[4]
    dev = dw.device

    dkbg = torch.empty((B, H, N, C, K), dtype=torch.float32, device=dev)
    dvb = torch.empty((B, H, N, C, V), dtype=torch.float32, device=dev)
    dkb = torch.empty((B, H, N, C, K), dtype=torch.float32, device=dev)
    dkt = torch.empty((B, H, N, C, K), dtype=torch.float32, device=dev)
    d_a = torch.empty((B, H, N, C, C), dtype=torch.float32, device=dev)
    # internal staged tensors (Module hand-offs carried through GM)
    dA_a = torch.empty((B, H, N, C, C), dtype=torch.float32, device=dev)
    dA_g1 = torch.empty((B, H, N, C, C), dtype=torch.float32, device=dev)
    dA_b = torch.empty((B, H, N, C, C), dtype=torch.float32, device=dev)
    k_beta = torch.empty((B, H, N, C, K), dtype=torch.float32, device=dev)

    # DESIGN §5: block_dim = min(max_blocks, tasks); max_blocks from the runtime
    # platform (A5 SKU .ini), never hard-coded.  Pure host scalar arithmetic.
    tasks = B * H * N
    max_blocks = get_platform_info().core_num or 1
    num_cores = max(1, min(tasks, max_blocks))

    gdr_bwd_wy_matmuls_module12345_kernel[None, num_cores](
        attn, k_f32, beta_act, decay, v_b, kbeta_t, dw, dv2,
        dkbg, dvb, dA_a, dA_g1, dA_b, d_a, k_beta, dkb, dkt)
    torch.npu.synchronize()

    return dkbg, dvb, dkb, dkt, d_a

