# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""gdr_fwd_train_scan -- Stage 4 delivery (single fusion module = FULL op).

DESIGN v2 (md5 ede2612c7cb597f9b2d71befa941138e) RE-TILING.  Math, SPEC, goldens
and the three outputs' shape/dtype/GM layout are unchanged from v1 (git 6ffbf3c,
hardware 7/7 PASS); this file re-homes that verified kernel onto the v2 geometry:

  * task grid (b, h, v_tile_group)  -- V columns are independent (DESIGN sec.0)
  * unified [64,64] fp32 tiles; K split into ktiles=nkob=ceil(K/64) (K,V <= 256)
  * AccToVecMode.DualModeSplitM row banding -- BOTH AIV sub-blocks work
  * s_hist is the cube's GM source for S; v_new is the cube's GM source for v_new
    (state_gm / vnew_gm / kft_gm workspaces deleted)
  * k^T materialised by a transpose-LOAD into a ZN L1 tile (no GM pre-transpose)
  * STREAMS in {1,2}: two V streams interleaved inside a core, credit-counted

v1 hardware fixes retained verbatim (DESIGN sec.12 row 16):
  D1 pl.store(o, ..., order=[1,3])        D2 no host layout op (k^T in kernel)
  D3 B is pl.DYNAMIC                      D4 no round_mode on VF casts
  D5 o fp32->bf16 via tile-op pl.cast(CAST_ROUND)   D6 no redundant vf.mem_bar
  + torch.npu.set_device() before every launch.

DECLARED DEVIATIONS from DESIGN v2 (mechanism only; math/契约 unchanged):
  V-A  DESIGN sec.1 C5 / sec.11 table 2 prescribe
       `pl.move(l0a, kf_l1, order=[1,0])`.  `pypto_pro.language.move` has NO
       `order` parameter (API doc SIMD-API/memory_data_movement/move.md prototype).
       The documented a5 mechanism for the same effect is the transpose-LOAD:
       L1 tile declared shape=[M,Kc] layout=pl.ZN + `pl.load(..., order=<descending>)`
       (SIMD-API/memory_data_movement/load.md "Cube侧转置场景", table row
       "左A[M,K] 转置=是").  Implemented as
       one full-K transpose-LOAD per chunk, `pl.load(kf_l1, k_f32, [b,h,ni,0,0],
       order=[4,3])` with kf_l1 = [KMAX,C] ZN, plus `pl.move(l0a, kf_l1,
       offset=[koff5,0])` per k-tile (PANKO b1).  Same intent (no GM pre-transpose, no extra
       workspace, no sync_all); only the API spelling differs.
  V-B  Partial-N and partial-M fixpipe are BOTH unsafe on this build.  Measured:
       with the L0C valid shape narrowed on the N axis (vcur=32 < VTP), the
       DualModeSplitM drain leaves the first rows of the destination UB tile
       holding stale data -- an A/B at [1,128,1,64,64] gave v_new max_abs 0.0
       at VT=64 and 7.5e-1 at VT=32, with the error confined to rows 0-7.
       Applying the official a5 matmul sample's rule fixed it: matmul AND
       fixpipe run on the FULL [64,64] tile; only the contraction axis and the
       GM load/store are narrowed.  So every L0A/L0B/L0C valid shape on the
       N axis is VTP, every L0C valid shape on the M axis is C, and `vcur`
       survives only in the GM loads/stores and the VF column masks.  Likewise
       C5 runs M=64 (kf_l1 is transpose-loaded narrow from GM, then widened to
       the full square for the L0A move) and the s_hist store is narrowed --
       which restores DESIGN sec.7's own banding, row0 = sub*CH,
       rows_k = clamp(kcur - row0, 0, CH).  Without this, k_tail (K=200,
       last kcur=8) failed with matched_ratio 0.891 while every other case passed.
  V-C  DESIGN sec.4 orders C5/V4 as `for s: for kob:`.  With a counting semaphore
       and depth-1 d_ub[s], two bootstrap credits would both be spent on d_ub[0]
       before the vector reads it (write-before-read race).  Loop order swapped to
       `for kob: for s:` on BOTH sides so consecutive credits land on different
       buffers.  Credit totals and FIFO pairing per DESIGN sec.6 are unchanged.
  V-D  DESIGN sec.5's VT/STREAMS selection code block and its own table disagree
       (the code block's key (cores, S, VT) selects S=1/VT=32/32 cores at the
       production shape, the table says S=2/VT=32/16 cores).  The CODE BLOCK is
       implemented verbatim.  `_streams` / `_vt` keyword-only overrides (default
       None = auto) exist solely for the DESIGN sec.11 G6 A/B arm; they change no
       math and no output.
  V-E  DESIGN sec.6 puts a `pl.system.bar_all()` after the V0 bootstrap.  Omitted:
       the bootstrap only writes this task's own disjoint GM band and is ordered to
       the cube by the S_READY (e0) MTE3 handshake on the same block; a grid-wide
       barrier inside a per-core task loop with unequal task counts would deadlock.
       Same as the hardware-verified sibling's M5 bootstrap.
  V-F  `vp_ub[s]` / `d_ub[s]` are declared [C, VTP], not DESIGN sec.2's [CH, VTP]:
       they are DualModeSplitM destinations and both hardware-verified references
       declare those with the full M.  UB terminal extent therefore moves from
       DESIGN sec.3's 0x14500 (83,200 B, 32.8%) to 0x1C500 (115,968 B, 45.7%) --
       still far under the 253,952 B limit and under the sec.11 60% guidance.
       Every other UB address shifts accordingly; the aliases are preserved
       (obf_ub[s] over vs_ub[s], d_acc[s] over vp_acc[s]).
  V-G  `pl.move(l0a, kf_l1)` needs no per-kob narrowing (see V-B), and every tile
       handle is hoisted once per loop iteration (auto_mutex SSA), per the
       hardware-verified sibling's explicit note.

Contract
--------
  wrapper  gdr_fwd_train_scan_wrapper(u, w, q_de, a_mat, k_f32, dfac,
                                              decay_exp) -> (o, s_hist, v_new)
  o       [B, 64N, H, V] bf16   -- public forward output (o[:, :T] == lean fwd o)
  s_hist  [B, H, N, K, V] fp32  -- ENTERING state per chunk; s_hist[:,:,0] == 0
  v_new   [B, H, N, C, V] fp32  -- u_i - w_i @ s_hist[:,:,i]
Supported range: K <= 256, V <= 256 (SPEC upper bound; v1's K,V <= 128 raise is gone).
"""

import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format="%(message)s")

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info


# ── frozen constants (DESIGN sec.2.1) ────────────────────────────────────────
C = 64            # chunk size (SPEC-frozen internal constant)
KT = 64           # K tile: contraction block AND delta/S output block
VTP = 64          # V tile PHYSICAL width (tiles are declared with it)
KMAX = 256        # SPEC upper bound on K -- physical width of the full-K L1 tiles
CH = 32           # rows per AIV sub-block after DualModeSplitM (C // 2)

_IB = pl.CrossCoreSyncMode.INTRA_BLOCK
_S = pl.PipeType.S
_V = pl.PipeType.V
_FIX = pl.PipeType.FIX
_MTE2 = pl.PipeType.MTE2
_MTE3 = pl.PipeType.MTE3

# cross-core event ids (DESIGN sec.6)
E_S_READY = 0     # Vec -> Cube : s_hist[ni] readable
E_VP_READY = 1    # Cube -> Vec : vp_ub[s] holds v_prime / o
E_VP_FREE = 2     # Vec -> Cube : vp_ub[s] consumed
E_VN_READY = 3    # Vec -> Cube : v_new / vscaled_gm written
E_VN_FREE = 4     # Cube -> Vec : v_new / vscaled_gm consumed
E_D_READY = 5     # Cube -> Vec : d_ub[s] holds delta
E_D_FREE = 6      # Vec -> Cube : d_ub[s] consumed


# =============================================================================
#   Vector functions (row pitch = VTP; each call runs ONE sub-block's band)
# =============================================================================
@pl.vector_function
def scan_sub_vf(vp_tile, uv_tile, dfac_tile, vs_tile,
                n_rows: pl.DT_INT64, n_cols: pl.DT_INT64):
    """v_new = u - v_prime (-> uv_tile); v_scaled = v_new * dfac[row] (-> vs_tile)."""
    for r in pl.range(0, n_rows):
        fac = vf.load_align(dfac_tile, r * 8, dist=pl.LoadDist.BRC_B32)
        mreg = vf.update_mask(n_cols, dtype=pl.DT_FP32)
        off = r * VTP
        uu = vf.load_align(uv_tile, off)
        vp = vf.load_align(vp_tile, off)
        vn = vf.sub(uu, vp, mreg)
        vs = vf.mul(vn, fac, mreg)
        vf.store_align(uv_tile + off, vn, mreg)
        vf.store_align(vs_tile + off, vs, mreg)


@pl.vector_function
def state_update_vf(state_tile, delta_tile, dexp_tile,
                    n_rows: pl.DT_INT64, n_cols: pl.DT_INT64, didx: pl.DT_INT64):
    """S = S * decay_exp[didx] + delta over [n_rows, n_cols] (pitch VTP)."""
    dreg = vf.load_align(dexp_tile, didx, dist=pl.LoadDist.BRC_B32)
    for r in pl.range(0, n_rows):
        mreg = vf.update_mask(n_cols, dtype=pl.DT_FP32)
        off = r * VTP
        s = vf.load_align(state_tile, off)
        d = vf.load_align(delta_tile, off)
        s = vf.mul(s, dreg, mreg)
        s = vf.add(s, d, mreg)
        vf.store_align(state_tile + off, s, mreg)


@pl.vector_function
def zero_state_vf(t, n_rows: pl.DT_INT64):
    """Zero [n_rows, VTP] of a fp32 tile (pitch VTP)."""
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    z = vf.full(0.0, active, dtype=pl.DT_FP32)
    for r in pl.range(0, n_rows):
        vf.store_align(t + r * VTP, z, active)


# =============================================================================
#   The single @pl.jit kernel
# =============================================================================
_DYN5 = [pl.DYNAMIC] * 5
_DYN4 = [pl.DYNAMIC] * 4


@pl.jit(auto_mutex=True)
def gdr_fwd_train_scan_module1_kernel(
    u: pl.Tensor[_DYN5, pl.DT_FP32],
    w: pl.Tensor[_DYN5, pl.DT_FP32],
    q_de: pl.Tensor[_DYN5, pl.DT_FP32],
    a_mat: pl.Tensor[_DYN5, pl.DT_FP32],
    k_f32: pl.Tensor[_DYN5, pl.DT_FP32],
    dfac: pl.Tensor[_DYN5, pl.DT_FP32],
    decay_exp: pl.Tensor[_DYN4, pl.DT_FP32],
    vscaled_gm: pl.Tensor[_DYN4, pl.DT_FP32],
    o: pl.Tensor[_DYN4, pl.DT_BF16],
    s_hist: pl.Tensor[_DYN5, pl.DT_FP32],
    v_new: pl.Tensor[_DYN5, pl.DT_FP32],
    B: pl.DT_INT32, H: pl.DT_INT32, Kd: pl.DT_INT32, Vd: pl.DT_INT32,
    N: pl.DT_INT32, VT: pl.DT_INT32, vtiles: pl.DT_INT32,
    vpairs: pl.DT_INT32, SR: pl.DT_INT32,
):
    # ---- UB (DESIGN sec.3 address map; 0x1C500 terminal = 115,968 B (deviation V-F)) ---------
    g_st_ub_0 = pl.make_tile_group(type=pl.TileType(shape=[CH, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=[0x00000, 0x1C800, 0x1E800, 0x20800], mutex_ids=[0, 27, 28, 29])
    g_uv_ub_0 = pl.make_tile_group(type=pl.TileType(shape=[CH, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x02000, mutex_ids=[1])
    g_vs_ub_0 = pl.make_tile_group(type=pl.TileType(shape=[CH, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x04000, mutex_ids=[2])
    g_vp_ub_0 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x06000, mutex_ids=[3])
    g_d_ub_0 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x0A000, mutex_ids=[4])
    g_st_ub_1 = pl.make_tile_group(type=pl.TileType(shape=[CH, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=[0x0E000, 0x22800, 0x24800, 0x26800], mutex_ids=[5, 30, 31, 27])
    g_uv_ub_1 = pl.make_tile_group(type=pl.TileType(shape=[CH, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x10000, mutex_ids=[6])
    g_vs_ub_1 = pl.make_tile_group(type=pl.TileType(shape=[CH, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x12000, mutex_ids=[7])
    g_vp_ub_1 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x14000, mutex_ids=[8])
    g_d_ub_1 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x18000, mutex_ids=[9])
    # obf_ub[s] aliases vs_ub[s] (same address, same mutex id; DESIGN sec.3 arg 1)
    g_obf_0 = pl.make_tile_group(type=pl.TileType(shape=[CH, VTP], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x04000, mutex_ids=[2])
    g_obf_1 = pl.make_tile_group(type=pl.TileType(shape=[CH, VTP], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x12000, mutex_ids=[7])
    g_dfac = pl.make_tile_group(type=pl.TileType(shape=[CH, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x1C000, mutex_ids=[10])
    g_dexp = pl.make_tile_group(type=pl.TileType(shape=[1, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x1C400, mutex_ids=[11])

    # ---- L1 (0x4C000 terminal = 311,296 B of 524,288; PANKO b1) ------------
    # PANKO b1 (cube-04 + cube-08): w / q_de / k^T are loaded ONCE PER CHUNK in
    # full-K form and sliced per k-tile with pl.move offsets, so the per-k-tile
    # GM->L1 loads disappear from inside the K loops AND from behind the
    # S_READY / VN_READY barriers.  k^T also gets its own mutex id (13, freed by
    # w no longer needing a ping/pong pair) -- id 14 now carries 2 tiles, not 3.
    g_w_l1 = pl.make_tile_group(type=pl.TileType(shape=[C, KMAX], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x00000, mutex_ids=[12])
    g_qde_l1 = pl.make_tile_group(type=pl.TileType(shape=[C, KMAX], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x10000, mutex_ids=[14])
    g_a_l1 = pl.make_tile_group(type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x30000, mutex_ids=[14])
    # V-A: transposed LEFT operand -> declare [M, Kc] = [KMAX, C] with layout ZN.
    g_kf_l1 = pl.make_tile_group(type=pl.TileType(shape=[KMAX, C], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1], layout=pl.ZN), addrs=0x20000, mutex_ids=[13])
    g_st_l1_0 = pl.make_tile_group(type=pl.TileType(shape=[KT, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x34000, mutex_ids=[15])
    g_st_l1_1 = pl.make_tile_group(type=pl.TileType(shape=[KT, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x38000, mutex_ids=[16])
    g_vn_l1_0 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x3C000, mutex_ids=[15])
    g_vn_l1_1 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x40000, mutex_ids=[16])
    g_vs_l1_0 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x44000, mutex_ids=[17])
    g_vs_l1_1 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x48000, mutex_ids=[18])

    # ---- L0A / L0B / L0C ---------------------------------------------------
    g_l0a = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1]), addrs=[0x0000, 0x4000], mutex_ids=[19, 20])
    g_l0b = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1]), addrs=[0x0000, 0x4000], mutex_ids=[21, 22])
    # vp_acc[s] and d_acc[s] are ONE tile (same addr, same id): disjoint lifetimes.
    g_vpd_0 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x00000, mutex_ids=[23])
    g_oacc_0 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x04000, mutex_ids=[24])
    g_vpd_1 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x08000, mutex_ids=[25])
    g_oacc_1 = pl.make_tile_group(type=pl.TileType(shape=[C, VTP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x0C000, mutex_ids=[26])

    # =====================================================================
    #   CUBE
    # =====================================================================
    with pl.section_cube():
        ncore = pl.get_block_num()
        core = pl.get_block_idx() // pl.get_subblock_num()
        ktiles = (Kd + KT - 1) // KT
        nkob = ktiles
        tasks = B * H * vpairs
        per = (tasks + ncore - 1) // ncore
        lo = per * core
        hi = pl.min(lo + per, tasks)

        for tid in pl.range(lo, hi, 1):
            bh = tid // vpairs
            vpx = tid % vpairs
            b = bh // H
            h = bh % H
            vv0 = vpx * SR
            vlim = pl.min(vtiles, vv0 + SR)
            s1_on = (vv0 + 1) < vlim
            voff0 = vv0 * VT
            vcur0 = pl.min(VT, Vd - voff0)
            voff1 = (vv0 + 1) * VT
            vcur1 = pl.min(VT, Vd - voff1)

            # bootstrap: VN_FREE credits (one per active stream)
            pl.system.set_cross_core(pipe=_FIX, event_id=E_VN_FREE, sync_mode=_IB)
            if s1_on:
                pl.system.set_cross_core(pipe=_FIX, event_id=E_VN_FREE, sync_mode=_IB)

            for ni in pl.range(0, N, 1):
                # hoist every tile handle once per iteration (auto_mutex SSA)
                stl1_0 = g_st_l1_0.current()
                vnl1_0 = g_vn_l1_0.current()
                vsl1_0 = g_vs_l1_0.current()
                acc_0 = g_vpd_0.current()
                oac_0 = g_oacc_0.current()
                vpu_0 = g_vp_ub_0.current()
                dub_0 = g_d_ub_0.current()
                stl1_1 = g_st_l1_1.current()
                vnl1_1 = g_vn_l1_1.current()
                vsl1_1 = g_vs_l1_1.current()
                acc_1 = g_vpd_1.current()
                oac_1 = g_oacc_1.current()
                vpu_1 = g_vp_ub_1.current()
                dub_1 = g_d_ub_1.current()
                ql1 = g_qde_l1.current()
                al1 = g_a_l1.current()
                kfl1 = g_kf_l1.current()
                wl1 = g_w_l1.current()
                # PANKO b1: w, q_de and k^T do NOT depend on S_READY.  Issue all
                # three full-K loads BEFORE the barrier so MTE2 works through the
                # recurrence handoff instead of idling in front of it (cube-08
                # "barrier audit"); each is now ONE descriptor over a fully
                # contiguous [C, Kd] GM block instead of ktiles strided ones.
                pl.set_validshape(wl1, [C, Kd])
                pl.load(wl1, w, [b, h, ni, 0, 0])
                pl.set_validshape(wl1, [C, KMAX])
                pl.set_validshape(ql1, [C, Kd])
                pl.load(ql1, q_de, [b, h, ni, 0, 0])
                pl.set_validshape(ql1, [C, KMAX])
                pl.set_validshape(kfl1, [Kd, C])
                pl.load(kfl1, k_f32, [b, h, ni, 0, 0], order=[4, 3])
                pl.set_validshape(kfl1, [KMAX, C])
                pl.system.wait_cross_core(pipe=_S, event_id=E_S_READY, sync_mode=_IB)
                for kk in pl.range(0, ktiles):
                    koff = kk * KT
                    kcur = pl.min(KT, Kd - koff)
                    pl.set_validshape(stl1_0, [kcur, vcur0])
                    pl.load(stl1_0, s_hist, [b, h, ni, koff, voff0])
                    pl.set_validshape(stl1_0, [kcur, VTP])
                    la = g_l0a.next()
                    pl.set_validshape(la, [C, kcur])
                    pl.move(la, wl1, offset=[0, koff])
                    lb = g_l0b.next()
                    pl.set_validshape(lb, [kcur, VTP])
                    pl.move(lb, stl1_0)
                    pl.set_validshape(acc_0, [C, VTP])
                    if kk == 0:
                        pl.matmul(acc_0, la, lb)
                    else:
                        pl.matmul_acc(acc_0, acc_0, la, lb)
                if s1_on:
                    pl.system.wait_cross_core(pipe=_S, event_id=E_S_READY, sync_mode=_IB)
                    for kk in pl.range(0, ktiles):
                        koff = kk * KT
                        kcur = pl.min(KT, Kd - koff)
                        pl.set_validshape(stl1_1, [kcur, vcur1])
                        pl.load(stl1_1, s_hist, [b, h, ni, koff, voff1])
                        pl.set_validshape(stl1_1, [kcur, VTP])
                        la = g_l0a.next()
                        pl.set_validshape(la, [C, kcur])
                        pl.move(la, wl1, offset=[0, koff])
                        lb = g_l0b.next()
                        pl.set_validshape(lb, [kcur, VTP])
                        pl.move(lb, stl1_1)
                        pl.set_validshape(acc_1, [C, VTP])
                        if kk == 0:
                            pl.matmul(acc_1, la, lb)
                        else:
                            pl.matmul_acc(acc_1, acc_1, la, lb)
                pl.system.wait_cross_core(pipe=_S, event_id=E_VP_FREE, sync_mode=_IB)
                pl.set_validshape(acc_0, [C, VTP])
                pl.move(vpu_0, acc_0, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                pl.system.set_cross_core(pipe=_FIX, event_id=E_VP_READY, sync_mode=_IB)
                if s1_on:
                    pl.system.wait_cross_core(pipe=_S, event_id=E_VP_FREE, sync_mode=_IB)
                    pl.set_validshape(acc_1, [C, VTP])
                    pl.move(vpu_1, acc_1, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                    pl.system.set_cross_core(pipe=_FIX, event_id=E_VP_READY, sync_mode=_IB)
                for kk2 in pl.range(0, ktiles):
                    koff2 = kk2 * KT
                    kcur2 = pl.min(KT, Kd - koff2)
                    pl.set_validshape(stl1_0, [kcur2, vcur0])
                    pl.load(stl1_0, s_hist, [b, h, ni, koff2, voff0])
                    pl.set_validshape(stl1_0, [kcur2, VTP])
                    la2 = g_l0a.next()
                    pl.set_validshape(la2, [C, kcur2])
                    pl.move(la2, ql1, offset=[0, koff2])
                    lb2 = g_l0b.next()
                    pl.set_validshape(lb2, [kcur2, VTP])
                    pl.move(lb2, stl1_0)
                    pl.set_validshape(oac_0, [C, VTP])
                    if kk2 == 0:
                        pl.matmul(oac_0, la2, lb2)
                    else:
                        pl.matmul_acc(oac_0, oac_0, la2, lb2)
                if s1_on:
                    for kk2 in pl.range(0, ktiles):
                        koff2 = kk2 * KT
                        kcur2 = pl.min(KT, Kd - koff2)
                        pl.set_validshape(stl1_1, [kcur2, vcur1])
                        pl.load(stl1_1, s_hist, [b, h, ni, koff2, voff1])
                        pl.set_validshape(stl1_1, [kcur2, VTP])
                        la2 = g_l0a.next()
                        pl.set_validshape(la2, [C, kcur2])
                        pl.move(la2, ql1, offset=[0, koff2])
                        lb2 = g_l0b.next()
                        pl.set_validshape(lb2, [kcur2, VTP])
                        pl.move(lb2, stl1_1)
                        pl.set_validshape(oac_1, [C, VTP])
                        if kk2 == 0:
                            pl.matmul(oac_1, la2, lb2)
                        else:
                            pl.matmul_acc(oac_1, oac_1, la2, lb2)
                pl.set_validshape(al1, [C, C])
                pl.load(al1, a_mat, [b, h, ni, 0, 0])
                pl.system.wait_cross_core(pipe=_S, event_id=E_VN_READY, sync_mode=_IB)
                pl.set_validshape(vnl1_0, [C, vcur0])
                pl.load(vnl1_0, v_new, [b, h, ni, 0, voff0])
                pl.set_validshape(vsl1_0, [C, vcur0])
                pl.load(vsl1_0, vscaled_gm, [b, h, 0, voff0])
                pl.system.set_cross_core(pipe=_MTE2, event_id=E_VN_FREE, sync_mode=_IB)
                pl.set_validshape(vnl1_0, [C, VTP])
                pl.set_validshape(vsl1_0, [C, VTP])
                la3 = g_l0a.next()
                pl.set_validshape(la3, [C, C])
                pl.move(la3, al1)
                lb3 = g_l0b.next()
                pl.set_validshape(lb3, [C, VTP])
                pl.move(lb3, vnl1_0)
                pl.set_validshape(oac_0, [C, VTP])
                pl.matmul_acc(oac_0, oac_0, la3, lb3)
                if s1_on:
                    pl.system.wait_cross_core(pipe=_S, event_id=E_VN_READY, sync_mode=_IB)
                    pl.set_validshape(vnl1_1, [C, vcur1])
                    pl.load(vnl1_1, v_new, [b, h, ni, 0, voff1])
                    pl.set_validshape(vsl1_1, [C, vcur1])
                    pl.load(vsl1_1, vscaled_gm, [b, h, 0, voff1])
                    pl.system.set_cross_core(pipe=_MTE2, event_id=E_VN_FREE, sync_mode=_IB)
                    pl.set_validshape(vnl1_1, [C, VTP])
                    pl.set_validshape(vsl1_1, [C, VTP])
                    la3 = g_l0a.next()
                    pl.set_validshape(la3, [C, C])
                    pl.move(la3, al1)
                    lb3 = g_l0b.next()
                    pl.set_validshape(lb3, [C, VTP])
                    pl.move(lb3, vnl1_1)
                    pl.set_validshape(oac_1, [C, VTP])
                    pl.matmul_acc(oac_1, oac_1, la3, lb3)
                pl.system.wait_cross_core(pipe=_S, event_id=E_VP_FREE, sync_mode=_IB)
                pl.set_validshape(oac_0, [C, VTP])
                pl.move(vpu_0, oac_0, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                pl.system.set_cross_core(pipe=_FIX, event_id=E_VP_READY, sync_mode=_IB)
                if s1_on:
                    pl.system.wait_cross_core(pipe=_S, event_id=E_VP_FREE, sync_mode=_IB)
                    pl.set_validshape(oac_1, [C, VTP])
                    pl.move(vpu_1, oac_1, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                    pl.system.set_cross_core(pipe=_FIX, event_id=E_VP_READY, sync_mode=_IB)
                for kob in pl.range(0, nkob):
                    koff5 = kob * KT
                    kcur5 = pl.min(KT, Kd - koff5)
                    la5 = g_l0a.next()
                    pl.set_validshape(la5, [C, C])
                    pl.move(la5, kfl1, offset=[koff5, 0])
                    lb5 = g_l0b.next()
                    pl.set_validshape(lb5, [C, VTP])
                    pl.move(lb5, vsl1_0)
                    pl.set_validshape(acc_0, [C, VTP])
                    pl.matmul(acc_0, la5, lb5)
                    pl.system.wait_cross_core(pipe=_S, event_id=E_D_FREE, sync_mode=_IB)
                    pl.set_validshape(acc_0, [C, VTP])
                    pl.move(dub_0, acc_0, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                    pl.system.set_cross_core(pipe=_FIX, event_id=E_D_READY, sync_mode=_IB)
                    if s1_on:
                        la5 = g_l0a.next()
                        pl.set_validshape(la5, [C, C])
                        pl.move(la5, kfl1, offset=[koff5, 0])
                        lb5 = g_l0b.next()
                        pl.set_validshape(lb5, [C, VTP])
                        pl.move(lb5, vsl1_1)
                        pl.set_validshape(acc_1, [C, VTP])
                        pl.matmul(acc_1, la5, lb5)
                        pl.system.wait_cross_core(pipe=_S, event_id=E_D_FREE, sync_mode=_IB)
                        pl.set_validshape(acc_1, [C, VTP])
                        pl.move(dub_1, acc_1, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                        pl.system.set_cross_core(pipe=_FIX, event_id=E_D_READY, sync_mode=_IB)

            # ---- cube drains: S_READY, VP_FREE, D_FREE (one per stream) ----
            pl.system.wait_cross_core(pipe=_S, event_id=E_S_READY, sync_mode=_IB)
            pl.system.wait_cross_core(pipe=_S, event_id=E_VP_FREE, sync_mode=_IB)
            pl.system.wait_cross_core(pipe=_S, event_id=E_D_FREE, sync_mode=_IB)
            if s1_on:
                pl.system.wait_cross_core(pipe=_S, event_id=E_S_READY, sync_mode=_IB)
                pl.system.wait_cross_core(pipe=_S, event_id=E_VP_FREE, sync_mode=_IB)
                pl.system.wait_cross_core(pipe=_S, event_id=E_D_FREE, sync_mode=_IB)

    # =====================================================================
    #   VECTOR (both AIV sub-blocks; no `if subblock == 0` guard)
    # =====================================================================
    with pl.section_vector():
        ncore2 = pl.get_block_num()
        core2 = pl.get_block_idx() // pl.get_subblock_num()
        sub = pl.get_subblock_idx()
        row0 = sub * CH
        ktiles2 = (Kd + KT - 1) // KT
        nkob2 = ktiles2
        tasks2 = B * H * vpairs
        per2 = (tasks2 + ncore2 - 1) // ncore2
        lo2 = per2 * core2
        hi2 = pl.min(lo2 + per2, tasks2)

        for tid in pl.range(lo2, hi2, 1):
            bh = tid // vpairs
            vpx = tid % vpairs
            b = bh // H
            h = bh % H
            vv0 = vpx * SR
            vlim = pl.min(vtiles, vv0 + SR)
            s1_on = (vv0 + 1) < vlim
            voff0 = vv0 * VT
            vcur0 = pl.min(VT, Vd - voff0)
            voff1 = (vv0 + 1) * VT
            vcur1 = pl.min(VT, Vd - voff1)

            # ---- V0 bootstrap: s_hist[...,0,...] = 0 for this task ---------
            # hoist every tile handle once per task (auto_mutex SSA)
            uvu_0 = g_uv_ub_0.current()
            vsu_0 = g_vs_ub_0.current()
            vpu_0 = g_vp_ub_0.current()
            dub_0 = g_d_ub_0.current()
            obu_0 = g_obf_0.current()
            uvu_1 = g_uv_ub_1.current()
            vsu_1 = g_vs_ub_1.current()
            vpu_1 = g_vp_ub_1.current()
            dub_1 = g_d_ub_1.current()
            obu_1 = g_obf_1.current()
            dfu = g_dfac.current()
            dxu = g_dexp.current()
            for kb0 in pl.range(0, nkob2):
                kf0 = kb0 * KT
                kc0 = pl.min(KT, Kd - kf0)
                ro0 = sub * CH
                rk0 = pl.min(CH, pl.max(0, kc0 - ro0))
                stu_0 = g_st_ub_0[kb0]
                if rk0 > 0:
                    pl.set_validshape(stu_0, [rk0, vcur0])
                    zero_state_vf(stu_0, rk0)
                    pl.set_validshape(stu_0, [rk0, vcur0])
                    pl.store(s_hist, stu_0, [b, h, 0, kf0 + ro0, voff0])
            if s1_on:
                for kb0 in pl.range(0, nkob2):
                    kf0 = kb0 * KT
                    kc0 = pl.min(KT, Kd - kf0)
                    ro0 = sub * CH
                    rk0 = pl.min(CH, pl.max(0, kc0 - ro0))
                    stu_1 = g_st_ub_1[kb0]
                    if rk0 > 0:
                        pl.set_validshape(stu_1, [rk0, vcur1])
                        zero_state_vf(stu_1, rk0)
                        pl.set_validshape(stu_1, [rk0, vcur1])
                        pl.store(s_hist, stu_1, [b, h, 0, kf0 + ro0, voff1])
            pl.system.set_cross_core(pipe=_MTE3, event_id=E_S_READY, sync_mode=_IB)
            pl.system.set_cross_core(pipe=_V, event_id=E_VP_FREE, sync_mode=_IB)
            pl.system.set_cross_core(pipe=_V, event_id=E_D_FREE, sync_mode=_IB)
            if s1_on:
                pl.system.set_cross_core(pipe=_MTE3, event_id=E_S_READY, sync_mode=_IB)
                pl.system.set_cross_core(pipe=_V, event_id=E_VP_FREE, sync_mode=_IB)
                pl.system.set_cross_core(pipe=_V, event_id=E_D_FREE, sync_mode=_IB)

            for ni in pl.range(0, N, 1):
                pl.set_validshape(uvu_0, [CH, vcur0])
                pl.load(uvu_0, u, [b, h, ni, row0, voff0])
                pl.set_validshape(dfu, [CH, 8])
                pl.load(dfu, dfac, [b, h, ni, row0, 0])
                pl.system.wait_cross_core(pipe=_S, event_id=E_VP_READY, sync_mode=_IB)
                pl.set_validshape(vpu_0, [CH, vcur0])
                scan_sub_vf(vpu_0, uvu_0, dfu, vsu_0, CH, vcur0)
                pl.system.set_cross_core(pipe=_V, event_id=E_VP_FREE, sync_mode=_IB)
                if s1_on:
                    pl.set_validshape(uvu_1, [CH, vcur1])
                    pl.load(uvu_1, u, [b, h, ni, row0, voff1])
                    pl.set_validshape(dfu, [CH, 8])
                    pl.load(dfu, dfac, [b, h, ni, row0, 0])
                    pl.system.wait_cross_core(pipe=_S, event_id=E_VP_READY, sync_mode=_IB)
                    pl.set_validshape(vpu_1, [CH, vcur1])
                    scan_sub_vf(vpu_1, uvu_1, dfu, vsu_1, CH, vcur1)
                    pl.system.set_cross_core(pipe=_V, event_id=E_VP_FREE, sync_mode=_IB)
                pl.system.wait_cross_core(pipe=_S, event_id=E_VN_FREE, sync_mode=_IB)
                pl.set_validshape(uvu_0, [CH, vcur0])
                pl.store(v_new, uvu_0, [b, h, ni, row0, voff0])
                pl.set_validshape(vsu_0, [CH, vcur0])
                pl.store(vscaled_gm, vsu_0, [b, h, row0, voff0])
                pl.system.set_cross_core(pipe=_MTE3, event_id=E_VN_READY, sync_mode=_IB)
                if s1_on:
                    pl.system.wait_cross_core(pipe=_S, event_id=E_VN_FREE, sync_mode=_IB)
                    pl.set_validshape(uvu_1, [CH, vcur1])
                    pl.store(v_new, uvu_1, [b, h, ni, row0, voff1])
                    pl.set_validshape(vsu_1, [CH, vcur1])
                    pl.store(vscaled_gm, vsu_1, [b, h, row0, voff1])
                    pl.system.set_cross_core(pipe=_MTE3, event_id=E_VN_READY, sync_mode=_IB)
                pl.set_validshape(dxu, [1, C])
                pl.load(dxu, decay_exp, [b, h, ni, 0])
                pl.system.wait_cross_core(pipe=_S, event_id=E_VP_READY, sync_mode=_IB)
                pl.set_validshape(vpu_0, [CH, vcur0])
                pl.set_validshape(obu_0, [CH, vcur0])
                pl.cast(obu_0, vpu_0, mode=pl.RoundMode.CAST_ROUND)
                pl.set_validshape(obu_0, [CH, vcur0])
                pl.store(o, obu_0, [b, ni * C + row0, h, voff0], order=[1, 3])
                pl.system.set_cross_core(pipe=_MTE3, event_id=E_VP_FREE, sync_mode=_IB)
                if s1_on:
                    pl.system.wait_cross_core(pipe=_S, event_id=E_VP_READY, sync_mode=_IB)
                    pl.set_validshape(vpu_1, [CH, vcur1])
                    pl.set_validshape(obu_1, [CH, vcur1])
                    pl.cast(obu_1, vpu_1, mode=pl.RoundMode.CAST_ROUND)
                    pl.set_validshape(obu_1, [CH, vcur1])
                    pl.store(o, obu_1, [b, ni * C + row0, h, voff1], order=[1, 3])
                    pl.system.set_cross_core(pipe=_MTE3, event_id=E_VP_FREE, sync_mode=_IB)
                for kob2 in pl.range(0, nkob2):
                    kf4 = kob2 * KT
                    kc4 = pl.min(KT, Kd - kf4)
                    ro4 = sub * CH
                    rk4 = pl.min(CH, pl.max(0, kc4 - ro4))
                    pl.system.wait_cross_core(pipe=_S, event_id=E_D_READY, sync_mode=_IB)
                    stu_0 = g_st_ub_0[kob2]
                    if rk4 > 0:
                        pl.set_validshape(dub_0, [rk4, vcur0])
                        state_update_vf(stu_0, dub_0, dxu, rk4, vcur0, C - 1)
                        if ni + 1 < N:
                            pl.set_validshape(stu_0, [rk4, vcur0])
                            pl.store(s_hist, stu_0, [b, h, ni + 1, kf4 + ro4, voff0])
                    pl.system.set_cross_core(pipe=_V, event_id=E_D_FREE, sync_mode=_IB)
                    if s1_on:
                        pl.system.wait_cross_core(pipe=_S, event_id=E_D_READY, sync_mode=_IB)
                        stu_1 = g_st_ub_1[kob2]
                        if rk4 > 0:
                            pl.set_validshape(dub_1, [rk4, vcur1])
                            state_update_vf(stu_1, dub_1, dxu, rk4, vcur1, C - 1)
                            if ni + 1 < N:
                                pl.set_validshape(stu_1, [rk4, vcur1])
                                pl.store(s_hist, stu_1, [b, h, ni + 1, kf4 + ro4, voff1])
                        pl.system.set_cross_core(pipe=_V, event_id=E_D_FREE, sync_mode=_IB)
                pl.system.set_cross_core(pipe=_MTE3, event_id=E_S_READY, sync_mode=_IB)
                if s1_on:
                    pl.system.set_cross_core(pipe=_MTE3, event_id=E_S_READY, sync_mode=_IB)

            # ---- vector drain: the cube's VN_FREE bootstrap credits --------
            pl.system.wait_cross_core(pipe=_S, event_id=E_VN_FREE, sync_mode=_IB)
            if s1_on:
                pl.system.wait_cross_core(pipe=_S, event_id=E_VN_FREE, sync_mode=_IB)


# =============================================================================
#   Wrapper -- op entry (HARD contract name)
# =============================================================================
def _pick_vt_streams(B, H, V, cube_cores):
    """DESIGN sec.5 selection, re-keyed on the critical path (PANKO u1 / P-0).

    DECLARED DEVIATION V-H.  The DESIGN sec.5 code block took the lexicographic
    max of (cores, STREAMS, VT) -- "use as many cube cores as possible".  On the
    32-cube-core host it was written for that coincided with tasks == cores at
    the production shape (B*H*ceil(V/32) = 32 tasks, exactly 1 wave).  On a
    28-cube-core host the same rule selects VT=32/S=1 -> 32 tasks over 28 cores
    = ceil(32/28) = 2 waves with 24 of 28 cores idle through the second;
    measured cube_utilization 56.4% == 32/(2*28) confirms it exactly.

    And because deviation V-B makes matmul AND fixpipe run on the FULL [64,64]
    tile whatever VT is, a narrower VT does not shrink a task's MAC work -- it
    only multiplies the number of full-width tasks.  Since aic_mac is this
    kernel's bounding pipe, "more cores" bought nothing and cost a second wave.

    So the cost of a configuration is the critical path in task-units,
        waves (= ceil(tasks / cores))  x  STREAMS (V-tiles serialised per task),
    minimised, with the widest VT as tie-break (fewest redundant full-width
    matmuls, least GM traffic).  Math, outputs, dtypes and supported range
    (K,V <= 256) are unchanged; this picks only how the independent (b, h,
    V-tile) item space is folded onto cores.
    """
    best = None
    for vt in (64, 48, 32):
        vtiles = (V + vt - 1) // vt
        for sr in (2, 1):
            vpairs = (vtiles + sr - 1) // sr
            tasks = B * H * vpairs
            cores = min(cube_cores, tasks)
            waves = (tasks + cores - 1) // cores
            # among minimum-critical-path options, fill the most cube cores
            # (u1: 16 items on 28 cores leaves 12 idle); widest VT breaks ties.
            cand = ((waves * sr, -min(tasks, cube_cores), -vt), (cores, sr, vt))
            best = cand if best is None or cand < best else best
    return best[1]  # (cores, STREAMS, VT)


def gdr_fwd_train_scan_wrapper(u, w, q_de, a_mat, k_f32, dfac, decay_exp,
                                       *, _streams=None, _vt=None):
    """Param checks + torch.empty outputs/workspace + ONE kernel launch.

    Returns (o [B,64N,H,V] bf16, s_hist [B,H,N,K,V] fp32, v_new [B,H,N,C,V] fp32).
    `_streams` / `_vt` are the DESIGN sec.11 G6 A/B overrides (deviation V-D);
    they alter neither the math nor any output.
    """
    B, H, N, Cin, V = u.shape
    K = w.shape[-1]
    if Cin != C:
        raise ValueError(f"chunk dim must be {C}, got {Cin}")
    if K > 256 or V > 256:
        raise ValueError(f"SPEC upper bound is K,V <= 256; got K={K}, V={V}")
    dev = u.device
    T = C * N                       # o is the padded public layout [B, 64N, H, V]

    o = torch.empty((B, T, H, V), dtype=torch.bfloat16, device=dev)
    s_hist = torch.empty((B, H, N, K, V), dtype=torch.float32, device=dev)
    v_new = torch.empty((B, H, N, C, V), dtype=torch.float32, device=dev)
    vscaled_gm = torch.empty((B, H, C, V), dtype=torch.float32, device=dev)

    cube_cores = get_platform_info().core_num or 1
    cores, streams, vt = _pick_vt_streams(B, H, V, cube_cores)
    if _vt is not None:
        vt = _vt
    if _streams is not None:
        streams = _streams
    vtiles = (V + vt - 1) // vt
    vpairs = (vtiles + streams - 1) // streams
    tasks = B * H * vpairs
    num_cores = max(1, min(cube_cores, tasks))

    gdr_fwd_train_scan_module1_kernel[None, num_cores](
        u, w, q_de, a_mat, k_f32, dfac, decay_exp, vscaled_gm,
        o, s_hist, v_new,
        B, H, K, V, N, vt, vtiles, vpairs, streams)
    torch.npu.synchronize()
    return o, s_hist, v_new

