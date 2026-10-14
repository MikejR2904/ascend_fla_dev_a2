# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO-Pro gdr_bwd_reverse_scan (OP4 / parent module M7, golden_step b2) —
L1 STAGED file, Module 1 (module_count=1, so this IS the complete operator).

BH-serial REVERSE inter-chunk dS scan (i=N-1..0), carrying loop-carried state
dS[K,TV] (init 0, V-subblocked TV=64) in Vec UB, bridged UB->GM->L1 for the two
cube phases.  Per reverse chunk:

    dh[:,:,i] = dS                                   (entering-state snapshot)
    dvc       = k_f32_i @ dS
    dv2[:,:,i]= dv_local_i + dvc * exp(g_last_i - cumsum(g)_i)
    dS        = dS*decay_exp_i[C-1] + q_t_i@(decay_exp_i . do_i) - w_t_i@dv2_i

PROVENANCE.  The kernel body (tile set, section structure, 4-handshake
cross_core protocol, loop nest, VF chains, matmul routing) is the Stage 3.5
KAL transpiled product `kal/gdr_bwd_reverse_scan/staged_module1.py`, simulator-verified 12/12
(4 shapes x 3 seeds, both outputs) and transpile-clean under the a5 guard
(UB 140,032 B / 55.1%, mutex 22 default / 17 merge, intra/cross flags 0/0,
L0A 64 KB exactly at cap via off-peak reuse).  It is ORGANISED into the
deliverable here, not re-authored.  Declared changes vs the transpiled DSL:

 X1  `pl.load(ub_do_bf, grad_o, ...)` gained `order=[1, 3]`.  grad_o is the one
     PUBLIC-layout [B,T,H,V] tensor read with a 2-D tile; without `order=` the
     load defaults to the trailing two axes (H,V) and walks `t*H+h` rows instead
     of T-rows at fixed h.  H=1 hides this completely, which is why the Python
     simulator passed.  This is the known sim<->hardware divergence found on
     gdr_bwd_dv_local; the fix matches the hardware-verified forward operator
     (the forward operator's acceptance tests).  Every §8
     case here has H>1 (H=4,4,2,3), so the fix is exercised.  All other
     loads/stores are on INTERNAL [B,H,N,...] / [B,H,*,*] layouts whose trailing
     two axes are already the intended ones -- audited one by one, unchanged.
 X2  Tensor dim0 (B) changed from the transpiler's literal `1` to `pl.DYNAMIC`.
 X3  Wrapper, GM scratch allocation, tests and the kernel name are added (the
     transpiled product contains none of these).
 X4  mutex_id split (perf round, 2026-10-14).  The transpiled product put three
     address-disjoint L1 operands on id 13 (dS 0x8000 / q_t 0x10000 / w_t 0x18000,
     96 KB) and two more on id 15 (do 0x20000 / dv2 0x24000), which auto_mutex
     turns into a false ordering between otherwise independent MTE2 streams.
     q_t->17, w_t->18, dv2->19 (20 of 32 ids used).  Splitting address-disjoint
     tiles only REMOVES ordering, so it cannot change the result, and does not:
     every component and composition error figure is unchanged.  Measured
     217.7 -> 216.8 us (median of 3, -0.41%): c_mte2 0.260 -> 0.246 but c_mte1
     0.076 -> 0.086, because L0A is pinned at its 64 KB cap with two aliased
     views sharing ids [0,1], so MTE2 relief backs up into MTE1.  Kept because
     it is free, not because 0.41% is a result.

None of these changes the algorithm, tile budget, handshake protocol or math.
"""

import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info

logging.basicConfig(level=logging.INFO, format="%(message)s")

C = 64      # SPEC-frozen chunk size
TV = 64     # V-subblock width (DESIGN §2)

# Some PyPTO builds hand back bare ints for a few DT_* constants, which the VF
# kwarg checks reject.  Rebuild them at module scope (the VF parser inlines
# helper calls and rejects isinstance, so these must be plain names by the time
# a decorated body mentions them).  A healthy build passes straight through.
_DTT = type(pl.DT_FP32)


def _dt_fix(v):
    return v if isinstance(v, _DTT) else _DTT(v)


_PL_BF16 = _dt_fix(pl.DT_BF16)
_PL_FP32 = _dt_fix(pl.DT_FP32)


# =============================================================================
#   Vector functions (verbatim from the sim-verified transpiled product)
# =============================================================================
@pl.vector_function
def cumsum_vf(ub_g, ub_decay):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    acc = vf.full(0.0, active, dtype=_PL_FP32)
    for csc in pl.range(0, 64, 1):
        gc = vf.load_align(ub_g, csc, dist=pl.LoadDist.BRC_B32)
        acc = vf.add(acc, gc, active)
        vf.store_align(ub_decay + csc, acc, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)


@pl.vector_function
def dS_update_vf(ub_dS, ub_delta, ub_dexp, ck, valid_v):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    dreg = vf.load_align(ub_dexp, 63, dist=pl.LoadDist.BRC_B32)
    for dsur in pl.range(0, ck, 1):
        s = vf.load_align(ub_dS, (dsur) * 64, dtype=_PL_FP32)
        d = vf.load_align(ub_delta, (dsur) * 64, dtype=_PL_FP32)
        s = vf.mul(s, dreg, _tmp_maskreg_0)
        s = vf.add(s, d, _tmp_maskreg_0)
        vf.store_align(ub_dS + (dsur) * 64, s, _tmp_maskreg_0)


@pl.vector_function
def do_scaled_vf(ub_do_bf, ub_do_f, ub_dexp, valid_c, valid_v):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    z = vf.full(0.0, active, dtype=_PL_FP32)
    for dsr in pl.range(0, valid_c, 1):
        de = vf.load_align(ub_dexp, dsr, dist=pl.LoadDist.BRC_B32)
        src = vf.load_align(ub_do_bf, (dsr) * 64, dist=pl.LoadDist.UNPK_B16)
        dst = vf.astype(src, active, dtype=_PL_FP32, layout=pl.CastLayout.ZERO)
        dst = vf.mul(dst, de, _tmp_maskreg_0)
        vf.store_align(ub_do_f + (dsr) * 64, dst, active)
    for dspr in pl.range(valid_c, 64, 1):
        vf.store_align(ub_do_f + (dspr) * 64, z, active)


@pl.vector_function
def dv2_gate_vf(ub_decay, ub_tdvc, ub_dvl, ub_dv2, valid_v):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    glast = vf.load_align(ub_decay, 63, dist=pl.LoadDist.BRC_B32)
    for dgc in pl.range(0, 64, 1):
        dc = vf.load_align(ub_decay, dgc, dist=pl.LoadDist.BRC_B32)
        gate = vf.exp_sub(glast, dc, active)
        dvc = vf.load_align(ub_tdvc, (dgc) * 64, dtype=_PL_FP32)
        dvl = vf.load_align(ub_dvl, (dgc) * 64, dtype=_PL_FP32)
        dvc = vf.mul(dvc, gate, _tmp_maskreg_0)
        dvc = vf.add(dvc, dvl, _tmp_maskreg_0)
        vf.store_align(ub_dv2 + (dgc) * 64, dvc, _tmp_maskreg_0)


@pl.vector_function
def neg_dv2_vf(ub_dv2, valid_v):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    for ndr in pl.range(0, 64, 1):
        x = vf.load_align(ub_dv2, (ndr) * 64, dtype=_PL_FP32)
        y = vf.neg(x, _tmp_maskreg_0)
        vf.store_align(ub_dv2 + (ndr) * 64, y, _tmp_maskreg_0)


@pl.vector_function
def zero_dS_vf(ub_dS, nrows):
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    z = vf.full(0.0, active, dtype=_PL_FP32)
    for zdr in pl.range(0, nrows, 1):
        vf.store_align(ub_dS + (zdr) * 64, z, active)


# =============================================================================
#   The single @pl.jit kernel
# =============================================================================
@pl.jit(auto_mutex=True)
def gdr_bwd_reverse_scan_module1_kernel(
    q_t: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    w_t: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    k_f32: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    dS_gm: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    do_gm: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    dv2neg_gm: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    grad_o: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    decay_exp: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    g_f32: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    dv_local: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    dh: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    dv2: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    B: pl.DT_INT32, T: pl.DT_INT32, H: pl.DT_INT32, Kd: pl.DT_INT32,
    Vd: pl.DT_INT32, N: pl.DT_INT32, Cc: pl.DT_INT32, VT: pl.DT_INT32,
):
    g__l0a__v0 = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left), addrs=[0x0, 0x8000], mutex_ids=[0, 1])
    g__l0a__v1 = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left), addrs=[0x0, 0x8000], mutex_ids=[0, 1])
    g__l0b__v0 = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right), addrs=[0x0, 0x8000], mutex_ids=[2, 3])
    g__l0b__v1 = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right), addrs=[0x0, 0x8000], mutex_ids=[2, 3])
    g_ub_dS = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x0, mutex_ids=[4])
    g_ub_delta = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x8000, mutex_ids=[5])
    g_ub_tdvc = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x10000, mutex_ids=[6])
    g_ub_dvl = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x14000, mutex_ids=[7])
    g_ub_dv2 = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x18000, mutex_ids=[8])
    g_ub_do_bf = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec), addrs=0x1c000, mutex_ids=[9])
    g_ub_do_f = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x1e000, mutex_ids=[8])
    g_ub_g = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x22000, mutex_ids=[10])
    g_ub_decay = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x22100, mutex_ids=[11])
    g_ub_dexp = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x22200, mutex_ids=[10])
    g_l1_k = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat), addrs=0x0, mutex_ids=[12])
    g_l1_dS = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat), addrs=0x8000, mutex_ids=[13])
    g_tdvc_l0c = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc), addrs=0x0, mutex_ids=[14])
    g_l1_qt = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat), addrs=0x10000, mutex_ids=[17])
    g_l1_wt = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat), addrs=0x18000, mutex_ids=[18])
    g_l1_do = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat), addrs=0x20000, mutex_ids=[15])
    g_l1_dv2 = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat), addrs=0x24000, mutex_ids=[19])
    g_delta_l0c = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc), addrs=0x4000, mutex_ids=[16])

    with pl.section_cube():
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=4, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=4, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        t1_ub_delta = g_ub_delta.next()
        t2_ub_tdvc = g_ub_tdvc.next()
        t3_l1_k = g_l1_k.next()
        t4_l1_dS = g_l1_dS.next()
        t5_tdvc_l0c = g_tdvc_l0c.next()
        t6_l1_qt = g_l1_qt.next()
        t7_l1_wt = g_l1_wt.next()
        t8_l1_do = g_l1_do.next()
        t9_l1_dv2 = g_l1_dv2.next()
        t10_delta_l0c = g_delta_l0c.next()
        _nv = ((Vd + 64 - 1) // 64)
        _nt = ((B * H) * _nv)
        _nb = pl.get_block_num()
        _bi = (pl.get_block_idx() // pl.get_subblock_num())
        for _it in pl.range(_bi, _nt, _nb):
            bh = (_it // _nv)
            vb = (_it - (bh * _nv))
            pl.system.bar_all()
            for rj in pl.range(0, N, 1):
                pl.set_validshape(t3_l1_k, [64, Kd])
                pl.load(t3_l1_k, k_f32, [(bh // H), (bh % H), ((N - 1) - rj), 0, 0])
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                pl.set_validshape(t4_l1_dS, [Kd, (pl.min((Vd - (vb * 64)), 64))])
                pl.load(t4_l1_dS, dS_gm, [(bh // H), (bh % H), vb, 0, 0])
                t11__l0a__v1 = g__l0a__v1.next()
                t12__l0a = g__l0a__v0.next()
                t13__l0b__v1 = g__l0b__v1.next()
                t14__l0b = g__l0b__v0.next()
                pl.move(t12__l0a, t3_l1_k, [0, 0])
                pl.move(t14__l0b, t4_l1_dS, [0, 0])
                pl.set_validshape(t12__l0a, [64, Kd])
                pl.set_validshape(t14__l0b, [Kd, 64])
                pl.set_validshape(t5_tdvc_l0c, [64, (pl.min((Vd - (vb * 64)), 64))])
                pl.matmul(t5_tdvc_l0c, t12__l0a, t14__l0b)
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=3, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                pl.set_validshape(t5_tdvc_l0c, [64, (pl.min((Vd - (vb * 64)), 64))])
                pl.move(t2_ub_tdvc, t5_tdvc_l0c, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                pl.set_validshape(t6_l1_qt, [Kd, 64])
                pl.load(t6_l1_qt, q_t, [(bh // H), (bh % H), ((N - 1) - rj), 0, 0])
                pl.set_validshape(t7_l1_wt, [Kd, 64])
                pl.load(t7_l1_wt, w_t, [(bh // H), (bh % H), ((N - 1) - rj), 0, 0])
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                pl.set_validshape(t8_l1_do, [64, (pl.min((Vd - (vb * 64)), 64))])
                pl.load(t8_l1_do, do_gm, [(bh // H), (bh % H), vb, 0, 0])
                pl.set_validshape(t9_l1_dv2, [64, (pl.min((Vd - (vb * 64)), 64))])
                pl.load(t9_l1_dv2, dv2neg_gm, [(bh // H), (bh % H), vb, 0, 0])
                t15__l0a__v0 = g__l0a__v0.next()
                t16__l0a = g__l0a__v1.next()
                t17__l0b__v0 = g__l0b__v0.next()
                t18__l0b = g__l0b__v1.next()
                pl.move(t16__l0a, t6_l1_qt, [0, 0])
                pl.move(t18__l0b, t8_l1_do, [0, 0])
                pl.set_validshape(t16__l0a, [128, 64])
                pl.set_validshape(t18__l0b, [64, 64])
                pl.set_validshape(t10_delta_l0c, [128, (pl.min((Vd - (vb * 64)), 64))])
                pl.matmul(t10_delta_l0c, t16__l0a, t18__l0b)
                t19__l0a__v0 = g__l0a__v0.next()
                t20__l0a = g__l0a__v1.next()
                t21__l0b__v0 = g__l0b__v0.next()
                t22__l0b = g__l0b__v1.next()
                pl.move(t20__l0a, t7_l1_wt, [0, 0])
                pl.move(t22__l0b, t9_l1_dv2, [0, 0])
                pl.set_validshape(t20__l0a, [128, 64])
                pl.set_validshape(t22__l0b, [64, 64])
                pl.set_validshape(t10_delta_l0c, [128, (pl.min((Vd - (vb * 64)), 64))])
                pl.matmul_acc(t10_delta_l0c, t10_delta_l0c, t20__l0a, t22__l0b)
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=4, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=7, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                pl.set_validshape(t10_delta_l0c, [128, (pl.min((Vd - (vb * 64)), 64))])
                pl.move(t1_ub_delta, t10_delta_l0c, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=6, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=3, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=3, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=7, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=7, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

    with pl.section_vector():
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=3, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=3, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=7, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=7, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        t23_ub_dS = g_ub_dS.next()
        t24_ub_delta = g_ub_delta.next()
        t25_ub_tdvc = g_ub_tdvc.next()
        t26_ub_dvl = g_ub_dvl.next()
        t27_ub_dv2 = g_ub_dv2.next()
        t28_ub_do_bf = g_ub_do_bf.next()
        t29_ub_do_f = g_ub_do_f.next()
        t30_ub_g = g_ub_g.next()
        t31_ub_decay = g_ub_decay.next()
        t32_ub_dexp = g_ub_dexp.next()
        t33_l1_k = g_l1_k.next()
        t34_l1_dS = g_l1_dS.next()
        t35_l1_qt = g_l1_qt.next()
        t36_l1_wt = g_l1_wt.next()
        t37_l1_do = g_l1_do.next()
        t38_l1_dv2 = g_l1_dv2.next()
        _nv = ((Vd + 64 - 1) // 64)
        _nt = ((B * H) * _nv)
        _nb = pl.get_block_num()
        _bi = (pl.get_block_idx() // pl.get_subblock_num())
        for _it in pl.range(_bi, _nt, _nb):
            bh = (_it // _nv)
            vb = (_it - (bh * _nv))
            _k0 = (pl.get_subblock_idx() * 64)
            if _k0 < 64:
                zero_dS_vf(t23_ub_dS, Kd)
            pl.system.bar_all()
            for rj in pl.range(0, N, 1):
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=0, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                if _k0 < 64:
                    pl.set_validshape(t23_ub_dS, [Kd, (pl.min((Vd - (vb * 64)), 64))])
                    pl.store(dh, t23_ub_dS, [(bh // H), (bh % H), ((N - 1) - rj), 0, (vb * 64)])
                    pl.set_validshape(t23_ub_dS, [Kd, (pl.min((Vd - (vb * 64)), 64))])
                    pl.store(dS_gm, t23_ub_dS, [(bh // H), (bh % H), vb, 0, 0])
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                if _k0 < 64:
                    pl.set_validshape(t26_ub_dvl, [64, (pl.min((Vd - (vb * 64)), 64))])
                    pl.load(t26_ub_dvl, dv_local, [(bh // H), (bh % H), ((N - 1) - rj), 0, (vb * 64)])
                    pl.set_validshape(t30_ub_g, [1, 64])
                    pl.load(t30_ub_g, g_f32, [(bh // H), (bh % H), ((N - 1) - rj), 0])
                    pl.set_validshape(t32_ub_dexp, [1, 64])
                    pl.load(t32_ub_dexp, decay_exp, [(bh // H), (bh % H), ((N - 1) - rj), 0])
                    if (pl.min((T - (((N - 1) - rj) * 64)), 64)) > 0:
                        pl.set_validshape(t28_ub_do_bf, [(pl.min((T - (((N - 1) - rj) * 64)), 64)), (pl.min((Vd - (vb * 64)), 64))])
                        # X1: PUBLIC [B,T,H,V] layout -> the 2-D tile must walk
                        # axes (T, V), not the default trailing (H, V).
                        pl.load(t28_ub_do_bf, grad_o, [(bh // H), (((N - 1) - rj) * 64), (bh % H), (vb * 64)], order=[1, 3])
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                if _k0 < 64:
                    cumsum_vf(t30_ub_g, t31_ub_decay)
                    dv2_gate_vf(t31_ub_decay, t25_ub_tdvc, t26_ub_dvl, t27_ub_dv2, (pl.min((Vd - (vb * 64)), 64)))
                    pl.set_validshape(t27_ub_dv2, [64, (pl.min((Vd - (vb * 64)), 64))])
                    pl.store(dv2, t27_ub_dv2, [(bh // H), (bh % H), ((N - 1) - rj), 0, (vb * 64)])
                    do_scaled_vf(t28_ub_do_bf, t29_ub_do_f, t32_ub_dexp, (pl.min((T - (((N - 1) - rj) * 64)), 64)), (pl.min((Vd - (vb * 64)), 64)))
                    neg_dv2_vf(t27_ub_dv2, (pl.min((Vd - (vb * 64)), 64)))
                pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=3, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=4, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                if _k0 < 64:
                    pl.set_validshape(t29_ub_do_f, [64, (pl.min((Vd - (vb * 64)), 64))])
                    pl.store(do_gm, t29_ub_do_f, [(bh // H), (bh % H), vb, 0, 0])
                    pl.set_validshape(t27_ub_dv2, [64, (pl.min((Vd - (vb * 64)), 64))])
                    pl.store(dv2neg_gm, t27_ub_dv2, [(bh // H), (bh % H), vb, 0, 0])
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=6, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                if _k0 < 64:
                    dS_update_vf(t23_ub_dS, t24_ub_delta, t32_ub_dexp, Kd, (pl.min((Vd - (vb * 64)), 64)))
                pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=7, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=0, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=0, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=4, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=4, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)


# =============================================================================
#   Wrapper — HARD contract name gdr_bwd_reverse_scan_wrapper
# =============================================================================
def gdr_bwd_reverse_scan_wrapper(q_t, w_t, k_f32, grad_o=None,
                                         decay_exp=None, g_f32=None, dv_local=None):
    """Module 1 entry (module_count=1, so this is the whole operator).

    DESIGN §0 "Wrapper 操作清单" is EMPTY: pure-Python shape arithmetic, output
    allocation, internal GM bridge allocation, and exactly ONE kernel launch.
    No cast / permute / contiguous / slice — all of it lives in the kernel.

    Returns (dh [B,H,N,K,V] fp32, dv2 [B,H,N,C,V] fp32).
    """
    B, T, H, V = grad_o.shape
    N = q_t.shape[2]
    K = q_t.shape[3]
    dev = q_t.device

    dh = torch.empty((B, H, N, K, V), dtype=torch.float32, device=dev)
    dv2 = torch.empty((B, H, N, C, V), dtype=torch.float32, device=dev)
    # internal kernel-scratch GM bridges (DESIGN §6): dS UB->GM->L1, and the two
    # cube-2 operands do_scaled / -dv2.  Not public I/O.
    NV = (V + TV - 1) // TV
    dS_gm = torch.empty((B, H, NV, K, TV), dtype=torch.float32, device=dev)
    do_gm = torch.empty((B, H, NV, C, TV), dtype=torch.float32, device=dev)
    dv2neg_gm = torch.empty((B, H, NV, C, TV), dtype=torch.float32, device=dev)

    max_blocks = get_platform_info().core_num or 1
    num_cores = max(1, min(B * H * ((V + TV - 1) // TV), max_blocks))

    gdr_bwd_reverse_scan_module1_kernel[None, num_cores](
        q_t, w_t, k_f32, dS_gm, do_gm, dv2neg_gm,
        grad_o, decay_exp, g_f32, dv_local, dh, dv2,
        B, T, H, K, V, N, C, TV)
    torch.npu.synchronize()
    return dh, dv2

