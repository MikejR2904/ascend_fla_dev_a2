# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""gdr_bwd_dqkw — final delivery (single fusion module == the FULL operator).

OP5 / parent module M8 (golden_step b3): the chunk-parallel `dqkwg` backward.
BHN-parallel (grid = B*H*N, every (b,h,i) chunk independent, no carried state),
a genuine multi-phase Cube+Vector fusion in ONE jit-compiled kernel:

    vec cast (grad_o bf16->fp32)
      -> cube-1   ds = grad_o @ v_new^T   and   ds_T = v_new @ grad_o^T
      -> vec      triangular mask * exp(decay_i - decay_j) * scale   (lower-incl-diag on ds,
                  upper-incl-diag on ds_T == transpose(mask(ds)))
      -> cube-2   dq_inter = grad_o @ h,  dq_intra = ds @ k,
                  dk_inter = v_new @ dh,  dk_intra = ds_T @ q_s,  dw = dv2 @ h
      -> vec      dq_p / dk_p / dw assemble + dg_p = Sum_K(dq_p*q_raw) - Sum_K(dk_p*k)
                  + the per-chunk scalar dg_last folded onto token C-1.

The kernel body (the 9 `@pl.vector_function` helpers and the single jit-decorated
`GDR_BWD_DQKW`) is the Stage 3.5 KAL transpile product `kal/gdr_bwd_dqkw/staged_module1.py`
carried over VERBATIM -- algorithm, tile set, addresses, loop structure and the
4 cube<->vec auto_mutex handoffs came from there.  Stage 4 added the delivery
wiring: the `gdr_bwd_dqkw_wrapper` entry, the GM workspace allocation and
the DESIGN §8 tests.

Budget: UB 238,432 B (93.9% of 253,952) / mutex 32 of 32 (0..31) /
L0A = L0B = 64 KB at cap / L0C 32 KB / intra+cross flags 0,0 / 11 `pl.matmul`.

Latency pass (production B=1 T=2048 H=8 K=V=128, 28 cube host, msprof warm
median) took this kernel 201.2 -> 160.5 us, aic_mac 0.635 -> 0.721.  Three
changes carry it, all leaving the maths bit-identical:

  * `ds_T` is not computed.  Masked `ds_T == transpose(masked ds)` exactly
    (mask_lower(ds)[j,i] = ds[j,i]*[i<=j]*exp(decay_j-decay_i)*sc, which is
    mask_upper(ds_T)[i,j]), so the second cube-1 matmul, its `do^T` L1 operand,
    its L0C drain, `mask_upper_vf` and the whole `dsT_gm` bridge are gone;
    `dk_intra` reads `ds_gm` back with `order=[4,3]`.  -1 of 12 matmuls.
  * the five L0C output accumulators and `g_l1_dh` got their own mutex pairs
    instead of sharing one (cube-05: a shared id serializes the groups, here
    MMAD against the FixPipe drains).
  * the once-per-tid `ds` rendezvous moved behind the three MMADs in block 0
    that do not depend on `ds`, so `dq_inter` / `dk_inter` / `dw` and the
    64 KB h/dh fetch now cover the vector's mask latency.  This is the big one.

Wrapper does NO core computation: pure-Python shape arithmetic, `torch.empty`
for the two GM bridges (do_gm / ds_gm) and the four outputs, and ONE kernel
launch (not in a loop).  grad_o chunk slicing, the bf16->fp32 cast, all
matmuls, the triangular mask and both reductions happen inside the kernel.
"""

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")

_HERE = os.path.dirname(os.path.abspath(__file__))
_OPDIR = os.path.join(_HERE, "modules")
for _p in (_HERE, _OPDIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info

# ── DESIGN §2.1 key constants (used verbatim) ──
C = 64    # fixed internal chunk rows
TV = 64   # V/K sub-block width

# The runner's older PyPTO build leaves gaps in its DT_* pybind
# coverage: some constants (DT_BF16 among them) arrive as bare ints
# that VF-layer kwarg checks reject ("expected ir::DataType, but got
# int"), while others on the same build are real DataType objects
# (add_rms_norm_dynamic_quant exercised FP16/FP32/INT8/INT16/UINT32
# there). Rebuild the broken ones from the enum value AT MODULE
# SCOPE -- the VF parser inlines helper calls and rejects
# isinstance, so the fixups must be plain names by the time a
# decorated body mentions them. A healthy build passes through.
_DTT = type(pl.DT_FP32)
def _dt_fix(v):
    return v if isinstance(v, _DTT) else _DTT(v)
_PL_BF16 = _dt_fix(pl.DT_BF16)
_PL_FP16 = _dt_fix(pl.DT_FP16)
_PL_FP32 = _dt_fix(pl.DT_FP32)
_PL_INT8 = _dt_fix(pl.DT_INT8)
_PL_INT16 = _dt_fix(pl.DT_INT16)
_PL_INT32 = _dt_fix(pl.DT_INT32)
_PL_INT64 = _dt_fix(pl.DT_INT64)
_PL_UINT8 = _dt_fix(pl.DT_UINT8)
_PL_UINT16 = _dt_fix(pl.DT_UINT16)
_PL_UINT32 = _dt_fix(pl.DT_UINT32)
_PL_UINT64 = _dt_fix(pl.DT_UINT64)

@pl.vector_function
def cast_go_vf(ub_bf, ub_f, valid_c, valid_v):
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    z = vf.full(0.0, active, dtype=_PL_FP32)
    for cgr in pl.range(0, valid_c, 1):
        src = vf.load_align(ub_bf, (cgr) * 64, dist=pl.LoadDist.UNPK_B16)
        dst = vf.astype(src, active, dtype=_PL_FP32, layout=pl.CastLayout.ZERO)
        vf.store_align(ub_f + (cgr) * 64, dst, active)
    for cgpr in pl.range(valid_c, 64, 1):
        vf.store_align(ub_f + (cgpr) * 64, z, active)

@pl.vector_function
def combine_vf(ub_dqi, ub_dqa, ub_dki, ub_dka, ub_dw, ub_qraw, ub_kf, ub_dexp, ub_decay, ub_const, ub_dgp, ub_dgl, valid_k):
    _tmp_maskreg_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    cols = vf.arange(0, dtype=_PL_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    kmask = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=_PL_INT32)
    kmask = vf.lt(cols, valid_k, _tmp_maskreg_0)
    zero = vf.full(0.0, active, dtype=_PL_FP32)
    sc = vf.load_align(ub_const, 1, dist=pl.LoadDist.BRC_B32)
    sk = vf.load_align(ub_const, 0, dist=pl.LoadDist.BRC_B32)
    glast = vf.load_align(ub_decay, 63, dist=pl.LoadDist.BRC_B32)
    for cbc in pl.range(0, 64, 1):
        de = vf.load_align(ub_dexp, cbc, dist=pl.LoadDist.BRC_B32)
        de = vf.mul(de, sc, _tmp_maskreg_1)
        dc = vf.load_align(ub_decay, cbc, dist=pl.LoadDist.BRC_B32)
        gate = vf.exp_sub(glast, dc, _tmp_maskreg_1)
        dqi = vf.load_align(ub_dqi, (cbc) * 64, dtype=_PL_FP32)
        dqa = vf.load_align(ub_dqa, (cbc) * 64, dtype=_PL_FP32)
        dki = vf.load_align(ub_dki, (cbc) * 64, dtype=_PL_FP32)
        dka = vf.load_align(ub_dka, (cbc) * 64, dtype=_PL_FP32)
        dwr = vf.load_align(ub_dw, (cbc) * 64, dtype=_PL_FP32)
        qrr = vf.load_align(ub_qraw, (cbc) * 64, dtype=_PL_FP32)
        kfr = vf.load_align(ub_kf, (cbc) * 64, dtype=_PL_FP32)
        dq = vf.mul(dqi, de, _tmp_maskreg_1)
        dq = vf.add(dq, dqa, _tmp_maskreg_1)
        vf.store_align(ub_dqa + (cbc) * 64, dq, _tmp_maskreg_1)
        dki = vf.mul(dki, gate, _tmp_maskreg_1)
        vf.store_align(ub_dki + (cbc) * 64, dki, _tmp_maskreg_1)
        dka = vf.mul(dka, sk, _tmp_maskreg_1)
        dk = vf.add(dki, dka, _tmp_maskreg_1)
        vf.store_align(ub_dka + (cbc) * 64, dk, _tmp_maskreg_1)
        ndw = vf.neg(dwr, _tmp_maskreg_1)
        vf.store_align(ub_dw + (cbc) * 64, ndw, _tmp_maskreg_1)
        p1v = vf.mul(dq, qrr, _tmp_maskreg_1)
        p1v = vf.select(p1v, zero, kmask)
        s1 = vf.reduce_sum(p1v, _tmp_maskreg_1)
        s1 = vf.mul(s1, sk, _tmp_maskreg_1)
        p2v = vf.mul(dk, kfr, _tmp_maskreg_1)
        p2v = vf.select(p2v, zero, kmask)
        s2 = vf.reduce_sum(p2v, _tmp_maskreg_1)
        prev = vf.load_align(ub_dgp, cbc, dist=pl.LoadDist.BRC_B32)
        acc = vf.add(prev, s1, _tmp_maskreg_1)
        acc = vf.sub(acc, s2, _tmp_maskreg_1)
        vf.store_align(ub_dgp + cbc, acc, _tmp_maskreg_1, dist=pl.StoreDist.FIRST_ELEMENT)
        p3v = vf.mul(dki, kfr, _tmp_maskreg_1)
        p3v = vf.select(p3v, zero, kmask)
        s3 = vf.reduce_sum(p3v, _tmp_maskreg_1)
        # Per-row PARTIAL at a distinct lane.  The transpiled product did a
        # load/add/store on the SINGLE address ub_dgl+0 inside this loop; on real a5
        # those read-modify-writes are not ordered across iterations, so updates are
        # lost.  Measured: every dg_p[..., C-1] whose dg_last != 0 was wrong (sign
        # flips), while all other columns -- which write a DISTINCT address per row,
        # see ub_dgp above -- were exact.  Reduced once after the loop (reduce_acc_vf).
        vf.store_align(ub_dgl + cbc, s3, _tmp_maskreg_1, dist=pl.StoreDist.FIRST_ELEMENT)

@pl.vector_function
def consts_vf(ub_const, Kd):
    _tmp_maskreg_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    ki = vf.full(Kd, _tmp_maskreg_0, dtype=_PL_INT32)
    kf = vf.astype(ki, _tmp_maskreg_1, dtype=_PL_FP32, layout=pl.CastLayout.ZERO)
    sk = vf.sqrt(kf, _tmp_maskreg_1)
    one = vf.full(1.0, _tmp_maskreg_1, dtype=_PL_FP32)
    sc = vf.div(one, sk, _tmp_maskreg_1)
    vf.store_align(ub_const + 0, sk, _tmp_maskreg_1, dist=pl.StoreDist.FIRST_ELEMENT)
    vf.store_align(ub_const + 1, sc, _tmp_maskreg_1, dist=pl.StoreDist.FIRST_ELEMENT)

@pl.vector_function
def dexp_vf(ub_decay, ub_dexp):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    d = vf.load_align(ub_decay, 0, dtype=_PL_FP32)
    d = vf.exp(d, _tmp_maskreg_0)
    vf.store_align(ub_dexp + 0, d, _tmp_maskreg_0)

@pl.vector_function
def dglast_fold_vf(ub_dgp, ub_dgl, ub_dgkv, ub_decay):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    glast = vf.load_align(ub_decay, 63, dist=pl.LoadDist.BRC_B32)
    eg = vf.exp(glast, _tmp_maskreg_0)
    kv = vf.load_align(ub_dgkv, 0, dist=pl.LoadDist.BRC_B32)
    dgl = vf.load_align(ub_dgl, 0, dist=pl.LoadDist.BRC_B32)
    dglast = vf.mul(eg, kv, _tmp_maskreg_0)
    dglast = vf.add(dglast, dgl, _tmp_maskreg_0)
    prev = vf.load_align(ub_dgp, 63, dist=pl.LoadDist.BRC_B32)
    prev = vf.add(prev, dglast, _tmp_maskreg_0)
    vf.store_align(ub_dgp + 63, prev, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)

@pl.vector_function
def mask_lower_vf(ub_decay, ub_ds, ub_const):
    _tmp_maskreg_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, active, dtype=_PL_FP32)
    cols = vf.arange(0, dtype=_PL_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    dvec = vf.load_align(ub_decay, 0, dtype=_PL_FP32)
    sc = vf.load_align(ub_const, 1, dist=pl.LoadDist.BRC_B32)
    for mli in pl.range(0, 64, 1):
        di = vf.load_align(ub_decay, mli, dist=pl.LoadDist.BRC_B32)
        gexp = vf.exp_sub(di, dvec, active)
        m = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=_PL_INT32)
        m = vf.le(cols, mli, _tmp_maskreg_0)
        gexp = vf.select(gexp, zero, m)
        gexp = vf.mul(gexp, sc, _tmp_maskreg_1)
        dsrow = vf.load_align(ub_ds, (mli) * 64, dtype=_PL_FP32)
        dsrow = vf.mul(dsrow, gexp, _tmp_maskreg_1)
        vf.store_align(ub_ds + (mli) * 64, dsrow, _tmp_maskreg_1)

@pl.vector_function
def mask_upper_vf(ub_decay, ub_dsT, ub_const):
    _tmp_maskreg_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, active, dtype=_PL_FP32)
    cols = vf.arange(0, dtype=_PL_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    dvec = vf.load_align(ub_decay, 0, dtype=_PL_FP32)
    ndvec = vf.neg(dvec, _tmp_maskreg_0)
    sc = vf.load_align(ub_const, 1, dist=pl.LoadDist.BRC_B32)
    for mui in pl.range(0, 64, 1):
        di = vf.load_align(ub_decay, mui, dist=pl.LoadDist.BRC_B32)
        ndi = vf.neg(di, _tmp_maskreg_0)
        gexp = vf.exp_sub(ndi, ndvec, active)
        m = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=_PL_INT32)
        m = vf.ge(cols, mui, _tmp_maskreg_1)
        gexp = vf.select(gexp, zero, m)
        gexp = vf.mul(gexp, sc, _tmp_maskreg_0)
        dsrow = vf.load_align(ub_dsT, (mui) * 64, dtype=_PL_FP32)
        dsrow = vf.mul(dsrow, gexp, _tmp_maskreg_0)
        vf.store_align(ub_dsT + (mui) * 64, dsrow, _tmp_maskreg_0)

@pl.vector_function
def shdh_vf(ub_sh, ub_dh, ub_dgkv, valid_kk, valid_v):
    _tmp_maskreg_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    cols = vf.arange(0, dtype=_PL_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    vmask = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=_PL_INT32)
    vmask = vf.lt(cols, valid_v, _tmp_maskreg_0)
    zero = vf.full(0.0, active, dtype=_PL_FP32)
    # Same-address RMW hazard as combine_vf: write a per-row PARTIAL to a distinct
    # lane, and zero the lanes beyond valid_kk so the later reduce sees no stale data.
    for shr in pl.range(0, valid_kk, 1):
        sh = vf.load_align(ub_sh, (shr) * 64, dtype=_PL_FP32)
        dhr = vf.load_align(ub_dh, (shr) * 64, dtype=_PL_FP32)
        pv = vf.mul(sh, dhr, _tmp_maskreg_1)
        pv = vf.select(pv, zero, vmask)
        s = vf.reduce_sum(pv, _tmp_maskreg_1)
        vf.store_align(ub_dgkv + shr, s, _tmp_maskreg_1, dist=pl.StoreDist.FIRST_ELEMENT)
    for shp in pl.range(valid_kk, 64, 1):
        vf.store_align(ub_dgkv + shp, zero, _tmp_maskreg_1, dist=pl.StoreDist.FIRST_ELEMENT)

@pl.vector_function
def reduce_acc_vf(ub_vec, ub_scalar):
    """sum the 64 per-row partials in ub_vec and accumulate into ub_scalar[0].

    Executed ONCE per producer loop, so the single read-modify-write on ub_scalar
    carries no cross-iteration ordering hazard.
    """
    _m = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    v = vf.load_align(ub_vec, 0, dtype=_PL_FP32)
    s = vf.reduce_sum(v, _m)
    prev = vf.load_align(ub_scalar, 0, dist=pl.LoadDist.BRC_B32)
    acc = vf.add(prev, s, _m)
    vf.store_align(ub_scalar + 0, acc, _m, dist=pl.StoreDist.FIRST_ELEMENT)

@pl.vector_function
def zero_row_vf(ub, n):
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    z = vf.full(0.0, active, dtype=_PL_FP32)
    vf.store_align(ub + 0, z, active)


@pl.jit(auto_mutex=True)
def GDR_BWD_DQKW(v_new: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], k_f32: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], q_s: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], s_hist: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], dh: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], dv2: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], do_gm: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], ds_gm: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], grad_o: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16], decay: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], dq_p: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], dk_p: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], dw: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], dg_p: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], B: pl.DT_INT32, T: pl.DT_INT32, H: pl.DT_INT32, Kd: pl.DT_INT32, Vd: pl.DT_INT32, N: pl.DT_INT32, Cc: pl.DT_INT32, VT: pl.DT_INT32):
    g__l0a__v0 = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1]), addrs=[0x0, 0x8000], mutex_ids=[0, 1])
    g__l0a__v1 = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1]), addrs=[0x0, 0x8000], mutex_ids=[0, 1])
    g__l0b__v0 = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1]), addrs=[0x0, 0x8000], mutex_ids=[2, 3])
    g__l0b__v1 = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1]), addrs=[0x0, 0x8000], mutex_ids=[2, 3])
    g_ub_do_bf = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x0, mutex_ids=[4])
    g_ub_do_f = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x2000, mutex_ids=[5])
    g_ub_ds = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x6000, mutex_ids=[6])
    g_ub_dsT = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0xa000, mutex_ids=[6])
    g_ub_decay = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0xe000, mutex_ids=[7])
    g_ub_dexp = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0xe100, mutex_ids=[8])
    g_ub_const = pl.make_tile_group(type=pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0xe200, mutex_ids=[8])
    g_ub_dqi = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0xe220, mutex_ids=[9, 10])
    g_ub_dqa = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x16220, mutex_ids=[11, 12])
    g_ub_dki = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x1e220, mutex_ids=[9, 10])
    g_ub_dka = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x26220, mutex_ids=[11, 12])
    g_ub_dw = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x2e220, mutex_ids=[11, 12])
    g_ub_kf = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x36220, mutex_ids=[13])
    g_ub_dgp = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x3a220, mutex_ids=[14])
    # dg_last accumulators.  dgl/dgkv hold 64 per-row PARTIALS (one lane per row) and
    # dgls/dgkvs the reduced scalars; see combine_vf / shdh_vf / reduce_acc_vf for why
    # the single-address accumulation the transpiler emitted had to be split.
    # UB tail: 0x3a220 dgp[1,64] | 0x3a320 dgl[1,64] | 0x3a420 dgkv[1,64]
    #          | 0x3a520 dgls[1,8] | 0x3a540 dgkvs[1,8]  -> ends 238,944 of 253,952.
    g_ub_dgl = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x3a320, mutex_ids=[8])
    g_ub_dgkv = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x3a420, mutex_ids=[8])
    g_ub_dgls = pl.make_tile_group(type=pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x3a520, mutex_ids=[8])
    g_ub_dgkvs = pl.make_tile_group(type=pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], pad=pl.TilePad.zero), addrs=0x3a540, mutex_ids=[8])
    g_l1_vn = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x8000, mutex_ids=[15])
    g_l0c_ds = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x0, mutex_ids=[16])
    g_l1_do2 = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x10000, mutex_ids=[15])
    g_l1_h = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x18000, mutex_ids=[17, 18])
    g_l1_k = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x28000, mutex_ids=[15])
    g_l1_vn2 = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x30000, mutex_ids=[15])
    g_l1_dh = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x38000, mutex_ids=[30, 31])
    g_l1_dv2 = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x48000, mutex_ids=[15])
    g_l1_dsL = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x50000, mutex_ids=[19])
    g_l1_dsTL = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.ZN, valid_shape=[-1, -1]), addrs=0x54000, mutex_ids=[19])
    g_l1_qraw = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]), addrs=0x58000, mutex_ids=[15])
    g_l0c_dqi = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x8000, mutex_ids=[20, 21])
    g_l0c_dqa = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x10000, mutex_ids=[22, 23])
    g_l0c_dki = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x18000, mutex_ids=[24, 25])
    g_l0c_dka = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x20000, mutex_ids=[26, 27])
    g_l0c_dw = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x28000, mutex_ids=[28, 29])
    with pl.section_cube():
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=4, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        t1_ub_ds = g_ub_ds.next()
        t4_l1_vn = g_l1_vn.next()
        t5_l0c_ds = g_l0c_ds.next()
        t10_l1_dv2 = g_l1_dv2.next()
        t12_l1_dsTL = g_l1_dsTL.next()
        for tid in pl.range((((((B * H) * N) + pl.get_block_num() - 1) // pl.get_block_num()) * (pl.get_block_idx() // pl.get_subblock_num())), (pl.min(((((((B * H) * N) + pl.get_block_num() - 1) // pl.get_block_num()) * (pl.get_block_idx() // pl.get_subblock_num())) + ((((B * H) * N) + pl.get_block_num() - 1) // pl.get_block_num())), ((B * H) * N))), 1):
            # One div + two mod per tid instead of three per GM access: the
            # (b, h, n) decode used to be re-emitted inline at all 29 load /
            # store sites, which is what put the cube SCALAR pipe at 0.079.
            _bb = (tid // (H * N))
            _hn = (tid % (H * N))
            _hh = (_hn // N)
            _nn = (_hn % N)
            # cube-05: rotate the tid-carried L1 operands so tid i+1's MTE2 can
            # issue while tid i's MMADs still read the other buffer.
            t7_l1_do2 = g_l1_do2.next()
            t8_l1_k = g_l1_k.next()
            t9_l1_vn2 = g_l1_vn2.next()
            t11_l1_dsL = g_l1_dsL.next()
            t13_l1_qraw = g_l1_qraw.next()
            pl.set_validshape(t4_l1_vn, [Vd, 64])
            pl.load(t4_l1_vn, v_new, [_bb, _hh, _nn, 0, 0], order=[4, 3])
            pl.set_validshape(t9_l1_vn2, [64, Vd])
            pl.load(t9_l1_vn2, v_new, [_bb, _hh, _nn, 0, 0])
            # cube-05: k / dv2 / q_s come straight from GM inputs -- no vector
            # dependency -- so issue them ABOVE the first rendezvous instead of
            # after the ds drain.  Previously these 96 KB of MTE2 sat behind
            # wait(ev1), so the cube idled at the barrier and then loaded.
            pl.set_validshape(t8_l1_k, [64, Kd])
            pl.load(t8_l1_k, k_f32, [_bb, _hh, _nn, 0, 0])
            pl.set_validshape(t10_l1_dv2, [64, Vd])
            pl.load(t10_l1_dv2, dv2, [_bb, _hh, _nn, 0, 0])
            pl.set_validshape(t13_l1_qraw, [64, Kd])
            pl.load(t13_l1_qraw, q_s, [_bb, _hh, _nn, 0, 0])
            # s_hist / dh are plain GM inputs too, and g_l1_h / g_l1_dh are
            # depth-2, so BOTH k-slices fit at once: pull all 128 KB above the
            # ev1 / ev3 / ev7 rendezvous rather than fetching each slice inside
            # its own kv block, where it sat behind a barrier.
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.set_validshape(t7_l1_do2, [64, Vd])
            pl.load(t7_l1_do2, do_gm, [_bb, _hh, _nn, 0, 0])
            t14__l0a__v1 = g__l0a__v1.next()
            t15__l0a = g__l0a__v0.next()
            t16__l0b__v1 = g__l0b__v1.next()
            t17__l0b = g__l0b__v0.next()
            pl.move(t15__l0a, t7_l1_do2, [0, 0])
            pl.move(t17__l0b, t4_l1_vn, [0, 0])
            pl.set_validshape(t15__l0a, [64, Vd])
            pl.set_validshape(t17__l0b, [Vd, 64])
            pl.set_validshape(t5_l0c_ds, [64, 64])
            pl.matmul(t5_l0c_ds, t15__l0a, t17__l0b)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=3, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.set_validshape(t5_l0c_ds, [64, 64])
            pl.move(t1_ub_ds, t5_l0c_ds, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            # --- the two k-slice blocks ----------------------------------------
            # Each block runs its three ds-free MMADs (dq_inter / dk_inter / dw)
            # first; block 0 then takes the once-per-tid ds rendezvous and the
            # two ds-dependent MMADs (dq_intra / dk_intra).  Kept unrolled: the
            # pl.range form measured +1.1% on trip/scalar overhead.
            if Kd > 0:
                t22_l1_h = g_l1_h.next()
                t23_l1_dh = g_l1_dh.next()
                # Keep the h / dh fetch adjacent to its consuming MMADs.  MTE2
                # runs at platform bandwidth whenever it is busy, so issuing
                # these 64 KB a barrier earlier does not make them land sooner
                # -- it only pushes the first-needed operand further back in a
                # saturated queue (measured: +2.4% when hoisted to a prologue).
                pl.set_validshape(t22_l1_h, [Vd, (pl.min(Kd, 64))])
                pl.load(t22_l1_h, s_hist, [_bb, _hh, _nn, 0, 0], order=[4, 3])
                pl.set_validshape(t23_l1_dh, [Vd, (pl.min(Kd, 64))])
                pl.load(t23_l1_dh, dh, [_bb, _hh, _nn, 0, 0], order=[4, 3])
                # dq_inter / dk_inter / dw need no `ds`
                t24_l0c_dqi = g_l0c_dqi.next()
                t25__l0a__v1 = g__l0a__v1.next()
                t26__l0a = g__l0a__v0.next()
                t27__l0b__v1 = g__l0b__v1.next()
                t28__l0b = g__l0b__v0.next()
                pl.move(t26__l0a, t7_l1_do2, [0, 0])
                pl.move(t28__l0b, t22_l1_h, [0, 0])
                pl.set_validshape(t26__l0a, [64, Vd])
                pl.set_validshape(t28__l0b, [Vd, 64])
                pl.set_validshape(t24_l0c_dqi, [64, (pl.min(Kd, 64))])
                pl.matmul(t24_l0c_dqi, t26__l0a, t28__l0b)
                t34_l0c_dki = g_l0c_dki.next()
                t35__l0a__v1 = g__l0a__v1.next()
                t36__l0a = g__l0a__v0.next()
                t37__l0b__v1 = g__l0b__v1.next()
                t38__l0b = g__l0b__v0.next()
                pl.move(t36__l0a, t9_l1_vn2, [0, 0])
                pl.move(t38__l0b, t23_l1_dh, [0, 0])
                pl.set_validshape(t36__l0a, [64, Vd])
                pl.set_validshape(t38__l0b, [Vd, 64])
                pl.set_validshape(t34_l0c_dki, [64, (pl.min(Kd, 64))])
                pl.matmul(t34_l0c_dki, t36__l0a, t38__l0b)
                t44_l0c_dw = g_l0c_dw.next()
                t45__l0a__v1 = g__l0a__v1.next()
                t46__l0a = g__l0a__v0.next()
                t47__l0b__v1 = g__l0b__v1.next()
                t48__l0b = g__l0b__v0.next()
                pl.move(t46__l0a, t10_l1_dv2, [0, 0])
                pl.move(t48__l0b, t22_l1_h, [0, 0])
                pl.set_validshape(t46__l0a, [64, Vd])
                pl.set_validshape(t48__l0b, [Vd, 64])
                pl.set_validshape(t44_l0c_dw, [64, (pl.min(Kd, 64))])
                pl.matmul(t44_l0c_dw, t46__l0a, t48__l0b)
                # ds rendezvous once per tid, behind the three ds-free MMADs
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                pl.set_validshape(t11_l1_dsL, [64, 64])
                pl.load(t11_l1_dsL, ds_gm, [_bb, _hh, _nn, 0, 0])
                pl.set_validshape(t12_l1_dsTL, [64, 64])
                pl.load(t12_l1_dsTL, ds_gm, [_bb, _hh, _nn, 0, 0], order=[4, 3])
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=4, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                t29_l0c_dqa = g_l0c_dqa.next()
                t30__l0a__v0 = g__l0a__v0.next()
                t31__l0a = g__l0a__v1.next()
                t32__l0b__v0 = g__l0b__v0.next()
                t33__l0b = g__l0b__v1.next()
                pl.move(t31__l0a, t11_l1_dsL, [0, 0])
                pl.move(t33__l0b, t8_l1_k, [0, 0])
                pl.set_validshape(t31__l0a, [64, 64])
                pl.set_validshape(t33__l0b, [64, 64])
                pl.set_validshape(t29_l0c_dqa, [64, (pl.min(Kd, 64))])
                pl.matmul(t29_l0c_dqa, t31__l0a, t33__l0b)
                t39_l0c_dka = g_l0c_dka.next()
                t40__l0a__v0 = g__l0a__v0.next()
                t41__l0a = g__l0a__v1.next()
                t42__l0b__v0 = g__l0b__v0.next()
                t43__l0b = g__l0b__v1.next()
                pl.move(t41__l0a, t12_l1_dsTL, [0, 0])
                pl.move(t43__l0b, t13_l1_qraw, [0, 0])
                pl.set_validshape(t41__l0a, [64, 64])
                pl.set_validshape(t43__l0b, [64, 64])
                pl.set_validshape(t39_l0c_dka, [64, (pl.min(Kd, 64))])
                pl.matmul(t39_l0c_dka, t41__l0a, t43__l0b)
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=7, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                t49_ub_dqi = g_ub_dqi.next()
                pl.set_validshape(t24_l0c_dqi, [64, (pl.min(Kd, 64))])
                pl.move(t49_ub_dqi, t24_l0c_dqi, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                t50_ub_dqa = g_ub_dqa.next()
                pl.set_validshape(t29_l0c_dqa, [64, (pl.min(Kd, 64))])
                pl.move(t50_ub_dqa, t29_l0c_dqa, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                t51_ub_dki = g_ub_dki.next()
                pl.set_validshape(t34_l0c_dki, [64, (pl.min(Kd, 64))])
                pl.move(t51_ub_dki, t34_l0c_dki, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                t52_ub_dka = g_ub_dka.next()
                pl.set_validshape(t39_l0c_dka, [64, (pl.min(Kd, 64))])
                pl.move(t52_ub_dka, t39_l0c_dka, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                t53_ub_dw = g_ub_dw.next()
                pl.set_validshape(t44_l0c_dw, [64, (pl.min(Kd, 64))])
                pl.move(t53_ub_dw, t44_l0c_dw, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=6, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            if Kd > 64:
                t54_l1_h = g_l1_h.next()
                _k4 = (64 + (pl.min((Kd - 64), 64)))
                _k8 = (_k4 - 64)
                pl.set_validshape(t54_l1_h, [Vd, _k8])
                pl.load(t54_l1_h, s_hist, [_bb, _hh, _nn, 64, 0], order=[4, 3])
                t55_l1_dh = g_l1_dh.next()
                pl.set_validshape(t55_l1_dh, [Vd, _k8])
                pl.load(t55_l1_dh, dh, [_bb, _hh, _nn, 64, 0], order=[4, 3])
                t56_l0c_dqi = g_l0c_dqi.next()
                t57__l0a__v1 = g__l0a__v1.next()
                t58__l0a = g__l0a__v0.next()
                t59__l0b__v1 = g__l0b__v1.next()
                t60__l0b = g__l0b__v0.next()
                pl.move(t58__l0a, t7_l1_do2, [0, 0])
                pl.move(t60__l0b, t54_l1_h, [0, 0])
                pl.set_validshape(t58__l0a, [64, Vd])
                pl.set_validshape(t60__l0b, [Vd, 64])
                pl.set_validshape(t56_l0c_dqi, [64, (pl.min((Kd - 64), 64))])
                pl.matmul(t56_l0c_dqi, t58__l0a, t60__l0b)
                t61_l0c_dqa = g_l0c_dqa.next()
                t62__l0a__v0 = g__l0a__v0.next()
                t63__l0a = g__l0a__v1.next()
                t64__l0b__v0 = g__l0b__v0.next()
                t65__l0b = g__l0b__v1.next()
                pl.move(t63__l0a, t11_l1_dsL, [0, 0])
                pl.move(t65__l0b, t8_l1_k, [0, 64])
                pl.set_validshape(t63__l0a, [64, 64])
                pl.set_validshape(t65__l0b, [64, 64])
                pl.set_validshape(t61_l0c_dqa, [64, (pl.min((Kd - 64), 64))])
                pl.matmul(t61_l0c_dqa, t63__l0a, t65__l0b)
                t66_l0c_dki = g_l0c_dki.next()
                t67__l0a__v1 = g__l0a__v1.next()
                t68__l0a = g__l0a__v0.next()
                t69__l0b__v1 = g__l0b__v1.next()
                t70__l0b = g__l0b__v0.next()
                pl.move(t68__l0a, t9_l1_vn2, [0, 0])
                pl.move(t70__l0b, t55_l1_dh, [0, 0])
                pl.set_validshape(t68__l0a, [64, Vd])
                pl.set_validshape(t70__l0b, [Vd, 64])
                pl.set_validshape(t66_l0c_dki, [64, (pl.min((Kd - 64), 64))])
                pl.matmul(t66_l0c_dki, t68__l0a, t70__l0b)
                t71_l0c_dka = g_l0c_dka.next()
                t72__l0a__v0 = g__l0a__v0.next()
                t73__l0a = g__l0a__v1.next()
                t74__l0b__v0 = g__l0b__v0.next()
                t75__l0b = g__l0b__v1.next()
                pl.move(t73__l0a, t12_l1_dsTL, [0, 0])
                pl.move(t75__l0b, t13_l1_qraw, [0, 64])
                pl.set_validshape(t73__l0a, [64, 64])
                pl.set_validshape(t75__l0b, [64, 64])
                pl.set_validshape(t71_l0c_dka, [64, (pl.min((Kd - 64), 64))])
                pl.matmul(t71_l0c_dka, t73__l0a, t75__l0b)
                t76_l0c_dw = g_l0c_dw.next()
                t77__l0a__v1 = g__l0a__v1.next()
                t78__l0a = g__l0a__v0.next()
                t79__l0b__v1 = g__l0b__v1.next()
                t80__l0b = g__l0b__v0.next()
                pl.move(t78__l0a, t10_l1_dv2, [0, 0])
                pl.move(t80__l0b, t54_l1_h, [0, 0])
                pl.set_validshape(t78__l0a, [64, Vd])
                pl.set_validshape(t80__l0b, [Vd, 64])
                pl.set_validshape(t76_l0c_dw, [64, (pl.min((Kd - 64), 64))])
                pl.matmul(t76_l0c_dw, t78__l0a, t80__l0b)
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=7, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                t81_ub_dqi = g_ub_dqi.next()
                pl.set_validshape(t56_l0c_dqi, [64, (pl.min((Kd - 64), 64))])
                pl.move(t81_ub_dqi, t56_l0c_dqi, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                t82_ub_dqa = g_ub_dqa.next()
                pl.set_validshape(t61_l0c_dqa, [64, (pl.min((Kd - 64), 64))])
                pl.move(t82_ub_dqa, t61_l0c_dqa, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                t83_ub_dki = g_ub_dki.next()
                pl.set_validshape(t66_l0c_dki, [64, (pl.min((Kd - 64), 64))])
                pl.move(t83_ub_dki, t66_l0c_dki, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                t84_ub_dka = g_ub_dka.next()
                pl.set_validshape(t71_l0c_dka, [64, (pl.min((Kd - 64), 64))])
                pl.move(t84_ub_dka, t71_l0c_dka, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                t85_ub_dw = g_ub_dw.next()
                pl.set_validshape(t76_l0c_dw, [64, (pl.min((Kd - 64), 64))])
                pl.move(t85_ub_dw, t76_l0c_dw, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=6, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=3, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=7, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
    with pl.section_vector():
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=3, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=7, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        t86_ub_do_bf = g_ub_do_bf.next()
        t87_ub_do_f = g_ub_do_f.next()
        t88_ub_ds = g_ub_ds.next()
        t89_ub_dsT = g_ub_dsT.next()
        t90_ub_decay = g_ub_decay.next()
        t91_ub_dexp = g_ub_dexp.next()
        t92_ub_const = g_ub_const.next()
        t93_ub_kf = g_ub_kf.next()
        t94_ub_dgp = g_ub_dgp.next()
        t95_ub_dgl = g_ub_dgl.next()
        t96_ub_dgkv = g_ub_dgkv.next()
        t95b_ub_dgls = g_ub_dgls.next()
        t96b_ub_dgkvs = g_ub_dgkvs.next()
        for tid in pl.range((((((B * H) * N) + pl.get_block_num() - 1) // pl.get_block_num()) * (pl.get_block_idx() // pl.get_subblock_num())), (pl.min(((((((B * H) * N) + pl.get_block_num() - 1) // pl.get_block_num()) * (pl.get_block_idx() // pl.get_subblock_num())) + ((((B * H) * N) + pl.get_block_num() - 1) // pl.get_block_num())), ((B * H) * N))), 1):
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=0, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            _bb = (tid // (H * N))
            _hn = (tid % (H * N))
            _hh = (_hn // N)
            _nn = (_hn % N)
            _k3 = (pl.get_subblock_idx() * 64)
            if _k3 < 64:
                for vb0 in pl.range(0, ((Vd + 64 - 1) // 64), 1):
                    _k2 = (_nn * 64)
                    _k7 = (T - _k2)
                    _k9 = (pl.min(_k7, 64))
                    if _k9 > 0:
                        pl.set_validshape(t86_ub_do_bf, [_k9, (pl.min((Vd - (vb0 * 64)), 64))])
                        pl.load(t86_ub_do_bf, grad_o, [_bb, _k2, _hh, (vb0 * 64)], order=[1, 3])
                    cast_go_vf(t86_ub_do_bf, t87_ub_do_f, _k9, (pl.min((Vd - (vb0 * 64)), 64)))
                    pl.set_validshape(t87_ub_do_f, [64, (pl.min((Vd - (vb0 * 64)), 64))])
                    pl.store(do_gm, t87_ub_do_f, [_bb, _hh, _nn, 0, (vb0 * 64)])
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            if _k3 < 64:
                pl.set_validshape(t90_ub_decay, [1, 64])
                pl.load(t90_ub_decay, decay, [_bb, _hh, _nn, 0])
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            if _k3 < 64:
                consts_vf(t92_ub_const, Kd)
                dexp_vf(t90_ub_decay, t91_ub_dexp)
                mask_lower_vf(t90_ub_decay, t88_ub_ds, t92_ub_const)
                pl.set_validshape(t88_ub_ds, [64, 64])
                pl.store(ds_gm, t88_ub_ds, [_bb, _hh, _nn, 0, 0])
            pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=3, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=4, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            if _k3 < 64:
                zero_row_vf(t94_ub_dgp, Cc)
                zero_row_vf(t95_ub_dgl, 1)
                zero_row_vf(t96_ub_dgkv, 1)
                zero_row_vf(t95b_ub_dgls, 1)
                zero_row_vf(t96b_ub_dgkvs, 1)
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            if Kd > 0:
                if _k3 < 64:
                    pl.set_validshape(t87_ub_do_f, [64, (pl.min(Kd, 64))])
                    pl.load(t87_ub_do_f, q_s, [_bb, _hh, _nn, 0, 0])
                    pl.set_validshape(t93_ub_kf, [64, (pl.min(Kd, 64))])
                    pl.load(t93_ub_kf, k_f32, [_bb, _hh, _nn, 0, 0])
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=6, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                t106_ub_dqi = g_ub_dqi.next()
                t107_ub_dqa = g_ub_dqa.next()
                t108_ub_dki = g_ub_dki.next()
                t109_ub_dka = g_ub_dka.next()
                t110_ub_dw = g_ub_dw.next()
                if _k3 < 64:
                    combine_vf(t106_ub_dqi, t107_ub_dqa, t108_ub_dki, t109_ub_dka, t110_ub_dw, t87_ub_do_f, t93_ub_kf, t91_ub_dexp, t90_ub_decay, t92_ub_const, t94_ub_dgp, t95_ub_dgl, (pl.min(Kd, 64)))
                    reduce_acc_vf(t95_ub_dgl, t95b_ub_dgls)
                    pl.set_validshape(t107_ub_dqa, [64, (pl.min(Kd, 64))])
                    pl.store(dq_p, t107_ub_dqa, [_bb, _hh, _nn, 0, 0])
                    pl.set_validshape(t109_ub_dka, [64, (pl.min(Kd, 64))])
                    pl.store(dk_p, t109_ub_dka, [_bb, _hh, _nn, 0, 0])
                    pl.set_validshape(t110_ub_dw, [64, (pl.min(Kd, 64))])
                    pl.store(dw, t110_ub_dw, [_bb, _hh, _nn, 0, 0])
                pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=7, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            if Kd > 64:
                if _k3 < 64:
                    pl.set_validshape(t87_ub_do_f, [64, (pl.min((Kd - 64), 64))])
                    pl.load(t87_ub_do_f, q_s, [_bb, _hh, _nn, 0, 64])
                    pl.set_validshape(t93_ub_kf, [64, (pl.min((Kd - 64), 64))])
                    pl.load(t93_ub_kf, k_f32, [_bb, _hh, _nn, 0, 64])
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=6, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                t111_ub_dqi = g_ub_dqi.next()
                t112_ub_dqa = g_ub_dqa.next()
                t113_ub_dki = g_ub_dki.next()
                t114_ub_dka = g_ub_dka.next()
                t115_ub_dw = g_ub_dw.next()
                if _k3 < 64:
                    combine_vf(t111_ub_dqi, t112_ub_dqa, t113_ub_dki, t114_ub_dka, t115_ub_dw, t87_ub_do_f, t93_ub_kf, t91_ub_dexp, t90_ub_decay, t92_ub_const, t94_ub_dgp, t95_ub_dgl, (pl.min((Kd - 64), 64)))
                    reduce_acc_vf(t95_ub_dgl, t95b_ub_dgls)
                    pl.set_validshape(t112_ub_dqa, [64, (pl.min((Kd - 64), 64))])
                    pl.store(dq_p, t112_ub_dqa, [_bb, _hh, _nn, 0, 64])
                    pl.set_validshape(t114_ub_dka, [64, (pl.min((Kd - 64), 64))])
                    pl.store(dk_p, t114_ub_dka, [_bb, _hh, _nn, 0, 64])
                    pl.set_validshape(t115_ub_dw, [64, (pl.min((Kd - 64), 64))])
                    pl.store(dw, t115_ub_dw, [_bb, _hh, _nn, 0, 64])
                pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=7, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            if _k3 < 64:
                for sk in pl.range(0, ((Kd + 64 - 1) // 64), 1):
                    for sv in pl.range(0, ((Vd + 64 - 1) // 64), 1):
                        _k0 = (Kd - (sk * 64))
                        _k1 = (Vd - (sv * 64))
                        _k5 = (pl.min(_k0, 64))
                        _k6 = (pl.min(_k1, 64))
                        pl.set_validshape(t88_ub_ds, [_k5, _k6])
                        pl.load(t88_ub_ds, s_hist, [_bb, _hh, _nn, (sk * 64), (sv * 64)])
                        pl.set_validshape(t89_ub_dsT, [_k5, _k6])
                        pl.load(t89_ub_dsT, dh, [_bb, _hh, _nn, (sk * 64), (sv * 64)])
                        shdh_vf(t88_ub_ds, t89_ub_dsT, t96_ub_dgkv, _k5, _k6)
                        reduce_acc_vf(t96_ub_dgkv, t96b_ub_dgkvs)
                dglast_fold_vf(t94_ub_dgp, t95b_ub_dgls, t96b_ub_dgkvs, t90_ub_decay)
                pl.set_validshape(t94_ub_dgp, [1, 64])
                pl.store(dg_p, t94_ub_dgp, [_bb, _hh, _nn, 0])
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=0, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=4, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)


# =============================================================================
#   Wrapper — the Stage 4 L1 module-1 entry point.
# =============================================================================
def gdr_bwd_dqkw_wrapper(grad_o, v_new, k_f32, q_s, s_hist, dh,
                                 dv2, g_f32=None, decay=None):
    """OP5 entry: allocate the 3 GM bridges + 4 outputs, launch the kernel ONCE.

    Signature mirrors `module_interfaces.yaml: primary_inputs` (== the golden
    `gdr_bwd_dqkw_golden_stage1` signature).  `g_f32` is a parent-fidelity
    contract input and is unused by the core math (DESIGN §0); it is accepted
    positionally and defaulted so a caller may omit it.

    Returns (dq_p, dk_p, dw, dg_p), all fp32 -- the operator's 4 final outputs.

    Host does only: python scalar/shape arithmetic, `torch.empty` allocation and
    one kernel launch.  No `.contiguous()`, `.to()`, `.permute()` or any other
    device op -- DESIGN §0 "Wrapper 操作清单" is empty by construction.
    """
    B, T, H, V = grad_o.shape
    K = k_f32.shape[-1]
    N = (T + C - 1) // C                 # chunk count (ceiling), pure python
    dev = grad_o.device
    total = B * H * N                    # BHN task grid

    # GM bridges (DESIGN §6: ds / ds_T / do routed UB->GM->L1; no Acc->Mat path).
    do_gm = torch.empty((B, H, N, C, V), dtype=torch.float32, device=dev)
    ds_gm = torch.empty((B, H, N, C, C), dtype=torch.float32, device=dev)

    # Outputs (module_interfaces.yaml: modules[0].outputs).
    dq_p = torch.empty((B, H, N, C, K), dtype=torch.float32, device=dev)
    dk_p = torch.empty((B, H, N, C, K), dtype=torch.float32, device=dev)
    dw = torch.empty((B, H, N, C, K), dtype=torch.float32, device=dev)
    dg_p = torch.empty((B, H, N, C), dtype=torch.float32, device=dev)

    # DESIGN §5: block_dim = min(max_cores, B*H*N); ONE launch, not in a loop.
    max_blocks = get_platform_info().core_num or 1
    num_cores = max(1, min(total, max_blocks))

    GDR_BWD_DQKW[None, num_cores](
        v_new, k_f32, q_s, s_hist, dh, dv2,
        do_gm, ds_gm,
        grad_o, decay,
        dq_p, dk_p, dw, dg_p,
        B, T, H, K, V, N, C, TV)
    torch.npu.synchronize()
    return dq_p, dk_p, dw, dg_p

