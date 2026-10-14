# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""gdr_bwd_dv_local -- Stage 4 L1 staged deliverable, Module 1 (single, FULL op).

OP3 of gated_delta_rule_bwd (parent module M6, golden_step b1), BHN-parallel,
cube -> vector -> cube fusion inside ONE @pl.jit kernel.

    A_l[C,C]      = k_f32[C,K] @ q_t[K,C]          (scale pre-folded into q_t upstream)
    A_l          *= exp(decay_j - decay_i),  decay = cumsum(g_f32)
    A_l           = upper-triangle INCLUSIVE of diagonal (keep j >= i, else 0)
    dv_local[C,V] = A_l @ grad_o_chunk[C,V]        (grad_o bf16 -> fp32 at load)

PROVENANCE -- the kernel below (the three vector_function helpers and the single
jit-decorated body) is the Stage 3.5 KAL transpiled product `kal/gdr_bwd_dv_local/staged_module1.py`, carried
over verbatim apart from the kernel's function name.  That product was simulator-verified
12/12 (4 shapes x 3 seeds, worst |d| 5.17e-4 << atol 1e-2) and transpile-clean under the
a5 budget guard (UB 111,104 B / 43.8%, mutex 22/32, flags 0/0, src_is_transpose into
L0A = 0, Vec _nz = 0).  See kal/gdr_bwd_dv_local/NOTES.md.  Stage 4 adds only delivery wiring: wrapper
naming/signature, torch.empty output allocation, and the DESIGN section 8 tests.

DEVIATION 1 -- grad_o load axis order (Stage 4 bug fix in the transpiled output;
DESIGN is NOT affected, Stage 3 does NOT need revisiting).  The transpiled product
emitted the grad_o chunk load as

    pl.load(ub_go_bf, grad_o, [b, t_base, h, 0])            # no order=

On real a5 a 2-D tile indexed into a 4-D tensor defaults to the TRAILING two axes,
i.e. (H,V) -- so the load walked t*H+h contiguous rows instead of 32 rows of T at a
fixed h.  Measured on hardware with crafted inputs: for [B,T,H,V]=[1,128,4,64],
chunk n=0 sub-block 1 row 31 read flattened row (32*4 + h + 31), giving
t'=(159+h)//4, h'=(159+h)%4 -- reproducing the observed 3040 / 41 / 1040 / 2040
for h=0..3 exactly.  H=1 hides the bug entirely (both readings coincide), which is
why the Stage 3.5 simulator, and the H=1 probe here, both pass.
The fix is the explicit `order=[1, 3]` used by the already-delivered, hardware-verified
forward operator for exactly this public-layout read
(the forward operator's acceptance tests, `tests/ops/pypto_pro/gated_delta_rule/test_gated_delta_rule_fwd.py`).
This changes load ADDRESSING only -- algorithm, tile set, loop structure, sync and
mask are untouched.  DESIGN section 10 already specifies this edge as
"grad_o[b, n*C:n*C+C, h, 0:Vd]" with "t-row stride = H*V" (section 2), so the fix
restores the frozen design intent rather than departing from it.
The other loads/stores need no order=: k_f32 [B,H,N,C,K], q_t [B,H,N,K,C],
g_f32 [B,H,N,C] and dv_local [B,H,N,C,V] all address their trailing axes already.

A5 STRUCTURE NOTE (declared deviation 2, inherited from Stage 3.5 -- see kal/gdr_bwd_dv_local/NOTES.md
"A5 sub-block reality"): DESIGN section 4/section 6 describe the vector phase as
"sub-block-0 guarded".  On real a5 that idiom is not expressible -- a cube-side L0C->UB
move may not sit inside a vec `if GetSubBlockIdx()==0` guard.  The delivered structure
realises the same intent as DUAL/SPLITM banding: each of the two vector sub-blocks owns
HC = C/2 = 32 rows (`row_off = get_subblock_idx()*32`), the mask uses the absolute row
`row_off + r`, and the L0C->UB move sits on the producer (cube) side of the handoff.
This is arithmetically identical, uses BOTH vector sub-blocks, and is the configuration
that was simulator-verified.  Stage 3 does NOT need to be revisited.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info

logging.basicConfig(level=logging.INFO, format="%(message)s")

C = 64      # SPEC-frozen internal chunk size
KMAX = 128  # K upper bound (v1); tiles declared at KMAX, runtime valid k = Kd
VMAX = 128  # V upper bound (v1); tiles declared at VMAX, runtime valid v = Vd

# The runner's older PyPTO build leaves gaps in its DT_* pybind
# coverage: some constants (DT_BF16 among them) arrive as bare ints
# that VF-layer kwarg checks reject ("expected ir::DataType, but got
# int"), while others on the same build are real DataType objects.
# Rebuild the broken ones from the enum value AT MODULE SCOPE -- the
# VF parser inlines helper calls and rejects isinstance, so the fixups
# must be plain names by the time a decorated body mentions them.
# A healthy build passes through.
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


# =============================================================================
#   Vector functions (verbatim from kal/gdr_bwd_dv_local/staged_module1.py)
# =============================================================================
@pl.vector_function
def cast_go_vf(ub_go_bf, ub_go_f, valid_sb):
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    z = vf.full(0.0, active, dtype=_PL_FP32)
    for gor in pl.range(0, valid_sb, 1):
        for cc in pl.range(0, 128, 64):
            src = vf.load_align(ub_go_bf, (gor) * 128 + cc, dist=pl.LoadDist.UNPK_B16)
            dst = vf.astype(src, active, dtype=_PL_FP32, layout=pl.CastLayout.ZERO)
            vf.store_align(ub_go_f + (gor) * 128 + cc, dst, active)
    for gopr in pl.range(valid_sb, 32, 1):
        for cc_1 in pl.range(0, 128, 64):
            vf.store_align(ub_go_f + (gopr) * 128 + cc_1, z, active)


@pl.vector_function
def cumsum_decay_vf(ub_g, ub_decay):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    acc = vf.full(0.0, active, dtype=_PL_FP32)
    for csc in pl.range(0, 64, 1):
        gc = vf.load_align(ub_g, csc, dist=pl.LoadDist.BRC_B32)
        acc = vf.add(acc, gc, active)
        vf.store_align(ub_decay + csc, acc, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)


@pl.vector_function
def gate_mask_vf(ub_decay, ub_al, row_off):
    _tmp_maskreg_1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_INT32)
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    zero = vf.full(0.0, active, dtype=_PL_FP32)
    cols = vf.arange(0, dtype=_PL_INT32, index_order=pl.IndexOrder.INCREASE_ORDER)
    dvec = vf.load_align(ub_decay, 0, dtype=_PL_FP32)
    ndvec = vf.neg(dvec, _tmp_maskreg_0)
    for gmi in pl.range(0, 32, 1):
        _m0 = (row_off + gmi)
        di = vf.load_align(ub_decay, _m0, dist=pl.LoadDist.BRC_B32)
        ndi = vf.neg(di, _tmp_maskreg_0)
        gexp = vf.exp_sub(ndi, ndvec, active)
        m = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=_PL_INT32)
        m = vf.ge(cols, _m0, _tmp_maskreg_1)
        gexp = vf.select(gexp, zero, m)
        alrow = vf.load_align(ub_al, (gmi) * 64, dtype=_PL_FP32)
        alrow = vf.mul(alrow, gexp, active)
        vf.store_align(ub_al + (gmi) * 64, alrow, _tmp_maskreg_0)


# =============================================================================
#   THE single @pl.jit kernel (verbatim from kal/gdr_bwd_dv_local/staged_module1.py)
# =============================================================================
@pl.jit(auto_mutex=True)
def gdr_bwd_dv_local_module1_kernel(k_f32: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], q_t: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], grad_o: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16], g_f32: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], dv_local: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], B: pl.DT_INT32, T: pl.DT_INT32, H: pl.DT_INT32, Kd: pl.DT_INT32, Vd: pl.DT_INT32, N: pl.DT_INT32, Cc: pl.DT_INT32):
    g__l0a__v0 = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left), addrs=[0x0, 0x8000], mutex_ids=[0, 1])
    g__l0a__v1 = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left), addrs=[0x0, 0x8000], mutex_ids=[0, 1])
    g__l0b__v0 = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right), addrs=[0x0, 0x8000], mutex_ids=[2, 3])
    g__l0b__v1 = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right), addrs=[0x0, 0x8000], mutex_ids=[2, 3])
    # PERF (cube-08): depth-2 ping/pong for the two GM->L1 cube operands so the
    # tid+1 refill can be in flight while tid's MAC runs.  Depth = len(mutex_ids),
    # addrs is a base and slot i lands at base + i*slot_size (make_tile_group doc).
    # L1 map: k 0x00000-0x10000 (2x32 KB), q_t 0x10000-0x20000 (2x32 KB),
    #         alB 0x20000-0x24000 (16 KB), goB 0x24000-0x2C000 (32 KB) = 176 KB
    #         of 512 KB.  mutex ids 22/23 were free (22 of 32 were in use).
    g_l1_k = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat), addrs=0x0, mutex_ids=[4, 22])
    g_l1_qt = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat), addrs=0x10000, mutex_ids=[5, 23])
    g_l0c_al = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc), addrs=0x0, mutex_ids=[6, 7])
    g_ub_al = pl.make_tile_group(type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0x0, mutex_ids=[8, 9])
    g_ub_g = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x4000, mutex_ids=[10])
    g_ub_decay = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x4100, mutex_ids=[11])
    g_ub_go_bf = pl.make_tile_group(type=pl.TileType(shape=[32, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec), addrs=0x4200, mutex_ids=[12])
    g_ub_go_f = pl.make_tile_group(type=pl.TileType(shape=[32, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x6200, mutex_ids=[13])
    g_l1_alB = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat), addrs=0x20000, mutex_ids=[14])
    g_l1_goB = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat), addrs=0x24000, mutex_ids=[15])
    g_l0c_dv = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1]), addrs=0x8000, mutex_ids=[16, 17])
    g_ub_dv = pl.make_tile_group(type=pl.TileType(shape=[32, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addrs=0xa200, mutex_ids=[18, 19])
    g_ub_al_nz = pl.make_tile_group(type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ), addrs=0x12200, mutex_ids=[20])
    g_ub_al_nz_c0 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x12200, mutex_ids=[20])
    g_ub_al_nz_c1 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x12600, mutex_ids=[20])
    g_ub_al_nz_c2 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x12a00, mutex_ids=[20])
    g_ub_al_nz_c3 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x12e00, mutex_ids=[20])
    g_ub_al_nz_c4 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x13200, mutex_ids=[20])
    g_ub_al_nz_c5 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x13600, mutex_ids=[20])
    g_ub_al_nz_c6 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x13a00, mutex_ids=[20])
    g_ub_al_nz_c7 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x13e00, mutex_ids=[20])
    g_ub_go_f_nz = pl.make_tile_group(type=pl.TileType(shape=[32, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ), addrs=0x15200, mutex_ids=[21])
    g_ub_go_f_nz_c0 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x15200, mutex_ids=[21])
    g_ub_go_f_nz_c1 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x15600, mutex_ids=[21])
    g_ub_go_f_nz_c2 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x15a00, mutex_ids=[21])
    g_ub_go_f_nz_c3 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x15e00, mutex_ids=[21])
    g_ub_go_f_nz_c4 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x16200, mutex_ids=[21])
    g_ub_go_f_nz_c5 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x16600, mutex_ids=[21])
    g_ub_go_f_nz_c6 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x16a00, mutex_ids=[21])
    g_ub_go_f_nz_c7 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x16e00, mutex_ids=[21])
    g_ub_go_f_nz_c8 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x17200, mutex_ids=[21])
    g_ub_go_f_nz_c9 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x17600, mutex_ids=[21])
    g_ub_go_f_nz_c10 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x17a00, mutex_ids=[21])
    g_ub_go_f_nz_c11 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x17e00, mutex_ids=[21])
    g_ub_go_f_nz_c12 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x18200, mutex_ids=[21])
    g_ub_go_f_nz_c13 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x18600, mutex_ids=[21])
    g_ub_go_f_nz_c14 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x18a00, mutex_ids=[21])
    g_ub_go_f_nz_c15 = pl.make_tile_group(type=pl.TileType(shape=[32, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ, valid_shape=[-1, -1]), addrs=0x18e00, mutex_ids=[21])
    with pl.section_cube():
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        t3_l1_alB = g_l1_alB.next()
        t4_l1_goB = g_l1_goB.next()
        # PERF (cube-08 step 3, after the barrier fix): software-pipeline the two
        # GM->L1 operand loads one tid ahead over a depth-2 group, addressed with the
        # documented runtime index form g[i] (a .next() handle cannot be carried across
        # loop iterations / traced branches).  The refill for tid+1 is ISSUED right
        # after tid's A_l matmul, so its MTE2 runs under that MAC, the ev1 wait, the
        # L0C->UB drain, bar_all, the ev3 wait and the dv matmul.  The last iteration
        # re-reads the final tid (clamped) rather than branching -- one redundant 64 KB
        # read per core in ~10, kept in exchange for not putting a traced `if` around
        # an indexed group access.
        _nt = ((B * H) * N)
        _per = (((_nt + pl.get_block_num()) - 1) // pl.get_block_num())
        _lo = (_per * (pl.get_block_idx() // pl.get_subblock_num()))
        _hi = (pl.min((_lo + _per), _nt))
        pl.set_validshape(g_l1_k[0], [64, Kd])
        pl.set_validshape(g_l1_k[1], [64, Kd])
        pl.set_validshape(g_l1_qt[0], [Kd, 64])
        pl.set_validshape(g_l1_qt[1], [Kd, 64])
        pl.load(g_l1_k[0], k_f32, [(_lo // (H * N)), ((_lo % (H * N)) // N), ((_lo % (H * N)) % N), 0, 0])
        pl.load(g_l1_qt[0], q_t, [(_lo // (H * N)), ((_lo % (H * N)) // N), ((_lo % (H * N)) % N), 0, 0])
        for tid in pl.range(_lo, _hi, 1):
            _cur = ((tid - _lo) % 2)
            _nxt = (((tid - _lo) + 1) % 2)
            _tn = (pl.min((tid + 1), (_hi - 1)))
            t1_l1_k = g_l1_k[_cur]
            t2_l1_qt = g_l1_qt[_cur]
            t5_l0c_al = g_l0c_al.next()
            t6__l0a__v1 = g__l0a__v1.next()
            t7__l0a = g__l0a__v0.next()
            t8__l0b__v1 = g__l0b__v1.next()
            t9__l0b = g__l0b__v0.next()
            pl.move(t7__l0a, t1_l1_k, [0, 0])
            pl.move(t9__l0b, t2_l1_qt, [0, 0])
            pl.set_validshape(t7__l0a, [64, Kd])
            pl.set_validshape(t9__l0b, [Kd, 64])
            pl.set_validshape(t5_l0c_al, [64, 64])
            pl.matmul(t5_l0c_al, t7__l0a, t9__l0b)
            # step-tail refill for tid+1 into the other L1 slot (independent mutex id)
            pl.load(g_l1_k[_nxt], k_f32, [(_tn // (H * N)), ((_tn % (H * N)) // N), ((_tn % (H * N)) % N), 0, 0])
            pl.load(g_l1_qt[_nxt], q_t, [(_tn // (H * N)), ((_tn % (H * N)) // N), ((_tn % (H * N)) % N), 0, 0])
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            t10_ub_al = g_ub_al.next()
            pl.set_validshape(t5_l0c_al, [64, 64])
            pl.move(t10_ub_al, t5_l0c_al, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.system.bar_all()
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=3, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            t11_l0c_dv = g_l0c_dv.next()
            t12__l0a__v0 = g__l0a__v0.next()
            t13__l0a = g__l0a__v1.next()
            t14__l0b__v0 = g__l0b__v0.next()
            t15__l0b = g__l0b__v1.next()
            pl.move(t13__l0a, t3_l1_alB, [0, 0])
            pl.move(t15__l0b, t4_l1_goB, [0, 0])
            pl.set_validshape(t13__l0a, [64, 64])
            pl.set_validshape(t15__l0b, [64, 128])
            pl.set_validshape(t11_l0c_dv, [64, Vd])
            pl.matmul(t11_l0c_dv, t13__l0a, t15__l0b)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            t16_ub_dv = g_ub_dv.next()
            pl.set_validshape(t11_l0c_dv, [64, Vd])
            pl.move(t16_ub_dv, t11_l0c_dv, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=4, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
    with pl.section_vector():
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        t17_l1_k = g_l1_k.next()
        t18_l1_qt = g_l1_qt.next()
        t19_ub_g = g_ub_g.next()
        t20_ub_decay = g_ub_decay.next()
        t21_ub_go_bf = g_ub_go_bf.next()
        t22_ub_go_f = g_ub_go_f.next()
        t23_l1_alB = g_l1_alB.next()
        t24_l1_goB = g_l1_goB.next()
        for tid in pl.range((((((B * H) * N) + pl.get_block_num() - 1) // pl.get_block_num()) * (pl.get_block_idx() // pl.get_subblock_num())), (pl.min(((((((B * H) * N) + pl.get_block_num() - 1) // pl.get_block_num()) * (pl.get_block_idx() // pl.get_subblock_num())) + ((((B * H) * N) + pl.get_block_num() - 1) // pl.get_block_num())), ((B * H) * N))), 1):
            # PERF (cube-08 "precondition zero" -- barrier placement before buffer
            # depth): the ev0 handshake only guards `ub_al`, which the cube fills from
            # L0C.  It used to sit HERE, at the top of the body, in front of the g_f32
            # load, the 64-step serial cumsum, the grad_o load and the bf16->fp32 cast
            # -- none of which read ub_al.  The vector core therefore idled on ev0 while
            # the cube ran the A_l matmul, and the cube then idled on ev3 while the
            # vector ran that whole independent block: one strictly serial chain
            # A_l matmul -> all vector work -> dv matmul.  The wait is now pushed down
            # to just before `gate_mask_vf`, its first real consumer, so the g/cumsum
            # and grad_o/cast work overlaps the cube's A_l phase.  Ordering of the
            # ev1 / ev2 / ev3 / ev4 / ev5 handshakes is unchanged.
            pl.set_validshape(t19_ub_g, [1, 64])
            pl.load(t19_ub_g, g_f32, [(tid // (H * N)), ((tid % (H * N)) // N), ((tid % (H * N)) % N), 0])
            cumsum_decay_vf(t19_ub_g, t20_ub_decay)
            _k1 = (pl.get_subblock_idx() * 32)
            _k0 = (((tid % (H * N)) % N) * 64)
            _k2 = (_k0 + _k1)
            _k3 = (T - _k2)
            _k4 = (pl.max(_k3, 0))
            _k5 = (pl.min(_k4, 32))
            if _k5 > 0:
                pl.set_validshape(t21_ub_go_bf, [_k5, Vd])
                # order=[1,3]: grad_o is the PUBLIC [B,T,H,V] layout, so the 2-D tile's
                # two axes are T (axis 1) and V (axis 3) -- axis 2 (H) is held fixed.
                # Without an explicit order= the trailing two axes (H,V) are taken
                # instead, which silently reads t*H+h contiguous rows.  See the
                # "grad_o load axis order" deviation note in the module docstring.
                pl.load(t21_ub_go_bf, grad_o, [(tid // (H * N)), _k2, ((tid % (H * N)) // N), 0], order=[1, 3])
            cast_go_vf(t21_ub_go_bf, t22_ub_go_f, _k5)
            # PERF (cube-08 precondition zero, second pass): ev0 gates ONLY ub_al, i.e.
            # gate_mask_vf and the l1_alB path (16 KB).  The grad_o NZ re-layout and its
            # 16 inserts into l1_goB (32 KB) need only ev2, so they now run BEFORE the
            # ev0 wait.  The cube blocks on ev3 for the whole of l1_alB + l1_goB; moving
            # two thirds of that production off the post-ev0 segment shortens that block.
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            t35_ub_go_f_nz = g_ub_go_f_nz.next()
            pl.move(t35_ub_go_f_nz, t22_ub_go_f)
            t36_ub_go_f_nz_c0 = g_ub_go_f_nz_c0.next()
            pl.set_validshape(t36_ub_go_f_nz_c0, [32, 8])
            pl.insert(t24_l1_goB, t36_ub_go_f_nz_c0, [_k1, 0])
            t37_ub_go_f_nz_c1 = g_ub_go_f_nz_c1.next()
            pl.set_validshape(t37_ub_go_f_nz_c1, [32, 8])
            pl.insert(t24_l1_goB, t37_ub_go_f_nz_c1, [_k1, (0) + 8])
            t38_ub_go_f_nz_c2 = g_ub_go_f_nz_c2.next()
            pl.set_validshape(t38_ub_go_f_nz_c2, [32, 8])
            pl.insert(t24_l1_goB, t38_ub_go_f_nz_c2, [_k1, (0) + 16])
            t39_ub_go_f_nz_c3 = g_ub_go_f_nz_c3.next()
            pl.set_validshape(t39_ub_go_f_nz_c3, [32, 8])
            pl.insert(t24_l1_goB, t39_ub_go_f_nz_c3, [_k1, (0) + 24])
            t40_ub_go_f_nz_c4 = g_ub_go_f_nz_c4.next()
            pl.set_validshape(t40_ub_go_f_nz_c4, [32, 8])
            pl.insert(t24_l1_goB, t40_ub_go_f_nz_c4, [_k1, (0) + 32])
            t41_ub_go_f_nz_c5 = g_ub_go_f_nz_c5.next()
            pl.set_validshape(t41_ub_go_f_nz_c5, [32, 8])
            pl.insert(t24_l1_goB, t41_ub_go_f_nz_c5, [_k1, (0) + 40])
            t42_ub_go_f_nz_c6 = g_ub_go_f_nz_c6.next()
            pl.set_validshape(t42_ub_go_f_nz_c6, [32, 8])
            pl.insert(t24_l1_goB, t42_ub_go_f_nz_c6, [_k1, (0) + 48])
            t43_ub_go_f_nz_c7 = g_ub_go_f_nz_c7.next()
            pl.set_validshape(t43_ub_go_f_nz_c7, [32, 8])
            pl.insert(t24_l1_goB, t43_ub_go_f_nz_c7, [_k1, (0) + 56])
            t44_ub_go_f_nz_c8 = g_ub_go_f_nz_c8.next()
            pl.set_validshape(t44_ub_go_f_nz_c8, [32, 8])
            pl.insert(t24_l1_goB, t44_ub_go_f_nz_c8, [_k1, (0) + 64])
            t45_ub_go_f_nz_c9 = g_ub_go_f_nz_c9.next()
            pl.set_validshape(t45_ub_go_f_nz_c9, [32, 8])
            pl.insert(t24_l1_goB, t45_ub_go_f_nz_c9, [_k1, (0) + 72])
            t46_ub_go_f_nz_c10 = g_ub_go_f_nz_c10.next()
            pl.set_validshape(t46_ub_go_f_nz_c10, [32, 8])
            pl.insert(t24_l1_goB, t46_ub_go_f_nz_c10, [_k1, (0) + 80])
            t47_ub_go_f_nz_c11 = g_ub_go_f_nz_c11.next()
            pl.set_validshape(t47_ub_go_f_nz_c11, [32, 8])
            pl.insert(t24_l1_goB, t47_ub_go_f_nz_c11, [_k1, (0) + 88])
            t48_ub_go_f_nz_c12 = g_ub_go_f_nz_c12.next()
            pl.set_validshape(t48_ub_go_f_nz_c12, [32, 8])
            pl.insert(t24_l1_goB, t48_ub_go_f_nz_c12, [_k1, (0) + 96])
            t49_ub_go_f_nz_c13 = g_ub_go_f_nz_c13.next()
            pl.set_validshape(t49_ub_go_f_nz_c13, [32, 8])
            pl.insert(t24_l1_goB, t49_ub_go_f_nz_c13, [_k1, (0) + 104])
            t50_ub_go_f_nz_c14 = g_ub_go_f_nz_c14.next()
            pl.set_validshape(t50_ub_go_f_nz_c14, [32, 8])
            pl.insert(t24_l1_goB, t50_ub_go_f_nz_c14, [_k1, (0) + 112])
            t51_ub_go_f_nz_c15 = g_ub_go_f_nz_c15.next()
            pl.set_validshape(t51_ub_go_f_nz_c15, [32, 8])
            pl.insert(t24_l1_goB, t51_ub_go_f_nz_c15, [_k1, (0) + 120])
            # ev0-gated segment, moved here from the top of the body: everything below
            # depends on ub_al, everything above does not.
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=0, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            t25_ub_al = g_ub_al.next()
            gate_mask_vf(t20_ub_decay, t25_ub_al, _k1)
            pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            t26_ub_al_nz = g_ub_al_nz.next()
            pl.move(t26_ub_al_nz, t25_ub_al)
            t27_ub_al_nz_c0 = g_ub_al_nz_c0.next()
            pl.set_validshape(t27_ub_al_nz_c0, [32, 8])
            pl.insert(t23_l1_alB, t27_ub_al_nz_c0, [_k1, 0])
            t28_ub_al_nz_c1 = g_ub_al_nz_c1.next()
            pl.set_validshape(t28_ub_al_nz_c1, [32, 8])
            pl.insert(t23_l1_alB, t28_ub_al_nz_c1, [_k1, (0) + 8])
            t29_ub_al_nz_c2 = g_ub_al_nz_c2.next()
            pl.set_validshape(t29_ub_al_nz_c2, [32, 8])
            pl.insert(t23_l1_alB, t29_ub_al_nz_c2, [_k1, (0) + 16])
            t30_ub_al_nz_c3 = g_ub_al_nz_c3.next()
            pl.set_validshape(t30_ub_al_nz_c3, [32, 8])
            pl.insert(t23_l1_alB, t30_ub_al_nz_c3, [_k1, (0) + 24])
            t31_ub_al_nz_c4 = g_ub_al_nz_c4.next()
            pl.set_validshape(t31_ub_al_nz_c4, [32, 8])
            pl.insert(t23_l1_alB, t31_ub_al_nz_c4, [_k1, (0) + 32])
            t32_ub_al_nz_c5 = g_ub_al_nz_c5.next()
            pl.set_validshape(t32_ub_al_nz_c5, [32, 8])
            pl.insert(t23_l1_alB, t32_ub_al_nz_c5, [_k1, (0) + 40])
            t33_ub_al_nz_c6 = g_ub_al_nz_c6.next()
            pl.set_validshape(t33_ub_al_nz_c6, [32, 8])
            pl.insert(t23_l1_alB, t33_ub_al_nz_c6, [_k1, (0) + 48])
            t34_ub_al_nz_c7 = g_ub_al_nz_c7.next()
            pl.set_validshape(t34_ub_al_nz_c7, [32, 8])
            pl.insert(t23_l1_alB, t34_ub_al_nz_c7, [_k1, (0) + 56])
            pl.system.bar_all()
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=3, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=4, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
            t52_ub_dv = g_ub_dv.next()
            pl.set_validshape(t52_ub_dv, [32, Vd])
            pl.store(dv_local, t52_ub_dv, [(tid // (H * N)), ((tid % (H * N)) // N), ((tid % (H * N)) % N), _k1, 0])
            pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)


# =============================================================================
#   Wrapper -- HARD contract name gdr_bwd_dv_local_wrapper
# =============================================================================
def gdr_bwd_dv_local_wrapper(grad_o, k_f32=None, q_t=None, g_f32=None):
    """Module 1 entry (= the FULL operator).  Returns dv_local [B,H,N,C,V] fp32.

    Host side does only what wrapper-boundary.md allows: pure-Python shape
    arithmetic, one torch.empty output allocation, and ONE kernel launch.
    No host tensor arithmetic, cast, transpose, slice, pad or cumsum.
    """
    B, T, H, V = grad_o.shape
    _, _, N, _, K = k_f32.shape          # k_f32 is [B, H, N, C, K]

    dv_local = torch.empty((B, H, N, C, V), dtype=torch.float32, device=grad_o.device)

    max_blocks = get_platform_info().core_num or 1
    num_cores = max(1, min(B * H * N, max_blocks))

    gdr_bwd_dv_local_module1_kernel[None, num_cores](
        k_f32, q_t, grad_o, g_f32, dv_local, B, T, H, K, V, N, C)
    torch.npu.synchronize()
    return dv_local

