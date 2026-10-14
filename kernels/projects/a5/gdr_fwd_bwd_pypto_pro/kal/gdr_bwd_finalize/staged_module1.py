import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

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
def cast_vf(t_in, t_out, Wd):
    _zero_fp32 = vf.full(0, dtype=_PL_FP32)
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_BF16)
    for cast_r in pl.range(0, 64, 1):
        for cast_h in pl.range(0, ((Wd + 64 - 1) // 64), 1):
            _m0 = (cast_h * 64)
            fr = vf.load_align(t_in, (cast_r) * 128 + _m0, dtype=_PL_FP32)
            br = vf.astype(fr, _tmp_maskreg_0, dtype=_PL_BF16, layout=pl.CastLayout.ZERO, round_mode=pl.VFRoundMode.CAST_ROUND)
            _t1 = vf.xor(br, _zero_fp32, _tmp_maskreg_0, dtype=_PL_FP32)
            vf.store_align(t_out + (cast_r) * 128 + _m0, _t1, _tmp_maskreg_0, dist=pl.StoreDist.PACK)

@pl.vector_function
def dbeta_calc_vf(t_dbw, t_sig, t_wide):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    one = vf.full(1.0, active, dtype=_PL_FP32)
    for db_r in pl.range(0, 64, 1):
        sgr = vf.load_align(t_sig, db_r, dist=pl.LoadDist.BRC_B32)
        dbr = vf.load_align(t_dbw, db_r, dist=pl.LoadDist.BRC_B32)
        omr = vf.sub(one, sgr, active)
        sdr = vf.mul(sgr, omr, active)
        res = vf.mul(dbr, sdr, active)
        vf.store_align(t_wide + (db_r) * 128, res, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)

@pl.vector_function
def dg_reduce_vf(t_dgp, t_dg2, t_wide):
    _tmp_maskreg_0 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    active = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    dgp = vf.load_align(t_dgp, 0, dtype=_PL_FP32)
    dg2r = vf.load_align(t_dg2, 0, dtype=_PL_FP32)
    dgsum = vf.add(dgp, dg2r, active)
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    full = vf.update_mask(pl.const(64, _PL_UINT32), dtype=_PL_FP32)
    total = vf.reduce_sum(dgsum, full)
    mpre = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=_PL_FP32)
    for dg_r in pl.range(0, 64, 1):
        mpre = vf.update_mask(pl.max(0, dg_r), dtype=_PL_FP32)
        pfx = vf.reduce_sum(dgsum, mpre)
        sr = vf.sub(total, pfx, active)
        vf.store_align(t_wide + (dg_r) * 128, sr, _tmp_maskreg_0, dist=pl.StoreDist.FIRST_ELEMENT)


@pl.jit(auto_mutex=True)
def gdr_bwd_finalize_kernel(dq_p: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], dk_in: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], dv_in: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], dg_p: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], dg2: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], dbeta_wy: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], sig: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], dq: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16], dk: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16], dv: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16], dg: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], dbeta: pl.Tensor[[1, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], B: pl.DT_INT32, T: pl.DT_INT32, H: pl.DT_INT32, Kd: pl.DT_INT32, Vd: pl.DT_INT32, N: pl.DT_INT32, Cc: pl.DT_INT32):
    g_in_big = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0x0, mutex_ids=[0])
    g_out_big = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec), addrs=0x8000, mutex_ids=[1])
    g_t_dgp = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0xc000, mutex_ids=[2])
    g_t_dg2 = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0xc100, mutex_ids=[3])
    g_t_dbw = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0xc200, mutex_ids=[4])
    g_t_sig = pl.make_tile_group(type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0xc300, mutex_ids=[5])
    g_col_src = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0xc400, mutex_ids=[6])
    with pl.section_vector():
        t1_in_big = g_in_big.next()
        t2_out_big = g_out_big.next()
        t3_t_dgp = g_t_dgp.next()
        t4_t_dg2 = g_t_dg2.next()
        t5_t_dbw = g_t_dbw.next()
        t6_t_sig = g_t_sig.next()
        t7_col_src = g_col_src.next()
        _k0 = ((B * H) * N)
        _k7 = (_k0 + pl.get_block_num() - 1)
        _k9 = (_k7 // pl.get_block_num())
        _k10 = (_k9 * pl.get_block_idx())
        for tid in pl.range(_k10, (pl.min((_k10 + _k9), _k0)), 1):
            pl.set_validshape(t3_t_dgp, [1, 64])
            _k1 = (tid % (H * N))
            _k2 = (tid // (H * N))
            _k3 = (_k1 % N)
            _k4 = (_k1 // N)
            pl.load(t3_t_dgp, dg_p, [_k2, _k4, _k3, 0])
            pl.set_validshape(t4_t_dg2, [1, 64])
            pl.load(t4_t_dg2, dg2, [_k2, _k4, _k3, 0])
            dg_reduce_vf(t3_t_dgp, t4_t_dg2, t7_col_src)
            _k5 = (_k3 * 64)
            _k6 = (T - _k5)
            _k8 = (pl.min(_k6, 64))
            pl.set_validshape(t7_col_src, [_k8, 1])
            pl.store(dg, t7_col_src, [_k2, _k5, _k4])
            pl.set_validshape(t5_t_dbw, [1, 64])
            pl.load(t5_t_dbw, dbeta_wy, [_k2, _k4, _k3, 0])
            pl.set_validshape(t6_t_sig, [1, 64])
            pl.load(t6_t_sig, sig, [_k2, _k4, _k3, 0])
            dbeta_calc_vf(t5_t_dbw, t6_t_sig, t7_col_src)
            pl.set_validshape(t7_col_src, [_k8, 1])
            pl.store(dbeta, t7_col_src, [_k2, _k5, _k4])
            pl.set_validshape(t1_in_big, [64, Kd])
            pl.load(t1_in_big, dq_p, [_k2, _k4, _k3, 0, 0])
            cast_vf(t1_in_big, t2_out_big, Kd)
            pl.set_validshape(t2_out_big, [_k8, Kd])
            pl.store(dq, t2_out_big, [_k2, _k5, _k4, 0])
            pl.set_validshape(t1_in_big, [64, Kd])
            pl.load(t1_in_big, dk_in, [_k2, _k4, _k3, 0, 0])
            cast_vf(t1_in_big, t2_out_big, Kd)
            pl.set_validshape(t2_out_big, [_k8, Kd])
            pl.store(dk, t2_out_big, [_k2, _k5, _k4, 0])
            pl.set_validshape(t1_in_big, [64, Vd])
            pl.load(t1_in_big, dv_in, [_k2, _k4, _k3, 0, 0])
            cast_vf(t1_in_big, t2_out_big, Vd)
            pl.set_validshape(t2_out_big, [_k8, Vd])
            pl.store(dv, t2_out_big, [_k2, _k5, _k4, 0])
