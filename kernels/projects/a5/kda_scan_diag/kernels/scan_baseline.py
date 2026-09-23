"""KDA kernel port; ABI and source provenance are owned by contract.json.

The arithmetic and work partition come from the reviewed source. Migration
uses a shared L0 operand pair with cube-phase barriers to fit the eight-flag
event budget; see the explicit scheduling amendment in contract.json.
"""

import math

from ascriptor.a5 import *

L = 64

D = 128

HALF_L = L // 2

HALF_D = D // 2

BF16_C0 = 16

QG_SCALE = 1.0 / 11.313708498984761

DAQK_SCALE = 1.0 / (D ** 0.5)

LN2 = math.log(2.0)

@vf()
def cast_dht_to_f32_vf(dht_h_ub: Tensor, dstate_f_ub: Tensor, row_begin_d: Var):
    reg_f32 = Reg(DT.float)
    dst_base = Var(row_begin_d * D)
    n_loops = Var(HALF_D * D // 64)  # M10: the static VF loop bound must be integral.
    for i in range(n_loops):
        reg_f32 <<= dht_h_ub[i * 64]
        dstate_f_ub[dst_base + i * 64] <<= reg_f32
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def snapshot_and_cast_state_vf(dstate_f_ub: Tensor, state_h_ub: Tensor, state_h_nz_ub: Tensor, row_begin_d: Var):
    row_lo = Reg(DT.float)
    row_hi = Reg(DT.float)
    lo_bf16 = Reg(DT.bfloat16)
    hi_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)
    for r in range(HALF_D):
        row_lo <<= dstate_f_ub[row_begin_d + r:row_begin_d + r + 1, 0:64]
        row_hi <<= dstate_f_ub[row_begin_d + r:row_begin_d + r + 1, 64:D]
        lo_bf16 <<= row_lo.astype(DT.bfloat16)
        hi_bf16 <<= row_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(state_h_ub[r:r + 1, 0:D], row_bf16)
        reg_to_ub(state_h_nz_ub[r * BF16_C0], row_bf16, HALF_D)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def add_dv_and_cast_bf16_vf(dv_delta_ub: Tensor, dv_sum_ub: Tensor, dv_bf16_ub: Tensor, dv_bf16_nz_ub: Tensor):
    delta_lo = Reg(DT.float)
    delta_hi = Reg(DT.float)
    sum_lo = Reg(DT.float)
    sum_hi = Reg(DT.float)
    lo_bf16 = Reg(DT.bfloat16)
    hi_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)
    for r in range(HALF_L):
        delta_lo <<= dv_delta_ub[r:r + 1, 0:64]
        delta_hi <<= dv_delta_ub[r:r + 1, 64:D]
        sum_lo <<= dv_sum_ub[r:r + 1, 0:64]
        sum_hi <<= dv_sum_ub[r:r + 1, 64:D]
        sum_lo <<= sum_lo + delta_lo
        sum_hi <<= sum_hi + delta_hi
        lo_bf16 <<= sum_lo.astype(DT.bfloat16)
        hi_bf16 <<= sum_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(dv_bf16_ub[r:r + 1, 0:D], row_bf16)
        reg_to_ub(dv_bf16_nz_ub[r * BF16_C0], row_bf16, HALF_L)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def exp2_glast_vf(g_last_half_ub: Tensor, exp_g_last_half_ub: Tensor):
    # Per-chunk decay 2^g_last = exp(g_last * ln2): g_last (= g_cumsum at the chunk
    # boundary) already carries the 1/ln2 factor. Replaces the old host
    # torch.exp(g_last*ln2). One HALF_D-lane shot per sub-block.
    r = Reg(DT.float)
    r <<= g_last_half_ub[0:1, 0:HALF_D]   # bf16 -> fp32 upcast
    r <<= r * LN2
    r <<= r.exp()
    exp_g_last_half_ub[0:1, 0:HALF_D] <<= r
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def update_dstate_vf(
    seed_ub: Tensor,
    corr_ub: Tensor,
    dstate_f_ub: Tensor,
    state_h_ub: Tensor,
    state_h_nz_ub: Tensor,
    exp_g_last_half_ub: Tensor,
    row_begin_d: Var,
):
    seed_lo = Reg(DT.float)
    seed_hi = Reg(DT.float)
    corr_lo = Reg(DT.float)
    corr_hi = Reg(DT.float)
    state_lo = Reg(DT.float)
    state_hi = Reg(DT.float)
    scale = Reg(DT.float)
    lo_bf16 = Reg(DT.bfloat16)
    hi_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)
    for r in range(HALF_D):
        scale <<= exp_g_last_half_ub[0:1, r:r + 1].single()
        seed_lo <<= seed_ub[r:r + 1, 0:64]
        seed_hi <<= seed_ub[r:r + 1, 64:D]
        seed_lo <<= seed_lo * QG_SCALE   # 1/sqrt(D): was host qg pre-scale
        seed_hi <<= seed_hi * QG_SCALE
        corr_lo <<= corr_ub[r:r + 1, 0:64]
        corr_hi <<= corr_ub[r:r + 1, 64:D]
        state_lo <<= dstate_f_ub[row_begin_d + r:row_begin_d + r + 1, 0:64]
        state_hi <<= dstate_f_ub[row_begin_d + r:row_begin_d + r + 1, 64:D]
        state_lo <<= state_lo * scale
        state_hi <<= state_hi * scale
        seed_lo <<= seed_lo + state_lo
        seed_hi <<= seed_hi + state_hi
        seed_lo <<= seed_lo - corr_lo
        seed_hi <<= seed_hi - corr_hi
        dstate_f_ub[row_begin_d + r:row_begin_d + r + 1, 0:64] <<= seed_lo
        dstate_f_ub[row_begin_d + r:row_begin_d + r + 1, 64:D] <<= seed_hi
        lo_bf16 <<= seed_lo.astype(DT.bfloat16)
        hi_bf16 <<= seed_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(state_h_ub[r:r + 1, 0:D], row_bf16)
        reg_to_ub(state_h_nz_ub[r * BF16_C0], row_bf16, HALF_D)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def update_dstate_final_vf(
    seed_ub: Tensor,
    corr_ub: Tensor,
    dstate_f_ub: Tensor,
    state_h_ub: Tensor,
    exp_g_last_half_ub: Tensor,
    row_begin_d: Var,
):
    seed_lo = Reg(DT.float)
    seed_hi = Reg(DT.float)
    corr_lo = Reg(DT.float)
    corr_hi = Reg(DT.float)
    state_lo = Reg(DT.float)
    state_hi = Reg(DT.float)
    scale = Reg(DT.float)
    lo_bf16 = Reg(DT.bfloat16)
    hi_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)
    for r in range(HALF_D):
        scale <<= exp_g_last_half_ub[0:1, r:r + 1].single()
        seed_lo <<= seed_ub[r:r + 1, 0:64]
        seed_hi <<= seed_ub[r:r + 1, 64:D]
        seed_lo <<= seed_lo * QG_SCALE   # 1/sqrt(D): was host qg pre-scale
        seed_hi <<= seed_hi * QG_SCALE
        corr_lo <<= corr_ub[r:r + 1, 0:64]
        corr_hi <<= corr_ub[r:r + 1, 64:D]
        state_lo <<= dstate_f_ub[row_begin_d + r:row_begin_d + r + 1, 0:64]
        state_hi <<= dstate_f_ub[row_begin_d + r:row_begin_d + r + 1, 64:D]
        state_lo <<= state_lo * scale
        state_hi <<= state_hi * scale
        seed_lo <<= seed_lo + state_lo
        seed_hi <<= seed_hi + state_hi
        seed_lo <<= seed_lo - corr_lo
        seed_hi <<= seed_hi - corr_hi
        lo_bf16 <<= seed_lo.astype(DT.bfloat16)
        hi_bf16 <<= seed_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(state_h_ub[r:r + 1, 0:D], row_bf16)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@func()
def _phase_matmul(dst: Tensor, lhs: Tensor, rhs: Tensor, l0a: Tensor, l0b: Tensor, m: Var, n: Var, k: Var):
    # Shared operands trade overlap for an explicit finite event footprint.
    bar_all()
    l1_to_l0(l0a, lhs)
    l1_to_l0(l0b, rhs)
    bar_all()
    mmad(dst, l0a, l0b, M=m, N=n, K=k, is_init=True)
    bar_all()


@kernel()
def a5k03_scan_baseline_kernel(kg: GM[bf16, ('B', 'T', 'HV', 128)], qg: GM[bf16, ('B', 'T', 'HV', 128)], w: GM[bf16, ('B', 'T', 'HV', 128)], g_last: GM[bf16, ('B', 'C', 'HV', 128)], grad_out: GM[bf16, ('B', 'T', 'HV', 128)], Aqk: GM[bf16, ('B', 'T', 'HV', 64)], v_new: GM[bf16, ('B', 'T', 'HV', 128)], dht: GM[bf16, ('B', 'HV', 64, 256)], dAqk: GM[bf16, ('B', 'T', 'HV', 64)], dh: GM[bf16, ('B', 'C', 'HV', 128, 128)], dv: GM[bf16, ('B', 'T', 'HV', 128)], dh0: GM[bf16, ('B', 'HV', 128, 128)], B: i32, HV: i32, C: i32):
    phase_l0a = Tensor(DT.bfloat16, [D, D], Position.L0A)
    phase_l0b = Tensor(DT.bfloat16, [D, D], Position.L0B)
    state_bridge = VcMutex(
        0,
        depth=1,
        src_start_pipe=Pipe.MTE3,
        src_end_pipe=Pipe.MTE3,
        dst_start_pipe=Pipe.MTE1,
        dst_end_pipe=Pipe.MTE1,
    )
    dv_bridge = VcMutex(
        1,
        depth=1,
        src_start_pipe=Pipe.MTE3,
        src_end_pipe=Pipe.MTE3,
        dst_start_pipe=Pipe.MTE1,
        dst_end_pipe=Pipe.MTE1,
    )
    dvdelta_mutex = CvMutex(2, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    seed_mutex = CvMutex(3, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    corr_mutex = CvMutex(4, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    dv0_mutex = CvMutex(5, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

    l1_kg = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_qg = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_w = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_do = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_Aqk = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_vnew = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_dv = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_dstate = Tensor(DT.bfloat16, [D, D], Position.L1)

    l0c_dvdelta = Tensor(DT.float, [L, D], Position.L0C)
    l0c_seed = DBuff(DT.float, [D, D], Position.L0C)
    l0c_corr = Tensor(DT.float, [D, D], Position.L0C)
    l0c_dv0 = Tensor(DT.float, [L, D], Position.L0C)
    # dAqk shares the dvdelta accumulator: its matmul runs only after the
    # dvdelta fixpipe of the current chunk has drained.
    l0c_daqk = l0c_dvdelta

    dstate_f_ub = Tensor(DT.float, [D, D], Position.UB)
    state_h_ub = Tensor(DT.bfloat16, [HALF_D, D], Position.UB)
    state_h_nz_ub = Tensor(DT.bfloat16, [HALF_D, D], Position.UB)
    dv_delta_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    dht_h_ub = dv_delta_ub.reinterpret(DT.bfloat16)
    dv_sum_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    dv_bf16_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    dv_bf16_nz_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    seed_ub = DBuff(DT.float, [HALF_D, D], Position.UB)
    corr_ub = Tensor(DT.float, [HALF_D, D], Position.UB)
    g_last_half_ub = Tensor(DT.bfloat16, [1, HALF_D], Position.UB)
    exp_g_last_half_ub = Tensor(DT.float, [1, HALF_D], Position.UB)

    bhv_count = B * HV
    bhv_per_core = CeilDiv(bhv_count, GetCubeNum())
    bhv_begin = Var(bhv_per_core * GetCubeIdx())
    bhv_end = Min(bhv_begin + bhv_per_core, bhv_count)
    row_begin_l = Var(GetSubBlockIdx() * HALF_L)
    row_end_l = Var(row_begin_l + HALF_L)
    row_begin_d = Var(GetSubBlockIdx() * HALF_D)
    row_end_d = Var(row_begin_d + HALF_D)
    # dht GM rides as [B, HV, D/2, 2*D]: each sub-block owns HALF_L bf16 rows.
    dht_row_begin = Var(GetSubBlockIdx() * HALF_L)
    dht_row_end = Var(dht_row_begin + HALF_L)

    with auto_sync():
        for bhv in range(bhv_begin, bhv_end):
            b_idx = Var(bhv / HV)
            hv_idx = Var(bhv % HV)
            dht_h_ub[0:HALF_L, 0:2 * D] <<= dht[b_idx, hv_idx, dht_row_begin:dht_row_end, 0:2 * D]
            cast_dht_to_f32_vf(dht_h_ub, dstate_f_ub, row_begin_d)
            snapshot_and_cast_state_vf(dstate_f_ub, state_h_ub, state_h_nz_ub, row_begin_d)

            # Initial (last) chunk C-1: strided BTHVD/BTHVL loads from the frozen
            # public seams (explicit pitch HV*D / HV*L).
            tok0 = Var((C - 1) * L)
            tok1 = Var(tok0 + L)
            gm_to_l1_nd2nz(l1_qg[0][0:L, 0:D], qg[b_idx, tok0:tok1, hv_idx, 0:D], L, D, HV * D, L)
            gm_to_l1_nd2nz(l1_w[0][0:L, 0:D], w[b_idx, tok0:tok1, hv_idx, 0:D], L, D, HV * D, L)
            gm_to_l1_nd2nz(l1_do[0][0:L, 0:D], grad_out[b_idx, tok0:tok1, hv_idx, 0:D], L, D, HV * D, L)
            gm_to_l1_nd2nz(l1_Aqk[0][0:L, 0:L], Aqk[b_idx, tok0:tok1, hv_idx, 0:L], L, L, HV * L, L)
            gm_to_l1_nd2nz(l1_vnew[0][0:L, 0:D], v_new[b_idx, tok0:tok1, hv_idx, 0:D], L, D, HV * D, L)
            _phase_matmul(l0c_seed[0], l1_qg[0].T, l1_do[0].T, phase_l0a, phase_l0b, D, D, L)
            seed_mutex.lock()
            l0c_to_ub(seed_ub[0], l0c_seed[0], M=D, N=D, N_dst=D, M_src=D, dual_mode=DualMode.SPLITM, sub_block_id=0)
            seed_mutex.ready()
            _phase_matmul(l0c_daqk, l1_do[0], l1_vnew[0], phase_l0a, phase_l0b, L, L, D)
            # dAqk -> public BTHVL, scaled by 1/sqrt(D) in the fixpipe (finalize_pre
            # re-trils, so the below-diagonal values here are masked downstream).
            l0c_to_gm_nz2nd(dAqk[b_idx, tok0:tok1, hv_idx, 0:L], l0c_daqk, L, L, HV * L, L, scale=DAQK_SCALE)
            _phase_matmul(l0c_dv0, l1_Aqk[0].T, l1_do[0].T, phase_l0a, phase_l0b, L, D, L)
            dv0_mutex.lock()
            l0c_to_ub(dv_sum_ub, l0c_dv0, M=L, N=D, N_dst=D, M_src=L, dual_mode=DualMode.SPLITM, sub_block_id=0)
            dv0_mutex.ready()

            for rev_c in range(C):
                c_idx = Var(C - 1 - rev_c)
                next_c_idx = Var(c_idx - 1)
                ctok0 = Var(c_idx * L)
                # Shared by both the next-chunk preload and the later dAqk writeback.
                ntok0 = Var(next_c_idx * L)
                # g_last read raw strided from [B,C,HV,D]; exp2 computed in-kernel.
                g_last_half_ub[0:1, 0:HALF_D] <<= g_last[b_idx, c_idx, hv_idx, row_begin_d:row_end_d]
                exp2_glast_vf(g_last_half_ub, exp_g_last_half_ub)

                state_bridge.lock()
                l1_dstate[row_begin_d:row_end_d, 0:D] <<= state_h_nz_ub[0:HALF_D, 0:D].nz()
                state_bridge.ready()

                gm_to_l1_nd2nz(l1_kg[0:L, 0:D], kg[b_idx, ctok0:ctok0 + L, hv_idx, 0:D], L, D, HV * D, L)
                state_bridge.wait()
                _phase_matmul(l0c_dvdelta, l1_kg, l1_dstate.T, phase_l0a, phase_l0b, L, D, D)
                state_bridge.free()
                # dh -> inverse seam BCHVDD [B,C,HV,D,D] (contiguous [D,D] sub-block).
                dh[b_idx, c_idx, hv_idx, row_begin_d:row_end_d, 0:D] <<= state_h_ub[0:HALF_D, 0:D]

                dvdelta_mutex.lock()
                l0c_to_ub(dv_delta_ub, l0c_dvdelta, M=L, N=D, N_dst=D, M_src=L, dual_mode=DualMode.SPLITM, sub_block_id=0)
                dvdelta_mutex.ready()

                dvdelta_mutex.wait()
                dv0_mutex.wait()
                add_dv_and_cast_bf16_vf(dv_delta_ub, dv_sum_ub, dv_bf16_ub, dv_bf16_nz_ub)
                dvdelta_mutex.free()
                # dv_sum_ub is consumed; the next chunk's dv0 fixpipe may
                # overwrite it as soon as the prefetch runs.
                dv0_mutex.free()

                dv_bridge.lock()
                l1_dv[row_begin_l:row_end_l, 0:D] <<= dv_bf16_nz_ub[0:HALF_L, 0:D].nz()
                dv_bridge.ready()
                # dv -> public BTHVD [B,T,HV,D], strided (gap (HV-1)*D).
                ub_to_gm_pad(dv[b_idx, ctok0 + row_begin_l:ctok0 + row_end_l, hv_idx, 0:D], dv_bf16_ub[0:HALF_L, 0:D], HALF_L, D, 0, (HV - 1) * D)

                if rev_c + 1 < C:
                    gm_to_l1_nd2nz(l1_qg[rev_c + 1][0:L, 0:D], qg[b_idx, ntok0:ntok0 + L, hv_idx, 0:D], L, D, HV * D, L)
                    gm_to_l1_nd2nz(l1_w[rev_c + 1][0:L, 0:D], w[b_idx, ntok0:ntok0 + L, hv_idx, 0:D], L, D, HV * D, L)
                    gm_to_l1_nd2nz(l1_do[rev_c + 1][0:L, 0:D], grad_out[b_idx, ntok0:ntok0 + L, hv_idx, 0:D], L, D, HV * D, L)
                    gm_to_l1_nd2nz(l1_Aqk[rev_c + 1][0:L, 0:L], Aqk[b_idx, ntok0:ntok0 + L, hv_idx, 0:L], L, L, HV * L, L)
                    gm_to_l1_nd2nz(l1_vnew[rev_c + 1][0:L, 0:D], v_new[b_idx, ntok0:ntok0 + L, hv_idx, 0:D], L, D, HV * D, L)
                    _phase_matmul(l0c_seed[rev_c + 1], l1_qg[rev_c + 1].T, l1_do[rev_c + 1].T, phase_l0a, phase_l0b, D, D, L)
                    seed_mutex.lock()
                    l0c_to_ub(seed_ub[rev_c + 1], l0c_seed[rev_c + 1], M=D, N=D, N_dst=D, M_src=D, dual_mode=DualMode.SPLITM, sub_block_id=0)
                    seed_mutex.ready()

                dv_bridge.wait()
                _phase_matmul(l0c_corr, l1_w[rev_c].T, l1_dv.T, phase_l0a, phase_l0b, D, D, L)
                dv_bridge.free()

                corr_mutex.lock()
                l0c_to_ub(corr_ub, l0c_corr, M=D, N=D, N_dst=D, M_src=D, dual_mode=DualMode.SPLITM, sub_block_id=0)
                corr_mutex.ready()

                if rev_c + 1 < C:
                    _phase_matmul(l0c_daqk, l1_do[rev_c + 1], l1_vnew[rev_c + 1], phase_l0a, phase_l0b, L, L, D)
                    l0c_to_gm_nz2nd(dAqk[b_idx, ntok0:ntok0 + L, hv_idx, 0:L], l0c_daqk, L, L, HV * L, L, scale=DAQK_SCALE)
                    _phase_matmul(l0c_dv0, l1_Aqk[rev_c + 1].T, l1_do[rev_c + 1].T, phase_l0a, phase_l0b, L, D, L)
                    dv0_mutex.lock()
                    l0c_to_ub(dv_sum_ub, l0c_dv0, M=L, N=D, N_dst=D, M_src=L, dual_mode=DualMode.SPLITM, sub_block_id=0)
                    dv0_mutex.ready()
                seed_mutex.wait()
                corr_mutex.wait()
                if rev_c + 1 < C:
                    update_dstate_vf(seed_ub[rev_c], corr_ub, dstate_f_ub, state_h_ub, state_h_nz_ub, exp_g_last_half_ub, row_begin_d)
                else:
                    update_dstate_final_vf(seed_ub[rev_c], corr_ub, dstate_f_ub, state_h_ub, exp_g_last_half_ub, row_begin_d)
                corr_mutex.free()
                seed_mutex.free()

            dh0[b_idx, hv_idx, row_begin_d:row_end_d, 0:D] <<= state_h_ub[0:HALF_D, 0:D]
            bar_all()

    return dAqk, dh, dv, dh0
