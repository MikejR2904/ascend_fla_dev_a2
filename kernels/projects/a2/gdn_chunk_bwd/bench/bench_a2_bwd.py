"""Device driver + host-wall timer for the a2 GDN chunk backward (8-kernel chain).

Compiles every backward stage for the a2 (910B3) profile and runs the full
reverse chain under a warmup+timed loop, so `msprof` captures per-stage AI Vector
Core device time and this script reports host wall (`torch.npu.synchronize`).

    ASCEND_RT_VISIBLE_DEVICES=<d> python bench_a2_bwd.py --shape s512 --iters 50

The chain mirrors the sim-validated reference order (replay -> reverse_rec ->
reverse_dq -> reverse_dbeta -> reverse_dg -> reverse_dk_bz -> reverse_dk_ddr ->
group_reduce); it is the current shipped structure, i.e. the pre-optimization
baseline. HV defaults to H (the fla-comparable, ratio-1 case).
"""
import argparse
import importlib.util
import pathlib as _pl
import sys
import time

import torch
import torch_npu  # noqa: F401

_REPO = str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "ascend_fla").is_dir()))
sys.path.insert(0, _REPO)
from ascend_fla.runtime.compile import compile_kernel  # noqa: E402

_KDIR = _pl.Path(__file__).resolve().parent.parent / "kernels"
DEV = "npu:0"
BLOCK_DIM = 40
SCALE = 128 ** -0.5
D = 128

# shape -> (B, T, H); T must be a multiple of the chunk size 64.
SHAPES = {
    "s128": (1, 128, 2),
    "s512": (1, 512, 4),
    "s1024": (1, 1024, 4),
    "s2048": (1, 2048, 4),
}


def _kernel(module_stem, fn_name):
    spec = importlib.util.spec_from_file_location("bwd_" + module_stem, _KDIR / (module_stem + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, fn_name)


_STAGES = (
    ("replay", "gdn_chunk_bwd_replay_a2_kernel"),
    ("reverse_rec", "gdn_chunk_bwd_reverse_rec_a2_kernel"),
    ("reverse_dq", "gdn_chunk_bwd_reverse_dq_a2_kernel"),
    ("reverse_dbeta", "gdn_chunk_bwd_reverse_dbeta_a2_kernel"),
    ("reverse_dg", "gdn_chunk_bwd_reverse_dg_a2_kernel"),
    ("reverse_dk_bz", "gdn_chunk_bwd_reverse_dk_bz_a2_kernel"),
    ("reverse_dk_ddr", "gdn_chunk_bwd_reverse_dk_ddr_a2_kernel"),
    ("group_reduce", "gdn_chunk_bwd_group_reduce_a2_kernel"),
)


def compile_stages():
    return {stem: compile_kernel(_kernel(stem, fn), device="a2", block_dim=BLOCK_DIM, backend="cce")
            for stem, fn in _STAGES}


def make_inputs(B, T, H, HV, seed=17):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s, sc=1.0: (torch.randn(*s, generator=gen) * sc).float().to(DEV)
    q = r(B * T, H * D, sc=0.05)
    k = r(B * T, H * D, sc=0.05)
    v = r(B * T, HV * D, sc=0.05)
    dout = r(B * T, HV * D, sc=0.05)
    g = (-torch.rand(B * T, HV, generator=gen) * 0.1).float().to(DEV)
    beta = torch.rand(B * T, HV, generator=gen).float().to(DEV)
    h0 = torch.zeros(B, HV, D, D, device=DEV)
    dht = torch.zeros(B, HV, D, D, device=DEV)
    return dict(q=q, k=k, v=v, dout=dout, g=g, beta=beta, h0=h0, dht=dht)


def build_call(stg, x, B, T, H, HV):
    N = T // 64
    z2 = lambda cols: torch.zeros(B * T, cols, device=DEV)
    zhv = lambda: torch.zeros(B * T, HV, device=DEV)
    tape5 = lambda: torch.zeros(B, HV, T, D, D, device=DEV)
    dims = dict(B=B, T=T, H=H, HV=HV, N=N, BT=B * T, HD=H * D, HVD=HV * D)

    tape_d = tape5()
    back_tape = tape5()
    dv = z2(HV * D)
    dh0 = torch.zeros(B, HV, D, D, device=DEV)
    dq_parts = z2(HV * D)
    dkbz = z2(HV * D)
    dkddr = z2(HV * D)
    dbeta = zhv()
    dg = zhv()
    dq = z2(H * D)
    dk = z2(H * D)

    def call():
        stg["replay"](dict(k=x["k"], v=x["v"], g=x["g"], beta=x["beta"], initial_state=x["h0"]),
                      dims, dict(tape_d=tape_d))
        stg["reverse_rec"](dict(q=x["q"], k=x["k"], g=x["g"], beta=x["beta"], dout=x["dout"], dht=x["dht"]),
                           dict(scale=SCALE, **dims), dict(dv=dv, dh0=dh0, back_tape=back_tape))
        stg["reverse_dq"](dict(k=x["k"], v=x["v"], beta=x["beta"], dout=x["dout"], tape_d=tape_d),
                          dict(scale=SCALE, **dims), dict(dq_parts=dq_parts))
        stg["reverse_dbeta"](dict(k=x["k"], v=x["v"], tape_d=tape_d, back_tape=back_tape),
                             dims, dict(dbeta=dbeta))
        stg["reverse_dg"](dict(k=x["k"], beta=x["beta"], tape_d=tape_d, back_tape=back_tape),
                          dims, dict(dg=dg))
        stg["reverse_dk_bz"](dict(k=x["k"], v=x["v"], beta=x["beta"], tape_d=tape_d, back_tape=back_tape),
                             dims, dict(dkbz_parts=dkbz))
        stg["reverse_dk_ddr"](dict(k=x["k"], beta=x["beta"], tape_d=tape_d, back_tape=back_tape),
                              dims, dict(dkddr_parts=dkddr))
        stg["group_reduce"](dict(dq_parts=dq_parts, dkbz_parts=dkbz, dkddr_parts=dkddr),
                            dims, dict(dq=dq, dk=dk))
        return dict(dq=dq, dk=dk, dv=dv, dg=dg, dbeta=dbeta, dh0=dh0)

    return call


def timed(call, iters, warmup):
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        call()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=tuple(SHAPES), default="s512")
    ap.add_argument("--hv", type=int, default=0, help="value heads; 0 => equal to H")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    a = ap.parse_args()
    B, T, H = SHAPES[a.shape]
    HV = a.hv or H

    stg = compile_stages()
    x = make_inputs(B, T, H, HV)
    call = build_call(stg, x, B, T, H, HV)
    us = timed(call, a.iters, a.warmup)
    print(f"[a2-bwd {a.shape} B{B}T{T}H{H}HV{HV}] host wall {us:.2f} us/call over {a.iters} iters "
          f"(warmup {a.warmup}, block_dim {BLOCK_DIM})")


if __name__ == "__main__":
    main()
