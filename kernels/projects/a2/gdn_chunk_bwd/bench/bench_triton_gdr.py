"""fla Triton `chunk_gated_delta_rule` on the Ascend NPU — the backward baseline.

Same op as the a2 GDN chunk kernel (scalar per-head gate delta rule), same NPU, run
under a warmup+timed loop so `msprof` captures its device time and host wall is
reported. Measures forward, and forward+backward (backward-only = the difference).

    ASCEND_RT_VISIBLE_DEVICES=<d> PYTHONPATH=<fla_root> \
        python bench_triton_gdr.py --shape s512 --mode fb --iters 50

Conventions matched to the a2 kernel: raw (un-normalized) q/k, external scale
1/sqrt(128), beta in (0,1) used directly, g a log-decay <= 0. fla does the scale
internally (`scale=`), l2norm and beta-sigmoid off.
"""
import argparse
import time

import torch
import torch_npu  # noqa: F401
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

DEV = "npu:0"
SCALE = 128 ** -0.5
D = 128
SHAPES = {"s128": (1, 128, 2), "s512": (1, 512, 4), "s1024": (1, 1024, 4), "s2048": (1, 2048, 4)}


def make_inputs(B, T, H, seed=17, req=False):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s, sc=1.0: (torch.randn(*s, generator=gen) * sc).float().to(DEV)
    q = r(B, T, H, D, sc=0.05)
    k = r(B, T, H, D, sc=0.05)
    v = r(B, T, H, D, sc=0.05)
    g = (-torch.rand(B, T, H, generator=gen) * 0.1).float().to(DEV)
    beta = torch.rand(B, T, H, generator=gen).float().to(DEV)
    if req:
        for t in (q, k, v, g, beta):
            t.requires_grad_(True)
    return q, k, v, g, beta


def fwd(q, k, v, g, beta):
    return chunk_gated_delta_rule(q, k, v, g, beta, scale=SCALE,
                                  use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False)


def timed(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=tuple(SHAPES), default="s512")
    ap.add_argument("--mode", choices=("fwd", "fb"), default="fb")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    a = ap.parse_args()
    B, T, H = SHAPES[a.shape]
    do = torch.randn(B, T, H, D, device=DEV)

    if a.mode == "fwd":
        q, k, v, g, beta = make_inputs(B, T, H)
        call = lambda: fwd(q, k, v, g, beta)
    else:
        def call():
            q, k, v, g, beta = make_inputs(B, T, H, req=True)
            o, *_ = fwd(q, k, v, g, beta)
            (o * do).sum().backward()

    us = timed(call, a.iters, a.warmup)
    print(f"[triton-gdr {a.mode} {a.shape} B{B}T{T}H{H}] host wall {us:.2f} us/call over {a.iters} iters")


if __name__ == "__main__":
    main()
