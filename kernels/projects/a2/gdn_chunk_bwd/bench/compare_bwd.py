"""End-to-end backward comparison: a2 GDN chunk kernel vs fla Triton, same NPU.

Runs both backwards on identical raw inputs (HV == H), checks each gradient's
rel-L2 agreement, and reports the host-wall speedup. This is the correctness
cross-check plus the wall-time headline; device time / pipe utilization come from
`msprof_bwd.sh` / `pipe_bwd.sh` wrapping the per-side bench drivers.

    ASCEND_RT_VISIBLE_DEVICES=<d> PYTHONPATH=<fla_root> \
        python compare_bwd.py --shape s512 --iters 50
"""
import argparse

import torch
import torch_npu  # noqa: F401

import bench_a2_bwd as A
import bench_triton_gdr as Tg

DEV = A.DEV
D = A.D
SCALE = A.SCALE


def rel_l2(x, ref):
    x = x.reshape(-1).float().detach().cpu()
    ref = ref.reshape(-1).float().detach().cpu()
    return (x - ref).norm().item() / max(ref.norm().item(), 1e-30)


_STEMS = {"v1": "V1", "v2": "V2", "v3": "V3"}
_BUILDERS = {"v1": "build_call", "v2": "build_call_v2", "v3": "build_call_v3"}


def _stems_builder(variant):
    return getattr(A, _STEMS[variant]), getattr(A, _BUILDERS[variant])


def run_a2(x, B, T, H, variant="v1"):
    stems, builder = _stems_builder(variant)
    stg = A.compile_stages(stems)
    call = builder(stg, x, B, T, H, H)
    return call()


def run_fla(x, B, T, H, do):
    q = x["q"].reshape(B, T, H, D).clone().requires_grad_(True)
    k = x["k"].reshape(B, T, H, D).clone().requires_grad_(True)
    v = x["v"].reshape(B, T, H, D).clone().requires_grad_(True)
    g = x["g"].reshape(B, T, H).clone().requires_grad_(True)
    beta = x["beta"].reshape(B, T, H).clone().requires_grad_(True)
    o, *_ = Tg.fwd(q, k, v, g, beta)
    (o * do.reshape(B, T, H, D)).sum().backward()
    return dict(dq=q.grad, dk=k.grad, dv=v.grad, dg=g.grad, dbeta=beta.grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=tuple(A.SHAPES), default="s512")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--variant", choices=("v1", "v2", "v3"), default="v1")
    a = ap.parse_args()
    B, T, H = A.SHAPES[a.shape]
    x = A.make_inputs(B, T, H, H)

    mine = run_a2(x, B, T, H, a.variant)
    ref = run_fla(x, B, T, H, x["dout"])

    print(f"[compare {a.variant} {a.shape} B{B}T{T}H{H}] rel-L2 a2-kernel vs fla-triton:")
    for name in ("dq", "dk", "dv", "dg", "dbeta"):
        m = mine[name].reshape(B, T, H, -1) if name in ("dq", "dk", "dv") else mine[name].reshape(B, T, H)
        print(f"    {name:6s} {rel_l2(m, ref[name]):.3e}")

    stems, builder = _stems_builder(a.variant)
    stg = A.compile_stages(stems)
    a2_call = builder(stg, x, B, T, H, H)
    q, k, v, g, beta = Tg.make_inputs(B, T, H, req=True)
    do = x["dout"].reshape(B, T, H, D)
    fla_call = lambda: (lambda o: (o * do).sum().backward())(Tg.fwd(q, k, v, g, beta)[0])
    t_a2 = A.timed(a2_call, a.iters, a.warmup)
    t_fla = Tg.timed(fla_call, a.iters, a.warmup)
    print(f"[compare {a.variant} {a.shape}] host wall  a2-kernel {t_a2:.1f}us  fla-triton(fwd+bwd) {t_fla:.1f}us")


if __name__ == "__main__":
    main()
