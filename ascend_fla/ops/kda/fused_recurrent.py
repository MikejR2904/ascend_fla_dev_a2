"""Token-major KDA decode, T=1..16, native BF16 or FP32 with FP32 state.

The custom kernel addresses GVA heads and performs scaling/conversion directly.
Default flags=False uses only host metadata, output allocation and launch.
Enabled raw flags use the typed KDA preparation kernels before native decode.
"""
from __future__ import annotations

import functools
import pathlib
from typing import Any

import torch

from ascend_fla.platform import capability

from .chunk import HEAD_DIM, VALUE_DIM, _load_kernel, _prepare_kernel_inputs, _prep_runtime

#: a5 decode 路径的 block_dim，从能力表读（a5 的别名）。真值住在 ``platform.CAPABILITIES``。
#: 这个 kernel 只用向量核、不碰 cube，``GetVecNum() == 2 * block_dim``，比 chunk 的 4 宽；
#: 上限仍要实测（超过物理核数会在硬件 barrier 死锁，见 AGENTS.md §5）。
SUPPORTED_BLOCK_DIM = capability("a5")["supported_block_dim"]["decode"]


def _unit_root() -> pathlib.Path:
    repo = pathlib.Path(__file__).resolve().parents[3]
    root = repo / "kernels/projects/a5/kda_fused_recurrent"
    if not (root / "kernels").is_dir():
        raise FileNotFoundError(f"找不到本仓的 kda_fused_recurrent 单元（试了 {root}）")
    return root


@functools.lru_cache(maxsize=1)
def kda_fused_recurrent_kernel() -> Any:
    return _load_kernel("fr", _unit_root() / "kernels", "step",
                        "kda_fused_recurrent_kernel")


@functools.lru_cache(maxsize=None)
def _compiled_pair(device: str, block_dim: int) -> dict:
    from ...runtime.compile import compile_kernel
    _prep_runtime().prepare(device, block_dim, "decode")
    return {dtype: compile_kernel(_native_kernel(dtype), device=device,
                                  block_dim=block_dim)
            for dtype in (torch.float32, torch.bfloat16)}


@functools.lru_cache(maxsize=2)
def _native_kernel(dtype: torch.dtype) -> Any:
    root = pathlib.Path(__file__).resolve().parents[3]
    directory = root / "kernels/projects/a5/kda_fused_recurrent_bf16/kernels"
    suffix = "bf16" if dtype == torch.bfloat16 else "fp32"
    return _load_kernel("fr_native", directory, "step", f"kda_decode_{suffix}_kernel")


@functools.lru_cache(maxsize=None)
def _compiled(device: str, block_dim: int, dtype: torch.dtype = torch.float32) -> Any:
    # prepare(decode=True) retains its existing two-argument call and registers
    # BOTH dtype vendors before the first custom execution in the process.
    return _compiled_pair(device, block_dim)[dtype]


def _check(q, k, v, g, beta, initial_state, block_dim) -> tuple[int, int, int, int]:
    """门控。返回 ``(B, T, H, HV)``。不满足就报错，绝不静默降级（AGENTS.md §7）。"""
    if block_dim not in SUPPORTED_BLOCK_DIM:
        raise ValueError(
            f"block_dim 只支持 {SUPPORTED_BLOCK_DIM}，收到 {block_dim}；"
            f"本单元只用向量核（GetVecNum() == 2*block_dim，物理 56 个），"
            f"但超过物理核数会在硬件 barrier 死锁"
        )
    for name, value, rank in (("q", q, 4), ("k", k, 4), ("v", v, 4),
                              ("g", g, 4), ("beta", beta, 3)):
        if not isinstance(value, torch.Tensor) or value.dim() != rank:
            raise ValueError(f"{name} must be a rank-{rank} tensor")
    if q.dim() != 4:
        raise ValueError(f"q 应为 4 维 [B,T,H,D]，收到 {tuple(q.shape)}")
    b, t, h, kd = q.shape
    hv, vd = v.shape[2], v.shape[3]
    if min(b, t, h, hv) < 1:
        raise ValueError("B, T, H and HV must all be positive")
    if kd != HEAD_DIM or vd != VALUE_DIM:
        raise ValueError(f"定尺要求 K=V={HEAD_DIM}，收到 K={kd} V={vd}")
    if hv % h:
        raise ValueError(f"HV({hv}) 必须是 H({h}) 的整数倍")
    if k.shape != q.shape:
        raise ValueError(f"k 的形状应与 q 相同，收到 {tuple(k.shape)} vs {tuple(q.shape)}")
    if v.shape != (b, t, hv, vd):
        raise ValueError(f"v 应为 [B,T,HV,V]，收到 {tuple(v.shape)}")
    if g.shape != (b, t, hv, kd):
        raise ValueError(f"g 应为 [B,T,HV,K]，收到 {tuple(g.shape)}")
    if beta.shape != (b, t, hv):
        raise ValueError(f"beta 应为 [B,T,HV]，收到 {tuple(beta.shape)}")
    if initial_state is not None and initial_state.shape != (b, hv, kd, vd):
        raise ValueError(
            f"initial_state 应为 [B,HV,K,V]={(b, hv, kd, vd)}（K 在前），"
            f"收到 {tuple(initial_state.shape)}；fla 的 KDA layer 用 V 在前，"
            f"见 gaps.json 的 state-layout-k-first"
        )
    # T 的上限由 kernel 里流式缓冲的行数定（step.py 的 T_MAX）。超了要分批调用 ——
    # 报错而不是自动分批：自动分批会把一次调用的语义悄悄变成多次，state 串接的
    # 正确性得另外验，那不是这层该默默做的决定。
    if t > T_MAX:
        raise ValueError(
            f"一次调用最多 {T_MAX} 个 token（kernel 内流式缓冲的行数），收到 T={t}；"
            f"更长的序列请走 chunk 路径，或自己按 {T_MAX} 分批并串接 state"
        )
    return b, t, h, hv


def _check_types(q, k, v, g, beta, initial_state):
    if v.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("decode q/k/v must be all FP32 or all BF16; FP16/FP64 are unsupported")
    if q.dtype != v.dtype or k.dtype != v.dtype:
        raise ValueError("decode q/k/v must share one dtype: FP32 or BF16")
    for name, value in (("q", q), ("k", k), ("v", v), ("g", g),
                         ("beta", beta), ("initial_state", initial_state)):
        if value is None:
            continue
        if name in ("g", "beta", "initial_state") and value.dtype != torch.float32:
            raise ValueError(f"{name} must be FP32 for native decode")
        if value.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous for native token-major decode")


#: 与 ``kernels/.../step.py`` 里流式缓冲的行数一致。改那边要同时改这里 ——
#: 由 tests/test_kda_decode.py 锁住。
T_MAX = 16


def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    *,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    check_domain: bool = True,
    device: str | None = None,
    block_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """逐 token 递推的 KDA 前向（decode）。

    Args:
        q, k: ``[B,T,H,128]``；v, g: ``[B,T,HV,128]``；beta: ``[B,T,HV]``。
            q/k/v 全 BF16 或全 FP32；g/beta 为 FP32，所有输入连续且同设备。
        scale: q 的缩放，默认 ``128 ** -0.5``，在 kernel 内以 FP32 相乘。
        initial_state: ``[B,HV,128,128]`` float32，K 在前。
        A_log, dt_bias: Raw-gate parameters with shapes ``[HV]`` and ``[HV*K]``;
            both are required when ``use_gate_in_kernel=True``.
        use_qk_l2norm_in_kernel: Normalize raw q/k in FP32 with additive
            squared epsilon 1e-6, then round to v.dtype before the existing ABI.
        use_gate_in_kernel: Apply ``-exp(A_log)*softplus(g+dt_bias)`` in FP32.
        use_beta_sigmoid_in_kernel: Apply sigmoid to raw beta logits in FP32.
            All three preparations execute in typed custom kernels.
        check_domain: Accepted for API symmetry; heuristic domain checks run
            only in chunk mode, never on each decode step. Shape checks remain.

    Returns:
        ``(o, final_state)``，``o`` 为 ``[B,T,HV,128]``（与输入同 dtype），
        ``final_state`` 为 ``[B,HV,128,128]`` float32 或 ``None``。
    """
    from ascend_fla.platform import require_qualified, resolve_soc
    device = resolve_soc(device)
    require_qualified(device)

    b, t, h, hv = _check(q, k, v, g, beta, initial_state, block_dim)
    q, k, g, beta = _prepare_kernel_inputs(
        q, k, g, beta, A_log=A_log, dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        qk_dtype=v.dtype,
        device=device, block_dim=block_dim, namespace="decode",
    )
    _check_types(q, k, v, g, beta, initial_state)
    sc = HEAD_DIM ** -0.5 if scale is None else float(scale)
    dev = q.device
    o = torch.empty(b, t, hv, VALUE_DIM, dtype=v.dtype, device=dev)
    final_state = torch.empty(b, hv, HEAD_DIM, VALUE_DIM, dtype=torch.float32, device=dev)
    # The absent initial-state pointer is never read: initialization is in VF.
    state0 = initial_state if initial_state is not None else final_state
    _compiled(device, block_dim, v.dtype)(
        {"q": q, "k": k, "v": v, "g": g, "beta": beta, "initial_state": state0},
        {"B": b, "HV": hv, "H": h, "T": t,
         "has_initial": int(initial_state is not None), "scale": sc},
        {"o": o, "final_state": final_state},
    )
    return o, (final_state if output_final_state else None)
