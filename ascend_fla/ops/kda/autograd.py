"""Differentiable KDA chunk attention with explicit raw-input flags.

The first five arguments follow FLA's q/k/v/g/beta order. By default they are
already prepared: normalized q/k, log-decay g, and sigmoid beta. The optional
``use_*_in_kernel`` flags request FP32 preparation in custom kernels. Training
uses custom preparation derivatives, including deterministic gate-parameter
reductions, and preserves the raw input gradient dtypes.

Chunk mode checks the prepared-input domain by default. This is a heuristic:
small unnormalized q/k cannot be distinguished from prepared inputs. Opting out
with ``check_domain=False`` leaves ABI and gate-span checks enabled.

反向依赖九个前向检查点，forward 里一并算出并 ``save_for_backward``。这比重算前向省
一次完整前向，代价是显存：检查点里 ``h`` 是 ``[B,C,HV,128,128]``，在
kimi_linear_layer 形状（B1/HV32/C16）下是 16MB（bf16）。
"""
from __future__ import annotations

import torch

from .chunk import (
    BWD_CACHE_NAMES,
    HEAD_DIM,
    VALUE_DIM,
    _check_input_domain,
    _prepare_inputs,
    _prepare_kernel_inputs,
    _validate_raw_inputs,
    _prep_runtime,
    _layout_runtime,
    chunk_kda_fwd,
    chunk_kda_fwd_with_caches,
)
from .chunk_bwd import chunk_kda_bwd

__all__ = ["chunk_kda"]


class _RawNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, device, block_dim):
        ctx.save_for_backward(source)
        ctx.options = dict(device=device, block_dim=block_dim)
        return _prep_runtime().norm(source, torch.bfloat16, **ctx.options)

    @staticmethod
    def backward(ctx, sensitivity):
        source, = ctx.saved_tensors
        sensitivity = _layout_runtime().cast(sensitivity, torch.bfloat16, **ctx.options)
        return _prep_runtime().norm_backward(source, sensitivity, **ctx.options), None, None


class _RawGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, alog, bias, device, block_dim):
        ctx.save_for_backward(source, alog, bias)
        ctx.options = dict(device=device, block_dim=block_dim)
        return _prep_runtime().gate(source, alog, bias, **ctx.options)

    @staticmethod
    def backward(ctx, sensitivity):
        source, alog, bias = ctx.saved_tensors
        sensitivity = _layout_runtime().cast(sensitivity, torch.float32, **ctx.options)
        dg, da, db = _prep_runtime().gate_backward(source, alog, bias, sensitivity, **ctx.options)
        need = ctx.needs_input_grad
        return dg if need[0] else None, da if need[1] else None, db if need[2] else None, None, None


class _RawBeta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, device, block_dim):
        ctx.options = dict(device=device, block_dim=block_dim)
        probability = _prep_runtime().beta(source, **ctx.options)
        ctx.save_for_backward(source, probability)
        return probability

    @staticmethod
    def backward(ctx, sensitivity):
        source, probability = ctx.saved_tensors
        sensitivity = _layout_runtime().cast(sensitivity, torch.float32, **ctx.options)
        result = _prep_runtime().beta_backward(source, probability, sensitivity, **ctx.options)
        return result, None, None


def _prepare_training_inputs(q, k, g, beta, *, A_log=None, dt_bias=None,
                             use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                             use_beta_sigmoid_in_kernel=False, device=None, block_dim=1,
                             impl='stable'):
    """Retain graph edges at each enabled native preparation boundary."""
    _validate_raw_inputs(q, k, g, beta, A_log=A_log, dt_bias=dt_bias,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                        use_gate_in_kernel=use_gate_in_kernel,
                        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel)
    if not (use_qk_l2norm_in_kernel or use_gate_in_kernel or use_beta_sigmoid_in_kernel):
        return q, k, g, beta
    from ascend_fla.platform import resolve_soc
    device = resolve_soc(device)
    runtime = _prep_runtime()
    sources = []
    if use_qk_l2norm_in_kernel:
        if q.shape[-1] != HEAD_DIM or min(q.shape) <= 0:
            raise ValueError('raw q/k require positive [B,T,H,128]')
        sources += [('q', q), ('k', k)]
    if use_gate_in_kernel:
        if g.shape[-1] != HEAD_DIM or min(g.shape) <= 0:
            raise ValueError('raw g requires positive [B,T,HV,128]')
        sources += [('g', g), ('A_log', A_log), ('dt_bias', dt_bias)]
    if use_beta_sigmoid_in_kernel:
        sources += [('beta', beta)]
    for name, source in sources:
        runtime._check_source(name, source)
    # Vendor installation after the first custom launch is unsafe. Precompile
    # the complete forward, inherited backward, layout and new prep chain here.
    from .chunk import _compiled_chain as fwd_chain
    from .chunk_bwd import _compiled_chain as bwd_chain
    fwd_chain(device, block_dim, impl)
    bwd_chain(device, block_dim, impl)
    runtime.prepare_backward(device, block_dim)
    if use_gate_in_kernel:
        g = _RawGate.apply(g, A_log, dt_bias, device, block_dim)
    if use_qk_l2norm_in_kernel:
        q = _RawNorm.apply(q, device, block_dim)
        k = _RawNorm.apply(k, device, block_dim)
    if use_beta_sigmoid_in_kernel:
        beta = _RawBeta.apply(beta, device, block_dim)
    return q, k, g, beta


class _ChunkKDA(torch.autograd.Function):
    """KDA 的 autograd 封装。不支持二阶导（检查点不参与建图）。"""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state,
                device, block_dim, layout_device, check_gate_range, impl):
        o, final_state, caches = chunk_kda_fwd_with_caches(
            q, k, v, g, beta, scale, initial_state,
            device=device, block_dim=block_dim, layout_device=layout_device,
            check_gate_range=check_gate_range, impl=impl,
        )
        # beta 在前向 ABI 里是 fp32、反向 ABI 里是 bf16。这一步降精度的代价已量化
        # （见 gaps.json 的 kda-fwd-bwd-dtype-mismatch），**显式**做，不当无害的类型适配。
        beta_bf16 = _layout_runtime().cast(beta, torch.bfloat16, device=device, block_dim=block_dim)
        ctx.save_for_backward(q, k, v, beta_bf16, *(caches[n] for n in BWD_CACHE_NAMES))
        ctx.bwd_options = dict(device=device, block_dim=block_dim, impl=impl)
        ctx.state_shape = (q.shape[0], v.shape[2], HEAD_DIM, VALUE_DIM)
        return o, final_state

    @staticmethod
    def backward(ctx, do, dht):
        q, k, v, beta_bf16, *cache_list = ctx.saved_tensors
        caches = dict(zip(BWD_CACHE_NAMES, cache_list))

        # 上游梯度的 dtype/连续性都不能假定：下游算子可能升到 fp32，也可能给非连续视图
        layout_options = {name: ctx.bwd_options[name] for name in ("device", "block_dim")}
        do = _layout_runtime().cast(do, torch.bfloat16, **layout_options)
        if dht is None:
            # final_state 没参与 loss。反向 kernel 没有"省略 dht"的入口，只能喂零。
            dht = _layout_runtime().zeros(ctx.state_shape, torch.bfloat16, do.device, **layout_options)
        else:
            dht = _layout_runtime().cast(dht, torch.bfloat16, **layout_options)

        grads = chunk_kda_bwd(q=q, k=k, v=v, beta=beta_bf16, do=do, dht=dht,
                              caches=caches, **ctx.bwd_options)

        # 反向 ABI 全出 bf16，而 g / beta / initial_state 的前向入口是 fp32 ——
        # 升回去，否则 autograd 会因 dtype 不匹配报错。
        # 用 ctx.needs_input_grad 而不是在 forward 里记 requires_grad：前者看的是整张图，
        # 后者只看那一刻的叶子标记。顺序与 forward 的参数表一致。
        need = ctx.needs_input_grad
        return (
            grads["dq"] if need[0] else None,
            grads["dk"] if need[1] else None,
            grads["dv"] if need[2] else None,
            _layout_runtime().cast(grads["dg"], torch.float32, **layout_options) if need[3] else None,
            _layout_runtime().cast(grads["dbeta"], torch.float32, **layout_options) if need[4] else None,
            None,                                              # scale
            _layout_runtime().cast(grads["dh0"], torch.float32, **layout_options) if need[6] else None,          # initial_state
            # output_final_state / device / block_dim / layout_device /
            # check_gate_range / impl
            None, None, None, None, None, None,
        )


def chunk_kda(
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
    layout_device: str = "auto",
    check_gate_range: bool = True,
    impl: str = "stable",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """可求导的 KDA 分块注意力。

    Args:
        q, k: ``[B, T, H, 128]`` bfloat16, already normalized by default.
            With ``use_qk_l2norm_in_kernel=True``, BF16 or FP32 raw inputs are
            normalized in FP32 with ``x/sqrt(sum(x*x)+1e-6)`` then cast to BF16.
        v: ``[B, T, HV, 128]`` bfloat16，``HV % H == 0``。
        g: ``[B, T, HV, 128]`` float32，log 空间的 per-channel 衰减**增量**。
            Already ``-exp(A_log) * softplus(g_raw + dt_bias)`` by default;
            ``use_gate_in_kernel=True`` accepts BF16 or FP32 raw gate inputs instead.
        beta: ``[B, T, HV]`` float32, post-sigmoid by default;
            ``use_beta_sigmoid_in_kernel=True`` accepts BF16 or FP32 raw logits.
        A_log, dt_bias: Required for raw gate preparation, with exact shapes
            ``[HV]`` and flattened ``[HV*128]``. Gate/beta preparation is FP32.
        check_domain: On prepared routes (their flag is False), require finite
            ``g <= 0``, ``0 < beta < 1`` and q/k L2 norm <= 1 + 2^-8.
            The BF16 margin is measured; small raw vectors can still pass.
            False disables only this heuristic, never ABI or gate-span guards.
        scale: q 的缩放，默认 ``128 ** -0.5``。
        initial_state: ``[B, HV, 128, 128]`` float32，可选、可求导。
            注意布局是 **K 在前**；fla 的 KDA layer 用 ``state_v_first=True``，
            即 V 在前 —— 对接它的 cache 要转置（见 gaps.json 的 ``state-layout-k-first``）。
        output_final_state: 是否返回末态。
        device: ascriptor 设备名。
        block_dim: 启动核组数，只接受 ``SUPPORTED_BLOCK_DIM``。
        layout_device: 见 :func:`~ascend_fla.ops.kda.chunk.chunk_kda_fwd`。
        check_gate_range: 校验 chunk 内门控跨度不超过
            :data:`~ascend_fla.ops.kda.chunk.MAX_GATE_SPAN` ``[impl]["backward"]``
            （``stable`` 下 105）。**这条闸守的是精度而不是有限性** —— 反向到跨度 169.8
            都还是有限值，但 ``dq`` 对 fp32 递推参考的相对 L2 在 130 处就越过契约预算
            0.05。让它"有限但超预算"地跑过去就是静默降级（AGENTS.md §7）。曲线见
            ``kernels/projects/a5/kda_bwd_stable/contract.json`` 的
            ``domain.gate_span.accuracy_vs_span``。
        impl: ``"stable"``（默认；前向可用跨度 155、反向 105）或 ``"upstream"``（两条都 80）。
            **同时选前向与反向两条链**，没有分开的开关 —— 两条链的失效点不同
            （前向在 87.3 下溢、反向在 88.72 上溢），混用会让门控检查的上限对不上实际
            会失效的那一侧。fla 默认初始化的 KDA 层跨度约 94，``upstream`` 两边都撑不住。
            见 ``gaps.json`` 的 ``gate-range-beyond-declared`` 与 ``bwd-gate-range-overflow``。

    Returns:
        ``(o, final_state)``。``o`` 为 ``[B, T, HV, 128]`` bfloat16。

    Note:
        **不需要梯度时（``no_grad`` 或所有输入都不 ``requires_grad``）本函数直接走
        :func:`~ascend_fla.ops.kda.chunk.chunk_kda_fwd`**：不产那九个检查点（在那种情况下是
        纯浪费，实测占训练步 21%），并且用前向那条更宽的闸。``o`` / ``final_state`` 逐位相同，
        因为两条路共用同一次 kernel 调用。

    Raises:
        ValueError: 任何定尺/dtype/设备约束不满足。绝不静默降级（AGENTS.md §7）。
    """
    from ascend_fla.platform import require_qualified, resolve_soc
    device = resolve_soc(device)
    require_qualified(device)

    # Choose before preparation; gate-parameter-only training must keep its graph.
    train_inputs = (q, k, v, g, beta, initial_state)
    if use_gate_in_kernel:
        train_inputs += (A_log, dt_bias)
    training = torch.is_grad_enabled() and any(
        isinstance(t, torch.Tensor) and t.requires_grad for t in train_inputs)
    options = dict(use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                   use_gate_in_kernel=use_gate_in_kernel,
                   use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel)
    if training:
        q, k, g, beta = _prepare_training_inputs(
            q, k, g, beta, A_log=A_log, dt_bias=dt_bias, **options,
            device=device, block_dim=block_dim, impl=impl)
    else:
        q, k, g, beta = _prepare_kernel_inputs(
            q, k, g, beta, A_log=A_log, dt_bias=dt_bias, **options,
            device=device, block_dim=block_dim, namespace="chunk", impl=impl)
    if check_domain:
        _check_input_domain(q, k, g, beta, **options)

    # 不需要梯度时绕开 autograd.Function：走纯前向那条路。
    # 两个实打实的好处，不是"优化"：
    #   ① 不产那九个检查点 —— 它们在 no_grad 下是纯浪费（实测占训练步 21%，bd=4）；
    #   ② 用**前向**那条更宽的闸（stable 下 155 而不是 105）。推理本来就不受反向精度约束，
    #      被反向的闸挡住是错的。
    # o 与 final_state 逐位相同 —— 两条路共用同一次 kernel 调用（见 chunk.py 的 _run_chain），
    # 所以这不是"近似等价的替换路径"（AGENTS.md §7 禁止的那种）。
    needs_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (q, k, v, g, beta, initial_state))
    if not needs_grad:
        return chunk_kda_fwd(
            q, k, v, g, beta, scale, initial_state, output_final_state,
            device=device, block_dim=block_dim, layout_device=layout_device,
            check_gate_range=check_gate_range, impl=impl,
        )
    return _ChunkKDA.apply(q, k, v, g, beta, scale, initial_state, output_final_state,
                           device, block_dim, layout_device, check_gate_range, impl)
