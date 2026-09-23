"""GDN-2 chunk prefill through five repository-owned Ascriptor CCE stages.

NPU tensors use the in-process bridge. CPU tensors require an explicit aclnn
or board launcher, allowing CPU-only Torch plus an external device harness.
There is no reference fallback and no backward implementation.
"""
from __future__ import annotations

import functools
import importlib.util
import math
from pathlib import Path
import sys

import torch

from .gdn2.fused_recurrent import ATTENTION_SCALE, QK_NORM_EPS


#: Public dtype -> derived unit. The BF16 unit takes BF16 q/k/v and writes BF16 o; everything
#: between stays FP32. Neither path converts dtype on the host (D-PM-35 / D-PM-37).
_UNITS = {torch.float32: ('gdn2_chunk_fwd', '_afla_gdn2_chunk_kernels'),
          torch.bfloat16: ('gdn2_chunk_fwd_bf16', '_afla_gdn2_chunk_bf16_kernels')}


@functools.lru_cache(maxsize=2)
def _pipeline(dtype=torch.float32):
    unit, name = _UNITS[dtype]
    path = Path(__file__).resolve().parents[2] / 'kernels/projects/a5' / unit / 'kernels'
    spec = importlib.util.spec_from_file_location(name, path / '__init__.py', submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    from importlib import import_module
    return import_module(name + '.pipeline')


@functools.lru_cache(maxsize=8)
def _compiled(block_dim, dtype=torch.float32):
    from ..runtime.compile import compile_kernel
    return tuple(compile_kernel(entry, device='a5', block_dim=block_dim, backend='cce')
                 for entry in _pipeline(dtype).entries())


def prepare(*, device='a5', block_dim=8, dtypes=(torch.float32, torch.bfloat16)):
    """Compile every chunk stage before CANN first resolves an operator.

    Both dtype paths are separate operator sets, so a process that will use both must compile
    both here: CANN resolves the vendor tree once, on the first execution.
    """
    _check_options(device, block_dim)
    for dtype in dtypes:
        _compiled(block_dim, dtype)


def _check_options(device, block_dim):
    if device != 'a5' or type(block_dim) is not int or block_dim not in (1, 2, 4, 8):
        raise ValueError('chunk requires device=a5 and block_dim in (1,2,4,8)')


def _validate(q, k, v, g, b, w, initial_state, *, scale, use_qk_l2norm,
              qk_norm_eps, device, block_dim, launcher):
    _check_options(device, block_dim)
    if launcher not in ('inprocess', 'aclnn', 'board'):
        raise ValueError('launcher must be inprocess, aclnn or board')
    if not use_qk_l2norm or not math.isclose(qk_norm_eps, QK_NORM_EPS, rel_tol=0., abs_tol=1e-12):
        raise ValueError('chunk CCE requires q/k normalization with eps=1e-6')
    if not math.isclose(scale, ATTENTION_SCALE, rel_tol=0., abs_tol=1e-12):
        raise ValueError('chunk CCE requires attention scale=128**-0.5')
    if q.ndim != 4:
        raise ValueError('q must be [B,T,H,128]')
    batch, time, heads, width = q.shape
    if batch != 1 or not 1 <= time <= 4096 or heads not in (1, 16) or width != 128:
        raise ValueError(f'chunk domain B=1,T=1..4096,H=1/16,K=V=128; got {tuple(q.shape)}')
    tensors = dict(q=q, k=k, v=v, g=g, erase_gate=b, w=w)
    for name, tensor in tensors.items():
        if tensor.shape != q.shape:
            raise ValueError(f'{name} must match q shape {tuple(q.shape)}')
        if name in ('g', 'erase_gate', 'w'):
            if tensor.dtype != torch.float32:
                raise ValueError(f'{name} must be float32')
        elif tensor.dtype != q.dtype or tensor.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError('q/k/v must share float32 or bfloat16 dtype')
    if initial_state is not None:
        if initial_state.shape != (batch, heads, 128, 128) or initial_state.dtype != torch.float32:
            raise ValueError('initial_state must be FP32 [B,H,128,128]')
        tensors['initial_state'] = initial_state
    expected_device = 'npu' if launcher == 'inprocess' else 'cpu'
    for name, tensor in tensors.items():
        if tensor.device != q.device or tensor.device.type != expected_device:
            raise ValueError(f'{launcher} requires all tensors on one {expected_device} device')
        if not tensor.is_contiguous():
            raise ValueError(f'{name} must be contiguous')
        if torch.is_grad_enabled() and tensor.requires_grad:
            raise RuntimeError('chunk CCE supports inference only; no backward is available')
    # CPU validation has no device synchronization. NPU numerical domain is a
    # caller precondition, matching the existing bridge's no-host-read hot path.
    if expected_device == 'cpu':
        if not all(bool(torch.isfinite(t).all()) for t in tensors.values()):
            raise ValueError('all inputs must be finite')
        if bool((g > 0).any()) or bool(((b < 0) | (b > 2)).any()) or bool(((w < 0) | (w > 1)).any()):
            raise ValueError('g<=0, b in [0,2] and w in [0,1] are required')


def chunk_gdn2(q, k, v, g, b, w, *, initial_state=None, output_final_state=False,
               scale=ATTENTION_SCALE, use_qk_l2norm=True, qk_norm_eps=QK_NORM_EPS,
               device='a5', block_dim=8, launcher='inprocess', board=None,
               out_dir=None, timeout=300):
    """Return token-major output and optional fresh K-major FP32 final state.

    CPU mode is explicit: launcher='aclnn' on a CANN host, or 'board' with an
    external board configuration. Its transfer/build/host costs are not NPU
    model performance. Numeric gate constraints are checked for CPU inputs;
    NPU callers must provide finite inputs in the documented activated domain.
    """
    _validate(q, k, v, g, b, w, initial_state, scale=scale,
              use_qk_l2norm=use_qk_l2norm, qk_norm_eps=qk_norm_eps,
              device=device, block_dim=block_dim, launcher=launcher)
    state = initial_state
    if state is None:
        # Allocation on the caller's device: building on the CPU and moving it is a host-side copy.
        state = torch.zeros(q.shape[0], q.shape[2], 128, 128, dtype=torch.float32, device=q.device)
    # q/k/v go to the kernel in the caller's dtype; the kernel widens and narrows internally.
    inputs = dict(q=q, k=k, v=v, g=g, erase_gate=b, w=w, initial_state=state)
    pipeline = _pipeline(q.dtype)
    if launcher == 'inprocess':
        compiled = dict(zip((entry.name for entry in pipeline.entries()), _compiled(block_dim, q.dtype)))
        def launch(entry, sources, outputs, scalars):
            # The compiler includes only scalar names consumed by its ABI.
            op = compiled[entry.name]
            op(sources, {name: scalars[name] for name in op.scalar_names}, outputs)
            return outputs
    else:
        from ascriptor.runtime import OpExec
        def launch(entry, sources, outputs, scalars):
            root = None if out_dir is None else Path(out_dir) / entry.name
            ex = OpExec(entry, launcher=launcher, device=device, backend='cce',
                        block_dim=block_dim, board=board, out_dir=root, timeout=timeout)
            result = ex(*(tuple(sources.values()) + tuple(outputs.values()) + tuple(scalars.values())))
            return dict(zip(outputs, (result,) if len(outputs) == 1 else result))
    checkpoints = pipeline.run(inputs, launch, retain_stages=False)
    # The kernel already wrote o in the caller's dtype: no cast on the way out.
    return checkpoints['o'], checkpoints['final_state'] if output_final_state else None
