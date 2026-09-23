"""Stable FP32 chunk composition, independent of the token recurrence oracle."""
from __future__ import annotations

import torch


def gdn2_chunk_reference(q, k, v, g, b, w, initial_state=None, *,
                        scale=None, use_qk_l2norm=True, qk_norm_eps=1e-6,
                        chunk_size=64):
    """Solve each chunk's triangular system without positive decay exponents.

    Pairwise scores are built a row at a time: memory is O(H*chunk_size*K),
    rather than allocating a [B,H,C,chunk_size,chunk_size,K] decay tensor.
    The return dtype follows q; state arithmetic and storage stay FP32.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    dtype = q.dtype
    q, k, v, g, b, w = (x.transpose(1, 2).float() for x in (q, k, v, g, b, w))
    batch, heads, time, width = q.shape
    if time == 0:
        raise ValueError("GDN-2 does not accept empty sequences")
    if use_qk_l2norm:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + qk_norm_eps)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + qk_norm_eps)
    q = q * (width**-0.5 if scale is None else scale)
    state = (torch.zeros(batch, heads, width, v.shape[-1], device=q.device)
             if initial_state is None else initial_state.float().clone())
    result = []
    for start in range(0, time, chunk_size):
        stop = min(start + chunk_size, time)
        qc, kc, vc, gc, bc, wc = (x[:, :, start:stop] for x in (q, k, v, g, b, w))
        prefix = gc.cumsum(-2)
        count = stop - start
        lower = torch.zeros(batch, heads, count, count, device=q.device)
        scores = torch.zeros_like(lower)
        bk = bc * kc
        for i in range(count):
            decayed = kc[:, :, :i+1] * (prefix[:, :, i:i+1] - prefix[:, :, :i+1]).exp()
            scores[:, :, i, :i+1] = (qc[:, :, i:i+1] * decayed).sum(-1)
            if i:
                lower[:, :, i, :i] = (bk[:, :, i:i+1] * decayed[:, :, :i]).sum(-1)
        system = lower + torch.eye(count, device=q.device)
        rhs = torch.cat((wc * vc, bk * prefix.exp()), dim=-1)
        solved = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
        u, wy = solved.split((vc.shape[-1], width), dim=-1)
        delta = u - wy @ state
        result.append((qc * prefix.exp()) @ state + scores @ delta)
        tail = kc * (prefix[:, :, -1:] - prefix).exp()
        state = prefix[:, :, -1].exp().unsqueeze(-1) * state + tail.transpose(-1, -2) @ delta
    return torch.cat(result, dim=2).transpose(1, 2).contiguous().to(dtype), state
