#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""Per-step accumulative golden — the four chunk-prep steps (chunk_prep) of gated_delta_rule_fwd.

Self-contained, pure-torch fp32 CPU reference. Implements the four chunk-prep steps from
PRIMARY inputs and returns the u/w step's (uw_cube) outputs.

the u/w step (golden_steps s10):
    u = attn @ v_b       (k_cumsum, the new v)
    w = attn @ kbeta_de  (k_cumdecay)

Derived from gated_delta_rule_fwd_golden_cpu.py — CPU FP32 source of truth.
Returns (u, w).
"""

import torch
import torch.nn.functional as F

_CHUNK_SIZE = 64


def _state_through_wy(q, k, v, g, beta):
    C = _CHUNK_SIZE
    q = q.to(torch.float32); k = k.to(torch.float32); v = v.to(torch.float32)
    g = g.to(torch.float32); beta = torch.sigmoid(beta.to(torch.float32))
    scale = 1.0 / (q.shape[-1] ** 0.5)
    q = q.transpose(1, 2).contiguous(); k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous(); g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous()
    T = q.shape[-2]
    pad = (C - (T % C)) % C
    if pad > 0:
        q = F.pad(q, (0, 0, 0, pad)); k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad)); g = F.pad(g, (0, pad)); beta = F.pad(beta, (0, pad))
    b, h, Tp, dk = q.shape
    dv = v.shape[-1]
    n = Tp // C
    # the preprocess step
    q_s = (q * scale).reshape(b, h, n, C, dk)
    k_f32 = k.reshape(b, h, n, C, dk)
    v_b = (v * beta[..., None]).reshape(b, h, n, C, dv)
    k_beta = (k * beta[..., None]).reshape(b, h, n, C, dk)
    g_f32 = g.reshape(b, h, n, C)
    # the gram step
    decay = g_f32.cumsum(-1)
    kk_gram = k_beta @ k_f32.transpose(-1, -2)
    qk_gram = q_s @ k_f32.transpose(-1, -2)
    # the WY step
    decay_exp = decay.exp()
    q_de = q_s * decay_exp[..., None]
    kbeta_de = k_beta * decay_exp[..., None]
    dfac = (decay[..., -1:] - decay).exp()[..., None].expand(b, h, n, C, dk).contiguous()
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp()).tril()
    up_incl_diag = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=0)
    attn = -((kk_gram) * L_mask).masked_fill(up_incl_diag, 0)
    for i in range(1, C):
        attn[..., i, :i] = attn[..., i, :i].clone() + (
            attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(C, dtype=torch.float32)
    up_strict = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=1)
    A = ((qk_gram) * L_mask).masked_fill(up_strict, 0)
    return dict(q_s=q_s, k_f32=k_f32, v_b=v_b, k_beta=k_beta, g_f32=g_f32,
                decay=decay, kk_gram=kk_gram, qk_gram=qk_gram, decay_exp=decay_exp, q_de=q_de,
                kbeta_de=kbeta_de, dfac=dfac, L_mask=L_mask, attn=attn, A=A, T=T)


def gated_delta_rule_fwd_golden_chunk_prep(q, k, v, g, beta):
    """the four chunk-prep steps accumulative golden.

    Returns:
        (u, w) with shapes [B,H,N,C,V], [B,H,N,C,K]  (fp32)
    """
    st = _state_through_wy(q, k, v, g, beta)
    u = st['attn'] @ st['v_b']
    w = st['attn'] @ st['kbeta_de']
    return u, w


# ─────────────────────────────────────────────
# Independent oracle: verbatim golden_cpu._chunk math, capturing intermediates.
# ─────────────────────────────────────────────

def _oracle(q, k, v, g, beta, chunk_size=_CHUNK_SIZE):
    K = q.shape[-1]
    scale = 1.0 / (K ** 0.5)
    q = q.to(torch.float32); k = k.to(torch.float32); v = v.to(torch.float32)
    g = g.to(torch.float32); beta = torch.sigmoid(beta.to(torch.float32))
    BT = chunk_size
    q, k, v, beta, g = [x.transpose(1, 2).contiguous() for x in (q, k, v, beta, g)]
    T = q.shape[-2]
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len)); k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len)); beta = F.pad(beta, (0, pad_len)); g = F.pad(g, (0, pad_len))
    decay = g
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * scale
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    n = l // chunk_size
    q = q.reshape(b, h, n, chunk_size, d_k)
    k = k.reshape(b, h, n, chunk_size, d_k)
    v = v.reshape(b, h, n, chunk_size, d_v)
    k_beta = k_beta.reshape(b, h, n, chunk_size, d_k)
    decay = decay.reshape(b, h, n, chunk_size)
    decay = decay.cumsum(-1)
    decay_exp = decay.exp()[..., None]
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()
    up_incl_diag = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=0)
    attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(up_incl_diag, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (
            attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float32)
    k_cumsum = attn @ v
    k_cumdecay = attn @ (k_beta * decay_exp)
    return {'u': k_cumsum.clone(), 'w': k_cumdecay.clone()}


def _mk_case(B, T, H, K, V, seed):
    torch.manual_seed(seed)
    q = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    k = torch.randn((B, T, H, K), dtype=torch.bfloat16)
    v = torch.randn((B, T, H, V), dtype=torch.bfloat16)
    g = torch.empty((B, T, H), dtype=torch.float32).uniform_(-4.0, 0.0)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16)
    return q, k, v, g, beta


def _validate():
    print("=" * 60)
    print("gated_delta_rule_fwd_golden_chunk_prep (the u/w step) 验证")
    print("=" * 60)
    shapes = [(1, 128, 4, 64, 64), (1, 100, 4, 64, 64),
              (1, 192, 2, 128, 128), (1, 200, 3, 128, 96)]
    seeds = [42, 123, 456]
    names = ['u', 'w']
    all_pass = True
    for (B, T, H, K, V) in shapes:
        for seed in seeds:
            q, k, v, g, beta = _mk_case(B, T, H, K, V, seed)
            got = gated_delta_rule_fwd_golden_chunk_prep(q, k, v, g, beta)
            ref = _oracle(q, k, v, g, beta)
            worst = 0.0
            for nm, a in zip(names, got):
                b_ = ref[nm]
                if tuple(a.shape) != tuple(b_.shape):
                    print(f"  [FAIL] {nm} shape {tuple(a.shape)} != {tuple(b_.shape)}")
                    all_pass = False; continue
                if not torch.isfinite(a).all():
                    print(f"  [FAIL] {nm} not finite"); all_pass = False; continue
                d = (a - b_).abs().max().item()
                worst = max(worst, d)
                if not torch.allclose(a, b_, rtol=1e-4, atol=1e-4):
                    print(f"  [FAIL] {nm} max_abs={d:.3e} shape={B,T,H,K,V} seed={seed}")
                    all_pass = False
            print(f"  [{'PASS' if worst < 1e-3 else 'CHK '}] B{B}T{T}H{H}K{K}V{V} seed{seed} max_abs={worst:.2e}")
    print("=" * 60)
    print("所有验证通过" if all_pass else "存在验证失败项")
    return all_pass


if __name__ == "__main__":
    ok = _validate()
    if not ok:
        exit(1)
