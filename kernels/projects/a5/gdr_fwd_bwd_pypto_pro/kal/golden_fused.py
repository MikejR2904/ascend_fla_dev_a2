#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""Per-step accumulative golden — all five steps (fused) of gated_delta_rule_fwd.

Self-contained, pure-torch fp32 CPU reference. Implements all five steps from
PRIMARY inputs and returns the scan step's (interchunk_scan_fusion) final output o.

the scan step (golden_steps s12..s16): loop-carried inter-chunk scan with persistent
state S[K,V] (init 0). Per chunk i:
    v_prime = w_i @ S
    v_new   = u_i - v_prime
    o_inter = q_de_i @ S
    o_i     = o_inter + A_i @ v_new
    S       = S * decay_exp_i[C-1] + (k_i * dfac_i)^T @ v_new
then slice [:T], transpose back to [B,T,H,V].

The returned o is numerically identical (fp32 round-off) to
gated_delta_rule_fwd_golden_cpu(q,k,v,g,beta). gen_cleanup.py substitutes
golden_cpu for this file downstream, so they MUST match.

Derived from gated_delta_rule_fwd_golden_cpu.py — CPU FP32 source of truth.
Returns o (fp32, layout [B,T,H,V]).
"""

import torch
import torch.nn.functional as F

_CHUNK_SIZE = 64


def _state_through_uw(q, k, v, g, beta):
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
    # the u/w step
    u = attn @ v_b
    w = attn @ kbeta_de
    return dict(u=u, w=w, A=A, q_de=q_de, decay_exp=decay_exp, dfac=dfac,
                k_f32=k_f32, T=T, C=C)


def gated_delta_rule_fwd_golden_fused(q, k, v, g, beta):
    """all five steps accumulative golden = final output o.

    Returns:
        o : [B, T, H, V]  fp32  (numerically identical to golden_cpu's o)
    """
    st = _state_through_uw(q, k, v, g, beta)
    u = st['u']; w = st['w']; A = st['A']; q_de = st['q_de']
    decay_exp = st['decay_exp']; dfac = st['dfac']; k_f32 = st['k_f32']
    T = st['T']; C = st['C']
    b, h, n, _, dv = u.shape
    dk = w.shape[-1]

    S = u.new_zeros(b, h, dk, dv)
    o = torch.zeros_like(u)
    for i in range(0, n):
        A_i = A[:, :, i]
        u_i = u[:, :, i]
        w_i = w[:, :, i]
        v_prime = w_i @ S
        v_new = u_i - v_prime
        o_inter = q_de[:, :, i] @ S
        o[:, :, i] = o_inter + A_i @ v_new
        S = S * decay_exp[:, :, i, -1, None, None] + (
            k_f32[:, :, i] * dfac[:, :, i]).transpose(-1, -2) @ v_new

    o = o.reshape(b, h, n * C, dv)
    o = o[:, :, :T]
    o = o.transpose(1, 2).contiguous()
    return o  # fp32


# ─────────────────────────────────────────────
# Independent oracle: VERBATIM transcription of gated_delta_rule_fwd_golden_cpu
# (_chunk + public sigmoid), producing the final o.
# ─────────────────────────────────────────────

def _oracle_o(q, k, v, g, beta, chunk_size=_CHUNK_SIZE):
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
    v = k_cumsum
    S = k.new_zeros(b, h, d_k, d_v)
    o = torch.zeros_like(v)
    up_strict = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=1)
    for i in range(0, n):
        q_i = q[:, :, i]; k_i = k[:, :, i]; v_i = v[:, :, i]
        A_i = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill(up_strict, 0)
        v_prime = k_cumdecay[:, :, i] @ S
        v_new = v_i - v_prime
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + A_i @ v_new
        S = S * decay[:, :, i, -1, None, None].exp() + (
            k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]
        ).transpose(-1, -2) @ v_new
    o = o.reshape(b, h, n * chunk_size, d_v)
    o = o[:, :, :T]
    o = o.transpose(1, 2).contiguous()
    return o


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
    print("gated_delta_rule_fwd_golden_fused (the scan step, final o) 验证")
    print("=" * 60)
    shapes = [(1, 128, 4, 64, 64), (1, 100, 4, 64, 64),
              (1, 192, 2, 128, 128), (1, 200, 3, 128, 96)]
    seeds = [42, 123, 456]
    all_pass = True
    global_worst = 0.0
    for (B, T, H, K, V) in shapes:
        for seed in seeds:
            q, k, v, g, beta = _mk_case(B, T, H, K, V, seed)
            o = gated_delta_rule_fwd_golden_fused(q, k, v, g, beta)
            o_ref = _oracle_o(q, k, v, g, beta)   # verbatim golden_cpu math
            ok = True
            if tuple(o.shape) != (B, T, H, V):
                print(f"  [FAIL] o shape {tuple(o.shape)} != {(B,T,H,V)}"); ok = False
            if o.dtype != torch.float32:
                print(f"  [FAIL] o dtype {o.dtype} != fp32"); ok = False
            if not torch.isfinite(o).all():
                print(f"  [FAIL] o not finite"); ok = False
            d = (o - o_ref).abs()
            max_abs = d.max().item()
            denom = o_ref.abs().max().clamp_min(1e-6).item()
            rel = max_abs / denom
            global_worst = max(global_worst, max_abs)
            # identical ops -> must be within fp32 round-off; relative-to-peak tight
            if not (rel < 1e-5 or max_abs < 1e-3):
                print(f"  [FAIL] o vs golden_cpu max_abs={max_abs:.3e} rel_peak={rel:.3e} "
                      f"shape={B,T,H,K,V} seed={seed}"); ok = False
            if not ok:
                all_pass = False
            print(f"  [{'PASS' if ok else 'FAIL'}] B{B}T{T}H{H}K{K}V{V} seed{seed} "
                  f"max_abs={max_abs:.3e} rel_peak={rel:.3e} peak={denom:.2e}")
    print("-" * 60)
    print(f"fused-o vs golden_cpu-o : global max_abs = {global_worst:.3e}")
    print("=" * 60)
    print("所有验证通过" if all_pass else "存在验证失败项")
    return all_pass


if __name__ == "__main__":
    ok = _validate()
    if not ok:
        exit(1)
