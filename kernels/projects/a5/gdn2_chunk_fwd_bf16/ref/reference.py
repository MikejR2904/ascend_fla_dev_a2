"""Generated inputs and an independent CPU implementation; no DSL imports."""
from __future__ import annotations

from typing import Any

import torch


HEAD_DIM = 128
VALUE_DIM = 128
QK_EPS = 1e-6
Q_SCALE = HEAD_DIM**-0.5


def make_inputs(case: dict[str, Any]) -> dict[str, torch.Tensor]:
    parameters = case["parameters"]
    batch = int(parameters["B"])
    time = int(parameters["T"])
    heads = int(parameters["H"])
    state_scale = float(parameters.get("state_scale", 0.1))
    gate_scale = float(parameters.get("gate_scale", 3.0))
    erase_scale = float(parameters.get("erase_scale", 1.0))
    generator = torch.Generator().manual_seed(int(case["seed"]))
    shape = (batch, time, heads, HEAD_DIM)

    # Same draw order and scales as the FP32 unit, then rounded: the BF16 path's public ABI is BF16
    # q/k/v, so the rounding belongs to input generation, not to a host conversion.
    q = (torch.randn(shape, generator=generator, dtype=torch.float32) * 0.5).bfloat16()
    k = (torch.randn(shape, generator=generator, dtype=torch.float32) * 0.5).bfloat16()
    v = (torch.randn(shape, generator=generator, dtype=torch.float32) * 0.25).bfloat16()
    g = -torch.rand(shape, generator=generator, dtype=torch.float32) * gate_scale
    erase_gate = torch.rand(shape, generator=generator, dtype=torch.float32) * erase_scale
    w = torch.rand(shape, generator=generator, dtype=torch.float32)
    if state_scale == 0.0:
        initial_state = torch.zeros(
            batch, heads, HEAD_DIM, VALUE_DIM, dtype=torch.float32
        )
    else:
        initial_state = (
            torch.randn(
                batch,
                heads,
                HEAD_DIM,
                VALUE_DIM,
                generator=generator,
                dtype=torch.float32,
            )
            * state_scale
        )
    return {
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": v.contiguous(),
        "g": g.contiguous(),
        "erase_gate": erase_gate.contiguous(),
        "w": w.contiguous(),
        "initial_state": initial_state.contiguous(),
    }


def validate_inputs(inputs: dict[str, torch.Tensor], case: dict[str, Any] | None = None) -> None:
    q = inputs["q"]
    if q.ndim != 4:
        raise ValueError(f"q must be [B,T,H,128], got {tuple(q.shape)}")
    batch, time, heads, width = q.shape
    expected = (batch, time, heads, HEAD_DIM)
    if batch != 1 or not 1 <= time <= 4096 or heads not in (1, 16) or width != HEAD_DIM:
        raise ValueError(f"outside declared B/T/H/D domain: {tuple(q.shape)}")
    # BF16 path: q/k/v are BF16; the gates, w and the state stay FP32.
    for name in ("k", "v", "g", "erase_gate", "w"):
        tensor = inputs[name]
        want = torch.bfloat16 if name in ("k", "v") else torch.float32
        if tuple(tensor.shape) != expected or tensor.dtype != want:
            raise ValueError(f"{name} must be {want} {expected}, got {tensor.dtype} {tuple(tensor.shape)}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    state = inputs["initial_state"]
    state_shape = (batch, heads, HEAD_DIM, VALUE_DIM)
    if tuple(state.shape) != state_shape or state.dtype != torch.float32:
        raise ValueError(
            f"initial_state must be float32 {state_shape}, got {state.dtype} {tuple(state.shape)}"
        )
    if q.dtype != torch.bfloat16 or not q.is_contiguous() or not state.is_contiguous():
        raise ValueError("q must be a contiguous bfloat16 tensor and initial_state a contiguous float32 one")
    if not all(bool(torch.isfinite(tensor).all()) for tensor in inputs.values()):
        raise ValueError("inputs must be finite")
    if bool((inputs["g"] > 0).any()):
        raise ValueError("activated g must be non-positive")
    if bool((inputs["erase_gate"] < 0).any()) or bool((inputs["erase_gate"] > 2).any()):
        raise ValueError("activated b must be in [0,2]")
    if bool((inputs["w"] < 0).any()) or bool((inputs["w"] > 1).any()):
        raise ValueError("activated w must be in [0,1]")
    if case is not None and int(case.get("block_dim", 8)) not in (1, 2, 4, 8):
        raise ValueError("block_dim must be 1, 2, 4 or 8")


def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    validate_inputs(inputs)
    # The reference widens the BF16 inputs once and runs the same FP32 recurrence as the FP32 unit;
    # o is rounded to BF16 at the end because that is what the kernel writes.
    q = inputs["q"].float()
    k = inputs["k"].float()
    v = inputs["v"].float()
    g = inputs["g"]
    erase_gate = inputs["erase_gate"]
    w = inputs["w"]
    state = inputs["initial_state"].clone()

    qn = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + QK_EPS) * Q_SCALE
    kn = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + QK_EPS)
    outputs = []
    for index in range(q.shape[1]):
        q_t = qn[:, index]
        k_t = kn[:, index]
        state = state * torch.exp(g[:, index]).unsqueeze(-1)
        erase = torch.matmul(
            (erase_gate[:, index] * k_t).unsqueeze(-2), state
        ).squeeze(-2)
        delta = w[:, index] * v[:, index] - erase
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append(torch.matmul(q_t.unsqueeze(-2), state).squeeze(-2))
    return {"o": torch.stack(outputs, dim=1).bfloat16(), "final_state": state}


def validate_reference(
    inputs: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    case: dict[str, Any] | None = None,
) -> None:
    del case
    batch, time, heads, _ = inputs["q"].shape
    expected = {
        "o": (batch, time, heads, VALUE_DIM),
        "final_state": (batch, heads, HEAD_DIM, VALUE_DIM),
    }
    if set(outputs) != set(expected):
        raise ValueError(f"reference outputs must be {sorted(expected)}, got {sorted(outputs)}")
    want = {"o": torch.bfloat16, "final_state": torch.float32}
    for name, shape in expected.items():
        tensor = outputs[name]
        if tuple(tensor.shape) != shape or tensor.dtype != want[name]:
            raise ValueError(f"{name} must be {want[name]} {shape}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains non-finite values")
