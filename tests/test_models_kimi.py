"""Tests for injecting the repository KDA layer into a Kimi-Linear model.

Everything here runs on CPU: :func:`cpu_reference` substitutes the torch recurrence for
the NPU op, so no device is required. The kernel's C=1 output-corruption gate lives inside
``chunk_kda`` and is therefore not reachable once the op is substituted; the routing test
instead asserts that :func:`prefill_then_decode` steers the single-chunk case onto the
exact decode path, which is how that gate is avoided on real hardware.
"""
from __future__ import annotations

import importlib
from unittest import mock

import pytest
import torch

from ascend_fla.layers.kda import KimiDeltaAttention
from ascend_fla.models.kimi_linear import (
    DECODE_MAX_TOKENS,
    _SOURCE_TO_REPO_NAME,
    cpu_reference,
    prefill_then_decode,
    transfer_kda_weights,
)

HIDDEN_SIZE = 128
NUM_HEADS = 1
NUM_V_HEADS = 2
STATE_HEAD_DIM = 128
KIMI_HIDDEN_SIZE = 2304
KIMI_NUM_HEADS = 32
_REPO_TO_SOURCE_NAME = {repo: source for source, repo in _SOURCE_TO_REPO_NAME.items()}


def _build_layer(**overrides) -> KimiDeltaAttention:
    config = dict(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        num_v_heads=NUM_V_HEADS,
        use_short_conv=True,
        block_dim=1,
    )
    config.update(overrides)
    return KimiDeltaAttention(**config)


class _RenamedSource(torch.nn.Module):
    """Carries a fixed ``state_dict`` under HuggingFace Kimi-Linear parameter names."""

    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self._state = state

    def state_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        return self._state


def _hf_style_source(origin: KimiDeltaAttention) -> tuple[_RenamedSource, dict[str, torch.Tensor]]:
    """Rename ``origin``'s parameters to the upstream split-projection names, dropping the
    output-gate bias that upstream Kimi-Linear does not carry."""
    renamed = {}
    for name, tensor in origin.state_dict().items():
        if name == "g_proj.1.bias":
            continue
        renamed[_REPO_TO_SOURCE_NAME.get(name, name)] = tensor
    return _RenamedSource(renamed), renamed


def _decode_whole_sequence(layer: KimiDeltaAttention, hidden_states: torch.Tensor) -> torch.Tensor:
    """Independent oracle: run every token through the decode recurrence in fixed batches,
    threading one cache. Uses no chunking, so it never shares code with the routing path."""
    cache: dict = {}
    outputs = []
    for start in range(0, hidden_states.shape[1], DECODE_MAX_TOKENS):
        step_out, _ = layer(
            hidden_states[:, start:start + DECODE_MAX_TOKENS], cache=cache, mode="fused_recurrent"
        )
        outputs.append(step_out)
    return torch.cat(outputs, dim=1)


def test_transfer_maps_names_and_zeros_absent_gate_bias():
    torch.manual_seed(0)
    origin = _build_layer()
    source, source_state = _hf_style_source(origin)
    target = _build_layer()

    transfer_kda_weights(source, target)

    target_state = target.state_dict()
    for source_name, tensor in source_state.items():
        target_name = _SOURCE_TO_REPO_NAME.get(source_name, source_name)
        torch.testing.assert_close(target_state[target_name], tensor)
    assert torch.count_nonzero(target_state["g_proj.1.bias"]) == 0


def test_transfer_rejects_unmapped_source_parameter():
    origin = _build_layer()
    source, source_state = _hf_style_source(origin)
    source_state["unexpected.weight"] = torch.zeros(1)
    with pytest.raises(KeyError):
        transfer_kda_weights(source, _build_layer())


def test_transfer_rejects_absent_gate_bias_when_not_zeroing():
    origin = _build_layer()
    source, _ = _hf_style_source(origin)
    with pytest.raises(KeyError):
        transfer_kda_weights(source, _build_layer(), zero_absent_gate_bias=False)


@pytest.mark.parametrize("total_tokens", [63, 64, 127, 128, 129])
def test_prefill_then_decode_matches_pure_recurrence(total_tokens):
    torch.manual_seed(total_tokens)
    layer = _build_layer()
    hidden_states = torch.randn(1, total_tokens, HIDDEN_SIZE)
    with cpu_reference(), torch.no_grad():
        routed = prefill_then_decode(layer, hidden_states, cache={})
        reference = _decode_whole_sequence(layer, hidden_states)
    torch.testing.assert_close(routed, reference, rtol=1e-4, atol=1e-4)


def test_decode_continues_after_prefill():
    torch.manual_seed(7)
    layer = _build_layer()
    prefill = torch.randn(1, 2 * DECODE_MAX_TOKENS * 4, HIDDEN_SIZE)  # two full chunks
    decode_steps = torch.randn(1, 5, HIDDEN_SIZE)
    full_sequence = torch.cat([prefill, decode_steps], dim=1)
    cache: dict = {}
    with cpu_reference(), torch.no_grad():
        prefill_then_decode(layer, prefill, cache=cache)
        stepped = torch.cat(
            [layer(decode_steps[:, i:i + 1], cache=cache, mode="fused_recurrent")[0] for i in range(5)],
            dim=1,
        )
        reference = _decode_whole_sequence(layer, full_sequence)[:, prefill.shape[1]:]
    torch.testing.assert_close(stepped, reference, rtol=1e-4, atol=1e-4)


def test_chunk_path_rejects_non_multiple_length():
    layer = _build_layer()
    with cpu_reference(), pytest.raises(ValueError):
        layer(torch.randn(1, 65, HIDDEN_SIZE), cache={}, mode="chunk")


def test_decode_path_rejects_overlong_input():
    layer = _build_layer()
    with cpu_reference(), torch.no_grad(), pytest.raises(ValueError):
        layer(torch.randn(1, DECODE_MAX_TOKENS + 1, HIDDEN_SIZE), cache={}, mode="fused_recurrent")


def test_decode_path_rejects_grad():
    layer = _build_layer()
    with cpu_reference(), pytest.raises(ValueError):
        layer(torch.randn(1, 1, HIDDEN_SIZE), cache={}, mode="fused_recurrent")


def test_cu_seqlens_rejected():
    layer = _build_layer()
    with cpu_reference(), torch.no_grad(), pytest.raises(ValueError):
        layer(torch.randn(1, 64, HIDDEN_SIZE), cu_seqlens=torch.tensor([0, 64]), mode="chunk")


def test_cache_and_initial_state_mutually_exclusive():
    layer = _build_layer()
    initial_state = torch.zeros(1, NUM_V_HEADS, STATE_HEAD_DIM, STATE_HEAD_DIM)
    with cpu_reference(), torch.no_grad(), pytest.raises(ValueError):
        layer(torch.randn(1, 64, HIDDEN_SIZE), initial_state=initial_state, cache={}, mode="chunk")


def _load_fla_naive():
    try:
        naive = importlib.import_module("fla.ops.kda.naive")
    except ImportError:
        return None
    for name in ("naive_chunk_kda", "naive_recurrent_kda", "chunk_kda_ref"):
        fn = getattr(naive, name, None)
        if fn is not None:
            return fn
    return None


def _fla_naive_kernel(fla_naive):
    """Wrap ``fla.ops.kda.naive`` to the op signature the layer calls, upcasting to fp32."""
    def kernel(q, k, v, g, beta, *_, initial_state=None, output_final_state=False, **__):
        return fla_naive(
            q.float(), k.float(), v.float(), g.float(), beta.float(),
            initial_state=initial_state, output_final_state=output_final_state,
        )
    return kernel


@pytest.mark.skipif(_load_fla_naive() is None, reason="fla.ops.kda.naive unavailable")
def test_injected_layer_matches_fla_naive_kernel():
    """Layer-level acceptance: the repository layer with its torch reference substitute must
    match the same layer driven by ``fla.ops.kda.naive`` to relative L2 <= 1e-5, at the real
    Kimi-Linear KDA shape (H = HV = 32, K = V = 128). The full HuggingFace model is not built
    because its ``modeling_kimi`` requires a newer transformers than is installed; per the
    task's sanctioned fallback the check is made at the layer against ``fla.ops.kda.naive``.
    """
    fla_naive = _load_fla_naive()
    torch.manual_seed(11)
    layer = KimiDeltaAttention(
        hidden_size=KIMI_HIDDEN_SIZE, num_heads=KIMI_NUM_HEADS, num_v_heads=KIMI_NUM_HEADS,
        head_dim=STATE_HEAD_DIM, use_short_conv=True, block_dim=1,
    )
    hidden_states = torch.randn(1, 128, KIMI_HIDDEN_SIZE)
    with torch.no_grad():
        with cpu_reference():
            reference_output, _ = layer(hidden_states, cache={}, mode="chunk")
        with mock.patch("ascend_fla.layers.kda.chunk_kda", _fla_naive_kernel(fla_naive)):
            fla_output, _ = layer(hidden_states, cache={}, mode="chunk")
    relative_l2 = (reference_output - fla_output).norm() / fla_output.norm()
    assert relative_l2 <= 1e-5, relative_l2.item()
