"""Inject the repository's ``KimiDeltaAttention`` into a HuggingFace Kimi-Linear model.

The upstream ``modeling_kimi`` file is used as-is (``trust_remote_code``); only the KDA
mixer of each linear-attention layer is swapped for this repository's layer and its
weights are carried over. Upstream spec updates then cost zero maintenance here.

On a host without an NPU the swapped layer runs its projection/gate/norm pipeline with
the torch recurrence in :mod:`ascend_fla.reference.kda` substituted for the NPU op; see
:func:`cpu_reference`.
"""
from __future__ import annotations

import contextlib
from unittest import mock

import torch
from torch import nn

from ascend_fla.layers.kda import KimiDeltaAttention
from ascend_fla.ops.kda.chunk import L_PER_CHUNK as CHUNK_LENGTH
from ascend_fla.ops.kda.fused_recurrent import T_MAX as DECODE_MAX_TOKENS
from ascend_fla.reference.kda import kda_recurrent_ref

_SOURCE_TO_REPO_NAME = {
    "f_a_proj.weight": "f_proj.0.weight",
    "f_b_proj.weight": "f_proj.1.weight",
    "g_a_proj.weight": "g_proj.0.weight",
    "g_b_proj.weight": "g_proj.1.weight",
}
_DECLARED_TARGET_ONLY = ("g_proj.1.bias",)


def _reshape_to_target(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape == target.shape:
        return source
    if source.numel() != target.numel():
        raise ValueError(
            f"element count differs: source {tuple(source.shape)} vs target {tuple(target.shape)}"
        )
    return source.reshape(target.shape)


def transfer_kda_weights(
    source: nn.Module,
    target: KimiDeltaAttention,
    *,
    zero_absent_gate_bias: bool = True,
) -> None:
    """Copy ``source``'s KDA weights into ``target`` under an explicit name map.

    Every source parameter must land on exactly one target parameter and every target
    parameter must be filled, otherwise this raises. The single sanctioned exception is
    ``g_proj.1.bias``: upstream Kimi-Linear has no output-gate bias, so when the source
    lacks it and ``zero_absent_gate_bias`` is set it is zeroed to reproduce that math.
    """
    source_state = dict(source.state_dict())
    target_state = dict(target.state_dict())

    filled: set[str] = set()
    for source_name, tensor in source_state.items():
        target_name = _SOURCE_TO_REPO_NAME.get(source_name, source_name)
        if target_name not in target_state:
            raise KeyError(f"source parameter {source_name!r} has no target {target_name!r}")
        target_state[target_name].copy_(_reshape_to_target(tensor, target_state[target_name]))
        filled.add(target_name)

    unfilled = set(target_state) - filled
    absent_gate_bias = unfilled & set(_DECLARED_TARGET_ONLY)
    if zero_absent_gate_bias:
        for name in absent_gate_bias:
            target_state[name].zero_()
            filled.add(name)
    unfilled -= filled
    if unfilled:
        raise KeyError(f"target parameters left unfilled: {sorted(unfilled)}")


def build_repo_kda(reference: nn.Module, *, layer_idx: int, block_dim: int = 1) -> KimiDeltaAttention:
    """Construct a repository ``KimiDeltaAttention`` whose shapes match ``reference``."""
    hidden_size = reference.q_proj.in_features
    head_dim = reference.head_k_dim
    num_heads = reference.num_heads
    num_v_heads = getattr(reference, "num_v_heads", num_heads)
    conv_size = reference.q_conv1d.kernel_size if hasattr(reference, "q_conv1d") else 4
    return KimiDeltaAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        conv_size=conv_size,
        layer_idx=layer_idx,
        block_dim=block_dim,
    )


def _kda_layer_indices(model) -> list[int]:
    linear_attn = model.config.linear_attn_config
    return [i - 1 for i in linear_attn["kda_layers"]]


def _mixer_attribute(decoder_layer: nn.Module) -> str:
    for name in ("self_attn", "attn", "mixer", "linear_attn"):
        if hasattr(decoder_layer, name):
            return name
    raise AttributeError(f"no known KDA mixer attribute on {type(decoder_layer).__name__}")


def inject_kda_layers(model, *, block_dim: int = 1, zero_absent_gate_bias: bool = True):
    """Replace every linear-attention layer's KDA mixer with the repository layer in place."""
    decoder_layers = model.model.layers
    for layer_idx in _kda_layer_indices(model):
        decoder_layer = decoder_layers[layer_idx]
        attribute = _mixer_attribute(decoder_layer)
        source = getattr(decoder_layer, attribute)
        repo_layer = build_repo_kda(source, layer_idx=layer_idx, block_dim=block_dim).to(
            dtype=source.q_proj.weight.dtype, device=source.q_proj.weight.device
        )
        transfer_kda_weights(source, repo_layer, zero_absent_gate_bias=zero_absent_gate_bias)
        setattr(decoder_layer, attribute, repo_layer)
    return model


def _reference_core(q, k, v, g, beta, *_, initial_state=None, output_final_state=False, **__):
    return kda_recurrent_ref(
        q.float(), k.float(), v.float(), g.float(), beta.float(),
        initial_state=initial_state, output_final_state=output_final_state,
    )


@contextlib.contextmanager
def cpu_reference():
    """Run the repository KDA layer on CPU by substituting the torch recurrence for the op.

    Both mixer entry points are patched, so ``mode="chunk"`` and ``mode="fused_recurrent"``
    share the single-source-of-truth recurrence and cannot diverge on the host path.
    """
    with mock.patch("ascend_fla.layers.kda.chunk_kda", _reference_core), \
            mock.patch("ascend_fla.layers.kda.fused_recurrent_kda", _reference_core):
        yield


def _decode_in_batches(layer, hidden_states, cache):
    outputs = []
    for start in range(0, hidden_states.shape[1], DECODE_MAX_TOKENS):
        step_out, _ = layer(
            hidden_states[:, start:start + DECODE_MAX_TOKENS], cache=cache, mode="fused_recurrent"
        )
        outputs.append(step_out)
    return torch.cat(outputs, dim=1)


def prefill_then_decode(
    layer: KimiDeltaAttention,
    hidden_states: torch.Tensor,
    *,
    cache: dict | None = None,
) -> torch.Tensor:
    """Route a full prefill of ``T`` tokens explicitly, with no approximate substitution.

    ``T = CHUNK_LENGTH*n + r``. When the chunk prefix spans at least two chunks it runs
    through ``mode="chunk"`` and the trailing ``r`` tokens continue on the decode path in
    ``DECODE_MAX_TOKENS``-sized batches. The single-chunk case (``64 <= T < 128``) trips the
    ``c1-multihead-o-corrupt`` gate (``C == 1 and B*HV > block_dim``); rather than approximate,
    the whole sequence is routed through the exact decode path instead. Both branches share
    the same cache, so prefill hands off to per-token decode unchanged.
    """
    batch_size, total_tokens, _ = hidden_states.shape
    if cache is None:
        cache = {}

    chunk_count = total_tokens // CHUNK_LENGTH
    chunked_tokens = chunk_count * CHUNK_LENGTH
    c1_gate_trips = chunk_count == 1 and batch_size * layer.num_v_heads > layer.block_dim
    decode_from = 0 if c1_gate_trips else chunked_tokens

    outputs = []
    if decode_from:
        chunk_out, _ = layer(hidden_states[:, :decode_from], cache=cache, mode="chunk")
        outputs.append(chunk_out)
    if decode_from < total_tokens:
        outputs.append(_decode_in_batches(layer, hidden_states[:, decode_from:], cache))
    return torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
