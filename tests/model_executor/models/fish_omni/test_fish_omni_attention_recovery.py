# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for source-based attention recovery and physical INT8 packing."""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.linear import LinearMethodBase

from vllm_omni.model_executor.models.fish_omni.attention_recovery import (
    RecoveredAttentionMethod,
    install_attention_recovery,
    quantize_attention_int8,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def unpack(packed, scales):
    words = packed.long() & 0xFFFFFFFF
    codes = torch.stack([(words >> shift) & 255 for shift in (0, 8, 16, 24)], dim=1)
    codes = codes.reshape(packed.shape[0] * 4, packed.shape[1]).T - 128
    return (codes.float() * scales.T.float().repeat_interleave(128, dim=1)).to(scales.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_physical_int8_scale_and_signed_word_packing(dtype):
    generator = torch.Generator().manual_seed(91)
    weight = torch.randn(16, 256, generator=generator).to(dtype)
    weight[0].zero_()
    weight[1].zero_()
    weight[1, :4] = torch.tensor([-127.0, -1.0, 0.0, 127.0], dtype=dtype)
    packed, scales = quantize_attention_int8(weight, dtype)
    assert packed.dtype == torch.int32 and scales.dtype == dtype
    assert packed.shape == (64, 16) and scales.shape == (2, 16)
    decoded = unpack(packed, scales)
    assert torch.equal(decoded[0], weight[0])
    assert bool((packed < 0).any()), "high byte must survive signed int32 storage"
    assert (decoded.float() - weight.float()).norm() / weight.float().norm() < 0.01
    torch.testing.assert_close(decoded[1, :4], weight[1, :4], rtol=0, atol=0)


@pytest.mark.parametrize("weight", [torch.zeros(8, 126), torch.zeros(8), torch.full((8, 128), float("nan"))])
def test_rejects_invalid_parent_matrix(weight):
    with pytest.raises(ValueError, match="attention"):
        quantize_attention_int8(weight, torch.bfloat16)


class OriginalMethod(LinearMethodBase):
    def create_weights(self, *args, **kwargs):
        raise AssertionError("already loaded")

    def process_weights_after_loading(self, layer):
        layer.processed = True

    def apply(self, layer, x, bias=None):
        raise AssertionError("recovery must not execute the old INT4 projection")


@pytest.mark.parametrize("with_bias", [False, True])
def test_full_projection_recovery_uses_new_weights_and_preserves_bias(with_bias):
    generator = torch.Generator().manual_seed(23)
    x = torch.randn(3, 8, generator=generator)
    replacement = torch.nn.Linear(8, 12, bias=False)
    layer = SimpleNamespace(attention_recovery=replacement)
    bias = torch.arange(12, dtype=torch.float32) if with_bias else None
    method = RecoveredAttentionMethod(OriginalMethod())
    method.process_weights_after_loading(layer)
    assert layer.processed
    output = method.apply(layer, x, bias)
    expected = replacement(x) + (bias if with_bias else 0)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_default_does_not_touch_model_or_require_parent_checkpoint():
    install_attention_recovery(SimpleNamespace(config=SimpleNamespace()))


def test_recovery_source_survives_config_serialization():
    from vllm_omni.model_executor.models.fish_omni.configuration_fish_omni import FishOmniConfig

    config = FishOmniConfig(int4_attention_source="/models/original")
    restored = FishOmniConfig.from_dict(config.to_dict())
    assert restored.int4_attention_source == "/models/original"
    assert FishOmniConfig().int4_attention_source is None


@pytest.mark.parametrize("source", ["", " ", 12, False])
def test_invalid_recovery_source_is_rejected_by_config(source):
    from vllm_omni.model_executor.models.fish_omni.configuration_fish_omni import FishOmniConfig

    with pytest.raises(ValueError, match="int4_attention_source"):
        FishOmniConfig(int4_attention_source=source)


@pytest.mark.parametrize("outer_dtype", [torch.float16, torch.bfloat16])
def test_quantization_is_independent_of_loader_default_dtype(outer_dtype):
    generator = torch.Generator().manual_seed(29)
    weight = torch.randn(32, 512, generator=generator, dtype=torch.float32).bfloat16()
    expected = quantize_attention_int8(weight, torch.bfloat16)
    saved = torch.get_default_dtype()
    try:
        torch.set_default_dtype(outer_dtype)
        actual = quantize_attention_int8(weight, torch.bfloat16)
    finally:
        torch.set_default_dtype(saved)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
