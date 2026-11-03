# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Codec checkpoint precision must not depend on the surrounding AR dtype."""

import pytest
import torch
from torch import nn
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.model_executor.models.fish_speech import dac_encoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _SmallCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2, bias=False)
        self.decoder = nn.Linear(2, 2, bias=False)


@pytest.fixture
def codec_checkpoint(tmp_path, monkeypatch):
    (tmp_path / "codec.pth").touch()
    weights = torch.tensor([[1.0013, -0.9971], [0.12345, -0.23456]], dtype=torch.float32)
    monkeypatch.setattr(dac_encoder, "_codec_cache", {})
    monkeypatch.setattr(dac_encoder, "build_dac_codec", _SmallCodec)
    monkeypatch.setattr(dac_encoder.torch, "load", lambda *args, **kwargs: {"encoder.weight": weights})
    return str(tmp_path), weights


@pytest.mark.parametrize("ar_dtype", [torch.bfloat16, torch.float16])
def test_reference_encoder_preserves_fp32_checkpoint(codec_checkpoint, ar_dtype):
    model_path, expected = codec_checkpoint
    with set_default_torch_dtype(ar_dtype):
        codec = dac_encoder._load_dac_codec(model_path, device="cpu", dtype=torch.float32)
        assert torch.get_default_dtype() == ar_dtype
    assert codec.encoder.weight.dtype == torch.float32
    torch.testing.assert_close(codec.encoder.weight, expected, rtol=0, atol=0)


def test_requested_runtime_dtype_rounds_checkpoint_only_once(codec_checkpoint):
    model_path, expected = codec_checkpoint
    with set_default_torch_dtype(torch.bfloat16):
        codec = dac_encoder._load_dac_codec(model_path, device="cpu", dtype=torch.float16)
        assert torch.get_default_dtype() == torch.bfloat16
    assert codec.encoder.weight.dtype == torch.float16
    torch.testing.assert_close(codec.encoder.weight, expected.half(), rtol=0, atol=0)


def test_codec_construction_error_restores_default_dtype(codec_checkpoint, monkeypatch):
    model_path, _ = codec_checkpoint

    def fail():
        raise RuntimeError("codec construction failed")

    monkeypatch.setattr(dac_encoder, "build_dac_codec", fail)
    with set_default_torch_dtype(torch.bfloat16):
        with pytest.raises(RuntimeError, match="codec construction failed"):
            dac_encoder._load_dac_codec(model_path, device="cpu")
        assert torch.get_default_dtype() == torch.bfloat16
