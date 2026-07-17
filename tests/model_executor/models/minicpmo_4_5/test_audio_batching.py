# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audio preprocessing regressions for MiniCPM-o 4.5 batching."""

from __future__ import annotations

import torch

from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni_llm


def test_flatten_audio_feature_lens_keeps_one_scalar_per_chunk() -> None:
    feature_lens = [
        torch.tensor([[120, 80]]),
        torch.tensor([55]),
    ]

    assert minicpmo_4_5_omni_llm._flatten_audio_feature_lens(feature_lens) == [
        120,
        80,
        55,
    ]


def test_audio_field_config_groups_chunks_by_audio() -> None:
    config = minicpmo_4_5_omni_llm._minicpmo_field_config(
        {
            "audio_features": [
                torch.zeros(80, 120),
                torch.zeros(80, 80),
                torch.zeros(80, 55),
            ],
            "audio_feature_lens": [
                torch.tensor([120, 80]),
                torch.tensor([55]),
            ],
        }
    )

    assert config["audio_features"].field.slices == [
        slice(0, 2),
        slice(2, 3),
    ]
