# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Characterization tests for MiniCPM-o 4.5 Talker AR sampling."""

from __future__ import annotations

import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    _apply_repetition_penalty,
    _apply_top_k_top_p,
    _restore_weight_norm_weight,
)


def test_restore_weight_norm_matches_row_normalization() -> None:
    weight_g = torch.tensor([[2.0], [3.0]])
    weight_v = torch.tensor([[3.0, 4.0], [0.0, 5.0]])

    restored = _restore_weight_norm_weight(weight_g, weight_v)

    expected = torch.tensor([[1.2, 1.6], [0.0, 3.0]])
    torch.testing.assert_close(restored, expected)


def test_repetition_penalty_matches_upstream_windowed_frequency_rule() -> None:
    logits = torch.tensor([[4.0, -4.0, 2.0, 1.0]])
    history = torch.tensor([0, 0, 1, 2, 2, 2])

    penalized = _apply_repetition_penalty(
        logits,
        history,
        penalty=2.0,
        window_size=4,
    )

    # The last four ids are [1, 2, 2, 2]. Positive scores are divided by
    # penalty**frequency, while negative scores are multiplied.
    expected = torch.tensor([[4.0, -8.0, 0.25, 1.0]])
    torch.testing.assert_close(penalized, expected)


def test_top_k_top_p_keep_at_least_three_candidates() -> None:
    logits = torch.tensor([[8.0, 7.0, 6.0, 5.0, 4.0]])

    filtered = _apply_top_k_top_p(
        logits,
        top_k=2,
        top_p=0.01,
        min_tokens_to_keep=3,
    )

    assert torch.isfinite(filtered[0, :3]).all()
    assert torch.isneginf(filtered[0, 3:]).all()
