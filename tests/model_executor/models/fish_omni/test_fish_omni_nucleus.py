# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The semantic nucleus includes its crossing token; acoustic heads do not."""

import math

import pytest
import torch

from vllm_omni.model_executor.models.fish_omni.kernels import draw_top_k_top_p

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("keep_crossing,top_k,expected", [(False, 30, 0), (True, 30, 1), (True, 1, 0)])
def test_nucleus_boundary_and_top_k(keep_crossing, top_k, expected):
    logits = torch.full((1, 32), -80.0)
    logits[0, :3] = torch.tensor([math.log(p) for p in (0.55, 0.35, 0.10)])
    # Make rank 1 win whenever it is eligible. It crosses top_p=0.6.
    noise = torch.full((1, 32), 1e-20)
    noise[0, 1] = 0.9999
    actual = draw_top_k_top_p(
        logits,
        torch.tensor([1.0]),
        torch.tensor([top_k]),
        torch.tensor([0.6]),
        noise,
        32,
        keep_top_p_crossing_token=keep_crossing,
    )
    assert actual.item() == expected
