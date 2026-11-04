# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real Marlin INT8 V correctness and replay, independent of speech acceptance."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from tests.model_executor.models.fish_omni.test_fish_omni_attention_recovery import unpack
from vllm_omni.model_executor.models.fish_omni.attention_recovery import (
    Int8AttentionProjection,
    quantize_attention_int8,
)

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "H100"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Marlin"),
]


@pytest.mark.parametrize("shape", [(128, 256), (4096, 2048)])
@pytest.mark.parametrize("rows", [1, 17])
def test_real_int8_value_matches_independent_fp64_reference(shape, rows):
    generator = torch.Generator().manual_seed(73)
    weight = (torch.randn(shape, generator=generator) * 0.03).bfloat16()
    packed, scales = quantize_attention_int8(weight, torch.bfloat16)
    reference = unpack(packed, scales).double()
    value = Int8AttentionProjection(weight, torch.device("cuda"), torch.bfloat16)
    x = torch.randn(rows, shape[1], generator=generator).bfloat16().to("cuda")
    actual = value(x).float().cpu()
    expected = (x.cpu().double() @ reference.T).float()
    assert (actual - expected).norm() / expected.norm() < 0.006


def test_cuda_graph_replay_reads_new_input():
    generator = torch.Generator().manual_seed(74)
    weight = torch.randn(128, 256, generator=generator).bfloat16()
    value = Int8AttentionProjection(weight, torch.device("cuda"), torch.bfloat16)
    x = torch.randn(1, 256, generator=generator).bfloat16().to("cuda")
    value(x)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = value(x)
    x.mul_(0.5)
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(output, value(x), rtol=0, atol=0)
