# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.covo_audio import (
    llm2code2wav_token_only as covo_token_only,
)
from vllm_omni.model_executor.stage_input_processors.mimo_audio import (
    llm2code2wav_token_only as mimo_token_only,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _params(namespace, values):
    return SimpleNamespace(extra_args={"model_runtime": {namespace: values}})


def test_covo_sync_processor_uses_downstream_sampling_params_without_output_metadata():
    source = SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=[9, 100, 101], multimodal_output=None)]
    )

    prompts = covo_token_only(
        [source],
        source_sampling_params=_params("covo_audio", {"audio_token_index": 100}),
    )

    assert prompts[0]["prompt_token_ids"] == [0, 0]


def test_mimo_sync_processor_uses_downstream_sampling_params_without_output_metadata():
    source = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                token_ids=[],
                multimodal_output={
                    "codes": {
                        "audio": torch.ones(2, 1, 8, 4, dtype=torch.long),
                    }
                },
            )
        ]
    )

    prompts = mimo_token_only(
        [source],
        source_sampling_params=_params(
            "mimo_audio",
            {
                "empty_token_id": 151667,
                "max_code2wav_tokens": 18192,
            },
        ),
    )

    assert len(prompts[0]["prompt_token_ids"]) == 2 * 32 + 2 * 4
