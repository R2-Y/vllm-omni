# SPDX-License-Identifier: Apache-2.0

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.omni_base import OmniBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_custom_sampling_params_preserve_internal_model_runtime_defaults():
    omni = object.__new__(OmniBase)
    omni.default_sampling_params_list = [
        SamplingParams(
            temperature=0.9,
            extra_args={
                "model_runtime": {
                    "mimo_audio": {
                        "empty_token_id": 151667,
                        "max_code2wav_tokens": 18192,
                    }
                },
                "default_only": "kept",
            },
        )
    ]
    omni.num_stages = 1
    user = SamplingParams(
        temperature=0.2,
        extra_args={
            "model_runtime": {
                "mimo_audio": {
                    "empty_token_id": -1,
                }
            },
            "user_option": "kept",
        },
    )

    resolved = omni.resolve_sampling_params_list([user])

    assert resolved[0] is not user
    assert resolved[0].temperature == 0.2
    assert resolved[0].extra_args["default_only"] == "kept"
    assert resolved[0].extra_args["user_option"] == "kept"
    assert resolved[0].extra_args["model_runtime"]["mimo_audio"] == {
        "empty_token_id": 151667,
        "max_code2wav_tokens": 18192,
    }
