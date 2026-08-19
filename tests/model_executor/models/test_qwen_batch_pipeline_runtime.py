# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniConfig,
)

from vllm_omni.model_executor.models.aura_omni.pipeline import (
    resolve_aura_omni_pipeline_with_qwen3_tts,
)
from vllm_omni.model_executor.models.qwen2_5_omni.pipeline import (
    resolve_qwen2_5_omni_pipeline,
)
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
    Qwen3TTSConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.pipeline import (
    resolve_qwen3_tts_pipeline,
)


def test_qwen2_5_pipeline_propagates_checkpoint_stop_and_runtime():
    config = Qwen2_5OmniConfig(
        vllm_omni_qwen2_5_speaker_token_ids={"custom": 81},
        vllm_omni_qwen2_5_default_speaker="custom",
        talker_config={
            "tts_codec_pad_token_id": 21,
            "tts_codec_start_token_id": 22,
            "tts_codec_end_token_id": 23,
            "tts_codec_mask_token_id": 24,
            "vocab_size": 100,
        }
    )

    pipeline = resolve_qwen2_5_omni_pipeline(config)
    constraints = pipeline.get_stage(1).sampling_constraints

    assert constraints["stop_token_ids"] == [23]
    assert constraints["extra_args"]["model_runtime"]["qwen2_5_omni"]["codec_pad_token_id"] == 21
    assert constraints["extra_args"]["model_runtime"]["qwen2_5_omni"]["speaker_token_ids"] == {"custom": 81}


def _qwen3_tts_config():
    return Qwen3TTSConfig(
        tts_model_type="base",
        talker_config={
            "vocab_size": 100,
            "num_code_groups": 4,
            "codec_pad_id": 64,
            "codec_bos_id": 65,
            "codec_eos_token_id": 66,
            "codec_nothink_id": 67,
            "codec_think_id": 68,
            "codec_think_bos_id": 69,
            "codec_think_eos_id": 70,
            "code_predictor_config": {
                "vocab_size": 64,
                "num_code_groups": 4,
            },
        },
    )


def test_qwen3_tts_pipeline_propagates_checkpoint_stop():
    constraints = resolve_qwen3_tts_pipeline(_qwen3_tts_config()).get_stage(0).sampling_constraints

    assert constraints["stop_token_ids"] == [66]
    assert constraints["extra_args"]["model_runtime"]["qwen3_tts"]["num_code_groups"] == 4


def test_aura_reuses_qwen3_tts_pipeline_resolver():
    constraints = resolve_aura_omni_pipeline_with_qwen3_tts(_qwen3_tts_config()).get_stage(2).sampling_constraints

    assert constraints["stop_token_ids"] == [66]


def test_pipeline_resolvers_ignore_wrong_config_type():
    assert resolve_qwen2_5_omni_pipeline(SimpleNamespace()) is None
    assert resolve_qwen3_tts_pipeline(SimpleNamespace()) is None
