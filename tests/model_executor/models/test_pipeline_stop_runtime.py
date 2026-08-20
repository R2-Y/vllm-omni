# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.cosyvoice3.pipeline import resolve_cosyvoice3_pipeline
from vllm_omni.model_executor.models.covo_audio.config_covo_audio import CovoAudioConfig
from vllm_omni.model_executor.models.covo_audio.pipeline import resolve_covo_audio_pipeline
from vllm_omni.model_executor.models.fish_speech.configuration_fish_speech import FishSpeechConfig
from vllm_omni.model_executor.models.fish_speech.pipeline import resolve_fish_speech_pipeline
from vllm_omni.model_executor.models.glm_tts.pipeline import resolve_glm_tts_pipeline
from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import HiggsAudioV2Config
from vllm_omni.model_executor.models.higgs_audio_v2.pipeline import resolve_higgs_audio_v2_pipeline
from vllm_omni.model_executor.models.higgs_audio_v3.pipeline import resolve_higgs_audio_v3_pipeline
from vllm_omni.model_executor.models.mimo_audio.config_mimo_audio import MiMoAudioConfig
from vllm_omni.model_executor.models.mimo_audio.pipeline import resolve_mimo_audio_pipeline
from vllm_omni.model_executor.models.minimax_music3.pipeline import MINIMAX_MUSIC3_PIPELINE
from vllm_omni.model_executor.models.minimax_music3.prompt import SPECIAL_TOKEN_IDS, validate_tokenizer_ids
from vllm_omni.model_executor.models.moss_tts.configuration_moss_tts import MossTTSLocalConfig
from vllm_omni.model_executor.models.moss_tts.pipeline import resolve_moss_tts_local_pipeline
from vllm_omni.model_executor.models.moss_tts_nano.pipeline import resolve_moss_tts_nano_pipeline
from vllm_omni.model_executor.models.moss_tts_nano.runtime_config import MossTTSNanoConfig
from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config
from vllm_omni.transformers_utils.configs.glm_tts import GLMTTSConfig
from vllm_omni.transformers_utils.configs.higgs_audio_v3 import HiggsAudioV3Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _stop_ids(pipeline) -> list[int]:
    return pipeline.get_stage(0).sampling_constraints["stop_token_ids"]


def _runtime(pipeline, namespace: str) -> dict[str, int]:
    return pipeline.get_stage(0).sampling_constraints["extra_args"]["model_runtime"][namespace]


def test_cosyvoice3_nondefault_stop_propagates_to_pipeline_and_model_config() -> None:
    config = CosyVoice3Config(speech_token_size=7000, speech_stop_token_id=7001)

    assert _stop_ids(resolve_cosyvoice3_pipeline(config)) == [7001]
    assert config.llm["eos_token_id"] == 7001


def test_cosyvoice3_inconsistent_model_stop_fails() -> None:
    config = CosyVoice3Config()
    config.llm["eos_token_id"] += 1

    with pytest.raises(ValueError, match="inconsistent speech stop fields"):
        resolve_cosyvoice3_pipeline(config)


def test_glm_tts_tokenizer_stop_propagates_and_is_required() -> None:
    config = GLMTTSConfig(eoa_token_id=60000)

    assert _stop_ids(resolve_glm_tts_pipeline(config)) == [60000]

    config.eoa_token_id = None
    with pytest.raises(ValueError, match="eoa_token_id"):
        resolve_glm_tts_pipeline(config)


@pytest.mark.parametrize(
    ("resolver", "config", "expected"),
    [
        (
            resolve_covo_audio_pipeline,
            CovoAudioConfig(audio_token_index=180010, eos_token_id=170001, vocab_size=200000),
            [170001],
        ),
        (resolve_fish_speech_pipeline, FishSpeechConfig(im_end_token_id=170002), [170002]),
        (
            resolve_higgs_audio_v2_pipeline,
            HiggsAudioV2Config(eos_token_id=170003, audio_eos_token_id=170004),
            [170003, 170004],
        ),
        (
            resolve_higgs_audio_v3_pipeline,
            HiggsAudioV3Config(eos_token_id=170005, audio_end_token_id=170006),
            [170005, 170006],
        ),
        (
            resolve_mimo_audio_pipeline,
            MiMoAudioConfig(
                no_interleave_next_token_id=170007,
                im_end_token_id=170008,
                vocab_size=200000,
            ),
            [170007, 170008],
        ),
        (
            resolve_moss_tts_local_pipeline,
            MossTTSLocalConfig(im_end_token_id=170009),
            [170009],
        ),
        (
            resolve_moss_tts_nano_pipeline,
            MossTTSNanoConfig(
                eos_token_id=9,
                streaming_continue_token_id=8,
                vocab_size=32,
                hidden_size=16,
            ),
            [9],
        ),
    ],
)
def test_checkpoint_stop_ids_propagate(resolver, config, expected) -> None:
    assert _stop_ids(resolver(config)) == expected


def test_covo_runtime_propagates_and_invalid_boundary_fails() -> None:
    config = CovoAudioConfig(audio_token_index=180010, eos_token_id=170001, vocab_size=200000)
    assert _runtime(resolve_covo_audio_pipeline(config), "covo_audio") == {
        "audio_token_index": 180010,
        "eos_token_id": 170001,
        "vocab_size": 200000,
    }

    with pytest.raises(ValueError, match="below audio_token_index"):
        resolve_covo_audio_pipeline(CovoAudioConfig(audio_token_index=100, eos_token_id=100))


def test_mimo_runtime_propagates_and_duplicate_tokens_fail() -> None:
    config = MiMoAudioConfig(
        empty_token_id=170000,
        span_codec_start_token_id=170001,
        span_codec_end_token_id=170002,
        no_interleave_next_token_id=170003,
        speech_start_token_id=170004,
        speech_end_token_id=170005,
        endoftext_token_id=170006,
        im_end_token_id=170007,
        vocab_size=200000,
    )
    pipeline = resolve_mimo_audio_pipeline(config)
    assert _runtime(pipeline, "mimo_audio")["empty_token_id"] == 170000

    config.im_end_token_id = config.empty_token_id
    with pytest.raises(ValueError, match="distinct special token IDs"):
        resolve_mimo_audio_pipeline(config)


class _MiniMaxTokenizer:
    def convert_tokens_to_ids(self, token: str) -> int:
        return SPECIAL_TOKEN_IDS[token]


def test_minimax_pipeline_and_tokenizer_share_one_contract() -> None:
    validate_tokenizer_ids(_MiniMaxTokenizer())
    assert _stop_ids(MINIMAX_MUSIC3_PIPELINE) == [SPECIAL_TOKEN_IDS["<|audio_end|>"]]
