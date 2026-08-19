# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniConfig,
)

from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
)
from vllm_omni.model_executor.models.qwen2_5_omni.runtime_config import (
    resolve_qwen2_5_omni_runtime_config,
)

_DEFAULT_SPEAKER_TOKEN_IDS = object()


def _config(
    *,
    speaker_token_ids=_DEFAULT_SPEAKER_TOKEN_IDS,
    default_speaker="voice_a",
    **talker_overrides,
):
    if speaker_token_ids is _DEFAULT_SPEAKER_TOKEN_IDS:
        speaker_token_ids = {"voice_a": 151, "voice_b": 152}
    talker = {
        "tts_codec_pad_token_id": 101,
        "tts_codec_start_token_id": 102,
        "tts_codec_end_token_id": 103,
        "tts_codec_mask_token_id": 104,
        "vocab_size": 200,
    }
    talker.update(talker_overrides)
    return SimpleNamespace(
        talker_config=SimpleNamespace(**talker),
        vllm_omni_qwen2_5_speaker_token_ids=speaker_token_ids,
        vllm_omni_qwen2_5_default_speaker=default_speaker,
    )


def test_resolves_custom_checkpoint_values():
    runtime = resolve_qwen2_5_omni_runtime_config(_config(tts_codec_end_token_id=177))

    assert runtime.codec_stop_token_id == 177
    assert runtime.codec_pad_token_id == 101
    assert runtime.speaker_token_ids == {"voice_a": 151, "voice_b": 152}
    assert runtime.default_speaker == "voice_a"
    assert runtime.to_sampling_extra_args()["model_runtime"]["qwen2_5_omni"]["speaker_token_ids"] == {
        "voice_a": 151,
        "voice_b": 152,
    }


def test_standard_transformers_config_uses_bundled_speaker_schema_defaults():
    runtime = resolve_qwen2_5_omni_runtime_config(Qwen2_5OmniConfig())

    assert runtime.speaker_token_ids == {
        "m02": 151870,
        "Ethan": 151870,
        "f030": 151872,
        "Chelsie": 151872,
        "prefix_caching": 151870,
    }
    assert runtime.default_speaker == "m02"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tts_codec_end_token_id", None),
        ("tts_codec_start_token_id", "102"),
        ("tts_codec_pad_token_id", True),
    ],
)
def test_rejects_missing_none_and_type_conflicts(field, value):
    with pytest.raises(ValueError, match=field):
        resolve_qwen2_5_omni_runtime_config(_config(**{field: value}))


def test_rejects_stop_outside_talker_vocabulary():
    with pytest.raises(ValueError, match="inside Talker logits vocabulary"):
        resolve_qwen2_5_omni_runtime_config(_config(tts_codec_end_token_id=200))


def test_rejects_missing_speaker_mapping():
    config = _config()
    del config.vllm_omni_qwen2_5_speaker_token_ids

    with pytest.raises(ValueError, match="vllm_omni_qwen2_5_speaker_token_ids.*<missing>"):
        resolve_qwen2_5_omni_runtime_config(config)


@pytest.mark.parametrize(
    "speaker_token_ids",
    [
        None,
        {},
        {"voice": "151"},
        {"voice": True},
        {1: 151},
    ],
)
def test_rejects_invalid_speaker_mapping(speaker_token_ids):
    with pytest.raises(ValueError, match="vllm_omni_qwen2_5_speaker_token_ids"):
        resolve_qwen2_5_omni_runtime_config(
            _config(speaker_token_ids=speaker_token_ids),
        )


def test_rejects_default_speaker_outside_mapping():
    with pytest.raises(ValueError, match="vllm_omni_qwen2_5_default_speaker"):
        resolve_qwen2_5_omni_runtime_config(
            _config(default_speaker="unknown"),
        )


def test_model_uses_runtime_speaker_mapping_and_default():
    model = Qwen2_5OmniForConditionalGeneration.__new__(
        Qwen2_5OmniForConditionalGeneration,
    )
    model.tts_text_spk_token_ids = {"voice_a": 151, "voice_b": 152}
    model.default_tts_text_spk_type = "voice_b"

    assert model._get_text_spk_token_id("voice_a") == 151
    assert model._get_text_spk_token_id("unknown") == 152
