# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.qwen3_tts.runtime_config import (
    resolve_qwen3_tts_runtime_config,
)


def _config(**talker_overrides):
    predictor = SimpleNamespace(vocab_size=64, num_code_groups=4)
    talker = {
        "code_predictor_config": predictor,
        "vocab_size": 96,
        "num_code_groups": 4,
        "codec_pad_id": 64,
        "codec_bos_id": 65,
        "codec_eos_token_id": 66,
        "codec_nothink_id": 67,
        "codec_think_id": 68,
        "codec_think_bos_id": 69,
        "codec_think_eos_id": 70,
        "spk_id": {"custom": 80},
        "spk_is_dialect": {"custom": False},
    }
    talker.update(talker_overrides)
    return SimpleNamespace(
        tts_model_type="custom_voice",
        tts_pad_token_id=90,
        tts_bos_token_id=91,
        tts_eos_token_id=92,
        talker_config=SimpleNamespace(**talker),
    )


def test_resolves_codec_stop_not_text_eos():
    runtime = resolve_qwen3_tts_runtime_config(
        _config(codec_eos_token_id=73),
    )

    assert runtime.codec_stop_token_id == 73
    assert runtime.codec_stop_token_id != 92


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("codec_eos_token_id", None),
        ("codec_pad_id", "64"),
        ("num_code_groups", True),
    ],
)
def test_rejects_missing_none_and_type_conflicts(field, value):
    with pytest.raises(ValueError, match=field):
        resolve_qwen3_tts_runtime_config(_config(**{field: value}))


def test_rejects_code_group_conflict():
    with pytest.raises(ValueError, match="conflicting num_code_groups"):
        resolve_qwen3_tts_runtime_config(
            _config(num_code_groups=8),
        )


def test_custom_voice_requires_speaker_mapping():
    with pytest.raises(ValueError, match="spk_id"):
        resolve_qwen3_tts_runtime_config(_config(spk_id=None))
