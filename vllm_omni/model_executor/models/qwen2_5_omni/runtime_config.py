# SPDX-License-Identifier: Apache-2.0
"""Validated checkpoint-owned runtime contract for Qwen2.5-Omni."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniConfig,
)

from vllm_omni.config.stage_config import get_required_config_field

_MODEL = "qwen2_5_omni"
_NAMESPACE = "qwen2_5_omni"


@dataclass(frozen=True)
class Qwen2_5OmniSpeakerConfig:
    """Project-owned adapter for the official checkpoint's speaker tokens."""

    speaker_token_ids: dict[str, int]
    default_speaker: str


OFFICIAL_QWEN2_5_OMNI_SPEAKER_CONFIG = Qwen2_5OmniSpeakerConfig(
    speaker_token_ids={
        "m02": 151870,
        "Ethan": 151870,
        "f030": 151872,
        "Chelsie": 151872,
        "prefix_caching": 151870,
    },
    default_speaker="m02",
)


@dataclass(frozen=True)
class Qwen2_5OmniRuntimeConfig:
    codec_pad_token_id: int
    codec_start_token_id: int
    codec_stop_token_id: int
    codec_mask_token_id: int
    talker_vocab_size: int
    speaker_token_ids: dict[str, int]
    default_speaker: str

    def to_sampling_extra_args(self) -> dict[str, dict[str, Any]]:
        return {"model_runtime": {_NAMESPACE: asdict(self)}}


def _required(config: object, field: str) -> int:
    return get_required_config_field(config, field, expected_type=int, model=_MODEL)


def _resolve_speaker_config(config: object) -> Qwen2_5OmniSpeakerConfig:
    speaker_token_ids = getattr(config, "vllm_omni_qwen2_5_speaker_token_ids", None)
    default_speaker = getattr(config, "vllm_omni_qwen2_5_default_speaker", None)
    if speaker_token_ids is None and default_speaker is None:
        return OFFICIAL_QWEN2_5_OMNI_SPEAKER_CONFIG
    return Qwen2_5OmniSpeakerConfig(
        speaker_token_ids=get_required_config_field(
            config,
            "vllm_omni_qwen2_5_speaker_token_ids",
            expected_type=dict,
            model=_MODEL,
        ),
        default_speaker=get_required_config_field(
            config,
            "vllm_omni_qwen2_5_default_speaker",
            expected_type=str,
            model=_MODEL,
        ),
    )


def resolve_qwen2_5_omni_runtime_config(
    config: Qwen2_5OmniConfig,
) -> Qwen2_5OmniRuntimeConfig:
    vocab_size = _required(config, "talker_config.vocab_size")
    stop_token_id = _required(config, "talker_config.tts_codec_end_token_id")
    if not 0 <= stop_token_id < vocab_size:
        raise ValueError(
            f"Model {_MODEL!r} sampled stop token {stop_token_id} must be inside "
            f"Talker logits vocabulary [0, {vocab_size})"
        )
    speaker_config = _resolve_speaker_config(config)
    speaker_token_ids = speaker_config.speaker_token_ids
    if not speaker_token_ids or any(
        not isinstance(name, str)
        or not name
        or not isinstance(token_id, int)
        or isinstance(token_id, bool)
        for name, token_id in speaker_token_ids.items()
    ):
        raise ValueError(
            f"Model {_MODEL!r} requires config field "
            "'vllm_omni_qwen2_5_speaker_token_ids' to be a non-empty "
            "str-to-int mapping"
        )
    default_speaker = speaker_config.default_speaker
    if not default_speaker or default_speaker not in speaker_token_ids:
        raise ValueError(
            f"Model {_MODEL!r} requires config field "
            "'vllm_omni_qwen2_5_default_speaker' to name a key in "
            "'vllm_omni_qwen2_5_speaker_token_ids'"
        )
    return Qwen2_5OmniRuntimeConfig(
        codec_pad_token_id=_required(config, "talker_config.tts_codec_pad_token_id"),
        codec_start_token_id=_required(config, "talker_config.tts_codec_start_token_id"),
        codec_stop_token_id=stop_token_id,
        codec_mask_token_id=_required(config, "talker_config.tts_codec_mask_token_id"),
        talker_vocab_size=vocab_size,
        speaker_token_ids=dict(speaker_token_ids),
        default_speaker=default_speaker,
    )


def qwen2_5_runtime_from_sampling_params(sampling_params: Any) -> Mapping[str, Any]:
    extra_args = getattr(sampling_params, "extra_args", None)
    model_runtime = extra_args.get("model_runtime") if isinstance(extra_args, Mapping) else None
    runtime = model_runtime.get(_NAMESPACE) if isinstance(model_runtime, Mapping) else None
    if not isinstance(runtime, Mapping):
        raise ValueError("Qwen2.5-Omni requires sampling_params.extra_args['model_runtime']['qwen2_5_omni']")
    return runtime


def get_required_qwen2_5_runtime_int(runtime: Mapping[str, Any], field: str) -> int:
    value = runtime.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Qwen2.5-Omni requires integer runtime field {field!r}; got {value!r}")
    return value
