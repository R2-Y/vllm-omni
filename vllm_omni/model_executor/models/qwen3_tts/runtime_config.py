# SPDX-License-Identifier: Apache-2.0
"""Validated checkpoint-owned runtime contract for Qwen3-TTS."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from vllm_omni.config.stage_config import get_required_config_field

from .configuration_qwen3_tts import Qwen3TTSConfig

_MODEL = "qwen3_tts"


@dataclass(frozen=True)
class Qwen3TTSRuntimeConfig:
    codec_pad_token_id: int
    codec_bos_token_id: int
    codec_stop_token_id: int
    codec_nothink_token_id: int
    codec_think_token_id: int
    codec_think_bos_token_id: int
    codec_think_eos_token_id: int
    codebook_size: int
    talker_vocab_size: int
    num_code_groups: int
    tts_pad_token_id: int
    tts_bos_token_id: int
    tts_eos_token_id: int

    def to_sampling_extra_args(self) -> dict[str, dict[str, Any]]:
        return {"model_runtime": {"qwen3_tts": asdict(self)}}


def qwen3_tts_runtime_from_sampling_params(sampling_params: Any) -> Mapping[str, Any]:
    extra_args = getattr(sampling_params, "extra_args", None)
    model_runtime = extra_args.get("model_runtime") if isinstance(extra_args, Mapping) else None
    runtime = model_runtime.get("qwen3_tts") if isinstance(model_runtime, Mapping) else None
    if not isinstance(runtime, Mapping):
        raise ValueError("Qwen3-TTS requires sampling_params.extra_args['model_runtime']['qwen3_tts']")
    return runtime


def qwen3_tts_runtime_from_meta(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    model_runtime = meta.get("model_runtime")
    runtime = model_runtime.get("qwen3_tts") if isinstance(model_runtime, Mapping) else None
    if not isinstance(runtime, Mapping):
        raise ValueError("Qwen3-TTS payload requires meta['model_runtime']['qwen3_tts']")
    return runtime


def get_required_qwen3_tts_runtime_int(runtime: Mapping[str, Any], field: str) -> int:
    value = runtime.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Qwen3-TTS requires integer runtime field {field!r}; got {value!r}")
    return value


def _required(config: object, field: str) -> int:
    return get_required_config_field(config, field, expected_type=int, model=_MODEL)


def resolve_qwen3_tts_runtime_config(config: Qwen3TTSConfig) -> Qwen3TTSRuntimeConfig:
    codebook_size = _required(config, "talker_config.code_predictor_config.vocab_size")
    predictor_groups = _required(config, "talker_config.code_predictor_config.num_code_groups")
    num_code_groups = _required(config, "talker_config.num_code_groups")
    if predictor_groups != num_code_groups:
        raise ValueError(
            f"Model {_MODEL!r} has conflicting num_code_groups: "
            f"talker_config.num_code_groups={num_code_groups}, "
            f"talker_config.code_predictor_config.num_code_groups={predictor_groups}"
        )

    talker_vocab_size = _required(config, "talker_config.vocab_size")
    codec_stop_token_id = _required(config, "talker_config.codec_eos_token_id")
    if not codebook_size <= codec_stop_token_id < talker_vocab_size:
        raise ValueError(
            f"Model {_MODEL!r} sampled codec stop token {codec_stop_token_id} must be "
            f"outside codebook [0, {codebook_size}) and inside Talker logits "
            f"vocabulary [0, {talker_vocab_size})"
        )

    if getattr(config, "tts_model_type", None) == "custom_voice":
        speaker_ids = get_required_config_field(
            config,
            "talker_config.spk_id",
            expected_type=dict,
            model=_MODEL,
        )
        if not speaker_ids or any(
            not isinstance(name, str) or not isinstance(token_id, int) or isinstance(token_id, bool)
            for name, token_id in speaker_ids.items()
        ):
            raise ValueError(f"Model {_MODEL!r} requires talker_config.spk_id to be a non-empty str-to-int mapping")

    return Qwen3TTSRuntimeConfig(
        codec_pad_token_id=_required(config, "talker_config.codec_pad_id"),
        codec_bos_token_id=_required(config, "talker_config.codec_bos_id"),
        codec_stop_token_id=codec_stop_token_id,
        codec_nothink_token_id=_required(config, "talker_config.codec_nothink_id"),
        codec_think_token_id=_required(config, "talker_config.codec_think_id"),
        codec_think_bos_token_id=_required(config, "talker_config.codec_think_bos_id"),
        codec_think_eos_token_id=_required(config, "talker_config.codec_think_eos_id"),
        codebook_size=codebook_size,
        talker_vocab_size=talker_vocab_size,
        num_code_groups=num_code_groups,
        tts_pad_token_id=_required(config, "tts_pad_token_id"),
        tts_bos_token_id=_required(config, "tts_bos_token_id"),
        tts_eos_token_id=_required(config, "tts_eos_token_id"),
    )
