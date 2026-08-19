# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validated runtime contract for Qwen3-Omni checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from transformers import Qwen3OmniMoeConfig

from vllm_omni.config.stage_config import get_required_config_field

_MODEL_NAME = "qwen3_omni_moe"
_MODEL_RUNTIME_KEY = "model_runtime"
_RUNTIME_NAMESPACE = "qwen3_omni"


@dataclass(frozen=True)
class Qwen3OmniRuntimeConfig:
    im_start_token_id: int
    system_token_id: int
    user_token_id: int
    assistant_token_id: int
    tts_pad_token_id: int
    tts_bos_token_id: int
    tts_eos_token_id: int
    thinker_audio_token_id: int
    thinker_image_token_id: int
    thinker_video_token_id: int
    accept_hidden_layer: int
    thinker_hidden_size: int
    codebook_size: int
    num_quantizers: int
    codec_pad_token_id: int
    codec_bos_token_id: int
    codec_nothink_token_id: int
    codec_think_bos_token_id: int
    codec_think_eos_token_id: int
    sampled_stop_token_id: int
    talker_vocab_size: int
    speaker_ids: dict[str, int]

    def to_sampling_extra_args(self) -> dict[str, dict[str, Any]]:
        return {_MODEL_RUNTIME_KEY: {_RUNTIME_NAMESPACE: asdict(self)}}

    def to_payload_meta(self) -> dict[str, Any]:
        return {_MODEL_RUNTIME_KEY: {_RUNTIME_NAMESPACE: asdict(self)}}


def _runtime_namespace(
    container: Mapping[str, Any],
    *,
    location: str,
) -> Mapping[str, Any]:
    model_runtime = container.get(_MODEL_RUNTIME_KEY)
    runtime = model_runtime.get(_RUNTIME_NAMESPACE) if isinstance(model_runtime, Mapping) else None
    if not isinstance(runtime, Mapping):
        raise ValueError(
            f"Qwen3-Omni requires {location}[{_MODEL_RUNTIME_KEY!r}][{_RUNTIME_NAMESPACE!r}] to be a mapping"
        )
    return runtime


def qwen3_omni_runtime_from_sampling_params(
    sampling_params: Any,
) -> Mapping[str, Any]:
    extra_args = getattr(sampling_params, "extra_args", None)
    if not isinstance(extra_args, Mapping):
        raise ValueError("Qwen3-Omni requires sampling_params.extra_args to be a mapping")
    return _runtime_namespace(
        extra_args,
        location="sampling_params.extra_args",
    )


def qwen3_omni_runtime_from_payload(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        raise ValueError("Qwen3-Omni handoff payload requires meta to be a mapping")
    return _runtime_namespace(meta, location="payload.meta")


def qwen3_omni_runtime_from_flat_pooling_output(
    pooling_output: Mapping[str, Any],
) -> Mapping[str, Any]:
    return _runtime_namespace(
        {_MODEL_RUNTIME_KEY: pooling_output.get("meta.model_runtime")},
        location="pooling_output.meta",
    )


def get_required_qwen3_omni_runtime_int(
    runtime: Mapping[str, Any],
    field: str,
) -> int:
    value = runtime.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Qwen3-Omni requires integer runtime field {field!r}; got {value!r}")
    return value


def _required(config: object, field_path: str) -> int:
    return get_required_config_field(
        config,
        field_path,
        expected_type=int,
        model=_MODEL_NAME,
    )


def resolve_qwen3_omni_runtime_config(
    config: Qwen3OmniMoeConfig,
) -> Qwen3OmniRuntimeConfig:
    """Resolve checkpoint-owned values and reject ambiguous model contracts.

    The published checkpoint stores the Talker sampled stop in the historically
    named ``codec_eos_token_id`` field. It is valid as a sampled stop only when
    it lies in the Talker logits vocabulary but outside the acoustic codebook.
    This rejects the Transformers class default that describes a different
    codec EOS value instead of silently swapping the two meanings.
    """
    codebook_size = _required(config, "code2wav_config.codebook_size")
    predictor_vocab_size = _required(
        config,
        "talker_config.code_predictor_config.vocab_size",
    )
    if predictor_vocab_size != codebook_size:
        raise ValueError(
            f"Model {_MODEL_NAME!r} has conflicting acoustic codebook sizes: "
            f"code2wav_config.codebook_size={codebook_size}, "
            "talker_config.code_predictor_config.vocab_size="
            f"{predictor_vocab_size}"
        )

    talker_vocab_size = _required(config, "talker_config.text_config.vocab_size")
    sampled_stop_token_id = _required(config, "talker_config.codec_eos_token_id")
    if not codebook_size <= sampled_stop_token_id < talker_vocab_size:
        raise ValueError(
            f"Model {_MODEL_NAME!r} sampled stop token {sampled_stop_token_id} "
            f"must be outside acoustic codebook [0, {codebook_size}) and inside "
            f"Talker logits vocabulary [0, {talker_vocab_size}); "
            "talker_config.codec_eos_token_id does not satisfy that contract"
        )

    speaker_ids = get_required_config_field(
        config,
        "talker_config.speaker_id",
        expected_type=dict,
        model=_MODEL_NAME,
    )
    if not speaker_ids or any(
        not isinstance(name, str) or not isinstance(token_id, int) or isinstance(token_id, bool)
        for name, token_id in speaker_ids.items()
    ):
        raise ValueError(
            f"Model {_MODEL_NAME!r} requires talker_config.speaker_id to be a non-empty str-to-int mapping"
        )

    return Qwen3OmniRuntimeConfig(
        im_start_token_id=_required(config, "im_start_token_id"),
        system_token_id=_required(config, "system_token_id"),
        user_token_id=_required(config, "user_token_id"),
        assistant_token_id=_required(config, "assistant_token_id"),
        tts_pad_token_id=_required(config, "tts_pad_token_id"),
        tts_bos_token_id=_required(config, "tts_bos_token_id"),
        tts_eos_token_id=_required(config, "tts_eos_token_id"),
        thinker_audio_token_id=_required(config, "thinker_config.audio_token_id"),
        thinker_image_token_id=_required(config, "thinker_config.image_token_id"),
        thinker_video_token_id=_required(config, "thinker_config.video_token_id"),
        accept_hidden_layer=_required(config, "talker_config.accept_hidden_layer"),
        thinker_hidden_size=_required(config, "talker_config.thinker_hidden_size"),
        codebook_size=codebook_size,
        num_quantizers=_required(config, "code2wav_config.num_quantizers"),
        codec_pad_token_id=_required(config, "talker_config.codec_pad_id"),
        codec_bos_token_id=_required(config, "talker_config.codec_bos_id"),
        codec_nothink_token_id=_required(config, "talker_config.codec_nothink_id"),
        codec_think_bos_token_id=_required(
            config,
            "talker_config.codec_think_bos_id",
        ),
        codec_think_eos_token_id=_required(
            config,
            "talker_config.codec_think_eos_id",
        ),
        sampled_stop_token_id=sampled_stop_token_id,
        talker_vocab_size=talker_vocab_size,
        speaker_ids=dict(speaker_ids),
    )
