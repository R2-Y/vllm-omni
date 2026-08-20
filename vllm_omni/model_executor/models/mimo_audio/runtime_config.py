# SPDX-License-Identifier: Apache-2.0
"""Validated checkpoint/runtime contract for MiMo Audio."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from vllm_omni.config.stage_config import get_required_config_field

from .config_mimo_audio import MiMoAudioConfig

_MODEL = "mimo_audio"
_NAMESPACE = "mimo_audio"


@dataclass(frozen=True)
class MiMoAudioRuntimeConfig:
    empty_token_id: int
    span_codec_start_token_id: int
    span_codec_end_token_id: int
    no_interleave_next_token_id: int
    speech_start_token_id: int
    speech_end_token_id: int
    endoftext_token_id: int
    im_end_token_id: int
    vocab_size: int
    max_code2wav_tokens: int

    def to_sampling_extra_args(self) -> dict[str, dict[str, Any]]:
        return {"model_runtime": {_NAMESPACE: asdict(self)}}


def _required(config: object, field: str) -> int:
    return get_required_config_field(config, field, expected_type=int, model=_MODEL)


def resolve_mimo_audio_runtime_config(config: MiMoAudioConfig) -> MiMoAudioRuntimeConfig:
    runtime = MiMoAudioRuntimeConfig(
        empty_token_id=_required(config, "empty_token_id"),
        span_codec_start_token_id=_required(config, "span_codec_start_token_id"),
        span_codec_end_token_id=_required(config, "span_codec_end_token_id"),
        no_interleave_next_token_id=_required(config, "no_interleave_next_token_id"),
        speech_start_token_id=_required(config, "speech_start_token_id"),
        speech_end_token_id=_required(config, "speech_end_token_id"),
        endoftext_token_id=_required(config, "endoftext_token_id"),
        im_end_token_id=_required(config, "im_end_token_id"),
        vocab_size=_required(config, "vocab_size"),
        max_code2wav_tokens=_required(config, "max_code2wav_tokens"),
    )
    values = asdict(runtime)
    token_values = {name: value for name, value in values.items() if name.endswith("_token_id")}
    invalid = {name: value for name, value in token_values.items() if value < 0}
    if invalid:
        raise ValueError(f"Model {_MODEL!r} requires non-negative token IDs; got {invalid}")
    if len(set(token_values.values())) != len(token_values):
        raise ValueError(f"Model {_MODEL!r} requires distinct special token IDs; got {token_values}")
    if runtime.vocab_size <= max(token_values.values()):
        raise ValueError(f"Model {_MODEL!r} requires special token IDs below vocab_size; got {values}")
    return runtime


def mimo_audio_runtime_from_sampling_params(sampling_params: Any) -> Mapping[str, Any]:
    extra_args = getattr(sampling_params, "extra_args", None)
    model_runtime = extra_args.get("model_runtime") if isinstance(extra_args, Mapping) else None
    runtime = model_runtime.get(_NAMESPACE) if isinstance(model_runtime, Mapping) else None
    if not isinstance(runtime, Mapping):
        raise ValueError("MiMo Audio requires sampling_params.extra_args['model_runtime']['mimo_audio']")
    return runtime


def mimo_audio_runtime_from_meta(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    model_runtime = meta.get("model_runtime")
    runtime = model_runtime.get(_NAMESPACE) if isinstance(model_runtime, Mapping) else None
    if not isinstance(runtime, Mapping):
        raise ValueError("MiMo Audio payload requires meta['model_runtime']['mimo_audio']")
    return runtime


def get_required_mimo_audio_runtime_int(runtime: Mapping[str, Any], field: str) -> int:
    value = runtime.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"MiMo Audio requires integer runtime field {field!r}; got {value!r}")
    return value


__all__ = [
    "MiMoAudioRuntimeConfig",
    "get_required_mimo_audio_runtime_int",
    "mimo_audio_runtime_from_meta",
    "mimo_audio_runtime_from_sampling_params",
    "resolve_mimo_audio_runtime_config",
]
