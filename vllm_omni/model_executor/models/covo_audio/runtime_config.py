# SPDX-License-Identifier: Apache-2.0
"""Validated checkpoint/runtime contract for Covo Audio."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from vllm_omni.config.stage_config import get_required_config_field

from .config_covo_audio import CovoAudioConfig

_MODEL = "covo_audio"
_NAMESPACE = "covo_audio"


@dataclass(frozen=True)
class CovoAudioRuntimeConfig:
    audio_token_index: int
    eos_token_id: int
    vocab_size: int

    def to_sampling_extra_args(self) -> dict[str, dict[str, Any]]:
        return {"model_runtime": {_NAMESPACE: asdict(self)}}


def resolve_covo_audio_runtime_config(config: CovoAudioConfig) -> CovoAudioRuntimeConfig:
    runtime = CovoAudioRuntimeConfig(
        audio_token_index=get_required_config_field(
            config,
            "audio_token_index",
            expected_type=int,
            model=_MODEL,
        ),
        eos_token_id=get_required_config_field(
            config,
            "eos_token_id",
            expected_type=int,
            model=_MODEL,
        ),
        vocab_size=get_required_config_field(
            config,
            "vocab_size",
            expected_type=int,
            model=_MODEL,
        ),
    )
    if runtime.audio_token_index < 0 or runtime.eos_token_id < 0:
        raise ValueError(f"Model {_MODEL!r} requires non-negative token IDs; got {runtime}")
    if runtime.eos_token_id >= runtime.audio_token_index:
        raise ValueError(f"Model {_MODEL!r} requires eos_token_id below audio_token_index; got {runtime}")
    if runtime.audio_token_index >= runtime.vocab_size:
        raise ValueError(f"Model {_MODEL!r} requires audio_token_index below vocab_size; got {runtime}")
    return runtime


def covo_audio_runtime_from_sampling_params(sampling_params: Any) -> Mapping[str, Any]:
    extra_args = getattr(sampling_params, "extra_args", None)
    model_runtime = extra_args.get("model_runtime") if isinstance(extra_args, Mapping) else None
    runtime = model_runtime.get(_NAMESPACE) if isinstance(model_runtime, Mapping) else None
    if not isinstance(runtime, Mapping):
        raise ValueError("Covo Audio requires sampling_params.extra_args['model_runtime']['covo_audio']")
    return runtime


def covo_audio_runtime_from_meta(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    model_runtime = meta.get("model_runtime")
    runtime = model_runtime.get(_NAMESPACE) if isinstance(model_runtime, Mapping) else None
    if not isinstance(runtime, Mapping):
        raise ValueError("Covo Audio payload requires meta['model_runtime']['covo_audio']")
    return runtime


def get_required_covo_audio_runtime_int(runtime: Mapping[str, Any], field: str) -> int:
    value = runtime.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Covo Audio requires integer runtime field {field!r}; got {value!r}")
    return value


__all__ = [
    "CovoAudioRuntimeConfig",
    "covo_audio_runtime_from_meta",
    "covo_audio_runtime_from_sampling_params",
    "get_required_covo_audio_runtime_int",
    "resolve_covo_audio_runtime_config",
]
