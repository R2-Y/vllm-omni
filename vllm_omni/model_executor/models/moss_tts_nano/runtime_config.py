# SPDX-License-Identifier: Apache-2.0
"""Validated checkpoint runtime contract for MOSS-TTS-Nano."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import AutoConfig, PretrainedConfig

from vllm_omni.config.stage_config import get_required_config_field


class MossTTSNanoConfig(PretrainedConfig):
    model_type = "moss_tts_nano"

    def __init__(
        self,
        eos_token_id: int = 2,
        streaming_continue_token_id: int = 1,
        audio_tokenizer_sample_rate: int = 48000,
        **kwargs: Any,
    ) -> None:
        super().__init__(eos_token_id=eos_token_id, **kwargs)
        self.streaming_continue_token_id = streaming_continue_token_id
        self.audio_tokenizer_sample_rate = audio_tokenizer_sample_rate


@dataclass(frozen=True)
class MossTTSNanoRuntimeConfig:
    eos_token_id: int
    streaming_continue_token_id: int
    vocab_size: int
    hidden_size: int
    audio_tokenizer_sample_rate: int


def resolve_moss_tts_nano_runtime_config(config: object) -> MossTTSNanoRuntimeConfig:
    required = lambda field: get_required_config_field(  # noqa: E731
        config,
        field,
        expected_type=int,
        model="moss_tts_nano",
    )
    runtime = MossTTSNanoRuntimeConfig(
        eos_token_id=required("eos_token_id"),
        streaming_continue_token_id=required("streaming_continue_token_id"),
        vocab_size=required("vocab_size"),
        hidden_size=required("hidden_size"),
        audio_tokenizer_sample_rate=required("audio_tokenizer_sample_rate"),
    )
    if not 0 <= runtime.eos_token_id < runtime.vocab_size:
        raise ValueError("MOSS-TTS-Nano eos_token_id must be inside vocab_size")
    if not 0 <= runtime.streaming_continue_token_id < runtime.vocab_size:
        raise ValueError("MOSS-TTS-Nano streaming_continue_token_id must be inside vocab_size")
    if runtime.streaming_continue_token_id == runtime.eos_token_id:
        raise ValueError("MOSS-TTS-Nano streaming continue and EOS tokens must differ")
    return runtime


AutoConfig.register("moss_tts_nano", MossTTSNanoConfig)

__all__ = [
    "MossTTSNanoConfig",
    "MossTTSNanoRuntimeConfig",
    "resolve_moss_tts_nano_runtime_config",
]
