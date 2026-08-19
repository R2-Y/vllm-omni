from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class StepAudio2Config:
    """Checkpoint contract used by the thinker and Token2Wav stages."""

    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    dummy_audio_seconds: int = 25
    text_max: int = 151688
    audio_start: int = 151696
    audio_vocab_size: int = 6562
    audio_eos: int = 6561
    audio_patch_token_id: int = 151690
    n_mels: int = 128
    n_audio_ctx: int = 1500
    n_audio_state: int = 512
    n_audio_head: int = 8
    n_audio_layer: int = 6
    kernel_size: int = 3
    adapter_stride: int = 2
    encoder_downsample_rate: int = 4
    max_audio_tokens_per_item: int = 250
    n_timesteps: int = 10
    chunk_size: int = 25
    pre_lookahead_len: int = 3
    up_rate: int = 2
    mel_cache_len: int = 8
    estimator_cache_keep: int = 100
    source_cache_samples_per_mel: int = 480
    prompt_n_fft: int = 1920
    prompt_num_mels: int = 80
    prompt_hop_size: int = 480
    prompt_win_size: int = 1920
    prompt_fmin: int = 0
    prompt_fmax: int = 8000

    @classmethod
    def from_hf_config(cls, hf_config: object) -> StepAudio2Config:
        raw: Any = getattr(hf_config, "step_audio2", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError(
                "Step-Audio2 config field 'step_audio2' must be a mapping; "
                f"got {raw!r}"
            )
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(
                "Step-Audio2 config contains unsupported fields: "
                + ", ".join(unknown)
            )
        values: dict[str, Any] = {}
        audio_encoder = getattr(hf_config, "audio_encoder_config", None)
        if isinstance(audio_encoder, Mapping):
            for name in (
                "n_mels",
                "n_audio_ctx",
                "n_audio_state",
                "n_audio_head",
                "n_audio_layer",
                "kernel_size",
                "adapter_stride",
            ):
                if name in audio_encoder:
                    values[name] = audio_encoder[name]
        elif audio_encoder is not None:
            for name in (
                "n_mels",
                "n_audio_ctx",
                "n_audio_state",
                "n_audio_head",
                "n_audio_layer",
                "kernel_size",
                "adapter_stride",
            ):
                if hasattr(audio_encoder, name):
                    values[name] = getattr(audio_encoder, name)
        values.update(raw)
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> StepAudio2Config:
        known = {field.name for field in fields(cls)}
        missing = sorted(known - set(values))
        if missing:
            raise ValueError(
                "Step-Audio2 runtime config is missing fields: "
                + ", ".join(missing)
            )
        config = cls(**dict(values))
        config.validate()
        return config

    @property
    def audio_end(self) -> int:
        return self.audio_start + self.audio_vocab_size - 1

    @property
    def stream_source_cache_len(self) -> int:
        return self.mel_cache_len * self.source_cache_samples_per_mel

    def validate(self) -> None:
        integer_fields = {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }
        invalid = [
            name
            for name, value in integer_fields.items()
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ]
        if self.prompt_fmin == 0:
            invalid = [name for name in invalid if name != "prompt_fmin"]
        if invalid:
            raise ValueError(
                "Step-Audio2 config requires positive integer fields; invalid: "
                + ", ".join(sorted(invalid))
            )
        if self.audio_eos >= self.audio_vocab_size:
            raise ValueError(
                "Step-Audio2 audio_eos must be smaller than audio_vocab_size; "
                f"got {self.audio_eos} >= {self.audio_vocab_size}"
            )
        if not self.text_max < self.audio_patch_token_id < self.audio_start:
            raise ValueError(
                "Step-Audio2 token ranges must satisfy "
                "text_max < audio_patch_token_id < audio_start"
            )
