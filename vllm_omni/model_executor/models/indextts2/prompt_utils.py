# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt helpers for IndexTTS2 talker prefill."""

from __future__ import annotations

from functools import lru_cache

from vllm_omni.config.stage_config import get_required_config_field
from vllm_omni.model_executor.models.indextts2.configuration_indextts2 import (
    IndexTTS2Config,
    IndexTTS25Config,
)
from vllm_omni.model_executor.models.indextts2.preprocess_utils import resolve_model_file
from vllm_omni.model_executor.models.indextts2.text_processing_v2_5 import (
    prepare_indextts25_text,
)
from vllm_omni.model_executor.models.indextts2.tokenizer import IndexTTS2Tokenizer
from vllm_omni.model_executor.models.indextts2.tokenizer_v2_5 import (
    INDEXTTS25_TOKENIZER_FILE,
)


def _resolve_bpe_model_path(model_id_or_path: str) -> str:
    path = resolve_model_file(model_id_or_path, "bpe.model")
    if path is None:
        raise FileNotFoundError(f"Could not resolve bpe.model for {model_id_or_path!r}")
    return path


@lru_cache(maxsize=16)
def _get_text_tokenizer(model_id_or_path: str) -> IndexTTS2Tokenizer:
    return IndexTTS2Tokenizer(_resolve_bpe_model_path(model_id_or_path), model_dir=model_id_or_path)


def estimate_indextts2_prefill_prompt_len(
    model_id_or_path: str,
    text: str,
    *,
    model_type: str = "indextts2",
    lang: str = "zh",
    text_normalization: bool = True,
    tokenizer_file: str = INDEXTTS25_TOKENIZER_FILE,
    hf_config: IndexTTS2Config | None = None,
) -> int:
    """Return the placeholder prompt length expected by the IndexTTS2 talker.

    IndexTTS 2 uses 34 conditioning tokens. IndexTTS 2.5 uses one projected
    CAMPPlus speaker token followed by two zero tokens.
    """
    if hf_config is None:
        config_cls = IndexTTS25Config if model_type == "indextts2_5" else IndexTTS2Config
        hf_config = config_cls.from_pretrained(model_id_or_path)
    conditioning_prefix_tokens = get_required_config_field(
        hf_config,
        "conditioning_prefix_tokens",
        expected_type=int,
        model=model_type,
    )
    start_text_token = get_required_config_field(
        hf_config,
        "gpt.start_text_token",
        expected_type=int,
        model=model_type,
    )
    stop_text_token = get_required_config_field(
        hf_config,
        "gpt.stop_text_token",
        expected_type=int,
        model=model_type,
    )
    if model_type == "indextts2_5":
        text_ids, _ = prepare_indextts25_text(
            text,
            lang=lang,
            model_dir=model_id_or_path,
            text_normalization=text_normalization,
            tokenizer_file=tokenizer_file,
        )
        text_ids = [token_id for token_id in text_ids if token_id not in {start_text_token, stop_text_token}]
    else:
        tokenizer = _get_text_tokenizer(model_id_or_path)
        text_ids = tokenizer.encode(text, add_special_tokens=False)
    text_token_count = len(text_ids) + 2  # Config-owned start/stop text wrappers.
    return conditioning_prefix_tokens + text_token_count + 1  # One config-owned start-mel token.


def build_indextts2_prefill_prompt_ids(
    model_id_or_path: str,
    text: str,
    *,
    model_type: str = "indextts2",
    lang: str = "zh",
    text_normalization: bool = True,
    tokenizer_file: str = INDEXTTS25_TOKENIZER_FILE,
    placeholder_token_id: int = 1,
    hf_config: IndexTTS2Config | None = None,
) -> list[int]:
    prompt_len = estimate_indextts2_prefill_prompt_len(
        model_id_or_path,
        text,
        model_type=model_type,
        lang=lang,
        text_normalization=text_normalization,
        tokenizer_file=tokenizer_file,
        hf_config=hf_config,
    )
    return [placeholder_token_id] * prompt_len
