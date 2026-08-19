# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.step_audio2.configuration_step_audio2 import (
    StepAudio2Config,
)
from vllm_omni.model_executor.stage_input_processors.step_audio2 import (
    thinker2token2wav_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _runtime(config: StepAudio2Config) -> dict:
    return {"meta": {"model_runtime": {"step_audio2": asdict(config)}}}


def _req(external_req_id: str, *, prompt_token_ids: list[int], all_token_ids: list[int], finished: bool):
    return SimpleNamespace(
        external_req_id=external_req_id,
        prompt_token_ids=prompt_token_ids,
        all_token_ids=all_token_ids,
        is_finished=lambda: finished,
    )


def test_step_audio2_async_chunk_uses_decode_only_tokens_not_prompt_history():
    config = StepAudio2Config(audio_start=120000, text_max=119998, audio_patch_token_id=119999)
    audio_start = config.audio_start
    audio_eos = config.audio_eos
    transfer_manager = SimpleNamespace(code_prompt_token_ids=defaultdict(list))

    prompt = [11, 12, audio_start + 5]  # historical prompt audio token
    generated = [99, audio_start + 1, audio_start + 2, audio_start + audio_eos]
    request = _req(
        "rid-decode-only",
        prompt_token_ids=prompt,
        all_token_ids=prompt + generated,
        finished=True,
    )

    payload = thinker2token2wav_async_chunk(
        transfer_manager=transfer_manager,
        multimodal_output=_runtime(config),
        request=request,
    )

    assert payload is not None
    assert payload.codes.audio.tolist() == [1, 2]
    assert payload.meta.left_context_size == 1
    assert payload.meta.finished.item() is True


def test_step_audio2_async_chunk_returns_none_when_not_enough_tokens():
    config = StepAudio2Config(chunk_size=7, pre_lookahead_len=2)
    audio_start = config.audio_start
    required = config.chunk_size + config.pre_lookahead_len
    transfer_manager = SimpleNamespace(code_prompt_token_ids=defaultdict(list))

    generated = [audio_start + i for i in range(required - 1)]
    request = _req(
        "rid-not-ready",
        prompt_token_ids=[1, 2, 3],
        all_token_ids=[1, 2, 3] + generated,
        finished=False,
    )

    payload = thinker2token2wav_async_chunk(
        transfer_manager=transfer_manager,
        multimodal_output=_runtime(config),
        request=request,
    )

    assert payload is None


def test_step_audio2_async_chunk_emits_non_last_chunk_and_advances_consumed_by_chunk_size():
    config = StepAudio2Config(chunk_size=7, pre_lookahead_len=2)
    audio_start = config.audio_start
    chunk_size = config.chunk_size
    required = chunk_size + config.pre_lookahead_len
    transfer_manager = SimpleNamespace(code_prompt_token_ids=defaultdict(list))

    generated = [audio_start + i for i in range(required + 10)]
    request = _req(
        "rid-ready",
        prompt_token_ids=[7, 8],
        all_token_ids=[7, 8] + generated,
        finished=False,
    )

    payload = thinker2token2wav_async_chunk(
        transfer_manager=transfer_manager,
        multimodal_output=_runtime(config),
        request=request,
    )

    assert payload is not None
    assert payload.meta.left_context_size == 0
    assert payload.meta.finished.item() is False
    assert payload.codes.audio.tolist() == list(range(required))
    assert transfer_manager.code_prompt_token_ids["rid-ready"] == list(range(chunk_size))


def test_step_audio2_async_chunk_emits_eof_when_finished_with_no_remaining_audio():
    config = StepAudio2Config()
    transfer_manager = SimpleNamespace(code_prompt_token_ids=defaultdict(list))
    transfer_manager.code_prompt_token_ids["rid-eof"] = [1, 2, 3]

    request = _req(
        "rid-eof",
        prompt_token_ids=[10, 11],
        all_token_ids=[
            10,
            11,
            config.audio_start + 1,
            config.audio_start + 2,
            config.audio_start + 3,
        ],
        finished=True,
    )

    payload = thinker2token2wav_async_chunk(
        transfer_manager=transfer_manager,
        multimodal_output=_runtime(config),
        request=request,
    )

    assert payload is not None
    assert payload.codes.audio.numel() == 0
    assert payload.meta.left_context_size == 1
    assert payload.meta.finished.item() is True


def test_step_audio2_async_chunk_requires_model_runtime_meta():
    request = _req(
        "rid-missing-runtime",
        prompt_token_ids=[],
        all_token_ids=[],
        finished=True,
    )
    with pytest.raises(ValueError, match="model runtime metadata"):
        thinker2token2wav_async_chunk(
            transfer_manager=SimpleNamespace(code_prompt_token_ids=defaultdict(list)),
            multimodal_output=None,
            request=request,
        )
