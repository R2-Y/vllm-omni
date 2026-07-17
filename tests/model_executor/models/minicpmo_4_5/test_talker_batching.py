# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-alignment tests for MiniCPM-o 4.5's blocking talker batch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
)
from vllm_omni.utils.mm_outputs import to_payload_element

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeTalker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_request_ids: list[str | None] = []

    def forward(
        self,
        input_ids=None,
        positions=None,
        inputs_embeds=None,
        additional_information=None,
    ):
        info = additional_information or {}
        self.seen_request_ids.append(info.get("request_id"))
        if not info.get("emit_audio", False):
            return torch.zeros(input_ids.shape[0], 4)
        value = float(info["waveform_value"])
        return None, torch.full((2,), value, dtype=torch.float32)


def _make_talker_wrapper() -> MiniCPMO45OmniForConditionalGeneration:
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "tts"
    model.config = SimpleNamespace(hidden_size=4)
    model.talker = _FakeTalker()
    return model


def test_talker_batch_preserves_request_order_and_empty_slots() -> None:
    model = _make_talker_wrapper()
    runtime_info = [
        {"request_id": "req-0", "emit_audio": True, "waveform_value": 1},
        {"request_id": "req-1", "emit_audio": False},
        {"request_id": "req-2", "emit_audio": True, "waveform_value": 3},
    ]

    output = model(
        input_ids=torch.tensor([1, 0, 2, 1, 0, 2, 1, 0, 2]),
        positions=torch.arange(9),
        model_intermediate_buffer=runtime_info,
        runtime_additional_information=[{"request_id": "legacy", "emit_audio": True, "waveform_value": 9}],
    )

    assert model.talker.seen_request_ids == ["req-0", "req-1", "req-2"]
    assert output.multimodal_outputs is not None
    waveforms = output.multimodal_outputs["model_outputs"]
    assert len(waveforms) == len(runtime_info)
    assert waveforms[0].tolist() == [1.0, 1.0]
    assert waveforms[1].numel() == 0
    assert waveforms[2].tolist() == [3.0, 3.0]

    routed = [
        to_payload_element(output.multimodal_outputs, idx, idx * 3, (idx + 1) * 3) for idx in range(len(runtime_info))
    ]
    assert routed[0]["model_outputs"].tolist() == [1.0, 1.0]
    assert routed[1]["model_outputs"].numel() == 0
    assert routed[2]["model_outputs"].tolist() == [3.0, 3.0]


def test_talker_single_request_keeps_batch_aligned_output() -> None:
    model = _make_talker_wrapper()

    output = model(
        input_ids=torch.tensor([1, 0, 2]),
        positions=torch.arange(3),
        runtime_additional_information=[{"request_id": "req-0", "emit_audio": True, "waveform_value": 7}],
    )

    assert output.multimodal_outputs is not None
    waveforms = output.multimodal_outputs["model_outputs"]
    assert len(waveforms) == 1
    assert waveforms[0].tolist() == [7.0, 7.0]


def test_talker_cleans_vocoder_state_after_request_error(mocker) -> None:
    talker = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    nn.Module.__init__(talker)
    talker.audio_tokenizer = SimpleNamespace(
        stream_cache={"stale": True},
        hift_cache_dict={"stale": True},
    )
    mocker.patch.object(talker, "generate_speech", side_effect=RuntimeError("vocoder failed"))

    with pytest.raises(RuntimeError, match="vocoder failed"):
        talker(
            input_ids=torch.tensor([1, 0, 2]),
            additional_information={
                "tts_token_ids": torch.tensor([10]),
                "tts_hidden_states": torch.zeros(1, 4),
            },
        )

    assert talker.audio_tokenizer.stream_cache is None
    assert talker.audio_tokenizer.hift_cache_dict == {}


def test_talker_dummy_logits_keep_one_row_per_request() -> None:
    talker = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    nn.Module.__init__(talker)

    logits = talker.compute_logits(torch.zeros(3, 4))

    assert logits.shape == (3, 2)


def _make_continuous_talker() -> MiniCPMO45OmniTTSForConditionalGeneration:
    talker = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    nn.Module.__init__(talker)
    talker.continuous_batching = True
    talker._num_audio_tokens = 8
    talker._batch_stop_logits = None
    talker._request_generators = {}
    talker._vocoder_states = {}
    talker._deferred_cleanup_ids = set()
    return talker


def test_continuous_talker_uses_request_spans_and_keeps_code_state_aligned(
    mocker,
) -> None:
    talker = _make_continuous_talker()
    seen: list[tuple[str, list[float]]] = []

    def sample(hidden, history, request_id):
        seen.append((request_id, hidden.reshape(-1).tolist()))
        return torch.tensor(2 if request_id == "req-a" else 3)

    mocker.patch.object(talker, "_sample_audio_code", side_effect=sample)
    infos = [
        {"request_id": "req-a", "audio_codes": {"accumulated": torch.tensor([1])}},
        {"request_id": "req-b", "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)}},
    ]

    output = talker.make_omni_output(
        torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 2), (2, 3)],
    )

    assert seen == [("req-a", [2.0, 0.0]), ("req-b", [3.0, 0.0])]
    assert infos[0]["audio_codes"]["accumulated"].tolist() == [1, 2]
    assert infos[1]["audio_codes"]["accumulated"].tolist() == [3]
    assert output.multimodal_outputs == {}
    logits = talker.compute_logits(output.text_hidden_states)
    assert logits.argmax(dim=-1).tolist() == [0, 0]


def test_continuous_talker_does_not_advance_incomplete_prefill(mocker) -> None:
    talker = _make_continuous_talker()
    sample = mocker.patch.object(talker, "_sample_audio_code")
    infos = [
        {
            "request_id": "req-prefill",
            "audio_state": {"step": 0},
            "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
        },
        {
            "request_id": "req-decode",
            "audio_state": {"step": 4},
            "audio_codes": {"accumulated": torch.tensor([1])},
        },
    ]
    sample.return_value = torch.tensor(2)

    talker.make_omni_output(
        torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 2), (2, 3)],
        request_sample_eligible=[False, True],
    )

    sample.assert_called_once()
    assert sample.call_args.args[2] == "req-decode"
    assert infos[0]["audio_state"]["step"] == 0
    assert infos[0]["audio_codes"]["accumulated"].numel() == 0
    assert infos[1]["audio_state"]["step"] == 5


def test_continuous_talker_emits_stop_channel_without_appending_eos(mocker) -> None:
    talker = _make_continuous_talker()
    mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(7))
    mocker.patch.object(talker, "_run_vocoder_chunk", return_value=torch.ones(4))
    infos = [
        {
            "request_id": "req-stop",
            "audio_codes": {"accumulated": torch.tensor([4, 5])},
        }
    ]

    talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 1)],
    )

    assert infos[0]["audio_codes"]["accumulated"].tolist() == [4, 5]
    assert infos[0]["audio_state"]["finished"] is True
    assert talker.compute_logits(torch.ones(1, 2)).argmax(dim=-1).tolist() == [1]


def test_continuous_talker_flushes_vocoder_at_internal_length_limit(mocker) -> None:
    talker = _make_continuous_talker()
    mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(3))
    run_vocoder = mocker.patch.object(talker, "_run_vocoder_chunk", return_value=torch.ones(4))
    infos = [
        {
            "request_id": "req-limit",
            "audio_state": {"step": 1, "max_tokens": 2},
            "audio_codes": {"accumulated": torch.tensor([4, 5])},
        }
    ]

    talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 1)],
    )

    assert infos[0]["audio_codes"]["accumulated"].tolist() == [4, 5, 3]
    assert infos[0]["audio_state"]["finished"] is True
    assert run_vocoder.call_args.kwargs["last_chunk"] is True
    assert talker.compute_logits(torch.ones(1, 2)).argmax(dim=-1).tolist() == [1]


def test_continuous_talker_cleans_request_generator_after_finish() -> None:
    talker = _make_continuous_talker()
    talker._request_generators["req-done"] = torch.Generator()

    talker.on_requests_finished(["req-done"])
    talker._flush_deferred_cleanup()

    assert "req-done" not in talker._request_generators


def test_continuous_talker_swaps_vocoder_cache_per_request(mocker) -> None:
    class FakeTokenizer:
        def __init__(self):
            self.cache = None
            self.stream_cache = None
            self.hift_cache_dict = {}

        def set_stream_cache(self, prompt):
            self.cache = {"prompt": prompt}
            return {"seen": []}, {"speech": []}

        def stream(self, codes, prompt, last_chunk=False, return_waveform=False):
            assert return_waveform
            self.stream_cache["seen"].extend(codes)
            self.hift_cache_dict["speech"].extend(codes)
            return torch.tensor(codes, dtype=torch.float32).numpy()

    talker = _make_continuous_talker()
    talker.audio_tokenizer = FakeTokenizer()
    talker._vocoder_loaded = True
    talker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model="/tmp/not-a-model")
    )
    mocker.patch("os.path.exists", return_value=False)

    talker._run_vocoder_chunk("req-a", torch.tensor([1, 2]), last_chunk=False)
    talker._run_vocoder_chunk("req-b", torch.tensor([9]), last_chunk=False)
    talker._run_vocoder_chunk("req-a", torch.tensor([3]), last_chunk=True)

    assert talker._vocoder_states["req-b"]["stream_cache"]["seen"] == [9]
    assert "req-a" not in talker._vocoder_states
    assert talker.audio_tokenizer.cache is None
    assert talker.audio_tokenizer.stream_cache is None
    assert talker.audio_tokenizer.hift_cache_dict == {}
