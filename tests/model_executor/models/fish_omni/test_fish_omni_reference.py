# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU reference selection and cache contracts; no alignment weights required."""

import numpy as np
import pytest

from vllm_omni.model_executor.models.fish_omni.reference import (
    ReferenceSentencePreprocessor,
    SentenceSpan,
    _word_boundaries,
    first_sentence_crop,
    reference_sentences,
)
from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_alignment_checkpoint_is_configurable_and_serialized():
    from vllm_omni.model_executor.models.fish_omni.configuration_fish_omni import FishOmniConfig

    config = FishOmniConfig(reference_alignment_model="/models/small.pt")
    assert config.to_dict()["reference_alignment_model"] == "/models/small.pt"
    assert FishOmniConfig().reference_alignment_model == "small"
    with pytest.raises(ValueError, match="reference_alignment_model"):
        FishOmniConfig(reference_alignment_model=" ")


@pytest.mark.parametrize(
    ("text", "prefix", "sentences"),
    [
        ("……山です。秋です。", "", ["山です。", "秋です。"]),
        ("<|speaker:2|> ……One. Two!", "<|speaker:2|> ", ["One.", "Two!"]),
        ("「山です。」 秋です！", "", ["「山です。」", "秋です！"]),
        ("Dr. Li paid 1.25. Next?", "", ["Dr. Li paid 1.25.", "Next?"]),
        ("A single phrase", "", ["A single phrase"]),
    ],
)
def test_sentence_text_is_preserved(text, prefix, sentences):
    assert reference_sentences(text) == (prefix, sentences)


@pytest.mark.parametrize("text", ["One. unfinished", "One. <|im_end|> Two.", "One. [sigh] Two."])
def test_unsafe_reference_is_rejected_before_cropping(text):
    with pytest.raises(ValueError):
        reference_sentences(text)


def test_first_is_selected_even_when_later_sentence_is_longer():
    audio = np.arange(1000, dtype=np.float32)
    crop = first_sentence_crop(
        audio, 100, ["Short.", "Much longer second sentence."], [SentenceSpan(0.5, 1.5), SentenceSpan(3.0, 7.0)]
    )
    np.testing.assert_array_equal(crop, audio[25:175])
    assert not np.shares_memory(crop, audio)


def test_margins_do_not_include_adjacent_sentence_speech():
    audio = np.arange(1000, dtype=np.float32)
    crop = first_sentence_crop(audio, 100, ["Long.", "Short."], [SentenceSpan(0, 3.0), SentenceSpan(3.1, 4.0)])
    np.testing.assert_array_equal(crop, audio[:305])


def test_observed_word_edges_restore_audited_reference_cut():
    # Supplied-text alignment ends at 3.50s; the audited blind words end at
    # 3.56s. Keep those observed edges instead of silently changing the cut.
    spans = _word_boundaries(
        [SentenceSpan(0, 3.5), SentenceSpan(5.3, 8.3)],
        [SentenceSpan(0, 2.34), SentenceSpan(2.34, 2.34), SentenceSpan(2.34, 3.56), SentenceSpan(5.28, 8.32)],
        8.6,
    )
    audio = np.arange(379260, dtype=np.float32)
    crop = first_sentence_crop(audio, 44100, ["山の木々が、きれいに色づいてきました。", "秋です。"], spans)
    np.testing.assert_array_equal(crop, audio[:168021])


@pytest.mark.parametrize(
    "words",
    [
        [],
        [SentenceSpan(0, 1)],
        [SentenceSpan(0, 1), SentenceSpan(1.2, 1.4), SentenceSpan(2, 3)],
        [SentenceSpan(0, float("nan"))],
        [SentenceSpan(0, 1), SentenceSpan(2, 4.1)],
    ],
)
def test_unmatched_or_invalid_observed_words_are_rejected(words):
    with pytest.raises(ValueError, match="reference ASR"):
        _word_boundaries([SentenceSpan(0, 1), SentenceSpan(2, 3)], words, 4)


@pytest.mark.parametrize(
    "spans",
    [
        [SentenceSpan(0, 1)],
        [SentenceSpan(0, 4), SentenceSpan(3, 6)],
        [SentenceSpan(0, 1), SentenceSpan(2, float("nan"))],
        [SentenceSpan(0, 1), SentenceSpan(2, 20)],
    ],
)
def test_bad_alignment_has_no_guessed_fallback(spans):
    with pytest.raises(ValueError):
        first_sentence_crop(np.zeros(1000, dtype=np.float32), 100, ["One.", "Two."], spans)


def test_cached_pair_tracks_audio_transcript_and_model_and_single_sentence_skips_alignment(monkeypatch):
    cache = SpeakerEmbeddingCache(max_bytes=100_000)
    processor = ReferenceSentencePreprocessor("local-small.pt", cache)
    calls = []

    def align(waveform, rate, sentences):
        calls.append((rate, sentences))
        return [SentenceSpan(0, 1), SentenceSpan(2, 5)]

    monkeypatch.setattr(processor, "_align", align)
    audio = np.arange(1000, dtype=np.float32)
    assert processor.prepare(audio, 100, "One sentence.") is None
    first = processor.prepare(audio, 100, "One. Two.")
    again = processor.prepare(audio.copy(), 100, "One. Two.")
    assert first.transcript == again.transcript == "One."
    np.testing.assert_array_equal(again.waveform, audio[:125])
    assert len(calls) == 1
    assert cache.memory_bytes() == again.waveform.nbytes
    changed = processor.prepare(audio, 100, "One. Another.")
    assert changed.transcript == "One." and changed.cache_key != first.cache_key
    processor.prepare(audio + 1, 100, "One. Two.")
    assert len(calls) == 3
    other_model = ReferenceSentencePreprocessor("another-small.pt", cache)
    monkeypatch.setattr(other_model, "_align", align)
    other = other_model.prepare(audio, 100, "One. Two.")
    assert other.cache_key != first.cache_key and len(calls) == 4


def test_disabled_cache_recomputes_and_never_stores_unbounded_arrays(monkeypatch):
    processor = ReferenceSentencePreprocessor("local-small.pt", SpeakerEmbeddingCache(max_bytes=0))
    calls = []

    def align(waveform, rate, sentences):
        calls.append(1)
        return [SentenceSpan(0, 1), SentenceSpan(2, 4)]

    monkeypatch.setattr(processor, "_align", align)
    for _ in range(2):
        processor.prepare(np.zeros(500, dtype=np.float32), 100, "One. Two.")
    assert len(calls) == 2 and processor.cache.memory_bytes() == 0
