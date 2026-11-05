# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Select the first reference sentence before estimating/encoding it.

Only reference audio is processed. Supplied text locates each sentence;
blind ASR word timestamps supply crop edges, without replacing that text.
There is no generated-audio ASR or proportional cut.
"""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass

import numpy as np
import regex as re
import torch

from vllm_omni.model_executor.models.fish_omni.prompt import (
    _SPEAKER_TAG,
    _reference_transcript,
    _validate_text,
)
from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

_CLOSERS = frozenset("\"'”’」』）)]}")
_ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc"})
_MARGIN_S = 0.25


def reference_sentences(transcript: str) -> tuple[str, list[str]]:
    """Preserve sentence text and an explicit initial speaker; validate all input."""
    _validate_text(transcript, "reference transcript")
    normalized = _reference_transcript(transcript)
    tag = _SPEAKER_TAG.match(normalized)
    assert tag is not None
    body = normalized[tag.end() :].strip()
    prefix = tag.group(0) + " " if _SPEAKER_TAG.search(transcript) else ""
    sentences: list[str] = []
    begin, index = 0, 0
    while index < len(body):
        char = body[index]
        terminal = char in "。！？!?"
        if char == ".":
            following = body[index + 1 : index + 2]
            token = re.search(r"[\w.]+$", body[begin:index])
            abbreviation = token is not None and (
                token.group().lower() in _ABBREVIATIONS or re.fullmatch(r"(?:[A-Za-z]\.)*[A-Za-z]", token.group())
            )
            terminal = not abbreviation and (not following or following.isspace() or following in _CLOSERS)
        if terminal:
            end = index + 1
            while end < len(body) and (body[end] in _CLOSERS or body[end] in "。！？!?"):
                end += 1
            piece = body[begin:end].strip()
            if any(c.isalnum() for c in piece):
                sentences.append(piece)
            begin, index = end, end
        else:
            index += 1
    tail = body[begin:].strip()
    if tail:
        if sentences:
            raise ValueError("multi-sentence ref_text must end with sentence punctuation")
        sentences.append(tail)
    if len(sentences) > 1 and re.search(r"[\[\]<>]", body):
        raise ValueError("automatic reference selection requires plain single-speaker ref_text")
    return prefix, sentences


@dataclass(frozen=True)
class SentenceSpan:
    start: float
    end: float


@dataclass(frozen=True)
class PreparedReference:
    transcript: str
    waveform: np.ndarray
    cache_key: str


def _word_boundaries(spans: list[SentenceSpan], words: list[SentenceSpan], duration: float) -> list[SentenceSpan]:
    """Assign observed words to supplied sentences without guessing across gaps."""
    grouped: list[list[SentenceSpan]] = [[] for _ in spans]
    previous_end = 0.0
    for word in words:
        if not (
            math.isfinite(word.start) and math.isfinite(word.end) and previous_end <= word.start <= word.end <= duration
        ):
            raise ValueError("reference ASR has invalid word timestamps")
        previous_end = word.end
        matches = [
            i
            for i, span in enumerate(spans)
            if (word.start == word.end and span.start <= word.start <= span.end)
            or max(word.start, span.start) < min(word.end, span.end)
        ]
        if len(matches) != 1:
            raise ValueError("reference ASR words do not map unambiguously to supplied sentences")
        grouped[matches[0]].append(word)
    if not all(grouped):
        raise ValueError("reference ASR did not cover every supplied sentence")
    return [SentenceSpan(group[0].start, group[-1].end) for group in grouped]


def first_sentence_crop(
    waveform: np.ndarray, sample_rate: int, sentences: list[str], spans: list[SentenceSpan]
) -> np.ndarray:
    """Crop the first sentence, retaining source margins within its neighbors."""
    if sample_rate <= 0 or waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
        raise ValueError("reference selection requires finite, non-empty mono audio and a positive sample rate")
    if len(sentences) != len(spans) or not spans:
        raise ValueError("reference alignment must cover every sentence")
    duration = waveform.size / sample_rate
    previous_end = 0.0
    for span in spans:
        if not (
            math.isfinite(span.start) and math.isfinite(span.end) and previous_end <= span.start < span.end <= duration
        ):
            raise ValueError("reference alignment has invalid or overlapping sentence boundaries")
        previous_end = span.end
    span = spans[0]
    right = (span.end + spans[1].start) / 2 if len(spans) > 1 else duration
    first = math.floor(max(0.0, span.start - _MARGIN_S) * sample_rate)
    last = min(waveform.size, math.ceil(min(right, span.end + _MARGIN_S) * sample_rate))
    return waveform[first:last].copy()


class ReferenceSentencePreprocessor:
    """One CPU aligner per serving adapter, sharing the bounded speaker LRU.

    Alignment registers temporary decoder hooks. The worker-thread lock protects
    those hooks and coalesces duplicate misses; it is never held on an event loop.
    """

    def __init__(self, model_name: str, cache: SpeakerEmbeddingCache) -> None:
        self.model_name = model_name
        self.cache = cache
        self._model = None
        self._lock = threading.Lock()

    def prepare(self, waveform: np.ndarray, sample_rate: int, transcript: str) -> PreparedReference | None:
        prefix, sentences = reference_sentences(transcript)
        if len(sentences) < 2:
            return None
        waveform = np.ascontiguousarray(waveform, dtype=np.float32)
        if sample_rate <= 0 or waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
            raise ValueError("reference selection requires finite, non-empty mono audio and a positive sample rate")
        if waveform.size / sample_rate > 30:
            raise ValueError("automatic reference sentence selection supports at most 30 seconds of reference audio")
        digest = hashlib.blake2b(digest_size=32)
        digest.update(repr(("first-v1-asr-word-boundaries", self.model_name, sample_rate, transcript)).encode())
        digest.update(waveform.tobytes())
        selection_key = digest.hexdigest()
        key = self.cache.make_cache_key(selection_key, "fish_omni_reference_sentence")
        with self._lock:
            cached = self.cache.get(key)
            if cached is not None:
                return PreparedReference(cached["ref_text"], cached["waveform"].numpy(), selection_key)
            spans = self._align(waveform, sample_rate, sentences)
            cropped = first_sentence_crop(waveform, sample_rate, sentences, spans)
            selected_text = prefix + sentences[0]
            self.cache.put(key, {"ref_text": selected_text, "waveform": torch.from_numpy(cropped)})
            return PreparedReference(selected_text, cropped, selection_key)

    def _align(self, waveform: np.ndarray, sample_rate: int, sentences: list[str]) -> list[SentenceSpan]:
        try:
            import whisper
            from whisper.audio import HOP_LENGTH, N_SAMPLES, SAMPLE_RATE, log_mel_spectrogram, pad_or_trim
            from whisper.timing import find_alignment
            from whisper.tokenizer import get_tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Drama 3 multi-sentence reference selection needs pip install 'vllm-omni[fish-reference]'"
            ) from exc
        from scipy.signal import resample_poly

        if self._model is None:
            self._model = whisper.load_model(self.model_name, device="cpu").eval()
        model = self._model
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        audio = resample_poly(waveform, SAMPLE_RATE // divisor, sample_rate // divisor).astype(np.float32)
        mel = log_mel_spectrogram(pad_or_trim(audio, N_SAMPLES), n_mels=model.dims.n_mels)
        with torch.inference_mode():
            if model.is_multilingual:
                _, probabilities = model.detect_language(mel)
                language = max(probabilities, key=probabilities.get)
            else:
                language = "en"
            tokenizer = get_tokenizer(
                model.is_multilingual, num_languages=model.num_languages, language=language, task="transcribe"
            )
            encoded = [tokenizer.encode((" " if i else "") + text) for i, text in enumerate(sentences)]
            tokens = [token for group in encoded for token in group]
            if len(tokens) + len(tokenizer.sot_sequence) + 2 > model.dims.n_text_ctx:
                raise ValueError("ref_text exceeds the reference aligner's text context")
            words = find_alignment(model, tokenizer, tokens, mel, math.ceil(audio.size / HOP_LENGTH))
        if [token for word in words for token in word.tokens] != tokens:
            raise ValueError("reference aligner did not cover the complete ref_text")
        spans = []
        offset = 0
        for group in encoded:
            end = offset + len(group)
            covered = []
            cursor = 0
            for word in words:
                word_end = cursor + len(word.tokens)
                if cursor < end and word_end > offset and any(c.isalnum() for c in word.word):
                    covered.append(word)
                cursor = word_end
            if not covered:
                raise ValueError("reference aligner found no speech for a sentence")
            spans.append(
                SentenceSpan(float(covered[0].start), min(float(covered[-1].end), waveform.size / sample_rate))
            )
            offset = end
        # Restore the audited preparation method: observed word edges plus the
        # original 0.25s margin. Forced alignment only associates words with the
        # supplied sentences; its shorter edges must not determine the crop.
        result = model.transcribe(
            audio,
            language=language,
            task="transcribe",
            temperature=0,
            beam_size=5,
            fp16=False,
            condition_on_previous_text=False,
            word_timestamps=True,
            initial_prompt=None,
            verbose=None,
        )
        observed = [
            SentenceSpan(float(word["start"]), float(word["end"]))
            for segment in result["segments"]
            for word in segment.get("words", [])
            if any(c.isalnum() for c in word["word"])
        ]
        return _word_boundaries(spans, observed, waveform.size / sample_rate)
