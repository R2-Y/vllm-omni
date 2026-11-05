# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drama 3 prompt frames in the ``reference-bracket-v1`` layout.

A frame row is ``[token id, code 0, ..., code Q-1]``. Text rows carry zero
codes; a reference audio frame's token id is ``semantic_begin + code 0``.
Reference transcripts drop leading ellipses; other punctuation, event tags
and user text are preserved. Length estimation uses the same normalization.
"""

from __future__ import annotations

from dataclasses import dataclass

import regex as re
import torch

TTS_INSTRUCTION = "convert the provided text to speech"
REFERENCE_INSTRUCTION = "convert the provided text to speech reference to the following:"
_SPEAKER_TAG = re.compile(r"<\|speaker:([0-9]+)\|>")
_SPEAKER_IDS = frozenset(str(speaker_id) for speaker_id in range(16))
_STRUCTURAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:")
# Match the native builder's Unicode White_Space set, excluding U+001C..U+001F.
_WHITE_SPACE = "".join(map(chr, (*range(9, 14), 32, 133, 160, 5760, *range(8192, 8203), 8232, 8233, 8239, 8287, 12288)))
_LEADING_REFERENCE_ELLIPSIS = re.compile(rf"\A([{_WHITE_SPACE}]*)(?:(?:[…⋯]|\.{{3,}})[{_WHITE_SPACE}]*)+")


def _validate_text(text: str, location: str) -> None:
    if not text.strip(_WHITE_SPACE):
        raise ValueError(f"{location}: text must not be empty")
    for token in _STRUCTURAL_TOKENS:
        if token in text:
            raise ValueError(f"{location}: {token} is written by the prompt builder")
    for marker in re.finditer(r"<\|speaker", text):
        tag = _SPEAKER_TAG.match(text, marker.start())
        digits = tag.group(1) if tag else ""
        if digits not in _SPEAKER_IDS:
            raise ValueError(f"{location}: speaker tags must be <|speaker:0|> to <|speaker:15|>")


@dataclass(frozen=True)
class Reference:
    transcript: str
    codes: torch.Tensor  # [frames, num_codebooks]


def with_speaker_tag(text: str) -> str:
    """Prefix ``<|speaker:0|> `` unless the text already starts with a speaker tag."""
    if _SPEAKER_TAG.match(text.lstrip(_WHITE_SPACE)):
        return text
    return f"<|speaker:0|> {text}"


def _reference_transcript(transcript: str) -> str:
    """A reference transcript starts with its speaker tag, directly after "Text: "."""
    # Only remove leading pause notation, not sentence punctuation or inline tags.
    transcript = _LEADING_REFERENCE_ELLIPSIS.sub(r"\1", transcript)
    tagged = with_speaker_tag(transcript)
    body = tagged.lstrip(_WHITE_SPACE)
    match = _SPEAKER_TAG.match(body)
    assert match is not None
    leading = tagged[: len(tagged) - len(body)]
    content = _LEADING_REFERENCE_ELLIPSIS.sub(r"\1", body[match.end() :])
    if not content.strip(_WHITE_SPACE):
        raise ValueError("reference transcript: text must not be empty after removing leading ellipses")
    return match.group(0) + leading + content


def _segments(text: str, references: list[Reference]) -> list[str | torch.Tensor]:
    _validate_text(text, "text")
    for index, reference in enumerate(references, 1):
        _validate_text(reference.transcript, f"reference {index}")
    segments: list[str | torch.Tensor] = [
        "<|im_start|>system\n",
        REFERENCE_INSTRUCTION if references else TTS_INSTRUCTION,
    ]
    for index, reference in enumerate(references):
        segments += [
            f"\n\n[Reference {index + 1}]\nText: {_reference_transcript(reference.transcript)}\nSpeech: ",
            reference.codes,
        ]
    segments += ["<|im_end|>\n", "<|im_start|>user\n", with_speaker_tag(text)]
    segments += ["<|im_end|>\n", "<|im_start|>assistant\n<|voice|>"]
    return segments


def build_prompt_frames(
    tokenizer,
    text: str,
    references: list[Reference],
    *,
    semantic_begin_id: int,
    num_codebooks: int,
) -> torch.Tensor:
    """CPU int64 rows [prompt_len, 1 + num_codebooks]. Each literal is tokenized on
    its own, because tokenizing the joined string changes ids at the seams."""
    rows: list[torch.Tensor] = []
    for segment in _segments(text, references):
        if isinstance(segment, str):
            ids = torch.tensor(tokenizer.encode(segment, add_special_tokens=False), dtype=torch.long)
            block = torch.zeros((ids.numel(), 1 + num_codebooks), dtype=torch.long)
            block[:, 0] = ids
        else:
            codes = segment.to(device="cpu", dtype=torch.long)
            if codes.ndim != 2 or codes.shape[1] != num_codebooks:
                raise ValueError(f"reference codes must be [frames, {num_codebooks}], got {tuple(codes.shape)}")
            block = torch.cat((codes[:, :1] + semantic_begin_id, codes), dim=1)
        rows.append(block)
    return torch.cat(rows)


def prompt_length(tokenizer, text: str, transcripts: list[str], frame_counts: list[int]) -> int:
    """Prompt rows for the given references without encoding their audio."""
    total = 0
    placeholder = [Reference(t, torch.zeros((n, 1), dtype=torch.long)) for t, n in zip(transcripts, frame_counts)]
    for segment in _segments(text, placeholder):
        if isinstance(segment, str):
            total += len(tokenizer.encode(segment, add_special_tokens=False))
        else:
            total += int(segment.shape[0])
    return total
