# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Caller text cannot close a role or masquerade as reference audio frames."""

import pytest
import torch

from vllm_omni.model_executor.models.fish_omni.prompt import Reference, build_prompt_frames, prompt_length

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


@pytest.mark.parametrize("field", ["input", "reference"])
@pytest.mark.parametrize(
    "marker", ["<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:7|>"]
)
def test_structural_tokens_rejected_before_encoding(field, marker):
    text = "Hello " + marker if field == "input" else "Hello"
    transcript = "Reference " + marker if field == "reference" else "Reference"
    with pytest.raises(ValueError, match="prompt builder"):
        build_prompt_frames(
            Tokenizer(),
            text,
            [Reference(transcript, torch.zeros((1, 3), dtype=torch.long))],
            semantic_begin_id=1000,
            num_codebooks=3,
        )
    with pytest.raises(ValueError, match="prompt builder"):
        prompt_length(Tokenizer(), text, [transcript], [1])


@pytest.mark.parametrize("tag", ["<|speaker:16|>", "<|speaker:01|>", "<|speaker:-1|>", "<|speaker:１|>", "<|speaker:0"])
def test_invalid_speaker_tag_rejected(tag):
    with pytest.raises(ValueError, match="speaker tags"):
        prompt_length(Tokenizer(), tag + " text", [], [])


def test_reference_leading_whitespace_moves_after_speaker_tag():
    tokenizer = Tokenizer()
    frames = build_prompt_frames(
        tokenizer,
        "こんにちは。「入力中」です。",
        [Reference(" \t<|speaker:2|> Reference", torch.zeros((1, 3), dtype=torch.long))],
        semantic_begin_id=1000,
        num_codebooks=3,
    )
    decoded = bytes(frames[frames[:, 0] < 1000, 0].tolist()).decode()
    assert "Text: <|speaker:2|> \t Reference\nSpeech: " in decoded
