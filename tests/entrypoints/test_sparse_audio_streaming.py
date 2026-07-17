# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat


def test_sparse_audio_terminal_update_without_pcm_is_ignored() -> None:
    omni_output = SimpleNamespace(
        request_output=SimpleNamespace(outputs=[SimpleNamespace()]),
        multimodal_output={},
    )

    choices = OmniOpenAIServingChat._create_audio_choice(
        None,
        omni_output,
        "assistant",
        SimpleNamespace(),
        stream=True,
    )

    assert choices == []
