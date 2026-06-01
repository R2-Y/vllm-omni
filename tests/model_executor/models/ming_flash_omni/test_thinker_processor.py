# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_thinker import (
    MingFlashOmniThinkerMultiModalProcessor,
)
from vllm_omni.transformers_utils.processors.ming import MingFlashOmniProcessor


class _Tokenizer:
    def __call__(self, text, return_tensors=None, **_kwargs):
        del text, return_tensors
        return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}


class _Info:
    def __init__(self, hf_processor):
        self._hf_processor = hf_processor

    def get_hf_processor(self):
        return self._hf_processor

    def get_tokenizer(self):
        return _Tokenizer()


class _HFProcessor:
    def __init__(self):
        self.image_processor = Qwen2VLImageProcessor()
        self.audio_processor = None


def _dummy_video():
    return [np.full((2, 64, 64, 3), 255, dtype=np.uint8)]


def test_ming_thinker_call_hf_processor_handles_current_qwen2vl_video_processor() -> None:
    processor = object.__new__(MingFlashOmniThinkerMultiModalProcessor)
    processor.info = _Info(_HFProcessor())

    out = processor._call_hf_processor(
        prompt="<VIDEO>",
        mm_data={"videos": _dummy_video()},
        mm_kwargs={},
        tok_kwargs={},
    )

    assert "pixel_values_videos" in out
    assert "video_grid_thw" in out
    assert "pixel_values" not in out
    assert "image_grid_thw" not in out


def test_ming_hf_processor_handles_current_qwen2vl_video_processor() -> None:
    processor = object.__new__(MingFlashOmniProcessor)
    processor.image_processor = Qwen2VLImageProcessor()
    processor.audio_processor = object()
    processor.tokenizer = _Tokenizer()
    processor.spatial_merge_size = 2

    out = processor(
        text="<VIDEO>",
        videos=_dummy_video(),
    )

    assert isinstance(out, BatchFeature)
    assert "pixel_values_videos" in out
    assert "video_grid_thw" in out
    assert "pixel_values" not in out
    assert "image_grid_thw" not in out
