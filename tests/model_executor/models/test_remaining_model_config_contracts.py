from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


step_audio2_config = _load_module(
    "remaining_step_audio2_config",
    "vllm_omni/model_executor/models/step_audio2/configuration_step_audio2.py",
)
StepAudio2Config = step_audio2_config.StepAudio2Config


def test_step_audio2_non_default_config_propagates_to_runtime_meta() -> None:
    config = StepAudio2Config.from_hf_config(
        SimpleNamespace(
            step_audio2={
                "input_sample_rate": 22050,
                "output_sample_rate": 32000,
                "audio_start": 120000,
                "audio_patch_token_id": 119999,
                "text_max": 119998,
            }
        )
    )
    runtime = StepAudio2Config.from_mapping(asdict(config))
    assert runtime.input_sample_rate == 22050
    assert runtime.output_sample_rate == 32000
    assert runtime.audio_start == 120000


def test_step_audio2_runtime_meta_rejects_missing_and_invalid_fields() -> None:
    values = asdict(StepAudio2Config())
    values.pop("output_sample_rate")
    with pytest.raises(ValueError, match="output_sample_rate"):
        StepAudio2Config.from_mapping(values)

    values = asdict(StepAudio2Config())
    values["audio_eos"] = values["audio_vocab_size"]
    with pytest.raises(ValueError, match="audio_eos"):
        StepAudio2Config.from_mapping(values)
