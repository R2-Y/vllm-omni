from __future__ import annotations

import ast
import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


config_contract = _load_module(
    "remaining_model_config_contract",
    "vllm_omni/model_executor/config_contract.py",
)
get_required_config_field = config_contract.get_required_config_field


def _required_fields(relative_path: str) -> set[str]:
    tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    fields: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "get_required_config_field"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            fields.add(node.args[1].value)
    return fields


def _nested_config(field_path: str, value: object) -> dict[str, object]:
    root: dict[str, object] = {}
    current = root
    parts = field_path.split(".")
    for part in parts[:-1]:
        child: dict[str, object] = {}
        current[part] = child
        current = child
    current[parts[-1]] = value
    return root


@pytest.mark.parametrize(
    ("family", "relative_path", "field_path"),
    [
        (
            "ming_flash",
            "vllm_omni/model_executor/models/ming_flash_omni/"
            "ming_flash_omni_thinker.py",
            "num_query_tokens",
        ),
        (
            "indextts2",
            "vllm_omni/model_executor/models/indextts2/"
            "indextts2_s2mel_decoder.py",
            "semantic_codec.codebook_size",
        ),
        (
            "minicpmo_4_5",
            "vllm_omni/model_executor/models/minicpmo_4_5/"
            "minicpmo_4_5_omni_llm.py",
            "audio_pool_step",
        ),
        (
            "nemotron_voicechat",
            "vllm_omni/model_executor/models/nemotron_voicechat/"
            "nemotron_voicechat_talker.py",
            "target_sample_rate",
        ),
        (
            "personaplex",
            "vllm_omni/model_executor/models/personaplex/personaplex_code2wav.py",
            "sample_rate",
        ),
        (
            "dots_tts",
            "vllm_omni/model_executor/models/dots_tts/dots_tts_talker.py",
            "campplus_embedding_size",
        ),
        (
            "bagel",
            "vllm_omni/model_executor/models/bagel/bagel.py",
            "max_latent_size",
        ),
        (
            "hunyuan_image3",
            "vllm_omni/model_executor/models/hunyuan_image3/hunyuan_image3.py",
            "image_base_size",
        ),
        (
            "mammoth_moda2",
            "vllm_omni/model_executor/models/mammoth_moda2/mammoth_moda2.py",
            "gen_vocab_size",
        ),
        (
            "voxcpm2",
            "vllm_omni/model_executor/models/voxcpm2/voxcpm2_talker.py",
            "sample_rate",
        ),
        (
            "qwen3_code_predictor",
            "vllm_omni/model_executor/models/common/qwen3_code_predictor.py",
            "num_code_groups",
        ),
        (
            "whisper_vq",
            "vllm_omni/model_executor/models/common/whisper_vq.py",
            "quantize_position",
        ),
    ],
)
def test_remaining_family_required_fields_propagate_and_fail_strictly(
    family: str,
    relative_path: str,
    field_path: str,
) -> None:
    assert field_path in _required_fields(relative_path)

    non_default = 777
    assert (
        get_required_config_field(
            _nested_config(field_path, non_default),
            field_path,
            expected_type=int,
            model=family,
        )
        == non_default
    )
    with pytest.raises(ValueError, match=field_path):
        get_required_config_field(
            {},
            field_path,
            expected_type=int,
            model=family,
        )
    with pytest.raises(ValueError, match=field_path):
        get_required_config_field(
            _nested_config(field_path, True),
            field_path,
            expected_type=int,
            model=family,
        )


def test_ming_tts_schema_owns_runtime_token_defaults() -> None:
    source = (
        REPO_ROOT
        / "vllm_omni/model_executor/models/ming_tts/config_ming_tts.py"
    ).read_text(encoding="utf-8")
    for field in (
        "audio_dummy_token_id",
        "audio_start_token_id",
        "audio_end_token_id",
        "audio_eos_token_id",
        "text_eos_token_id",
        "speaker_placeholder_token_id",
        "stop_head_min_steps",
        "stop_head_threshold",
    ):
        assert field in source
    assert "Ming MoE config requires llm_config" in source
    assert "raise ValueError" in source


def test_mimo_vocoder_window_is_schema_owned_and_strict() -> None:
    config_source = (
        REPO_ROOT
        / "vllm_omni/model_executor/models/mimo_audio/config_mimo_audio.py"
    ).read_text(encoding="utf-8")
    runtime_source = (
        REPO_ROOT
        / "vllm_omni/model_executor/models/mimo_audio/mimo_audio_code2wav.py"
    ).read_text(encoding="utf-8")
    assert "vocoder_attn_window_size: list[int] | None = None" in config_source
    assert "def parsed_vocoder_attn_window_size" in config_source
    assert 'getattr(self.config, "vocoder_attn_window_size"' not in runtime_source
    assert "config.parsed_vocoder_attn_window_size()" in runtime_source


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
