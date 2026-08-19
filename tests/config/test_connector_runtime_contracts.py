# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from vllm_omni.config.stage_config import (
    DeployConfig,
    PipelineConfig,
    StageDeployConfig,
    StagePipelineConfig,
    merge_pipeline_deploy,
)
from vllm_omni.model_executor.models.cosyvoice3.pipeline import COSYVOICE3_PIPELINE
from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_sampling_extra_args_are_deep_merged_with_internal_runtime_protected():
    pipeline = PipelineConfig(
        model_type="merge_test",
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="talker",
                sampling_constraints={
                    "extra_args": {
                        "model_runtime": {
                            "internal": {
                                "token_id": 7,
                                "limit": 10,
                            }
                        },
                        "pipeline_option": "kept",
                    }
                },
            ),
        ),
    )
    deploy = DeployConfig(
        stages=[
            StageDeployConfig(
                stage_id=0,
                default_sampling_params={
                    "extra_args": {
                        "model_runtime": {
                            "internal": {"token_id": -1},
                            "user_namespace": {"value": 3},
                        },
                        "user_option": "kept",
                    }
                },
            )
        ]
    )

    params = merge_pipeline_deploy(pipeline, deploy)[0].yaml_extras[
        "default_sampling_params"
    ]

    assert params["extra_args"] == {
        "model_runtime": {
            "internal": {"token_id": 7, "limit": 10},
            "user_namespace": {"value": 3},
        },
        "pipeline_option": "kept",
        "user_option": "kept",
    }


@pytest.mark.parametrize(
    ("pipeline", "missing_field"),
    [
        (QWEN3_TTS_PIPELINE, "ref_code_context_frames"),
        (COSYVOICE3_PIPELINE, "codec_pre_lookahead_frames"),
    ],
)
def test_required_connector_runtime_fields_fail_during_stage_construction(
    pipeline,
    missing_field,
):
    deploy = DeployConfig(
        stages=[
            StageDeployConfig(
                stage_id=0,
                output_connectors={"to_stage_1": "shared"},
            )
        ],
        connectors={
            "shared": {
                "name": "SharedMemoryConnector",
                "extra": {},
            }
        }
    )

    with pytest.raises(ValueError, match=missing_field):
        merge_pipeline_deploy(pipeline, deploy)


def test_invalid_connector_runtime_value_fails_during_stage_construction():
    deploy = DeployConfig(
        stages=[
            StageDeployConfig(
                stage_id=0,
                output_connectors={"to_stage_1": "shared"},
            )
        ],
        connectors={
            "shared": {
                "name": "SharedMemoryConnector",
                "extra": {
                    "codec_chunk_frames": 25,
                    "codec_left_context_frames": 72,
                    "initial_codec_chunk_frames": 1,
                    "ref_code_context_frames": -1,
                },
            }
        }
    )

    with pytest.raises(ValueError, match=r"ref_code_context_frames.*>= 0"):
        merge_pipeline_deploy(QWEN3_TTS_PIPELINE, deploy)


def test_idle_connector_without_contract_fields_is_ignored():
    deploy = DeployConfig(
        stages=[
            StageDeployConfig(
                stage_id=0,
                output_connectors={"to_stage_1": "active"},
            )
        ],
        connectors={
            "active": {
                "name": "SharedMemoryConnector",
                "extra": {
                    "codec_chunk_frames": 25,
                    "codec_left_context_frames": 72,
                    "initial_codec_chunk_frames": 1,
                    "ref_code_context_frames": 72,
                },
            },
            "idle": {
                "name": "SharedMemoryConnector",
                "extra": {"ref_code_context_frames": -1},
            },
        }
    )

    merge_pipeline_deploy(QWEN3_TTS_PIPELINE, deploy)


def test_active_connector_with_invalid_contract_field_fails():
    deploy = DeployConfig(
        stages=[
            StageDeployConfig(
                stage_id=0,
                output_connectors={"to_stage_1": "active"},
            )
        ],
        connectors={
            "active": {
                "name": "SharedMemoryConnector",
                "extra": {
                    "codec_chunk_frames": 25,
                    "codec_left_context_frames": 72,
                    "initial_codec_chunk_frames": 1,
                    "ref_code_context_frames": -1,
                },
            },
            "idle": {
                "name": "SharedMemoryConnector",
                "extra": {},
            },
        },
    )

    with pytest.raises(ValueError, match="active.*ref_code_context_frames"):
        merge_pipeline_deploy(QWEN3_TTS_PIPELINE, deploy)


def test_async_pipeline_requires_an_active_contract_connector():
    deploy = DeployConfig(
        async_chunk=True,
        connectors={
            "idle": {
                "name": "SharedMemoryConnector",
                "extra": {},
            }
        },
    )

    with pytest.raises(ValueError, match="requires an active connector"):
        merge_pipeline_deploy(QWEN3_TTS_PIPELINE, deploy)


@pytest.mark.parametrize(
    "profile",
    [
        "qwen3_tts.yaml",
        "qwen3_tts_high_concurrency.yaml",
        "aura_omni.yaml",
        "cosyvoice3.yaml",
    ],
)
def test_bundled_profiles_satisfy_connector_runtime_contract(profile):
    deploy_dir = Path(__file__).parents[2] / "vllm_omni" / "deploy"
    from vllm_omni.config.stage_config import load_deploy_config

    pipeline = QWEN3_TTS_PIPELINE if "qwen3_tts" in profile else COSYVOICE3_PIPELINE
    if profile == "aura_omni.yaml":
        from vllm_omni.model_executor.models.aura_omni.pipeline import (
            AURA_OMNI_PIPELINE,
        )

        pipeline = AURA_OMNI_PIPELINE

    merge_pipeline_deploy(pipeline, load_deploy_config(deploy_dir / profile))
