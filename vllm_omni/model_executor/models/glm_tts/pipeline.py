# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-TTS pipeline: Stage 0 (AR) → Stage 1 (DiT)."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
    get_required_config_field,
    pipeline_cfg_resolver,
    replace_stage_sampling_constraints,
)
from vllm_omni.transformers_utils.configs.glm_tts import GLMTTSConfig

_PROC = "vllm_omni.model_executor.stage_input_processors.glm_tts"

GLM_TTS_PIPELINE = PipelineConfig(
    model_type="glm_tts",
    default_deploy_config_name="glm_tts.yaml",
    model_arch="GLMTTSForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="glm_tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.ar_to_dit_async_chunk"),
            sampling_constraints={},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="glm_tts_dit",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="latent",
            sync_process_input_func=f"{_PROC}.ar_to_dit",
        ),
    ),
)


@pipeline_cfg_resolver(
    config_type=GLMTTSConfig,
    default_pipeline_config=GLM_TTS_PIPELINE,
)
def resolve_glm_tts_pipeline(hf_config: GLMTTSConfig) -> PipelineConfig:
    stop_token_id = get_required_config_field(
        hf_config,
        "eoa_token_id",
        expected_type=int,
        model="glm_tts",
    )
    if stop_token_id < 0:
        raise ValueError(
            "Model 'glm_tts' requires tokenizer-resolved config field 'eoa_token_id'; "
            "populate it through the existing GLM tokenizer/config loading path"
        )
    return replace_stage_sampling_constraints(
        GLM_TTS_PIPELINE,
        stage_id=0,
        updates={"stop_token_ids": [stop_token_id]},
    )
