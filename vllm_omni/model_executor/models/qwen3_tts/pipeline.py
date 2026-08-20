# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-TTS pipeline: Talker (text → RVQ codec) → Code2Wav (codec → audio).

Chunked vs end-to-end mode is dispatched from ``deploy.async_chunk``.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
    pipeline_cfg_resolver,
    replace_stage_sampling_constraints,
)

from .configuration_qwen3_tts import Qwen3TTSConfig
from .runtime_config import resolve_qwen3_tts_runtime_config

_PROC = "vllm_omni.model_executor.stage_input_processors.qwen3_tts"

QWEN3_TTS_PIPELINE = PipelineConfig(
    model_type="qwen3_tts",
    default_deploy_config_name="qwen3_tts.yaml",
    # Pipeline-level default; the code2wav stage overrides per-stage below.
    model_arch="Qwen3TTSTalkerForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="qwen3_tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.talker2code2wav_async_chunk"),
            custom_process_next_stage_input_func=f"{_PROC}.talker2code2wav_full_payload",
            sampling_constraints={
                "detokenize": False,
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="Qwen3TTSCode2Wav",
            # ``sync_process_input_func`` is the only input-proc override for
            # this stage in sync (non-async-chunk) mode: a length-only
            # ``_token_only`` placeholder.  The bulk codec payload itself
            # ships via the worker connector from stage 0's
            # ``talker2code2wav_full_payload`` producer.  Under async_chunk
            # mode no pre-stage processing is needed -- chunks deliver
            # directly to the consumer.
            sync_process_input_func=f"{_PROC}.talker2code2wav_token_only",
            sampling_constraints={"detokenize": True},
            extras={"tts_args": {"max_instructions_length": 500}},
        ),
    ),
)


def apply_qwen3_tts_sampling_constraints(
    pipeline: PipelineConfig,
    hf_config: Qwen3TTSConfig,
    *,
    stage_id: int,
) -> PipelineConfig:
    """Apply the Qwen3-TTS Talker's checkpoint-owned sampling contract."""
    runtime = resolve_qwen3_tts_runtime_config(hf_config)
    return replace_stage_sampling_constraints(
        pipeline,
        stage_id=stage_id,
        updates={
            "stop_token_ids": [runtime.codec_stop_token_id],
            "extra_args": runtime.to_sampling_extra_args(),
        },
    )


@pipeline_cfg_resolver(
    config_type=Qwen3TTSConfig,
    default_pipeline_config=QWEN3_TTS_PIPELINE,
)
def resolve_qwen3_tts_pipeline(hf_config: Qwen3TTSConfig) -> PipelineConfig:
    return apply_qwen3_tts_sampling_constraints(
        QWEN3_TTS_PIPELINE,
        hf_config,
        stage_id=0,
    )
