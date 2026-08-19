# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Covo-Audio pipeline topology (frozen).

Stage 0: fused_thinker_talker — multimodal understanding + interleaved text/audio
         token generation (AR).
Stage 1: code2wav              — audio codes → waveform (generation).
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
    replace_stage_sampling_constraints,
)

from .config_covo_audio import CovoAudioConfig
from .runtime_config import resolve_covo_audio_runtime_config

_PROC = "vllm_omni.model_executor.stage_input_processors.covo_audio"

COVO_AUDIO_PIPELINE = PipelineConfig(
    model_type="covo_audio",
    default_deploy_config_name="covo_audio.yaml",
    model_arch="CovoAudioForConditionalGeneration",
    hf_architectures=("CovoAudioForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="fused_thinker_talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            custom_process_next_stage_input_func=f"{_PROC}.llm2code2wav_full_payload",
            sampling_constraints={
                "detokenize": True,
                "ignore_eos": True,
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
            sync_process_input_func=f"{_PROC}.llm2code2wav_token_only",
            sampling_constraints={"detokenize": False},
        ),
    ),
)


def resolve_covo_audio_pipeline(hf_config) -> PipelineConfig | None:
    if hf_config is None:
        return COVO_AUDIO_PIPELINE
    architectures = set(getattr(hf_config, "architectures", ()) or ())
    if not isinstance(hf_config, CovoAudioConfig) and not architectures.intersection(
        COVO_AUDIO_PIPELINE.hf_architectures
    ):
        return COVO_AUDIO_PIPELINE
    config = hf_config if isinstance(hf_config, CovoAudioConfig) else CovoAudioConfig(**hf_config.to_dict())
    runtime = resolve_covo_audio_runtime_config(config)
    return replace_stage_sampling_constraints(
        COVO_AUDIO_PIPELINE,
        stage_id=0,
        updates={
            "stop_token_ids": [runtime.eos_token_id],
            "extra_args": runtime.to_sampling_extra_args(),
        },
    )


resolve_covo_audio_pipeline.default_pipeline_config = COVO_AUDIO_PIPELINE
