# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""higgs-audio v2 pipeline: Talker (text -> 8-codebook codec) -> Code2Wav (codec -> 24 kHz PCM)."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
    get_required_config_field,
    pipeline_cfg_resolver,
    replace_stage_sampling_constraints,
)

from .configuration_higgs_audio_v2 import HiggsAudioV2Config

_PROC = "vllm_omni.model_executor.stage_input_processors.higgs_audio_v2"

HIGGS_AUDIO_V2_PIPELINE = PipelineConfig(
    model_type="higgs_audio_v2",
    default_deploy_config_name="higgs_audio_v2.yaml",
    model_arch="HiggsAudioV2ForConditionalGeneration",
    hf_architectures=("HiggsAudioV2ForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="higgs_audio_v2",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.talker2code2wav_async_chunk"),
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
            model_arch="HiggsAudioV2Code2WavForConditionalGeneration",
            sync_process_input_func=f"{_PROC}.talker2code2wav",
            sampling_constraints={"detokenize": True},
        ),
    ),
)


@pipeline_cfg_resolver(
    config_type=HiggsAudioV2Config,
    default_pipeline_config=HIGGS_AUDIO_V2_PIPELINE,
)
def resolve_higgs_audio_v2_pipeline(hf_config: HiggsAudioV2Config) -> PipelineConfig:
    """Use the checkpoint's sequence and audio-ramp termination tokens."""
    stop_token_ids = [
        get_required_config_field(hf_config, field, expected_type=int, model="higgs_audio_v2")
        for field in ("eos_token_id", "audio_eos_token_id")
    ]
    return replace_stage_sampling_constraints(
        HIGGS_AUDIO_V2_PIPELINE,
        stage_id=0,
        updates={"stop_token_ids": stop_token_ids},
    )
