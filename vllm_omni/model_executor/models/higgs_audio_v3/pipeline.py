# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""higgs-audio v3 pipeline: Talker (text -> 8-codebook codec) -> Code2Wav (codec -> 24 kHz PCM).

Two delivery modes are wired here:

* ``sync_process_input_func`` runs when the deploy YAML has
  ``async_chunk: false``. The orchestrator collects the entire Stage-0 emit,
  reverts the delay pattern once, and hands a single payload to Stage 1.
* ``async_chunk_process_next_stage_input_func`` runs when the deploy YAML
  has ``async_chunk: true``. Stage 0 dispatches per AR step; the streaming
  adapter buffers raw delay-pattern rows, slides a window with left context
  and right holdback, and emits codec-ready frames per chunk. Stage 1
  trims the overlap on both ends so the client sees a coherent PCM stream.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
    get_required_config_field,
    pipeline_cfg_resolver,
    replace_stage_sampling_constraints,
)
from vllm_omni.transformers_utils.configs.higgs_audio_v3 import HiggsAudioV3Config

_PROC = "vllm_omni.model_executor.stage_input_processors.higgs_audio_v3"

HIGGS_AUDIO_V3_PIPELINE = PipelineConfig(
    model_type="higgs_multimodal_qwen3",
    default_deploy_config_name="higgs_multimodal_qwen3.yaml",
    model_arch="HiggsAudioV3TalkerForConditionalGeneration",
    hf_architectures=("HiggsMultimodalQwen3ForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="higgs_audio_v3",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            sampling_constraints={
                "detokenize": False,
            },
            async_chunk_process_next_stage_input_func=f"{_PROC}.talker2code2wav_async_chunk",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="HiggsAudioV3Code2WavForConditionalGeneration",
            sync_process_input_func=f"{_PROC}.talker2code2wav",
            sampling_constraints={"detokenize": True},
        ),
    ),
)


@pipeline_cfg_resolver(
    config_type=HiggsAudioV3Config,
    default_pipeline_config=HIGGS_AUDIO_V3_PIPELINE,
)
def resolve_higgs_audio_v3_pipeline(hf_config: HiggsAudioV3Config) -> PipelineConfig:
    """Bind safety stops to tokenizer-resolved checkpoint config fields."""
    if hf_config.eos_token_id is None or hf_config.audio_end_token_id is None:
        model_path = get_required_config_field(
            hf_config,
            "_name_or_path",
            expected_type=str,
            model="higgs_audio_v3",
        )
        if not model_path:
            raise ValueError("Model 'higgs_audio_v3' requires a checkpoint path to resolve tokenizer-owned stop IDs")
        hf_config.resolve_special_tokens(model_path)
    stop_token_ids = [
        get_required_config_field(hf_config, field, expected_type=int, model="higgs_audio_v3")
        for field in ("eos_token_id", "audio_end_token_id")
    ]
    return replace_stage_sampling_constraints(
        HIGGS_AUDIO_V3_PIPELINE,
        stage_id=0,
        updates={"stop_token_ids": stop_token_ids},
    )
