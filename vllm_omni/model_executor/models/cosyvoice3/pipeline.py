# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CosyVoice3 pipeline topology (frozen).

Stage 0: Talker   — text prompt → speech tokens (LLM autoregressive).
Stage 1: Code2Wav — flow-matching decoder → acoustic features → waveform.
  * ``sync_process_input_func`` (``text2flow_token_only``) runs when
    ``deploy.async_chunk=false``: stage 1 allocates placeholder slots while
    bulk tensors arrive via ``text2flow_full_payload`` on the connector.
  * ``async_chunk_process_next_stage_input_func`` runs when
    ``deploy.async_chunk=true``: stage 0 streams codec chunks to stage 1
    through the shared-memory connector.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
    get_required_config_field,
    pipeline_cfg_resolver,
    replace_stage_sampling_constraints,
)
from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

_PROC = "vllm_omni.model_executor.stage_input_processors.cosyvoice3"

COSYVOICE3_PIPELINE = PipelineConfig(
    model_type="cosyvoice3",
    default_deploy_config_name="cosyvoice3.yaml",
    connector_extra_int_minimums=(
        ("codec_chunk_frames", 1),
        ("codec_vocab_size", 1),
        ("codec_pre_lookahead_frames", 0),
        ("codec_max_chunk_frames", 1),
        ("codec_stream_scale_factor", 1),
    ),
    model_arch="CosyVoice3Model",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="cosyvoice3_talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.talker2code2wav_async_chunk"),
            custom_process_next_stage_input_func=f"{_PROC}.text2flow_full_payload",
            sampling_constraints={},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="cosyvoice3_code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="latent",
            sync_process_input_func=f"{_PROC}.text2flow_token_only",
        ),
    ),
)


@pipeline_cfg_resolver(
    config_type=CosyVoice3Config,
    default_pipeline_config=COSYVOICE3_PIPELINE,
)
def resolve_cosyvoice3_pipeline(hf_config: CosyVoice3Config) -> PipelineConfig:
    """Bind the engine stop to the same merged stop logit used by the model."""
    stop_token_id = get_required_config_field(
        hf_config,
        "speech_stop_token_id",
        expected_type=int,
        model="cosyvoice3",
    )
    speech_token_size = get_required_config_field(
        hf_config,
        "speech_token_size",
        expected_type=int,
        model="cosyvoice3",
    )
    model_speech_token_size = get_required_config_field(
        hf_config,
        "llm.speech_token_size",
        expected_type=int,
        model="cosyvoice3",
    )
    model_stop_token_id = get_required_config_field(
        hf_config,
        "llm.eos_token_id",
        expected_type=int,
        model="cosyvoice3",
    )
    if (
        stop_token_id != speech_token_size + 1
        or model_speech_token_size != speech_token_size
        or model_stop_token_id != stop_token_id
    ):
        raise ValueError(
            "Model 'cosyvoice3' has inconsistent speech stop fields: "
            f"speech_token_size={speech_token_size}, speech_stop_token_id={stop_token_id}, "
            f"llm.speech_token_size={model_speech_token_size}, llm.eos_token_id={model_stop_token_id}"
        )
    return replace_stage_sampling_constraints(
        COSYVOICE3_PIPELINE,
        stage_id=0,
        updates={"stop_token_ids": [stop_token_id]},
    )
