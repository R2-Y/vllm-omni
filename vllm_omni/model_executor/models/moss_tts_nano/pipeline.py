# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MOSS-TTS-Nano pipeline topology (frozen).

Single-stage AR TTS: text -> speech waveform in one pass. The 0.1B AR LM
and the MOSS-Audio-Tokenizer-Nano codec both run inside
``MossTTSNanoForGeneration.forward()``, which uses the VoxCPM-style
generator pattern (``inference_stream()`` stored per-request; one audio
chunk yielded per forward) to drive progressive streaming through the
AR scheduler.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
    pipeline_cfg_resolver,
    replace_stage_sampling_constraints,
)

from .runtime_config import MossTTSNanoConfig, resolve_moss_tts_nano_runtime_config

MOSS_TTS_NANO_PIPELINE = PipelineConfig(
    model_type="moss_tts_nano",
    default_deploy_config_name="moss_tts_nano.yaml",
    model_arch="MossTTSNanoForCausalLM",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="moss_tts_nano",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            engine_output_type="audio",
            sampling_constraints={
                "detokenize": False,
            },
        ),
    ),
)


@pipeline_cfg_resolver(
    config_type=MossTTSNanoConfig,
    default_pipeline_config=MOSS_TTS_NANO_PIPELINE,
)
def resolve_moss_tts_nano_pipeline(hf_config: MossTTSNanoConfig) -> PipelineConfig:
    runtime = resolve_moss_tts_nano_runtime_config(hf_config)
    return replace_stage_sampling_constraints(
        MOSS_TTS_NANO_PIPELINE,
        stage_id=0,
        updates={"stop_token_ids": [runtime.eos_token_id]},
    )
