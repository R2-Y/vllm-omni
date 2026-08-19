# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline registry and factory for vllm-omni.

``OMNI_PIPELINES`` maps each ``model_type`` to either a ``PipelineConfig``
instance or a resolver callable that accepts an optional HF config and returns
a ``PipelineConfig``.

To add a new pipeline:
    1. Define the ``PipelineConfig`` instance as a module-level variable in
       ``vllm_omni/.../pipeline.py``.
    2. If the model needs to support several configurations, e.g., because some
       stages are optional, implement a resolver that consumes the HF config
       and returns a ``PipelineConfig``.
    3. Update the registry to map the key to the new config object (in the case
       of new keys) or to the resolver func.

Out of tree pipeline configs or resolvers can also be registered with register_pipeline.

NOTE: Single-stage diffusion models continue to use the
``_create_default_diffusion_stage_cfg`` fallback in
``async_omni_engine.py``; for now we do not add them to registry.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from transformers import PretrainedConfig
from vllm.logger import init_logger

from vllm_omni.config.stage_config import (
    PipelineConfig,
)
from vllm_omni.diffusion.models.pi0_pipeline_config import PI0_PIPELINE
from vllm_omni.model_executor.models.audex.pipeline import (
    AUDEX_THINKER_ONLY_PIPELINE,
    resolve_audex_s2s_pipeline,
    resolve_audex_tta_pipeline,
    resolve_audex_tts_pipeline,
)
from vllm_omni.model_executor.models.aura_omni.pipeline import (
    resolve_aura_omni_pipeline,
)
from vllm_omni.model_executor.models.bagel.pipeline import (
    BAGEL_PIPELINE,
    BAGEL_SINGLE_STAGE_PIPELINE,
    BAGEL_THINK_PIPELINE,
)
from vllm_omni.model_executor.models.cosyvoice3.pipeline import resolve_cosyvoice3_pipeline
from vllm_omni.model_executor.models.covo_audio.pipeline import resolve_covo_audio_pipeline
from vllm_omni.model_executor.models.dots_tts.pipeline import DOTS_TTS_PIPELINE
from vllm_omni.model_executor.models.dreamzero.pipeline import DREAMZERO_PIPELINE
from vllm_omni.model_executor.models.dynin_omni.pipeline import DYNIN_OMNI_PIPELINE
from vllm_omni.model_executor.models.fish_speech.pipeline import resolve_fish_speech_pipeline
from vllm_omni.model_executor.models.gepard.pipeline import GEPARD_PIPELINE
from vllm_omni.model_executor.models.glm_image.pipeline import GLM_IMAGE_PIPELINE
from vllm_omni.model_executor.models.glm_tts.pipeline import resolve_glm_tts_pipeline
from vllm_omni.model_executor.models.gr00t.pipeline import GR00T_N1D7_PIPELINE
from vllm_omni.model_executor.models.higgs_audio_v2.pipeline import resolve_higgs_audio_v2_pipeline
from vllm_omni.model_executor.models.higgs_audio_v3.pipeline import resolve_higgs_audio_v3_pipeline
from vllm_omni.model_executor.models.hunyuan_image3.pipeline import (
    HUNYUAN_IMAGE3_AR_PIPELINE,
    HUNYUAN_IMAGE3_DIT_PIPELINE,
    HUNYUAN_IMAGE3_PIPELINE,
)
from vllm_omni.model_executor.models.hunyuan_video.pipeline import HUNYUAN_VIDEO_15_PIPELINE
from vllm_omni.model_executor.models.indextts2.pipeline import (
    INDEXTTS2_PIPELINE,
    INDEXTTS25_PIPELINE,
    resolve_indextts2_pipeline,
)
from vllm_omni.model_executor.models.lance.pipeline import LANCE_PIPELINE
from vllm_omni.model_executor.models.mammoth_moda2.pipeline import (
    MAMMOTH_MODA2_AR_PIPELINE,
    MAMMOTH_MODA2_PIPELINE,
)
from vllm_omni.model_executor.models.mimo_audio.pipeline import resolve_mimo_audio_pipeline
from vllm_omni.model_executor.models.ming_flash_omni.pipeline import (
    MING_FLASH_OMNI_IMAGE_PIPELINE,
    MING_FLASH_OMNI_PIPELINE,
    MING_FLASH_OMNI_THINKER_ONLY_PIPELINE,
    MING_FLASH_OMNI_TTS_PIPELINE,
)
from vllm_omni.model_executor.models.ming_tts.pipeline import (
    MING_TTS_MOE_PIPELINE,
    MING_TTS_PIPELINE,
)
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE
from vllm_omni.model_executor.models.minimax_music3.pipeline import MINIMAX_MUSIC3_PIPELINE
from vllm_omni.model_executor.models.moss_tts.pipeline import (
    MOSS_TTS_PIPELINE,
    MOSS_TTS_REALTIME_PIPELINE,
    resolve_moss_tts_local_pipeline,
)
from vllm_omni.model_executor.models.moss_tts_nano.pipeline import resolve_moss_tts_nano_pipeline
from vllm_omni.model_executor.models.nemotron_voicechat.pipeline import (
    NEMOTRON_VOICECHAT_PIPELINE,
)
from vllm_omni.model_executor.models.omnivoice.pipeline import OMNIVOICE_PIPELINE
from vllm_omni.model_executor.models.personaplex.pipeline import PERSONAPLEX_PIPELINE
from vllm_omni.model_executor.models.qwen2_5_omni.pipeline import (
    QWEN2_5_OMNI_THINKER_ONLY_PIPELINE,
    resolve_qwen2_5_omni_pipeline,
)
from vllm_omni.model_executor.models.qwen3_omni.pipeline import resolve_qwen3_omni_pipeline
from vllm_omni.model_executor.models.qwen3_tts.pipeline import resolve_qwen3_tts_pipeline
from vllm_omni.model_executor.models.soulx_singer.pipeline import (
    SOULXSINGER_SVC_PIPELINE,
    SOULXSINGER_SVS_PIPELINE,
)
from vllm_omni.model_executor.models.step_audio2.pipeline import (
    STEP_AUDIO2_ASR_PIPELINE,
    STEP_AUDIO2_PIPELINE,
)
from vllm_omni.model_executor.models.voxcpm2.pipeline import VOXCPM2_PIPELINE
from vllm_omni.model_executor.models.voxtral_tts.pipeline import VOXTRAL_TTS_PIPELINE
from vllm_omni.model_executor.models.wan2_2.pipeline import WAN2_2_TI2V_PIPELINE

logger = init_logger(__name__)

PipelineResolverFunc: TypeAlias = Callable[[PretrainedConfig | None], PipelineConfig | None]

# --- Multi-stage omni pipelines (LLM-centric; audio / video I/O) ---
OMNI_PIPELINES: dict[str, PipelineConfig | PipelineResolverFunc] = {
    "aura_omni": resolve_aura_omni_pipeline,
    "qwen2_5_omni": resolve_qwen2_5_omni_pipeline,
    "qwen2_5_omni_thinker_only": QWEN2_5_OMNI_THINKER_ONLY_PIPELINE,
    "personaplex": PERSONAPLEX_PIPELINE,
    "nemotron_voicechat": NEMOTRON_VOICECHAT_PIPELINE,
    # Alias: lets bare `vllm-omni serve <NVIDIA-NemotronLabs-VoiceChat-11B dir>`
    # auto-detect through the path-basename fallback (the checkpoint config.json
    # has no model_type key).
    "nemotron_labs_voicechat": NEMOTRON_VOICECHAT_PIPELINE,
    "qwen3_omni_moe": resolve_qwen3_omni_pipeline,
    "qwen3_tts": resolve_qwen3_tts_pipeline,
    "step_audio_2": STEP_AUDIO2_PIPELINE,
    "step_audio_2_asr": STEP_AUDIO2_ASR_PIPELINE,
    "covo_audio": resolve_covo_audio_pipeline,
    "bagel": BAGEL_PIPELINE,
    "bagel_think": BAGEL_THINK_PIPELINE,
    "bagel_single_stage": BAGEL_SINGLE_STAGE_PIPELINE,
    "lance": LANCE_PIPELINE,
    "dreamzero": DREAMZERO_PIPELINE,
    "Gr00tN1d7": GR00T_N1D7_PIPELINE,
    "pi0": PI0_PIPELINE,
    "gepard": GEPARD_PIPELINE,
    "glm_image": GLM_IMAGE_PIPELINE,
    "hunyuan_image_3_moe": HUNYUAN_IMAGE3_PIPELINE,
    "hunyuan_image3_ar": HUNYUAN_IMAGE3_AR_PIPELINE,
    "hunyuan_image3_dit": HUNYUAN_IMAGE3_DIT_PIPELINE,
    "hunyuan_video_15": HUNYUAN_VIDEO_15_PIPELINE,
    "wan2_2_ti2v": WAN2_2_TI2V_PIPELINE,
    "voxcpm2": VOXCPM2_PIPELINE,
    "dots_tts": DOTS_TTS_PIPELINE,
    "cosyvoice3": resolve_cosyvoice3_pipeline,
    "audex_tts": resolve_audex_tts_pipeline,
    "audex_tta": resolve_audex_tta_pipeline,
    "audex_thinker_only": AUDEX_THINKER_ONLY_PIPELINE,
    "audex_s2s": resolve_audex_s2s_pipeline,
    # Alias: the Nemotron-Labs-Audex-2B repo-root config.json reports
    # ``model_type: nemotron_labs_audex``; bare ``vllm-omni serve <repo>``
    # auto-detects through it and must land on the default (TTS) pipeline.
    "nemotron_labs_audex": resolve_audex_tts_pipeline,
    "mimo_audio": resolve_mimo_audio_pipeline,
    "ming_tts": MING_TTS_PIPELINE,
    "ming_tts_moe": MING_TTS_MOE_PIPELINE,
    "voxtral_tts": VOXTRAL_TTS_PIPELINE,
    "glm_tts": resolve_glm_tts_pipeline,
    "fish_qwen3_omni": resolve_fish_speech_pipeline,
    "ming_flash_omni": MING_FLASH_OMNI_PIPELINE,
    "ming_flash_omni_tts": MING_FLASH_OMNI_TTS_PIPELINE,
    "ming_flash_omni_thinker_only": MING_FLASH_OMNI_THINKER_ONLY_PIPELINE,
    "ming_flash_omni_image": MING_FLASH_OMNI_IMAGE_PIPELINE,
    "moss_tts_nano": resolve_moss_tts_nano_pipeline,
    "omnivoice": OMNIVOICE_PIPELINE,
    "mammoth_moda2": MAMMOTH_MODA2_PIPELINE,
    "mammoth_moda2_ar": MAMMOTH_MODA2_AR_PIPELINE,
    "moss_tts_delay": MOSS_TTS_PIPELINE,
    "moss_tts_realtime": MOSS_TTS_REALTIME_PIPELINE,
    "moss_tts_local": resolve_moss_tts_local_pipeline,
    "minicpmo_4_5": MINICPMO_4_5_PIPELINE,
    "minimax_music3": MINIMAX_MUSIC3_PIPELINE,
    "higgs_audio_v2": resolve_higgs_audio_v2_pipeline,
    "higgs_multimodal_qwen3": resolve_higgs_audio_v3_pipeline,
    "dynin_omni": DYNIN_OMNI_PIPELINE,
    "indextts2": resolve_indextts2_pipeline,
    "indextts2_5": resolve_indextts2_pipeline,
    "soulxsinger_svc": SOULXSINGER_SVC_PIPELINE,
    "soulxsinger_svs": SOULXSINGER_SVS_PIPELINE,
}

_RESOLVER_DEFAULT_OVERRIDES: dict[str, PipelineConfig] = {
    "indextts2": INDEXTTS2_PIPELINE,
    "indextts2_5": INDEXTTS25_PIPELINE,
}


def register_pipeline(pipeline: PipelineConfig | PipelineResolverFunc, model_type: str | None = None):
    """Register an out of tree pipeline or PipelineResolverFunc to a model_type key.
    If a PipelineConfig is provided, model_type is optional, and pipeline.model_type
    will be used by default. If a callable is provided, model_type must be provided,
    since resolvers can return multiple different PipelineConfigs depending on the
    consumed config.
    """
    errors: list[str] = []
    if isinstance(pipeline, PipelineConfig):
        errors = pipeline.validate()
        model_type = model_type if model_type is not None else pipeline.model_type
    else:
        if model_type is None:
            raise ValueError("Model type must be explicitly provided when registering a pipeline resolver")

    if model_type in OMNI_PIPELINES:
        errors.append(f"Model type {model_type} is already registered; the old mapping will be clobbered")
    if errors:
        logger.warning("Registration for pipeline of type %s produced the following issues: %s", model_type, errors)
    OMNI_PIPELINES[model_type] = pipeline


def resolve_pipeline_config(
    model_type: str,
    hf_config: PretrainedConfig | None = None,
) -> PipelineConfig | None:
    """Resolve a registry key to a concrete pipeline config."""
    if model_type not in OMNI_PIPELINES:
        logger.warning("Model type %s is not registered to OMNI_PIPELINES", model_type)
        return None
    pipeline = OMNI_PIPELINES[model_type]
    if not callable(pipeline):
        return pipeline
    default = _RESOLVER_DEFAULT_OVERRIDES.get(
        model_type,
        getattr(pipeline, "default_pipeline_config", None),
    )
    config_type = getattr(pipeline, "config_type", None)
    if hf_config is None or (config_type is not None and not isinstance(hf_config, config_type)):
        return default
    if model_type in _RESOLVER_DEFAULT_OVERRIDES and getattr(hf_config, "model_type", None) != model_type:
        return default
    resolved = pipeline(hf_config)
    return resolved if resolved is not None else default
