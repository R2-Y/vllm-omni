# Copyright 2026 Tencent.
from collections.abc import Mapping
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.covo_audio.runtime_config import (
    covo_audio_runtime_from_meta,
    covo_audio_runtime_from_sampling_params,
    get_required_covo_audio_runtime_int,
)

logger = init_logger(__name__)

# Per-model REPLACE-keys for the full-payload accumulator (none for covo_audio:
# the producer side does not emit per-step hidden_states / model_outputs;
# llm2code2wav_full_payload reads token_ids directly from `request`).
_FULL_PAYLOAD_REPLACE_KEYS: frozenset[str] = frozenset()


def _filter_audio_codes(token_ids: list[int], *, audio_token_index: int) -> list[int]:
    """Filter codec-range token ids and rebase by checkpoint runtime."""
    audio_codes = [token_id - audio_token_index for token_id in token_ids if token_id >= audio_token_index]
    if not audio_codes:
        audio_codes = [-1]
    return audio_codes


def llm2code2wav_token_only(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    *,
    sampling_params: Any | None = None,
    source_sampling_params: Any | None = None,
    target_sampling_params: Any | None = None,
) -> list[OmniTokensPrompt]:
    """Sync-side placeholder for the non-async-chunk Stage-1 input.

    Returns an OmniTokensPrompt sized to the code2wav stage's expected
    prefill length (one slot per audio code).  The actual codec ids are
    delivered via the worker connector payload built by
    ``llm2code2wav_full_payload``.
    """
    code2wav_inputs: list[OmniTokensPrompt] = []
    for output_wrapper in source_outputs:
        output = output_wrapper.outputs[0]
        mm = getattr(output, "multimodal_output", None)
        mm_meta = mm.get("meta") if isinstance(mm, Mapping) else None
        runtime_params = (
            source_sampling_params
            if source_sampling_params is not None
            else target_sampling_params
            if target_sampling_params is not None
            else sampling_params
        )
        if runtime_params is not None:
            runtime = covo_audio_runtime_from_sampling_params(runtime_params)
        elif isinstance(mm_meta, Mapping) and "model_runtime" in mm_meta:
            runtime = covo_audio_runtime_from_meta(mm_meta)
        else:
            sampling_params = getattr(output_wrapper, "sampling_params", None) or getattr(
                output,
                "sampling_params",
                None,
            )
            runtime = covo_audio_runtime_from_sampling_params(sampling_params)
        audio_token_index = get_required_covo_audio_runtime_int(runtime, "audio_token_index")
        audio_codes = _filter_audio_codes(list(output.token_ids), audio_token_index=audio_token_index)
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * len(audio_codes),
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return code2wav_inputs


def llm2code2wav_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: Any,
) -> dict[str, Any] | None:
    """Producer-side payload builder for the worker connector data plane.

    covo_audio's fused_thinker_talker stage emits codec ids via
    ``request.output_token_ids`` (token-ids only -- no
    hidden_states or embed tensors), so the connector payload is
    just the filtered audio codes plus a finished marker.
    """
    output_token_ids = list(getattr(request, "output_token_ids", None) or [])
    if not output_token_ids:
        logger.warning(
            "covo_audio.llm2code2wav_full_payload: empty output_token_ids for req=%s; consumer wait gate may hang.",
            getattr(request, "request_id", "?"),
        )
        return None
    flat_runtime = pooling_output.get("meta.model_runtime") if isinstance(pooling_output, Mapping) else None
    if isinstance(flat_runtime, Mapping):
        runtime = covo_audio_runtime_from_meta({"model_runtime": flat_runtime})
    else:
        runtime = covo_audio_runtime_from_sampling_params(getattr(request, "sampling_params", None))
    audio_token_index = get_required_covo_audio_runtime_int(runtime, "audio_token_index")
    audio_codes = _filter_audio_codes(output_token_ids, audio_token_index=audio_token_index)
    return {
        "codes": {"audio": audio_codes},
        "meta": {
            "finished": torch.tensor(True, dtype=torch.bool),
            "model_runtime": {"covo_audio": dict(runtime)},
        },
    }
