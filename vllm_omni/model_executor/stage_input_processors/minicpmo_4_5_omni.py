# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for MiniCPM-o 4.5: Thinker (LLM) -> Talker (TTS).

This is the original vLLM-Omni bridge: it converts the thinker stage's
hidden states + token ids into the talker stage's prompt payload. The
talker model itself is adapted from openbmb/MiniCPM-o-4_5 (see the headers
on vllm_omni/model_executor/models/minicpmo_4_5/*.py).
"""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)

_MINICPMO45_ASYNC_STATE = "_minicpmo45_async_codec_state"
_MINICPMO45_REGISTRY = "_minicpmo45_async_codec_registry"
_MINICPMO45_SILENCE_CODE = 4218


def _payload_value(payload: Any, key: str) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(key)
    return getattr(payload, key, None)


def _codec_config(transfer_manager: Any) -> tuple[int, int]:
    connector = getattr(transfer_manager, "connector", None)
    raw_config = getattr(connector, "config", {}) or {}
    config = raw_config.get("extra", raw_config) if isinstance(raw_config, dict) else {}
    config = config if isinstance(config, dict) else {}
    chunk_frames = int(config.get("codec_chunk_frames", 25))
    left_context_frames = int(config.get("codec_left_context_frames", 3))
    if chunk_frames <= 0 or left_context_frames < 0:
        raise ValueError(
            "Invalid MiniCPM-o codec chunk config: "
            f"codec_chunk_frames={chunk_frames}, "
            f"codec_left_context_frames={left_context_frames}"
        )
    return chunk_frames, left_context_frames


def _codec_scalars(value: Any) -> list[int]:
    """Normalize one request-routed codec delta to CPU scalar token IDs."""
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return []
        return value.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        scalars: list[int] = []
        for item in value:
            scalars.extend(_codec_scalars(item))
        return scalars
    if isinstance(value, (int, bool)):
        return [int(value)]
    raise TypeError(f"Unsupported MiniCPM-o codec delta type: {type(value).__name__}")


def _extract_codec_delta(pooling_output: Any, request_id: str) -> list[int]:
    if pooling_output is None:
        return []
    if isinstance(pooling_output, Mapping):
        meta = pooling_output.get("meta")
        routed_id = meta.get("request_id") if isinstance(meta, Mapping) else None
        routed_id = routed_id or pooling_output.get("request_id")
        if routed_id is not None and str(routed_id) != request_id:
            return []
        codes = pooling_output.get("codes")
        audio = codes.get("audio") if isinstance(codes, Mapping) else None
        return _codec_scalars(audio)
    if isinstance(pooling_output, Sequence) and not isinstance(
        pooling_output,
        (str, bytes, bytearray),
    ):
        delta: list[int] = []
        for item in pooling_output:
            values = _extract_codec_delta(item, request_id) if isinstance(item, Mapping) else _codec_scalars(item)
            delta.extend(values)
        return delta
    return _codec_scalars(pooling_output)


def _drop_codec_state(transfer_manager: Any, request_id: str) -> None:
    request_payload = getattr(transfer_manager, "request_payload", None)
    if isinstance(request_payload, dict):
        container = request_payload.get(request_id)
        if isinstance(container, dict):
            container.pop(_MINICPMO45_ASYNC_STATE, None)
            if not container:
                request_payload.pop(request_id, None)
        else:
            request_payload.pop(request_id, None)
    code_accumulators = getattr(transfer_manager, "code_prompt_token_ids", None)
    if hasattr(code_accumulators, "pop"):
        code_accumulators.pop(request_id, None)


def _is_aborted(request: Any) -> bool:
    status_name = getattr(getattr(request, "status", None), "name", "")
    return any(marker in status_name for marker in ("ABORT", "CANCEL", "IGNORED", "ERROR"))


def tts2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: Any,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Stream request-owned MiniCPM-o codec windows to Code2Wav."""
    external_id = getattr(request, "external_req_id", None)
    internal_id = getattr(request, "request_id", None)
    request_id = str(external_id if external_id is not None else internal_id)
    internal_id = str(internal_id if internal_id is not None else request_id)

    registry = getattr(transfer_manager, _MINICPMO45_REGISTRY, None)
    if registry is None:
        registry = {}
        setattr(transfer_manager, _MINICPMO45_REGISTRY, registry)
    record = registry.get(request_id)
    if record is None:
        record = {
            "internal_id": internal_id,
            "cache_epoch": 0,
            "retired_internal_ids": set(),
        }
        registry[request_id] = record
    elif internal_id in record["retired_internal_ids"]:
        return None
    elif record["internal_id"] != internal_id:
        record["retired_internal_ids"].add(record["internal_id"])
        record["internal_id"] = internal_id
        record["cache_epoch"] = int(record["cache_epoch"]) + 1
        _drop_codec_state(transfer_manager, request_id)

    if _is_aborted(request):
        record["retired_internal_ids"].add(internal_id)
        _drop_codec_state(transfer_manager, request_id)
        return None

    request_payload = getattr(transfer_manager, "request_payload", None)
    if request_payload is None:
        request_payload = {}
        transfer_manager.request_payload = request_payload
    container = request_payload.get(request_id)
    if not isinstance(container, dict):
        container = {}
        request_payload[request_id] = container
    state = container.get(_MINICPMO45_ASYNC_STATE)
    if not isinstance(state, dict):
        state = {
            "internal_id": internal_id,
            "pending": [],
            "left_context": [],
            "codec_end": 0,
        }
        container[_MINICPMO45_ASYNC_STATE] = state

    pending = state["pending"]
    pending.extend(_extract_codec_delta(multimodal_output, request_id))
    request_finished = getattr(request, "is_finished", None)
    finished = bool(is_finished or (callable(request_finished) and request_finished()))
    chunk_frames, left_context_frames = _codec_config(transfer_manager)
    if not finished and len(pending) < chunk_frames:
        return None

    new_token_count = len(pending) if finished else chunk_frames
    new_codes = pending[:new_token_count]
    del pending[:new_token_count]
    codec_start = int(state["codec_end"])
    codec_end = codec_start + new_token_count

    if new_token_count:
        if codec_start == 0:
            context = [_MINICPMO45_SILENCE_CODE] * left_context_frames
        else:
            context = list(state["left_context"])
        output_codes = [*context, *new_codes]
        history = [*state["left_context"], *new_codes]
        state["left_context"] = history[-left_context_frames:] if left_context_frames else []
    elif finished and codec_start > 0 and state["left_context"]:
        # A non-final flow chunk holds its final lookahead tokens. Even when
        # the generated-code boundary is exact, those tokens still need a
        # final ``last_chunk=True`` decode to flush the encoder/HiFT tail.
        context = list(state["left_context"])
        output_codes = context
    else:
        context = []
        output_codes = []
    state["codec_end"] = codec_end

    last_chunk = bool(finished and not pending)
    if last_chunk:
        record["retired_internal_ids"].add(internal_id)
        _drop_codec_state(transfer_manager, request_id)

    chunk_seq = int(getattr(transfer_manager, "put_req_chunk", {}).get(request_id, 0))
    finished_tensor = torch.tensor(last_chunk, dtype=torch.bool)
    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.tensor(output_codes, dtype=torch.long)),
        meta=MetaStruct(
            request_id=request_id,
            chunk_seq=chunk_seq,
            cache_epoch=int(record["cache_epoch"]),
            codec_start=codec_start,
            codec_end=codec_end,
            new_token_count=new_token_count,
            code_flat_numel=len(output_codes),
            codec_chunk_frames=new_token_count,
            codec_left_context_frames=len(context),
            left_context_size=len(context),
            last_chunk=last_chunk,
            stream_finished=finished_tensor,
            finished=finished_tensor,
            req_id=[request_id],
        ),
        request_id=request_id,
    )


def llm2tts(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | dict | list | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
):
    """Convert thinker stage output to talker stage input for MiniCPMO Omni.

    The signature matches the framework's ``custom_process_input_func`` call
    convention used by ``StageEngineCoreClientBase.process_engine_inputs``:

        (source_outputs, prompt, requires_multimodal_data, streaming_context)

    ``source_outputs`` is the already-resolved list of upstream engine
    outputs (one entry per request), so we do not need to look anything up
    via ``stage_list[source_stage_id].engine_outputs``.

    Extracts from thinker output:
      - Full hidden states (prompt + generated) for speaker embedding extraction
      - Prompt token IDs (for finding spk_bos/spk_eos positions)
      - Generated token IDs (for decoding TTS text)

    The talker model will:
      1. Find <|spk_bos|>/<|spk_eos|> positions in prompt_token_ids
      2. Extract speaker embedding from hidden states at those positions
      3. Decode generated text and extract TTS content
      4. Run ConditionalChatTTS pipeline
    """
    del streaming_context  # not used by MiniCPM-o 4.5 turn-taking pipeline

    if not source_outputs:
        raise ValueError("source_outputs cannot be empty")

    llm_outputs = source_outputs
    tts_inputs = []

    if not isinstance(prompt, list):
        prompt = [prompt]

    multi_modal_data = {
        llm_output.request_id: p.get("multi_modal_data", None) if isinstance(p, dict) else None
        for llm_output, p in zip(llm_outputs, prompt)
    }

    for i, llm_output in enumerate(llm_outputs):
        output = llm_output.outputs[0]
        prompt_token_ids = llm_output.prompt_token_ids
        llm_output_ids = output.token_ids
        prompt_token_ids_len = len(prompt_token_ids)

        multimodal_output = getattr(llm_output, "multimodal_output", None)
        if multimodal_output is None:
            multimodal_output = getattr(output, "multimodal_output", None)
        latent = _payload_value(multimodal_output, "latent")
        if latent is None:
            latent = getattr(llm_output, "hidden_states", None)
        if latent is None:
            latent = getattr(output, "hidden_states", None)
            if latent is None:
                payload_keys = (
                    sorted(multimodal_output)
                    if isinstance(multimodal_output, Mapping)
                    else [
                        key
                        for key in ("latent", "hidden", "hidden_states")
                        if _payload_value(multimodal_output, key) is not None
                    ]
                )
                raise ValueError(
                    "No latent or hidden_states found in thinker output; "
                    f"request_output={type(llm_output).__name__}, "
                    f"completion_output={type(output).__name__}, "
                    f"multimodal_output={type(multimodal_output).__name__}, "
                    f"payload_keys={payload_keys}"
                )

        thinker_hidden_states = latent.clone().detach()

        # Split hidden states: prompt portion has speaker embedding,
        # generated portion has the text content
        prompt_hidden = thinker_hidden_states[:prompt_token_ids_len].to(torch.float32)

        # Extract decoded text from thinker output for TTS text extraction
        thinker_text = getattr(output, "text", "") or ""

        # Build full token sequence and extract TTS region
        full_token_ids = list(prompt_token_ids) + (
            list(llm_output_ids) if not isinstance(llm_output_ids, list) else llm_output_ids
        )
        full_hidden = thinker_hidden_states.to(torch.float32)

        # Detect TTS token IDs (4.5: 151703/151704, 2.6: 151691/151692)
        tts_bos_id, tts_eos_id = 151691, 151692
        for _id in [151703, 151704]:
            if _id in full_token_ids:
                tts_bos_id, tts_eos_id = 151703, 151704
                break

        tts_bos_idx = tts_eos_idx = None
        for idx_t, tid in enumerate(full_token_ids):
            if tid == tts_bos_id:
                tts_bos_idx = idx_t + 1
            elif tid == tts_eos_id:
                tts_eos_idx = idx_t

        tts_token_ids_slice = tts_hidden_slice = None
        if tts_bos_idx is not None and full_hidden.shape[0] > tts_bos_idx:
            end_idx = tts_eos_idx if tts_eos_idx is not None else full_hidden.shape[0]
            tts_token_ids_slice = torch.tensor(full_token_ids[tts_bos_idx:end_idx], dtype=torch.long)
            tts_hidden_slice = full_hidden[tts_bos_idx:end_idx]

        additional_information = {
            "request_id": llm_output.request_id,
            "prompt_embeds": prompt_hidden,
            "prompt_token_ids": list(prompt_token_ids),
            "llm_output_token_ids": list(llm_output_ids) if not isinstance(llm_output_ids, list) else llm_output_ids,
            "llm_output_text": [thinker_text],
        }
        if tts_token_ids_slice is not None:
            additional_information["tts_token_ids"] = tts_token_ids_slice
        if tts_hidden_slice is not None:
            additional_information["tts_hidden_states"] = tts_hidden_slice

        # The native Talker replaces these token embeddings in preprocess(), but
        # the prompt length must still reserve exactly one KV position per
        # conditioning row: TTS text + text_eos + audio_bos.
        condition_length = (
            max(
                int(tts_token_ids_slice.numel()),
                int(tts_hidden_slice.shape[0]),
            )
            + 2
            if isinstance(tts_token_ids_slice, torch.Tensor) and isinstance(tts_hidden_slice, torch.Tensor)
            else 1
        )
        tts_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * condition_length,
                additional_information=additional_information,
                multi_modal_data=(
                    multi_modal_data[llm_output.request_id]
                    if requires_multimodal_data and multi_modal_data.get(llm_output.request_id) is not None
                    else None
                ),
                mm_processor_kwargs=None,
            )
        )

    logger.info(
        "MiniCPM-o llm2tts batch: requests=%d request_ids=%s tts_token_counts=%s",
        len(tts_inputs),
        [llm_output.request_id for llm_output in llm_outputs],
        [
            int(tts_input["additional_information"].get("tts_token_ids", torch.empty(0)).numel())
            for tts_input in tts_inputs
        ],
    )

    return tts_inputs
