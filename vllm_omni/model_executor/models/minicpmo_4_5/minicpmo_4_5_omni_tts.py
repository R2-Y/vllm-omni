# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from:
# https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/modeling_minicpmo.py
"""MiniCPM-o 4.5 Talker + Token2Wav: MiniCPMTTS with hidden_text_merge condition.

Pipeline:
  1. Receive thinker hidden_states + full token IDs via additional_information
  2. Extract tts_bos..tts_eos region
  3. Build condition: emb_text(tokens) + projector_semantic(hidden) (hidden_text_merge)
  4. Run MiniCPMTTS.generate() -> discrete audio tokens
  5. Run Token2wav(tokens) -> waveform bytes -> numpy array
"""

import io
import logging
import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import maybe_prefix
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms import current_omni_platform

try:
    from stepaudio2 import Token2wav as _Token2wav

    _stepaudio2_available = True
except ImportError:
    _Token2wav = None
    _stepaudio2_available = False

logger = logging.getLogger(__name__)


def _restore_weight_norm_weight(weight_g: torch.Tensor, weight_v: torch.Tensor) -> torch.Tensor:
    """Materialize ``weight_norm(..., dim=0)`` checkpoint parameters."""
    row_norm = torch.linalg.vector_norm(
        weight_v.float(),
        dim=1,
        keepdim=True,
    ).clamp_min(torch.finfo(torch.float32).tiny)
    return weight_v * (weight_g.to(dtype=weight_v.dtype) / row_norm.to(dtype=weight_v.dtype))


def _apply_repetition_penalty(
    logits: torch.Tensor,
    history: torch.Tensor,
    *,
    penalty: float,
    window_size: int,
) -> torch.Tensor:
    """Match MiniCPMTTS' frequency-aware repetition penalty."""
    if penalty == 1.0 or history.numel() == 0:
        return logits
    recent = history.reshape(-1)[-window_size:].to(device=logits.device, dtype=torch.long)
    frequencies = torch.bincount(recent, minlength=logits.shape[-1]).to(dtype=logits.dtype)
    alpha = torch.pow(torch.as_tensor(penalty, device=logits.device, dtype=logits.dtype), frequencies)
    return torch.where(logits < 0, logits * alpha, logits / alpha)


def _apply_top_k_top_p(
    logits: torch.Tensor,
    *,
    top_k: int | None,
    top_p: float | None,
    min_tokens_to_keep: int = 3,
) -> torch.Tensor:
    """Apply the same candidate floors as the upstream Transformers warpers."""
    filtered = logits.clone()
    vocab_size = filtered.shape[-1]
    # MiniCPM-o's gen_logits() appends TopPLogitsWarper before
    # TopKLogitsWarper. The order is observable for fixed-seed sampling.
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=False, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative_probs <= (1.0 - float(top_p))
        remove[..., -min_tokens_to_keep:] = False
        remove = remove.scatter(-1, sorted_indices, remove)
        filtered.masked_fill_(remove, float("-inf"))
    if top_k is not None and top_k > 0:
        keep = min(vocab_size, max(int(top_k), min_tokens_to_keep))
        threshold = torch.topk(filtered, keep, dim=-1).values[..., -1, None]
        filtered.masked_fill_(filtered < threshold, float("-inf"))
    return filtered


def _install_torchaudio_soundfile_shim() -> None:
    """Monkey-patch torchaudio.load to use soundfile instead of the default
    torchcodec backend, which requires libtorchcodec/ffmpeg shared libs that
    may be missing on the deployment machine."""
    try:
        import torchaudio

        if getattr(torchaudio, "_soundfile_shim_installed", False):
            return
        _orig_load = torchaudio.load

        def _patched_load(uri, *args, **kwargs):
            try:
                return _orig_load(uri, *args, **kwargs)
            except Exception:
                import numpy as _np
                import soundfile as _sf

                data, sr = _sf.read(uri, dtype="float32", always_2d=True)
                wav = torch.from_numpy(_np.ascontiguousarray(data.T))
                return wav, sr

        torchaudio.load = _patched_load
        torchaudio._soundfile_shim_installed = True
        logger.info("Installed torchaudio.load soundfile shim")
    except Exception as _e:
        logger.warning("Could not install torchaudio shim: %s", _e)


_install_torchaudio_soundfile_shim()


class _MiniCPMTTSProjector(nn.Module):
    """Checkpoint-compatible hidden-state projector used by MiniCPMTTS."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(hidden_states)))


class MiniCPMO45OmniTTSForConditionalGeneration(nn.Module, SupportsPP):
    """MiniCPM-o 4.5 Talker: MiniCPMTTS + Token2wav in a single forward pass."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import MiniCPMOConfig

        config: MiniCPMOConfig = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        additional_config = getattr(vllm_config, "additional_config", {}) or {}
        self.continuous_batching = additional_config.get("minicpmo_talker_mode") == "continuous"

        self.tts = None
        self.audio_tokenizer = None
        self._assets_loaded = False
        self._vocoder_loaded = False
        self._batch_stop_logits: torch.Tensor | None = None
        self._request_generators: dict[str, torch.Generator] = {}
        self._vocoder_states: dict[str, dict[str, Any]] = {}
        self._deferred_cleanup_ids: set[str] = set()

        tts_config = getattr(config, "tts_config", None)
        if tts_config is None and getattr(config, "model_type", None) == "minicpmtts":
            tts_config = config
        if tts_config is not None:
            self._tts_config = tts_config
            self._tts_bos_id = getattr(tts_config, "audio_bos_token_id", 151687)
            self._text_eos_id = getattr(tts_config, "text_eos_token_id", 151692)
            self._num_audio_tokens = getattr(tts_config, "num_audio_tokens", 6562)
            self._hidden_size = getattr(tts_config, "hidden_size", 768)
            self._normalize = getattr(tts_config, "normalize_projected_hidden", True)
        else:
            self._tts_config = None

        self.has_preprocess = self.continuous_batching
        self.has_postprocess = False
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("audio_codes", "current"),
            ("audio_codes", "accumulated"),
        }
        if getattr(self, "continuous_batching", False):
            self._init_native_talker(prefix)

    def _init_native_talker(self, prefix: str) -> None:
        if self._tts_config is None:
            raise ValueError("MiniCPM-o continuous Talker requires tts_config")
        cfg = self._tts_config
        if int(getattr(cfg, "num_vq", 1)) != 1:
            raise ValueError(
                "MiniCPM-o continuous Talker currently requires num_vq=1; "
                f"checkpoint reports {getattr(cfg, 'num_vq', None)}"
            )
        llama_config = LlamaConfig(
            vocab_size=32000,
            hidden_size=int(cfg.hidden_size),
            intermediate_size=int(cfg.intermediate_size),
            num_hidden_layers=int(cfg.num_hidden_layers),
            num_attention_heads=int(cfg.num_attention_heads),
            num_key_value_heads=int(cfg.num_key_value_heads),
            hidden_act=getattr(cfg, "hidden_act", "silu"),
            max_position_embeddings=int(cfg.max_position_embeddings),
            rms_norm_eps=float(getattr(cfg, "rms_norm_eps", 1e-6)),
            tie_word_embeddings=False,
        )
        talker_config = self.vllm_config.with_hf_config(llama_config, architectures=["LlamaForCausalLM"])
        talker_config.model_config.hf_text_config = llama_config
        self.tts_model = LlamaModel(
            vllm_config=talker_config,
            prefix=maybe_prefix(prefix, "tts_obj.model"),
        )
        self.emb_text = nn.Embedding(int(cfg.num_text_tokens), int(cfg.hidden_size))
        self.projector_semantic = _MiniCPMTTSProjector(int(cfg.llm_dim), int(cfg.hidden_size))
        self.emb_code = nn.ModuleList(
            [nn.Embedding(int(cfg.num_audio_tokens), int(cfg.hidden_size)) for _ in range(int(cfg.num_vq))]
        )
        self.head_code = nn.ModuleList(
            [nn.Linear(int(cfg.hidden_size), int(cfg.num_audio_tokens), bias=False) for _ in range(int(cfg.num_vq))]
        )
        self.make_empty_intermediate_tensors = self.tts_model.make_empty_intermediate_tensors
        logger.info(
            "MiniCPM-o native Talker initialized: layers=%d hidden=%d audio_vocab=%d num_vq=%d",
            int(cfg.num_hidden_layers),
            int(cfg.hidden_size),
            int(cfg.num_audio_tokens),
            int(cfg.num_vq),
        )

    def _lazy_init_tts(self):
        if self._assets_loaded or self._tts_config is None:
            return
        if getattr(self, "continuous_batching", False):
            self._lazy_init_vocoder()
            self._assets_loaded = True
            return
        try:
            model_path = self.vllm_config.model_config.model
            import os
            import sys

            if model_path not in sys.path:
                sys.path.insert(0, model_path)
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            MiniCPMTTS = get_class_from_dynamic_module("modeling_minicpmo.MiniCPMTTS", model_path)

            # MiniCPMTTS.__init__ reads `config.top_p / top_k / repetition_penalty`
            # directly (modeling_minicpmo.py L4112-4114), but the model repo's
            # config.json `tts_config` block does not declare these fields and
            # PretrainedConfig in recent transformers no longer surfaces
            # generation-style params on `self.config`. Inject the defaults the
            # upstream code itself ships with (modeling_minicpmo.py L2212-2214,
            # L3132-3133) so attribute access does not raise.
            for _attr, _default in (("top_p", 0.8), ("top_k", 100), ("repetition_penalty", 1.02)):
                if not hasattr(self._tts_config, _attr):
                    setattr(self._tts_config, _attr, _default)

            prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            try:
                self.tts_obj = MiniCPMTTS(config=self._tts_config, audio_tokenizer=None)
            finally:
                torch.set_default_dtype(prev_dtype)
            self.emb_text = self.tts_obj.emb_text
            self.projector_semantic = self.tts_obj.projector_semantic

            token2wav_dir = os.path.join(model_path, "assets", "token2wav")
            if os.path.isdir(token2wav_dir):
                if not _stepaudio2_available:
                    raise ImportError(
                        "MiniCPM-o 4.5 token2wav stage requires the `stepaudio2` Python "
                        "module (a MiniCPM-o-flavored Token2wav vocoder, NOT the upstream "
                        "stepfun-ai/Step-Audio2 — the upstream signature does not accept "
                        "n_timesteps and will fail at __init__). Install via:\n"
                        "    pip install 'vllm-omni[minicpmo]'   # recommended, declared as PR extra\n"
                        "Equivalent direct installs of the same `from stepaudio2 import Token2wav`\n"
                        "entry point used by openbmb/MiniCPM-o-4_5/modeling_minicpmo.py:\n"
                        "    pip install stepaudio2-minicpmo     # bare token2wav package\n"
                        "    pip install 'minicpmo-utils[all]'   # MiniCPM-o umbrella (also brings image/video deps)"
                    )
                prev_dtype2 = torch.get_default_dtype()
                torch.set_default_dtype(torch.float32)
                try:
                    # NB: this must be the MiniCPM-o-flavored Token2wav from
                    # the `stepaudio2-minicpmo` PyPI package (or the
                    # `minicpmo-utils[all]` umbrella), not the upstream
                    # `stepfun-ai/Step-Audio2` repo. The MiniCPM-o variant's
                    # __init__ accepts n_timesteps; the upstream signature is
                    # (model_path, float16=False) and will raise
                    # TypeError on n_timesteps. See ImportError message below
                    # for installation guidance.
                    self.audio_tokenizer = _Token2wav(token2wav_dir, float16=False, n_timesteps=10)
                finally:
                    torch.set_default_dtype(prev_dtype2)
                self.tts_obj.audio_tokenizer = self.audio_tokenizer
                logger.info("Loaded Token2wav from %s", token2wav_dir)
            # Only mark init as complete after every step succeeds, so a
            # partial failure leaves the next call free to retry the full
            # init instead of short-circuiting back to a silent empty path.
            self._assets_loaded = True
        except ImportError:
            # Surface missing dependencies directly so users can act on them
            # instead of getting a silent None waveform downstream.
            raise
        except Exception:
            # Re-raise init failures so the server does not return silent audio.
            logger.error("Failed to init 4.5 TTS", exc_info=True)
            raise

    def _lazy_init_vocoder(self) -> None:
        if self._vocoder_loaded:
            return
        model_path = self.vllm_config.model_config.model
        token2wav_dir = os.path.join(model_path, "assets", "token2wav")
        if not os.path.isdir(token2wav_dir):
            raise FileNotFoundError(f"MiniCPM-o Token2Wav assets not found: {token2wav_dir}")
        if not _stepaudio2_available:
            raise ImportError("MiniCPM-o continuous Talker requires the `stepaudio2-minicpmo` package")
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            self.audio_tokenizer = _Token2wav(token2wav_dir, float16=False, n_timesteps=10)
        finally:
            torch.set_default_dtype(previous_dtype)
        self._vocoder_loaded = True
        logger.info("Loaded request-scoped Token2Wav from %s", token2wav_dir)

    def _build_condition_embeddings(
        self,
        tts_token_ids: torch.Tensor,
        tts_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype
        token_ids = tts_token_ids.to(device=device, dtype=torch.long).reshape(-1)
        hidden = tts_hidden_states.to(device=device, dtype=dtype)
        if hidden.shape[0] != token_ids.shape[0] and token_ids.shape[0] != 1:
            raise ValueError(
                "MiniCPM-o Talker condition length mismatch: "
                f"token_ids={token_ids.shape[0]} hidden_states={hidden.shape[0]}"
            )
        text_embeds = self.emb_text(token_ids)
        hidden_embeds = self.projector_semantic(hidden)
        if self._normalize:
            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
        text_eos = self.emb_text(torch.tensor([self._text_eos_id], device=device, dtype=torch.long))
        audio_bos = self.emb_text(torch.tensor([self._tts_bos_id], device=device, dtype=torch.long))
        return torch.cat([text_embeds + hidden_embeds, text_eos, audio_bos], dim=0)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build request-local prefill/decode embeddings for the vLLM runner."""
        del input_embeds
        span_len = int(input_ids.shape[0])
        is_prefill = bool(info_dict.get("_omni_is_prefill", False))
        state = info_dict.get("audio_state")
        first_call = not isinstance(state, dict)

        if is_prefill or first_call:
            token_ids = info_dict.get("tts_token_ids")
            hidden_states = info_dict.get("tts_hidden_states")
            if not isinstance(token_ids, torch.Tensor) or not isinstance(hidden_states, torch.Tensor):
                return (
                    input_ids,
                    self._dummy_hidden_states(input_ids, None, None),
                    {
                        "audio_state": {
                            "step": 0,
                            "finished": True,
                            "invalid_conditioning": True,
                        }
                    },
                )
            full_embeds = self._build_condition_embeddings(token_ids, hidden_states)
            offset = int(info_dict.get("_omni_num_computed_tokens", 0))
            embeds = full_embeds[offset : offset + span_len]
            if embeds.shape[0] != span_len:
                raise ValueError(
                    "MiniCPM-o Talker prefill span exceeds condition: "
                    f"offset={offset} span={span_len} condition={full_embeds.shape[0]}"
                )
            max_tokens = max(2048, min(16384, int(token_ids.numel()) * 10))
            state = {
                "step": 0,
                "max_tokens": max_tokens,
                "finished": False,
            }
            empty_codes = torch.empty(0, dtype=torch.long, device=embeds.device)
            return (
                input_ids,
                embeds,
                {
                    "audio_state": state,
                    "audio_codes": {
                        "current": empty_codes,
                        "accumulated": empty_codes,
                    },
                },
            )

        current = (info_dict.get("audio_codes", {}) or {}).get("current")
        if not isinstance(current, torch.Tensor) or current.numel() != 1:
            raise RuntimeError("MiniCPM-o Talker decode is missing the previous request-local audio code")
        code = current.to(device=self.emb_code[0].weight.device, dtype=torch.long).reshape(1)
        embeds = self.emb_code[0](code)
        return input_ids, embeds, {}

    def _request_generator(self, request_id: str, device: torch.device) -> torch.Generator:
        generator = self._request_generators.get(request_id)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(42)
            self._request_generators[request_id] = generator
        return generator

    def _sample_audio_code(
        self,
        hidden_state: torch.Tensor,
        history: torch.Tensor,
        request_id: str,
    ) -> torch.Tensor:
        logits = self.head_code[0](hidden_state).float() / 0.8
        logits = _apply_repetition_penalty(
            logits,
            history,
            penalty=1.05,
            window_size=16,
        )
        if history.numel() < 50:
            logits[..., self._num_audio_tokens - 1] = float("-inf")
        logits = _apply_top_k_top_p(
            logits,
            top_k=25,
            top_p=0.85,
            min_tokens_to_keep=3,
        )
        probabilities = torch.softmax(logits, dim=-1)
        return torch.multinomial(
            probabilities,
            num_samples=1,
            generator=self._request_generator(request_id, probabilities.device),
        ).reshape(())

    def _run_vocoder_chunk(
        self,
        request_id: str,
        codes: torch.Tensor,
        *,
        last_chunk: bool,
    ) -> torch.Tensor:
        """Run one bounded Token2Wav chunk with request-scoped mutable caches."""
        self._lazy_init_vocoder()
        assert self.audio_tokenizer is not None
        model_path = self.vllm_config.model_config.model
        default_ref = os.path.join(model_path, "assets", "HT_ref_audio.wav")
        prompt_wav_path = default_ref if os.path.exists(default_ref) else None
        state = self._vocoder_states.get(request_id)
        try:
            if state is None:
                stream_cache, hift_cache = self.audio_tokenizer.set_stream_cache(prompt_wav_path)
                state = {
                    "cache": self.audio_tokenizer.cache,
                    "stream_cache": stream_cache,
                    "hift_cache_dict": hift_cache,
                }
            self.audio_tokenizer.cache = state["cache"]
            self.audio_tokenizer.stream_cache = state["stream_cache"]
            self.audio_tokenizer.hift_cache_dict = state["hift_cache_dict"]
            with torch.amp.autocast("cuda", enabled=False):
                waveform = self.audio_tokenizer.stream(
                    codes.detach().to("cpu", dtype=torch.long).tolist(),
                    prompt_wav_path,
                    last_chunk=last_chunk,
                    return_waveform=True,
                )
            state["cache"] = self.audio_tokenizer.cache
            state["stream_cache"] = self.audio_tokenizer.stream_cache
            state["hift_cache_dict"] = self.audio_tokenizer.hift_cache_dict
            if last_chunk:
                self._vocoder_states.pop(request_id, None)
            else:
                self._vocoder_states[request_id] = state
            return torch.as_tensor(np.asarray(waveform).reshape(-1), dtype=torch.float32)
        except Exception:
            self._vocoder_states.pop(request_id, None)
            raise
        finally:
            self.audio_tokenizer.cache = None
            self.audio_tokenizer.stream_cache = None
            self.audio_tokenizer.hift_cache_dict = {}

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs
        infos = kwargs.get("model_intermediate_buffer") or []
        spans = kwargs.get("request_token_spans")
        if spans is None or len(spans) != len(infos):
            raise RuntimeError("MiniCPM-o continuous Talker requires one request_token_span per request")
        sample_eligible = kwargs.get("request_sample_eligible")
        if sample_eligible is None:
            sample_eligible = [True] * len(infos)
        if len(sample_eligible) != len(infos):
            raise RuntimeError(
                "MiniCPM-o continuous Talker received "
                f"{len(sample_eligible)} sampling flags for {len(infos)} requests"
            )

        stop_rows: list[torch.Tensor] = []
        audio_by_request: dict[str, torch.Tensor] = {}
        for index, info in enumerate(infos):
            if not isinstance(info, dict):
                stop_rows.append(hidden.new_tensor([0.0, float("-inf")]))
                continue
            start, end = spans[index]
            end = min(int(end), int(hidden.shape[0]))
            if int(start) >= end:
                stop_rows.append(hidden.new_tensor([0.0, float("-inf")]))
                continue
            state = dict(info.get("audio_state", {}) or {})
            if state.get("invalid_conditioning"):
                stop_rows.append(hidden.new_tensor([float("-inf"), 0.0]))
                continue
            if not sample_eligible[index]:
                # vLLM computes a logit row for incomplete chunked prefills but
                # discards its sampled token. Advancing codec/RNG state here
                # would make output depend on prefill chunking and compaction.
                stop_rows.append(hidden.new_tensor([0.0, float("-inf")]))
                continue
            codes = (info.get("audio_codes", {}) or {}).get("accumulated")
            if not isinstance(codes, torch.Tensor):
                codes = torch.empty(0, dtype=torch.long, device=hidden.device)
            else:
                codes = codes.to(device=hidden.device, dtype=torch.long).reshape(-1)
            request_id = str(info.get("request_id", index))
            sampled = self._sample_audio_code(hidden[end - 1 : end], codes, request_id)
            is_eos = int(sampled.item()) == self._num_audio_tokens - 1
            state["step"] = int(state.get("step", 0)) + 1
            reached_limit = int(state["step"]) >= int(state.get("max_tokens", 2048))
            finished = is_eos or reached_limit
            state["finished"] = finished
            if not is_eos:
                codes = torch.cat([codes, sampled.reshape(1)])
            info["audio_state"] = state
            info["audio_codes"] = {
                "current": sampled.reshape(1),
                "accumulated": codes,
            }
            cursor = int(state.get("vocoder_cursor", 0))
            pending = int(codes.numel()) - cursor
            chunk_size = int(
                getattr(
                    getattr(self, "_tts_config", None),
                    "streaming_audio_chunk_size",
                    50,
                )
            )
            chunk_end: int | None = None
            if finished and pending > 0:
                chunk_end = int(codes.numel())
            elif pending >= chunk_size + 6:
                chunk_end = cursor + chunk_size
            if chunk_end is not None:
                try:
                    waveform = self._run_vocoder_chunk(
                        request_id,
                        codes[cursor:chunk_end],
                        last_chunk=finished,
                    )
                    state["vocoder_cursor"] = chunk_end
                    info["audio_state"] = state
                    if waveform.numel() > 0:
                        audio_by_request[request_id] = waveform
                except Exception:
                    # Keep a single corrupt vocoder stream from aborting every
                    # other request in the same AR scheduler step.
                    logger.exception(
                        "MiniCPM-o Token2Wav failed for request %s; terminating only that request",
                        request_id,
                    )
                    finished = True
                    state["finished"] = True
                    state["vocoder_failed"] = True
                    info["audio_state"] = state
            stop_rows.append(hidden.new_tensor([float("-inf"), 0.0] if finished else [0.0, float("-inf")]))

        self._batch_stop_logits = torch.stack(stop_rows, dim=0) if stop_rows else hidden.new_empty((0, 2))
        multimodal_outputs: dict[str, Any] = {}
        if audio_by_request:
            ready_ids = list(audio_by_request)
            multimodal_outputs = {
                "model_outputs": [
                    audio_by_request[request_id] for request_id in ready_ids
                ],
                "sr": [
                    torch.tensor(24000, dtype=torch.int32) for _ in ready_ids
                ],
                "meta": {
                    "req_id": ready_ids,
                    "sparse_audio": ["1"],
                },
            }
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs=multimodal_outputs,
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        self._deferred_cleanup_ids.update(str(req_id) for req_id in finished_req_ids)

    def _flush_deferred_cleanup(self) -> None:
        for request_id in self._deferred_cleanup_ids:
            self._request_generators.pop(request_id, None)
            self._vocoder_states.pop(request_id, None)
        self._deferred_cleanup_ids.clear()

    def generate_speech(
        self,
        tts_token_ids: torch.Tensor,
        tts_hidden_states: torch.Tensor,
    ) -> np.ndarray | None:
        """Run full 4.5 TTS pipeline using original MiniCPMTTS.generate."""
        self._lazy_init_tts()
        if not hasattr(self, "tts_obj") or self.tts_obj is None:
            logger.warning("generate_speech: tts_obj not initialized")
            return None

        tts = self.tts_obj
        device = tts.emb_text.weight.device
        dtype = tts.emb_text.weight.dtype

        llm_embeds = tts.emb_text(tts_token_ids.to(device))
        hidden_embeds = tts.projector_semantic(tts_hidden_states.to(device=device, dtype=dtype))
        if getattr(tts.config, "normalize_projected_hidden", False):
            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
        tts_embeds = llm_embeds + hidden_embeds

        text_eos = tts.emb_text(torch.tensor([tts.config.text_eos_token_id], device=device, dtype=torch.long))
        audio_bos = tts.emb_text(torch.tensor([tts.audio_bos_token_id], device=device, dtype=torch.long))
        spk_embeds = torch.zeros(0, tts.config.hidden_size, device=device, dtype=tts_embeds.dtype)

        inputs_embeds = torch.cat([spk_embeds, tts_embeds, text_eos, audio_bos], dim=0).unsqueeze(0)
        logger.info("generate_speech: inputs_embeds shape=%s", list(inputs_embeds.shape))

        # Scale max_new_token with input text length to avoid mid-stream truncation on long
        # responses (default 2048 can only cover ~300 text tokens at ~6x audio/text ratio).
        # Empirically 511 text tokens → 1951 audio tokens (~3.8x) finishes cleanly, so use 10x
        # as a safe upper bound with a floor of 2048 and a hard cap of 16384 to bound latency/mem.
        num_text = int(tts_token_ids.shape[-1]) if tts_token_ids.ndim > 0 else 0
        max_new_token = max(2048, min(16384, num_text * 10))

        eos_token = torch.tensor([tts.config.num_audio_tokens - 1], dtype=torch.long, device=device)
        outputs = tts.generate(
            inputs_embeds=inputs_embeds,
            eos_token=eos_token,
            max_new_token=max_new_token,
            show_tqdm=False,
        )
        generated_tokens = outputs.new_ids.squeeze(-1)
        logger.info(
            "generate_speech: generated %d audio tokens (cap=%d, text_tokens=%d)",
            generated_tokens.shape[-1],
            max_new_token,
            num_text,
        )

        if self.audio_tokenizer is None:
            logger.warning("No audio_tokenizer")
            return None

        import torchaudio

        model_path = self.vllm_config.model_config.model
        default_ref = os.path.join(model_path, "assets", "HT_ref_audio.wav")
        prompt_wav_path = default_ref if os.path.exists(default_ref) else None

        _orig_save = torchaudio.save

        def _patched_save(uri, src, sample_rate, **kw):
            kw.pop("backend", None)
            if hasattr(uri, "write"):
                sf.write(uri, src.cpu().numpy().T, sample_rate, format="WAV")
                return
            return _orig_save(uri, src, sample_rate, backend="soundfile", **kw)

        torchaudio.save = _patched_save
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            with torch.amp.autocast("cuda", enabled=False):
                token_list = generated_tokens.squeeze(0).tolist()
                num_tokens = len(token_list)

                # For long outputs, the one-shot vocoder path
                # (Token2wav.__call__ -> flow.inference) runs full O(N^2) self-
                # attention over all audio tokens and OOMs on a 24GB card once
                # N exceeds a few thousand (e.g. 4964 tokens needs ~3GiB for a
                # single attention matmul). Switch to the chunked / streaming
                # vocoder (set_stream_cache + stream) which truncates the flow
                # attention caches to prompt_len + 100 steps on every chunk,
                # keeping peak memory bounded regardless of total length.
                STREAM_THRESHOLD = int(os.environ.get("MINICPMO45_TTS_STREAM_THRESHOLD", "2500"))  # ~100s @ 25Hz
                CHUNK_SIZE = int(os.environ.get("MINICPMO45_TTS_STREAM_CHUNK", "50"))  # ~2s per chunk
                MIN_TAIL = 6  # must exceed flow.pre_lookahead_len (typically 3)

                if num_tokens <= STREAM_THRESHOLD:
                    wav_bytes = self.audio_tokenizer(token_list, prompt_wav_path)
                    waveform, sr = sf.read(io.BytesIO(wav_bytes))
                    waveform = waveform.astype(np.float32)
                else:
                    # Build chunk boundaries, merging a too-small tail into the
                    # previous chunk so every chunk satisfies MIN_TAIL.
                    boundaries = []
                    i = 0
                    while i < num_tokens:
                        end = min(i + CHUNK_SIZE, num_tokens)
                        if 0 < num_tokens - end < MIN_TAIL:
                            end = num_tokens
                        boundaries.append((i, end))
                        i = end

                    logger.info(
                        "generate_speech: streaming vocoder, %d tokens -> %d chunks (chunk=%d)",
                        num_tokens,
                        len(boundaries),
                        CHUNK_SIZE,
                    )

                    stream_cache, hift_cache_dict = self.audio_tokenizer.set_stream_cache(prompt_wav_path)
                    self.audio_tokenizer.stream_cache = stream_cache
                    self.audio_tokenizer.hift_cache_dict = hift_cache_dict

                    try:
                        pieces = []
                        for idx, (s, e) in enumerate(boundaries):
                            is_last = idx == len(boundaries) - 1
                            wav_np = self.audio_tokenizer.stream(
                                token_list[s:e],
                                prompt_wav_path,
                                last_chunk=is_last,
                                return_waveform=True,
                            )
                            pieces.append(np.asarray(wav_np).reshape(-1))
                        waveform = np.concatenate(pieces, axis=0).astype(np.float32)
                        sr = 24000
                    finally:
                        # Free per-request streaming state so the next request starts clean
                        self.audio_tokenizer.stream_cache = None
                        self.audio_tokenizer.hift_cache_dict = {}
        finally:
            torch.set_default_dtype(prev_dtype)
            torchaudio.save = _orig_save

        logger.info("generate_speech: waveform %d samples, sr=%d", waveform.shape[0], sr)
        return waveform

    def _generate_tokens(self, inputs_embeds: torch.Tensor, max_new_token: int = 2048) -> torch.Tensor | None:
        """Autoregressive generation of audio tokens using the TTS LlamaModel."""
        device = inputs_embeds.device
        eos_token = self._num_audio_tokens - 1
        condition_length = inputs_embeds.shape[1]
        num_vq = len(self.emb_code)

        new_tokens = torch.zeros(1, max_new_token, num_vq, device=device, dtype=torch.long)
        past_key_values = None
        finished = False

        for t in range(max_new_token):
            if t == 0:
                emb = inputs_embeds
                position_ids = torch.arange(condition_length, device=device).unsqueeze(0)
            else:
                code_emb = [self.emb_code[q](new_tokens[:, t - 1 : t, q]) for q in range(num_vq)]
                emb = torch.stack(code_emb, -1).sum(-1)
                position_ids = torch.tensor([[condition_length + t - 1]], device=device)

            outputs = self.tts_model(
                inputs_embeds=emb,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            hidden = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

            logits = torch.stack([self.head_code[q](hidden[:, -1]) for q in range(num_vq)], dim=-1)
            logits = logits.float() / 0.8

            if t < 50:
                logits[:, eos_token, :] = -float("inf")

            probs = F.softmax(logits, dim=1)
            idx = torch.multinomial(probs.view(-1, probs.shape[1]), 1).view(1, num_vq)
            new_tokens[:, t] = idx

            if (idx == eos_token).any():
                finished = True
                break

        return new_tokens[:, : t + 1 if finished else t, :]

    def _dummy_hidden_states(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        """Shape-correct zero tensor for vllm KV cache profiling.

        vllm's gpu_model_runner._dummy_run takes forward()'s return value as
        ``hidden_states`` and does ``hidden_states[logit_indices_device]``;
        returning None on the dummy path crashes with
        ``TypeError: 'NoneType' object is not subscriptable``.
        """
        for ref in (input_ids, positions, inputs_embeds):
            if isinstance(ref, torch.Tensor):
                num_tokens = int(ref.shape[0]) if ref.ndim >= 1 else 1
                device = ref.device
                break
        else:
            num_tokens = 1
            device = current_omni_platform.get_torch_device()
        hidden_size = int(getattr(self, "_hidden_size", 768) or 768)
        return torch.zeros((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)

    def _reset_vocoder_request_state(self) -> None:
        """Clear mutable Token2Wav state at request boundaries."""
        if self.audio_tokenizer is None:
            return
        if hasattr(self.audio_tokenizer, "stream_cache"):
            self.audio_tokenizer.stream_cache = None
        if hasattr(self.audio_tokenizer, "hift_cache_dict"):
            self.audio_tokenizer.hift_cache_dict = {}

    def forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        additional_information=None,
        **kwargs,
    ):
        if getattr(self, "continuous_batching", False):
            self._flush_deferred_cleanup()
            if input_ids is None and inputs_embeds is None:
                return self._dummy_hidden_states(input_ids, positions, inputs_embeds)
            return self.tts_model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )

        if additional_information is None:
            additional_information = {}

        tts_token_ids = additional_information.get("tts_token_ids")
        tts_hidden_states = additional_information.get("tts_hidden_states")
        tts_text = additional_information.get("llm_output_text", [""])
        if isinstance(tts_text, list):
            tts_text = tts_text[0] if tts_text else ""

        if tts_token_ids is None or tts_hidden_states is None:
            # KV cache profiling / dummy run path — no real TTS input yet.
            logger.debug("4.5 Talker: dummy forward (missing tts_token_ids/tts_hidden_states)")
            return self._dummy_hidden_states(input_ids, positions, inputs_embeds)

        logger.info("4.5 Talker: generating speech for %d tokens", tts_token_ids.shape[0])
        self._reset_vocoder_request_state()
        try:
            waveform = self.generate_speech(tts_token_ids, tts_hidden_states)
        finally:
            self._reset_vocoder_request_state()
        # Tuple layout: (mel_spec, waveform). 4.5 talker emits only waveform,
        # so mel_spec stays None; the wrapper unpacks in this order and
        # packages the waveform into ``multimodal_outputs["model_outputs"]``.
        if waveform is not None:
            return None, torch.tensor(waveform, dtype=torch.float32)
        return None, None

    def compute_logits(self, hidden_states, *args, **kwargs):
        if getattr(self, "continuous_batching", False):
            if self._batch_stop_logits is None:
                if not isinstance(hidden_states, torch.Tensor):
                    return None
                return torch.zeros(
                    hidden_states.shape[0],
                    2,
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
            logits = self._batch_stop_logits
            self._batch_stop_logits = None
            return logits
        if not isinstance(hidden_states, torch.Tensor):
            return None
        return torch.zeros(
            hidden_states.shape[0],
            2,
            device=hidden_states.device,
            dtype=torch.float32,
        )

    def sample(self, logits, sampling_metadata):
        if not getattr(self, "continuous_batching", False):
            return None
        return Sampler()(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        if getattr(self, "continuous_batching", False):
            return self._load_native_weights(weights)
        loaded = set()
        tts_weights = {}
        for k, v in weights:
            if k.startswith("tts."):
                tts_weights[k.replace("tts.", "", 1)] = v
                # vllm sanity-checks `loaded` against `named_parameters()`.
                # The submodule is attached at `self.tts_obj`, not `self.tts`,
                # so report the loaded name under the on-module path.
                loaded.add(k.replace("tts.", "tts_obj.", 1))

        if tts_weights and self._tts_config is not None:
            self._lazy_init_tts()
            if hasattr(self, "tts_obj") and self.tts_obj is not None:
                missing, unexpected = self.tts_obj.load_state_dict(tts_weights, strict=False)
                if missing:
                    logger.warning("TTS missing keys (%d): %s", len(missing), missing[:5])
                if unexpected:
                    logger.warning("TTS unexpected keys (%d): %s", len(unexpected), unexpected[:5])
                self.tts_obj = self.tts_obj.to("cuda")
                self.emb_text = self.tts_obj.emb_text
                self.projector_semantic = self.tts_obj.projector_semantic
                logger.info("Loaded %d TTS weights, moved to cuda", len(tts_weights))

        return loaded

    def _load_native_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        direct_params = dict(self.named_parameters())
        head_g = head_v = None

        for name, tensor in weights:
            if not name.startswith("tts."):
                continue
            stripped = name[len("tts.") :]
            if stripped.startswith("model."):
                backbone_weights.append((stripped[len("model.") :], tensor))
                continue
            if stripped == "head_code.0.parametrizations.weight.original0":
                head_g = tensor
                continue
            if stripped == "head_code.0.parametrizations.weight.original1":
                head_v = tensor
                continue
            target = stripped
            parameter = direct_params.get(target)
            if parameter is None:
                logger.debug("MiniCPM-o native Talker skipped weight %s", name)
                continue
            parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
            loaded.add(target)

        for name in self.tts_model.load_weights(backbone_weights):
            loaded.add(f"tts_model.{name}")

        if head_g is None or head_v is None:
            raise ValueError("MiniCPM-o checkpoint is missing weight-norm Talker head parameters")
        restored = _restore_weight_norm_weight(head_g, head_v)
        self.head_code[0].weight.data.copy_(
            restored.to(
                device=self.head_code[0].weight.device,
                dtype=self.head_code[0].weight.dtype,
            )
        )
        loaded.add("head_code.0.weight")
        logger.info(
            "Loaded MiniCPM-o native Talker weights: backbone=%d direct=%d",
            len(backbone_weights),
            len(loaded),
        )
        return loaded

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None, **kwargs):
        if hasattr(self, "emb_text") and self.emb_text is not None:
            return self.emb_text(input_ids)
        return torch.zeros(input_ids.shape[0], 1)

    def embed_input_ids(self, input_ids, **kwargs):
        return self.get_input_embeddings(input_ids, **kwargs)
