# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

# ---------------------------------------------------------------------------
# Architectural constants (confirmed from original config.json)
# ---------------------------------------------------------------------------

LATENT_DIM = 64
PATCH_SIZE = 4
HISTORY_PATCH_SIZE = 32
LLM_HIDDEN_SIZE = 896
LLM_VOCAB_SIZE = 151936
AGGREGATOR_HIDDEN_SIZE = 1024
VAE_PATCH_SIZE = 4
SAMPLE_RATE = 44100
SPEAKER_EMBEDDING_DIM = 192  # CAMPPlus output width and speaker projection input width

# AudioVAE frame/hop geometry (confirmed)
AUDIO_FRAME_HOP = 882  # enc input_dim / hop_size / dec output_dim

# FlowLoss sampling defaults
DEFAULT_CFG = 2.0
DEFAULT_SIGMA = 0.25
DEFAULT_TEMPERATURE = 0.0

# Connector / Stage-2 streaming defaults (runtime tuning)
LATENT_CHUNK_SIZE = 25
INITIAL_LATENT_CHUNK_SIZE = 4
LATENT_LEFT_CONTEXT = 0
MAX_DECODE_STEPS = 200


# ---------------------------------------------------------------------------
# seq_data.extra_data keys
# ---------------------------------------------------------------------------

KEY_LATENT_HISTORY = "ming_latent_history"
KEY_DECODE_STEP = "ming_decode_step"
KEY_LAST_STOP_PROB = "ming_last_stop_prob"
KEY_NEXT_EMBEDS = "ming_next_embeds"
KEY_PROMPT_LATENTS = "ming_prompt_latents"
KEY_SPEAKER_EMBEDDING = "ming_speaker_embedding"
KEY_SPEAKER_WAVEFORM = "ming_speaker_waveform"
KEY_SPEAKER_WAVEFORM_LENGTHS = "ming_speaker_waveform_lengths"
KEY_SPEAKER_SAMPLE_RATES = "ming_speaker_sample_rates"
KEY_REQUEST_ID = "ming_request_id"
KEY_CHUNK_ID = "ming_chunk_id"
KEY_CFG = "ming_cfg"
KEY_SIGMA = "ming_sigma"
KEY_TEMPERATURE = "ming_temperature"
KEY_MAX_DECODE_STEPS = "ming_max_decode_steps"
KEY_MIN_DECODE_STEPS = "ming_min_decode_steps"
KEY_TEXT_MODE = "ming_text_mode"
