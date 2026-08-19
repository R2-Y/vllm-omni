# Copyright 2026 Tencent.
from typing import Any

from transformers import Qwen2Config
from transformers.configuration_utils import PretrainedConfig


class CovoAudioConfig(Qwen2Config):
    """Stage-0 config extension for Covo tokenizer/runtime boundaries."""

    def __init__(
        self,
        audio_token_index: int = 151671,
        audio_sample_rate: int = 16000,
        max_audio_tokens: int = 188,
        max_audio_seconds: int = 30,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.audio_token_index = audio_token_index
        self.audio_sample_rate = audio_sample_rate
        self.max_audio_tokens = max_audio_tokens
        self.max_audio_seconds = max_audio_seconds

class CovoAudioCode2WavConfig(PretrainedConfig):
    model_type = "covo_audio_code2wav"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wav_input_sr = 24000
        self.n_speakers = 1
        self.n_styles = 1

        self.token2latent = {
            "upsample_factor": 4,
            "token_vocab_size": 16384,
            "token_input_dim": 512,
            "z_dim": 64,
            "spkr_embed_dim": 1024,
            "spkr_mask_ratio": 0.0,
            "bert_mask_rate0": 0.7,
            "bert_mask_rate1": 1.0,
            "random_maskrate": 0.3,
            "cfg_dropout": 0.2,
            "sigma": 1e-05,
            "transformer": {
                "num_layers": 12,
                "hidden_size": 1024,
                "ffn_hidden_size": 4096,
                "num_heads": 16,
                "modulation": True,
                "alibi_bias": False,
                "rotary_bias": True,
                "qk_norm": False,
                "max_position_embeddings": 4096,
                "attn_dropout": 0.0,
                "dropout": 0.1,
            },
        }

        self.wavegan = {
            "type": "BigVGANFlowVAE",
            "upsample_rates": [5, 3, 2, 2, 2, 2],
            "upsample_kernel_sizes": [10, 6, 4, 4, 4, 4],
            "upsample_initial_channel": 1536,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "downsample_rates": [2, 2, 2, 2, 3, 5],
            "downsample_channels": [12, 24, 48, 96, 192, 384, 768],
            "activation": "snakebeta",
            "snake_logscale": True,
            "latent_dim": 64,
            "use_flow": True,
            "use_vae": True,
            "kl_weight": 5,
            "causal": True,
            "flow_hidden_channels": 256,
        }

        self.inference = {
            "s_steps": 10,
            "cfg_alpha": 1.0,
            "rescale_logits": False,
        }
