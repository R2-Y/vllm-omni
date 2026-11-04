# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Recover INT4 attention projections from an unquantized parent checkpoint.

The Drama 3 RTN INT4 export can generate speech from the reference text. This
opt-in path re-quantizes the exported attention targets to INT8. Experts and
untargeted projections keep their existing checkpoint methods. It cannot reconstruct lost values
from the INT4 checkpoint itself and requires the matching original parent.
"""

from __future__ import annotations

import json
from pathlib import Path

import regex as re
import torch
from safetensors import safe_open
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase

logger = init_logger(__name__)


def quantize_attention_int8(weight: torch.Tensor, dtype: torch.dtype, group_size: int = 128):
    """Pack symmetric INT8 input-axis words; optimize physical scale rounding."""
    if weight.device.type != "cpu" or weight.ndim != 2:
        raise ValueError("attention quantization requires a CPU matrix")
    n, k = weight.shape
    if dtype not in (torch.float16, torch.bfloat16) or group_size <= 0 or k % group_size or k % 4:
        raise ValueError("attention requires FP16/BF16 scales and aligned input groups")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("attention parent weights must be finite")
    w = weight.float().reshape(n, k // group_size, group_size)
    maximum = w.abs().amax(-1, keepdim=True)
    best_error = torch.full_like(maximum, float("inf"))
    best_scale = torch.ones_like(maximum)
    # The loader may set BF16/FP16 as the surrounding default dtype.
    for clip in torch.linspace(1.0, 0.98, 5, dtype=torch.float32).tolist():
        scale = (maximum * clip / 127).to(dtype)
        scale = scale.masked_fill(scale == 0, 1).float()
        codes = (w / scale).round().clamp(-128, 127)
        decoded = (codes * scale).to(dtype).float()
        error = ((decoded - w) ** 2).sum(-1, keepdim=True)
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_scale = torch.where(better, scale, best_scale)
    codes = ((w / best_scale).round().clamp(-128, 127) + 128).reshape(n, k).long()
    columns = codes.T.contiguous().reshape(k // 4, 4, n)
    words = sum(columns[:, offset, :] << (8 * offset) for offset in range(4))
    packed = (words - (words >= 2**31).long() * 2**32).to(torch.int32)
    return packed, best_scale.squeeze(-1).T.contiguous().to(dtype)


class Int8AttentionProjection(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, device: torch.device, dtype: torch.dtype):
        super().__init__()
        import vllm._custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            MARLIN_MAX_BLOCKS_PER_SM,
            marlin_make_empty,
            marlin_make_workspace_new,
            marlin_permute_scales,
        )

        self.n, self.k = weight.shape
        packed, scales = quantize_attention_int8(weight, dtype)
        self.register_buffer(
            "weight",
            ops.gptq_marlin_repack(packed.to(device), size_k=self.k, size_n=self.n, num_bits=8, is_a_8bit=False),
            persistent=False,
        )
        self.register_buffer(
            "scales",
            marlin_permute_scales(scales.to(device), size_k=self.k, size_n=self.n, group_size=128, is_a_8bit=False),
            persistent=False,
        )
        self.register_buffer("zero", marlin_make_empty(device), persistent=False)
        self.register_buffer("workspace", marlin_make_workspace_new(device, MARLIN_MAX_BLOCKS_PER_SM), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import apply_gptq_marlin_linear
        from vllm.scalar_type import scalar_types

        return apply_gptq_marlin_linear(
            input=x,
            weight=self.weight,
            weight_scale=self.scales,
            weight_zp=self.zero,
            workspace=self.workspace,
            wtype=scalar_types.uint8b128,
            output_size_per_partition=self.n,
            input_size_per_partition=self.k,
            use_fp32_reduce=True,
        )


class RecoveredAttentionMethod(LinearMethodBase):
    """Static linear method, so compilation captures the recovered projections."""

    def __init__(self, original: LinearMethodBase):
        self.original = original

    def create_weights(self, *args, **kwargs):
        return self.original.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.original.process_weights_after_loading(layer)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        output = layer.attention_recovery(x)
        return output if bias is None else output + bias


def install_attention_recovery(model) -> None:
    """Install once after checkpoint load, before quantization packing/graphs."""
    source_name = getattr(model.config, "int4_attention_source", None)
    if source_name is None:
        return
    from vllm.distributed import get_tensor_model_parallel_world_size

    if model.lm_head.device.type != "cuda" or get_tensor_model_parallel_world_size() != 1:
        raise ValueError("int4_attention_source currently requires CUDA tensor parallel size 1")
    quantization = getattr(model.config, "quantization_config", None) or {}
    if quantization.get("quant_method") != "drama3-int4":
        raise ValueError("int4_attention_source applies only to a drama3-int4 checkpoint")
    if not isinstance(source_name, str) or not source_name:
        raise ValueError("int4_attention_source must name the matching local original checkpoint")
    source = Path(source_name)
    if not source.is_dir():
        raise ValueError("int4_attention_source must be an existing local checkpoint directory")
    config = json.loads((source / "config.json").read_text())
    from .configuration_fish_omni import to_qwen3_5_moe_text_config

    parent = to_qwen3_5_moe_text_config(config["text_config"])
    text = model.text_config
    for name in (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "linear_num_key_heads",
        "linear_key_head_dim",
        "linear_num_value_heads",
        "linear_value_head_dim",
        "layer_types",
    ):
        if getattr(parent, name) != getattr(text, name):
            raise ValueError(f"int4_attention_source text_config.{name} differs from the loaded model")
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    targets = tuple(quantization["dense"]["targets"])
    count = 0
    for name, layer in list(model.model.named_modules()):
        if not re.fullmatch(r"layers\.\d+\.(self_attn\.(qkv_proj|o_proj)|linear_attn\.(in_proj_qkvz|out_proj))", name):
            continue
        parent_name, projection = name.rsplit(".", 1)
        parts = {
            "qkv_proj": ("q_proj", "k_proj", "v_proj"),
            "in_proj_qkvz": ("in_proj_qkv", "in_proj_z"),
        }.get(projection, (projection,))
        target_prefix = getattr(layer, "prefix", None) or f"model.{name}"
        target_parent = target_prefix.rsplit(".", 1)[0]
        if not all(any(f"{target_parent}.{part}".endswith(target) for target in targets) for part in parts):
            continue
        pieces = []
        for part in parts:
            key = f"model.language_model.{parent_name}.{part}.weight"
            if key not in index:
                raise ValueError(f"int4_attention_source has no unquantized projection {key}")
            shard = Path(index[key])
            if shard.is_absolute() or ".." in shard.parts:
                raise ValueError("int4_attention_source has an invalid shard path")
            with safe_open(source / shard, framework="pt", device="cpu") as reader:
                pieces.append(reader.get_tensor(key))
        weight = torch.cat(pieces, dim=0).contiguous()
        if weight.shape != (layer.output_size, layer.input_size) or weight.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError(f"int4_attention_source has incompatible unquantized weights for {name}")
        layer.add_module(
            "attention_recovery", Int8AttentionProjection(weight, model.lm_head.device, model.lm_head.dtype)
        )
        layer.quant_method = RecoveredAttentionMethod(layer.quant_method)
        count += 1
    if not count:
        raise ValueError("int4_attention_source found no quantized attention projections")
    logger.info(
        "Recovered %d attention projections as INT8 from %s; experts and untargeted projections unchanged",
        count,
        source,
    )
