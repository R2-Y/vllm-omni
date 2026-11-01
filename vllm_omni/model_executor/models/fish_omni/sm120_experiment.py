# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drama 3 SM120 single-row tower GEMM; packed quant tuning is excluded."""

import os

import torch
import triton
import triton.language as tl
from vllm.logger import init_logger

logger = init_logger(__name__)
_dense_shapes = set()


@triton.jit
def _dense(A, W, O, N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr):
    # Only M=1 is selected: larger batches lost to cuBLAS in the shape sweep.
    ni = tl.program_id(0) * 64 + tl.arange(0, 64)
    split = tl.program_id(1)
    ki = split * tl.cdiv(K, SK * 64) * 64 + tl.arange(0, 64)
    mi = tl.arange(0, 16)
    acc = tl.full((16, 64), 0, tl.float32)
    for _ in range(tl.cdiv(K, SK * 64)):
        a = tl.load(A + mi[:, None] * K + ki[None, :], (mi[:, None] == 0) & (ki[None, :] < K), 0)
        w = tl.load(W + ni[None, :] * K + ki[:, None],
                    (ni[None, :] < N) & (ki[:, None] < K), 0)
        acc += tl.dot(a, w)
        ki += 64
    tl.store(O + split * N + mi[:, None] * N + ni[None, :], acc,
             (mi[:, None] == 0) & (ni[None, :] < N))


@torch.library.custom_op("drama3_sm120::dense", mutates_args=())
def dense_one(x: torch.Tensor, weight: torch.Tensor, split: int) -> torch.Tensor:
    n, k = weight.shape
    # Keep the batch dispatch INSIDE the opaque op. A Python M==1 branch in
    # Linear.forward is specialized away when AOT first traces a large prefill.
    if x.shape[0] != 1:
        return torch.nn.functional.linear(x, weight)
    if (n, k) not in _dense_shapes:
        logger.info("[SM120DenseLaunch] M=1 N=%s K=%s split_k=%s", n, k, split)
        _dense_shapes.add((n, k))
    partial = torch.empty((split, n), device=x.device, dtype=torch.float32)
    _dense[(triton.cdiv(n, 64), split)](x, weight, partial, n, k, split, num_warps=4)
    return partial.sum(0).to(x.dtype).unsqueeze(0)


@dense_one.register_fake
def _dense_fake(x, weight, split):
    return x.new_empty((x.shape[0], weight.shape[0]))


def _linear_forward(self, x):
    choices = {(2048, 2048): 4, (9216, 2048): 4, (12288, 2048): 4, (2048, 6144): 8}
    split = choices.get(tuple(self.weight.shape))
    if x.ndim == 2 and x.dtype == torch.bfloat16 and self.bias is None and split:
        return dense_one(x, self.weight, split)
    return torch.nn.functional.linear(x, self.weight, self.bias)


class Sm120Linear(torch.nn.Linear):
    forward = _linear_forward


def _install_ranges(model):
    # Profiling uses eager mode. Hooks are absent in all latency measurements.
    for name, module in model.named_modules():
        if not name or not (isinstance(module, torch.nn.Linear)
                            or any(key in type(module).__name__ for key in ["Linear", "FusedMoE"])):
            continue
        category = "tower" if name.startswith("tower_") else "shared_expert" if "shared_expert" in name else "backbone"
        label = f"drama3/{category}/{name}"
        def before(mod, args, label=label):
            torch.cuda.nvtx.range_push(label)
            context = torch.profiler.record_function(label)
            context.__enter__()
            mod._sm120_profile_context = context
        def after(mod, args, output):
            mod._sm120_profile_context.__exit__(None, None, None)
            torch.cuda.nvtx.range_pop()
        module.register_forward_pre_hook(before)
        module.register_forward_hook(after, always_call=True)


def install(model):
    if torch.cuda.get_device_capability() != (12, 0):
        return
    dense = os.getenv("DRAMA3_SM120_DENSE", "1") == "1"
    if dense:
        # additional_config participates in VllmConfig.compute_hash and hence
        # in aot_compile_hash_factors. Runtime environment flags alone do not.
        config = model.vllm_config
        if not isinstance(config.additional_config, dict):
            raise TypeError("SM120 experiment requires dictionary additional_config")
        config.additional_config["drama3_sm120"] = {"revision": 3, "dense": True}
    if dense:
        for name, module in model.named_modules():
            if name.startswith("tower_layers.") and isinstance(module, torch.nn.Linear):
                module.__class__ = Sm120Linear
    if os.getenv("DRAMA3_SM120_NVTX") == "1":
        _install_ranges(model)
