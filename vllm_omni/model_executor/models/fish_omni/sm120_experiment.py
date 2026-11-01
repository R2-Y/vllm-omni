# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in Drama 3 SM120 experiment, installed only in the isolated worktree."""

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
    if x.ndim == 2 and x.dtype == torch.bfloat16 and split:
        return dense_one(x, self.weight, split)
    return torch.nn.functional.linear(x, self.weight, self.bias)


class Sm120Linear(torch.nn.Linear):
    forward = _linear_forward


def _install_packed():
    from vllm.model_executor.layers.fused_moe.experts import deep_gemm_moe
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    original = fp8_utils.silu_mul_quant_fp8_packed_triton
    seen = set()

    def tuned(input, group_size=128, output_q=None, clamp_limit=None, alpha=1., beta=0.,
              *, expert_ends=None, expert_alignment=0):
        m, n = input.shape
        # Actual SM120 legacy layout; tuning selected BM=2/4 warps only in
        # the validated padded-row interval. Larger prefills had no gain.
        if (n != 1024 or not 16384 <= m <= 32768 or group_size != 128
                or expert_ends is not None or expert_alignment != 0):
            return original(input, group_size, output_q, clamp_limit, alpha, beta,
                            expert_ends=expert_ends, expert_alignment=expert_alignment)
        if m not in seen:
            logger.info("[SM120PackedLaunch] M=%s N=1024 BM=2 warps=4 legacy=true", m)
            seen.add(m)
        q = output_q if output_q is not None else torch.empty((m, 512), device=input.device, dtype=torch.float8_e4m3fn)
        s = torch.empty((1, triton.cdiv(m, 4) * 4), device=input.device, dtype=torch.int32).T[:m]
        fp8_utils._silu_mul_quant_fp8_packed_kernel[(1, min(triton.cdiv(m, 2), 4096))](
            input, q, s, None, m, input.stride(0), q.stride(0), s.stride(1),
            clamp_limit if clamp_limit is not None else 0., alpha, beta,
            N=n, GROUPS_PER_ROW=4, PACKS_PER_ROW=1, fp8_min=-448., fp8_max=448.,
            GROUP_SIZE=128, PACKS_PER_CTA=1, BLOCK_M=2, HAS_CLAMP=clamp_limit is not None,
            EXPERT_ALIGNMENT=0, num_warps=4, num_stages=2)
        return q, s

    deep_gemm_moe.fused_silu_mul_fp8_quant_packed = tuned


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
    packed = os.getenv("DRAMA3_SM120_PACKED") == "1"
    dense = os.getenv("DRAMA3_SM120_DENSE") == "1"
    if packed or dense:
        # additional_config participates in VllmConfig.compute_hash and hence
        # in aot_compile_hash_factors. Runtime environment flags alone do not.
        config = model.vllm_config
        if not isinstance(config.additional_config, dict):
            raise TypeError("SM120 experiment requires dictionary additional_config")
        config.additional_config["drama3_sm120"] = {"revision": 2, "packed": packed, "dense": dense}
    if os.getenv("DRAMA3_SM120_PACKED") == "1":
        _install_packed()
    if os.getenv("DRAMA3_SM120_DENSE") == "1":
        for name, module in model.named_modules():
            if name.startswith("tower_layers.") and isinstance(module, torch.nn.Linear):
                module.__class__ = Sm120Linear
    if os.getenv("DRAMA3_SM120_NVTX") == "1":
        _install_ranges(model)
