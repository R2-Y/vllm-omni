"""Shared immutable snapshots for asynchronous Omni output construction."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import torch
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

from vllm_omni.outputs import OmniModelRunnerOutput


def to_cpu_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if tensor.device.type == "cpu":
        return tensor.contiguous()
    return tensor.to("cpu").contiguous()


def _clone_cuda_tensor_payload(value: Any, sources: list[torch.Tensor]) -> Any:
    """Own graph-reused tensors before scheduling copies on another stream."""
    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            cloned = value.detach().clone()
            sources.append(cloned)
            return cloned
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_cuda_tensor_payload(item, sources) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cuda_tensor_payload(item, sources) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cuda_tensor_payload(item, sources) for item in value)
    return value


def _copy_tensor_payload_to_cpu(value: Any, pin_memory: bool) -> Any:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cuda":
            return value
        cpu = torch.empty_like(value, device="cpu", pin_memory=pin_memory)
        cpu.copy_(value, non_blocking=True)
        return cpu
    if isinstance(value, dict):
        return {key: _copy_tensor_payload_to_cpu(item, pin_memory) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_tensor_payload_to_cpu(item, pin_memory) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_tensor_payload_to_cpu(item, pin_memory) for item in value)
    return value


class AsyncCPUPayloadSnapshot:
    def __init__(
        self,
        payload: Any,
        ready_event: torch.cuda.Event | None,
        cuda_sources: list[torch.Tensor],
    ) -> None:
        self.payload = payload
        self._ready_event = ready_event
        self._cuda_sources = cuda_sources
        self._waited = False

    def wait(self) -> None:
        if self._waited:
            return
        if self._ready_event is not None:
            self._ready_event.synchronize()
        self._cuda_sources.clear()
        self._waited = True


def snapshot_tensor_payload_to_cpu_async(
    value: Any,
    *,
    copy_stream: torch.cuda.Stream,
    pin_memory: bool,
) -> AsyncCPUPayloadSnapshot:
    cuda_sources: list[torch.Tensor] = []
    cloned = _clone_cuda_tensor_payload(value, cuda_sources)
    if not cuda_sources:
        return AsyncCPUPayloadSnapshot(cloned, None, cuda_sources)

    source_stream = torch.cuda.current_stream()
    ready_event = torch.cuda.Event()
    with torch.cuda.stream(copy_stream):
        copy_stream.wait_stream(source_stream)
        cpu_payload = _copy_tensor_payload_to_cpu(cloned, pin_memory)
        ready_event.record(copy_stream)
    return AsyncCPUPayloadSnapshot(cpu_payload, ready_event, cuda_sources)


def get_or_create_payload_copy_stream(owner: Any) -> torch.cuda.Stream:
    stream = getattr(owner, "_omni_payload_copy_stream", None)
    if stream is None:
        stream = torch.cuda.Stream()
        owner._omni_payload_copy_stream = stream
    return stream


class OmniAsyncGPUModelRunnerOutput(AsyncGPUModelRunnerOutput):
    def __init__(
        self,
        *,
        model_runner_output_builder: Callable[[], OmniModelRunnerOutput],
        cuda_device: torch.device | int | str | None = None,
        **kwargs: Any,
    ) -> None:
        sampled_token_ids = kwargs.pop("sampled_token_ids")
        logprobs_tensors = kwargs.pop("logprobs_tensors")
        invalid_req_indices = kwargs.pop("invalid_req_indices")
        async_output_copy_stream = kwargs.pop("async_output_copy_stream")
        vocab_size = kwargs.pop("vocab_size")
        routed_experts = kwargs.pop("routed_experts", None)
        kwargs.pop("check_ep_fault", False)
        if kwargs:
            raise TypeError(f"Unexpected OmniAsyncGPUModelRunnerOutput kwargs: {sorted(kwargs)}")

        self._model_runner_output = None
        self._invalid_req_indices = invalid_req_indices
        self.async_copy_ready_event = torch.Event()
        self._sampled_token_ids = sampled_token_ids
        self.vocab_size = vocab_size
        self._logprobs_tensors = logprobs_tensors
        self._routed_experts = routed_experts
        self._has_fault: torch.Tensor | None = None

        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self.sampled_token_ids_cpu = self._sampled_token_ids.to("cpu", non_blocking=True)
            self._logprobs_tensors_cpu = self._logprobs_tensors.to_cpu_nonblocking() if self._logprobs_tensors else None
            self._routed_experts_cpu = (
                self._routed_experts.to_cpu_nonblocking() if self._routed_experts is not None else None
            )
            self.async_copy_ready_event.record()

        self._model_runner_output_builder = model_runner_output_builder
        self._background_exception: BaseException | None = None
        self._background_thread: threading.Thread | None = threading.Thread(
            target=self._build_output_in_background,
            daemon=True,
            name="omni-async-output-builder",
        )
        self._cuda_device = cuda_device
        self._background_thread.start()

    def _build_model_runner_output_once(self) -> None:
        if self._model_runner_output is not None:
            return
        with record_function_or_nullcontext("omni_async_output:get_output/build_model_runner_output"):
            self._model_runner_output = self._model_runner_output_builder()
        self._model_runner_output_builder = None

    def _build_output_in_background(self) -> None:
        try:
            if self._cuda_device is not None:
                torch.cuda.set_device(self._cuda_device)
            self._build_model_runner_output_once()
        except BaseException as exc:  # noqa: BLE001 - re-raised by get_output().
            self._background_exception = exc

    def get_output(self) -> OmniModelRunnerOutput:
        background_thread = getattr(self, "_background_thread", None)
        if background_thread is not None:
            background_thread.join()
            self._background_thread = None
            background_exception = getattr(self, "_background_exception", None)
            if background_exception is not None:
                raise background_exception
        self._build_model_runner_output_once()
        if not hasattr(self, "_has_fault"):
            self._has_fault = None
        with record_function_or_nullcontext("omni_async_output:get_output/finalize_async_sampled_tokens"):
            return super().get_output()
