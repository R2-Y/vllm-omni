from types import SimpleNamespace

import pytest
import torch

import vllm_omni.worker.gpu_generation_model_runner as generation_runner_module
from vllm_omni.worker.gpu_generation_model_runner import (
    ExecuteModelState,
    GPUGenerationModelRunner,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyInputBatch:
    def __init__(self):
        self.req_ids = ["req-1"]
        self.req_id_to_index = {"req-1": 0}
        self.num_reqs = 1
        self.vocab_size = 10


def _make_runner(multimodal_outputs):
    runner = object.__new__(GPUGenerationModelRunner)
    runner.execute_model_state = ExecuteModelState(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        multimodal_outputs,
        None,
    )
    runner.kv_connector_output = None
    runner.input_batch = _DummyInputBatch()
    runner.use_async_scheduling = False
    runner.device = torch.device("cpu")
    runner.supports_mm_inputs = False
    runner.speculative_config = None
    runner.routed_experts_initialized = False
    runner._async_chunk = False
    return runner


def test_async_omni_output_guard_uses_generic_model_capability():
    runner = object.__new__(GPUGenerationModelRunner)
    runner.use_async_scheduling = True
    runner.speculative_config = None
    runner.device = torch.device("cuda")
    runner.model_config = SimpleNamespace(async_chunk=True)
    runner.model = SimpleNamespace(use_async_omni_output=True)
    runner._should_accumulate_full_payload_output = lambda: False

    assert runner._should_use_async_omni_output()

    runner.device = torch.device("cpu")
    assert not runner._should_use_async_omni_output()

    runner.device = torch.device("cuda")
    runner._should_accumulate_full_payload_output = lambda: True
    assert not runner._should_use_async_omni_output()


def test_async_sample_tokens_builds_from_immutable_snapshot(monkeypatch):
    runner = _make_runner({"audio": [torch.tensor([1.0, 2.0])]})
    runner.use_async_scheduling = True
    runner.async_output_copy_stream = object()
    runner._should_use_async_omni_output = lambda **_: True
    runner._should_accumulate_full_payload_output = lambda: False
    runner.get_omni_connector_output = lambda: None
    waited = []

    snapshot = SimpleNamespace(
        payload={"audio": [torch.tensor([1.0, 2.0])]},
        wait=lambda: waited.append("wait"),
    )
    monkeypatch.setattr(
        generation_runner_module,
        "snapshot_tensor_payload_to_cpu_async",
        lambda *args, **kwargs: snapshot,
    )
    monkeypatch.setattr(generation_runner_module, "get_or_create_payload_copy_stream", lambda owner: object())

    class FakeAsyncOutput:
        def __init__(self, *, model_runner_output_builder, **kwargs):
            runner.input_batch.req_ids[0] = "mutated"
            runner.input_batch.req_id_to_index = {"mutated": 0}
            self.output = model_runner_output_builder()
            self.kwargs = kwargs

    monkeypatch.setattr(generation_runner_module, "OmniAsyncGPUModelRunnerOutput", FakeAsyncOutput)

    result = GPUGenerationModelRunner.sample_tokens(runner)

    assert waited == ["wait"]
    assert result.output.req_ids == ["req-1"]
    assert result.output.req_id_to_index == {"req-1": 0}
    assert torch.equal(result.output.multimodal_outputs[0]["audio"], torch.tensor([1.0, 2.0]))


def test_mapping_tensor_is_copied_to_cpu_once_per_batch(monkeypatch):
    calls = []
    tensor = torch.tensor([1.0, 2.0])

    def copy_once(value):
        calls.append(value)
        return value

    monkeypatch.setattr(generation_runner_module, "to_cpu_contiguous", copy_once)

    payloads = GPUGenerationModelRunner._build_per_request_payloads({"shared": tensor}, num_reqs=2)

    assert calls == [tensor]
    assert payloads[0]["shared"] is tensor
    assert payloads[1]["shared"] is tensor


def test_sample_tokens_tensor_output():
    multimodal_outputs = torch.randn(1, 2, 3)
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 1
    assert output.multimodal_outputs[0]["model_outputs"].shape == (2, 3)


def test_sample_tokens_list_output():
    multimodal_outputs = [torch.randn(2, 1)]
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 1
    assert output.multimodal_outputs[0]["model_outputs"].shape == (2, 1)


def test_sample_tokens_list_allows_none_output():
    multimodal_outputs = [None]
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 1
    assert output.multimodal_outputs[0]["model_outputs"] is None


def test_sample_tokens_dict_output():
    multimodal_outputs = {"audio": torch.randn(1, 4), "unused": None}
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 1
    assert "audio" in output.multimodal_outputs[0]
    assert "unused" not in output.multimodal_outputs[0]
    assert output.multimodal_outputs[0]["audio"].shape == (1, 4)
