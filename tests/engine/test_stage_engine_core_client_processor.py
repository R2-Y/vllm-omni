# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClientBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _client(processor):
    client = object.__new__(StageEngineCoreClientBase)
    client.custom_process_input_func = processor
    client.requires_multimodal_data = False
    return client


def test_processor_supports_legacy_arbitrarily_named_fourth_positional_parameter():
    received = {}

    def processor(source, prompt, requires_mm, legacy_context):
        received["args"] = (source, prompt, requires_mm, legacy_context)
        return ["ok"]

    result = _client(processor).process_engine_inputs(
        ["source"],
        "prompt",
        streaming_context="stream",
        sampling_params="sampling",
    )

    assert result == ["ok"]
    assert received["args"] == (["source"], "prompt", False, "stream")


def test_processor_supports_varargs_and_kwargs():
    received = {}

    def processor(source, prompt, requires_mm, *args, **kwargs):
        received["args"] = args
        received["kwargs"] = kwargs
        return ["ok"]

    _client(processor).process_engine_inputs(
        [],
        None,
        streaming_context="stream",
        sampling_params="sampling",
    )

    assert received["args"] == ("stream",)
    assert received["kwargs"]["sampling_params"] == "sampling"


def test_processor_supports_new_sampling_params_keyword():
    received = {}

    def processor(source, prompt, requires_mm, *, sampling_params=None):
        received["sampling_params"] = sampling_params
        return ["ok"]

    _client(processor).process_engine_inputs([], sampling_params="sampling")

    assert received["sampling_params"] == "sampling"


def test_processor_receives_explicit_source_and_target_sampling_params():
    received = {}

    def processor(
        source,
        prompt,
        requires_mm,
        *,
        source_sampling_params=None,
        target_sampling_params=None,
        sampling_params=None,
    ):
        received.update(
            source=source_sampling_params,
            target=target_sampling_params,
            legacy=sampling_params,
        )
        return ["ok"]

    _client(processor).process_engine_inputs(
        [],
        source_sampling_params="source",
        target_sampling_params="target",
        sampling_params="legacy-target",
    )

    assert received == {
        "source": "source",
        "target": "target",
        "legacy": "legacy-target",
    }


@pytest.mark.parametrize(
    ("parameter_name", "expected"),
    [
        ("source_sampling_params", "source"),
        ("target_sampling_params", "target"),
        ("sampling_params", "legacy-target"),
    ],
)
def test_processor_supports_sampling_params_as_fourth_positional_parameter(
    parameter_name,
    expected,
):
    received = {}

    if parameter_name == "source_sampling_params":

        def processor(source, prompt, requires_mm, source_sampling_params):
            received["value"] = source_sampling_params
            return ["ok"]

    elif parameter_name == "target_sampling_params":

        def processor(source, prompt, requires_mm, target_sampling_params):
            received["value"] = target_sampling_params
            return ["ok"]

    else:

        def processor(source, prompt, requires_mm, sampling_params):
            received["value"] = sampling_params
            return ["ok"]

    _client(processor).process_engine_inputs(
        [],
        streaming_context="stream",
        source_sampling_params="source",
        target_sampling_params="target",
        sampling_params="legacy-target",
    )

    assert received["value"] == expected


def test_processor_supports_sampling_params_as_fourth_positional_only_parameter():
    received = []

    def source_processor(source, prompt, requires_mm, source_sampling_params, /):
        received.append(source_sampling_params)
        return ["ok"]

    def target_processor(source, prompt, requires_mm, target_sampling_params, /):
        received.append(target_sampling_params)
        return ["ok"]

    def legacy_processor(source, prompt, requires_mm, sampling_params, /):
        received.append(sampling_params)
        return ["ok"]

    for processor in (source_processor, target_processor, legacy_processor):
        _client(processor).process_engine_inputs(
            [],
            streaming_context="stream",
            source_sampling_params="source",
            target_sampling_params="target",
            sampling_params="legacy-target",
        )

    assert received == ["source", "target", "legacy-target"]
