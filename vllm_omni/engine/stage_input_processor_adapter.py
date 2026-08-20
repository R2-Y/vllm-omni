# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-independent invocation adapter for inter-stage input processors."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any


@functools.cache
def _processor_parameters(
    processor: Callable[..., Any],
) -> dict[str, inspect.Parameter]:
    """Inspect a processor once instead of once per streamed chunk."""
    return dict(inspect.signature(processor).parameters)


def invoke_stage_input_processor(
    processor: Callable[..., Any],
    source_outputs: list[Any],
    prompt: Any,
    requires_multimodal_data: bool,
    *,
    streaming_context: Any | None = None,
    sampling_params: Any | None = None,
    source_sampling_params: Any | None = None,
    target_sampling_params: Any | None = None,
) -> Any:
    """Invoke a processor across legacy and sampling-aware signatures."""
    parameters = _processor_parameters(processor)
    has_varargs = any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters.values())
    has_varkw = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    extra_args: list[Any] = []
    extra_kwargs: dict[str, Any] = {}
    target_params = target_sampling_params if target_sampling_params is not None else sampling_params
    supplied_values = {
        "streaming_context": streaming_context,
        "_streaming_context": streaming_context,
        "sampling_params": sampling_params,
        "source_sampling_params": source_sampling_params,
        "target_sampling_params": target_params,
    }
    positional = [
        parameter
        for parameter in parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    handled: set[str] = set()
    context_supplied = False

    # Positional-only parameters cannot be supplied by keyword. Fill any such
    # slots after the three stable base arguments in declaration order.
    for index, parameter in enumerate(positional[3:], start=3):
        if parameter.kind is not inspect.Parameter.POSITIONAL_ONLY:
            break
        if parameter.name in supplied_values:
            extra_args.append(supplied_values[parameter.name])
            handled.add(parameter.name)
            context_supplied = context_supplied or parameter.name in {
                "streaming_context",
                "_streaming_context",
            }
        elif index == 3:
            extra_args.append(streaming_context)
            context_supplied = True
        elif parameter.default is not inspect.Parameter.empty:
            extra_args.append(parameter.default)
        else:
            break

    fourth = positional[3] if len(positional) >= 4 else None
    if fourth is not None and fourth.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
        if fourth.name in supplied_values:
            extra_kwargs[fourth.name] = supplied_values[fourth.name]
            handled.add(fourth.name)
            context_supplied = context_supplied or fourth.name in {
                "streaming_context",
                "_streaming_context",
            }
        else:
            extra_args.append(streaming_context)
            context_supplied = True
    elif fourth is None:
        if has_varargs:
            extra_args.append(streaming_context)
            context_supplied = True
        elif has_varkw:
            extra_kwargs["streaming_context"] = streaming_context
            context_supplied = True

    declared_context_name = next(
        (name for name in ("streaming_context", "_streaming_context") if name in parameters),
        None,
    )
    if not context_supplied and declared_context_name is not None:
        parameter = parameters[declared_context_name]
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            extra_kwargs[declared_context_name] = streaming_context
            handled.add(declared_context_name)
    elif not context_supplied and has_varkw:
        extra_kwargs["streaming_context"] = streaming_context

    for name in (
        "sampling_params",
        "source_sampling_params",
        "target_sampling_params",
    ):
        value = supplied_values[name]
        if name in handled:
            continue
        parameter = parameters.get(name)
        if parameter is not None:
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                extra_kwargs[name] = value
        elif has_varkw:
            extra_kwargs[name] = value

    return processor(
        source_outputs,
        prompt,
        requires_multimodal_data,
        *extra_args,
        **extra_kwargs,
    )
