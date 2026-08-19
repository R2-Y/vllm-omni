from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

_MISSING = object()
_ConfigFieldT = TypeVar("_ConfigFieldT")


def get_required_config_field(
    config: object,
    field_path: str,
    *,
    expected_type: type[_ConfigFieldT],
    model: str,
) -> _ConfigFieldT:
    """Read a required dotted field from an object/mapping config tree."""
    value: object = config
    for field_name in field_path.split("."):
        if isinstance(value, Mapping):
            value = value.get(field_name, _MISSING)
        else:
            value = getattr(value, field_name, _MISSING)
        if value is _MISSING:
            break

    is_invalid_bool = expected_type is int and isinstance(value, bool)
    if value is _MISSING or value is None or is_invalid_bool or not isinstance(value, expected_type):
        actual = "<missing>" if value is _MISSING else repr(value)
        raise ValueError(
            f"Model {model!r} requires config field {field_path!r} "
            f"to be {expected_type.__name__}; actual value: {actual}"
        )
    return value
