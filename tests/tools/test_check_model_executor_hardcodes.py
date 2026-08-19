# SPDX-License-Identifier: Apache-2.0
"""Tests for the model-executor hardcode regression gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "pre_commit"))

from check_model_executor_hardcodes import (  # noqa: E402
    CONFIG_NUMERIC_DEFAULT,
    MODULE_SPECIAL_TOKEN,
    PIPELINE_STOP_TOKEN_IDS,
    VALID_CATEGORIES,
    audit,
    load_baseline,
    main,
    scan_source,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("path", "source", "rule"),
    [
        (
            Path("vllm_omni/model_executor/models/example/pipeline.py"),
            'PIPELINE = {"stop_token_ids": [42]}',
            PIPELINE_STOP_TOKEN_IDS,
        ),
        (
            Path("vllm_omni/model_executor/models/example/model.py"),
            'value = getattr(self.config, "hidden_size", 4096)',
            CONFIG_NUMERIC_DEFAULT,
        ),
        (
            Path("vllm_omni/model_executor/models/example/model.py"),
            'value = getattr(self.config, "windows", [40, 10])',
            CONFIG_NUMERIC_DEFAULT,
        ),
        (
            Path("vllm_omni/model_executor/models/example/model.py"),
            'value = getattr(model_config, "shape", (2, 4))',
            CONFIG_NUMERIC_DEFAULT,
        ),
        (
            Path("vllm_omni/model_executor/models/example/model.py"),
            'value = getattr(model_config, "limits", {"audio": 188})',
            CONFIG_NUMERIC_DEFAULT,
        ),
        (
            Path("vllm_omni/model_executor/models/example/model.py"),
            'value = config.get("sample_rate", 24000)',
            CONFIG_NUMERIC_DEFAULT,
        ),
        (
            Path("vllm_omni/model_executor/models/example/model.py"),
            'value = getattr(codec_cfg, "sample_rates", [16000, 24000])',
            CONFIG_NUMERIC_DEFAULT,
        ),
        (
            Path("vllm_omni/model_executor/stage_input_processors/example.py"),
            "AUDIO_EOS_TOKEN_ID = 32000",
            MODULE_SPECIAL_TOKEN,
        ),
    ],
)
def test_detects_required_hardcode_shapes(path, source, rule):
    findings = scan_source(path, source)
    assert [finding.rule for finding in findings] == [rule]


def test_named_pipeline_stop_token_is_not_a_bare_integer():
    path = Path("vllm_omni/model_executor/models/example/pipeline.py")
    source = 'AUDIO_EOS_TOKEN_ID = lookup()\nPIPELINE = {"stop_token_ids": [AUDIO_EOS_TOKEN_ID]}'
    assert all(finding.rule != PIPELINE_STOP_TOKEN_IDS for finding in scan_source(path, source))


def test_unrelated_numeric_constants_and_defaults_are_ignored():
    path = Path("vllm_omni/model_executor/models/example/model.py")
    source = 'MAX_BATCH_SIZE = 8\nvalue = getattr(runtime, "limit", 4)\nother = metadata.get("limit", 4)'
    assert scan_source(path, source) == []


def test_new_same_pattern_fails_without_allowlist(tmp_path):
    source_path = tmp_path / "vllm_omni/model_executor/models/example/model.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text('value = getattr(config, "hidden_size", 4096)')
    baseline_path = tmp_path / "tools/pre_commit/model_executor_hardcodes_baseline.json"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text(json.dumps({"allowlist": []}))

    errors = audit(tmp_path)

    assert len(errors) == 1
    assert "new config-numeric-default" in errors[0]


def test_current_tree_matches_reviewed_baseline():
    assert main([]) == 0


def test_baseline_uses_only_valid_ownership_categories():
    entries = load_baseline().values()
    categories = {entry["category"] for entry in entries}
    assert categories <= VALID_CATEGORIES
    assert categories == {"algorithm-protocol"}


def test_baseline_protocol_entries_document_rationale():
    for entry in load_baseline().values():
        assert isinstance(entry.get("rationale"), str)
        assert entry["rationale"].strip()
