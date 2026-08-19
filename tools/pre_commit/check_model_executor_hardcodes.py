#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reject new model-executor hardcodes while preserving a reviewed legacy baseline.

The audit is source-only so it can run in pre-commit without importing vLLM.
Each accepted finding has a stable, line-number-independent fingerprint and an
ownership classification in ``model_executor_hardcodes_baseline.json``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = Path("tools/pre_commit/model_executor_hardcodes_baseline.json")
AUDIT_ROOTS = (
    Path("vllm_omni/model_executor/models"),
    Path("vllm_omni/model_executor/stage_input_processors"),
)
VALID_CATEGORIES = {
    "config-owned",
    "tokenizer-owned",
    "algorithm-protocol",
    "request-runtime",
    "dead-constant",
}

PIPELINE_STOP_TOKEN_IDS = "pipeline-stop-token-ids"
CONFIG_NUMERIC_DEFAULT = "config-numeric-default"
MODULE_SPECIAL_TOKEN = "module-special-control-token"

_TOKEN_NAME = re.compile(r"(?:TOKEN|BOS|EOS|PAD|STOP|CONTROL|SPECIAL|SENTINEL)")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    scope: str
    source: str
    occurrence: int = 1

    @property
    def fingerprint(self) -> str:
        identity = "\0".join((self.path, self.rule, self.scope, self.source, str(self.occurrence)))
        return hashlib.sha256(identity.encode()).hexdigest()[:16]


def _numeric_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int | float) and not isinstance(node.value, bool)
    return isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd) and _numeric_literal(node.operand)


def _contains_numeric_literal(node: ast.AST) -> bool:
    return any(_numeric_literal(child) for child in ast.walk(node))


def _is_config_reference(node: ast.AST) -> bool:
    return any(
        (
            isinstance(child, ast.Name)
            and (child.id.lower().endswith("config") or child.id.lower().endswith("cfg"))
        )
        or (
            isinstance(child, ast.Attribute)
            and (child.attr.lower().endswith("config") or child.attr.lower().endswith("cfg"))
        )
        for child in ast.walk(node)
    )


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Tuple | ast.List):
        return [name for child in node.elts for name in _target_names(child)]
    return []


class _ScopedVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.scopes: list[str] = []
        self.findings: list[Finding] = []

    @property
    def scope(self) -> str:
        return ".".join(self.scopes) or "<module>"

    def _add(self, node: ast.AST, rule: str) -> None:
        self.findings.append(
            Finding(
                path=self.path.as_posix(),
                line=node.lineno,
                rule=rule,
                scope=self.scope,
                source=ast.unparse(node),
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self.visit_FunctionDef(node)

    def visit_Call(self, node: ast.Call) -> None:
        is_numeric_getattr = (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 3
            and _is_config_reference(node.args[0])
            and _contains_numeric_literal(node.args[2])
        )
        is_numeric_config_get = (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and len(node.args) >= 2
            and _is_config_reference(node.func.value)
            and _contains_numeric_literal(node.args[1])
        )
        if is_numeric_getattr or is_numeric_config_get:
            self._add(node, CONFIG_NUMERIC_DEFAULT)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        if self.path.name == "pipeline.py":
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "stop_token_ids" and _contains_numeric_literal(value):
                    self._add(value, PIPELINE_STOP_TOKEN_IDS)
        self.generic_visit(node)


def _module_token_findings(path: Path, tree: ast.Module) -> list[Finding]:
    findings: list[Finding] = []
    for node in tree.body:
        targets: list[ast.AST]
        value: ast.AST | None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if value is None or not _contains_numeric_literal(value):
            continue
        for name in (name for target in targets for name in _target_names(target)):
            if _TOKEN_NAME.search(name.upper()):
                findings.append(
                    Finding(
                        path=path.as_posix(),
                        line=node.lineno,
                        rule=MODULE_SPECIAL_TOKEN,
                        scope="<module>",
                        source=f"{name} = {ast.unparse(value)}",
                    )
                )
    return findings


def scan_source(path: Path, source: str) -> list[Finding]:
    tree = ast.parse(source, filename=path.as_posix())
    visitor = _ScopedVisitor(path)
    visitor.visit(tree)
    findings = visitor.findings + _module_token_findings(path, tree)

    occurrences: Counter[tuple[str, str, str, str]] = Counter()
    numbered: list[Finding] = []
    for finding in sorted(findings, key=lambda item: (item.line, item.rule, item.source)):
        key = (finding.path, finding.rule, finding.scope, finding.source)
        occurrences[key] += 1
        numbered.append(
            Finding(
                path=finding.path,
                line=finding.line,
                rule=finding.rule,
                scope=finding.scope,
                source=finding.source,
                occurrence=occurrences[key],
            )
        )
    return numbered


def scan_tree(root: Path = REPO_ROOT) -> list[Finding]:
    findings: list[Finding] = []
    for audit_root in AUDIT_ROOTS:
        for path in sorted((root / audit_root).rglob("*.py")):
            relative_path = path.relative_to(root)
            findings.extend(scan_source(relative_path, path.read_text(encoding="utf-8")))
    return findings


def _group_digest(findings: list[Finding]) -> str:
    identities = sorted(f"{finding.scope}\0{finding.source}" for finding in findings)
    return hashlib.sha256("\n".join(identities).encode()).hexdigest()[:16]


def load_baseline(root: Path = REPO_ROOT) -> dict[tuple[str, str], dict[str, object]]:
    path = root / BASELINE_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("allowlist")
    if not isinstance(entries, list):
        raise ValueError(f"{BASELINE_PATH}: 'allowlist' must be a list")

    baseline: dict[tuple[str, str], dict[str, object]] = {}
    for entry in entries:
        path_value = entry.get("path")
        rule = entry.get("rule")
        category = entry.get("category")
        count = entry.get("count")
        digest = entry.get("digest")
        if (
            not isinstance(path_value, str)
            or rule not in {PIPELINE_STOP_TOKEN_IDS, CONFIG_NUMERIC_DEFAULT, MODULE_SPECIAL_TOKEN}
            or category not in VALID_CATEGORIES
            or not isinstance(count, int)
            or count < 1
            or not isinstance(digest, str)
        ):
            raise ValueError(f"{BASELINE_PATH}: invalid baseline entry: {entry!r}")
        key = (path_value, rule)
        if key in baseline:
            raise ValueError(f"{BASELINE_PATH}: duplicate path/rule entry {key}")
        baseline[key] = entry
    return baseline


def audit(root: Path = REPO_ROOT) -> list[str]:
    findings = scan_tree(root)
    baseline = load_baseline(root)
    current: dict[tuple[str, str], list[Finding]] = {}
    for finding in findings:
        current.setdefault((finding.path, finding.rule), []).append(finding)
    errors: list[str] = []

    for key, grouped_findings in current.items():
        entry = baseline.get(key)
        if entry is None:
            for finding in grouped_findings:
                errors.append(
                    f"{finding.path}:{finding.line}: new {finding.rule}: {finding.source} "
                    f"[fingerprint: {finding.fingerprint}]"
                )
            continue
        actual_digest = _group_digest(grouped_findings)
        if len(grouped_findings) != entry["count"] or actual_digest != entry["digest"]:
            locations = ", ".join(str(finding.line) for finding in grouped_findings)
            errors.append(
                f"{key[0]}:{locations}: {key[1]} baseline changed "
                f"(expected count={entry['count']} digest={entry['digest']}; "
                f"actual count={len(grouped_findings)} digest={actual_digest}). "
                "Review the diff, remediate additions, and update the classified baseline only for intentional legacy."
            )
    for key, entry in baseline.items():
        if key not in current:
            errors.append(f"{BASELINE_PATH}: stale {entry['category']} allowlist entry {key[0]} / {key[1]} — remove it")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "filenames",
        nargs="*",
        help="Accepted for pre-commit; the complete audit roots are always scanned.",
    )
    parser.parse_args(argv)

    try:
        errors = audit()
    except (OSError, SyntaxError, ValueError, json.JSONDecodeError) as exc:
        print(f"check_model_executor_hardcodes: {exc}", file=sys.stderr)
        return 1
    for error in errors:
        print(f"check_model_executor_hardcodes: {error}", file=sys.stderr)
    if errors:
        print(
            "Classify intentional findings in the reviewed baseline; do not copy a fingerprint "
            "without documenting its ownership.",
            file=sys.stderr,
        )
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
