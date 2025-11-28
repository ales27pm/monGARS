from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Set

from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parent.parent
SETUP_PATH = REPO_ROOT / "setup.py"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
VENDOR_SENTINEL = "llm2vec-local"
TEST_DEPENDENCIES = {"pytest", "pytest-asyncio", "coverage", "packaging"}


def _normalize_requirement(raw: str) -> str:
    cleaned = raw.strip()
    if not cleaned:
        return cleaned
    base_part = cleaned.split(";", 1)[0].strip()
    if base_part.startswith("./vendor/") or base_part.startswith("vendor/"):
        return VENDOR_SENTINEL
    if base_part.startswith("llm2vec @"):
        return VENDOR_SENTINEL
    return base_part


def _detect_group(raw: str, requirement: Requirement | None) -> str:
    if requirement and requirement.marker:
        marker_text = str(requirement.marker).replace("'", '"')
        if 'extra == "grpo"' in marker_text:
            return "grpo"
    if requirement and requirement.name.lower() in TEST_DEPENDENCIES:
        return "test"
    if requirement is None:
        base_part = raw.split(";", 1)[0].strip()
        for candidate in TEST_DEPENDENCIES:
            if base_part.lower().startswith(candidate):
                return "test"
    return "base"


def _parse_requirements() -> Dict[str, Set[str]]:
    grouped: DefaultDict[str, Set[str]] = defaultdict(set)
    for line in REQUIREMENTS_PATH.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        requirement: Requirement | None = None
        try:
            requirement = Requirement(stripped)
        except Exception:
            requirement = None
        group = _detect_group(stripped, requirement)
        grouped[group].add(_normalize_requirement(stripped))
    return grouped


def _evaluate_string(node) -> str:
    import ast

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: List[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                if (
                    isinstance(value.value, ast.Name)
                    and value.value.id == "LLM2VEC_URI"
                ):
                    uri = (REPO_ROOT / "vendor" / "llm2vec_monGARS").resolve().as_uri()
                    parts.append(uri)
                else:
                    raise ValueError("Unsupported formatted value in setup.py")
            else:
                raise ValueError("Unsupported node structure in setup.py")
        return "".join(parts)
    raise ValueError("Unsupported requirement node in setup.py")


def _parse_setup_requirements() -> Dict[str, Set[str]]:
    import ast

    with SETUP_PATH.open("r", encoding="utf-8") as setup_file:
        tree = ast.parse(setup_file.read())

    grouped: DefaultDict[str, Set[str]] = defaultdict(set)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg == "install_requires" and isinstance(
                keyword.value, ast.List
            ):
                normalized = (
                    _normalize_requirement(_evaluate_string(item))
                    for item in keyword.value.elts
                )
                grouped["base"].update(normalized)
            if keyword.arg == "extras_require" and isinstance(keyword.value, ast.Dict):
                for key_node, value_node in zip(
                    keyword.value.keys, keyword.value.values
                ):
                    if not isinstance(key_node, ast.Constant) or not isinstance(
                        key_node.value, str
                    ):
                        continue
                    extra_key = key_node.value
                    if isinstance(value_node, ast.List):
                        normalized_extra = (
                            _normalize_requirement(_evaluate_string(item))
                            for item in value_node.elts
                        )
                        grouped[extra_key].update(normalized_extra)
    return grouped


def _assert_group_alignment(
    expected: Dict[str, Set[str]], actual: Dict[str, Set[str]], group: str
) -> None:
    assert expected.get(group, set()) == actual.get(
        group, set()
    ), f"Mismatch for {group} dependencies"


def test_requirements_alignment():
    requirements = _parse_requirements()
    setup_requirements = _parse_setup_requirements()

    _assert_group_alignment(requirements, setup_requirements, "base")
    _assert_group_alignment(requirements, setup_requirements, "grpo")
    _assert_group_alignment(requirements, setup_requirements, "test")
