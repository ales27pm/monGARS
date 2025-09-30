from __future__ import annotations

from pathlib import Path

import pytest

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]

# Only deliberate test doubles are allowed to raise NotImplementedError.
ALLOWED_NOT_IMPLEMENTED: dict[Path, set[int]] = {
    Path("tests/test_dynamic_response.py"): {15}
}


def _iter_python_files(root: Path) -> list[Path]:
    """Return python files under ``root`` ignoring virtualenv and cache dirs."""

    excluded_dir_names = {
        "__pycache__",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "build",
        "dist",
        "node_modules",
        "venv",
    }

    candidates: list[Path] = []
    for path in root.rglob("*.py"):
        if path == THIS_FILE:
            continue
        if any(part in excluded_dir_names for part in path.parts):
            continue
        candidates.append(path)
    return candidates


def _collect_marker_violations() -> list[str]:
    todo_markers = {"TODO", "FIXME", "XXX"}
    violations: list[str] = []

    for path in _iter_python_files(REPO_ROOT):
        try:
            contents = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Non-UTF8 sources are out of scope for this enforcement.
            continue

        relative_path = path.relative_to(REPO_ROOT)

        for line_number, line in enumerate(contents.splitlines(), start=1):
            stripped = line.strip()

            # Quick skip for comments referencing "pass-through" style notes.
            if stripped.startswith("# pragma"):
                continue

            for marker in todo_markers:
                if marker in line:
                    violations.append(
                        f"{relative_path}:{line_number} contains {marker}"
                    )

            if "NotImplementedError" in line:
                allowed_lines = ALLOWED_NOT_IMPLEMENTED.get(relative_path, set())
                if line_number not in allowed_lines:
                    violations.append(
                        "{}:{} raises NotImplementedError without an explicit allowlist entry".format(
                            relative_path, line_number
                        )
                    )

    return violations


@pytest.mark.parametrize("violations", [_collect_marker_violations()])
def test_incomplete_logic_markers_absent(violations: list[str]) -> None:
    """Ensure TODO/FIXME/XXX markers and unfinished stubs stay out of the repo."""

    assert (
        not violations
    ), "Found incomplete-logic markers that require attention:\n" + "\n".join(
        sorted(violations)
    )
