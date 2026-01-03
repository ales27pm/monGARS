from __future__ import annotations

import io
import tokenize
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]

# Only deliberate test doubles are allowed to raise NotImplementedError.
ALLOWED_NOT_IMPLEMENTED: dict[str, set[int]] = {}


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
        "unsloth_compiled_cache",
        "venv",
    }

    candidates: list[Path] = []
    for path in root.rglob("*.py"):
        if path == THIS_FILE:
            continue
        if all(part not in excluded_dir_names for part in path.parts):
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

        relative_path = path.relative_to(REPO_ROOT).as_posix()

        try:
            tokens = tokenize.generate_tokens(io.StringIO(contents).readline)
        except (SyntaxError, tokenize.TokenError):
            comment_tokens: list[tuple[int, str]] = []
        else:
            comment_tokens = [
                (token.start[0], token.string)
                for token in tokens
                if token.type == tokenize.COMMENT
            ]

        for line_number, line in enumerate(contents.splitlines(), start=1):
            stripped = line.strip()

            # Skip pragma directives (e.g., coverage or type-checking pragmas).
            if stripped.startswith("# pragma"):
                continue

            if "NotImplementedError" in line:
                allowed_lines = ALLOWED_NOT_IMPLEMENTED.get(relative_path, set())
                if line_number not in allowed_lines:
                    violations.append(
                        f"{relative_path}:{line_number} raises NotImplementedError without an explicit allowlist entry"
                    )

        for comment_line, comment in comment_tokens:
            if comment.lstrip().startswith("# pragma"):
                continue
            for marker in todo_markers:
                if marker in comment:
                    violations.append(
                        f"{relative_path}:{comment_line} contains {marker}"
                    )

    return violations


def test_incomplete_logic_markers_absent() -> None:
    """Ensure TODO/FIXME/XXX markers and unfinished placeholders stay out of the repo."""

    violations = _collect_marker_violations()
    assert (
        not violations
    ), "Found incomplete-logic markers that require attention:\n" + "\n".join(
        sorted(violations)
    )
