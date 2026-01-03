"""Synchronise Markdown metadata across the repository.

This utility injects/updates a standard `Last updated` line near the top of
every Markdown document (excluding `AGENTS.md`). The value is derived from the
file's most recent Git commit and falls back to today's date for new files.

Running the script keeps documentation timestamps dynamic without requiring
manual edits. It is idempotent and safe to run repeatedly.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
LAST_UPDATED_PREFIX = "> **Last updated:** "
INSTRUCTION_SUFFIX = " _(auto-synced; run `python scripts/update_docs_metadata.py`)_"
SKIP_FILENAMES = {"AGENTS.md"}
DEFAULT_SECTION_TITLES = {
    "README.md": "Project overview",
    "ROADMAP.md": "Roadmap",
}


def normalize_heading(title: str) -> str:
    """Return a string guaranteed to be a Markdown heading."""

    cleaned = title.strip()
    if not cleaned:
        return "# Untitled"
    if not cleaned.startswith("#"):
        return f"# {cleaned}"
    return cleaned


def iter_markdown_files(paths: Iterable[Path] | None) -> Iterable[Path]:
    """Yield Markdown files under the provided paths."""

    if not paths:
        paths = [REPO_ROOT]

    excluded_parts = {
        "node_modules",
        "vendor",
        "build",
        "dist",
        ".git",
        "venv",
        ".venv",
    }
    allowed_exts = {".md", ".mdx"}

    for base in paths:
        base = (base if base.is_absolute() else (Path.cwd() / base)).resolve()
        if not base.exists():
            continue
        if base.is_file():
            if (
                base.suffix.lower() in allowed_exts
                and base.name not in SKIP_FILENAMES
                and not any(part in excluded_parts for part in base.parts)
            ):
                yield base
            continue

        for path in base.rglob("*"):
            if any(part in excluded_parts for part in path.parts):
                continue
            if not path.is_file():
                continue
            if path.suffix.lower() not in allowed_exts:
                continue
            if path.name in SKIP_FILENAMES:
                continue
            yield path


def git_last_updated(path: Path) -> str:
    """Return the last commit date for `path` in ISO format."""

    try:
        rel_path = path.relative_to(REPO_ROOT)
        out = subprocess.check_output(
            [
                "git",
                "log",
                "-1",
                "--date=short",
                "--format=%cd",
                str(rel_path),
            ],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        out = b""

    date_str = out.decode().strip()
    if not date_str:
        date_str = _dt.date.today().isoformat()
    return date_str


def find_heading_index(lines: list[str]) -> int | None:
    """Return the index of the first Markdown heading line, if any."""

    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            return idx
    return None


def ensure_last_updated(path: Path) -> bool:
    """Ensure the document contains the canonical Last updated line.

    Returns True if the file was modified.
    """

    original_text = path.read_text(encoding="utf-8")
    lines = original_text.splitlines()

    if not lines:
        default_title = DEFAULT_SECTION_TITLES.get(path.name, path.stem.title())
        lines = [normalize_heading(default_title)]

    heading_idx = find_heading_index(lines)

    if heading_idx is None:
        # Prepend a generated heading if missing.
        generated_heading = DEFAULT_SECTION_TITLES.get(
            path.name, path.stem.replace("_", " ").title()
        )
        lines.insert(0, normalize_heading(generated_heading))
        heading_idx = 0

    metadata_line = f"{LAST_UPDATED_PREFIX}{git_last_updated(path)}{INSTRUCTION_SUFFIX}"

    # Determine insertion point (line immediately after heading, skipping blank lines once).
    insert_idx = heading_idx + 1
    if insert_idx < len(lines) and not lines[insert_idx].strip():
        insert_idx += 1

    # Remove existing metadata line if present.
    existing_idx = None
    for idx, line in enumerate(lines):
        if line.startswith(LAST_UPDATED_PREFIX):
            existing_idx = idx
            break

    modified = False
    if existing_idx is not None:
        if lines[existing_idx] != metadata_line:
            lines.pop(existing_idx)
            if existing_idx < insert_idx:
                insert_idx -= 1
            lines.insert(insert_idx, metadata_line)
            modified = True
        else:
            insert_idx = existing_idx
    else:
        lines.insert(insert_idx, metadata_line)
        modified = True

    # Remove any duplicate canonical banners elsewhere in the document.
    duplicate_indices = [
        idx for idx, line in enumerate(lines) if line.startswith(LAST_UPDATED_PREFIX)
    ]
    for idx in reversed(duplicate_indices):
        if idx == insert_idx:
            continue
        lines.pop(idx)
        modified = True

    # Drop redundant banner immediately following the metadata line even if formatted differently.
    redundant_idx = insert_idx + 1
    while redundant_idx < len(lines) and not lines[redundant_idx].strip():
        redundant_idx += 1
    if redundant_idx < len(lines):
        candidate = lines[redundant_idx].strip().lower().lstrip(">*_ ").strip()
        if candidate.startswith("last updated"):
            lines.pop(redundant_idx)
            modified = True

    # Ensure a blank line separates metadata from the next block for readability.
    after_metadata = insert_idx + 1
    if after_metadata >= len(lines) or lines[after_metadata].strip():
        lines.insert(after_metadata, "")
        modified = True

    new_text = "\n".join(lines) + "\n"

    if not modified and new_text == original_text:
        return False

    path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synchronise Markdown metadata across the repo."
    )
    parser.add_argument(
        "paths", nargs="*", type=Path, help="Optional directories or files to process"
    )
    args = parser.parse_args()

    changed = 0
    for md_file in iter_markdown_files(args.paths):
        if ensure_last_updated(md_file):
            changed += 1

    print(f"Updated metadata for {changed} Markdown file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
