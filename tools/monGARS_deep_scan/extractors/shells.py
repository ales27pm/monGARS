from __future__ import annotations

from pathlib import Path
from typing import List

from ..utils.text_clean import normalise_whitespace
from .types import ExtractionRecord


def _collect_comment_blocks(lines: List[str]) -> List[tuple[str, int, int]]:
    blocks: List[tuple[str, int, int]] = []
    current: List[str] = []
    start_line = 1
    for idx, line in enumerate(lines, start=1):
        if line.strip().startswith("#"):
            content = line.lstrip("# ")
            if not current:
                start_line = idx
            current.append(content)
        else:
            if current:
                blocks.append(("\n".join(current), start_line, idx - 1))
                current = []
    if current:
        blocks.append(("\n".join(current), start_line, len(lines)))
    return blocks


def extract(path: Path, text: str) -> List[ExtractionRecord]:
    lines = text.splitlines()
    records: List[ExtractionRecord] = []

    for block, start, end in _collect_comment_blocks(lines):
        cleaned = normalise_whitespace(block)
        if len(cleaned) >= 60:
            records.append(
                ExtractionRecord.for_embedding(
                    text=cleaned,
                    source_file=str(path),
                    start_line=start,
                    end_line=end,
                    type_label="shell_comment",
                )
            )

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("echo") and "Usage" in stripped:
            message = stripped.split("Usage", 1)[1].strip(" :\"')")
            records.append(
                ExtractionRecord.for_agent(
                    instruction="Display shell usage guidance",
                    output={"echo": message},
                    source_file=str(path),
                    start_line=idx,
                    end_line=idx,
                    type_label="shell_usage_echo",
                )
            )
    return records
