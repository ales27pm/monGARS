from __future__ import annotations

from pathlib import Path
from typing import List

from ..utils.text_clean import (
    chunk_lines,
    find_dialog_blocks,
    normalise_whitespace,
    split_paragraphs,
    strip_code_fences,
    strip_html_tags,
)
from .types import ExtractionRecord

_USER_ROLES = {"user", "client", "utilisateur", "moi", "tu", "vous"}


def extract(path: Path, text: str) -> List[ExtractionRecord]:
    cleaned = strip_html_tags(strip_code_fences(text))
    records: List[ExtractionRecord] = []

    paragraphs = split_paragraphs(cleaned)
    for paragraph, start, end in paragraphs:
        records.append(
            ExtractionRecord.for_embedding(
                text=normalise_whitespace(paragraph),
                source_file=str(path),
                start_line=start,
                end_line=end,
                type_label="doc_paragraph",
            )
        )

    lines = chunk_lines(cleaned)
    for block in find_dialog_blocks(lines):
        user_lines = [
            line["content"] for line in block["lines"] if line["role"] in _USER_ROLES
        ]
        assistant_lines = [
            line["content"]
            for line in block["lines"]
            if line["role"] not in _USER_ROLES
        ]
        if user_lines and assistant_lines:
            records.append(
                ExtractionRecord.for_sft(
                    instruction="\n".join(user_lines),
                    output="\n".join(assistant_lines),
                    source_file=str(path),
                    start_line=block["start_line"],
                    end_line=block["end_line"],
                    type_label="doc_dialog",
                )
            )

    return records
