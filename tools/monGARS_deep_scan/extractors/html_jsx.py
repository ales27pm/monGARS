from __future__ import annotations

from pathlib import Path
from typing import List

from bs4 import BeautifulSoup

from ..utils.text_clean import (
    chunk_lines,
    find_dialog_blocks,
    normalise_whitespace,
    split_paragraphs,
)
from .types import ExtractionRecord

_USER_ROLES = {"user", "client", "utilisateur", "moi", "tu", "vous"}


def extract(path: Path, text: str) -> List[ExtractionRecord]:
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    raw_text = soup.get_text(separator="\n")
    records: List[ExtractionRecord] = []

    paragraphs = split_paragraphs(raw_text)
    for paragraph, start, end in paragraphs:
        cleaned = normalise_whitespace(paragraph)
        records.append(
            ExtractionRecord.for_embedding(
                text=cleaned,
                source_file=str(path),
                start_line=start,
                end_line=end,
                type_label="html_paragraph",
            )
        )

    for block in find_dialog_blocks(chunk_lines(raw_text)):
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
                    type_label="html_dialog",
                )
            )
    return records
