from __future__ import annotations

import re
from typing import Iterable, List, Sequence

PARAGRAPH_MIN_LENGTH = 60

_DIALOG_ROLES = (
    "user",
    "client",
    "utilisateur",
    "moi",
    "tu",
    "vous",
    "assistant",
    "system",
    "bot",
    "agent",
)

DIALOG_PATTERN = re.compile(
    r"^(?P<role>(?:" + "|".join(_DIALOG_ROLES) + r"))\s*[:\-—]\s*(?P<content>.+)",
    re.IGNORECASE,
)

CODE_FENCE_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def strip_code_fences(text: str) -> str:
    return re.sub(CODE_FENCE_PATTERN, "", text)


def strip_html_tags(text: str) -> str:
    return re.sub(HTML_TAG_PATTERN, " ", text)


def normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_paragraphs(text: str) -> List[tuple[str, int, int]]:
    paragraphs: List[tuple[str, int, int]] = []
    current_lines: List[str] = []
    start_line = 1
    line_no = 0
    for raw_line in text.splitlines():
        line_no += 1
        if raw_line.strip():
            if not current_lines:
                start_line = line_no
            current_lines.append(raw_line)
        else:
            if current_lines:
                paragraph = "\n".join(current_lines).strip()
                if len(paragraph) >= PARAGRAPH_MIN_LENGTH:
                    paragraphs.append((paragraph, start_line, line_no - 1))
                current_lines = []
    if current_lines:
        paragraph = "\n".join(current_lines).strip()
        if len(paragraph) >= PARAGRAPH_MIN_LENGTH:
            paragraphs.append((paragraph, start_line, line_no))
    return paragraphs


def find_dialog_blocks(lines: Sequence[str]) -> List[dict]:
    dialog: List[dict] = []
    buffer: List[dict] = []
    start_line = 1
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        match = DIALOG_PATTERN.match(stripped)
        if match:
            if not buffer:
                start_line = idx
            buffer.append(
                {
                    "role": match.group("role").lower(),
                    "content": match.group("content").strip(),
                    "line": idx,
                }
            )
        else:
            if len(buffer) >= 2:
                dialog.append(
                    {
                        "lines": buffer.copy(),
                        "start_line": start_line,
                        "end_line": buffer[-1]["line"],
                    }
                )
            buffer = []
    if len(buffer) >= 2:
        dialog.append(
            {
                "lines": buffer,
                "start_line": start_line,
                "end_line": buffer[-1]["line"],
            }
        )
    return dialog


def chunk_lines(text: str) -> List[str]:
    return text.splitlines()
