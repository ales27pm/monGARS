from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .log import get_logger

logger = get_logger()


def read_text_file(path: Path, max_lines: Optional[int] = None) -> Optional[str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            if max_lines is None:
                return handle.read()

            lines: list[str] = []
            for idx, line in enumerate(handle):
                if max_lines is not None and idx >= max_lines:
                    logger.warning(
                        "Skipping %s because it exceeds max_lines=%s", path, max_lines
                    )
                    return None
                lines.append(line)
            return "".join(lines)
    except UnicodeDecodeError:
        logger.warning("Skipping %s due to decode error", path)
    except OSError as exc:
        logger.error("Failed to read %s: %s", path, exc)
    return None


def stream_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_lines_with_numbers(text: str) -> Iterator[tuple[int, str]]:
    for idx, line in enumerate(text.splitlines(), start=1):
        yield idx, line
