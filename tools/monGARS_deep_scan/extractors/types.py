from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExtractionRecord:
    dataset: str  # "sft" | "agent" | "embeddings"
    instruction: str = ""
    input_text: str = ""
    output: Any = ""
    text: str = ""
    source_file: str = ""
    start_line: int = 1
    end_line: int = 1
    type_label: str = "unknown"

    @classmethod
    def for_sft(
        cls,
        *,
        instruction: str,
        output: str,
        source_file: str,
        start_line: int,
        end_line: int,
        type_label: str,
        input_text: str = "",
    ) -> "ExtractionRecord":
        return cls(
            dataset="sft",
            instruction=instruction,
            output=output,
            input_text=input_text,
            source_file=source_file,
            start_line=start_line,
            end_line=end_line,
            type_label=type_label,
        )

    @classmethod
    def for_agent(
        cls,
        *,
        instruction: str,
        output: Any,
        source_file: str,
        start_line: int,
        end_line: int,
        type_label: str,
        input_text: str = "",
    ) -> "ExtractionRecord":
        return cls(
            dataset="agent",
            instruction=instruction,
            output=output,
            input_text=input_text,
            source_file=source_file,
            start_line=start_line,
            end_line=end_line,
            type_label=type_label,
        )

    @classmethod
    def for_embedding(
        cls,
        *,
        text: str,
        source_file: str,
        start_line: int,
        end_line: int,
        type_label: str,
    ) -> "ExtractionRecord":
        return cls(
            dataset="embeddings",
            text=text,
            source_file=source_file,
            start_line=start_line,
            end_line=end_line,
            type_label=type_label,
        )
