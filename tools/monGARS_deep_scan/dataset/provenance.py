from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class ProvenanceRecord:
    record_id: str
    dataset: str
    source_file: str
    start_line: int
    end_line: int
    type: str
    qc_fr_ca: bool


class ProvenanceTracker:
    def __init__(self) -> None:
        self._records: List[ProvenanceRecord] = []

    def add(self, record: ProvenanceRecord) -> None:
        self._records.append(record)

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "record_id",
                    "dataset",
                    "source_file",
                    "start_line",
                    "end_line",
                    "type",
                    "qc_fr_ca",
                ]
            )
            for record in self._records:
                writer.writerow(
                    [
                        record.record_id,
                        record.dataset,
                        record.source_file,
                        record.start_line,
                        record.end_line,
                        record.type,
                        "true" if record.qc_fr_ca else "false",
                    ]
                )

    @property
    def records(self) -> Iterable[ProvenanceRecord]:
        return list(self._records)
