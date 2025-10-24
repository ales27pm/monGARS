from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from ..utils.hashing import stable_hash
from ..utils.io import stream_jsonl
from .provenance import ProvenanceRecord, ProvenanceTracker
from .qc_filter import QCFilter

DATASET_FILE_NAMES = {
    "sft": "sft_dataset.jsonl",
    "agent": "agent_handoff_dataset.jsonl",
    "embeddings": "embeddings_corpus.jsonl",
}

DATASET_LABELS = {
    "sft": "sft_dataset",
    "agent": "agent_handoff_dataset",
    "embeddings": "embeddings_corpus",
}


@dataclass
class RecordContainer:
    record_id: str
    dataset: str
    payload: dict
    source_file: str
    start_line: int
    end_line: int
    type_label: str
    qc_flag: bool
    sort_key: Tuple


class DatasetBuilders:
    def __init__(self, provenance: ProvenanceTracker, qc_filter: QCFilter) -> None:
        self._provenance = provenance
        self._qc_filter = qc_filter
        self._records: Dict[str, List[RecordContainer]] = defaultdict(list)
        self._dataset_counts: Counter[str] = Counter()
        self._qc_counts: Counter[str] = Counter()
        self._file_breakdown: Dict[str, Counter[str]] = defaultdict(Counter)
        self._type_breakdown: Counter[str] = Counter()
        self._samples: Dict[str, List[dict]] = defaultdict(list)

    def add_sft(
        self,
        instruction: str,
        output: str,
        *,
        input_text: str = "",
        source_file: str,
        start_line: int,
        end_line: int,
        type_label: str,
    ) -> str:
        record = {
            "instruction": instruction.strip(),
            "input": input_text.strip(),
            "output": output.strip(),
        }
        return self._add_record(
            "sft", record, source_file, start_line, end_line, type_label
        )

    def add_agent(
        self,
        instruction: str,
        output,
        *,
        input_text: str = "",
        source_file: str,
        start_line: int,
        end_line: int,
        type_label: str,
    ) -> str:
        record = {
            "instruction": instruction.strip(),
            "input": input_text.strip(),
            "output": output,
        }
        return self._add_record(
            "agent", record, source_file, start_line, end_line, type_label
        )

    def add_embedding(
        self,
        text: str,
        *,
        source_file: str,
        start_line: int,
        end_line: int,
        type_label: str,
    ) -> str:
        record = {
            "text": text.strip(),
        }
        return self._add_record(
            "embeddings", record, source_file, start_line, end_line, type_label
        )

    def _add_record(
        self,
        dataset: str,
        record: dict,
        source_file: str,
        start_line: int,
        end_line: int,
        type_label: str,
    ) -> str:
        dataset_label = DATASET_LABELS[dataset]
        qc_flag = self._qc_filter.flag_text(*(str(value) for value in record.values()))
        payload = dict(record)
        payload["_meta"] = {
            "source_file": source_file,
            "start_line": start_line,
            "end_line": end_line,
            "type": type_label,
            "qc_fr_ca": qc_flag,
        }
        record_id = stable_hash(
            [
                dataset_label,
                source_file,
                str(start_line),
                str(end_line),
                str(payload.get("instruction", "")),
                str(payload.get("input", "")),
                str(payload.get("output", "")),
                str(payload.get("text", "")),
            ]
        )

        container = RecordContainer(
            record_id=record_id,
            dataset=dataset,
            payload=payload,
            source_file=source_file,
            start_line=start_line,
            end_line=end_line,
            type_label=type_label,
            qc_flag=qc_flag,
            sort_key=(source_file, start_line, end_line, record_id),
        )
        self._records[dataset].append(container)
        self._dataset_counts[dataset_label] += 1
        if qc_flag:
            self._qc_counts[dataset_label] += 1
        self._file_breakdown[dataset_label][source_file] += 1
        self._type_breakdown[type_label] += 1
        if len(self._samples[dataset_label]) < 10:
            self._samples[dataset_label].append(payload)

        self._provenance.add(
            ProvenanceRecord(
                record_id=record_id,
                dataset=dataset_label,
                source_file=source_file,
                start_line=start_line,
                end_line=end_line,
                type=type_label,
                qc_fr_ca=qc_flag,
            )
        )

        return record_id

    def write_all(self, out_dir: Path) -> None:
        for dataset in DATASET_FILE_NAMES:
            containers = self._records.get(dataset, [])
            file_path = out_dir / DATASET_FILE_NAMES[dataset]
            serialised = [
                container.payload
                for container in sorted(containers, key=lambda c: c.sort_key)
            ]
            stream_jsonl(file_path, serialised)

    def report(self) -> dict:
        def qc_ratio(label: str) -> float:
            total = self._dataset_counts.get(label, 0)
            if total == 0:
                return 0.0
            return self._qc_counts.get(label, 0) / total

        dataset_counts = {
            label: self._dataset_counts.get(label, 0)
            for label in DATASET_LABELS.values()
        }
        qc_ratios = {label: qc_ratio(label) for label in dataset_counts}
        file_breakdown = {
            label: dict(self._file_breakdown.get(label, Counter()))
            for label in dataset_counts
        }
        return {
            "dataset_counts": dataset_counts,
            "qc_ratios": qc_ratios,
            "file_breakdown": file_breakdown,
            "type_breakdown": dict(self._type_breakdown),
            "samples": {label: samples for label, samples in self._samples.items()},
        }
