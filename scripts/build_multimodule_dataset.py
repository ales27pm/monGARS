#!/usr/bin/env python3
"""Build a module-tagged multitask dataset from analyzer artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Sequence

RANDOM_SEED = 42
MIN_LEN = 12
DEFAULT_VAL_PCT = 0.06

QC_TERMS = {
    "dépanneur",
    "poutine",
    "cégep",
    "tuque",
    "magasiner",
    "char",
    "chum",
    "blonde",
    "icitte",
    "ben là",
    "patente",
    "tabarnak",
}

MODULE_RULES: Sequence[tuple[str, str]] = (
    (r"(core/hippocampus|hippocampus)", "Hippocampus"),
    (r"(core/conversation|cortex|bouche)", "Cortex"),
    (r"(evolution_engine|sommeil|self_training)", "Evolution"),
    (r"(neuro_symbolic|reasoner|tronc)", "NeuroSymbolic"),
    (r"(rag|retriev|vector|embedding)", "RAG"),
    (r"(api/|fastapi|server|routes|ws)", "API"),
    (r"(webapp|django|frontend|ui|console)", "WebApp"),
    (r"(ray|serve|distributed|actors|replica)", "Distributed"),
    (r"(scripts/|ops/|infra/|docker|k8s|compose)", "Ops"),
)


@dataclass(frozen=True)
class Record:
    module: str
    instruction: str
    input: str
    output: str
    record_id: str
    source_paths: Sequence[str]

    def to_payload(self) -> Dict[str, str]:
        return {
            "module": self.module,
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a module-tagged multitask dataset using artifacts produced by "
            "scripts/ultimate_repo_analyzer.py"
        )
    )
    parser.add_argument(
        "--repo_sft",
        default="data/ultimate/processed_repo/sft_repo.jsonl",
        help="Path to the repository SFT samples",
    )
    parser.add_argument(
        "--agent_sft",
        default="data/ultimate/processed_repo/agent_instruct_repo.jsonl",
        help="Path to the agent hand-off style samples",
    )
    parser.add_argument(
        "--provenance",
        default="data/ultimate/processed_repo/provenance.csv",
        help="Provenance CSV emitted by the analyzer",
    )
    parser.add_argument(
        "--outdir",
        default="data/multimodule",
        help="Directory to write train/val JSONL outputs",
    )
    parser.add_argument(
        "--val_pct",
        type=float,
        default=DEFAULT_VAL_PCT,
        help="Validation split ratio (0-1)",
    )
    parser.add_argument(
        "--strict_qc",
        action="store_true",
        help="Filter repo/agent samples for Québec French keywords",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed controlling shuffling",
    )
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def is_valid_text(value: str) -> bool:
    return isinstance(value, str) and len(value.strip()) >= MIN_LEN


def clamp_output(value: str) -> str:
    return value.strip()


def normalize_sft(record: dict) -> dict | None:
    if not isinstance(record, dict):
        return None
    instruction = (record.get("instruction") or "").strip()
    input_text = (record.get("input") or "").strip()
    output = record.get("output")
    if not isinstance(output, str):
        output = json.dumps(
            output,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    output = clamp_output(output)
    if not (is_valid_text(instruction) and is_valid_text(output)):
        return None
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }


def qc_ok(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in QC_TERMS)


def choose_module(source_file: str) -> str:
    lowered = source_file.lower()
    for pattern, module_name in MODULE_RULES:
        if re.search(pattern, lowered):
            return module_name
    parts = [part for part in lowered.split("/") if part and part != "."]
    if parts:
        return parts[0].capitalize()
    return "General"


def compute_record_id(payload: dict) -> str:
    serialized = json.dumps(payload, ensure_ascii=False)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:12]


def load_provenance(path: Path) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            record_id = (row.get("record_id") or "").strip()
            source_file = (row.get("source_file") or "").strip()
            if record_id and source_file:
                mapping[record_id].append(source_file)
    return mapping


def derive_module(candidate_paths: Sequence[str], default_hint: str) -> str:
    if not candidate_paths:
        return default_hint
    counter: Counter[str] = Counter()
    for path in candidate_paths:
        counter[choose_module(path)] += 1
    most_common = counter.most_common(1)
    if most_common:
        return most_common[0][0]
    return default_hint


def build_records(
    path: Path,
    provenance: dict[str, list[str]],
    default_module: str,
    enforce_qc: bool,
) -> list[Record]:
    records: list[Record] = []
    seen: set[str] = set()
    for payload in load_jsonl(path):
        normalized = normalize_sft(payload)
        if not normalized:
            continue
        text_for_qc = " ".join((normalized["instruction"], normalized["output"]))
        if enforce_qc and not qc_ok(text_for_qc):
            continue
        record_payload = {
            "instruction": normalized["instruction"],
            "input": normalized["input"],
            "output": normalized["output"],
        }
        record_id = compute_record_id(record_payload)
        if record_id in seen:
            continue
        seen.add(record_id)
        module = derive_module(provenance.get(record_id, ()), default_module)
        tagged_instruction = f"[MOD={module}] {normalized['instruction']}"
        records.append(
            Record(
                module=module,
                instruction=tagged_instruction,
                input=normalized["input"],
                output=normalized["output"],
                record_id=record_id,
                source_paths=provenance.get(record_id, ()),
            )
        )
    return records


def split_train_val(
    records: Sequence[Record], val_pct: float, seed: int
) -> tuple[list[Record], list[Record]]:
    if not 0 <= val_pct < 1:
        raise ValueError("val_pct must be within [0, 1)")
    random.Random(seed).shuffle(records := list(records))
    val_size = int(len(records) * val_pct)
    val_records = list(records[:val_size])
    train_records = list(records[val_size:])
    return train_records, val_records


def write_jsonl(path: Path, records: Iterable[Record]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_payload(), ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)

    repo_path = Path(args.repo_sft)
    agent_path = Path(args.agent_sft)
    provenance_path = Path(args.provenance)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    provenance = load_provenance(provenance_path)

    repo_records = build_records(
        repo_path,
        provenance,
        default_module="General",
        enforce_qc=args.strict_qc,
    )
    agent_records = build_records(
        agent_path,
        provenance,
        default_module="API",
        enforce_qc=args.strict_qc,
    )

    combined = repo_records + agent_records
    if not combined:
        raise SystemExit(
            "No eligible records found. Run the analyzer first or adjust filters."
        )

    train_records, val_records = split_train_val(combined, args.val_pct, args.seed)

    write_jsonl(out_dir / "train.jsonl", train_records)
    write_jsonl(out_dir / "val.jsonl", val_records)

    def describe(records: Sequence[Record]) -> str:
        module_counts = Counter(rec.module for rec in records)
        return ", ".join(
            f"{module}:{count}" for module, count in module_counts.most_common()
        )

    print("[DONE] Multi-module dataset built")
    print(f"       output_dir={out_dir}")
    print(
        "       summary="
        f"train={len(train_records)} ({describe(train_records)}) "
        f"val={len(val_records)} ({describe(val_records)})"
    )


if __name__ == "__main__":
    main()
