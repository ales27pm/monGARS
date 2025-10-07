"""Assemble curated train/validation splits for the custom dataset workflow."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable

from tools.monGARS_deep_scan.utils.hashing import stable_hash

MIN_TEXT = 12
MAX_OUTPUT_CHARS = 3000


def _load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"::warning::Skipping invalid JSON in {path} line {line_no}: {exc}",
                    file=sys.stderr,
                )


def _normalise(record: dict) -> dict | None:
    instruction = (record.get("instruction") or "").strip()
    input_text = (record.get("input") or "").strip()
    output = record.get("output")

    if isinstance(output, (dict, list)):
        output = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
    elif output is None:
        output = ""
    else:
        output = str(output).strip()

    if len(instruction) < MIN_TEXT or len(output) < MIN_TEXT:
        return None
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS].rsplit(" ", 1)[0] + " …"

    return {"instruction": instruction, "input": input_text, "output": output}


def _parse_ratios(raw: str) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            print(
                f"::warning::Ignoring malformed ratio entry '{part}'", file=sys.stderr
            )
            continue
        key, value = part.split(":", 1)
        try:
            ratios[key.strip()] = float(value)
        except ValueError:
            print(
                f"::warning::Invalid ratio value '{value}' for key '{key.strip()}'",
                file=sys.stderr,
            )

    if not ratios:
        ratios = {"frca": 0.5, "agent": 0.4, "repo": 0.1}

    total = sum(ratios.values())
    if total > 0:
        ratios = {key: value / total for key, value in ratios.items()}

    for bucket in ("frca", "agent", "repo"):
        ratios.setdefault(bucket, 0.0)

    return ratios


def _hash_row(row: dict) -> str:
    return stable_hash(
        [row.get("instruction", ""), row.get("input", ""), row.get("output", "")]
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-dir",
        default=os.environ.get("SCAN_OUTPUT_DIR", "data/deep_scan"),
        help="Directory containing deep scan outputs",
    )
    parser.add_argument(
        "--final-dir",
        default=os.environ.get("FINAL_OUTPUT_DIR", "data/final"),
        help="Directory that will receive train/val splits",
    )
    parser.add_argument(
        "--ratios",
        default=os.environ.get("DATASET_RATIOS", "frca:0.50,agent:0.40,repo:0.10"),
        help="Comma-delimited sampling ratios, e.g. frca:0.5,agent:0.4,repo:0.1",
    )
    parser.add_argument(
        "--val-pct",
        type=float,
        default=float(os.environ.get("DATASET_VAL_PCT", 0.06)),
        help="Validation split fraction expressed as a decimal",
    )
    parser.add_argument(
        "--strict-qc",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("DATASET_STRICT_QC", True),
        help="Require fr-CA QC flag for inclusion in repo/frca buckets",
    )
    parser.add_argument(
        "--seed",
        default=(
            os.environ.get("DATASET_SEED")
            or os.environ.get("GITHUB_RUN_ID")
            or os.environ.get("GITHUB_SHA")
            or "42"
        ),
        help="Random seed for deterministic sampling",
    )
    return parser.parse_args(argv)


def assemble_dataset(
    scan_dir: Path,
    final_dir: Path,
    *,
    ratios_raw: str,
    val_pct: float,
    strict_qc: bool,
    seed_value: str | int,
) -> dict:
    try:
        random.seed(int(seed_value))
    except ValueError:
        random.seed(seed_value)

    if not scan_dir.exists():
        raise SystemExit(f"Scan directory {scan_dir} not found")
    final_dir.mkdir(parents=True, exist_ok=True)

    ratios = _parse_ratios(ratios_raw)

    if val_pct < 0:
        val_pct = 0.0
    elif val_pct > 1:
        val_pct = 1.0

    sft_path = scan_dir / "sft_dataset.jsonl"
    agent_path = scan_dir / "agent_handoff_dataset.jsonl"
    if not sft_path.exists() or not agent_path.exists():
        raise SystemExit("Deep scan outputs missing required dataset files")

    sft_records: list[tuple[dict, bool]] = []
    for record in _load_jsonl(sft_path):
        parsed = _normalise(record)
        if not parsed:
            continue
        qc_flag = bool(record.get("_meta", {}).get("qc_fr_ca", False))
        sft_records.append((parsed, qc_flag))

    buckets: dict[str, list[dict]] = {"frca": [], "agent": [], "repo": []}
    for parsed, qc_flag in sft_records:
        if qc_flag or not strict_qc:
            buckets["frca"].append(parsed)
        if qc_flag or not strict_qc:
            buckets["repo"].append(parsed)

    for record in _load_jsonl(agent_path):
        parsed = _normalise(record)
        if not parsed:
            continue
        buckets["agent"].append(parsed)

    record_sources: dict[str, set[str]] = {}
    all_records: list[dict] = []

    for bucket_name, records in buckets.items():
        deduped: list[dict] = []
        seen: set[str] = set()
        for row in records:
            row_hash = _hash_row(row)
            if row_hash in seen:
                continue
            seen.add(row_hash)
            deduped.append(row)
            record_sources.setdefault(row_hash, set()).add(bucket_name)
        buckets[bucket_name] = deduped
        all_records.extend(deduped)

    if not all_records:
        raise SystemExit("No qualifying records available for dataset assembly")

    unique_total = len({_hash_row(row) for row in all_records})
    source_counts = {key: len(value) for key, value in buckets.items()}
    requested_total = max(unique_total, 1)

    mixed: list[dict] = []
    selected_ids: set[str] = set()
    selected_breakdown: dict[str, int] = {"frca": 0, "agent": 0, "repo": 0}

    for key in ("frca", "agent", "repo"):
        records = buckets.get(key, [])
        if not records:
            continue
        random.shuffle(records)
        target = int(round(ratios.get(key, 0.0) * requested_total))
        if ratios.get(key, 0.0) > 0 and target == 0:
            target = 1
        target = min(target, len(records))

        taken = 0
        for row in records:
            row_hash = _hash_row(row)
            if row_hash in selected_ids:
                continue
            mixed.append(row)
            selected_ids.add(row_hash)
            selected_breakdown[key] = selected_breakdown.get(key, 0) + 1
            taken += 1
            if taken >= target:
                break

    min_required = min(unique_total, requested_total)
    if len(mixed) < min_required:
        pool = all_records.copy()
        random.shuffle(pool)
        for row in pool:
            if len(mixed) >= min_required:
                break
            row_hash = _hash_row(row)
            if row_hash in selected_ids:
                continue
            mixed.append(row)
            selected_ids.add(row_hash)
            origins = sorted(record_sources.get(row_hash, {"repo"}))
            origin = origins[0] if origins else "repo"
            selected_breakdown[origin] = selected_breakdown.get(origin, 0) + 1

    random.shuffle(mixed)

    val_count = int(len(mixed) * val_pct)
    validation = mixed[:val_count]
    training = mixed[val_count:]

    datasets = {
        final_dir / "train.jsonl": training,
        final_dir / "val.jsonl": validation,
    }

    for path, records in datasets.items():
        with path.open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    generated_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    summary = {
        "generated_at": generated_at.replace("+00:00", "Z"),
        "commit": os.environ.get("GITHUB_SHA"),
        "requested_ratios": ratios,
        "strict_qc": strict_qc,
        "validation_fraction": val_pct,
        "train_records": len(training),
        "validation_records": len(validation),
        "total_records": len(mixed),
        "source_counts": source_counts,
        "selected_counts": selected_breakdown,
        "actual_ratios": {
            key: (selected_breakdown.get(key, 0) / len(mixed) if mixed else 0.0)
            for key in sorted(selected_breakdown)
        },
    }

    summary_path = final_dir / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = assemble_dataset(
        Path(args.scan_dir).resolve(),
        Path(args.final_dir).resolve(),
        ratios_raw=args.ratios,
        val_pct=args.val_pct,
        strict_qc=args.strict_qc,
        seed_value=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
