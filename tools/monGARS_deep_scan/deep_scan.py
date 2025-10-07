from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional
from zipfile import ZipFile

from .dataset.builders import DatasetBuilders
from .dataset.provenance import ProvenanceTracker
from .dataset.qc_filter import QCFilter
from .extractors import code_py, configs_yaml, dockerfiles, html_jsx, shells, text_docs
from .extractors.types import ExtractionRecord
from .utils.io import ensure_directory, read_text_file
from .utils.log import configure_logging, get_logger

DEFAULT_EXTENSIONS = {
    ".py",
    ".md",
    ".rst",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".sh",
    ".sql",
    ".html",
    ".htm",
    ".jsx",
    ".tsx",
}

EXCLUDE_DIRECTORIES = {
    ".git",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "dist",
    "build",
    "output",
    "logs",
}

DOCKERFILE_NAMES = {
    "dockerfile",
    "dockerfile.gpu",
    "dockerfile.embedded",
    "dockerfile.embedded.gpu",
}


@dataclass
class ScanConfig:
    input_path: Path
    output_dir: Path
    allow_network: bool
    max_lines: int
    jobs: int
    dry_run: bool
    qc_terms: Optional[Path]
    include_ext: Optional[List[str]]
    exclude_dirs: Optional[List[str]]


def parse_args(argv: Optional[List[str]] = None) -> ScanConfig:
    parser = argparse.ArgumentParser(
        description="monGARS deep scan and dataset builder"
    )
    parser.add_argument("--input", required=True, help="Path to repo directory or ZIP")
    parser.add_argument("--out", default="output", help="Output directory")
    parser.add_argument(
        "--allow-network", action="store_true", help="Enable network access (unused)"
    )
    parser.add_argument(
        "--max-lines", type=int, default=50000, help="Skip files longer than this"
    )
    parser.add_argument("--jobs", type=int, default=0, help="Parallel workers")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without writing output",
    )
    parser.add_argument("--qc-terms", type=str, help="Path to custom QC terms list")
    parser.add_argument(
        "--include-ext", type=str, help="Comma-separated list of extensions to include"
    )
    parser.add_argument(
        "--exclude-dir", type=str, help="Comma-separated directories to exclude"
    )
    args = parser.parse_args(argv)

    jobs = args.jobs
    if jobs <= 0:
        jobs = min(8, max(1, os.cpu_count() or 1) * 2)

    include_ext = (
        [
            ext.strip() if ext.startswith(".") else f".{ext.strip()}"
            for ext in args.include_ext.split(",")
        ]
        if args.include_ext
        else None
    )
    exclude_dirs = (
        [part.strip() for part in args.exclude_dir.split(",") if part.strip()]
        if args.exclude_dir
        else None
    )

    return ScanConfig(
        input_path=Path(args.input).resolve(),
        output_dir=Path(args.out).resolve(),
        allow_network=args.allow_network,
        max_lines=args.max_lines,
        jobs=jobs,
        dry_run=args.dry_run,
        qc_terms=Path(args.qc_terms).resolve() if args.qc_terms else None,
        include_ext=include_ext,
        exclude_dirs=exclude_dirs,
    )


def _resolve_input_path(
    config: ScanConfig,
) -> tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    input_path = config.input_path
    if input_path.is_dir():
        return input_path, None
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        temp_dir = tempfile.TemporaryDirectory(prefix="monGARS_scan_")
        with ZipFile(input_path, "r") as archive:
            archive.extractall(temp_dir.name)
        return Path(temp_dir.name), temp_dir
    raise FileNotFoundError(f"Unsupported input path: {input_path}")


def _extension_matches(path: Path, include_ext: Iterable[str]) -> bool:
    if path.name.lower() in DOCKERFILE_NAMES:
        return True
    return path.suffix.lower() in include_ext


EXTRACTOR_MAP: Dict[str, Callable[[Path, str], List[ExtractionRecord]]] = {
    ".py": code_py.extract,
    ".md": text_docs.extract,
    ".rst": text_docs.extract,
    ".txt": text_docs.extract,
    ".json": text_docs.extract,
    ".yaml": configs_yaml.extract,
    ".yml": configs_yaml.extract,
    ".sh": shells.extract,
    ".sql": text_docs.extract,
    ".html": html_jsx.extract,
    ".htm": html_jsx.extract,
    ".jsx": html_jsx.extract,
    ".tsx": html_jsx.extract,
}


DOCKERFILE_EXTRACTOR = dockerfiles.extract


def _iter_files(
    root: Path, include_ext: Iterable[str], exclude_dirs: Iterable[str]
) -> List[Path]:
    results: List[Path] = []
    for current_root, dirs, files in os.walk(root):
        current_path = Path(current_root)
        dirs[:] = [
            d for d in dirs if d.lower() not in {e.lower() for e in exclude_dirs}
        ]
        for file_name in files:
            path = current_path / file_name
            if path.name.lower() in DOCKERFILE_NAMES:
                results.append(path)
                continue
            if _extension_matches(path, include_ext):
                results.append(path)
    return sorted(results)


def _load_text(root: Path, path: Path, max_lines: int) -> Optional[str]:
    return read_text_file(path, max_lines)


def _relative_path(root: Path, path: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def _dispatch_extractor(
    path: Path,
) -> Optional[Callable[[Path, str], List[ExtractionRecord]]]:
    if path.name.lower() in DOCKERFILE_NAMES:
        return DOCKERFILE_EXTRACTOR
    return EXTRACTOR_MAP.get(path.suffix.lower())


def _write_report(out_dir: Path, report: dict) -> None:
    report_path = out_dir / "report.md"
    lines: List[str] = []
    lines.append("# Deep Scan Report")
    lines.append("")
    lines.append("## Dataset Counts")
    for dataset, count in sorted(report.get("dataset_counts", {}).items()):
        lines.append(f"- **{dataset}**: {count}")
    lines.append("")
    lines.append("## Québécois French Ratios")
    for dataset, ratio in sorted(report.get("qc_ratios", {}).items()):
        lines.append(f"- **{dataset}**: {ratio * 100:.1f}%")
    lines.append("")
    lines.append("## Top Source Files")
    file_breakdown = report.get("file_breakdown", {})
    for dataset, files in sorted(file_breakdown.items()):
        lines.append(f"### {dataset}")
        for source, count in sorted(
            files.items(), key=lambda item: item[1], reverse=True
        )[:10]:
            lines.append(f"- {source}: {count}")
        lines.append("")
    lines.append("## Type Breakdown")
    for type_label, count in sorted(
        report.get("type_breakdown", {}).items(), key=lambda item: item[1], reverse=True
    ):
        lines.append(f"- {type_label}: {count}")
    lines.append("")
    lines.append("## Sample Records")
    for dataset, samples in report.get("samples", {}).items():
        lines.append(f"### {dataset}")
        for sample in samples[:10]:
            lines.append("```json")
            lines.append(json.dumps(sample, ensure_ascii=False, indent=2))
            lines.append("```")
        lines.append("")
    warnings: List[str] = []
    for dataset, count in report.get("dataset_counts", {}).items():
        if count == 0:
            warnings.append(f"Dataset {dataset} is empty")
    if warnings:
        lines.append("## Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    config = parse_args(argv)
    root, temp_dir = _resolve_input_path(config)
    exclude_dirs = config.exclude_dirs or list(EXCLUDE_DIRECTORIES)
    include_ext = config.include_ext or list(DEFAULT_EXTENSIONS)

    if config.dry_run:
        files = _iter_files(root, include_ext, exclude_dirs)
        print(f"Planned scan of {len(files)} files under {root}")
        if temp_dir:
            temp_dir.cleanup()
        return 0

    ensure_directory(config.output_dir)
    logs_dir = config.output_dir / "logs"
    ensure_directory(logs_dir)
    configure_logging(logs_dir / "scan.log")
    logger = get_logger()

    logger.info("Starting deep scan of %s", root)
    qc_filter = QCFilter.from_path(config.qc_terms)
    provenance = ProvenanceTracker()
    builders = DatasetBuilders(provenance, qc_filter)

    files = _iter_files(root, include_ext, exclude_dirs)
    logger.info("Identified %d files to scan", len(files))

    all_records: List[ExtractionRecord] = []
    for file_path in files:
        extractor = _dispatch_extractor(file_path)
        if extractor is None:
            continue
        text = _load_text(root, file_path, config.max_lines)
        if text is None:
            continue
        relative = _relative_path(root, file_path)
        try:
            records = extractor(relative, text)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Extractor failed for %s: %s", relative, exc)
            continue
        all_records.extend(records)

    logger.info("Collected %d candidate records", len(all_records))
    all_records.sort(
        key=lambda record: (
            record.source_file,
            record.start_line,
            record.end_line,
            record.dataset,
        )
    )

    for record in all_records:
        if record.dataset == "sft":
            builders.add_sft(
                instruction=record.instruction,
                input_text=record.input_text,
                output=str(record.output),
                source_file=record.source_file,
                start_line=record.start_line,
                end_line=record.end_line,
                type_label=record.type_label,
            )
        elif record.dataset == "agent":
            builders.add_agent(
                instruction=record.instruction,
                input_text=record.input_text,
                output=record.output,
                source_file=record.source_file,
                start_line=record.start_line,
                end_line=record.end_line,
                type_label=record.type_label,
            )
        elif record.dataset == "embeddings":
            builders.add_embedding(
                text=record.text,
                source_file=record.source_file,
                start_line=record.start_line,
                end_line=record.end_line,
                type_label=record.type_label,
            )

    builders.write_all(config.output_dir)
    provenance.write_csv(config.output_dir / "provenance.csv")
    _write_report(config.output_dir, builders.report())

    logger.info("Scan complete. Outputs written to %s", config.output_dir)

    if temp_dir:
        temp_dir.cleanup()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
