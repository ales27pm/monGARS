"""Render the dataset summary markdown for the GitHub Actions job summary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build_markdown(summary: dict) -> str:
    lines = [
        "# Québec-French dataset build",
        "",
        f"* Total records: {summary['total_records']}",
        f"* Train records: {summary['train_records']}",
        f"* Validation records: {summary['validation_records']}",
        f"* Strict QC: {summary['strict_qc']}",
        "",
        "## Source buckets",
    ]

    for key, value in summary.get("source_counts", {}).items():
        lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("## Requested ratios")
    for key, value in summary.get("requested_ratios", {}).items():
        lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("## Selected counts")
    for key, value in summary.get("selected_counts", {}).items():
        lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("## Actual ratios")
    for key, value in summary.get("actual_ratios", {}).items():
        lines.append(f"- {key}: {value:.4f}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-dir",
        default=os.environ.get("FINAL_OUTPUT_DIR", "data/final"),
        help="Directory containing dataset_summary.json",
    )
    parser.add_argument(
        "--output",
        default="summary.md",
        help="Path for the rendered markdown summary",
    )
    args = parser.parse_args()

    final_dir = Path(args.final_dir)
    summary_path = final_dir / "dataset_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Dataset summary not found at {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = build_markdown(summary)
    output_path = Path(args.output)
    output_path.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
