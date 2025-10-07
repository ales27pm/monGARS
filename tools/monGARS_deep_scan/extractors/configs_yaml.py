from __future__ import annotations

from pathlib import Path
from typing import Any, List

try:  # pragma: no cover - import guard exercised in tests
    from ruamel.yaml import YAML

    pyyaml = None  # type: ignore[assignment]
except ModuleNotFoundError:  # pragma: no cover - fallback path
    YAML = None  # type: ignore[assignment]
    try:
        import yaml as pyyaml
    except ModuleNotFoundError:  # pragma: no cover
        pyyaml = None  # type: ignore[assignment]

from ..utils.text_clean import normalise_whitespace, split_paragraphs
from .types import ExtractionRecord


def _load_yaml_documents(text: str) -> List[Any]:
    if YAML is not None:
        yaml_parser = YAML(typ="safe")
        yaml_parser.preserve_quotes = False
        try:
            documents = list(yaml_parser.load_all(text))
        except Exception:
            return []
        return [doc for doc in documents if doc is not None]
    if pyyaml is None:
        return []
    try:
        documents = list(pyyaml.safe_load_all(text))
    except Exception:
        return []
    return [doc for doc in documents if doc is not None]


def _build_step_instruction(job_name: str, step: dict) -> str:
    name = step.get("name") or step.get("id") or "step"
    return f"Execute workflow step '{name}' in job '{job_name}'"


def extract(path: Path, text: str) -> List[ExtractionRecord]:
    records: List[ExtractionRecord] = []
    documents = _load_yaml_documents(text)
    for doc in documents:
        if not isinstance(doc, dict):
            continue

        if "jobs" in doc and isinstance(doc["jobs"], dict):
            for job_name, job in doc["jobs"].items():
                steps = job.get("steps") if isinstance(job, dict) else None
                if not isinstance(steps, list):
                    continue
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    run_command = step.get("run")
                    uses = step.get("uses")
                    if run_command or uses:
                        lc = getattr(step, "lc", None)
                        start_line = getattr(lc, "data", [None])[0]
                        if start_line is None and hasattr(lc, "line"):
                            start_line = lc.line
                        if start_line is None:
                            start_line = 0
                        start_line = int(start_line) + 1
                        output_payload = {
                            key: value
                            for key, value in step.items()
                            if key in {"run", "uses", "with", "env", "shell"}
                        }
                        records.append(
                            ExtractionRecord.for_agent(
                                instruction=_build_step_instruction(job_name, step),
                                output=output_payload,
                                source_file=str(path),
                                start_line=start_line,
                                end_line=start_line,
                                type_label="workflow_step",
                            )
                        )

        for key in ("description", "summary", "notes"):
            value = doc.get(key)
            if isinstance(value, str) and len(value) >= 60:
                for paragraph, start, end in split_paragraphs(value):
                    records.append(
                        ExtractionRecord.for_embedding(
                            text=normalise_whitespace(paragraph),
                            source_file=str(path),
                            start_line=start,
                            end_line=end,
                            type_label=f"yaml_{key}",
                        )
                    )
    return records
