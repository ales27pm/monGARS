"""Tests for dataset helpers used in fine-tuning pipelines."""

from __future__ import annotations

from pathlib import Path

import pytest

from monGARS.mlops import dataset as dataset_module


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "dataset.jsonl"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_jsonl_records_supports_pretty_printed_objects(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """{
  \"prompt\": \"Alpha\",
  \"completion\": \"Bravo\"
}

{
  \"prompt\": \"Charlie\",
  \"completion\": \"Delta\"
}
""",
    )

    records = dataset_module._load_jsonl_records(path)

    assert records == [
        {"prompt": "Alpha", "completion": "Bravo"},
        {"prompt": "Charlie", "completion": "Delta"},
    ]


def test_load_jsonl_records_accepts_single_line_json(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        '{"prompt": "Ping", "completion": "Pong"}\n{"prompt": "Foo", "completion": "Bar"}\n',
    )

    records = dataset_module._load_jsonl_records(path)

    assert records == [
        {"prompt": "Ping", "completion": "Pong"},
        {"prompt": "Foo", "completion": "Bar"},
    ]


def test_load_jsonl_records_raises_on_unterminated_object(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """{
  \"prompt\": \"Unfinished\",
  \"completion\": \"Record\"
""",
    )

    with pytest.raises(ValueError):
        dataset_module._load_jsonl_records(path)
