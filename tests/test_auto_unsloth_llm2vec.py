import json
from pathlib import Path

from scripts.auto_unsloth_llm2vec import _discover_dataset_prep_exports


def _write_metadata(base: Path, data: dict) -> Path:
    metadata = base / "dataset_metadata.json"
    metadata.write_text(json.dumps(data), encoding="utf-8")
    return metadata


def test_discover_dataset_prep_exports_uses_metadata(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    prep_dir.mkdir()
    unsloth = prep_dir / "unsloth_prompt_completion.jsonl"
    unsloth.write_text("{}\n", encoding="utf-8")
    _write_metadata(
        prep_dir,
        {
            "derived_outputs": {
                "unsloth_prompt_completion": {
                    "path": unsloth.name,
                    "records": 1,
                }
            }
        },
    )

    discovered = _discover_dataset_prep_exports([prep_dir])
    assert unsloth.resolve() in discovered


def test_discover_dataset_prep_exports_accepts_metadata_file(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    prep_dir.mkdir()
    export = prep_dir / "custom.jsonl"
    export.write_text("{}\n", encoding="utf-8")
    metadata = _write_metadata(
        prep_dir,
        {"derived_outputs": {"unsloth_prompt_completion": {"path": "custom.jsonl"}}},
    )

    discovered = _discover_dataset_prep_exports([metadata])
    assert export.resolve() in discovered


def test_discover_dataset_prep_exports_falls_back_to_default(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    prep_dir.mkdir()
    fallback = prep_dir / "combined_instruct.unsloth.jsonl"
    fallback.write_text("{}\n", encoding="utf-8")

    discovered = _discover_dataset_prep_exports([prep_dir])
    assert fallback.resolve() in discovered
