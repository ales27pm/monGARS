from pathlib import Path

from modules.neurons.registry import (
    MANIFEST_FILENAME,
    load_manifest,
    update_manifest,
)


def _create_summary(registry: Path, run_name: str) -> dict[str, object]:
    adapter_dir = registry / run_name / "adapter"
    adapter_dir.mkdir(parents=True)
    weights_path = adapter_dir / "weights.json"
    weights_path.write_text(f"{{\"run\": \"{run_name}\"}}")
    return {
        "status": "success",
        "artifacts": {
            "adapter": adapter_dir.as_posix(),
            "weights": weights_path.as_posix(),
        },
        "metrics": {"loss": 0.1},
    }


def test_update_manifest_tracks_current_and_history(tmp_path: Path) -> None:
    registry = tmp_path / "encoders"
    registry.mkdir()
    manifest = update_manifest(registry, _create_summary(registry, "run-1"))
    manifest_path = registry / MANIFEST_FILENAME
    assert manifest_path.exists()
    assert manifest.current is not None
    first_version = manifest.current.version
    payload = manifest.build_payload()
    assert payload["adapter_path"].endswith("run-1/adapter")

    manifest_second = update_manifest(registry, _create_summary(registry, "run-2"))
    assert manifest_second.current is not None
    assert manifest_second.current.version != first_version
    assert manifest_second.history
    assert manifest_second.history[0].version == first_version

    loaded = load_manifest(registry)
    assert loaded is not None
    assert loaded.current is not None
    assert loaded.current.version == manifest_second.current.version


def test_update_manifest_rejects_missing_adapter(tmp_path: Path) -> None:
    registry = tmp_path / "encoders"
    registry.mkdir()
    summary = {"status": "success", "artifacts": {}}
    try:
        update_manifest(registry, summary)
    except ValueError as exc:
        assert "artifacts.adapter" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError for missing adapter path")
