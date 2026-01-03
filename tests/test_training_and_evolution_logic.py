import json
from pathlib import Path

import pytest

from monGARS.core.persistence import PersistenceRepository
from modules.evolution_engine.energy import EnergyTracker
from modules.evolution_engine.self_training import _extract_text, collect_curated_data


def test_persistence_vector_normalisation_and_payload():
    repo = PersistenceRepository(enable_embeddings=False)
    vector = repo._normalise_vector([1, 2, 3])  # type: ignore[attr-defined]
    assert len(vector) == int(repo._settings.llm2vec_vector_dimensions)
    assert vector[:3] == [1.0, 2.0, 3.0]

    payload = repo._compose_history_payload(" hi ", " there ")  # type: ignore[attr-defined]
    assert "User: hi" in payload and "Assistant: there" in payload


def test_collect_curated_data_reads_jsonl(tmp_path: Path):
    dataset_dir = tmp_path / "curated"
    dataset_dir.mkdir()
    file_path = dataset_dir / "dataset.jsonl"
    file_path.write_text(
        """{"text": "hello"}
{"prompt": "ask", "response": "reply", "embedding": [0.1]}
{"text_preview": " ", "response": "  "}
""",
        encoding="utf-8",
    )

    latest_path = dataset_dir / "latest.json"
    latest_path.write_text(json.dumps({"dataset_file": str(file_path)}), encoding="utf-8")

    class DummyDatasetCatalog:
        def __init__(self, root):
            self.root = root

        def latest(self):
            return type("Latest", (), {"dataset_file": file_path})

    # Patch catalog to avoid touching real filesystem layout
    from modules.evolution_engine import self_training

    self_training.DatasetCatalog = DummyDatasetCatalog  # type: ignore[assignment]
    records = collect_curated_data(dataset_root=dataset_dir, limit=5)
    if hasattr(records, "to_dict"):
        # Dataset-like object
        as_list = records.to_dict()
        assert any(as_list.values())
    else:
        assert isinstance(records, list)
        assert records[0]["text"] == "hello"
        assert records[1]["text"] == "reply"
    assert _extract_text({"text_preview": "preview"}) == "preview"


def test_energy_tracker_reports_usage(monkeypatch: pytest.MonkeyPatch):
    class FakeProcess:
        def __init__(self):
            self.calls = []
            self.cpu_time = 0.0

        def cpu_times(self):  # noqa: D401
            self.calls.append("cpu_times")
            self.cpu_time += 1.0
            return type("Times", (), {"user": self.cpu_time, "system": 0.0})

    tracker = EnergyTracker(baseline_cpu_power_watts=10.0, process=FakeProcess())
    tracker.start()
    report = tracker.stop()
    assert report.energy_wh >= 0.0
    assert report.backend in {"psutil", "codecarbon"}
    assert tracker.last_report == report
