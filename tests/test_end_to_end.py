from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.monGARS_deep_scan import deep_scan


@pytest.fixture()
def fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()

    docs_dir = repo / "docs"
    docs_dir.mkdir()
    docs_dir.joinpath("guide.md").write_text(
        """
# Guide

User: Salut, peux-tu vérifier la configuration?
Assistant: Pas de stress, on règle ça icitte avec la meilleure pratique.

Ce paragraphe dépasse largement les soixante caractères pour être considéré
comme un bon candidat dans le corpus d'embedding, incluant une référence à la
poutine et au dépanneur du quartier.
""",
        encoding="utf-8",
    )

    repo.joinpath("script.py").write_text(
        '"""\nUser: Comment magasiner un bon modem?\nAssistant: Commence au dépanneur puis consulte la doc officielle.\n"""\n\n'
        "def plan():\n"
        '    """Ce docstring long explique comment orchestrer les étapes du pipeline pour générer des artefacts cohérents et déterministes."""\n'
        "    return True\n",
        encoding="utf-8",
    )

    workflows = repo / ".github" / "workflows"
    workflows.mkdir(parents=True)
    workflows.joinpath("build.yml").write_text(
        """
name: Build
jobs:
  test:
    steps:
      - name: Run tests
        run: pytest -q
        shell: bash
""",
        encoding="utf-8",
    )

    repo.joinpath("Dockerfile").write_text(
        """
FROM python:3.11-slim
RUN echo "icitte" && echo "on prépare une poutine"
CMD [\"python\", \"main.py\"]
""",
        encoding="utf-8",
    )

    repo.joinpath("script.sh").write_text(
        """
# Ce script explique comment magasiner de la sauce à poutine
# dans un dépanneur de Montréal pour le brunch.
echo "Usage: ./script.sh --qc"
""",
        encoding="utf-8",
    )

    return repo


def test_end_to_end_scan_creates_expected_outputs(
    tmp_path: Path, fixture_repo: Path
) -> None:
    output_dir = tmp_path / "output"
    exit_code = deep_scan.main(
        [
            "--input",
            str(fixture_repo),
            "--out",
            str(output_dir),
        ]
    )
    assert exit_code == 0

    sft_path = output_dir / "sft_dataset.jsonl"
    agent_path = output_dir / "agent_handoff_dataset.jsonl"
    emb_path = output_dir / "embeddings_corpus.jsonl"
    provenance_path = output_dir / "provenance.csv"
    report_path = output_dir / "report.md"
    log_path = output_dir / "logs" / "scan.log"

    for path in [
        sft_path,
        agent_path,
        emb_path,
        provenance_path,
        report_path,
        log_path,
    ]:
        assert path.exists(), f"Expected output file {path} to exist"

    with sft_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
    assert first_line, "SFT dataset should contain at least one record"
    record = json.loads(first_line)
    assert "instruction" in record and "output" in record
    assert record["_meta"]["qc_fr_ca"] is True

    with agent_path.open("r", encoding="utf-8") as handle:
        agent_record = json.loads(handle.readline())
    assert isinstance(agent_record["output"], dict)

    with emb_path.open("r", encoding="utf-8") as handle:
        emb_record = json.loads(handle.readline())
    assert len(emb_record["text"]) >= 60

    with provenance_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert rows, "Provenance CSV should not be empty"
    assert {
        "record_id",
        "dataset",
        "source_file",
        "start_line",
        "end_line",
        "type",
        "qc_fr_ca",
    }.issubset(reader.fieldnames or [])

    report_text = report_path.read_text(encoding="utf-8")
    assert "Deep Scan Report" in report_text
    assert "sft_dataset" in report_text

    log_text = log_path.read_text(encoding="utf-8")
    assert "Starting deep scan" in log_text
