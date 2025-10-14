import json
from pathlib import Path

import pytest

from monGARS.mlops.code_analysis import LLMUsage, ModuleInteraction
from monGARS.mlops.dataset import (
    build_module_interaction_dataset,
    build_mongars_strategy_dataset,
    prepare_local_instruction_dataset,
)


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int,
        return_attention_mask: bool,
        padding: str | None = None,
    ) -> dict[str, list[int]]:
        tokens = list(range(min(len(text), max_length)))
        if padding == "max_length":
            tokens = tokens + [0] * (max_length - len(tokens))
            attention = [1] * len(tokens)
        else:
            attention = [1] * len(tokens)
        return {"input_ids": tokens, "attention_mask": attention}


@pytest.fixture()
def tokenizer() -> DummyTokenizer:
    return DummyTokenizer()


def _usage(idx: int) -> LLMUsage:
    return LLMUsage(
        file_path=Path(f"module_{idx}.py"),
        line=idx,
        symbol=f"Class{idx}::method",
        framework="transformers" if idx % 2 == 0 else "llm2vec",
        call="AutoModel" if idx % 2 == 0 else "LLM2Vec",
        snippet="0010: pass",
    )


def test_build_strategy_dataset_and_prepare(
    tokenizer: DummyTokenizer, tmp_path: Path
) -> None:
    usages = [_usage(i) for i in range(1, 5)]
    dataset_path = tmp_path / "dataset.jsonl"
    metadata_path = tmp_path / "dataset.meta.json"

    build_mongars_strategy_dataset(
        usages,
        dataset_path,
        metadata_path=metadata_path,
        min_examples=4,
    )

    assert dataset_path.exists()
    assert json.loads(metadata_path.read_text())["num_examples"] == 4

    dataset = prepare_local_instruction_dataset(dataset_path, tokenizer, max_seq_len=32)
    assert len(dataset) == 4


def test_build_module_interaction_dataset(tmp_path: Path) -> None:
    interactions = [
        ModuleInteraction(
            source_path=Path("monGARS/mlops/dataset.py"),
            line=10,
            source_module="monGARS.mlops.dataset",
            target_module="monGARS.mlops.code_analysis",
            import_names=("build_strategy_recommendation",),
            kind="from",
            snippet="0008: from monGARS.mlops import code_analysis",
        ),
        ModuleInteraction(
            source_path=Path("monGARS/mlops/pipelines/unsloth.py"),
            line=20,
            source_module="monGARS.mlops.pipelines.unsloth",
            target_module="monGARS.mlops.dataset",
            import_names=("build_module_interaction_dataset",),
            kind="import",
            snippet="0019: import monGARS.mlops.dataset as dataset",
        ),
        ModuleInteraction(
            source_path=Path("modules/evolution_engine/orchestrator.py"),
            line=30,
            source_module="modules.evolution_engine.orchestrator",
            target_module="monGARS.core.persistence",
            import_names=("PersistenceRepository",),
            kind="from",
            snippet="0030: from monGARS.core.persistence import PersistenceRepository",
        ),
        ModuleInteraction(
            source_path=Path("monGARS/mlops/code_analysis.py"),
            line=15,
            source_module="monGARS.mlops.code_analysis",
            target_module="modules.neurons.training",
            import_names=("MNTPTrainer",),
            kind="from",
            snippet="0015: from modules.neurons.training import MNTPTrainer",
        ),
        ModuleInteraction(
            source_path=Path("monGARS/mlops/code_analysis.py"),
            line=40,
            source_module="monGARS.mlops.code_analysis",
            target_module="monGARS.core.logging",
            import_names=("get_logger",),
            kind="from",
            snippet="0040: from monGARS.core.logging import get_logger",
        ),
        ModuleInteraction(
            source_path=Path("monGARS/mlops/code_analysis.py"),
            line=55,
            source_module="monGARS.mlops.code_analysis",
            target_module="monGARS.core.settings",
            import_names=("get_settings",),
            kind="from",
            snippet="0055: from monGARS.core.settings import get_settings",
        ),
        ModuleInteraction(
            source_path=Path("monGARS/mlops/code_analysis.py"),
            line=60,
            source_module="monGARS.mlops.code_analysis",
            target_module="monGARS.core.telemetry",
            import_names=("Telemetry",),
            kind="from",
            snippet="0060: from monGARS.core.telemetry import Telemetry",
        ),
        ModuleInteraction(
            source_path=Path("monGARS/mlops/code_analysis.py"),
            line=70,
            source_module="monGARS.mlops.code_analysis",
            target_module="monGARS.core.telemetry",
            import_names=("Span",),
            kind="from",
            snippet="0070: from monGARS.core.telemetry import Span",
        ),
    ]
    dataset_path = tmp_path / "module_interactions.jsonl"
    metadata_path = tmp_path / "module_interactions.meta.json"

    build_module_interaction_dataset(
        interactions,
        dataset_path,
        metadata_path=metadata_path,
        min_examples=4,
    )

    payload = dataset_path.read_text(encoding="utf-8").splitlines()
    assert len(payload) == len(interactions)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["num_examples"] == len(interactions)
