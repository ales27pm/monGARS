import json
from pathlib import Path

import pytest

from monGARS.mlops.code_analysis import LLMUsage
from monGARS.mlops.dataset import (
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
