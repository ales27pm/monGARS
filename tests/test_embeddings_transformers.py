import numpy as np
import torch

from monGARS.config import get_settings
from monGARS.core import embeddings as embeddings_module


class DummyTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    padding_side = "right"

    def __call__(
        self,
        texts,
        *,
        padding,
        truncation,
        max_length,
        return_tensors,
    ):
        del padding, truncation, return_tensors
        batch = len(texts)
        seq_len = min(max_length, 4)
        input_ids = torch.arange(batch * seq_len, dtype=torch.long).reshape(
            batch, seq_len
        )
        attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class DummyOutput:
    def __init__(self, hidden_state: torch.Tensor) -> None:
        self.hidden_states = [hidden_state, hidden_state]


class DummyModel:
    def __init__(self) -> None:
        self.config = type("Cfg", (), {"hidden_size": 16})()

    def to(self, device: torch.device) -> "DummyModel":
        self._device = device
        return self

    def eval(self) -> "DummyModel":
        return self

    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch, seq_len = input_ids.shape
        hidden_state = torch.ones(
            batch,
            seq_len,
            self.config.hidden_size,
            dtype=torch.float32,
            device=input_ids.device,
        )
        return DummyOutput(hidden_state)


def test_transformers_embedding_backend_normalises_vectors(monkeypatch):
    monkeypatch.setenv("TRANSFORMERS_EMBEDDING_BATCH_SIZE", "2")
    monkeypatch.setenv("TRANSFORMERS_EMBEDDING_MAX_LENGTH", "8")
    monkeypatch.setenv("TRANSFORMERS_EMBEDDING_MODEL", "stub-model")
    get_settings.cache_clear()

    def fake_components(settings):
        del settings
        return DummyTokenizer(), DummyModel(), torch.device("cpu"), 16

    monkeypatch.setattr(
        embeddings_module,
        "_ensure_transformers_components",
        fake_components,
    )

    vectors = embeddings_module._encode_with_transformers(
        ["hello", "world"],
        "Embed the text",
    )

    assert vectors.shape == (2, 16)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.all(norms > 0.99)
    assert np.all(norms < 1.01)

    get_settings.cache_clear()
