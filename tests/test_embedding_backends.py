from monGARS.core.embedding_backends import normalise_embedding_backend


def test_dolphin_backend_is_supported() -> None:
    assert normalise_embedding_backend("Dolphin-X1-LLM2Vec") == "dolphin-x1-llm2vec"
