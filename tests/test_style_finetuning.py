import pytest

from monGARS.core.style_finetuning import StyleFineTuner, StyleFineTuningConfig

pytestmark = pytest.mark.filterwarnings(
    "ignore:AutoAWQ is officially deprecated.*:DeprecationWarning"
)


@pytest.mark.asyncio
async def test_style_finetuner_trains_and_generates(tmp_path) -> None:
    config = StyleFineTuningConfig(
        base_model="hf-internal-testing/tiny-random-gpt2",
        adapter_repository=tmp_path,
        max_steps=1,
        training_epochs=1.0,
        micro_batch_size=1,
        min_samples=1,
        max_history_messages=4,
        max_new_tokens=16,
    )
    tuner = StyleFineTuner(config=config, device="cpu")
    interactions = [
        {"message": "Salut", "response": "Bonjour, ravi de vous aider."},
        {"message": "Merci", "response": "Toujours heureux d'aider!"},
    ]

    analysis = await tuner.estimate_personality("user-42", interactions)

    assert analysis.sample_count >= 1
    assert 0 <= analysis.traits["openness"] <= 1
    assert 0 <= analysis.style["formality"] <= 1

    adapted = tuner.apply_style("user-42", "Voici la réponse.", analysis.style)

    assert isinstance(adapted, str)
    assert adapted
