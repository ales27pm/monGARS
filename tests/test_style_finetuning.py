import pytest

from monGARS.core.style_finetuning import (
    PromptBuilder,
    StyleAdapterState,
    StyleFineTuner,
    StyleFineTuningConfig,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:AutoAWQ is officially deprecated.*:DeprecationWarning"
)


@pytest.mark.asyncio
async def test_style_finetuner_trains_and_generates(tmp_path) -> None:
    config = StyleFineTuningConfig(
        base_model="hf-internal-testing/tiny-random-gpt2",
        allow_placeholder_training=True,
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


@pytest.mark.asyncio
async def test_estimate_personality_skips_placeholder_training_by_default(
    tmp_path,
) -> None:
    def fail_train(*_args, **_kwargs):
        raise AssertionError("train should not run")

    tuner = StyleFineTuner(
        config=StyleFineTuningConfig(
            base_model="hf-internal-testing/tiny-random-gpt2",
            adapter_repository=tmp_path,
            min_samples=1,
        ),
        device="cpu",
    )
    tuner._trainer = type("_Trainer", (), {"train": staticmethod(fail_train)})()
    interactions = [
        {"message": "Salut", "response": "Bonjour, ravi de vous aider."},
    ]

    analysis = await tuner.estimate_personality("user-skip", interactions)

    assert analysis.sample_count == 1
    assert analysis.confidence >= 0.4


def test_apply_style_skips_placeholder_model(tmp_path) -> None:
    tuner = StyleFineTuner.__new__(StyleFineTuner)
    tuner.config = StyleFineTuningConfig(
        base_model="hf-internal-testing/tiny-random-gpt2",
        adapter_repository=tmp_path,
    )
    tuner._prompt_builder = PromptBuilder(StyleFineTuner.STYLE_KEYS)
    tuner._adapter_cache = type(
        "_Cache",
        (),
        {
            "get": staticmethod(
                lambda _user_id: StyleAdapterState(
                    fingerprint="fingerprint",
                    model=object(),
                    tokenizer=None,  # type: ignore[arg-type]
                    sample_count=2,
                    updated_at=0.0,
                )
            )
        },
    )()
    tuner._inference = type(
        "_Inference",
        (),
        {
            "apply_style": staticmethod(
                lambda _state, _prompt: "generated output that should not be used"
            )
        },
    )()

    adapted = tuner.apply_style("user-1", "Voici la reponse.", {"formality": 0.8})

    assert adapted == "Voici la reponse."


def test_looks_like_degraded_output_detects_gibberish() -> None:
    assert StyleFineTuner._looks_like_degraded_output(
        "Slot fallback unavailable",
        (
            "comp 1ann;Mectend Briturn 00her v beganTge@ high&Yedish "
            "charralral Itif comm setward aptbumwayensensassftve"
        ),
    )
