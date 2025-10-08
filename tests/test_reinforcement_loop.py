from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from modules.neurons.training.reinforcement_loop import (
    AdaptiveScalingStrategy,
    BatchStatistics,
    EpisodeResult,
    PreferenceAlignmentLoop,
    PreferenceDatasetCurator,
    PreferenceSample,
    ReasoningRunSummary,
    ReinforcementLearningLoop,
    ReinforcementLoop,
    ThroughputAwareScalingStrategy,
    Transition,
)
from monGARS.core.operator_approvals import OperatorApprovalRegistry
from monGARS.core.self_training import SelfTrainingEngine


class _RecordingSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.attributes: dict[str, Any] = {}

    def __enter__(self) -> "_RecordingSpan":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - interface stub
        return None

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append((name, attributes or {}))

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value


class _RecordingTracer:
    def __init__(self) -> None:
        self.spans: list[_RecordingSpan] = []

    def start_as_current_span(self, name: str) -> _RecordingSpan:
        span = _RecordingSpan(name)
        self.spans.append(span)
        return span


class FixedScalingStrategy:
    """Scaling strategy that keeps the worker count constant."""

    def recommend_worker_count(
        self, current_workers: int, batch_index: int, stats: BatchStatistics
    ):
        return current_workers, "fixed"


class SimpleBanditEnvironment:
    """Single-step environment that returns configured rewards."""

    def __init__(self, rewards: list[float]) -> None:
        self._rewards = rewards

    def reset(self) -> int:
        return 0

    def step(self, action: int) -> tuple[int, float, bool, dict[str, float]]:
        reward = float(self._rewards[int(action)])
        return 0, reward, True, {"reward": reward}


@dataclass
class EpsilonGreedyPolicy:
    """Minimal epsilon-greedy policy for testing purposes."""

    action_count: int
    epsilon: float = 0.1
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.action_counts = [0 for _ in range(self.action_count)]
        self.action_values = [0.0 for _ in range(self.action_count)]

    def select_action(self, _: int) -> int:
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(self.action_count)
        best_index = max(
            range(self.action_count), key=lambda idx: self.action_values[idx]
        )
        return best_index

    def update(self, transitions: list[Transition]) -> None:
        for transition in transitions:
            action_index = int(transition.action)
            self.action_counts[action_index] += 1
            count = self.action_counts[action_index]
            step_size = 1.0 / count
            previous = self.action_values[action_index]
            self.action_values[action_index] = previous + step_size * (
                transition.reward - previous
            )

    def clone(self) -> "EpsilonGreedyPolicy":
        seed = self._rng.randrange(1, 1_000_000)
        clone = EpsilonGreedyPolicy(
            action_count=self.action_count,
            epsilon=self.epsilon,
            seed=seed,
        )
        clone.action_counts = list(self.action_counts)
        clone.action_values = list(self.action_values)
        return clone


def test_reinforcement_loop_updates_policy_towards_best_action() -> None:
    rewards = [0.0, 1.0]
    policy = EpsilonGreedyPolicy(action_count=len(rewards), epsilon=0.25, seed=123)
    loop = ReinforcementLearningLoop(
        environment_factory=lambda: SimpleBanditEnvironment(rewards),
        policy=policy,
        max_steps=1,
        scaling_strategy=FixedScalingStrategy(),
        initial_workers=2,
        max_workers=2,
    )

    summary = loop.run(total_episodes=40)

    assert summary.total_episodes == 40
    assert summary.failures == 0
    assert policy.action_counts[1] > policy.action_counts[0]
    assert summary.average_reward > 0.6
    assert len(summary.worker_history) == 1
    assert summary.worker_history[0].worker_count == 2
    assert summary.worker_history[0].reason == "initial"


def test_adaptive_scaling_strategy_reacts_to_rewards() -> None:
    strategy = AdaptiveScalingStrategy(
        min_workers=1,
        max_workers=6,
        increase_factor=1.4,
        decrease_factor=0.5,
        improvement_threshold=0.1,
        regression_tolerance=0.15,
        variance_threshold=0.05,
        max_duration_per_episode=1.0,
        min_duration_per_episode=0.01,
        reward_window=3,
    )

    def _make_results(values: list[float]) -> list[EpisodeResult]:
        return [
            EpisodeResult(
                index=i,
                reward=value,
                steps=1,
                duration=0.2,
                transitions=[],
            )
            for i, value in enumerate(values)
        ]

    baseline_stats = BatchStatistics(
        recent_results=_make_results([0.5, 0.55, 0.52]),
        completed_episodes=3,
        total_episodes=30,
        batch_duration=0.6,
        worker_count=2,
    )
    workers, reason = strategy.recommend_worker_count(2, 0, baseline_stats)
    assert workers == 2
    assert reason in {"stable", "fast-rollouts", "slow-rollouts"}

    improved_stats = BatchStatistics(
        recent_results=_make_results([0.75, 0.82, 0.79]),
        completed_episodes=6,
        total_episodes=30,
        batch_duration=0.6,
        worker_count=2,
    )
    boosted_workers, reason = strategy.recommend_worker_count(
        workers, 1, improved_stats
    )
    assert boosted_workers > workers
    assert reason == "reward-improved"

    regressed_stats = BatchStatistics(
        recent_results=_make_results([0.2, 0.65, 0.1]),
        completed_episodes=9,
        total_episodes=30,
        batch_duration=0.45,
        worker_count=4,
    )
    reduced_workers, reason = strategy.recommend_worker_count(
        boosted_workers, 2, regressed_stats
    )
    assert reduced_workers < boosted_workers
    assert reason == "reward-regressed"


def test_throughput_scaling_strategy_balances_throughput() -> None:
    base_strategy = AdaptiveScalingStrategy(
        min_workers=1,
        max_workers=6,
        increase_factor=1.2,
        decrease_factor=0.8,
        improvement_threshold=10.0,
        regression_tolerance=10.0,
        variance_threshold=1.0,
        max_duration_per_episode=10.0,
        min_duration_per_episode=0.0,
    )
    strategy = ThroughputAwareScalingStrategy(
        target_throughput=2.0,
        reward_strategy=base_strategy,
        throughput_window=2,
        adjustment_tolerance=0.05,
        cooldown_batches=0,
        minimum_success_rate=0.0,
    )

    def _stats(
        rewards: list[float], duration: float, worker_count: int, completed: int
    ) -> BatchStatistics:
        return BatchStatistics(
            recent_results=[
                EpisodeResult(
                    index=i,
                    reward=value,
                    steps=1,
                    duration=duration / max(1, len(rewards)),
                    transitions=[],
                )
                for i, value in enumerate(rewards)
            ],
            completed_episodes=completed,
            total_episodes=64,
            batch_duration=duration,
            worker_count=worker_count,
        )

    slow_stats = _stats([0.5, 0.55], duration=4.0, worker_count=2, completed=2)
    increased_workers, reason = strategy.recommend_worker_count(2, 0, slow_stats)
    assert increased_workers > 2
    assert reason == "throughput-low"

    fast_stats = _stats(
        [0.5 for _ in range(increased_workers)],
        duration=0.4,
        worker_count=increased_workers,
        completed=2 + increased_workers,
    )
    reduced_workers, reason = strategy.recommend_worker_count(
        increased_workers, 1, fast_stats
    )
    assert reduced_workers < increased_workers
    assert reason == "throughput-high"


def test_reinforcement_loop_invokes_batch_callback() -> None:
    rewards = [0.1, 0.5, 0.9]
    policy = EpsilonGreedyPolicy(action_count=len(rewards), epsilon=0.0, seed=42)
    observed: list[BatchStatistics] = []

    def _callback(stats: BatchStatistics) -> None:
        observed.append(stats)

    loop = ReinforcementLearningLoop(
        environment_factory=lambda: SimpleBanditEnvironment(rewards),
        policy=policy,
        max_steps=1,
        scaling_strategy=FixedScalingStrategy(),
        initial_workers=1,
        max_workers=1,
        batch_callback=_callback,
    )

    loop.run(total_episodes=5)

    assert observed
    assert all(isinstance(item, BatchStatistics) for item in observed)
    assert observed[0].worker_count == 1
    assert observed[0].episode_count >= 1


def test_reinforcement_loop_emits_tracing_and_metrics() -> None:
    rewards = [0.2, 0.8]
    policy = EpsilonGreedyPolicy(action_count=len(rewards), epsilon=0.1, seed=7)
    tracer = _RecordingTracer()
    metrics: list[tuple[str, dict[str, Any]]] = []

    def _metrics_sink(name: str, payload: dict[str, Any]) -> None:
        metrics.append((name, dict(payload)))

    loop = ReinforcementLearningLoop(
        environment_factory=lambda: SimpleBanditEnvironment(rewards),
        policy=policy,
        max_steps=1,
        scaling_strategy=FixedScalingStrategy(),
        initial_workers=1,
        max_workers=1,
        tracer_factory=lambda _: tracer,
        metrics_sink=_metrics_sink,
    )

    summary = loop.run(total_episodes=6)

    assert summary.total_episodes == 6
    assert tracer.spans and tracer.spans[0].name == "reinforcement.loop.run"
    recorded_events = [event for span in tracer.spans for event in span.events]
    assert any(name == "batch.completed" for name, _ in recorded_events)
    metric_names = [name for name, _ in metrics]
    assert "reinforcement.loop.batch" in metric_names
    assert metric_names[-1] == "reinforcement.loop.summary"


class _CuriosityStub:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def detect_gaps(self, conversation_context: dict) -> dict:
        self.calls.append(conversation_context)
        return {
            "status": "insufficient_knowledge",
            "additional_context": "cached knowledge base insight",
        }


class _MemoryItem:
    def __init__(self, response: str) -> None:
        self.response = response


class _HippocampusStub:
    def __init__(self) -> None:
        self.requests: list[tuple[str, int]] = []

    async def history(self, user_id: str, limit: int = 10) -> list[_MemoryItem]:
        self.requests.append((user_id, limit))
        return [_MemoryItem("prior contextual response")] if user_id else []


def test_preference_dataset_curator_builds_samples_with_context() -> None:
    curiosity = _CuriosityStub()
    hippocampus = _HippocampusStub()
    curator = PreferenceDatasetCurator(
        curiosity_engine=curiosity,
        hippocampus=hippocampus,
    )

    dataset = [
        {
            "metadata": {
                "user_id": "user-123",
                "query": "Summarise the monGARS roadmap",
            },
            "response": "The roadmap focuses on cognition and reinforcement.",
        }
    ]

    samples = curator.build(dataset, limit=5)

    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, PreferenceSample)
    assert "Context:" in sample.prompt
    assert sample.chosen == "The roadmap focuses on cognition and reinforcement."
    assert sample.rejected != sample.chosen
    assert sample.metadata["user_id"] == "user-123"
    assert curiosity.calls and curiosity.calls[0]["last_query"]
    assert hippocampus.requests == [("user-123", 1)]


class _SlotStub:
    def __init__(self, slot_name: str, **kwargs: Any) -> None:
        self.slot_name = slot_name
        self.kwargs = kwargs
        self.closed = False

    def __enter__(self) -> tuple[object, object]:
        _SlotStub.last_slot = self
        return object(), object()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.closed = True


class _TrainerStub:
    created: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _TrainerStub.created.append(kwargs)

    def train(self) -> SimpleNamespace:
        return SimpleNamespace(metrics={"preference_accuracy": 1.0})


def test_preference_alignment_loop_invokes_trainer_with_slot() -> None:
    loop = PreferenceAlignmentLoop(
        slot_manager_cls=_SlotStub,
        dpo_trainer_cls=_TrainerStub,
        dpo_config_cls=lambda **kwargs: SimpleNamespace(**kwargs),
        output_dir="unit-test",
    )

    dataset = [
        {
            "prompt": "Explain alignment.",
            "chosen": "Alignment ensures consistent assistant behaviour.",
            "rejected": "I cannot help with that.",
        }
    ]

    result = loop.reinforcement_loop(dataset)

    assert isinstance(result, SimpleNamespace)
    assert result.metrics["preference_accuracy"] == 1.0
    assert _TrainerStub.created
    trainer_kwargs = _TrainerStub.created[-1]
    assert trainer_kwargs["max_length"] == 1024
    assert trainer_kwargs["max_prompt_length"] == 512
    assert trainer_kwargs["train_dataset"][0]["prompt"] == "Explain alignment."
    assert _SlotStub.last_slot.kwargs["max_seq_length"] == 1024
    assert "model_id" in _SlotStub.last_slot.kwargs


def test_reasoning_reward_function_awards_bonus() -> None:
    loop = ReinforcementLoop(
        model_id="test-model",
        slot_manager_cls=_SlotStub,
        self_training_engine=SelfTrainingEngine(),
        trainer_cls=None,
        trainer_config_cls=None,
        fast_model_cls=None,
        torch_module=None,
    )
    dataset = [{"answer": "42"}]
    reward_fn = loop._build_reward_function(dataset)
    completion = (
        "<reasoning>" + " ".join(["step"] * 60) + "</reasoning><answer>42</answer>"
    )
    rewards = reward_fn(completions=[completion], completion_ids=[0])
    assert rewards == [1.5]


def test_reinforcement_reasoning_requires_operator_approval(
    tmp_path: Path, monkeypatch
) -> None:
    approvals = OperatorApprovalRegistry(tmp_path / "approvals.json")
    loop = ReinforcementLoop(
        model_id="test-model",
        slot_manager_cls=_SlotStub,
        self_training_engine=SelfTrainingEngine(),
        trainer_cls=None,
        trainer_config_cls=None,
        fast_model_cls=None,
        torch_module=None,
        approval_registry=approvals,
    )

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    manifest_calls: list[dict[str, Any]] = []

    def _fake_update_manifest(registry: Path, summary: dict[str, Any]):
        manifest_calls.append(summary)
        return SimpleNamespace(build_payload=lambda: summary)

    monkeypatch.setenv("USE_RAY_SERVE", "false")
    monkeypatch.setattr(
        "modules.neurons.training.reinforcement_loop.update_manifest",
        _fake_update_manifest,
    )

    evaluation = {"accuracy": 0.75, "evaluated": 10.0}
    loop._rollout_to_manifest(evaluation, 12, adapter_dir, None)

    pending = list(approvals.pending(source="reinforcement.reasoning"))
    assert pending and not manifest_calls

    approvals.approve(pending[0].request_id, operator="tester")

    loop._rollout_to_manifest(evaluation, 12, adapter_dir, None)

    assert manifest_calls


def test_train_reasoning_grpo_invokes_injected_dependencies(
    monkeypatch, tmp_path
) -> None:
    prompt = [
        {"role": "system", "content": SelfTrainingEngine.SYSTEM_PROMPT.strip()},
        {"role": "user", "content": "2 + 2"},
    ]
    dataset_entry = {"prompt": prompt, "answer": "4"}

    class StubSelfTraining(SelfTrainingEngine):
        def curate_reasoning_dataset(
            self, num_samples: int = 200, internal_ratio: float = 0.5
        ):
            return [dataset_entry], [dataset_entry]

    class DummyTokenizer:
        def apply_chat_template(self, *_: Any, **__: Any) -> str:
            return "prompt"

        def __call__(self, *_: Any, **__: Any) -> SimpleNamespace:
            return SimpleNamespace(to=lambda _device: None)

        def decode(self, *_: Any, **__: Any) -> str:
            return "<answer>4</answer>"

        def save_pretrained(self, directory: str) -> None:
            Path(directory).mkdir(parents=True, exist_ok=True)

    class DummyModel:
        device = "cpu"

        def generate(self, **_: Any) -> list[str]:
            return ["ignored"]

        def save_pretrained(self, directory: str) -> None:
            Path(directory).mkdir(parents=True, exist_ok=True)

    class DummySlotManager:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def __enter__(self) -> tuple[DummyModel, DummyTokenizer]:
            return DummyModel(), DummyTokenizer()

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    class DummyTrainer:
        def __init__(self, **_: Any) -> None:
            self.model = DummyModel()
            self.state = SimpleNamespace(global_step=9)

        def train(self) -> None:
            return None

    def fake_config(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)

    def fake_save(trainer: Any, tokenizer: Any) -> tuple[Path, Path | None]:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        return adapter_dir, None

    loop = ReinforcementLoop(
        model_id="test-model",
        slot_name="test",
        output_dir=tmp_path / "output",
        registry_path=tmp_path / "registry",
        self_training_engine=StubSelfTraining(),
        slot_manager_cls=DummySlotManager,
        trainer_cls=DummyTrainer,
        trainer_config_cls=fake_config,
        fast_model_cls=SimpleNamespace(get_peft_model=lambda model, **_: model),
        torch_module=SimpleNamespace(no_grad=lambda: contextlib.nullcontext()),
    )

    monkeypatch.setattr(
        loop,
        "_evaluate_reasoning",
        lambda model, dataset, tokenizer: {"accuracy": 1.0, "evaluated": 1.0},
    )
    monkeypatch.setattr(loop, "_save_artifacts", fake_save)
    monkeypatch.setattr(loop, "_rollout_to_manifest", lambda *args, **kwargs: None)

    summary = loop.train_reasoning_grpo(num_samples=1)

    assert isinstance(summary, ReasoningRunSummary)
    assert summary.accuracy == 1.0
    assert summary.steps == 9
    assert summary.adapter_dir is not None
