from __future__ import annotations

import random
from dataclasses import dataclass

from modules.neurons.training.reinforcement_loop import (
    AdaptiveScalingStrategy,
    BatchStatistics,
    EpisodeResult,
    ReinforcementLearningLoop,
    ThroughputAwareScalingStrategy,
    Transition,
)


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
