"""Utilities for lightweight reinforcement learning research loops.

This module keeps the training orchestration intentionally lightweight so it
can execute in constrained environments while still surfacing the key
telemetry that larger production runs depend on.  Heavy dependencies (Torch,
Ray, etc.) are intentionally avoided; instead, the classes defined here focus
on orchestrating experience collection, tracking worker utilisation, and
providing hooks for adaptive scaling strategies.

Typical usage::

    loop = ReinforcementLearningLoop(
        environment_factory=lambda: MyGymWrapper(...),
        policy=my_policy,
        max_steps=256,
        scaling_strategy=AdaptiveScalingStrategy(),
    )
    summary = loop.run(total_episodes=128)

``summary`` contains per-episode outcomes, worker-allocation history, and
aggregate reward metrics suitable for downstream manifest updates or
observability pipelines.
"""

from __future__ import annotations

import copy
import logging
import statistics
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Transition:
    """Single interaction with the environment."""

    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool
    info: dict[str, Any]


@dataclass(slots=True)
class EpisodeResult:
    """Outcome for a single training episode."""

    index: int
    reward: float
    steps: int
    duration: float
    transitions: list[Transition]
    metadata: dict[str, Any] = field(default_factory=dict)
    failed: bool = False
    error: str | None = None


@dataclass(slots=True)
class WorkerAdjustment:
    """History entry describing a worker-count decision."""

    batch_index: int
    worker_count: int
    reason: str


@dataclass(slots=True)
class BatchStatistics:
    """Snapshot describing the most recent batch of rollouts."""

    recent_results: Sequence[EpisodeResult]
    completed_episodes: int
    total_episodes: int
    batch_duration: float


@dataclass(slots=True)
class ReinforcementLearningSummary:
    """Aggregate metrics for a reinforcement-learning run."""

    total_episodes: int
    total_reward: float
    average_reward: float
    episode_results: list[EpisodeResult]
    failures: int
    wall_clock_seconds: float
    worker_history: list[WorkerAdjustment]

    def successful_episodes(self) -> Iterable[EpisodeResult]:
        """Return an iterator over the successful episode results."""

        return (result for result in self.episode_results if not result.failed)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the summary into a JSON-friendly payload."""

        return {
            "total_episodes": self.total_episodes,
            "total_reward": self.total_reward,
            "average_reward": self.average_reward,
            "failures": self.failures,
            "wall_clock_seconds": self.wall_clock_seconds,
            "worker_history": [
                {
                    "batch_index": item.batch_index,
                    "worker_count": item.worker_count,
                    "reason": item.reason,
                }
                for item in self.worker_history
            ],
            "episodes": [
                {
                    "index": episode.index,
                    "reward": episode.reward,
                    "steps": episode.steps,
                    "duration": episode.duration,
                    "failed": episode.failed,
                    "error": episode.error,
                    "metadata": episode.metadata,
                }
                for episode in self.episode_results
            ],
        }


class EnvironmentProtocol(Protocol):
    """Minimal contract expected of an environment."""

    def reset(self) -> Any:
        """Reset the environment and return the initial observation."""

    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        """Apply an action returning ``(state, reward, done, info)``."""


class PolicyProtocol(Protocol):
    """Contract that policy implementations must fulfil."""

    def select_action(self, state: Any) -> Any:
        """Choose the next action for the provided state."""

    def update(self, transitions: Sequence[Transition]) -> None:
        """Update the policy parameters using the collected trajectory."""

    def clone(self) -> "PolicyProtocol":  # pragma: no cover - optional
        """Return a copy of the policy suitable for isolated rollouts."""


class ScalingStrategy(Protocol):
    """Strategy interface for dynamic worker-allocation decisions."""

    def recommend_worker_count(
        self,
        current_workers: int,
        batch_index: int,
        stats: BatchStatistics,
    ) -> tuple[int, str]:
        """Return the recommended worker count and a human-readable reason."""


class AdaptiveScalingStrategy:
    """Reward-aware scaling policy for reinforcement-learning rollouts."""

    def __init__(
        self,
        *,
        min_workers: int = 1,
        max_workers: int = 8,
        increase_factor: float = 1.5,
        decrease_factor: float = 0.75,
        improvement_threshold: float = 0.15,
        regression_tolerance: float = 0.25,
        variance_threshold: float = 0.2,
        max_duration_per_episode: float = 1.0,
        min_duration_per_episode: float = 0.05,
        reward_window: int = 4,
    ) -> None:
        if min_workers < 1:
            raise ValueError("Minimum workers must be at least 1")
        if max_workers < min_workers:
            raise ValueError("Maximum workers must be >= minimum workers")
        if increase_factor <= 1.0:
            raise ValueError("Increase factor should be greater than 1.0")
        if not (0.0 < decrease_factor < 1.0):
            raise ValueError("Decrease factor must be between 0 and 1")

        self.min_workers = min_workers
        self.max_workers = max_workers
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.improvement_threshold = improvement_threshold
        self.regression_tolerance = regression_tolerance
        self.variance_threshold = variance_threshold
        self.max_duration_per_episode = max_duration_per_episode
        self.min_duration_per_episode = min_duration_per_episode
        self.reward_window = max(1, reward_window)
        self._reward_history: list[float] = []

    def recommend_worker_count(
        self,
        current_workers: int,
        batch_index: int,
        stats: BatchStatistics,
    ) -> tuple[int, str]:
        """Recommend a worker-count adjustment based on recent results."""

        if not stats.recent_results:
            return current_workers, "no-results"

        rewards = [
            result.reward for result in stats.recent_results if not result.failed
        ]
        if not rewards:
            # If the batch failed entirely, retreat towards the minimum.
            downgraded = max(
                self.min_workers, int(current_workers * self.decrease_factor)
            )
            return downgraded, "batch-failed"

        avg_reward = statistics.fmean(rewards)
        variance = statistics.pvariance(rewards) if len(rewards) > 1 else 0.0
        duration_per_episode = stats.batch_duration / max(1, len(rewards))

        window_mean = (
            statistics.fmean(self._reward_history)
            if self._reward_history
            else avg_reward
        )
        improvement = avg_reward - window_mean

        self._reward_history.append(avg_reward)
        if len(self._reward_history) > self.reward_window:
            self._reward_history.pop(0)

        recommendation = current_workers
        reason = "stable"

        if (
            improvement >= self.improvement_threshold
            and variance <= self.variance_threshold
        ):
            proposed = max(
                current_workers + 1, int(current_workers * self.increase_factor)
            )
            recommendation = min(self.max_workers, proposed)
            reason = "reward-improved"
        elif (
            improvement <= -self.regression_tolerance
            or variance >= self.variance_threshold * 2
        ):
            proposed = max(
                self.min_workers, int(current_workers * self.decrease_factor)
            )
            recommendation = proposed
            reason = "reward-regressed"

        if (
            duration_per_episode > self.max_duration_per_episode
            and recommendation < self.max_workers
        ):
            recommendation = min(self.max_workers, recommendation + 1)
            reason = "slow-rollouts"
        elif (
            duration_per_episode < self.min_duration_per_episode
            and recommendation > self.min_workers
        ):
            recommendation = max(self.min_workers, recommendation - 1)
            reason = "fast-rollouts"

        if recommendation != current_workers:
            logger.debug(
                "Scaling decision",  # pragma: no cover - log formatting
                extra={
                    "batch_index": batch_index,
                    "reason": reason,
                    "current_workers": current_workers,
                    "recommendation": recommendation,
                    "avg_reward": avg_reward,
                    "variance": variance,
                    "duration_per_episode": duration_per_episode,
                },
            )

        return recommendation, reason


class ReinforcementLearningLoop:
    """Coordinate reinforcement-learning rollouts with adaptive scaling."""

    def __init__(
        self,
        *,
        environment_factory: Callable[[], EnvironmentProtocol],
        policy: PolicyProtocol,
        max_steps: int,
        scaling_strategy: ScalingStrategy | None = None,
        initial_workers: int = 1,
        max_workers: int | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if initial_workers < 1:
            raise ValueError("initial_workers must be at least 1")

        self._environment_factory = environment_factory
        self._policy = policy
        self._max_steps = max_steps
        self._scaling_strategy = scaling_strategy or AdaptiveScalingStrategy(
            min_workers=1, max_workers=max_workers or max(initial_workers, 1)
        )
        self._initial_workers = initial_workers
        self._max_workers = max_workers or getattr(
            self._scaling_strategy, "max_workers", initial_workers
        )
        self._policy_lock = threading.Lock()

    def run(self, total_episodes: int) -> ReinforcementLearningSummary:
        """Execute the reinforcement-learning loop for the requested episodes."""

        if total_episodes < 1:
            raise ValueError("total_episodes must be at least 1")

        start_time = time.perf_counter()
        worker_count = min(self._initial_workers, self._max_workers)
        completed = 0
        all_results: list[EpisodeResult] = []
        worker_history: list[WorkerAdjustment] = [
            WorkerAdjustment(batch_index=0, worker_count=worker_count, reason="initial")
        ]
        batch_index = 0

        while completed < total_episodes:
            remaining = total_episodes - completed
            episodes_this_batch = min(remaining, worker_count)
            batch_start = time.perf_counter()
            results = self._execute_batch(
                batch_index, episodes_this_batch, worker_count
            )
            batch_duration = time.perf_counter() - batch_start

            for result in results:
                if not result.failed:
                    self._apply_policy_update(result.transitions)
            all_results.extend(results)
            completed += len(results)

            stats = BatchStatistics(
                recent_results=results,
                completed_episodes=completed,
                total_episodes=total_episodes,
                batch_duration=batch_duration,
            )

            next_workers, reason = self._scaling_strategy.recommend_worker_count(
                worker_count, batch_index + 1, stats
            )
            next_workers = max(1, min(self._max_workers, next_workers))
            if next_workers != worker_count:
                worker_history.append(
                    WorkerAdjustment(
                        batch_index=batch_index + 1,
                        worker_count=next_workers,
                        reason=reason,
                    )
                )
                worker_count = next_workers

            batch_index += 1

        duration = time.perf_counter() - start_time
        total_reward = sum(result.reward for result in all_results if not result.failed)
        success_count = sum(1 for result in all_results if not result.failed)
        average_reward = total_reward / success_count if success_count else 0.0
        failures = sum(1 for result in all_results if result.failed)

        logger.info(
            "RL loop complete",
            extra={
                "episodes": total_episodes,
                "failures": failures,
                "total_reward": round(total_reward, 4),
                "average_reward": round(average_reward, 4),
            },
        )

        return ReinforcementLearningSummary(
            total_episodes=total_episodes,
            total_reward=total_reward,
            average_reward=average_reward,
            episode_results=all_results,
            failures=failures,
            wall_clock_seconds=duration,
            worker_history=worker_history,
        )

    def _apply_policy_update(self, transitions: Sequence[Transition]) -> None:
        if not transitions:
            return
        with self._policy_lock:
            try:
                self._policy.update(transitions)
            except Exception:  # pragma: no cover - defensive logging
                logger.exception(
                    "Policy update failed", extra={"transitions": len(transitions)}
                )

    def _execute_batch(
        self, batch_index: int, episodes: int, worker_count: int
    ) -> list[EpisodeResult]:
        results: list[EpisodeResult] = []
        if episodes <= 0:
            return results

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures: list[Future[EpisodeResult]] = [
                executor.submit(self._run_episode, batch_index, idx)
                for idx in range(episodes)
            ]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:  # pragma: no cover - unexpected worker error
                    logger.exception("Episode execution failed", exc_info=True)
                    results.append(
                        EpisodeResult(
                            index=-1,
                            reward=0.0,
                            steps=0,
                            duration=0.0,
                            transitions=[],
                            failed=True,
                            error=str(exc),
                        )
                    )
        return results

    def _run_episode(self, batch_index: int, offset: int) -> EpisodeResult:
        environment = self._environment_factory()
        policy = self._clone_policy()
        transitions: list[Transition] = []
        total_reward = 0.0
        steps = 0
        index = batch_index * max(1, self._max_workers) + offset
        episode_start = time.perf_counter()

        try:
            state = environment.reset()
            for steps in range(1, self._max_steps + 1):
                action = policy.select_action(state)
                next_state, reward, done, info = environment.step(action)
                reward_value = float(reward)
                transitions.append(
                    Transition(
                        state=state,
                        action=action,
                        reward=reward_value,
                        next_state=next_state,
                        done=bool(done),
                        info=dict(info) if isinstance(info, dict) else {},
                    )
                )
                total_reward += reward_value
                state = next_state
                if done:
                    break
            duration = time.perf_counter() - episode_start
            metadata = {"batch_index": batch_index, "offset": offset}
            return EpisodeResult(
                index=index,
                reward=total_reward,
                steps=steps,
                duration=duration,
                transitions=transitions,
                metadata=metadata,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            duration = time.perf_counter() - episode_start
            logger.exception(
                "Episode crashed",
                extra={"batch_index": batch_index, "offset": offset},
            )
            return EpisodeResult(
                index=index,
                reward=0.0,
                steps=steps,
                duration=duration,
                transitions=transitions,
                metadata={"batch_index": batch_index, "offset": offset},
                failed=True,
                error=str(exc),
            )

    def _clone_policy(self) -> PolicyProtocol:
        try:
            return self._policy.clone()
        except AttributeError:
            pass
        except Exception:  # pragma: no cover - best-effort fallback
            logger.exception("Policy clone() failed; falling back to deepcopy")

        try:
            return copy.deepcopy(self._policy)
        except Exception:  # pragma: no cover - final fallback
            logger.warning(
                "Falling back to shared policy instance; concurrency may degrade"
            )
            return self._policy
