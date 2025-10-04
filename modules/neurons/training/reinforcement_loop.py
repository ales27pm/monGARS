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

import asyncio
import copy
import logging
import statistics
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

try:  # pragma: no cover - optional dependency at runtime
    from trl import DPOConfig, DPOTrainer  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - trl is optional in tests
    DPOConfig = None  # type: ignore[assignment]
    DPOTrainer = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency at runtime
    from datasets import Dataset  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - datasets optional
    Dataset = None  # type: ignore[assignment]

from monGARS.core.model_slot_manager import ModelSlotManager

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
    worker_count: int

    @property
    def episode_count(self) -> int:
        """Number of episode results contained in the batch."""

        return len(self.recent_results)

    @property
    def failure_count(self) -> int:
        """Count of failed episodes in the batch."""

        return sum(1 for result in self.recent_results if result.failed)

    @property
    def success_count(self) -> int:
        """Count of successful episodes in the batch."""

        return self.episode_count - self.failure_count

    @property
    def success_rate(self) -> float:
        """Success ratio for the batch."""

        if self.episode_count == 0:
            return 0.0
        return self.success_count / self.episode_count

    @property
    def rewards(self) -> list[float]:
        """Return the reward values for successful episodes."""

        return [result.reward for result in self.recent_results if not result.failed]

    @property
    def average_reward(self) -> float | None:
        """Average reward for successful episodes, if present."""

        rewards = self.rewards
        if not rewards:
            return None
        return statistics.fmean(rewards)

    @property
    def reward_variance(self) -> float | None:
        """Population variance of rewards for successful episodes."""

        rewards = self.rewards
        if len(rewards) <= 1:
            return 0.0 if rewards else None
        return statistics.pvariance(rewards)

    @property
    def average_duration(self) -> float:
        """Average duration of each episode in the batch."""

        return self.batch_duration / max(1, self.episode_count)

    @property
    def throughput(self) -> float | None:
        """Episodes executed per second for the batch."""

        if self.batch_duration <= 0:
            return None
        return self.episode_count / self.batch_duration


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


@dataclass(slots=True)
class PreferenceSample:
    """Single prompt-preference pair for DPO alignment."""

    prompt: str
    chosen: str
    rejected: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Return a mapping compatible with :class:`~trl.DPOTrainer`."""

        record = {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }
        if self.metadata:
            record["metadata"] = dict(self.metadata)
        return record

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

        avg_reward = stats.average_reward
        if avg_reward is None:
            # If the batch failed entirely, retreat towards the minimum.
            downgraded = max(
                self.min_workers, int(current_workers * self.decrease_factor)
            )
            return downgraded, "batch-failed"

        variance = stats.reward_variance or 0.0
        duration_per_episode = stats.average_duration

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


class ThroughputAwareScalingStrategy:
    """Compose reward-aware scaling with throughput monitoring."""

    def __init__(
        self,
        *,
        target_throughput: float,
        reward_strategy: ScalingStrategy | None = None,
        throughput_window: int = 5,
        adjustment_tolerance: float = 0.1,
        cooldown_batches: int = 1,
        minimum_success_rate: float = 0.6,
    ) -> None:
        if target_throughput <= 0:
            raise ValueError("target_throughput must be positive")
        if throughput_window < 1:
            raise ValueError("throughput_window must be at least 1")
        if adjustment_tolerance < 0:
            raise ValueError("adjustment_tolerance must be >= 0")
        if cooldown_batches < 0:
            raise ValueError("cooldown_batches must be >= 0")
        if not (0.0 <= minimum_success_rate <= 1.0):
            raise ValueError("minimum_success_rate must be between 0 and 1")

        self._reward_strategy = reward_strategy or AdaptiveScalingStrategy()
        self._target_throughput = target_throughput
        self._throughput_history: deque[float] = deque(maxlen=throughput_window)
        self._adjustment_tolerance = adjustment_tolerance
        self._cooldown_batches = cooldown_batches
        self._batches_since_change = cooldown_batches
        self._minimum_success_rate = minimum_success_rate
        self.max_workers = getattr(self._reward_strategy, "max_workers", 8)
        self.min_workers = getattr(self._reward_strategy, "min_workers", 1)

    def recommend_worker_count(
        self,
        current_workers: int,
        batch_index: int,
        stats: BatchStatistics,
    ) -> tuple[int, str]:
        base_workers, base_reason = self._reward_strategy.recommend_worker_count(
            current_workers, batch_index, stats
        )

        if base_workers != current_workers:
            self._throughput_history.clear()
            self._batches_since_change = 0
            return base_workers, base_reason

        throughput = stats.throughput
        if throughput is None:
            return base_workers, base_reason

        self._throughput_history.append(throughput)
        self._batches_since_change += 1

        if not self._throughput_history:
            return base_workers, base_reason

        if self._batches_since_change <= self._cooldown_batches:
            return base_workers, base_reason

        avg_throughput = statistics.fmean(self._throughput_history)
        if avg_throughput <= 0:
            return base_workers, base_reason

        gap = (avg_throughput - self._target_throughput) / self._target_throughput
        recommendation = base_workers
        reason = base_reason

        min_workers = getattr(self._reward_strategy, "min_workers", self.min_workers)
        max_workers = getattr(self._reward_strategy, "max_workers", self.max_workers)

        if gap < -self._adjustment_tolerance:
            if stats.success_rate >= self._minimum_success_rate:
                desired = min(max_workers, base_workers + 1)
                if desired != base_workers:
                    recommendation = desired
                    reason = "throughput-low"
        elif gap > self._adjustment_tolerance:
            desired = max(min_workers, base_workers - 1)
            if desired != base_workers:
                recommendation = desired
                reason = "throughput-high"

        if recommendation != base_workers:
            self._batches_since_change = 0
            self._throughput_history.clear()
            return recommendation, reason

        return base_workers, base_reason


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
        batch_callback: Callable[[BatchStatistics], None] | None = None,
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
        self._batch_callback = batch_callback
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
            active_workers = worker_count
            episodes_this_batch = min(remaining, active_workers)
            batch_start = time.perf_counter()
            results = self._execute_batch(
                batch_index, episodes_this_batch, active_workers
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
                worker_count=active_workers,
            )

            if self._batch_callback is not None:
                try:
                    self._batch_callback(stats)
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception(
                        "Batch callback failed",
                        extra={
                            "batch_index": batch_index,
                            "worker_count": active_workers,
                        },
                    )

            next_workers, reason = self._scaling_strategy.recommend_worker_count(
                active_workers, batch_index + 1, stats
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
            episode_idx = 0
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:  # pragma: no cover - unexpected worker error
                    logger.exception("Episode execution failed", exc_info=True)
                    failed_index = batch_index * max(1, self._max_workers) + episode_idx
                    results.append(
                        EpisodeResult(
                            index=failed_index,
                            reward=0.0,
                            steps=0,
                            duration=0.0,
                            transitions=[],
                            failed=True,
                            error=str(exc),
                        )
                    )
                episode_idx += 1
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
            for step in range(1, self._max_steps + 1):
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
                steps = step
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
        finally:
            close = getattr(environment, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover - best-effort cleanup
                    logger.debug("Environment close() failed", exc_info=True)

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


class PreferenceDatasetCurator:
    """Build DPO preference datasets from curated memory and curiosity signals."""

    def __init__(
        self,
        *,
        curiosity_engine: Any | None = None,
        hippocampus: Any | None = None,
    ) -> None:
        self._curiosity = curiosity_engine
        self._hippocampus = hippocampus

    async def build_async(
        self,
        dataset: Iterable[Any] | Dataset | None,
        *,
        limit: int | None = None,
    ) -> list[PreferenceSample]:
        """Asynchronously transform ``dataset`` into preference samples."""

        if dataset is None:
            return []

        samples: list[PreferenceSample] = []
        for idx, record in enumerate(self._iter_dataset(dataset)):
            if limit is not None and idx >= limit:
                break
            sample = await self._record_to_sample(record)
            if sample is None:
                continue
            samples.append(sample)
        return samples

    def build(
        self,
        dataset: Iterable[Any] | Dataset | None,
        *,
        limit: int | None = None,
    ) -> list[PreferenceSample]:
        """Synchronously construct preference samples from ``dataset``.

        ``CuriosityEngine`` and ``Hippocampus`` integrations are optional; when
        provided they enrich prompts with recent context to stabilise DPO
        alignment.
        """

        async def _inner() -> list[PreferenceSample]:
            return await self.build_async(dataset, limit=limit)

        try:
            return asyncio.run(_inner())
        except RuntimeError as exc:  # pragma: no cover - running loop guard
            if "asyncio.run() cannot be called" in str(exc):
                raise RuntimeError(
                    "PreferenceDatasetCurator.build() cannot run inside an active "
                    "event loop; use build_async() instead."
                ) from exc
            raise

    def _iter_dataset(self, dataset: Iterable[Any] | Dataset) -> Iterable[Any]:
        if isinstance(dataset, PreferenceSample):
            return [dataset]
        if Dataset is not None and isinstance(dataset, Dataset):  # pragma: no cover
            return dataset  # type: ignore[return-value]
        if isinstance(dataset, Iterable):
            return dataset
        return [dataset]

    async def _record_to_sample(self, record: Any) -> PreferenceSample | None:
        if isinstance(record, PreferenceSample):
            return record
        if not isinstance(record, Mapping):
            return None

        metadata = self._extract_metadata(record)
        prompt = self._extract_prompt(record, metadata)
        chosen = self._extract_response(record, metadata)
        if not prompt or not chosen:
            return None

        contexts = await self._gather_context(prompt, metadata)
        prompt_with_context = self._apply_context(prompt, contexts)
        rejected = self._select_rejected(record, metadata, chosen)

        return PreferenceSample(
            prompt=prompt_with_context,
            chosen=chosen,
            rejected=rejected,
            metadata=metadata,
        )

    def _extract_metadata(self, record: Mapping[str, Any]) -> dict[str, Any]:
        metadata = record.get("metadata")
        if isinstance(metadata, Mapping):
            return {str(key): value for key, value in metadata.items()}

        skip_keys = {
            "prompt",
            "chosen",
            "rejected",
            "text",
            "response",
            "answer",
            "completion",
            "output",
        }
        collected = {
            str(key): value for key, value in record.items() if key not in skip_keys
        }
        return collected

    def _extract_prompt(
        self, record: Mapping[str, Any], metadata: Mapping[str, Any]
    ) -> str | None:
        prompt = self._first_string(
            record,
            ("prompt", "query", "input", "question"),
        )
        if prompt:
            return prompt

        prompt = self._first_string(
            metadata,
            ("prompt", "query", "input", "question", "user_query"),
        )
        if prompt:
            return prompt

        text_value = record.get("text")
        if isinstance(text_value, str) and text_value.strip():
            return text_value.strip()

        return None

    def _extract_response(
        self, record: Mapping[str, Any], metadata: Mapping[str, Any]
    ) -> str | None:
        chosen = self._first_string(
            record,
            ("chosen", "answer", "response", "output", "completion", "text"),
        )
        if chosen:
            return chosen

        chosen = self._first_string(
            metadata,
            ("chosen", "answer", "response", "output", "completion", "text"),
        )
        return chosen

    def _select_rejected(
        self,
        record: Mapping[str, Any],
        metadata: Mapping[str, Any],
        chosen: str,
    ) -> str:
        rejected = self._first_string(
            record,
            ("rejected", "baseline_response", "negative", "worst"),
        )
        if not rejected:
            rejected = self._first_string(
                metadata,
                ("rejected", "baseline_response", "negative", "fallback"),
            )
        if not rejected or rejected.strip() == chosen.strip():
            topic = self._first_string(
                metadata, ("topic", "subject", "intent", "query")
            )
            if topic:
                rejected = f"I do not have enough trusted information about {topic}."
            else:
                rejected = (
                    "I'm unsure about that request right now, but I'll keep learning."
                )
        return rejected

    async def _gather_context(
        self, prompt: str, metadata: Mapping[str, Any]
    ) -> list[str]:
        contexts: list[str] = []

        meta_context = self._first_string(
            metadata, ("additional_context", "context", "background")
        )
        if meta_context:
            contexts.append(meta_context)

        if self._curiosity is not None:
            try:
                gap = await self._curiosity.detect_gaps(
                    {
                        "last_query": prompt,
                        "history": metadata.get("history"),
                    }
                )
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("reinforcement.curator.curiosity_failed")
            else:
                additional = (
                    gap.get("additional_context") if isinstance(gap, Mapping) else None
                )
                if isinstance(additional, str) and additional.strip():
                    contexts.append(additional.strip())

        if self._hippocampus is not None:
            user_id = metadata.get("user_id")
            if isinstance(user_id, str) and user_id:
                try:
                    history_items = await self._hippocampus.history(user_id, limit=1)
                except Exception:  # pragma: no cover - hippocampus optional
                    logger.exception("reinforcement.curator.hippocampus_failed")
                else:
                    for item in history_items:
                        response = getattr(item, "response", None)
                        if isinstance(response, str) and response.strip():
                            contexts.append(response.strip())
                            break

        return contexts

    def _apply_context(self, prompt: str, contexts: list[str]) -> str:
        if not contexts:
            return prompt
        unique: list[str] = []
        seen: set[str] = set()
        for context in contexts:
            cleaned = context.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            unique.append(cleaned)
        if not unique:
            return prompt
        context_block = "\n\n".join(unique)
        return f"{prompt}\n\nContext:\n{context_block}"

    @staticmethod
    def _first_string(
        source: Mapping[str, Any] | None, keys: Sequence[str]
    ) -> str | None:
        if source is None:
            return None
        for key in keys:
            value = source.get(key)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    return stripped
        return None


class PreferenceAlignmentLoop:
    """Execute a DPO-based reinforcement loop using Unsloth-backed models."""

    def __init__(
        self,
        *,
        slot_manager_cls: type[ModelSlotManager] | None = None,
        dpo_trainer_cls: type[Any] | None = None,
        dpo_config_cls: type[Any] | None = None,
        slot_name: str = "rl_slot",
        output_dir: str = "dpo_outputs",
        beta: float = 0.1,
        max_length: int = 1024,
        max_prompt_length: int = 512,
    ) -> None:
        self._slot_name = slot_name
        self._slot_manager_cls = slot_manager_cls or ModelSlotManager
        self._dpo_trainer_cls = dpo_trainer_cls or DPOTrainer
        self._dpo_config_cls = dpo_config_cls or DPOConfig
        self._output_dir = output_dir
        self._beta = float(beta)
        self._max_length = max(1, int(max_length))
        self._max_prompt_length = max(1, int(max_prompt_length))
        self._last_training_metrics: dict[str, Any] | None = None

    def reinforcement_loop(
        self, dataset: Sequence[PreferenceSample | Mapping[str, Any]] | Dataset
    ) -> Any:
        """Fine-tune the active model on ``dataset`` using DPO."""

        if self._slot_manager_cls is None:
            raise RuntimeError(
                "ModelSlotManager is unavailable for reinforcement training"
            )

        if self._dpo_trainer_cls is None or self._dpo_config_cls is None:
            raise RuntimeError(
                "trl.DPOTrainer is unavailable. Install the 'trl' package to enable reinforcement alignment."
            )

        prepared_dataset = self._prepare_dataset(dataset)
        if prepared_dataset is None:
            logger.info("reinforcement.alignment.dataset_empty")
            return None

        with self._slot_manager_cls(
            self._slot_name, max_seq_length=self._max_length
        ) as slot:
            model, tokenizer = slot
            if model is None or tokenizer is None:
                raise RuntimeError("ModelSlotManager returned an empty slot")

            config = self._dpo_config_cls(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_ratio=0.1,
                num_train_epochs=1,
                optim="adamw_8bit",
                beta=self._beta,
                output_dir=self._output_dir,
                fp16=False,
                bf16=False,
                remove_unused_columns=False,
                report_to=[],
                logging_steps=1,
                save_strategy="no",
            )

            trainer = self._dpo_trainer_cls(
                model=model,
                ref_model=None,
                train_dataset=prepared_dataset,
                tokenizer=tokenizer,
                args=config,
                beta=self._beta,
                max_length=self._max_length,
                max_prompt_length=self._max_prompt_length,
            )

            train_output = trainer.train()
            metrics = getattr(train_output, "metrics", None)
            if isinstance(metrics, Mapping):
                self._last_training_metrics = dict(metrics)
            else:
                self._last_training_metrics = {"status": "completed"}
            logger.info(
                "reinforcement.alignment.completed",
                extra={
                    "samples": len(prepared_dataset),
                    "max_length": self._max_length,
                    "max_prompt_length": self._max_prompt_length,
                },
            )
            return train_output

    def _prepare_dataset(
        self, dataset: Sequence[PreferenceSample | Mapping[str, Any]] | Dataset
    ) -> Any:
        if Dataset is not None and isinstance(dataset, Dataset):  # pragma: no cover
            if len(dataset) == 0:
                return None
            return dataset

        records: list[dict[str, Any]] = []
        for item in dataset:
            if isinstance(item, PreferenceSample):
                records.append(item.to_record())
            elif isinstance(item, Mapping):
                prompt = item.get("prompt")
                chosen = item.get("chosen")
                rejected = item.get("rejected")
                if not all(
                    isinstance(value, str) and value.strip()
                    for value in (prompt, chosen, rejected)
                ):
                    continue
                record: dict[str, Any] = {
                    "prompt": str(prompt),
                    "chosen": str(chosen),
                    "rejected": str(rejected),
                }
                metadata = item.get("metadata")
                if isinstance(metadata, Mapping):
                    record["metadata"] = dict(metadata)
                records.append(record)

        if not records:
            return None

        if Dataset is not None:  # pragma: no cover - optional dependency branch
            try:
                return Dataset.from_list(records)
            except Exception:
                logger.exception("reinforcement.alignment.dataset_conversion_failed")

        return records


__all__ = [
    "AdaptiveScalingStrategy",
    "BatchStatistics",
    "PreferenceAlignmentLoop",
    "PreferenceDatasetCurator",
    "PreferenceSample",
    "ReinforcementLearningLoop",
    "ReinforcementLearningSummary",
    "ThroughputAwareScalingStrategy",
    "Transition",
]
