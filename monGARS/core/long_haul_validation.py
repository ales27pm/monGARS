from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, MutableMapping

from modules.evolution_engine.energy import EnergyUsageReport
from modules.neurons.training.reinforcement_loop import (
    ReinforcementLearningLoop,
    ReinforcementLearningSummary,
)
from monGARS.config import get_settings
from monGARS.core.monitor import get_tracer
from monGARS.core.operator_approvals import OperatorApprovalRegistry

logger = logging.getLogger(__name__)


EnergyTrackerFactory = Callable[[], Any]
MetricsSink = Callable[[str, MutableMapping[str, float | int]], None]
ReinforcementLoopFactory = Callable[[], ReinforcementLearningLoop]
MNTPCallback = Callable[[], Any]


@dataclass(slots=True)
class LongHaulCycleReport:
    """Snapshot of a single long-haul validation cycle."""

    index: int
    status: str
    episodes: int
    total_reward: float
    average_reward: float
    failures: int
    duration_seconds: float
    energy_wh: float | None
    approval_pending: int | None
    incidents: tuple[str, ...] = field(default_factory=tuple)
    mnpt_executed: bool = False


@dataclass(slots=True)
class LongHaulValidationSummary:
    """Aggregate view of the research long-haul validation run."""

    started_at: str
    duration_seconds: float
    total_cycles: int
    total_episodes: int
    total_reward: float
    average_reward: float
    total_failures: int
    success_rate: float
    energy_wh: float | None
    approval_pending_final: int | None
    mnpt_runs: int
    cycles: list[LongHaulCycleReport]
    incidents: tuple[str, ...]


class ResearchLoopLongHaulValidator:
    """Execute sustained research loop validation with telemetry correlation."""

    def __init__(
        self,
        *,
        reinforcement_loop_factory: ReinforcementLoopFactory,
        approval_registry: OperatorApprovalRegistry | None = None,
        energy_tracker_factory: EnergyTrackerFactory | None = None,
        metrics_sink: MetricsSink | None = None,
        tracer_factory: Callable[[str], Any] | None = get_tracer,
        mnpt_callback: MNTPCallback | None = None,
        approval_source: str | None = None,
    ) -> None:
        if reinforcement_loop_factory is None:
            raise ValueError("reinforcement_loop_factory is required")

        settings = get_settings()
        self._reinforcement_loop_factory = reinforcement_loop_factory
        self._approval_registry = approval_registry
        self._energy_tracker_factory = energy_tracker_factory
        self._metrics_sink = metrics_sink
        self._tracer_factory = tracer_factory
        self._mnpt_callback = mnpt_callback
        self._default_cycles = max(1, int(settings.research_long_haul_cycles))
        self._default_episodes = max(
            1, int(settings.research_long_haul_episodes_per_cycle)
        )
        self._default_cooldown = max(
            0.0, float(settings.research_long_haul_cooldown_seconds)
        )
        self._approval_source = (
            approval_source
            if approval_source is not None
            else settings.research_long_haul_approval_source
        )

    async def execute(
        self,
        *,
        cycles: int | None = None,
        episodes_per_cycle: int | None = None,
        cooldown_seconds: float | None = None,
    ) -> LongHaulValidationSummary:
        """Run long-haul validation and return aggregated metrics."""

        cycle_count = cycles if cycles is not None else self._default_cycles
        episode_target = (
            episodes_per_cycle
            if episodes_per_cycle is not None
            else self._default_episodes
        )
        cooldown = (
            cooldown_seconds if cooldown_seconds is not None else self._default_cooldown
        )

        if cycle_count <= 0:
            raise ValueError("cycles must be greater than zero")
        if episode_target <= 0:
            raise ValueError("episodes_per_cycle must be greater than zero")
        if cooldown < 0:
            raise ValueError("cooldown_seconds must not be negative")

        logger.info(
            "research.longhaul.start",
            extra={
                "cycles": cycle_count,
                "episodes_per_cycle": episode_target,
                "cooldown_seconds": cooldown,
            },
        )

        started_at = datetime.now(timezone.utc)
        start_monotonic = time.perf_counter()
        tracer = (
            self._tracer_factory("llm.research.longhaul")
            if self._tracer_factory
            else None
        )

        span_attributes: dict[str, Any] = {
            "longhaul.cycles.target": int(cycle_count),
            "longhaul.episodes.per_cycle": int(episode_target),
        }

        incidents: list[str] = []
        cycle_reports: list[LongHaulCycleReport] = []
        total_reward = 0.0
        total_episodes = 0
        total_failures = 0
        energy_total = 0.0
        energy_observed = False
        mnpt_runs = 0

        with self._start_span(
            tracer, "research.longhaul.execute", span_attributes
        ) as span:
            for index in range(cycle_count):
                cycle_report, energy_captured, mnpt_success = await self._run_cycle(
                    index=index,
                    episodes=episode_target,
                    span=span,
                )
                cycle_reports.append(cycle_report)
                mnpt_runs += 1 if mnpt_success else 0
                if cycle_report.status != "completed":
                    incidents.extend(cycle_report.incidents)
                total_reward += cycle_report.total_reward
                total_episodes += cycle_report.episodes
                total_failures += cycle_report.failures
                if energy_captured is not None:
                    energy_total += energy_captured
                    energy_observed = True
                if cooldown and index < cycle_count - 1:
                    await asyncio.sleep(cooldown)

        success_count = max(0, total_episodes - total_failures)
        average_reward = total_reward / success_count if success_count else 0.0
        success_rate = success_count / total_episodes if total_episodes > 0 else 0.0
        duration_seconds = time.perf_counter() - start_monotonic
        approval_pending_final = self._count_pending_approvals()

        summary = LongHaulValidationSummary(
            started_at=started_at.isoformat(),
            duration_seconds=duration_seconds,
            total_cycles=len(cycle_reports),
            total_episodes=total_episodes,
            total_reward=total_reward,
            average_reward=average_reward,
            total_failures=total_failures,
            success_rate=success_rate,
            energy_wh=energy_total if energy_observed else None,
            approval_pending_final=approval_pending_final,
            mnpt_runs=mnpt_runs,
            cycles=cycle_reports,
            incidents=tuple(incidents),
        )

        summary_metrics: MutableMapping[str, float | int] = {
            "cycles": summary.total_cycles,
            "episodes": summary.total_episodes,
            "failures": summary.total_failures,
            "success_rate": summary.success_rate,
            "average_reward": summary.average_reward,
            "duration_seconds": summary.duration_seconds,
        }
        if summary.energy_wh is not None:
            summary_metrics["energy_wh"] = summary.energy_wh
        if summary.approval_pending_final is not None:
            summary_metrics["approvals_pending"] = summary.approval_pending_final

        self._record_metrics("research.longhaul.summary", summary_metrics)

        logger.info(
            "research.longhaul.complete",
            extra={
                "cycles": summary.total_cycles,
                "episodes": summary.total_episodes,
                "failures": summary.total_failures,
                "success_rate": round(summary.success_rate, 4),
                "energy_wh": summary.energy_wh,
                "duration_seconds": round(summary.duration_seconds, 3),
            },
        )

        return summary

    async def _run_cycle(
        self,
        *,
        index: int,
        episodes: int,
        span: Any,
    ) -> tuple[LongHaulCycleReport, float | None, bool]:
        start = time.perf_counter()
        incidents: list[str] = []
        energy_wh: float | None = None
        approvals: int | None = None
        status = "completed"
        summary: ReinforcementLearningSummary | None = None
        mnpt_executed = False

        try:
            loop = self._reinforcement_loop_factory()
        except Exception as exc:  # pragma: no cover - factory errors are unexpected
            status = "failed"
            message = f"reinforcement-factory-error: {exc!r}"
            incidents.append(message)
            logger.exception("research.longhaul.factory_failed", exc_info=True)
            loop = None

        tracker = (
            self._energy_tracker_factory() if self._energy_tracker_factory else None
        )
        tracker_started = False
        tracker_context = None
        if tracker is not None and callable(getattr(tracker, "track", None)):
            tracker_context = tracker.track()
        else:
            tracker_context = contextlib.nullcontext()
            if tracker is not None and hasattr(tracker, "start"):
                try:
                    tracker.start()
                    tracker_started = True
                except Exception as exc:  # pragma: no cover - defensive logging
                    incidents.append(f"energy-start-error: {exc!r}")
                    logger.exception(
                        "research.longhaul.energy_start_failed", exc_info=True
                    )
                    tracker_started = False

        with tracker_context:
            try:
                mnpt_executed = await self._invoke_mnpt_callback()
            except Exception as exc:  # pragma: no cover - unexpected callback failure
                status = "failed"
                incidents.append(f"mnpt-callback-error: {exc!r}")
                logger.exception("research.longhaul.mnpt_failed", exc_info=True)
            if loop is not None:
                try:
                    summary = await self._execute_loop(loop, episodes)
                except Exception as exc:  # pragma: no cover - defensive logging
                    status = "failed"
                    incidents.append(f"reinforcement-run-error: {exc!r}")
                    logger.exception("research.longhaul.run_failed", exc_info=True)

        if tracker is not None:
            try:
                if tracker_started and hasattr(tracker, "stop"):
                    report = tracker.stop()
                else:
                    report = getattr(tracker, "last_report", None)
            except Exception as exc:  # pragma: no cover - defensive logging
                report = None
                incidents.append(f"energy-stop-error: {exc!r}")
                logger.exception("research.longhaul.energy_stop_failed", exc_info=True)
            if isinstance(report, EnergyUsageReport):
                energy_wh = float(report.energy_wh)
            elif isinstance(report, dict) and "energy_wh" in report:
                try:
                    energy_wh = float(report["energy_wh"])
                except (TypeError, ValueError):
                    energy_wh = None

        if summary is not None:
            approvals = self._count_pending_approvals()
            episodes_completed = int(summary.total_episodes)
            total_reward = float(summary.total_reward)
            average_reward = float(summary.average_reward)
            failures = int(summary.failures)
        else:
            approvals = self._count_pending_approvals()
            episodes_completed = 0
            total_reward = 0.0
            average_reward = 0.0
            failures = 0

        duration = time.perf_counter() - start

        cycle_metrics: MutableMapping[str, float | int] = {
            "cycle": index,
            "episodes": episodes_completed,
            "failures": failures,
            "average_reward": average_reward,
            "duration_seconds": duration,
            "status": 1 if status == "completed" else 0,
        }
        if energy_wh is not None:
            cycle_metrics["energy_wh"] = energy_wh
        if approvals is not None:
            cycle_metrics["approvals_pending"] = approvals
        self._record_metrics("research.longhaul.cycle", cycle_metrics)

        self._emit_event(
            span,
            "cycle.completed",
            {
                "cycle": index,
                "status": status,
                "episodes": episodes_completed,
                "failures": failures,
                "energy_wh": energy_wh or 0.0,
                "mnpt_executed": int(mnpt_executed),
            },
        )

        return (
            LongHaulCycleReport(
                index=index,
                status=status,
                episodes=episodes_completed,
                total_reward=total_reward,
                average_reward=average_reward,
                failures=failures,
                duration_seconds=duration,
                energy_wh=energy_wh,
                approval_pending=approvals,
                incidents=tuple(incidents),
                mnpt_executed=mnpt_executed,
            ),
            energy_wh,
            mnpt_executed,
        )

    async def _execute_loop(
        self, loop: ReinforcementLearningLoop, episodes: int
    ) -> ReinforcementLearningSummary:
        run_method = getattr(loop, "run")
        if inspect.iscoroutinefunction(run_method):
            return await run_method(episodes)
        return await asyncio.to_thread(run_method, episodes)

    async def _invoke_mnpt_callback(self) -> bool:
        if self._mnpt_callback is None:
            return False
        result = self._mnpt_callback()
        if inspect.isawaitable(result):
            await result
            return True
        return True

    def _count_pending_approvals(self) -> int | None:
        if self._approval_registry is None:
            return None
        try:
            if self._approval_source:
                return sum(
                    1
                    for _ in self._approval_registry.pending(
                        source=self._approval_source
                    )
                )
            return sum(1 for _ in self._approval_registry.pending())
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("research.longhaul.approval_query_failed", exc_info=True)
            return None

    @contextlib.contextmanager
    def _start_span(
        self, tracer: Any, name: str, attributes: MutableMapping[str, Any]
    ) -> contextlib.AbstractContextManager[Any]:
        if tracer is None:
            yield None
            return
        try:  # pragma: no cover - depends on OpenTelemetry instrumentation
            with tracer.start_as_current_span(name) as span:
                self._set_span_attributes(span, attributes)
                yield span
        except Exception:  # pragma: no cover - tracing must not break validation
            logger.debug("research.longhaul.tracing_unavailable", exc_info=True)
            yield None

    def _emit_event(
        self, span: Any, name: str, attributes: MutableMapping[str, Any]
    ) -> None:
        if span is None:
            return
        try:  # pragma: no cover - depends on OpenTelemetry instrumentation
            span.add_event(name, attributes=dict(attributes))
        except Exception:
            logger.debug("research.longhaul.event_emit_failed", exc_info=True)

    def _set_span_attributes(
        self, span: Any, attributes: MutableMapping[str, Any]
    ) -> None:
        if span is None:
            return
        for key, value in attributes.items():
            try:  # pragma: no cover - depends on OpenTelemetry instrumentation
                span.set_attribute(key, value)
            except Exception:
                logger.debug("research.longhaul.attr_set_failed", exc_info=True)

    def _record_metrics(
        self, name: str, payload: MutableMapping[str, float | int]
    ) -> None:
        if self._metrics_sink is None:
            return
        try:
            self._metrics_sink(name, payload)
        except Exception:  # pragma: no cover - metrics sinks are best effort
            logger.debug("research.longhaul.metrics_failed", exc_info=True)


__all__ = [
    "ResearchLoopLongHaulValidator",
    "LongHaulValidationSummary",
    "LongHaulCycleReport",
]
