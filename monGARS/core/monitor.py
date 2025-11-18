import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any

import GPUtil
import psutil
from opentelemetry import metrics, trace

from monGARS.config import get_settings
from monGARS.telemetry import (
    LLM_DURATION_MILLISECONDS,
    LLM_ERRORS_TOTAL,
    LLM_TOKENS_TOTAL,
)

from .ui_events import event_bus, make_event

logger = logging.getLogger(__name__)

meter = metrics.get_meter(__name__)
TRAINING_CYCLE_COUNTER = meter.create_counter(
    "llm.training.cycles",
    description="Number of MNTP training cycles started and completed.",
)
TRAINING_FAILURE_COUNTER = meter.create_counter(
    "llm.training.failures",
    description="Count of MNTP training cycles that failed.",
)
TRAINING_TOKEN_COUNTER = meter.create_counter(
    "llm.training.tokens",
    unit="token",
    description="Approximate number of tokens processed during MNTP fine-tuning.",
)
LLM_TOKEN_COUNTER = meter.create_counter(
    "llm.tokens",
    unit="tokens",
    description="Tokens processed",
)
LLM_LATENCY_HISTOGRAM = meter.create_histogram(
    "llm.duration",
    unit="ms",
    description="LLM response time",
)
_otel_llm_error_counter = meter.create_counter(
    "llm.errors",
    unit="errors",
    description="LLM errors",
)


class _PrometheusAwareCounter:
    """Proxy OpenTelemetry counters into Prometheus."""

    def __init__(self, otel_counter: Any) -> None:
        self._otel_counter = otel_counter

    def add(self, amount: int, attributes: dict[str, Any] | None = None) -> None:
        attributes = attributes or {}
        self._otel_counter.add(amount, attributes)
        _record_prometheus_error(amount, attributes)


LLM_ERROR_COUNTER = _PrometheusAwareCounter(_otel_llm_error_counter)


def _generate_uuid() -> str:
    """Return a random hexadecimal UUID string."""

    return uuid.uuid4().hex


def generate_request_id() -> str:
    """Return a globally unique identifier for a single LLM request."""

    return _generate_uuid()


def generate_conversation_id() -> str:
    """Return a globally unique UUID4 identifier for ad-hoc conversations."""

    return _generate_uuid()


def _base_llm_attrs(
    user_id: str | None,
    conversation_id: str | None,
) -> tuple[dict[str, str], str, str]:
    resolved_user_id = user_id or "anonymous"
    resolved_conversation_id = conversation_id or generate_conversation_id()
    attributes = {
        "user.id": resolved_user_id,
        "conversation.id": resolved_conversation_id,
    }
    return attributes, resolved_user_id, resolved_conversation_id


def _prometheus_labels(attributes: dict[str, Any]) -> dict[str, str]:
    return {
        "model": str(attributes.get("model") or "unknown"),
    }


def _prometheus_enabled() -> bool:
    try:
        return get_settings().otel_prometheus_enabled
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("monitor.prometheus_setting_lookup_failed", exc_info=True)
        return False


def _record_prometheus_tokens(
    attributes: dict[str, Any], *, token_type: str, amount: int
) -> None:
    if amount <= 0:
        return
    if not _prometheus_enabled():
        return
    labels = _prometheus_labels(attributes)
    labels["token_type"] = token_type
    try:
        LLM_TOKENS_TOTAL.labels(**labels).inc(amount)
    except Exception:  # pragma: no cover - defensive guard
        logger.debug(
            "monitor.prometheus_token_metric_failed",
            extra={"token_type": token_type},
            exc_info=True,
        )


def _record_prometheus_latency(attributes: dict[str, Any], latency_ms: float) -> None:
    if latency_ms < 0:
        latency_ms = 0.0
    if not _prometheus_enabled():
        return
    try:
        LLM_DURATION_MILLISECONDS.labels(**_prometheus_labels(attributes)).observe(
            latency_ms
        )
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("monitor.prometheus_latency_metric_failed", exc_info=True)


def _record_prometheus_error(amount: int, attributes: dict[str, Any]) -> None:
    if amount <= 0:
        return
    if not _prometheus_enabled():
        return
    labels = _prometheus_labels(attributes)
    labels["error_type"] = str(attributes.get("error.type") or "unknown")
    try:
        LLM_ERRORS_TOTAL.labels(**labels).inc(amount)
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("monitor.prometheus_error_metric_failed", exc_info=True)


def annotate_llm_span(
    span: trace.Span,
    *,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    temperature: float | None,
    max_tokens: int | None,
    user_id: str | None,
    conversation_id: str | None,
    request_id: str | None = None,
) -> tuple[str, str, str]:
    """Attach the required Dolphin-X1 metadata to an OpenTelemetry span.

    Returns the resolved ``(user_id, conversation_id, request_id)`` tuple so the
    caller can forward identifiers to downstream metrics sinks.
    """

    base_attributes, resolved_user_id, resolved_conversation_id = _base_llm_attrs(
        user_id, conversation_id
    )
    resolved_request_id = request_id or generate_request_id()
    attributes: dict[str, Any] = {
        "llm.model_name": model_id,
        "llm.system": "dolphin-x1",
        "llm.token_count.prompt": input_tokens,
        "llm.token_count.completion": output_tokens,
        "request.id": resolved_request_id,
        **base_attributes,
    }
    if temperature is not None:
        attributes["llm.temperature"] = temperature
    if max_tokens is not None:
        attributes["llm.max_tokens"] = max_tokens
    span.set_attributes(attributes)
    return resolved_user_id, resolved_conversation_id, resolved_request_id


def record_llm_metrics(
    *,
    model_id: str,
    user_id: str,
    conversation_id: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    extra_attributes: dict[str, Any] | None = None,
) -> None:
    """Record token counts, latency, and correlation identifiers for an LLM call."""

    base_attributes, _, _ = _base_llm_attrs(user_id, conversation_id)
    attributes: dict[str, Any] = {"model": model_id, **base_attributes}
    if extra_attributes:
        attributes |= extra_attributes

    prompt_attrs = dict(attributes)
    prompt_attrs["llm.token_type"] = "prompt"
    completion_attrs = dict(attributes)
    completion_attrs["llm.token_type"] = "completion"

    LLM_TOKEN_COUNTER.add(input_tokens, prompt_attrs)
    LLM_TOKEN_COUNTER.add(output_tokens, completion_attrs)
    LLM_LATENCY_HISTOGRAM.record(latency_ms, attributes)
    _record_prometheus_tokens(attributes, token_type="prompt", amount=input_tokens)
    _record_prometheus_tokens(attributes, token_type="completion", amount=output_tokens)
    _record_prometheus_latency(attributes, latency_ms)


def get_tracer(name: str) -> trace.Tracer:
    """Return an OpenTelemetry tracer configured for ``name``."""

    return trace.get_tracer(name)


@dataclass
class SystemStats:
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    gpu_usage: float | None = None
    gpu_memory_usage: float | None = None


class SystemMonitor:
    def __init__(self, update_interval: int = 5):
        self.update_interval = update_interval

    async def get_system_stats(self) -> SystemStats:
        cpu = await asyncio.to_thread(psutil.cpu_percent, self.update_interval)
        memory = await asyncio.to_thread(psutil.virtual_memory)
        disk = await asyncio.to_thread(psutil.disk_usage, "/")
        gpu_stats = await asyncio.to_thread(self._get_gpu_stats)
        return SystemStats(
            cpu_usage=cpu,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            gpu_usage=gpu_stats.get("gpu_usage"),
            gpu_memory_usage=gpu_stats.get("gpu_memory_usage"),
        )

    def _get_gpu_stats(self) -> dict[str, float | None]:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    "gpu_usage": min(gpu.load * 100, 85),
                    "gpu_memory_usage": gpu.memoryUtil * 100,
                }
        except Exception as exc:  # pragma: no cover - optional GPU dependency
            logger.exception("Failed to query GPU stats", exc_info=exc)
        return {"gpu_usage": None, "gpu_memory_usage": None}


async def maybe_alert(
    user_id: str | None = None,
    cpu: float | None = None,
    ttfb_ms: int | None = None,
) -> None:
    """Publish performance alerts when metrics are provided."""

    data: dict[str, float | int] = {}
    if cpu is not None:
        data["cpu"] = cpu
    if ttfb_ms is not None:
        data["ttfb_ms"] = ttfb_ms
    if not data:
        return
    await event_bus().publish(make_event("performance.alert", user_id, data))


__all__ = [
    "SystemMonitor",
    "SystemStats",
    "maybe_alert",
    "TRAINING_CYCLE_COUNTER",
    "TRAINING_FAILURE_COUNTER",
    "TRAINING_TOKEN_COUNTER",
    "LLM_TOKEN_COUNTER",
    "LLM_LATENCY_HISTOGRAM",
    "LLM_ERROR_COUNTER",
    "get_tracer",
    "generate_request_id",
    "generate_conversation_id",
    "annotate_llm_span",
    "record_llm_metrics",
]
