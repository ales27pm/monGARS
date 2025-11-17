import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any

import GPUtil
import psutil
from opentelemetry import metrics, trace

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
    "Tokens processed",
    "tokens",
)
LLM_LATENCY_HISTOGRAM = meter.create_histogram(
    "llm.duration",
    "LLM response time",
    "ms",
)
LLM_ERROR_COUNTER = meter.create_counter(
    "llm.errors",
    "LLM errors",
    "errors",
)


def generate_request_id() -> str:
    """Return a globally unique identifier for a single LLM request."""

    return uuid.uuid4().hex


def generate_conversation_id() -> str:
    """Return a deterministic placeholder for ad-hoc conversations."""

    return uuid.uuid4().hex


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
) -> tuple[str, str, str]:
    """Attach the required Dolphin-X1 metadata to an OpenTelemetry span.

    Returns the resolved ``(user_id, conversation_id, request_id)`` tuple so the
    caller can forward identifiers to downstream metrics sinks.
    """

    resolved_user_id = user_id or "anonymous"
    resolved_conversation_id = conversation_id or generate_conversation_id()
    request_id = generate_request_id()
    span.set_attributes(
        {
            "llm.model_name": model_id,
            "llm.system": "dolphin-x1",
            "llm.token_count.prompt": input_tokens,
            "llm.token_count.completion": output_tokens,
            "llm.temperature": temperature,
            "llm.max_tokens": max_tokens,
            "user.id": resolved_user_id,
            "conversation.id": resolved_conversation_id,
            "request.id": request_id,
        }
    )
    return resolved_user_id, resolved_conversation_id, request_id


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

    attributes = {
        "model": model_id,
        "user.id": user_id,
        "conversation.id": conversation_id,
    }
    if extra_attributes:
        attributes.update(extra_attributes)

    prompt_attrs = dict(attributes)
    prompt_attrs["llm.token_type"] = "prompt"
    completion_attrs = dict(attributes)
    completion_attrs["llm.token_type"] = "completion"

    LLM_TOKEN_COUNTER.add(input_tokens, prompt_attrs)
    LLM_TOKEN_COUNTER.add(output_tokens, completion_attrs)
    LLM_LATENCY_HISTOGRAM.record(latency_ms, attributes)


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
