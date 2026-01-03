from __future__ import annotations

from prometheus_client import REGISTRY, Counter, Histogram

PROMETHEUS_REGISTRY = REGISTRY

HTTP_REQUESTS_TOTAL = Counter(
    "mongars_http_requests_total",
    "Total number of HTTP requests processed by the monGARS API.",
    ("method", "route", "status"),
    registry=PROMETHEUS_REGISTRY,
)

HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "mongars_http_request_duration_seconds",
    "Latency distribution for HTTP requests processed by the monGARS API.",
    ("method", "route"),
    registry=PROMETHEUS_REGISTRY,
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
    ),
)

LLM_TOKENS_TOTAL = Counter(
    "mongars_llm_tokens_total",
    "Number of tokens processed by the LLM runtime partitioned by token type.",
    ("token_type", "model"),
    registry=PROMETHEUS_REGISTRY,
)

LLM_DURATION_MILLISECONDS = Histogram(
    "mongars_llm_duration_milliseconds",
    "Distribution of LLM response times in milliseconds.",
    ("model",),
    registry=PROMETHEUS_REGISTRY,
    buckets=(
        10,
        25,
        50,
        75,
        100,
        250,
        500,
        1000,
        2000,
    ),
)

LLM_ERRORS_TOTAL = Counter(
    "mongars_llm_errors_total",
    "Number of LLM errors grouped by error type.",
    ("error_type", "model"),
    registry=PROMETHEUS_REGISTRY,
)

__all__ = [
    "HTTP_REQUEST_LATENCY_SECONDS",
    "HTTP_REQUESTS_TOTAL",
    "PROMETHEUS_REGISTRY",
    "LLM_TOKENS_TOTAL",
    "LLM_DURATION_MILLISECONDS",
    "LLM_ERRORS_TOTAL",
]
