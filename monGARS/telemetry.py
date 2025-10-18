from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

PROMETHEUS_REGISTRY = CollectorRegistry(auto_describe=True)

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

__all__ = [
    "HTTP_REQUEST_LATENCY_SECONDS",
    "HTTP_REQUESTS_TOTAL",
    "PROMETHEUS_REGISTRY",
]
