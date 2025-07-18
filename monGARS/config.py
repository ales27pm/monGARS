import asyncio
import logging
import os
from functools import lru_cache

import hvac
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from pydantic import Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "monGARS"
    api_version: str = "1.0.0"

    debug: bool = os.getenv("DEBUG", "False").lower() in ("true", "1")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", 8000))
    workers: int = 4

    SECRET_KEY: str = Field("unsafe-secret", min_length=1)
    JWT_ALGORITHM: str = "RS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost/mongars_db"
    )
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")

    IN_MEMORY_CACHE_SIZE: int = int(os.getenv("IN_MEMORY_CACHE_SIZE", 10000))
    DISK_CACHE_PATH: str = os.getenv("DISK_CACHE_PATH", "/tmp/mongars_cache")
    DOC_RETRIEVAL_URL: str = os.getenv("DOC_RETRIEVAL_URL", "http://localhost:8080")

    otel_service_name: str = os.getenv("OTEL_SERVICE_NAME", "mongars-api")
    otel_debug: bool = os.getenv("OTEL_DEBUG", "False").lower() in ("true", "1")
    otel_collector_url: str = os.getenv("OTEL_COLLECTOR_URL", "http://localhost:4318")
    otel_metrics_enabled: bool = os.getenv("OTEL_METRICS_ENABLED", "True").lower() in (
        "true",
        "1",
    )
    otel_traces_enabled: bool = os.getenv("OTEL_TRACES_ENABLED", "True").lower() in (
        "true",
        "1",
    )

    VAULT_URL: str = os.getenv("VAULT_URL", "")
    VAULT_TOKEN: str = os.getenv("VAULT_TOKEN", "")


async def fetch_secrets_from_vault(settings: Settings) -> dict:
    if not settings.VAULT_URL or not settings.VAULT_TOKEN:
        log.warning("Vault not configured; using .env values.")
        return {}

    try:
        client = hvac.Client(url=settings.VAULT_URL, token=settings.VAULT_TOKEN)
        secret_response = client.secrets.kv.v2.read_secret_version(path="monGARS")
        secrets = secret_response["data"]["data"]
        log.info("Secrets successfully fetched from Vault.")
        return secrets
    except Exception as exc:  # pragma: no cover - vault not used in tests
        log.error("Error fetching secrets from Vault: %s", exc)
        return {}


def configure_telemetry(settings: Settings) -> None:
    resource = Resource(
        attributes={
            "service.name": settings.otel_service_name,
            "service.version": settings.api_version,
        }
    )
    trace_provider = TracerProvider(resource=resource)
    meter_provider = MeterProvider(resource=resource)

    if settings.otel_traces_enabled:
        exporter = (
            ConsoleSpanExporter()
            if settings.otel_debug
            else OTLPSpanExporter(endpoint=settings.otel_collector_url)
        )
        trace_provider.add_span_processor(BatchSpanProcessor(exporter))

    if settings.otel_metrics_enabled and not settings.otel_debug:
        try:
            metric_exporter = OTLPMetricExporter(
                endpoint=f"{settings.otel_collector_url}/v1/metrics"
            )
            meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[metric_exporter],
            )
        except Exception as exc:  # pragma: no cover - optional metrics
            log.warning("Failed to configure metrics: %s", exc)
            meter_provider = MeterProvider(resource=resource)

    trace.set_tracer_provider(trace_provider)
    metrics.set_meter_provider(meter_provider)


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    loop = asyncio.get_event_loop()
    vault_secrets = loop.run_until_complete(fetch_secrets_from_vault(settings))
    for key, value in vault_secrets.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    configure_telemetry(settings)
    return settings
