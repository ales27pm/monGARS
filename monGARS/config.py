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
from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from monGARS.utils.hardware import recommended_worker_count

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "monGARS"
    api_version: str = "1.0.0"

    debug: bool = os.getenv("DEBUG", "False").lower() in ("true", "1")
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", 8000))
    workers: int = recommended_worker_count()
    worker_deployment_name: str = Field(
        default="mongars-workers", validation_alias="WORKER_DEPLOYMENT_NAME"
    )
    worker_deployment_namespace: str = Field(
        default="default", validation_alias="WORKER_DEPLOYMENT_NAMESPACE"
    )

    SECRET_KEY: str = Field(..., min_length=1)
    JWT_ALGORITHM: str = "RS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost/mongars_db"
    )
    db_pool_size: int = int(os.getenv("DB_POOL_SIZE", 5))
    db_max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", 10))
    db_pool_timeout: int = int(os.getenv("DB_POOL_TIMEOUT", 30))
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")

    IN_MEMORY_CACHE_SIZE: int = int(os.getenv("IN_MEMORY_CACHE_SIZE", 10000))
    DISK_CACHE_PATH: str = os.getenv("DISK_CACHE_PATH", "/tmp/mongars_cache")
    DOC_RETRIEVAL_URL: str = os.getenv("DOC_RETRIEVAL_URL", "http://localhost:8080")
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    FASTAPI_URL: str = os.getenv("FASTAPI_URL", "http://localhost:8000")

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

    AI_MODEL_NAME: str = os.getenv("AI_MODEL_NAME", "gpt-3.5-turbo")
    AI_MODEL_TEMPERATURE: float = float(os.getenv("AI_MODEL_TEMPERATURE", 0.7))
    USE_GPU: bool = os.getenv("USE_GPU", "False").lower() in ("true", "1")
    default_language: str = "fr-CA"
    otel_logs_enabled: bool = os.getenv("OTEL_LOGS_ENABLED", "True").lower() in (
        "true",
        "1",
    )

    @field_validator("database_url")
    @classmethod
    def validate_db(cls, value: PostgresDsn) -> PostgresDsn:
        if "postgresql+asyncpg" not in str(value):
            raise ValueError("Invalid async PostgreSQL URL")
        return value


async def fetch_secrets_from_vault(
    settings: Settings, attempts: int = 3, delay: float = 1.0
) -> dict:
    if not settings.VAULT_URL or not settings.VAULT_TOKEN:
        log.warning("Vault not configured; using .env values.")
        return {}

    for attempt in range(1, attempts + 1):
        try:
            client = hvac.Client(url=settings.VAULT_URL, token=settings.VAULT_TOKEN)
            secret_response = client.secrets.kv.v2.read_secret_version(path="monGARS")
            secrets = secret_response["data"]["data"]
            log.info("Secrets successfully fetched from Vault.")
            return secrets
        except Exception as exc:  # pragma: no cover - vault not used in tests
            log.error(
                "Error fetching secrets from Vault (attempt %s/%s): %s",
                attempt,
                attempts,
                exc,
            )
            if attempt < attempts:
                await asyncio.sleep(delay)

    log.critical("Failed to fetch secrets from Vault after %s attempts", attempts)
    return {}


def configure_telemetry(settings: Settings) -> None:
    resource = Resource(
        attributes={
            "service.name": settings.otel_service_name,
            "service.version": settings.api_version,
        }
    )

    metric_readers = []
    if settings.otel_metrics_enabled and not settings.otel_debug:
        try:
            metric_exporter = OTLPMetricExporter(
                endpoint=f"{settings.otel_collector_url}/v1/metrics"
            )
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

            metric_readers.append(PeriodicExportingMetricReader(metric_exporter))
        except Exception as exc:  # pragma: no cover - optional metrics
            log.warning("Failed to configure metrics: %s", exc)

    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=metric_readers or [],
    )

    trace_provider = TracerProvider(resource=resource)
    if settings.otel_traces_enabled:
        exporter = (
            ConsoleSpanExporter()
            if settings.otel_debug
            else OTLPSpanExporter(endpoint=settings.otel_collector_url)
        )
        trace_provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(trace_provider)
    metrics.set_meter_provider(meter_provider)


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    try:
        loop = asyncio.get_running_loop()
        vault_secrets = {}
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        vault_secrets = loop.run_until_complete(fetch_secrets_from_vault(settings))
        loop.close()
    for key, value in vault_secrets.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    if not settings.SECRET_KEY and not settings.debug:
        raise ValueError("SECRET_KEY must be provided in production")
    configure_telemetry(settings)
    return settings
