import asyncio
import logging
import os
import secrets
import sys
from functools import lru_cache
from pathlib import Path

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

    debug: bool = Field(default=False, validation_alias="DEBUG")
    host: str = Field(default="127.0.0.1", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")
    workers: int = recommended_worker_count()
    worker_deployment_name: str = Field(
        default="mongars-workers", validation_alias="WORKER_DEPLOYMENT_NAME"
    )
    worker_deployment_namespace: str = Field(
        default="default", validation_alias="WORKER_DEPLOYMENT_NAMESPACE"
    )

    SECRET_KEY: str | None = Field(
        default=None,
        min_length=1,
        description="Application secret used for JWT signing; override in production.",
    )
    JWT_ALGORITHM: str = Field(default="HS256", validation_alias="JWT_ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=60, validation_alias="ACCESS_TOKEN_EXPIRE_MINUTES"
    )

    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost/mongars_db",
        validation_alias="DATABASE_URL",
    )
    db_pool_size: int = Field(default=5, validation_alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=10, validation_alias="DB_MAX_OVERFLOW")
    db_pool_timeout: int = Field(default=30, validation_alias="DB_POOL_TIMEOUT")
    redis_url: RedisDsn = Field(
        default="redis://localhost:6379/0", validation_alias="REDIS_URL"
    )

    IN_MEMORY_CACHE_SIZE: int = Field(
        default=10000, validation_alias="IN_MEMORY_CACHE_SIZE"
    )
    DISK_CACHE_PATH: str = Field(
        default="/tmp/mongars_cache", validation_alias="DISK_CACHE_PATH"
    )
    DOC_RETRIEVAL_URL: str = Field(
        default="http://localhost:8080", validation_alias="DOC_RETRIEVAL_URL"
    )
    llm_adapter_registry_path: Path = Field(
        default=Path("models/encoders"),
        validation_alias="LLM_ADAPTER_REGISTRY_PATH",
        description="Directory storing adapter artifacts and manifest.",
    )
    curiosity_similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        validation_alias="CURIOSITY_SIMILARITY_THRESHOLD",
        description="Cosine similarity threshold used to determine whether prior queries satisfy the current prompt.",
    )
    curiosity_minimum_similar_history: int = Field(
        default=3,
        ge=0,
        validation_alias="CURIOSITY_MIN_SIMILAR_HISTORY",
        description="Minimum number of similar historical interactions required before skipping external research.",
    )
    curiosity_graph_gap_cutoff: int = Field(
        default=1,
        ge=1,
        validation_alias="CURIOSITY_GRAPH_GAP_CUTOFF",
        description="Minimum number of missing entities detected in the knowledge graph before triggering research.",
    )
    MLFLOW_TRACKING_URI: str = Field(
        default="http://localhost:5000", validation_alias="MLFLOW_TRACKING_URI"
    )
    FASTAPI_URL: str = Field(
        default="http://localhost:8000", validation_alias="FASTAPI_URL"
    )

    otel_service_name: str = Field(
        default="mongars-api", validation_alias="OTEL_SERVICE_NAME"
    )
    otel_debug: bool = Field(default=False, validation_alias="OTEL_DEBUG")
    otel_collector_url: str = Field(
        default="http://localhost:4318", validation_alias="OTEL_COLLECTOR_URL"
    )
    otel_metrics_enabled: bool = Field(
        default=True, validation_alias="OTEL_METRICS_ENABLED"
    )
    otel_traces_enabled: bool = Field(
        default=True, validation_alias="OTEL_TRACES_ENABLED"
    )

    VAULT_URL: str = Field(default="", validation_alias="VAULT_URL")
    VAULT_TOKEN: str = Field(default="", validation_alias="VAULT_TOKEN")

    AI_MODEL_NAME: str = Field(
        default="gpt-3.5-turbo", validation_alias="AI_MODEL_NAME"
    )
    AI_MODEL_TEMPERATURE: float = Field(
        default=0.7, validation_alias="AI_MODEL_TEMPERATURE"
    )
    USE_GPU: bool = Field(default=False, validation_alias="USE_GPU")
    default_language: str = "fr-CA"
    caption_prefix: str = Field(
        default="Description de l'image:", validation_alias="CAPTION_PREFIX"
    )
    otel_logs_enabled: bool = Field(default=True, validation_alias="OTEL_LOGS_ENABLED")
    style_base_model: str = Field(
        default="hf-internal-testing/tiny-random-gpt2",
        validation_alias="STYLE_BASE_MODEL",
    )
    style_adapter_dir: str = Field(
        default="/tmp/mongars_style",
        validation_alias="STYLE_ADAPTER_DIR",
    )
    style_max_history: int = Field(default=20, validation_alias="STYLE_MAX_HISTORY")
    style_min_samples: int = Field(default=2, validation_alias="STYLE_MIN_SAMPLES")
    style_max_steps: int = Field(default=6, validation_alias="STYLE_MAX_STEPS")
    style_learning_rate: float = Field(
        default=5e-4, validation_alias="STYLE_LEARNING_RATE"
    )
    style_use_qlora: bool = Field(
        default=False,
        validation_alias="STYLE_USE_QLORA",
    )
    style_max_concurrent_trainings: int = Field(
        default=2, validation_alias="STYLE_MAX_CONCURRENT_TRAININGS"
    )
    style_adapter_ttl_seconds: int = Field(
        default=3600, validation_alias="STYLE_ADAPTER_TTL"
    )
    style_adapter_maxsize: int = Field(
        default=64, validation_alias="STYLE_ADAPTER_MAXSIZE"
    )
    mimicry_positive_lexicon_path: str | None = Field(
        default=None,
        validation_alias="MIMICRY_POSITIVE_LEXICON_PATH",
        description="Optional path to a file containing additional positive sentiment terms.",
    )
    mimicry_negative_lexicon_path: str | None = Field(
        default=None,
        validation_alias="MIMICRY_NEGATIVE_LEXICON_PATH",
        description="Optional path to a file containing additional negative sentiment terms.",
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
    if os.getenv("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
        log.debug("Skipping telemetry configuration in test environment.")
        return
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
    if not settings.SECRET_KEY:
        if settings.debug:
            generated_key = secrets.token_urlsafe(64)
            log.warning(
                "SECRET_KEY not configured; generated ephemeral key for debug use only."
            )
            settings = settings.model_copy(update={"SECRET_KEY": generated_key})
        else:
            raise ValueError("SECRET_KEY must be provided in production")
    configure_telemetry(settings)
    return settings
