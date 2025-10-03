import asyncio
import json
import logging
import os
import secrets
import sys
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import hvac
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from pydantic import (
    AnyUrl,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PostgresDsn,
    RedisDsn,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from monGARS.utils.hardware import recommended_worker_count

log = logging.getLogger(__name__)


def _parse_env_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    raise ValueError(f"Invalid boolean value: {value!r}")


EnvBool = Annotated[bool, BeforeValidator(_parse_env_bool)]


class HardwareHeuristics(BaseModel):
    """Tunable parameters for hardware-aware scaling and power estimation."""

    model_config = ConfigDict(alias_generator=str.upper, populate_by_name=True)

    base_power_draw: float = Field(default=20.0, ge=0.0)
    power_per_core: float = Field(default=5.0, ge=0.0)
    power_per_gpu: float = Field(default=75.0, ge=0.0)
    minimum_power_draw: float = Field(default=15.0, ge=0.0)
    low_memory_power_threshold_gb: float = Field(default=8.0, ge=0.0)
    low_memory_power_scale: float = Field(default=0.8, ge=0.0)
    cpu_capacity_divisor: int = Field(default=2, ge=1)
    gpu_worker_bonus: int = Field(default=2, ge=0)
    worker_low_memory_soft_limit_gb: float = Field(default=8.0, ge=0.0)
    worker_memory_floor_gb: float = Field(default=4.0, ge=0.0)
    worker_low_memory_increment: int = Field(default=1, ge=0)
    worker_default_increment: int = Field(default=2, ge=0)
    warm_pool_memory_threshold_gb: float = Field(default=2.0, ge=0.0)
    warm_pool_divisor: int = Field(default=4, ge=1)
    warm_pool_cap: int = Field(default=2, ge=1)
    warm_pool_floor: int = Field(default=1, ge=1)


class Settings(BaseSettings):
    """Application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        alias_generator=str.upper,
        populate_by_name=True,
        env_nested_delimiter="__",
    )

    app_name: str = "monGARS"
    api_version: str = "1.0.0"

    debug: EnvBool = Field(default=False)
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)
    workers: int = recommended_worker_count()
    hardware_heuristics: HardwareHeuristics = Field(
        default_factory=HardwareHeuristics,
        description="Parameters controlling hardware-aware scaling and power estimation.",
    )
    worker_deployment_name: str = Field(default="mongars-workers")
    worker_deployment_namespace: str = Field(default="default")

    SECRET_KEY: str | None = Field(
        default=None,
        min_length=1,
        description="Application secret used for JWT signing; override in production.",
    )
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_PRIVATE_KEY: str | None = Field(
        default=None,
        description=(
            "Deprecated placeholder for RSA support. Ignored while JWT_ALGORITHM is locked to HS256."
        ),
    )
    JWT_PUBLIC_KEY: str | None = Field(
        default=None,
        description=(
            "Deprecated placeholder for RSA support. Ignored while JWT_ALGORITHM is locked to HS256."
        ),
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60)

    @model_validator(mode="after")
    def validate_jwt_keys(self) -> "Settings":
        """Ensure the configured JWT algorithm matches the provided key material."""

        algorithm = self.JWT_ALGORITHM.upper()
        if algorithm != "HS256":
            raise ValueError(
                "JWT_ALGORITHM must be set to HS256 until managed key storage is available."
            )
        if self.JWT_PRIVATE_KEY or self.JWT_PUBLIC_KEY:
            raise ValueError(
                "JWT_PRIVATE_KEY and JWT_PUBLIC_KEY are not supported when JWT_ALGORITHM is HS256."
            )
        object.__setattr__(self, "JWT_ALGORITHM", "HS256")
        return self

    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost/mongars_db"
    )
    db_pool_size: int = Field(default=5)
    db_max_overflow: int = Field(default=10)
    db_pool_timeout: int = Field(default=30)
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")

    IN_MEMORY_CACHE_SIZE: int = Field(default=10000)
    DISK_CACHE_PATH: str = Field(default="/tmp/mongars_cache")
    DOC_RETRIEVAL_URL: str = Field(default="http://localhost:8080")
    rag_enabled: EnvBool = Field(
        default=False,
        description="Enable repository-aware RAG context enrichment.",
    )
    rag_repo_list: list[str] = Field(
        default_factory=list,
        description="Repositories eligible for RAG enrichment when overrides are not supplied.",
    )
    rag_max_results: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Maximum number of references requested from the RAG service.",
    )
    rag_service_url: AnyUrl | None = Field(
        default=None,
        description="Optional override for the RAG context enrichment service URL.",
    )
    llm_adapter_registry_path: Path = Field(
        default=Path("models/encoders"),
        description="Directory storing adapter artifacts and manifest.",
    )
    llm_models_config_path: Path = Field(
        default=Path("configs/llm_models.json"),
        description="JSON configuration describing LLM model profiles and provisioning rules.",
    )
    llm_models_profile: str = Field(
        default="default",
        description="Name of the LLM model profile to activate for inference.",
    )
    llm_models_auto_download: EnvBool = Field(
        default=True,
        description="Automatically download missing local models when providers support it.",
    )
    llm_general_model: str | None = Field(
        default=None,
        description="Optional override for the general-purpose conversational model.",
    )
    llm_coding_model: str | None = Field(
        default=None,
        description="Optional override for the code-focused model.",
    )
    curiosity_similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold used to determine whether prior queries satisfy the current prompt.",
    )
    curiosity_minimum_similar_history: int = Field(
        default=3,
        ge=0,
        description="Minimum number of similar historical interactions required before skipping external research.",
    )
    curiosity_graph_gap_cutoff: int = Field(
        default=1,
        ge=1,
        description="Minimum number of missing entities detected in the knowledge graph before triggering research.",
    )
    MLFLOW_TRACKING_URI: str = Field(default="http://localhost:5000")
    FASTAPI_URL: str = Field(default="http://localhost:8000")

    otel_service_name: str = Field(default="mongars-api")
    otel_debug: EnvBool = Field(default=False)
    otel_collector_url: str = Field(default="http://localhost:4318")
    otel_metrics_enabled: EnvBool = Field(default=True)
    otel_traces_enabled: EnvBool = Field(default=True)

    WS_ENABLE_EVENTS: bool = Field(default=True)
    WS_ALLOWED_ORIGINS: list[AnyUrl] = Field(
        default_factory=lambda: [
            "http://localhost:8000",
            "https://your.app",
        ],
        description=(
            "Comma separated or JSON array of origins permitted to use the WebSocket API."
        ),
    )
    WS_TICKET_TTL_SECONDS: int = Field(default=45, ge=1)
    WS_CONNECTION_QUEUE_SIZE: int = Field(
        default=32,
        ge=1,
        description="Maximum number of outbound events buffered per WebSocket connection.",
    )
    WS_HEARTBEAT_INTERVAL_SECONDS: float = Field(
        default=20.0,
        gt=0.0,
        description="Interval in seconds between ping heartbeats for WebSocket connections.",
    )
    WS_HEARTBEAT_TIMEOUT_SECONDS: float = Field(
        default=60.0,
        gt=0.0,
        description="Maximum allowed silence after a ping before the WebSocket connection is closed.",
    )
    REDIS_URL: AnyUrl | None = Field(
        default=None,
        description="Optional override for the Redis connection string, e.g. redis://localhost:6379/0.",
    )
    EVENTBUS_MEMORY_QUEUE_MAXSIZE: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of events buffered per in-memory subscriber before backpressure is applied.",
    )
    WS_RATE_LIMIT_MAX_TOKENS: int = Field(
        default=0,
        ge=0,
        description=(
            "Maximum burst of WebSocket events allowed per user. Set to 0 to disable per-user throttling."
        ),
    )
    WS_RATE_LIMIT_REFILL_SECONDS: float = Field(
        default=1.0,
        gt=0.0,
        description="Number of seconds required to refill a single WebSocket event token.",
    )

    VAULT_URL: str = Field(default="")
    VAULT_TOKEN: str = Field(default="")

    AI_MODEL_NAME: str = Field(default="gpt-3.5-turbo")
    AI_MODEL_TEMPERATURE: float = Field(default=0.7)
    USE_GPU: EnvBool = Field(default=False)
    default_language: str = "fr-CA"
    caption_prefix: str = Field(default="Description de l'image:")
    otel_logs_enabled: EnvBool = Field(default=True)
    style_base_model: str = Field(default="hf-internal-testing/tiny-random-gpt2")
    style_adapter_dir: str = Field(default="/tmp/mongars_style")
    style_max_history: int = Field(default=20)
    style_min_samples: int = Field(default=2)
    style_max_steps: int = Field(default=6)
    style_learning_rate: float = Field(default=5e-4)
    style_use_qlora: EnvBool = Field(default=False)
    style_max_concurrent_trainings: int = Field(default=2)
    style_adapter_ttl_seconds: int = Field(default=3600)
    style_adapter_maxsize: int = Field(default=64)
    mimicry_positive_lexicon_path: str | None = Field(
        default=None,
        description="Optional path to a file containing additional positive sentiment terms.",
    )
    mimicry_negative_lexicon_path: str | None = Field(
        default=None,
        description="Optional path to a file containing additional negative sentiment terms.",
    )

    @field_validator("database_url")
    @classmethod
    def validate_db(cls, value: PostgresDsn) -> PostgresDsn:
        if "postgresql+asyncpg" not in str(value):
            raise ValueError("Invalid async PostgreSQL URL")
        return value

    @field_validator("WS_ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_ws_origins(cls, value: Any) -> Any:
        """Allow comma separated or JSON encoded origins."""

        if value is None:
            return []
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return []
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                parsed = [item.strip() for item in cleaned.split(",") if item.strip()]
            return parsed
        return value

    @field_validator("rag_repo_list", mode="before")
    @classmethod
    def parse_rag_repo_list(cls, value: Any) -> list[str]:
        """Normalise repository lists passed via environment variables."""

        if value is None:
            return []
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return []
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                parsed = [item.strip() for item in cleaned.split(",") if item.strip()]
            else:
                if isinstance(parsed, str):
                    parsed = [parsed]
                elif not isinstance(parsed, Sequence):
                    raise ValueError("rag_repo_list must be a sequence of strings")
            value = parsed
        if isinstance(value, Sequence):
            cleaned_values: list[str] = []
            for item in value:
                if not isinstance(item, str):
                    continue
                trimmed = item.strip()
                if trimmed:
                    cleaned_values.append(trimmed)
            return cleaned_values
        raise ValueError("rag_repo_list must be a sequence or comma separated string")

    @model_validator(mode="after")
    def sync_redis_url(self) -> "Settings":
        """Normalise the Redis override onto the canonical redis_url field."""

        if not self.REDIS_URL:
            return self

        try:
            override = RedisDsn(str(self.REDIS_URL))
        except ValueError as exc:  # pragma: no cover - configuration error
            raise ValueError("Invalid REDIS_URL provided") from exc

        object.__setattr__(self, "redis_url", override)
        return self


def ensure_secret_key(
    settings: Settings, *, log_message: str | None = None
) -> tuple[Settings, bool]:
    """Ensure the settings object contains a SECRET_KEY."""

    if settings.SECRET_KEY:
        return settings, False
    if not settings.debug:
        raise ValueError("SECRET_KEY must be provided in production")
    message = (
        log_message
        if log_message is not None
        else "SECRET_KEY not configured; generated ephemeral key for debug use only."
    )
    log.warning(message)
    generated_key = secrets.token_urlsafe(64)
    return settings.model_copy(update={"SECRET_KEY": generated_key}), True


def validate_jwt_configuration(settings: Settings) -> None:
    """Validate that JWT settings have consistent key material."""

    algorithm = settings.JWT_ALGORITHM.upper()
    if algorithm != "HS256":
        raise ValueError(
            "JWT_ALGORITHM must be HS256 to match the deployed secret management strategy."
        )
    if settings.JWT_PRIVATE_KEY or settings.JWT_PUBLIC_KEY:
        raise ValueError(
            "JWT_PRIVATE_KEY and JWT_PUBLIC_KEY must not be defined when using HS256."
        )
    if not settings.SECRET_KEY:
        raise ValueError("HS256 requires SECRET_KEY to be configured.")


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
    settings, _ = ensure_secret_key(settings)
    validate_jwt_configuration(settings)
    configure_telemetry(settings)
    return settings
