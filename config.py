import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import PostgresDsn, RedisDsn, Field, validator
from opentelemetry.sdk.resources import Resource
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
import logging
import hvac

log = logging.getLogger(__name__)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    app_name: str = "monGARS"
    api_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "False").lower() in ("true", "1")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", 8000))
    workers: int = 4

    SECRET_KEY: str = Field(..., min_length=32, max_length=128)
    JWT_ALGORITHM: str = "RS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    database_url: PostgresDsn = Field(default="postgresql+asyncpg://postgres:postgres@localhost/mongars_db")
    db_pool_size: int = int(os.getenv("DB_POOL_SIZE", 5))
    db_max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", 10))
    db_pool_timeout: int = int(os.getenv("DB_POOL_TIMEOUT", 30))
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")

    otel_service_name: str = os.getenv("OTEL_SERVICE_NAME", "mongars-api")
    otel_debug: bool = os.getenv("OTEL_DEBUG", "False").lower() in ("true", "1")
    otel_metrics_enabled: bool = os.getenv("OTEL_METRICS_ENABLED", "True").lower() in ("true", "1")
    otel_traces_enabled: bool = os.getenv("OTEL_TRACES_ENABLED", "True").lower() in ("true", "1")
    otel_logs_enabled: bool = os.getenv("OTEL_LOGS_ENABLED", "True").lower() in ("true", "1")
    otel_collector_url: str = os.getenv("OTEL_COLLECTOR_URL", "http://localhost:4318")

    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    DOC_RETRIEVAL_URL: str = os.getenv("DOC_RETRIEVAL_URL", "http://localhost:8080")

    VAULT_URL: str = os.getenv("VAULT_URL", "")
    VAULT_TOKEN: str = os.getenv("VAULT_TOKEN", "")

    AI_MODEL_NAME: str = os.getenv("AI_MODEL_NAME", "gpt-3.5-turbo")
    AI_MODEL_TEMPERATURE: float = float(os.getenv("AI_MODEL_TEMPERATURE", 0.7))
    USE_GPU: bool = os.getenv("USE_GPU", "False").lower() in ("true", "1")

    IN_MEMORY_CACHE_SIZE: int = int(os.getenv("IN_MEMORY_CACHE_SIZE", 10000))
    DISK_CACHE_PATH: str = os.getenv("DISK_CACHE_PATH", "/tmp/mongars_cache")
    FASTAPI_URL: str = os.getenv("FASTAPI_URL", "http://localhost:8000")
    default_language: str = "fr-CA"

    @validator("database_url")
    def validate_db(cls, v):
        if "postgresql+asyncpg" not in v:
            raise ValueError("Invalid async PostgreSQL URL")
        return v

async def fetch_secrets_from_vault(settings: Settings) -> dict:
    if not settings.VAULT_URL or not settings.VAULT_TOKEN:
        if settings.debug:
            log.warning("Vault not configured; using .env values.")
            return {}
        else:
            raise RuntimeError("Vault configuration required in production")
    try:
        client = hvac.Client(url=settings.VAULT_URL, token=settings.VAULT_TOKEN)
        secret_response = client.secrets.kv.v2.read_secret_version(path="monGARS")
        secrets = secret_response["data"]["data"]
        log.info("Secrets successfully fetched from Vault.")
        return secrets
    except Exception as e:
        log.error(f"Error fetching secrets from Vault: {e}")
        return {}

def configure_telemetry(settings: Settings):
    resource = Resource(attributes={
        "service.name": settings.otel_service_name,
        "service.version": settings.api_version,
    })
    trace.set_tracer_provider(TracerProvider(resource=resource))
    metrics.set_meter_provider(MeterProvider(resource=resource))
    if settings.otel_traces_enabled:
        trace_exporter = ConsoleSpanExporter() if settings.otel_debug else OTLPSpanExporter(endpoint=settings.otel_collector_url)
        trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(trace_exporter))
    if settings.otel_metrics_enabled:
        metric_exporter = OTLPMetricExporter(endpoint=f"{settings.otel_collector_url}/v1/metrics") if not settings.otel_debug else None
        if metric_exporter:
            meter_provider = MeterProvider(resource=resource, metric_readers=[metric_exporter])
            metrics.set_meter_provider(meter_provider)

@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    # In production, secrets are fetched from Vault asynchronously.
    # For simplicity in this synchronous context, we use a blocking call.
    import asyncio
    loop = asyncio.get_event_loop()
    vault_secrets = loop.run_until_complete(fetch_secrets_from_vault(settings))
    for key, value in vault_secrets.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    configure_telemetry(settings)
    return settings