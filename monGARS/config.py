import json
import logging
import os
import re
import secrets
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Iterable, Literal, Optional, TypeAlias, get_args

import hvac
from dotenv import dotenv_values, set_key
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
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
    PrivateAttr,
    RedisDsn,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import ArgumentError

from monGARS.core.constants import (
    DEFAULT_EMBEDDING_BACKEND,
    SUPPORTED_EMBEDDING_BACKENDS,
)
from monGARS.core.embedding_backends import normalise_embedding_backend
from monGARS.telemetry import PROMETHEUS_REGISTRY
from monGARS.utils.database import apply_database_url_overrides
from monGARS.utils.hardware import recommended_worker_count

log = logging.getLogger(__name__)


EMBEDDING_BACKEND_CHOICES: tuple[str, ...] = tuple(sorted(SUPPORTED_EMBEDDING_BACKENDS))
"""Supported embedding backends exposed to runtime configuration."""

EmbeddingBackend: TypeAlias = Literal[
    "dolphin-x1-llm2vec",
    "huggingface",
    "ollama",
    "transformers",
]


if set(get_args(EmbeddingBackend)) != set(
    EMBEDDING_BACKEND_CHOICES
):  # pragma: no cover - defensive guard
    raise RuntimeError(
        "EmbeddingBackend literal choices must match SUPPORTED_EMBEDDING_BACKENDS. "
        "Update monGARS.config.EmbeddingBackend when the supported backends change."
    )


# --- helpers (top-level) ---


def _generate_secret_key() -> str:
    """Create a high-entropy secret key suitable for symmetric JWT signing."""

    return secrets.token_urlsafe(64)


def _vault_configured(s) -> bool:
    return bool(getattr(s, "VAULT_URL", None)) and bool(getattr(s, "VAULT_TOKEN", None))


def _iter_env_files(settings: "Settings") -> Iterable[Path]:
    """Yield candidate env files configured for the settings model."""

    env_file = settings.model_config.get("env_file")
    if not env_file:
        return []

    env_files: Iterable[str | Path] = (
        (env_file,) if isinstance(env_file, (str, Path)) else env_file
    )

    resolved: list[Path] = []
    for entry in env_files:
        path = Path(entry)
        if not path.is_absolute():
            path = Path.cwd() / path
        resolved.append(path)
    return resolved


def _load_secret_from_env_files(settings: "Settings") -> str | None:
    """Return the last non-empty SECRET_KEY discovered in configured env files."""

    discovered_secret: str | None = None
    for env_path in _iter_env_files(settings):
        try:
            env_values = dotenv_values(env_path, encoding="utf-8") or {}
        except (OSError, UnicodeDecodeError) as exc:
            # Reading env files can fail when the path is missing, unreadable, or misencoded.
            # The caller treats the result as best-effort so we log and skip.
            log.debug(
                "Skipping env file %s while resolving SECRET_KEY: %s", env_path, exc
            )
            continue

        if candidate := (env_values.get("SECRET_KEY") or "").strip():
            discovered_secret = candidate

    return discovered_secret


@dataclass(frozen=True)
class _SecretKeyInputs:
    """Collect the different SECRET_KEY candidates for downstream validation."""

    resolved_value: str | None
    env_var: str | None
    env_file: str | None
    vault_configured: bool


def _collect_secret_key_inputs(settings: "Settings") -> _SecretKeyInputs:
    """Gather SECRET_KEY from config, environment variables, and env files."""

    resolved_value = (getattr(settings, "SECRET_KEY", None) or "").strip() or None
    env_var = (os.environ.get("SECRET_KEY") or "").strip() or None
    env_file = _load_secret_from_env_files(settings)
    return _SecretKeyInputs(
        resolved_value=resolved_value,
        env_var=env_var,
        env_file=env_file,
        vault_configured=_vault_configured(settings),
    )


SecretKeyOrigin = Literal[
    "missing",
    "provided",
    "vault",
    "ephemeral",
    "generated",
    "persisted",
    "deferred",
]


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


class ModelRuntimeSettings(BaseModel):
    """Quantization and sampling controls for the unified LLM runtime."""

    model_config = ConfigDict(extra="forbid")

    quantize_4bit: bool = Field(
        default=True,
        description=(
            "Enable 4-bit NF4 quantization when CUDA is available to reduce memory usage."
        ),
    )
    bnb_4bit_quant_type: str = Field(
        default="nf4",
        description="Quantization scheme applied by bitsandbytes when quantizing in 4-bit mode.",
    )
    bnb_4bit_compute_dtype: str = Field(
        default="bfloat16",
        description="Torch dtype used for computation when 4-bit quantization is active.",
    )
    bnb_4bit_use_double_quant: bool = Field(
        default=True,
        description="Enable double quantization for improved accuracy when quantizing to 4-bit.",
    )
    max_new_tokens: int = Field(
        default=512,
        ge=1,
        description="Default number of tokens the unified runtime may generate per request.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        description="Sampling temperature applied to unified runtime generations by default.",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p nucleus sampling parameter for unified runtime generations.",
    )
    top_k: int = Field(
        default=40,
        ge=0,
        description="Top-k sampling parameter for unified runtime generations.",
    )
    repetition_penalty: float = Field(
        default=1.05,
        ge=0.0,
        description="Penalty applied to repeated tokens while sampling.",
    )


class LLMPooling(str, Enum):
    MEAN = "mean"
    MAX = "max"
    CLS = "cls"


class LLMQuantization(str, Enum):
    NONE = "none"
    NF4 = "nf4"
    GPTQ = "gptq"
    FP8 = "fp8"


class LLMSettings(BaseModel):
    """LLM-specific runtime configuration."""

    model_config = ConfigDict(
        extra="forbid", alias_generator=str.upper, populate_by_name=True
    )

    quantization: LLMQuantization = Field(
        default=LLMQuantization.NF4,
        description="BitsAndBytes quantization strategy applied to LLM weights.",
    )
    load_in_4bit: bool = Field(
        default=True,
        description="Toggle 4-bit quantized loading when CUDA resources are available.",
    )
    embedding_pooling: LLMPooling = Field(
        default=LLMPooling.MEAN,
        description="Pooling strategy applied to encoder embeddings.",
    )
    serve_backend: Literal["local", "ray"] = Field(
        default="local",
        description=(
            "Backend used by the /llm API router. When set to 'ray' the API"
            " will attempt to route requests through the Ray Serve deployment"
            " before falling back to the local runtime."
        ),
    )
    use_gpu: bool = Field(
        default=False,
        description="Request GPU resources for Ray Serve replicas when available.",
    )


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

    SECRET_KEY: Optional[str] = None
    _secret_key_origin: SecretKeyOrigin = PrivateAttr(default="missing")
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_PRIVATE_KEY: str | None = Field(
        default=None,
        description=(
            "PEM-encoded private key used for asymmetric JWT algorithms (e.g. RS256)."
        ),
    )
    JWT_PUBLIC_KEY: str | None = Field(
        default=None,
        description=(
            "PEM-encoded public key paired with JWT_PRIVATE_KEY for asymmetric algorithms."
        ),
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60)

    @model_validator(mode="after")
    def _derive_secret_key_and_validate(self) -> "Settings":
        """Generate ephemeral secrets for debug builds and validate JWT configuration."""

        inputs = _collect_secret_key_inputs(self)

        # Secret precedence:
        #   1. Explicit environment variables override everything.
        #   2. Env files override config defaults using "last one wins" semantics.
        #   3. Remaining config or persisted values are treated as provided.
        secret_value = inputs.resolved_value
        if inputs.env_var and secret_value == inputs.env_var:
            secret_source = (
                "env_var"  # noqa: S105 - provenance label, not a secret value
            )
        elif inputs.env_file and secret_value == inputs.env_file:
            secret_source = (
                "env_file"  # noqa: S105 - provenance label, not a secret value
            )
        elif secret_value is not None:
            secret_source = (
                "config"  # noqa: S105 - provenance label, not a secret value
            )
        else:
            secret_source = None

        if inputs.vault_configured:
            if secret_value is None or secret_source == "env_file":
                secret_origin: SecretKeyOrigin = (
                    "deferred"  # noqa: S105 - provenance label
                )
                secret_value = None
            else:
                secret_origin = "provided"  # noqa: S105 - provenance label
        elif secret_value is not None:
            secret_origin = "provided"  # noqa: S105 - provenance label
        elif self.debug:
            secret_origin = "ephemeral"  # noqa: S105 - provenance label
        else:
            secret_origin = "missing"  # noqa: S105 - provenance label

        object.__setattr__(self, "SECRET_KEY", secret_value)
        object.__setattr__(self, "_secret_key_origin", secret_origin)

        algorithm = (self.JWT_ALGORITHM or "").strip().upper()
        object.__setattr__(self, "JWT_ALGORITHM", algorithm)

        private_key = (self.JWT_PRIVATE_KEY or "").strip() or None
        public_key = (self.JWT_PUBLIC_KEY or "").strip() or None
        object.__setattr__(self, "JWT_PRIVATE_KEY", private_key)
        object.__setattr__(self, "JWT_PUBLIC_KEY", public_key)

        symmetric_match = re.fullmatch(r"HS\d+", algorithm)
        if symmetric_match:
            if private_key or public_key:
                raise ValueError(
                    "JWT_PRIVATE_KEY and JWT_PUBLIC_KEY are not supported with symmetric JWT algorithms."
                )
        else:
            if not (private_key and public_key):
                raise ValueError(
                    "Asymmetric JWT algorithms require both JWT_PRIVATE_KEY and JWT_PUBLIC_KEY."
                )

        return self

    database_url: AnyUrl = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost/mongars_db"
    )
    db_user: str | None = Field(default=None)
    db_password: str | None = Field(default=None)
    db_host: str | None = Field(default=None)
    db_port: int | str | None = Field(default=None)
    db_name: str | None = Field(default=None)
    db_pool_size: int = Field(default=5)
    db_max_overflow: int = Field(default=10)
    db_pool_timeout: int = Field(default=30)
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")

    @model_validator(mode="after")
    def apply_database_overrides(self) -> "Settings":
        """Ensure DATABASE_URL honours discrete DB_* overrides."""

        try:
            url = make_url(str(self.database_url))
        except ArgumentError as exc:
            raise ValueError("Invalid DATABASE_URL provided") from exc

        overridden_url = apply_database_url_overrides(
            url,
            username=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
            logger=log,
            field_sources={
                "username": "DB_USER",
                "password": "DB_PASSWORD",
                "host": "DB_HOST",
                "port": "DB_PORT",
                "database": "DB_NAME",
            },
        )

        if overridden_url is not url:
            object.__setattr__(
                self,
                "database_url",
                AnyUrl(overridden_url.render_as_string(hide_password=False)),
            )

        return self

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
    rag_curated_default_provenance: str = Field(
        default="self-training",
        description=(
            "Label applied to curated datasets indicating their source pipeline."
        ),
    )
    rag_curated_default_sensitivity: str = Field(
        default="restricted",
        description="Sensitivity classification applied to curated dataset exports.",
    )
    rag_curated_reviewer: str = Field(
        default="self-training-automation",
        description="Reviewer recorded against automatically curated datasets.",
    )
    rag_curated_default_tags: list[str] = Field(
        default_factory=lambda: ["rag", "curated"],
        description="Default governance tags applied to curated dataset versions.",
    )
    rag_curated_retention_days: int = Field(
        default=30,
        ge=1,
        description="Number of days curated datasets remain valid before re-review.",
    )
    rag_curated_export_window_days: int = Field(
        default=7,
        ge=0,
        description="Duration in days that curated datasets remain exportable post-review.",
    )
    unified_model_dir: Path = Field(
        default=Path("models/dolphin_x1_unified_enhanced"),
        description=(
            "Path to the unified Dolphin-X1 bundle used by the quantized LLM/LLM2Vec runtime."
        ),
    )
    model: ModelRuntimeSettings = Field(
        default_factory=ModelRuntimeSettings,
        description="Runtime quantization and sampling controls for the unified model.",
    )
    llm: LLMSettings = Field(
        default_factory=LLMSettings,
        description="LLM runtime configuration covering quantization and pooling.",
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
    embedding_backend: EmbeddingBackend = Field(
        default=DEFAULT_EMBEDDING_BACKEND,
        description="Embedding backend provider used for semantic vector generation.",
    )
    ollama_host: AnyUrl | None = Field(
        default=None,
        description=(
            "Optional base URL for the Ollama runtime when using the Ollama embedding backend."
        ),
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        min_length=1,
        description="Model identifier requested from Ollama when embedding_backend='ollama'.",
    )
    transformers_embedding_model: str = Field(
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        description=(
            "Hugging Face model identifier used when embedding_backend='transformers'."
        ),
    )
    transformers_embedding_max_length: int = Field(
        default=4096,
        ge=1,
        description="Maximum sequence length applied during transformers embedding tokenisation.",
    )
    transformers_embedding_batch_size: int = Field(
        default=2,
        ge=1,
        description="Batch size used when pooling transformers embeddings.",
    )
    transformers_embedding_device: str | None = Field(
        default=None,
        description=(
            "Optional torch device override (e.g. 'cuda', 'cpu', 'mps') for transformers embeddings."
        ),
    )
    dolphin_x1_llm2vec_service_url: AnyUrl = Field(
        default="http://127.0.0.1:8080",
        description=(
            "Base URL for the Dolphin-X1-LLM2Vec embedding service when "
            "embedding_backend='dolphin-x1-llm2vec'."
        ),
    )
    dolphin_x1_llm2vec_service_timeout: float = Field(
        default=30.0,
        ge=0.1,
        description=(
            "Request timeout, in seconds, applied to Dolphin-X1-LLM2Vec service calls."
        ),
    )
    dolphin_x1_llm2vec_service_token: str | None = Field(
        default=None,
        description=(
            "Optional bearer token added to Authorization headers for the Dolphin-X1-LLM2Vec service."
        ),
    )

    @field_validator("embedding_backend", mode="before")
    @classmethod
    def _normalise_embedding_backend(cls, value: Any) -> str:
        return normalise_embedding_backend(
            value,
            default=DEFAULT_EMBEDDING_BACKEND,
            strict=True,
        )

    llm2vec_base_model: str = Field(
        default="nomic-ai/llm2vec-large",
        description="Base checkpoint used when instantiating the LLM2Vec encoder.",
    )
    llm2vec_encoder: str | None = Field(
        default=None,
        description="Optional adapter/peft checkpoint path for LLM2Vec fine-tuning.",
    )
    llm2vec_instruction: str = Field(
        default="Represent this conversation turn for high-precision recall.",
        description="Instruction prompt applied when generating conversational embeddings.",
    )
    llm2vec_max_batch_size: int = Field(
        default=16,
        ge=1,
        le=256,
        description="Maximum number of payloads encoded per LLM2Vec batch invocation.",
    )
    llm2vec_max_concurrency: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Upper bound on concurrent LLM2Vec encode jobs to balance throughput and memory.",
    )
    llm2vec_device_map: str = Field(
        default="auto",
        description="Device placement strategy passed to LLM2Vec when a GPU is available.",
    )
    llm2vec_torch_dtype: str | None = Field(
        default="bfloat16",
        description="Torch dtype used when loading LLM2Vec; falls back to library defaults when unset.",
    )
    llm2vec_vector_dimensions: int = Field(
        default=3072,
        ge=1,
        description="Embedding dimensionality expected from LLM2Vec fallbacks and pgvector schema.",
    )
    llm2vec_context_limit: int = Field(
        default=3,
        ge=0,
        le=12,
        description="Maximum number of semantic recall snippets injected into conversation prompts.",
    )
    llm2vec_context_max_distance: float | None = Field(
        default=0.4,
        ge=0.0,
        le=2.0,
        description=(
            "Optional cosine distance cutoff applied to semantic recall candidates; set to 0 to disable the filter."
        ),
    )
    llm2vec_fallback_candidate_window: int = Field(
        default=64,
        ge=1,
        le=512,
        description=(
            "Number of historical rows inspected when native pgvector search is unavailable."
        ),
    )
    curiosity_similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold used to determine whether prior queries satisfy the current prompt.",
    )

    @field_validator("database_url", mode="before")
    @classmethod
    def normalise_database_url(cls, value: Any) -> Any:
        if isinstance(value, str):
            if value.startswith("postgres://"):
                value = value.replace("postgres://", "postgresql://", 1)
            if value.startswith("postgresql://") and not value.startswith(
                "postgresql+"
            ):
                return value.replace("postgresql://", "postgresql+asyncpg://", 1)
        return value

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
    search_searx_enabled: EnvBool = Field(
        default=True,
        description=(
            "Enable the SearxNG search provider so the orchestrator can query a self-hosted instance."
        ),
    )
    search_searx_base_url: AnyUrl | None = Field(
        default="http://localhost:8082",
        description="Base URL for the SearxNG deployment, e.g. https://searx.example.com.",
    )
    search_searx_api_key: str | None = Field(
        default=None,
        description="Optional API key or token expected by the SearxNG instance.",
    )
    search_searx_categories: list[str] = Field(
        default_factory=list,
        description=(
            "Optional list of SearxNG categories to query (e.g. ['general', 'news']). Leave empty to use server defaults."
        ),
    )
    search_searx_safesearch: int | None = Field(
        default=None,
        ge=0,
        le=2,
        description=(
            "Optional SearxNG safesearch level (0=off, 1=moderate, 2=strict). Leave unset to defer to server defaults."
        ),
    )
    search_searx_default_language: str | None = Field(
        default="en",
        description=(
            "Fallback language passed to SearxNG when the orchestrator does not supply an explicit locale."
        ),
    )
    search_searx_result_cap: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Upper bound on SearxNG results processed per query before local ranking and truncation.",
    )
    search_searx_timeout_seconds: float = Field(
        default=6.0,
        ge=1.0,
        le=60.0,
        description=(
            "Per-request timeout when querying SearxNG. Increase for slower upstream engines or reduce to enforce snappier cutoffs."
        ),
    )
    search_searx_engines: list[str] = Field(
        default_factory=list,
        description=(
            "Optional list of SearxNG engines to target (e.g. ['google', 'bing']). Leave empty to use the server defaults."
        ),
    )
    search_searx_time_range: str | None = Field(
        default=None,
        description=(
            "Optional time range filter passed to SearxNG (e.g. 'day', 'week', 'month', 'year')."
        ),
    )
    search_searx_sitelimit: str | None = Field(
        default=None,
        description="Restrict SearxNG results to a specific domain or hostname (e.g. 'site:example.com').",
    )
    search_searx_page_size: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description=(
            "Override the per-page result count requested from SearxNG. Leave unset to allow the orchestrator to adapt automatically."
        ),
    )
    search_searx_max_pages: int = Field(
        default=2,
        ge=1,
        le=5,
        description=(
            "Maximum number of result pages to fetch from SearxNG for a single query before ranking locally."
        ),
    )
    search_searx_language_strict: EnvBool = Field(
        default=True,
        description=(
            "Force the Accept-Language header to match the orchestrator locale so SearxNG favours language-specific engines."
        ),
    )

    @field_validator("search_searx_page_size", mode="before")
    @classmethod
    def _normalize_optional_int(cls, value: Any) -> Any:
        """Allow empty strings from env vars to fall back to the default."""

        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            return stripped
        return value

    # --- SearxNG helpers ---
    @field_validator("search_searx_categories", "search_searx_engines", mode="before")
    @classmethod
    def _parse_searx_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return []
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                return [item.strip() for item in s.split(",") if item.strip()]
            if isinstance(parsed, str):
                return [parsed.strip()] if parsed.strip() else []
            if isinstance(parsed, Sequence):
                return [str(item).strip() for item in parsed if str(item).strip()]
            return []
        if isinstance(value, Sequence):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @model_validator(mode="after")
    def _validate_searx_config(self) -> "Settings":
        if self.search_searx_enabled and not self.search_searx_base_url:
            raise ValueError(
                "search_searx_base_url must be set when search_searx_enabled=True"
            )
        return self

    MLFLOW_TRACKING_URI: str = Field(default="http://localhost:5000")
    FASTAPI_URL: str = Field(default="http://localhost:8000")

    otel_service_name: str = Field(default="mongars-api")
    otel_debug: EnvBool = Field(default=False)
    otel_collector_url: str = Field(default="http://localhost:4318")
    otel_metrics_enabled: EnvBool = Field(default=True)
    otel_traces_enabled: EnvBool = Field(default=True)
    otel_prometheus_enabled: EnvBool = Field(
        default=True,
        description=(
            "Expose OpenTelemetry metrics via the in-process Prometheus registry served by the API."
        ),
    )

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
    EVENTBUS_USE_REDIS: EnvBool = Field(
        default=False,
        description=(
            "Enable the Redis-backed event bus. When false, the in-memory backend is always used."
        ),
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
    training_pipeline_enabled: EnvBool = Field(
        default=True,
        description="Flag controlling whether the background evolution training workflow runs.",
    )
    training_pipeline_user_id: str = Field(
        default="system-training",
        description="User identifier recorded for background training cycles.",
    )
    training_pipeline_version_prefix: str = Field(
        default="enc-auto",
        description="Prefix used when generating automatic training version identifiers.",
    )
    training_cycle_interval_seconds: int = Field(
        default=7200,
        ge=60,
        description="Base interval in seconds between background training cycles.",
    )
    training_cycle_jitter_seconds: int = Field(
        default=300,
        ge=0,
        description="Maximum random jitter in seconds applied to the training interval.",
    )
    research_long_haul_enabled: EnvBool = Field(
        default=True,
        description="Toggle for background long-haul validation when reinforcement tooling is available.",
    )
    research_long_haul_cycles: int = Field(
        default=3,
        ge=1,
        description="Number of validation cycles executed during research long-haul checks.",
    )
    research_long_haul_episodes_per_cycle: int = Field(
        default=64,
        ge=1,
        description="Number of reinforcement episodes executed in each validation cycle.",
    )
    research_long_haul_cooldown_seconds: float = Field(
        default=30.0,
        ge=0.0,
        description="Cooldown delay between validation cycles to surface stability issues.",
    )
    research_long_haul_approval_source: str = Field(
        default="reinforcement.reasoning",
        description="Approval source monitored while aggregating long-haul validation signals.",
    )
    research_long_haul_interval_seconds: float = Field(
        default=7200.0,
        ge=0.0,
        description="Base interval between automated research long-haul validation runs.",
    )
    research_long_haul_jitter_seconds: float = Field(
        default=600.0,
        ge=0.0,
        description="Maximum jitter applied when scheduling long-haul validation runs.",
    )
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
    def validate_db(cls, value: AnyUrl) -> AnyUrl:
        url_str = str(value)
        if url_str.startswith("postgres://"):
            raise ValueError("Invalid async PostgreSQL URL")
        if url_str.startswith("postgresql") and "postgresql+asyncpg" not in url_str:
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

    origin: SecretKeyOrigin = getattr(settings, "_secret_key_origin", "missing")

    if settings.SECRET_KEY and origin not in {"ephemeral"}:
        return settings, False

    if origin == "deferred":
        raise ValueError("SECRET_KEY must be provided in production")

    if settings.debug or origin == "ephemeral":
        message = (
            log_message
            if log_message is not None
            else "Generated ephemeral SECRET_KEY for debug mode; do not use in production."
        )
        log.warning(message)
        generated_key = _generate_secret_key()
        new_settings = settings.model_copy(update={"SECRET_KEY": generated_key})
        object.__setattr__(new_settings, "_secret_key_origin", "generated")
        return new_settings, True

    raise ValueError("SECRET_KEY must be provided in production")


def _persist_secret_key(settings: Settings) -> Settings:
    """Persist an auto-generated SECRET_KEY for misconfigured environments.

    When the runtime is configured for production but no SECRET_KEY is defined,
    containerised deployments can end up crash-looping. To keep the service
    available we generate a high-entropy key, inject it into the current process,
    and attempt to update the configured ``.env`` file so subsequent runs reuse
    the same value. The caller is responsible for logging any warnings.
    """

    generated_key = _generate_secret_key()
    os.environ["SECRET_KEY"] = generated_key

    persisted_key: str | None = None

    for env_path in _iter_env_files(settings):
        try:
            env_values = dotenv_values(env_path, encoding="utf-8") or {}
        except OSError as exc:  # pragma: no cover - unlikely but defend anyway
            log.warning("Unable to read env file %s: %s", env_path, exc)
            continue

        existing_key = (env_values.get("SECRET_KEY") or "").strip() or None
        if existing_key:
            persisted_key = existing_key
            log.info(
                "SECRET_KEY already persisted in %s; reusing existing value.", env_path
            )
            break

        try:
            env_path.parent.mkdir(parents=True, exist_ok=True)
            set_key(str(env_path), "SECRET_KEY", generated_key)
        except OSError as exc:  # pragma: no cover - writing may fail on RO filesystems
            log.warning("Unable to persist SECRET_KEY to %s: %s", env_path, exc)
            continue

        log.info("Persisted generated SECRET_KEY to %s", env_path)

        try:
            updated_values = dotenv_values(env_path, encoding="utf-8") or {}
        except OSError as exc:  # pragma: no cover - best effort re-read
            log.warning(
                "Unable to re-read SECRET_KEY from %s after persistence: %s",
                env_path,
                exc,
            )
            persisted_key = generated_key
        else:
            final_key = (updated_values.get("SECRET_KEY") or "").strip()
            if final_key:
                persisted_key = final_key
                if final_key != generated_key:
                    log.info(
                        "Adopting SECRET_KEY written by another process in %s.",
                        env_path,
                    )
        break

    final_key = (persisted_key or generated_key).strip()
    os.environ["SECRET_KEY"] = final_key

    origin: SecretKeyOrigin = "generated" if final_key == generated_key else "persisted"
    new_settings = settings.model_copy(update={"SECRET_KEY": final_key})
    object.__setattr__(new_settings, "_secret_key_origin", origin)
    return new_settings


def validate_jwt_configuration(settings: Settings) -> None:
    """Validate that JWT settings have consistent key material."""

    algorithm = settings.JWT_ALGORITHM.upper()
    symmetric_algorithms = {"HS256", "HS384", "HS512"}

    if algorithm in symmetric_algorithms:
        if settings.JWT_PRIVATE_KEY or settings.JWT_PUBLIC_KEY:
            raise ValueError(
                "JWT_PRIVATE_KEY and JWT_PUBLIC_KEY must not be defined when using symmetric JWT algorithms."
            )
        if not settings.SECRET_KEY:
            raise ValueError(
                "Symmetric JWT algorithms require SECRET_KEY to be configured."
            )
        return

    if not (settings.JWT_PRIVATE_KEY and settings.JWT_PUBLIC_KEY):
        raise ValueError(
            "Asymmetric JWT algorithms require both JWT_PRIVATE_KEY and JWT_PUBLIC_KEY."
        )


def fetch_secrets_from_vault(
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
                time.sleep(delay)

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

    if settings.otel_prometheus_enabled:
        try:
            metric_readers.append(PrometheusMetricReader())
        except Exception as exc:  # pragma: no cover - optional metrics
            log.warning("Failed to configure Prometheus metrics exporter: %s", exc)

    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=metric_readers or [],
    )

    trace_provider = TracerProvider(resource=resource)
    if settings.otel_traces_enabled:
        exporter = (
            ConsoleSpanExporter()
            if settings.otel_debug
            else OTLPSpanExporter(endpoint=f"{settings.otel_collector_url}/v1/traces")
        )
        trace_provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(trace_provider)
    metrics.set_meter_provider(meter_provider)


@lru_cache()
def get_settings() -> Settings:
    """Load configuration with debug, Vault, and production policies applied."""

    settings = Settings()

    deprecated_model_overrides: list[str] = []
    for attr in ("llm_general_model", "llm_coding_model"):
        value = getattr(settings, attr, None)
        if value:
            deprecated_model_overrides.append(attr)
    if deprecated_model_overrides:
        log.warning(
            "Legacy llm_* model overrides are deprecated; configure unified_model_dir instead.",
            extra={"deprecated_fields": deprecated_model_overrides},
        )

    if settings.debug:
        debug_secret = _generate_secret_key()
        debug_settings = settings.model_copy(update={"SECRET_KEY": debug_secret})
        object.__setattr__(debug_settings, "_secret_key_origin", "generated")
        validate_jwt_configuration(debug_settings)
        configure_telemetry(debug_settings)
        return debug_settings

    if _vault_configured(settings):
        secrets_map = fetch_secrets_from_vault(settings)
        if secrets_map:
            updates = {
                key: value
                for key, value in secrets_map.items()
                if hasattr(settings, key)
            }
            if updates:
                settings = settings.model_copy(update=updates)
            extra: dict[str, Any] = dict(
                getattr(settings, "__pydantic_extra__", {}) or {}
            )
            for key, value in secrets_map.items():
                if not hasattr(settings, key):
                    extra[key] = value
                    object.__setattr__(settings, key, value)
            if extra:
                object.__setattr__(settings, "__pydantic_extra__", extra)
            if "SECRET_KEY" in updates:
                object.__setattr__(settings, "_secret_key_origin", "vault")
        if (
            re.fullmatch(r"HS\d+", settings.JWT_ALGORITHM.strip().upper())
            and not settings.SECRET_KEY
        ):
            raise ValueError(
                "Symmetric JWT algorithms require SECRET_KEY to be configured."
            )
        validate_jwt_configuration(settings)
        configure_telemetry(settings)
        return settings

    if not settings.SECRET_KEY:
        log.critical(
            "SECRET_KEY missing while DEBUG is disabled; generating ephemeral key. "
            "Persist SECRET_KEY in your environment or Vault to avoid token invalidation."
        )
        settings = _persist_secret_key(settings)

    validate_jwt_configuration(settings)
    configure_telemetry(settings)
    return settings
