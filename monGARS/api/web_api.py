from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Mapping

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from httpx import HTTPError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from monGARS.api.authentication import (
    authenticate_user,
    get_current_admin_user,
    get_current_operator_user,
    get_current_user,
)
from monGARS.api.dependencies import (
    get_adaptive_response_generator,
    get_approval_db_session,
    get_hippocampus,
    get_peer_communicator,
    get_persistence_repository,
    get_personality_engine,
)
from monGARS.api.schemas import (
    ApprovalResponse,
    ChatRequest,
    ChatResponse,
    LLMHealthResponse,
    PasswordChangeRequest,
    PeerLoadSnapshot,
    PeerMessage,
    PeerRegistration,
    PeerTelemetryEnvelope,
    PeerTelemetryPayload,
    SpeechSegmentSchema,
    SpeechTurnSchema,
    UserListResponse,
    UserRegistration,
)
from monGARS.api.ws_ticket import router as ws_ticket_router
from monGARS.config import get_settings
from monGARS.core.conversation import ConversationalModule, PromptTooLargeError
from monGARS.core.dynamic_response import AdaptiveResponseGenerator
from monGARS.core.hippocampus import MemoryItem
from monGARS.core.llm_integration import (
    CircuitBreakerOpenError,
    GuardRejectionError,
    LLMIntegration,
)
from monGARS.core.operator_approvals import get_operator_approval_registry
from monGARS.core.peer import PeerCommunicator
from monGARS.core.persistence import PersistenceRepository
from monGARS.core.personality import PersonalityEngine
from monGARS.core.security import SecurityManager, validate_user_input
from monGARS.core.ui_events import BackendUnavailable, event_bus, make_event
from monGARS.db import OperatorApproval
from monGARS.telemetry import (
    HTTP_REQUEST_LATENCY_SECONDS,
    HTTP_REQUESTS_TOTAL,
    PROMETHEUS_REGISTRY,
)

from . import authentication as auth_routes
from . import model_management
from . import rag as rag_routes
from . import ui as ui_routes
from . import ws_manager
from .rate_limiter import InMemoryRateLimiter

try:  # pragma: no cover - ray is optional
    import ray
    from ray import serve
    from ray.serve.exceptions import RayServeException as _RayServeException
except Exception:  # pragma: no cover - defensive fallback
    ray = None  # type: ignore[assignment]
    serve = None  # type: ignore[assignment]
    _RayServeException = None


class _RayServeUnavailable(Exception):
    """Sentinel used when Ray Serve is not installed."""


try:
    from opentelemetry import trace
except ImportError:  # pragma: no cover - optional dependency
    trace = None  # type: ignore[assignment]

try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
except ImportError:  # pragma: no cover - optional dependency
    FastAPIInstrumentor = None  # type: ignore[assignment]

try:
    from opentelemetry.trace import Status, StatusCode
except ImportError:  # pragma: no cover - optional dependency
    Status = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment]

UTC = datetime.UTC if hasattr(datetime, "UTC") else timezone.utc
RayServeException = _RayServeException or _RayServeUnavailable

_ws_manager = ws_manager.ws_manager


class _SecurityManagerProxy:
    """Ensure API security manager stays in sync with refreshed settings."""

    def __init__(self) -> None:
        self._manager: SecurityManager | None = None
        self._cached_settings = None

    def _resolve(self) -> SecurityManager:
        settings = get_settings()
        if self._manager is None or self._cached_settings is not settings:
            self._manager = SecurityManager(settings=settings)
            self._cached_settings = settings
        return self._manager

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)


sec_manager = _SecurityManagerProxy()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate dependency overrides and prepare application state."""

    override = app.dependency_overrides.get(get_persistence_repository)
    if override is not None and not callable(override):
        logger.error("lifespan.invalid_override", extra={"override": override})
        raise TypeError("Dependency override must be callable")
    yield


STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"

app = FastAPI(title="monGARS API", lifespan=lifespan)
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/llm", tags=["llm"])

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

_RAY_DEPLOYMENT_NAME = "RayLLMDeployment"

_settings = get_settings()
settings = _settings
if _settings.otel_traces_enabled:
    if FastAPIInstrumentor is None:
        logger.warning(
            "web_api.otel_instrumentation_missing",
            extra={"package": "opentelemetry-instrumentation-fastapi"},
        )
    else:
        FastAPIInstrumentor.instrument_app(app, excluded_urls="/metrics")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:  # pragma: no cover - build-time issue
    logger.warning(
        "web_api.static_missing",
        extra={"path": str(STATIC_DIR)},
    )

app.include_router(ws_manager.router)
app.include_router(auth_routes.router)
app.include_router(ui_routes.router)
app.include_router(model_management.router)
app.include_router(ws_ticket_router)
app.include_router(rag_routes.router)

conversation_module: ConversationalModule | None = None
ws_manager = _ws_manager

_CHAT_RATE_LIMIT_SECONDS = 1.0
_CHAT_RATE_LIMIT_PRUNE_AFTER = 60.0


def _log_rate_limit(user_id: str) -> None:
    logger.warning(
        "web_api.chat_rate_limited",
        extra={"user": _redact_user_id(user_id)},
    )


_chat_rate_limiter = InMemoryRateLimiter(
    interval_seconds=_CHAT_RATE_LIMIT_SECONDS,
    prune_after_seconds=_CHAT_RATE_LIMIT_PRUNE_AFTER,
    on_reject=_log_rate_limit,
)


@app.middleware("http")
async def record_request_metrics(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    if request.scope.get("type") != "http":  # pragma: no cover - defensive guard
        return await call_next(request)

    method = request.method.upper()
    route_template = request.url.path
    start = perf_counter()
    status_code = 500
    response: Response | None = None

    current_span = trace.get_current_span() if trace is not None else None
    if current_span is not None and not current_span.is_recording():
        current_span = None

    try:
        response = await call_next(request)
        status_code = response.status_code
    finally:
        route = request.scope.get("route")
        if route is not None and getattr(route, "path", None):
            route_template = route.path

        exc = sys.exc_info()[1]
        if exc is not None and current_span is not None:
            current_span.record_exception(exc)
            if Status is not None and StatusCode is not None:
                current_span.set_status(Status(StatusCode.ERROR, str(exc)))

        if current_span is not None:
            current_span.set_attribute("http.method", method)
            current_span.set_attribute("http.scheme", request.url.scheme)
            current_span.set_attribute("http.route", route_template)
            current_span.set_attribute("http.target", request.url.path)
            current_span.set_attribute("http.status_code", status_code)

        elapsed = perf_counter() - start
        if _settings.otel_prometheus_enabled and route_template != "/metrics":
            HTTP_REQUESTS_TOTAL.labels(
                method=method, route=route_template, status=str(status_code)
            ).inc()
            HTTP_REQUEST_LATENCY_SECONDS.labels(
                method=method, route=route_template
            ).observe(elapsed)

    assert (
        response is not None
    )  # pragma: no cover - response guaranteed unless exception
    return response


@app.get("/", include_in_schema=False)
async def frontend() -> FileResponse:
    """Serve the compiled web frontend."""

    if not STATIC_DIR.exists():
        logger.error(
            "web_api.static_directory_missing",
            extra={"path": str(STATIC_DIR)},
        )
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Static assets are unavailable.",
        )

    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        logger.error(
            "web_api.index_missing",
            extra={"path": str(index_path)},
        )
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail="Frontend build not found.",
        )

    return FileResponse(index_path)


def _redact_user_id(user_id: str) -> str:
    """Return a short, stable identifier suitable for logs."""

    if not isinstance(user_id, str) or not user_id:
        return "u:unknown"
    digest = hashlib.blake2s(user_id.encode("utf-8"), digest_size=4).hexdigest()
    return f"u:{digest}"


def _get_adaptive_response_generator_for_personality(
    personality: Annotated[PersonalityEngine, Depends(get_personality_engine)],
) -> AdaptiveResponseGenerator:
    """Resolve the adaptive response generator for the provided personality."""

    return get_adaptive_response_generator(personality)


async def reset_chat_rate_limiter_async() -> None:
    """Asynchronously reset in-memory chat rate limiting state."""

    await _chat_rate_limiter.reset()


def reset_chat_rate_limiter() -> None:
    """Reset in-memory chat rate limiting state (primarily for tests)."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        raise RuntimeError(
            "reset_chat_rate_limiter() cannot be used while an event loop is running. "
            "Use reset_chat_rate_limiter_async() instead."
        )
    asyncio.run(_chat_rate_limiter.reset())


def get_chat_rate_limiter() -> InMemoryRateLimiter:
    return _chat_rate_limiter


async def enforce_chat_rate_limit(
    current_user: Annotated[dict, Depends(get_current_user)],
    limiter: Annotated[InMemoryRateLimiter, Depends(get_chat_rate_limiter)],
) -> dict:
    await limiter.ensure_permitted(current_user["sub"])
    return current_user


def get_conversational_module(
    personality: Annotated[PersonalityEngine, Depends(get_personality_engine)],
    dynamic: Annotated[
        AdaptiveResponseGenerator,
        Depends(_get_adaptive_response_generator_for_personality),
    ],
) -> ConversationalModule:
    global conversation_module
    if conversation_module is None:
        conversation_module = ConversationalModule(
            personality=personality,
            dynamic=dynamic,
        )
    return conversation_module


def _ray_backend_configured() -> bool:
    backend = getattr(settings.llm, "serve_backend", "local")
    return backend == "ray"


def _ray_backend_available() -> bool:
    return (
        _ray_backend_configured()
        and ray is not None
        and serve is not None
        and hasattr(serve, "get_deployment_handle")
    )


async def _invoke_ray_chat(
    prompt: str,
    *,
    max_new_tokens: int | None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not _ray_backend_available():
        if _ray_backend_configured() and ray is None:
            logger.warning("web_api.ray_dependency_missing")
        return None
    try:
        handle = serve.get_deployment_handle(_RAY_DEPLOYMENT_NAME)
    except Exception as exc:  # pragma: no cover - transport errors
        logger.warning(
            "web_api.ray_handle_unavailable",
            extra={"detail": str(exc)},
        )
        return None
    payload: dict[str, Any] = {"prompt": prompt}
    if max_new_tokens is not None:
        payload["max_new_tokens"] = max_new_tokens
    if context:
        payload["context"] = dict(context)
    try:
        return await handle.remote(payload)
    except (RayServeException, asyncio.TimeoutError) as exc:
        logger.warning(
            "web_api.ray_generation_failed",
            extra={"detail": str(exc)},
        )
        return None


async def _invoke_ray_health_check() -> dict[str, Any] | None:
    if not _ray_backend_available():
        return None
    try:
        handle = serve.get_deployment_handle(_RAY_DEPLOYMENT_NAME)
    except Exception as exc:  # pragma: no cover - transport errors
        logger.warning(
            "web_api.ray_handle_unavailable",
            extra={"detail": str(exc)},
        )
        return None
    try:
        return await handle.options(method_name="health_check").remote()
    except (RayServeException, asyncio.TimeoutError) as exc:
        logger.warning(
            "web_api.ray_health_failed",
            extra={"detail": str(exc)},
        )
        return None


def _build_default_speech_turn(text: str) -> SpeechTurnSchema:
    created_at = datetime.now(UTC)
    word_count = len(text.split())
    estimated_duration = max(word_count / 2.5, 0.0)
    segment = SpeechSegmentSchema(
        text=text,
        estimated_duration=estimated_duration,
        pause_after=0.0,
    )
    avg_wps = word_count / estimated_duration if estimated_duration else 0.0
    return SpeechTurnSchema(
        turn_id=f"llm-{hashlib.blake2s(text.encode('utf-8'), digest_size=6).hexdigest()}",
        text=text,
        created_at=created_at,
        segments=[segment],
        average_words_per_second=avg_wps,
        tempo=0.0,
    )


def _chat_response_from_payload(payload: dict[str, Any]) -> ChatResponse:
    response_text = str(payload.get("response", ""))
    speech_turn_data = payload.get("speech_turn")
    speech_turn: SpeechTurnSchema | None = None
    if isinstance(speech_turn_data, SpeechTurnSchema):
        speech_turn = speech_turn_data
    elif isinstance(speech_turn_data, dict):
        try:
            speech_turn = SpeechTurnSchema(**speech_turn_data)
        except (ValidationError, TypeError, ValueError):
            speech_turn = None
    if speech_turn is None:
        speech_turn = _build_default_speech_turn(response_text)
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        processing_time = float(payload.get("processing_time", 0.0))
    except (TypeError, ValueError):
        processing_time = 0.0
    return ChatResponse(
        response=response_text,
        confidence=confidence,
        processing_time=processing_time,
        speech_turn=speech_turn,
    )


def _build_guard_context(
    chat: ChatRequest, current_user: Mapping[str, Any]
) -> dict[str, Any]:
    """Construct the security guard context passed to the LLM runtime."""

    allowed = chat.allowed_actions or current_user.get("allowed_actions")
    if allowed is None:
        allowed = current_user.get("actions")
    normalised = [
        str(action).strip()
        for action in allowed or []
        if isinstance(action, str) and action.strip()
    ]
    if not normalised:
        normalised = ["personal_data_access"]
    context: dict[str, Any] = {
        "user_id": current_user.get("sub") or current_user.get("id"),
        "allowed_actions": normalised,
    }
    token_ref = (
        chat.token_ref
        or current_user.get("token_ref")
        or current_user.get("approval_token_ref")
    )
    if token_ref:
        context["token_ref"] = token_ref
    approval_token = chat.approval_token or current_user.get("approval_token")
    if approval_token:
        context["approval_token"] = approval_token
    return context


async def _generate_local_llm_payload(
    prompt: str,
    *,
    max_new_tokens: int | None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    llm = LLMIntegration.instance()
    start = perf_counter()
    kwargs: dict[str, Any] = {}
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = max_new_tokens
    guard_context = dict(context) if context else {}
    response = await asyncio.to_thread(
        llm.generate,
        prompt,
        context=guard_context,
        **kwargs,
    )
    duration = perf_counter() - start
    return {"response": response, "processing_time": duration}


@app.post("/token")
async def login(
    repo: Annotated[PersistenceRepository, Depends(get_persistence_repository)],
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> dict:
    """Return a simple access token."""
    user = await authenticate_user(
        repo,
        form_data.username,
        form_data.password,
        sec_manager,
    )
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    token = sec_manager.create_access_token(
        {
            "sub": user.username,
            "admin": user.is_admin,
        }
    )
    return {"access_token": token, "token_type": "bearer"}


async def _persist_registration(
    repo: PersistenceRepository,
    reg: UserRegistration,
    *,
    is_admin: bool,
) -> dict:
    try:
        await repo.create_user_atomic(
            reg.username,
            sec_manager.get_password_hash(reg.password),
            is_admin=is_admin,
        )
    except ValueError as exc:
        logger.debug(
            "auth.register.conflict",
            extra={"username": reg.username},
            exc_info=exc,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    return {"status": "registered", "is_admin": is_admin}


@app.post("/api/v1/user/register")
async def register_user(
    reg: UserRegistration,
    repo: Annotated[PersistenceRepository, Depends(get_persistence_repository)],
) -> dict:
    return await _persist_registration(repo, reg, is_admin=False)


@app.post("/api/v1/user/register/admin")
async def register_admin_user(
    reg: UserRegistration,
    repo: Annotated[PersistenceRepository, Depends(get_persistence_repository)],
) -> dict:
    try:
        if await repo.has_admin_user():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin already exists",
            )
    except HTTPException:
        raise
    except (
        RuntimeError,
        SQLAlchemyError,
    ) as exc:  # pragma: no cover - unexpected failure
        logger.error(
            "auth.register_admin.state_check_failed",
            extra={"username": reg.username},
            exc_info=exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to determine admin availability",
        ) from exc
    return await _persist_registration(repo, reg, is_admin=True)


@app.get("/api/v1/user/list", response_model=UserListResponse)
async def list_users(
    current_admin: Annotated[dict, Depends(get_current_admin_user)],
    repo: Annotated[PersistenceRepository, Depends(get_persistence_repository)],
) -> UserListResponse:
    try:
        usernames = await repo.list_usernames()
    except (RuntimeError, SQLAlchemyError) as exc:
        logger.error(
            "auth.list_users_failed",
            extra={"admin": _redact_user_id(current_admin.get("sub"))},
            exc_info=exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to load users",
        ) from exc

    return UserListResponse(users=usernames)


@app.post("/api/v1/user/change-password")
async def change_password(
    payload: PasswordChangeRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
    repo: Annotated[PersistenceRepository, Depends(get_persistence_repository)],
) -> dict:
    username = current_user.get("sub")
    if not isinstance(username, str) or not username:
        logger.warning(
            "auth.change_password.invalid_subject",
            extra={"subject": _redact_user_id(username)},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing subject",
        )

    user = await repo.get_user_by_username(username)
    if user is None:
        logger.warning(
            "auth.change_password.user_missing",
            extra={"username": _redact_user_id(username)},
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if not sec_manager.verify_password(payload.old_password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Incorrect password",
        )

    updated = await repo.update_user_password(
        username, sec_manager.get_password_hash(payload.new_password)
    )
    if not updated:
        logger.error(
            "auth.change_password.update_failed",
            extra={"username": _redact_user_id(username)},
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return {"status": "changed"}


@app.get("/metrics", include_in_schema=False)
async def secured_metrics_endpoint(
    _: Annotated[dict, Depends(get_current_user)],
) -> Response:
    """Expose Prometheus metrics collected from the API process."""

    if not _settings.otel_prometheus_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    payload = generate_latest(PROMETHEUS_REGISTRY)
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict:
    return {"status": "ready"}


@app.get("/api/v1/conversation/history", response_model=list[MemoryItem])
async def conversation_history(
    user_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    store: Annotated[Any, Depends(get_hippocampus)],
    limit: int = 10,
) -> list[MemoryItem]:
    if not isinstance(user_id, str):
        logger.warning(
            "conversation.history_invalid_user_id",
            extra={"user": _redact_user_id(user_id)},
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="user_id must be a non-empty string",
        )
    normalized_user_id = user_id.strip()
    if not normalized_user_id:
        logger.warning(
            "conversation.history_invalid_user_id",
            extra={"user": _redact_user_id(user_id)},
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="user_id must be a non-empty string",
        )
    if not isinstance(limit, int) or limit <= 0:
        logger.warning(
            "conversation.history_invalid_limit",
            extra={"user": _redact_user_id(normalized_user_id), "limit": limit},
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="limit must be a positive integer",
        )
    if normalized_user_id != current_user.get("sub"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    if not isinstance(user_id, str) or not user_id.strip():
        logger.warning(
            "conversation.history_invalid_user_id",
            extra={"user": _redact_user_id(user_id)},
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="user_id must be a non-empty string",
        )
    if not isinstance(limit, int) or limit <= 0:
        logger.warning(
            "conversation.history_invalid_limit",
            extra={"user": _redact_user_id(user_id), "limit": limit},
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="limit must be a positive integer",
        )
    try:
        return await store.history(normalized_user_id, limit=limit)
    except (
        RuntimeError,
        SQLAlchemyError,
    ) as exc:  # pragma: no cover - unexpected errors
        logger.exception(
            "conversation.history_failed",
            extra={"user": _redact_user_id(normalized_user_id), "limit": limit},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to load conversation history",
        ) from exc


@app.post("/api/v1/conversation/chat", response_model=ChatResponse)
async def chat(
    chat: ChatRequest,
    current_user: Annotated[dict, Depends(enforce_chat_rate_limit)],
    conv: Annotated[ConversationalModule, Depends(get_conversational_module)],
) -> ChatResponse:
    user_id = current_user["sub"]
    try:
        data = validate_user_input({"user_id": user_id, "query": chat.message})
    except ValueError as exc:
        logger.warning(
            "web_api.chat_invalid_input",
            extra={"user": _redact_user_id(user_id), "detail": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    try:
        result = await conv.generate_response(
            user_id, data["query"], session_id=chat.session_id
        )
    except PromptTooLargeError as exc:
        logger.warning(
            "web_api.chat_prompt_too_large",
            extra={
                "user": _redact_user_id(user_id),
                "prompt_tokens": exc.prompt_tokens,
                "token_limit": exc.limit,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                "Prompt exceeds the maximum supported token limit. "
                "Please shorten your request and try again."
            ),
        ) from exc
    except asyncio.TimeoutError as exc:
        logger.warning(
            "web_api.chat_inference_timeout",
            extra={"user": _redact_user_id(user_id)},
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Chat response timed out",
        ) from exc
    except HTTPException:
        raise
    except (
        CircuitBreakerOpenError,
        HTTPError,
        RuntimeError,
        SQLAlchemyError,
        ValueError,
    ) as exc:
        logger.exception(
            "web_api.chat_inference_failed",
            extra={"user": _redact_user_id(user_id)},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate chat response",
        ) from exc
    response_payload = {
        "query": data["query"],
        "response": result.get("text", ""),
        "timestamp": datetime.now(UTC).isoformat(),
        "speech_turn": result.get("speech_turn"),
    }
    try:
        event = make_event(
            "chat.message",
            user_id,
            response_payload,
        )
        await event_bus().publish(event)
    except BackendUnavailable:
        logger.exception(
            "web_api.chat_event_publish_failed",
            extra={"user": _redact_user_id(user_id)},
        )
    except (OSError, RuntimeError):
        logger.exception(
            "web_api.chat_event_publish_failed",
            extra={"user": _redact_user_id(user_id)},
        )
    return ChatResponse(
        response=response_payload["response"],
        confidence=result.get("confidence", 0.0),
        processing_time=result.get("processing_time", 0.0),
        speech_turn=result.get("speech_turn"),
    )


@app.post("/api/v1/peer/message")
async def peer_message(
    message: PeerMessage,
    current_user: dict = Depends(get_current_user),
    communicator: PeerCommunicator = Depends(get_peer_communicator),
) -> dict:
    """Receive an encrypted message from a peer."""
    try:
        data = communicator.decode(message.payload)
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning(
            "peer.message_invalid_payload",
            extra={"detail": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid peer payload",
        ) from exc
    except (RuntimeError, OSError, TypeError) as exc:
        logger.exception("peer.message_decode_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process peer payload",
        ) from exc
    return {"status": "received", "data": data}


@app.post("/api/v1/peer/register")
async def peer_register(
    registration: PeerRegistration,
    current_user: dict = Depends(get_current_admin_user),
    communicator: PeerCommunicator = Depends(get_peer_communicator),
) -> dict:
    """Register a peer URL for future broadcasts."""
    url = registration.url
    if url in communicator.peers:
        return {"status": "already registered", "count": len(communicator.peers)}
    communicator.peers.add(url)
    return {"status": "registered", "count": len(communicator.peers)}


@app.post("/api/v1/peer/unregister")
async def peer_unregister(
    registration: PeerRegistration,
    current_user: dict = Depends(get_current_admin_user),
    communicator: PeerCommunicator = Depends(get_peer_communicator),
) -> dict:
    """Remove a previously registered peer URL."""
    url = registration.url
    if url in communicator.peers:
        communicator.peers.remove(url)
        return {"status": "unregistered", "count": len(communicator.peers)}
    return {"status": "not registered", "count": len(communicator.peers)}


@app.get("/api/v1/peer/list", response_model=list[str])
async def peer_list(
    current_user: dict = Depends(get_current_admin_user),
    communicator: PeerCommunicator = Depends(get_peer_communicator),
) -> list[str]:
    """Return the list of registered peer URLs."""
    return sorted(communicator.peers)


@app.get("/api/v1/peer/load", response_model=PeerLoadSnapshot)
async def peer_load(
    current_user: dict = Depends(get_current_admin_user),
    communicator: PeerCommunicator = Depends(get_peer_communicator),
) -> PeerLoadSnapshot:
    """Expose this node's scheduler load metrics to peers."""

    snapshot = await communicator.get_local_load()
    return PeerLoadSnapshot(**snapshot)


@app.post("/api/v1/peer/telemetry", status_code=status.HTTP_202_ACCEPTED)
async def peer_telemetry_ingest(
    report: PeerTelemetryPayload,
    current_user: dict = Depends(get_current_admin_user),
    communicator: PeerCommunicator = Depends(get_peer_communicator),
) -> dict[str, str]:
    """Accept telemetry published by peer schedulers."""

    communicator.ingest_remote_telemetry(report.source, report.model_dump())
    return {"status": "accepted"}


@app.get("/api/v1/peer/telemetry", response_model=PeerTelemetryEnvelope)
async def peer_telemetry_snapshot(
    current_user: dict = Depends(get_current_admin_user),
    communicator: PeerCommunicator = Depends(get_peer_communicator),
) -> PeerTelemetryEnvelope:
    """Return cached telemetry for local and remote schedulers."""

    telemetry = communicator.get_peer_telemetry(include_self=True)
    payloads = [PeerTelemetryPayload(**entry) for entry in telemetry]
    return PeerTelemetryEnvelope(telemetry=payloads)


@router.post("/security/approve", response_model=ApprovalResponse)
async def approve_request(
    token: str,
    operator_id: str,
    current_operator: dict = Depends(get_current_operator_user),
    db: Session | None = Depends(get_approval_db_session),
) -> ApprovalResponse:
    if current_operator.get("sub") != operator_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"status": "error", "reason": "operator_mismatch"},
        )

    token_value = (token or "").strip()
    if not token_value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"status": "error", "reason": "missing_token"},
        )

    registry = get_operator_approval_registry()
    request_entry = registry.find_by_token(token_value)
    if request_entry is None:
        logger.warning(
            "web_api.approval.invalid_token",
            extra={"operator": _redact_user_id(operator_id)},
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="invalid_token",
        )
    if request_entry.is_approved:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="already_approved",
        )

    registry.approve(request_entry.request_id, operator=operator_id)
    token_ref = request_entry.request_id

    if db is not None:
        try:
            approval = (
                db.query(OperatorApproval).filter_by(approval_token=token_value).first()
            )
            if approval is None:
                approval = OperatorApproval(
                    id=token_ref,
                    user_id=str(request_entry.payload.get("user_id", "anonymous")),
                    prompt_hash=str(request_entry.payload.get("prompt_hash", ""))[:8],
                    pii_entities=request_entry.payload.get("pii_entities", []),
                    approval_token=token_value,
                )
                db.add(approval)
            approval.approved_at = datetime.now(UTC)
            approval.approved_by = operator_id
            db.commit()
        except SQLAlchemyError:  # pragma: no cover - database failure
            db.rollback()
            logger.exception("web_api.approval.commit_failed")

    return ApprovalResponse(token_ref=token_ref)


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    chat: ChatRequest,
    current_user: Annotated[dict, Depends(enforce_chat_rate_limit)],
) -> ChatResponse:
    user_id = current_user["sub"]
    try:
        data = validate_user_input({"user_id": user_id, "query": chat.message})
    except ValueError as exc:
        logger.warning(
            "web_api.llm_chat_invalid_input",
            extra={"user": _redact_user_id(user_id), "detail": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    prompt = data["query"]
    max_tokens = getattr(settings.model, "max_new_tokens", None)
    guard_context = _build_guard_context(chat, current_user)
    ray_result = await _invoke_ray_chat(
        prompt,
        max_new_tokens=max_tokens,
        context=guard_context,
    )
    if ray_result is not None:
        if ray_result.get("error") == "approval_required":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=ray_result,
            )
        if ray_result.get("error"):
            logger.warning(
                "web_api.llm_chat_ray_error",
                extra={"user": _redact_user_id(user_id), "error": ray_result},
            )
        else:
            return _chat_response_from_payload(ray_result)

    try:
        local_payload = await _generate_local_llm_payload(
            prompt,
            max_new_tokens=max_tokens,
            context=guard_context,
        )
    except GuardRejectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=exc.payload,
        ) from exc
    except Exception as exc:  # pragma: no cover - unexpected runtime failure
        logger.exception(
            "web_api.llm_chat_local_failed",
            extra={"user": _redact_user_id(user_id)},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate chat response",
        ) from exc
    return _chat_response_from_payload(local_payload)


@router.get("/health", response_model=LLMHealthResponse)
async def llm_health() -> LLMHealthResponse:
    model_name = getattr(settings.llm, "model_name", None)
    if _ray_backend_configured():
        health_payload = await _invoke_ray_health_check()
        if health_payload is not None:
            status_value = str(health_payload.get("status", "healthy"))
            detail = health_payload.get("detail") or health_payload.get("error")
            return LLMHealthResponse(
                status="healthy" if status_value == "healthy" else "unhealthy",
                backend="ray",
                model=health_payload.get("model", model_name),
                last_check=health_payload.get("last_check"),
                detail=detail,
            )
        backend = "ray" if _ray_backend_available() else "unavailable"
        detail = (
            "Ray Serve deployment is unavailable"
            if backend == "ray"
            else "Ray Serve integration is not installed"
        )
        return LLMHealthResponse(
            status="unhealthy",
            backend=backend,
            model=model_name,
            detail=detail,
        )

    return LLMHealthResponse(status="healthy", backend="local", model=model_name)


@router.get("/metrics", include_in_schema=False)
async def metrics_endpoint() -> Response:
    if not _settings.otel_prometheus_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return Response(
        generate_latest(PROMETHEUS_REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )


app.include_router(router)
