from __future__ import annotations

import json
import logging
import math
import pickle
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import DBAPIError, IntegrityError, InterfaceError, OperationalError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import Settings, get_settings
from ..init_db import (
    ConversationHistory,
    Interaction,
    UserAccount,
    UserPreferences,
    async_session_factory,
)
from .embeddings import EmbeddingBackendError, LLM2VecEmbedder, get_llm2vec_embedder
from .inference_utils import render_chat_prompt_from_text

try:  # pragma: no cover - optional dependency
    from transformers import AutoTokenizer  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    AutoTokenizer = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


SessionCallable = Callable[[Any], Awaitable[Any]]


@dataclass(slots=True)
class VectorMatch:
    """Result row returned by :meth:`PersistenceRepository.vector_search_history`."""

    record: ConversationHistory
    distance: float


@dataclass(slots=True)
class ModelSnapshot:
    """Container for model snapshot artefacts on disk."""

    path: Path
    state_dict: dict[str, Any]
    tokenizer: Any | None
    metadata: dict[str, Any] | None


class PersistenceRepository:
    def __init__(
        self,
        session_factory=async_session_factory,
        *,
        settings: Settings | None = None,
        embedder: LLM2VecEmbedder | None = None,
        enable_embeddings: bool = True,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings or get_settings()
        if enable_embeddings:
            self._embedder = (
                embedder if embedder is not None else get_llm2vec_embedder()
            )
        else:
            self._embedder = None
        self._vector_support_native: bool | None = None

    async def _execute_with_retry(
        self,
        operation: SessionCallable,
        *,
        operation_name: str,
        retry_exceptions: tuple[type[Exception], ...] = (
            OperationalError,
            InterfaceError,
            DBAPIError,
        ),
    ) -> Any:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
            retry=retry_if_exception_type(retry_exceptions),
            reraise=True,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    async with self._session_factory() as session:
                        try:
                            return await operation(session)
                        except Exception as exc:  # pragma: no cover - defensive
                            in_tx = getattr(session, "in_transaction", None)
                            if callable(in_tx) and in_tx():
                                await session.rollback()
                            max_attempts = getattr(
                                attempt.retry_state.retry_object.stop,
                                "max_attempt_number",
                                None,
                            )
                            if (
                                max_attempts is None
                                or attempt.retry_state.attempt_number < max_attempts
                            ):
                                logger.warning(
                                    "persistence.%s.retry", operation_name, exc_info=exc
                                )
                            raise
        except Exception:
            logger.exception("persistence.%s.failed", operation_name)
            raise

    def _compose_history_payload(self, query: str | None, response: str | None) -> str:
        """Combine ``query`` and ``response`` into a deterministic embedding payload."""

        segments: list[str] = []
        if query:
            segments.append(f"User: {query.strip()}")
        if response:
            segments.append(f"Assistant: {response.strip()}")
        combined = "\n".join(segments)
        if not combined.strip():
            return ""
        system_prompt = getattr(self._settings, "llm2vec_instruction", None)
        return render_chat_prompt_from_text(
            combined,
            system_prompt=system_prompt,
            include_assistant_stub=False,
        ).chatml

    async def _history_embedding_vector(
        self, query: str | None, response: str | None
    ) -> list[float] | None:
        if self._embedder is None:
            return None
        payload = self._compose_history_payload(query, response)
        if not payload.strip():
            return None
        try:
            vector, used_fallback = await self._embedder.embed_text(
                payload, instruction=self._settings.llm2vec_instruction
            )
        except EmbeddingBackendError:
            logger.error(
                "persistence.embedding.backend_unavailable",
                extra={"payload_length": len(payload)},
            )
            return None
        if used_fallback:
            logger.warning(
                "persistence.embedding.used_fallback",
                extra={"payload_length": len(payload)},
            )
        return vector

    @staticmethod
    def _vector_search_supported(session) -> bool:
        bind = getattr(session, "bind", None)
        if bind is None:
            return False
        if bind.dialect.name != "postgresql":
            return False
        comparator = getattr(ConversationHistory.vector, "comparator", None)
        return hasattr(comparator, "cosine_distance")

    def _normalise_vector(self, vector: Sequence[float] | None) -> list[float] | None:
        if vector is None:
            return None
        if hasattr(vector, "tolist"):
            vector = vector.tolist()  # type: ignore[assignment]
        values = list(vector)
        if not values:
            return None
        try:
            floats = [float(component) for component in values]
        except (TypeError, ValueError):
            return None

        dimensions = int(self._settings.llm2vec_vector_dimensions)
        if len(floats) > dimensions:
            floats = floats[:dimensions]
        elif len(floats) < dimensions:
            floats.extend(0.0 for _ in range(dimensions - len(floats)))
        return floats

    @staticmethod
    def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError(
                "Vectors must have the same dimensions for cosine distance."
            )
        dot = sum(
            left_component * right_component
            for left_component, right_component in zip(left, right)
        )
        norm_left = math.sqrt(sum(component * component for component in left))
        norm_right = math.sqrt(sum(component * component for component in right))
        if norm_left == 0 or norm_right == 0:
            return 1.0
        cosine_similarity = dot / (norm_left * norm_right)
        # Numerical noise can push the value slightly outside the [-1, 1] range.
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
        return 1.0 - cosine_similarity

    async def save_interaction(
        self,
        interaction: Interaction,
        *,
        history_query: str | None = None,
        history_response: str | None = None,
    ) -> None:
        embedding_vector = await self._history_embedding_vector(
            history_query or interaction.message,
            history_response or interaction.response,
        )
        prepared_vector = self._normalise_vector(embedding_vector)

        async def operation(session) -> None:
            async with session.begin():
                await session.merge(interaction)
                if interaction.user_id and (interaction.message or history_query):
                    session.add(
                        ConversationHistory(
                            user_id=interaction.user_id,
                            query=history_query or interaction.message,
                            response=history_response or interaction.response,
                            vector=prepared_vector,
                        )
                    )

        await self._execute_with_retry(operation, operation_name="save_interaction")

    async def save_history_entry(
        self, *, user_id: str, query: str, response: str
    ) -> None:
        embedding_vector = await self._history_embedding_vector(query, response)
        prepared_vector = self._normalise_vector(embedding_vector)

        async def operation(session) -> None:
            async with session.begin():
                session.add(
                    ConversationHistory(
                        user_id=user_id,
                        query=query,
                        response=response,
                        vector=prepared_vector,
                    )
                )

        await self._execute_with_retry(operation, operation_name="save_history_entry")

    async def get_history(self, user_id: str, limit: int = 10):
        async def operation(session):
            result = await session.execute(
                select(ConversationHistory)
                .where(ConversationHistory.user_id == user_id)
                .order_by(desc(ConversationHistory.timestamp))
                .limit(limit)
            )
            return result.scalars().all()

        return await self._execute_with_retry(operation, operation_name="get_history")

    async def vector_search_history(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = 5,
        max_distance: float | None = None,
    ) -> list[VectorMatch]:
        if limit <= 0 or not query.strip():
            return []

        query_vector = await self._history_embedding_vector(query, None)
        prepared_vector = self._normalise_vector(query_vector)
        if prepared_vector is None:
            return []

        async def operation(session):
            native_supported = self._vector_support_native
            if native_supported is None:
                native_supported = self._vector_search_supported(session)
                self._vector_support_native = native_supported

            if native_supported:
                distance_metric = ConversationHistory.vector.cosine_distance(
                    prepared_vector
                )
                stmt = (
                    select(ConversationHistory, distance_metric.label("distance"))
                    .where(ConversationHistory.user_id == user_id)
                    .where(ConversationHistory.vector.isnot(None))
                    .order_by(distance_metric)
                    .limit(limit)
                )
                if max_distance is not None:
                    stmt = stmt.where(distance_metric <= max_distance)

                result = await session.execute(stmt)
                rows = result.all()
                matches: list[VectorMatch] = []
                for record, distance in rows:
                    try:
                        numeric_distance = (
                            float(distance) if distance is not None else 0.0
                        )
                    except (TypeError, ValueError):
                        numeric_distance = 0.0
                    matches.append(
                        VectorMatch(record=record, distance=numeric_distance)
                    )
                return matches

            fallback_window = max(
                limit,
                int(self._settings.llm2vec_fallback_candidate_window),
            )
            stmt = (
                select(ConversationHistory)
                .where(ConversationHistory.user_id == user_id)
                .where(ConversationHistory.vector.isnot(None))
                .order_by(desc(ConversationHistory.timestamp))
                .limit(fallback_window)
            )
            result = await session.execute(stmt)
            records = result.scalars().all()
            matches: list[VectorMatch] = []
            for record in records:
                candidate_vector = self._normalise_vector(record.vector)
                if candidate_vector is None:
                    continue
                try:
                    distance = self._cosine_distance(prepared_vector, candidate_vector)
                except ValueError:
                    logger.warning(
                        "persistence.vector_search.dimension_mismatch",
                        extra={"record_id": record.id},
                    )
                    continue
                if max_distance is not None and distance > max_distance:
                    continue
                matches.append(VectorMatch(record=record, distance=distance))

            matches.sort(key=lambda item: item.distance)
            return matches[:limit]

        return await self._execute_with_retry(
            operation, operation_name="vector_search_history"
        )

    async def get_user_by_username(self, username: str) -> UserAccount | None:
        async def operation(session):
            result = await session.execute(
                select(UserAccount).where(UserAccount.username == username)
            )
            return result.scalar_one_or_none()

        return await self._execute_with_retry(
            operation, operation_name="get_user_by_username"
        )

    async def update_user_password(self, username: str, password_hash: str) -> bool:
        async def operation(session) -> bool:
            async with session.begin():
                result = await session.execute(
                    select(UserAccount)
                    .where(UserAccount.username == username)
                    .with_for_update()
                )
                user = result.scalar_one_or_none()
                if user is None:
                    return False
                user.password_hash = password_hash
                return True

        return await self._execute_with_retry(
            operation, operation_name="update_user_password"
        )

    async def has_admin_user(self) -> bool:
        """Return ``True`` when at least one admin account exists."""

        async def operation(session) -> bool:
            result = await session.execute(
                select(func.count())
                .select_from(UserAccount)
                .where(UserAccount.is_admin.is_(True))
            )
            count = result.scalar_one()
            return bool(count)

        return await self._execute_with_retry(
            operation,
            operation_name="has_admin_user",
        )

    async def list_usernames(self) -> list[str]:
        """Return all registered usernames sorted alphabetically."""

        async def operation(session) -> list[str]:
            result = await session.execute(
                select(UserAccount.username).order_by(UserAccount.username)
            )
            rows = result.scalars().all()
            return [username for username in rows if isinstance(username, str)]

        return await self._execute_with_retry(
            operation,
            operation_name="list_usernames",
        )

    async def get_user_preferences(self, user_id: str) -> UserPreferences | None:
        async def operation(session):
            result = await session.execute(
                select(UserPreferences).where(UserPreferences.user_id == user_id)
            )
            return result.scalar_one_or_none()

        return await self._execute_with_retry(
            operation, operation_name="get_user_preferences"
        )

    async def upsert_user_preferences(
        self,
        *,
        user_id: str,
        interaction_style: dict,
        preferred_topics: dict | None = None,
    ) -> None:
        topics = preferred_topics or {}

        async def operation(session) -> None:
            async with session.begin():
                stmt = insert(UserPreferences).values(
                    user_id=user_id,
                    interaction_style=interaction_style,
                    preferred_topics=topics,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=[UserPreferences.user_id],
                    set_={
                        "interaction_style": interaction_style,
                        "preferred_topics": topics,
                    },
                )
                await session.execute(stmt)

        await self._execute_with_retry(
            operation, operation_name="upsert_user_preferences"
        )

    async def create_user(
        self, username: str, password_hash: str, *, is_admin: bool = False
    ) -> UserAccount:
        async def operation(session):
            async with session.begin():
                user = UserAccount(
                    username=username,
                    password_hash=password_hash,
                    is_admin=is_admin,
                )
                session.add(user)
            return user

        try:
            return await self._execute_with_retry(
                operation,
                operation_name="create_user",
                retry_exceptions=(OperationalError, InterfaceError),
            )
        except IntegrityError as exc:
            raise ValueError("Username already exists") from exc

    async def create_user_atomic(
        self,
        username: str,
        password_hash: str,
        *,
        is_admin: bool = False,
        reserved_usernames: Iterable[str] | None = None,
    ) -> UserAccount:
        reserved = set(reserved_usernames or ())

        async def operation(session):
            if username in reserved:
                raise ValueError("Username already exists")
            async with session.begin():
                result = await session.execute(
                    select(UserAccount).where(UserAccount.username == username)
                )
                if result.scalar_one_or_none() is not None:
                    raise ValueError("Username already exists")
                user = UserAccount(
                    username=username,
                    password_hash=password_hash,
                    is_admin=is_admin,
                )
                session.add(user)
            return user

        try:
            return await self._execute_with_retry(
                operation,
                operation_name="create_user_atomic",
                retry_exceptions=(OperationalError, InterfaceError),
            )
        except IntegrityError as exc:
            raise ValueError("Username already exists") from exc


class PersistenceManager:
    """Utility helpers for persisting heavyweight artefacts to disk."""

    @staticmethod
    def _resolve_snapshot_root(base_path: Path | None = None) -> Path:
        settings = get_settings()
        if base_path is not None:
            return Path(base_path)
        return Path(settings.llm_adapter_registry_path).parent / "snapshots"

    @staticmethod
    def _import_torch() -> Any:
        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PyTorch is required to handle model snapshots. Install torch to enable persistence snapshots."
            ) from exc
        return torch

    @staticmethod
    def snapshot_model(
        model: Any,
        tokenizer: Any,
        *,
        slot_name: str,
        metadata: dict[str, Any] | None = None,
        base_path: Path | None = None,
    ) -> Path:
        """Persist the model state dict and tokenizer to disk."""

        root_dir = PersistenceManager._resolve_snapshot_root(base_path)
        slot_dir = root_dir / slot_name
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snapshot_dir = slot_dir / timestamp
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        model_state = getattr(model, "state_dict", None)
        if not callable(model_state):
            raise TypeError("model does not expose a callable state_dict")
        model_path = snapshot_dir / "model.pt"
        torch = PersistenceManager._import_torch()
        torch.save(model_state(), model_path)

        tokenizer_dir = snapshot_dir / "tokenizer"
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(tokenizer_dir)
        else:
            tokenizer_dir.mkdir(parents=True, exist_ok=True)
            fallback_path = tokenizer_dir / "tokenizer.pkl"
            with fallback_path.open("wb") as handle:
                pickle.dump(tokenizer, handle)

        if metadata:
            metadata_path = snapshot_dir / "metadata.json"
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, sort_keys=True)

        logger.info(
            "persistence.snapshot.saved",
            extra={
                "slot": slot_name,
                "path": str(snapshot_dir),
                "metadata_keys": sorted(metadata.keys()) if metadata else [],
            },
        )

        return snapshot_dir

    @staticmethod
    def find_latest_snapshot(
        slot_name: str, *, base_path: Path | None = None
    ) -> Path | None:
        root_dir = PersistenceManager._resolve_snapshot_root(base_path)
        slot_dir = root_dir / slot_name
        if not slot_dir.exists():
            return None
        candidates = [path for path in slot_dir.iterdir() if path.is_dir()]
        if not candidates:
            return None
        latest = max(candidates, key=lambda candidate: candidate.name)
        return latest

    @staticmethod
    def load_snapshot(
        snapshot_path: Path,
        *,
        map_location: Any | None = None,
        load_tokenizer: bool = True,
    ) -> ModelSnapshot:
        snapshot_path = Path(snapshot_path)
        if not snapshot_path.exists():
            raise FileNotFoundError(snapshot_path)

        model_path = snapshot_path / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        torch = PersistenceManager._import_torch()
        state_dict = torch.load(model_path, map_location=map_location)

        tokenizer_obj: Any | None = None
        metadata: dict[str, Any] | None = None

        metadata_path = snapshot_path / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)

        if load_tokenizer:
            tokenizer_dir = snapshot_path / "tokenizer"
            fallback_path = tokenizer_dir / "tokenizer.pkl"
            if fallback_path.exists():
                with fallback_path.open("rb") as handle:
                    tokenizer_obj = pickle.load(handle)
            elif tokenizer_dir.exists():
                if AutoTokenizer is None:
                    logger.warning(
                        "persistence.snapshot.tokenizer_unavailable",
                        extra={"path": str(tokenizer_dir)},
                    )
                else:
                    try:
                        tokenizer_obj = AutoTokenizer.from_pretrained(tokenizer_dir)
                    except Exception:  # pragma: no cover - defensive logging
                        logger.exception(
                            "persistence.snapshot.tokenizer_load_failed",
                            extra={"path": str(tokenizer_dir)},
                        )

        return ModelSnapshot(
            path=snapshot_path,
            state_dict=state_dict,
            tokenizer=tokenizer_obj,
            metadata=metadata,
        )
