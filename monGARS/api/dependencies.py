from __future__ import annotations

import logging
from collections.abc import Iterator
from threading import Lock
from weakref import WeakKeyDictionary

from sqlalchemy import create_engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session, sessionmaker

from monGARS.config import get_settings
from monGARS.core.dynamic_response import AdaptiveResponseGenerator
from monGARS.core.hippocampus import Hippocampus
from monGARS.core.model_manager import LLMModelManager
from monGARS.core.peer import PeerCommunicator
from monGARS.core.persistence import PersistenceRepository
from monGARS.core.personality import PersonalityEngine
from monGARS.core.rag import RagContextEnricher

logger = logging.getLogger(__name__)

hippocampus = Hippocampus()
peer_communicator = PeerCommunicator()
_personality_engine: PersonalityEngine | None = None
_adaptive_generators: WeakKeyDictionary[
    PersonalityEngine, AdaptiveResponseGenerator
] = WeakKeyDictionary()
_persistence_repository = PersistenceRepository()
_model_manager: LLMModelManager | None = None
_rag_context_enricher: RagContextEnricher | None = None
_approval_engine = None
_approval_session_factory: sessionmaker | None = None
_approval_session_lock = Lock()


def _build_approval_sync_url() -> str | None:
    settings = get_settings()
    try:
        url = make_url(str(settings.database_url))
    except Exception as exc:  # pragma: no cover - defensive configuration guard
        logger.exception("dependencies.approval.invalid_database_url")
        return None

    driver = url.drivername
    if driver.endswith("+asyncpg"):
        url = url.set(drivername=driver.replace("+asyncpg", "+psycopg"))
    elif driver.endswith("+aiosqlite"):
        url = url.set(drivername="sqlite")
    elif driver.endswith("+psycopg_async"):
        url = url.set(drivername=driver.replace("+psycopg_async", "+psycopg"))
    return url.render_as_string(hide_password=False)


def _resolve_approval_session_factory() -> sessionmaker | None:
    global _approval_engine, _approval_session_factory
    if _approval_session_factory is not None:
        return _approval_session_factory

    with _approval_session_lock:
        if _approval_session_factory is not None:
            return _approval_session_factory
        url = _build_approval_sync_url()
        if not url:
            return None
        try:
            engine = create_engine(url, future=True)
            factory = sessionmaker(
                bind=engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False,
            )
        except Exception as exc:  # pragma: no cover - engine initialisation
            logger.exception("dependencies.approval.engine_initialization_failed")
            return None

        _approval_engine = engine
        _approval_session_factory = factory
        return factory


def get_approval_db_session() -> Iterator[Session | None]:
    factory = _resolve_approval_session_factory()
    if factory is None:
        yield None
        return
    session: Session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _resolve_personality_engine() -> PersonalityEngine:
    global _personality_engine
    if _personality_engine is None:
        _personality_engine = PersonalityEngine()
    return _personality_engine


def get_hippocampus() -> Hippocampus:
    """Return the shared Hippocampus instance."""
    return hippocampus


def get_peer_communicator() -> PeerCommunicator:
    """Return the shared PeerCommunicator instance."""
    return peer_communicator


def get_persistence_repository() -> PersistenceRepository:
    """Return the shared PersistenceRepository instance."""
    return _persistence_repository


def get_personality_engine() -> PersonalityEngine:
    """Return the shared PersonalityEngine instance."""
    return _resolve_personality_engine()


def get_adaptive_response_generator(
    personality: PersonalityEngine | None = None,
) -> AdaptiveResponseGenerator:
    """Return a cached AdaptiveResponseGenerator for the given engine."""

    engine = personality or _resolve_personality_engine()
    generator = _adaptive_generators.get(engine)
    if generator is None:
        generator = AdaptiveResponseGenerator(engine)
        _adaptive_generators[engine] = generator
    return generator


def get_model_manager() -> LLMModelManager:
    """Return the shared LLM model manager instance."""

    global _model_manager
    if _model_manager is None:
        _model_manager = LLMModelManager()
    return _model_manager


def get_rag_context_enricher() -> RagContextEnricher:
    """Return the shared RAG context enricher instance."""

    global _rag_context_enricher
    if _rag_context_enricher is None:
        _rag_context_enricher = RagContextEnricher()
    return _rag_context_enricher
