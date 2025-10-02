from __future__ import annotations

from weakref import WeakKeyDictionary

from monGARS.core.dynamic_response import AdaptiveResponseGenerator
from monGARS.core.hippocampus import Hippocampus
from monGARS.core.model_manager import LLMModelManager
from monGARS.core.peer import PeerCommunicator
from monGARS.core.persistence import PersistenceRepository
from monGARS.core.personality import PersonalityEngine
from monGARS.core.rag import RagContextEnricher

hippocampus = Hippocampus()
peer_communicator = PeerCommunicator()
_personality_engine: PersonalityEngine | None = None
_adaptive_generators: WeakKeyDictionary[
    PersonalityEngine, AdaptiveResponseGenerator
] = WeakKeyDictionary()
_persistence_repository = PersistenceRepository()
_model_manager: LLMModelManager | None = None
_rag_context_enricher: RagContextEnricher | None = None


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
