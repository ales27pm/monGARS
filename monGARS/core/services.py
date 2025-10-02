from __future__ import annotations

from collections.abc import Callable

from monGARS.core.bouche import Bouche, SpeechTurn, SpeechTurnManager
from monGARS.core.hippocampus import Hippocampus


class MemoryService:
    def __init__(self, hippocampus: Hippocampus):
        self._hippocampus = hippocampus

    async def store(self, user_id: str, query: str, response: str):
        await self._hippocampus.store(user_id, query, response)

    async def history(self, user_id: str, limit: int = 10):
        return await self._hippocampus.history(user_id, limit)


class SpeakerService:
    """Coordinate speech planning across concurrent conversations."""

    def __init__(
        self,
        bouche: Bouche | None = None,
        *,
        manager_factory: Callable[[], SpeechTurnManager] | None = None,
    ) -> None:
        self._manager_factory = manager_factory or SpeechTurnManager
        self._default_bouche = bouche or Bouche(manager=self._manager_factory())
        self._sessions: dict[str, Bouche] = {}

    async def speak(self, text: str, *, session_id: str | None = None) -> SpeechTurn:
        """Plan speech for ``text`` while preserving per-session state."""

        bouche = self._resolve_bouche(session_id)
        return await bouche.speak(text)

    def conversation_profile(
        self, session_id: str | None = None
    ) -> dict[str, float | int]:
        """Expose pacing metrics for a given session."""

        bouche = self._resolve_bouche(session_id)
        return bouche.conversation_profile()

    def drop_session(self, session_id: str) -> None:
        """Forget cached state for ``session_id`` (used by tests)."""

        self._sessions.pop(session_id, None)

    def _resolve_bouche(self, session_id: str | None) -> Bouche:
        if not session_id:
            return self._default_bouche
        bouche = self._sessions.get(session_id)
        if bouche is None:
            bouche = Bouche(manager=self._manager_factory())
            self._sessions[session_id] = bouche
        return bouche
