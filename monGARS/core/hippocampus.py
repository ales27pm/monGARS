import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    user_id: str
    query: str
    response: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Hippocampus:
    """Simple in-memory store for conversation history."""

    def __init__(self) -> None:
        self._memory: Dict[str, List[MemoryItem]] = {}

    async def store(self, user_id: str, query: str, response: str) -> None:
        """Persist a query/response pair for a user."""
        logger.debug("Storing interaction for %s", user_id)
        self._memory.setdefault(user_id, []).append(
            MemoryItem(user_id=user_id, query=query, response=response)
        )

    async def history(self, user_id: str, limit: int = 10) -> List[MemoryItem]:
        """Return recent conversation history."""
        items = self._memory.get(user_id, [])
        return list(reversed(items[-limit:]))
