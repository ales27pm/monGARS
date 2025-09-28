from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class AdvancedReasoner:
    """Lightweight rule-based reasoner for quick follow-up hints."""

    async def reason(self, query: str, user_id: str) -> dict:
        """Return supplemental reasoning hints based on keyword heuristics."""

        try:
            lowered = query.lower()
            if "why" in lowered or "pourquoi" in lowered:
                return {"result": "La cause principale est en cours d'analyse."}
            if "how" in lowered or "comment" in lowered:
                return {"result": "Voici quelques étapes possibles à considérer."}
            if "what" in lowered or "quoi" in lowered:
                return {"result": "Voici quelques informations pertinentes."}
            if "when" in lowered or "quand" in lowered:
                return {"result": "Cela dépend du contexte."}
            return {}
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "advanced_reasoner.error",
                exc_info=True,
                extra={"user_id": user_id, "error": str(exc)},
            )
            return {}
