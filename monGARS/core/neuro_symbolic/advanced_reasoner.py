from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class AdvancedReasoner:
    """Lightweight rule-based reasoner for quick follow-up hints."""

    async def reason(self, query: str, user_id: str) -> dict:
        """Return supplemental reasoning hints based on keyword heuristics."""

        try:
            tokens = {token for token in re.findall(r"\b\w+\b", query.lower())}
            if not tokens:
                return {}
            if tokens & {"why", "pourquoi"}:
                return {"result": "La cause principale est en cours d'analyse."}
            if tokens & {"how", "comment"}:
                return {"result": "Voici quelques étapes possibles à considérer."}
            if tokens & {"what", "quoi"}:
                return {"result": "Voici quelques informations pertinentes."}
            if tokens & {"when", "quand"}:
                return {"result": "Cela dépend du contexte."}
            return {}
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "advanced_reasoner.error",
                exc_info=True,
                extra={"user_id": user_id, "error": str(exc)},
            )
            return {}
