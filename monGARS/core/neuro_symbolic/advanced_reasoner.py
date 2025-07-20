from __future__ import annotations


class AdvancedReasoner:
    """Very small reasoning helper used for demos."""

    async def reason(self, query: str, user_id: str) -> dict:
        """Return additional reasoning or empty dict if nothing applies."""

        lowered = query.lower()
        if "why" in lowered or "pourquoi" in lowered:
            return {"result": "La cause principale est en cours d'analyse."}
        if "how" in lowered or "comment" in lowered:
            return {"result": "Voici quelques étapes possibles à considérer."}
        return {}
