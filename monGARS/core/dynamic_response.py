from __future__ import annotations


class AdaptiveResponseGenerator:
    """Simple adaptive response generator based on personality."""

    async def generate_adaptive_response(self, text: str, personality: dict) -> str:
        """Return an adapted response based on user personality."""

        formality = personality.get("formality", 0.5)
        humor = personality.get("humor", 0.5)
        enthusiasm = personality.get("enthusiasm", 0.5)

        adapted = text
        if formality > 0.7:
            adapted = adapted.replace(" tu ", " vous ")
        elif formality < 0.3:
            adapted = adapted.replace(" vous ", " tu ")

        if enthusiasm > 0.7:
            adapted += "!"
        if humor > 0.7:
            adapted += " \U0001f603"

        return adapted
