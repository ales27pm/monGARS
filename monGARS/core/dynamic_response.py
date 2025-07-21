from __future__ import annotations


class AdaptiveResponseGenerator:
    """Simple adaptive response generator based on personality."""

    def generate_adaptive_response(self, text: str, personality: dict) -> str:
        """Return an adapted response based on user personality."""

        formality = personality.get("formality", 0.5)
        humor = personality.get("humor", 0.5)
        enthusiasm = personality.get("enthusiasm", 0.5)

        adapted = text

        # Handle formality with more comprehensive pronoun replacement
        if formality > 0.7:
            import re

            adapted = re.sub(r"\btu\b", "vous", adapted, flags=re.IGNORECASE)
        elif formality < 0.3:
            import re

            adapted = re.sub(r"\bvous\b", "tu", adapted, flags=re.IGNORECASE)

        if enthusiasm > 0.7 and not adapted.endswith("!"):
            adapted += "!"

        SMILING_EMOJI = "\U0001f603"
        if humor > 0.7 and SMILING_EMOJI not in adapted:
            adapted += f" {SMILING_EMOJI}"

        return adapted
