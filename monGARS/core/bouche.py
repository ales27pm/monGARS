import logging

logger = logging.getLogger(__name__)


class Bouche:
    """Output interface for delivering responses."""

    async def speak(self, text: str) -> str:
        logger.info("Bouche delivers response: %s", text)
        return text
