import asyncio
import logging

import aiohttp

from monGARS.config import get_settings
from monGARS.core.security import decrypt_token

logger = logging.getLogger(__name__)


class SocialMediaManager:
    """Simple interface for posting content to social platforms."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def post_to_twitter(self, content: str, encrypted_token: str) -> bool:
        """Post a tweet using an encrypted bearer token."""
        try:
            access_token = decrypt_token(encrypted_token)
        except ValueError as exc:
            logger.error("Token decryption failed: %s", exc)
            return False

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {access_token}"}
                async with session.post(
                    "https://api.twitter.com/2/tweets",
                    json={"text": content},
                    headers=headers,
                ) as response:
                    if response.status != 201:
                        logger.error("Twitter API returned status %s", response.status)
                        return False
                    return True
        except asyncio.TimeoutError:
            logger.error("Twitter request timed out")
            return False
        except aiohttp.ClientError as exc:
            logger.error("Twitter request failed: %s", exc)
            return False

    async def analyze_social_sentiment(self, post: str) -> dict:
        """Placeholder sentiment analysis for a social post."""
        score = 0
        positive_words = {"great", "love", "good", "happy"}
        negative_words = {"bad", "hate", "terrible", "sad"}
        for word in post.lower().split():
            if word in positive_words:
                score += 1
            if word in negative_words:
                score -= 1
        sentiment = "neutral"
        if score > 0:
            sentiment = "positive"
        elif score < 0:
            sentiment = "negative"
        return {"sentiment": sentiment, "score": score}
