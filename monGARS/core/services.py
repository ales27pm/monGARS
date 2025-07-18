from monGARS.core.bouche import Bouche
from monGARS.core.hippocampus import Hippocampus


class MemoryService:
    def __init__(self, hippocampus: Hippocampus):
        self._hippocampus = hippocampus

    async def store(self, user_id: str, query: str, response: str):
        await self._hippocampus.store(user_id, query, response)

    async def history(self, user_id: str, limit: int = 10):
        return await self._hippocampus.history(user_id, limit)


class SpeakerService:
    def __init__(self, bouche: Bouche):
        self._bouche = bouche

    async def speak(self, text: str) -> str:
        return await self._bouche.speak(text)
