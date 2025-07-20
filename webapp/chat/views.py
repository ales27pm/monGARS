import logging
import os

import httpx
from django.shortcuts import render

logger = logging.getLogger(__name__)


async def index(request):
    fastapi_url = os.environ.get("FASTAPI_URL", "http://localhost:8000")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{fastapi_url}/api/v1/conversation/history?user_id=testuser"
            )
            if response.status_code == 200:
                data = await response.json()
            else:
                data = {
                    "error": f"Erreur lors de la récupération de l'historique: {response.status_code}"
                }
    except Exception as e:  # pragma: no cover - network failure
        logger.error("Erreur de connexion: %s", e)
        data = {"error": f"Impossible de se connecter au serveur FastAPI: {e}"}
    return render(
        request, "chat/index.html", {"data": data, "fastapi_url": fastapi_url}
    )
