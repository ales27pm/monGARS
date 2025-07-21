import logging
import os

import httpx
from django.shortcuts import redirect, render

logger = logging.getLogger(__name__)


async def index(request):
    fastapi_url = os.environ.get("FASTAPI_URL", "http://localhost:8000")
    token = request.session.get("token")
    user_id = request.session.get("user_id")
    if not token or not user_id:
        return redirect("login")

    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{fastapi_url}/api/v1/conversation/history?user_id={user_id}",
                headers=headers,
            )
            if response.status_code == 200:
                data = response.json()
            else:
                data = {
                    "error": f"Erreur lors de la récupération de l'historique: {response.status_code}"
                }
    except Exception as e:  # pragma: no cover - network failure
        logger.error("Erreur de connexion: %s", e)
        data = {"error": f"Impossible de se connecter au serveur FastAPI: {e}"}
    return render(
        request,
        "chat/index.html",
        {
            "data": data,
            "fastapi_url": fastapi_url,
            "user_id": user_id,
            "token": token,
        },
    )


async def login_view(request):
    fastapi_url = os.environ.get("FASTAPI_URL", "http://localhost:8000")
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        if not username or not password:
            return render(request, "chat/login.html", {"error": "Credentials required"})
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{fastapi_url}/token",
                    data={"username": username, "password": password},
                )
            if resp.status_code == 200:
                token = resp.json().get("access_token")
                request.session["token"] = token
                request.session["user_id"] = username
                return redirect("index")
            error = "Invalid credentials"
        except Exception as exc:  # pragma: no cover - network failure
            logger.error("Login failed: %s", exc)
            error = f"Connexion impossible: {exc}"
        return render(request, "chat/login.html", {"error": error})

    return render(request, "chat/login.html")
