import logging
import os
import re

from django.shortcuts import redirect, render

from .decorators import require_token
from .services import authenticate_user, fetch_history

logger = logging.getLogger(__name__)


@require_token
async def index(request):
    """Show conversation history for the logged-in user."""
    data = await fetch_history(request.user_id, request.token)
    fastapi_url = os.environ.get("FASTAPI_URL", "http://localhost:8000")
    return render(
        request,
        "chat/index.html",
        {
            "data": data,
            "fastapi_url": fastapi_url,
            "user_id": request.user_id,
            "token": request.token,
        },
    )


async def login_view(request):
    fastapi_url = os.environ.get("FASTAPI_URL", "http://localhost:8000")
    debug = os.environ.get("DJANGO_DEBUG", "False").lower() in ("true", "1")
    if (
        not debug
        and "localhost" not in fastapi_url
        and not fastapi_url.startswith("https://")
    ):
        logger.warning("Insecure FASTAPI_URL: %s", fastapi_url)

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        if not username or not password:
            return render(request, "chat/login.html", {"error": "Credentials required"})
        if not 1 <= len(username) <= 150 or not re.fullmatch(r"[\w.@+-]+", username):
            return render(
                request, "chat/login.html", {"error": "Nom d'utilisateur invalide"}
            )
        if len(password) < 8:
            return render(
                request, "chat/login.html", {"error": "Mot de passe trop court"}
            )

        token = await authenticate_user(username, password)
        if token:
            # downstream modules treat user_id as an arbitrary string
            request.session["token"] = token
            request.session["user_id"] = username
            return redirect("index")
        error = "Connexion impossible. Veuillez réessayer plus tard."
        return render(request, "chat/login.html", {"error": error})

    return render(request, "chat/login.html")
