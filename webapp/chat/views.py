import logging
import os
import re
from typing import Any

from django.shortcuts import redirect, render

from .decorators import require_token
from .services import authenticate_user, fetch_history, post_chat_message

logger = logging.getLogger(__name__)


@require_token
async def index(request):
    """Render the chat interface for the authenticated operator.

    Context:
        history (list[dict[str, Any]]): Chronological conversation history
            (oldest first). Empty when history retrieval fails.
        history_json (list[dict[str, Any]]): Raw history payload passed to the
            progressive-enhancement bootstrap script.
        history_error (str | None): Error message displayed when the history API
            is unavailable.
        flash_message (str | None): Ephemeral acknowledgement shown after a
            successful POST when JavaScript is disabled.
        form_error (str | None): Validation or transport error displayed above
            the composer.
        prompt_value (str): Current value shown in the composer input when a
            submission fails validation.
        fastapi_url (str): Base URL for the FastAPI service used by the
            JavaScript layer.
        user_id (str): Identifier for the authenticated operator.
        token (str): JWT used by the client-side enhancements.
    """

    flash_message = request.session.pop("chat_flash", None)
    form_error = None
    prompt_value = ""

    if request.method == "POST":
        prompt_value = request.POST.get("prompt", "").strip()
        if not prompt_value:
            form_error = "Le message ne peut pas être vide."
        else:
            result = await post_chat_message(
                request.user_id, request.token, prompt_value
            )
            error = result.get("error") if isinstance(result, dict) else None
            if error:
                form_error = error
            else:
                request.session["chat_flash"] = "Message envoyé avec succès."
                return redirect("index")

    history_raw: Any = await fetch_history(request.user_id, request.token)
    history_error: str | None = None
    history: list[dict[str, Any]] = []
    history_json: list[dict[str, Any]] = []
    if isinstance(history_raw, dict) and history_raw.get("error"):
        history_error = str(history_raw.get("error"))
    elif isinstance(history_raw, list):
        history_json = history_raw
        history = list(reversed(history_raw))
    else:
        history_error = "Historique indisponible pour le moment."

    fastapi_url = os.environ.get("FASTAPI_URL", "http://localhost:8000")
    return render(
        request,
        "chat/index.html",
        {
            "history": history,
            "history_json": history_json,
            "history_error": history_error,
            "flash_message": flash_message,
            "form_error": form_error,
            "prompt_value": prompt_value,
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
