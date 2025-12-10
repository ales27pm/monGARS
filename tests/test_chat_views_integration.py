import os
import sys
from pathlib import Path
from types import SimpleNamespace

import django
import pytest
from django.http import HttpResponse
from django.urls import reverse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(PROJECT_ROOT / "webapp"), str(PROJECT_ROOT)])
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webapp.settings")
django.setup()

from chat import views  # noqa: E402  pylint: disable=wrong-import-position


def _make_request(method: str = "GET", data: dict | None = None) -> SimpleNamespace:
    payload = data or {}
    return SimpleNamespace(method=method, POST=payload, session={}, token=None, user_id=None)


@pytest.mark.asyncio
async def test_index_redirects_without_session():
    request = _make_request()
    response = await views.index(request)
    assert response.status_code == 302
    assert response.url == reverse("login")


@pytest.mark.asyncio
async def test_index_renders_history_and_handles_post(monkeypatch: pytest.MonkeyPatch):
    request = _make_request("POST", {"prompt": "hello"})
    request.session.update({"token": "tkn", "user_id": "alice"})

    history_payload = [{"text": "hi"}, {"text": "there"}]
    captured_context: dict | None = None

    async def fake_post_chat_message(user_id, token, prompt):  # noqa: ANN001
        return {"ok": True, "message": prompt, "user_id": user_id, "token": token}

    async def fake_fetch_history(user_id, token):  # noqa: ANN001
        assert user_id == "alice"
        assert token == "tkn"
        return history_payload

    def fake_render(request, template, context):  # noqa: ANN001
        nonlocal captured_context
        captured_context = context
        return HttpResponse("rendered", status=200)

    monkeypatch.setattr(views, "post_chat_message", fake_post_chat_message)
    monkeypatch.setattr(views, "fetch_history", fake_fetch_history)
    monkeypatch.setattr(views, "render", fake_render)

    response = await views.index(request)
    assert response.status_code == 302
    assert captured_context is None  # redirect path skips render


@pytest.mark.asyncio
async def test_index_validation_error(monkeypatch: pytest.MonkeyPatch):
    request = _make_request("POST", {"prompt": "  "})
    request.session.update({"token": "token", "user_id": "user"})

    def fake_render(request, template, context):  # noqa: ANN001
        return HttpResponse(context["form_error"], status=200)

    monkeypatch.setattr(views, "render", fake_render)

    response = await views.index(request)
    assert response.status_code == 200
    assert "peut pas être vide" in response.content.decode()


@pytest.mark.asyncio
async def test_login_view_auth_flow(monkeypatch: pytest.MonkeyPatch):
    request = _make_request("POST", {"username": "alice", "password": "password1"})

    def fake_render(request, template, context=None):  # noqa: ANN001
        return HttpResponse(context["error"] if context else "ok", status=200)

    async def fake_authenticate(username, password):  # noqa: ANN001
        return f"token-for-{username}-{password}"

    monkeypatch.setattr(views, "render", fake_render)
    monkeypatch.setattr(views, "authenticate_user", fake_authenticate)

    response = await views.login_view(request)
    assert response.status_code == 302
    assert request.session["token"].startswith("token-for-alice")
    assert request.session["user_id"] == "alice"

    invalid_request = _make_request("POST", {"username": "bad user", "password": "pw"})
    monkeypatch.setattr(views, "render", lambda r, t, c=None: HttpResponse(c["error"], status=200))  # noqa: E731
    invalid_response = await views.login_view(invalid_request)
    assert invalid_response.status_code == 200
    assert "Nom d'utilisateur invalide" in invalid_response.content.decode()
