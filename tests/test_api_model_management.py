"""Tests covering the model management API surface."""

from __future__ import annotations

import os
from typing import Any

import pytest
from fastapi.testclient import TestClient

from monGARS.api.dependencies import get_model_manager
from monGARS.api.web_api import app
from monGARS.core.model_manager import (
    ModelDefinition,
    ModelProfile,
    ModelProvisionReport,
    ModelProvisionStatus,
)

os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("SECRET_KEY", "test-secret")


pytestmark = pytest.mark.usefixtures("ensure_test_users")


class FakeModelManager:
    def __init__(self) -> None:
        self._profile = ModelProfile(
            name="default",
            models={
                "general": ModelDefinition(
                    role="general",
                    name="fake/general",
                    provider="ollama",
                    parameters={"temperature": 0.2},
                    description="General test model",
                ),
                "coding": ModelDefinition(
                    role="coding",
                    name="fake/coder",
                    provider="ollama",
                    auto_download=False,
                ),
            },
        )
        self._available = ["default", "research"]
        self.calls: list[dict[str, Any]] = []

    def get_profile_snapshot(self, name: str | None = None) -> ModelProfile:
        if name and name.lower() != self._profile.name:
            raise KeyError(name)
        return ModelProfile(name=self._profile.name, models=dict(self._profile.models))

    def available_profile_names(self) -> list[str]:
        return list(self._available)

    def active_profile_name(self) -> str:
        return self._profile.name

    async def ensure_models_installed(
        self, roles: list[str] | None = None, *, force: bool = False
    ) -> ModelProvisionReport:
        self.calls.append({"roles": roles, "force": force})
        return ModelProvisionReport(
            statuses=[
                ModelProvisionStatus(
                    role="general",
                    name="fake/general",
                    provider="ollama",
                    action="exists",
                ),
                ModelProvisionStatus(
                    role="coding",
                    name="fake/coder",
                    provider="ollama",
                    action="skipped",
                    detail="auto_download_disabled",
                ),
            ]
        )


@pytest.fixture
def fake_model_manager() -> FakeModelManager:
    return FakeModelManager()


@pytest.fixture
def client(fake_model_manager: FakeModelManager):
    app.dependency_overrides[get_model_manager] = lambda: fake_model_manager
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.pop(get_model_manager, None)


def _get_token(client: TestClient, username: str, password: str) -> str:
    response = client.post("/token", data={"username": username, "password": password})
    assert response.status_code == 200
    return response.json()["access_token"]


@pytest.mark.asyncio
async def test_model_configuration_requires_admin(client: TestClient):
    token = _get_token(client, "u2", "y")
    response = client.get(
        "/api/v1/models",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_model_configuration_returns_active_profile(
    client: TestClient, fake_model_manager: FakeModelManager
):
    token = _get_token(client, "u1", "x")
    response = client.get(
        "/api/v1/models",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["active_profile"] == fake_model_manager.active_profile_name()
    assert data["available_profiles"] == fake_model_manager.available_profile_names()
    general = data["profile"]["models"]["general"]
    assert general["name"] == "fake/general"
    assert general["parameters"]["temperature"] == 0.2
    coding = data["profile"]["models"]["coding"]
    assert coding["auto_download"] is False


@pytest.mark.asyncio
async def test_model_provision_invokes_manager(
    client: TestClient, fake_model_manager: FakeModelManager
):
    token = _get_token(client, "u1", "x")
    response = client.post(
        "/api/v1/models/provision",
        json={"roles": ["GENERAL", "coding"], "force": True},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["statuses"][0]["action"] == "exists"
    assert payload["statuses"][1]["detail"] == "auto_download_disabled"
    assert fake_model_manager.calls == [{"roles": ["general", "coding"], "force": True}]
