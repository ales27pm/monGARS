"""Administrative endpoints for managing local LLM model provisioning."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from httpx import HTTPError

from monGARS.api.authentication import get_current_admin_user
from monGARS.api.dependencies import get_model_manager
from monGARS.api.schemas import (
    LLMModelConfigurationResponse,
    LLMModelProvisionReportResponse,
    LLMModelProvisionRequest,
)
from monGARS.core.model_manager import LLMModelManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("", response_model=LLMModelConfigurationResponse)
async def read_model_configuration(
    _admin: dict = Depends(get_current_admin_user),
    manager: LLMModelManager = Depends(get_model_manager),
) -> LLMModelConfigurationResponse:
    """Return the active profile and available model definitions."""

    try:
        profile = manager.get_profile_snapshot()
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    response = LLMModelConfigurationResponse.from_profile(
        active_profile=manager.active_profile_name(),
        available_profiles=manager.available_profile_names(),
        profile=profile,
    )
    logger.debug(
        "api.models.profile", extra={"active_profile": response.active_profile}
    )
    return response


@router.post("/provision", response_model=LLMModelProvisionReportResponse)
async def provision_models(
    payload: LLMModelProvisionRequest,
    _admin: dict = Depends(get_current_admin_user),
    manager: LLMModelManager = Depends(get_model_manager),
) -> LLMModelProvisionReportResponse:
    """Ensure local providers have the configured models for the requested roles."""

    roles = payload.roles
    try:
        report = await manager.ensure_models_installed(roles, force=payload.force)
    except (
        HTTPError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:  # pragma: no cover - provider failure bubble-up
        logger.exception(
            "api.models.provision.failed",
            extra={"roles": roles},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to provision requested models.",
        ) from exc

    logger.info(
        "api.models.provision",
        extra={"roles": roles or "all", "actions": report.actions_by_role()},
    )
    return LLMModelProvisionReportResponse.from_report(report)
