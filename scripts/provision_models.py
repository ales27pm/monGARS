"""Utility script for provisioning configured LLM models locally."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from pydantic import ValidationError

from monGARS.api.schemas import LLMModelProvisionRequest
from monGARS.config import get_settings
from monGARS.core.model_manager import LLMModelManager

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensure configured LLM models are available locally."
    )
    parser.add_argument(
        "--roles",
        nargs="*",
        help="Specific model roles to provision (default: all roles in the active profile).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-provisioning even if models were previously ensured.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit provisioning results as JSON instead of human readable text.",
    )
    return parser.parse_args()


async def _provision_models(args: argparse.Namespace) -> int:
    settings = get_settings()
    manager = LLMModelManager(settings)
    try:
        request = LLMModelProvisionRequest(roles=args.roles, force=args.force)
    except ValidationError as exc:
        logger.error(
            "scripts.models.provision.invalid_roles", extra={"error": exc.errors()}
        )
        print("Invalid roles provided; see logs for details.")
        return 1
    roles = request.roles
    report = await manager.ensure_models_installed(roles, force=request.force)
    if args.as_json:
        print(json.dumps(report.to_payload(), indent=2))
    else:
        if not report.statuses:
            print("No models were provisioned.")
        for status in report.statuses:
            detail_suffix = f" ({status.detail})" if status.detail else ""
            print(
                f"[{status.provider}] {status.role}: {status.action} -> {status.name}{detail_suffix}"
            )
    logger.info(
        "scripts.models.provision",
        extra={"roles": roles or "all", "actions": report.actions_by_role()},
    )
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    try:
        return asyncio.run(_provision_models(args))
    except KeyboardInterrupt:  # pragma: no cover - CLI convenience
        return 130


if __name__ == "__main__":  # pragma: no branch - CLI entry point
    raise SystemExit(main())
