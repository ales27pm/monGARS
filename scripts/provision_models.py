"""Utility script for provisioning configured LLM models locally."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any

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
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Also curate reasoning datasets and warm the GRPO slot for alignment runs.",
    )
    parser.add_argument(
        "--reasoning-samples",
        type=int,
        default=200,
        help=(
            "Number of mixed internal/external reasoning samples to curate when --reasoning is set. "
            "Defaults to 200."
        ),
    )
    parser.add_argument(
        "--reasoning-internal-ratio",
        type=float,
        default=0.5,
        help=(
            "Fraction of reasoning samples drawn from internal history logs when --reasoning is set. "
            "Value is clamped to the [0.0, 1.0] interval. Defaults to 0.5."
        ),
    )
    parser.add_argument(
        "--reasoning-slot",
        default="reasoning-grpo",
        help="Model slot name to warm for GRPO reasoning runs (default: reasoning-grpo).",
    )
    parser.add_argument(
        "--reasoning-model-id",
        default="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        help=(
            "Model identifier to load when preparing the reasoning slot (default: unsloth/Meta-Llama-3.1-8B-bnb-4bit)."
        ),
    )
    parser.add_argument(
        "--reasoning-max-seq",
        type=int,
        default=2048,
        help="Maximum sequence length for the warmed reasoning slot (default: 2048).",
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
    reasoning_summary: dict[str, Any] | None = None
    if args.reasoning:
        reasoning_summary = await _prepare_reasoning_assets(args)

    if args.as_json:
        payload = report.to_payload()
        if reasoning_summary is not None:
            payload["reasoning"] = reasoning_summary
        print(json.dumps(payload, indent=2))
    else:
        if not report.statuses:
            print("No models were provisioned.")
        for status in report.statuses:
            detail_suffix = f" ({status.detail})" if status.detail else ""
            print(
                f"[{status.provider}] {status.role}: {status.action} -> {status.name}{detail_suffix}"
            )

        if reasoning_summary is not None:
            _emit_reasoning_summary(reasoning_summary)
    logger.info(
        "scripts.models.provision",
        extra={"roles": roles or "all", "actions": report.actions_by_role()},
    )
    return 0


async def _prepare_reasoning_assets(args: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    try:  # Import lazily so the script can run without optional dependencies.
        from monGARS.core.self_training import SelfTrainingEngine
    except Exception as exc:  # pragma: no cover - optional dependency missing
        logger.warning(
            "scripts.models.reasoning.engine_unavailable",
            extra={"error": str(exc)},
        )
        summary["dataset"] = {"status": "unavailable", "error": str(exc)}
    else:
        engine = SelfTrainingEngine()
        num_samples = max(1, int(getattr(args, "reasoning_samples", 1)))
        ratio = float(getattr(args, "reasoning_internal_ratio", 0.5) or 0.0)
        ratio = max(0.0, min(1.0, ratio))

        def _curate() -> tuple[Any, Any]:
            return engine.curate_reasoning_dataset(
                num_samples=num_samples,
                internal_ratio=ratio,
            )

        try:
            train_ds, eval_ds = await asyncio.to_thread(_curate)
        except Exception as exc:  # pragma: no cover - dataset download issues
            logger.warning(
                "scripts.models.reasoning.dataset_failed",
                extra={"error": str(exc)},
            )
            summary["dataset"] = {"status": "failed", "error": str(exc)}
        else:
            train_len = _safe_length(train_ds)
            eval_len = _safe_length(eval_ds)
            summary["dataset"] = {
                "status": "ok",
                "train_samples": train_len,
                "eval_samples": eval_len,
            }
            logger.info(
                "scripts.models.reasoning.dataset_ready",
                extra={
                    "train_samples": train_len,
                    "eval_samples": eval_len,
                    "num_samples": num_samples,
                    "internal_ratio": ratio,
                },
            )

    try:  # Import lazily for environments without GPU dependencies.
        from monGARS.core.model_slot_manager import ModelSlotManager
    except Exception as exc:  # pragma: no cover - optional dependency missing
        logger.warning(
            "scripts.models.reasoning.slot_unavailable",
            extra={"error": str(exc)},
        )
        summary["slot"] = {"status": "unavailable", "error": str(exc)}
        return summary

    slot_name = getattr(args, "reasoning_slot", "reasoning-grpo")
    model_id = getattr(args, "reasoning_model_id", "unsloth/Meta-Llama-3.1-8B-bnb-4bit")
    max_seq = max(1, int(getattr(args, "reasoning_max_seq", 2048)))

    def _warm_slot() -> None:
        with ModelSlotManager(
            slot_name=slot_name,
            model_id=model_id,
            max_seq_length=max_seq,
        ):
            return None

    try:
        await asyncio.to_thread(_warm_slot)
    except Exception as exc:  # pragma: no cover - slot warm failures
        logger.warning(
            "scripts.models.reasoning.slot_failed",
            extra={
                "slot": slot_name,
                "model_id": model_id,
                "max_seq_length": max_seq,
                "error": str(exc),
            },
        )
        summary["slot"] = {
            "status": "failed",
            "slot": slot_name,
            "model_id": model_id,
            "max_seq_length": max_seq,
            "error": str(exc),
        }
    else:
        summary["slot"] = {
            "status": "ok",
            "slot": slot_name,
            "model_id": model_id,
            "max_seq_length": max_seq,
        }
        logger.info(
            "scripts.models.reasoning.slot_ready",
            extra={
                "slot": slot_name,
                "model_id": model_id,
                "max_seq_length": max_seq,
            },
        )

    return summary


def _emit_reasoning_summary(summary: dict[str, Any]) -> None:
    dataset = summary.get("dataset")
    slot = summary.get("slot")

    if isinstance(dataset, dict):
        status = dataset.get("status")
        if status == "ok":
            train_samples = dataset.get("train_samples")
            eval_samples = dataset.get("eval_samples")
            print(
                "Reasoning dataset curated: "
                f"train={train_samples if train_samples is not None else 'n/a'}, "
                f"eval={eval_samples if eval_samples is not None else 'n/a'}"
            )
        else:
            print(
                "Reasoning dataset unavailable: "
                f"{dataset.get('error', 'unknown error')}"
            )

    if isinstance(slot, dict):
        status = slot.get("status")
        if status == "ok":
            print(
                "Reasoning slot warmed: "
                f"slot={slot.get('slot')} model={slot.get('model_id')} "
                f"max_seq={slot.get('max_seq_length')}"
            )
        else:
            print(
                "Reasoning slot preparation failed: "
                f"{slot.get('error', 'unknown error')}"
            )


def _safe_length(dataset: Any) -> int | None:
    try:
        return int(len(dataset))  # type: ignore[arg-type]
    except Exception:
        return None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    try:
        return asyncio.run(_provision_models(args))
    except KeyboardInterrupt:  # pragma: no cover - CLI convenience
        return 130


if __name__ == "__main__":  # pragma: no branch - CLI entry point
    raise SystemExit(main())
