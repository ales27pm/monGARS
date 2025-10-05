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
    """
    Builds and parses the command-line arguments for the model provisioning script.
    
    Returns:
        args (argparse.Namespace): Parsed CLI arguments with the following attributes:
            roles (list[str] | None): Specific model roles to provision; None means use all roles in the active profile.
            force (bool): If true, force re-provisioning even if models are already ensured.
            as_json (bool): If true, emit provisioning results as JSON.
            reasoning (bool): If true, curate reasoning datasets and warm the GRPO slot.
            reasoning_samples (int): Number of reasoning samples to curate when `reasoning` is true (default 200; coerced to at least 1 by callers).
            reasoning_internal_ratio (float): Fraction of samples drawn from internal history when `reasoning` is true (clamped to [0.0, 1.0]; default 0.5).
            reasoning_slot (str): Model slot name to warm for reasoning runs (default "reasoning-grpo").
            reasoning_model_id (str): Model identifier to load when preparing the reasoning slot (default "unsloth/Meta-Llama-3.1-8B-bnb-4bit").
            reasoning_max_seq (int): Maximum sequence length for the warmed reasoning slot (default 2048).
    """
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
    """
    Provision configured LLM models according to CLI arguments and emit a summary.
    
    Parameters:
        args (argparse.Namespace): Parsed CLI arguments. Expected attributes:
            - roles: Optional[list[str]] of model roles to provision; when empty or None, all configured roles are used.
            - force: bool indicating whether to forcibly re-provision models.
            - as_json: bool; when true, emits a JSON payload instead of human-readable lines.
            - reasoning: bool; when true, prepares reasoning assets and includes a reasoning summary in output.
            - reasoning_*: additional reasoning-related fields used when `reasoning` is true (e.g., reasoning_samples, reasoning_internal_ratio, reasoning_slot, reasoning_model_id, reasoning_max_seq).
    
    Returns:
        int: Exit code where `0` indicates success and `1` indicates invalid role input was provided.
    """
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
    """
    Prepare an optional reasoning dataset and warm a model slot, returning a summary of dataset and slot statuses.
    
    Parameters:
        args (argparse.Namespace): CLI arguments; recognized attributes:
            reasoning_samples (int): Number of reasoning samples to curate (coerced to at least 1).
            reasoning_internal_ratio (float): Fraction of internal reasoning samples, clamped to [0.0, 1.0].
            reasoning_slot (str): Slot name to warm (default "reasoning-grpo").
            reasoning_model_id (str): Model identifier to load into the slot.
            reasoning_max_seq (int): Maximum sequence length for the warmed slot (coerced to at least 1).
    
    Returns:
        summary (dict[str, Any]): A dictionary containing:
            - "dataset": dict with keys:
                - "status": one of "ok", "failed", or "unavailable".
                - On "ok": "train_samples" (int|None) and "eval_samples" (int|None).
                - On failure/unavailable: "error" (str).
            - "slot": dict with keys:
                - "status": one of "ok", "failed", or "unavailable".
                - On "ok": "slot" (str), "model_id" (str), "max_seq_length" (int).
                - On failure/unavailable: "error" (str) and, if failed, "slot", "model_id", "max_seq_length".
    """
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
            """
            Invoke the SelfTrainingEngine to curate a reasoning dataset.
            
            Returns:
                train_ds, eval_ds: The curated training and evaluation datasets returned by the engine.
            """
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
    model_id = getattr(
        args, "reasoning_model_id", "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    )
    max_seq = max(1, int(getattr(args, "reasoning_max_seq", 2048)))

    def _warm_slot() -> None:
        """
        Warm the configured model slot by instantiating a ModelSlotManager context.
        
        This triggers any slot preparation or resource allocation associated with the configured
        slot_name, model_id, and max_seq_length.
        """
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
    """
    Print a human-readable summary of reasoning dataset curation and slot warming results.
    
    Parameters:
    	summary (dict[str, Any]): A summary map that may contain:
    		- "dataset": dict with keys:
    			- "status": either "ok" or an error status.
    			- "train_samples" (optional): number of training samples or None.
    			- "eval_samples" (optional): number of evaluation samples or None.
    			- "error" (optional): error message when status is not "ok".
    		- "slot": dict with keys:
    			- "status": either "ok" or an error status.
    			- "slot" (optional): slot name when status is "ok".
    			- "model_id" (optional): warmed model identifier when status is "ok".
    			- "max_seq_length" (optional): max sequence length when status is "ok".
    			- "error" (optional): error message when status is not "ok".
    
    Description:
    	Prints lines describing the dataset curation results (train/eval counts or an error)
    	and the slot warming results (slot/model/max sequence length or an error) when present.
    """
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
    """
    Return the length of a dataset if it can be determined.
    
    Attempts to obtain and coerce the dataset's length to an int; if the length cannot be determined or an error occurs, returns `None`.
    
    Parameters:
        dataset (Any): Object whose length is to be measured.
    
    Returns:
        int | None: The length as an `int` if determinable, `None` otherwise.
    """
    try:
        return int(len(dataset))  # type: ignore[arg-type]
    except Exception:
        return None


def main() -> int:
    """
    Run the provisioning command-line flow and return a process exit code.
    
    Invokes argument parsing, executes the asynchronous provisioning routine, and maps a keyboard interrupt to a standard exit code.
    
    Returns:
        int: Process exit code — `0` on success, `1` for invalid roles (validation failure from provisioning), `130` if interrupted by the user, or other codes returned by the provisioning routine.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    try:
        return asyncio.run(_provision_models(args))
    except KeyboardInterrupt:  # pragma: no cover - CLI convenience
        return 130


if __name__ == "__main__":  # pragma: no branch - CLI entry point
    raise SystemExit(main())