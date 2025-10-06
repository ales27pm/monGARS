"""Command-line entrypoint for CUDA and Unsloth diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from .analysis import analyse_cuda_state
from .cuda_metrics import gather_cuda_metrics
from .environment import configure_logging, gather_environment, import_optional

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether Unsloth is available and inspect CUDA memory headroom.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run the Unsloth patch even if it was previously initialised.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index to inspect (default: 0).",
    )
    parser.add_argument(
        "--all-devices",
        action="store_true",
        help="Collect diagnostics for every visible CUDA device instead of a single index.",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Skip CUDA diagnostics to avoid touching GPU state.",
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=1.0,
        help="Minimum free VRAM (GiB) required before flagging OOM risk (default: 1.0).",
    )
    parser.add_argument(
        "--min-free-ratio",
        type=float,
        default=0.1,
        help="Minimum free/total VRAM ratio required before flagging OOM risk (default: 0.1).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output for troubleshooting.",
    )

    args = parser.parse_args(argv)

    if args.min_free_gib <= 0:
        parser.error("--min-free-gib must be a positive value.")
    if args.min_free_ratio <= 0:
        parser.error("--min-free-ratio must be a positive value.")

    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging(args.verbose)

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    os.environ.setdefault("DEBUG", "true")
    os.environ.setdefault("SECRET_KEY", "diagnostics-only")

    try:
        from monGARS.core.llm_integration import initialize_unsloth
    except Exception:  # pragma: no cover - defensive guardrail
        logger.exception("unable to import Unsloth integration helper")
        return 2

    try:
        unsloth_state = initialize_unsloth(force=args.force)
    except Exception:  # pragma: no cover - defensive guardrail
        logger.exception("initialising Unsloth failed")
        return 3

    torch_module = import_optional("torch")

    payload: dict[str, Any] = {
        "environment": gather_environment(torch_module),
        "unsloth": {**unsloth_state},
    }

    skip_reason: str | None = None
    if args.no_cuda:
        payload["cuda"] = None
        skip_reason = "cuda_diagnostics_disabled"
    elif torch_module is None:
        payload["cuda"] = None
        skip_reason = "torch_missing"
    else:
        selector = (
            (lambda: list(range(torch_module.cuda.device_count())))
            if args.all_devices
            else (lambda: [args.device])
        )
        payload["cuda"] = gather_cuda_metrics(torch_module, selector)

    payload.setdefault("analysis", {})["oom_risk"] = analyse_cuda_state(
        payload.get("cuda"),
        min_free_gib=args.min_free_gib,
        min_free_ratio=args.min_free_ratio,
        skip_reason=skip_reason,
    )

    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


__all__ = ["main"]
