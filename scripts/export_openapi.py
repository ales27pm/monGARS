"""Export the FastAPI OpenAPI schema to a JSON file."""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

from fastapi.encoders import jsonable_encoder

os.environ.setdefault("SECRET_KEY", "export-secret")
os.environ.setdefault("JWT_ALGORITHM", "HS256")


def _install_kubernetes_stub() -> None:
    exceptions = types.SimpleNamespace(ApiException=Exception)
    client = types.SimpleNamespace(AppsV1Api=object, exceptions=exceptions)
    config = types.SimpleNamespace(
        load_kube_config=lambda *args, **kwargs: None,
        load_incluster_config=lambda *args, **kwargs: None,
    )
    module = types.ModuleType("kubernetes")
    module.client = client
    module.config = config
    sys.modules["kubernetes"] = module
    sys.modules["kubernetes.client"] = client
    sys.modules["kubernetes.config"] = config


def _ensure_lightweight_kubernetes() -> None:
    force_stub = os.environ.get("MON_GARS_FORCE_KUBE_STUB", "1") == "1"
    if not force_stub:
        try:
            import kubernetes  # type: ignore  # noqa: F401

            return
        except Exception:
            pass
    if "kubernetes" not in sys.modules:
        _install_kubernetes_stub()


def _generate_schema() -> dict[str, Any]:
    _ensure_lightweight_kubernetes()
    from monGARS.api.web_api import app

    return jsonable_encoder(app.openapi())


def _write_schema(schema: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatted = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    path.write_text(formatted)
    return path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Export the monGARS OpenAPI schema.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("openapi.json"),
        help="Path where openapi.json will be written.",
    )
    parser.add_argument(
        "--update-lock",
        action="store_true",
        help="Also update openapi.lock.json alongside the primary output.",
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=Path("openapi.lock.json"),
        help="Location of the lock file used for contract testing.",
    )
    args = parser.parse_args(argv)

    schema = _generate_schema()
    output_path = _write_schema(schema, args.output)

    if args.update_lock:
        _write_schema(schema, args.lock_path)

    return output_path


if __name__ == "__main__":
    main()
