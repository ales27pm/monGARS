"""Utilities for building distributable monGARS SDK packages."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


class BuildError(RuntimeError):
    """Raised when an SDK packaging step fails."""


def _run_command(command: Sequence[str], *, cwd: Path) -> None:
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except FileNotFoundError as exc:  # pragma: no cover - depends on host env
        raise BuildError(
            f"Required command '{command[0]}' is not available on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        joined = " ".join(command)
        raise BuildError(
            f"Command '{joined}' failed with exit code {exc.returncode}."
        ) from exc


def build_python_sdk(repo_root: Path, *, output_dir: Path | None = None) -> Path:
    """Build the Python SDK wheel and sdist.

    Parameters
    ----------
    repo_root:
        Path pointing at the repository root. ``sdks/python`` is resolved from
        here.
    output_dir:
        Optional directory for output artefacts. When omitted the SDK's local
        ``dist`` directory is reused.
    """

    if importlib.util.find_spec("build") is None:  # pragma: no cover - env guard
        raise BuildError(
            "The 'build' package is required. Install it via 'pip install build'."
        )

    sdk_root = repo_root / "sdks" / "python"
    if not sdk_root.exists():
        raise BuildError(f"Python SDK directory not found: {sdk_root}")

    destination = output_dir or sdk_root / "dist"
    destination.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--sdist",
        "--outdir",
        str(destination),
    ]
    _run_command(command, cwd=sdk_root)
    return destination


def build_typescript_sdk(repo_root: Path, *, output_dir: Path | None = None) -> Path:
    """Build the TypeScript SDK and create an npm package tarball."""

    sdk_root = repo_root / "sdks" / "typescript"
    if not sdk_root.exists():
        raise BuildError(f"TypeScript SDK directory not found: {sdk_root}")

    destination = output_dir or sdk_root / "dist"
    destination.mkdir(parents=True, exist_ok=True)

    steps: Iterable[Sequence[str]] = (
        ("npm", "ci"),
        ("npm", "run", "build"),
        ("npm", "pack", "--pack-destination", str(destination)),
    )
    for step in steps:
        _run_command(list(step), cwd=sdk_root)

    return destination


def package_all(repo_root: Path, output_dir: Path | None = None) -> dict[str, Path]:
    """Build both SDKs and return their output directories."""

    if output_dir is not None:
        python_output_dir = output_dir / "python"
        typescript_output_dir = output_dir / "typescript"
    else:
        python_output_dir = None
        typescript_output_dir = None

    outputs: dict[str, Path] = {}
    python_output = build_python_sdk(repo_root, output_dir=python_output_dir)
    outputs["python"] = python_output

    ts_output = build_typescript_sdk(repo_root, output_dir=typescript_output_dir)
    outputs["typescript"] = ts_output

    return outputs


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package monGARS SDKs")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional directory where build artefacts are stored. "
            "Defaults to language-specific 'dist' folders."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    repo_root = Path(__file__).resolve().parents[1]
    try:
        outputs = package_all(repo_root, output_dir=args.output)
    except BuildError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1

    for name, path in outputs.items():
        sys.stdout.write(f"{name} artefacts written to {path}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI invocation
    raise SystemExit(main())
