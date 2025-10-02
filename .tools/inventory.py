#!/usr/bin/env python3
"""Generate a repository inventory for drift detection."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from fastapi import FastAPI
from fastapi.routing import APIRoute, APIWebSocketRoute

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "inventory.json"
EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTEST_CURRENT_TEST", "inventory")
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("SECRET_KEY", "inventory-placeholder")


@dataclass
class SourceEntry:
    path: Path
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.as_posix(),
            "sha256": self.sha256,
        }


class EnvKeyCollector(ast.NodeVisitor):
    """Collect environment/config keys referenced in an AST."""

    def __init__(self) -> None:
        self.keys: set[str] = set()

    def visit_Call(self, node: ast.Call) -> Any:  # type: ignore[override]
        func = node.func
        if isinstance(func, ast.Attribute):
            if self._is_os_attribute(func, target="getenv"):
                key = self._extract_literal(node.args, node.keywords)
                if key:
                    self.keys.add(key)
            elif self._is_os_environ_attribute(func, target="get"):
                key = self._extract_literal(node.args, node.keywords)
                if key:
                    self.keys.add(key)
        elif isinstance(func, ast.Name) and func.id in {"Field", "field"}:
            for keyword in node.keywords:
                if keyword.arg == "env":
                    value = self._resolve_literal(keyword.value)
                    if value:
                        self.keys.add(value)
        return self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> Any:  # type: ignore[override]
        if self._is_os_environ(node.value):
            key = self._resolve_literal(node.slice)
            if key:
                self.keys.add(key)
        return self.generic_visit(node)

    @staticmethod
    def _extract_literal(
        args: Iterable[ast.expr], keywords: Iterable[ast.keyword]
    ) -> str | None:
        if args:
            first = next(iter(args))
            literal = EnvKeyCollector._resolve_literal(first)
            if literal:
                return literal
        for keyword in keywords:
            if keyword.arg == "key":
                literal = EnvKeyCollector._resolve_literal(keyword.value)
                if literal:
                    return literal
        return None

    @staticmethod
    def _resolve_literal(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                else:
                    return None
            return "".join(parts)
        return None

    @staticmethod
    def _is_os_attribute(node: ast.Attribute, *, target: str) -> bool:
        return (
            isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr == target
        )

    @staticmethod
    def _is_os_environ_attribute(node: ast.Attribute, *, target: str) -> bool:
        return EnvKeyCollector._is_os_environ(node.value) and node.attr == target

    @staticmethod
    def _is_os_environ(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr == "environ"
        )


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )


def collect_packages() -> list[dict[str, str]]:
    result = run([sys.executable, "-m", "pip", "freeze", "--disable-pip-version-check"])
    packages: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name: str
        version: str
        if "==" in stripped:
            name, version = stripped.split("==", 1)
        elif " @ " in stripped:
            name, version = stripped.split(" @ ", 1)
        else:
            name, version = stripped, ""
        packages.append({"name": name, "version": version})
    return sorted(packages, key=lambda item: item["name"].lower())


def iter_python_files() -> Iterator[Path]:
    for path in ROOT.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        yield path


def collect_source_tree() -> list[dict[str, Any]]:
    entries: list[SourceEntry] = []
    for path in iter_python_files():
        relative = path.relative_to(ROOT)
        try:
            data = path.read_bytes()
        except OSError:
            continue
        sha = hashlib.sha256(data).hexdigest()
        entries.append(SourceEntry(path=relative, sha256=sha))
    return [
        entry.to_dict()
        for entry in sorted(entries, key=lambda entry: entry.path.as_posix())
    ]


def load_fastapi_app() -> FastAPI:
    try:
        from monGARS.api.web_api import app
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError("Failed to import FastAPI application") from exc
    if not isinstance(app, FastAPI):
        raise RuntimeError("monGARS.api.web_api.app is not a FastAPI instance")
    return app


def collect_routes(app: FastAPI) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    routes: list[dict[str, Any]] = []
    websockets: list[dict[str, Any]] = []
    for route in app.routes:
        if isinstance(route, APIRoute):
            methods = sorted(
                m for m in (route.methods or set()) if m not in {"HEAD", "OPTIONS"}
            )
            routes.append(
                {
                    "path": route.path,
                    "name": route.name,
                    "methods": methods,
                }
            )
        elif isinstance(route, APIWebSocketRoute):
            websockets.append(
                {
                    "path": route.path,
                    "name": route.name,
                }
            )
    routes.sort(key=lambda item: (item["path"], "-".join(item["methods"])))
    websockets.sort(key=lambda item: item["path"])
    return routes, websockets


def collect_env_keys() -> list[str]:
    collector = EnvKeyCollector()
    for path in iter_python_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        collector.visit(tree)
    return sorted(collector.keys)


def validate_sections(sections: dict[str, Any]) -> None:
    missing = [name for name, value in sections.items() if not value]
    if missing:
        raise SystemExit(
            "Inventory generation failed; empty sections: " + ", ".join(missing)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a repository inventory")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="Path to write the inventory JSON file",
    )
    return parser.parse_args()


def resolve_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def main() -> None:
    args = parse_args()
    output_path = resolve_output_path(args.output)

    packages = collect_packages()
    source_tree = collect_source_tree()
    app = load_fastapi_app()
    fastapi_routes, websockets = collect_routes(app)
    config_keys = collect_env_keys()

    inventory = {
        "python_packages": packages,
        "source_tree": source_tree,
        "fastapi_routes": fastapi_routes,
        "websocket_endpoints": websockets,
        "config_keys": config_keys,
    }

    validate_sections(inventory)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
