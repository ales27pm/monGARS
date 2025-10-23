"""Deployment simulation and configuration validation tooling.

This module emulates a production deployment by loading configuration, parsing
container orchestration manifests, and surfacing actionable issues before an
operator pushes changes to staging or production. The simulator is deliberately
defensive: it favours deterministic checks over shelling out to Docker or
Kubernetes so it can run in CI and development environments without
containerisation support.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import yaml
from pydantic import ValidationError
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from monGARS.config import Settings, ensure_secret_key, validate_jwt_configuration

Severity = Literal["error", "warning", "info"]


@dataclass(slots=True)
class DeploymentIssue:
    """Represents a potential deployment problem discovered by the simulator."""

    severity: Severity
    message: str
    context: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "message": self.message,
            "context": self.context or {},
        }


@dataclass(slots=True)
class DeploymentSimulationReport:
    """Aggregates issues emitted by each simulation stage."""

    issues: list[DeploymentIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    @property
    def info_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "info")

    def extend(self, new_issues: Iterable[DeploymentIssue]) -> None:
        self.issues.extend(new_issues)

    def as_dict(self) -> dict[str, Any]:
        return {
            "issues": [issue.as_dict() for issue in self.issues],
            "errors": self.error_count,
            "warnings": self.warning_count,
            "infos": self.info_count,
        }


class DeploymentSimulator:
    """Run best-effort deployment checks against the repository."""

    def __init__(
        self,
        root: Path,
        compose_files: Sequence[Path] | None = None,
        kubernetes_manifests: Sequence[Path] | None = None,
    ) -> None:
        self.root = root
        self.compose_files = list(compose_files or [root / "docker-compose.yml"])
        self.kubernetes_manifests = list(
            kubernetes_manifests or [root / "k8s" / "deployment.yaml"]
        )

    # --- public API -----------------------------------------------------

    def run(self) -> DeploymentSimulationReport:
        report = DeploymentSimulationReport()

        report.extend(self._simulate_settings())
        report.extend(self._simulate_compose())
        report.extend(self._simulate_kubernetes())

        return report

    # --- helpers --------------------------------------------------------

    def _simulate_settings(self) -> list[DeploymentIssue]:
        issues: list[DeploymentIssue] = []

        env_file = self.root / ".env"
        env_file_arg = str(env_file) if env_file.exists() else None

        try:
            settings = Settings(_env_file=env_file_arg)
        except (
            ValidationError,
            ValueError,
            TypeError,
        ) as exc:  # pragma: no cover - configuration errors are surfaced
            issues.append(
                DeploymentIssue(
                    "error",
                    "Failed to load application settings.",
                    {"exception": repr(exc)},
                )
            )
            return issues

        origin = getattr(settings, "_secret_key_origin", "missing")
        if origin in {"missing", "deferred"}:
            issues.append(
                DeploymentIssue(
                    "error",
                    "SECRET_KEY is not configured for a production deployment.",
                    {"origin": origin},
                )
            )
        else:
            try:
                validate_jwt_configuration(settings)
            except ValueError as exc:
                issues.append(
                    DeploymentIssue(
                        "error",
                        "JWT configuration is invalid for the selected algorithm.",
                        {"exception": str(exc)},
                    )
                )

        # Detect deployments that rely on ephemeral secret generation.
        if origin in {"ephemeral", "generated"}:
            issues.append(
                DeploymentIssue(
                    "warning",
                    "Deployment will generate a SECRET_KEY at runtime; persist it before production rollout.",
                    {"origin": origin},
                )
            )

        if not settings.debug and origin not in {"missing", "deferred"}:
            try:
                ensure_secret_key(settings)
            except ValueError as exc:
                issues.append(
                    DeploymentIssue(
                        "error",
                        "Production deployments require a SECRET_KEY; generation failed.",
                        {"exception": str(exc)},
                    )
                )

        # Highlight local-only database defaults when running in production mode.
        try:
            url = make_url(str(settings.database_url))
        except ArgumentError as exc:
            issues.append(
                DeploymentIssue(
                    "error",
                    "DATABASE_URL is invalid and cannot be parsed by SQLAlchemy.",
                    {"exception": str(exc)},
                )
            )
        else:
            if (url.host or "").lower() in {"localhost", "127.0.0.1"}:
                issues.append(
                    DeploymentIssue(
                        "warning",
                        "DATABASE_URL points to localhost; confirm production networking and secrets are configured.",
                        {"database_url": str(settings.database_url)},
                    )
                )

        return issues

    def _simulate_compose(self) -> list[DeploymentIssue]:
        issues: list[DeploymentIssue] = []

        for path in self.compose_files:
            if not path.exists():
                issues.append(
                    DeploymentIssue(
                        "warning",
                        "Docker Compose file is missing; skipping simulation.",
                        {"path": str(path)},
                    )
                )
                continue

            try:
                data = yaml.safe_load(path.read_text()) or {}
            except yaml.YAMLError as exc:
                issues.append(
                    DeploymentIssue(
                        "error",
                        "Failed to parse Docker Compose file.",
                        {"path": str(path), "exception": str(exc)},
                    )
                )
                continue

            issues.extend(self._evaluate_compose_dict(data, path))

        return issues

    def _evaluate_compose_dict(
        self, compose: Any, origin: Path
    ) -> list[DeploymentIssue]:
        issues: list[DeploymentIssue] = []
        if not isinstance(compose, dict):
            issues.append(
                DeploymentIssue(
                    "error",
                    "Compose file must be a mapping at the top level.",
                    {"path": str(origin), "type": type(compose).__name__},
                )
            )
            return issues

        services = compose.get("services", {})

        if not isinstance(services, dict):
            issues.append(
                DeploymentIssue(
                    "error",
                    "Compose services block is not a mapping.",
                    {"path": str(origin)},
                )
            )
            return issues

        published_ports: dict[str, str] = {}

        for service_name, service in services.items():
            if not isinstance(service, dict):
                issues.append(
                    DeploymentIssue(
                        "error",
                        "Service definition is not a mapping.",
                        {"path": str(origin), "service": service_name},
                    )
                )
                continue

            if "image" not in service and "build" not in service:
                issues.append(
                    DeploymentIssue(
                        "error",
                        "Service is missing both image and build directives.",
                        {"service": service_name, "compose": str(origin)},
                    )
                )

            for env_file in _as_iterable(service.get("env_file")):
                resolved = (origin.parent / Path(env_file)).resolve()
                if not resolved.exists():
                    issues.append(
                        DeploymentIssue(
                            "warning",
                            "Referenced env_file is missing.",
                            {
                                "service": service_name,
                                "compose": str(origin),
                                "env_file": str(resolved),
                            },
                        )
                    )

            for volume in _as_iterable(service.get("volumes", [])):
                host_path = _extract_volume_host_path(volume, origin.parent)
                if host_path and not host_path.exists():
                    issues.append(
                        DeploymentIssue(
                            "warning",
                            "Host volume path does not exist; container will mount an empty directory.",
                            {
                                "service": service_name,
                                "compose": str(origin),
                                "host_path": str(host_path),
                            },
                        )
                    )

            for binding in _as_iterable(service.get("ports", [])):
                host_port = _extract_host_port(binding)
                if not host_port:
                    continue
                first_consumer = published_ports.get(host_port)
                if first_consumer and first_consumer != service_name:
                    issues.append(
                        DeploymentIssue(
                            "error",
                            "Host port is published by multiple services.",
                            {
                                "compose": str(origin),
                                "host_port": host_port,
                                "services": sorted({first_consumer, service_name}),
                            },
                        )
                    )
                else:
                    published_ports[host_port] = service_name

        return issues

    def _simulate_kubernetes(self) -> list[DeploymentIssue]:
        issues: list[DeploymentIssue] = []

        for manifest in self.kubernetes_manifests:
            if not manifest.exists():
                issues.append(
                    DeploymentIssue(
                        "warning",
                        "Kubernetes manifest not found; skipping validation.",
                        {"path": str(manifest)},
                    )
                )
                continue

            try:
                documents = list(yaml.safe_load_all(manifest.read_text()))
            except yaml.YAMLError as exc:
                issues.append(
                    DeploymentIssue(
                        "error",
                        "Failed to parse Kubernetes manifest.",
                        {"path": str(manifest), "exception": str(exc)},
                    )
                )
                continue

            for document in documents:
                if not isinstance(document, dict):
                    continue
                kind = document.get("kind")
                if kind != "Deployment":
                    continue
                issues.extend(self._evaluate_kubernetes_deployment(document, manifest))

        return issues

    def _evaluate_kubernetes_deployment(
        self, deployment: dict[str, Any], manifest_path: Path
    ) -> list[DeploymentIssue]:
        issues: list[DeploymentIssue] = []

        spec = deployment.get("spec", {})
        template = spec.get("template", {})
        pod_spec = template.get("spec", {})
        containers = pod_spec.get("containers", [])

        if not containers:
            issues.append(
                DeploymentIssue(
                    "error",
                    "Deployment defines no containers.",
                    {
                        "manifest": str(manifest_path),
                        "name": deployment.get("metadata", {}).get("name"),
                    },
                )
            )
            return issues

        for container in containers:
            name = container.get("name", "<unnamed>")
            image = (container.get("image") or "").strip()
            if not image:
                issues.append(
                    DeploymentIssue(
                        "error",
                        "Container image is missing; Kubernetes will refuse to schedule the pod.",
                        {"manifest": str(manifest_path), "container": name},
                    )
                )

            resources = container.get("resources") or {}
            if not resources.get("limits") or not resources.get("requests"):
                issues.append(
                    DeploymentIssue(
                        "warning",
                        "Container does not declare resource requests and limits; scheduling may be unpredictable.",
                        {"manifest": str(manifest_path), "container": name},
                    )
                )

            if "livenessProbe" not in container or "readinessProbe" not in container:
                missing: list[str] = []
                if "livenessProbe" not in container:
                    missing.append("livenessProbe")
                if "readinessProbe" not in container:
                    missing.append("readinessProbe")
                issues.append(
                    DeploymentIssue(
                        "warning",
                        "Container is missing health probes; rollouts may stall or crash loops may go undetected.",
                        {
                            "manifest": str(manifest_path),
                            "container": name,
                            "missing_probes": missing,
                        },
                    )
                )

        return issues


def _as_iterable(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return value
    return [value]


def _extract_volume_host_path(entry: Any, base: Path) -> Path | None:
    if isinstance(entry, str):
        parts = entry.split(":", 2)
        if not parts:
            return None
        candidate = parts[0]
        if (
            candidate.startswith("./")
            or candidate.startswith("../")
            or candidate.startswith("/")
        ):
            return (
                (base / candidate).resolve()
                if not Path(candidate).is_absolute()
                else Path(candidate)
            )
        return None

    if isinstance(entry, dict):
        if entry.get("type") == "volume":
            return None
        source = entry.get("source") or entry.get("hostPath")
        if isinstance(source, str) and (
            source.startswith("./")
            or source.startswith("../")
            or source.startswith("/")
        ):
            src_path = Path(source)
            return (
                (base / src_path).resolve() if not src_path.is_absolute() else src_path
            )

    return None


def _extract_host_port(entry: Any) -> str | None:
    if isinstance(entry, str):
        parts = entry.split(":")
        if len(parts) == 2:
            return parts[0]
        if len(parts) == 3:
            return parts[1]
        return None

    if isinstance(entry, dict):
        published = entry.get("published") or entry.get("host_port")
        if isinstance(published, int):
            return str(published)
        if isinstance(published, str):
            return published

    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate monGARS deployment")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to two directories above this script).",
    )
    parser.add_argument(
        "--compose",
        action="append",
        type=Path,
        default=None,
        help="Additional Docker Compose files to validate.",
    )
    parser.add_argument(
        "--k8s",
        action="append",
        type=Path,
        default=None,
        help="Additional Kubernetes manifests to validate.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures and exit with a non-zero status.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON report instead of human-readable output.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    compose_files = [Path(p) for p in args.compose] if args.compose else None
    kubernetes_manifests = [Path(p) for p in args.k8s] if args.k8s else None

    simulator = DeploymentSimulator(
        root=args.root,
        compose_files=compose_files,
        kubernetes_manifests=kubernetes_manifests,
    )

    report = simulator.run()

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        for issue in report.issues:
            context = json.dumps(issue.context or {}, indent=2)
            print(f"[{issue.severity.upper()}] {issue.message}")
            if issue.context:
                print(context)

        print(
            f"Summary: {report.error_count} error(s), {report.warning_count} warning(s), {report.info_count} info message(s)."
        )

    if report.error_count > 0:
        return 1

    if args.strict and report.warning_count > 0:
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution path
    raise SystemExit(main())
