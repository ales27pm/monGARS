#!/usr/bin/env python3
"""Interactive Docker Compose orchestrator for the monGARS stack."""

from __future__ import annotations

import json
import os
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PROJECT_DEFAULT_NAME = "mongars"
WEAK_SECRET_DEFAULTS = {
    "SECRET_KEY": {"dev-secret-change-me", ""},
    "DJANGO_SECRET_KEY": {"django-insecure-change-me", ""},
    "DB_PASSWORD": {"changeme", ""},
    "VAULT_TOKEN": {"dev-root-token", ""},
}


@dataclass(frozen=True)
class PortSpec:
    key: str
    default: int
    description: str


@dataclass(frozen=True)
class DiagnosticResult:
    name: str
    passed: bool
    details: str
    remediation: str | None = None


class ComposeError(RuntimeError):
    """Raised when docker compose commands fail."""

    def __init__(self, message: str, *, stdout: str = "", stderr: str = "") -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


class DockerMenu:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parent.parent
        self.compose_file = self.project_root / "docker-compose.yml"
        self.env_file = self.project_root / ".env"
        self.env_template = self.project_root / ".env.example"
        self.project_name = os.environ.get("COMPOSE_PROJECT_NAME", PROJECT_DEFAULT_NAME)
        self.compose_binary = self._resolve_compose_binary()
        self.active_profiles: list[str] = []
        self._last_diagnostics: list[DiagnosticResult] = []

        if not self.compose_file.exists():
            raise FileNotFoundError(
                f"Docker Compose definition not found at {self.compose_file}"
            )

    # ------------------------------------------------------------------
    # Logging utilities
    # ------------------------------------------------------------------
    @staticmethod
    def log(message: str, *, error: bool = False) -> None:
        stream = sys.stderr if error else sys.stdout
        print(f"[monGARS] {message}", file=stream)

    def log_block(self, heading: str, lines: Iterable[str]) -> None:
        self.log(heading)
        for line in lines:
            self.log(f"  {line}")

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _model_install_dir(self) -> Path:
        """Return the canonical location for wrapper artefacts."""

        return self.project_root / "models" / "encoders" / "chat_and_embed"

    def _default_dataset_path(self) -> Path:
        """Return the default dataset path used for fine-tuning."""

        return self.project_root / "datasets" / "formatted_dataset 2.jsonl"

    def _resolve_path(self, raw_path: str) -> Path:
        """Resolve ``raw_path`` relative to the project root."""

        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (self.project_root / path).resolve()
        else:
            path = path.resolve()
        return path

    # ------------------------------------------------------------------
    # Subprocess helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _run_command(
        command: Sequence[str],
        *,
        capture_output: bool = True,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        safe_command = [str(arg) for arg in command]
        if not safe_command:
            raise ValueError("Command sequence must contain at least one argument")
        return subprocess.run(
            safe_command,
            text=True,
            capture_output=capture_output,
            check=check,
        )

    # ------------------------------------------------------------------
    # Environment helpers
    # ------------------------------------------------------------------
    def ensure_env_file(self) -> None:
        if not self.env_file.exists():
            if not self.env_template.exists():
                raise FileNotFoundError(
                    "Missing .env.example template; cannot bootstrap environment."
                )
            shutil.copyfile(self.env_template, self.env_file)
            self.log("Created .env from .env.example")

        self._refresh_secret_defaults()

    def _refresh_secret_defaults(self) -> None:
        env_values = self._read_env_file()
        updates: Dict[str, str] = {}

        for key, weak_values in WEAK_SECRET_DEFAULTS.items():
            current = env_values.get(key, "").strip()
            if current in weak_values:
                if key.endswith("SECRET_KEY"):
                    updates[key] = secrets.token_urlsafe(64)
                elif key == "DB_PASSWORD":
                    updates[key] = secrets.token_urlsafe(24)
                elif key == "VAULT_TOKEN":
                    updates[key] = secrets.token_hex(16)
                else:
                    updates[key] = secrets.token_urlsafe(32)

        if updates:
            self._write_env_updates(updates)
            for key in updates:
                self.log(f"Regenerated strong secret for {key}")

    def _read_env_file(self) -> Dict[str, str]:
        return self._parse_env_file(self.env_file)

    @staticmethod
    def _parse_env_file(path: Path) -> Dict[str, str]:
        values: Dict[str, str] = {}
        if not path.exists():
            return values
        for raw in path.read_text().splitlines():
            if "=" not in raw:
                continue
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            key, _, rest = raw.partition("=")
            values[key.strip()] = rest.strip()
        return values

    def _write_env_updates(self, updates: Dict[str, str]) -> None:
        lines: List[str] = []
        seen: set[str] = set()
        if self.env_file.exists():
            for raw in self.env_file.read_text().splitlines():
                if "=" in raw and not raw.lstrip().startswith("#"):
                    key, _, _ = raw.partition("=")
                    key = key.strip()
                    if key in updates:
                        lines.append(f"{key}={updates[key]}")
                        seen.add(key)
                    else:
                        lines.append(raw)
                else:
                    lines.append(raw)
        lines.extend(
            f"{key}={value}" for key, value in updates.items() if key not in seen
        )
        self.env_file.write_text("\n".join(lines).rstrip() + "\n")

    # ------------------------------------------------------------------
    # Port management
    # ------------------------------------------------------------------
    def prepare_ports(self) -> None:
        specs = self._port_specs()
        env_values = self._read_env_file()
        reserved: set[int] = set()

        for spec in specs:
            value = env_values.get(spec.key) or os.environ.get(spec.key)
            if not value:
                candidate = spec.default
            else:
                try:
                    candidate = int(value)
                except ValueError:
                    self.log(
                        f"Invalid port '{value}' for {spec.key}; falling back to {spec.default}",
                        error=True,
                    )
                    candidate = spec.default

            candidate = self._find_available_port(candidate, reserved)
            reserved.add(candidate)
            env_values[spec.key] = str(candidate)
            self._write_env_updates({spec.key: str(candidate)})

        self._synchronise_ws_origins(env_values)

    @staticmethod
    def _port_specs() -> list[PortSpec]:
        return [
            PortSpec("API_PORT", 8000, "FastAPI service"),
            PortSpec("WEBAPP_PORT", 8001, "Django webapp"),
            PortSpec("POSTGRES_PORT", 5432, "PostgreSQL database"),
            PortSpec("REDIS_PORT", 6379, "Redis cache"),
            PortSpec("MLFLOW_PORT", 5000, "MLflow tracking server"),
            PortSpec("VAULT_PORT", 8200, "Vault dev server"),
            PortSpec("OLLAMA_PORT", 11434, "Ollama runtime"),
            PortSpec("RAY_HTTP_PORT", 8005, "Ray Serve HTTP endpoint"),
            PortSpec("RAY_DASHBOARD_PORT", 8265, "Ray dashboard"),
            PortSpec("RAY_CLIENT_PORT", 10001, "Ray client"),
            PortSpec("RAY_MIN_WORKER_PORT", 20000, "Ray worker port range (min)"),
            PortSpec("RAY_MAX_WORKER_PORT", 20100, "Ray worker port range (max)"),
        ]

    def _find_available_port(self, start: int, reserved: set[int]) -> int:
        if start < 1 or start > 65535:
            raise ValueError(f"Port {start} outside valid range (1-65535)")
        candidate = start
        while candidate <= 65535:
            if candidate in reserved:
                candidate += 1
                continue
            if self._is_port_available(candidate) or self._port_owned_by_project(
                candidate
            ):
                return candidate
            candidate += 1
        raise RuntimeError("Unable to find free port in range 1-65535")

    def _is_port_available(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
            except OSError:
                return False
        return True

    def _port_owned_by_project(self, port: int) -> bool:
        try:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    f"label=com.docker.compose.project={self.project_name}",
                    "--format",
                    "{{json .}}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return False

        if result.returncode != 0:
            return False

        target = str(port)
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            ports_field = data.get("Ports") or ""
            for chunk in ports_field.split(","):
                chunk = chunk.strip()
                if "->" not in chunk:
                    continue
                host_part = chunk.split("->", 1)[0]
                if not host_part:
                    continue
                host_port = host_part.rsplit(":", 1)[-1]
                host_port = host_port.split("-", 1)[0]
                if host_port == target:
                    return True
        return False

    def _synchronise_ws_origins(self, env_values: Dict[str, str]) -> None:
        api_port = env_values.get("API_PORT", "8000")
        webapp_port = env_values.get("WEBAPP_PORT", "8001")
        existing = env_values.get("WS_ALLOWED_ORIGINS", "")

        origins: list[str] = []
        if existing:
            try:
                parsed = json.loads(existing)
            except json.JSONDecodeError:
                parsed = [
                    entry.strip().strip("'\"")
                    for entry in existing.strip("[]").split(",")
                    if entry.strip()
                ]
            else:
                parsed = (
                    [str(item) for item in parsed] if isinstance(parsed, list) else []
                )
            origins = parsed

        candidates = [
            f"http://localhost:{api_port}",
            f"http://127.0.0.1:{api_port}",
            f"http://localhost:{webapp_port}",
            f"http://127.0.0.1:{webapp_port}",
        ]
        deduped = list(dict.fromkeys([*origins, *candidates]))

        self._write_env_updates({"WS_ALLOWED_ORIGINS": json.dumps(deduped)})

    # ------------------------------------------------------------------
    # Compose command helpers
    # ------------------------------------------------------------------
    def _resolve_compose_binary(self) -> List[str]:
        candidates = (["docker", "compose"], ["docker-compose"])
        for candidate in candidates:
            try:
                result = self._run_command([*candidate, "version"])
            except FileNotFoundError:
                continue
            if result.returncode == 0:
                return list(candidate)
        raise FileNotFoundError(
            "Docker Compose plugin or docker-compose binary is required but was not found."
        )

    def compose(
        self,
        *args: str,
        profiles: Sequence[str] | None = None,
        capture_output: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        profile_args: list[str] = []
        selected = list(dict.fromkeys(profiles or self.active_profiles))
        for profile in selected:
            profile_args.extend(["--profile", profile])

        cmd = [
            *self.compose_binary,
            "-f",
            str(self.compose_file),
            "--project-name",
            self.project_name,
            *profile_args,
            *args,
        ]

        safe_cmd = [str(arg) for arg in cmd]

        self.log(f"Executing: {' '.join(safe_cmd)}")
        result = subprocess.run(
            safe_cmd,
            text=True,
            capture_output=capture_output,
            check=False,
        )

        if capture_output:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

        if check and result.returncode != 0:
            raise ComposeError(
                f"Command {' '.join(safe_cmd)} failed with exit code {result.returncode}",
                stdout=result.stdout or "",
                stderr=result.stderr or "",
            )
        return result

    def _handle_compose_failure(self, error: ComposeError) -> None:
        self.log(
            "Docker Compose reported a failure — running triage steps...", error=True
        )
        hints: list[str] = []
        lowered = error.stderr.lower()
        if (
            "address already in use" in lowered
            or "port is already allocated" in lowered
        ):
            hints.append(
                "Detected host port conflict. Attempting to auto-reconcile the port map via prepare_ports()."
            )
            try:
                self.prepare_ports()
                hints.append(
                    "Ports refreshed successfully. Re-run your command to apply the updated mapping."
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                hints.append(f"Automatic port remediation failed: {exc}")

        if "no such network" in lowered:
            hints.append(
                "Docker network missing. Recreating via 'docker compose up -d' will rebuild networks."
            )

        if hints:
            self.log_block("Suggested remediation", hints)

        try:
            status = self.compose("ps", capture_output=True, check=False)
            if status.stdout:
                self.log("Service status snapshot:")
                for line in status.stdout.strip().splitlines():
                    self.log(f"  {line}")
        except Exception as exc:  # pragma: no cover - diagnostic best effort
            self.log(f"Unable to collect docker compose status: {exc}", error=True)

        self.log("Running non-invasive diagnostics for additional context...")
        self.run_diagnostics(auto_remediate=False)

    # ------------------------------------------------------------------
    # Diagnostics & auto-remediation
    # ------------------------------------------------------------------
    def run_diagnostics(self, *, auto_remediate: bool = True) -> list[DiagnosticResult]:
        checks = [
            self._diagnose_docker_daemon,
            self._diagnose_compose_version,
            lambda: self._diagnose_env_file(auto_remediate=auto_remediate),
            lambda: self._diagnose_ports(auto_remediate=auto_remediate),
            self._diagnose_compose_config,
        ]

        results: list[DiagnosticResult] = []
        for check in checks:
            try:
                result = check()
            except Exception as exc:  # pragma: no cover - defensive logging
                result = DiagnosticResult(
                    getattr(check, "__name__", "diagnostic"),
                    False,
                    f"Unexpected failure: {exc}",
                )
            results.append(result)

        self._last_diagnostics = results
        self._print_diagnostics(results)
        return results

    def _diagnose_docker_daemon(self) -> DiagnosticResult:
        try:
            result = self._run_command(
                [
                    "docker",
                    "info",
                    "--format",
                    "{{json .ServerVersion}}",
                ]
            )
        except FileNotFoundError:
            return DiagnosticResult(
                "Docker CLI",
                False,
                "docker binary not found in PATH",
                "Install Docker Desktop or the Docker Engine CLI tools.",
            )

        if result.returncode != 0:
            details = result.stderr.strip() or "docker info failed"
            return DiagnosticResult(
                "Docker daemon",
                False,
                details,
                "Ensure the Docker daemon is running and your user has permission to access it.",
            )

        version = (result.stdout or "").strip().strip('"') or "unknown"
        return DiagnosticResult(
            "Docker daemon",
            True,
            f"Connected to Docker engine version {version}",
        )

    def _diagnose_compose_version(self) -> DiagnosticResult:
        result = self._run_command([*self.compose_binary, "version"])
        if result.returncode != 0:
            return DiagnosticResult(
                "Docker Compose",
                False,
                result.stderr.strip() or "Unable to determine docker compose version",
                "Verify the Docker Compose plugin/binary is installed and runnable.",
            )

        version_text = (result.stdout or result.stderr or "").strip().splitlines()[0]
        version = self._extract_compose_version(version_text)
        return DiagnosticResult(
            "Docker Compose",
            True,
            f"Using docker compose {version}",
        )

    def _diagnose_env_file(self, *, auto_remediate: bool) -> DiagnosticResult:
        template_values = self._parse_env_file(self.env_template)
        env_values = self._read_env_file()
        missing = sorted(key for key in template_values if key not in env_values)
        auto_fixed = False

        if missing and auto_remediate:
            if updates := {key: template_values[key] for key in missing}:
                self._write_env_updates(updates)
                env_values = self._read_env_file()
                missing = sorted(
                    key for key in template_values if key not in env_values
                )
                auto_fixed = not missing

        if missing:
            return DiagnosticResult(
                ".env completeness",
                auto_fixed,
                f"Missing keys: {', '.join(missing)}",
                "Regenerate your .env from .env.example or rerun the menu's Deploy option.",
            )

        if weak := [
            key
            for key, defaults in WEAK_SECRET_DEFAULTS.items()
            if env_values.get(key, "") in defaults
        ]:
            if auto_remediate:
                self._refresh_secret_defaults()
                return DiagnosticResult(
                    ".env secrets",
                    True,
                    f"Regenerated weak secrets: {', '.join(weak)}",
                )
            return DiagnosticResult(
                ".env secrets",
                False,
                f"Weak secrets present: {', '.join(weak)}",
                "Run the menu's Deploy flow to rotate secrets automatically.",
            )

        return DiagnosticResult(
            ".env completeness",
            True,
            ".env file is in sync with template and secrets are strong",
        )

    def _diagnose_ports(self, *, auto_remediate: bool) -> DiagnosticResult:
        env_values = self._read_env_file()
        conflicts: list[str] = []
        reserved: set[int] = set()
        updates: Dict[str, str] = {}

        for spec in self._port_specs():
            current_value = env_values.get(spec.key)
            try:
                port = int(current_value) if current_value else spec.default
            except ValueError:
                port = spec.default
                conflicts.append(
                    f"{spec.key} invalid ('{current_value}'); reset to {spec.default}"
                )

            original = port
            port = self._find_available_port(port, reserved)
            reserved.add(port)
            if port != original:
                conflicts.append(
                    f"{spec.key} reassigned from {original} to {port} ({spec.description})"
                )
                updates[spec.key] = str(port)

        remedied = False
        if conflicts and auto_remediate:
            self._write_env_updates(updates)
            self._synchronise_ws_origins({**env_values, **updates})
            remedied = True

        if conflicts:
            remediation = None
            if not remedied:
                remediation = "Re-run deployment to apply regenerated port assignments."
            return DiagnosticResult(
                "Port availability",
                remedied,
                "; ".join(conflicts),
                remediation,
            )

        return DiagnosticResult(
            "Port availability",
            True,
            "All declared ports available for use",
        )

    def _diagnose_compose_config(self) -> DiagnosticResult:
        result = self.compose("config", "--quiet", capture_output=True, check=False)
        if result.returncode != 0:
            return DiagnosticResult(
                "Compose config",
                False,
                result.stderr.strip() or "docker compose config --quiet failed",
                "Inspect docker-compose.yml for syntax issues.",
            )
        return DiagnosticResult(
            "Compose config",
            True,
            "docker-compose.yml parsed successfully",
        )

    def _print_diagnostics(self, results: Sequence[DiagnosticResult]) -> None:
        self.log("Diagnostic summary:")
        for result in results:
            status = "✅" if result.passed else "❌"
            self.log(f"  {status} {result.name}: {result.details}")
            if result.remediation and not result.passed:
                self.log(f"     ↳ Fix: {result.remediation}")

    @staticmethod
    def _extract_compose_version(raw: str) -> str:
        for token in raw.replace(",", " ").split():
            token = token.strip()
            if token.startswith("v") and any(ch.isdigit() for ch in token):
                return token.lstrip("v")
        return raw

    # ------------------------------------------------------------------
    # Interactive flows
    # ------------------------------------------------------------------
    def deploy(self, build: bool) -> None:
        self.ensure_env_file()
        self.prepare_ports()
        profiles = self.prompt_profiles()
        self.active_profiles = profiles
        args = ["up", "-d"]
        if build:
            args.append("--build")
        self.compose(*args, profiles=profiles)
        self.compose("ps", profiles=profiles)

    def prompt_profiles(self) -> list[str]:
        profiles: list[str] = []
        if self._prompt_yes_no("Enable Ollama inference profile?", default=False):
            profiles.append("inference")
        if self._prompt_yes_no("Enable Ray orchestration profile?", default=False):
            profiles.append("ray")
        return profiles

    def stop(self) -> None:
        self.compose("stop")

    def restart(self) -> None:
        services = self._prompt_services()
        self.compose("restart", *services)

    def status(self) -> None:
        self.compose("ps")

    def logs(self) -> None:
        services = self._prompt_services()
        tail_raw = input("Tail how many lines? [200]: ").strip() or "200"
        tail = tail_raw if tail_raw.isdigit() else "200"
        args = ["logs", "-f", "--tail", tail]
        args.extend(services)
        try:
            self.compose(*args, capture_output=False)
        except ComposeError:
            self.log("Log streaming terminated or failed.", error=True)

    def rebuild_images(self) -> None:
        self.compose("build", "--pull")

    def pull_images(self) -> None:
        self.compose("pull")

    def destroy(self) -> None:
        if self._prompt_yes_no(
            "This removes containers, volumes, and networks. Continue?", default=False
        ):
            self.compose("down", "-v", "--remove-orphans")

    def exec_shell(self) -> None:
        services = self._prompt_services(single=True)
        if not services:
            self.log("No service selected.", error=True)
            return
        command = input("Command to run inside container [bash]: ").strip() or "bash"
        try:
            cmd_args = shlex.split(command)
        except ValueError:
            self.log("Invalid command syntax.", error=True)
            return
        self.compose("exec", services[0], *cmd_args, capture_output=False)

    def config(self) -> None:
        self.compose("config")

    def diagnostics(self) -> None:
        auto_fix = self._prompt_yes_no(
            "Attempt automatic remediation for detected issues?", default=True
        )
        self.run_diagnostics(auto_remediate=auto_fix)

    def auto_heal(self) -> None:
        self.log("Initiating auto-heal workflow...")
        results = self.run_diagnostics(auto_remediate=True)
        if unresolved := [result for result in results if not result.passed]:
            self.log(
                "Some diagnostics require manual intervention before auto-heal can continue:",
                error=True,
            )
            for result in unresolved:
                self.log(f" - {result.name}: {result.details}", error=True)
                if result.remediation:
                    self.log(f"   ↳ Suggested fix: {result.remediation}", error=True)
            return

        state = self._collect_service_state()
        restart_targets = [
            name
            for name, info in state.items()
            if info.get("state") != "running"
            or info.get("health") not in {None, "healthy"}
        ]

        if not restart_targets:
            self.log("All services report healthy. No restart required.")
            return

        for service in restart_targets:
            self.log(f"Restarting {service}...")
            try:
                self.compose("restart", service, capture_output=True, check=False)
            except (
                ComposeError
            ) as exc:  # pragma: no cover - compose restart rarely fails
                self.log(f"Restart of {service} failed: {exc}", error=True)
        self.log("Auto-heal pass completed. Re-run status to confirm service health.")

    def show_environment_summary(self) -> None:
        env_values = self._read_env_file()
        if not env_values:
            self.log(
                ".env file not found. Run Deploy to bootstrap configuration.",
                error=True,
            )
            return
        summary_lines = [
            f"{key}={self._mask_env_value(key, env_values[key])}"
            for key in sorted(env_values)
        ]
        self.log_block("Environment snapshot", summary_lines)

    def generate_base_model_bundle(self) -> None:
        """Run the LLM pipeline to fine-tune and install the project wrapper."""

        script_path = self.project_root / "scripts" / "run_mongars_llm_pipeline.py"
        if not script_path.exists():
            self.log(
                f"LLM pipeline script missing at {script_path}.",
                error=True,
            )
            return

        default_dataset = self._default_dataset_path()
        dataset_prompt = input(
            f"Instruction dataset JSONL [{default_dataset}]: "
        ).strip()
        if dataset_prompt:
            dataset_path = self._resolve_path(dataset_prompt)
        else:
            dataset_path = default_dataset

        if not dataset_path.exists():
            self.log(
                f"Dataset not found at {dataset_path}. Provide a valid JSONL dataset.",
                error=True,
            )
            return

        install_root = self._model_install_dir()
        install_root.parent.mkdir(parents=True, exist_ok=True)
        if install_root.exists():
            if not self._prompt_yes_no(
                f"Existing artefacts detected at {install_root}. Overwrite?",
                default=False,
            ):
                self.log("Aborted base model generation.")
                return
            shutil.rmtree(install_root)

        model_default = "dphn/Dolphin3.0-Llama3.1-8B"
        model_id = (
            input(f"Base model identifier [{model_default}]: ").strip() or model_default
        )

        command = [
            sys.executable,
            str(script_path),
            "finetune",
            "--model-id",
            model_id,
            "--dataset-path",
            str(dataset_path),
            "--output-dir",
            str(install_root),
            "--skip-smoke-tests",
            "--skip-merge",
        ]

        summary = [
            f"Model: {model_id}",
            f"Dataset: {dataset_path}",
            f"Output: {install_root}",
        ]
        self.log_block("Launching fine-tuning pipeline", summary)

        try:
            result = self._run_command(command, capture_output=False, check=False)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.log(f"Failed to launch fine-tuning pipeline: {exc}", error=True)
            return

        if result.returncode != 0:
            self.log(
                "Fine-tuning pipeline failed. Inspect the logs above for details.",
                error=True,
            )
            return

        wrapper_dir = install_root / "wrapper"
        if not wrapper_dir.exists():
            self.log(
                f"Pipeline completed but wrapper bundle missing at {wrapper_dir}.",
                error=True,
            )
            return

        chat_lora_dir = install_root / "chat_lora"
        self.log_block(
            "Base model + wrapper installation complete",
            [
                f"Wrapper: {wrapper_dir}",
                f"LoRA adapters: {chat_lora_dir if chat_lora_dir.exists() else 'pending'}",
                "Smoke tests skipped; rerun pipeline without --skip-smoke-tests if desired.",
            ],
        )

    def _collect_service_state(self) -> Dict[str, Dict[str, str]]:
        state: Dict[str, Dict[str, str]] = {}
        result = self.compose(
            "ps",
            "--format",
            "json",
            capture_output=True,
            check=False,
        )

        entries: list[Dict[str, str]] = []
        if result.returncode == 0:
            try:
                parsed = json.loads(result.stdout or "[]")
            except json.JSONDecodeError:
                parsed = []
            else:
                if isinstance(parsed, list):
                    entries = [entry for entry in parsed if isinstance(entry, dict)]

        if not entries:
            fallback = self.compose("ps", capture_output=True, check=False)
            entries = self._parse_compose_ps_table(fallback.stdout or "")

        for entry in entries:
            service = (
                entry.get("Service")
                or entry.get("service")
                or entry.get("Name")
                or entry.get("name")
            )
            if not service:
                continue
            raw_state = (
                entry.get("State")
                or entry.get("state")
                or entry.get("STATUS")
                or entry.get("status")
                or ""
            )
            raw_state_lower = raw_state.lower()
            health_value = entry.get("Health") or entry.get("health")
            if "(" in raw_state_lower and ")" in raw_state_lower:
                base, _, remainder = raw_state_lower.partition("(")
                candidate_health = remainder.rstrip(")")
                if not health_value and candidate_health:
                    health_value = candidate_health
                raw_state_lower = base.strip()
            state[service] = {
                "state": raw_state_lower.strip(),
                "health": (
                    health_value.strip()
                    if isinstance(health_value, str)
                    else health_value
                ),
            }
        return state

    @staticmethod
    def _mask_env_value(key: str, value: str) -> str:
        lowered = key.lower()
        if any(token in lowered for token in ("secret", "password", "token")):
            if len(value) <= 6:
                return "***"
            return f"{value[:4]}…{value[-2:]}"
        return value

    @staticmethod
    def _parse_compose_ps_table(output: str) -> list[Dict[str, str]]:
        entries: list[Dict[str, str]] = []
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if len(lines) <= 1:
            return entries
        headers = lines[0].split()
        for line in lines[1:]:
            parts = line.split(None, len(headers) - 1)
            if len(parts) < len(headers):
                parts.extend([""] * (len(headers) - len(parts)))
            entry = {header: parts[idx] for idx, header in enumerate(headers)}
            if len(headers) >= 4:
                status_key = headers[2]
                ports_key = headers[3]
                status_val = entry.get(status_key, "")
                ports_val = entry.get(ports_key, "")
                if ports_val.startswith("(") and "(" not in status_val:
                    prefix, sep, remainder = ports_val.partition(")")
                    status_suffix = f"{prefix}{sep}".strip()
                    entry[status_key] = f"{status_val} {status_suffix}".strip()
                    entry[ports_key] = remainder.strip()
                    parts[2] = entry[status_key]
                    parts[3] = entry[ports_key]
            for idx, header in enumerate(headers):
                entry[header.lower()] = parts[idx]
            entries.append(entry)
        return entries

    # ------------------------------------------------------------------
    # Input helpers
    # ------------------------------------------------------------------
    def _prompt_yes_no(self, prompt: str, *, default: bool) -> bool:
        suffix = "[Y/n]" if default else "[y/N]"
        while True:
            choice = input(f"{prompt} {suffix} ").strip().lower()
            if not choice:
                return default
            if choice in {"y", "yes"}:
                return True
            if choice in {"n", "no"}:
                return False
            self.log("Please respond with 'y' or 'n'.", error=True)

    def _prompt_services(self, *, single: bool = False) -> list[str]:
        services = input(
            "Enter service names separated by spaces (leave empty for all): "
        ).strip()
        if not services:
            return []
        parts = services.split()
        if single and len(parts) > 1:
            self.log("Only one service allowed; using the first entry.", error=True)
            return parts[:1]
        return parts

    def menu(self) -> None:
        actions = {
            "1": ("Deploy/Update stack (build + up)", lambda: self.deploy(build=True)),
            "2": ("Start stack without rebuild", lambda: self.deploy(build=False)),
            "3": ("Stop running services", self.stop),
            "4": ("Restart services", self.restart),
            "5": ("Show service status", self.status),
            "6": ("Tail service logs", self.logs),
            "7": ("Rebuild images", self.rebuild_images),
            "8": ("Pull latest images", self.pull_images),
            "9": ("Open shell inside a service", self.exec_shell),
            "10": ("View resolved docker-compose config", self.config),
            "11": ("Destroy stack (containers + volumes)", self.destroy),
            "12": ("Run diagnostics & auto-remediation checks", self.diagnostics),
            "13": ("Auto-heal unhealthy services", self.auto_heal),
            "14": ("Show environment variable summary", self.show_environment_summary),
            "15": (
                "Fine-tune base model and install wrapper",
                self.generate_base_model_bundle,
            ),
            "0": ("Exit", None),
        }

        intro = textwrap.dedent(
            """
            ========================================
            monGARS Docker Deployment Menu
            ========================================
            """
        )
        print(intro)

        while True:
            for key, (label, _) in actions.items():
                print(f" {key.rjust(2)}. {label}")
            choice = input("\nSelect an option: ").strip()
            if choice == "0":
                print("Goodbye! 👋")
                return
            action = actions.get(choice)
            if not action:
                self.log("Invalid selection. Please try again.\n", error=True)
                continue
            _, handler = action
            assert handler is not None
            try:
                handler()
            except ComposeError as exc:
                self.log(str(exc), error=True)
                self._handle_compose_failure(exc)
            except KeyboardInterrupt:
                self.log("Operation cancelled by user.", error=True)
            print("")


def main() -> None:
    try:
        menu = DockerMenu()
    except Exception as exc:  # pragma: no cover - startup validation
        print(f"[monGARS] {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        menu.menu()
    except KeyboardInterrupt:
        print("\nGoodbye! 👋")


if __name__ == "__main__":
    main()
