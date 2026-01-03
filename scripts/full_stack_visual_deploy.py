#!/usr/bin/env python3
"""Full-stack deployment automation with a visual stepper."""
from __future__ import annotations

import argparse
import logging
import os
import secrets
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Sequence
from urllib.parse import urlsplit, urlunsplit


class DeploymentError(RuntimeError):
    """Raised when an automation step fails."""


class StepStatus(Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class StepResult:
    key: str
    title: str
    status: StepStatus
    message: str | None = None


@dataclass
class Step:
    key: str
    title: str
    description: str
    handler: Callable[["VisualDeployer"], StepStatus | None]
    optional: bool = False


class TerminalUI:
    """Render simple visual blocks without third-party dependencies."""

    def __init__(self) -> None:
        self._width = max(80, shutil.get_terminal_size((120, 40)).columns)

    def banner(self, title: str) -> None:
        line = f"== {title} =="
        print("\n" + line)
        print("=" * len(line))

    def describe(self, text: str) -> None:
        wrapped = textwrap.fill(text, width=min(self._width, 100))
        print(wrapped)

    def start_step(self, title: str, description: str) -> None:
        border = "╭" + "─" * (len(title) + 2) + "╮"
        print("\n" + border)
        print(f"│ {title} │")
        print("╰" + "─" * (len(title) + 2) + "╯")
        if description:
            self.describe(description)

    def finish_step(
        self, title: str, status: StepStatus, message: str | None = None
    ) -> None:
        symbol = {
            StepStatus.SUCCESS: "✔",
            StepStatus.SKIPPED: "⚠",
            StepStatus.FAILED: "✖",
        }[status]
        summary = f"[{symbol}] {title}: {status.value.upper()}"
        if message:
            summary += f" — {message}"
        print(summary)

    def final_summary(self, results: Iterable[StepResult]) -> None:
        print("\nSummary")
        print("-" * 40)
        for result in results:
            symbol = {
                StepStatus.SUCCESS: "✔",
                StepStatus.SKIPPED: "⚠",
                StepStatus.FAILED: "✖",
            }[result.status]
            line = f"{symbol} {result.title}: {result.status.value}"
            if result.message:
                line += f" ({result.message})"
            print(line)


class EnvFileManager:
    """Utility that keeps the operator .env file consistent."""

    def __init__(self, project_root: Path, logger: logging.Logger) -> None:
        self.project_root = project_root
        self.logger = logger
        self.path = project_root / ".env"
        self.example_path = project_root / ".env.example"

    def ensure_env_file(self) -> None:
        if not self.path.exists():
            if not self.example_path.exists():
                raise DeploymentError(
                    ".env.example is missing; cannot bootstrap environment file"
                )
            shutil.copyfile(self.example_path, self.path)
            self.logger.info("Created .env from .env.example")
        else:
            self.logger.info("Using existing .env file")

    def ensure_secure_defaults(self) -> None:
        replacements = {
            "SECRET_KEY": self._random_value(48),
            "DJANGO_SECRET_KEY": self._random_value(48),
            "DB_PASSWORD": self._random_value(20),
            "SEARXNG_SECRET": self._random_value(48),
            "SEARCH_SEARX_API_KEY": self._random_value(48),
        }
        self._replace_if_default(
            "SECRET_KEY", {"dev-secret-change-me"}, replacements["SECRET_KEY"]
        )
        self._replace_if_default(
            "DJANGO_SECRET_KEY",
            {"django-insecure-change-me"},
            replacements["DJANGO_SECRET_KEY"],
        )
        self._replace_if_default(
            "DB_PASSWORD", {"changeme", "password"}, replacements["DB_PASSWORD"]
        )
        self._replace_if_default(
            "SEARXNG_SECRET", {"", "change-me"}, replacements["SEARXNG_SECRET"]
        )
        self._replace_if_default(
            "SEARCH_SEARX_API_KEY",
            {"", "change-me"},
            replacements["SEARCH_SEARX_API_KEY"],
        )
        self._ensure_key("SEARCH_SEARX_ENABLED", "true")
        self._ensure_key("SEARXNG_PORT", "8082")
        self._ensure_key("SEARXNG_BASE_URL", "http://localhost:8082")
        self._ensure_key("SEARCH_SEARX_BASE_URL", "http://localhost:8082")
        self._ensure_key("SEARCH_SEARX_INTERNAL_BASE_URL", "http://searxng:8080")
        self._align_searx_urls()

    def _replace_if_default(self, key: str, defaults: set[str], new_value: str) -> None:
        current = self._read_value(key)
        if current is None or current.strip() in defaults:
            self.logger.info("Updating %s in .env", key)
            self._write_value(key, new_value)

    def _ensure_key(self, key: str, value: str) -> None:
        if self._read_value(key) is None:
            self.logger.info("Adding %s to .env", key)
            self._write_value(key, value)

    def _read_value(self, key: str) -> str | None:
        if not self.path.exists():
            return None
        for line in self.path.read_text().splitlines():
            if not line or line.strip().startswith("#"):
                continue
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1]
        return None

    def _write_value(self, key: str, value: str) -> None:
        lines = []
        replaced = False
        if self.path.exists():
            for raw_line in self.path.read_text().splitlines():
                if raw_line.startswith(f"{key}="):
                    lines.append(f"{key}={value}")
                    replaced = True
                else:
                    lines.append(raw_line)
        if not replaced:
            lines.append(f"{key}={value}")
        with self.path.open("w", encoding="utf8") as handle:
            handle.write("\n".join(lines) + "\n")

    def _align_searx_urls(self) -> None:
        port_raw = self._read_value("SEARXNG_PORT")
        if port_raw is None:
            return
        try:
            port = int(port_raw.strip())
        except (TypeError, ValueError):
            self.logger.warning(
                "Invalid SEARXNG_PORT=%s; skipping SearxNG URL alignment", port_raw
            )
            return

        def _normalise_local_url(raw_url: str | None) -> str | None:
            raw = (raw_url or "").strip()
            if not raw:
                return f"http://localhost:{port}"
            try:
                parsed = urlsplit(raw)
            except ValueError:
                return None
            hostname = parsed.hostname
            path = parsed.path or ""
            if not hostname:
                if raw.startswith(("localhost", "127.0.0.1")):
                    host_part, _, remainder = raw.partition("/")
                    name, _, _ = host_part.partition(":")
                    hostname = name or "localhost"
                    path = f"/{remainder}" if remainder else ""
                else:
                    return None
            if hostname not in {"localhost", "127.0.0.1"}:
                return None
            scheme = parsed.scheme or "http"
            netloc = f"{hostname}:{port}"
            if path in {"", "/"}:
                path = ""
            new_url = urlunsplit((scheme, netloc, path, parsed.query, parsed.fragment))
            return new_url if new_url != raw else None

        for key in ("SEARXNG_BASE_URL", "SEARCH_SEARX_BASE_URL"):
            current = self._read_value(key)
            replacement = _normalise_local_url(current)
            if replacement is None:
                continue
            self.logger.info("Aligning %s with SEARXNG_PORT=%s", key, port)
            self._write_value(key, replacement)

    @staticmethod
    def _random_value(length: int) -> str:
        return secrets.token_urlsafe(length)[:length]


@dataclass
class VisualDeployer:
    project_root: Path
    include_searx: bool
    non_interactive: bool
    step_filter: set[str]
    logger: logging.Logger = field(init=False)
    ui: TerminalUI = field(default_factory=TerminalUI, init=False)

    def __post_init__(self) -> None:
        log_path = self.project_root / "deployment.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.FileHandler(log_path, encoding="utf8"),
                logging.StreamHandler(sys.stderr),
            ],
        )
        self.logger = logging.getLogger("deploy")
        self.env_manager = EnvFileManager(self.project_root, self.logger)

    def run(self) -> int:
        self.ui.banner("monGARS Full-Stack Deployment")
        self.ui.describe(
            "This wizard prepares Python dependencies, installs frontend packages, "
            "creates a secure .env file, provisions SearxNG, and launches the dockerised stack."
        )

        results: list[StepResult] = []
        for step in self._build_steps():
            if self.step_filter and step.key not in self.step_filter:
                continue
            self.ui.start_step(step.title, step.description)
            try:
                status = step.handler(self) or StepStatus.SUCCESS
            except DeploymentError as exc:  # noqa: PERF203
                self.logger.error("Step %s failed: %s", step.key, exc)
                self.ui.finish_step(step.title, StepStatus.FAILED, str(exc))
                failure_result = StepResult(
                    step.key, step.title, StepStatus.FAILED, str(exc)
                )
                if not step.optional or self.non_interactive:
                    results.append(failure_result)
                    self.ui.final_summary(results)
                    return 1
                if not self._confirm_continue():
                    results.append(failure_result)
                    self.ui.final_summary(results)
                    return 1
                results.append(
                    StepResult(
                        step.key,
                        step.title,
                        StepStatus.SKIPPED,
                        "operator skipped after failure",
                    )
                )
                continue
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("Unexpected failure during %s", step.key)
                self.ui.finish_step(step.title, StepStatus.FAILED, str(exc))
                results.append(
                    StepResult(step.key, step.title, StepStatus.FAILED, str(exc))
                )
                self.ui.final_summary(results)
                return 1
            else:
                self.ui.finish_step(step.title, status)
                results.append(StepResult(step.key, step.title, status))
        self.ui.final_summary(results)
        return 0

    # --- Step handlers -------------------------------------------------

    def step_prerequisites(self) -> StepStatus:
        missing: list[str] = []
        for command in ("python3", "pip", "npm", "node", "docker"):
            if shutil.which(command) is None:
                missing.append(command)
        compose = self._compose_invocation()
        if not compose:
            missing.append("docker compose")
        if missing:
            raise DeploymentError(
                "Missing required commands: " + ", ".join(sorted(set(missing)))
            )
        return StepStatus.SUCCESS

    def step_env_file(self) -> StepStatus:
        self.env_manager.ensure_env_file()
        self.env_manager.ensure_secure_defaults()
        return StepStatus.SUCCESS

    def step_python(self) -> StepStatus:
        venv_dir = self.project_root / ".venv"
        python_exec = self._venv_python(venv_dir)
        if not venv_dir.exists():
            self.logger.info("Creating virtual environment at %s", venv_dir)
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        self.logger.info("Installing backend dependencies")
        self._run_command(
            [python_exec, "-m", "pip", "install", "-U", "pip"], cwd=self.project_root
        )
        self._run_command(
            [python_exec, "-m", "pip", "install", "-r", "requirements.txt"],
            cwd=self.project_root,
        )
        return StepStatus.SUCCESS

    def step_root_node(self) -> StepStatus:
        self.logger.info("Installing root npm dependencies")
        self._run_command(["npm", "install"], cwd=self.project_root)
        return StepStatus.SUCCESS

    def step_mobile_node(self) -> StepStatus:
        app_dir = self.project_root / "mobile-app"
        if not app_dir.exists():
            self.logger.warning(
                "mobile-app directory not found; skipping React Native setup"
            )
            return StepStatus.SKIPPED
        self._run_command(["npm", "install"], cwd=app_dir)
        if sys.platform == "darwin" and shutil.which("pod"):
            self.logger.info("Running pod install for iOS dependencies")
            self._run_command(["npx", "pod-install"], cwd=app_dir)
        return StepStatus.SUCCESS

    def step_searx_config(self) -> StepStatus:
        if not self.include_searx:
            return StepStatus.SKIPPED
        config_dir = self.project_root / "configs" / "searxng"
        config_dir.mkdir(parents=True, exist_ok=True)
        settings_file = config_dir / "settings.yml"
        if not settings_file.exists():
            settings_file.write_text(
                textwrap.dedent(
                    """
                    use_default_settings: true
                    general:
                      instance_name: "monGARS SearxNG"
                      contact_url: "https://example.com/contact"
                    server:
                      bind_address: 0.0.0.0
                      port: 8080
                      base_url: "${SEARXNG_BASE_URL}"
                      secret_key: "${SEARXNG_SECRET}"
                      limiter: false
                      image_proxy: true
                    search:
                      safe_search: 1
                      formats:
                        - html
                        - json
                      default_lang: en
                      max_results: 20
                    ui:
                      static_use_hash: true
                      query_in_title: true
                    """
                ).strip()
                + "\n",
                encoding="utf8",
            )
            self.logger.info("Wrote default SearxNG settings to %s", settings_file)
        return StepStatus.SUCCESS

    def step_containers(self) -> StepStatus:
        compose = self._compose_invocation()
        if not compose:
            raise DeploymentError("Docker Compose is not installed")
        base_args = list(compose)
        compose_files = [self.project_root / "docker-compose.yml"]
        if self.include_searx:
            compose_files.append(self.project_root / "docker-compose.searxng.yml")
        for path in compose_files:
            if not path.exists():
                raise DeploymentError(f"Missing compose file: {path}")
            base_args.extend(["-f", str(path)])
        self._run_command(base_args + ["pull"], cwd=self.project_root)
        self._run_command(base_args + ["build"], cwd=self.project_root)
        self._run_command(base_args + ["up", "-d"], cwd=self.project_root)
        return StepStatus.SUCCESS

    # --- Helpers -------------------------------------------------------

    def _build_steps(self) -> list[Step]:
        steps = [
            Step(
                "prerequisites",
                "Validate prerequisites",
                "Check required CLI tooling",
                VisualDeployer.step_prerequisites,
            ),
            Step(
                "env",
                "Prepare environment file",
                "Create .env and secure secrets",
                VisualDeployer.step_env_file,
            ),
            Step(
                "python",
                "Install Python backend",
                "Provision the virtual environment and backend dependencies",
                VisualDeployer.step_python,
            ),
            Step(
                "root-node",
                "Install web dependencies",
                "Install npm packages for shared tooling and the Django operator console build assets",
                VisualDeployer.step_root_node,
            ),
            Step(
                "mobile",
                "Install React Native UI",
                "Install npm packages for the React Native client and pods on macOS",
                VisualDeployer.step_mobile_node,
            ),
            Step(
                "searxng",
                "Provision SearxNG",
                "Create configuration for the SearxNG metasearch service",
                VisualDeployer.step_searx_config,
                optional=not self.include_searx,
            ),
            Step(
                "containers",
                "Launch containers",
                "Build and start the docker-compose stack",
                VisualDeployer.step_containers,
            ),
        ]
        return steps

    def _confirm_continue(self) -> bool:
        if self.non_interactive:
            return False
        try:
            response = input("A step failed. Continue? [y/N]: ").strip().lower()
        except EOFError:
            return False
        return response in {"y", "yes"}

    def _compose_invocation(self) -> Sequence[str] | None:
        docker = shutil.which("docker")
        if docker is not None:
            completed = subprocess.run(
                [docker, "compose", "version"], capture_output=True, text=True
            )
            if completed.returncode == 0:
                return (docker, "compose")
        docker_compose = shutil.which("docker-compose")
        if docker_compose is not None:
            return (docker_compose,)
        return None

    def _run_command(self, command: Sequence[str], cwd: Path | None = None) -> None:
        self.logger.info("Running command: %s", " ".join(command))
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(f"    {line.rstrip()}")
        process.wait()
        if process.returncode != 0:
            raise DeploymentError(
                f"Command failed with exit code {process.returncode}: {' '.join(command)}"
            )

    @staticmethod
    def _venv_python(venv_dir: Path) -> str:
        if os.name == "nt":
            return str(venv_dir / "Scripts" / "python.exe")
        return str(venv_dir / "bin" / "python")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate monGARS full-stack deployment"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Path to the monGARS repository root",
    )
    parser.add_argument(
        "--include-searx",
        dest="include_searx",
        action="store_true",
        default=True,
        help="Provision the SearxNG metasearch container",
    )
    parser.add_argument(
        "--skip-searx",
        dest="include_searx",
        action="store_false",
        help="Skip SearxNG provisioning",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Fail immediately when a step errors instead of prompting",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only the specified steps (keys: prerequisites, env, python, root-node, mobile, searxng, containers)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    project_root = args.project_root.resolve()
    if not project_root.exists():
        raise SystemExit(f"Project root {project_root} does not exist")
    deployer = VisualDeployer(
        project_root=project_root,
        include_searx=args.include_searx,
        non_interactive=args.non_interactive,
        step_filter=set(args.only or []),
    )
    return deployer.run()


if __name__ == "__main__":
    raise SystemExit(main())
