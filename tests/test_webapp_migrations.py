"""Regression tests that guard the Django migration workflow."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANAGE_PY = REPO_ROOT / "webapp" / "manage.py"


def _run_manage_command(
    args: list[str], env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(MANAGE_PY), *args],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_django_migrations_apply_cleanly(tmp_path: Path) -> None:
    """Ensure Django migrations run and report as fully applied."""

    env = os.environ.copy()
    env.setdefault("DJANGO_SECRET_KEY", "test-secret-key")
    env.setdefault("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1")
    env.setdefault("DJANGO_DEBUG", "False")
    env["DJANGO_USE_SQLITE"] = "1"
    env["DJANGO_SQLITE_PATH"] = str(tmp_path / "test-webapp.sqlite3")

    migrate_result = _run_manage_command(["migrate", "--noinput"], env)
    assert (
        migrate_result.returncode == 0
    ), f"migrate failed: {migrate_result.stdout}\n{migrate_result.stderr}"

    check_result = _run_manage_command(["migrate", "--check"], env)
    assert (
        check_result.returncode == 0
    ), f"Pending migrations detected: {check_result.stdout}\n{check_result.stderr}"
