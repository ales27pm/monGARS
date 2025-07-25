#!/usr/bin/env python
import os
import sys


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webapp.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:  # pragma: no cover - runtime error
        raise ImportError("Couldn't import Django. Ensure it is installed.") from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
