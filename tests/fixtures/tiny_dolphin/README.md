# Project overview
> **Last updated:** 2025-11-16 _(auto-synced; run `python scripts/update_docs_metadata.py`)_

This directory hosts the tiny Dolphin fixture referenced by runtime tests.
The tests monkeypatch model loaders to read from this path, so only its
existence on disk is required.
