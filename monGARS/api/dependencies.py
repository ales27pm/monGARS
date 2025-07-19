from __future__ import annotations

from monGARS.core.hippocampus import Hippocampus

hippocampus = Hippocampus()


def get_hippocampus() -> Hippocampus:
    """Return the shared Hippocampus instance."""
    return hippocampus
