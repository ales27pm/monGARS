from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set

DEFAULT_TERMS = {
    "dépanneur",
    "tuque",
    "magasiner",
    "char",
    "chum",
    "blonde",
    "icitte",
    "ben là",
    "poutine",
    "cégep",
    "patente",
}


class QCFilter:
    def __init__(self, extra_terms: Iterable[str] | None = None) -> None:
        self.terms: Set[str] = {term.lower() for term in DEFAULT_TERMS}
        if extra_terms:
            self.terms.update(
                term.lower().strip() for term in extra_terms if term.strip()
            )

    @classmethod
    def from_path(cls, path: Path | None) -> "QCFilter":
        if path is None:
            return cls()
        if not path.exists():
            raise FileNotFoundError(f"QC terms file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            terms = [line.strip() for line in handle if line.strip()]
        return cls(terms)

    def flag_text(self, *parts: str) -> bool:
        candidate = " ".join(parts).lower()
        return any(term in candidate for term in self.terms)
