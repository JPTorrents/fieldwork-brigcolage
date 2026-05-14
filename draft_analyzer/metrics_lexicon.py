from __future__ import annotations

from pathlib import Path

RES = Path(__file__).parent / "resources"


def load_terms(file_name: str) -> set[str]:
    return {line.strip().lower() for line in (RES / file_name).read_text().splitlines() if line.strip()}


def count_matches(text: str, terms: set[str]) -> int:
    t = text.lower()
    return sum(1 for w in terms if w in t)
