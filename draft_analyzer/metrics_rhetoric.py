from __future__ import annotations

CONTRIBUTION_MARKERS = {"contribute", "contribution", "show", "theorize", "develop", "identify", "reveal", "argue"}


def has_marker(text: str, markers: set[str]) -> bool:
    low = text.lower()
    return any(m in low for m in markers)
