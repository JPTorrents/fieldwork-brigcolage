from __future__ import annotations

import re

SECTION_PATTERNS: list[tuple[str, str]] = [
    ("abstract", r"^abstract$"),
    ("introduction", r"^introduction$"),
    ("theory", r"theor|conceptual|background"),
    ("literature", r"literature|review"),
    ("methods", r"methods?|methodology|data and methods?"),
    ("findings", r"findings?"),
    ("results", r"results?"),
    ("discussion", r"discussion"),
    ("conclusion", r"conclusion|concluding"),
    ("references", r"references|bibliography|works cited"),
]


def classify_section_heading(text: str) -> str:
    t = text.strip().lower()
    for section_type, pattern in SECTION_PATTERNS:
        if re.search(pattern, t):
            return section_type
    return "unknown"
