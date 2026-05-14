from __future__ import annotations

THRESHOLDS = {
    "sentence_len_p2": 35,
    "sentence_len_p1": 50,
    "paragraph_len_p2": 220,
    "paragraph_len_p1": 320,
    "no_short_sentence_limit": 25,
    "nominalizations_p2": 3,
    "nominalizations_p1": 5,
    "vague_words_p2": 3,
    "hedges_p2": 4,
    "specificity_min": 35,
}

SECTION_TYPES = [
    "abstract",
    "introduction",
    "theory",
    "literature",
    "methods",
    "findings",
    "results",
    "discussion",
    "conclusion",
    "references",
    "unknown",
]
