from __future__ import annotations

import statistics
from typing import Any

import textstat


def compute_readability(text: str, sentence_lengths: list[int], paragraph_lengths: list[int]) -> dict[str, Any]:
    if not text.strip():
        return {}
    metrics: dict[str, Any] = {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "dale_chall_score": textstat.dale_chall_readability_score(text),
        "linsear_write_formula": textstat.linsear_write_formula(text),
        "avg_sentence_length": statistics.mean(sentence_lengths) if sentence_lengths else 0,
        "median_sentence_length": statistics.median(sentence_lengths) if sentence_lengths else 0,
        "p90_sentence_length": sorted(sentence_lengths)[int(0.9 * (len(sentence_lengths)-1))] if sentence_lengths else 0,
        "avg_paragraph_length": statistics.mean(paragraph_lengths) if paragraph_lengths else 0,
        "median_paragraph_length": statistics.median(paragraph_lengths) if paragraph_lengths else 0,
    }
    return metrics
