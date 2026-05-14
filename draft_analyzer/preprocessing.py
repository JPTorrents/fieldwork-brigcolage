from __future__ import annotations

import spacy


def load_nlp(model: str = "en_core_web_sm"):
    try:
        return spacy.load(model)
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is missing. Install with: python -m spacy download en_core_web_sm"
        ) from exc
