from __future__ import annotations


def detect_nominalizations(tokens: list[tuple[str, str]], suffixes: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    for token, pos in tokens:
        low = token.lower()
        if pos in {"NOUN", "PROPN"} and low.endswith(suffixes):
            found.append(low)
    return found


def detect_passive(doc_sent) -> bool:
    deps = {t.dep_ for t in doc_sent}
    if "auxpass" in deps or "nsubjpass" in deps:
        return True
    for t in doc_sent:
        if t.lemma_ in {"be", "get"} and t.head.tag_ == "VBN":
            return True
    return False
