from __future__ import annotations

import re
from collections import Counter

from .config import THRESHOLDS
from .metrics_lexicon import count_matches, load_terms
from .metrics_rhetoric import CONTRIBUTION_MARKERS, has_marker
from .metrics_syntax import detect_nominalizations, detect_passive
from .models import Flag


SUFFIXES = tuple(load_terms("nominalization_suffixes.txt"))
HEDGES = load_terms("hedges.txt")
VAGUE = load_terms("vague_academic_words.txt")
OVERCLAIM = {"prove", "proves", "proved", "establish", "establishes", "established", "demonstrate", "demonstrates", "demonstrated"}


def sentence_flags(sent, section_type: str, p_idx: int, s_idx: int) -> tuple[list[Flag], dict]:
    text = sent.text.strip()
    words = [t for t in sent if t.is_alpha]
    wc = len(words)
    flags: list[Flag] = []
    toks = [(t.text, t.pos_) for t in sent]
    nom = detect_nominalizations(toks, SUFFIXES)
    hedge_count = sum(1 for t in words if t.lemma_.lower() in HEDGES)
    vague_count = count_matches(text, VAGUE)
    passive = detect_passive(sent)
    if wc > THRESHOLDS["sentence_len_p1"]:
        flags.append(_mk("P1", "sentence", "long_sentence", "Sentence exceeds 50 words", str(wc), section_type, p_idx, s_idx, text))
    elif wc > THRESHOLDS["sentence_len_p2"]:
        flags.append(_mk("P2", "sentence", "long_sentence", "Sentence exceeds 35 words", str(wc), section_type, p_idx, s_idx, text))
    if len(re.findall(r"\([^\)]*\)", text)) > 2:
        flags.append(_mk("P2", "sentence", "citation_density", "Many parenthetical citations", text, section_type, p_idx, s_idx, text))
    if len(re.findall(r"[;:\-–—]", text)) > 3:
        flags.append(_mk("P3", "sentence", "punctuation_density", "Heavy punctuation chaining", text, section_type, p_idx, s_idx, text))
    if re.search(r"\b[Tt]his\b(?!\s+[a-z]+\b)", text):
        flags.append(_mk("P2", "sentence", "vague_demonstrative", "Vague demonstrative 'this'", text, section_type, p_idx, s_idx, text))
    if len(nom) >= THRESHOLDS["nominalizations_p1"]:
        flags.append(_mk("P1", "sentence", "nominalization_cluster", "High nominalization load", ", ".join(nom[:8]), section_type, p_idx, s_idx, text))
    elif len(nom) >= THRESHOLDS["nominalizations_p2"]:
        flags.append(_mk("P2", "sentence", "nominalization_cluster", "Nominalization cluster", ", ".join(nom[:8]), section_type, p_idx, s_idx, text))
    if vague_count >= THRESHOLDS["vague_words_p2"]:
        flags.append(_mk("P2", "sentence", "vague_lexicon", "Cluster of vague academic terms", str(vague_count), section_type, p_idx, s_idx, text))
    if passive:
        sev = "P3" if section_type == "methods" else "P2"
        flags.append(_mk(sev, "sentence", "passive_voice", "Possible passive voice", text, section_type, p_idx, s_idx, text))
    if hedge_count >= THRESHOLDS["hedges_p2"]:
        flags.append(_mk("P2", "sentence", "hedge_density", "Heavy hedging", str(hedge_count), section_type, p_idx, s_idx, text))
    if any(t.lemma_.lower() in OVERCLAIM for t in words) and section_type in {"findings", "discussion", "theory", "literature", "unknown"}:
        flags.append(_mk("P2", "sentence", "overclaim", "Possible overclaim verb", text, section_type, p_idx, s_idx, text))
    return flags, {"word_count": wc, "nominalizations": nom, "hedges": hedge_count, "vague_terms": vague_count, "passive": passive}


def _mk(sev, level, cat, msg, evidence, section_type, p_idx=None, s_idx=None, excerpt=None):
    return Flag(severity=sev, level=level, category=cat, message=msg, evidence=evidence, suggestion="Revise for clarity and specificity.", section_type=section_type, paragraph_index=p_idx, sentence_index=s_idx, text_excerpt=excerpt)
