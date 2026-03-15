#!/usr/bin/env python3
"""Normalize parsed cited references into conservative deduplicated work clusters."""

from __future__ import annotations

import argparse
import hashlib
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Optional

try:
    from rapidfuzz import fuzz
except ImportError as e:
    raise SystemExit(
        "Missing dependency: rapidfuzz. Install with:\n"
        "python3 -m pip install rapidfuzz"
    ) from e

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from unidecode import unidecode


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "i", "if", "in",
    "into", "is", "it", "its", "of", "on", "or", "our", "she", "that", "the",
    "their", "them", "they", "this", "to", "was", "were", "will", "with",
    "within", "without", "you", "your", "we", "us", "not", "can", "could",
    "should", "would", "may", "might", "do", "does", "did", "done", "using",
    "use", "used", "via", "toward", "towards", "between", "among", "than",
    "these", "those", "such", "how", "what", "when", "where", "which", "who",
    "whom", "whose", "also", "there", "here", "because", "while", "during",
    "over", "under", "about", "across", "after", "before", "again", "further",
    "more", "most", "some", "any", "each", "other", "same", "own", "so",
    "very", "s", "t", "d", "ll", "m", "o", "re", "ve", "y",
}

DOI_REGEX = re.compile(r"(?i)\b(10\.\d{4,9}/[-._;()/:a-z0-9]+)\b")
DOI_URL_REGEX = re.compile(r"(?i)https?://(?:dx\.)?doi\.org/\S+")
YEAR_REGEX = re.compile(r"\b(?:18|19|20)\d{2}\b")
PUNCT_REGEX = re.compile(r"[^\w\s]")
WS_REGEX = re.compile(r"\s+")

BRACKET_LABEL_RE = re.compile(r"\[(?:[A-Za-z]{1,3}|\d+\s*paragraphs?)\]")
EDITION_RE = re.compile(
    r"\b(?:\d{1,2}(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+ed(?:ition)?\b",
    re.I,
)
VOLUME_MARKER_RE = re.compile(r"\b(?:vol\.?|volume)\s*\d+\b", re.I)
PAGE_DEBRIS_RE = re.compile(r"\bpp?\.?\s*[A-Za-z]?\d+[A-Za-z]?(?:\s*[-–—]\s*[A-Za-z]?\d+[A-Za-z]?)?\b", re.I)
ISSUE_DEBRIS_RE = re.compile(r"\b(?:\d+\s*\(\s*\d+\s*\)|issue\s*\d+)\b", re.I)
PROCEEDINGS_RE = re.compile(
    r"\b(?:in\s+)?proceedings\s+of\b|\bpaper\s+presented\s+at\b|\bconference\s+on\b|\bsymposium\s+on\b|\bworkshop\s+on\b",
    re.I,
)
VENUE_TAIL_RE = re.compile(
    r"\b(?:journal\s+of|transactions\s+on|interact\.?\s*comput\.?\b|interacting\s+with\s+computers\b|psychol\.?\s*bull\.?\b|acad\.?\s*med\.?\b|organ\.?\s*sci\.?\b|ecol\.?\s*psychol\.?\b|mis\s*q\b|acad\.?\s*manag\.?\s*rev\.?\b|chi\s*['’]?\d{2}|hicss\b|acm\b)\b",
    re.I,
)
TRAILING_SOURCE_FRAGMENT_RE = re.compile(r"(?:[,;:]\s*)?(international|information|european|practice|work)\s*$", re.I)
DOI_CONTAM_RE = re.compile(r"\bdoi\b|doi\.org|10\.\d{4,9}/", re.I)
URL_RE = re.compile(r"https?://\S+", re.I)
DUPLICATED_PHRASE_RE = re.compile(r"\b(.{20,}?)\s+\1\b", re.I)
TITLE_LABEL_TAIL_RE = re.compile(r"\[[^\]]{1,40}\]\s*$")
OCR_HYPHEN_BREAK_RE = re.compile(r"\b([a-z]{3,})\s*[-‐]\s*([a-z]{3,})\b", re.I)
OCR_SPACE_BREAK_RE = re.compile(r"\b([a-z]{1,3})\s+([a-z]{5,})\b", re.I)
LEADING_LABEL_RE = re.compile(r"^(?:introduction|foreword|preface|editor'?s?\s+comments?|editorial|guest\s+editorial|commentary)\b", re.I)
AUTHOR_LEAK_RE = re.compile(r",\s*[A-Z][A-Za-z'\-]+\s+[A-Z](?:\.[A-Z])?\.?\s*(?:;\s*[A-Z][A-Za-z'\-]+\s+[A-Z](?:\.[A-Z])?\.?\s*)+$")
CONSERVATIVE_JOIN_MAP = {
    "af fordances": "affordances",
    "aff ordances": "affordances",
    "concep tual": "conceptual",
    "lan guage": "language",
    "coll ision": "collision",
    "a case": "acase",
    "pyschology": "psychology",
}
SUSPICIOUS_TITLE_TOKENS = {
    "doi", "proceedings", "conference", "symposium", "workshop", "journal",
    "transactions", "vol", "volume", "pp", "pages", "presented", "http", "acm", "international",
}
OCR_SPLIT_RE = re.compile(r"\b([a-z]{2,})\s*[-‐]\s+([a-z]{2,})\b")

DEFAULT_AUTO_TITLE_THRESHOLD = 92
DEFAULT_REVIEW_TITLE_MIN = 85
DEFAULT_TITLE_OVERLAP_MIN = 0.50
CONTAINMENT_AUTO_SIM_MIN = 90
CONTAINMENT_AUTO_OVERLAP_MIN = 0.85

INTERNAL_CITATION_TITLE_OVERLAP_MIN = 0.50
INTERNAL_CITATION_TITLE_SIM_MIN = 92

UK_US_MAP = {
    "behaviour": "behavior",
    "organisation": "organization",
    "organisations": "organizations",
    "organise": "organize",
    "organised": "organized",
    "colour": "color",
    "labour": "labor",
    "centre": "center",
    "modelling": "modeling",
    "analyse": "analyze",
    "analysed": "analyzed",
    "catalogue": "catalog",
}


VENUE_ABBREV_MAP = {
    "psychol bull": "psychological bulletin",
    "acad med": "academic medicine",
    "organ sci": "organization science",
    "ecol psychol": "ecological psychology",
    "interact comput": "interacting with computers",
}


@dataclass
class RefRecord:
    source_ref_id: Any
    citing_doc_id: Any
    raw_title: str
    raw_first_author: str
    raw_year: Optional[int]
    raw_doi: str
    title_display_norm: str
    title_display_tokens: list[str]
    title_norm: str
    title_tokens: list[str]
    work_title_norm: str
    work_title_tokens: list[str]
    title_key8: str
    work_key8: str
    first_author_norm: str
    year: Optional[int]
    doi_norm: str
    exact_key: str
    work_exact_key: str


class DSU:
    def __init__(self, items: Iterable[Any]) -> None:
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: Any) -> Any:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: Any, b: Any) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def as_str_id(value: Any) -> str:
    return str(value).strip()


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    normalized = unidecode(str(text)).lower()
    normalized = PUNCT_REGEX.sub(" ", normalized)
    normalized = WS_REGEX.sub(" ", normalized).strip()
    return normalized


def normalize_display_title(title: Optional[str]) -> tuple[str, list[str]]:
    text = normalize_text(title)
    if not text:
        return "", []
    text = strip_doi_fragments(text)
    text = AUTHOR_LEAK_RE.sub("", text).strip(" ,;:")
    text = LEADING_LABEL_RE.sub("", text).strip(" ,;:")
    text = TITLE_LABEL_TAIL_RE.sub(" ", text)
    text = TRAILING_SOURCE_FRAGMENT_RE.sub("", text)
    text = WS_REGEX.sub(" ", text).strip()
    tokens = [tok for tok in text.split() if tok]
    return " ".join(tokens), tokens


def normalize_title(title: Optional[str]) -> tuple[str, list[str]]:
    text, _ = normalize_display_title(title)
    if not text:
        return "", []
    tokens = [tok for tok in text.split() if tok and tok not in STOPWORDS]
    return " ".join(tokens), tokens


def strip_doi_fragments(text: str) -> str:
    text = DOI_URL_REGEX.sub(" ", text)
    text = re.sub(r"\bdoi\s*[:=]?\s*", " ", text, flags=re.I)
    text = DOI_REGEX.sub(" ", text)
    return text


def strip_source_tail(text: str) -> str:
    """Remove source/container tails with strong metadata evidence."""
    cleaned = normalize_text(text)
    if not cleaned:
        return ""

    parts = [p.strip() for p in re.split(r"\s*[,;]\s*", cleaned) if p.strip()]
    if len(parts) <= 1:
        return TRAILING_SOURCE_FRAGMENT_RE.sub("", cleaned).strip(" ,;:")

    kept: list[str] = []
    for idx, part in enumerate(parts):
        if idx > 0 and (PROCEEDINGS_RE.search(part) or VENUE_TAIL_RE.search(part)):
            break
        if idx > 0 and re.search(r"\b(?:vol\.?|volume|pp?\.?)\b", part, re.I):
            break
        if idx > 0 and ISSUE_DEBRIS_RE.search(part):
            break
        if idx > 0 and TRAILING_SOURCE_FRAGMENT_RE.fullmatch(part):
            break
        kept.append(part)

    candidate = ", ".join(kept) if kept else parts[0]
    candidate = TRAILING_SOURCE_FRAGMENT_RE.sub("", candidate).strip(" ,;:")
    return candidate


def repair_ocr_breaks(text: str) -> str:
    """Repair OCR-only splits conservatively; never join arbitrary compounds."""
    def _join_hyphen_break(match: re.Match[str]) -> str:
        left, right = match.group(1).lower(), match.group(2).lower()
        pair = f"{left} {right}"
        return CONSERVATIVE_JOIN_MAP.get(pair, match.group(0))

    text = OCR_HYPHEN_BREAK_RE.sub(_join_hyphen_break, text)

    def _repair_space_break(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        pair = f"{left.lower()} {right.lower()}"
        if pair in CONSERVATIVE_JOIN_MAP:
            return CONSERVATIVE_JOIN_MAP[pair]
        return match.group(0)

    text = OCR_SPACE_BREAK_RE.sub(_repair_space_break, text)
    text = re.sub(r"\bpyschology\b", "psychology", text, flags=re.I)
    text = re.sub(r"\blan\s*[-‐]\s*guage\b", "language", text, flags=re.I)
    return text


def normalize_spelling_for_match(text: str) -> str:
    tokens = text.split()
    mapped = [UK_US_MAP.get(token, token) for token in tokens]
    return " ".join(mapped)


def normalize_work_title(title: Optional[str]) -> tuple[str, list[str]]:
    """Aggressive work-level normalization used for clustering keys and matching only."""
    if not title:
        return "", []

    text = unidecode(str(title)).lower()
    text = strip_doi_fragments(text)
    text = URL_RE.sub(" ", text)
    text = AUTHOR_LEAK_RE.sub("", text)
    text = BRACKET_LABEL_RE.sub(" ", text)
    text = TITLE_LABEL_TAIL_RE.sub(" ", text)
    text = LEADING_LABEL_RE.sub(" ", text)
    text = strip_source_tail(text)
    text = EDITION_RE.sub(" ", text)
    text = VOLUME_MARKER_RE.sub(" ", text)
    text = PAGE_DEBRIS_RE.sub(" ", text)
    text = ISSUE_DEBRIS_RE.sub(" ", text)
    text = PROCEEDINGS_RE.sub(" ", text)
    text = VENUE_TAIL_RE.sub(" ", text)
    for short, long_form in VENUE_ABBREV_MAP.items():
        text = re.sub(rf"\b{re.escape(short)}\b", long_form, text)
    text = repair_ocr_breaks(text)

    text = TRAILING_SOURCE_FRAGMENT_RE.sub("", text)
    text = PUNCT_REGEX.sub(" ", text)
    text = normalize_spelling_for_match(text)
    text = re.sub(r"\bwell\s+being\b", "wellbeing", text)
    text = WS_REGEX.sub(" ", text).strip()

    tokens = [tok for tok in text.split() if tok and tok not in STOPWORDS]
    return " ".join(tokens), tokens


def normalize_author(author: Optional[str]) -> str:
    text = normalize_text(author)
    if not text:
        return ""
    text = re.split(r"\b(?:and|et al)\b|,|&", text)[0].strip()
    parts = text.split()
    return parts[-1] if parts else ""


def normalize_doi(doi: Optional[str]) -> str:
    if not doi:
        return ""
    doi_text = unidecode(str(doi)).strip().lower()
    doi_text = doi_text.replace("https://doi.org/", "").replace("http://doi.org/", "")
    doi_text = doi_text.replace("doi:", "").strip()
    match = DOI_REGEX.search(doi_text)
    return match.group(1).lower().rstrip(" .;,)") if match else doi_text.rstrip(" .;,)")


def safe_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value if 1800 <= value <= 2100 else None
    text = str(value).strip()
    if not text:
        return None
    match = YEAR_REGEX.search(text)
    if not match:
        return None
    year = int(match.group(0))
    return year if 1800 <= year <= 2100 else None


def token_overlap(tokens_a: list[str], tokens_b: list[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = len(set_a & set_b)
    denominator = min(len(set_a), len(set_b))
    return (intersection / denominator) if denominator else 0.0


def stable_cluster_id(seed: str) -> str:
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
    return f"refcl_{digest}"


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def fetch_table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({quote_ident(table_name)})").fetchall()
    return [row[1] for row in rows]


def resolve_column(columns: list[str], candidates: list[str], table_name: str) -> str:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise ValueError(
        f"Could not resolve required column in table '{table_name}'. "
        f"Tried candidates: {candidates}. Available columns: {columns}"
    )


def ensure_output_tables(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.executescript(
            """
            DROP TABLE IF EXISTS internal_citations;
            DROP TABLE IF EXISTS reference_cluster_review;
            DROP TABLE IF EXISTS document_reference_links;
            DROP TABLE IF EXISTS reference_clusters;

            CREATE TABLE reference_clusters (
                cluster_id TEXT PRIMARY KEY,
                canonical_title TEXT,
                canonical_first_author TEXT,
                canonical_year INTEGER,
                canonical_source_title TEXT,
                canonical_doi TEXT,
                cluster_confidence TEXT,
                cluster_method TEXT,
                member_count INTEGER,
                source_ref_ids TEXT
            );

            CREATE TABLE document_reference_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                source_ref_id TEXT,
                cluster_id TEXT,
                link_method TEXT,
                link_confidence TEXT,
                FOREIGN KEY(cluster_id) REFERENCES reference_clusters(cluster_id)
            );

            CREATE TABLE reference_cluster_review (
                review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ref_id_1 TEXT,
                ref_id_2 TEXT,
                cluster_id_1 TEXT,
                cluster_id_2 TEXT,
                first_author_norm TEXT,
                year_1 INTEGER,
                year_2 INTEGER,
                title_1 TEXT,
                title_2 TEXT,
                title_similarity REAL,
                title_overlap REAL,
                suggested_action TEXT,
                reason TEXT
            );

            CREATE TABLE internal_citations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                citing_doc_id TEXT,
                cited_doc_id TEXT,
                cluster_id TEXT,
                match_method TEXT,
                match_confidence TEXT
            );

            CREATE INDEX idx_document_reference_links_doc_id
                ON document_reference_links(doc_id);

            CREATE INDEX idx_document_reference_links_cluster_id
                ON document_reference_links(cluster_id);

            CREATE INDEX idx_reference_clusters_canonical_doi
                ON reference_clusters(canonical_doi);

            CREATE INDEX idx_internal_citations_citing_doc_id
                ON internal_citations(citing_doc_id);

            CREATE INDEX idx_internal_citations_cited_doc_id
                ON internal_citations(cited_doc_id);
            """
        )
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def title_noise_score(raw_title: str) -> int:
    normalized = normalize_text(raw_title)
    if not normalized:
        return 100
    penalty = 0
    tokens = normalized.split()
    suspicious = sum(1 for token in tokens if token in SUSPICIOUS_TITLE_TOKENS)
    penalty += suspicious * 3
    penalty += 4 if DOI_REGEX.search(raw_title) or DOI_URL_REGEX.search(raw_title) else 0
    penalty += 2 if EDITION_RE.search(raw_title) else 0
    penalty += 2 if BRACKET_LABEL_RE.search(raw_title) else 0
    penalty += 2 if re.search(r"\b\w+[-‐]\s+\w+\b", raw_title) else 0
    penalty += 1 if PROCEEDINGS_RE.search(raw_title) else 0
    penalty += 1 if VENUE_TAIL_RE.search(raw_title) else 0
    penalty += 4 if DOI_CONTAM_RE.search(raw_title) else 0
    penalty += 3 if URL_RE.search(raw_title) else 0
    penalty += 3 if TRAILING_SOURCE_FRAGMENT_RE.search(raw_title or "") else 0
    penalty += 4 if DUPLICATED_PHRASE_RE.search(normalized) else 0
    penalty += 2 if OCR_HYPHEN_BREAK_RE.search(raw_title) else 0
    penalty += 5 if AUTHOR_LEAK_RE.search(raw_title) else 0
    penalty += 4 if LEADING_LABEL_RE.search(normalized) else 0
    penalty += 4 if re.search(r"\b(?:pp?\.?\s*\d+|vol\.?\s*\d+|issue\s*\d+)\b", normalized) else 0
    return penalty


def choose_canonical(records: list[RefRecord]) -> RefRecord:
    """Choose the cleanest representative, not the longest string."""

    def score(record: RefRecord) -> tuple[int, int, int, int, int, int, int, int]:
        noise = title_noise_score(record.raw_title)
        return (
            1 if record.year is not None and record.first_author_norm else 0,
            1 if record.doi_norm and not DOI_CONTAM_RE.search(record.raw_title) else 0,
            1 if record.title_display_norm and not LEADING_LABEL_RE.search(record.title_display_norm) else 0,
            -noise,
            -len(record.work_title_tokens),
            -len(record.title_display_tokens),
            -len(record.title_tokens),
            len(record.raw_first_author or ""),
        )

    return sorted(records, key=score, reverse=True)[0]


def cluster_confidence_for_members(records: list[RefRecord]) -> tuple[str, str]:
    if any(record.doi_norm for record in records):
        return "high", "doi_exact"
    exact_keys = {record.work_exact_key for record in records if record.work_exact_key}
    if len(exact_keys) == 1 and exact_keys:
        return "high", "work_exact_key"
    return "medium", "fuzzy_title"


def detect_source_rows(conn: sqlite3.Connection, refs_table: str) -> list[dict[str, Any]]:
    columns = fetch_table_columns(conn, refs_table)

    ref_id_col = resolve_column(columns, ["cited_ref_id", "reference_id", "id", "raw_ref_id"], refs_table)
    doc_id_col = resolve_column(columns, ["doc_id", "source_doc_id", "document_id", "citing_doc_id"], refs_table)
    title_col = resolve_column(columns, ["title_candidate", "title", "ref_title", "parsed_title"], refs_table)
    author_col = resolve_column(columns, ["first_author", "author", "ref_first_author", "parsed_first_author"], refs_table)
    year_col = resolve_column(columns, ["year", "ref_year", "parsed_year"], refs_table)
    doi_col = resolve_column(columns, ["doi", "ref_doi", "parsed_doi"], refs_table)

    sql = f"""
        SELECT
            {quote_ident(ref_id_col)} AS source_ref_id,
            {quote_ident(doc_id_col)} AS citing_doc_id,
            {quote_ident(title_col)} AS raw_title,
            {quote_ident(author_col)} AS raw_first_author,
            {quote_ident(year_col)} AS raw_year,
            {quote_ident(doi_col)} AS raw_doi
        FROM {quote_ident(refs_table)}
    """
    cursor = conn.execute(sql)
    names = [description[0] for description in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def build_ref_records(rows: list[dict[str, Any]]) -> list[RefRecord]:
    records: list[RefRecord] = []

    for row in rows:
        raw_title = str(row.get("raw_title") or "").strip()
        raw_first_author = str(row.get("raw_first_author") or "").strip()
        raw_doi = str(row.get("raw_doi") or "").strip()
        year = safe_year(row.get("raw_year"))

        doi_norm = normalize_doi(raw_doi)
        title_display_norm, title_display_tokens = normalize_display_title(raw_title)
        title_norm, title_tokens = normalize_title(raw_title)
        work_title_norm, work_title_tokens = normalize_work_title(raw_title)
        first_author_norm = normalize_author(raw_first_author)
        title_key8 = "_".join(title_tokens[:8]) if title_tokens else ""
        work_key8 = "_".join(work_title_tokens[:8]) if work_title_tokens else ""

        exact_key = ""
        if first_author_norm and year and title_key8:
            exact_key = f"{first_author_norm}_{year}_{title_key8}"

        work_exact_key = ""
        if first_author_norm and year and work_key8:
            work_exact_key = f"{first_author_norm}_{year}_{work_key8}"

        records.append(
            RefRecord(
                source_ref_id=row["source_ref_id"],
                citing_doc_id=row["citing_doc_id"],
                raw_title=raw_title,
                raw_first_author=raw_first_author,
                raw_year=year,
                raw_doi=raw_doi,
                title_display_norm=title_display_norm,
                title_display_tokens=title_display_tokens,
                title_norm=title_norm,
                title_tokens=title_tokens,
                work_title_norm=work_title_norm,
                work_title_tokens=work_title_tokens,
                title_key8=title_key8,
                work_key8=work_key8,
                first_author_norm=first_author_norm,
                year=year,
                doi_norm=doi_norm,
                exact_key=exact_key,
                work_exact_key=work_exact_key,
            )
        )

    return records


def apply_tier1_doi(records: list[RefRecord], dsu: DSU) -> dict[Any, str]:
    methods: dict[Any, str] = {}
    by_doi: dict[str, list[RefRecord]] = defaultdict(list)

    for record in records:
        if record.doi_norm:
            by_doi[record.doi_norm].append(record)

    for members in by_doi.values():
        if len(members) < 2:
            continue
        anchor = members[0].source_ref_id
        methods[anchor] = "doi_exact"
        for member in members[1:]:
            dsu.union(anchor, member.source_ref_id)
            methods[member.source_ref_id] = "doi_exact"

    return methods


def apply_tier2_exact_key(records: list[RefRecord], dsu: DSU, methods: dict[Any, str]) -> None:
    by_key: dict[str, list[RefRecord]] = defaultdict(list)

    for record in records:
        if record.doi_norm:
            continue
        if record.work_exact_key:
            by_key[record.work_exact_key].append(record)

    for members in by_key.values():
        if len(members) < 2:
            continue
        anchor = members[0].source_ref_id
        methods.setdefault(anchor, "work_exact_key")
        for member in members[1:]:
            dsu.union(anchor, member.source_ref_id)
            methods.setdefault(member.source_ref_id, "work_exact_key")


def is_containment_variant(rec_a: RefRecord, rec_b: RefRecord) -> bool:
    """Return True when one work title is mostly another plus metadata debris."""
    if not rec_a.work_title_norm or not rec_b.work_title_norm:
        return False

    shorter, longer = (rec_a, rec_b)
    if len(rec_a.work_title_tokens) > len(rec_b.work_title_tokens):
        shorter, longer = rec_b, rec_a

    if len(shorter.work_title_tokens) < 4:
        return False

    overlap = token_overlap(shorter.work_title_tokens, longer.work_title_tokens)
    if overlap < CONTAINMENT_AUTO_OVERLAP_MIN:
        return False

    sim = float(fuzz.partial_ratio(shorter.work_title_norm, longer.work_title_norm))
    if sim < CONTAINMENT_AUTO_SIM_MIN:
        return False

    longer_extra = set(longer.work_title_tokens) - set(shorter.work_title_tokens)
    if len(longer_extra) > max(6, len(shorter.work_title_tokens) // 2):
        return False

    return True


def apply_tier3_fuzzy(
    conn: sqlite3.Connection,
    records: list[RefRecord],
    dsu: DSU,
    methods: dict[Any, str],
    auto_title_threshold: int,
    review_title_min: int,
    title_overlap_min: float,
) -> None:
    grouped: dict[tuple[str, int], list[RefRecord]] = defaultdict(list)

    for record in records:
        if not record.first_author_norm or record.year is None or not record.work_title_norm:
            continue
        grouped[(record.first_author_norm, record.year)].append(record)

    review_rows: list[tuple[Any, ...]] = []

    for (_, _), group in tqdm(grouped.items(), desc="Tier 3 fuzzy"):
        if len(group) < 2:
            continue

        rep_to_record: dict[Any, RefRecord] = {}
        for record in group:
            root = dsu.find(record.source_ref_id)
            current = rep_to_record.get(root)
            rep_to_record[root] = record if current is None else choose_canonical([current, record])

        reps = list(rep_to_record.items())
        for i in range(len(reps)):
            root_i, rec_i = reps[i]
            for j in range(i + 1, len(reps)):
                root_j, rec_j = reps[j]

                if root_i == root_j:
                    continue

                overlap = token_overlap(rec_i.work_title_tokens, rec_j.work_title_tokens)
                if overlap < title_overlap_min:
                    continue

                similarity = float(fuzz.token_sort_ratio(rec_i.work_title_norm, rec_j.work_title_norm))
                containment = is_containment_variant(rec_i, rec_j)

                if similarity >= auto_title_threshold or containment:
                    if dsu.union(root_i, root_j):
                        method = "containment_title" if containment and similarity < auto_title_threshold else "fuzzy_work_title"
                        methods.setdefault(rec_i.source_ref_id, method)
                        methods.setdefault(rec_j.source_ref_id, method)
                elif review_title_min <= similarity < auto_title_threshold:
                    if overlap >= 0.80 and similarity >= review_title_min + 2:
                        if dsu.union(root_i, root_j):
                            methods.setdefault(rec_i.source_ref_id, "high_overlap_fuzzy")
                            methods.setdefault(rec_j.source_ref_id, "high_overlap_fuzzy")
                        continue
                    review_rows.append(
                        (
                            as_str_id(rec_i.source_ref_id),
                            as_str_id(rec_j.source_ref_id),
                            None,
                            None,
                            rec_i.first_author_norm,
                            rec_i.year,
                            rec_j.year,
                            rec_i.raw_title,
                            rec_j.raw_title,
                            similarity,
                            float(overlap),
                            "manual_review",
                            "same_author_same_year_work_title_ambiguous",
                        )
                    )

    if review_rows:
        conn.executemany(
            """
            INSERT INTO reference_cluster_review (
                ref_id_1, ref_id_2, cluster_id_1, cluster_id_2,
                first_author_norm, year_1, year_2,
                title_1, title_2, title_similarity, title_overlap,
                suggested_action, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            review_rows,
        )
        conn.commit()


def materialize_clusters(
    conn: sqlite3.Connection,
    records: list[RefRecord],
    dsu: DSU,
    methods: dict[Any, str],
) -> dict[str, str]:
    clusters: dict[Any, list[RefRecord]] = defaultdict(list)

    for record in records:
        clusters[dsu.find(record.source_ref_id)].append(record)

    source_to_cluster: dict[str, str] = {}
    cluster_rows: list[tuple[Any, ...]] = []
    link_rows: list[tuple[Any, ...]] = []
    used_cluster_ids: set[str] = set()

    def build_cluster_seed(canonical: RefRecord, root: Any) -> str:
        if canonical.doi_norm:
            return f"doi::{canonical.doi_norm}"
        if canonical.work_exact_key:
            return f"key::{canonical.work_exact_key}"
        if canonical.first_author_norm and canonical.year and canonical.work_title_norm:
            return f"meta::{canonical.first_author_norm}|{canonical.year}|{canonical.work_title_norm}"
        return f"root::{as_str_id(root)}"

    def make_unique_cluster_id(seed: str, root: Any) -> str:
        base_id = stable_cluster_id(seed)
        if base_id not in used_cluster_ids:
            used_cluster_ids.add(base_id)
            return base_id

        fallback_id = stable_cluster_id(f"{seed}||root::{as_str_id(root)}")
        if fallback_id not in used_cluster_ids:
            used_cluster_ids.add(fallback_id)
            return fallback_id

        counter = 2
        while True:
            candidate = stable_cluster_id(f"{seed}||root::{as_str_id(root)}||n::{counter}")
            if candidate not in used_cluster_ids:
                used_cluster_ids.add(candidate)
                return candidate
            counter += 1

    for root, members in clusters.items():
        canonical = choose_canonical(members)
        seed = build_cluster_seed(canonical, root)
        cluster_id = make_unique_cluster_id(seed, root)

        confidence, cluster_method = cluster_confidence_for_members(members)
        member_ids = sorted(as_str_id(member.source_ref_id) for member in members)

        cluster_rows.append(
            (
                cluster_id,
                canonical.title_display_norm or canonical.raw_title or canonical.title_norm,
                canonical.raw_first_author or canonical.first_author_norm,
                canonical.year,
                None,
                canonical.doi_norm or None,
                confidence,
                cluster_method,
                len(members),
                "|".join(member_ids),
            )
        )

        for member in members:
            source_ref_id = as_str_id(member.source_ref_id)
            citing_doc_id = as_str_id(member.citing_doc_id)
            source_to_cluster[source_ref_id] = cluster_id

            link_method = methods.get(member.source_ref_id, cluster_method)
            link_confidence = "high" if link_method in {"doi_exact", "work_exact_key"} else confidence

            link_rows.append(
                (
                    citing_doc_id,
                    source_ref_id,
                    cluster_id,
                    link_method,
                    link_confidence,
                )
            )

    cluster_id_counts = Counter(row[0] for row in cluster_rows)
    duplicate_cluster_ids = {cid: n for cid, n in cluster_id_counts.items() if n > 1}
    if duplicate_cluster_ids:
        raise RuntimeError(
            f"Duplicate cluster_ids generated before insert: {list(duplicate_cluster_ids.items())[:10]}"
        )

    conn.executemany(
        """
        INSERT INTO reference_clusters (
            cluster_id, canonical_title, canonical_first_author, canonical_year,
            canonical_source_title, canonical_doi, cluster_confidence, cluster_method,
            member_count, source_ref_ids
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        cluster_rows,
    )

    conn.executemany(
        """
        INSERT INTO document_reference_links (
            doc_id, source_ref_id, cluster_id, link_method, link_confidence
        ) VALUES (?, ?, ?, ?, ?)
        """,
        link_rows,
    )

    conn.commit()
    return source_to_cluster


def update_review_cluster_ids(conn: sqlite3.Connection, source_to_cluster: dict[str, str]) -> None:
    rows = conn.execute(
        "SELECT review_id, ref_id_1, ref_id_2 FROM reference_cluster_review"
    ).fetchall()

    updates: list[tuple[Any, ...]] = []
    for review_id, ref_id_1, ref_id_2 in rows:
        cluster_id_1 = source_to_cluster.get(as_str_id(ref_id_1))
        cluster_id_2 = source_to_cluster.get(as_str_id(ref_id_2))
        updates.append((cluster_id_1, cluster_id_2, review_id))

    if updates:
        conn.executemany(
            """
            UPDATE reference_cluster_review
            SET cluster_id_1 = ?, cluster_id_2 = ?
            WHERE review_id = ?
            """,
            updates,
        )
        conn.commit()


def detect_document_rows(conn: sqlite3.Connection, docs_table: str) -> list[dict[str, Any]]:
    columns = fetch_table_columns(conn, docs_table)

    doc_id_col = resolve_column(columns, ["doc_id", "document_id", "id"], docs_table)
    title_col = resolve_column(columns, ["title", "document_title", "article_title"], docs_table)
    year_col = resolve_column(columns, ["year", "pub_year", "publication_year"], docs_table)

    doi_col = None
    lowered = {column.lower(): column for column in columns}
    for candidate in ["doi", "document_doi", "article_doi"]:
        if candidate.lower() in lowered:
            doi_col = lowered[candidate.lower()]
            break

    select_parts = [
        f"{quote_ident(doc_id_col)} AS doc_id",
        f"{quote_ident(title_col)} AS raw_title",
        f"{quote_ident(year_col)} AS raw_year",
    ]
    select_parts.append(f"{quote_ident(doi_col)} AS raw_doi" if doi_col else "NULL AS raw_doi")

    sql = f"SELECT {', '.join(select_parts)} FROM {quote_ident(docs_table)}"
    cursor = conn.execute(sql)
    names = [description[0] for description in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def build_internal_citations(conn: sqlite3.Connection, docs_table: str, skip_internal_citations: bool) -> None:
    if skip_internal_citations:
        return

    cluster_rows = conn.execute(
        """
        SELECT cluster_id, canonical_title, canonical_year, canonical_doi
        FROM reference_clusters
        """
    ).fetchall()

    doc_rows = detect_document_rows(conn, docs_table)

    doi_to_docs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    year_title_docs: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for row in doc_rows:
        title_norm, title_tokens = normalize_title(row.get("raw_title"))
        doc = {
            "doc_id": as_str_id(row["doc_id"]),
            "title_norm": title_norm,
            "title_tokens": title_tokens,
            "year": safe_year(row.get("raw_year")),
            "doi_norm": normalize_doi(row.get("raw_doi")),
        }

        if doc["doi_norm"]:
            doi_to_docs[doc["doi_norm"]].append(doc)
        if doc["year"] is not None and doc["title_norm"]:
            year_title_docs[doc["year"]].append(doc)

    cluster_match_rows: list[tuple[Any, ...]] = []

    for cluster_id, canonical_title, canonical_year, canonical_doi in cluster_rows:
        matched_docs: list[tuple[str, str, str]] = []

        doi_norm = normalize_doi(canonical_doi)
        title_norm, title_tokens = normalize_title(canonical_title)

        if doi_norm and doi_norm in doi_to_docs:
            for doc in doi_to_docs[doi_norm]:
                matched_docs.append((doc["doc_id"], "doi", "high"))
        elif canonical_year is not None and title_norm:
            best_doc: Optional[dict[str, Any]] = None
            best_similarity = -1.0

            for doc in year_title_docs.get(canonical_year, []):
                overlap = token_overlap(title_tokens, doc["title_tokens"])
                if overlap < INTERNAL_CITATION_TITLE_OVERLAP_MIN:
                    continue

                similarity = float(fuzz.token_sort_ratio(title_norm, doc["title_norm"]))
                if similarity > best_similarity:
                    best_doc = doc
                    best_similarity = similarity

            if best_doc is not None and best_similarity >= INTERNAL_CITATION_TITLE_SIM_MIN:
                matched_docs.append((best_doc["doc_id"], "title_year", "medium"))

        if not matched_docs:
            continue

        citing_rows = conn.execute(
            """
            SELECT DISTINCT doc_id
            FROM document_reference_links
            WHERE cluster_id = ?
            """,
            (cluster_id,),
        ).fetchall()

        for cited_doc_id, match_method, match_confidence in matched_docs:
            for (citing_doc_id,) in citing_rows:
                citing_doc_id_str = as_str_id(citing_doc_id)
                cited_doc_id_str = as_str_id(cited_doc_id)
                if citing_doc_id_str == cited_doc_id_str:
                    continue
                cluster_match_rows.append(
                    (
                        citing_doc_id_str,
                        cited_doc_id_str,
                        cluster_id,
                        match_method,
                        match_confidence,
                    )
                )

    if cluster_match_rows:
        deduped_rows = list(dict.fromkeys(cluster_match_rows))
        conn.executemany(
            """
            INSERT INTO internal_citations (
                citing_doc_id, cited_doc_id, cluster_id, match_method, match_confidence
            ) VALUES (?, ?, ?, ?, ?)
            """,
            deduped_rows,
        )
        conn.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize cited references into conservative deduplicated works."
    )
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--refs-table", default="cited_references", help="Parsed cited references table")
    parser.add_argument("--docs-table", default="documents", help="Corpus documents table")
    parser.add_argument(
        "--auto-title-threshold",
        type=int,
        default=DEFAULT_AUTO_TITLE_THRESHOLD,
        help="Fuzzy title similarity threshold for auto-clustering",
    )
    parser.add_argument(
        "--review-title-min",
        type=int,
        default=DEFAULT_REVIEW_TITLE_MIN,
        help="Lower fuzzy title similarity threshold for manual review",
    )
    parser.add_argument(
        "--title-overlap-min",
        type=float,
        default=DEFAULT_TITLE_OVERLAP_MIN,
        help="Minimum title token overlap for fuzzy candidate comparison",
    )
    parser.add_argument(
        "--skip-internal-citations",
        action="store_true",
        help="Skip optional matching of normalized references back to corpus documents",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.review_title_min > args.auto_title_threshold:
        raise ValueError("review-title-min cannot be greater than auto-title-threshold")

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        ensure_output_tables(conn)

        source_rows = detect_source_rows(conn, args.refs_table)
        if not source_rows:
            print(f"No rows found in {args.refs_table}", file=sys.stderr)
            return 1

        records = build_ref_records(source_rows)
        dsu = DSU([record.source_ref_id for record in records])

        methods = apply_tier1_doi(records, dsu)
        apply_tier2_exact_key(records, dsu, methods)
        apply_tier3_fuzzy(
            conn=conn,
            records=records,
            dsu=dsu,
            methods=methods,
            auto_title_threshold=args.auto_title_threshold,
            review_title_min=args.review_title_min,
            title_overlap_min=args.title_overlap_min,
        )

        source_to_cluster = materialize_clusters(conn, records, dsu, methods)
        update_review_cluster_ids(conn, source_to_cluster)
        build_internal_citations(
            conn=conn,
            docs_table=args.docs_table,
            skip_internal_citations=args.skip_internal_citations,
        )

        counts = {
            "input_references": conn.execute(
                f"SELECT COUNT(*) FROM {quote_ident(args.refs_table)}"
            ).fetchone()[0],
            "reference_clusters": conn.execute(
                "SELECT COUNT(*) FROM reference_clusters"
            ).fetchone()[0],
            "document_reference_links": conn.execute(
                "SELECT COUNT(*) FROM document_reference_links"
            ).fetchone()[0],
            "review_pairs": conn.execute(
                "SELECT COUNT(*) FROM reference_cluster_review"
            ).fetchone()[0],
            "internal_citations": conn.execute(
                "SELECT COUNT(*) FROM internal_citations"
            ).fetchone()[0],
        }

        print("Normalization complete")
        for key, value in counts.items():
            print(f"{key}: {value}")

        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
