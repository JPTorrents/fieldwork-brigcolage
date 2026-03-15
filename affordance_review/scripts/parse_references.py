#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional

from unidecode import unidecode


DB_PATH = Path("data/processed/affordance_lit.sqlite")

DOI_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)
DOI_URL_RE = re.compile(r"https?://(?:dx\.)?doi\.org/\S+", re.I)
DOI_PREFIX_RE = re.compile(r"\bdoi\s*[:=]?\s*(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)
YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
PAREN_YEAR_RE = re.compile(r"\((19\d{2}|20\d{2})\)")
PAGES_RE = re.compile(
    r"\bpp?\.\s*([A-Za-z0-9]+(?:\s*[-–—]\s*[A-Za-z0-9]+)?)",
    re.I,
)
VOL_ISSUE_RE = re.compile(r",\s*(\d{1,4})\s*,\s*([A-Za-z0-9\-]{1,20})(?=,|\s*\(|\s*$)")
VOL_ONLY_RE = re.compile(r",\s*(\d{1,4})(?=,?\s*pp?\.|,?\s*\(|\s*$)", re.I)
SPACE_RE = re.compile(r"\s+")
TRAILING_PUNCT_RE = re.compile(r"[\s,;:.]+$")
EDITION_RE = re.compile(
    r"\b(?:\d{1,2}(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+ed(?:ition)?\b",
    re.I,
)
VENUE_SPLIT_HINT_RE = re.compile(
    r"\b(?:proceedings\s+of|paper\s+presented\s+at|conference\s+on|international\s+conference|symposium\s+on|workshop\s+on|journal\s+of|transactions\s+on|mis\s+q\b|acad\.?\s*manag\.?\s*rev\.?\b|chi\s*['’]?\d{2})\b",
    re.I,
)
SOURCE_BOILERPLATE_RE = re.compile(
    r"\b(?:proceedings\s+of\s+the|in\s+proceedings\s+of|paper\s+presented\s+at|conference\s+on|symposium\s+on|workshop\s+on|journal\s+of|transactions\s+on|mis\s+q\.?|acad\.?\s*manag\.?\s*rev\.?)\b",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse raw references into cited_references fields.")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Path to SQLite database")
    return parser.parse_args()


def normalize_whitespace(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def text_norm(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = unidecode(text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_doi(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = text.strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    text = re.sub(r"[\]\)\.,;:]+$", "", text)
    return text or None


def normalize_ref_string(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = unidecode(text.lower())
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\bdoi:\s*", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = normalize_whitespace(text)
    return text or None


def strip_doi_noise(raw: str) -> str:
    """Remove DOI URLs/fragments so they do not pollute title/source extraction."""
    cleaned = DOI_URL_RE.sub(" ", raw)
    cleaned = DOI_PREFIX_RE.sub(" ", cleaned)
    cleaned = DOI_RE.sub(" ", cleaned)
    return normalize_whitespace(cleaned)


def extract_doi(raw: str) -> Optional[str]:
    m = DOI_RE.search(raw)
    return normalize_doi(m.group(1)) if m else None


def extract_year(raw: str) -> Optional[int]:
    paren_years = [int(y) for y in PAREN_YEAR_RE.findall(raw)]
    if paren_years:
        return paren_years[-1]
    years = [int(y) for y in YEAR_RE.findall(raw)]
    return years[-1] if years else None


def strip_doi(raw: str) -> str:
    return strip_doi_noise(raw)


def strip_reference_prefix_noise(raw: str) -> str:
    """Drop leading labels and boilerplate that are not bibliographic content."""
    text = raw.strip()
    text = re.sub(r"^\s*\[\d+\]\s*", "", text)
    text = re.sub(r"^\s*\(?\d+\)?[\.)]\s*", "", text)
    text = re.sub(r"^[\"'“”‘’]\s*", "", text)
    text = re.sub(r"\s*[\"'“”‘’]\s*$", "", text)
    return normalize_whitespace(text)


def strip_terminal_year_block(raw: str) -> tuple[str, Optional[int]]:
    s = raw.strip()

    m = re.search(r",?\s*\((19\d{2}|20\d{2})\)\s*$", s)
    if m:
        return s[:m.start()].strip(" ,;"), int(m.group(1))

    m = re.search(r",?\s*(19\d{2}|20\d{2})\s*$", s)
    if m:
        return s[:m.start()].strip(" ,;"), int(m.group(1))

    return s, extract_year(s)


def looks_like_author_block(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    if ";" in s:
        return True
    patterns = [
        r"^[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+(?:\s+[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+)*\s+[A-Z](?:\.[A-Z])?\.?$",
        r"^[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+(?:\s+[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+)*,\s*[A-Z](?:\.[A-Z])?\.?$",
        r"^[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+(?:\s+[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+)*\s+[A-Z](?:\.[A-Z])?\.?(?:\s*;\s*[A-ZÀ-ÖØ-Ý].+)?$",
    ]
    return any(re.search(p, s) for p in patterns)


def is_probable_fragment(raw: str) -> bool:
    s = normalize_whitespace(raw)
    if not s:
        return True
    if looks_like_author_block(s) and not extract_year(s) and "," not in s:
        return True
    if looks_like_author_block(s) and len(s) < 40 and not extract_year(s):
        return True
    return False


def clean_first_author(author_block: Optional[str]) -> Optional[str]:
    if not author_block:
        return None
    s = author_block.strip().strip(",;: ")
    if not s:
        return None

    first = s.split(";")[0].strip()
    if "," in first:
        surname = first.split(",")[0].strip()
    else:
        tokens = first.split()
        if len(tokens) >= 2 and re.fullmatch(r"[A-Z](?:\.[A-Z])?\.?", tokens[-1]):
            surname = " ".join(tokens[:-1]).strip()
        else:
            surname = tokens[0].strip()

    surname = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", surname)
    surname = TRAILING_PUNCT_RE.sub("", surname).strip()
    return surname or None


def split_author_and_rest(raw: str) -> tuple[Optional[str], str]:
    s = normalize_whitespace(raw)

    comma_positions = [m.start() for m in re.finditer(",", s)]
    for pos in comma_positions:
        left = s[:pos].strip()
        right = s[pos + 1 :].strip()
        if looks_like_author_block(left) and right:
            return left, right

    if "," in s:
        left, right = s.split(",", 1)
        return left.strip(), right.strip()

    return None, s


def strip_edition_markers(text: str) -> tuple[str, Optional[str]]:
    """Extract edition markers from a title-like segment."""
    match = EDITION_RE.search(text)
    marker = normalize_whitespace(match.group(0)) if match else None
    cleaned = EDITION_RE.sub(" ", text)
    cleaned = re.sub(r"\b(?:vol\.?|volume)\s*\d+\b", " ", cleaned, flags=re.I)
    return normalize_whitespace(cleaned.strip(" ,;:")), marker


def extract_pages(text: str) -> Optional[str]:
    m = PAGES_RE.search(text)
    if m:
        return normalize_whitespace(m.group(1))
    return None


def extract_volume_issue(text: str) -> tuple[Optional[str], Optional[str]]:
    m = VOL_ISSUE_RE.search(text)
    if m:
        return m.group(1), m.group(2)

    m = VOL_ONLY_RE.search(text)
    if m:
        return m.group(1), None

    return None, None


def strip_right_metadata(text: str) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Remove pages / volume / issue from the right edge.
    Returns cleaned text plus parsed volume/issue/pages.
    """
    working = text.strip(" ,;")
    pages = extract_pages(working)
    volume, issue = extract_volume_issue(working)

    working = re.sub(r",?\s*pp?\.\s*[A-Za-z0-9]+(?:\s*[-–—]\s*[A-Za-z0-9]+)?", "", working, flags=re.I)
    working = re.sub(r",\s*\d{1,4}\s*,\s*[A-Za-z0-9\-]{1,20}(?=,|\s*$)", "", working)
    working = re.sub(r",\s*\d{1,4}(?=,?\s*$)", "", working)
    working = normalize_whitespace(working.strip(" ,;"))

    return working, volume, issue, pages


def split_title_source_by_hints(working: str) -> tuple[str, Optional[str]]:
    """Split residual citation text into title and source using venue/proceedings hints."""
    match = VENUE_SPLIT_HINT_RE.search(working)
    if match and match.start() > 10:
        left = working[: match.start()].strip(" ,;:.-")
        right = working[match.start() :].strip(" ,;:.-")
        if left and right:
            return left, right

    parts = [p.strip(" ,;:.") for p in working.split(",") if p.strip(" ,;:.")]
    if len(parts) >= 3:
        return ", ".join(parts[:-1]).strip(), parts[-1].strip()
    if len(parts) == 2:
        left, right = parts
        if VENUE_SPLIT_HINT_RE.search(right) or len(right.split()) >= 3:
            return left, right
    return working.strip(" ,;:."), None


def split_title_and_source(rest: str) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    if not rest:
        return None, None, None, None, None

    working = strip_reference_prefix_noise(rest)
    working = normalize_whitespace(strip_doi(working))
    working, _ = strip_terminal_year_block(working)
    working, volume, issue, pages = strip_right_metadata(working)

    if not working:
        return None, None, volume, issue, pages

    title, source_title = split_title_source_by_hints(working)
    title, _edition_marker = strip_edition_markers(title)

    if title:
        title = normalize_whitespace(title.strip(" \"'“”‘’"))
    if source_title:
        source_title = normalize_whitespace(source_title.strip(" \"'“”‘’"))
        source_title = SOURCE_BOILERPLATE_RE.sub(lambda m: normalize_whitespace(m.group(0)), source_title)

    if title and re.search(r"\bpp?\.\b", title, re.I):
        title = re.sub(r",?\s*pp?\..*$", "", title, flags=re.I).strip(" ,;:.") or None

    if title and re.search(r",\s*\d{1,4}\s*,\s*[A-Za-z0-9\-]{1,20}$", title):
        title = re.sub(r",\s*\d{1,4}\s*,\s*[A-Za-z0-9\-]{1,20}$", "", title).strip(" ,;:.") or None

    return title, source_title, volume, issue, pages


def has_internal_year_boundary(raw: str) -> bool:
    s = normalize_whitespace(raw)
    return bool(re.search(r"\((19\d{2}|20\d{2})\);\s+[A-ZÀ-ÖØ-Ý]", s))


def has_multiple_years(raw: str) -> bool:
    return len(re.findall(r"(?<!\d)(17\d{2}|18\d{2}|19\d{2}|20\d{2})(?!\d)", raw)) >= 2


def score_parse_quality(
    raw: str,
    first_author: Optional[str],
    year: Optional[int],
    title: Optional[str],
    source_title: Optional[str],
) -> str:
    if is_probable_fragment(raw):
        return "failed"

    if has_internal_year_boundary(raw) or has_multiple_years(raw):
        if first_author and year:
            return "medium"
        return "low"

    if first_author and year and title:
        if len(title) >= 12 and not re.fullmatch(r"[A-Z][a-z]?$", title):
            return "high"

    if first_author and year:
        return "medium"

    if first_author or year or title or source_title:
        return "low"

    return "failed"


def parse_reference(raw: str) -> dict[str, Any]:
    raw = normalize_whitespace(raw)
    doi = extract_doi(raw)
    year = extract_year(raw)

    author_block, rest = split_author_and_rest(raw)
    first_author = clean_first_author(author_block) if author_block else None

    title, source_title, volume, issue, pages = split_title_and_source(rest)

    parse_quality = score_parse_quality(raw, first_author, year, title, source_title)

    return {
        "first_author": first_author,
        "year": year,
        "title": title,
        "source_title": source_title,
        "volume": volume,
        "issue": issue,
        "pages": pages,
        "doi": doi,
        "ref_string_norm": normalize_ref_string(raw),
        "parse_quality": parse_quality,
    }


def assert_db_ready(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    present = {row[0] for row in rows}
    required = {"references_raw", "cited_references"}
    missing = required - present
    if missing:
        raise RuntimeError(f"Missing tables: {sorted(missing)}")


def clear_cited_references(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM cited_references")


def fetch_raw_references(conn: sqlite3.Connection) -> list[tuple[int, int, int, str]]:
    return conn.execute(
        """
        SELECT raw_ref_id, doc_id, ref_order, raw_reference
        FROM references_raw
        ORDER BY doc_id, ref_order
        """
    ).fetchall()


def insert_cited_reference(conn: sqlite3.Connection, doc_id: int, raw_ref_id: int, parsed: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO cited_references (
            doc_id,
            raw_ref_id,
            first_author,
            year,
            title,
            source_title,
            volume,
            issue,
            pages,
            doi,
            ref_string_norm,
            parse_quality
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc_id,
            raw_ref_id,
            parsed["first_author"],
            parsed["year"],
            parsed["title"],
            parsed["source_title"],
            parsed["volume"],
            parsed["issue"],
            parsed["pages"],
            parsed["doi"],
            parsed["ref_string_norm"],
            parsed["parse_quality"],
        ),
    )


def main() -> int:
    args = parse_args()
    if not args.db.exists():
        raise FileNotFoundError(f"SQLite DB not found: {args.db}")

    conn = sqlite3.connect(args.db)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        assert_db_ready(conn)

        raw_rows = fetch_raw_references(conn)
        if not raw_rows:
            print("No rows found in references_raw.")
            return 0

        with conn:
            clear_cited_references(conn)

            inserted = 0
            quality_counts = {"high": 0, "medium": 0, "low": 0, "failed": 0}

            for raw_ref_id, doc_id, ref_order, raw_reference in raw_rows:
                try:
                    parsed = parse_reference(raw_reference)
                except Exception:
                    parsed = {
                        "first_author": None,
                        "year": None,
                        "title": None,
                        "source_title": None,
                        "volume": None,
                        "issue": None,
                        "pages": None,
                        "doi": extract_doi(raw_reference),
                        "ref_string_norm": normalize_ref_string(raw_reference),
                        "parse_quality": "failed",
                    }

                insert_cited_reference(conn, doc_id, raw_ref_id, parsed)
                quality_counts[parsed["parse_quality"]] = quality_counts.get(parsed["parse_quality"], 0) + 1
                inserted += 1

        print(f"Parsed raw references: {len(raw_rows)}")
        print(f"Inserted cited_references rows: {inserted}")
        print("Parse quality counts:")
        print(f"  high:   {quality_counts.get('high', 0)}")
        print(f"  medium: {quality_counts.get('medium', 0)}")
        print(f"  low:    {quality_counts.get('low', 0)}")
        print(f"  failed: {quality_counts.get('failed', 0)}")
        print(f"Database updated: {args.db}")

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
