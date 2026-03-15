#!/usr/bin/env python3
from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path
from typing import Iterable
from typing import Optional

import pandas as pd
from unidecode import unidecode


CSV_PATH = Path("data/raw/scopus_affordance_2010_2026.csv")
DB_PATH = Path("data/processed/affordance_lit.sqlite")

REQUIRED_TABLES = {
    "documents",
    "authors",
    "document_authors",
    "sources",
    "keywords",
    "document_keywords",
    "references_raw",
}

DOCUMENT_COLUMN_MAP = {
    "EID": "eid",
    "Title": "title",
    "Year": "year",
    "Source title": "source_title",
    "Volume": "volume",
    "Issue": "issue",
    "Art. No.": "article_no",
    "Page start": "page_start",
    "Page end": "page_end",
    "Cited by": "cited_by",
    "DOI": "doi",
    "Link": "link",
    "Abstract": "abstract",
    "Document Type": "document_type",
    "Publication Stage": "publication_stage",
    "Open Access": "open_access",
    "Source": "source_db",
}


def load_csv_robust(path: Path) -> tuple[pd.DataFrame, str]:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            return df, enc
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to read CSV: {path}\nLast error: {last_error}")


def normalize_blank_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        df[col] = df[col].replace(r"^\s*$", pd.NA, regex=True)
    return df

def text_or_none(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None

def int_or_none(value) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None

def normalize_doi(value) -> str | None:
    text = text_or_none(value)
    if not text:
        return None
    text = text.strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    text = text.strip()
    return text or None


def normalize_name(value: str | None) -> str | None:
    if not value:
        return None
    text = unidecode(value.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_title(value: str | None) -> str | None:
    if not value:
        return None
    text = unidecode(value.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def split_semicolon_field(value) -> list[str]:
    text = text_or_none(value)
    if not text:
        return []
    parts = [part.strip() for part in text.split(";")]
    return [part for part in parts if part]


###
### START REF SPLITTING BLOCK
###

YEAR_END_RE = re.compile(r"^(.*?\((19\d{2}|20\d{2})\))(?:;\s+)(.+)$")
YEAR_END_BARE_RE = re.compile(r"^(.*?\b(19\d{2}|20\d{2})\b)(?:;\s+)(.+)$")

AUTHOR_ONLY_RE = re.compile(
    r"^[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+(?:\s+[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+)*"
    r"(?:\s+[A-Z](?:\.[A-Z]){0,3}\.?|,\s*[A-Z](?:\.[A-Z]){0,3}\.?)$"
)

AUTHOR_START_RE = re.compile(
    r"^[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+(?:\s+[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+)*"
    r"(?:\s+[A-Z](?:\.[A-Z]){0,3}\.?|,\s*[A-Z](?:\.[A-Z]){0,3}\.?)"
)

DOI_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)
PAGES_RE = re.compile(r"\bpp?\.\s*[A-Za-z0-9]+(?:\s*[-–—]\s*[A-Za-z0-9]+)?", re.I)
YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2}|17\d{2}|18\d{2})(?!\d)")
PAREN_YEAR_RE = re.compile(r"\((17\d{2}|18\d{2}|19\d{2}|20\d{2})\)")
YEAR_SEMI_SPLIT_RE = re.compile(r"^(.*?\((17\d{2}|18\d{2}|19\d{2}|20\d{2})\)|.*?\b(17\d{2}|18\d{2}|19\d{2}|20\d{2})\b);\s+(.+)$")

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def extract_years(text: str) -> list[int]:
    years = []
    for y in YEAR_RE.findall(text):
        try:
            years.append(int(y))
        except ValueError:
            pass
    return years

def count_years(text: str) -> int:
    return len(extract_years(text))

def has_year(text: str) -> bool:
    return bool(PAREN_YEAR_RE.search(text) or YEAR_RE.search(text))


def has_doi(text: str) -> bool:
    return bool(DOI_RE.search(text))


def has_pages(text: str) -> bool:
    return bool(PAGES_RE.search(text))


def looks_like_author_only_fragment(text: str) -> bool:
    s = normalize_ws(text.strip(" ,"))
    if not s:
        return False
    if has_year(s) or has_doi(s) or has_pages(s):
        return False
    return bool(AUTHOR_ONLY_RE.match(s))


def looks_like_reference_candidate(text: str) -> bool:
    s = normalize_ws(text.strip(" ,"))
    if not s:
        return False
    if not AUTHOR_START_RE.match(s):
        return False
    if has_year(s) or has_doi(s) or has_pages(s):
        return True
    if "," in s:
        left, right = s.split(",", 1)
        if len(right.strip()) >= 12:
            return True
    return False

def ends_with_year(text: str) -> bool:
    s = normalize_ws(text).rstrip(" .,;:")
    return bool(re.search(r"(\((19\d{2}|20\d{2})\)|(19\d{2}|20\d{2}))$", s))

def find_year_boundary_split(ref: str, max_lookahead_parts: int = 4) -> Optional[tuple[str, str]]:
    """
    Detect merged pattern:
    <complete citation ending in year>; <start of next citation ...>

    Return (left_ref, right_ref) if a plausible split is found.
    """
    s = normalize_ws(ref)

    for rx in (YEAR_END_RE, YEAR_END_BARE_RE):
        m = rx.match(s)
        if not m:
            continue

        left = m.group(1).strip(" ,;")
        suffix = m.group(3).strip(" ,;")

        parts = [normalize_ws(p) for p in suffix.split(";") if normalize_ws(p)]
        if not parts:
            continue

        candidate = parts[0]
        if looks_like_reference_candidate(candidate):
            return left, candidate

        for j in range(1, min(len(parts), max_lookahead_parts)):
            candidate = f"{candidate}; {parts[j]}"
            if looks_like_reference_candidate(candidate):
                remainder = "; ".join(parts[:j+1]).strip(" ,;")
                tail = "; ".join(parts[j+1:]).strip(" ,;")
                right = remainder if not tail else f"{remainder}; {tail}"
                return left, right

    return None

def lookahead_candidate(parts: list[str], start_idx: int, max_parts: int = 4) -> Optional[str]:
    """
    Assemble up to max_parts semicolon fragments starting at start_idx.
    Return the shortest plausible citation candidate, else None.
    """
    candidate = parts[start_idx]
    if looks_like_reference_candidate(candidate):
        return candidate

    for j in range(start_idx + 1, min(len(parts), start_idx + max_parts)):
        candidate = f"{candidate}; {parts[j]}"
        if looks_like_reference_candidate(candidate):
            return candidate

    return None

def current_chunk_is_complete(text: str) -> bool:
    s = normalize_ws(text)
    if not s:
        return False

    if has_doi(s) or has_pages(s):
        return True

    if ends_with_year(s):
        return True

    if has_year(s) and "," in s:
        left, right = s.split(",", 1)
        if right.strip():
            return True

    return False

def should_start_new_reference(current: str, parts: list[str], idx: int) -> bool:
    """
    Start a new reference only with positive evidence.

    Key rule:
    If current chunk already ends as a complete citation, especially at a year boundary,
    and the next few fragments can be assembled into a plausible new citation,
    then split.
    """
    if idx >= len(parts):
        return False

    current = normalize_ws(current)
    next_frag = normalize_ws(parts[idx])

    if not next_frag:
        return False

    if not current_chunk_is_complete(current):
        return False

    candidate = lookahead_candidate(parts, idx, max_parts=4)
    if not candidate:
        return False

    # Strongest case: current citation ends with year and the next sequence forms a citation
    if ends_with_year(current):
        return True

    # Also allow split if next fragment is itself already a strong reference start
    if looks_like_reference_candidate(next_frag):
        return True

    return False

def split_references_conservative(value) -> list[str]:
    """
    Reference splitter for Scopus exports.

    Principles:
    - start from naive semicolon split
    - preserve semicolons inside author lists
    - split after completed citations when a lookahead sequence forms a plausible new citation
    """
    text = text_or_none(value)
    if not text:
        return []

    parts = [normalize_ws(part) for part in text.split(";")]
    parts = [part for part in parts if part]

    if not parts:
        return []

    refs: list[str] = []
    current = parts[0]
    i = 1

    while i < len(parts):
        if should_start_new_reference(current, parts, i):
            refs.append(current.strip(" ,;"))
            current = parts[i]
        else:
            current = f"{current}; {parts[i]}"
        i += 1

    refs.append(current.strip(" ,;"))
    return [r for r in refs if r]


def split_at_first_terminal_year_semicolon(text: str) -> Optional[tuple[str, str]]:
    """
    Split:
    <left citation ending in year> ; <right remainder>
    """
    s = normalize_ws(text)
    m = YEAR_SEMI_SPLIT_RE.match(s)
    if not m:
        return None

    left = m.group(1).strip(" ,;")
    right = m.group(4).strip(" ,;")

    if not left or not right:
        return None

    return left, right

def recursively_split_multi_year_reference(ref: str, max_rounds: int = 8) -> list[str]:
    """
    If one raw string contains multiple year markers, repeatedly split after the
    earliest terminal year followed by semicolon.
    """
    pending = [normalize_ws(ref)]
    out = []

    rounds = 0
    while pending and rounds < max_rounds:
        current = pending.pop(0)

        # Only attempt demerge when multiple years exist
        if count_years(current) <= 1:
            out.append(current)
            rounds += 1
            continue

        split_result = split_at_first_terminal_year_semicolon(current)
        if split_result is None:
            out.append(current)
            rounds += 1
            continue

        left, right = split_result
        out.append(left)
        pending.insert(0, right)
        rounds += 1

    out.extend(pending)
    return [x.strip(" ,;") for x in out if x.strip(" ,;")]


def split_references_final(value) -> list[str]:
    first_pass = split_references_conservative(value)
    final_refs = []
    for ref in first_pass:
        final_refs.extend(recursively_split_multi_year_reference(ref))
    return final_refs

###
### END REF SPLITTING BLOCK
###



def assert_db_ready(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    present = {row[0] for row in rows}
    missing = REQUIRED_TABLES - present
    if missing:
        raise RuntimeError(
            "Database schema is incomplete. Run scripts/init_db.py first.\n"
            f"Missing tables: {sorted(missing)}"
        )


def clear_ingest_tables(conn: sqlite3.Connection) -> None:
    """
    Rebuild ingest layer deterministically.
    This intentionally does not touch parsed or downstream tables.
    """
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        for table in [
            "document_reference_links",
            "references_raw",
            "document_keywords",
            "keywords",
            "document_authors",
            "authors",
            "sources",
            "documents",
        ]:
            conn.execute(f"DELETE FROM {table}")
    finally:
        conn.execute("PRAGMA foreign_keys = ON")



def ingest_documents(conn: sqlite3.Connection, df: pd.DataFrame) -> dict[str, int]:
    insert_sql = """
    INSERT INTO documents (
        eid, title, year, source_title, volume, issue, article_no, page_start,
        page_end, cited_by, doi, link, abstract, document_type,
        publication_stage, open_access, source_db
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    count = 0
    for _, row in df.iterrows():
        record = {
            "eid": text_or_none(row.get("EID")),
            "title": text_or_none(row.get("Title")),
            "year": int_or_none(row.get("Year")),
            "source_title": text_or_none(row.get("Source title")),
            "volume": text_or_none(row.get("Volume")),
            "issue": text_or_none(row.get("Issue")),
            "article_no": text_or_none(row.get("Art. No.")),
            "page_start": text_or_none(row.get("Page start")),
            "page_end": text_or_none(row.get("Page end")),
            "cited_by": int_or_none(row.get("Cited by")),
            "doi": normalize_doi(row.get("DOI")),
            "link": text_or_none(row.get("Link")),
            "abstract": text_or_none(row.get("Abstract")),
            "document_type": text_or_none(row.get("Document Type")),
            "publication_stage": text_or_none(row.get("Publication Stage")),
            "open_access": text_or_none(row.get("Open Access")),
            "source_db": text_or_none(row.get("Source")),
        }
        if not record["eid"]:
            continue
        conn.execute(
            insert_sql,
            (
                record["eid"],
                record["title"],
                record["year"],
                record["source_title"],
                record["volume"],
                record["issue"],
                record["article_no"],
                record["page_start"],
                record["page_end"],
                record["cited_by"],
                record["doi"],
                record["link"],
                record["abstract"],
                record["document_type"],
                record["publication_stage"],
                record["open_access"],
                record["source_db"],
            ),
        )
        count += 1

    doc_map = {
        eid: doc_id
        for doc_id, eid in conn.execute("SELECT doc_id, eid FROM documents")
    }
    return {"documents_inserted": count, "doc_map_size": len(doc_map)}


def ingest_sources(conn: sqlite3.Connection) -> int:
    source_titles = conn.execute(
        """
        SELECT DISTINCT source_title
        FROM documents
        WHERE source_title IS NOT NULL AND TRIM(source_title) <> ''
        """
    ).fetchall()

    count = 0
    for (source_title,) in source_titles:
        source_title_norm = normalize_title(source_title)
        conn.execute(
            """
            INSERT OR IGNORE INTO sources (source_title, source_title_norm)
            VALUES (?, ?)
            """,
            (source_title, source_title_norm),
        )
        count += 1
    return count


def get_doc_map(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        eid: doc_id
        for doc_id, eid in conn.execute("SELECT doc_id, eid FROM documents")
    }


def ingest_authors(conn: sqlite3.Connection, df: pd.DataFrame, doc_map: dict[str, int]) -> dict[str, int]:
    author_cache: dict[tuple, int] = {}
    author_rows = 0
    bridge_rows = 0

    for _, row in df.iterrows():
        eid = text_or_none(row.get("EID"))
        if not eid or eid not in doc_map:
            continue

        doc_id = doc_map[eid]
        display_names = split_semicolon_field(row.get("Authors"))
        full_names = split_semicolon_field(row.get("Author full names"))
        scopus_ids = split_semicolon_field(row.get("Author(s) ID"))

        n = max(len(display_names), len(full_names), len(scopus_ids))
        for idx in range(n):
            display_name = display_names[idx] if idx < len(display_names) else None
            full_name = full_names[idx] if idx < len(full_names) else None
            scopus_author_id = scopus_ids[idx] if idx < len(scopus_ids) else None

            chosen_name = full_name or display_name
            author_name_norm = normalize_name(chosen_name)

            if not scopus_author_id and not chosen_name:
                continue

            if scopus_author_id:
                key = ("scopus_id", scopus_author_id)
            else:
                key = ("name_norm", author_name_norm, full_name or "", display_name or "")

            if key not in author_cache:
                cur = conn.execute(
                    """
                    INSERT INTO authors (
                        scopus_author_id,
                        author_full_name,
                        author_display_name,
                        author_name_norm
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (scopus_author_id, full_name, display_name, author_name_norm),
                )
                author_cache[key] = cur.lastrowid
                author_rows += 1

            author_id = author_cache[key]
            conn.execute(
                """
                INSERT INTO document_authors (doc_id, author_order, author_id)
                VALUES (?, ?, ?)
                """,
                (doc_id, idx + 1, author_id),
            )
            bridge_rows += 1

    return {
        "authors_inserted": author_rows,
        "document_authors_inserted": bridge_rows,
    }


def ingest_keywords(conn: sqlite3.Connection, df: pd.DataFrame, doc_map: dict[str, int]) -> dict[str, int]:
    keyword_cache: dict[tuple[str, str], int] = {}
    keyword_rows = 0
    bridge_rows = 0

    field_specs = [
        ("Author Keywords", "author"),
        ("Index Keywords", "index"),
    ]

    for _, row in df.iterrows():
        eid = text_or_none(row.get("EID"))
        if not eid or eid not in doc_map:
            continue
        doc_id = doc_map[eid]

        for field_name, keyword_type in field_specs:
            terms = split_semicolon_field(row.get(field_name))
            for term in terms:
                keyword_norm = normalize_title(term)
                if not keyword_norm:
                    continue

                key = (keyword_norm, keyword_type)
                if key not in keyword_cache:
                    cur = conn.execute(
                        """
                        INSERT OR IGNORE INTO keywords (keyword, keyword_norm, keyword_type)
                        VALUES (?, ?, ?)
                        """,
                        (term, keyword_norm, keyword_type),
                    )
                    if cur.lastrowid:
                        keyword_id = cur.lastrowid
                    else:
                        keyword_id = conn.execute(
                            """
                            SELECT keyword_id
                            FROM keywords
                            WHERE keyword_norm = ? AND keyword_type = ?
                            """,
                            (keyword_norm, keyword_type),
                        ).fetchone()[0]
                    keyword_cache[key] = keyword_id
                    keyword_rows += 1
                keyword_id = keyword_cache[key]

                conn.execute(
                    """
                    INSERT OR IGNORE INTO document_keywords (doc_id, keyword_id)
                    VALUES (?, ?)
                    """,
                    (doc_id, keyword_id),
                )
                bridge_rows += 1

    return {
        "keywords_distinct_touched": keyword_rows,
        "document_keywords_inserted": bridge_rows,
    }


def ingest_references_raw(conn: sqlite3.Connection, df: pd.DataFrame, doc_map: dict[str, int]) -> dict[str, int]:
    docs_with_refs = 0
    raw_ref_rows = 0
    suspicious_short_refs = 0

    for _, row in df.iterrows():
        eid = text_or_none(row.get("EID"))
        if not eid or eid not in doc_map:
            continue
        doc_id = doc_map[eid]

        raw_refs = split_references_final(row.get("References"))
        if not raw_refs:
            continue

        docs_with_refs += 1
        for ref_order, raw_reference in enumerate(raw_refs, start=1):
            if len(raw_reference) < 40 and not has_year(raw_reference):
                suspicious_short_refs += 1

            conn.execute(
                """
                INSERT INTO references_raw (doc_id, ref_order, raw_reference)
                VALUES (?, ?, ?)
                """,
                (doc_id, ref_order, raw_reference),
            )
            raw_ref_rows += 1

    return {
        "docs_with_references": docs_with_refs,
        "references_raw_inserted": raw_ref_rows,
        "suspicious_short_refs": suspicious_short_refs,
    }


def main() -> int:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"SQLite DB not found: {DB_PATH}\nRun scripts/init_db.py first."
        )

    df, encoding_used = load_csv_robust(CSV_PATH)
    df = normalize_blank_strings(df)

    if "EID" not in df.columns:
        raise RuntimeError("Input CSV is missing required column: EID")

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        assert_db_ready(conn)

        with conn:
            clear_ingest_tables(conn)

            doc_stats = ingest_documents(conn, df)
            source_count = ingest_sources(conn)
            doc_map = get_doc_map(conn)

            author_stats = ingest_authors(conn, df, doc_map)
            keyword_stats = ingest_keywords(conn, df, doc_map)
            ref_stats = ingest_references_raw(conn, df, doc_map)

        print(f"CSV loaded with encoding: {encoding_used}")
        print(f"Documents inserted: {doc_stats['documents_inserted']}")
        print(f"Distinct sources inserted: {source_count}")
        print(f"Authors inserted: {author_stats['authors_inserted']}")
        print(f"Document-author links inserted: {author_stats['document_authors_inserted']}")
        print(f"Distinct keywords touched: {keyword_stats['keywords_distinct_touched']}")
        print(f"Document-keyword links inserted: {keyword_stats['document_keywords_inserted']}")
        print(f"Documents with references: {ref_stats['docs_with_references']}")
        print(f"Raw references inserted: {ref_stats['references_raw_inserted']}")
        print(f"Suspicious short raw refs: {ref_stats['suspicious_short_refs']}")
        print(f"Database populated: {DB_PATH}")

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

