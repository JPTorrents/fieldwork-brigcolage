#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from unidecode import unidecode

from parse_references import segment_reference_blob


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

REQUIRED_CSV_COLUMNS = ["EID", "Title", "Year"]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SQLite ingest tables from Scopus CSV.")
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="Input Scopus CSV path")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="SQLite database path")
    return parser.parse_args()

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


def validate_input_df(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_CSV_COLUMNS if col not in df.columns]
    if missing:
        raise RuntimeError(f"Input CSV is missing required columns: {missing}")

    eid_norm = df["EID"].fillna("").astype(str).str.strip()
    duplicate_eids = eid_norm[eid_norm.ne("") & eid_norm.duplicated(keep=False)]
    if not duplicate_eids.empty:
        sample = sorted(duplicate_eids.unique())[:5]
        raise RuntimeError(
            "Input CSV has non-unique EID values, refusing to ingest. "
            f"Example duplicate EIDs: {sample}"
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
            "reference_fragments_rejected",
            "cited_references",
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
    # Align loader with active DB schema instead of hard-coding retired columns.
    column_map = {
        "eid": text_or_none,
        "title": text_or_none,
        "year": int_or_none,
        "source_title": text_or_none,
        "volume": text_or_none,
        "cited_by": int_or_none,
        "doi": normalize_doi,
        "link": text_or_none,
        "abstract": text_or_none,
        "document_type": text_or_none,
        "source_db": text_or_none,
    }
    csv_map = {
        "eid": "EID",
        "title": "Title",
        "year": "Year",
        "source_title": "Source title",
        "volume": "Volume",
        "cited_by": "Cited by",
        "doi": "DOI",
        "link": "Link",
        "abstract": "Abstract",
        "document_type": "Document Type",
        "source_db": "Source",
    }

    available_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(documents)").fetchall()
    }
    insert_cols = [c for c in csv_map if c in available_cols]
    if "eid" not in insert_cols:
        raise RuntimeError("documents table missing required column: eid")

    placeholders = ", ".join(["?"] * len(insert_cols))
    cols_sql = ", ".join(insert_cols)
    insert_sql = f"INSERT INTO documents ({cols_sql}) VALUES ({placeholders})"

    count = 0
    for _, row in df.iterrows():
        eid = text_or_none(row.get("EID"))
        if not eid:
            continue

        record = []
        for col in insert_cols:
            value = row.get(csv_map[col])
            record.append(column_map[col](value))

        conn.execute(insert_sql, tuple(record))
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
                INSERT OR IGNORE INTO document_authors (doc_id, author_order, author_id)
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

    field_specs = [("Author Keywords", "author")]

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
    rejected_rows = 0

    for _, row in df.iterrows():
        eid = text_or_none(row.get("EID"))
        if not eid or eid not in doc_map:
            continue
        doc_id = doc_map[eid]

        retained, rejected = segment_reference_blob(text_or_none(row.get("References")))
        if not retained and not rejected:
            continue

        docs_with_refs += 1
        ref_order = 1
        for segment in retained:
            conn.execute(
                """
                INSERT INTO references_raw (
                    doc_id,
                    ref_order,
                    raw_reference,
                    raw_reference_len,
                    has_year,
                    has_doi,
                    has_url,
                    semicolon_count,
                    segmentation_quality,
                    segmentation_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    ref_order,
                    segment["raw"],
                    segment["raw_len"],
                    segment["has_year"],
                    segment["has_doi"],
                    segment["has_url"],
                    segment["semicolon_count"],
                    segment["quality"],
                    segment["reason"],
                ),
            )
            raw_ref_rows += 1
            ref_order += 1

        # Persist obvious fragments for traceability instead of dropping silently.
        parent_context = text_or_none(row.get("References"))
        for frag in rejected:
            conn.execute(
                """
                INSERT INTO reference_fragments_rejected (
                    doc_id,
                    ref_order,
                    raw_fragment,
                    raw_fragment_len,
                    has_year,
                    has_doi,
                    has_url,
                    semicolon_count,
                    rejection_reason,
                    parent_context
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    None,
                    frag["raw"],
                    frag["raw_len"],
                    frag["has_year"],
                    frag["has_doi"],
                    frag["has_url"],
                    frag["semicolon_count"],
                    frag["reason"],
                    parent_context,
                ),
            )
            rejected_rows += 1

    return {
        "docs_with_references": docs_with_refs,
        "references_raw_inserted": raw_ref_rows,
        "fragments_rejected": rejected_rows,
    }


def main() -> int:
    args = parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not args.db.exists():
        raise FileNotFoundError(
            f"SQLite DB not found: {args.db}\nRun scripts/init_db.py first."
        )

    df, encoding_used = load_csv_robust(args.csv)
    df = normalize_blank_strings(df)
    validate_input_df(df)

    conn = sqlite3.connect(args.db)
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
        print(f"Rejected reference fragments: {ref_stats['fragments_rejected']}")
        print(f"Database populated: {args.db}")

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
