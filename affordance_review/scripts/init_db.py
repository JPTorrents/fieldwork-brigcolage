#!/usr/bin/env python3
from pathlib import Path
import sqlite3

DB_PATH = Path("data/processed/affordance_lit.sqlite")

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

BEGIN;

CREATE TABLE IF NOT EXISTS documents (
    doc_id INTEGER PRIMARY KEY,
    eid TEXT UNIQUE,
    title TEXT,
    year INTEGER,
    source_title TEXT,
    volume TEXT,
    issue TEXT,
    article_no TEXT,
    page_start TEXT,
    page_end TEXT,
    cited_by INTEGER,
    doi TEXT,
    link TEXT,
    abstract TEXT,
    document_type TEXT,
    publication_stage TEXT,
    open_access TEXT,
    source_db TEXT
);

CREATE TABLE IF NOT EXISTS authors (
    author_id INTEGER PRIMARY KEY,
    scopus_author_id TEXT,
    author_full_name TEXT,
    author_display_name TEXT,
    author_name_norm TEXT
);

CREATE TABLE IF NOT EXISTS document_authors (
    doc_id INTEGER NOT NULL,
    author_order INTEGER NOT NULL,
    author_id INTEGER NOT NULL,
    PRIMARY KEY (doc_id, author_order),
    UNIQUE (doc_id, author_id),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    FOREIGN KEY (author_id) REFERENCES authors(author_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS sources (
    source_id INTEGER PRIMARY KEY,
    source_title TEXT,
    source_title_norm TEXT,
    UNIQUE (source_title_norm)
);

CREATE TABLE IF NOT EXISTS keywords (
    keyword_id INTEGER PRIMARY KEY,
    keyword TEXT,
    keyword_norm TEXT,
    keyword_type TEXT CHECK (keyword_type IN ('author', 'index')),
    UNIQUE (keyword_norm, keyword_type)
);

CREATE TABLE IF NOT EXISTS document_keywords (
    doc_id INTEGER NOT NULL,
    keyword_id INTEGER NOT NULL,
    PRIMARY KEY (doc_id, keyword_id),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    FOREIGN KEY (keyword_id) REFERENCES keywords(keyword_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS references_raw (
    raw_ref_id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL,
    ref_order INTEGER NOT NULL,
    raw_reference TEXT NOT NULL,
    UNIQUE (doc_id, ref_order),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS cited_references (
    cited_ref_id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL,
    raw_ref_id INTEGER NOT NULL,
    first_author TEXT,
    year INTEGER,
    title TEXT,
    source_title TEXT,
    volume TEXT,
    issue TEXT,
    pages TEXT,
    doi TEXT,
    ref_string_norm TEXT,
    parse_quality TEXT CHECK (
        parse_quality IN ('high', 'medium', 'low', 'failed')
    ),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    FOREIGN KEY (raw_ref_id) REFERENCES references_raw(raw_ref_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    UNIQUE (raw_ref_id)
);

CREATE TABLE IF NOT EXISTS reference_clusters (
    ref_cluster_id INTEGER PRIMARY KEY,
    canonical_first_author TEXT,
    canonical_year INTEGER,
    canonical_title TEXT,
    canonical_source_title TEXT,
    canonical_doi TEXT,
    cluster_method TEXT,
    cluster_confidence REAL CHECK (
        cluster_confidence IS NULL OR
        (cluster_confidence >= 0.0 AND cluster_confidence <= 1.0)
    )
);

CREATE TABLE IF NOT EXISTS document_reference_links (
    doc_id INTEGER NOT NULL,
    ref_cluster_id INTEGER NOT NULL,
    PRIMARY KEY (doc_id, ref_cluster_id),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    FOREIGN KEY (ref_cluster_id) REFERENCES reference_clusters(ref_cluster_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_documents_year
    ON documents(year);

CREATE INDEX IF NOT EXISTS idx_documents_doi
    ON documents(doi);

CREATE INDEX IF NOT EXISTS idx_documents_source_title
    ON documents(source_title);

CREATE INDEX IF NOT EXISTS idx_authors_scopus_author_id
    ON authors(scopus_author_id);

CREATE INDEX IF NOT EXISTS idx_keywords_keyword_norm
    ON keywords(keyword_norm);

CREATE INDEX IF NOT EXISTS idx_references_raw_doc_id
    ON references_raw(doc_id);

CREATE INDEX IF NOT EXISTS idx_cited_references_doi
    ON cited_references(doi);

CREATE INDEX IF NOT EXISTS idx_cited_references_first_author_year
    ON cited_references(first_author, year);

CREATE INDEX IF NOT EXISTS idx_document_reference_links_doc_ref
    ON document_reference_links(doc_id, ref_cluster_id);

COMMIT;
"""

def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()

    print(f"Initialized SQLite database at: {DB_PATH}")

if __name__ == "__main__":
    main()
