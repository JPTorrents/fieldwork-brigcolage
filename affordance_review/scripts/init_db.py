#!/usr/bin/env python3
from __future__ import annotations

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
    cited_by INTEGER,
    doi TEXT,
    link TEXT,
    abstract TEXT,
    source_db TEXT
);

CREATE TABLE IF NOT EXISTS authors (
    author_id INTEGER PRIMARY KEY,
    scopus_author_id TEXT,
    author_full_name TEXT,
    author_display_name TEXT,
    author_name_norm TEXT,
    UNIQUE (scopus_author_id, author_full_name, author_display_name)
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
    keyword_type TEXT CHECK (keyword_type IN ('author')),
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
    raw_reference_len INTEGER,
    has_year INTEGER DEFAULT 0 CHECK (has_year IN (0, 1)),
    has_doi INTEGER DEFAULT 0 CHECK (has_doi IN (0, 1)),
    has_url INTEGER DEFAULT 0 CHECK (has_url IN (0, 1)),
    semicolon_count INTEGER DEFAULT 0,
    segmentation_quality TEXT CHECK (
        segmentation_quality IN ('clean', 'needs_review', 'fragment')
    ),
    segmentation_reason TEXT,
    UNIQUE (doc_id, ref_order),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reference_fragments_rejected (
    rejected_ref_id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL,
    ref_order INTEGER,
    raw_fragment TEXT NOT NULL,
    raw_fragment_len INTEGER,
    has_year INTEGER DEFAULT 0 CHECK (has_year IN (0, 1)),
    has_doi INTEGER DEFAULT 0 CHECK (has_doi IN (0, 1)),
    has_url INTEGER DEFAULT 0 CHECK (has_url IN (0, 1)),
    semicolon_count INTEGER DEFAULT 0,
    rejection_reason TEXT NOT NULL,
    parent_context TEXT,
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
    cluster_id TEXT PRIMARY KEY,
    canonical_first_author TEXT,
    canonical_year INTEGER,
    canonical_title TEXT,
    canonical_source_title TEXT,
    canonical_doi TEXT,
    cluster_method TEXT,
    cluster_confidence TEXT,
    member_count INTEGER,
    source_ref_ids TEXT
);

CREATE TABLE IF NOT EXISTS document_reference_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL,
    source_ref_id INTEGER,
    cluster_id TEXT NOT NULL,
    link_method TEXT,
    link_confidence TEXT,
    UNIQUE (doc_id, source_ref_id),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    FOREIGN KEY (cluster_id) REFERENCES reference_clusters(cluster_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reference_cluster_review (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ref_id_1 INTEGER,
    ref_id_2 INTEGER,
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

CREATE TABLE IF NOT EXISTS internal_citations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    citing_doc_id INTEGER NOT NULL,
    cited_doc_id INTEGER NOT NULL,
    cluster_id TEXT,
    match_method TEXT,
    match_confidence TEXT,
    UNIQUE (citing_doc_id, cited_doc_id, cluster_id),
    FOREIGN KEY (citing_doc_id) REFERENCES documents(doc_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    FOREIGN KEY (cited_doc_id) REFERENCES documents(doc_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    FOREIGN KEY (cluster_id) REFERENCES reference_clusters(cluster_id)
        ON UPDATE CASCADE
        ON DELETE SET NULL
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

CREATE INDEX IF NOT EXISTS idx_references_raw_segmentation_quality
    ON references_raw(segmentation_quality);

CREATE INDEX IF NOT EXISTS idx_references_raw_has_year
    ON references_raw(has_year);

CREATE INDEX IF NOT EXISTS idx_reference_fragments_rejected_doc_id
    ON reference_fragments_rejected(doc_id);

CREATE INDEX IF NOT EXISTS idx_reference_fragments_rejected_reason
    ON reference_fragments_rejected(rejection_reason);

CREATE INDEX IF NOT EXISTS idx_cited_references_doi
    ON cited_references(doi);

CREATE INDEX IF NOT EXISTS idx_cited_references_first_author_year
    ON cited_references(first_author, year);

CREATE INDEX IF NOT EXISTS idx_document_reference_links_doc_ref
    ON document_reference_links(doc_id, cluster_id);

CREATE INDEX IF NOT EXISTS idx_reference_clusters_canonical_doi
    ON reference_clusters(canonical_doi);

CREATE INDEX IF NOT EXISTS idx_internal_citations_docs
    ON internal_citations(citing_doc_id, cited_doc_id);

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
    main()    scopus_author_id TEXT,
    author_full_name TEXT,
    author_display_name TEXT,
    author_name_norm TEXT,
    UNIQUE (scopus_author_id, author_full_name, author_display_name)
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
    cluster_id TEXT PRIMARY KEY,
    canonical_first_author TEXT,
    canonical_year INTEGER,
    canonical_title TEXT,
    canonical_source_title TEXT,
    canonical_doi TEXT,
    cluster_method TEXT,
    cluster_confidence TEXT,
    member_count INTEGER,
    source_ref_ids TEXT
);

CREATE TABLE IF NOT EXISTS document_reference_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL,
    source_ref_id INTEGER,
    cluster_id TEXT NOT NULL,
    link_method TEXT,
    link_confidence TEXT,
    UNIQUE (doc_id, source_ref_id),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    FOREIGN KEY (cluster_id) REFERENCES reference_clusters(cluster_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reference_cluster_review (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ref_id_1 INTEGER,
    ref_id_2 INTEGER,
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

CREATE TABLE IF NOT EXISTS internal_citations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    citing_doc_id INTEGER NOT NULL,
    cited_doc_id INTEGER NOT NULL,
    cluster_id TEXT,
    match_method TEXT,
    match_confidence TEXT,
    UNIQUE (citing_doc_id, cited_doc_id, cluster_id),
    FOREIGN KEY (citing_doc_id) REFERENCES documents(doc_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    FOREIGN KEY (cited_doc_id) REFERENCES documents(doc_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    FOREIGN KEY (cluster_id) REFERENCES reference_clusters(cluster_id)
        ON UPDATE CASCADE
        ON DELETE SET NULL
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
    ON document_reference_links(doc_id, cluster_id);

CREATE INDEX IF NOT EXISTS idx_reference_clusters_canonical_doi
    ON reference_clusters(canonical_doi);

CREATE INDEX IF NOT EXISTS idx_internal_citations_docs
    ON internal_citations(citing_doc_id, cited_doc_id);

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
