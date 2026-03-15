#!/usr/bin/env python3
"""
normalize_references.py

Normalize parsed cited references into deduplicated cited works.

Pipeline:
1. Tier 1: exact DOI clustering
2. Tier 2: exact normalized key clustering
3. Tier 3: fuzzy matching inside blocking groups
4. Write:
   - reference_clusters
   - document_reference_links
   - reference_cluster_review
   - internal_citations (optional)

Assumptions:
- Input SQLite DB already contains a parsed cited references table, default: cited_references
- Input DB also contains a corpus documents table, default: documents
- The script attempts to autodetect common column name variants.

Recommended usage:
    python scripts/normalize_references.py --db data/processed/affordance_lit.sqlite

Core logic:
- DOI exact match is high-confidence auto-cluster
- Exact normalized key is high-confidence auto-cluster
- Fuzzy match within strict blocking:
    * same normalized first-author surname
    * same year
    * title token overlap >= threshold
  Then:
    * similarity >= 92 -> auto-cluster
    * 85-91 -> manual review
    * < 85 -> separate
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from rapidfuzz import fuzz
from tqdm import tqdm
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

DOI_REGEX = re.compile(r"""(?i)\b(10\.\d{4,9}/[-._;()/:a-z0-9]+)\b""")
YEAR_REGEX = re.compile(r"\b(18|19|20)\d{2}\b")
PUNCT_REGEX = re.compile(r"[^\w\s]")
WS_REGEX = re.compile(r"\s+")


@dataclass
class RefRecord:
    source_ref_id: Any
    citing_doc_id: Any
    raw_title: str
    raw_first_author: str
    raw_year: Optional[int]
    raw_doi: str
    title_norm: str
    title_tokens: list[str]
    title_key8: str
    first_author_norm: str
    year: Optional[int]
    doi_norm: str
    exact_key: str


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


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = unidecode(str(text)).lower()
    text = PUNCT_REGEX.sub(" ", text)
    text = WS_REGEX.sub(" ", text).strip()
    return text


def normalize_title(title: Optional[str]) -> tuple[str, list[str]]:
    text = normalize_text(title)
    if not text:
        return "", []
    tokens = [tok for tok in text.split() if tok and tok not in STOPWORDS]
    return " ".join(tokens), tokens


def normalize_author(author: Optional[str]) -> str:
    text = normalize_text(author)
    if not text:
        return ""
    text = re.split(r"\b(and|et al|&)\b|,", text)[0].strip()
    parts = text.split()
    if not parts:
        return ""
    return parts[-1]


def normalize_doi(doi: Optional[str]) -> str:
    if not doi:
        return ""
    doi = unidecode(str(doi)).strip().lower()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    doi = doi.replace("doi:", "").strip()
    m = DOI_REGEX.search(doi)
    return m.group(1).lower().rstrip(" .;,)") if m else doi.rstrip(" .;,)")


def safe_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        if 1800 <= value <= 2100:
            return value
        return None
    text = str(value).strip()
    if not text:
        return None
    m = YEAR_REGEX.search(text)
    if not m:
        return None
    year = int(m.group(0))
    return year if 1800 <= year <= 2100 else None


def token_overlap(tokens_a: list[str], tokens_b: list[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    a = set(tokens_a)
    b = set(tokens_b)
    inter = len(a & b)
    denom = min(len(a), len(b))
    if denom == 0:
        return 0.0
    return inter / denom


def stable_cluster_id(seed: str) -> str:
    h = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
    return f"refcl_{h}"


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def fetch_table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({quote_ident(table_name)})").fetchall()
    return [r[1] for r in rows]


def resolve_column(columns: list[str], candidates: list[str], table_name: str) -> str:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    raise ValueError(
        f"Could not resolve required column in table '{table_name}'. "
        f"Tried candidates: {candidates}. Available columns: {columns}"
    )


def ensure_output_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS reference_clusters;
        DROP TABLE IF EXISTS document_reference_links;
        DROP TABLE IF EXISTS reference_cluster_review;
        DROP TABLE IF EXISTS internal_citations;

        CREATE TABLE reference_clusters (
            cluster_id TEXT PRIMARY KEY,
            canonical_title TEXT,
            canonical_first_author TEXT,
            canonical_year INTEGER,
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
        """
    )
    conn.commit()


def choose_canonical(records: list[RefRecord]) -> RefRecord:
    def score(r: RefRecord) -> tuple[int, int, int, int]:
        return (
            1 if r.doi_norm else 0,
            len(r.title_norm),
            len(r.raw_first_author or ""),
            1 if r.year is not None else 0,
        )
    return sorted(records, key=score, reverse=True)[0]


def cluster_confidence_for_members(records: list[RefRecord]) -> tuple[str, str]:
    if any(r.doi_norm for r in records):
        return "high", "doi_exact"
    exact_keys = {r.exact_key for r in records if r.exact_key}
    if len(exact_keys) == 1 and exact_keys:
        return "high", "exact_key"
    return "medium", "fuzzy_title"


def detect_source_rows(
    conn: sqlite3.Connection,
    refs_table: str,
) -> list[dict[str, Any]]:
    cols = fetch_table_columns(conn, refs_table)

    ref_id_col = resolve_column(cols, ["cited_ref_id", "reference_id", "id"], refs_table)
    doc_id_col = resolve_column(cols, ["doc_id", "source_doc_id", "document_id", "citing_doc_id"], refs_table)
    title_col = resolve_column(cols, ["title_candidate", "title", "ref_title", "parsed_title"], refs_table)
    author_col = resolve_column(cols, ["first_author", "author", "ref_first_author", "parsed_first_author"], refs_table)
    year_col = resolve_column(cols, ["year", "ref_year", "parsed_year"], refs_table)
    doi_col = resolve_column(cols, ["doi", "ref_doi", "parsed_doi"], refs_table)

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
    cur = conn.execute(sql)
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def build_ref_records(rows: list[dict[str, Any]]) -> list[RefRecord]:
    records: list[RefRecord] = []
    for row in rows:
        raw_title = str(row.get("raw_title") or "").strip()
        raw_first_author = str(row.get("raw_first_author") or "").strip()
        raw_doi = str(row.get("raw_doi") or "").strip()
        year = safe_year(row.get("raw_year"))

        doi_norm = normalize_doi(raw_doi)
        title_norm, title_tokens = normalize_title(raw_title)
        first_author_norm = normalize_author(raw_first_author)
        title_key8 = "_".join(title_tokens[:8]) if title_tokens else ""
        exact_key = ""
        if first_author_norm and year and title_key8:
            exact_key = f"{first_author_norm}_{year}_{title_key8}"

        records.append(
            RefRecord(
                source_ref_id=row["source_ref_id"],
                citing_doc_id=row["citing_doc_id"],
                raw_title=raw_title,
                raw_first_author=raw_first_author,
                raw_year=year,
                raw_doi=raw_doi,
                title_norm=title_norm,
                title_tokens=title_tokens,
                title_key8=title_key8,
                first_author_norm=first_author_norm,
                year=year,
                doi_norm=doi_norm,
                exact_key=exact_key,
            )
        )
    return records


def apply_tier1_doi(records: list[RefRecord], dsu: DSU) -> dict[Any, str]:
    methods: dict[Any, str] = {}
    by_doi: dict[str, list[RefRecord]] = defaultdict(list)
    for r in records:
        if r.doi_norm:
            by_doi[r.doi_norm].append(r)

    for _, members in by_doi.items():
        if len(members) < 2:
            continue
        anchor = members[0].source_ref_id
        methods[anchor] = "doi_exact"
        for m in members[1:]:
            dsu.union(anchor, m.source_ref_id)
            methods[m.source_ref_id] = "doi_exact"
    return methods


def apply_tier2_exact_key(records: list[RefRecord], dsu: DSU, methods: dict[Any, str]) -> None:
    by_key: dict[str, list[RefRecord]] = defaultdict(list)
    for r in records:
        if r.doi_norm:
            continue
        if r.exact_key:
            by_key[r.exact_key].append(r)

    for _, members in by_key.items():
        if len(members) < 2:
            continue
        anchor = members[0].source_ref_id
        methods.setdefault(anchor, "exact_key")
        for m in members[1:]:
            dsu.union(anchor, m.source_ref_id)
            methods.setdefault(m.source_ref_id, "exact_key")


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
    for r in records:
        if not r.first_author_norm or r.year is None or not r.title_norm:
            continue
        grouped[(r.first_author_norm, r.year)].append(r)

    review_rows: list[tuple[Any, ...]] = []

    for (author_key, year), group in tqdm(grouped.items(), desc="Tier 3 fuzzy"):
        if len(group) < 2:
            continue

        rep_to_record: dict[Any, RefRecord] = {}
        for r in group:
            root = dsu.find(r.source_ref_id)
            current = rep_to_record.get(root)
            if current is None:
                rep_to_record[root] = r
            else:
                rep_to_record[root] = choose_canonical([current, r])

        reps = list(rep_to_record.items())
        n = len(reps)
        for i in range(n):
            root_i, rec_i = reps[i]
            for j in range(i + 1, n):
                root_j, rec_j = reps[j]

                if root_i == root_j:
                    continue

                overlap = token_overlap(rec_i.title_tokens, rec_j.title_tokens)
                if overlap < title_overlap_min:
                    continue

                sim = fuzz.token_sort_ratio(rec_i.title_norm, rec_j.title_norm)

                if sim >= auto_title_threshold:
                    merged = dsu.union(root_i, root_j)
                    if merged:
                        methods.setdefault(rec_i.source_ref_id, "fuzzy_title")
                        methods.setdefault(rec_j.source_ref_id, "fuzzy_title")
                elif review_title_min <= sim < auto_title_threshold:
                    review_rows.append(
                        (
                            str(rec_i.source_ref_id),
                            str(rec_j.source_ref_id),
                            None,
                            None,
                            author_key,
                            rec_i.year,
                            rec_j.year,
                            rec_i.raw_title,
                            rec_j.raw_title,
                            float(sim),
                            float(overlap),
                            "manual_review",
                            "same_author_same_year_overlap_borderline_title_similarity",
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
) -> dict[Any, str]:
    clusters: dict[Any, list[RefRecord]] = defaultdict(list)
    for r in records:
        clusters[dsu.find(r.source_ref_id)].append(r)

    source_to_cluster: dict[Any, str] = {}
    cluster_rows: list[tuple[Any, ...]] = []
    link_rows: list[tuple[Any, ...]] = []

    for root, members in clusters.items():
        canonical = choose_canonical(members)

        seed = (
            canonical.doi_norm
            or canonical.exact_key
            or f"{canonical.first_author_norm}|{canonical.year}|{canonical.title_norm}"
            or str(root)
        )
        cluster_id = stable_cluster_id(seed)

        conf, method = cluster_confidence_for_members(members)
        member_ids = sorted(str(m.source_ref_id) for m in members)

        cluster_rows.append(
            (
                cluster_id,
                canonical.raw_title or canonical.title_norm,
                canonical.raw_first_author or canonical.first_author_norm,
                canonical.year,
                canonical.doi_norm or None,
                conf,
                method,
                len(members),
                "|".join(member_ids),
            )
        )

        for m in members:
            source_to_cluster[m.source_ref_id] = cluster_id
            link_method = methods.get(m.source_ref_id, method)
            link_conf = "high" if link_method in {"doi_exact", "exact_key"} else conf
            link_rows.append(
                (
                    str(m.citing_doc_id),
                    str(m.source_ref_id),
                    cluster_id,
                    link_method,
                    link_conf,
                )
            )

    conn.executemany(
        """
        INSERT INTO reference_clusters (
            cluster_id, canonical_title, canonical_first_author, canonical_year,
            canonical_doi, cluster_confidence, cluster_method, member_count, source_ref_ids
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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


def update_review_cluster_ids(
    conn: sqlite3.Connection,
    source_to_cluster: dict[Any, str],
) -> None:
    rows = conn.execute(
        "SELECT review_id, ref_id_1, ref_id_2 FROM reference_cluster_review"
    ).fetchall()
    updates = []
    for review_id, ref1, ref2 in rows:
        c1 = source_to_cluster.get(ref1) or source_to_cluster.get(int(ref1)) if str(ref1).isdigit() else source_to_cluster.get(ref1)
        c2 = source_to_cluster.get(ref2) or source_to_cluster.get(int(ref2)) if str(ref2).isdigit() else source_to_cluster.get(ref2)
        updates.append((c1, c2, review_id))

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
    cols = fetch_table_columns(conn, docs_table)
    doc_id_col = resolve_column(cols, ["doc_id", "document_id", "id"], docs_table)
    title_col = resolve_column(cols, ["title", "document_title", "article_title"], docs_table)
    year_col = resolve_column(cols, ["year", "pub_year", "publication_year"], docs_table)

    doi_col = None
    lowered = {c.lower(): c for c in cols}
    for cand in ["doi", "document_doi", "article_doi"]:
        if cand.lower() in lowered:
            doi_col = lowered[cand.lower()]
            break

    select_parts = [
        f"{quote_ident(doc_id_col)} AS doc_id",
        f"{quote_ident(title_col)} AS raw_title",
        f"{quote_ident(year_col)} AS raw_year",
    ]
    if doi_col:
        select_parts.append(f"{quote_ident(doi_col)} AS raw_doi")
    else:
        select_parts.append("NULL AS raw_doi")

    sql = f"SELECT {', '.join(select_parts)} FROM {quote_ident(docs_table)}"
    cur = conn.execute(sql)
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def build_internal_citations(
    conn: sqlite3.Connection,
    docs_table: str,
    skip_internal_citations: bool,
) -> None:
    if skip_internal_citations:
        return

    cluster_rows = conn.execute(
        """
        SELECT cluster_id, canonical_title, canonical_year, canonical_doi, cluster_confidence
        FROM reference_clusters
        """
    ).fetchall()

    doc_rows = detect_document_rows(conn, docs_table)

    docs = []
    doi_to_docs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    year_title_docs: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for row in doc_rows:
        doc = {
            "doc_id": str(row["doc_id"]),
            "title_raw": str(row.get("raw_title") or "").strip(),
            "title_norm": normalize_title(row.get("raw_title"))[0],
            "title_tokens": normalize_title(row.get("raw_title"))[1],
            "year": safe_year(row.get("raw_year")),
            "doi_norm": normalize_doi(row.get("raw_doi")),
        }
        docs.append(doc)
        if doc["doi_norm"]:
            doi_to_docs[doc["doi_norm"]].append(doc)
        if doc["year"] is not None and doc["title_norm"]:
            year_title_docs[doc["year"]].append(doc)

    cluster_match_rows: list[tuple[Any, ...]] = []

    for cluster_id, canonical_title, canonical_year, canonical_doi, cluster_confidence in cluster_rows:
        matched_docs: list[tuple[str, str, str]] = []

        doi_norm = normalize_doi(canonical_doi)
        title_norm, title_tokens = normalize_title(canonical_title)

        if doi_norm and doi_norm in doi_to_docs:
            for doc in doi_to_docs[doi_norm]:
                matched_docs.append((doc["doc_id"], "doi", "high"))
        elif canonical_year is not None and title_norm:
            candidates = year_title_docs.get(canonical_year, [])
            best_doc = None
            best_sim = -1
            for doc in candidates:
                overlap = token_overlap(title_tokens, doc["title_tokens"])
                if overlap < 0.5:
                    continue
                sim = fuzz.token_sort_ratio(title_norm, doc["title_norm"])
                if sim > best_sim:
                    best_doc = doc
                    best_sim = sim
            if best_doc is not None and best_sim >= 92:
                matched_docs.append((best_doc["doc_id"], "title_year", "medium"))

        for cited_doc_id, method, conf in matched_docs:
            citing_rows = conn.execute(
                """
                SELECT DISTINCT doc_id
                FROM document_reference_links
                WHERE cluster_id = ?
                """,
                (cluster_id,),
            ).fetchall()
            for (citing_doc_id,) in citing_rows:
                citing_doc_id = str(citing_doc_id)
                if citing_doc_id == cited_doc_id:
                    continue
                cluster_match_rows.append(
                    (citing_doc_id, cited_doc_id, cluster_id, method, conf)
                )

    if cluster_match_rows:
        deduped = list(dict.fromkeys(cluster_match_rows))
        conn.executemany(
            """
            INSERT INTO internal_citations (
                citing_doc_id, cited_doc_id, cluster_id, match_method, match_confidence
            ) VALUES (?, ?, ?, ?, ?)
            """,
            deduped,
        )
        conn.commit()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalize cited references into deduplicated works.")
    p.add_argument("--db", required=True, help="Path to SQLite database")
    p.add_argument("--refs-table", default="cited_references", help="Parsed cited references table")
    p.add_argument("--docs-table", default="documents", help="Corpus documents table")
    p.add_argument("--auto-title-threshold", type=int, default=92, help="Auto-cluster threshold for fuzzy title similarity")
    p.add_argument("--review-title-min", type=int, default=85, help="Lower threshold for manual review band")
    p.add_argument("--title-overlap-min", type=float, default=0.50, help="Minimum title token overlap for fuzzy candidate comparison")
    p.add_argument("--skip-internal-citations", action="store_true", help="Skip optional internal article-to-article citation linking")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row

    ensure_output_tables(conn)

    source_rows = detect_source_rows(conn, args.refs_table)
    if not source_rows:
        print(f"No rows found in {args.refs_table}", file=sys.stderr)
        return 1

    records = build_ref_records(source_rows)
    dsu = DSU([r.source_ref_id for r in records])

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
        "input_references": conn.execute(f"SELECT COUNT(*) FROM {quote_ident(args.refs_table)}").fetchone()[0],
        "reference_clusters": conn.execute("SELECT COUNT(*) FROM reference_clusters").fetchone()[0],
        "document_reference_links": conn.execute("SELECT COUNT(*) FROM document_reference_links").fetchone()[0],
        "review_pairs": conn.execute("SELECT COUNT(*) FROM reference_cluster_review").fetchone()[0],
        "internal_citations": conn.execute("SELECT COUNT(*) FROM internal_citations").fetchone()[0],
    }

    print("Normalization complete")
    for k, v in counts.items():
        print(f"{k}: {v}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
