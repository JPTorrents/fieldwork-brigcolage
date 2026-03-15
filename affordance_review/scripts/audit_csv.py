#!/usr/bin/env python3
"""
Audit a Scopus CSV export and write:
- outputs/logs/csv_audit.txt
- outputs/tables/missingness.csv
- outputs/tables/duplicate_docs.csv

Checks:
- robust CSV loading with encoding fallback
- null / blank rates per column
- duplicate EID
- duplicate DOI
- count missing abstracts
- count missing references
- semicolon-separated field consistency
- summary counts for DOI / Abstract / Author Keywords / References
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Iterable, Optional

import pandas as pd


INPUT_CSV = Path("data/raw/scopus_affordance_2010_2026.csv")
OUTPUT_LOG = Path("outputs/logs/csv_audit.txt")
OUTPUT_MISSINGNESS = Path("outputs/tables/missingness.csv")
OUTPUT_DUPLICATES = Path("outputs/tables/duplicate_docs.csv")

# High-value fields to surface in reports when available
KEY_FIELDS = [
    "Authors",
    "Author full names",
    "Author(s) ID",
    "Title",
    "Year",
    "Source title",
    "Cited by",
    "DOI",
    "Abstract",
    "Author Keywords",
    "Index Keywords",
    "References",
    "EID",
]

# Semicolon-delimited fields typically found in Scopus exports
SEMICOLON_FIELDS = [
    "Authors",
    "Author full names",
    "Author(s) ID",
    "Author Keywords",
    "Index Keywords",
    "References",
]

# Fields where aligned counts matter across columns
# Example: Authors / Author full names / Author(s) ID should usually have matching item counts
ALIGNED_GROUPS = [
    ["Authors", "Author full names", "Author(s) ID"],
]

REQUIRED_COLUMNS = ["EID"]


def ensure_output_dirs() -> None:
    OUTPUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MISSINGNESS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DUPLICATES.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a Scopus CSV export.")
    parser.add_argument("--input", type=Path, default=INPUT_CSV, help="Input CSV path")
    parser.add_argument("--output-log", type=Path, default=OUTPUT_LOG, help="Output audit report path")
    parser.add_argument("--output-missingness", type=Path, default=OUTPUT_MISSINGNESS, help="Output missingness CSV path")
    parser.add_argument("--output-duplicates", type=Path, default=OUTPUT_DUPLICATES, help="Output duplicates CSV path")
    return parser.parse_args()


def load_csv_robust(path: Path) -> tuple[pd.DataFrame, str, str]:
    """
    Load CSV with a small set of common encoding fallbacks.
    """
    encodings_to_try = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1",
    ]

    last_error: Optional[Exception] = None
    separators = [",", ";", "\t"]

    for enc in encodings_to_try:
        for sep in separators:
            try:
                df = pd.read_csv(path, encoding=enc, low_memory=False, sep=sep)
                if df.shape[1] <= 1:
                    continue
                return df, enc, sep
            except UnicodeDecodeError as e:
                last_error = e
            except Exception as e:
                last_error = e

    raise RuntimeError(f"Failed to read CSV: {path}\nLast error: {last_error}")


def normalize_blank_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert empty or whitespace-only strings to NA for object columns.
    """
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].replace(r"^\s*$", pd.NA, regex=True)
    return df


def has_content(series: pd.Series) -> pd.Series:
    """
    True if value is non-null and not just whitespace.
    """
    return series.fillna("").astype(str).str.strip().ne("")


def split_semicolon_items(value) -> list[str]:
    """
    Split a semicolon-delimited field into cleaned items.
    Empty items are dropped.
    """
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    items = [x.strip() for x in text.split(";")]
    return [x for x in items if x]


def item_count(series: pd.Series) -> pd.Series:
    """
    Number of semicolon-delimited items per row.
    """
    return series.apply(lambda x: len(split_semicolon_items(x)))


def inspect_semicolon_field(series: pd.Series) -> dict:
    """
    Produce simple consistency diagnostics for a semicolon-separated field.
    """
    nonmissing = series[has_content(series)]
    if nonmissing.empty:
        return {
            "rows_with_content": 0,
            "mean_items": 0.0,
            "median_items": 0.0,
            "max_items": 0,
            "rows_with_double_semicolon": 0,
            "rows_ending_semicolon": 0,
        }

    counts = item_count(nonmissing)
    as_text = nonmissing.astype(str)

    return {
        "rows_with_content": int(len(nonmissing)),
        "mean_items": round(float(counts.mean()), 3),
        "median_items": round(float(counts.median()), 3),
        "max_items": int(counts.max()),
        "rows_with_double_semicolon": int(as_text.str.contains(r";\s*;").sum()),
        "rows_ending_semicolon": int(as_text.str.contains(r";\s*$").sum()),
    }


def aligned_group_mismatches(df: pd.DataFrame, fields: Iterable[str]) -> dict:
    """
    For rows where at least one field in the group has content, compare semicolon-item counts.
    Returns mismatch diagnostics.
    """
    fields = [f for f in fields if f in df.columns]
    if len(fields) < 2:
        return {"checked_fields": fields, "rows_checked": 0, "rows_mismatch": 0}

    counts_df = pd.DataFrame({f: item_count(df[f]) for f in fields})
    any_content = pd.DataFrame({f: has_content(df[f]) for f in fields}).any(axis=1)
    counts_df = counts_df.loc[any_content]

    if counts_df.empty:
        return {"checked_fields": fields, "rows_checked": 0, "rows_mismatch": 0}

    mismatch_mask = counts_df.nunique(axis=1) > 1

    return {
        "checked_fields": fields,
        "rows_checked": int(len(counts_df)),
        "rows_mismatch": int(mismatch_mask.sum()),
    }


def compute_missingness(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "column": df.columns,
        "n_missing": df.isna().sum().values,
        "pct_missing": (df.isna().mean() * 100).round(3).values,
        "n_nonmissing": df.notna().sum().values,
    })
    out = out.sort_values(["pct_missing", "n_missing"], ascending=[False, False]).reset_index(drop=True)
    return out


def duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a table of duplicates for EID and DOI.
    Includes useful identifying fields when present.
    """
    frames = []

    if "EID" in df.columns:
        eid_nonblank = has_content(df["EID"])
        eid_dupe_mask = eid_nonblank & df["EID"].duplicated(keep=False)
        if eid_dupe_mask.any():
            sub = df.loc[eid_dupe_mask].copy()
            sub["duplicate_type"] = "EID"
            frames.append(sub)

    if "DOI" in df.columns:
        doi_norm = df["DOI"].fillna("").astype(str).str.strip().str.lower()
        doi_nonblank = doi_norm.ne("")
        doi_dupe_mask = doi_nonblank & doi_norm.duplicated(keep=False)
        if doi_dupe_mask.any():
            sub = df.loc[doi_dupe_mask].copy()
            sub["duplicate_type"] = "DOI"
            sub["_doi_norm"] = doi_norm.loc[doi_dupe_mask]
            frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["duplicate_type"] + [c for c in KEY_FIELDS if c in df.columns])

    dup = pd.concat(frames, ignore_index=True)

    preferred_cols = ["duplicate_type"] + [c for c in KEY_FIELDS if c in dup.columns]
    extra_cols = [c for c in dup.columns if c not in preferred_cols and c != "_doi_norm"]
    dup = dup[preferred_cols + extra_cols]

    sort_cols = [c for c in ["duplicate_type", "EID", "DOI", "Title", "Year"] if c in dup.columns]
    if sort_cols:
        dup = dup.sort_values(sort_cols).reset_index(drop=True)

    return dup


def is_probable_duplicate_header(column_name: str) -> bool:
    return bool(re.search(r"\.\d+$", str(column_name)))


def main() -> int:
    args = parse_args()

    global INPUT_CSV, OUTPUT_LOG, OUTPUT_MISSINGNESS, OUTPUT_DUPLICATES
    INPUT_CSV = args.input
    OUTPUT_LOG = args.output_log
    OUTPUT_MISSINGNESS = args.output_missingness
    OUTPUT_DUPLICATES = args.output_duplicates

    ensure_output_dirs()

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df_raw, encoding_used, separator_used = load_csv_robust(INPUT_CSV)
    df = normalize_blank_strings(df_raw)

    duplicate_header_cols = [c for c in df.columns if is_probable_duplicate_header(c)]
    missing_required_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    n_rows, n_cols = df.shape

    missingness = compute_missingness(df)
    missingness.to_csv(OUTPUT_MISSINGNESS, index=False)

    duplicates = duplicate_rows(df)
    duplicates.to_csv(OUTPUT_DUPLICATES, index=False)

    # Required counts
    missing_abstracts = int(df["Abstract"].isna().sum()) if "Abstract" in df.columns else None
    missing_references = int(df["References"].isna().sum()) if "References" in df.columns else None

    docs_with_doi = int(has_content(df["DOI"]).sum()) if "DOI" in df.columns else None
    docs_with_abstract = int(has_content(df["Abstract"]).sum()) if "Abstract" in df.columns else None
    docs_with_author_keywords = int(has_content(df["Author Keywords"]).sum()) if "Author Keywords" in df.columns else None
    docs_with_references = int(has_content(df["References"]).sum()) if "References" in df.columns else None

    duplicate_eid_count = 0
    duplicate_doi_count = 0

    if "EID" in df.columns:
        eid_nonblank = has_content(df["EID"])
        duplicate_eid_count = int((eid_nonblank & df["EID"].duplicated(keep=False)).sum())

    if "DOI" in df.columns:
        doi_norm = df["DOI"].fillna("").astype(str).str.strip().str.lower()
        duplicate_doi_count = int((doi_norm.ne("") & doi_norm.duplicated(keep=False)).sum())

    # Semicolon field diagnostics
    semicolon_report = {}
    for col in SEMICOLON_FIELDS:
        if col in df.columns:
            semicolon_report[col] = inspect_semicolon_field(df[col])

    aligned_report = []
    for group in ALIGNED_GROUPS:
        result = aligned_group_mismatches(df, group)
        aligned_report.append(result)

    # Assertions / hard checks
    hard_check_messages = []
    if missing_required_columns:
        hard_check_messages.append(f"FAIL: Missing required columns: {missing_required_columns}.")
    else:
        if duplicate_eid_count > 0:
            hard_check_messages.append(f"FAIL: Non-unique EID detected ({duplicate_eid_count} rows involved).")
        else:
            hard_check_messages.append("PASS: EID column present and unique.")

    if duplicate_header_cols:
        hard_check_messages.append(
            f"WARN: Potential duplicate-header columns detected: {duplicate_header_cols}."
        )

    # Build text report
    lines = []
    lines.append("SCOPUS CSV AUDIT REPORT")
    lines.append("=" * 80)
    lines.append(f"Input file: {INPUT_CSV}")
    lines.append(f"Encoding used: {encoding_used}")
    lines.append(f"Delimiter used: {separator_used!r}")
    lines.append(f"Rows: {n_rows:,}")
    lines.append(f"Columns: {n_cols:,}")
    lines.append("")

    lines.append("HIGH-VALUE FIELD PRESENCE")
    lines.append("-" * 80)
    for field in KEY_FIELDS:
        status = "present" if field in df.columns else "missing"
        lines.append(f"{field}: {status}")
    lines.append("")

    lines.append("HARD CHECKS")
    lines.append("-" * 80)
    lines.extend(hard_check_messages)
    lines.append("")

    lines.append("DOCUMENT-LEVEL COVERAGE")
    lines.append("-" * 80)
    if docs_with_doi is not None:
        lines.append(f"Docs with DOI: {docs_with_doi:,} / {n_rows:,}")
    if docs_with_abstract is not None:
        lines.append(f"Docs with Abstract: {docs_with_abstract:,} / {n_rows:,}")
    if docs_with_author_keywords is not None:
        lines.append(f"Docs with Author Keywords: {docs_with_author_keywords:,} / {n_rows:,}")
    if docs_with_references is not None:
        lines.append(f"Docs with References: {docs_with_references:,} / {n_rows:,}")
    lines.append("")

    lines.append("MISSINGNESS")
    lines.append("-" * 80)
    if missing_abstracts is not None:
        lines.append(f"Missing Abstracts: {missing_abstracts:,}")
    if missing_references is not None:
        lines.append(f"Missing References: {missing_references:,}")
    lines.append("")
    lines.append("Top 10 columns by missingness:")
    top10 = missingness.head(10)
    for _, row in top10.iterrows():
        lines.append(
            f"- {row['column']}: {int(row['n_missing']):,} missing "
            f"({float(row['pct_missing']):.3f}%)"
        )
    lines.append("")

    lines.append("DUPLICATES")
    lines.append("-" * 80)
    lines.append(f"Rows involved in duplicate EID groups: {duplicate_eid_count:,}")
    lines.append(f"Rows involved in duplicate DOI groups: {duplicate_doi_count:,}")
    lines.append(f"Duplicate rows exported: {len(duplicates):,}")
    lines.append("")

    lines.append("SEMICOLON-DELIMITED FIELD DIAGNOSTICS")
    lines.append("-" * 80)
    if semicolon_report:
        for col, stats in semicolon_report.items():
            lines.append(f"{col}:")
            lines.append(f"  rows_with_content: {stats['rows_with_content']:,}")
            lines.append(f"  mean_items: {stats['mean_items']}")
            lines.append(f"  median_items: {stats['median_items']}")
            lines.append(f"  max_items: {stats['max_items']}")
            lines.append(f"  rows_with_double_semicolon: {stats['rows_with_double_semicolon']:,}")
            lines.append(f"  rows_ending_semicolon: {stats['rows_ending_semicolon']:,}")
    else:
        lines.append("No configured semicolon-delimited fields found.")
    lines.append("")

    lines.append("ALIGNED FIELD CONSISTENCY")
    lines.append("-" * 80)
    for result in aligned_report:
        fields = result["checked_fields"]
        lines.append(f"Fields: {fields}")
        lines.append(f"  rows_checked: {result['rows_checked']:,}")
        lines.append(f"  rows_mismatch: {result['rows_mismatch']:,}")
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 80)
    lines.append(f"- {OUTPUT_LOG}")
    lines.append(f"- {OUTPUT_MISSINGNESS}")
    lines.append(f"- {OUTPUT_DUPLICATES}")
    lines.append("")

    OUTPUT_LOG.write_text("\n".join(lines), encoding="utf-8")

    # Console summary
    print(f"Audit complete for {INPUT_CSV}")
    print(f"Rows: {n_rows:,} | Columns: {n_cols:,} | Encoding: {encoding_used}")
    print(f"Missingness table: {OUTPUT_MISSINGNESS}")
    print(f"Duplicate rows: {OUTPUT_DUPLICATES}")
    print(f"Audit log: {OUTPUT_LOG}")

    # Fail at the end if EID duplicates exist, after all outputs are written.
    if missing_required_columns:
        raise AssertionError(f"Missing required columns: {missing_required_columns}")
    if duplicate_eid_count > 0:
        raise AssertionError(f"Non-unique EID detected ({duplicate_eid_count} rows involved).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
