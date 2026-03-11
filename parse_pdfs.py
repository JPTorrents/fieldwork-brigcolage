from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import median

import fitz
from sqlalchemy import text

from pipelines.db import engine

OUT_DIR = Path("data/parsed/text")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARSER_NAME = "pymupdf"
PARSER_VERSION = fitz.VersionBind
NORMALIZER_VERSION = "v1"

SELECT_SQL = text("""
    SELECT
        p.product_id,
        p.publication_id,
        p.product_title,
        p.local_path,
        p.file_hash,
        p.mime_type,
        p.language,
        p.page_count,
        p.has_extractable_text,
        p.parse_status
    FROM products p
    WHERE p.mime_type = 'application/pdf'
      AND p.download_status = 'success'
      AND p.local_path IS NOT NULL
      AND p.has_extractable_text = TRUE
""")

UPSERT_METADATA_SQL = text("""
    INSERT INTO product_metadata (
        product_id,
        detected_title,
        detected_language,
        detected_sections_json,
        detected_references_count,
        detected_tables_count,
        detected_figures_count,
        ocr_used,
        parser_name,
        parser_version
    )
    VALUES (
        :product_id,
        :detected_title,
        :detected_language,
        CAST(:detected_sections_json AS jsonb),
        :detected_references_count,
        :detected_tables_count,
        :detected_figures_count,
        :ocr_used,
        :parser_name,
        :parser_version
    )
    ON CONFLICT (product_id) DO UPDATE SET
        detected_title = EXCLUDED.detected_title,
        detected_language = EXCLUDED.detected_language,
        detected_sections_json = EXCLUDED.detected_sections_json,
        detected_references_count = EXCLUDED.detected_references_count,
        detected_tables_count = EXCLUDED.detected_tables_count,
        detected_figures_count = EXCLUDED.detected_figures_count,
        ocr_used = EXCLUDED.ocr_used,
        parser_name = EXCLUDED.parser_name,
        parser_version = EXCLUDED.parser_version,
        parse_timestamp = now()
""")

MARK_SUCCESS_SQL = text("""
    UPDATE products
    SET parse_status = 'success',
        parse_error_text = NULL,
        last_parse_finished_at = now()
    WHERE product_id = :product_id
""")

MARK_FAILED_SQL = text("""
    UPDATE products
    SET parse_status = 'failed',
        parse_error_text = :parse_error_text,
        last_parse_finished_at = now()
    WHERE product_id = :product_id
""")

MARK_STARTED_SQL = text("""
    UPDATE products
    SET last_parse_started_at = now(),
        parse_error_text = NULL
    WHERE product_id = :product_id
""")

HEADING_RE = re.compile(
    r"^("
    r"\d+(\.\d+)*\s+\S.+"
    r"|"
    r"(annexe|appendix|chapitre|chapter)\s+[A-Z0-9IVXLC]+(\s*[:\-]\s*.+)?"
    r"|"
    r"[A-ZÉÈÀÙÂÊÎÔÛÇ0-9][A-ZÉÈÀÙÂÊÎÔÛÇ0-9'’()\-,:;/ ]{4,}"
    r")$"
)
NUMBERED_HEADING_RE = re.compile(r"^\d+(\.\d+)*\s+\S+")
SECTION_KEYWORD_RE = re.compile(r"^(annexe|appendix|chapter|chapitre)\b", re.IGNORECASE)


PAGE_NUMBER_ONLY_RE = re.compile(
    r"^\s*(page\s+\d+(\s+of\s+\d+)?|\d+\s*/\s*\d+|\d+)\s*$",
    re.IGNORECASE,
)
DOT_LEADER_RE = re.compile(r"\.{2,}\s*\d+\s*$")
MONTH_LINE_RE = re.compile(
    r"^(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)"
    r"(\s+\d{4})?$",
    re.IGNORECASE,
)
ALL_CAPS_BRANDISH_RE = re.compile(r"^[A-ZÉÈÀÙÂÊÎÔÛÇ]\s(?:[A-ZÉÈÀÙÂÊÎÔÛÇ]\s){2,}[A-ZÉÈÀÙÂÊÎÔÛÇ]$")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_inline_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def norm_for_repeat(text: str) -> str:
    t = clean_inline_text(text).lower()
    t = re.sub(r"\d+", "#", t)
    return t


def detect_language(text: str, db_hint: str | None) -> str | None:
    pages = text.split("\n\n<<<PAGE_BREAK>>>\n\n")
    if not pages:
        return db_hint

    sample_pages = pages[:3] + (pages[-2:] if len(pages) > 3 else [])
    sample = " ".join(sample_pages).lower()

    fr_markers = [
        " le ",
        " la ",
        " les ",
        " des ",
        " et ",
        " pour ",
        "résumé",
        "table des matières",
        "méthode",
        "références",
        "annexe",
    ]
    en_markers = [
        " the ",
        " and ",
        " for ",
        " with ",
        " by ",
        "summary",
        "table of contents",
        "methods",
        "references",
        "appendix",
    ]

    fr_score = sum(sample.count(m) for m in fr_markers)
    en_score = sum(sample.count(m) for m in en_markers)
    if fr_score > en_score:
        return "fr"
    if en_score > fr_score:
        return "en"
    return db_hint


def extract_block_candidates(page_dict: dict, page_num: int, page_height: float) -> list[dict]:
    stats = page_font_stats(page_dict)
    body_median = stats["median"] or 0.0
    out = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        lines = block.get("lines", [])
        spans = [span for line in lines for span in line.get("spans", [])]
        if not spans:
            continue

        text = clean_inline_text(" ".join(span.get("text", "") for span in spans))
        if not text:
            continue

        sizes = [float(span.get("size", 0) or 0) for span in spans if float(span.get("size", 0) or 0) > 0]
        if not sizes:
            continue

        bbox = block.get("bbox") or [0, 0, 0, 0]
        y0 = float(bbox[1])
        y1 = float(bbox[3])

        out.append(
            {
                "page_num": page_num,
                "text": text,
                "bbox": bbox,
                "y0": y0,
                "y1": y1,
                "page_height": page_height,
                "max_font_size": max(sizes),
                "median_font_size": median(sizes),
                "body_median_font_size": body_median,
                "is_boldish": any(
                    (int(span.get("flags", 0)) & 16) or "bold" in str(span.get("font", "")).lower() for span in spans
                ),
                "line_count": len(lines),
                "span_count": len(spans),
            }
        )
    return out


def is_probable_heading(c: dict, repeated_texts: set[str], toc_pages: set[int]) -> bool:
    text = c["text"]
    tnorm = norm_for_repeat(text)
    text_l = text.lower()

    if not text or len(text) < 4:
        return False

    if tnorm in repeated_texts:
        return False

    if PAGE_NUMBER_ONLY_RE.match(text):
        return False

    if DOT_LEADER_RE.search(text):
        return False

    if MONTH_LINE_RE.match(text):
        return False

    if ALL_CAPS_BRANDISH_RE.match(text):
        return False

    if c["page_num"] in toc_pages:
        return text_l in {"table des matières", "table of contents"}

    top_margin = c["page_height"] * 0.08
    bottom_margin = c["page_height"] * 0.90

    is_explicitly_numbered = bool(NUMBERED_HEADING_RE.match(text))
    is_section_keyword = bool(SECTION_KEYWORD_RE.match(text))

    if c["y0"] > bottom_margin and not (is_explicitly_numbered or is_section_keyword):
        return False

    if c["page_num"] > 1 and c["y1"] < top_margin:
        return False

    rel_size = c["max_font_size"] / c["body_median_font_size"] if c["body_median_font_size"] > 0 else 1.0

    looks_like_heading_text = is_explicitly_numbered or is_section_keyword or bool(HEADING_RE.match(text))
    strong_typography = rel_size >= 1.20 or (c["is_boldish"] and rel_size >= 1.08)

    if not looks_like_heading_text:
        return False

    if not (strong_typography or is_explicitly_numbered):
        return False

    return True


def sanitize_blocks(blocks: list[dict]) -> list[dict]:
    safe = []
    for block in blocks:
        if block.get("type") != 0:
            continue

        lines = block.get("lines", [])
        spans = [span for line in lines for span in line.get("spans", [])]
        if not spans:
            continue

        text = clean_inline_text(" ".join(span.get("text", "") for span in spans))
        if not text:
            continue

        sizes = [float(span.get("size", 0) or 0) for span in spans if float(span.get("size", 0) or 0) > 0]
        max_font_size = max(sizes) if sizes else 0.0
        is_boldish = any((int(span.get("flags", 0)) & 16) or "bold" in str(span.get("font", "")).lower() for span in spans)

        safe.append(
            {
                "text": text,
                "bbox": block.get("bbox"),
                "max_font_size": max_font_size,
                "is_boldish": is_boldish,
            }
        )
    return safe


# Mention counts only, not true table/figure/reference object extraction counts.
def count_markers(text: str) -> tuple[int, int, int]:
    t = text.lower()
    refs = t.count("références") + t.count("references")
    tables = t.count("tableau ") + t.count("table ")
    figures = t.count("figure ")
    return refs, tables, figures


def page_font_stats(page_dict: dict) -> dict:
    sizes = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                s = float(span.get("size", 0) or 0)
                if s > 0:
                    sizes.append(s)
    if not sizes:
        return {"median": 0.0, "max": 0.0}
    return {"median": median(sizes), "max": max(sizes)}


def repeated_page_furniture(candidates: list[dict], total_pages: int) -> set[str]:
    page_sets = defaultdict(set)

    for c in candidates:
        zone = "middle"
        if c["y1"] < c["page_height"] * 0.12:
            zone = "top"
        elif c["y0"] > c["page_height"] * 0.88:
            zone = "bottom"

        key = (norm_for_repeat(c["text"]), zone)
        page_sets[key].add(c["page_num"])

    repeated = set()
    threshold = 2 if total_pages <= 5 else max(3, int(total_pages * 0.2))

    for (text_norm, zone), pages in page_sets.items():
        if zone in {"top", "bottom"} and len(pages) >= threshold:
            repeated.add(text_norm)

    return repeated


def extract_title_from_first_page(pages: list[dict]) -> str | None:
    if not pages:
        return None

    first_page = pages[0]
    blocks = sorted(
        [b for b in first_page.get("blocks", []) if b.get("text")],
        key=lambda b: (b.get("bbox") or [0, 0, 0, 0])[1],
    )
    if not blocks:
        return None

    page_height = first_page.get("height") or 0
    top_cutoff = page_height * 0.35 if page_height else None
    top_blocks = [b for b in blocks if top_cutoff is None or (b.get("bbox") and b["bbox"][1] <= top_cutoff)]
    candidates = top_blocks or blocks[:6]
    if not candidates:
        return None

    best = max(candidates, key=lambda b: (b.get("max_font_size") or 0, -((b.get("bbox") or [0, 0, 0, 0])[1])))
    title = clean_inline_text(best.get("text", ""))
    return title[:500] if title else None


def parse_pdf(path: Path, db_title: str | None, db_lang: str | None, file_hash: str | None) -> dict:
    with fitz.open(path) as doc:
        pages = []
        full_parts = []
        heading_candidates = []
        total_refs_mentions = 0
        total_tables_mentions = 0
        total_figures_mentions = 0
        toc_pages = []
        all_block_candidates = []

        cursor = 0
        for i, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict")
            raw_text = page.get_text("text")
            normalized = normalize_text(raw_text)

            char_start = cursor
            full_parts.append(normalized)
            cursor += len(normalized)
            char_end = cursor
            cursor += len("\n\n<<<PAGE_BREAK>>>\n\n")

            refs, tables, figures = count_markers(normalized)
            total_refs_mentions += refs
            total_tables_mentions += tables
            total_figures_mentions += figures

            if "table des matières" in normalized.lower() or "table of contents" in normalized.lower():
                toc_pages.append(i)

            block_candidates = extract_block_candidates(page_dict, i, page.rect.height)
            all_block_candidates.extend(block_candidates)

            pages.append(
                {
                    "page_num": i,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "char_start": char_start,
                    "char_end": char_end,
                    "raw_text": raw_text,
                    "normalized_text": normalized,
                    "blocks": sanitize_blocks(page_dict.get("blocks", [])),
                    "images_present": bool(page.get_images(full=True)),
                    "heading_candidates": [],
                }
            )

        repeated_texts = repeated_page_furniture(all_block_candidates, len(doc))
        toc_page_set = set(toc_pages)
        heading_candidates = [c for c in all_block_candidates if is_probable_heading(c, repeated_texts, toc_page_set)]

        full_text = "\n\n<<<PAGE_BREAK>>>\n\n".join(full_parts)
        detected_language = detect_language(full_text, db_lang)

        by_page: dict[int, list[dict]] = {}
        for c in heading_candidates:
            by_page.setdefault(c["page_num"], []).append(c)

        for p in pages:
            p["heading_candidates"] = by_page.get(p["page_num"], [])

        detected_title = extract_title_from_first_page(pages) or db_title or path.stem

        return {
            "product_id": None,
            "source": {
                "local_path": str(path),
                "file_hash": file_hash,
                "mime_type": "application/pdf",
            },
            "parser": {
                "name": PARSER_NAME,
                "version": PARSER_VERSION,
                "normalizer_version": NORMALIZER_VERSION,
            },
            "document": {
                "page_count": len(doc),
                "title_hint": detected_title,
                "language_hint": db_lang,
                "detected_language": detected_language,
            },
            "text": {
                "full_text": full_text,
                "full_text_char_count": len(full_text),
            },
            "pages": pages,
            "diagnostics": {
                "toc_pages": toc_pages,
                "heading_candidates": heading_candidates,
                "warnings": [],
            },
            "metadata_summary": {
                "detected_title": detected_title,
                "detected_language": detected_language,
                "detected_sections_json": json.dumps(heading_candidates, ensure_ascii=False),
                "detected_references_count": total_refs_mentions,
                "detected_tables_count": total_tables_mentions,
                "detected_figures_count": total_figures_mentions,
                "ocr_used": False,
            },
        }


def main() -> None:
    with engine.begin() as conn:
        rows = conn.execute(SELECT_SQL).mappings().all()

    for row in rows:
        product_id = row["product_id"]
        path = Path(row["local_path"])

        try:
            with engine.begin() as conn:
                conn.execute(MARK_STARTED_SQL, {"product_id": product_id})

            payload = parse_pdf(
                path=path,
                db_title=row["product_title"],
                db_lang=row["language"],
                file_hash=row["file_hash"],
            )
            payload["product_id"] = product_id

            tmp_path = OUT_DIR / f"{product_id}.json.tmp"
            out_path = OUT_DIR / f"{product_id}.json"

            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(out_path)

            with engine.begin() as conn:
                conn.execute(
                    UPSERT_METADATA_SQL,
                    {
                        "product_id": product_id,
                        **payload["metadata_summary"],
                        "parser_name": PARSER_NAME,
                        "parser_version": PARSER_VERSION,
                    },
                )
                conn.execute(MARK_SUCCESS_SQL, {"product_id": product_id})

        except Exception as e:
            error_text = f"{type(e).__name__}: {e}"
            print(f"ERROR parsing {product_id}: {error_text}")
            with engine.begin() as conn:
                conn.execute(
                    MARK_FAILED_SQL,
                    {"product_id": product_id, "parse_error_text": error_text},
                )


if __name__ == "__main__":
    main()
