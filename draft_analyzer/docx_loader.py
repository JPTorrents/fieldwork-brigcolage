from __future__ import annotations

from docx import Document as DocxDocument

from .models import ParagraphModel


def _heading_level(style_name: str | None) -> int | None:
    if not style_name:
        return None
    s = style_name.lower()
    if "heading" in s:
        parts = s.split()
        for p in parts:
            if p.isdigit():
                return int(p)
    return None


def load_docx_paragraphs(path: str) -> list[ParagraphModel]:
    doc = DocxDocument(path)
    paragraphs: list[ParagraphModel] = []
    for idx, p in enumerate(doc.paragraphs):
        text = p.text.strip()
        if not text:
            continue
        style_name = p.style.name if p.style else None
        level = _heading_level(style_name)
        is_heading = level is not None or (style_name and "heading" in style_name.lower())
        is_block_quote = bool(style_name and "quote" in style_name.lower())
        paragraphs.append(
            ParagraphModel(
                index=idx,
                section_type="unknown",
                text=text,
                style_name=style_name,
                heading_level=level,
                is_heading=is_heading,
                is_block_quote=is_block_quote,
                is_reference_like=False,
            )
        )
    return paragraphs
