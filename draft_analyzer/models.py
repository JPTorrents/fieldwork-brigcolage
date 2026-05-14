from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


Severity = Literal["P1", "P2", "P3", "P4"]
Level = Literal["document", "section", "paragraph", "sentence"]


class Flag(BaseModel):
    severity: Severity
    level: Level
    category: str
    message: str
    evidence: str
    suggestion: str
    section_type: str
    section_heading: str | None = None
    paragraph_index: int | None = None
    sentence_index: int | None = None
    text_excerpt: str | None = None


class SentenceModel(BaseModel):
    index: int
    paragraph_index: int
    section_type: str
    text: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    flags: list[Flag] = Field(default_factory=list)


class ParagraphModel(BaseModel):
    index: int
    section_type: str
    text: str
    style_name: str | None = None
    heading_level: int | None = None
    is_heading: bool = False
    is_block_quote: bool = False
    is_reference_like: bool = False
    sentences: list[SentenceModel] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    flags: list[Flag] = Field(default_factory=list)


class SectionModel(BaseModel):
    heading: str
    section_type: str
    start_paragraph_index: int
    paragraphs: list[ParagraphModel] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    flags: list[Flag] = Field(default_factory=list)


class DocumentModel(BaseModel):
    path: str
    title_guess: str | None = None
    sections: list[SectionModel] = Field(default_factory=list)
    global_metrics: dict[str, Any] = Field(default_factory=dict)
    flags: list[Flag] = Field(default_factory=list)
