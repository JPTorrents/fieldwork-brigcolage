from __future__ import annotations

import argparse
from collections import Counter

from .diagnostics import sentence_flags
from .docx_loader import load_docx_paragraphs
from .metrics_readability import compute_readability
from .models import DocumentModel, SectionModel, SentenceModel
from .preprocessing import load_nlp
from .reporting import write_outputs
from .sectioning import classify_section_heading


def build_document(path: str) -> DocumentModel:
    nlp = load_nlp()
    paragraphs = load_docx_paragraphs(path)
    sections: list[SectionModel] = []
    current = SectionModel(heading="Front matter", section_type="unknown", start_paragraph_index=0, paragraphs=[])
    for p in paragraphs:
        if p.is_heading:
            if current.paragraphs:
                sections.append(current)
            st = classify_section_heading(p.text)
            current = SectionModel(heading=p.text, section_type=st, start_paragraph_index=p.index, paragraphs=[])
            continue
        if current.section_type == "references":
            p.is_reference_like = True
        p.section_type = current.section_type
        sp = nlp(p.text)
        sent_models = []
        for si, s in enumerate(sp.sents):
            sflags, sm = sentence_flags(s, current.section_type, p.index, si)
            sent_models.append(SentenceModel(index=si, paragraph_index=p.index, section_type=current.section_type, text=s.text.strip(), metrics=sm, flags=sflags))
        p.sentences = sent_models
        p.metrics = {"word_count": len([t for t in sp if t.is_alpha])}
        current.paragraphs.append(p)
    if current.paragraphs:
        sections.append(current)

    doc = DocumentModel(path=path, title_guess=paragraphs[0].text if paragraphs else None, sections=sections)
    for sec in doc.sections:
        s_text = "\n".join(p.text for p in sec.paragraphs)
        s_lens = [s.metrics.get("word_count", 0) for p in sec.paragraphs for s in p.sentences]
        p_lens = [p.metrics.get("word_count", 0) for p in sec.paragraphs]
        sec.metrics = compute_readability(s_text, s_lens, p_lens)
        sec.metrics["word_count"] = sum(p_lens)
        sec.metrics["sentence_count"] = len(s_lens)
        c = Counter(f.severity for p in sec.paragraphs for s in p.sentences for f in s.flags)
        sec.metrics.update(c)
        for p in sec.paragraphs:
            for s in p.sentences:
                doc.flags.extend(s.flags)
    full_text = "\n".join(p.text for s in doc.sections if s.section_type != "references" for p in s.paragraphs)
    all_s_lens = [s.metrics.get("word_count", 0) for sec in doc.sections for p in sec.paragraphs for s in p.sentences]
    all_p_lens = [p.metrics.get("word_count", 0) for sec in doc.sections for p in sec.paragraphs]
    doc.global_metrics = compute_readability(full_text, all_s_lens, all_p_lens)
    return doc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_docx")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    doc = build_document(args.input_docx)
    write_outputs(doc, args.out)
    print(f"Wrote reports to {args.out}")


if __name__ == "__main__":
    main()
