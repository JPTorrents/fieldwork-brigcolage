from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pandas as pd


def write_outputs(doc_model, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(doc_model.model_dump_json(indent=2), encoding="utf-8")
    flags_rows = [f.model_dump() for f in doc_model.flags]
    pd.DataFrame(flags_rows).to_csv(out / "flags.csv", index=False)

    section_rows = []
    for s in doc_model.sections:
        section_rows.append({
            "section_name": s.heading,
            "section_type": s.section_type,
            "word_count": s.metrics.get("word_count", 0),
            "sentence_count": s.metrics.get("sentence_count", 0),
            "avg_sentence_length": s.metrics.get("avg_sentence_length", 0),
            "flesch_reading_ease": s.metrics.get("flesch_reading_ease", 0),
            "flesch_kincaid_grade": s.metrics.get("flesch_kincaid_grade", 0),
            "P1": s.metrics.get("P1", 0),
            "P2": s.metrics.get("P2", 0),
            "P3": s.metrics.get("P3", 0),
        })
    pd.DataFrame(section_rows).to_csv(out / "metrics_by_section.csv", index=False)

    (out / "report.md").write_text(build_markdown(doc_model), encoding="utf-8")


def build_markdown(doc_model) -> str:
    sev = Counter(f.severity for f in doc_model.flags)
    lines = ["# Draft Writing Diagnostic Report", "", "## Executive diagnosis"]
    lines += [f"- Number of sections: {len(doc_model.sections)}", f"- Total flags: {len(doc_model.flags)} (P1={sev.get('P1',0)}, P2={sev.get('P2',0)}, P3={sev.get('P3',0)}, P4={sev.get('P4',0)})"]
    lines += ["", "## Caveat", "Scientific writing can be dense; these metrics indicate relative burden, not universal quality."]
    lines += ["", "## Priority flags"]
    for f in [x for x in doc_model.flags if x.severity in {"P1", "P2"}][:10]:
        lines.append(f"- [{f.severity}] {f.section_type} p{f.paragraph_index} s{f.sentence_index}: {f.message} | {f.evidence}")
    lines += ["", "## Appendix", "Nominalizations are not automatically wrong; clusters can hide actors/actions."]
    return "\n".join(lines)
