# Draft Analyzer MVP

CLI tool to analyze `.docx` scientific drafts (organization/management studies) for writing burdens and rhetorical risks using local, rule-based heuristics.

## Caveat
“This tool does not judge scientific quality. It identifies likely writing and rhetorical burdens that may make scholarly contribution, evidence, and argument harder to see.”

## Installation
- Python 3.11+
- `pip install -r requirements.txt`
- Install spaCy model: `python -m spacy download en_core_web_sm`

## Usage
```bash
python analyze_draft.py input.docx --out reports/
```

## Outputs
- `report.json`
- `report.md`
- `flags.csv`
- `metrics_by_section.csv`

## Limitations
- Heuristic section/rhetoric detection may miss edge cases.
- Passive and nominalization detection are approximate.
- No LLM rewriting or external API calls.
