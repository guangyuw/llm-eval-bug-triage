# Annotation Guide · human_labels.csv

Your task: judge whether the **3 LLM-extracted fields faithfully and reasonably reflect the bug text**.
(You are NOT judging whether the bug is important or should be fixed — only whether the extraction is accurate. No Firefox engineering knowledge required.)

For each row, read the 4 source columns (`component` / `summary` / `description`) and the 3 LLM output columns (`llm_area` / `llm_severity` / `llm_kind`), then fill in `human_good`:

- `1` (good): `area` matches the text topic; `severity`/`kind` are consistent with the tone of the text
    (e.g. text mentions crash / use-after-free / security → `kind=security` or `kind=crash`, `severity=high` is reasonable)
- `0` (bad): `area` is clearly off-topic (text is about PDF but labelled as Networking), OR `kind`/`severity` is obviously wrong
    (e.g. a minor UI tweak labelled as `kind=crash`, `severity=high`)
- (blank): text is too short or too technical to judge — **skip it; blank rows are excluded from the κ calculation**

Field definitions:
- `llm_area`: affected subsystem / topic (short phrase)
- `llm_severity`: `low` | `med` | `high`
- `llm_kind`: `crash` | `ui` | `performance` | `security` | `data` | `other`

Label 20–40 rows to get a stable Cohen's κ estimate. Save the file and re-run Step 6 in the notebook.
