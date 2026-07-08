# Bugzilla Defect Intelligence — Readout

## Corpus
- 13,019 bugs (Firefox / Core / Thunderbird), 281 components, 93% with descriptions
- 4,987 hard-labeled duplicate pairs for retrieval eval

## Duplicate retrieval (semantic, hard labels) — headline
- precision@1=0.362  precision@5=0.595  precision@10=0.667  MRR=0.468
- reading: a duplicate's true master is surfaced in the top-10 67% of the time

- ceiling: cross-component dups (35% of pairs) P@10=0.57 vs 0.72 within-component

## Clustering (externally validated by dup labels)
- 18 themes; duplicate pairs land in the SAME cluster 72% of the time (random 6%, ×11)

## Extraction agent (LangGraph supervisor + tool loop)
- 18 distinct trajectories over 1762 bugs; 46% solved with zero retrieval; mean 5.4 supervisor steps
- valid yield: first-pass 58% -> after critique-retry 80% (agent self-correction +22%)

## Auto-rater reliability
- LLM-judge vs human Cohen's κ: 0.390

## Themes (top, 95% Wilson CI)
- Performance Issues Firefox: 9.7% [9.2%, 10.2%]
- Security Vulnerabilities: 8.4% [8.0%, 8.9%]
- UI Rendering Issues: 7.5% [7.1%, 8.0%]
- Intermittent bugs cluster: 7.4% [7.0%, 7.9%]
- WPT Sync Tasks: 7.4% [6.9%, 7.8%]
- Inbox display issues: 7.0% [6.6%, 7.5%]
