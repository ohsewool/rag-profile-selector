# Decisions

## Approved decisions

### D-001: Bounded empirical contribution

The project tests whether query-plus-probe features improve per-query retrieval-profile selection over fixed, rule-based, and query-only baselines. It will not claim to be the first general RAG Auto-Tuner or a wholly new framework.

### D-002: HotpotQA-only MVP

Use HotpotQA for MVP development and evaluation. SciFact is an optional final dataset after primary experiments; QASPER is a later stretch goal. Neither is part of MVP completion.

### D-003: Four-profile pilot first

The MVP candidate set is BM25 `k=4`, dense `k=4`, hybrid RRF `k=4`, and hybrid RRF `k=8`. Profile expansion is conditional on validation-only diversity evidence and explicit approval.

### D-004: One configuration-conditioned quality model

Train one model on query-profile examples to predict gold evidence quality from query features, optional probe features, and the candidate profile descriptor. Do not train separate models per profile or models that predict cost or latency.

### D-005: Evidence-quality supervision

Use gold evidence quality as the primary target. Evidence F1, supporting-fact recall, and threshold attainment are candidate formulations. Use answer-quality metrics only for held-out end-to-end comparisons; do not generate answers for all training query-profile pairs.

### D-006: Probe contribution isolation

Compare query-only and query-plus-probe selectors using the same conditioned modeling procedure and candidate profiles. Probe features come only from low-cost sparse/dense retrieval signals and may not use gold evidence or execution history.

### D-007: Deterministic cost, measured latency

Calculate cost from versioned token counts, call counts, and fixed prices. Measure retrieval latency directly under a frozen protocol and report median and p95. Do not fit cost or latency predictors.

### D-008: Grouped, leakage-free splits

Keep all results from one query in exactly one train, validation, or test split. Freeze manifests, corpus/index versions, profiles, and feature extractors. Test outcomes cannot influence development decisions.

### D-009: Exact regret and paired comparison

Compute per-query configuration-selection regret against the gold-quality oracle over the same candidate set. Use paired bootstrap confidence intervals resampled by query for system differences.

### D-010: Negative results are complete results

No performance improvement is required for project completion. Fixed-profile dominance, negligible diversity, quality/cost tradeoff failures, and selector underperformance must be reported rather than worked around.

### D-011: Minimal local research architecture

Prefer Python and modular local components. Exclude APIs, dashboards, Docker, MLflow, unnecessary distributed systems, and unrelated workflow infrastructure from the MVP.

## Decisions requiring approval

- Begin application implementation or create functional retrieval, feature, training, or evaluation code.
- Select, install, or add any new dependency, sparse/dense model, tokenizer, index, framework, or model weight.
- Download HotpotQA, a corpus, a model, or any dataset artifact.
- Expand beyond the four pilot profiles, including BM25/dense `k=8` or any reranker.
- Add SciFact or QASPER.
- Add an evidence verifier, retry, escalation, abstention, rewriting, decomposition, routing, or execution-history feature.
- Freeze the primary target formulation, minimum-recall threshold, diversity threshold, profile tie-break, or numerical non-inferiority/success threshold for the evaluated run.
- Add a feature, baseline, metric, dataset, API, paid service, dashboard, Docker, MLflow, or other scope change.
- Run index construction, retrieval experiments, selector training, end-to-end answer generation, or another long-running/heavy job.

## Permanently rejected within the approved project

- LLM-based verifier.
- Complex query rewriting.
- Query decomposition.
- More than one retry.
- Top-one or top-two domain routing.
- General LLM workflow scheduling.
- Multimodal PDF processing.
- Claims of being the first general RAG Auto-Tuner or a completely new framework.

## Decision procedure

Record proposals under `NEEDS_APPROVAL` in `docs/TASKS.md` with rationale, alternatives, validation-only evidence, leakage risk, compute/storage/cost impact, reproducibility impact, and scope effect. Do not proceed until approval is explicit.
