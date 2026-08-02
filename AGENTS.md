# Repository Instructions: Probe-Conditioned RAG Retrieval Profile Selection

Korean title: **Probe 검색 신호 기반 질의별 RAG 검색 프로필 선택**

## Mission

Conduct a reproducible empirical study of whether low-cost sparse and dense probe-retrieval signals improve per-query retrieval-profile selection compared with fixed, rule-based, and query-only learned baselines.

Do not describe this project as the first general RAG Auto-Tuner or as a completely new framework. Claims must remain limited to the approved empirical contribution and observed results.

## Scope authority

- `docs/PROJECT_SPEC.md` is the approved scope baseline.
- The MVP uses HotpotQA only and exactly four pilot profiles: BM25 `k=4`, dense `k=4`, hybrid RRF `k=4`, and hybrid RRF `k=8`.
- The primary contribution is the held-out comparison of query-only and query-plus-probe selection using gold evidence quality and exact configuration-selection regret.
- Use one configuration-conditioned model to predict evidence-retrieval quality. Do not train separate models per profile or cost/latency prediction models.
- Calculate cost deterministically from tokens, calls, and a fixed versioned price schedule. Measure latency experimentally and report median and p95.
- Keep execution-history features out of the core contribution.
- Positive and negative findings are equally valid. Never alter the corpus, profiles, split, targets, or reporting to manufacture an improvement.

## Work controls

- Before work, read `docs/PROJECT_SPEC.md`, `docs/DECISIONS.md`, `docs/TASKS.md`, and `docs/STATUS.md`.
- Work only on `AUTO_READY` tasks unless the user explicitly approves an item under `NEEDS_APPROVAL`.
- `AUTO_READY` is limited to repository inspection, environment bootstrap planning, data-protocol planning, profile-specification planning, evaluation-runner planning, and test planning.
- Current approval does not authorize application implementation, dataset/model downloads, index construction, retrieval experiments, or long-running jobs.
- Prefer Python and a modular design manageable by one undergraduate developer.
- Do not add APIs, dashboards, Docker, MLflow, unnecessary distributed systems, paid services, or production infrastructure.
- Do not add a dependency, dataset, retrieval profile, feature group, baseline, metric, verifier, retry, or scope change without approval.
- Keep fixed dataset manifests, index/corpus versions, profile versions, feature-extractor versions, seeds, environment details, and raw per-query/profile results for reproducibility.
- Keep all results for a query in one grouped split. Test results must never influence training, tuning, rule construction, feature selection, or profile selection.
- Never claim an experiment result until it has been run and recorded in `docs/RESULTS.md`.

## Prohibited shortcuts

- Do not generate answers for every training query/profile pair.
- Do not use dataset-native answer-quality metrics as selector training labels; reserve them for held-out end-to-end comparisons.
- Do not add LLM-based verification, complex rewriting, query decomposition, more than one retry, domain routing, general workflow scheduling, multimodal PDF processing, web search, or MCP tools.

## Current phase

The approved research specification and planning protocol are established. Implementation, downloads, indexing, feature extraction, model training, and experiments have not started.
