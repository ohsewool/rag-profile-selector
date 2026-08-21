# Project Specification: Probe-Conditioned RAG Retrieval Profile Selection

<!-- historical: 프로젝트 착수 시점 -->
> **이 문서는 기록이다.** 착수 시점의 사양이다. "MVP는 HotpotQA만 쓴다"가 여기서 나온다.
>
> 그 뒤로 달라진 것: **코퍼스가 HotpotQA에서 한국어 법령으로 바뀌었다.** HotpotQA는 한 번도 내려받지 않았다. 경위와 확인 가능한 근거는 [`DECISIONS.md`의 D-002 정정](DECISIONS.md)에 있다.
>
> 지금 상태는 [README](../README.md)에 있다. 여기 적힌 "아직 하지 않았다"·"구현하지 말라"는
> 항목들은 **당시의 사실이자 당시의 제약**이다. 체크박스를 지금 채우면 계획을 그대로
> 따른 것처럼 보이고, 실제로 어디서 갈라졌는지가 사라진다. 그래서 고치지 않고 선언한다.
>
> 낡았다는 것이 선언이면 기록이고, 선언이 아니면 사고다.

Korean title: **Probe 검색 신호 기반 질의별 RAG 검색 프로필 선택**

## 1. Project goal and claim boundary

Conduct a reproducible empirical study of whether low-cost sparse and dense probe-retrieval signals improve per-query retrieval-profile selection compared with fixed, rule-based, and query-only learned baselines.

The approved core contribution is to evaluate whether query-plus-probe features improve retrieval-profile selection on held-out queries, using gold evidence quality and exact configuration-selection regret.

This project must not be described as the first general RAG Auto-Tuner or as a completely new framework. Conclusions must be empirical, bounded to the evaluated datasets, profiles, features, and splits, and valid whether the result is positive or negative.

## 2. Approved core MVP scope

### 2.1 Dataset

The MVP uses **HotpotQA only**. The HotpotQA corpus, query identifiers, gold evidence mapping, and dataset manifest must be validated and versioned before retrieval runs.

### 2.2 Four pilot retrieval profiles

1. BM25, `k=4`.
2. Dense retrieval, `k=4`.
3. Hybrid retrieval with reciprocal rank fusion (RRF), `k=4`.
4. Hybrid retrieval with RRF, `k=8`.

Every implementation-defining detail must be frozen and versioned before experiments, including corpus/index version, text unit, tokenizer/analyzer, dense representation, similarity function, RRF constant, duplicate handling, tie breaking, and returned-evidence unit.

### 2.3 Prediction design

Use one configuration-conditioned model to predict evidence-retrieval quality. Each training example represents a query-profile pair and combines:

- query features;
- probe-retrieval features; and
- a versioned descriptor of the candidate retrieval profile.

The trained selector scores each candidate profile for a query and selects according to the frozen predicted-quality rule. Do not train:

- a cost-prediction model;
- a latency-prediction model; or
- separate models for every retrieval profile.

Cost is calculated deterministically from token counts, call counts, and a fixed versioned price schedule. Latency is measured experimentally and reported using median and p95 values.

### 2.4 Training labels

Use gold evidence quality as the primary training target. Candidate targets are:

- Evidence F1;
- supporting-fact recall; or
- probability of meeting a frozen minimum evidence-recall threshold.

The primary target, any threshold, tie-breaking rule, and target transformations must be frozen using training/validation data before test evaluation.

Dataset-native answer-quality metrics may be used only for held-out end-to-end system comparisons. Do not generate answers for every training query and every profile.

### 2.5 Approved feature groups

Query features may include:

- token count;
- character count;
- numbers or dates;
- comparison expressions;
- rare-word ratio;
- average IDF; and
- named-entity or acronym indicators.

Probe-retrieval features may include:

- BM25 top-1 score;
- BM25 top-1 minus top-2 margin;
- dense top-1 score;
- dense top-1 minus top-2 margin;
- sparse-dense top-10 overlap;
- ranking agreement;
- score-decay slope;
- duplicate-document ratio; and
- unique-source count.

Probe features must be computable from the low-cost sparse and dense probe results without using gold evidence, test outcomes, downstream answer quality, or execution history.

### 2.6 Approved baselines

1. Best fixed profile.
2. Rule-based selector.
3. Query-only learned selector.
4. Query-plus-probe learned selector.
5. Oracle selector.

All learned selectors must use the same grouped split and candidate profiles. The query-only and query-plus-probe systems should use the same configuration-conditioned modeling procedure except for the probe-feature group, so the contribution of probe signals can be isolated.

## 3. Conditional profile stages

### Stage 1: MVP pilot

Run only the four approved pilot profiles on HotpotQA.

### Stage 2: Six core profiles

Only after pilot validation demonstrates meaningful profile diversity and approval is obtained, the candidate set may expand to:

1. BM25, `k=4`.
2. BM25, `k=8`.
3. Dense retrieval, `k=4`.
4. Dense retrieval, `k=8`.
5. Hybrid RRF, `k=4`.
6. Hybrid RRF, `k=8`.

The diversity decision must use a validation-only criterion frozen before test analysis. A proposed criterion is that multiple profiles are oracle-best for non-trivial, predeclared validation subsets or that multiple profiles occupy the validation quality-latency or quality-cost Pareto frontier. The exact non-trivial-share threshold requires approval before execution.

### Stage 3: Optional reranker profiles

Only after the core selector and primary experiments are complete, and only with approval, consider:

7. Hybrid plus reranker, `20 -> 4`.
8. Hybrid plus reranker, `20 -> 8`.

A reranker profile is eligible only if validation evidence shows it is oracle-best for a meaningful portion of queries or adds a new Pareto-efficient point. Reranker model/dependency selection and the meaningful-portion threshold require approval.

## 4. Dataset sequence

- **MVP dataset:** HotpotQA only.
- **Approved final datasets:** HotpotQA and, after the core HotpotQA experiments and separate approval, SciFact.
- **Stretch goal:** QASPER only after the primary experiments are complete and with separate approval.

SciFact and QASPER are not part of MVP completion.

## 5. Split and versioning rules

- Use grouped train, validation, and test splits by query.
- Keep all profile results for one query in the same split.
- Never use a test execution result in selector training, tuning, feature selection, rule design, threshold selection, or profile expansion decisions.
- Freeze dataset manifests and query IDs.
- Freeze corpus and index versions.
- Version every retrieval profile and feature extractor.
- Record seeds, environment, dependencies, model/index identifiers, and raw query-profile outcomes.

## 6. Approved evaluation

Measure:

- Recall@4;
- Recall@8;
- Evidence precision;
- Evidence recall;
- Evidence F1;
- MRR or nDCG, with the choice frozen before test evaluation;
- exact configuration-selection regret;
- profile-selection distribution;
- retrieval latency median;
- retrieval latency p95;
- context token count;
- quality-latency Pareto analysis;
- quality-cost Pareto analysis; and
- paired bootstrap 95% confidence intervals.

For query `q`, candidate set `C`, gold evidence-quality function `Q`, and selected profile `s(q)`, exact selection regret is:

`regret(q) = max[c in C] Q(q, c) - Q(q, s(q))`

The primary regret quality function must match the frozen primary training/evaluation target. Ties must use a deterministic, predeclared rule that does not inspect the test aggregate.

## 7. MVP exclusions

The following are excluded from the MVP:

- domain or shard routing;
- SciFact;
- QASPER;
- reranker profiles;
- execution-history conditioning;
- evidence verifier;
- retry;
- abstention;
- query rewriting;
- query decomposition;
- three operating modes;
- FastAPI;
- dashboard;
- Docker;
- MLflow;
- multimodal PDF processing;
- web search; and
- MCP tools.

## 8. Optional final extensions

Only after the core selector and primary experiments are complete, and only with approval, the project may add:

- SciFact;
- execution-history features as a final ablation;
- reranker profiles justified by validation results;
- simple deterministic evidence checks;
- one escalation to the strongest validated profile; or
- abstention when evidence remains insufficient.

Execution-history features must remain outside the primary contribution.

## 9. Rejected features

Do not add:

- an LLM-based verifier;
- complex query rewriting;
- query decomposition;
- more than one retry;
- top-one or top-two domain routing;
- general LLM workflow scheduling; or
- multimodal PDF processing.

The project also excludes unapproved APIs, dashboards, Docker, MLflow, web search, MCP tools, and any claim of being the first general RAG Auto-Tuner.

## 10. Implementation constraints

- Prefer Python.
- Use a modular architecture manageable by one undergraduate developer.
- Avoid unnecessary distributed systems.
- Keep data protocol, retrieval profiles, probe feature extraction, label calculation, selector training, evaluation, cost calculation, latency measurement, and reporting separable.
- Do not begin implementation until separately approved.

## 11. MVP completion criteria

1. The HotpotQA corpus and gold evidence mapping are validated against a fixed manifest.
2. The four pilot profiles run reproducibly from frozen specifications.
3. The validation and held-out oracle profile distributions are analyzed without leaking test results into development.
4. Best fixed, rule-based, query-only, query-plus-probe, and oracle systems are compared on the held-out test split.
5. Held-out exact configuration-selection regret is calculated from gold evidence quality.
6. Paired bootstrap 95% confidence intervals are reported with the resampling protocol and seed.
7. Negative or null results are accepted, analyzed, and reported without scope manipulation.
8. No prohibited scope expansion occurs.

## 12. Research success criteria

Positive and negative findings are both valid outcomes. The project is complete even if:

- query-plus-probe does not outperform query-only selection;
- one fixed profile dominates most queries;
- reranking later dominates all simpler profiles;
- profile differences are too small to justify adaptation;
- quality is preserved but cost is not reduced; or
- cost is reduced but quality non-inferiority is not achieved.

Success is a reproducible, leakage-free answer to the research question, not a required improvement.

This specification approves planning, not implementation, downloads, or experiment execution.
