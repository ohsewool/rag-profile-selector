# Experiment Plan

## 1. Purpose and current status

Test whether query-plus-probe features improve per-query retrieval-profile selection over fixed, rule-based, and query-only learned baselines on held-out HotpotQA queries. The primary evidence is gold evidence-retrieval quality and exact configuration-selection regret.

This document defines a protocol only. No dataset/model has been downloaded, no index exists, and no retrieval or training experiment is authorized or running.

## 2. MVP candidate profiles

Freeze exactly four Stage 1 profiles before the primary run:

| ID | Retrieval method | Returned evidence cutoff |
|---|---|---:|
| P1 | BM25 | 4 |
| P2 | Dense | 4 |
| P3 | Hybrid BM25+dense with RRF | 4 |
| P4 | Hybrid BM25+dense with RRF | 8 |

The frozen profile manifest must include corpus/index checksums, evidence unit, preprocessing, tokenizer/analyzer, dense representation and similarity, RRF formula/constant, candidate depths, duplicate handling, score/rank direction, tie breaking, output ordering, and profile version.

Stage 2 or reranker profiles are not part of this protocol until validation evidence and approval are recorded.

## 3. Data protocol

### 3.1 HotpotQA validation

Before profile execution, verify a fixed HotpotQA dataset/corpus version, query identifiers, supporting-fact/evidence mapping, document-title normalization, missing-document cases, duplicate evidence, and corpus/index correspondence. Record source, license/usage terms, checksums, and validation failures in an immutable manifest.

### 3.2 Grouped split

Assign each query ID to exactly one train, validation, or test split with a fixed, versioned manifest and recorded seed. All query-profile rows, gold labels, probe results, features, predictions, answer-quality results, and reruns for that query inherit the same split.

Fit IDF statistics, normalizers, encoders, imputation, feature selection, model parameters, and other learned state using training data only unless a transformation is explicitly defined as corpus-level and query-label independent. Use validation data for model/profile/threshold selection. Keep the test split sealed until the full primary protocol is frozen.

No test result may influence rule design, hyperparameters, features, target choice, thresholds, profile expansion, reranker justification, retry/escalation logic, or experiment stopping.

## 4. Labels and features

### 4.1 Gold evidence targets

Produce a gold quality value for every query-profile pair from evidence retrieval, not generated answers. Candidate primary targets are Evidence F1, supporting-fact recall, and probability of meeting a frozen minimum-recall threshold. Freeze one primary target and all edge-case rules before test evaluation; retain other approved measures as secondary outcomes.

Do not generate answers for every training query/profile pair. Dataset-native answer metrics are reserved for an approved held-out end-to-end comparison.

### 4.2 Query features

Version definitions for token and character counts, numbers/dates, comparison expressions, rare-word ratio, average IDF, and named-entity/acronym indicators. Fit any vocabulary or statistic without test leakage.

### 4.3 Probe features

Run the same frozen low-cost sparse and dense probes for every eligible query before profile selection. Derive only approved signals: top-1 scores, top-1/top-2 margins, top-10 overlap, rank agreement, score-decay slope, duplicate-document ratio, and unique-source count.

Score calibration and cross-retriever comparisons must be defined carefully because sparse and dense raw scores are not naturally comparable. Probe features may not include gold labels, downstream profile outcomes, answer quality, or execution history.

## 5. Selector and baseline protocols

### B1: Best fixed profile

Choose the single profile with the best validation aggregate under the frozen primary quality metric and deterministic tie-break. Apply it unchanged to every test query.

### B2: Rule-based selector

Create rules using training/validation data only and only approved observable features. Freeze thresholds and tie behavior before test evaluation.

### B3: Query-only learned selector

Train one configuration-conditioned quality model on query-profile rows using query features and profile descriptors.

### B4: Query-plus-probe learned selector

Use the same model family, fitting/tuning budget, candidate profiles, training rows, target, and selection rule as B3, adding only the approved probe-feature group.

### B5: Oracle selector

For analysis only, choose the profile with the highest realized gold evidence quality for each query, using the frozen tie-break. The oracle is not deployable and no oracle/test outcome may become an input feature.

For B3 and B4, select the profile with the highest predicted primary evidence quality. Do not predict cost or latency. If predicted qualities tie, use the predeclared deterministic profile tie-break; a proposed rule is lower deterministic cost, then lower profile ID, subject to approval before execution.

## 6. Run sequence

1. Freeze repository commit, environment, HotpotQA manifest, evidence mapping, grouped split, and validation checks.
2. Freeze the four profile specifications and build plan.
3. Freeze label, feature, model, tuning, baseline, cost, latency, metric, bootstrap, and reporting protocols.
4. Obtain approval before downloads, index construction, or execution.
5. After approval, build the fixed corpus/index artifacts and run all four profiles for each query with complete query-profile rows.
6. Validate profile reproducibility and analyze oracle-profile diversity on training/validation data.
7. Freeze selectors and all primary-analysis choices; keep Stage 2 and extensions disabled.
8. Execute the sealed held-out test comparison once, apart from predeclared deterministic reruns for failures.
9. Compute metrics and paired confidence intervals from immutable raw results.
10. Report positive, null, or negative results and limitations without changing scope.

Any missing profile row, manifest mismatch, split leakage, non-versioned feature, or test-informed decision invalidates the affected primary comparison.

## 7. Metric definitions

- **Recall@4 / Recall@8:** fraction of gold supporting evidence units present within the first 4 or 8 unique retrieved evidence units.
- **Evidence precision:** retrieved evidence units judged relevant divided by unique evidence units returned at the evaluated cutoff.
- **Evidence recall:** retrieved gold evidence units divided by all gold evidence units for the query.
- **Evidence F1:** harmonic mean of evidence precision and evidence recall, with zero-denominator behavior frozen before execution.
- **MRR or nDCG:** freeze one rank-sensitive primary measure, relevance grading, cutoff, and tie handling before test evaluation.
- **Configuration-selection regret:** oracle primary evidence quality minus selected-profile primary evidence quality for the same query and candidate set. Report mean, median, p95, zero-regret share, and distribution.
- **Profile-selection distribution:** count and proportion of queries assigned to each profile by each selector, plus the oracle-best distribution.
- **Retrieval latency:** measure profile retrieval wall-clock latency under a frozen hardware, concurrency, cache, warm-up, and repetition protocol; report median and p95 with sample counts.
- **Context token count:** deterministically tokenize the returned context with a frozen tokenizer/version and report per-query and aggregate values.
- **Deterministic cost:** calculate from versioned token counts, call counts, and fixed prices; report the exact formula and price-manifest version. Do not infer cost with a model.
- **Quality-latency / quality-cost Pareto analysis:** identify non-dominated profiles and selectors using the frozen quality, latency, and cost aggregations; retain all points, not only frontier points.

## 8. Paired bootstrap confidence intervals

For each predeclared pairwise comparison, resample held-out query IDs with replacement so every metric contribution for a query stays paired. Use a recorded pseudorandom seed and a predeclared number of resamples; the planned default is 10,000, subject to approval before execution. Report the observed paired difference, percentile 95% confidence interval, query count, and resample count.

Do not treat overlapping/non-overlapping intervals as a substitute for the exact predeclared comparison. Report all primary comparisons, including unfavorable and inconclusive results.

## 9. Stage gates

### Stage 1 acceptance

- The HotpotQA manifest and evidence mapping validate.
- Every query has exactly one result row for each of P1-P4 or an explicitly classified predeclared failure.
- Repeated deterministic checks reproduce profile ordering and evidence metrics under the frozen environment.
- Oracle profile distribution and profile-level quality/latency/cost diversity are reported.

### Stage 2 consideration

Stage 2 requires a validation-only, predeclared meaningful-diversity rule and explicit approval. Test results may not justify expansion. If one profile dominates or differences are negligible, retain the four-profile study and report that result.

### Stage 3 consideration

Rerankers require completion of the primary experiments, validation evidence that they are oracle-best for a meaningful share or add a Pareto-efficient point, and explicit approval. No reranker result may retroactively alter the primary comparison.

## 10. MVP acceptance criteria

The study is complete when:

1. HotpotQA corpus/evidence mappings and grouped splits are validated and versioned.
2. P1-P4 are executed reproducibly from fixed profile/index specifications after approval.
3. Oracle profile distribution and diversity are analyzed without test leakage.
4. B1-B5 are compared on the same held-out queries and candidate set.
5. Exact per-query selection regret is calculated and summarized.
6. All approved retrieval, latency, token, cost, Pareto, distribution, and paired-bootstrap outputs are reported or explicitly marked inapplicable with justification.
7. Raw query-profile results, manifests, versions, seeds, and protocol deviations permit recomputation.
8. Null or negative findings are reported as valid outcomes, and no prohibited scope expansion occurs.

No superiority threshold is required. If query-plus-probe fails to outperform query-only, a fixed profile dominates, diversity is inadequate, or quality/cost objectives conflict, the experiment still passes when the protocol is valid and the finding is fully reported.

## 11. Result separation

Label results as pilot, validation, primary held-out, or optional extension. Do not merge SciFact, QASPER, reranker, history, verifier, retry, escalation, or abstention results into the core HotpotQA four-profile claim.
