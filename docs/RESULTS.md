# Results

## Result status

**이 문장은 더 이상 사실이 아니다.** 원래 여기에는 "내려받은 데이터셋도 모델도 없고, 검색·학습·평가를 실행한 적이 없으며, 경험적 결과가 없다"고 적혀 있었다. 착수 시점에는 맞았고, 그 뒤로 고쳐지지 않았다.

실제로는 한국어 법령 14건(745조문)을 받아 프로파일 5개를 비교했고, 봉인했던 test split을 열었으며, dense 모델 2종을 비교했다. **결과는 [`experiments/KR_LAW_RESULTS.md`](../experiments/KR_LAW_RESULTS.md)에 있다.**

아래 템플릿은 그대로 둔다 — 보고 구조를 정의하는 것이 이 파일의 나머지 역할이고, 그 부분은 여전히 쓰인다.

This file defines the reporting structure only. Planned metrics, stage gates, and acceptance criteria are not results.

## Experiment record template

### Identification

- Result class: pilot / validation / primary held-out / optional extension
- Experiment version and commit hash:
- Date/time and timezone:
- Dataset, corpus, query, evidence-map, and split-manifest versions:
- Index and retrieval-profile versions:
- Feature-extractor and target versions:
- Selector/model and tuning-protocol versions:
- Environment, hardware, seed, and run-manifest versions:

### Reproducibility and leakage checks

- Corpus manifest / evidence mapping validated:
- All query-profile rows remain in one grouped split:
- Test results excluded from development:
- Four profile specifications/checksums matched:
- Feature fitting boundaries validated:
- Raw results complete and immutable:
- Protocol deviations:

### Profile-level retrieval results

For P1-P4, report raw counts, per-query values, aggregation, and sample sizes:

- Recall@4:
- Recall@8:
- Evidence precision:
- Evidence recall:
- Evidence F1:
- MRR or nDCG:
- Retrieval latency median:
- Retrieval latency p95:
- Context token count:
- Deterministic cost and price-manifest version:
- Quality-latency Pareto status:
- Quality-cost Pareto status:

### Selector comparison

Report B1 best fixed, B2 rule-based, B3 query-only, B4 query-plus-probe, and B5 oracle on the same held-out query set:

- Primary gold evidence quality:
- Mean/median/p95 configuration-selection regret:
- Zero-regret share:
- Profile-selection distribution:
- Paired B4-minus-B3 difference:
- Paired bootstrap 95% confidence interval:
- Query and bootstrap-resample counts:
- Held-out answer-quality comparison, if separately approved:

### Stage-gate assessment

- Multiple profiles oracle-best on meaningful validation subsets:
- Validation quality-latency Pareto diversity:
- Validation quality-cost Pareto diversity:
- Stage 2 recommendation: STOP / REQUEST APPROVAL
- Reranker eligibility: NOT EVALUATED / REQUEST APPROVAL

### MVP completion assessment

- Corpus / evidence mapping validated:
- P1-P4 reproducible:
- Oracle distribution analyzed:
- B1-B5 held-out comparison complete:
- Exact regret reported:
- Confidence intervals reported:
- Raw artifacts permit recomputation:
- Prohibited scope expansion absent:
- Overall result: COMPLETE / INVALID / INCOMPLETE

### Interpretation

- Does query-plus-probe outperform query-only selection?
- Does a fixed profile dominate?
- Is profile diversity sufficient to justify adaptation?
- Are quality changes accompanied by cost or latency improvement?
- Negative/null findings:
- Limitations and generalizability:
- Follow-up requiring approval:

## Current findings

이 파일에는 없다. 실제 결과는 [`experiments/KR_LAW_RESULTS.md`](../experiments/KR_LAW_RESULTS.md)에 있다 — 프로파일 비교, headroom, 규칙 선택기, 봉인 split 개봉, dense 모델 비교.
