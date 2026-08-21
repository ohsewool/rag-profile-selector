# Decisions

## Approved decisions

### D-001: Bounded empirical contribution

The project tests whether query-plus-probe features improve per-query retrieval-profile selection over fixed, rule-based, and query-only baselines. It will not claim to be the first general RAG Auto-Tuner or a wholly new framework.

### D-002: HotpotQA-only MVP

Use HotpotQA for MVP development and evaluation. SciFact is an optional final dataset after primary experiments; QASPER is a later stretch goal. Neither is part of MVP completion.

### D-002 정정 (2026-08-21): MVP 코퍼스는 HotpotQA가 아니라 한국어 법령이다

**D-002는 뒤집혔고, 이 로그에 적히지 않았다.** 위에는 "HotpotQA를 MVP 개발·평가에
쓴다"고 남아 있는데 실제로 만들어진 것은 국가법령정보 OPEN API에서 받은 한국어 법령
14건(745조문) 위의 실험이다. HotpotQA는 **한 번도 내려받은 적이 없다** — 받는
스크립트조차 없다.

이 저장소들은 범위 변경을 기록할 줄 안다. 형제 저장소 `mcp-gateway`의 SPEC은 같은
일을 정확히 해두었다 — *"원래 범위는 테스트베드를 합성 MCP 서버로 제한했다. 다음에
한해, 그리고 다음에만 그 제한을 해제한다."* **여기서만 하지 않았다.**

**당시의 이유는 적혀 있지 않다.** 그래서 지어내지 않는다. 지금 확인할 수 있는 것만
적는다:

- 실제로 쓴 코퍼스는 **공공누리 제1유형**이라 체크섬·매니페스트를 저장소에 남기고
  재현 절차를 공개할 수 있다. 이 저장소의 코퍼스 출처 관문이 요구하는 조건이다.
- 합성 코퍼스 실험이 먼저 있었고(README "초기 실험"), 그 다음이 실제 코퍼스다.
  합성에서 실제로 넘어갈 때 어떤 코퍼스를 고를지는 남아 있지 않다.
- HotpotQA를 받지 않았다는 사실은 검증 가능하다: `scripts/`에 받는 코드가 없고
  `data/`에 흔적이 없다.

**결정을 바꾼 것이 문제가 아니라, 바꾼 것을 적지 않은 것이 문제다.** 이 저장소의
원칙이 "뒤집은 판단은 근거와 함께 남긴다"인데, 정작 이 프로젝트가 **무엇인지**를
정하는 결정이 빠져 있었다. 늦게라도 적되, 당시 이유를 아는 척하지는 않는다.

`docs/PROJECT_SPEC.md`·`EXPERIMENT_PLAN.md`·`TASKS.md`에 남은 HotpotQA 서술은
착수 시점 계획의 기록으로 선언했다.


### D-003: Four-profile pilot first

The MVP candidate set is BM25 `k=4`, dense `k=4`, hybrid RRF `k=4`, and hybrid RRF `k=8`. Profile expansion is conditional on validation-only diversity evidence and explicit approval.

### D-003 정정 (2026-08-22): 프로파일 집합도 바뀌었고, 승인 목록은 배선되지 않았다

D-002가 코퍼스에 대해 그랬듯 **D-003의 프로파일 집합도 뒤집혔다.** 위에는 pilot이
`BM25 k=4`·`dense k=4`·`hybrid RRF k=4`·`hybrid RRF k=8` 넷이라고 적혀 있는데,
한국어 법령 실험이 돌리는 것은 다른 다섯이다:

| 실험이 돌리는 것 | 무엇 |
|---|---|
| `bm25-word` | 어절 토큰화 |
| `bm25-char` | 문자 3-gram |
| `hybrid-rrf` | 위 둘의 RRF 융합 |
| `dense` | e5-small |
| `hybrid-all` | 셋 전부 융합 |

**더 중요한 것**: `src/rag_profile_selector/profiles.py`가 승인되지 않은 설정을
거부하려고 존재하는데, **설정을 고르는 코드가 그것을 부르지 않는다.**
`experiments/`는 `resolve_profile`도 `validate_profile`도 import하지 않고, 이 모듈을
쓰는 것은 테스트뿐이다. 이 프로젝트가 처음 만난 결함과 같은 모양이다 — `access.py`에
권한 헬퍼가 전부 있었고 `ledger.py`가 하나도 import하지 않았다.

**지금 배선하지 않는다.** `RetrievalProfile`은 `(method, k)`이고 실제 프로파일은
그것으로 표현되지 않는다 — `bm25-word`와 `bm25-char`는 **토큰화**가 다른데 모델에
그 축이 없고, `hybrid-all`은 셋 융합이라 둘을 융합하는 `HYBRID_RRF`가 아니다.
모델을 급히 뜯어고치는 것은 지금 상태를 정확히 적어두는 것보다 나쁘다. 다음 코퍼스에서
어휘 중복으로 층화하기로 한 결정과 함께 다뤄야 할 설계 변경이다.

그때까지 `profiles.py`는 **계획의 기록이자 거부 로직의 시험대**이지 활성 관문이
아니며, 모듈 자신이 그렇게 적고 있다.

**성립하는 계약 하나는 확인했다**: 실험이 돌리는 다섯과 `KR_LAW_RESULTS.md`의 표에
실린 다섯이 같다. `tests/test_published_profiles.py`가 고정한다 — 표에 없는 프로파일을
돌리거나 돌리지 않은 프로파일을 표에 싣는 것을 막는다.


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
