# Repository Instructions: Probe-Conditioned RAG Retrieval Profile Selection

Korean title: **Probe 검색 신호 기반 질의별 RAG 검색 프로필 선택**

## Mission

Conduct a reproducible empirical study of whether low-cost sparse and dense probe-retrieval signals improve per-query retrieval-profile selection compared with fixed, rule-based, and query-only learned baselines.

Do not describe this project as the first general RAG Auto-Tuner or as a completely new framework. Claims must remain limited to the approved empirical contribution and observed results.

## Scope authority

- `docs/PROJECT_SPEC.md` is the approved scope baseline.
- The MVP corpus is **Korean statutes** fetched from the 국가법령정보 OPEN API under 공공누리 제1유형 (`scripts/fetch_kr_law_corpus.py`), not HotpotQA. The original plan said HotpotQA and it was never downloaded; the reversal and what can still be verified about it are recorded as the D-002 correction in `docs/DECISIONS.md`. This line said HotpotQA for months and would have steered the next piece of work back to a corpus this project does not use.
- The profiles the experiments actually run are `bm25-word`, `bm25-char`, `hybrid-rrf` (word+char), `dense` (e5-small), and `hybrid-all` (all three fused). The four pilot profiles named in D-003 — BM25 `k=4`, dense `k=4`, hybrid RRF `k=4`, hybrid RRF `k=8` — belong to the HotpotQA plan and are still encoded in `src/rag_profile_selector/profiles.py`, which the experiments do not consult. **I corrected this line once already and got it half right**, keeping the stale four and appending `hybrid-all`; the D-003 correction in `docs/DECISIONS.md` records what the set actually is.
- The primary contribution is the held-out comparison of query-only and query-plus-probe selection using gold evidence quality and exact configuration-selection regret.
- Use one configuration-conditioned model to predict evidence-retrieval quality. Do not train separate models per profile or cost/latency prediction models.
- Calculate cost deterministically from tokens, calls, and a fixed versioned price schedule. Measure latency experimentally and report median and p95.
- Keep execution-history features out of the core contribution.
- Positive and negative findings are equally valid. Never alter the corpus, profiles, split, targets, or reporting to manufacture an improvement.

## Work controls

- Before work, read `docs/PROJECT_SPEC.md`, `docs/DECISIONS.md`, `docs/TASKS.md`, and `docs/STATUS.md`.
- Work only on `AUTO_READY` tasks unless the user explicitly approves an item under `NEEDS_APPROVAL`.
- `AUTO_READY` is limited to repository inspection, environment bootstrap planning, data-protocol planning, profile-specification planning, evaluation-runner planning, and test planning.

> **착수 단계 게이트는 소진됐다 (2026-08-21).** 위의 "AUTO_READY is limited to ... planning"는 착수 시점의 제약이고,
> 그때는 맞았다. 지금 이 저장소에는 한국어 법령 745조문 위의 실험과 프로파일 5종 비교, 봉인 split 개봉까지 끝났고 테스트 346개가 돈다. 그 문장을 그대로 두면 **다음 작업이 이미
> 끝난 단계로 되돌아간다** — 형제 저장소 `rag-profile-selector`의 `AGENTS.md`가 몇 달간
> 쓰지 않는 코퍼스를 지시하고 있던 것과 같은 종류의 사고다.
>
> `docs/TASKS.md`와 `docs/STATUS.md`는 착수 계획의 **기록으로 선언**돼 있다. 지금 상태를
> 알려면 [README](README.md)와 그 문서들이 가리키는 실제 결과를 본다.
>
> **아래 안전 제약은 그대로 유효하다** — 실서비스·실계정·실크리덴셜 금지, 승인 없는
> 다운로드·장시간 작업 금지, 측정하지 않은 결과를 주장하지 않기. 소진된 것은 단계
> 게이트뿐이다.

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
