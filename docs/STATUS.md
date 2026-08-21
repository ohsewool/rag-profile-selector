# Status

<!-- historical: 프로젝트 착수 시점 -->
> **이 문서는 기록이다.** 착수 시점의 단계 게이트 상태다 — 무엇을 아직 하지
> 않았고 다음에 무엇이 승인됐는지를 적어둔 것이다.
>
> 그 뒤로 달라진 것: 한국어 법령 코퍼스 14건(745조문)을 받아 BM25·dense·융합 5개 프로파일을 비교했고, 봉인했던 test split을 열어 결론을 확인했으며, dense 모델 2종을 비교했다. 결과는 [`experiments/KR_LAW_RESULTS.md`](../experiments/KR_LAW_RESULTS.md).
>
> 지금 상태는 [README](../README.md)에 있다. 여기 적힌 "아직 하지 않았다"는 문장들은
> **당시의 사실**이고 지금은 맞지 않는다. 조용히 고치면 착수 때 무엇을 의도적으로
> 미뤘는지가 사라지므로, 고치는 대신 선언한다.
>
> 낡았다는 것이 선언이면 기록이고, 선언이 아니면 사고다.

## Current phase

**Approved specification and planning documentation.** The repository contains the controlled scaffold and the bounded probe-conditioned RAG study design. Application implementation, data/model downloads, index construction, retrieval runs, feature extraction, selector training, and experiments have not started.

## Completed

- Repository initialized on `main`.
- HotpotQA-only MVP, four pilot profiles, configuration-conditioned prediction design, feature groups, labels, baselines, split rules, metrics, exclusions, and completion criteria documented.
- Core scope, conditional final extensions, and rejected features separated.
- Leakage controls, profile stage gates, exact regret, deterministic cost, measured latency, and paired-bootstrap protocol documented.
- Planning-only `AUTO_READY` tasks and approval boundaries defined.

## Not started

- Repository inspection and environment bootstrap planning.
- HotpotQA manifest/evidence-map planning or validation.
- Dependency, sparse/dense retriever, tokenizer, model, or index selection.
- Application or test implementation.
- Dataset/model downloads or index construction.
- Profile runs, feature extraction, selector training, answer generation, or evaluation.
- Stage 2, SciFact, QASPER, reranking, history ablation, verifier, retry/escalation, or abstention.

## Safety and artifact state

- No datasets, corpora, models, or weights have been downloaded.
- No indexes, features, learned models, or result artifacts have been created.
- No APIs, dashboards, Docker, or MLflow have been added.
- No retrieval experiment or long-running job has been started.

## Next authorized work

Start with A1 in `docs/TASKS.md`: inspect and inventory the repository. This is a bounded planning task and does not authorize implementation or downloads.
