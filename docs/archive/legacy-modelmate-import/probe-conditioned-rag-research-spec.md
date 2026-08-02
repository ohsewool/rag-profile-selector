# Probe 검색 신호 기반 질의별 RAG 검색 프로필 선택

영문 제목: **Probe-Conditioned RAG Retrieval Profile Selection**

문서 상태: 승인된 최종 연구 사양  
기준일: 2026-07-30

## 범위 선언

이 문서는 다음 연구를 위한 최종 사양이다.

> 저비용 sparse/dense probe 검색 신호가 질의별 검색 프로필 선택의 품질을 개선하는지 재현 가능한 실험으로 검증한다.

이 연구는 일반적인 "RAG Auto-Tuner" 전체를 제안하지 않는다. 새로운 범용 RAG 프레임워크, 완전 자율형 RAG Agent, enterprise 검색 플랫폼을 주장하지 않는다.

첨부된 기존 ModelMate 보고서는 CSV 기반 Agentic AutoML 재설계를 다루며 RAG/document analysis를 범위 밖으로 둔다. 따라서 해당 보고서는 저장소 우선 점검, 단계적 구현, 과장 방지 원칙에만 참고한다. 본 연구의 내용과 충돌할 때는 이 문서와 승인된 12개 결정이 우선한다.

## 범위 구분

### 승인된 핵심 범위

- HotpotQA를 이용한 MVP
- HotpotQA와 SciFact를 이용한 최종 실험
- 4개 pilot 검색 프로필
- 검증 결과에 따른 6개 core 프로필 확장
- query-only와 query-plus-probe selector 비교
- configuration-conditioned `EvidenceUtilityPredictor` 하나
- gold evidence retrieval quality 기반 학습
- 결정론적 비용 계산
- 실측 latency의 median 및 p95 보고
- validation에서 선택한 하나의 budget
- 품질-비용 및 품질-latency Pareto 분석

### 선택적 최종 확장

- 사전 정의된 조건을 통과한 경우에만 2개 reranker 프로필 추가
- 핵심 실험 종료 후 execution-history feature ablation
- 결정론적 evidence check, 최강 검증 프로필로 1회 escalation, 필요 시 abstention
- QASPER는 HotpotQA와 SciFact 실험이 완료된 뒤 stretch goal로만 검토

### 명시적으로 제외한 기능

- top-one/top-two domain-shard routing
- cost prediction model
- latency prediction model
- 학습 질의와 모든 검색 프로필에 대한 answer generation
- MVP verifier, retry, abstention
- LLM 기반 evidence verifier
- query decomposition
- 복잡한 query rewriting
- 2회 이상의 retry
- quality-first, balanced, low-cost의 별도 시스템 3개
- 범용 RAG Auto-Tuner 또는 최초 프레임워크 주장

---

## 1. 최종 문제 정의

RAG 시스템은 같은 질의에도 sparse, dense, hybrid, reranking 등 여러 검색 프로필을 적용할 수 있다. 가장 강한 프로필을 모든 질의에 고정 적용하면 품질은 높을 수 있지만 비용과 latency가 불필요하게 증가할 수 있다. 반대로 저비용 프로필을 고정하면 일부 질의에서 필요한 evidence를 놓칠 수 있다.

본 연구는 각 질의에 대해 저비용 sparse/dense probe 검색을 먼저 수행하고, 그 결과의 score shape, rank stability, sparse-dense agreement 같은 신호를 사용해 주어진 budget 안에서 적합한 검색 프로필을 선택하는 문제를 다룬다.

입력은 질의 `q`, probe 결과 `z(q)`, 후보 프로필 설정 `c_p`이다. 모델은 각 후보 프로필 `p`의 evidence retrieval quality를 예측한다.

```text
U_hat(q, p) = EvidenceUtilityPredictor(query_features(q), probe_features(z(q)), profile_features(c_p))
```

선택기는 validation에서 고정한 budget `B` 안에서 예측 utility가 가장 높은 프로필을 선택한다.

```text
p*(q) = argmax_p U_hat(q, p)
        subject to deterministic_cost(q, p, probe) <= B
```

## 2. 연구 질문과 가설

### 주 연구 질문

저비용 sparse/dense probe 검색 신호를 query feature와 함께 사용하면 query-only selector보다 질의별 evidence retrieval profile 선택 성능이 향상되는가?

### 보조 연구 질문

1. 적응형 선택이 validation에서 고른 단일 fixed profile보다 evidence quality를 유지하거나 높이는가?
2. 적응형 선택이 oracle profile selection과의 regret를 줄이는가?
3. 질의별 profile diversity가 실제로 충분하여 adaptation을 정당화하는가?
4. 개선이 HotpotQA에서만 나타나는지, SciFact에도 재현되는가?
5. 개선이 있더라도 probe overhead를 포함한 비용과 latency 관점에서 의미가 있는가?

### 사전 등록 가설

- **H1:** query-plus-probe selector의 평균 `EvidenceRecall`은 query-only selector보다 높다.
- **H2:** query-plus-probe selector의 oracle regret는 query-only selector보다 낮다.
- **H3:** query-plus-probe selector는 동일 budget에서 validation-best fixed profile보다 높은 평균 `EvidenceRecall`을 보인다.
- **H4:** sparse-dense agreement, top-score margin, score entropy 중 적어도 하나가 profile utility 차이에 추가 설명력을 제공한다.
- **H5:** profile oracle-best 분포가 한 프로필에 지나치게 집중되면 적응형 선택의 이점은 작거나 없다.

H1-H5가 지지되지 않아도 연구는 실패로 간주하지 않는다. 고정 프로필 지배, reranking 지배, profile 차이 부족, 품질 비열등성 실패도 유효한 연구 결과다.

## 3. 고유 기여와 한계

### 고유 기여

1. 동일 질의에 대한 sparse/dense probe 결과를 저비용 진단 신호로 사용한다.
2. 질의와 프로필 설정을 함께 입력하는 단일 configuration-conditioned utility predictor를 사용한다.
3. evidence quality label, 결정론적 비용, 실측 latency를 분리하여 예측 대상의 의미를 명확히 한다.
4. profile diversity gate를 사용해 불필요한 프로필 확장을 막는다.
5. positive/negative result를 모두 허용하는 재현 가능한 평가 프로토콜을 제공한다.

### 한계

- 두 데이터셋만으로 모든 RAG domain을 일반화할 수 없다.
- evidence retrieval quality가 최종 answer quality를 완전히 대변하지 않는다.
- HotpotQA의 Wikipedia 질의와 SciFact의 과학 claim은 실제 기업 문서 검색과 다르다.
- 고정된 모델 및 가격표에 기반한 비용은 실제 provider와 hardware에 따라 달라진다.
- latency는 실험 hardware와 implementation에 의존한다.
- selector 이점이 profile set 설계에 민감할 수 있다.
- 본 연구는 generation, grounding verifier, retry orchestration을 핵심 기여로 다루지 않는다.

## 4. 정확한 4개 pilot 프로필

모든 profile version, tokenizer, index checksum, corpus checksum을 manifest에 고정한다.

### 공통 corpus unit

- **HotpotQA:** 문서 title과 문장 순서를 보존한 paragraph 단위. gold supporting fact가 포함된 title을 relevant evidence unit으로 정의한다.
- **SciFact:** `doc_id`, title, abstract 문장 순서를 보존한 abstract 단위. gold evidence가 연결된 `doc_id`를 relevant evidence unit으로 정의한다.

### 공통 sparse 설정

- Engine: Pyserini/Lucene BM25
- Analyzer: English analyzer
- `k1=0.9`
- `b=0.4`
- 동일 corpus와 동일 analyzer를 모든 sparse/hybrid profile에서 사용

### 공통 dense 설정

- Encoder: `intfloat/e5-base-v2`
- Similarity: L2-normalized embedding의 cosine similarity
- Query prefix: `query: `
- Passage prefix: `passage: `
- 최대 입력 길이: 512 tokens
- 정확한 model revision을 manifest에 기록

### P1. Sparse-10

```yaml
profile_id: p1_bm25_k10
retriever: bm25
candidate_k: 10
output_k: 10
```

### P2. Sparse-50

```yaml
profile_id: p2_bm25_k50
retriever: bm25
candidate_k: 50
output_k: 50
```

### P3. Dense-10

```yaml
profile_id: p3_e5_k10
retriever: e5-base-v2
candidate_k: 10
output_k: 10
```

### P4. Hybrid-20

```yaml
profile_id: p4_hybrid_rrf_k20
sparse_candidates: 50
dense_candidates: 50
fusion: reciprocal_rank_fusion
rrf_constant: 60
output_k: 20
```

Sparse와 dense 결과에서 동일 evidence unit은 하나로 합친다. 동점은 `document_id`의 정렬 순서로 결정한다.

## 5. 6개 또는 8개 프로필로 확장하는 기준

확장 여부는 validation split만 사용해 결정한다. test 결과를 본 뒤 profile set을 바꾸지 않는다.

### 4개 pilot의 다양성 통과 조건

다음 조건을 모두 만족해야 6개 core profile을 검토한다.

1. `epsilon=0.01`의 quality 동점 범위에서 최소 2개 profile이 각각 validation 질의의 5% 이상에서 oracle-best다.
2. 한 profile이 validation 질의의 85%를 초과해 단독 지배하지 않는다.
3. best fixed와 unconstrained oracle의 평균 `EvidenceRecall` 차이가 0.01 이상이다.

조건을 통과하지 못하면 4개 profile로 실험을 종료하고 "profile diversity 부족"을 결과로 보고한다.

### 추가하는 2개 core profile

```yaml
profile_id: p5_e5_k50
retriever: e5-base-v2
candidate_k: 50
output_k: 50
```

```yaml
profile_id: p6_hybrid_rrf_k50
sparse_candidates: 100
dense_candidates: 100
fusion: reciprocal_rank_fusion
rrf_constant: 60
output_k: 50
```

P5/P6를 실제 core set에 남기려면 둘 중 하나 이상이 다음 중 하나를 만족해야 한다.

- validation 질의의 5% 이상에서 oracle-best
- 4-profile oracle 평균 `EvidenceRecall`을 0.01 이상 개선
- 기존 점들에 지배되지 않는 새로운 quality-cost 또는 quality-latency Pareto point 생성

### 선택적 reranker profile

Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`, exact revision 고정

```yaml
profile_id: p7_bm25_rerank_k10
first_stage: bm25
first_stage_k: 100
reranker: cross-encoder/ms-marco-MiniLM-L6-v2
output_k: 10
```

```yaml
profile_id: p8_hybrid_rerank_k10
sparse_candidates: 100
dense_candidates: 100
fusion: reciprocal_rank_fusion
rrf_constant: 60
rerank_pool_k: 100
reranker: cross-encoder/ms-marco-MiniLM-L6-v2
output_k: 10
```

P7/P8은 각 profile이 다음 중 하나를 만족할 때만 최종 실험에 포함한다.

- validation 질의의 5% 이상에서 oracle-best
- 기존 profile set에 없는 Pareto-efficient point를 추가

Reranker가 모든 profile을 지배하면 그 사실 자체를 결과로 보고하며, 적응형 selector의 필요성이 약하다는 결론을 허용한다.

## 6. HotpotQA MVP 데이터셋 프로토콜

HotpotQA는 약 113k개의 multi-hop question과 sentence-level supporting facts를 제공한다. 원본 schema와 official evaluation은 [HotpotQA 공식 사이트](https://hotpotqa.github.io/)와 [공식 저장소](https://github.com/hotpotqa/hotpot)를 따른다.

### 사용 범위

- 데이터: HotpotQA fullwiki train 및 labeled dev
- corpus: official processed Wikipedia corpus
- MVP task: evidence retrieval profile selection
- primary evidence unit: supporting title이 포함된 paragraph
- answer generation: selector 학습에는 사용하지 않음

### 고정 cohort

- seed: `20260730`
- train: official train에서 grouped/stratified sample 15,000 queries
- validation: official train의 나머지에서 grouped/stratified sample 3,000 queries
- test: official labeled dev에서 grouped/stratified sample 3,000 queries
- sampling strata: `type`(`bridge`, `comparison`)와 `level`
- 정확한 query ID 목록을 split manifest로 저장

실험 자원이 허용되면 전체 official train/dev로 scale-up할 수 있지만, 이는 MVP 결과를 덮어쓰지 않는 별도 final run으로 기록한다.

### relevance label

- query의 gold supporting title 집합을 `G_q`로 정의한다.
- profile의 top-k 검색 결과 title 집합을 `R_qp`로 정의한다.
- primary label:

```text
EvidenceRecall(q, p) = |G_q ∩ R_qp| / |G_q|
```

- secondary binary label:

```text
CompleteEvidence(q, p) = 1 if G_q ⊆ R_qp else 0
```

### 데이터 검증

- `_id` 중복 금지
- question, supporting facts, context/corpus title 누락 검사
- supporting sentence index 범위 검사
- corpus에 없는 supporting title 비율 기록
- answer와 supporting facts는 label 생성 외 feature에 사용하지 않음

## 7. SciFact 최종 버전 프로토콜

SciFact는 scientific claim과 evidence/rationale annotation을 제공한다. 데이터 구조는 [공식 저장소](https://github.com/allenai/scifact), 평가 정의는 [공식 evaluation guide](https://github.com/allenai/scifact/blob/master/doc/evaluation.md), 데이터셋 배경은 [원 논문](https://arxiv.org/abs/2004.14974)을 따른다.

### 사용 범위

- corpus: `corpus.jsonl`
- labeled claims: official train + dev
- official test는 label이 공개되지 않으므로 selector 학습/평가에 사용하지 않음
- primary evidence unit: abstract `doc_id`
- secondary evidence unit: rationale sentence

### grouped split

- seed: `20260730`
- labeled train+dev를 합친 뒤 70/15/15 train/validation/test
- group: gold evidence `doc_id`가 연결된 claim들의 connected component
- strata: `SUPPORT`/`CONTRADICT` label과 evidence document count
- 같은 evidence document group이 여러 split에 등장하지 않게 한다.
- exact claim ID 목록과 group assignment를 manifest로 저장한다.

gold evidence가 없는 claim은 retrieval utility predictor 학습에서 제외하고 제외 수를 보고한다. dataset-native held-out end-to-end evaluation에는 해당 claim을 별도 category로 남길 수 있다.

### relevance 및 평가

- document-level `EvidenceRecall`을 primary selector label로 사용
- sentence-level evidence precision/recall/F1은 secondary evaluation으로 사용
- final end-to-end comparison에서만 SciFact의 abstract-level 및 sentence-level evidence/label metric을 사용

## 8. Query feature

모든 feature는 gold answer, gold evidence, test label 없이 계산한다.

### lexical/shape feature

- character count
- whitespace token count
- punctuation count
- digit ratio
- uppercase ratio
- quoted phrase count
- named-entity-like capitalized span count
- unique token ratio
- stopword ratio
- average corpus IDF
- maximum corpus IDF
- rare-token count
- question word indicators: who/what/when/where/why/how
- comparison cue indicators: versus, compared, both, difference
- conjunction count

### semantic feature

- E5 query embedding
- train split에서만 fit한 PCA로 32차원 축소
- PCA transformer를 validation/test에 그대로 적용

### dataset feature

- dataset ID one-hot: HotpotQA 또는 SciFact
- dataset-native gold field는 사용하지 않음

## 9. Probe-retrieval feature

모든 질의에 동일한 probe를 수행한다.

```yaml
sparse_probe:
  retriever: bm25
  top_k: 5
dense_probe:
  retriever: e5-base-v2
  top_k: 5
```

### sparse probe feature

- top-1 score
- top-5 score mean/std
- top1-top2 margin
- top1-top5 margin
- normalized score entropy
- retrieved title lexical overlap with query의 mean/max
- unique result count

### dense probe feature

- top-1 cosine score
- top-5 score mean/std
- top1-top2 margin
- top1-top5 margin
- normalized score entropy
- query-result embedding cosine mean/max
- unique result count

### cross-probe agreement feature

- sparse/dense top-1 same-document indicator
- top-5 set overlap
- top-5 Jaccard similarity
- reciprocal-rank overlap
- rank correlation on shared results
- sparse-only/dense-only result count
- top result title agreement count

### probe 실행 원칙

- probe 결과는 profile 실행 전에 계산하고 cache한다.
- selected profile이 동일 retrieval 결과를 재사용할 수 있으면 중복 검색하지 않는다.
- query-plus-probe selector의 비용과 latency에는 probe overhead를 포함한다.
- fixed, rule-based, query-only baseline에는 사용하지 않은 probe 비용을 부과하지 않는다.

## 10. EvidenceUtilityPredictor 설계

### 입력

각 학습 row는 `(query_id, profile_id)` 쌍이다.

```text
x(q, p) = [
  query_features(q),
  probe_features(q),          # query-plus-probe 모델만
  profile_features(p)
]
```

Profile feature:

- sparse candidate depth
- dense candidate depth
- output depth
- hybrid indicator
- reranker indicator
- rerank pool depth
- profile ID one-hot
- deterministic estimated cost

### 출력

- scalar `predicted_evidence_recall` in `[0, 1]`
- 모든 후보 profile에 대해 동일 모델을 반복 적용
- budget을 만족하는 profile 중 예측값 최대 profile 선택

### 학습 label

- primary: gold evidence 기반 `EvidenceRecall(q, p)`
- secondary diagnostics: `CompleteEvidence`, nDCG, MRR
- cost와 latency는 label로 학습하지 않음

### 기본 모델

재현성과 작은 tabular feature set을 위해 다음 모델을 기본으로 고정한다.

```text
sklearn.ensemble.HistGradientBoostingRegressor
loss="squared_error"
learning_rate=0.05
max_iter=300
max_leaf_nodes=31
l2_regularization=1.0
random_state=20260730
```

- train query의 모든 profile row를 사용
- 각 query의 총 sample weight가 동일하도록 row weight를 `1 / profile_count`로 설정
- hyperparameter 변경은 validation에서 사전 정의된 작은 grid로만 허용
- test split을 model fit, PCA fit, threshold 선택, budget 선택에 사용하지 않음
- prediction은 `[0, 1]`로 clip하되 raw score도 artifact에 저장

### 비용

비용 모델은 학습하지 않는다.

```text
cost_usd =
  embedding_input_tokens × embedding_price_per_token
  + reranker_pair_tokens × reranker_price_per_token
  + generation_input_tokens × generation_input_price_per_token
  + generation_output_tokens × generation_output_price_per_token
  + calls × fixed_call_price
```

- 가격은 `cost_table.yaml`에 source, currency, snapshot date와 함께 고정
- local-only 실행에서 API 비용이 0이면 token/pair 기반 normalized compute cost를 함께 보고
- 가격표는 test 결과를 확인한 뒤 변경하지 않음

### latency

- latency prediction model 없음
- 동일 hardware, warmed index에서 query당 wall-clock 측정
- profile별 1회 warm-up 후 최소 5회 반복
- median과 p95 보고
- index load time과 cold-start time은 별도 보고

## 11. Baseline

### Fixed

- validation에서 평균 `EvidenceRecall`이 가장 높은 단일 profile
- 동일 budget을 만족해야 함
- 동점은 낮은 deterministic cost, 낮은 latency, profile ID 순으로 결정

### Rule-based

Train/validation에서만 고정한 규칙을 사용한다.

- sparse top-score margin이 높고 sparse/dense top-5 overlap이 높으면 낮은 비용 sparse profile
- dense margin이 높고 lexical overlap이 낮으면 dense profile
- sparse/dense disagreement가 크면 hybrid profile
- 불확실하면 validation에서 선택한 안전 profile

Rule threshold는 validation에서 한 번 고정하고 test에서 변경하지 않는다.

### Query-only

- proposed predictor와 동일한 model class와 profile feature 사용
- probe feature만 제외
- query embedding PCA와 lexical feature 사용

### Query-plus-probe

- 본 연구의 proposed selector
- query, sparse/dense probe, profile feature 모두 사용
- probe overhead를 비용과 latency에 포함

### Oracle

- test의 true `EvidenceRecall(q, p)`가 최대인 profile을 질의별 선택
- selector가 도달 가능한 upper bound
- 동점은 낮은 deterministic cost, profile ID 순으로 결정
- 학습 또는 실제 deployment에 사용하지 않음

## 12. Grouped train/validation/test split

### 공통 원칙

- query ID 중복 금지
- normalized question 중복 및 near-duplicate 제거
- 동일 evidence group이 split을 넘지 않도록 group 단위 분할
- split manifest를 version control에 저장
- corpus 자체의 공유는 retrieval benchmark 특성상 허용하지만 gold label은 split별로 격리

### HotpotQA group key

```text
group_key = sorted(unique(supporting_title_set))
```

official dev와 동일 group key가 있는 train query는 train/validation 후보에서 제거한다. 그 후 train/validation을 group 단위로 분할한다.

### SciFact group key

같은 gold evidence `doc_id`를 공유하는 claim을 connected component로 묶어 하나의 group으로 사용한다.

### split 고정

- seed와 ID manifest 고정
- profile, feature, model을 변경해도 같은 split 사용
- split 수정이 필요하면 test 실행 전에 protocol amendment를 작성하고 이전 결과와 분리

## 13. Leakage 방지 규칙

1. test label은 최종 평가 전까지 selector 개발 코드에서 읽지 않는다.
2. PCA, scaler, imputer, feature vocabulary는 train에서만 fit한다.
3. profile expansion, budget, threshold, model hyperparameter는 validation만 사용한다.
4. gold answer, gold evidence title, rationale sentence는 query/probe feature에 포함하지 않는다.
5. supporting title과 동일한 문자열을 feature로 직접 주입하지 않는다.
6. profile utility label 생성 cache와 inference feature cache를 물리적으로 구분한다.
7. test query의 oracle result는 최종 분석에만 사용한다.
8. 동일 query의 profile rows는 항상 같은 split에 둔다.
9. 동일 evidence group이 split을 넘지 않게 한다.
10. answer generator 출력은 selector 학습 label로 사용하지 않는다.
11. test 결과를 본 뒤 profile set, budget, metric, non-inferiority margin을 변경하지 않는다.
12. 모든 artifact에 dataset checksum, split hash, profile manifest hash를 기록한다.

## 14. 평가 지표

### Primary

- mean `EvidenceRecall`
- query-plus-probe와 query-only의 paired delta

### Selector quality

- `CompleteEvidence` rate
- oracle regret:

```text
Regret(q) = OracleEvidenceRecall(q) - SelectedEvidenceRecall(q)
```

- oracle profile classification accuracy
- top-2 profile accuracy
- selected profile distribution

### Retrieval diagnostics

- nDCG@k
- MRR@k
- evidence precision
- HotpotQA complete supporting-title recall
- SciFact document/sentence evidence F1

### Efficiency

- deterministic cost per query: mean, median, p95
- latency per query: median, p95
- probe overhead
- quality-cost Pareto frontier
- quality-latency Pareto frontier

### Held-out end-to-end

Answer generation은 모든 training query/profile에 수행하지 않는다. 최종 test 비교에서 selected subset 또는 전체 test의 selected profile에만 수행한다.

- HotpotQA: dataset-native answer EM/F1과 supporting fact metric
- SciFact: label accuracy/F1과 official evidence metric
- fixed, query-only, query-plus-probe의 동일 generator/prompt 사용

## 15. Ablation study

핵심 ablation:

1. query-only
2. query + sparse probe
3. query + dense probe
4. query + sparse+dense probe
5. cross-probe agreement feature 제거
6. score-shape feature 제거
7. query semantic embedding 제거
8. probe depth `k ∈ {1, 3, 5, 10}`
9. profile configuration numeric feature 제거하고 profile ID만 사용

선택적 final ablation:

- core 실험 종료 후 execution-history feature 추가
- reranker가 gate를 통과한 경우 reranker 포함/제외 비교
- 결정론적 1회 escalation extension

## 16. 통계 검정

- primary comparison: query-plus-probe vs query-only
- group-clustered paired bootstrap 10,000회
- 평균 `EvidenceRecall` 차이와 oracle regret 차이의 95% CI
- paired permutation test 10,000회
- `CompleteEvidence`는 McNemar test
- secondary comparison의 다중 검정은 Holm correction
- latency median/p95는 query-level bootstrap CI
- effect size와 CI를 p-value보다 우선 보고

### 비열등성 기준

비용 절감과 품질 유지 주장을 할 경우:

- quality non-inferiority margin: absolute `EvidenceRecall` 0.01
- 95% CI 하한이 `-0.01`보다 커야 함
- 동시에 deterministic cost 또는 measured median latency가 감소해야 함

조건을 만족하지 않으면 "비용 절감과 품질 유지가 입증되지 않음"으로 보고한다.

## 17. MVP 범위

MVP 완료 범위:

- HotpotQA only
- 4 pilot profile
- BM25/E5 index
- fixed sparse/dense probe
- gold evidence label generation
- fixed/rule/query-only/query-plus-probe/oracle baseline
- one validation-selected budget
- `EvidenceUtilityPredictor`
- grouped split 및 leakage checks
- retrieval quality, cost, median/p95 latency 평가
- profile diversity 결과
- 재현 가능한 config, manifest, CLI, report

MVP 제외:

- SciFact
- reranker profile
- QASPER
- answer generation at scale
- verifier/retry/abstention
- history feature
- UI, API, SaaS integration

## 18. 최종 범위

MVP를 재현한 뒤 다음을 수행한다.

1. HotpotQA core result 고정
2. diversity gate 통과 시 6 profile 확장
3. SciFact에 동일 protocol 적용
4. gate 통과 시 reranker profile 최대 2개 추가
5. held-out end-to-end answer evaluation
6. cross-dataset 결과와 failure mode 분석
7. positive/negative finding을 포함한 최종 보고서

선택적 확장은 core result를 변경하지 않는 별도 appendix로 수행한다.

## 19. 거절된 기능

- domain-shard routing
- multi-domain router
- cost predictor
- latency predictor
- 모든 profile의 answer generation label
- 별도 3-mode 운영 시스템
- MVP verifier/retry/abstention
- LLM verifier
- query decomposition
- complex rewriting
- multi-retry loop
- history-conditioned primary selector
- QASPER primary dataset
- UI dashboard
- 범용 autonomous RAG agent
- enterprise RAG platform

## 20. 마일스톤 순서

1. **M0 저장소/실현 가능성 감사**
   - 현재 저장소 구조, Python runtime, storage, existing dependencies 조사
   - 기존 ModelMate 앱과 연구 코드의 격리 방식 결정
   - dataset license, storage, compute estimate 작성
2. **M1 데이터 및 split**
   - HotpotQA loader, checksum, group split, leakage test
3. **M2 검색 기반**
   - BM25/E5 index와 4개 profile executor
4. **M3 label 및 probe**
   - profile run cache, evidence label, sparse/dense probe feature
5. **M4 selector baseline**
   - fixed, rule, query-only, query-plus-probe, oracle
6. **M5 HotpotQA MVP 평가**
   - primary metric, cost, latency, statistical test
7. **M6 profile diversity gate**
   - 4개 유지 또는 6개 확장 결정
8. **M7 SciFact 재현**
   - grouped split, evidence mapping, final comparison
9. **M8 reranker gate**
   - 사전 기준 통과 시에만 P7/P8 추가
10. **M9 held-out end-to-end**
    - selected profile만 answer generation, native metric 평가
11. **M10 선택적 확장**
    - history ablation 또는 deterministic one-escalation
12. **M11 재현 패키지**
    - one-command pipeline, manifests, tables, plots, limitations

각 마일스톤은 이전 단계 artifact와 검증 결과가 없으면 시작하지 않는다.

## 21. 측정 가능한 완료 기준

### 구현 완료

- 고정 split manifest와 checksum 존재
- 4개 pilot profile이 동일 query set에서 재현 가능
- 모든 train/validation profile utility label 생성
- query-only/query-plus-probe predictor가 동일 interface로 실행
- test label이 model selection에 사용되지 않았음을 자동 검사
- deterministic cost와 latency log 생성
- one-command 또는 단계별 documented pipeline 실행 가능

### 실험 완료

- HotpotQA primary comparison 및 95% CI 보고
- fixed/rule/query-only/query-plus-probe/oracle 비교
- profile oracle-best distribution 보고
- quality-cost, quality-latency Pareto plot 생성
- diversity gate 결정과 근거 기록
- SciFact final replication 완료
- negative finding도 숨기지 않고 보고

### 연구 결론 완료

다음 중 어느 결과가 나와도 위 실험이 정직하게 완료되면 연구는 완료다.

- query-plus-probe가 query-only보다 우수
- 유의한 차이 없음
- fixed profile이 대부분 지배
- reranker가 단순 profile 지배
- 품질은 유지되지만 비용 감소 없음
- 비용은 줄지만 quality non-inferiority 실패

## 22. Codex용 저장소 수준 구현 요구사항

### 구현 전 필수 조사

Codex는 코드를 수정하기 전에 다음을 수행해야 한다.

1. root 및 하위 `AGENTS.md` 읽기
2. repository tree, dependency, runtime, tests, deployment 구조 조사
3. 기존 ModelMate production flow와 RAG 연구 코드의 충돌 여부 확인
4. dataset download/storage/index size와 예상 실행 비용 산정
5. 기존 dependency로 Pyserini/FAISS/E5를 실행할 수 있는지 확인
6. 별도 repository가 더 안전한지 판단
7. 조사 결과와 최소 구현 계획을 먼저 보고

현재 저장소 지침은 RAG/document analysis를 기존 제품 roadmap의 범위 밖으로 둔다. 따라서 기본 선택은 다음 중 하나다.

- **권장:** 별도 research repository
- **허용:** 기존 저장소의 격리된 `research/probe_conditioned_rag/` 디렉터리

기존 `frontend`, `backend/main.py`, `main_parts`, CSV AutoML endpoint를 이 연구 때문에 변경하지 않는다.

### 권장 디렉터리

```text
research/probe_conditioned_rag/
├── README.md
├── pyproject.toml
├── configs/
│   ├── datasets/
│   ├── profiles/
│   ├── experiments/
│   └── cost_table.yaml
├── manifests/
│   ├── datasets/
│   ├── splits/
│   └── models/
├── src/probe_rag/
│   ├── data/
│   ├── indexing/
│   ├── retrieval/
│   ├── features/
│   ├── labels/
│   ├── models/
│   ├── selection/
│   ├── costing/
│   └── evaluation/
├── scripts/
│   ├── download_data.py
│   ├── build_indexes.py
│   ├── run_profiles.py
│   ├── build_features.py
│   ├── train_selector.py
│   └── evaluate.py
├── tests/
├── reports/
└── artifacts/                 # gitignored
```

### 구현 규칙

- config-driven, deterministic, idempotent pipeline
- seed, model revision, package version, corpus/split/profile hash 기록
- import 시 network download 금지
- raw data, index, model artifact, large result는 Git에 commit하지 않음
- query/profile 결과는 Parquet 또는 JSONL로 저장
- 각 artifact key에 dataset/profile/split/version 포함
- interrupted run 재개 가능
- test split label은 evaluation command에서만 접근
- 비용은 `cost_table.yaml`로 계산하며 학습 모델을 만들지 않음
- latency는 실측하고 raw timing log를 보존
- answer generation은 final held-out command로 분리
- UI/API/LLM key 없이 core experiment 실행 가능

### 최소 테스트

- toy corpus에서 BM25/dense/hybrid ranking 결정성
- profile config schema validation
- HotpotQA/SciFact schema parsing
- group split overlap 0 검증
- gold evidence가 feature에 포함되지 않는지 검사
- query의 모든 profile row가 같은 split인지 검사
- label 계산 unit test
- probe cache 재사용 및 cost accounting test
- query-only가 probe feature를 읽지 않는지 검사
- model fit이 test split을 읽지 않는지 검사
- profile expansion gate test
- reranker gate test
- answer generation이 training pipeline에서 호출되지 않는지 검사

### 범위 변경 규칙

profile, split, metric, budget rule을 바꾸려면:

1. test 결과 확인 전 변경
2. protocol amendment 작성
3. config version 증가
4. 이전 결과와 새 결과를 분리

## 참고 구현 자원

- [HotpotQA 공식 사이트](https://hotpotqa.github.io/)
- [HotpotQA 공식 저장소](https://github.com/hotpotqa/hotpot)
- [SciFact 공식 저장소](https://github.com/allenai/scifact)
- [SciFact 공식 평가 문서](https://github.com/allenai/scifact/blob/master/doc/evaluation.md)
- [Pyserini 공식 저장소](https://github.com/castorini/pyserini)
- [BEIR 공식 저장소](https://github.com/beir-cellar/beir)
- [E5 base v2 model card](https://huggingface.co/intfloat/e5-base-v2)
- [MS MARCO MiniLM cross-encoder model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- [Sentence Transformers retrieve-rerank 문서](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)

