# 프로토콜 동결 — test split을 열기 전에 확정한 것

**동결 시각**: 2026-08-19 · **동결 전 사용 데이터**: train 19건 + validation 9건 = 28건

test split(12건)은 이 문서가 쓰이기 전까지 한 번도 읽히지 않았다. 봉인의 목적은 **test 결과가 설계를 바꾸지 못하게** 하는 것이므로, 무엇이 확정됐는지를 먼저 적어야 나중에 "결과를 보고 맞춰 쓴 것 아니냐"는 질문에 답할 수 있다.

아래는 전부 train+validation만 보고 정한 것이다.

## 1. 프로파일 집합 (5개, 추가·제거 없음)

| id | 정의 |
|---|---|
| `bm25-word` | Okapi BM25, 어절 토큰 (k1=1.5, b=0.75) |
| `bm25-char` | 동일한 BM25, 문자 3-gram (공백 제거) |
| `hybrid-rrf` | 위 둘의 RRF 융합 (k=60) |
| `dense` | `intfloat/multilingual-e5-small`, mean pooling, `query:`/`passage:` 접두사, 코사인 |
| `hybrid-all` | 셋 전부의 RRF 융합 (k=60) |

## 2. 측정 지표

**MRR@4** — 첫 정답 조문의 역순위. recall이 아니라 순위를 재는 이유는, 4번째에 있는 정답은 사람이 확인하지 않는 정답이기 때문이다.

**regret** — 그 질의에서 최선 프로파일과의 차이. **headroom** — 최선 고정 프로파일의 평균 regret.

## 3. 선택기 후보 (4개, 추가·제거·임계값 조정 없음)

```
rule:lexical-when-confident      top1_margin >= 0.25 → bm25-char, else dense
rule:lexical-when-probes-agree   overlap_at_k >= 0.5 → bm25-char, else dense
rule:dense-for-long-queries      query_length >= 8   → dense, else bm25-char
rule:fuse-when-uncertain         score_decay < 0.5   → dense, else hybrid-all
```

기준선은 `fixed:dense`(train+val에서 최선 고정), 상한은 `oracle`.

## 4. 동결 전 결론

train+validation 28건에서:

- 최선 고정 = `dense`, 평균 regret **0.1071**, headroom **0.1071**
- 규칙 넷 중 하나(`lexical-when-confident`)가 기준선을 이겼으나 **순이득 1건**(개선 2·악화 1) — 우연과 구분되지 않음
- 오라클이 이기는 5건은 전부 `hybrid-rrf`가 정답인데, 이긴 규칙은 `bm25-char`를 고름 → 겨냥해서 이긴 것이 아님
- **결론: 이 코퍼스·프로파일·특징 집합에서 질의별 선택은 만들 근거가 없다**

## 5. test split이 답해야 할 질문

**하나뿐이다.** *위 결론이 보지 않은 데이터에서도 유지되는가.*

test에서 무엇이 나오든 프로파일·지표·규칙·임계값은 바꾸지 않는다. 바꾸면 test는 두 번째 validation이 되고, 그 순간 이 실험에는 검증되지 않은 결론만 남는다.

**결과가 결론과 어긋날 경우**에도 규칙을 손보지 않는다. 어긋났다는 사실 자체를 보고한다 — 28건 표본에서 내린 판단이 12건에서 뒤집힌다면, 그것은 규칙의 문제가 아니라 **표본 크기에 대한 정보**다.
