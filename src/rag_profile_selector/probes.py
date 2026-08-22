"""Signals a selector may look at before choosing a retrieval profile.

The selector's job is to pick a profile per query. To do that honestly it may
look only at things knowable *before* retrieval runs: the query itself, and what
a cheap probe of each retriever returns. It may not look at gold labels, at how
the chosen profile scored, or at what happened last time — those are the answer,
and a feature computed from the answer produces a selector that works in
evaluation and nowhere else.

That rule is enforced structurally rather than stated. The extractor accepts
probe results and nothing else, so there is no parameter through which a label
could arrive.

One measurement caution the experiment plan already names: sparse and dense
scores are not comparable as numbers. BM25 scores are unbounded and corpus
dependent; cosine similarities sit in a fixed range. So every cross-retriever
feature here is computed from *ranks* or from within-retriever score ratios,
never from raw score differences across retrievers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

APPROVED_FEATURES = (
    "query_token_count",
    "query_char_count",
    "query_has_number",
    "query_has_comparison",
    "top1_margin",
    "score_decay",
    "rank_agreement",
    "overlap_at_k",
    "unique_source_count",
    "duplicate_ratio",
)


class ProbeError(ValueError):
    """Raised when probe input is malformed or a feature is not approved."""


@dataclass(frozen=True)
class ProbeResult:
    """What one retriever returned for one query, cheaply.

    ``scores`` are that retriever's own numbers and are only ever compared with
    other scores from the same retriever.
    """

    retriever: str
    identifiers: tuple[str, ...]
    scores: tuple[float, ...] = ()
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.scores and len(self.scores) != len(self.identifiers):
            raise ProbeError(f"{self.retriever}: scores and identifiers differ in length")
        if self.sources and len(self.sources) != len(self.identifiers):
            raise ProbeError(f"{self.retriever}: sources and identifiers differ in length")

    def top1_margin(self) -> float:
        """How far ahead the leader is, as a fraction of its own score.

        A ratio rather than a difference, so it means the same thing for a BM25
        score of 40 and a cosine similarity of 0.4.
        """
        if len(self.scores) < 2 or self.scores[0] == 0:
            return 0.0
        return round((self.scores[0] - self.scores[1]) / abs(self.scores[0]), 6)

    def score_decay(self) -> float:
        """How steeply relevance falls off, again within this retriever only.

        A flat list means the retriever could not distinguish its candidates,
        which is the case where a more expensive profile tends to earn its cost.
        """
        if len(self.scores) < 2 or self.scores[0] == 0:
            return 0.0
        return round((self.scores[0] - self.scores[-1]) / abs(self.scores[0]), 6)

    def duplicate_ratio(self) -> float:
        if not self.identifiers:
            return 0.0
        unique = len(set(self.identifiers))
        return round(1 - unique / len(self.identifiers), 6)

    def unique_source_count(self) -> int:
        return len(set(self.sources)) if self.sources else len(set(self.identifiers))


def _cutoff(k: object) -> int:
    """Validate a top-k cutoff.

    `overlap_at_k` raised `ProbeError` for `k < 1`; `rank_agreement` checked
    nothing and returned **0.5** — its documented "nothing to compare" value. Two
    functions, the same parameter, different contracts, and no reason recorded
    for the difference.

    That mattered here more than it looks. 0.5 is supposed to be a fact about the
    data: the retrievers shared fewer than two results. A 0.5 produced by `k=0`
    is a fact about the caller, and the two were indistinguishable in the output.
    This repository exists to keep published numbers honest, so a metric that
    answers a meaningless question with a plausible number is the wrong default.

    `bool` is excluded because it is a subclass of `int` and `True` silently
    means top-1. `profiles.py` and `fusion.py` already exclude it; `probes.py`
    was the one module that did not.

    Non-integers used to reach the slice and raise `TypeError` from deep inside
    (`nan`, `inf`, `2.5`). Same refusal, named at the boundary.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise ProbeError("k must be an integer")
    if k < 1:
        raise ProbeError("k must be at least 1")
    return k


def overlap_at_k(first: ProbeResult, second: ProbeResult, *, k: int = 10) -> float:
    """Share of the top-k that both retrievers returned, in any order."""
    k = _cutoff(k)
    left = set(first.identifiers[:k])
    right = set(second.identifiers[:k])
    if not left or not right:
        return 0.0
    return round(len(left & right) / len(left | right), 6)


def rank_agreement(first: ProbeResult, second: ProbeResult, *, k: int = 10) -> float:
    """Do the retrievers order their shared results the same way?

    Rank-based on purpose: comparing BM25 and cosine scores directly would be
    comparing two different units. Returns 1.0 for identical ordering, 0.0 for
    reversed, and 0.5 when there is nothing to compare.

    **0.5 is a statement about the data**, not about the call. `k` is validated
    so that a meaningless cutoff is refused rather than answered — see `_cutoff`.
    """
    k = _cutoff(k)
    left = list(first.identifiers[:k])
    right = list(second.identifiers[:k])
    shared = [item for item in left if item in right]
    if len(shared) < 2:
        # **단축이지 통제가 아니다.** 이 줄을 지워도 답은 같다 — 공유 항목이 0개나
        # 1개면 아래 이중 루프가 순서쌍을 하나도 만들지 않아 `total == 0`으로 떨어지고
        # 거기서도 0.5가 나온다. 2026-08-23에 변이 감사가 그것을 말했다: 이 조건을
        # `False`로 바꿔도 스위트가 초록이었고, **잡히지 않은 유일한 변이**였다.
        #
        # 남겨두는 이유는 읽는 사람이다. "비교할 쌍이 없으면 0.5"라는 규칙이 여기
        # 한 줄로 보이는 편이, 빈 루프의 부수 효과로 같은 값이 나오는 것보다 낫다.
        # 대신 **통제인 척하지 않도록** 적어둔다 — 실제 통제는 아래의 `0.5`이고,
        # 그 값이 0이나 1로 바뀌면 모르는 것이 아는 것처럼 특징에 실린다.
        return 0.5

    left_rank = {item: index for index, item in enumerate(left)}
    right_rank = {item: index for index, item in enumerate(right)}
    concordant = discordant = 0
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            a, b = shared[i], shared[j]
            same = (left_rank[a] - left_rank[b]) * (right_rank[a] - right_rank[b])
            if same > 0:
                concordant += 1
            elif same < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 0.5
    return round((concordant / total), 6)


def query_features(query: str) -> dict[str, float]:
    """Cheap properties of the question itself."""
    tokens = [token for token in query.split() if token]
    return {
        "query_token_count": float(len(tokens)),
        "query_char_count": float(len(query)),
        "query_has_number": float(any(character.isdigit() for character in query)),
        "query_has_comparison": float(
            any(word in query.lower() for word in
                ("more", "less", "than", "compare", "difference", "보다", "차이", "비교"))
        ),
    }


def extract(query: str, probes: Sequence[ProbeResult], *, k: int = 10) -> dict[str, float]:
    """Every approved feature for one query.

    Takes probes and a query; there is deliberately no parameter for labels or
    outcomes, so a leaking feature cannot be added without changing this
    signature and explaining why.
    """
    if not probes:
        raise ProbeError("at least one probe result is required")

    features = query_features(query)
    primary = probes[0]
    features["top1_margin"] = primary.top1_margin()
    features["score_decay"] = primary.score_decay()
    features["duplicate_ratio"] = primary.duplicate_ratio()
    features["unique_source_count"] = float(primary.unique_source_count())

    if len(probes) >= 2:
        features["overlap_at_k"] = overlap_at_k(probes[0], probes[1], k=k)
        features["rank_agreement"] = rank_agreement(probes[0], probes[1], k=k)
    else:
        # One retriever cannot agree or disagree with anything; 0.5 is the same
        # "no information" value rank_agreement uses, rather than a fabricated 0.
        features["overlap_at_k"] = 0.0
        features["rank_agreement"] = 0.5

    unknown = sorted(set(features) - set(APPROVED_FEATURES))
    if unknown:
        raise ProbeError(f"unapproved features present: {', '.join(unknown)}")
    return features


def assert_no_leakage(features: Mapping[str, float],
                      forbidden: Iterable[str] = ()) -> None:
    """Fail loudly if a feature set contains something it could not have known.

    Called by the training path. The default forbidden set names the shapes a
    label usually arrives in when someone adds a feature without thinking.
    """
    default = {"gold", "label", "answer", "quality", "regret", "outcome", "score_of_chosen"}
    banned = default | set(forbidden)
    offenders = sorted(
        name for name in features
        if any(word in name.lower() for word in banned)
    )
    if offenders:
        raise ProbeError(
            "these features are derived from the answer and would not exist at "
            f"selection time: {', '.join(offenders)}"
        )
