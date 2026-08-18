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


def overlap_at_k(first: ProbeResult, second: ProbeResult, *, k: int = 10) -> float:
    """Share of the top-k that both retrievers returned, in any order."""
    if k < 1:
        raise ProbeError("k must be at least 1")
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
    """
    left = list(first.identifiers[:k])
    right = list(second.identifiers[:k])
    shared = [item for item in left if item in right]
    if len(shared) < 2:
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
