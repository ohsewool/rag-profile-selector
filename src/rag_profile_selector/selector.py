"""Choosing a retrieval profile per query, and the baselines that must be beaten.

The project's question is whether per-query selection beats picking one profile
and using it everywhere. That question is only answerable if the comparison is
fair, so the baselines are part of this module rather than an afterthought:

fixed
    always the same profile. The one most systems actually use, and the bar a
    learned selector has to clear to justify existing.

oracle
    always the profile that turned out best. Not achievable — it reads the
    answer — but it bounds how much selection could possibly buy. If the gap
    between `fixed` and `oracle` is small, no selector is worth building, and
    that is a result worth having early.

rule
    a stated heuristic over probe features. Cheap, inspectable, and often
    surprisingly hard to beat.

Regret is the measure: how much quality was given up compared with the profile
that would have been best for that query. Zero means the choice was optimal for
that query, not that the retrieval was good.

Nothing here trains a model. Learned selection needs the corpus that has not
arrived, and putting the scaffolding in first means the comparison is defined
before any number exists to be tempted by.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .probes import ProbeError, assert_no_leakage


class SelectorError(ValueError):
    """Raised when a selection request is malformed or a profile is unknown."""


@dataclass(frozen=True)
class QueryOutcome:
    """What each profile achieved for one query, known only after the fact."""

    query_id: str
    quality: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.quality:
            raise SelectorError(f"{self.query_id}: no profile outcomes supplied")

    @property
    def best_profile(self) -> str:
        return max(self.quality, key=lambda profile: (self.quality[profile], profile))

    @property
    def best_quality(self) -> float:
        return self.quality[self.best_profile]

    def regret_of(self, profile: str) -> float:
        if profile not in self.quality:
            raise SelectorError(f"{self.query_id}: no outcome recorded for {profile}")
        return round(self.best_quality - self.quality[profile], 6)


class Selector:
    """Anything that maps query features to a profile id."""

    name = "selector"

    def choose(self, query_id: str, features: Mapping[str, float]) -> str:
        raise NotImplementedError


class FixedSelector(Selector):
    """Always the same profile — what most systems do."""

    def __init__(self, profile: str) -> None:
        self.profile = profile
        self.name = f"fixed:{profile}"

    def choose(self, query_id: str, features: Mapping[str, float]) -> str:
        return self.profile


class RuleSelector(Selector):
    """A stated heuristic over probe features.

    The rule is supplied as a function so it stays readable and reviewable; a
    selector nobody can explain is not usable in the setting this project cares
    about.
    """

    def __init__(self, rule: Callable[[Mapping[str, float]], str], *,
                 name: str = "rule") -> None:
        self._rule = rule
        self.name = name

    def choose(self, query_id: str, features: Mapping[str, float]) -> str:
        assert_no_leakage(features)
        return self._rule(features)


class OracleSelector(Selector):
    """Always the profile that turned out best. Not achievable, only a bound.

    It reads outcomes, which is exactly what a real selector may not do — hence
    the separate constructor argument rather than a features-based choice, so it
    can never be mistaken for something shippable.
    """

    name = "oracle"

    def __init__(self, outcomes: Mapping[str, QueryOutcome]) -> None:
        self._outcomes = dict(outcomes)

    def choose(self, query_id: str, features: Mapping[str, float]) -> str:
        outcome = self._outcomes.get(query_id)
        if outcome is None:
            raise SelectorError(f"no recorded outcome for {query_id}")
        return outcome.best_profile


@dataclass(frozen=True)
class Evaluation:
    """How a selector did across a set of queries."""

    selector: str
    queries: int
    mean_regret: float
    max_regret: float
    optimal_choices: int
    chosen: Mapping[str, str]

    @property
    def optimal_rate(self) -> float:
        return round(self.optimal_choices / self.queries, 6) if self.queries else 0.0


def evaluate(selector: Selector, outcomes: Sequence[QueryOutcome],
             features: Mapping[str, Mapping[str, float]]) -> Evaluation:
    """Run a selector over recorded outcomes and measure the regret it incurred."""
    if not outcomes:
        raise SelectorError("evaluation requires at least one query outcome")

    regrets: list[float] = []
    chosen: dict[str, str] = {}
    optimal = 0

    for outcome in outcomes:
        query_features = features.get(outcome.query_id)
        if query_features is None:
            raise SelectorError(f"no features supplied for {outcome.query_id}")
        try:
            profile = selector.choose(outcome.query_id, query_features)
        except ProbeError as error:
            raise SelectorError(f"{selector.name}: {error}") from error
        chosen[outcome.query_id] = profile
        regret = outcome.regret_of(profile)
        regrets.append(regret)
        if regret == 0:
            optimal += 1

    return Evaluation(
        selector=selector.name,
        queries=len(outcomes),
        mean_regret=round(sum(regrets) / len(regrets), 6),
        max_regret=round(max(regrets), 6),
        optimal_choices=optimal,
        chosen=chosen,
    )


def headroom(outcomes: Sequence[QueryOutcome], profiles: Sequence[str]) -> dict[str, float]:
    """How much selection could buy at most, before building anything.

    The gap between the best fixed profile and the oracle. If it is near zero,
    per-query selection cannot help on this data no matter how good the selector
    is, and the honest response is to stop rather than to tune.
    """
    if not outcomes:
        raise SelectorError("headroom requires at least one query outcome")

    fixed_means = {
        profile: sum(outcome.regret_of(profile) for outcome in outcomes) / len(outcomes)
        for profile in profiles
    }
    best_fixed = min(fixed_means, key=lambda profile: (fixed_means[profile], profile))
    return {
        "best_fixed_profile": best_fixed,
        "best_fixed_mean_regret": round(fixed_means[best_fixed], 6),
        "oracle_mean_regret": 0.0,
        "headroom": round(fixed_means[best_fixed], 6),
        "per_profile_mean_regret": {
            profile: round(value, 6) for profile, value in sorted(fixed_means.items())
        },
    }
