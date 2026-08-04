"""Exact configuration-selection regret for synthetic profile qualities.

This module deliberately accepts already supplied quality values.  It does not
define a quality metric or select between profiles with equal quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Mapping, Sequence


class RegretValidationError(ValueError):
    """Raised when synthetic profile-quality outcomes are not well formed."""


@dataclass(frozen=True)
class QueryProfileQualityOutcomes:
    """Immutable supplied quality outcomes for every candidate of one query.

    ``candidate_profiles`` defines the complete candidate set.  ``qualities``
    must provide one finite real-valued quality for exactly that set.
    """

    query_id: str
    candidate_profiles: Sequence[str]
    qualities: Mapping[str, Real]

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id:
            raise RegretValidationError("query_id must be a non-empty string")

        candidates = tuple(self.candidate_profiles)
        if not candidates:
            raise RegretValidationError("candidate_profiles must not be empty")
        if any(not isinstance(profile, str) or not profile for profile in candidates):
            raise RegretValidationError(
                "candidate_profiles must contain non-empty string identifiers"
            )
        if len(set(candidates)) != len(candidates):
            raise RegretValidationError("candidate_profiles must not contain duplicates")

        if not isinstance(self.qualities, Mapping):
            raise TypeError("qualities must be a mapping from profile to quality")
        quality_values = dict(self.qualities)
        if set(quality_values) != set(candidates):
            raise RegretValidationError(
                "qualities must contain exactly one score for every candidate profile"
            )
        for profile, quality in quality_values.items():
            if isinstance(quality, bool) or not isinstance(quality, Real):
                raise TypeError("quality scores must be real numbers")
            if not isfinite(quality):
                raise RegretValidationError("quality scores must be finite")

        object.__setattr__(self, "candidate_profiles", candidates)
        object.__setattr__(self, "qualities", MappingProxyType(quality_values))


def calculate_exact_regret(
    outcomes: QueryProfileQualityOutcomes, selected_profile: str
) -> Real:
    """Return ``max Q(q, c) - Q(q, s(q))`` for supplied outcomes.

    Equal maximum quality values yield the same regret regardless of which
    tied profile was selected.  This function intentionally does not choose a
    profile when qualities tie.
    """

    if not isinstance(outcomes, QueryProfileQualityOutcomes):
        raise TypeError("outcomes must be QueryProfileQualityOutcomes")
    if selected_profile not in outcomes.qualities:
        raise RegretValidationError("selected_profile must be a candidate profile")

    return max(outcomes.qualities.values()) - outcomes.qualities[selected_profile]
