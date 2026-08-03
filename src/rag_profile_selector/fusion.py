"""Deterministic reciprocal-rank fusion for the approved hybrid profiles.

The inputs to this module are already ranked evidence identifiers.  It does
not retrieve, index, or otherwise resolve evidence itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


RRF_RANK_CONSTANT = 60
"""The fixed reciprocal-rank-fusion constant used by the pilot profiles."""

_APPROVED_HYBRID_K = frozenset((4, 8))


@dataclass(frozen=True)
class FusedEvidence:
    """An evidence identifier together with its reciprocal-rank-fusion score."""

    identifier: str
    score: float


def resolve_hybrid_rrf_k(k: int) -> int:
    """Validate and return an approved hybrid RRF result cutoff.

    Only the approved ``hybrid RRF k=4`` and ``hybrid RRF k=8`` profiles are
    represented here.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k not in _APPROVED_HYBRID_K:
        raise ValueError("hybrid RRF k must be exactly 4 or 8")
    return k


def fuse_reciprocal_rank(
    ranked_identifier_lists: Sequence[Sequence[str]], *, k: int
) -> tuple[FusedEvidence, ...]:
    """Fuse pre-ranked identifier lists for an approved hybrid RRF profile.

    Duplicate identifiers in one input list contribute only at their first
    observed rank.  Ties are broken by the first list/rank at which an
    identifier occurs, then lexicographically by identifier, making output
    independent of mapping iteration order.
    """

    cutoff = resolve_hybrid_rrf_k(k)
    _validate_ranked_lists(ranked_identifier_lists)

    scores: dict[str, float] = {}
    first_seen: dict[str, tuple[int, int]] = {}
    for list_index, identifiers in enumerate(ranked_identifier_lists):
        seen_in_list: set[str] = set()
        for identifier in identifiers:
            if identifier in seen_in_list:
                continue
            seen_in_list.add(identifier)
            rank = len(seen_in_list)
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (
                RRF_RANK_CONSTANT + rank
            )
            first_seen.setdefault(identifier, (list_index, rank))

    ordered_identifiers = sorted(
        scores,
        key=lambda identifier: (-scores[identifier], first_seen[identifier], identifier),
    )
    return tuple(
        FusedEvidence(identifier, scores[identifier])
        for identifier in ordered_identifiers[:cutoff]
    )


def _validate_ranked_lists(ranked_identifier_lists: Sequence[Sequence[str]]) -> None:
    if isinstance(ranked_identifier_lists, (str, bytes)) or not isinstance(
        ranked_identifier_lists, Sequence
    ):
        raise TypeError("ranked_identifier_lists must be a sequence of identifier lists")

    for identifiers in ranked_identifier_lists:
        if isinstance(identifiers, (str, bytes)) or not isinstance(identifiers, Sequence):
            raise TypeError("each ranked input must be a sequence of identifiers")
        for identifier in identifiers:
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("evidence identifiers must be non-empty strings")
