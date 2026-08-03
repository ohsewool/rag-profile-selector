"""Approved retrieval-profile definitions and strict validation.

This module intentionally contains configuration only.  It does not implement
retrieval or ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Final, Mapping


class ProfileValidationError(ValueError):
    """Raised when a retrieval profile is outside the approved catalog."""


class RetrievalMethod(str, Enum):
    """The complete, fixed set of approved retrieval methods."""

    BM25 = "bm25"
    DENSE = "dense"
    HYBRID_RRF = "hybrid_rrf"


_APPROVED_COMBINATIONS: Final[frozenset[tuple[RetrievalMethod, int]]] = frozenset(
    {
        (RetrievalMethod.BM25, 4),
        (RetrievalMethod.DENSE, 4),
        (RetrievalMethod.HYBRID_RRF, 4),
        (RetrievalMethod.HYBRID_RRF, 8),
    }
)


def _profile_id(method: RetrievalMethod, k: int) -> str:
    """Return the stable identifier for an approved method/k combination."""
    return f"{method.value.replace('_', '-')}-k{k}"


@dataclass(frozen=True, slots=True)
class RetrievalProfile:
    """An immutable, approved retrieval-profile configuration."""

    method: RetrievalMethod
    k: int

    def __post_init__(self) -> None:
        if not isinstance(self.method, RetrievalMethod):
            raise ProfileValidationError("method must be a RetrievalMethod")
        if isinstance(self.k, bool) or not isinstance(self.k, int):
            raise ProfileValidationError("k must be an integer")
        if (self.method, self.k) not in _APPROVED_COMBINATIONS:
            raise ProfileValidationError(
                f"unsupported approved-profile combination: {self.method.value!r}, k={self.k}"
            )

    @property
    def profile_id(self) -> str:
        """Stable profile identifier derived only from the immutable fields."""
        return _profile_id(self.method, self.k)

    def to_dict(self) -> dict[str, object]:
        """Return the fixed serialization payload for this profile."""
        return {"k": self.k, "method": self.method.value, "profile_id": self.profile_id}

    def serialize(self) -> str:
        """Return a canonical, deterministic JSON representation."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


_BM25_K4: Final = RetrievalProfile(RetrievalMethod.BM25, 4)
_DENSE_K4: Final = RetrievalProfile(RetrievalMethod.DENSE, 4)
_HYBRID_RRF_K4: Final = RetrievalProfile(RetrievalMethod.HYBRID_RRF, 4)
_HYBRID_RRF_K8: Final = RetrievalProfile(RetrievalMethod.HYBRID_RRF, 8)

APPROVED_PROFILES: Final[tuple[RetrievalProfile, ...]] = (
    _BM25_K4,
    _DENSE_K4,
    _HYBRID_RRF_K4,
    _HYBRID_RRF_K8,
)

_PROFILES_BY_ID: Final[Mapping[str, RetrievalProfile]] = {
    profile.profile_id: profile for profile in APPROVED_PROFILES
}


def resolve_profile(profile_id: str) -> RetrievalProfile:
    """Resolve an approved profile identifier, rejecting every unknown value."""
    if not isinstance(profile_id, str):
        raise ProfileValidationError("profile_id must be a string")
    try:
        return _PROFILES_BY_ID[profile_id]
    except KeyError as error:
        raise ProfileValidationError(f"unknown approved profile_id: {profile_id!r}") from error


def validate_profile(profile: RetrievalProfile) -> RetrievalProfile:
    """Return the canonical catalog value only when *profile* is unaltered."""
    if not isinstance(profile, RetrievalProfile):
        raise ProfileValidationError("profile must be a RetrievalProfile")
    canonical = resolve_profile(profile.profile_id)
    if profile != canonical:
        raise ProfileValidationError("profile fields do not match its approved profile_id")
    return canonical
