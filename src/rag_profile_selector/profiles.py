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

# **이 목록은 실험이 쓰지 않는다 (2026-08-22 확인).**
#
# 여기 넷은 착수 계획의 pilot 프로파일이다(`docs/DECISIONS.md` D-003, HotpotQA 시절).
# 실제로 돌아간 실험은 한국어 법령 코퍼스에서 **다른 다섯**을 쓴다 —
# `bm25-word`, `bm25-char`, `hybrid-rrf`, `dense`, `hybrid-all`.
#
# 이 모듈을 import하는 곳은 테스트뿐이다. `experiments/`는 `resolve_profile`도
# `validate_profile`도 부르지 않는다. **승인되지 않은 설정이 비교표에 들어오는 것을
# 막으려고 만든 관문이, 설정을 고르는 코드에서 참조되지 않는다** — 이 프로젝트가
# 처음 만난 결함과 같은 모양이다(`access.py`에 권한 헬퍼가 다 있었고 `ledger.py`가
# 하나도 import하지 않았다).
#
# 지금 배선하지 않는 이유를 적어둔다. `RetrievalProfile`은 `(method, k)`이고,
# 실제 프로파일은 그것으로 표현되지 않는다: `bm25-word`와 `bm25-char`는 **토큰화**가
# 다르고(모델에 그 축이 없다), `hybrid-all`은 셋을 융합한 것이라 두 개를 융합하는
# `HYBRID_RRF`가 아니다. 모델을 급히 뜯어고치는 것은 지금 상태를 정확히 적어두는
# 것보다 나쁘다 - 그건 다음 코퍼스에서 어휘 중복으로 층화하기로 한 결정과 함께
# 다뤄야 할 설계 변경이다.
#
# 그때까지 이 모듈은 **계획의 기록이자 거부 로직의 시험대**다. 활성 관문이 아니다.
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
    """Return the canonical catalog value for whatever *profile*'s fields name.

    This used to say "only when *profile* is unaltered", and that was more than
    the code can do. `profile_id` is derived from `method` and `k`, so altering
    either alters the id in step: a profile tampered from dense to bm25 resolves
    to `bm25-k4` and matches it exactly. The mismatch branch below cannot become
    true while identity is derived from the fields it is meant to police.

    Measured on 2026-08-22 by deleting each rejection in turn and running the
    suite - this one and the `isinstance` one above were among six that no test
    noticed. Writing the test is what showed the branch to be unreachable.

    What does hold: an unapproved combination cannot get through. `k=8` on a
    dense profile makes the id `dense-k8`, and `resolve_profile` refuses it. The
    guarantee is "this names an approved profile", not "this object is the one
    the catalog issued".
    """
    if not isinstance(profile, RetrievalProfile):
        raise ProfileValidationError("profile must be a RetrievalProfile")
    canonical = resolve_profile(profile.profile_id)
    if profile != canonical:  # pragma: no cover - unreachable while the id is derived
        # Kept for the day `profile_id` becomes a stored field, at which point a
        # record can name one profile and carry another's numbers.
        raise ProfileValidationError("profile fields do not match its approved profile_id")
    return canonical
