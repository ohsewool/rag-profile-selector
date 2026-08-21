"""같은 인자에 두 함수의 계약이 달랐다.

`overlap_at_k`는 `k < 1`에 `ProbeError`를 냈고, `rank_agreement`는 **아무것도
검사하지 않고 0.5를 돌려줬다** — 그 함수가 문서화한 "비교할 것이 없다"는 값이다.
두 함수, 같은 파라미터, 다른 계약, 그리고 왜 다른지는 어디에도 없었다.

여기서는 그 차이가 보기보다 크다. **0.5는 데이터에 대한 사실이어야 한다**: 두
검색기가 공유한 결과가 둘 미만이었다는 뜻이다. `k=0`이 만든 0.5는 호출자에 대한
사실이고, 출력에서 둘은 구별되지 않는다. 이 저장소는 공개하는 수치가 진짜인지를
지키려고 있으니, 뜻 없는 질문에 그럴듯한 숫자로 답하는 것이 가장 나쁜 기본값이다.

`nan`·`inf`·`2.5`는 예전에 슬라이스까지 내려가 `TypeError`를 냈다. 같은 거절이지만
경계에서 이름을 갖고 난다.

`bool`은 `int`의 하위형이라 `True`가 조용히 top-1이 된다. `profiles.py`와
`fusion.py`는 이미 배제하고 있었고 `probes.py`만 안 하고 있었다 — **한 저장소가 같은
판단을 두 곳에서 하고 한 곳에서 빠뜨린** 모양이다.

2026-08-22 경계 감사에서 나왔다. 같은 회차에 `agent-safety-core`는 `nan` TTL로
만료하지 않는 lease를, `mcp-gateway`는 `nan` 레이트로 100/100 통과를 만들 수 있었다.
"""

import pytest

from rag_profile_selector.probes import (ProbeError, ProbeResult, overlap_at_k,
                                         rank_agreement)

METRICS = (overlap_at_k, rank_agreement)


@pytest.fixture
def pair():
    first = ProbeResult(retriever="bm25", identifiers=("a", "b", "c"), sources=("s",) * 3)
    second = ProbeResult(retriever="dense", identifiers=("a", "b", "z"), sources=("s",) * 3)
    return first, second


@pytest.mark.parametrize("metric", METRICS, ids=lambda fn: fn.__name__)
class TestBothMetricsRefuseTheSameCutoffs:
    @pytest.mark.parametrize("k", [0, -1, -10])
    def test_a_cutoff_below_one_is_refused(self, metric, pair, k):
        with pytest.raises(ProbeError, match="at least 1"):
            metric(*pair, k=k)

    @pytest.mark.parametrize("k", [float("nan"), float("inf"), 2.5, "2", None, True])
    def test_a_non_integer_cutoff_is_refused(self, metric, pair, k):
        with pytest.raises(ProbeError, match="integer"):
            metric(*pair, k=k)

    def test_a_usable_cutoff_still_answers(self, metric, pair):
        """전부 거절하는 검증은 전부 거절하는 것으로도 통과한다."""
        assert isinstance(metric(*pair, k=2), float)

    def test_the_default_cutoff_is_usable(self, metric, pair):
        """기본값이 새 검증에 걸리면 인자 없이 부르는 모든 곳이 죽는다."""
        assert isinstance(metric(*pair), float)


class TestZeroPointFiveMeansSomethingAboutTheData:
    def test_it_is_returned_when_the_retrievers_share_too_little(self, pair):
        first, second = pair
        apart = ProbeResult(retriever="dense", identifiers=("x", "y", "z"), sources=("s",) * 3)
        assert rank_agreement(first, apart, k=3) == 0.5

    def test_it_is_no_longer_reachable_by_a_meaningless_cutoff(self, pair):
        """예전에는 `k=0`이 같은 0.5를 냈고, 출력만 보면 구별할 수 없었다."""
        with pytest.raises(ProbeError):
            rank_agreement(*pair, k=0)

    def test_a_real_ordering_still_scores_one(self, pair):
        assert rank_agreement(*pair, k=3) == 1.0


class TestTheRepositoryIsConsistentAboutBool:
    """`profiles.py`와 `fusion.py`가 이미 하던 판단이다. 세 곳이 같아야 한다."""

    def test_profiles_refuses_a_bool(self):
        from rag_profile_selector.profiles import (ProfileValidationError,
                                                   RetrievalMethod, RetrievalProfile)

        with pytest.raises(ProfileValidationError, match="integer"):
            RetrievalProfile(method=RetrievalMethod.BM25, k=True)

    def test_fusion_refuses_a_bool(self):
        from rag_profile_selector.fusion import resolve_hybrid_rrf_k

        with pytest.raises(ValueError):
            resolve_hybrid_rrf_k(True)

    def test_probes_now_refuses_a_bool(self, pair):
        with pytest.raises(ProbeError):
            overlap_at_k(*pair, k=True)
