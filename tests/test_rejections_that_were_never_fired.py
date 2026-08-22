"""거부 여섯 개가 한 번도 발동한 적이 없었다.

2026-08-22에 다섯 저장소에 같은 감사를 걸었다. 문자열 메시지를 가진 `raise`
**31개를 하나씩 `pass`로 바꾸고** 매번 스위트를 돌렸다.

    잡힘        25건
    안 잡힘      6건

여섯 중 둘이 이 저장소의 주장 한복판에 있다.

`validate_profile`은 **"승인된 프로파일이 그대로인가"**를 묻는 함수다. 그 함수의 두
거부 — 프로파일이 아닌 것, 그리고 **필드가 승인된 값과 다른 것** — 이 둘 다 한 번도
발동한 적이 없었다. 즉 이 저장소는 "승인 목록 밖의 설정으로는 실험하지 않는다"고
말하면서, **그 문지기가 실제로 막는지는 확인한 적이 없었다.**

나머지 넷은 코퍼스 매니페스트의 빈 체크섬, 융합 입력이 리스트가 아닌 경우, 후보
프로파일에 빈 문자열이 섞인 경우, 그리고 `document_intelligence`가 없을 때의 안내다.
마지막 것은 형제 저장소가 없는 환경에서 나는 것이라 **하위 프로세스에서 확인한다** —
이 프로세스에는 이미 설치돼 있고, 설치된 채로는 그 분기가 영영 죽어 있다.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_profile_selector.corpus import CorpusError, CorpusManifest  # noqa: E402
from rag_profile_selector.fusion import fuse_reciprocal_rank  # noqa: E402
from rag_profile_selector.profiles import (  # noqa: E402
    ProfileValidationError,
    RetrievalMethod,
    RetrievalProfile,
    resolve_profile,
    validate_profile,
)
from rag_profile_selector.regret import RegretValidationError  # noqa: E402
from rag_profile_selector.regret import QueryProfileQualityOutcomes  # noqa: E402

COMPLETE = {
    "corpus_id": "c", "version": "1", "source_url": "https://example.test",
    "licence": "CC0", "retrieved_at": "2026-01-01T00:00:00+00:00",
    "evidence_unit": "article", "checksums": {"a.json": "0" * 64},
}


class TestTheApprovedProfileGate:
    """이 저장소는 "승인 목록 밖의 설정으로는 실험하지 않는다"고 말한다.
    그 문장을 지키는 것이 `validate_profile`인데, **그 함수의 거부 둘이 한 번도
    발동한 적이 없었다.** 문지기가 서 있는지 확인하지 않은 채로 문지기가 있다고
    말해온 셈이다."""

    def test_something_that_is_not_a_profile_is_refused(self):
        with pytest.raises(ProfileValidationError, match="must be a RetrievalProfile"):
            validate_profile("dense-4")

    def test_tampering_cannot_be_detected_and_the_docstring_now_says_so(self):
        """두 번째 거부("필드가 승인된 id와 다르다")는 **절대 참이 될 수 없다.**

        `profile_id`가 `method`와 `k`에서 파생되기 때문이다. dense를 bm25로 바꾸면
        id도 함께 `bm25-k4`가 되고 그 정본과 정확히 일치한다 — 즉 이 함수는
        "네가 준 객체가 카탈로그가 발행한 그것이다"를 확인하지 못한다. 원래
        독스트링은 "only when *profile* is unaltered"라고 적혀 있었고, 그것이
        코드가 할 수 있는 것보다 큰 약속이었다.

        **테스트를 쓰려다 도달 불가라는 것을 알았다.** 감사가 "안 잡힘"이라고 한
        여섯 중 하나가 이것이고, 잡히지 않은 이유가 "테스트가 없어서"가 아니라
        "일어날 수 없어서"였다. 그 구분은 재보기 전에는 알 수 없었다.
        """
        canonical = resolve_profile("dense-k4")
        tampered = RetrievalProfile(method=canonical.method, k=canonical.k)
        object.__setattr__(tampered, "method", RetrievalMethod.BM25)
        assert validate_profile(tampered) is resolve_profile("bm25-k4")

    def test_what_does_hold_is_that_the_combination_is_approved(self):
        """실제 보장은 이것이다 — 승인되지 않은 조합은 통과하지 못한다."""
        canonical = resolve_profile("dense-k4")
        unapproved = RetrievalProfile(method=canonical.method, k=canonical.k)
        object.__setattr__(unapproved, "k", 8)      # dense-k8은 카탈로그에 없다
        with pytest.raises(ProfileValidationError, match="unknown approved profile_id"):
            validate_profile(unapproved)

    def test_an_untouched_profile_passes(self):
        """거부만 확인하면 전부 거부하는 구현도 통과한다."""
        profile = resolve_profile("dense-k4")
        assert validate_profile(profile) is resolve_profile("dense-k4")


class TestTheManifestMustCarryChecksums:
    """체크섬 없는 매니페스트를 받아들이면 **코퍼스가 매니페스트가 말하는 그것인지**를
    확인할 방법이 사라진다. 주간 워크플로가 그 대조로 서 있다."""

    # `{}`와 `[]`는 falsy라 위쪽 "incomplete" 검사가 먼저 잡는다. 여기서 확인하려는
    # 것은 **비어 있지 않은데 dict가 아닌** 경우 - 두 검사가 다른 것을 막는다.
    @pytest.mark.parametrize("checksums", ["none", ["a.json"], 5])
    def test_a_manifest_without_them_is_refused(self, tmp_path, checksums):
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({**COMPLETE, "checksums": checksums}), encoding="utf-8")
        with pytest.raises(CorpusError, match="at least one file checksum"):
            CorpusManifest.load(path)

    def test_a_complete_manifest_loads(self, tmp_path):
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(COMPLETE), encoding="utf-8")
        assert CorpusManifest.load(path).corpus_id == "c"


class TestFusionInputsMustBeListsOfLists:
    @pytest.mark.parametrize("value", ["abc", b"abc", 5, None])
    def test_something_that_is_not_a_sequence_of_lists_is_refused(self, value):
        """문자열은 시퀀스라서 조용히 통과한다 — 그러면 **글자 하나하나가 식별자**가
        되고, 융합 결과는 나오지만 아무 의미가 없다."""
        with pytest.raises(TypeError, match="sequence of identifier lists"):
            fuse_reciprocal_rank(value, k=4)

    def test_real_ranked_lists_fuse(self):
        assert fuse_reciprocal_rank([["a", "b"], ["b", "a"]], k=4)


class TestCandidateProfilesMustBeNamed:
    @pytest.mark.parametrize("candidates", [("dense-k4", ""), ("dense-k4", None), ("", )])
    def test_a_blank_candidate_is_refused(self, candidates):
        with pytest.raises(RegretValidationError, match="non-empty string identifiers"):
            QueryProfileQualityOutcomes(
                query_id="q1", candidate_profiles=candidates,
                qualities={name: 1.0 for name in candidates if name})

    def test_named_candidates_are_accepted(self):
        outcomes = QueryProfileQualityOutcomes(
            query_id="q1", candidate_profiles=("dense-k4", "bm25-k4"),
            qualities={"dense-k4": 1.0, "bm25-k4": 0.5})
        assert outcomes.query_id == "q1"


class TestTheSiblingIsRequiredForGrounding:
    """`grounding`은 `document_intelligence` 없이는 아무것도 못 한다. 그 사실을
    말하는 `ImportError`는 **형제가 설치된 환경에서는 영영 죽어 있다** — 그래서
    하위 프로세스에서 import를 막고 확인한다.

    이 저장소는 형제 가용성 판단에서 이미 한 번 데였다: 절대 경로로 판단하던 검사가
    다른 기계에서 조용히 skip됐다. **없을 때 무슨 일이 일어나는지**를 확인하는 것이
    그 교훈의 나머지 절반이다.
    """

    def test_without_the_sibling_the_import_says_so(self):
        script = textwrap.dedent(f"""
            import sys

            class Block:
                def find_spec(self, name, path=None, target=None):
                    if name == "document_intelligence" or name.startswith("document_intelligence."):
                        return None
                    return None

            sys.meta_path.insert(0, Block())
            for name in [n for n in sys.modules if n.startswith("document_intelligence")]:
                del sys.modules[name]
            sys.path = [p for p in sys.path if "document" not in p]
            sys.path.insert(0, {str(ROOT / "src")!r})
            try:
                import rag_profile_selector.grounding  # noqa: F401
            except ImportError as error:
                print("REFUSED:", error)
            else:
                print("IMPORTED")
        """)
        finished = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                  text=True, timeout=120, cwd=ROOT)
        output = finished.stdout.strip()
        if output.startswith("IMPORTED"):
            pytest.skip("이 환경에서는 형제를 경로에서 떼어낼 수 없다 — 설치돼 있다")
        assert "grounding requires the document_intelligence package" in output


class TestTheAuditIsRecorded:
    def test_the_readme_says_what_it_found(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "31개를 하나씩" in readme
        assert "안 잡힘 6" in readme


class TestLinesNoTestEverRan:
    """거부 감사 뒤에 질문을 넓혔다 — **한 번도 실행되지 않는 줄이 무엇인가.**
    574줄 중 9줄이었다. 대부분이 **퇴화 입력의 이른 반환**이다: 점수가 0이거나,
    목록이 비었거나, 비교할 쌍이 하나도 없는 경우. 그런 갈래가 안 돌았다는 것은
    **빈 검색 결과로 특징을 계산해본 적이 없다**는 뜻이고, 실험은 늘 결과가 있는
    질의로만 돌았다는 뜻이다."""

    def test_a_summary_says_ok_when_nothing_is_wrong(self):
        from rag_profile_selector.corpus import ValidationReport

        assert ValidationReport([]).summary().startswith("OK —")

    def test_identifiers_that_are_not_iterable_are_refused(self):
        from rag_profile_selector.evidence_metrics import calculate_evidence_metrics

        with pytest.raises(TypeError, match="must be an iterable of identifiers"):
            calculate_evidence_metrics(5, ["a"])

    def test_a_decay_of_zero_when_the_top_score_is_zero(self):
        """상위 점수가 0이면 감쇠를 비율로 말할 수 없다. 0으로 나누는 대신 0을
        돌려주는데, **그 갈래는 검색이 아무것도 못 찾았을 때만 나온다.**"""
        from rag_profile_selector.probes import ProbeResult

        assert ProbeResult("bm25", ("a", "b"), (0.0, 0.0)).score_decay() == 0.0
        assert ProbeResult("bm25", ("a",), (1.0,)).score_decay() == 0.0

    def test_a_duplicate_ratio_of_zero_for_an_empty_result(self):
        from rag_profile_selector.probes import ProbeResult

        assert ProbeResult("bm25", (), ()).duplicate_ratio() == 0.0

    def test_overlap_is_zero_when_one_side_found_nothing(self):
        """한쪽이 빈 결과면 겹침은 정의되지 않는다. 0을 주는 것과 1을 주는 것은
        **"둘이 완전히 다르다"와 "둘이 같다"만큼 다르고**, 선택기는 그 값을 본다."""
        from rag_profile_selector.probes import ProbeResult, overlap_at_k

        empty = ProbeResult("bm25", (), ())
        full = ProbeResult("dense", ("a", "b"), (1.0, 0.5))
        assert overlap_at_k(empty, full, k=2) == 0.0
        assert overlap_at_k(full, empty, k=2) == 0.0

    def test_rank_agreement_is_a_half_when_there_is_nothing_to_compare(self):
        """비교할 쌍이 하나도 없으면 일치도 불일치도 아니다. 0.5는 "모른다"이고,
        0이나 1로 접으면 **모르는 것이 아는 것처럼 특징에 실린다.**"""
        from rag_profile_selector.probes import ProbeResult, rank_agreement

        one = ProbeResult("bm25", ("a",), (1.0,))
        other = ProbeResult("dense", ("b",), (1.0,))
        assert rank_agreement(one, other) == 0.5

    def test_the_base_selector_refuses_to_pretend(self):
        """`Selector.choose`는 규칙이 채워야 하는 자리다. 기본 구현이 조용히
        무언가를 고르면 **아무 규칙도 없는 선택기가 결과를 낸다.**"""
        from rag_profile_selector.selector import Selector

        with pytest.raises(NotImplementedError):
            Selector().choose("q1", {})

    def test_rank_agreement_is_a_half_when_every_pair_ties(self):
        """공유 항목이 둘인데 **같은 식별자가 두 번**이면 비교할 순서쌍이 없다.
        중복은 이 저장소가 실제로 측정하는 것이다(`duplicate_ratio`) — 즉 중복이
        섞인 결과로 특징을 뽑는 것은 상상한 상황이 아니다."""
        from rag_profile_selector.probes import ProbeResult, rank_agreement

        duplicated = ProbeResult("bm25", ("a", "a"), (1.0, 0.9))
        single = ProbeResult("dense", ("a",), (1.0,))
        assert rank_agreement(duplicated, single) == 0.5

    def test_a_coarse_gold_entry_counts_as_region_correct(self):
        """조문 단위 정답에는 구역이 없다(`None`). 페이지가 맞으면 맞은 것이고,
        구역까지 요구하면 **정답이 구역을 말하지 않는 질의를 전부 오답으로 센다** —
        이 코퍼스의 정답이 정확히 그 모양이다."""
        pytest.importorskip("document_intelligence")
        from rag_profile_selector.grounding import (
            GroundedCitation, GroundingResult, measure_citation_accuracy)

        citation = GroundedCitation(identifier="c1", document_id="d1", page_number=3,
                                    region_identifier=None, bounding_box=None,
                                    rank=1, score=1.0)
        accuracy = measure_citation_accuracy(GroundingResult((citation,), ()),
                                             {"c1": (3, None)})
        assert accuracy.page_correct == 1
        assert accuracy.region_correct == 1
