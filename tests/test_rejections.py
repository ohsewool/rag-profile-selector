"""거부가 실제로 거부하는가.

커버리지로 훑었더니 미실행 줄이 거의 전부 거부 분기였다 — `regret.py` 82%,
`profiles.py` 92%, `probes.py` 94%. 잘못된 입력을 막는 검사들이 한 번도 발동한 적이
없었다. 전부 쏴봤고 **전부 동작한다.** 결함은 없었다.

다섯 저장소에 같은 방법을 적용했고, `agent-safety-core`에서 하나가 나왔다 — 경로
traversal 검사가 `.resolve()` 뒤에 `..`를 찾고 있어 발동할 수 없었다. 그래서 이
훑기는 빈손으로 끝나도 값이 있다: 다음에 조건이 뒤집히면 여기서 걸린다.

**`probes.py`의 승인 목록 관문은 성격이 다르다.** 그것은 호출자 입력이 아니라
`extract()` 자신의 출력을 검사한다 — 오늘 어떤 입력으로도 발동하지 않고, 그게
정상이다. 지키는 대상이 미래의 편집이기 때문이다: 누군가 답에서 파생된 특징을
추가하면 그때 걸린다. 발동할 수 없는 검사와 오늘 발동할 일이 없는 불변식은 다르고,
아래 테스트는 승인 목록을 좁혀 **관문이 실제로 물 줄 안다**는 것을 보인다.

`grounding`의 거부는 여기 없다. 그 모듈만 형제 저장소를 필요로 하고, 이 파일에
넣었더니 `test_optional_sibling.py`가 **"의존이 퍼졌다"**고 잡았다 - 그 파일이
존재하는 이유가 정확히 그것이다. 허용 목록을 넓히는 대신 이미 의존하는
`test_grounding.py`로 옮겼다. 가드가 잡았을 때 가드를 고치는 것은 가드를 없애는 것과
같다.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rag_profile_selector.probes as probes_module  # noqa: E402
from rag_profile_selector.corpus import CorpusError, CorpusManifest, SplitAssignment  # noqa: E402
from rag_profile_selector.probes import ProbeError, ProbeResult, extract  # noqa: E402
from rag_profile_selector.profiles import (  # noqa: E402
    ProfileValidationError,
    RetrievalMethod,
    RetrievalProfile,
    resolve_profile,
    validate_profile,
)
from rag_profile_selector.regret import (  # noqa: E402
    QueryProfileQualityOutcomes,
    RegretValidationError,
    calculate_exact_regret,
)

PROBES = [
    ProbeResult(retriever="bm25-word", identifiers=("x", "y"), scores=(1.0, 0.5),
                sources=("s1", "s2")),
    ProbeResult(retriever="bm25-char", identifiers=("y", "x"), scores=(0.9, 0.4),
                sources=("s2", "s1")),
]


class TestTheApprovedFeatureGateCanActuallyBite:
    """호출자 입력이 아니라 `extract()` 자신의 출력을 검사하는 불변식이다.

    오늘 어떤 입력으로도 발동하지 않는 것이 정상이고, 그래서 커버리지에 미실행으로
    남는다. 그것과 "발동할 수 없는 검사"는 다르다 — 후자는 이 프로젝트가 자매
    저장소에서 실제로 만났다.
    """

    def test_narrowing_the_approved_list_makes_the_gate_fire(self, monkeypatch):
        monkeypatch.setattr(probes_module, "APPROVED_FEATURES", ("query_token_count",))
        with pytest.raises(ProbeError, match="unapproved features present"):
            extract("전기통신사업자의 손해배상 책임", PROBES, k=2)

    def test_and_the_message_names_what_was_not_approved(self, monkeypatch):
        monkeypatch.setattr(probes_module, "APPROVED_FEATURES", ("query_token_count",))
        with pytest.raises(ProbeError) as caught:
            extract("전기통신사업자의 손해배상 책임", PROBES, k=2)
        assert "overlap_at_k" in str(caught.value)

    def test_the_real_list_lets_everything_through(self):
        """관문이 무엇이든 막으면 실험이 아무것도 못 돈다."""
        assert extract("전기통신사업자의 손해배상 책임", PROBES, k=2)


class TestProfilesRefuseWhatIsNotApproved:
    def test_k_must_be_an_integer(self):
        with pytest.raises(ProfileValidationError, match="k must be an integer"):
            RetrievalProfile(method=list(RetrievalMethod)[0], k="4")

    def test_a_profile_id_must_be_a_string(self):
        with pytest.raises(ProfileValidationError, match="profile_id must be a string"):
            resolve_profile(5)

    def test_an_unapproved_combination_is_refused(self):
        """승인 조합 밖의 프로파일은 실험에 들어올 수 없다. 들어오면 비교표에
        아무도 검토하지 않은 설정이 한 줄 생긴다."""
        with pytest.raises(ProfileValidationError, match="unsupported approved-profile"):
            validate_profile(RetrievalProfile(method=list(RetrievalMethod)[0], k=99))

    def test_an_approved_profile_resolves(self):
        assert resolve_profile("dense-k4").k == 4


class TestRegretRefusesIncoherentOutcomes:
    def test_an_empty_query_id_is_refused(self):
        with pytest.raises(RegretValidationError, match="query_id must be"):
            QueryProfileQualityOutcomes("", ["dense"], {"dense": 1.0})

    def test_no_candidate_profiles_is_refused(self):
        """후보가 없으면 regret은 무엇에 대한 후회인지 말할 수 없다."""
        with pytest.raises(RegretValidationError, match="must not be empty"):
            QueryProfileQualityOutcomes("q1", [], {})

    def test_duplicate_candidates_are_refused(self):
        with pytest.raises(RegretValidationError, match="must not contain duplicates"):
            QueryProfileQualityOutcomes("q1", ["dense", "dense"], {"dense": 1.0})

    def test_qualities_must_be_a_mapping(self):
        with pytest.raises(TypeError, match="must be a mapping"):
            QueryProfileQualityOutcomes("q1", ["dense"], [1.0])

    def test_a_quality_must_be_a_number(self):
        with pytest.raises(TypeError, match="must be real numbers"):
            QueryProfileQualityOutcomes("q1", ["dense"], {"dense": "high"})

    def test_outcomes_must_be_outcome_objects(self):
        with pytest.raises(TypeError, match="must be QueryProfileQualityOutcomes"):
            calculate_exact_regret(["not-an-outcome"], "dense")

    def test_a_well_formed_outcome_is_accepted(self):
        outcome = QueryProfileQualityOutcomes("q1", ["dense", "bm25-char"],
                                              {"dense": 1.0, "bm25-char": 0.5})
        assert outcome.query_id == "q1"
        assert set(outcome.candidate_profiles) == {"dense", "bm25-char"}


class TestTheCorpusRefusesAnIncompleteRecord:
    def test_a_manifest_without_checksums_is_refused(self, tmp_path):
        """체크섬 없는 매니페스트는 무엇을 받았는지 말하지 않는다."""
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"licence": "공공누리", "source": "x",
                                    "fetched_at": "t", "checksums": {}}), encoding="utf-8")
        with pytest.raises(CorpusError, match="incomplete|checksum"):
            CorpusManifest.load(path)

    def test_a_split_manifest_with_no_queries_is_refused(self, tmp_path):
        path = tmp_path / "splits.json"
        path.write_text(json.dumps({"seed": 42, "assignments": {}}), encoding="utf-8")
        with pytest.raises(CorpusError, match="assigns no queries"):
            SplitAssignment.load(path)

    def test_asking_for_a_split_that_does_not_exist_is_refused(self, tmp_path):
        """오타로 빈 목록을 받으면 그 위의 모든 결과가 조용히 0건이 된다."""
        path = tmp_path / "splits.json"
        path.write_text(json.dumps({"seed": 42, "assignments": {"q1": "train"}}),
                        encoding="utf-8")
        with pytest.raises(CorpusError, match="unknown split"):
            SplitAssignment.load(path).members("nope")

    def test_a_known_split_returns_its_members(self, tmp_path):
        path = tmp_path / "splits.json"
        path.write_text(json.dumps({"seed": 42, "assignments": {"q1": "train"}}),
                        encoding="utf-8")
        assert SplitAssignment.load(path).members("train") == ("q1",)
