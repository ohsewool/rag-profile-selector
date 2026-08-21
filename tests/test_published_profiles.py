"""실험이 돌리는 프로파일과 결과 표에 실린 프로파일이 같은가.

`KR_LAW_RESULTS.md`의 표는 프로파일별 수치를 싣는다. 그 표와 실험의 프로파일 목록은
**같은 것을 두 곳에 적은 것**이고, 어긋나는지 보는 것이 없었다. 표에 없는 프로파일을
돌리면 결과가 조용히 사라지고, 돌리지 않은 프로파일을 표에 실으면 어디서도 나오지
않은 숫자가 공개된다.

이 저장소들이 반복해서 찾아온 모양이라 여기도 확인해 두는 것이 맞다. 그리고 실제로
**옆에서 어긋난 것이 나왔다**: `src/rag_profile_selector/profiles.py`의
`APPROVED_PROFILES`는 착수 계획의 네 pilot(`bm25-k4`·`dense-k4`·`hybrid-rrf-k4`·
`hybrid-rrf-k8`)을 담고 있는데 실험은 다른 다섯을 돌린다. 승인되지 않은 설정을
거부하려고 만든 그 모듈을 **설정을 고르는 코드가 부르지 않는다** — `experiments/`는
`resolve_profile`도 `validate_profile`도 import하지 않는다.

그쪽은 지금 배선하지 않는다. `RetrievalProfile`이 `(method, k)`라 토큰화가 다른
`bm25-word`/`bm25-char`도, 셋을 융합한 `hybrid-all`도 표현하지 못한다 — 모델 변경이
필요하고, 급히 뜯어고치는 것은 상태를 정확히 적어두는 것보다 나쁘다. 경위는
`docs/DECISIONS.md`의 D-003 정정에 있다.

여기서 고정하는 것은 **성립하는 쪽**이다: 돌린 것과 실은 것이 같다.

코퍼스가 없어도 돈다. 실험의 상수와 마크다운만 읽는다.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

RESULTS = ROOT / "experiments" / "KR_LAW_RESULTS.md"
SELECTION = ROOT / "experiments" / "kr_law_selection.py"

pytestmark = pytest.mark.skipif(
    not (RESULTS.exists() and SELECTION.exists()), reason="실험 파일이 없다")


def experiment_profiles() -> list[str]:
    """실험의 `PROFILES` 상수. import하지 않고 소스에서 읽는다 —
    import하면 무거운 의존이 딸려오고, 확인하려는 것은 **적혀 있는 목록**이다."""
    text = SELECTION.read_text(encoding="utf-8")
    found = re.search(r"^PROFILES\s*=\s*\[([^\]]+)\]", text, re.MULTILINE)
    assert found, "kr_law_selection.py에서 PROFILES를 찾지 못했다"
    return re.findall(r'["\']([^"\']+)["\']', found.group(1))


def published_profiles() -> list[str]:
    """결과 문서의 프로파일 비교표 첫 열.

    표 제목이 아니라 헤더 행을 기준으로 찾는다 - 제목은 문장이라 다듬어지고,
    헤더는 열 이름이라 바뀌면 표 자체가 바뀐 것이다.
    """
    text = RESULTS.read_text(encoding="utf-8")
    header = "| 프로파일 | MRR@4 | top-1 | 미검색 |"
    assert header in text, "프로파일 비교표를 찾지 못했다 — 열이 바뀌었나?"
    # 헤더 **줄 다음부터** 자른다. 헤더 문자열 길이만큼만 건너뛰면 그 줄의 남은
    # 부분(빈 문자열)이 첫 줄이 되고, `|`로 시작하지 않으니 즉시 멈춘다 - 표를
    # 하나도 못 읽으면서 "표에만 있는 프로파일 없음"은 통과한다. 아래
    # `TestTheComparisonIsNotVacuous`가 그것을 잡아냈다.
    body = text[text.index(header):].split("\n", 1)[1]
    found = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            break
        cell = line.split("|")[1].strip()
        if set(cell) <= set("-: ") or not cell:
            continue
        # `**dense** (e5-small)` → `dense`
        name = re.sub(r"[*`]", "", cell).split("(")[0].strip()
        if name:
            found.append(name)
    return found


class TestWhatRanIsWhatIsPublished:
    def test_the_same_profiles_appear_in_both(self):
        assert set(experiment_profiles()) == set(published_profiles())

    def test_the_table_has_no_profile_the_experiment_never_ran(self):
        extra = set(published_profiles()) - set(experiment_profiles())
        assert not extra, f"표에만 있는 프로파일: {extra} — 어디서도 나오지 않은 숫자다"

    def test_the_experiment_runs_nothing_the_table_omits(self):
        missing = set(experiment_profiles()) - set(published_profiles())
        assert not missing, f"돌렸는데 표에 없는 프로파일: {missing} — 결과가 사라진다"


class TestTheApprovedListIsNotWiredAndSaysSo:
    """상태를 고정한다. 배선하는 날 이 테스트가 실패하고, 그때 D-003 정정과 모듈
    docstring을 함께 고치게 된다 — 한 곳만 고쳐지는 것이 이 저장소들의 단골 결함이다."""

    def test_the_approved_list_still_holds_the_planned_four(self):
        from rag_profile_selector.profiles import APPROVED_PROFILES

        assert {profile.profile_id for profile in APPROVED_PROFILES} == {
            "bm25-k4", "dense-k4", "hybrid-rrf-k4", "hybrid-rrf-k8"}

    def test_the_experiments_do_not_consult_it(self):
        used = [path.name for path in sorted((ROOT / "experiments").glob("*.py"))
                if re.search(r"resolve_profile|validate_profile|APPROVED_PROFILES",
                             path.read_text(encoding="utf-8"))]
        assert not used, (
            f"{used}가 이제 승인 목록을 부른다. 좋은 변화지만 D-003 정정과 "
            f"profiles.py의 설명이 함께 바뀌어야 한다."
        )

    def test_the_module_says_it_is_not_a_live_gate(self):
        """설명 없이 두면 읽는 사람은 관문이 동작한다고 믿는다."""
        source = (ROOT / "src" / "rag_profile_selector" / "profiles.py").read_text(
            encoding="utf-8")
        assert "실험이 쓰지 않는다" in source


class TestTheComparisonIsNotVacuous:
    def test_both_sides_produced_something(self):
        assert len(experiment_profiles()) >= 4
        assert len(published_profiles()) >= 4

    def test_the_table_parser_strips_decoration(self):
        """`**dense** (e5-small)`가 `dense`로 읽히지 않으면 두 집합이 영원히
        달라지고, 이 파일은 통과하지 못하거나 잘못된 이유로 실패한다."""
        assert "dense" in published_profiles()
        assert "hybrid-rrf" in published_profiles()

    def test_a_missing_profile_would_be_noticed(self):
        experiment = set(experiment_profiles())
        assert experiment - {next(iter(experiment))} != experiment
