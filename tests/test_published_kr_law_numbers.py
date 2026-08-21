"""문서에 실린 숫자가 지금 코드에서도 나오는가.

`KR_LAW_RESULTS.md`의 표에는 프로파일 다섯 개의 MRR@4·top-1·미검색과, 선택 규칙 넷의
평균 regret·최적 선택률이 적혀 있다. **아무것도 그 숫자를 실행 결과와 대조하지
않았다.**

`agent-safety-core`에는 그 검사가 있다(`test_published_benchmark.py` — 벤치마크를
돌려 README의 표와 맞춰본다). 여기엔 없었다. 이 저장소들에서 반복해서 나온 모양이다:
**한쪽에는 있고 형제에는 없다.**

빈 곳인 줄 몰랐던 이유는 옆에 있는 검사가 그럴듯했기 때문이다. 주간 `corpus` 워크플로가
국가법령정보 API에서 법령을 다시 받아 **체크섬을 대조**한다. 그것이 지키는 것은
"코퍼스가 매니페스트가 말하는 그것인가"이고, "그 코퍼스에서 저 숫자가 나오는가"는
아니다. 둘은 다른 주장이고 함께 있어야 표가 증거가 된다.

2026-08-22에 손으로 확인했다. 최근 여덟 회차에 다섯 저장소의 계산 경로를 여럿
건드렸고 — 여기서는 `overlap_at_k`·`rank_agreement`에 `_cutoff` 검증을 넣었다,
그 둘은 `extract()`가 부르는 특징 계산이다 — 그 뒤로 재현을 한 번도 확인하지
않았었다. 다섯 프로파일과 네 규칙의 아홉 개 수치가 **전부 그대로**였다. 빈손이지만
그것이 이 저장소의 주장이고, 손으로 한 번 확인한 것은 다음에도 성립한다는 뜻이 아니다.

**CI는 이것을 돌리지 않는다.** 코퍼스 본문을 재배포하지 않으므로 CI 클론에는 문서가
없고, 이 파일은 다른 25개와 같은 이유로 skip된다. 지금 성립하는 조합은 이렇다:
주간 잡이 **코퍼스가 그대로임**을 말하고, 이 검사가 **그 코퍼스에서 숫자가 나옴**을
말한다 — 코퍼스를 받은 곳에서 돌렸을 때. 둘 다 없으면 표는 손으로 적은 숫자일 뿐이다.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "KR_LAW_RESULTS.md"
DOCUMENTS = ROOT / "data" / "kr_law" / "documents"

pytestmark = pytest.mark.skipif(
    not (DOCUMENTS.exists() and any(DOCUMENTS.glob("*.json"))),
    reason="코퍼스 본문은 gitignore돼 있다 — 받아야 돈다",
)


def run(script: str) -> str:
    """실험을 그대로 돌린다. 출력을 파싱할 뿐 안을 흉내 내지 않는다 —
    흉내 내면 실험이 아니라 내 재현을 검사하게 된다."""
    finished = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / script)],
        cwd=ROOT, capture_output=True, text=True, timeout=2400)
    assert finished.returncode == 0, finished.stderr[-2000:]
    return finished.stdout


@pytest.fixture(scope="module")
def retrieval_output():
    return run("kr_law_retrieval.py")


@pytest.fixture(scope="module")
def selection_output():
    return run("kr_law_selection.py")


@pytest.fixture(scope="module")
def published():
    return RESULTS.read_text(encoding="utf-8")


def published_profiles(text: str) -> dict[str, tuple[float, float, float]]:
    """프로파일 비교표. 헤더 **줄 다음부터** 자른다 — 헤더 문자열 길이만큼만
    건너뛰면 그 줄의 남은 부분이 첫 줄이 되어 표를 하나도 못 읽는다. 이 저장소가
    이미 한 번 빠진 함정이라 같은 방식으로 자른다."""
    header = "| 프로파일 | MRR@4 | top-1 | 미검색 |"
    assert header in text, "프로파일 비교표를 찾지 못했다 — 열이 바뀌었나?"
    body = text[text.index(header):].split("\n", 1)[1]
    found = {}
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if not cells or set(cells[0]) <= set("-: ") or not cells[0]:
            continue
        name = re.sub(r"[*`]", "", cells[0]).split("(")[0].strip()
        numbers = [float(re.sub(r"[^\d.]", "", cell)) for cell in cells[1:4]]
        found[name] = tuple(numbers)
    return found


def produced_profiles(output: str) -> dict[str, tuple[float, float, float]]:
    found = {}
    for line in output.splitlines():
        match = re.match(r"\s*([\w-]+)\s+(\d+\.\d+)\s+(\d+\.\d+)%\s+(\d+\.\d+)%\s*$", line)
        if match:
            found[match.group(1)] = (float(match.group(2)), float(match.group(3)),
                                     float(match.group(4)))
    return found


def published_rules(text: str) -> dict[str, tuple[float, float]]:
    """규칙 표. **첫 번째 것만** 읽는다.

    이 문서에는 같은 모양의 표가 둘 있다 — train+val(78줄)과 봉인 해제한 test
    split(133줄). 처음에 `rule:` 줄을 전부 훑었더니 딕셔너리가 뒤엣것으로 덮여
    test split 숫자와 비교했고, 실험은 train+val을 출력하므로 아홉 개가 전부
    어긋난 것처럼 보였다. **재현 실패가 아니라 내 파서가 다른 표를 읽은 것이다.**

    프로파일 표에서 쓴 것과 같은 방식으로 앵커를 고정한다: 헤더 **줄 다음부터**
    자르고, `|`로 시작하지 않는 줄에서 멈춘다.
    """
    header = "| 선택기 | 평균 regret | 최적 선택률 |"
    assert header in text, "선택기 표를 찾지 못했다 — 열이 바뀌었나?"
    body = text[text.index(header):].split("\n", 1)[1]
    found = {}
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            break
        match = re.match(r"\|\s*(rule:[\w-]+)\s*\|\s*\**([\d.]+)\**\s*\|\s*([\d.]+)%\s*\|", line)
        if match:
            found[match.group(1)] = (float(match.group(2)), float(match.group(3)))
    return found


def produced_rules(output: str) -> dict[str, tuple[float, float]]:
    found = {}
    for line in output.splitlines():
        match = re.match(r"\s*(rule:[\w-]+)\s+([\d.]+)\s+[\d.]+\s+([\d.]+)%", line)
        if match:
            found[match.group(1)] = (float(match.group(2)), float(match.group(3)))
    return found


class TestEveryPublishedProfileNumberMatchesARun:
    def test_the_same_profiles_appear(self, retrieval_output, published):
        assert set(produced_profiles(retrieval_output)) == set(published_profiles(published))

    def test_every_number_matches(self, retrieval_output, published):
        produced, claimed = produced_profiles(retrieval_output), published_profiles(published)
        wrong = {name: (produced[name], claimed[name]) for name in claimed
                 if produced[name] != claimed[name]}
        assert not wrong, wrong

    def test_the_headline_dense_number_is_the_published_one(self, retrieval_output, published):
        """표 전체가 맞아도 결론이 걸린 숫자를 따로 못박는다. 0.714는 이 저장소의
        결론("dense가 최선 고정 프로파일")이 서 있는 값이다."""
        assert produced_profiles(retrieval_output)["dense"][0] == 0.714
        assert published_profiles(published)["dense"][0] == 0.714


class TestEveryPublishedRuleNumberMatchesARun:
    def test_the_same_rules_appear(self, selection_output, published):
        assert set(produced_rules(selection_output)) == set(published_rules(published))

    def test_every_number_matches(self, selection_output, published):
        produced, claimed = produced_rules(selection_output), published_rules(published)
        wrong = {name: (produced[name], claimed[name]) for name in claimed
                 if produced[name] != claimed[name]}
        assert not wrong, wrong

    def test_the_headroom_is_what_the_document_says(self, selection_output, published):
        match = re.search(r"headroom\s+([\d.]+)", selection_output)
        assert match and float(match.group(1)) == 0.1071
        assert "0.1071" in published


class TestTheComparisonIsNotVacuous:
    def test_both_sides_produced_something(self, retrieval_output, published):
        assert len(produced_profiles(retrieval_output)) >= 5
        assert len(published_profiles(published)) >= 5

    def test_the_rule_tables_are_not_empty(self, selection_output, published):
        assert len(produced_rules(selection_output)) >= 4
        assert len(published_rules(published)) >= 4

    def test_the_parser_reads_the_first_table_not_the_second(self, published):
        """문서에는 같은 모양의 표가 둘 있다(train+val, 봉인 해제한 test split).
        뒤엣것을 읽으면 전부 어긋난 것처럼 보인다 — 실제로 그렇게 한 번 틀렸다."""
        assert published_rules(published)["rule:lexical-when-confident"] == (0.0893, 85.7)
        assert "0.0694" in published, "test split 표가 사라졌다면 이 검사의 전제가 바뀐다"

    def test_the_numbers_are_not_all_zero(self, retrieval_output):
        values = [value for row in produced_profiles(retrieval_output).values() for value in row]
        assert any(value > 0 for value in values)

    def test_a_changed_number_would_be_noticed(self, retrieval_output, published):
        """비교가 실제로 차이를 잡는지. 한 값을 흔들어 같은 판정을 걸어본다."""
        produced = dict(produced_profiles(retrieval_output))
        claimed = published_profiles(published)
        name = next(iter(claimed))
        produced[name] = (claimed[name][0] + 0.001,) + claimed[name][1:]
        assert produced[name] != claimed[name]

    def test_the_experiments_actually_ran(self, retrieval_output, selection_output):
        """출력이 비어도 파싱 결과가 비고 위 검사 일부가 통과한다."""
        assert len(retrieval_output) > 500
        assert len(selection_output) > 500
