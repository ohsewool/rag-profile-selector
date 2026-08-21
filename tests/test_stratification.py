"""층화 기준은 라벨이 아니라 측정값이어야 한다 — 그리고 그게 실제로 낫다.

이 코퍼스의 split은 난이도 **라벨**로 층화했다. 라벨 분포는 고르다(test는
paraphrase 5·situational 4·lexical 3). 그런데 검색 난이도를 지배한 성질은 라벨이
아니라 **질의와 정답 조문의 어휘 중복**이었고, 중복이 0인 질의 9건 중 **6건이 test에
몰렸다.** test는 우연히 하드 모드가 됐고, 그래서 test split의 절대 수치가 절반
이하로 떨어졌다.

`check_query_set.py`가 그 중복을 이미 계산하고 있었다. **계산해둔 값을 안 쓴 것이지
없어서 못 쓴 것이 아니다** — 이 프로젝트가 반복해서 만나는 모양이다.

`KR_LAW_RESULTS.md`는 "다음 코퍼스에서는 측정값으로 층화한다"고 적었다. 계획을
문장으로만 남기면 다음 코퍼스가 올 때 다시 라벨로 나눈다. 그래서 도구를 만들었고,
이 테스트가 **그 도구가 실제로 나은지**와 **봉인된 split을 건드리지 않는지**를 고정한다.

현재 코퍼스는 다시 나누지 않는다. 다시 나누면 test가 이미 본 데이터가 된다.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

CORPUS = ROOT / "data" / "kr_law"

stratify_by_overlap = pytest.importorskip("stratify_by_overlap")
from check_query_set import load_articles  # noqa: E402

# 본문이 없으면 **모든** 테스트가 skip이다. 처음에는 `measured` 픽스처에만 걸었는데,
# 봉인 검사 셋은 그 픽스처를 쓰지 않아 그대로 돌았고 **잘못된 이유로 통과했다**:
# 본문이 없으면 `main()`이 "질의가 없다"로 1을 내는데, 테스트는 그것을 "봉인을
# 거부했다"로 읽었다. 같은 종료 코드가 두 가지를 뜻하면 단언은 아무것도 고정하지
# 않는다. 코퍼스를 받아 도는 주간 워크플로가 이 파일을 실제로 실행한다.
pytestmark = pytest.mark.skipif(
    not (CORPUS / "documents").exists(),
    reason="코퍼스 본문은 gitignore돼 있다 — 받아야 돈다")


@pytest.fixture(scope="module")
def measured():
    articles = load_articles(CORPUS)
    queries = json.loads((CORPUS / "queries.json").read_text(encoding="utf-8"))["queries"]
    return {q["id"]: stratify_by_overlap.overlap(q["text"], q["evidence"][0], articles)
            for q in queries}


@pytest.fixture(scope="module")
def existing():
    return json.loads((CORPUS / "splits.json").read_text(encoding="utf-8"))["assignments"]


def spread_of(assignment, measured):
    return stratify_by_overlap._spread(stratify_by_overlap.describe(assignment, measured))


class TestTheMeasuredSplitIsMoreBalanced:
    def test_it_narrows_the_gap_between_splits(self, measured, existing):
        """요지. 라벨 층화는 split 간 평균 중복이 0.1488 벌어졌다."""
        proposed = stratify_by_overlap.stratify(list(measured.items()), seed=42)
        assert spread_of(proposed, measured) < spread_of(existing, measured)

    def test_the_improvement_is_large_rather_than_marginal(self, measured, existing):
        before, after = spread_of(existing, measured), spread_of(
            stratify_by_overlap.stratify(list(measured.items()), seed=42), measured)
        assert after < before / 5, f"{before:.4f} → {after:.4f}"

    def test_the_hard_queries_stop_piling_into_one_split(self, measured, existing):
        """중복 0인 질의 9건 중 6건이 test에 있었다. 그것이 test를 하드 모드로
        만든 직접적 원인이다."""
        def worst(assignment):
            counts = stratify_by_overlap.describe(assignment, measured)
            return max(item["zero"] for item in counts.values())

        proposed = stratify_by_overlap.stratify(list(measured.items()), seed=42)
        assert worst(proposed) < worst(existing)

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 99])
    def test_the_balance_does_not_depend_on_the_seed(self, measured, existing, seed):
        """seed로 좋은 결과를 골랐다면 그건 결과가 아니다.

        이 코퍼스에서는 질의가 짧아 중복 값이 성기고, 28건 전부가 동점군에 속한다.
        그래서 어떤 동점 처리를 하든 균형은 유지된다 — seed는 어느 질의가 어디로
        가는지만 바꾼다.
        """
        proposed = stratify_by_overlap.stratify(list(measured.items()), seed=seed)
        assert spread_of(proposed, measured) < spread_of(existing, measured)

    def test_the_seed_actually_changes_the_assignment(self, measured):
        """균형이 seed에 무관하다는 것이 seed가 아무것도 안 한다는 뜻은 아니다.
        아무것도 안 한다면 인자를 없애야 한다."""
        first = stratify_by_overlap.stratify(list(measured.items()), seed=0)
        second = stratify_by_overlap.stratify(list(measured.items()), seed=1)
        assert first != second


class TestTheSealedSplitIsNotTouched:
    def test_writing_over_an_existing_split_is_refused(self, tmp_path, monkeypatch):
        """봉인된 split을 덮어쓰면 test가 이미 본 데이터가 된다. 실수로 그렇게
        되는 경로가 없어야 한다."""
        assert stratify_by_overlap.main([str(CORPUS), "--write"]) == 1

    def test_the_existing_file_is_unchanged_after_that_attempt(self):
        before = (CORPUS / "splits.json").read_bytes()
        stratify_by_overlap.main([str(CORPUS), "--write"])
        assert (CORPUS / "splits.json").read_bytes() == before

    def test_comparing_without_write_succeeds(self):
        assert stratify_by_overlap.main([str(CORPUS)]) == 0


class TestTheComparisonIsNotVacuous:
    def test_the_two_stratifications_really_differ(self, measured, existing):
        proposed = stratify_by_overlap.stratify(list(measured.items()), seed=42)
        moved = [q for q in existing if existing[q] != proposed.get(q)]
        assert moved, "제안이 현재 split과 같다면 비교할 것이 없다"

    def test_both_assign_every_query(self, measured, existing):
        proposed = stratify_by_overlap.stratify(list(measured.items()), seed=42)
        assert set(proposed) == set(measured)
        assert set(existing) == set(measured)

    def test_the_split_sizes_are_preserved(self, measured, existing):
        """크기까지 바꾸면 무엇이 개선을 만들었는지 알 수 없다."""
        from collections import Counter
        proposed = stratify_by_overlap.stratify(list(measured.items()), seed=42)
        assert Counter(proposed.values()) == Counter(existing.values())

    def test_overlap_is_measured_not_assumed(self, measured):
        """전부 0이면 어떤 층화도 완벽해 보인다."""
        assert any(value > 0 for value in measured.values())
        assert len(set(measured.values())) > 3
