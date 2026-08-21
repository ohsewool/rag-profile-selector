"""규칙이 읽는 특징은 실제로 생성돼야 하고, 규칙은 갈려야 한다.

`rule:dense-for-long-queries`가 `query_length`를 읽고 있었다. 그런 특징은 만들어지지
않는다 — 있는 것은 `query_token_count`와 `query_char_count`다. `f.get(이름, 0.0)`이
조용히 0.0을 돌려주니 `>= 8`은 언제나 거짓이었고, 이 규칙은 28건 전부에서
`bm25-char`를 골랐다. **이름과 정반대로 한 번도 dense를 고르지 않았다.**

규칙이 아니라 상수였고, 그런 채로 "규칙 넷을 시험했으나 하나만 기준선을 이겼다"는
문장 안에 한 줄을 차지했다. 사실상 `fixed:bm25-char`를 다른 이름으로 다시 센 것이다.
**오타 하나가 실험 하나를 조용히 무효로 만든다.**

두 가지를 고정한다:

  - 선언한 특징이 실제로 생성되는가. `dict.get`의 기본값은 편리하지만, 없는 키와
    0인 값을 구분하지 못한다.
  - 규칙이 두 가지 이상을 고르는가. 하나만 고르는 것은 선택기가 아니라 고정
    프로파일이고, 표에서 한 줄을 차지하면서 아무것도 시험하지 않는다.

여기서 쓰는 특징 값은 실제 코퍼스가 아니라 손으로 만든다. 코퍼스는 gitignore돼
있어 매 push마다 없고, **이 성질은 코퍼스와 무관하다** — 규칙이 읽는 이름이
`extract()`가 내놓는 이름 안에 있는지의 문제다.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from rag_profile_selector.probes import ProbeResult, extract  # noqa: E402

kr_law_selection = pytest.importorskip("kr_law_selection")
RULES = kr_law_selection.RULES


def produced_features() -> dict[str, float]:
    """`extract()`가 실제로 내놓는 특징 한 벌.

    두 탐침에 서로 다른 순위를 줘서 교차 특징이 자명한 값으로 접히지 않게 한다.
    """
    probes = [
        ProbeResult(retriever="bm25-word", identifiers=("a", "b", "c", "d"),
                    scores=(4.0, 2.0, 1.5, 1.0), sources=("s1", "s2", "s3", "s3")),
        ProbeResult(retriever="bm25-char", identifiers=("b", "a", "e", "c"),
                    scores=(3.0, 2.5, 1.0, 0.5), sources=("s2", "s1", "s4", "s3")),
    ]
    return extract("전기통신사업자의 손해배상 책임은 어떻게 정해지는가", probes, k=4)


def varied_feature_sets() -> list[dict[str, float]]:
    """규칙이 갈리는지 보려면 서로 다른 입력이 필요하다."""
    base = produced_features()
    found = []
    for margin, overlap, tokens, decay in (
        (0.05, 0.10, 3, 0.20), (0.40, 0.80, 12, 0.90),
        (0.25, 0.50, 8, 0.50), (0.60, 0.20, 20, 0.10),
    ):
        found.append({**base, "top1_margin": margin, "overlap_at_k": overlap,
                      "query_token_count": tokens, "score_decay": decay})
    return found


class TestEveryRuleReadsAFeatureThatExists:
    def test_each_declared_feature_is_produced(self):
        available = set(produced_features())
        missing = {name: [f for f in declared if f not in available]
                   for name, (_, _, declared) in RULES.items()
                   if any(f not in available for f in declared)}
        assert not missing, (
            f"규칙이 존재하지 않는 특징을 읽는다: {missing}. "
            f"생성되는 특징: {sorted(available)}"
        )

    def test_the_experiment_refuses_to_run_with_a_missing_feature(self):
        """선언만 하고 검사하지 않으면 선언은 주석이다."""
        features = {"q1": {k: v for k, v in produced_features().items()
                           if k != "query_token_count"}}
        with pytest.raises(SystemExit, match="query_token_count"):
            kr_law_selection.validate_rules(features)

    def test_it_accepts_a_complete_feature_set(self):
        kr_law_selection.validate_rules({"q1": produced_features()})


class TestNoRuleIsSecretlyAConstant:
    @pytest.mark.parametrize("name", sorted(RULES))
    def test_the_rule_picks_more_than_one_profile(self, name):
        rule, _, _ = RULES[name]
        picked = {rule(features) for features in varied_feature_sets()}
        assert len(picked) >= 2, (
            f"{name}이(가) 언제나 {picked}만 고른다. 고정 프로파일을 다른 이름으로 "
            f"세는 것이고, 표에서 한 줄을 차지하면서 아무것도 시험하지 않는다."
        )

    def test_the_detector_would_notice_a_constant(self):
        """검사가 실패할 줄 아는지. 늘 같은 것을 고르는 규칙을 넣어 확인한다."""
        assert kr_law_selection.rule_is_constant(
            lambda f: "dense", {f"q{i}": f for i, f in enumerate(varied_feature_sets())})

    def test_and_does_not_cry_wolf_on_a_real_one(self):
        rule, _, _ = RULES["rule:lexical-when-confident"]
        assert not kr_law_selection.rule_is_constant(
            rule, {f"q{i}": f for i, f in enumerate(varied_feature_sets())})


class TestTheFixtureIsNotVacuous:
    """특징이 만들어지지 않으면 위 검사들은 아무것도 비교하지 않는다."""

    def test_extract_produced_the_features_the_rules_need(self):
        available = produced_features()
        for name in ("top1_margin", "overlap_at_k", "query_token_count", "score_decay"):
            assert name in available, name

    def test_query_length_is_not_among_them(self):
        """결함의 정체를 그대로 고정한다. 언젠가 이 이름이 생기면 이 테스트가
        먼저 알려주고, 그때는 규칙을 되돌릴지 결정하면 된다."""
        assert "query_length" not in produced_features()

    def test_the_varied_sets_really_vary(self):
        sets = varied_feature_sets()
        assert len({f["top1_margin"] for f in sets}) > 1
        assert len({f["query_token_count"] for f in sets}) > 1
