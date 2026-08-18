"""Selection and the baselines it must beat to be worth building."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_profile_selector.selector import (
    Evaluation,
    FixedSelector,
    OracleSelector,
    QueryOutcome,
    RuleSelector,
    SelectorError,
    evaluate,
    headroom,
)

PROFILES = ("bm25-k4", "dense-k4", "hybrid-rrf-k4")


@pytest.fixture
def outcomes():
    """Three queries where no single profile is best for all of them."""
    return [
        QueryOutcome("q1", {"bm25-k4": 1.0, "dense-k4": 0.4, "hybrid-rrf-k4": 0.8}),
        QueryOutcome("q2", {"bm25-k4": 0.3, "dense-k4": 1.0, "hybrid-rrf-k4": 0.7}),
        QueryOutcome("q3", {"bm25-k4": 0.5, "dense-k4": 0.5, "hybrid-rrf-k4": 1.0}),
    ]


@pytest.fixture
def features():
    return {
        "q1": {"top1_margin": 0.9, "rank_agreement": 0.9},
        "q2": {"top1_margin": 0.1, "rank_agreement": 0.2},
        "q3": {"top1_margin": 0.4, "rank_agreement": 0.5},
    }


class TestOutcomes:
    def test_the_best_profile_is_identified(self, outcomes):
        assert outcomes[0].best_profile == "bm25-k4"
        assert outcomes[1].best_profile == "dense-k4"

    def test_regret_is_zero_for_the_best_choice(self, outcomes):
        assert outcomes[0].regret_of("bm25-k4") == 0.0

    def test_regret_measures_what_was_given_up(self, outcomes):
        assert outcomes[0].regret_of("dense-k4") == pytest.approx(0.6)

    def test_an_unrecorded_profile_is_an_error_not_a_zero(self, outcomes):
        """Silently scoring an unmeasured profile as perfect would flatter it."""
        with pytest.raises(SelectorError):
            outcomes[0].regret_of("reranker-k8")

    def test_an_outcome_with_no_profiles_is_refused(self):
        with pytest.raises(SelectorError):
            QueryOutcome("q9", {})

    def test_ties_resolve_deterministically(self):
        outcome = QueryOutcome("q", {"b": 1.0, "a": 1.0})
        assert outcome.best_profile == "b"  # stable, not arbitrary per run


class TestFixedBaseline:
    def test_a_fixed_selector_always_returns_its_profile(self, outcomes, features):
        result = evaluate(FixedSelector("bm25-k4"), outcomes, features)
        assert set(result.chosen.values()) == {"bm25-k4"}

    def test_a_fixed_selector_is_optimal_only_where_its_profile_wins(self, outcomes, features):
        result = evaluate(FixedSelector("bm25-k4"), outcomes, features)
        assert result.optimal_choices == 1
        assert result.mean_regret > 0

    def test_the_evaluation_names_the_selector(self, outcomes, features):
        assert evaluate(FixedSelector("dense-k4"), outcomes, features).selector == "fixed:dense-k4"


class TestOracleBound:
    def test_the_oracle_is_always_optimal(self, outcomes, features):
        oracle = OracleSelector({outcome.query_id: outcome for outcome in outcomes})
        result = evaluate(oracle, outcomes, features)
        assert result.mean_regret == 0.0
        assert result.optimal_rate == 1.0

    def test_the_oracle_refuses_a_query_it_has_no_outcome_for(self, features):
        oracle = OracleSelector({})
        with pytest.raises(SelectorError):
            evaluate(oracle, [QueryOutcome("q1", {"bm25-k4": 1.0})], features)

    def test_the_oracle_reads_outcomes_rather_than_features(self, outcomes, features):
        """It is a bound, not a candidate: it cannot be mistaken for shippable."""
        import inspect

        parameters = set(inspect.signature(OracleSelector.__init__).parameters)
        assert "outcomes" in parameters


class TestRuleBaseline:
    def test_a_rule_chooses_from_features(self, outcomes, features):
        rule = RuleSelector(
            lambda f: "bm25-k4" if f["top1_margin"] > 0.5 else "dense-k4",
            name="margin-rule",
        )
        result = evaluate(rule, outcomes, features)
        assert result.chosen["q1"] == "bm25-k4"
        assert result.chosen["q2"] == "dense-k4"

    def test_a_good_rule_beats_a_fixed_profile(self, outcomes, features):
        rule = RuleSelector(
            lambda f: "bm25-k4" if f["top1_margin"] > 0.5
            else "dense-k4" if f["rank_agreement"] < 0.3 else "hybrid-rrf-k4"
        )
        assert (evaluate(rule, outcomes, features).mean_regret
                < evaluate(FixedSelector("bm25-k4"), outcomes, features).mean_regret)

    def test_a_rule_reading_a_leaking_feature_is_refused(self, outcomes):
        """A rule cannot be allowed to see the answer just because it is a rule."""
        leaking = {"q1": {"gold_recall": 1.0}, "q2": {"gold_recall": 0.0},
                   "q3": {"gold_recall": 0.5}}
        rule = RuleSelector(lambda f: "bm25-k4")
        with pytest.raises(SelectorError):
            evaluate(rule, outcomes, leaking)


class TestEvaluation:
    def test_missing_features_are_an_error(self, outcomes):
        with pytest.raises(SelectorError):
            evaluate(FixedSelector("bm25-k4"), outcomes, {})

    def test_an_empty_evaluation_is_refused(self, features):
        with pytest.raises(SelectorError):
            evaluate(FixedSelector("bm25-k4"), [], features)

    def test_max_regret_reports_the_worst_single_query(self, outcomes, features):
        result = evaluate(FixedSelector("dense-k4"), outcomes, features)
        assert result.max_regret == pytest.approx(0.6)

    def test_optimal_rate_is_a_fraction_of_the_queries(self, outcomes, features):
        result = evaluate(FixedSelector("bm25-k4"), outcomes, features)
        assert result.optimal_rate == pytest.approx(1 / 3)


class TestHeadroom:
    def test_headroom_is_the_gap_between_the_best_fixed_and_the_oracle(self, outcomes):
        result = headroom(outcomes, PROFILES)
        assert result["headroom"] > 0
        assert result["oracle_mean_regret"] == 0.0

    def test_the_best_fixed_profile_is_named(self, outcomes):
        result = headroom(outcomes, PROFILES)
        assert result["best_fixed_profile"] in PROFILES

    def test_every_profile_is_reported_so_the_choice_is_inspectable(self, outcomes):
        result = headroom(outcomes, PROFILES)
        assert set(result["per_profile_mean_regret"]) == set(PROFILES)

    def test_no_headroom_when_one_profile_wins_everywhere(self):
        """The result that says stop: selection cannot help on this data."""
        outcomes = [
            QueryOutcome("q1", {"bm25-k4": 1.0, "dense-k4": 0.2}),
            QueryOutcome("q2", {"bm25-k4": 1.0, "dense-k4": 0.3}),
        ]
        assert headroom(outcomes, ("bm25-k4", "dense-k4"))["headroom"] == 0.0

    def test_headroom_requires_outcomes(self):
        with pytest.raises(SelectorError):
            headroom([], PROFILES)


class TestComparability:
    def test_all_selectors_are_scored_the_same_way(self, outcomes, features):
        """The comparison is only fair if the measure does not vary by arm."""
        oracle = OracleSelector({outcome.query_id: outcome for outcome in outcomes})
        results = [
            evaluate(FixedSelector("bm25-k4"), outcomes, features),
            evaluate(RuleSelector(lambda f: "hybrid-rrf-k4"), outcomes, features),
            evaluate(oracle, outcomes, features),
        ]
        assert all(isinstance(result, Evaluation) for result in results)
        assert all(result.queries == len(outcomes) for result in results)
        assert results[-1].mean_regret <= min(result.mean_regret for result in results[:-1])
