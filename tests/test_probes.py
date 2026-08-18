"""Probe features: what a selector may look at, and what it must not."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_profile_selector.probes import (
    APPROVED_FEATURES,
    ProbeError,
    ProbeResult,
    assert_no_leakage,
    extract,
    overlap_at_k,
    query_features,
    rank_agreement,
)


def probe(retriever, identifiers, scores=(), sources=()):
    return ProbeResult(retriever=retriever, identifiers=tuple(identifiers),
                       scores=tuple(scores), sources=tuple(sources))


class TestProbeValidation:
    def test_scores_must_match_the_identifiers(self):
        with pytest.raises(ProbeError):
            probe("bm25", ["a", "b"], scores=[1.0])

    def test_sources_must_match_the_identifiers(self):
        with pytest.raises(ProbeError):
            probe("bm25", ["a", "b"], sources=["doc1"])

    def test_scores_are_optional(self):
        assert probe("bm25", ["a", "b"]).top1_margin() == 0.0


class TestWithinRetrieverFeatures:
    def test_a_clear_leader_produces_a_large_margin(self):
        assert probe("bm25", ["a", "b"], scores=[10.0, 2.0]).top1_margin() == 0.8

    def test_a_tie_produces_no_margin(self):
        assert probe("bm25", ["a", "b"], scores=[5.0, 5.0]).top1_margin() == 0.0

    def test_the_margin_is_a_ratio_so_units_do_not_matter(self):
        """BM25's 40 and cosine's 0.4 must describe the same situation."""
        sparse = probe("bm25", ["a", "b"], scores=[40.0, 20.0])
        dense = probe("dense", ["a", "b"], scores=[0.4, 0.2])
        assert sparse.top1_margin() == dense.top1_margin()

    def test_a_flat_ranking_shows_little_decay(self):
        """The case where a cheap retriever could not tell its candidates apart."""
        flat = probe("bm25", ["a", "b", "c"], scores=[5.0, 4.9, 4.8])
        steep = probe("bm25", ["a", "b", "c"], scores=[5.0, 1.0, 0.2])
        assert flat.score_decay() < steep.score_decay()

    def test_duplicates_are_measured(self):
        assert probe("bm25", ["a", "a", "b"]).duplicate_ratio() > 0
        assert probe("bm25", ["a", "b", "c"]).duplicate_ratio() == 0.0

    def test_unique_sources_are_counted_when_supplied(self):
        result = probe("bm25", ["a", "b", "c"], sources=["d1", "d1", "d2"])
        assert result.unique_source_count() == 2

    def test_identifiers_stand_in_when_sources_are_absent(self):
        assert probe("bm25", ["a", "b"]).unique_source_count() == 2


class TestCrossRetrieverFeatures:
    def test_identical_results_overlap_completely(self):
        first = probe("bm25", ["a", "b", "c"])
        assert overlap_at_k(first, first) == 1.0

    def test_disjoint_results_do_not_overlap(self):
        assert overlap_at_k(probe("bm25", ["a", "b"]), probe("dense", ["c", "d"])) == 0.0

    def test_partial_overlap_is_measured(self):
        value = overlap_at_k(probe("bm25", ["a", "b"]), probe("dense", ["b", "c"]))
        assert 0 < value < 1

    def test_overlap_ignores_order(self):
        forward = probe("bm25", ["a", "b"])
        reversed_ = probe("dense", ["b", "a"])
        assert overlap_at_k(forward, reversed_) == 1.0

    def test_agreement_is_full_when_orders_match(self):
        assert rank_agreement(probe("bm25", ["a", "b", "c"]),
                              probe("dense", ["a", "b", "c"])) == 1.0

    def test_agreement_is_zero_when_orders_are_reversed(self):
        assert rank_agreement(probe("bm25", ["a", "b", "c"]),
                              probe("dense", ["c", "b", "a"])) == 0.0

    def test_agreement_is_neutral_when_nothing_is_shared(self):
        """No information is 0.5, not 0 - disagreement would be a claim."""
        assert rank_agreement(probe("bm25", ["a", "b"]), probe("dense", ["c", "d"])) == 0.5

    def test_agreement_uses_ranks_not_scores(self):
        """Comparing BM25 and cosine numbers directly compares different units."""
        sparse = probe("bm25", ["a", "b"], scores=[40.0, 20.0])
        dense = probe("dense", ["a", "b"], scores=[0.4, 0.39])
        assert rank_agreement(sparse, dense) == 1.0

    def test_a_zero_cutoff_is_refused(self):
        with pytest.raises(ProbeError):
            overlap_at_k(probe("bm25", ["a"]), probe("dense", ["a"]), k=0)


class TestQueryFeatures:
    def test_token_and_character_counts_are_reported(self):
        features = query_features("who wrote this book")
        assert features["query_token_count"] == 4
        assert features["query_char_count"] == len("who wrote this book")

    def test_numbers_are_detected(self):
        assert query_features("population in 1990")["query_has_number"] == 1.0
        assert query_features("population growth")["query_has_number"] == 0.0

    def test_comparisons_are_detected_in_both_languages(self):
        assert query_features("which is larger than the other")["query_has_comparison"] == 1.0
        assert query_features("둘 중 어느 쪽이 더 큰 차이인가")["query_has_comparison"] == 1.0

    def test_an_empty_query_does_not_crash(self):
        assert query_features("")["query_token_count"] == 0


class TestExtraction:
    def test_every_extracted_feature_is_approved(self):
        features = extract("who wrote this", [probe("bm25", ["a", "b"], scores=[5.0, 1.0]),
                                              probe("dense", ["a", "c"], scores=[0.9, 0.5])])
        assert set(features) <= set(APPROVED_FEATURES)

    def test_extraction_needs_at_least_one_probe(self):
        with pytest.raises(ProbeError):
            extract("query", [])

    def test_a_single_probe_yields_neutral_agreement(self):
        features = extract("query", [probe("bm25", ["a", "b"], scores=[5.0, 1.0])])
        assert features["rank_agreement"] == 0.5
        assert features["overlap_at_k"] == 0.0

    def test_features_are_numeric_and_finite(self):
        features = extract("query 42", [probe("bm25", ["a", "b"], scores=[5.0, 1.0]),
                                        probe("dense", ["b", "a"], scores=[0.9, 0.5])])
        for name, value in features.items():
            assert isinstance(value, float), name
            assert value == value, name  # not NaN

    def test_extraction_is_deterministic(self):
        probes = [probe("bm25", ["a", "b"], scores=[5.0, 1.0]),
                  probe("dense", ["a", "c"], scores=[0.9, 0.5])]
        assert extract("query", probes) == extract("query", probes)


class TestLeakageGuard:
    def test_clean_features_pass(self):
        assert_no_leakage(extract("query", [probe("bm25", ["a", "b"], scores=[5.0, 1.0])]))

    @pytest.mark.parametrize("name", [
        "gold_recall", "evidence_label", "answer_quality", "profile_regret",
        "chosen_outcome",
    ])
    def test_a_feature_derived_from_the_answer_is_refused(self, name):
        """These would not exist at selection time; a model using them cannot ship."""
        with pytest.raises(ProbeError) as error:
            assert_no_leakage({name: 1.0, "top1_margin": 0.5})
        assert name in str(error.value)

    def test_additional_forbidden_terms_can_be_supplied(self):
        with pytest.raises(ProbeError):
            assert_no_leakage({"future_click_rate": 1.0}, forbidden=["future"])

    def test_the_extractor_has_no_parameter_for_labels(self):
        """Structural, not documented: there is nowhere for a label to enter."""
        import inspect

        parameters = set(inspect.signature(extract).parameters)
        assert parameters == {"query", "probes", "k"}
