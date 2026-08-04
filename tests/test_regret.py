"""Tests for exact regret over synthetic profile-quality outcomes."""

import math
import unittest

from rag_profile_selector.regret import (
    QueryProfileQualityOutcomes,
    RegretValidationError,
    calculate_exact_regret,
)


class ExactRegretTests(unittest.TestCase):
    def test_best_selected_profile_has_zero_regret(self) -> None:
        outcomes = QueryProfileQualityOutcomes(
            "query-1", ("bm25-k4", "dense-k4"), {"bm25-k4": 0.4, "dense-k4": 0.8}
        )

        self.assertEqual(0.0, calculate_exact_regret(outcomes, "dense-k4"))

    def test_lower_quality_selection_has_positive_regret(self) -> None:
        outcomes = QueryProfileQualityOutcomes(
            "query-1", ("bm25-k4", "dense-k4"), {"bm25-k4": 0.25, "dense-k4": 0.75}
        )

        self.assertEqual(0.5, calculate_exact_regret(outcomes, "bm25-k4"))

    def test_tied_best_profiles_both_have_zero_regret_without_selecting_a_tie(self) -> None:
        outcomes = QueryProfileQualityOutcomes(
            "query-1",
            ("bm25-k4", "dense-k4", "hybrid-k4"),
            {"bm25-k4": 0.6, "dense-k4": 0.9, "hybrid-k4": 0.9},
        )

        self.assertEqual(0.0, calculate_exact_regret(outcomes, "dense-k4"))
        self.assertEqual(0.0, calculate_exact_regret(outcomes, "hybrid-k4"))
        self.assertAlmostEqual(0.3, calculate_exact_regret(outcomes, "bm25-k4"))

    def test_incomplete_or_extra_profile_qualities_are_rejected(self) -> None:
        for qualities in (
            {"bm25-k4": 0.5},
            {"bm25-k4": 0.5, "dense-k4": 0.7, "hybrid-k4": 0.8},
        ):
            with self.subTest(qualities=qualities):
                with self.assertRaises(RegretValidationError):
                    QueryProfileQualityOutcomes(
                        "query-1", ("bm25-k4", "dense-k4"), qualities
                    )

    def test_non_finite_scores_and_invalid_selection_are_rejected(self) -> None:
        for score in (math.nan, math.inf, -math.inf):
            with self.subTest(score=score):
                with self.assertRaises(RegretValidationError):
                    QueryProfileQualityOutcomes(
                        "query-1", ("bm25-k4",), {"bm25-k4": score}
                    )

        outcomes = QueryProfileQualityOutcomes(
            "query-1", ("bm25-k4",), {"bm25-k4": 0.5}
        )
        with self.assertRaises(RegretValidationError):
            calculate_exact_regret(outcomes, "dense-k4")

    def test_equivalent_inputs_produce_stable_results_and_are_immutable(self) -> None:
        source_qualities = {"bm25-k4": 0.2, "dense-k4": 0.8}
        first = QueryProfileQualityOutcomes(
            "query-1", ["bm25-k4", "dense-k4"], source_qualities
        )
        source_qualities["bm25-k4"] = 1.0
        second = QueryProfileQualityOutcomes(
            "query-1", ("bm25-k4", "dense-k4"), {"dense-k4": 0.8, "bm25-k4": 0.2}
        )

        self.assertAlmostEqual(0.6, calculate_exact_regret(first, "bm25-k4"))
        self.assertEqual(calculate_exact_regret(first, "bm25-k4"), calculate_exact_regret(second, "bm25-k4"))
        with self.assertRaises(TypeError):
            first.qualities["bm25-k4"] = 1.0  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
