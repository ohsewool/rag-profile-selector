"""Tests for deterministic synthetic evidence metrics."""

import unittest

from rag_profile_selector.evidence_metrics import (
    EvidenceMetrics,
    calculate_evidence_metrics,
)


class EvidenceMetricsTests(unittest.TestCase):
    def test_exact_match_has_perfect_scores(self) -> None:
        result = calculate_evidence_metrics(("e1", "e2"), ("e2", "e1"))

        self.assertEqual(EvidenceMetrics(1.0, 1.0, 1.0), result)

    def test_partial_match_has_exact_scores(self) -> None:
        result = calculate_evidence_metrics(("e1", "e2", "e3"), ("e2", "e4"))

        self.assertEqual(1.0 / 3.0, result.precision)
        self.assertEqual(0.5, result.recall)
        self.assertEqual(0.4, result.f1)

    def test_no_match_has_zero_scores(self) -> None:
        result = calculate_evidence_metrics(("e1",), ("e2",))

        self.assertEqual(EvidenceMetrics(0.0, 0.0, 0.0), result)

    def test_duplicates_are_ignored_deterministically(self) -> None:
        result = calculate_evidence_metrics(
            ("e1", "e1", "e2", "e2"), ("e2", "e2", "e3")
        )

        self.assertEqual(EvidenceMetrics(0.5, 0.5, 0.5), result)

    def test_empty_predictions_have_zero_scores(self) -> None:
        result = calculate_evidence_metrics((), ("e1",))

        self.assertEqual(EvidenceMetrics(0.0, 0.0, 0.0), result)

    def test_empty_gold_has_zero_scores(self) -> None:
        result = calculate_evidence_metrics(("e1",), ())

        self.assertEqual(EvidenceMetrics(0.0, 0.0, 0.0), result)

    def test_both_empty_inputs_have_zero_scores(self) -> None:
        self.assertEqual(
            EvidenceMetrics(0.0, 0.0, 0.0), calculate_evidence_metrics((), ())
        )

    def test_invalid_identifiers_are_rejected(self) -> None:
        for identifiers in (("",), ("contains space",), ("\t",)):
            with self.subTest(identifiers=identifiers):
                with self.assertRaises(ValueError):
                    calculate_evidence_metrics(identifiers, ("e1",))

        with self.assertRaises(TypeError):
            calculate_evidence_metrics((1,), ("e1",))  # type: ignore[arg-type]

    def test_swapping_inputs_swaps_precision_and_recall(self) -> None:
        forward = calculate_evidence_metrics(("e1", "e2", "e3"), ("e2", "e4"))
        reverse = calculate_evidence_metrics(("e2", "e4"), ("e1", "e2", "e3"))

        self.assertEqual(forward.precision, reverse.recall)
        self.assertEqual(forward.recall, reverse.precision)
        self.assertEqual(forward.f1, reverse.f1)

    def test_equivalent_repeated_calls_are_stable(self) -> None:
        first = calculate_evidence_metrics(["e1", "e2", "e1"], ["e2", "e3"])
        second = calculate_evidence_metrics(("e2", "e1"), ("e3", "e2"))

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
