"""Tests for bounded deterministic reciprocal-rank fusion."""

import unittest

from rag_profile_selector.fusion import (
    RRF_RANK_CONSTANT,
    FusedEvidence,
    fuse_reciprocal_rank,
    resolve_hybrid_rrf_k,
)


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_exact_scores(self) -> None:
        result = fuse_reciprocal_rank((("e1", "e2"), ("e2", "e3")), k=4)

        self.assertEqual([item.identifier for item in result], ["e2", "e1", "e3"])
        self.assertEqual(result[0].score, 1 / 62 + 1 / 61)
        self.assertEqual(result[1].score, 1 / 61)
        self.assertEqual(result[2].score, 1 / 62)
        self.assertEqual(RRF_RANK_CONSTANT, 60)

    def test_duplicate_in_one_list_counts_at_first_rank_only(self) -> None:
        result = fuse_reciprocal_rank((("e1", "e1", "e2"),), k=4)

        self.assertEqual(
            result,
            (FusedEvidence("e1", 1 / 61), FusedEvidence("e2", 1 / 62)),
        )

    def test_deterministic_ties_follow_first_seen_order(self) -> None:
        result = fuse_reciprocal_rank((("left",), ("right",)), k=4)

        self.assertEqual([item.identifier for item in result], ["left", "right"])

    def test_empty_input(self) -> None:
        self.assertEqual(fuse_reciprocal_rank((), k=4), ())
        self.assertEqual(fuse_reciprocal_rank(((), ()), k=8), ())

    def test_invalid_identifiers_and_rank_cutoffs(self) -> None:
        with self.assertRaises(ValueError):
            fuse_reciprocal_rank((("",),), k=4)
        with self.assertRaises(ValueError):
            fuse_reciprocal_rank((("e1",),), k=3)
        with self.assertRaises(ValueError):
            resolve_hybrid_rrf_k(True)
        with self.assertRaises(ValueError):
            resolve_hybrid_rrf_k(8.0)

    def test_invalid_ranked_inputs(self) -> None:
        with self.assertRaises(TypeError):
            fuse_reciprocal_rank("e1", k=4)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            fuse_reciprocal_rank((("e1",), "e2"), k=4)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fuse_reciprocal_rank((("e1", 2),), k=4)  # type: ignore[arg-type]

    def test_order_is_stable(self) -> None:
        inputs = (("e3", "e2", "e1"), ("e2", "e3", "e1"))
        first = fuse_reciprocal_rank(inputs, k=4)
        second = fuse_reciprocal_rank(inputs, k=4)

        self.assertEqual(first, second)
        self.assertEqual([item.identifier for item in first], ["e3", "e2", "e1"])

    def test_exact_k_four_and_k_eight_cutoffs(self) -> None:
        identifiers = tuple(f"e{number}" for number in range(1, 10))

        k_four = fuse_reciprocal_rank((identifiers,), k=4)
        k_eight = fuse_reciprocal_rank((identifiers,), k=8)

        self.assertEqual([item.identifier for item in k_four], ["e1", "e2", "e3", "e4"])
        self.assertEqual(
            [item.identifier for item in k_eight],
            ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"],
        )


if __name__ == "__main__":
    unittest.main()
