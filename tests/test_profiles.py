"""Tests for the bounded retrieval-profile configuration catalog."""

from dataclasses import FrozenInstanceError
import unittest

from rag_profile_selector.profiles import (
    APPROVED_PROFILES,
    ProfileValidationError,
    RetrievalMethod,
    RetrievalProfile,
    resolve_profile,
    validate_profile,
)


class RetrievalProfileTests(unittest.TestCase):
    def test_catalog_contains_exactly_the_four_approved_profiles(self) -> None:
        self.assertEqual(
            tuple((profile.method, profile.k) for profile in APPROVED_PROFILES),
            (
                (RetrievalMethod.BM25, 4),
                (RetrievalMethod.DENSE, 4),
                (RetrievalMethod.HYBRID_RRF, 4),
                (RetrievalMethod.HYBRID_RRF, 8),
            ),
        )

    def test_identifiers_and_serialization_are_stable(self) -> None:
        profiles = APPROVED_PROFILES
        self.assertEqual(
            tuple(profile.profile_id for profile in profiles),
            ("bm25-k4", "dense-k4", "hybrid-rrf-k4", "hybrid-rrf-k8"),
        )
        self.assertEqual(
            profiles[0].serialize(),
            '{"k":4,"method":"bm25","profile_id":"bm25-k4"}',
        )
        self.assertEqual(profiles[0].serialize(), profiles[0].serialize())

    def test_valid_lookup_returns_the_canonical_catalog_value(self) -> None:
        profile = resolve_profile("hybrid-rrf-k8")
        self.assertIs(profile, APPROVED_PROFILES[3])
        self.assertIs(validate_profile(profile), profile)

    def test_unknown_profile_id_is_rejected(self) -> None:
        with self.assertRaises(ProfileValidationError):
            resolve_profile("dense-k8")

    def test_unsupported_methods_and_k_values_are_rejected(self) -> None:
        with self.assertRaises(ProfileValidationError):
            RetrievalProfile("sparse", 4)  # type: ignore[arg-type]
        with self.assertRaises(ProfileValidationError):
            RetrievalProfile(RetrievalMethod.BM25, 8)
        with self.assertRaises(ProfileValidationError):
            RetrievalProfile(RetrievalMethod.HYBRID_RRF, 6)

    def test_profile_values_are_immutable(self) -> None:
        profile = APPROVED_PROFILES[0]
        with self.assertRaises(FrozenInstanceError):
            profile.k = 8  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
