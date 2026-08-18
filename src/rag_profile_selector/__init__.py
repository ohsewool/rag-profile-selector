"""Retrieval profiles, fusion, evidence metrics, and citation grounding."""

from .corpus import (
    Corpus,
    CorpusError,
    CorpusManifest,
    SealedSplitError,
    SplitAssignment,
    validate_evidence_mapping,
)
from .probes import (
    APPROVED_FEATURES,
    ProbeError,
    ProbeResult,
    assert_no_leakage,
    extract,
)
from .profiles import (
    APPROVED_PROFILES,
    ProfileValidationError,
    RetrievalMethod,
    RetrievalProfile,
    resolve_profile,
    validate_profile,
)

__all__ = [
    "APPROVED_FEATURES",
    "ProbeError",
    "ProbeResult",
    "assert_no_leakage",
    "extract",
    "Corpus",
    "CorpusError",
    "CorpusManifest",
    "SealedSplitError",
    "SplitAssignment",
    "validate_evidence_mapping",
    "APPROVED_PROFILES",
    "ProfileValidationError",
    "RetrievalMethod",
    "RetrievalProfile",
    "resolve_profile",
    "validate_profile",
]
