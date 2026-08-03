"""Deterministic evidence-identifier precision, recall, and F1 metrics.

The functions in this module operate only on synthetic evidence identifiers.
They intentionally make no retrieval, benchmark, or thresholding decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


EvidenceIdentifier = str


@dataclass(frozen=True)
class EvidenceMetrics:
    """Evidence-overlap metrics for one prediction and gold-identifier pair."""

    precision: float
    recall: float
    f1: float


def _normalise_identifiers(
    identifiers: Iterable[EvidenceIdentifier], *, argument_name: str
) -> frozenset[EvidenceIdentifier]:
    """Validate identifiers and return their duplicate-insensitive representation.

    An evidence identifier is a non-empty, whitespace-free string.  Duplicate
    identifiers have no additional effect because evidence matching is set
    based.
    """

    try:
        values = tuple(identifiers)
    except TypeError as error:
        raise TypeError(f"{argument_name} must be an iterable of identifiers") from error

    for identifier in values:
        if not isinstance(identifier, str):
            raise TypeError(f"{argument_name} identifiers must be strings")
        if not identifier or identifier.isspace() or any(
            character.isspace() for character in identifier
        ):
            raise ValueError(
                f"{argument_name} identifiers must be non-empty and whitespace-free"
            )

    return frozenset(values)


def calculate_evidence_metrics(
    predicted_identifiers: Iterable[EvidenceIdentifier],
    gold_identifiers: Iterable[EvidenceIdentifier],
) -> EvidenceMetrics:
    """Calculate deterministic set-based evidence precision, recall, and F1.

    Empty predictions have precision 0.0.  Empty gold evidence has recall 0.0.
    Consequently, F1 is 0.0 whenever either input is empty.  This makes the
    metric defined without treating an empty pair as an exact evidence match.
    """

    predicted = _normalise_identifiers(
        predicted_identifiers, argument_name="predicted_identifiers"
    )
    gold = _normalise_identifiers(gold_identifiers, argument_name="gold_identifiers")

    matches = len(predicted & gold)
    precision = matches / len(predicted) if predicted else 0.0
    recall = matches / len(gold) if gold else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    return EvidenceMetrics(precision=precision, recall=recall, f1=f1)
