"""Resolve retrieved evidence identifiers back to where they came from.

Retrieval evaluation normally stops at "was the right chunk returned?".  That
leaves the question a reader actually asks unanswered: *where in the document is
this, and is the citation pointing at the right place?*

This module joins the two halves.  Retrieval (`fusion`, `evidence_metrics`)
works with opaque identifiers; the document model works with pages, regions, and
bounding boxes.  A grounding index maps one to the other, which makes three
failure modes measurable that identifier-level metrics cannot see:

ungrounded citation
    the retriever returned an identifier that resolves to nothing.  At the
    identifier level this looks like an ordinary result; to a reader it is a
    citation that goes nowhere.

misplaced citation
    the identifier resolves, but to a different page or region than the passage
    it is standing in for — the answer is right and the pointer is wrong.

coarse citation
    the identifier resolves to a whole page where a region was available.
    Correct, but weaker evidence than the document could support.

None of these are visible in precision/recall over identifiers, which is the
gap this project is testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

try:  # the document model lives in its own repository during the merge
    from document_intelligence import (
        BoundingBox,
        Document,
        EvidenceCitation,
        PageReference,
        RegionReference,
    )
except ImportError as error:  # pragma: no cover - surfaced by the caller
    raise ImportError(
        "grounding requires the document_intelligence package; add its src/ to the path"
    ) from error


class GroundingError(ValueError):
    """Raised when an index or citation cannot be built consistently."""


@dataclass(frozen=True)
class GroundedCitation:
    """A retrieved identifier resolved to a location in a document."""

    identifier: str
    document_id: str
    page_number: int
    region_identifier: str | None
    bounding_box: BoundingBox | None
    rank: int
    score: float

    @property
    def is_region_level(self) -> bool:
        return self.region_identifier is not None


@dataclass(frozen=True)
class GroundingResult:
    """What grounding produced for one query."""

    grounded: tuple[GroundedCitation, ...]
    ungrounded: tuple[str, ...]

    @property
    def coverage(self) -> float:
        total = len(self.grounded) + len(self.ungrounded)
        return len(self.grounded) / total if total else 0.0


class GroundingIndex:
    """Maps evidence identifiers to the document location they came from.

    Built from documents rather than declared separately, so an identifier can
    only be grounded to a region that actually exists in the document — a stale
    index cannot invent a location.
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[str, int, str | None, BoundingBox | None]] = {}

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        identifier_for: "callable[[str, int, str], str] | None" = None,
    ) -> "GroundingIndex":
        """Index every region of every document.

        ``identifier_for`` decides how a retrieval identifier is derived from a
        location; the default matches the ``doc:page:region`` convention used by
        the evaluation fixtures.
        """
        index = cls()
        derive = identifier_for or (lambda doc, page, region: f"{doc}:{page}:{region}")
        for document in documents:
            for page in document.pages:
                for region in page.regions:
                    index.add(
                        derive(document.identifier, page.number, region.identifier),
                        document_id=document.identifier,
                        page_number=page.number,
                        region_identifier=region.identifier,
                        bounding_box=region.bounding_box,
                    )
        return index

    def add(self, identifier: str, *, document_id: str, page_number: int,
            region_identifier: str | None = None,
            bounding_box: BoundingBox | None = None) -> None:
        if not identifier:
            raise GroundingError("evidence identifier must not be empty")
        if identifier in self._entries:
            raise GroundingError(f"identifier is already grounded: {identifier}")
        self._entries[identifier] = (document_id, page_number, region_identifier, bounding_box)

    def __contains__(self, identifier: str) -> bool:
        return identifier in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def resolve(self, identifier: str, *, rank: int = 0,
                score: float = 0.0) -> GroundedCitation | None:
        entry = self._entries.get(identifier)
        if entry is None:
            return None
        document_id, page_number, region_identifier, bounding_box = entry
        return GroundedCitation(
            identifier=identifier,
            document_id=document_id,
            page_number=page_number,
            region_identifier=region_identifier,
            bounding_box=bounding_box,
            rank=rank,
            score=score,
        )

    def ground(self, ranked: Sequence[object]) -> GroundingResult:
        """Ground a ranked result list of identifiers or ``FusedEvidence``."""
        grounded: list[GroundedCitation] = []
        ungrounded: list[str] = []
        for rank, item in enumerate(ranked, start=1):
            identifier = getattr(item, "identifier", item)
            score = float(getattr(item, "score", 0.0))
            if not isinstance(identifier, str):
                raise GroundingError("ranked results must be identifiers or FusedEvidence")
            citation = self.resolve(identifier, rank=rank, score=score)
            if citation is None:
                ungrounded.append(identifier)
            else:
                grounded.append(citation)
        return GroundingResult(tuple(grounded), tuple(ungrounded))


def to_evidence_citation(citation: GroundedCitation) -> EvidenceCitation:
    """Express a grounded citation in the document model's own vocabulary."""
    if citation.region_identifier is None:
        return EvidenceCitation(
            identifier=citation.identifier,
            references=(PageReference(citation.page_number),),
        )
    return EvidenceCitation(
        identifier=citation.identifier,
        references=(RegionReference(citation.page_number, citation.region_identifier),),
        bounding_box=citation.bounding_box,
    )


@dataclass(frozen=True)
class CitationAccuracy:
    """How well the citations point at where the evidence actually is."""

    grounded: int
    ungrounded: int
    page_correct: int
    region_correct: int
    misplaced: int
    coarse: int

    @property
    def grounding_rate(self) -> float:
        total = self.grounded + self.ungrounded
        return self.grounded / total if total else 0.0

    @property
    def page_accuracy(self) -> float:
        return self.page_correct / self.grounded if self.grounded else 0.0

    @property
    def region_accuracy(self) -> float:
        return self.region_correct / self.grounded if self.grounded else 0.0


def measure_citation_accuracy(
    result: GroundingResult,
    gold: Mapping[str, tuple[int, str | None]],
) -> CitationAccuracy:
    """Compare grounded citations with where the evidence really is.

    ``gold`` maps an identifier to ``(page_number, region_identifier or None)``.
    An identifier absent from ``gold`` is counted as misplaced rather than
    ignored: a citation nobody can confirm is not a correct citation.
    """
    page_correct = region_correct = misplaced = coarse = 0
    for citation in result.grounded:
        expected = gold.get(citation.identifier)
        if expected is None:
            misplaced += 1
            continue
        expected_page, expected_region = expected
        if citation.page_number != expected_page:
            misplaced += 1
            continue
        page_correct += 1
        if expected_region is None:
            region_correct += 1
        elif citation.region_identifier == expected_region:
            region_correct += 1
        elif citation.region_identifier is None:
            coarse += 1  # right page, but page-level where a region existed
        else:
            misplaced += 1
    return CitationAccuracy(
        grounded=len(result.grounded),
        ungrounded=len(result.ungrounded),
        page_correct=page_correct,
        region_correct=region_correct,
        misplaced=misplaced,
        coarse=coarse,
    )
