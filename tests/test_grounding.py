"""Citation grounding: the failure modes identifier-level metrics cannot see."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "document-intelligence" / "src"))

pytest.importorskip("document_intelligence", reason="document-intelligence is unavailable")

from document_intelligence import (  # noqa: E402
    BoundingBox,
    Document,
    Page,
    RegionReference,
    TableRegion,
    TextRegion,
)

from rag_profile_selector.evidence_metrics import calculate_evidence_metrics  # noqa: E402
from rag_profile_selector.fusion import fuse_reciprocal_rank  # noqa: E402
from rag_profile_selector.grounding import (  # noqa: E402
    GroundingError,
    GroundingIndex,
    measure_citation_accuracy,
    to_evidence_citation,
)


def box(top: float = 0.1) -> BoundingBox:
    """Normalized coordinates, which is the document model's default space."""
    return BoundingBox(left=0.1, top=top, right=0.9, bottom=top + 0.1)


@pytest.fixture
def document() -> Document:
    return Document(
        identifier="report",
        checksum="c" * 64,
        pages=(
            Page(number=1, width=612.0, height=792.0, regions=(
                TextRegion(identifier="p1-intro", bounding_box=box(0.1)),
                TextRegion(identifier="p1-body", bounding_box=box(0.4)),
            )),
            Page(number=2, width=612.0, height=792.0, regions=(
                TableRegion(identifier="p2-table", bounding_box=box(0.1)),
            )),
        ),
    )


@pytest.fixture
def index(document) -> GroundingIndex:
    return GroundingIndex.from_documents([document])


class TestIndexConstruction:
    def test_every_region_is_indexed(self, index):
        assert len(index) == 3
        assert "report:1:p1-intro" in index
        assert "report:2:p2-table" in index

    def test_resolution_carries_the_bounding_box(self, index):
        citation = index.resolve("report:2:p2-table")
        assert citation.page_number == 2
        assert citation.region_identifier == "p2-table"
        assert citation.bounding_box == box(0.1)
        assert citation.is_region_level

    def test_unknown_identifier_resolves_to_nothing(self, index):
        assert index.resolve("report:9:missing") is None

    def test_duplicate_identifiers_are_refused(self, index):
        with pytest.raises(GroundingError):
            index.add("report:1:p1-intro", document_id="report", page_number=1)

    def test_custom_identifier_scheme_is_honoured(self, document):
        index = GroundingIndex.from_documents(
            [document], identifier_for=lambda doc, page, region: f"{doc}#{region}"
        )
        assert "report#p1-intro" in index


class TestGroundingRankedResults:
    def test_fused_retrieval_results_ground_in_rank_order(self, index):
        fused = fuse_reciprocal_rank(
            [["report:1:p1-body", "report:1:p1-intro"],
             ["report:1:p1-body", "report:2:p2-table"]],
            k=8,   # only 4 and 8 are approved hybrid RRF constants
        )
        result = index.ground(fused)
        assert [citation.rank for citation in result.grounded] == [1, 2, 3]
        assert result.grounded[0].identifier == "report:1:p1-body"
        assert result.grounded[0].score > 0
        assert result.coverage == 1.0

    def test_a_hallucinated_identifier_is_reported_as_ungrounded(self, index):
        """Identifier metrics would score this as an ordinary miss; a reader gets a dead link."""
        result = index.ground(["report:1:p1-intro", "report:7:invented"])
        assert result.ungrounded == ("report:7:invented",)
        assert result.coverage == 0.5

    def test_plain_identifier_strings_are_accepted(self, index):
        assert len(index.ground(["report:1:p1-intro"]).grounded) == 1

    def test_non_string_results_are_refused(self, index):
        with pytest.raises(GroundingError):
            index.ground([42])


class TestCitationAccuracy:
    def test_correct_citations_score_fully(self, index):
        result = index.ground(["report:1:p1-intro", "report:2:p2-table"])
        accuracy = measure_citation_accuracy(result, {
            "report:1:p1-intro": (1, "p1-intro"),
            "report:2:p2-table": (2, "p2-table"),
        })
        assert accuracy.grounding_rate == 1.0
        assert accuracy.page_accuracy == 1.0
        assert accuracy.region_accuracy == 1.0
        assert accuracy.misplaced == 0

    def test_right_answer_wrong_page_is_misplaced(self, index):
        """The distinguishing case: retrieval is right, the pointer is not."""
        result = index.ground(["report:1:p1-intro"])
        accuracy = measure_citation_accuracy(result, {"report:1:p1-intro": (2, "p2-table")})
        assert accuracy.misplaced == 1
        assert accuracy.page_accuracy == 0.0

    def test_right_page_wrong_region_is_misplaced(self, index):
        result = index.ground(["report:1:p1-intro"])
        accuracy = measure_citation_accuracy(result, {"report:1:p1-intro": (1, "p1-body")})
        assert accuracy.page_correct == 1
        assert accuracy.region_accuracy == 0.0
        assert accuracy.misplaced == 1

    def test_page_level_citation_where_a_region_exists_is_coarse(self, index):
        index.add("report:page-1", document_id="report", page_number=1)
        result = index.ground(["report:page-1"])
        accuracy = measure_citation_accuracy(result, {"report:page-1": (1, "p1-intro")})
        assert accuracy.coarse == 1
        assert accuracy.page_accuracy == 1.0
        assert accuracy.region_accuracy == 0.0

    def test_unverifiable_citation_counts_against_accuracy(self, index):
        """An identifier nobody can confirm is not a correct citation."""
        result = index.ground(["report:1:p1-intro"])
        accuracy = measure_citation_accuracy(result, {})
        assert accuracy.misplaced == 1

    def test_ungrounded_results_lower_the_grounding_rate_only(self, index):
        result = index.ground(["report:1:p1-intro", "report:9:missing"])
        accuracy = measure_citation_accuracy(result, {"report:1:p1-intro": (1, "p1-intro")})
        assert accuracy.grounding_rate == 0.5
        assert accuracy.page_accuracy == 1.0  # computed over what was grounded


class TestComplementarityWithRetrievalMetrics:
    def test_identical_retrieval_scores_can_hide_different_citation_quality(self, index):
        """Why this layer exists: precision/recall cannot separate these two runs."""
        gold_identifiers = ["report:1:p1-intro"]
        predicted = ["report:1:p1-intro"]

        retrieval = calculate_evidence_metrics(predicted, gold_identifiers)
        assert retrieval.precision == 1.0 and retrieval.recall == 1.0

        result = index.ground(predicted)
        accurate = measure_citation_accuracy(result, {"report:1:p1-intro": (1, "p1-intro")})
        misplaced = measure_citation_accuracy(result, {"report:1:p1-intro": (2, "p2-table")})

        assert accurate.region_accuracy == 1.0
        assert misplaced.region_accuracy == 0.0  # same retrieval score, different citations


class TestDocumentModelInterop:
    def test_region_citation_round_trips_into_the_document_model(self, index):
        citation = to_evidence_citation(index.resolve("report:2:p2-table"))
        assert citation.references == (RegionReference(2, "p2-table"),)
        assert citation.bounding_box == box(0.1)

    def test_page_level_citation_round_trips(self, index):
        index.add("report:page-2", document_id="report", page_number=2)
        citation = to_evidence_citation(index.resolve("report:page-2"))
        assert citation.references[0].page_number == 2
