"""Pin the experiment's claims so they cannot drift unnoticed."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "document-intelligence" / "src"))

pytest.importorskip("document_intelligence", reason="document-intelligence is unavailable")

from experiments.citation_quality import (  # noqa: E402
    build_corpus,
    build_queries,
    render,
    run,
)


@pytest.fixture(scope="module")
def outcomes():
    return run()


class TestCorpus:
    def test_corpus_has_region_structure_on_every_page(self):
        """"Every page" over an empty corpus is a sentence about nothing.

        All the assertions here live inside the loop, so a `build_corpus()` that
        returned nothing would pass this test while checking no page at all.
        The count makes the claim say what it means.
        """
        corpus = build_corpus()
        pages = 0
        for document in corpus:
            for page in document.pages:
                assert page.regions, f"{document.identifier} page {page.number} has no regions"
                pages += 1
        assert len(corpus) == 3 and pages == 12

    def test_queries_cover_the_distinguishing_cases(self):
        queries = build_queries(build_corpus())
        # exact hit, right-page/wrong-region, cross-document drift, and a ghost id
        assert len(queries) == 4
        assert any("ghost" in identifier for query in queries for identifier in query.bm25_ranking)


class TestFindings:
    def test_retrieval_recall_does_not_separate_the_profiles(self, outcomes):
        """The premise: on this corpus every profile finds the gold evidence."""
        assert len({outcome.retrieval_recall for outcome in outcomes}) == 1

    def test_citation_quality_does_separate_them(self, outcomes):
        """The finding that justifies the grounding layer."""
        assert len({outcome.top1_exact for outcome in outcomes}) > 1

    def test_hybrid_fusion_cites_at_least_as_well_as_either_component(self, outcomes):
        by_id = {outcome.profile_id: outcome for outcome in outcomes}
        best_single = max(by_id["bm25-k4"].top1_exact, by_id["dense-k4"].top1_exact)
        assert by_id["hybrid-rrf-k4"].top1_exact >= best_single

    def test_a_ghost_identifier_shows_up_as_ungrounded_not_as_a_hit(self, outcomes):
        by_id = {outcome.profile_id: outcome for outcome in outcomes}
        assert by_id["bm25-k4"].ungrounded >= 1

    def test_averaged_accuracy_is_reported_with_its_caveat(self, outcomes):
        """The denominator artifact must be visible in the report, not buried."""
        report = render(outcomes)
        assert "raises the" in report and "top-1 exact" in report


class TestReproducibility:
    def test_two_runs_agree(self):
        assert [outcome.__dict__ for outcome in run()] == [outcome.__dict__ for outcome in run()]
