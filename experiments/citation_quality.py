"""Does choosing a retrieval profile change *citation* quality, or only recall?

Profile selection is normally judged on retrieval quality. If citation quality
moved with it, measuring retrieval would be enough and the grounding layer would
be redundant. This experiment exists to find out, and it is built so that the
answer can come back "no difference" — that would be a real finding too.

The corpus is synthetic and says so. It is not evidence about real documents; it
is a controlled setting where the two kinds of quality can be varied
independently, so the *relationship* between them is observable at all. Numbers
from a real corpus would say something about the world; these say something
about the measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "document-intelligence" / "src"))

from document_intelligence import BoundingBox, Document, Page, TableRegion, TextRegion  # noqa: E402

from rag_profile_selector.evidence_metrics import calculate_evidence_metrics  # noqa: E402
from rag_profile_selector.fusion import fuse_reciprocal_rank  # noqa: E402
from rag_profile_selector.grounding import (  # noqa: E402
    GroundingIndex,
    measure_citation_accuracy,
)
from rag_profile_selector.profiles import RetrievalMethod, RetrievalProfile  # noqa: E402

PAGES = 4
REGIONS_PER_PAGE = 3


def build_corpus(document_count: int = 3) -> tuple[Document, ...]:
    """A small synthetic corpus with region-level structure on every page."""
    documents = []
    for index in range(document_count):
        pages = []
        for page_number in range(1, PAGES + 1):
            regions = []
            for slot in range(REGIONS_PER_PAGE):
                top = 0.05 + slot * 0.3
                identifier = f"p{page_number}-r{slot}"
                region_class = TableRegion if slot == REGIONS_PER_PAGE - 1 else TextRegion
                regions.append(region_class(
                    identifier=identifier,
                    bounding_box=BoundingBox(left=0.1, top=top, right=0.9, bottom=top + 0.2),
                ))
            pages.append(Page(number=page_number, width=612.0, height=792.0,
                              regions=tuple(regions)))
        documents.append(Document(identifier=f"doc{index}", checksum=f"{index}" * 64,
                                  pages=tuple(pages)))
    return tuple(documents)


@dataclass(frozen=True)
class Query:
    """One query with the identifier that answers it and where it really lives."""

    query_id: str
    gold_identifier: str
    gold_page: int
    gold_region: str | None
    bm25_ranking: tuple[str, ...]
    dense_ranking: tuple[str, ...]


def build_queries(documents: Sequence[Document]) -> tuple[Query, ...]:
    """Rankings are fixtures, not a retriever.

    Each query fixes what BM25 and dense retrieval would return, chosen to cover
    the cases the grounding layer distinguishes: exact hits, right-page/wrong-
    region hits, and identifiers that resolve nowhere.
    """
    document = documents[0].identifier
    other = documents[1].identifier

    def identifier(doc: str, page: int, slot: int) -> str:
        return f"{doc}:{page}:p{page}-r{slot}"

    return (
        # Both retrievers find the exact region.
        Query("q1", identifier(document, 1, 0), 1, "p1-r0",
              bm25_ranking=(identifier(document, 1, 0), identifier(document, 1, 1)),
              dense_ranking=(identifier(document, 1, 0), identifier(document, 2, 0))),
        # BM25 lands on the right page but the wrong region; dense is exact.
        Query("q2", identifier(document, 2, 1), 2, "p2-r1",
              bm25_ranking=(identifier(document, 2, 0), identifier(document, 2, 1)),
              dense_ranking=(identifier(document, 2, 1), identifier(document, 3, 0))),
        # Dense drifts to another document; BM25 is exact.
        Query("q3", identifier(document, 3, 2), 3, "p3-r2",
              bm25_ranking=(identifier(document, 3, 2), identifier(document, 3, 0)),
              dense_ranking=(identifier(other, 3, 2), identifier(document, 3, 2))),
        # An identifier that does not exist in the corpus at all.
        Query("q4", identifier(document, 4, 0), 4, "p4-r0",
              bm25_ranking=(f"{document}:9:ghost", identifier(document, 4, 0)),
              dense_ranking=(identifier(document, 4, 0), identifier(document, 4, 1))),
    )


def rank_for(profile: RetrievalProfile, query: Query) -> tuple[str, ...]:
    if profile.method is RetrievalMethod.BM25:
        return query.bm25_ranking
    if profile.method is RetrievalMethod.DENSE:
        return query.dense_ranking
    fused = fuse_reciprocal_rank([list(query.bm25_ranking), list(query.dense_ranking)],
                                 k=profile.k)
    return tuple(item.identifier for item in fused)


@dataclass
class ProfileOutcome:
    profile_id: str
    retrieval_precision: float
    retrieval_recall: float
    grounding_rate: float
    page_accuracy: float
    region_accuracy: float
    misplaced: int
    coarse: int
    ungrounded: int
    top1_exact: float      # share of queries whose *first* citation points exactly right


def evaluate(profile: RetrievalProfile, queries: Sequence[Query],
             index: GroundingIndex, *, cutoff: int = 2) -> ProfileOutcome:
    precisions: list[float] = []
    recalls: list[float] = []
    grounding_rates: list[float] = []
    page_scores: list[float] = []
    region_scores: list[float] = []
    misplaced = coarse = ungrounded = 0
    top1_hits = 0

    for query in queries:
        ranked = rank_for(profile, query)[:cutoff]
        retrieval = calculate_evidence_metrics(ranked, [query.gold_identifier])
        precisions.append(retrieval.precision)
        recalls.append(retrieval.recall)

        result = index.ground(ranked)
        gold: Mapping[str, tuple[int, str | None]] = {
            query.gold_identifier: (query.gold_page, query.gold_region)
        }
        accuracy = measure_citation_accuracy(result, gold)
        grounding_rates.append(accuracy.grounding_rate)
        page_scores.append(accuracy.page_accuracy)
        region_scores.append(accuracy.region_accuracy)
        misplaced += accuracy.misplaced
        coarse += accuracy.coarse
        ungrounded += accuracy.ungrounded

        # Averaging over every returned citation is distorted by the denominator:
        # an ungrounded result is excluded and so *raises* the average, and with
        # one gold per query the ceiling is 1/cutoff. The first citation is what a
        # reader actually follows, so it is scored separately and without that bias.
        first = result.grounded[0] if result.grounded else None
        if (first is not None
                and first.identifier == query.gold_identifier
                and first.page_number == query.gold_page
                and first.region_identifier == query.gold_region):
            top1_hits += 1

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return ProfileOutcome(
        profile_id=profile.profile_id,
        retrieval_precision=mean(precisions),
        retrieval_recall=mean(recalls),
        grounding_rate=mean(grounding_rates),
        page_accuracy=mean(page_scores),
        region_accuracy=mean(region_scores),
        misplaced=misplaced,
        coarse=coarse,
        ungrounded=ungrounded,
        top1_exact=round(top1_hits / len(queries), 4) if queries else 0.0,
    )


def run() -> list[ProfileOutcome]:
    documents = build_corpus()
    queries = build_queries(documents)
    index = GroundingIndex.from_documents(documents)
    profiles = (
        RetrievalProfile(RetrievalMethod.BM25, 4),
        RetrievalProfile(RetrievalMethod.DENSE, 4),
        RetrievalProfile(RetrievalMethod.HYBRID_RRF, 4),
        RetrievalProfile(RetrievalMethod.HYBRID_RRF, 8),
    )
    return [evaluate(profile, queries, index) for profile in profiles]


def render(outcomes: Sequence[ProfileOutcome]) -> str:
    lines = [
        "# Retrieval profile vs citation quality (synthetic corpus)",
        "",
        "Synthetic fixtures, not real documents: this measures whether the two kinds",
        "of quality move together, not how any retriever performs in the world.",
        "",
        "| profile | precision | recall | grounding | top-1 exact | page acc.\* | region acc.\* | misplaced | ungrounded |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for outcome in outcomes:
        lines.append(
            f"| {outcome.profile_id} | {outcome.retrieval_precision} | {outcome.retrieval_recall} | "
            f"{outcome.grounding_rate} | {outcome.top1_exact} | {outcome.page_accuracy} | "
            f"{outcome.region_accuracy} | {outcome.misplaced} | {outcome.ungrounded} |"
        )

    lines += ["",
              r"\* averaged over every returned citation, so an ungrounded result raises the",
              "  average by leaving the denominator; read `top-1 exact` for the unbiased view.",
              ""]
    retrieval_spread = _spread(o.retrieval_recall for o in outcomes)
    citation_spread = _spread(o.top1_exact for o in outcomes)
    lines += ["", "## Reading", ""]
    lines.append(f"- retrieval recall spread across profiles: {retrieval_spread}")
    lines.append(f"- citation top-1 exact spread across profiles: {citation_spread}")
    if citation_spread > retrieval_spread:
        lines.append("- citation quality separates the profiles **more** than retrieval quality does,")
        lines.append("  so a selector tuned on recall alone would be choosing blind on this axis.")
    elif citation_spread == 0:
        lines.append("- citation quality does not separate these profiles at all: on this corpus")
        lines.append("  the grounding layer adds no signal for profile selection.")
    else:
        lines.append("- citation quality separates the profiles less than retrieval quality does;")
        lines.append("  the axes are related but not identical on this corpus.")
    return "\n".join(lines)


def _spread(values) -> float:
    materialised = list(values)
    return round(max(materialised) - min(materialised), 4) if materialised else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args(argv)

    outcomes = run()
    if arguments.json:
        print(json.dumps([asdict(outcome) for outcome in outcomes], indent=2))
        return 0
    report = render(outcomes)
    if arguments.out:
        arguments.out.write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
