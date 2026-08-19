"""The fetched Korean corpus, and the manifest that stands in for it.

The documents are not in this repository and are not meant to be. What is
committed is `data/kr_law/manifest.json` - source, licence, retrieval date and a
SHA-256 per file - so anyone can re-fetch and prove they hold the bytes the
numbers were computed on. Redistribution is avoided rather than argued about.

That means most of these tests run against the manifest alone. The ones needing
the documents skip when they are absent, which is the normal state of a fresh
clone.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_profile_selector.corpus import CorpusError, CorpusManifest, file_checksum

CORPUS = ROOT / "data" / "kr_law"
MANIFEST = CORPUS / "manifest.json"


@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST.exists():
        pytest.skip("run scripts/fetch_kr_law_corpus.py first")
    return CorpusManifest.load(MANIFEST)


@pytest.fixture(scope="module")
def documents_present():
    if not (CORPUS / "documents").exists():
        pytest.skip("documents are gitignored; fetch them to run this")
    return True


class TestManifestRecordsProvenance:
    def test_the_licence_is_named(self, manifest):
        """An unnamed licence is the same as no permission."""
        assert "공공누리" in manifest.licence

    def test_the_source_is_a_url_someone_can_check(self, manifest):
        assert manifest.source_url.startswith("https://")

    def test_the_evidence_unit_is_the_article(self, manifest):
        """제N조 is a location a claim can be checked against; a statute is not."""
        assert manifest.evidence_unit == "article"

    def test_a_retrieval_date_is_recorded(self, manifest):
        assert manifest.retrieved_at

    def test_every_document_has_a_checksum(self, manifest):
        assert len(manifest.checksums) > 1
        assert all(len(digest) == 64 for digest in manifest.checksums.values())


class TestTheGateActuallyGates:
    def test_a_manifest_missing_a_licence_does_not_load(self, tmp_path, manifest):
        broken = {
            "corpus_id": manifest.corpus_id, "version": manifest.version,
            "source_url": manifest.source_url, "retrieved_at": manifest.retrieved_at,
            "evidence_unit": manifest.evidence_unit, "checksums": dict(manifest.checksums),
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(CorpusError, match="licence"):
            CorpusManifest.load(path)

    def test_an_altered_file_is_rejected(self, tmp_path):
        """The checksum has to be load-bearing, not decorative."""
        document = tmp_path / "doc.txt"
        document.write_text("제1조 원문", encoding="utf-8")
        recorded = file_checksum(document)
        item = CorpusManifest(
            corpus_id="c", version="1", source_url="https://example.invalid",
            licence="test", retrieved_at="2026-01-01", evidence_unit="article",
            checksums={"doc.txt": recorded},
        )
        item.verify_files(tmp_path)              # unchanged: passes

        document.write_text("제1조 고쳐진 원문", encoding="utf-8")
        with pytest.raises(CorpusError, match="checksum"):
            item.verify_files(tmp_path)

    def test_a_missing_file_is_rejected(self, tmp_path):
        item = CorpusManifest(
            corpus_id="c", version="1", source_url="https://example.invalid",
            licence="test", retrieved_at="2026-01-01", evidence_unit="article",
            checksums={"absent.txt": "0" * 64},
        )
        with pytest.raises(CorpusError, match="missing"):
            item.verify_files(tmp_path)


class TestTheFetchedDocuments:
    def test_the_files_match_their_recorded_checksums(self, manifest, documents_present):
        manifest.verify_files(CORPUS)

    def test_statutes_are_segmented_into_articles(self, manifest, documents_present):
        names = [n for n in manifest.checksums if n.startswith("documents/")]
        payload = json.loads((CORPUS / names[0]).read_text(encoding="utf-8"))
        assert payload["articles"]
        assert all("text" in article for article in payload["articles"])

    def test_articles_carry_their_number(self, manifest, documents_present):
        """Without 조문번호 a citation cannot say where it came from."""
        names = [n for n in manifest.checksums if n.startswith("documents/")]
        numbered = 0
        for name in names:
            payload = json.loads((CORPUS / name).read_text(encoding="utf-8"))
            numbered += sum(1 for a in payload["articles"] if a["article_no"])
        assert numbered > 0

    def test_no_document_is_the_frameset_that_fooled_the_first_parser(
        self, manifest, documents_present
    ):
        """type=HTML returns 3kB of layout. Eleven characters of text per statute
        looked like a short law rather than a broken parser."""
        for name in manifest.checksums:
            if not name.startswith("documents/"):
                continue
            payload = json.loads((CORPUS / name).read_text(encoding="utf-8"))
            body = "".join(a["text"] for a in payload["articles"])
            assert len(body) > 500, f"{name} holds {len(body)} characters"
            assert "<html" not in body.lower()


class TestQuerySet:
    """The queries, and the property that makes them worth running."""

    @pytest.fixture(scope="class")
    def queries(self):
        path = CORPUS / "queries.json"
        if not path.exists():
            pytest.skip("no query set")
        return json.loads(path.read_text(encoding="utf-8"))["queries"]

    def test_every_query_names_its_evidence(self, queries):
        assert queries
        assert all(item["evidence"] for item in queries)

    def test_evidence_ids_are_statute_and_article(self, queries):
        """`270351:7-4` is 제7조의4 of one statute - one article, not a document."""
        for item in queries:
            for gold in item["evidence"]:
                statute, _, article = gold.partition(":")
                assert statute.isdigit() and article

    def test_query_ids_are_unique(self, queries):
        ids = [item["id"] for item in queries]
        assert len(ids) == len(set(ids))

    def test_the_gold_articles_exist(self, queries, manifest, documents_present):
        from rag_profile_selector.corpus import validate_evidence_mapping
        known = set()
        for name in manifest.checksums:
            if not name.startswith("documents/"):
                continue
            payload = json.loads((CORPUS / name).read_text(encoding="utf-8"))
            for article in payload["articles"]:
                if article["is_article"] and not article["is_repealed"]:
                    known.add(f"{payload['mst']}:{article['article_id']}")
        report = validate_evidence_mapping(
            {item["id"]: item["evidence"] for item in queries}, known)
        assert report.ok, report.summary()

    def test_hard_queries_exist(self, queries):
        """A set of only term-matching queries reports that profile choice does
        not matter, which is a fact about the queries rather than retrieval."""
        situational = [q for q in queries if q["difficulty"] == "situational"]
        assert len(situational) >= len(queries) // 5


class TestArticleIdentityIsUnique:
    """제7조 and 제7조의2 are different provisions and must not share an id."""

    def test_no_two_articles_share_an_identifier(self, manifest, documents_present):
        for name in manifest.checksums:
            if not name.startswith("documents/"):
                continue
            payload = json.loads((CORPUS / name).read_text(encoding="utf-8"))
            ids = [a["article_id"] for a in payload["articles"]
                   if a["is_article"] and not a["is_repealed"]]
            assert len(ids) == len(set(ids)), f"{payload['name']} has colliding ids"

    def test_branch_articles_keep_their_branch_number(self, manifest, documents_present):
        """Dropping 조문가지번호 collapsed fourteen provisions onto the id "7"."""
        found = False
        for name in manifest.checksums:
            if not name.startswith("documents/"):
                continue
            payload = json.loads((CORPUS / name).read_text(encoding="utf-8"))
            for article in payload["articles"]:
                if article.get("branch_no"):
                    assert "-" in article["article_id"]
                    assert "의" in article["label"]
                    found = True
        assert found, "no branch article in the corpus; this test proves nothing"


class TestSealedSplit:
    def test_the_test_split_is_sealed_until_the_protocol_is_frozen(self, documents_present):
        from rag_profile_selector.corpus import Corpus, SealedSplitError
        if not (CORPUS / "splits.json").exists():
            pytest.skip("no splits")
        corpus = Corpus.open(CORPUS)
        corpus.queries("train")
        with pytest.raises(SealedSplitError):
            corpus.queries("test")
        assert corpus.freeze_protocol().queries("test")

    def test_the_split_seed_is_recorded(self, documents_present):
        from rag_profile_selector.corpus import Corpus
        if not (CORPUS / "splits.json").exists():
            pytest.skip("no splits")
        assert Corpus.open(CORPUS).provenance()["split_seed"]


class TestRetrievalExperiment:
    """The retrieval components, checked without needing the whole corpus."""

    @pytest.fixture(scope="class")
    def retrieval(self):
        sys.path.insert(0, str(ROOT / "experiments"))
        import kr_law_retrieval
        return kr_law_retrieval

    def test_char_ngrams_survive_korean_particles(self, retrieval):
        """개인정보를 and 개인정보는 are different words and near-identical n-grams.

        This is the whole reason bm25-char is in the comparison.
        """
        first, second = "개인정보를 수집한다", "개인정보는 수집된다"

        # The claim is qualitative: n-grams meet where word tokens cannot. A
        # numeric floor here would be a threshold invented to pass - the first
        # version asserted three and the true overlap is two (개인정, 인정보).
        assert set(retrieval.word_tokens(first)) & set(retrieval.word_tokens(second)) == set()
        assert set(retrieval.char_ngrams(first)) & set(retrieval.char_ngrams(second))

    def test_char_ngrams_ignore_inconsistent_spacing(self, retrieval):
        assert set(retrieval.char_ngrams("한 번")) == set(retrieval.char_ngrams("한번"))

    def test_bm25_ranks_the_matching_document_first(self, retrieval):
        engine = retrieval.BM25({
            "a": retrieval.word_tokens("개인정보 파기 의무"),
            "b": retrieval.word_tokens("전자금융 거래 지시 철회"),
            "c": retrieval.word_tokens("소비자 분쟁 해결 절차"),
        })
        assert engine.search(retrieval.word_tokens("전자금융 철회"), 2)[0] == "b"

    def test_an_unmatched_query_returns_nothing_rather_than_anything(self, retrieval):
        """Returning an arbitrary document for a query with no matching term
        would turn a miss into a wrong citation."""
        engine = retrieval.BM25({"a": retrieval.word_tokens("개인정보 파기")})
        assert engine.search(retrieval.word_tokens("zzzz"), 4) == []

    def test_reciprocal_rank_rewards_position(self, retrieval):
        assert retrieval.reciprocal_rank(["x", "gold"], ["gold"]) == 0.5
        assert retrieval.reciprocal_rank(["gold"], ["gold"]) == 1.0
        assert retrieval.reciprocal_rank(["x", "y"], ["gold"]) == 0.0

    def test_fusion_promotes_what_both_rankings_agree_on(self, retrieval):
        fused = retrieval.reciprocal_rank_fusion([["a", "b"], ["b", "c"]], 3)
        assert fused[0] == "b"
