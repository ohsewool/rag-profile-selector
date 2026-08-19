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
