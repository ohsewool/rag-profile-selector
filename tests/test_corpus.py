"""Provenance gates: the checks that keep results reproducible six months later."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_profile_selector.corpus import (
    Corpus,
    CorpusError,
    CorpusManifest,
    SealedSplitError,
    SplitAssignment,
    file_checksum,
    validate_evidence_mapping,
)

MANIFEST = {
    "corpus_id": "kr-public-docs",
    "version": "2026.08",
    "source_url": "https://example.test/corpus",
    "licence": "CC BY 4.0",
    "retrieved_at": "2026-08-19",
    "evidence_unit": "region",
    "checksums": {},
}

SPLITS = {"seed": 42, "assignments": {"q1": "train", "q2": "validation", "q3": "test"}}


@pytest.fixture
def corpus_root(tmp_path):
    documents = tmp_path / "documents.jsonl"
    documents.write_text('{"id": "d1"}\n', encoding="utf-8")
    manifest = dict(MANIFEST, checksums={"documents.jsonl": file_checksum(documents)})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "splits.json").write_text(json.dumps(SPLITS), encoding="utf-8")
    return tmp_path


class TestManifestCompleteness:
    def test_a_complete_manifest_loads(self, corpus_root):
        manifest = CorpusManifest.load(corpus_root / "manifest.json")
        assert manifest.corpus_id == "kr-public-docs"
        assert manifest.licence == "CC BY 4.0"

    @pytest.mark.parametrize("field", [
        "corpus_id", "version", "source_url", "licence", "retrieved_at", "evidence_unit",
    ])
    def test_every_required_field_is_actually_required(self, corpus_root, field):
        data = json.loads((corpus_root / "manifest.json").read_text(encoding="utf-8"))
        data.pop(field)
        (corpus_root / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(CorpusError) as error:
            CorpusManifest.load(corpus_root / "manifest.json")
        assert field in str(error.value)

    def test_a_licence_may_not_be_left_blank(self, corpus_root):
        """Unknown licensing is a decision, not a default."""
        data = json.loads((corpus_root / "manifest.json").read_text(encoding="utf-8"))
        data["licence"] = ""
        (corpus_root / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(CorpusError):
            CorpusManifest.load(corpus_root / "manifest.json")

    def test_a_manifest_without_checksums_is_refused(self, corpus_root):
        data = json.loads((corpus_root / "manifest.json").read_text(encoding="utf-8"))
        data["checksums"] = {}
        (corpus_root / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(CorpusError):
            CorpusManifest.load(corpus_root / "manifest.json")


class TestChecksumVerification:
    def test_matching_files_verify(self, corpus_root):
        CorpusManifest.load(corpus_root / "manifest.json").verify_files(corpus_root)

    def test_an_edited_corpus_file_is_caught(self, corpus_root):
        """The silent failure this exists for: data changed, version string did not."""
        (corpus_root / "documents.jsonl").write_text('{"id": "d2"}\n', encoding="utf-8")
        with pytest.raises(CorpusError) as error:
            CorpusManifest.load(corpus_root / "manifest.json").verify_files(corpus_root)
        assert "does not match" in str(error.value)

    def test_a_missing_corpus_file_is_caught(self, corpus_root):
        (corpus_root / "documents.jsonl").unlink()
        with pytest.raises(CorpusError):
            CorpusManifest.load(corpus_root / "manifest.json").verify_files(corpus_root)


class TestSplits:
    def test_a_split_manifest_records_its_seed(self, corpus_root):
        assert SplitAssignment.load(corpus_root / "splits.json").seed == 42

    def test_a_split_without_a_seed_is_refused(self, tmp_path):
        """Without the seed the split cannot be reproduced, so it is not a split."""
        path = tmp_path / "splits.json"
        path.write_text(json.dumps({"assignments": {"q1": "train"}}), encoding="utf-8")
        with pytest.raises(CorpusError):
            SplitAssignment.load(path)

    def test_unknown_split_names_are_refused(self, tmp_path):
        path = tmp_path / "splits.json"
        path.write_text(json.dumps({"seed": 1, "assignments": {"q1": "holdout"}}),
                        encoding="utf-8")
        with pytest.raises(CorpusError):
            SplitAssignment.load(path)

    def test_members_are_returned_per_split(self, corpus_root):
        splits = SplitAssignment.load(corpus_root / "splits.json")
        assert splits.members("train") == ("q1",)
        assert splits.of("q2") == "validation"


class TestEvidenceMapping:
    def test_a_consistent_mapping_passes(self):
        report = validate_evidence_mapping({"q1": ["e1", "e2"]}, ["e1", "e2", "e3"])
        assert report.ok

    def test_a_query_with_no_gold_evidence_is_reported(self):
        report = validate_evidence_mapping({"q1": []}, ["e1"])
        assert not report.ok
        assert "no gold evidence" in report.summary()

    def test_gold_evidence_missing_from_the_corpus_is_reported(self):
        """A label pointing at a document that is not there silently deflates recall."""
        report = validate_evidence_mapping({"q1": ["e1", "ghost"]}, ["e1"])
        assert not report.ok
        assert "not present" in report.summary()

    def test_duplicate_gold_evidence_is_reported(self):
        report = validate_evidence_mapping({"q1": ["e1", "e1"]}, ["e1"])
        assert not report.ok

    def test_unassigned_queries_are_reported(self, corpus_root):
        splits = SplitAssignment.load(corpus_root / "splits.json")
        report = validate_evidence_mapping({"q1": ["e1"], "q9": ["e1"]}, ["e1"], splits)
        assert not report.ok
        assert "no split assignment" in report.summary()

    def test_stray_split_assignments_are_reported(self, corpus_root):
        splits = SplitAssignment.load(corpus_root / "splits.json")
        report = validate_evidence_mapping({"q1": ["e1"]}, ["e1"], splits)
        assert not report.ok
        assert "unknown queries" in report.summary()

    def test_problems_accumulate_rather_than_stopping_at_the_first(self):
        report = validate_evidence_mapping({"q1": [], "q2": ["ghost"]}, ["e1"])
        assert len(report.problems) == 2


class TestOpeningACorpus:
    def test_a_valid_corpus_opens(self, corpus_root):
        corpus = Corpus.open(corpus_root)
        assert corpus.manifest.corpus_id == "kr-public-docs"

    def test_a_corpus_without_a_manifest_does_not_open(self, tmp_path):
        (tmp_path / "documents.jsonl").write_text("{}\n", encoding="utf-8")
        with pytest.raises(CorpusError) as error:
            Corpus.open(tmp_path)
        assert "provenance" in str(error.value)

    def test_a_corpus_without_splits_does_not_open(self, corpus_root):
        (corpus_root / "splits.json").unlink()
        with pytest.raises(CorpusError):
            Corpus.open(corpus_root)

    def test_opening_verifies_checksums(self, corpus_root):
        (corpus_root / "documents.jsonl").write_text("tampered\n", encoding="utf-8")
        with pytest.raises(CorpusError):
            Corpus.open(corpus_root)


class TestSealedTestSplit:
    def test_train_and_validation_are_readable(self, corpus_root):
        corpus = Corpus.open(corpus_root)
        assert corpus.queries("train") == ("q1",)
        assert corpus.queries("validation") == ("q2",)

    def test_the_test_split_is_sealed_until_the_protocol_is_frozen(self, corpus_root):
        """Makes 'no test result may influence design' a mechanism, not a promise."""
        corpus = Corpus.open(corpus_root)
        with pytest.raises(SealedSplitError):
            corpus.queries("test")

    def test_freezing_the_protocol_unseals_it(self, corpus_root):
        corpus = Corpus.open(corpus_root).freeze_protocol()
        assert corpus.queries("test") == ("q3",)

    def test_freezing_returns_a_new_corpus_leaving_the_original_sealed(self, corpus_root):
        corpus = Corpus.open(corpus_root)
        corpus.freeze_protocol()
        with pytest.raises(SealedSplitError):
            corpus.queries("test")


class TestProvenanceRecord:
    def test_provenance_carries_what_a_results_table_needs(self, corpus_root):
        provenance = Corpus.open(corpus_root).provenance()
        assert provenance["version"] == "2026.08"
        assert provenance["licence"] == "CC BY 4.0"
        assert provenance["split_seed"] == 42
        assert provenance["split_sizes"] == {"train": 1, "validation": 1, "test": 1}

    def test_provenance_is_json_serialisable(self, corpus_root):
        json.dumps(Corpus.open(corpus_root).provenance())
