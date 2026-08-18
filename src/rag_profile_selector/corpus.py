"""Refuse to run an experiment on data whose provenance is not recorded.

The experiment plan commits to a rule that is easy to state and easy to skip
under time pressure: before any profile executes, the dataset version, evidence
mapping, licence, and checksums are recorded in an immutable manifest, and
validation failures are recorded rather than worked around.

Skipping it is how results become unreproducible without anyone noticing. Six
months later nobody can say which corpus version produced a number, whether the
evidence mapping had gaps, or whether the licence permitted the use at all.

So the loader is deliberately obstructive. A corpus without a complete manifest
does not load; a manifest whose checksum does not match the file does not load;
a split that leaks a query across train and test does not load. There is no
force flag, because the point is that there is no force flag.

Sealed splits are enforced here too. `test` is unreadable until the protocol is
frozen, which makes the plan's "no test result may influence design" rule a
mechanism rather than a promise.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

SPLITS = ("train", "validation", "test")
REQUIRED_MANIFEST_FIELDS = (
    "corpus_id", "version", "source_url", "licence", "retrieved_at",
    "evidence_unit", "checksums",
)


class CorpusError(RuntimeError):
    """Raised when a corpus cannot be trusted enough to run an experiment on."""


class SealedSplitError(CorpusError):
    """Raised when the test split is read before the protocol is frozen."""


def file_checksum(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CorpusManifest:
    """Everything that must be known about a corpus before it is used."""

    corpus_id: str
    version: str
    source_url: str
    licence: str
    retrieved_at: str
    evidence_unit: str
    checksums: Mapping[str, str]
    notes: str = ""

    @classmethod
    def load(cls, path: Path | str) -> "CorpusManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        missing = [field for field in REQUIRED_MANIFEST_FIELDS if not data.get(field)]
        if missing:
            raise CorpusError(
                "manifest is incomplete; refusing to load: " + ", ".join(sorted(missing))
            )
        if not isinstance(data["checksums"], dict) or not data["checksums"]:
            raise CorpusError("manifest must record at least one file checksum")
        return cls(
            corpus_id=data["corpus_id"], version=data["version"],
            source_url=data["source_url"], licence=data["licence"],
            retrieved_at=data["retrieved_at"], evidence_unit=data["evidence_unit"],
            checksums=dict(data["checksums"]), notes=data.get("notes", ""),
        )

    def verify_files(self, root: Path | str) -> None:
        """Every named file must exist and match its recorded checksum."""
        root = Path(root)
        for name, expected in self.checksums.items():
            path = root / name
            if not path.exists():
                raise CorpusError(f"manifest names a file that is missing: {name}")
            actual = file_checksum(path)
            if actual != expected:
                raise CorpusError(
                    f"{name} does not match the manifest checksum "
                    f"(expected {expected[:12]}…, found {actual[:12]}…)"
                )


@dataclass(frozen=True)
class SplitAssignment:
    """Which queries belong to which split, and the seed that decided it."""

    seed: int
    assignments: Mapping[str, str]

    @classmethod
    def load(cls, path: Path | str) -> "SplitAssignment":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "seed" not in data:
            raise CorpusError("split manifest must record the seed that produced it")
        assignments = data.get("assignments") or {}
        if not assignments:
            raise CorpusError("split manifest assigns no queries")
        unknown = sorted({split for split in assignments.values()} - set(SPLITS))
        if unknown:
            raise CorpusError(f"unknown split names: {', '.join(unknown)}")
        return cls(seed=int(data["seed"]), assignments=dict(assignments))

    def of(self, query_id: str) -> str | None:
        return self.assignments.get(query_id)

    def members(self, split: str) -> tuple[str, ...]:
        if split not in SPLITS:
            raise CorpusError(f"unknown split: {split}")
        return tuple(sorted(q for q, s in self.assignments.items() if s == split))


@dataclass
class ValidationReport:
    """What is wrong with a corpus, stated rather than worked around."""

    problems: list[str] = field(default_factory=list)

    def add(self, problem: str) -> None:
        self.problems.append(problem)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        if self.ok:
            return "OK — corpus, evidence mapping, and splits are consistent"
        return "FAILED —\n" + "\n".join(f"  - {problem}" for problem in self.problems)


def validate_evidence_mapping(
    queries: Mapping[str, Iterable[str]],
    known_evidence: Iterable[str],
    splits: SplitAssignment | None = None,
) -> ValidationReport:
    """Check that gold evidence exists, is not empty, and splits do not leak."""
    report = ValidationReport()
    available = set(known_evidence)

    for query_id, evidence in queries.items():
        gold = list(evidence)
        if not gold:
            report.add(f"{query_id}: has no gold evidence")
            continue
        if len(gold) != len(set(gold)):
            report.add(f"{query_id}: gold evidence contains duplicates")
        missing = [item for item in gold if item not in available]
        if missing:
            report.add(
                f"{query_id}: gold evidence not present in the corpus "
                f"({', '.join(sorted(missing)[:3])})"
            )

    if splits is not None:
        unassigned = sorted(set(queries) - set(splits.assignments))
        if unassigned:
            report.add(f"{len(unassigned)} query/queries have no split assignment")
        stray = sorted(set(splits.assignments) - set(queries))
        if stray:
            report.add(f"{len(stray)} split assignment(s) refer to unknown queries")

    return report


class Corpus:
    """A corpus that can only be opened once its provenance checks out."""

    def __init__(self, root: Path | str, *, manifest: CorpusManifest,
                 splits: SplitAssignment, protocol_frozen: bool = False) -> None:
        self.root = Path(root)
        self.manifest = manifest
        self.splits = splits
        self._protocol_frozen = protocol_frozen

    @classmethod
    def open(cls, root: Path | str, *, protocol_frozen: bool = False) -> "Corpus":
        root = Path(root)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise CorpusError(
                f"no manifest at {manifest_path}; a corpus without recorded provenance "
                "cannot be used for an experiment"
            )
        manifest = CorpusManifest.load(manifest_path)
        manifest.verify_files(root)
        splits_path = root / "splits.json"
        if not splits_path.exists():
            raise CorpusError(f"no split manifest at {splits_path}")
        splits = SplitAssignment.load(splits_path)
        return cls(root, manifest=manifest, splits=splits, protocol_frozen=protocol_frozen)

    def queries(self, split: str) -> tuple[str, ...]:
        """Read a split. The test split stays sealed until the protocol is frozen."""
        if split == "test" and not self._protocol_frozen:
            raise SealedSplitError(
                "the test split is sealed until the primary protocol is frozen; "
                "reading it now would let a test result influence the design"
            )
        return self.splits.members(split)

    def freeze_protocol(self) -> "Corpus":
        """Unseal the test split. Deliberately explicit and one-way."""
        return Corpus(self.root, manifest=self.manifest, splits=self.splits,
                      protocol_frozen=True)

    def provenance(self) -> dict[str, Any]:
        """What a results table must carry alongside every number."""
        return {
            "corpus_id": self.manifest.corpus_id,
            "version": self.manifest.version,
            "licence": self.manifest.licence,
            "source_url": self.manifest.source_url,
            "retrieved_at": self.manifest.retrieved_at,
            "evidence_unit": self.manifest.evidence_unit,
            "split_seed": self.splits.seed,
            "split_sizes": {split: len(self.splits.members(split)) for split in SPLITS},
        }
