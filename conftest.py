"""Make both packages importable without environment setup.

`document-intelligence` is a sibling repository while the merge is in progress
(see docs/ADR-001-citation-grounding.md); its tests are skipped when it is absent.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for candidate in (ROOT / "src", ROOT.parent / "document-intelligence" / "src"):
    if candidate.exists():
        sys.path.insert(0, str(candidate))


# ---------------------------------------------------------------------------
# 무엇이 건너뛰어지는지 지킨다.
#
# CI는 `299 passed, 49 skipped`로 초록불이다. **건너뛴 검사는 아무것도 확인하지
# 않은 초록불**이고, 마흔아홉은 새 하나가 숨기에 충분한 수다. 이유가 다른 skip이
# 하나 끼어들어도 총계는 50이 되고 아무도 안 본다.
#
# 그래서 **집합으로 둔다**: 어느 파일이 몇 개를, 어떤 이유로 건너뛰는가. 하한선이
# 아니라 정확한 일치라야 늘어나는 것도, 고쳐서 줄어드는 것도 걸린다.
#
# 전체 스위트를 돌릴 때만 본다. 파일 하나만 돌릴 때 걸리면 그건 검사가 아니라
# 방해다 — 그리고 다음 사람이 이 훅을 지운다.

EXPECTED_SKIPS = {
    "tests/test_kr_law_corpus.py": 9,
    "tests/test_published_corpus_size.py": 12,
    "tests/test_published_kr_law_numbers.py": 12,
    "tests/test_stratification.py": 16,
}
# 이유 문장은 `tests/corpus_absence.py` 한 곳에 있다.
SKIP_REASON = "코퍼스 본문은 gitignore돼 있다"

_skipped: list[tuple[str, str]] = []


def pytest_runtest_logreport(report):
    if report.skipped and report.when in ("setup", "call"):
        reason = ""
        if isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
            reason = report.longrepr[2]
        _skipped.append((report.nodeid.split("::")[0], reason))


def pytest_sessionfinish(session, exitstatus):
    if getattr(session.config.option, "file_or_dir", None):
        return                      # 일부만 돌렸다 — 집합을 비교할 수 없다
    if session.testsfailed:
        return                      # 이미 빨간불이다. 이유를 하나 더 얹지 않는다

    counted: dict[str, int] = {}
    other_reasons = []
    for path, reason in _skipped:
        counted[path] = counted.get(path, 0) + 1
        if SKIP_REASON not in reason:
            other_reasons.append(f"{path}: {reason[:70]}")

    problems = []
    if counted != EXPECTED_SKIPS:
        problems.append(
            "건너뛴 검사의 집합이 다르다.\n"
            f"    실제: {dict(sorted(counted.items()))}\n"
            f"    기대: {dict(sorted(EXPECTED_SKIPS.items()))}\n"
            "  코퍼스를 받아 돌렸다면 전부 0이어야 하고, 그때는 이 훅이 걸린다 —\n"
            "  `KR_LAW_CORPUS_PRESENT=1`을 주면 건너뛴다.")
    if other_reasons:
        problems.append("코퍼스 말고 다른 이유로 건너뛴 것이 있다:\n    "
                        + "\n    ".join(other_reasons))
    if problems and not os.getenv("KR_LAW_CORPUS_PRESENT"):
        raise SystemExit("SKIP 집합 검사 실패 —\n  " + "\n  ".join(problems))
