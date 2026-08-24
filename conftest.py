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



def _collected_every_test_file(session) -> bool:
    """이번 실행이 **디스크의 테스트 파일을 전부** 모았는가.

    부르는 방법은 여럿이다(`pytest`, `pytest tests/`, `pytest tests/ -q`). 셋 다
    같은 것을 돌린다. 반면 `pytest tests/test_one.py`는 아니다. 인자의 모양이
    아니라 **모인 결과**로 가른다.
    """
    from pathlib import Path as _Path

    root = _Path(str(session.config.rootpath)) / "tests"
    if not root.is_dir():
        return False
    on_disk = {p.resolve() for p in root.rglob("test_*.py")}
    if not on_disk:
        return False
    collected = set()
    for item in getattr(session, "items", []):
        try:
            collected.add(_Path(str(item.fspath)).resolve())
        except Exception:
            pass
    return on_disk <= collected

def pytest_sessionfinish(session, exitstatus):
    """**돌린 것이 있을 때만 판정한다.**

    처음엔 이 두 줄이 없었고 CI가 잡았다. 이 저장소에는 `pytest --collect-only`로
    "스위트가 조용히 줄지 않았는가"를 보는 단계가 있는데, 거기서는 **아무것도 안
    돈다** — 건너뛴 것도 0이다. 훅은 그것을 "기대 집합과 다르다"로 읽고 세션을
    죽였고, 그래서 `N tests collected` 줄이 아예 안 찍혔다.

    세션 훅은 **내가 생각하지 않은 모드에서도 돈다.** collect-only, `-k` 선택,
    파일 하나. 판정할 근거가 없는 실행에서 걸리면 그건 검사가 아니라 방해다.
    """
    if getattr(session.config.option, "collectonly", False):
        return                      # 아무것도 안 돌았다
    if not _collected_every_test_file(session):
        # **"경로를 줬다"와 "일부만 돌렸다"는 다르다.**
        #
        # 예전에는 `file_or_dir`가 있으면 무조건 건너뛰었다. 그래서 README가
        # 시키는 `pytest tests/ -q`에서 이 훅이 통째로 침묵했다 — CI는 인자 없이
        # `pytest -q`를 쓰므로 거기서만 살아 있었고, **README를 따르는 사람에게는
        # 없는 것과 같았다.** 2026-08-24에 새 clone에서 재보다 걸렸다:
        # 33개가 건너뛰어졌는데 종료 코드가 0이었다.
        #
        # 이제 "무엇을 줬는가"가 아니라 **"전부 모였는가"**를 본다. 디스크의
        # `test_*.py`가 전부 수집됐으면 어떻게 불렀든 판정할 근거가 있다.
        return
    if getattr(session.config.option, "keyword", None):
        return                      # `-k`로 골랐다
    if not getattr(session, "testscollected", 0):
        return
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
