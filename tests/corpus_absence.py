"""코퍼스가 없을 때 건너뛴다 — **한 곳에 적힌 조건과 한 문장.**

네 파일이 같은 사실을 말하고 있었다. 셋은 한국어로 똑같이, 하나는 영어로 다르게.

    tests/test_published_corpus_size.py      "코퍼스 본문은 gitignore돼 있다 — 받아야 돈다"
    tests/test_published_kr_law_numbers.py   같은 문장
    tests/test_stratification.py             같은 문장
    tests/test_kr_law_corpus.py              "documents are gitignored; fetch them to run this"

`test_published_corpus_size.py`는 바로 그 아래에 **"같은 사실을 말하는 문서들. 한 곳만
고치고 나머지가 남는 것이 이 저장소가 이미 겪은 일이다"**라고 적어두고 있었다. 문서에
대해서는 그 규칙을 세웠고, **건너뛰는 이유 자체는 네 번 적혀 있었다.**

2026-08-23에 건너뛴 검사의 집합을 지키는 훅을 쓰다 나왔다. 훅이 "이유가 다른 skip이
있다"고 걸었는데, 실제로 다른 것은 사실이 아니라 문장이었다.

    CI:  299 passed, 49 skipped
    로컬(코퍼스 있음): 348 passed

**건너뛴 검사는 아무것도 확인하지 않은 초록불이다.** 마흔아홉은 새 하나가 숨기에
충분한 수이고, 그래서 이 조건과 문장을 여기 한 곳에 둔다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "kr_law"
DOCUMENTS = CORPUS / "documents"

REASON = "코퍼스 본문은 gitignore돼 있다 — 받아야 돈다"


def documents_present() -> bool:
    """조문 JSON이 실제로 있는가. **디렉터리만 보지 않는다** — `fetch`가 중간에
    죽으면 빈 디렉터리가 남고, 그때 "있다"고 하면 검사가 빈손으로 돈다."""
    return DOCUMENTS.exists() and any(DOCUMENTS.glob("*.json"))


#: 모듈 맨 위에 `pytestmark = SKIP_WITHOUT_CORPUS`로 붙인다.
SKIP_WITHOUT_CORPUS = pytest.mark.skipif(not documents_present(), reason=REASON)


def skip_without_corpus() -> None:
    """함수 안에서 부르는 쪽. 데코레이터를 못 붙이는 자리에 쓴다."""
    if not documents_present():
        pytest.skip(REASON)
