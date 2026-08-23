"""문서가 말하는 코퍼스 크기가 실제 코퍼스와 같은가.

`test_published_kr_law_numbers.py`는 표의 **결과 수치**를 실행과 대조한다. 그런데
그 결과가 무엇 위에서 나왔는지를 말하는 숫자 — 법령 14건, 검색 대상 조문 745개,
질의 40건 — 는 README·`STATUS.md`·`RESULTS.md`·`DECISIONS.md`·`TASKS.md`·
`KR_LAW_RESULTS.md` **여섯 문서에 흩어져 있고 아무것도 재지 않았다.**

2026-08-22에 재봤다. 셋 다 맞았다. **빈손이지만 그것을 확인하는 데 값이 있었다** —
745는 조문 전체가 아니라 `is_article`이고 `is_repealed`가 아닌 것만 센 값이고,
파일 안의 조문을 그냥 세면 863이 나온다. 재보지 않고 "863인데 745라고 적혀 있다"고
적었으면 그것이 오보였다.

그래서 이 검사는 **로더를 거쳐서** 센다. 숫자를 만든 정의와 같은 정의로 세지 않으면
같은 것을 세는 것이 아니다.

숫자는 문서에서 읽어온다. 박아두면 문서와 데이터가 갈릴 때 어느 편을 들지 알 수 없다.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from corpus_absence import SKIP_WITHOUT_CORPUS

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "kr_law"
DOCUMENTS = CORPUS / "documents"

pytestmark = SKIP_WITHOUT_CORPUS

# 같은 사실을 말하는 문서들. 한 곳만 고치고 나머지가 남는 것이 이 저장소가
# 이미 겪은 일이다 — README의 산문 숫자가 두 회차 동안 옛 값에 멈춰 있었다.
CLAIMS = (
    ("README.md", r"법령 (\d+)개 조문", "articles"),
    ("README.md", r"질의 (\d+)건으로 답했다", "queries"),
    ("experiments/KR_LAW_RESULTS.md", r"법령 (\d+)건", "laws"),
    ("experiments/KR_LAW_RESULTS.md", r"조문 \*\*(\d+)개\*\*", "articles"),
    ("experiments/KR_LAW_RESULTS.md", r"질의 (\d+)개", "queries"),
    ("docs/STATUS.md", r"(\d+)건\((\d+)조문\)", "laws_and_articles"),
    ("docs/RESULTS.md", r"(\d+)건\((\d+)조문\)", "laws_and_articles"),
    ("docs/DECISIONS.md", r"(\d+)건\((\d+)조문\)", "laws_and_articles"),
)


@pytest.fixture(scope="module")
def measured() -> dict[str, int]:
    """로더를 거쳐 센다. 745는 `is_article`이고 폐지되지 않은 조문의 수이지
    파일 안의 조문 수(863)가 아니다."""
    sys.path.insert(0, str(ROOT))
    from experiments.kr_law_retrieval import load_articles

    queries = json.loads((CORPUS / "queries.json").read_text(encoding="utf-8"))
    if isinstance(queries, dict):
        queries = queries.get("queries", queries)
    return {
        "laws": len(sorted(DOCUMENTS.glob("*.json"))),
        "articles": len(load_articles()),
        "queries": len(queries),
    }


@pytest.mark.parametrize("document,pattern,kind", CLAIMS)
def test_a_published_size_matches_the_corpus(document, pattern, kind, measured):
    text = (ROOT / document).read_text(encoding="utf-8")
    match = re.search(pattern, text)
    assert match, f"{document}에서 {pattern!r}를 찾지 못했다. 문장이 바뀌었으면 여기도 고쳐라."
    if kind == "laws_and_articles":
        assert int(match.group(1)) == measured["laws"], document
        assert int(match.group(2)) == measured["articles"], document
    else:
        assert int(match.group(1)) == measured[kind], document


class TestTheMeasurementIsNotVacuous:
    def test_the_corpus_is_actually_there(self, measured):
        assert measured["laws"] >= 5
        assert measured["articles"] >= 100
        assert measured["queries"] >= 10

    def test_the_loader_filters_something(self, measured):
        """로더를 거치지 않고 그냥 세면 다른 값이 나온다. 같지 않다는 것이
        **로더를 거쳐 세는 이유**이고, 같아지면 이 검사는 정의를 잃는다."""
        raw = sum(len(json.loads(path.read_text(encoding="utf-8"))["articles"])
                  for path in sorted(DOCUMENTS.glob("*.json")))
        assert raw > measured["articles"], (
            f"파일 안 조문 {raw}개와 검색 대상 {measured['articles']}개가 같다 — "
            "폐지 조문이 사라졌거나 필터가 없어졌다."
        )

    def test_every_claim_is_still_findable(self):
        """정규식이 조용히 안 맞게 되는 것이 이 방식의 실패 모드다."""
        for document, pattern, _ in CLAIMS:
            text = (ROOT / document).read_text(encoding="utf-8")
            assert re.search(pattern, text), f"{document}: {pattern}"

    def test_the_claim_list_covers_more_than_one_document(self):
        """한 문서만 보면 나머지가 갈라지는 것을 못 본다 — 그게 이 검사의 이유다."""
        assert len({document for document, _, _ in CLAIMS}) >= 4
