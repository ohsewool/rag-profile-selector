""""아직 아무것도 하지 않았다"고 말하는 살아 있는 문서가 없어야 한다.

`docs/STATUS.md`와 `docs/RESULTS.md`는 착수 시점 템플릿에서 나왔고, 그때는 사실을
적고 있었다 — 내려받은 것도 실행한 것도 없다고. **그 뒤로 고쳐지지 않았다.**

세 저장소에서 같은 문장이 남아 있었다. `rag-profile-selector/docs/RESULTS.md`는
"내려받은 데이터셋도 모델도 없고 경험적 결과가 없다"고 적혀 있었는데, 그 저장소에는
법령 745조문과 dense 모델 두 종의 비교와 311줄짜리 결과 문서가 있다.

**낡은 것을 넘어 모순이다.** README는 결과를 앞세우고 `docs/`는 결과가 없다고 한다.
그리고 `docs/`는 읽는 사람이 결과를 찾으러 가는 자리다.

착수 시점의 판단(무엇을 의도적으로 미뤘는지)은 지울 이유가 없으므로 `STATUS.md`는
`<!-- historical: -->`로 선언했고, 이름이 "결과"인 `RESULTS.md`는 사실로 고치고
진짜 결과를 가리키게 했다. 선언된 기록은 이 검사에서 제외된다 —
**낡았다는 것이 선언이면 기록이고, 선언이 아니면 사고다.**
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# **선언**과 **언급**은 다르다.
#
# 예전 정규식은 `<!--\s*historical:`을 파일 어디에서든 찾았다. 그래서 이 관례를
# **설명하는 문장**이 있는 문서가 통째로 면제됐다. `modelmate`에서 실제로 그 일이
# 벌어졌다(2026-08-22): README 349줄의 "…은 각자 `<!-- historical: -->`로 선언돼
# 있다"는 한 문장이 **가장 많이 읽히는 문서를 모든 검사에서 빼버렸고**, 두 회차
# 동안 아무도 몰랐다. 다시 넣자마자 가려져 있던 죽은 링크 둘이 나왔다.
#
# 여기 문서들은 지금 그 상태가 아니다 — 재봤고, 언급만으로 면제된 문서는 없다.
# 장치가 같으므로 미리 고친다. 진짜 선언은 **줄 시작에, 문서 앞쪽에** 있다.
HISTORICAL = re.compile(r"^\s*<!--\s*historical:", re.MULTILINE)
DECLARATION_WITHIN_LINES = 15


def declared_historical(text: str) -> bool:
    """문서 앞쪽에 줄 시작으로 놓인 선언만 인정한다."""
    head = "\n".join(text.splitlines()[:DECLARATION_WITHIN_LINES])
    return bool(HISTORICAL.search(head))

SKIP = {".git", "node_modules", "__pycache__", "archive"}

# 착수 시점 템플릿이 남긴 문장들. 원문을 인용하는 것은 이 검사의 일부다 -
# 문장을 조금 바꿔 다시 넣는 것을 잡으려면 정확한 문구를 알아야 한다.
STALE = (
    "The project remains in the specification and planning phase",
    "There are no empirical findings",
    "no retrieval, training, or evaluation run has occurred",
)


def documents() -> list[Path]:
    listed = subprocess.run(["git", "ls-files", "*.md", "**/*.md"], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    names = set(listed.stdout.split()) if listed.returncode == 0 else set()
    return sorted(ROOT / name for name in names
                  if not SKIP & set(Path(name).parts) and (ROOT / name).exists())


def living() -> list[Path]:
    return [path for path in documents()
            if not declared_historical(path.read_text(encoding="utf-8", errors="replace"))]


def test_no_living_document_says_nothing_has_been_done():
    offenders = []
    for path in living():
        text = path.read_text(encoding="utf-8", errors="replace")
        for phrase in STALE:
            # 정정문이 원문을 인용하고 있을 수 있다. 인용은 그 줄에 정정 표시가
            # 함께 있으므로 구분된다 - 없으면 그것은 여전히 주장이다.
            for line in text.splitlines():
                if phrase in line and "사실이 아니" not in line:
                    offenders.append(f"{path.relative_to(ROOT)}: {phrase}")
    assert not offenders, (
        "착수 시점 문장이 살아 있는 문서에 남아 있다:\n  " + "\n  ".join(offenders)
        + "\n사실로 고치거나 <!-- historical: 시점 -->으로 선언하라."
    )


def test_no_living_document_names_a_corpus_this_project_did_not_use():
    """계획은 HotpotQA였고 실제로는 한국어 법령을 썼다.

    `docs/PROJECT_SPEC.md`는 "MVP는 HotpotQA만 쓴다"고 적혀 있었고, HotpotQA는 한 번도
    내려받지 않았다 - 받는 스크립트조차 없다. 여섯 문서에 22번 나왔다. **사양이
    만들어지지 않은 프로젝트를 서술하고 있었다.**

    형제 저장소 `mcp-gateway`의 SPEC은 같은 일을 제대로 해두었다("원래 범위는 합성
    서버로 제한했다. 다음에 한해 해제한다"). 할 줄 알면서 여기서만 안 했다.

    착수 계획은 기록으로 선언했고, 뒤집힘은 `DECISIONS.md`의 D-002 정정에 적었다.
    이 검사는 그 이름이 다시 **살아 있는** 문서로 새어 나오는 것을 막는다.
    """
    planned_but_unused = ("HotpotQA", "SciFact", "QASPER")
    # 결정 로그는 예외다. **폐기된 결정을 보관하는 것이 그 문서의 본분**이고,
    # D-002를 지우면 무엇이 뒤집혔는지가 사라진다. 나머지 문서는 지금을 서술한다.
    offenders = []
    for path in living():
        if path.name == "DECISIONS.md":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name in planned_but_unused:
            for line in text.splitlines():
                # 정정문은 원문을 인용해야 무엇이 뒤집혔는지 읽힌다. 인용은 같은
                # 줄에 정정 표시가 있어 구분된다 - 없으면 그것은 여전히 지시다.
                # (이 예외를 만든 계기: `AGENTS.md`의 D-003 정정을 쓰면서 내가
                # 이 검사에 걸렸다. 걸린 것이 맞았고, 구분이 없던 것이 문제였다.)
                corrected = any(mark in line for mark in
                                ("D-002", "D-003", "쓰지 않", "not HotpotQA",
                                 "never downloaded", "belong to the"))
                if name in line and not corrected:
                    offenders.append(f"{path.relative_to(ROOT)}: {name}")
    assert not offenders, (
        "쓰지 않은 데이터셋을 살아 있는 문서가 말한다:\n  " + "\n  ".join(offenders)
        + "\n계획의 기록이라면 <!-- historical: 시점 -->으로 선언하라."
    )


class TestTheCheckIsNotVacuous:
    def test_it_looked_at_documents(self):
        assert len(living()) >= 3

    def test_something_is_declared_historical(self):
        """선언 분기가 한 번도 쓰이지 않으면 그것은 실행되지 않는 코드다."""
        assert set(documents()) - set(living())

    def test_it_would_catch_the_phrase(self, tmp_path):
        doc = tmp_path / "d.md"
        doc.write_text("None. The project remains in the specification and planning phase.",
                       encoding="utf-8")
        text = doc.read_text(encoding="utf-8")
        assert any(phrase in text for phrase in STALE)
        assert not declared_historical(text)

    def test_a_declared_record_may_still_contain_it(self, tmp_path):
        doc = tmp_path / "d.md"
        doc.write_text("<!-- historical: 2026-06 -->\nThere are no empirical findings.",
                       encoding="utf-8")
        assert declared_historical(doc.read_text(encoding="utf-8"))

    def test_the_unused_corpus_names_are_still_findable_somewhere(self):
        """선언된 기록에서도 사라졌다면 계획이 지워진 것이고, 그것은 이 처리의
        의도가 아니다 - 어디서 갈라졌는지가 남아야 한다."""
        declared = set(documents()) - set(living())
        assert any("HotpotQA" in path.read_text(encoding="utf-8", errors="replace")
                   for path in declared), "계획 기록에서 HotpotQA가 사라졌다"

    def test_the_phrase_list_is_not_empty(self):
        """`STALE = ()`는 모든 문서를 통과시키면서 검사처럼 보인다."""
        assert len(STALE) >= 3


class TestAMentionIsNotADeclaration:
    """관례를 **설명하는 문장**이 문서를 면제시키면 안 된다.

    `modelmate`에서 그 일이 벌어졌다(2026-08-22). 정규식 사본이 파일마다 있으므로
    대조도 파일마다 있어야 한다 — 여기서 느슨하게 되돌려도 다른 파일의 대조는
    아무 말을 하지 않는다.
    """

    def test_a_prose_mention_does_not_exempt(self):
        assert not declared_historical(
            "# 제목\n\n관례는 `<!-- historical: 시점 -->`으로 적는다.\n")

    def test_a_real_declaration_exempts(self):
        assert declared_historical("# 제목\n<!-- historical: 2026-06 -->\n본문\n")

    def test_a_declaration_far_down_does_not_count(self):
        """문서 끝에 붙인 표시는 선언이 아니다 — 읽는 사람은 앞을 보고 판단한다."""
        assert not declared_historical("# 제목\n" + "본문\n" * 40
                                       + "<!-- historical: 2026-06 -->\n")
