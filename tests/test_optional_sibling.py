"""This repository is supposed to work without document-intelligence.

The grounding layer needs the sibling; everything else does not. That was
asserted in the README and in requirements.txt and never run, because every
local environment had the sibling installed and CI installs it on purpose.

The same blind spot in mcp-gateway turned out to hide two real faults - a test
importing its sibling unguarded, and three deciding availability from one
machine's absolute path. Here it turned out clean, and that is worth recording:
an audit that finds nothing is a result, and it is not the same as an audit
nobody ran.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_no_test_decides_availability_from_a_filesystem_path():
    """Availability is "does it import", not "is there a directory".

    A path check answers a question about one machine's layout and gets it wrong
    in both directions: elsewhere the directory is missing and the test skips
    silently, and on the original machine with the package uninstalled the
    directory is there while the import still fails.
    """
    import re

    # 예전 패턴은 `Path("/...형제이름` 하나만 봤다. 좁게 잡은 이유는 기록돼 있었고
    # 맞는 걱정이었다 — 첫 판이 `Path(__file__)`과 이 파일의 산문까지 잡았다.
    #
    # 그런데 좁힌 결과가 **한 가지 철자만** 잡는 상태였다. 2026-08-22에 재봤더니
    # 이 셋이 전부 통과했다:
    #
    #     os.path.exists("/home/jovyan/work/document-intelligence")
    #     SIBLING = "/home/jovyan/work/document-intelligence"
    #     from pathlib import Path as P; P("/home/jovyan/work/document-intelligence")
    #
    # 셋 다 이 검사가 막으려는 바로 그 실수다. 이제 **주석과 독스트링을 걷어낸 뒤**
    # 기계 고유 경로가 문자열 리터럴로 들어 있는지 본다 — 걷어내기가 `Path(__file__)`
    # 오탐과 산문 인용을 함께 없애므로, 넓히면서 정밀도를 잃지 않는다.
    #
    # 인용과 사용을 구분하는 이 방식은 이 저장소들이 이미 여러 번 쓴 것이다.
    without_strings_of_prose = re.compile(r'("""|\'\'\')(?:.|\n)*?\1')
    pattern = re.compile(r'["\'][^"\']*/home/[^"\']*document-intelligence')
    offenders = []
    for path in sorted((ROOT / "tests").glob("*.py")):
        text = without_strings_of_prose.sub('""', path.read_text(encoding="utf-8"))
        text = "\n".join(re.sub(r"#.*$", "", line) for line in text.splitlines())
        if pattern.search(text):
            offenders.append(path.name)
    assert not offenders, f"hardcoded sibling paths in {offenders}"


@pytest.mark.slow
def test_the_suite_skips_rather_than_errors_without_the_sibling():
    """Run the suite in a subprocess with the sibling genuinely unimportable.

    ModuleNotFoundError rather than a bare ImportError, because that is what
    Python raises for a missing module and what importorskip is written to
    catch. Simulating it with a plain ImportError produces collection errors
    that are an artefact of the harness - a mistake made once already while
    checking mcp-gateway, and nearly reported as a finding.
    """
    script = textwrap.dedent('''
        import sys
        from importlib.abc import MetaPathFinder

        class Absent(MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "document_intelligence" or name.startswith("document_intelligence."):
                    raise ModuleNotFoundError(f"No module named {name!r}", name=name)
                return None

        sys.meta_path.insert(0, Absent())
        import pytest
        sys.exit(pytest.main(["tests/", "-q", "-p", "no:cacheprovider",
                              "-m", "not slow"]))
    ''')
    finished = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                              capture_output=True, text=True, timeout=600)
    assert finished.returncode == 0, finished.stdout[-2000:]
    assert "skipped" in finished.stdout, "nothing skipped; the sibling was still reachable"


def test_only_the_grounding_tests_depend_on_it():
    """If that list grows, the dependency has spread and someone should know.

    Named rather than counted, so a file appearing here is visible in the diff
    instead of arriving as a number that moved.
    """
    dependent = {
        path.name for path in sorted((ROOT / "tests").glob("*.py"))
        if "document_intelligence" in path.read_text(encoding="utf-8")
        and path.name != Path(__file__).name
    }
    # `test_rejections_that_were_never_fired.py`는 다른 이유로 이 이름을 든다:
    # **형제가 없을 때 나는 ImportError**를 하위 프로세스에서 확인한다. 형제를 쓰는
    # 것이 아니라 없을 때의 안내를 확인하는 것이므로 형제가 있든 없든 돈다. 목록에
    # 넣는 이유는 이 검사가 "이름을 드는 파일"을 세기 때문이고, 그 방식이 맞다 —
    # 2026-08-22에 이 가드가 그 파일을 정확히 잡아냈다.
    assert dependent == {"test_grounding.py", "test_citation_experiment.py",
                         "test_rejections_that_were_never_fired.py"}
