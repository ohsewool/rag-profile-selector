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

    pattern = re.compile(r'Path\(\s*"/[^"]*document-intelligence')
    offenders = [path.name for path in sorted((ROOT / "tests").glob("*.py"))
                 if pattern.search(path.read_text(encoding="utf-8"))]
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
    assert dependent == {"test_grounding.py", "test_citation_experiment.py"}
