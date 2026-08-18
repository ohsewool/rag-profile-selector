"""Make both packages importable without environment setup.

`document-intelligence` is a sibling repository while the merge is in progress
(see docs/ADR-001-citation-grounding.md); its tests are skipped when it is absent.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for candidate in (ROOT / "src", ROOT.parent / "document-intelligence" / "src"):
    if candidate.exists():
        sys.path.insert(0, str(candidate))
