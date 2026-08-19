#!/usr/bin/env python3
"""Fetch Korean statutes from 국가법령정보 OPEN API into a provenanced corpus.

Why statutes. The project needs Korean documents whose licence is unambiguous
and whose structure is worth retrieving over. Korean law is published by 법제처
under 공공누리 제1유형 - free to use, redistribute and modify with attribution -
and it is written in a numbered hierarchy (조/항/호) that gives citation a real
target. "Article 31 paragraph 1" is a location a claim can be checked against,
which is exactly what the grounding layer measures.

Nothing is committed to the repository. The corpus directory is gitignored and
what gets versioned is `manifest.json`: source URL, licence, retrieval date and
a SHA-256 per file. Anyone can re-fetch and verify they hold the same bytes
this repo's numbers were computed on. Redistribution is not the point, and not
redistributing avoids the question entirely.

The API is unauthenticated for the trial identifier `OC=test`, which is
sufficient here; pass --oc to use your own.

    python3 scripts/fetch_kr_law_corpus.py --out data/kr_law --limit 30

Re-running is safe: files are rewritten and the manifest recomputed, so a
changed statute shows up as a checksum change rather than silently sliding into
an old experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://www.law.go.kr/DRF"
LICENCE = "공공누리 제1유형 (출처표시) — 법제처 국가법령정보센터"
SOURCE = "https://www.law.go.kr/DRF/lawSearch.do"

# Topics chosen for coverage rather than convenience: each is a distinct domain
# with its own vocabulary, so a retrieval profile that suits one need not suit
# the next - which is the question this project exists to ask.
TOPICS = ("개인정보", "인공지능", "전자금융", "정보통신", "소비자", "근로", "공공기관")

_CONTEXT = ssl.create_default_context()
_CONTEXT.check_hostname = False
_CONTEXT.verify_mode = ssl.CERT_NONE


class FetchError(RuntimeError):
    """The source did not answer in a form worth writing to disk."""


def _get(path: str, params: dict[str, str], *, retries: int = 3) -> str:
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(params, encoding="utf-8")
    request = urllib.request.Request(url, headers={"User-Agent": "rag-profile-selector/0.1"})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30, context=_CONTEXT) as response:
                return response.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError) as error:
            last = error
            time.sleep(1.5 * (attempt + 1))
    raise FetchError(f"{url}: {last}")


def search(query: str, display: int) -> list[dict]:
    raw = _get("lawSearch.do", {"OC": OC, "target": "law", "type": "JSON",
                                "query": query, "display": str(display)})
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise FetchError(f"search for {query!r} returned non-JSON: {raw[:120]}") from error
    found = payload.get("LawSearch", {}).get("law", [])
    return found if isinstance(found, list) else [found]


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def articles(mst: str) -> list[dict]:
    """Statute as a list of articles, each keeping its 조문번호.

    The JSON endpoint rather than the HTML one. `type=HTML` returns a frameset -
    three kilobytes of layout and no statute at all - so the first version of
    this script stripped the tags, got eleven characters per document, and would
    have written a corpus of empty files if the length floor had not rejected
    every single one. That floor exists to skip statutes too short to retrieve
    over; it caught a broken parser instead.

    JSON also arrives already segmented by article, which is the better outcome:
    the evidence unit becomes 제N조 rather than a whole statute, so a citation
    points at a provision instead of a document.
    """
    raw = _get("lawService.do", {"OC": OC, "target": "law", "MST": mst, "type": "JSON"})
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise FetchError(f"MST {mst} returned non-JSON: {raw[:120]}") from error

    collected: list[dict] = []
    for unit in _as_list(payload.get("법령", {}).get("조문", {}).get("조문단위")):
        parts = [str(unit.get("조문내용") or "")]
        # 항 hang off the article when it has them. Without descending, every
        # multi-paragraph provision would be stored as its heading alone.
        for paragraph in _as_list(unit.get("항")):
            parts.append(str(paragraph.get("항내용") or ""))
            for clause in _as_list(paragraph.get("호")):
                parts.append(str(clause.get("호내용") or ""))
        text = re.sub(r"[^\S\n]+", " ", "\n".join(parts))
        text = re.sub(r"\n\s*\n+", "\n", text).strip()
        if not text:
            continue
        collected.append({
            "article_no": str(unit.get("조문번호") or "").strip(),
            "is_article": str(unit.get("조문여부") or "") == "조문",
            "text": text,
        })
    return collected


def slugify(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣]+", "_", name).strip("_")
    return cleaned[:60] or "statute"


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    global OC
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/kr_law")
    parser.add_argument("--limit", type=int, default=30, help="statutes in total")
    parser.add_argument("--oc", default="test", help="국가법령정보 OPEN API identifier")
    parser.add_argument("--min-chars", type=int, default=800,
                        help="skip statutes too short to retrieve over")
    args = parser.parse_args()
    OC = args.oc

    out = Path(args.out)
    (out / "documents").mkdir(parents=True, exist_ok=True)

    per_topic = max(1, args.limit // len(TOPICS) + 1)
    seen: set[str] = set()
    written: list[dict] = []

    for topic in TOPICS:
        if len(written) >= args.limit:
            break
        try:
            results = search(topic, per_topic)
        except FetchError as error:
            print(f"  ! {topic}: {error}", file=sys.stderr)
            continue
        for item in results:
            if len(written) >= args.limit:
                break
            mst = str(item.get("법령일련번호") or "")
            name = str(item.get("법령명한글") or "").strip()
            if not mst or mst in seen or not name:
                continue
            seen.add(mst)
            try:
                provisions = articles(mst)
            except FetchError as error:
                print(f"  ! {name}: {error}", file=sys.stderr)
                continue
            body = "\n\n".join(item["text"] for item in provisions)
            if len(body) < args.min_chars:
                print(f"  - {name}: {len(body)}자, 너무 짧아 건너뜀")
                continue
            filename = f"documents/{mst}_{slugify(name)}.json"
            (out / filename).write_text(json.dumps({
                "mst": mst, "name": name, "articles": provisions,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            written.append({
                "file": filename, "mst": mst, "name": name, "topic": topic,
                "ministry": item.get("소관부처명", ""),
                "kind": item.get("법령구분명", ""),
                "articles": len(provisions),
                "chars": len(body),
            })
            print(f"  + {name} ({len(provisions)}개 조문 / {len(body):,}자)")
            time.sleep(0.4)   # the source is a public service, not a load target

    if not written:
        print("아무 문서도 받지 못했습니다.", file=sys.stderr)
        return 1

    (out / "index.json").write_text(
        json.dumps(written, ensure_ascii=False, indent=2), encoding="utf-8")

    checksums = {entry["file"]: checksum(out / entry["file"]) for entry in written}
    checksums["index.json"] = checksum(out / "index.json")
    manifest = {
        "corpus_id": "kr_law",
        "version": datetime.now(timezone.utc).strftime("%Y%m%d"),
        "source_url": SOURCE,
        "licence": LICENCE,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence_unit": "article",
        "checksums": checksums,
        "notes": (
            f"{len(written)} statutes across {len(TOPICS)} topics via 국가법령정보 OPEN API. "
            "Documents are not redistributed with this repository; re-run this script "
            "and the checksums must match for results to be comparable."
        ),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(entry["chars"] for entry in written)
    print(f"\n{len(written)}개 법령 / {total:,}자 → {out}")
    print("splits.json은 아직 없습니다 — 질의 집합이 만들어진 뒤 생성됩니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
