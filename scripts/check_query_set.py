#!/usr/bin/env python3
"""Check that the query set is answerable and not trivially so.

Two failures would make the benchmark meaningless in opposite directions.

A query whose gold article does not exist is unanswerable, and every profile
scores zero on it - noise dressed as difficulty. `validate_evidence_mapping`
catches that.

The subtler one: a query built by copying the article's own heading is solved by
term overlap alone. Every retriever finds it, no profile distinguishes itself,
and a benchmark of such queries reports that profile choice does not matter -
which is a conclusion about the query set, not about retrieval.

So this measures the overlap rather than trusting the difficulty label the author
attached. Labels are claims; the numbers below are the check on them.

    python3 scripts/check_query_set.py data/kr_law
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_profile_selector.corpus import validate_evidence_mapping

# Korean particles and function words carry no topical signal, so leaving them in
# would inflate every overlap score by the same amount and flatten the contrast.
STOPWORDS = {
    "이", "그", "저", "것", "수", "등", "및", "또는", "때", "경우", "대한", "관한",
    "있다", "없다", "하는", "하여", "위한", "따라", "대해", "인가요", "있나요",
    "되나요", "하나요", "무엇", "어떤", "어떻게", "뭔가요", "누구", "얼마나",
}


def tokens(text: str) -> set[str]:
    words = re.findall(r"[가-힣A-Za-z0-9]+", text)
    out = set()
    for word in words:
        if word in STOPWORDS or len(word) < 2:
            continue
        out.add(word)
        # Korean is agglutinative: 개인정보를 and 개인정보는 should meet. Trimming
        # one trailing syllable is crude but symmetric across query and article,
        # so it cannot favour one side.
        if len(word) > 2:
            out.add(word[:-1])
    return out


def load_articles(root: Path) -> dict[str, dict]:
    articles: dict[str, dict] = {}
    for path in sorted((root / "documents").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for article in payload["articles"]:
            if not article["is_article"] or article["is_repealed"]:
                continue
            articles[f"{payload['mst']}:{article['article_id']}"] = {
                "text": article["text"], "label": article["label"],
                "statute": payload["name"],
            }
    return articles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="data/kr_law")
    args = parser.parse_args()
    root = Path(args.root)

    articles = load_articles(root)
    queries = json.loads((root / "queries.json").read_text(encoding="utf-8"))["queries"]
    print(f"조문 {len(articles)}개 / 질의 {len(queries)}개\n")

    mapping = {item["id"]: item["evidence"] for item in queries}
    report = validate_evidence_mapping(mapping, articles.keys())
    if not report.ok:
        print("근거 매핑 실패:")
        print(report.summary())
        return 1
    print("근거 매핑: 모든 gold 조문이 코퍼스에 존재")

    # Overlap against the gold article, and against the best non-gold article.
    # A query only distinguishes retrievers when the two are close; when the gold
    # wins on raw vocabulary by a wide margin, any retriever finds it.
    rows, by_label = [], {}
    for item in queries:
        query_terms = tokens(item["text"])
        gold_id = item["evidence"][0]
        gold = len(query_terms & tokens(articles[gold_id]["text"])) / max(1, len(query_terms))
        rival = max(
            (len(query_terms & tokens(data["text"])) / max(1, len(query_terms))
             for key, data in articles.items() if key != gold_id),
            default=0.0,
        )
        rows.append((item["id"], item["difficulty"], gold, rival, gold - rival))
        by_label.setdefault(item["difficulty"], []).append(gold)

    print("\n난이도 라벨 대 실제 gold 어휘 중복")
    for label in ("lexical", "paraphrase", "situational"):
        values = by_label.get(label, [])
        if values:
            print(f"  {label:<12} n={len(values):<3} 평균 {sum(values)/len(values):.3f}")

    ordered = [sum(by_label[l]) / len(by_label[l])
               for l in ("lexical", "paraphrase", "situational") if by_label.get(l)]
    if ordered == sorted(ordered, reverse=True):
        print("  → 라벨 순서와 실제 중복 순서가 일치한다")
    else:
        print("  → 라벨이 실제 중복 순서와 어긋난다. 라벨을 고치거나 질의를 다시 써야 한다")

    # This used to flag queries whose gold beat the runner-up by more than 0.30
    # and report the count. The count was always zero, because the largest
    # margin this set produces is 0.200 - the check could not fire, and its "0"
    # read as a clean result rather than as a threshold placed out of reach.
    #
    # There is no honest replacement constant. What the retrieval experiment
    # actually shows, on 28 queries: every profile finds the gold article for
    # all three queries with a margin above 0.1, and for none of the fifteen
    # with a margin at or below zero. Three queries is not a boundary, so the
    # distribution is printed and the reader draws the line.
    print("\ngold가 차점 조문을 앞서는 정도(margin) 분포")
    bands = ((-1.0, 0.0), (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 1.0))
    for low, high in bands:
        in_band = [row for row in rows if low <= row[4] < high]
        if in_band:
            names = ", ".join(row[0] for row in sorted(in_band, key=lambda r: -r[4])[:4])
            print(f"  {low:+.1f} ~ {high:+.1f}  {len(in_band):>2}개   {names}")
    top = max(rows, key=lambda row: row[4])
    print(f"  최대 {top[4]:+.2f} ({top[0]})")
    print("  margin이 클수록 어휘 일치만으로 풀리고 프로파일을 구분하지 못한다. "
          "이 집합에서 margin > 0.1인 질의는 모든 프로파일이 찾아냈다(3건).")

    unanswered = [row for row in rows if row[2] == 0.0]
    if unanswered:
        print(f"\n⚠ gold와 어휘가 하나도 겹치지 않는 질의 {len(unanswered)}개: "
              f"{', '.join(r[0] for r in unanswered)}")
        print("  어려운 것일 수도, 매핑이 틀린 것일 수도 있다. 사람이 봐야 한다.")

    statutes = Counter(articles[item['evidence'][0]]["statute"] for item in queries)
    print("\n법령별 질의 분포")
    for name, count in statutes.most_common():
        print(f"  {count:>3}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
