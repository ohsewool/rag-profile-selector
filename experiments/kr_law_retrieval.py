#!/usr/bin/env python3
"""Retrieval over the Korean statute corpus, and whether selection is worth building.

This answers `headroom()` first, before any selector exists. If the best single
profile is already as good as an oracle that picks per query, then per-query
selection cannot help on this data and the honest response is to stop rather
than to tune. That question is cheap to ask now and expensive to ignore later.

What is compared, and what is not:

    bm25-word    BM25 over whitespace-ish word tokens. The obvious baseline.
    bm25-char    BM25 over character 3-grams. Not a variation for its own sake:
                 Korean is agglutinative, so 개인정보를 and 개인정보는 are
                 different word tokens and identical over character n-grams.
                 This is the axis on which Korean retrieval actually splits.
    hybrid-rrf   Reciprocal-rank fusion of the two.

    dense        NOT RUN. No embedding model is available in this environment,
                 and TF-IDF cosine is not a dense retriever whatever it is
                 labelled. Reporting it as `dense` would make every conclusion
                 about lexical-versus-semantic retrieval false. It is absent,
                 and the results table says so.

Only the train and validation splits are read. The test split stays sealed.

    python3 experiments/kr_law_retrieval.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_profile_selector.corpus import Corpus
from rag_profile_selector.selector import QueryOutcome, headroom

CORPUS = ROOT / "data" / "kr_law"


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[가-힣A-Za-z0-9]+", text.lower())


def char_ngrams(text: str, n: int = 3) -> list[str]:
    """Character n-grams over the whole string, spaces removed.

    Removing spaces is deliberate: Korean spacing is inconsistent in practice
    (한 번 / 한번), and n-grams that straddle a space boundary are exactly the
    ones that survive that inconsistency.
    """
    packed = re.sub(r"[^가-힣A-Za-z0-9]", "", text.lower())
    if len(packed) < n:
        return [packed] if packed else []
    return [packed[i:i + n] for i in range(len(packed) - n + 1)]


class BM25:
    """Okapi BM25. Implemented here rather than pulled in as a dependency.

    Two hundred lines of library for one formula would obscure the one thing
    that matters in this experiment: both profiles run the identical ranking
    function and differ only in the tokenizer. Any gap between them is
    attributable to tokenization and to nothing else.
    """

    def __init__(self, documents: dict[str, list[str]], *, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = list(documents)
        self.lengths = {key: len(terms) for key, terms in documents.items()}
        self.average_length = sum(self.lengths.values()) / max(1, len(self.lengths))
        self.frequencies = {key: Counter(terms) for key, terms in documents.items()}
        appearances: Counter[str] = Counter()
        for terms in documents.values():
            appearances.update(set(terms))
        total = len(documents)
        self.idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in appearances.items()
        }
        self.postings: dict[str, list[str]] = defaultdict(list)
        for key, terms in documents.items():
            for term in set(terms):
                self.postings[term].append(key)

    def search(self, query_terms: list[str], limit: int) -> list[str]:
        scores: dict[str, float] = defaultdict(float)
        for term in query_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for key in self.postings[term]:
                frequency = self.frequencies[key][term]
                norm = 1 - self.b + self.b * self.lengths[key] / self.average_length
                scores[key] += idf * frequency * (self.k1 + 1) / (frequency + self.k1 * norm)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [key for key, _ in ranked[:limit]]


def reciprocal_rank_fusion(rankings: list[list[str]], limit: int, k: int = 60) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for position, key in enumerate(ranking):
            scores[key] += 1 / (k + position + 1)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [key for key, _ in ranked[:limit]]


def load_articles() -> dict[str, str]:
    articles: dict[str, str] = {}
    for path in sorted((CORPUS / "documents").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for article in payload["articles"]:
            if article["is_article"] and not article["is_repealed"]:
                articles[f"{payload['mst']}:{article['article_id']}"] = article["text"]
    return articles


def reciprocal_rank(ranking: list[str], gold: list[str]) -> float:
    """1/rank of the first correct article, 0 if none is retrieved.

    Chosen over recall because a citation the reader has to hunt for is a
    citation they will not check. Where the right article lands matters.
    """
    for position, key in enumerate(ranking, start=1):
        if key in gold:
            return 1 / position
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=4, help="results per query")
    args = parser.parse_args()

    corpus = Corpus.open(CORPUS)
    articles = load_articles()
    queries = {
        item["id"]: item
        for item in json.loads((CORPUS / "queries.json").read_text(encoding="utf-8"))["queries"]
    }
    evaluated = list(corpus.queries("train")) + list(corpus.queries("validation"))
    print(f"조문 {len(articles)}개 / 평가 질의 {len(evaluated)}개 (test split은 봉인 상태)\n")

    engines = {
        "bm25-word": BM25({key: word_tokens(text) for key, text in articles.items()}),
        "bm25-char": BM25({key: char_ngrams(text) for key, text in articles.items()}),
    }

    outcomes: list[QueryOutcome] = []
    per_difficulty: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for query_id in evaluated:
        item = queries[query_id]
        text, gold = item["text"], item["evidence"]
        rankings = {
            "bm25-word": engines["bm25-word"].search(word_tokens(text), args.k),
            "bm25-char": engines["bm25-char"].search(char_ngrams(text), args.k),
        }
        rankings["hybrid-rrf"] = reciprocal_rank_fusion(
            [rankings["bm25-word"], rankings["bm25-char"]], args.k)
        quality = {name: reciprocal_rank(ranking, gold) for name, ranking in rankings.items()}
        outcomes.append(QueryOutcome(query_id=query_id, quality=quality))
        for name, score in quality.items():
            per_difficulty[item["difficulty"]][name].append(score)

    profiles = ["bm25-word", "bm25-char", "hybrid-rrf"]
    print(f"{'프로파일':<14}{'MRR@' + str(args.k):>9}{'top-1':>9}{'미검색':>9}")
    print("-" * 42)
    for name in profiles:
        scores = [outcome.quality[name] for outcome in outcomes]
        mrr = sum(scores) / len(scores)
        top1 = sum(1 for score in scores if score == 1.0) / len(scores)
        missed = sum(1 for score in scores if score == 0.0) / len(scores)
        print(f"{name:<14}{mrr:>9.3f}{top1:>9.1%}{missed:>9.1%}")
    print(f"{'dense':<14}{'—':>9}{'—':>9}{'—':>9}   임베딩 모델 없음, 실행 안 함")

    print("\n난이도별 MRR")
    print(f"{'':14}" + "".join(f"{name:>12}" for name in profiles))
    for difficulty in ("lexical", "paraphrase", "situational"):
        row = per_difficulty.get(difficulty)
        if not row:
            continue
        cells = "".join(f"{sum(row[n]) / len(row[n]):>12.3f}" for n in profiles)
        print(f"{difficulty:<14}{cells}")

    space = headroom(outcomes, profiles)
    print("\n" + "=" * 42)
    print("선택기를 만들 값어치가 있는가")
    print(f"  최선 고정 프로파일 : {space['best_fixed_profile']}")
    print(f"  그 평균 regret     : {space['best_fixed_mean_regret']:.4f}")
    print(f"  오라클 평균 regret : {space['oracle_mean_regret']:.4f}")
    print(f"  headroom           : {space['headroom']:.4f}")
    print("  프로파일별 평균 regret:")
    for name, value in space["per_profile_mean_regret"].items():
        print(f"    {name:<14}{value:.4f}")

    if space["headroom"] < 0.02:
        print("\n  → headroom이 사실상 0이다. 어떤 선택기도 최선 고정 프로파일을 "
              "의미 있게 이길 수 없으므로, 만들 이유가 없다.")
    else:
        oracle_wins = sum(
            1 for outcome in outcomes
            if outcome.regret_of(space["best_fixed_profile"]) > 0)
        print(f"\n  → 오라클이 최선 고정보다 나은 질의: {oracle_wins}/{len(outcomes)}건. "
              "선택이 얻을 수 있는 최대치가 여기까지다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
