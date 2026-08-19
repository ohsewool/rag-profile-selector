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

    dense        BAAI/bge-m3 embeddings, cosine similarity. Added after the
                 first run of this experiment reported a headroom of 0.033 and
                 a 50% both-miss bucket - the two lexical profiles were failing
                 on the same queries, which is not a finding about selection so
                 much as about running two views of the same signal.
    hybrid-all   RRF over all three.

Only the train and validation splits are read. The test split stays sealed.

    python3 experiments/kr_law_retrieval.py
"""

from __future__ import annotations

import argparse
import json
import hashlib
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_profile_selector.corpus import Corpus
from rag_profile_selector.selector import QueryOutcome, headroom

# e5-small (118M) rather than bge-m3 (568M). The larger model was tried first
# and never finished embedding 745 articles on CPU before the run was killed.
# A result that cannot be produced is not a result.
DENSE_MODEL = "intfloat/multilingual-e5-small"

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

    def search_scored(self, query_terms: list[str], limit: int) -> list[tuple[str, float]]:
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
        return ranked[:limit]

    def search(self, query_terms: list[str], limit: int) -> list[str]:
        return [key for key, _ in self.search_scored(query_terms, limit)]


def reciprocal_rank_fusion(rankings: list[list[str]], limit: int, k: int = 60) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for position, key in enumerate(ranking):
            scores[key] += 1 / (k + position + 1)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [key for key, _ in ranked[:limit]]


class DenseRetriever:
    """Cosine similarity over sentence embeddings.

    Present so the experiment can answer the question the lexical-only run could
    not: whether profiles that fail differently leave anything to select between.
    Two BM25 variants disagree about tokenization and agree about everything
    else, so they miss together; this is the profile that can miss elsewhere.
    """

    def __init__(self, documents: dict[str, str], *, model_name: str = DENSE_MODEL,
                 batch_size: int = 32, max_length: int = 384,
                 cache: Path | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        # torch defaults to a thread count derived from the container's view of
        # the machine, which here was two. The first run spent 67 minutes of CPU
        # without finishing 745 articles; at sixteen threads the same work takes
        # about ten. Stating it explicitly is the difference between an
        # experiment that produces a result and one that gets killed.
        torch.set_num_threads(min(16, os.cpu_count() or 4))
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, use_safetensors=True).eval()
        self.max_length = max_length
        self.ids = list(documents)

        # Embedding the corpus is the expensive step and it does not change
        # unless the corpus does. Keyed by model and by a digest of the article
        # ids and text, so a re-fetch that alters a statute invalidates it rather
        # than silently reusing vectors for text that is no longer there.
        signature = hashlib.sha256(
            (model_name + "\x00".join(f"{k}\x01{documents[k]}" for k in self.ids))
            .encode("utf-8")).hexdigest()[:16]
        cache_file = (cache or CORPUS / ".embeddings") / f"{signature}.pt"
        if cache_file.exists():
            self.matrix = torch.load(cache_file, weights_only=True)
            print(f"  임베딩 캐시 사용: {cache_file.name}")
            return

        self.matrix = self._encode(
            [f"passage: {documents[key]}" for key in self.ids], batch_size, progress=True)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.matrix, cache_file)

    def _encode(self, texts: list[str], batch_size: int, *, progress: bool = False):
        torch = self._torch
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = self.tokenizer(
                    texts[start:start + batch_size], padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                )
                hidden = self.model(**batch).last_hidden_state
                # Mean pooling over real tokens: e5 is trained that way, and CLS
                # pooling on an e5 model would silently produce worse vectors
                # rather than an error.
                mask = batch["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                chunks.append(torch.nn.functional.normalize(pooled, p=2, dim=1))
                if progress and (start // batch_size) % 5 == 0:
                    print(f"    {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
        return torch.cat(chunks)

    def search_scored(self, query: str, limit: int) -> list[tuple[str, float]]:
        # e5 requires the asymmetric prefixes; without them query and passage
        # vectors sit in different regions and the ranking degrades quietly.
        vector = self._encode([f"query: {query}"], 1)
        scores = (self.matrix @ vector.T).squeeze(1)
        order = scores.argsort(descending=True)[:limit].tolist()
        return [(self.ids[index], float(scores[index])) for index in order]

    def search(self, query: str, limit: int) -> list[str]:
        return [key for key, _ in self.search_scored(query, limit)]


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
    parser.add_argument("--no-dense", action="store_true",
                        help="skip the embedding profile (no model download)")
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
    dense = None
    if not args.no_dense:
        print(f"{DENSE_MODEL} 로 조문 {len(articles)}개 임베딩 중...")
        dense = DenseRetriever(articles)
        print("완료\n")

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
        if dense is not None:
            rankings["dense"] = dense.search(text, args.k)
            rankings["hybrid-all"] = reciprocal_rank_fusion(
                [rankings["bm25-word"], rankings["bm25-char"], rankings["dense"]], args.k)
        quality = {name: reciprocal_rank(ranking, gold) for name, ranking in rankings.items()}
        outcomes.append(QueryOutcome(query_id=query_id, quality=quality))
        for name, score in quality.items():
            per_difficulty[item["difficulty"]][name].append(score)

    profiles = ["bm25-word", "bm25-char", "hybrid-rrf"]
    if dense is not None:
        profiles += ["dense", "hybrid-all"]
    print(f"{'프로파일':<14}{'MRR@' + str(args.k):>9}{'top-1':>9}{'미검색':>9}")
    print("-" * 42)
    for name in profiles:
        scores = [outcome.quality[name] for outcome in outcomes]
        mrr = sum(scores) / len(scores)
        top1 = sum(1 for score in scores if score == 1.0) / len(scores)
        missed = sum(1 for score in scores if score == 0.0) / len(scores)
        print(f"{name:<14}{mrr:>9.3f}{top1:>9.1%}{missed:>9.1%}")
    if dense is None:
        print(f"{'dense':<14}{'—':>9}{'—':>9}{'—':>9}   --no-dense 로 생략됨")

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

    print("\n프로파일 쌍별 실패 상관 (둘 다 놓친 질의 비율)")
    for i, first in enumerate(profiles):
        for second in profiles[i + 1:]:
            both = sum(1 for o in outcomes
                       if o.quality[first] == 0 and o.quality[second] == 0)
            split = sum(1 for o in outcomes
                        if (o.quality[first] == 0) != (o.quality[second] == 0))
            print(f"  {first:>11} / {second:<11} 동시실패 {both:>2}  갈림 {split:>2}")

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
