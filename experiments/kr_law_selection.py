#!/usr/bin/env python3
"""Can a selector find the queries where the best fixed profile is wrong?

The retrieval run established the ceiling: headroom 0.107, and an oracle that
picks per query beats always-dense on five of twenty-eight. This asks the only
question left worth asking - whether those five are identifiable from features a
selector is allowed to see, before it sees any answer.

The rules here are hypotheses, not tuned settings. Each one states a belief
about when a cheap lexical profile is the better choice, and each is checked
against the same baselines:

    fixed:dense     always the best single profile. The bar.
    fixed:*         the other profiles, for context.
    oracle          picks the best per query. Unreachable; it reads the answer.
    rule:*          the hypotheses.

A rule that fails to beat fixed:dense is a finding. It means the divergence is
not visible in these features, and a selector built anyway would be a selector
guessing.

    python3 experiments/kr_law_selection.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from kr_law_retrieval import (
    BM25, CORPUS, DenseRetriever, char_ngrams, load_articles,
    reciprocal_rank, reciprocal_rank_fusion, word_tokens,
)
from rag_profile_selector.corpus import Corpus
from rag_profile_selector.probes import ProbeResult, assert_no_leakage, extract
from rag_profile_selector.selector import (
    FixedSelector, OracleSelector, QueryOutcome, RuleSelector, evaluate, headroom,
)

PROFILES = ["bm25-word", "bm25-char", "hybrid-rrf", "dense", "hybrid-all"]


def build(args):
    corpus = Corpus.open(CORPUS)
    articles = load_articles()
    queries = {
        item["id"]: item
        for item in json.loads((CORPUS / "queries.json").read_text(encoding="utf-8"))["queries"]
    }
    evaluated = list(corpus.queries("train")) + list(corpus.queries("validation"))

    word = BM25({key: word_tokens(text) for key, text in articles.items()})
    char = BM25({key: char_ngrams(text) for key, text in articles.items()})
    dense = DenseRetriever(articles)

    outcomes: list[QueryOutcome] = []
    features: dict[str, dict[str, float]] = {}

    for query_id in evaluated:
        item = queries[query_id]
        text, gold = item["text"], item["evidence"]

        word_hits = word.search_scored(word_tokens(text), args.k)
        char_hits = char.search_scored(char_ngrams(text), args.k)
        dense_hits = dense.search_scored(text, args.k)

        rankings = {
            "bm25-word": [key for key, _ in word_hits],
            "bm25-char": [key for key, _ in char_hits],
            "dense": [key for key, _ in dense_hits],
        }
        rankings["hybrid-rrf"] = reciprocal_rank_fusion(
            [rankings["bm25-word"], rankings["bm25-char"]], args.k)
        rankings["hybrid-all"] = reciprocal_rank_fusion(
            [rankings["bm25-word"], rankings["bm25-char"], rankings["dense"]], args.k)

        outcomes.append(QueryOutcome(
            query_id=query_id,
            quality={name: reciprocal_rank(rank, gold) for name, rank in rankings.items()},
        ))

        # Features come from the two cheap probes only. Using the dense ranking
        # here would make the selector's input cost as much as running dense
        # everywhere, which is the thing selection is supposed to avoid.
        probes = [
            ProbeResult(retriever="bm25-word",
                        identifiers=tuple(k for k, _ in word_hits),
                        scores=tuple(v for _, v in word_hits)),
            ProbeResult(retriever="bm25-char",
                        identifiers=tuple(k for k, _ in char_hits),
                        scores=tuple(v for _, v in char_hits)),
        ]
        found = extract(text, probes, k=args.k)
        assert_no_leakage(found)
        features[query_id] = found

    return outcomes, features


# Each rule is a stated belief about when lexical retrieval is trustworthy.
RULES = {
    "rule:lexical-when-confident":
        # A clear leader in the cheap ranking suggests the term match is real.
        (lambda f: "bm25-char" if f.get("top1_margin", 0.0) >= 0.25 else "dense",
         "cheap profile when its top hit stands clearly ahead"),
    "rule:lexical-when-probes-agree":
        # Two tokenizers landing on the same articles is corroboration.
        (lambda f: "bm25-char" if f.get("overlap_at_k", 0.0) >= 0.5 else "dense",
         "cheap profile when word and char tokenization agree"),
    "rule:dense-for-long-queries":
        # Longer questions tend to be phrased, not keyworded.
        (lambda f: "dense" if f.get("query_length", 0.0) >= 8 else "bm25-char",
         "dense for longer, more conversational queries"),
    "rule:fuse-when-uncertain":
        (lambda f: "dense" if f.get("score_decay", 0.0) < 0.5 else "hybrid-all",
         "dense when the cheap ranking is flat, fusion otherwise"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    outcomes, features = build(args)
    print(f"질의 {len(outcomes)}개 / 프로파일 {len(PROFILES)}개\n")
    print("사용 가능한 특징:", ", ".join(sorted(next(iter(features.values())))))

    space = headroom(outcomes, PROFILES)
    best_fixed = space["best_fixed_profile"]
    print(f"\n최선 고정 프로파일 {best_fixed}, headroom {space['headroom']:.4f}\n")

    selectors = [FixedSelector(name) for name in PROFILES]
    selectors.append(OracleSelector({o.query_id: o for o in outcomes}))
    selectors += [RuleSelector(rule, name=name) for name, (rule, _) in RULES.items()]

    print(f"{'선택기':<32}{'평균 regret':>12}{'최대':>8}{'최적 선택률':>12}")
    print("-" * 66)
    results = {}
    for selector in selectors:
        report = evaluate(selector, outcomes, features)
        results[report.selector] = report
        marker = "  ←" if report.selector.startswith("rule") and \
            report.mean_regret < results.get(f"fixed:{best_fixed}", report).mean_regret else ""
        print(f"{report.selector:<32}{report.mean_regret:>12.4f}"
              f"{report.max_regret:>8.3f}{report.optimal_rate:>11.1%}{marker}")

    baseline = results[f"fixed:{best_fixed}"].mean_regret
    winners = [name for name, r in results.items()
               if name.startswith("rule") and r.mean_regret < baseline]

    print("\n" + "=" * 66)
    if winners:
        print(f"고정 기준선({best_fixed}, regret {baseline:.4f})을 이긴 규칙:")
        for name in winners:
            # A mean over 28 queries hides how many of them the rule actually
            # touched. A win built on a net of one query is a win built on one
            # query, and reporting only the mean would let it pass as a result.
            rule, description = RULES[name]
            better = worse = 0
            for query_id, found in features.items():
                outcome = next(o for o in outcomes if o.query_id == query_id)
                delta = outcome.quality[rule(found)] - outcome.quality[best_fixed]
                better += delta > 0
                worse += delta < 0
            print(f"  {name}  regret {results[name].mean_regret:.4f}  — {description}")
            print(f"     기준선과 달라진 질의: 개선 {better} / 악화 {worse} "
                  f"/ 동일 {len(features) - better - worse}  → 순이득 {better - worse}건")
            if better - worse <= 2:
                print(f"     ⚠ 순이득 {better - worse}건은 {len(features)}건 표본에서 "
                      "우연과 구분되지 않는다. 이것을 결과로 보고하면 안 된다.")
        print(f"\n규칙 {len(RULES)}개를 시험했다. 그중 가장 좋은 하나를 사후에 고르는 것은")
        print("선택기를 찾은 것이 아니라 표본에 과적합한 것이다.")
    else:
        print(f"규칙 넷 중 어느 것도 고정 {best_fixed}(regret {baseline:.4f})를 이기지 못했다.")
        print("오라클이 이기는 질의가 이 특징들로는 식별되지 않는다는 뜻이다.")
        print("headroom이 존재한다는 것과 그것에 닿을 수 있다는 것은 다른 이야기다.")

    print("\n오라클이 고정 기준선보다 나은 질의:")
    for outcome in outcomes:
        regret = outcome.regret_of(best_fixed)
        if regret > 0:
            print(f"  {outcome.query_id}  {best_fixed} {outcome.quality[best_fixed]:.2f} "
                  f"→ 최선 {outcome.best_profile} {outcome.best_quality:.2f}  "
                  f"(top1_margin {features[outcome.query_id].get('top1_margin', 0):.2f}, "
                  f"overlap {features[outcome.query_id].get('overlap_at_k', 0):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
