"""측정한 어휘 중복으로 split을 층화한다.

이 코퍼스의 split은 난이도 **라벨**로 층화했다. 라벨 분포는 고르다 —
test는 paraphrase 5·situational 4·lexical 3. 그런데 검색 난이도를 지배하는 성질은
라벨이 아니라 **질의와 정답 조문의 어휘 중복**이었고, 중복이 0인 질의 9건 중 6건이
test에 몰렸다. test는 우연히 하드 모드가 됐다.

`check_query_set.py`가 그 중복을 이미 계산하고 있었는데 split은 쓰지 않았다.
**계산해둔 값을 안 쓴 것이지 없어서 못 쓴 것이 아니다.**

`KR_LAW_RESULTS.md`는 "다음 코퍼스에서는 측정값으로 층화한다"고 적어두었다. 계획을
문장으로만 남기면 다음 코퍼스가 올 때 다시 라벨로 나눌 것이다. 그래서 도구를 지금
만들고, 지금 데이터로 라벨 층화와 비교해 실제로 나아지는지 보인다.

**현재 코퍼스의 split은 다시 나누지 않는다.** 다시 나누면 test가 이미 본 데이터가
되고 이 실험에 검증된 결론이 남지 않는다. 이 스크립트는 기본적으로 비교만 출력하며,
`--write`를 줘야 파일을 쓴다 — 그것도 새 코퍼스 디렉터리에서만.

    python3 scripts/stratify_by_overlap.py                  # 현재 split과 비교
    python3 scripts/stratify_by_overlap.py <새코퍼스> --write
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from check_query_set import load_articles, tokens  # noqa: E402

SPLITS = ("train", "validation", "test")
# 현재 코퍼스와 같은 비율. 층화가 바꾸는 것은 크기가 아니라 어느 질의가 어디 가느냐다.
SHARES = (19, 9, 12)


def overlap(query_text: str, gold_id: str, articles: dict) -> float:
    """질의 어휘 중 정답 조문에도 나타나는 비율.

    `check_query_set.py`의 정의를 그대로 쓴다. 여기서 다시 정의하면 두 곳이
    어긋날 수 있고, 어긋나면 "층화한 값"과 "보고한 값"이 다른 것이 된다.
    """
    query_terms = tokens(query_text)
    if not query_terms or gold_id not in articles:
        return 0.0
    return len(query_terms & tokens(articles[gold_id]["text"])) / len(query_terms)


def stratify(measured: list[tuple[str, float]], *, seed: int,
             shares: tuple[int, ...] = SHARES) -> dict[str, str]:
    """중복이 낮은 것부터 순서대로 split에 돌아가며 배분한다.

    무작위 층화가 아니라 결정적 배분이다. 표본이 28건이라 무작위로 뽑으면 층 안에서
    또 치우칠 수 있고, 그 치우침이 정확히 지금 고치려는 문제다. 순서대로 돌리면
    어떤 split도 어려운 쪽이나 쉬운 쪽으로 몰릴 수 없다.

    seed는 동점 처리에만 쓴다 — 중복이 같은 질의들의 상대 순서.
    """
    import random

    rng = random.Random(seed)
    shuffled = list(measured)
    rng.shuffle(shuffled)
    ordered = sorted(shuffled, key=lambda item: item[1])

    quota = dict(zip(SPLITS, shares))
    assignment: dict[str, str] = {}
    cycle = [name for name in SPLITS for _ in range(quota[name])]
    # 층 안에서 순환: 가장 어려운 것부터 train/validation/test/train/... 이 아니라
    # 비율에 맞춰 고르게 섞이도록 인덱스로 배분한다.
    total = sum(shares)
    positions = {name: [] for name in SPLITS}
    for index in range(total):
        # 각 split이 자기 몫만큼 균등 간격을 차지한다.
        best = max(SPLITS, key=lambda name: quota[name] * (index + 1) / total - len(positions[name]))
        positions[best].append(index)
    for name, indexes in positions.items():
        for index in indexes:
            if index < len(ordered):
                assignment[ordered[index][0]] = name
    del cycle
    return assignment


def describe(assignment: dict[str, str], measured: dict[str, float]) -> dict[str, dict]:
    found = {}
    for name in SPLITS:
        values = [measured[q] for q, s in assignment.items() if s == name]
        if not values:
            continue
        found[name] = {
            "n": len(values),
            "mean": round(statistics.mean(values), 4),
            "zero": sum(1 for v in values if v == 0.0),
        }
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", default="data/kr_law")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write", action="store_true",
                        help="splits.json을 쓴다. 기존 파일이 있으면 거부한다.")
    arguments = parser.parse_args(argv)
    root = Path(arguments.root)

    articles = load_articles(root)
    queries = json.loads((root / "queries.json").read_text(encoding="utf-8"))["queries"]
    measured = {item["id"]: overlap(item["text"], item["evidence"][0], articles)
                for item in queries}
    if not measured:
        print("질의가 없다 — 이 결과는 아무 뜻도 없다")
        return 1

    proposed = stratify(list(measured.items()), seed=arguments.seed)

    existing_path = root / "splits.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))["assignments"]
        print("현재 split (난이도 라벨로 층화)")
        for name, stats in describe(existing, measured).items():
            print(f"  {name:<12} n={stats['n']:<3} 평균 중복 {stats['mean']:.4f}  "
                  f"중복 0인 질의 {stats['zero']}건")
        spread_before = _spread(describe(existing, measured))
        print(f"  → split 간 평균 중복 격차 {spread_before:.4f}\n")
    else:
        spread_before = None

    print(f"제안 split (측정한 어휘 중복으로 층화, seed={arguments.seed})")
    for name, stats in describe(proposed, measured).items():
        print(f"  {name:<12} n={stats['n']:<3} 평균 중복 {stats['mean']:.4f}  "
              f"중복 0인 질의 {stats['zero']}건")
    spread_after = _spread(describe(proposed, measured))
    print(f"  → split 간 평균 중복 격차 {spread_after:.4f}")

    if spread_before is not None:
        print(f"\n격차 {spread_before:.4f} → {spread_after:.4f}"
              f" ({'개선' if spread_after < spread_before else '악화'})")
        print("\n**현재 코퍼스는 다시 나누지 않는다.** 다시 나누면 test가 이미 본 "
              "데이터가 되고, 이 실험에 검증된 결론이 남지 않는다.")

    if arguments.write:
        if existing_path.exists():
            print(f"\n거부: {existing_path}가 이미 있다. 봉인된 split을 덮어쓰지 않는다.")
            return 1
        existing_path.write_text(json.dumps(
            {"seed": arguments.seed, "stratified_on": "measured_lexical_overlap",
             "assignments": proposed}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n{existing_path}를 썼다.")
    return 0


def _spread(stats: dict[str, dict]) -> float:
    """split 간 평균 중복의 최대-최소. 0에 가까울수록 고르게 나뉜 것이다."""
    means = [item["mean"] for item in stats.values()]
    return round(max(means) - min(means), 4) if means else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
