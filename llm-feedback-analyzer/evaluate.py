"""Evaluate the classifier against the labelled gold set.

Loads `data/gold_set.csv`, classifies every review, and reports
per-field accuracy plus a quick error analysis.

Usage:
    python evaluate.py
    python evaluate.py --gold data/gold_set.csv --limit 5
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from classifier import analyse


def main(gold_path: str, limit: int | None) -> int:
    path = Path(gold_path)
    if not path.exists():
        print(f"Gold set not found: {gold_path}", file=sys.stderr)
        return 1

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if limit:
        rows = rows[:limit]
    if not rows:
        print("Gold set is empty.", file=sys.stderr)
        return 1

    sentiment_correct = 0
    churn_correct = 0
    misses: list[tuple[int, str, str, str, str, str]] = []

    for i, row in enumerate(rows, 1):
        pred = analyse(row["review"])
        s_ok = pred.sentiment.value == row["sentiment"]
        c_ok = pred.churn_risk.value == row["churn_risk"]
        sentiment_correct += int(s_ok)
        churn_correct += int(c_ok)
        if not (s_ok and c_ok):
            misses.append((
                i, row["review"][:60],
                row["sentiment"], pred.sentiment.value,
                row["churn_risk"], pred.churn_risk.value,
            ))
        flag = lambda ok: "✓" if ok else "✗"
        print(f"[{i:>2}/{len(rows)}] sentiment {flag(s_ok)} | churn {flag(c_ok)}")

    n = len(rows)
    print("\n=== Results ===")
    print(f"Sentiment  accuracy: {sentiment_correct}/{n} = {sentiment_correct/n:.1%}")
    print(f"Churn-risk accuracy: {churn_correct}/{n} = {churn_correct/n:.1%}")

    if misses:
        print(f"\n=== Error analysis ({len(misses)} misses, showing up to 5) ===")
        for i, snippet, s_t, s_p, c_t, c_p in misses[:5]:
            print(f"  #{i}: '{snippet}...'")
            print(f"      sentiment  expected={s_t:<8}  predicted={s_p}")
            print(f"      churn_risk expected={c_t:<8}  predicted={c_p}")

    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gold", default="data/gold_set.csv")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    sys.exit(main(args.gold, args.limit))
