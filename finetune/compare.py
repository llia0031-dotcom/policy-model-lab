"""Paired comparisons between text classifiers: accuracy difference with a 95% bootstrap interval.

Each pair is scored on the items both methods answered in a test set. The interval comes from 4,000 resamples
of those items (seed 0), keeping each item's two answers together. Added after the preregistered runs, to
report how far apart two methods are; it changes no score. A test set whose labels are not in this copy (the
private real cases) keeps its stored rows.

Run: python finetune/compare.py      -> results/text_paired.md
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from finetune.predict import test_rows  # noqa: E402
from finetune.score_text import GROUP_ORDER, SET_ORDER, group  # noqa: E402
from lab.common import RESULTS, read_jsonl  # noqa: E402

PAIRS = [  # (a, b): the table reports b minus a
    ("qwen3-1.7b-zs", "qwen3-1.7b-lora"),
    ("qwen3-0.6b-zs", "qwen3-0.6b-lora"),
    ("haiku", "qwen3-1.7b-lora"),
    ("haiku-400", "qwen3-1.7b-lora"),
    ("mmbert", "qwen3-1.7b-lora"),
    ("qwen3-0.6b-lora", "qwen3-1.7b-lora"),
]
REPS = 4000


def right(method: str, gold: dict) -> dict:
    out = {}
    path = RESULTS / "text" / f"{method}.jsonl"
    for p in read_jsonl(path) if path.exists() else []:
        g = gold.get((p["set"], p["id"]))
        if g:
            out[(group(g), p["id"])] = bool(g["gold"]) == bool(p["pred"])
    return out


def main() -> int:
    gold = {(r["set"], r["id"]): r for r in test_rows(SET_ORDER)}
    available = {group(r) for r in gold.values()}
    stored = RESULTS / "text_paired.md"
    old = stored.read_text(encoding="utf-8").splitlines() if stored.exists() else []
    lines = ["# Text classifiers: paired differences", "",
             "Decision accuracy of the second method minus the first, in percentage points, on the items both answered. "
             f"95% interval from {REPS:,} paired bootstrap resamples (seed 0).", "",
             "| test set | second method | first method | items | difference | 95% interval |", "|---|---|---|---|---|---|"]
    for grp in GROUP_ORDER:
        if grp not in available:  # labels not in this copy (the private real cases): keep the stored rows
            lines += [l for l in old if l.startswith(f"| {grp} |")]
            continue
        for a, b in PAIRS:
            A, B = right(a, gold), right(b, gold)
            ids = sorted(k for k in A if k[0] == grp and k in B)
            if not ids:
                continue
            xa, xb = [A[i] for i in ids], [B[i] for i in ids]
            n = len(ids)
            rnd = random.Random(0)
            diffs = sorted(sum(xb[j] - xa[j] for j in (rnd.randrange(n) for _ in range(n))) / n for _ in range(REPS))
            d = (sum(xb) - sum(xa)) / n
            lo, hi = diffs[int(0.025 * REPS)], diffs[int(0.975 * REPS)]
            lines.append(f"| {grp} | {b} | {a} | {n} | {100 * d:+.1f} | {100 * lo:+.1f} to {100 * hi:+.1f} |")
    md = "\n".join(lines) + "\n"
    (RESULTS / "text_paired.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
