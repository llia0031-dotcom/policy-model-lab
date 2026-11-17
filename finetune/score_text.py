"""Score the text classifiers (results/text/*.jsonl) against the fixed labels.

Per method and test set: decision accuracy (violating vs compliant), violating caught, compliant wrongly
rejected, look-alikes wrongly rejected, label recall, exact category set, macro-F1 over the categories of each
language, unparsed answers, and median seconds per item. The real cases are reported only as these totals; where
their labels are not available (the public repository), their stored totals are kept.

Run: python finetune/score_text.py      -> results/text_summary.md and .json
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from finetune.predict import test_rows  # noqa: E402
from lab.common import LANG_CATS, RESULTS, read_jsonl  # noqa: E402
from lab.score import wilson  # noqa: E402

METHOD_ORDER = ["qwen3-0.6b-zs", "qwen3-0.6b-lora", "qwen3-1.7b-zs", "qwen3-1.7b-lora", "mmbert", "haiku", "haiku-400"]
SET_ORDER = ["public", "real", "synth-val", "video-ocr"]
GROUP_ORDER = ["grounded-review", "video-text", "real", "synth-val", "video-ocr"]


def group(r: dict) -> str:
    """The public set is reported in two parts: grounded-review's items and this project's video items as text."""
    if r["set"] != "public":
        return r["set"]
    return "grounded-review" if r["split"].startswith("gr-") else "video-text"


def metrics(gold_rows: dict, preds: list[dict]) -> dict:
    m = defaultdict(int)
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    secs = []
    langs = set()
    for p in preds:
        g = gold_rows[(p["set"], p["id"])]
        langs.add(g["lang"])
        gold, pred = set(g["gold"]), set(p["pred"])
        m["n"] += 1
        m["unparsed"] += not p["parsed"]
        m["decision_ok"] += bool(gold) == bool(pred)
        m["exact"] += gold == pred
        if gold:
            m["violating"] += 1
            m["caught"] += bool(pred)
        else:
            m["compliant"] += 1
            m["wrong_reject"] += bool(pred)
            if g.get("trap"):
                m["traps"] += 1
                m["trap_wrong"] += bool(pred)
        m["labels"] += len(gold)
        m["label_hit"] += len(gold & pred)
        for c in LANG_CATS[g["lang"]]:
            key = (g["lang"], c)
            tp[key] += c in gold and c in pred
            fp[key] += c not in gold and c in pred
            fn[key] += c in gold and c not in pred
        if p.get("seconds") is not None:
            secs.append(p["seconds"])
    f1s = []
    for key in tp.keys() | fn.keys():
        if tp[key] + fn[key] == 0:
            continue  # category absent from this set's gold labels
        prec = tp[key] / (tp[key] + fp[key]) if tp[key] + fp[key] else 0.0
        rec = tp[key] / (tp[key] + fn[key])
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    out = dict(m)
    out["macro_f1"] = round(statistics.mean(f1s), 3) if f1s else None
    out["seconds_median"] = round(statistics.median(secs), 4) if secs else None
    return out


def pct(k, n):
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "–"


def main() -> int:
    gold_rows = {(r["set"], r["id"]): r for r in test_rows(SET_ORDER)}
    available = {group(r) for r in gold_rows.values()}
    stored = RESULTS / "text_summary.json"
    old = json.loads(stored.read_text(encoding="utf-8")) if stored.exists() else {}
    table = {}
    for f in sorted((RESULTS / "text").glob("*.jsonl")):
        rows = read_jsonl(f)
        by_set = defaultdict(list)
        for r in rows:
            by_set[group(r)].append(r)
        for s, ps in by_set.items():
            if s in available:
                table[(rows[0]["method"], s)] = metrics(gold_rows, ps)
    # A test set whose labels are not in this copy (the private real cases, or the synthetic validation set when
    # $POLICY_LAB_DATA is not there) keeps its stored totals instead of being dropped.
    for k, v in old.items():
        meth, s = k.split("|", 1)
        if s not in available and (meth, s) not in table:
            table[(meth, s)] = v
    lines = ["# Text classifiers: results", "",
             "| test set | method | decision accuracy | violating caught | compliant wrongly rejected | look-alikes wrongly rejected | labels found | exact | macro-F1 | unparsed | median s/item |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in GROUP_ORDER:
        for meth in METHOD_ORDER:
            x = table.get((meth, s))
            if not x:
                continue
            lo, hi = wilson(x["decision_ok"], x["n"])
            lines.append(f"| {s} | {meth} | {pct(x['decision_ok'], x['n'])} [{100*lo:.0f}–{100*hi:.0f}] | {pct(x['caught'], x['violating'])} | "
                         f"{pct(x['wrong_reject'], x['compliant'])} | {pct(x.get('trap_wrong', 0), x.get('traps', 0))} | "
                         f"{pct(x['label_hit'], x['labels'])} | {pct(x['exact'], x['n'])} | {x['macro_f1']} | {x['unparsed']} | {x['seconds_median']} |")
    md = "\n".join(lines) + "\n"
    (RESULTS / "text_summary.md").write_text(md, encoding="utf-8")
    order = sorted(table, key=lambda k: (GROUP_ORDER.index(k[1]) if k[1] in GROUP_ORDER else 99,
                                         METHOD_ORDER.index(k[0]) if k[0] in METHOD_ORDER else 99, k))
    (RESULTS / "text_summary.json").write_text(json.dumps({f"{a}|{b}": table[(a, b)] for a, b in order}, indent=1), encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
