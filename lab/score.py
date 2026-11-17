"""Score answers against the fixed labels.

Per item the gold answer is a set of categories (empty = compliant) and the prediction is the set of
categories the model listed. Reported:
  decision accuracy   violating vs compliant called right
  caught              violating items with at least one category listed (1 - miss rate)
  wrongly rejected    compliant items with any category listed (traps reported separately)
  label recall        gold labels whose category was listed, overall and by where the violation sits
  exact               the listed set equals the gold set
  evidence            for labels with an exact span that were caught, the evidence overlaps the span
An answer that cannot be parsed counts as "nothing listed" and is counted in `unparsed`.

Run: python -m lab.score            (video results -> results/video_summary.md and .json)
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
import unicodedata
from collections import defaultdict

from lab.common import RESULTS, load_video_items, read_jsonl

CONDITION_ORDER = ["text", "cover", "frames8", "video", "frames16", "ocr8", "ocr16", "siglip-frames8", "siglip-frames16"]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return "".join(ch for ch in s if ch.isalnum())


def grams(s: str, n: int) -> set:
    return {s[i:i + n] for i in range(max(len(s) - n + 1, 1))}


def evidence_matches(evidence: str, span: str) -> bool:
    e, s = norm(evidence), norm(span)
    if not e or not s:
        return False
    if s in e or (len(e) >= 4 and e in s):
        return True
    n = 2 if any("一" <= ch <= "鿿" for ch in s) else 4
    gs = grams(s, n)
    return len(gs & grams(e, n)) / len(gs) >= 0.5


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def location(label: dict) -> str:
    w = label["where"]
    if w == "screen":
        return "screen-cover" if label.get("shot") == 1 else "screen-later"
    return w


def score(items: list[dict], rows: list[dict]) -> dict:
    by_id = {r["id"]: r for r in rows}
    m = defaultdict(int)
    loc_hit, loc_tot = defaultdict(int), defaultdict(int)
    per_item = {}
    secs = []
    for it in items:
        r = by_id.get(it["id"])
        if r is None:
            m["missing"] += 1
            continue
        gold = {l["cat"] for l in it["labels"]}
        pred = {v["category"] for v in r["violations"]}
        ev = defaultdict(list)
        for v in r["violations"]:
            ev[v["category"]].append(v.get("evidence", ""))
        m["n"] += 1
        m["unparsed"] += 0 if r.get("parsed", True) else 1
        if r.get("seconds") is not None:  # batch API answers carry no per-item time
            secs.append(r["seconds"])
        correct_decision = bool(gold) == bool(pred)
        m["decision_ok"] += correct_decision
        m["exact"] += gold == pred
        if gold:
            m["violating"] += 1
            m["caught"] += bool(pred)
            m["extra_cats"] += len(pred - gold)
        else:
            m["compliant"] += 1
            m["wrong_reject"] += bool(pred)
            if it.get("trap") and not it["trap"].startswith(("普通商品介绍", "Plain product")):
                m["traps"] += 1
                m["trap_wrong"] += bool(pred)
        for l in it["labels"]:
            loc = location(l)
            m["labels"] += 1
            loc_tot[loc] += 1
            hit = l["cat"] in pred
            m["label_hit"] += hit
            loc_hit[loc] += hit
            if hit and "span" in l:
                m["ev_checked"] += 1
                m["ev_ok"] += any(evidence_matches(e, l["span"]) for e in ev[l["cat"]])
        per_item[it["id"]] = {"decision_ok": correct_decision, "gold": sorted(gold), "pred": sorted(pred)}
    out = {k: 0 for k in ("n", "unparsed", "decision_ok", "exact", "violating", "caught", "extra_cats", "compliant",
                          "wrong_reject", "traps", "trap_wrong", "labels", "label_hit", "ev_checked", "ev_ok", "missing")}
    out.update(m)
    out["by_location"] = {k: [loc_hit[k], loc_tot[k]] for k in sorted(loc_tot)}
    out["seconds_median"] = round(statistics.median(secs), 2) if secs else None
    out["seconds_mean"] = round(statistics.mean(secs), 2) if secs else None
    out["peak_gb"] = max((r.get("peak_gb", 0) for r in rows), default=0)
    out["per_item"] = per_item
    return out


def paired_bootstrap(a: dict, b: dict, key="decision_ok", reps=4000, seed=0) -> tuple[float, float, float]:
    """Difference in mean of per-item `key` (b - a), with a 95% percentile interval over items."""
    ids = sorted(set(a["per_item"]) & set(b["per_item"]))
    xa = [a["per_item"][i][key] for i in ids]
    xb = [b["per_item"][i][key] for i in ids]
    rnd = random.Random(seed)
    diffs = []
    for _ in range(reps):
        s = [rnd.randrange(len(ids)) for _ in ids]
        diffs.append(sum(xb[i] - xa[i] for i in s) / len(ids))
    diffs.sort()
    return (sum(xb) - sum(xa)) / len(ids), diffs[int(0.025 * reps)], diffs[int(0.975 * reps)]


def pct(k, n):
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "–"


def video_report() -> str:
    items = load_video_items()
    files = sorted((RESULTS / "video").glob("*.jsonl"))
    runs = {}
    for f in files:
        if f.name.startswith("dev-") or ".prompt-" in f.name:
            continue
        rows = read_jsonl(f)
        groups = defaultdict(list)
        for r in rows:
            groups[(r["model"], r["condition"])].append(r)
        for k, rs in groups.items():
            runs[k] = score(items, rs)
    order = lambda k: (k[0], CONDITION_ORDER.index(k[1]) if k[1] in CONDITION_ORDER else 99)
    lines = ["# Video test: results", "",
             f"{len(items)} videos (48 Chinese, 48 English): {sum(1 for i in items if i['labels'])} violating with {sum(len(i['labels']) for i in items)} labels, {sum(1 for i in items if not i['labels'])} compliant (look-alike traps are counted in their own column).", "",
             "| model | input | decision accuracy | violating caught | compliant wrongly rejected | traps wrongly rejected | labels found | exact | evidence matches span | unparsed | median s | peak GB |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(runs, key=order):
        s = runs[k]
        lo, hi = wilson(s["decision_ok"], s["n"])
        lines.append(f"| {k[0]} | {k[1]} | {pct(s['decision_ok'], s['n'])} [{100*lo:.0f}–{100*hi:.0f}] | {pct(s['caught'], s['violating'])} | "
                     f"{pct(s['wrong_reject'], s['compliant'])} | {pct(s.get('trap_wrong', 0), s.get('traps', 0))} | {pct(s['label_hit'], s['labels'])} | "
                     f"{pct(s['exact'], s['n'])} | {pct(s.get('ev_ok', 0), s.get('ev_checked', 0))} | {s.get('unparsed', 0)} | {s['seconds_median']} | {s['peak_gb'] or '–'} |")
    lines += ["", "## Labels found, by where the violation sits", "",
              "| model | input | caption | subtitle on cover | subtitle later | 0.5 s sticker | footage only |", "|---|---|---|---|---|---|---|"]
    for k in sorted(runs, key=order):
        b = runs[k]["by_location"]
        cell = lambda loc: pct(*b.get(loc, [0, 0]))
        lines.append(f"| {k[0]} | {k[1]} | {cell('caption')} | {cell('screen-cover')} | {cell('screen-later')} | {cell('flash')} | {cell('visual')} |")
    (RESULTS / "video_summary.json").write_text(json.dumps({f"{a}|{b}": {kk: vv for kk, vv in v.items()} for (a, b), v in runs.items()},
                                                          ensure_ascii=False, indent=1), encoding="utf-8")
    return "\n".join(lines) + "\n"


def main() -> int:
    md = video_report()
    (RESULTS / "video_summary.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
