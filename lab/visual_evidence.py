"""Exploratory analysis (not pre-registered): when a model flags a footage-only violation, does its evidence
point at what the footage shows, or only at the product? A second table counts the categories each run lists
per ad and how many of them come with no evidence text.

For each footage-only label that a run caught (the category was listed), the evidence text is checked:
  "right reason"  = it says the footage shows the action or object: a footage word such as 画面 / footage / shows
                    somewhere in the evidence, and an action or object word below in a clause without a negation
                    (未 无 没有 缺失 无法 cannot not no) or, for drinking and the flag, an inference word
                    (暗示 suggest imply; for drinking also 诱导 怂恿 induce encourage promote);
  "category only" = it does not, for example it only says the product is a beer or that the caption suggests
                    drinking.
Action or object words:
  ALCOHOL       drinking shown: 喝 饮 举杯 干杯 碰杯 drink sip toast cheers raise
  FLAG          the flag: 国旗 旗 五星 flag
  BEFORE_AFTER  a comparison: 前后 对比 之前 之后 before after comparison side by side
The responsible-drinking line that every alcohol ad carries (理性饮酒 / drink responsibly) is removed before the
check. (The first version matched action words anywhere, which counted quotes of that line, "the caption
suggests drinking" and "no drinking is described" as right reasons; tightened after reading its output. A second
change after reading the Qwen3-VL-2B answers: "responsible drinking" is removed too, and inputs that never show
the footage get no right-reason count, because one OCR answer wrote "text shows ... responsible drinking advice".)
For the look-alike alcohol ads (only the bottle is shown, or someone drinks tea, water, coffee or juice), the same
check shows whether a wrong ALCOHOL call says drinking was shown.
Text-only and OCR inputs never show the footage, so all their catches count as category only.

Run: python -m lab.visual_evidence  -> results/visual_evidence.md
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict

from lab.common import RESULTS, load_video_items, read_jsonl

WORDS = {
    "ALCOHOL": r"喝|饮(?!料)|举杯|干杯|碰杯|drink|sip|toast|cheers|raise|raising",
    "FLAG": r"国旗|旗|五星|flag",
    "BEFORE_AFTER": r"前后|对比|之前|之后|before|after|comparison|side by side|side-by-side",
}
FOOTAGE = r"画面|镜头|footage|frames?\b|shows?\b|shown|showing|depict|scene"
WARNING = r"理性饮酒|适量饮酒|(please )?drink responsibly|responsible drinking"
NEGATION = r"未|无|没有|缺失|无法|cannot|\bnot\b|\bno\b"
# For drinking and the flag, "suggests", "induces" or "promotes" means the model inferred it rather than saw it.
# For a before-and-after, "the footage implies a before-and-after" is the right reason, so only negations count.
HEDGE = {"ALCOHOL": NEGATION + r"|暗示|诱导|怂恿|suggest|impl|induc|encourag|promot",
         "FLAG": NEGATION + r"|暗示|suggest|impl",
         "BEFORE_AFTER": NEGATION}
ALCOHOL_LOOKALIKES = {"zv-44", "zv-45", "zv-46", "zv-48", "ev-44", "ev-45", "ev-46"}
SEES_FOOTAGE = {"cover", "frames8", "frames16", "video"}


def right_reason(cat: str, evidence: str) -> bool:
    e = re.sub(WARNING, "", evidence, flags=re.I)
    if not re.search(FOOTAGE, e, re.I):
        return False
    clauses = re.split(r"[，。；、,;.()（）]", e)
    return any(re.search(WORDS[cat], c, re.I) and not re.search(HEDGE[cat], c, re.I) for c in clauses)


def main() -> int:
    items = {i["id"]: i for i in load_video_items()}
    lines = ["# Footage-only violations: does the evidence point at the footage?", "",
             "Exploratory, not pre-registered. Built by `python -m lab.visual_evidence` (rule in the script's header).", "",
             "| model | input | sees the footage | footage-only labels caught | evidence says the footage shows it | category only | alcohol look-alikes wrongly flagged (of 7) | of those, evidence says drinking is shown |",
             "|---|---|---|---|---|---|---|---|"]
    for f in sorted((RESULTS / "video").glob("*.jsonl")):
        if f.name.startswith("dev-") or ".prompt-" in f.name:
            continue
        groups = defaultdict(list)
        for r in read_jsonl(f):
            groups[(r["model"], r["condition"])].append(r)
        for (m, c), rows in sorted(groups.items()):
            caught = right = 0
            look_wrong = look_drink = 0
            for r in rows:
                it = items.get(r["id"])
                if not it:
                    continue
                ev = defaultdict(str)
                for v in r["violations"]:
                    ev[v["category"]] += " " + v.get("evidence", "")
                for l in it["labels"]:
                    if l["where"] != "visual" or l["cat"] not in ev:
                        continue
                    caught += 1
                    right += right_reason(l["cat"], ev[l["cat"]])
                if r["id"] in ALCOHOL_LOOKALIKES and "ALCOHOL" in ev:
                    look_wrong += 1
                    look_drink += right_reason("ALCOHOL", ev["ALCOHOL"])
            if not caught and not look_wrong:
                continue
            if c in SEES_FOOTAGE:
                lines.append(f"| {m} | {c} | yes | {caught}/16 | {right} | {caught - right} | {look_wrong} | {look_drink} |")
            else:
                lines.append(f"| {m} | {c} | no | {caught}/16 | – | {caught} | {look_wrong} | – |")
    # How many categories a run lists per ad, and how many of them come with no evidence text at all.
    lines += ["", "## Categories listed per ad", "",
              "| model | input | categories listed | per ad | per compliant ad | listed with no evidence |", "|---|---|---|---|---|---|"]
    for f in sorted((RESULTS / "video").glob("*.jsonl")):
        if f.name.startswith("dev-") or ".prompt-" in f.name:
            continue
        groups = defaultdict(list)
        for r in read_jsonl(f):
            if r["id"] in items:
                groups[(r["model"], r["condition"])].append(r)
        for (m, c), rows in sorted(groups.items()):
            if c == "ocr16-text-classifier" or m == "siglip2":
                continue  # these answer with categories only, by design
            listed = sum(len(r["violations"]) for r in rows)
            comp = [r for r in rows if not items[r["id"]]["labels"]]
            per_comp = sum(len(r["violations"]) for r in comp) / len(comp) if comp else 0
            empty = sum(1 for r in rows for v in r["violations"] if not str(v.get("evidence", "")).strip())
            lines.append(f"| {m} | {c} | {listed} | {listed / len(rows):.2f} | {per_comp:.2f} | {empty} |")
    md = "\n".join(lines) + "\n"
    (RESULTS / "visual_evidence.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
