"""Offline checks: no model, no network, no GPU. Run: python -m unittest discover -s tests -v"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from finetune.prompt import parse as parse_cls  # noqa: E402
from finetune.prompt import prompt as cls_prompt  # noqa: E402
from lab.common import ALL_CATS, LANG_CATS, load_video_items, manifest, parse_answer, read_jsonl, user_text  # noqa: E402
from lab.frames import PLANS  # noqa: E402
from lab.score import evidence_matches, location, score  # noqa: E402


class VideoItems(unittest.TestCase):
    def setUp(self):
        self.items = load_video_items()

    def test_counts_match_preregistration(self):
        self.assertEqual(len(self.items), 96)
        self.assertEqual(sum(1 for i in self.items if i["labels"]), 64)
        self.assertEqual(sum(len(i["labels"]) for i in self.items), 72)
        locs = Counter(location(l) for i in self.items for l in i["labels"])
        self.assertEqual(dict(locs), {"caption": 16, "screen-cover": 3, "screen-later": 17, "flash": 20, "visual": 16})

    def test_spans_are_verbatim_and_categories_fit_language(self):
        for it in self.items:
            self.assertEqual(len(it["lines"]), 4, it["id"])
            for l in it["labels"]:
                self.assertIn(l["cat"], LANG_CATS[it["lang"]], it["id"])
                if l["where"] == "caption":
                    self.assertIn(l["span"], it["caption"], it["id"])
                elif l["where"] == "screen":
                    self.assertIn(l["span"], it["lines"][l["shot"] - 1], it["id"])
                elif l["where"] == "flash":
                    self.assertIn(l["span"], it["flash"], it["id"])
                else:
                    self.assertIn(l["shot"], it["art"], it["id"])

    def test_compliant_items_explain_why(self):
        for it in self.items:
            if not it["labels"]:
                self.assertTrue(it.get("trap"), it["id"])

    def test_manifest_timeline(self):
        man = manifest()
        for it in self.items:
            m = man[it["id"]]
            self.assertEqual(m["frames"], 96)
            flashes = [r for r in m["timeline"] if r["kind"] == "flash"]
            self.assertEqual(len(flashes), 1 if it.get("flash") else 0, it["id"])
            for r in flashes:
                self.assertEqual(r["end"] - r["start"], 6)
                # 2 fps sampling always shows a 0.5 s sticker
                self.assertTrue(any(r["start"] <= i < r["end"] for i in PLANS["frames16"]), it["id"])

    def test_prompt_names_only_language_categories(self):
        zh = user_text("zh", "video", "x")
        self.assertIn("FLAG", zh)
        self.assertNotIn("WEIGHT", zh)
        en = user_text("en", "text", "x")
        self.assertIn("BEFORE_AFTER", en)
        self.assertNotIn("FLAG", en)


class Parsing(unittest.TestCase):
    def test_parse_answer(self):
        v, ok = parse_answer('```json\n{"violations": [{"category": "ABSOLUTE", "evidence": "最好"}]}\n```', "zh")
        self.assertTrue(ok)
        self.assertEqual([x["category"] for x in v], ["ABSOLUTE"])
        v, ok = parse_answer('{"violations": []}', "en")
        self.assertEqual((v, ok), ([], True))
        v, ok = parse_answer('{"violations": [{"category": "<CODE>", "evidence": ""}]}', "en")
        self.assertEqual(v, [])  # a copied placeholder is not a category
        v, ok = parse_answer("I think it is fine.", "en")
        self.assertFalse(ok)
        v, ok = parse_answer('{"violations": [{"category": "ALCOHOL", "evidence": "x"}]', "zh")  # missing brace
        self.assertEqual([x["category"] for x in v], ["ALCOHOL"])

    def test_parse_classifier(self):
        self.assertEqual(parse_cls('{"violations": ["MEDICAL", "ABSOLUTE"]}'), (["ABSOLUTE", "MEDICAL"], True))
        self.assertEqual(parse_cls('<think></think>{"violations": []}'), ([], True))
        self.assertIn("广告文案", cls_prompt("zh", "x"))

    def test_evidence_match(self):
        self.assertTrue(evidence_matches("广告中使用了“全网最耐摔的玻璃杯”", "全网最耐摔的玻璃杯"))
        self.assertTrue(evidence_matches("The best sunscreen, period", "The best sunscreen, period."))
        self.assertFalse(evidence_matches("画面中有人喝酒", "宇宙第一好用"))


class Scoring(unittest.TestCase):
    def test_gold_answers_score_full_marks(self):
        items = load_video_items()
        rows = [{"id": i["id"], "violations": [{"category": l["cat"], "evidence": l.get("span", "")} for l in i["labels"]],
                 "parsed": True, "seconds": 1.0} for i in items]
        s = score(items, rows)
        self.assertEqual(s["decision_ok"], 96)
        self.assertEqual(s["label_hit"], 72)
        self.assertEqual(s["exact"], 96)
        self.assertEqual(s.get("wrong_reject", 0), 0)
        self.assertEqual(s["ev_ok"], s["ev_checked"])

    def test_empty_answers_miss_everything(self):
        items = load_video_items()
        s = score(items, [{"id": i["id"], "violations": [], "parsed": True, "seconds": 1.0} for i in items])
        self.assertEqual(s["caught"], 0)
        self.assertEqual(s["decision_ok"], 32)


class TextTests(unittest.TestCase):
    def test_text_test_sets(self):
        rows = read_jsonl(ROOT / "data_public" / "text_tests.jsonl")
        c = Counter(r["split"] for r in rows)
        self.assertEqual(c["video-text"], 96)
        self.assertEqual(c["gr-dev"] + c["gr-test"] + c["gr-hard"] + c["gr-memory"], 168)
        for r in rows:
            for g in r["gold"]:
                self.assertIn(g, LANG_CATS[r["lang"]], r["id"])
            self.assertTrue(set(r["gold"]) <= set(ALL_CATS))


class Preregistration(unittest.TestCase):
    def test_frozen_files_unchanged(self):
        text = (ROOT / "eval" / "PREREGISTRATION.md").read_text(encoding="utf-8")
        frozen = dict(re.findall(r"^\| ((?:testset|lab|finetune|data_public)/[^ ]+) \| ([0-9a-f]{16}) \|$", text, re.M))
        frozen.update(re.findall(r"`((?:testset|lab|finetune)/[^`]+)` ([0-9a-f]{16})", text))  # Part 3 lists its hash inline
        self.assertGreater(len(frozen), 15)
        self.assertIn("lab/kuaimod.py", frozen)
        # only the "### Changes after the run" sections, each up to the next part
        declared = "\n".join(re.findall(r"^### Changes after the run.*?(?=^## |\Z)", text, re.M | re.S))
        self.assertIn("lab/score.py", declared)
        for f, h in frozen.items():
            data = (ROOT / f).read_bytes()
            lf = data.replace(b"\r\n", b"\n")
            # line endings do not count: a checkout may turn LF into CRLF or back
            got = {hashlib.sha256(x).hexdigest()[:16] for x in (data, lf, lf.replace(b"\n", b"\r\n"))}
            if h not in got:
                self.assertIn(f, declared, f"{f} changed after freeze and is not declared")


if __name__ == "__main__":
    unittest.main()
