"""Build the text test sets for the fine-tuning comparison.

Sources (none is ever used for training):
  gr-dev, gr-test, gr-hard, gr-memory   the public ad items of grounded-review (MIT, same author), written with
                                        fixed answers on 2026-11-13/14/16. Image ads become text: the caption, then the
                                        words drawn in the image. Labels that live only in the picture (a flag, someone
                                        drinking) cannot be seen in text and are removed; the item keeps its other labels.
  video-text                            the 96 video items of this project as text: caption, the four subtitles and the
                                        sticker. Footage-only labels are removed the same way.
  real                                  grounded-review's 46 real cases (UK ASA rulings, PRC SAMR cases). They quote the
                                        original ads, so they stay out of this repository: built only when
                                        GROUNDED_REVIEW_DIR points at a local copy, written to $POLICY_LAB_DATA/text_real.jsonl,
                                        and reported as totals only.

Output: data_public/text_tests.jsonl (one row per item: id, split, lang, text, gold, removed_visual, trap)
Run:    python finetune/make_testsets.py [--grounded-review PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lab.common import DATA, load_video_items  # noqa: E402

OUT = ROOT / "data_public" / "text_tests.jsonl"
LABEL_SHAPES = {"bottle", "liquor", "wine", "oil", "beer", "pills", "tube", "jar", "candle", "tumbler", "teaware", "phone",
                "card", "powerbank", "book", "box", "tea", "blender", "bag", "jacket", "sofa", "bedding"}


def gr_text(item: dict) -> str:
    """Caption, then the words in the image, as grounded-review's testset/load.py ad_text() lists them."""
    lines = []
    if item["kind"] == "image":
        spec = item["image"]
        lines = list(spec["lines"]) + list(spec.get("small", []))
        if spec["layout"] in {"poster", "banner"} and spec.get("shape") in LABEL_SHAPES and spec.get("brand"):
            lines.append(spec["brand"])
    return "\n".join([item["copy"]] + lines)


def row(item_id, split, lang, text, labels, trap):
    gold = sorted({l["cat"] for l in labels if "visual" not in l})
    removed = sorted({l["cat"] for l in labels if "visual" in l} - set(gold))
    return {"id": item_id, "split": split, "lang": lang, "text": text, "gold": gold, "removed_visual": removed, "trap": bool(trap)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grounded-review", default=os.environ.get("GROUNDED_REVIEW_DIR", str(ROOT.parent / "grounded-review")))
    a = ap.parse_args(argv)
    gr = Path(a.grounded_review) / "testset"
    rows = []
    for split, files in [("gr-dev", ["items_zh.yaml", "items_en.yaml"]), ("gr-test", ["items_zh.yaml", "items_en.yaml"]),
                         ("gr-hard", ["items_hard_zh.yaml", "items_hard_en.yaml"]),
                         ("gr-memory", ["items_memory_zh.yaml", "items_memory_en.yaml"])]:
        want = split.split("-", 1)[1]
        for fn in files:
            lang = "zh" if "_zh" in fn else "en"
            for it in yaml.safe_load((gr / fn).read_text(encoding="utf-8")):
                if it.get("split") != want:
                    continue
                rows.append(row(it["id"], split, lang, gr_text(it), it["labels"], it.get("trap")))
    for it in load_video_items():
        text = "\n".join([it["caption"]] + it["lines"] + ([it["flash"]] if it.get("flash") else []))
        labels = [dict(l, visual=l["visual"]) if l["where"] == "visual" else l for l in it["labels"]]
        rows.append(row(it["id"], "video-text", it["lang"], text, labels, it.get("trap") and not it["trap"].startswith(("普通商品介绍", "Plain product"))))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    real = []
    for fn in ["items_real_en.yaml", "items_real_zh.yaml"]:
        p = gr / fn
        if p.exists():
            lang = "zh" if "_zh" in fn else "en"
            for it in yaml.safe_load(p.read_text(encoding="utf-8")):
                real.append(row(it["id"], "real", lang, gr_text(it), it["labels"], it.get("trap")))
    if real:
        DATA.mkdir(parents=True, exist_ok=True)
        (DATA / "text_real.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in real), encoding="utf-8")
    from collections import Counter
    c = Counter((r["split"], r["lang"]) for r in rows)
    print(dict(c), "real:", len(real))
    print("violating:", sum(1 for r in rows if r["gold"]), "compliant:", sum(1 for r in rows if not r["gold"]),
          "items that lost a picture-only label:", sum(1 for r in rows if r["removed_visual"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
