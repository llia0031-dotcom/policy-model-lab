"""claude-haiku-5-5 on the video test items, as the large-model reference.

Claude takes images, not video, so the footage is sent as frames: the cover, 8 frames (1 per second) or 16
frames (2 per second). The prompt text is the same as for the local models (lab/common.py). Thinking is off,
as for the local models.

Run: python -m lab.claude_video [--conditions text,cover,frames8,frames16,ocr8] [--split test|dev]
"""
from __future__ import annotations

import argparse
import sys

from lab.claude import MODEL, jpeg_block, params, run_batch
from lab.common import RESULTS, load_video_items, parse_answer, read_jsonl, user_text, write_jsonl
from lab.frames import PLANS, decode, load_ocr, ocr_text


def content_for(item: dict, cond: str, ocr_cache: dict) -> list[dict]:
    ocr_lines = None
    blocks = []
    if cond in ("cover", "frames8", "frames16"):
        blocks = [jpeg_block(f) for f in decode(item["id"], PLANS[cond])]
    if cond in ("ocr8", "ocr16"):
        ocr_lines = ocr_text(item["id"], "frames8" if cond == "ocr8" else "frames16", ocr_cache)
    return blocks + [{"type": "text", "text": user_text(item["lang"], cond, item["caption"], ocr_lines)}]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="text,cover,frames8,frames16,ocr8")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    a = ap.parse_args(argv)
    items = load_video_items(split=a.split)
    ocr_cache = load_ocr()
    out = RESULTS / "video" / ("haiku.jsonl" if a.split == "test" else "dev-haiku.jsonl")
    done = {(r["condition"], r["id"]) for r in read_jsonl(out)}
    reqs, meta = {}, {}
    for cond in a.conditions.split(","):
        for it in items:
            if (cond, it["id"]) in done:
                continue
            cid = f"{cond}__{it['id']}"
            reqs[cid] = params(content_for(it, cond, ocr_cache))
            meta[cid] = (cond, it)
    if not reqs:
        print("nothing to do")
        return 0
    res = run_batch(f"video-{a.split}", reqs)
    rows = []
    for cid, (cond, it) in meta.items():
        r = res.get(cid)
        if not r:
            continue
        vs, ok = parse_answer(r["text"], it["lang"])
        rows.append({"model": "haiku", "model_id": MODEL, "condition": cond, "id": it["id"], "lang": it["lang"],
                     "violations": vs or [], "parsed": ok, "raw": r["text"], "seconds": None,
                     "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"]})
    write_jsonl(out, rows)
    print(f"wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
