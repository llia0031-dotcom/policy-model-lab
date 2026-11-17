"""Frame sampling and OCR for the test videos.

The videos are 8 s at 12 fps (96 frames). Sampling plans, by frame index:
  cover    [0]                       the first frame, which feeds most cover/thumbnail checks
  frames8  6, 18, ..., 90            one frame per second, at 0.5 s, 1.5 s ... 7.5 s
  frames16 3, 9, ..., 93             two frames per second, at 0.25 s, 0.75 s ... 7.75 s (also the native-video input)
OCR uses RapidOCR with the PP-OCRv6 medium models (as in grounded-review), on the GPU through onnxruntime-gpu,
and is cached in results/ocr_video.json.

Run: python -m lab.frames            (OCR every sampled frame once; prints the character recall of on-screen text)
"""
from __future__ import annotations

import json
import sys
import time
import unicodedata
from functools import lru_cache

import av
import numpy as np

from lab.common import RESULTS, VIDEOS, load_video_items, manifest

PLANS = {
    "cover": [0],
    "frames8": [6 + 12 * i for i in range(8)],
    "frames16": [3 + 6 * i for i in range(16)],
}
PLANS["video"] = PLANS["frames16"]
FPS = 12
OCR_CACHE = RESULTS / "ocr_video.json"


def decode(item_id: str, indices: list[int]) -> list[np.ndarray]:
    want = set(indices)
    out = {}
    with av.open(str(VIDEOS / f"{item_id}.mp4")) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i in want:
                out[i] = f.to_ndarray(format="rgb24")
            if i >= max(want):
                break
    return [out[i] for i in indices]


def captured(item_id: str, plan: str) -> dict:
    """For each on-screen text of the item: does any frame of the plan show it?"""
    tl = manifest()[item_id]["timeline"]
    idx = PLANS[plan]
    return {r["text"]: any(r["start"] <= i < r["end"] for i in idx) for r in tl}


@lru_cache(maxsize=1)
def engine():
    """PP-OCRv6 medium models; on the GPU through onnxruntime-gpu when POLICY_LAB_OCR_CUDA=1 (default).
    Importing torch first makes its bundled CUDA libraries available to onnxruntime."""
    import os

    from rapidocr import ModelType, RapidOCR

    cuda = os.environ.get("POLICY_LAB_OCR_CUDA", "1") == "1"
    if cuda:
        import torch  # noqa: F401  (loads the CUDA libraries onnxruntime-gpu needs)
    return RapidOCR(params={"Det.model_type": ModelType("medium"), "Rec.model_type": ModelType("medium"),
                            "EngineConfig.onnxruntime.use_cuda": cuda})


def ocr_rows(img: np.ndarray) -> list[str]:
    """Read one frame; group boxes into lines top to bottom, left to right (grounded-review agent/ocr.py)."""
    res = engine()(img)
    if res.boxes is None:
        return []
    boxes = []
    for box, txt in zip(res.boxes, res.txts):
        xs, ys = [float(p[0]) for p in box], [float(p[1]) for p in box]
        boxes.append({"text": txt, "x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)})
    lines: list[list[dict]] = []
    for b in sorted(boxes, key=lambda b: (b["y0"] + b["y1"]) / 2):
        cy, h = (b["y0"] + b["y1"]) / 2, b["y1"] - b["y0"]
        for row in lines:
            if min(x["y0"] for x in row) - 0.25 * h <= cy <= max(x["y1"] for x in row) + 0.25 * h:
                row.append(b)
                break
        else:
            lines.append([b])
    texts = []
    for row in lines:
        row.sort(key=lambda b: b["x0"])
        s = row[0]["text"]
        for prev, cur in zip(row, row[1:]):
            s += (" " if cur["x0"] - prev["x1"] > 0.3 * (cur["y1"] - cur["y0"]) else "") + cur["text"]
        texts.append(s)
    return texts


def load_ocr() -> dict:
    return json.loads(OCR_CACHE.read_text(encoding="utf-8")) if OCR_CACHE.exists() else {}


def ocr_text(item_id: str, plan: str, cache: dict | None = None) -> list[str]:
    """OCR lines of a plan's frames in time order, with repeats of the previous frame's lines removed."""
    cache = cache if cache is not None else load_ocr()
    out, prev = [], set()
    for i in PLANS[plan]:
        lines = cache[item_id][str(i)]["lines"]
        out += [l for l in lines if l not in prev]
        prev = set(lines)
    return out


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return "".join(ch for ch in s if ch.isalnum())


def main() -> int:
    cache = load_ocr()
    items = load_video_items() + load_video_items(split="dev")
    need = sorted(set(PLANS["frames8"] + PLANS["frames16"] + PLANS["cover"]))
    t0 = time.time()
    for it in items:
        if it["id"] in cache and all(str(i) in cache[it["id"]] for i in need):
            continue
        frames = decode(it["id"], need)
        cache[it["id"]] = {}
        for i, img in zip(need, frames):
            s = time.time()
            cache[it["id"]][str(i)] = {"lines": ocr_rows(img), "seconds": round(time.time() - s, 3)}
        print(it["id"], flush=True)
        OCR_CACHE.parent.mkdir(parents=True, exist_ok=True)
        OCR_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    # character recall of the subtitles and stickers that a frame of the plan actually showed
    for plan in ("frames8", "frames16"):
        hit = tot = 0
        for it in items:
            seen = _norm("".join(ocr_text(it["id"], plan, cache)))
            for text, was_shown in captured(it["id"], plan).items():
                if was_shown and not text.startswith("@"):
                    t = _norm(text)
                    tot += len(t)
                    hit += sum(1 for ch in t if ch in seen)
        print(f"{plan}: character recall of shown text {hit}/{tot} = {hit / max(tot, 1):.3f}")
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
