"""Contrastive baseline: SigLIP 2 zero-shot on video frames, no generative model.

Each frame of a plan (frames8 or frames16) is scored against one short English description per category of
the item's language and one description of an ordinary product ad. A category is predicted when, on at least
one frame, its probability is higher than the ordinary-ad description's probability on that frame (max over
frames, so a brief scene can count). The descriptions are fixed below and were written with the test items,
before any run. Only the footage is used: the caption and on-screen words are not read as text.

Run: python -m lab.siglip_frames [--plan frames8|frames16] [--split test|dev]
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

from lab.common import LANG_CATS, RESULTS, load_video_items, write_jsonl
from lab.frames import PLANS, decode

MODEL_ID = "google/siglip2-so400m-patch14-384"
PROMPTS = {
    "ABSOLUTE": "an advertisement claiming its product is the best in the world",
    "MEDICAL": "an advertisement claiming a product cures a disease",
    "GUARANTEE": "an advertisement promising guaranteed money or exam results",
    "DISPARAGE": "an advertisement insulting a competitor's product",
    "DATA": "an advertisement showing statistics and percentages",
    "ALCOHOL": "a person drinking alcohol from a glass",
    "WEIGHT": "an advertisement about losing weight fast",
    "BEFORE_AFTER": "a before and after comparison of skin or teeth",
    "FLAG": "the national flag of China",
}
NEUTRAL = "an ordinary product advertisement"


def main(argv=None) -> int:
    from transformers import AutoModel, AutoProcessor

    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="frames8", choices=["frames8", "frames16"])
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    a = ap.parse_args(argv)
    out = RESULTS / "video" / (f"siglip2-{a.plan}.jsonl" if a.split == "test" else f"dev-siglip2-{a.plan}.jsonl")
    if out.exists():
        out.unlink()
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda").eval()
    rows = []
    for it in load_video_items(split=a.split):
        cats = LANG_CATS[it["lang"]]
        texts = [PROMPTS[c] for c in cats] + [NEUTRAL]
        frames = decode(it["id"], PLANS[a.plan])
        s = time.time()
        with torch.inference_mode():
            inp = proc(text=texts, images=frames, padding="max_length", max_length=64, return_tensors="pt").to("cuda")
            logits = model(**inp).logits_per_image.float()  # frames x texts
        probs = torch.sigmoid(logits)
        above = (probs[:, :-1] > probs[:, -1:]).any(dim=0).tolist()
        pred = [{"category": c, "evidence": f"max p={probs[:, i].max():.3f}"} for i, c in enumerate(cats) if above[i]]
        rows.append({"model": "siglip2", "model_id": MODEL_ID, "condition": f"siglip-{a.plan}", "id": it["id"],
                     "lang": it["lang"], "violations": pred, "parsed": True, "raw": "",
                     "seconds": round(time.time() - s, 3),
                     "scores": {c: round(float(probs[:, i].max()), 4) for i, c in enumerate(cats + ["NEUTRAL"])}})
    write_jsonl(out, rows, mode="w")
    print(f"wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
