"""Run a local vision-language model on the video test items under each input condition.

Conditions (what the model is given besides the post caption):
  text      nothing else
  cover     the first frame
  frames8   8 frames, one per second, as separate images
  video     16 frames at 2 fps, as one video with timestamps
  ocr8      the text OCR read from the 8 frames (no images)
  ocr16     the text OCR read from the 16 frames (no images)

Models:
  qwen3vl-2b   Qwen/Qwen3-VL-2B-Instruct, bf16
  qwen35-4b    Qwen/Qwen3.5-4B, 4-bit NF4 (bitsandbytes), thinking turned off

Decoding is greedy (do_sample=False), so a rerun gives the same answer. Results are appended to
results/video/<model>.jsonl; finished (condition, item) pairs are skipped, so a run can be resumed.

Run: python -m lab.local_vlm --model qwen3vl-2b [--conditions text,cover,frames8,video,ocr8,ocr16] [--limit N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

from lab.common import RESULTS, load_video_items, parse_answer, read_jsonl, user_text, write_jsonl
from lab.frames import PLANS, decode, load_ocr, ocr_text

MODELS = {
    "qwen3vl-2b": {"id": "Qwen/Qwen3-VL-2B-Instruct", "quant": False, "template": {}},
    "qwen35-4b": {"id": "Qwen/Qwen3.5-4B", "quant": True, "template": {"enable_thinking": False}},
}
CONDITIONS = ["text", "cover", "frames8", "video", "ocr8", "ocr16"]
MAX_NEW = 200


def load(model_key: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    spec = MODELS[model_key]
    kw = dict(dtype=torch.bfloat16, device_map="cuda")
    if spec["quant"]:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    proc = AutoProcessor.from_pretrained(spec["id"])
    model = AutoModelForImageTextToText.from_pretrained(spec["id"], **kw).eval()
    return proc, model


def build(proc, model_key: str, item: dict, cond: str, ocr_cache: dict):
    lang = item["lang"]
    ocr_lines = None
    if cond in ("ocr8", "ocr16"):
        ocr_lines = ocr_text(item["id"], "frames8" if cond == "ocr8" else "frames16", ocr_cache)
    content, extra = [], {}
    if cond == "cover":
        content.append({"type": "image"})
        extra["images"] = decode(item["id"], PLANS["cover"])
    elif cond == "frames8":
        content += [{"type": "image"}] * 8
        extra["images"] = decode(item["id"], PLANS["frames8"])
    elif cond == "video":
        content.append({"type": "video"})
        extra["videos"] = [np.stack(decode(item["id"], PLANS["video"]))]
        extra["video_metadata"] = [{"fps": 12.0, "total_num_frames": 96, "frames_indices": PLANS["video"], "duration": 8.0}]
        extra["do_sample_frames"] = False
    content.append({"type": "text", "text": user_text(lang, cond, item["caption"], ocr_lines)})
    chat = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                    tokenize=False, **MODELS[model_key]["template"])
    return proc(text=[chat], return_tensors="pt", **extra).to("cuda")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    a = ap.parse_args(argv)
    out = RESULTS / "video" / (f"{a.model}.jsonl" if a.split == "test" else f"dev-{a.model}.jsonl")
    done = {(r["condition"], r["id"]) for r in read_jsonl(out)}
    items = load_video_items(split=a.split)
    if a.limit:
        items = items[: a.limit // 2] + [i for i in items if i["lang"] == "en"][: a.limit - a.limit // 2]
    conds = a.conditions.split(",")
    todo = [(c, it) for c in conds for it in items if (c, it["id"]) not in done]
    if not todo:
        print("nothing to do")
        return 0
    t0 = time.time()
    proc, model = load(a.model)
    print(f"loaded {MODELS[a.model]['id']} in {time.time() - t0:.1f} s", flush=True)
    ocr_cache = load_ocr()
    for n, (cond, it) in enumerate(todo, 1):
        inputs = build(proc, a.model, it, cond, ocr_cache)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        s = time.time()
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
        torch.cuda.synchronize()
        secs = time.time() - s
        new = gen[:, inputs["input_ids"].shape[1]:]
        raw = proc.batch_decode(new, skip_special_tokens=True)[0]
        vs, ok = parse_answer(raw, it["lang"])
        write_jsonl(out, [{
            "model": a.model, "model_id": MODELS[a.model]["id"], "condition": cond, "id": it["id"], "lang": it["lang"],
            "violations": vs or [], "parsed": ok, "raw": raw, "seconds": round(secs, 3),
            "input_tokens": int(inputs["input_ids"].shape[1]), "output_tokens": int(new.shape[1]),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }])
        if n % 20 == 0 or n == len(todo):
            print(f"{n}/{len(todo)} {cond} {it['id']} {secs:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
