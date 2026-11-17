"""Run every text classifier on every text test set.

Methods:
  qwen3-1.7b-zs / qwen3-0.6b-zs       the base models with the prompt, thinking off, greedy
  qwen3-1.7b-lora / qwen3-0.6b-lora   the same models with the LoRA adapters from train_lora.py
  mmbert                              the fine-tuned encoder from train_encoder.py
  haiku                               claude-haiku-5-5 with the same prompt (Message Batches API, thinking off),
                                      answer limit 120 tokens
  haiku-400                           the same with an answer limit of 400 tokens (added after the run: with 120,
                                      some answers reasoned first and were cut off before the list)
Test sets:
  public     data_public/text_tests.jsonl (grounded-review public items + this project's video items as text)
  real       $POLICY_LAB_DATA/text_real.jsonl (46 real cases; totals only)
  synth-val  $POLICY_LAB_DATA/synth/val.jsonl (held-out synthetic buckets)
  video-ocr  each video item's caption plus the OCR text of its 16 frames (2 fps): the cheap video pipeline.
             These rows are also written to results/video/ so they appear next to the video models.

Run: python finetune/predict.py --method qwen3-1.7b-lora [--sets public,real,synth-val,video-ocr]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from finetune.prompt import parse, prompt  # noqa: E402
from lab.common import ALL_CATS, DATA, LANG_CATS, RESULTS, load_video_items, read_jsonl, write_jsonl  # noqa: E402

BASES = {"qwen3-1.7b": "Qwen/Qwen3-1.7B", "qwen3-0.6b": "Qwen/Qwen3-0.6B"}


def test_rows(sets: list[str]) -> list[dict]:
    rows = []
    if "public" in sets:
        rows += [dict(r, set="public") for r in read_jsonl(ROOT / "data_public" / "text_tests.jsonl")]
    if "real" in sets:
        rows += [dict(r, set="real") for r in read_jsonl(DATA / "text_real.jsonl")]
    if "synth-val" in sets:
        rows += [dict(r, set="synth-val", id=f"sv-{i:04d}", split="synth-val", trap=r["kind"] == "TRAP")
                 for i, r in enumerate(read_jsonl(DATA / "synth" / "val.jsonl"))]
    if "video-ocr" in sets:
        from lab.frames import load_ocr, ocr_text

        cache = load_ocr()
        for it in load_video_items():
            text = "\n".join([it["caption"]] + ocr_text(it["id"], "frames16", cache))
            rows.append({"id": it["id"], "set": "video-ocr", "split": "video-ocr", "lang": it["lang"], "text": text,
                         "gold": sorted({l["cat"] for l in it["labels"]}), "trap": False})
    return rows


def run_causal(method: str, rows: list[dict], bs: int = 16) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    size, kind = method.rsplit("-", 1)
    base = BASES[size]
    tok = AutoTokenizer.from_pretrained(base, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
    if kind == "lora":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(DATA / "lora" / f"{size}-lora"))
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        texts = [tok.apply_chat_template([{"role": "user", "content": prompt(r["lang"], r["text"])}], add_generation_prompt=True,
                                         tokenize=False, enable_thinking=False) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        torch.cuda.synchronize()
        s = time.time()
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        torch.cuda.synchronize()
        per = (time.time() - s) / len(chunk)
        for r, g in zip(chunk, gen[:, enc["input_ids"].shape[1]:]):
            raw = tok.decode(g, skip_special_tokens=True)
            pred, ok = parse(raw)
            out.append({"pred": [c for c in pred if c in LANG_CATS[r["lang"]]], "parsed": ok, "raw": raw, "seconds": round(per, 4)})
    return out


def run_encoder(rows: list[dict], bs: int = 32) -> list[dict]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = DATA / "encoder" / "mmbert-base"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).cuda().eval()
    out = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        enc = tok([r["text"] for r in chunk], padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        s = time.time()
        with torch.inference_mode():
            p = torch.sigmoid(model(**enc).logits.float()).cpu()
        torch.cuda.synchronize()
        per = (time.time() - s) / len(chunk)
        for r, pr in zip(chunk, p):
            pred = [c for k, c in enumerate(ALL_CATS) if pr[k] > 0.5 and c in LANG_CATS[r["lang"]]]
            out.append({"pred": pred, "parsed": True, "raw": "", "seconds": round(per, 4),
                        "probs": {c: round(float(pr[k]), 3) for k, c in enumerate(ALL_CATS) if c in LANG_CATS[r["lang"]]}})
    return out


def run_haiku(rows: list[dict], max_tokens: int = 120, tag: str = "text-haiku") -> list[dict]:
    from lab.claude import params, run_batch

    reqs = {f"{r['set']}__{r['id']}": params([{"type": "text", "text": prompt(r["lang"], r["text"])}], max_tokens=max_tokens) for r in rows}
    res = run_batch(tag, reqs)
    out = []
    for r in rows:
        x = res.get(f"{r['set']}__{r['id']}")
        if not x:
            out.append({"pred": [], "parsed": False, "raw": "", "seconds": None})
            continue
        pred, ok = parse(x["text"])
        out.append({"pred": [c for c in pred if c in LANG_CATS[r["lang"]]], "parsed": ok, "raw": x["text"], "seconds": None,
                    "input_tokens": x["input_tokens"], "output_tokens": x["output_tokens"]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--sets", default="public,real,synth-val,video-ocr")
    a = ap.parse_args(argv)
    rows = test_rows(a.sets.split(","))
    if a.method == "mmbert":
        preds = run_encoder(rows)
    elif a.method == "haiku":
        preds = run_haiku(rows)
    elif a.method == "haiku-400":  # added after the run: same prompt, answer limit 400 tokens instead of 120
        preds = run_haiku(rows, max_tokens=400, tag="text-haiku-400")
    else:
        preds = run_causal(a.method, rows)
    outp = RESULTS / "text" / f"{a.method}.jsonl"
    old = [r for r in read_jsonl(outp) if r["set"] not in a.sets.split(",")]
    new = [{"method": a.method, "set": r["set"], "id": r["id"], "split": r["split"], "lang": r["lang"], **p} for r, p in zip(rows, preds)]
    write_jsonl(outp, old + new, mode="w")
    vid = [{"model": a.method, "condition": "ocr16-text-classifier", "id": r["id"], "lang": r["lang"],
            "violations": [{"category": c, "evidence": ""} for c in p["pred"]], "parsed": p["parsed"], "raw": p["raw"],
            "seconds": p["seconds"]} for r, p in zip(rows, preds) if r["set"] == "video-ocr"]
    if vid:
        write_jsonl(RESULTS / "video" / f"{a.method}-ocr16.jsonl", vid, mode="w")
    print(f"{a.method}: {len(new)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
