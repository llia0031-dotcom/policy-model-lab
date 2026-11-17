"""Encoder baseline: full fine-tuning of jhu-clsp/mmBERT-base (307M, MIT) as a multi-label classifier.

Same synthetic training set as the LoRA runs. Input is the ad text alone; output is one sigmoid per category.
Settings: AdamW, learning rate 3e-5, 10% warm-up then linear decay, batch 16, 4 epochs, maximum 256 tokens,
seed 0. At test time a category is predicted when its probability is above 0.5 and it belongs to the ad's
language (Chinese ads: 7 categories under PRC rules; English ads: 7 under TikTok's policies).

Run: python finetune/train_encoder.py
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lab.common import ALL_CATS, DATA, read_jsonl  # noqa: E402

BASE = "jhu-clsp/mmBERT-base"
OUT = DATA / "encoder" / "mmbert-base"


def main() -> int:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(0)
    rnd = random.Random(0)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=len(ALL_CATS),
                                                               problem_type="multi_label_classification").cuda()
    train = read_jsonl(DATA / "synth" / "train.jsonl")
    val = read_jsonl(DATA / "synth" / "val.jsonl")

    def batch(rows):
        enc = tok([r["text"] for r in rows], padding=True, truncation=True, max_length=256, return_tensors="pt")
        y = torch.zeros((len(rows), len(ALL_CATS)))
        for i, r in enumerate(rows):
            for c in r["gold"]:
                y[i, ALL_CATS.index(c)] = 1
        return {k: v.cuda() for k, v in enc.items()}, y.cuda()

    epochs, bs = 4, 16
    steps = epochs * math.ceil(len(train) / bs)
    warm = int(0.1 * steps)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    log = {"base": BASE, "train_examples": len(train), "epochs": []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        idx = list(range(len(train)))
        rnd.shuffle(idx)
        tl = 0.0
        for i in range(0, len(idx), bs):
            x, y = batch([train[j] for j in idx[i:i + bs]])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**x, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            tl += loss.item()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for i in range(0, len(val), 32):
                x, y = batch(val[i:i + 32])
                vl += model(**x, labels=y).loss.item()
        log["epochs"].append({"epoch": ep + 1, "train_loss": round(tl / math.ceil(len(train) / bs), 4),
                              "val_loss": round(vl / max(1, math.ceil(len(val) / 32)), 4), "minutes": round((time.time() - t0) / 60, 1)})
        print(log["epochs"][-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    log["minutes"] = round((time.time() - t0) / 60, 1)
    (ROOT / "results" / "train").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "train" / "mmbert-base.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
