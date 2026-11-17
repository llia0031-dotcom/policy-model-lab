"""LoRA fine-tuning of a small Qwen3 model to answer the classification prompt (finetune/prompt.py).

Input: the synthetic training set ($POLICY_LAB_DATA/synth/train.jsonl). Each example is the prompt with the ad
text, and the answer is the JSON list of its categories. Loss is computed on the answer tokens only.
Settings: LoRA rank 16, alpha 32, dropout 0.05 on every linear layer of the language model; AdamW, learning
rate 2e-4 with 5% warm-up and cosine decay, effective batch 16, 3 epochs, bf16, maximum 512 tokens, seed 0.
The synthetic validation set is only used to report loss after each epoch; the last epoch is kept.

Run: python finetune/train_lora.py --base Qwen/Qwen3-1.7B --out qwen3-1.7b-lora
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from finetune.prompt import prompt, target  # noqa: E402
from lab.common import DATA, read_jsonl  # noqa: E402


def encode(tok, ex, max_len=512):
    msgs = [{"role": "user", "content": prompt(ex["lang"], ex["text"])}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    a = target(ex["gold"]) + tok.eos_token
    pi = tok(p, add_special_tokens=False)["input_ids"]
    ai = tok(a, add_special_tokens=False)["input_ids"]
    ids = (pi + ai)[:max_len]
    labels = ([-100] * len(pi) + ai)[:max_len]
    return ids, labels


def batches(data, bs, pad_id, shuffle, rnd):
    idx = list(range(len(data)))
    if shuffle:
        rnd.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [data[j] for j in idx[i:i + bs]]
        L = max(len(x[0]) for x in chunk)
        ids = torch.full((len(chunk), L), pad_id)
        lab = torch.full((len(chunk), L), -100)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for k, (a, b) in enumerate(chunk):
            ids[k, :len(a)] = torch.tensor(a)
            lab[k, :len(b)] = torch.tensor(b)
            att[k, :len(a)] = 1
        yield ids, att, lab


def main(argv=None) -> int:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    rnd = random.Random(0)
    tok = AutoTokenizer.from_pretrained(a.base)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    train = [encode(tok, ex) for ex in read_jsonl(DATA / "synth" / "train.jsonl")]
    val = [encode(tok, ex) for ex in read_jsonl(DATA / "synth" / "val.jsonl")]
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = a.epochs * math.ceil(len(train) / (a.micro * a.accum))
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    log = {"base": a.base, "train_examples": len(train), "val_examples": len(val), "trainable_params": trainable,
           "total_params": total, "optimizer_steps": steps, "epochs": []}
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    step = 0
    for ep in range(a.epochs):
        model.train()
        run_loss, n = 0.0, 0
        for k, (ids, att, lab) in enumerate(batches(train, a.micro, pad, True, rnd)):
            out = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda())
            (out.loss / a.accum).backward()
            run_loss += out.loss.item()
            n += 1
            if (k + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for ids, att, lab in batches(val, 8, pad, False, rnd):
                vl += model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda()).loss.item()
                vn += 1
        log["epochs"].append({"epoch": ep + 1, "train_loss": round(run_loss / max(n, 1), 4), "val_loss": round(vl / max(vn, 1), 4),
                              "minutes": round((time.time() - t0) / 60, 1)})
        print(log["epochs"][-1], flush=True)
    log["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    log["minutes"] = round((time.time() - t0) / 60, 1)
    out_dir = DATA / "lora" / a.out
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (ROOT / "results" / "train").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "train" / f"{a.out}.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(json.dumps(log, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
