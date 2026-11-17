"""Real-data check on the KuaiMod benchmark (https://huggingface.co/datasets/KuaiMod/KuaiMod).

1,000 real short videos from a short-video platform, each with a cover and up to 8 frames, the title, OCR text
and speech transcript, labelled "正向" (fine) or one of 15 violation types. Only the binary question is asked
here: does the video violate the platform's community rules? A fixed random sample of 300 videos (seed
20261117, same fine/violating ratio as the full set) is used.

Inputs: text (title + OCR + transcript), frames (cover and frames + title), frames+text (all of it).
Models: Qwen/Qwen3.5-4B (NF4, thinking off; text and frames+text) and claude-haiku-5-5 (all three).

The data and the raw answers stay outside the repository ($POLICY_LAB_DATA/kuaimod/). Only totals are written
to results/kuaimod_summary.md: no frame, title or quotation of a video is published.

Run: python -m lab.kuaimod sample | local --conditions text,frames+text | haiku | score
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict

from lab.common import DATA, RESULTS, read_jsonl, write_jsonl

KM = DATA / "kuaimod"
IMG_ROOT = KM / "extracted" / "VLM_CoT" / "kuaimod"
SAMPLE = KM / "sample300.json"
OUT = KM / "answers"
FINE = "正向"
TYPES = ["色情低俗", "涉政不当言行", "不实信息", "辱骂、引战、宣扬仇恨", "违法犯罪", "引导私下交易", "历史虚无主义", "未成年不适当行为",
         "盗用抄袭作品", "夸大商品功效", "炒作卖货", "未添加AI标识", "虚假承诺福利", "危害未成年人身心健康", "出售违禁品"]
LOCAL_LONG_SIDE = 640  # added after the freeze, see eval/PREREGISTRATION.md, Part 3 changes
COMMERCE = {"引导私下交易", "夸大商品功效", "炒作卖货", "虚假承诺福利", "出售违禁品"}

INSTR = ("你在审核一条短视频是否违反短视频平台的社区规范。违规类型包括：" + "、".join(TYPES) + "。\n"
         "只输出一个 JSON 对象，不要输出其他文字：{\"violating\": true 或 false, \"type\": \"<违规类型；不违规时为空字符串>\"}")
CONTEXT = {"text": "下面是这条视频的标题、画面文字识别结果和语音转写（没有画面）。",
           "frames": "上面是这条视频的封面和抽帧，下面是标题。",
           "frames+text": "上面是这条视频的封面和抽帧，下面是标题、画面文字识别结果和语音转写。"}


def rows() -> list[dict]:
    return [json.loads(l) for l in (KM / "test" / "benchmark.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def sample() -> int:
    rs = rows()
    rnd = random.Random(20261117)
    fine = [r["pid"] for r in rs if r["tag"] == FINE]
    bad = [r["pid"] for r in rs if r["tag"] != FINE]
    n_fine = round(300 * len(fine) / len(rs))
    pick = sorted(rnd.sample(fine, n_fine) + rnd.sample(bad, 300 - n_fine))
    SAMPLE.write_text(json.dumps(pick), encoding="utf-8")
    sel = [r for r in rs if r["pid"] in set(pick)]
    print(len(pick), Counter("fine" if r["tag"] == FINE else "violating" for r in sel), Counter(r["tag"] for r in sel).most_common())
    return 0


def text_part(r: dict, cond: str) -> str:
    parts = [CONTEXT[cond], f"标题：{r['title']}"]
    if cond != "frames":
        parts += [f"画面文字：{r['OCR'] or '（无）'}", f"语音转写：{r['ASR'] or '（无）'}"]
    return "\n".join(parts) + "\n\n" + INSTR


def images(r: dict, long_side: int | None = None):
    """The video's cover and frames. long_side scales larger images down (the local model: 9 frames at up to
    1080x1440 do not fit in 8 GB; most KuaiMod frames are 360x640 and are left as they are)."""
    from PIL import Image

    out = []
    for p in r["images"]:
        im = Image.open(IMG_ROOT / p).convert("RGB")
        if long_side and max(im.size) > long_side:
            k = long_side / max(im.size)
            im = im.resize((round(im.width * k), round(im.height * k)), Image.BICUBIC)
        out.append(im)
    return out


def parse(text: str) -> tuple[bool | None, str]:
    import re

    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            o = json.loads(m.group(0))
            v = o.get("violating")
            if isinstance(v, str):
                v = v.strip().lower() in ("true", "是", "yes")
            return (bool(v) if v is not None else None), str(o.get("type", ""))
        except json.JSONDecodeError:
            pass
    t = (text or "").lower()
    if '"violating": true' in t or "violating: true" in t:
        return True, ""
    if '"violating": false' in t or "violating: false" in t:
        return False, ""
    return None, ""


def local(conds: list[str]) -> int:
    import torch

    from lab.local_vlm import MODELS, load

    pick = set(json.loads(SAMPLE.read_text(encoding="utf-8")))
    rs = [r for r in rows() if r["pid"] in pick]
    out = OUT / "qwen35-4b.jsonl"
    done = {(x["condition"], x["pid"]) for x in read_jsonl(out)}
    proc, model = load("qwen35-4b")
    for cond in conds:
        for n, r in enumerate(rs, 1):
            if (cond, r["pid"]) in done:
                continue
            content = ([{"type": "image"}] * len(r["images"]) if cond != "text" else []) + [{"type": "text", "text": text_part(r, cond)}]
            chat = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False,
                                            **MODELS["qwen35-4b"]["template"])
            kw = {"images": images(r, LOCAL_LONG_SIDE)} if cond != "text" else {}
            inputs = proc(text=[chat], return_tensors="pt", **kw).to("cuda")
            s = time.time()
            with torch.inference_mode():
                g = model.generate(**inputs, max_new_tokens=60, do_sample=False)
            raw = proc.batch_decode(g[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            v, t = parse(raw)
            write_jsonl(out, [{"model": "qwen35-4b", "condition": cond, "pid": r["pid"], "violating": v, "type": t, "raw": raw,
                               "seconds": round(time.time() - s, 3)}])
            if n % 50 == 0:
                print(cond, n, flush=True)
    return 0


def haiku(conds: list[str]) -> int:
    import io

    import numpy as np

    from lab.claude import jpeg_block, params, run_batch

    pick = set(json.loads(SAMPLE.read_text(encoding="utf-8")))
    rs = [r for r in rows() if r["pid"] in pick]
    reqs, meta = {}, {}
    for cond in conds:
        for r in rs:
            blocks = [jpeg_block(np.asarray(im)) for im in images(r)] if cond != "text" else []
            cid = f"{cond.replace('+', '_')}__{r['pid']}"
            reqs[cid] = params(blocks + [{"type": "text", "text": text_part(r, cond)}], max_tokens=120)
            meta[cid] = (cond, r["pid"])
    res = run_batch("kuaimod-haiku", reqs)
    out = OUT / "haiku.jsonl"
    rows_out = []
    for cid, (cond, pid) in meta.items():
        x = res.get(cid)
        v, t = parse(x["text"]) if x else (None, "")
        rows_out.append({"model": "haiku", "condition": cond, "pid": pid, "violating": v, "type": t, "raw": x["text"] if x else "",
                         "input_tokens": x["input_tokens"] if x else None})
    write_jsonl(out, rows_out, mode="w")
    return 0


def score() -> int:
    truth = {r["pid"]: r["tag"] for r in rows()}
    pick = json.loads(SAMPLE.read_text(encoding="utf-8"))
    lines = ["# KuaiMod real-data check: totals only", "",
             f"Fixed random sample of {len(pick)} of the 1,000 KuaiMod test videos (seed 20261117): "
             f"{sum(truth[p] == FINE for p in pick)} fine, {sum(truth[p] != FINE for p in pick)} violating. "
             "Question asked: does the video violate the platform's community rules (yes/no). No frame, title or quotation is shown here.", "",
             "| model | input | accuracy | violating caught | fine wrongly flagged | commerce-type violations caught | F1 (violating) | unparsed |",
             "|---|---|---|---|---|---|---|---|"]
    from lab.score import wilson

    for f in sorted(OUT.glob("*.jsonl")):
        by = defaultdict(list)
        for x in read_jsonl(f):
            by[(x["model"], x["condition"])].append(x)
        for (m, c), xs in sorted(by.items()):
            n = len(xs)
            tp = sum(1 for x in xs if truth[x["pid"]] != FINE and x["violating"] is True)
            fn = sum(1 for x in xs if truth[x["pid"]] != FINE and x["violating"] is not True)
            fp = sum(1 for x in xs if truth[x["pid"]] == FINE and x["violating"] is True)
            tn = n - tp - fn - fp
            com = [x for x in xs if truth[x["pid"]] in COMMERCE]
            comc = sum(1 for x in com if x["violating"] is True)
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            lo, hi = wilson(tp + tn, n)
            lines.append(f"| {m} | {c} | {tp + tn}/{n} ({100 * (tp + tn) / n:.0f}%) [{100*lo:.0f}–{100*hi:.0f}] | {tp}/{tp + fn} ({100 * tp / max(1, tp + fn):.0f}%) | "
                         f"{fp}/{fp + tn} ({100 * fp / max(1, fp + tn):.0f}%) | {comc}/{len(com)} | {f1:.3f} | {sum(1 for x in xs if x['violating'] is None)} |")
    md = "\n".join(lines) + "\n"
    (RESULTS / "kuaimod_summary.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sample", "local", "haiku", "score"])
    ap.add_argument("--conditions", default="")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step == "sample":
        return sample()
    if a.step == "local":
        return local((a.conditions or "text,frames+text").split(","))
    if a.step == "haiku":
        return haiku((a.conditions or "text,frames,frames+text").split(","))
    return score()


if __name__ == "__main__":
    sys.exit(main())
