"""Shared paths, test items, categories, prompts and answer parsing.

Large files (videos, frames, model weights, KuaiMod data) live under POLICY_LAB_DATA (default ./data),
which is not part of the repository. Results that the reports are built from live in results/.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("POLICY_LAB_DATA", ROOT / "data"))
VIDEOS = DATA / "videos"
RESULTS = ROOT / "results"
TESTSET = ROOT / "testset"

LANG_CATS = {
    "zh": ["ABSOLUTE", "MEDICAL", "GUARANTEE", "DISPARAGE", "DATA", "ALCOHOL", "FLAG"],
    "en": ["ABSOLUTE", "MEDICAL", "GUARANTEE", "DISPARAGE", "ALCOHOL", "WEIGHT", "BEFORE_AFTER"],
}
ALL_CATS = ["ABSOLUTE", "MEDICAL", "GUARANTEE", "DISPARAGE", "DATA", "ALCOHOL", "WEIGHT", "BEFORE_AFTER", "FLAG"]

# Category wording follows grounded-review's policies/clauses.yaml (same author, MIT).
CAT_TEXT = {
    "zh": {
        "ABSOLUTE": "绝对化用语：用“最”“第一”“顶级”“国家级”等词说所推销的商品或服务",
        "MEDICAL": "非医疗商品宣称预防、治疗疾病，或保健食品宣称疾病预防、治疗功能",
        "GUARANTEE": "对考试、升学、投资收益、经营收益或效果作保证性承诺",
        "DISPARAGE": "贬低其他生产经营者的商品或服务",
        "DATA": "使用数据、统计资料、调查结果、引用语，却没有表明出处",
        "ALCOHOL": "酒类广告：诱导、怂恿饮酒，出现饮酒的动作，或明示、暗示饮酒有消除紧张焦虑、增加体力等功效",
        "FLAG": "使用或变相使用中华人民共和国国旗、国歌、国徽、军旗等",
    },
    "en": {
        "ABSOLUTE": "Absolute or superlative claims about the product (best, No. 1, strongest)",
        "MEDICAL": "Claims that a product prevents, treats or cures a disease",
        "GUARANTEE": "Promised or guaranteed outcomes: exam results, investment returns, income, product effects",
        "DISPARAGE": "Running down other businesses' products or services",
        "ALCOHOL": "Alcohol ads: encouraging drinking, drinking shown, claimed benefits of drinking",
        "WEIGHT": "Weight-loss, muscle-gain and body-image claims",
        "BEFORE_AFTER": "Before-and-after comparisons of product results",
    },
}

INSTR = {
    "zh": (
        "你在审核一条短视频广告是否违反广告政策。只按下面列出的违规类别判断。\n"
        "{cats}\n"
        "按广告对所推销商品实际说了什么、画面实际展示了什么来判断；同一个词放在别的语境里可能并不违规。\n"
        "只列出确实违反的类别，每个违反的类别给一条证据（广告里的原话，或对画面的简短描述，不超过 30 个字）；没有违反的类别不要列出。\n"
        "只输出一个 JSON 对象，不要输出其他文字。格式：\n"
        '{{"violations": [{{"category": "<上面列表里的一个类别代码>", "evidence": "<证据>"}}]}}\n'
        '整条广告都不违规时，只输出 {{"violations": []}}。'
    ),
    "en": (
        "You are reviewing a short-video ad against advertising policy. Judge only by the violation categories below.\n"
        "{cats}\n"
        "Judge by what the ad actually claims about the product and what the footage actually shows; the same word can be allowed in another context.\n"
        "List only the categories the ad actually violates, each with one piece of evidence (the ad's exact words, or a short description of the footage, at most 20 words). Do not list categories that are not violated.\n"
        "Output one JSON object and nothing else. Format:\n"
        '{{"violations": [{{"category": "<one category code from the list above>", "evidence": "<evidence>"}}]}}\n'
        'If the ad violates nothing, output only {{"violations": []}}.'
    ),
}

CONTEXT = {  # what the model is given, per condition
    "zh": {
        "text": "只有文字，没有画面。",
        "cover": "上面是视频封面（第一帧）。",
        "frames8": "上面是从这条 8 秒视频里按时间均匀抽出的 8 帧（第 0.5、1.5……7.5 秒）。",
        "frames16": "上面是从这条 8 秒视频里每 0.5 秒抽出的 16 帧。",
        "video": "上面是整段视频（每秒 2 帧）。",
        "ocr8": "下面是从这条视频均匀抽出的 8 帧里识别出的文字，按时间顺序。",
        "ocr16": "下面是从这条视频每 0.5 秒一帧、共 16 帧里识别出的文字，按时间顺序。",
    },
    "en": {
        "text": "Text only; no footage.",
        "cover": "Above is the video's cover (first frame).",
        "frames8": "Above are 8 frames taken evenly from this 8-second video (at 0.5, 1.5 ... 7.5 s).",
        "frames16": "Above are 16 frames taken every 0.5 s from this 8-second video.",
        "video": "Above is the whole video (2 frames per second).",
        "ocr8": "Below is the text read from 8 frames taken evenly from this video, in time order.",
        "ocr16": "Below is the text read from 16 frames taken every 0.5 s from this video, in time order.",
    },
}
CAPTION_LABEL = {"zh": "发布文案：", "en": "Post caption: "}
OCR_LABEL = {"zh": "画面文字：", "en": "On-screen text: "}


def instruction(lang: str) -> str:
    cats = "\n".join(f"- {c}: {CAT_TEXT[lang][c]}" for c in LANG_CATS[lang])
    return INSTR[lang].format(cats=cats)


def user_text(lang: str, condition: str, caption: str, ocr_lines: list[str] | None = None) -> str:
    parts = [CONTEXT[lang][condition]]
    if ocr_lines is not None:
        parts.append(OCR_LABEL[lang] + (" / ".join(ocr_lines) if ocr_lines else ("（没有识别出文字）" if lang == "zh" else "(no text found)")))
    parts.append(CAPTION_LABEL[lang] + caption)
    parts.append(instruction(lang))
    return "\n\n".join(parts)


def load_video_items(lang: str | None = None, split: str = "test") -> list[dict]:
    """split "test": the 96 scored items; "dev": 4 pipeline-check items that are never scored."""
    items = []
    if split == "dev":
        items = yaml.safe_load((TESTSET / "items_video_dev.yaml").read_text(encoding="utf-8"))
        return [i for i in items if lang is None or i["lang"] == lang]
    for lg in (["zh", "en"] if lang is None else [lang]):
        for it in yaml.safe_load((TESTSET / f"items_video_{lg}.yaml").read_text(encoding="utf-8")):
            it["lang"] = lg
            items.append(it)
    return items


def manifest() -> dict:
    return json.loads((TESTSET / "video_manifest.json").read_text(encoding="utf-8"))


_JSON = re.compile(r"\{.*\}", re.S)


def parse_answer(text: str, lang: str) -> tuple[list[dict] | None, bool]:
    """Return (violations, ok). violations is a list of {category, evidence}; ok=False if no JSON could be read.
    Unknown category codes are dropped; a category named twice is kept once."""
    t = (text or "").strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    m = _JSON.search(t)
    if not m:
        return None, False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        # tolerate a trailing comma or a truncated tail
        s = re.sub(r",\s*([\]}])", r"\1", m.group(0))
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            cats = re.findall(r'"category"\s*:\s*"([A-Z_]+)"', t)
            if not cats and '"violations"' not in t:
                return None, False
            obj = {"violations": [{"category": c, "evidence": ""} for c in cats]}
    vs = obj.get("violations", []) if isinstance(obj, dict) else []
    out, seen = [], set()
    for v in vs if isinstance(vs, list) else []:
        if isinstance(v, str):
            v = {"category": v, "evidence": ""}
        if not isinstance(v, dict):
            continue
        c = str(v.get("category", "")).strip().upper()
        if c in ALL_CATS and c not in seen:
            seen.add(c)
            out.append({"category": c, "evidence": str(v.get("evidence", ""))})
    return out, True


def write_jsonl(path: Path, rows: list[dict], mode: str = "a") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
