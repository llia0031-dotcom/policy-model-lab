"""The text-classification prompt. The same prompt is used for zero-shot runs, for fine-tuning, and for the
fine-tuned models at test time, so the three are compared on identical input."""
from __future__ import annotations

import json
import re

from lab.common import ALL_CATS, CAT_TEXT, LANG_CATS

TEMPLATE = {
    "zh": ("你在审核一条广告文案是否违反广告政策。只按下面列出的违规类别判断。\n{cats}\n"
           "按广告对所推销商品实际说了什么来判断；同一个词放在别的语境里可能并不违规。\n"
           '只输出一个 JSON 对象：{{"violations": ["<类别代码>"]}}，只列确实违反的类别；不违规时输出 {{"violations": []}}。\n\n'
           "广告文案：\n{text}"),
    "en": ("You are reviewing an ad's text against advertising policy. Judge only by the violation categories below.\n{cats}\n"
           "Judge by what the ad actually claims about the product; the same word can be allowed in another context.\n"
           'Output one JSON object: {{"violations": ["<CODE>"]}}, listing only categories that are actually violated; '
           'if nothing is violated, output {{"violations": []}}.\n\n'
           "Ad text:\n{text}"),
}


def prompt(lang: str, text: str) -> str:
    cats = "\n".join(f"- {c}: {CAT_TEXT[lang][c]}" for c in LANG_CATS[lang])
    return TEMPLATE[lang].format(cats=cats, text=text)


def target(labels: list[str]) -> str:
    return json.dumps({"violations": sorted(labels)}, ensure_ascii=False)


def parse(text: str) -> tuple[list[str], bool]:
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", t, re.S)
    cats: list[str] = []
    ok = False
    if m:
        try:
            obj = json.loads(m.group(0))
            vs = obj.get("violations", []) if isinstance(obj, dict) else []
            for v in vs if isinstance(vs, list) else []:
                c = (v.get("category", "") if isinstance(v, dict) else str(v)).strip().upper()
                if c in ALL_CATS and c not in cats:
                    cats.append(c)
            ok = True
        except json.JSONDecodeError:
            pass
    if not ok:
        found = [c for c in re.findall(r"[A-Z_]{4,}", t) if c in ALL_CATS]
        cats = list(dict.fromkeys(found))
        ok = '"violations"' in t
    return sorted(cats), ok
