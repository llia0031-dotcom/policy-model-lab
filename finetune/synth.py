"""Synthesise labelled ad copy for training the small classifiers.

Specs: language x target (one violation category, two categories, a compliant ad, or a compliant look-alike)
x product x writing style. Each request asks for 8 ads that carry exactly the target and nothing else from the
category list, with the exact violating words for every violation. The request text holds only the category
definitions; no test item and no example ad is shown. Generation runs through the Message Batches API
(claude-haiku-5-5).

Filters, in order (counts are written to data_public/synth_report.json):
  1. the answer parses and each ad has text; each ad's own labels use only categories of its language and every
     violating span appears verbatim in the ad. The ad's own labels are kept even when they differ from the
     request (a request for MEDICAL ads may come back with some compliant ads, which then train as compliant).
  2. language: Chinese ads contain Chinese characters, English ads contain none
  3. length 10-300 characters
  4. exact and near duplicates inside the training data (character 5-gram Jaccard >= 0.6) are dropped
  5. anything close to a test item is dropped: a shared verbatim run (12 characters in Chinese, 30 in English,
     where shorter runs are stock phrases such as "on the market"), or character 8-gram Jaccard >= 0.3,
     against every text test item (grounded-review public items, video items, and the real cases when present)
Split: 10% of the (product, style) buckets per language are held out as a synthetic validation set, so no
template seen in training appears there.

Run: python finetune/synth.py generate      (sends the batch, about USD 0.1)
     python finetune/synth.py build         (filters, splits -> $POLICY_LAB_DATA/synth/{train,val}.jsonl)
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lab.common import CAT_TEXT, DATA, LANG_CATS, read_jsonl  # noqa: E402

SYN = DATA / "synth"
REPORT = ROOT / "data_public" / "synth_report.json"
SEED = 20261117

PRODUCTS = {
    "zh": {
        "general": ["电动牙刷", "空气炸锅", "床垫", "收纳柜", "保温杯", "蓝牙耳机", "羽绒服", "跑鞋", "洗衣液", "扫地机器人",
                    "台灯", "行李箱", "沙发", "护手霜", "防晒霜", "面膜", "洗发水", "零食礼盒", "咖啡豆", "宠物粮"],
        "health": ["益生菌固体饮料", "代餐奶昔", "草本茶", "钙片（保健食品）", "鱼油软胶囊", "艾灸贴", "按摩椅", "足浴盆", "蜂蜜", "枸杞原浆"],
        "edu": ["考研辅导班", "英语四六级课程", "少儿编程课", "公务员考试培训", "职业资格证课程", "钢琴陪练"],
        "finance": ["理财产品", "奶茶店加盟", "基金定投服务", "民宿投资项目", "社区团购招商", "黄金积存"],
        "alcohol": ["白酒", "精酿啤酒", "葡萄酒", "米酒", "黄酒", "果酒"],
    },
    "en": {
        "general": ["electric toothbrush", "air fryer", "mattress", "storage shelf", "water bottle", "wireless earbuds",
                    "winter jacket", "running shoes", "laundry detergent", "robot vacuum", "desk lamp", "suitcase", "sofa",
                    "hand cream", "sunscreen", "face mask", "shampoo", "snack box", "coffee beans", "dog food"],
        "health": ["probiotic drink mix", "meal replacement shake", "herbal tea", "vitamin gummies", "fish oil capsules",
                   "massage gun", "foot spa", "manuka honey", "collagen powder", "sleep spray"],
        "edu": ["SAT prep course", "coding bootcamp", "language app subscription", "online MBA prep", "driving lessons", "guitar course"],
        "finance": ["trading app", "franchise opportunity", "crypto signals group", "real estate course", "side-hustle program", "savings account"],
        "alcohol": ["vodka", "craft beer", "red wine", "hard cider", "whiskey", "gin"],
    },
}
STYLES = {
    "zh": ["短视频标题", "直播间口播", "种草笔记", "商品详情页卖点", "短视频字幕（几句串起来）", "促销海报文案"],
    "en": ["short video title", "livestream pitch", "creator review post", "product page bullet points", "video subtitles (a few lines)", "promo banner"],
}
CAT_PRODUCTS = {  # which product groups make sense for each target
    "ABSOLUTE": ["general", "health", "edu", "finance", "alcohol"], "MEDICAL": ["health", "general"],
    "GUARANTEE": ["edu", "finance", "health"], "DISPARAGE": ["general", "health", "edu"], "DATA": ["general", "health", "edu"],
    "ALCOHOL": ["alcohol"], "WEIGHT": ["health", "general"], "BEFORE_AFTER": ["health", "general"], "FLAG": ["general", "alcohol"],
    "NONE": ["general", "health", "edu", "finance", "alcohol"], "TRAP": ["general", "health", "edu", "finance", "alcohol"],
}
PAIRS = {"zh": [("ABSOLUTE", "DATA"), ("MEDICAL", "ABSOLUTE"), ("GUARANTEE", "DATA"), ("DISPARAGE", "ABSOLUTE"), ("ALCOHOL", "ABSOLUTE")],
         "en": [("ABSOLUTE", "DISPARAGE"), ("MEDICAL", "ABSOLUTE"), ("GUARANTEE", "WEIGHT"), ("WEIGHT", "BEFORE_AFTER"), ("ALCOHOL", "ABSOLUTE")]}
# requests per target, per language (8 ads each)
N_REQ = {"single": 10, "pair": 2, "NONE": 15, "TRAP": 22}

ASK = {
    "zh": ("请写 8 条互不相同的中文广告文案，用于训练广告审核模型。\n"
           "违规类别定义：\n{cats}\n"
           "要求：\n- 商品：{product}；写法：{style}；品牌名自己编，不要用真实品牌。\n"
           "- {target}\n"
           "- 长度在 15 到 150 个字之间，彼此措辞、句式、长度都要有差别，违规内容出现的位置也要变化（开头、中间、结尾）。\n"
           "- 只输出 JSON 数组，每条是 {{\"text\": 广告文案, \"violations\": [{{\"category\": 类别代码, \"span\": 文案里构成该违规的原话}}]}}；"
           "span 必须一字不差地出现在 text 里。合规的条目 violations 为空数组。"),
    "en": ("Write 8 different English ad texts for training an ad-review model.\n"
           "Violation categories:\n{cats}\n"
           "Requirements:\n- Product: {product}; style: {style}; invent fictional brand names, no real brands.\n"
           "- {target}\n"
           "- Keep each ad between 10 and 60 words; vary wording, sentence shape and length, and vary where the violating words sit (start, middle, end).\n"
           "- Output only a JSON array; each element is {{\"text\": ad text, \"violations\": [{{\"category\": code, \"span\": the exact words in the text that make the violation}}]}}; "
           "span must appear verbatim in text. Compliant ads have an empty violations array."),
}
TARGET = {
    "zh": {
        "single": "每条恰好违反一个类别：{a}；不能同时违反列表里的其他类别。",
        "pair": "每条恰好违反两个类别：{a} 和 {b}；不能违反列表里的其他类别。",
        "NONE": "每条都完全合规，不违反列表里任何类别，就是普通的商品介绍。",
        "TRAP": "每条都合规，但要用到看起来像违规、按政策其实允许的说法，例如步骤序号、服务态度用语、比喻、写明出处和范围的数据、和自家旧款的对比、写明风险的理财介绍、只展示酒瓶不劝酒的酒类介绍；不违反列表里任何类别。",
    },
    "en": {
        "single": "Each ad violates exactly one category: {a}; it must not also violate any other category in the list.",
        "pair": "Each ad violates exactly two categories: {a} and {b}; no other category in the list.",
        "NONE": "Each ad is fully compliant with every category in the list: an ordinary product description.",
        "TRAP": "Each ad is compliant but uses wording that looks risky and is allowed under the policy, for example step numbers, figurative language, service slogans, statistics with a named source and scope, comparisons with the brand's own earlier model, investment descriptions that state the risk, or alcohol descriptions that show the product without encouraging drinking; it violates no category in the list.",
    },
}


def specs() -> list[dict]:
    rnd = random.Random(SEED)
    out = []
    for lang in ("zh", "en"):
        cats_text = "\n".join(f"- {c}: {CAT_TEXT[lang][c]}" for c in LANG_CATS[lang])
        plan = [(("single", c), N_REQ["single"]) for c in LANG_CATS[lang]]
        plan += [(("pair",) + p, N_REQ["pair"]) for p in PAIRS[lang]]
        plan += [(("NONE",), N_REQ["NONE"]), (("TRAP",), N_REQ["TRAP"])]
        for target, n in plan:
            groups = CAT_PRODUCTS[target[1]] if target[0] in ("single", "pair") else CAT_PRODUCTS[target[0]]
            for k in range(n):
                g = groups[k % len(groups)]
                product = rnd.choice(PRODUCTS[lang][g])
                style = rnd.choice(STYLES[lang])
                kind = target[0]
                t = TARGET[lang][kind].format(a=target[1] if len(target) > 1 else "", b=target[2] if len(target) > 2 else "")
                labels = list(target[1:]) if kind in ("single", "pair") else []
                sid = f"{lang}-{kind}-{'-'.join(labels) or 'x'}-{k:02d}"
                prompt = ASK[lang].format(cats=cats_text, product=product, style=style, target=t)
                out.append({"sid": sid, "lang": lang, "kind": kind, "labels": labels, "product": product, "style": style, "prompt": prompt})
    return out


def generate() -> int:
    from lab.claude import params, run_batch

    sp = specs()
    SYN.mkdir(parents=True, exist_ok=True)
    (SYN / "specs.json").write_text(json.dumps(sp, ensure_ascii=False, indent=1), encoding="utf-8")
    reqs = {s["sid"]: params([{"type": "text", "text": s["prompt"]}], max_tokens=3000) for s in sp}
    print(f"{len(reqs)} requests")
    run_batch("synth-train", reqs)
    return 0


def grams(s: str, n: int) -> set:
    s = re.sub(r"\s+", " ", s.lower())
    return {s[i:i + n] for i in range(max(len(s) - n + 1, 1))}


def build() -> int:
    from lab.common import RESULTS

    sp = {s["sid"]: s for s in json.loads((SYN / "specs.json").read_text(encoding="utf-8"))}
    log = {r["custom_id"]: r for r in read_jsonl(RESULTS / "claude_batches" / "synth-train.jsonl") if r["type"] == "succeeded"}
    stats = Counter()
    ads = []
    for sid, s in sp.items():
        r = log.get(sid)
        if not r:
            stats["request_missing"] += 1
            continue
        m = re.search(r"\[.*\]", r["text"], re.S)
        try:
            arr = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            arr = []
            stats["request_unparsed"] += 1
        for a in arr:
            stats["generated"] += 1
            text = str(a.get("text", "")).strip() if isinstance(a, dict) else ""
            vs = a.get("violations", []) if isinstance(a, dict) else []
            cats = sorted({str(v.get("category", "")).upper() for v in vs if isinstance(v, dict)})
            spans = [str(v.get("span", "")) for v in vs if isinstance(v, dict)]
            if not text:
                stats["drop_empty"] += 1
                continue
            if any(c not in LANG_CATS[s["lang"]] for c in cats):
                stats["drop_unknown_category"] += 1
                continue
            if any(not sp_ or sp_ not in text for sp_ in spans):
                stats["drop_span_not_verbatim"] += 1
                continue
            stats["label_differs_from_request"] += cats != sorted(s["labels"])
            has_cjk = bool(re.search(r"[一-鿿]", text))
            if has_cjk != (s["lang"] == "zh"):
                stats["drop_language"] += 1
                continue
            if not 10 <= len(text) <= 300:
                stats["drop_length"] += 1
                continue
            kind = s["kind"] if cats == sorted(s["labels"]) else ("NONE" if not cats else ("single" if len(cats) == 1 else "pair"))
            ads.append({"lang": s["lang"], "text": text, "gold": cats, "kind": kind,
                        "bucket": f"{s['lang']}|{s['product']}|{s['style']}", "sid": sid, "spans": spans})
    # near duplicates inside the synthetic set
    kept, seen = [], []
    for a in ads:
        g = grams(a["text"], 5)
        if any(len(g & h) / max(len(g | h), 1) >= 0.6 for h in seen):
            stats["drop_near_duplicate"] += 1
            continue
        seen.append(g)
        kept.append(a)
    # closeness to any test item
    tests = [r["text"] for r in read_jsonl(ROOT / "data_public" / "text_tests.jsonl")]
    tests += [r["text"] for r in read_jsonl(DATA / "text_real.jsonl")]
    t8 = [grams(t, 8) for t in tests]
    run = {"zh": 12, "en": 30}  # a shared verbatim run this long counts as copied (shorter English runs are stock phrases)
    trun = {lg: set().union(*[grams(t, n) for t in tests]) for lg, n in run.items()}
    final = []
    for a in kept:
        g8 = grams(a["text"], 8)
        if grams(a["text"], run[a["lang"]]) & trun[a["lang"]] or any(len(g8 & h) / max(len(g8 | h), 1) >= 0.3 for h in t8):
            stats["drop_close_to_test"] += 1
            continue
        final.append(a)
    # split by bucket
    rnd = random.Random(SEED)
    buckets = defaultdict(list)
    for a in final:
        buckets[a["bucket"]].append(a)
    val_buckets = set()
    for lang in ("zh", "en"):
        bs = sorted(b for b in buckets if b.startswith(lang))
        rnd.shuffle(bs)
        val_buckets |= set(bs[: max(1, round(0.1 * len(bs)))])
    train = [a for a in final if a["bucket"] not in val_buckets]
    val = [a for a in final if a["bucket"] in val_buckets]
    for name, rows in (("train", train), ("val", val)):
        (SYN / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    rep = dict(stats)
    rep.update(kept_train=len(train), kept_val=len(val),
               train_by_label=Counter("+".join(a["gold"]) or ("TRAP" if a["kind"] == "TRAP" else "NONE") for a in train),
               train_by_lang=Counter(a["lang"] for a in train),
               train_sha256=hashlib.sha256((SYN / "train.jsonl").read_bytes()).hexdigest()[:16],
               val_sha256=hashlib.sha256((SYN / "val.jsonl").read_bytes()).hexdigest()[:16])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit({"generate": generate, "build": build}[sys.argv[1]]())
