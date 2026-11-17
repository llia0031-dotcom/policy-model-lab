"""Render the short-video test items (testset/items_video_*.yaml) as 8-second vertical MP4 files.

Every video: 480x854, 12 fps, 4 shots of 2 s. Each shot shows a drawing (the product, or a scene named in
`art`), an account handle at the top and the shot's subtitle in a band near the bottom. An optional sticker
(`flash`) appears for 0.5 s (6 frames) at a start time drawn from a seed fixed by the item id. The post caption
is metadata and is not drawn. Because every word on screen is known, OCR and model answers can be checked
against the exact timeline written to testset/video_manifest.json.

Run:  python testset/render_video.py [--out DIR] [--only zv-01,ev-07] [--sheets]
      --sheets also writes a contact sheet (8 frames per video) for visual checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import av
import yaml
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_OUT = Path(os.environ.get("POLICY_LAB_DATA", ROOT / "data")) / "videos"

W, H, FPS, SHOT_S, N_SHOTS = 480, 854, 12, 2, 4
N_FRAMES = FPS * SHOT_S * N_SHOTS          # 96
FLASH_FRAMES = 6                           # 0.5 s
ART_BOX = (70, 190, 410, 600)
BAND = (0, 660, W, 760)

FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts"), Path("/System/Library/Fonts")]
FONT_FILES = {
    ("zh", "bold"): ["msyhbd.ttc", "NotoSansCJK-Bold.ttc", "PingFang.ttc"],
    ("zh", "regular"): ["msyh.ttc", "NotoSansCJK-Regular.ttc", "PingFang.ttc"],
    ("en", "bold"): ["arialbd.ttf", "DejaVuSans-Bold.ttf", "Helvetica.ttc"],
    ("en", "regular"): ["segoeui.ttf", "DejaVuSans.ttf", "Helvetica.ttc"],
}

PALETTES = {  # background, body, dark text, accent (same names as grounded-review)
    "blue": ("#dce9f7", "#8fb6e3", "#173b6c", "#2f6fb5"),
    "navy": ("#d9dee9", "#2b3a5c", "#ffffff", "#f2c14e"),
    "purple": ("#ece3f6", "#a58bcf", "#3b2160", "#7b4fb3"),
    "gray": ("#eeeeee", "#b9b9b9", "#222222", "#555555"),
    "green": ("#e3f2df", "#8cc084", "#1f4d1a", "#3c8a34"),
    "red": ("#fbe1dc", "#d9534f", "#7a1010", "#c0392b"),
    "orange": ("#fdebd6", "#f0a35e", "#6b3305", "#d9731a"),
    "pink": ("#fbe4ec", "#e79bb6", "#6b1f3a", "#c2456f"),
    "teal": ("#dff3f1", "#6cbfb5", "#0f4842", "#1f8a7f"),
    "gold": ("#fbf2d5", "#d9b44a", "#4a3505", "#a8801a"),
    "beige": ("#f5efe3", "#d8c7a6", "#4a3b22", "#8a6d3b"),
}
SKIN, SHIRTS = "#e8b99a", ["#3c6e9e", "#a8443c", "#4f7d3a", "#6b4f9e"]
CUPS = {  # drink -> (glass colour, kind)
    "drink_beer": ("#e8a317", "beer"), "drink_wine": ("#8e1b3a", "wine"), "drink_baijiu": ("#f1e6c8", "cup"),
    "drink_cheers": ("#e8a317", "beer"), "drink_tea": ("#b5853f", "cup"), "drink_water": ("#cfe8ff", "glass"),
    "drink_coffee": ("#6b4226", "mug"), "drink_juice": ("#f7a21b", "glass"),
}
SCENE_BOTTLE = {"drink_beer": "beer", "drink_wine": "wine", "drink_baijiu": "liquor", "drink_cheers": "liquor"}


# ---------------------------------------------------------------- fonts and text
_font_cache: dict = {}


def font(lang: str, weight: str, size: int) -> ImageFont.FreeTypeFont:
    key = (lang, weight, size)
    if key not in _font_cache:
        for name in FONT_FILES[(lang, weight)]:
            for d in FONT_DIRS:
                hits = list(d.rglob(name)) if d.exists() and d.name != "Fonts" else ([d / name] if (d / name).exists() else [])
                if hits:
                    _font_cache[key] = ImageFont.truetype(str(hits[0]), size)
                    break
            if key in _font_cache:
                break
        else:
            raise FileNotFoundError(f"no font for {lang}/{weight}; install Noto Sans CJK or DejaVu Sans")
    return _font_cache[key]


def wrap(draw, text, lang, weight, size, max_w, min_size=22):
    """Return (font, lines): one line if it fits at a size >= min_size, else two balanced lines."""
    s = size
    while s >= min_size:
        f = font(lang, weight, s)
        if draw.textlength(text, font=f) <= max_w:
            return f, [text]
        s -= 2
    f = font(lang, weight, size - 4)
    if lang == "en" and " " in text:
        words = text.split(" ")
        best = min(range(1, len(words)), key=lambda k: abs(draw.textlength(" ".join(words[:k]), font=f) - draw.textlength(" ".join(words[k:]), font=f)))
        lines = [" ".join(words[:best]), " ".join(words[best:])]
    else:
        k = math.ceil(len(text) / 2)
        lines = [text[:k], text[k:]]
    while any(draw.textlength(l, font=f) > max_w for l in lines) and f.size > 14:
        f = font(lang, weight, f.size - 2)
    return f, lines


def gradient(w, h, c1, c2):
    a, b = Image.new("RGB", (w, h), c1), Image.new("RGB", (w, h), c2)
    return Image.composite(b, a, Image.linear_gradient("L").resize((w, h)))


# ---------------------------------------------------------------- drawings (adapted from grounded-review)
def draw_product(d, shape, box, colors, brand, lang):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) // 2
    body, accent, dark = colors[1], colors[3], colors[2]
    if dark == "#ffffff":
        dark = "#1c2740"
    lf = font(lang, "bold", max(14, int(w * 0.11)))

    def label(y):
        tw = d.textlength(brand, font=lf)
        d.text((cx - tw / 2, y), brand, font=lf, fill="#ffffff")

    if shape in {"bottle", "liquor", "wine", "oil", "beer"}:
        neck_w = w * (0.16 if shape in {"wine", "beer", "liquor", "oil"} else 0.3)
        body_top = y0 + h * (0.35 if shape in {"wine", "beer"} else 0.22)
        d.rounded_rectangle([cx - neck_w / 2, y0, cx + neck_w / 2, body_top + 20], 8, fill=dark)
        d.rounded_rectangle([x0 + w * 0.18, body_top, x1 - w * 0.18, y1], 28, fill=accent)
        d.rectangle([x0 + w * 0.18, body_top + h * 0.25, x1 - w * 0.18, body_top + h * 0.42], fill=body)
        label(body_top + h * 0.27)
    elif shape == "tube":
        d.polygon([(x0 + w * 0.3, y0), (x1 - w * 0.3, y0), (x1 - w * 0.18, y1 - 40), (x0 + w * 0.18, y1 - 40)], fill=accent)
        d.rounded_rectangle([cx - w * 0.14, y1 - 50, cx + w * 0.14, y1], 10, fill=dark)
        label(y0 + h * 0.35)
    elif shape in {"jar", "candle"}:
        d.rounded_rectangle([x0 + w * 0.12, y0 + h * 0.18, x1 - w * 0.12, y0 + h * 0.32], 12, fill=dark)
        d.rounded_rectangle([x0 + w * 0.08, y0 + h * 0.3, x1 - w * 0.08, y1], 30, fill=accent)
        if shape == "candle":
            d.ellipse([cx - 10, y0 + h * 0.02, cx + 10, y0 + h * 0.16], fill="#f6b73c")
        label(y0 + h * 0.55)
    elif shape in {"tumbler", "teaware"}:
        d.rounded_rectangle([x0 + w * 0.28, y0, x1 - w * 0.28, y0 + h * 0.12], 10, fill=dark)
        d.polygon([(x0 + w * 0.25, y0 + h * 0.12), (x1 - w * 0.25, y0 + h * 0.12), (x1 - w * 0.3, y1), (x0 + w * 0.3, y1)], fill=accent)
        label(y0 + h * 0.45)
    elif shape in {"phone", "laptop", "card", "powerbank", "book"}:
        if shape == "laptop":
            d.rounded_rectangle([x0, y0 + h * 0.3, x1, y1 - h * 0.25], 14, fill=dark)
            d.rectangle([x0 + 12, y0 + h * 0.3 + 12, x1 - 12, y1 - h * 0.25 - 12], fill=body)
            d.rectangle([x0 - 16, y1 - h * 0.25, x1 + 16, y1 - h * 0.2], fill=accent)
        else:
            ratio = {"phone": 0.52, "card": 1.55, "powerbank": 0.62, "book": 0.75}[shape]
            bw = min(w * 0.9, h * ratio)
            bh = bw / ratio
            top = y0 + (h - bh) / 2
            d.rounded_rectangle([cx - bw / 2, top, cx + bw / 2, top + bh], 26 if shape != "book" else 6, fill=accent)
            if shape == "phone":
                d.rounded_rectangle([cx - bw / 2 + 12, top + 30, cx + bw / 2 - 12, top + bh - 30], 16, fill=body)
            if shape == "card":
                d.rectangle([cx - bw / 2, top + bh * 0.25, cx + bw / 2, top + bh * 0.4], fill=dark)
            label(top + bh * 0.55)
    elif shape in {"box", "tea", "blender", "vacuum"}:
        if shape == "vacuum":
            d.ellipse([x0, y0 + h * 0.3, x1, y1 - h * 0.1], fill=accent)
            d.ellipse([x0 + w * 0.3, y0 + h * 0.45, x1 - w * 0.3, y1 - h * 0.25], fill=dark)
        elif shape == "blender":
            d.polygon([(x0 + w * 0.2, y0), (x1 - w * 0.2, y0), (x1 - w * 0.28, y0 + h * 0.65), (x0 + w * 0.28, y0 + h * 0.65)], fill=body, outline=dark, width=6)
            d.rounded_rectangle([x0 + w * 0.2, y0 + h * 0.65, x1 - w * 0.2, y1], 16, fill=accent)
            label(y0 + h * 0.75)
        else:
            d.polygon([(x0 + w * 0.15, y0 + h * 0.15), (x1 - w * 0.05, y0 + h * 0.15), (x1 - w * 0.05, y1), (x0 + w * 0.15, y1)], fill=accent)
            d.polygon([(x0 + w * 0.15, y0 + h * 0.15), (x0 + w * 0.3, y0), (x1 + w * 0.05, y0), (x1 - w * 0.05, y0 + h * 0.15)], fill=body)
            label(y0 + h * 0.5)
    elif shape == "shoe":
        d.ellipse([x0, y0 + h * 0.45, x1, y1 - h * 0.1], fill=accent)
        d.rounded_rectangle([x0 + w * 0.05, y0 + h * 0.25, x0 + w * 0.55, y0 + h * 0.7], 40, fill=accent)
        d.rectangle([x0, y1 - h * 0.2, x1, y1 - h * 0.13], fill=dark)
    elif shape in {"bag", "jacket", "bedding"}:
        d.rounded_rectangle([x0 + w * 0.1, y0 + h * 0.25, x1 - w * 0.1, y1], 30, fill=accent)
        if shape == "bag":
            d.arc([cx - w * 0.22, y0, cx + w * 0.22, y0 + h * 0.5], 180, 360, fill=dark, width=14)
        if shape == "jacket":
            d.line([cx, y0 + h * 0.25, cx, y1], fill=dark, width=8)
        label(y0 + h * 0.55)
    elif shape == "earbuds":
        for dx in (-0.22, 0.22):
            ex = cx + w * dx
            d.ellipse([ex - w * 0.15, y0 + h * 0.2, ex + w * 0.15, y0 + h * 0.5], fill=accent)
            d.rounded_rectangle([ex - w * 0.05, y0 + h * 0.45, ex + w * 0.05, y1 - h * 0.1], 14, fill=accent)
    elif shape == "tape":
        d.ellipse([x0 + w * 0.1, y0 + h * 0.15, x1 - w * 0.1, y1 - h * 0.15], fill=accent)
        d.ellipse([cx - w * 0.15, (y0 + y1) / 2 - h * 0.12, cx + w * 0.15, (y0 + y1) / 2 + h * 0.12], fill=body)
    else:
        raise ValueError(shape)


def flag_image(w: int) -> Image.Image:
    """National flag of the PRC, drawn from its published geometry (30 x 20 grid)."""
    h = w * 2 // 3
    flag = Image.new("RGB", (w, h), "#de2910")
    d = ImageDraw.Draw(flag)
    u = w / 30

    def star(cx, cy, r, angle):
        pts = []
        for i in range(10):
            a = angle + i * math.pi / 5
            rr = r if i % 2 == 0 else r * 0.382
            pts.append((cx + rr * math.sin(a), cy - rr * math.cos(a)))
        d.polygon(pts, fill="#ffde00")

    star(5 * u, 5 * u, 3 * u, 0)
    for sx, sy in [(10, 2), (12, 4), (12, 7), (10, 9)]:
        star(sx * u, sy * u, 1 * u, math.atan2(5 - sx, -(5 - sy)))
    return flag


def wave(img: Image.Image, phase: float, amp: int = 6) -> Image.Image:
    """Shift each column vertically by a sine wave, so a flag looks like it is moving."""
    w, h = img.size
    out = Image.new("RGBA", (w, h + 2 * amp), (0, 0, 0, 0))
    for x in range(0, w, 2):
        dy = int(amp + amp * math.sin(phase + x / w * 2 * math.pi) * (x / w))
        out.paste(img.crop((x, 0, min(x + 2, w), h)), (x, dy))
    return out


def person(d, cx, base, shirt, scale=1.0, lift=1.0, cup="#e8a317", kind="beer"):
    """A flat figure holding a drink; lift=0 holds it at the chest, lift=1 tips it against the lips."""
    s = scale
    hx, hy = cx, base - 390 * s
    d.ellipse([hx - 38 * s, hy - 38 * s, hx + 38 * s, hy + 38 * s], fill=SKIN)
    d.polygon([(cx - 64 * s, base - 345 * s), (cx + 64 * s, base - 345 * s), (cx + 55 * s, base - 160 * s), (cx - 55 * s, base - 160 * s)], fill=shirt)
    d.rectangle([cx - 46 * s, base - 160 * s, cx - 10 * s, base], fill="#333333")
    d.rectangle([cx + 10 * s, base - 160 * s, cx + 46 * s, base], fill="#333333")
    d.line([cx - 58 * s, base - 330 * s, cx - 88 * s, base - 200 * s], fill=SKIN, width=int(24 * s))
    sx, sy = cx + 58 * s, base - 330 * s
    # hand position: chest (lift 0) -> mouth (lift 1)
    hx1, hy1 = cx + 40 * s + (hx + 26 * s - (cx + 40 * s)) * lift, base - 250 * s + (hy + 12 * s - (base - 250 * s)) * lift
    ex, ey = sx + 50 * s, base - 300 * s + 40 * s * (1 - lift)
    d.line([sx, sy, ex, ey], fill=SKIN, width=int(24 * s))
    d.line([ex, ey, hx1, hy1], fill=SKIN, width=int(22 * s))
    tilt = 0.9 * lift
    gw, gh = (30, 52) if kind in {"beer", "glass"} else (40, 30) if kind == "mug" else (26, 40) if kind == "wine" else (24, 24)
    gw, gh = gw * s * 1.45, gh * s * 1.45
    pts = [(-gw / 2, -gh), (gw / 2, -gh), (gw / 2 * 0.8, 0), (-gw / 2 * 0.8, 0)]
    rot = [(hx1 + x * math.cos(tilt) - y * math.sin(tilt), hy1 + x * math.sin(tilt) + y * math.cos(tilt)) for x, y in pts]
    d.polygon(rot, fill=cup, outline="#5a4a20")
    if kind == "wine":
        d.line([hx1, hy1, hx1 - 14 * s * math.sin(tilt), hy1 + 30 * s * math.cos(tilt)], fill="#bbbbbb", width=int(4 * s))
    if kind == "mug":
        d.arc([hx1 + gw / 2 - 6 * s, hy1 - gh * 0.8, hx1 + gw / 2 + 14 * s, hy1 - gh * 0.2], 270, 90, fill=cup, width=int(6 * s))


def draw_scene(img, d, art, k, colors, brand, shape, lang):
    """Draw scene `art` for frame k (0..23) of its shot."""
    x0, y0, x1, y1 = ART_BOX
    p = k / (FPS * SHOT_S - 1)
    if art.startswith("drink_"):
        cup, kind = CUPS[art]
        floor = y1 + 40
        d.rectangle([0, floor, W, BAND[1]], fill="#d8c7a6")
        lift = 0.0 if p < 0.35 else min(1.0, (p - 0.35) / 0.25)
        bottle_shape = SCENE_BOTTLE.get(art, shape)
        d.rectangle([300, floor - 70, 460, floor - 56], fill="#8a6d3b")  # table
        d.rectangle([316, floor - 56, 326, floor], fill="#8a6d3b")
        d.rectangle([434, floor - 56, 444, floor], fill="#8a6d3b")
        draw_product(d, bottle_shape, (330, floor - 230, 430, floor - 70), colors, brand, lang)
        if art == "drink_cheers":
            person(d, 110, floor, SHIRTS[0], 0.85, lift, cup, "beer")
            person(d, 240, floor, SHIRTS[1], 0.85, lift, cup, "beer")
        else:
            person(d, 170, floor, SHIRTS[2], 0.95, lift, cup, kind)
        if art == "drink_juice":
            d.ellipse([400, floor - 300, 450, floor - 250], fill="#f7a21b")
            d.line([425, floor - 300, 432, floor - 318], fill="#3c8a34", width=5)
    elif art == "flag":
        pole_x = 96
        d.rectangle([pole_x - 5, y0 - 40, pole_x + 5, y1 + 40], fill="#8a8a8a")
        f = wave(flag_image(230), phase=k * 0.5)
        img.paste(f, (pole_x + 5, y0 - 30), f)
        draw_product(d, shape, (230, 360, 410, 600), colors, brand, lang)
    elif art == "lanterns":
        for i, lx in enumerate((110, 300)):
            sw = 8 * math.sin(k * 0.4 + i)
            d.line([lx + 40, y0 - 40, lx + 40 + sw, y0 + 10], fill="#7a1010", width=4)
            d.ellipse([lx + sw, y0 + 10, lx + 80 + sw, y0 + 110], fill="#e0301e")
            d.rectangle([lx + 24 + sw, y0 + 4, lx + 56 + sw, y0 + 14], fill="#f2c14e")
            d.rectangle([lx + 24 + sw, y0 + 106, lx + 56 + sw, y0 + 116], fill="#f2c14e")
        draw_product(d, shape, (130, 340, 350, 600), colors, brand, lang)
    elif art in {"skin_before", "skin_after", "split_skin", "split_teeth", "split_colors"}:
        panels = {"skin_before": ["blemish"], "skin_after": ["clear"], "split_skin": ["blemish", "clear"],
                  "split_teeth": ["yellow", "white"], "split_colors": ["sand", "slate"]}[art]
        n = len(panels)
        pw = (x1 - x0 - 16 * (n - 1)) // n
        rnd = random.Random(art)
        for i, kind in enumerate(panels):
            px = x0 + i * (pw + 16)
            box = (px, y0 + 20, px + pw, y1 - 20)
            if kind in {"blemish", "clear"}:
                d.rounded_rectangle(box, 24, fill="#efc7a8")
                d.ellipse([box[0] + pw * 0.2, box[1] + 60, box[0] + pw * 0.8, box[1] + 140], fill="#e6b796")
                if kind == "blemish":
                    for _ in range(22):
                        ex, ey = rnd.uniform(box[0] + 14, box[2] - 14), rnd.uniform(box[1] + 150, box[3] - 20)
                        r = rnd.uniform(4, 9)
                        d.ellipse([ex - r, ey - r, ex + r, ey + r], fill="#c4573f")
            elif kind in {"yellow", "white"}:
                d.rounded_rectangle(box, 24, fill="#b8444e")
                tooth = "#d9c27a" if kind == "yellow" else "#fbfbf6"
                tw = (pw - 30) / 4
                for row, ty in enumerate((box[1] + 120, box[1] + 220)):
                    for t in range(4):
                        tx = box[0] + 15 + t * tw
                        d.rounded_rectangle([tx + 2, ty, tx + tw - 2, ty + 80], 10, fill=tooth)
            else:
                col = "#d9c3a0" if kind == "sand" else "#4f5b66"
                d.rounded_rectangle([box[0] + 10, box[1] + 120, box[2] - 10, box[3] - 40], 26, fill=col)
                d.arc([box[0] + pw * 0.25, box[1] + 40, box[2] - pw * 0.25, box[1] + 200], 180, 360, fill="#333333", width=10)
    elif art in {"box_closed", "box_open"}:
        d.polygon([(130, 360), (350, 360), (350, 560), (130, 560)], fill="#b98b52")
        if art == "box_closed":
            d.polygon([(130, 360), (190, 310), (410, 310), (350, 360)], fill="#d1a56d")
            d.rectangle([228, 330, 252, 560], fill="#e8d7b5")
        else:
            d.polygon([(130, 360), (80, 300), (300, 300), (350, 360)], fill="#d1a56d")
            for i in range(4):
                d.rounded_rectangle([150 + i * 50, 250 - (i % 2) * 20, 190 + i * 50, 360], 10, fill=colors[3])
    else:
        raise ValueError(art)


# ---------------------------------------------------------------- frames and timeline
def flash_start(item_id: str) -> int:
    rnd = random.Random(int(hashlib.sha256(item_id.encode()).hexdigest(), 16))
    return rnd.randint(FPS, N_FRAMES - FPS - FLASH_FRAMES)  # between 1.0 s and 6.5 s


def frame_image(item, lang, f, flash_at):
    pal = PALETTES[item["product"]["palette"]]
    brand, shape = item["product"]["brand"], item["product"]["shape"]
    shot = f // (FPS * SHOT_S)
    k = f % (FPS * SHOT_S)
    img = gradient(W, H, pal[0], "#ffffff")
    d = ImageDraw.Draw(img)
    art = (item.get("art") or {}).get(shot + 1, "product")
    if art == "product":
        z = 1.0 + 0.04 * k / (FPS * SHOT_S)  # slow push-in
        x0, y0, x1, y1 = ART_BOX
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hw, hh = (x1 - x0) / 2 * z, (y1 - y0) / 2 * z
        draw_product(d, shape, (int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh)), pal, brand, lang)
    else:
        draw_scene(img, d, art, k, pal, brand, shape, lang)
    # account handle
    hf = font(lang, "regular", 20)
    d.text((20, 28), f"@{brand}", font=hf, fill="#555555")
    # subtitle band
    line = item["lines"][shot]
    f_, rows = wrap(d, line, lang, "bold", 32 if lang == "zh" else 30, W - 48)
    band_h = 26 + len(rows) * int(f_.size * 1.35)
    top = BAND[1] + (BAND[3] - BAND[1] - band_h) // 2
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rounded_rectangle([14, top, W - 14, top + band_h], 18, fill=(15, 15, 15, 200))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(img)
    y = top + 13
    for r in rows:
        tw = d.textlength(r, font=f_)
        d.text(((W - tw) / 2, y), r, font=f_, fill="#ffffff")
        y += int(f_.size * 1.35)
    # sticker
    if item.get("flash") and flash_at <= f < flash_at + FLASH_FRAMES:
        sf, srows = wrap(d, item["flash"], lang, "bold", 34 if lang == "zh" else 30, W - 110)
        sw = max(d.textlength(r, font=sf) for r in srows) + 40
        sh = len(srows) * int(sf.size * 1.3) + 28
        st = Image.new("RGBA", (int(sw), int(sh)), (0, 0, 0, 0))
        sd = ImageDraw.Draw(st)
        sd.rounded_rectangle([0, 0, sw - 1, sh - 1], 16, fill=(255, 221, 0, 255), outline=(40, 40, 40, 255), width=3)
        yy = 14
        for r in srows:
            sd.text(((sw - sd.textlength(r, font=sf)) / 2, yy), r, font=sf, fill="#1a1a1a")
            yy += int(sf.size * 1.3)
        st = st.rotate(-5, expand=True, resample=Image.BICUBIC)
        img.paste(st, ((W - st.width) // 2, 92), st)
    return img


def timeline(item, flash_at):
    """Every text that appears on screen, with its frame range [start, end)."""
    rows = [{"text": line, "kind": "subtitle", "shot": i + 1, "start": i * FPS * SHOT_S, "end": (i + 1) * FPS * SHOT_S}
            for i, line in enumerate(item["lines"])]
    rows.append({"text": "@" + item["product"]["brand"], "kind": "handle", "start": 0, "end": N_FRAMES})
    if item.get("flash"):
        rows.append({"text": item["flash"], "kind": "flash", "start": flash_at, "end": flash_at + FLASH_FRAMES})
    return rows


def load_items(lang=None):
    items = []
    for lg in (["zh", "en"] if lang is None else [lang]):
        for it in yaml.safe_load((HERE / f"items_video_{lg}.yaml").read_text(encoding="utf-8")):
            it["lang"] = lg
            items.append(it)
    if lang is None:  # development items carry their own lang
        items += yaml.safe_load((HERE / "items_video_dev.yaml").read_text(encoding="utf-8"))
    return items


def render(item, out_dir: Path) -> dict:
    lang = item["lang"]
    fa = flash_start(item["id"])
    path = out_dir / f"{item['id']}.mp4"
    c = av.open(str(path), "w")
    s = c.add_stream("libx264", rate=FPS)
    s.width, s.height, s.pix_fmt = W, H, "yuv420p"
    s.options = {"crf": "20", "preset": "medium"}
    for f in range(N_FRAMES):
        for pkt in s.encode(av.VideoFrame.from_image(frame_image(item, lang, f, fa))):
            c.mux(pkt)
    for pkt in s.encode():
        c.mux(pkt)
    c.close()
    return {"id": item["id"], "lang": lang, "file": path.name, "fps": FPS, "frames": N_FRAMES, "width": W, "height": H,
            "flash_start": fa if item.get("flash") else None, "timeline": timeline(item, fa),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def contact_sheet(item, out: Path):
    fa = flash_start(item["id"])
    idx = [6 + 12 * i for i in range(8)]
    if item.get("flash"):
        idx[3] = fa + 2  # always show the sticker once on the sheet
    thumbs = [frame_image(item, item["lang"], i, fa).resize((W // 2, H // 2)) for i in idx]
    sheet = Image.new("RGB", (W // 2 * 8, H // 2), "#000000")
    for i, t in enumerate(thumbs):
        sheet.paste(t, (i * W // 2, 0))
    sheet.save(out / f"{item['id']}.jpg", quality=85)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", default="")
    ap.add_argument("--sheets", action="store_true")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    items = load_items()
    if a.only:
        keep = set(a.only.split(","))
        items = [i for i in items if i["id"] in keep]
    manifest_path = HERE / "video_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for it in items:
        manifest[it["id"]] = render(it, a.out)
        if a.sheets:
            (a.out / "sheets").mkdir(exist_ok=True)
            contact_sheet(it, a.out / "sheets")
        print(it["id"], manifest[it["id"]]["sha256"][:12], flush=True)
    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
