"""Contact sheet of four test videos for the README: one frame from each 2-second shot, with the frame that
carries the 0.5 s sticker taken instead where the item has one.

Run: python testset/contact_sheet.py   -> media/examples.zh.jpg, media/examples.en.jpg
Needs the rendered videos ($POLICY_LAB_DATA/videos) and a font with Chinese glyphs (Noto Sans SC by default,
or POLICY_LAB_FONT).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lab.common import VIDEOS  # noqa: E402

ROWS = {
    "zh": [("zv-17", "第 2 镜闪过 0.5 秒贴纸：宣称治病"), ("zv-23", "违规只在画面里：第 2 镜举杯喝酒"),
           ("zv-26", "违规只在画面里：封面镜头有国旗"), ("zv-44", "合规的陷阱题：喝的是茶，不是酒")],
    "en": [("ev-17", "0.5 s sticker in shot 2: a disease claim"), ("ev-23", "Footage only: a person drinks beer in shot 2"),
           ("ev-26", "Footage only: before-and-after panels in shot 2"), ("ev-44", "Compliant look-alike: the drink is water")],
}
HEAD = {"zh": ["镜头 1", "镜头 2", "镜头 3", "镜头 4"], "en": ["Shot 1", "Shot 2", "Shot 3", "Shot 4"]}
W, H = 180, 320  # thumbnail
PAD, LABEL = 16, 40
BG, FG, MUTED, ACCENT = (250, 249, 246), (28, 28, 30), (110, 110, 115), (196, 62, 44)


def font(size: int):
    path = os.environ.get("POLICY_LAB_FONT", "C:/Windows/Fonts/NotoSansSC-VF.ttf")
    return ImageFont.truetype(path, size)


def frames(item_id: str, indices: list[int]) -> list[Image.Image]:
    want, out = set(indices), {}
    with av.open(str(VIDEOS / f"{item_id}.mp4")) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                out[i] = fr.to_image()
    return [out[i] for i in indices]


def main() -> int:
    man = json.loads((ROOT / "testset" / "video_manifest.json").read_text(encoding="utf-8"))
    out_dir = ROOT / "media"
    out_dir.mkdir(exist_ok=True)
    for lang, rows in ROWS.items():
        width = PAD + 4 * (W + PAD)
        height = PAD + 28 + len(rows) * (LABEL + H + PAD)
        sheet = Image.new("RGB", (width, height), BG)
        d = ImageDraw.Draw(sheet)
        for c, h in enumerate(HEAD[lang]):
            d.text((PAD + c * (W + PAD), PAD), h, fill=MUTED, font=font(15))
        for r, (item_id, label) in enumerate(rows):
            m = man[item_id]
            idx = [12, 36, 60, 84]
            flash = m.get("flash_start")
            if flash is not None:
                idx[flash // 24] = flash + 3
            y = PAD + 28 + r * (LABEL + H + PAD)
            d.text((PAD, y + 6), f"{item_id}  {label}", fill=FG, font=font(18))
            for c, im in enumerate(frames(item_id, idx)):
                x = PAD + c * (W + PAD)
                sheet.paste(im.resize((W, H), Image.LANCZOS), (x, y + LABEL))
                if flash is not None and c == flash // 24:
                    d.rectangle([x - 3, y + LABEL - 3, x + W + 2, y + LABEL + H + 2], outline=ACCENT, width=3)
        p = out_dir / f"examples.{lang}.jpg"
        sheet.save(p, quality=88)
        print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
