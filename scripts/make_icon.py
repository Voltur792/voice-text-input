"""Generate app/icon.ico and app/icon.png — a microphone glyph.

Run once from the repo root:  .venv\\Scripts\\python.exe scripts\\make_icon.py
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 256
OUT = Path(__file__).resolve().parent.parent / "app"


def make() -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = SIZE / 64  # the 64px sketch scaled up
    body = (36, 84, 196, 255)
    # mic capsule
    d.rounded_rectangle((24 * s, 6 * s, 40 * s, 34 * s), radius=8 * s, fill=body)
    # cradle
    d.rounded_rectangle((19 * s, 28 * s, 45 * s, 37 * s), radius=4 * s, fill=body)
    # stem + base
    d.rectangle((30 * s, 37 * s, 34 * s, 47 * s), fill=body)
    d.rounded_rectangle((24 * s, 50 * s, 40 * s, 55 * s), radius=2 * s, fill=body)
    # sound arcs
    d.arc((10 * s, 16 * s, 22 * s, 40 * s), start=110, end=250,
          fill=(255, 170, 40, 255), width=int(3 * s))
    d.arc((42 * s, 16 * s, 54 * s, 40 * s), start=-70, end=70,
          fill=(255, 170, 40, 255), width=int(3 * s))
    return img


if __name__ == "__main__":
    img = make()
    OUT.mkdir(exist_ok=True)
    img.save(OUT / "icon.png")
    img.save(OUT / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32),
                                      (48, 48), (64, 64), (128, 128), (256, 256)])
    print("written:", OUT / "icon.png", OUT / "icon.ico")
    sys.exit(0)
