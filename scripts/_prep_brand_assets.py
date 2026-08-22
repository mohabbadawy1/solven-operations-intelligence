"""One-off local script: derive clean, trimmed logo assets for the Solven
design system from the source files supplied on the user's Desktop.

Not part of the runtime pipeline -- run once to populate assets/brand/,
which IS committed and IS what ai/html_report_renderer.py and
ai/email_renderer.py read at render time. Re-run only if the source
logo files change. Requires Pillow (`pip install pillow`), which is
not in requirements.txt since nothing at runtime needs it.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

SRC = "/Users/hoba/Desktop/solven_pics/solven-logo-transparent.png"
OUT_DIR = "/Users/hoba/Desktop/solven-operations-intelligence/assets/brand"

CREAM = (244, 241, 234)  # #F4F1EA


def trim(im: Image.Image, pad: int = 24) -> Image.Image:
    arr = np.array(im)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 8)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, arr.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, arr.shape[0])
    return im.crop((x0, y0, x1, y1))


def main() -> None:
    im = Image.open(SRC).convert("RGBA")
    arr = np.array(im).astype(np.float64)
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]

    # The mark (orange slash) has hue far from grey; the wordmark glyphs
    # are near-black/grey (r≈g≈b). Separate by saturation, not a hard
    # color match, so anti-aliased edge pixels blend smoothly instead of
    # producing a halo.
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    sat = np.where(maxc > 0, (maxc - minc) / np.maximum(maxc, 1), 0)
    is_glyph = sat < 0.15  # near-neutral -> wordmark letterform, not the orange mark

    # Light-background variant: source glyphs are already near-black --
    # just trim. Save as-is.
    on_light = trim(im.copy())
    on_light.save(f"{OUT_DIR}/solven-logo-on-light.png")

    # Dark-background variant: recolor only the near-neutral glyph pixels
    # to cream, weighted by each pixel's own alpha/darkness so
    # anti-aliased edges stay smooth; the orange mark is left untouched.
    arr2 = arr.copy()
    for ch, cream_v in zip(range(3), CREAM):
        arr2[:, :, ch] = np.where(is_glyph, cream_v, arr[:, :, ch])
    on_dark = Image.fromarray(arr2.astype(np.uint8), mode="RGBA")
    on_dark = trim(on_dark)
    on_dark.save(f"{OUT_DIR}/solven-logo-on-dark.png")

    # Mark alone (orange slash), cropped from the same source, for
    # small compact usages (e.g. a single accent glyph).
    arr3 = arr.copy()
    arr3[:, :, 3] = np.where(is_glyph, 0, a)  # drop the glyph pixels' alpha
    mark = Image.fromarray(arr3.astype(np.uint8), mode="RGBA")
    mark = trim(mark, pad=16)
    mark.save(f"{OUT_DIR}/solven-mark.png")

    for name in ("solven-logo-on-light.png", "solven-logo-on-dark.png", "solven-mark.png"):
        i = Image.open(f"{OUT_DIR}/{name}")
        print(name, i.size, i.mode)


if __name__ == "__main__":
    main()
