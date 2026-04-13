#!/usr/bin/env python3
"""Apply synapt branded watermark to a blog hero image.

Usage:
    python scripts/watermark.py docs/blog/images/my-hero.png
    python scripts/watermark.py docs/blog/images/my-hero.png --output /tmp/watermarked.png

Watermark style: synapt logo icon (cropped, transparent) + purple "synapt"
text on subtle dark pill, top-left corner. Matches the pixel-perfect style
iterated in the 2026-03-27 session.

Hero image standard:
    - Canonical size: 1408x768 (exact). All blog hero images must be this size.
    - Generated via fal-ai/nano-banana-2 (nano-banana-2 may produce 1376x768 or
      1584x672; resize to 1408x768 with cover-crop before watermarking).
    - Naming: {slug}-hero-raw.png (pre-watermark), {slug}-hero.png (final).
    - Style: cyberpunk owl theme, purple/teal accent lighting, dark background.
    - Owl theme is enforced by hero.py and generate_hero.py via STYLE_SUFFIX.
      Owls must always be the focal subject. Humans may appear alongside owls
      when the post context calls for it (e.g. human-AI collaboration themes).
    - The blog CSS uses aspect-ratio: 16/9 + object-fit: cover on card thumbnails,
      so non-standard sizes will crop unevenly and may hide the watermark.

Requires: Pillow, and the birefnet-cleaned icon at /tmp/icon-nobg.png
or assets/logo.png as fallback.
"""

import argparse
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def find_icon() -> Path:
    """Find the best available logo icon."""
    candidates = [
        Path("/tmp/icon-nobg.png"),  # birefnet-cleaned (best)
        Path(__file__).parent.parent / "docs" / "assets" / "logo.png",  # transparent bg
        Path(__file__).parent.parent / "assets" / "logo.png",  # fallback (has dark bg)
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No logo icon found. Need /tmp/icon-nobg.png or assets/logo.png")


HERO_SIZE = (1408, 768)  # Canonical blog hero dimensions


def resize_to_standard(img: Image.Image) -> Image.Image:
    """Resize image to canonical HERO_SIZE using cover-crop."""
    if img.size == HERO_SIZE:
        return img
    scale = max(HERO_SIZE[0] / img.width, HERO_SIZE[1] / img.height)
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - HERO_SIZE[0]) // 2
    top = (new_h - HERO_SIZE[1]) // 2
    return img.crop((left, top, left + HERO_SIZE[0], top + HERO_SIZE[1]))


def apply_watermark(image_path: str, output_path: str | None = None, resize: bool = True) -> str:
    """Apply synapt watermark to an image. Returns output path."""
    hero = Image.open(image_path).convert("RGBA")
    if resize and hero.size != HERO_SIZE:
        old = hero.size
        hero = resize_to_standard(hero)
        print(f"Resized: {old[0]}x{old[1]} -> {HERO_SIZE[0]}x{HERO_SIZE[1]}")
    icon = Image.open(find_icon()).convert("RGBA")
    icon_cropped = icon.crop(icon.getbbox())

    # Scale icon proportionally to image height
    icon_h = max(36, int(hero.height * 0.052))
    aspect = icon_cropped.width / icon_cropped.height
    icon_w = int(icon_h * aspect)
    icon_small = icon_cropped.resize((icon_w, icon_h), Image.LANCZOS)

    # Font — scale with image
    font_size = max(26, int(hero.height * 0.039))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except OSError:
        font = ImageFont.load_default()

    text = "synapt"
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    padding_x = 10
    padding_y = 8
    pill_w = icon_w + text_w + padding_x * 3 + 4
    pill_h = icon_h + padding_y * 2

    # Subtle dark pill
    pill = Image.new("RGBA", (pill_w, pill_h), (15, 15, 25, 120))
    draw = ImageDraw.Draw(pill)

    # Icon at top of pill content area
    icon_y = padding_y
    pill.paste(icon_small, (padding_x, icon_y), icon_small)

    # Purple text — center on icon's visual midpoint, accounting for font ascent
    text_y_offset = bbox[1]  # font ascent offset for true centering
    icon_center_y = icon_y + icon_h // 2
    text_render_y = icon_center_y - text_h // 2 - text_y_offset
    draw.text(
        (icon_w + padding_x * 2 + 4, text_render_y),
        text,
        fill=(160, 120, 220, 200),
        font=font,
    )

    # Top-left with margin
    margin = 14
    hero.paste(pill, (margin, margin), pill)

    out = output_path or image_path
    hero.convert("RGB").save(out, quality=95)
    print(f"Watermarked: {out} ({hero.size[0]}x{hero.size[1]})")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply synapt watermark to hero image")
    parser.add_argument("image", help="Path to hero image")
    parser.add_argument("--output", "-o", help="Output path (default: overwrite input)")
    parser.add_argument("--no-resize", action="store_true", help="Skip auto-resize to 1408x768")
    args = parser.parse_args()
    apply_watermark(args.image, args.output, resize=not args.no_resize)
