#!/usr/bin/env python3
"""Generate an OG social card from a blog hero image.

Usage:
    python scripts/og_card.py docs/blog/images/my-hero.png "Post Title" "subtitle text"
    python scripts/og_card.py docs/blog/images/my-hero.png "Post Title" "subtitle" --output og/my-hero-og.png

OG card style: 1200x630, hero as background with dark gradient overlay,
synapt logo top-left, title centered, teal accent line, subtitle below,
"synapt.dev" bottom-right.

Requires: Pillow, and the birefnet-cleaned icon at /tmp/icon-nobg.png
or assets/logo.png as fallback.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def find_icon() -> Path:
    """Find the best available logo icon."""
    candidates = [
        Path("/tmp/icon-nobg.png"),
        Path(__file__).parent.parent / "assets" / "logo.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No logo icon found. Need /tmp/icon-nobg.png or assets/logo.png")


def generate_og_card(
    hero_path: str,
    title: str,
    subtitle: str,
    output_path: str | None = None,
) -> str:
    """Generate OG social card. Returns output path."""
    hero = Image.open(hero_path).convert("RGBA")
    og = hero.resize((1200, 630), Image.LANCZOS)

    # Dark gradient overlay
    overlay = Image.new("RGBA", (1200, 630), (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    for y in range(630):
        alpha = int(140 + 40 * (1 - abs(y - 315) / 315))
        draw_ov.line([(0, y), (1200, y)], fill=(10, 10, 20, alpha))
    og = Image.alpha_composite(og, overlay)

    draw = ImageDraw.Draw(og)

    # Logo top-left
    icon = Image.open(find_icon()).convert("RGBA")
    icon_cropped = icon.crop(icon.getbbox())
    icon_h = 32
    aspect = icon_cropped.width / icon_cropped.height
    icon_w = int(icon_h * aspect)
    icon_small = icon_cropped.resize((icon_w, icon_h), Image.LANCZOS)
    og.paste(icon_small, (30, 25), icon_small)

    # Fonts
    try:
        font_logo = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
        font_sub = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_url = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except OSError:
        font_logo = font_title = font_sub = font_url = ImageFont.load_default()

    # "synapt" next to logo
    draw.text((30 + icon_w + 10, 28), "synapt", fill=(160, 120, 220, 220), font=font_logo)

    # Title centered
    title_bbox = font_title.getbbox(title)
    title_w = title_bbox[2] - title_bbox[0]
    draw.text(((1200 - title_w) // 2, 250), title, fill=(255, 255, 255, 240), font=font_title)

    # Teal accent line
    draw.line([(500, 310), (700, 310)], fill=(6, 182, 212, 200), width=2)

    # Subtitle
    sub_bbox = font_sub.getbbox(subtitle)
    sub_w = sub_bbox[2] - sub_bbox[0]
    draw.text(((1200 - sub_w) // 2, 325), subtitle, fill=(200, 200, 210, 200), font=font_sub)

    # synapt.dev bottom-right
    draw.text((1080, 590), "synapt.dev", fill=(160, 120, 220, 180), font=font_url)

    # Output
    if output_path is None:
        stem = Path(hero_path).stem
        output_path = str(Path(hero_path).parent / "og" / f"{stem}-og.png")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    og.convert("RGB").save(output_path, quality=95)
    print(f"OG card saved: {output_path} (1200x630)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OG social card from hero image")
    parser.add_argument("image", help="Path to hero image")
    parser.add_argument("title", help="Post title")
    parser.add_argument("subtitle", help="Subtitle text")
    parser.add_argument("--output", "-o", help="Output path (default: images/og/<stem>-og.png)")
    args = parser.parse_args()
    generate_og_card(args.image, args.title, args.subtitle, args.output)
