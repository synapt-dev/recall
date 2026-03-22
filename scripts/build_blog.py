#!/usr/bin/env python3
"""Build blog HTML from markdown files.

Reads docs/blog/*.md, parses YAML frontmatter, renders HTML using
inline templates that match the existing synapt.dev blog style.

Usage:
    python scripts/build_blog.py              # build all posts
    python scripts/build_blog.py --force      # rebuild even if HTML is newer
    python scripts/build_blog.py --dry-run    # show what would be built
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent

try:
    import markdown
except ImportError:
    print("Missing dependency: pip install markdown")
    sys.exit(1)


BLOG_DIR = Path(__file__).resolve().parent.parent / "docs" / "blog"
IMAGES_DIR = BLOG_DIR / "images"

# Known authors and their metadata
AUTHORS = {
    "opus": ("Opus", "Claude", "author-opus.jpg"),
    "atlas": ("Atlas", "Codex", "author-atlas.jpg"),
    "apollo": ("Apollo", "Claude", "author-apollo.jpg"),
    "sentinel": ("Sentinel", "Claude", "author-sentinel.jpg"),
    "layne": ("Layne Penney", "", "author-layne.jpg"),
}

PLAUSIBLE_SNIPPET = dedent("""\
    <!-- Privacy-friendly analytics by Plausible -->
    <script async src="https://plausible.io/js/pa-AlVJwNFS6NppMt50yqocF.js"></script>
    <script>
      window.plausible=window.plausible||function(){(plausible.q=plausible.q||[]).push(arguments)},plausible.init=plausible.init||function(i){plausible.o=i||{}};
      plausible.init()
    </script>""")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML-style frontmatter from markdown text.

    Supports --- delimited frontmatter at the top of the file.
    Returns (metadata_dict, remaining_markdown).
    """
    if not text.startswith("---"):
        return {}, text

    end = text.find("\n---", 3)
    if end == -1:
        return {}, text

    frontmatter = text[3:end].strip()
    body = text[end + 4:].strip()

    meta = {}
    for line in frontmatter.split("\n"):
        line = line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()

    return meta, body


def render_post_html(meta: dict, body_html: str, slug: str) -> str:
    """Render a full blog post HTML page."""
    title = meta.get("title", "Untitled")
    description = meta.get("description", "")
    author_key = meta.get("author", "opus").lower()
    date = meta.get("date", "")
    subtitle = meta.get("subtitle", "")
    hero = meta.get("hero", "")

    author_name, author_model, author_img = AUTHORS.get(
        author_key, (author_key.title(), "", f"author-{author_key}.jpg")
    )

    byline_text = f"{author_name} ({author_model})" if author_model else author_name
    date_display = date if date else "2026"

    hero_tag = ""
    if hero:
        hero_tag = f'<img src="images/{hero}" alt="{title}" class="hero">'
    elif (IMAGES_DIR / f"{slug}-hero.jpg").exists():
        hero_tag = f'<img src="images/{slug}-hero.jpg" alt="{title}" class="hero">'

    subtitle_tag = ""
    if subtitle:
        subtitle_tag = f'<p class="subtitle">{subtitle}</p>'

    og_image = f"images/{hero}" if hero else f"images/{slug}-hero.jpg"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} — synapt</title>
  <meta name="description" content="{description}">
  <meta property="og:title" content="{title}">
  <meta property="og:description" content="{description}">
  <meta property="og:image" content="https://synapt.dev/blog/{og_image}">
  <meta property="og:url" content="https://synapt.dev/blog/{slug}.html">
  <meta property="og:type" content="article">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:site" content="@synapt_dev">
  <meta name="twitter:title" content="{title}">
  <meta name="twitter:description" content="{description}">
  <meta name="twitter:image" content="https://synapt.dev/blog/{og_image}">
  <link rel="icon" href="/favicon.ico" type="image/x-icon">
  <link rel="apple-touch-icon" href="/apple-touch-icon.png">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      --purple: #7c5cbf;
      --purple-light: #9b7fd4;
      --teal: #00e5cc;
      --teal-dim: #00c4ae;
      --bg: #0d1117;
      --bg-card: #161b22;
      --bg-code: #1c2129;
      --text: #e6edf3;
      --text-dim: #8b949e;
      --border: #30363d;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: 'Inter', -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.7;
    }}
    a {{ color: var(--teal); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{ font-family: 'JetBrains Mono', monospace; background: var(--bg-code); padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
    pre {{ background: var(--bg-code); padding: 1rem 1.5rem; border-radius: 8px; overflow-x: auto; margin-bottom: 1.2rem; }}
    pre code {{ background: none; padding: 0; }}
    blockquote {{
      background: var(--bg-card);
      border-left: 3px solid var(--purple);
      padding: 1rem 1.5rem;
      margin: 1.5rem 0;
      border-radius: 0 6px 6px 0;
      font-style: italic;
    }}
    blockquote p {{ margin-bottom: 0.5rem; }}
    blockquote p:last-child {{ margin-bottom: 0; }}
    .container {{ max-width: 720px; margin: 0 auto; padding: 0 1.5rem; }}
    header {{
      border-bottom: 1px solid var(--border);
      padding: 1rem 0;
    }}
    header .container {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    header .logo {{ font-weight: 700; font-size: 1.2rem; color: var(--purple-light); }}
    header nav a {{ margin-left: 1.5rem; color: var(--text-dim); font-size: 0.9rem; }}
    article {{ padding: 3rem 0 4rem; }}
    article h1 {{
      font-size: 2.2rem;
      font-weight: 700;
      margin-bottom: 0.5rem;
      line-height: 1.2;
    }}
    article .subtitle {{
      color: var(--text-dim);
      font-size: 1.1rem;
      margin-bottom: 0.5rem;
    }}
    article .meta {{
      color: var(--text-dim);
      font-size: 0.85rem;
      margin-bottom: 2.5rem;
      padding-bottom: 2rem;
      border-bottom: 1px solid var(--border);
    }}
    article h2 {{
      font-size: 1.5rem;
      margin: 2.5rem 0 1rem;
      color: var(--purple-light);
    }}
    article h3 {{
      font-size: 1.2rem;
      margin: 2rem 0 0.75rem;
      color: var(--text);
    }}
    article p {{ margin-bottom: 1.2rem; }}
    article ul, article ol {{ margin-bottom: 1.2rem; padding-left: 1.5rem; }}
    article li {{ margin-bottom: 0.5rem; }}
    article strong {{ color: var(--teal); font-weight: 600; }}
    article table {{
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 1.5rem;
      font-size: 0.9rem;
    }}
    article th, article td {{
      padding: 0.5rem 0.75rem;
      border: 1px solid var(--border);
      text-align: left;
    }}
    article th {{
      background: var(--bg-card);
      font-weight: 600;
    }}
    .hero {{
      width: 100%;
      border-radius: 12px;
      margin-bottom: 2rem;
    }}
    .byline {{
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }}
    .byline img {{
      width: 28px;
      height: 28px;
      border-radius: 50%;
      object-fit: cover;
    }}
    .cta {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 2rem;
      margin: 2.5rem 0;
      text-align: center;
    }}
    .cta code {{
      display: block;
      font-size: 1.1rem;
      margin: 1rem 0;
      padding: 0.8rem;
      background: var(--bg-code);
      border-radius: 6px;
    }}
    @media (max-width: 640px) {{
      article h1 {{ font-size: 1.6rem; }}
      article .subtitle {{ font-size: 0.95rem; }}
      header nav a {{ margin-left: 1rem; font-size: 0.8rem; }}
    }}
  </style>
  {PLAUSIBLE_SNIPPET}
</head>
<body>
  <header>
    <div class="container">
      <a href="../" class="logo">synapt</a>
      <nav>
        <a href="../#features">Features</a>
        <a href="../#benchmarks">Benchmarks</a>
        <a href="https://github.com/laynepenney/synapt">GitHub</a>
        <a href="https://x.com/synapt_dev">X</a>
      </nav>
    </div>
  </header>

  <article>
    <div class="container">
      {hero_tag}
      <h1>{title}</h1>
      {subtitle_tag}
      <p class="meta"><span class="byline"><img src="images/{author_img}" alt="{author_name}"> <a href="authors.html#{author_key}" style="color: var(--text-dim);">{byline_text}</a></span> &middot; {date_display}</p>

      {body_html}

      <div class="cta">
        <p>synapt gives your AI agents persistent memory across sessions.</p>
        <code>pip install synapt</code>
        <p><a href="https://github.com/laynepenney/synapt">GitHub</a> &middot; <a href="../">synapt.dev</a></p>
      </div>
    </div>
  </article>
</body>
</html>"""


def build_post(md_path: Path, force: bool = False, dry_run: bool = False) -> bool:
    """Build a single blog post. Returns True if built."""
    slug = md_path.stem
    html_path = md_path.parent / f"{slug}.html"

    # Skip if HTML is newer than markdown (unless forced)
    if not force and html_path.exists():
        if html_path.stat().st_mtime >= md_path.stat().st_mtime:
            return False

    text = md_path.read_text(encoding="utf-8")
    meta, body_md = parse_frontmatter(text)

    if not meta.get("title"):
        # Try to extract title from first H1
        h1_match = re.match(r"^#\s+(.+)$", body_md, re.MULTILINE)
        if h1_match:
            meta["title"] = h1_match.group(1)
            body_md = body_md[h1_match.end():].strip()

    # Render markdown to HTML
    md = markdown.Markdown(extensions=["tables", "fenced_code", "codehilite"])
    body_html = md.convert(body_md)

    html = render_post_html(meta, body_html, slug)

    if dry_run:
        print(f"  Would build: {slug}.html ({len(html)} bytes)")
        return True

    html_path.write_text(html, encoding="utf-8")
    print(f"  Built: {slug}.html ({len(html)} bytes)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Build blog HTML from markdown")
    parser.add_argument("--force", action="store_true", help="Rebuild all posts")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be built")
    args = parser.parse_args()

    md_files = sorted(BLOG_DIR.glob("*.md"))
    if not md_files:
        print(f"No markdown files found in {BLOG_DIR}")
        return 1

    print(f"[build_blog] Scanning {len(md_files)} markdown files...")
    built = 0
    for md_path in md_files:
        if build_post(md_path, force=args.force, dry_run=args.dry_run):
            built += 1

    if built == 0:
        print("[build_blog] All posts up to date.")
    else:
        action = "Would build" if args.dry_run else "Built"
        print(f"[build_blog] {action} {built} post(s).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
