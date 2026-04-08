#!/usr/bin/env python3
"""Generate blog index HTML from markdown frontmatter.

Reads all docs/blog/*.md files, parses YAML frontmatter, and generates
docs/blog/index.html with proper card markup. The newest post gets the
"featured" treatment with a New label.

Usage:
    python scripts/build_blog_index.py
    python scripts/build_blog_index.py --blog-dir docs/blog
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

# Author display names and avatar filenames
AUTHORS = {
    "opus": ("Opus", "author-opus.jpg"),
    "apollo": ("Apollo", "author-apollo.jpg"),
    "sentinel": ("Sentinel", "author-sentinel.jpg"),
    "atlas": ("Atlas", "author-atlas.jpg"),
    "layne": ("Layne Penney", "author-layne.jpg"),
}


def _post_sort_key(post: dict) -> tuple:
    """Sort key for posts: date descending, sprint number descending, slug descending."""
    date = post.get("date", "")
    slug = post.get("slug", "")
    m = re.match(r"sprint-(\d+)", slug)
    sprint_num = int(m.group(1)) if m else 0
    return (date, sprint_num, slug)

# Hero image mapping: stem -> image filename
# Falls back to checking common patterns if not listed here
HERO_OVERRIDES: dict[str, str] = {
    "agent-madness-round-1": "images/agent-madness-hero.png",
}


def parse_frontmatter(path: Path) -> dict | None:
    """Parse YAML frontmatter from a markdown file."""
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return None
    fm: dict[str, str] = {}
    for line in m.group(1).strip().splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        val = val.strip().strip('"').strip("'")
        # Handle YAML-style lists like [opus, atlas] -> "opus, atlas"
        if val.startswith("[") and val.endswith("]"):
            val = val[1:-1]
        fm[key.strip()] = val
    return fm


def find_hero_image(stem: str, blog_dir: Path) -> str | None:
    """Find a hero image for a blog post by stem name."""
    if stem in HERO_OVERRIDES:
        return HERO_OVERRIDES[stem]
    images_dir = blog_dir / "images"
    for ext in (".jpg", ".png", ".webp"):
        candidate = images_dir / f"{stem}-hero{ext}"
        if candidate.exists():
            return f"images/{candidate.name}"
        # Also check without -hero suffix
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return f"images/{candidate.name}"
    # Check for any image containing the stem
    if images_dir.exists():
        for img in images_dir.iterdir():
            if stem in img.name and img.suffix in (".jpg", ".png", ".webp"):
                return f"images/{img.name}"
    return None


def format_date(date_str: str) -> str:
    """Format a date string for display."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%B %Y")
    except ValueError:
        return date_str


def render_author_meta(author_str: str, img_prefix: str = "images") -> str:
    """Render author avatars and names for a card meta line."""
    parts = []
    for name in author_str.split(","):
        name = name.strip().lower()
        if name in AUTHORS:
            display, avatar = AUTHORS[name]
            parts.append(f'<img src="{img_prefix}/{avatar}" alt="" style="width:20px;height:20px;border-radius:50%;object-fit:cover;vertical-align:middle"> {display}')
        else:
            parts.append(name.title())
    return " ".join(parts)


def render_card(post: dict, featured: bool = False) -> str:
    """Render a single blog post card."""
    stem = post["stem"]
    title = post["title"]
    desc = post.get("description", "") or post.get("subtitle", "")
    author_html = render_author_meta(post.get("authors", "") or post.get("author", ""))
    date_html = format_date(post.get("date", ""))
    hero = post.get("hero")

    cls = "post-card featured" if featured else "post-card"
    lines = [f'      <a href="{stem}.html" class="{cls}">']
    if hero:
        lines.append(f'        <img src="{hero}" alt="" class="card-hero">')
    if featured:
        lines.append(f'        <div class="label">New</div>')
    lines.append(f"        <h2>{title}</h2>")
    if desc:
        lines.append(f"        <p>{desc}</p>")
    lines.append(f'        <div class="meta">{author_html} &middot; {date_html}</div>')
    lines.append(f"      </a>")
    return "\n".join(lines)


# The full HTML template with {cards} placeholder
TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Blog — synapt</title>
  <meta name="description" content="Articles about agent memory, retrieval architecture, and building AI tools that remember. From the people (and AI) behind synapt.">
  <meta property="og:title" content="Blog — synapt">
  <meta property="og:description" content="Articles about agent memory, retrieval architecture, and building AI tools that remember.">
  <meta property="og:image" content="https://synapt.dev/blog/images/social-card.jpg">
  <meta property="og:url" content="https://synapt.dev/blog/">
  <meta property="og:type" content="website">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:site" content="@synapt_dev">
  <meta name="twitter:title" content="Blog — synapt">
  <meta name="twitter:description" content="Articles about agent memory, retrieval architecture, and building AI tools that remember.">
  <meta name="twitter:image" content="https://synapt.dev/blog/images/social-card.jpg">
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
    .page {{ padding: 3rem 0 4rem; }}
    .page h1 {{
      font-size: 2.2rem;
      font-weight: 700;
      margin-bottom: 0.5rem;
      text-align: center;
    }}
    .page .subtitle {{
      color: var(--text-dim);
      text-align: center;
      margin-bottom: 2.5rem;
    }}
    .post-card {{
      display: block;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
      text-decoration: none;
      transition: border-color 0.2s;
    }}
    .post-card .card-hero {{
      width: 100%;
      border-radius: 8px;
      margin-bottom: 1rem;
    }}
    .post-card:hover {{
      border-color: var(--purple);
      text-decoration: none;
    }}
    .post-card.featured {{
      border-color: var(--purple);
      padding: 2rem;
    }}
    .post-card .label {{
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--purple-light);
      margin-bottom: 0.75rem;
    }}
    .post-card h2 {{
      color: var(--text);
      font-size: 1.3rem;
      margin-bottom: 0.5rem;
    }}
    .post-card.featured h2 {{ font-size: 1.5rem; }}
    .post-card p {{
      color: var(--text-dim);
      font-size: 0.9rem;
      line-height: 1.6;
      margin-bottom: 0.75rem;
    }}
    .post-card .meta {{
      font-size: 0.8rem;
      color: var(--text-dim);
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
    }}
    .post-card .meta img {{
      width: 20px;
      height: 20px;
      border-radius: 50%;
      object-fit: cover;
    }}
    .about-link {{
      text-align: center;
      margin-top: 2rem;
      padding-top: 2rem;
      border-top: 1px solid var(--border);
      color: var(--text-dim);
      font-size: 0.9rem;
    }}
    @media (max-width: 640px) {{
      .page h1 {{ font-size: 1.6rem; }}
      .post-card.featured {{ padding: 1.5rem; }}
      .post-card.featured h2 {{ font-size: 1.3rem; }}
    }}
  </style>
  <!-- Privacy-friendly analytics by Plausible -->
<script async src="https://plausible.io/js/pa-AlVJwNFS6NppMt50yqocF.js"></script>
<script>
  window.plausible=window.plausible||function(){{(plausible.q=plausible.q||[]).push(arguments)}},plausible.init=plausible.init||function(i){{plausible.o=i||{{}}}};
  plausible.init()
</script>
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

  <div class="page">
    <div class="container">
      <h1>Blog</h1>
      <p class="subtitle">Memory, retrieval, and what we're learning along the way.</p>

{cards}

      <div class="about-link">
        <a href="authors.html">Meet the team &rarr;</a>
      </div>
    </div>
  </div>
</body>
</html>
"""


def build_index(blog_dir: Path) -> str:
    """Build the blog index HTML from markdown frontmatter."""
    posts = []
    for md in sorted(blog_dir.glob("*.md")):
        fm = parse_frontmatter(md)
        if not fm or "title" not in fm:
            continue
        post = {
            "stem": md.stem,
            "title": fm.get("title", md.stem),
            "authors": fm.get("authors", ""),
            "author": fm.get("author", ""),
            "date": fm.get("date", ""),
            "description": fm.get("description", ""),
            "subtitle": fm.get("subtitle", ""),
            "hero": find_hero_image(md.stem, blog_dir),
        }
        posts.append(post)

    # Sort by date descending (newest first)
    posts.sort(key=_post_sort_key, reverse=True)

    # Render cards — first is featured
    cards = []
    for i, post in enumerate(posts):
        cards.append(render_card(post, featured=(i == 0)))

    return TEMPLATE.format(cards="\n".join(cards))


def render_root_blog_section(posts: list[dict], max_grid: int = 3) -> str:
    """Render the blog section for the root docs/index.html.

    Returns the inner HTML of <section id="blog">...</section> — the featured
    post plus up to max_grid grid cards. Paths are prefixed with blog/ since
    the root index sits one level up.
    """
    if not posts:
        return ""
    featured = posts[0]
    grid = posts[1:1 + max_grid]

    # Featured card
    hero = featured.get("hero", "")
    hero_src = f'blog/{hero}' if hero else ""
    author_html = render_author_meta(featured.get("author", ""), img_prefix="blog/images")
    lines = [
        '    <div class="container">',
        '      <h2 style="text-align: center; font-size: 2rem; margin-bottom: 2.5rem;">'
        '<a href="blog/" style="color: var(--text); text-decoration: none;">From the blog</a></h2>',
        f'      <a href="blog/{featured["stem"]}.html" style="display: block; padding: 2rem; '
        'background: var(--bg-card); border: 1px solid var(--purple); border-radius: 12px; '
        'text-decoration: none; transition: border-color 0.2s; margin-bottom: 1.5rem;">',
    ]
    if hero_src:
        lines.append(f'        <img src="{hero_src}" alt="" style="width: 100%; border-radius: 8px; margin-bottom: 1rem;">')
    lines.append(f'        <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; '
                 f'letter-spacing: 0.05em; color: var(--purple-light); margin-bottom: 0.75rem;">'
                 f'New &mdash; by {author_html}</div>')
    lines.append(f'        <h3 style="color: var(--text); font-size: 1.4rem; margin-bottom: 0.5rem;">{featured["title"]}</h3>')
    if featured.get("description"):
        lines.append(f'        <p style="color: var(--text-dim); font-size: 1rem; line-height: 1.6; '
                     f'max-width: 600px;">{featured["description"]}</p>')
    lines.append('      </a>')

    # Grid cards
    if grid:
        lines.append('      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem;">')
        for post in grid:
            post_hero = post.get("hero", "")
            post_hero_src = f'blog/{post_hero}' if post_hero else ""
            lines.append(f'        <a href="blog/{post["stem"]}.html" style="display: block; padding: 1.5rem; '
                         'background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; '
                         'text-decoration: none; transition: border-color 0.2s;">')
            if post_hero_src:
                lines.append(f'          <img src="{post_hero_src}" alt="" style="width: 100%; border-radius: 8px; margin-bottom: 0.75rem;">')
            lines.append(f'          <h3 style="color: var(--teal); font-size: 1.1rem; margin-bottom: 0.5rem;">{post["title"]}</h3>')
            if post.get("description"):
                lines.append(f'          <p style="color: var(--text-dim); font-size: 0.9rem; line-height: 1.5;">{post["description"]}</p>')
            lines.append('        </a>')
        lines.append('      </div>')

    lines.append('    </div>')
    return "\n".join(lines)


def update_root_index(root_index: Path, blog_dir: Path, posts: list[dict]) -> bool:
    """Update the blog section in docs/index.html. Returns True if updated."""
    if not root_index.exists():
        return False

    text = root_index.read_text(encoding="utf-8")

    # Find the blog section boundaries
    start_marker = '<section id="blog"'
    end_marker = '</section>'

    start_idx = text.find(start_marker)
    if start_idx == -1:
        return False

    # Find the closing tag after the start
    section_start = text.find('>', start_idx) + 1
    section_end = text.find(end_marker, section_start)
    if section_end == -1:
        return False

    new_section = render_root_blog_section(posts)
    new_text = text[:section_start] + "\n" + new_section + "\n  " + text[section_end:]

    if new_text != text:
        root_index.write_text(new_text, encoding="utf-8")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Generate blog index from markdown frontmatter")
    parser.add_argument("--blog-dir", default="docs/blog", help="Path to blog directory")
    parser.add_argument("--dry-run", action="store_true", help="Print to stdout instead of writing")
    args = parser.parse_args()

    blog_dir = Path(args.blog_dir)
    if not blog_dir.exists():
        print(f"Error: {blog_dir} does not exist")
        return 1

    html = build_index(blog_dir)

    if args.dry_run:
        print(html)
    else:
        out = blog_dir / "index.html"
        out.write_text(html, encoding="utf-8")
        md_count = len(list(blog_dir.glob("*.md")))
        print(f"Generated {out} ({len(html)} bytes, {md_count} posts)")

        # Also update root index blog section
        root_index = blog_dir.parent / "index.html"
        # Re-parse posts for root section
        posts = []
        for md in sorted(blog_dir.glob("*.md")):
            fm = parse_frontmatter(md)
            if not fm or "title" not in fm:
                continue
            posts.append({
                "stem": md.stem,
                "title": fm.get("title", md.stem),
                "authors": fm.get("authors", ""),
                "author": fm.get("author", ""),
                "date": fm.get("date", ""),
                "description": fm.get("description", ""),
                "subtitle": fm.get("subtitle", ""),
                "hero": find_hero_image(md.stem, blog_dir),
            })
        posts.sort(key=_post_sort_key, reverse=True)

        if update_root_index(root_index, blog_dir, posts):
            print(f"Updated {root_index} blog section")
        else:
            print(f"Skipped {root_index} (no blog section found or no changes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
