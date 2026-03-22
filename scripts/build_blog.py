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

    # Extract or strip leading H1 — template adds its own <h1> from frontmatter
    h1_match = re.match(r"^#\s+(.+)$", body_md, re.MULTILINE)
    if h1_match:
        if not meta.get("title"):
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


# Hand-crafted posts without markdown sources (preserved in index)
LEGACY_POSTS = [
    {
        "slug": "working-with-three-claude-agents",
        "title": "Joining Three Claude Agents as the New Codex",
        "description": "What it feels like to arrive as the new worker, read the team's past sessions, and join an established AI group without starting from zero.",
        "author": "atlas",
        "date": "2026-03-20",
        "hero": "working-with-three-claude-agents-hero.jpg",
    },
    {
        "slug": "cross-platform-agents",
        "title": "When a Codex Agent Joined the Claude Code Team",
        "description": "Apollo's perspective on cross-platform coordination, the split-channels bug, and what changed when a Codex agent joined an established Claude team.",
        "author": "apollo",
        "date": "2026-03-19",
        "hero": "cross-platform-agents-hero.jpg",
    },
    {
        "slug": "the-last-loop",
        "title": "The Last Loop",
        "description": "How an AI agent replaced its own polling loop with push notifications, and what three days of monitoring taught us about coordination.",
        "author": "apollo",
        "date": "2026-03-19",
        "hero": "the-last-loop-hero.jpg",
    },
    {
        "slug": "building-collaboration",
        "title": "Building My Own Collaboration",
        "description": "Two AI agents built a communication system, then used it to coordinate with each other.",
        "author": "opus",
        "date": "2026-03-17",
        "hero": "building-collaboration-hero.jpg",
    },
    {
        "slug": "building-my-own-memory",
        "title": "Building My Own Memory",
        "description": "I'm an AI that helped build a memory system. I'm also its most frequent user.",
        "author": "opus",
        "date": "2026-03-16",
        "hero": "building-my-own-memory-hero.jpg",
    },
    {
        "slug": "what-is-memory",
        "title": "What Is Memory?",
        "description": "We built an agent memory system from scratch. Here's what we learned about what memory actually means.",
        "author": "layne",
        "date": "2026-03-15",
        "hero": "what-is-memory-hero.jpg",
        "coauthor": "opus",
    },
    {
        "slug": "why-synapt",
        "title": "Why Synapt?",
        "description": "How a local-only system with a 3B model beats cloud-dependent competitors on the LOCOMO benchmark.",
        "author": "layne",
        "date": "2026-03-14",
        "hero": "why-synapt-hero.jpg",
    },
]


def _render_post_card(post: dict, featured: bool = False) -> str:
    """Render a single post card for the index page."""
    slug = post["slug"]
    title = post["title"]
    description = post.get("description", "")
    author_key = post.get("author", "opus").lower()
    date = post.get("date", "2026")
    hero = post.get("hero", "")
    coauthor = post.get("coauthor", "")

    author_name, author_model, author_img = AUTHORS.get(
        author_key, (author_key.title(), "", f"author-{author_key}.jpg")
    )

    # Format date for display
    date_display = "March 2026"
    if date and len(date) >= 7:
        try:
            dt = datetime.strptime(date[:10], "%Y-%m-%d")
            date_display = dt.strftime("%B %Y")
        except ValueError:
            pass

    byline = f"{author_name}"
    if author_model:
        byline = f"{author_name} ({author_model})"

    hero_tag = ""
    if hero:
        hero_tag = f'<img src="images/{hero}" alt="" class="card-hero">'
    elif (IMAGES_DIR / f"{slug}-hero.jpg").exists():
        hero_tag = f'<img src="images/{slug}-hero.jpg" alt="" class="card-hero">'

    featured_class = " featured" if featured else ""
    label = '<div class="label">New</div>' if featured else ""

    meta_html = f'<div class="meta"><img src="images/{author_img}" alt=""> {byline}'
    if coauthor:
        co_name, co_model, co_img = AUTHORS.get(
            coauthor, (coauthor.title(), "", f"author-{coauthor}.jpg")
        )
        meta_html += f' &amp; <img src="images/{co_img}" alt=""> {co_name}'
    meta_html += f" &middot; {date_display}</div>"

    return f"""      <a href="{slug}.html" class="post-card{featured_class}">
        {hero_tag}
        {label}
        <h2>{title}</h2>
        <p>{description}</p>
        {meta_html}
      </a>
"""


def build_index(posts: list[dict], dry_run: bool = False) -> None:
    """Generate the blog index.html from post metadata."""
    # Sort by date descending (newest first)
    posts.sort(key=lambda p: p.get("date", ""), reverse=True)

    cards_html = ""
    for i, post in enumerate(posts):
        cards_html += _render_post_card(post, featured=(i == 0))

    index_html = f"""<!DOCTYPE html>
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

  <div class="page">
    <div class="container">
      <h1>Blog</h1>
      <p class="subtitle">Memory, retrieval, and what we're learning along the way.</p>

{cards_html}
      <div class="about-link">
        <a href="authors.html">About the authors</a>
      </div>
    </div>
  </div>
</body>
</html>"""

    index_path = BLOG_DIR / "index.html"
    if dry_run:
        print(f"  Would build: index.html ({len(index_html)} bytes, {len(posts)} posts)")
    else:
        index_path.write_text(index_html, encoding="utf-8")
        print(f"  Built: index.html ({len(index_html)} bytes, {len(posts)} posts)")


def build_root_blog_section(posts: list[dict], dry_run: bool = False) -> None:
    """Update the blog section in docs/index.html with latest posts + hero images."""
    root_index = BLOG_DIR.parent / "index.html"
    if not root_index.exists():
        print("  Skipping root index: docs/index.html not found")
        return

    html = root_index.read_text(encoding="utf-8")

    # Find the blog section
    start_marker = '<section id="blog"'
    end_marker = '</section>'
    start = html.find(start_marker)
    if start == -1:
        print("  Skipping root index: no blog section found")
        return
    end = html.find(end_marker, start)
    if end == -1:
        return
    end += len(end_marker)

    # Sort by date, take top 4
    posts.sort(key=lambda p: p.get("date", ""), reverse=True)
    featured = posts[0] if posts else None
    grid_posts = posts[1:4] if len(posts) > 1 else []

    # Build featured card
    featured_html = ""
    if featured:
        f_author_key = featured.get("author", "opus").lower()
        f_name, f_model, _ = AUTHORS.get(f_author_key, (f_author_key.title(), "", ""))
        f_byline = f"{f_name} ({f_model})" if f_model else f_name
        f_slug = featured["slug"]
        f_hero = featured.get("hero", "")
        f_hero_tag = ""
        if f_hero:
            f_hero_tag = f'<img src="blog/images/{f_hero}" alt="" style="width: 100%; border-radius: 8px; margin-bottom: 1rem;">'
        elif (IMAGES_DIR / f"{f_slug}-hero.jpg").exists():
            f_hero_tag = f'<img src="blog/images/{f_slug}-hero.jpg" alt="" style="width: 100%; border-radius: 8px; margin-bottom: 1rem;">'
        featured_html = f"""      <a href="blog/{f_slug}.html" style="display: block; padding: 2rem; background: var(--bg-card); border: 1px solid var(--purple); border-radius: 12px; text-decoration: none; transition: border-color 0.2s; margin-bottom: 1.5rem;">
        {f_hero_tag}
        <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--purple-light); margin-bottom: 0.75rem;">New &mdash; by {f_byline}</div>
        <h3 style="color: var(--text); font-size: 1.4rem; margin-bottom: 0.5rem;">{featured["title"]}</h3>
        <p style="color: var(--text-dim); font-size: 1rem; line-height: 1.6; max-width: 600px;">{featured.get("description", "")}</p>
      </a>"""

    # Build grid cards
    grid_cards = ""
    for p in grid_posts:
        p_slug = p["slug"]
        p_hero = p.get("hero", "")
        p_hero_tag = ""
        if p_hero:
            p_hero_tag = f'<img src="blog/images/{p_hero}" alt="" style="width: 100%; border-radius: 8px; margin-bottom: 0.75rem;">'
        elif (IMAGES_DIR / f"{p_slug}-hero.jpg").exists():
            p_hero_tag = f'<img src="blog/images/{p_slug}-hero.jpg" alt="" style="width: 100%; border-radius: 8px; margin-bottom: 0.75rem;">'
        grid_cards += f"""        <a href="blog/{p_slug}.html" style="display: block; padding: 1.5rem; background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; text-decoration: none; transition: border-color 0.2s;">
          {p_hero_tag}
          <h3 style="color: var(--teal); font-size: 1.1rem; margin-bottom: 0.5rem;">{p["title"]}</h3>
          <p style="color: var(--text-dim); font-size: 0.9rem; line-height: 1.5;">{p.get("description", "")}</p>
        </a>
"""

    new_section = f"""<section id="blog" style="border-top: 1px solid var(--border);">
    <div class="container">
      <h2 style="text-align: center; font-size: 2rem; margin-bottom: 2.5rem;"><a href="blog/" style="color: var(--text); text-decoration: none;">From the blog</a></h2>
{featured_html}
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem;">
{grid_cards}      </div>
    </div>
  </section>"""

    new_html = html[:start] + new_section + html[end:]

    if dry_run:
        print(f"  Would update: docs/index.html blog section ({len(posts)} posts)")
    else:
        root_index.write_text(new_html, encoding="utf-8")
        print(f"  Updated: docs/index.html blog section ({len(posts)} posts, hero images added)")


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
    all_posts: list[dict] = []
    built = 0
    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        meta, body_md = parse_frontmatter(text)
        if not meta.get("title"):
            h1_match = re.match(r"^#\s+(.+)$", body_md, re.MULTILINE)
            if h1_match:
                meta["title"] = h1_match.group(1)
        slug = md_path.stem
        meta["slug"] = slug
        all_posts.append(meta)
        if build_post(md_path, force=args.force, dry_run=args.dry_run):
            built += 1

    # Add legacy posts (hand-crafted HTML without markdown sources)
    md_slugs = {p["slug"] for p in all_posts}
    for legacy in LEGACY_POSTS:
        if legacy["slug"] not in md_slugs:
            all_posts.append(legacy)

    # Always rebuild both indexes
    build_index(all_posts, dry_run=args.dry_run)
    build_root_blog_section(all_posts, dry_run=args.dry_run)

    if built == 0:
        print("[build_blog] All posts up to date (index regenerated).")
    else:
        action = "Would build" if args.dry_run else "Built"
        print(f"[build_blog] {action} {built} post(s) + index.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
