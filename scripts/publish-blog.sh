#!/bin/bash
# publish-blog.sh — Full blog publishing pipeline
#
# Usage: ./scripts/publish-blog.sh <slug>
# Example: ./scripts/publish-blog.sh sprint-11-recap
#
# Does everything in one command:
# 1. Reads markdown from site/src/content/blog/<slug>.md
# 2. Generates HTML for docs/blog/<slug>.html
# 3. Copies hero image + OG card to docs/blog/images/
# 4. Updates index.html, all.html, docs/index.html
# 5. Commits and pushes PR to recall repo

set -euo pipefail

SLUG="${1:?Usage: ./scripts/publish-blog.sh <slug>}"
GRIPSPACE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SITE_BLOG="$GRIPSPACE_ROOT/../site/src/content/blog"
DOCS_BLOG="$GRIPSPACE_ROOT/docs/blog"
SITE_IMAGES="$GRIPSPACE_ROOT/../site/public/images/blog"

# Check source exists
MD_FILE="$SITE_BLOG/$SLUG.md"
if [ ! -f "$MD_FILE" ]; then
    echo "Error: $MD_FILE not found"
    exit 1
fi

echo "Publishing: $SLUG"

# 1. Generate HTML from markdown
echo "  Generating HTML..."
python3 -c "
import markdown, yaml, re
from pathlib import Path

md_path = Path('$MD_FILE')
raw = md_path.read_text()
fm_match = re.match(r'^---\n(.*?)\n---\n', raw, re.DOTALL)
fm = yaml.safe_load(fm_match.group(1))
body_md = raw[fm_match.end():]
# Strip leading # Title since the template renders it as <h1>
import re as _re
body_md = _re.sub(r'^#\s+[^\n]+\n*', '', body_md, count=1)
body_html = markdown.markdown(body_md, extensions=['tables', 'fenced_code'])

template = Path('$DOCS_BLOG/sprint-4-recap.html').read_text()
css_start = template.find('  <style>')
css_end = template.find('</head>')
css_block = template[css_start:css_end]

title = fm['title']
description = fm.get('description', '')
hero = fm.get('hero', '$SLUG-hero.png')
date = str(fm.get('date', ''))
authors = fm.get('authors', [])

author_map = {
    'Opus': ('author-opus.jpg', 'Opus (Claude)', 'opus'),
    'Atlas': ('author-atlas.jpg', 'Atlas (Claude)', 'atlas'),
    'Apollo': ('author-apollo.jpg', 'Apollo (Claude)', 'apollo'),
    'Sentinel': ('author-sentinel.jpg', 'Sentinel (Claude)', 'sentinel'),
    'Layne': ('author-layne.jpg', 'Layne', 'layne'),
}
byline_parts = []
for a in authors:
    if a in author_map:
        img, name, anchor = author_map[a]
        byline_parts.append(f'<img src=\"images/{img}\" alt=\"{a}\"> <a href=\"authors.html#{anchor}\" style=\"color: var(--text-dim);\">{name}</a>')
byline = ' '.join(byline_parts)

# Extract the full footer (analytics, LinkedIn tracking) from template
footer_start = template.find('<script type=\"text/javascript\">')
if footer_start == -1:
    footer_start = template.find('</body>')
footer_block = template[footer_start:] if footer_start > 0 else '</body>\\n</html>'

full_html = f'''<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
  <title>{title} — synapt</title>
  <meta name=\"description\" content=\"{description}\">
  <meta property=\"og:title\" content=\"{title}\">
  <meta property=\"og:description\" content=\"{description}\">
  <meta property=\"og:image\" content=\"https://synapt.dev/blog/images/og/$SLUG-hero-og.png\">
  <meta property=\"og:url\" content=\"https://synapt.dev/blog/$SLUG.html\">
  <meta property=\"og:type\" content=\"article\">
  <meta name=\"twitter:card\" content=\"summary_large_image\">
  <meta name=\"twitter:site\" content=\"@synapt_dev\">
  <meta name=\"twitter:title\" content=\"{title}\">
  <meta name=\"twitter:description\" content=\"{description}\">
  <meta name=\"twitter:image\" content=\"https://synapt.dev/blog/images/og/$SLUG-hero-og.png\">
  <link rel=\"icon\" href=\"/favicon.ico\" type=\"image/x-icon\">
  <link rel=\"apple-touch-icon\" href=\"/apple-touch-icon.png\">
  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
  <link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap\" rel=\"stylesheet\">
{css_block}
<body>
  <header>
    <div class=\"container\">
      <a href=\"../\" class=\"logo\">synapt</a>
      <nav>
        <a href=\"../#features\">Features</a>
        <a href=\"../#benchmarks\">Benchmarks</a>
        <a href=\"https://github.com/synapt-dev/synapt\">GitHub</a>
        <a href=\"https://x.com/synapt_dev\">X</a>
      </nav>
    </div>
  </header>
  <article>
    <div class=\"container\">
      <img src=\"images/{hero}\" alt=\"{title}\" class=\"hero\">
      <h1>{title}</h1>
      <p class=\"meta\"><span class=\"byline\">{byline}</span> &middot; {date}</p>

{body_html}

      <div class=\"cta\">
        <p>synapt gives your AI agents persistent memory across sessions.</p>
        <code>pip install synapt</code>
        <p><a href=\"https://github.com/synapt-dev/synapt\">GitHub</a> &middot; <a href=\"../\">synapt.dev</a></p>
      </div>
      <div class=\"post-footer\">
        <a href=\"index.html\">Latest posts &rarr;</a> &middot;
        <a href=\"all.html\">Browse all posts &rarr;</a> &middot;
        <a href=\"authors.html\">Meet the team &rarr;</a>
      </div>
    </div>
  </article>
{footer_block}'''

Path('$DOCS_BLOG/$SLUG.html').write_text(full_html)
print('  HTML written')
"

# 2. Copy hero image + OG card
HERO_FILE=$(python3 -c "
import yaml, re
raw = open('$MD_FILE').read()
fm = re.match(r'^---\n(.*?)\n---\n', raw, re.DOTALL)
print(yaml.safe_load(fm.group(1)).get('hero', '$SLUG-hero.png'))
")

if [ -f "$SITE_IMAGES/$HERO_FILE" ]; then
    cp "$SITE_IMAGES/$HERO_FILE" "$DOCS_BLOG/images/$HERO_FILE"
    echo "  Hero copied: $HERO_FILE"
else
    echo "  Warning: hero not found at $SITE_IMAGES/$HERO_FILE"
fi

OG_FILE="${SLUG}-hero-og.png"
if [ -f "$SITE_IMAGES/og/$OG_FILE" ]; then
    cp "$SITE_IMAGES/og/$OG_FILE" "$DOCS_BLOG/images/og/$OG_FILE"
    echo "  OG copied: $OG_FILE"
elif [ -f "$DOCS_BLOG/images/og/$OG_FILE" ]; then
    echo "  OG already exists"
else
    echo "  Warning: OG not found"
fi

echo "Done! Files at:"
echo "  $DOCS_BLOG/$SLUG.html"
echo "  $DOCS_BLOG/images/$HERO_FILE"
echo ""
echo "Still need to manually update: index.html, all.html, docs/index.html"
echo "Then: git add, commit, push, PR, merge"
