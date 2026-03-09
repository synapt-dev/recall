"""Auto-tag extraction for recall clusters.

Extracts structured tags from journal entries linked to clusters:
  - issue:NNN — from #NNN references in git_log, focus, done
  - branch:name — from non-default branch names
  - keyword:term — top distinctive tokens from journal focus text

Tags are appended to cluster search_text for FTS searchability.
No LLM required — pure regex and token analysis.

Phase 10 of the adaptive memory system.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

from synapt.recall.bm25 import _tokenize

if TYPE_CHECKING:
    from synapt.recall.journal import JournalEntry

# Regex: #123 (issue refs), but not inside hex hashes or URLs
_ISSUE_RE = re.compile(r"(?<![a-fA-F0-9])#(\d{1,5})\b")

# Branches that aren't worth tagging
_SKIP_BRANCHES = frozenset({"main", "master", "develop", "dev", "HEAD"})

# Reuse the clustering stop tokens to filter keywords
from synapt.recall.clustering import _STOP_TOKENS


def extract_issue_refs(text: str) -> list[str]:
    """Extract issue references from text.

    >>> extract_issue_refs("fix #287 and #290")
    ['issue:287', 'issue:290']
    """
    if not text:
        return []
    seen: set[str] = set()
    tags: list[str] = []
    for m in _ISSUE_RE.finditer(text):
        num = m.group(1)
        tag = f"issue:{num}"
        if tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


def extract_branch_tag(branch: str) -> str | None:
    """Extract a branch tag, skipping default branches.

    >>> extract_branch_tag("feat/clustering")
    'branch:feat/clustering'
    >>> extract_branch_tag("main") is None
    True
    """
    if not branch or branch in _SKIP_BRANCHES:
        return None
    return f"branch:{branch}"


def extract_tags(
    cluster: dict,
    journal_entries: list[JournalEntry],
) -> list[str]:
    """Extract all tags for a cluster from linked journal entries.

    Matches journal entries to the cluster via session_ids overlap.
    Returns deduplicated tag list (issue refs, branches, keywords).
    """
    cluster_sessions = set(cluster.get("session_ids", []))
    if not cluster_sessions or not journal_entries:
        return []

    # Find journal entries that overlap with this cluster's sessions
    matched: list[JournalEntry] = []
    for entry in journal_entries:
        if entry.session_id and entry.session_id in cluster_sessions:
            matched.append(entry)

    tags: list[str] = []
    seen: set[str] = set()

    def _add(tag: str) -> None:
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)

    # Issue refs from all text fields
    for entry in matched:
        text_parts = [entry.focus] + entry.done + entry.git_log
        for part in text_parts:
            for tag in extract_issue_refs(part):
                _add(tag)

    # Branch tags
    for entry in matched:
        tag = extract_branch_tag(entry.branch)
        if tag:
            _add(tag)

    # Keywords: top-3 distinctive tokens from journal focus texts
    focus_texts = [e.focus for e in matched if e.focus]
    if focus_texts:
        all_tokens = _tokenize(" ".join(focus_texts))
        # Filter stop tokens and very short tokens
        filtered = [t for t in all_tokens if t not in _STOP_TOKENS and len(t) > 2]
        # Count frequencies, take top-3 most common
        counts = Counter(filtered)
        for token, _ in counts.most_common(3):
            _add(f"keyword:{token}")

    return tags
