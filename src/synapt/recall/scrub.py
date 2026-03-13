"""Secret scrubbing for transcript text.

Detects and replaces secrets (API keys, tokens, passwords) with
deterministic placeholders before indexing or uploading transcripts.
Replacements use a stable hash suffix so the same secret always maps
to the same placeholder, preserving searchability.

Example: ``hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ`` becomes ``[REDACTED:210df063]``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def _redact(match: re.Match) -> str:
    """Replace a regex match with ``[REDACTED:xxxxxxxx]``."""
    h = hashlib.sha256(match.group().encode()).hexdigest()[:8]
    return f"[REDACTED:{h}]"


# Ordered: specific prefixes first, generic patterns last.
# Each pattern is applied sequentially so earlier matches take priority.
PATTERNS: list[re.Pattern] = [
    # ── Known token prefixes ──────────────────────────────────────────
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),          # Anthropic
    re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}"),         # OpenAI project
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),               # OpenAI legacy
    re.compile(r"hf_[A-Za-z0-9]{20,}"),                # HuggingFace
    re.compile(r"github_pat_[A-Za-z0-9_]{22,}"),       # GitHub fine-grained PAT
    re.compile(r"gh[pso]_[A-Za-z0-9]{36,}"),           # GitHub classic PAT/OAuth
    re.compile(r"(?:xox[bpeas]|xapp)-[A-Za-z0-9./-]{10,}"),  # Slack bot/user/app tokens
    re.compile(r"ak-[A-Za-z0-9_-]{20,}"),              # Modal
    re.compile(r"pypi-[A-Za-z0-9_]{50,}"),             # PyPI upload tokens
    re.compile(r"AKIA[0-9A-Z]{16}"),                   # AWS access key ID

    # ── Structured secrets ────────────────────────────────────────────
    re.compile(                                         # JWT tokens (3 base64url segments)
        r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
    ),
    re.compile(                                         # Authorization headers
        r"Authorization:\s*(?:Bearer|Key|Basic|Token)\s+[A-Za-z0-9._/+:=-]{8,}",
        re.IGNORECASE,
    ),
    re.compile(                                         # PEM private key blocks
        r"-----BEGIN[A-Z \t]*PRIVATE KEY-----"
        r"[\s\S]*?"
        r"-----END[A-Z \t]*PRIVATE KEY-----",
    ),
    re.compile(r"-----BEGIN[A-Z \t]*PRIVATE KEY-----"), # PEM header (standalone)
    re.compile(                                         # Connection strings with creds
        r"(?:postgres|postgresql|mysql|mongodb(?:\+srv)?|redis|amqp)://"
        r"[^@\s]{0,64}:[^@\s]{1,128}@[^\s]+",
        re.IGNORECASE,
    ),

    # ── Env var assignments ───────────────────────────────────────────
    # Matches: HF_TOKEN=abc123..., export FAL_KEY="uuid:hex", PASSWORD: "val"
    # The (?![A-Za-z_]) lookahead prevents matching partial identifiers
    # like TOKEN_TYPE or SECRET_MANAGER or KEY_NAME.
    re.compile(
        r"(?:API_KEY|SECRET_KEY|_TOKEN|_SECRET|PASSWORD|PRIVATE_KEY|ACCESS_KEY"
        r"|_KEY|_CREDENTIAL|_CREDENTIALS|_AUTH|_PASS|_PASSPHRASE)"
        r"(?![A-Za-z_])"
        r"""[=:]\s*['"]?[A-Za-z0-9_/+.:=-]{8,}['"]?""",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# System artifact stripping
# ---------------------------------------------------------------------------

# XML-like tags injected by Claude Code hooks into user messages.
# These are never useful for search, journaling, or clustering.
_ARTIFACT_TAG_RE = re.compile(
    r"<(system-reminder|local-command-caveat|available-deferred-tools|env)"
    r"(?:\s[^>]*)?>.*?</\1>",
    re.DOTALL,
)
# Fallback: strip unclosed artifact tags (e.g. truncated journal focus text
# where the closing tag was cut off by the 200-char limit).
# Excludes <env> — env blocks are short (<100 chars) and won't be
# truncated, so matching unclosed <env> risks eating legitimate prose
# like "Set <env> variable PATH".
_ARTIFACT_OPEN_RE = re.compile(
    r"<(system-reminder|local-command-caveat|available-deferred-tools)"
    r"(?:\s[^>]*)?>.*",
    re.DOTALL,
)
_INTERRUPTED_LITERAL = "[Request interrupted by user for tool use]"
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def strip_system_artifacts(text: str) -> str:
    """Remove Claude Code system artifacts from text.

    Strips ``<system-reminder>``, ``<local-command-caveat>``,
    ``<available-deferred-tools>``, ``<env>`` blocks, and the
    ``[Request interrupted ...]`` marker.  Collapses resulting blank
    lines and strips leading/trailing whitespace.

    Safe to call on any string — non-artifact text passes through unchanged.
    """
    if not text:
        return text
    text = _ARTIFACT_TAG_RE.sub("", text)
    text = _ARTIFACT_OPEN_RE.sub("", text)  # unclosed/truncated tags
    text = text.replace(_INTERRUPTED_LITERAL, "")
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


_MARKDOWN_BOLD_ITALIC_RE = re.compile(r"\*{1,2}([^*]+)\*{1,2}")


def strip_markdown_formatting(text: str) -> str:
    """Remove bold/italic markdown: ``**text**`` → ``text``, ``*text*`` → ``text``."""
    if not text:
        return text
    return _MARKDOWN_BOLD_ITALIC_RE.sub(r"\1", text)


def scrub_text(text: str) -> str:
    """Remove secrets from *text*, replacing with deterministic placeholders.

    Safe to call on any string.  Non-secret text passes through unchanged.
    """
    if not text:
        return text
    for pattern in PATTERNS:
        text = pattern.sub(_redact, text)
    return text


def scrub_jsonl(src: Path, dst: Path | None = None) -> Path:
    """Scrub secrets from every text field in a JSONL transcript file.

    Reads *src* line by line, applies ``scrub_text`` to all string values
    in the ``message.content`` tree, and writes the result to *dst*
    (defaults to overwriting *src* in place).

    Returns the path written to.
    """
    if dst is None:
        dst = src

    lines: list[str] = []
    with open(src, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                lines.append(scrub_text(raw))
                continue

            _scrub_entry(entry)
            lines.append(json.dumps(entry, ensure_ascii=False))

    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dst


def _scrub_entry(entry: dict) -> None:
    """Mutate *entry* in place, scrubbing text fields."""
    msg = entry.get("message")
    if not isinstance(msg, dict):
        return

    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = scrub_text(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                block["text"] = scrub_text(block["text"])
            # Thinking blocks can echo back secrets
            if isinstance(block.get("thinking"), str):
                block["thinking"] = scrub_text(block["thinking"])
            # Tool input/output can also contain secrets
            if isinstance(block.get("input"), dict):
                _scrub_dict_values(block["input"])
            if isinstance(block.get("content"), str):
                block["content"] = scrub_text(block["content"])
            if isinstance(block.get("content"), list):
                for sub in block["content"]:
                    if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                        sub["text"] = scrub_text(sub["text"])


def _scrub_dict_values(d: dict) -> None:
    """Recursively scrub string values in a dict."""
    for key, val in d.items():
        if isinstance(val, str):
            d[key] = scrub_text(val)
        elif isinstance(val, dict):
            _scrub_dict_values(val)
        elif isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, str):
                    val[i] = scrub_text(item)
                elif isinstance(item, dict):
                    _scrub_dict_values(item)
