"""Content profile detection for adaptive filtering and retrieval.

Classifies conversation content as code-project or personal/prose based on
chunk-level signals. The profile adjusts consolidation filters and retrieval
parameters so that code-project heuristics (low-specificity filter, knowledge
caps) don't damage personal conversational memory.

Analogous to how human memory works differently for procedural vs episodic:
you don't apply the same recall strategies to "how to compile the project"
and "what did Sarah say about moving to Denver."
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------

# File path patterns (code projects reference files heavily)
_FILE_PATH_RE = re.compile(
    r"(?:^|\s)(?:\.?/)?(?:src|lib|app|pkg|cmd|internal|test|spec|docs)/"
    r"|\.(?:py|ts|tsx|js|jsx|rs|go|java|kt|swift|rb|cpp|c|h|css|html|yaml|yml|toml|json|md)\b"
    r"|(?:^|\s)(?:requirements\.txt|package\.json|Cargo\.toml|go\.mod|Makefile|Dockerfile)\b"
)

# Tool use indicators (Claude Code sessions have tool_use/tool_result blocks)
_TOOL_BLOCK_RE = re.compile(r'"type":\s*"tool_(?:use|result)"')

# Code block indicators (markdown fences in assistant messages)
_CODE_FENCE_RE = re.compile(r"```\w*\n")

# Technical identifiers
_TECH_IDENT_RE = re.compile(
    r"\b[A-Z][a-z]+[A-Z]\w*\b"   # CamelCase
    r"|\b[a-z]\w+_\w+\b"          # snake_case
    r"|\b(?:def|class|fn|func|import|from|require|include|use)\s"
)

# Personal conversation indicators
_PERSONAL_RE = re.compile(
    r"\b(?:feel|felt|feelings?|happy|sad|excited|worried|nervous|love|hate|enjoy"
    r"|friend|family|mom|dad|sister|brother|wife|husband|boyfriend|girlfriend"
    r"|birthday|wedding|vacation|trip|dinner|lunch|concert|movie|book|pet"
    r"|dog|cat|turtle|hobby|hobbies|weekend|summer|winter|holiday|miss you"
    r"|how are you|how've you been|long time no see)\b",
    re.IGNORECASE,
)

# Person name addressing (e.g., "Hi Sarah", "What did John say")
_NAME_ADDRESS_RE = re.compile(
    r"\b(?:Hi|Hey|Dear|Thanks?|What (?:did|does|has|is) )\s+[A-Z][a-z]{2,}\b"
)


@dataclass
class ContentProfile:
    """Content classification result with signal counts."""

    total_chunks: int = 0

    # Code signals
    file_refs: int = 0
    tool_uses: int = 0
    code_fences: int = 0
    tech_idents: int = 0

    # Personal signals
    personal_refs: int = 0
    name_addresses: int = 0

    # Derived
    _type: str = ""  # "code", "personal", "mixed"

    @property
    def content_type(self) -> str:
        """Classify as code, personal, or mixed."""
        if self._type:
            return self._type
        if self.total_chunks == 0:
            return "mixed"

        code_score = (
            min(self.file_refs / self.total_chunks, 1.0) * 3.0
            + min(self.tool_uses / self.total_chunks, 1.0) * 2.0
            + min(self.code_fences / self.total_chunks, 1.0) * 1.5
            + min(self.tech_idents / self.total_chunks, 1.0) * 1.0
        )
        personal_score = (
            min(self.personal_refs / self.total_chunks, 1.0) * 3.0
            + min(self.name_addresses / self.total_chunks, 1.0) * 2.0
        )

        # Thresholds tuned so that LOCOMO (pure personal) → "personal"
        # and typical Claude Code sessions → "code"
        if code_score > 2.0 and code_score > personal_score * 2:
            self._type = "code"
        elif personal_score > 1.0 and personal_score > code_score * 0.5:
            self._type = "personal"
        else:
            self._type = "mixed"
        return self._type

    @property
    def is_code(self) -> bool:
        return self.content_type == "code"

    @property
    def is_personal(self) -> bool:
        return self.content_type == "personal"


def detect_content_profile(chunks: list) -> ContentProfile:
    """Classify content type from a list of transcript chunks.

    Args:
        chunks: List of TranscriptChunk objects (or anything with a .text attribute).

    Returns:
        ContentProfile with signal counts and derived content type.
    """
    profile = ContentProfile(total_chunks=len(chunks))

    for chunk in chunks:
        text = getattr(chunk, "text", str(chunk))

        if _FILE_PATH_RE.search(text):
            profile.file_refs += 1
        if _TOOL_BLOCK_RE.search(text):
            profile.tool_uses += 1
        if _CODE_FENCE_RE.search(text):
            profile.code_fences += 1
        if _TECH_IDENT_RE.search(text):
            profile.tech_idents += 1
        if _PERSONAL_RE.search(text):
            profile.personal_refs += 1
        if _NAME_ADDRESS_RE.search(text):
            profile.name_addresses += 1

    return profile


# ---------------------------------------------------------------------------
# Adaptive parameters based on content profile
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveParams:
    """Adjusted parameters based on content profile."""

    # Consolidation filters
    specificity_threshold: int = 120       # chars — below this, low-specificity filter activates
    generic_filter_enabled: bool = True
    garbled_filter_enabled: bool = True

    # Indexing
    subchunk_min_text: int = 1200              # Min chars to trigger sub-chunk splitting (0 = disabled)

    # Retrieval
    dedup_jaccard: float = 0.70
    max_knowledge_default: int | None = None   # None = no cap
    knowledge_boost_adjust: float = 0.0        # Added to intent-based boost

    @property
    def specificity_filter_enabled(self) -> bool:
        return self.specificity_threshold < 10000


def adaptive_params(profile: ContentProfile) -> AdaptiveParams:
    """Return adjusted parameters based on content profile.

    Code projects get tight filters (the current defaults).
    Personal conversations get relaxed filters.
    Mixed gets middle ground.
    """
    if profile.is_code:
        return AdaptiveParams(
            specificity_threshold=80,           # Stricter — catch short generic tool output
            generic_filter_enabled=True,
            garbled_filter_enabled=True,
            subchunk_min_text=1200,          # Code turns benefit from tool-boundary splitting
            dedup_jaccard=0.8,              # Coding chunks are naturally repetitive (same files)
            max_knowledge_default=None,
            knowledge_boost_adjust=0.0,
        )
    elif profile.is_personal:
        return AdaptiveParams(
            specificity_threshold=10000,    # Effectively disabled
            generic_filter_enabled=False,   # Personal facts look "generic" to code filters
            garbled_filter_enabled=True,    # Still catch LLM artifacts
            subchunk_min_text=0,            # Disable sub-chunking — personal turns need full context
            dedup_jaccard=0.6,              # Preserve distinct personal evidence chunks
            max_knowledge_default=0,        # Disable knowledge in retrieval — raw chunks only
            knowledge_boost_adjust=0.0,     # N/A with max_knowledge=0
        )
    else:  # mixed
        return AdaptiveParams(
            specificity_threshold=200,      # Relaxed but not disabled
            generic_filter_enabled=True,
            garbled_filter_enabled=True,
            subchunk_min_text=2000,          # Higher threshold — only split very long turns
            dedup_jaccard=0.70,             # Lowered from 0.75 — formatted blocks with context lines dilute Jaccard
            max_knowledge_default=5,
            knowledge_boost_adjust=0.0,
        )


def forced_content_profile(total_chunks: int = 0) -> ContentProfile | None:
    """Return an env-forced content profile, if configured.

    Supported values:
    - code
    - personal
    - mixed
    """
    forced = os.environ.get("SYNAPT_FORCE_PROFILE", "").strip().lower()
    if forced not in {"code", "personal", "mixed"}:
        return None
    return ContentProfile(total_chunks=total_chunks, _type=forced)
