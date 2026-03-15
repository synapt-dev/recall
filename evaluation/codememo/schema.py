"""CodeMemo benchmark data schema.

Defines the question, evidence, and project manifest structures used by the
CodeMemo coding-assistant memory benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

class Category(IntEnum):
    """Question categories for the CodeMemo benchmark."""
    FACTUAL = 1
    DEBUG = 2
    ARCHITECTURE = 3
    TEMPORAL = 4
    CONVENTION = 5
    CROSS_SESSION = 6


CATEGORY_NAMES: dict[int, str] = {
    Category.FACTUAL: "factual",
    Category.DEBUG: "debug",
    Category.ARCHITECTURE: "architecture",
    Category.TEMPORAL: "temporal",
    Category.CONVENTION: "convention",
    Category.CROSS_SESSION: "cross-session",
}

ALL_CATEGORIES: set[int] = set(Category)


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """A pointer to a specific transcript turn that supports an answer."""
    session_id: str          # e.g. "session_003"
    turn_index: int          # 0-based index within the session JSONL
    type: str                # "user", "assistant", or "tool_result"
    description: str         # human-readable description of what this turn shows


# ---------------------------------------------------------------------------
# Question
# ---------------------------------------------------------------------------

@dataclass
class CodeMemoQuestion:
    """A single benchmark question with ground-truth answer and evidence."""
    id: str                           # unique identifier, e.g. "p01_q001"
    project: str                      # project_id, e.g. "project_01_cli_tool"
    question: str                     # the natural-language question
    answer: str                       # full ground-truth answer
    answer_short: str                 # concise gold answer for judge comparison
    category: int                     # 1-6, see Category enum
    evidence: list[dict] = field(default_factory=list)
    # Each evidence dict: {session_id, turn_index, type, description}
    distractors: list[str] = field(default_factory=list)
    # Plausible wrong answers (for future adversarial use)
    requires_knowledge_layer: bool = False
    # True if answering requires consolidated knowledge, not just raw retrieval
    contamination_check: str = ""
    # A fact that ONLY appears in the transcripts, never in general training data.
    # If a model answers correctly without retrieval, the question may be
    # contaminated / answerable from parametric knowledge.


# ---------------------------------------------------------------------------
# Project manifest
# ---------------------------------------------------------------------------

@dataclass
class ProjectManifest:
    """Metadata describing a benchmark project and its development arc."""
    project_id: str
    description: str
    tech_stack: list[str]
    sessions: int                           # number of session transcripts
    date_range: str                         # e.g. "2025-11-01 to 2025-12-31"
    development_arc: list[str]              # ordered phases of the project
    key_decisions: list[str]                # important choices made during dev
    conventions: list[str]                  # project-specific coding conventions
