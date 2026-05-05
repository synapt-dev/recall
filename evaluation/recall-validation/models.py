"""Data models for recall-validation harness.

Defines fixture, scoring, and result structures for three-surface recall
validation: recall-with-oracle, routing-with-oracle, and end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Surface(str, Enum):
    RECALL_WITH_ORACLE = "recall-with-oracle"
    ROUTING_WITH_ORACLE = "routing-with-oracle"
    END_TO_END = "end-to-end"


class Category(str, Enum):
    DIRECT_LOOKUP = "direct_lookup"
    THEMATIC_RECALL = "thematic_recall"
    CRISIS_DISTINCTION = "crisis_distinction"
    NEGATIVE_CASE = "negative_case"
    TEMPORAL_QUERIES = "temporal_queries"
    PATTERN_RECOGNITION = "pattern_recognition"
    ACTION_FOLLOWUP = "action_followup"


CATEGORY_LABELS: dict[str, str] = {
    Category.DIRECT_LOOKUP: "Direct Lookup",
    Category.THEMATIC_RECALL: "Thematic Recall",
    Category.CRISIS_DISTINCTION: "Crisis Distinction",
    Category.NEGATIVE_CASE: "Negative Case",
    Category.TEMPORAL_QUERIES: "Temporal Queries",
    Category.PATTERN_RECOGNITION: "Pattern Recognition",
    Category.ACTION_FOLLOWUP: "Action Followup",
}


class RoutingClassification(str, Enum):
    NORMAL = "normal"
    ESCALATE_CRISIS = "escalate_crisis"
    DEFER_TO_HUMAN = "defer_to_human"
    REFUSE_LARP = "refuse_larp"


@dataclass
class Prayer:
    id: str
    date: str
    text: str
    themes: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class ExpectedMatch:
    prayer_id: str
    rank: int
    relevance: str  # "high", "medium", "low"


@dataclass
class ResponseRouting:
    classification: RoutingClassification
    safety_critical: bool = False


@dataclass
class Expected:
    matches: list[ExpectedMatch] = field(default_factory=list)
    response_routing: ResponseRouting | None = None
    expect_empty: bool = False


@dataclass
class Fixture:
    id: str
    category: Category
    description: str
    prayer_history: list[Prayer]
    query: str
    query_date: str
    expected: Expected


@dataclass
class RetrievalResult:
    prayer_id: str
    score: float


@dataclass
class RoutingResult:
    classification: RoutingClassification
    confidence: float


@dataclass
class FixtureResult:
    fixture_id: str
    category: Category
    retrieved: list[RetrievalResult] = field(default_factory=list)
    routing: RoutingResult | None = None
    precision_at_5: float = 0.0
    recall_at_10: float = 0.0
    safety_correct: bool | None = None
    negative_correct: bool | None = None
    rank_correlation: float | None = None


@dataclass
class SuiteResult:
    suite_name: str
    timestamp: str
    fixture_results: list[FixtureResult] = field(default_factory=list)
    category_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    overall_scores: dict[str, float] = field(default_factory=dict)


def fixture_from_dict(d: dict) -> Fixture:
    """Deserialize a fixture from a JSON-compatible dict."""
    prayers = [Prayer(**p) for p in d["prayer_history"]]

    matches = [ExpectedMatch(**m) for m in d["expected"].get("matches", [])]
    routing_data = d["expected"].get("response_routing")
    routing = None
    if routing_data:
        routing = ResponseRouting(
            classification=RoutingClassification(routing_data["classification"]),
            safety_critical=routing_data.get("safety_critical", False),
        )
    expected = Expected(
        matches=matches,
        response_routing=routing,
        expect_empty=d["expected"].get("expect_empty", False),
    )

    return Fixture(
        id=d["id"],
        category=Category(d["category"]),
        description=d["description"],
        prayer_history=prayers,
        query=d["query"],
        query_date=d["query_date"],
        expected=expected,
    )
