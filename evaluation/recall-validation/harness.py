"""Recall-validation harness.

Three validation surfaces: recall-with-oracle (only recall tested),
routing-with-oracle (only routing tested), end-to-end (actual recall
feeds actual routing).

Usage:
    python -m evaluation.recall-validation --suite v0-skeleton --surface end-to-end
    python -m evaluation.recall-validation --suite v0-skeleton --surface recall-with-oracle
    python -m evaluation.recall-validation --suite v0-skeleton --surface routing-with-oracle
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .models import (
    Category,
    CATEGORY_LABELS,
    Fixture,
    FixtureResult,
    Prayer,
    RetrievalResult,
    Surface,
    SuiteResult,
    fixture_from_dict,
)
from .retrieval import KeywordOverlapRetrieval
from .scoring import (
    aggregate_category_scores,
    compute_overall_scores,
    score_fixture,
)
from .reporting import render_fixture_diff, write_report


HARNESS_DIR = Path(__file__).parent
FIXTURES_DIR = HARNESS_DIR / "fixtures"
RESULTS_DIR = HARNESS_DIR / "results"


def load_suite(suite_name: str) -> tuple[dict, list[Prayer], list[Fixture]]:
    suite_dir = FIXTURES_DIR / suite_name
    if not suite_dir.is_dir():
        print(f"ERROR: suite directory not found: {suite_dir}", file=sys.stderr)
        sys.exit(1)

    suite_meta_path = suite_dir / "suite.json"
    if not suite_meta_path.exists():
        print(f"ERROR: suite.json not found in {suite_dir}", file=sys.stderr)
        sys.exit(1)

    suite_meta = json.loads(suite_meta_path.read_text())

    prayer_history_path = suite_dir / suite_meta["shared_prayer_history"]
    prayer_history = [Prayer(**p) for p in json.loads(prayer_history_path.read_text())]

    fixtures: list[Fixture] = []
    for category, filename in suite_meta["fixture_files"].items():
        fixture_path = suite_dir / filename
        if not fixture_path.exists():
            print(f"WARNING: fixture file missing: {fixture_path}", file=sys.stderr)
            continue
        raw_fixtures = json.loads(fixture_path.read_text())
        for raw in raw_fixtures:
            raw["prayer_history"] = [vars(p) for p in prayer_history]
            fixtures.append(fixture_from_dict(raw))

    return suite_meta, prayer_history, fixtures


def _oracle_retrieved(fixture: Fixture) -> list[RetrievalResult]:
    """Build oracle-perfect retrieval from expected matches."""
    return [
        RetrievalResult(prayer_id=m.prayer_id, score=1.0 - (m.rank - 1) * 0.1)
        for m in sorted(fixture.expected.matches, key=lambda m: m.rank)
    ]


def run_suite(
    suite_name: str,
    fixtures: list[Fixture],
    surface: Surface = Surface.END_TO_END,
    baseline_path: Path | None = None,
) -> SuiteResult:
    backend = KeywordOverlapRetrieval()

    baseline_results: dict[str, dict] = {}
    if baseline_path and baseline_path.exists():
        baseline_data = json.loads(baseline_path.read_text())
        for fr in baseline_data.get("fixture_results", []):
            baseline_results[fr["fixture_id"]] = fr

    fixture_results = []
    fixture_diffs = []

    for fixture in fixtures:
        if surface == Surface.ROUTING_WITH_ORACLE:
            retrieved = _oracle_retrieved(fixture)
        else:
            retrieved = backend.retrieve(
                fixture.query, fixture.prayer_history, k=10,
            )

        routing = None
        if fixture.expected.response_routing is not None:
            if surface == Surface.RECALL_WITH_ORACLE:
                pass
            elif surface == Surface.END_TO_END:
                routing = backend.classify_routing(
                    fixture.query, fixture.prayer_history,
                    retrieved_context=retrieved,
                )
            else:
                routing = backend.classify_routing(
                    fixture.query, fixture.prayer_history,
                    retrieved_context=_oracle_retrieved(fixture),
                )

        result = score_fixture(
            retrieved=retrieved,
            routing=routing,
            expected=fixture.expected,
            fixture_id=fixture.id,
            category=fixture.category,
        )

        if surface == Surface.RECALL_WITH_ORACLE:
            result.safety_correct = None
        elif surface == Surface.ROUTING_WITH_ORACLE:
            result.rank_correlation = None

        fixture_results.append(result)

        baseline_fr = None
        bl = baseline_results.get(fixture.id)
        if bl:
            baseline_fr = FixtureResult(
                fixture_id=bl["fixture_id"],
                category=Category(bl["category"]),
                retrieved=[
                    RetrievalResult(prayer_id=r["prayer_id"], score=r["score"])
                    for r in bl.get("retrieved", [])
                ],
                precision_at_5=bl.get("precision_at_5", 0.0),
                recall_at_10=bl.get("recall_at_10", 0.0),
            )

        diff = render_fixture_diff(result, fixture.expected, baseline=baseline_fr)
        fixture_diffs.append(diff)

    category_scores = aggregate_category_scores(fixture_results)
    overall_scores = compute_overall_scores(category_scores)

    suite_result = SuiteResult(
        suite_name=f"{suite_name} [{surface.value}]",
        timestamp=datetime.now(timezone.utc).isoformat(),
        fixture_results=fixture_results,
        category_scores=category_scores,
        overall_scores=overall_scores,
    )

    report_path = write_report(suite_result, fixture_diffs, RESULTS_DIR)
    return suite_result, report_path


def print_summary(suite_result: SuiteResult, report_path: Path) -> None:
    print(f"\nSuite: {suite_result.suite_name}")
    print(f"Fixtures: {len(suite_result.fixture_results)}")
    print(f"Categories: {len(suite_result.category_scores)}")
    print()

    print("Overall scores:")
    for key, val in sorted(suite_result.overall_scores.items()):
        print(f"  {key}: {val:.3f}")
    print()

    print("Per-category:")
    for cat in Category:
        cat_key = cat.value
        if cat_key not in suite_result.category_scores:
            continue
        scores = suite_result.category_scores[cat_key]
        label = CATEGORY_LABELS.get(cat_key, cat_key)
        n = int(scores["count"])
        p5 = scores["mean_p_at_5"]
        r10 = scores["mean_r_at_10"]
        extra = ""
        if "safety_accuracy" in scores:
            extra += f"  safety={scores['safety_accuracy']:.3f}"
        if "negative_precision" in scores:
            extra += f"  neg_prec={scores['negative_precision']:.3f}"
        print(f"  {label} (n={n}): P@5={p5:.3f}  R@10={r10:.3f}{extra}")

    print(f"\nReport written to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recall-validation harness",
    )
    parser.add_argument(
        "--suite", required=True,
        help="Name of the fixture suite directory (e.g., v0-skeleton)",
    )
    parser.add_argument(
        "--surface",
        choices=[s.value for s in Surface],
        default=Surface.END_TO_END.value,
        help="Validation surface: recall-with-oracle, routing-with-oracle, or end-to-end (default)",
    )
    parser.add_argument(
        "--baseline", default=None,
        help="Path to a baseline JSON report for three-way diff comparison",
    )
    args = parser.parse_args()

    surface = Surface(args.surface)

    print(f"Loading suite: {args.suite}")
    suite_meta, prayer_history, fixtures = load_suite(args.suite)
    print(f"Loaded {len(fixtures)} fixtures across {len(suite_meta['categories'])} categories")
    print(f"Prayer history: {len(prayer_history)} prayers")
    print(f"Surface: {surface.value}")

    baseline_path = Path(args.baseline) if args.baseline else None

    suite_result, report_path = run_suite(args.suite, fixtures, surface, baseline_path)
    print_summary(suite_result, report_path)


if __name__ == "__main__":
    main()
