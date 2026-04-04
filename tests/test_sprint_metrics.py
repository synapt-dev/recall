"""Tests for sprint metrics computation."""

import sys
from pathlib import Path

# Add scripts dir to path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from sprint_metrics import compute_metrics, format_report


def _make_pr(number, author="agent-a", created="2026-04-04T17:00:00Z",
             merged="2026-04-04T17:30:00Z", additions=50, deletions=10,
             reviews=None):
    return {
        "number": number,
        "title": f"PR #{number}",
        "user": author,
        "created_at": created,
        "merged_at": merged,
        "additions": additions,
        "deletions": deletions,
        "changed_files": 2,
        "reviews": reviews or [],
    }


def test_cycle_time():
    """Cycle time is created → merged in hours."""
    prs = [_make_pr(1, created="2026-04-04T17:00:00Z", merged="2026-04-04T18:00:00Z")]
    m = compute_metrics(prs)
    assert m["cycle_time"]["mean_hours"] == 1.0


def test_review_turnaround():
    """Review turnaround is created → first review."""
    prs = [_make_pr(1, reviews=[
        {"user": "reviewer", "state": "COMMENTED", "submitted_at": "2026-04-04T17:15:00Z"},
        {"user": "reviewer", "state": "APPROVED", "submitted_at": "2026-04-04T17:30:00Z"},
    ])]
    m = compute_metrics(prs)
    assert m["review_turnaround_gh"]["mean_hours"] == 0.25


def test_first_pass_approval():
    """First-pass = APPROVED without CHANGES_REQUESTED."""
    prs = [
        _make_pr(1, reviews=[
            {"user": "r", "state": "APPROVED", "submitted_at": "2026-04-04T17:10:00Z"},
        ]),
        _make_pr(2, reviews=[
            {"user": "r", "state": "CHANGES_REQUESTED", "submitted_at": "2026-04-04T17:10:00Z"},
            {"user": "r", "state": "APPROVED", "submitted_at": "2026-04-04T17:20:00Z"},
        ]),
    ]
    m = compute_metrics(prs)
    assert m["first_pass_approval_rate"] == 50.0


def test_prs_per_author():
    """PRs counted per author."""
    prs = [
        _make_pr(1, author="atlas"),
        _make_pr(2, author="atlas"),
        _make_pr(3, author="apollo"),
    ]
    m = compute_metrics(prs)
    assert m["prs_per_author"] == {"atlas": 2, "apollo": 1}


def test_lines_changed():
    """Lines changed = additions + deletions."""
    prs = [_make_pr(1, additions=100, deletions=20)]
    m = compute_metrics(prs)
    assert m["lines_changed"]["total"] == 120
    assert m["lines_changed"]["mean"] == 120


def test_format_report_includes_title():
    """Format report includes sprint number in title."""
    m = compute_metrics([_make_pr(1)])
    report = format_report(m, sprint="4")
    assert "Sprint 4 Metrics" in report


def test_claim_to_pr():
    """Claim → PR latency when claim times provided."""
    prs = [_make_pr(1, created="2026-04-04T17:30:00Z")]
    prs[0]["title"] = "feat: broadcast mentions (#466)"
    claim_times = {"466": "2026-04-04T17:00:00Z"}
    m = compute_metrics(prs, claim_times)
    assert m["claim_to_pr"]["mean_hours"] == 0.5
    assert m["claim_to_pr"]["count"] == 1


def test_empty_prs():
    """Empty PR list returns error."""
    m = compute_metrics([])
    assert "error" in m
