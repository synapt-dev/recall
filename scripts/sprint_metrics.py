#!/usr/bin/env python3
"""Sprint metrics — cycle time, review turnaround, approval rate.

Scrapes GitHub PRs for velocity/quality data, and optionally channel
JSONL for claim timestamps (claim → PR created latency).

Note: review turnaround and first-pass approval rate are GitHub-only
metrics. In-channel approvals (which are common in multi-agent sprints)
are not captured here — those would need channel JSONL parsing for
"approved"/"LGTM" signals.

Usage:
    python scripts/sprint_metrics.py [--sprint 4] [--repo laynepenney/synapt]
    python scripts/sprint_metrics.py --prs 471,472,475,477,478,479,481
    python scripts/sprint_metrics.py --prs 471,472 --channel-dir ~/.synapt/recall/channels
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _gh_json(cmd: list[str]) -> list[dict] | dict:
    """Run a gh CLI command and parse JSON output."""
    result = subprocess.run(
        ["gh"] + cmd, capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"gh error: {result.stderr.strip()}", file=sys.stderr)
        return []
    return json.loads(result.stdout)


def _parse_iso(ts: str) -> datetime:
    """Parse ISO timestamp to datetime."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def parse_claim_times(channel_dir: Path | None) -> dict[str, str]:
    """Parse claim timestamps from channel JSONL.

    Returns {issue_number: timestamp} for claim messages.
    Looks for type="claim" messages in dev.jsonl.
    """
    if not channel_dir:
        return {}
    dev_path = channel_dir / "dev.jsonl"
    if not dev_path.exists():
        return {}
    claims: dict[str, str] = {}
    for line in dev_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("type") == "claim":
            # body format: "display claimed 466"
            body = msg.get("body", "")
            parts = body.rsplit(" ", 1)
            if len(parts) == 2:
                issue = parts[-1].strip()
                if issue.isdigit():
                    claims[issue] = msg.get("timestamp", "")
    return claims


def _hours_between(start: str, end: str) -> float:
    """Hours between two ISO timestamps."""
    return (_parse_iso(end) - _parse_iso(start)).total_seconds() / 3600


def fetch_pr_data(repo: str, pr_numbers: list[int]) -> list[dict]:
    """Fetch PR metadata + reviews from GitHub."""
    prs = []
    for num in pr_numbers:
        pr = _gh_json([
            "api", f"repos/{repo}/pulls/{num}",
            "--jq", ".",
        ])
        if not pr:
            continue

        reviews = _gh_json([
            "api", f"repos/{repo}/pulls/{num}/reviews",
            "--jq", ".",
        ])

        prs.append({
            "number": pr.get("number", num),
            "title": pr.get("title", ""),
            "user": pr.get("user", {}).get("login", ""),
            "created_at": pr.get("created_at", ""),
            "merged_at": pr.get("merged_at", ""),
            "additions": pr.get("additions", 0),
            "deletions": pr.get("deletions", 0),
            "changed_files": pr.get("changed_files", 0),
            "reviews": [
                {
                    "user": r.get("user", {}).get("login", ""),
                    "state": r.get("state", ""),
                    "submitted_at": r.get("submitted_at", ""),
                }
                for r in (reviews if isinstance(reviews, list) else [])
            ],
        })
    return prs


def compute_metrics(prs: list[dict], claim_times: dict[str, str] | None = None) -> dict:
    """Compute sprint metrics from PR data."""
    if not prs:
        return {"error": "No PRs found"}

    claim_times = claim_times or {}
    cycle_times: list[float] = []
    claim_to_pr: list[float] = []
    review_turnarounds: list[float] = []
    first_pass_approvals = 0
    lines_changed: list[int] = []
    prs_per_author: dict[str, int] = {}
    review_iterations: list[int] = []

    for pr in prs:
        author = pr["user"]
        prs_per_author[author] = prs_per_author.get(author, 0) + 1
        lines_changed.append(pr["additions"] + pr["deletions"])

        # Cycle time: created → merged
        if pr["created_at"] and pr["merged_at"]:
            hours = _hours_between(pr["created_at"], pr["merged_at"])
            cycle_times.append(hours)

        # Claim → PR created (from channel JSONL)
        # PR title format: "feat: ... (#NNN)" — extract issue number
        title = pr.get("title", "")
        issue_match = re.search(r"#(\d+)", title)
        if issue_match and pr["created_at"]:
            issue_num = issue_match.group(1)
            if issue_num in claim_times:
                hours = _hours_between(claim_times[issue_num], pr["created_at"])
                if hours >= 0:
                    claim_to_pr.append(hours)

        # Review turnaround (GitHub-only): created → first review
        reviews = pr.get("reviews", [])
        if reviews and pr["created_at"]:
            first_review = min(
                r["submitted_at"] for r in reviews if r["submitted_at"]
            )
            hours = _hours_between(pr["created_at"], first_review)
            review_turnarounds.append(hours)

        # First-pass approval: approved without CHANGES_REQUESTED
        states = [r["state"] for r in reviews]
        if "APPROVED" in states and "CHANGES_REQUESTED" not in states:
            first_pass_approvals += 1

        # Review iterations: count of reviews per PR
        review_iterations.append(len(reviews))

    total = len(prs)
    metrics = {
        "total_prs": total,
        "cycle_time": {
            "mean_hours": round(sum(cycle_times) / len(cycle_times), 2) if cycle_times else None,
            "min_hours": round(min(cycle_times), 2) if cycle_times else None,
            "max_hours": round(max(cycle_times), 2) if cycle_times else None,
        },
        "claim_to_pr": {
            "mean_hours": round(sum(claim_to_pr) / len(claim_to_pr), 2) if claim_to_pr else None,
            "count": len(claim_to_pr),
        },
        "review_turnaround_gh": {
            "mean_hours": round(sum(review_turnarounds) / len(review_turnarounds), 2) if review_turnarounds else None,
            "min_hours": round(min(review_turnarounds), 2) if review_turnarounds else None,
            "max_hours": round(max(review_turnarounds), 2) if review_turnarounds else None,
        },
        "first_pass_approval_rate": round(first_pass_approvals / total * 100, 1) if total else 0,
        "lines_changed": {
            "mean": round(sum(lines_changed) / len(lines_changed)) if lines_changed else 0,
            "total": sum(lines_changed),
        },
        "review_iterations": {
            "mean": round(sum(review_iterations) / len(review_iterations), 1) if review_iterations else 0,
        },
        "prs_per_author": prs_per_author,
    }
    return metrics


def format_report(metrics: dict, sprint: str = "") -> str:
    """Format metrics as a readable report."""
    title = f"Sprint {sprint} Metrics" if sprint else "Sprint Metrics"
    lines = [f"## {title}", ""]

    ct = metrics.get("cycle_time", {})
    if ct.get("mean_hours") is not None:
        lines.append(f"**Cycle time** (created → merged): {ct['mean_hours']:.1f}h avg "
                      f"(min {ct['min_hours']:.1f}h, max {ct['max_hours']:.1f}h)")

    ctp = metrics.get("claim_to_pr", {})
    if ctp.get("mean_hours") is not None:
        lines.append(f"**Claim → PR** (channel claim → PR created): {ctp['mean_hours']:.1f}h avg "
                      f"({ctp['count']} PRs matched)")

    rt = metrics.get("review_turnaround_gh", {})
    if rt.get("mean_hours") is not None:
        lines.append(f"**Review turnaround (GH)** (created → first GH review): {rt['mean_hours']:.1f}h avg "
                      f"(min {rt['min_hours']:.1f}h, max {rt['max_hours']:.1f}h)")

    lines.append(f"**First-pass approval rate**: {metrics.get('first_pass_approval_rate', 0)}%")

    lc = metrics.get("lines_changed", {})
    lines.append(f"**Lines changed**: {lc.get('total', 0)} total, {lc.get('mean', 0)} avg/PR")

    ri = metrics.get("review_iterations", {})
    lines.append(f"**Reviews per PR**: {ri.get('mean', 0)}")

    lines.append(f"**Total PRs**: {metrics.get('total_prs', 0)}")

    ppa = metrics.get("prs_per_author", {})
    if ppa:
        lines.append("")
        lines.append("**PRs per author:**")
        for author, count in sorted(ppa.items(), key=lambda x: -x[1]):
            lines.append(f"  {author}: {count}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Sprint metrics")
    parser.add_argument("--repo", default="laynepenney/synapt")
    parser.add_argument("--prs", help="Comma-separated PR numbers")
    parser.add_argument("--sprint", default="")
    parser.add_argument("--channel-dir", help="Path to channel JSONL dir for claim times")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if not args.prs:
        print("Usage: python scripts/sprint_metrics.py --prs 471,472,475,...", file=sys.stderr)
        sys.exit(1)

    pr_numbers = [int(n.strip()) for n in args.prs.split(",")]
    print(f"Fetching {len(pr_numbers)} PRs from {args.repo}...", file=sys.stderr)

    prs = fetch_pr_data(args.repo, pr_numbers)
    claim_times = parse_claim_times(Path(args.channel_dir) if args.channel_dir else None)
    metrics = compute_metrics(prs, claim_times)

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print(format_report(metrics, args.sprint))


if __name__ == "__main__":
    main()
