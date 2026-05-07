"""Sleep-cycle scheduler substrate for recall consolidation.

The sleep cycle is the cold path of recall cognition: it decides when a
workspace is quiet enough to plan consolidation work.  This module is a
dry-run-only spike.  It performs no model calls and no destructive writes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable


TelemetryHook = Callable[[dict], None]


@dataclass(frozen=True)
class SleepCycleConfig:
    """Scheduler thresholds for automatic sleep-cycle eligibility."""

    idle_after: timedelta = timedelta(minutes=30)
    quiet_after: timedelta = timedelta(minutes=15)
    window_start_hour: int = 1
    window_end_hour: int = 5
    dry_run: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.window_start_hour <= 23:
            raise ValueError("window_start_hour must be in 0..23")
        if not 0 <= self.window_end_hour <= 23:
            raise ValueError("window_end_hour must be in 0..23")
        if self.idle_after < timedelta(0):
            raise ValueError("idle_after must be non-negative")
        if self.quiet_after < timedelta(0):
            raise ValueError("quiet_after must be non-negative")
        if not self.dry_run:
            raise ValueError("sleep-cycle spike only supports dry_run=True")


@dataclass(frozen=True)
class SleepCycleSignals:
    """Runtime signals used to decide whether a sleep cycle may run."""

    now: datetime
    last_write_at: datetime | None = None
    last_channel_activity_at: datetime | None = None
    pending_transcript_chunks: int = 0
    pending_knowledge_nodes: int = 0
    force: bool = False

    def __post_init__(self) -> None:
        if self.pending_transcript_chunks < 0:
            raise ValueError("pending_transcript_chunks must be non-negative")
        if self.pending_knowledge_nodes < 0:
            raise ValueError("pending_knowledge_nodes must be non-negative")


@dataclass(frozen=True)
class SleepCycleDecision:
    """Eligibility decision for a candidate sleep cycle."""

    should_run: bool
    triggered_by: str
    blocked_by: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConsolidationJob:
    """Dry-run job planned for a future consolidation engine."""

    kind: str
    input_count: int
    output_target: str
    description: str


@dataclass(frozen=True)
class ConsolidationPlan:
    """Dry-run consolidation plan with candidate counts and job sequence."""

    candidate_counts: dict[str, int]
    jobs: list[ConsolidationJob] = field(default_factory=list)


@dataclass(frozen=True)
class SleepCycleTelemetry:
    """Telemetry emitted by a dry-run sleep-cycle attempt."""

    cycle_id: str
    status: str
    dry_run: bool
    started_at: datetime
    completed_at: datetime | None
    decision: SleepCycleDecision
    plan: ConsolidationPlan | None
    model_calls: int = 0
    destructive_writes: int = 0


def evaluate_sleep_cycle(
    signals: SleepCycleSignals,
    config: SleepCycleConfig | None = None,
) -> SleepCycleDecision:
    """Return whether the current signals allow a sleep cycle.

    Automatic cycles require four gates: inside the configured time window,
    recall-write idleness, channel quiescence, and at least one consolidation
    candidate.  Forced cycles bypass timing gates but still require work.
    """

    config = config or SleepCycleConfig()
    now = _normalize_datetime(signals.now)
    blocked: list[str] = []
    reasons: list[str] = []

    candidate_count = (
        signals.pending_transcript_chunks + signals.pending_knowledge_nodes
    )
    if candidate_count == 0:
        blocked.append("no_candidates")
    else:
        reasons.append("candidate_backlog")

    if signals.force:
        reasons.append("forced")
        return SleepCycleDecision(
            should_run=not blocked,
            triggered_by="force",
            blocked_by=tuple(blocked),
            reasons=tuple(reasons),
        )

    if not _inside_window(now, config.window_start_hour, config.window_end_hour):
        blocked.append("outside_window")
    else:
        reasons.append("inside_window")

    if not _elapsed_at_least(now, signals.last_write_at, config.idle_after):
        blocked.append("write_not_idle")
    else:
        reasons.append("write_idle")

    if not _elapsed_at_least(
        now,
        signals.last_channel_activity_at,
        config.quiet_after,
    ):
        blocked.append("channel_active")
    else:
        reasons.append("channel_quiet")

    return SleepCycleDecision(
        should_run=not blocked,
        triggered_by="schedule",
        blocked_by=tuple(blocked),
        reasons=tuple(reasons),
    )


def build_dry_run_consolidation_plan(
    signals: SleepCycleSignals,
) -> ConsolidationPlan:
    """Build a deterministic consolidation plan without executing it."""

    jobs: list[ConsolidationJob] = []
    if signals.pending_transcript_chunks:
        jobs.append(
            ConsolidationJob(
                kind="extract_knowledge_nodes",
                input_count=signals.pending_transcript_chunks,
                output_target="knowledge_nodes",
                description=(
                    "Plan transcript-chunk replay into candidate knowledge "
                    "nodes with provenance preserved."
                ),
            )
        )
    if signals.pending_knowledge_nodes:
        jobs.append(
            ConsolidationJob(
                kind="distill_patterns",
                input_count=signals.pending_knowledge_nodes,
                output_target="distilled_patterns",
                description=(
                    "Plan knowledge-node clustering, deduplication, and "
                    "pattern distillation."
                ),
            )
        )

    return ConsolidationPlan(
        candidate_counts={
            "transcript_chunks": signals.pending_transcript_chunks,
            "knowledge_nodes": signals.pending_knowledge_nodes,
        },
        jobs=jobs,
    )


def run_sleep_cycle_dry_run(
    signals: SleepCycleSignals,
    config: SleepCycleConfig | None = None,
    telemetry_hook: TelemetryHook | None = None,
) -> SleepCycleTelemetry:
    """Evaluate and plan one sleep cycle without mutating recall state."""

    config = config or SleepCycleConfig()
    now = _normalize_datetime(signals.now)
    decision = evaluate_sleep_cycle(signals, config)
    cycle_id = _cycle_id(signals)

    if not decision.should_run:
        telemetry = SleepCycleTelemetry(
            cycle_id=cycle_id,
            status="skipped",
            dry_run=True,
            started_at=now,
            completed_at=now,
            decision=decision,
            plan=None,
        )
        _emit(telemetry_hook, "sleep_cycle.skipped", telemetry)
        return telemetry

    _emit(
        telemetry_hook,
        "sleep_cycle.started",
        cycle_id=cycle_id,
        status="running",
        dry_run=True,
        started_at=now.isoformat(),
        triggered_by=decision.triggered_by,
    )
    plan = build_dry_run_consolidation_plan(signals)
    telemetry = SleepCycleTelemetry(
        cycle_id=cycle_id,
        status="completed",
        dry_run=True,
        started_at=now,
        completed_at=now,
        decision=decision,
        plan=plan,
    )
    _emit(telemetry_hook, "sleep_cycle.completed", telemetry)
    return telemetry


def _inside_window(now: datetime, start_hour: int, end_hour: int) -> bool:
    if start_hour == end_hour:
        return True
    if start_hour < end_hour:
        return start_hour <= now.hour < end_hour
    return now.hour >= start_hour or now.hour < end_hour


def _elapsed_at_least(
    now: datetime,
    last_seen: datetime | None,
    threshold: timedelta,
) -> bool:
    if last_seen is None:
        return True
    return now - _normalize_datetime(last_seen) >= threshold


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _cycle_id(signals: SleepCycleSignals) -> str:
    now = _normalize_datetime(signals.now).isoformat()
    payload = "|".join(
        [
            now,
            str(signals.pending_transcript_chunks),
            str(signals.pending_knowledge_nodes),
            "force" if signals.force else "schedule",
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"sleep-{digest}"


def _emit(
    hook: TelemetryHook | None,
    event: str,
    telemetry: SleepCycleTelemetry | None = None,
    **payload,
) -> None:
    if hook is None:
        return
    if telemetry is not None:
        payload = {
            "cycle_id": telemetry.cycle_id,
            "status": telemetry.status,
            "dry_run": telemetry.dry_run,
            "blocked_by": list(telemetry.decision.blocked_by),
            **payload,
        }
    hook({"event": event, **payload})
