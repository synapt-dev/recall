"""Sleep-cycle scheduler spike tests.

The spike is intentionally dry-run only.  It locks the substrate seam for
nightly recall consolidation without performing model calls or destructive
storage operations.
"""

from datetime import datetime, timedelta, timezone

from synapt.recall.sleep_cycle import (
    SleepCycleConfig,
    SleepCycleSignals,
    evaluate_sleep_cycle,
    run_sleep_cycle_dry_run,
)


NOW = datetime(2026, 5, 7, 8, 30, tzinfo=timezone.utc)


def _config() -> SleepCycleConfig:
    return SleepCycleConfig(
        idle_after=timedelta(minutes=30),
        quiet_after=timedelta(minutes=15),
        window_start_hour=0,
        window_end_hour=23,
        dry_run=True,
    )


def _signals(**overrides) -> SleepCycleSignals:
    values = {
        "now": NOW,
        "last_write_at": NOW - timedelta(hours=2),
        "last_channel_activity_at": NOW - timedelta(hours=1),
        "pending_transcript_chunks": 3,
        "pending_knowledge_nodes": 2,
    }
    values.update(overrides)
    return SleepCycleSignals(**values)


def test_idle_quiescent_window_triggers_dry_run_plan():
    events: list[dict] = []

    telemetry = run_sleep_cycle_dry_run(
        _signals(),
        _config(),
        telemetry_hook=events.append,
    )

    assert telemetry.status == "completed"
    assert telemetry.dry_run is True
    assert telemetry.decision.should_run is True
    assert telemetry.model_calls == 0
    assert telemetry.destructive_writes == 0
    assert telemetry.plan is not None
    assert [job.kind for job in telemetry.plan.jobs] == [
        "extract_knowledge_nodes",
        "distill_patterns",
    ]
    assert telemetry.plan.candidate_counts == {
        "transcript_chunks": 3,
        "knowledge_nodes": 2,
    }
    assert [event["event"] for event in events] == [
        "sleep_cycle.started",
        "sleep_cycle.completed",
    ]


def test_active_channel_blocks_automatic_cycle():
    decision = evaluate_sleep_cycle(
        _signals(last_channel_activity_at=NOW - timedelta(minutes=5)),
        _config(),
    )

    assert decision.should_run is False
    assert "channel_active" in decision.blocked_by


def test_outside_window_blocks_automatic_cycle():
    config = SleepCycleConfig(
        idle_after=timedelta(minutes=30),
        quiet_after=timedelta(minutes=15),
        window_start_hour=1,
        window_end_hour=5,
        dry_run=True,
    )

    telemetry = run_sleep_cycle_dry_run(_signals(), config)

    assert telemetry.status == "skipped"
    assert telemetry.plan is None
    assert "outside_window" in telemetry.decision.blocked_by
    assert telemetry.model_calls == 0
    assert telemetry.destructive_writes == 0


def test_no_candidates_skips_without_mutation():
    telemetry = run_sleep_cycle_dry_run(
        _signals(pending_transcript_chunks=0, pending_knowledge_nodes=0),
        _config(),
    )

    assert telemetry.status == "skipped"
    assert telemetry.plan is None
    assert "no_candidates" in telemetry.decision.blocked_by
    assert telemetry.model_calls == 0
    assert telemetry.destructive_writes == 0


def test_force_trigger_bypasses_timing_gates_but_stays_dry_run():
    config = SleepCycleConfig(
        idle_after=timedelta(hours=4),
        quiet_after=timedelta(hours=2),
        window_start_hour=1,
        window_end_hour=5,
        dry_run=True,
    )
    telemetry = run_sleep_cycle_dry_run(
        _signals(
            force=True,
            last_write_at=NOW - timedelta(minutes=1),
            last_channel_activity_at=NOW - timedelta(minutes=1),
        ),
        config,
    )

    assert telemetry.status == "completed"
    assert telemetry.decision.should_run is True
    assert telemetry.decision.triggered_by == "force"
    assert telemetry.model_calls == 0
    assert telemetry.destructive_writes == 0


def test_window_that_wraps_midnight_allows_late_night_cycle():
    late_night = datetime(2026, 5, 7, 2, 15, tzinfo=timezone.utc)
    config = SleepCycleConfig(
        idle_after=timedelta(minutes=30),
        quiet_after=timedelta(minutes=15),
        window_start_hour=22,
        window_end_hour=5,
        dry_run=True,
    )

    decision = evaluate_sleep_cycle(
        _signals(
            now=late_night,
            last_write_at=late_night - timedelta(hours=2),
            last_channel_activity_at=late_night - timedelta(hours=1),
        ),
        config,
    )

    assert decision.should_run is True
