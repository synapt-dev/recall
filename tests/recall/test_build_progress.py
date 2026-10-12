"""TDD spec: `synapt recall build` must not go silent after "build: clustering N transcript chunks...".

Contract under test
-------------------
A real 250,638-chunk build printed that line and then nothing for 55+ minutes at ~95% CPU, so a watcher could not tell
working from hung. The rule is that no loop of the build is silent for longer than the progress interval (30 s): every
loop after the clustering line reports ``n/N``, a rate and an ETA at least that often, and a phase that is one opaque
SQL statement says when it starts and when it ends.

The first half pins the helper that does the reporting, including a property over random per-item costs (the gap
between two lines never exceeds the interval plus the slowest single item, and a line is never sent sooner than the
interval). The second half pins that ``cluster_chunks`` ticks in each of its own loops, and that doing so changes
nothing it returns.
"""

from __future__ import annotations

import logging
import random
import re

import pytest

from synapt.recall import progress
from synapt.recall.progress import ProgressLog


class _Clock:
    """A hand-driven clock: the test says how long each item 'took'."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _log(total=100, interval=30.0, label="clustering: group", unit="chunks"):
    clock = _Clock()
    lines: list[tuple[float, str]] = []
    log = ProgressLog(label, total, unit, interval=interval, clock=clock, emit=lambda text: lines.append((clock.now, text)))
    return log, clock, lines


# ---------------------------------------------------------------------------
# A. The helper
# ---------------------------------------------------------------------------

def test_no_line_before_the_interval_has_passed():
    log, clock, lines = _log()
    for i in range(1, 11):
        clock.now += 2.0  # 20 s in total, under the 30 s interval
        log.tick(i)
    assert lines == [], f"a line was emitted before the interval: {lines}"


def test_a_line_names_label_count_total_rate_and_eta():
    log, clock, lines = _log(total=100)
    for i in range(1, 16):
        clock.now += 2.0  # 30 s in total at i == 15
        log.tick(i)
    assert len(lines) == 1, lines
    text = lines[0][1]
    assert text.startswith("clustering: group: 15/100 chunks"), text
    assert "0.5/s" in text or "0/s" in text or "1/s" in text, f"no rate in {text!r}"
    # 85 items left at 0.5 items/s is 170 s
    assert "170s remaining" in text, text


def test_the_eta_follows_the_recent_rate_not_the_lifetime_average():
    """The greedy clustering loop gets slower as it goes, so an ETA off the lifetime average is far too optimistic."""
    log, clock, lines = _log(total=1000)
    done = 0
    for _ in range(30):  # 10 items a second for 30 s
        clock.now += 1.0
        done += 10
        log.tick(done)
    for _ in range(30):  # then 1 item a second for the next 30 s
        clock.now += 1.0
        done += 1
        log.tick(done)
    assert len(lines) == 2, lines
    assert "(10/s, 70s remaining)" in lines[0][1], lines[0]
    # recent rate 1/s over 670 items left is 670 s; the lifetime average (330 items in 60 s) would say 122 s
    assert "670s remaining" in lines[1][1], lines[1]


def test_a_line_is_not_sent_sooner_than_the_interval_and_not_later_than_one_item_past_it():
    """Property over random per-item costs: gaps between lines are in [interval, interval + the slowest item]."""
    for seed in range(25):
        rng = random.Random(seed)
        log, clock, lines = _log(total=5000, interval=30.0)
        slowest = 0.0
        for i in range(1, 5001):
            cost = rng.choice([0.001, 0.01, 0.2, 2.0]) if rng.random() < 0.97 else rng.uniform(5, 20)
            slowest = max(slowest, cost)
            clock.now += cost
            log.tick(i)
        times = [t for t, _ in lines]
        assert times, f"seed {seed}: a run of {clock.now - 1000.0:.0f} s produced no line"
        gaps = [b - a for a, b in zip(times, times[1:])]
        assert all(g >= 30.0 - 1e-9 for g in gaps), f"seed {seed}: a line came sooner than the interval: {min(gaps)}"
        assert all(g <= 30.0 + slowest + 1e-9 for g in gaps), f"seed {seed}: silent for {max(gaps)} s with items of at most {slowest}"
        assert times[0] - 1000.0 <= 30.0 + slowest + 1e-9, f"seed {seed}: the first line was {times[0] - 1000.0} s in"


def test_done_always_emits_the_final_line_even_when_the_interval_never_passed():
    log, clock, lines = _log(total=10)
    for i in range(1, 11):
        clock.now += 0.1
        log.tick(i)
    assert lines == []
    log.done()
    assert len(lines) == 1 and "10/10 chunks" in lines[0][1], lines


def test_done_with_nothing_to_do_does_not_divide_by_zero():
    log, _clock, lines = _log(total=0)
    log.done()
    assert len(lines) == 1 and "0/0" in lines[0][1], lines


def test_a_zero_elapsed_tick_does_not_divide_by_zero():
    log, _clock, lines = _log(total=10, interval=0.0)
    log.tick(1)  # the clock has not moved: elapsed is 0
    assert len(lines) == 1, lines


def test_a_count_past_the_total_is_reported_as_given_and_never_negative_eta():
    log, clock, lines = _log(total=10, interval=0.0)
    clock.now += 1.0
    log.tick(12)
    assert "12/10" in lines[0][1] and "-" not in lines[0][1].split("(")[1], lines


def test_tick_without_a_count_counts_one_item_at_a_time():
    log, clock, lines = _log(total=3, interval=0.0)
    for _ in range(3):
        clock.now += 1.0
        log.tick()
    assert [t.split(": ")[-1].split(" ")[0] for _, t in lines] == ["1/3", "2/3", "3/3"], lines


def test_the_default_interval_is_thirty_seconds_and_is_read_when_the_log_is_made(monkeypatch):
    assert progress.PROGRESS_INTERVAL_S == 30.0
    monkeypatch.setattr(progress, "PROGRESS_INTERVAL_S", 0.0)
    lines: list[str] = []
    log = ProgressLog("x", 5, "things", clock=_Clock(), emit=lines.append)
    log.tick(1)
    assert len(lines) == 1, "the module interval was not read at construction"


def test_note_sends_one_line_as_given():
    lines: list[str] = []
    progress.note("clusters: replacing the stored clusters with 3 clusters and 9 memberships", emit=lines.append)
    assert lines == ["clusters: replacing the stored clusters with 3 clusters and 9 memberships"]


def test_phase_names_its_start_and_its_end_for_one_opaque_statement():
    clock = _Clock()
    lines: list[str] = []
    with progress.phase("clusters: FTS rebuild", clock=clock, emit=lines.append):
        clock.now += 4.2
    assert lines[0] == "clusters: FTS rebuild: starting", lines
    assert lines[1].startswith("clusters: FTS rebuild: done in 4.2s"), lines


def test_phase_still_reports_its_end_when_the_statement_raises():
    clock = _Clock()
    lines: list[str] = []
    with pytest.raises(RuntimeError):
        with progress.phase("p", clock=clock, emit=lines.append):
            raise RuntimeError("boom")
    assert len(lines) == 2 and "failed after" in lines[1], lines


# ---------------------------------------------------------------------------
# B. cluster_chunks ticks in every loop of its own, and returns what it returned before
# ---------------------------------------------------------------------------

def _corpus(n_topics=6, per_topic=12):
    from synapt.recall.core import TranscriptChunk

    topics = [
        "alpha deploy pipeline kubernetes rollout canary staging release rollback manifest helm chart",
        "beta database migration postgres index vacuum replica failover backup restore schema table",
        "gamma frontend component render hydrate bundle webpack typescript hooks state props store",
        "delta incident postmortem outage latency alert pager oncall mitigation runbook timeline",
        "epsilon benchmark retrieval recall embedding vector rerank hybrid score judge metric corpus",
        "zeta billing invoice subscription plan stripe webhook refund proration tax ledger account",
    ][:n_topics]
    chunks = []
    for t, text in enumerate(topics):
        for i in range(per_topic):
            chunks.append(
                TranscriptChunk(
                    id=f"s{t}:t{i}",
                    session_id=f"sess-{t}",
                    timestamp=f"2026-03-{(i % 27) + 1:02d}T10:{i:02d}:00Z",
                    turn_index=i,
                    user_text=f"{text} question number {i}",
                    assistant_text=f"{text} answer {i} with detail {text}",
                )
            )
    return chunks


def _strip(clusters):
    return [{k: v for k, v in c.items() if k not in ("created_at", "updated_at")} for c in clusters]


@pytest.fixture
def capture(monkeypatch):
    """Every progress line, in order, with the interval at zero so each tick reports."""
    lines: list[str] = []
    monkeypatch.setattr(progress, "PROGRESS_INTERVAL_S", 0.0)
    monkeypatch.setattr(progress, "_emit", lines.append)
    return lines


def test_cluster_chunks_ticks_in_its_tokenise_group_and_topic_loops(capture):
    from synapt.recall.clustering import cluster_chunks

    chunks = _corpus()
    clusters = cluster_chunks(chunks)
    assert clusters, "the corpus must cluster, or the topic loop never runs"
    for label in ("clustering: tokenize", "clustering: group", "clustering: vocabulary", "clustering: topics"):
        _assert_reports_from_inside(capture, label)
    assert _loop_lines(capture, "clustering: group")[1] == [(len(chunks), len(chunks))]


_PROGRESS = re.compile(r": (\d+)/(\d+) \w+ \(")          # "label: 12/72 chunks (3/s, 40s remaining)"
_CLOSING = re.compile(r": (\d+)/(\d+) \w+ in [\d.]+s$")  # "label: 72/72 chunks in 1.2s"


def _loop_lines(lines, label):
    """(progress lines as (n, total), closing lines as (n, total)) for one label. A closing line is NOT a progress line:
    a loop that never ticked still closes, and says 0/N."""
    mine = [l for l in lines if l.startswith(label + ": ")]
    progress_lines = [(int(m.group(1)), int(m.group(2))) for l in mine if (m := _PROGRESS.search(l))]
    closing = [(int(m.group(1)), int(m.group(2))) for l in mine if (m := _CLOSING.search(l))]
    return progress_lines, closing


def _assert_reports_from_inside(lines, label):
    progress_lines, closing = _loop_lines(lines, label)
    assert progress_lines, f"no progress line from {label!r}: {[l for l in lines if l.startswith(label)]}"
    assert any(n < total for n, total in progress_lines), (
        f"{label!r} never reported before its end, so a long loop would still be silent: {progress_lines}"
    )
    assert len(closing) == 1, f"{label!r} must close exactly once: {closing}"
    n, total = closing[0]
    assert n == total, f"{label!r} closed at {n}/{total}: the loop's own count never reached its total"


def test_progress_changes_nothing_cluster_chunks_returns(monkeypatch):
    from synapt.recall.clustering import cluster_chunks

    chunks = _corpus()
    monkeypatch.setattr(progress, "PROGRESS_INTERVAL_S", 1e9)
    quiet = _strip(cluster_chunks(chunks))
    monkeypatch.setattr(progress, "PROGRESS_INTERVAL_S", 0.0)
    monkeypatch.setattr(progress, "_emit", lambda text: None)
    loud = _strip(cluster_chunks(chunks))
    assert quiet == loud and quiet, "progress reporting changed the clusters"


def test_cluster_chunks_reports_nothing_for_nothing(capture):
    from synapt.recall.clustering import cluster_chunks

    assert cluster_chunks([]) == []
    assert capture == []


# ---------------------------------------------------------------------------
# C. The whole build: every loop after the clustering line reports from inside itself
# ---------------------------------------------------------------------------

def _topic_transcript(path, topic_words, *, session, turns=10):
    from conftest import assistant_entry, user_text_entry, write_jsonl

    entries = []
    for i in range(turns):
        entries.append(user_text_entry(f"{topic_words} question number {i}", uuid=f"{session}-u{i}", ts=f"2026-03-01T10:{i:02d}:00Z"))
        entries.append(assistant_entry(text=f"{topic_words} answer {i} with detail {topic_words}", uuid=f"{session}-a{i}", ts=f"2026-03-01T10:{i:02d}:30Z"))
    write_jsonl(path, entries)


@pytest.fixture
def built(tmp_path, capture):
    from synapt.recall.cli import _archive_and_build

    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _topic_transcript(source / "s1.jsonl", "alpha deploy pipeline kubernetes rollout canary staging release rollback manifest helm chart", session="s1")
    _topic_transcript(source / "s2.jsonl", "beta database migration postgres index vacuum replica failover backup restore schema table", session="s2")
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)
    return capture


BUILD_LOOPS = (
    "clusters: search text",
    "clusters: save rows",
    "clusters: save memberships",
    "clusters: save signatures",
    "clusters: summaries",
    "clusters: tags",
    "timeline: sessions",
    "timeline: arcs",
    "timeline: save arcs",
)


@pytest.mark.parametrize("label", BUILD_LOOPS)
def test_each_loop_after_the_clustering_line_reports_from_inside_itself(built, label):
    _assert_reports_from_inside(built, label)


@pytest.mark.parametrize(
    "label", ("clusters: FTS rebuild", "clusters: drop orphan summaries", "timeline: session info", "timeline: FTS rebuild")
)
def test_each_opaque_statement_names_its_start_and_its_end(built, label):
    starts = [i for i, l in enumerate(built) if l == f"{label}: starting"]
    ends = [i for i, l in enumerate(built) if l.startswith(f"{label}: done in ")]
    assert len(starts) == 1 and len(ends) == 1 and starts[0] < ends[0], f"{label!r}: starts {starts} ends {ends}"


def _cluster_rows(project):
    import sqlite3

    from synapt.recall.core import project_index_dir

    conn = sqlite3.connect(project_index_dir(project) / "recall.db")
    try:
        clusters = conn.execute("SELECT cluster_id, topic, chunk_count, tags FROM clusters ORDER BY cluster_id").fetchall()
        members = conn.execute("SELECT cluster_id, chunk_id FROM cluster_chunks ORDER BY cluster_id, chunk_id").fetchall()
        summaries = conn.execute("SELECT cluster_id, summary FROM cluster_summaries ORDER BY cluster_id").fetchall()
    finally:
        conn.close()
    return clusters, members, summaries


def test_progress_changes_nothing_a_whole_build_stores(tmp_path, monkeypatch):
    """The same transcripts built with progress silent and with every tick reporting store the same clusters,
    memberships and summaries: the change is observation only."""
    from synapt.recall.cli import _archive_and_build

    stored = []
    for name, interval in (("quiet", 1e9), ("loud", 0.0)):
        monkeypatch.setattr(progress, "PROGRESS_INTERVAL_S", interval)
        monkeypatch.setattr(progress, "_emit", lambda text: None)
        project = tmp_path / name
        project.mkdir()
        source = tmp_path / f"{name}-source"
        source.mkdir()
        _topic_transcript(source / "s1.jsonl", "alpha deploy pipeline kubernetes rollout canary staging release rollback manifest helm chart", session="s1")
        _topic_transcript(source / "s2.jsonl", "beta database migration postgres index vacuum replica failover backup restore schema table", session="s2")
        _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)
        stored.append(_cluster_rows(project))
    clusters, members, summaries = stored[0]
    assert clusters and members and summaries, "the build must have stored clusters or this compares nothing"
    assert stored[0] == stored[1], "progress reporting changed what the build stored"


def test_save_clusters_says_how_big_a_replacement_it_is_about_to_make(built):
    sized = [l for l in built if l.startswith("clusters: replacing the stored clusters with ")]
    assert len(sized) == 1 and "2 clusters and 20 memberships" in sized[0], sized


def test_a_run_of_skipped_singleton_clusters_still_moves_the_topic_count(capture):
    """The topic loop skips clusters below MIN_CLUSTER_SIZE. The tick comes before the skip, so a corpus that is mostly
    singletons still reports every position: the count would otherwise sit still through the longest stretch."""
    from synapt.recall.clustering import cluster_chunks
    from synapt.recall.core import TranscriptChunk

    chunks = _corpus(n_topics=3, per_topic=10)
    for i in range(40):  # 40 chunks that share no vocabulary with anything: 40 singleton clusters
        chunks.append(
            TranscriptChunk(
                id=f"solo:{i}", session_id=f"solo-{i}", timestamp=f"2026-04-{(i % 27) + 1:02d}T10:00:00Z", turn_index=0,
                user_text=" ".join(f"uniq{i}x{j}" for j in range(12)),
                assistant_text=" ".join(f"only{i}y{j}" for j in range(12)),
            )
        )
    cluster_chunks(chunks)
    topics, _closing = _loop_lines(capture, "clustering: topics")
    counts = [n for n, _total in topics]
    total = topics[-1][1]
    assert total >= 40, f"expected the singletons among the clusters, got total {total}"
    assert counts == list(range(1, total + 1)), f"the topic count skipped positions: {counts}"


def test_the_default_channel_is_a_logger_record_at_info(caplog):
    """The lines reach a watcher through logging, the same channel as the 'FTS5 index: n/N chunks' lines."""
    with caplog.at_level(logging.INFO):
        log = ProgressLog("probe", 2, "things", interval=0.0)
        log.tick(1)
        log.done()
    records = [r for r in caplog.records if r.name.startswith("synapt.recall") and r.getMessage().startswith("probe: ")]
    assert [r.levelno for r in records] == [logging.INFO, logging.INFO], [(r.name, r.getMessage()) for r in caplog.records]


def test_the_summary_and_tag_loops_count_the_clusters_they_skip(tmp_path, monkeypatch):
    """A second FULL build over the same transcripts holds an LLM summary (the summary loop skips that cluster) and the
    first build's timeline arcs (the tag loop skips every non-topic row). The tick comes before each skip, so both
    counts advance by one through every position."""
    from synapt.recall.cli import _archive_and_build
    from synapt.recall.core import project_index_dir
    from synapt.recall.storage import RecallDB

    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _topic_transcript(source / "s1.jsonl", "alpha deploy pipeline kubernetes rollout canary staging release rollback manifest helm chart", session="s1")
    _topic_transcript(source / "s2.jsonl", "beta database migration postgres index vacuum replica failover backup restore schema table", session="s2")
    monkeypatch.setattr(progress, "PROGRESS_INTERVAL_S", 1e9)
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)

    db = RecallDB(project_index_dir(project) / "recall.db")
    first_topic = db._conn.execute("SELECT cluster_id FROM clusters WHERE cluster_type = 'topic' ORDER BY cluster_id").fetchone()[0]
    db.save_cluster_summary(first_topic, "an upgraded summary", method="llm")
    db._conn.commit()
    del db

    lines: list[str] = []
    monkeypatch.setattr(progress, "PROGRESS_INTERVAL_S", 0.0)
    monkeypatch.setattr(progress, "_emit", lines.append)
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=False)

    for label in ("clusters: summaries", "clusters: tags"):
        counted, closing = _loop_lines(lines, label)
        totals = {total for _n, total in counted}
        assert counted and len(totals) == 1, f"{label!r}: {counted}"
        assert [n for n, _t in counted] == list(range(1, totals.pop() + 1)), f"{label!r} skipped a position: {counted}"
        assert len(closing) == 1


# ---------------------------------------------------------------------------
# D. Cross-session links: the n x n step an embedding build runs after the timeline
# ---------------------------------------------------------------------------

def _links_index(per_session=8, sessions=4, dim=6, seed=3):
    """The REAL ``TranscriptIndex.build_cross_session_links`` over synthetic embeddings, through a stubbed store."""
    np = pytest.importorskip("numpy")
    from types import SimpleNamespace

    from synapt.recall.core import TranscriptIndex

    rng = np.random.default_rng(seed)
    rowids = list(range(1, per_session * sessions + 1))
    saved: list[list] = []
    index = object.__new__(TranscriptIndex)
    index._db = SimpleNamespace(
        chunk_session_map=lambda: {r: f"s{(r - 1) // per_session}" for r in rowids},
        chunk_id_map=lambda: {r: f"c{r}" for r in rowids},
        save_chunk_links=lambda links: saved.append(list(links)),
    )
    index._all_embeddings = {r: rng.normal(size=dim).astype("float32") for r in rowids}
    index._emb_matrix = None
    index._emb_rowids = []
    index._ensure_embeddings_loaded = lambda: None
    index.CROSS_LINK_MIN_SIM = -1.0  # keep every top-k candidate, so the stored links are never empty
    return index, saved


@pytest.mark.parametrize("label", ("cross-session links: same-session mask", "cross-session links: top-k"))
def test_the_cross_session_loops_report_from_inside_themselves(capture, label):
    index, saved = _links_index()
    assert index.build_cross_session_links() > 0 and saved
    _assert_reports_from_inside(capture, label)
    assert _loop_lines(capture, label)[1] == [(32, 32)]


@pytest.mark.parametrize(
    "label", ("cross-session links: load embeddings", "cross-session links: similarity matrix", "cross-session links: save links")
)
def test_each_cross_session_statement_names_its_start_and_its_end(capture, label):
    index, _saved = _links_index()
    index.build_cross_session_links()
    starts = [i for i, l in enumerate(capture) if l == f"{label}: starting"]
    ends = [i for i, l in enumerate(capture) if l.startswith(f"{label}: done in ")]
    assert len(starts) == 1 and len(ends) == 1 and starts[0] < ends[0], f"{label!r}: starts {starts} ends {ends}"


def test_the_cross_session_step_says_how_big_its_matrix_is_before_it_is_built(capture):
    index, _saved = _links_index()
    index.build_cross_session_links()
    sized = [i for i, l in enumerate(capture) if l.startswith("cross-session links: 32 chunks with embeddings; the similarity matrix is ")]
    matrix_start = capture.index("cross-session links: similarity matrix: starting")
    assert len(sized) == 1 and sized[0] < matrix_start, (sized, matrix_start)
    assert capture[sized[0]].endswith(" 0.000 GB"), capture[sized[0]]


def test_the_matrix_size_in_that_line_is_n_squared_float32s(capture):
    """2,000 chunks: 2,000 x 2,000 x 4 bytes is 0.016 GB. The real step runs; only the number in the line is checked."""
    index, _saved = _links_index(per_session=250, sessions=8)
    index.build_cross_session_links()
    sized = [l for l in capture if l.startswith("cross-session links: 2000 chunks with embeddings; the similarity matrix is ")]
    assert sized == ["cross-session links: 2000 chunks with embeddings; the similarity matrix is 0.016 GB"], sized


def test_progress_changes_nothing_the_cross_session_links_store(monkeypatch):
    stored = []
    for interval in (1e9, 0.0):
        monkeypatch.setattr(progress, "PROGRESS_INTERVAL_S", interval)
        monkeypatch.setattr(progress, "_emit", lambda text: None)
        index, saved = _links_index()
        index.build_cross_session_links()
        stored.append(saved)
    assert stored[0] and stored[0] == stored[1], "progress reporting changed the links the step stored"


def test_the_disable_flag_returns_before_any_line_is_sent(capture, monkeypatch):
    monkeypatch.setenv("SYNAPT_DISABLE_CROSS_LINKS", "1")
    index, saved = _links_index()
    assert index.build_cross_session_links() == 0 and not saved
    assert capture == [], capture


# ---------------------------------------------------------------------------
# E. The other steps after the timeline are bracketed, so none of them is a silent stretch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label",
    ("build: promotions", "build: decay scores", "build: archive cold clusters", "build: compact access log", "knowledge: dedup"),
)
def test_each_step_after_the_timeline_names_its_start_and_its_end(built, label):
    starts = [i for i, l in enumerate(built) if l == f"{label}: starting"]
    ends = [i for i, l in enumerate(built) if l.startswith(f"{label}: done in ")]
    assert len(starts) == 1 and len(ends) == 1 and starts[0] < ends[0], f"{label!r}: starts {starts} ends {ends}"


# ---------------------------------------------------------------------------
# F. save_clusters re-homes rows that an incremental merge placed; those two loops tick as well
# ---------------------------------------------------------------------------

_MERGE_WORDS = [
    "kubernetes", "deployment", "container", "cluster", "pod", "service",
    "ingress", "namespace", "volume", "secret", "configmap", "replica",
]


def _merge_topic(path, turns=8):
    from conftest import assistant_entry, user_text_entry, write_jsonl

    entries = []
    for i in range(turns):
        words = " ".join(w for j, w in enumerate(_MERGE_WORDS) if (j + i) % 3 != 0)
        entries.append(user_text_entry(f"question about {words}", uuid=f"topic-u{i}", ts=f"2026-03-01T10:{i:02d}:00Z"))
        entries.append(assistant_entry(text=f"answer about {words}", uuid=f"topic-a{i}", ts=f"2026-03-01T10:{i:02d}:30Z"))
    write_jsonl(path, entries)


def _merge_similar(path, tag):
    from conftest import assistant_entry, user_text_entry, write_jsonl

    words = " ".join(_MERGE_WORDS)
    write_jsonl(path, [
        user_text_entry(f"question about {words}", uuid=f"{tag}-u", ts="2026-03-01T11:00:00Z"),
        assistant_entry(text=f"answer about {words}", uuid=f"{tag}-a", ts="2026-03-01T11:00:30Z"),
    ])


def _lonely(path, tag, words):
    from conftest import assistant_entry, user_text_entry, write_jsonl

    # no word is shared between two lonely chunks, so the fresh grouping leaves each one out
    write_jsonl(path, [
        user_text_entry(" ".join(words[:5]), uuid=f"{tag}-u", ts="2026-03-01T11:00:00Z"),
        assistant_entry(text=" ".join(words[5:]), uuid=f"{tag}-a", ts="2026-03-01T11:00:30Z"),
    ])


def test_the_rehome_and_recount_loops_report_when_an_incremental_merge_left_rows_behind(tmp_path, monkeypatch):
    """Two chunks that no fresh grouping places, merged into an existing cluster with a run id, leave two preserved rows.
    The next ordinary build's save_clusters walks them (one loop over the rows) and recounts the cluster they landed in
    (one loop over clusters). Both loops report from inside themselves."""
    from conftest import assistant_entry, user_text_entry, write_jsonl
    from synapt.recall.cli import _archive_and_build
    from synapt.recall.clustering import stale_transcript_chunk_ids
    from synapt.recall.core import project_index_dir
    from synapt.recall.storage import RecallDB

    monkeypatch.setattr(progress, "PROGRESS_INTERVAL_S", 1e9)
    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _merge_topic(source / "topic.jsonl")
    _lonely(source / "lonely-a.jsonl", "lone-a", ("zebra", "quantum", "bagpipe", "okapi", "violin", "marmalade", "trombone", "glacier", "pumice", "saffron"))
    _lonely(source / "lonely-b.jsonl", "lone-b", ("walrus", "tundra", "sextant", "heron", "cobalt", "lantern", "anvil", "juniper", "obsidian", "quasar"))
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)

    db = RecallDB(project_index_dir(project) / "recall.db")
    try:
        clusters = db._conn.execute("SELECT cluster_id FROM clusters WHERE cluster_type = 'topic'").fetchall()
        assert len(clusters) == 1, clusters
        stale = stale_transcript_chunk_ids(db)
        assert len(stale) == 2, f"the two lonely chunks must be stale (the fresh pass leaves them out): {stale}"
        db.merge_chunks_into_cluster(clusters[0][0], stale, "appended", "2026-03-01T12:00:00Z", run_id="merge-run")
        db._conn.commit()
        assert db._conn.execute("SELECT COUNT(*) FROM cluster_chunks WHERE run_id = 'merge-run'").fetchone()[0] == 2
    finally:
        db.close()

    write_jsonl(source / "unrelated.jsonl", [
        user_text_entry("completely unrelated filler turn", uuid="fill-u", ts="2026-03-02T12:00:00Z"),
        assistant_entry(text="completely unrelated filler answer", uuid="fill-a", ts="2026-03-02T12:00:30Z"),
    ])
    lines: list[str] = []
    monkeypatch.setattr(progress, "PROGRESS_INTERVAL_S", 0.0)
    monkeypatch.setattr(progress, "_emit", lines.append)
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)

    _assert_reports_from_inside(lines, "clusters: re-home preserved rows")
    recount, closing = _loop_lines(lines, "clusters: recount re-homed clusters")
    assert recount and closing == [(1, 1)], f"the recount loop must have run over the one cluster: {recount} {closing}"


def test_the_knowledge_compact_step_is_bracketed_when_a_knowledge_file_exists(tmp_path, capture):
    """`knowledge: compact` only runs when the project has a knowledge file, which the plain build fixture does not."""
    from synapt.recall.cli import _archive_and_build
    from synapt.recall.knowledge import VALID_CATEGORIES, KnowledgeNode, _knowledge_path, append_node

    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _topic_transcript(source / "s1.jsonl", "alpha deploy pipeline kubernetes rollout canary staging release rollback manifest helm chart", session="s1")
    kn = _knowledge_path(project)
    kn.parent.mkdir(parents=True, exist_ok=True)
    for revision in (1, 2):  # two records of one id: compaction has something to remove
        append_node(
            KnowledgeNode(id="k1", content="a durable fact", category=sorted(VALID_CATEGORIES)[0], confidence=0.9,
                          created_at="2026-03-01T10:00:00Z", updated_at=f"2026-03-01T10:0{revision}:00Z", revision=revision),
            path=kn,
        )
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)
    starts = [i for i, l in enumerate(capture) if l == "knowledge: compact: starting"]
    ends = [i for i, l in enumerate(capture) if l.startswith("knowledge: compact: done in ")]
    assert len(starts) == 1 and len(ends) == 1 and starts[0] < ends[0], (starts, ends)
