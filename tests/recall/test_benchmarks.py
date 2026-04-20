"""Benchmarks for recall system performance (#305).

Priority metrics:
1. recall_search latency (most frequent operation)
2. recall_build incremental time (runs every session start)
3. Peak memory during build (constrains M2 Air 16GB)

Run benchmarks (disabled by default via pyproject.toml):
    pytest tests/recall/test_benchmarks.py -v --benchmark-enable --benchmark-sort=mean
    pytest tests/recall/test_benchmarks.py -v --benchmark-only
"""

from __future__ import annotations

import random
import statistics
import time
import tracemalloc
from pathlib import Path

import pytest

from synapt.recall.clustering import (
    _chunk_tokens,
    cluster_chunks,
)
from synapt.recall.core import TranscriptChunk, TranscriptIndex
from synapt.recall.storage import RecallDB


# ---------------------------------------------------------------------------
# Synthetic corpus generator
# ---------------------------------------------------------------------------

# Vocabulary pools — realistic code/engineering tokens grouped by topic
_TOPICS = {
    "flock": ["flock", "advisory", "locking", "toctou", "compact", "journal",
              "truncate", "atomic", "mutex", "deadlock", "contention", "writeback"],
    "clustering": ["jaccard", "cluster", "centroid", "threshold", "similarity",
                   "overlap", "token", "union", "greedy", "merging", "dendrogram"],
    "swift": ["swift", "adapter", "lora", "patcher", "critic", "xctassert",
              "harness", "evaluate", "inference", "quantize", "finetune"],
    "recall": ["recall", "transcript", "session", "chunk", "bm25", "fts5",
               "sqlite", "embedding", "search", "ranking", "relevance"],
    "training": ["training", "epoch", "gradient", "backprop", "checkpoint",
                 "validation", "overfitting", "convergence", "warmup", "scheduler"],
    "git": ["commit", "branch", "rebase", "cherry", "merge", "conflict",
            "stash", "bisect", "worktree", "reflog", "upstream", "origin"],
    "database": ["postgres", "migration", "schema", "constraint", "foreign",
                 "trigger", "transaction", "rollback", "vacuum", "replication"],
    "network": ["socket", "handshake", "tls", "certificate", "proxy",
                "latency", "throughput", "bandwidth", "congestion", "retry"],
}

_NOISE_WORDS = [
    "implement", "review", "debug", "refactor", "optimize", "analyze",
    "configure", "deploy", "monitor", "troubleshoot", "investigate",
    "document", "benchmark", "profile", "measure", "validate",
]


def _generate_chunks(
    n: int,
    n_sessions: int = 0,
    seed: int = 42,
) -> list[TranscriptChunk]:
    """Generate n synthetic chunks with realistic token distributions.

    Each chunk belongs to 1-2 topics. Chunks within a session share a
    primary topic (simulating real conversation coherence).
    """
    rng = random.Random(seed)
    topics = list(_TOPICS.keys())
    if n_sessions <= 0:
        n_sessions = max(1, n // 8)

    # Assign each session a primary topic
    session_topics = {
        f"sess-{i:04d}": rng.choice(topics) for i in range(n_sessions)
    }

    chunks: list[TranscriptChunk] = []
    for i in range(n):
        session_id = f"sess-{i % n_sessions:04d}"
        turn_index = i // n_sessions
        primary_topic = session_topics[session_id]

        # 70% primary topic tokens, 30% secondary/noise
        primary_words = rng.sample(
            _TOPICS[primary_topic], k=min(5, len(_TOPICS[primary_topic]))
        )
        secondary_topic = rng.choice(topics)
        secondary_words = rng.sample(
            _TOPICS[secondary_topic], k=min(2, len(_TOPICS[secondary_topic]))
        )
        noise = rng.sample(_NOISE_WORDS, k=2)

        user_words = primary_words[:3] + noise[:1]
        assistant_words = primary_words[2:] + secondary_words + noise[1:]

        ts_hour = 10 + (i % 8)
        ts_min = (i * 7) % 60
        day = 1 + (i % 28)
        timestamp = f"2026-03-{day:02d}T{ts_hour:02d}:{ts_min:02d}:00Z"

        chunks.append(TranscriptChunk(
            id=f"{session_id}:t{turn_index}",
            session_id=session_id,
            timestamp=timestamp,
            turn_index=turn_index,
            user_text=" ".join(user_words),
            assistant_text=" ".join(assistant_words),
            tools_used=rng.sample(["Read", "Edit", "Bash", "Grep"], k=rng.randint(0, 2)),
            files_touched=[f"src/module_{i % 20}.py"] if rng.random() > 0.3 else [],
        ))

    return chunks


def _build_populated_db(tmp_path: Path, chunks: list[TranscriptChunk]) -> RecallDB:
    """Create a RecallDB populated with chunks and FTS5 index."""
    db = RecallDB(tmp_path / "recall.db")
    db.save_chunks(chunks)
    return db


def _build_index(
    chunks: list[TranscriptChunk],
    db: RecallDB | None = None,
) -> TranscriptIndex:
    """Build a TranscriptIndex over chunks, optionally with FTS5 backend."""
    return TranscriptIndex(chunks, db=db)


def _median_lookup_latency(
    index: TranscriptIndex,
    query: str,
    *,
    max_chunks: int = 5,
    rounds: int = 7,
    min_total_seconds: float = 0.05,
    min_iters: int = 20,
) -> float:
    """Measure stable per-lookup latency for very fast searches.

    Sub-millisecond FTS lookups are common on fast runners. When the baseline
    is ~0.1ms, a simple ratio against a 4x larger corpus becomes dominated by
    timer noise and fixed overhead. Measure enough iterations to exceed a small
    minimum wall time, then take the median across rounds.
    """
    samples: list[float] = []

    # Warm the SQLite page cache and Python call path before timing.
    index.lookup(query, max_chunks=max_chunks)

    for _ in range(rounds):
        n_iters = min_iters
        while True:
            start = time.perf_counter()
            for _ in range(n_iters):
                index.lookup(query, max_chunks=max_chunks)
            elapsed = time.perf_counter() - start
            if elapsed >= min_total_seconds or n_iters >= 10000:
                samples.append(elapsed / n_iters)
                break
            n_iters *= 2

    return statistics.median(samples)


# ---------------------------------------------------------------------------
# Fixtures — generate corpora once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def corpus_500():
    return _generate_chunks(500, n_sessions=20)


@pytest.fixture(scope="session")
def corpus_2000():
    return _generate_chunks(2000, n_sessions=80)


@pytest.fixture(scope="session")
def corpus_5000():
    return _generate_chunks(5000, n_sessions=200)


@pytest.fixture
def db_500(tmp_path, corpus_500):
    return _build_populated_db(tmp_path, corpus_500)


@pytest.fixture
def db_2000(tmp_path, corpus_2000):
    return _build_populated_db(tmp_path, corpus_2000)


@pytest.fixture
def db_5000(tmp_path, corpus_5000):
    return _build_populated_db(tmp_path, corpus_5000)


@pytest.fixture
def index_500(corpus_500, db_500):
    return _build_index(corpus_500, db_500)


@pytest.fixture
def index_2000(corpus_2000, db_2000):
    return _build_index(corpus_2000, db_2000)


# Sample queries (one per topic for realistic mix)
_QUERIES = [
    "flock advisory locking race condition",
    "jaccard clustering threshold",
    "swift adapter training evaluation",
    "recall search transcript session",
    "training epoch gradient convergence",
    "git rebase merge conflict",
    "postgres migration schema constraint",
    "socket tls handshake latency",
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. SEARCH LATENCY
# ═══════════════════════════════════════════════════════════════════════════

class TestSearchLatency:
    """Benchmark recall_search — the most frequent operation."""

    def test_fts_search_500(self, benchmark, index_500):
        """FTS5 search over 500 chunks."""
        benchmark(index_500.lookup, "flock advisory locking", max_chunks=5)

    def test_fts_search_2000(self, benchmark, index_2000):
        """FTS5 search over 2000 chunks."""
        benchmark(index_2000.lookup, "flock advisory locking", max_chunks=5)

    def test_fts_search_varied_queries(self, benchmark, index_2000):
        """FTS5 search with rotating queries (cache-hostile)."""
        queries = _QUERIES[:]
        call_count = [0]

        def search_varied():
            q = queries[call_count[0] % len(queries)]
            call_count[0] += 1
            return index_2000.lookup(q, max_chunks=5)

        benchmark(search_varied)

    def test_fts_search_max_chunks_10(self, benchmark, index_2000):
        """FTS5 search returning 10 chunks (larger result set)."""
        benchmark(index_2000.lookup, "training epoch gradient", max_chunks=10)

    def test_fts_search_concise(self, benchmark, index_2000):
        """Concise mode — cluster summaries only."""
        benchmark(
            index_2000.lookup,
            "flock advisory locking",
            max_chunks=5,
            depth="concise",
        )

    def test_bm25_search_500(self, benchmark, corpus_500):
        """In-memory BM25 search (no SQLite, legacy path)."""
        index = TranscriptIndex(corpus_500)
        benchmark(index.lookup, "flock advisory locking", max_chunks=5)

    def test_bm25_search_2000(self, benchmark, corpus_2000):
        """In-memory BM25 search over 2000 chunks."""
        index = TranscriptIndex(corpus_2000)
        benchmark(index.lookup, "flock advisory locking", max_chunks=5)


# ═══════════════════════════════════════════════════════════════════════════
# 2. BUILD / CLUSTERING TIME
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildTime:
    """Benchmark recall_build components."""

    def test_clustering_500(self, benchmark, corpus_500):
        """Jaccard clustering over 500 chunks."""
        benchmark(cluster_chunks, corpus_500)

    def test_clustering_2000(self, benchmark, corpus_2000):
        """Jaccard clustering over 2000 chunks."""
        benchmark(cluster_chunks, corpus_2000)

    def test_clustering_5000(self, benchmark, corpus_5000):
        """Jaccard clustering over 5000 chunks."""
        benchmark(cluster_chunks, corpus_5000)

    def test_index_construction_500(self, benchmark, corpus_500, db_500):
        """TranscriptIndex construction (BM25 build + rowid mapping)."""
        benchmark(_build_index, corpus_500, db_500)

    def test_index_construction_2000(self, benchmark, corpus_2000, db_2000):
        """TranscriptIndex construction over 2000 chunks."""
        benchmark(_build_index, corpus_2000, db_2000)

    def test_db_save_chunks_500(self, benchmark, tmp_path, corpus_500):
        """SQLite save_chunks + FTS5 rebuild for 500 chunks."""
        db = RecallDB(tmp_path / "bench_save.db")

        def save():
            db.save_chunks(corpus_500)

        benchmark(save)

    def test_db_save_chunks_2000(self, benchmark, tmp_path, corpus_2000):
        """SQLite save_chunks + FTS5 rebuild for 2000 chunks."""
        db = RecallDB(tmp_path / "bench_save_2k.db")

        def save():
            db.save_chunks(corpus_2000)

        benchmark(save)

    def test_tokenization_throughput(self, benchmark, corpus_2000):
        """_chunk_tokens extraction over all chunks."""

        def tokenize_all():
            for chunk in corpus_2000:
                _chunk_tokens(chunk)

        benchmark(tokenize_all)


# ═══════════════════════════════════════════════════════════════════════════
# 3. PEAK MEMORY
# ═══════════════════════════════════════════════════════════════════════════

class TestMemoryUsage:
    """Measure peak memory for key operations via tracemalloc.

    These are not pytest-benchmark tests — they use plain assertions
    with tracemalloc snapshots to report and cap peak memory.
    """

    @staticmethod
    def _measure_peak(func, *args, **kwargs):
        """Run func and return (result, peak_mb) via tracemalloc."""
        tracemalloc.start()
        tracemalloc.reset_peak()
        result = func(*args, **kwargs)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result, peak / (1024 * 1024)

    def test_clustering_peak_memory_2000(self, corpus_2000, capsys):
        """Peak memory during clustering of 2000 chunks."""
        _, peak_mb = self._measure_peak(cluster_chunks, corpus_2000)
        print(f"\n  clustering(2000): {peak_mb:.1f}MB peak")
        assert peak_mb < 50, f"Clustering peak memory {peak_mb:.1f}MB exceeds 50MB"

    def test_clustering_peak_memory_5000(self, corpus_5000, capsys):
        """Peak memory during clustering of 5000 chunks."""
        _, peak_mb = self._measure_peak(cluster_chunks, corpus_5000)
        print(f"\n  clustering(5000): {peak_mb:.1f}MB peak")
        assert peak_mb < 100, f"Clustering peak memory {peak_mb:.1f}MB exceeds 100MB"

    def test_index_construction_memory_2000(self, corpus_2000, tmp_path, capsys):
        """Peak memory during TranscriptIndex construction."""
        db = _build_populated_db(tmp_path, corpus_2000)
        _, peak_mb = self._measure_peak(TranscriptIndex, corpus_2000, db=db)
        print(f"\n  index_construction(2000): {peak_mb:.1f}MB peak")
        assert peak_mb < 50, f"Index construction peak {peak_mb:.1f}MB exceeds 50MB"

    def test_search_memory_2000(self, corpus_2000, tmp_path, capsys):
        """Peak memory during a single search query."""
        db = _build_populated_db(tmp_path, corpus_2000)
        index = TranscriptIndex(corpus_2000, db=db)

        def run_queries():
            for q in _QUERIES:
                index.lookup(q, max_chunks=5)

        _, peak_mb = self._measure_peak(run_queries)
        print(f"\n  search(8 queries, 2000 chunks): {peak_mb:.1f}MB peak")
        assert peak_mb < 20, f"Search peak memory {peak_mb:.1f}MB exceeds 20MB"


# ═══════════════════════════════════════════════════════════════════════════
# 4. SCALING CHARACTERISTICS
# ═══════════════════════════════════════════════════════════════════════════

class TestScaling:
    """Verify subquadratic scaling by comparing times at different sizes.

    These tests run operations at two sizes and check that the larger
    takes less than 10x the smaller (rough subquadratic bound). This
    catches accidental O(n²) regressions.
    """

    def test_clustering_scaling(self, corpus_500, corpus_2000):
        """Measure clustering scaling factor (currently O(n×k), ~quadratic).

        The greedy algorithm compares each chunk against all existing clusters,
        giving O(n×k) where k grows with n. For 500→2000 (4x data), expect
        ~12-16x time. This test documents the current scaling and will catch
        regressions (>20x) or improvements.
        """
        import time

        start = time.perf_counter()
        cluster_chunks(corpus_500)
        t_500 = time.perf_counter() - start

        start = time.perf_counter()
        cluster_chunks(corpus_2000)
        t_2000 = time.perf_counter() - start

        ratio = t_2000 / max(t_500, 1e-6)
        # Current: ~12x for 4x data (O(n²) territory).
        # Alert if it gets worse than 20x (regression) or document
        # improvement if it drops below 8x (optimization landed).
        assert ratio < 20, (
            f"Clustering scaling regressed: 500→2000 = {ratio:.1f}x "
            f"({t_500:.3f}s → {t_2000:.3f}s)"
        )

    def test_fts_search_scales_with_index(self, corpus_500, corpus_2000, tmp_path):
        """FTS5 search latency should not increase linearly with corpus size."""
        db_small = _build_populated_db(tmp_path / "small", corpus_500)
        idx_small = TranscriptIndex(corpus_500, db=db_small)

        db_large = _build_populated_db(tmp_path / "large", corpus_2000)
        idx_large = TranscriptIndex(corpus_2000, db=db_large)

        q = "flock advisory locking"
        t_small = _median_lookup_latency(idx_small, q, max_chunks=5)
        t_large = _median_lookup_latency(idx_large, q, max_chunks=5)

        # When the baseline is sub-millisecond, ratio alone becomes unstable:
        # 0.1ms -> 1.5ms is still very fast, but reads as a 15x regression.
        if t_small < 0.0005:
            delta_ms = (t_large - t_small) * 1000
            assert t_large < 0.003 and delta_ms < 2.5, (
                "FTS absolute slowdown is too high for a 4x larger corpus: "
                f"{t_small*1000:.2f}ms → {t_large*1000:.2f}ms "
                f"(+{delta_ms:.2f}ms)"
            )
        else:
            ratio = t_large / t_small
            # 4x data should be <5x search time (FTS5 is sublinear)
            assert ratio < 5, (
                f"FTS search scales too steeply: 500→2000 = {ratio:.1f}x "
                f"({t_small*1000:.1f}ms → {t_large*1000:.1f}ms)"
            )
