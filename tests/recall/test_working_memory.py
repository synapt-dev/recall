"""Tests for the session-scoped working memory."""

import time

from synapt.recall.working_memory import WorkingMemory, WorkingMemorySlot, MAX_SLOTS


class TestWorkingMemory:
    """Unit tests for WorkingMemory."""

    def test_record_and_query(self):
        """Record an item, query with overlapping tokens."""
        wm = WorkingMemory()
        wm.record("chunk", "s1:t0", "Fixed flock TOCTOU race condition in locking")

        results = wm.query({"flock", "race", "condition"})
        assert len(results) == 1
        assert results[0].item_id == "s1:t0"
        assert results[0].access_count == 1

    def test_query_requires_two_token_overlap(self):
        """Query needs 2+ shared tokens to match."""
        wm = WorkingMemory()
        wm.record("chunk", "s1:t0", "Fixed flock TOCTOU race condition")

        # Only 1 token overlap → no match
        results = wm.query({"flock", "unrelated"})
        assert len(results) == 0

        # 2 tokens → match
        results = wm.query({"flock", "race"})
        assert len(results) == 1

    def test_repeated_access_increments_count(self):
        """Recording the same item again increments access_count."""
        wm = WorkingMemory()
        wm.record("chunk", "s1:t0", "flock race fix")
        wm.record("chunk", "s1:t0", "flock race fix")
        wm.record("chunk", "s1:t0", "flock race fix")

        assert "s1:t0" in wm
        results = wm.query({"flock", "race"})
        assert results[0].access_count == 3

    def test_lru_eviction(self):
        """When at capacity, evicts least-recently-accessed slot."""
        wm = WorkingMemory()

        # Fill to capacity
        for i in range(MAX_SLOTS):
            wm.record("chunk", f"s1:t{i}", f"topic{i} content{i} extra{i}")

        assert len(wm) == MAX_SLOTS

        # Access the first item to make it recent
        wm.record("chunk", "s1:t0", "topic0 content0 extra0")

        # Add one more — should evict s1:t1 (LRU, not s1:t0 which was just accessed)
        wm.record("chunk", "s1:t99", "new topic99 content99 extra99")

        assert len(wm) == MAX_SLOTS
        assert "s1:t0" in wm   # kept (recently accessed)
        assert "s1:t99" in wm  # just added
        assert "s1:t1" not in wm  # evicted (LRU)

    def test_boost_score_not_in_memory(self):
        """Items not in working memory get no boost."""
        wm = WorkingMemory()
        assert wm.boost_score(5.0, "unknown") == 5.0

    def test_boost_score_in_memory(self):
        """Items in working memory get 1.5x boost."""
        wm = WorkingMemory()
        wm.record("chunk", "s1:t0", "something")

        assert wm.boost_score(4.0, "s1:t0") == 6.0  # 4.0 * 1.5

    def test_boost_score_frequent(self):
        """Items accessed 3+ times get 2.0x boost."""
        wm = WorkingMemory()
        wm.record("chunk", "s1:t0", "something")
        wm.record("chunk", "s1:t0", "something")
        wm.record("chunk", "s1:t0", "something")

        assert wm.boost_score(4.0, "s1:t0") == 8.0  # 4.0 * 2.0

    def test_empty_query(self):
        """Empty query tokens return nothing."""
        wm = WorkingMemory()
        wm.record("chunk", "s1:t0", "some content here")
        assert wm.query(set()) == []

    def test_query_ranking(self):
        """Results ranked by overlap * log(access_count)."""
        wm = WorkingMemory()
        wm.record("chunk", "s1:t0", "swift adapter training eval")
        wm.record("cluster", "clust-abc", "swift adapter patcher critic training")
        # Access cluster more to boost its score
        wm.record("cluster", "clust-abc", "swift adapter patcher critic training")

        results = wm.query({"swift", "adapter", "training"}, max_results=2)
        assert len(results) == 2
        # Cluster should rank higher (same overlap but higher access_count)
        assert results[0].item_id == "clust-abc"

    def test_len_and_contains(self):
        """__len__ and __contains__ work correctly."""
        wm = WorkingMemory()
        assert len(wm) == 0
        assert "s1:t0" not in wm

        wm.record("chunk", "s1:t0", "content")
        assert len(wm) == 1
        assert "s1:t0" in wm
