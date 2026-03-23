"""Tests for hybrid search: RRF fusion, embedding search, query intent, temporal."""

from datetime import datetime, timezone

import pytest
from synapt.recall.hybrid import (
    rrf_merge,
    weighted_rrf_merge,
    embedding_search,
    classify_query_intent,
    intent_search_params,
    extract_temporal_range,
    extract_entities,
    RRF_K,
    SPARSE_RESULT_THRESHOLD,
)


# ---------------------------------------------------------------------------
# RRF merge
# ---------------------------------------------------------------------------

class TestRRFMerge:
    def test_single_list(self):
        """RRF with one list is just reciprocal rank scoring."""
        ranked = [(10, 5.0), (20, 3.0), (30, 1.0)]
        merged = rrf_merge(ranked)
        assert len(merged) == 3
        # First item should have highest RRF score
        assert merged[0][0] == 10
        assert merged[1][0] == 20
        assert merged[2][0] == 30

    def test_two_lists_agreement(self):
        """When both lists agree on top item, it should rank first."""
        list1 = [(10, 5.0), (20, 3.0), (30, 1.0)]
        list2 = [(10, 0.9), (30, 0.7), (20, 0.5)]
        merged = rrf_merge(list1, list2)
        # Item 10 is top in both lists
        assert merged[0][0] == 10

    def test_two_lists_different_items(self):
        """Items from only one list still appear in merged results."""
        list1 = [(10, 5.0), (20, 3.0)]
        list2 = [(30, 0.9), (40, 0.7)]
        merged = rrf_merge(list1, list2)
        ids = {item_id for item_id, _ in merged}
        assert ids == {10, 20, 30, 40}

    def test_empty_lists(self):
        """Empty input produces empty output."""
        assert rrf_merge() == []
        assert rrf_merge([]) == []
        assert rrf_merge([], []) == []

    def test_rrf_scores_are_positive(self):
        """All RRF scores should be positive."""
        ranked = [(1, 10.0), (2, 5.0), (3, 1.0)]
        merged = rrf_merge(ranked)
        for _, score in merged:
            assert score > 0

    def test_rrf_score_formula(self):
        """Verify exact RRF score for a single list."""
        ranked = [(10, 5.0)]
        merged = rrf_merge(ranked, k=60)
        # Score should be 1/(60+1) = 1/61
        assert abs(merged[0][1] - 1.0 / 61) < 1e-10

    def test_rrf_two_lists_exact(self):
        """Item appearing first in both lists gets sum of reciprocal ranks."""
        list1 = [(10, 5.0)]
        list2 = [(10, 0.9)]
        merged = rrf_merge(list1, list2, k=60)
        expected = 1.0 / 61 + 1.0 / 61  # Rank 0 in both
        assert abs(merged[0][1] - expected) < 1e-10


class TestWeightedRRFMerge:
    def test_balanced_weights(self):
        """With equal weights, same as regular RRF."""
        bm25 = [(10, 5.0), (20, 3.0)]
        emb = [(20, 0.9), (10, 0.7)]
        merged = weighted_rrf_merge(bm25, emb, bm25_weight=1.0, emb_weight=1.0)
        assert len(merged) == 2

    def test_bm25_weight_dominates(self):
        """Higher BM25 weight preserves BM25 ranking."""
        bm25 = [(10, 5.0), (20, 3.0)]
        emb = [(20, 0.9), (10, 0.7)]
        merged = weighted_rrf_merge(bm25, emb, bm25_weight=10.0, emb_weight=0.1)
        # BM25 has item 10 first with much higher weight
        assert merged[0][0] == 10

    def test_sparse_bm25_boosts_embeddings(self):
        """When BM25 returns few results, embedding weight auto-increases."""
        bm25 = [(10, 5.0)]  # Only 1 result < SPARSE_RESULT_THRESHOLD
        emb = [(20, 0.9), (30, 0.8), (10, 0.7)]
        merged = weighted_rrf_merge(bm25, emb, bm25_weight=1.0, emb_weight=1.0)
        # Item 20 (embedding-only) should appear thanks to auto-boost
        ids = [item_id for item_id, _ in merged]
        assert 20 in ids

    def test_empty_embedding_list(self):
        """Graceful handling when no embedding results."""
        bm25 = [(10, 5.0), (20, 3.0)]
        merged = weighted_rrf_merge(bm25, [], bm25_weight=1.0, emb_weight=1.0)
        assert len(merged) == 2
        assert merged[0][0] == 10

    def test_bm25_floor_zero_is_noop(self):
        """bm25_floor=0 (default) does not change output."""
        bm25 = [(10, 5.0), (20, 3.0)]
        emb = [(30, 0.9), (40, 0.8)]
        without = weighted_rrf_merge(bm25, emb)
        with_floor = weighted_rrf_merge(bm25, emb, bm25_floor=0)
        assert without == with_floor

    def test_bm25_floor_preserves_top_n(self):
        """bm25_floor=2 ensures top-2 BM25 results appear in output."""
        # Embedding dominates — BM25 items 10, 20 might not rank highly
        bm25 = [(10, 5.0), (20, 3.0), (30, 1.0)]
        emb = [(40, 0.99), (50, 0.98), (60, 0.97)]
        merged = weighted_rrf_merge(bm25, emb, emb_weight=100.0, bm25_floor=2)
        merged_ids = {item_id for item_id, _ in merged}
        # Items 10 and 20 must appear even though embedding weight is huge
        assert 10 in merged_ids
        assert 20 in merged_ids

    def test_bm25_floor_no_duplicates(self):
        """Floor items that already rank naturally are not duplicated."""
        bm25 = [(10, 5.0), (20, 3.0)]
        emb = [(10, 0.9), (20, 0.8)]  # Same items
        merged = weighted_rrf_merge(bm25, emb, bm25_floor=2)
        ids = [item_id for item_id, _ in merged]
        assert len(ids) == len(set(ids)), "No duplicate IDs"

    def test_bm25_floor_items_survive_positive_filter(self):
        """Floor-preserved items have score > 0 so they survive ``if s > 0``."""
        bm25 = [(10, 5.0), (20, 3.0)]
        emb = [(30, 0.99), (40, 0.98)]  # Disjoint — 10, 20 only from BM25
        merged = weighted_rrf_merge(bm25, emb, emb_weight=100.0, bm25_floor=2)
        # Simulate the downstream filter used in core.py retrieval path
        surviving = [(i, s) for i, s in merged if s > 0]
        surviving_ids = {i for i, _ in surviving}
        assert 10 in surviving_ids, "BM25 floor item 10 must survive s > 0 filter"
        assert 20 in surviving_ids, "BM25 floor item 20 must survive s > 0 filter"


# ---------------------------------------------------------------------------
# Embedding search
# ---------------------------------------------------------------------------

class TestEmbeddingSearch:
    def _make_embeddings(self):
        """Create simple test embeddings."""
        # Unit vectors along different axes
        emb_a = [1.0] + [0.0] * 383  # Points along dim 0
        emb_b = [0.0, 1.0] + [0.0] * 382  # Points along dim 1
        emb_c = [0.7, 0.7] + [0.0] * 382  # Between a and b
        return {1: emb_a, 2: emb_b, 3: emb_c}

    def test_exact_match(self):
        """Query vector identical to stored vector returns similarity ~1.0."""
        all_embs = self._make_embeddings()
        query = [1.0] + [0.0] * 383
        results = embedding_search(query, all_embs, limit=10, threshold=0.0)
        assert results[0][0] == 1  # Exact match
        assert results[0][1] > 0.99

    def test_threshold_filters(self):
        """Results below threshold are excluded."""
        all_embs = self._make_embeddings()
        query = [1.0] + [0.0] * 383
        # With high threshold, only exact/near matches survive
        results = embedding_search(query, all_embs, limit=10, threshold=0.9)
        assert all(sim >= 0.9 for _, sim in results)

    def test_limit_respected(self):
        """Limit parameter caps result count."""
        all_embs = {i: [float(i == j) for j in range(384)] for i in range(100)}
        query = [1.0 / 384**0.5] * 384  # Roughly equidistant from all
        results = embedding_search(query, all_embs, limit=5, threshold=0.0)
        assert len(results) <= 5

    def test_empty_embeddings(self):
        """Empty embedding dict returns empty results."""
        query = [1.0] + [0.0] * 383
        assert embedding_search(query, {}) == []

    def test_empty_query(self):
        """Empty query vector returns empty results."""
        all_embs = self._make_embeddings()
        assert embedding_search([], all_embs) == []

    def test_sorted_descending(self):
        """Results are sorted by similarity descending."""
        all_embs = self._make_embeddings()
        query = [0.8, 0.6] + [0.0] * 382  # Closer to a than b
        results = embedding_search(query, all_embs, limit=10, threshold=0.0)
        sims = [s for _, s in results]
        assert sims == sorted(sims, reverse=True)


# ---------------------------------------------------------------------------
# Query intent classification
# ---------------------------------------------------------------------------

class TestQueryIntentClassification:
    def test_factual(self):
        assert classify_query_intent("what is the database port") == "factual"
        assert classify_query_intent("which API endpoint handles auth") == "factual"
        assert classify_query_intent("what does the config setting do") == "factual"

    def test_factual_indirect_what(self):
        """'What [noun] did/is X' with named entities → aggregation (scattered facts)."""
        assert classify_query_intent("What state did Nate visit") == "aggregation"
        assert classify_query_intent("What book did Tim recommend") == "aggregation"
        assert classify_query_intent("What color is Caroline's car") == "factual"

    def test_factual_inference(self):
        """Inference questions about personality/traits need knowledge nodes."""
        # "Would X be considered" → aggregation (needs multiple facts to infer)
        assert classify_query_intent("Would Caroline be considered religious") == "aggregation"
        assert classify_query_intent("Would Melanie likely enjoy classical music") == "aggregation"
        assert classify_query_intent("Did James have a girlfriend") == "factual"
        assert classify_query_intent("Is Deborah married") == "factual"
        assert classify_query_intent("Was James feeling lonely") == "factual"
        assert classify_query_intent("Does John live close to a beach") == "factual"

    def test_factual_expanded(self):
        """Expanded factual patterns: negation, modals, comparisons, possessives."""
        # "Why didn't X" — causal reasoning
        assert classify_query_intent("Why didn't John want to go to Starbucks") == "factual"
        # "Are X and Y [predicate]" — comparison
        assert classify_query_intent("Are John and James fans of the same football team") == "factual"
        # "Does X [expanded verbs]" — broader verb coverage
        assert classify_query_intent("Does Calvin wish to become more popular") == "factual"
        assert classify_query_intent("Does Calvin love music tours") == "factual"
        # "Does X's Y [verb]" — possessive subjects
        assert classify_query_intent("Does Dave's shop employ a lot of people") == "factual"
        # "Did X and Y [verb]" — compound subjects
        assert classify_query_intent("Did John and James study together") == "factual"
        # "Is the [noun] who" — identity questions
        assert classify_query_intent("Is the friend who wrote Deborah the quote alive") == "factual"
        # "What [adj] [noun] is/was" — 2-word gap
        assert classify_query_intent("What card game is Deborah talking about") == "factual"
        # Modal questions — "What can/could/would [person]"
        # "What would be" → aggregation (needs person's interests to infer)
        assert classify_query_intent("What would be a good hobby for Tim") == "aggregation"
        assert classify_query_intent("What can Andrew do to improve his stress") == "factual"
        assert classify_query_intent("What electronic device could Evan gift Sam") == "factual"
        # Context prefix — "Based on" / "Considering"
        assert classify_query_intent("Based on the conversation, did Calvin meet Dave") == "aggregation"
        assert classify_query_intent("Considering their growth, what advice might they give") == "aggregation"
        # "What X wouldn't" — negative conditionals
        assert classify_query_intent("What pets wouldn't cause discomfort to Joanna") == "factual"

    def test_debug(self):
        assert classify_query_intent("why did the build fail") == "debug"
        assert classify_query_intent("error in the deployment") == "debug"
        assert classify_query_intent("crash when loading the model") == "debug"

    def test_decision(self):
        assert classify_query_intent("important product decisions") == "decision"
        assert classify_query_intent("Stripe vs RevenueCat") == "decision"
        assert classify_query_intent("why did we chose Stripe") == "decision"
        assert classify_query_intent("tradeoffs we considered") == "decision"

    def test_exploratory(self):
        assert classify_query_intent("how did we solve the auth problem") == "exploratory"
        assert classify_query_intent("what did we try for caching") == "exploratory"
        assert classify_query_intent("tell me about the migration history") == "exploratory"

    def test_aggregation(self):
        """Aggregation queries gather scattered facts across sessions."""
        assert classify_query_intent("What activities does Melanie partake in") == "aggregation"
        assert classify_query_intent("Where has Melanie camped") == "aggregation"
        assert classify_query_intent("What martial arts has John done") == "aggregation"
        assert classify_query_intent("What types of yoga has Maria practiced") == "aggregation"
        assert classify_query_intent("What do Jon and Gina have in common") == "aggregation"
        assert classify_query_intent("How many times has Melanie gone to the beach") == "aggregation"

    def test_aggregation_excludes_we(self):
        """'what did we try' is exploratory, not aggregation."""
        assert classify_query_intent("what did we try for caching") == "exploratory"
        assert classify_query_intent("what did we decide about auth") != "aggregation"

    def test_aggregation_no_false_positives(self):
        """Generic questions should not be misclassified as aggregation."""
        assert classify_query_intent("who was the first person there") != "aggregation"
        assert classify_query_intent("this is in common use") != "aggregation"
        assert classify_query_intent("both methods and functions work") != "aggregation"
        assert classify_query_intent("what configuration is needed to deploy") != "aggregation"

    def test_procedural(self):
        assert classify_query_intent("how to deploy the service") == "procedural"
        assert classify_query_intent("steps to configure the database") == "procedural"
        assert classify_query_intent("how should we run the tests") == "procedural"

    def test_general(self):
        assert classify_query_intent("latent critic pipeline") == "general"
        assert classify_query_intent("quality curve weighting") == "general"

    def test_temporal(self):
        assert classify_query_intent("when did we discuss the migration") == "temporal"
        assert classify_query_intent("when was the last deployment") == "temporal"
        assert classify_query_intent("what happened last week") == "temporal"
        assert classify_query_intent("how recently was the config changed") == "temporal"
        assert classify_query_intent("what changed between March and June") == "temporal"
        assert classify_query_intent("what happened since March") == "temporal"
        assert classify_query_intent("what happened before June 2023") == "temporal"
        assert classify_query_intent("what happened during May 2023") == "temporal"

    def test_temporal_beats_factual_for_when(self):
        """'When did X' should classify as temporal, not factual."""
        assert classify_query_intent("when did we add the cache") == "temporal"
        assert classify_query_intent("when was the API key rotated") == "temporal"

    def test_intent_params_keys(self):
        """All intents return the expected parameter keys."""
        for intent in ["temporal", "factual", "debug", "decision", "exploratory", "procedural", "aggregation", "general"]:
            params = intent_search_params(intent)
            assert "knowledge_boost" in params
            assert "half_life" in params
            assert "emb_weight" in params

    def test_decision_has_low_knowledge_boost(self):
        """Decision queries should prefer journal entries over knowledge nodes."""
        decision_params = intent_search_params("decision")
        assert decision_params["knowledge_boost"] < 1.0

    def test_debug_has_short_half_life(self):
        """Debug queries should have shorter half_life (recency matters)."""
        debug_params = intent_search_params("debug")
        general_params = intent_search_params("general")
        assert debug_params["half_life"] < general_params["half_life"]

    def test_factual_has_high_knowledge_boost(self):
        """Factual queries should prioritize knowledge nodes."""
        factual_params = intent_search_params("factual")
        general_params = intent_search_params("general")
        assert factual_params["knowledge_boost"] >= general_params["knowledge_boost"]

    def test_temporal_has_low_knowledge_boost(self):
        """Temporal queries need conversation sequences, not knowledge summaries."""
        temporal_params = intent_search_params("temporal")
        general_params = intent_search_params("general")
        assert temporal_params["knowledge_boost"] < general_params["knowledge_boost"]

    def test_temporal_has_max_knowledge_cap(self):
        """Temporal queries should cap knowledge to leave room for raw chunks."""
        temporal_params = intent_search_params("temporal")
        assert "max_knowledge" in temporal_params
        assert temporal_params["max_knowledge"] <= 5

    def test_factual_has_no_max_knowledge_cap(self):
        """Factual queries should NOT cap knowledge — knowledge nodes are primary."""
        factual_params = intent_search_params("factual")
        assert "max_knowledge" not in factual_params

    def test_decision_has_max_knowledge_cap(self):
        """Decision queries prefer journal entries, so knowledge is capped."""
        decision_params = intent_search_params("decision")
        assert "max_knowledge" in decision_params

    def test_decision_intent_classification(self):
        """Decision queries should be classified correctly."""
        assert classify_query_intent("what decisions did we make") == "decision"
        assert classify_query_intent("why did we choose TypeScript") == "decision"
        assert classify_query_intent("Stripe vs RevenueCat tradeoffs") == "decision"
        assert classify_query_intent("what did we switch to") == "decision"

    def test_decision_params_prefer_journals(self):
        """Decision params should de-boost knowledge and boost embeddings."""
        decision_params = intent_search_params("decision")
        general_params = intent_search_params("general")
        # Knowledge boost lower → journal entries rank higher
        assert decision_params["knowledge_boost"] < general_params["knowledge_boost"]
        # Embedding weight higher → semantic matching for paraphrased decisions
        assert decision_params["emb_weight"] > general_params["emb_weight"]


# ---------------------------------------------------------------------------
# Temporal date extraction
# ---------------------------------------------------------------------------


class TestTemporalExtraction:
    """Test extract_temporal_range from query strings."""

    # Fixed reference time for deterministic tests
    NOW = datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc)

    def test_yesterday(self):
        after, before = extract_temporal_range("what happened yesterday", now=self.NOW)
        assert after == "2026-03-09"
        assert before == "2026-03-10"

    def test_today(self):
        after, before = extract_temporal_range("what are we doing today", now=self.NOW)
        assert after == "2026-03-10"
        assert before == "2026-03-11"

    def test_last_week(self):
        after, before = extract_temporal_range("what did we discuss last week", now=self.NOW)
        assert after == "2026-03-03"
        assert before == "2026-03-10"

    def test_last_month(self):
        after, before = extract_temporal_range("changes from last month", now=self.NOW)
        assert after == "2026-02-08"
        assert before == "2026-03-10"

    def test_n_days_ago(self):
        after, before = extract_temporal_range("what happened 3 days ago", now=self.NOW)
        assert after == "2026-03-07"
        assert before == "2026-03-10"

    def test_n_weeks_ago(self):
        after, before = extract_temporal_range("from 2 weeks ago", now=self.NOW)
        assert after == "2026-02-24"
        assert before == "2026-03-10"

    def test_month_day(self):
        after, before = extract_temporal_range("on March 5th", now=self.NOW)
        assert after == "2026-03-05"
        assert before == "2026-03-06"

    def test_month_day_year(self):
        after, before = extract_temporal_range("on February 28, 2025", now=self.NOW)
        assert after == "2025-02-28"
        assert before == "2025-03-01"

    def test_iso_date(self):
        after, before = extract_temporal_range("changes from 2026-03-01", now=self.NOW)
        assert after == "2026-03-01"
        assert before == "2026-03-02"

    def test_no_temporal_expression(self):
        after, before = extract_temporal_range("how does authentication work", now=self.NOW)
        assert after is None
        assert before is None

    def test_this_week(self):
        after, before = extract_temporal_range("what happened this week", now=self.NOW)
        assert after == "2026-03-09"  # Monday (weekday=0)
        assert before == "2026-03-11"

    def test_abbreviated_month(self):
        after, before = extract_temporal_range("on Feb 14", now=self.NOW)
        assert after == "2026-02-14"
        assert before == "2026-02-15"

    def test_month_only_defaults_to_current_year(self):
        after, before = extract_temporal_range("what happened in March", now=self.NOW)
        assert after == "2026-03-01"
        assert before == "2026-04-01"

    def test_month_only_with_explicit_year(self):
        after, before = extract_temporal_range("what happened during May 2023", now=self.NOW)
        assert after == "2023-05-01"
        assert before == "2023-06-01"

    def test_year_only(self):
        after, before = extract_temporal_range("what happened in 2023", now=self.NOW)
        assert after == "2023-01-01"
        assert before == "2024-01-01"

    def test_between_months_same_year(self):
        after, before = extract_temporal_range("what changed between March and June", now=self.NOW)
        assert after == "2026-03-01"
        assert before == "2026-07-01"

    def test_since_month(self):
        after, before = extract_temporal_range("what happened since March", now=self.NOW)
        assert after == "2026-03-01"
        assert before is None

    def test_before_month_with_year(self):
        after, before = extract_temporal_range("what happened before June 2023", now=self.NOW)
        assert after is None
        assert before == "2023-06-01"

    def test_last_n_months(self):
        after, before = extract_temporal_range("what happened over the last 3 months", now=self.NOW)
        assert after == "2025-12-01"
        assert before == "2026-03-10"


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

class TestExtractEntities:
    def test_single_name(self):
        entities = extract_entities("What is Caroline's identity?")
        assert "caroline" in entities

    def test_multiple_names(self):
        entities = extract_entities("Did Caroline visit Melanie last week?")
        assert "caroline" in entities
        assert "melanie" in entities

    def test_no_entities(self):
        entities = extract_entities("how does authentication work?")
        assert len(entities) == 0

    def test_filters_question_words(self):
        """Question words at start of sentence (capitalized) should not be entities."""
        entities = extract_entities("What is the configuration?")
        assert "what" not in entities

    def test_filters_stop_words(self):
        entities = extract_entities("Would Caroline still want to pursue counseling?")
        assert "would" not in entities
        assert "caroline" in entities

    def test_possessive_form(self):
        entities = extract_entities("What are Caroline's hobbies?")
        assert "caroline" in entities
        # Should not include the possessive form
        assert "caroline's" not in entities

    def test_possessive_name_ending_in_s(self):
        """Possessive stripping must not corrupt names ending in 's'."""
        entities = extract_entities("What are James's plans?")
        assert "james" in entities
        # rstrip("'s") would produce "jame" — verify full name preserved
        assert "jame" not in entities

    def test_location_names(self):
        entities = extract_entities("Where did Caroline move from Sweden?")
        assert "caroline" in entities
        assert "sweden" in entities

    def test_short_words_filtered(self):
        """Single-character capitalized words should be filtered."""
        entities = extract_entities("Is A the right choice?")
        assert len(entities) == 0
