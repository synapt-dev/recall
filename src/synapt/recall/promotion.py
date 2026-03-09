"""Promotion pipeline for the adaptive memory layer.

Checks access stats against promotion thresholds and returns actions
to execute. Cheap actions (clustering orphans, tier flagging) run
inline after search. Expensive actions (LLM summaries, knowledge
promotion) are deferred to recall_build.

Promotion tier state machine:
    raw -> clustered -> summarized -> promoted -> knowledge

Only explicit_count (user-initiated search + context) drives promotions.
Hook-triggered accesses do NOT count toward promotion thresholds.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synapt.recall.storage import RecallDB

logger = logging.getLogger(__name__)

# Tier constants
TIER_RAW = "raw"
TIER_CLUSTERED = "clustered"
TIER_SUMMARIZED = "summarized"
TIER_PROMOTED = "promoted"
TIER_KNOWLEDGE = "knowledge"

# Action constants
ACTION_ENSURE_CLUSTER = "ensure_cluster"
ACTION_GENERATE_LLM_SUMMARY = "generate_llm_summary"
ACTION_FLAG_KNOWLEDGE_CANDIDATE = "flag_knowledge_candidate"
ACTION_AUTO_PROMOTE_KNOWLEDGE = "auto_promote_knowledge"


def check_promotions(db: RecallDB, item_type: str, item_id: str) -> list[str]:
    """Check if an item qualifies for promotion based on access stats.

    Returns a list of promotion actions to take (may be empty).
    Called after recording an access -- promotions happen lazily.
    """
    stats = db.get_access_stats(item_type, item_id)
    if stats is None:
        return []

    actions: list[str] = []
    tier = stats.get("promotion_tier", TIER_RAW)

    if (tier == TIER_RAW
            and stats["explicit_count"] >= 3
            and stats.get("distinct_sessions", 0) >= 2):
        actions.append(ACTION_ENSURE_CLUSTER)

    elif (tier == TIER_CLUSTERED
            and stats["explicit_count"] >= 3):
        actions.append(ACTION_GENERATE_LLM_SUMMARY)

    elif (tier == TIER_SUMMARIZED
            and stats["explicit_count"] >= 5
            and stats.get("distinct_sessions", 0) >= 3):
        actions.append(ACTION_FLAG_KNOWLEDGE_CANDIDATE)

    elif (tier == TIER_PROMOTED
            and stats["explicit_count"] >= 10
            and stats.get("distinct_queries", 0) >= 5):
        actions.append(ACTION_AUTO_PROMOTE_KNOWLEDGE)

    return actions


def execute_cheap_promotions(
    db: RecallDB,
    item_type: str,
    item_id: str,
    actions: list[str],
) -> list[str]:
    """Execute zero-cost promotion actions inline.

    Returns list of completed action descriptions.
    Only handles clustering orphans and tier flagging.
    LLM-dependent actions are skipped (deferred to recall_build).
    """
    completed: list[str] = []

    for action in actions:
        if action == ACTION_ENSURE_CLUSTER and item_type == "chunk":
            cluster_id = db.create_singleton_cluster(item_id)
            if cluster_id:
                db.update_promotion_tier(item_type, item_id, TIER_CLUSTERED)
                completed.append(
                    f"Created singleton cluster {cluster_id} for chunk {item_id}"
                )
                logger.debug("Promotion: %s", completed[-1])

        elif action == ACTION_FLAG_KNOWLEDGE_CANDIDATE:
            db.update_promotion_tier(item_type, item_id, TIER_PROMOTED)
            completed.append(
                f"Flagged {item_type}:{item_id} as knowledge candidate"
            )
            logger.debug("Promotion: %s", completed[-1])

        # LLM-dependent actions are NOT executed here:
        # ACTION_GENERATE_LLM_SUMMARY -> deferred to recall_build
        # ACTION_AUTO_PROMOTE_KNOWLEDGE -> deferred to recall_build

    return completed


def _resolve_cluster_id(db: RecallDB, item: dict) -> str | None:
    """Find the cluster_id for a promotion item.

    Cluster-type items use item_id directly. Chunk-type items look up
    their cluster membership via cluster_chunks.
    """
    if item["item_type"] == "cluster":
        return item["item_id"]
    if item["item_type"] == "chunk":
        return db.get_chunk_cluster_id(item["item_id"])
    return None


def process_build_promotions(
    db: RecallDB,
    max_knowledge_promotions: int = 3,
    max_llm_summaries: int = 5,
) -> dict:
    """Execute pending expensive promotions during recall_build.

    Scans access_stats for items at tiers that qualify for advancement.
    LLM summary generation and knowledge promotion happen here, capped
    by budget limits.

    Returns dict with counts of actions taken.
    """
    from synapt.recall.clustering import generate_llm_summary, create_summary_client

    result = {
        "summaries_upgraded": 0,
        "llm_summaries_generated": 0,
        "knowledge_promoted": 0,
        "candidates_flagged": 0,
    }

    # 1. Advance 'clustered' items that meet the LLM summary threshold.
    #    Try LLM summary for the hottest clusters (budget-capped).
    #    Fall back to advancing the tier without LLM if unavailable.
    #    Client is created lazily on first qualifying cluster and reused.
    _sentinel = object()
    llm_client: object = _sentinel  # Not yet attempted
    llm_count = 0

    clustered_items = db.items_at_tier(TIER_CLUSTERED)
    for item in clustered_items:
        if item["explicit_count"] >= 3:
            # Try LLM summary for this cluster (budget-capped)
            if llm_count < max_llm_summaries:
                cluster_id = _resolve_cluster_id(db, item)
                if cluster_id:
                    chunk_texts = db.get_cluster_chunk_texts(cluster_id)
                    if chunk_texts:
                        # Lazy client creation — once per build
                        if llm_client is _sentinel:
                            llm_client = create_summary_client()
                        cluster_info = db.get_cluster(cluster_id)
                        topic = cluster_info.get("topic", "unknown") if cluster_info else "unknown"
                        summary = generate_llm_summary(
                            chunk_texts, topic, client=llm_client,
                        )
                        if summary:
                            db.save_cluster_summary(
                                cluster_id, summary, method="llm",
                            )
                            llm_count += 1
                            result["llm_summaries_generated"] += 1
                            logger.debug(
                                "LLM summary generated for cluster %s",
                                cluster_id,
                            )

            db.update_promotion_tier(
                item["item_type"], item["item_id"], TIER_SUMMARIZED,
            )
            result["summaries_upgraded"] += 1

    # 2. Advance 'summarized' items to 'promoted' (knowledge candidate)
    summarized_items = db.items_at_tier(TIER_SUMMARIZED)
    for item in summarized_items:
        if (item["explicit_count"] >= 5
                and item.get("distinct_sessions", 0) >= 3):
            db.update_promotion_tier(
                item["item_type"], item["item_id"], TIER_PROMOTED,
            )
            result["candidates_flagged"] += 1

    # 3. Advance 'promoted' items to 'knowledge' (auto-promote).
    #    Budget-capped to avoid expensive LLM calls.
    promoted_items = db.items_at_tier(TIER_PROMOTED)
    promoted_count = 0
    for item in promoted_items:
        if promoted_count >= max_knowledge_promotions:
            break
        if (item["explicit_count"] >= 10
                and item.get("distinct_queries", 0) >= 5):
            db.update_promotion_tier(
                item["item_type"], item["item_id"], TIER_KNOWLEDGE,
            )
            promoted_count += 1
    result["knowledge_promoted"] = promoted_count

    return result
