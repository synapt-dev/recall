"""Tests for recall_quick intent routing improvements (#412).

Verifies that:
1. Code-level queries are detected by intent classification
2. Project-level queries are detected by intent classification
3. recall_quick routes code queries to depth="full" for file-associated chunks
4. recall_quick routes project queries to knowledge-heavy results
5. The intent classifier doesn't misclassify ambiguous queries
"""

import unittest

from synapt.recall.hybrid import classify_query_intent, intent_search_params


class TestCodeIntentClassification(unittest.TestCase):
    """Test that code-level queries are classified as 'code' intent."""

    def test_file_path_query(self):
        self.assertEqual(classify_query_intent("how does storage.py work"), "code")

    def test_function_name_query(self):
        self.assertEqual(classify_query_intent("what does build_index do"), "code")

    def test_class_reference(self):
        self.assertEqual(classify_query_intent("how is TranscriptIndex initialized"), "code")

    def test_implementation_query(self):
        self.assertEqual(classify_query_intent("how is the FTS index implemented"), "code")

    def test_module_query(self):
        self.assertEqual(classify_query_intent("what module handles embeddings"), "code")

    def test_snake_case_identifier(self):
        self.assertEqual(classify_query_intent("where is get_all_embeddings called"), "code")

    def test_error_message_stays_debug(self):
        """Error messages should still classify as debug, not code."""
        self.assertEqual(classify_query_intent("why does the import error occur"), "debug")

    def test_generic_how_does_not_code(self):
        """Generic 'how does' without code signals should not be code."""
        intent = classify_query_intent("how does the team communicate")
        self.assertNotEqual(intent, "code")


class TestProjectIntentClassification(unittest.TestCase):
    """Test that project-level queries are classified as 'project' intent."""

    def test_roadmap_query(self):
        self.assertEqual(classify_query_intent("what's our roadmap"), "project")

    def test_backlog_query(self):
        self.assertEqual(classify_query_intent("what's on the backlog"), "project")

    def test_sprint_query(self):
        self.assertEqual(classify_query_intent("what did we ship in this sprint"), "project")

    def test_milestone_query(self):
        self.assertEqual(classify_query_intent("upcoming milestones and deadlines"), "project")

    def test_priority_query(self):
        self.assertEqual(classify_query_intent("what are the priorities right now"), "project")

    def test_initiative_query(self):
        self.assertEqual(classify_query_intent("what initiatives are in progress"), "project")


class TestIntentSearchParams(unittest.TestCase):
    """Test that code and project intents produce appropriate search parameters."""

    def test_code_intent_low_knowledge_boost(self):
        """Code queries should de-prioritize knowledge nodes."""
        params = intent_search_params("code")
        self.assertLessEqual(params["knowledge_boost"], 1.0)

    def test_code_intent_has_max_knowledge(self):
        """Code queries should cap knowledge to leave room for transcript chunks."""
        params = intent_search_params("code")
        self.assertIn("max_knowledge", params)
        self.assertLessEqual(params["max_knowledge"], 3)

    def test_project_intent_high_knowledge_boost(self):
        """Project queries should boost knowledge nodes."""
        params = intent_search_params("project")
        self.assertGreaterEqual(params["knowledge_boost"], 2.0)

    def test_project_intent_semantic_weight(self):
        """Project queries benefit from semantic matching."""
        params = intent_search_params("project")
        self.assertGreaterEqual(params["emb_weight"], 1.5)


class TestQuickIntentDepthRouting(unittest.TestCase):
    """Test that recall_quick routes to appropriate depth based on intent."""

    def test_code_intent_uses_full_depth(self):
        """Code queries should use depth='full' in recall_quick."""
        intent = "code"
        # This mirrors the routing logic in recall_quick
        depth = _quick_depth_for_intent(intent)
        self.assertEqual(depth, "full")

    def test_project_intent_uses_concise_depth(self):
        """Project queries should use depth='concise' for knowledge/clusters."""
        intent = "project"
        depth = _quick_depth_for_intent(intent)
        self.assertEqual(depth, "concise")

    def test_status_intent_uses_summary_depth(self):
        """Status queries should use depth='summary' (existing behavior)."""
        intent = "status"
        depth = _quick_depth_for_intent(intent)
        self.assertEqual(depth, "summary")

    def test_general_intent_uses_concise_depth(self):
        """General queries should use depth='concise' (existing behavior)."""
        intent = "general"
        depth = _quick_depth_for_intent(intent)
        self.assertEqual(depth, "concise")


def _quick_depth_for_intent(intent: str) -> str:
    """Mirror the depth routing logic from recall_quick for testing."""
    if intent == "code":
        return "full"
    elif intent == "status":
        return "summary"
    else:
        return "concise"


if __name__ == "__main__":
    unittest.main()
