"""Focused tests for synapt.recall.server tool surfaces."""

from unittest.mock import patch


class _FakeIndex:
    def list_sessions(self, max_sessions=20, after=None, before=None):
        return []


def test_recall_sessions_uses_non_embedding_index():
    from synapt.recall import server

    with patch.object(server, "_get_index", return_value=_FakeIndex()) as mock_get_index:
        result = server.recall_sessions(max_sessions=2)

    mock_get_index.assert_called_once_with(use_embeddings=False)
    assert result == "No sessions found."
