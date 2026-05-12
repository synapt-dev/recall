from __future__ import annotations

from types import SimpleNamespace

from synapt.recall import server


def test_get_index_reuses_embedding_cache_without_false_missing_index(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "recall.db").write_bytes(b"placeholder")

    fake_index = SimpleNamespace(chunks=[], _db=None)
    calls: list[tuple[object, bool]] = []

    def fake_load(directory, use_embeddings=False):
        calls.append((directory, use_embeddings))
        return fake_index

    monkeypatch.setattr(server, "_cached_index", None)
    monkeypatch.setattr(server, "_cached_mtime", 0.0)
    monkeypatch.setattr(server, "_cached_dir", None)
    monkeypatch.setattr(server, "_cached_has_embeddings", False)
    monkeypatch.setattr(server, "project_index_dir", lambda: index_dir)
    monkeypatch.setattr(server.TranscriptIndex, "load", fake_load)

    first = server._get_index(use_embeddings=True)
    second = server._get_index(use_embeddings=True)

    assert first is fake_index
    assert second is fake_index
    assert calls == [(index_dir, True)]
    assert server._cached_has_embeddings is True
