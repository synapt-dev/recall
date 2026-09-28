"""Tests for user-configurable model selection."""

from __future__ import annotations

import json

import pytest

from synapt.recall.config import (
    DEFAULTS,
    RecallConfig,
    clear_config_cache,
    load_config,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear config cache before each test."""
    clear_config_cache()
    yield
    clear_config_cache()


class TestRecallConfig:
    """Test the RecallConfig dataclass."""

    def test_defaults(self):
        cfg = RecallConfig()
        assert cfg.get_model("embedding") == DEFAULTS["embedding"]
        assert cfg.get_model("summarization") == DEFAULTS["summarization"]
        assert cfg.get_model("enrichment") == DEFAULTS["enrichment"]
        assert cfg.backend == "auto"
        assert cfg.get_session_start_continuity() == "automatic"

    def test_session_start_continuity_env_override(self, monkeypatch):
        monkeypatch.setenv("SYNAPT_SESSION_START_CONTINUITY", "explicit")
        assert RecallConfig().get_session_start_continuity() == "explicit"

    def test_invalid_session_start_continuity_falls_back(self):
        cfg = RecallConfig(session_start_continuity="surprise-me")
        assert cfg.get_session_start_continuity() == "automatic"

    def test_custom_model(self):
        cfg = RecallConfig(models={**DEFAULTS, "embedding": "custom-model"})
        assert cfg.get_model("embedding") == "custom-model"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("SYNAPT_SUMMARY_MODEL", "google/flan-t5-large")
        cfg = RecallConfig()
        assert cfg.get_model("summarization") == "google/flan-t5-large"

    def test_env_override_enrichment(self, monkeypatch):
        monkeypatch.setenv("SYNAPT_ENRICHMENT_MODEL", "custom/enrichment")
        cfg = RecallConfig()
        assert cfg.get_model("enrichment") == "custom/enrichment"

    def test_env_embedding_override_is_NOT_honoured(self, monkeypatch):
        """The regression witness for the removal.

        SYNAPT_EMBEDDING_MODEL is no longer in _ENV_MAP. It never reached the
        constructor -- LocalEmbeddings() is always built with the default -- so
        honouring it here only moved the stats row, making the status table name
        an embedding model the product never loaded. Setting it must now leave
        the resolved model at the default, which is what the runtime uses.
        """
        monkeypatch.setenv("SYNAPT_EMBEDDING_MODEL", "all-MiniLM-L12-v2")
        cfg = RecallConfig()
        assert cfg.get_model("embedding") == DEFAULTS["embedding"]

    def test_env_override_reranker(self, monkeypatch):
        monkeypatch.setenv("SYNAPT_RERANKER_MODEL", "custom/reranker")
        cfg = RecallConfig()
        assert cfg.get_model("reranker") == "custom/reranker"

    def test_env_override_consolidation(self, monkeypatch):
        monkeypatch.setenv("SYNAPT_CONSOLIDATION_MODEL", "custom/consolidation")
        cfg = RecallConfig()
        assert cfg.get_model("consolidation") == "custom/consolidation"

    def test_active_models_includes_all(self):
        cfg = RecallConfig()
        models = cfg.active_models()
        for key in DEFAULTS:
            assert key in models

    def test_query_freshness_env_overrides_configured_values(self, monkeypatch):
        cfg = RecallConfig(
            query_freshness={
                "age_threshold_seconds": 120.0,
                "byte_trigger": 2048.0,
                "step_bytes": 4096.0,
                "byte_cap": 8192.0,
                "wall_seconds": 2.0,
            }
        )
        monkeypatch.setenv("SYNAPT_QUERY_FRESHNESS_BYTE_CAP", "16384")
        monkeypatch.setenv("SYNAPT_QUERY_FRESHNESS_WALL_SECONDS", "3.5")

        values = cfg.get_query_freshness()

        assert values["age_threshold_seconds"] == 120.0
        assert values["byte_cap"] == 16384.0
        assert values["wall_seconds"] == 3.5

    def test_unknown_key_returns_empty(self):
        cfg = RecallConfig()
        assert cfg.get_model("nonexistent") == ""


class TestLoadConfig:
    """Test config loading from files."""

    def test_loads_defaults_when_no_files(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.get_model("embedding") == DEFAULTS["embedding"]

    def test_loads_global_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        monkeypatch.chdir(tmp_path)

        config_dir = tmp_path / ".synapt"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({
            "models": {"embedding": "custom-global-model"}
        }))

        cfg = load_config()
        assert cfg.get_model("embedding") == "custom-global-model"
        # Other models stay at defaults
        assert cfg.get_model("summarization") == DEFAULTS["summarization"]

    def test_project_config_overrides_global(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        monkeypatch.chdir(tmp_path)

        # Global config
        global_dir = tmp_path / ".synapt"
        global_dir.mkdir()
        (global_dir / "config.json").write_text(json.dumps({
            "models": {"embedding": "global-model", "summarization": "global-summary"}
        }))

        # Project config
        project_dir = tmp_path / ".synapt" / "recall"
        project_dir.mkdir(parents=True)
        (project_dir / "config.json").write_text(json.dumps({
            "models": {"embedding": "project-model"}
        }))

        cfg = load_config()
        assert cfg.get_model("embedding") == "project-model"
        assert cfg.get_model("summarization") == "global-summary"

    def test_project_config_controls_session_start_continuity(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        project_dir = tmp_path / ".synapt" / "recall"
        project_dir.mkdir(parents=True)
        (project_dir / "config.json").write_text(json.dumps({
            "session_start": {"continuity": "explicit"}
        }))

        assert load_config().get_session_start_continuity() == "explicit"

    def test_project_config_controls_query_freshness(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        project_dir = tmp_path / ".synapt" / "recall"
        project_dir.mkdir(parents=True)
        (project_dir / "config.json").write_text(
            json.dumps(
                {
                    "query_freshness": {
                        "age_threshold_seconds": 45,
                        "byte_trigger": 1024,
                    }
                }
            )
        )

        values = load_config().get_query_freshness()

        assert values["age_threshold_seconds"] == 45.0
        assert values["byte_trigger"] == 1024.0
        assert values["step_bytes"] == 4 * 1024 * 1024

    def test_env_var_overrides_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SYNAPT_SUMMARY_MODEL", "env-model")

        # Config file
        config_dir = tmp_path / ".synapt"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({
            "models": {"summarization": "file-model"}
        }))

        cfg = load_config()
        # Env var wins
        assert cfg.get_model("summarization") == "env-model"

    # Roles deliberately WITHOUT an env override, each with its reason. This is
    # an ALLOW-LIST OF ONE, not a relaxation of the invariant: a sixth role added
    # to DEFAULTS that has no override still reddens this test, which is what it
    # exists for.
    NOT_OVERRIDABLE = {
        # SYNAPT_EMBEDDING_MODEL deliberately has no override: it never reached the
        # constructor -- LocalEmbeddings() is always built with the default -- so
        # honouring it only moved the stats row and made the table name a model
        # the product never loaded. Removed rather than wired. See the comment at
        # _ENV_MAP in config.py.
        "embedding",
    }

    def test_every_model_role_has_an_env_override(self):
        """The set-level invariant, so a sixth role cannot be added silently.

        No test referenced `_ENV_MAP`/`_KEY_TO_ENV` at all before this: the per-role
        witnesses would stay green while a new role in DEFAULTS had no override, which
        is precisely how consolidation sat without one. Both directions are asserted --
        every role has a var, and no var names a role that does not exist.

        `NOT_OVERRIDABLE` carries the roles the product deliberately does not
        override, with the reason on the entry. It is a named exception list rather
        than a weakened assertion, so the invariant keeps its full force for every
        other role."""
        from synapt.recall.config import _ENV_MAP

        roles = set(DEFAULTS)
        mapped = set(_ENV_MAP.values())
        expected = roles - self.NOT_OVERRIDABLE
        assert mapped == expected, (
            f"roles without an env override: {sorted(expected - mapped)}; "
            f"overrides naming unknown roles: {sorted(mapped - roles)}"
        )

    def test_consolidation_precedence_default_then_file_then_env(self, tmp_path, monkeypatch):
        """The three precedence levels for the consolidation role, the fifth role.

        The default level is read from DEFAULTS rather than restated as a literal, so
        this witness keeps its meaning if the default itself changes for platform
        reasons; what it pins is the precedence, not the model name.
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        monkeypatch.chdir(tmp_path)

        # 1. nothing set -> the default
        assert load_config().get_model("consolidation") == DEFAULTS["consolidation"]

        # 2. the config file wins over the default
        config_dir = tmp_path / ".synapt"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({
            "models": {"consolidation": "file-model"}
        }))
        assert load_config().get_model("consolidation") == "file-model"

        # 3. the env var wins over the file
        monkeypatch.setenv("SYNAPT_CONSOLIDATION_MODEL", "env-model")
        assert load_config().get_model("consolidation") == "env-model"

    def test_backend_from_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        monkeypatch.chdir(tmp_path)

        config_dir = tmp_path / ".synapt"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({
            "backend": "transformers"
        }))

        cfg = load_config()
        assert cfg.backend == "transformers"

    def test_backend_env_overrides_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SYNAPT_SUMMARY_BACKEND", "mlx")

        config_dir = tmp_path / ".synapt"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({
            "backend": "transformers"
        }))

        cfg = load_config()
        assert cfg.backend == "mlx"

    def test_config_caching(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        monkeypatch.chdir(tmp_path)

        cfg1 = load_config()
        cfg2 = load_config()
        assert cfg1 is cfg2

    def test_malformed_json_uses_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        monkeypatch.chdir(tmp_path)

        config_dir = tmp_path / ".synapt"
        config_dir.mkdir()
        (config_dir / "config.json").write_text("not valid json{{{")

        cfg = load_config()
        assert cfg.get_model("embedding") == DEFAULTS["embedding"]


class TestRouterConfigIntegration:
    """Test that the model router uses config."""

    def test_get_encoder_decoder_model_uses_config(self, tmp_path, monkeypatch):
        from synapt.recall._model_router import get_encoder_decoder_model

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        monkeypatch.chdir(tmp_path)

        config_dir = tmp_path / ".synapt"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({
            "models": {"summarization": "google/flan-t5-large"}
        }))

        assert get_encoder_decoder_model() == "google/flan-t5-large"
