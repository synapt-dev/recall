"""User-configurable model selection and query defaults for the recall system.

Loads preferences from config files and environment variables.
Priority (highest wins): env vars → project config → global config → defaults.

Config locations:
  - Global: ~/.synapt/config.json
  - Project: .synapt/recall/config.json (relative to project root)

Example config:
  {
    "models": {
      "embedding": "all-MiniLM-L6-v2",
      "summarization": "google/flan-t5-base",
      "enrichment": "laynepro/t5-enrichment-v2",
      "consolidation": "mlx-community/Ministral-3-3B-Instruct-2512-4bit",
      "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    },
    "backend": "auto",
    "max_tokens": 1500
  }
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Default model names — these match the current hardcoded defaults
DEFAULTS = {
    "embedding": "all-MiniLM-L6-v2",
    "summarization": "google/flan-t5-base",
    "enrichment": "laynepro/t5-enrichment-v2",
    "consolidation": "mlx-community/Ministral-3-3B-Instruct-2512-4bit",
    "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}

# Env var → config key mapping (model overrides)
_ENV_MAP = {
    "SYNAPT_SUMMARY_MODEL": "summarization",
    "SYNAPT_ENRICHMENT_MODEL": "enrichment",
    "SYNAPT_RERANKER_MODEL": "reranker",
    "SYNAPT_CONSOLIDATION_MODEL": "consolidation",
    # SYNAPT_EMBEDDING_MODEL is deliberately NOT here. It never reached the
    # constructor -- LocalEmbeddings() is always built with the default -- so
    # honouring it only moved the stats row, making the table name an embedding
    # model the product never loaded. Removed rather than wired: on a
    # fixed-width store the switch could only ever accept another 384-dimension
    # model, which the width check would then have to explain. Rewire if a user
    # asks. See the status-table row in resolve_model_states().
}

# Default query parameters
DEFAULT_MAX_TOKENS = 1500
DEFAULT_SESSION_START_CONTINUITY = "automatic"
SESSION_START_CONTINUITY_MODES = {"off", "explicit", "automatic", "always"}
DEFAULT_QUERY_FRESHNESS = {
    "age_threshold_seconds": 600.0,
    "byte_trigger": 4 * 1024 * 1024,
    "step_bytes": 4 * 1024 * 1024,
    "byte_cap": 32 * 1024 * 1024,
    "wall_seconds": 5.0,
}

# Reverse lookup: config key → env var name
_KEY_TO_ENV = {v: k for k, v in _ENV_MAP.items()}


@dataclass
class RecallConfig:
    """Resolved model configuration."""

    models: dict[str, str] = field(default_factory=lambda: dict(DEFAULTS))
    backend: str = "auto"
    max_tokens: int = DEFAULT_MAX_TOKENS
    session_start_continuity: str = DEFAULT_SESSION_START_CONTINUITY
    query_freshness: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_QUERY_FRESHNESS)
    )
    # Guard settings a user set. Each entry carries the value AND the layer it came
    # from, because a reader has to report the SOURCE beside the number: a bare number with
    # no source is what let the 6 GB build floor outlive the measurement contradicting it.
    # Only keys a user actually set appear here -- a missing key means unset, and the reader
    # applies its own default, so exactly one place owns each default.
    #
    # The value is stored RAW, not coerced here: validation lives in the reader so that a
    # bad setting can be reported back to the person who typed it ("requested true, using 6")
    # rather than only logged. Coercing here would also be wrong -- float(True) is 1.0, so a
    # boolean typo would silently become a 1 GB build floor.
    memory: dict[str, tuple[object, str]] = field(default_factory=dict)
    # `memory.*` keys found in the PROJECT layer, by name. These guards are host properties
    # and never read that layer; they are recorded so `maintenance status` can say so instead
    # of letting a project setting look applied.
    memory_ignored_project_keys: list[str] = field(default_factory=list)

    def get_model(self, key: str) -> str:
        """Get a model name by key, with env var override."""
        # Check env var first (highest priority)
        env_var = _KEY_TO_ENV.get(key)
        if env_var:
            env_val = os.environ.get(env_var)
            if env_val:
                return env_val

        return self.models.get(key, DEFAULTS.get(key, ""))

    def get_max_tokens(self) -> int:
        """Get the configured max_tokens default, with env var override."""
        env_val = os.environ.get("SYNAPT_MAX_TOKENS")
        if env_val:
            try:
                return int(env_val)
            except ValueError:
                logger.warning("Invalid SYNAPT_MAX_TOKENS=%r, using %d", env_val, self.max_tokens)
        return self.max_tokens

    def get_session_start_continuity(self) -> str:
        """Return the SessionStart recovery policy, with env override."""
        value = os.environ.get(
            "SYNAPT_SESSION_START_CONTINUITY", self.session_start_continuity,
        ).strip().lower()
        if value not in SESSION_START_CONTINUITY_MODES:
            logger.warning(
                "Invalid SessionStart continuity mode %r, using %s",
                value,
                DEFAULT_SESSION_START_CONTINUITY,
            )
            return DEFAULT_SESSION_START_CONTINUITY
        return value

    def get_query_freshness(self) -> dict[str, float]:
        """Return bounded query-refresh controls with environment overrides."""
        result = dict(self.query_freshness)
        env_names = {
            "age_threshold_seconds": "SYNAPT_QUERY_FRESHNESS_AGE_SECONDS",
            "byte_trigger": "SYNAPT_QUERY_FRESHNESS_BYTE_TRIGGER",
            "step_bytes": "SYNAPT_QUERY_FRESHNESS_STEP_BYTES",
            "byte_cap": "SYNAPT_QUERY_FRESHNESS_BYTE_CAP",
            "wall_seconds": "SYNAPT_QUERY_FRESHNESS_WALL_SECONDS",
        }
        for key, env_name in env_names.items():
            value = os.environ.get(env_name)
            if value is None:
                continue
            try:
                result[key] = float(value)
            except ValueError:
                logger.warning("Invalid %s=%r, using configured value", env_name, value)
        return result

    def active_models(self) -> dict[str, str]:
        """Get all active model names (with env overrides applied)."""
        result = {}
        for key in DEFAULTS:
            result[key] = self.get_model(key)
        return result


# Module-level cache
_cached_config: RecallConfig | None = None
_cached_mtime: float = 0.0
_cached_project_path: str | None = None


def _find_project_config() -> str | None:
    """Find the project-level config file by walking up from cwd."""
    cwd = os.getcwd()
    for _ in range(20):  # Max depth
        candidate = os.path.join(cwd, ".synapt", "recall", "config.json")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(cwd)
        if parent == cwd:
            break
        cwd = parent
    return None


def _load_json(path: str) -> dict:
    """Load a JSON file, returning empty dict on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def load_config() -> RecallConfig:
    """Load and merge configuration from all sources.

    Priority: env vars → project config → global config → defaults.
    Results are cached and reloaded when config files change.
    """
    global _cached_config, _cached_mtime, _cached_project_path

    # Check if we need to reload
    global_path = os.path.expanduser("~/.synapt/config.json")

    # Reuse cached project path to avoid walking the directory tree every call
    if _cached_config is not None:
        project_path = _cached_project_path
    else:
        project_path = _find_project_config()
        _cached_project_path = project_path

    # Use max mtime of both config files for cache invalidation
    current_mtime = 0.0
    for path in (global_path, project_path):
        if path:
            try:
                current_mtime = max(current_mtime, os.path.getmtime(path))
            except OSError:
                pass

    if _cached_config is not None and current_mtime == _cached_mtime:
        return _cached_config

    # Start with defaults
    models = dict(DEFAULTS)

    # Layer 1: Global config
    global_data = _load_json(global_path)
    if "models" in global_data:
        models.update(global_data["models"])

    # Layer 2: Project config (overrides global)
    project_data = {}
    if project_path:
        project_data = _load_json(project_path)
        if "models" in project_data:
            models.update(project_data["models"])

    backend = (
        project_data.get("backend")
        or global_data.get("backend")
        or "auto"
    )

    # Env var for backend override
    env_backend = os.environ.get("SYNAPT_SUMMARY_BACKEND", "").lower()
    if env_backend:
        backend = env_backend

    # max_tokens: project overrides global overrides default
    max_tokens = DEFAULT_MAX_TOKENS
    if "max_tokens" in global_data:
        try:
            max_tokens = int(global_data["max_tokens"])
        except (ValueError, TypeError):
            logger.warning("Invalid max_tokens in global config, using default")
    if "max_tokens" in project_data:
        try:
            max_tokens = int(project_data["max_tokens"])
        except (ValueError, TypeError):
            logger.warning("Invalid max_tokens in project config, using default")

    continuity = DEFAULT_SESSION_START_CONTINUITY
    for data in (global_data, project_data):
        session_start = data.get("session_start", {})
        if isinstance(session_start, dict) and "continuity" in session_start:
            continuity = str(session_start["continuity"]).strip().lower()

    query_freshness = dict(DEFAULT_QUERY_FRESHNESS)
    for data in (global_data, project_data):
        configured = data.get("query_freshness", {})
        if not isinstance(configured, dict):
            continue
        for key in DEFAULT_QUERY_FRESHNESS:
            if key not in configured:
                continue
            try:
                query_freshness[key] = float(configured[key])
            except (TypeError, ValueError):
                logger.warning("Invalid query_freshness.%s, using prior value", key)

    # memory: the GLOBAL layer only. These guards are HOST properties, so the
    # precedence is env > global config > default and the PROJECT file is deliberately not
    # consulted: it is found from the caller's cwd, so one host would gate differently by
    # working directory (the cwd-keyed store defect family), and a project config committed
    # to a repository could lower a safety floor for everyone who clones it. A project key is
    # recorded by name so `maintenance status` can say it is ignored.
    # The value is kept RAW and its validation stays in the reader
    # (`_resolve_build_min_free_gb`), which is where the "requested X, using Y" note is made.
    memory: dict[str, tuple[object, str]] = {}
    configured_memory = global_data.get("memory", {})
    if isinstance(configured_memory, dict):
        for name, raw in configured_memory.items():
            memory[name] = (raw, "global config")
    elif configured_memory:
        logger.warning("Invalid memory section in the global config, ignoring it")

    memory_ignored_project_keys: list[str] = []
    project_memory = project_data.get("memory", {})
    if isinstance(project_memory, dict):
        memory_ignored_project_keys = sorted(project_memory)

    config = RecallConfig(
        models=models,
        backend=backend,
        max_tokens=max_tokens,
        session_start_continuity=continuity,
        query_freshness=query_freshness,
        memory=memory,
        memory_ignored_project_keys=memory_ignored_project_keys,
    )
    _cached_config = config
    _cached_mtime = current_mtime

    logger.debug("Loaded config: models=%s, backend=%s", models, backend)
    return config


def clear_config_cache() -> None:
    """Clear the config cache. Useful for testing."""
    global _cached_config, _cached_mtime, _cached_project_path
    _cached_config = None
    _cached_mtime = 0.0
    _cached_project_path = None
