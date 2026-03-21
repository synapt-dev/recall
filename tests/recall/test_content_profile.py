from synapt.recall.content_profile import (
    ContentProfile,
    adaptive_params,
    forced_content_profile,
)


def test_adaptive_params_keep_code_dedup_high():
    params = adaptive_params(ContentProfile(total_chunks=10, _type="code"))

    assert params.dedup_jaccard == 0.8
    assert params.max_knowledge_default is None


def test_adaptive_params_keep_personal_dedup_conservative():
    params = adaptive_params(ContentProfile(total_chunks=10, _type="personal"))

    assert params.dedup_jaccard == 0.6
    assert params.max_knowledge_default == 0


def test_adaptive_params_keep_mixed_profile_defaults():
    params = adaptive_params(ContentProfile(total_chunks=10, _type="mixed"))

    assert params.dedup_jaccard == 0.75
    assert params.max_knowledge_default == 5


def test_forced_content_profile_uses_env_override(monkeypatch):
    monkeypatch.setenv("SYNAPT_FORCE_PROFILE", "personal")

    profile = forced_content_profile(total_chunks=7)

    assert profile is not None
    assert profile.total_chunks == 7
    assert profile.content_type == "personal"


def test_forced_content_profile_ignores_unknown_value(monkeypatch):
    monkeypatch.setenv("SYNAPT_FORCE_PROFILE", "not-a-profile")

    assert forced_content_profile(total_chunks=3) is None
