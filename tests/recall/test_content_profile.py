from synapt.recall.content_profile import ContentProfile, adaptive_params


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
