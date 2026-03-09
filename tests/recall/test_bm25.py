"""Tests for the internalized BM25 implementation."""

from synapt.recall.bm25 import BM25, _stem, _tokenize


def test_tokenize_basic():
    tokens = _tokenize("Hello World! foo-bar baz_qux")
    assert "hello" in tokens
    assert "world" in tokens
    assert "baz_qux" in tokens


def test_tokenize_filters_short():
    tokens = _tokenize("a b cd ef")
    assert "a" not in tokens
    assert "b" not in tokens
    assert "cd" in tokens
    assert "ef" in tokens


def test_tokenize_preserves_dots():
    tokens = _tokenize("api_index.py os.path.join")
    assert "api_index.py" in tokens
    assert "os.path.join" in tokens


def test_bm25_empty_corpus():
    bm25 = BM25()
    bm25.index([])
    scores = bm25.score(["hello"])
    assert scores == []


def test_bm25_single_doc():
    bm25 = BM25()
    bm25.index([["hello", "world"]])
    scores = bm25.score(["hello"])
    assert len(scores) == 1
    assert scores[0] > 0


def test_bm25_multi_doc_ranking():
    bm25 = BM25()
    bm25.index([
        ["quality", "curve", "hermite", "spline"],
        ["harness", "bug", "swift", "tests"],
        ["quality", "curve", "zone", "weight", "quality"],  # repeated term
    ])
    scores = bm25.score(["quality", "curve"])
    # Doc 2 (index 2) has more "quality" occurrences, should score highest
    assert scores[2] > scores[0]
    # Doc 1 (index 1) has neither term
    assert scores[1] == 0.0


def test_bm25_idf_behavior():
    """Terms in fewer documents get higher IDF."""
    bm25 = BM25()
    bm25.index([
        ["common", "rare"],
        ["common", "other"],
        ["common", "another"],
    ])
    assert bm25.idf["rare"] > bm25.idf["common"]


# ---------------------------------------------------------------------------
# Tests: suffix stripper
# ---------------------------------------------------------------------------

def test_stem_plurals():
    """Strip -s, -es, -sses, -ies."""
    assert _stem("fixes") == "fix"
    assert _stem("decisions") == "decision"
    assert _stem("caresses") == "caress"
    assert _stem("ponies") == "poni"


def test_stem_past_tense():
    """Strip -ed."""
    assert _stem("orphaned") == "orphan"
    assert _stem("decided") == "decid"
    assert _stem("fixed") == "fix"
    assert _stem("implemented") == "implement"


def test_stem_progressive():
    """Strip -ing."""
    assert _stem("orphaning") == "orphan"
    assert _stem("deciding") == "decid"
    assert _stem("fixing") == "fix"
    assert _stem("implementing") == "implement"


def test_stem_base_matches_inflected():
    """Critical: base form must match inflected forms after stemming."""
    # This is the core issue from #213 — "orphan" should match "orphaned"
    assert _stem("orphan") == _stem("orphaned")
    assert _stem("orphan") == _stem("orphaning")
    assert _stem("fix") == _stem("fixed")
    assert _stem("fix") == _stem("fixing")
    assert _stem("fix") == _stem("fixes")
    # Known limitations of suffix stripping without a dictionary:
    # 1. Base forms ending in 'e': decide→decide, decided→decid (mismatch)
    # 2. -ement words: implement→implement, implemented→implement (OK!),
    #    but movement→mov, move→move (mismatch via -ement compound rule)
    # Inflected forms still match each other in both cases:
    assert _stem("decided") == _stem("deciding")
    assert _stem("implemented") == _stem("implementing")
    # FTS5 Porter tokenizer handles all these correctly — this stemmer
    # is only the BM25 fallback path.


def test_stem_short_words_unchanged():
    """Words <= 3 chars should not be stemmed."""
    assert _stem("go") == "go"
    assert _stem("an") == "an"
    assert _stem("the") == "the"
    assert _stem("bus") == "bus"


def test_stem_no_false_stripping():
    """Words that happen to end in a suffix shouldn't be incorrectly stripped."""
    assert _stem("thing") == "thing"   # -ing too short
    assert _stem("king") == "king"     # -ing too short
    assert _stem("red") == "red"       # -ed too short
    assert _stem("bass") == "bass"     # -ss excluded from -s rule
    # RISK-4 fix: -ment/-ness require stem >= 5 to avoid over-stripping
    assert _stem("comment") == "comment"   # NOT "com"
    assert _stem("harness") == "harness"   # NOT "har"
    assert _stem("moment") == "moment"     # NOT "mom"
    # Long words still strip correctly
    assert _stem("replacement") != "replacement"  # "replac" + "ement"
    assert _stem("awareness") != "awareness"      # "aware" + "ness"


def test_stem_derivational():
    """Strip derivational suffixes."""
    assert _stem("effectiveness") == "effect"
    assert _stem("deployment") == "deploy"
    assert _stem("readable") == "read"


def test_stem_adverbs():
    """Strip -ly."""
    assert _stem("actually") == "actual"
    assert _stem("quickly") == "quick"


def test_tokenize_stems_english_words():
    """Tokenizer applies stemming to pure-alphabetic tokens."""
    tokens = _tokenize("orphaned decisions fixing")
    assert "orphan" in tokens
    assert "decision" in tokens
    assert "fix" in tokens


def test_tokenize_skips_stemming_code_tokens():
    """Tokens with dots, underscores, or digits are not stemmed."""
    tokens = _tokenize("api_index.py session_002 fixing")
    assert "api_index.py" in tokens     # code token: kept as-is
    assert "session_002" in tokens      # code token: kept as-is
    assert "fix" in tokens              # English word: stemmed


def test_stemming_enables_morphological_matching():
    """BM25 index + query both stem, so 'orphan' matches 'orphaned'."""
    bm25 = BM25()
    # Index a document containing 'orphaned'
    bm25.index([
        _tokenize("the keychain items were orphaned by the service name change"),
        _tokenize("unrelated document about swift testing"),
    ])
    # Query for 'orphan' (base form) — should match because both stem to 'orphan'
    scores = bm25.score(_tokenize("orphan"))
    assert scores[0] > 0   # orphaned → orphan matches query orphan → orphan
    assert scores[1] == 0   # unrelated doc
