"""Channel post citations: the m_ id parse, the citation graph, cited_by,
and the search ranker seam."""

import pytest

from synapt.recall import channel as ch
from synapt.recall.channel import (
    build_citation_graph,
    channel_post,
    channel_search,
    extract_post_refs,
    set_search_ranker,
)

A, B, C = "m_aaaaaaaa", "m_bbbbbbbb", "m_cccccccc"


@pytest.fixture(autouse=True)
def _no_ranker():
    previous = set_search_ranker(None)
    yield
    set_search_ranker(previous)


def _id_of(text, project_dir):
    [hit] = channel_search(text, project_dir=project_dir)
    return hit["message_id"]


def test_parse_ids_in_order_deduplicated_with_word_before():
    refs = extract_post_refs(f"see {A}, and head {B}; again see {A} m_short")
    assert refs == [("see", A), ("head", B)]
    assert extract_post_refs(A) == [(None, A)]


def test_adjacent_ids_are_both_parsed_and_neither_is_a_word():
    assert extract_post_refs(f"{A} {B}") == [(None, A), (None, B)]


def test_adjacent_ids_are_both_cited():
    g = build_citation_graph({"dev": [(A, "x"), (B, "y"), (C, f"{A} {B}")]})
    assert g["cited_by"][("dev", A)] == 1
    assert g["cited_by"][("dev", B)] == 1


def test_three_post_fixture_cited_by(tmp_path):
    channel_post("dev", "alpha first post", "x", project_dir=tmp_path)
    a = _id_of("alpha", tmp_path)
    channel_post("dev", f"bravo cites {a}", "y", project_dir=tmp_path)
    channel_post("dev", f"charlie cites {a} twice {a}", "z", project_dir=tmp_path)
    hits = {h["body"].split()[0]: h for h in channel_search("i", project_dir=tmp_path)}
    hits["alpha"] = channel_search("alpha first", project_dir=tmp_path)[0]
    assert hits["alpha"]["cited_by"] == 2
    assert hits["bravo"]["cited_by"] == 0
    assert hits["charlie"]["cited_by"] == 0


def test_branch_a_named_channel_wins_over_own_channel():
    g = build_citation_graph({
        "dev": [(A, "x"), (B, f"head {A}")],
        "head": [(A, "y")],
    })
    assert g["edges"] == [("dev", B, "head", A)]
    assert g["cited_by"][("head", A)] == 1
    assert g["cited_by"][("dev", A)] == 0


def test_branch_a_named_channel_without_the_id_falls_through():
    g = build_citation_graph({"dev": [(A, "x"), (B, f"head {A}")], "head": []})
    assert g["edges"] == [("dev", B, "dev", A)]


def test_branch_b_own_channel():
    g = build_citation_graph({
        "dev": [(A, "x"), (B, f"see {A}")],
        "head": [(A, "y")],
    })
    assert g["edges"] == [("dev", B, "dev", A)]
    assert g["ambiguous"] == []


def test_branch_c_exactly_one_channel_in_store():
    g = build_citation_graph({"dev": [(B, f"see {A}")], "head": [(A, "y")]})
    assert g["edges"] == [("dev", B, "head", A)]
    assert g["cited_by"][("head", A)] == 1


def test_branch_d_collision_is_ambiguous_and_never_attributed():
    g = build_citation_graph({
        "dev": [(B, f"see {A}")],
        "head": [(A, "y")],
        "ops": [(A, "z")],
    })
    assert g["edges"] == []
    assert g["ambiguous"] == [("dev", B, A)]
    assert g["cited_by"][("head", A)] == 0
    assert g["cited_by"][("ops", A)] == 0


def test_branch_d_unknown_id_is_ambiguous():
    g = build_citation_graph({"dev": [(B, f"see {C}")]})
    assert g["edges"] == [] and g["ambiguous"] == [("dev", B, C)]


def test_collision_resolved_when_a_channel_is_named():
    g = build_citation_graph({
        "dev": [(B, f"#ops {A}")],
        "head": [(A, "y")],
        "ops": [(A, "z")],
    })
    assert g["edges"] == [("dev", B, "ops", A)]
    assert g["ambiguous"] == []


def _three(tmp_path):
    for word in ("one", "two", "three"):
        channel_post("dev", f"needle {word}", "x", project_dir=tmp_path)
    return channel_search("needle", project_dir=tmp_path)


def test_ranker_reorder_and_drop_honoured(tmp_path):
    default = _three(tmp_path)
    set_search_ranker(lambda q, rs: list(reversed(rs))[:2])
    got = channel_search("needle", project_dir=tmp_path)
    assert got == list(reversed(default))[:2]


def test_ranker_that_adds_is_refused(tmp_path):
    default = _three(tmp_path)
    extra = dict(default[0], message_id="m_00000000")
    set_search_ranker(lambda q, rs: rs + [extra])
    assert channel_search("needle", project_dir=tmp_path) == default


def test_ranker_that_repeats_is_refused(tmp_path):
    default = _three(tmp_path)
    set_search_ranker(lambda q, rs: [rs[0], rs[0]])
    assert channel_search("needle", project_dir=tmp_path) == default


def test_ranker_that_raises_gives_default_order(tmp_path):
    default = _three(tmp_path)

    def boom(q, rs):
        raise RuntimeError("ranker failed")

    set_search_ranker(boom)
    assert channel_search("needle", project_dir=tmp_path) == default


def test_ranker_unhashable_channel_gives_default_order(tmp_path):
    default = _three(tmp_path)
    set_search_ranker(lambda q, rs: [dict(rs[0], channel=["dev"])])
    assert channel_search("needle", project_dir=tmp_path) == default


def test_ranker_unhashable_message_id_gives_default_order(tmp_path):
    default = _three(tmp_path)
    set_search_ranker(lambda q, rs: [dict(rs[0], message_id={"id": 1})])
    assert channel_search("needle", project_dir=tmp_path) == default


def test_ranker_runs_before_the_cut(tmp_path):
    default = _three(tmp_path)
    set_search_ranker(lambda q, rs: list(reversed(rs)))
    assert channel_search("needle", max_results=1, project_dir=tmp_path) == [default[-1]]


def test_no_ranker_ships():
    assert ch._search_ranker is None
