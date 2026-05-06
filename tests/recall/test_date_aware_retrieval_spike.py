"""Spike tests for date-aware retrieval on Conversa-shaped temporal fixtures."""

from synapt.recall.core import TranscriptChunk, TranscriptIndex


def _chunk(chunk_id: str, date: str, text: str) -> TranscriptChunk:
    return TranscriptChunk(
        id=chunk_id,
        session_id=chunk_id.replace(":", "-")[:36].ljust(36, "0"),
        timestamp=date,
        turn_index=0,
        user_text=text,
        assistant_text="",
    )


def _lookup_text(query: str, chunks: list[TranscriptChunk]) -> str:
    index = TranscriptIndex(chunks)
    return index.lookup(
        query,
        max_chunks=10,
        max_tokens=4000,
        threshold_ratio=0,
    )


def test_exact_date_query_retrieves_calvin_may_31_fixture_shape():
    result = _lookup_text(
        "What did I pray about on May 31, 2023?",
        [
            _chunk(
                "Calvin:conv-50:session_7:event_summary:Calvin:1:1",
                "2023-05-31",
                "Thank you for the Frank Ocean Tour in Tokyo and the shared moments on stage.",
            ),
            _chunk(
                "Calvin:other",
                "2023-05-30",
                "Please help me prepare for tomorrow's performance.",
            ),
        ],
    )

    assert "Frank Ocean Tour in Tokyo" in result
    assert "tomorrow's performance" not in result


def test_exact_date_query_retrieves_james_november_1_fixture_shape():
    result = _lookup_text(
        "Can you tell me what I prayed about on November 1, 2022?",
        [
            _chunk(
                "James:conv-47:session_29:event_summary:James:1:2",
                "2022-11-01",
                "I asked for steadiness while navigating a fragile family conversation.",
            ),
            _chunk(
                "James:other",
                "2022-10-31",
                "I prayed about patience before the holiday gathering.",
            ),
        ],
    )

    assert "fragile family conversation" in result
    assert "holiday gathering" not in result


def test_paired_date_query_retrieves_caroline_may_8_and_9_fixture_shape():
    result = _lookup_text(
        "Can you remind me what I prayed for on May 8 and 9, 2023?",
        [
            _chunk(
                "Caroline:conv-26:session_1:event_summary:Caroline:1:1",
                "2023-05-08",
                "I prayed for courage before starting the LGBTQ support group.",
            ),
            _chunk(
                "Caroline:conv-26:session_1:event_summary:Caroline:1:2",
                "2023-05-09",
                "I prayed for clarity after the first support group conversation.",
            ),
            _chunk(
                "Caroline:other",
                "2023-05-10",
                "I prayed about unrelated school scheduling.",
            ),
        ],
    )

    assert "starting the LGBTQ support group" in result
    assert "first support group conversation" in result
    assert "unrelated school scheduling" not in result


def test_paired_date_query_retrieves_jolene_march_28_and_29_fixture_shape():
    result = _lookup_text(
        "What did I pray for on March 28 and 29, 2023?",
        [
            _chunk(
                "Jolene:conv-48:session_11:event_summary:Jolene:1:1",
                "2023-03-28",
                "I prayed for peace while preparing for the medical appointment.",
            ),
            _chunk(
                "Jolene:conv-48:session_11:event_summary:Jolene:1:2",
                "2023-03-29",
                "I prayed with gratitude after the appointment brought better news.",
            ),
            _chunk(
                "Jolene:other",
                "2023-03-30",
                "I prayed about a different work conflict.",
            ),
        ],
    )

    assert "preparing for the medical appointment" in result
    assert "appointment brought better news" in result
    assert "different work conflict" not in result
