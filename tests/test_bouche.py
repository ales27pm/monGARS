from datetime import UTC, datetime

import pytest

from monGARS.core.bouche import Bouche


def _assert_segment_payload(turn_payload: dict) -> None:
    assert "segments" in turn_payload
    for segment in turn_payload["segments"]:
        assert segment["estimated_duration"] > 0
        assert segment["pause_after"] >= 0.18


@pytest.mark.asyncio
async def test_bouche_speak_creates_structured_turn() -> None:
    bouche = Bouche()

    text = "Hello there! How are you today?"
    turn = await bouche.speak(text)

    assert turn.text == text
    assert len(turn.segments) == 2
    assert turn.segments[-1].pause_after > turn.segments[0].pause_after
    assert 0.65 <= turn.tempo <= 1.35

    payload = turn.to_payload()
    assert payload["turn_id"] == turn.turn_id
    assert datetime.fromisoformat(payload["created_at"]).tzinfo == UTC
    _assert_segment_payload(payload)


@pytest.mark.asyncio
async def test_bouche_conversation_profile_tracks_turns() -> None:
    bouche = Bouche()

    first = await bouche.speak("We can start now. I want to review goals.")
    second = await bouche.speak(
        "Great, let's dive into the agenda together. Sound good?"
    )

    assert second.turn_id != first.turn_id

    profile = bouche.conversation_profile()
    assert profile["turn_count"] == 2

    expected_tempo = (first.tempo * 0.4) + (second.tempo * 0.6)
    assert profile["tempo"] == pytest.approx(expected_tempo)

    first_avg_pause = sum(seg.pause_after for seg in first.segments) / len(
        first.segments
    )
    second_avg_pause = sum(seg.pause_after for seg in second.segments) / len(
        second.segments
    )
    assert profile["average_pause"] == pytest.approx(
        (first_avg_pause + second_avg_pause) / 2
    )
    assert profile["average_pause"] >= 0.18
