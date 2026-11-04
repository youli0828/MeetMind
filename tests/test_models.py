import pytest
from pydantic import ValidationError

from meetmind.models import Minutes, Transcript


@pytest.mark.parametrize(
    "change",
    [
        {"start": -1.0},
        {"end": -1.0},
        {"end": float("nan")},
        {"start": 6.0},
        {"text": " "},
        {"id": "0"},
        {"unexpected": True},
    ],
)
def test_bad_segments_rejected(transcript, change):
    data = transcript.model_dump()
    data["segments"][0].update(change)
    with pytest.raises(ValidationError):
        Transcript.model_validate(data)


def test_empty_duplicate_and_unordered_segments_rejected(transcript):
    data = transcript.model_dump()
    with pytest.raises(ValidationError):
        Transcript(language=None, segments=[])
    data["segments"][1]["id"] = 0
    with pytest.raises(ValidationError):
        Transcript.model_validate(data)
    data = transcript.model_dump()
    data["segments"].reverse()
    with pytest.raises(ValidationError):
        Transcript.model_validate(data)


def test_nullable_owner_due_are_required_and_preserved(minutes):
    data = minutes.model_dump()
    assert data["action_items"][1]["owner"] is None
    assert data["action_items"][1]["due"] is None
    del data["action_items"][1]["owner"]
    with pytest.raises(ValidationError):
        Minutes.model_validate(data)


def test_unknown_sources_rejected(minutes, transcript):
    minutes.decisions[0].source_segment_ids = [999]
    with pytest.raises(ValueError, match="unknown segment"):
        minutes.validate_sources(transcript)
