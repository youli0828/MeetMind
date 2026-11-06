from unittest.mock import Mock

import pytest
from test_llm import OpenAICompletion

from meetmind.config import Settings
from meetmind.models import Decision, Minutes, Segment, Transcript
from meetmind.pipeline import process_transcript
from meetmind.summarize.chunking import chunk_transcript, deduplicate_minutes
from meetmind.summarize.openai import OpenAISummarizer


@pytest.fixture
def oversized():
    return Transcript(
        language="en",
        segments=[
            Segment(id=i, start=float(i), end=float(i + 1), text="meeting " * 12, speaker=None)
            for i in range(12)
        ],
    )


def test_chunking_is_bounded_chronological_and_overlaps(oversized):
    limit = len(Transcript(language="en", segments=oversized.segments[:4]).model_dump_json())
    chunks = chunk_transcript(oversized, limit, 2)
    assert len(chunks) > 1
    assert all(len(chunk.model_dump_json()) <= limit for chunk in chunks)
    assert {s.id for chunk in chunks for s in chunk.segments} == set(range(12))
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.segments[0].id > previous.segments[0].id
        assert current.segments[0].id <= previous.segments[-1].id
        assert current.segments[-1].id > previous.segments[-1].id
    assert chunks[0].segments[-2:] == chunks[1].segments[:2]


def test_oversized_single_segment_keeps_text_and_source_id():
    transcript = Transcript(
        language="en", segments=[Segment(id=42, start=0.0, end=1.0, text="a" * 1500, speaker=None)]
    )
    chunks = chunk_transcript(transcript, 300, 2)
    assert "".join(c.segments[0].text for c in chunks) == transcript.segments[0].text
    assert all(c.segments[0].id == 42 for c in chunks)
    assert all(len(c.model_dump_json()) <= 300 for c in chunks)


def decision_minutes(ids):
    return Minutes(
        title="Meeting",
        summary="Summary",
        topics=[],
        action_items=[],
        decisions=[Decision(text="Ship the feature", source_segment_ids=ids)],
    )


def test_chunk_retry_merge_and_dedupe_preserve_citations(tmp_path, oversized):
    settings = Settings(max_transcript_chars=800, summarize_chunk_chars=650, max_attempts=2)
    chunks = chunk_transcript(oversized, 650, 2)
    responses = ["invalid"]
    for chunk in chunks:
        responses.append(decision_minutes([s.id for s in chunk.segments]).model_dump_json())
    merged = decision_minutes(list(range(6)))
    merged.decisions.append(
        Decision(text="Ship the feature", source_segment_ids=list(range(6, 12)))
    )
    responses.append(merged.model_dump_json())
    client = Mock()
    client.chat.completions.create.side_effect = [OpenAICompletion(r) for r in responses]
    output = tmp_path / "meeting"
    summarizer = OpenAISummarizer(settings, client=client, artifacts_dir=output / "chunks")
    result = process_transcript(oversized, output, summarizer)
    assert len(result.decisions) == 1
    assert result.decisions[0].source_segment_ids == list(range(12))
    assert client.chat.completions.create.call_count == len(chunks) + 2
    assert len(list((output / "chunks").glob("*.minutes.json"))) == len(chunks)
    merge_payload = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    combined = Minutes.model_validate_json(merge_payload)
    assert len(combined.decisions) == 1
    assert combined.decisions[0].source_segment_ids == list(range(12))
    assert (output / "minutes.json").exists()


def test_merge_failure_retains_chunks_in_default_api_location(tmp_path, oversized):
    settings = Settings(max_transcript_chars=650, max_attempts=2, data_dir=str(tmp_path))
    chunks = chunk_transcript(oversized, 650, 2)
    client = Mock()
    client.chat.completions.create.side_effect = [
        *[
            OpenAICompletion(decision_minutes([s.id for s in c.segments]).model_dump_json())
            for c in chunks
        ],
        OpenAICompletion("private-invalid"),
        OpenAICompletion("private-invalid"),
    ]
    with pytest.raises(ValueError, match="per-chunk minutes retained") as error:
        OpenAISummarizer(settings, client=client).summarize(oversized)
    assert "private-invalid" not in str(error.value)
    files = list((tmp_path / "summarize").glob("*/*.minutes.json"))
    assert len(files) == len(chunks)
    for path, chunk in zip(sorted(files), chunks, strict=True):
        minutes = Minutes.model_validate_json(path.read_text("utf-8"))
        minutes.validate_sources(chunk)


def test_failed_chunk_retries_only_itself_and_keeps_earlier_files(tmp_path, oversized):
    client = Mock()
    chunks = chunk_transcript(oversized, 650, 2)
    client.chat.completions.create.side_effect = [
        OpenAICompletion(decision_minutes([s.id for s in chunks[0].segments]).model_dump_json()),
        OpenAICompletion("bad"),
        OpenAICompletion("bad"),
    ]
    with pytest.raises(ValueError, match="failed validation"):
        OpenAISummarizer(
            Settings(max_transcript_chars=650, max_attempts=2),
            client=client,
            artifacts_dir=tmp_path / "chunks",
        ).summarize(oversized)
    assert client.chat.completions.create.call_count == 3
    assert (tmp_path / "chunks/chunk-0000.minutes.json").exists()
    assert not (tmp_path / "chunks/chunk-0001.minutes.json").exists()


def test_dedupe_all_collections(minutes):
    for field in ("topics", "decisions", "action_items"):
        item = getattr(minutes, field)[0].model_copy(deep=True)
        item.source_segment_ids = [99]
        getattr(minutes, field).append(item)
    result = deduplicate_minutes(minutes)
    for field in ("topics", "decisions", "action_items"):
        assert len(getattr(result, field)) == len(getattr(minutes, field)) - 1
        assert 99 in getattr(result, field)[0].source_segment_ids


def test_chunk_config_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SUMMARIZE_CHUNK_CHARS", "1000")
    monkeypatch.setenv("SUMMARIZE_CHUNK_OVERLAP_SEGS", "1")
    result = Settings.load(tmp_path / ".env")
    assert result.summarize_chunk_chars == 1000
    assert result.summarize_chunk_overlap_segs == 1


@pytest.mark.parametrize("bad_ids", [[0], list(range(12))])
def test_merge_rejects_lost_or_added_citations(tmp_path, oversized, bad_ids):
    chunks = chunk_transcript(oversized, 650, 2)
    client = Mock()
    client.chat.completions.create.side_effect = [
        *[
            OpenAICompletion(
                decision_minutes([s.id for s in c.segments if s.id != 11]).model_dump_json()
            )
            for c in chunks
        ],
        OpenAICompletion(decision_minutes(bad_ids).model_dump_json()),
    ]
    with pytest.raises(ValueError, match="Minutes merge failed"):
        OpenAISummarizer(
            Settings(max_transcript_chars=650, max_attempts=1),
            client=client,
            artifacts_dir=tmp_path / "chunks",
        ).summarize(oversized)


def test_api_long_meeting_uses_chunked_summarizer(tmp_path, oversized):
    from fastapi.testclient import TestClient

    from meetmind.api import create_app, get_service
    from meetmind.service import Service
    from meetmind.storage import Store

    settings = Settings(max_transcript_chars=650, data_dir=str(tmp_path))
    chunks = chunk_transcript(oversized, 650, 2)
    client = Mock()
    client.chat.completions.create.side_effect = [
        *[
            OpenAICompletion(decision_minutes([s.id for s in c.segments]).model_dump_json())
            for c in chunks
        ],
        OpenAICompletion(decision_minutes(list(range(12))).model_dump_json()),
    ]
    service = Service(Store(tmp_path), Mock(), Mock(), OpenAISummarizer(settings, client=client))
    app = create_app()
    app.dependency_overrides[get_service] = lambda: service
    with TestClient(app) as api:
        response = api.post("/meetings", json={"transcript": oversized.model_dump()})
    assert response.status_code == 201
    meeting = response.json()
    assert meeting["minutes"]["decisions"][0]["source_segment_ids"] == list(range(12))
    assert service.store.get(meeting["id"]).transcript == oversized
    assert len(list((tmp_path / "summarize").glob("*/*.minutes.json"))) == len(chunks)
