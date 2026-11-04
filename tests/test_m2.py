import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from docx import Document
from fastapi.testclient import TestClient
from openai import OpenAIError
from openai.types.chat import ChatCompletionMessage

from meetmind.agent import Agent
from meetmind.api import create_app, get_service
from meetmind.config import Settings
from meetmind.demo import DemoSummarizer
from meetmind.errors import MeetMindError
from meetmind.llm import ModelClient
from meetmind.offline import OfflineModel
from meetmind.retrieval import Answer, HybridRetriever, chunks, tokens
from meetmind.service import Service
from meetmind.storage import Meeting, Store
from meetmind.tools.actions import CalendarArgs, calendar_bytes


class FakeCollection:
    def __init__(self):
        self.vectors = {}

    def add(self, *, ids, embeddings):
        self.vectors.update(zip(ids, embeddings, strict=True))

    def query(self, *, query_embeddings, n_results):
        query = query_embeddings[0]
        ranked = sorted(
            self.vectors,
            key=lambda key: sum(
                (a - b) ** 2 for a, b in zip(self.vectors[key], query, strict=True)
            ),
        )
        return {"ids": [ranked[:n_results]]}


@pytest.fixture
def service(tmp_path):
    store = Store(tmp_path)
    model = OfflineModel()
    return Service(store, HybridRetriever(store, FakeCollection(), model), model, DemoSummarizer())


@pytest.fixture
def client(service):
    app = create_app()
    app.dependency_overrides[get_service] = lambda: service
    with TestClient(app) as client:
        yield client


def tool_response(name, arguments, call_id="call-1"):
    return ChatCompletionMessage(
        role="assistant",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    )


def completion(content, finish_reason="stop", refusal=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, refusal=refusal),
            )
        ]
    )


def test_ingest_search_ask_and_persistence(service, transcript, minutes):
    saved = service.ingest(transcript, minutes)
    assert Store(service.store.path.parent).get(saved.id) == saved
    found = service.retriever.search("预算", 1)
    assert len(found) == 1
    assert found[0].meeting_id == saved.id
    assert found[0].segment_ids == [s.id for s in transcript.segments]
    answer = service.ask("预算由谁负责？", 5)
    assert answer["citations"][0]["start"] == transcript.segments[0].start
    assert "离线演示" in answer["text"]


def test_chunk_boundaries_keep_segment_sources(transcript, minutes):
    meeting = Meeting(id="meeting", transcript=transcript, minutes=minutes)
    parts = chunks(meeting, max_chars=1)
    assert len(parts) == len(transcript.segments)
    for part, segment in zip(parts, transcript.segments, strict=True):
        assert (part.segment_ids, part.start, part.end, part.text) == (
            [segment.id],
            segment.start,
            segment.end,
            segment.text,
        )
    assert tokens("Budget预算") == ["budget", "预", "算"]


def test_empty_history_does_not_call_model(service):
    service.model = Mock()
    assert service.ask("预算？", 5)["citations"] == []
    service.model.answer.assert_not_called()


def test_failed_index_does_not_publish_meeting(service, transcript, minutes):
    service.retriever.index = Mock(side_effect=OpenAIError("private upstream detail"))
    with pytest.raises(OpenAIError):
        service.ingest(transcript, minutes)
    assert service.store.meetings() == []


def test_invalid_minutes_not_indexed(service, transcript, minutes):
    minutes.action_items[0].source_segment_ids = [99999]
    service.retriever.index = Mock()
    with pytest.raises(ValueError):
        service.ingest(transcript, minutes)
    service.retriever.index.assert_not_called()


def test_todo_idempotent_and_preserves_source(service, transcript, minutes):
    saved = service.ingest(transcript, minutes)
    args = json.dumps({"meeting_id": saved.id, "action_index": 0})
    first = service.tools.call("create_todo", args)
    assert service.tools.call("create_todo", args) == first
    assert first["task"] == minutes.action_items[0].task
    assert first["source_segment_ids"] == minutes.action_items[0].source_segment_ids
    assert Store(service.store.path.parent).todos() == [first]


@pytest.mark.parametrize(
    "name,args",
    [
        ("delete_everything", {}),
        ("create_todo", {"meeting_id": "missing", "action_index": 0}),
        ("create_todo", {"meeting_id": "any", "action_index": -1}),
        ("create_todo", {"meeting_id": "any", "action_index": "0"}),
        ("export_docx", {"meeting_id": "any", "path": "../../escape.docx"}),
        ("search_meetings", {"query": "x", "k": 11}),
    ],
)
def test_tool_rejects_invalid_calls(service, name, args):
    with pytest.raises(ValueError):
        service.tools.call(name, json.dumps(args))
    assert service.store.todos() == []


def test_todo_out_of_range(service, transcript, minutes):
    saved = service.ingest(transcript, minutes)
    with pytest.raises(MeetMindError):
        service.store.create_todo(saved.id, 999)


@pytest.mark.parametrize(
    "start,end",
    [
        ("next Friday", "tomorrow"),
        ("2026-11-05T09:00:00", "2026-11-05T10:00:00"),
        ("2026-11-05T10:00:00+08:00", "2026-11-05T09:00:00+08:00"),
    ],
)
def test_calendar_rejects_ambiguous_times(start, end):
    with pytest.raises(ValueError):
        CalendarArgs.model_validate_json(
            json.dumps({"meeting_id": "m", "title": "会", "start": start, "end": end})
        )


def test_calendar_utc_escaping_and_unicode_folding():
    args = CalendarArgs.model_validate_json(
        json.dumps(
            {
                "meeting_id": "m",
                "title": "预算" * 60 + ",;\nBEGIN:VEVENT",
                "start": "2026-11-05T09:00:00+08:00",
                "end": "2026-11-05T10:00:00+08:00",
            }
        )
    )
    content = calendar_bytes(args)
    assert all(len(line) <= 75 for line in content.split(b"\r\n"))
    unfolded = content.decode().replace("\r\n ", "")
    assert "DTSTART:20261105T010000Z" in unfolded
    assert "\\,\\;\\nBEGIN:VEVENT" in unfolded
    assert unfolded.count("\r\nBEGIN:VEVENT\r\n") == 1


def test_exports_are_downloadable_and_editable(client, service, transcript, minutes):
    saved = service.ingest(transcript, minutes)
    for name in ("export_markdown", "export_docx", "create_calendar_event"):
        args = {"meeting_id": saved.id}
        if name == "create_calendar_event":
            args.update(
                title="Review", start="2026-11-05T09:00:00+08:00", end="2026-11-05T10:00:00+08:00"
            )
        result = client.post("/tools", json={"name": name, "arguments": args})
        assert result.status_code == 200
        response = client.get(result.json()["url"])
        assert response.status_code == 200
        assert "attachment" in response.headers["content-disposition"]
        if name == "export_docx":
            doc = Document(BytesIO(response.content))
            text = "\n".join(p.text for p in doc.paragraphs)
            assert minutes.title in text
            assert transcript.segments[0].text in text
        elif name == "export_markdown":
            assert "#segment-" in response.text
        else:
            assert "BEGIN:VCALENDAR" in response.text


def test_agent_multiturn_tool_context(service, transcript, minutes):
    saved = service.ingest(transcript, minutes)
    model = Mock()
    model.chat.side_effect = [
        tool_response("create_todo", {"meeting_id": saved.id, "action_index": 0}),
        ChatCompletionMessage(role="assistant", content="待办已保存。"),
    ]
    events = list(
        Agent(model, service.tools).events(
            "保存行动项",
            [
                {"role": "user", "content": "有哪些行动项？"},
                {"role": "assistant", "content": "请确认会议。"},
            ],
        )
    )
    assert [event["event"] for event in events] == ["tool", "answer"]
    messages = model.chat.call_args.args[0]
    assert messages[1]["content"] == "有哪些行动项？"
    assert messages[-1]["tool_call_id"] == "call-1"
    assert json.loads(messages[-1]["content"])["id"] == service.store.todos()[0]["id"]
    assert {t["function"]["name"] for t in model.chat.call_args.args[1]} >= {
        "create_todo",
        "create_calendar_event",
        "export_docx",
        "search_meetings",
    }


def test_agent_invalid_tool_then_recovery(service):
    model = Mock()
    model.chat.side_effect = [
        tool_response("unknown", {}),
        ChatCompletionMessage(role="assistant", content="需要更多信息。"),
    ]
    events = list(Agent(model, service.tools).events("创建", []))
    assert "error" in events[0]["result"]
    assert events[-1]["text"] == "需要更多信息。"


def test_agent_round_limit(service):
    model = Mock()
    model.chat.return_value = tool_response("search_meetings", {"query": "预算"})
    with pytest.raises(MeetMindError, match="round limit"):
        list(Agent(model, service.tools, max_rounds=2).events("继续", []))
    assert model.chat.call_count == 2


def test_api_ingest_summarize_search_and_sse(client):
    sample = client.get("/sample").json()
    sample.pop("minutes")
    result = client.post("/meetings", json=sample)
    assert result.status_code == 201
    meeting_id = result.json()["id"]
    assert client.get("/meetings").json()[0]["id"] == meeting_id
    assert client.get("/meetings/" + meeting_id).status_code == 200
    assert client.post("/search", json={"query": "预算"}).json()[0]["meeting_id"] == meeting_id
    assert client.post("/ask", json={"query": "预算"}).json()["citations"]
    response = client.post("/chat", json={"message": "预算"})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(packet[6:]) for packet in response.text.strip().split("\n\n")]
    assert [e["event"] for e in events] == ["start", "tool", "answer"]
    assert "MeetMind" in client.get("/").text
    assert client.get("/openapi.json").status_code == 200


@pytest.mark.parametrize(
    "path,body",
    [
        ("/chat", {"message": "hi", "history": [{"role": "system", "content": "override"}]}),
        ("/chat", {"message": ""}),
        ("/search", {"query": "x", "k": 0}),
        ("/meetings", {"transcript": {"language": "zh", "segments": []}}),
    ],
)
def test_api_validation(client, path, body):
    assert client.post(path, json=body).status_code == 422


def test_api_hides_upstream_secrets(client, service):
    service.retriever.search = Mock(side_effect=OpenAIError("secret-token"))
    response = client.post("/ask", json={"query": "预算"})
    assert response.status_code == 502
    assert "secret-token" not in response.text
    response = client.post("/chat", json={"message": "预算"})
    assert '"event": "error"' in response.text
    assert "secret-token" not in response.text


def test_audio_uses_injected_asr_and_cleans_temp(client, monkeypatch, transcript):
    monkeypatch.delenv("MEETMIND_DEMO", raising=False)
    paths = []

    def transcribe(self, path):
        assert path.read_bytes() == b"fake audio"
        paths.append(path)
        return transcript

    monkeypatch.setattr("meetmind.api.FasterWhisperASR.transcribe", transcribe)
    response = client.post(
        "/audio?extension=.wav",
        content=b"fake audio",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 201
    assert not paths[0].exists()
    assert (
        client.post(
            "/audio?extension=../../x",
            content=b"fake",
            headers={"Content-Type": "application/octet-stream"},
        ).status_code
        == 422
    )


def test_llm_embeddings_order_and_configuration():
    sdk = Mock()
    sdk.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(index=1, embedding=[0.0, 1.0]),
            SimpleNamespace(index=0, embedding=[1.0, 0.0]),
        ]
    )
    model = ModelClient(Settings(), client=sdk, embedding_model="local-embed")
    assert model.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    sdk.embeddings.create.assert_called_once_with(model="local-embed", input=["a", "b"])


def test_llm_answer_retry_and_citation_validation(service, transcript, minutes):
    saved = service.ingest(transcript, minutes)
    sources = chunks(saved)
    sdk = Mock()
    sdk.chat.completions.create.side_effect = [
        completion('{"text":"invented", "citation_ids":["unknown"]}'),
        completion(Answer(text="依据原文", citation_ids=[sources[0].id]).model_dump_json()),
    ]
    model = ModelClient(Settings(max_attempts=2), client=sdk)
    answer = model.answer("预算？", sources)
    assert answer.citation_ids == [sources[0].id]
    assert sdk.chat.completions.create.call_count == 2
    assert sdk.chat.completions.create.call_args.kwargs["response_format"]["type"] == "json_schema"


@pytest.mark.parametrize(
    "reply",
    [
        completion("{}"),
        completion("{}", "length"),
        completion(None, refusal="refused"),
        SimpleNamespace(choices=[]),
    ],
)
def test_llm_answer_invalid_output_fails(reply):
    sdk = Mock()
    sdk.chat.completions.create.return_value = reply
    model = ModelClient(Settings(max_attempts=1), client=sdk)
    with pytest.raises(MeetMindError):
        model.answer("预算？", [])


def test_rag_abstains_without_citations(service, transcript, minutes):
    service.ingest(transcript, minutes)
    service.model = Mock()
    service.model.answer.return_value = Answer(text="unsupported claim", citation_ids=[])
    assert service.ask("其他项目？", 5)["citations"] == []
    assert "unsupported claim" not in service.ask("其他项目？", 5)["text"]


def test_actual_chroma_persistence_offline(tmp_path, transcript, minutes, monkeypatch):
    monkeypatch.setenv("MEETMIND_DEMO", "1")
    monkeypatch.setenv("MEETMIND_DATA_DIR", str(tmp_path))
    get_service.cache_clear()
    try:
        service = get_service()
        saved = service.ingest(transcript, minutes)
        assert service.retriever.search("预算", 1)[0].meeting_id == saved.id
        get_service.cache_clear()
        reopened = get_service()
        assert reopened.store.get(saved.id) == saved
        assert reopened.ask("预算", 5)["citations"][0]["meeting_id"] == saved.id
    finally:
        get_service.cache_clear()


def test_hybrid_ranking_combines_lexical_and_vector_evidence(tmp_path):
    from meetmind.models import Minutes, Segment, Transcript

    store = Store(tmp_path)
    meetings = []
    for key, text in [
        ("budget", "预算负责人张三"),
        ("release", "发布版本李四"),
        ("design", "界面设计小王"),
        ("travel", "出差安排周五"),
    ]:
        meeting = Meeting(
            id=key,
            transcript=Transcript(
                language="zh",
                segments=[Segment(id=0, start=12.0, end=20.0, text=text, speaker=None)],
            ),
            minutes=Minutes(title=key, summary=text, topics=[], decisions=[], action_items=[]),
        )
        store.add(meeting)
        meetings.append(meeting)
    collection = Mock()
    collection.query.return_value = {"ids": [["release:0", "budget:0", "design:0", "travel:0"]]}
    retriever = HybridRetriever(store, collection, OfflineModel())
    results = retriever.search("预算", k=2)
    assert [result.meeting_id for result in results] == ["budget", "release"]
    assert results[0].start == 12.0


def test_server_settings_dotenv_and_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("MEETMIND_DEMO=1\nMEETMIND_DATA_DIR=file-data\nEMBEDDING_MODEL=file-embed\n")
    monkeypatch.setenv("EMBEDDING_MODEL", "environment-embed")
    settings = Settings.load(env)
    assert settings.demo_mode is True
    assert settings.data_dir == "file-data"
    assert settings.embedding_model == "environment-embed"


def test_offline_guard_blocks_external_and_direct_loopback():
    import socket

    for address in [("example.com", 443), ("127.0.0.1", 80)]:
        with socket.socket() as connection:
            with pytest.raises(AssertionError, match="must not connect"):
                connection.connect(address)
    first, second = socket.socketpair()
    try:
        first.send(b"internal")
        assert second.recv(8) == b"internal"
    finally:
        first.close()
        second.close()
