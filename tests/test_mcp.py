import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from openai import OpenAIError

from meetmind.api import create_app
from meetmind.mcp_server import create_server, main
from meetmind.retrieval import chunks
from meetmind.storage import Meeting, Store
from meetmind.tools.actions import Tools


@pytest.fixture
def mcp_setup(tmp_path, transcript, minutes):
    store = Store(tmp_path)
    meeting = Meeting(id="test-meeting", transcript=transcript, minutes=minutes)
    store.add(meeting)
    retriever = Mock()
    retriever.search.return_value = chunks(meeting)
    factory = Mock(return_value=SimpleNamespace(tools=Tools(store, retriever)))
    return create_server(service_factory=factory), store, retriever, factory


def invoke(server, name, arguments):
    result = asyncio.run(server.call_tool(name, arguments))
    # FastMCP returns (content, structured data) for structured tool outputs.
    content = result[0] if isinstance(result, tuple) else result
    return json.loads(content[0].text)


def test_registration_is_lazy_and_lists_exact_tools(mcp_setup):
    server, _, _, factory = mcp_setup
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == {
        "search_meetings",
        "get_meeting_summary",
        "create_todo",
        "create_calendar_event",
        "export_markdown",
    }
    assert next(t for t in tools if t.name == "search_meetings").inputSchema["required"] == [
        "query"
    ]
    factory.assert_not_called()


def test_search_delegates_and_preserves_timestamped_sources(mcp_setup, transcript):
    server, _, retriever, _ = mcp_setup
    result = invoke(server, "search_meetings", {"query": "预算", "k": 2})
    retriever.search.assert_called_once_with("预算", 2)
    source = result["sources"][0]
    assert source["meeting_id"] == "test-meeting"
    assert source["start"] == transcript.segments[0].start
    assert source["end"] == transcript.segments[-1].end
    assert transcript.segments[0].text in source["text"]


def test_summary_reads_existing_minutes(mcp_setup, minutes):
    server, _, _, _ = mcp_setup
    assert invoke(server, "get_meeting_summary", {"meeting_id": "test-meeting"}) == (
        minutes.model_dump()
    )


def test_todo_reuses_idempotent_action(mcp_setup):
    server, store, _, _ = mcp_setup
    args = {"meeting_id": "test-meeting", "action_index": 0}
    first = invoke(server, "create_todo", args)
    assert invoke(server, "create_todo", args) == first
    assert store.todos() == [first]


@pytest.mark.parametrize("name", ["export_markdown", "create_calendar_event"])
def test_artifacts_use_existing_export_tools(mcp_setup, name):
    server, store, _, _ = mcp_setup
    args = {"meeting_id": "test-meeting"}
    if name == "create_calendar_event":
        args.update(
            title="Review", start="2026-11-05T09:00:00+08:00", end="2026-11-05T10:00:00+08:00"
        )
    result = invoke(server, name, args)
    filename, content = store.read_artifact(result["url"].rsplit("/", 1)[1])
    assert filename.endswith(".md" if name == "export_markdown" else ".ics")
    assert (b"#segment-" if name == "export_markdown" else b"BEGIN:VCALENDAR") in content


@pytest.mark.parametrize(
    "name,args",
    [
        ("search_meetings", {"query": "budget", "k": 11}),
        ("get_meeting_summary", {"meeting_id": "missing"}),
        ("create_todo", {"meeting_id": "test-meeting", "action_index": -1}),
        (
            "create_calendar_event",
            {"meeting_id": "test-meeting", "title": "x", "start": "tomorrow", "end": "later"},
        ),
    ],
)
def test_invalid_calls_do_not_write(mcp_setup, name, args):
    server, store, _, _ = mcp_setup
    with pytest.raises(Exception, match="Meeting tool failed"):
        invoke(server, name, args)
    assert store.todos() == []


def test_upstream_error_does_not_expose_details(mcp_setup):
    server, _, retriever, _ = mcp_setup
    retriever.search.side_effect = OpenAIError("private upstream detail")
    with pytest.raises(Exception) as error:
        invoke(server, "search_meetings", {"query": "x"})
    assert "private upstream detail" not in str(error.value)


@pytest.mark.parametrize("transport", ["stdio", "sse"])
def test_cli_transport(monkeypatch, transport):
    factory = Mock()
    monkeypatch.setattr("meetmind.mcp_server.create_server", factory)
    args = [] if transport == "stdio" else ["--transport", "sse", "--port", "9001"]
    assert main(args) == 0
    factory.assert_called_once_with(host="127.0.0.1", port=8001 if transport == "stdio" else 9001)
    factory.return_value.run.assert_called_once_with(transport=transport)


def test_health_without_model_or_storage(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("meetmind.api.get_service", Mock(side_effect=AssertionError("not needed")))
    with TestClient(create_app()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert list(tmp_path.iterdir()) == []
