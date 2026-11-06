import json
from unittest.mock import Mock

import httpx
import pytest
from openai import OpenAI

from meetmind.config import Settings
from meetmind.summarize.openai import OpenAISummarizer


def completion(content, *, finish="stop", refusal=None):
    return {
        "id": "test",
        "object": "chat.completion",
        "created": 0,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": content, "refusal": refusal},
            }
        ],
    }


def test_real_sdk_request_uses_configured_endpoint_and_schema(transcript, minutes):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=completion(minutes.model_dump_json()))

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        with OpenAI(
            api_key="test-only", base_url="https://example.invalid/v1", http_client=http
        ) as client:
            result = OpenAISummarizer(Settings(model="configured-gpt"), client=client).summarize(
                transcript
            )
    assert result == minutes
    assert str(requests[0].url) == "https://example.invalid/v1/chat/completions"
    body = json.loads(requests[0].content)
    assert body["model"] == "configured-gpt"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert json.loads(body["messages"][1]["content"]) == transcript.model_dump()
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert "owner" in schema["$defs"]["ActionItem"]["required"]


@pytest.mark.parametrize("bad", ["not json", '{"summary": "only"}', "unknown-source"])
def test_invalid_output_retried(bad, transcript, minutes):
    if bad == "unknown-source":
        altered = minutes.model_dump()
        altered["decisions"][0]["source_segment_ids"] = [999]
        bad = json.dumps(altered)
    client = Mock()
    client.chat.completions.create.side_effect = [
        OpenAICompletion(bad),
        OpenAICompletion(minutes.model_dump_json()),
    ]
    result = OpenAISummarizer(Settings(), client=client).summarize(transcript)
    assert result == minutes
    assert client.chat.completions.create.call_count == 2


def OpenAICompletion(content, **kwargs):
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate(completion(content, **kwargs))


def test_retries_exhausted_without_raw_response(transcript):
    client = Mock()
    client.chat.completions.create.return_value = OpenAICompletion("private invalid content")
    with pytest.raises(ValueError, match="failed validation after") as exc:
        OpenAISummarizer(Settings(max_attempts=2), client=client).summarize(transcript)
    assert client.chat.completions.create.call_count == 2
    assert "private" not in str(exc.value)


@pytest.mark.parametrize("mode", ["json_object", "text"])
def test_compatible_modes(mode, transcript, minutes):
    client = Mock()
    client.chat.completions.create.return_value = OpenAICompletion(minutes.model_dump_json())
    OpenAISummarizer(Settings(response_format=mode), client=client).summarize(transcript)
    options = client.chat.completions.create.call_args.kwargs
    if mode == "text":
        assert "response_format" not in options
    else:
        assert options["response_format"] == {"type": "json_object"}


def test_refusal_is_not_retried(transcript):
    client = Mock()
    client.chat.completions.create.return_value = OpenAICompletion(None, refusal="no")
    with pytest.raises(ValueError, match="refused"):
        OpenAISummarizer(Settings(), client=client).summarize(transcript)
    assert client.chat.completions.create.call_count == 1


def test_truncation_retried_even_when_json_valid(transcript, minutes):
    client = Mock()
    client.chat.completions.create.side_effect = [
        OpenAICompletion(minutes.model_dump_json(), finish="length"),
        OpenAICompletion(minutes.model_dump_json()),
    ]
    assert OpenAISummarizer(Settings(), client=client).summarize(transcript) == minutes
    assert client.chat.completions.create.call_count == 2


def test_impossibly_small_chunk_limit_rejected_before_request(transcript):
    client = Mock()
    with pytest.raises(ValueError, match="too small for transcript metadata"):
        OpenAISummarizer(Settings(max_transcript_chars=1), client=client).summarize(transcript)
    client.chat.completions.create.assert_not_called()


def test_api_key_required():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAISummarizer(Settings())
