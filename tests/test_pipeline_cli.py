import json
import wave
from unittest.mock import Mock

import httpx
from openai import APIConnectionError

from meetmind.cli import main
from meetmind.pipeline import process_transcript, run_audio
from meetmind.tools.export import render_markdown, timestamp


def test_audio_to_json_and_markdown_with_mocks(tmp_path, transcript, minutes):
    audio = tmp_path / "synthetic.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 1600)
    asr = Mock()
    asr.transcribe.return_value = transcript
    summarizer = Mock()
    summarizer.summarize.return_value = minutes
    output = tmp_path / "result"
    assert run_audio(audio, output, asr, summarizer) == minutes
    asr.transcribe.assert_called_once_with(audio)
    summarizer.summarize.assert_called_once_with(transcript)
    assert json.loads((output / "transcript.json").read_text("utf-8")) == transcript.model_dump()
    assert json.loads((output / "minutes.json").read_text("utf-8")) == minutes.model_dump()
    markdown = (output / "minutes.md").read_text("utf-8")
    assert "00:00:05.000–00:00:11.500" in markdown
    assert "负责人：未指定" in markdown
    assert "截止：未指定" in markdown
    assert "(#segment-1)" in markdown


def test_cli_run_wires_adapters(monkeypatch, tmp_path, transcript, minutes, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"synthetic-test-input")
    asr = Mock()
    asr.transcribe.return_value = transcript
    llm = Mock()
    llm.summarize.return_value = minutes
    monkeypatch.setattr("meetmind.cli.FasterWhisperASR", Mock(return_value=asr))
    monkeypatch.setattr("meetmind.cli.OpenAISummarizer", Mock(return_value=llm))
    assert main(["run", str(audio), "--output-dir", "result"]) == 0
    assert (tmp_path / "result/minutes.md").is_file()
    assert "Saved" in capsys.readouterr().out


def test_transcript_saved_on_llm_failure(tmp_path, transcript):
    llm = Mock()
    llm.summarize.side_effect = ValueError("failure")
    import pytest

    output = tmp_path / "failed"
    with pytest.raises(ValueError):
        process_transcript(transcript, output, llm)
    assert (output / "transcript.json").is_file()
    assert not (output / "minutes.json").exists()


def test_demo_cli_is_offline_and_refuses_overwrite(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_MAX_ATTEMPTS", "invalid")
    assert main(["demo", "--output-dir", "demo"]) == 0
    before = (tmp_path / "demo/minutes.json").read_bytes()
    assert main(["demo", "--output-dir", "demo"]) == 1
    assert (tmp_path / "demo/minutes.json").read_bytes() == before
    assert "already exists" in capsys.readouterr().err


def test_cli_missing_key_and_input(tmp_path, monkeypatch, transcript, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["run", "missing.wav"]) == 1
    assert "does not exist" in capsys.readouterr().err
    path = tmp_path / "transcript.json"
    path.write_text(transcript.model_dump_json(), encoding="utf-8")
    assert main(["summarize", str(path)]) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_cli_summarize_recovers_saved_transcript(tmp_path, monkeypatch, transcript, minutes):
    monkeypatch.chdir(tmp_path)
    llm = Mock()
    llm.summarize.return_value = minutes
    monkeypatch.setattr("meetmind.cli.OpenAISummarizer", Mock(return_value=llm))
    path = tmp_path / "transcript.json"
    path.write_text(transcript.model_dump_json(), encoding="utf-8")
    assert main(["summarize", str(path), "--output-dir", "retry"]) == 0
    assert (tmp_path / "retry/minutes.json").is_file()
    llm.summarize.assert_called_once_with(transcript)


def test_cli_api_error_does_not_expose_credentials(tmp_path, monkeypatch, transcript, capsys):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "transcript.json"
    path.write_text(transcript.model_dump_json(), encoding="utf-8")
    llm = Mock()
    llm.summarize.side_effect = APIConnectionError(
        message="private-secret", request=httpx.Request("POST", "https://example.invalid")
    )
    monkeypatch.setattr("meetmind.cli.OpenAISummarizer", Mock(return_value=llm))
    assert main(["summarize", str(path)]) == 1
    error = capsys.readouterr().err
    assert "LLM request failed" in error
    assert "private-secret" not in error


def test_markdown_escapes_untrusted_content(transcript, minutes):
    minutes.summary = '<script>alert("x")</script>\n# heading'
    markdown = render_markdown(minutes, transcript)
    assert "<script>" not in markdown
    assert "&lt;script&gt;" in markdown
    assert "\n# heading" not in markdown


def test_timestamp_rounding():
    assert timestamp(3599.9999) == "01:00:00.000"
