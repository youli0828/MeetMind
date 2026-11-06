import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from meetmind.asr.diarization import PyannoteBackend, SpeakerTurn, assign_speakers
from meetmind.asr.faster_whisper import FasterWhisperASR
from meetmind.config import Settings
from meetmind.models import Segment, Transcript


def test_overlap_sums_turns_and_keeps_unmatched_null():
    transcript = Transcript(
        language="en",
        segments=[
            Segment(id=0, start=0.0, end=10.0, text="hello", speaker=None),
            Segment(id=1, start=11.0, end=12.0, text="world", speaker=None),
        ],
    )
    result = assign_speakers(
        transcript,
        [
            SpeakerTurn(start=0.0, end=3.0, speaker="A"),
            SpeakerTurn(start=3.0, end=7.0, speaker="B"),
            SpeakerTurn(start=7.0, end=10.0, speaker="A"),
        ],
    )
    assert [s.speaker for s in result.segments] == ["A", None]
    assert transcript.segments[0].speaker is None
    assert result.segments[0].model_dump(exclude={"speaker"}) == (
        transcript.segments[0].model_dump(exclude={"speaker"})
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_asr_optional_backend(monkeypatch, tmp_path, enabled):
    model = Mock()
    model.transcribe.return_value = (
        iter([SimpleNamespace(start=0.0, end=1.0, text="hello")]),
        SimpleNamespace(language="en"),
    )
    monkeypatch.setitem(
        sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Mock(return_value=model))
    )
    backend = Mock()
    backend.diarize.return_value = [SpeakerTurn(start=0.0, end=1.0, speaker="SPEAKER_00")]
    audio = tmp_path / "test.wav"
    result = FasterWhisperASR(Settings(diarization=enabled), diarizer=backend).transcribe(audio)
    assert result.segments[0].speaker == ("SPEAKER_00" if enabled else None)
    if enabled:
        backend.diarize.assert_called_once_with(audio)
    else:
        backend.diarize.assert_not_called()


def test_missing_token_is_explicit(tmp_path):
    with pytest.raises(ValueError, match="HUGGINGFACE_TOKEN or HF_TOKEN"):
        PyannoteBackend(Settings(diarization=True)).diarize(tmp_path / "audio.wav")


def test_missing_package_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)
    with pytest.raises(ValueError, match=r"\[diarization\]"):
        PyannoteBackend(Settings(huggingface_token="test-only")).diarize(tmp_path / "audio.wav")


def test_pyannote_adapter_and_safe_errors(monkeypatch, tmp_path):
    pipeline = Mock()
    pipeline.return_value.speaker_diarization.itertracks.return_value = [
        (SimpleNamespace(start=0.0, end=1.0), None, "SPEAKER_00")
    ]
    factory = Mock(return_value=pipeline)
    monkeypatch.setitem(
        sys.modules,
        "pyannote.audio",
        SimpleNamespace(Pipeline=SimpleNamespace(from_pretrained=factory)),
    )
    backend = PyannoteBackend(Settings(huggingface_token="test-only"))
    assert backend.diarize(tmp_path / "audio.wav") == [
        SpeakerTurn(start=0.0, end=1.0, speaker="SPEAKER_00")
    ]
    assert factory.call_args.kwargs == {"token": "test-only"}
    factory.side_effect = RuntimeError("private-token")
    with pytest.raises(ValueError, match="gated model access") as error:
        backend.diarize(tmp_path / "audio.wav")
    assert "private-token" not in str(error.value)


def test_env_alias_and_secret(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DIARIZATION=1\nHF_TOKEN=file-test-only\n", encoding="utf-8")
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "env-test-only")
    settings = Settings.load(env)
    assert settings.diarization
    assert settings.huggingface_token.get_secret_value() == "env-test-only"
    assert "env-test-only" not in repr(settings)
