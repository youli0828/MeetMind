import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from meetmind.asr.faster_whisper import FasterWhisperASR
from meetmind.config import Settings


def test_local_adapter_maps_timestamps_and_consumes_generator(monkeypatch, tmp_path):
    model = Mock()
    model.transcribe.return_value = (
        iter([SimpleNamespace(start=1.25, end=2.75, text=" hello ")]),
        SimpleNamespace(language="en"),
    )
    constructor = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=constructor))
    audio = tmp_path / "speech.wav"
    result = FasterWhisperASR(Settings(asr_language="en")).transcribe(audio)
    constructor.assert_called_once_with("small", device="cpu", compute_type="int8")
    model.transcribe.assert_called_once_with(str(audio), language="en", vad_filter=True)
    assert result.segments[0].start == 1.25
    assert result.segments[0].end == 2.75
    assert result.segments[0].text == "hello"
    assert result.segments[0].speaker is None


def test_no_speech_is_explicit_error(monkeypatch, tmp_path):
    model = Mock()
    model.transcribe.return_value = (iter([]), SimpleNamespace(language="en"))
    monkeypatch.setitem(
        sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Mock(return_value=model))
    )
    with pytest.raises(ValueError, match="No speech"):
        FasterWhisperASR(Settings()).transcribe(tmp_path / "silent.wav")


def test_missing_optional_dependency(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(ValueError, match="Install local ASR"):
        FasterWhisperASR(Settings()).transcribe(tmp_path / "speech.wav")
