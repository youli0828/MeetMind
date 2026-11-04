from pathlib import Path

from meetmind.asr import ASR
from meetmind.errors import MeetMindError
from meetmind.models import Minutes, Transcript
from meetmind.summarize import Summarizer
from meetmind.tools.export import write_minutes


def process_transcript(transcript: Transcript, output: Path, summarizer: Summarizer) -> Minutes:
    output.mkdir(parents=True, exist_ok=False)
    (output / "transcript.json").write_text(
        transcript.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    minutes = summarizer.summarize(transcript)
    write_minutes(output, minutes, transcript)
    return minutes


def run_audio(audio: Path, output: Path, asr: ASR, summarizer: Summarizer) -> Minutes:
    if not audio.is_file():
        raise MeetMindError("Audio file does not exist")
    if output.exists():
        raise MeetMindError("Output directory already exists; choose a new directory")
    return process_transcript(asr.transcribe(audio), output, summarizer)
