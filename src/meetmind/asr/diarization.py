from collections import defaultdict
from pathlib import Path
from typing import Protocol

from meetmind.config import Settings
from meetmind.errors import MeetMindError
from meetmind.models import Model, Transcript


class SpeakerTurn(Model):
    start: float
    end: float
    speaker: str


class DiarizationBackend(Protocol):
    def diarize(self, audio: Path) -> list[SpeakerTurn]: ...


class PyannoteBackend:
    def __init__(self, settings: Settings):
        self.settings = settings

    def diarize(self, audio: Path) -> list[SpeakerTurn]:
        token = self.settings.huggingface_token.get_secret_value().strip()
        if not token:
            raise MeetMindError("DIARIZATION=1 requires HUGGINGFACE_TOKEN or HF_TOKEN")
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise MeetMindError('DIARIZATION=1 requires: pip install -e ".[diarization]"') from None
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1", token=token
            )
            output = pipeline(str(audio))
            return [
                SpeakerTurn(start=float(turn.start), end=float(turn.end), speaker=speaker)
                for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True)
            ]
        except Exception:
            # Upstream errors can contain request URLs or credentials.
            raise MeetMindError(
                "Diarization failed; check HF token, gated model access, model download "
                "and pyannote audio decoding dependencies (FFmpeg)"
            ) from None


def assign_speakers(transcript: Transcript, turns: list[SpeakerTurn]) -> Transcript:
    segments = []
    for segment in transcript.segments:
        overlaps: dict[str, float] = defaultdict(float)
        for turn in turns:
            overlap = min(segment.end, turn.end) - max(segment.start, turn.start)
            if overlap > 0:
                overlaps[turn.speaker] += overlap
        speaker = max(overlaps, key=overlaps.get) if overlaps else None
        segments.append(segment.model_copy(update={"speaker": speaker}))
    return transcript.model_copy(update={"segments": segments})
