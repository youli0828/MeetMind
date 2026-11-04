from pathlib import Path

from meetmind.config import Settings
from meetmind.errors import MeetMindError
from meetmind.models import Segment, Transcript


class FasterWhisperASR:
    def __init__(self, settings: Settings):
        self.settings = settings

    def transcribe(self, audio: Path) -> Transcript:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise MeetMindError('Install local ASR with: pip install -e ".[asr]"') from exc

        model = WhisperModel(
            self.settings.asr_model,
            device=self.settings.asr_device,
            compute_type=self.settings.asr_compute_type,
        )
        segments, info = model.transcribe(
            str(audio), language=self.settings.asr_language, vad_filter=True
        )
        result = [
            Segment(id=i, start=s.start, end=s.end, text=s.text.strip(), speaker=None)
            for i, s in enumerate(segments)
            if s.text.strip()
        ]
        if not result:
            raise MeetMindError("No speech detected in audio")
        return Transcript(language=info.language, segments=result)
