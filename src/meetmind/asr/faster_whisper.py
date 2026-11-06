from pathlib import Path

from meetmind.asr.diarization import DiarizationBackend, PyannoteBackend, assign_speakers
from meetmind.config import Settings
from meetmind.errors import MeetMindError
from meetmind.models import Segment, Transcript


class FasterWhisperASR:
    def __init__(self, settings: Settings, *, diarizer: DiarizationBackend | None = None):
        self.settings = settings
        self.diarizer = diarizer

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
        transcript = Transcript(language=info.language, segments=result)
        if self.settings.diarization:
            backend = self.diarizer if self.diarizer is not None else PyannoteBackend(self.settings)
            transcript = assign_speakers(transcript, backend.diarize(audio))
        return transcript
