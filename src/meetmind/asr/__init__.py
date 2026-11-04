from pathlib import Path
from typing import Protocol

from meetmind.models import Transcript


class ASR(Protocol):
    def transcribe(self, audio: Path) -> Transcript: ...
