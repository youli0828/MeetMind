from typing import Protocol

from meetmind.models import Minutes, Transcript


class Summarizer(Protocol):
    def summarize(self, transcript: Transcript) -> Minutes: ...
