from importlib.resources import files

from meetmind.models import Minutes, Transcript


def demo_transcript() -> Transcript:
    return Transcript.model_validate_json(
        files("meetmind").joinpath("samples/meeting.transcript.json").read_text(encoding="utf-8")
    )


class DemoSummarizer:
    """Recorded fixture response for the bundled fictional transcript; no model inference."""

    def summarize(self, transcript: Transcript) -> Minutes:
        if transcript != demo_transcript():
            raise ValueError("Demo only supports the bundled transcript")
        return Minutes.model_validate_json(
            files("meetmind").joinpath("samples/meeting.minutes.json").read_text(encoding="utf-8")
        )
