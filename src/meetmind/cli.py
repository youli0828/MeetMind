import argparse
import sys
from pathlib import Path
from uuid import uuid4

from openai import OpenAIError
from pydantic import ValidationError

from meetmind.asr.faster_whisper import FasterWhisperASR
from meetmind.config import Settings
from meetmind.demo import DemoSummarizer, demo_transcript
from meetmind.errors import MeetMindError
from meetmind.models import Transcript
from meetmind.pipeline import process_transcript, run_audio
from meetmind.summarize.openai import OpenAISummarizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Timestamped audio to meeting minutes")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in [
        ("run", "Transcribe local audio and generate minutes using an LLM"),
        ("summarize", "Generate minutes from a timestamped transcript JSON"),
        ("demo", "Offline demo using a fictional transcript and recorded minutes"),
    ]:
        sub = subparsers.add_parser(command, help=help_text)
        if command != "demo":
            sub.add_argument("input", type=Path)
        sub.add_argument("--output-dir", type=Path, help="New output directory (must not exist)")
        if command != "demo":
            sub.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args(argv)
    output = args.output_dir or Path("outputs") / f"{args.command}-{uuid4().hex[:8]}"
    try:
        if output.exists():
            raise MeetMindError("Output directory already exists; choose a new directory")
        if args.command == "demo":
            process_transcript(demo_transcript(), output, DemoSummarizer())
            print("Offline fixture demo: no ASR or LLM inference performed.")
        else:
            if not args.input.is_file():
                raise MeetMindError("Input file does not exist")
            settings = Settings.load(args.env_file)
            summarizer = OpenAISummarizer(settings, artifacts_dir=output / "chunks")
            if args.command == "run":
                run_audio(args.input, output, FasterWhisperASR(settings), summarizer)
            else:
                transcript = Transcript.model_validate_json(args.input.read_text(encoding="utf-8"))
                process_transcript(transcript, output, summarizer)
    except ValidationError:
        print(
            "Error: invalid configuration or transcript fields; check documented schema.",
            file=sys.stderr,
        )
        return 1
    except OpenAIError:
        print(
            "Error: LLM request failed; check endpoint, credentials and model. "
            "If saved, transcript.json can be retried with summarize.",
            file=sys.stderr,
        )
        return 1
    except MeetMindError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError):
        print(
            "Error: unable to process input or write output; check audio, local ASR "
            "installation and filesystem permissions.",
            file=sys.stderr,
        )
        return 1
    print(f"Saved transcript.json, minutes.json and minutes.md to {output}")
    return 0
