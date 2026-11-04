"""Real AMI evaluation; audio and downloaded annotations remain in ignored local storage."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import time
import wave
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import chromadb
import httpx
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

from meetmind.asr.faster_whisper import FasterWhisperASR
from meetmind.config import Settings
from meetmind.llm import ModelClient
from meetmind.models import Minutes
from meetmind.retrieval import HybridRetriever
from meetmind.service import Service
from meetmind.storage import Store
from meetmind.summarize.openai import OpenAISummarizer
from meetmind.tools.export import write_minutes

AUDIO_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/"
    "ES2002a/audio/ES2002a.Mix-Headset.wav"
)
ANNOTATIONS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["base", "small"], default="base")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    local = Path("data/external")
    local.mkdir(parents=True, exist_ok=True)
    metrics = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "audio_url": AUDIO_URL,
        "annotations_url": ANNOTATIONS_URL,
        "license": "CC BY 4.0; AMI Consortium / University of Edinburgh",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["faster-whisper", "ctranslate2", "openai", "chromadb", "meetmind"]
        },
    }
    stage = "download"
    try:
        for url, path in [
            (AUDIO_URL, local / "ES2002a.Mix-Headset.wav"),
            (ANNOTATIONS_URL, local / "ami_public_manual_1.6.2.zip"),
        ]:
            if not path.exists():
                temporary = path.with_suffix(path.suffix + ".part")
                with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as stream:
                        for block in response.iter_bytes():
                            stream.write(block)
                temporary.rename(path)
            print(f"Downloaded/cached {path.name}: {path.stat().st_size} bytes", flush=True)
        clip = local / "ES2002a.first300s.wav"
        with wave.open(str(local / "ES2002a.Mix-Headset.wav"), "rb") as source:
            metrics["source_audio_seconds"] = source.getnframes() / source.getframerate()
            if not clip.exists():
                with wave.open(str(clip), "wb") as target:
                    target.setparams(source.getparams())
                    target.writeframes(source.readframes(300 * source.getframerate()))
        with wave.open(str(clip), "rb") as audio:
            metrics["audio_seconds"] = audio.getnframes() / audio.getframerate()
            metrics["sample_rate"] = audio.getframerate()
            metrics["channels"] = audio.getnchannels()
        metrics["clip_sha256"] = hashlib.sha256(clip.read_bytes()).hexdigest()
        settings = Settings.load().model_copy(
            update={
                "asr_model": args.model,
                "asr_device": "cpu",
                "asr_compute_type": "int8",
                "asr_language": "en",
                "demo_mode": False,
            }
        )
        metrics.update(
            asr_model=args.model,
            asr_device="cpu",
            asr_compute_type="int8",
            asr_language="en",
            llm_model=settings.model,
            response_format=settings.response_format,
            embedding_model=settings.embedding_model,
            base_url=settings.base_url,
        )
        stage = "asr"
        print("ASR started (includes model initialization/download)", flush=True)
        started = time.perf_counter()
        transcript = FasterWhisperASR(settings).transcribe(clip)
        metrics["transcription_seconds"] = time.perf_counter() - started
        metrics["real_time_factor"] = metrics["transcription_seconds"] / metrics["audio_seconds"]
        metrics["segments"] = len(transcript.segments)
        (output / "transcript.json").write_text(
            transcript.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        print(f"ASR finished: {metrics['transcription_seconds']:.3f}s", flush=True)
        stage = "wer"
        words = []
        with zipfile.ZipFile(local / "ami_public_manual_1.6.2.zip") as archive:
            names = sorted(
                name
                for name in archive.namelist()
                if re.search(r"(?:^|/)ES2002a\.[A-D]\.words\.xml$", name)
            )
            if len(names) != 4:
                raise ValueError("Expected four AMI speaker reference files")
            for name in names:
                for element in ET.fromstring(archive.read(name)).iter("w"):
                    if "starttime" not in element.attrib or element.get("punc") == "true":
                        continue
                    start = float(element.attrib["starttime"])
                    if 0 <= start < metrics["audio_seconds"] and element.text:
                        words.append((start, name, element.text))
        reference = " ".join(word for _, _, word in sorted(words))
        hypothesis = " ".join(segment.text for segment in transcript.segments)
        ref = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", reference.lower())
        hyp = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", hypothesis.lower())
        previous = list(range(len(hyp) + 1))
        for i, word in enumerate(ref, 1):
            current = [i]
            for j, other in enumerate(hyp, 1):
                current.append(
                    min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (word != other))
                )
            previous = current
        metrics["wer"] = {
            "rate": previous[-1] / len(ref),
            "edit_distance": previous[-1],
            "reference_words": len(ref),
            "hypothesis_words": len(hyp),
            "reference_files": names,
            "method": "All speaker words sorted by start time, 0 <= start < 300s; "
            "lowercase, punctuation removed, internal apostrophes retained; "
            "Levenshtein / reference length. Overlap ordering is not speaker-aware.",
        }
        (output / "reference.txt").write_text(reference + "\n", encoding="utf-8")
        (output / "wer_tokens.json").write_text(
            json.dumps({"reference": ref, "hypothesis": hyp}, indent=2) + "\n", encoding="utf-8"
        )
        stage = "minutes"
        calls = []

        def count_request(request):
            calls.append(request.url.path)

        with httpx.Client(event_hooks={"request": [count_request]}) as transport:
            client = OpenAI(
                api_key=settings.api_key.get_secret_value(),
                base_url=settings.base_url,
                timeout=settings.timeout_seconds,
                max_retries=2,
                http_client=transport,
            )
            summarizer = OpenAISummarizer(settings, client=client)
            started = time.perf_counter()
            minutes = summarizer.summarize(transcript)
            metrics["minutes_seconds"] = time.perf_counter() - started
            metrics["minutes_http_requests"] = len(calls)
            metrics["minutes_retry_count"] = len(calls) - 1
            Minutes.model_validate_json(minutes.model_dump_json()).validate_sources(transcript)
            metrics["minutes_schema_valid"] = True
            write_minutes(output, minutes, transcript)
            print(f"Minutes validated; HTTP requests: {len(calls)}", flush=True)
            stage = "retrieval_qa"
            model = ModelClient(settings, client=client, embedding_model=settings.embedding_model)
            store = Store(Path("outputs") / ("real-e2e-" + uuid4().hex))
            chroma = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
            collection = chroma.create_collection("e2e-" + uuid4().hex, embedding_function=None)
            retriever = HybridRetriever(store, collection, model)
            service = Service(store, retriever, model, summarizer)
            service.ingest(transcript, minutes)
            question = "What product is the team designing, and what are its main goals?"
            answer = service.ask(question, k=3)
            if not answer["citations"]:
                raise ValueError("Q&A returned no citations")
            by_id = {s.id: s for s in transcript.segments}
            for citation in answer["citations"]:
                segments = [by_id[i] for i in citation["segment_ids"]]
                assert citation["start"] == segments[0].start
                assert citation["end"] == max(s.end for s in segments)
                assert citation["text"] == "\n".join(s.text for s in segments)
            (output / "qa.json").write_text(
                json.dumps({"question": question, **answer}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            metrics["qa_citations_valid"] = True
            metrics["qa_citations"] = len(answer["citations"])
            metrics["http_requests_by_path"] = {
                path: calls.count(path) for path in sorted(set(calls))
            }
        metrics["status"] = "passed"
    except Exception as exc:
        metrics["status"] = "failed"
        metrics["failed_stage"] = stage
        metrics["error_type"] = type(exc).__name__
        print(f"Failed at {stage}: {type(exc).__name__} (details suppressed)", flush=True)
    finally:
        (output / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0 if metrics["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
