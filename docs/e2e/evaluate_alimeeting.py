"""Real Chinese ASR, minutes and hybrid retrieval evaluation; never logs credentials."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Set before importing any Hugging Face / faster-whisper modules.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import chromadb  # noqa: E402
import httpx  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402
from openai import OpenAI  # noqa: E402

from meetmind.asr.faster_whisper import FasterWhisperASR  # noqa: E402
from meetmind.config import Settings  # noqa: E402
from meetmind.llm import ModelClient  # noqa: E402
from meetmind.models import Minutes  # noqa: E402
from meetmind.retrieval import HybridRetriever  # noqa: E402
from meetmind.service import Service  # noqa: E402
from meetmind.storage import Store  # noqa: E402
from meetmind.summarize.openai import OpenAISummarizer  # noqa: E402
from meetmind.tools.export import write_minutes  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", default="base", help="base or local base weight directory")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    source = json.loads(Path("data/external/alimeeting/source.json").read_text(encoding="utf-8"))
    clip = Path(source["clip"])
    assert hashlib.sha256(clip.read_bytes()).hexdigest() == source["sha256"]
    settings = Settings.load().model_copy(
        update={
            "asr_model": args.model_path,
            "asr_device": "cpu",
            "asr_compute_type": "int8",
            "asr_language": "zh",
            "demo_mode": False,
        }
    )
    assert settings.model == "glm-4-flash"
    metrics = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["faster-whisper", "ctranslate2", "av", "openai", "chromadb", "meetmind"]
        },
        "asr_model": "base",
        "asr_model_path": args.model_path,
        "asr_device": "cpu",
        "asr_compute_type": "int8",
        "hf_endpoint": os.environ["HF_ENDPOINT"],
        "audio_seconds": source["audio_seconds"],
        "llm_model": settings.model,
        "response_format": settings.response_format,
        "embedding_model": settings.embedding_model,
        "transcription_timing_scope": "Includes model initialization; excludes weight download",
        "cer": None,
        "wer": None,
        "reference_status": "Official TextGrid references exist; none obtained locally. "
        "audio-only gzip prefix does not include annotations.",
        "ami": "Skipped: previous local download failed with ReadTimeout; prioritize Chinese.",
    }
    stage = "asr"
    requests = []
    http_stages = []
    completions = []

    def on_request(request):
        requests.append(request.url.path)
        http_stages.append(stage)

    class CountedCompletions:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def create(self, **kwargs):
            record = {"stage": stage, "validation_retry": len(kwargs["messages"]) > 2}
            completions.append(record)
            result = self.wrapped.create(**kwargs)
            record["finish_reason"] = result.choices[0].finish_reason if result.choices else None
            return result

    try:
        print("ASR started: base / CPU / int8 / zh", flush=True)
        started = time.perf_counter()
        transcript = FasterWhisperASR(settings).transcribe(clip)
        metrics["transcription_wall_seconds"] = time.perf_counter() - started
        metrics["real_time_factor"] = (
            metrics["transcription_wall_seconds"] / source["audio_seconds"]
        )
        metrics["segments"] = len(transcript.segments)
        (output / "transcript.json").write_text(
            transcript.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        print(f"ASR finished: {metrics['transcription_wall_seconds']:.3f}s", flush=True)
        stage = "minutes"
        with httpx.Client(event_hooks={"request": [on_request]}) as transport:
            client = OpenAI(
                api_key=settings.api_key.get_secret_value(),
                base_url=settings.base_url,
                timeout=settings.timeout_seconds,
                max_retries=2,
                http_client=transport,
            )
            client.chat.completions = CountedCompletions(client.chat.completions)
            summarizer = OpenAISummarizer(settings, client=client)
            started = time.perf_counter()
            minutes = summarizer.summarize(transcript)
            metrics["minutes_wall_seconds"] = time.perf_counter() - started
            Minutes.model_validate_json(minutes.model_dump_json()).validate_sources(transcript)
            metrics["minutes_schema_valid"] = True
            metrics["minutes_source_ids_valid"] = True
            write_minutes(output, minutes, transcript)
            print("Minutes schema and source IDs validated", flush=True)
            stage = "retrieval_qa"
            model = ModelClient(settings, client=client, embedding_model=settings.embedding_model)
            store = Store(Path("outputs") / ("alimeeting-e2e-" + uuid4().hex))
            chroma = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
            collection = chroma.create_collection("ali-e2e-" + uuid4().hex, embedding_function=None)
            service = Service(store, HybridRetriever(store, collection, model), model, summarizer)
            started = time.perf_counter()
            service.ingest(transcript, minutes)
            question = "这段会议讨论了哪些主要问题？请依据原文回答。"
            answer = service.ask(question, k=3)
            metrics["retrieval_index_and_qa_wall_seconds"] = time.perf_counter() - started
            (output / "qa.json").write_text(
                json.dumps({"question": question, **answer}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            assert answer["citations"]
            by_id = {s.id: s for s in transcript.segments}
            for citation in answer["citations"]:
                segments = [by_id[i] for i in citation["segment_ids"]]
                assert citation["start"] == segments[0].start
                assert citation["end"] == max(s.end for s in segments)
                assert citation["text"] == "\n".join(s.text for s in segments)
            metrics["qa_citations_valid"] = True
            metrics["qa_citations"] = len(answer["citations"])
        metrics["status"] = "passed"
    except Exception as exc:
        metrics.update(status="failed", failed_stage=stage, error_type=type(exc).__name__)
        print(f"Failed at {stage}: {type(exc).__name__}; raw error suppressed", flush=True)
    finally:
        metrics["completion_attempts"] = completions
        attempts = [c for c in completions if c["stage"] == "minutes"]
        metrics["minutes_schema_attempts"] = len(attempts)
        metrics["minutes_validation_retries"] = sum(c["validation_retry"] for c in attempts)
        metrics["minutes_http_requests"] = http_stages.count("minutes")
        metrics["minutes_transport_retries"] = http_stages.count("minutes") - len(attempts)
        metrics["http_requests_by_path"] = {p: requests.count(p) for p in sorted(set(requests))}
        (output / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0 if metrics["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
