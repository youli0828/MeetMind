"""Score independently decoded, reference-aligned AliMeeting excerpt subset."""

import argparse
import ast
import hashlib
import json
import os
import time
import unicodedata
import wave
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import httpx  # noqa: E402

from meetmind.asr.faster_whisper import FasterWhisperASR  # noqa: E402
from meetmind.config import Settings  # noqa: E402

URL = (
    "https://datasets-server.huggingface.co/first-rows?"
    "dataset=playwithmino/alimeeting-eval-8k&config=default&split=test"
)
CHECK_URL = (
    "https://huggingface.co/desh2608/icefall-asr-alimeeting-pruned-transducer-stateless7/"
    "resolve/main/log/greedy_search/"
    "recogs-eval_sdm-greedy_search-epoch-15-avg-8-context-2-max-sym-per-frame-1.txt"
)


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).lower() if c.isalnum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    local = Path("data/external/alimeeting")
    cached = local / "hf-eval-first-rows.json"
    crosscheck = local / "icefall-eval-sdm-recogs.txt"
    for url, path in [(URL, cached), (CHECK_URL, crosscheck)]:
        if not path.exists():
            with httpx.Client(timeout=60, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                path.write_bytes(response.content)
    data = json.loads(cached.read_text(encoding="utf-8"))
    rows = [
        r["row"]
        for r in data["rows"]
        if r["row"]["session_id"] == "R8001_M8004"
        and 0 <= r["row"]["clip_start_s"] < r["row"]["clip_end_s"] <= 300
        and r["row"]["num_speakers"] == 1
    ]
    assert len(rows) == 7
    known = {
        normalize("".join(ast.literal_eval(line.split("ref=", 1)[1])))
        for line in crosscheck.read_text(encoding="utf-8").splitlines()
        if line.startswith("R8001_M8004-") and "\tref=" in line
    }
    settings = Settings.load().model_copy(
        update={
            "asr_model": "data/external/faster-whisper-base",
            "asr_device": "cpu",
            "asr_compute_type": "int8",
            "asr_language": "zh",
            "demo_mode": False,
        }
    )
    results = []
    started = time.perf_counter()
    for row in rows:
        reference = "".join(s["text_raw"] for s in row["sources"])
        ref = normalize(reference)
        assert ref in known, "Subset reference does not match corpus-derived Icefall ref"
        clip = local / (row["sample_id"] + ".wav")
        with wave.open(str(local / "R8001_M8004_MS801.ch0.first300s.wav"), "rb") as source:
            rate = source.getframerate()
            start = round(row["clip_start_s"] * rate)
            end = round(row["clip_end_s"] * rate)
            source.setpos(start)
            pcm = source.readframes(end - start)
            with wave.open(str(clip), "wb") as target:
                target.setparams(source.getparams())
                target.writeframes(pcm)
        transcript = FasterWhisperASR(settings).transcribe(clip)
        hypothesis = "".join(s.text for s in transcript.segments)
        hyp = normalize(hypothesis)
        previous = list(range(len(hyp) + 1))
        for i, char in enumerate(ref, 1):
            current = [i]
            for j, other in enumerate(hyp, 1):
                current.append(
                    min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char != other))
                )
            previous = current
        results.append(
            {
                "sample_id": row["sample_id"],
                "start": row["clip_start_s"],
                "end": row["clip_end_s"],
                "reference": reference,
                "hypothesis": hypothesis,
                "reference_normalized": ref,
                "hypothesis_normalized": hyp,
                "edit_distance": previous[-1],
                "reference_characters": len(ref),
            }
        )
        print("Scored", row["sample_id"], previous[-1], "/", len(ref), flush=True)
    edits = sum(r["edit_distance"] for r in results)
    count = sum(r["reference_characters"] for r in results)
    report = {
        "scope": "Seven single-speaker reference-aligned excerpts within first 300s; "
        "independently decoded, not CER for the continuous five-minute transcript",
        "reference_url": URL,
        "crosscheck_url": CHECK_URL,
        "reference_cache_sha256": hashlib.sha256(cached.read_bytes()).hexdigest(),
        "crosscheck": "All normalized references exactly match corpus-derived Icefall ref entries; "
        "no Icefall hypotheses used",
        "method": "Per-excerpt Levenshtein character distance; NFKC, lowercase, "
        "keep Unicode alphanumeric only; no simplified/traditional conversion. "
        "Sum edit distances / sum reference lengths. No Chinese word-based WER.",
        "model": "base",
        "device": "cpu",
        "compute_type": "int8",
        "reference_audio_seconds": sum(r["end"] - r["start"] for r in results),
        "evaluation_wall_seconds": time.perf_counter() - started,
        "edit_distance": edits,
        "reference_characters": count,
        "cer": edits / count,
        "excerpts": results,
    }
    (args.run_dir / "cer_subset.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metrics_path = args.run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["reference_status"] = (
        "No complete aligned reference for continuous 300s. "
        "Seven reference excerpts obtained and crosschecked; see cer_subset.json."
    )
    metrics["cer_subset"] = {
        key: report[key]
        for key in [
            "cer",
            "edit_distance",
            "reference_characters",
            "reference_audio_seconds",
            "scope",
        ]
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Subset CER", edits, "/", count, "=", edits / count, flush=True)


if __name__ == "__main__":
    main()
