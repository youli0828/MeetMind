"""Measure one local ASR model/condition, without loading .env or calling an LLM."""

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import numpy as np  # noqa: E402
from evaluate_cer_subset import normalize  # noqa: E402

LOCAL = Path("data/external/alimeeting")
BASELINE = Path("docs/e2e/alimeeting-20261105-base-rerun/cer_subset.json")


def save_metrics(path, report):
    pending = path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, char in enumerate(reference, 1):
        current = [i]
        for j, other in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def peak_memory():
    if os.name != "nt":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t)
            for name in [
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            ]
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize


def decode(model, audio):
    segments, info = model.transcribe(audio, language="zh", vad_filter=True)
    return {
        "language": info.language,
        "segments": [
            {"id": i, "start": s.start, "end": s.end, "text": s.text.strip(), "speaker": None}
            for i, s in enumerate(segments)
            if s.text.strip()
        ],
    }


def score(model, pcm, excerpts):
    results = []
    started = time.perf_counter()
    for excerpt in excerpts:
        start, end = excerpt["start"], excerpt["end"]
        transcript = decode(model, pcm[round(start * 16000) : round(end * 16000)])
        hypothesis = "".join(s["text"] for s in transcript["segments"])
        ref, hyp = normalize(excerpt["reference"]), normalize(hypothesis)
        results.append(
            {
                **excerpt,
                "hypothesis": hypothesis,
                "reference_normalized": ref,
                "hypothesis_normalized": hyp,
                "edit_distance": distance(ref, hyp),
                "reference_characters": len(ref),
            }
        )
    elapsed = time.perf_counter() - started
    edits = sum(r["edit_distance"] for r in results)
    count = sum(r["reference_characters"] for r in results)
    return {
        "cer": edits / count,
        "edit_distance": edits,
        "reference_characters": count,
        "reference_audio_seconds": sum(r["end"] - r["start"] for r in results),
        "evaluation_wall_seconds": elapsed,
        "excerpts": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, choices=["base", "small", "medium", "large-v3-turbo"]
    )
    parser.add_argument("--condition", choices=["far", "near"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir / f"{args.model}-{args.condition}"
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    if metrics_path.exists():
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        if saved.get("status") == "completed":
            print(args.model, args.condition, "already completed; preserved", flush=True)
            return
        raise RuntimeError(
            "Incomplete checkpoint exists; preserve it and use a new output directory"
        )
    clip = LOCAL / (
        "R8001_M8004_MS801.ch0.first300s.wav"
        if args.condition == "far"
        else "R8001_M8004.headset-mean.first300s.wav"
    )
    with wave.open(str(clip), "rb") as source:
        assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (
            1,
            2,
            16000,
        )
        assert source.getnframes() == 300 * 16000
        pcm = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32)
        pcm /= 32768.0
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    excerpts = [
        {k: r[k] for k in ["sample_id", "start", "end", "reference"]} for r in baseline["excerpts"]
    ]
    assert sum(len(normalize(r["reference"])) for r in excerpts) == 344
    weights = Path("data/external/faster-whisper-" + args.model)
    provenance_path = weights / "download_provenance.json"
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    )
    expected_hash = next(
        (
            item["lfs"]["sha256"]
            for item in provenance.get("siblings", [])
            if item["rfilename"] == "model.bin"
        ),
        "d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9",
    )
    hasher = hashlib.sha256()
    with (weights / "model.bin").open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    assert hasher.hexdigest() == expected_hash
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "condition": args.condition,
        "status": "running",
        "device": "cpu",
        "compute_type": "int8",
        "language": "zh",
        "cpu_threads": 4,
        "num_workers": 1,
        "random_seed": 0,
        "hf_endpoint": os.environ["HF_ENDPOINT"],
        "decode_options": "faster-whisper defaults; language=zh, vad_filter=True; beam_size=5",
        "model_revision": provenance.get("sha", "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"),
        "model_sha256": hasher.hexdigest(),
        "preprocessor_config_sha256": (
            hashlib.sha256((weights / "preprocessor_config.json").read_bytes()).hexdigest()
            if (weights / "preprocessor_config.json").exists()
            else None
        ),
        "audio": clip.as_posix(),
        "audio_sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
        "audio_seconds": 300,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["faster-whisper", "ctranslate2", "av", "numpy"]
        },
        "timing_scope": "ASR import + model initialization + continuous 300s decode; "
        "excludes downloads, CER excerpt decoding and output writes",
        "cer_method": baseline["method"],
        "cer_scope": baseline["scope"],
        "baseline_reference_file": BASELINE.as_posix(),
        "reference_url": baseline["reference_url"],
        "crosscheck_url": baseline["crosscheck_url"],
        "peak_memory_method": "Windows GetProcessMemoryInfo PeakWorkingSetSize for this fresh "
        "model/condition process; continuous peak before excerpt scoring; "
        "process peak after all scoring; includes Python and imports",
        "baseline_reference_sha256": hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
    }
    print(args.model, args.condition, "starting", flush=True)
    started = time.perf_counter()
    import ctranslate2
    from faster_whisper import WhisperModel

    ctranslate2.set_random_seed(0)
    model = WhisperModel(str(weights), device="cpu", compute_type="int8", cpu_threads=4)
    report["initialization_wall_seconds"] = time.perf_counter() - started
    transcript = decode(model, str(clip))
    report["transcription_wall_seconds"] = time.perf_counter() - started
    report["real_time_factor"] = report["transcription_wall_seconds"] / 300
    report["continuous_peak_working_set_bytes"] = peak_memory()
    (output / "transcript.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_metrics(metrics_path, report)
    print(
        args.model,
        args.condition,
        "full decode seconds",
        round(report["transcription_wall_seconds"], 3),
        flush=True,
    )
    report["cer_subset"] = score(model, pcm, excerpts)
    save_metrics(metrics_path, report)
    expanded = args.output_dir / "expanded_references.json"
    if expanded.exists():
        expanded_data = json.loads(expanded.read_text(encoding="utf-8"))
        report["cer_expanded"] = score(model, pcm, expanded_data["excerpts"])
        report["expanded_reference_sha256"] = hashlib.sha256(expanded.read_bytes()).hexdigest()
        save_metrics(metrics_path, report)
    prefix = args.output_dir / "prefix_reference.json"
    if prefix.exists():
        prefix_data = json.loads(prefix.read_text(encoding="utf-8"))
        report["cer_long_prefix"] = score(model, pcm, prefix_data["excerpts"])
        report["prefix_reference_sha256"] = hashlib.sha256(prefix.read_bytes()).hexdigest()
    report["process_peak_working_set_bytes"] = peak_memory()
    report["status"] = "completed"
    save_metrics(metrics_path, report)
    print(args.model, args.condition, "CER", report["cer_subset"]["cer"], flush=True)


if __name__ == "__main__":
    main()
