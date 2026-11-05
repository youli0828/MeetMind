"""Prepare a synchronized headset mix and official non-overlapping CER excerpts."""

import argparse
import hashlib
import json
import re
import wave
from pathlib import Path

import numpy as np
from evaluate_cer_subset import normalize

LOCAL = Path("data/external/alimeeting")


def read_intervals(text):
    intervals = []
    for tier in re.split(r"\n\s*item \[\d+\]:", text)[1:]:
        speaker = re.search(r'name = "([^"]*)"', tier).group(1)
        for match in re.finditer(
            r"intervals \[\d+\]:\s+xmin = ([\d.]+)\s+xmax = ([\d.]+)"
            r'\s+text = "((?:[^"\n]|"")*)"',
            tier,
        ):
            start, end, reference = match.groups()
            if normalize(reference):
                intervals.append(
                    {
                        "start": float(start),
                        "end": float(end),
                        "reference": reference.replace('""', '"'),
                        "speaker": speaker,
                    }
                )
    return sorted(intervals, key=lambda x: (x["start"], x["end"], x["speaker"]))


def nonoverlapping(intervals):
    return [
        {"sample_id": f"official-{i:03d}", **item}
        for i, item in enumerate(intervals)
        if 0 <= item["start"] < item["end"] <= 300
        and not any(
            i != j and other["start"] < item["end"] and item["start"] < other["end"]
            for j, other in enumerate(intervals)
        )
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads((LOCAL / "matching_sources.json").read_text(encoding="utf-8"))
    headsets = sorted(LOCAL.glob("R8001_M8004_N*.wav"))
    assert headsets, "Matching headset audio not extracted"
    signals = []
    members = []
    for path in headsets:
        with wave.open(str(path), "rb") as audio:
            assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (
                1,
                2,
                16000,
            )
            assert audio.getnframes() >= 300 * 16000
            signals.append(np.frombuffer(audio.readframes(300 * 16000), dtype="<i2"))
        members.append(
            {
                "file": path.as_posix(),
                "first300s_pcm_sha256": hashlib.sha256(signals[-1].tobytes()).hexdigest(),
            }
        )
    mix = np.rint(np.mean(np.stack(signals).astype(np.float32), axis=0)).astype("<i2")
    clip = LOCAL / "R8001_M8004.headset-mean.first300s.wav"
    with wave.open(str(clip), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(mix.tobytes())
    annotation = LOCAL / "R8001_M8004.TextGrid"
    raw = annotation.read_bytes()
    text = raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
    intervals = read_intervals(text)
    assert intervals
    alignment = []
    for path in headsets:
        speaker = path.stem.removeprefix("R8001_M8004_")
        near_text = path.with_suffix(".TextGrid").read_text(encoding="utf-8-sig")
        near_intervals = read_intervals(near_text)
        far_intervals = [x for x in intervals if x["speaker"] == speaker]
        fields = ["start", "end", "reference"]
        identical = [{k: x[k] for k in fields} for x in near_intervals] == [
            {k: x[k] for k in fields} for x in far_intervals
        ]
        alignment.append(
            {
                "speaker": speaker,
                "intervals": len(near_intervals),
                "near_far_annotations_identical": identical,
            }
        )
        assert identical, "Near/far time alignment needs inspection"
    excerpts = nonoverlapping(intervals)
    assert len(excerpts) > 7
    expanded = {
        "source": annotation.as_posix(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "method": "All complete lexical intervals in official far-field TextGrid within 0..300s "
        "that do not overlap any other lexical interval; independently decoded. "
        "Same boundaries and texts for both acoustic conditions. "
        "No continuous-300s CER and no speaker-order ambiguity from overlap.",
        "reference_characters": sum(len(normalize(x["reference"])) for x in excerpts),
        "reference_audio_seconds": sum(x["end"] - x["start"] for x in excerpts),
        "excerpts": excerpts,
    }
    (args.output_dir / "expanded_references.json").write_text(
        json.dumps(expanded, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    boundary = max(
        x["end"]
        for x in intervals
        if x["end"] <= 300 and not any(y["start"] < x["end"] < y["end"] for y in intervals)
    )
    prefix_intervals = [x for x in intervals if 0 <= x["start"] < x["end"] <= boundary]
    prefix = {
        "source": annotation.as_posix(),
        "source_sha256": expanded["source_sha256"],
        "method": "Continuous prefix ending at the last complete-utterance boundary <=300s "
        "with no utterance crossing it. All speakers' references concatenated by "
        "(start, end, speaker). Includes overlapping speech: this chronological "
        "linearization is order-sensitive and is not an official overlap-aware score. "
        "Decode prefix independently; do not align by ASR segment timestamps.",
        "reference_characters": sum(len(normalize(x["reference"])) for x in prefix_intervals),
        "intervals": prefix_intervals,
        "excerpts": [
            {
                "sample_id": "official-chronological-prefix",
                "start": 0,
                "end": boundary,
                "reference": "".join(x["reference"] for x in prefix_intervals),
            }
        ],
    }
    (args.output_dir / "prefix_reference.json").write_text(
        json.dumps(prefix, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    baseline = json.loads(
        Path("docs/e2e/alimeeting-20261105-base-rerun/cer_subset.json").read_text(encoding="utf-8")
    )
    checks = []
    for excerpt in baseline["excerpts"]:
        matches = [
            x
            for x in intervals
            if abs(x["start"] - excerpt["start"]) < 0.011
            and abs(x["end"] - excerpt["end"]) < 0.011
            and normalize(x["reference"]) == normalize(excerpt["reference"])
        ]
        checks.append({"sample_id": excerpt["sample_id"], "official_exact_match": bool(matches)})
    metadata = {
        **source,
        "license": "CC BY-SA 4.0",
        "attribution": "Alibaba Group / AliMeeting (Yu et al., ICASSP 2022)",
        "near": {
            "clip": clip.as_posix(),
            "sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
            "headsets": members,
            "transformation": "First 300 seconds at original time origin; arithmetic mean "
            "of all participant headset PCM16 channels, rounded to PCM16. "
            "No peak normalization, time shift or oracle speaker selection.",
        },
        "far": json.loads((LOCAL / "source.json").read_text(encoding="utf-8")),
        "baseline_official_reference_checks": checks,
        "near_far_annotation_alignment": alignment,
        "expanded_reference_characters": expanded["reference_characters"],
        "expanded_reference_audio_seconds": expanded["reference_audio_seconds"],
    }
    (args.output_dir / "data_provenance.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "Headsets",
        len(headsets),
        "expanded excerpts",
        len(excerpts),
        "characters",
        expanded["reference_characters"],
        "seconds",
        expanded["reference_audio_seconds"],
        flush=True,
    )
    print("Baseline official matches", sum(x["official_exact_match"] for x in checks), flush=True)


if __name__ == "__main__":
    main()
