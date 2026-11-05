"""Check CER arithmetic and independent reference-aligned slicing."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def comparison(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "docs" / "e2e"))
    return importlib.import_module("compare_asr")


def test_completed_run_is_preserved_without_loading_audio_or_model(
    comparison, monkeypatch, tmp_path
):
    output = tmp_path / "base-far"
    output.mkdir()
    metrics = output / "metrics.json"
    original = b'{"status": "completed", "transcription_wall_seconds": 17.9}\n'
    metrics.write_bytes(original)
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare_asr.py", "--model", "base", "--condition", "far", "--output-dir", str(tmp_path)],
    )
    comparison.main()
    assert metrics.read_bytes() == original
    assert list(output.iterdir()) == [metrics]


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ("手机", "手机", 0),
        ("手机", "手", 1),
        ("手机", "手机啊", 1),
        ("手机", "耳机", 1),
        ("手机", "", 2),
    ],
)
def test_character_distance(comparison, reference, hypothesis, expected):
    assert comparison.distance(reference, hypothesis) == expected


def test_normalization_preserves_traditional_and_number_differences(comparison):
    assert comparison.normalize("Ａ，B！手机１２") == "ab手机12"
    assert comparison.normalize("臺灣二十") != comparison.normalize("台湾20")


def test_excerpt_cer_is_weighted_by_reference_length(comparison, monkeypatch):
    calls = []

    def decode(model, audio):
        calls.append(audio.copy())
        text = "手机" if len(calls) == 1 else ""
        return {"segments": [{"text": text}]}

    monkeypatch.setattr(comparison, "decode", decode)
    pcm = np.arange(48000, dtype=np.float32)
    result = comparison.score(
        None,
        pcm,
        [
            {"start": 0.5, "end": 1, "reference": "手机"},
            {"start": 2, "end": 3, "reference": "开会讨论"},
        ],
    )
    assert np.array_equal(calls[0], pcm[8000:16000])
    assert np.array_equal(calls[1], pcm[32000:48000])
    assert result["edit_distance"] == 4
    assert result["reference_characters"] == 6
    assert result["reference_audio_seconds"] == 1.5
    assert result["cer"] == 4 / 6


def test_official_reference_filter_excludes_overlap_and_clip_boundary(comparison):
    preparation = importlib.import_module("prepare_asr_comparison")
    intervals = [
        {"start": 1, "end": 2, "reference": "独立", "speaker": "a"},
        {"start": 3, "end": 5, "reference": "重叠甲", "speaker": "a"},
        {"start": 4, "end": 6, "reference": "重叠乙", "speaker": "b"},
        {"start": 298, "end": 299, "reference": "完整", "speaker": "b"},
        {"start": 299, "end": 301, "reference": "跨边界", "speaker": "a"},
    ]
    excerpts = preparation.nonoverlapping(intervals)
    assert [x["reference"] for x in excerpts] == ["独立", "完整"]


def test_textgrid_parser_reads_tiers_and_escaped_quotes(comparison):
    preparation = importlib.import_module("prepare_asr_comparison")
    text = '''item []:
    item [1]:
        name = "speaker1"
        intervals [1]:
            xmin = 0
            xmax = 1.2
            text = ""
        intervals [2]:
            xmin = 1.2
            xmax = 2.34
            text = "说""手机"""
    item [2]:
        name = "speaker2"
        intervals [1]:
            xmin = 5
            xmax = 6
            text = "讨论"
'''
    assert preparation.read_intervals(text) == [
        {"start": 1.2, "end": 2.34, "reference": '说"手机"', "speaker": "speaker1"},
        {"start": 5.0, "end": 6.0, "reference": "讨论", "speaker": "speaker2"},
    ]
