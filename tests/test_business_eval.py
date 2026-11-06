import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "business_eval", ROOT / "docs/e2e/business-eval/evaluate.py"
)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_action_normalization_and_field_separation():
    predicted = [{"task": "整理ＡＰＩ说明！", "owner": "陈", "due": None}]
    gold = [{"task": "整理api说明", "owner": "陈", "due": None}]
    assert evaluation.action_scores(predicted, gold)["field_token_set"]["f1"] == 1
    assert evaluation.action_scores(predicted, gold)["exact_item_set"]["f1"] == 1
    assert evaluation.action_tokens([{"task": "陈"}]) != evaluation.action_tokens([{"owner": "陈"}])


def test_action_omission_and_hallucination():
    gold = [{"task": "安装"}]
    scores = evaluation.action_scores([{"task": "安装测试"}], gold)["field_token_set"]
    assert scores["precision"] == 0.5
    assert scores["recall"] == 1
    assert scores["f1"] == pytest.approx(2 / 3)
    assert evaluation.action_scores([], gold)["field_token_set"]["f1"] == 0
    assert evaluation.action_scores(gold, [])["field_token_set"]["f1"] == 0
    assert evaluation.action_scores([], [])["field_token_set"]["f1"] == 1


def test_recall_uses_meeting_and_segment_ids():
    relevant = [
        {"meeting_id": "right", "segment_id": 1},
        {"meeting_id": "right", "segment_id": 2},
    ]
    assert (
        evaluation.recall_at_k([SimpleNamespace(meeting_id="wrong", segment_ids=[1, 2])], relevant)
        == 0
    )
    assert (
        evaluation.recall_at_k([SimpleNamespace(meeting_id="right", segment_ids=[1])], relevant)
        == 0.5
    )


def test_offline_eval_matches_saved_measurements(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation must not read .env or call a model service")

    monkeypatch.setattr("meetmind.config.Settings.load", forbidden)
    measured = evaluation.evaluate()
    saved = json.loads((ROOT / "docs/e2e/business-eval/results.json").read_text(encoding="utf-8"))
    assert measured["retrieval"] == saved["retrieval"]
    assert measured["action_items"] == saved["action_items"]
