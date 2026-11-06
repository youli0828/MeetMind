from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from meetmind.api import create_app
from meetmind.config import Settings
from meetmind.health import dependency_health


def test_default_medium_and_health_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Settings.load().asr_model == "medium"
    monkeypatch.setenv("MEETMIND_HEALTH_DEEP", "1")
    assert Settings.load().health_deep is True


@pytest.mark.parametrize("deep,code", [(False, 200), (True, 503)])
def test_health_missing_configuration(monkeypatch, deep, code):
    monkeypatch.setattr("meetmind.api.Settings.load", lambda: Settings(health_deep=deep))
    with TestClient(create_app()) as client:
        response = client.get("/health")
    assert response.status_code == code
    report = response.json()
    assert report["alive"] is True
    assert report["ready"] is False
    assert report["dependencies"]["llm"]["status"] == "missing_config"
    assert report["dependencies"]["embedding"]["configured"] is False


def test_basic_health_does_not_probe(monkeypatch):
    probe = MagicMock(side_effect=AssertionError("must not probe"))
    monkeypatch.setattr("meetmind.health.OpenAI", probe)
    report = dependency_health(Settings(api_key="test-secret"))
    assert report["ready"] is True
    assert all(d["status"] == "configured" for d in report["dependencies"].values())
    probe.assert_not_called()


@pytest.mark.parametrize("failed", [None, "llm", "embedding"])
def test_deep_probes_and_redaction(monkeypatch, tmp_path, failed):
    (tmp_path / "model.bin").write_bytes(b"test")
    factory = MagicMock()
    client = factory.return_value.__enter__.return_value
    if failed:
        method = client.chat.completions.create if failed == "llm" else client.embeddings.create
        method.side_effect = RuntimeError("test-secret upstream response")
    monkeypatch.setattr("meetmind.health.OpenAI", factory)
    monkeypatch.setattr("meetmind.health.find_spec", lambda name: object())
    settings = Settings(api_key="test-secret", asr_model=str(tmp_path), health_deep=True)
    monkeypatch.setattr("meetmind.api.Settings.load", lambda: settings)
    with TestClient(create_app()) as api:
        response = api.get("/health")
    assert response.status_code == (200 if failed is None else 503)
    report = response.json()
    assert report["ready"] is (failed is None)
    assert report["dependencies"]["asr"]["status"] == "local_weights_present"
    assert "test-secret" not in str(report)
    client.chat.completions.create.assert_called_once()
    client.embeddings.create.assert_called_once()
    assert factory.call_args.kwargs["timeout"] == 3
    assert factory.call_args.kwargs["max_retries"] == 0


@pytest.mark.parametrize(
    "installed,status", [(False, "missing_package"), (True, "weights_unverified")]
)
def test_deep_asr_does_not_download(monkeypatch, installed, status):
    monkeypatch.setattr("meetmind.health.find_spec", lambda name: object() if installed else None)
    report = dependency_health(Settings(health_deep=True))
    assert report["dependencies"]["asr"]["status"] == status
    assert report["ready"] is False


def test_deep_demo_bypasses_dependencies(monkeypatch):
    probe = MagicMock(side_effect=AssertionError("must not probe"))
    monkeypatch.setattr("meetmind.health.OpenAI", probe)
    report = dependency_health(Settings(demo_mode=True, health_deep=True))
    assert report["ready"] is True
    assert all(d["status"] == "demo_bypass" for d in report["dependencies"].values())
    probe.assert_not_called()
