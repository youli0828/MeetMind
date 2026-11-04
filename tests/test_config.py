import pytest
from pydantic import ValidationError

from meetmind.config import Settings


def test_dotenv_and_environment_precedence(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "OPENAI_API_KEY=test-only\nOPENAI_MODEL=file-model\nLLM_MAX_ATTEMPTS=2\nASR_LANGUAGE=zh\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_MODEL", "environment-model")
    result = Settings.load(env)
    assert result.model == "environment-model"
    assert result.max_attempts == 2
    assert result.asr_language == "zh"
    assert "test-only" not in repr(result)


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_invalid_attempt_count(tmp_path, monkeypatch, value):
    monkeypatch.setenv("LLM_MAX_ATTEMPTS", value)
    with pytest.raises(ValidationError):
        Settings.load(tmp_path / ".env")
