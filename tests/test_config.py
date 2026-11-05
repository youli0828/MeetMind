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


@pytest.mark.parametrize(
    "legacy,alias,field,value",
    [
        ("OPENAI_API_KEY", "LLM_API_KEY", "api_key", "test-only"),
        ("OPENAI_BASE_URL", "LLM_BASE_URL", "base_url", "https://example.invalid/v1"),
        ("OPENAI_MODEL", "LLM_MODEL", "model", "custom-model"),
        ("ASR_MODEL", "WHISPER_MODEL", "asr_model", "medium"),
    ],
)
def test_aliases_and_cross_name_environment_precedence(
    tmp_path, monkeypatch, legacy, alias, field, value
):
    env = tmp_path / ".env"
    monkeypatch.delenv(legacy, raising=False)
    monkeypatch.delenv(alias, raising=False)
    env.write_text(f"{legacy}=legacy-file\n{alias}={value}\n", encoding="utf-8")

    def read_value():
        result = getattr(Settings.load(env), field)
        return result.get_secret_value() if field == "api_key" else result

    assert read_value() == value
    monkeypatch.setenv(legacy, "legacy-environment")
    assert read_value() == "legacy-environment"
    monkeypatch.setenv(alias, "alias-environment")
    assert read_value() == "alias-environment"
    monkeypatch.setenv(alias, "")
    assert read_value() == "legacy-environment"
