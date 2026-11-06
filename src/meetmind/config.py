import os
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import Field, SecretStr

from meetmind.models import Model


class Settings(Model):
    api_key: SecretStr = SecretStr("")
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    response_format: Literal["json_schema", "json_object", "text"] = "json_schema"
    timeout_seconds: float = Field(default=60, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=10)
    max_transcript_chars: int = Field(default=60000, ge=1)
    asr_model: str = "medium"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    asr_language: str | None = None
    embedding_model: str = "text-embedding-3-small"
    data_dir: str = "outputs/server"
    demo_mode: bool = False
    health_deep: bool = False

    @classmethod
    def load(cls, env_file: Path = Path(".env")) -> "Settings":
        mapping = {
            "OPENAI_API_KEY": "api_key",
            "OPENAI_BASE_URL": "base_url",
            "OPENAI_MODEL": "model",
            "LLM_RESPONSE_FORMAT": "response_format",
            "LLM_TIMEOUT_SECONDS": "timeout_seconds",
            "LLM_MAX_ATTEMPTS": "max_attempts",
            "MAX_TRANSCRIPT_CHARS": "max_transcript_chars",
            "ASR_MODEL": "asr_model",
            "ASR_DEVICE": "asr_device",
            "ASR_COMPUTE_TYPE": "asr_compute_type",
            "ASR_LANGUAGE": "asr_language",
            "EMBEDDING_MODEL": "embedding_model",
            "MEETMIND_DATA_DIR": "data_dir",
            "MEETMIND_DEMO": "demo_mode",
            "MEETMIND_HEALTH_DEEP": "health_deep",
            "LLM_API_KEY": "api_key",
            "LLM_BASE_URL": "base_url",
            "LLM_MODEL": "model",
            "WHISPER_MODEL": "asr_model",
        }
        data = {}
        for values in (dotenv_values(env_file), os.environ):
            data.update({field: values[key] for key, field in mapping.items() if values.get(key)})
        return cls.model_validate(data, strict=False)
