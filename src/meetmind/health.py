from importlib.util import find_spec
from pathlib import Path

from openai import OpenAI

from meetmind.config import Settings


def dependency_health(settings: Settings) -> dict:
    credentials = bool(settings.api_key.get_secret_value().strip() and settings.base_url.strip())
    dependencies = {
        "llm": {"configured": credentials and bool(settings.model.strip())},
        "embedding": {"configured": credentials and bool(settings.embedding_model.strip())},
        "asr": {"configured": bool(settings.asr_model.strip())},
    }
    for dependency in dependencies.values():
        dependency["status"] = "configured" if dependency["configured"] else "missing_config"
    if settings.demo_mode:
        for dependency in dependencies.values():
            dependency["status"] = "demo_bypass"
    elif settings.health_deep:
        for name in ("llm", "embedding"):
            dependency = dependencies[name]
            if not dependency["configured"]:
                continue
            try:
                with OpenAI(
                    api_key=settings.api_key.get_secret_value(),
                    base_url=settings.base_url,
                    timeout=3,
                    max_retries=0,
                ) as client:
                    if name == "llm":
                        client.chat.completions.create(
                            model=settings.model,
                            messages=[{"role": "user", "content": "Reply OK"}],
                            max_tokens=1,
                        )
                    else:
                        client.embeddings.create(model=settings.embedding_model, input=["health"])
                dependency["status"] = "ok"
            except Exception:
                # Upstream exception messages may include credentials or request contents.
                dependency["status"] = "unavailable"
        asr = dependencies["asr"]
        if asr["configured"]:
            if find_spec("faster_whisper") is None:
                asr["status"] = "missing_package"
            elif (Path(settings.asr_model) / "model.bin").is_file():
                asr["status"] = "local_weights_present"
            else:
                asr["status"] = "weights_unverified"
    ready = all(d["configured"] for d in dependencies.values())
    if settings.demo_mode:
        ready = True
    elif settings.health_deep:
        ready = ready and all(
            d["status"] in {"ok", "local_weights_present"} for d in dependencies.values()
        )
    return {
        "status": "degraded" if settings.health_deep and not ready else "ok",
        "alive": True,
        "ready": ready,
        "demo": settings.demo_mode,
        "deep": settings.health_deep,
        "dependencies": dependencies,
    }
