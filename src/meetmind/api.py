import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal

from fastapi import Body, Depends, FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from openai import OpenAIError
from pydantic import Field

from meetmind.asr.faster_whisper import FasterWhisperASR
from meetmind.config import Settings
from meetmind.errors import MeetMindError
from meetmind.llm import ModelClient
from meetmind.models import Minutes, Model, Transcript
from meetmind.retrieval import HybridRetriever
from meetmind.service import Service
from meetmind.storage import Store
from meetmind.summarize.openai import OpenAISummarizer
from meetmind.tools.actions import SearchArgs


class IngestRequest(Model):
    transcript: Transcript
    minutes: Minutes | None = None


class HistoryMessage(Model):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=16000)


class ChatRequest(Model):
    message: str = Field(min_length=1, max_length=2000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=20)


class ToolRequest(Model):
    name: str
    arguments: dict


@lru_cache
def get_service() -> Service:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    settings = Settings.load()
    directory = Path(settings.data_dir)
    store = Store(directory)
    demo = settings.demo_mode
    if demo:
        from meetmind.demo import DemoSummarizer
        from meetmind.offline import OfflineModel

        model = OfflineModel()
        summarizer = DemoSummarizer()
        collection_name = "demo-hash-v1"
    else:
        model = ModelClient(settings, embedding_model=settings.embedding_model)
        summarizer = OpenAISummarizer(settings)
        # Separate incompatible embedding spaces rather than silently reusing an index.
        import hashlib

        identity = settings.base_url + "\n" + model.embedding_model
        collection_name = "meetings-" + hashlib.sha256(identity.encode()).hexdigest()[:20]
    client = chromadb.PersistentClient(
        path=str(directory / "chroma"), settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(collection_name, embedding_function=None)
    retriever = HybridRetriever(store, collection, model)
    # Recover the selected embedding index from authoritative meetings after a model switch
    # or an interrupted ingestion. Existing entries are immutable.
    existing = set(collection.get(include=[])["ids"])
    from meetmind.retrieval import chunks

    for meeting in store.meetings():
        if not {c.id for c in chunks(meeting)} <= existing:
            parts = chunks(meeting)
            collection.upsert(
                ids=[c.id for c in parts], embeddings=model.embed([c.text for c in parts])
            )
    return Service(store, retriever, model, summarizer)


def create_app() -> FastAPI:
    app = FastAPI(title="MeetMind", version="0.2.0")

    @app.exception_handler(MeetMindError)
    def application_error(request, exc):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(OpenAIError)
    def upstream_error(request, exc):
        return JSONResponse(status_code=502, content={"detail": "Model service request failed"})

    @app.exception_handler(ValueError)
    def invalid_value(request, exc):
        return JSONResponse(status_code=400, content={"detail": "Invalid data or tool arguments"})

    @app.get("/", response_class=HTMLResponse)
    def home():
        return files("meetmind").joinpath("web/index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        return {"status": "ok", "demo": Settings.load().demo_mode}

    @app.get("/sample")
    def sample():
        from meetmind.demo import DemoSummarizer, demo_transcript

        transcript = demo_transcript()
        return {
            "transcript": transcript.model_dump(),
            "minutes": DemoSummarizer().summarize(transcript).model_dump(),
        }

    @app.get("/meetings")
    def meetings(service: Annotated[Service, Depends(get_service)]):
        return [{"id": m.id, "title": m.minutes.title} for m in service.store.meetings()]

    @app.get("/meetings/{meeting_id}")
    def meeting(meeting_id: str, service: Annotated[Service, Depends(get_service)]):
        return service.store.get(meeting_id)

    @app.post("/meetings", status_code=201)
    def ingest(body: IngestRequest, service: Annotated[Service, Depends(get_service)]):
        return service.ingest(body.transcript, body.minutes)

    @app.post("/audio", status_code=201)
    def audio(
        data: Annotated[
            bytes,
            Body(media_type="application/octet-stream", min_length=1, max_length=50 * 1024 * 1024),
        ],
        extension: Annotated[str, Query(pattern=r"^\.(wav|mp3|m4a|flac|ogg|webm)$")],
        service: Annotated[Service, Depends(get_service)],
    ):
        if Settings.load().demo_mode:
            raise MeetMindError("Offline demo accepts only the bundled transcript, not audio")
        settings = Settings.load()
        with TemporaryDirectory() as directory:
            path = Path(directory) / ("audio" + extension)
            path.write_bytes(data)
            transcript = FasterWhisperASR(settings).transcribe(path)
        return service.ingest(transcript)

    @app.post("/search")
    def search(body: SearchArgs, service: Annotated[Service, Depends(get_service)]):
        return service.retriever.search(body.query, body.k)

    @app.post("/ask")
    def ask(body: SearchArgs, service: Annotated[Service, Depends(get_service)]):
        return service.ask(body.query, body.k)

    @app.post("/chat")
    def chat(body: ChatRequest, service: Annotated[Service, Depends(get_service)]):
        def events():
            yield "data: " + json.dumps({"event": "start"}) + "\n\n"
            try:
                for event in service.agent.events(
                    body.message, [m.model_dump() for m in body.history]
                ):
                    yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
            except (ValueError, OpenAIError):
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "event": "error",
                            "text": "Agent request failed; completed actions remain saved. "
                            "Check tasks/files "
                            "before retrying.",
                        }
                    )
                    + "\n\n"
                )

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/tools")
    def tool(body: ToolRequest, service: Annotated[Service, Depends(get_service)]):
        return service.tools.call(body.name, json.dumps(body.arguments))

    @app.get("/todos")
    def todos(service: Annotated[Service, Depends(get_service)]):
        return service.store.todos()

    @app.get("/artifacts/{artifact_id}")
    def artifact(artifact_id: str, service: Annotated[Service, Depends(get_service)]):
        filename, content = service.store.read_artifact(artifact_id)
        types = {
            "md": "text/markdown; charset=utf-8",
            "ics": "text/calendar; charset=utf-8",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }
        return Response(
            content,
            media_type=types[filename.rsplit(".", 1)[1]],
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return app


app = create_app()
