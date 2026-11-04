import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from meetmind.errors import MeetMindError
from meetmind.models import Minutes, Model, Transcript


class Meeting(Model):
    id: str
    transcript: Transcript
    minutes: Minutes


class Store:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "meetmind.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS meetings (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS todos (
                    id TEXT PRIMARY KEY, meeting_id TEXT NOT NULL, action_index INTEGER NOT NULL,
                    payload TEXT NOT NULL, UNIQUE(meeting_id, action_index));
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY, filename TEXT NOT NULL, content BLOB NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path)
        try:
            with db:
                yield db
        finally:
            db.close()

    def add(self, meeting: Meeting):
        meeting.minutes.validate_sources(meeting.transcript)
        with self.connect() as db:
            db.execute(
                "INSERT INTO meetings VALUES (?, ?)", (meeting.id, meeting.model_dump_json())
            )

    def meetings(self) -> list[Meeting]:
        with self.connect() as db:
            return [
                Meeting.model_validate_json(row[0])
                for row in db.execute("SELECT payload FROM meetings ORDER BY rowid")
            ]

    def get(self, meeting_id: str) -> Meeting:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
        if row is None:
            raise MeetMindError("Meeting not found")
        return Meeting.model_validate_json(row[0])

    def create_todo(self, meeting_id: str, action_index: int) -> dict:
        meeting = self.get(meeting_id)
        if not 0 <= action_index < len(meeting.minutes.action_items):
            raise MeetMindError("Action item not found")
        item = meeting.minutes.action_items[action_index]
        payload = item.model_dump_json()
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO todos VALUES (?, ?, ?, ?)",
                (uuid4().hex, meeting_id, action_index, payload),
            )
            row = db.execute(
                "SELECT id FROM todos WHERE meeting_id = ? AND action_index = ?",
                (meeting_id, action_index),
            ).fetchone()
        return {"id": row[0], "meeting_id": meeting_id, **item.model_dump()}

    def todos(self) -> list[dict]:
        import json

        with self.connect() as db:
            return [
                {"id": row[0], "meeting_id": row[1], **json.loads(row[2])}
                for row in db.execute("SELECT id, meeting_id, payload FROM todos ORDER BY rowid")
            ]

    def artifact(self, filename: str, content: bytes) -> dict:
        artifact_id = uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO artifacts VALUES (?, ?, ?)", (artifact_id, filename, content))
        return {"url": f"/artifacts/{artifact_id}", "filename": filename}

    def read_artifact(self, artifact_id: str) -> tuple[str, bytes]:
        with self.connect() as db:
            row = db.execute(
                "SELECT filename, content FROM artifacts WHERE id = ?", (artifact_id,)
            ).fetchone()
        if row is None:
            raise MeetMindError("Artifact not found")
        return row
