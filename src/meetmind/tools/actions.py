from datetime import datetime, timezone
from io import BytesIO
from uuid import uuid4

from docx import Document
from pydantic import Field, model_validator

from meetmind.models import Model
from meetmind.storage import Store
from meetmind.tools.export import render_markdown, timestamp


class MeetingArgs(Model):
    meeting_id: str = Field(min_length=1)


class TodoArgs(MeetingArgs):
    action_index: int = Field(ge=0, description="Zero-based index in the meeting action_items")


class CalendarArgs(MeetingArgs):
    title: str = Field(min_length=1, max_length=300)
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def valid_times(self):
        if self.start.utcoffset() is None or self.end.utcoffset() is None:
            raise ValueError("Calendar times must include explicit timezone offsets")
        if self.end <= self.start:
            raise ValueError("Calendar end must be after start")
        return self


class SearchArgs(Model):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=10)


def ics_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "\\n")
        .replace(";", "\\;")
        .replace(",", "\\,")
    )


def calendar_bytes(args: CalendarArgs) -> bytes:
    def utc(value):
        return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//MeetMind//MeetMind//EN",
        "BEGIN:VEVENT",
        f"UID:{uuid4().hex}@meetmind",
        f"DTSTAMP:{utc(datetime.now(timezone.utc))}",
        f"DTSTART:{utc(args.start)}",
        f"DTEND:{utc(args.end)}",
        "SUMMARY:" + ics_escape(args.title),
        "END:VEVENT",
        "END:VCALENDAR",
    ]
    # RFC 5545 folds at 75 octets without splitting UTF-8 characters.
    folded = []
    for line in lines:
        part = ""
        for char in line:
            if len((part + char).encode("utf-8")) > 75:
                folded.append(part)
                part = " "
            part += char
        folded.append(part)
    return ("\r\n".join(folded) + "\r\n").encode("utf-8")


class Tools:
    arguments = {
        "get_meeting": MeetingArgs,
        "create_todo": TodoArgs,
        "create_calendar_event": CalendarArgs,
        "export_markdown": MeetingArgs,
        "export_docx": MeetingArgs,
        "search_meetings": SearchArgs,
    }
    descriptions = {
        "get_meeting": "Read the full meeting including its indexed action items.",
        "create_todo": "Save an existing action item to the local task table, idempotently.",
        "create_calendar_event": "Generate an ICS file. Only use user-confirmed absolute times "
        "with explicit timezone offsets; never guess relative dates.",
        "export_markdown": "Export minutes and timestamped transcript as Markdown.",
        "export_docx": "Export editable minutes and timestamped transcript as DOCX.",
        "search_meetings": "Hybrid search of meeting transcripts with source timestamps.",
    }

    def __init__(self, store: Store, retriever):
        self.store = store
        self.retriever = retriever

    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": self.descriptions[name],
                    "parameters": model.model_json_schema(),
                },
            }
            for name, model in self.arguments.items()
        ]

    def call(self, name: str, raw_arguments: str) -> dict:
        if name not in self.arguments:
            raise ValueError("Unknown tool")
        args = self.arguments[name].model_validate_json(raw_arguments)
        if name == "search_meetings":
            return {"sources": [s.model_dump() for s in self.retriever.search(args.query, args.k)]}
        meeting = self.store.get(args.meeting_id)
        if name == "get_meeting":
            return meeting.model_dump()
        if name == "create_todo":
            return self.store.create_todo(args.meeting_id, args.action_index)
        if name == "create_calendar_event":
            return self.store.artifact("event.ics", calendar_bytes(args))
        if name == "export_markdown":
            return self.store.artifact(
                "minutes.md", render_markdown(meeting.minutes, meeting.transcript).encode("utf-8")
            )
        document = Document()
        document.add_heading(meeting.minutes.title, 0)
        document.add_heading("摘要", 1)
        document.add_paragraph(meeting.minutes.summary)
        for title, items in [
            ("议题", meeting.minutes.topics),
            ("决策", meeting.minutes.decisions),
            ("行动项", meeting.minutes.action_items),
        ]:
            document.add_heading(title, 1)
            for item in items:
                if title == "议题":
                    text = f"{item.title}: {item.discussion}"
                elif title == "决策":
                    text = item.text
                else:
                    text = (
                        f"{item.task}；负责人：{item.owner or '未指定'}；"
                        f"截止：{item.due or '未指定'}"
                    )
                document.add_paragraph(
                    text + " [分段 " + ", ".join(str(i) for i in item.source_segment_ids) + "]"
                )
        document.add_heading("带时间戳的转写", 1)
        for segment in meeting.transcript.segments:
            document.add_paragraph(
                f"#{segment.id} {timestamp(segment.start)}–"
                f"{timestamp(segment.end)} {segment.speaker or '说话人未知'}："
                f"{segment.text}"
            )
        buffer = BytesIO()
        document.save(buffer)
        return self.store.artifact("minutes.docx", buffer.getvalue())
