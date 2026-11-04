import re
from pathlib import Path

from meetmind.models import Minutes, Transcript


def timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def escape(text: str) -> str:
    text = " ".join(text.split())
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", text)


def render_markdown(minutes: Minutes, transcript: Transcript) -> str:
    segments = {s.id: s for s in transcript.segments}

    def references(ids: list[int]) -> str:
        return " ".join(
            f"[#{i} {timestamp(segments[i].start)}–{timestamp(segments[i].end)}](#segment-{i})"
            for i in ids
        )

    lines = [f"# {escape(minutes.title)}", "", "## 摘要", "", escape(minutes.summary), ""]
    lines.extend(["## 议题 / 议程", ""])
    for topic in minutes.topics:
        lines.append(
            f"- **{escape(topic.title)}**：{escape(topic.discussion)} "
            + references(topic.source_segment_ids)
        )
    if not minutes.topics:
        lines.append("未提及。")
    lines.extend(["", "## 决策", ""])
    lines.extend(
        f"- {escape(d.text)} {references(d.source_segment_ids)}" for d in minutes.decisions
    )
    if not minutes.decisions:
        lines.append("未提及。")
    lines.extend(["", "## 行动项", ""])
    for item in minutes.action_items:
        lines.append(
            f"- [ ] {escape(item.task)}；负责人：{escape(item.owner or '未指定')}；"
            f"截止：{escape(item.due or '未指定')} {references(item.source_segment_ids)}"
        )
    if not minutes.action_items:
        lines.append("未提及。")
    lines.extend(["", "## 带时间戳的转写", ""])
    for segment in transcript.segments:
        lines.extend(
            [
                f'<a id="segment-{segment.id}"></a>',
                "",
                f"- **#{segment.id} {timestamp(segment.start)}–{timestamp(segment.end)}** "
                f"{escape(segment.speaker or '说话人未知')}：{escape(segment.text)}",
                "",
            ]
        )
    return "\n".join(lines)


def write_minutes(output: Path, minutes: Minutes, transcript: Transcript) -> None:
    minutes.validate_sources(transcript)
    (output / "minutes.json").write_text(minutes.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (output / "minutes.md").write_text(render_markdown(minutes, transcript), encoding="utf-8")
