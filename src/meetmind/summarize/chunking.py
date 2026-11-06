from meetmind.errors import MeetMindError
from meetmind.models import Minutes, Transcript


def chunk_transcript(transcript: Transcript, limit: int, overlap: int) -> list[Transcript]:
    def make(segments):
        return Transcript(language=transcript.language, segments=segments)

    segments = []
    for segment in transcript.segments:
        pending = [segment]
        while pending:
            part = pending.pop()
            if len(make([part]).model_dump_json()) <= limit:
                segments.append(part)
            elif len(part.text) > 1:
                midpoint = len(part.text) // 2
                pending.extend(
                    [
                        part.model_copy(update={"text": part.text[midpoint:]}),
                        part.model_copy(update={"text": part.text[:midpoint]}),
                    ]
                )
            else:
                raise MeetMindError("Chunk character limit is too small for transcript metadata")

    chunks = []
    start = 0
    while start < len(segments):
        end = start + 1
        # Fragments of one oversized segment retain its ID and live in separate chunks.
        while end < len(segments):
            candidate = segments[start : end + 1]
            if len({s.id for s in candidate}) != len(candidate):
                break
            if len(make(candidate).model_dump_json()) > limit:
                break
            end += 1
        chunks.append(make(segments[start:end]))
        if end == len(segments):
            break
        next_start = max(start + 1, end - overlap)
        while next_start < end:
            candidate = segments[next_start : end + 1]
            if (
                len({s.id for s in candidate}) == len(candidate)
                and len(make(candidate).model_dump_json()) <= limit
            ):
                break
            next_start += 1
        start = next_start
    return chunks


def deduplicate_minutes(minutes: Minutes) -> Minutes:
    result = minutes.model_copy(deep=True)
    for field in ("topics", "decisions", "action_items"):
        unique = {}
        for item in getattr(result, field):
            key = tuple(
                " ".join(value.split()).casefold() if isinstance(value, str) else value
                for value in item.model_dump(exclude={"source_segment_ids"}).values()
            )
            if key in unique:
                unique[key].source_segment_ids = sorted(
                    set(unique[key].source_segment_ids + item.source_segment_ids)
                )
            else:
                item.source_segment_ids = sorted(set(item.source_segment_ids))
                unique[key] = item
        setattr(result, field, list(unique.values()))
    return result
