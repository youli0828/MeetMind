import re
from collections import defaultdict

from pydantic import Field
from rank_bm25 import BM25Okapi

from meetmind.models import Model
from meetmind.storage import Meeting, Store


class Chunk(Model):
    id: str
    meeting_id: str
    title: str
    segment_ids: list[int]
    start: float
    end: float
    text: str


class Answer(Model):
    text: str = Field(min_length=1)
    citation_ids: list[str]


def chunks(meeting: Meeting, max_chars: int = 1200) -> list[Chunk]:
    # Keep segments intact so every excerpt maps back to exact source timestamps.
    groups = []
    current = []
    size = 0
    for segment in meeting.transcript.segments:
        if current and size + len(segment.text) > max_chars:
            groups.append(current)
            current = []
            size = 0
        current.append(segment)
        size += len(segment.text)
    if current:
        groups.append(current)
    return [
        Chunk(
            id=f"{meeting.id}:{i}",
            meeting_id=meeting.id,
            title=meeting.minutes.title,
            segment_ids=[s.id for s in group],
            start=group[0].start,
            end=max(s.end for s in group),
            text="\n".join(s.text for s in group),
        )
        for i, group in enumerate(groups)
    ]


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]", text.lower())


class HybridRetriever:
    def __init__(self, store: Store, collection, embedder):
        self.store = store
        self.collection = collection
        self.embedder = embedder

    def index(self, meeting: Meeting):
        parts = chunks(meeting)
        vectors = self.embedder.embed([c.text for c in parts])
        self.collection.add(ids=[c.id for c in parts], embeddings=vectors)

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        parts = [chunk for meeting in self.store.meetings() for chunk in chunks(meeting)]
        if not parts:
            return []
        by_id = {c.id: c for c in parts}
        corpus = [tokens(c.text) or [c.text] for c in parts]
        scores = BM25Okapi(corpus, epsilon=0.5).get_scores(tokens(query))
        lexical = sorted(range(len(parts)), key=lambda i: (-scores[i], i))
        lexical_ids = [parts[i].id for i in lexical if scores[i] > 0]
        result = self.collection.query(
            query_embeddings=self.embedder.embed([query]), n_results=min(len(parts), max(k * 4, 20))
        )
        vector_ids = [key for key in result["ids"][0] if key in by_id]
        fused = defaultdict(float)
        for ranking in (lexical_ids, vector_ids):
            for rank, key in enumerate(ranking):
                fused[key] += 1 / (60 + rank + 1)
        return [by_id[key] for key in sorted(fused, key=lambda key: (-fused[key], key))[:k]]
