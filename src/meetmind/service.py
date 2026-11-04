from uuid import uuid4

from meetmind.agent import Agent
from meetmind.storage import Meeting, Store
from meetmind.tools.actions import Tools


class Service:
    def __init__(self, store: Store, retriever, model, summarizer):
        self.store = store
        self.retriever = retriever
        self.model = model
        self.summarizer = summarizer
        self.tools = Tools(store, retriever)
        self.agent = Agent(model, self.tools)

    def ingest(self, transcript, minutes=None) -> Meeting:
        if minutes is None:
            minutes = self.summarizer.summarize(transcript)
        minutes.validate_sources(transcript)
        meeting = Meeting(id=uuid4().hex, transcript=transcript, minutes=minutes)
        # Only publish to the source-of-truth store after embeddings are indexed successfully.
        self.retriever.index(meeting)
        self.store.add(meeting)
        return meeting

    def ask(self, question: str, k: int) -> dict:
        sources = self.retriever.search(question, k)
        if not sources:
            return {"text": "没有可供回答的会议记录。", "citations": []}
        answer = self.model.answer(question, sources)
        cited = set(answer.citation_ids)
        if not cited:
            return {"text": "会议记录中没有足够依据回答该问题。", "citations": []}
        if not cited <= {s.id for s in sources}:
            raise ValueError("Unknown answer citations")
        return {
            "text": answer.text,
            "citations": [s.model_dump() for s in sources if s.id in cited],
        }
