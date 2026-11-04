import hashlib
import json

from openai.types.chat import ChatCompletionMessage

from meetmind.retrieval import Answer, tokens
from meetmind.tools.export import timestamp


class OfflineModel:
    """Deterministic demonstration only: hashed tokens and verbatim excerpts, no inference."""

    def embed(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * 64
            for token in tokens(text):
                index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % 64
                vector[index] += 1
            norm = sum(v * v for v in vector) ** 0.5 or 1
            vectors.append([v / norm for v in vector])
        return vectors

    def answer(self, question, sources):
        return Answer(
            text="[离线演示：原文摘录，并非模型回答]\n" + sources[0].text,
            citation_ids=[sources[0].id],
        )

    def chat(self, messages, tools):
        if messages[-1]["role"] == "tool":
            sources = json.loads(messages[-1]["content"]).get("sources", [])
            excerpts = "\n\n".join(
                f"{s['title']} [{timestamp(s['start'])}–{timestamp(s['end'])}]\n{s['text']}"
                for s in sources
            )
            content = "[离线演示：固定搜索流程，并非模型推理]\n" + (excerpts or "没有会议记录。")
            return ChatCompletionMessage(role="assistant", content=content)
        return ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "offline-search",
                    "type": "function",
                    "function": {
                        "name": "search_meetings",
                        "arguments": json.dumps({"query": messages[-1]["content"], "k": 3}),
                    },
                }
            ],
        )
