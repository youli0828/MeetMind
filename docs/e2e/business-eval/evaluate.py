"""Offline evaluation of production hybrid retrieval and saved action-item predictions."""

import argparse
import hashlib
import json
import platform
import unicodedata
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import chromadb
from chromadb.config import Settings as ChromaSettings

from meetmind.models import Minutes, Transcript
from meetmind.offline import OfflineModel
from meetmind.retrieval import HybridRetriever, chunks, tokens
from meetmind.storage import Meeting, Store

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[2]


def normalize(text):
    return "".join(tokens(unicodedata.normalize("NFKC", text or "")))


def action_tokens(items):
    # Field tags stop an owner's name from matching task text or a deadline.
    return {
        (field, token)
        for item in items
        for field in ("task", "owner", "due")
        for token in tokens(unicodedata.normalize("NFKC", item.get(field) or ""))
    }


def set_scores(predicted, gold):
    overlap = len(predicted & gold)
    precision = overlap / len(predicted) if predicted else float(not gold)
    recall = overlap / len(gold) if gold else float(not predicted)
    f1 = 2 * overlap / (len(predicted) + len(gold)) if predicted or gold else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "overlap": overlap,
        "predicted": len(predicted),
        "gold": len(gold),
    }


def action_scores(predicted, gold):
    strict_predicted = {
        tuple(normalize(i.get(f)) for f in ("task", "owner", "due")) for i in predicted
    }
    strict_gold = {tuple(normalize(i.get(f)) for f in ("task", "owner", "due")) for i in gold}
    return {
        "field_token_set": set_scores(action_tokens(predicted), action_tokens(gold)),
        "exact_item_set": set_scores(strict_predicted, strict_gold),
    }


def recall_at_k(found, relevant):
    covered = {(c.meeting_id, segment) for c in found for segment in c.segment_ids}
    expected = {(r["meeting_id"], r["segment_id"]) for r in relevant}
    return len(covered & expected) / len(expected)


def evaluate(gold_path=DIRECTORY / "gold.json"):
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    meetings = []
    provenance = {}
    action_results = []
    for entry in gold["meetings"]:
        for field in ("transcript", "predictions"):
            path = ROOT / entry[field]
            provenance[entry[field]] = hashlib.sha256(path.read_bytes()).hexdigest()
        meeting = Meeting(
            id=entry["id"],
            transcript=Transcript.model_validate_json(
                (ROOT / entry["transcript"]).read_text(encoding="utf-8")
            ),
            minutes=Minutes.model_validate_json(
                (ROOT / entry["predictions"]).read_text(encoding="utf-8")
            ),
        )
        meeting.minutes.validate_sources(meeting.transcript)
        known = {s.id for s in meeting.transcript.segments}
        assert all(set(i["source_segment_ids"]) <= known for i in entry["action_items"])
        meetings.append(meeting)
        action_results.append(
            {
                "meeting_id": entry["id"],
                "prediction_kind": entry["prediction_kind"],
                "predicted_items": len(meeting.minutes.action_items),
                "gold_items": len(entry["action_items"]),
                **action_scores(
                    [i.model_dump() for i in meeting.minutes.action_items], entry["action_items"]
                ),
            }
        )
    with TemporaryDirectory(prefix="meetmind-business-eval-") as directory:
        store = Store(Path(directory))
        client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
        collection = client.create_collection("business-eval", embedding_function=None)
        retriever = HybridRetriever(store, collection, OfflineModel())
        try:
            for meeting in meetings:
                store.add(meeting)
                retriever.index(meeting)
            query_results = []
            for query in gold["queries"]:
                for relevant in query["relevant"]:
                    meeting = next(m for m in meetings if m.id == relevant["meeting_id"])
                    assert relevant["segment_id"] in {s.id for s in meeting.transcript.segments}
                found = retriever.search(query["query"], max(gold["k"]))
                query_results.append(
                    {
                        "id": query["id"],
                        "returned_chunk_ids": [c.id for c in found],
                        "recall": {
                            str(k): recall_at_k(found[:k], query["relevant"]) for k in gold["k"]
                        },
                    }
                )
        finally:
            client.delete_collection("business-eval")
    return {
        "mode": "offline_hash_embeddings_saved_minutes",
        "versions": {"python": platform.python_version(), "chromadb": version("chromadb")},
        "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "input_sha256": provenance,
        "retrieval": {
            "meeting_count": len(meetings),
            "chunk_count": sum(len(chunks(m)) for m in meetings),
            "query_count": len(query_results),
            "recall_at_k": {
                str(k): sum(q["recall"][str(k)] for q in query_results) / len(query_results)
                for k in gold["k"]
            },
            "queries": query_results,
        },
        "action_items": action_results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DIRECTORY / "results.json")
    args = parser.parse_args()
    results = evaluate()
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
