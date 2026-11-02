"""Keyword search over the in-memory document corpus."""

from __future__ import annotations

from typing import Any

from actproto.tools.fixtures import DOCUMENTS


def search_docs(query: str) -> dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        return {"ok": False, "error": {"type": "invalid_query", "message": "query must be a non-empty string"}}
    tokens = [t.lower() for t in query.split() if t]
    hits = []
    for doc in DOCUMENTS:
        blob = f"{doc['title']} {doc['text']}".lower()
        score = sum(1 for t in tokens if t in blob)
        if score:
            hits.append({"id": doc["id"], "title": doc["title"], "text": doc["text"], "score": score})
    hits.sort(key=lambda h: (-h["score"], h["id"]))
    return {"ok": True, "data": {"hits": hits, "count": len(hits)}}
