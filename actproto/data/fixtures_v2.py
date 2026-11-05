"""Fixture-backed tool environment, separate from the historical global fixtures."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import random
from actproto.experiments.manifest import hash_object
from actproto.tools.registry import call_tool
from actproto.tools.schemas import validate_args

ENVIRONMENT_ID = "tooluse_v2.fixtures.1"


@dataclass(frozen=True)
class FixtureEnvironment:
    identifier: str
    records: tuple[dict, ...]
    sales: tuple[dict, ...]
    documents: tuple[dict, ...]

    @cached_property
    def digest(self):
        return hash_object({"identifier": self.identifier, "records": self.records,
                            "sales": self.sales, "documents": self.documents})

    def call(self, environment_id, name, arguments):
        if environment_id != self.identifier:
            raise ValueError("unknown fixture environment")
        ok, reason = validate_args(name, arguments)
        if not ok:
            return {"ok": False, "error": {"type": "invalid_arguments", "message": reason}}
        if name in {"calculator", "finish", "no_tool"}:
            return call_tool(name, arguments)
        if name == "lookup_record":
            found = next((row for row in self.records if row["id"] == arguments["record_id"]), None)
            return ({"ok": True, "data": dict(found)} if found else
                    {"ok": False, "error": {"type": "not_found", "message": f"no record {arguments['record_id']}"}})
        if name == "query_database":
            rows = [dict(row) for row in self.sales if all(
                str(row[k]).lower() == str(v).lower() for k, v in arguments.items())]
            return {"ok": True, "data": {"rows": rows, "count": len(rows)}}
        if name == "search_docs":
            query = arguments["query"]
            if not query.strip():
                return {"ok": False, "error": {"type": "invalid_query", "message": "query must be a non-empty string"}}
            tokens = query.lower().split()
            hits = []
            for doc in self.documents:
                blob = f"{doc['title']} {doc['text']}".lower()
                score = sum(token in blob for token in tokens)
                if score:
                    hits.append({**doc, "score": score})
            hits.sort(key=lambda hit: (-hit["score"], hit["id"]))
            return {"ok": True, "data": {"hits": hits, "count": len(hits)}}
        return {"ok": False, "error": {"type": "unknown_tool", "message": f"unknown tool {name}"}}


def build_environment(count=1800, seed=42):
    if not 0 <= count <= 1_000_000:
        raise ValueError("fixture count outside identifier pool")
    # Independent shuffled, disjoint pools prevent record, product, and document
    # IDs from revealing one another; separate RNG streams decouple their values.
    record_ids = random.Random(f"{seed}:record").sample(range(1_000_000, 2_000_000), count)
    product_ids = random.Random(f"{seed}:product").sample(range(2_000_000, 3_000_000), count)
    document_ids = random.Random(f"{seed}:document").sample(range(3_000_000, 4_000_000), count)
    values = random.Random(f"{seed}:values")
    records, sales, documents = [], [], []
    for i in range(count):
        product, record = f"item{product_ids[i]}", f"V2-{record_ids[i]}"
        year, amount = values.randint(2010, 2025), values.randint(1000, 9999)
        region = values.choice(("north", "south", "west", "east"))
        days = values.randint(20, 90)
        records.append({"id": record, "product": product, "region": region, "year": year})
        sales.append({"product": product, "region": region, "year": year, "sales": amount})
        documents.append({"id": f"doc-{document_ids[i]}", "title": f"Guide {product}",
                          "text": f"The service interval for {product} is {days} days."})
    return FixtureEnvironment(ENVIRONMENT_ID, tuple(records), tuple(sales), tuple(documents))
