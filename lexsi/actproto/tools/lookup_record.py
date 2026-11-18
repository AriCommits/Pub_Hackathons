"""Fetch one record by id. Missing ids return an explicit error object."""

from __future__ import annotations

from typing import Any

from actproto.tools.fixtures import RECORDS


def lookup_record(record_id: str) -> dict[str, Any]:
    if not isinstance(record_id, str) or not record_id.strip():
        return {"ok": False, "error": {"type": "invalid_id", "message": "record_id must be a non-empty string"}}
    for row in RECORDS:
        if str(row["id"]) == record_id:
            return {"ok": True, "data": dict(row)}
    return {"ok": False, "error": {"type": "not_found", "message": f"no record {record_id}"}}
