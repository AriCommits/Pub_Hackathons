"""Filter the fixed sales table. Missing filters that match nothing return ok with empty rows."""

from __future__ import annotations

from typing import Any

from actproto.tools.fixtures import SALES_ROWS


def query_database(
    product: str | None = None,
    region: str | None = None,
    year: int | None = None,
) -> dict[str, Any]:
    rows = list(SALES_ROWS)
    if product is not None:
        if not isinstance(product, str):
            return {"ok": False, "error": {"type": "invalid_filter", "message": "product must be a string"}}
        rows = [r for r in rows if str(r["product"]).lower() == product.lower()]
    if region is not None:
        if not isinstance(region, str):
            return {"ok": False, "error": {"type": "invalid_filter", "message": "region must be a string"}}
        rows = [r for r in rows if str(r["region"]).lower() == region.lower()]
    if year is not None:
        if not isinstance(year, int):
            return {"ok": False, "error": {"type": "invalid_filter", "message": "year must be an integer"}}
        rows = [r for r in rows if int(r["year"]) == year]
    return {"ok": True, "data": {"rows": rows, "count": len(rows)}}
