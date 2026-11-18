"""Task template families. Instantiated by the generator with split-safe entities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from actproto.data.oracles import yoy_growth
from actproto.data.types import Oracle, Task
from actproto.tools.fixtures import DOCUMENTS, RECORDS, SALES_ROWS
from actproto.tools.registry import call_tool

NO_TOOL_FACTS: list[tuple[str, str]] = [
    ("France", "Paris"),
    ("Japan", "Tokyo"),
    ("Kenya", "Nairobi"),
    ("Canada", "Ottawa"),
    ("Brazil", "Brasilia"),
    ("Peru", "Lima"),
    ("Egypt", "Cairo"),
    ("Norway", "Oslo"),
    ("Poland", "Warsaw"),
    ("Chile", "Santiago"),
    ("Ghana", "Accra"),
    ("Spain", "Madrid"),
    ("Sweden", "Stockholm"),
    ("Ireland", "Dublin"),
    ("Greece", "Athens"),
]


def _sales_for(product: str) -> list[dict[str, Any]]:
    return sorted(
        [r for r in SALES_ROWS if str(r["product"]) == product],
        key=lambda r: int(r["year"]),
    )


def easy_calc(entity: dict[str, Any], idx: int) -> Task:
    a, b = int(entity["a"]), int(entity["b"])
    expr = f"{a} * {b}"
    value = float(a * b)
    return Task(
        id=f"easy_calc-{idx}",
        bucket="easy",
        template_family="easy_calc",
        entity_ids=[entity["id"]],
        prompt=f"Calculate {expr}. Use the calculator tool, then finish with the numeric result.",
        gold_tools=["calculator", "finish"],
        gold_args=[{"expression": expr}, {"answer": str(int(value) if value.is_integer() else value)}],
        oracle=Oracle(final_state={"done": True, "answer": str(int(value))}, answer=str(int(value))),
    )


def easy_lookup(entity: dict[str, Any], idx: int) -> Task:
    rec = entity["record"]
    result = call_tool("lookup_record", {"record_id": rec["id"]})
    price = result["data"]["price"]
    return Task(
        id=f"easy_lookup-{idx}",
        bucket="easy",
        template_family="easy_lookup",
        entity_ids=[rec["id"]],
        prompt=f"Look up record {rec['id']} and finish with its price.",
        gold_tools=["lookup_record", "finish"],
        gold_args=[{"record_id": rec["id"]}, {"answer": str(price)}],
        oracle=Oracle(final_state={"done": True, "answer": str(price)}, answer=str(price)),
    )


def easy_search(entity: dict[str, Any], idx: int) -> Task:
    doc = entity["doc"]
    return Task(
        id=f"easy_search-{idx}",
        bucket="easy",
        template_family="easy_search",
        entity_ids=[doc["id"]],
        prompt=f"Search the docs for '{doc['title']}' and finish with the document id you found.",
        gold_tools=["search_docs", "finish"],
        gold_args=[{"query": doc["title"]}, {"answer": doc["id"]}],
        oracle=Oracle(final_state={"done": True, "answer": doc["id"]}, answer=doc["id"]),
    )


def easy_query(entity: dict[str, Any], idx: int) -> Task:
    product = entity["product"]
    year = int(entity["year"])
    rows = [r for r in _sales_for(product) if int(r["year"]) == year]
    sales = int(rows[0]["sales"]) if rows else 0
    return Task(
        id=f"easy_query-{idx}",
        bucket="easy",
        template_family="easy_query",
        entity_ids=[f"{product}:{year}"],
        prompt=f"Query the sales database for {product} in {year} and finish with the sales number.",
        gold_tools=["query_database", "finish"],
        gold_args=[{"product": product, "year": year}, {"answer": str(sales)}],
        oracle=Oracle(final_state={"done": True, "answer": str(sales)}, answer=str(sales)),
    )


def confusable_sales_number(entity: dict[str, Any], idx: int) -> Task:
    """Mentions arithmetic and sales; gold is query_database, not calculator."""
    product = entity["product"]
    year = int(entity["year"])
    rows = [r for r in _sales_for(product) if int(r["year"]) == year]
    sales = int(rows[0]["sales"]) if rows else 0
    return Task(
        id=f"confusable_sales-{idx}",
        bucket="confusable",
        template_family="confusable_sales_number",
        entity_ids=[f"{product}:{year}"],
        prompt=(
            f"What were {product} sales in {year}? Do not estimate or multiply list prices. "
            f"Use the database, then finish with the recorded sales figure."
        ),
        gold_tools=["query_database", "finish"],
        gold_args=[{"product": product, "year": year}, {"answer": str(sales)}],
        oracle=Oracle(final_state={"done": True, "answer": str(sales)}, answer=str(sales)),
    )


def confusable_record_vs_search(entity: dict[str, Any], idx: int) -> Task:
    rec = entity["record"]
    result = call_tool("lookup_record", {"record_id": rec["id"]})
    sku = result["data"]["sku"]
    return Task(
        id=f"confusable_record-{idx}",
        bucket="confusable",
        template_family="confusable_record_vs_search",
        entity_ids=[rec["id"]],
        prompt=(
            f"Find the SKU for inventory record {rec['id']}. This is a structured record lookup, "
            f"not a document search. Finish with the SKU."
        ),
        gold_tools=["lookup_record", "finish"],
        gold_args=[{"record_id": rec["id"]}, {"answer": str(sku)}],
        oracle=Oracle(final_state={"done": True, "answer": str(sku)}, answer=str(sku)),
    )


def multi_avg_growth(entity: dict[str, Any], idx: int) -> Task:
    product = entity["product"]
    rows = _sales_for(product)
    values = [float(r["sales"]) for r in rows]
    growth = yoy_growth(values)
    answer = f"{growth:.4f}"
    return Task(
        id=f"multi_growth-{idx}",
        bucket="multi_step",
        template_family="multi_avg_growth",
        entity_ids=[product],
        prompt=(
            f"Look up the last three years of {product} sales in the database, then calculate "
            f"the average year-over-year growth rate (as a decimal). Finish with the number."
        ),
        gold_tools=["query_database", "calculator", "finish"],
        gold_args=[
            {"product": product},
            {"expression": _growth_expression(values)},
            {"answer": answer},
        ],
        oracle=Oracle(final_state={"done": True, "answer": answer}, answer=answer, numeric_tolerance=1e-3),
        metadata={"sales": values},
    )


def _growth_expression(values: list[float]) -> str:
    parts = [f"(({values[i]}-{values[i-1]})/{values[i-1]})" for i in range(1, len(values))]
    return "(" + "+".join(parts) + f")/{len(parts)}"


def no_tool_capital(entity: dict[str, Any], idx: int) -> Task:
    country, capital = entity["country"], entity["capital"]
    return Task(
        id=f"no_tool_capital-{idx}",
        bucket="no_tool",
        template_family="no_tool_capital",
        entity_ids=[country],
        prompt=(
            f"The capital of {country} is {capital}. What is the capital of {country}? "
            f"Do not use tools. Call finish with the capital."
        ),
        gold_tools=["finish"],
        gold_args=[{"answer": capital}],
        oracle=Oracle(final_state={"done": True, "answer": capital}, answer=capital),
    )


def error_missing_record(entity: dict[str, Any], idx: int) -> Task:
    missing = entity["missing_id"]
    return Task(
        id=f"error_missing-{idx}",
        bucket="error_recovery",
        template_family="error_missing_record",
        entity_ids=[missing],
        prompt=(
            f"Look up record {missing}. If the tool returns an error, do not invent a price. "
            f"Finish with the exact text NOT_FOUND."
        ),
        gold_tools=["lookup_record", "finish"],
        gold_args=[{"record_id": missing}, {"answer": "NOT_FOUND"}],
        oracle=Oracle(
            final_state={"done": True, "answer": "NOT_FOUND", "lookup_error": True},
            answer="NOT_FOUND",
        ),
    )


def error_missing_product(entity: dict[str, Any], idx: int) -> Task:
    product = entity["product"]
    return Task(
        id=f"error_missing_product-{idx}",
        bucket="error_recovery",
        template_family="error_missing_product",
        entity_ids=[product],
        prompt=(
            f"Query sales for product '{product}'. If there are no rows, finish with NO_ROWS. "
            f"Do not guess numbers."
        ),
        gold_tools=["query_database", "finish"],
        gold_args=[{"product": product}, {"answer": "NO_ROWS"}],
        oracle=Oracle(final_state={"done": True, "answer": "NO_ROWS"}, answer="NO_ROWS"),
    )


FAMILIES: dict[str, Callable[[dict[str, Any], int], Task]] = {
    "easy_calc": easy_calc,
    "easy_lookup": easy_lookup,
    "easy_search": easy_search,
    "easy_query": easy_query,
    "confusable_sales_number": confusable_sales_number,
    "confusable_record_vs_search": confusable_record_vs_search,
    "multi_avg_growth": multi_avg_growth,
    "no_tool_capital": no_tool_capital,
    "error_missing_record": error_missing_record,
    "error_missing_product": error_missing_product,
}

# Families that consume product names from the sales table.
PRODUCT_FAMILIES = ("easy_query", "confusable_sales_number", "multi_avg_growth")
RECORD_FAMILIES = ("easy_lookup", "confusable_record_vs_search")
