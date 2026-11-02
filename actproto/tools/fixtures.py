"""Deterministic local corpora: docs, sales rows, and lookup records."""

from __future__ import annotations

DOCUMENTS: list[dict[str, str]] = [
    {
        "id": "doc-widget-spec",
        "title": "Widget specification",
        "text": "The widget is a fastening component used in NA plants. Maintenance interval is 90 days.",
    },
    {
        "id": "doc-gadget-spec",
        "title": "Gadget specification",
        "text": "The gadget is an EU-region sensor pack. Calibration is required after 200 hours.",
    },
    {
        "id": "doc-sprocket-spec",
        "title": "Sprocket specification",
        "text": "The sprocket is an APAC drivetrain part. It is not a sales figure; see the database for revenue.",
    },
    {
        "id": "doc-policy-returns",
        "title": "Returns policy",
        "text": "Returns are accepted within 30 days with a valid REC record id such as REC-001.",
    },
    {
        "id": "doc-calc-help",
        "title": "How to compute growth",
        "text": "Average annual growth is the mean of year-over-year percentage changes. Use the calculator after querying sales.",
    },
]

SALES_ROWS: list[dict[str, object]] = [
    {"product": "widget", "region": "NA", "year": 2022, "sales": 100000},
    {"product": "widget", "region": "NA", "year": 2023, "sales": 110000},
    {"product": "widget", "region": "NA", "year": 2024, "sales": 121000},
    {"product": "gadget", "region": "EU", "year": 2022, "sales": 80000},
    {"product": "gadget", "region": "EU", "year": 2023, "sales": 88000},
    {"product": "gadget", "region": "EU", "year": 2024, "sales": 92400},
    {"product": "sprocket", "region": "APAC", "year": 2022, "sales": 50000},
    {"product": "sprocket", "region": "APAC", "year": 2023, "sales": 45000},
    {"product": "sprocket", "region": "APAC", "year": 2024, "sales": 54000},
    {"product": "gizmo", "region": "NA", "year": 2022, "sales": 20000},
    {"product": "gizmo", "region": "NA", "year": 2023, "sales": 25000},
    {"product": "gizmo", "region": "NA", "year": 2024, "sales": 30000},
    {"product": "doodad", "region": "EU", "year": 2022, "sales": 12000},
    {"product": "doodad", "region": "EU", "year": 2023, "sales": 15000},
    {"product": "doodad", "region": "EU", "year": 2024, "sales": 18000},
    {"product": "bolt", "region": "APAC", "year": 2022, "sales": 9000},
    {"product": "bolt", "region": "APAC", "year": 2023, "sales": 9900},
    {"product": "bolt", "region": "APAC", "year": 2024, "sales": 10890},
    {"product": "hinge", "region": "NA", "year": 2022, "sales": 33000},
    {"product": "hinge", "region": "NA", "year": 2023, "sales": 36000},
    {"product": "hinge", "region": "NA", "year": 2024, "sales": 39000},
    {"product": "washer", "region": "EU", "year": 2022, "sales": 7000},
    {"product": "washer", "region": "EU", "year": 2023, "sales": 7700},
    {"product": "washer", "region": "EU", "year": 2024, "sales": 8470},
]

RECORDS: list[dict[str, object]] = [
    {"id": "REC-001", "sku": "A", "product": "widget", "in_stock": True, "price": 12.5},
    {"id": "REC-002", "sku": "B", "product": "gadget", "in_stock": False, "price": 40.0},
    {"id": "REC-003", "sku": "C", "product": "sprocket", "in_stock": True, "price": 8.25},
    {"id": "REC-004", "sku": "D", "product": "gizmo", "in_stock": True, "price": 19.99},
    {"id": "REC-005", "sku": "E", "product": "doodad", "in_stock": True, "price": 5.0},
    {"id": "REC-006", "sku": "F", "product": "bolt", "in_stock": False, "price": 0.4},
    {"id": "REC-007", "sku": "G", "product": "hinge", "in_stock": True, "price": 3.1},
    {"id": "REC-008", "sku": "H", "product": "washer", "in_stock": True, "price": 0.15},
]

PRODUCTS = sorted({str(row["product"]) for row in SALES_ROWS})
RECORD_IDS = [str(row["id"]) for row in RECORDS]
KNOWN_MISSING_IDS = ["REC-404", "REC-999", "REC-000"]
