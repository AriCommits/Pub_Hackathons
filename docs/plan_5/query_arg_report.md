# Query argument log

Mismatched query_database calls: 39

A blank extra argument counts as missing_field. A non-blank extra argument counts as wrong_value.

- wrong_value: 39
- missing_field: 0
- unfilled: 0

The assistant-only adapter was this run. It did not fix the query arguments. First-tool accuracy fell from 1.000 to 0.317 because the prototypes were fit on the old adapter, and the new calls are `lookup_record` with a `record_id`. `results/adapters/b1` stays the eval adapter.

- `easy_query-19` wrong_value predicted `{"record_id": "REC-001"}` gold `{"product": "washer", "year": 2023}` tool `{"count": null, "ok": true}`
- `easy_query-20` wrong_value predicted `{"record_id": "REC-001"}` gold `{"product": "washer", "year": 2024}` tool `{"count": null, "ok": true}`
- `easy_query-21` wrong_value predicted `{"record_id": "REC-001"}` gold `{"product": "widget", "year": 2022}` tool `{"count": null, "ok": true}`
- `easy_query-22` wrong_value predicted `{"record_id": "REC-001"}` gold `{"product": "widget", "year": 2023}` tool `{"count": null, "ok": true}`
- `easy_query-23` wrong_value predicted `{"record_id": "REC-001"}` gold `{"product": "widget", "year": 2024}` tool `{"count": null, "ok": true}`
- `confusable_sales-19` wrong_value predicted `{"record_id": "REC-2023"}` gold `{"product": "washer", "year": 2023}` tool `{"count": null, "error": {"message": "no record REC-2023", "type": "not_found"}, "ok": false}`
- `confusable_sales-20` wrong_value predicted `{"record_id": "REC-2024"}` gold `{"product": "washer", "year": 2024}` tool `{"count": null, "error": {"message": "no record REC-2024", "type": "not_found"}, "ok": false}`
- `confusable_sales-21` wrong_value predicted `{"record_id": "REC-2022"}` gold `{"product": "widget", "year": 2022}` tool `{"count": null, "error": {"message": "no record REC-2022", "type": "not_found"}, "ok": false}`
- `confusable_sales-22` wrong_value predicted `{"record_id": "REC-2023"}` gold `{"product": "widget", "year": 2023}` tool `{"count": null, "error": {"message": "no record REC-2023", "type": "not_found"}, "ok": false}`
- `confusable_sales-23` wrong_value predicted `{"record_id": "REC-2024"}` gold `{"product": "widget", "year": 2024}` tool `{"count": null, "error": {"message": "no record REC-2024", "type": "not_found"}, "ok": false}`
- `multi_growth-6` wrong_value predicted `{"record_id": "REC-2023"}` gold `{"product": "washer"}` tool `{"count": null, "error": {"message": "no record REC-2023", "type": "not_found"}, "ok": false}`
- `multi_growth-7` wrong_value predicted `{"record_id": "REC-2023-2024-2025"}` gold `{"product": "widget"}` tool `{"count": null, "error": {"message": "no record REC-2023-2024-2025", "type": "not_found"}, "ok": false}`
- `error_missing_product-4` wrong_value predicted `{"record_id": "REC-unobtainium"}` gold `{"product": "unobtainium"}` tool `{"count": null, "error": {"message": "no record REC-unobtainium", "type": "not_found"}, "ok": false}`
- `easy_query-19-p0` wrong_value predicted `{"record_id": "REC-001"}` gold `{"product": "washer", "year": 2023}` tool `{"count": null, "ok": true}`
- `easy_query-20-p0` wrong_value predicted `{"record_id": "REC-001"}` gold `{"product": "washer", "year": 2024}` tool `{"count": null, "ok": true}`
- `easy_query-21-p0` wrong_value predicted `{"record_id": "REC-001"}` gold `{"product": "widget", "year": 2022}` tool `{"count": null, "ok": true}`
- `easy_query-22-p0` wrong_value predicted `{"record_id": "REC-001"}` gold `{"product": "widget", "year": 2023}` tool `{"count": null, "ok": true}`
- `easy_query-23-p0` wrong_value predicted `{"record_id": "REC-001"}` gold `{"product": "widget", "year": 2024}` tool `{"count": null, "ok": true}`
- `easy_query-19-p1` wrong_value predicted `{"record_id": "REC-001"}` gold `{"product": "washer", "year": 2023}` tool `{"count": null, "ok": true}`
- `easy_query-20-p1` wrong_value predicted `{"record_id": "REC-001"}` gold `{"product": "washer", "year": 2024}` tool `{"count": null, "ok": true}`
- `easy_query-21-p1` wrong_value predicted `{"record_id": "REC-001"}` gold `{"product": "widget", "year": 2022}` tool `{"count": null, "ok": true}`
- `easy_query-22-p1` wrong_value predicted `{"record_id": "REC-001"}` gold `{"product": "widget", "year": 2023}` tool `{"count": null, "ok": true}`
- `easy_query-23-p1` wrong_value predicted `{"record_id": "REC-001"}` gold `{"product": "widget", "year": 2024}` tool `{"count": null, "ok": true}`
- `confusable_sales-19-p0` wrong_value predicted `{"record_id": "REC-2023"}` gold `{"product": "washer", "year": 2023}` tool `{"count": null, "error": {"message": "no record REC-2023", "type": "not_found"}, "ok": false}`
- `confusable_sales-20-p0` wrong_value predicted `{"record_id": "REC-2024"}` gold `{"product": "washer", "year": 2024}` tool `{"count": null, "error": {"message": "no record REC-2024", "type": "not_found"}, "ok": false}`
- `confusable_sales-21-p0` wrong_value predicted `{"record_id": "REC-2022"}` gold `{"product": "widget", "year": 2022}` tool `{"count": null, "error": {"message": "no record REC-2022", "type": "not_found"}, "ok": false}`
- `confusable_sales-22-p0` wrong_value predicted `{"record_id": "REC-2023"}` gold `{"product": "widget", "year": 2023}` tool `{"count": null, "error": {"message": "no record REC-2023", "type": "not_found"}, "ok": false}`
- `confusable_sales-23-p0` wrong_value predicted `{"record_id": "REC-2024"}` gold `{"product": "widget", "year": 2024}` tool `{"count": null, "error": {"message": "no record REC-2024", "type": "not_found"}, "ok": false}`
- `confusable_sales-19-p1` wrong_value predicted `{"record_id": "REC-2023"}` gold `{"product": "washer", "year": 2023}` tool `{"count": null, "error": {"message": "no record REC-2023", "type": "not_found"}, "ok": false}`
- `confusable_sales-20-p1` wrong_value predicted `{"record_id": "REC-2024"}` gold `{"product": "washer", "year": 2024}` tool `{"count": null, "error": {"message": "no record REC-2024", "type": "not_found"}, "ok": false}`
- `confusable_sales-21-p1` wrong_value predicted `{"record_id": "REC-2022"}` gold `{"product": "widget", "year": 2022}` tool `{"count": null, "error": {"message": "no record REC-2022", "type": "not_found"}, "ok": false}`
- `confusable_sales-22-p1` wrong_value predicted `{"record_id": "REC-2023"}` gold `{"product": "widget", "year": 2023}` tool `{"count": null, "error": {"message": "no record REC-2023", "type": "not_found"}, "ok": false}`
- `confusable_sales-23-p1` wrong_value predicted `{"record_id": "REC-2024"}` gold `{"product": "widget", "year": 2024}` tool `{"count": null, "error": {"message": "no record REC-2024", "type": "not_found"}, "ok": false}`
- `multi_growth-6-p0` wrong_value predicted `{"record_id": "REC-2023"}` gold `{"product": "washer"}` tool `{"count": null, "error": {"message": "no record REC-2023", "type": "not_found"}, "ok": false}`
- `multi_growth-7-p0` wrong_value predicted `{"record_id": "REC-2023-2024-2025"}` gold `{"product": "widget"}` tool `{"count": null, "error": {"message": "no record REC-2023-2024-2025", "type": "not_found"}, "ok": false}`
- `multi_growth-6-p1` wrong_value predicted `{"record_id": "REC-001"}` gold `{"product": "washer"}` tool `{"count": null, "ok": true}`
- `multi_growth-7-p1` wrong_value predicted `{"record_id": "REC-001"}` gold `{"product": "widget"}` tool `{"count": null, "ok": true}`
- `error_missing_product-4-p0` wrong_value predicted `{"record_id": "REC-unobtainium"}` gold `{"product": "unobtainium"}` tool `{"count": null, "error": {"message": "no record REC-unobtainium", "type": "not_found"}, "ok": false}`
- `error_missing_product-4-p1` wrong_value predicted `{"record_id": "REC-unobtainium"}` gold `{"product": "unobtainium"}` tool `{"count": null, "error": {"message": "no record REC-unobtainium", "type": "not_found"}, "ok": false}`
