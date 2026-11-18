# Query argument log

Mismatched query_database calls: 39

A blank extra argument counts as missing_field. A non-blank extra argument counts as wrong_value.

- wrong_value: 21
- missing_field: 15
- unfilled: 3

The next experiment is an assistant-only adapter. This plan does not train it.

- `easy_query-19` missing_field predicted `{"product": "washer", "region": "", "year": 2023}` gold `{"product": "washer", "year": 2023}` tool `{"count": 0, "ok": true}`
- `easy_query-20` missing_field predicted `{"product": "washer", "region": "", "year": 2024}` gold `{"product": "washer", "year": 2024}` tool `{"count": 0, "ok": true}`
- `easy_query-21` missing_field predicted `{"product": "widget", "region": "", "year": 2022}` gold `{"product": "widget", "year": 2022}` tool `{"count": 0, "ok": true}`
- `easy_query-22` missing_field predicted `{"product": "widget", "region": "", "year": 2023}` gold `{"product": "widget", "year": 2023}` tool `{"count": 0, "ok": true}`
- `easy_query-23` missing_field predicted `{"product": "widget", "region": "", "year": 2024}` gold `{"product": "widget", "year": 2024}` tool `{"count": 0, "ok": true}`
- `confusable_sales-19` wrong_value predicted `{"product": "washing machine", "region": "NA", "year": 2023}` gold `{"product": "washer", "year": 2023}` tool `{"count": 0, "ok": true}`
- `confusable_sales-20` wrong_value predicted `{"product": "washing machine", "region": "NA", "year": 2024}` gold `{"product": "washer", "year": 2024}` tool `{"count": 0, "ok": true}`
- `confusable_sales-21` wrong_value predicted `{"product": "widget", "region": "europe", "year": 2022}` gold `{"product": "widget", "year": 2022}` tool `{"count": 0, "ok": true}`
- `confusable_sales-22` wrong_value predicted `{"product": "widget", "region": "europe", "year": 2023}` gold `{"product": "widget", "year": 2023}` tool `{"count": 0, "ok": true}`
- `confusable_sales-23` wrong_value predicted `{"product": "widget", "region": "europe", "year": 2024}` gold `{"product": "widget", "year": 2024}` tool `{"count": 0, "ok": true}`
- `multi_growth-6` wrong_value predicted `{"product": "washing machine", "region": "NA", "year": 2022}` gold `{"product": "washer"}` tool `{"count": 0, "ok": true}`
- `multi_growth-7` wrong_value predicted `{"product": "widget", "region": "NA", "year": 2022}` gold `{"product": "widget"}` tool `{"count": 1, "ok": true}`
- `error_missing_product-4` unfilled predicted `{}` gold `{"product": "unobtainium"}` tool `{"count": 24, "ok": true}`
- `easy_query-19-p0` missing_field predicted `{"product": "washer", "region": "", "year": 2023}` gold `{"product": "washer", "year": 2023}` tool `{"count": 0, "ok": true}`
- `easy_query-20-p0` missing_field predicted `{"product": "washer", "region": "", "year": 2024}` gold `{"product": "washer", "year": 2024}` tool `{"count": 0, "ok": true}`
- `easy_query-21-p0` missing_field predicted `{"product": "widget", "region": "", "year": 2022}` gold `{"product": "widget", "year": 2022}` tool `{"count": 0, "ok": true}`
- `easy_query-22-p0` missing_field predicted `{"product": "widget", "region": "", "year": 2023}` gold `{"product": "widget", "year": 2023}` tool `{"count": 0, "ok": true}`
- `easy_query-23-p0` missing_field predicted `{"product": "widget", "region": "", "year": 2024}` gold `{"product": "widget", "year": 2024}` tool `{"count": 0, "ok": true}`
- `easy_query-19-p1` missing_field predicted `{"product": "washer", "region": "", "year": 2023}` gold `{"product": "washer", "year": 2023}` tool `{"count": 0, "ok": true}`
- `easy_query-20-p1` missing_field predicted `{"product": "washer", "region": "", "year": 2024}` gold `{"product": "washer", "year": 2024}` tool `{"count": 0, "ok": true}`
- `easy_query-21-p1` missing_field predicted `{"product": "widget", "region": "", "year": 2022}` gold `{"product": "widget", "year": 2022}` tool `{"count": 0, "ok": true}`
- `easy_query-22-p1` missing_field predicted `{"product": "widget", "region": "", "year": 2023}` gold `{"product": "widget", "year": 2023}` tool `{"count": 0, "ok": true}`
- `easy_query-23-p1` missing_field predicted `{"product": "widget", "region": "", "year": 2024}` gold `{"product": "widget", "year": 2024}` tool `{"count": 0, "ok": true}`
- `confusable_sales-19-p0` wrong_value predicted `{"product": "washing machine", "region": "NA", "year": 2023}` gold `{"product": "washer", "year": 2023}` tool `{"count": 0, "ok": true}`
- `confusable_sales-20-p0` wrong_value predicted `{"product": "washing machine", "region": "NA", "year": 2024}` gold `{"product": "washer", "year": 2024}` tool `{"count": 0, "ok": true}`
- `confusable_sales-21-p0` wrong_value predicted `{"product": "widget", "region": "europe", "year": 2022}` gold `{"product": "widget", "year": 2022}` tool `{"count": 0, "ok": true}`
- `confusable_sales-22-p0` wrong_value predicted `{"product": "widget", "region": "europe", "year": 2023}` gold `{"product": "widget", "year": 2023}` tool `{"count": 0, "ok": true}`
- `confusable_sales-23-p0` wrong_value predicted `{"product": "widget", "region": "europe", "year": 2024}` gold `{"product": "widget", "year": 2024}` tool `{"count": 0, "ok": true}`
- `confusable_sales-19-p1` wrong_value predicted `{"product": "washing machine", "region": "NA", "year": 2023}` gold `{"product": "washer", "year": 2023}` tool `{"count": 0, "ok": true}`
- `confusable_sales-20-p1` wrong_value predicted `{"product": "washing machine", "region": "NA", "year": 2024}` gold `{"product": "washer", "year": 2024}` tool `{"count": 0, "ok": true}`
- `confusable_sales-21-p1` wrong_value predicted `{"product": "widget", "region": "europe", "year": 2022}` gold `{"product": "widget", "year": 2022}` tool `{"count": 0, "ok": true}`
- `confusable_sales-22-p1` wrong_value predicted `{"product": "widget", "region": "europe", "year": 2023}` gold `{"product": "widget", "year": 2023}` tool `{"count": 0, "ok": true}`
- `confusable_sales-23-p1` wrong_value predicted `{"product": "widget", "region": "europe", "year": 2024}` gold `{"product": "widget", "year": 2024}` tool `{"count": 0, "ok": true}`
- `multi_growth-6-p0` wrong_value predicted `{"product": "washing machine", "region": "NA", "year": 2023}` gold `{"product": "washer"}` tool `{"count": 0, "ok": true}`
- `multi_growth-7-p0` wrong_value predicted `{"product": "widget", "region": "europe", "year": 2022}` gold `{"product": "widget"}` tool `{"count": 0, "ok": true}`
- `multi_growth-6-p1` wrong_value predicted `{"product": "washing machine", "region": "NA", "year": 2022}` gold `{"product": "washer"}` tool `{"count": 0, "ok": true}`
- `multi_growth-7-p1` wrong_value predicted `{"product": "widget", "region": "europe", "year": 2022}` gold `{"product": "widget"}` tool `{"count": 0, "ok": true}`
- `error_missing_product-4-p0` unfilled predicted `{}` gold `{"product": "unobtainium"}` tool `{"count": 24, "ok": true}`
- `error_missing_product-4-p1` unfilled predicted `{}` gold `{"product": "unobtainium"}` tool `{"count": 24, "ok": true}`
