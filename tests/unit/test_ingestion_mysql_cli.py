from __future__ import annotations

import json

import numpy as np
import pandas as pd

from china_auto_market.ingestion.mysql_cli import insert_statements, json_payload, sql_literal


def test_sql_literal_uses_utf8_hex_not_quote_escaping() -> None:
    literal = sql_literal("车系'A\\B\n下一行")

    assert literal.startswith("CONVERT(0x")
    assert literal.endswith(" USING utf8mb4)")
    assert "车系" not in literal
    assert "'A" not in literal


def test_json_payload_normalizes_numpy_and_missing_values() -> None:
    payload = json.loads(
        json_payload({"count": np.int64(2), "missing": np.nan, "when": pd.Timestamp("2026-01-01")})
    )

    assert payload == {"count": 2, "missing": None, "when": "2026-01-01 00:00:00"}


def test_insert_statements_are_chunked_and_validate_row_width() -> None:
    statements = list(
        insert_statements(
            "auto_raw.raw_sales",
            ["batch_id", "source_record_id"],
            [(1, "a"), (1, "b"), (1, "c")],
            chunk_size=2,
        )
    )

    assert len(statements) == 2
    assert statements[0].count("CONVERT(0x") == 2
