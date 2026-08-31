from __future__ import annotations

import json
from types import SimpleNamespace

from china_auto_market.warehouse import schema


def test_query_json_rows_uses_raw_batch_mode_and_parses_objects(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(
            stdout='{"text":"line\\nvalue"}\n{"text":"left\u2028right"}\n',
            stderr="",
        )

    monkeypatch.setattr(schema.subprocess, "run", fake_run)
    monkeypatch.setattr(schema, "resolve_mysql_binary", lambda _: "/mysql")
    rows = schema.query_json_rows("safe-path", "SELECT payload;")

    assert rows == [{"text": "line\nvalue"}, {"text": "left\u2028right"}]
    assert "--raw" in observed["command"]
    assert "--skip-column-names" in observed["command"]
    assert observed["kwargs"]["check"] is True


def test_query_json_rows_rejects_non_object(monkeypatch) -> None:
    monkeypatch.setattr(
        schema.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps([1, 2]) + "\n", stderr=""),
    )
    monkeypatch.setattr(schema, "resolve_mysql_binary", lambda _: "/mysql")
    try:
        schema.query_json_rows("safe-path", "SELECT payload;")
    except ValueError as error:
        assert "JSON object" in str(error)
    else:
        raise AssertionError("non-object JSON must fail")
