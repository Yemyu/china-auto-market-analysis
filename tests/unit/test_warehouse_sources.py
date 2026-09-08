from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from china_auto_market.warehouse import schema
from china_auto_market.warehouse import sources


def test_configuration_bound_batch_does_not_query_latest(monkeypatch):
    queries = []

    def query(login, sql):
        queries.append(sql)
        return [{"series_name": "A", "year": 2024}]

    monkeypatch.setattr(sources, "query_json_rows", query)
    result = sources.load_raw_configuration("local", batch_id=7)
    assert len(queries) == 1
    assert "c.batch_id = 7" in queries[0]
    assert "MAX(batch_id)" not in queries[0]
    assert result.attrs["configuration_batch_id"] == 7


@pytest.mark.parametrize("batch_id", [True, 0, -1, 1.5, "7; SELECT 1"])
def test_invalid_configuration_batch_is_rejected(batch_id):
    with pytest.raises(ValueError, match="batch"):
        sources.load_raw_configuration("local", batch_id=batch_id)


def test_forecast_consumer_propagates_bound_batch(monkeypatch):
    from china_auto_market.forecasting.core import load_configuration_source

    observed = {}

    def loader(login_path, batch_id):
        observed.update(login_path=login_path, batch_id=batch_id)
        return pd.DataFrame()

    monkeypatch.setattr(sources, "load_raw_configuration", loader)
    frame = pd.DataFrame()
    frame.attrs["configuration_batch_id"] = 7
    load_configuration_source(backend="mysql", login_path="test-local", frame=frame)
    assert observed == {"login_path": "test-local", "batch_id": 7}


@pytest.mark.parametrize("payload", [
    {"configuration_policy": "raw-batch-reference-v1"},
    {"configuration_policy": "unknown", "configuration_batch_id": 4},
    {"configuration_policy": "raw-batch-reference-v1", "configuration_batch_id": -1},
])
def test_mart_rejects_incomplete_or_invalid_configuration_reference(monkeypatch, payload):
    monkeypatch.setattr(sources, "query_json_rows", lambda *args: [{"date": "2026-01-01", **payload}])
    with pytest.raises(ValueError, match="configuration"):
        sources.load_forecast_feature_mart("test")


def test_mart_reference_binds_config_and_preserves_split_metadata(monkeypatch):
    from china_auto_market.forecasting import core

    rows = [{"date": "2026-01-01", "configuration_policy": "raw-batch-reference-v1",
             "configuration_batch_id": 4, "split": "test",
             **dict.fromkeys(["series_name", "series_id", "year", "month", "brand", "category",
                              "category_en", "monthly_sales", *core.LAG_COLS, *core.CAL, *core.SEASONAL_LAG_COLS], 1)}]
    monkeypatch.setattr(sources, "query_json_rows", lambda *args: rows)
    _, _, frame = core.load_splits(backend="mysql")
    assert frame.attrs["configuration_batch_id"] == 4
    assert frame[core.CFG_COLS].isna().all().all()


def test_staging_does_not_silently_mix_config_batches(monkeypatch):
    monkeypatch.setattr(sources, "query_json_rows", lambda *args: [{"batch_id": 4}, {"batch_id": 5}])
    with pytest.raises(ValueError, match="exactly one"):
        sources.staging_configuration_batch("test")


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
