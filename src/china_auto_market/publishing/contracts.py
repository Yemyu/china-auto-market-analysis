"""Lightweight schema checks for the pre-baked dashboard payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DASHBOARD_REQUIRED_KEYS = {
    "overview.json": {"findings", "kpis", "monthly_trend", "stages"},
    "forecast.json": {
        "best_model",
        "class_wmape",
        "conclusion",
        "features",
        "fixed_stress",
        "meta",
        "primary_models",
    },
    "attribution.json": {
        "comparison",
        "conclusion",
        "meta",
        "models",
        "shap",
        "wmape_by_variant",
    },
    "absa.json": {"aspects", "conclusion", "distribution", "meta", "monthly_trends"},
    "alerts.json": {"alerts", "conclusion", "meta", "monthly", "risk_dist", "rule"},
    "drilldown.json": {"brand_en", "brands", "data", "series"},
    "brand_drilldown.json": {"brand_en", "brands", "data", "market_radar"},
    "forecast_evidence.json": {
        "conclusion",
        "correlation",
        "fusion",
        "granger",
        "scenario",
        "timeseries",
    },
}


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject arrays or scalar payloads."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def validate_dashboard_payloads(data_dir: Path) -> dict[str, list[str]]:
    """Return missing top-level keys by file; an empty result means success."""
    failures: dict[str, list[str]] = {}
    for filename, required in DASHBOARD_REQUIRED_KEYS.items():
        path = data_dir / filename
        if not path.is_file():
            failures[filename] = ["<missing file>"]
            continue
        payload = load_json_object(path)
        missing = sorted(required - payload.keys())
        if missing:
            failures[filename] = missing
    return failures


def assert_dashboard_payloads(data_dir: Path) -> None:
    """Fail with a compact contract report when dashboard payloads drift."""
    failures = validate_dashboard_payloads(data_dir)
    if failures:
        raise ValueError(f"Dashboard payload contract failed: {failures}")
