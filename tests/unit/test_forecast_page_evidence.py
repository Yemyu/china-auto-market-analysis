"""Keep the dashboard's small historical-results table tied to saved backtests."""

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_historical_table_matches_saved_results():
    page = (ROOT / "app/forecast.html").read_text()
    displayed = re.findall(
        r'<tr data-origin="([^"]+)"><td>[^<]+</td>'
        r'<td>([\d.]+)%</td><td>([\d.]+)%</td></tr>', page
    )
    with (ROOT / "data/processed/forecast/rolling_origin_validation.csv").open(
        encoding="utf-8-sig"
    ) as source:
        rows = list(csv.DictReader(source))
    expected = []
    for origin in sorted({row["origin"] for row in rows}):
        scores = {
            row["version"]: float(row["global_volume_weighted_WMAPE"])
            for row in rows if row["origin"] == origin
        }
        expected.append((origin, f'{scores["BASE"]:.2f}', f'{scores["SEASONAL_D5"]:.2f}'))
    assert len(expected) == 4
    assert displayed == expected


def test_rolling_evidence_precedes_fixed_stress_test():
    page = (ROOT / "app/forecast.html").read_text()
    sections = re.findall(r'<section[^>]+aria-labelledby="([^"]+)"', page)
    assert sections == [
        "fc-performance-title", "fc-diagnostics-title", "fc-error-title",
        "fc-stress-title", "fc-signal-title", "fc-review-value-title",
    ]
    ids = re.findall(r'\bid="([^"]+)"', page)
    assert len(ids) == len(set(ids))
