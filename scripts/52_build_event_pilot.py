#!/usr/bin/env python3
"""Select a small event-data pilot from historical forecast evidence only."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "artifacts" / "conditional_experts" / "candidate_predictions.csv.gz"
OUT = ROOT / "data" / "events"
ROSTER = OUT / "event_pilot_series.csv"
EVENTS = OUT / "event_records.csv"
TOP_N_PER_AXIS = 18
EXPECTED_MIN_SERIES = 20
EXPECTED_MAX_SERIES = 30

EVENT_COLUMNS = [
    "series_name",
    "event_date",
    "event_type",
    "event_subtype",
    "event_title",
    "event_end_date",
    "price_change_wan",
    "promotion_amount_wan",
    "source_name",
    "source_url",
    "published_at",
    "known_at",
    "verification_status",
    "notes",
]


def build_roster() -> pd.DataFrame:
    if not PREDICTIONS.exists():
        raise FileNotFoundError(
            f"Missing {PREDICTIONS}; run scripts/49_evaluate_conditional_experts.py first"
        )
    predictions = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    predictions = predictions.loc[predictions["version"].eq("CONFIG_SEASONAL")].copy()
    if predictions["date"].max() >= pd.Timestamp("2026-01-01"):
        raise AssertionError("Event pilot selection must not use locked-test targets")
    predictions["absolute_error"] = np.abs(
        predictions["actual"] - predictions["pred"]
    )
    summary = (
        predictions.groupby("series_name", as_index=False)
        .agg(
            historical_actual_volume=("actual", "sum"),
            historical_absolute_error=("absolute_error", "sum"),
            historical_months=("date", "size"),
            historical_origins=("origin", "nunique"),
        )
    )
    summary["historical_WMAPE"] = (
        summary["historical_absolute_error"]
        / summary["historical_actual_volume"].replace(0, np.nan)
        * 100
    )
    summary["absolute_error_rank"] = summary["historical_absolute_error"].rank(
        method="min", ascending=False
    ).astype(int)
    summary["actual_volume_rank"] = summary["historical_actual_volume"].rank(
        method="min", ascending=False
    ).astype(int)
    error_names = set(
        summary.nlargest(TOP_N_PER_AXIS, "historical_absolute_error")["series_name"]
    )
    volume_names = set(
        summary.nlargest(TOP_N_PER_AXIS, "historical_actual_volume")["series_name"]
    )
    selected = summary.loc[
        summary["series_name"].isin(error_names | volume_names)
    ].copy()
    if not EXPECTED_MIN_SERIES <= len(selected) <= EXPECTED_MAX_SERIES:
        raise AssertionError(f"Unexpected event-pilot size: {len(selected)}")
    selected["selected_for_high_error"] = selected["series_name"].isin(error_names)
    selected["selected_for_high_volume"] = selected["series_name"].isin(volume_names)
    selected["selection_score"] = (
        selected["absolute_error_rank"] + selected["actual_volume_rank"]
    )
    selected = selected.sort_values(
        ["selection_score", "historical_absolute_error", "series_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    selected.insert(0, "priority_rank", np.arange(1, len(selected) + 1))
    selected["event_types_requested"] = "launch|facelift|price_cut|promotion"
    selected["selection_window"] = "rolling origins 2024-01..2025-07; targets through 2025-12"
    selected["locked_test_used"] = False
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset-empty-events",
        action="store_true",
        help="Replace event_records.csv with an empty schema. Existing records are preserved by default.",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    roster = build_roster()
    roster.to_csv(ROSTER, index=False, encoding="utf-8-sig")
    if args.reset_empty_events or not EVENTS.exists():
        pd.DataFrame(columns=EVENT_COLUMNS).to_csv(EVENTS, index=False, encoding="utf-8-sig")
    print(
        f"event_pilot_series={len(roster)} "
        f"high_error={int(roster['selected_for_high_error'].sum())} "
        f"high_volume={int(roster['selected_for_high_volume'].sum())}"
    )
    print(roster[[
        "priority_rank", "series_name", "historical_actual_volume",
        "historical_absolute_error", "historical_WMAPE",
        "selected_for_high_error", "selected_for_high_volume",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
