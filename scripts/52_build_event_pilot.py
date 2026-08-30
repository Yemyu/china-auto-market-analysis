#!/usr/bin/env python3
"""Select a small event-data pilot from historical forecast evidence only."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "artifacts" / "conditional_experts" / "candidate_predictions.csv.gz"
OUT = ROOT / "data" / "events"
ROSTER = OUT / "event_pilot_series.csv"
EVENTS = OUT / "event_records.csv"
DIAGNOSTIC_OUT = ROOT / "artifacts" / "event_pilot"
TOP_N_PER_AXIS = 18
EXPECTED_MIN_SERIES = 20
EXPECTED_MAX_SERIES = 30
PILOT_SELECTION_ORIGINS = {"2024-01-01", "2024-07-01", "2025-01-01"}
PROMOTION_ORIGIN = "2025-07-01"
SIGNAL_WINDOW_MONTHS = 3
SIGNAL_GATE = {
    "minimum_calibration_recent_rows": 20,
    "minimum_calibration_recent_series": 5,
    "minimum_recent_window_WMAPE_uplift_pp": 5.0,
    "minimum_absolute_recent_signed_bias_pp": 5.0,
    "minimum_promotion_recent_rows": 10,
    "minimum_promotion_recent_series": 3,
}
ALLOWED_EVENT_TYPES = {"launch", "facelift", "price_cut", "promotion"}
ALLOWED_VERIFICATION_STATUSES = {"verified_primary", "verified_secondary"}

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
    predictions = predictions.loc[
        predictions["origin"].isin(PILOT_SELECTION_ORIGINS)
    ].copy()
    if set(predictions["origin"]) != PILOT_SELECTION_ORIGINS:
        raise AssertionError("Event pilot selection origins are incomplete")
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
    selected["selection_window"] = (
        "calibration origins 2024-01, 2024-07, 2025-01; targets through 2025-06"
    )
    selected["locked_test_used"] = False
    return selected


def validate_event_records(events: pd.DataFrame, roster: pd.DataFrame) -> None:
    """Reject ambiguous, untraceable, or point-in-time unsafe event rows."""
    if list(events.columns) != EVENT_COLUMNS:
        raise ValueError("event_records.csv does not match the declared schema")
    if events.empty:
        return
    required = [
        "series_name", "event_date", "event_type", "event_subtype", "event_title",
        "source_name", "source_url", "published_at", "known_at",
        "verification_status",
    ]
    missing = events[required].isna() | events[required].astype(str).apply(
        lambda column: column.str.strip().eq("")
    )
    if missing.any(axis=None):
        rows = sorted(set(events.index[missing.any(axis=1)] + 2))
        raise ValueError(f"Required event fields are blank on CSV rows {rows}")
    unknown_series = sorted(set(events["series_name"]) - set(roster["series_name"]))
    if unknown_series:
        raise ValueError(f"Events contain series outside the frozen pilot: {unknown_series}")
    unknown_types = sorted(set(events["event_type"]) - ALLOWED_EVENT_TYPES)
    if unknown_types:
        raise ValueError(f"Unsupported event types: {unknown_types}")
    unknown_statuses = sorted(
        set(events["verification_status"]) - ALLOWED_VERIFICATION_STATUSES
    )
    if unknown_statuses:
        raise ValueError(f"Unsupported verification statuses: {unknown_statuses}")
    if not events["source_url"].astype(str).str.startswith(("https://", "http://")).all():
        raise ValueError("Every event must have a traceable HTTP(S) source")

    event_dates = pd.to_datetime(events["event_date"], errors="raise")
    published = pd.to_datetime(events["published_at"], errors="raise", utc=True)
    known = pd.to_datetime(events["known_at"], errors="raise", utc=True)
    if (known < published).any():
        raise ValueError("known_at cannot precede published_at")
    known_shanghai = known.dt.tz_convert("Asia/Shanghai")
    if (known_shanghai >= pd.Timestamp("2026-01-01", tz="Asia/Shanghai")).any():
        raise ValueError("Pilot collection must remain outside the locked-test period")
    end_dates = pd.to_datetime(events["event_end_date"], errors="coerce")
    if (end_dates.notna() & end_dates.lt(event_dates)).any():
        raise ValueError("event_end_date cannot precede event_date")
    duplicate_key = ["series_name", "event_date", "event_type", "event_subtype"]
    if events.duplicated(duplicate_key).any():
        raise ValueError(f"Duplicate atomic events found on key {duplicate_key}")


def build_point_in_time_features(events: pd.DataFrame) -> pd.DataFrame:
    """Attach event history that was public before each forecast month began."""
    predictions = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    predictions = predictions.loc[
        predictions["version"].eq("CONFIG_SEASONAL")
    ].copy()
    if predictions["date"].max() >= pd.Timestamp("2026-01-01"):
        raise AssertionError("Event diagnostics must not read locked-test targets")

    prepared = events.copy()
    prepared["known_at_local"] = pd.to_datetime(
        prepared["known_at"], errors="raise", utc=True
    ).dt.tz_convert("Asia/Shanghai")
    prepared["event_month"] = pd.to_datetime(
        prepared["event_date"], errors="raise"
    ).dt.to_period("M")
    prepared["price_change_wan"] = pd.to_numeric(
        prepared["price_change_wan"], errors="coerce"
    )
    prepared["promotion_amount_wan"] = pd.to_numeric(
        prepared["promotion_amount_wan"], errors="coerce"
    )
    covered_series = set(prepared["series_name"])

    rows: list[dict[str, object]] = []
    for record in predictions.itertuples(index=False):
        target_date = pd.Timestamp(record.date)
        information_cutoff = target_date.tz_localize("Asia/Shanghai")
        known = prepared.loc[
            prepared["series_name"].eq(record.series_name)
            & prepared["known_at_local"].lt(information_cutoff)
        ].copy()
        target_month = target_date.to_period("M")
        known["months_since_event"] = np.asarray([
            target_month.ordinal - month.ordinal for month in known["event_month"]
        ], dtype=int)
        known = known.loc[known["months_since_event"].ge(0)].copy()
        recent_3m = known["months_since_event"].between(1, 3)
        recent_6m = known["months_since_event"].between(1, 6)
        price_cuts = known.loc[
            recent_6m & known["event_type"].eq("price_cut"), "price_change_wan"
        ].dropna()
        promotions = known.loc[
            recent_3m & known["event_type"].eq("promotion"), "promotion_amount_wan"
        ].dropna()
        rows.append({
            "origin": record.origin,
            "series_name": record.series_name,
            "date": target_date,
            "actual": float(record.actual),
            "prediction": float(record.pred),
            "event_coverage_available": int(record.series_name in covered_series),
            "event_count_known_prior": int(len(known)),
            "event_count_recent_3m": int(recent_3m.sum()),
            "event_count_recent_6m": int(recent_6m.sum()),
            "has_recent_event_3m": int(recent_3m.any()),
            "has_recent_event_6m": int(recent_6m.any()),
            "months_since_last_event": (
                int(known["months_since_event"].min()) if not known.empty else np.nan
            ),
            "price_cut_magnitude_6m_wan": float(-price_cuts.clip(upper=0).sum()),
            "promotion_amount_3m_wan": float(promotions.sum()),
            "launch_count_6m": int(
                (recent_6m & known["event_type"].eq("launch")).sum()
            ),
            "facelift_count_6m": int(
                (recent_6m & known["event_type"].eq("facelift")).sum()
            ),
        })
    result = pd.DataFrame(rows)
    expected = len(predictions)
    if len(result) != expected or result.duplicated(
        ["origin", "series_name", "date"]
    ).any():
        raise ValueError("Event feature coverage does not match historical predictions")
    return result


def pooled_diagnostic(frame: pd.DataFrame) -> dict[str, float | int | None]:
    if frame.empty:
        return {
            "rows": 0, "series": 0, "actual_volume": 0.0,
            "WMAPE": None, "signed_error_share": None,
        }
    actual_volume = float(frame["actual"].abs().sum())
    error = frame["actual"] - frame["prediction"]
    return {
        "rows": int(len(frame)),
        "series": int(frame["series_name"].nunique()),
        "actual_volume": actual_volume,
        "WMAPE": float(error.abs().sum() / actual_volume * 100),
        "signed_error_share": float(error.sum() / actual_volume * 100),
    }


def write_signal_diagnostic(features: pd.DataFrame) -> dict[str, object]:
    """Measure whether events target hard residuals; do not fit a correction."""
    covered = features.loc[features["event_coverage_available"].eq(1)].copy()
    calibration = covered.loc[covered["origin"].isin(PILOT_SELECTION_ORIGINS)]
    promotion = covered.loc[covered["origin"].eq(PROMOTION_ORIGIN)]
    calibration_recent = calibration.loc[calibration["has_recent_event_3m"].eq(1)]
    calibration_inactive = calibration.loc[calibration["has_recent_event_3m"].eq(0)]
    promotion_recent = promotion.loc[promotion["has_recent_event_3m"].eq(1)]

    calibration_recent_metrics = pooled_diagnostic(calibration_recent)
    calibration_inactive_metrics = pooled_diagnostic(calibration_inactive)
    promotion_recent_metrics = pooled_diagnostic(promotion_recent)
    uplift = (
        float(calibration_recent_metrics["WMAPE"])
        - float(calibration_inactive_metrics["WMAPE"])
    )
    calibration_signal = bool(
        calibration_recent_metrics["rows"]
        >= SIGNAL_GATE["minimum_calibration_recent_rows"]
        and calibration_recent_metrics["series"]
        >= SIGNAL_GATE["minimum_calibration_recent_series"]
        and uplift >= SIGNAL_GATE["minimum_recent_window_WMAPE_uplift_pp"]
        and abs(float(calibration_recent_metrics["signed_error_share"]))
        >= SIGNAL_GATE["minimum_absolute_recent_signed_bias_pp"]
    )
    promotion_coverage = bool(
        promotion_recent_metrics["rows"]
        >= SIGNAL_GATE["minimum_promotion_recent_rows"]
        and promotion_recent_metrics["series"]
        >= SIGNAL_GATE["minimum_promotion_recent_series"]
    )
    if calibration_signal and not promotion_coverage:
        decision = "complete_seed_series_promotion_window_before_roster_expansion"
    elif calibration_signal and promotion_coverage:
        decision = "evaluate_predeclared_event_correction_on_promotion_origin"
    else:
        decision = "stop_event_expansion_due_to_weak_calibration_targeting_signal"
    summary: dict[str, object] = {
        "schema_version": "v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "exploratory point-in-time event residual diagnostic",
        "interpretation": (
            "Association and targeting diagnostic only; it is not a causal effect "
            "or an event-model performance estimate."
        ),
        "locked_test_targets_read": False,
        "signal_window_months": SIGNAL_WINDOW_MONTHS,
        "gate": SIGNAL_GATE,
        "calibration_recent": calibration_recent_metrics,
        "calibration_inactive": calibration_inactive_metrics,
        "calibration_recent_WMAPE_uplift_pp": uplift,
        "calibration_targeting_signal": calibration_signal,
        "promotion_recent": promotion_recent_metrics,
        "promotion_coverage_sufficient": promotion_coverage,
        "decision": decision,
    }
    DIAGNOSTIC_OUT.mkdir(parents=True, exist_ok=True)
    features.to_csv(
        DIAGNOSTIC_OUT / "point_in_time_event_features.csv.gz",
        index=False,
        compression="gzip",
    )
    (DIAGNOSTIC_OUT / "signal_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return summary


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
    events = pd.read_csv(EVENTS)
    validate_event_records(events, roster)
    diagnostic = write_signal_diagnostic(build_point_in_time_features(events))
    print(
        f"event_pilot_series={len(roster)} "
        f"high_error={int(roster['selected_for_high_error'].sum())} "
        f"high_volume={int(roster['selected_for_high_volume'].sum())} "
        f"verified_events={len(events)}"
    )
    print(
        "event_signal_decision=" + str(diagnostic["decision"])
    )
    print(roster[[
        "priority_rank", "series_name", "historical_actual_volume",
        "historical_absolute_error", "historical_WMAPE",
        "selected_for_high_error", "selected_for_high_volume",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
