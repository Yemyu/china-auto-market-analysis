#!/usr/bin/env python3
"""Run a full-population integrity audit and prioritize manual sales verification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from _sales_repair import apply_verified_sales_corrections, load_sales_correction_register


ROOT = Path(__file__).resolve().parent.parent
RAW_SALES = ROOT / "data" / "raw" / "monthly_sales.csv"
TEST_SPLIT = ROOT / "data" / "processed" / "splits" / "test.csv"
QUALITY = ROOT / "data" / "processed" / "data_quality"
STATUS_REGISTER = QUALITY / "sales_zero_status_register.csv"
TRAIN_END = pd.Timestamp("2025-06-01")
TEST_START = pd.Timestamp("2026-01-01")


def longest_true_run(values: pd.Series) -> int:
    best = current = 0
    for value in values.fillna(False).astype(bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def previous_12_month_average(group: pd.DataFrame, last_positive: pd.Timestamp) -> float:
    if pd.isna(last_positive):
        return 0.0
    start = last_positive - pd.DateOffset(months=11)
    window = group[group["date"].between(start, last_positive)]
    return float(window["monthly_sales_raw"].mean()) if len(window) else 0.0


def build_series_audit(
    repaired: pd.DataFrame,
    model_cohort: set[str],
    statuses: pd.DataFrame,
) -> pd.DataFrame:
    status_lookup = statuses.set_index("series_name").to_dict("index")
    rows: list[dict[str, object]] = []

    for name, group in repaired.groupby("series_name", sort=True):
        group = group.sort_values("date").copy()
        raw_positive = group["monthly_sales_raw"].gt(0)
        repaired_positive = group["monthly_sales"].gt(0)
        raw_positive_dates = group.loc[raw_positive, "date"]
        repaired_positive_dates = group.loc[repaired_positive, "date"]
        raw_first = raw_positive_dates.min() if len(raw_positive_dates) else pd.NaT
        raw_last = raw_positive_dates.max() if len(raw_positive_dates) else pd.NaT
        repaired_first = repaired_positive_dates.min() if len(repaired_positive_dates) else pd.NaT
        repaired_last = repaired_positive_dates.max() if len(repaired_positive_dates) else pd.NaT

        internal = (
            group[group["date"].between(repaired_first, repaired_last)]
            if pd.notna(repaired_first) and pd.notna(repaired_last)
            else group.iloc[0:0]
        )
        train = group[group["date"].le(TRAIN_END)]
        test = group[group["date"].ge(TEST_START)]
        status = status_lookup.get(name, {})
        audit_status = status.get("status", "unreviewed")

        row = {
            "series_id": int(group["series_id"].iloc[0]),
            "series_name": name,
            "brand": group["brand"].iloc[0],
            "category": group["category"].iloc[0],
            "in_model_cohort": name in model_cohort,
            "row_months": int(len(group)),
            "raw_positive_months": int(raw_positive.sum()),
            "repaired_positive_months": int(repaired_positive.sum()),
            "raw_first_positive_date": raw_first,
            "raw_last_positive_date": raw_last,
            "repaired_first_positive_date": repaired_first,
            "repaired_last_positive_date": repaired_last,
            "prelaunch_zero_months": (
                int(group["date"].lt(repaired_first).sum()) if pd.notna(repaired_first) else 0
            ),
            "internal_zero_months": int(internal["monthly_sales"].eq(0).sum()),
            "longest_internal_zero_run": longest_true_run(internal["monthly_sales"].eq(0)),
            "trailing_zero_months": (
                int(group["date"].gt(repaired_last).sum())
                if pd.notna(repaired_last)
                else int(len(group))
            ),
            "raw_train_positive_months": int(train["monthly_sales_raw"].gt(0).sum()),
            "raw_test_sales": int(test["monthly_sales_raw"].sum()),
            "repaired_test_sales": int(test["monthly_sales"].sum()),
            "test_sales_delta": int(
                test["monthly_sales"].sum() - test["monthly_sales_raw"].sum()
            ),
            "reference_avg_12_before_raw_cutoff": previous_12_month_average(group, raw_last),
            "audit_status": audit_status,
            "current_modeling_action": status.get("modeling_action", "not_reviewed"),
            "evidence_url": status.get("evidence_url", ""),
            "status_note": status.get("note", ""),
        }
        row.update(
            {
                "flag_never_positive": row["repaired_positive_months"] == 0,
                "flag_no_positive_in_training": row["raw_train_positive_months"] == 0,
                "flag_all_zero_raw_test": len(test) > 0 and row["raw_test_sales"] == 0,
                "flag_all_zero_repaired_test": len(test) > 0
                and row["repaired_test_sales"] == 0,
                "flag_internal_gap_ge3": row["longest_internal_zero_run"] >= 3,
                "flag_recent_trailing_cutoff": pd.notna(repaired_last)
                and repaired_last >= pd.Timestamp("2025-01-01")
                and row["trailing_zero_months"] >= 6,
                "flag_high_volume_cutoff": row["reference_avg_12_before_raw_cutoff"] >= 500
                and (
                    row["trailing_zero_months"] >= 6
                    or row["longest_internal_zero_run"] >= 6
                ),
            }
        )
        rows.append(row)

    audit = pd.DataFrame(rows)
    priority = audit.apply(classify_priority, axis=1, result_type="expand")
    priority.columns = ["verification_priority", "risk_score", "verification_reason"]
    audit = pd.concat([audit, priority], axis=1)
    order = pd.CategoricalDtype(
        ["critical", "high", "medium", "reviewed_keep", "low"], ordered=True
    )
    audit["verification_priority"] = audit["verification_priority"].astype(order)
    return audit.sort_values(
        ["verification_priority", "risk_score", "in_model_cohort", "series_name"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def classify_priority(row: pd.Series) -> tuple[str, int, str]:
    reasons: list[str] = []
    score = 0
    status = row["audit_status"]

    if status == "confirmed_post_discontinuation":
        return "reviewed_keep", 0, "same-source discontinuation already verified"
    if status == "confirmed_source_gap":
        reasons.append("same-source gap confirmed")
        score += 100
    if status == "source_page_stops_before_period":
        reasons.append("source page stops before audit period")
        score += 45
    elif status == "unresolved":
        reasons.append("previous high-risk review unresolved")
        score += 30

    anomaly = any(
        bool(row[column])
        for column in [
            "flag_never_positive",
            "flag_all_zero_repaired_test",
            "flag_internal_gap_ge3",
            "flag_recent_trailing_cutoff",
            "flag_high_volume_cutoff",
        ]
    )
    if row["in_model_cohort"] and anomaly:
        reasons.append("affects 371-series model cohort")
        score += 30
    if row["flag_never_positive"]:
        reasons.append("no positive sales in entire 54-month panel")
        score += 35
    if row["flag_all_zero_repaired_test"] and row["repaired_positive_months"] > 0:
        reasons.append("entire repaired test window is zero")
        score += 15
    if row["flag_internal_gap_ge3"]:
        reasons.append(f"internal zero run={int(row['longest_internal_zero_run'])} months")
        score += min(30, 2 * int(row["longest_internal_zero_run"]))
    if row["flag_recent_trailing_cutoff"]:
        reasons.append("positive in 2025 then at least 6 trailing zeros")
        score += 20
    if row["flag_high_volume_cutoff"]:
        reasons.append("high-volume history followed by a zero gap")
        score += 25

    if status == "confirmed_source_gap":
        priority = "critical"
    elif status == "source_page_stops_before_period":
        priority = "high"
    elif row["in_model_cohort"] and (
        row["flag_never_positive"]
        or row["flag_high_volume_cutoff"]
        or (row["flag_recent_trailing_cutoff"] and row["reference_avg_12_before_raw_cutoff"] >= 100)
    ):
        priority = "high"
    elif status == "unresolved" or row["flag_high_volume_cutoff"]:
        priority = "high"
    elif row["in_model_cohort"] and (
        row["flag_internal_gap_ge3"] or row["flag_recent_trailing_cutoff"]
    ):
        priority = "medium"
    elif row["flag_internal_gap_ge3"] or row["flag_recent_trailing_cutoff"]:
        priority = "medium"
    else:
        priority = "low"

    return priority, int(score), "; ".join(reasons) if reasons else "no prioritized anomaly"


def build_cutoff_clusters(audit: pd.DataFrame) -> pd.DataFrame:
    eligible = audit[audit["raw_last_positive_date"].notna()].copy()
    clusters = eligible.groupby("raw_last_positive_date", as_index=False).agg(
        series_count=("series_name", "size"),
        model_cohort_series=("in_model_cohort", "sum"),
        historical_reference_sales=("reference_avg_12_before_raw_cutoff", "sum"),
        high_volume_cutoff_series=("flag_high_volume_cutoff", "sum"),
    )
    clusters["raw_last_positive_date"] = clusters["raw_last_positive_date"].dt.strftime(
        "%Y-%m-%d"
    )
    return clusters.sort_values(
        ["series_count", "historical_reference_sales"], ascending=False
    ).reset_index(drop=True)


def build_integrity_summary(
    raw: pd.DataFrame,
    repaired: pd.DataFrame,
    audit: pd.DataFrame,
    model_cohort: set[str],
) -> dict[str, object]:
    raw = raw.copy()
    raw["date"] = pd.to_datetime(dict(year=raw["year"], month=raw["month"], day=1))
    series_year = raw.groupby(["series_name", "year"], as_index=False).agg(
        monthly_sum=("monthly_sales", "sum"),
        declared_cumulative=("website_cumulative_sales", "first"),
        cumulative_values=("website_cumulative_sales", "nunique"),
    )
    brand_month = raw.groupby(["brand", "date"], as_index=False).agg(
        monthly_sum=("monthly_sales", "sum"),
        declared_brand_total=("品牌月总销量", "first"),
        positive_series=("monthly_sales", lambda values: int(values.gt(0).sum())),
        declared_positive_series=("品牌车型数", "first"),
        total_values=("品牌月总销量", "nunique"),
        count_values=("品牌车型数", "nunique"),
    )
    zero_label_expected = np.where(
        raw["monthly_sales"].eq(0), "真实0销量（页面明确为0）", "有销量"
    )
    rank_presence_expected = raw["monthly_sales"].gt(0)
    row_counts = raw.groupby("series_name").size()

    priority_counts = (
        audit["verification_priority"].astype(str).value_counts().to_dict()
    )
    cohort = audit[audit["in_model_cohort"]]
    return {
        "schema_version": "v1",
        "population": {
            "rows": int(len(raw)),
            "series": int(raw["series_name"].nunique()),
            "months": int(raw["date"].nunique()),
            "date_start": raw["date"].min().strftime("%Y-%m-%d"),
            "date_end": raw["date"].max().strftime("%Y-%m-%d"),
        },
        "hard_integrity_checks": {
            "duplicate_series_month_keys": int(
                raw.duplicated(["series_name", "date"]).sum()
            ),
            "negative_sales_rows": int(raw["monthly_sales"].lt(0).sum()),
            "non_54_month_series": int(row_counts.ne(54).sum()),
            "series_names_with_multiple_ids": int(
                raw.groupby("series_name")["series_id"].nunique().gt(1).sum()
            ),
            "series_ids_with_multiple_names": int(
                raw.groupby("series_id")["series_name"].nunique().gt(1).sum()
            ),
        },
        "derived_field_checks": {
            "series_year_groups": int(len(series_year)),
            "annual_cumulative_exact_groups": int(
                series_year["monthly_sum"].eq(series_year["declared_cumulative"]).sum()
            ),
            "annual_cumulative_constant_groups": int(
                series_year["cumulative_values"].eq(1).sum()
            ),
            "brand_month_groups": int(len(brand_month)),
            "brand_total_exact_groups": int(
                brand_month["monthly_sum"].eq(brand_month["declared_brand_total"]).sum()
            ),
            "brand_positive_count_exact_groups": int(
                brand_month["positive_series"]
                .eq(brand_month["declared_positive_series"])
                .sum()
            ),
            "zero_label_exact_rows": int((raw["零销量类型"] == zero_label_expected).sum()),
            "rank_presence_exact_rows": int(
                raw["source_rank"].notna().eq(rank_presence_expected).sum()
            ),
            "interpretation": (
                "These fields are deterministic restatements of monthly_sales and cannot independently "
                "validate whether a source zero is real."
            ),
        },
        "series_risk_scan": {
            "never_positive_series": int(audit["flag_never_positive"].sum()),
            "internal_gap_ge3_series": int(audit["flag_internal_gap_ge3"].sum()),
            "recent_trailing_cutoff_series": int(audit["flag_recent_trailing_cutoff"].sum()),
            "high_volume_cutoff_series": int(audit["flag_high_volume_cutoff"].sum()),
            "priority_counts": {key: int(value) for key, value in priority_counts.items()},
        },
        "model_cohort_risk": {
            "series": int(len(model_cohort)),
            "never_positive_series": int(cohort["flag_never_positive"].sum()),
            "no_positive_in_training_series": int(
                cohort["flag_no_positive_in_training"].sum()
            ),
            "all_zero_raw_test_series": int(cohort["flag_all_zero_raw_test"].sum()),
            "all_zero_repaired_test_series": int(
                cohort["flag_all_zero_repaired_test"].sum()
            ),
            "verified_test_sales_delta": int(cohort["test_sales_delta"].sum()),
        },
        "cohort_construction_finding": (
            "The prior >=24 consecutive-month filter counts padded rows, not positive-sales history. "
            "Because every raw series has all 54 calendar rows, all 1,017 series pass that row-continuity "
            "test before the configuration join. Rebuild eligibility with lifecycle-aware positive-history "
            "rules after source repair."
        ),
        "decision": (
            "Use the full-population scan to prioritize external verification. Do not manually browse all "
            "54,918 rows and do not rebuild the model cohort until critical/high risks are resolved."
        ),
    }


def main() -> None:
    QUALITY.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_SALES, low_memory=False)
    correction_register = load_sales_correction_register()
    repaired, _ = apply_verified_sales_corrections(raw, correction_register)
    statuses = pd.read_csv(STATUS_REGISTER).fillna("")
    test = pd.read_csv(TEST_SPLIT, usecols=["series_name"], low_memory=False)
    model_cohort = set(test["series_name"].astype(str).unique())

    audit = build_series_audit(repaired, model_cohort, statuses)
    queue = audit[
        audit["verification_priority"].astype(str).isin(["critical", "high", "medium"])
    ].copy()
    clusters = build_cutoff_clusters(audit)
    summary = build_integrity_summary(raw, repaired, audit, model_cohort)

    date_columns = [
        "raw_first_positive_date",
        "raw_last_positive_date",
        "repaired_first_positive_date",
        "repaired_last_positive_date",
    ]
    for table in (audit, queue):
        table["verification_priority"] = table["verification_priority"].astype(str)
        for column in date_columns:
            table[column] = pd.to_datetime(table[column]).dt.strftime("%Y-%m-%d")

    audit.to_csv(QUALITY / "sales_series_risk_audit.csv", index=False, encoding="utf-8-sig")
    queue.to_csv(
        QUALITY / "sales_manual_verification_queue.csv", index=False, encoding="utf-8-sig"
    )
    clusters.to_csv(
        QUALITY / "sales_cutoff_cluster_audit.csv", index=False, encoding="utf-8-sig"
    )
    (QUALITY / "sales_panel_integrity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
