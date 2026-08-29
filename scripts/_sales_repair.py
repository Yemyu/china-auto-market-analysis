#!/usr/bin/env python3
"""Apply manually verified sales corrections without mutating the raw snapshot."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTER = (
    ROOT / "data" / "processed" / "data_quality" / "sales_correction_register.csv"
)
DEFAULT_ANNUAL_REGISTER = (
    ROOT / "data" / "processed" / "data_quality" / "annual_sales_correction_register.csv"
)
REQUIRED_REGISTER_COLUMNS = {
    "series_name",
    "date",
    "original_sales",
    "corrected_sales",
    "source_name",
    "source_url",
    "evidence_status",
    "verified_at",
    "note",
}
APPLICABLE_STATUSES = {
    "verified_same_source",
    "verified_cross_source_consensus",
}
ANNUAL_REQUIRED_REGISTER_COLUMNS = {
    "series_name",
    "year",
    "original_annual_sales",
    "corrected_annual_sales",
    "source_name",
    "source_url",
    "evidence_status",
    "verified_at",
    "note",
}


def load_sales_correction_register(path: Path = DEFAULT_REGISTER) -> pd.DataFrame:
    """Load and strictly validate the small, reviewable correction register."""
    register = pd.read_csv(path)
    missing = REQUIRED_REGISTER_COLUMNS - set(register.columns)
    if missing:
        raise ValueError(f"Sales correction register is missing columns: {sorted(missing)}")

    register = register.copy()
    register["date"] = pd.to_datetime(register["date"], errors="raise")
    if register["date"].dt.day.ne(1).any():
        raise ValueError("Correction dates must be calendar-month starts")
    if register.duplicated(["series_name", "date"]).any():
        raise ValueError("Correction register contains duplicate series-month keys")
    if register[["original_sales", "corrected_sales"]].isna().any().any():
        raise ValueError("Correction sales values cannot be missing")
    if register[["original_sales", "corrected_sales"]].lt(0).any().any():
        raise ValueError("Correction sales values cannot be negative")
    if register["original_sales"].eq(register["corrected_sales"]).any():
        raise ValueError("Correction rows must change the original value")
    return register


def apply_verified_sales_corrections(
    sales: pd.DataFrame,
    register: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a repaired copy and an applied-row audit.

    Only same-source verifications or explicit cross-source consensus rows are
    eligible. Every eligible row must match exactly one raw series-month and its
    declared original value, otherwise the function fails closed.
    """
    if register is None:
        register = load_sales_correction_register()
    else:
        register = register.copy()
        register["date"] = pd.to_datetime(register["date"], errors="raise")

    applicable = register.loc[
        register["evidence_status"].isin(APPLICABLE_STATUSES)
    ].copy()
    repaired = sales.copy()
    if "date" not in repaired:
        repaired["date"] = pd.to_datetime(
            dict(year=repaired["year"], month=repaired["month"], day=1)
        )
    else:
        repaired["date"] = pd.to_datetime(repaired["date"], errors="raise")

    if repaired.duplicated(["series_name", "date"]).any():
        raise ValueError("Sales data contains duplicate series-month keys")

    applicable = applicable.rename(columns={"original_sales": "declared_original_sales"})
    match = applicable.merge(
        repaired[["series_name", "date", "monthly_sales"]],
        on=["series_name", "date"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if match["_merge"].ne("both").any():
        missing = match.loc[match["_merge"].ne("both"), ["series_name", "date"]]
        raise ValueError(f"Correction rows are absent from sales data: {missing.to_dict('records')}")
    mismatch = match["monthly_sales"].ne(match["declared_original_sales"])
    if mismatch.any():
        columns = ["series_name", "date", "declared_original_sales", "monthly_sales"]
        raise ValueError(
            "Correction original value does not match the raw snapshot: "
            f"{match.loc[mismatch, columns].to_dict('records')}"
        )

    repaired["monthly_sales_raw"] = repaired["monthly_sales"]
    repaired["sales_repair_applied"] = False
    repaired["sales_repair_evidence_status"] = ""
    repaired["sales_repair_source_url"] = ""

    corrections = applicable.set_index(["series_name", "date"])
    repaired_index = repaired.set_index(["series_name", "date"])
    keys = corrections.index
    repaired_index.loc[keys, "monthly_sales"] = corrections["corrected_sales"].to_numpy()
    repaired_index.loc[keys, "sales_repair_applied"] = True
    repaired_index.loc[keys, "sales_repair_evidence_status"] = applicable[
        "evidence_status"
    ].to_numpy()
    repaired_index.loc[keys, "sales_repair_source_url"] = applicable["source_url"].to_numpy()
    repaired = repaired_index.reset_index()

    audit_columns = [
        "series_name",
        "date",
        "declared_original_sales",
        "corrected_sales",
        "source_name",
        "source_url",
        "evidence_status",
        "verified_at",
        "note",
    ]
    audit = match[audit_columns].rename(
        columns={"declared_original_sales": "original_sales"}
    )
    audit["sales_delta"] = audit["corrected_sales"] - audit["original_sales"]
    return repaired, audit.sort_values(["series_name", "date"]).reset_index(drop=True)


def load_annual_sales_correction_register(
    path: Path = DEFAULT_ANNUAL_REGISTER,
) -> pd.DataFrame:
    """Load the explicit, source-verified annual target correction register."""
    register = pd.read_csv(path)
    missing = ANNUAL_REQUIRED_REGISTER_COLUMNS - set(register.columns)
    if missing:
        raise ValueError(
            f"Annual correction register is missing columns: {sorted(missing)}"
        )
    if register.duplicated(["series_name", "year"]).any():
        raise ValueError("Annual correction register contains duplicate series-year keys")
    if register["evidence_status"].isin(APPLICABLE_STATUSES).eq(False).any():
        raise ValueError("Annual correction register contains an ineligible evidence status")
    numeric = ["original_annual_sales", "corrected_annual_sales"]
    if register[numeric].isna().any().any() or register[numeric].lt(0).any().any():
        raise ValueError("Annual correction sales values must be non-negative and non-missing")
    if register["original_annual_sales"].eq(register["corrected_annual_sales"]).any():
        raise ValueError("Annual correction rows must change the original value")
    return register


def apply_verified_annual_sales_corrections(
    features: pd.DataFrame,
    sales: pd.DataFrame,
    register: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply only explicitly verified annual targets derived from full calendar years.

    The raw configuration table remains untouched.  Each register row must match
    the declared original annual value, have exactly twelve monthly observations,
    and agree with the corrected monthly sum.  Any mismatch fails closed.
    """
    if register is None:
        register = load_annual_sales_correction_register()
    else:
        register = register.copy()
    required_feature = {"series_name", "year", "annual_sales"}
    if not required_feature.issubset(features.columns):
        raise ValueError(f"Features are missing columns: {sorted(required_feature - set(features.columns))}")
    sales = sales.copy()
    if "date" in sales.columns:
        sales["date"] = pd.to_datetime(sales["date"], errors="raise")
    else:
        sales["date"] = pd.to_datetime(
            dict(year=sales["year"], month=sales["month"], day=1), errors="raise"
        )
    monthly = (
        sales.groupby(["series_name", sales["date"].dt.year.rename("year")], as_index=False)
        .agg(corrected_annual_sales=("monthly_sales", "sum"), months=("date", "nunique"))
    )
    monthly = monthly.loc[monthly["months"].eq(12)].copy()

    feature_keys = features[["series_name", "year", "annual_sales"]].copy()
    feature_keys["series_name"] = feature_keys["series_name"].astype(str)
    register["series_name"] = register["series_name"].astype(str)
    register["year"] = register["year"].astype(int)
    match = register.merge(
        feature_keys,
        on=["series_name", "year"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if match["_merge"].ne("both").any():
        missing = match.loc[match["_merge"].ne("both"), ["series_name", "year"]]
        raise ValueError(f"Annual correction rows are absent from features: {missing.to_dict('records')}")
    mismatch = match["annual_sales"].ne(match["original_annual_sales"])
    if mismatch.any():
        columns = ["series_name", "year", "original_annual_sales", "annual_sales"]
        raise ValueError(
            "Annual correction original value does not match the feature snapshot: "
            f"{match.loc[mismatch, columns].to_dict('records')}"
        )
    match = match.drop(columns="_merge").merge(
        monthly, on=["series_name", "year"], how="left", validate="one_to_one"
    )
    if match["corrected_annual_sales_y"].isna().any():
        missing = match.loc[match["corrected_annual_sales_y"].isna(), ["series_name", "year"]]
        raise ValueError(f"Annual correction rows lack a complete monthly year: {missing.to_dict('records')}")
    if match["corrected_annual_sales_x"].ne(match["corrected_annual_sales_y"]).any():
        columns = ["series_name", "year", "corrected_annual_sales_x", "corrected_annual_sales_y"]
        raise ValueError(
            "Annual correction does not agree with the verified monthly sum: "
            f"{match.loc[match['corrected_annual_sales_x'].ne(match['corrected_annual_sales_y']), columns].to_dict('records')}"
        )

    repaired = features.copy()
    repaired["annual_sales_raw"] = repaired["annual_sales"]
    repaired["annual_sales_repair_applied"] = False
    corrections = match.set_index(["series_name", "year"])
    repaired_index = repaired.set_index(["series_name", "year"])
    keys = corrections.index
    repaired_index.loc[keys, "annual_sales"] = corrections["corrected_annual_sales_x"].to_numpy()
    repaired_index.loc[keys, "annual_sales_repair_applied"] = True
    repaired = repaired_index.reset_index()

    audit = match[
        [
            "series_name",
            "year",
            "annual_sales",
            "corrected_annual_sales_x",
            "source_name",
            "source_url",
            "evidence_status",
            "verified_at",
            "note",
        ]
    ].rename(
        columns={
            "annual_sales": "original_annual_sales",
            "corrected_annual_sales_x": "corrected_annual_sales",
        }
    )
    audit["sales_delta"] = audit["corrected_annual_sales"] - audit["original_annual_sales"]
    return repaired, audit.sort_values(["series_name", "year"]).reset_index(drop=True)
