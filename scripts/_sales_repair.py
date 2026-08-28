#!/usr/bin/env python3
"""Apply manually verified sales corrections without mutating the raw snapshot."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTER = (
    ROOT / "data" / "processed" / "data_quality" / "sales_correction_register.csv"
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
APPLICABLE_STATUS = "verified_same_source"


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

    Only rows marked ``verified_same_source`` are eligible. Every eligible row must
    match exactly one raw series-month and its declared original value, otherwise
    the function fails closed.
    """
    if register is None:
        register = load_sales_correction_register()
    else:
        register = register.copy()
        register["date"] = pd.to_datetime(register["date"], errors="raise")

    applicable = register.loc[register["evidence_status"].eq(APPLICABLE_STATUS)].copy()
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
