from __future__ import annotations

import pandas as pd
import pytest

from china_auto_market.quality.sales_repair import apply_verified_sales_corrections


def register(original: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "series_name": "A",
                "date": "2024-01-01",
                "original_sales": original,
                "corrected_sales": 100,
                "source_name": "source",
                "source_url": "https://example.invalid/evidence",
                "evidence_status": "verified_same_source",
                "verified_at": "2026-01-01",
                "note": "fixture",
            }
        ]
    )


def test_verified_correction_preserves_raw_value_and_emits_audit() -> None:
    sales = pd.DataFrame(
        {"series_name": ["A"], "date": ["2024-01-01"], "monthly_sales": [0]}
    )

    repaired, audit = apply_verified_sales_corrections(sales, register())

    assert repaired.iloc[0]["monthly_sales_raw"] == 0
    assert repaired.iloc[0]["monthly_sales"] == 100
    assert bool(repaired.iloc[0]["sales_repair_applied"])
    assert audit.iloc[0]["sales_delta"] == 100


def test_correction_fails_closed_when_declared_original_does_not_match() -> None:
    sales = pd.DataFrame(
        {"series_name": ["A"], "date": ["2024-01-01"], "monthly_sales": [0]}
    )

    with pytest.raises(ValueError, match="does not match"):
        apply_verified_sales_corrections(sales, register(original=1))
