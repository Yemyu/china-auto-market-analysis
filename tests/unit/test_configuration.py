from __future__ import annotations

import pandas as pd

from china_auto_market.features import configuration


def config_row(series_name: str, year: int, value: float) -> dict[str, object]:
    row: dict[str, object] = {"series_name": series_name, "year": year}
    for column in configuration.CFG_COLS:
        row[column] = value
    return row


def test_configuration_fallback_never_uses_a_future_model_year(monkeypatch) -> None:
    config = pd.DataFrame(
        [
            config_row("A", 2022, 22.0),
            config_row("A", 2024, 24.0),
        ]
    )
    monkeypatch.setattr(configuration, "_load_cfg_frame", lambda: config)
    sales = pd.DataFrame(
        {
            "series_name": ["A", "A"],
            "year": [2023, 2024],
            "monthly_sales": [10, 20],
        }
    )

    joined = configuration.join_cfg(sales)

    assert joined.loc[joined["year"].eq(2023), configuration.CFG_NUM[0]].iloc[0] == 22.0
    assert joined.loc[joined["year"].eq(2024), configuration.CFG_NUM[0]].iloc[0] == 24.0


def test_keep_unmatched_preserves_pre_configuration_month(monkeypatch) -> None:
    config = pd.DataFrame([config_row("A", 2022, 22.0)])
    monkeypatch.setattr(configuration, "_load_cfg_frame", lambda: config)
    sales = pd.DataFrame(
        {
            "series_name": ["A", "A"],
            "year": [2021, 2022],
            "monthly_sales": [10, 20],
        }
    )

    joined = configuration.join_cfg(sales, keep_unmatched=True)

    assert len(joined) == 2
    unknown = joined.loc[joined["year"].eq(2021)].iloc[0]
    assert unknown[configuration.CFG_NUM[0]] == 22.0
    assert unknown[f"{configuration.CFG_CAT[0]}_enc"] == -1.0
