from __future__ import annotations

import pandas as pd

from china_auto_market.marts.forecast import build_forecast_panel


def test_forecast_panel_rejects_unlocked_cohort() -> None:
    try:
        build_forecast_panel(pd.DataFrame(), {"only-one"})
    except ValueError as error:
        assert "371" in str(error)
    else:
        raise AssertionError("an unlocked cohort must fail")
